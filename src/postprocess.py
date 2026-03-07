import pandas as pd

def step1_build_summary_tables(step1_results, input_data):
    """
    Returns tables:
    - generator dispatch & profit
    - wind dispatch & profit
    - demand served & utility
    """
    price = step1_results.market_price

    gen_rows = []
    for g in input_data.GENERATORS:
        pg = step1_results.pG[g]
        cg = input_data.generator_cost[g]
        gen_rows.append({
            "generator": g,
            "p_MW": pg,
            "marginal_cost": cg,
            "profit": (price - cg) * pg
        })
    df_gen = pd.DataFrame(gen_rows).sort_values("generator")

    wind_rows = []
    for w in input_data.WINDS:
        pw = step1_results.pW[w]
        wind_rows.append({
            "wind": w,
            "p_MW": pw,
            "profit": price * pw
        })
    df_wind = pd.DataFrame(wind_rows).sort_values("wind")

    dem_rows = []
    for n in input_data.LOAD_BUSES:
        dn = step1_results.d[n]
        bn = input_data.demand_bid[n]
        dem_rows.append({
            "bus": n,
            "d_MW": dn,
            "bid_price": bn,
            "utility": (bn - price) * dn
        })
    df_dem = pd.DataFrame(dem_rows).sort_values("bus")

    total_operating_cost = (df_gen["marginal_cost"] * df_gen["p_MW"]).sum()

    totals = {
        "market_price": price,
        "objective_welfare": step1_results.objective_value,
        "total_operating_cost": total_operating_cost,
        "total_gen_MW": df_gen["p_MW"].sum(),
        "total_wind_MW": df_wind["p_MW"].sum(),
        "total_demand_served_MW": df_dem["d_MW"].sum(),
        "total_gen_profit": df_gen["profit"].sum(),
        "total_wind_profit": df_wind["profit"].sum(),
        "total_demand_utility": df_dem["utility"].sum(),
    }

    return df_gen, df_wind, df_dem, totals

# ========================
# STEP 2: 24-HOUR DISPATCH RESULTS
# ========================

def step2_build_summary_tables(step2_results, input_data):
    """
    Returns:
    - df_gen: generator dispatch & profit (24h totals)
    - df_wind: wind dispatch & profit (24h totals)
    - df_dem: demand served & utility (24h totals)
    - df_storage: storage operation & profit (hourly breakdown)
    - df_prices: hourly market prices
    - totals: dict with 24h aggregates
    """
    T = input_data.hours
    prices_t = step2_results.market_prices_t
    
    # Generator dispatch & profit (aggregated 24h)

    gen_rows = []
    for g in input_data.GENERATORS:
        total_pg = sum(step2_results.pG[(t, g)] for t in T)
        cg = input_data.generator_cost[g]
        # Profit = sum_t [price_t * pG_tg - cg * pG_tg]
        profit = sum(prices_t[t] * step2_results.pG[(t, g)] - cg * step2_results.pG[(t, g)] for t in T)
        gen_rows.append({
            "generator": g,
            "total_p_MWh": total_pg,  # MWh over 24h
            "marginal_cost": cg,
            "total_profit_24h": profit
        })
    df_gen = pd.DataFrame(gen_rows).sort_values("generator")
    
    # Wind dispatch & profit (aggregated 24h)

    wind_rows = []
    for w in input_data.WINDS:
        total_pw = sum(step2_results.pW[(t, w)] for t in T)
        # Profit = sum_t [price_t * pW_tw] (cost=0)
        profit = sum(prices_t[t] * step2_results.pW[(t, w)] for t in T)
        wind_rows.append({
            "wind": w,
            "total_p_MWh": total_pw,
            "total_profit_24h": profit
        })
    df_wind = pd.DataFrame(wind_rows).sort_values("wind")
    
    # Demand served & utility (aggregated 24h)

    dem_rows = []
    for n in input_data.LOAD_BUSES:
        total_dn = sum(step2_results.d[(t, n)] for t in T)
        # Utility = sum_t [(bid_tn - price_t) * d_tn]
        utility = sum((input_data.demand_bid_t[t][n] - prices_t[t]) * step2_results.d[(t, n)] for t in T)
        dem_rows.append({
            "bus": n,
            "total_d_MWh": total_dn,
            "total_utility_24h": utility
        })
    df_dem = pd.DataFrame(dem_rows).sort_values("bus")
    
    # Storage operation (hourly breakdown)
    
    storage_rows = []
    for t in T:
        pch = step2_results.pch[t]
        pdis = step2_results.pdis[t]
        e = step2_results.e[t]
        # Profit = price_t * (pdis_t - pch_t)  (bids=0, arbitrage)
        profit_t = prices_t[t] * (pdis - pch)
        storage_rows.append({
            "hour": t,
            "pch_MW": pch,
            "pdis_MW": pdis,
            "e_MWh": e,
            "price": prices_t[t],
            "profit_t": profit_t
        })
    df_storage = pd.DataFrame(storage_rows)
    
    # Hourly prices
    df_prices = pd.DataFrame([{"hour": t, "price": prices_t[t]} for t in T])

    # Total operating cost (24h)
    
    total_operating_cost = sum(
        input_data.generator_cost[g] * step2_results.pG[(t, g)]
        for t in T for g in input_data.GENERATORS
    )
    
    # Aggregated totals
    
    totals = {
        "objective_welfare_24h": step2_results.objective_value,
        "total_operating_cost_24h": total_operating_cost,
        "total_gen_MWh": df_gen["total_p_MWh"].sum(),
        "total_wind_MWh": df_wind["total_p_MWh"].sum(),
        "total_demand_served_MWh": df_dem["total_d_MWh"].sum(),
        "total_gen_profit_24h": df_gen["total_profit_24h"].sum(),
        "total_wind_profit_24h": df_wind["total_profit_24h"].sum(),
        "total_demand_utility_24h": df_dem["total_utility_24h"].sum(),
        "total_storage_profit_24h": df_storage["profit_t"].sum(),
        "avg_price": df_prices["price"].mean(),
        "max_price": df_prices["price"].max(),
        "min_price": df_prices["price"].min(),
    }

    return df_gen, df_wind, df_dem, df_storage, df_prices, totals