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

# ========================
# STEP 3A: NODAL RESULTS
# ========================

def step3_build_summary_tables(step3_results, input_data):
    """
    Returns:
    - df_gen: generator dispatch, nodal price, profit
    - df_wind: wind dispatch, nodal price, profit
    - df_dem: demand served, nodal price, utility
    - df_bus: nodal injections, angles, prices
    - df_line: line flows and congestion status
    - totals: dict with aggregate quantities
    """

    gen_rows = []
    for g in input_data.GENERATORS:
        bus = input_data.generator_bus[g]
        pg = step3_results.pG[g]
        cg = input_data.generator_cost[g]
        price = step3_results.nodal_prices[bus]

        gen_rows.append({
            "generator": g,
            "bus": bus,
            "p_MW": pg,
            "marginal_cost": cg,
            "nodal_price": price,
            "profit": (price - cg) * pg
        })
    df_gen = pd.DataFrame(gen_rows).sort_values(["bus", "generator"])

    wind_rows = []
    for w in input_data.WINDS:
        bus = input_data.wind_bus[w]
        pw = step3_results.pW[w]
        price = step3_results.nodal_prices[bus]

        wind_rows.append({
            "wind": w,
            "bus": bus,
            "p_MW": pw,
            "nodal_price": price,
            "profit": price * pw
        })
    df_wind = pd.DataFrame(wind_rows).sort_values(["bus", "wind"])

    dem_rows = []
    for n in input_data.LOAD_BUSES:
        dn = step3_results.d[n]
        bid = input_data.demand_bid[n]
        price = step3_results.nodal_prices[n]

        dem_rows.append({
            "bus": n,
            "d_MW": dn,
            "bid_price": bid,
            "nodal_price": price,
            "utility": (bid - price) * dn
        })
    df_dem = pd.DataFrame(dem_rows).sort_values("bus")

    bus_rows = []
    for n in input_data.BUSES:
        gen_n = sum(step3_results.pG[g] for g in input_data.GENERATORS if input_data.generator_bus[g] == n)
        wind_n = sum(step3_results.pW[w] for w in input_data.WINDS if input_data.wind_bus[w] == n)
        dem_n = step3_results.d[n] if n in input_data.LOAD_BUSES else 0.0

        bus_rows.append({
            "bus": n,
            "theta_rad": step3_results.theta[n],
            "nodal_price": step3_results.nodal_prices[n],
            "gen_MW": gen_n,
            "wind_MW": wind_n,
            "demand_MW": dem_n,
            "net_injection_MW": gen_n + wind_n - dem_n
        })
    df_bus = pd.DataFrame(bus_rows).sort_values("bus")

    line_rows = []
    tol = 1e-5
    for l in input_data.LINES:
        flow = step3_results.f[l]
        fmax = input_data.line_fmax[l]

        line_rows.append({
            "line": l,
            "from_bus": input_data.line_from[l],
            "to_bus": input_data.line_to[l],
            "B": input_data.line_B[l],
            "flow_MW": flow,
            "Fmax_MW": fmax,
            "loading_pct": 100 * abs(flow) / fmax if fmax > 0 else None,
            "congested": abs(abs(flow) - fmax) <= tol
        })
    df_line = pd.DataFrame(line_rows).sort_values("line")

    total_operating_cost = (df_gen["marginal_cost"] * df_gen["p_MW"]).sum()

    totals = {
        "objective_welfare": step3_results.objective_value,
        "total_operating_cost": total_operating_cost,
        "total_gen_MW": df_gen["p_MW"].sum(),
        "total_wind_MW": df_wind["p_MW"].sum(),
        "total_demand_served_MW": df_dem["d_MW"].sum(),
        "total_gen_profit": df_gen["profit"].sum(),
        "total_wind_profit": df_wind["profit"].sum(),
        "total_demand_utility": df_dem["utility"].sum(),
        "min_nodal_price": df_bus["nodal_price"].min(),
        "max_nodal_price": df_bus["nodal_price"].max(),
        "price_spread": df_bus["nodal_price"].max() - df_bus["nodal_price"].min(),
        "n_congested_lines": int(df_line["congested"].sum()),
    }

    return df_gen, df_wind, df_dem, df_bus, df_line, totals

# ========================
# STEP 3B: ZONAL RESULTS
# ========================

def step3_zonal_build_summary_tables(step3z_results, input_data):
    """
    Returns:
    - df_gen: generator dispatch, zonal price, profit
    - df_wind: wind dispatch, zonal price, profit
    - df_dem: demand served, zonal price, utility
    - df_zone: zone balances and prices
    - df_transfer: interzonal transfer results
    - totals: dict with aggregate quantities
    """

    gen_rows = []
    for g in input_data.GENERATORS:
        bus = input_data.generator_bus[g]
        zone = input_data.BUS_ZONE[bus]
        pg = step3z_results.pG[g]
        cg = input_data.generator_cost[g]
        price = step3z_results.zonal_prices[zone]

        gen_rows.append({
            "generator": g,
            "bus": bus,
            "zone": zone,
            "p_MW": pg,
            "marginal_cost": cg,
            "zonal_price": price,
            "profit": (price - cg) * pg
        })
    df_gen = pd.DataFrame(gen_rows).sort_values(["zone", "bus", "generator"])

    wind_rows = []
    for w in input_data.WINDS:
        bus = input_data.wind_bus[w]
        zone = input_data.BUS_ZONE[bus]
        pw = step3z_results.pW[w]
        price = step3z_results.zonal_prices[zone]

        wind_rows.append({
            "wind": w,
            "bus": bus,
            "zone": zone,
            "p_MW": pw,
            "zonal_price": price,
            "profit": price * pw
        })
    df_wind = pd.DataFrame(wind_rows).sort_values(["zone", "bus", "wind"])

    dem_rows = []
    for n in input_data.LOAD_BUSES:
        zone = input_data.BUS_ZONE[n]
        dn = step3z_results.d[n]
        bid = input_data.demand_bid[n]
        price = step3z_results.zonal_prices[zone]

        dem_rows.append({
            "bus": n,
            "zone": zone,
            "d_MW": dn,
            "bid_price": bid,
            "zonal_price": price,
            "utility": (bid - price) * dn
        })
    df_dem = pd.DataFrame(dem_rows).sort_values(["zone", "bus"])

    zone_rows = []
    for z in input_data.ZONES:
        gen_z = sum(step3z_results.pG[g] for g in input_data.GENERATORS if input_data.BUS_ZONE[input_data.generator_bus[g]] == z)
        wind_z = sum(step3z_results.pW[w] for w in input_data.WINDS if input_data.BUS_ZONE[input_data.wind_bus[w]] == z)
        dem_z = sum(step3z_results.d[n] for n in input_data.LOAD_BUSES if input_data.BUS_ZONE[n] == z)

        net_export = sum(
            step3z_results.T[k] if k[0] == z else -step3z_results.T[k]
            for k in input_data.INTERFACES if z in k
        )

        zone_rows.append({
            "zone": z,
            "zonal_price": step3z_results.zonal_prices[z],
            "gen_MW": gen_z,
            "wind_MW": wind_z,
            "demand_MW": dem_z,
            "net_export_MW": net_export
        })
    df_zone = pd.DataFrame(zone_rows).sort_values("zone")

    transfer_rows = []
    for k in input_data.INTERFACES:
        transfer_rows.append({
            "interface": f"{k[0]}->{k[1]}",
            "transfer_MW": step3z_results.T[k],
            "ATC_MW": input_data.atc[k],
            "loading_pct": 100 * abs(step3z_results.T[k]) / input_data.atc[k] if input_data.atc[k] > 0 else None
        })
    df_transfer = pd.DataFrame(transfer_rows)

    total_operating_cost = (df_gen["marginal_cost"] * df_gen["p_MW"]).sum()

    totals = {
        "objective_welfare": step3z_results.objective_value,
        "total_operating_cost": total_operating_cost,
        "total_gen_MW": df_gen["p_MW"].sum(),
        "total_wind_MW": df_wind["p_MW"].sum(),
        "total_demand_served_MW": df_dem["d_MW"].sum(),
        "total_gen_profit": df_gen["profit"].sum(),
        "total_wind_profit": df_wind["profit"].sum(),
        "total_demand_utility": df_dem["utility"].sum(),
    }

    return df_gen, df_wind, df_dem, df_zone, df_transfer, totals