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

    totals = {
        "market_price": price,
        "objective_welfare": step1_results.objective_value,
        "total_gen_MW": df_gen["p_MW"].sum(),
        "total_wind_MW": df_wind["p_MW"].sum(),
        "total_demand_served_MW": df_dem["d_MW"].sum(),
        "total_gen_profit": df_gen["profit"].sum(),
        "total_wind_profit": df_wind["profit"].sum(),
        "total_demand_utility": df_dem["utility"].sum(),
    }

    return df_gen, df_wind, df_dem, totals