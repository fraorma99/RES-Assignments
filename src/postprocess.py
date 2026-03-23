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
# ========================
# STEP 5: BALANCING MARKET RESULTS
# ========================
# Append this block to postprocess.py


def step5_build_summary_tables(
    step5_results,
    step5_input,
    da_results,
    da_input,
    failed_generator,
    wind_da_dispatch: dict,
    wind_real: dict,
    system_imbalance: float,
):
    λ_DA = da_results.market_price
    λ_B  = step5_results.balancing_price

    # Sets of BSPs (activated balancing providers) — remunerated at λ_B directly,
    # NOT subject to one-price/two-price imbalance settlement
    BSP_UP = {g for g in step5_input.GENERATORS_UP if step5_results.r_up.get(g, 0.0) > 1e-6}
    BSP_DN = {g for g in step5_input.GENERATORS_DN if step5_results.r_dn.get(g, 0.0) > 1e-6}
    BSPS   = BSP_UP | BSP_DN

    # ── Offer stack ──────────────────────────────────────────────────────────
    offer_rows = []
    all_providers = set(step5_input.GENERATORS_UP) | set(step5_input.GENERATORS_DN)
    for g in sorted(all_providers):
        offer_rows.append({
            "generator":         g,
            "up_offer_price":    step5_input.up_price.get(g, None),
            "up_capacity_MW":    step5_input.up_cap.get(g, 0.0),
            "r_up_activated_MW": step5_results.r_up.get(g, 0.0),
            "dn_offer_price":    step5_input.dn_price.get(g, None),
            "dn_capacity_MW":    step5_input.dn_cap.get(g, 0.0),
            "r_dn_activated_MW": step5_results.r_dn.get(g, 0.0),
        })
    df_offers = pd.DataFrame(offer_rows)

    # ── Generator profits ────────────────────────────────────────────────────
    # Three categories:
    #   BSP: activated balancing provider → remunerated at λ_B, no BRP settlement
    #   BRP with deviation: has DA schedule but deviates (G8) → BRP settlement applies
    #   Neutral: delivered exactly DA schedule → same under both schemes
    gen_rows = []
    for g in da_input.GENERATORS:
        pG_DA = da_results.pG[g]
        c_g   = da_input.generator_cost[g]

        da_profit = (λ_DA - c_g) * pG_DA

        if g == failed_generator:
            r_up_g, r_dn_g = 0.0, 0.0
            deviation = -pG_DA
            role = "BRP (failed)"

            # G8 doesn't produce → saves production cost c_g * pG_DA
            # but must buy back missing energy at λ_B
            cost_savings = c_g * pG_DA
            settlement_1p = λ_B * deviation     # cash penalty
            settlement_2p = λ_B * deviation     # system is short, Δ<0 → same

            balancing_profit_1p = settlement_1p + cost_savings
            balancing_profit_2p = settlement_2p + cost_savings
            total_profit_1p = da_profit + balancing_profit_1p
            total_profit_2p = da_profit + balancing_profit_2p

        elif g in BSPS:
            # BSP: activated by TSO, remunerated directly at λ_B
            # Not subject to BRP one/two-price settlement
            r_up_g = step5_results.r_up.get(g, 0.0)
            r_dn_g = step5_results.r_dn.get(g, 0.0)
            deviation = r_up_g - r_dn_g
            role = "BSP"

            balancing_profit = (λ_B - c_g) * r_up_g - (λ_B - c_g) * r_dn_g
            # Same under both schemes — BSPs are not BRPs
            balancing_profit_1p = balancing_profit
            balancing_profit_2p = balancing_profit
            total_profit_1p = da_profit + balancing_profit_1p
            total_profit_2p = da_profit + balancing_profit_2p

        else:
            # Neutral: delivered DA schedule exactly, no deviation
            r_up_g, r_dn_g = 0.0, 0.0
            deviation = 0.0
            role = ""
            balancing_profit_1p = 0.0
            balancing_profit_2p = 0.0
            total_profit_1p = da_profit
            total_profit_2p = da_profit

        gen_rows.append({
            "generator":          g,
            "role":               role,
            "pG_DA_MW":           pG_DA,
            "marginal_cost":      c_g,
            "deviation_MW":       deviation,
            "da_profit":          da_profit,
            "balancing_profit_1p": balancing_profit_1p,
            "balancing_profit_2p": balancing_profit_2p,
            "total_profit_1p":    total_profit_1p,
            "total_profit_2p":    total_profit_2p,
        })
    df_gen = pd.DataFrame(gen_rows).sort_values("generator")

    # ── Wind profits ─────────────────────────────────────────────────────────
    wind_rows = []
    for w in da_input.WINDS:
        pW_DA   = wind_da_dispatch[w]
        pW_real = wind_real[w]
        delta   = pW_real - pW_DA

        da_profit = λ_DA * pW_DA

        # One-price: all deviations at λ_B
        settlement_1p = λ_B * delta

        # Two-price: depends on system direction
        if system_imbalance > 0:
            # System is short: under-producers penalised (λ_B), over-producers at λ_DA
            settlement_2p = λ_B * delta if delta < 0 else λ_DA * delta
        else:
            # System is long: over-producers penalised (λ_B), under-producers at λ_DA
            settlement_2p = λ_B * delta if delta > 0 else λ_DA * delta

        wind_rows.append({
            "wind":                  w,
            "pW_DA_MW":              pW_DA,
            "pW_real_MW":            pW_real,
            "deviation_MW":          delta,
            "da_profit":             da_profit,
            "settlement_1price":     settlement_1p,
            "settlement_2price":     settlement_2p,
            "total_profit_1price":   da_profit + settlement_1p,
            "total_profit_2price":   da_profit + settlement_2p,
        })
    df_wind = pd.DataFrame(wind_rows).sort_values("wind")

    # ── Totals ───────────────────────────────────────────────────────────────
    totals = {
        "da_price":                    λ_DA,
        "balancing_price":             λ_B,
        "system_imbalance_MW":         system_imbalance,
        "total_upward_activated_MW":   sum(step5_results.r_up[g] for g in step5_input.GENERATORS_UP),
        "total_downward_activated_MW": sum(step5_results.r_dn[g] for g in step5_input.GENERATORS_DN),
        "load_curtailment_MW":         step5_results.LC,
        "total_gen_profit_1p":         df_gen["total_profit_1p"].sum(),
        "total_gen_profit_2p":         df_gen["total_profit_2p"].sum(),
        "total_wind_profit_1price":    df_wind["total_profit_1price"].sum(),
        "total_wind_profit_2price":    df_wind["total_profit_2price"].sum(),
    }

    return df_offers, df_gen, df_wind, totals


def step5_build_comparison_table(df_gen, df_wind):
    """
    Builds a concise grouped summary table for the one-price vs two-price
    comparison, organised by party role as the assignment requests:

    Groups:
      - 'Causes imbalance (BRP)' : G8 (failed) + wind farms with Δ < 0
      - 'Helps system (BRP)'     : wind farms with Δ > 0
      - 'Provides service (BSP)' : generators activated by TSO (G1, G2, G3)
      - 'Neutral'                : delivered DA schedule exactly

    Returns df_comparison: one row per group with aggregated profits.
    """
    rows = []

    # ── BSPs (activated generators) ───────────────────────────────────────────
    bsp_df = df_gen[df_gen["role"] == "BSP"]
    if not bsp_df.empty:
        rows.append({
            "group":           "Provides service (BSP)",
            "members":         ", ".join(f"G{g}" for g in bsp_df["generator"]),
            "da_profit":       bsp_df["da_profit"].sum(),
            "bal_settlement_1p": bsp_df["balancing_profit_1p"].sum(),
            "bal_settlement_2p": bsp_df["balancing_profit_2p"].sum(),
            "total_profit_1p": bsp_df["total_profit_1p"].sum(),
            "total_profit_2p": bsp_df["total_profit_2p"].sum(),
            "note": "Remunerated at λ_B; settlement scheme does not apply"
        })

    # ── G8: causes imbalance (BRP, failed generator) ──────────────────────────
    brp_gen_df = df_gen[df_gen["role"] == "BRP (failed)"]
    if not brp_gen_df.empty:
        rows.append({
            "group":           "Causes imbalance (BRP)",
            "members":         ", ".join(f"G{g}" for g in brp_gen_df["generator"]),
            "da_profit":       brp_gen_df["da_profit"].sum(),
            "bal_settlement_1p": brp_gen_df["balancing_profit_1p"].sum(),
            "bal_settlement_2p": brp_gen_df["balancing_profit_2p"].sum(),
            "total_profit_1p": brp_gen_df["total_profit_1p"].sum(),
            "total_profit_2p": brp_gen_df["total_profit_2p"].sum(),
            "note": "Δ < 0 in short system → penalised at λ_B under both schemes"
        })

    # ── Wind farms causing imbalance (Δ < 0) ──────────────────────────────────
    wind_bad = df_wind[df_wind["deviation_MW"] < 0]
    if not wind_bad.empty:
        rows.append({
            "group":           "Causes imbalance (BRP)",
            "members":         ", ".join(f"W{w}" for w in wind_bad["wind"]),
            "da_profit":       wind_bad["da_profit"].sum(),
            "bal_settlement_1p": (wind_bad["total_profit_1price"] - wind_bad["da_profit"]).sum(),
            "bal_settlement_2p": (wind_bad["total_profit_2price"] - wind_bad["da_profit"]).sum(),
            "total_profit_1p": wind_bad["total_profit_1price"].sum(),
            "total_profit_2p": wind_bad["total_profit_2price"].sum(),
            "note": "Δ < 0 in short system → penalised at λ_B under both schemes"
        })

    # ── Wind farms helping system (Δ > 0) ─────────────────────────────────────
    wind_good = df_wind[df_wind["deviation_MW"] > 0]
    if not wind_good.empty:
        rows.append({
            "group":           "Helps system (BRP)",
            "members":         ", ".join(f"W{w}" for w in wind_good["wind"]),
            "da_profit":       wind_good["da_profit"].sum(),
            "bal_settlement_1p": (wind_good["total_profit_1price"] - wind_good["da_profit"]).sum(),
            "bal_settlement_2p": (wind_good["total_profit_2price"] - wind_good["da_profit"]).sum(),
            "total_profit_1p": wind_good["total_profit_1price"].sum(),
            "total_profit_2p": wind_good["total_profit_2price"].sum(),
            "note": "Δ > 0 in short system → 1-price rewards at λ_B, 2-price only at λ_DA"
        })

    # ── Neutral generators (no deviation, no balancing service) ───────────────
    neutral_df = df_gen[df_gen["role"] == ""]
    neutral_dispatched = neutral_df[neutral_df["pG_DA_MW"] > 1e-6]
    if not neutral_dispatched.empty:
        rows.append({
            "group":           "Neutral (no deviation)",
            "members":         ", ".join(f"G{g}" for g in neutral_dispatched["generator"]),
            "da_profit":       neutral_dispatched["da_profit"].sum(),
            "bal_settlement_1p": 0.0,
            "bal_settlement_2p": 0.0,
            "total_profit_1p": neutral_dispatched["total_profit_1p"].sum(),
            "total_profit_2p": neutral_dispatched["total_profit_2p"].sum(),
            "note": "Delivered DA schedule exactly; unaffected by settlement scheme"
        })

    df_comparison = pd.DataFrame(rows, columns=[
        "group", "members", "da_profit",
        "bal_settlement_1p", "bal_settlement_2p",
        "total_profit_1p", "total_profit_2p", "note"
    ])

    return df_comparison

# ========================
# STEP 6: RESERVE MARKET RESULTS
# ========================



def step6_build_summary_tables(
    res6a,          # Step6aReserveMarket results
    inp6a,          # Step6aInputData
    res6b,          # Step6bDAMarket results
    inp6b,          # Step6bInputData
    res_step1,      # Step1 results (for comparison)
    inp_step1,      # Step1 input (for comparison)
):

    λ_DA_new   = res6b.market_price      # new DA price after reserve
    λ_DA_old   = res_step1.market_price  # Step 1 DA price (no reserve)
    λ_up_res   = res6a.lambda_up_res     # upward reserve price
    λ_dn_res   = res6a.lambda_dn_res     # downward reserve price

    # ── Reserve commitment table ──────────────────────────────────────────────
    res_rows = []
    all_res_gens = set(inp6a.GENERATORS_UP_RES) | set(inp6a.GENERATORS_DN_RES)
    for g in sorted(all_res_gens):
        r_up = res6a.r_up.get(g, 0.0)
        r_dn = res6a.r_dn.get(g, 0.0)
        # Revenue = committed capacity * reserve price
        up_revenue = λ_up_res * r_up
        dn_revenue = λ_dn_res * r_dn
        res_rows.append({
            "generator":        g,
            "up_offer_price":   inp6a.up_res_price.get(g, None),
            "R_up_max_MW":      inp6a.R_up_max.get(g, 0.0),
            "r_up_committed_MW": r_up,
            "up_reserve_revenue": up_revenue,
            "dn_offer_price":   inp6a.dn_res_price.get(g, None),
            "R_dn_max_MW":      inp6a.R_dn_max.get(g, 0.0),
            "r_dn_committed_MW": r_dn,
            "dn_reserve_revenue": dn_revenue,
            "total_reserve_revenue": up_revenue + dn_revenue,
        })
    df_reserve = pd.DataFrame(res_rows)

    # ── Generator DA dispatch & total profit (reserve revenue + DA profit) ───
    gen_rows = []
    for g in inp6b.GENERATORS:
        pG_new   = res6b.pG[g]
        pG_old   = res_step1.pG[g]
        c_g      = inp6b.generator_cost[g]

        # DA profit at new price
        da_profit_new  = (λ_DA_new - c_g) * pG_new
        # DA profit at old price (Step 1, for comparison)
        da_profit_old  = (λ_DA_old - c_g) * pG_old

        # Reserve revenue
        r_up = res6a.r_up.get(g, 0.0)
        r_dn = res6a.r_dn.get(g, 0.0)
        res_revenue = λ_up_res * r_up + λ_dn_res * r_dn

        # Total profit = DA profit + reserve revenue
        total_profit_new = da_profit_new + res_revenue

        gen_rows.append({
            "generator":          g,
            "marginal_cost":      c_g,
            "pG_step1_MW":        pG_old,
            "pG_new_MW":          pG_new,
            "dispatch_change_MW": pG_new - pG_old,
            "r_up_MW":            r_up,
            "r_dn_MW":            r_dn,
            "da_profit_step1":    da_profit_old,
            "da_profit_new":      da_profit_new,
            "reserve_revenue":    res_revenue,
            "total_profit_new":   total_profit_new,
        })
    df_gen = pd.DataFrame(gen_rows).sort_values("generator")

    # ── Wind dispatch & profit ────────────────────────────────────────────────
    wind_rows = []
    for w in inp6b.WINDS:
        pw_new = res6b.pW[w]
        pw_old = res_step1.pW[w]
        wind_rows.append({
            "wind":               w,
            "p_MW_step1":         pw_old,
            "p_MW_new":           pw_new,
            "profit_step1":       λ_DA_old * pw_old,
            "profit_new":         λ_DA_new * pw_new,
        })
    df_wind = pd.DataFrame(wind_rows).sort_values("wind")

    # ── Demand served & utility ───────────────────────────────────────────────
    dem_rows = []
    for n in inp6b.LOAD_BUSES:
        dn_new = res6b.d[n]
        dn_old = res_step1.d[n]
        bn     = inp6b.demand_bid[n]
        dem_rows.append({
            "bus":              n,
            "d_MW_step1":       dn_old,
            "d_MW_new":         dn_new,
            "bid_price":        bn,
            "utility_step1":    (bn - λ_DA_old) * dn_old,
            "utility_new":      (bn - λ_DA_new) * dn_new,
        })
    df_dem = pd.DataFrame(dem_rows).sort_values("bus")

    # ── Totals ────────────────────────────────────────────────────────────────
    total_op_cost_new = (df_gen["marginal_cost"] * df_gen["pG_new_MW"]).sum()
    total_op_cost_old = (df_gen["marginal_cost"] * df_gen["pG_step1_MW"]).sum()

    totals = {
        # Prices
        "da_price_step1":           λ_DA_old,
        "da_price_new":             λ_DA_new,
        "da_price_change":          λ_DA_new - λ_DA_old,
        "lambda_up_res":            λ_up_res,
        "lambda_dn_res":            λ_dn_res,
        # Reserve market
        "R_up_req_MW":              inp6a.R_up_req,
        "R_dn_req_MW":              inp6a.R_dn_req,
        "total_reserve_cost":       res6a.objective_value,
        # DA market
        "social_welfare_step1":     res_step1.objective_value,
        "social_welfare_new":       res6b.objective_value,
        "total_op_cost_step1":      total_op_cost_old,
        "total_op_cost_new":        total_op_cost_new,
        "total_gen_MW_new":         df_gen["pG_new_MW"].sum(),
        "total_wind_MW_new":        df_wind["p_MW_new"].sum(),
        "total_demand_served_new":  df_dem["d_MW_new"].sum(),
        # Profits
        "total_gen_profit_step1":   df_gen["da_profit_step1"].sum(),
        "total_gen_profit_new":     df_gen["total_profit_new"].sum(),
        "total_wind_profit_step1":  df_wind["profit_step1"].sum(),
        "total_wind_profit_new":    df_wind["profit_new"].sum(),
        "total_demand_utility_new": df_dem["utility_new"].sum(),
        # Combined reserve + DA cost (total system cost, the relevant metric)
        "total_system_cost":        res6a.objective_value + total_op_cost_new,
        "total_system_cost_step1":  total_op_cost_old,
    }

    return df_reserve, df_gen, df_wind, df_dem, totals
