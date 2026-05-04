"""
Reporting helpers: print summaries and produce the standard plots used in the assignment.

This module is purely cosmetic / I/O for results — no optimization
or data transformation logic lives here.
"""

import numpy as np
import matplotlib.pyplot as plt


def print_offering_summary(results, scheme_name: str = "one-price"):
    """
    Pretty-print the summary of an offering strategy result.

    Parameters
    ----------
    results : Expando
        The .results object returned by an offering model's run() method.
        Must have attributes: objective_value, expected_profit, p_DA,
        profit_per_scenario.
    scheme_name : str
        Label for the header (e.g. "one-price", "two-price",
        "two-price (CVaR, beta=0.5)").
    """
    print()
    print(f"Offering strategy — {scheme_name}")
    print()
    print(f"Expected profit: {results.expected_profit:>15,.2f} EUR")
    print(f"Objective value: {results.objective_value:>15,.2f} EUR  (sanity check)")
    print()
    print("Hourly DA offers [MW]:")
    for t, p in enumerate(results.p_DA):
        print(f"  Hour {t:2d}: {p:7.2f} MW")
    print()
    print("Profit distribution across scenarios:")
    p = results.profit_per_scenario
    print(f"  min:    {p.min():>15,.2f} EUR")
    print(f"  median: {np.median(p):>15,.2f} EUR")
    print(f"  mean:   {p.mean():>15,.2f} EUR")
    print(f"  max:    {p.max():>15,.2f} EUR")
    print(f"  std:    {p.std():>15,.2f} EUR")


def plot_offering_results(results, scheme_name: str = "one-price",
                          P_nom: float = 500.0, figsize=(14, 4.5)):
    """
    Standard 2-panel plot used in Tasks 1.1, 1.2, 1.4:
      left:  bar chart of optimal hourly DA offers
      right: histogram of profit distribution across scenarios
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    hours = np.arange(len(results.p_DA))

    # --- Left: hourly DA offers ---
    axes[0].bar(hours, results.p_DA, color="steelblue", edgecolor="black")
    axes[0].axhline(P_nom, color="red", linestyle="--", alpha=0.5,
                    label=f"P_nom = {P_nom:.0f} MW")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("DA offer [MW]")
    axes[0].set_title(f"Optimal day-ahead offers ({scheme_name})")
    axes[0].set_xticks(hours)
    axes[0].set_ylim(0, P_nom * 1.1)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # --- Right: profit distribution ---
    p = results.profit_per_scenario / 1000
    axes[1].hist(p, bins=50, color="steelblue", edgecolor="black", alpha=0.85)
    axes[1].axvline(results.expected_profit / 1000, color="red", linestyle="--",
                    linewidth=2,
                    label=f"E[profit] = {results.expected_profit/1000:.1f} k EUR")
    axes[1].axvline(np.median(results.profit_per_scenario) / 1000,
                    color="orange", linestyle="--", linewidth=2,
                    label=f"Median = {np.median(results.profit_per_scenario)/1000:.1f} k EUR")
    axes[1].set_xlabel("Daily profit [k EUR]")
    axes[1].set_ylabel("Number of scenarios")
    axes[1].set_title(f"Profit distribution across scenarios ({scheme_name})")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig, axes