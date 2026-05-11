"""
Task 2.2: Out-of-sample verification of the P90 requirement.

No new optimization is performed here. The reserve bids obtained in
Task 2.1 are fixed and evaluated on the 200 out-of-sample load profiles.

The available FCR-D UP reserve of each profile is defined as the minimum
load during the hour, because the reserve must be available for all
60 minutes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def available_reserve_per_profile(profiles: np.ndarray) -> np.ndarray:
    """
    Compute the available FCR-D UP reserve for each load profile.

    Since the reserve must be available during the full hour, the available
    reserve of a profile is the minimum load over its 60 minutes.
    """
    return profiles.min(axis=1)


def evaluate_out_of_sample(profiles_out: np.ndarray, reserve_bid: float) -> dict:
    """
    Evaluate a fixed FCR-D UP reserve bid on out-of-sample profiles.
    """
    available_reserve = available_reserve_per_profile(profiles_out)

    reliability = np.mean(available_reserve >= reserve_bid)
    violation_rate = 1.0 - reliability
    expected_shortfall = np.mean(np.maximum(reserve_bid - available_reserve, 0.0))
    n_violations = int(np.sum(available_reserve < reserve_bid))

    return {
        "reliability": reliability,
        "violation_rate": violation_rate,
        "expected_shortfall": expected_shortfall,
        "n_violations": n_violations,
    }


def run_task22(
    profiles_out: np.ndarray,
    R_alsox: float,
    R_cvar: float,
) -> pd.DataFrame:
    """
    Run Task 2.2 for the ALSO-X and CVaR reserve bids.
    """
    alsox_oos = evaluate_out_of_sample(profiles_out, R_alsox)
    cvar_oos = evaluate_out_of_sample(profiles_out, R_cvar)

    results = pd.DataFrame({
        "Method": ["ALSO-X", "CVaR"],
        "Reserve bid from Task 2.1 [kW]": [R_alsox, R_cvar],
        "Out-of-sample reliability": [
            alsox_oos["reliability"],
            cvar_oos["reliability"],
        ],
        "Out-of-sample violation rate": [
            alsox_oos["violation_rate"],
            cvar_oos["violation_rate"],
        ],
        "Out-of-sample expected shortfall [kW]": [
            alsox_oos["expected_shortfall"],
            cvar_oos["expected_shortfall"],
        ],
        "Number of violating profiles": [
            alsox_oos["n_violations"],
            cvar_oos["n_violations"],
        ],
    })

    return results


def plot_oos_available_reserve_distribution(
    profiles_out: np.ndarray,
    R_alsox: float,
    R_cvar: float,
    save_path=None,
):
    """
    Plot the distribution of out-of-sample available reserve A_omega
    across the 200 profiles.

    Vertical lines show the reserve bids obtained in Task 2.1 using
    ALSO-X and CVaR. Profiles to the left of each line are violations.
    """
    A_out = available_reserve_per_profile(profiles_out)

    n_viol_alsox = int(np.sum(A_out < R_alsox))
    n_viol_cvar = int(np.sum(A_out < R_cvar))

    plt.figure(figsize=(9, 5))

    plt.hist(A_out, bins=20, alpha=0.75)

    plt.axvline(
        R_alsox,
        linestyle="--",
        linewidth=2,
        label=f"ALSO-X bid = {R_alsox:.2f} kW ({n_viol_alsox} violations)",
    )

    plt.axvline(
        R_cvar,
        linestyle="-.",
        linewidth=2,
        label=f"CVaR bid = {R_cvar:.2f} kW ({n_viol_cvar} violations)",
    )

    plt.xlabel(r"Available reserve $A_\omega$ [kW]")
    plt.ylabel("Number of profiles")
    plt.title("Task 2.2: Out-of-sample available reserve distribution")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()