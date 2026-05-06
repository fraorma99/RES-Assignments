"""
Task 2.2: Out-of-sample verification of the P90 requirement.

No new optimization is performed here. The reserve bids obtained in
Task 2.1 are fixed and evaluated on the 200 out-of-sample load profiles.
"""

import numpy as np
import pandas as pd


def evaluate_out_of_sample(profiles_out: np.ndarray, reserve_bid: float) -> dict:
    """
    Evaluate a fixed FCR-D UP reserve bid on out-of-sample profiles.

    Since the reserve must be available during the full hour, the available
    reserve of each profile is the minimum load over the 60 minutes.
    """
    available_reserve = profiles_out.min(axis=1)

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
    
    #Run Task 2.2 for the ALSO-X and CVaR reserve bids.
    
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