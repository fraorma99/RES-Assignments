"""
Task 2.3: Energinet perspective.

Study how changes in the reliability threshold, from P80 to P100,
affect the optimal ALSO-X reserve bid in-sample and the corresponding
out-of-sample reserve shortfall.

The implementation follows the same empirical ALSO-X logic used in Task 2.1:
for a required reliability q, the optimal reserve bid is the largest bid such
that at most (1 - q) of the in-sample profiles violate availability.
"""

import numpy as np
import pandas as pd

from task2_2_verification import evaluate_out_of_sample


def compute_alsox_bid(A_in: np.ndarray, p_req: float) -> float:
    """
    Compute the empirical ALSO-X reserve bid for a required reliability p_req.

    Parameters
    ----------
    A_in : np.ndarray
        In-sample available reserve per profile, i.e. min load over the hour.
    p_req : float
        Required reliability level in [0, 1], e.g. 0.90 for P90.

    Returns
    -------
    float
        Optimal empirical ALSO-X reserve bid.
    """
    if not (0.0 < p_req <= 1.0):
        raise ValueError("p_req must be in the interval (0, 1].")

    n_in = len(A_in)
    epsilon = 1.0 - p_req
    allowed_violations = int(np.floor(epsilon * n_in))

    A_sorted = np.sort(A_in)
    reserve_bid = A_sorted[allowed_violations]

    return float(reserve_bid)


def evaluate_in_sample(A_in: np.ndarray, reserve_bid: float) -> dict:
    """
    Evaluate a fixed reserve bid on the in-sample profiles.
    """
    reliability = np.mean(A_in >= reserve_bid)
    violation_rate = 1.0 - reliability
    expected_shortfall = np.mean(np.maximum(reserve_bid - A_in, 0.0))
    n_violations = int(np.sum(A_in < reserve_bid))

    return {
        "reliability": float(reliability),
        "violation_rate": float(violation_rate),
        "expected_shortfall": float(expected_shortfall),
        "n_violations": n_violations,
    }


def run_task23(
    profiles_in: np.ndarray,
    profiles_out: np.ndarray,
    reliability_levels: np.ndarray | list = None,
) -> pd.DataFrame:
    """
    Run Task 2.3 for a range of reliability requirements using ALSO-X.

    Parameters
    ----------
    profiles_in : np.ndarray
        In-sample load profiles, shape (n_in, n_minutes).
    profiles_out : np.ndarray
        Out-of-sample load profiles, shape (n_out, n_minutes).
    reliability_levels : array-like, optional
        Reliability thresholds to test. If None, defaults to
        [0.80, 0.82, ..., 1.00].

    Returns
    -------
    pd.DataFrame
        Table with in-sample and out-of-sample metrics for each threshold.
    """
    if reliability_levels is None:
        reliability_levels = np.round(np.arange(0.80, 1.001, 0.02), 2)

    A_in = profiles_in.min(axis=1)

    rows = []

    for p_req in reliability_levels:
        reserve_bid = compute_alsox_bid(A_in, p_req)

        in_sample = evaluate_in_sample(A_in, reserve_bid)
        out_of_sample = evaluate_out_of_sample(profiles_out, reserve_bid)

        rows.append({
            "Reliability requirement": f"P{int(round(100 * p_req))}",
            "Required reliability level": p_req,
            "Optimal ALSO-X reserve bid [kW]": reserve_bid,

            "In-sample reliability": in_sample["reliability"],
            "In-sample violation rate": in_sample["violation_rate"],
            "In-sample expected shortfall [kW]": in_sample["expected_shortfall"],
            "In-sample violating profiles": in_sample["n_violations"],

            "Out-of-sample reliability": out_of_sample["reliability"],
            "Out-of-sample violation rate": out_of_sample["violation_rate"],
            "Out-of-sample expected shortfall [kW]": out_of_sample["expected_shortfall"],
            "Out-of-sample violating profiles": out_of_sample["n_violations"],
        })

    results = pd.DataFrame(rows)

    results["Reserve bid reduction vs P80 [kW]"] = (
        results.loc[0, "Optimal ALSO-X reserve bid [kW]"]
        - results["Optimal ALSO-X reserve bid [kW]"]
    )

    results["Out-of-sample shortfall reduction vs P80 [kW]"] = (
        results.loc[0, "Out-of-sample expected shortfall [kW]"]
        - results["Out-of-sample expected shortfall [kW]"]
    )

    return results


if __name__ == "__main__":
    from task2_1_fcrd_up import profiles_in, profiles_out

    task23_results = run_task23(profiles_in, profiles_out)
    print(task23_results.round(4).to_string(index=False))