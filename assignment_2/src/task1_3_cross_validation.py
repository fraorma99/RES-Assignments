"""
8-fold cross-validation for the wind farm offering strategy.

Implements Task 1.3 of Assignment 2.

For each fold k = 1, ..., 8:
    - 200 scenarios are used as in-sample (training) set
    - The remaining 1400 scenarios are used as out-of-sample (test) set
    - The offering model is solved on the in-sample set, yielding p_DA*
    - Expected profit is computed both in-sample and out-of-sample
      using the same p_DA*

Results are averaged across the 8 folds.
"""

import numpy as np
from dataclasses import dataclass

from scenario_generation import ScenarioSet



# Output container

@dataclass
class CVResults:
    """
    Container for cross-validation results.

    Attributes
    ----------
    in_sample_profits : np.ndarray
        Expected profits computed on the in-sample (training) set,
        one per fold. Shape (n_folds,).
    out_sample_profits : np.ndarray
        Expected profits computed on the out-of-sample (test) set,
        one per fold. Shape (n_folds,).
    p_DA_per_fold : np.ndarray
        Optimal day-ahead offers from each fold's training problem.
        Shape (n_folds, 24).
    """
    in_sample_profits: np.ndarray
    out_sample_profits: np.ndarray
    p_DA_per_fold: np.ndarray

    @property
    def n_folds(self) -> int:
        return len(self.in_sample_profits)

    @property
    def mean_in_sample(self) -> float:
        return float(self.in_sample_profits.mean())

    @property
    def mean_out_sample(self) -> float:
        return float(self.out_sample_profits.mean())

    @property
    def gap_pct(self) -> float:
        """Percentage gap between in-sample and out-of-sample mean profits."""
        return 100.0 * (self.mean_in_sample - self.mean_out_sample) / self.mean_in_sample



# Helpers to build a sub-ScenarioSet from a slice of the original

def _subset_scenarios(S: ScenarioSet, idx: np.ndarray) -> ScenarioSet:
    """
    Create a new ScenarioSet containing only the scenarios in `idx`.
    Probabilities are renormalized to sum to 1 (uniform over the subset).
    """
    n = len(idx)
    return ScenarioSet(
        wind=S.wind[idx],
        price_da=S.price_da[idx],
        imbalance=S.imbalance[idx],
        probabilities=np.full(n, 1.0 / n),
        P_nom=S.P_nom,
    )



# Profit evaluation given fixed p_DA (no optimization)

def _evaluate_profit_oneprice(p_DA: np.ndarray, S: ScenarioSet) -> float:
    """
    Evaluate the expected profit of a fixed offer schedule p_DA on a
    given scenario set, under the one-price scheme.

    For each (t, w):
        delta[t,w] = p_real[w,t] - p_DA[t]
        balancing price: 1.25*lambda_DA if SI=1, else 0.85*lambda_DA
        profit per scenario: sum_t [ lambda_DA * p_DA + lambda_B * delta ]
    """
    coeff = np.where(S.imbalance == 1, 1.25, 0.85)
    lambda_B = coeff * S.price_da                                # (n_scen, 24)
    delta = S.wind - p_DA[np.newaxis, :]                         # (n_scen, 24)

    da_part = (S.price_da * p_DA[np.newaxis, :]).sum(axis=1)     # (n_scen,)
    bal_part = (lambda_B * delta).sum(axis=1)                    # (n_scen,)
    profit_per_scenario = da_part + bal_part

    return float((S.probabilities * profit_per_scenario).sum())


def _evaluate_profit_twoprice(p_DA: np.ndarray, S: ScenarioSet) -> float:
    """
    Evaluate the expected profit of a fixed offer schedule p_DA on a
    given scenario set, under the two-price scheme. Includes the
    negative-price override (psi_up = psi_dn = system-side price)
    consistent with task1_2_offering_twoprice.py.
    """
    # Standard two-price coefficients
    coeff_up = np.where(S.imbalance == 1, 1.00, 0.85)
    coeff_dn = np.where(S.imbalance == 1, 1.25, 1.00)

    # Negative-price override: collapse to uniform settlement
    coeff_uniform = np.where(S.imbalance == 1, 1.25, 0.85)
    neg_mask = S.price_da < 0
    coeff_up = np.where(neg_mask, coeff_uniform, coeff_up)
    coeff_dn = np.where(neg_mask, coeff_uniform, coeff_dn)

    psi_up = coeff_up * S.price_da
    psi_dn = coeff_dn * S.price_da

    delta = S.wind - p_DA[np.newaxis, :]                         # (n_scen, 24)
    delta_up = np.maximum(delta, 0.0)
    delta_dn = np.maximum(-delta, 0.0)

    da_part = (S.price_da * p_DA[np.newaxis, :]).sum(axis=1)     # (n_scen,)
    up_part = (psi_up * delta_up).sum(axis=1)                    # (n_scen,)
    dn_part = (psi_dn * delta_dn).sum(axis=1)                    # (n_scen,)
    profit_per_scenario = da_part + up_part - dn_part

    return float((S.probabilities * profit_per_scenario).sum())



# Main cross-validation routine

def run_8fold_cv(S: ScenarioSet,
                 scheme: str,
                 fold_size: int = 200,
                 seed: int = 42) -> CVResults:
    """
    Run 8-fold cross-validation on the given scenario set.

    Parameters
    ----------
    S : ScenarioSet
        Full set of 1600 combined scenarios.
    scheme : {'one-price', 'two-price'}
        Which offering model to use during training.
    fold_size : int
        Number of in-sample scenarios per fold (default 200).
    seed : int
        Seed for reproducible scenario shuffling.

    Returns
    -------
    CVResults with arrays of in-sample and out-of-sample expected profits
    for each fold, and the optimal p_DA from each fold.
    """
    if scheme == "one-price":
        from task1_1_offering_oneprice import (
            OneprcInputData, StochasticOfferingOnePrice
        )
        InputDataClass = OneprcInputData
        ModelClass = StochasticOfferingOnePrice
        evaluator = _evaluate_profit_oneprice
    elif scheme == "two-price":
        from task1_2_offering_twoprice import (
            TwoprcInputData, StochasticOfferingTwoPrice
        )
        InputDataClass = TwoprcInputData
        ModelClass = StochasticOfferingTwoPrice
        evaluator = _evaluate_profit_twoprice
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    n_total = S.n_scenarios
    if n_total % fold_size != 0:
        raise ValueError(
            f"Total scenarios ({n_total}) not divisible by fold_size ({fold_size})"
        )
    n_folds = n_total // fold_size

    # Shuffle scenarios reproducibly
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    folds = perm.reshape(n_folds, fold_size)

    in_sample = np.zeros(n_folds)
    out_sample = np.zeros(n_folds)
    p_DA_all = np.zeros((n_folds, S.n_hours))

    for k in range(n_folds):
        in_idx  = folds[k]
        out_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])

        S_in  = _subset_scenarios(S, in_idx)
        S_out = _subset_scenarios(S, out_idx)

        # Train on in-sample
        data_in = InputDataClass(S_in)
        model = ModelClass(data_in)
        res = model.run()

        p_DA_all[k] = res.p_DA
        in_sample[k]  = res.expected_profit                 # already evaluated on S_in
        out_sample[k] = evaluator(res.p_DA, S_out)          # re-evaluate on S_out

    return CVResults(
        in_sample_profits=in_sample,
        out_sample_profits=out_sample,
        p_DA_per_fold=p_DA_all,
    )