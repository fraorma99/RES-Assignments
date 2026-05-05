"""
Stochastic offering strategy for a price-taking wind farm under the
TWO-PRICE balancing scheme. Implements Task 1.2 of Assignment 2.

The structural difference compared to the one-price scheme is that the
balancing settlement price now depends on the sign of the wind farm's
imbalance:

    psi_up[t,w]   (price for excess production Delta_up):
        - if SI = 1 (system deficit, our excess HELPS):  psi_up = lambda_DA
        - if SI = 0 (system surplus, our excess HURTS):  psi_up = 0.85 * lambda_DA

    psi_dn[t,w]   (price for shortfall Delta_dn):
        - if SI = 1 (system deficit, our shortfall HURTS):  psi_dn = 1.25 * lambda_DA
        - if SI = 0 (system surplus, our shortfall HELPS):  psi_dn = lambda_DA

The model maximizes expected profit:

    sum_t sum_w  pi_w * [ lambda_DA[t,w] * p_DA[t]
                          + psi_up[t,w]   * delta_up[t,w]
                          - psi_dn[t,w]   * delta_dn[t,w] ]

subject to the same imbalance-decomposition constraints as Task 1.1.

Note on negative DA prices:
    The decomposition delta = delta_up - delta_dn is an LP relaxation of the
    physical complementarity (delta_up * delta_dn = 0). With strictly positive
    DA prices, the LP optimum naturally enforces this complementarity because
    psi_dn >= psi_up always holds. With negative DA prices the inequality
    can flip, and the LP could in principle inflate both delta_up and delta_dn
    indefinitely, leading to an unbounded model.
    To prevent this, we cap both delta_up and delta_dn at P_nom (a physical
    upper bound, since the hourly imbalance cannot exceed the nominal capacity).
    We then verify ex-post that the strict complementarity holds in the
    optimal solution, and warn the user otherwise.
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from scenario_generation import ScenarioSet



# Generic Expando container (same idiom as Task 1.1 and class exercises)

class Expando:
    """Generic data container; attributes can be added on the fly."""
    pass



# Input data wrapper

class TwoprcInputData:
    """
    Pre-processed input data for the two-price stochastic offering model.

    Builds the per-scenario settlement prices psi_up and psi_dn from the
    day-ahead price and the binary system-imbalance signal of each scenario.

    Standard two-price rule (lambda_DA >= 0):
        - excess (Delta_up):    paid at lambda_DA  if SI=1, at 0.85*lambda_DA if SI=0
        - shortfall (Delta_dn): charged at 1.25*lambda_DA if SI=1, at lambda_DA if SI=0

    Negative-price hours (lambda_DA < 0):
        - both directions are settled at the uniform system-side balancing price
          (1.25*lambda_DA if SI=1, 0.85*lambda_DA if SI=0).
        This collapses to the one-price scheme and prevents LP degeneracy.
    """

    def __init__(self, scenarios: ScenarioSet):
        self.S = scenarios

        self.T = list(range(scenarios.n_hours))               # 0..23
        self.W = list(range(scenarios.n_scenarios))           # 0..1599
        self.P_nom = scenarios.P_nom

        self.pi = scenarios.probabilities                     # (n_scen,)

        self.p_real = scenarios.wind                          # (n_scen, 24) MW
        self.lambda_DA = scenarios.price_da                   # (n_scen, 24) EUR/MWh

        # Standard two-price coefficients (apply when lambda_DA >= 0)
        # SI = 1 (deficit):  psi_up = lambda_DA,        psi_dn = 1.25 * lambda_DA
        # SI = 0 (surplus):  psi_up = 0.85 * lambda_DA, psi_dn = lambda_DA
        coeff_up = np.where(scenarios.imbalance == 1, 1.00, 0.85)
        coeff_dn = np.where(scenarios.imbalance == 1, 1.25, 1.00)

        # Negative-price override: collapse to uniform balancing settlement
        # (one-price-like). Use the system-side balancing coefficient on both
        # directions to remove the LP degeneracy that occurs when psi_dn < psi_up.
        coeff_uniform = np.where(scenarios.imbalance == 1, 1.25, 0.85)
        neg_mask = scenarios.price_da < 0
        coeff_up = np.where(neg_mask, coeff_uniform, coeff_up)
        coeff_dn = np.where(neg_mask, coeff_uniform, coeff_dn)

        self.psi_up = coeff_up * scenarios.price_da           # (n_scen, 24)
        self.psi_dn = coeff_dn * scenarios.price_da           # (n_scen, 24)



# Optimization model

class StochasticOfferingTwoPrice:
    """
    Stochastic LP for the wind farm's day-ahead offering strategy under
    the two-price balancing scheme.

    Usage
    -----
    >>> data  = TwoprcInputData(scenario_set)
    >>> model = StochasticOfferingTwoPrice(data)
    >>> res   = model.run()
    >>> print(res.expected_profit)
    >>> print(res.p_DA)        # length 24
    """

    def __init__(self, input_data: TwoprcInputData):
        self.data = input_data
        self.variables = Expando()
        self.constraints = Expando()
        self.results = Expando()

        self.model = gp.Model(name="TwoPriceOffering")
        self._build_model()

    
    # Model construction

    def _build_variables(self):
        d = self.data

        # First-stage: hourly day-ahead offer
        self.variables.p_DA = self.model.addVars(
            d.T, lb=0.0, ub=d.P_nom, name="p_DA",
        )

        # Second-stage: total imbalance (free) and its non-negative split.
        # delta_up and delta_dn are upper-bounded at P_nom: physically
        # |delta| <= P_nom since both p_real and p_DA lie in [0, P_nom].
        # This bound also ensures the LP is bounded in hours with negative
        # DA prices (see module docstring).
        self.variables.delta = self.model.addVars(
            d.T, d.W, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="delta",
        )
        self.variables.delta_up = self.model.addVars(
            d.T, d.W, lb=0.0, ub=d.P_nom, name="delta_up",
        )
        self.variables.delta_dn = self.model.addVars(
            d.T, d.W, lb=0.0, ub=d.P_nom, name="delta_dn",
        )

    def _build_constraints(self):
        d = self.data
        v = self.variables

        # delta[t,w] = p_real[t,w] - p_DA[t]
        self.constraints.delta_def = self.model.addConstrs(
            (v.delta[t, w] == d.p_real[w, t] - v.p_DA[t]
             for t in d.T for w in d.W),
            name="delta_def",
        )

        # delta[t,w] = delta_up[t,w] - delta_dn[t,w]
        self.constraints.delta_split = self.model.addConstrs(
            (v.delta[t, w] == v.delta_up[t, w] - v.delta_dn[t, w]
             for t in d.T for w in d.W),
            name="delta_split",
        )

    def _build_objective(self):
        d = self.data
        v = self.variables

        # DA revenue (depends only on first-stage decision and DA price)
        da_revenue = gp.quicksum(
            d.pi[w] * d.lambda_DA[w, t] * v.p_DA[t]
            for t in d.T for w in d.W
        )

        # Balancing settlement: revenue from excess minus cost of shortfall
        bal_revenue = gp.quicksum(
            d.pi[w] * (
                d.psi_up[w, t] * v.delta_up[t, w]
                - d.psi_dn[w, t] * v.delta_dn[t, w]
            )
            for t in d.T for w in d.W
        )

        self.model.setObjective(da_revenue + bal_revenue, GRB.MAXIMIZE)

    def _build_model(self):
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self.model.update()


    # Solve and extract results

    def _save_results(self):
        d = self.data
        v = self.variables
        r = self.results

        r.objective_value = self.model.ObjVal

        # Hourly DA offer (length 24)
        r.p_DA = np.array([v.p_DA[t].X for t in d.T])

        # Per-hour, per-scenario imbalance variables (24 x n_scen)
        n_t, n_w = len(d.T), len(d.W)
        r.delta    = np.zeros((n_t, n_w))
        r.delta_up = np.zeros((n_t, n_w))
        r.delta_dn = np.zeros((n_t, n_w))
        for t in d.T:
            for w in d.W:
                r.delta[t, w]    = v.delta[t, w].X
                r.delta_up[t, w] = v.delta_up[t, w].X
                r.delta_dn[t, w] = v.delta_dn[t, w].X

        # Per-scenario realized profit (sum over 24 hours):
        # profit_w = sum_t [ lambda_DA[t,w] * p_DA[t]
        #                    + psi_up[t,w] * delta_up[t,w]
        #                    - psi_dn[t,w] * delta_dn[t,w] ]
        da_part = (d.lambda_DA * r.p_DA[np.newaxis, :]).sum(axis=1)             # (n_w,)
        up_part = (d.psi_up    * r.delta_up.T).sum(axis=1)                      # (n_w,)
        dn_part = (d.psi_dn    * r.delta_dn.T).sum(axis=1)                      # (n_w,)
        r.profit_per_scenario = da_part + up_part - dn_part                     # (n_w,)
        r.expected_profit = float((d.pi * r.profit_per_scenario).sum())

        # Diagnostic: count cells where the LP picked one of multiple
        # equivalent optima (delta_up and delta_dn both > 0). This is
        # harmless when psi_up == psi_dn (negative-price hours under our
        # override), because the objective contribution depends only on
        # the difference (delta_up - delta_dn), not on each individually.
        # We track it for completeness; in well-posed cells (psi_up < psi_dn)
        # the violation count must be zero.
        both_positive = (r.delta_up > 1e-6) & (r.delta_dn > 1e-6)
        psi_up_TW = d.psi_up.T  # (n_t, n_w)
        psi_dn_TW = d.psi_dn.T  # (n_t, n_w)
        well_posed_violations = both_positive & (psi_dn_TW > psi_up_TW + 1e-9)
        r.complementarity_violations = int(well_posed_violations.sum())
        r.harmless_degenerate_cells = int(both_positive.sum() - well_posed_violations.sum())

        if r.complementarity_violations > 0:
            print(
                f"Warning: {r.complementarity_violations} (t,w) cells have "
                f"both delta_up and delta_dn strictly positive in well-posed "
                f"hours (psi_dn > psi_up). This indicates a model bug."
            )

    def run(self, verbose: bool = False):
        self.model.Params.OutputFlag = 1 if verbose else 0
        self.model.optimize()

        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Solver did not return an optimal solution. Status: {self.model.Status}"
            )

        self._save_results()
        return self.results