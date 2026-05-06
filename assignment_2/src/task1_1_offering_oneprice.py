"""
Stochastic offering strategy for a price-taking wind farm under the
ONE-PRICE balancing scheme. Implements Task 1.1 of Assignment 2.

Following the formulation of Lecture 8 and the OOP pattern of
ex8_solution.py / ex9_solution.py from class:

    max  sum_t sum_w  pi_w * [ lambda_DA[t,w] * p_DA[t]
                              + lambda_B[t,w] * (delta_up[t,w] - delta_dn[t,w]) ]

    s.t. 0 <= p_DA[t] <= P_nom                   for all t
         delta[t,w] = p_real[t,w] - p_DA[t]      for all t, w
         delta[t,w] = delta_up[t,w] - delta_dn[t,w]
         delta_up[t,w], delta_dn[t,w] >= 0

Under the one-price scheme the balancing price is the same regardless
of the sign of the wind farm's individual imbalance:

    lambda_B[t,w] = 1.25 * lambda_DA[t,w]   if system in deficit  (SI=1)
    lambda_B[t,w] = 0.85 * lambda_DA[t,w]   if system in surplus  (SI=0)
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from scenario_generation import ScenarioSet



# Generic Expando container (same idiom as class exercises and A1)

class Expando:
    """Generic data container; attributes can be added on the fly."""
    pass



# Input data wrapper

class OneprcInputData:
    """
    Pre-processed input data for the one-price stochastic offering model.

    Builds the per-scenario balancing price lambda_B from the day-ahead
    price and the binary system-imbalance signal of each scenario,
    according to the assignment's scaling rule:
        SI = 1 (deficit)  -> lambda_B = 1.25 * lambda_DA
        SI = 0 (surplus)  -> lambda_B = 0.85 * lambda_DA
    """

    def __init__(self, scenarios: ScenarioSet):
        self.S = scenarios

        self.T = list(range(scenarios.n_hours))               # 0..23
        self.W = list(range(scenarios.n_scenarios))           # 0..1599
        self.P_nom = scenarios.P_nom

        self.pi = scenarios.probabilities                     # (n_scen,)

        self.p_real = scenarios.wind                          # (n_scen, 24) MW
        self.lambda_DA = scenarios.price_da                   # (n_scen, 24) EUR/MWh

        # Per-scenario balancing price (one-price scheme)
        coeff = np.where(scenarios.imbalance == 1, 1.25, 0.85)
        self.lambda_B = coeff * scenarios.price_da            # (n_scen, 24)



# Optimization model

class StochasticOfferingOnePrice:
    """
    Stochastic LP for the wind farm's day-ahead offering strategy under
    the one-price balancing scheme.

    Usage
    -----
    >>> data  = OneprcInputData(scenario_set)
    >>> model = StochasticOfferingOnePrice(data)
    >>> model.run()
    >>> print(model.results.objective_value)
    >>> print(model.results.p_DA)        # length 24
    """

    def __init__(self, input_data: OneprcInputData):
        self.data = input_data
        self.variables = Expando()
        self.constraints = Expando()
        self.results = Expando()

        self.model = gp.Model(name="OnePriceOffering")
        self._build_model()

    
    # Model construction

    def _build_variables(self):
        d = self.data

        # First-stage: hourly day-ahead offer
        self.variables.p_DA = self.model.addVars(
            d.T,
            lb=0.0,
            ub=d.P_nom,
            name="p_DA",
        )

        # Second-stage: total imbalance (free), and its non-negative parts
        self.variables.delta = self.model.addVars(
            d.T, d.W,
            lb=-GRB.INFINITY, ub=GRB.INFINITY,
            name="delta",
        )
        self.variables.delta_up = self.model.addVars(
            d.T, d.W,
            lb=0.0,
            name="delta_up",
        )
        self.variables.delta_dn = self.model.addVars(
            d.T, d.W,
            lb=0.0,
            name="delta_dn",
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

        # Expected DA revenue: sum_t sum_w pi_w * lambda_DA[t,w] * p_DA[t]
        # Note: p_DA[t] does not depend on w, but the price does.
        da_revenue = gp.quicksum(
            d.pi[w] * d.lambda_DA[w, t] * v.p_DA[t]
            for t in d.T for w in d.W
        )

        # Expected balancing revenue (signed: up sold, dn bought back)
        bal_revenue = gp.quicksum(
            d.pi[w] * d.lambda_B[w, t] * (v.delta_up[t, w] - v.delta_dn[t, w])
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

        # Per-scenario realized profit (sum over 24 hours)
        # Useful for plotting the profit distribution (Task 1.1 deliverable)
        # and for reuse in CVaR analysis (Task 1.4).
        # profit_w = sum_t [ lambda_DA[t,w] * p_DA[t]
        #                    + lambda_B[t,w]  * (delta_up[t,w] - delta_dn[t,w]) ]
        da_part = (d.lambda_DA * r.p_DA[np.newaxis, :]).sum(axis=1)         # (n_w,)
        bal_part = (d.lambda_B * (r.delta_up.T - r.delta_dn.T)).sum(axis=1) # (n_w,)
        r.profit_per_scenario = da_part + bal_part                          # (n_w,)
        r.expected_profit = float((d.pi * r.profit_per_scenario).sum())

    def run(self, verbose: bool = False):
        self.model.Params.OutputFlag = 1 if verbose else 0
        self.model.optimize()

        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Solver did not return an optimal solution. Status: {self.model.Status}"
            )

        self._save_results()
        return self.results