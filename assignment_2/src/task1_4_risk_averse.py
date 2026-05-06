"""
Risk-averse offering strategies for the wind farm under both ONE-PRICE
and TWO-PRICE balancing schemes. Implements Task 1.4 of Assignment 2.

The risk-averse extension adds a Conditional Value-at-Risk (CVaR) term
to the objective of Tasks 1.1 / 1.2, following the classical
Rockafellar-Uryasev linearization. The new objective is

    max  (1 - beta) * E[profit]  +  beta * CVaR_alpha(profit)

with CVaR_alpha represented through the auxiliary variables zeta (scalar)
and eta_w >= 0, plus the constraint  zeta - profit_w <= eta_w  for each
scenario w. The CVaR_alpha value at the optimum is

    zeta - (1 / (1 - alpha)) * sum_w pi_w * eta_w.

beta = 0 recovers the risk-neutral models of Tasks 1.1 / 1.2.
beta -> 1 yields the ultra-conservative strategy that maximizes CVaR alone.
Intermediate values trace out the Pareto frontier (E[profit], CVaR).

The two classes below differ only in the per-scenario balancing
coefficients (one-price uses a single lambda_B per scenario; two-price
uses asymmetric psi_up and psi_dn with the negative-price override
inherited from task1_2_offering_twoprice.py).
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from scenario_generation import ScenarioSet



# Generic Expando container (same idiom as the other tasks)

class Expando:
    """Generic data container; attributes can be added on the fly."""
    pass



# ONE-PRICE risk-averse model

class RiskAverseOneprcInputData:
    """
    Input data for the risk-averse one-price model.

    Builds the per-scenario balancing price lambda_B from the day-ahead
    price and the binary system imbalance, identically to Task 1.1.
    """

    def __init__(self, scenarios: ScenarioSet, alpha: float = 0.90, beta: float = 0.0):
        self.S = scenarios

        self.T = list(range(scenarios.n_hours))
        self.W = list(range(scenarios.n_scenarios))
        self.P_nom = scenarios.P_nom

        self.pi = scenarios.probabilities
        self.p_real = scenarios.wind
        self.lambda_DA = scenarios.price_da

        coeff = np.where(scenarios.imbalance == 1, 1.25, 0.85)
        self.lambda_B = coeff * scenarios.price_da

        self.alpha = alpha
        self.beta  = beta


class RiskAverseOfferingOnePrice:
    """
    Risk-averse stochastic LP for the one-price scheme, with CVaR.
    """

    def __init__(self, input_data: RiskAverseOneprcInputData):
        self.data = input_data
        self.variables = Expando()
        self.constraints = Expando()
        self.results = Expando()

        self.model = gp.Model(name="OnePriceRiskAverse")
        self._build_model()

    def _build_variables(self):
        d = self.data

        self.variables.p_DA = self.model.addVars(
            d.T, lb=0.0, ub=d.P_nom, name="p_DA",
        )
        self.variables.delta = self.model.addVars(
            d.T, d.W, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="delta",
        )
        self.variables.delta_up = self.model.addVars(
            d.T, d.W, lb=0.0, name="delta_up",
        )
        self.variables.delta_dn = self.model.addVars(
            d.T, d.W, lb=0.0, name="delta_dn",
        )

        # CVaR auxiliary variables (Rockafellar-Uryasev formulation)
        self.variables.zeta = self.model.addVar(
            lb=-GRB.INFINITY, ub=GRB.INFINITY, name="zeta",
        )
        self.variables.eta = self.model.addVars(
            d.W, lb=0.0, name="eta",
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

        # CVaR constraint: zeta - profit[w] <= eta[w]   for all w
        # i.e. -profit[w] + zeta - eta[w] <= 0
        # where profit[w] = sum_t [ lambda_DA[t,w] * p_DA[t]
        #                            + lambda_B[t,w] * (delta_up - delta_dn) ]
        self.constraints.cvar = self.model.addConstrs(
            (
                -gp.quicksum(
                    d.lambda_DA[w, t] * v.p_DA[t]
                    + d.lambda_B[w, t] * (v.delta_up[t, w] - v.delta_dn[t, w])
                    for t in d.T
                )
                + v.zeta - v.eta[w] <= 0
                for w in d.W
            ),
            name="cvar",
        )

    def _build_objective(self):
        d = self.data
        v = self.variables

        # Expected profit term
        expected_profit = gp.quicksum(
            d.pi[w] * (
                d.lambda_DA[w, t] * v.p_DA[t]
                + d.lambda_B[w, t] * (v.delta_up[t, w] - v.delta_dn[t, w])
            )
            for t in d.T for w in d.W
        )

        # CVaR term: zeta - (1 / (1-alpha)) * sum_w pi_w * eta_w
        cvar_term = v.zeta - (1.0 / (1.0 - d.alpha)) * gp.quicksum(
            d.pi[w] * v.eta[w] for w in d.W
        )

        # Convex combination
        self.model.setObjective(
            (1.0 - d.beta) * expected_profit + d.beta * cvar_term,
            GRB.MAXIMIZE,
        )

    def _build_model(self):
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self.model.update()

    def _save_results(self):
        d = self.data
        v = self.variables
        r = self.results

        r.objective_value = self.model.ObjVal
        r.p_DA = np.array([v.p_DA[t].X for t in d.T])
        r.zeta = v.zeta.X
        r.eta  = np.array([v.eta[w].X for w in d.W])

        # Per-scenario realized profit (re-computed for clarity)
        n_t, n_w = len(d.T), len(d.W)
        delta_up = np.zeros((n_t, n_w))
        delta_dn = np.zeros((n_t, n_w))
        for t in d.T:
            for w in d.W:
                delta_up[t, w] = v.delta_up[t, w].X
                delta_dn[t, w] = v.delta_dn[t, w].X

        da_part  = (d.lambda_DA * r.p_DA[np.newaxis, :]).sum(axis=1)
        bal_part = (d.lambda_B * (delta_up.T - delta_dn.T)).sum(axis=1)
        r.profit_per_scenario = da_part + bal_part
        r.expected_profit = float((d.pi * r.profit_per_scenario).sum())

        # CVaR_alpha computed from optimal zeta and eta
        r.cvar = float(r.zeta - (1.0 / (1.0 - d.alpha)) * (d.pi * r.eta).sum())

    def run(self, verbose: bool = False):
        self.model.Params.OutputFlag = 1 if verbose else 0
        self.model.optimize()

        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Solver did not return an optimal solution. Status: {self.model.Status}"
            )

        self._save_results()
        return self.results


# TWO-PRICE risk-averse model

class RiskAverseTwoprcInputData:
    """
    Input data for the risk-averse two-price model.

    Builds psi_up and psi_dn identically to Task 1.2, including the
    negative-price override that collapses the scheme to a uniform
    settlement when lambda_DA < 0.
    """

    def __init__(self, scenarios: ScenarioSet, alpha: float = 0.90, beta: float = 0.0):
        self.S = scenarios

        self.T = list(range(scenarios.n_hours))
        self.W = list(range(scenarios.n_scenarios))
        self.P_nom = scenarios.P_nom

        self.pi = scenarios.probabilities
        self.p_real = scenarios.wind
        self.lambda_DA = scenarios.price_da

        # Standard two-price coefficients
        coeff_up = np.where(scenarios.imbalance == 1, 1.00, 0.85)
        coeff_dn = np.where(scenarios.imbalance == 1, 1.25, 1.00)

        # Negative-price override: collapse to uniform settlement
        coeff_uniform = np.where(scenarios.imbalance == 1, 1.25, 0.85)
        neg_mask = scenarios.price_da < 0
        coeff_up = np.where(neg_mask, coeff_uniform, coeff_up)
        coeff_dn = np.where(neg_mask, coeff_uniform, coeff_dn)

        self.psi_up = coeff_up * scenarios.price_da
        self.psi_dn = coeff_dn * scenarios.price_da

        self.alpha = alpha
        self.beta  = beta


class RiskAverseOfferingTwoPrice:
    """
    Risk-averse stochastic LP for the two-price scheme, with CVaR.
    """

    def __init__(self, input_data: RiskAverseTwoprcInputData):
        self.data = input_data
        self.variables = Expando()
        self.constraints = Expando()
        self.results = Expando()

        self.model = gp.Model(name="TwoPriceRiskAverse")
        self._build_model()

    def _build_variables(self):
        d = self.data

        self.variables.p_DA = self.model.addVars(
            d.T, lb=0.0, ub=d.P_nom, name="p_DA",
        )
        self.variables.delta = self.model.addVars(
            d.T, d.W, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="delta",
        )
        self.variables.delta_up = self.model.addVars(
            d.T, d.W, lb=0.0, ub=d.P_nom, name="delta_up",
        )
        self.variables.delta_dn = self.model.addVars(
            d.T, d.W, lb=0.0, ub=d.P_nom, name="delta_dn",
        )

        self.variables.zeta = self.model.addVar(
            lb=-GRB.INFINITY, ub=GRB.INFINITY, name="zeta",
        )
        self.variables.eta = self.model.addVars(
            d.W, lb=0.0, name="eta",
        )

    def _build_constraints(self):
        d = self.data
        v = self.variables

        self.constraints.delta_def = self.model.addConstrs(
            (v.delta[t, w] == d.p_real[w, t] - v.p_DA[t]
             for t in d.T for w in d.W),
            name="delta_def",
        )

        self.constraints.delta_split = self.model.addConstrs(
            (v.delta[t, w] == v.delta_up[t, w] - v.delta_dn[t, w]
             for t in d.T for w in d.W),
            name="delta_split",
        )

        # CVaR constraint:
        # profit[w] = sum_t [ lambda_DA[t,w]*p_DA[t]
        #                     + psi_up[t,w]*delta_up[t,w]
        #                     - psi_dn[t,w]*delta_dn[t,w] ]
        self.constraints.cvar = self.model.addConstrs(
            (
                -gp.quicksum(
                    d.lambda_DA[w, t] * v.p_DA[t]
                    + d.psi_up[w, t] * v.delta_up[t, w]
                    - d.psi_dn[w, t] * v.delta_dn[t, w]
                    for t in d.T
                )
                + v.zeta - v.eta[w] <= 0
                for w in d.W
            ),
            name="cvar",
        )

    def _build_objective(self):
        d = self.data
        v = self.variables

        expected_profit = gp.quicksum(
            d.pi[w] * (
                d.lambda_DA[w, t] * v.p_DA[t]
                + d.psi_up[w, t] * v.delta_up[t, w]
                - d.psi_dn[w, t] * v.delta_dn[t, w]
            )
            for t in d.T for w in d.W
        )

        cvar_term = v.zeta - (1.0 / (1.0 - d.alpha)) * gp.quicksum(
            d.pi[w] * v.eta[w] for w in d.W
        )

        self.model.setObjective(
            (1.0 - d.beta) * expected_profit + d.beta * cvar_term,
            GRB.MAXIMIZE,
        )

    def _build_model(self):
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self.model.update()

    def _save_results(self):
        d = self.data
        v = self.variables
        r = self.results

        r.objective_value = self.model.ObjVal
        r.p_DA = np.array([v.p_DA[t].X for t in d.T])
        r.zeta = v.zeta.X
        r.eta  = np.array([v.eta[w].X for w in d.W])

        n_t, n_w = len(d.T), len(d.W)
        delta_up = np.zeros((n_t, n_w))
        delta_dn = np.zeros((n_t, n_w))
        for t in d.T:
            for w in d.W:
                delta_up[t, w] = v.delta_up[t, w].X
                delta_dn[t, w] = v.delta_dn[t, w].X

        da_part = (d.lambda_DA * r.p_DA[np.newaxis, :]).sum(axis=1)
        up_part = (d.psi_up    * delta_up.T).sum(axis=1)
        dn_part = (d.psi_dn    * delta_dn.T).sum(axis=1)
        r.profit_per_scenario = da_part + up_part - dn_part
        r.expected_profit = float((d.pi * r.profit_per_scenario).sum())

        r.cvar = float(r.zeta - (1.0 / (1.0 - d.alpha)) * (d.pi * r.eta).sum())

    def run(self, verbose: bool = False):
        self.model.Params.OutputFlag = 1 if verbose else 0
        self.model.optimize()

        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Solver did not return an optimal solution. Status: {self.model.Status}"
            )

        self._save_results()
        return self.results



# Pareto frontier sweep

def sweep_beta(scenarios: ScenarioSet,
               scheme: str,
               betas: np.ndarray,
               alpha: float = 0.90) -> dict:
    """
    Solve the risk-averse model for a sequence of beta values and return
    the (expected_profit, CVaR, p_DA) triplets needed to plot the
    Pareto frontier.

    Parameters
    ----------
    scenarios : ScenarioSet
        The 1600-scenario set.
    scheme : {'one-price', 'two-price'}
        Which model variant to use.
    betas : np.ndarray
        1-D array of beta values to sweep, e.g. np.linspace(0, 1, 11).
    alpha : float
        CVaR confidence level (default 0.90).

    Returns
    -------
    dict with keys 'betas', 'expected_profits', 'cvars', 'p_DA_per_beta',
    'profits_per_scenario_per_beta'.
    """
    if scheme == "one-price":
        InputDataClass = RiskAverseOneprcInputData
        ModelClass     = RiskAverseOfferingOnePrice
    elif scheme == "two-price":
        InputDataClass = RiskAverseTwoprcInputData
        ModelClass     = RiskAverseOfferingTwoPrice
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    n_beta = len(betas)
    expected_profits = np.zeros(n_beta)
    cvars = np.zeros(n_beta)
    p_DA_per_beta = np.zeros((n_beta, scenarios.n_hours))
    profits_per_scenario_per_beta = np.zeros((n_beta, scenarios.n_scenarios))

    for i, beta in enumerate(betas):
        data = InputDataClass(scenarios, alpha=alpha, beta=float(beta))
        res = ModelClass(data).run()
        expected_profits[i] = res.expected_profit
        cvars[i] = res.cvar
        p_DA_per_beta[i] = res.p_DA
        profits_per_scenario_per_beta[i] = res.profit_per_scenario

    return {
        "betas": np.asarray(betas, dtype=float),
        "expected_profits": expected_profits,
        "cvars": cvars,
        "p_DA_per_beta": p_DA_per_beta,
        "profits_per_scenario_per_beta": profits_per_scenario_per_beta,
    }