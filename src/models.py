import gurobipy as gp
from gurobipy import GRB

class Expando(object):
    """A class which can have attributes set"""
    pass


class Step1InputData:
    """
    Copper-plate (single-node) market clearing with:
    - conventional generators (marginal cost bids)
    - wind farms (zero marginal cost, limited by availability)
    - elastic demand at buses (bid price, limited by Dmax)
    """

    def __init__(
        self,
        GENERATORS: list,
        WINDS: list,
        LOAD_BUSES: list,
        generator_cost: dict,
        generator_pmax: dict,
        wind_avail: dict,
        demand_pmax: dict,
        demand_bid: dict,
    ):
        self.GENERATORS = GENERATORS
        self.WINDS = WINDS
        self.LOAD_BUSES = LOAD_BUSES

        self.generator_cost = generator_cost
        self.generator_pmax = generator_pmax

        self.wind_avail = wind_avail

        self.demand_pmax = demand_pmax
        self.demand_bid = demand_bid


class Step1MarketClearing:
    """
    Maximize social welfare:
        max  sum_n bid_n * d_n  -  sum_g c_g * p_g  - 0 * w
    s.t.  0 <= p_g <= Pmax_g
          0 <= w_k <= Wavail_k
          0 <= d_n <= Dmax_n
          balance: sum_g p_g + sum_k w_k - sum_n d_n = 0
    """

    def __init__(self, input_data: Step1InputData, output_flag: int = 0):
        self.data = input_data
        self.variables = Expando()
        self.constraints = Expando()
        self.results = Expando()
        self._build_model(output_flag=output_flag)

    def _build_variables(self):
        # Generator dispatch
        self.variables.pG = {
            g: self.model.addVar(lb=0, ub=self.data.generator_pmax[g],
                                 name=f'Gen production {g}')
            for g in self.data.GENERATORS
        }

        # Wind dispatch (bounded by availability)
        self.variables.pW = {
            w: self.model.addVar(lb=0, ub=self.data.wind_avail[w],
                                 name=f'Wind production {w}')
            for w in self.data.WINDS
        }

        # Elastic demand served
        self.variables.d = {
            n: self.model.addVar(lb=0, ub=self.data.demand_pmax[n],
                                 name=f'Demand served at bus {n}')
            for n in self.data.LOAD_BUSES
        }

    def _build_constraints(self):
        # System balance (copper plate)
        # Note: in welfare maximization, the economic "market price" is typically -Pi of this constraint.
        self.constraints.balance = self.model.addLConstr(
            gp.quicksum(self.variables.pG[g] for g in self.data.GENERATORS)
            + gp.quicksum(self.variables.pW[w] for w in self.data.WINDS)
            - gp.quicksum(self.variables.d[n] for n in self.data.LOAD_BUSES),
            GRB.EQUAL,
            0,
            name='Balance constraint'
        )

    def _build_objective_function(self):
        welfare = (
            gp.quicksum(self.data.demand_bid[n] * self.variables.d[n] for n in self.data.LOAD_BUSES)
            - gp.quicksum(self.data.generator_cost[g] * self.variables.pG[g] for g in self.data.GENERATORS)
            # wind marginal cost = 0, so omitted
        )
        self.model.setObjective(welfare, GRB.MAXIMIZE)

    def _build_model(self, output_flag: int = 0):
        self.model = gp.Model(name='Step 1 - Copper plate market clearing')
        self.model.setParam('OutputFlag', output_flag)
        self._build_variables()
        self._build_constraints()
        self._build_objective_function()
        self.model.update()

    def _save_results(self):
        self.results.objective_value = self.model.ObjVal

        self.results.pG = {g: self.variables.pG[g].X for g in self.data.GENERATORS}
        self.results.pW = {w: self.variables.pW[w].X for w in self.data.WINDS}
        self.results.d = {n: self.variables.d[n].X for n in self.data.LOAD_BUSES}

        # Dual of balance constraint
        self.results.balance_dual = self.constraints.balance.Pi

        # Market price convention:
        # For welfare maximization with balance written as (supply - demand = 0),
        # market price is usually -Pi.
        self.results.market_price = -self.results.balance_dual

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            raise RuntimeError(f"Optimization of {self.model.ModelName} was not successful")