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
        
# ========================
# STEP 2: 24-HOUR DISPATCH
# ========================

class Step2InputData:
    """
    Copper-plate market clearing over 24 hours with:
    - conventional generators (marginal cost bids)
    - wind farms (zero marginal cost, limited by hourly availability)
    - elastic demand at buses (hourly bid price/quantity)
    - energy storage (pch/pdis/e with dynamics, zero bids)
    """

    def __init__(
        self,
        GENERATORS: list,
        WINDS: list,
        LOAD_BUSES: list,
        generator_cost: dict,
        generator_pmax: dict,
        wind_avail_t: dict,
        demand_pmax_t: dict,
        demand_bid_t: dict,
        hours: int
    ):
        self.GENERATORS = GENERATORS
        self.WINDS = WINDS
        self.LOAD_BUSES = LOAD_BUSES

        self.generator_cost = generator_cost
        self.generator_pmax = generator_pmax

        self.wind_avail_t = wind_avail_t

        self.demand_pmax_t = demand_pmax_t
        self.demand_bid_t = demand_bid_t
        self.hours = hours
        
        # Storage
        self.P_ch = 400  # We consider that we cover approximately 12% of the demand with storage during peak load hours.
        self.P_dis = 400
        self.E = 1600
        # As stated in ESR report, we consider a round-trip efficiency close to 0.8 which is a reasonable assumption for Pump Hydro Storage.
        self.eta_ch = 0.9
        self.eta_dis = 0.93

class Step2MarketClearing:
    """
    Copper-plate market clearing over 24 hours with storage:
    max sum_t [sum_n bid_tn * d_tn - sum_g c_g * pG_tg]
    s.t. balance_t: sum_g pG_tg + sum_w pW_tw + pch_t - pdis_t = sum_n d_tn  ∀t
         storage dyn: e_t = e_{t-1} + η_ch pch_t - pdis_t / η_dis  ∀t
         bounds ∀t,g,w,n
    """      
    def __init__(self, input_data: Step2InputData, output_flag: int = 0):
        self.data = input_data
        self.variables = Expando()
        self.constraints = Expando()
        self.results = Expando()
        self._build_model(output_flag=output_flag)

    def _build_variables(self):
        # Time index
        T = self.data.hours
        
        # Generator dispatch
        self.variables.pG = {(t, g): self.model.addVar(lb=0, ub=self.data.generator_pmax[g],
                                                       name=f'pG_t{t}_g{g}')
                             for t in T for g in self.data.GENERATORS}
        
        # Wind dispatch (bounded by availability)
        self.variables.pW = {
            (t, w): self.model.addVar(lb=0, ub=self.data.wind_avail_t[t][w],
                                      name=f'pW_t{t}_w{w}')
            for t in T for w in self.data.WINDS
        }

        # Elastic demand served
        self.variables.d = {
            (t, n): self.model.addVar(lb=0, ub=self.data.demand_pmax_t[t][n],
                                      name=f'Demand served at bus {n} in hour {t}')
            for t in T for n in self.data.LOAD_BUSES
        }
        
        # Storage variables
        self.variables.pch = {t: self.model.addVar(lb=0, ub=self.data.P_ch, name=f'pch_t{t}')
                              for t in T}
        self.variables.pdis = {t: self.model.addVar(lb=0, ub=self.data.P_dis, name=f'pdis_t{t}')
                               for t in T}
        self.variables.e = {t: self.model.addVar(lb=0, ub=self.data.E, name=f'e_t{t}')
                            for t in T}
        
    def _build_constraints(self):
        T = self.data.hours
        
        # System balance (copper plate)
        # Note: in welfare maximization, the economic "market price" is typically -Pi of this constraint.
        
        self.constraints.balance = {}
        for t in T:
            
            self.constraints.balance[t] = self.model.addLConstr(
                gp.quicksum(self.variables.pG[(t, g)] for g in self.data.GENERATORS) +
                gp.quicksum(self.variables.pW[(t, w)] for w in self.data.WINDS) -
                self.variables.pch[t] + self.variables.pdis[t] -
                gp.quicksum(self.variables.d[(t, n)] for n in self.data.LOAD_BUSES) == 0,
                name=f'Balance_t{t}'
            )
        
        # Storage dynamics (e_0 assume 0)
        t0 = T[0]
        self.constraints.dyn = {}
        self.constraints.dyn[t0] = self.model.addLConstr(
            self.variables.e[t0] == (self.data.eta_ch * self.variables.pch[t0]) -
            (self.variables.pdis[t0] / self.data.eta_dis),
            name='e_t0'
        )
        for t in T[1:]:
            self.constraints.dyn[t] = self.model.addLConstr(
                self.variables.e[t] == self.variables.e[t-1] +
                self.data.eta_ch * self.variables.pch[t] -
                self.variables.pdis[t] / self.data.eta_dis,
                name=f'dyn_t{t}'
            )

    def _build_objective_function(self):
        T = self.data.hours
        welfare = (
            gp.quicksum(self.data.demand_bid_t[t][n] * self.variables.d[(t, n)]
                        for t in T for n in self.data.LOAD_BUSES) -
            gp.quicksum(self.data.generator_cost[g] * self.variables.pG[(t, g)]
                        for t in T for g in self.data.GENERATORS)
            # wind/storage cost=0 omitted
        )
        self.model.setObjective(welfare, GRB.MAXIMIZE)

    def _build_model(self, output_flag: int = 0):
        self.model = gp.Model(name='Step 2 - Copper plate 24h with storage')
        self.model.setParam('OutputFlag', output_flag)
        self._build_variables()
        self.model.update()
        self._build_constraints()
        self._build_objective_function()
        self.model.update()

    def _save_results(self):
        T = self.data.hours
        self.results.objective_value = self.model.ObjVal  # Total welfare 24h

        # Dispatch por hora
        self.results.pG = {(t, g): self.variables.pG[(t, g)].X for t in T for g in self.data.GENERATORS}
        self.results.pW = {(t, w): self.variables.pW[(t, w)].X for t in T for w in self.data.WINDS}
        self.results.d = {(t, n): self.variables.d[(t, n)].X for t in T for n in self.data.LOAD_BUSES}
        self.results.pch = {t: self.variables.pch[t].X for t in T}
        self.results.pdis = {t: self.variables.pdis[t].X for t in T}
        self.results.e = {t: self.variables.e[t].X for t in T}

        # Hourly prices: -Pi balance_t
        self.results.market_prices_t = {t: -self.constraints.balance[t].Pi for t in T}
        self.results.balance_duals = {t: self.constraints.balance[t].Pi for t in T}

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            raise RuntimeError(f"Optimization of {self.model.ModelName} was not successful")
  