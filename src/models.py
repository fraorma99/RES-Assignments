import gurobipy as gp
from gurobipy import GRB
from typing import List

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
        hours: List[int],
        P_ch: float = 400.0,
        P_dis: float = 400.0,
        E: float = 1600.0,
        eta_ch: float = 0.9,
        eta_dis: float = 0.93,
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
        self.P_ch = P_ch
        self.P_dis = P_dis
        self.E = E
        self.eta_ch = eta_ch
        self.eta_dis = eta_dis

class Step2MarketClearing:
    """
    Copper-plate market clearing over 24 hours with storage:
    max sum_t [sum_n bid_tn * d_tn - sum_g c_g * pG_tg]
    s.t. balance_t: sum_g pG_tg + sum_w pW_tw - pch_t + pdis_t - sum_n d_tn = 0  ∀t
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
            
        # Final SOC constraint: e_T = 0 (cyclic, since e_0 = 0)
        self.constraints.final_soc = self.model.addLConstr(
            self.variables.e[T[-1]] == 0,
            name='final_soc'
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
  
# ========================
# STEP 3A: 1-HOUR NODAL MARKET CLEARING
# ========================

class Step3InputData:
    """
    Network-constrained (nodal) 1-hour market clearing with:
    - conventional generators (marginal cost bids)
    - wind farms (zero marginal cost, limited by availability)
    - elastic demand at buses (bid price, limited by Dmax)
    - DC power flow with line limits
    """

    def __init__(
        self,
        BUSES: list,
        GENERATORS: list,
        WINDS: list,
        LOAD_BUSES: list,
        LINES: list,
        slack_bus: int,
        generator_cost: dict,
        generator_pmax: dict,
        generator_bus: dict,
        wind_avail: dict,
        wind_bus: dict,
        demand_pmax: dict,
        demand_bid: dict,
        line_from: dict,
        line_to: dict,
        line_B: dict,
        line_fmax: dict,
    ):
        self.BUSES = BUSES
        self.GENERATORS = GENERATORS
        self.WINDS = WINDS
        self.LOAD_BUSES = LOAD_BUSES
        self.LINES = LINES
        self.slack_bus = slack_bus

        self.generator_cost = generator_cost
        self.generator_pmax = generator_pmax
        self.generator_bus = generator_bus

        self.wind_avail = wind_avail
        self.wind_bus = wind_bus

        self.demand_pmax = demand_pmax
        self.demand_bid = demand_bid

        self.line_from = line_from
        self.line_to = line_to
        self.line_B = line_B
        self.line_fmax = line_fmax


class Step3MarketClearing:
    """
    Maximize social welfare with nodal balances and DC power flow:
        max  sum_n bid_n * d_n  -  sum_g c_g * p_g

    s.t.
        0 <= p_g <= Pmax_g
        0 <= w_k <= Wavail_k
        0 <= d_n <= Dmax_n

        f_l = B_l * (theta_i - theta_j)
        -Fmax_l <= f_l <= Fmax_l

        For each bus n:
            generation at n + wind at n + inflow - outflow - demand at n = 0

        theta_slack = 0
    """

    def __init__(self, input_data: Step3InputData, output_flag: int = 0):
        self.data = input_data
        self.variables = Expando()
        self.constraints = Expando()
        self.results = Expando()
        self._build_model(output_flag=output_flag)

    def _build_variables(self):
        self.variables.pG = {
            g: self.model.addVar(
                lb=0,
                ub=self.data.generator_pmax[g],
                name=f'Gen production {g}'
            )
            for g in self.data.GENERATORS
        }

        self.variables.pW = {
            w: self.model.addVar(
                lb=0,
                ub=self.data.wind_avail[w],
                name=f'Wind production {w}'
            )
            for w in self.data.WINDS
        }

        self.variables.d = {
            n: self.model.addVar(
                lb=0,
                ub=self.data.demand_pmax[n],
                name=f'Demand served at bus {n}'
            )
            for n in self.data.LOAD_BUSES
        }

        self.variables.theta = {
            n: self.model.addVar(
                lb=-GRB.INFINITY,
                ub=GRB.INFINITY,
                name=f'Theta at bus {n}'
            )
            for n in self.data.BUSES
        }

        self.variables.f = {
            l: self.model.addVar(
                lb=-GRB.INFINITY,
                ub=GRB.INFINITY,
                name=f'Flow on line {l}'
            )
            for l in self.data.LINES
        }

    def _build_constraints(self):
        self.constraints.flow_def = {}
        self.constraints.line_up = {}
        self.constraints.line_dn = {}
        self.constraints.balance = {}

        # DC flow equations
        for l in self.data.LINES:
            i = self.data.line_from[l]
            j = self.data.line_to[l]
            B = self.data.line_B[l]

            self.constraints.flow_def[l] = self.model.addLConstr(
                self.variables.f[l] - B * (self.variables.theta[i] - self.variables.theta[j]),
                GRB.EQUAL,
                0,
                name=f'Flow definition line {l}'
            )

        # Line limits
        for l in self.data.LINES:
            fmax = self.data.line_fmax[l]

            self.constraints.line_up[l] = self.model.addLConstr(
                self.variables.f[l],
                GRB.LESS_EQUAL,
                fmax,
                name=f'Line upper limit {l}'
            )

            self.constraints.line_dn[l] = self.model.addLConstr(
                self.variables.f[l],
                GRB.GREATER_EQUAL,
                -fmax,
                name=f'Line lower limit {l}'
            )

        # Slack angle
        self.constraints.slack = self.model.addLConstr(
            self.variables.theta[self.data.slack_bus],
            GRB.EQUAL,
            0,
            name=f'Slack bus angle {self.data.slack_bus}'
        )

        # Nodal balances
        for n in self.data.BUSES:
            gen_at_n = gp.quicksum(
                self.variables.pG[g]
                for g in self.data.GENERATORS
                if self.data.generator_bus[g] == n
            )

            wind_at_n = gp.quicksum(
                self.variables.pW[w]
                for w in self.data.WINDS
                if self.data.wind_bus[w] == n
            )

            demand_at_n = self.variables.d[n] if n in self.data.LOAD_BUSES else 0

            inflow = gp.quicksum(
                self.variables.f[l]
                for l in self.data.LINES
                if self.data.line_to[l] == n
            )

            outflow = gp.quicksum(
                self.variables.f[l]
                for l in self.data.LINES
                if self.data.line_from[l] == n
            )

            self.constraints.balance[n] = self.model.addLConstr(
                gen_at_n + wind_at_n + inflow - outflow - demand_at_n,
                GRB.EQUAL,
                0,
                name=f'Nodal balance at bus {n}'
            )

    def _build_objective_function(self):
        welfare = (
            gp.quicksum(
                self.data.demand_bid[n] * self.variables.d[n]
                for n in self.data.LOAD_BUSES
            )
            - gp.quicksum(
                self.data.generator_cost[g] * self.variables.pG[g]
                for g in self.data.GENERATORS
            )
        )
        self.model.setObjective(welfare, GRB.MAXIMIZE)

    def _build_model(self, output_flag: int = 0):
        self.model = gp.Model(name='Step 3 - Nodal market clearing')
        self.model.setParam('OutputFlag', output_flag)
        self._build_variables()
        self.model.update()
        self._build_constraints()
        self._build_objective_function()
        self.model.update()

    def _save_results(self):
        self.results.objective_value = self.model.ObjVal

        self.results.pG = {g: self.variables.pG[g].X for g in self.data.GENERATORS}
        self.results.pW = {w: self.variables.pW[w].X for w in self.data.WINDS}
        self.results.d = {n: self.variables.d[n].X for n in self.data.LOAD_BUSES}
        self.results.theta = {n: self.variables.theta[n].X for n in self.data.BUSES}
        self.results.f = {l: self.variables.f[l].X for l in self.data.LINES}

        self.results.balance_duals = {
            n: self.constraints.balance[n].Pi for n in self.data.BUSES
        }

        # Keep same sign convention as Step 1 and Step 2
        self.results.nodal_prices = {
            n: -self.results.balance_duals[n] for n in self.data.BUSES
        }

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            raise RuntimeError(f"Optimization of {self.model.ModelName} was not successful")
        
# ========================
# STEP 3B: 1-HOUR ZONAL MARKET CLEARING
# ========================

class Step3ZonalInputData:
    """
    Zonal 1-hour market clearing with:
    - bus-level generation/wind/demand data
    - one uniform price per zone
    - ATC-limited transfers between zones
    """

    def __init__(
        self,
        ZONES: list,
        BUS_ZONE: dict,
        INTERFACES: list,
        atc: dict,
        GENERATORS: list,
        WINDS: list,
        LOAD_BUSES: list,
        generator_cost: dict,
        generator_pmax: dict,
        generator_bus: dict,
        wind_avail: dict,
        wind_bus: dict,
        demand_pmax: dict,
        demand_bid: dict,
    ):
        self.ZONES = ZONES
        self.BUS_ZONE = BUS_ZONE
        self.INTERFACES = INTERFACES
        self.atc = atc

        self.GENERATORS = GENERATORS
        self.WINDS = WINDS
        self.LOAD_BUSES = LOAD_BUSES

        self.generator_cost = generator_cost
        self.generator_pmax = generator_pmax
        self.generator_bus = generator_bus

        self.wind_avail = wind_avail
        self.wind_bus = wind_bus

        self.demand_pmax = demand_pmax
        self.demand_bid = demand_bid


class Step3ZonalMarketClearing:
    """
    Zonal welfare maximization with ATC between zones.
    Transfer T[z1,z2] is positive from z1 to z2.
    """

    def __init__(self, input_data: Step3ZonalInputData, output_flag: int = 0):
        self.data = input_data
        self.variables = Expando()
        self.constraints = Expando()
        self.results = Expando()
        self._build_model(output_flag=output_flag)

    def _build_variables(self):
        self.variables.pG = {
            g: self.model.addVar(lb=0, ub=self.data.generator_pmax[g], name=f'Gen production {g}')
            for g in self.data.GENERATORS
        }

        self.variables.pW = {
            w: self.model.addVar(lb=0, ub=self.data.wind_avail[w], name=f'Wind production {w}')
            for w in self.data.WINDS
        }

        self.variables.d = {
            n: self.model.addVar(lb=0, ub=self.data.demand_pmax[n], name=f'Demand served at bus {n}')
            for n in self.data.LOAD_BUSES
        }

        self.variables.T = {
            k: self.model.addVar(
                lb=-self.data.atc[k],
                ub=self.data.atc[k],
                name=f'Transfer_{k[0]}_{k[1]}'
            )
            for k in self.data.INTERFACES
        }

    def _build_constraints(self):
        self.constraints.zone_balance = {}

        for z in self.data.ZONES:
            gen_z = gp.quicksum(
                self.variables.pG[g]
                for g in self.data.GENERATORS
                if self.data.BUS_ZONE[self.data.generator_bus[g]] == z
            )

            wind_z = gp.quicksum(
                self.variables.pW[w]
                for w in self.data.WINDS
                if self.data.BUS_ZONE[self.data.wind_bus[w]] == z
            )

            dem_z = gp.quicksum(
                self.variables.d[n]
                for n in self.data.LOAD_BUSES
                if self.data.BUS_ZONE[n] == z
            )

            net_export = gp.quicksum(
                self.variables.T[k] if k[0] == z else -self.variables.T[k]
                for k in self.data.INTERFACES
                if z in k
            )

            self.constraints.zone_balance[z] = self.model.addLConstr(
                gen_z + wind_z - dem_z - net_export,
                GRB.EQUAL,
                0,
                name=f'Zone balance {z}'
            )

    def _build_objective_function(self):
        welfare = (
            gp.quicksum(self.data.demand_bid[n] * self.variables.d[n] for n in self.data.LOAD_BUSES)
            - gp.quicksum(self.data.generator_cost[g] * self.variables.pG[g] for g in self.data.GENERATORS)
        )
        self.model.setObjective(welfare, GRB.MAXIMIZE)

    def _build_model(self, output_flag: int = 0):
        self.model = gp.Model(name='Step 3 - Zonal market clearing')
        self.model.setParam('OutputFlag', output_flag)
        self._build_variables()
        self.model.update()
        self._build_constraints()
        self._build_objective_function()
        self.model.update()

    def _save_results(self):
        self.results.objective_value = self.model.ObjVal

        self.results.pG = {g: self.variables.pG[g].X for g in self.data.GENERATORS}
        self.results.pW = {w: self.variables.pW[w].X for w in self.data.WINDS}
        self.results.d = {n: self.variables.d[n].X for n in self.data.LOAD_BUSES}
        self.results.T = {k: self.variables.T[k].X for k in self.data.INTERFACES}

        self.results.zone_balance_duals = {
            z: self.constraints.zone_balance[z].Pi for z in self.data.ZONES
        }

        self.results.zonal_prices = {
            z: -self.results.zone_balance_duals[z] for z in self.data.ZONES
        }

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            raise RuntimeError(f"Optimization of {self.model.ModelName} was not successful")        