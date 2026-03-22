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
# ========================
# STEP 5: BALANCING MARKET
# ========================
# Append this block to models.py


class Step5InputData:


    def __init__(
        self,
        GENERATORS_UP: list,
        GENERATORS_DN: list,
        up_price: dict,
        dn_price: dict,
        up_cap: dict,
        dn_cap: dict,
        system_imbalance: float,
        load_curtailment_cost: float = 500.0,
    ):
        self.GENERATORS_UP = GENERATORS_UP
        self.GENERATORS_DN = GENERATORS_DN
        self.up_price = up_price
        self.dn_price = dn_price
        self.up_cap = up_cap
        self.dn_cap = dn_cap
        self.system_imbalance = system_imbalance
        self.load_curtailment_cost = load_curtailment_cost


class Step5BalancingMarket:
    """
    Minimize cost of balancing:

        min  sum_g [up_price_g * r_up_g]
           - sum_g [dn_price_g * r_dn_g]
           + LC_cost * LC

    s.t.
        sum_g r_up_g - sum_g r_dn_g + LC = system_imbalance   (balance)
        0 <= r_up_g <= up_cap_g       ∀g in GENERATORS_UP
        0 <= r_dn_g <= dn_cap_g       ∀g in GENERATORS_DN
        0 <= LC

    The dual of the balance constraint (Pi) gives the balancing price λ_B.
    For a minimisation problem: λ_B = +Pi  (increasing shortage by 1 MW
    raises cost by exactly the marginal offer price of the last activated unit).
    """

    def __init__(self, input_data: Step5InputData, output_flag: int = 0):
        self.data = input_data
        self.variables = Expando()
        self.constraints = Expando()
        self.results = Expando()
        self._build_model(output_flag=output_flag)

    def _build_variables(self):
        d = self.data
        self.variables.r_up = {
            g: self.model.addVar(lb=0, ub=d.up_cap[g], name=f'r_up_{g}')
            for g in d.GENERATORS_UP
        }
        self.variables.r_dn = {
            g: self.model.addVar(lb=0, ub=d.dn_cap[g], name=f'r_dn_{g}')
            for g in d.GENERATORS_DN
        }
        self.variables.LC = self.model.addVar(lb=0, ub=GRB.INFINITY, name='load_curtailment')

    def _build_constraints(self):
        d = self.data
        self.constraints.balance = self.model.addLConstr(
            gp.quicksum(self.variables.r_up[g] for g in d.GENERATORS_UP)
            - gp.quicksum(self.variables.r_dn[g] for g in d.GENERATORS_DN)
            + self.variables.LC
            == d.system_imbalance,
            name='Balancing_balance'
        )

    def _build_objective_function(self):
        d = self.data
        cost = (
            gp.quicksum(d.up_price[g] * self.variables.r_up[g] for g in d.GENERATORS_UP)
            - gp.quicksum(d.dn_price[g] * self.variables.r_dn[g] for g in d.GENERATORS_DN)
            + d.load_curtailment_cost * self.variables.LC
        )
        self.model.setObjective(cost, GRB.MINIMIZE)

    def _build_model(self, output_flag: int = 0):
        self.model = gp.Model(name='Step 5 - Balancing Market')
        self.model.setParam('OutputFlag', output_flag)
        self._build_variables()
        self._build_constraints()
        self._build_objective_function()
        self.model.update()

    def _save_results(self):
        d = self.data
        self.results.r_up = {g: self.variables.r_up[g].X for g in d.GENERATORS_UP}
        self.results.r_dn = {g: self.variables.r_dn[g].X for g in d.GENERATORS_DN}
        self.results.LC = self.variables.LC.X
        self.results.objective_value = self.model.ObjVal
        # For minimisation: λ_B = +Pi (not -Pi as in the welfare-max steps)
        self.results.balance_dual = self.constraints.balance.Pi
        self.results.balancing_price = self.results.balance_dual

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            raise RuntimeError(f"Optimization of {self.model.ModelName} was not successful")


# ========================
# STEP 6: RESERVE MARKET (European Sequential)
# ========================

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1: RESERVE MARKET
# ──────────────────────────────────────────────────────────────────────────────

class Step6aInputData:
    """
    Reserve market input data.

    The TSO minimizes reserve procurement cost subject to meeting
    system-wide upward and downward reserve requirements.

    Reserve offer prices:
        up_res_price[g]  = 0.25 * c_g   $/MW
        dn_res_price[g]  = 0.40 * c_g   $/MW

    Reserve capacity limits per generator:
        R_up_max[g]  = 0.80 * (Pmax_g - pG_DA_step1_g)
        R_dn_max[g]  = 0.20 * pG_DA_step1_g

    System requirements:
        R_up_req  = 0.15 * total_demand
        R_dn_req  = 0.10 * total_demand
    """

    def __init__(
        self,
        GENERATORS: list,
        GENERATORS_UP_RES: list,   # subset offering upward reserve
        GENERATORS_DN_RES: list,   # subset offering downward reserve
        up_res_price: dict,        # g -> $/MW
        dn_res_price: dict,        # g -> $/MW
        R_up_max: dict,            # g -> MW
        R_dn_max: dict,            # g -> MW
        R_up_req: float,           # MW system upward requirement
        R_dn_req: float,           # MW system downward requirement
    ):
        self.GENERATORS = GENERATORS
        self.GENERATORS_UP_RES = GENERATORS_UP_RES
        self.GENERATORS_DN_RES = GENERATORS_DN_RES
        self.up_res_price = up_res_price
        self.dn_res_price = dn_res_price
        self.R_up_max = R_up_max
        self.R_dn_max = R_dn_max
        self.R_up_req = R_up_req
        self.R_dn_req = R_dn_req


class Step6aReserveMarket:
    """
    Stage 1: Reserve market clearing.

    min  Σ_g [ up_res_price_g * r_up_g + dn_res_price_g * r_dn_g ]

    s.t.
        Σ_g r_up_g  = R_up_req           (upward reserve requirement)   → dual: λ_up_res
        Σ_g r_dn_g  = R_dn_req           (downward reserve requirement) → dual: λ_dn_res
        0 ≤ r_up_g ≤ R_up_max_g          ∀g in GENERATORS_UP_RES
        0 ≤ r_dn_g ≤ R_dn_max_g          ∀g in GENERATORS_DN_RES

    Reserve prices = +Pi of requirements constraints (minimization problem).
    """

    def __init__(self, input_data: Step6aInputData, output_flag: int = 0):
        self.data = input_data
        self.variables = Expando()
        self.constraints = Expando()
        self.results = Expando()
        self._build_model(output_flag=output_flag)

    def _build_variables(self):
        d = self.data
        self.variables.r_up = {
            g: self.model.addVar(lb=0, ub=d.R_up_max[g], name=f'r_up_{g}')
            for g in d.GENERATORS_UP_RES
        }
        self.variables.r_dn = {
            g: self.model.addVar(lb=0, ub=d.R_dn_max[g], name=f'r_dn_{g}')
            for g in d.GENERATORS_DN_RES
        }

    def _build_constraints(self):
        d = self.data
        self.constraints.up_req = self.model.addLConstr(
            gp.quicksum(self.variables.r_up[g] for g in d.GENERATORS_UP_RES)
            == d.R_up_req,
            name='Upward_reserve_requirement'
        )
        self.constraints.dn_req = self.model.addLConstr(
            gp.quicksum(self.variables.r_dn[g] for g in d.GENERATORS_DN_RES)
            == d.R_dn_req,
            name='Downward_reserve_requirement'
        )

    def _build_objective_function(self):
        d = self.data
        cost = (
            gp.quicksum(d.up_res_price[g] * self.variables.r_up[g]
                        for g in d.GENERATORS_UP_RES)
            + gp.quicksum(d.dn_res_price[g] * self.variables.r_dn[g]
                          for g in d.GENERATORS_DN_RES)
        )
        self.model.setObjective(cost, GRB.MINIMIZE)

    def _build_model(self, output_flag: int = 0):
        self.model = gp.Model(name='Step 6a - Reserve Market')
        self.model.setParam('OutputFlag', output_flag)
        self._build_variables()
        self._build_constraints()
        self._build_objective_function()
        self.model.update()

    def _save_results(self):
        d = self.data
        self.results.r_up = {g: self.variables.r_up[g].X for g in d.GENERATORS_UP_RES}
        self.results.r_dn = {g: self.variables.r_dn[g].X for g in d.GENERATORS_DN_RES}
        self.results.objective_value = self.model.ObjVal  # total reserve procurement cost

        # Reserve prices: +Pi for minimization problem
        self.results.up_res_price_dual = self.constraints.up_req.Pi
        self.results.dn_res_price_dual = self.constraints.dn_req.Pi
        self.results.lambda_up_res = self.results.up_res_price_dual
        self.results.lambda_dn_res = self.results.dn_res_price_dual

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            raise RuntimeError(f"Optimization of {self.model.ModelName} was not successful")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2: DAY-AHEAD MARKET WITH RESERVE CONSTRAINTS
# ──────────────────────────────────────────────────────────────────────────────

class Step6bInputData:
    """
    Day-ahead market input data for Stage 2, incorporating reserve commitments.

    Identical to Step1InputData but with two extra constraints per generator:
        pG_g <= Pmax_g - r_up_g   (upward reserve headroom preserved)
        pG_g >= r_dn_g            (downward reserve providers must be online)

    r_up and r_dn come from Step6aReserveMarket results.
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
        r_up: dict,   # g -> MW committed upward reserve (0 if not a provider)
        r_dn: dict,   # g -> MW committed downward reserve (0 if not a provider)
    ):
        self.GENERATORS = GENERATORS
        self.WINDS = WINDS
        self.LOAD_BUSES = LOAD_BUSES
        self.generator_cost = generator_cost
        self.generator_pmax = generator_pmax
        self.wind_avail = wind_avail
        self.demand_pmax = demand_pmax
        self.demand_bid = demand_bid
        self.r_up = r_up
        self.r_dn = r_dn


class Step6bDAMarket:
    """
    Stage 2: Day-ahead market clearing after reserve commitments.

    Identical objective to Step 1 (maximize social welfare) but with
    tighter generator bounds reflecting reserve commitments:

    max  Σ_n bid_n * d_n  -  Σ_g c_g * pG_g

    s.t.
        r_dn_g  <=  pG_g  <=  Pmax_g - r_up_g    ∀g
        0       <=  pW_w  <=  Wavail_w             ∀w
        0       <=  d_n   <=  Dmax_n               ∀n
        Σ_g pG_g + Σ_w pW_w - Σ_n d_n = 0         (balance → market price = -Pi)

    The upper bound Pmax_g - r_up_g ensures headroom for upward reserve.
    The lower bound r_dn_g ensures the generator is online for downward reserve.
    """

    def __init__(self, input_data: Step6bInputData, output_flag: int = 0):
        self.data = input_data
        self.variables = Expando()
        self.constraints = Expando()
        self.results = Expando()
        self._build_model(output_flag=output_flag)

    def _build_variables(self):
        d = self.data
        self.variables.pG = {
            g: self.model.addVar(
                lb=d.r_dn.get(g, 0.0),                    # must be online if downward reserve provider
                ub=d.generator_pmax[g] - d.r_up.get(g, 0.0),  # headroom reserved for upward
                name=f'Gen production {g}'
            )
            for g in d.GENERATORS
        }
        self.variables.pW = {
            w: self.model.addVar(lb=0, ub=d.wind_avail[w], name=f'Wind production {w}')
            for w in d.WINDS
        }
        self.variables.d = {
            n: self.model.addVar(lb=0, ub=d.demand_pmax[n], name=f'Demand served at bus {n}')
            for n in d.LOAD_BUSES
        }

    def _build_constraints(self):
        d = self.data
        self.constraints.balance = self.model.addLConstr(
            gp.quicksum(self.variables.pG[g] for g in d.GENERATORS)
            + gp.quicksum(self.variables.pW[w] for w in d.WINDS)
            - gp.quicksum(self.variables.d[n] for n in d.LOAD_BUSES),
            GRB.EQUAL,
            0,
            name='Balance constraint'
        )

    def _build_objective_function(self):
        d = self.data
        welfare = (
            gp.quicksum(d.demand_bid[n] * self.variables.d[n] for n in d.LOAD_BUSES)
            - gp.quicksum(d.generator_cost[g] * self.variables.pG[g] for g in d.GENERATORS)
        )
        self.model.setObjective(welfare, GRB.MAXIMIZE)

    def _build_model(self, output_flag: int = 0):
        self.model = gp.Model(name='Step 6b - DA Market with Reserve Constraints')
        self.model.setParam('OutputFlag', output_flag)
        self._build_variables()
        self._build_constraints()
        self._build_objective_function()
        self.model.update()

    def _save_results(self):
        d = self.data
        self.results.objective_value = self.model.ObjVal
        self.results.pG = {g: self.variables.pG[g].X for g in d.GENERATORS}
        self.results.pW = {w: self.variables.pW[w].X for w in d.WINDS}
        self.results.d  = {n: self.variables.d[n].X  for n in d.LOAD_BUSES}
        self.results.balance_dual = self.constraints.balance.Pi
        self.results.market_price = -self.results.balance_dual

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            raise RuntimeError(f"Optimization of {self.model.ModelName} was not successful")
