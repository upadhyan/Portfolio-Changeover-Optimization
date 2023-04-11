import gurobipy as gp
from gurobipy import GRB
from abc import ABC, abstractmethod
from experimental_config import *
from time import time


class TradingPolicy(ABC):
    def __init__(self, experiment: ExperimentInfo, verbose=True, **kwargs):
        self.exp = experiment
        self.trading_times = experiment.full_trading_times
        self.price_dict = self.convert_to_price_dict()
        self.known_dict = self.develop_known_dict()

    @abstractmethod
    def get_trades(self, portfolio, t):
        pass

    def convert_to_price_dict(self):
        price_dict = {}
        for i in range(len(self.trading_times)):
            price_dict[self.trading_times[i]] = self.exp.forecasts[i]
        return price_dict

    def develop_known_dict(self):
        known_dict = {}
        for i in range(len(self.trading_times)):
            if i == 0:
                known_dict[self.trading_times[i]] = self.exp.initial_prices
            else:
                known_dict[self.trading_times[i]] = self.exp.truth.loc[self.trading_times[i - 1].strftime("%Y-%m-%d")]
        return known_dict


class DayTradingPolicy(TradingPolicy):
    def __init__(self, experiment: ExperimentInfo, verbose=True, **kwargs):
        super().__init__(experiment)

    def get_trades(self, portfolio, t):
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", False)
        # env.setParam("TimeLimit",time_limit)
        env.start()
        #######################
        p = portfolio.values[:-1]  # portfolio of number of positions
        cash = portfolio.values[-1]  # cash amount

        value = self.known_dict[t] @ p + cash
        assert value > 0.0
        previous_time_string = self.known_dict[t].name.strftime("%Y-%m-%d")
        print(f"Current Portfolio Value at {previous_time_string}: {value}")
        prob_arr = []
        z_arr = []
        cash_arr = []
        p_arr = []

        F = np.ones(p.shape) * self.exp.trading_cost
        M = value * 1e3

        trading_information = pd.concat([pd.DataFrame(self.known_dict[t]).T, self.price_dict[t]])
        p_next = None
        m = gp.Model(env=env)
        n_timesteps = len(trading_information.index)
        for time_step in trading_information.index:
            prices = trading_information.loc[time_step].values

            z_buy = m.addMVar(p.shape, vtype=GRB.INTEGER, lb=0)
            z_sell = m.addMVar(p.shape, vtype=GRB.INTEGER, lb=0)
            y_sell = m.addMVar(p.shape, vtype=GRB.BINARY)
            y_buy = m.addMVar(p.shape, vtype=GRB.BINARY)

            # Next Portfolio
            p_next = p + z_buy - z_sell
            cash_next = prices @ (z_sell - z_buy) - F @ y_sell - F @ y_buy + cash

            ## Trading fees
            ### if we sell a stock, we pay a trading price
            m.addConstr(M * (y_sell) >= z_sell)

            ## If we buy a stock, we pay a trading fee
            m.addConstr(M * (y_buy) >= z_buy)

            ## No borrowing
            m.addConstr(cash_next >= 0)

            ## No shorting
            m.addConstr(p_next >= 0)

            prob_arr.append(prices @ p_next - F @ z_sell - F @ z_buy)
            cash = cash_next
            p = p_next

            z_arr.append(z_buy - z_sell)
            cash_arr.append(cash_next)
            p_arr.append(p_next)

        final_p = self.exp.final_portfolio.values[:-1]
        # Terminal constraint.
        m.addConstr(p_next >= final_p)

        # Combine all time instances
        obj = sum(prob_arr)

        m.setObjective(obj, GRB.MAXIMIZE)
        m.update()
        # get model constraints
        print(
            f"\t Optimizing with {n_timesteps} time steps, "
            f"{len(m.getConstrs())} constraints, and {len(m.getVars())} variables..."
        )
        t1 = time()
        m.optimize()
        t2 = time()
        print("\t Optimized. Time taken: ", t2 - t1)
        try:
            self.p_vals = [p.getValue() for p in p_arr]
            self.cash_vals = [cash.getValue() for cash in cash_arr]
            self.z_vals = [z.getValue() for z in z_arr]
        except Exception as e:
            print(e)
        assert (self.p_vals[0] >= 0).all()
        del m
        del env
        gc.collect()
        return pd.Series(index=portfolio.index, data=(np.append(self.z_vals[0], self.cash_vals[0])), name=t)


class NaivePolicy(TradingPolicy):
    def __init__(self, experiment: ExperimentInfo):
        super().__init__(experiment)
        self.initial = True

    def get_trades(self, portfolio, t):
        if not self.initial:
            return pd.Series(index=portfolio.index, data=0, name=t)
        self.initial = False


class RigidMultiPeriodPolicy(TradingPolicy):
    def __init__(self, experiment: ExperimentInfo):
        super().__init__(experiment)
        self.initial = True
        self.p_vals = None
        self.cash_vals = None
        self.z_vals = None

    def get_trades(self, portfolio, t):
        if self.initial:
            self.calculate_trades()
        portfolio = self.z_vals[t]
        cash = self.cash_vals[t]
        return pd.Series(index=portfolio.index, data=(np.append(portfolio, cash)), name=t)

    def calculate_trades(self):
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", False)
        # env.setParam("TimeLimit",time_limit)
        env.start()
        #######################

        prob_arr = []
        z_arr = []
        cash_arr = []
        p_arr = []

        F = np.ones(self.exp.num_stocks) * self.exp.trading_cost
        M = self.exp.initial_portfolio_value * 1e3

        trading_information = self.exp.forecasts[0]
        p_next = None
        m = gp.Model(env=env)
        n_timesteps = len(trading_information.index)
        for time_step in trading_information.index:
            prices = trading_information.loc[time_step].values

            z_buy = m.addMVar(p.shape, vtype=GRB.INTEGER, lb=0)
            z_sell = m.addMVar(p.shape, vtype=GRB.INTEGER, lb=0)
            y_sell = m.addMVar(p.shape, vtype=GRB.BINARY)
            y_buy = m.addMVar(p.shape, vtype=GRB.BINARY)

            # Next Portfolio
            p_next = p + z_buy - z_sell
            cash_next = prices @ (z_sell - z_buy) - F @ y_sell - F @ y_buy + cash

            ## Trading fees
            ### if we sell a stock, we pay a trading price
            m.addConstr(M * (y_sell) >= z_sell)

            ## If we buy a stock, we pay a trading fee
            m.addConstr(M * (y_buy) >= z_buy)

            ## No borrowing
            m.addConstr(cash_next >= 0)

            ## No shorting
            m.addConstr(p_next >= 0)

            prob_arr.append(prices @ p_next - F @ z_sell - F @ z_buy)
            cash = cash_next
            p = p_next

            z_arr.append(z_buy - z_sell)
            cash_arr.append(cash_next)
            p_arr.append(p_next)

        final_p = self.exp.final_portfolio.values[:-1]
        # Terminal constraint.
        m.addConstr(p_next >= final_p)

        # Combine all time instances
        obj = sum(prob_arr)

        m.setObjective(obj, GRB.MAXIMIZE)
        m.update()
        # get model constraints
        print(
            f"\t Optimizing with {n_timesteps} time steps, "
            f"{len(m.getConstrs())} constraints, and {len(m.getVars())} variables..."
        )
        t1 = time()
        m.optimize()
        t2 = time()
        print("\t Optimized. Time taken: ", t2 - t1)

        try:
            self.p_vals = {}
            self.cash_vals = {}
            self.z_vals = {}
            for i in range(len(p_arr)):
                self.p_vals[trading_information.index[i]] = p_arr[i].getValue()
                self.cash_vals[trading_information.index[i]] = cash_arr[i].getValue()
                self.z_vals[trading_information.index[i]] = z_arr[i].getValue()
        except Exception as e:
            print(e)


class MarketSimulator:
    def __init__(self, experiment: ExperimentInfo, policy: TradingPolicy):
        self.configuration = experiment
        self.trading_times = experiment.full_trading_times
        self.policy = policy
        self.historical_trades = None
        self.trading_dict = {}

    def run(self):
        self.historical_trades = []
        # Get initial portfolio
        portfolio = self.configuration.initial_portfolio.copy()
        print("Starting Simulation")
        pbar = trange(len(self.trading_times))
        for i in pbar:
            # Get trade for the previous time step
            t = self.trading_times[i]
            relevant_time = self.trading_times[i - 1] if i > 0 else self.configuration.initial_prices.name
            trades = self.policy.get_trades(portfolio, t)
            # Save Trades
            self.historical_trades.append(trades)

            # Apply trades
            portfolio = self._apply_trades(portfolio, trades)
            gc.collect()
        return portfolio

    def _apply_trades(self, portfolio, trades):
        portfolio[:-1] = portfolio[:-1] + trades[:-1]  # apply trades
        portfolio[-1] = trades[-1]  # update cash
        return portfolio
