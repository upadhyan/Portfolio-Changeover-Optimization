import gc

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
from abc import ABC, abstractmethod
from experimental_config import *
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm


class TradingPolicy(ABC):
    def __init__(self, experiment: ExperimentInfo, verbose=False, **kwargs):
        self.exp = experiment
        self.trading_times = experiment.full_trading_times + [experiment.full_trading_times[-1] + pd.Timedelta(days=1)]
        self.price_dict = self.convert_to_price_dict()
        self.known_dict = self.develop_known_dict()
        self.verbose = verbose

    @abstractmethod
    def get_trades(self, portfolio, t):
        pass

    def convert_to_price_dict(self):
        price_dict = {}
        for i in range(len(self.trading_times)):
            if i != len(self.exp.full_trading_times):
                price_dict[self.trading_times[i]] = self.exp.forecasts[i]
            else:
                price_dict[self.trading_times[i]] = None
        return price_dict

    def develop_known_dict(self):
        known_dict = {}
        for i in range(len(self.trading_times)):
            if i == 0:
                known_dict[self.trading_times[i]] = self.exp.initial_prices
            else:
                known_dict[self.trading_times[i]] = self.exp.truth.loc[self.trading_times[i - 1].strftime("%Y-%m-%d")]
        last_known_dict = self.trading_times[-1] + pd.Timedelta(days=1)
        known_dict[last_known_dict] = self.exp.truth.loc[self.trading_times[-2].strftime("%Y-%m-%d")]
        return known_dict

    def vprint(self, print_statement):
        if self.verbose:
            print(print_statement)


class DayTradingPolicy(TradingPolicy):
    def __init__(self, experiment: ExperimentInfo, verbose=True, **kwargs):
        super().__init__(experiment, verbose, **kwargs)
        self.F = np.ones(self.exp.initial_portfolio.values[:-1].shape) * self.exp.trading_cost

    def get_trades(self, portfolio, t):
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", False)
        env.setParam("TimeLimit", 300)
        env.start()
        #######################
        p = portfolio.values[:-1]  # portfolio of number of positions
        current_cash = portfolio.values[-1]  # cash amount

        value = self.known_dict[t] @ p + current_cash
        assert value > 0.0
        previous_time_string = self.known_dict[t].name.strftime("%Y-%m-%d")
        self.vprint(f"Current Portfolio Value at {previous_time_string}: {value}")
        prob_arr = []
        z_arr = []
        cash_arr = []
        p_arr = []

        F = self.F
        M = value * 1e3

        trading_information = pd.concat([pd.DataFrame(self.known_dict[t]).T, self.price_dict[t]])
        p_next = None
        m = gp.Model(env=env)
        n_timesteps = len(trading_information.index)
        cash = current_cash
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

            prob_arr.append(prices @ p_next - F @ y_sell - F @ y_buy)
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
        self.vprint(
            f"\t Optimizing with {n_timesteps} time steps, "
            f"{len(m.getConstrs())} constraints, and {len(m.getVars())} variables..."
        )
        t1 = time()
        m.optimize()
        t2 = time()
        self.vprint(
            f"\t Optimized. Time taken: {t2 - t1}",
        )
        try:
            self.p_vals = [p.getValue() for p in p_arr]
            self.cash_vals = [_cash.getValue() for _cash in cash_arr]
            self.z_vals = [z.getValue() for z in z_arr]
        except Exception as e:
            self.vprint(e)
            return (
                pd.Series(index=portfolio.index, data=0, name=t),
                self.known_dict[t] @ portfolio[:-1] + portfolio[-1],
            )
        assert (np.round(self.p_vals[0]) >= 0).all()
        del m
        del env
        gc.collect()
        return (
            pd.Series(
                index=portfolio.index, data=(np.append(self.z_vals[0], self.cash_vals[0] - current_cash)), name=t
            ),
            value,
        )


class DirectionalTradingPolicy(TradingPolicy):
    def __init__(self, experiment: ExperimentInfo, verbose=True, **kwargs):
        super().__init__(experiment, verbose, **kwargs)
        self.F = np.ones(self.exp.initial_portfolio.values[:-1].shape) * self.exp.trading_cost

    def get_trades(self, portfolio, t):
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", False)
        env.setParam("TimeLimit", 300)
        env.start()
        #######################
        p = portfolio.values[:-1]  # portfolio of number of positions
        current_cash = portfolio.values[-1]  # cash amount

        value = self.known_dict[t] @ p + current_cash
        assert value > 0.0
        previous_time_string = self.known_dict[t].name.strftime("%Y-%m-%d")
        self.vprint(f"Current Portfolio Value at {previous_time_string}: {value}")
        prob_arr = []
        z_arr = []
        cash_arr = []
        p_arr = []

        ## Direction Definition
        initial_portfolio = self.exp.initial_portfolio.copy()[:-1]
        final = self.exp.final_portfolio.copy()[:-1]

        difference = final - initial_portfolio

        buy_constraint = (difference >= 0) * 1
        F = self.F
        M = value * 1e3

        trading_information = pd.concat([pd.DataFrame(self.known_dict[t]).T, self.price_dict[t]])
        p_next = None
        m = gp.Model(env=env)
        n_timesteps = len(trading_information.index)
        cash = current_cash
        for time_step in trading_information.index:
            prices = trading_information.loc[time_step].values

            z_buy = m.addMVar(p.shape, vtype=GRB.INTEGER, lb=0)
            z_sell = m.addMVar(p.shape, vtype=GRB.INTEGER, lb=0)
            y_sell = m.addMVar(p.shape, vtype=GRB.BINARY)
            y_buy = m.addMVar(p.shape, vtype=GRB.BINARY)

            ## Directional Constraints
            m.addConstr(z_buy <= buy_constraint.values * M)
            m.addConstr(z_sell <= (1 - buy_constraint.values) * M)
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

            prob_arr.append(prices @ p_next - F @ y_sell - F @ y_buy)
            cash = cash_next
            p = p_next

            z_arr.append(z_buy - z_sell)
            cash_arr.append(cash_next)
            p_arr.append(p_next)

        final_p = self.exp.final_portfolio.values[:-1]
        # Terminal constraint.
        m.addConstr(p_next >= final_p, name="terminal")

        # Combine all time instances
        obj = sum(prob_arr)

        m.setObjective(obj, GRB.MAXIMIZE)
        m.update()
        # get model constraints
        self.vprint(
            f"\t Optimizing with {n_timesteps} time steps, "
            f"{len(m.getConstrs())} constraints, and {len(m.getVars())} variables..."
        )
        t1 = time()
        m.optimize()
        t2 = time()
        self.vprint(
            f"\t Optimized. Time taken: {t2 - t1}",
        )
        try:
            self.p_vals = [p.getValue() for p in p_arr]
            self.cash_vals = [_cash.getValue() for _cash in cash_arr]
            self.z_vals = [z.getValue() for z in z_arr]
        except Exception as e:
            self.vprint(e)
            return (
                pd.Series(index=portfolio.index, data=0, name=t),
                self.known_dict[t] @ portfolio[:-1] + portfolio[-1],
            )
        assert (np.round(self.p_vals[0]) >= 0).all()
        del m
        del env
        gc.collect()
        return (
            pd.Series(
                index=portfolio.index, data=(np.append(self.z_vals[0], self.cash_vals[0] - current_cash)), name=t
            ),
            value,
        )


class DirectionalIncentiveTradingPolicy(TradingPolicy):
    def __init__(self, experiment: ExperimentInfo, verbose=True, lambda_=0.5, **kwargs):
        super().__init__(experiment, verbose, **kwargs)
        self.F = np.ones(self.exp.initial_portfolio.values[:-1].shape) * self.exp.trading_cost
        self.lambda_ = lambda_

    def get_trades(self, portfolio, t):
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", False)
        env.setParam("TimeLimit", 300)
        env.start()
        #######################
        p = portfolio.values[:-1]  # portfolio of number of positions
        current_cash = portfolio.values[-1]  # cash amount

        value = self.known_dict[t] @ p + current_cash
        assert value > 0.0
        previous_time_string = self.known_dict[t].name.strftime("%Y-%m-%d")
        self.vprint(f"Current Portfolio Value at {previous_time_string}: {value}")
        prob_arr = []
        z_arr = []
        cash_arr = []
        p_arr = []

        ## Direction Definition
        initial_portfolio = portfolio.copy()[:-1]
        final = self.exp.final_portfolio.copy()[:-1]

        difference = final - initial_portfolio

        buy_constraint = (difference >= 0) * 1  # 1 means we are allowed to buy, 0 means we are not allowed to buy

        F = self.F
        M = value * 1e3

        trading_information = pd.concat([pd.DataFrame(self.known_dict[t]).T, self.price_dict[t]])
        p_next = None
        m = gp.Model(env=env)
        n_timesteps = len(trading_information.index)
        cash = current_cash
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

            # ## Directional Constraints
            # m.addConstr(z_buy <= buy_constraint.values * M)
            # m.addConstr(z_sell <= (1 - buy_constraint.values) * M)
            # 1 means we are allowed to buy, 0 means we are not allowed to buy

            sell_penalty = (self.lambda_ * buy_constraint.values * prices) @ z_sell
            buy_penalty = (self.lambda_ * (1 - buy_constraint.values) * prices) @ z_buy
            prob_arr.append(- F @ y_sell - F @ y_buy - sell_penalty - buy_penalty)
            cash = cash_next
            p = p_next

            z_arr.append(z_buy - z_sell)
            cash_arr.append(cash_next)
            p_arr.append(p_next)

        final_p = self.exp.final_portfolio.values[:-1]
        # Terminal constraint.
        m.addConstr(p_next >= final_p, name="terminal")
        prob_arr.append(prices @ p_next)
        # Combine all time instances
        obj = sum(prob_arr)

        m.setObjective(obj, GRB.MAXIMIZE)
        m.update()
        # get model constraints
        self.vprint(
            f"\t Optimizing with {n_timesteps} time steps, "
            f"{len(m.getConstrs())} constraints, and {len(m.getVars())} variables..."
        )
        t1 = time()
        m.optimize()
        t2 = time()
        self.vprint(
            f"\t Optimized. Time taken: {t2 - t1}",
        )
        try:
            self.p_vals = [p.getValue() for p in p_arr]
            self.cash_vals = [_cash.getValue() for _cash in cash_arr]
            self.z_vals = [z.getValue() for z in z_arr]
        except Exception as e:
            self.vprint(e)
            return (
                pd.Series(index=portfolio.index, data=0, name=t),
                self.known_dict[t] @ portfolio[:-1] + portfolio[-1],
            )
        assert (np.round(self.p_vals[0]) >= 0).all()
        del m
        del env
        gc.collect()
        return (
            pd.Series(
                index=portfolio.index, data=(np.append(self.z_vals[0], self.cash_vals[0] - current_cash)), name=t
            ),
            value,
        )


class NaivePolicy(TradingPolicy):
    def __init__(self, experiment: ExperimentInfo, verbose=True, **kwargs):
        super().__init__(experiment, verbose, **kwargs)
        self.initial = True
        self.F = np.ones(self.exp.initial_portfolio.values[:-1].shape) * self.exp.trading_cost

    def get_trades(self, portfolio, t):
        if not self.initial:
            return (
                pd.Series(index=portfolio.index, data=0, name=t),
                self.known_dict[t] @ portfolio[:-1] + portfolio[-1],
            )
        self.initial = False

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", False)
        env.start()
        #######################
        p = portfolio.values[:-1]  # portfolio of number of positions
        cash = portfolio.values[-1]  # cash amount

        value = self.known_dict[t] @ p + cash
        assert value > 0.0
        previous_time_string = self.known_dict[t].name.strftime("%Y-%m-%d")
        self.vprint(f"Current Portfolio Value at {previous_time_string}: {value}")

        F = self.F
        M = value * 1e3

        trading_information = pd.concat([pd.DataFrame(self.known_dict[t]).T, self.price_dict[t]])
        m = gp.Model(env=env)

        prices = self.known_dict[t].values
        # prices = trading_information.loc[time_step].values

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

        obj = prices @ p_next - F @ y_sell - F @ y_buy

        final_p = self.exp.final_portfolio.values[:-1]
        # Terminal constraint.
        m.addConstr(p_next >= final_p)

        m.setObjective(obj, GRB.MAXIMIZE)
        m.update()
        # get model constraints
        # self.vprint(
        #     f"\t Optimizing with {n_timesteps} time steps, "
        #     f"{len(m.getConstrs())} constraints, and {len(m.getVars())} variables..."
        # )
        t1 = time()
        m.optimize()
        t2 = time()
        self.vprint(
            f"\t Optimized. Time taken: {t2 - t1}",
        )
        try:
            self.z = (z_buy - z_sell).getValue()
            self.cash_next = cash_next.getValue()
            self.p_next = p_next.getValue()
        except Exception as e:
            self.vprint(e)
            return (
                pd.Series(index=portfolio.index, data=0, name=t),
                self.known_dict[t] @ portfolio[:-1] + portfolio[-1],
            )
        assert (np.round(self.p_next) >= 0).all()
        del m
        del env
        gc.collect()
        return pd.Series(index=portfolio.index, data=(np.append(self.z, self.cash_next - cash)), name=t), value


class RigidDayTrading(TradingPolicy):
    def __init__(self, experiment: ExperimentInfo, verbose=True, **kwargs):
        super().__init__(experiment, verbose, **kwargs)
        self.initial = True
        self.p_vals = None
        self.cash_vals = None
        self.z_vals = None
        self.value_dictionary = None
        self.F = np.ones(self.exp.initial_portfolio.values[:-1].shape) * self.exp.trading_cost

    def get_trades(self, portfolio, t):
        if self.initial:
            self.calculate_trades(portfolio, t)
            self.initial = False
        value = portfolio.values[:-1] @ self.known_dict[t] + portfolio.values[-1]
        trades = self.z_vals[t]
        trade_binary = self.y_vals[t]
        cash_change = self.known_dict[t] @ trades * -1 - self.F @ trade_binary
        return pd.Series(index=portfolio.index, data=(np.append(trades, cash_change)), name=t), value

    def calculate_trades(self, portfolio, t):
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
        self.vprint(f"Current Portfolio Value at {previous_time_string}: {value}")
        prob_arr = []
        z_arr = []
        cash_arr = []
        p_arr = []
        y_arr = []
        F = self.F
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

            prob_arr.append(prices @ p_next - F @ y_sell - F @ y_buy)
            cash = cash_next
            p = p_next

            z_arr.append(z_buy - z_sell)
            y_arr.append(y_buy + y_sell)
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
        self.vprint(
            f"\t Optimizing with {n_timesteps} time steps, "
            f"{len(m.getConstrs())} constraints, and {len(m.getVars())} variables..."
        )
        t1 = time()
        m.optimize()
        t2 = time()
        self.vprint(
            f"\t Optimized. Time taken: {t2 - t1}",
        )
        try:
            self.z_vals = {}
            self.y_vals = {}
            self.value_dictionary = {}
            self.cash_vals = {}
            for i, time_step in enumerate(trading_information.index):
                self.z_vals[time_step] = z_arr[i].getValue()
                self.y_vals[time_step] = y_arr[i].getValue()
                if i == 0:
                    self.value_dictionary[time_step] = value
                else:
                    self.value_dictionary[time_step] = (
                            p_arr[i - 1].getValue() @ self.known_dict[time_step] + cash_arr[i - 1].getValue()
                    )
                self.cash_vals[time_step] = cash_arr[i].getValue()
            self.p_vals = [p.getValue() for p in p_arr]
        except Exception as e:
            self.vprint(e)

        assert (np.round(self.p_vals[0]) >= 0).all()
        del m
        del env
        gc.collect()


class MarketSimulator:
    def __init__(self, experiment: ExperimentInfo, policy: TradingPolicy, verbose=True):
        self.configuration = experiment
        self.stored_times = experiment.full_trading_times + [experiment.full_trading_times[-1] + pd.Timedelta(days=1)]
        self.policy = policy
        self.historical_trades = None
        self.trading_dict = {}
        self.portfolio_value = []
        self.trading_times = []
        self.solve_times = []
        self.verbose = verbose
        self.gain = 0
        self.status = "Not Run"
        self.current_portfolio = None
        self.historical_portfolios = []
        self.total_trading_cost = 0

    def run(self):
        self.status = "Running"
        self.historical_trades = []
        # Get initial portfolio
        portfolio = self.configuration.initial_portfolio.copy()
        if self.verbose:
            print("Starting Simulation")
            pbar = trange(len(self.stored_times))
        else:
            pbar = range(len(self.stored_times))
        for i in pbar:
            # Get trade for the previous time step
            t = self.stored_times[i]
            relevant_time = self.stored_times[i - 1] if i > 0 else self.configuration.initial_prices.name
            t1 = time()
            trades, value = self.policy.get_trades(portfolio, t)
            asset_trades = trades[:-1]
            self.total_trading_cost = trading_cost_incurred = ((np.round(
                asset_trades) != 0) * 1).sum() * self.configuration.trading_cost
            t2 = time()
            self.solve_times.append(t2 - t1)
            # Save Trades
            self.historical_trades.append(trades)
            self.trading_dict[relevant_time] = trades
            # Apply trades
            portfolio = self._apply_trades(portfolio, trades)
            self.current_portfolio = portfolio
            self.historical_portfolios.append(portfolio)
            self.portfolio_value.append(value)
            self.trading_times.append(relevant_time)
            gc.collect()
        ## make trade corrections

        if self.check_portfolio(portfolio):
            self.status = "Complete"
        else:
            self.status = "Infeasible"
        self.gain = self.evaluate_gain()

        return portfolio

    def check_portfolio(self, portfolio):
        return (np.round(portfolio) >= self.configuration.final_portfolio).all()

    def evaluate_gain(self):
        return (self.portfolio_value[-1] - self.portfolio_value[0]) / self.portfolio_value[0]

    def _apply_trades(self, portfolio, trades):
        new_portfolio = portfolio + trades
        if not (np.round(new_portfolio) >= 0).all():
            # get all items in the list except the last one
            self.historical_trades = self.historical_trades[:-1]
            self.portfolio_value = [np.nan] * 3
            print("Infeasible Trade")
            return None
        else:
            return new_portfolio

    def plot_value(self, ax=None, **kwargs):
        if ax is None:
            plt.plot(self.trading_times, self.portfolio_value, **kwargs)
            plt.xlabel("Time")
            plt.ylabel("Portfolio Value")
            plt.title("Portfolio Value over Time")
        else:
            ax.plot(self.trading_times, self.portfolio_value, **kwargs)
            ax.set_xlabel("Time")
            ax.set_ylabel("Portfolio Value")
            ax.set_title("Portfolio Value over Time")
        plt.show()

    def plot_solve_time(self, ax=None, **kwargs):
        if ax is None:
            plt.plot(range(len(self.solve_times)), self.solve_times, **kwargs)
            plt.xlabel("Time")
            plt.ylabel("Solve Time")
            plt.title("Solve Time over Time")
        else:
            ax.plot(range(len(self.solve_times)), self.solve_times, **kwargs)
            ax.set_xlabel("Time")
            ax.set_ylabel("Solve Time")
            ax.set_title("Solve Time over Time")
        plt.show()


NAIVE = "Naive"
RIGID = "RigidDayTrading"
DAY_TRADING = "DayTrading"
DIRECTIONAL_TRADING = "Directional"


def DIRECTIONAL_INCENTIVE_TRADING(lambda_=0.5):
    assert lambda_ >= 0 and lambda_ <= 1
    return f"DirectionalIncentive_{lambda_ * 10}"


class MultiSimRunner:
    def __init__(self, experiments_directory, policies, trim=None):
        self.experiments_directory = experiments_directory
        self.experiments = os.listdir(experiments_directory)
        if trim is not None:
            self.experiments = self.experiments[trim:]
        self.policies = policies
        self.results = []
        self.simulators = {}
        self.trim = None

    def run(self, save_file=None):
        result_list = []
        pbar = tqdm(self.experiments)
        for experiment in pbar:
            with open(f"{self.experiments_directory}/{experiment}", 'rb') as f:
                exp = pickle.load(f)
            for policy in self.policies:
                if policy == "RigidDayTrading":
                    policy_instance = RigidDayTrading(exp, verbose=False)
                elif policy == "DayTrading":
                    policy_instance = DayTradingPolicy(exp, verbose=False)
                elif policy == "Naive":
                    policy_instance = NaivePolicy(exp, verbose=False)
                elif policy == "Directional":
                    policy_instance = DirectionalTradingPolicy(exp, verbose=False)
                elif policy.contains("DirectionalIncentive"):
                    lambda_ = float(policy.split("_")[1]) / 10
                    policy_instance = DirectionalIncentiveTradingPolicy(exp, lambda_=lambda_, verbose=False)
                else:
                    raise Exception("Policy not found")
                pbar.set_description(f"Running {policy} on {exp.exp_id}")
                t1 = time()
                simulator = MarketSimulator(exp, policy_instance, verbose=False)
                final_portfolio = simulator.run()
                t2 = time()
                pbar.set_description(f"Finished Running {policy} on {exp.exp_id}")
                self.simulators[(exp.exp_id, policy)] = simulator
                result_list.append({
                    "experiment": exp.exp_id,
                    "policy": policy,
                    "Gain": simulator.evaluate_gain(),
                    "Final Value": simulator.portfolio_value[-1],
                    "Initial Value": simulator.portfolio_value[0],
                    "num_stocks": exp.num_stocks,
                    "pct_variance": exp.pct_variance,
                    "initial_budget": exp.budget,
                    "trading_fee": exp.trading_cost,
                    "average_error": exp.average_error,
                    "status": simulator.status,
                    "runtime": t2 - t1,
                    "leftover_cash": final_portfolio[-1]
                })
                gc.collect()
                if save_file is not None:
                    self.results = pd.DataFrame(result_list)
                    self.results.to_csv(save_file)
            del exp
            gc.collect()
        self.results = pd.DataFrame(result_list)

    def get_results(self, save_file=None):
        self.run(save_file=save_file)
        return self.results
