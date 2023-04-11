import gurobipy as gp
from gurobipy import GRB
from abc import ABC, abstractmethod
from experimental_config import *
from time import time
import matplotlib.pyplot as plt


class TradingPolicy(ABC):
    def __init__(self, experiment: ExperimentInfo, verbose=False, **kwargs):
        self.exp = experiment
        self.trading_times = experiment.full_trading_times
        self.price_dict = self.convert_to_price_dict()
        self.known_dict = self.develop_known_dict()
        self.verbose = verbose

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

    def vprint(self, print_statement):
        if self.verbose:
            print(print_statement)


class DayTradingPolicy(TradingPolicy):
    def __init__(self, experiment: ExperimentInfo, verbose=True, **kwargs):
        super().__init__(experiment, verbose, **kwargs)

    def get_trades(self, portfolio, t):
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", False)
        # env.setParam("TimeLimit",time_limit)
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

        F = np.ones(p.shape) * self.exp.trading_cost
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
        self.vprint(
            f"\t Optimizing with {n_timesteps} time steps, "
            f"{len(m.getConstrs())} constraints, and {len(m.getVars())} variables..."
        )
        t1 = time()
        m.optimize()
        t2 = time()
        self.vprint(f"\t Optimized. Time taken: {t2 - t1}", )
        try:
            self.p_vals = [p.getValue() for p in p_arr]
            self.cash_vals = [_cash.getValue() for _cash in cash_arr]
            self.z_vals = [z.getValue() for z in z_arr]
        except Exception as e:
            self.vprint(e)
        assert (self.p_vals[0] >= 0).all()
        del m
        del env
        gc.collect()
        return pd.Series(index=portfolio.index, data=(np.append(self.z_vals[0], self.cash_vals[0] - current_cash)),
                         name=t), value


class NaivePolicy(TradingPolicy):
    def __init__(self, experiment: ExperimentInfo):
        super().__init__(experiment)
        self.initial = True

    def get_trades(self, portfolio, t):
        if not self.initial:
            return pd.Series(index=portfolio.index, data=0, name=t)
        self.initial = False


class RigidDayTrading(TradingPolicy):
    def __init__(self, experiment: ExperimentInfo, verbose=True, **kwargs):
        super().__init__(experiment, verbose, **kwargs)
        self.initial = True
        self.p_vals = None
        self.cash_vals = None
        self.z_vals = None
        self.value_dictionary = None

    def get_trades(self, portfolio, t):
        if self.initial:
            self.calculate_trades(portfolio, t)
            self.initial = False
        value = portfolio.values[:-1] @ self.known_dict[t] + portfolio.values[-1]
        trades = self.z_vals[t]
        cash = self.known_dict[t] @ trades
        return pd.Series(index=portfolio.index, data=(np.append(trades, cash)), name=t), value

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
        self.vprint(
            f"\t Optimizing with {n_timesteps} time steps, "
            f"{len(m.getConstrs())} constraints, and {len(m.getVars())} variables..."
        )
        t1 = time()
        m.optimize()
        t2 = time()
        self.vprint(f"\t Optimized. Time taken: {t2 - t1}", )
        try:
            self.z_vals = {}
            self.value_dictionary = {}
            self.cash_vals = {}
            for i, time_step in enumerate(trading_information.index):
                self.z_vals[time_step] = z_arr[i].getValue()
                if i == 0:
                    self.value_dictionary[time_step] = value
                else:
                    self.value_dictionary[time_step] = p_arr[i - 1].getValue() @ self.known_dict[time_step] + cash_arr[
                        i - 1].getValue()
                self.cash_vals[time_step] = cash_arr[i].getValue()
            self.p_vals = [p.getValue() for p in p_arr]
        except Exception as e:
            self.vprint(e)
        assert (self.p_vals[0] >= 0).all()
        del m
        del env
        gc.collect()


class MarketSimulator:
    def __init__(self, experiment: ExperimentInfo, policy: TradingPolicy):
        self.configuration = experiment
        self.stored_times = experiment.full_trading_times
        self.policy = policy
        self.historical_trades = None
        self.trading_dict = {}
        self.portfolio_value = []
        self.trading_times = []
        self.solve_times = []

    def run(self):
        self.historical_trades = []
        # Get initial portfolio
        portfolio = self.configuration.initial_portfolio.copy()
        print("Starting Simulation")
        pbar = trange(len(self.stored_times))
        for i in pbar:
            # Get trade for the previous time step
            t = self.stored_times[i]
            relevant_time = self.stored_times[i - 1] if i > 0 else self.configuration.initial_prices.name
            t1 = time()
            trades, value = self.policy.get_trades(portfolio, t)
            t2 = time()
            self.solve_times.append(t2 - t1)
            # Save Trades
            self.historical_trades.append(trades)
            self.trading_dict[relevant_time] = trades
            # Apply trades
            portfolio = self._apply_trades(portfolio, trades)
            self.portfolio_value.append(value)
            self.trading_times.append(relevant_time)
            gc.collect()
        return portfolio

    def evaluate_gain(self):
        return (self.portfolio_value[-1] - self.portfolio_value[0]) / self.portfolio_value[0]

    def _apply_trades(self, portfolio, trades):
        # portfolio[:-1] = portfolio[:-1] + trades[:-1]  # apply trades
        # portfolio[-1] = trades[-1]  # update cash
        # return portfolio
        return portfolio + trades

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


class MultiSimRunner:
    def __init__(self, experiments, policies):
        self.experiments = experiments
        self.policies = policies
        self.results = {}
        self.simulators = {}

    def run(self):
        for experiment in self.experiments:
            for policy in self.policies:
                simulator = MarketSimulator(experiment, policy)
                simulator.run()
                self.simulators[(experiment, policy)] = simulator
