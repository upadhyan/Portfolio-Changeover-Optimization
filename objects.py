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

        trading_information = pd.concat([pd.DataFrame(self.known_dict[t]).T, self.price_dict[t]])
        returns = trading_information.pct_change().fillna(0) + 1

        port_value_bounds = value * returns.max(axis=1).cumprod()

        p_next = None
        m = gp.Model(env=env)
        n_timesteps = len(trading_information.index)
        cash = current_cash
        ### Generate all possible trade matrices -> store it in a matrix list of size len(trading_info)
        for time_step in trading_information.index:
            prices = trading_information.loc[time_step].values

            z_buy = m.addMVar(p.shape, vtype=GRB.INTEGER, lb=0)
            z_sell = m.addMVar(p.shape, vtype=GRB.INTEGER, lb=0)
            y_sell = m.addMVar(p.shape, vtype=GRB.BINARY)
            y_buy = m.addMVar(p.shape, vtype=GRB.BINARY)
            ## lambda_it_sell = admvar(L(## of choice), vtype = binary )
            ## lambda_it_buy = admvar(L(## of choice), vtype = binary )
            ## m.add constr (sum lambdas  == 1)

            ## Directional Constraints
            bound_at_time = port_value_bounds.loc[time_step]
            M = np.ceil(bound_at_time / prices)
            # m.addConstr(z_buy <= buy_constraint.values * M)
            # m.addConstr(z_sell <= (1 - buy_constraint.values) * M)
            m.addConstr(y_buy <= buy_constraint.values)
            m.addConstr(y_sell <= (1 - buy_constraint.values))

            # pull relevant vectors
            # vectors_buy = [buy_matrix[:,timestep] for matrix in buy_matrix_list]
            # vectors_buy = [sell_matrix[:,timestep] for matrix in sell_matrix_list]

            # Next Portfolio
            p_next = p + z_buy - z_sell
            # y_sell = lambda_it_sell^T @ vectors_sell
            # y_buy = lambda_it_buy^T @ vectors_buy
            cash_next = prices @ (z_sell - z_buy) - F @ y_sell - F @ y_buy + cash

            ## Trading fees
            ### if we sell a stock, we pay a trading price
            m.addConstr(M * y_sell >= z_sell)

            ## If we buy a stock, we pay a trading fee
            m.addConstr(M * y_buy >= z_buy)

            ## No borrowing
            m.addConstr(cash_next >= 0)

            ## No shorting
            m.addConstr(p_next >= 0)

            prob_arr.append(- F @ y_sell - F @ y_buy)
            cash = cash_next
            p = p_next

            z_arr.append(z_buy - z_sell)
            cash_arr.append(cash_next)
            p_arr.append(p_next)

        final_p = self.exp.final_portfolio.values[:-1]
        # Terminal constraint.
        m.addConstr(p_next >= final_p, name="terminal")

        prob_arr.append(prices @ p_next + cash_next)

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


class DirectionalPenaltyTradingPolicy(TradingPolicy):
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
        returns = trading_information.pct_change().fillna(0) + 1

        port_value_bounds = value * returns.max(axis=1).cumprod()
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

            bound_at_time = port_value_bounds.loc[time_step]
            M = np.ceil(bound_at_time / prices)
            ## Trading fees
            ### if we sell a stock, we pay a trading price
            m.addConstr(M * y_sell >= z_sell)

            ## If we buy a stock, we pay a trading fee
            m.addConstr(M * y_buy >= z_buy)

            ## Valid Inequality
            m.addConstr(y_sell + y_buy <= 1)

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

        prob_arr.append(prices @ p_next + cash_next)
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
        M_buy = np.ceil(value / prices)
        M_sell = p
        ## Trading fees
        ### if we sell a stock, we pay a trading price
        m.addConstr(M_sell * y_sell >= z_sell)

        ## If we buy a stock, we pay a trading fee
        m.addConstr(M_buy * y_buy >= z_buy)

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
        self.total_trades = 0

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
            self.total_trading_cost += ((np.round(
                asset_trades) != 0) * 1).sum() * self.configuration.trading_cost
            self.total_trades += (np.round(asset_trades) != 0).sum()
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
RIGID = "RigidDirectional"
DIRECTIONAL_TRADING = "Directional"


def RIGID_INCENTIVE_TRADING(lambda_=0.5):
    return f"RigidIncentive_{lambda_ * 100}"


def DIRECTIONAL_INCENTIVE_TRADING(lambda_=0.5):
    return f"DirP_{lambda_ * 100}"





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

    def get_policy(self, policy, exp):
        penalty = None
        if policy == "Naive":
            policy_instance = NaivePolicy(exp, verbose=False)
        elif "DirP" in policy:
            lambda_ = float(policy.split("_")[1]) / 100
            policy_instance = DirectionalPenaltyTradingPolicy(exp, lambda_=lambda_, verbose=False)
            penalty = lambda_
        elif policy == "Directional":
            policy_instance = DirectionalTradingPolicy(exp, verbose=False)
        else:
            raise ValueError("Policy not found")
        return policy_instance, penalty

    def provide_run_stats(self, exp, simulator, policy, result, t1, t2, final_portfolio, penalty):
        self.simulators[(exp.exp_id, policy)] = simulator
        return {
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
            "leftover_cash": final_portfolio[-1],
            "penalty": penalty,
            "total_trades": simulator.total_trades,
            "total_trading_costs": simulator.total_trading_cost
        }

    def run(self, save_file=None):
        result_list = []
        pbar = tqdm(self.experiments)
        for experiment in pbar:
            with open(f"{self.experiments_directory}/{experiment}", 'rb') as f:
                exp = pickle.load(f)
            for policy in self.policies:
                policy_instance, penalty = self.get_policy(policy, exp)
                pbar.set_description(f"Running {policy} on {exp.exp_id}")
                t1 = time()
                simulator = MarketSimulator(exp, policy_instance, verbose=False)
                final_portfolio = simulator.run()
                t2 = time()
                pbar.set_description(f"Finished Running {policy} on {exp.exp_id}")
                self.simulators[(exp.exp_id, policy)] = simulator
                result_list.append(
                    self.provide_run_stats(exp, simulator, policy, result_list, t1, t2, final_portfolio, penalty))
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

    def run_specific(self, experiment_list):
        result_list = []
        pbar = tqdm(experiment_list)
        for exp in pbar:
            for policy in self.policies:
                policy_instance, penalty = self.get_policy(policy, exp)
                pbar.set_description(f"Running {policy} on {exp.exp_id}")
                t1 = time()
                simulator = MarketSimulator(exp, policy_instance, verbose=False)
                final_portfolio = simulator.run()
                t2 = time()
                pbar.set_description(f"Finished Running {policy} on {exp.exp_id}")
                self.simulators[(exp.exp_id, policy)] = simulator
                result_list.append(
                    self.provide_run_stats(exp, simulator, policy, result_list, t1, t2, final_portfolio, penalty))
                gc.collect()
        return pd.DataFrame(result_list)

