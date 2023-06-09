import pandas as pd
import numpy as np
from tqdm import trange
from src.policies import *
from src.forecasting import *

NAIVE = "Naive"
RIGID = "RigidDirectional"
DIRECTIONAL_TRADING = "Directional"


def DIRECTIONAL_INCENTIVE_TRADING(lambda_=0.5):
    return f"DirP_{lambda_ * 100}"


def COL_GEN(switch=True):
    return f"ColGen_{switch}"


class MarketSimulator:
    def __init__(self, experiment: ExperimentInfo, policy: TradingPolicy, forecast: Forecast, verbose=True):
        self.configuration = experiment
        self.stored_times = experiment.full_trading_times + [experiment.full_trading_times[-1] + pd.Timedelta(days=1)]
        self.policy = policy
        self.forecast = forecast
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
            t1_forecast = time()
            price_data = self.forecast.update(t, portfolio.index.tolist())
            t1 = time()
            trades, value = self.policy.get_trades(portfolio, t, price_data)
            asset_trades = trades[:-1]
            self.total_trading_cost += ((np.round(asset_trades) != 0) * 1).sum() * self.configuration.trading_cost
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

        self.status = "Complete" if self.check_portfolio(portfolio) else "Infeasible"
        self.gain = self.evaluate_gain()

        return portfolio

    def check_portfolio(self, portfolio):
        return (np.round(portfolio) >= self.configuration.final_portfolio).all()

    def evaluate_gain(self):
        return (self.portfolio_value[-1] - self.portfolio_value[0]) / self.portfolio_value[0]

    def _apply_trades(self, portfolio, trades):
        new_portfolio = portfolio + trades
        if (np.round(new_portfolio) >= 0).all():
            return new_portfolio
        # get all items in the list except the last one
        self.historical_trades = self.historical_trades[:-1]
        self.portfolio_value = [np.nan] * 3
        print("Infeasible Trade")
        return None

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
        elif "ColGen" in policy:
            switch = policy.split("_")[1] == "True"
            policy_instance = ColumnGenerationPolicy(exp, sell_switch=switch, verbose=False)
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
            "total_trading_costs": simulator.total_trading_cost,
        }

    def run(self, save_file=None):
        result_list = []
        pbar = tqdm(self.experiments)
        for experiment in pbar:
            with open(f"{self.experiments_directory}/{experiment}", "rb") as f:
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
                    self.provide_run_stats(exp, simulator, policy, result_list, t1, t2, final_portfolio, penalty)
                )
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
                    self.provide_run_stats(exp, simulator, policy, result_list, t1, t2, final_portfolio, penalty)
                )
                gc.collect()
        return pd.DataFrame(result_list)
