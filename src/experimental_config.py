import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import pickle
import random
from tqdm import trange
import os
import logging
import gc

from src.forecasting import NBEATSForecast, ABM, GBM

torch.set_float32_matmul_precision("medium")
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)


# Create a custom exception to handle errors
class ExperimentGenerationError(Exception):
    pass


class PriceData:
    def __init__(self, time: pd.Timestamp, history: pd.DataFrame, forecast: pd.DataFrame):
        """
        Args:
            time: A pandas timestamp representing the time of the forecast
            history: A dataframe with the columns being the stocks and the index being the dates.
                     This represents the history of the stocks up to the time of the forecast.
            forecast: A dataframe with the columns being the stocks and the index being the dates.
        """
        assert time == history.index[-1], "The time must be the same as the last row of the history"
        # combine the index of history and forecast
        self.time = time
        self.history = history
        self.current_prices = pd.DataFrame(self.history.iloc[-1]).T
        self.forecast = forecast
        # stack the known on top of the forecast
        self.data = pd.concat([self.current_prices, self.forecast])

    def update(self):
        # TODO: make sure everything that needs updating is updated
        pass

    @staticmethod
    def convert_from_compact(data):
        # TODO: ds(datestamp), unique(ticket), y (price)
        pass

    @staticmethod
    def convert_from_long(data):
        # TODO: columns: tickers, data: values, index: date
        pass


class ExperimentInfo:
    """Create an experiment

    Objects:
        stock_prices (pd.DataFrame): Stock Price Dataframe
        covariates (pd.DataFrame): Covariance Matrix for the stocks
        lookback (int): Number of lookback days to use for estimates
        min_horizon (int): Minimum investment horizon for experiments (hint: set min = max for a fixed value)
        max_horizon (int): Maximum investment horizon for experiments
        min_num_stocks (int): Minimum number of stocks to use in experiments (hint: set min = max for a fixed value)
        max_num_stocks (int): Maximum number of stocks to use in experiments
        rng (np.random.default_rng): Random number generator
        number_of_stocks (int): Number of stocks to use in the experiment
        stocks (list): List of stocks to use in the experiment
        pct_variance (pd.Series): Percentage of variance explained by each stock
        initial_prices (pd.Series): Initial prices of the stocks
        forecast (pd.DataFrame): Forecast model for the stocks
        errors (pd.DataFrame): Errors for the stocks
        average_error (float): Average error for the stocks
        budget (int): Budget for the experiment
        initial_portfolio (pd.Series): Initial portfolio for the experiment
        final_portfolio (pd.Series): Final portfolio for the experiment
        initial_date (pd.Timestamp): Initial date for the experiment
        final_date (pd.Timestamp): Final date for the experiment
        truth (pd.Series): Truth for the experiment
        initial_portfolio_value (float): Initial portfolio value for the experiment
        final_portfolio_value (float): Final portfolio value for the experiment
        returns (pd.Series): Returns for the experiment
        full_trading_times (pd.Series): Full trading times for the experiment
        exp_name (str): Name of the experiment
        exp_id (str): ID of the experiment
    """

    MODEL_DICT = {"GBM": GBM, "ABM": ABM, "NBEATS": NBEATSForecast}

    def __init__(
        self,
        stock_prices: str,
        covariates: pd.DataFrame,
        lookback: int,
        horizon: int,
        budget: float,
        num_stocks: int,
        tx_cost: int,
        date: pd.Timestamp,
        forecast_model: str,
        forecast_params: dict,
    ):
        """Create an experiment

        Args:
            price_data_dir (pd.DataFrame): Stock Price Dataframe
            covariates (pd.DataFrame): Covariance Matrix for the stocks
            lookback (int): Number of lookback days to use for estimates
            horizon (int): Investment horizon for experiments
            budget (float): Budget for the experiment
            num_stocks (int): Number of stocks to use in experiments
            tx_cost (int): Transaction cost to use in experiments
            date (pd.Timestamp): Starting date for the experiment
            forecast_model (str): Forecast model to use for experiments
        """

        # Validations
        assert lookback > 0, "Lookback must be greater than 0"
        assert horizon > 0, "Horizon must be greater than 0"
        assert num_stocks > 0, "Number of stocks must be greater than 0"
        assert tx_cost >= 0, "Transaction cost must be greater or equal to 0"
        assert (
            num_stocks <= stock_prices.shape[1]
        ), "Number of stocks must be less than or equal to the number of stocks in the data"
        assert horizon <= stock_prices.shape[0], "Horizon must be less than or equal to the number of days in the data"
        assert date > pd.Timestamp("2018-01-01"), "Date must be after 2018-01-01"
        assert budget > 0, "Budget must be greater than 0"

        # Set variables
        self.num_stocks = num_stocks
        self.tx_cost = tx_cost
        self.lookback = lookback
        self.horizon = horizon
        self.budget = budget
        self.covariates = covariates
        self.start_date = date

        # None variables
        self.tickers = None
        self.initial_portfolio = None
        self.target_portfolio = None
        self.initial_prices = None
        
        # Configure experiment
        self.choose_tickers(stock_prices)
        # order self.tickers alphabetically
        self.tickers.sort()
        stock_prices = stock_prices[self.tickers]
        # re-index the stock price data
        new_index = pd.date_range(start=min(stock_prices.index), end=max(stock_prices.index), freq="B")
        stock_prices = stock_prices.reindex(new_index, fill_value=np.nan).interpolate()
        self.create_portfolios(stock_prices)
        self.forecast_model_name = forecast_model
        self.forecast_model = self.MODEL_DICT[forecast_model](
            price_data=stock_prices[self.tickers],
            lookback=self.lookback,
            horizon=self.horizon,
            **forecast_params,
        )
        ## generate experiment id
        self.generate_exp_id()

    def choose_tickers(self, stock_prices):
        # subset the stock prices to a year before the start date and 90 days after the start date
        subset_prices = stock_prices.loc[
            self.start_date - pd.Timedelta(days=365) : self.start_date + pd.Timedelta(days=90)
        ]
        # replace 0 values with nan
        subset_prices = subset_prices.replace(0, np.nan)
        # drop columns with nan or zero values
        subset_prices = subset_prices.dropna(axis=1)
        tickers = subset_prices.columns.tolist()
        # choose a random number of stocks between 5 and 30
        rng = np.random.default_rng()
        self.tickers = rng.choice(tickers, size=self.num_stocks, replace=False)

    def create_portfolios(self, stock_prices):
        rng = np.random.default_rng()
        # get stock prices on the start date for the chosen tickers
        self.initial_prices = stock_prices.loc[self.start_date]
        initial_portfolio = pd.Series(0, index=self.tickers)
        purchase = pd.Series(0, index=self.tickers)
        # create a random initial portfolio with a budget using the current prices
        while initial_portfolio @ self.initial_prices < self.budget:
            purchase[rng.choice(self.tickers)] += rng.integers(1, 5)
            if purchase @ self.initial_prices + initial_portfolio @ self.initial_prices > self.budget:
                break
            initial_portfolio = initial_portfolio.add(purchase)
            purchase = pd.Series(0, index=self.tickers)
        leftover_cash = self.budget - initial_portfolio @ self.initial_prices
        # order the initial portfolio alphabetically
        initial_portfolio = initial_portfolio.sort_index(ascending=True)
        initial_portfolio["cash"] = leftover_cash
        self.initial_portfolio = initial_portfolio

        # create a random target portfolio with a budget using the current prices
        target = pd.Series(0, index=self.tickers)
        purchase = pd.Series(0, index=self.tickers)
        while target @ self.initial_prices < self.budget:
            purchase[rng.choice(self.tickers)] += rng.integers(1, 5)
            if purchase @ self.initial_prices + target @ self.initial_prices > self.budget:
                break
            target += purchase
        # order the target portfolio alphabetically
        target = target.sort_index(ascending=True)
        target["cash"] = leftover_cash
        self.target_portfolio = target

    def generate_exp_id(self):
        t = pd.to_datetime(str(self.start_date))
        timestring = t.strftime("%Y%m%d")
        self.exp_id = f"{timestring}_{self.horizon}_{self.num_stocks}_{self.tx_cost}_{int(self.budget)}"


def generate_experiments(
    price_data_dir: str,
    covariates,
    num_experiments,
    output_dir,
    min_horizon: int,
    max_horizon: int,
    max_num_stocks: int,
    min_num_stocks: int,
    min_tx_cost: int,
    max_tx_cost: int,
    max_budget: float,
    min_budget: float,
    forecast_model: str,
    forecast_params: dict,
    lookback: int = 60,
):
    """Generates a set of n experiments and saves them to a directory
    Args:
        price_data_dir (_type_): Directory of Stock Price Dataframe
        covariates (_type_): Covariance Matrix for the stocks
        num_experiments (_type_): Number of experiments to generate
        output_dir (_type_): Output directory for experiments
        min_horizon (int): Minimum investment horizon for experiments (hint: set min = max for a fixed value)
        max_horizon (int): Maximum investment horizon for experiments
        max_num_stocks (int): Maximum number of stocks to use in experiments (hint: set min = max for a fixed value)
        min_num_stocks (int): Minimum number of stocks to use in experiments
        min_tx_cost(int): Minimum transaction cost (hint: set min = max for a fixed value)
        max_tx_cost(int): Maximum transaction cost
        max_budget(float): Maximum budget (hint: set min = max for a fixed value)
        min_budget(float): Minimum budget
        forecast_model(str): Forecast model to use for experiments
    """

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the stock price data
    stock_price_df = pd.read_parquet(price_data_dir)

    # re-index the stock price data
    new_index = pd.date_range(start=min(stock_price_df.index), end=max(stock_price_df.index), freq="B")
    stock_price_df = stock_price_df.reindex(new_index, fill_value=np.nan).interpolate()

    rng = np.random.default_rng()

    pbar = trange(num_experiments)
    for i in pbar:
        # Try to generate a valid experiment with the given parameters
        try:
            lookback = lookback
            horizon = rng.integers(min_horizon, max_horizon)
            number_of_stocks = rng.integers(min_num_stocks, max_num_stocks)
            trading_cost = rng.integers(min_tx_cost, max_tx_cost)
            budget = np.round(rng.uniform(min_budget, max_budget), 2)
            temp = stock_price_df.loc["2018-03-01":]
            start_date = rng.choice(temp.index[:-horizon])
            # Create an experiment
            exp = ExperimentInfo(
                stock_prices=stock_price_df,
                covariates=covariates,
                lookback=lookback,
                horizon=horizon,
                budget=budget,
                num_stocks=number_of_stocks,
                tx_cost=trading_cost,
                date=start_date,
                forecast_model=forecast_model,
                forecast_params=forecast_params,
            )
            # Save the experiment to a file in the output directory

            with open(os.path.join(output_dir, f"{exp.exp_id}.pkl"), "wb") as f:
                pickle.dump(exp, f)
        except ExperimentGenerationError as e:
            pbar.set_description(f"Experiment {i} failed with error {e}.")
