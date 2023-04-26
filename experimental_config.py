import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import NHiTSModel
from darts.metrics import mape, smape
from darts.utils.losses import MapeLoss, SmapeLoss
import pickle
import torch
import random
from tqdm import trange
import os
import logging
import gc

torch.set_float32_matmul_precision("medium")
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)


# Create a custom exception to handle errors
class ExperimentGenerationError(Exception):
    pass


class ExperimentInfo:
    def __init__(self, stock_prices: pd.DataFrame, covariates: pd.DataFrame, lookback):
        self.num_stocks = None
        self.stocks = None
        self.pct_variance = None
        self.initial_prices = None
        self.forecasts = None
        self.errors = None
        self.average_error = None
        self.budget = None
        self.initial_portfolio = None
        self.final_portfolio = None
        self.initial_date = None
        self.final_date = None
        self.truth = None
        self.initial_portfolio_value = None
        self.lookback = lookback
        self.full_trading_times = None
        self.trading_cost = np.random.randint(2, 10)
        self.exp_id = None
        self.generate_experiment(stock_prices, covariates)

    # Make a function that accepts a list as an input
    def set_stocks(self, stocks: list):
        assert len(stocks) > 0, "Must have at least one stock"
        self.stocks = stocks
        self.num_stocks = len(stocks)

    def set_initial_prices(self, initial_prices: pd.Series):
        assert self.stocks is not None, "Must set stocks before setting initial prices"
        assert len(initial_prices) == self.num_stocks, "Initial prices must have the correct number of stocks"
        assert initial_prices.isna().sum() == 0, "Initial prices must not have any missing values"
        assert initial_prices.min() > 0, "Initial prices must not have any negative values"
        self.initial_prices = initial_prices

    def set_truth(self, truth: pd.DataFrame):
        """Set the truth for the experiment. This is the actual price of the stocks over time.
        Args:
            truth: A dataframe with the columns being the stocks and the index being the dates.
        returns:
            None
        """
        assert self.stocks is not None, "Must set stocks before setting truth"
        assert truth.index.is_monotonic_increasing, "Truth must be sorted by date"
        assert truth.index.is_unique, "Truth must have unique dates"
        assert truth.shape[1] == self.num_stocks, "Truth must have the correct number of stocks"
        assert truth.isna().sum().sum() == 0, "Truth must not have any missing values"
        assert truth.min().min() > 0, "Truth must not have any negative values"
        assert truth.columns.tolist() == self.stocks, "Truth must have the correct stocks"
        self.truth = truth
        self.pct_variance = self.truth.pct_change().var().mean()

    def create_initial_portfolio(self):
        assert self.initial_prices is not None, "Must set initial prices before creating initial portfolio"
        # Set self.budget to a random dollar value between 15000 and 350000
        self.budget = random.randint(15000, 350000)
        self.initial_portfolio = np.zeros(self.num_stocks)
        self.final_portfolio = np.zeros(self.num_stocks)
        add = np.zeros(self.num_stocks)
        # Set forbidden stock
        # Set self.initial_portfolio to a random portfolio of stocks
        while (self.initial_portfolio + add) @ self.initial_prices < self.budget:
            # Make a random entry in add equal to 1
            self.initial_portfolio = self.initial_portfolio + add
            add[random.randint(0, self.num_stocks - 1)] = 1
        self.initial_portfolio_value = self.initial_portfolio @ self.initial_prices
        self.starting_cash = self.budget - self.initial_portfolio_value
        # append starting cash to initial portfolio
        self.initial_portfolio = np.append(self.initial_portfolio, self.starting_cash)
        add = np.zeros(self.num_stocks)

        # Create a target portfolio
        while (self.final_portfolio + add) @ self.initial_prices < self.initial_portfolio_value:
            self.final_portfolio = self.final_portfolio + add
            add[random.randint(0, self.num_stocks - 1)] = 1
        # append zero to final portfolio
        self.final_portfolio = np.append(self.final_portfolio, 0)
        self.initial_portfolio = pd.Series(self.initial_portfolio, index=self.stocks + ["cash"])
        self.initial_portfolio.name = self.initial_date
        self.final_portfolio = pd.Series(self.final_portfolio, index=self.stocks + ["cash"])
        self.initial_portfolio.name = self.final_date

    def generate_experiment(self, stock_prices: pd.DataFrame, covariates: pd.DataFrame):
        # Choose a random date
        random_date = self.choose_split_date()
        # get the date two years before random date
        start_date = random_date - pd.DateOffset(years=5)
        cov_start_date = random_date - pd.DateOffset(years=6)
        # get the date 30 business days after random date
        end_date = random_date + pd.DateOffset(days=30)
        cov_end_date = random_date + pd.DateOffset(days=90)
        # convert random date to a string
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")
        cov_end_date = cov_end_date.strftime("%Y-%m-%d")
        subset = stock_prices.loc[start_date:end_date]
        # Replace all zeros with nan
        subset = subset.replace(0, np.nan)
        # Drop all columns with nans
        subset = subset.dropna(axis=1, how="any")
        n = 3
        # Choose n random stocks
        stocks = random.sample(subset.columns.tolist(), n)
        self.set_stocks(stocks)

        # subset the data to only the stocks we chose
        subset = subset[stocks]
        full_idx = pd.date_range(start_date, end_date, freq="B")
        new_index = pd.date_range(cov_start_date, cov_end_date, freq="B")

        # Reindex the covariates
        covariates = covariates.reindex(new_index, method="ffill")
        # copy the subset
        subset_copy = subset.copy()
        # Rename the subset columns just in case
        subset_copy.columns = [f"{col}_price" for col in subset_copy.columns]
        # Add in the subset
        covariates = covariates.join(subset_copy, how="left")
        covariates = covariates.fillna(method="bfill")
        # Set the index name to 'date'
        covariates.index.name = "date"
        # Drop all columns with nans
        covariates = covariates.dropna(axis=1, how="any")
        # Get the list of covariates
        cov_names = covariates.columns.tolist()
        # Reset the index to create a date column
        covariates = covariates.reset_index()
        # Create a TimeSeries object from the covariates
        cov_ts = TimeSeries.from_dataframe(covariates, time_col="date", value_cols=cov_names)

        # Reindex the subset
        subset = subset.reindex(full_idx, method="ffill")
        subset.index.name = "date"

        # Get the date 30 trading periods before the end
        split_date = subset.index[-31]

        # reset the index
        subset = subset.reset_index()

        # Create a TimeSeries object from the subset
        ts = TimeSeries.from_dataframe(subset, time_col="date", value_cols=stocks)
        # split the data into training and validation
        train, val = ts.split_after(split_date)

        # Set the initial date and final date of the trading period
        self.initial_date = val.time_index.min()
        self.final_date = val.time_index.max()
        # Turn datetime index into a list of timestamps
        self.full_trading_times = [pd.Timestamp(time) for time in val.time_index]

        # Set the true prices
        self.set_truth(val.pd_dataframe())
        # Set the initial prices
        self.set_initial_prices(train.pd_dataframe().iloc[-1])
        # Set the initial portfolio
        self.create_initial_portfolio()
        # Generate Forecasts
        self.create_forecasts(ts, subset, train, val, cov_ts)
        self.generate_exp_id()

    def generate_exp_id(self):
        self.exp_id = (
            f"{self.num_stocks}_{self.budget}" f"_{int(self.initial_portfolio_value)}" f"_{int(self.trading_cost)}"
        )

    def create_forecasts(self, time_series, subset, train, val, cov_ts):
        model = NHiTSModel(
            input_chunk_length=self.lookback,
            output_chunk_length=30,
            n_epochs=125,
            random_state=42,
            layer_widths=128,
            num_layers=4,
            num_blocks=2,
            loss_fn=MapeLoss(),
        )

        model.fit(train, verbose=False, past_covariates=cov_ts)
        predictions = [None] * len(val)
        errors = []
        for i in range(len(val)):
            updated_split = subset.index[-31 + i]
            updated_train, updated_val = time_series.split_after(updated_split)
            forward_steps = len(val) - i
            current_prediction = model.predict(
                forward_steps, series=updated_train, verbose=False, past_covariates=cov_ts
            )
            current_prediction_df = current_prediction.pd_dataframe()
            # if any element of current_prediction_df is nan, raise an error
            if current_prediction_df.isna().sum().sum() > 0:
                raise ExperimentGenerationError("Forecast contains nan values")
            predictions[i] = current_prediction_df
            errors.append(mape(updated_val, current_prediction))
        self.errors = errors
        self.forecasts = predictions
        self.average_error = np.mean(errors)

    @staticmethod
    def choose_split_date():
        minimum_date = pd.Timestamp("2006-01-01")
        maximum_date = pd.Timestamp("2021-12-31")
        random_date = pd.Timestamp(random.randint(minimum_date.value, maximum_date.value))
        random_date = random_date.date()
        return random_date


def generate_experiments(stock_price_df, covariates, num_experiments, output_dir, lookback=128, error_max=9):
    pbar = trange(num_experiments)
    average_errors = []
    for i in pbar:
        too_much_error = True
        while too_much_error:
            try:
                experiment = ExperimentInfo(stock_price_df, covariates, lookback)
                # check if the output directory exists, if not, create it
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                f_name = (
                    f"exp_{experiment.num_stocks}_{experiment.budget}"
                    f"_{int(experiment.initial_portfolio_value)}"
                    f"_{int(experiment.trading_cost)}.pkl"
                )
                # save the experiment to a pickle file in output_dir
                if experiment.average_error < error_max:
                    too_much_error = False
                    with open(os.path.join(output_dir, f_name), "wb") as f:
                        pickle.dump(experiment, f)
                    average_errors.append(experiment.average_error)
                    pbar.set_description(f"Experiment {i} saved to {f_name}. Error is {experiment.average_error}")
                    del experiment
                    gc.collect()
                else:
                    pbar.set_description(f"Retrying experiment {i}. Error is {experiment.average_error}")
            except ExperimentGenerationError as e:
                pbar.set_description(f"Experiment {i} failed with error {e}.")
    print(f"Average error is {np.mean(average_errors)}")


if __name__ == "__main__":
    print("reading data")
    prices = pd.read_parquet("raw_data/spx_stock_prices.parquet")
    treasury_rate_files = ["daily-treasury-rates.csv"] + [f"daily-treasury-rates ({i}).csv" for i in range(1, 25)]
    rates_df = [pd.read_csv(f"raw_data/{file}", index_col=0) for file in treasury_rate_files]
    rates_df = pd.concat(rates_df)
    rates_df.index = pd.to_datetime(rates_df.index)
    # sort rates_df by date
    rates_df = rates_df.sort_index()
    print("Start Generation")
    generate_experiments(prices, rates_df, 5, "experiments", lookback=48, error_max=10)
