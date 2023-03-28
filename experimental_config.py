import pandas as pd
import numpy as np

from darts import TimeSeries
from darts.models import ExponentialSmoothing, NBEATSModel, NLinearModel, RNNModel, TCNModel, TransformerModel
from darts.metrics import mape, r2_score, mase, smape, mse
import matplotlib.pyplot as plt
import seaborn as sns

import torch

torch.set_float32_matmul_precision('medium')

import time
import random

from tqdm.notebook import trange

import os
import contextlib

import logging

logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)


class Experiment:
    def __init__(self, stock_prices: pd.DataFrame):
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
        self.target_stock = None
        self.initial_portfolio_value = None
        self.final_portfolio_value = None
        self.generate_experiment(stock_prices)

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
        forbidden_stock = random.randint(0, self.num_stocks - 1)
        self.target_stock = forbidden_stock
        # Set self.initial_portfolio to a random portfolio of stocks
        while (self.initial_portfolio + add) @ (self.initial_prices) < self.budget:
            # Make a random entry in add equal to 1
            self.initial_portfolio = self.initial_portfolio + add
            add[random.randint(0, self.num_stocks - 1)] = 1
            add[forbidden_stock] = 0
        add = np.zeros(self.num_stocks)
        self.initial_portfolio_value = self.initial_portfolio @ self.initial_prices
        # Create a target portfolio
        while (self.final_portfolio + add) @ (self.initial_prices) < self.initial_portfolio_value:
            self.final_portfolio = self.final_portfolio + add
            add[random.randint(0, self.num_stocks - 1)] = 1
            temp_final = self.final_portfolio + add
            temp_final[forbidden_stock] = 0
            if (temp_final > self.initial_portfolio).any():
                add = np.zeros(self.num_stocks)
        self.final_portfolio_value = self.final_portfolio @ self.initial_prices
    def generate_experiment(self, stock_prices):
        random_date = self.choose_split_date()
        # get the date two years before random date
        start_date = random_date - pd.DateOffset(years=3)
        # get the date 30 business days after random date
        end_date = random_date + pd.DateOffset(days=30)
        # convert random date to a string
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
        # Subset the data
        print(f"Start Date: {start_date}")
        print(f"End Date: {end_date}")
        subset = stock_prices.loc[start_date:end_date]
        # Replace all zeros with nan
        subset = subset.replace(0, np.nan)
        # Drop all columns with nans
        subset = subset.dropna(axis=1, how='any')
        n = random.randint(20, 50)
        # Choose n random stocks
        stocks = random.sample(subset.columns.tolist(), n)
        self.set_stocks(stocks)

        # subset the data to only the stocks we chose
        subset = subset[stocks]

        # reindex the subset with business days and set the index name to 'date'
        subset = subset.reindex(pd.date_range(start_date, end_date, freq='B'), method='ffill')
        subset.index.name = 'date'

        split_date = subset.index[-31]
        self.initial_date = split_date
        self.final_date = subset.index[-1]

        subset = subset.reset_index()

        ts = TimeSeries.from_dataframe(subset, time_col='date', value_cols=stocks)

        train, val = ts.split_after(split_date)
        self.set_truth(val.pd_dataframe())
        self.set_initial_prices(train.pd_dataframe().iloc[-1])
        self.create_initial_portfolio()
        self.create_forecasts(ts, subset, train, val)

    def create_forecasts(self, time_series, subset, train, val):
        model = NLinearModel(
            input_chunk_length=120,
            output_chunk_length=30,
            n_epochs=100,
            random_state=42
        )
        model.fit(train, verbose=False)
        predictions = [None] * len(val)
        errors = []
        for i in range(len(val)):
            updated_split = subset.index[-31 + i]
            updated_train, updated_val = time_series.split_after(updated_split)
            forward_steps = len(val) - i
            pred = model.predict(forward_steps, series=updated_train, verbose=False)
            if i == 0:
                predictions[i] = pred.pd_dataframe()
            else:
                df1 = val[:i].pd_dataframe()
                df2 = pred.pd_dataframe()
                combined_df = pd.concat([df1, df2])
                predictions[i] = combined_df
            errors.append(mape(updated_val, pred))
        self.errors = errors
        self.forecasts = predictions
        self.average_error = np.mean(errors)

    @staticmethod
    def choose_split_date():
        minimum_date = pd.Timestamp('2002-01-01')
        maximum_date = pd.Timestamp('2021-12-31')
        random_date = pd.Timestamp(random.randint(minimum_date.value, maximum_date.value))
        random_date = random_date.date()
        return random_date


# write a if name == main statement
if __name__ == '__main__':
    # read in stock_data.parquet
    stock_prices = pd.read_parquet('spx_stock_prices.parquet')
    print(stock_prices)
    stock_prices.index = pd.to_datetime(stock_prices.index)
    # create a new experiment
    experiment = Experiment(stock_prices)
    print(experiment.average_error)
    truth = experiment.truth
    first_predict = experiment.forecasts[0]
    stock_one = experiment.stocks[0]
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    ax.plot(truth[stock_one], label='truth')
    ax.plot(first_predict[stock_one], label='prediction')
    ax.legend()
    plt.show()