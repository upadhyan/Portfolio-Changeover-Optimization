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

from forecasting import NBEATSForecast, ABM, GBM

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


class ExperimentalConfig:
    def __init__(
        self,
        budget: int,
        stocks: list,
        trading_cost: int,
        horizon: int,
        starting_step: pd.Timestamp,
        initial_portfolio: pd.Series,
        target_portfolio: pd.Series,
        data_file: str,
        model_directory: str = "./models",
    ):
        """_summary_

        Args:
            budget (int): _description_
            stocks (list): _description_
            trading_cost (int): _description_
            horizon (int): _description_
            starting_step (pd.Timestamp): _description_
            initial_portfolio (pd.Series): _description_
            target_portfolio (pd.Series): _description_
            data_file (str): _description_
            model_directory (str, optional): _description_. Defaults to "./models".
        """
        self.budget = budget
        self.stocks = stocks
        self.trading_cost = trading_cost
        self.horizon = horizon
        self.starting_step = starting_step
        self.initial_portfolio = initial_portfolio
        self.target_portfolio = target_portfolio
        self.data_file = data_file
        self.model_directory = model_directory
        self.num_stocks = len(self.stocks)

    @staticmethod
    def create_config(
        budget: int = random.randint(5000, 350000),
        stocks: list = None,
        num_stocks: int = random.randint(5, 30),
        trading_cost: int = random.randint(2, 10),
        horizon: int = random.randint(30, 90),
        starting_step: pd.Timestamp = None,
        data_file: str = "./raw_data/spx_stock_prices.parquet",
        model_directory: str = "./models",
    ):
        """
        Args:
            budget: The budget for the experiment
            stocks: The stocks to use in the experiment
            num_stocks: The number of stocks to use in the experiment
            trading_cost: The trading cost for the experiment
            horizon: The horizon for the experiment
            starting_step: The starting step for the experiment
            data_file: The file to load the data from
            model_directory: The directory to load the models from
        """

        # load the data
        stock_prices = pd.read_parquet(data_file)
        stock_prices = stock_prices.loc["2018-01-01":"2022-03-01"]
        # drop columns with nan values
        stock_prices = stock_prices.dropna(axis=1)

        if stocks is None:
            stocks = stock_prices.columns.tolist()
            stocks = random.sample(stocks, num_stocks)
        assert set(stocks).issubset(stock_prices.columns.tolist()), "Stocks must be in the data"

        # choose a random starting step
        if starting_step is None:
            starting_step = random.choice(stock_prices.index[: -(horizon + 1)])
        assert (
            starting_step in stock_prices.index[:-horizon]
        ), f"Starting step must be in the data at least {horizon + 1} days before the end of the data"
        current_prices = stock_prices.loc[starting_step, stocks]
        initial_portfolio = pd.Series(0, index=stocks)
        add = pd.Series(0, index=stocks)
        # create a random initial portfolio with a budget using the current prices
        while initial_portfolio @ current_prices < budget:
            add[random.choice(stocks)] += 1
            if add @ current_prices + initial_portfolio @ current_prices > budget:
                break
            initial_portfolio += add
            add = pd.Series(0, index=stocks)

        target_portfolio = pd.Series(0, index=stocks)
        add = pd.Series(0, index=stocks)
        # create a random initial portfolio with a budget using the current prices
        while target_portfolio @ current_prices < budget:
            add[random.choice(stocks)] += 1
            if add @ current_prices + target_portfolio @ current_prices > budget:
                break
            target_portfolio += add
            add = pd.Series(0, index=stocks)
        return ExperimentalConfig(
            budget=budget,
            stocks=stocks,
            trading_cost=trading_cost,
            horizon=horizon,
            starting_step=starting_step,
            initial_portfolio=initial_portfolio,
            target_portfolio=target_portfolio,
            data_file=data_file,
            model_directory=model_directory,
        )


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
    MODEL_DICT = {
        "GBM": GBM,
        "ABM": ABM,
        "NBEATS": NBEATSForecast
    }

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
        forecast_model: str
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
        assert num_stocks <= stock_prices.shape[1], "Number of stocks must be less than or equal to the number of stocks in the data"
        assert horizon <= stock_prices.shape[0], "Horizon must be less than or equal to the number of days in the data"
        assert date > pd.Timestamp("2019-01-01"), "Date must be after 2019-01-01"
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
        stock_prices = stock_prices[self.tickers]
        self.create_portfolios(stock_prices)

        self.forecast_model = self.MODEL_DICT[forecast_model](price_data = stock_prices[self.tickers], 
                                                              lookback=self.lookback, 
                                                              horizon=self.horizon)
        ## generate experiment id
        self.generate_exp_id()
        
    def choose_tickers(self, stock_prices):
        # subset the stock prices to a year before the start date and 90 days after the start date
        subset_prices = stock_prices.loc[self.start_date - pd.Timedelta(days=365) : self.start_date + pd.Timedelta(days=90)]
        # drop columns with nan or zero values
        subset_prices = subset_prices.dropna(axis=1)
        subset_prices = subset_prices.loc[:, (subset_prices != 0).any(axis=0)]
        tickers = subset_prices.columns.tolist()
        # choose a random number of stocks between 5 and 30
        rng = np.random.default_rng()
        self.tickers = rng.choice(tickers, size=self.num_stocks, replace=False)

    def create_portfolios(self):
        rng  = np.random.default_rng()
        # get stock prices on the start date for the chosen tickers
        self.initial_prices = self.stock_prices.loc[self.start_date]
        initial_portfolio = pd.Series(0, index=self.tickers)
        purchase = pd.Series(0, index=self.tickers)
        increment = int(self.budget / (10 * self.initial_prices.mean()))
        # create a random initial portfolio with a budget using the current prices
        while initial_portfolio @ self.initial_prices < self.budget:
            purchase[rng.choice(self.tickers)] += increment
            if purchase @ self.initial_prices + initial_portfolio @ self.initial_prices > self.budget:
                break
            initial_portfolio = initial_portfolio.add(purchase)
            purchase = pd.Series(0, index=self.tickers)
        self.initial_portfolio = initial_portfolio

        # create a random target portfolio with a budget using the current prices
        target = pd.Series(0, index=self.tickers)
        purchase = pd.Series(0, index=self.tickers)
        while target @ self.initial_prices < self.budget:
            purchase[rng.choice(self.tickers)] += increment
            if purchase @ self.initial_prices + target @ self.initial_prices > self.budget:
                break
            target += purchase
        self.target_portfolio = target

    def generate_exp_id(self):
        t= pd.to_datetime(str(self.start_date)) 
        timestring = t.strftime('%Y%m%d')
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
    lookback: int = 60
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
            temp = stock_price_df.loc['2019-01-01':]
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
                forecast_model=forecast_model
            )
            # Save the experiment to a file in the output directory
            
            with open(os.path.join(output_dir, f"{exp.exp_id}.pkl"), "wb") as f:
                pickle.dump(exp, f)
        except ExperimentGenerationError as e:
            pbar.set_description(f"Experiment {i} failed with error {e}.")

if __name__ == "__main__":
    generate_experiments(
        price_data_dir = './raw_data/spx_stock_prices.parquet',
        covariates = None,
        num_experiments = 10,
        output_dir = './experiments',
        min_horizon = 30,
        max_horizon = 90,
        max_num_stocks = 30,
        min_num_stocks = 5,
        min_tx_cost = 2,
        max_tx_cost = 10,
        max_budget = 350000,
        min_budget = 5000,
        forecast_model='GBM'
    )