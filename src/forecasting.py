import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import plotly.express as px

from neuralforecast import NeuralForecast

import sys
import os


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Forecast(ABC):
    def __init__(self, price_data: pd.DataFrame, lookback: int, horizon: int):
        self.price_data = price_data
        self.lookback = lookback

    @abstractmethod
    def update(self, t, horizon, **kwargs):
        pass

class NBEATSForecast(Forecast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.price_data = self.melt_data(self.price_data)
        self.model = None

    def update(self, t, horizon):
        with HiddenPrints():
            if self.model is None:
                self.model = NeuralForecast.load(path='./models/')
            price_data = self.price_data[self.price_data['ds'] <= t]
            fcst = self.model.predict(df=price_data).reset_index()
            fcst = fcst.rename(columns={'PatchTST': 'yhat_patchtst', 'NBEATSx': 'yhat', 'NHITS': 'yhat_nhits'})
            current_day = self.price_data[self.price_data['ds'] == t].copy()
            current_day = current_day.rename(columns={'y': 'yhat'})
            # stack current day with forecast
            fcst = pd.concat([current_day, fcst], axis=0)
            fcst = self.unmelt_data(fcst)
            # re-index this to ensure the index is continuous with respect to business days
            fcst = fcst.reindex(pd.date_range(start=fcst.index[0], end=fcst.index[-1], freq='B'))
            fcst = fcst.fillna(method='ffill')
            # get the first n+1 rows
            fcst = fcst.iloc[:horizon+1]
            # sort the columns alphabetically
            tickers = fcst.columns
            tickers.sort_values(ascending=True)
            fcst = fcst[tickers]
        return fcst

    def melt_data(self, price_data):
        price_data.index.name = 'ds'
        # Convert from wide to long format
        prices = price_data.reset_index()
        melted_df = pd.melt(prices, id_vars=['ds'], value_vars=prices.columns[1:], var_name='unique_id', value_name='y')
        melted_df['ds'] = pd.to_datetime(melted_df['ds'])
        melted_df = melted_df.sort_values(['unique_id', 'ds'])
        melted_df = melted_df.reset_index(drop=True)
        melted_df = melted_df.dropna()
        return melted_df 

    def unmelt_data(self, price_data):
        return price_data.pivot(index='ds', columns='unique_id', values='yhat')

    

class ABM(Forecast):
    def __init__(self, boost_variance=1, **kwargs):
        self.boost_variance = boost_variance
        super().__init__(**kwargs)

    def update(self, t, horizon, simulations=1000):
        t = pd.to_datetime(t)
        rng = np.random.default_rng()

        price_data = self.price_data
        price_change = price_data.diff().dropna()

        idx = price_change.index.get_indexer([t], method="pad")[0]
        end_dt = idx
        start_dt = max(idx - self.lookback, 0)
        assert start_dt >= 0, "start_dt must be greater than 0"

        price_observations = price_change.iloc[start_dt:end_dt]

        ## ABM params ##
        # numpy objects
        mu = np.mean(price_observations, axis=0) * horizon
        # cov = np.cov(price_observations, rowvar=False)
        # sigma = np.sqrt(np.diag(cov))
        sigma = np.std(price_observations, axis=0) * np.sqrt(horizon) * self.boost_variance

        # reshape mu and sigma based on number of simulations
        # mu = np.reshape(mu, (len(tickers), simulations))
        # sigma = np.reshape(sigma, (len(tickers), simulations))
        print(np.mean(price_observations, axis=0))

        T = 1  # Use this for scaling mu and sigma to horizon if different unit
        dt = T / horizon

        # time horizon
        periods = price_change.iloc[idx - 1 : idx + horizon].index

        p_list = []

        for ticker in price_data.columns:
            S0 = price_data[ticker].iloc[idx]
            p_list.append(self.simulate(S0, mu[ticker], sigma[ticker], dt, horizon, simulations))

        self.price_est = pd.DataFrame(p_list).T
        self.price_est.columns = price_data.columns
        self.price_est.index = periods
        return self.price_est

    def simulate(self, S0, mu, sigma, dt, num_steps, simulations):
        # TODO: This can be simplified by taking cumsum of all random numbers
        # generate random numbers
        rng = np.random.default_rng()
        Z = rng.standard_normal(size=(num_steps, simulations))  # Z ~ N(0,1)
        # Z2 = Z * sigma.loc[ticker] + mu.loc[ticker]
        S = np.zeros(shape=(num_steps + 1, simulations))
        S[0] = S0 * np.ones(simulations)

        print(mu * dt, sigma * np.sqrt(dt))

        S[1:, :] = S[0] + np.cumsum(
            mu * dt + sigma * np.sqrt(dt) * Z, axis=0
        )  # Use this if mu and sigma are scaled to dt
        # S[1:,:] = S[0] + np.cumsum(mu + sigma * Z, axis=0)

        return np.mean(S, axis=1)


class GBM(Forecast):
    """Geometric Brownian Motion"""

    def __init__(self, boost_variance=1, **kwargs):
        self.boost_variance = boost_variance
        super().__init__(**kwargs)

    def update(self, t, horizon, simulations=1000):
        """Update the forecasted price

        Args:
            t (datetime): current trading day
            tickers (list): list of tickers to forecast
            simulations (int, optional): Number of simulated random walks to generate forecast. Defaults to 1000.

        Returns:
            DataFrame: DataFrame of forecasted prices
        """
        t = pd.to_datetime(t)
        rng = np.random.default_rng()

        price_data = self.price_data
        ret_data = self.price_data.pct_change().dropna()

        idx = ret_data.index.get_indexer([t], method="pad")[0]
        end_dt = idx
        start_dt = max(idx - self.lookback, 0)
        assert start_dt >= 0, "start_dt must be greater than 0"

        price_observations = ret_data.iloc[start_dt:end_dt]

        ## ABM params ##
        # numpy objects
        mu = np.mean(price_observations, axis=0) * horizon
        # cov = np.cov(price_observations, rowvar=False)
        # sigma = np.sqrt(np.diag(cov))
        sigma = np.std(price_observations, axis=0) * np.sqrt(horizon)

        # reshape mu and sigma based on number of simulations
        # mu = np.reshape(mu, (len(tickers), simulations))
        # sigma = np.reshape(sigma, (len(tickers), simulations))

        T = 1  # Use this for scaling mu and sigma to horizon if different unit
        dt = T / horizon

        # time horizon
        periods = ret_data.iloc[idx - 1 : idx + horizon].index

        p_list = []

        for ticker in price_data.columns:
            S0 = price_data[ticker].iloc[idx]  # use ret_data if not using exp in simulate
            p_list.append(self.simulate(S0, mu[ticker], sigma[ticker], dt, horizon, simulations))

        self.price_est = pd.DataFrame(p_list).T
        self.price_est.columns = price_data.columns
        self.price_est.index = periods
        return self.price_est

    def simulate(self, S0, mu, sigma, dt, num_steps, simulations):
        # generate random numbers
        rng = np.random.default_rng()
        Z = rng.standard_normal(size=(num_steps, simulations))  # Z ~ N(0,1)
        # Z2 = Z * sigma.loc[ticker] + mu.loc[ticker]
        S = np.zeros(shape=(num_steps + 1, simulations))
        S[0] = S0 * np.ones(simulations)

        print(mu * dt, sigma * np.sqrt(dt))

        S[1:, :] = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        S = S.cumprod(axis=0)

        return np.mean(S, axis=1)
