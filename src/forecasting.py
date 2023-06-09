import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import plotly.express as px


class Forecast(ABC):
    def __init__(self, price_date: pd.DataFrame, lookback: int, horizon: int):
        self.price_data = price_date
        self.lookback = lookback
        self.horizon = horizon

    @abstractmethod
    def update(self):
        pass


class ABM(Forecast):
    def __init__(self, boost_variance=1, **kwargs):
        self.boost_variance = boost_variance
        super().__init__(**kwargs)

    def update(self, t, tickers, simulations=1000):
        t = pd.to_datetime(t)
        rng = np.random.default_rng()

        price_data = self.price_data[tickers]
        price_change = price_data.diff().dropna()

        idx = price_change.index.get_indexer([t], method="pad")[0]
        end_dt = idx
        start_dt = max(idx - self.lookback, 0)
        assert start_dt >= 0, "start_dt must be greater than 0"

        price_observations = price_change.iloc[start_dt:end_dt]

        ## ABM params ##
        # numpy objects
        mu = np.mean(price_observations, axis=0) * self.horizon
        # cov = np.cov(price_observations, rowvar=False)
        # sigma = np.sqrt(np.diag(cov))
        sigma = np.std(price_observations, axis=0) * np.sqrt(self.horizon) * self.boost_variance

        # reshape mu and sigma based on number of simulations
        # mu = np.reshape(mu, (len(tickers), simulations))
        # sigma = np.reshape(sigma, (len(tickers), simulations))
        print(np.mean(price_observations, axis=0))

        T = 1  # Use this for scaling mu and sigma to horizon if different unit
        dt = T / self.horizon

        # time horizon
        periods = price_change.iloc[idx - 1 : idx + self.horizon].index

        p_list = []

        for ticker in tickers:
            S0 = price_data[ticker].iloc[idx]
            p_list.append(self.simulate(S0, mu[ticker], sigma[ticker], dt, self.horizon, simulations))

        self.price_est = pd.DataFrame(p_list).T
        self.price_est.columns = tickers
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

    def update(self, t, tickers, simulations=1000):
        t = pd.to_datetime(t)
        rng = np.random.default_rng()

        price_data = self.price_data[tickers]
        ret_data = self.price_data[tickers].pct_change().dropna()

        idx = ret_data.index.get_indexer([t], method="pad")[0]
        end_dt = idx
        start_dt = max(idx - self.lookback, 0)
        assert start_dt >= 0, "start_dt must be greater than 0"

        price_observations = ret_data.iloc[start_dt:end_dt]

        ## ABM params ##
        # numpy objects
        mu = np.mean(price_observations, axis=0) * self.horizon
        # cov = np.cov(price_observations, rowvar=False)
        # sigma = np.sqrt(np.diag(cov))
        sigma = np.std(price_observations, axis=0) * np.sqrt(self.horizon)

        # reshape mu and sigma based on number of simulations
        # mu = np.reshape(mu, (len(tickers), simulations))
        # sigma = np.reshape(sigma, (len(tickers), simulations))

        T = 1  # Use this for scaling mu and sigma to horizon if different unit
        dt = T / self.horizon

        # time horizon
        periods = ret_data.iloc[idx - 1 : idx + self.horizon].index

        p_list = []

        for ticker in tickers:
            S0 = price_data[ticker].iloc[idx]  # use ret_data if not using exp in simulate
            p_list.append(self.simulate(S0, mu[ticker], sigma[ticker], dt, self.horizon, simulations))

        self.price_est = pd.DataFrame(p_list).T
        self.price_est.columns = tickers
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