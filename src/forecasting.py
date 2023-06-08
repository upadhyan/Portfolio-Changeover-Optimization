import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class Forecast(ABC):
    def __init__(self, price_date: pd.DataFrame, lookback: int, horizon: int):
        self.price_data = price_date
        self.lookback = lookback
        self.horizon = horizon

    @abstractmethod
    def update(self):
        pass


class ABM(Forecast):
    """Arithmatic Brownian Motion"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, t, tickers):
        price_data = self.price_data[tickers]
        ret_data = self.price_data[tickers].pct_change().dropna()
        rng = np.random.default_rng()
        sample_size = 10

        idx = ret_data.index.get_indexer([t], method="pad")[0]
        end_dt = idx
        start_dt = max(idx - self.lookback, 0)
        assert start_dt >= 0, "start_dt must be greater than 0"

        price_observations = ret_data.iloc[start_dt:end_dt]
        # numpy objects
        mu = np.mean(price_observations, axis=0)
        cov = np.cov(price_observations, rowvar=False) * 1e2

        periods = ret_data.iloc[idx : idx + self.horizon].index

        # create planning matrix
        mvn = rng.multivariate_normal(mu, cov, size=(sample_size, self.horizon))
        self.mvn_avg = pd.DataFrame(index=periods, data=mvn.mean(axis=0), columns=ret_data.columns)

        print((self.mvn_avg + 1).cumprod())
        self.price_est = price_data.loc[t].multiply((self.mvn_avg + 1).cumprod())
