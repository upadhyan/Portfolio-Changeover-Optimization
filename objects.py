import numpy as np
import pandas as pd
import datetime as dt

import cvxpy as cvx
import gurobipy as gp
from gurobipy import GRB


class OnlineReturnsForecasts:
    def __init__(self, prices, trading_dates, samples=None):
        self.prices = prices
        self.ret = prices.pct_change().iloc[1:]
        self.trading_dates = trading_dates
        self.r_hat = None
        super().__init__()

    def get_prices(self, t, tau, assets):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        if t == tau:
            self._update(t, assets)
        # return self.r_hat.loc[tau].values.T @ wplus
        return self.p_hat.loc[tau].values

    def _update(self, t, assets=None):
        rng = np.random.default_rng()
        # samples = 1000
        if assets is not None:
            ret = self.ret[assets]
        else:
            ret = self.ret
        mean = ret.loc[:t].mean()
        cov = ret.loc[:t].cov()
        # r_hat = rng.multivariate_normal(mean = mean, cov = cov, size = (samples, 30)).mean(axis=0)
        r_hat = rng.multivariate_normal(mean=mean, cov=cov, size=len(self.trading_dates))
        self.r_hat = pd.DataFrame(data=r_hat, columns=ret.columns, index=self.trading_dates)
        self.p_hat = (self.r_hat + 1).cumprod().multiply(self.prices[assets].loc[t])


class CustomMultiPeriodOpt:
    def __init__(
        self,
        trading_times,
        terminal_weights,
        asset_prices,
        fixed_cost,
        return_forecast,
        lookahead_periods=None,
        *args,
        **kwargs
    ):
        """
        trading_times: list, all times at which get_trades will be called
        lookahead_periods: int or None. if None uses all remaining periods
        """
        # Number of periods to look ahead.
        self.lookahead_periods = lookahead_periods
        self.trading_times = trading_times
        self.terminal_weights = terminal_weights
        self.prices = asset_prices
        self.fixed_cost = fixed_cost
        self.return_forecast = return_forecast
        # super().__init__(*args, **kwargs)

    def get_trades(self, portfolio, t=dt.datetime.today()):
        ### GUROBI SETTINGS ###
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", False)
        # env.setParam("TimeLimit",time_limit)
        env.start()
        #######################

        assets = portfolio.index[:-1]
        value = self.prices[assets].loc[t] @ portfolio[assets].T + portfolio[-1]
        assert value > 0.0

        prob_arr = []
        z_arr = []
        cash_arr = []
        p_arr = []

        p = portfolio.values[:-1]  # portfolio of number of positions
        cash = portfolio.values[-1]  # cash amount

        F = np.ones(p.shape) * self.fixed_cost
        M = value * 1e3

        m = gp.Model(env=env)

        # planning_periods = self.lookahead_model.get_periods(t)
        for tau in self.trading_times[
            self.trading_times.index(t) : self.trading_times.index(t) + self.lookahead_periods
        ]:
            prices = self.return_forecast.get_prices(t, tau, assets)

            # Define in loop vars
            # Define Variables
            z_buy = m.addMVar(p.shape, vtype=GRB.INTEGER)
            z_sell = m.addMVar(p.shape, vtype=GRB.INTEGER)
            y_sell = m.addMVar(p.shape, vtype=GRB.BINARY)
            y_buy = m.addMVar(p.shape, vtype=GRB.BINARY)

            # print(prices @ z_sell)

            # This is the next iteration of portfolio
            p_next = p + z_buy - z_sell
            cash_next = prices @ (z_sell - z_buy) - F @ y_sell - F @ y_buy + cash

            ## Trading fees
            ### if we sell a stock, we pay a trading price
            m.addConstr(M * (y_sell) >= z_sell)

            ### If we buy a stock, we pay a trading fee
            m.addConstr(M * (y_buy) >= z_buy)

            # No borrowing
            m.addConstr(cash_next >= 0)

            # No shorting
            m.addConstr(p_next >= 0)

            prob_arr.append(prices @ p_next - F @ z_sell - F @ z_buy)
            cash = cash_next
            p = p_next
            # prob_arr.append(m)

            z_arr.append(z_buy - z_sell)
            cash_arr.append(cash_next)
            p_arr.append(p_next)

        # Terminal constraint.
        if self.terminal_weights is not None:
            # Terminal weights constraints
            m.addConstr(p_next >= self.terminal_weights.values)

        # Combine all time instances
        obj = sum(prob_arr)
        m.setObjective(obj, GRB.MAXIMIZE)
        m.optimize()

        try:
            self.p_vals = [p.getValue() for p in p_arr]
            self.cash_vals = [cash.getValue() for cash in cash_arr]
            self.z_vals = [z.getValue() for z in z_arr]
            print(f"Z:{self.z_vals[0]}")
            print(f"P:{self.p_vals[0]}")
        except Exception as e:
            print(e)

        assert (self.p_vals[0] >= 0).all()
        return pd.Series(index=portfolio.index, data=(np.append(self.z_vals[0], self.cash_vals[0])))


class MarketSimulator:
    def __init__(self, trading_times, asset_prices, policy):
        self.trading_times = trading_times
        self.asset_prices = asset_prices
        self.policy = policy

    def run(self, starting_portfolio):
        self.hist_trades = []
        portfolio = starting_portfolio.copy()
        print(f"SP: {portfolio.to_numpy()}")
        for t in self.trading_times:
            trades = self.policy.get_trades(portfolio, t)
            self.hist_trades.append(trades)
            portfolio = self._apply_trades(portfolio, trades)
        return portfolio

    def _apply_trades(self, portfolio, trades):
        portfolio[:-1] = portfolio[:-1] + trades[:-1]  # apply trades
        portfolio[-1] = trades[-1]  # update cash
        return portfolio
