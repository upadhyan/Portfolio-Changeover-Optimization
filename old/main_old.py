# %%
import pandas as pd
import numpy as np

from objects import *
import cvxpy as cvx

import plotly.graph_objs as go
import plotly.express as px

import datetime as dt

# %%
start_date = "2012-01-01"
end_date = "2012-12-31"
prices = pd.read_parquet("spx_stock_prices.parquet").loc[start_date:end_date].replace(0.0, np.nan).dropna(axis=1)
returns = prices.pct_change().iloc[1:]


# %%
def generate_portfolios(stock_prices, portfolio_size, start_date, end_date, budget):
    rng = np.random.default_rng()
    # Select a random set of n assets
    asset_names = rng.choice(stock_prices.columns.tolist(), portfolio_size, replace=False)

    # Calculate maximum number of positions in each asset to respect budget
    max_positions = {}
    for asset in asset_names:
        max_positions[asset] = budget // stock_prices.loc[start_date, asset]

    # Select random number of positions in each asset within maximum limit
    def get_positions(t):
        positions = {}
        total_spent = 0
        for asset in asset_names:
            max_pos = max_positions[asset]
            positions[asset] = rng.integers(1, max_pos)
            total_spent += positions[asset] * stock_prices.loc[t, asset]

        # Adjust positions to respect budget
        while total_spent > budget:
            asset = rng.choice(asset_names)
            if positions[asset] > 0:
                positions[asset] -= 1
                total_spent -= stock_prices.loc[t, asset]

        return positions, total_spent

    # Calculate starting portfolio value
    positions, total_spent = get_positions(start_date)
    starting_portfolio = pd.Series(positions, name=start_date)
    starting_portfolio["cash"] = budget - total_spent
    starting_value = (
        starting_portfolio[asset_names] * stock_prices.loc[start_date, asset_names]
    ).sum() + starting_portfolio["cash"]

    # Calculate ending portfolio value
    positions, total_spent = get_positions(end_date)
    ending_portfolio = pd.Series(positions, name=end_date)
    ending_portfolio["cash"] = budget - total_spent
    ending_value = (ending_portfolio[asset_names] * stock_prices.loc[end_date, asset_names]).sum() + ending_portfolio[
        "cash"
    ]

    return starting_portfolio, ending_portfolio, starting_value, ending_value


# %%
trading_period = 42
sim_st_dt = dt.datetime.strptime("2012-01-01", "%Y-%m-%d") + dt.timedelta(days=30)
sim_end_dt = sim_st_dt + dt.timedelta(days=trading_period)
trading_times = returns[sim_st_dt:sim_end_dt].index.to_list()

# %%
start_portf, end_portf, start_val, end_val = generate_portfolios(prices, 10, sim_st_dt, sim_end_dt, 10000)

# %%
params = dict(
    lookahead_periods=5,
    trading_times=trading_times,
    terminal_weights=end_portf[:-1],
    asset_prices=prices,
    return_forecast=OnlineReturnsForecasts(prices, trading_dates=trading_times),
    fixed_cost=5.99,
    costs=[],
    constraints=[],
    solver=cvx.GUROBI,
    solver_opts={"verbose": True},
)

# %%
# risk_model = cp.EmpSigma(returns=returns, lookback=252)
# gamma_risk = 5
# gamma_trade = 1
# gamma_hold = 1

mpo_policy = CustomMultiPeriodOpt(**params)

market_sim = MarketSimulator(
    trading_times=trading_times,
    asset_prices=prices,
    policy=mpo_policy,
)

final_portf = market_sim.run(start_portf)




# market_sim = cp.MarketSimulator(
#     returns,
#     costs=[cp.FixedTxModel(cost=5.99, prices=prices)],
# )

# results = market_sim.run_backtest(
#     initial_portfolio=start_portf,
#     start_time=sim_st_dt,
#     end_time=sim_end_dt,
#     policy=mpo_policy,
# )

# results.summary()
