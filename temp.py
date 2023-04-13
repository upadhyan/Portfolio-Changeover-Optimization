# %% [markdown]
# # Main Runner

# %%
import pandas as pd

from objects import *
import pickle
import os
import matplotlib.pyplot as plt

# %%
# Get all files in the experiments folder
experiments = os.listdir("experiments")

# read in pickle object
with open(f"experiments/{experiments[0]}", "rb") as f:
    exp = pickle.load(f)

# %%
csmpo = RigidDayTrading(exp, verbose=True)

# %%
market_sim = MarketSimulator(exp, csmpo)

nmpo = RigidDayTrading(exp, verbose=True)

# %%
market_sim = MarketSimulator(exp, nmpo)

# %%
final_portf = market_sim.run()

# %%
market_sim.plot_value()

# %%
csmpo = DayTradingPolicy(exp, verbose=False)
day_trading_runner = MarketSimulator(exp, csmpo)
day_trading_runner.run()
day_trading_runner.plot_value()
