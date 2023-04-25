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
experiment_names = os.listdir("experiments")
experiments = []
# read in pickle object
for e in experiment_names:
    with open(f"experiments/{e}", "rb") as f:
        exp = pickle.load(f)
        experiments.append(exp)
exp = experiments[0]

# %%
# csmpo = RigidDayTrading(exp, verbose=False)
# market_sim = MarketSimulator(exp, csmpo)
# final_portf = market_sim.run()
# market_sim.plot_value()

# %%
# nmpo = NaivePolicy(exp, verbose=True)
# market_sim = MarketSimulator(exp, nmpo)
# final_portf = market_sim.run()
# market_sim.plot_value()

# %%
csmpo = DayTradingPolicy(exp, verbose=False)
day_trading_runner = MarketSimulator(exp, csmpo)
day_trading_runner.run()
day_trading_runner.plot_value()

# %%
multi_sim = MultiSimRunner(experiments[:6], ["RigidDayTrading", "DayTrading"])
multi_sim.run()

# %%
multi_sim.get_results()
