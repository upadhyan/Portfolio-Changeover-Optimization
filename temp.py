# %% [markdown]
# # Main Runner

# %%
from objects import *
import pickle
import os
import plotly.express as px

# %%
# Get all files in the experiments folder
experiment_names = os.listdir("experiments")
experiments = []
# read in pickle object
for e in experiment_names:
    with open(f"experiments/{e}", "rb") as f:
        exp = pickle.load(f)
        experiments.append(exp)

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
# policy = DirectionalPenaltyTradingPolicy(experiments[0], lambda_=0.5)
policy = ColumnGenerationPolicy(experiments[2])
simulator = MarketSimulator(experiments[2], policy, verbose=False)
simulator.run()
simulator.total_trades
print("")
# %%
# multi_sim = MultiSimRunner(experiments[:6], ["RigidDayTrading", "DayTrading"])
# multi_sim.run()

# # %%
# multi_sim.get_results()
