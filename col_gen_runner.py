from experimental_config import *
import pickle
from objects import *

# read in pickle from small experiments
with open("small_experiments/exp_3_20506_20471_3.pkl", "rb") as f:
    small_experiments = pickle.load(f)

multi_sim = MultiSimRunner('small_experiments', [COL_GEN(True), COL_GEN(False), DIRECTIONAL_INCENTIVE_TRADING(0),
                                                 DIRECTIONAL_INCENTIVE_TRADING(.25), NAIVE, DIRECTIONAL_TRADING])
multi_sim.get_results("results/col_Gen.csv")