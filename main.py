import pandas as pd
from objects import *
import pickle
import os
import matplotlib.pyplot as plt

# Get all files in the experiments folder
experiment_names = os.listdir('experiments')
experiments = []
# read in pickle object
for e in experiment_names:
    with open(f"experiments/{e}", 'rb') as f:
        exp = pickle.load(f)
        experiments.append(exp)

multi_sim = MultiSimRunner(experiments, ["Naive", "DayTrading"])
multi_sim.get_results('naive_vs_daytrading.csv')

