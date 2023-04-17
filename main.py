import pandas as pd
from objects import *
import pickle
import os
import matplotlib.pyplot as plt


# Write if name == main
if __name__ == "__main__":
    multi_sim = MultiSimRunner("experiments", ["Naive", "DayTrading"])
    multi_sim.get_results('naive_vs_daytrading_vs_directional.csv')
