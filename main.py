from objects import *
import pickle
import os
import plotly.express as px

# Write if name == main
if __name__ == "__main__":
    # Get all files in the experiments folder
    experiment_names = os.listdir('experiments')
    experiments = []
    # read in pickle object
    for e in experiment_names:
        with open(f"experiments/{e}", 'rb') as f:
            exp = pickle.load(f)
            experiments.append(exp)
    exp = experiments[0]
    policy = RigidDirectionalIncentivePolicy(exp, lambda_=0.5)
    simulator = MarketSimulator(exp, policy)
    res = simulator.run()
    print(simulator.status)
    print(simulator)