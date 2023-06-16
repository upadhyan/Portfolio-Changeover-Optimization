
from src.experimental_config import *
from src.policies import *
from src.simulator import *
from src.forecasting import *

import os


filename = '20220210_35_30_4_218480.pkl'
file_path = os.path.join('./experiments', filename)
with open(file_path, "rb") as f:
    exp = pickle.load(f)
# display the experiment id in the progress bar
print(f"Running {exp.exp_id}")
t1 = time()
policy = DirectionalTradingPolicy(exp, verbose=False)
simulator = MarketSimulator(exp, policy, verbose=False)

result = simulator.run()
