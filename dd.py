import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
t = 30
n = 3

buy_set = np.random.choice([0, 1], size=n)
relevant_matrices = [np.zeros((t,t)) if i == 0 else np.identity(t) for i in buy_set]
combinations = []
for i, vec in enumerate(buy_set):
    if vec != 0:
        asset_comb = [relevant_matrices[i][j] for j in range(t)]
    else:
        asset_comb = [np.zeros(t)]
    combinations.append(asset_comb)
iter_combinations = itertools.product(*combinations)
print("Buy Set: ", buy_set)
#print(len(list(iter_combinations)))

for c in tqdm(iter_combinations):
    print("---")
    print(np.array(c))

# for c in iter_combinations:
#     #print("---")
#     new_c = tuple(np.zeros(t) if type(item_) == np.float64 else item_ for item_ in c)
#     #print(pd.DataFrame(new_c))

#
# combinations = [c for c in itertools.product(c1, c2)]
# for c in combinations:
#     print(c)
