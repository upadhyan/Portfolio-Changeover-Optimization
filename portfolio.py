import numpy as np
import pandas as pd

class Portfolio:
    """Portfolio Object
    """
    
    def __init__(self, budget):
        self.value = budget

    def set_init_pos(self, weights, cash):
        self.w = np.concatenate([weights, cash])
    


    