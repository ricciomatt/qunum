"""try:
    import cupy as cp 
except:
    import numpy as cp
"""
import numpy as cp
import pandas as pd
import torch
from typing import Callable
def ord_to_grid(x):
    X_Grid= cp.array(cp.meshgrid(*cp.array(x)))
    X_Grid = X_Grid.T.reshape(-1, x.shape[0])
    return X_Grid

def construct_cont_gen(df:pd.DataFrame, cols, num_cont_pts:int=10, n_sig:int= 2)->tuple[cp.array, cp.array]:
    x_ = cp.empty((cols.shape[0], num_cont_pts))
    for i in range(cols.shape[0]):
        tdf = df[(df[cols[i]].isna() == False) & (df[cols[i]].isnull() == False)]
        v = tdf[cols[i]].values
        sig = cp.std(v)
        if(sig == 0):
            sig = 1e-5
        mn = cp.min(v) - n_sig * cp.abs(sig)
        mx = cp.max(v) + n_sig * cp.abs(sig)
        x_[i] = cp.linspace(mn, mx, num_cont_pts)
    delta = x_[:, 1] - x_[:, 0]
    return x_, delta


