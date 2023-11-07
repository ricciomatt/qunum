import numpy as np
import polars as pl
import pandas as pd

def rbst_sig(x, ax = 0):
    return 1.4826*np.nanmedian(np.abs(np.nanmedian(x, axis = ax) - x),axis=ax)

def rbst_cov(x, ax = 0):
    
    pass 

def rbst_sig_pl(df, col):
    return 1.4826*(df[col].median() - df[col]).abs().median()


