import numpy as np
import polars as pl
import pandas as pd
import scipy as sp
import scipy.stats as st
from typing import Callable
class QuantileSum:
    def __init__(self, quant:float):
        self.qaunt = quant
        return
    def __call__(self, x:np.array)->np.float32:
        return np.nanquantile(x, self.qaunt)

class Anderson:
    def __init__(self, dist:str = 'norm'):
        self.dist = dist
    
    def __call__(self, x:np.array)->np.float32:
        return sp.stats.anderson(x, dist=self.dist)[0]
    
def Sum_Sq(x:np.array)->np.float32:
    return np.nansum(np.power(x, 2))

def Fisher_Skew(x:np.array)->np.float32:
    return np.mean(((x - np.mean(x))/np.nanstd(x))**3)

def Kurtosis(x:np.array)->np.float32:
    return np.mean(((x - np.mean(x))/np.nanstd(x))**4)

def shapiro_wilke(x):
    return sp.stats.shapiro(x)[0]

def jaque(x):
    return sp.stats.jarque_bera(x)[0]

def quantile_objs(a):
    return QuantileSum(a)

class Entropy:
    def __init__(self, tp:Callable):
        self.density = st.kde
        self.entropy = tp
        return

    
vq_objs = np.vectorize(quantile_objs)
def summarize_pd(df:pd.DataFrame, group_col:str|list[str], val_cols:str|list[str])->pd.DataFrame:
    di_base = [np.nanmean, sp.stats.sem, 
               np.nanmedian,
               np.nanstd, rbst_sig,
               np.nanmax,
               QuantileSum(.95), QuantileSum(.75), QuantileSum(.25), QuantileSum(.05),
               np.nanmin, 
               np.nansum,
               
               Sum_Sq, Fisher_Skew, Kurtosis, sp.stats.skew, 
               
               shapiro_wilke, jaque,
               Anderson('norm'), Anderson('expon'), Anderson('logistic'),
               Anderson('gumbel'), Anderson('extreme1'),
               
               
               sp.stats.differential_entropy, sp.stats.entropy]
    cols_base = ['Mean', 'Mean_Error',
                 'Median',
                 'Sigma', 'Sigma_Tilde',
                 'Max', 'Q95', 'Q75', 'Q25', 'Q05', 'Min',
                 'Sum', 'Sum(x^2)', 
                 
                 'Fisher_Skew', 'Kurtosis', 'Skewness',
                 
                 'Shapiro_Wilk', 'Jarque_Bera',
                 'Anderson(norm)', 'Anderson(expon)', 
                 'Anderson(logistic)',
                 'Anderson(gumbel)', 'Anderson(exreme1)',
                 
                 'Differential_Entropy', 'Shannon_Entropy']
    cols = []
    for v in val_cols:
        for c in cols_base:
            cols.append(f"{v}_{c}")
    
    sdf = df.groupby(group_col)[val_cols].agg(di_base)
    sdf.columns = cols
    return sdf

def summarize_pl(df:pl.DataFrame, group_col:list[str], val_cols:list[str])->pl.DataFrame:
    pass



def rbst_sig(x, ax = 0):
    return 1.4826*np.nanmedian(np.abs(np.nanmedian(x, axis = ax) - x),axis=ax)

def rbst_cov(x, ax = 0):
    pass 

def rbst_sig_pl(df, col):
    return 1.4826*(df[col].median() - df[col]).abs().median()


