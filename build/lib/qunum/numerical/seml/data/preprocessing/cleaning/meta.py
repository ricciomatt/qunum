import polars as pl
import numpy as np
import torch
from typing import Iterable, Sequence
from ..scaling import NormScaler
class CleanerMeta:
    def __init__(self,
                 df:pl.DataFrame,
                 x_cols:np.ndarray|list|set|Sequence|Iterable, 
                 y_cols:np.ndarray|list|set|Sequence|Iterable,
                 th_drop:float = .4, 
                 tokenize_cols:np.ndarray|list|set|Sequence|Iterable|None = None,
                 tokenize:bool = True,
                 fill_method:str = 'dist_match',
                 
                 dummify_cols:np.ndarray|list|set|Sequence|Iterable|None = None,
                 dummify_sep:bool = True,
                 
                 comp_stats:bool = True,
                 scaler:NormScaler|None = None 
                 )->None:
        self.dummify_cols = dummify_cols
        self.dummify_sep = dummify_sep
        self.tokenize_cols  = tokenize_cols
        self.tokenize_do = tokenize
        self.x_cols = x_cols
        self.y_cols = y_cols
        if(scaler is None ):
            scaler = NormScaler(df, self.x_cols, max_ = None)
        self.Scaler = scaler
        
        pass