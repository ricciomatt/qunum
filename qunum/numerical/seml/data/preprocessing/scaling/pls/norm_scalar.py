import numpy as np
import torch
import polars as pl
from typing import Iterable, Sequence
class NormScaler:
    def __init__(self, df:pl.DataFrame, x_cols:np.ndarray|pl.Expr|Iterable|Sequence|list=None, min_:int = 1e-6, max_:int = 1, **kwargs):
        if('cols' in kwargs and x_cols is None):
            x_cols = kwargs['cols']
        elif(x_cols is None):
            raise RuntimeError('Must Provide Columns')
        if('min_' in kwargs):
            min_ = float(kwargs['min_'])
        if('max_' in kwargs):
            max_ = float(kwargs['max_'])
        self.x_cols = np.array(x_cols)
        self.min_ = min_
        self.max_ = max_
        self.scale_build(df)
        return
    
    def scale_build(self, df:pl.DataFrame)->None:
        """_summary_
        builds the min max scale args self.min/max_scle_facts
        and computes the max_kern_scale, and stats of dummified df
        Args:
            df (pd.DataFrame): _description_
        """        
        self.max_scle_facts = df[self.x_cols].max()
        self.min_scle_facts = df[self.x_cols].min() 
        if(self.max_ is None):
            mn = np.abs(self.max_scle_facts.to_numpy())
            mx = np.abs(self.max_scle_facts.to_numpy())
            ix = np.where(mn != 0)
            self.max_= np.mean(mx[ix]/mn[ix])
        return
    
    def scale_do(self,cols:np.array= None, df:pl.DataFrame = None, x:np.array = None, 
                   h:np.array = None, min_:float = None, max_:float=None, 
                   kern:bool = False)->tuple[pl.DataFrame, np.ndarray|torch.Tensor, np.ndarray|torch.Tensor]:
        """_summary_

        Args:
            cols (np.array, optional): _description_. Defaults to None.
            df (pd.DataFrame, optional): _description_. Defaults to None.
            x (np.array, optional): _description_. Defaults to None.
            h (np.array, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if(min_ is None):
            min_ = self.min_
            
        if(max_ is None):
            max_ = self.max_
            
        if(cols is None):
            cols = self.x_cols
        cols = np.array(cols)
        mn = self.min_scle_facts.to_numpy()[0]
        mx = self.max_scle_facts.to_numpy()[0]
        m = max_/((mx-mn)+min_)
        a = -mn +min_
        return self.do(m, a, cols, df, x, h) 
    
    
    def scale_undo(self,cols:np.array= None, df:pl.DataFrame = None, x:np.array = None,
                   h:np.array = None, min_:float = None, max_:float=None, 
                   kern:bool = False)->tuple[pl.DataFrame, np.ndarray, np.ndarray]:
        """_summary_

        Args:
            cols (np.array, optional): _description_. Defaults to None.
            df (pd.DataFrame, optional): _description_. Defaults to None.
            x (np.array, optional): _description_. Defaults to None.
            h (np.array, optional): _description_. Defaults to None.
            min_ (_type_, optional): _description_. Defaults to None.
            max_ (_type_, optional): _description_. Defaults to None.
            kern (bool, optional): Defaults to False

        Returns:
            _type_: _description_
        """
        if(min_ is None):
            min_ = self.min_
            
        if(max_ is None):
            max_ = self.max_
            
        if(cols is None):
            cols = self.x_cols
        cols = np.array(cols)
        mn = self.min_scle_facts.to_numpy()[0]
        mx = self.max_scle_facts.to_numpy()[0]
        m = ((mx-mn)+min_)/max_
        a = mn -min_
        return self.do(m, a, cols, df, x, h)
    
    def do(self, m:np.ndarray, a:np.ndarray, cols:np.ndarray, 
           df:pl.DataFrame, x:np.ndarray|torch.Tensor, h:np.ndarray|torch.Tensor)->tuple[pl.DataFrame, np.ndarray|torch.Tensor, np.ndarray|torch.Tensor]:
        if(df is not None):
            df = df.with_columns(
                list(
                    map(
                        lambda x: (
                            (
                                pl.col(cols[x])+a[x]
                            )*(m[x]
                            )
                        ), 
                        np.arange(cols.shape[0])
                    )
                )
            )
        
        if(x is not None):
            ix = []
            if(isinstance(x, torch.Tensor)):
                m = torch.tensor(m)
                a = torch.tensor(a)
            else:
                m = np.array(m)
                a = np.array(a)
            for c in cols:  
                ix.append(np.where(self.x_cols == c )[0][0])
            
            if(len(cols) != self.x_cols.shape[0]):
                x = (x+a[ix])*m[ix]
            else:
                x[:, ix] = (x[:,ix] + a)*(m)
            
        if(h is not None):
            ix = []
            if(isinstance(h, torch.Tensor)):
                m = torch.tensor(m)
                a = torch.tensor(a)
            else:
                m = np.array(m)
                a = np.array(a)
            for c in cols:  
                ix.append(np.where(self.x_cols == c )[0][0])
            if(len(cols) != self.x_cols.shape[0]):
                h = (h+a[ix])*m[ix]
            else:
                h[ix] = (h[ix] + a)*(m)
            
        return df, x, h 
   