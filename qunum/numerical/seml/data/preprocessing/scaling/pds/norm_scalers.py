import torch
"""try:
    import cupy as cp
except:
    import numpy as cp
"""
import numpy as cp
import numpy as np
import pandas as pd
import polars as pl
from imblearn.over_sampling import ADASYN,SMOTE,SMOTEN,SMOTENC, SVMSMOTE, KMeansSMOTE, RandomOverSampler


class NormScaler:
    def __init__(self, df:pd.DataFrame, x_cols:np.array, min_:int = 1e-6, max_:int = 1):
        self.x_cols = np.array(x_cols)
        self.min_ = min_
        self.max_ = max_
        self.scale_build(df)
        return
    
    def scale_build(self, df:pd.DataFrame)->None:
        """_summary_
        builds the min max scale args self.min/max_scle_facts
        and computes the max_kern_scale, and stats of dummified df
        Args:
            df (pd.DataFrame): _description_
        """        
        self.max_scle_facts = df[self.x_cols].max()
        self.min_scle_facts = df[self.x_cols].min()
        
        
        
        if(self.max_ is None):
            mn = np.abs(self.max_scle_facts.values)
            mx = np.abs(self.max_scle_facts.values)
            ix = np.where(mn != 0)
            self.max_= np.mean(mx[ix]/mn[ix])
        return
    
    def scale_do(self,cols:np.array= None, df:pd.DataFrame = None, x:np.array = None, 
                 h:np.array = None,
                 min_:float = None, max_:float=None, 
                kern:bool = False)->tuple[pd.DataFrame, np.array, np.array]:
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
        mn = self.min_scle_facts[cols].values
        mx = self.max_scle_facts[cols].values
        
        mn = mn.reshape((cols.shape[0], 1))
        mx = mx.reshape((cols.shape[0], 1))
        if(df is not None):
            vals = df[cols].values
            
            if(len(vals.shape)!=2):
                vals = vals.reshape(vals.shape[0], cols.shape[0])
            vals -= mn[:,0]
            vals += min_
            vals  *= max_/((mx[:,0]-mn[:,0])+min_)
            
            for i,c in enumerate(cols):
                df[c] = vals[:,i]
        if(x is not None):
            ix = []
            
            try:
                x = np.array(x)
            except:
                x = x.get()
            for c in cols:  
                ix.append(np.where(self.x_cols == c )[0][0])
            
            if(len(cols) != self.x_cols.shape[0]):
                x -= mn[:,0]
                x += min_
                x *= max_/((mx[0,:]-mn[:,0])+min_)
            else:
                x[:,ix] -= mn[:,0]
                x[:,ix] += min_
                x[:,ix] *= max_/((mx[0,:]-mn[:,0])+min_)
            
        if(h is not None):
            ix = []
            try:
                h = np.array(h)
            except:
                h = h.get()
            for c in cols:  
                ix.append(np.where(self.x_cols == c )[0][0])
            if(len(cols) != self.x_cols.shape[0]):
                h *= max_/((mx[0,ix]-mn[0,ix])+min_)
            else:
                h[ix] *= max_/((mx[0,ix]-mn[0,ix])+min_)
            
            
        return df, x, h 
    
    def scale_undo(self,cols:np.array= None, df:pd.DataFrame = None, x:np.array = None,
                   h:np.array = None, min_:float = None, max_:float=None, 
                   kern:bool = False)->tuple[pd.DataFrame, np.array, np.array]:
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
        mn = self.min_scle_facts[cols].values
        mx = self.max_scle_facts[cols].values
        mn = mn.reshape((cols.shape[0], 1))
        mx = mx.reshape((cols.shape[0], 1))
        
        if(df is not None):
            
            vals = df[cols].values
            if(len(vals.shape)!=2):
                vals = vals.reshape(vals.shape[0], cols.shape[0])
           
            vals  *= ((mx[0,:]-mn[:,0])+min_)/max_
            vals += mn[:,0]
            vals -= min_
            i = 0
            for c in cols:
                df[c] = vals[:,i]
                i+=1
        if(x is not None):
            ix = []    
            try:
                x = np.array(x)
            except:
                x = x.get()
            for c in cols:  
                ix.append(np.where(self.x_cols == c)[0][0])
            
            
            if(len(cols) != self.x_cols.shape[0]):
                x *= ((mx[0,:]-mn[:,0]) + min_)/max_
                x += mn[:,0]
                x -= min_
            else:
                x[:,ix] *= ((mx[0,:]-mn[:,0])+min_)/max_
                x[:,ix] += mn[:,0]
                x[:,ix] -= min_
            
            
        if(h is not None):
            ix = []
            try:
                h = np.array(h)
            except:
                h = h.get()
            for c in cols:  
                ix.append(np.where(self.x_cols == c )[0][0])
            if(len(cols) != self.x_cols.shape[0]):
                h *= ((mx[0,:]-mn[:,0])+min_)/max_
            else:
                h[ix] *= ((mx[0,:]-mn[:,0])+min_)/max_
            
        return df, x, h 
   
    