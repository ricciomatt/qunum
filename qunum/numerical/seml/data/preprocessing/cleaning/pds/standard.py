import torch
try:
    #import cupy as cp
    import numpy as np
except:
    import numpy as cp
import numpy as np
import pandas as pd
import polars as pl
from ...scaling import NormScaler
from .....stats.stats_rbst import rbst_sig
from ......numerics.grid_space import ord_to_grid
from .....kernals import Kernal
from ...scaling import NormScaler

class PdDataCleaner:
    def __init__(self,
                 df:pd.DataFrame,
                 x_cols:np.array, 
                 y_cols:np.array,
                 th_drop:float = .4, 
                 tokenize_cols:np.array = None,
                 tokenize:bool = True,
                 fill_method:str = 'dist_match',
                 
                 dummify_cols:np.array = None,
                 dummify_sep:bool = True,
                 
                 comp_stats:bool = True,
                 scaler:object = None 
                 ):
        self.df_og = df.copy(deep = True)
        self.dummify_cols = dummify_cols
        self.dummify_sep = dummify_sep
        self.tokenize_cols  = tokenize_cols
        self.tokenize_do = tokenize
        self.x_cols = x_cols
        self.y_cols = y_cols
        if(scaler is None ):
            scaler = NormScaler(df, self.x_cols, max_ = None)
        self.Scaler = scaler
        return
    
    def filter_data(self,):
        pass 
    
    def clean_data(self)->None:
        df = self.df_og.copy(deep=True)
        
        if(self.tokenize_do):
            self.tokenize_data(self.tokenize_cols, df)
        
        df_og_dummified, fill_cols = self.dummify_and_drop(df)
        self.df_og = df_og_dummified
        if(fill_cols != []):
            self.df_filled = self.fill_nas(self.df_og, fill_cols=fill_cols)  
        else:
            self.df_filled = self.df_og.copy(deep = True)
        print(f'Cleaned & Filled Data')  
        return 
    
    def tokenize_data(self):
        import nltk.tokenize as tokenizer
        
        pass 
    
    def dummify_and_drop(self,df:pd.DataFrame)->tuple[pd.DataFrame, np.array]:
        """_summary_

        Args:
            df (pd.DataFrame): _description_

        Returns:
            tuple[pd.DataFrame, np.array]: _description_
        """                
        
        x_cols = []
        for x in self.dummify_cols:
            cond = (df[x].isna()) | (df[x].isnull())
            tdf = df[cond == False]
            U = tdf.unique()
            t = df[x].values
            ix_null = np.where(cond.values)[0][0]
            rg = np.random.choice(U,(ix_null.shape[0]))
            t[ix_null] = rg
            df[x] = t
            if(self.dummify_sep):
                for u in range(U):
                    t = np.zeros((df.values.shape[0],), dtype=np.float64)
                    t = np.where((df[x] == u).values)[0] = 1.0
                    df[f"{x}-{u}"] = t
                    x_cols.append(f'{x}-{u}')
            else:
                ct = 0.0
                t = np.zeros((df.values.shape[0],), dtype=np.float64)
                for u in range(U):
                    t = np.where((df[x] == u).values)[0] = ct
                    
                    ct+=1.0
                df[f"dummy_{x}"] = t
                x_cols.append(f'dummy-{x}')
        fill_cols = []
        num_tot = df.shape[0]
        for x in self.x_cols:

            tdf = df[((df[x].isna()==False) | (df[x].isnull() == False))]
            num_nna = tdf.values.shape[0]
            if(num_nna/num_tot > self.th_drop and x not in list(self.dummify_cols)):
                x_cols.append(x)
            if(num_nna != num_tot):
                fill_cols.append(x)
        self.x_cols = np.copy(np.array(x_cols))
        return df, fill_cols
    
    def fill_nas(self, df:pd.DataFrame, fill_cols:np.array = None)->pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            fill_cols (np.array, optional): _description_. Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """        
        if(fill_cols is None):
            fill_cols = self.x_cols.copy()
        self.Kern = Kernal(df, x_cols=self.x_cols, compute_h= False ,compute_cov=False, scaler = self.Scaler, df_scaled=False, do_scale=True)
        for f in fill_cols:
            t = self.Kern.df[f].values
            cond = self.Kern.df[f].isna()
            use_ix = np.where((cond == False).values)[0]
            nuse_ix = np.where((cond).values)[0]
            n = self.Kern.df[cond].shape[0]
            
            if(self.na_fill == 'dist_match'):
                RX = self.Kern.random_sample(df, cols = [f], size=(n,1), use_ix=use_ix, scale_back=False,)
                vals = RX[f][:,0]
            elif(self.na_fill == 'norm_dist_rbst'):
                md = np.nanmedian(self.Kern.df[f].values )
                sig = 1.4826*(np.nanmedian(np.abs(md- self.Kern.df[f].values)))
                vals = np.random.normal(md,sig,size = use_ix.shape[0])
                
            elif(self.na_fill == 'norm_dist'):
                
                md = np.nanmean(self.Kern.df[f].values )
                sig = np.nanstd(self.Kern.df[f].values)
                vals = np.random.normal(md,sig,size = use_ix.shape[0])
            elif(self.na_fill=='median'):
                md = np.nanmedian(self.Kern.df[f].values )
                vals = np.empty(use_ix.shape[0])
                vals[:] = md
                
            elif(self.na_fill=='mean'):
                md = np.nanmean(self.Kern.df[f].values )
                vals = np.empty(use_ix.shape[0])
                vals[:] = md
            else:
                md = np.nanmedian(self.Kern.df[f].values )
                vals = np.empty(use_ix.shape[0])
                vals[:] = md
            t[nuse_ix] = vals
            self.Kern.df[f] = t
        df = self.Scaler.scale_undo(self.x_cols, df = self.Kern.df,)
        return df[0]

    def test_train_split(self, train_pct:float=.75, under_sample:bool = False, over_sample:bool = False, min_:float=0.0, max_:float=1.0)->tuple[np.array, np.array, np.array, np.array]:
        pass 
    
    def test_train_split_oversample_syn(self, train_pct:float=.75, under_sample:bool = False, over_sample:bool = False, min_:float=0.0, max_:float=1.0, sampler = None)->tuple[np.array, np.array, np.array, np.array]:
        from imblearn.over_sampling import ADASYN,SMOTE,SMOTEN,SMOTENC, SVMSMOTE, KMeansSMOTE, RandomOverSampler
        
        pass 
    
    def test_train_split_undersample_syn(self, train_pct:float=.75, under_sample:bool = False, over_sample:bool = False, min_:float=0.0, max_:float=1.0, sampler = None)->tuple[np.array, np.array, np.array, np.array]:
        
        from imblearn.over_sampling import ADASYN,SMOTE,SMOTEN,SMOTENC, SVMSMOTE, KMeansSMOTE, RandomOverSampler
        import nltk.tokenize as tokenizer
        if(sampler is None):
            sampler = SMOTE
        
        pass 
    
