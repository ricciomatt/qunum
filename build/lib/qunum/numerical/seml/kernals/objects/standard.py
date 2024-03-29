import torch
'''try:
    import cupy as cp
    from cupy.typing import NDArray as CPArray
except:
    import numpy as cp
    from numpy.typing import NDArray as CPArray
'''

import numpy as cp
from numpy.typing import NDArray as CPArray
import numpy as np
import pandas as pd
import polars as pl
from imblearn.over_sampling import ADASYN,SMOTE,SMOTEN,SMOTENC, SVMSMOTE, KMeansSMOTE, RandomOverSampler

from ..functions import gauss_functs
from ...fit import grad_descent
from ...nn import optimizers as optimizer


from ....mathematics.numerics.grid_space import ord_to_grid, construct_cont_gen
import nltk.tokenize as tokenizer
from ...data.preprocessing.scaling import NormScaler
from ....mathematics.numerics import integrators_

from typing import Callable
from numpy.typing import NDArray


def rbst_sig(x, ax = 0):
    return 1.4826*np.nanmedian(np.abs(np.nanmedian(x, axis = ax) - x),axis=ax)

class Kernal:
    def __init__(self,
                 df:pd.DataFrame = None, 
                 x_cols:NDArray = None, 
                 y_cols:NDArray = None,
                 x:NDArray = None,
                 y:NDArray = None,
                 compute_cov:bool = True,
                 h:NDArray = None, 
                 h_estimation_method:str = 'silvermans_rbst',
                 h_fact:float = 1.0,
                 th_non_full:int = int(1e5),
                 num_cont_pts:int = int(1e3),
                 nint:int = 10,
                 kernal_funct:object = gauss_functs.partition,
                 kernal_pdf:object = gauss_functs.pdf,
                 kernal_fisher:object = gauss_functs.fisher_torch,
                 do_scale:bool = True,
                 scaler:object = None,
                 df_scaled:bool = False,
                 compute_h:bool = True,
                 )->object:
        """_summary_

        Args:
            df (pd.DataFrame, optional): _description_. Defaults to None.
            x_cols (NDArray, optional): _description_. Defaults to None.
            y_cols (NDArray, optional): _description_. Defaults to None.
            x (NDArray, optional): _description_. Defaults to None.
            y (NDArray, optional): _description_. Defaults to None.
            compute_cov (bool, optional): _description_. Defaults to True.
            h (NDArray, optional): _description_. Defaults to None.
            h_estimation_method (str, optional): _description_. Defaults to 'silvermans_rbst'.
            h_fact (float, optional): _description_. Defaults to 1.0.
            th_non_full (int, optional): _description_. Defaults to int(1e5).
            num_cont_pts (int, optional): _description_. Defaults to int(1e3).
            nint (int, optional): _description_. Defaults to 10.
            kernal_funct (object, optional): _description_. Defaults to gauss_functs.partition.
            kernal_pdf (object, optional): _description_. Defaults to gauss_functs.pdf.
            kernal_fisher (object, optional): _description_. Defaults to gauss_functs.fisher_torch.
            do_scale (bool, optional): _description_. Defaults to True.
            scaler (object, optional): _description_. Defaults to None.
            df_scaled (bool, optional): _description_. Defaults to False.
            compute_h (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: _description_

        Returns:
            object: _description_
        """        
        if(x is not None and y is not None):
            x_cols = v_str(np.arange(x.shape[1]))
            y_cols = v_str(np.arange(y.shape[1]), st='y')
            cols = np.concatenate((x_cols,y_cols), axis = 0)
            vals = np.concatenate((x,y), axis = 1)
            df = pd.DataFrame(data=vals, columns=cols)
        elif(x_cols is not None and y_cols is not None and df is not None):
            pass 
        else:
            raise ValueError(f'''
                             Need To Pass Data Into Kernal, Either as a DataFrame with x and y cols specified or x and y arrays
                             IE. Kernal(df = pd.DataFrame(data=data, columns=['a', 'b', 'c']), x_cols = ['a','b'], y_cols = 'c')
                             or Kenral(x=np.array(x_vals), y = np.array(y_vals))
                             ''')
        df = df.copy(deep=True)
        self.x_cols = np.array(x_cols)
        self.y_cols = np.array(y_cols)
        if(scaler is None):
            scaler = NormScaler(df, x_cols, min_ = 1e-8, max_ = None)
        self.Scaler = scaler
        self.do_scaler = do_scale
        if(not df_scaled and do_scale):
            df, temp, temp1 = scaler.scale_do(x_cols, df)
        self.df = df.copy(deep = True)
        
        self.num_cont_pts = num_cont_pts
        self.nint = nint
        self.th_non_full = th_non_full
        
        self.build_kern(
            h = h,
            h_estimation_method=h_estimation_method,
            h_fact=h_fact,
            compute_cov=compute_cov,
            kernal_fisher=kernal_fisher, 
            kernal_funct=kernal_funct, 
            kernal_pdf=kernal_pdf,
            compute_h= compute_h
            )
        return
    
    def build_kern(self,
                   df:pd.DataFrame = None,
                   h:CPArray = None,
                   h_estimation_method:str = 'silvermans_rbst',
                   h_fact:float = 1.0,
                   kernal_funct:object = gauss_functs.partition,
                   kernal_pdf:object = gauss_functs.pdf,
                   kernal_fisher :object= gauss_functs.fisher_torch,
                   compute_cov:bool = True,
                   compute_h: bool = True,
                   scale:bool = True
        )->None:
        
        if(df is None):
            df = self.df.copy(deep=True)
        else:
            if(scale):
                df,m,n = self.Scaler.scale_do(self.x_cols, df,)
        
        self.kern_scaled = scale
        self.kernal_pdf_funct = kernal_pdf
        self.kernal_prd_funct = kernal_funct
        self.kernal_fsh_funct = kernal_fisher               
        
        N = self.df.shape[0]
        d = self.x_cols.shape[0]
        fact = ((N * (d + 2)) / 4) ** (-1 / (d + 4))*h_fact
        if(h is None and compute_h):
            if(h_estimation_method == 'silvermans_rbst'):
                std = 1.4826*(df[self.x_cols].median() - df[self.x_cols]).abs().median()
                
            elif(h_estimation_method == 'silvermans'):
                std = df[self.x_cols].std().values
            else:
                pass 
            self.h = cp.array(std*fact)+1e-6
        else:
            self.h = h
        if(compute_cov):
            self.cov_ij = (cp.cov(cp.array(self.df[self.x_cols].values.transpose())))
            cols = list(self.y_cols.copy())
            cols.extend(self.x_cols)
            self.cov_y_xij = cp.cov(cp.array(self.df[cols].values.transpose()))[:self.y_cols.shape[0],self.y_cols.shape[0]:]
        self.kern_built = True
        return
    
    def df_stats(self,df:pd.DataFrame)->dict:
        """_summary_
        computes stats of input df

        Args:
            df (pd.DataFrame): _description_

        Returns:
            dict: _description_
        """        
        return {
            'min':df[self.x_cols].min(), 
            'max':df[self.x_cols].max(), 
            'mean':df[self.x_cols].mean(),
            'median':df[self.x_cols].median(),
            'sig':df[self.x_cols].std(),
            'rbst_sig':1.5*(df[self.x_cols].median() - df[self.x_cols]).abs().median(),
            'sum_of_squares':df[self.x_cols].pow(2).sum(),
            'sum':df[self.x_cols].sum(),
            'mode':df[self.x_cols].mode(),
            'PCT25':df[self.x_cols].quantile(.25),
            'PCT75':df[self.x_cols].mode(.75),
            'PCT95':df[self.x_cols].mode(.95),
            'PCT05':df[self.x_cols].mode(.05),
            }
       
    def random_sample(self, df:pd.DataFrame=None, cols:NDArray = None, size:tuple = (100, 1), use_ix:NDArray=None,
                    scale_back:bool = True, edge_tuning_scale:float = 10.0, num_cont_pts:int=int(5e3),
                    h:NDArray =None, )->dict[NDArray]:
        RX = {}
        if(df is None):
            df = self.df.copy()
        if(h is None):
            h = self.h
        
        if(use_ix is not None):
            df = df.loc[use_ix]
        N = df.shape[0]
        d = self.x_cols.shape[0]
        for c in cols:
            X = df[cols].values
            X = X.reshape((X.shape[0],1))
            
            if(h is None):
                std = rbst_sig(df[c].values)
                th = cp.array([std* ((N * (d + 2)) / 4) ** (-1 / (d + 4))])
            else:
                ix = np.where(self.x_cols == c)[0]
                th = h[ix]
            
            x, delta = construct_cont_gen(df=df,
                                          cols=np.array([cols]),
                                          num_cont_pts=num_cont_pts,)
            x = cp.array(ord_to_grid(x))
            pdf = gauss_functs.pdf(x,X,th,W = np.ones((X.shape[0], 1)))
            
            
            cdf = cp.cumsum(pdf*delta[0], axis = 0)
            
            th = delta/edge_tuning_scale
            y = cp.empty(size, dtype=cp.float64)
            xin = np.array([np.linspace(0,1, 1000)]).T
            ty = gauss_functs.partition(xin, cdf, th, x)
            
            for i in range(size[1]):
                xin = np.random.random((size[0],1))
                ty = gauss_functs.partition(xin, cdf, th, x)
                y[:,i] = ty[:,0]

            try:
                y = y.get()
            except:
                pass
            if(scale_back):
                f, y, mh = self.Scaler.scale_undo(cols=np.array([c]), x=y , kern=True)
                
            RX[c] = y
                
        return RX
    
    def diff_entropy(self,
                     xin:CPArray = None,
                     scaled_in:bool = False,
                     rel_cols:NDArray = None,
                     h_override:CPArray = None,
                     num_int_pts:int = 10,
                     w_cols = None,
                     df:pd.DataFrame = None,
                     kern_pdf_override:object = None,
                     n_sig:int = 2,
                     integrator:Callable = integrators_.simpsons_1_3
                     )->tuple[NDArray, NDArray]:
        """_summary_

        Args:
            xin (cp.array, optional): _description_. Defaults to None.
            scaled_in (bool, optional): _description_. Defaults to False.
            rel_cols (np.array, optional): _description_. Defaults to None.
            h_override (cp.array, optional): _description_. Defaults to None.
            num_int_pts (int, optional): _description_. Defaults to 10.
            w_cols (_type_, optional): _description_. Defaults to None.
            df (pd.DataFrame, optional): _description_. Defaults to None.
            kern_pdf_override (object, optional): _description_. Defaults to None.
            n_sig (int, optional): _description_. Defaults to 2.
            scale_back (bool, optional): _description_. Defaults to True.

        Returns:
            cp.array: _description_
        """
        if(df is None):
            df = self.df.copy(deep=True)
        
        if(rel_cols is None):
            rel_cols = self.x_cols
        rel_cols=np.array(rel_cols)
        
        if(kern_pdf_override is None):
            kern_pdf_override = self.kernal_pdf_funct
        
        if(h_override is None):
            h_override = self.h
        if(xin is None):
            xin, delta = construct_cont_gen(self.df, cols = rel_cols, num_cont_pts=num_int_pts, n_sig=n_sig)
            xin = ord_to_grid(xin).get()
        
        else:
            if not scaled_in and self.do_scaler:
                xin = self.Scaler.scale_do(rel_cols,x=xin,kern=True)
                xin = xin[1]
        
        if(w_cols is None):
            W = cp.ones((df.shape[0], 1))
        
        elif(w_cols == 'y'):
            W = df[self.y_cols].values
            W = W.reshape((W.shape[0], self.y_cols.shape[0]))
        
        else:
            W = df[w_cols].values
            W = W.reshape((W.shape[0], w_cols.shape[0]))
        
        X = df[rel_cols].values
        X = X.reshape((X.shape[0], rel_cols.shape[0]))
        
        return gauss_functs.diff_entropy(xin, X, h_override, W, integrator=integrator)        
        
    def kern_pdf(self,
                 xin:CPArray = None,
                 scaled_in:bool = False,
                 rel_cols:NDArray = None,
                 h_override:CPArray=None, 
                 num_cont_pts:int = int(2e3), 
                 w_cols:NDArray = None, 
                 df:pd.DataFrame = None, 
                 kern_pdf_override:object = None,
                 n_sig:int = 2,
                 scale_back:bool = True,
                 )->tuple[NDArray, NDArray]:
        """_summary_

        Args:
            xin (cp.array, optional): _description_. Defaults to None.
            scaled_in (bool, optional): _description_. Defaults to False.
            rel_cols (np.array, optional): _description_. Defaults to None.
            h_override (cp.array, optional): _description_. Defaults to None.
            num_cont_pts (int, optional): _description_. Defaults to int(2e3).
            w_cols (np.array, optional): _description_. Defaults to None.
            df (pd.DataFrame, optional): _description_. Defaults to None.
            kern_pdf_override (object, optional): _description_. Defaults to None.
            n_sig (int, optional): _description_. Defaults to 2.
            scale_back (bool, optional): _description_. Defaults to True.

        Returns:
            tuple[np.array, np.array]: _description_
        """        
        
        if(df is None):
            df = self.df.copy(deep=True)
        
        if(rel_cols is None):
            rel_cols = self.x_cols
        rel_cols=np.array(rel_cols)
        
        cols_ix = []
        for r in rel_cols:
            cols_ix.append(np.where(self.x_cols == r)[0][0])
            
        
        if(kern_pdf_override is None):
            kern_pdf_override = self.kernal_pdf_funct
        
        if(h_override is None):
            h_override = self.h[cols_ix]
        
        if(xin is None):
            xin, delta = construct_cont_gen(self.df, cols = rel_cols, num_cont_pts=num_cont_pts, n_sig=n_sig)
            xin = ord_to_grid(xin).get()
        else:
            if not scaled_in and self.do_scaler:
                xin = self.Scaler.scale_do(rel_cols,x=xin, kern=True)
                xin = xin[1]
        
        if(w_cols is None):
            W = cp.ones((df.shape[0], 1))     
        elif(w_cols == 'y'):
            W = df[self.y_cols].values
            W = W.reshape((W.shape[0], self.y_cols.shape[0]))      
        else:
            W = df[w_cols].values
            W = W.reshape((W.shape[0], w_cols.shape[0]))
        
        X = df[rel_cols].values
        X = X.reshape((X.shape[0], rel_cols.shape[0])) 
        yh = kern_pdf_override(xin, X, h_override, W)
        
        try:
            yh = yh.get()
        except:
            pass 
        
        if(scale_back):
            d, x_, th = self.Scaler.scale_undo(cols=rel_cols,x=xin, h=h_override, kern=True)
            yh*=np.prod(h_override.get())/np.prod(th)
            
            return yh, x_
        else:
            return yh, xin
    
    def kern_predict(self,
                 xin:CPArray = None,
                 scaled_in:bool = False,
                 rel_cols:NDArray = None,
                 h_override:CPArray=None, 
                 num_cont_pts:int = int(2e3), 
                 w_cols:NDArray = 'y', 
                 df:pd.DataFrame = None, 
                 kern_prd_override:object = None,
                 n_sig:int = 2,
                 scale_back:bool = True,
                 sample_size:float = 1
                 )->tuple[NDArray, NDArray]:
        """_summary_

        Args:
            xin (cp.array, optional): _description_. Defaults to None.
            scaled_in (bool, optional): _description_. Defaults to False.
            rel_cols (np.array, optional): _description_. Defaults to None.
            h_override (cp.array, optional): _description_. Defaults to None.
            num_cont_pts (int, optional): _description_. Defaults to int(2e3).
            w_cols (np.array, optional): _description_. Defaults to 'y'.
            df (pd.DataFrame, optional): _description_. Defaults to None.
            kern_prd_override (object, optional): _description_. Defaults to None.
            n_sig (int, optional): _description_. Defaults to 2.
            scale_back (bool, optional): _description_. Defaults to True.
            sample_size (float): size of sampling for kernal function. Defaults to 1 for full sampling.

        Returns:
            tuple[np.array, np.array]: _description_
        """        
        
        if(df is None):
            df = self.df.copy(deep=True)
        if(rel_cols is None):
            rel_cols = self.x_cols
        rel_cols=np.array(rel_cols)
        cols_ix = []
        for r in rel_cols:
            cols_ix.append(np.where(self.x_cols == r)[0][0])
            
        
        if(kern_prd_override is None):
            kern_prd_override = self.kernal_prd_funct
        
        if(h_override is None):
            h_override = self.h[cols_ix]
        if(xin is None):
            xin, delta = construct_cont_gen(self.df, cols = rel_cols, num_cont_pts=num_cont_pts, n_sig=n_sig)
            xin = ord_to_grid(xin).get()
        else:
            if not scaled_in and self.do_scaler:
                xin = self.Scaler.scale_do(rel_cols,x=xin,kern=True)
                xin = xin[1]
            
        
        if(w_cols is None):
            W = cp.ones((df.shape[0], 1))
        elif(w_cols == 'y'):
            W = df[self.y_cols].values
            W = W.reshape((W.shape[0], self.y_cols.shape[0]))
        else:
            W = df[w_cols].values
            W = W.reshape((W.shape[0], w_cols.shape[0]))
        X = df[rel_cols].values
        X = X.reshape((X.shape[0], rel_cols.shape[0]))
        sample = np.random.choice(np.arange((X.shape[0])),(X.shape[0]*sample_size), replace = True)
        yh = kern_prd_override(xin, X[sample], h_override, W[sample])
        try:
            yh = yh.get()
        except:
            pass
        if(scale_back):
            d, xin, th = self.Scaler.scale_undo(cols=rel_cols,x=xin, h=h_override, kern=True)
            
            return yh, xin
        else:
            return yh, xin
    
    def clt_convergence(self, cols:NDArray, num_steps:int = 30, edge_tuning_scale:int = 3):
        mn = self.df[cols].mean()
        std = self.df[cols].std()
        conv = np.empty((num_steps-1, len(cols)))
        fact = ((1_000* 2) / 4) ** (-1 / (4))
        for n in range(2, num_steps+1):
            X = self.random_sample(cols = cols, edge_tuning_scale=edge_tuning_scale,size=(n, 1_000), scale_back=False)
            i = 0 
            for c in X:
                t = X[c].sum(axis = 0).reshape((X[c].shape[1],1))
                h = t.std(axis = 0)*fact
                x = ord_to_grid(np.array([np.linspace(mn[c]*n - 10* np.sqrt(n)*std[c], mn[c]*n + 10* np.sqrt(n)*std[c], 1_000)])).get()

                pdf = gauss_functs.pdf(x, t, h, np.ones((t.shape[0], 1))).get()
                
                gf = (1/(np.sqrt(2*np.pi*n)*std[c])) * np.exp(-(1/2)*((n*mn[c]-x)/(np.sqrt(n)*std[c]))**2)
                
                conv[n-2, i] = np.sum((gf - pdf)**2)/1_000
                i+=1
        return conv
    
    #here
    def kern_unsup(self,
                    xin:CPArray = None,
                    scaled_in:bool = False,
                    rel_cols:NDArray = None,
                    h_override:CPArray=None, 
                    num_cont_pts:int = int(2e3), 
                    w_cols:NDArray = 'y', 
                    df:pd.DataFrame = None, 
                    X:NDArray = None,
                    W:NDArray = None,
                    kern_prd_override:object = None,
                    n_sig:int = 2,
                    scale_back:bool = True,
                   ):
        
        
        return

    def kern_overlap(self):
        pass 
    
    def kern_fisher(self):
        pass
    
    def lin_reg_fit(self):
        pass
    
    def lin_reg_errors(self):
        pass 
   
    def grad_des_fit(self):
        pass 
  
    def monte_fit(self):
        pass
   
    def confusion_matrix(self):
        pass 
    


def to_str(a, st = 'x'):
    return f"{st}_{a}"
v_str = np.vectorize(to_str)