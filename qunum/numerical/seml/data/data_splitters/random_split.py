import numpy as np
import polars as pl
from ..data_loaders import Dataset, make_data_loader
class RandomSplit:
    def __init__(self, train_pct:float=.9, test_pct:float=0.05, validation_pct:float=0.05, mx_num:int=int(1e3), randomize=True, to_dataLoader:bool=True, **kwargs)->None:
        self.trn_pct = train_pct
        self.test_pct = test_pct
        self.validation_pct = validation_pct
        self.mx_num = int(mx_num)
        self.ixs = None
        self.randomize = randomize
        self.to_dataLoader = to_dataLoader
        return 
    
    def __call__(self, df:pl.DataFrame, resample:bool = False):
        ldf = df.lazy().with_row_count()
        if(self.ixs is None or resample):
            if(self.randomize):
                self.ixs = np.random.randint(0, df.shape[0], min(df.shape[0], self.mx_num))
            else:
                self.ixs = np.arange(0, min(df.shape[0], self.mx_num))
        ltrn = int(np.ceil(self.trn_pct*self.ixs.shape[0]))
        ltst = int(np.ceil(self.test_pct*self.ixs.shape[0]))
        trn = ldf.filter(pl.col('row_nr').is_in(self.ixs[:ltrn])).collect()
        tst = ldf.filter(pl.col('row_nr').is_in(self.ixs[ltrn:ltrn+ltst])).collect()
        vld = ldf.filter(pl.col('row_nr').is_in(self.ixs[ltrn+ltst:])).collect()
        return trn, tst, vld

