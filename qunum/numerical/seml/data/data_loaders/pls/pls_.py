import polars as pl 
from typing import Sequence
import numpy as np 
from ...pipelines import PrePipe
from torch.utils.data import DataLoader, Dataset
from torch import from_numpy, Tensor


def make_pls_data_loader(df:pl.DataFrame, x_cols:Sequence[str], y_cols:Sequence[str], 
                         batch_size:int = None, 
                         batch_pct:float = .1, 
                         pipeline:PrePipe = PrePipe(), 
                         randomize:bool = True, 
                         requires_grad:bool = False,
                         ToTesnor:bool = True, **kwargs):
    if(batch_size is None):
        batch_size = int(batch_pct*df.shape[0])
    lds = LazyPolarsDs(
                df, x_cols= x_cols, y_cols=y_cols, pipe=pipeline
            )
    return DataLoader(lds, 
        batch_size=batch_size, 
        shuffle=randomize)
    

class LazyPolarsDs(Dataset):
    def __init__(self, df:pl.DataFrame, x_cols:Sequence[str], y_cols:Sequence[str], pipe:PrePipe)->None:
        super(LazyPolarsDs, self).__init__()
        self.n = 0
        self.df = df.with_row_count().lazy()
        self.shp = df.shape[0]
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.pipe = pipe 
        return

    def __iter__(self)->None:
        return 
    
    def __getitem__(self, ix:int)->tuple[Tensor, Tensor]:
        t = pl.DataFrame(self.pipe(self.df.filter(pl.col('row_nr')==ix).fetch(1)))
        return (
                    from_numpy(t[self.x_cols].to_arrow().to_numpy()), 
                    from_numpy(t[self.y_cols].to_arrow().to_numpy())
                )
    
    def __len__(self)->int:
        return self.shp