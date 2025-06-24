import polars as pl
from typing import Self, Iterable
import torch
class PlsMinMaxScaler:
    def __init__(self, data:pl.LazyFrame)->Self:
        oper = [pl.col(col).max().alias('Max_{col}'.format(col=col)) for col in data.columns if data.schema in [pl.Float32, pl.Float64]]
        oper.extend([pl.col(col).min().alias('Min_{col}'.format(col=col)) for col in data.columns if data.schema in [pl.Float32, pl.Float64]])
        self.data = data.select(*oper).collect()     
        return 
    def __get_data(self, x_col:Iterable[str]|str)->pl.DataFrame:
        match x_col:
            case str():

                return self.data
            case _:
                pass 
        return 
    def scale_tensor(self, x:torch.Tensor, x_col:Iterable[str]|str)->torch.Tensor:
        return
    def scale_dataframe(self,df:pl.DataFrame, x_col:Iterable[str]|str)->pl.DataFrame:
        return 
    def invscale_tensor(self,x:torch.Tensor, x_col:Iterable[str]|str)->torch.Tensor:
        return
    def invscale_dataframe(self, df:pl.DataFrame, x_col:Iterable[str]|str)->pl.DataFrame:
        return

