from typing import Any, Iterable
from plotly import express as px
from polars import DataFrame
class PlotIt:
    def __init__(self, engine:str='plotly', offline:bool = False)->None:
        self.engine = engine
        self.offline = offline
        self.Configs = {''}
        self.plot = None
        pass
    
    def configure(self, *args, **kwargs):
        self.Configs.update(kwargs)
        return
    def __getattribute__(self, *args:str|Iterable[str]) -> Any:
        D = self.Configs
        for a in args:
            D = D[a]
        return D
    def __setattr__(self, __name: str, __value: Any) -> None:
        self.Configs[__name] = __value
        return 
    
    def show(self):

        return
    
    def plot_it(self):
        pass
    
    def remove_data(self, *args, **kwargs):
        return

    def __repr__(self):
        return 
