from plotly import express as px
from polars import DataFrame
class PlotIt:
    def __init__(self, engine:str='plotly', offline:bool = False)->None:
        self.engine = engine
        self.offline = offline
        pass

    def show(self):
        pass 
    
    def add_data(self, *args, **kwargs):
        pass
    
    def remove_data(self, *args, **kwargs):
        pass

    def __repr__(self):
        pass 

