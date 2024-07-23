import polars as pl
from typing import Self
class PauliMatrix:
    def __init__(self,):
        self.Data:pl.LazyFrame= pl.LazyFrame()
        pass
    def doit(self )->None|Self:
        pass