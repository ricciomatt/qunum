import polars as pl
from .meta import SmartMeta

class SmartDf(pl.DataFrame):
    def __init__(self, *args,meta:None = None, **kwargs)->None:
        super(SmartDf,self).__init__(*args, **kwargs)
        if(meta is not None):
            self._metadata = meta
        else:
            self._metadata = SmartMeta()
        return 
    def clean(self):
        pass
    def _filter_(self):
        pass
    