from typing import Iterable, Any, Callable
class PrePipe:
    def __init__(self, mods:Iterable[Callable]=[], splitData:Callable = None )->None:
        self.mods = tuple(mods)
        self.splitData = splitData
        return
    
    def __call__(self, *args:tuple[Any], **kwargs:dict[Any])->Any:
        for i, m in enumerate(self.mods):
            args = m(*args, **kwargs)
        return args
    
    def __len__(self)->Callable:
        return self.mods.__len__()
    
    def __iter__(self)->Callable:
        return self.mods.__iter__()
    
    def __next__(self)->Callable:
        return self.mods.__next__()
    
    def __getitem__(self, ix:int)->Callable:
        return self.mods.__getitem__(ix)
    
    def split(self, *args:tuple[Any], **kwargs:dict[Any])->tuple[Any]:
        return self.splitData(*args, **kwargs)
    
    def add(self, inp_mod:Callable)->None:
        mods = list(self.mods)
        mods.append(inp_mod)
        self.mods = tuple(mods)
        return 
    
    def insert(self, inp_mod:Callable, ix:int = 0)->None:
        mods = list(self.mods)
        mods.insert(ix, inp_mod)
        self.mods = tuple(mods)
        return  

    def extend(self, inp_mod:Iterable[Callable])->None:
        mods = list(self.mods)
        mods.extend(inp_mod)
        self.mods = tuple(mods)
        return 