from typing import Any, Generator, Iterable, Self
import torch
import numpy as np 
from warnings import warn
from copy import copy
class YieldArgCombos:
    def __init__(self, *args:tuple[torch.Tensor|Any], ret_tensor:bool = True)->Self:
        self.idx:torch.Tensor = torch.from_numpy(np.array([len(a) for a in args]))
        self.cix:torch.Tensor = torch.zeros(self.idx.shape[0], dtype = torch.int32)
        self.args:tuple[torch.Tensor] = args
        self.ret_tensor:bool = ret_tensor
        if(ret_tensor):
            assert all(map(lambda x: isinstance(x, torch.Tensor), args)), TypeError('Must provide tensor args or instantiate as YiledArgCombos((arg1,arg2...argN), ret_tensor=False)')
        return

    def __iter__(self, dim:int|Iterable[int]|None = None)->Generator[torch.Tensor, None, None]:
        def yield_arg_combos_inner(args:tuple[tuple[Iterable[Any]]], cix:torch.Tensor, idx:torch.Tensor, ret_tensor:bool = True, dim:int|Iterable[int]|None = None)->Generator[tuple[Any], None, None]:
            fst:bool = True
            if(dim is None):
                dim = torch.arange(cix.shape[0])
            elif(isinstance(dim, int)):
                dim = [dim]
            while True:
                if(fst):
                    fst = False

                    if(ret_tensor):
                        if(len(dim)>1):
                            yield torch.from_numpy(
                                np.array(
                                    list(
                                        map(
                                            lambda x: args[x[0]][x[1]], 
                                            zip(dim, cix[dim])
                                        )
                                    )
                                )
                            )
                        else:
                            if(isinstance(args[dim[0]][cix[dim[0]]], torch.Tensor)):
                                yield args[dim[0]][cix[dim[0]]]
                            else:

                                yield torch.tensor(args[dim[0]][cix[dim[0]]])
                    else:
                        yield tuple(
                            map(
                                lambda x: args[x[0]][x[1]], 
                                zip(dim,cix[dim])
                            )
                        )
                if not (torch.all(cix == idx-1)) :
                    a = 1
                    k = True
                    while k:
                        if (cix[-a] == idx[-a]-1):
                            if(a == cix.shape[0]):
                                break
                            cix[-a] = 0
                            a += 1
                        else:
                            cix[-a] += 1
                            break
                    
                    if(ret_tensor):
                        if(len(dim)>1):
                            yield torch.from_numpy(
                                np.array(
                                    list(
                                        map(
                                            lambda x: args[x[0]][x[1]], 
                                            zip(dim, cix[dim])
                                        )
                                    )
                                )
                            )
                        else:
                            if(isinstance(args[dim[0]][cix[dim[0]]], torch.Tensor)):
                                yield args[dim[0]][cix[dim[0]]]
                            else:

                                yield torch.tensor(args[dim[0]][cix[dim[0]]])
                        
                    else:
                        yield tuple(
                            map(
                                lambda x: args[x[0]][x[1]], 
                                zip(dim,cix[dim])
                            )
                        )
                else:
                    break
        assert any([dim is None, isinstance(dim,int), isinstance(dim, tuple), isinstance(dim,list), isinstance(dim, torch.Tensor), isinstance(dim, np.ndarray)]), 'Dim must be int, None, or Iterable[int]'
        return yield_arg_combos_inner(self.args, self.cix.clone(), self.idx.clone(), self.ret_tensor, dim = dim)
    
    def __list__(self, dim:int|Iterable[int]|None = None)->list[torch.Tensor|Any]:
        return list(self.__iter__(dim=dim))
    
    def __tuple__(self, dim:int|Iterable[int]|None = None)->tuple[torch.Tensor|Any]:
        return tuple(self.__iter__(dim))

    def __set__(self, dim:int|Iterable[int]|None = None)->set[tuple]:
        ret_ten = copy(self.ret_tensor)
        self.ret_tensor = ret_ten
        a = set(self.__iter__(dim=dim))
        self.ret_tensor = ret_ten
        return a

    def __call__(self, dim:int|Iterable[int]|None = None)->torch.Tensor|np.ndarray[Any]|list[Any]:
        a = self.__list__(dim=dim)
        try:
            if(self.ret_tensor):
                return torch.stack(a)
            else: 
                return np.array(a)
        except:
            try:
                return np.array(a, dtype=np.object_)
            except:
                return a
            
    def __array__(self)->np.ndarray:
        try:
            return np.array(list(self))
        except:
            return np.array(list(self), dtype = np.object_)
    
    def __tensor__(self)->torch.Tensor:
        return torch.stack(list(self))




def yield_arg_combos(*args:tuple[tuple[Iterable[Any]]], ret_tensor:bool = True,)->Generator[tuple[Any], None, None]:
    idx = torch.from_numpy(np.array([len(a) for a in args]))
    cix = torch.zeros(idx.shape[0], dtype = torch.int32)
    fst:bool = True
    while True:
        if(fst):
            fst = False
            if(ret_tensor):
                yield torch.from_numpy(np.array(list(map(lambda x: args[x[0]][x[1]], enumerate(cix)))))
            else:
                yield tuple(map(lambda x: args[x[0]][x[1]], enumerate(cix)))
        if not (torch.all(cix == idx-1)) :
            a = 1
            k = True
            while k:
                if (cix[-a] == idx[-a]-1):
                    if(a == cix.shape[0]):
                        break
                    cix[-a] = 0
                    a += 1
                else:
                    cix[-a] += 1
                    break
            if(ret_tensor):
                yield torch.from_numpy(np.array(list(map(lambda x: args[x[0]][x[1]], enumerate(cix)))))
            else:
                yield tuple(map(lambda x: args[x[0]][x[1]], enumerate(cix)))
        else:
            break
