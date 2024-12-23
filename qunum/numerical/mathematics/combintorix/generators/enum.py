from typing import Any, Generator, Iterable, Self, Callable
import torch
import numpy as np 
from warnings import warn
from copy import copy

class EnumerateArgCombos:
    def __init__(self, *args:tuple[torch.Tensor|range], ret_tensor:bool = True, ignore_arange:bool = False)->Self:
        self.idx:torch.Tensor = torch.from_numpy(np.array([a.stop if(isinstance(a,range)) else len(a) for a in args]))
        self.cix:torch.Tensor = torch.zeros(self.idx.shape[0], dtype = torch.int32)
        self.args:tuple[torch.Tensor] = args
        self.ret_tensor:bool = ret_tensor
        self.ignore_arange:bool = ignore_arange
        if(ret_tensor): 
            if(not ignore_arange):
                self.args = self.getArange()
            
            def getIdxs(x:torch.Tensor):
                N = x.clone()
                x = torch.ones(self.idx.shape[0], dtype = torch.int64)*x
                x[-1] = N%self.idx[-1]
                for j in range(-2, -self.idx.shape[0]-1, -1):
                    x[j] = (N//self.idx[j+1:].prod())%self.idx[j]
                return x
            self.getTorchTensor:Callable[[torch.Tensor], torch.Tensor]= torch.vmap(getIdxs)            
        return 
    def getArange(self)->bool:
        def convertRanges(x:torch.Tensor|range|np.ndarray):
            if(isinstance(x, range)):
                return torch.arange(x.start, x.stop, x.step)
            elif(isinstance(x,np.ndarray)):
                try:
                    return torch.from_numpy(x)
                except:
                    raise TypeError('Could not convert numpy type:{dt} to torch data type, User EnumerateArgCombos(..., ret_tensor = False)'.format(dt=x.dtype))
            elif(isinstance(x,torch.Tensor)):
                return x
            else:
                raise TypeError('Must Be Tensor or range to use ret_tensor = True')
        
        return tuple(map(convertRanges, self.args))
    
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
        if(self.ignore_arange and self.ret_tensor):
            return yield_arg_combos_inner(self.getArange(), self.cix.clone(), self.idx.clone(), self.ret_tensor, dim = dim)
        else:
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

    def __call__(self, dim:int|Iterable[int]|None = None, rawIdx:bool=False)->torch.Tensor|np.ndarray[Any]|list[Any]:
            if(self.ret_tensor):
                return self.__tensor__(dim=dim, rawIdx=rawIdx)
            else: 
                try:
                    a = self.__list__(dim=dim)
                    return np.array(a)
                except:
                    a = self.__list__(dim=dim)
                    try:
                        return np.array(a, dtype=np.object_)
                    except:
                        return a
            
    def __array__(self)->np.ndarray:
        try:
            return np.array(list(self))
        except:
            return np.array(list(self), dtype = np.object_)
    
    def __tensor__(self, dim:int|Iterable[int]|None = None, rawIdx:bool=False)->torch.Tensor:
        assert self.ret_tensor, 'Must Be tensor arguments and enable ret tensor, ie. EnumerateArgCombos(Tensor1,Tensor2,..., ret_tensor = True)'
        if(rawIdx or self.ignore_arange):
            if(dim is None):
                return self.getTorchTensor(torch.arange(0,self.idx.prod(), step = 1))
            else:
                return self.getTorchTensor(torch.arange(0,self.idx.prod(), step = 1))[:,dim]
        else:
            K = self.getTorchTensor(torch.arange(0,self.idx.prod(), step = 1))
            if(dim is None):
                return torch.stack([a[K[:,i]] for i, a in enumerate(self.args)], ).swapdims(dim0=0, dim1=1)
            else:
                match dim:
                    case dim if isinstance(dim, int):
                        return self.args[dim][K[:,dim]]
                    case dim if isinstance(dim,list) or isinstance(dim, tuple) or isinstance(dim, torch.Tensor) or isinstance(dim, np.ndarray):
                        if(len(dim) == 1):
                            return self.args[dim][K[:,dim]]
                    
                        else:
                            return torch.stack([self.args[d][K[:,d]] for d in dim], ).swapdims(dim0=0, dim1=1)




def enumerate_arg_combos(*args:tuple[tuple[Iterable[Any]]], ret_tensor:bool = True,)->Generator[tuple[Any], None, None]:
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
