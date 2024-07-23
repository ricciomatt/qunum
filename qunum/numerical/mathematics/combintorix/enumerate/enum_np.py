

import numpy as np 
import torch
from typing import Self, Generator, Any, Iterable
from warnings import warn
from copy import copy

class YieldArgCombosNpFromIter:
    def __init__(self, *args:tuple[torch.Tensor|Any], ret_tensor:bool = True)->Self:
        
        self.idx:torch.Tensor = torch.from_numpy(np.array([len(a) for a in args]))
        self.cix:torch.Tensor = torch.zeros(self.idx.shape[0], dtype = torch.int32)
        self.args:tuple[torch.Tensor] = args
        
        self.ret_tensor:bool = ret_tensor
        self.infer_type(args)
            
        return
    
    def infer_type(self, args):
        torch_type_map = {
            torch.float16:np.float16, 
            torch.float32:np.float32, 
            torch.float64:np.float64, 
            torch.complex32:np.complex128, 
            torch.complex64:np.complex64, 
            torch.complex128:np.complex128,
            
            torch.int16:np.int16, 
            torch.int32:np.int32, 
            torch.int64:np.int64,
            torch.int8:np.int8,
            
            torch.uint8:np.uint8,
            
        }
        if(self.ret_tensor):
            if not (all(
                map(
                    lambda x: isinstance(x, torch.Tensor), args
                )
                )): 
                raise TypeError('Must provide tensor args or instantiate as YiledArgCombos((arg1,arg2...argN), ret_tensor=False)')
            dtype = args[0].dtype
            try:
                if not (all(map(lambda x: x.dtype==dtype, args))):
                    raise TypeError('To Return Tensor output datatypes must match for all args')
                self.ret_type = {'np':torch_type_map[dtype], 'torch':dtype}
                return 
            except TypeError as t:
                warn(str(t))
                self.ret_tensor = False
                self.infer_type()
        else:
            
            try:
                dtype = args[0].dtype
                if not (all(map(lambda x: x.dtype==dtype, args)) and (dtype in torch_type_map.values())):
                    if not (dtype in torch_type_map.keys()):
                        raise TypeError('Unable to infer numpy data type, np.object is set by default')
                    else:
                        if not (all(map(lambda x: x.dtype==dtype, args))):
                            raise TypeError('Unable to infer numpy data type, np.object is set by default')
                        else:
                            self.ret_type = {'np':torch_type_map[dtype], 'torch':dtype}
                            return
                self.ret_type = {'np':dtype, 'torch':None}
            except TypeError as t:
                
                warn(str(t))
                self.ret_type = {'np':np.object_, 'torch':None}
        print(self.ret_type)
        return 
        

    def __iter__(self)->Generator[torch.Tensor, None, None]:
        def yield_arg_combos_inner(args:tuple[tuple[Iterable[Any]]],  dtp:dict[str:torch.dtype, str:np.dtype], cix:torch.Tensor, idx:torch.Tensor, ret_tensor:bool = True,)->Generator[tuple[Any], None, None]:
            fst:bool = True
            while True:
                if(fst):
                    fst = False
                    if(ret_tensor):
                        print(next(map(
                                        lambda x: x, enumerate(cix)
                        )))
                        print(next(map(
                                        lambda x: args[x[0]][x[1]], enumerate(cix)
                        )))
                        print(np.fromiter(map(
                                        lambda x: args[x[0]][x[1]], enumerate(cix)
                                    ),
                                         dtp['np']))
                        yield torch.from_numpy(
                            np.fromiter(
                                    map(
                                        lambda x: args[x[0]][x[1]], enumerate(cix)
                                    ),
                                dtp['np']
                            )
                        )
                    else:
                        yield np.fromiter(map(lambda x: args[x[0]][x[1]], enumerate(cix)), dtp['np'])
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
                        yield torch.from_numpy(
                            np.fromiter(
                                    map(
                                        lambda x: args[x[0]][x[1]], enumerate(cix)
                                    ),
                                dtp['np']
                            )
                        )
                    else:
                        yield np.fromiter(map(lambda x: args[x[0]][x[1]], enumerate(cix)), dtp['np'])
                else:
                    break
        return yield_arg_combos_inner(self.args, self.ret_type, self.cix.clone(), self.idx.clone(), self.ret_tensor, )
    
    def __list__(self,)->list[torch.Tensor|Any]:
        return list(iter(self))
    
    def __call__(self)->torch.Tensor|np.ndarray[Any]:
        if(self.ret_tensor):
            try:
                a = np.fromiter(iter(self),dtype=np.dtype((self.ret_type['np'], self.idx.shape[0])))
            except:
                a =  np.fromiter(iter(self), dtype = np.object)
            return torch.from_numpy(a).to(dtype = self.ret_type['torch'])
        else:
            try:
                return np.fromiter(iter(self),dtype=np.dtype((self.ret_type['np'], self.idx.shape[0])))
            except:
                return np.fromiter(iter(self), dtype = np.object_)
    def __array__(self)->np.ndarray:
            try:
                return np.fromiter(iter(self),dtype=np.dtype((self.ret_type['np'], self.idx.shape[0])))
            except:
                return np.fromiter(iter(self), dtype = np.object_)
    
    def __tensor__(self)->torch.Tensor:
        ret = copy(self.ret_tensor)
        self.ret_tensor = True
        a = self()
        self.ret_tensor = ret
        return a

