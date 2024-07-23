from typing import Self, Generator, Iterable
import numpy as np
class EnumUniqueUnorderedIdx:
    def __init__(self, *args)->Self:
        self.lower = np.array([a.start for a in args])
        self.upper = np.array([a.stop for a in args])
        return 
    def __iter__(self, tuple_ = False)->Generator[Iterable[int], None, None]:
        def yield_ixs(cix:np.ndarray[np.int32], idx:np.ndarray[np.int32], tuple_:bool = False):
            fst:bool = True
            inDer = []
            while True:
                if(fst):
                    fst = False
                    yield cix
                if not (np.all(cix == idx-1)):
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
                    
                    m:dict[int:int] = {k : (cix == k).astype(np.int32).sum() for k in np.unique(cix)}
                    if(m not in inDer):
                        inDer.append(m)
                        if(tuple_):
                            yield tuple(cix.copy().tolist())
                        else:
                            yield cix.copy()

                else:
                    break
        return yield_ixs(self.lower.copy(), self.upper.copy(), tuple_)
    
    def __list__(self,tuple_:bool = False)->list[np.ndarray[np.int32]]:
        return list((self).__iter__(tuple_=tuple_))

    def __set__(self)->set[tuple]:
        return tuple(self.__iter__(tuple_=True))
    
    def __tuple__(self, tuple_:bool = True)->tuple[np.ndarray[np.int32]|tuple]:
        return tuple(self.__iter__(tuple_=tuple_))

    def __call__(self, tuple_ = False)->np.ndarray[np.int32]:
       if(tuple_):
            return np.array(self)
       else:
           return self.__tuple__(tuple_=True)
    
    def __array__(self):
        return np.fromiter(iter(self), np.dtype((np.int32, self.lower.shape[0])))