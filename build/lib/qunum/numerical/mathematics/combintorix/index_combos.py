import torch
from typing import Iterator, Set
class LazyEnumIndex:
    def __init__(self, inpt:torch.Tensor)->None:
        self.idx = torch.tensor(inpt.shape, dtype = torch.int32)
        self.cix = torch.zeros(self.idx.shape[0], dtype = torch.int32)
        self.fst:bool = True
        return 
    def __next__(self)->torch.Tensor:
        if(self.fst):
            self.fst = False
            return self.cix
        if not (torch.all(self.cix == self.idx-1)) :
            a = 1
            k = True
            while k:
                if (self.cix[-a] == self.idx[-a]-1):
                    if(a == self.cix.shape[0]):
                        break
                    self.cix[-a] = 0
                    a += 1
                else:
                    self.cix[-a] += 1
                    break
            return self.cix.clone()
        
        else:
            self.fst = True
            self.cix = torch.zeros(self.idx.shape[0], dtype = torch.int32)
            raise StopIteration
            
    
    def __len__(self)->int:
        return torch.prod(self.idx)
    
    def __iter__(self):
        return self
