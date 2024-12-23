from .....qobjs import TQobj
import torch
def aDag_(n:int,offset:int = 0, spacing:int = 1, dtype = torch.complex128)->TQobj:
    aDag = torch.zeros(n,n, dtype = dtype)
    aDag[torch.arange(n-1), torch.arange(1,n)] = torch.sqrt(spacing*(torch.arange(n-1)+1+offset).to(dtype))    
    return TQobj(aDag)

def a_(n:int, offset:int = 0, spacing:int = 1, dtype = torch.complex128)->TQobj:
    a = torch.zeros(n,n, dtype = dtype)
    a[torch.arange(1, n), torch.arange(n-1)] = torch.sqrt(spacing*(torch.arange(n-1)+1+offset).to(dtype))    
    return TQobj(a)
