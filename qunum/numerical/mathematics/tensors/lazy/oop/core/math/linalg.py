from typing import Callable, Any, Self
from torch import Tensor, diag_embed
from torch.linalg import eig, eigh, eigvals, eigvalsh
class LazyDiag:
    def __init__(self, A:Callable[[tuple[Any|Tensor],dict[Any|Tensor]],Tensor], depth:int, assume_hermitian:bool = False)->Self:
        self.A = A
        self.depth = depth
        self.assume_hermitian = assume_hermitian
        return 
    def __call__(self, *args:tuple[Any|Tensor], **kwargs:dict[Any,Tensor])->Tensor: 
        try:
            if(self.assume_hermitian):
                return diag_embed(eigvalsh(self.A(*args, **kwargs)))
            else:
                return diag_embed(eigvals(self.A(*args, **kwargs)))
        except Exception as e:
            raise Exception('Error: \n {e} \n on .diagonilize(assume_hermitian={ah}, depth={d})'.format(ah= str(self.assume_hermitian), depth = str(self.depth), e=str(e)))

class LazyEig:
    def __init__(self, A:Callable[[tuple[Any|Tensor],dict[Any|Tensor]],Tensor], depth:int, eigenvectors:bool = False, justVectors:bool= False, hermitian:bool = False)->Self:
        self.A = A
        self.depth = depth
        self.eigenvectors = eigenvectors
        self.justVects = justVectors
        self.hermitian = hermitian
        return 
    
    def __call__(self, *args:tuple[Any|Tensor], **kwargs:dict[Any,Tensor])->Tensor:
        try:
            match (self.eigenvectors, self.justVects, self.hermitian):
                case (False, _, True):
                    return eigvalsh(self.A(*args, **kwargs))
                case (False, _, False):
                    return eigvals(self.A(*args, **kwargs))
                case (True, False, True):
                    return eigh(self.A(*args, **kwargs))
                case (True, True, True):
                    return eigh(self.A(*args, **kwargs))[0]
                case (True, True, False):
                    return eig(self.A(*args, **kwargs))[0]

        except Exception as e:
            raise Exception('Error on .eig(eigenvectors={ev}, justVectors={justV}, hermitian={ah}, depth={d}) with except {e}'.format(ah= str(self.hermitian), ev = str(self.eigenvectors), justV = str(self.justVects), depth = str(self.depth), e=str(e)))
        
class LazyApplyFofM:
    def __init__(self, A:Callable[[tuple[Any|Tensor],dict[Any|Tensor]],Tensor],  fofM:Callable[[Tensor],Tensor], depth:int,*additional_args:tuple[Any], hermitian:bool = False, **additional_kwargs:dict[Any])->Self:
        self.A = A
        self.fofM = fofM
        self.depth = depth
        self.hermitian = hermitian
        self.additional_args = additional_args
        self.additional_kwargs = additional_kwargs
        return
    def __call__(self, *args:tuple[Any|Tensor], **kwargs:dict[Any|Tensor])->Self:
        A = self.A(*args, **kwargs)
        try:
            if(self.hermitian):
                V, U = eigh(A)
            else:
                V, U = eig(A)
            return U @ self.fofM(diag_embed(V), *self.additional_args, **self.additional_kwargs) @ U.conj().swapdims(-1,-2)
             
        except Exception as e:
            raise Exception('Error on .fofM(fofM={f}, assume_hermitian={ah} depth={d}) with exception {e}'.format(ah= str(self.hermitian), f=str(self.fofM), depth = str(self.depth), e=str(e)))
        