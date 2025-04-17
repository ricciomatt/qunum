from ....physics.quantum.qobjs import TQobj
import torch
def gamma_matricies(basis = 'dirac', tqobj:bool = True, to_tensor:bool = True, dtype:torch.dtype = torch.complex128, device:int|str = 'cpu')->TQobj|torch.Tensor:
    from ..sun import get_pauli
    y = torch.zeros(5,4,4, dtype=dtype, device=device)
    s = get_pauli(ret_type='tensor', dtype=dtype, device =device)
    match basis:
        case 'dirac':
            y[0, :2, :2] = s[0]
            y[0,2:, 2:] = -s[0]
            
            y[1:4, 2:, :2] = -s[1:]
            y[1:4, :2, 2:] = s[1:]
            
            y[4, :2, 2:] = s[0]
            y[4, 2:, :2] = s[0]
            
        case basis if basis in ['weyl', 'chiral']:
            y[0, :2, 2:] = s[0]
            y[0, 2:, :2] = s[0]
            
            y[1:4, 2:, :2] = -s[1:]
            y[1:4, :2, 2:] = s[1:]
            
            y[4, :2, :2] = -s[0]
            y[4,2:, 2:] = s[0] 
            
            
        case basis if basis in ['weyl_alt', 'chiral_alt']:
            y[0, :2, 2:] = -s[0]
            y[0, 2:, :2] = -s[0]
            
            y[1:4, 2:, :2] = -s[1:]
            y[1:4, :2, 2:] = s[1:]
            
            y[4, :2, :2] = s[0]
            y[4,2:, 2:] = -s[0] 
            
        case 'majorana':
            y[0, :2, :2] = s[2]
            y[0,2:, 2:] = s[2]
            
            y[1, 2:, 2:] = 1j*s[3]
            y[1, :2, :2] = 1j*s[3]
            
            y[2, 2:, :2] = -s[2]
            y[2, :2, 2:] = s[2]
            
            y[3, :2, :2] = -1j*s[1]
            y[3:4, 2:, 2:] = -1j*s[1]
            
            y[4, :2, 2:] = s[2]
            y[4, 2:, :2] = -s[2]

    if(tqobj):
        return TQobj(y, dims = {0:2, 1:2}, dtype=dtype, device=device)
    else:
        return y



