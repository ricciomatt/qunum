import torch
from ...physics.quantum.qobjs import TQobj
from itertools import product
from .. import commutator as comm

def compose_magnus_basis(HBasis:TQobj)->set[TQobj]:
    Hamiltonian_Basis:set = set(map(lambda i: HBasis[i], range(HBasis.shape[0])))
    a:bool = 1
    i:int = 0
    while a:
        t:set = Hamiltonian_Basis.copy()
        for h in Hamiltonian_Basis:
            for b in Hamiltonian_Basis:
                tCom = comm(h, b)
                A = map(
                    lambda i: 
                        (
                            torch.all(torch.round(tCom.real, decimals = 3) == torch.round(i.real, decimals = 3)) 
                                and 
                            torch.all(torch.round(tCom.imag, decimals = 3) == torch.round(i.imag, decimals = 3))
                        )
                        or
                        (
                            torch.all(torch.round(tCom.real, decimals = 3) == torch.round((-i).real, decimals = 3)) 
                                and 
                            torch.all(torch.round(tCom.imag, decimals = 3) == torch.round((-i).imag, decimals = 3))
                        )
                        
                        , t
                    )
                B = (
                        torch.all(torch.round(tCom.imag, decimals= 6) == 0)
                            and
                        torch.all(torch.round(tCom.real, decimals= 6) == 0)
                    )
                if not (any(A) or B):
                    t.add(tCom)
        if(len(t) == len(Hamiltonian_Basis)):
            a:bool = 0
        else:
            Hamiltonian_Basis:set = t.copy()
    return Hamiltonian_Basis