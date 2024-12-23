from .....qobjs import TQobj
import torch
from warnings import warn
from typing import Self


class ProductFormula:
    def __init__(self, order:int = 2)->Self:
        assert int(order) >0 , ValueError('Order must be an even integer')
        if(order%2 and order !=3):
            self.order = int(order) + int(order)%2
            warn(f'Only implemented for 3rd and even order Product Fromulae {self.order}')
        else:
            self.order = int(order)
        return 
    
    def __call__(self, A:TQobj, B:TQobj)->TQobj:
        return Sn(A, B, self.order)

def Sn(A:TQobj, B:TQobj, order:int, sin:float=1.0)->TQobj:
    match order:
        case 1:
            return A.expm() @ B.expm()
        case 2:
            return (A*(s/2)).expm()@ (B*s).expm() @ (A*(s/2)).expm()
        case 3:
            s = (2-2**(1/3))**(-1)
            return Sn(A, B, order=2, sin = s*sin) @ Sn(A, B, order = 2, sin=(1-2*s)*sin ) @ Sn(A,B,order=2, sin =s*sin)
        case _:
            s = (4-4**(1/(2*order+1)))**(-1)
            return torch.matrix_power(Sn(A,B,order-2, sin=s*sin), 2) @ Sn(A,B,order-2, sin=(1-4*s)*sin) @ torch.matrix_power(Sn(A,B, order-2, sin=sin*s), 2)
