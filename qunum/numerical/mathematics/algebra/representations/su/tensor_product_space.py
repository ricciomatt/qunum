import torch
from .....physics.quantum.qobjs.torch_qobj import TQobj, direct_prod
def get_Jrepeat(n:int, sigma:TQobj)->TQobj:
    '''Pass in \sigma_\mu this will take \ket{s_1}\ket{s_2}...\ket{s_n}'''
    sigma[1:] /= 2
    J = TQobj(torch.zeros((sigma.shape[0], int(sigma.shape[1]**n), int(sigma.shape[1]**n)), dtype = sigma.dtype), n_particles=n, hilbert_space_dims=sigma.shape[1])
    return mkJ(J, n , sigma)

def getJ(*args:tuple[TQobj]):
    '''Pass in \lambda_{\mu} this will tensor these object together and return a definite J'''
    pass 


def mkJ(J:TQobj, n:int, sigma:TQobj)->TQobj:
    for i in range(sigma.shape[0]):
        for j in range(n):
            if(j == 0):
                temp = sigma[i]
            else:
                temp = sigma[0]
            for k in range(1,n):
                if(k == j):
                    temp = direct_prod(temp, sigma[i])
                else:
                    temp = direct_prod(temp, sigma[0])
            J[i]+=temp
    return J

