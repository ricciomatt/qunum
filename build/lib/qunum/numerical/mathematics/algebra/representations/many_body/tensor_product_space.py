import torch
from .....physics.quantum.qobjs.dense.core.torch_qobj import TQobj, direct_prod

def getJ(*args:tuple[TQobj])->tuple[TQobj]:
    I = get_Ids(*args)
    J = []
    for i, a in enumerate(args):
        J.append(direct_prod(*(I[j] if j != i else a for j in range(len(args)))))
    return tuple(J)

def getJtot(*args:tuple[TQobj])->tuple[TQobj]:
    I = get_Ids(*args)
    J = []
    for i, a in enumerate(args):
        J.append(direct_prod(*(I[j] if j != i else a for j in range(len(args)))))
    return sum(J)

def get_Jrepeat(n:int, sigma:TQobj)->TQobj:
    '''Pass in \sigma_\mu this will take \ket{s_1}\ket{s_2}...\ket{s_n}'''
    I = TQobj(torch.eye(sigma.shape[1], sigma.shape[2]).reshape(1, sigma.shape[1], sigma.shape[2]), meta = sigma._metadata)
    return sum(
        [
            direct_prod(
                *(
                    sigma if i == j else I for i in range(n)
                )
            )
            for j in range(n)
        ]
    )

def get_Jn(n:int, N:int, sigma:TQobj)->tuple[TQobj]:
    I = TQobj(torch.eye(sigma.shape[1], sigma.shape[2]).reshape((1, sigma.shape[1], sigma.shape[2])), meta = sigma._metadata)
    return direct_prod(*(sigma if i == n else I for i in range(N)))

def get_Ids(*args:tuple[TQobj])->tuple[TQobj]:
    return list(
            map(
            lambda a: TQobj(
                    torch.eye(
                        a.shape[1], a.shape[2]
                    ).reshape(
                        (
                            1, a.shape[1], a.shape[2]
                        )
                    ), 
                    meta=a._metadata
                ),
                args
            )
    )
