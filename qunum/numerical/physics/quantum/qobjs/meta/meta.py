import polars as pl
from numpy.typing import NDArray
import numpy as np
from .core import  vgc
import polars as pl 
import numpy as np
from typing import Iterable, Any, Self
import itertools
from torch import Tensor, arange
from .....mathematics.combintorix import EnumerateArgCombos as enumIt
class QobjMeta:
    def __init__(self, 
                 n_particles:int= 1,
                 hilbert_space_dims:int = 2, 
                 shp:tuple[int] = None, 
                 dims:None|dict[int:int] = None,
                 is_hermitian:bool = False,
                 )->Self:
        if(shp[-1] == 1  and shp[-2] == 1):
            self.obj_tp = 'scaler'
            l = 1
        elif(shp[-1] == shp[-2]):
            self.obj_tp = 'operator'
            l = shp[-1]
        elif(shp[-2] == 1):
            self.obj_tp = 'bra'
            l = shp[-1]
        elif(shp[-1] == 1):
            self.obj_tp = 'ket'
            l = shp[-2]
        else:
            raise ValueError(str(shp) + 'Is not a valid shape')
        self.shp = shp
       
        if(dims is None):
            dims, hilbert_space_dims, n_particles = self.infer_dims(l, n_particles, hilbert_space_dims)
        elif(self.obj_tp != 'scaler'):
            assert np.prod(list(dims.values())) == l, f'''The ProductSum(dims) must be equivalent to the number of dimensions of the hilbert space'''
        else:
            dims:dict[int:int] = dict.fromkeys(range(n_particles),0)
        self.hilbert_space_dims:int = l
        self.shp:tuple = shp
        self.refactor_dims(dims)
        self.eigenBasis:Tensor  = None 
        self.eigenVals:Tensor = None
        self.is_hermitian:bool = is_hermitian
        return
    
    def set_eig(self, eigenBasis:Tensor|None = None , eigenVals:Tensor|None = None):
        if(eigenBasis is not None): self.eigenBasis = eigenBasis.to_sparse()
        if(eigenVals is not None): self.eigenVals = eigenVals.to_sparse()
        return
    
    def get_eig_vals(self)->Tensor|None:
        if(self.eigenVals is not None):
            eigV = self.eigenVals.to_dense()
        else:
            eigV = None
        return eigV


    def get_eig(self)->tuple[Tensor, Tensor]|tuple[Tensor,None]|tuple[None, None]:
        if(self.eigenBasis is not None):
            eigB = self.eigenBasis.to_dense()
        else:
            eigB = None
        if(self.eigenVals is not None):
            eigV = self.eigenVals.to_dense()
        else:
            eigV = None
        return eigV, eigB

    def refactor_dims(self, dims:dict[int:int])->None:
        self.dims = dims 
        if(self.obj_tp != 'scaler'):
            A = enumIt(*(range(0,dims[x]) for x in dims), ignore_arange=True)
            self.ixs = pl.LazyFrame(A.__tensor__(rawIdx=True).numpy()).with_row_index().with_columns(pl.col('index').cast(pl.Int32))
        else:
            self.ixs = None
        self.n_particles = len(dims)
        return

    def update_dims(self, keep_ixs:Iterable, reorder:bool = False)->None:
        self.ixs = self.ixs.select(vgc(keep_ixs))
        if(reorder):
            self.dims = {int(i):self.dims[k] for i, k in enumerate(keep_ixs)}
            self.ixs = self.ixs.rename({f"column_{k}":f"column_{i}" for i, k in enumerate(keep_ixs)}).with_row_index()
        else:
            self.dims = {int(k):self.dims[k] for k in keep_ixs}
            self.ixs = self.ixs.with_row_index()
        self.n_particles = int(len(self.dims))
        self.hilbert_space_dims = int(np.prod(list(self.dims.values())))
        return 
    
    def _reset_(self):
        self.eigenBasis = None
        self.eigenVals = None 
        self.is_hermitian = False
        return
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return f'''TQobj(n_particles={str(self.n_particles)}, dims={str(self.dims)}, total_hilbert_space_dims={self.hilbert_space_dims}, shape={self.shp}, object_tp='{str(self.obj_tp)}')'''
        
    def infer_dims(self, l:int, n_particles:int, hilbert_space_dims:int)->tuple[dict[int:int], int, int]:
        if(self.obj_tp == 'scaler'):
            hilbert_space_dims = 0
        elif( n_particles == 1):
            hilbert_space_dims = l
        elif not (hilbert_space_dims**n_particles == l or self.obj_tp == 'scaler'):
            n_particles = np.log(l)/np.log(hilbert_space_dims)
            assert (n_particles == int(n_particles)), f'''Defaulting to same dimensional hilbert space, dimension of Object must be an integer value(ie log_{hilbert_space_dims}(n_particles) is int) this is not the case for input values of n_particles = {n_particles} and hilbert_space_dims = {hilbert_space_dims}'''
        elif(hilbert_space_dims**n_particles == l):
            pass
        else:
            raise ValueError(f'''Cannot Resolve the dimensions of this object, dimension of Object must be an integer value(ie log_{hilbert_space_dims}(n_particles) is int) this is not the case for input values of n_particles = {n_particles} and hilbert_space_dims = {hilbert_space_dims}', \n OR Pass object dimensions of each particle in a dictionary ie {{0:2, 1:3... N:2}} for a object that has N particles indexed by the keys with hilbert space dimensions of the value''')
        dims = dict.fromkeys(range(int(n_particles)), int(hilbert_space_dims))
        return dims, hilbert_space_dims, n_particles
    
    def particle_in(self, ixs:Iterable)->None:
        return all(map(lambda x: x in self.dims , ixs))
    
    def check_particle_ixs(self, ix:Iterable|int)->NDArray:
        if(is_iterable(ix)): 
            ix_ = np.array(ix, dtype=np.int32)
        elif(isinstance(ix, int)): 
            ix_ = np.array([ix], dtype=np.int32)
        else: 
            raise TypeError('tr_out must be Iterable[int] or int')
        assert self.particle_in(ix_), ValueError('Particle Not found')
        return np.array(ix_)
    
    def query_particle_ixs(self, ix:NDArray)->NDArray:
        a:np.ndarray = vgc(ix)
        return self.ixs.group_by(
                pl.col(
                        a.tolist()
                    )
                ).agg(
                    pl.col('index').implode().alias('ix')
                ).with_columns(
                    pl.col('ix').cast(
                        pl.Array(pl.Int32,
                                 (
                                     int(
                                         np.prod(
                                            list(
                                                (
                                                    self.dims[i] 
                                                    for i in (self.dims) 
                                                    if i not in ix)
                                            )
                                        )
                                    )
                                ,)
                            )
                        )
                ).collect().sort(a)['ix'].to_numpy()
    


def is_iterable(a:Any)->bool:
    try:
        iter(a)
        return True
    except:
        return False