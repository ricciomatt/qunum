import polars as pl
from numpy.typing import NDArray
import numpy as np
from .density_operations import ptrace_torch_ix as ptrace_ix, vgc
import polars as pl 
import numpy as np
from typing import Iterable, Any
from IPython.display import display as disp, Markdown as md, Math as mt
import itertools
class QobjMeta:
    def __init__(self, 
                 n_particles:int= 1,
                 hilbert_space_dims:int = 2, 
                 shp:tuple[int] = None, 
                 dims:None|dict[int:int] = None,
                 check_hermitian:bool = False,
                 is_sparse:bool = False
                 )->None:
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
        if(check_hermitian):
            self.check_hermitian = check_hermitian
            self.herm = False
        else:
            self.check_hermitian = check_hermitian
        if(dims is None):
            dims, hilbert_space_dims, n_particles = self.infer_dims(l, n_particles, hilbert_space_dims)
        elif(self.obj_tp != 'scaler'):
            assert np.prod(list(dims.values())) == l, f'''The ProductSum(dims) must be equivalent to the number of dimensions of the hilbert space'''
        else:
            dims = dict.fromkeys(range(n_particles),0)
        self.hilbert_space_dims = l
        self.shp = shp
        self.refactor_dims(dims)
        self.eigenBasis = None 
        self.eigenVals = None
        self.is_sparse = is_sparse
        return
    
    def refactor_dims(self, dims:dict[int:int])->None:
        self.dims = dims 
        if(self.obj_tp != 'scaler'):
            self.ixs = pl.LazyFrame(itertools.product(*[range(dims[x]) for x in dims])).with_row_count()
        else:
            self.ixs = None
        self.n_particles = len(dims)
        return

    def update_dims(self, keep_ixs:Iterable, reorder:bool = False)->None:
        self.ixs = self.ixs.select(vgc(keep_ixs))
        if(reorder):
            self.dims = {i:self.dims[k] for i, k in enumerate(keep_ixs)}
            self.ixs = self.ixs.rename({f"column_{k}":f"column_{i}" for i, k in enumerate(keep_ixs)}).with_row_count()
        else:
            self.dims = {k:self.dims[k] for k in keep_ixs}
            self.ixs = self.ixs.with_row_count()
        self.n_particles = int(len(self.dims))
        self.hilbert_space_dims = np.prod(self.dims.values())
        return 
    
    def _reset_(self):
        self.eigenBasis = None
        self.eigenVals = None 
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
    
    def check_particle_ixs(self, ix:Iterable|int)->Iterable[int]:
        if(is_iterable(ix)): 
            ix_ = np.array(ix, dtype=np.int32)
        elif(isinstance(ix, int)): 
            ix_ = np.array([ix], dtype=np.int32)
        else: 
            raise TypeError('tr_out must be Iterable[int] or int')
        assert self.particle_in(ix_), ValueError('Particle Not found')
        return ix_
    
    def query_particle_ixs(self, ix:NDArray):
        a:np.ndarray = vgc(ix)
        return self.ixs.group_by(
                pl.col(
                        a.tolist()
                    )
                ).agg(
                    pl.col('row_nr').implode().alias('ix')
                ).collect().sort(a)['ix'].to_list()
    
    
class GenQobjMeta:
    def __init__(self, n_particles:int= None,
                 hilbert_space_dims:int = 2, 
                 shp:tuple[int] = None, 
                 check_hermitian:bool = False,)->None:
        if(len(shp) == 2):
            if(shp[0] == 1):
                self.obj_tp = 'bra'
                l = shp[1]
            elif(shp[1] == 1):
                self.obj_tp = 'ket'
                l = shp[0]
            else:
                self.obj_tp = 'operator'
                l = shp[1]
        elif(len(shp) == 3):
            if(shp[1] == 1):
                self.obj_tp = 'bra'
                l = shp[2]
            elif(shp[2] == 1):
                self.obj_tp = 'ket'
                l = shp[1]
            else:
                self.obj_tp = 'operator'
                l = shp[1]
        else:
            raise IndexError('Only Object of Size 2 and 3, if more than that specify hilbert axis')
            
        if(check_hermitian):
            self.check_hermitian = check_hermitian
            self.herm = False
        else:
            self.check_hermitian = check_hermitian
            
        if(hilbert_space_dims**n_particles == l):
            self.n_particles = n_particles
            self.hilbert_space_dims = hilbert_space_dims
            
        elif(hilbert_space_dims == 2):
            self.hilbert_space_dims = hilbert_space_dims
            self.n_particles = int(np.log2(l))
            from warnings import warn
            warn('Assuming that this is a 2d hilbert space')
        else:
            raise RuntimeError('Operators must have dimensions specified')
        self.ixs = pl.DataFrame(
                np.array(
                    list
                        (itertools.product(np.arange(self.hilbert_space_dims), 
                                           repeat=n_particles)
                         )
                        )
                ).with_row_count().lazy()
        return
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return '$$n_{particles}= '+str(self.n_particles)+'\\\\'+' n_{hilbert\\;dims}= '+str(self.hilbert_space_dims)+'\\\\type='+str(self.obj_tp)+'$$'
        

def is_iterable(a:Any)->bool:
    try:
        iter(a)
        return True
    except:
        return False