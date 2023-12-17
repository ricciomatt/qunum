import polars as pl
from numpy.typing import NDArray
import warnings
import numpy as np
import torch
from ..operators.density_operations import ptrace_torch_ix as ptrace_ix, vgc
import polars as pl 
import numpy as np
import warnings
from IPython.display import display as disp, Markdown as md, Math as mt
import itertools
class QobjMeta:
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
           
            warnings.warn('Assuming that this is a 2d hilbert space')
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
           
            warnings.warn('Assuming that this is a 2d hilbert space')
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
        
