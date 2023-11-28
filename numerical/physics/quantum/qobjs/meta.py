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

class QobjMeta:
    def __init__(self, n_particles:int= None, hilbert_space_dims:int = 2, shp:tuple[int] = None, check_hermitian:bool = False)->None:
        if(len(shp) == 2):
            if(shp[0] == 1):
                self.obj_tp = 'bra'
            elif(shp[1] == 1):
                self.obj_tp = 'ket'
            else:
                self.obj_tp = 'operator'
        elif(len(shp) == 3):
            if(shp[1] == 1):
                self.obj_tp = 'ket'
            elif(shp[2] == 1):
                self.obj_tp = 'bra'
            else:
                self.obj_tp = 'operator'
        else:
            raise IndexError('Only Object of Size 2 and 3')
        if(check_hermitian):
            self.check_hermitian = check_hermitian
            self.herm = False
        else:
            self.check_hermitian = check_hermitian
            
        if(hilbert_space_dims**n_particles == shp[0]):
            self.n_particles = n_particles
            self.hilbert_space_dims = hilbert_space_dims
            self.ixs = pl.DataFrame(
                np.array(
                    np.meshgrid(
                        *[np.arange(hilbert_space_dims)]*n_particles)
                    ).T.reshape(
                    -1, 
                    n_particles
                    )).with_row_count().lazy()
        elif(hilbert_space_dims == 2):
            self.hilbert_space_dims = hilbert_space_dims
            self.n_particles = n_particles
            self.ixs = pl.DataFrame(
                np.array(
                    np.meshgrid(
                        *[np.arange(hilbert_space_dims)]*n_particles)
                    ).T.reshape(
                    -1, 
                    n_particles
                    )).with_row_count().lazy()
            warnings.warn('Assuming that this is a 2d hilbert space')
        else:
            raise RuntimeError('Operators must have dimensions specified')
        return    
    def __str__(self):
        return mt('$$n_{particles}= '+str(self.n_particles)+'\\\\'+' n_{hilbert\\;dims}= '+str(self.hilbert_space_dims)+'\\\\type='+str(self.obj_tp)+'$$')
        
