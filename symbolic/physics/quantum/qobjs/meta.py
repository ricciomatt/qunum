from sympy import Matrix, eye, sqrt
import numpy as np
import torch
import numba as nb
import warnings 
import polars as pl
from numpy.typing import NDArray


class OperMeta:
    def __init__(self, n_particles:int= None, hilbert_space_dims:int = 2, shp:tuple[int] = None)->None:
        warnings.warn('Operator is deprecated use the more general SQobj instead')
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
    
    

class SQobjMeta:
    def __init__(self, n_particles:int= None, hilbert_space_dims:int = 2, shp:tuple[int] = None, check_hermitian:bool = False)->None:
        if(len(shp) == 2):
            if(shp[0] == 1):
                self.obj_tp = 'bra'
                l = shp[1]
            elif(shp[1] == 1):
                self.obj_tp = 'ket'
                l = shp[0]
            else:
                self.obj_tp = 'operator'
                l = shp[0]
        elif(len(shp) == 3):
            if(shp[2] == 1):
                self.obj_tp = 'ket'
                l = shp[1]
            elif(shp[1] == 1):
                self.obj_tp = 'bra'
                l = shp[2]
            else:
                self.obj_tp = 'operator'
                l = shp[1]
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
            self.n_particles = np.log2(l)
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