import polars as pl
from numpy.typing import NDArray
import warnings
import numpy as np


class OperMeta:
    def __init__(self, n_particles:int= None, hilbert_space_dims:int = 2, shp:tuple[int] = None)->None:
        if(hilbert_space_dims**n_particles == shp[0]):
            self.n_particles = n_particles
            self.hilber_space_dims = hilbert_space_dims
            self.ixs = pl.DataFrame(
                np.array(
                    np.meshgrid(
                        *[np.arange(hilbert_space_dims)]*n_particles)
                    ).T.reshape(
                    -1, 
                    n_particles
                    )).with_row_count().lazy()
        elif(hilbert_space_dims == 2):
            self.hilber_space_dims = hilbert_space_dims
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