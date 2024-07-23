from ..pauli_matrix import PauliMatrix
from typing import Self, Generator, Iterable
import polars as pl
import numpy as np 
import torch

class JordanWignerPauliMatrix:
    def __init__(self, N:int )->Self:
        self.N = N
        return
    def __iter__(self, nix:Iterable[int]|None = None, create_or_annhilate:str = 'annhilate')->Generator[PauliMatrix, None, None]:
        assert create_or_annhilate in ['create', 'annhilate'], ValueError("Keyword create_or_annhilate must be in the {'create', 'annhilate'}")
        def yield_args(nix:np.ndarray[np.int32], N:int, create_or_annhilate:str):
            match create_or_annhilate:
                case 'annhilate':
                    v = -1.0/2.0
                case 'create':
                    v = 1.0/2.0
            for i in nix:
                a = pl.LazyFrame(
                    dict(
                        particle_index = [i,i], 
                        function_=[None, None],
                        basis=['X','Y'],
                        coef_real = [1.0/2.0,0.0],
                        coef_imag=[0.0, v]
                    ),
                    schema = dict(
                        particle_index = pl.Int32, 
                        function_ = pl.Null,
                        basis = pl.String,
                        coef_real = pl.Float64,
                        coef_imag=pl.Float64
                    )
                )
                D = pl.LazyFrame(
                    dict(
                        particle_index = pl.arange(i+1,N, eager = True),    
                    ), 
                    schema = dict(
                        particle_index = pl.Int32
                        )
                ).with_columns(
                    basis = pl.lit('Z'),
                    coef_real = 1.0,
                    coef_imag = 0.0,
                    function_=None,
                )
                yield PauliMatrix(set_direct=pl.concat((
                    a.select(['particle_index', 'basis','function_', 'coef_real', 'coef_imag']),
                    D.select(['particle_index', 'basis','function_', 'coef_real', 'coef_imag'])
                )).collect()[['particle_index', 'basis','function_', 'coef_real', 'coef_imag']].lazy())
        if(nix is None):
            nix = range(0, self.N)
        return yield_args(nix, self.N, create_or_annhilate=create_or_annhilate)
    
    def __list__(self, create_or_annhilate:str = 'annhilate')->list[PauliMatrix]:
        return list(self.__iter__(create_or_annhilate=create_or_annhilate))

    def __call__(self, n:np.ndarray[np.int32]|int|None = None, create_or_annhilate:str = 'annhilate')->np.ndarray[PauliMatrix]|PauliMatrix:
        assert  (
                    (isinstance(n,int)) or 
                    (isinstance(n, np.ndarray) and n.dtype in [np.int16, np.int32, np.int64, np.int128, np.int8, np.int_]) or
                    (isinstance(n, torch.Tensor) and n.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]) or
                    (isinstance(n, list)) or
                    n is None
                ), TypeError(f'Key Word argument n must be less than {self.N} ie in range[0, {self.N-1}]')
        if(isinstance(n, int)):
            assert n<self.N and n >=0, ValueError('Key Word argument n must be an integer or iterable less than {self.N} ie in range[0, {self.N-1}]')
            n = np.array([int(n)])
            return next(self.__iter__(nix=n, create_or_annhilate=create_or_annhilate))
        assert is_iter(n) and n.max()<self.N and n.min()>=0, ValueError(f'Key Word argument n must be an integer or iterable less than {self.N} ie in range[0, {self.N-1}]')
        return np.fromiter(self.__iter__(nix=n, create_or_annhilate=create_or_annhilate))
             
    
    def __array__(self, create_or_annhilate:str = 'annhilate')->np.ndarray[PauliMatrix]:
        return np.fromiter(self.__iter__(create_or_annhilate=create_or_annhilate), dtype = np.object_)
        
def is_iter(a):
    try:
        iter(a)
        return True
    except:
        return False

