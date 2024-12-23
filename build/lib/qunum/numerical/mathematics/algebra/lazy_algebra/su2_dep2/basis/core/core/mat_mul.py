from typing import Self, Iterable
from sympy import Symbol
import numpy as np 
import polars as pl
from .......combintorix import EnumerateArgCombos
from .......tensors import levi_cevita_tensor
class MatrixBasis:
    def __init__(self,):
        self.Data:pl.LazyFrame= pl.LazyFrame()
        pass

class MatMul:
    def __init__(self):
        e = levi_cevita_tensor(3)
        Mapping={
            f'I{i}':(dict(basisNew=i, coef_realNew = 1.0, coef_imagNew=0.0))
            for i in ['I', 'X', 'Y','Z']
        }
        for i, j in enumerate(['X','Y','Z']):
            Mapping.update(
                {
                    (f"{j}{m}"):(
                        (dict(
                            basisNew=j,
                            coef_realNew = 1.0, 
                            coef_imagNew=0.0
                        )) if(m == 'I') 
                        else 
                       (dict(
                           basisNew='I', 
                           coef_realNew = 1.0, 
                           coef_imagNew=0.0
                       )) if(j == m)
                        else 
                       (dict(
                            basisNew=['X',"Y","Z"][np.where(np.abs(e[i,n-1]))[0][0]],
                            coef_realNew = 0.0,
                            coef_imagNew = e[i,n-1,np.where(np.abs(e[i,n-1]))[0][0]]
                        ) )
                    )
                    for n,m in enumerate(['I', 'X', 'Y','Z'])
                }
            )

        self.Data = pl.LazyFrame(
            EnumerateArgCombos(['I', 'X', 'Y', 'Z'], ['I', "X", "Y", 'Z'], ret_tensor = False).__iter__(), 
            schema = {'column_0':pl.String, 'column_1':pl.String}
        ).rename({'column_0':'basis', 'column_1':'basis(1)'}).with_columns(
            pl.concat_str('basis', 'basis(1)').replace(Mapping, return_dtype=pl.Struct({'basisNew':pl.String, 'coef_realNew':pl.Float64, 'coef_imagNew':pl.Float64})).alias('res')
        ).unnest('res')
        return
    
    def __call__(self, O1:MatrixBasis, O2:MatrixBasis, lazy_ops:bool = True)->MatrixBasis:
        from ..basis import  MatrixBasis as PM
        Data = O1.Data.join(
            O2.Data,  on='particle_index',how = 'outer',suffix = '(1)'
        ).with_columns(
            pl.when(
                pl.col('particle_index').is_null().not_()
            ).then(
                pl.col('particle_index')
            ).otherwise(
                pl.col('particle_index(1)')
            ).alias('particle_index')
        ).drop('particle_index(1)').with_columns(
            pl.col('basis','basis(1)').fill_null('I'), 
            pl.col('coef_imag','coef_imag(1)').fill_null(0.0),
            pl.col('coef_real','coef_real(1)').fill_null(1.0)
        ).join(
            self.Data, 
            on=['basis','basis(1)'], 
            how='inner'
        ).select(
            ['particle_index', 'basisNew', 'function_', 'coef_real', 'coef_imag','coef_real(1)', 'coef_imag(1)', 'coef_realNew', 'coef_imagNew']
        ).with_columns(
            (
                pl.col('coef_real')*pl.col('coef_real(1)')*pl.col('coef_realNew')
                    -
                pl.col('coef_imag')*pl.col('coef_imag(1)')*pl.col('coef_realNew')
                    -
                pl.col('coef_imag')*pl.col('coef_real(1)')*pl.col('coef_imagNew')
                    -
                pl.col('coef_real')*pl.col('coef_imag(1)')*pl.col('coef_imagNew')
            ).alias('coef_real'),
            (
                pl.col('coef_imag')*pl.col('coef_real(1)')*pl.col('coef_realNew')
                    +
                pl.col('coef_real')*pl.col('coef_imag(1)')*pl.col('coef_realNew')
                    +
                pl.col('coef_real')*pl.col('coef_real(1)')*pl.col('coef_imagNew')
                    -
                pl.col('coef_imag')*pl.col('coef_imag(1)')*pl.col('coef_imagNew')
            ).alias('coef_imag')
        ).select(
            ['particle_index', 'basisNew', 'function_', 'coef_real', 'coef_imag']
        ).rename(
            {'basisNew':'basis'}
        ).group_by(
            ['particle_index', 'basis','function_']
        ).agg(
            pl.col('coef_real').sum(),
            pl.col('coef_imag').sum()
        ).select(
            ['particle_index', 'basis', 'function_', 'coef_real', 'coef_imag']
        )
        if not (lazy_ops):
            Data = Data.collect().lazy()
        return PM(set_direct= Data.clone().lazy(), lazy_ops=lazy_ops)