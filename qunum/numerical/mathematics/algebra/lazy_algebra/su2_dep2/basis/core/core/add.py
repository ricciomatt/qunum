
from .basis_mirror import MatrixBasis
import polars as pl
def add_pauliMatricies(O1:MatrixBasis,O2:MatrixBasis, lazy_ops:bool = True)->MatrixBasis:
    from ..basis import  MatrixBasis as PM
    Data = O1.Data.join(
        O2.Data, on= ['particle_index', 'basis'], how='outer', suffix="(1)"
    )
    Data = Data.with_columns(
        pl.when(pl.col('particle_index').is_null()).then(pl.col('particle_index(1)')).otherwise(pl.col('particle_index')).alias('particle_index'),
        pl.when(
            pl.col('basis') == pl.col('basis(1)')
        ).then(
            pl.struct(basis = pl.col('basis'), coef_real = pl.col('coef_real')+pl.col('coef_real(1)'), coef_imag = pl.col('coef_imag')+pl.col('coef_imag(1)'))
        ).otherwise(
            pl.when(
                pl.col('basis').is_null().not_()
            ).then(
                pl.struct(basis = pl.col('basis'), coef_real = pl.col('coef_real'), coef_imag = pl.col('coef_imag'))
            ).otherwise(
                pl.struct(basis = pl.col('basis(1)'), coef_real = pl.col('coef_real(1)'), coef_imag = pl.col('coef_imag(1)'))
            )
        ).alias('Data')
    ).select(
        ['particle_index', 'function_', 'Data']
    ).unnest('Data').select(
        [
            'particle_index',
             'basis', 
             'function_', 
             'coef_real',
             'coef_imag'
        ]
    )
    return PM(set_direct=Data.clone().lazy(), lazy_ops=lazy_ops)

def sub_pauliMatricies(O1:MatrixBasis,O2:MatrixBasis, lazy_ops:bool = True)->MatrixBasis:
    from ..basis import  MatrixBasis as PM
    Data = O1.Data.join(
        O2.Data, on= ['particle_index', 'basis'], how='outer', suffix="(1)"
    )
    Data = Data.with_columns(
        pl.when(pl.col('particle_index').is_null()).then(pl.col('particle_index(1)')).otherwise(pl.col('particle_index')).alias('particle_index'),
        pl.when(
            pl.col('basis') == pl.col('basis(1)')
        ).then(
            pl.struct(basis = pl.col('basis'), coef_real = pl.col('coef_real')-pl.col('coef_real(1)'), coef_imag = pl.col('coef_imag')-pl.col('coef_imag(1)'))
        ).otherwise(
            pl.when(
                pl.col('basis').is_null().not_()
            ).then(
                pl.struct(basis = pl.col('basis'), coef_real = pl.col('coef_real'), coef_imag = pl.col('coef_imag'))
            ).otherwise(
                pl.struct(basis = pl.col('basis(1)'), coef_real = pl.col('coef_real(1)'), coef_imag = pl.col('coef_imag(1)'))
            )
        ).alias('Data')
    ).select(
        ['particle_index', 'function_', 'Data']
    ).unnest('Data').select(
        [
            'particle_index',
             'basis', 
             'function_', 
             'coef_real',
             'coef_imag'
        ]
    )

    return PM(set_direct=Data.clone().lazy(), lazy_ops=lazy_ops)