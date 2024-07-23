from typing import Self
import numpy as np 
import polars as pl
from .core import MatMul, add_pauliMatricies, sub_pauliMatricies
class Functions:
    def __init__(self):
        return 
    
    def __mul__(self):
        return
    
    def conjugate(self):
        return

    def __call__(self):
        return getattr(self)


class PauliMatrix:
    def __init__(
            self, 
            basis:np.ndarray[str] = np.array(['I'], dtype= np.object_), 
            particle_index:np.ndarray[np.int32] = np.array([0], dtype= np.int32),
            coef:np.ndarray[np.float_]|np.ndarray[np.complex_]|np.ndarray[np.int_] = np.array([1.0], dtype = np.complex128),
            lazy_ops:bool = False,
            set_direct:pl.DataFrame|pl.LazyFrame|None = None,
            is_zero:bool = False,
            **kwargs,
        )->Self:
        assert isinstance(set_direct, pl.LazyFrame) or set_direct is None, 'set_direct Must be LazyFrame or None' 
        if('MulOper' not in globals()):
            global MulOper 
            MulOper = MatMul()
        self.lazy_ops = lazy_ops
        self.is_zero = is_zero

        if(set_direct is None):
            self.Data:pl.LazyFrame = pl.LazyFrame(
            dict(
                particle_index = particle_index,
                basis = basis, 
                function_ = [None for ix in particle_index], 
                coef_real = coef.real, 
                coef_imag = coef.imag, 
            ), 
            schema = dict(
                particle_index=pl.Int32,
                basis = pl.String, 
                function_=pl.Null, 
                coef_real=pl.Float64, 
                coef_imag=pl.Float64, 
            )
            )
        else:
            self.Data :pl.LazyFrame = set_direct
        
        return
    def zero(self)->None:
        self.Data = pl.LazyFrame(
                dict(
                    particle_index = [0],
                    basis = ['I'], 
                    function_ = [None], 
                    coef_real = [0.0], 
                    coef_imag = [0.0], 
                ), 
                schema = dict(
                    particle_index=pl.Int32,
                    basis = pl.String, 
                    function_=pl.Null, 
                    coef_real=pl.Float64, 
                    coef_imag=pl.Float64, 
            )
        )
        self.is_zero = True
        return

    def __mul__(self, coef:int|float|complex)->Self:
        assert any([isinstance(coef, t) for t in [float, int, complex]]), 'Must be int|float|complex type'
        if(isinstance(coef, int) or isinstance(float, int)):
            return self.Data.with_columns((pl.col('coef_real')*float(coef)), ((pl.col('coef_imag')*float(coef))))
        else:
            return self.Data.with_columns((pl.col('coef_real')*(coef).real - pl.col('coef_imag')*(coef).imag), ((pl.col('coef_real')*coef.imag + pl.col('coef_imag')*coef.real)))
   
    def __rmul__(self, coef:int|float|complex)->Self:
        return self.__mul__(coef)
    
    
    def __matmul__(self, Op:Self)->Self:
        assert isinstance(Op, PauliMatrix), 'Must be a Pauli Expression'
        if(self.is_zero or Op.is_zero):
            return PauliMatrix().zero()
        return MulOper(self, Op, lazy_ops = self.lazy_ops)
    
    def __rmatmul__(self,Op:Self)->Self:
        assert isinstance(Op, PauliMatrix), 'Must be a Pauli Expression'
        return MulOper(Op, self, lazy_ops = self.lazy_ops)

    def __radd__(self, Op:Self)->Self:
        assert isinstance(Op, PauliMatrix), 'Must be a Pauli Expression'
        return add_pauliMatricies(Op, self, lazy_ops = self.lazy_ops)
     
    def dag(self, doit:bool =False, inplace:bool=False)->None|Self:
        Data = self.Data.with_columns(
            (-1*pl.col('coef_imag')).alias('coef_imag')
        )
        if(inplace):
            self.Data = Data
            if(doit):
                self.doit(ck_z= False)
            return
        else:
            return PauliMatrix(set_direct=Data.collect().lazy())
    
    def transpose(self, doit:bool=False, inplace:bool = False)->None|Self:
        Data = self.Data.with_columns(
                pl.when(
                    pl.col('basis')=='Y'
                ).then(
                    pl.struct(
                        coef_real = -1*pl.col('coef_real'), 
                        coef_imag = -1*pl.col('coef_imag')
                    )
                ).otherwise(
                    pl.struct(
                        coef_real = pl.col('coef_real'), 
                        coef_imag = pl.col('coef_imag')
                    )
                ).alias('res')
            ).select(
                ['particle_index', 'basis', 'function_', 'res']
            ).unnest('res').select(
                ['particle_index', 'basis', 'function_', 'coef_real', 'coef_imag']
            )
        if(inplace):
            self.Data = Data
            if(doit):
                self.doit(ck_z= False)
            return
        else:
            return PauliMatrix(set_direct=Data.collect().lazy())
        


    def __add__(self, Op:Self)->Self:
        assert isinstance(Op, PauliMatrix), 'Must be a Pauli Expression'
        return add_pauliMatricies(self, Op, lazy_ops = self.lazy_ops)
    
    def __sub__(self, Op:Self)->Self:
        assert isinstance(Op, PauliMatrix), 'Must be a Pauli Expression'
        return sub_pauliMatricies(self, Op, lazy_ops = self.lazy_ops)
    
    def __rsub__(self, Op:Self)->Self:
        assert isinstance(Op, PauliMatrix), 'Must be a Pauli Expression'
        return sub_pauliMatricies(Op, self, lazy_ops= self.lazy_ops)

    def doit(self, inplace:bool  = True, ck_z:bool = True):
        Data = self.Data.collect()
        if(ck_z):
            shp = self.check_zero(Data=Data)
            if(shp):
                self.is_zero = True
                if(inplace):
                    self.zero()
                else:
                    P = PauliMatrix()
                    P.zero()
                    return P
                return
            else:
                Data = Data.filter(
                    (
                        (pl.col('coef_real')==0.0) & 
                        (pl.col('coef_imag')==0.0)
                    ).not_()
                )
        if(inplace):
            self.Data = Data.lazy()
            return 
        else:
            P = PauliMatrix(set_direct=Data.lazy(), lazy_ops= self.lazy_ops)
            return P
    
    def check_zero(self, Data:None|pl.DataFrame = None, mut_to_zero:bool = False):
        if(Data is not None):
            shp = bool(
                Data.filter(
                pl.col('particle_index').is_in(
                    Data.filter(
                (pl.col('coef_real')== 0.0) 
                    &
                (pl.col('coef_real')== 0.0)
                ).select(
                    pl.col('particle_index')
                )
            )
            ).group_by(
                ['particle_index', 'basis']).agg(
                pl.col('coef_real').sum(), 
                pl.col('coef_imag').sum()
            ).group_by(['particle_index']).agg(
                pl.col('coef_real').cast(pl.Boolean).not_().all(), 
                pl.col('coef_imag').cast(pl.Boolean).not_().all()
            ).filter(
                pl.col('coef_real') & pl.col('coef_imag')
            ).shape[0])
            
        else:
            shp = bool(
                self.Data.filter(
                pl.col('particle_index').is_in(
                    self.Data.filter(
                (pl.col('coef_real')== 0.0) 
                    &
                (pl.col('coef_real')== 0.0)
                ).select(
                    pl.col('particle_index')
                ).collect()
            )
            ).group_by(
                ['particle_index', 'basis']).agg(
                pl.col('coef_real').sum(), 
                pl.col('coef_imag').sum()
            ).group_by(['particle_index']).agg(
                pl.col('coef_real').cast(pl.Boolean).not_().all(), 
                pl.col('coef_imag').cast(pl.Boolean).not_().all()
            ).filter(
                pl.col('coef_real') & pl.col('coef_imag')
            ).collect().shape[0])
        if(mut_to_zero and shp):
            self.zero()
        return shp

    def __str__(self)->str:
        def sign(u:float, v:float):
            if(abs(u)>1_000 or abs(u)<1e-2):
                ufmt = '.2e'
            else:
                ufmt = '.2f'
            if(abs(v)>1_000 or abs(v)<1e-2):
                vfmt = '.2e'
            else:
                vfmt = '.2f'
            match (u,v):
                case (u,v) if u!=0 and v>0:
                    return (('{u:'+ufmt+'}+{v:'+vfmt+'}i').format(u=u,v=abs(v)))
                case (u,v) if u!=0 and v<0:
                    return (('{u:'+ufmt+'}-{v:'+vfmt+'}i').format(u=u,v=abs(v)))
                case (u,v) if u!=0 and v==0:
                    return (('{u:'+ufmt+'}').format(u=u))
                case (u,v) if u==0 and v!=0:
                    return (('{v:'+vfmt+'}i').format(v=v))
                case (u,v) if u==0 and v==0:
                    return "0.0"
        self.check_zero(mut_to_zero=True)
        if(self.is_zero):
            return '$$ 0 $$'
        disp = self.Data.sort(
            'particle_index', 
            descending=True).fetch(11).group_by('particle_index').agg(
            pl.struct(basis= pl.col('basis'), function_ = pl.col('function_'), coef_real = pl.col('coef_real'), coef_imag = pl.col('coef_imag')).implode().alias('struct')
        )
        if(disp.shape[0]>10):
            a = '\\right\\} \otimes \\hdots $$'
        else:
            a = '\\right\\}$$'
        return str('$$\\left\\{' + "\\right\\}\\otimes\\left\\{".join(
            ["+".join([
                "({sgn})\\sigma_{{{basis},{idx}}}".format(
                    idx=row['particle_index'],sgn = sign(struct['coef_real'], struct['coef_imag']), **struct
                )  
                for struct in row['struct'][0]]
            ) for row in disp.iter_rows(named = True)])+a )

    def __repr__(self)->str:
        return self.__str__()