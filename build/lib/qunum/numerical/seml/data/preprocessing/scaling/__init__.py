from .pds import NormScaler as PdNormScalar
from .pls import NormScaler as PlNormScalar
from .pds import NormScaler

def getPlsMapping():
    return {'Norm':PlNormScalar}