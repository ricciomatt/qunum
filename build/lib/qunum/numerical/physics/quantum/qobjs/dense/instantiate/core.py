
from typing import Iterable
def extract_shape(obj_tp:str, l:int, shp:Iterable[int]|int|None)->list[int]:
    if(shp is None):
        shp = []
    elif(isinstance(shp,int)):
        shp = [shp]
    elif not (isinstance(shp, list)):
        shp = list(shp)
    assert isinstance(shp, list), TypeError('Shape must be Iterable[int] or int')
    match obj_tp:
        case 'ket':
            shp.extend([l,1])
        case 'bra':
            shp.extend([1,l])
        case 'operator':
            shp.extend([l,l])
        case 'scaler':
            shp.extend([1,1])
    return tuple(shp)

