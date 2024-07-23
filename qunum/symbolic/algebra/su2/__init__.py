from typing import Self
from sympy import Symbol, I
class Pauli:
    def __init__(self, pauli_op:str = 'x', coefficent:Symbol|int|float = 1)->Self:
        assert pauli_op.lower() in ['i','x','y','z']
        self.pauli_op = pauli_op.lower()
        self.coefficent = coefficent
        return
    def __mul__(self, Obj:Self):
        
        assert any(isinstance(Obj, Pauli) ,isinstance(Obj, Symbol), isinstance(Obj, int), isinstance(Obj, float))
        {'ix':Pauli('x', coefficent=1), 'xi':Pauli('x', coefficent=1), }
        if(isinstance(Obj,Pauli)):
            coefficent=Obj.coefficent * self.coefficent

            match self.pauli_op:
                case 'x':
                    match Obj.pauli_op:
                        case 'i':
                            return Pauli('x', coefficent= I*coefficent)
                        case 'x':
                            return Pauli('i', coefficent=coefficent)
                        case 'y':
                            return Pauli('z', coefficent= I*coefficent)
                        case 'y':
                            return Pauli('y', coefficent= -I*coefficent)
                case 'x':
                    match Obj.pauli_op:
                        case 'i':
                            return Pauli('x', coefficent= I*coefficent)
                        case 'x':
                            return Pauli('i', coefficent=coefficent)
                        case 'y':
                            return Pauli('z', coefficent= I*coefficent)
                        case 'y':
                            return Pauli('y', coefficent= -I*coefficent)
        else:
            return 
        return
    def commute(self ):
        pass
