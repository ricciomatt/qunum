from .....qobjs import TQobj

class Variational:
    def __init__(self, Psi:TQobj, HamiltonianOperator:TQobj)->None:
        self.Psi:TQobj = Psi
        pass

    def doVary(self)->None:
        pass

    def transpileCircuit(self)->TQobj:
        pass


