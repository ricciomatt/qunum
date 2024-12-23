from .....qobjs.dense.core.torch_qobj import TQobj
from qiskit.circuit import QuantumCircuit


def euler_rotation()->TQobj:
    pass 

def euler_circuit(nqubits:int, alpha:float = 0.0, beta:float = 0.0, gamma:float = 0.0,**kwargs)->QuantumCircuit:
    Qc = QuantumCircuit(nqubits)
    for n in range(nqubits):
        Qc.rz(alpha, n)
        Qc.ry(beta, n)
        Qc.rz(gamma, n)
    pass

def wignerdMatrix(alpha:float = 0.0, beta:float = 0.0, gamma:float = 0.0,**kwargs)->None:
    pass 