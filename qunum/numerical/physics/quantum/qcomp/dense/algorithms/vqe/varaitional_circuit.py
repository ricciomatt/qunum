from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators import SparsePauliOp, Operator
from qiskit.primitives import Estimator
from typing import Generator, Self
import numpy as np
from scipy.optimize import minimize

class VQEPauliOp:
    def __init__(self, AnsatzCirq:QuantumCircuit, Hamiltonian:Operator, num_parameters:int=3, p0:np.ndarray|None = None,  num_steps:int = int(1e2), method:str= 'COYBAL')->Self:
        self.num_steps = num_steps
        if(p0 is None):
            self.p = np.random.random(num_parameters)
        else:
            self.p = p0
        self.estimator:Estimator = Estimator()
        self.H = Hamiltonian
        self.AnsatzCirq = AnsatzCirq
        self.E = np.empty((self.num_steps))
        self.n = 0
        self.nresets = 0
        self. method = method
        return

    def __next__(self)->None:
        if(self.n<self.num_steps):
            min_ =  minimize(lambda x: self.estimator.run(self.AnsatzCirq, [self.H], x).result().values[0], self.p, method=self.method)
            self.E[self.n+self.num_steps*self.nresets] = (min_.fun)
            self.n+=1
            return
        else:
            raise StopIteration
        
    def __iter__(self)->Generator[None, None,None]:
        return self
    
    def __call__(self)->np.ndarray:
        return self.estimator.run(self.AnsatzCirq, [self.H], self.p).result().values
    
    def reset_iter(self)->None:
        self.n = 0
        self.nresets += 1
        self.E = np.stack((self.E, np.empty_like(self.num_steps)))
        return 
    
    def getMinE(self)->np.float32:
        return self.E.min()


def vqe_fit(parms:np.ndarray, Ansatz:QuantumCircuit, estimator:Estimator, hamiltonian:SparsePauliOp, num_steps:int= int(1e3), method:str= 'COYBAL')->Generator[tuple[np.ndarray, np.ndarray], None,None]:
    for i in range(int(num_steps)):
        min_ = minimize(lambda x: estimator.run(Ansatz, [hamiltonian], x).result().values[0], parms, method)
        E = min_.fun
        parms = min_.x
        yield E, parms

