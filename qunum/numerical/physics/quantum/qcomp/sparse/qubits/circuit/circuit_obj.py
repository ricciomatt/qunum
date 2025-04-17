from typing import Self, Iterable, Iterator
from .....qobjs.sparse_sun import SUNMatrix, SUNState
from .....qobjs.sparse_sun.instantiate import init_sparse_basis as mk_sparse_basis
from ..gateset import *
import torch
from copy import copy
from warnings import warn, WarningMessage
class SuQCirquit:
    def __init__(
            self, 
            n_qubits:int, 
            *params:tuple, 
            psi0:None|SUNMatrix = None, 
            errorModel:None=None, 
            fidelity:tuple[float,float] = (1.,1.), 
            device:torch.device = 'cpu', 
            dtype:torch.dtype = torch.complex128,
            **kwargs:dict

        )->None:
        
        self.n_qubits:int = n_qubits

        self.q:list[Gate] = []


        self.fidelity:tuple[int,int] = fidelity
        self.device:torch.device = device
        self.dtype:torch.dtype = dtype
        self.n:int = 0
        self.errorModel = errorModel
        self.cirq_depth:int = len(self.q)
        self.check_impl()
        if(psi0 is not None):
            assert isinstance(psi0, SUNState), TypeError('psi0 must be SUNState')
            assert psi0.basis.shape[-2] == self.n_qubits, ValueError("psi0 must contain the correct number of qubits")
            self.psi0 = psi0
        else:
            self.psi0 = None
        return 
    
    def check_impl(self)->bool:
        if self.errorModel is not None:
            warn(WarningMessage('Circuits with errors not yet implemented. Error model {err} ignored'.format(err=self.errorModel), NotImplementedError))
            self.errorModel = None
        if self.fidelity != (1.,1.):
            warn(WarningMessage('Circuits with errors not yet implemented. Fidelity is set to (1.,1.)', NotImplementedError))
            self.fidelity = (1.,1.)
        return 

    def add_gate(self, Gate:QuantumGate|CustomCallableOperator):
        assert Gate.n_qubits == self.n_qubits, ValueError('Number of qubits on Gate or Operator must match the number of qubits in the circuit')
        self.q.append(Gate)
        self.cirq_depth+=1
        return

    def add_gate_from_context(self, target_qubit:int|Iterable[int], gate_tp:str, *params, **kwargs)->None:
        match gate_tp.lower():
            case gate_tp if gate_tp in ['x', 'y', 'z']:
                self.q.append(PauliGate(self.n_qubits, target=target_qubit,*params, dir=gate_tp, dtype=self.dtype, **kwargs))
            case 'rot':
                self.q.append(RotationGate(self.n_qubits, target_qubit, *params, device=self.device, **kwargs))
            case gate_tp if gate_tp in ['cx','cy','cz']:
                self.q.append(ControlledPauli(self.n_qubits, target_qubit, *params, device = self.device, dtype = self.dtype, **kwargs))
            case 's':
                self.q.append(SGate(self.n_qubits, target_qubit, *params, device=self.device, **kwargs))
            case 't':
                self.q.append(TGate(self.n_qubits, target_qubit, *params, device=self.device, **kwargs))
            case 'tofoli':
                self.q.append(ToffoliGate(self.n_qubits, target_qubit, *params, device=self.device, dtype=self.dtype, **kwargs))
            case _:
                raise NotImplementedError('Gate Operation Not Found')
        self.cirq_depth+=1
        return 

    def __call__(self)->SUNMatrix:
        psi = self.init_psi()
        return self.act_opers(psi, 0)
    
    def act_opers(self, psi:None|SUNState=None, n:int = 0)->SUNMatrix:
        gate = self.compile_gate(n)
        psi = gate @ psi
        if(n < self.cirq_depth-1):
            n+=1
            return self.act_opers(psi, n = n)
        else:
            return psi

    def init_psi(self)->SUNState:
        if(self.psi0 is None):
            return mk_sparse_basis(self.n_qubits)
        else:
            return self.psi0.clone()
    
    def add_circuit(self, circuit:Self, inplace:bool = False)->None:
        if(inplace):
            self.q.extend(circuit.q)
            self.cirq_depth = len(self.q)
            return     
        else:
            q = copy(self.q)
            q.extend(circuit.q)
            C = SuQCirquit(
                self.n_qubits, 
                psi0=self.psi0, 
                errorModel=self.errorModel, 
                fidelity=self.fidelity, 
                device=self.device, 
                dtype=self.dtype
            )
            C.q = q
            C.cirq_depth = len(q)
            return C

    def compile_gate(self, n):
        return self.q[n]()

    def __next__(self)->None:
        if(self.psi0 is None):
            self.psi = self.init_psi()
        if(self.n<len(self.q)):
            self.psi = self.compile_gate(self.n) @ self.psi
            self.n+=1
            return self.psi
        else:
            self.n = 0
            raise StopIteration
    
    def __iter__(self)->Iterator[Self]:
        return self

    def run(self):
        return self()