# Qunum
## Blurb
- Qunum is a physics library for general purpose simulations, symbolic quantum mechanics, physics inspired and general purpose machine learning library implemented with torch.
## Quantum Simulations:
### TQobj
- TQobj is a general purpose Torch Quantum object for numerical simulations and capable of doing backpropegation and auto-differentiation
#### Example Euler Rotation by an angle theta :
```
from qunum.numerical import TQobj
from qunum.numerical.algebra import su
import torch

sigma_matricies = TQobj(su.get_pauli(to_tensor = True), n_particles = 1)
# Declaring a new ket state randomly evaluated at 1000 points
Psi = TQobj(torch.rand((1000, 8, 1), dtype = torch.complex128), n_particles = 3, hilbert_space_dims = 2)
Psi /= torch.sqrt((Psi.dag() @ Psi))

# making a density matrix
rho = Psi.to_density() # Psi @ Psi.dag()

# Operator
Rz = (sigma_matricies[3]*torch.pi/4).expm()

I = TQobj(torch.eye(2,2), dtype = rho.dtype)
Full_Operator = Rz^I^I

rho = Full_Operator @ rho @ Full_Operator.dag()

S0 = rho.Tr(0).entropy()
I01 = rho.mutual_info(0,1)
```
#### Differentiation

```
from qunum import qunum as qn 
from torch.autograd import grad as D
t = torch.linspace(0, 1, 1_000)
t = t.type(torch.complex128).requires_grad_(True)

Psi = qn.TQobj(torch.zeros((1000, 2, 1), dtype = torch.complex128))

Psi[:,0,0] = torch.sin(t)
Psi[:, 1, 0] = torch.cos(t)

D(Psi[:, 1, 0] , t, grad_outputs=torch.ones_like(t), retain_graph = True, create_graph=False)

```