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
Psi = TQobj(torch.rand((1000, 8, 1), dtype = torch.complex128), n_particles = 3)
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
import torch
from qunum import qunum as qn 
t = torch.linspace(0, 1, 1_000)
t = t.type(torch.complex128).requires_grad_(True)

Psi = qn.TQobj(torch.zeros((1000, 2, 1), dtype = torch.complex128))

Psi[:,0,0] = torch.sin(t)
Psi[:, 1, 0] = torch.cos(t)
H = 1j * qn.Dx(Psi, t, der_dim = 0)

```

#### Trotter Expansion
```
import torch as torch
from qunum import qunum as qn
from qunum.jupyter_tools.plotting import *
setup_plotly()
Sx = qn.TQobj(qn.algebra.representations.su.get_pauli(to_tensor=True)[1], n_particles = 1)
B = 1e1
H = Sx*B
t = torch.linspace(0, 3, 1000, requires_grad=True)
e1, e0 = qn.TQobj([[0],[1]], dtype = torch.complex128), qn.TQobj(torch.tensor([[1],[0]], dtype = torch.complex128))
p = e0 @ e0.dag()
U = qn.einsum('A, ij->Aij',t,(-1j*H)).expm() 
p = U.dag() @ p @ U
fig = go.Figure()
fig.add_trace(go.Scatter(y=p[:,0,0].detach().real, x = t.detach().real, name='$P_{\\left|0\\right>}$'))
fig.add_trace(go.Scatter(y=p[:,1,1].detach().real, x = t.detach().real, name='$P_{\\left|1\\right>}$'))
fig.update_xaxes(title = 'Time(s)')
fig.update_yaxes(title = 'Probability')
fig.update_layout(title = 'Probability of State', height = 500, width= 1000)
iplot(fig)
```
![Graph of probability](image.png)