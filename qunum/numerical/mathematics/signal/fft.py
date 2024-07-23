import torch
def torch_fft1d(inp_tensor:torch.Tensor, f_or_i = 1)->torch.Tensor:
    n = torch.arange(inp_tensor.shape)
    return torch.exp(torch.tensor(f_or_i)*torch.complex(0,torch.einsum('A,B->AB', n, n)*2*torch.pi)) @ inp_tensor.to(torch.complex128)
