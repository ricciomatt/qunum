
from typing import Any, Callable
import torch

from ..linear import custmom_layers as cl
from torch import tensor

class ChannelAttention_CBAM(torch.nn.Module):
    def __init__(self, 
                 num_channels:int, 
                 channel_ratio:int = 2, 
                 attention_out:Callable = torch.nn.Sigmoid(),
                 ):
        super(ChannelAttention_CBAM, self).__init__()
        self.max_pool_1 = torch.nn.AdaptiveMaxPool2d(1)
        self.avg_pool_1 = torch.nn.AdaptiveAvgPool2d(1)
        print(num_channels, channel_ratio, num_channels//channel_ratio, )
        self.linear_encoder = torch.nn.Sequential(
            torch.nn.Linear(num_channels, int(num_channels//channel_ratio), bias=False),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(int(num_channels//channel_ratio), num_channels, bias=False)
            )
        self.paritition = attention_out
        
    def __call__(self, x:torch.tensor) -> torch.tensor:
        return self.forward(x)
    
    def forward(self, x:torch.tensor) -> torch.tensor:
        weight_x = self.paritition(self.linear_encoder(self.max_pool_1(x).squeeze(-1).squeeze(-1)) + 
                                   self.linear_encoder(self.avg_pool_1(x).squeeze(-1).squeeze(-1))
                                   ).unsqueeze(-1).unsqueeze(-1)
        return weight_x*x
        
class SpatialAttention_CBAM(torch.nn.Module):
    def __init__(self, 
                 kernal_size:tuple[int,int] = (2,2),
                 attention_out:Callable = torch.nn.Sigmoid()):
        super(SpatialAttention_CBAM, self).__init__()
        self.sequence = torch.nn.Sequential(
            torch.nn.Conv2d(2, 1, kernel_size=kernal_size, padding='same', bias=False ),
            attention_out
        )
        
    def __call__(self, x:torch.tensor) -> torch.tensor:
        return self.forward(x)
    def forward(self, x:torch.tensor) -> torch.tensor:
        weight_x = self.sequence(torch.cat([torch.mean(x,dim=1,keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1))
        return weight_x*x
            
class CBAMAttention(torch.nn.Module):
    def __init__(self, 
                 inp_size:tuple[int,int,int], 
                 channel_ratio:int,
                 spatial_kernal:tuple[int,int], 
                 channel_attention_out:Callable = torch.nn.Sigmoid(),
                 spatial_attention_out:Callable = torch.nn.Sigmoid()
                 ):
        super(CBAMAttention, self).__init__()
        self.CBAM_Apply = torch.nn.Sequential(
            ChannelAttention_CBAM(inp_size[0], channel_ratio=channel_ratio, attention_out=channel_attention_out),
            SpatialAttention_CBAM(kernal_size=spatial_kernal, attention_out=spatial_attention_out)     
        )
    
    def __call__(self, x:torch.tensor)->torch.tensor:
        return self.forward(x)
    
    def forward(self, x:torch.tensor)->torch.tensor:
        return self.CBAM_Apply(x)
    
class ChannelAttentionHadamard(torch.nn.Module):
    def __init__(self, 
                 inp_size:tuple[int,int,int],
                 channel_ratio:float = 2.0,
                 attention_out:Callable = torch.nn.Softmax(1)
                 ) -> torch.nn.Module:
        
        super(ChannelAttentionHadamard, self).__init__()
        
        self.hadamard = torch.nn.Sequential(
            cl.HadamardLayer(inp_size),
            torch.nn.LeakyReLU(),
            )
        self.avg = torch.nn.AdaptiveAvgPool2d(1)
        self.max = torch.nn.AdaptiveAvgPool2d(1)
        
        self.out_seq = torch.nn.Sequential(
            torch.nn.Linear(inp_size[0], int(inp_size[0]//channel_ratio)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(inp_size[0]//channel_ratio), inp_size[0]),
            attention_out
        )
    def __call__(self,x):
        return self.forward(x)
    def forward(self, x):
        xt = self.hadamard(x)
        xt = self.out_seq(self.avg(xt).squeeze(-1).squeeze(-1) + self.max(xt).squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
        return x*xt
                
            

class SpatialAttentionHadamard(torch.nn.Module):
    def __init__(self, 
                 inp_size:tuple[int,int,int], 
                 kernal_size:tuple[int,int] = (2,2),
                 attention_out:Callable = torch.nn.Softmax2d()) -> torch.nn.Module:
        super(SpatialAttentionHadamard, self).__init__()
        self.hadamard = torch.nn.Sequential(
            cl.HadamardLayer(inp_size),
            torch.nn.LeakyReLU(),
            )
        self.out_seq = torch.nn.Sequential(
            torch.nn.Conv2d(2, 1, kernel_size=kernal_size, padding='same', bias=False ),
            attention_out
        )
        
    def __call__(self,x):
        return self.forward(x)
    def forward(self, x):
        xt = self.hadamard(x)
        xt = torch.cat([torch.mean(xt,dim=1,keepdim=True), torch.max(xt, dim=1, keepdim=True)[0]], dim=1)
        xt = self.out_seq(xt)
        return x*xt


class HadamardAttention(torch.nn.Module):
    def __init__(self,
                 inp_size:tuple[int,int,int], 
                 channel_ratio:int,
                 spatial_kernal:tuple[int,int], 
                 channel_attention_out:Callable = torch.nn.Sigmoid(),
                 spatial_attention_out:Callable = torch.nn.Sigmoid()
                 ):
        super(HadamardAttention, self).__init__()
        self.ApplyAttention = torch.nn.Sequential(
            ChannelAttentionHadamard(inp_size, channel_ratio=channel_ratio, attention_out=channel_attention_out),
            SpatialAttentionHadamard(inp_size, kernal_size=spatial_kernal, attention_out=spatial_attention_out)
        )
    
    def __call__(self, x:tensor)->tensor:
        return self.forward(x)
    
    def forward(self, x:tensor)->tensor:
        return self.ApplyAttention(x)