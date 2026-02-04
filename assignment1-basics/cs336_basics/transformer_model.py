import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__() 
        
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        
        sigma = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return einsum(x, self.W, "... din, dout din -> ... dout")
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        
        self.W = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        
        rms = torch.sqrt(torch.mean(x_f32**2, dim=-1, keepdim=True) + self.eps)

        x_normed = (x_f32 / rms) * self.g
        
        return x_normed.to(in_dtype)