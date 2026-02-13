import math
import torch
import torch.nn as nn
from einops import rearrange, einsum
import torch.nn.functional as F


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

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff = None, device=None, dtype=None):
        super().__init__()

        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = 64 * ((d_ff + 63) // 64)

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x3 = self.w3(x)

        SiLU = x1* torch.sigmoid(x1)
        ffn = self.w2(SiLU * x3)

        
        return ffn
    

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k

        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))

        t = torch.arange(max_seq_len, device=device).float() 
        angles = torch.outer(t, freqs)

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions].unsqueeze(0).unsqueeze(1)
        sin = self.sin[token_positions].unsqueeze(0).unsqueeze(1)

        x_even = x[...,0::2]
        x_odd = x[..., 1::2]

        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        
        return out

    

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    e_x = torch.exp(x - x_max)
    return e_x / torch.sum(e_x, dim=dim, keepdim=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))

    p = softmax(scores, dim = -1)

    output = einsum(p, V, "... q k , ... k v -> ... q v")
    return output


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_k = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False, device=device)
        self.o_proj = nn.Linear(d_model, d_model, device=device, bias=False)

        self.rope_cache = None


    def forward(self, x: torch.Tensor, rope_encoder=None, token_positions = None):
        B, T, _ = x.shape
        
        qkv = self.qkv_proj(x) 
        q, k, v = rearrange(qkv, "b t (qkv h d) -> qkv b h t d", qkv=3, h=self.num_heads)

        if rope_encoder is not None:
            q = rope_encoder(q, token_positions)
            k = rope_encoder(k, token_positions)
        # Causal Mask
        # mask = torch.ones((T, T), device=x.device, dtype=torch.bool)
        # mask = torch.tril(mask) 

        # values = scaled_dot_product_attention(q, k, v, mask=mask)
        values = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = rearrange(values, "... h n d -> ... n (h d)")
        return self.o_proj(out)
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        self.ffn = SwiGLU(d_model, d_ff=d_ff, device=device, dtype=dtype) 

    def forward(self, x: torch.Tensor, rope_encoder, positions: torch.Tensor):
        # ln + residual
        residual = x
        x = self.ln1(x)

        # mha(with rope) + residual
        x = self.attn(x, rope_encoder, token_positions=positions)
        x = residual + x

        # ln + ffn + residual
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x
    
        # # ln + residual
        # residual = x

        # # mha(with rope) + residual
        # x = self.attn(x, rope_encoder, token_positions=positions)
        # x = residual + x

        # x = self.ln1(x)

        # # ln + ffn + residual
        # residual = x

        # x = self.ffn(x)

        # x = residual + x
        # x = self.ln2(x)
        
        # return x
    

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int,device=None, dtype=None, rope_theta=10000.0):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.rope_encoder = RotaryPositionalEmbedding(
            theta=rope_theta, 
            d_k=d_model // num_heads, 
            max_seq_len=context_length, 
            device=device
        )

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) 
            for _ in range(num_layers)
        ])
        
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        

    def forward(self, in_indices: torch.Tensor):

        seq_len = in_indices.shape[1]
        positions = torch.arange(seq_len, device=in_indices.device)

        # embedding
        x = self.token_embeddings(in_indices)
        
        # transformer
        for layer in self.layers:
            x = layer(x, self.rope_encoder, positions=positions)
        
        # ln_final
        x = self.ln_final(x)
        
        # affine
        
        logits = self.lm_head(x)
        
        return logits
    
        # fake_logits = torch.randn(x.shape[0], x.shape[1], 50257, 
        #                          device=x.device, requires_grad=True)
        # return fake_logits