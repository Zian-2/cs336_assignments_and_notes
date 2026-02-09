import math
import torch
import torch.nn as nn
from typing import Iterable
from einops import rearrange, einsum


def cross_entropy(inputs, targets):

    inputs_stable = inputs - torch.max(inputs, dim = -1, keepdim = True)[0]
    range = torch.arange(inputs.shape[0])
    entropy = - inputs_stable[range, targets] + torch.log(torch.sum(torch.exp(inputs_stable), dim = -1)) 
    l = torch.mean(entropy)

    return l

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)


    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["t"] += 1
                beta1, beta2 = group["betas"]
                lr = group["lr"]
                wd = group["weight_decay"]
                eps = group["eps"]
                t = state["t"]

                m = state["m"]
                v = state["v"]

                m *= beta1 
                m += (1 - beta1) * grad
                v *= beta2
                v += (1 - beta2) * grad**2
                
                alpha_t = lr * (math.sqrt(1 - beta2**t) / (1 - beta1**t))
                
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * wd * p.data

        return loss
    
def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return max_learning_rate* it / warmup_iters
    if warmup_iters <= it <= cosine_cycle_iters:
       cos = math.cos((it - warmup_iters)*math.pi/(cosine_cycle_iters - warmup_iters))
       return min_learning_rate + (1+cos) * (max_learning_rate - min_learning_rate) / 2
    if it > cosine_cycle_iters:
        return min_learning_rate
    

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    epsilon = 1e-6
    total_norm = 0.0

    parameters = [p for p in parameters if p.grad is not None]

    with torch.no_grad():
        for p in parameters:
            param_norm = torch.norm(p.grad, 2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        if total_norm > max_l2_norm:
            scaling_factor = max_l2_norm / (total_norm + epsilon)

            for p in parameters:
                p.grad.mul_(scaling_factor)
