import os
import math
import torch
import torch.nn as nn
import torch
import numpy as np
import numpy.typing as npt
from typing import Iterable, BinaryIO, IO
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


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:

    max_idx = len(dataset) - context_length - 1

    ix = torch.randint(0, max_idx + 1, (batch_size,))

    x_list = []
    y_list = []
    
    for i in ix:
        i = i.item()
        x_list.append(torch.from_numpy(dataset[i : i + context_length].astype(np.int64)))
        y_list.append(torch.from_numpy(dataset[i + 1 : i + context_length + 1].astype(np.int64)))

    x = torch.stack(x_list)
    y = torch.stack(y_list)

    return x.to(device), y.to(device)



def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }

    torch.save(checkpoint, out)



def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:

    checkpoint = torch.load(src, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['iteration']