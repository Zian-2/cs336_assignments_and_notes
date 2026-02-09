import torch
import math


from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # 获取学习率。
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # 获取与 p 相关的状态。
                t = state.get("t", 0) # 从状态中获取迭代次数，或初始值。
                grad = p.grad.data # 获取损失相对于 p 的梯度。
                p.data -= lr / math.sqrt(t + 1) * grad # 就地更新权重张量。
                state["t"] = t + 1 # 增加迭代次数。
        return loss

def tune_lr():
    # 题目要求的三个值
    lrs = [10.0, 100.0, 1000.0]
    
    for lr in lrs:
        print(f"\n===== Testing LR: {lr} =====")
        # 初始化权重：一定要每次循环都重新初始化，保证公平
        weights = torch.nn.Parameter(torch.tensor([5.0])) 
        opt = SGD([weights], lr=lr)
        
        for t in range(10): # 仅 10 次迭代
            opt.zero_grad()
            # 损失函数：w^2
            loss = weights**2 
            loss.backward()
            
            # 先打印当前损失，再更新
            print(f"Step {t}: Loss = {loss.item():.2e}")
            
            opt.step()

if __name__ == "__main__":
    tune_lr()