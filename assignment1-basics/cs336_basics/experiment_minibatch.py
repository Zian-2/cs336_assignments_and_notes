import torch
import numpy as np
from transformer_model import TransformerLM
from transformer_training import AdamW, cross_entropy, get_batch

def overfit_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 同样的 17M 配置
    model = TransformerLM(
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=5e-4) # 稍微大一点的学习率
    
    # 2. 加载数据并固定一个 Batch
    data = np.load("../../data/tinystories_valid.npy", mmap_mode='r')
    x, y = get_batch(data, batch_size=4, context_length=256, device=device)
    
    print(f"过拟合测试 (Device: {device})...")
    
    model.train()
    for i in range(1000):
        logits = model(x, theta=10000.0)
        
        # View 
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Step {i:3d} | Loss: {loss.item():.6f}")
            
        if loss.item() < 1e-4:
            print(f"成功")
            break

if __name__ == "__main__":
    overfit_test()