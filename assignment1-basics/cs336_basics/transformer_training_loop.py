import os
import torch
import numpy as np
import argparse

# 导入你提供的两个模块
from transformer_training import (
    AdamW, 
    lr_cosine_schedule, 
    gradient_clipping, 
    get_batch, 
    save_checkpoint, 
    load_checkpoint,
    cross_entropy
)
from transformer_model import TransformerLM

def train():
    parser = argparse.ArgumentParser(description="Transformer 训练脚本")
    
    parser.add_argument("--data_path", type=str, required=True, help="训练数据 .npy 文件路径")
    parser.add_argument("--val_data_path", type=str, help="验证数据 .npy 文件路径")
    parser.add_argument("--out_dir", type=str, default="out", help="检查点保存目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # TransformerLM 的 __init__
    parser.add_argument("--vocab_size", type=int, default=50257, help="词表大小 (GPT-2 默认 50257)")
    parser.add_argument("--num_layers", type=int, default=12, help="Transformer 层数")
    parser.add_argument("--num_heads", type=int, default=12, help="多头注意力的头数")
    parser.add_argument("--d_model", type=int, default=768, help="嵌入维度/模型宽度")
    parser.add_argument("--d_ff", type=int, default=2048, help="FFN 中间层维度")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE 的基础频率 theta")
    
    # 超参数
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--max_iters", type=int, default=10000)
    
    # 优化器与调度
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # 辅助参数
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--resume", action="store_true", help="是否从最新检查点恢复")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 数据加载
    train_data = np.load(args.data_path, mmap_mode='r')
    val_data = np.load(args.val_data_path, mmap_mode='r') if args.val_data_path else None

    # 实例化 TransformerLM
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff
    )
    model.to(args.device)

    # AdamW
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 断点续传
    start_iter = 0
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        # load_checkpoint
        start_iter = load_checkpoint(ckpt_path, model, optimizer)
        print(f"从迭代步数 {start_iter} 恢复训练")

    # loop
    model.train()
    for it in range(start_iter, args.max_iters):
        
        # 余弦退火
        curr_lr = lr_cosine_schedule(
            it, args.lr, args.min_lr, args.warmup_iters, args.max_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr

        # 采样数据批次
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)

        # 前向传播
        logits = model(x, theta=args.rope_theta)

        # loss
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # 更新
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 梯度裁剪
        if args.grad_clip != 0.0:
            gradient_clipping(model.parameters(), args.grad_clip)

        # 优化器步进
        optimizer.step()

        # 打印日志
        if it % args.log_interval == 0:
            print(f"迭代 {it}: 损失 {loss.item():.4f}, 学习率 {curr_lr:.2e}")

        # val
        if it > 0 and it % args.eval_interval == 0 and val_data is not None:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, args.batch_size, args.context_length, args.device)
                v_logits = model(vx, theta=args.rope_theta)
                v_loss = cross_entropy(v_logits.view(-1, v_logits.size(-1)), vy.view(-1))
                print(f"步数 {it}: 损失 {v_loss.item():.4f}")
            model.train()

        # save
        if it > 0 and it % args.save_interval == 0:
            save_checkpoint(model, optimizer, it, ckpt_path)
            print(f"检查点已保存至 {ckpt_path}")

if __name__ == "__main__":
    train()