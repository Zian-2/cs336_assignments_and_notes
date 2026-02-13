import os
import time
import wandb
import torch
import numpy as np
import argparse
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

    parser.add_argument("--use_wandb", action="store_true", help="是否启动 W&B 记录")
    
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

    from torch.amp import autocast, GradScaler
    scaler = GradScaler(device='cuda')

    if args.use_wandb:
        wandb.init(
            project="cs336-assignment1",
            name=f"lr_{args.lr}_dff_{args.d_ff}",
            config=vars(args)
        )
    
    if "cuda" in args.device:
        torch.set_float32_matmul_precision('high')
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
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    )
    model.to(args.device)

    # AdamW
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True)

    model.to(args.device)

    # 断点续传
    start_iter = 0
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        # load_checkpoint
        start_iter = load_checkpoint(ckpt_path, model, optimizer)
        model.to(args.device)
        print(f"从迭代步数 {start_iter} 恢复训练")


    training_start_time = time.time()


    # loop
    model.train()

    dt_data, dt_forward, dt_backward = 0, 0, 0
    

    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=5, warmup=2, active=4, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profile_report'),
    record_shapes=True,
    with_stack=True
    ) as prof:
        
        lr_lookup = [lr_cosine_schedule(
                i, args.lr, args.min_lr, args.warmup_iters, args.max_iters
            ) for i in range(args.max_iters)]

        for it in range(start_iter, args.max_iters):
            t_start = time.time()

            # 余弦退火
            curr_lr = lr_lookup[it]

            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr

            # 采样数据批次
            t0 = time.time()
            x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)

            optimizer.zero_grad(set_to_none=True)
            
            # 前向传播
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))


            # 更新
            loss.backward()

            # 梯度裁剪
            if args.grad_clip != 0.0:
                gradient_clipping(model.parameters(), args.grad_clip)

            # 优化器步进
            optimizer.step()

            prof.step() 

            # log
            if it % args.log_interval == 0:
                print(f"step {it:4d} | loss: {loss.item():.4f} | lr: {curr_lr:.2e} | ")
                
                if args.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": curr_lr,
                        "global_step": it
                    })
                

            


            # 重点：第 10 步强制退出，开始分析日志
            # if it >= 10:
            #     print("Profiler 采样完成，正在生成分析表格...")
            #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
            #     break
            # val
            if it > 0 and it % args.eval_interval == 0 and val_data is not None:
                model.eval()
                # 1. 验证集也要用 autocast，保持算子模式一致
                with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    vx, vy = get_batch(val_data, args.batch_size, args.context_length, args.device)
                    v_logits = model(vx)
                    # 2. 避免使用 view，改用 reshape 处理潜在的非连续显存
                    v_loss = cross_entropy(v_logits.reshape(-1, v_logits.size(-1)), vy.reshape(-1))
                    print(f"step {it}: val loss {v_loss.item():.4f}")

                    if args.use_wandb:
                        wandb.log({"val/loss": v_loss.item(), "global_step": it})
                
                # 3. 极其重要：显式清理验证产生的显存占用
                del vx, vy, v_logits, v_loss

            # save
            if it > 0 and it % args.save_interval == 0:
                save_checkpoint(model, optimizer, it, ckpt_path)
                print(f"checkpoint saved to {ckpt_path}")

        final_path = os.path.join(args.out_dir, "final_model.pt")
        save_checkpoint(model, optimizer, args.max_iters, final_path)
        print(f"training completed, with final_model saved to {final_path}")

    if args.use_wandb:
        wandb.finish()



if __name__ == "__main__":
    train()