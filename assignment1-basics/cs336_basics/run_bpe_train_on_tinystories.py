import os
import json
import psutil
import cProfile
import pstats
import time
from tokenizer import train_bpe

def main():
    # 路径配置
    input_path = "../../data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 50257 
    special_tokens = ["<|endoftext|>"] 

    process = psutil.Process(os.getpid())

    print("开始训练 (Profiling Mode)...")
    
    # --- 启动 Profile ---
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_real_time = time.time()

    # 调用训练函数
    tokenizer = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_chunks=20
    )

    # --- 停止 Profile ---
    profiler.disable()
    end_real_time = time.time()

    print("\n" + "="*50)
    print("性能分析报告 (Top 20 Functions):")


    # 打印最耗时的函数
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(20) 
    print("="*50 + "\n")

    # 训练统计
    print(f"训练完成，耗时: {end_real_time - start_real_time:.2f} 秒")
    print(f"词表大小: {len(tokenizer.vocab)}")

    # 内存峰值统计 (Windows peak_wset)
    mem_info = process.memory_info()
    peak_mem_gb = getattr(mem_info, 'peak_wset', mem_info.rss) / (1024 ** 3)
    print(f"内存峰值: {peak_mem_gb:.4f} GB")

    # 最长 Token 统计
    longest_token_bytes = max(tokenizer.vocab.values(), key=len)
    print(f"最长 Token 内容: '{longest_token_bytes.decode('utf-8', errors='replace')}'")
    print(f"最长 Token 字节长度: {len(longest_token_bytes)}")

    # 保存 profile 原始数据到文件，可以用 snakeviz 等工具可视化
    stats.dump_stats("bpe_training_profile.prof")

    output_dir = "run_bpe_train_on_tinystories_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    readable_vocab = {int(k): list(v) for k, v in tokenizer.vocab.items()}
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(readable_vocab, f, indent=4)

    merges_path = os.path.join(output_dir, "merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        for p1, p2 in tokenizer.merges:
            # 将 bytes 转换为逗号分隔的数字字符串
            s1 = ",".join(map(str, list(p1)))
            s2 = ",".join(map(str, list(p2)))
            f.write(f"{s1} {s2}\n")

if __name__ == "__main__":
    main()

