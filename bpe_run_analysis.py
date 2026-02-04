import sys
import os
import time
import json
import numpy as np
from pathlib import Path

# 1. 动态添加路径，确保能导入你的 tokenizer
# 指向包含 cs336_basics 的父目录
sys.path.append(os.path.join(os.getcwd(), "assignment1-basics"))

from cs336_basics.tokenizer import Tokenizer

def load_tokenizer(vocab_path, merges_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    # 假设你的 merges 是文本格式，且键值对由空格分隔
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p0, p1 = line.strip().split()
                merges.append((p0.encode("utf-8"), p1.encode("utf-8")))
    # 注意：这里的 vocab 里的 key 如果是字符串，需要转为 bytes
    byte_vocab = {int(v): k.encode("utf-8") for k, v in vocab.items()}
    return Tokenizer(byte_vocab, merges)

def main():
    # 路径配置
    data_path = Path("data/TinyStoriesV2-GPT4-valid.txt")
    vocab_file = Path(r"D:\cs336_assignments_and_notes\assignment1-basics\run_bpe_train_on_tinystories_output\vocab.json")
    merges_file = Path(r"D:\cs336_assignments_and_notes\assignment1-basics\run_bpe_train_on_tinystories_output\merges.txt")
    
    print(f"--- 正在加载分词器 ---")
    tokenizer = load_tokenizer(vocab_file, merges_file)
    
    if not data_path.exists():
        print(f"错误: 找不到数据文件 {data_path}")
        return

    print(f"--- 正在读取并处理: {data_path.name} ---")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    raw_bytes = len(text.encode("utf-8"))
    
    # 测速开始
    start_time = time.perf_counter()
    ids = tokenizer.encode(text)
    end_time = time.perf_counter()
    
    # 计算指标
    duration = end_time - start_time
    num_tokens = len(ids)
    compression_ratio = raw_bytes / num_tokens
    throughput = (raw_bytes / 1024 / 1024) / duration
    
    print(f"\n[结果统计]")
    print(f"- 原始大小: {raw_bytes / 1024:.2f} KB")
    print(f"- Token 数量: {num_tokens}")
    print(f"- 压缩比 (Bytes/Token): {compression_ratio:.2f}")
    print(f"- 耗时: {duration:.4f} 秒")
    print(f"- 吞吐量: {throughput:.2f} MB/s")
    
    # 估算 Pile (825GB)
    pile_time_hrs = (825 * 1024) / throughput / 3600
    print(f"- 估算处理 Pile 数据集所需时间: {pile_time_hrs:.2f} 小时")

    # 序列化为 uint16
    out_path = "tinystories_valid.npy"
    np.save(out_path, np.array(ids, dtype=np.uint16))
    print(f"\n--- 已保存序列化文件至: {out_path} ---")

if __name__ == "__main__":
    main()