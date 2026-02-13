import os
import json
import numpy as np
from pathlib import Path
import gc

from tokenizer import Tokenizer

def preprocess():
    me = Path(__file__).resolve()
    # 假设项目结构没变
    project_root = me.parent.parent.parent

    data_dir = project_root / "data"
    train_txt = data_dir / "owt_train.txt"
    valid_txt = data_dir / "owt_valid.txt"

    vocab_path = "run_bpe_train_on_owt_output/vocab.json"
    merges_path = "run_bpe_train_on_owt_output/merges.txt"
    
    if not train_txt.exists():
        print(f"错误: 找不到训练文件 {train_txt}")
        return

    # --- 1. 加载 Tokenizer ---
    print(f"加载 Tokenizer...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw_vocab = json.load(f)
    
    # 转换回字节格式
    vocab = {int(k): bytes(v) for k, v in raw_vocab.items()}

    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                p1 = bytes(map(int, parts[0].split(',')))
                p2 = bytes(map(int, parts[1].split(',')))
                merges.append((p1, p2))
    
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    print(f"Tokenizer 加载完成，词表大小: {len(tokenizer.byte_to_id)}")

    # --- 2. 处理文本任务 ---
    tasks = [
        ("owt_valid", valid_txt),
        ("owt_train", train_txt)
    ]

    for name, path in tasks:
        print(f"\n正在处理 {name}...")
        
        # 临时二进制文件路径
        temp_bin = data_dir / f"{name}.tmp.bin"
        output_npy = data_dir / f"{name}.npy"
        
        token_count = 0
        
        # 核心修改：流式读取文本，并立即以二进制形式写入硬盘，不占用 Python 列表内存
        with open(path, "r", encoding="utf-8") as f, open(temp_bin, "wb") as f_out:
            chunk_size = 50 * 1024 * 1024  # 每次读 50MB 文本
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                # 编码当前块
                chunk_ids = tokenizer.encode(chunk)
                # 使用 uint32 (4字节) 存储，对于 5w 词表足够了，节省内存
                ids_array = np.array(chunk_ids, dtype=np.uint32)
                
                # 直接将二进制数据写入文件
                f_out.write(ids_array.tobytes())
                
                token_count += len(chunk_ids)
                print(f"已处理 Token 数: {token_count:,}", end='\r')

        print(f"\n{name} 编码完成。正在转换为最终的 .npy 格式...")

        # --- 3. 使用 Memory-mapped 将二进制转为标准 .npy ---
        # 这种方式不会占用物理内存，而是直接映射到磁盘文件
        mmap_array = np.lib.format.open_memmap(
            output_npy, 
            mode='w+', 
            dtype=np.uint32, 
            shape=(token_count,)
        )
        
        # 分段从临时文件读入 memmap
        with open(temp_bin, "rb") as f_temp:
            offset = 0
            read_batch = 10 * 1024 * 1024  # 每次搬运 1000 万个 token
            while offset < token_count:
                num_to_read = min(read_batch, token_count - offset)
                # 读取二进制数据并转回 numpy 数组
                chunk_data = np.frombuffer(f_temp.read(num_to_read * 4), dtype=np.uint32)
                mmap_array[offset : offset + num_to_read] = chunk_data
                offset += num_to_read
                print(f"正在保存: {offset/token_count*100:.1f}%", end='\r')
        
        # 确保数据刷入磁盘并清理
        mmap_array.flush()
        del mmap_array
        if temp_bin.exists():
            os.remove(temp_bin)
            
        print(f"\n✅ {name} 保存成功！路径: {output_npy}")
        gc.collect()

if __name__ == "__main__":
    preprocess()