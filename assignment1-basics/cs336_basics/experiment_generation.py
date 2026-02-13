import torch
import json
import os
from transformer_model import TransformerLM
from transformer_generation import generate
from tokenizer import Tokenizer # 记得确保 tokenizer.py 里的导入去掉了那个“.”

def run_experiment():
    # --- 1. 路径与配置 ---
    checkpoint_path = "out_owt_v1/final_model.pt"
    vocab_path = "run_bpe_train_on_owt_output/vocab.json"
    merges_path = "run_bpe_train_on_owt_output/merges.txt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 架构参数 (请根据你训练时的设置微调)
    # 注意：vocab_size 必须是你训练模型时的那个值 (比如之前脚本里写的 50257)
    # 而不是训练 BPE 脚本里的 10000。这两个要匹配！
    model_args = {
        "vocab_size": 50257, 
        "context_length": 1024,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "rope_theta": 10000.0
    }

    # --- 2. 针对性加载 Tokenizer (手动构建以匹配你的保存格式) ---
    print(f"reading voc: {vocab_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw_vocab = json.load(f)
    
    # 关键修复：你的保存脚本存的是 list(bytes)，这里必须转回 bytes
    vocab = {}
    for k, v in raw_vocab.items():
        # v 是 [226, 130, 158] 这种 list
        vocab[int(k)] = bytes(v)

    print(f"reading mg: {merges_path}")
    merges = []
    if os.path.exists(merges_path):
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    # 匹配你的保存格式：用逗号分割再转 bytes
                    p1 = bytes(map(int, parts[0].split(',')))
                    p2 = bytes(map(int, parts[1].split(',')))
                    merges.append((p1, p2))

    # 初始化 Tokenizer
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    eos_id = tokenizer.byte_to_id.get(b"<|endoftext|>", None)

    # --- 3. 加载模型 ---
    print(f"checkpoint: {checkpoint_path}")
    model = TransformerLM(**model_args)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()



    # --- 4. 生成 ---
    prompts = [
        "Once upon a time, there was a little girl named Sue."
    ]

    print("\n" + "="*50)
    for p_text in prompts:
        print(f"\n[Prompt]: {p_text}")
        prompt_tokens = tokenizer.encode(p_text)
        
        output_ids = generate(
            model, 
            prompt_tokens, 
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.9,
            eos_token_id=eos_id,
            rope_theta=model_args["rope_theta"]
        )
        
        # 将 tensor 转为 list 交给你的 decode
        decoded_text = tokenizer.decode(output_ids[0].tolist())
        print(f"[Generated]:\n{decoded_text}\n")
        print("-" * 30)

if __name__ == "__main__":
    run_experiment()