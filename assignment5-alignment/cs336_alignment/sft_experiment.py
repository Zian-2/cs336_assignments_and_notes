import os
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from vllm import LLM, SamplingParams
from tqdm import tqdm
import wandb

from .tokenize_prompt import tokenize_prompt_and_output
from .log_probs import get_response_log_probs
from .sft_microbatch_train_step import sft_microbatch_train_step
from .log_generations import init_vllm, load_policy_into_vllm_instance
from .drgrpo_grader import r1_zero_reward_fn, extract_answer
from .baseline import r1_zero_prompts

MODEL_PATH = "/root/autodl-tmp/assignment5-alignment/models/Qwen2.5-Math-1.5B"
DATA_DIR = "/root/autodl-tmp/assignment5-alignment/data/sft-reason/sft_gpt-oss-120b.json"

class MathDataset(Dataset):
    def __init__(self, data_path, max_size=None):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        if max_size and max_size < len(self.data):
            self.data = self.data[:max_size]
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]

def collate(batch, tokenizer):
    prompts = [x["problem"] for x in batch]
    responses = [x["reasoning_trace"] for x in batch]
    return tokenize_prompt_and_output(prompts, responses, tokenizer)

def evaluate(model, vllm_llm, val_data):
    subset = val_data[:10] 
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256, stop=["</answer>"]) # 缩短 max_tokens
    load_policy_into_vllm_instance(model, vllm_llm)
    correct = 0
    for item in tqdm(val_data, desc="Evaluating"):
        question = item["problem"]
        gt = item["reasoning_trace"]
        answer = gt.split("<answer>")[-1].split("</answer>")[0].strip() if "<think>" in gt and "<answer>" in gt else (extract_answer(gt) or gt)
        prompt = r1_zero_prompts(question)
        out = vllm_llm.generate(prompt, sampling_params)
        response = "<think>" + out[0].outputs[0].text + "</answer>"
        if r1_zero_reward_fn(response, answer)["reward"] == 1.0:
            correct += 1
    return correct / len(val_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--filtered", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    # Test mode: small data, 1 epoch
    if args.test:
        args.dataset_size = 64
        epochs = 1
    else:
        epochs = 1
    
    print(f"=== SFT: size={args.dataset_size}, lr={args.lr}, bs={args.batch_size}, epochs={epochs} ===")
    print("Loading model & data...")

    os.environ["WANDB_API_KEY"] = "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"
    run_name = f"sft_{args.dataset_size or 'full'}_lr{args.lr}_bs{args.batch_size}" + ("_filtered" if args.filtered else "")
    wandb.init(project="sft_math", name=run_name, config={
        "dataset_size": args.dataset_size,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": epochs,
    })
    wandb.define_metric("train_step"); wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    
    device = "cuda:0"
    vllm_device = "cuda:1"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=device)
    vllm_llm = init_vllm(MODEL_PATH, vllm_device, 42, gpu_memory_utilization=0.6)
    
    train_data = MathDataset(f"{DATA_DIR}", args.dataset_size)
    val_data = MathDataset(f"{DATA_DIR}", 20)
    
    if args.filtered:
        print("Filtering to correct answers...")
        sampling_params = SamplingParams(temperature=0.0, max_tokens=1024, stop=["</answer>"])
        load_policy_into_vllm_instance(model, vllm_llm)
        filtered = []
        for item in tqdm(train_data, desc="Filtering"):
            prompt = r1_zero_prompts(item["problem"])
            out = vllm_llm.generate(prompt, sampling_params)
            response = "<think>" + out[0].outputs[0].text + "</answer>"
            answer = item["reasoning_trace"].split("<answer>")[-1].split("</answer>")[0].strip()
            if r1_zero_reward_fn(response, answer)["reward"] == 1.0:
                filtered.append(item)
        train_data = filtered
        print(f"Filtered size: {len(train_data)}")
        wandb.log({"filtered_size": len(train_data)})
    
    batch_size = args.batch_size
    lr = args.lr
    grad_accum = 1
    
    train_loader = DataLoader(train_data, batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1*len(train_loader)*epochs/grad_accum), int(len(train_loader)*epochs/grad_accum))
    
    global_step = 0
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)
            

            log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"]
            loss, _ = sft_microbatch_train_step(log_probs, response_mask, grad_accum, response_mask.sum().float())
            
            if (global_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            
            wandb.log({"train_step": global_step // grad_accum, "train/loss": loss.item()})
            global_step += 1

            if global_step % 10 == 0:
                acc = evaluate(model, vllm_llm, val_data)
                wandb.log({"eval_step": global_step, "eval/accuracy": acc})
        
        acc = evaluate(model, vllm_llm, val_data)
        print(f"Epoch {epoch+1}: Val Acc = {acc:.4f}")
    
    wandb.finish()

if __name__ == "__main__":
    main()
