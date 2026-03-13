import torch
import json
from vllm import LLM, SamplingParams
from unittest.mock import patch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import PreTrainedModel
# from transformers import AutoModelForCausalLM, AutoTokenizer
from .baseline import r1_zero_prompts
from .per_token_entropy import compute_entropy


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# MODEL_PATH = "/root/autodl-tmp/assignment5-alignment/models/Qwen2.5-Math-1.5B"

# data_path = "autodl-tmp/assignment5-alignment/data/sft-reason/sft_gpt-oss-120b_filtered.jsonl"
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, 
#                                              torch_dtype = torch.bfloat16, 
#                                              attn_implementation = "flash_attention_2")

# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def generation(
        model, 
        tokenizer,
        data_path: str,
        reward_fn, 
        device = 'cuda',
        ):
        model.eval()
        with torch.no_grad():
                avg_length = 0
                total_number = 0
                correct_length = 0
                correct_number = 0
                incorrect_length = 0
                incorrect_number = 0

                with open(data_path, 'r', encoding = 'utf-8') as f: 
                        prompts = json.load(f)
                for prompt in prompts[:10]: 
                        result = {}
                        # 1. input
                        problem = prompt["problem"]
                        result["prompt"] = problem

                        # 2.generation
                        prompt_id = tokenizer.encode(r1_zero_prompts(problem), return_tensors = "pt").to(device)
                        out = model.generate(
                                prompt_id,
                                return_dict_in_generate = True, 
                                output_scores = True, 
                                )
                        generation_id = out.sequences[:, prompt_id.shape[-1]:]
                        generation = tokenizer.decode(generation_id[0])

                        result["generation"] = generation
                        full_generation = "<think>" + generation + "</answer>"

                        reasoning_trace = prompt["reasoning_trace"]

                        # 3. ground_truth
                        answer= prompt["expected_answer"]
                        result["ground truth"] = answer

                        # 4 reward
                        eval_score = reward_fn(full_generation, answer)
                        result["score"] = eval_score

                        # 5.average_token_entropy
                        logits = torch.stack(out.scores)
                        avg_entropy = torch.mean(compute_entropy(logits))
                        result["average_entropy"] = avg_entropy

                        # 6. length
                        if result["score"]["reward"] == 1.0: 
                                total_number += 1
                                correct_number += 1
                                avg_length += len(result["generation"])
                                correct_length += len(result["generation"])
                        if result["score"]["reward"] == 0.0: 
                                total_number += 1
                                incorrect_number += 1
                                avg_length += len(result["generation"])
                                incorrect_length += len(result["generation"])
                
                if correct_length > 0 and incorrect_length > 0: 
                        print(f"""avg_length: {avg_length / total_number}, 
                        correct_length: {correct_length / correct_number}, 
                        incorrect_length: {incorrect_length / incorrect_number}, 
                        {result}""")

        model.train() 