import json
from typing import Callable, List
from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

def r1_zero_prompts(prompts: List[str]):
    """
    Turn prompts into standard r1_zero prompts.
    """
    zero_shot = [f"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {prompts}
Assistant: <think>"""]
    return zero_shot
    
def read_json(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    results = []

    total = len(prompts)
    for i, item in enumerate(prompts):
        if i % 10 == 1 : 
            print(f"loading data {i}/{total}")

        question = item["problem"]
        answer = item["expected_answer"]

        generation = vllm_model.generate(r1_zero_prompts(question), eval_sampling_params)
        generated_text = generation[0].outputs[0].text

        full_response = "<think>" + generated_text + "</answer>"

        # evaluation
        eval_score = reward_fn(full_response, str(answer))

        # save
        results.append({
            "question": question, 
            "ground_truth": answer, 
            "model_response": full_response, 
            "scores": eval_score
        })

    # serialize to disk
    with open("math_baseline.json", 'w', encoding = 'utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("results saved to math_baseline.json")



def generation():
    sampling_params = SamplingParams(
        temperature = 0.0, 
        top_p = 1.0, 
        max_tokens = 1024, 
        stop = ["</answer>"]
    )

    MODEL_PATH = "/root/autodl-tmp/assignment5-alignment/models/Qwen2.5-Math-1.5B"
    llm = LLM(model = MODEL_PATH, trust_remote_code=True)

    prompts = read_json("/root/autodl-tmp/assignment5-alignment/data/math_val.jsonl")

    evaluate_vllm(llm, r1_zero_reward_fn, prompts[:200], sampling_params)


if __name__ == "__main__":
    generation()