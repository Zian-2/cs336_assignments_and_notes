import torch
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

MODEL_PATH = "/root/autodl-tmp/assignment5-alignment/models/Qwen2.5-Math-1.5B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def tokenize_prompt_and_output(
        prompt_strs: list[str], 
        output_strs: list[str], 
        tokenizer
) -> dict[str, torch.Tensor]:

    all_input_ids = []
    all_labels = []
    all_response_masks = []

    for prompt_str, output_str in zip(prompt_strs, output_strs): 
        
        # tokenizing
        prompt_ids = torch.tensor(tokenizer.encode(prompt_str))
        output_ids = torch.tensor(tokenizer.encode(output_str))

        # concatenating
        prompt_and_output = torch.cat([prompt_ids, output_ids])

        all_input_ids.append(prompt_and_output)
        all_labels.append(prompt_and_output[1:])
        all_response_masks.append((torch.arange(len(prompt_and_output)-1) >= len(prompt_ids)-1 ))

    result = {}
    input_ids = pad_sequence(all_input_ids, batch_first = True, padding_value = tokenizer.eos_token_id)
    result["input_ids"] = input_ids[:, :-1]
    result["labels"] = pad_sequence(all_labels, batch_first = True, padding_value = tokenizer.eos_token_id)
    result["response_mask"] = pad_sequence(all_response_masks, batch_first = True, padding_value = False)


    return result