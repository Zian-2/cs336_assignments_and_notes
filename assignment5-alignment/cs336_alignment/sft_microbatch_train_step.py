import torch
from .masked_normalize import masked_normalize

def sft_microbatch_train_step( 
	policy_log_probs: torch.Tensor, 
	response_mask: torch.Tensor, 
	gradient_accumulation_steps: int, 
	normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: 
    
    negative_logp = -policy_log_probs

    total_mask_sum = response_mask.sum()
    loss = masked_normalize(
            negative_logp, 
            response_mask, 
            normalize_constant=total_mask_sum, 
            dim=None 
            )

    loss = loss / gradient_accumulation_steps
    loss.backward()

    metadata = {"loss": loss.detach()}
    return loss, metadata
