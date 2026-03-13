import torch
from transformers import PreTrainedModel
import torch.nn.functional as F
from .per_token_entropy import compute_entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Args:
    model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
        and in inference mode if gradients should not be computed).
    input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
        response tokens as produced by your tokenization method.
    labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your
        tokenization method.
    return_token_entropy: bool If True, also return per-token entropy by calling
        compute_entropy.

    Returns:
    dict[str, torch.Tensor].
        "log_probs" shape (batch_size, sequence_length), conditional log-probabilities
            log pθ(xt | x<t).
        "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
            for each position (present only if return_token_entropy=True).
    """
    device = model.device

    labels = labels.to(device)
    input_ids = input_ids.to(device)
    model.train()

    logits = model(input_ids).logits

    logp = F.log_softmax(logits, dim = -1)
    logp_label = torch.gather(logp, dim = -1, index = labels.unsqueeze(-1)).squeeze(-1)

    res = {"log_probs": logp_label}
    if return_token_entropy is True: 
        res["token_entropy"] = compute_entropy(logits)
    return res
