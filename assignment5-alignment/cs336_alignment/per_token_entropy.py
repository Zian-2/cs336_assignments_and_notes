import torch

def compute_entropy(logits: torch.Tensor) ->torch.Tensor:
    """
    Args:logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
    containing unnormalized logits.
    Returns:torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
    prediction.

    """
    logsumexp = torch.logsumexp(logits, dim=-1)
    p = torch.softmax(logits, dim = -1)
    return logsumexp - torch.sum(p * logits, dim = -1)