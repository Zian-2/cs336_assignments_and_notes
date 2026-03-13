import torch

def masked_normalize(
        tensor: torch.tensor, 
        mask: torch.tensor, 
        normalize_constant: float, 
        dim: int | None = None,
) -> torch.tensor: 
    mask = mask.to(tensor.dtype)
    masked_tensor = tensor * mask
    summed = torch.sum(masked_tensor, dim = dim) if dim is not None else torch.sum(masked_tensor)

    return summed / normalize_constant