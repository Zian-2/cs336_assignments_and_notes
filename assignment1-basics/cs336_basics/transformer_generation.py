import torch
import torch.nn.functional as F


def top_p_sampling(p, top_p):
    sorted_p, sorted_indices = torch.sort(p, descending = True)
    
    cumulative_p = torch.cumsum(sorted_p, dim =-1)

    mask = cumulative_p > top_p
    mask[..., 1:] = mask[..., :-1].clone() 
    mask[..., 0] = False

    sorted_p[mask] = 0.0
    new_p = torch.zeros_like(p)
    new_p.scatter_(-1, sorted_indices, sorted_p)
    
    new_p /= torch.sum(new_p, axis = -1, keepdim = True)

    return new_p

@torch.no_grad()
def generate(model, prompt_tokens, max_new_tokens, temperature=1.0, top_p=1.0, eos_token_id=None, rope_theta=10000.0):
    model.eval()
    device = next(model.parameters()).device
    
    x = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    if x.ndim == 1:
        x = x.unsqueeze(0) # (1, Seq_len)
    
    batch_size = x.size(0)
    is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):

        logits = model(x, rope_theta)
        next_token_logits = logits[:, -1, :]

        if temperature != 1.0:
            next_token_logits /= temperature
        p = F.softmax(next_token_logits, dim = -1)

        if top_p != 1.0:
            p = top_p_sampling(p, top_p)

        next_id_tensor = torch.multinomial(p, num_samples = 1)

        x = torch.cat([x, next_id_tensor], dim=1)
        
        if eos_token_id is not None:
            is_finished |= (next_id_tensor.squeeze(1) == eos_token_id)
        if is_finished.all():
            break

    return x

    