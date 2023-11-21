import torch
import torch.nn.functional as F

def greedy(logits, **kwargs):
    pred = torch.argmax(logits, dim=-1).unsqueeze(0)
    return pred

def sampling(logits, top_k, top_p, **kwargs):
    if top_k:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')

    if top_p:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('Inf')

    probs = F.softmax(logits, dim=-1)
    pred = torch.multinomial(probs, num_samples=1)
    return pred

def generate(model, input_ids, max_length, method="greedy", temperature=1.0, top_k=None, top_p=None):
    method_funcs = {"greedy": greedy, "sampling": sampling}

    if method not in method_funcs:
        raise  ValueError(f"Method not found, the methods available are {list(method_funcs.keys())}")

    kwargs = {'temperature': temperature, 'top_k': top_k, 'top_p': top_p}
    original_length = input_ids.shape[-1]
    with torch.no_grad():
        model.eval()
        kv_cache = None
        while (input_ids.shape[-1] - original_length) < max_length:
            input_ids = input_ids[-model.context_length:]
            logits, kv_cache = model(input_ids, None)
            logits = logits[:, -1, :] / temperature
            pred = method_funcs[method](logits, **kwargs)
            input_ids = torch.cat((input_ids, pred), dim=-1)
        model.train()

    return input_ids