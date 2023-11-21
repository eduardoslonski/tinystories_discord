import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def build_rope_cache(config, base=10_000):
    d = config.n_embed
    theta = 1. / (base ** (torch.arange(0, d, 2)/d)).view(1, -1)
    seq_idx = torch.arange(config.context_length).view(-1, 1)
    m_theta = seq_idx * theta
    m_theta = m_theta.repeat(1, 2).to(config.device)
    cos = m_theta.cos()
    sin = m_theta.sin()
    return cos, sin

def apply_rope(x, cos, sin):
    d = x.shape[-1]
    neg_half = torch.cat([-x[..., d//2:], x[..., :d//2]], dim=-1).to(x.device)
    x_rope = x * cos + neg_half * sin
    return x_rope
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.n_embed, config.n_embed * 4)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(config.n_embed * 4, config.n_embed)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.query = nn.Linear(config.n_embed, config.n_embed)
        self.key = nn.Linear(config.n_embed, config.n_embed)
        self.value = nn.Linear(config.n_embed, config.n_embed)
    
        self.register_buffer("attn_mask", torch.tril(torch.ones(config.context_length, config.context_length)).unsqueeze(0))
    
        self.linear = nn.Linear(config.n_embed, config.n_embed)

    def forward(self, x, kv_cache=None, rope_cache=None):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        if kv_cache:
            k_cache, v_cache = kv_cache
            k = torch.cat((k_cache, k), dim=1)
            v = torch.cat((v_cache, v), dim=1)
            self.attn_mask = torch.ones(1, 1, k.shape[-2]).to(self.config.device)
        
        kv_cache_new = [k, v]

        cos, sin = rope_cache

        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])

        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        T = k.shape[-2]
        k = k.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        v = v.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        attn_scores = attn_scores.masked_fill(self.attn_mask[:, :x.shape[1], :x.shape[1]] == 0, float('-inf'))
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_values = attn_scores @ v
        attn_values = attn_values.transpose(1, 2).contiguous().view(B, x.shape[1], C)

        x = self.linear(attn_values)

        return x, kv_cache_new
    
class AttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.n_embed)
        self.attn = MultiHeadAttention(config)
        self.layer_norm_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None, rope_cache=None):
        x = self.layer_norm_1(x)
        attn_out, kv_cache_new = self.attn(x, kv_cache, rope_cache)
        x = x + attn_out
        x = self.layer_norm_2(x)

        mlp_out = self.mlp(x)

        x = x + mlp_out

        return x, kv_cache_new

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            embed = nn.Embedding(config.vocab_size, config.n_embed),
        
            attn_blocks = nn.ModuleList([AttentionBlock(config) for _ in range(config.n_layers)])
        ))
    
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)

        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.context_length = config.context_length
        self.rope_cache = None
        self.config = config

    def forward(self, x, kv_cache=None):
        if kv_cache:
            x = x[:, -1].unsqueeze(0)
        else:
            kv_cache = [None]*len(self.transformer.attn_blocks)

        if self.rope_cache == None:
            self.rope_cache = build_rope_cache(self.config)

        x = self.transformer.embed(x)

        kv_cache_new = []
        for attn_block, kv_cache_block in zip(self.transformer.attn_blocks, kv_cache):
            x, kv_cache_new_block = attn_block(x, kv_cache_block, self.rope_cache)
            kv_cache_new.append(kv_cache_new_block)

        x = self.lm_head(x)

        return x, kv_cache_new
    
    def generate(self, input_ids, max_length):
        original_length = input_ids.shape[-1]
        while (input_ids.shape[-1] - original_length) < max_length:
            input_ids = input_ids[-self.config.context_length:]
            logits = self(input_ids)[:, -1, :]
            pred = torch.argmax(logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat((input_ids, pred), dim=-1)

        return input_ids