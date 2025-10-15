from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class FMConfig:
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        # key, query, value stored in one big matrix
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # final output layer
        y = self.c_proj(y)
        return y


class CrossAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        # key, value stored in one big matrix, query stored in a separate matrix
        self.q_attn = nn.Linear(config.n_embd, config.n_embd)
        self.c_attn = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, query):
        # x.shape [B, T, C] query.shape [B, C]
        assert x.shape[0] == query.shape[0] and x.shape[-1] == query.shape[-1]

        B, T, C = x.shape
        q = self.q_attn(query.unsqueeze(1).expand(B, T, C))
        kv = self.c_attn(x)
        k, v = kv.split(self.n_embd, dim=-1)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # final output layer
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.GPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class SABlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CABlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CrossAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, query):
        x = x + self.attn(self.ln_1(x), query)
        x = x + self.mlp(self.ln_2(x))
        return x



class FlowMatching(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.query_proj = nn.Linear(1, config.n_embd)
        self.transformer = nn.ModuleDict(
            dict(
                emb_proj = nn.Linear(1, config.n_embd),
                sa_h = nn.ModuleList([SABlock(config) for _ in range(config.n_layer // 2)]),
                ca_h = nn.ModuleList([CABlock(config) for _ in range(config.n_layer // 2)]),
                ln_f = nn.LayerNorm(config.n_embd),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, 1)


    
    def generate_t_embedding(self, t, max_positions=10000):
        B, T = t.shape
        assert T == 1
        
        d_model = self.config.n_embd
        div_term = torch.exp(torch.arange(0, d_model, 2, device=t.device) * (-math.log(10000.0) / d_model))

        # [B, n_embd / 2]
        angles = (t * max_positions) * div_term[None, :]
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        # interleave [sin, cos] along the last dimension
        pos_encoding = torch.stack((sin, cos), dim=-1).reshape(B, d_model)
        return pos_encoding


    """
    Forward function
        img_tensor: image tensor, shape [B, C, H, W]
        t:          timestamp tensor, shape [B, 1]
        cls:        class token tensor, shape [B, 1]
    """
    def forward(self, img_tensor, t, cls):

        # in our case, [C, H, W] is [1, 28, 28]
        B, C, H, W = img_tensor.shape
        idx = img_tensor.view(B, -1, 1)
       
        # B, C * H * W, n_embd
        tok_emb = self.transformer.emb_proj(idx)
        t_emb = self.generate_t_embedding(t).unsqueeze(1).expand(B, C * H * W, -1)

        x = tok_emb + t_emb
        query = self.query_proj(cls)

        # transformer blocks
        for i in range(self.config.n_layer // 2):
            ca_block = self.transformer.ca_h[i]
            sa_block = self.transformer.sa_h[i]

            x = sa_block(ca_block(x, query))

        # the final layer norm
        x = self.transformer.ln_f(x)

        # [B, C * H * W, 1]
        logits = self.lm_head(x).view(B, C, H, W)

        return logits
    