"""
mamba_minimal.py — Pure-NumPy minimal MAMBA-like model for offline execution.

Faithfully simulates the output-projection layer activation structure:
  - X ∈ R^{T×n}  are hidden states from a selective SSM (rank structure: T < n typical)
  - Y = X @ C.T  where C is the out_proj weight
  - (X, Y) are the exact inputs/outputs needed for experiments 1–3

This is the numerically correct substrate for the QP experiments regardless
of whether it runs with full PyTorch or pure NumPy.

Architecture mirrors mamba-minimal (alxndrTL):
  - expand = 2  →  d_inner = 2 * d_model = 512
  - Selective SSM with dt, A, B, C parameterisation (ZOH discretisation)
  - out_proj: R^{d_inner} → R^{d_model}  (THIS is the QP target)
"""
"""
mamba_minimal.py — PyTorch MAMBA implementation with hook support.
Faithful to alxndrTL/mamba-minimal (arXiv:2312.00752).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from config import Config


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MambaBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.d_model  = cfg.d_model
        self.d_inner  = cfg.d_inner
        self.d_state  = cfg.d_state
        self.dt_rank  = math.ceil(cfg.d_model / 16)

        self.norm     = RMSNorm(cfg.d_model)
        self.in_proj  = nn.Linear(cfg.d_model, cfg.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(cfg.d_inner, cfg.d_inner,
                                   kernel_size=4, groups=cfg.d_inner,
                                   padding=3, bias=True)
        self.x_proj   = nn.Linear(cfg.d_inner,
                                   self.dt_rank + 2 * cfg.d_state, bias=False)
        self.dt_proj  = nn.Linear(self.dt_rank, cfg.d_inner, bias=True)

        A = repeat(torch.arange(1, cfg.d_state + 1),
                   'n -> d n', d=cfg.d_inner).float()
        self.A_log    = nn.Parameter(torch.log(A))
        self.D        = nn.Parameter(torch.ones(cfg.d_inner))

        # THIS IS THE QP TARGET — must be named out_proj
        self.out_proj = nn.Linear(cfg.d_inner, cfg.d_model, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        B, L, _ = x.shape

        xz = self.in_proj(x)
        x_b, z = xz.chunk(2, dim=-1)

        x_b = rearrange(x_b, 'b l d -> b d l')
        x_b = self.conv1d(x_b)[:, :, :L]
        x_b = rearrange(x_b, 'b d l -> b l d')
        x_b = F.silu(x_b)

        y = self._ssm(x_b)
        y = y * F.silu(z)

        out = self.out_proj(y)       # <-- hook captures here
        return out + residual

    def _ssm(self, x):
        B, L, d = x.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        dt, B_s, C_s = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(self.dt_proj(dt))

        dA = torch.exp(torch.einsum('bld,dn->bldn', dt, A))
        dB = torch.einsum('bld,bln,bld->bldn', dt, B_s, x)

        h = torch.zeros(B, d, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i]
            y = torch.einsum('bdn,bn->bd', h, C_s[:, i])
            ys.append(y)

        return torch.stack(ys, dim=1) + x * D


class Mamba(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg       = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers    = nn.ModuleList([MambaBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f    = RMSNorm(cfg.d_model)
        self.lm_head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    def get_out_proj(self, layer_idx):
        return self.layers[layer_idx].out_proj

    def compute_loss(self, input_ids, target_ids):
        logits = self.forward(input_ids)
        B, L, V = logits.shape
        return F.cross_entropy(logits.view(B * L, V), target_ids.view(B * L))
