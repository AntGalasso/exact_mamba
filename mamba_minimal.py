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

import numpy as np
from config import Config


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _rms_norm(x: np.ndarray, w: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * w


class MambaBlockNumpy:
    """
    Single MAMBA block — pure NumPy.
    Forward: [B, L, d_model] → [B, L, d_model]
    Also returns (X_proj, Y_proj) for the out_proj activation extraction.
    """

    def __init__(self, cfg: Config, rng: np.random.Generator):
        d       = cfg.d_model          # 256
        n       = cfg.d_inner          # 512  (d_model * expand)
        dt_rank = max(1, d // 16)      # 16
        d_state = cfg.d_state          # 16

        scale = 0.02

        self.norm_w   = np.ones(d)
        self.in_proj  = rng.normal(0, scale, (2 * n, d))   # maps d → 2n

        # SSM parameters
        A_base = np.arange(1, d_state + 1, dtype=np.float64)
        self.A_log     = np.tile(np.log(A_base), (n, 1))  # [n, d_state]
        self.D         = np.ones(n)
        self.x_proj    = rng.normal(0, scale, (dt_rank + 2 * d_state, n))
        self.dt_proj_W = rng.normal(0, scale, (n, dt_rank))
        self.dt_proj_b = np.zeros(n)

        # THE TARGET: out_proj weight C  [d_model, d_inner]
        self.out_proj  = rng.normal(0, scale, (d, n))

        self.d_model  = d
        self.d_inner  = n
        self.d_state  = d_state
        self.dt_rank  = dt_rank

    def forward(self, x: np.ndarray):
        """
        x: [B, L, d_model]
        Returns:
            output    [B, L, d_model]
            X_proj    [B*L, d_inner]  — inputs to out_proj
            Y_proj    [B*L, d_model]  — outputs of out_proj
        """
        residual = x.copy()
        x_norm   = _rms_norm(x, self.norm_w)

        B, L, _ = x.shape

        # in_proj → x_branch + z gate
        xz       = x_norm.reshape(B * L, self.d_model) @ self.in_proj.T
        xz       = xz.reshape(B, L, 2 * self.d_inner)
        x_branch = _silu(xz[:, :, :self.d_inner])    # [B, L, n]
        z        = xz[:, :, self.d_inner:]            # [B, L, n]

        # SSM
        y_ssm    = self._ssm(x_branch)               # [B, L, n]

        # Gate + out_proj
        y_gated  = y_ssm * _silu(z)                  # [B, L, n]
        Y_out    = y_gated.reshape(B * L, self.d_inner) @ self.out_proj.T  # [BL, d]
        Y_out    = Y_out.reshape(B, L, self.d_model)

        # Extraction
        X_proj = y_gated.reshape(B * L, self.d_inner).copy()   # [BL, n]
        Y_proj = Y_out.reshape(B * L, self.d_model).copy()     # [BL, d]

        output = Y_out + residual
        return output, X_proj, Y_proj

    def _ssm(self, x: np.ndarray) -> np.ndarray:
        """Selective SSM scan. x: [B, L, n] → [B, L, n]"""
        B_sz, L, n = x.shape
        d_state    = self.d_state
        dt_rank    = self.dt_rank

        A = -np.exp(self.A_log)   # [n, d_state]

        xbl = x.reshape(B_sz * L, n)
        x_dbl = (xbl @ self.x_proj.T).reshape(B_sz, L, -1)  # [B,L, dt_rank+2*ds]

        dt_raw = x_dbl[:, :, :dt_rank]
        B_ssm  = x_dbl[:, :, dt_rank : dt_rank + d_state]
        C_ssm  = x_dbl[:, :, dt_rank + d_state:]

        dt = _softplus(
            (dt_raw.reshape(B_sz * L, dt_rank) @ self.dt_proj_W.T
             + self.dt_proj_b).reshape(B_sz, L, n)
        )  # [B, L, n]

        # Discretise: dA [B,L,n,ds], dBu [B,L,n,ds]
        dA   = np.exp(dt[:, :, :, None] * A[None, None, :, :])
        dB_u = dt[:, :, :, None] * B_ssm[:, :, None, :] * x[:, :, :, None]

        # Sequential scan
        h  = np.zeros((B_sz, n, d_state), dtype=np.float64)
        ys = []
        for i in range(L):
            h   = dA[:, i] * h + dB_u[:, i]            # [B, n, ds]
            y_i = np.einsum('bnd,bd->bn', h, C_ssm[:, i])  # [B, n]
            ys.append(y_i)

        y = np.stack(ys, axis=1) + x * self.D[None, None, :]
        return y.astype(np.float64)


class MambaNumpyModel:
    """Full MAMBA model — pure NumPy."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)

        self.embedding = rng.normal(0, 0.02, (cfg.vocab_size, cfg.d_model))
        self.layers    = [MambaBlockNumpy(cfg, rng) for _ in range(cfg.n_layers)]
        self.norm_f_w  = np.ones(cfg.d_model)

    def forward_extract(
        self,
        input_ids: np.ndarray,
        target_layer: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Full forward pass; extracts (X, Y) from out_proj of target_layer.
        input_ids: [B, L]  int
        Returns:
            X: [B*L, d_inner]  float64
            Y: [B*L, d_model]  float64
        """
        x = self.embedding[input_ids].astype(np.float64)  # [B, L, d_model]

        X_out = Y_out = None
        for i, layer in enumerate(self.layers):
            x, Xp, Yp = layer.forward(x)
            if i == target_layer:
                X_out, Y_out = Xp, Yp

        return X_out.astype(np.float64), Y_out.astype(np.float64)

    def get_out_proj(self, layer_idx: int) -> np.ndarray:
        return self.layers[layer_idx].out_proj.copy()

    def set_out_proj(self, layer_idx: int, C: np.ndarray):
        self.layers[layer_idx].out_proj = C.copy()

    def compute_loss(
        self,
        input_ids: np.ndarray,
        target_ids: np.ndarray,
    ) -> float:
        """Cross-entropy LM loss (pure NumPy)."""
        x = self.embedding[input_ids].astype(np.float64)
        for layer in self.layers:
            x, _, _ = layer.forward(x)
        # rms norm
        x = _rms_norm(x, self.norm_f_w)
        # logits: [B, L, vocab_size]
        BL = x.shape[0] * x.shape[1]
        logits = x.reshape(BL, -1) @ self.embedding.T   # [BL, V]
        # Stable softmax cross-entropy
        logits -= logits.max(axis=-1, keepdims=True)
        log_sum_exp = np.log(np.exp(logits).sum(axis=-1))  # [BL]
        targets_flat = target_ids.reshape(-1)
        correct_logits = logits[np.arange(BL), targets_flat]  # [BL]
        nll = -(correct_logits - log_sum_exp).mean()
        return float(nll)
