"""extract.py — PyTorch hook-based activation extraction."""

import torch
import numpy as np
from config import Config


def extract_XY(model, input_ids, layer_idx, cfg):
    """
    Forward pass with hook on out_proj of layer_idx.
    Returns X [B*T, d_inner] and Y [B*T, d_model] as float64 numpy arrays.
    """
    model.eval()
    captured = {}

    def hook(module, inp, out):
        captured['X'] = inp[0].detach().cpu().float()
        captured['Y'] = out.detach().cpu().float()

    handle = model.get_out_proj(layer_idx).register_forward_hook(hook)

    device = next(model.parameters()).device

    # Handle both numpy arrays and torch tensors
    if isinstance(input_ids, np.ndarray):
        input_ids = torch.from_numpy(input_ids)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        _ = model(input_ids)

    handle.remove()

    B, L, d_inner = captured['X'].shape
    X = captured['X'].reshape(B * L, d_inner).numpy().astype(np.float64)
    Y = captured['Y'].reshape(B * L, -1).numpy().astype(np.float64)

    return X, Y


def verify_dimensions(X, Y, cfg):
    B, T = cfg.batch_size, cfg.seq_len
    T_total = B * T
    n, d = cfg.d_inner, cfg.d_model

    print(f"\n── Dimension Verification ──────────────────────────────────")
    print(f"  Expected X: [{T_total}, {n}]   Got: {list(X.shape)}")
    print(f"  Expected Y: [{T_total}, {d}]   Got: {list(Y.shape)}")

    assert X.shape == (T_total, n)
    assert Y.shape == (T_total, d)
    assert X.dtype == np.float64

    sv   = np.linalg.svd(X.T @ X, compute_uv=False)
    tol  = sv[0] * max(X.shape) * np.finfo(np.float64).eps
    rank = int(np.sum(sv > tol))

    print(f"  dtype: float64  ✓")
    print(f"  rank(XᵀX): {rank} / {n}")
    print(f"  cond(XᵀX): {sv[0]/sv[rank-1]:.4e}")
    print(f"  null_dim:  {n - rank}")
    print(f"  ‖Y‖_F:     {np.linalg.norm(Y, 'fro'):.6e}")
    print(f"────────────────────────────────────────────────────────────\n")