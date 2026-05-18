"""
main.py — Entry point. Run all 5 experiments and save outputs.

Usage:
    python main.py                  # full pipeline
    python main.py --skip-training  # Exp 1–3 only (algebraic, fast)
    python main.py --smoke-test     # tiny config, quick check
"""

import sys
import os
import random
import argparse
import time
import pathlib
try:
    import torch
except ImportError:
    torch = None

import numpy as np

# ── ensure working directory = script directory ──────────────────────────────
os.chdir(pathlib.Path(__file__).parent)

from config import Config
from data import load_wikitext2, get_fixed_batch, get_sequential_batches
try:
    from mamba_minimal import Mamba as MambaModel
    USE_TORCH = True
except ImportError:
    from mamba_minimal import MambaNumpyModel as MambaModel
    USE_TORCH = False
from extract import extract_XY, verify_dimensions
from experiments import (
    experiment_1_residual_comparison,
    experiment_2_anchor_sweep,
    experiment_3_null_space,
    experiment_4_hybrid_training,
    experiment_5_algebraic_monitoring,
)
from plots import (
    plot_exp1_residual,
    plot_exp2_anchor_sweep,
    plot_exp3_singular_values,
    plot_exp4_perplexity,
    plot_exp5_monitoring,
    generate_latex_tables,
)


def set_all_seeds(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Summary printers
# ─────────────────────────────────────────────────────────────────────────────

def _print_exp4_summary(res4: dict):
    w = 60
    full    = res4.get("_full", {})
    k_steps = res4.get("_k_step_values", [])
    b_final = res4.get("_b_final", float("nan"))

    print(f"\n  ┌{'─'*w}┐")
    print(f"  │{'EXPERIMENT 4  —  Hybrid Training Protocol':^{w}}│")
    print(f"  ├{'─'*w}┤")
    print(f"  │  Metric: ‖X_eval Cᵀ − Y_eval‖_F{'':>21}│")
    print(f"  ├{'─'*w}┤")
    print(f"  │  Adam baseline  final residual : {b_final:.6e}{'':>8}│")

    for k in k_steps:
        if k not in full:
            continue

        hf = full[k]["residual"][-1]
        imp = (b_final - hf) / b_final * 100 if b_final > 0 else 0.0
        sgn = "▼" if imp > 0 else "▲"

        n_corr = len(full[k]["correction_steps"])
        corr_data = full[k].get("residual_at_correction", [])
        avg_drop = (float(np.mean([r["drop"] for r in corr_data]))
                    if corr_data else float("nan"))

        print(f"  │  Hybrid k_step={k:3d}  residual: {hf:.6e}  "
              f"({sgn}{abs(imp):.2f}%)  corr: {n_corr:3d}  "
              f"Δres: {avg_drop:.3e}  │")

    best_k = None
    best_res = b_final

    for k in k_steps:
        if k in full:
            val = full[k]["residual"][-1]
            if val < best_res:
                best_res = val
                best_k   = k

    print(f"  ├{'─'*w}┤")
    if best_k is not None:
        imp = (b_final - best_res) / b_final * 100
        print(f"  │  Best hybrid: k_step={best_k}  residual={best_res:.6e}  "
              f"improvement={imp:.2f}% vs baseline{'':>2}│")
    else:
        print(f"  │  No hybrid variant outperformed baseline.{'':>18}│")

    print(f"  └{'─'*w}┘")


def _print_exp5_summary(res5: dict):
    """Extended console summary for Experiment 5."""
    w = 60
    ranks     = res5.get("ranks",        res5.get("rank",        []))
    conds     = res5.get("cond_numbers", res5.get("cond",        []))
    null_dims = res5.get("null_dims",    res5.get("null_dim",    []))
    is_uniq   = res5.get("is_unique", [])
    r_before  = res5.get("residual_before", [])
    r_after   = res5.get("residual_after",  [])
    steps     = res5.get("steps", res5.get("correction_step", []))
    n_corr    = len(ranks)

    print(f"\n  ┌{'─'*w}┐")
    print(f"  │{'EXPERIMENT 5  —  Algebraic Monitoring':^{w}}│")
    print(f"  ├{'─'*w}┤")
    print(f"  │  Total exact corrections recorded : {n_corr:4d}{'':>20}│")

    if n_corr > 0:
        rank_stable = min(ranks) == max(ranks)
        n_sing      = sum(1 for nd in null_dims if nd > 0)
        all_uniq    = all(is_uniq)
        avg_drop    = (float(np.mean([b - a for b, a in zip(r_before, r_after)]))
                       if r_before else float("nan"))

        print(f"  │  rank(XᵀX)  min={min(ranks):3d}  max={max(ranks):3d}  "
              f"stable={str(rank_stable):<5}{'':>17}│")
        print(f"  │  null_dim   min={min(null_dims):3d}  max={max(null_dims):3d}{'':>34}│")
        print(f"  │  cond(XᵀX)  min={min(conds):.3e}  max={max(conds):.3e}{'':>19}│")
        print(f"  │  Batches with nontrivial null space : "
              f"{n_sing}/{n_corr} ({100*n_sing/n_corr:.1f}%){'':>12}│")
        print(f"  │  Uniqueness always satisfied        : "
              f"{str(all_uniq):<5}{'':>24}│")
        print(f"  │  Avg residual drop per correction   : "
              f"{avg_drop:.4e}{'':>14}│")

        # Print a compact table of every recorded correction step
        print(f"  ├{'─'*w}┤")
        header = (f"  │  {'step':>5}  {'rank':>4}  {'null':>4}  "
                  f"{'cond':>9}  {'uniq':>5}  {'res_before':>10}  {'res_after':>10}  │")
        print(header)
        print(f"  ├{'─'*w}┤")
        for i in range(n_corr):
            rb_str = f"{r_before[i]:.3e}" if r_before else "  n/a    "
            ra_str = f"{r_after[i]:.3e}"  if r_after  else "  n/a    "
            print(f"  │  {steps[i]:>5}  {ranks[i]:>4}  {null_dims[i]:>4}  "
                  f"{conds[i]:>9.2e}  {str(is_uniq[i] if is_uniq else '?'):>5}  "
                  f"{rb_str:>10}  {ra_str:>10}  │")

    print(f"  └{'─'*w}┘")


def print_final_summary(all_results: dict):
    res1 = all_results.get("exp1", {})
    res2 = all_results.get("exp2", [])
    res3 = all_results.get("exp3", {})

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    exact_r = res1.get("exact_residual", float("nan"))
    Y_norm  = res1.get("Y_norm", float("nan"))
    print(f"  ‖Y‖_F (Adam step 0):      {Y_norm:.6e}")
    print(f"  Exact residual (1 step):  {exact_r:.6e}")

    best_lr = res1.get("best_adam_lr")
    curves  = res1.get("adam_curves", {})
    if best_lr and best_lr in curves:
        print(f"  Best Adam (lr={best_lr:.0e}) final: {curves[best_lr][-1]:.6e} "
              f"(after {len(curves[best_lr])} steps)")

    s8, s10, s14 = (res1.get(k) for k in ("steps_to_1e8", "steps_to_1e10", "steps_to_1e14"))
    print(f"  Adam steps to 1e-8:    {s8}")
    print(f"  Adam steps to 1e-10:   {s10}")
    print(f"  Adam steps to 1e-14:   {s14}   ← exact solver: 1 step")

    rank = res3.get("rank_XtX")
    if rank is not None:
        print(f"\n  rank(XᵀX):     {rank}")
        print(f"  null_dim:      {res3.get('null_dim_XtX')}")
        print(f"  cond(XᵀX):     {res3.get('cond_XtX'):.4e}")
        grad_ok = res3.get("grad_null_verified")
        print(f"  Gradient zero on null space: {'✓ verified' if grad_ok else '✗ FAILED'}")

    # Uniqueness transition per strategy
    first_unique = {}
    for r in res2:
        s = r.get("strategy", "?")
        if r.get("is_unique") and s not in first_unique:
            first_unique[s] = r["k"]
    if first_unique:
        print("\n  Uniqueness first achieved at k:")
        for strat, k in sorted(first_unique.items()):
            print(f"    {strat:24s}: k = {k}")

    print("=" * 60)
    print(f"  PRIMARY RESULT FOR ABSTRACT:")
    print(f"    Exact residual = {exact_r:.2e}  (1 algebraic step)")
    if s8 is not None:
        print(f"    Adam steps to 1e-8: {s8}  (best lr={best_lr:.0e})")
    print("=" * 60)

    # Experiments 4 and 5 extended summaries
    if "exp4" in all_results:
        print()
        _print_exp4_summary(all_results["exp4"])
    if "exp5" in all_results:
        print()
        _print_exp5_summary(all_results["exp5"])


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip Experiments 4–5")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Tiny parameters for quick sanity check")
    args = parser.parse_args()

    cfg = Config()

    if args.smoke_test:
        cfg.adam_n_steps       = 30
        cfg.n_tokens_total     = 500
        cfg.hybrid_train_steps = 1000      # enough steps for a few corrections
        cfg.batch_size         = 4
        cfg.seq_len            = 16
        cfg.d_model            = 64
        cfg.d_state            = 4
        cfg.n_layers           = 2
        cfg.hybrid_k_steps     = [5, 10]
        cfg.hybrid_eval_every  = 10
        print("⚠  SMOKE TEST MODE — reduced parameters")

    set_all_seeds(cfg.seed)

    print("=" * 60)
    print("EXACT QP SOLVER FOR MAMBA — EXPERIMENTAL PIPELINE")
    print("=" * 60)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("\n[1/7] Loading corpus...")
    train_ids, val_ids = load_wikitext2(cfg)
    input_ids, target_ids = get_fixed_batch(train_ids, cfg)
    print(f"  Fixed batch: input {input_ids.shape}  target {target_ids.shape}")

    # ── 2. Model ─────────────────────────────────────────────────────────────
    print("\n[2/7] Building MAMBA model (NumPy)...")
    set_all_seeds(cfg.seed)
    model = MambaModel(cfg)
    if USE_TORCH:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        print(f"  Device: {device}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Approximate param count: {total_params:,}")
    print(f"  d_model={cfg.d_model}  d_inner={cfg.d_inner}  n_layers={cfg.n_layers}")

    # ── 3. Extract X, Y ──────────────────────────────────────────────────────
    print(f"\n[3/7] Extracting activations from layer {cfg.target_layer}...")
    X, Y = extract_XY(model, input_ids, cfg.target_layer, cfg)
    verify_dimensions(X, Y, cfg)

    # ── 4. Experiment 1 ──────────────────────────────────────────────────────
    print("\n[4/7] Experiment 1: Residual comparison...")
    res1 = experiment_1_residual_comparison(X, Y, cfg)
    plot_exp1_residual(res1, cfg.output_dir)

    # ── 5. Experiments 2–3 ───────────────────────────────────────────────────
    # ── DEBUG CHECK ─────────────────────────────────────────────────────────
    print("\n[DEBUG] Checking X and Y before Experiments 2–3...")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    if hasattr(X, "dtype"):
        print("X dtype:", X.dtype)
    if hasattr(Y, "dtype"):
        print("Y dtype:", Y.dtype)
    assert X.ndim == 2, f"X must be 2D, got {X.shape}"
    assert Y.ndim == 2, f"Y must be 2D, got {Y.shape}"
    assert X.shape[0] == Y.shape[0], "Mismatch in number of samples between X and Y"
    try:
        X_np = X.detach().cpu().numpy() if hasattr(X, "detach") else X
        Y_np = Y.detach().cpu().numpy() if hasattr(Y, "detach") else Y
        print("X mean/std:", np.mean(X_np), np.std(X_np))
        print("Y mean/std:", np.mean(Y_np), np.std(Y_np))
    except Exception as e:
        print("Could not compute stats:", e)
    print("[DEBUG] OK → proceeding to Experiments 2–3\n")

    print("\n[5/7] Experiments 2–3: Structural characterisation...")
    res2 = experiment_2_anchor_sweep(X, Y, cfg)
    res3 = experiment_3_null_space(X, Y, cfg)
    plot_exp2_anchor_sweep(res2, cfg.output_dir)
    plot_exp3_singular_values(res3, cfg.output_dir)

    all_results = {"exp1": res1, "exp2": res2, "exp3": res3}

    # ── 6. Experiments 4–5 ───────────────────────────────────────────────────
    if not args.skip_training:
        print("\n[6/7] Experiments 4–5: Hybrid training + algebraic monitoring...")
        print(f"  Using fixed (X, Y) batch extracted above "
              f"[shape X={X.shape}, Y={Y.shape}]")
        print(f"  Batch variation simulated via 5% Gaussian noise per step.")

        set_all_seeds(cfg.seed)
        print("\n  ── Starting Experiment 4 ────────────────────────────────────")
        res4 = experiment_4_hybrid_training(X, Y, cfg)
        plot_exp4_perplexity(res4, cfg.output_dir)

        set_all_seeds(cfg.seed)
        print("\n  ── Starting Experiment 5 ────────────────────────────────────")
        res5 = experiment_5_algebraic_monitoring(X, Y, cfg)
        plot_exp5_monitoring(res5, cfg.output_dir)

        all_results["exp4"] = res4
        all_results["exp5"] = res5
    else:
        print("\n[6/7] Skipping Experiments 4–5 (--skip-training)")

    # ── 7. Tables + summary ──────────────────────────────────────────────────
    print("\n[7/7] Generating LaTeX tables and final summary...")
    generate_latex_tables(all_results, cfg.output_dir)
    print_final_summary(all_results)

    out = cfg.output_dir.resolve()
    print(f"\n✓ All outputs saved to: {out}")
    print(f"  Reproducibility table: {out}/reproducibility_table.tex")


if __name__ == "__main__":
    main()
