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

import numpy as np

# ── ensure working directory = script directory ──────────────────────────────
os.chdir(pathlib.Path(__file__).parent)

from config import Config
from data import load_wikitext2, get_fixed_batch, get_sequential_batches
from mamba_minimal import MambaNumpyModel
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip Experiments 4–5")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Tiny parameters for quick sanity check")
    args = parser.parse_args()

    cfg = Config()

    if args.smoke_test:
        cfg.adam_n_steps      = 30
        cfg.n_tokens_total    = 500
        cfg.hybrid_train_steps = 10
        cfg.batch_size         = 4
        cfg.seq_len            = 16
        cfg.d_model            = 64
        cfg.d_state            = 4
        cfg.n_layers           = 2
        cfg.hybrid_k_steps     = [5]
        print("⚠  SMOKE TEST MODE — reduced parameters")

    set_all_seeds(cfg.seed)

    print("=" * 60)
    print("EXACT QP SOLVER FOR MAMBA — EXPERIMENTAL PIPELINE")
    print("=" * 60)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("\n[1/7] Loading corpus...")
    data = load_wikitext2(cfg)
    input_ids, target_ids = get_fixed_batch(data["train_ids"], cfg)
    print(f"  Fixed batch: input {input_ids.shape}  target {target_ids.shape}")

    # ── 2. Model ─────────────────────────────────────────────────────────────
    print("\n[2/7] Building MAMBA model (NumPy)...")
    set_all_seeds(cfg.seed)
    model = MambaNumpyModel(cfg)
    total_params = sum(
        v.size for layer in model.layers
        for v in [layer.in_proj, layer.out_proj, layer.x_proj, layer.dt_proj_W]
    )
    print(f"  Approximate param count: {total_params:,}")
    print(f"  d_model={cfg.d_model}  d_inner={cfg.d_inner}  n_layers={cfg.n_layers}")

    # ── 3. Extract X, Y ──────────────────────────────────────────────────────
    print(f"\n[3/7] Extracting activations from layer {cfg.target_layer}...")
    X, Y = extract_XY(model, input_ids, target_ids, cfg.target_layer, cfg)
    verify_dimensions(X, Y, cfg)

    # ── 4. Experiment 1 ──────────────────────────────────────────────────────
    print("\n[4/7] Experiment 1: Residual comparison...")
    res1 = experiment_1_residual_comparison(X, Y, cfg)
    plot_exp1_residual(res1, cfg.output_dir)

    # ── 5. Experiments 2–3 ───────────────────────────────────────────────────
    print("\n[5/7] Experiments 2–3: Structural characterisation...")
    res2 = experiment_2_anchor_sweep(X, Y, cfg)
    res3 = experiment_3_null_space(X, Y, cfg)
    plot_exp2_anchor_sweep(res2, cfg.output_dir)
    plot_exp3_singular_values(res3, cfg.output_dir)

    all_results = {"exp1": res1, "exp2": res2, "exp3": res3}

    # ── 6. Experiments 4–5 ───────────────────────────────────────────────────
    if not args.skip_training:
        print("\n[6/7] Experiments 4–5: Hybrid training loop...")
        train_batches = get_sequential_batches(data["train_ids"], cfg)
        val_batches   = get_sequential_batches(data["val_ids"],   cfg)
        print(f"  Train batches: {len(train_batches)}  Val batches: {len(val_batches)}")

        set_all_seeds(cfg.seed)
        res4 = experiment_4_hybrid_training(model, train_batches, cfg, val_batches)
        plot_exp4_perplexity(res4, cfg.output_dir)

        set_all_seeds(cfg.seed)
        res5 = experiment_5_algebraic_monitoring(model, train_batches, cfg)
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
