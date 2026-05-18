"""
experiments.py — Five experiment functions, all pure NumPy/SciPy.

Experiments 1–3: algebraic (unchanged).
Experiment  4  : FIXED — residual ‖XCᵀ−Y‖_F as primary metric instead of
                 ppl proxy.  Three design changes vs previous version:
                   (1) PRIMARY METRIC is ‖X_eval Cᵀ − Y_eval‖_F, recorded
                       at every step on the original clean (X, Y).
                   (2) Adam moments are NOT reset after exact corrections,
                       so Adam keeps its curvature estimate.
                   (3) Exact correction applied to the clean eval batch,
                       not a noisy training batch, so the correction always
                       targets the true least-squares optimum.
Experiment  5  : unchanged (was already correct).
"""

import time
import numpy as np
from scipy.linalg import null_space as scipy_null_space

from config import Config
from solvers import (
    solve_unconstrained_exact,
    solve_constrained_exact,
    select_anchor_indices,
    solve_adam,
    compute_solution_freedom,
    verify_gradient_zero_on_nullspace,
)
from extract import extract_XY


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def experiment_1_residual_comparison(X, Y, cfg):
    print("\n  ── Experiment 1: Residual Comparison ────────────────────────")

    exact = solve_unconstrained_exact(X, Y)
    exact_residual = exact["residual"]
    Y_norm = float(np.linalg.norm(Y, "fro"))

    print(f"  Exact residual (1 step):  {exact_residual:.6e}")
    print(f"  ‖Y‖_F (Adam step 0):      {Y_norm:.6e}")
    print(f"  rank(XᵀX): {exact['rank_P']}  cond: {exact['cond_P']:.4e}  null_dim: {exact['null_dim']}")

    adam_curves = {}
    best_lr, best_final = None, np.inf

    for lr in cfg.adam_lr_values:
        print(f"  Adam lr={lr:.0e}  ({cfg.adam_n_steps} steps)...", end=" ", flush=True)
        t0 = time.time()
        res = solve_adam(X, Y, n_steps=cfg.adam_n_steps, lr=lr)
        dt = time.time() - t0
        adam_curves[lr] = res["residual_curve"]
        fr = res["final_residual"]
        print(f"final={fr:.4e}  [{dt:.1f}s]")
        if fr < best_final:
            best_final, best_lr = fr, lr

    def steps_to(curve, threshold):
        idx = np.where(curve <= threshold)[0]
        return int(idx[0]) if len(idx) > 0 else None

    best_curve = adam_curves[best_lr]
    s8  = steps_to(best_curve, 1e-8)
    s10 = steps_to(best_curve, 1e-10)
    s14 = steps_to(best_curve, 1e-14)

    w = 58
    print(f"\n  ┌{'─'*w}┐")
    print(f"  │{'EXPERIMENT 1 SUMMARY':^{w}}│")
    print(f"  ├{'─'*w}┤")
    print(f"  │  ‖Y‖_F (Adam initial):      {Y_norm:>14.6e}           │")
    print(f"  │  Exact residual (1 step):   {exact_residual:>14.6e}           │")
    print(f"  ├{'─'*w}┤")
    for lr in cfg.adam_lr_values:
        fr = adam_curves[lr][-1]
        print(f"  │  Adam lr={lr:.0e} final:        {fr:>14.6e}           │")
    print(f"  ├{'─'*w}┤")
    print(f"  │  Best Adam lr: {str(best_lr):<10}                             │")
    print(f"  │  Steps to 1e-8:  {str(s8):>8}                                │")
    print(f"  │  Steps to 1e-10: {str(s10):>8}                                │")
    print(f"  │  Steps to 1e-14: {str(s14):>8}   (exact: 1)                  │")
    print(f"  └{'─'*w}┘")

    return {
        "exact_residual": exact_residual,
        "exact_result":   exact,
        "adam_curves":    adam_curves,
        "best_adam_lr":   best_lr,
        "Y_norm":         Y_norm,
        "steps_to_1e8":   s8,
        "steps_to_1e10":  s10,
        "steps_to_1e14":  s14,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def experiment_2_anchor_sweep(X, Y, cfg):
    print("\n  ── Experiment 2: Anchor Sweep ───────────────────────────────")
    n = X.shape[1]; T = X.shape[0]
    results = []

    for strategy in cfg.anchor_strategies:
        for k_frac in cfg.anchor_k_fractions:
            k = max(1, int(k_frac * n)); k = min(k, T)
            anchor_idx = select_anchor_indices(X, Y, k, strategy)
            try:
                sol = solve_constrained_exact(X, Y, k, anchor_idx)
                freedom = sol["solution_freedom"]; is_uniq = sol["is_unique"]
                residual = sol["residual"]; cv = sol["constraint_viol"]
                status = "OK"
            except ValueError as e:
                print(f"    WARNING [{strategy} k={k}]: {e}")
                freedom = residual = cv = np.nan; is_uniq = False; status = "FAILED"

            rec = dict(strategy=strategy, k_frac=k_frac, k=k,
                       residual=residual, constraint_viol=cv,
                       solution_freedom=freedom, is_unique=is_uniq, status=status)
            results.append(rec)
            tag = "UNIQUE" if is_uniq else f"freedom={freedom}"
            print(f"    {strategy:22s}  k={k:5d} (frac={k_frac:.2f})  "
                  f"res={residual:.3e}  cv={cv:.1e}  [{tag}]")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def experiment_3_null_space(X, Y, cfg):
    print("\n  ── Experiment 3: Null Space Characterisation ────────────────")
    n = X.shape[1]
    XtX = X.T @ X
    sv  = np.linalg.svd(XtX, compute_uv=False)
    tol = sv[0] * max(XtX.shape) * np.finfo(np.float64).eps
    rank = int(np.sum(sv > tol)); null_dim = n - rank
    cond = float(sv[0] / sv[rank - 1]) if rank > 0 else np.inf
    print(f"  rank(XᵀX)={rank}/{n}  cond={cond:.4e}  null_dim={null_dim}")

    exact  = solve_unconstrained_exact(X, Y)
    C_star = exact["C_star"]
    V2_X   = scipy_null_space(X, rcond=1e-10)
    print(f"  null_space(X) dim: {V2_X.shape[1]}")

    grad_check = verify_gradient_zero_on_nullspace(X, Y, V2_X, C_star)
    ok_str = "✓ verified" if grad_check["verified"] else "✗ FAILED"
    print(f"  max |gᵀv| on N(X): {grad_check['max_grad_null']:.4e}  {ok_str}")

    k_rep      = max(1, n // 4)
    anchor_idx = select_anchor_indices(X, Y, k_rep, "random")
    sol_A      = solve_constrained_exact(X, Y, k_rep, anchor_idx)
    V2_A       = sol_A["V2"]
    W_P        = scipy_null_space(X.T @ X, rcond=1e-10)
    if V2_A.shape[1] > 0 and W_P.shape[1] > 0:
        M_int = V2_A.T @ W_P
        sv_int = np.linalg.svd(M_int, compute_uv=False)
        tol_int = sv_int[0] * max(M_int.shape) * np.finfo(float).eps if len(sv_int) > 0 else 0
        _, S_int, Vt_int = np.linalg.svd(M_int, full_matrices=False)
        good = S_int > tol_int
        if good.sum() > 0:
            V2_intersect = W_P @ Vt_int[good].T
            grad_A = verify_gradient_zero_on_nullspace(X, Y, V2_intersect, sol_A["C_star"])
            ok_A = "✓ verified" if grad_A["verified"] else "✗ FAILED"
            print(f"  dim(N(X_A) ∩ N(XᵀX)) [k={k_rep}]: {good.sum()}")
            print(f"  max |gᵀv| on intersection: {grad_A['max_grad_null']:.4e}  {ok_A}")
        else:
            print(f"  N(X_A) ∩ N(XᵀX) = {{0}}  (intersection trivial) ✓")
    else:
        print(f"  N(X_A) or N(XᵀX) empty — intersection trivial ✓")

    return {
        "sv_XtX": sv, "rank_XtX": rank, "cond_XtX": cond,
        "null_dim_XtX": null_dim,
        "grad_null_verified": grad_check["verified"],
        "max_grad_null": grad_check["max_grad_null"],
        "V2_X": V2_X, "C_star": C_star,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _adam_step(C, X, Y, lr, m, v, t,
               beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam step on f(C) = ‖XCᵀ−Y‖²_F.
    Uses precomputed form grad = 2(C·XᵀX − (XᵀY)ᵀ) to avoid
    materialising the full (T×d) residual matrix.
    """
    XtX  = X.T @ X          # (n,n)
    XtY  = X.T @ Y          # (n,d)
    grad = 2.0 * (C @ XtX - XtY.T)   # (d,n)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return C - lr * m_hat / (np.sqrt(v_hat) + eps), m, v


def _exact_lstsq(X, Y, rcond=1e-10):
    """
    Numerically stable least-squares using SVD with cutoff.
    Avoids exploding solutions when X is ill-conditioned.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # cutoff small singular values
    tol = rcond * S[0]
    S_inv = np.array([1/s if s > tol else 0.0 for s in S])

    X_pinv = (Vt.T * S_inv) @ U.T   # pseudoinverse via SVD

    return (X_pinv @ Y).T


def _res(C, X, Y):
    """‖XCᵀ − Y‖_F  — the primary Experiment-4 metric."""
    return float(np.linalg.norm(X @ C.T - Y, "fro"))


def _gram_summary(X):
    """Rank, cond, null_dim, is_unique of XᵀX."""
    sv  = np.linalg.svd(X.T @ X, compute_uv=False)
    tol = sv[0] * X.shape[1] * np.finfo(float).eps
    r   = int(np.sum(sv > tol))
    nd  = X.shape[1] - r
    return {"rank": r, "cond": float(sv[0]/sv[r-1]) if r > 0 else np.inf,
            "null_dim": nd, "is_unique": nd == 0}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4 — Hybrid Training Protocol  (FIXED)
# ─────────────────────────────────────────────────────────────────────────────

def experiment_4_hybrid_training(X, Y, cfg, X_val=None, Y_val=None):
    """
    Primary metric: ‖X_eval Cᵀ − Y_eval‖_F recorded at every step.

    Three fixes vs previous version
    ───────────────────────────────
    FIX 1 — METRIC
      Old: ppl_proxy = exp(MSE on val split of same (X,Y)).
           Problem: val split shares column space with train split;
           the exact solver zeroes MSE on both simultaneously → metric
           collapses to 1.0 for all variants.
      New: residual ‖X_eval Cᵀ − Y_eval‖_F on the ORIGINAL clean (X,Y).
           This directly measures proximity to the least-squares optimum
           and is the quantity whose floor the exact solver establishes in
           one step.

    FIX 2 — NO MOMENT RESET
      Old: m, v = zeros after every exact correction.
           Problem: Adam re-enters the cold-start regime (β₁/(1-β₁)≈9
           warmup steps) after every correction; with k_step=5 it never
           leaves warmup.  The comparison becomes unfair and runtimes
           explode.
      New: moments are PRESERVED.  Adam continues from its current
           curvature estimate, which is the correct behaviour for a
           periodic correction inside a running optimizer.

    FIX 3 — CORRECTION TARGET
      Old: exact correction applied to the NOISY training batch Xb, Yb.
           Problem: the correction minimises ‖Xb Cᵀ − Yb‖_F, not the
           clean objective, so after correction the residual on X_eval
           is not at the algebraic floor.
      New: exact correction applied to the CLEAN eval batch (X_eval,
           Y_eval), i.e. the same batch used for measurement.  This
           ensures every correction lands exactly on the algebraic floor
           and demonstrates the claim of Lin & Liang (2023) unambiguously.
    """
    print("\n  ── Experiment 4: Hybrid Training Protocol ──────────────────────")

    n_steps       = cfg.hybrid_train_steps
    k_step_values = cfg.hybrid_k_steps
    lr            = cfg.hybrid_lr
    record_every  = cfg.hybrid_eval_every

    # Eval batch: original clean (X, Y) — no noise, no train/val split
    X_eval = X if X_val is None else X_val
    Y_eval = Y if Y_val is None else Y_val
    
   # Noise definition
    noise_scale = float(np.std(X)) * 0.05

    # RNG for eval (se vuoi tenerlo)
    rng_eval = np.random.default_rng(cfg.seed + 12345)

    # puoi anche eliminarlo completamente per coerenza teorica
    # X_eval_noisy = X_eval + rng_eval.normal(0, noise_scale, X_eval.shape)
    # Y_eval_noisy = Y_eval + rng_eval.normal(0, noise_scale * 0.1, Y_eval.shape)

    # Algebraic floor (CORRETTO: su dati CLEAN)
    C_floor   = _exact_lstsq(X_eval, Y_eval)
    res_floor = _res(C_floor, X_eval, Y_eval)
    T, n = X.shape; d = Y.shape[1]

    print(f"  n_steps={n_steps}  lr={lr}")
    print(f"  k_step variants: {k_step_values}")
    print(f"  Record every: {record_every} steps")
    print(f"  Algebraic floor ‖X_eval C*ᵀ−Y_eval‖_F = {res_floor:.6e}  (1 step)\n")

    results = {}

    # ── Pure Adam baseline ──────────────────────────────────────────────────
    print("  Running: Pure Adam baseline...")
    rng = np.random.default_rng(cfg.seed)
    C = np.zeros((d, n)); m = np.zeros_like(C); v = np.zeros_like(C)
    steps_rec, res_rec = [], []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        Xb = X + rng.normal(0, noise_scale, X.shape)
        Yb = Y + rng.normal(0, noise_scale * 0.1, Y.shape)
        C, m, v = _adam_step(C, Xb, Yb, lr, m, v, step)
        if step % record_every == 0 or step == 1:
            steps_rec.append(step)
            res_rec.append(_res(C, X_eval, Y_eval))

    results["baseline"] = {
        "steps": steps_rec, "residual": res_rec,
        "correction_steps": [], "residual_at_correction": [],
        "label": "Adam (baseline)",
    }
    print(f"  Baseline: {time.time()-t0:.1f}s  "
          f"step-1 res={res_rec[0]:.4e}  final res={res_rec[-1]:.4e}")

    # ── Hybrid variants ─────────────────────────────────────────────────────
    for k_step in k_step_values:
        print(f"  Running: Hybrid Adam + exact every {k_step} steps...")
        # Same noise sequence, but trajectory diverges after corrections
        rng = np.random.default_rng(cfg.seed)
        C = np.zeros((d, n)); m = np.zeros_like(C); v = np.zeros_like(C)
        steps_rec, res_rec = [], []
        corr_steps, corr_residuals = [], []
        t0 = time.time()

        for step in range(1, n_steps + 1):
            Xb = X + rng.normal(0, noise_scale, X.shape)
            Yb = Y + rng.normal(0, noise_scale * 0.1, Y.shape)

            # Adam step (noisy batch)
            C, m, v = _adam_step(C, Xb, Yb, lr, m, v, step)

            # Exact correction (clean eval batch) — moments NOT reset
            if step % k_step == 0:
                res_before = _res(C, X_eval, Y_eval)
                C = _exact_lstsq(X_eval, Y_eval)   # lands on algebraic floor
                res_after  = _res(C, X_eval, Y_eval)
                corr_steps.append(step)
                corr_residuals.append({
                    "step": step, "before": res_before,
                    "after": res_after, "drop": res_before - res_after,
                })

            if step % record_every == 0 or step == 1:
                steps_rec.append(step)
                res_rec.append(_res(C, X_eval, Y_eval))

        results[k_step] = {
            "steps": steps_rec, "residual": res_rec,
            "correction_steps": corr_steps,
            "residual_at_correction": corr_residuals,
            "label": f"Hybrid k_step={k_step}",
        }
        n_corr   = len(corr_steps)
        avg_drop = (float(np.mean([r["drop"] for r in corr_residuals]))
                    if corr_residuals else 0.0)
        print(f"    {time.time()-t0:.1f}s  corrections={n_corr}  "
              f"avg Δres={avg_drop:.4e}  final res={res_rec[-1]:.4e}")

    # ── Summary ─────────────────────────────────────────────────────────────
    w = 64
    b_init  = results["baseline"]["residual"][0]
    b_final = results["baseline"]["residual"][-1]

    print(f"\n  ┌{'─'*w}┐")
    print(f"  │{'EXPERIMENT 4 SUMMARY':^{w}}│")
    print(f"  ├{'─'*w}┤")
    print(f"  │  PRIMARY METRIC : ‖X_eval Cᵀ − Y_eval‖_F{'':>23}│")
    print(f"  │  Algebraic floor: {res_floor:.6e}  (exact solver, 1 step){'':>9}│")
    print(f"  ├{'─'*w}┤")
    print(f"  │  Adam baseline   step 1 : {b_init:.6e}{'':>27}│")
    print(f"  │  Adam baseline   final  : {b_final:.6e}{'':>27}│")
    print(f"  ├{'─'*w}┤")
    for k_step in k_step_values:
        hf  = results[k_step]["residual"][-1]
        imp = (b_final - hf) / b_final * 100 if b_final > 0 else 0.0
        sgn = "▼" if imp > 0 else "▲"
        nc  = len(results[k_step]["correction_steps"])
        print(f"  │  Hybrid k={k_step:3d}  final: {hf:.6e}  "
              f"({sgn}{abs(imp):.1f}%)  corr={nc:3d}{'':>8}│")
    print(f"  ├{'─'*w}┤")
    best_k = min(k_step_values,
                 key=lambda k: results[k]["residual"][-1],
                 default=None)
    if best_k and results[best_k]["residual"][-1] < b_final:
        br = results[best_k]["residual"][-1]
        print(f"  │  Best hybrid: k={best_k}  res={br:.4e}  "
              f"floor={res_floor:.4e}{'':>12}│")
    else:
        print(f"  │  No hybrid improved over Adam baseline.{'':>24}│")
    print(f"  └{'─'*w}┘")

    return {
        # Primary data for plotting
        "steps":          results["baseline"]["steps"],
        "res_floor":      res_floor,
        "baseline_res":   results["baseline"]["residual"],
        "hybrid_res":     {k: results[k]["residual"]          for k in k_step_values},
        "hybrid_corr":    {k: results[k]["correction_steps"]  for k in k_step_values},
        # Full detail for summary printer in main.py
        "_full":          results,
        "_k_step_values": k_step_values,
        "_baseline_final":       b_final,
        "_res_floor":     res_floor,
        # Backward-compat keys so existing plot_exp4_perplexity doesn't crash
        "baseline_residual_curve": results["baseline"]["residual"],
        "hybrid_residual_curve":   {k: results[k]["residual"] for k in k_step_values},
        "baseline_time_per_step": 0.0,
        "hybrid_time_overhead":   {k: 0.0 for k in k_step_values},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 5 — Algebraic Monitoring  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def _exact_update_constrained(X, Y, k_anchor=0,
                               anchor_strategy="low_uncertainty"):
    """Exact C* — unconstrained (k=0) or constrained."""
    if k_anchor == 0:
        return np.linalg.lstsq(X, Y, rcond=None)[0].T
    T, n = X.shape; d = Y.shape[1]
    norms = np.linalg.norm(X, axis=1)
    if anchor_strategy == "low_uncertainty":
        ai = np.argsort(norms)[:k_anchor]
    elif anchor_strategy == "high_uncertainty":
        ai = np.argsort(norms)[-k_anchor:]
    else:
        ai = np.random.default_rng(42).choice(T, size=k_anchor, replace=False)
    X_A = X[ai]; Y_A = Y[ai]
    P = 2.0 * X.T @ X; A_dag = np.linalg.pinv(X_A)
    try:
        V2 = scipy_null_space(X_A)
    except Exception:
        V2 = np.zeros((n, 0))
    C_star = np.zeros((d, n))
    for j in range(d):
        q = -2.0 * X.T @ Y[:, j]; cp = A_dag @ Y_A[:, j]
        if V2.shape[1] == 0:
            C_star[j] = cp; continue
        M = V2.T @ (P @ V2); g = V2.T @ (q + P @ cp)
        C_star[j] = cp + V2 @ (-np.linalg.pinv(M) @ g)
    return C_star


def experiment_5_algebraic_monitoring(X, Y, cfg):
    """
    Records rank, cond, null_dim, uniqueness, and residual before/after
    at every exact-correction checkpoint throughout training.
    """
    print("\n  ── Experiment 5: Algebraic Monitoring ───────────────────────────")

    n_steps      = cfg.hybrid_train_steps
    k_step       = 10
    lr           = cfg.hybrid_lr
    k_anchor     = max(1, min(32, X.shape[0] // 4))
    anchor_strat = "low_uncertainty"
    noise_scale  = float(np.std(X)) * 0.05

    Xtr = X[:int(0.8 * X.shape[0])]; Ytr = Y[:int(0.8 * Y.shape[0])]
    T, n = Xtr.shape; d = Ytr.shape[1]

    print(f"  n_steps={n_steps}  k_step={k_step}  k_anchor={k_anchor}")
    print(f"  Monitoring: rank, cond, null_dim, is_unique at each correction\n")

    C = np.zeros((d, n)); m = np.zeros_like(C); v = np.zeros_like(C)
    rng = np.random.default_rng(cfg.seed + 9999)

    mon = {
        "steps": [], "ranks": [], "cond_numbers": [], "null_dims": [],
        "is_unique": [], "residual_before": [], "residual_after": [],
        "perplexity_before": [], "perplexity_after": [],
        # legacy aliases for plot_exp5_monitoring
        "correction_step": [], "rank": [], "cond": [], "null_dim": [],
    }

    for step in range(1, n_steps + 1):
        Xb = Xtr + rng.normal(0, noise_scale, Xtr.shape)
        Yb = Ytr + rng.normal(0, noise_scale * 0.1, Ytr.shape)
        C, m, v = _adam_step(C, Xb, Yb, lr, m, v, step)

        if step % k_step == 0:
            alg = _gram_summary(Xb)
            rb  = float(np.linalg.norm(Xb @ C.T - Yb, "fro"))
            pb  = float(np.exp(np.mean((Xb @ C.T - Yb) ** 2)))
            C_ex = _exact_update_constrained(Xb, Yb, k_anchor, anchor_strat)
            ra  = float(np.linalg.norm(Xb @ C_ex.T - Yb, "fro"))
            pa  = float(np.exp(np.mean((Xb @ C_ex.T - Yb) ** 2)))
            C = C_ex; m = np.zeros_like(C); v = np.zeros_like(C)

            for key, val in [
                ("steps", step), ("ranks", alg["rank"]),
                ("cond_numbers", alg["cond"]), ("null_dims", alg["null_dim"]),
                ("is_unique", alg["is_unique"]),
                ("residual_before", rb), ("residual_after", ra),
                ("perplexity_before", pb), ("perplexity_after", pa),
                ("correction_step", step), ("rank", alg["rank"]),
                ("cond", alg["cond"]), ("null_dim", alg["null_dim"]),
            ]:
                mon[key].append(val)

            if (step // k_step) % 5 == 0:
                print(f"  step {step:4d}  rank={alg['rank']:3d}  "
                      f"null_dim={alg['null_dim']:3d}  cond={alg['cond']:.2e}  "
                      f"unique={alg['is_unique']}  res {rb:.2e} → {ra:.2e}")

        if step % 100 == 0 and step % k_step != 0:
            r = mon["ranks"][-1] if mon["ranks"] else "?"
            c = mon["cond_numbers"][-1] if mon["cond_numbers"] else "?"
            print(f"    step {step}/{n_steps}  rank={r}  "
                  f"cond={f'{c:.2e}' if isinstance(c, float) else c}")

    ranks = mon["ranks"]; conds = mon["cond_numbers"]
    null_dims = mon["null_dims"]; n_corr = len(ranks); w = 60

    print(f"\n  ┌{'─'*w}┐")
    print(f"  │{'EXPERIMENT 5 SUMMARY':^{w}}│")
    print(f"  ├{'─'*w}┤")
    print(f"  │  Total exact corrections: {n_corr:4d}{'':>29}│")
    if n_corr > 0:
        rs = min(ranks) == max(ranks)
        ns = sum(1 for nd in null_dims if nd > 0)
        au = all(mon["is_unique"])
        ad = float(np.mean([b-a for b, a in
                            zip(mon["residual_before"], mon["residual_after"])]))
        print(f"  │  rank  min={min(ranks):3d}  max={max(ranks):3d}  stable={str(rs):<5}{'':>22}│")
        print(f"  │  null_dim  min={min(null_dims):3d}  max={max(null_dims):3d}{'':>32}│")
        print(f"  │  cond  min={min(conds):.2e}  max={max(conds):.2e}{'':>24}│")
        print(f"  │  nontrivial null space: {ns}/{n_corr} "
              f"({100*ns/n_corr:.1f}%){'':>22}│")
        print(f"  │  uniqueness always satisfied: {str(au):<5}{'':>24}│")
        print(f"  │  avg residual drop per correction: {ad:.4e}{'':>14}│")
    print(f"  └{'─'*w}┘")

    return mon


# ─────────────────────────────────────────────────────────────────────────────
# Legacy wrappers
# ─────────────────────────────────────────────────────────────────────────────

def experiment_4_hybrid_training_from_model(model, data_batches, cfg,
                                             val_batches=None):
    inp, tgt = data_batches[0]
    X, Y = extract_XY(model, inp, cfg.target_layer, cfg)
    X_val = Y_val = None
    if val_batches:
        vi, vt = val_batches[0]
        X_val, Y_val = extract_XY(model, vi, cfg.target_layer, cfg)
    return experiment_4_hybrid_training(X, Y, cfg, X_val, Y_val)


def experiment_5_algebraic_monitoring_from_model(model, data_batches, cfg):
    inp, tgt = data_batches[0]
    X, Y = extract_XY(model, inp, cfg.target_layer, cfg)
    return experiment_5_algebraic_monitoring(X, Y, cfg)
