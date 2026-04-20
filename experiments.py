"""
experiments.py — Five experiment functions, all pure NumPy/SciPy.

Experiments 1–3: algebraic (unchanged).
Experiments 4–5: hybrid training + algebraic monitoring, fully implemented
                 using the exact Adam simulation and QP solver from
                 experiments_4_5.py (integrated here).
"""

import time
import copy
import math
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
# Experiment 1 — Residual Comparison  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def experiment_1_residual_comparison(
    X: np.ndarray,
    Y: np.ndarray,
    cfg: Config,
) -> dict:
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
        "exact_residual":  exact_residual,
        "exact_result":    exact,
        "adam_curves":     adam_curves,
        "best_adam_lr":    best_lr,
        "Y_norm":          Y_norm,
        "steps_to_1e8":    s8,
        "steps_to_1e10":   s10,
        "steps_to_1e14":   s14,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 — Anchor Sweep  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def experiment_2_anchor_sweep(
    X: np.ndarray,
    Y: np.ndarray,
    cfg: Config,
) -> list:
    print("\n  ── Experiment 2: Anchor Sweep ───────────────────────────────")
    n  = X.shape[1]
    T  = X.shape[0]
    results = []

    for strategy in cfg.anchor_strategies:
        for k_frac in cfg.anchor_k_fractions:
            k = max(1, int(k_frac * n))
            k = min(k, T)

            anchor_idx = select_anchor_indices(X, Y, k, strategy)

            try:
                sol = solve_constrained_exact(X, Y, k, anchor_idx)
                freedom  = sol["solution_freedom"]
                is_uniq  = sol["is_unique"]
                residual = sol["residual"]
                cv       = sol["constraint_viol"]
                status   = "OK"
            except ValueError as e:
                print(f"    WARNING [{strategy} k={k}]: {e}")
                freedom = residual = cv = np.nan
                is_uniq = False
                status  = "FAILED"

            rec = dict(strategy=strategy, k_frac=k_frac, k=k,
                       residual=residual, constraint_viol=cv,
                       solution_freedom=freedom, is_unique=is_uniq, status=status)
            results.append(rec)

            tag = "UNIQUE" if is_uniq else f"freedom={freedom}"
            print(f"    {strategy:22s}  k={k:5d} (frac={k_frac:.2f})  "
                  f"res={residual:.3e}  cv={cv:.1e}  [{tag}]")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3 — Null Space Characterisation  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def experiment_3_null_space(
    X: np.ndarray,
    Y: np.ndarray,
    cfg: Config,
) -> dict:
    print("\n  ── Experiment 3: Null Space Characterisation ────────────────")
    n = X.shape[1]

    XtX  = X.T @ X
    sv   = np.linalg.svd(XtX, compute_uv=False)
    tol  = sv[0] * max(XtX.shape) * np.finfo(np.float64).eps
    rank = int(np.sum(sv > tol))
    null_dim = n - rank
    cond = float(sv[0] / sv[rank - 1]) if rank > 0 else np.inf

    print(f"  rank(XᵀX)={rank}/{n}  cond={cond:.4e}  null_dim={null_dim}")

    exact   = solve_unconstrained_exact(X, Y)
    C_star  = exact["C_star"]

    V2_X = scipy_null_space(X, rcond=1e-10)
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
        M_int   = V2_A.T @ W_P
        sv_int  = np.linalg.svd(M_int, compute_uv=False)
        tol_int = sv_int[0] * max(M_int.shape) * np.finfo(float).eps if len(sv_int) > 0 else 0
        U_int, S_int, Vt_int = np.linalg.svd(M_int, full_matrices=False)
        good = S_int > tol_int
        if good.sum() > 0:
            V2_intersect = W_P @ Vt_int[good].T
            grad_A = verify_gradient_zero_on_nullspace(X, Y, V2_intersect, sol_A["C_star"])
            ok_A   = "✓ verified" if grad_A["verified"] else "✗ FAILED"
            dim_int = good.sum()
            print(f"  dim(N(X_A) ∩ N(XᵀX)) [k={k_rep}]: {dim_int}")
            print(f"  max |gᵀv| on intersection: {grad_A['max_grad_null']:.4e}  {ok_A}")
        else:
            print(f"  N(X_A) ∩ N(XᵀX) = {{0}}  (intersection trivial) ✓")
    else:
        print(f"  N(X_A) or N(XᵀX) empty — intersection trivial ✓")

    return {
        "sv_XtX":             sv,
        "rank_XtX":           rank,
        "cond_XtX":           cond,
        "null_dim_XtX":       null_dim,
        "grad_null_verified": grad_check["verified"],
        "max_grad_null":      grad_check["max_grad_null"],
        "V2_X":               V2_X,
        "C_star":             C_star,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers shared by Experiments 4 and 5
# ─────────────────────────────────────────────────────────────────────────────

def _adam_step(
    C: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    lr: float,
    m: np.ndarray,
    v: np.ndarray,
    t: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
):
    """One Adam step on 0.5‖XCᵀ − Y‖²_F w.r.t. C.  Gradient shape: (d, n)."""
    grad = (X @ C.T - Y).T @ X       # (d, n)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    C = C - lr * m_hat / (np.sqrt(v_hat) + eps)
    return C, m, v


def _exact_update(
    X: np.ndarray,
    Y: np.ndarray,
    k_anchor: int = 0,
    anchor_strategy: str = "low_uncertainty",
) -> np.ndarray:
    """
    Exact optimal C* for a batch (X, Y).

    k_anchor=0  →  unconstrained minimum-norm solution via lstsq.
    k_anchor>0  →  equality-constrained QP solver (Lin & Liang 2023).
    """
    if k_anchor == 0:
        return np.linalg.lstsq(X, Y, rcond=None)[0].T   # (d, n)

    T, n = X.shape
    d    = Y.shape[1]

    norms = np.linalg.norm(X, axis=1)
    if anchor_strategy == "low_uncertainty":
        anchor_idx = np.argsort(norms)[:k_anchor]
    elif anchor_strategy == "high_uncertainty":
        anchor_idx = np.argsort(norms)[-k_anchor:]
    else:
        rng = np.random.default_rng(42)
        anchor_idx = rng.choice(T, size=k_anchor, replace=False)

    X_A   = X[anchor_idx]
    Y_A   = Y[anchor_idx]
    P     = 2.0 * X.T @ X
    A_dag = np.linalg.pinv(X_A)

    try:
        V2 = scipy_null_space(X_A)
    except Exception:
        V2 = np.zeros((n, 0))

    C_star = np.zeros((d, n))
    for j in range(d):
        q           = -2.0 * X.T @ Y[:, j]
        c_part      = A_dag @ Y_A[:, j]
        if V2.shape[1] == 0:
            C_star[j] = c_part
            continue
        M     = V2.T @ (P @ V2)
        g     = V2.T @ (q + P @ c_part)
        y_star = -np.linalg.pinv(M) @ g
        C_star[j] = c_part + V2 @ y_star

    return C_star


def _ppl_proxy(C: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    """exp(MSE) — monotone proxy for language-model perplexity."""
    return float(np.exp(np.mean((X @ C.T - Y) ** 2)))


def _gram_summary(X: np.ndarray) -> dict:
    """Rank, cond, null_dim, is_unique of XᵀX."""
    XtX = X.T @ X
    sv  = np.linalg.svd(XtX, compute_uv=False)
    tol = sv[0] * XtX.shape[0] * np.finfo(float).eps
    rank = int(np.sum(sv > tol))
    null_dim = XtX.shape[0] - rank
    cond = float(sv[0] / sv[rank - 1]) if rank > 0 else np.inf
    return {"rank": rank, "cond": cond, "null_dim": null_dim, "is_unique": null_dim == 0}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4 — Hybrid Training Protocol
# ─────────────────────────────────────────────────────────────────────────────

def experiment_4_hybrid_training(
    X: np.ndarray,
    Y: np.ndarray,
    cfg: Config,
    X_val: np.ndarray = None,
    Y_val: np.ndarray = None,
) -> dict:
    """
    Pure Adam baseline vs Adam + periodic exact-QP correction.

    Uses the (X, Y) pair from the fixed batch (same as Experiments 1–3)
    with 5% Gaussian noise to simulate batch variation across steps.
    A static 80/20 train/val split is applied for perplexity reporting.

    Returns a dict compatible with plot_exp4_perplexity.
    """
    print("\n  ── Experiment 4: Hybrid Training Protocol ──────────────────────")

    n_steps       = cfg.hybrid_train_steps
    k_step_values = cfg.hybrid_k_steps
    lr            = cfg.hybrid_lr
    record_every  = cfg.hybrid_eval_every
    k_anchor      = max(1, min(32, X.shape[0] // 4))
    anchor_strat  = "low_uncertainty"

    # train / val split
    split = int(0.8 * X.shape[0])
    Xtr, Ytr = X[:split], Y[:split]
    Xv,  Yv  = X[split:], Y[split:]
    if X_val is not None:
        Xv, Yv = X_val, Y_val

    T, n = Xtr.shape
    d    = Ytr.shape[1]
    noise_scale = float(np.std(Xtr)) * 0.05

    print(f"  n_steps={n_steps}  k_anchor={k_anchor}  "
          f"strategy={anchor_strat}  lr={lr}")
    print(f"  k_step variants: {k_step_values}")
    print(f"  Recording every {record_every} steps\n")

    rng_noise = np.random.default_rng(cfg.seed)

    def make_batch():
        """Return a lightly perturbed training batch."""
        Xb = Xtr + rng_noise.normal(0, noise_scale, Xtr.shape)
        Yb = Ytr + rng_noise.normal(0, noise_scale * 0.1, Ytr.shape)
        return Xb, Yb

    results = {}

    # ── Pure Adam baseline ──────────────────────────────────────────────────
    print("  Running: Pure Adam baseline...")
    C = np.zeros((d, n)); m = np.zeros_like(C); v = np.zeros_like(C)
    steps_rec, ppl_rec = [], []
    t0 = time.time()
    for step in range(1, n_steps + 1):
        Xb, Yb = make_batch()
        C, m, v = _adam_step(C, Xb, Yb, lr, m, v, step)
        if step % record_every == 0 or step == 1:
            steps_rec.append(step); ppl_rec.append(_ppl_proxy(C, Xv, Yv))
    results["baseline"] = {
        "steps": steps_rec, "perplexity": ppl_rec,
        "correction_steps": [], "residual_at_correction": [],
        "label": "Adam (baseline)",
    }
    print(f"  Baseline done in {time.time()-t0:.1f}s  "
          f"final ppl proxy = {ppl_rec[-1]:.6f}")

    # ── Hybrid variants ─────────────────────────────────────────────────────
    for k_step in k_step_values:
        print(f"  Running: Hybrid Adam + exact every {k_step} steps...")
        C = np.zeros((d, n)); m = np.zeros_like(C); v = np.zeros_like(C)
        steps_rec, ppl_rec = [], []
        corr_steps, corr_residuals = [], []
        rng_noise = np.random.default_rng(cfg.seed)   # reset noise for fairness
        t0 = time.time()
        for step in range(1, n_steps + 1):
            Xb, Yb = make_batch()
            C, m, v = _adam_step(C, Xb, Yb, lr, m, v, step)

            if step % k_step == 0:
                res_before = float(np.linalg.norm(Xb @ C.T - Yb, "fro"))
                C_exact = _exact_update(Xb, Yb, k_anchor=k_anchor,
                                        anchor_strategy=anchor_strat)
                res_after = float(np.linalg.norm(Xb @ C_exact.T - Yb, "fro"))
                C = C_exact
                m = np.zeros_like(C); v = np.zeros_like(C)   # reset moments
                corr_steps.append(step)
                corr_residuals.append({
                    "step": step,
                    "before": res_before,
                    "after":  res_after,
                    "drop":   res_before - res_after,
                })

            if step % record_every == 0 or step == 1:
                steps_rec.append(step); ppl_rec.append(_ppl_proxy(C, Xv, Yv))

        results[k_step] = {
            "steps": steps_rec, "perplexity": ppl_rec,
            "correction_steps": corr_steps,
            "residual_at_correction": corr_residuals,
            "label": f"Hybrid k_step={k_step}",
        }
        n_corr = len(corr_steps)
        avg_drop = (float(np.mean([r["drop"] for r in corr_residuals]))
                    if corr_residuals else 0.0)
        print(f"    Done in {time.time()-t0:.1f}s  |  corrections: {n_corr}  "
              f"|  avg Δresidual: {avg_drop:.4e}  "
              f"|  final ppl proxy: {ppl_rec[-1]:.6f}")

    # ── Summary box ─────────────────────────────────────────────────────────
    w = 60
    baseline_final = results["baseline"]["perplexity"][-1]
    print(f"\n  ┌{'─'*w}┐")
    print(f"  │{'EXPERIMENT 4 SUMMARY':^{w}}│")
    print(f"  ├{'─'*w}┤")
    print(f"  │  Adam baseline final ppl proxy:  {baseline_final:.6f}{'':>20}│")
    for k_step in k_step_values:
        hf  = results[k_step]["perplexity"][-1]
        imp = (baseline_final - hf) / baseline_final * 100
        sgn = "▼" if imp > 0 else "▲"
        print(f"  │  Hybrid k_step={k_step:3d} final ppl:   "
              f"{hf:.6f}  ({sgn}{abs(imp):.2f}%){'':>10}│")
    print(f"  └{'─'*w}┘")

    # Build backward-compatible keys for plot_exp4_perplexity
    return {
        "baseline_perplexity":    results["baseline"]["perplexity"],
        "hybrid_perplexity":      {k: results[k]["perplexity"] for k in k_step_values},
        "baseline_time_per_step": 0.0,
        "hybrid_time_overhead":   {k: 0.0 for k in k_step_values},
        # full detail for the extended summary
        "_full": results,
        "_k_step_values": k_step_values,
        "_baseline_final": baseline_final,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 5 — Algebraic Monitoring
# ─────────────────────────────────────────────────────────────────────────────

def experiment_5_algebraic_monitoring(
    X: np.ndarray,
    Y: np.ndarray,
    cfg: Config,
) -> dict:
    """
    Hybrid training loop that records rank, cond, null_dim, and
    uniqueness of XᵀX at every exact-correction checkpoint.

    Batch variation is simulated via small Gaussian perturbations
    (5% of std(X)) at each step, identical to Experiment 4.

    Returns a dict compatible with plot_exp5_monitoring.
    """
    print("\n  ── Experiment 5: Algebraic Monitoring ───────────────────────────")

    n_steps      = cfg.hybrid_train_steps
    k_step       = 10                              # correction frequency
    lr           = cfg.hybrid_lr
    k_anchor     = max(1, min(32, X.shape[0] // 4))
    anchor_strat = "low_uncertainty"
    noise_scale  = float(np.std(X)) * 0.05

    split = int(0.8 * X.shape[0])
    Xtr, Ytr = X[:split], Y[:split]

    T, n = Xtr.shape
    d    = Ytr.shape[1]

    print(f"  n_steps={n_steps}  k_step={k_step}  k_anchor={k_anchor}")
    print(f"  Monitoring: rank, cond, null_dim, is_unique at each correction\n")

    C = np.zeros((d, n))
    m = np.zeros_like(C)
    v = np.zeros_like(C)
    rng_noise = np.random.default_rng(cfg.seed + 9999)

    monitoring = {
        "steps":             [],
        "ranks":             [],
        "cond_numbers":      [],
        "null_dims":         [],
        "is_unique":         [],
        "residual_before":   [],
        "residual_after":    [],
        "perplexity_before": [],
        "perplexity_after":  [],
        # legacy keys expected by plot_exp5_monitoring
        "correction_step":   [],
        "rank":              [],
        "cond":              [],
        "null_dim":          [],
    }

    for step in range(1, n_steps + 1):
        Xb = Xtr + rng_noise.normal(0, noise_scale, Xtr.shape)
        Yb = Ytr + rng_noise.normal(0, noise_scale * 0.1, Ytr.shape)
        C, m, v = _adam_step(C, Xb, Yb, lr, m, v, step)

        if step % k_step == 0:
            alg  = _gram_summary(Xb)
            rb   = float(np.linalg.norm(Xb @ C.T - Yb, "fro"))
            pb   = _ppl_proxy(C, Xb, Yb)
            C_ex = _exact_update(Xb, Yb, k_anchor=k_anchor,
                                 anchor_strategy=anchor_strat)
            ra   = float(np.linalg.norm(Xb @ C_ex.T - Yb, "fro"))
            pa   = _ppl_proxy(C_ex, Xb, Yb)
            C = C_ex
            m = np.zeros_like(C); v = np.zeros_like(C)

            monitoring["steps"].append(step)
            monitoring["ranks"].append(alg["rank"])
            monitoring["cond_numbers"].append(alg["cond"])
            monitoring["null_dims"].append(alg["null_dim"])
            monitoring["is_unique"].append(alg["is_unique"])
            monitoring["residual_before"].append(rb)
            monitoring["residual_after"].append(ra)
            monitoring["perplexity_before"].append(pb)
            monitoring["perplexity_after"].append(pa)
            # legacy aliases
            monitoring["correction_step"].append(step)
            monitoring["rank"].append(alg["rank"])
            monitoring["cond"].append(alg["cond"])
            monitoring["null_dim"].append(alg["null_dim"])

            if (step // k_step) % 5 == 0:
                print(f"  step {step:4d}  rank={alg['rank']:3d}  "
                      f"null_dim={alg['null_dim']:3d}  "
                      f"cond={alg['cond']:.2e}  "
                      f"unique={alg['is_unique']}  "
                      f"res {rb:.2e} → {ra:.2e}")

        if (step % 100 == 0) and (step % k_step != 0):
            r  = monitoring["ranks"][-1]  if monitoring["ranks"]  else "?"
            c  = monitoring["cond_numbers"][-1] if monitoring["cond_numbers"] else "?"
            cs = f"{c:.2e}" if isinstance(c, float) else str(c)
            print(f"    step {step}/{n_steps}  rank={r}  cond={cs}")

    # ── Summary box ─────────────────────────────────────────────────────────
    ranks     = monitoring["ranks"]
    conds     = monitoring["cond_numbers"]
    null_dims = monitoring["null_dims"]
    n_corr    = len(ranks)
    w = 60

    print(f"\n  ┌{'─'*w}┐")
    print(f"  │{'EXPERIMENT 5 SUMMARY':^{w}}│")
    print(f"  ├{'─'*w}┤")
    print(f"  │  Total exact corrections recorded: {n_corr:4d}{'':>20}│")
    if n_corr > 0:
        rank_stable = min(ranks) == max(ranks)
        n_sing = sum(1 for nd in null_dims if nd > 0)
        all_uniq = all(monitoring["is_unique"])
        avg_drop = float(np.mean([
            b - a for b, a in zip(monitoring["residual_before"],
                                  monitoring["residual_after"])
        ]))
        print(f"  │  rank  — min: {min(ranks):3d}  max: {max(ranks):3d}  "
              f"stable: {str(rank_stable):<5}{'':>20}│")
        print(f"  │  null_dim — min: {min(null_dims):3d}  max: {max(null_dims):3d}{'':>30}│")
        print(f"  │  cond  — min: {min(conds):.2e}  max: {max(conds):.2e}{'':>22}│")
        print(f"  │  batches with nontrivial null space: "
              f"{n_sing}/{n_corr} ({100*n_sing/n_corr:.1f}%){'':>12}│")
        print(f"  │  uniqueness always satisfied: {str(all_uniq):<5}{'':>24}│")
        print(f"  │  avg residual drop per correction: {avg_drop:.4e}{'':>14}│")
    print(f"  └{'─'*w}┘")

    return monitoring


# ─────────────────────────────────────────────────────────────────────────────
# Legacy wrappers — keep old call signatures from main.py working
# (main.py passes model + data_batches; these wrappers extract X, Y first)
# ─────────────────────────────────────────────────────────────────────────────

def experiment_4_hybrid_training_from_model(model, data_batches, cfg, val_batches=None):
    """
    Legacy entry point used when model + raw batches are available
    (original main.py path). Extracts (X, Y) then delegates to the
    algebraically-correct implementation above.
    """
    inp, tgt = data_batches[0]
    X, Y = extract_XY(model, inp, cfg.target_layer, cfg)

    X_val = Y_val = None
    if val_batches:
        vi, vt = val_batches[0]
        X_val, Y_val = extract_XY(model, vi, cfg.target_layer, cfg)

    return experiment_4_hybrid_training(X, Y, cfg, X_val, Y_val)


def experiment_5_algebraic_monitoring_from_model(model, data_batches, cfg):
    """
    Legacy entry point. Extracts (X, Y) then delegates.
    """
    inp, tgt = data_batches[0]
    X, Y = extract_XY(model, inp, cfg.target_layer, cfg)
    return experiment_5_algebraic_monitoring(X, Y, cfg)
