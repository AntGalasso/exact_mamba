"""
plots.py — All plotting functions.

plot_exp1_residual         — improved (adds exact floor line)
plot_exp2_anchor_sweep     — improved (one curve per strategy)
plot_exp3_singular_values  — improved (reads sv_XtX key)
plot_exp4_residual         — NEW: residual curves + correction markers + floor
plot_exp4_perplexity       — alias → plot_exp4_residual
plot_exp5_monitoring       — FIXED: reads correct dict keys
generate_latex_tables      — extended with Exp 4 / Exp 5 rows
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

_COL = {
    "baseline": "#73726c",
    5:  "#1D9E75",
    10: "#378ADD",
    20: "#BA7517",
    50: "#D4537E",
    "floor": "#E24B4A",
}

def _c(k):
    return _COL.get(k, "#888787")


# ─────────────────────────────────────────────────────────────────────────────
def plot_exp1_residual(res1, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    curves = res1.get("adam_curves", {})
    exact  = res1.get("exact_residual")

    fig, ax = plt.subplots(figsize=(8, 4))
    for lr, vals in sorted(curves.items()):
        ax.plot(vals, label=f"Adam lr={lr:.0e}", linewidth=1.5)
    if exact is not None:
        ax.axhline(exact, color=_COL["floor"], linestyle="--",
                   linewidth=1.8, label=f"Exact (1 step) = {exact:.2e}")
    ax.set_yscale("log")
    ax.set_xlabel("Adam steps", fontsize=10)
    ax.set_ylabel("‖XCᵀ − Y‖_F", fontsize=10)
    ax.set_title("Experiment 1 — Residual: Adam vs exact solver", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp1_residual.png"), dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def plot_exp2_anchor_sweep(res2, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    by_strat = defaultdict(list)
    for r in res2:
        by_strat[r["strategy"]].append((r["k"], r.get("residual", np.nan)))

    fig, ax = plt.subplots(figsize=(8, 4))
    for (strat, pts), mk in zip(sorted(by_strat.items()), ["o", "s", "^"]):
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker=mk, label=strat, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("k (anchor positions)", fontsize=10)
    ax.set_ylabel("‖XCᵀ − Y‖_F", fontsize=10)
    ax.set_title("Experiment 2 — Anchor sweep", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp2_anchor_sweep.png"), dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def plot_exp3_singular_values(res3, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sv = res3.get("sv_XtX", res3.get("singular_values", []))
    if len(sv) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sv, color="#378ADD", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Index", fontsize=10)
    ax.set_ylabel("Singular value", fontsize=10)
    rank = res3.get("rank_XtX"); cond = res3.get("cond_XtX")
    title = "Experiment 3 — Singular values of XᵀX"
    if rank and cond:
        title += f"  (rank={rank}, κ={cond:.2e})"
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp3_sv_spectrum.png"), dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def plot_exp4_residual(res4, output_dir):
    """
    Left: ‖X_eval Cᵀ−Y_eval‖_F over training steps.
      grey dashed  = Adam baseline
      coloured     = hybrid variants
      red dotted   = algebraic floor (exact solver, 1 step)
      vertical marks = correction events
    Right: |Δresidual| at each correction event (k_step = smallest).
    """
    os.makedirs(output_dir, exist_ok=True)

    steps      = res4.get("steps", [])
    base_res   = res4.get("baseline_res",  res4.get("baseline_residual_curve", []))
    hybrid_res = res4.get("hybrid_res",    res4.get("hybrid_residual_curve",   {}))
    hybrid_corr= res4.get("hybrid_corr",   {})
    res_floor  = res4.get("res_floor",     res4.get("_res_floor"))
    k_steps    = res4.get("_k_step_values", sorted(hybrid_res.keys()))
    full       = res4.get("_full", {})

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1], wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # baseline
    ax1.plot(steps, base_res, color=_COL["baseline"], linestyle="--",
             linewidth=1.8, label="Adam baseline", zorder=3)

    # hybrid curves + correction tick marks
    for k in k_steps:
        if k not in hybrid_res:
            continue
        col = _c(k)
        ax1.plot(steps, hybrid_res[k], color=col, linewidth=1.6,
                 label=f"Hybrid k={k}", zorder=4)
        corr_s = hybrid_corr.get(k, [])
        if corr_s:
            step2res = dict(zip(steps, hybrid_res[k]))
            ys = [step2res.get(s, np.nan) for s in corr_s]
            ax1.vlines(corr_s, ymin=0, ymax=ys,
                       color=col, alpha=0.20, linewidth=0.8, zorder=2)

    # algebraic floor
    if res_floor is not None:
        ax1.axhline(res_floor, color=_COL["floor"], linestyle=":",
                    linewidth=2.0, zorder=5,
                    label=f"Algebraic floor = {res_floor:.2e}  (1 step)")

    ax1.set_yscale("log")
    ax1.set_xlabel("Training step", fontsize=10)
    ax1.set_ylabel("‖X_eval Cᵀ − Y_eval‖_F  (log scale)", fontsize=10)
    ax1.set_title(
        "Experiment 4 — Residual trajectory\n"
        "Adam baseline vs Adam + periodic exact correction", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.20)

    # right panel: drop per correction for smallest k_step
    best_k = k_steps[0] if k_steps else None
    if best_k and best_k in full:
        cd = full[best_k].get("residual_at_correction", [])
        if cd:
            cs    = [r["step"] for r in cd]
            drops = [abs(r["drop"]) + 1e-30 for r in cd]
            ax2.semilogy(cs, drops, color=_c(best_k), linewidth=1.5,
                         marker="o", markersize=2.5)
            ax2.set_xlabel("Correction step", fontsize=10)
            ax2.set_ylabel("|Δresidual|  (log scale)", fontsize=10)
            ax2.set_title(f"Residual drop per correction\n(k_step={best_k})",
                          fontsize=10)
            ax2.grid(True, alpha=0.20)
        else:
            ax2.text(0.5, 0.5, "No correction data", ha="center", va="center",
                     transform=ax2.transAxes, color="grey")
    else:
        ax2.text(0.5, 0.5, "No correction data", ha="center", va="center",
                 transform=ax2.transAxes, color="grey")

    fig.suptitle("Experiment 4 — Hybrid Training Protocol", fontsize=12, y=1.01)
    out_path = os.path.join(output_dir, "exp4_residual.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_exp4_perplexity(res4, output_dir):
    """Alias kept for backward compatibility."""
    plot_exp4_residual(res4, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
def plot_exp5_monitoring(res5, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    steps     = res5.get("steps",        res5.get("correction_step", []))
    ranks     = res5.get("ranks",        res5.get("rank",            []))
    conds     = res5.get("cond_numbers", res5.get("cond",            []))
    null_dims = res5.get("null_dims",    res5.get("null_dim",        []))

    if not steps:
        print("  plot_exp5_monitoring: no data, skipping")
        return

    log_cond = [np.log10(max(c, 1e-30)) for c in conds]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(steps, ranks, color="#378ADD", linewidth=1.6,
                 marker="o", markersize=2.5)
    axes[0].set_ylabel("rank(XᵀX)", fontsize=10)
    axes[0].set_title(
        "Experiment 5 — Algebraic monitoring throughout training", fontsize=10)
    axes[0].grid(True, alpha=0.20)

    axes[1].plot(steps, log_cond, color="#BA7517", linewidth=1.6,
                 marker="o", markersize=2.5)
    axes[1].set_ylabel("log₁₀ κ(XᵀX)", fontsize=10)
    axes[1].grid(True, alpha=0.20)

    bar_col = ["#E24B4A" if nd > 0 else "#1D9E75" for nd in null_dims]
    axes[2].bar(range(len(steps)), null_dims, color=bar_col,
                alpha=0.80, width=0.85)
    axes[2].set_ylabel("null_dim(XᵀX)", fontsize=10)
    axes[2].set_xlabel("Correction index", fontsize=10)
    axes[2].grid(True, alpha=0.20, axis="y")

    for i, nd in enumerate(null_dims):
        if nd > 0:
            axes[2].annotate(f"step {steps[i]}",
                             xy=(i, nd), xytext=(i, nd + 0.3),
                             fontsize=7, ha="center", color="#E24B4A")

    fig.tight_layout()
    out_path = os.path.join(output_dir, "exp5_monitoring.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def generate_latex_tables(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tex_path = os.path.join(output_dir, "reproducibility_table.tex")

    exp1 = all_results.get("exp1", {})
    exp4 = all_results.get("exp4", {})
    exp5 = all_results.get("exp5", {})

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated reproducibility table\n")
        f.write("\\begin{tabular}{llr}\n\\hline\n")
        f.write("Experiment & Metric & Value \\\\\n\\hline\n")

        er = exp1.get("exact_residual", float("nan"))
        yn = exp1.get("Y_norm",         float("nan"))
        bl = exp1.get("best_adam_lr",   float("nan"))
        cr = exp1.get("adam_curves",    {})
        bf = cr[bl][-1] if bl in cr else float("nan")
        f.write(f"Exp 1 & $\\|Y\\|_F$ & {yn:.6e} \\\\\n")
        f.write(f"Exp 1 & Exact residual (1 step) & {er:.6e} \\\\\n")
        f.write(f"Exp 1 & Best Adam final & {bf:.6e} \\\\\n")
        f.write(f"Exp 1 & Steps to $10^{{-8}}$ & "
                f"{exp1.get('steps_to_1e8','None')} \\\\\n\\hline\n")

        if exp4:
            floor = exp4.get("_res_floor", exp4.get("res_floor", float("nan")))
            bfin  = exp4.get("_b_final", float("nan"))
            f.write(f"Exp 4 & Algebraic floor & {floor:.6e} \\\\\n")
            f.write(f"Exp 4 & Adam baseline final & {bfin:.6e} \\\\\n")
            for k in exp4.get("_k_step_values", []):
                hr = exp4.get("hybrid_res", {}).get(k, [])
                if hr:
                    f.write(f"Exp 4 & Hybrid $k={k}$ final & {hr[-1]:.6e} \\\\\n")
            f.write("\\hline\n")

        if exp5:
            ranks = exp5.get("ranks", exp5.get("rank", []))
            conds = exp5.get("cond_numbers", exp5.get("cond", []))
            nds   = exp5.get("null_dims", exp5.get("null_dim", []))
            rb    = exp5.get("residual_before", [])
            ra    = exp5.get("residual_after",  [])
            if ranks:
                f.write(f"Exp 5 & rank range & {min(ranks)}--{max(ranks)} \\\\\n")
                f.write(f"Exp 5 & $\\kappa$ range & "
                        f"{min(conds):.2e}--{max(conds):.2e} \\\\\n")
                ns = sum(1 for nd in nds if nd > 0)
                f.write(f"Exp 5 & Null dim $> 0$ & {ns}/{len(nds)} \\\\\n")
                if rb and ra:
                    ad = float(np.mean([b-a for b, a in zip(rb, ra)]))
                    f.write(f"Exp 5 & Avg res drop & {ad:.4e} \\\\\n")
            f.write("\\hline\n")

        f.write("\\end{tabular}\n")
    print(f"  Saved: {tex_path}")
