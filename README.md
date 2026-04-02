# Exact QP Solver for MAMBA

> **An experiment to demonstrate that a part of the training of modern neural networks can be solved exactly, in a single algebraic step, instead of using hundreds of iterations of approximate optimization.**

---

## Table of Contents

1. [The starting point](#1-the-starting-point)
2. [The mathematical problem in plain words](#2-the-mathematical-problem-in-plain-words)
3. [The theory behind it — Lin & Liang (2023)](#3-the-theory-behind-it)
4. [The model used — MAMBA](#4-the-model-used--mamba)
5. [The data — WikiText-2](#5-the-data--wikitext-2)
6. [Experiment design](#6-experiment-design)
7. [How the code is structured](#7-how-the-code-is-structured)
8. [How to run it](#8-how-to-run-it)
9. [The results — how to read them](#9-the-results--how-to-read-them)
10. [What the numbers mean for the EEML abstract](#10-what-they-mean-for-the-eeml-abstract)
11. [Honest limitations](#11-honest-limitations)
12. [References](#12-references)

---

## 1. The Starting Point

### The underlying problem

When training a neural network like MAMBA, you almost always use an iterative optimizer — typically **Adam**. Adam updates all the network parameters simultaneously, one small step at a time, thousands of times.

The key point is this: Adam treats **all parameters the same way**, even those that mathematically do not need iteration.

### The central observation

Inside every MAMBA block there is a layer called the **output projection**. This layer does something very simple:

```
Y = X · Cᵀ
```

where `X` is a matrix of hidden states, `C` is the weight matrix to update, and `Y` is the output. **Finding the optimal C given a fixed batch of X and Y is a linear least-squares problem.** Linear least-squares problems are solved exactly, in a single step, with the pseudoinverse. No iteration is needed. No existing MAMBA implementation exploits this.

---

## 2. The Mathematical Problem in Plain Words

### The unconstrained problem

We want to minimize `‖X·Cᵀ - Y‖_F²`. The exact solution is `C* = (pinv(X) @ Y).T`, computed in a single algebraic step via the Moore–Penrose pseudoinverse.

### The constrained problem (the novelty)

We add an anchor constraint: for a subset of `k` selected rows (anchor positions), the projection must be **exactly correct**:

```
min_C ‖X·Cᵀ - Y‖_F²   subject to   X_A · Cᵀ = Y_A
```

This is also solved exactly in one step, using the null space of the constraint matrix.

### Why the null space matters

The null space of A contains all weight directions that produce zero change in the output at anchor positions. These vectors are completely **invisible** to Adam because the gradient is zero along them. The exact solver identifies all of them and handles them correctly.

---

## 3. The Theory Behind It

**Lin & Liang (2023)** — *Exact Optimization: Part I*, Taiwanese Journal of Mathematics, Vol. 27, No. 1.

This paper proves that a broad class of quadratic optimization problems admits closed-form solutions. The general formula for the constrained problem is:

```
c*_j = A†b - V₂ (V₂ᵀPV₂)† V₂ᵀ (q + PA†b)
```

where `A = X_A`, `V₂` is the null-space basis of A (via SVD), `P = 2XᵀX`, and `q = -2Xᵀy_j`. Our contribution is applying this formula to the output projection of MAMBA — a connection no SSM paper has yet established.

### The uniqueness condition

The solution is unique if and only if `N(X_A) ∩ N(XᵀX) = {0}`. If this fails, infinitely many equally optimal solutions exist. Adam converges toward one of them without knowing anything about the others.

---

## 4. The Model Used — MAMBA

MAMBA (Gu & Dao, 2023) is a family of State Space Models — alternatives to Transformers using a recurrent hidden state mechanism instead of attention. We use **mamba-minimal** (alxndrTL, GitHub), a pure PyTorch implementation of ~300 lines with no proprietary CUDA dependencies, ensuring full reproducibility on CPU-only hardware.

Architecture: `d_model=256`, `d_inner=512`, `d_state=16`, `n_layers=4`. The target layer is `out_proj` in layer 1.

---

## 5. The Data — WikiText-2

WikiText-2 is a standard English corpus (~2M tokens) from Wikipedia, used as a language modeling benchmark. In this experiment the data serves only to produce matrices (X, Y) via a forward hook on `out_proj`, yielding `X: [2048, 512]` and `Y: [2048, 256]` in float64. In offline environments a synthetic Zipf corpus is used; the algebraic properties relevant to experiments 1–3 are invariant to this choice. All solver computations use float64 to preserve machine-precision accuracy (~2.2×10⁻¹⁶).

---

## 6. Experiment Design

```
Experiment 1 → PRIMARY FALSIFICATION
Experiment 2 → ALGEBRAIC STRUCTURE
Experiment 3 → NULL SPACE
Experiment 4 → HYBRID TRAINING
Experiment 5 → ALGEBRAIC MONITORING
```

### Experiment 1 — Residual Comparison

Computes `‖XCᵀ - Y‖_F` for the exact solver (1 step) and Adam with lr ∈ {1e-2, 1e-3, 1e-4} for 500 iterations, both initialized at C=0. This is the direct falsification: either the exact solver reaches machine-epsilon in 1 step, or it does not. Output: `exp1_residual_comparison.png`.

### Experiment 2 — Anchor Sweep

Varies k over `{1, n/4, n/2, n-1}` for three anchor strategies (random, low-uncertainty, high-uncertainty) and records residual, constraint violation, solution freedom dimension, and uniqueness flag. Output: `exp2_anchor_sweep.png`.

### Experiment 3 — Null Space Verification

Computes the singular value spectrum of XᵀX and verifies the algebraic certificate `max_j |g_j^T v| < 1e-6` for all null-space vectors. Output: `exp3_sv_spectrum.png`.

### Experiment 4 — Hybrid Training

Integrates the exact solver as a periodic correction every `k_step ∈ {5, 10, 20, 50}` Adam steps. Training runs for 1000 steps on WikiText-2; perplexity recorded every 50 steps. Output: `exp4_perplexity.png`.

### Experiment 5 — Algebraic Monitoring

Records rank, condition number, null-space dimension, and uniqueness flag of XᵀX at each correction step. Output: `exp5_monitoring.png`.

---

## 7. How the Code is Structured

```
exact_qp_mamba/
├── config.py          ← all hyperparameters
├── data.py            ← loads WikiText-2 (real or synthetic)
├── mamba_minimal.py   ← MAMBA model (pure NumPy, offline-compatible)
├── extract.py         ← forward pass + hook to extract (X, Y)
├── solvers.py         ← exact QP, Adam, null space, constraints
├── experiments.py     ← 5 experiment functions
├── plots.py           ← all figures (300 DPI)
├── main.py            ← entry point
├── generate_tables.py ← LaTeX tables
└── outputs/           ← all generated files
```

### Key functions in `solvers.py`

| Function | What it does |
|---|---|
| `solve_unconstrained_exact` | Pseudoinverse via lstsq |
| `solve_constrained_exact` | Constrained QP, column by column |
| `select_anchor_indices` | Selects the k anchor positions |
| `solve_adam` | Adam in pure NumPy |
| `compute_solution_freedom` | dim(N(X_A) ∩ N(XᵀX)) via SVD |
| `verify_gradient_zero_on_nullspace` | Algebraic optimality certificate |

---

## 8. How to Run It

```bash
# Experiments 1–3 only (fast, ~5 min on CPU)
python main.py --skip-training

# Full pipeline
python main.py

# Smoke test
python main.py --smoke-test --skip-training

# LaTeX tables only
python generate_tables.py
```

---

## 9. The Results — How to Read Them

### `exp1_residual_comparison.png`

X axis: iterations (0–500). Y axis: `‖XCᵀ - Y‖_F` on log scale. The horizontal dashed line is the exact solver at `2.70×10⁻¹⁴`. Adam (lr=1e-3) reaches `1.75×10⁻⁴` after 500 steps and never crosses `10⁻⁸`. Gap: 9 orders of magnitude.

### `exp2_anchor_sweep.png`

With T=2048 > n=512, XᵀX is full-rank (κ=69.1), so N(XᵀX)={0} and the solution is always unique for any k. All points show `solution_freedom=0` and `is_unique=True`.

### `exp3_sv_spectrum.png`

All 512 singular values of XᵀX are positive (rank=512, κ=69.1). No real null space in the overdetermined regime.

### `exp4_perplexity.png`

Hybrid lines show whether periodic corrections accelerate convergence. Effect may be modest since X changes at every step.

### `exp5_monitoring.png`

Rank, condition number, and null-space dimension across training steps. A drop in rank or worsening κ signals potential instability invisible to Adam.

---

## 10. What They Mean for the EEML Abstract

**The key result:**
The exact closed-form QP solver reaches `‖XC* − Y‖_F = 2.70 × 10⁻¹⁴` in **one algebraic step**. Adam (lr=1e-3, 500 iterations) reaches `1.75 × 10⁻⁴` — a factor of 6.5×10⁹ larger — and never reaches `10⁻⁸`.

**Suggested abstract structure:**
Section 1 (motivation): Adam wastes iterations on a subproblem solvable in 1 step. Section 2 (method): constrained QP formula, anchor definition, closed-form solution. Section 3 (experiments): exp1 table and figure; exp2 anchor sweep; exp4 perplexity curve. Section 4 (conclusion): opens the door to hybrid training pipelines; next steps include applying the framework to B and dt projections.

---

## 11. Honest Limitations

| Limitation | Explanation |
|---|---|
| XᵀX full-rank in our case | With T=2048 > n=512 there is no real null space. The richer case (T < n) requires batch size below n. |
| Linearized subproblem | The solver optimizes C on a fixed batch; in real training X changes every step. |
| Experiments 4–5 on simulated data | The NumPy SSM loop is too slow for 1000 real steps. Results are indicative. |
| Untrained model | Random weights; behavior may differ after real pretraining. |

---

## 12. References

```
[1] Lin, L.-G. & Liang, Y.-W. (2023). Exact Optimization: Part I.
    Taiwanese Journal of Mathematics, Vol. 27, No. 1.

[2] Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling
    with Selective State Spaces. arXiv:2312.00752.

[3] alxndrTL (2023). mamba-minimal.
    https://github.com/alxndrTL/mamba-minimal

[4] Merity, S. et al. (2016). Pointer Sentinel Mixture Models.
    arXiv:1609.07843.

[5] Kingma, D. & Ba, J. (2015). Adam: A Method for Stochastic
    Optimization. ICLR 2015.
```

---

*Experiment designed by Antonio Galasso — EEML 2025/2026 Extended Abstract*
*Theory: Lin & Liang (2023) — Substrate: mamba-minimal (alxndrTL)*
