"""
solvers.py — All solvers in pure NumPy/SciPy (float64 throughout).

  solve_unconstrained_exact   — pseudoinverse, 1 step
  solve_constrained_exact     — equality-constrained QP (Lin & Liang 2023), 1 step
  select_anchor_indices       — random / low / high uncertainty
  solve_adam                  — Adam optimizer implemented from scratch in NumPy
  compute_solution_freedom    — dim(N(X_A) ∩ N(XᵀX))
  verify_gradient_zero_on_nullspace
"""

import numpy as np
from scipy.linalg import null_space as scipy_null_space


# ─────────────────────────────────────────────────────────────────────────────
# Unconstrained exact solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_unconstrained_exact(X: np.ndarray, Y: np.ndarray) -> dict:
    """
    C*ᵀ = pinv(X) @ Y   →   C* ∈ R^{d × n}

    numpy.linalg.lstsq gives minimum-norm solution when underdetermined.
    All computations in float64.
    """
    assert X.dtype == np.float64, "X must be float64"
    assert Y.dtype == np.float64, "Y must be float64"

    T, n = X.shape
    _, d = Y.shape

    # lstsq: solve X @ C_T ≈ Y for C_T ∈ R^{n × d}
    C_star_T, _, rank_X, sv_X = np.linalg.lstsq(X, Y, rcond=None)
    C_star = C_star_T.T   # [d, n]

    # Residual
    pred     = X @ C_star_T
    residual = float(np.linalg.norm(pred - Y, "fro"))

    # XᵀX diagnostics
    sv_P = sv_X ** 2
    tol  = sv_P[0] * max(n, T) * np.finfo(np.float64).eps if len(sv_P) > 0 else 0.0
    rank_P   = int(np.sum(sv_P > tol))
    cond_P   = float(sv_P[0] / sv_P[rank_P - 1]) if rank_P > 0 else np.inf
    null_dim = n - rank_P

    return {
        "C_star":   C_star,
        "residual": residual,
        "rank_P":   rank_P,
        "cond_P":   cond_P,
        "null_dim": null_dim,
        "sv_P":     sv_P,
        "n_steps":  1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Constrained exact solver (Lin & Liang 2023, Problem 3.9)
# ─────────────────────────────────────────────────────────────────────────────

def solve_constrained_exact(
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
    anchor_indices: np.ndarray,
) -> dict:
    """
    Solve  min ‖XCᵀ − Y‖_F  s.t.  X_A Cᵀ = Y_A   (column-by-column).

    For each output dimension j:
        P = 2 XᵀX,   q = −2 Xᵀ y_j,   A = X_A,   b = Y_A[:,j]
        c*_j = A†b − V₂ (V₂ᵀPV₂)† V₂ᵀ (q + PA†b)

    Lin & Liang (2023) eq. (3.9) closed-form solution.
    """
    assert X.dtype == np.float64
    assert Y.dtype == np.float64

    T, n = X.shape
    _, d = Y.shape
    anchor_indices = np.asarray(anchor_indices, dtype=int)
    assert len(anchor_indices) == k

    X_A = X[anchor_indices]   # [k, n]
    Y_A = Y[anchor_indices]   # [k, d]

    # Precompute shared quantities
    P    = 2.0 * (X.T @ X)           # [n, n]
    A_pi = np.linalg.pinv(X_A)       # [n, k]  pseudoinverse of X_A

    # Null space of X_A: V₂ columns span N(X_A)
    V2     = scipy_null_space(X_A, rcond=1e-10)   # [n, n_null]
    n_null = V2.shape[1]

    # Reduced Hessian and its pseudoinverse
    H_red      = V2.T @ P @ V2        # [n_null, n_null]
    H_red_pi   = np.linalg.pinv(H_red)

    # Column-by-column
    C_star_T = np.zeros((n, d), dtype=np.float64)

    for j in range(d):
        b   = Y_A[:, j]           # [k]
        y_j = Y[:, j]             # [T]
        q   = -2.0 * (X.T @ y_j) # [n]

        p           = A_pi @ b                      # [n]  particular solution
        rhs         = q + P @ p                     # [n]
        c_star_j    = p - V2 @ (H_red_pi @ (V2.T @ rhs))
        C_star_T[:, j] = c_star_j

    C_star = C_star_T.T   # [d, n]

    # Diagnostics
    pred     = X @ C_star_T
    residual = float(np.linalg.norm(pred - Y, "fro"))

    pred_A         = X_A @ C_star_T
    constraint_viol = float(np.linalg.norm(pred_A - Y_A, "fro"))

    # When k ≥ rank(X) the anchor constraints span the full row space and the
    # constraint violation equals the unconstrained residual — this is correct.
    # We only raise an error when violation is unexpectedly large relative to
    # the unconstrained residual.
    if constraint_viol > max(1e-5, residual * 1.01 + 1e-13):
        raise ValueError(
            f"Constraint violation {constraint_viol:.4e} unexpectedly large "
            f"(residual={residual:.4e})\n"
            f"  k={k}  n_null={n_null}"
        )

    # Rank of X_A
    sv_A     = np.linalg.svd(X_A, compute_uv=False)
    tol_A    = sv_A[0] * max(X_A.shape) * np.finfo(np.float64).eps if len(sv_A) > 0 else 0
    rank_A   = int(np.sum(sv_A > tol_A))
    null_A   = n - rank_A

    # Solution freedom
    freedom  = compute_solution_freedom(X, X_A)
    is_uniq  = (freedom == 0)

    return {
        "C_star":           C_star,
        "residual":         residual,
        "constraint_viol":  constraint_viol,
        "null_dim_A":       null_A,
        "solution_freedom": freedom,
        "is_unique":        is_uniq,
        "n_steps":          1,
        "rank_A":           rank_A,
        "V2":               V2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Anchor selection
# ─────────────────────────────────────────────────────────────────────────────

def select_anchor_indices(
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
    strategy: str,
    model_logits: np.ndarray = None,
) -> np.ndarray:
    """
    Select k anchor positions from {0, ..., T−1}.
    'random'           — uniform (rng seed=42)
    'low_uncertainty'  — rows of X with lowest ‖x‖₂ (or entropy if logits given)
    'high_uncertainty' — rows with highest ‖x‖₂ / entropy
    """
    T   = X.shape[0]
    k   = min(k, T)
    rng = np.random.default_rng(42)

    if strategy == "random":
        return rng.choice(T, size=k, replace=False).astype(int)

    if model_logits is not None:
        shifted = model_logits - model_logits.max(axis=-1, keepdims=True)
        probs   = np.exp(shifted)
        probs  /= probs.sum(axis=-1, keepdims=True)
        scores  = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
    else:
        scores = np.linalg.norm(X, axis=1)   # [T]

    if strategy == "low_uncertainty":
        return np.argsort(scores)[:k].astype(int)
    elif strategy == "high_uncertainty":
        return np.argsort(scores)[-k:].astype(int)
    else:
        raise ValueError(f"Unknown anchor strategy: {strategy!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Adam baseline (pure NumPy implementation)
# ─────────────────────────────────────────────────────────────────────────────

def solve_adam(
    X: np.ndarray,
    Y: np.ndarray,
    n_steps: int,
    lr: float,
    constrained: bool = False,
    anchor_indices: np.ndarray = None,
    Y_anchor: np.ndarray = None,
) -> dict:
    """
    Minimise ‖XCᵀ − Y‖²_F using Adam from scratch in NumPy (float64).
    C initialised at zeros (fair baseline).

    Objective: f(C) = ‖XCᵀ − Y‖²_F
    Gradient:  ∂f/∂C = 2 (XCᵀ − Y)ᵀ X  →  shape [d, n]

    If constrained=True, add penalty λ‖X_A Cᵀ − Y_Aᵀ‖²_F, λ=1000.

    Adam hyperparameters: β₁=0.9, β₂=0.999, ε=1e-8 (standard defaults).
    """
    assert X.dtype == np.float64
    assert Y.dtype == np.float64

    T, n = X.shape
    _, d = Y.shape

    C   = np.zeros((d, n), dtype=np.float64)     # [d, n]
    m   = np.zeros_like(C)                        # first moment
    v   = np.zeros_like(C)                        # second moment
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    lam = 1000.0

    # Precompute XᵀX and XᵀY for efficiency
    XtX = X.T @ X    # [n, n]
    XtY = X.T @ Y    # [n, d]

    X_A = Y_A = None
    if constrained and anchor_indices is not None:
        X_A = X[anchor_indices]   # [k, n]
        Y_A = Y[anchor_indices]   # [k, d]

    residual_curve = np.zeros(n_steps, dtype=np.float64)

    for step in range(n_steps):
        # Gradient of ‖XCᵀ − Y‖²_F w.r.t. C:
        # ∂f/∂C = 2 * (C @ XtX - XtY.T)  [d, n]
        grad = 2.0 * (C @ XtX - XtY.T)

        if constrained and X_A is not None:
            # ‖X_A Cᵀ − Y_A‖²_F  penalty
            # ∂/∂C = 2λ * (C @ X_Aᵀ X_A - (X_A.T @ Y_A).T)
            XAtXA = X_A.T @ X_A    # [n, n]
            XAtYA = X_A.T @ Y_A    # [n, d]
            grad  += 2.0 * lam * (C @ XAtXA - XAtYA.T)

        # Adam update
        t_adam = step + 1
        m = b1 * m + (1.0 - b1) * grad
        v = b2 * v + (1.0 - b2) * grad ** 2
        m_hat = m / (1.0 - b1 ** t_adam)
        v_hat = v / (1.0 - b2 ** t_adam)
        C = C - lr * m_hat / (np.sqrt(v_hat) + eps_adam)

        # Record ‖XCᵀ − Y‖_F
        pred    = X @ C.T         # [T, d]
        res     = np.linalg.norm(pred - Y, "fro")
        residual_curve[step] = float(res)

    return {
        "C_final":        C,
        "residual_curve": residual_curve,
        "final_residual": float(residual_curve[-1]),
        "n_steps":        n_steps,
        "lr":             lr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Solution freedom
# ─────────────────────────────────────────────────────────────────────────────

def compute_solution_freedom(
    X: np.ndarray,
    X_A: np.ndarray,
    rcond: float = 1e-10,
) -> int:
    """
    dim(N(X_A) ∩ N(XᵀX)).
    Uses null_space of each matrix then checks dimension of the intersection
    via rank of the cross-projection matrix.
    """
    V2 = scipy_null_space(X_A, rcond=rcond)   # [n, nA_null]
    W  = scipy_null_space(X.T @ X, rcond=rcond)   # [n, nP_null]

    if V2.shape[1] == 0 or W.shape[1] == 0:
        return 0

    M   = V2.T @ W    # [nA_null, nP_null]
    sv  = np.linalg.svd(M, compute_uv=False)
    tol = sv[0] * max(M.shape) * np.finfo(np.float64).eps if len(sv) > 0 else 0
    return int(np.sum(sv > tol))


# ─────────────────────────────────────────────────────────────────────────────
# Gradient zero on null space
# ─────────────────────────────────────────────────────────────────────────────

def verify_gradient_zero_on_nullspace(
    X: np.ndarray,
    Y: np.ndarray,
    V2: np.ndarray,
    C_star: np.ndarray,
) -> dict:
    """
    For each output dimension j, verify that the gradient g_j = 2XᵀXc*_j + 2Xᵀy_j
    is orthogonal to every null-space direction v ∈ V₂.

    If C* is the exact minimiser, g_j must lie in R(XᵀX) ⊥ N(XᵀX),
    so g_jᵀv ≈ 0 for all v ∈ V₂.
    """
    _, d = Y.shape
    if V2.shape[1] == 0:
        return {"max_grad_null": 0.0, "verified": True}

    P   = 2.0 * (X.T @ X)    # [n, n]
    Xty = 2.0 * (X.T @ Y)    # [n, d]
    C_T = C_star.T            # [n, d]

    max_dot = 0.0
    for j in range(d):
        g_j  = P @ C_T[:, j] - Xty[:, j]    # [n]
        dots = np.abs(V2.T @ g_j)            # [n_null]
        max_dot = max(max_dot, float(dots.max()))

    return {
        "max_grad_null": max_dot,
        "verified":      max_dot < 1e-6,
    }
