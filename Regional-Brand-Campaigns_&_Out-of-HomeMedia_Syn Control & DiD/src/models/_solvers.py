"""Numerical building blocks shared by the synthetic-control family.

All simplex problems are solved as convex QPs with OSQP (polished), after rescaling so tolerances
are meaningful regardless of the outcome scale. SLSQP is the fallback.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize


def _qp_simplex(P: np.ndarray, q: np.ndarray) -> np.ndarray:
    """min 0.5 w'Pw + q'w  s.t. w >= 0, sum(w) = 1."""
    n = len(q)
    P = 0.5 * (P + P.T) + 1e-10 * np.eye(n)
    try:
        import osqp

        A = sp.vstack([sp.csc_matrix(np.ones((1, n))), sp.identity(n, format="csc")], format="csc")
        lo = np.concatenate([[1.0], np.zeros(n)])
        hi = np.concatenate([[1.0], np.full(n, np.inf)])
        prob = osqp.OSQP()
        prob.setup(P=sp.csc_matrix(np.triu(P)), q=q, A=A, l=lo, u=hi, eps_abs=1e-10, eps_rel=1e-10,
                   max_iter=40000, polishing=True, verbose=False)
        res = prob.solve(raise_error=False)
        w = np.asarray(res.x, dtype=float)
        if np.all(np.isfinite(w)):
            w = np.clip(w, 0.0, None)
            s = w.sum()
            if s > 0:
                return w / s
    except Exception:
        pass
    r = minimize(lambda w: 0.5 * w @ P @ w + q @ w, np.full(n, 1.0 / n), jac=lambda w: P @ w + q,
                 bounds=[(0, 1)] * n, constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                                   "jac": lambda w: np.ones(n)},
                 method="SLSQP", options={"maxiter": 1000, "ftol": 1e-14})
    w = np.clip(r.x, 0, None)
    return w / w.sum()


def simplex_ls(X: np.ndarray, y: np.ndarray, ridge: float = 0.0, loss_w: np.ndarray | None = None) -> np.ndarray:
    """Weights w >= 0, sum w = 1 minimising ||y - X w||^2 (rows of X = time, columns = donors).

    `loss_w` optionally weights rows (e.g. SDID time weights); `ridge` adds ridge*||w||^2.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if loss_w is not None:
        s = np.sqrt(np.asarray(loss_w, float))
        X, y = X * s[:, None], y * s
    scale = max(np.sqrt(np.mean(y**2)), 1e-9)
    Xs, ys = X / scale, y / scale
    rs = ridge / scale**2                       # ridge is expressed in the original (unscaled) units
    P = 2.0 * (Xs.T @ Xs) + 2.0 * rs * np.eye(X.shape[1])
    q = -2.0 * (Xs.T @ ys)
    return _qp_simplex(P, q)


def simplex_ls_cols(X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Convenience alias (clarity at call sites): columns of X are candidate donors."""
    return simplex_ls(X, y, ridge=ridge)


def ridge_solve(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    """Solve (A'A + lam I) x = A'b via the SVD-stable normal equations."""
    n = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(n), A.T @ b)


def rmspe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
