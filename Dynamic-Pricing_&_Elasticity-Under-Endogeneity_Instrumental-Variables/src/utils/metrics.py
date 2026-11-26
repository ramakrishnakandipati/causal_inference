"""Evaluation metrics: PEHE, elasticity RMSE, Qini/AUUC (oracle), first-stage F, JSON helpers."""
from __future__ import annotations

import json
import math
from typing import Any, Dict, Tuple

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / torch scalars and arrays to plain JSON-safe Python types."""
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().numpy()
    except Exception:  # noqa: BLE001
        pass
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, "item") and callable(obj.item) and getattr(obj, "shape", None) == ():
        return to_jsonable(obj.item())
    return obj


def dump_json(obj: Any, path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(obj), fh, indent=2)


def pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """sqrt(PEHE): root mean squared error of the individual treatment effect estimates."""
    d = np.asarray(tau_hat, float) - np.asarray(tau_true, float)
    return float(np.sqrt(np.mean(d ** 2)))


def elasticity_rmse(theta_hat: np.ndarray, theta_true: np.ndarray) -> float:
    return pehe(theta_hat, theta_true)


def elasticity_bias(theta_hat: np.ndarray, theta_true: np.ndarray) -> float:
    return float(np.mean(np.asarray(theta_hat, float) - np.asarray(theta_true, float)))


def qini_curve(tau_hat: np.ndarray, tau_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Oracle Qini: target units in descending order of predicted effect and accumulate the *true* gain.

    Returns (fraction targeted, cumulative gain per capita * fraction).
    """
    tau_hat = np.asarray(tau_hat, float)
    tau_true = np.asarray(tau_true, float)
    order = np.argsort(-tau_hat, kind="stable")
    cum = np.cumsum(tau_true[order])
    n = len(tau_hat)
    frac = np.arange(1, n + 1) / n
    gain = cum / n
    return np.concatenate([[0.0], frac]), np.concatenate([[0.0], gain])


def auuc(tau_hat: np.ndarray, tau_true: np.ndarray, normalize: bool = True) -> float:
    """Area under the oracle uplift curve, minus the random-targeting baseline (Qini coefficient)."""
    frac, gain = qini_curve(tau_hat, tau_true)
    area = float(np.trapz(gain, frac))
    random_area = 0.5 * float(np.mean(tau_true))
    q = area - random_area
    if not normalize:
        return q
    f2, g2 = qini_curve(tau_true, tau_true)
    best = float(np.trapz(g2, f2)) - random_area
    return float(q / best) if best > 1e-12 else 0.0


def first_stage_f(t: np.ndarray, z: np.ndarray, x: np.ndarray | None = None) -> Dict[str, float]:
    """Partial F-statistic for H0: instruments Z have zero coefficients in T ~ Z + X (+ const)."""
    t = np.asarray(t, float).ravel()
    z = np.asarray(z, float).reshape(len(t), -1)
    n = len(t)
    ones = np.ones((n, 1))
    xs = ones if x is None else np.hstack([ones, np.asarray(x, float).reshape(n, -1)])
    full = np.hstack([xs, z])

    def rss(design: np.ndarray) -> float:
        beta, *_ = np.linalg.lstsq(design, t, rcond=None)
        r = t - design @ beta
        return float(r @ r)

    rss_r, rss_u = rss(xs), rss(full)
    q = z.shape[1]
    dof = n - full.shape[1]
    f = ((rss_r - rss_u) / q) / (rss_u / dof)
    partial_r2 = (rss_r - rss_u) / rss_r
    return {"f_stat": float(f), "partial_r2": float(partial_r2), "n_instruments": int(q), "dof": int(dof)}
