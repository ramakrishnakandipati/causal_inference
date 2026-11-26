"""Unified training / evaluation / persistence for the 9 CATE algorithms.

Evaluation protocol: out-of-fold CATE predictions with DMA-grouped 5-fold CV, so no DMA is
scored by a model that saw its rows. Effective sample size is 50 DMAs, not 1,400 DMA-days.
"""
from __future__ import annotations

import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.feature_engineering import cate_frame, cate_matrix
from src.data.synthetic_geo_generator import CATE_FEATURES
from src.models.deep_cate_gpu import DragonNet, TARNet
from src.models.dml_and_forests import CausalForestDMLModel, LinearDMLModel, NonParamDMLModel
from src.models.meta_learners import default_meta_learners, group_folds
from src.utils.config import path

log = logging.getLogger(__name__)
ALGOS = ["S-Learner", "T-Learner", "X-Learner", "R-Learner", "DR-Learner",
         "LinearDML", "CausalForestDML", "TARNet", "DragonNet"]


def build_models(seed: int = 0, include_nonparam: bool = False) -> dict:
    m = default_meta_learners(seed)
    m["LinearDML"] = LinearDMLModel(seed)
    m["CausalForestDML"] = CausalForestDMLModel(seed)
    m["TARNet"] = TARNet(seed)
    m["DragonNet"] = DragonNet(seed)
    if include_nonparam:
        m["NonParamDML"] = NonParamDMLModel(seed)
    return m


# ---------------------------------------------------------------- uplift metrics
def uplift_curves(score, y, t, n_points: int = 101):
    """Qini curve Q(k)=sum_yT(k) - sum_yC(k)*nT(k)/nC(k) ranked by predicted benefit of pausing."""
    order = np.argsort(-score, kind="stable")
    y, t = y[order], t[order]
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    syt, syc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    q = np.where(nc > 0, syt - syc * nt / np.maximum(nc, 1), 0.0)
    frac = np.arange(1, len(y) + 1) / len(y)
    grid = np.linspace(0, 1, n_points)
    return grid, np.interp(grid, np.r_[0, frac], np.r_[0, q])


def auuc_qini(score, y, t, true_tau=None):
    grid, q = uplift_curves(score, y, t)
    auuc = float(np.trapezoid(q, grid))
    qini = auuc - 0.5 * q[-1]                       # area above the random-targeting diagonal
    out = {"auuc": auuc, "qini": float(qini)}
    if true_tau is not None:
        _, qo = uplift_curves(true_tau, y, t)
        out["qini_oracle"] = float(np.trapezoid(qo, grid) - 0.5 * qo[-1])
    return out, grid, q


def evaluate(tau_hat, tau_true, y, t) -> dict:
    res, _, _ = auuc_qini(tau_hat, y, t, tau_true)
    sd = np.std(tau_true) + 1e-9
    res.update({
        "pehe": float(np.sqrt(np.mean((tau_hat - tau_true) ** 2))),
        "pehe_norm": float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)) / sd),
        "ate_hat": float(np.mean(tau_hat)), "ate_true": float(np.mean(tau_true)),
        "ate_abs_err": float(abs(np.mean(tau_hat) - np.mean(tau_true))),
        "rank_corr": float(spearmanr(tau_hat, tau_true).correlation),
        "qini_ratio": float(res["qini"] / res["qini_oracle"]) if res["qini_oracle"] else float("nan"),
    })
    return res


def permutation_importance(model, X, names, n_rep: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = model.predict(X)
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        for _ in range(n_rep):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            imp[j] += np.mean(np.abs(model.predict(Xp) - base))
    imp /= n_rep
    return imp / (imp.sum() + 1e-12)


# ---------------------------------------------------------------- cpu-ification for serving
def to_cpu(obj, _seen=None):
    """Best effort: move GBDT/torch components to CPU so artefacts load on machines without CUDA."""
    _seen = _seen if _seen is not None else set()
    if id(obj) in _seen:
        return obj
    _seen.add(id(obj))
    if hasattr(obj, "set_params") and hasattr(obj, "get_params"):
        try:
            if "device" in obj.get_params():
                obj.set_params(device="cpu")
        except Exception:
            pass
    if hasattr(obj, "parameters") and hasattr(obj, "cpu"):
        try:
            obj.cpu()
        except Exception:
            pass
    if hasattr(obj, "device") and isinstance(getattr(obj, "device"), str):
        try:
            obj.device = "cpu"
        except Exception:
            pass
    if isinstance(obj, (list, tuple)):
        for o in obj:
            to_cpu(o, _seen)
    elif isinstance(obj, dict):
        for o in obj.values():
            to_cpu(o, _seen)
    elif hasattr(obj, "__dict__"):
        for o in list(vars(obj).values()):
            if not isinstance(o, (int, float, str, np.ndarray, type(None))):
                to_cpu(o, _seen)
    return obj


# ---------------------------------------------------------------- main entry
def train_and_evaluate(panel: pd.DataFrame, seed: int = 0, algos=None, n_outer: int = 5, verbose: bool = True,
                       save: bool = True, target: str = "direct"):
    d = cate_frame(panel, target)
    X, names = cate_matrix(d)
    T, Y, g = d["T"].values.astype(int), d["Y"].values.astype(float), d["dma_id"].values
    tau_true = d["tau_true"].values
    folds = group_folds(g, n_outer, seed + 100)
    models = build_models(seed)
    algos = algos or ALGOS
    oof, rows, importance, curves, timings = {}, [], {}, {}, {}
    for name in algos:
        t0 = time.time()
        pred = np.zeros(len(Y))
        for tr, te in folds:
            m = build_models(seed)[name]
            m.fit(X[tr], T[tr], Y[tr], g[tr])
            pred[te] = m.predict(X[te])
        oof[name] = pred
        met = evaluate(pred, tau_true, Y, T)
        grid, q = uplift_curves(pred, Y, T)
        curves[name] = q
        full = models[name].fit(X, T, Y, g)                         # final model on all DMAs
        importance[name] = permutation_importance(full, X, names, seed=seed)
        met.update({"model": name, "seconds": round(time.time() - t0, 1)})
        rows.append(met)
        if verbose:
            log.info("%-16s PEHE %.0f (norm %.2f) AUUC %.0f Qini %.0f rank %.2f  %.1fs", name, met["pehe"],
                     met["pehe_norm"], met["auuc"], met["qini"], met["rank_corr"], met["seconds"])
        models[name] = full
    # reference row: ignore heterogeneity and predict the (DiD) average effect for every row
    const = np.full(len(Y), Y[T == 1].mean() - Y[T == 0].mean())
    met = evaluate(const, tau_true, Y, T)
    for k in ("auuc", "qini", "qini_ratio", "rank_corr"):   # a constant has no ranking: ties, not skill
        met[k] = float("nan")
    met.update({"model": "Constant ATE (DiD)", "seconds": 0.0})
    rows.append(met)
    lb = pd.DataFrame(rows).set_index("model").sort_values("pehe")
    d_out = d[["dma_id", "dma", "day", "T", "tier", "tau_true"]].copy()
    for k, v in oof.items():
        d_out[f"tau_hat_{k}"] = v
    qdf = pd.DataFrame(curves, index=grid)
    qdf.index.name = "fraction"
    _, _, q_or = auuc_qini(tau_true, Y, T)
    qdf["Oracle (true tau)"] = q_or
    rng = np.random.default_rng(seed)
    qdf["Random"] = np.linspace(0, 1, len(grid)) * qdf.iloc[-1, 0]
    imp = pd.DataFrame(importance, index=names)
    if save:
        out = path("metrics")
        lb.to_csv(out / f"cate_leaderboard_{target}.csv")
        (out / f"cate_leaderboard_{target}.json").write_text(json.dumps(lb.reset_index().to_dict("records"), indent=2))
        d_out.to_csv(out / f"cate_oof_predictions_{target}.csv", index=False)
        qdf.to_csv(out / f"qini_curves_{target}.csv")
        imp.to_csv(out / f"cate_feature_importance_{target}.csv")
        mdir = path("models")
        for k, m in models.items():
            joblib.dump(to_cpu(m), mdir / f"cate_{target}_{k}.joblib")
        joblib.dump({"features": names, "algos": list(algos)}, mdir / f"cate_meta_{target}.joblib")
    return {"leaderboard": lb, "oof": d_out, "qini": qdf, "importance": imp, "models": models, "X": X,
            "names": names, "frame": d}


def load_models(target: str = "direct", algos=None) -> dict:
    mdir = path("models")
    algos = algos or ALGOS
    return {a: joblib.load(mdir / f"cate_{target}_{a}.joblib") for a in algos
            if (mdir / f"cate_{target}_{a}.joblib").exists()}
