"""Policy simulation, persona quadrant assignment, CRM decision rule and SHAP CATE explanations."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.common import FEATURES, PERSONAS, get_logger, load_config, resolve, save_json
from src.metrics import policy_value

log = get_logger()


# ------------------------------------------------------------------------------------------
# Economics / decision rule (shared by evaluate, API and Streamlit)
# ------------------------------------------------------------------------------------------
def stay_value_proxy(X: pd.DataFrame) -> np.ndarray:
    """Expected full-rate stay revenue R ~ ADR x expected nights (same proxy the UpliftRF plug-in uses)."""
    return X["avg_adr"].values * (1 + (X["weekend_stay_ratio"].values > 0.5) + 0.8)


def points_tau_margin(tau_conv, p0, R, offer: dict, discount_tau=None):
    """Modelled margin effect of the 2,000-point offer: conversion lift = ratio * discount lift, and the points
    liability is paid on every booking made by a mailed guest (incl. would-have-booked-anyway guests):
        tau_pts = ratio*tau_conv*(R*m - P) - p0*P
    NB: this offer arm is *modelled* from the estimated conversion uplift, it is not observed in the DGP."""
    P = offer["points_offer"] * offer["points_cost_per_point"]
    return offer["points_conv_effect_ratio"] * tau_conv * (R * offer["margin_rate"] - P) - p0 * P


def assign_quadrant(tau_conv, tau_margin, p0, cfg=None):
    cfg = cfg or load_config()
    qc = cfg["evaluation"]["quadrant_conv_threshold"]
    qm = cfg["evaluation"]["quadrant_margin_threshold"]
    tau_conv, tau_margin, p0 = map(np.asarray, (tau_conv, tau_margin, p0))
    sd = cfg["evaluation"].get("sleeping_dog_conv_threshold", 0.06)
    return np.select([tau_conv < -sd, (tau_conv >= qc) & (tau_margin > qm), p0 >= 0.30],
                     ["Sleeping Dog", "Persuadable", "Always-Taker"], default="Lost Cause")


def decide(tau_margin, tau_conv, p0, R, quadrant, cfg=None, tau_points=None):
    """CRM decision: SEND 20% OFF / SEND 2,000 POINTS / SUPPRESS."""
    cfg = cfg or load_config()
    thr = cfg["offer"]["cate_threshold"]
    tp = points_tau_margin(tau_conv, p0, R, cfg["offer"]) if tau_points is None else tau_points
    tau_margin, tp, quadrant = np.asarray(tau_margin), np.asarray(tp), np.asarray(quadrant)
    qc = cfg["evaluation"]["quadrant_conv_threshold"]
    tau_conv = np.asarray(tau_conv)
    tp = np.where(tau_conv >= qc, tp, -np.inf)       # points offer needs a real conversion lift
    best = np.maximum(tau_margin, tp)
    best = np.where(tau_conv >= qc, best, -np.inf)    # never mail a guest without a real conversion lift
    out = np.where(best > thr, np.where(tau_margin >= tp, "SEND 20% OFF", "SEND 2,000 POINTS"), "SUPPRESS")
    return np.where(quadrant == "Sleeping Dog", "SUPPRESS", out)


# ------------------------------------------------------------------------------------------
def select_best(bench: pd.DataFrame) -> str:
    b = bench.dropna(subset=["pehe", "qini_norm_ipw"]).copy()
    b["score"] = b["pehe"].rank() + b["qini_norm_ipw"].rank(ascending=False)
    return b.sort_values("score").iloc[0]["model"]


def fit_baseline_conv(cfg, tr):
    """p0(x) = P(Y_conv=1 | X, T=0) baseline-conversion model (used for Always-Taker vs Lost-Cause + points offer)."""
    from src.models import make_classifier
    c = tr[tr["T"] == 0]
    m = make_classifier(cfg, small=False).fit(c[FEATURES].values, c["Y_conv"].values)
    return m


def simulate_policies(te, tau_hat, tau_conv_hat, p0_hat, resp_prob, e_hat, cfg, best_name):
    """Four CRM policies evaluated with ground-truth counterfactuals (primary) + an IPW observed estimate."""
    off = cfg["offer"]
    cc = off["comm_cost"]
    n = len(te)
    tau_true, tau_conv_true = te["tau_margin_true"].values, te["tau_conv_true"].values
    R, d = te["stay_value"].values, off["discount_pct"]
    causal = tau_hat > (off["cate_threshold"] + cc)
    k = int(causal.sum())
    prop_mask = np.zeros(n, bool); prop_mask[np.argsort(-resp_prob)[:k]] = True  # same budget as causal policy
    policies = {"Mass Blast": np.ones(n, bool), "No Promo": np.zeros(n, bool),
                "Propensity Targeting": prop_mask, f"Causal Uplift ({best_name})": causal}
    y, t = te["Y_margin"].values, te["T"].values
    rows = []
    for name, mask in policies.items():
        v = policy_value(tau_true, mask, cc)
        cannibal = float((te["Y0_conv"].values[mask] * R[mask] * d).sum())       # discount handed to would-have-booked guests
        # IPW estimate of the policy's total outcome (observed data only): treat mask -> Y(1), else Y(0)
        w1, w0 = mask * (t == 1) / e_hat, (~mask) * (t == 0) / (1 - e_hat)
        ipw_total = float((w1 * y + w0 * y).sum())
        ipw_base = float(((t == 0) / (1 - e_hat) * y).sum())
        rows.append({"policy": name, "n_targeted": v["n_targeted"], "pct_targeted": v["n_targeted"] / n,
                     "incremental_margin": v["incremental_margin"], "gross_incremental_margin": v["gross"],
                     "send_cost": v["send_cost"], "incremental_bookings": float(tau_conv_true[mask].sum()),
                     "discount_cannibalization": cannibal,
                     "margin_per_guest_targeted": v["incremental_margin"] / max(v["n_targeted"], 1),
                     "ipw_incremental_margin_est": ipw_total - ipw_base - cc * v["n_targeted"]})
    return pd.DataFrame(rows)


def shap_explain(cfg, tr_X, te, preds, best_name, n_sample=2000):
    """Fit XGBoost surrogates tau_hat(x) ~ X for every estimator -> TreeSHAP explanations of CATE drivers."""
    import shap
    from xgboost import XGBRegressor
    sdir = resolve(cfg["paths"]["models"]) / "surrogates"; sdir.mkdir(parents=True, exist_ok=True)
    out = {}
    Xte = te[FEATURES]
    for name in [c for c in preds.columns if c not in ("tau_margin_true", "tau_conv_true", "conv_companion", "UpliftRF__conv")
                 and "__" not in c]:
        sur = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.1, device="cpu", tree_method="hist",
                           random_state=cfg["seed"], verbosity=0).fit(Xte, preds[name].values)
        joblib.dump(sur, sdir / f"{name}.joblib")
        out[name] = float(np.corrcoef(sur.predict(Xte), preds[name])[0, 1] ** 2)
    sur = joblib.load(sdir / f"{best_name}.joblib")
    samp = Xte.sample(min(n_sample, len(Xte)), random_state=cfg["seed"])
    sv = shap.TreeExplainer(sur)(samp)
    pd.DataFrame(sv.values, columns=FEATURES, index=samp.index).to_parquet(resolve(cfg["paths"]["logs"]) / "shap_values.parquet")
    samp.to_parquet(resolve(cfg["paths"]["logs"]) / "shap_sample.parquet")
    imp = pd.Series(np.abs(sv.values).mean(0), index=FEATURES).sort_values(ascending=False)
    save_json({"model": best_name, "mean_abs_shap": imp.to_dict(), "surrogate_r2": out}, f"{cfg['paths']['logs']}/shap_summary.json")
    return imp, out


def save_propensity(cfg, tr):
    """Persist e(x) = P(T=1|X) (same learner as the IPW propensity in train.py) for the serving / flows layer."""
    from src.models import make_classifier
    m = make_classifier(cfg, small=True).fit(tr[FEATURES].values, tr["T"].values)
    joblib.dump(m, resolve(cfg["paths"]["models"]) / "propensity.joblib")
    return m


def run(cfg=None, best: str | None = None):
    cfg = cfg or load_config()
    logs = resolve(cfg["paths"]["logs"])
    bench = pd.read_csv(logs / "benchmark_margin.csv")
    preds = pd.read_parquet(logs / "test_predictions.parquet")
    te = pd.read_parquet(logs / "test_set.parquet")
    prop = pd.read_parquet(logs / "test_propensity.parquet")
    tr = pd.read_parquet(logs / "train_set.parquet")
    best = best or select_best(bench)
    log.info("Best estimator (rank of PEHE + Qini): %s", best)

    save_propensity(cfg, tr)
    base = fit_baseline_conv(cfg, tr)
    joblib.dump(base, resolve(cfg["paths"]["models"]) / "baseline_conv.joblib")
    p0 = base.predict_proba(te[FEATURES].values)[:, 1]
    resp = _response_model(cfg, tr).predict_proba(te[FEATURES].values)[:, 1]
    tau, tauc = preds[best].values, preds["conv_companion"].values
    R = stay_value_proxy(te)
    quad = assign_quadrant(tauc, tau, p0, cfg)
    dec = decide(tau, tauc, p0, R, quad, cfg)
    te_out = te[["persona"]].copy()
    te_out["quadrant"], te_out["decision"], te_out["tau_hat"], te_out["tau_conv_hat"] = quad, dec, tau, tauc
    te_out["resp_prob"], te_out["p0_hat"] = resp, p0
    te_out.to_parquet(logs / "test_decisions.parquet", index=False)
    conf = pd.crosstab(te_out["persona"], te_out["quadrant"], normalize="index").round(3)
    conf.to_csv(logs / "quadrant_confusion.csv")
    log.info("Quadrant assignment vs true persona (row-normalised):\n%s", conf)

    pol = simulate_policies(te, tau, tauc, p0, resp, prop["e_hat"].values, cfg, best)
    pol.to_csv(logs / "policy_simulation.csv", index=False)
    log.info("Policy simulation (test set):\n%s", pol[["policy", "pct_targeted", "incremental_margin", "discount_cannibalization"]].round(2))
    imp, r2 = shap_explain(cfg, tr[FEATURES], te, preds, best)
    log.info("Top SHAP drivers: %s", imp.head(5).round(3).to_dict())
    save_json({"best_model": best}, logs / "summary.json")
    return {"best": best, "policies": pol, "shap": imp}


def _response_model(cfg, tr):
    """P(Y_conv = 1 | X) ignoring treatment: the classic 'propensity-to-buy' targeting model."""
    from src.models import make_classifier
    return make_classifier(cfg).fit(tr[FEATURES].values, tr["Y_conv"].values)


if __name__ == "__main__":
    run()
