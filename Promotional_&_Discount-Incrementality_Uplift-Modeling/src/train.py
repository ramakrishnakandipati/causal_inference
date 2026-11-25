"""End-to-end training: propensity -> Optuna (R-loss) -> fit 11 estimators on GPU -> benchmark -> serialize."""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import pandas as pd

from src.common import FEATURES, get_logger, load_config, resolve, save_json
from src.features import (fit_propensity, ipw_weights, load_data, overlap_summary, split_data,
                          standardized_mean_diff)
from src.metrics import auuc, normalized_qini, pehe, qini_curve, uplift_at_k
from src.models import (ESTIMATORS, DRLearnerModel, device_info, make_classifier, make_regressor)

log = get_logger()


# ------------------------------------------------------------------------------------------
def tune_base_learner(cfg, tr, va, e_va):
    """Optuna search over the shared GBM base learner; objective = validation R-loss of a T-learner."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    A, T, Y = tr[FEATURES].values, tr["T"].values, tr["Y_margin"].values
    Av, Tv, Yv = va[FEATURES].values, va["T"].values, va["Y_margin"].values
    m_hat = make_regressor(cfg).fit(A, Y).predict(Av)          # E[Y|X] nuisance (train-fit, val-eval)
    res_y, res_t = Yv - m_hat, Tv - e_va

    def objective(trial):
        p = {"max_depth": trial.suggest_int("max_depth", 2, 6),
             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
             "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
             "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
             "subsample": trial.suggest_float("subsample", 0.6, 1.0),
             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)}
        m0 = make_regressor(cfg, p).fit(A[T == 0], Y[T == 0])
        m1 = make_regressor(cfg, p).fit(A[T == 1], Y[T == 1])
        tau = m1.predict(Av) - m0.predict(Av)
        return float(np.mean((res_y - res_t * tau) ** 2))

    o = cfg["training"]["optuna"]
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    t0 = time.time()
    study.optimize(objective, n_trials=o["n_trials"], timeout=o["timeout_s"])
    log.info("Optuna: %d trials in %.0fs, best R-loss=%.3f params=%s", len(study.trials), time.time() - t0,
             study.best_value, study.best_params)
    return study.best_params, {"best_value": study.best_value, "n_trials": len(study.trials),
                               "history": [t.value for t in study.trials]}


def _latency(est, X):
    X1k = X.iloc[:1000]
    ts = []
    for _ in range(3):
        t0 = time.perf_counter(); est.predict(X1k); ts.append(time.perf_counter() - t0)
    one = []
    for i in range(5):
        t0 = time.perf_counter(); est.predict(X.iloc[i:i + 1]); one.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts)), 1000 * float(np.median(one))


def evaluate_scores(tau, te, w_ipw, cfg):
    """Ranking metrics on the IPW-weighted test set and on the unbiased randomized holdout; PEHE vs truth."""
    ks = cfg["evaluation"]["uplift_k"]
    y, t = te["Y_margin"].values, te["T"].values
    oracle = te["tau_margin_true"].values
    rct = te["rct"].values == 1
    out = {"pehe": pehe(tau, oracle),
           "corr_true": float(np.corrcoef(tau, oracle)[0, 1]) if np.std(tau) > 0 else 0.0,
           "auuc_ipw": auuc(tau, y, t, w_ipw),
           "qini_norm_ipw": normalized_qini(tau, y, t, oracle, w_ipw),
           "auuc_rct": auuc(tau[rct], y[rct], t[rct]),
           "qini_norm_rct": normalized_qini(tau[rct], y[rct], t[rct], oracle[rct])}
    for k in ks:
        out[f"uplift@{int(k * 100)}_ipw"] = uplift_at_k(tau, y, t, k, w_ipw)
        out[f"true_tau@{int(k * 100)}"] = float(oracle[np.argsort(-tau)[:int(k * len(tau))]].mean())
    return out


def run(cfg=None, quick: bool = False, ablation: bool = False, only: list[str] | None = None):
    cfg = cfg or load_config()
    if quick:
        cfg["training"].update(ci_bootstrap={"default": 2, "dr": 1, "nn": 2})
        cfg["training"]["optuna"].update(n_trials=4, timeout_s=60)
        cfg["training"]["nn"]["epochs"] = 15
        cfg["training"]["forest"]["n_estimators"] = 80
        cfg["training"]["uplift_rf"]["n_estimators"] = 30
    dev = device_info(cfg)
    log.info("Device: %s", dev)
    df = load_data(cfg)
    if quick:
        df = df.sample(15000, random_state=cfg["seed"]).reset_index(drop=True)
    sp = split_data(df, cfg)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    log.info("Split sizes train/val/test = %d/%d/%d | test RCT rows=%d", len(tr), len(va), len(te), te["rct"].sum())

    # ---- propensity & balance diagnostics ---------------------------------------------------
    e_tr, (e_va, e_te), prop_model = fit_propensity(tr[FEATURES].values, tr["T"].values,
                                                    [va[FEATURES].values, te[FEATURES].values], cfg)
    w_te = ipw_weights(te["T"].values, e_te)
    w_tr = ipw_weights(tr["T"].values, e_tr)
    smd_raw = standardized_mean_diff(tr[FEATURES], tr["T"].values)
    smd_ipw = standardized_mean_diff(tr[FEATURES], tr["T"].values, w_tr)
    diag = {"overlap_train": overlap_summary(e_tr, tr["T"].values), "smd_raw": smd_raw.to_dict(), "smd_ipw": smd_ipw.to_dict(),
            "max_smd_raw": float(smd_raw.max()), "max_smd_ipw": float(smd_ipw.max())}
    log.info("Propensity overlap %s | max SMD raw=%.3f -> IPW=%.3f", diag["overlap_train"], smd_raw.max(), smd_ipw.max())
    save_json(diag, f"{cfg['paths']['logs']}/propensity_diagnostics.json")
    pd.DataFrame({"e_hat": e_tr, "T": tr["T"].values, "w_ipw": w_tr}).to_parquet(resolve(cfg["paths"]["logs"]) / "propensity_train.parquet")

    # ---- tuning ---------------------------------------------------------------------------------
    params, tune_info = {}, {}
    if cfg["training"]["optuna"]["enabled"] and cfg["training"]["base_learner"] == "xgboost":
        params, tune_info = tune_base_learner(cfg, tr, va, e_va)
    save_json({"best_params": params, **tune_info}, f"{cfg['paths']['logs']}/optuna.json")

    # ---- fit estimators --------------------------------------------------------------------------
    Xtr, Ttr, Ytr = tr[FEATURES], tr["T"].values, tr["Y_margin"].values
    aux = {"Y_conv": tr["Y_conv"].values}
    models_dir = resolve(cfg["paths"]["models"]); models_dir.mkdir(parents=True, exist_ok=True)
    rows, preds, histories = [], {"tau_margin_true": te["tau_margin_true"].values, "tau_conv_true": te["tau_conv_true"].values}, {}
    names = [n for n in ESTIMATORS if not only or n in only]
    for name in names:
        cls = ESTIMATORS[name]
        B = 0 if cls.native_ci else cfg["training"]["ci_bootstrap"][cls.ci_B_key]
        t0 = time.perf_counter()
        try:
            est = cls(cfg, params)
            est.fit_with_ci(Xtr, Ttr, Ytr, aux, B=B)
            fit_s = time.perf_counter() - t0
            tau = est.predict(te[FEATURES])
            ci = est.predict_interval(te[FEATURES])
            lat_1k, lat_1 = _latency(est, te[FEATURES])
            row = {"model": name, "fit_seconds": fit_s, "latency_ms_per_1k": lat_1k, "latency_ms_single": lat_1,
                   "gpu": bool(dev["cuda"] and cls.uses_gpu), "has_ci": ci is not None, **evaluate_scores(tau, te, w_te, cfg)}
            if ci is not None:
                row["ci_coverage_true"] = float(((te["tau_margin_true"].values >= ci[0]) & (te["tau_margin_true"].values <= ci[1])).mean())
                row["ci_mean_width"] = float((ci[1] - ci[0]).mean())
                preds[f"{name}__lo"], preds[f"{name}__hi"] = ci
            preds[name] = tau
            if hasattr(est, "history"):
                histories[name] = {**est.history, "best_epoch": est.best_epoch}
            if name == "UpliftRF":
                preds["UpliftRF__conv"] = est.predict_conv(te[FEATURES])
                row["pehe_conv_native"] = pehe(preds["UpliftRF__conv"], te["tau_conv_true"].values)
            joblib.dump(est.to_cpu(), models_dir / f"{name}.joblib")
            rows.append(row)
            log.info("%-16s fit=%6.1fs PEHE=%6.2f QiniNorm(ipw)=%5.3f AUUC=%7.3f lat/1k=%6.1fms", name, fit_s,
                     row["pehe"], row["qini_norm_ipw"], row["auuc_ipw"], lat_1k)
        except Exception as ex:  # keep going: one failing estimator must not abort the benchmark
            log.error("%s FAILED: %r", name, ex)
            rows.append({"model": name, "error": repr(ex)})

    # ---- conversion companion (tau_conv for quadrant assignment / points offer) ------------------------
    try:
        comp = DRLearnerModel(cfg, params)
        comp.fit_with_ci(Xtr, Ttr, tr["Y_conv"].values, None, B=cfg["training"]["ci_bootstrap"]["dr"])
        tc = comp.predict(te[FEATURES]); preds["conv_companion"] = tc
        joblib.dump(comp.to_cpu(), models_dir / "conv_companion.joblib")
        log.info("Conv companion (DR-Learner on Y_conv): PEHE=%.4f", pehe(tc, te["tau_conv_true"].values))
    except Exception as ex:
        log.error("conv companion failed: %r", ex)

    bench = pd.DataFrame(rows)
    logs = resolve(cfg["paths"]["logs"])
    bench.to_csv(logs / "benchmark_margin.csv", index=False)
    pd.DataFrame(preds).to_parquet(logs / "test_predictions.parquet", index=False)
    te.to_parquet(logs / "test_set.parquet", index=False)
    tr.to_parquet(logs / "train_set.parquet", index=False)
    va.to_parquet(logs / "val_set.parquet", index=False)
    pd.DataFrame({"e_hat": e_te, "w_ipw": w_te}).to_parquet(logs / "test_propensity.parquet", index=False)
    save_json(histories, logs / "deep_training_history.json")
    save_json({"device": dev, "features": FEATURES, "best_params": params, "n_train": len(tr), "models": names,
               "quick": quick}, models_dir / "meta.json")

    if ablation:
        run_ablation(cfg, params, tr, te, w_te, aux)
    return bench


def run_ablation(cfg, params, tr, te, w_te, aux):
    """S/T/X-learners with each base learner family (XGBoost-GPU, LightGBM-CPU, CatBoost-GPU)."""
    rows = []
    for kind in ["xgboost", "lightgbm", "catboost"]:
        for name in ["S-Learner", "T-Learner", "X-Learner"]:
            t0 = time.perf_counter()
            try:
                est = ESTIMATORS[name](cfg, params if kind == "xgboost" else {}, kind=kind)
                est.fit(tr[FEATURES], tr["T"].values, tr["Y_margin"].values, aux)
                rows.append({"base": kind, "model": name, "fit_seconds": time.perf_counter() - t0,
                             **evaluate_scores(est.predict(te[FEATURES]), te, w_te, cfg)})
                log.info("ablation %-9s %-10s PEHE=%.2f", kind, name, rows[-1]["pehe"])
            except Exception as ex:
                rows.append({"base": kind, "model": name, "error": repr(ex)})
    pd.DataFrame(rows).to_csv(resolve(cfg["paths"]["logs"]) / "benchmark_base_learner_ablation.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    run(quick=a.quick, ablation=a.ablation, only=a.only)
