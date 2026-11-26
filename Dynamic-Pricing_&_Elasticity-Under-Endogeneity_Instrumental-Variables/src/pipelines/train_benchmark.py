"""Train and benchmark every estimator; persist checkpoints, metrics and test-set predictions.

Run:  python -m src.pipelines.train_benchmark [--n-rows N] [--epochs E] [--skip-heavy]
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.generate_hospitality_data import FEATURES, TREATMENT, ln_reference_price  # noqa: E402
from src.models.baselines import NaiveOLS, NaiveXGB  # noqa: E402
from src.models.cate_metalearners import CateMeta, causalml_cross_check  # noqa: E402
from src.models.deep_cate_pytorch import CEVAETorch, DragonNetTorch, TARNetTorch  # noqa: E402
from src.models.deep_iv_pytorch import DeepGMMTorch, DeepIVTorch  # noqa: E402
from src.models.iv_econometric import EconmlIV, LinearIV, iv_diagnostics  # noqa: E402
from src.utils.gpu_utils import hardware_summary, load_config, log_vram, resolve, set_seed  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.metrics import auuc, dump_json, elasticity_bias, elasticity_rmse, pehe  # noqa: E402

warnings.filterwarnings("ignore")
log = get_logger("train_benchmark", "train_benchmark.log")

CORE_17 = ["Naive OLS", "Naive XGBoost", "2SLS", "IV-GMM", "DML-IV", "DR-IV (LinearDRIV)", "DeepIV (PyTorch)",
           "DeepGMM (PyTorch)", "S-Learner", "T-Learner", "X-Learner", "R-Learner (NonParamDML)", "DR-Learner",
           "Causal Forest DML", "Uplift RF (causalml)", "TARNet", "DragonNet"]


def build_models(cfg: Dict[str, Any], heavy_n: int, skip_heavy: bool) -> List[Any]:
    models: List[Any] = [
        NaiveOLS(cfg=cfg), NaiveXGB(cfg=cfg),
        LinearIV("2sls", cfg=cfg), LinearIV("liml", cfg=cfg), LinearIV("gmm", cfg=cfg),
        EconmlIV("dmliv", cfg=cfg), EconmlIV("orthoiv", cfg=cfg), EconmlIV("driv", cfg=cfg),
        DeepIVTorch(cfg=cfg), DeepGMMTorch(cfg=cfg),
        CateMeta("s", cfg=cfg), CateMeta("t", cfg=cfg), CateMeta("x", cfg=cfg), CateMeta("r", cfg=cfg),
        CateMeta("dr", cfg=cfg),
        TARNetTorch(cfg=cfg), DragonNetTorch(cfg=cfg), CEVAETorch(cfg=cfg),
    ]
    if not skip_heavy:
        models += [EconmlIV("forestdriv", cfg=cfg, max_rows=heavy_n),
                   CateMeta("causal_forest", cfg=cfg, max_rows=heavy_n),
                   CateMeta("uplift_rf", cfg=cfg, max_rows=heavy_n)]
    return models


def demand_curve_rmse(model, test: pd.DataFrame, cfg: Dict[str, Any], n: int = 1500) -> float:
    """RMSE of predicted log demand vs the structural truth (U=0) at +/-20% around each row's reference price."""
    from src.optimization.revenue_optimizer import truth_log_demand

    sub = test.sample(min(n, len(test)), random_state=0).reset_index(drop=True)
    errs = []
    for mult in (0.8, 1.0, 1.25):
        price = np.exp(ln_reference_price(sub, cfg)) * mult
        pred = model.predict_log_demand(sub, np.log(price))
        truth = truth_log_demand(sub, cfg, price, 0.0)
        errs.append(pred - truth)
    # level is only identified up to the model's treatment of U, so compare shapes: demean per row-price set
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean((e - e.mean()) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows", type=int, default=None, help="subsample the dataset (smoke tests)")
    ap.add_argument("--epochs", type=int, default=None, help="override deep-model epochs (smoke tests)")
    ap.add_argument("--skip-heavy", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    if args.epochs:
        for k in ("deep",):
            cfg["models"][k]["epochs"] = args.epochs
        cfg["models"]["deepgmm"]["epochs"] = args.epochs
        cfg["models"]["cevae"]["epochs"] = args.epochs
    set_seed(cfg["project"]["seed"])
    out_models, out_metrics = resolve(cfg["paths"]["models"]), resolve(cfg["paths"]["metrics"])
    log.info("Hardware: %s", hardware_summary())
    log_vram(log, "start")

    df = pd.read_parquet(resolve(cfg["paths"]["data"]) / "hospitality_bookings.parquet")
    if args.n_rows:
        df = df.sample(args.n_rows, random_state=cfg["project"]["seed"]).reset_index(drop=True)
    train, test = train_test_split(df, test_size=cfg["data"]["test_size"], random_state=cfg["project"]["seed"])
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    log.info("Dataset=%s train=%s test=%s", df.shape, train.shape, test.shape)

    # ---- econometric diagnostics ----------------------------------------------------------------
    diag = iv_diagnostics(train)
    dump_json(diag, out_metrics / "iv_first_stage_diagnostics.json")
    log.info("First-stage F (robust)=%.1f, partial R2=%.3f | Wu-Hausman=%.1f (p=%.2e) | Sargan p=%.3f",
             diag["first_stage"]["f_statistic_robust"], diag["first_stage"]["partial_r2"],
             diag["wu_hausman_endogeneity"]["statistic"], diag["wu_hausman_endogeneity"]["pvalue"],
             diag["sargan_overidentification"]["pvalue"])
    log.info("Price coefficient: OLS=%.3f  2SLS=%.3f  GMM=%.3f  (true mean %.3f)",
             diag["coefficients"]["ols_price_coefficient"], diag["coefficients"]["2sls_price_coefficient"],
             diag["coefficients"]["gmm_price_coefficient"], diag["coefficients"]["true_mean_elasticity"])

    # ---- train all models -------------------------------------------------------------------------
    th_true, tau_true = test["true_elasticity"].to_numpy(), test["true_promo_cate"].to_numpy()
    rows, fitted, preds, histories = [], {}, {"booking_id": test["booking_id"].to_numpy(),
                                                "true_elasticity": th_true, "true_promo_cate": tau_true}, {}
    for m in build_models(cfg, cfg["data"]["heavy_model_n"], args.skip_heavy):
        log.info("=== Training %s [%s] ===", m.name, m.family)
        try:
            t0 = time.time()
            m.fit(train)
            theta_hat, tau_hat = m.elasticity(test), m.promo_cate(test)
            rec = {
                "model": m.name, "family": m.family, "core17": m.name in CORE_17, "is_iv": bool(m.is_iv),
                "native_target": m.native_target,
                "elasticity_rmse": elasticity_rmse(theta_hat, th_true),
                "elasticity_bias": elasticity_bias(theta_hat, th_true),
                "pehe": pehe(tau_hat, tau_true),
                "auuc": auuc(tau_hat, tau_true),
                "mean_elasticity": float(np.mean(theta_hat)),
                "demand_curve_rmse": demand_curve_rmse(m, test, cfg),
                "rank_corr_with_truth": float(pd.Series(theta_hat).corr(pd.Series(th_true), method="spearman")),
                "fit_seconds": float(m.fit_seconds or time.time() - t0),
            }
            rows.append(rec)
            fitted[m.name] = m.to_cpu()
            preds[f"theta__{m.name}"], preds[f"tau__{m.name}"] = theta_hat, tau_hat
            if m.loss_history:
                histories[m.name] = m.loss_history
            log.info("%s -> elasticity RMSE=%.3f bias=%+.3f PEHE=%.4f AUUC=%.3f mean=%.3f (%.1fs)", m.name,
                     rec["elasticity_rmse"], rec["elasticity_bias"], rec["pehe"], rec["auuc"],
                     rec["mean_elasticity"], rec["fit_seconds"])
            log_vram(log, m.name)
        except Exception as exc:  # noqa: BLE001
            log.error("Model %s FAILED: %s\n%s", m.name, exc, traceback.format_exc())
            rows.append({"model": m.name, "family": m.family, "core17": m.name in CORE_17, "error": str(exc)})

    # ---- independent causalml cross-check ----------------------------------------------------------
    try:
        cc = causalml_cross_check(train, test, cfg)
        for k, v in cc.items():
            name = {"S": "S-Learner", "T": "T-Learner", "X": "X-Learner", "R": "R-Learner (NonParamDML)"}[k]
            if f"tau__{name}" in preds:
                log.info("causalml %s-learner PEHE=%.4f (econml %.4f)", k, pehe(v, tau_true), pehe(preds[f"tau__{name}"], tau_true))
    except Exception as exc:  # noqa: BLE001
        log.warning("causalml cross-check skipped: %s", str(exc)[:200])

    # ---- persist ------------------------------------------------------------------------------------
    summary = pd.DataFrame(rows)
    summary.to_csv(out_metrics / "benchmark_summary.csv", index=False)
    dump_json({"n_train": len(train), "n_test": len(test), "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
               "models": summary.to_dict(orient="records")}, out_metrics / "benchmark_summary.json")
    pd.DataFrame(preds).to_parquet(out_metrics / "test_predictions.parquet", index=False)
    dump_json(histories, out_metrics / "training_histories.json")
    joblib.dump(fitted, out_models / "causal_estimators.joblib")

    if "DeepIV (PyTorch)" in fitted:
        fitted["DeepIV (PyTorch)"].save_stages(out_models / "deep_iv_stage1_mdn.pt", out_models / "deep_iv_stage2_outcome.pt")
    deep = {}
    for nm in ("DeepGMM (PyTorch)", "TARNet", "DragonNet", "CEVAE"):
        if nm in fitted:
            deep[nm] = {k: v.state_dict() for k, v in fitted[nm]._modules_to_save().items()} if hasattr(fitted[nm], "_modules_to_save") \
                else fitted[nm].model.state_dict()
    torch.save(deep, out_models / "deep_cate_models.pt")

    cols = [c for c in ["model", "family", "elasticity_rmse", "elasticity_bias", "pehe", "auuc", "mean_elasticity", "fit_seconds"] if c in summary]
    log.info("\n%s", summary[cols].sort_values("elasticity_rmse").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log.info("Saved models -> %s ; metrics -> %s", out_models, out_metrics)


if __name__ == "__main__":
    main()
