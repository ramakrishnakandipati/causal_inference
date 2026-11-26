"""End-to-end training: data -> RDD + validity suite -> tuned CATE learners -> artifacts.

Usage:  python -m src.models.trainer --config configs/config.yaml [--quick] [--learners a,b]
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import synthetic_generator as gen
from src.evaluation import causal_metrics as M
from src.evaluation import visualizer as V
from src.models import rdd_estimators as R
from src.models.cate_learners import LEARNERS, make_learner, tune_learner
from src.utils import device
from src.utils.nuisance import CateData, dr_scores, oof_propensity, oof_regression

warnings.filterwarnings("ignore")
log = logging.getLogger("trainer")


# ----------------------------------------------------------------------------- helpers
def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.DataFrame,)):
        return o.to_dict("records")
    raise TypeError(type(o))


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def gpu_snapshot() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        u, mu, mt, t = [float(x) for x in out.split(",")]
        return {"gpu_util_pct": u, "mem_used_mb": mu, "mem_total_mb": mt, "temp_c": t}
    except Exception:
        return {}


def setup_logging(out: Path) -> Path:
    (out / "logs").mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(out / "logs" / "train.log", mode="w", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)
    return out / "logs" / "gpu_telemetry.jsonl"


def telemetry(path: Path, event: str, **kw) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%H:%M:%S"), "event": event, **gpu_snapshot(), **kw}) + "\n")


def oof_predict(name: str, params: dict, cd: CateData, k: int, seed: int, use_gpu: bool) -> np.ndarray:
    tau = np.zeros(len(cd))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(cd.X, cd.T):
        L = make_learner(name, params, use_gpu, seed).fit(cd.X[tr], cd.T[tr], cd.Y[tr], cd.w[tr], cd.Z[tr])
        tau[te] = L.predict(cd.X[te])
    return tau


# ----------------------------------------------------------------------------- RDD stage
def run_rdd_stage(df: pd.DataFrame, cfg: dict, out: Path) -> dict:
    log.info("RDD suite ...")
    suite = R.run_rdd_suite(df, cfg["rdd"])
    m = out / "metrics"
    suite["main"].to_csv(m / "rdd_late_estimates.csv", index=False)
    suite["sensitivity"].to_csv(m / "rdd_bandwidth_sensitivity.csv", index=False)
    suite["balance"].to_csv(m / "rdd_covariate_balance.csv", index=False)
    suite["placebo"].to_csv(m / "rdd_placebo_cutoffs.csv", index=False)
    suite["donut"].to_csv(m / "rdd_donut_hole.csv", index=False)
    truth = gen.local_truth(df)
    diag = M.diagnostics_summary(suite)
    diag["mccrary"] = suite["mccrary"]
    diag["true_local_late"] = truth
    # contrast: the same generator with simulated threshold gaming
    dm = gen.generate(n=len(df), regime=df.attrs.get("regime", "fuzzy"), manipulation=True,
                      seed=cfg["seed"], cutoff=cfg["rdd"]["cutoff"])
    mc_manip = R.mccrary_density_test(dm["qualifying_nights_t"], cfg["rdd"]["cutoff"])
    diag["mccrary_with_simulated_manipulation"] = mc_manip
    dump_json(diag, m / "rdd_diagnostics.json")
    joblib.dump({"suite": suite, "cutoff": cfg["rdd"]["cutoff"], "regime": df.attrs.get("regime")},
                out / "models" / "rdd_model.joblib")
    est = suite["main"].set_index("outcome")
    for oc, row in est.iterrows():
        log.info("  %-34s est=%9.3f se=%7.3f true=%9.3f", oc, row["estimate"], row["se"], truth[oc])
    log.info("  McCrary p=%.3f | balanced=%s | placebo null share=%.2f", suite["mccrary"]["p_value"],
             diag["covariates_balanced"], 1 - diag["placebo_rejection_share"])
    plots = out / "plots"
    V.plot_rdd_main(df, suite, cfg["rdd"]["cutoff"], plots)
    V.plot_mccrary(df, suite["mccrary"], cfg["rdd"]["cutoff"], plots, dm, mc_manip)
    V.plot_bandwidth_sensitivity(suite, plots)
    V.plot_falsification(suite, plots)
    return suite


# ----------------------------------------------------------------------------- CATE stage
def run_cate_stage(df: pd.DataFrame, cfg: dict, out: Path, tele: Path, outcome: str,
                   learners: List[str], suite: dict) -> pd.DataFrame:
    cc, gcfg = cfg["cate"], cfg["gpu"]
    use_gpu = bool(gcfg["use_gpu"]) and device.detect()["cuda"]
    h = cc.get("bandwidth") or suite["bandwidths"]["used"]
    cd = CateData(df, outcome, cfg["rdd"]["cutoff"], h, cc.get("kernel", "triangular"))
    short = outcome.replace("next_year_", "")
    log.info("CATE stage [%s]: window n=%d, h=%s, gpu=%s", short, len(cd), h, use_gpu)

    # evaluation scaffolding (ground-truth free): honest nuisance residuals + DR scores
    k, seed = int(cc["cv_folds"]), int(cfg["seed"])
    m_hat = oof_regression(cd.X, cd.Y, cd.w, k, seed, use_gpu)
    e_hat = oof_propensity(cd.X, cd.T, cd.w, k, seed, use_gpu)
    y_res, t_res = cd.Y - m_hat, cd.T - e_hat
    scores = dr_scores(cd.X, cd.T, cd.Y, cd.w, k, seed, use_gpu)

    oof = pd.DataFrame({"member_id": cd.df["member_id"], "r": cd.r, "T": cd.T, "Z": cd.Z, "Y": cd.Y, "w": cd.w,
                        "tau_true": cd.tau_true, "business_travel_ratio": cd.df["business_travel_ratio"],
                        "historical_adr": cd.df["historical_adr"]})
    rows, best_params, histories, curves = [], {}, {}, {}
    model_dir = out / "models" / short
    model_dir.mkdir(parents=True, exist_ok=True)
    if cd.tau_true is None:
        raise RuntimeError("synthetic ground truth missing")

    for name in learners:
        t0 = time.time()
        telemetry(tele, "learner_start", learner=name, outcome=short)
        row = {"outcome": short, "learner": name}
        try:
            tuned = tune_learner(name, cd, int(cfg["optuna"]["n_trials"]), cfg["optuna"].get("timeout_sec"),
                                 use_gpu, seed, cc["tune_val_frac"], log.info)
            params = tuned["best_params"]
            best_params[name] = params
            if tuned["trials"]:
                pd.DataFrame(tuned["trials"]).to_csv(out / "metrics" / f"optuna_{short}_{name}.csv", index=False)
            row.update(tuned_r_loss=tuned["best_value"], n_trials=len(tuned["trials"]))
            tau = oof_predict(name, params, cd, k, seed, use_gpu)           # cross-validated predictions
            oof[name] = tau
            row.update(M.cate_accuracy(tau, cd.tau_true, cd.w))
            row.update(M.uplift_metrics(tau, cd.T, cd.Y, cd.w))
            row.update(M.policy_value(tau, scores, cd.w, tau_true=cd.tau_true))
            row["r_loss_oof"] = M.r_loss(tau, y_res, t_res, cd.w)
            curves[name] = M.uplift_curve(tau, cd.T, cd.Y, cd.w)
            final = make_learner(name, params, use_gpu, seed).fit(cd.X, cd.T, cd.Y, cd.w, cd.Z)   # deployable model
            joblib.dump(final, model_dir / f"{name}.joblib")
            if final.history.get("train"):
                histories[name] = final.history
            row["status"] = "ok"
        except Exception as exc:
            log.error("learner %s failed: %s\n%s", name, exc, traceback.format_exc())
            row["status"] = f"failed: {exc}"
        row["seconds"] = round(time.time() - t0, 1)
        telemetry(tele, "learner_end", learner=name, outcome=short, seconds=row["seconds"], status=row["status"])
        log.info("  %-18s %-8s corr=%.3f rmse=%.3f auuc_gain=%.3f (%.0fs)", name, row["status"][:8],
                 row.get("pearson", np.nan), row.get("pehe_rmse", np.nan), row.get("auuc_over_random", np.nan), row["seconds"])
        rows.append(row)

    lb = pd.DataFrame(rows)
    ok = lb[lb["status"] == "ok"].sort_values("pehe_rmse")
    lb = pd.concat([ok, lb[lb["status"] != "ok"]], ignore_index=True)
    lb["rank_pehe"] = np.arange(1, len(lb) + 1)
    lb.to_csv(out / "metrics" / f"cate_leaderboard_{short}.csv", index=False)
    dump_json(lb.to_dict("records"), out / "metrics" / f"cate_leaderboard_{short}.json")
    dump_json(best_params, out / "models" / f"best_hyperparameters_{short}.json")
    dump_json(histories, out / "metrics" / f"loss_curves_{short}.json")
    oof.to_csv(out / "metrics" / f"oof_cate_{short}.csv.gz", index=False)

    plots = out / "plots"
    if len(ok):
        V.plot_leaderboard(ok, plots, short.replace("_", " "))
        best = ok.iloc[0]["learner"]
        V.plot_segments(oof, best, plots, short.replace("_", " "))
        top = ok.sort_values("auuc_over_random", ascending=False)["learner"].head(4)
        V.plot_uplift({n: curves[n] for n in top}, plots)
    if histories:
        V.plot_loss_curves(histories, plots)
    return lb


# ----------------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--quick", action="store_true", help="small, fast run for smoke tests")
    ap.add_argument("--learners", default=None, help="comma separated subset of learners")
    ap.add_argument("--regime", default=None, choices=["sharp", "fuzzy"])
    ap.add_argument("--outputs", default=None)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.regime:
        cfg["data"]["regime"] = args.regime
    if args.quick:
        cfg["data"]["n"] = 12000
        cfg["optuna"]["n_trials"] = 2
        cfg["cate"]["cv_folds"] = 2
        cfg["cate"]["learners"] = [l for l in cfg["cate"]["learners"] if l != "orthogonal_forest"]
    if args.learners:
        cfg["cate"]["learners"] = args.learners.split(",")
    out = Path(args.outputs or cfg["paths"]["outputs"])
    for sub in ("models", "metrics", "logs", "plots", "screenshots"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    tele = setup_logging(out)
    tele.write_text("", encoding="utf-8")
    log.info("hardware: %s", device.describe())
    telemetry(tele, "start")
    t_start = time.time()

    df = gen.generate(cfg["data"]["n"], cfg["data"]["regime"], cfg["data"]["manipulation"], cfg["rdd"]["cutoff"],
                      cfg["seed"], cfg["data"]["r_min"], cfg["data"]["r_max"])
    df.to_csv(out / "models" / "panel.csv.gz", index=False)
    log.info("generated %d guests (%s regime)", len(df), cfg["data"]["regime"])

    suite = run_rdd_stage(df, cfg, out)
    leaderboards = {}
    for oc in cfg["cate"]["outcomes"]:
        leaderboards[oc] = run_cate_stage(df, cfg, out, tele, oc, cfg["cate"]["learners"], suite)

    # metadata consumed by the API and the dashboard
    meta = {
        "cutoff": cfg["rdd"]["cutoff"], "regime": cfg["data"]["regime"], "covariates": gen.COVARIATES,
        "bandwidths": suite["bandwidths"], "cate_bandwidth": cfg["cate"].get("bandwidth") or suite["bandwidths"]["used"],
        "outcomes": cfg["cate"]["outcomes"], "targets": gen.TARGETS,
        "best_learner": {oc.replace("next_year_", ""): (lb[lb["status"] == "ok"].iloc[0]["learner"] if (lb["status"] == "ok").any() else None)
                         for oc, lb in leaderboards.items()},
        "feature_stats": df[gen.COVARIATES].describe().loc[["mean", "std", "min", "max"]].to_dict(),
        "home_regions": gen.HOME_REGIONS, "age_groups": gen.AGE_GROUPS, "n_panel": len(df),
        "hardware": device.detect(), "train_seconds": round(time.time() - t_start, 1),
    }
    dump_json(meta, out / "models" / "meta.json")
    try:                                    # end-to-end flows (8 worked examples with stage-, flow- and programme-level metrics)
        from src import flows
        log.info("building end-to-end flows -> %s", flows.build_and_save(out))
    except Exception as exc:                # never lose a finished training run over the flows
        log.warning("flows not built (%s); run `python -m src.flows` once the models exist", exc)
    telemetry(tele, "end")
    log.info("done in %.1f min; artifacts in %s", (time.time() - t_start) / 60, out)


if __name__ == "__main__":
    main()
    device.hard_exit(0)      # avoids a native abort during DLL unload (see device.hard_exit)
