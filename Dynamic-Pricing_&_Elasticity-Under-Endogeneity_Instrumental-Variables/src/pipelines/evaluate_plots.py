"""Five evaluation charts (300 DPI) into outputs/visualizations/evaluation/.  Run: python -m src.pipelines.evaluate_plots"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.generate_hospitality_data import ln_reference_price  # noqa: E402
from src.optimization.revenue_optimizer import (build_profile_df, revpar_curve, truth_curve,  # noqa: E402
                                                truth_occupancy, truth_log_demand)
from src.utils import plotstyle as ps  # noqa: E402
from src.utils.gpu_utils import load_config, resolve  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.metrics import auuc, qini_curve  # noqa: E402
from src.utils.registry import load_bookings, load_models  # noqa: E402
from src.utils.structural import expected_occupancy  # noqa: E402

log = get_logger("evaluate_plots", "evaluate_plots.log")


def training_curves(hist: dict, out: Path) -> None:
    panels = [
        ("DeepIV stage 1: MDN NLL", "DeepIV (PyTorch)", ("stage1_train_nll", "stage1_val_nll"), "NLL"),
        ("DeepIV stage 2: structural MSE", "DeepIV (PyTorch)", ("stage2_train_mse", "stage2_val_mse"), "MSE"),
        ("DeepGMM: moment objective", "DeepGMM (PyTorch)", ("train_moment_objective", "val_moment_objective"), "objective"),
        ("TARNet", "TARNet", ("train_loss", "val_loss"), "MSE"),
        ("DragonNet", "DragonNet", ("train_loss", "val_loss"), "loss"),
        ("CEVAE", "CEVAE", ("train_loss", "val_loss"), "negative ELBO (approx.)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (title, model, (tr, va), yl) in zip(axes.ravel(), panels):
        h = hist.get(model, {})
        if tr in h:
            ax.plot(range(1, len(h[tr]) + 1), h[tr], color=ps.CAT[0], label="train")
            ax.plot(range(1, len(h[va]) + 1), h[va], color=ps.CAT[1], label="validation")
            ax.legend()
        ax.set(title=title, xlabel="epoch", ylabel=yl)
    fig.suptitle("Neural estimators: training and validation curves", fontsize=14, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out / "01_neural_training_validation_curves.png")
    plt.close(fig)


def demand_curves(models: dict, test: pd.DataFrame, cfg: dict, out: Path) -> None:
    sub = test.sample(min(400, len(test)), random_state=1).reset_index(drop=True)
    mults = np.linspace(0.55, 1.7, 24)
    ln_ref = ln_reference_price(sub, cfg)
    eps = cfg["dgp"]["eps_sd"]
    truth = [float(truth_occupancy(sub, cfg, np.exp(ln_ref) * m, None).mean()) for m in mults]
    show = ["Naive OLS", "Naive XGBoost", "2SLS", "DML-IV", "DeepIV (PyTorch)"]
    color = {"Naive OLS": ps.CAT[4], "Naive XGBoost": "#E58AA3", "2SLS": ps.CAT[0], "DML-IV": ps.CAT[2], "DeepIV (PyTorch)": ps.CAT[3]}
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(mults, truth, color=ps.INK, linewidth=3, label="Ground-truth structural demand")
    for name in show:
        if name not in models:
            continue
        occ = [float(expected_occupancy(models[name].predict_log_demand(sub, ln_ref + np.log(m)), eps).mean()) for m in mults]
        ax.plot(mults, occ, color=color[name], linestyle="--" if name.startswith("Naive") else "-", label=name)
    ax.set(xlabel="ADR as a multiple of the reference rate", ylabel="expected occupancy (booking probability)",
           title="Structural demand curves: naive models slope the wrong way")
    ax.legend(loc="lower left")
    fig.savefig(out / "02_structural_demand_curves_ols_vs_iv.png")
    plt.close(fig)


def benchmark_bars(bench: pd.DataFrame, out: Path) -> None:
    b = bench.dropna(subset=["elasticity_rmse"]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.5))
    for ax, col, title in zip(axes, ["pehe", "elasticity_rmse"], ["PEHE (promo effect on log demand), lower is better", "Elasticity RMSE vs truth, lower is better"]):
        d = b.sort_values(col, ascending=False)
        ax.barh(d["model"], d[col], color=[ps.FAMILY_COLOR[f] for f in d["family"]], height=0.65)
        for y, v in enumerate(d[col]):
            ax.text(v, y, f" {v:.3f}", va="center", fontsize=8, color=ps.MUTED)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ps.FAMILY_COLOR.values()]
    fig.legend(handles, [ps.FAMILY_LABEL[k] for k in ps.FAMILY_COLOR], loc="lower center", ncol=5)
    fig.suptitle(f"CATE / elasticity benchmark across {len(b)} estimators", fontsize=14, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out / "03_cate_pehe_and_elasticity_rmse_benchmark.png")
    plt.close(fig)


def qini_plot(preds: pd.DataFrame, bench: pd.DataFrame, out: Path) -> None:
    names = ["S-Learner", "T-Learner", "X-Learner", "R-Learner (NonParamDML)", "DR-Learner", "Causal Forest DML",
             "Uplift RF (causalml)", "TARNet", "DragonNet", "CEVAE"]
    tau = preds["true_promo_cate"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    f0, g0 = qini_curve(tau, tau)
    ax.plot(f0, g0, color=ps.INK, linewidth=1.2, linestyle=":", label="Oracle ranking")
    ax.plot([0, 1], [0, g0[-1]], color=ps.MUTED, linewidth=1, label="Random targeting")
    for i, n in enumerate(names):
        c = f"tau__{n}"
        if c not in preds:
            continue
        f, g = qini_curve(preds[c].to_numpy(), tau)
        ax.plot(f, g, color=ps.CAT[i % 5], linestyle=["-", "--", ":"][i // 5 % 3], linewidth=1.7, label=f"{n} (AUUC {auuc(preds[c].to_numpy(), tau):.2f})")
    ax.set(xlabel="fraction of reservations targeted (ranked by predicted CATE)", ylabel="cumulative true uplift per capita",
           title="Qini curves: who should receive the promotion?")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(out / "04_qini_and_cumulative_uplift_curves.png")
    plt.close(fig)


def revpar_frontier(models: dict, cfg: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=False)
    for ax, key in zip(axes, ["D", "A"]):
        sc = cfg["scenarios"][key]
        row = build_profile_df(sc["profile"], sc["instruments"], cfg=cfg)
        t = truth_curve(row, cfg, None)
        ax.plot(t["adr"], t["net_revpar"], color=ps.INK, linewidth=3, label="Ground truth (expected over U)")
        for name, col in [("Naive OLS", ps.CAT[4]), ("DeepIV (PyTorch)", ps.CAT[3]), ("DR-IV (LinearDRIV)", ps.CAT[0])]:
            if name not in models:
                continue
            c = revpar_curve(models[name], row, cfg)
            ax.plot(c["adr"], c["net_revpar"], color=col, linestyle="--" if "Naive" in name else "-", label=f"{name} (predicted)")
            i = int(c["net_revpar"].to_numpy().argmax())
            # mark where that policy lands on the TRUE curve
            tr_val = float(np.interp(c["adr"].iloc[i], t["adr"], t["net_revpar"]))
            ax.scatter([c["adr"].iloc[i]], [tr_val], s=70, color=col, edgecolor="white", linewidth=1.5, zorder=5)
            ax.annotate(f"${c['adr'].iloc[i]:.0f}", (c["adr"].iloc[i], tr_val), textcoords="offset points", xytext=(6, 8), fontsize=8, color=ps.INK)
        ax.set(xlabel="offered ADR ($)", ylabel="expected net RevPAR ($)")
        ax.set_title(f"Scenario {key}: " + sc["name"].split(" (")[0], fontsize=11)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("RevPAR frontier: the dot is where each policy lands on the true curve", fontsize=14, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out / "05_revpar_optimization_frontier.png")
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    out = resolve(cfg["paths"]["eval_plots"])
    mdir = resolve(cfg["paths"]["metrics"])
    models = load_models()
    ps.apply()
    hist = json.loads((mdir / "training_histories.json").read_text())
    bench = pd.read_csv(mdir / "benchmark_summary.csv")
    preds = pd.read_parquet(mdir / "test_predictions.parquet")
    test = load_bookings().set_index("booking_id").loc[preds["booking_id"]].reset_index()
    training_curves(hist, out)
    demand_curves(models, test, cfg, out)
    benchmark_bars(bench, out)
    qini_plot(preds, bench, out)
    revpar_frontier(models, cfg, out)
    log.info("Saved 5 evaluation plots to %s", out)


if __name__ == "__main__":
    main()
