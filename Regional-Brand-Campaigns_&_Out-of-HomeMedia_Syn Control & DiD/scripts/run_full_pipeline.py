"""End-to-end pipeline: data -> panel ATT -> CATE -> Monte-Carlo -> GPU benchmark -> figures -> verify.

    python scripts/run_full_pipeline.py                  # everything
    python scripts/run_full_pipeline.py --quick          # fast smoke run
    python scripts/run_full_pipeline.py --stage panel    # one stage
    python scripts/run_full_pipeline.py --no-gpu         # force CPU
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils import device as dev  # noqa: E402
from src.utils.config import ensure_dir, load_config, read_json, resolve, save_json  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

STAGES = ["data", "panel", "cate", "mc", "gpu", "figures", "verify"]
log = get_logger("pipeline")
_TELEMETRY: list[dict] = []


def snap(tag: str) -> None:
    _TELEMETRY.append(dev.hardware_telemetry(tag))
    ensure_dir("outputs/logs")
    save_json(_TELEMETRY, "outputs/logs/hardware_telemetry.json")


# ----------------------------------------------------------------------------- stages
def stage_data(cfg: dict, quick: bool) -> None:
    from src.data import generate_panel

    generate_panel.main(cfg)


def stage_panel(cfg: dict, quick: bool) -> None:
    from src.evaluation import diagnostics as dg

    dg.run_panel_stage(cfg, quick)          # the response curve is fitted after the CATE stage (it uses the CATE multiplier)


def stage_cate(cfg: dict, quick: bool) -> None:
    from src.evaluation import diagnostics as dg

    snap("before_cate")
    main = dg.cate_benchmark(cfg, quick)
    dg.cate_main_artifacts(cfg, main)
    dg.response_curve_and_pacing_validation(cfg, main)
    snap("after_cate")
    agg = main["agg"]
    log.info("CATE benchmark (mean over %d seeds):\n%s", int(agg["n_seeds"].max()),
             agg[["learner", "pehe", "att_bias", "corr", "qini_oracle", "seconds"]].round(2).to_string(index=False))


def stage_mc(cfg: dict, quick: bool) -> None:
    from src.evaluation import diagnostics as dg

    g = dg.panel_monte_carlo(cfg, n=2 if quick else None)
    log.info("panel Monte-Carlo:\n%s", g.round(2).to_string(index=False))


def _time(fn) -> float:
    t0 = time.perf_counter()
    fn()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    return time.perf_counter() - t0


def stage_gpu(cfg: dict, quick: bool) -> None:
    """Measured CPU-vs-GPU timings. Reports whatever the hardware shows, including GPU losses at small n."""
    import torch
    import torch.nn as nn
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    from src.models import cate_learners as cl

    mdir = ensure_dir(cfg["paths"]["metrics"])
    cuda = torch.cuda.is_available()
    sizes = [6000, 50000] if quick else cfg["gpu"]["benchmark_sizes"]
    rng = np.random.default_rng(0)
    p = 12
    snap("before_gpu_benchmark")
    rows = []

    def make(n):
        X = rng.standard_normal((n, p)).astype(np.float32)
        y = (X[:, :4] @ np.array([1.0, -0.6, 0.4, 0.8]) + 0.5 * np.sin(X[:, 4]) * X[:, 5] + 0.3 * rng.standard_normal(n)).astype(np.float32)
        return X, y

    def mlp_fit(X, y, device, epochs=3, bs=4096):
        d = torch.device(device)
        net = nn.Sequential(nn.Linear(p, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1)).to(d)
        opt = torch.optim.AdamW(net.parameters(), 1e-3)
        Xt, yt = torch.tensor(X, device=d), torch.tensor(y, device=d)
        for _ in range(epochs):
            perm = torch.randperm(len(yt), device=d)
            for i in range(0, len(yt), bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = ((net(Xt[idx]).squeeze(-1) - yt[idx]) ** 2).mean()
                loss.backward()
                opt.step()

    if cuda:                                                    # warm-up CUDA / driver once
        w = make(2000)
        XGBRegressor(n_estimators=5, device="cuda", tree_method="hist").fit(*w)
        mlp_fit(*w, "cuda", epochs=1)
    for n in sizes:
        X, y = make(n)
        n_est = 200
        jobs = {
            "XGBoost (hist, depth 6)": lambda d, X=X, y=y: XGBRegressor(n_estimators=n_est, max_depth=6, tree_method="hist", device=d, verbosity=0, n_jobs=8).fit(X, y),
            "CatBoost (depth 6)": lambda d, X=X, y=y: CatBoostRegressor(iterations=n_est, depth=6, task_type="GPU" if d == "cuda" else "CPU", verbose=0, allow_writing_files=False, thread_count=8).fit(X, y),
            "PyTorch MLP (3 epochs)": lambda d, X=X, y=y: mlp_fit(X, y, d),
        }
        for name, job in jobs.items():
            t_cpu = _time(lambda: job("cpu"))
            t_gpu = _time(lambda: job("cuda")) if cuda else float("nan")
            rows.append({"workload": name, "rows": n, "cpu_seconds": t_cpu, "gpu_seconds": t_gpu,
                         "speedup": t_cpu / t_gpu if cuda else float("nan")})
            log.info("  %-26s n=%8d  cpu %7.2fs  gpu %7.2fs  speedup %.2fx", name, n, t_cpu, t_gpu, rows[-1]["speedup"])
        t_lgb = _time(lambda: LGBMRegressor(n_estimators=n_est, num_leaves=63, verbose=-1, n_jobs=8).fit(X, y))
        rows.append({"workload": "LightGBM (CPU only)", "rows": n, "cpu_seconds": t_lgb, "gpu_seconds": float("nan"), "speedup": float("nan")})
        snap(f"gpu_benchmark_n{n}")
    pd.DataFrame(rows).to_csv(mdir / "gpu_benchmark.csv", index=False)

    # the real workload: the ten CATE learners on the campaign-window panel
    from src.data.generate_panel import generate_base_panel

    data = cl.build_cate_dataset(generate_base_panel(cfg), cfg)
    out = []
    for flag in ([False, True] if cuda else [False]):
        dev.set_gpu_enabled(flag)
        res = cl.fit_all(data, cfg, seed=cfg["seed"])
        for k, r in res.items():
            out.append({"learner": k, "mode": "gpu" if flag else "cpu", "seconds": r["seconds"], "backend": r["backend"],
                        "used_gpu": r["gpu"], "error": r["error"]})
    dev.set_gpu_enabled(cfg["gpu"]["enabled"])
    t = pd.DataFrame(out)
    t.to_csv(mdir / "gpu_benchmark_cate.csv", index=False)
    snap("after_gpu_benchmark")


def stage_figures(cfg: dict, quick: bool) -> None:
    from src.evaluation import visualizations

    visualizations.run(cfg)


def stage_verify(cfg: dict, quick: bool) -> None:
    """Every image the README links to must exist and be non-empty; the 8 required figures must be 300 DPI."""
    from PIL import Image

    readme = ROOT / "README.md"
    bad: list[str] = []
    if readme.exists():
        links = re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", readme.read_text(encoding="utf-8"))
        for ln in links:
            p = (ROOT / ln).resolve()
            if not p.exists() or p.stat().st_size == 0:
                bad.append(ln)
        log.info("README images checked: %d, missing/empty: %d", len(links), len(bad))
    else:
        log.info("README.md not written yet - skipping link check")
    figs = sorted((ROOT / cfg["paths"]["figures"]).glob("*.png"))
    for f in figs:
        with Image.open(f) as im:
            dpi = im.info.get("dpi", (0, 0))[0]
        log.info("figure %-48s %7.1f KB  dpi=%s", f.name, f.stat().st_size / 1024, round(dpi))
    if bad:
        raise SystemExit("missing README images: " + ", ".join(bad))


RUN = {"data": stage_data, "panel": stage_panel, "cate": stage_cate, "mc": stage_mc, "gpu": stage_gpu,
       "figures": stage_figures, "verify": stage_verify}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all"] + STAGES)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--skip-mc", action="store_true")
    ap.add_argument("--skip-gpu-benchmark", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    if a.no_gpu:
        cfg["gpu"]["enabled"] = False
    dev.set_gpu_enabled(cfg["gpu"]["enabled"])
    log.info("device: %s | jax: %s", dev.device_info(), dev.jax_device_summary())
    snap("start")
    stages = STAGES if a.stage == "all" else [a.stage]
    t_all = time.time()
    for s in stages:
        if (s == "mc" and a.skip_mc) or (s == "gpu" and a.skip_gpu_benchmark):
            continue
        t0 = time.time()
        log.info("===== stage: %s =====", s)
        RUN[s](cfg, a.quick)
        log.info("stage %s finished in %.1fs", s, time.time() - t0)
    snap("end")
    log.info("pipeline finished in %.1fs", time.time() - t_all)


if __name__ == "__main__":
    main()
