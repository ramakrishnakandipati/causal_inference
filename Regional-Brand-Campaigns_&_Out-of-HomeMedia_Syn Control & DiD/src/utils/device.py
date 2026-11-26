"""Hardware detection, GPU/CPU backend selection and base-learner factories.

Every GPU path has an automatic CPU fallback:
  * XGBoost   -> device="cuda"            (CPU: device="cpu")
  * CatBoost  -> task_type="GPU"          (CPU: task_type="CPU")
  * LightGBM  -> CPU only (the pip wheel is built without CUDA)
  * PyTorch   -> torch.device("cuda")     (CPU: torch.device("cpu"))
"""
from __future__ import annotations

import os
import platform
import subprocess
import time
import warnings
from functools import lru_cache
from typing import Any

warnings.filterwarnings("ignore", message=".*mismatched devices.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# Expose 4 virtual CPU devices to JAX so NumPyro can run its MCMC chains in parallel.
# Must be set before JAX initialises; harmless if JAX is never imported.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

_GPU_ENABLED = True


def set_gpu_enabled(flag: bool) -> None:
    """Globally enable/disable GPU use (e.g. --no-gpu, or the CPU leg of the benchmark)."""
    global _GPU_ENABLED
    _GPU_ENABLED = bool(flag)


def gpu_enabled() -> bool:
    return _GPU_ENABLED


@lru_cache(maxsize=1)
def _cuda_probe() -> tuple[bool, str]:
    try:
        import torch

        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except Exception:
        pass
    return False, "cpu"


def device_info() -> dict:
    """Resolved backend choices given availability and the global GPU switch."""
    available, name = _cuda_probe()
    cuda = bool(available and _GPU_ENABLED)
    return {
        "cuda": cuda,
        "cuda_available": available,
        "gpu_name": name if cuda else "cpu",
        "torch_device": "cuda" if cuda else "cpu",
        "xgb_device": "cuda" if cuda else "cpu",
        "catboost_task": "GPU" if cuda else "CPU",
        "lightgbm": "cpu (pip wheel has no CUDA build)",
    }


def torch_device():
    import torch

    return torch.device(device_info()["torch_device"])


def jax_device_summary() -> dict:
    """What JAX sees (the Bayesian SC uses it). Native Windows wheels are CPU-only."""
    try:
        import jax

        devs = jax.devices()
        return {"backend": jax.default_backend(), "devices": [str(d) for d in devs]}
    except Exception as exc:  # pragma: no cover
        return {"backend": "unavailable", "devices": [], "error": str(exc)[:120]}


# ----------------------------------------------------------------------------- factories
def make_regressor(kind: str, seed: int = 0, n_estimators: int = 250, small: bool = False, **over: Any):
    """Gradient-boosting regressor on the best available backend."""
    dev = device_info()
    ne = max(60, n_estimators // 2) if small else n_estimators
    if kind == "xgboost":
        from xgboost import XGBRegressor

        p = dict(n_estimators=ne, learning_rate=0.05, max_depth=3 if small else 4, subsample=0.85,
                 colsample_bytree=0.85, min_child_weight=3, reg_lambda=2.0)
        p.update(over)
        return XGBRegressor(**p, device=dev["xgb_device"], tree_method="hist", random_state=seed,
                            n_jobs=4, verbosity=0)
    if kind == "catboost":
        from catboost import CatBoostRegressor

        p = dict(iterations=ne, depth=4 if small else 5, learning_rate=0.06, l2_leaf_reg=3.0)
        p.update(over)
        return CatBoostRegressor(**p, task_type=dev["catboost_task"], random_seed=seed, verbose=0,
                                 allow_writing_files=False)
    if kind == "lightgbm":
        from lightgbm import LGBMRegressor

        p = dict(n_estimators=ne, learning_rate=0.05, num_leaves=7 if small else 15, subsample=0.85,
                 subsample_freq=1, colsample_bytree=0.85, min_child_samples=8, reg_lambda=2.0)
        p.update(over)
        return LGBMRegressor(**p, random_state=seed, n_jobs=4, verbose=-1)
    raise ValueError(f"unknown base learner '{kind}'")


def make_classifier(kind: str, seed: int = 0, n_estimators: int = 200, small: bool = False, **over: Any):
    """Gradient-boosting classifier (used for propensity models)."""
    dev = device_info()
    ne = max(50, n_estimators // 2) if small else n_estimators
    if kind == "xgboost":
        from xgboost import XGBClassifier

        p = dict(n_estimators=ne, learning_rate=0.05, max_depth=3, subsample=0.85, colsample_bytree=0.85,
                 min_child_weight=3, reg_lambda=3.0)
        p.update(over)
        return XGBClassifier(**p, device=dev["xgb_device"], tree_method="hist", random_state=seed,
                             n_jobs=4, verbosity=0)
    if kind == "catboost":
        from catboost import CatBoostClassifier

        p = dict(iterations=ne, depth=4, learning_rate=0.06, l2_leaf_reg=5.0)
        p.update(over)
        return CatBoostClassifier(**p, task_type=dev["catboost_task"], random_seed=seed, verbose=0,
                                  allow_writing_files=False)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier

        p = dict(n_estimators=ne, learning_rate=0.05, num_leaves=7, subsample=0.85, subsample_freq=1,
                 colsample_bytree=0.85, min_child_samples=8, reg_lambda=3.0)
        p.update(over)
        return LGBMClassifier(**p, random_state=seed, n_jobs=4, verbose=-1)
    raise ValueError(f"unknown base learner '{kind}'")


# ----------------------------------------------------------------------------- telemetry
def gpu_telemetry() -> dict:
    """Snapshot from nvidia-smi (empty dict when unavailable)."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
        name, drv, util, mu, mt, temp, pw = [s.strip() for s in out.split(",")]
        return {"gpu_name": name, "driver": drv, "util_pct": float(util), "mem_used_mb": float(mu),
                "mem_total_mb": float(mt), "temp_c": float(temp), "power_w": _f(pw)}
    except Exception:
        return {}


def _f(x: str) -> float | None:
    try:
        return float(x)
    except ValueError:
        return None


def hardware_telemetry(tag: str = "") -> dict:
    info = device_info()
    snap: dict[str, Any] = {
        "tag": tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "device": info,
        "gpu": gpu_telemetry(),
        "jax": jax_device_summary(),
    }
    try:
        import torch

        snap["torch"] = torch.__version__
        if info["cuda"]:
            snap["torch_cuda_mem_allocated_mb"] = round(torch.cuda.memory_allocated() / 2**20, 1)
            snap["torch_cuda_max_mem_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except Exception:
        pass
    return snap
