"""CUDA detection, VRAM logging and version-aware GPU parameter builders with CPU fallback."""
from __future__ import annotations

import os
import platform
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[2]

# Environment shim: an old dask can fail to import against recent pandas (AttributeError, not ImportError),
# which makes `import lightgbm` crash. Marking dask.dataframe as unavailable triggers LightGBM's own fallback.
try:  # pragma: no cover - environment dependent
    import dask.dataframe  # noqa: F401
except Exception:  # noqa: BLE001
    sys.modules["dask.dataframe"] = None  # type: ignore[assignment]


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    p = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel: str) -> Path:
    """Resolve a repo-relative path and make sure it exists as a directory when it has no suffix."""
    p = ROOT / rel
    if not p.suffix:
        p.mkdir(parents=True, exist_ok=True)
    return p


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def want_gpu(cfg: Dict[str, Any] | None = None, key: str | None = None) -> bool:
    cfg = cfg or load_config()
    g = cfg.get("gpu", {})
    if not g.get("use_gpu", True) or not cuda_available():
        return False
    return bool(g.get(key, True)) if key else True


def get_device(cfg: Dict[str, Any] | None = None):
    import torch

    return torch.device("cuda" if want_gpu(cfg) else "cpu")


def to_device(obj, device):
    return obj.to(device)


def hardware_summary() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cuda_available": cuda_available(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cpu_threads"] = torch.get_num_threads()
        if cuda_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_total_mem_gb"] = round(props.total_memory / 1024 ** 3, 2)
            info["gpu_allocated_mb"] = round(torch.cuda.memory_allocated(0) / 1024 ** 2, 1)
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = str(exc)
    return info


def log_vram(logger, tag: str = "") -> None:
    if not cuda_available():
        logger.info("[VRAM]%s CUDA unavailable - running on CPU", f" {tag}" if tag else "")
        return
    import torch

    alloc = torch.cuda.memory_allocated(0) / 1024 ** 2
    reserved = torch.cuda.memory_reserved(0) / 1024 ** 2
    logger.info("[VRAM]%s allocated=%.1f MB reserved=%.1f MB", f" {tag}" if tag else "", alloc, reserved)


def _version_tuple(v: str):
    out = []
    for part in v.split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def xgb_params(cfg: Dict[str, Any] | None = None, **overrides) -> Dict[str, Any]:
    """XGBoost params that work on both 1.x (gpu_hist) and 2.x (device='cuda')."""
    import xgboost as xgb

    cfg = cfg or load_config()
    nu = cfg["models"]["nuisance"]
    params: Dict[str, Any] = dict(
        n_estimators=nu["n_estimators"],
        max_depth=nu["max_depth"],
        learning_rate=nu["learning_rate"],
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=cfg["project"]["seed"],
        n_jobs=-1,
        verbosity=0,
    )
    gpu = want_gpu(cfg, "xgboost")
    if _version_tuple(xgb.__version__) >= (2, 0, 0):
        params.update(tree_method="hist", device="cuda" if gpu else "cpu")
    else:
        params.update(tree_method="gpu_hist" if gpu else "hist")
        if gpu:
            params["predictor"] = "gpu_predictor"
    params.update(overrides)
    return params


def lgbm_params(cfg: Dict[str, Any] | None = None, **overrides) -> Dict[str, Any]:
    cfg = cfg or load_config()
    nu = cfg["models"]["nuisance"]
    params: Dict[str, Any] = dict(
        n_estimators=nu["n_estimators"],
        max_depth=nu["max_depth"],
        learning_rate=nu["learning_rate"],
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=cfg["project"]["seed"],
        n_jobs=-1,
        verbose=-1,
    )
    if want_gpu(cfg, "lightgbm"):
        params["device"] = "gpu"
    params.update(overrides)
    return params


def catboost_params(cfg: Dict[str, Any] | None = None, **overrides) -> Dict[str, Any]:
    cfg = cfg or load_config()
    nu = cfg["models"]["nuisance"]
    params: Dict[str, Any] = dict(
        iterations=nu["n_estimators"],
        depth=nu["max_depth"],
        learning_rate=nu["learning_rate"],
        random_seed=cfg["project"]["seed"],
        verbose=False,
        allow_writing_files=False,
        task_type="GPU" if want_gpu(cfg, "catboost") else "CPU",
    )
    if params["task_type"] == "CPU":
        params["thread_count"] = -1
    params.update(overrides)
    return params


def make_xgb_regressor(cfg=None, **overrides):
    from xgboost import XGBRegressor

    return XGBRegressor(**xgb_params(cfg, **overrides))


def make_xgb_classifier(cfg=None, **overrides):
    from xgboost import XGBClassifier

    return XGBClassifier(**xgb_params(cfg, **overrides))


def make_lgbm_regressor(cfg=None, **overrides):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**lgbm_params(cfg, **overrides))


def make_catboost_regressor(cfg=None, **overrides):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(**catboost_params(cfg, **overrides))


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if cuda_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001
        pass
