"""Hardware detection and GPU parameter helpers with graceful CPU fallback."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _ver(v: str) -> tuple:
    out = []
    for p in v.split(".")[:2]:
        digits = "".join(ch for ch in p if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


@lru_cache(maxsize=1)
def detect() -> Dict[str, Any]:
    """Probe the environment once and cache the result."""
    info: Dict[str, Any] = {"cuda": False, "gpu_name": None, "torch": None, "xgboost": None,
                            "catboost": False, "cupy": False, "cuml": False}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover
        logger.warning("torch unavailable: %s", exc)
    try:
        import xgboost
        info["xgboost"] = xgboost.__version__
    except Exception:  # pragma: no cover
        pass
    for mod in ("catboost", "cupy", "cuml"):
        try:
            __import__(mod)
            info[mod] = True
        except Exception:
            info[mod] = False
    return info


def torch_device(use_gpu: bool = True):
    import torch
    return torch.device("cuda" if (use_gpu and detect()["cuda"]) else "cpu")


def xgb_params(use_gpu: bool = True) -> Dict[str, Any]:
    """XGBoost >=2 uses device='cuda'; 1.x uses tree_method='gpu_hist'."""
    if use_gpu and detect()["cuda"]:
        ver = _ver(detect()["xgboost"] or "0.0")
        if ver >= (2, 0):
            return {"tree_method": "hist", "device": "cuda"}
        return {"tree_method": "gpu_hist"}
    return {"tree_method": "hist"}


def catboost_params(use_gpu: bool = True) -> Dict[str, Any]:
    if use_gpu and detect()["cuda"] and detect()["catboost"]:
        return {"task_type": "GPU"}
    return {"task_type": "CPU"}


def lgbm_params(use_gpu: bool = False) -> Dict[str, Any]:
    """The pip LightGBM wheel has no GPU build; request it only when explicitly enabled."""
    if use_gpu and detect()["cuda"]:
        return {"device_type": "gpu"}
    return {"device_type": "cpu"}


def hard_exit(status: int = 0) -> None:
    """Terminate without native DLL teardown.

    On Windows with xgboost 2.x (CUDA) + econml/joblib the process can abort (0xC0000409) *during DLL
    unload* even though all work finished; ``os._exit`` still runs DLL detach, ``TerminateProcess`` does not.
    Only call this at the very end of a command-line entry point, after outputs are written and flushed.
    """
    import os
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE          # 64-bit pseudo-handle must not be truncated
        k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.TerminateProcess(k32.GetCurrentProcess(), int(status))
    os._exit(int(status))


def describe() -> str:
    d = detect()
    return (f"cuda={d['cuda']} gpu={d['gpu_name']} torch={d['torch']} xgboost={d['xgboost']} "
            f"catboost={d['catboost']} cupy={d['cupy']} cuml={d['cuml']}")
