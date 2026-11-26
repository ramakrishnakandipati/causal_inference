"""Hardware detection with graceful CPU fallback. Logs the active devices once."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.utils.config import load_config

log = logging.getLogger("hardware")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


@dataclass
class Hardware:
    torch_device: str = "cpu"
    xgb_params: dict = field(default_factory=dict)
    catboost_params: dict = field(default_factory=dict)
    lgbm_params: dict = field(default_factory=dict)
    jax_platform: str = "cpu"
    rapids: bool = False
    gpu_name: str = ""

    def summary(self) -> dict:
        return {"torch": self.torch_device, "gpu": self.gpu_name or None,
                "xgboost": self.xgb_params.get("device", "cpu"),
                "catboost": self.catboost_params.get("task_type", "CPU"),
                "lightgbm": self.lgbm_params.get("device_type", "cpu"),
                "jax": self.jax_platform, "rapids": self.rapids}


_HW: Hardware | None = None


def get_hardware(force_cpu: bool = False) -> Hardware:
    global _HW
    if _HW is not None and not force_cpu:
        return _HW
    cfg = load_config()["gpu"]
    hw = Hardware()
    cuda = False
    try:
        import torch
        cuda = bool(cfg["use_gpu"]) and not force_cpu and torch.cuda.is_available()
        if cuda:
            hw.torch_device = "cuda"
            hw.gpu_name = torch.cuda.get_device_name(0)
    except Exception as e:  # torch missing
        log.warning("torch unavailable (%s); CPU only", e)
    if cuda:
        hw.xgb_params = dict(cfg["xgboost"])
        hw.catboost_params = dict(cfg["catboost"])
    else:
        hw.xgb_params = {"device": "cpu", "tree_method": "hist"}
        hw.catboost_params = {"task_type": "CPU"}
    hw.lgbm_params = dict(cfg["lightgbm"])
    try:
        import jax
        hw.jax_platform = jax.default_backend()  # 'cpu' on native Windows
    except Exception:
        hw.jax_platform = "unavailable"
    if cfg.get("rapids_optional"):
        try:
            import cudf, cuml  # noqa: F401
            hw.rapids = True
        except Exception:
            hw.rapids = False
    if not force_cpu:
        _HW = hw
        log.info("Active hardware: %s", hw.summary())
    return hw
