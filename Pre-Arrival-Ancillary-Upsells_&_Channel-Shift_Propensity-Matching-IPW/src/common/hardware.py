"""Hardware detection with clean CPU fallbacks."""
from __future__ import annotations

import importlib.util
import logging
from functools import lru_cache

log = logging.getLogger("hw")


@lru_cache(maxsize=1)
def get_device_report() -> dict:
    rep = {"cuda": False, "gpu_name": None, "torch_device": "cpu", "cuml": False,
           "xgb_device": "cpu", "catboost_task": "CPU", "lightgbm_device": "cpu"}
    try:
        import torch
        if torch.cuda.is_available():
            rep.update(cuda=True, gpu_name=torch.cuda.get_device_name(0), torch_device="cuda",
                       xgb_device="cuda", catboost_task="GPU")
    except Exception as e:  # pragma: no cover
        log.warning("torch probe failed: %s", e)
    if importlib.util.find_spec("cuml") is not None:
        try:
            import cuml  # noqa: F401
            rep["cuml"] = True
        except Exception:
            pass
    # LightGBM GPU needs a special build; probe with a tiny fit.
    if rep["cuda"]:
        try:
            import lightgbm as lgb
            import numpy as np
            X = np.random.rand(60, 3)
            y = np.random.rand(60)
            lgb.LGBMRegressor(device="gpu", n_estimators=2, verbose=-1).fit(X, y)
            rep["lightgbm_device"] = "gpu"
        except Exception:
            rep["lightgbm_device"] = "cpu"
    return rep


def torch_device():
    import torch
    return torch.device(get_device_report()["torch_device"])


def xgb_params() -> dict:
    return {"tree_method": "hist", "device": get_device_report()["xgb_device"]}


def lgb_params() -> dict:
    d = get_device_report()["lightgbm_device"]
    return {"device": d, "verbose": -1}


def catboost_params() -> dict:
    return {"task_type": get_device_report()["catboost_task"], "verbose": 0}
