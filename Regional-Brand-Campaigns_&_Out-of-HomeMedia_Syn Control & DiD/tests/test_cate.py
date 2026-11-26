import numpy as np
import pytest

from src.models import cate_learners as cl
from src.utils import device as dev


@pytest.fixture(scope="module")
def data(cfg, panel):
    return cl.build_cate_dataset(panel, cfg)


def test_dataset_structure(data):
    assert data.X.shape == (1500, 12) and data.D.sum() == 60 and data.eval_mask.sum() == 60
    assert set(np.unique(data.folds)) == {0, 1, 2, 3}
    for k in range(4):                                         # every cross-fit fold contains treated rows
        assert data.D[data.folds == k].sum() > 0
    assert abs(data.Y[data.D == 0].mean()) < 0.02              # donors' residuals are centred on zero


@pytest.mark.parametrize("name", ["S-Learner", "T-Learner", "X-Learner", "Domain Adaptation"])
def test_fast_learners_recover_positive_effect(cfg, data, name):
    out = cl.fit_all(data, cfg, seed=1, names=[name])[name]
    assert out["error"] is None
    t = out["tau_rel"][data.eval_mask] * data.base
    assert t.shape == (60,) and np.isfinite(t).all()
    truth = data.tau_true[data.eval_mask].mean()
    assert 0.3 * truth < t.mean() < 2.5 * truth


def test_linear_dml_uses_only_named_drivers_and_stays_sane(cfg, data):
    out = cl.fit_all(data, cfg, seed=4, names=["Linear DML"], keep_models=True)["Linear DML"]
    m = out["model"]
    drivers = set(cfg["cate"]["linear_dml_drivers"])
    nonzero = {n for n, c in zip(data.feature_names, m.coef_) if c != 0}
    assert nonzero <= drivers and len(nonzero) >= 4
    # coefficients must be on the scale of the effect itself (the collinear 12-feature fit gave +/-1,000 room nights per sd)
    assert np.abs(m.coef_ * data.base).max() < 400
    lo, hi = m.predict_interval(data.X)
    assert (lo <= hi).all()


def test_cpu_fallback_matches_interface(cfg, data):
    dev.set_gpu_enabled(False)
    try:
        assert dev.device_info()["xgb_device"] == "cpu" and not dev.device_info()["cuda"]
        out = cl.fit_all(data, cfg, seed=2, names=["S-Learner", "TARNet"])
        assert all(v["error"] is None and not v["gpu"] for v in out.values())
    finally:
        dev.set_gpu_enabled(True)


def test_deep_model_records_history(cfg, data):
    out = cl.fit_all(data, cfg, seed=3, names=["CFRNet-DragonNet"], keep_models=True)["CFRNet-DragonNet"]
    h = out["history"]
    assert {"train_loss", "val_loss", "train_mmd", "train_prop", "train_target"} <= set(h) and len(h["train_loss"]) >= 10
