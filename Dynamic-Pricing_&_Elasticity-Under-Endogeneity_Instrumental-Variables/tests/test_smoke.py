"""Fast checks of the DGP, optimizer and API (the API/optimizer tests need `make train` first)."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from data.generate_hospitality_data import FEATURES, INSTRUMENTS, OUTCOME, TREATMENT, generate  # noqa: E402
from src.utils.gpu_utils import load_config  # noqa: E402
from src.utils.metrics import auuc, first_stage_f  # noqa: E402
from src.utils.registry import load_models  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return generate(6000, 1)


def test_endogeneity_bias_and_strong_instruments(df):
    x = np.column_stack([np.ones(len(df)), df[TREATMENT], df[FEATURES].to_numpy(float)])
    ols = np.linalg.lstsq(x, df[OUTCOME].to_numpy(), rcond=None)[0][1]
    assert ols > 0 > df["true_elasticity"].mean()           # naive slope has the wrong sign
    assert first_stage_f(df[TREATMENT], df[INSTRUMENTS], df[FEATURES])["f_stat"] > 10
    assert (df["true_elasticity"] < 0).all()
    for z in INSTRUMENTS:                                    # exclusion: Z independent of the latent shock
        assert abs(np.corrcoef(df[z], df["demand_shock_u"])[0, 1]) < 0.06


def test_auuc_oracle_is_one():
    t = np.random.default_rng(0).normal(size=500)
    assert auuc(t, t) == pytest.approx(1.0)


@pytest.mark.skipif(not load_models(), reason="models not trained")
def test_optimizer_and_iv_beat_naive():
    from src.optimization.revenue_optimizer import build_profile_df, compare_policies
    cfg, m = load_config(), load_models()
    row = build_profile_df(cfg["scenarios"]["D"]["profile"], cfg["scenarios"]["D"]["instruments"], cfg=cfg)
    res = compare_policies({k: m[k] for k in ["Naive OLS", "DeepIV (PyTorch)"]}, row, cfg, 0.0)
    assert res["per_model"]["Naive OLS"]["estimated_elasticity"] > 0 > res["per_model"]["DeepIV (PyTorch)"]["estimated_elasticity"]
    assert res["revpar_lift_pct_causal_vs_naive"] > 0
