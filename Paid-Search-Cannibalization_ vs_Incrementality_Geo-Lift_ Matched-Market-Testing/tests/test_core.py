"""Fast unit tests (no GPU/long training needed beyond tiny fits)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import feature_engineering as fe  # noqa: E402
from src.data.synthetic_geo_generator import generate_panel  # noqa: E402
from src.economics import cannibalization_roi as econ  # noqa: E402
from src.models import macro_geolift as mg  # noqa: E402
from src.models.model_registry import auuc_qini  # noqa: E402


@pytest.fixture(scope="module")
def gp():
    return generate_panel(seed=7)


def test_design_shape_and_stratified_split(gp):
    d = gp.dma
    assert len(d) == 50 and d["T"].sum() == 10
    assert d.groupby("tier")["T"].sum().to_dict() == {1: 2, 2: 4, 3: 4}
    assert gp.panel.post.sum() == 50 * 28 and gp.panel.day.max() == 24 * 7 - 1
    assert d.commission_rate.between(0.15, 0.20).all()


def test_no_effect_before_intervention(gp):
    pre = gp.panel[gp.panel.post == 0]
    assert pre.treated_active.sum() == 0
    assert (pre.branded_spend > 0).all() or pre.branded_spend.mean() > 0


def test_truth_signs(gp):
    d = gp.dma
    assert (d.true_post_tau_direct < 0).all()      # pausing never raises direct bookings
    assert (d.true_post_tau_ota > 0).all()         # and always leaks to OTAs
    assert ((d.true_cannibalization >= 0.15) & (d.true_cannibalization <= 0.97)).all()


def test_net_margin_identity(gp):
    p = gp.panel
    lhs = p.ncm
    rhs = p.direct_gbv - p.branded_spend + (p.ota_gbv - p.ota_commission)
    assert np.allclose(lhs, rhs)


def test_smd_and_matching(gp):
    pairs, _ = fe.match_markets(gp.panel, gp.dma)
    assert len(pairs) == 10 and pairs.control_dma_id.is_unique
    bal = fe.balance_table(gp.dma, pairs)
    assert bal.smd_matched.abs().max() < 0.5


def test_sdid_and_mmt_recover_direct_effect(gp):
    pairs, _ = fe.match_markets(gp.panel, gp.dma)
    gd = mg.GeoData(gp.panel, gp.dma, "direct_bookings")
    truth = gp.dma[gp.dma["T"] == 1].true_post_tau_direct.mean()
    for r in (mg.mmt(gd, pairs, gd.T - 28), mg.sdid(gd, gd.T - 28, n_placebo=60)):
        assert r["att"] < 0
        assert abs(r["att"] - truth) < 1.5          # within 1.5 bookings/DMA-day
        assert r["ci_low"] < truth < r["ci_high"] + 1.0


def test_sdid_weights_on_simplex(gp):
    gd = mg.GeoData(gp.panel, gp.dma, "direct_bookings")
    r = mg.sdid(gd, gd.T - 28, n_placebo=10)
    assert abs(r["unit_weights"].sum() - 1) < 1e-6 and (r["unit_weights"] >= -1e-9).all()
    assert abs(r["time_weights"].sum() - 1) < 1e-6


def test_qini_oracle_beats_random(gp):
    from src.data.feature_engineering import cate_frame
    d = cate_frame(gp.panel)
    res, _, _ = auuc_qini(d.tau_true.values, d.Y.values, d["T"].values.astype(int), d.tau_true.values)
    # a single random ranking is noisy (10 treated DMAs): compare with the average over many
    rng = np.random.default_rng(0)
    rnd = np.mean([auuc_qini(rng.normal(size=len(d)), d.Y.values, d["T"].values.astype(int))[0]["qini"] for _ in range(200)])
    assert res["qini"] > rnd


def test_simulator_decision_flips_with_cannibalization():
    lo = econ.simulate_policy({"cannibalization_rate": 0.4})
    hi = econ.simulate_policy({"cannibalization_rate": 0.98, "ota_capture_rate": 0.02})
    assert lo["decision"] == "Keep Branded SEM" and hi["decision"] == "Pause Branded SEM"
    cstar = lo["break_even_cannibalization"]
    at = econ.simulate_policy({"cannibalization_rate": cstar})
    assert abs(at["net_margin_change_if_paused_usd"]) < 1e-6 * max(1, at["branded_spend_usd"])


def test_economics_roas_ordering(gp):
    tr = gp.dma[gp.dma["T"] == 1]
    e = econ.programme_economics(gp.panel, gp.dma, tr.true_post_tau_direct.mean(), tr.true_post_tau_ota.mean())
    assert e["iroas"] < e["last_click_roas"] and 0 < e["cannibalization_rate"] < 1


def test_tarnet_and_dragonnet_fit_runs(gp):
    from src.data.feature_engineering import cate_frame, cate_matrix
    from src.models.deep_cate_gpu import DragonNet, TARNet
    d = cate_frame(gp.panel)
    X, _ = cate_matrix(d)
    for cls in (TARNet, DragonNet):
        m = cls(seed=0, epochs=5).fit(X, d["T"].values.astype(int), d.Y.values, d.dma_id.values)
        assert np.isfinite(m.predict(X)).all()
