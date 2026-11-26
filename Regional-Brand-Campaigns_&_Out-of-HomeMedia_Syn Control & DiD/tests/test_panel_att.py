import numpy as np
import pandas as pd

from src.models import panel_att as pa
from src.models._solvers import simplex_ls


def _toy_panel(att=40.0, n=12, T=120, T0=90, seed=0):
    """Parallel-trends panel: unit FE + common time FE + noise; the first unit gets +att after T0."""
    rng = np.random.default_rng(seed)
    rows = []
    time_fe = np.cumsum(rng.normal(0, 3, T))
    for i in range(n):
        c = f"U{i:02d}"
        a = rng.normal(500, 60)
        for t in range(T):
            y = a + time_fe[t] + rng.normal(0, 4) + (att if (i == 0 and t >= T0) else 0)
            rows.append({"dma_code": c, "day_index": t + 1, "room_nights_booked": y, "ooh_ctv_spend_usd": 0.0, "is_weekend": 0.0,
                         "is_holiday_window": 0.0, **{k: rng.normal() for k in pa.TV_COVS},
                         "treatment_active": int(i == 0 and t >= T0), "post_period": int(t >= T0)})
    return pd.DataFrame(rows)


def test_simplex_ls_constraints_and_recovery():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 8))
    w_true = np.array([0.5, 0.3, 0.2, 0, 0, 0, 0, 0])
    w = simplex_ls(X, X @ w_true)
    assert abs(w.sum() - 1) < 1e-8 and (w >= -1e-10).all()
    assert np.allclose(w, w_true, atol=1e-4)


def test_estimators_recover_known_att_under_parallel_trends(cfg):
    df = _toy_panel(att=40.0)
    cfg = dict(cfg, estimators=dict(cfg["estimators"], calendar_adjust=False))
    pdm = pa.PanelData.from_df(df, "U00", outcome="room_nights_booked", start_day=91)
    assert abs(pa.estimator_twfe(pdm, cfg).att_mean - 40) < 4
    assert abs(pa.estimator_sdid(pdm, cfg).att_mean - 40) < 6
    assert abs(pa.estimator_csdid(pdm, cfg).att_mean - 40) < 8


def test_scm_family_on_real_panel(cfg, panel):
    pdm = pa.PanelData.from_df(panel, "MIA")
    scm = pa.estimator_scm(pdm, cfg)
    w = np.array(list(scm.weights.values()))
    assert abs(w.sum() - 1) < 1e-6 and (w >= -1e-9).all()
    asc = pa.estimator_ascm(pdm, cfg)
    assert abs(sum(asc.weights.values()) - 1) < 1e-6          # augmentation keeps weights summing to one
    truth = pa.truth_series(panel, "MIA", "room_nights_booked", pdm.T)[pdm.T0:].mean()
    for r in (scm, asc, pa.estimator_sdid(pdm, cfg)):
        assert abs(r.att_mean - truth) < 0.5 * truth, r.name


def test_event_study_grid_and_reference(cfg, panel):
    pdm = pa.PanelData.from_df(panel, "MIA")
    es = pa.estimator_event_study(pdm, cfg)
    k = es.meta["k"]
    assert k.min() == -14 and k.max() == 59 and len(k) == 74
    assert abs(es.meta["beta_full"][pdm.T0 - 1]) < 1e-9       # k = -1 is the reference period


def test_placebo_p_value_bounds(cfg, panel):
    pdm = pa.PanelData.from_df(panel, "MIA")
    res = pa.estimator_scm(pdm, cfg)
    pa.attach_placebo(res, pdm, pa.estimator_scm, cfg)
    assert len(res.meta["placebo"]["units"]) == 24
    assert 1 / 25 - 1e-12 <= res.meta["p_placebo"] <= 1.0
    assert res.se > 0 and res.ci_lo < res.att_mean < res.ci_hi


def test_in_time_placebo_is_near_zero(cfg, panel):
    from src.evaluation.diagnostics import in_time_placebo

    pdm = pa.PanelData.from_df(panel, "MIA")
    it = in_time_placebo(pdm, cfg).set_index("estimator")
    assert abs(it.loc["SCM", "att_pct_of_baseline"]) < 8 and abs(it.loc["ASCM (Ridge)", "att_pct_of_baseline"]) < 8


def test_response_curve_recovers_decay_on_noiseless_data():
    from src.models.response_curve import ResponseCurve, fit_response_curve

    rng = np.random.default_rng(0)
    spend = np.abs(rng.normal(50_000, 25_000, 60))
    base = np.full(60, 1500.0)
    true = ResponseCurve(0.7, 150.0, 1.0, 0.2, 1, 0, 60)
    fit = fit_response_curve(spend, true.predict(spend, base), base)
    assert abs(fit.theta - 0.7) < 0.05 and fit.r2 > 0.999
