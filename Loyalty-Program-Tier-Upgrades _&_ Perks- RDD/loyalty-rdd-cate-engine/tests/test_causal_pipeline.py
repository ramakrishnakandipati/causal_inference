"""Verification suite: RDD recovery, validity tests, CATE shapes/quality, metrics and API endpoints."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = 20
_KEEP_ALIVE: list = []


# ------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="session")
def sharp():
    from src.data.synthetic_generator import generate
    return generate(50_000, "sharp", seed=11)


@pytest.fixture(scope="session")
def fuzzy():
    from src.data.synthetic_generator import generate
    return generate(50_000, "fuzzy", seed=11)


@pytest.fixture(scope="session")
def cate_data(fuzzy):
    from src.utils.nuisance import CateData
    return CateData(fuzzy.sample(20_000, random_state=0), "next_year_room_nights", CUTOFF, 10)


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    """Quick end-to-end training run used by the API tests."""
    from src.models import trainer
    out = tmp_path_factory.mktemp("outputs")
    trainer.main(["--config", str(ROOT / "configs" / "config.yaml"), "--quick", "--outputs", str(out),
                  "--learners", "s_learner,t_learner,tarnet"])
    return out


@pytest.fixture(scope="session")
def client(artifacts):
    from fastapi.testclient import TestClient
    from src.api.main import create_app
    app = create_app(str(artifacts))
    c = TestClient(app)
    c.__enter__()                         # runs the lifespan (loads models)
    _KEEP_ALIVE.extend([app, c])          # never torn down: see tests/conftest.py (native abort on CUDA handle teardown)
    return c


GUEST = {"historical_adr": 165.0, "business_travel_ratio": 0.4, "lead_time_avg": 35.0, "ancillary_spend_t": 650.0,
         "direct_booking_ratio_t": 0.35, "tenure_years": 4.0, "co_brand_card_holder": False,
         "home_region": "North America", "age_group": "30-39", "weekend_stay_ratio": 0.3}


# ------------------------------------------------------------------------------------ data
def test_generator_hits_targets_and_regimes(sharp, fuzzy):
    from src.data.synthetic_generator import TARGETS, local_truth
    for df in (sharp, fuzzy):
        for k, v in local_truth(df).items():
            assert v == pytest.approx(TARGETS[k], rel=1e-6)
    assert (sharp["gold_status"] == (sharp["qualifying_nights_t"] >= CUTOFF)).all()
    below = fuzzy.loc[fuzzy.qualifying_nights_t < CUTOFF, "gold_status"].mean()
    above = fuzzy.loc[fuzzy.qualifying_nights_t >= CUTOFF, "gold_status"].mean()
    assert 0.09 < below < 0.15 and 0.91 < above < 0.97
    assert fuzzy["qualifying_nights_t"].between(1, 45).all()


def test_heterogeneity_corporate_lower_than_leisure(fuzzy):
    corp = fuzzy.loc[fuzzy.business_travel_ratio > 0.6, "tau_true_room_nights"].mean()
    leis = fuzzy.loc[fuzzy.business_travel_ratio < 0.3, "tau_true_room_nights"].mean()
    assert corp < leis


# ------------------------------------------------------------------------------------ RDD
def test_kernel_weights():
    from src.models.rdd_estimators import kernel_weights
    r = np.array([20, 22, 25, 30])
    assert kernel_weights(r, 20, 5, "triangular").tolist() == pytest.approx([1.0, 0.6, 0.0, 0.0])
    assert kernel_weights(r, 20, 5, "uniform")[1] == kernel_weights(r, 20, 5, "uniform")[0]
    assert kernel_weights(r, 20, 5, "epanechnikov")[2] == 0.0


@pytest.mark.parametrize("outcome,truth", [("next_year_room_nights", 3.4), ("next_year_ancillary_spend", 420.0),
                                           ("next_year_direct_booking_ratio", 0.11)])
def test_sharp_rdd_recovers_late(sharp, outcome, truth):
    from src.models.rdd_estimators import sharp_rdd
    res = sharp_rdd(sharp[outcome], sharp["qualifying_nights_t"], CUTOFF, 7, 1, "triangular")
    assert abs(res["estimate"] - truth) < 3.5 * res["se"] + 0.03 * truth
    assert res["p_value"] < 1e-6


def test_fuzzy_2sls_recovers_late_and_matches_linearmodels(fuzzy):
    from src.models.rdd_estimators import fuzzy_rdd, linearmodels_fuzzy_check
    res = fuzzy_rdd(fuzzy.next_year_room_nights, fuzzy.gold_status, fuzzy.qualifying_nights_t, CUTOFF, 6, 1)
    assert 0.75 < res["first_stage_jump"] < 0.88 and res["first_stage_f"] > 1000
    assert abs(res["estimate"] - 3.4) < 4 * res["se"] + 0.1          # truth is a complier-weighted local average
    lm = linearmodels_fuzzy_check(fuzzy, "next_year_room_nights", CUTOFF, 6)
    assert res["estimate"] == pytest.approx(lm["estimate"], rel=1e-6)
    assert res["se"] == pytest.approx(lm["se"], rel=0.02)


def test_polynomial_order_two_needs_enough_support(sharp):
    from src.models.rdd_estimators import sharp_rdd
    tiny = sharp_rdd(sharp.next_year_room_nights, sharp.qualifying_nights_t, CUTOFF, 3, 2, "triangular")
    assert np.isnan(tiny["estimate"])                                  # only 2 usable nights per side
    ok = sharp_rdd(sharp.next_year_room_nights, sharp.qualifying_nights_t, CUTOFF, 7, 2, "triangular")
    assert np.isfinite(ok["estimate"])


def test_bandwidth_selectors_are_sane(sharp):
    from src.models.rdd_estimators import cct_bandwidth, ik_bandwidth
    y, r = sharp.next_year_room_nights.to_numpy(), sharp.qualifying_nights_t.to_numpy(float)
    assert 1 <= ik_bandwidth(y, r, CUTOFF) <= 15
    assert 2 <= cct_bandwidth(y, r, CUTOFF) <= 20


def test_mccrary_passes_clean_and_detects_manipulation():
    from src.data.synthetic_generator import generate
    from src.models.rdd_estimators import mccrary_density_test
    ok = [mccrary_density_test(generate(50_000, "sharp", seed=s)["qualifying_nights_t"], CUTOFF)["p_value"] for s in (1, 2, 3, 4)]
    assert sum(p > 0.05 for p in ok) >= 3                              # 5% test: tolerate one false alarm
    bad = mccrary_density_test(generate(50_000, "sharp", manipulation=True, seed=1)["qualifying_nights_t"], CUTOFF)
    assert bad["reject_continuity_5pct"] and bad["p_value"] < 1e-6
    assert bad["log_density_ratio"] > 0.5                              # bunching just above the cutoff


def test_covariate_balance_has_no_jumps(sharp):
    from src.models.rdd_estimators import _COVS, covariate_balance
    bal = covariate_balance(sharp, _COVS, CUTOFF, 6)
    assert len(bal) == 10 and bal["balanced"].all()


def test_placebo_cutoffs_are_null_and_true_cutoff_is_not(sharp):
    from src.models.rdd_estimators import placebo_cutoffs, sharp_rdd
    pl = placebo_cutoffs(sharp, ["next_year_room_nights"], CUTOFF, [12, 15, 25, 30], 6)
    assert pl["null_not_rejected"].sum() >= 3
    assert (pl["estimate"].abs() < 1.0).all()                          # far below the true +3.4
    real = sharp_rdd(sharp.next_year_room_nights, sharp.qualifying_nights_t, CUTOFF, 6)
    assert real["estimate"] > 3 * pl["estimate"].abs().max()


def test_donut_hole_is_robust(sharp):
    from src.models.rdd_estimators import donut_rdd
    d = donut_rdd(sharp, ["next_year_room_nights"], CUTOFF, 6, (19, 20)).iloc[0]
    assert d["n_excluded"] > 0 and abs(d["donut"] - 3.4) < 0.9 and d["donut_p"] < 0.001


def test_full_rdd_suite_runs(fuzzy):
    import yaml
    from src.models.rdd_estimators import run_rdd_suite
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))["rdd"]
    s = run_rdd_suite(fuzzy, cfg)
    assert set(s) >= {"bandwidths", "main", "sensitivity", "mccrary", "balance", "placebo", "donut"}
    assert len(s["main"]) == 3 and len(s["sensitivity"]) == 3 * 3 * 2 * 5


# ------------------------------------------------------------------------------------ CATE
def test_cate_data_window_and_weights(cate_data):
    assert (np.abs(cate_data.r - CUTOFF) <= 10).all()
    assert cate_data.w.mean() == pytest.approx(1.0)
    assert cate_data.X.shape[1] == 10 and cate_data.tau_true is not None


@pytest.mark.parametrize("name", ["s_learner", "t_learner", "x_learner", "r_learner", "dr_learner",
                                  "nonparam_dml", "causal_forest_dml", "dmliv"])
def test_learner_shapes_and_signal(cate_data, name):
    from src.models.cate_learners import make_learner
    cd = cate_data
    n = 12_000
    L = make_learner(name).fit(cd.X[:n], cd.T[:n], cd.Y[:n], cd.w[:n], cd.Z[:n])
    tau = L.predict(cd.X[n:])
    assert tau.shape == (len(cd) - n,) and np.isfinite(tau).all()
    assert np.corrcoef(tau, cd.tau_true[n:])[0, 1] > 0.35
    assert 1.5 < tau.mean() < 5.0                                       # true average is about 3.4


@pytest.mark.parametrize("name", ["tarnet", "dragonnet"])
def test_neural_learners_train_and_pickle(cate_data, name, tmp_path):
    import joblib
    from src.models.cate_learners import make_learner
    cd = cate_data
    L = make_learner(name, {"epochs": 25, "patience": 5}).fit(cd.X[:8000], cd.T[:8000], cd.Y[:8000], cd.w[:8000])
    assert len(L.history["train"]) > 3 and len(L.history["train"]) == len(L.history["val"])
    a = L.predict(cd.X[8000:8200])
    joblib.dump(L, tmp_path / "m.joblib")
    b = joblib.load(tmp_path / "m.joblib").predict(cd.X[8000:8200])
    assert a.shape == (200,) and np.allclose(a, b, atol=1e-4)


def test_linear_dml_and_orthogonal_forest_interface(cate_data):
    from src.models.cate_learners import make_learner
    cd = cate_data
    lin = make_learner("linear_dml", {"degree": 1}).fit(cd.X[:6000], cd.T[:6000], cd.Y[:6000], cd.w[:6000])
    assert lin.predict(cd.X[:50]).shape == (50,)
    orf = make_learner("orthogonal_forest", {"n_trees": 8, "n_fit": 800, "n_ref": 40, "subsample_ratio": 0.4}) \
        .fit(cd.X[:6000], cd.T[:6000], cd.Y[:6000], cd.w[:6000])
    assert orf.predict(cd.X[:50]).shape == (50,)


def test_dmliv_requires_instrument(cate_data):
    from src.models.cate_learners import make_learner
    cd = cate_data
    with pytest.raises(ValueError):
        make_learner("dmliv").fit(cd.X[:500], cd.T[:500], cd.Y[:500], cd.w[:500], None)


def test_all_twelve_learners_registered():
    from src.models.cate_learners import LEARNERS
    assert len(LEARNERS) == 12 and LEARNERS["dmliv"].needs_instrument


def test_optuna_tuning_returns_params(cate_data):
    from src.models.cate_learners import tune_learner
    sub = cate_data
    r = tune_learner("s_learner", sub, n_trials=3, timeout=60)
    assert r["tuned"] and "n_estimators" in r["best_params"] and len(r["trials"]) >= 2


def test_slope_adjustment_removes_trend(cate_data):
    cd = cate_data
    for side in (0.0, 1.0):
        m = cd.Z == side
        slope = np.polyfit(cd.r[m] - CUTOFF, cd.Y[m], 1)[0]
        assert abs(slope) < abs(np.polyfit(cd.r[m] - CUTOFF, cd.Y_raw[m], 1)[0]) * 0.5


# ------------------------------------------------------------------------------------ metrics
def test_uplift_metrics_reward_good_rankings(cate_data):
    from src.evaluation import causal_metrics as M
    cd = cate_data
    good = M.uplift_metrics(cd.tau_true, cd.T, cd.Y, cd.w)
    bad = M.uplift_metrics(-cd.tau_true, cd.T, cd.Y, cd.w)
    assert good["auuc_over_random"] > bad["auuc_over_random"]
    assert good["uplift_at_10"] > bad["uplift_at_10"]
    curve = M.uplift_curve(cd.tau_true, cd.T, cd.Y, cd.w)
    assert curve["fraction"].is_monotonic_increasing and curve["fraction"].iloc[-1] == pytest.approx(1.0)


def test_policy_value_and_accuracy(cate_data):
    from src.evaluation import causal_metrics as M
    cd = cate_data
    scores = cd.tau_true + np.random.default_rng(0).normal(0, 1, len(cd))
    pv = M.policy_value(cd.tau_true, scores, cd.w, tau_true=cd.tau_true)
    assert pv["policy_value_oracle_eval"] == pytest.approx(pv["oracle_policy_value"])
    acc = M.cate_accuracy(cd.tau_true, cd.tau_true, cd.w)
    assert acc["pehe_rmse"] == 0 and acc["pearson"] == pytest.approx(1.0)


# ------------------------------------------------------------------------------------ trainer artifacts
def test_trainer_artifacts(artifacts):
    for rel in ("models/meta.json", "models/rdd_model.joblib", "models/panel.csv.gz", "metrics/rdd_late_estimates.csv",
                "metrics/cate_leaderboard_room_nights.csv", "models/room_nights/s_learner.joblib",
                "plots/01_rdd_discontinuity_main.png", "plots/02_mccrary_density_test.png",
                "logs/train.log", "logs/gpu_telemetry.jsonl"):
        assert (artifacts / rel).exists(), rel
    lb = pd.read_csv(artifacts / "metrics" / "cate_leaderboard_room_nights.csv")
    assert (lb["status"] == "ok").all() and {"pearson", "auuc", "policy_value_dr"} <= set(lb.columns)


# ------------------------------------------------------------------------------------ API
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["cutoff"] == 20 and "room_nights" in body["outcomes_trained"]


def test_predict_late(client):
    r = client.post("/predict/late", json={"outcome": "next_year_room_nights"})
    assert r.status_code == 200
    b = r.json()
    assert b["design"] == "fuzzy" and b["ci_low"] < b["estimate"] < b["ci_high"]
    assert abs(b["estimate"] - b["true_local_late"]) < 1.0
    r2 = client.post("/predict/late", json={"outcome": "next_year_ancillary_spend", "kernel": "uniform",
                                            "bandwidth": 8, "poly_order": 2, "design": "sharp"})
    assert r2.status_code == 200 and r2.json()["kernel"] == "uniform"


def test_predict_late_validation(client):
    assert client.post("/predict/late", json={"bandwidth": 1}).status_code == 422
    assert client.post("/predict/late", json={"outcome": "revenue"}).status_code == 422
    assert client.post("/predict/late", json={"bandwidth": 25}).status_code == 422                   # above the allowed range
    assert client.post("/predict/late", json={"poly_order": 3}).status_code == 422
    # the 4-night lower bound always leaves enough whole-night support points for a quadratic fit
    assert client.post("/predict/late", json={"bandwidth": 4, "poly_order": 2}).status_code == 200


def test_predict_cate(client):
    r = client.post("/predict/cate", json={"guest": GUEST})
    assert r.status_code == 200
    b = r.json()
    assert b["unit"] == "nights" and -2 < b["predicted_effect"] < 10 and 0 <= b["percentile_among_guests"] <= 100
    assert set(b["all_learner_estimates"]) == {"s_learner", "t_learner", "tarnet"}
    assert b["segment"] == "Bleisure traveller"
    r2 = client.post("/predict/cate", json={"guest": GUEST, "learner": "tarnet", "compare_all": False})
    assert r2.status_code == 200 and r2.json()["learner"] == "tarnet" and r2.json()["all_learner_estimates"] is None


def test_cate_orders_segments_sensibly(client):
    leisure = dict(GUEST, business_travel_ratio=0.1, weekend_stay_ratio=0.5)
    corp = dict(GUEST, business_travel_ratio=0.9, weekend_stay_ratio=0.05, historical_adr=420.0, lead_time_avg=5.0)
    a = client.post("/predict/cate", json={"guest": leisure, "compare_all": False}).json()["predicted_effect"]
    b = client.post("/predict/cate", json={"guest": corp, "compare_all": False}).json()["predicted_effect"]
    assert a > b


@pytest.mark.parametrize("patch", [{"business_travel_ratio": 1.5}, {"historical_adr": 5}, {"home_region": "Atlantis"},
                                   {"lead_time_avg": -1}, {"age_group": "12-17"}, {"tenure_years": 0.5, "co_brand_card_holder": True,
                                                                                    "ancillary_spend_t": 50000}])
def test_predict_cate_rejects_bad_profiles(client, patch):
    assert client.post("/predict/cate", json={"guest": dict(GUEST, **patch)}).status_code == 422


def test_predict_cate_unknown_learner_and_missing_field(client):
    assert client.post("/predict/cate", json={"guest": GUEST, "learner": "nope"}).status_code == 404
    g = dict(GUEST); g.pop("weekend_stay_ratio")
    assert client.post("/predict/cate", json={"guest": g}).status_code == 422


def test_simulate_threshold(client):
    lower = client.post("/simulate/threshold", json={"new_cutoff": 17, "perk_cost_per_member": 100}).json()
    assert lower["guests_affected"] > 0 and lower["incremental_total"] > 0 and "lower" in lower["direction"]
    assert lower["perk_cost_usd"] > 0 and lower["net_value_usd"] is not None
    higher = client.post("/simulate/threshold", json={"new_cutoff": 23}).json()
    assert higher["incremental_total"] < 0 and "raise" in higher["direction"]
    same = client.post("/simulate/threshold", json={"new_cutoff": 20}).json()
    assert same["guests_affected"] == 0
    far = client.post("/simulate/threshold", json={"new_cutoff": 12}).json()
    assert far["warnings"] and far["guests_affected"] > lower["guests_affected"]
    assert client.post("/simulate/threshold", json={"new_cutoff": 3}).status_code == 422


def test_leaderboard_and_docs(client):
    assert len(client.get("/leaderboard").json()) == 3
    assert client.get("/leaderboard", params={"outcome": "nope"}).status_code == 404
    assert client.get("/docs").status_code == 200 and "/predict/cate" in client.get("/openapi.json").json()["paths"]


@pytest.mark.parametrize("path,schema", [("/predict/late", "LateRequest"), ("/predict/cate", "CateRequest"),
                                         ("/simulate/threshold", "SimulateRequest")])
def test_swagger_examples_are_valid_requests(client, path, schema):
    """The pre-filled Swagger body must work as-is (it used to send learner='string' and 404)."""
    example = client.get("/openapi.json").json()["components"]["schemas"][schema]["example"]
    r = client.post(path, json=example)
    assert r.status_code == 200, r.text


def test_plotly_serialises_plain_lists_for_old_streamlit():
    """Streamlit 1.12 cannot decode plotly>=6 binary arrays (the dashboard charts would render empty)."""
    import plotly
    import plotly.graph_objects as go
    assert int(plotly.__version__.split(".")[0]) < 6
    assert "bdata" not in go.Figure(go.Bar(x=np.arange(5), y=np.arange(5))).to_json()


def test_api_without_models_returns_503(tmp_path):
    from fastapi.testclient import TestClient
    from src.api.main import create_app
    with TestClient(create_app(str(tmp_path))) as c:
        assert c.get("/health").status_code == 503
