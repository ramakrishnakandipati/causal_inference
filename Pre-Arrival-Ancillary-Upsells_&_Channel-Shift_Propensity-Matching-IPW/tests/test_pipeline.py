"""Unit + integration tests. The pipeline fixture runs the --fast pipeline into a temp output dir."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import synthetic_generator as sg  # noqa: E402
from src.evaluation import metrics_and_plots as mp  # noqa: E402
from src.models import propensity_ate as pa  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return sg.generate(8000, seed=7)


def test_dgp_shapes_and_truth(df):
    assert len(df) == 8000
    assert set(df.channel) == set(sg.CHANNELS)
    assert df.e_true.between(0.04, 0.96).all()
    assert np.allclose(df.tau_direct, df.p1_direct - df.p0_direct)
    assert set(df.segment) <= set(sg.SEGMENTS)
    X = sg.build_features(df)
    assert list(X.columns) == sg.feature_columns() and not X.isna().any().any()


def test_naive_is_confounded(df):
    t, c = df[df.treatment == 1], df[df.treatment == 0]
    naive = t.y_ancillary.mean() - c.y_ancillary.mean()
    assert naive > 1.8 * df.tau_anc.mean()          # strong selection bias by construction


def test_aipw_recovers_truth_and_smd_improves(df):
    X, T = sg.build_features(df), df.treatment.values
    e, _, _ = pa.fit_propensity(X, T, fast=True)
    y = df.y_ancillary.values
    mu1, mu0 = pa.outcome_models_crossfit(X, T, y, False, fast=True)
    ate = pa.aipw(e, T, y, mu1, mu0)[0]
    assert abs(ate - df.tau_anc.mean()) < 0.15 * df.tau_anc.mean() + 1.0
    bal = pa.balance_table(X, T, e)
    assert bal.smd_ipw.abs().max() < bal.smd_before.abs().max()
    assert bal.smd_ipw.abs().mean() < 0.1


def test_metrics_basic(df):
    T, y = df.treatment.values, df.y_ancillary.values
    e = df.e_true.values
    q_or, _, _ = mp.qini_auuc(df.tau_anc.values, T, y, e)
    q_rand, _, _ = mp.qini_auuc(np.random.default_rng(0).normal(size=len(df)), T, y, e)
    assert q_or > q_rand
    assert mp.pehe(df.tau_anc, df.tau_anc) == 0.0


@pytest.mark.parametrize("binary", [False, True])
def test_every_learner_fits(df, binary):
    from src.models.cate_learners import make_learners
    tr, te = sg.split(df)
    X, Xt = sg.build_features(tr), sg.build_features(te)
    col = "y_direct" if binary else "y_ancillary"
    for name, m in make_learners(binary, fast=True).items():
        m.fit(X, tr.treatment.values, tr[col].values)
        p = m.predict_cate(Xt)
        assert p.shape == (len(te),) and np.isfinite(p).all(), name


@pytest.fixture(scope="module")
def pipeline_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    env = dict(os.environ, CAUSAL_OUT=str(out))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--fast"], env=env, cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return out


def test_pipeline_artifacts(pipeline_out):
    for f in ("data/train.parquet", "data/test.parquet", "data/scored_guests.parquet", "metrics/ate_comparison.json",
              "metrics/cate_leaderboard.csv", "metrics/balance_smd.csv", "plots/love_plot.png", "plots/propensity_overlap.png",
              "plots/qini_curves_ancillary.png", "logs/pipeline.log"):
        assert (pipeline_out / f).exists(), f
    lb = pd.read_csv(pipeline_out / "metrics" / "cate_leaderboard.csv")
    assert lb[lb.outcome == "direct"].model.str.contains("Uplift RF").any()
    assert lb[lb.outcome == "ancillary"].shape[0] >= 10


def test_api_endpoints(pipeline_out, monkeypatch):
    monkeypatch.setenv("CAUSAL_OUT", str(pipeline_out))
    from fastapi.testclient import TestClient
    from src import serving
    from src.api.main import app
    serving.load_model.cache_clear()
    with TestClient(app) as c:
        h = c.get("/health").json()
        assert h["status"] == "ok"
        g = serving.PERSONAS["Expedia family, Suite, 60-day lead (iOS)"]
        r = c.post("/predict/cate", json={"guests": [g]})
        assert r.status_code == 200 and "tau_ancillary" in r.json()["predictions"][0]
        assert c.post("/predict/cate", json={"guests": [dict(g, channel="Nope")]}).status_code == 422
        o = c.post("/optimize/campaign", json={"budget_frac": 0.3}).json()
        assert o["n_selected"] > 0 and o["true_net_value"] >= o["random_net_value"]
        m = c.get("/models/metrics").json()
        assert len(m["leaderboard"]) > 10
