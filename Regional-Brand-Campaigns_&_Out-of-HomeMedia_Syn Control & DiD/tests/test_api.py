from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "outputs/models/cate_meta.joblib").exists(), reason="run the pipeline first")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and len(r.json()["cate_models"]) >= 8


def test_predict_att_cached_and_validation(client):
    r = client.post("/predict/att", json={"scenario": "mia_all24", "estimator": "ASCM (Ridge)", "include_series": False})
    j = r.json()
    assert r.status_code == 200 and j["source"] == "cache" and j["att_mean"] > 0 and j["economics"]["iroas"] > 0
    assert client.post("/predict/att", json={"estimator": "nope"}).status_code == 422


def test_predict_att_live_matches_cache(client):
    a = client.post("/predict/att", json={"scenario": "mia_flagship", "estimator": "SCM", "include_series": False}).json()
    b = client.post("/predict/att", json={"scenario": "mia_flagship", "estimator": "SCM", "live": True, "include_series": False}).json()
    assert abs(a["att_mean"] - b["att_mean"]) < 1e-6 and b["source"].startswith("live")


def test_bayes_live_is_rejected(client):
    r = client.post("/predict/att", json={"estimator": "Bayesian SC", "live": True})
    assert r.status_code == 422


def test_predict_cate(client):
    body = {"rows": [{"is_weekend": 1, "weather_severity_index": 2}, {"is_weekend": 0, "weather_severity_index": 60}],
            "learner": "Linear DML"}
    r = client.post("/predict/cate", json=body)
    j = r.json()
    assert r.status_code == 200 and len(j["lift_pct"]) == 2 and j["ci"] is not None
    assert client.post("/predict/cate", json={"rows": [{"is_weekend": 5}]}).status_code == 422


def test_simulate_campaign(client):
    r = client.post("/simulate/campaign", json={"budget_usd": 3_000_000, "strategy": "optimized"})
    j = r.json()
    assert r.status_code == 200 and j["totals"]["incremental_room_nights"] > 0 and len(j["strategy_comparison"]) == 8
    best = max(x["incremental_room_nights"] for x in j["strategy_comparison"])
    assert j["totals"]["incremental_room_nights"] >= best - 1e-6      # the optimiser is at least as good as every heuristic
