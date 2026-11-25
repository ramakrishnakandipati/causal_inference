import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import auuc, normalized_qini, pehe, qini_curve, random_auuc, uplift_at_k  # noqa: E402


def _rct(n=40000, seed=0):
    r = np.random.default_rng(seed)
    tau = r.normal(0.1, 1.0, n)
    t = r.integers(0, 2, n)
    y = r.normal(0, 1, n) + t * tau
    return tau, t, y


def test_oracle_beats_random():
    tau, t, y = _rct()
    assert auuc(tau, y, t) > random_auuc(y, t) + 0.01
    assert abs(normalized_qini(tau, y, t, tau) - 1) < 1e-9
    assert abs(normalized_qini(np.random.default_rng(1).normal(size=len(y)), y, t, tau)) < 0.15


def test_curve_ends_at_ate():
    tau, t, y = _rct()
    x, q = qini_curve(tau, y, t)
    assert x[-1] == 1 and abs(q[-1] - tau.mean()) < 0.05


def test_uplift_at_k_and_pehe():
    tau, t, y = _rct()
    assert uplift_at_k(tau, y, t, 0.1) > uplift_at_k(-tau, y, t, 0.1)
    assert pehe(tau, tau) == 0
