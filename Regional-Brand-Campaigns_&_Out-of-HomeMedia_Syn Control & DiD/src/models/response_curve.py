"""Media response curve (geometric adstock -> Hill saturation) fitted to the estimated daily lift.

    A_t   = spend_t / 1000 + theta * A_{t-1}
    ATT_t = c * A_t^s / (A_t^s + K^s) * base_t * m_t

`base_t` is the counterfactual demand level, `m_t` an optional market-condition multiplier (e.g. the CATE
learner's predicted lift relative to its mean). The fit is done on the *estimated* daily ATT of the exposed
market, never on ground truth, so it is available in a real engagement. It powers the what-if simulator
(/simulate/campaign) and a pacing optimiser.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.signal import lfilter


def adstock(spend: np.ndarray, theta: float) -> np.ndarray:
    return lfilter([1.0], [1.0, -theta], np.asarray(spend, float))


def hill(A: np.ndarray, K: float, s: float) -> np.ndarray:
    A = np.maximum(A, 0.0)
    return A**s / (A**s + K**s)


@dataclass
class ResponseCurve:
    theta: float
    K: float
    s: float
    c: float
    r2: float
    rmse: float
    n: int

    def predict(self, spend: np.ndarray, base: np.ndarray, mod: np.ndarray | None = None) -> np.ndarray:
        m = np.ones_like(base, dtype=float) if mod is None else mod
        return self.c * hill(adstock(np.asarray(spend, float) / 1000.0, self.theta), self.K, self.s) * base * m

    def to_dict(self) -> dict:
        return asdict(self)


def fit_response_curve(spend: np.ndarray, att: np.ndarray, base: np.ndarray, mod: np.ndarray | None = None,
                       s_fixed: float | None = 1.0, seed: int = 0) -> ResponseCurve:
    """Non-linear least squares over (theta, K, c[, s]) with multi-start; spend/att/base are campaign-window arrays."""
    spend, att, base = (np.asarray(a, float) for a in (spend, att, base))
    m = np.ones_like(base) if mod is None else np.asarray(mod, float)
    rng = np.random.default_rng(seed)

    def model(p):
        theta, K, c = p[0], p[1], p[2]
        s = s_fixed if s_fixed is not None else p[3]
        return c * hill(adstock(spend / 1000.0, theta), K, s) * base * m

    def resid(p):
        return (model(p) - att) / max(att.std(), 1e-9)

    lo = [0.0, 5.0, 0.01] + ([] if s_fixed is not None else [0.7])
    # Prior: adstock decay <= 0.9 (half-life <= ~6.6 days). Longer carry-over cannot be told apart from slow demand
    # drift inside a 60-day window, and an unbounded fit pegs at the boundary and drives absurd "front-load it all" plans.
    hi = [0.9, 3000.0, 1.0] + ([] if s_fixed is not None else [2.5])
    best = None
    for _ in range(25):
        x0 = [rng.uniform(0.2, 0.9), np.exp(rng.uniform(np.log(30), np.log(800))), rng.uniform(0.05, 0.5)]
        if s_fixed is None:
            x0.append(rng.uniform(0.9, 1.8))
        try:
            r = least_squares(resid, x0, bounds=(lo, hi), max_nfev=400)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    pred = model(p)
    ss_res = float(((att - pred) ** 2).sum())
    ss_tot = float(((att - att.mean()) ** 2).sum())
    return ResponseCurve(theta=float(p[0]), K=float(p[1]), s=float(s_fixed if s_fixed is not None else p[3]),
                         c=float(p[2]), r2=1 - ss_res / max(ss_tot, 1e-12), rmse=float(np.sqrt(ss_res / len(att))),
                         n=len(att))


# ----------------------------------------------------------------------------- pacing strategies
def pacing_plan(strategy: str, total: float, days: int, weights: np.ndarray | None = None,
                holiday: np.ndarray | None = None, weekend: np.ndarray | None = None) -> np.ndarray:
    """Daily spend (USD) for a given total budget under a named pacing strategy."""
    t = np.arange(days)
    if strategy == "even":
        w = np.ones(days)
    elif strategy == "front_loaded":
        w = np.exp(-t / (days / 2.5))
    elif strategy == "back_loaded":
        w = np.exp(t / (days / 2.5))
    elif strategy == "pulsed_7_on_7_off":
        w = (((t // 7) % 2) == 0) * 1.0 + 0.05
    elif strategy == "holiday_weighted":
        w = 1.0 + 2.0 * (holiday if holiday is not None else np.zeros(days))
    elif strategy == "weekend_weighted":
        w = 1.0 + 1.5 * (weekend if weekend is not None else np.zeros(days))
    elif strategy == "cate_weighted":
        w = np.maximum(weights if weights is not None else np.ones(days), 1e-3) ** 2
    else:
        raise ValueError(f"unknown pacing strategy '{strategy}'")
    return total * w / w.sum()


def optimise_pacing(curve: ResponseCurve, total: float, base: np.ndarray, mod: np.ndarray,
                    min_frac: float = 0.3, max_frac: float = 2.5) -> np.ndarray:
    """Spend allocation maximising projected incremental room nights for a fixed budget (SLSQP).

    Every day keeps between `min_frac` and `max_frac` times the even-pacing spend: a brand campaign needs an
    always-on presence, and an unconstrained optimiser on a weakly identified curve just front-loads everything.
    """
    n = len(base)
    scale = total / n

    def neg(x):
        return -curve.predict(x * scale, base, mod).sum()

    x0 = np.ones(n)
    cons = {"type": "eq", "fun": lambda x: x.sum() - n}
    bnds = [(min_frac, max_frac)] * n
    r = minimize(neg, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
    x = np.clip(r.x, 0, None)
    return x / x.sum() * total
