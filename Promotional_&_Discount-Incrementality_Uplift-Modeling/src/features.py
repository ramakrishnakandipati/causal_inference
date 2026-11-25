"""Feature preprocessing, train/val/test split, propensity model, IPW weights and balance diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.common import FEATURES, LOYALTY_LEVELS, load_config, resolve

SKEWED = ["hist_booking_freq", "days_since_last_stay", "avg_lead_time", "points_balance"]


class Preprocessor:
    """log1p on skewed columns + standardization (used by neural nets / linear final stages)."""

    def __init__(self):
        self.scaler = StandardScaler()

    def _prep(self, X: pd.DataFrame) -> np.ndarray:
        Z = X[FEATURES].copy().astype(float)
        for c in SKEWED:
            Z[c] = np.log1p(Z[c])
        return Z.values

    def fit(self, X):
        self.scaler.fit(self._prep(X))
        return self

    def transform(self, X) -> np.ndarray:
        return self.scaler.transform(self._prep(X)).astype(np.float32)


def load_data(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = resolve(cfg["data"]["path"])
    if not path.exists():
        from data.synthetic_generator import main as gen
        gen()
    return pd.read_parquet(path)


def split_data(df: pd.DataFrame, cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    s, seed = cfg["data"]["split"], cfg["seed"]
    strat = df["T"].astype(str) + df["rct"].astype(str)
    tr, rest = train_test_split(df.index, train_size=s["train"], stratify=strat, random_state=seed)
    va, te = train_test_split(rest, train_size=s["val"] / (s["val"] + s["test"]),
                              stratify=strat.loc[rest], random_state=seed)
    return {"train": df.loc[tr].reset_index(drop=True), "val": df.loc[va].reset_index(drop=True),
            "test": df.loc[te].reset_index(drop=True)}


def fit_propensity(X_train, T_train, X_all_list, cfg: dict | None = None, n_folds: int = 5):
    """Cross-fitted GBM propensity: out-of-fold scores for the training set, plain predictions for others."""
    from sklearn.model_selection import StratifiedKFold
    from src.models import make_classifier
    cfg = cfg or load_config()
    clip = cfg["evaluation"]["propensity_clip"]
    X_train, T_train = np.asarray(X_train), np.asarray(T_train)
    oof = np.zeros(len(X_train))
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=cfg["seed"]).split(X_train, T_train):
        m = make_classifier(cfg, small=True).fit(X_train[tr], T_train[tr])
        oof[te] = m.predict_proba(X_train[te])[:, 1]
    full = make_classifier(cfg, small=True).fit(X_train, T_train)
    others = [np.clip(full.predict_proba(np.asarray(X))[:, 1], *clip) for X in X_all_list]
    return np.clip(oof, *clip), others, full


def ipw_weights(T, e, stabilized: bool = True):
    T, e = np.asarray(T), np.asarray(e)
    w = np.where(T == 1, 1.0 / e, 1.0 / (1.0 - e))
    if stabilized:
        p = T.mean()
        w = w * np.where(T == 1, p, 1 - p)
    return w


def standardized_mean_diff(X: pd.DataFrame, T, w=None) -> pd.Series:
    """|SMD| between treated and control for every feature (weighted if w given)."""
    T = np.asarray(T)
    w = np.ones(len(T)) if w is None else np.asarray(w)
    out = {}
    for c in X.columns:
        v = X[c].values.astype(float)
        m1 = np.average(v[T == 1], weights=w[T == 1]); m0 = np.average(v[T == 0], weights=w[T == 0])
        v1 = np.average((v[T == 1] - m1) ** 2, weights=w[T == 1]); v0 = np.average((v[T == 0] - m0) ** 2, weights=w[T == 0])
        out[c] = abs(m1 - m0) / np.sqrt((v1 + v0) / 2 + 1e-12)
    return pd.Series(out)


def overlap_summary(e_hat, T) -> dict:
    e_hat, T = np.asarray(e_hat), np.asarray(T)
    return {"min": float(e_hat.min()), "max": float(e_hat.max()),
            "frac_below_0.1": float((e_hat < 0.1).mean()), "frac_above_0.9": float((e_hat > 0.9).mean()),
            "mean_treated": float(e_hat[T == 1].mean()), "mean_control": float(e_hat[T == 0].mean())}


def loyalty_label(i: int) -> str:
    return LOYALTY_LEVELS[int(i)]
