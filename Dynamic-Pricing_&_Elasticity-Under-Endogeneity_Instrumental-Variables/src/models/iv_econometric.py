"""Econometric IV estimators: 2SLS / LIML / IV-GMM (linearmodels) and OrthoIV / DMLIV / DRIV / ForestDRIV (econml)."""
from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS, IVGMM, IVLIML
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from data.generate_hospitality_data import INSTRUMENTS, OUTCOME, TREATMENT
from src.models.base import LINEAR_FEATURES, BaseElasticityModel
from src.utils.gpu_utils import make_xgb_regressor
from src.utils.metrics import first_stage_f, to_jsonable

# Effect modifiers for heterogeneous (interacted) linear IV.
HET_VARS: List[str] = ["seg_corporate", "seg_group", "tier_luxury", "tier_resort", "loyalty_tier",
                       "hist_adr_sensitivity", "lead_time", "room_type"]


def _col(df: pd.DataFrame, c: str) -> np.ndarray:
    return df[c].to_numpy(float)


class LinearIV(BaseElasticityModel):
    """2SLS / LIML / IV-GMM with price x effect-modifier interactions (heterogeneous elasticity)."""

    family = "iv"
    is_iv = True

    def __init__(self, method: str = "2sls", hetero: bool = True, cfg=None):
        super().__init__(cfg)
        assert method in {"2sls", "liml", "gmm"}
        self.method, self.hetero = method, hetero
        self.name = {"2sls": "2SLS", "liml": "LIML", "gmm": "IV-GMM"}[method]

    # -- design -------------------------------------------------------------------------------
    def _h(self, df: pd.DataFrame) -> np.ndarray:
        return self.h_scaler.transform(df[HET_VARS].to_numpy(float))

    def _exog(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.ones(len(df)), self.x_scaler.transform(df[LINEAR_FEATURES].to_numpy(float))])

    def fit(self, train: pd.DataFrame) -> "LinearIV":
        t0 = time.time()
        self.x_scaler = StandardScaler().fit(train[LINEAR_FEATURES].to_numpy(float))
        self.h_scaler = StandardScaler().fit(train[HET_VARS].to_numpy(float))
        z = StandardScaler().fit(train[INSTRUMENTS].to_numpy(float))
        self.z_scaler = z
        zz = z.transform(train[INSTRUMENTS].to_numpy(float))
        t = _col(train, TREATMENT)
        if self.hetero:
            h = self._h(train)
            endog = np.column_stack([t, t[:, None] * h])
            inst = np.column_stack([zz] + [zz * h[:, [j]] for j in range(h.shape[1])])
        else:
            endog, inst = t[:, None], zz
        kw = dict(dependent=_col(train, OUTCOME), exog=self._exog(train), endog=endog, instruments=inst)
        if self.method == "2sls":
            self.res = IV2SLS(**kw).fit(cov_type="robust")
        elif self.method == "liml":
            self.res = IVLIML(**kw).fit(cov_type="robust")
        else:
            self.res = IVGMM(**kw, weight_type="robust").fit(cov_type="robust")
        n_ex = self._exog(train).shape[1]
        self.params = np.asarray(self.res.params)
        self.b_price = self.params[n_ex:]          # [b0, b_h1 ... b_hk] (or [b0])
        self.fit_seconds = time.time() - t0
        self.fit_structural_intercept(train)
        return self

    def elasticity(self, df: pd.DataFrame) -> np.ndarray:
        if not self.hetero:
            return np.full(len(df), self.b_price[0])
        return self.b_price[0] + self._h(df) @ self.b_price[1:]

    def get_diagnostics(self) -> Dict[str, Any]:
        return {"mean_price_coefficient": float(self.b_price[0])}


def iv_diagnostics(train: pd.DataFrame) -> Dict[str, Any]:
    """Constant-elasticity IV specification with the full battery of weak-IV / endogeneity / over-id tests."""
    xs = StandardScaler().fit_transform(train[LINEAR_FEATURES].to_numpy(float))
    exog = np.column_stack([np.ones(len(train)), xs])
    z = StandardScaler().fit_transform(train[INSTRUMENTS].to_numpy(float))
    y, t = _col(train, OUTCOME), _col(train, TREATMENT)

    r2 = IV2SLS(y, exog, t, z).fit(cov_type="robust")
    gmm = IVGMM(y, exog, t, z, weight_type="robust").fit(cov_type="robust")
    liml = IVLIML(y, exog, t, z).fit(cov_type="robust")
    ols = IV2SLS(y, np.column_stack([exog, t]), None, None).fit(cov_type="robust")

    fs = r2.first_stage.diagnostics
    row = fs.iloc[0]
    manual = first_stage_f(t, z, xs)
    wu = r2.wu_hausman()
    sargan, basmann = r2.sargan, r2.basmann
    out: Dict[str, Any] = {
        "coefficients": {
            "ols_price_coefficient": float(ols.params.iloc[-1] if hasattr(ols.params, "iloc") else ols.params[-1]),
            "2sls_price_coefficient": float(np.asarray(r2.params)[exog.shape[1]]),
            "2sls_std_error": float(np.asarray(r2.std_errors)[exog.shape[1]]),
            "liml_price_coefficient": float(np.asarray(liml.params)[exog.shape[1]]),
            "gmm_price_coefficient": float(np.asarray(gmm.params)[exog.shape[1]]),
            "true_mean_elasticity": float(train["true_elasticity"].mean()) if "true_elasticity" in train else None,
        },
        "first_stage": {
            "f_statistic_robust": float(row["f.stat"]),
            "f_pvalue": float(row["f.pval"]),
            "partial_r2": float(row["partial.rsquared"]),
            "cragg_donald_equivalent_f_homoskedastic": manual["f_stat"],
            "rule_of_thumb_threshold": 10.0,
            "instruments_strong": bool(manual["f_stat"] > 10),
            "n_instruments": len(INSTRUMENTS),
        },
        "wu_hausman_endogeneity": {"statistic": float(wu.stat), "pvalue": float(wu.pval),
                                    "reject_exogeneity_5pct": bool(wu.pval < 0.05)},
        "sargan_overidentification": {"statistic": float(sargan.stat), "pvalue": float(sargan.pval),
                                       "reject_validity_5pct": bool(sargan.pval < 0.05)},
        "basmann_overidentification": {"statistic": float(basmann.stat), "pvalue": float(basmann.pval)},
        "hansen_j_gmm": {"statistic": float(gmm.j_stat.stat), "pvalue": float(gmm.j_stat.pval)},
        "instruments": INSTRUMENTS,
        "n_obs": int(len(train)),
    }
    # The constant-elasticity spec ignores effect heterogeneity, which can make the over-identification test reject
    # in large samples even with valid instruments. Report the test for the heterogeneous (interacted) spec as well.
    het = LinearIV("2sls", hetero=True).fit(train).res
    out["sargan_overidentification_heterogeneous_spec"] = {
        "statistic": float(het.sargan.stat), "pvalue": float(het.sargan.pval),
        "reject_validity_5pct": bool(het.sargan.pval < 0.05),
        "n_overidentifying_restrictions": int(het.sargan.df if hasattr(het.sargan, "df") else -1)}
    out["notes"] = ("Sargan on the constant-elasticity spec is sensitive to misspecification (true elasticity varies by "
                    "segment); the heterogeneous spec is the one used for the elasticity estimates.")
    return to_jsonable(out)


class EconmlIV(BaseElasticityModel):
    """Wrapper around econml IV estimators with GPU-capable GBDT nuisance models.

    kind: 'dmliv' | 'orthoiv' | 'driv' | 'forestdriv'
    """

    family = "iv"
    is_iv = True
    _NAMES = {"dmliv": "DML-IV", "orthoiv": "OrthoIV", "driv": "DR-IV (LinearDRIV)", "forestdriv": "ForestDR-IV"}

    def __init__(self, kind: str = "dmliv", cfg=None, max_rows: int | None = None):
        super().__init__(cfg)
        self.kind = kind
        self.name = self._NAMES[kind]
        self.max_rows = max_rows

    def _xg(self):
        return make_xgb_regressor(self.cfg)

    def _build(self):
        from econml.iv.dml import DMLIV, OrthoIV
        from econml.iv.dr import ForestDRIV, LinearDRIV
        from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression

        seed = self.cfg["project"]["seed"]
        poly = PolynomialFeatures(1, include_bias=False)
        if self.kind == "dmliv":
            return DMLIV(model_y_xw=self._xg(), model_t_xw=self._xg(), model_t_xwz=self._xg(),
                         model_final=StatsModelsLinearRegression(fit_intercept=False),
                         featurizer=PolynomialFeatures(1, include_bias=True),
                         discrete_treatment=False, discrete_instrument=False, cv=3, random_state=seed)
        if self.kind == "orthoiv":
            return OrthoIV(model_y_xw=self._xg(), model_t_xw=self._xg(), model_z_xw=self._xg(),
                           projection=False, featurizer=poly, cv=3, random_state=seed)
        if self.kind == "driv":
            return LinearDRIV(model_y_xw=self._xg(), model_t_xw=self._xg(), model_z_xw=self._xg(),
                              model_tz_xw=self._xg(), cv=3, random_state=seed)
        fo = self.cfg["models"]["forest"]
        return ForestDRIV(model_y_xw=self._xg(), model_t_xw=self._xg(), model_z_xw=self._xg(),
                          model_tz_xw=self._xg(), cv=3, n_estimators=fo["n_estimators"],
                          min_samples_leaf=fo["min_samples_leaf"], random_state=seed)

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        return df[LINEAR_FEATURES].to_numpy(float)

    def _z(self, df: pd.DataFrame) -> np.ndarray:
        """Scalar instrument: the 2SLS-style linear combination of the three instruments (DRIV needs 1-D Z)."""
        z = df[INSTRUMENTS].to_numpy(float)
        return z if self.kind in ("dmliv",) else z @ self.z_coef

    def fit(self, train: pd.DataFrame) -> "EconmlIV":
        t0 = time.time()
        if self.max_rows and len(train) > self.max_rows:
            train = train.sample(self.max_rows, random_state=self.cfg["project"]["seed"])
        lr = LinearRegression().fit(np.column_stack([self._x(train), train[INSTRUMENTS].to_numpy(float)]),
                                    _col(train, TREATMENT))
        self.z_coef = lr.coef_[-len(INSTRUMENTS):]
        self.est = self._build()
        self.est.fit(_col(train, OUTCOME), _col(train, TREATMENT), Z=self._z(train), X=self._x(train))
        self.fit_seconds = time.time() - t0
        self.fit_structural_intercept(train)
        return self

    def elasticity(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.est.effect(self._x(df)), float).ravel()
