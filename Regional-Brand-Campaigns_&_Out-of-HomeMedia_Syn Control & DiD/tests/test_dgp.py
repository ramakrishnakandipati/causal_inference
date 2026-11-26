import numpy as np

from src.data.generate_panel import COVARIATES, DMA_ORDER, assemble_panel, build_latent, scenario_specs, spend_path


def test_shape_and_columns(panel):
    assert len(panel) == 6000
    assert panel["dma_code"].nunique() == 25 and panel["day_index"].nunique() == 240
    assert set(DMA_ORDER) == set(panel["dma_code"].astype(str).unique())
    needed = ["dma_code", "dma_name", "region", "tier", "date", "day_index", "post_period", "treated_unit", "cohort_start_day",
              "treatment_active", "ooh_ctv_spend_usd", "room_nights_booked", "direct_web_sessions", "branded_search_index",
              "incremental_net_revenue_usd", "y0_true", "y1_true", "true_att_it"] + COVARIATES
    assert not set(needed) - set(panel.columns)
    assert not panel.isna().any().any()


def test_treatment_structure(panel):
    t = panel[panel.treated_unit == 1]
    assert set(t["dma_code"].astype(str)) == {"MIA"}
    assert (panel.loc[panel.treated_unit == 0, "cohort_start_day"] == 0).all()
    assert (t["cohort_start_day"] == 181).all()
    active = panel[panel.treatment_active == 1]
    assert len(active) == 60 and active["day_index"].min() == 181 and active["day_index"].max() == 240
    assert (panel.loc[panel.post_period == 1, "day_index"] >= 181).all()


def test_ground_truth_identity(panel):
    assert np.allclose(panel["y1_true"] - panel["y0_true"], panel["true_att_it"], atol=2e-3)
    assert np.allclose(panel["room_nights_booked"], panel["y0_true"] + panel["true_att_it"], atol=2e-3)
    assert (panel.loc[panel.treatment_active == 0, "true_att_it"] == 0).all()
    assert (panel.loc[panel.treatment_active == 1, "true_att_it"] > 0).all()
    assert (panel.loc[panel.treated_unit == 0, "ooh_ctv_spend_usd"] == 0).all()


def test_effect_is_heterogeneous_and_sized(panel):
    a = panel[panel.treatment_active == 1]
    assert a["true_att_it"].std() > 5
    assert 0.05 < a["true_att_it"].sum() / a["y0_true"].sum() < 0.12


def test_placebo_scenario_has_no_effect(cfg, lat):
    df = assemble_panel(lat, scenario_specs(cfg)["tpa_placebo"], 11)
    assert df["true_att_it"].abs().sum() == 0 and df["treatment_active"].sum() == 0
    assert df["ooh_ctv_spend_usd"].sum() == 0


def test_reproducible(cfg):
    a = build_latent(cfg, 3).y0["room_nights_booked"]
    b = build_latent(cfg, 3).y0["room_nights_booked"]
    assert np.array_equal(a, b)


def test_spend_path_sums_to_budget(lat):
    for pat in ["winter_airport", "convention_billboard", "awareness_push"]:
        s = spend_path(pat, 1_000_000, lat.cal, 181, 1)
        assert abs(s.sum() - 1_000_000) < 1e-6 and (s[:180] == 0).all() and (s[180:] > 0).all()
