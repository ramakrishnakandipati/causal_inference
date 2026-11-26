"""Streamlit dashboard.  streamlit run app/streamlit_app.py --server.port 8501"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.api import service  # noqa: E402
from src.data.synthetic_geo_generator import load_panel  # noqa: E402
from src.utils.config import path  # noqa: E402

st.set_page_config(page_title="Paid Search Incrementality - Grand Horizon", layout="wide")
M = path("metrics")
PLOTS = path("plots")


@st.cache_data
def read_csv(name, **kw):
    return pd.read_csv(M / name, **kw)


@st.cache_data
def read_json(name):
    return json.loads((M / name).read_text())


@st.cache_data
def panel():
    gp = load_panel()
    return gp.panel, gp.dma


def usd(x):
    return f"${x:,.0f}"


st.title("Paid Search Cannibalization vs. Incrementality")
st.caption("Grand Horizon Hotels & Resorts - 50-DMA geo-holdout of branded paid search (20-week pre, 4-week pause in 10 DMAs). "
           "Synthetic data with known ground truth.")
hw = read_json("hardware.json")
st.sidebar.markdown("**Active hardware**")
st.sidebar.json(hw)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Geo-Lift & Cannibalization", "2. CATE Leaderboard & Uplift",
                                        "3. Policy Simulator", "4. Refutation & DMA Policy", "5. End-to-End Flows"])

# ------------------------------------------------------------------------------------------- tab 1
with tab1:
    pan, dma = panel()
    prog = read_json("programme_economics.json")
    mac = read_json("macro_results.json")
    series = read_csv("macro_series.csv")
    pairs = read_csv("matched_pairs.csv")
    hold = dma[dma["T"] == 1].sort_values("dma")
    c1, c2 = st.columns([1, 3])
    with c1:
        choice = st.selectbox("DMA", ["All holdout DMAs (aggregate)"] + list(hold.dma) + ["-- control DMAs --"] + list(dma[dma["T"] == 0].dma.sort_values()),
                              key="dma_sel")
        method = st.selectbox("Estimator for programme economics", ["SDID", "MMT", "BSTS", "Truth"], key="est_sel")
    cut = 140
    if choice.startswith("--"):
        st.info("Select a DMA.")
    elif choice.startswith("All"):
        s = series[(series.outcome == "direct_bookings")]
        so = series[(series.outcome == "ota_bookings")]
        with c2:
            for ttl, ss in (("Direct bookings per holdout DMA-day", s), ("OTA bookings per holdout DMA-day", so)):
                piv = pd.DataFrame({"observed": ss[ss.method == "BSTS"].set_index("day").observed})
                for m_ in ("BSTS", "SDID", "MMT"):
                    piv[f"{m_} counterfactual"] = ss[ss.method == m_].set_index("day").counterfactual
                b = ss[ss.method == "BSTS"].set_index("day")
                piv["BSTS 95% low"], piv["BSTS 95% high"] = b.cf_low, b.cf_high
                st.markdown(f"**{ttl}** (pause begins day {cut})")
                st.line_chart(piv, height=260)
    else:
        row = dma[dma.dma == choice].iloc[0]
        d = pan[pan.dma_id == row.dma_id].set_index("day")
        out = pd.DataFrame({"direct": d.direct_bookings, "OTA": d.ota_bookings})
        if row["T"] == 1:
            pr = pairs[pairs.treated_dma_id == row.dma_id].iloc[0]
            dc = pan[pan.dma_id == pr.control_dma_id].set_index("day")
            ratio = d.direct_bookings[:cut].mean() / dc.direct_bookings[:cut].mean()
            out["direct counterfactual (matched control x pre ratio)"] = dc.direct_bookings * ratio
            ratio_o = d.ota_bookings[:cut].mean() / dc.ota_bookings[:cut].mean()
            out["OTA counterfactual (matched control x pre ratio)"] = dc.ota_bookings * ratio_o
            st.caption(f"Matched control: {pr.control_dma} (DTW {pr.dtw:.2f}, Mahalanobis {pr.mahalanobis:.2f})")
        with c2:
            st.line_chart(out.rolling(7, min_periods=1).mean(), height=300)
            pre, post = d[d.index < cut], d[d.index >= cut]
            k = st.columns(4)
            k[0].metric("Direct/day, pre -> post", f"{pre.direct_bookings.mean():.1f} -> {post.direct_bookings.mean():.1f}")
            k[1].metric("OTA/day, pre -> post", f"{pre.ota_bookings.mean():.1f} -> {post.ota_bookings.mean():.1f}")
            k[2].metric("OTA impression share", f"{pre.ota_impression_share.mean():.0%} -> {post.ota_impression_share.mean():.0%}")
            k[3].metric("Branded spend/day", f"{usd(pre.branded_spend.mean())} -> {usd(post.branded_spend.mean())}")
    e = prog[method]
    st.subheader(f"Last-click ROAS vs incremental ROAS - programme level ({method})")
    k = st.columns(5)
    k[0].metric("Last-click ROAS", f"{e['last_click_roas']:.1f}x")
    k[1].metric("Incremental ROAS", f"{e['iroas']:.1f}x")
    k[2].metric("Cannibalization rate", f"{e['cannibalization_rate']:.0%}")
    k[3].metric("OTA commission leakage", usd(e["ota_commission_leakage_usd"]))
    k[4].metric("Net value of branded SEM", usd(e["net_value_of_branded_search_usd"]),
                help="Change in net contribution margin from KEEPING branded search = -(effect of pausing)")
    st.image(str(PLOTS / "iroas_waterfall.png"), width="stretch")
    st.image(str(PLOTS / "att_estimators.png"), width="stretch")

# ------------------------------------------------------------------------------------------- tab 2
with tab2:
    target = st.radio("Outcome", ["direct", "ota"], horizontal=True, key="target",
                      format_func=lambda t: {"direct": "Direct bookings lift", "ota": "OTA bookings lift"}[t])
    lb = read_csv(f"cate_leaderboard_{target}.csv", index_col=0)
    show = lb[["pehe", "pehe_norm", "auuc", "qini", "qini_ratio", "rank_corr", "ate_hat", "ate_true", "seconds"]].round(3)
    st.dataframe(show, width="stretch")
    st.caption("Out-of-fold, DMA-grouped 5-fold evaluation. PEHE is in units of lift (0.05 = 5 percentage points of baseline "
               "bookings). PEHE/sd(true CATE) > 1 means the learner is worse at the DMA level than the constant average effect: "
               "with 10 treated DMAs, average effects are recovered far better than heterogeneity.")
    q = read_csv(f"qini_curves_{target}.csv", index_col=0)
    algos = st.multiselect("Qini curves", list(q.columns), default=[c for c in q.columns if c in ("CausalForestDML", "TARNet", "S-Learner", "Oracle (true tau)", "Random")], key="qsel")
    if algos:
        st.line_chart(q[algos], height=320)
    imp = read_csv(f"cate_feature_importance_{target}.csv", index_col=0)
    a = st.selectbox("Feature importance (permutation, share of CATE movement)", list(imp.columns), key="imp_sel")
    st.bar_chart(imp[a].sort_values(ascending=False), height=280)
    if (PLOTS / f"leaderboard_{target}.png").exists():
        st.image(str(PLOTS / f"leaderboard_{target}.png"), width="stretch")
    st.image(str(PLOTS / "cate_heterogeneity.png"), width="stretch")

# ------------------------------------------------------------------------------------------- tab 3
from src.api.presets import PRESETS  # noqa: E402

SLIDERS = {  # key: (label, min, max, step)
    "baseline_direct_bookings_day": ("Baseline direct bookings / day", 2.0, 120.0, 1.0),
    "ota_to_direct_ratio": ("OTA : direct booking ratio", 0.1, 1.5, 0.05),
    "lastclick_share": ("Share of direct bookings last-click credited to branded paid", 0.05, 0.6, 0.01),
    "cannibalization_rate": ("Cannibalization rate (would book via organic anyway)", 0.0, 1.0, 0.01),
    "ota_capture_rate": ("OTA hijack rate of organic clickers when paused", 0.0, 1.0, 0.01),
    "diversion_rate": ("Share of lost incremental bookings that re-book on an OTA", 0.0, 1.0, 0.05),
    "adr": ("ADR ($)", 80.0, 450.0, 5.0), "commission_rate": ("OTA commission rate", 0.15, 0.20, 0.0025),
    "cpc": ("Branded CPC ($)", 0.3, 6.0, 0.05),
}
COVS = {"organic_rank": ("Organic SERP rank (1 = top)", 1.0, 5.0, 0.1), "ota_pressure": ("OTA auction pressure", 0.0, 1.0, 0.01),
        "loyalty_pen": ("Loyalty penetration", 0.05, 0.5, 0.01), "lead_time": ("Booking lead time (days)", 3.0, 90.0, 1.0),
        "competitor_adr": ("Competitor ADR ($)", 80.0, 450.0, 5.0)}


def load_preset(k):
    for key, v in PRESETS[k].items():
        if key != "label":
            st.session_state[f"sim_{key}"] = v
    st.session_state["preset"] = k


with tab3:
    st.markdown("**One-click presets** (calibrated to the experiment's tier archetypes) - then adjust the sliders.")
    pc = st.columns(3)
    for col, k in zip(pc, PRESETS):
        col.button(f"Preset {k}: {PRESETS[k]['label']}", key=f"btn_{k}", on_click=load_preset, args=(k,), width="stretch")
    if "sim_tier" not in st.session_state:
        load_preset("B")
    st.markdown(f"### Scenario: {PRESETS[st.session_state['preset']]['label']}")
    left, right = st.columns([1, 1])
    vals = {}
    with left:
        st.markdown("**Economics & cannibalization**")
        for key, (lab, lo, hi, stp) in SLIDERS.items():
            vals[key] = st.slider(lab, lo, hi, float(st.session_state[f"sim_{key}"]), stp, key=f"sim_{key}")
        horizon = st.slider("Horizon (days)", 7, 90, 28, 7, key="sim_h")
        st.markdown("**DMA profile for the CATE models**")
        st.selectbox("Market tier", [1, 2, 3], index=[1, 2, 3].index(st.session_state["sim_tier"]), key="sim_tier")
        for key, (lab, lo, hi, stp) in COVS.items():
            vals[key] = st.slider(lab, lo, hi, float(st.session_state[f"sim_{key}"]), stp, key=f"sim_{key}")
    sim_in = {k: float(vals[k]) for k in SLIDERS} | {"cvr": 0.025, "horizon_days": int(horizon)}
    sim = service.simulate_holdout_policy(sim_in)
    with right:
        st.markdown("**Structural simulation of pausing branded SEM**")
        (st.success if sim["decision"].startswith("Pause") else st.warning)(f"Policy decision: **{sim['decision']}**")
        k = st.columns(3)
        k[0].metric("Last-click ROAS", f"{sim['last_click_roas']:.1f}x")
        k[1].metric("Incremental ROAS", f"{sim['iroas']:.1f}x")
        k[2].metric("Cannibalization", f"{sim['cannibalization_rate']:.0%}")
        k = st.columns(3)
        k[0].metric(f"Net margin change if paused ({horizon}d)", usd(sim["net_margin_change_if_paused_usd"]))
        k[1].metric("OTA commission leakage", usd(sim["ota_commission_leakage_usd"]))
        k[2].metric("Break-even cannibalization", f"{sim['break_even_cannibalization']:.0%}",
                    help="Pausing pays off only above this cannibalization rate")
        st.bar_chart(pd.Series({"Branded spend saved": sim["branded_spend_usd"],
                                "Direct GBV lost (incl. hijack)": -sim["incremental_direct_bookings_period"] * sim_in["adr"] * 2.4,
                                "OTA margin gained": sim["ota_leakage_bookings_period"] * sim_in["adr"] * 2.4 * 0.95 * (1 - sim_in["commission_rate"])}), height=240)
        st.markdown("**CATE-model prediction for this DMA profile** (top-3 ensemble, trained on the experiment)")
        prof = {"tier": int(st.session_state["sim_tier"]), "flight_idx": 100.0, "baseline_direct_bookings_day": vals["baseline_direct_bookings_day"],
                "ota_to_direct_ratio": vals["ota_to_direct_ratio"], "lastclick_share": vals["lastclick_share"], "adr": vals["adr"],
                "commission_rate": vals["commission_rate"], "cpc": vals["cpc"]} | {k_: float(vals[k_]) for k_ in COVS}
        try:
            pr = service.predict_dma_cate(prof)
            k = st.columns(3)
            k[0].metric("Direct lift if paused", f"{pr['direct_lift_pct']:.1f}%")
            k[1].metric("OTA lift if paused", f"{pr['ota_lift_pct']:.1f}%")
            k[2].metric("Model decision", pr["decision"].replace(" Branded SEM", ""))
            st.caption(f"Ensemble: direct = {', '.join(pr['ensemble_models']['direct'])}; OTA = {', '.join(pr['ensemble_models']['ota'])}. "
                       "Individual DMA-level CATEs are noisy (see Tab 2); treat this as a second opinion, not the decision rule.")
        except Exception as ex:  # models not trained
            st.info(f"CATE models unavailable: {ex}")

# ------------------------------------------------------------------------------------------- tab 4
with tab4:
    pol = read_csv("dma_policy_recommendations.csv")
    val = read_json("policy_value.json")
    st.subheader("Optimal bidding policy by DMA")
    k = st.columns(4)
    k[0].metric("Model policy value (4 wk)", usd(val["model_policy_usd"]), help="Realised margin gain vs always-keep, using ground-truth effects")
    k[1].metric("Oracle value", usd(val["oracle_usd"]))
    k[2].metric("Always pause", usd(val["always_pause_usd"]))
    k[3].metric("Decision accuracy", f"{val['decision_accuracy']:.0%}")
    st.caption(f"Rule: {val['rule']}. Realised value uses the synthetic ground truth, which a real experiment never has.")
    rc1, rc2 = st.columns(2)
    rc1.markdown("**Alternative rules (realised 4-week value)**")
    rc1.dataframe(read_csv("policy_rules.csv").round(2), width="stretch")
    rc2.markdown("**True value of pausing by tier** (USD, 4 weeks; negative = keep branded SEM)")
    rc2.dataframe(read_csv("policy_by_tier.csv").round(0), width="stretch")
    tiers = st.multiselect("Tier", [1, 2, 3], default=[1, 2, 3], key="tier_filter")
    cols = ["dma", "tier", "decision", "oracle_decision", "pred_ncm_change_if_paused_day", "true_ncm_change_if_paused_day",
            "evidence_margin_day", "pred_commission_leakage_day", "spend_saved_day", "true_cannibalization", "ota_pressure", "organic_rank"]
    st.dataframe(pol[pol.tier.isin(tiers)][cols].round(2), width="stretch", height=320)
    st.image(str(PLOTS / "policy_by_dma.png"), width="stretch")
    st.subheader("Refutation tests")
    c1, c2 = st.columns(2)
    c1.dataframe(read_csv("placebo_in_time.csv").round(3), width="stretch")
    mc = read_csv("macro_montecarlo.csv")
    c2.dataframe(mc.groupby("method").agg(bias=("err", "mean"), rmse=("err", lambda e: float(np.sqrt((e ** 2).mean()))),
                                          ci_coverage=("covered", "mean"), n=("seed", "count")).round(2), width="stretch")
    st.image(str(PLOTS / "placebo_tests.png"), width="stretch")
    st.image(str(PLOTS / "montecarlo_error.png"), width="stretch")
    st.image(str(PLOTS / "balance_smd.png"), width="stretch")


# ------------------------------------------------------------------------------------------- tab 5
from src import flows as flows_mod  # noqa: E402


@st.cache_data
def flows_data():
    return flows_mod.load()


def metric_row(metrics, per_row=4):
    for i in range(0, len(metrics), per_row):
        chunk = metrics[i:i + per_row]
        cols = st.columns(per_row)
        for c, m in zip(cols, chunk):
            c.metric(m["label"], m["value"], help=m.get("help"))


def draw_chart(ch):
    if not ch:
        return
    if ch["kind"] == "line":
        df = pd.DataFrame(ch["series"], index=ch["x"])
        st.markdown(f"**{ch['title']}** - pause begins day {ch['cut']}")
        st.line_chart(df, height=240)
    elif ch["kind"] == "bar":
        st.markdown(f"**{ch['title']}**")
        st.bar_chart(pd.Series(ch["y"], index=ch["x"]), height=220)
    elif ch["kind"] == "hist":
        st.markdown("**Placebo-in-space distribution of effects on fake holdouts** (real estimates: "
                    + ", ".join(f"{k} {v:.2f}" for k, v in ch["marks"].items()) + ")")
        st.bar_chart(pd.Series(ch["y"], index=ch["x"]), height=200)


FLAG = {"pass": "PASS", "warn": "CAUTION", "fail": "FAIL", None: ""}

with tab5:
    D = flows_data()
    st.subheader("Programme scorecard")
    st.caption("Overall metrics for the whole engine (macro lift, economics, validity, CATE, policy). Every number is read from `outputs/metrics/flows.json`, "
               "the same file served by `GET /flows`.")
    for g in D["programme"]["groups"]:
        st.markdown(f"**{g['group']}**")
        metric_row(g["metrics"], per_row=max(len(g["metrics"]), 1) if len(g["metrics"]) <= 5 else 5)
    st.markdown("**Flow roll-up** (4 flow types x 2 examples)")
    st.dataframe(pd.DataFrame(D["programme"]["rollup"]), width="stretch", hide_index=True)

    st.subheader("End-to-end flow")
    fid = st.radio("Choose a flow", D["order"], horizontal=True, key="flow_sel",
                   format_func=lambda i: D["flows"][i]["short"])
    F = D["flows"][fid]
    st.markdown(f"### {F['id']}: {F['title']}")
    st.caption(f"Flow type {F['type']} - {F['type_title']}: {F['question']}")
    st.write(F["story"])
    last = F["stages"][-1].get("flag")
    (st.success if last == "pass" else st.error if last == "fail" else st.warning)(f"Flow verdict: **{F['verdict']}**")
    st.caption(F["action"])
    st.markdown("**Flow metrics**")
    metric_row(F["flow_metrics"], per_row=min(len(F["flow_metrics"]), 6))
    for stg in F["stages"]:
        badge = f"  `{FLAG[stg.get('flag')]}`" if stg.get("flag") else ""
        st.markdown(f"#### {stg['title']}{badge}")
        metric_row(stg["metrics"], per_row=4)
        if stg.get("note"):
            st.caption(stg["note"])
        draw_chart(stg.get("chart"))
