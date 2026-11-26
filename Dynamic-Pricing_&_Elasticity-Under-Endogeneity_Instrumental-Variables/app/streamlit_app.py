"""Grand Horizon Hotels - Dynamic Pricing & Elasticity under Endogeneity (Streamlit).

Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.optimization.flows import run_all_flows  # noqa: E402
from src.optimization.revenue_optimizer import (build_profile_df, compare_policies, crm_segments, rack_rate,  # noqa: E402
                                                revpar_curve, truth_curve)
from src.utils.gpu_utils import cuda_available, load_config  # noqa: E402
from src.utils.metrics import qini_curve  # noqa: E402
from src.utils.registry import OUT, load_benchmark, load_bookings, load_json, load_models  # noqa: E402
from src.utils.structural import expected_occupancy  # noqa: E402

st.set_page_config(page_title="Grand Horizon - Causal Pricing Engine", page_icon="🏨", layout="wide")

BLUE, ORANGE, TEAL, PURPLE, ROSE, INK = "#2A6FDB", "#E8833A", "#1F9D8A", "#7B5CD6", "#C8466B", "#1F2430"
FAMILY_COLOR = {"naive": ROSE, "iv": BLUE, "deep_iv": PURPLE, "cate_meta": TEAL, "cate_deep": ORANGE}
_cache = getattr(st, "cache_resource", None) or st.experimental_singleton


@_cache
def get_assets():
    return load_config(), load_models()


@_cache
def get_bookings() -> pd.DataFrame:
    return load_bookings()


cfg, models = get_assets()


@_cache
def get_flows():
    return run_all_flows(models, cfg)
if not models:
    st.error("No trained models found in outputs/models. Run `make train` (or `bash run_all.sh`) first.")
    st.stop()

PRICE_MODELS = [m for m in ["Naive OLS", "Naive XGBoost", "2SLS", "DML-IV", "DR-IV (LinearDRIV)", "DeepIV (PyTorch)"] if m in models]
COLORS = {"Naive OLS": ROSE, "Naive XGBoost": "#E58AA3", "2SLS": BLUE, "DML-IV": TEAL, "DR-IV (LinearDRIV)": ORANGE,
          "DeepIV (PyTorch)": PURPLE, "Ground truth": INK}

# ----------------------------------------------------------------------------------------------------------- sidebar
st.sidebar.title("🏨 Grand Horizon")
st.sidebar.caption("IV + CATE dynamic pricing engine")
scn_key = st.sidebar.radio("Scenario", list(cfg["scenarios"].keys()),
                           format_func=lambda k: f"{k}: {cfg['scenarios'][k]['name']}")
scn = cfg["scenarios"][scn_key]
st.sidebar.info(scn["description"])
prof, inst, u_shock = dict(scn["profile"]), dict(scn["instruments"]), float(scn["demand_shock_u"])

st.sidebar.markdown("**Booking context**")
if scn_key == "D":
    k = "d_"
    prof["segment"] = st.sidebar.selectbox("Guest segment", ["leisure", "corporate", "group"], key=k + "seg")
    prof["property_tier"] = st.sidebar.selectbox("Property tier", ["luxury", "urban_business", "resort"], index=1, key=k + "tier")
    prof["lead_time"] = st.sidebar.slider("Lead time (days)", 0, 365, int(prof["lead_time"]), key=k + "lead")
    prof["los"] = st.sidebar.slider("Length of stay", 1, 14, int(prof["los"]), key=k + "los")
    prof["dow"] = st.sidebar.slider("Arrival day (Mon=0)", 0, 6, int(prof["dow"]), key=k + "dow")
    prof["seasonality"] = st.sidebar.slider("Seasonality index", -2.5, 2.5, float(prof["seasonality"]), 0.1, key=k + "ssn")
    prof["loyalty_tier"] = st.sidebar.slider("Loyalty tier", 0, 3, int(prof["loyalty_tier"]), key=k + "loy")
    prof["hist_adr_sensitivity"] = st.sidebar.slider("Historical ADR sensitivity (z)", -2.0, 2.5, float(prof["hist_adr_sensitivity"]), 0.1, key=k + "sens")
    prof["room_type"] = st.sidebar.selectbox("Room type", [0, 1, 2], index=1, format_func=lambda i: ["Standard", "Deluxe", "Suite"][i], key=k + "room")
    st.sidebar.markdown("**Instruments Z**")
    inst["competitor_room_outage_pct"] = st.sidebar.slider("Competitor room outage (%)", 0.0, 35.0, float(inst["competitor_room_outage_pct"]), 0.5, key=k + "z1")
    inst["airline_seat_capacity_shock"] = st.sidebar.slider("Airline seat shock (x10%)", -4.0, 4.0, float(inst["airline_seat_capacity_shock"]), 0.1, key=k + "z2")
    inst["pricing_algo_rounding_threshold"] = int(st.sidebar.checkbox("Rate-rounding trigger", bool(inst["pricing_algo_rounding_threshold"]), key=k + "z3"))
    u_shock = st.sidebar.slider("Latent demand shock U (hidden from models)", -3.0, 3.0, u_shock, 0.1, key=k + "u")
else:
    st.sidebar.json({**prof, **inst, "demand_shock_u (hidden)": u_shock})

row = build_profile_df(prof, inst, cfg=cfg)
st.sidebar.caption(f"Compute: {'CUDA GPU' if cuda_available() else 'CPU fallback'} | models loaded: {len(models)}")

st.title("Dynamic Pricing & Price Elasticity under Endogeneity")
st.caption(f"**Scenario {scn_key} - {scn['name']}**")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1 · EDA & Endogeneity Lab", "2 · Price & RevPAR Optimizer",
                                        "3 · CATE & Deep Causal Arena", "4 · Batch CRM Targeting", "5 · End-to-End Flows"])

# ----------------------------------------------------------------------------------------------------------- tab 1
with tab1:
    diag = load_json("iv_first_stage_diagnostics.json")
    if diag:
        fs, wu, co = diag["first_stage"], diag["wu_hausman_endogeneity"], diag["coefficients"]
        sg = diag.get("sargan_overidentification_heterogeneous_spec", diag["sargan_overidentification"])
        c = st.columns([1, 1, 1, 1, 1.5])
        c[0].metric("First-stage F (robust)", f"{fs['f_statistic_robust']:,.0f}", "strong (>10)" if fs["instruments_strong"] else "WEAK")
        c[1].metric("Partial R² of Z", f"{fs['partial_r2']:.3f}")
        c[2].metric("Wu-Hausman stat", f"{wu['statistic']:,.0f}", "endogeneity detected" if wu["reject_exogeneity_5pct"] else "no endogeneity")
        c[3].metric("Sargan p (heterog. spec)", f"{sg['pvalue']:.3f}", "instruments valid" if not sg["reject_validity_5pct"] else "REJECTED")
        c[4].metric("Price coef: OLS → 2SLS", f"{co['ols_price_coefficient']:+.2f} → {co['2sls_price_coefficient']:+.2f}",
                    f"truth {co['true_mean_elasticity']:+.2f}", delta_color="off")
        with st.expander("Full diagnostics JSON"):
            st.json(diag)
    st.markdown("#### Exploratory evidence")
    imgs = [("eda/01_endogeneity_trap_simpsons_paradox.png", "The endogeneity trap"),
            ("eda/02_instrument_relevance_first_stage.png", "Instrument relevance"),
            ("eda/04_confounder_instrument_correlation_heatmap.png", "Relevance and exogeneity"),
            ("evaluation/02_structural_demand_curves_ols_vs_iv.png", "Structural demand: OLS vs IV")]
    cols = st.columns(2)
    for i, (rel, cap) in enumerate(imgs):
        p = OUT / "visualizations" / rel
        if p.exists():
            # pass bytes: a str path is treated as a URL (file://) which the browser refuses to load
            cols[i % 2].image(p.read_bytes(), caption=cap, use_column_width=True)

# ----------------------------------------------------------------------------------------------------------- tab 2
with tab2:
    res = compare_policies({m: models[m] for m in PRICE_MODELS}, row, cfg, u_shock)
    pm = res["per_model"]
    naive, deep = pm.get("Naive OLS"), pm.get("DeepIV (PyTorch)")
    if naive and deep:
        c = st.columns(4)
        c[0].metric("Naive OLS recommends", f"${naive['optimal_adr']:,.0f}", f"elasticity {naive['estimated_elasticity']:+.2f}", delta_color="off")
        c[1].metric("DeepIV recommends", f"${deep['optimal_adr']:,.0f}", f"elasticity {deep['estimated_elasticity']:+.2f}", delta_color="off")
        c[2].metric("Promo depth (DeepIV)", f"{deep['optimal_promo_pct']:.0f}%", f"rack ${res['rack_rate']:,.0f}", delta_color="off")
        lift = res.get("revpar_lift_pct_causal_vs_naive")
        c[3].metric("Realised RevPAR lift vs naive", f"{lift:+.1f}%" if lift is not None else "n/a",
                    f"truth elasticity {res['true_elasticity']:+.2f}", delta_color="off")
    t = truth_curve(row, cfg, None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t["adr"], y=t["net_revpar"], name="Ground truth", line=dict(color=INK, width=4)))
    for m in ["Naive OLS", "DR-IV (LinearDRIV)", "DeepIV (PyTorch)"]:
        if m not in models:
            continue
        cv = revpar_curve(models[m], row, cfg)
        fig.add_trace(go.Scatter(x=cv["adr"], y=cv["net_revpar"], name=f"{m} (predicted)", line=dict(color=COLORS[m], width=2, dash="dash" if "Naive" in m else "solid")))
        a = pm[m]["optimal_net_price"]
        y_true = float(np.interp(a, t["adr"], t["net_revpar"]))
        fig.add_trace(go.Scatter(x=[a], y=[y_true], mode="markers+text", text=[f"${a:,.0f}"], textposition="top center",
                                 marker=dict(size=13, color=COLORS[m], line=dict(color="white", width=2)),
                                 name=f"{m} lands here (true curve)", showlegend=False))
    fig.update_layout(title="Net RevPAR vs ADR (dots mark where each policy lands on the true curve)", xaxis_title="Offered net ADR ($)",
                      yaxis_title="Expected net RevPAR ($)", height=430, margin=dict(t=60), legend=dict(orientation="h", y=-0.25))
    left, right = st.columns([3, 2])
    left.plotly_chart(fig, use_container_width=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t["adr"], y=t["occupancy"], name="Ground truth", line=dict(color=INK, width=4)))
    for m in ["Naive OLS", "DeepIV (PyTorch)"]:
        cv = revpar_curve(models[m], row, cfg)
        fig2.add_trace(go.Scatter(x=cv["adr"], y=cv["occupancy"], name=m, line=dict(color=COLORS[m], width=2, dash="dash" if "Naive" in m else "solid")))
    fig2.update_layout(title="Booking conversion vs ADR", xaxis_title="Offered net ADR ($)", yaxis_title="Expected conversion", height=430,
                       margin=dict(t=60), legend=dict(orientation="h", y=-0.25))
    right.plotly_chart(fig2, use_container_width=True)

    tbl = pd.DataFrame(pm).T[["estimated_elasticity", "optimal_adr", "optimal_promo_pct", "predicted_net_revpar",
                              "realised_net_revpar", "realised_net_revpar_at_scenario_u"]]
    tbl.columns = ["Est. elasticity", "Optimal ADR $", "Promo %", "Predicted RevPAR $", "Realised RevPAR $ (E over U)", "Realised RevPAR $ (scenario U)"]
    st.dataframe(tbl.style.format("{:,.2f}"), width=1100)
    st.caption(f"Ground-truth optimal ADR ≈ \${res['truth_optimal_adr']:,.0f}; market-following status-quo ADR ≈ \${res['status_quo_adr']:,.0f}. "
               "Realised RevPAR is evaluated on the true structural demand, averaged over the unobserved shock U (decision taken before U is known); "
               "the last column shows the value if this scenario's specific shock occurs.")
    if "profile_alt" in scn:
        st.markdown("#### Segment contrast on the same night")
        alt = build_profile_df(scn["profile_alt"], inst, cfg=cfg)
        rows = []
        for nm, r_ in [(f"{prof['segment'].title()} ({prof['property_tier']})", row), (f"{scn['profile_alt']['segment'].title()} ({scn['profile_alt']['property_tier']})", alt)]:
            rr = compare_policies({m: models[m] for m in PRICE_MODELS}, r_, cfg, u_shock)
            d = rr["per_model"]["DeepIV (PyTorch)"]
            rows.append({"Segment": nm, "True elasticity": rr["true_elasticity"], "DeepIV elasticity": d["estimated_elasticity"],
                         "Naive OLS elasticity": rr["per_model"]["Naive OLS"]["estimated_elasticity"], "Optimal ADR $": d["optimal_adr"],
                         "Optimal promo %": d["optimal_promo_pct"]})
        st.dataframe(pd.DataFrame(rows).style.format({c: "{:,.2f}" for c in ["True elasticity", "DeepIV elasticity", "Naive OLS elasticity", "Optimal ADR $", "Optimal promo %"]}), width=1100)

# ----------------------------------------------------------------------------------------------------------- tab 3
with tab3:
    bench = load_benchmark()
    if not bench.empty:
        b = bench.dropna(subset=["elasticity_rmse"]).sort_values("pehe")
        cols = ["model", "family", "pehe", "elasticity_rmse", "elasticity_bias", "auuc", "mean_elasticity", "demand_curve_rmse", "fit_seconds"]
        st.dataframe(b[cols].style.format({c: "{:.3f}" for c in cols[2:]}), width=1100, height=330)
        g1, g2 = st.columns(2)
        for col, container, title in [("pehe", g1, "PEHE (lower is better)"), ("elasticity_rmse", g2, "Elasticity RMSE (lower is better)")]:
            d = b.sort_values(col, ascending=False)
            f = go.Figure(go.Bar(x=d[col], y=d["model"], orientation="h", marker_color=[FAMILY_COLOR[x] for x in d["family"]],
                                 text=[f"{v:.3f}" for v in d[col]], textposition="outside"))
            f.update_layout(title=title, height=620, margin=dict(l=10, r=40, t=50), xaxis=dict(showgrid=True))
            container.plotly_chart(f, use_container_width=True)
    pr = OUT / "metrics" / "test_predictions.parquet"
    if pr.exists():
        preds = pd.read_parquet(pr)
        tau = preds["true_promo_cate"].to_numpy()
        fq = go.Figure()
        fo, go_ = qini_curve(tau, tau)
        fq.add_trace(go.Scatter(x=fo, y=go_, name="Oracle", line=dict(color=INK, dash="dot")))
        fq.add_trace(go.Scatter(x=[0, 1], y=[0, go_[-1]], name="Random", line=dict(color="#8A8F98")))
        for n_ in ["S-Learner", "T-Learner", "X-Learner", "R-Learner (NonParamDML)", "DR-Learner", "Causal Forest DML", "Uplift RF (causalml)", "TARNet", "DragonNet", "CEVAE"]:
            if f"tau__{n_}" in preds:
                x_, y_ = qini_curve(preds[f"tau__{n_}"].to_numpy(), tau)
                fq.add_trace(go.Scatter(x=x_[::40], y=y_[::40], name=n_, mode="lines"))
        fq.update_layout(title="Qini curves (oracle uplift on held-out reservations)", xaxis_title="fraction targeted", yaxis_title="cumulative true uplift", height=450)
        st.plotly_chart(fq, use_container_width=True)
    st.markdown("#### Live promo CATE for the selected guest profile")
    adr0 = float(rack_rate(row, cfg)[0])
    row2 = row.copy()
    row2["ln_adr"], row2["adr"] = np.log(adr0), adr0
    names = [n_ for n_ in ["S-Learner", "T-Learner", "X-Learner", "R-Learner (NonParamDML)", "DR-Learner", "Causal Forest DML", "Uplift RF (causalml)", "TARNet", "DragonNet", "CEVAE", "DeepIV (PyTorch)", "DR-IV (LinearDRIV)"] if n_ in models]
    vals = [float(models[n_].promo_cate(row2)[0]) for n_ in names]
    truth_tau = float(row["true_elasticity"].iloc[0] * np.log(1 - cfg["dgp"]["promo"]["discount"]))
    fb = go.Figure(go.Bar(x=names, y=vals, marker_color=TEAL))
    fb.add_hline(y=truth_tau, line_dash="dash", line_color=INK, annotation_text=f"truth {truth_tau:.3f}")
    fb.update_layout(title=f"Effect of a {cfg['dgp']['promo']['discount']*100:.0f}% promo on log demand (consensus median {np.median(vals):.3f})", height=420, yaxis_title="CATE (log demand)")
    st.plotly_chart(fb, use_container_width=True)

# ----------------------------------------------------------------------------------------------------------- tab 4
with tab4:
    st.markdown("Score upcoming reservations with GPU-batched CATE inference and split guests into uplift quadrants.")
    crm_models = [m for m in ["Causal Forest DML", "DragonNet", "TARNet", "R-Learner (NonParamDML)", "DeepIV (PyTorch)", "DR-IV (LinearDRIV)"] if m in models]
    c1, c2 = st.columns([1, 2])
    which = c1.selectbox("CATE model", crm_models)
    up = c2.file_uploader("Optional: upload reservations CSV (same columns as hospitality_bookings)", type=["csv"])
    n_res = c1.slider("Reservations to generate", 100, 2000, 500, 100)
    seed = c1.number_input("Random seed", 0, 9999, 7)
    if up is not None:
        try:
            batch = pd.read_csv(up)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read CSV: {exc}")
            batch = get_bookings().sample(n_res, random_state=int(seed))
    else:
        batch = get_bookings().sample(n_res, random_state=int(seed))
    batch = batch.reset_index(drop=True)
    if "ln_adr" not in batch and "adr" in batch:
        batch["ln_adr"] = np.log(batch["adr"])
    scored = crm_segments(batch, models[which], cfg)
    order = ["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"]
    counts = scored["uplift_segment"].value_counts().reindex(order).fillna(0).astype(int)
    k = st.columns(4)
    for i, s_ in enumerate(order):
        k[i].metric(s_, int(counts[s_]), f"{counts[s_] / len(scored) * 100:.0f}% of guests", delta_color="off")
    colors = {"Persuadables": TEAL, "Sure Things": BLUE, "Lost Causes": "#8A8F98", "Sleeping Dogs": ROSE}
    fs_ = go.Figure()
    for s_ in order:
        d = scored[scored["uplift_segment"] == s_]
        fs_.add_trace(go.Scatter(x=d["baseline_conv"], y=d["net_value_usd"], mode="markers", name=s_, marker=dict(color=colors[s_], size=6, opacity=0.7)))
    fs_.update_layout(title="Baseline booking probability vs net value of the promotion", xaxis_title="baseline conversion (no promo)", yaxis_title="net RevPAR change from promo ($)", height=430)
    st.plotly_chart(fs_, use_container_width=True)
    persuad = scored[scored["uplift_segment"] == "Persuadables"]
    st.success(f"Targeting only the {len(persuad)} Persuadables adds ≈ \${persuad['net_value_usd'].sum():,.0f} of expected net RevPAR; "
               f"blanket-discounting everyone changes it by \${scored['net_value_usd'].sum():,.0f}.")
    st.dataframe(scored.head(100), width=1100)
    st.download_button("⬇ Download optimal promo offers (CSV)", scored.to_csv(index=False).encode("utf-8"), "optimal_promo_offers.csv", "text/csv")


# ----------------------------------------------------------------------------------------------------------- tab 5
with tab5:
    flows = get_flows()
    st.markdown("### End-to-end decision flows")
    st.caption("Four flows, each run on several worked examples. Every stage reports its own metrics, every flow reports "
               "overall metrics, and every estimate is checked against the synthetic ground truth. "
               "These examples are fixed and do not depend on the sidebar scenario.")
    diag = load_json("iv_first_stage_diagnostics.json")
    bench = load_benchmark()
    st.markdown("#### Programme scorecard")
    sc = flows["scorecard"]
    cols = st.columns(4)
    for col, key in zip(cols, ["elasticity", "pricing", "promo", "crm"]):
        col.markdown(f"**{flows[key]['title']}**")
        for lab, val in sc[key].items():
            col.metric(lab, val)
    if diag and not bench.empty:
        b = bench.dropna(subset=["elasticity_rmse"])
        best = b.sort_values("elasticity_rmse").iloc[0]
        g = st.columns(4)
        g[0].metric("Models benchmarked", f"{len(b)}")
        g[1].metric("Best elasticity RMSE", f"{best['elasticity_rmse']:.2f}", best["model"], delta_color="off")
        g[2].metric("Naive OLS elasticity RMSE", f"{float(b[b['model'] == 'Naive OLS']['elasticity_rmse'].iloc[0]):.2f}")
        g[3].metric("First-stage F", f"{diag['first_stage']['f_statistic_robust']:,.0f}")

    keys = ["elasticity", "pricing", "promo", "crm"]
    pick = st.radio("Choose a flow", keys, format_func=lambda k: flows[k]["title"], key="flow_pick")
    flow = flows[pick]
    st.markdown(f"#### {flow['title']}")
    st.info(flow["question"])
    for ex in flow["examples"]:
        st.markdown(f"**Example: {ex['label']}**")
        scols = st.columns(len(ex["stages"]))
        for col, stg in zip(scols, ex["stages"]):
            col.markdown(f"**{stg['name']}**")
            col.caption(stg["detail"])
            for m in stg["metrics"]:
                col.metric(m["label"], m["value"])
        st.success(ex["headline"])
        st.markdown("---")
    ov = pd.DataFrame([{"Example": e["label"], **{k.replace("_", " "): (round(v, 3) if isinstance(v, float) else v) for k, v in e["overall"].items()}}
                       for e in flow["examples"]])
    st.markdown("**Overall metrics for this flow**")
    st.dataframe(ov, width=1100)
