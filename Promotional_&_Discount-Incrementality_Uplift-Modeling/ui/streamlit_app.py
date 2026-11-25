"""Interactive CATE simulator.   Run: streamlit run ui/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.common import FEATURES, LOYALTY_LEVELS, load_config, resolve
from src.serving import PRESETS, Scorer
from ui.flows_ui import render_flows_tab

st.set_page_config(page_title="Hospitality Uplift CATE Simulator", page_icon="🏨", layout="wide")
cfg = load_config()
logs, plots = resolve(cfg["paths"]["logs"]), resolve(cfg["paths"]["plots"])


@st.cache_resource
def get_scorer():
    return Scorer(cfg)


@st.cache_data
def load_logs():
    return (pd.read_csv(logs / "benchmark_margin.csv"), pd.read_parquet(logs / "test_set.parquet"),
            pd.read_parquet(logs / "test_predictions.parquet"))


DEFAULTS = {"hist_booking_freq": 3.0, "days_since_last_stay": 170, "loyalty_tier": "Silver", "hist_price_sensitivity": 0.5,
            "avg_lead_time": 30, "avg_adr": 150, "weekend_stay_ratio": 0.5, "cancellation_rate": 0.1,
            "email_open_rate_90d": 0.3, "points_balance": 4000}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("preset", None)


def apply_preset(name):
    st.session_state["preset"] = name
    for k in DEFAULTS:
        st.session_state[k] = PRESETS[name][k]


scorer = get_scorer()
st.title("🏨 Hospitality Promotional Incrementality — CATE Simulator")
st.caption("Who should receive the offer? Persuadables yes; Always-Takers, Lost Causes and Sleeping Dogs no.")
tab1, tab2, tab3, tab4 = st.tabs(["Guest CATE Simulator", "Model Benchmark & Diagnostics", "Campaign ROI & Policy Simulator",
                                  "End-to-End Flows"])

# ------------------------------------------------------------------ Tab 1
with tab1:
    st.subheader("One-click guest presets")
    cols = st.columns(4)
    labels = ["Example 1", "Example 2", "Example 3", "Example 4"]
    for c, (name, p), lab in zip(cols, PRESETS.items(), labels):
        c.button(f"{lab}: {name}", key=f"btn_{name}", on_click=apply_preset, args=(name,), width="stretch",
                 help=f"Expected persona: {p['persona']}")
    if st.session_state["preset"]:
        st.info(f"Loaded preset **{st.session_state['preset']}** (designed persona: {PRESETS[st.session_state['preset']]['persona']})")

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("**Guest features**")
        st.slider("Historical bookings / year", 0.0, 40.0, key="hist_booking_freq", step=0.5)
        st.slider("Days since last stay", 0, 1500, key="days_since_last_stay")
        st.selectbox("Loyalty tier", LOYALTY_LEVELS, key="loyalty_tier")
        st.slider("Historical price sensitivity", 0.0, 1.0, key="hist_price_sensitivity", step=0.01)
        st.slider("Average lead time (days)", 0, 180, key="avg_lead_time")
        st.slider("Average daily rate ($)", 60, 600, key="avg_adr")
        st.slider("Weekend stay ratio", 0.0, 1.0, key="weekend_stay_ratio", step=0.01)
        st.slider("Cancellation rate", 0.0, 0.9, key="cancellation_rate", step=0.01)
        st.slider("Email open rate (90d)", 0.0, 1.0, key="email_open_rate_90d", step=0.01)
        st.slider("Loyalty points balance", 0, 150000, key="points_balance", step=500)
        est = st.selectbox("CATE estimator", list(scorer.models), index=list(scorer.models).index(scorer.best))
    guest = {k: st.session_state[k] for k in FEATURES}
    res = scorer.score([guest], est, explain=True)[0]
    with right:
        badge = {"Persuadable": "🟢", "Always-Taker": "🟡", "Lost Cause": "⚪", "Sleeping Dog": "🔴"}[res["quadrant"]]
        st.markdown(f"### {badge} Persona quadrant: **{res['quadrant']}**")
        d = res["decision"]
        (st.success if d.startswith("SEND") else st.error)(f"CRM decision: **{d}**")
        m1, m2, m3 = st.columns(3)
        m1.metric("τ̂ margin (20% off)", f"${res['tau_margin_hat']:.2f}")
        m2.metric("τ̂ conversion", f"{100 * res['tau_conv_hat']:+.1f} pp")
        m3.metric("Baseline conversion", f"{100 * res['baseline_conv_prob']:.0f}%")
        if res["ci95_low"] is not None:
            st.write(f"**95% CI for τ̂ margin:** [{res['ci95_low']:.2f}, {res['ci95_high']:.2f}]  "
                     f"{'(excludes 0)' if res['ci95_low'] > 0 or res['ci95_high'] < 0 else '(includes 0)'}")
        st.caption(f"Modelled τ̂ margin for 2,000-point offer: ${res['tau_margin_points_hat']:.2f} · inference {res['latency_ms']:.1f} ms")
        if "shap" in res:
            s = res["shap"]
            vals = pd.Series(s["values"]).sort_values(key=abs)
            fig, ax = plt.subplots(figsize=(6.5, 4))
            ax.barh([f"{k} = {s['data'][k]:.2f}" for k in vals.index], vals.values,
                    color=["#009E73" if v >= 0 else "#D55E00" for v in vals.values])
            ax.axvline(0, color="black", lw=0.8)
            ax.set_title(f"Local SHAP (surrogate of {est}) — base ${s['base_value']:.1f}", fontsize=10)
            ax.set_xlabel("Contribution to τ̂ margin ($)")
            st.pyplot(fig)

# ------------------------------------------------------------------ Tab 2
with tab2:
    bench, te, preds = load_logs()
    st.subheader("Algorithm benchmark (test set, 25% hold-out)")
    show = bench[["model", "qini_norm_ipw", "auuc_ipw", "pehe", "uplift@20_ipw", "latency_ms_per_1k", "fit_seconds", "gpu", "has_ci"]].rename(
        columns={"qini_norm_ipw": "Norm. Qini", "auuc_ipw": "AUUC", "pehe": "PEHE", "uplift@20_ipw": "Uplift@20% ($)",
                 "latency_ms_per_1k": "Latency ms/1k", "fit_seconds": "Fit s"})
    st.dataframe(show.style.format({"Norm. Qini": "{:.3f}", "AUUC": "{:.3f}", "PEHE": "{:.2f}", "Uplift@20% ($)": "{:.2f}",
                                    "Latency ms/1k": "{:.1f}", "Fit s": "{:.1f}"}).background_gradient(subset=["Norm. Qini"], cmap="Greens")
                 .background_gradient(subset=["PEHE"], cmap="Reds"), width="stretch", hide_index=True)
    abl = logs / "benchmark_base_learner_ablation.csv"
    if abl.exists():
        with st.expander("Base-learner ablation (XGBoost-GPU / LightGBM-CPU / CatBoost-GPU)"):
            st.dataframe(pd.read_csv(abl), width="stretch", hide_index=True)
    for f, cap in [("02_multi_model_qini_curves.png", "Qini curves"), ("03_uplift_by_decile.png", "Uplift by CATE decile"),
                   ("01_propensity_overlap_balance.png", "Propensity overlap & covariate balance")]:
        if (plots / f).exists():
            st.image(str(plots / f), caption=cap, width="stretch")

# ------------------------------------------------------------------ Tab 3
with tab3:
    bench, te, preds = load_logs()
    st.subheader("Campaign ROI & policy simulator")
    c = st.columns(4)
    disc = c[0].slider("Offer discount %", 5, 40, int(100 * cfg["offer"]["discount_pct"])) / 100
    pts_cost = c[1].slider("Loyalty points cost ($)", 5.0, 60.0, 20.0, 1.0)
    db = c[2].number_input("CRM database size", 10_000, 50_000_000, cfg["offer"]["crm_database_size"], step=100_000)
    thr = c[3].slider("CATE targeting threshold ($)", -5.0, 40.0, float(cfg["offer"]["cate_threshold"]), 0.5)
    o = cfg["offer"]; m = o["margin_rate"]
    est3 = st.selectbox("Estimator driving the targeting policy", [n for n in bench["model"] if n in preds.columns],
                        index=[n for n in bench["model"] if n in preds.columns].index(scorer.best), key="roi_est")
    tau = preds[est3].values
    R, d0, cc = te["stay_value"].values, o["discount_pct"], o["comm_cost"]
    dconv = te["tau_conv_true"].values
    # ground truth under a new discount d:  tau(d) = dConv*R*(m-d) - p0*R*d   (p0 recovered exactly from the DGP at d0)
    p0 = (dconv * R * (m - d0) - te["tau_margin_true"].values) / (R * d0)
    tau_true_new = dconv * R * (m - disc) - p0 * R * disc
    # model side: re-price the stored CATE by -R*(d-d0)*p1_hat using the estimated treated-conversion probability
    dec = pd.read_parquet(logs / "test_decisions.parquet")
    p1_hat = np.clip(dec["p0_hat"].values + preds["conv_companion"].values, 0, 1)
    tau_hat_new = tau - R * (disc - d0) * p1_hat
    # modelled points arm: conv lift = ratio * discount lift, liability P paid on every booking by a mailed guest
    P = pts_cost
    tau_pts_true = o["points_conv_effect_ratio"] * dconv * (R * m - P) - p0 * P
    tau_pts_hat = o["points_conv_effect_ratio"] * preds["conv_companion"].values * (R * m - P) - dec["p0_hat"].values * P
    use_pts = (tau_pts_hat > tau_hat_new) & (tau_pts_hat > thr + cc)
    use_disc = (~use_pts) & (tau_hat_new > thr + cc)
    scale = db / len(te)
    rows = []
    for name, mask, mask_p in [("Mass Blast (discount)", np.ones(len(te), bool), np.zeros(len(te), bool)),
                               ("No Promo", np.zeros(len(te), bool), np.zeros(len(te), bool)),
                               ("Causal Uplift (discount only)", tau_hat_new > thr + cc, np.zeros(len(te), bool)),
                               ("Causal Uplift (discount / points routing)", use_disc, use_pts)]:
        net = (tau_true_new[mask].sum() + tau_pts_true[mask_p].sum() - cc * (mask.sum() + mask_p.sum())) * scale
        cann = (te["Y0_conv"].values[mask] * R[mask] * disc).sum() * scale
        rows.append({"Policy": name, "Targeted %": 100 * (mask | mask_p).mean(), "Net incremental profit ($)": net,
                     "Discount cannibalization ($)": cann})
    tab = pd.DataFrame(rows)
    k = st.columns(4)
    for col, r in zip(k, rows):
        col.metric(r["Policy"], f"${r['Net incremental profit ($)']:,.0f}", f"{r['Targeted %']:.0f}% targeted", delta_color="off")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].bar([x.replace(" (", chr(10) + "(") for x in tab["Policy"]], tab["Net incremental profit ($)"], color=["#D55E00" if v < 0 else "#009E73" for v in tab["Net incremental profit ($)"]])
    ax[0].axhline(0, color="black", lw=0.8); ax[0].set_title("Net incremental profit")
    ax[1].bar([x.replace(" (", chr(10) + "(") for x in tab["Policy"]], tab["Discount cannibalization ($)"], color="#E69F00"); ax[1].set_title("Discount cannibalization")
    st.pyplot(fig)
    st.dataframe(tab.style.format({"Targeted %": "{:.1f}", "Net incremental profit ($)": "${:,.0f}", "Discount cannibalization ($)": "${:,.0f}"}),
                 hide_index=True, width="stretch")
    st.caption(f"Scaled from the {len(te):,}-guest test set to {db:,} guests using ground-truth counterfactuals. "
               "Points arm is modelled (conv lift = 80% of discount lift); discount arm uses DGP ground truth.")

# ------------------------------------------------------------------ Tab 4
with tab4:
    render_flows_tab(scorer, cfg)
