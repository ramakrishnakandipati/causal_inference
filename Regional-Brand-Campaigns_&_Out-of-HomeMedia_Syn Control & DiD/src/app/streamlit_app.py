"""Streamlit workbench: Regional Brand & OOH Incrementality (Aurelia Hotels & Resorts).

    streamlit run src/app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from src.evaluation import visualizations as viz  # noqa: E402
from src.models import panel_att as pa  # noqa: E402
from src.models.cate_learners import LEARNER_ORDER  # noqa: E402
from src.serving import OUTCOME_LABELS, PRESET_KEYS, STRATEGIES, Engine  # noqa: E402
from src.utils.config import load_config, read_json, resolve  # noqa: E402

st.set_page_config(page_title="Aurelia - Regional Brand & OOH Incrementality", layout="wide", initial_sidebar_state="expanded")
viz._style()
BLUE, ORANGE, AQUA, INK, INK2, GRAY = viz.BLUE, viz.ORANGE, viz.AQUA, viz.INK, viz.INK2, viz.GRAY

st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1500px}
[data-testid="stMetricValue"]{font-size:1.65rem}
[data-testid="stMetricLabel"] p{color:#666666}
.note{color:#666666;font-size:0.86rem}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models and cached results...")
def get_engine() -> Engine:
    return Engine()


@st.cache_data(show_spinner=False)
def att_cached(_eng: Engine, scenario: str, estimator: str, outcome: str, live: bool, treated: str | None, donors: tuple | None) -> dict:
    return _eng.att(scenario, estimator, outcome, live=live, treated=treated, donors=list(donors) if donors else None)


@st.cache_data(show_spinner=False)
def read_csv(name: str) -> pd.DataFrame:
    p = resolve("outputs/metrics") / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


eng = get_engine()
cfg = eng.cfg
PRESETS = {p["key"]: p for p in eng.presets()}
SHORT = {"mia_flagship": "1. Miami Flagship Resort", "las_convention": "2. Las Vegas Strip Property",
         "nyc_spring_break": "3. New York Feeder Market", "tpa_placebo": "4. Negative / Placebo Control"}
CAPTION = {"mia_flagship": "Winter Peak CTV + Airport Campaign (MIA vs. Coastal/Sunbelt Donors)",
           "las_convention": "Convention & Billboard Blitz (LAS vs. ORL, SAN, PHX, AUS)",
           "nyc_spring_break": "Spring Break Awareness Push (NYC vs. BOS, CHI, DCA)",
           "tpa_placebo": "Unexposed Market - TPA Placebo Validation"}

st.session_state.setdefault("preset", "mia_flagship")


def set_preset(k: str) -> None:
    st.session_state["preset"] = k


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Aurelia Hotels & Resorts")
    st.caption("Regional brand & out-of-home incrementality engine")
    st.markdown("**One-click examples**")
    for k in PRESET_KEYS:
        st.button(f"{SHORT[k]}", key=f"preset_btn_{k}", on_click=set_preset, args=(k,), width="stretch",
                  type="primary" if st.session_state["preset"] == k else "secondary", help=CAPTION[k])
    st.markdown("---")
    estimator = st.selectbox("Panel estimator (headline)", pa.ESTIMATOR_ORDER + ["ASCM (ElasticNet)"], index=4, key="estimator")
    outcome = st.selectbox("Outcome", list(OUTCOME_LABELS), format_func=OUTCOME_LABELS.get, key="outcome")
    st.caption("Non-room-night outcomes are precomputed for the Miami headline analysis only.")
    use_all24 = st.toggle("Use all 24 control DMAs (Miami headline)", value=False, key="all24",
                          help="Applies to the Miami preset only; the other presets use their fixed donor pools.")

preset = st.session_state["preset"]
scenario = "mia_all24" if (preset == "mia_flagship" and use_all24) else preset
if scenario != "mia_all24" and outcome != "room_nights_booked":
    scenario = "mia_all24"
    st.sidebar.info("Showing the Miami all-donor analysis because other outcomes are only computed there.")
spec = eng.scenario_spec(scenario)
st.title("Regional Brand & Out-of-Home Incrementality")
st.markdown(f"**{PRESETS[preset]['label']}**  \n<span class='note'>Treated market **{spec['treated']}**, {len(spec['donors'])} donor DMAs, "
            f"campaign days {spec['start_day']}-240, spend ${spec['total_spend_usd'] / 1e6:.1f}M. Ground truth is known because the panel is simulated.</span>",
            unsafe_allow_html=True)

res = att_cached(eng, scenario, estimator, outcome, False, None, None)
T0 = spec["start_day"] - 1
panel = eng.panel(scenario)
dates = panel[panel.dma_code == spec["treated"]].sort_values("day_index")["date"].reset_index(drop=True)
ser = pd.DataFrame(res["series"])
unit = "USD" if outcome == "incremental_net_revenue_usd" else "room nights" if outcome == "room_nights_booked" else "units"

tabs = st.tabs(["1. Executive Summary & iROAS", "2. DiD & Event-Study", "3. Synthetic Control & Donor Lab",
                "4. Geo-CATE & Meta-Learner Studio", "5. Live FastAPI & Batch Scoring", "6. End-to-End Flows"])


def fmt(x, d=0, pre="", suf=""):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{pre}{x:,.{d}f}{suf}"


def paths_figure(ser: pd.DataFrame, title: str, band: bool = True):
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    d = dates.iloc[: len(ser)]
    true_y0 = ser["y_obs"] - ser["true_att"].fillna(0) if ser["true_att"].notna().any() else None
    if true_y0 is not None:
        ax.plot(d, viz._roll(true_y0.to_numpy()), color=GRAY, lw=1.8, ls=(0, (5, 3)), label="True counterfactual")
    ax.plot(d, viz._roll(ser["y0_hat"].to_numpy()), color=BLUE, lw=2.1, label="Estimated counterfactual")
    if band and ser["att_lo"].notna().any():
        lo = ser["y_obs"] - ser["att_hi"]
        hi = ser["y_obs"] - ser["att_lo"]
        ax.fill_between(d.iloc[T0:], viz._roll(lo.to_numpy()[T0:]), viz._roll(hi.to_numpy()[T0:]), color=BLUE, alpha=0.14, lw=0)
    ax.plot(d, viz._roll(ser["y_obs"].to_numpy()), color=INK, lw=2.1, label="Observed (treated market)")
    ax.axvline(d.iloc[T0], color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_ylabel(f"{OUTCOME_LABELS[outcome]} (7-day mean)")
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


# ============================================================================ TAB 1
with tabs[0]:
    with st.container(key="tab1_kpis"):
        e = res.get("economics") or {}
        c = st.columns(5)
        c[0].metric("Incremental " + unit, fmt(res["att_total"]), help="Sum of daily ATT over the 60 campaign days")
        c[1].metric("Avg. daily lift", fmt(res["att_mean"], 1), delta=f"95% CI {fmt(res['ci_lo'])} to {fmt(res['ci_hi'])}" if res["ci_lo"] is not None else None, delta_color="off")
        c[2].metric("CPIRN", fmt(e.get("cpirn_usd"), 0, "$") if e else "n/a", help="Cost per incremental room night")
        c[3].metric("iROAS", fmt(e.get("iroas"), 2, suf="x") if e else "n/a",
                    delta=f"CI {fmt(e.get('iroas_lo'), 2)} to {fmt(e.get('iroas_hi'), 2)}" if e and e.get("iroas_lo") is not None else None, delta_color="off")
        c[4].metric("Placebo p-value", fmt(res.get("p_placebo"), 3) if res.get("p_placebo") is not None else "n/a",
                    help="Exact in-space permutation p-value. Smallest attainable = 1 / (donors + 1).")
        if res.get("true_att_mean") is not None:
            st.markdown(f"<span class='note'>Simulation truth: mean daily effect <b>{res['true_att_mean']:,.1f}</b> "
                        f"(estimator bias {res['att_bias']:+,.1f}).  Estimator: <b>{estimator}</b>  |  source: {res.get('source', '?')}</span>", unsafe_allow_html=True)
    left, right = st.columns([1.55, 1])
    with left:
        with st.container(key="tab1_chart"):
            st.pyplot(paths_figure(ser, f"{spec['treated']}: observed vs. counterfactual"), width="stretch")
    with right:
        with st.container(key="tab1_iroas_calc"):
            st.markdown("#### iROAS calculator")
            spend = st.number_input("Campaign spend (USD)", 0.0, 50_000_000.0, float(spec["total_spend_usd"] or 3_000_000), 100_000.0, key="calc_spend")
            default_nrpr = (e["incremental_net_revenue_usd"] / e["incremental_room_nights"]) if e and e.get("incremental_room_nights", 0) > 0 else eng._mia_nrpr()
            nrpr = st.number_input("Net revenue per incremental room night (USD)", 50.0, 3000.0, float(round(default_nrpr, 2)), 10.0, key="calc_nrpr",
                                   help="ADR x (1 + ancillary) x (1 - distribution cost)")
            lo, hi = (res["ci_lo"], res["ci_hi"]) if res["ci_lo"] is not None else (res["att_mean"], res["att_mean"])
            T1 = 240 - T0
            inc, inc_lo, inc_hi = res["att_mean"] * T1, lo * T1, hi * T1
            ir = inc * nrpr / spend if spend > 0 else float("nan")
            st.metric("iROAS (point)", fmt(ir, 2, suf="x"))
            st.markdown(f"Range from the estimator's 95% interval: **{fmt(inc_lo * nrpr / spend if spend else None, 2, suf='x')}** to "
                        f"**{fmt(inc_hi * nrpr / spend if spend else None, 2, suf='x')}**")
            st.markdown(f"Break-even incremental room nights: **{fmt(spend / nrpr if nrpr else None)}**  \n"
                        f"CPIRN: **{fmt(spend / inc if inc > 0 else None, 0, '$')}**")
    econ = read_csv("economics.csv")
    if len(econ):
        e1 = econ[econ.scenario == scenario].copy()
        if len(e1):
            st.markdown("#### Economics by estimator")
            show = e1[["estimator", "incremental_room_nights", "inc_rn_lo", "inc_rn_hi", "cpirn_usd", "iroas", "iroas_lo", "iroas_hi", "true_iroas"]].round(2)
            st.dataframe(show, hide_index=True, width="stretch")

# ============================================================================ TAB 2
with tabs[1]:
    es = read_csv("event_study.csv")
    es = es[(es.scenario == scenario) & (es.outcome == outcome)]
    meta = read_csv("event_study_pretrend.csv")
    meta = meta[(meta.scenario == scenario) & (meta.outcome == outcome)]
    l, r = st.columns(2)
    with l:
        with st.container(key="tab2_trends"):
            wide = panel.pivot(index="day_index", columns="dma_code", values=outcome)
            idx = wide / wide.iloc[:T0].mean() * 100
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            pool = spec["donors"]
            ax.plot(dates, viz._roll(idx[pool].mean(axis=1).to_numpy()), color=GRAY, lw=2, label=f"Average of {len(pool)} donor DMAs")
            ax.plot(dates, viz._roll(idx[spec["treated"]].to_numpy()), color=INK, lw=2, label=f"{spec['treated']} (observed)")
            ax.axvline(dates.iloc[T0], color=INK2, lw=1, ls=(0, (4, 3)))
            ax.set_title("Parallel-trends check: indexed outcome")
            ax.set_ylabel("Index (pre-period mean = 100), 7-day mean")
            ax.legend(loc="upper left", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
    with r:
        with st.container(key="tab2_event_study"):
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            if len(es):
                k = es["k"].to_numpy()
                ax.axhline(0, color=INK2, lw=1)
                if es["se_placebo"].notna().any():
                    ax.fill_between(k, es["beta"] - 1.96 * es["se_placebo"], es["beta"] + 1.96 * es["se_placebo"], color=ORANGE, alpha=0.13, lw=0)
                ax.plot(k, es["beta"], color=ORANGE, marker="o", ms=3.5, label="Event-study coefficient (placebo 95% band)")
                ax.plot(k[k >= 0], es["true_att"][k >= 0], color=INK, ls=(0, (5, 3)), lw=1.6, label="True effect")
                ax.axvline(0, color=INK2, lw=1, ls=(0, (4, 3)))
                ax.set_xlabel("Days relative to launch (k); reference k = -1")
                ax.set_ylabel("Effect vs. reference")
                ax.legend(loc="upper left", fontsize=8)
            ax.set_title("Dynamic DiD: leads (-14..-2) and lags (0..59)")
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
    if len(meta):
        m = meta.iloc[0]
        c = st.columns(4)
        c[0].metric("Pre-trend RMS", fmt(m.pretrend_rms, 1), help="RMS of lead coefficients k=-14..-2")
        c[1].metric("Pre-trend slope / day", fmt(m.pretrend_slope, 2))
        c[2].metric("Permutation p (pre-trend)", fmt(m.p_pretrend_perm, 2), help="Share of placebo markets with at least as large a pre-trend")
        c[3].metric("Early vs late effect", f"{m.post_early:,.0f} -> {m.post_late:,.0f}", help="Mean event-study coefficient, first 20 vs last 20 campaign days")
    sm = read_csv("panel_att_summary.csv")
    sm = sm[(sm.scenario == scenario) & (sm.outcome == outcome) & (sm.estimator.isin(["TWFE DiD", "Event Study", "CS-DiD (DR)"]))]
    if len(sm):
        st.markdown("#### DiD-family estimates")
        st.dataframe(sm[["estimator", "att_mean", "se", "ci_lo", "ci_hi", "att_true_mean", "att_bias", "p_placebo"]].round(2), hide_index=True, width="stretch")
        st.markdown("<span class='note'>The DiD family assumes parallel trends. Miami's winter ramp is steeper than the average control DMA, so with an "
                    "unweighted donor pool these estimators are biased - the synthetic-control family on the next tab fixes that by re-weighting the donors.</span>", unsafe_allow_html=True)

# ============================================================================ TAB 3
with tabs[2]:
    c1, c2 = st.columns([1, 2])
    with c1:
        live = st.toggle("Re-run estimators live with a custom donor pool", value=False, key="live_donors")
        pool = spec["donors"]
        sel = st.multiselect("Donor pool", pool if not live else sorted(set(panel.dma_code) - {spec["treated"]}), default=pool, key="donor_sel") if live else pool
    sc_list = ["SCM", "ASCM (Ridge)", "SDID"]
    results = {}
    for nm in sc_list:
        results[nm] = att_cached(eng, scenario, nm, outcome, live, None, tuple(sel) if live and sel else None) if (live and len(sel) >= 2) else att_cached(eng, scenario, nm, outcome, False, None, None)
    with c2:
        with st.container(key="tab3_counterfactual"):
            fig, ax = plt.subplots(figsize=(10.5, 4.3))
            d = dates
            y0t = ser["y_obs"] - ser["true_att"].fillna(0) if ser["true_att"].notna().any() else None
            if y0t is not None:
                ax.plot(d, viz._roll(y0t.to_numpy()), color=GRAY, lw=1.8, ls=(0, (5, 3)), label="True counterfactual")
            for nm, col in zip(sc_list, [ORANGE, BLUE, AQUA]):
                s = pd.DataFrame(results[nm]["series"])
                ax.plot(d, viz._roll(s["y0_hat"].to_numpy()), color=col, lw=1.8, label=f"{nm}  (ATT {results[nm]['att_mean']:,.0f}/day)")
            ax.plot(d, viz._roll(ser["y_obs"].to_numpy()), color=INK, lw=2.1, label="Observed")
            ax.axvline(d.iloc[T0], color=INK2, lw=1, ls=(0, (4, 3)))
            ax.set_title("SCM vs. ASCM vs. SDID counterfactuals")
            ax.set_ylabel(f"{OUTCOME_LABELS[outcome]} (7-day mean)")
            ax.legend(loc="upper left", ncol=2, fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
    c = st.columns(3)
    for col, nm in zip(c, sc_list):
        r_ = results[nm]
        col.metric(nm, fmt(r_["att_mean"], 1) + " /day", delta=f"CI {fmt(r_['ci_lo'])} to {fmt(r_['ci_hi'])} | p={fmt(r_.get('p_placebo'), 3)}", delta_color="off")
    wl, pl_ = st.columns(2)
    with wl:
        with st.container(key="tab3_weights"):
            wdf = pd.DataFrame({nm: pd.Series(results[nm]["weights"]) for nm in sc_list}).fillna(0)
            wdf = wdf.loc[wdf.abs().max(axis=1).sort_values(ascending=False).index[:10]][::-1]
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            yy = np.arange(len(wdf))
            for off, nm, col in zip([-0.25, 0, 0.25], sc_list, [ORANGE, BLUE, AQUA]):
                ax.barh(yy + off, wdf[nm], height=0.24, color=col, label=nm, zorder=2)
            ax.axvline(0, color=INK2, lw=1)
            ax.set_yticks(yy, wdf.index)
            ax.set_xlabel("Donor weight")
            ax.set_title("Donor weights")
            ax.legend(fontsize=8, loc="lower right")
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
    with pl_:
        with st.container(key="tab3_placebo"):
            est_pl = results["ASCM (Ridge)"].get("placebo")
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            if est_pl:
                t = pd.DataFrame(est_pl)
                t["ratio"] = t["post_rmspe"] / t["pre_rmspe"].clip(lower=1e-9)
                t = t.sort_values("ratio", ascending=False).reset_index(drop=True)
                ax.bar(range(len(t)), t["ratio"], color=[BLUE if u == "TREATED" else "#b9b9b9" for u in t["unit"]], zorder=2)
                ax.set_xticks(range(len(t)), ["TRT" if u == "TREATED" else u for u in t["unit"]], rotation=90, fontsize=7.5)
                rank = int(t.index[t.unit == "TREATED"][0]) + 1
                ax.set_title(f"In-space placebo: treated ranks {rank} of {len(t)}  (exact p = {rank / len(t):.3f})")
            ax.set_ylabel("Post / pre RMSPE ratio")
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
    if scenario == "mia_all24":
        l2, r2 = st.columns(2)
        it = read_csv("in_time_placebo.csv")
        lodo = read_csv("lodo_sensitivity.csv")
        with l2:
            st.markdown("#### In-time placebo (campaign back-dated to day 120)")
            st.dataframe(it.round(2), hide_index=True, width="stretch")
        with r2:
            st.markdown("#### Leave-one-donor-out / spillover sensitivity")
            if len(lodo):
                st.dataframe(lodo.pivot(index="dropped", columns="estimator", values="att_mean").round(1).reset_index(), hide_index=True, width="stretch")

# ============================================================================ TAB 4
with tabs[3]:
    bench = read_csv("cate_benchmark.csv")
    with st.container(key="tab4_benchmark"):
        st.markdown("#### 10-model Geo-CATE benchmark (Miami campaign window, mean over noise draws)")
        if len(bench):
            b = bench.copy()
            lead = b[~b.reference].sort_values("pehe")
            fig, ax = plt.subplots(figsize=(10.5, 4.3))
            yy = np.arange(len(lead))[::-1]
            for yi, r_ in zip(yy, lead.itertuples()):
                ax.barh(yi, r_.pehe, height=0.55, color=viz.LEARNER_FAMILY[r_.learner], zorder=2)
                ax.text(r_.pehe + 1, yi, f"{r_.pehe:.1f}", va="center", fontsize=8.5)
            for r_ in b[b.reference].itertuples():
                ax.axvline(r_.pehe, color=INK, lw=1.2, ls=(0, (5, 3)) if "Constant" in r_.learner else (0, (1, 2)))
            ax.set_yticks(yy, list(lead["learner"]))
            ax.set_xlabel("PEHE (room nights)  |  dashed = constant-ATE baseline, dotted = oracle X-regression")
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            st.dataframe(b[["learner", "pehe", "pehe_sd", "att_bias", "corr", "qini_oracle", "seconds", "gpu", "backend"]].round(2), hide_index=True, width="stretch")
    st.markdown("---")
    with st.container(key="tab4_whatif"):
        st.markdown("#### What-if market simulator")
        st.caption("Set market conditions for a single campaign day; every learner scores the conditional lift.")
        win = eng.window
        feats = eng.feature_names
        a1, a2, a3 = st.columns(3)
        with a1:
            weekend = st.checkbox("Weekend", value=True, key="wi_wk")
            holiday = st.checkbox("Holiday window", value=False, key="wi_hol")
            dsl = st.slider("Days since launch", 0, 60, 20, key="wi_dsl")
        with a2:
            weather = st.slider("Weather severity index", 0, 100, int(win["weather_severity_index"].median()), key="wi_w")
            cadr = st.slider("Competitor ADR (USD)", 200, 600, int(win["competitor_adr_usd"].median()), key="wi_c")
        with a3:
            flight = st.slider("Flight-search index", 60, 220, int(win["flight_search_index"].median()), key="wi_f")
            ota = st.slider("OTA promo pressure", 0, 100, int(win["ota_promo_pressure_index"].median()), key="wi_o")
        row = {f: float(win[f].median()) for f in feats}
        row.update({"is_weekend": float(weekend), "is_holiday_window": float(holiday), "days_since_launch": float(dsl),
                    "weather_severity_index": float(weather), "competitor_adr_usd": float(cadr), "flight_search_index": float(flight),
                    "ota_promo_pressure_index": float(ota), "dow": 5.0 if weekend else 2.0})
        out = []
        for ln in LEARNER_ORDER:
            if ln in eng.cate_models:
                o = eng.cate([row], ln)
                out.append({"learner": ln, "lift_pct": o["lift_pct"][0], "incremental_room_nights": o["incremental_room_nights"][0]})
        wf = pd.DataFrame(out)
        if len(wf):
            fig, ax = plt.subplots(figsize=(10.5, 3.6))
            ax.bar(range(len(wf)), wf["incremental_room_nights"], color=[viz.LEARNER_FAMILY[k] for k in wf["learner"]], width=0.62, zorder=2)
            ax.axhline(wf["incremental_room_nights"].median(), color=INK, lw=1.2, ls=(0, (5, 3)))
            ax.set_xticks(range(len(wf)), wf["learner"], rotation=25, ha="right", fontsize=8)
            ax.set_ylabel("Incremental room nights / day")
            ax.set_title(f"Predicted effect under these conditions (median across learners {wf['incremental_room_nights'].median():.0f} room nights, dashed)")
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
    st.markdown("---")
    with st.container(key="tab4_pacing"):
        st.markdown("#### Flighting and budget pacing")
        if eng.curve:
            p1, p2 = st.columns([1, 2])
            with p1:
                bud = st.slider("Budget (USD M)", 0.5, 8.0, 3.0, 0.25, key="pace_budget") * 1e6
                strat = st.selectbox("Pacing strategy", STRATEGIES, index=STRATEGIES.index("even"), key="pace_strategy")
                wsev = st.slider("Weather severity override (0 = leave as observed)", 0, 100, 0, key="pace_w")
                ov = {"weather_severity_index": float(wsev)} if wsev > 0 else {}
                sim = eng.simulate(bud, strat, overrides=ov)
                st.metric("Projected incremental room nights", fmt(sim["totals"]["incremental_room_nights"]))
                st.metric("CPIRN / iROAS", f"${sim['totals']['cpirn_usd']:,.0f} / {sim['totals']['iroas']:.2f}x")
                st.caption(f"Highest *projected* strategy: **{sim['best_strategy']}** (projection only - see the validation table). "
                           f"Response curve fitted on estimated effects: adstock theta={eng.curve.theta:.2f}, half-saturation K={eng.curve.K:.0f}k, R2={eng.curve.r2:.2f}.")
                if sim["curve_reliability"]["weak"]:
                    st.warning(sim["curve_reliability"]["note"])
            with p2:
                cmp_ = pd.DataFrame(sim["strategy_comparison"])
                fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1, 1]})
                d_ = sim["daily"]
                axs[0].bar(d_["day"], np.array(d_["spend_usd"]) / 1000, color="#b9d6f8", width=0.8, zorder=2, label="spend ($k)")
                ax2 = axs[0].twinx()
                ax2.grid(False)
                ax2.plot(d_["day"], d_["incremental_room_nights"], color=BLUE, lw=2)
                axs[0].set_xlabel("Campaign day")
                axs[0].set_ylabel("Spend ($k / day)")
                ax2.set_ylabel("Incremental room nights / day", color=INK2)
                axs[0].set_title(f"{sim['strategy']} plan")
                cs = cmp_.sort_values("incremental_room_nights")
                axs[1].barh(cs["strategy"], cs["incremental_room_nights"], color=[BLUE if s == sim["strategy"] else "#b9b9b9" for s in cs["strategy"]], zorder=2)
                axs[1].set_xlabel("Incremental room nights (60 days)")
                axs[1].set_title("Strategy comparison")
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
            pv = read_csv("pacing_validation.csv")
            if len(pv):
                st.markdown("**Does the projection survive contact with the truth?** Each plan scored with the fitted curve vs. the simulation's true response "
                            "(change vs. even pacing, same budget):")
                show = pv[["strategy", "projected_vs_even_pct", "true_vs_even_pct"]].rename(columns={
                    "projected_vs_even_pct": "Projected vs even (%)", "true_vs_even_pct": "True vs even (%)"}).round(1)
                st.dataframe(show, hide_index=True, width="stretch")
                st.markdown("<span class='note'>The fitted curve correctly flags back-loading as costly, but it cannot rank the reasonable plans: gains it projects "
                            "for front-loading or the optimiser are not borne out by the truth. Prefer even, weekend- or CATE-weighted pacing.</span>", unsafe_allow_html=True)

# ============================================================================ TAB 5
with tabs[4]:
    with st.container(key="tab5_api"):
        st.markdown("#### Live FastAPI integration")
        url = st.text_input("API base URL", f"http://{cfg['api']['host']}:{cfg['api']['port']}", key="api_url")
        try:
            h = requests.get(url + "/health", timeout=4).json()
            st.success(f"API online | learners loaded: {len(h['cate_models'])} | best learner: {h['best_learner']} | serving device: {h['serving_device']}")
            online = True
        except Exception as exc:
            st.error(f"API not reachable at {url} ({type(exc).__name__}). Start it with: uvicorn src.api.main:app --port {cfg['api']['port']}")
            online = False
        if online:
            q1, q2 = st.columns(2)
            with q1:
                st.markdown("**POST /predict/att**")
                req = {"scenario": scenario, "estimator": estimator, "outcome": outcome, "include_series": False}
                st.code(str(req), language="json")
                try:
                    rr = requests.post(url + "/predict/att", json=req, timeout=30)
                    st.json({k: v for k, v in rr.json().items() if k in ("estimator", "scenario", "att_mean", "ci_lo", "ci_hi", "att_total", "p_placebo", "economics", "source")} if rr.ok else rr.text)
                except Exception as exc:
                    st.warning(str(exc))
            with q2:
                st.markdown("**POST /simulate/campaign**")
                req2 = {"budget_usd": 3_000_000, "strategy": "optimized"}
                st.code(str(req2), language="json")
                try:
                    rr = requests.post(url + "/simulate/campaign", json=req2, timeout=30)
                    j = rr.json()
                    st.json({"best_strategy": j.get("best_strategy"), "totals": j.get("totals")} if rr.ok else rr.text)
                except Exception as exc:
                    st.warning(str(exc))
    st.markdown("---")
    with st.container(key="tab5_batch"):
        st.markdown("#### Batch scoring via /predict/cate")
        st.caption("Score many market-days at once. Upload a CSV with any of the feature columns, or use the Miami campaign-window sample.")
        up = st.file_uploader("CSV of market-days", type=["csv"], key="batch_upload")
        sample = eng.window_features().head(60)
        df_in = pd.read_csv(up) if up else sample
        st.dataframe(df_in.head(8), hide_index=True, width="stretch")
        learner = st.selectbox("Learner", [k for k in LEARNER_ORDER if k in eng.cate_models], index=[k for k in LEARNER_ORDER if k in eng.cate_models].index(eng.best_learner), key="batch_learner")
        if st.button("Score batch through the API", key="batch_go") and online:
            body = {"rows": df_in[[c for c in df_in.columns if c in eng.feature_names]].head(5000).to_dict("records"), "learner": learner}
            rr = requests.post(url + "/predict/cate", json=body, timeout=60)
            if rr.ok:
                j = rr.json()
                scored = df_in.head(len(j["lift_pct"])).copy()
                scored["lift_pct"] = j["lift_pct"]
                scored["incremental_room_nights"] = j["incremental_room_nights"]
                if j.get("ci"):
                    scored["ci_lo_rn"], scored["ci_hi_rn"] = j["ci"]["incremental_room_nights_lo"], j["ci"]["incremental_room_nights_hi"]
                st.success(f"{len(scored)} market-days scored by {j['learner']} | mean lift {j['mean_lift_pct']:.2f}%")
                st.dataframe(scored.round(2), hide_index=True, width="stretch")
                st.download_button("Download scored CSV", scored.to_csv(index=False), "scored_market_days.csv", key="batch_dl")
            else:
                st.error(rr.text)

# ============================================================================ TAB 6
with tabs[5]:
    from src import flows as fl  # noqa: E402
    from src.app import flows_ui  # noqa: E402

    st.markdown("#### End-to-end flows")
    st.caption("Four flow types, two worked examples each. Every stage shows its own metrics; the scorecard on top rolls them up. "
               "All numbers come from the same engine as the API; the 'truth' columns exist only because the panel is simulated.")
    labels = {f["id"]: f["short"] for f in fl.FLOWS}
    pick = st.radio("Choose a flow", list(labels), format_func=labels.get, horizontal=True, key="flow_pick")
    flows_ui.render_flow(eng, pick)
    st.markdown("---")
    with st.container(key="flow_compare"):
        st.markdown("#### All eight flows side by side")
        flows_ui.render_comparison(eng)
