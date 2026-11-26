"""Streamlit rendering for the end-to-end flows in `src/flows.py` (tab 6 of the workbench).

Every stage of a flow is a keyed container (`flowstep_<key>`) so `scripts/capture_screenshots.py` can screenshot it on its own;
the whole flow sits in `flow_full`, and the scorecard in `flow_overall`.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import flows as fl
from src.evaluation import visualizations as viz
from src.serving import STRATEGIES, Engine

BLUE, ORANGE, AQUA, INK, INK2, GRAY = viz.BLUE, viz.ORANGE, viz.AQUA, viz.INK, viz.INK2, viz.GRAY
GOOD = {"INCREMENTAL AND PROFITABLE", "SAFE TO RUN", "LEAN IN", "NO DETECTABLE EFFECT", "BASELINE"}
BAD = {"FALSE POSITIVE", "DO NOT TRUST THE PROJECTION", "PULL BACK", "INCREMENTAL BUT BELOW BREAK-EVEN"}


@st.cache_data(show_spinner=False)
def cached_report(_eng: Engine, flow_id: str) -> dict:
    return fl.report(_eng, fl.FLOW_BY_ID[flow_id])


@st.cache_data(show_spinner=False)
def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(fl.resolve("outputs/metrics") / name)


def _fig(h: float = 3.3):
    f, ax = plt.subplots(figsize=(10.5, h))
    return f, ax


def _roll(x):
    return viz._roll(np.asarray(x, float))


# ----------------------------------------------------------------------------- charts
def chart(eng: Engine, rep: dict, step: dict) -> None:
    c = step["chart"]
    if c == "none":
        return
    f, ax = getattr(_Charts, c)(eng, rep)
    f.tight_layout()
    st.pyplot(f, width="stretch")
    plt.close(f)


class _Charts:
    @staticmethod
    def trends(eng, rep):
        sc = rep["scenario"]
        spec, panel = eng.scenario_spec(sc), eng.panel(sc)
        T0 = spec["start_day"] - 1
        wide = panel.pivot(index="day_index", columns="dma_code", values="room_nights_booked")
        idx = wide / wide.iloc[:T0].mean() * 100
        dates = panel[panel.dma_code == spec["treated"]].sort_values("day_index")["date"].reset_index(drop=True)
        f, ax = _fig()
        ax.plot(dates, _roll(idx[spec["donors"]].mean(axis=1)), color=GRAY, lw=2, label=f"Average of {len(spec['donors'])} donor DMAs")
        ax.plot(dates, _roll(idx[spec["treated"]]), color=INK, lw=2, label=f"{spec['treated']} (observed)")
        ax.axvline(dates.iloc[T0], color=INK2, lw=1, ls=(0, (4, 3)))
        ax.set_ylabel("Index (pre mean = 100)")
        ax.set_title("Parallel-trends check: indexed room nights, 7-day mean")
        ax.legend(loc="upper left", fontsize=8)
        return f, ax

    @staticmethod
    def did(eng, rep):
        S = _csv("panel_att_summary.csv")
        S = S[(S.scenario == rep["scenario"]) & (S.outcome == "room_nights_booked")].set_index("estimator")
        names = fl.DID_FAMILY + ["ASCM (Ridge)"]
        f, ax = _fig(2.9)
        yy = np.arange(len(names))[::-1]
        for yi, nm in zip(yy, names):
            r = S.loc[nm]
            col = BLUE if nm == "ASCM (Ridge)" else ORANGE
            ax.errorbar(r.att_mean, yi, xerr=[[r.att_mean - r.ci_lo], [r.ci_hi - r.att_mean]], fmt="o", color=col, capsize=3, lw=1.6)
            ax.text(r.ci_hi + 8, yi, f"{r.att_mean:,.0f}", va="center", fontsize=8.5)
        truth = float(S.loc["ASCM (Ridge)", "att_true_mean"])
        ax.axvline(truth, color=INK, ls=(0, (5, 3)), lw=1.4, label=f"Truth ({truth:,.0f})")
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(yy, names)
        ax.margins(x=0.1)
        ax.set_xlabel("ATT, room nights / day (95% interval)")
        ax.set_title("DiD family vs. ASCM reference")
        ax.legend(fontsize=8, loc="lower right")
        return f, ax

    @staticmethod
    def sc(eng, rep):
        sc = rep["scenario"]
        spec, panel = eng.scenario_spec(sc), eng.panel(sc)
        T0 = spec["start_day"] - 1
        dates = panel[panel.dma_code == spec["treated"]].sort_values("day_index")["date"].reset_index(drop=True)
        f, ax = _fig()
        for nm, col in zip(["SCM", "ASCM (Ridge)", "SDID"], [ORANGE, BLUE, AQUA]):
            s = pd.DataFrame(eng.att(sc, nm, "room_nights_booked")["series"])
            if nm == "SCM":
                true_y0 = s["y_obs"] - s["true_att"].fillna(0)
                ax.plot(dates, _roll(true_y0), color=GRAY, lw=1.8, ls=(0, (5, 3)), label="True counterfactual")
            ax.plot(dates, _roll(s["y0_hat"]), color=col, lw=1.8, label=nm)
        ax.plot(dates, _roll(s["y_obs"]), color=INK, lw=2.1, label="Observed")
        ax.axvline(dates.iloc[T0], color=INK2, lw=1, ls=(0, (4, 3)))
        ax.set_ylabel("Room nights (7-day mean)")
        ax.set_title("Synthetic-control counterfactuals")
        ax.legend(loc="upper left", ncol=3, fontsize=8)
        return f, ax

    @staticmethod
    def placebo(eng, rep):
        pl = eng.att(rep["scenario"], fl.HEADLINE, "room_nights_booked").get("placebo")
        f, ax = _fig(3.0)
        t = pd.DataFrame(pl)
        t["ratio"] = t["post_rmspe"] / t["pre_rmspe"].clip(lower=1e-9)
        t = t.sort_values("ratio", ascending=False).reset_index(drop=True)
        ax.bar(range(len(t)), t["ratio"], color=[BLUE if u == "TREATED" else "#b9b9b9" for u in t["unit"]], zorder=2)
        ax.set_xticks(range(len(t)), ["TRT" if u == "TREATED" else u for u in t["unit"]], rotation=90, fontsize=7.5)
        rank = int(t.index[t.unit == "TREATED"][0]) + 1
        ax.set_title(f"In-space placebo: treated ranks {rank} of {len(t)} (exact p = {rank / len(t):.3f})")
        ax.set_ylabel("Post / pre RMSPE")
        return f, ax

    @staticmethod
    def econ(eng, rep):
        E = _csv("economics.csv")
        E = E[E.scenario == rep["scenario"]]
        f, ax = _fig(3.0)
        if not len(E):
            ax.text(0.5, 0.5, "No spend in this market: no economics to show", ha="center", va="center", transform=ax.transAxes, color=INK2)
            ax.set_axis_off()
            return f, ax
        E = E[np.isfinite(E.iroas_hi) & np.isfinite(E.iroas_lo)].sort_values("iroas")
        yy = np.arange(len(E))
        ax.errorbar(E.iroas, yy, xerr=[(E.iroas - E.iroas_lo).clip(lower=0), (E.iroas_hi - E.iroas).clip(lower=0)], fmt="o", color=BLUE, capsize=3)
        ax.axvline(float(E.true_iroas.iloc[0]), color=INK, ls=(0, (5, 3)), lw=1.4, label=f"True iROAS {float(E.true_iroas.iloc[0]):.2f}x")
        ax.axvline(1.0, color=ORANGE, lw=1.2, label="Break-even (1.0x)")
        ax.set_yticks(yy, E.estimator)
        ax.set_xlabel("iROAS (95% interval)")
        ax.set_title("iROAS by estimator")
        ax.legend(fontsize=8)
        return f, ax

    # --- CATE
    @staticmethod
    def learners(eng, rep):
        p = rep["preds"]
        f, ax = _fig(3.0)
        names = list(p)
        ax.bar(range(len(p)), list(p.values()), color=[viz.LEARNER_FAMILY.get(n, BLUE) for n in names], width=0.6, zorder=2)
        ax.axhline(rep["headline"]["median"], color=INK, ls=(0, (5, 3)), lw=1.3, label=f"median {rep['headline']['median']:,.0f}")
        ax.axhline(rep["avg"], color=GRAY, ls=(0, (1, 2)), lw=1.3, label=f"campaign average {rep['avg']:,.0f}")
        ax.set_xticks(range(len(p)), names, fontsize=8.5)
        ax.set_ylabel("Incremental room nights / day")
        ax.set_title("Per-learner prediction for this day")
        ax.legend(fontsize=8)
        return f, ax

    @staticmethod
    def index(eng, rep):
        f, ax = _fig(2.3)
        vals = [rep["avg"], rep["headline"]["median"]]
        ax.barh(["Campaign average", "This day"], vals, color=[GRAY, BLUE], height=0.5, zorder=2)
        for i, v in enumerate(vals):
            ax.text(v + 2, i, f"{v:,.0f}", va="center", fontsize=9)
        ax.set_xlabel("Median-learner incremental room nights / day")
        ax.set_title(f"Relative index {rep['headline']['index']:.2f}x")
        return f, ax

    # --- pacing
    @staticmethod
    def plan(eng, rep):
        d = rep["sim"]["daily"]
        f, ax = _fig(3.2)
        ax.bar(d["day"], np.array(d["spend_usd"]) / 1000, color="#b9d6f8", width=0.8, zorder=2)
        ax2 = ax.twinx()
        ax2.grid(False)
        ax2.plot(d["day"], d["incremental_room_nights"], color=BLUE, lw=2)
        ax.set_xlabel("Campaign day")
        ax.set_ylabel("Spend ($k / day)")
        ax2.set_ylabel("Incremental room nights / day", color=INK2)
        ax.set_title(f"Daily plan: {rep['sim']['strategy']}")
        return f, ax

    @staticmethod
    def compare(eng, rep):
        c = pd.DataFrame(rep["sim"]["strategy_comparison"]).sort_values("incremental_room_nights")
        f, ax = _fig(3.2)
        ax.barh(c["strategy"], c["incremental_room_nights"], color=[BLUE if s == rep["sim"]["strategy"] else "#b9b9b9" for s in c["strategy"]], zorder=2)
        ax.set_xlabel("Projected incremental room nights (60 days)")
        ax.set_title("Eight plans, same budget (projected)")
        return f, ax

    @staticmethod
    def audit(eng, rep):
        pv = _csv("pacing_validation.csv")
        pv = pv[pv.strategy.isin(STRATEGIES)].set_index("strategy").loc[STRATEGIES]
        f, ax = _fig(3.4)
        x = np.arange(len(pv))
        ax.bar(x - 0.2, pv.projected_vs_even_pct, 0.38, color=BLUE, label="Projected (fitted curve)", zorder=2)
        ax.bar(x + 0.2, pv.true_vs_even_pct, 0.38, color=ORANGE, label="True response", zorder=2)
        sel = STRATEGIES.index(rep["flow"]["strategy"])
        ax.axvspan(sel - 0.5, sel + 0.5, color="#000000", alpha=0.06, lw=0)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x, pv.index, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Change vs. even pacing (%)")
        ax.set_title("Does the projection survive the truth? (this flow's plan shaded)")
        ax.legend(fontsize=8)
        return f, ax


# ----------------------------------------------------------------------------- layout
def _metrics(ms: list[dict]) -> None:
    ms = [m for m in ms if m["label"] != "Verdict"]      # the verdict is shown in the banner; long text truncates in a metric tile
    cols = st.columns(max(len(ms), 1))
    for col, m in zip(cols, ms):
        col.metric(m["label"], m["value"], help=m.get("help"))


def _banner(rep: dict) -> None:
    v = rep["verdict"]
    msg = f"**{v}** - {rep['action']}"
    (st.success if v in GOOD else st.error if v in BAD else st.warning)(msg)


def render_flow(eng: Engine, flow_id: str) -> None:
    flow = fl.FLOW_BY_ID[flow_id]
    ftype = fl.FLOW_TYPES[flow["type"]]
    rep = cached_report(eng, flow_id)
    with st.container(key="flow_full"):
        st.markdown(f"### Flow {flow['id']} - {flow['title']}")
        st.caption(f"Flow type {flow['type']}: **{ftype['title']}** - {ftype['question']}")
        st.markdown(flow["story"])
        track = "  ->  ".join(s["title"].split(". ", 1)[1] for s in rep["steps"])
        st.markdown(f"<span class='note'>Stages: {track}</span>", unsafe_allow_html=True)
        with st.container(key="flow_overall"):
            st.markdown("#### Overall scorecard")
            _banner(rep)
            _metrics(rep["overall"])
        for step in rep["steps"]:
            with st.container(key=f"flowstep_{step['key']}", border=True):
                st.markdown(f"**{step['title']}**")
                if step.get("verdict"):
                    _banner(rep)
                _metrics(step["metrics"])
                chart(eng, rep, step)
                st.markdown(f"<span class='note'>{step['note']}</span>", unsafe_allow_html=True)


def render_comparison(eng: Engine) -> None:
    rows = []
    for f in fl.FLOWS:
        r, h = cached_report(eng, f["id"]), None
        h = r["headline"]
        if r["kind"] == "panel":
            key = f"ATT {h['att_mean']:,.0f}/day (truth {h['truth']:,.0f}); p={h['p']:.3f}; iROAS {h['iroas']:.2f}x" if r["has_econ"] else f"ATT {h['att_mean']:,.0f}/day (truth 0); p={h['p']:.3f}"
        elif r["kind"] == "cate":
            key = f"median lift {h['median']:,.0f} RN/day ({h['lo']:,.0f}-{h['hi']:,.0f}); index {h['index']:.2f}x"
        else:
            key = f"projected {h['projected']:+.1f}% vs even; true {h['true']:+.1f}%; iROAS {h['iroas']:.2f}x"
        rows.append({"Flow": f["id"], "Type": fl.FLOW_TYPES[f["type"]]["title"], "Example": f["title"], "Key metrics": key, "Verdict": r["verdict"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
