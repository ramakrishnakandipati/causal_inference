"""Streamlit rendering of the end-to-end flows in `src/flows.py` (the last tab of the dashboard).

Every stage starts with an invisible anchor `<div id="flowstep-<flow>-<stage>">` so scripts/capture_screenshots.py can crop
each stage, the flow summary (`flowsummary-<flow>`), and the programme scorecard (`flow-overall` ... `flow-end`) on their own.
Compatible with Streamlit 1.12 (no nested columns, no container keys).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import flows as fl

BLUE, ORANGE, AQUA, AMBER, RED, GREY, INK = "#257de0", "#f47230", "#00b894", "#f2a600", "#ec4647", "#666666", "#191919"
STATUS_COLOR = {"PASS": "#008a17", "WARN": "#b57d00", "FAIL": RED, "INFO": GREY}
LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=36, b=10), height=330, legend=dict(orientation="h", y=-0.25))


def md(text: str) -> str:
    """Escape dollar signs: Streamlit markdown treats $...$ as LaTeX, which garbles any sentence with two dollar amounts."""
    return str(text).replace("$", "\\$")


@st.experimental_memo(show_spinner=False)
def _load_report(path: str, mtime: float) -> dict:
    return fl.json.loads(open(path, encoding="utf-8").read())


def get_report(svc) -> dict:
    p = svc.out / "metrics" / fl.REPORT_NAME
    if not p.exists():
        with st.spinner("Building the flows (one-off, about a minute) ..."):
            fl.save(fl.build_all(svc), svc.out)
    return _load_report(str(p), os.path.getmtime(p))


# ----------------------------------------------------------------------------- charts
def _vline(fig, x, color=INK, dash="dash", row=None, col=None):
    fig.add_vline(x=x, line_dash=dash, line_color=color, line_width=1, row=row, col=col)


def make_chart(ch: dict):
    t = ch["type"]
    if t == "takeup":
        fig = go.Figure(go.Scatter(x=ch["r"], y=[v * 100 for v in ch["t"]], mode="lines+markers", line=dict(color=BLUE, width=3), name="Gold take-up"))
        _vline(fig, ch["cutoff"])
        fig.update_layout(title="Share of members holding Gold, by qualifying nights", xaxis_title="Qualifying nights", yaxis_title="% with Gold", **LAYOUT)
    elif t == "density":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ch["x"], y=ch["n"], mode="markers", marker=dict(color=GREY, size=8), name="Guests per night"))
        fig.add_trace(go.Scatter(x=ch["left"]["x"], y=ch["left"]["y"], mode="lines", line=dict(color=BLUE, width=3), name="Local fit, below"))
        fig.add_trace(go.Scatter(x=ch["right"]["x"], y=ch["right"]["y"], mode="lines", line=dict(color=ORANGE, width=3), name="Local fit, at/above"))
        _vline(fig, ch["cutoff"] - 0.5)
        fig.update_layout(title="Guests per qualifying night around the cutoff", xaxis_title="Qualifying nights", yaxis_title="Guests", **LAYOUT)
    elif t == "balance":
        order = np.argsort(ch["z"])
        cov = [ch["covariate"][i] for i in order]
        z = [ch["z"][i] for i in order]
        ok = [ch["ok"][i] for i in order]
        fig = go.Figure(go.Bar(x=z, y=cov, orientation="h", marker_color=[BLUE if o else ORANGE for o in ok]))
        for v in (-1.96, 1.96):
            _vline(fig, v, GREY, "dot")
        fig.update_layout(title="Jump at the cutoff in standard errors (dotted = ±1.96)", xaxis_title="Jump / SE", **{**LAYOUT, "height": 360})
    elif t == "rdd":
        outs = list(ch["outcomes"])
        fig = make_subplots(rows=1, cols=3, subplot_titles=[ch["labels"][o] for o in outs], horizontal_spacing=0.07)
        for i, oc in enumerate(outs, start=1):
            d = ch["outcomes"][oc]
            fig.add_trace(go.Scatter(x=d["bins_x"], y=d["bins_y"], mode="markers", marker=dict(color=GREY, size=6), showlegend=(i == 1), name="Mean per night"), row=1, col=i)
            for side, color, fill, nm in (("left", BLUE, "rgba(37,125,224,0.18)", "Below threshold"), ("right", ORANGE, "rgba(244,114,48,0.18)", "At or above")):
                s = d[side]
                fig.add_trace(go.Scatter(x=s["x"] + s["x"][::-1], y=s["hi"] + s["lo"][::-1], fill="toself", mode="none", fillcolor=fill, showlegend=False, hoverinfo="skip"), row=1, col=i)
                fig.add_trace(go.Scatter(x=s["x"], y=s["f"], mode="lines", line=dict(color=color, width=3), showlegend=(i == 1), name=nm), row=1, col=i)
            _vline(fig, ch["cutoff"], row=1, col=i)
        fig.update_layout(**{**LAYOUT, "height": 360, "margin": dict(l=10, r=10, t=30, b=10)})
    elif t == "robust":
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Room-night estimate by bandwidth", "Placebo cutoffs (true effect there: 0)"], horizontal_spacing=0.1)
        s = ch["sens"]
        fig.add_trace(go.Scatter(x=[a["bandwidth"] for a in s], y=[a["estimate"] for a in s], mode="markers+lines", marker=dict(color=BLUE, size=8),
                                 error_y=dict(type="data", array=[1.96 * a["se"] for a in s], color=BLUE), name="Estimate ±95%"), row=1, col=1)
        fig.add_hline(y=ch["headline"], line_dash="dash", line_color=INK, line_width=1, row=1, col=1)
        p = ch["placebo"]
        fig.add_trace(go.Scatter(x=[str(int(a["placebo_cutoff"])) for a in p], y=[a["estimate"] for a in p], mode="markers", marker=dict(color=ORANGE, size=9),
                                 error_y=dict(type="data", array=[1.96 * a["se"] for a in p], color=ORANGE), name="Placebo ±95%"), row=1, col=2)
        fig.add_hline(y=0, line_color=GREY, line_width=1, row=1, col=2)
        fig.update_layout(**LAYOUT)
    elif t == "caterpillar":
        order = np.argsort(ch["est"])
        est, lo, hi = (np.array(ch[k])[order] for k in ("est", "lo", "hi"))
        miss = (lo > ch["truth"]) | (hi < ch["truth"])
        fig = go.Figure(go.Scatter(x=list(range(len(est))), y=est.tolist(), mode="markers", marker=dict(color=[ORANGE if m else BLUE for m in miss], size=8),
                                   error_y=dict(type="data", array=(hi - est).tolist(), arrayminus=(est - lo).tolist(), color=GREY), name="Estimate ±95%"))
        fig.add_hline(y=ch["truth"], line_dash="dash", line_color=INK, line_width=1)
        fig.update_layout(title="Room-night estimate in each re-drawn panel (orange = interval misses the truth)", xaxis_title="Replication (sorted)", yaxis_title="Nights", **LAYOUT)
    elif t == "rates":
        fig = go.Figure(go.Bar(x=ch["labels"], y=[v * 100 for v in ch["values"]], marker_color=[ORANGE, BLUE], text=[f"{v:.0%}" for v in ch["values"]], textposition="outside"))
        fig.update_layout(title="How often the density test fires", yaxis_title="% of re-drawn panels", yaxis_range=[0, 115], **{**LAYOUT, "height": 300})
    elif t == "waterfall":
        vals = ch["values"]
        colors = [AQUA if v >= 0 else ORANGE for v in vals]
        fig = go.Figure(go.Bar(x=ch["labels"], y=vals, marker_color=colors, text=[fl._money(v) for v in vals], textposition="outside"))
        lo, hi = min(0.0, min(vals)), max(vals)
        fig.update_layout(title="From revenue to net value", yaxis_title="USD", yaxis_range=[lo - 0.22 * (hi - lo), hi + 0.18 * (hi - lo)],
                          **{**LAYOUT, "height": 320})
    elif t == "distribution":
        e = np.array(ch["edges"])
        mid = ((e[:-1] + e[1:]) / 2).tolist()
        fig = go.Figure(go.Bar(x=mid, y=ch["counts"], marker_color="#b6d5f8", name="Guests near the cutoff"))
        _vline(fig, ch["mean"], GREY, "dot")
        _vline(fig, ch["guest"], ORANGE, "solid")
        fig.add_annotation(x=ch["guest"], y=max(ch["counts"]), text="this guest", showarrow=False, yshift=12, font=dict(color=ORANGE))
        fig.update_layout(title="Predicted Gold effect across guests near the cutoff (dotted = average)", xaxis_title="Predicted extra room nights", yaxis_title="Guests", **LAYOUT)
    elif t == "learners":
        names, vals, corr = ch["learners"], ch["values"], ch["corr"]
        fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=[BLUE if c >= 0.85 else "#b0b0b0" for c in corr],
                               text=[f"{v:+.2f}" for v in vals], textposition="outside"))
        _vline(fig, ch["headline"], ORANGE, "solid")
        _vline(fig, ch["weighted"], INK, "dot")
        fig.update_layout(title="Predicted effect by learner (blue = corr ≥ 0.85; orange = headline, dotted = accuracy-weighted)",
                          xaxis_title="Extra room nights", **{**LAYOUT, "height": 420})
    elif t == "nights_hist":
        fig = go.Figure(go.Bar(x=ch["x"], y=ch["n"], marker_color=["#b6d5f8" if x < ch["cutoff"] else "#84b9f3" for x in ch["x"]]))
        _vline(fig, ch["cutoff"] - 0.5, INK)
        _vline(fig, ch["new_cutoff"] - 0.5, ORANGE, "solid")
        fig.update_layout(title="Members by qualifying nights (black = today's cutoff, orange = candidate)", xaxis_title="Qualifying nights", yaxis_title="Members", **LAYOUT)
    elif t == "mix":
        fig = go.Figure(go.Bar(x=ch["labels"], y=[v * 100 for v in ch["values"]], marker_color=[AQUA, BLUE, ORANGE, "#b0b0b0"],
                               text=[f"{v:.0%}" for v in ch["values"]], textposition="outside"))
        fig.update_layout(title="Who is affected: guest-type mix", yaxis_title="% of affected guests", yaxis_range=[0, 100], **{**LAYOUT, "height": 300})
    elif t == "policy_grid":
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Net value by threshold (assumed margin and perk cost)", "Net value of the candidate by perk cost"], horizontal_spacing=0.12)
        g = ch["grid"]
        fig.add_trace(go.Bar(x=[a["threshold"] for a in g], y=[a["net_m"] for a in g], marker_color=[AQUA if a["net_m"] >= 0 else ORANGE for a in g], showlegend=False), row=1, col=1)
        _vline(fig, ch["cutoff"], INK, row=1, col=1)
        _vline(fig, ch["new_cutoff"], BLUE, "solid", row=1, col=1)
        cg = ch["cost_grid"]
        fig.add_trace(go.Bar(x=[f"${a['perk_cost']}" for a in cg], y=[a["net_m"] for a in cg], marker_color=[AQUA if a["net_m"] >= 0 else ORANGE for a in cg], showlegend=False), row=1, col=2)
        fig.update_layout(**LAYOUT)
    else:
        return None
    return fig


# ----------------------------------------------------------------------------- stage / summary rendering
def _metrics_grid(metrics: list) -> None:
    wide = any(len(m["value"]) > 16 for m in metrics)
    per = 3 if wide else 4
    for i in range(0, len(metrics), per):
        cols = st.columns(per)
        for col, m in zip(cols, metrics[i:i + per]):
            col.metric(m["label"], m["value"], help=m["help"] or None)


def _chip(status: str) -> str:
    return (f"<span style='background:{STATUS_COLOR[status]};color:#fff;padding:2px 10px;border-radius:10px;"
            f"font-size:12px;font-weight:600;vertical-align:middle'>{status}</span>")


def render_stage(flow_id: str, n: int, s: dict) -> None:
    st.markdown(f"<div id='flowstep-{flow_id}-{s['key']}'></div>", unsafe_allow_html=True)
    st.markdown(f"### {n}. {s['title']} &nbsp; {_chip(s['status'])}", unsafe_allow_html=True)
    st.caption(md(s["question"]))
    _metrics_grid(s["metrics"])
    st.markdown(f"**{md(s['takeaway'])}**")
    fig = make_chart(s["chart"])
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    if s.get("table"):
        st.table(pd.DataFrame(s["table"]))


def render_summary(flow: dict) -> None:
    sm = flow["summary"]
    st.markdown(f"<div id='flowsummary-{flow['id']}'></div>", unsafe_allow_html=True)
    st.markdown(f"### Flow summary: {flow['short']}")
    v = sm["verdict"]
    box = st.success if v in fl.GOOD else st.error if v in fl.BAD else st.warning
    box(f"**{v}**  \n{md(sm['reason'])}")
    _metrics_grid(sm["metrics"])


def render_programme(rep: dict) -> None:
    prog = rep["programme"]
    st.markdown("<div id='flow-overall'></div>", unsafe_allow_html=True)
    st.markdown("## Programme scorecard: all flows and all models")
    a = rep["assumptions"]
    st.caption(md(f"Money assumptions: {a['contribution_margin']:.0%} contribution margin on incremental revenue, ${a['perk_cost_per_member']:.0f} perk cost per Gold member; "
                  "change them in configs/config.yaml (flows) and rerun `python -m src.flows`."))
    groups = prog["groups"]
    for i in range(0, len(groups), 3):
        cols = st.columns(3)
        for col, g in zip(cols, groups[i:i + 3]):
            col.markdown(f"**{g['title']}**")
            col.table(pd.DataFrame([{"Metric": m["label"], "Value": m["value"]} for m in g["metrics"]]).set_index("Metric"))
    st.markdown("**All eight flows**")
    comp = pd.DataFrame(prog["compare"])
    st.table(comp.set_index("Flow"))
    pers = prog["models"]["personas"]
    names = list(next(iter(pers.values())))
    fig = go.Figure()
    for (persona, est), color in zip(pers.items(), (BLUE, ORANGE)):
        fig.add_trace(go.Bar(x=names, y=[est[n] for n in names], name=persona, marker_color=color))
    fig.update_layout(barmode="group", title="Model comparison across the two guest flows: predicted Gold effect by learner",
                      yaxis_title="Extra room nights", xaxis_tickangle=-35,
                      **{**LAYOUT, "height": 420, "margin": dict(l=10, r=10, t=36, b=100), "legend": dict(orientation="h", y=-0.4)})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<div id='flow-end'></div>", unsafe_allow_html=True)


def render(svc) -> None:
    rep = get_report(svc)
    flows = rep["flows"]
    st.markdown("### End-to-end flows")
    st.write("Four questions an analyst actually faces, each worked through two examples. Every stage has its own metrics and a pass/warn/fail "
             "gate; every flow ends in a verdict; the scorecard at the bottom rolls up all models and all flows.")
    ids = [f["id"] for f in flows]
    qp = st.experimental_get_query_params().get("flow", [ids[0]])[0]
    default = ids.index(qp) if qp in ids else 0
    fid = st.selectbox("Choose a flow", ids, index=default, format_func=lambda i: next(f["short"] for f in flows if f["id"] == i), key="flow_pick")
    flow = next(f for f in flows if f["id"] == fid)
    ft = rep["flow_types"][flow["type"]]
    st.markdown("<div id='flow-top'></div>", unsafe_allow_html=True)
    st.markdown(f"## {flow['short']}")
    st.markdown(f"**{md(ft['title'])}**: {md(ft['question'])}")
    st.markdown(f"*{md(flow['title'])}*. {md(flow['story'])}")
    st.markdown("---")
    for n, s in enumerate(flow["stages"], start=1):
        render_stage(fid, n, s)
    render_summary(flow)
    st.markdown("---")
    render_programme(rep)
