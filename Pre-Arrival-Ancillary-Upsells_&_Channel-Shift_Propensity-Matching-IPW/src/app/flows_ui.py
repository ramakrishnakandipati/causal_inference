"""Tab 5: end-to-end flows rendered from outputs/metrics/flows.json."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.common.config import out_dir

TYPE_NAMES = {"A": "A · Guest decision", "B": "B · Cohort read-out", "C": "C · Validity checks", "D": "D · Campaign policy"}
PIPE = ["Data", "Propensity", "Balance", "ATE / ATT", "Refutation", "CATE models", "Policy"]
BADGE = {"ok": "✅", "warn": "⚠️"}


def _chart(ch: dict, key: str):
    fig = go.Figure()
    if ch["kind"] == "estimates":
        fig.add_trace(go.Bar(x=ch["values"], y=ch["labels"], orientation="h", marker_color="#1f77b4", text=ch["values"], textposition="outside"))
        if ch.get("truth") is not None:
            fig.add_vline(x=ch["truth"], line_dash="dash", line_color="black", annotation_text=f"ref {ch['truth']}")
        fig.update_yaxes(autorange="reversed")
    else:
        if ch.get("values2") is not None:
            fig.add_trace(go.Bar(x=ch["labels"], y=ch["values"], name=ch.get("name1", ""), marker_color="#d62728"))
            fig.add_trace(go.Bar(x=ch["labels"], y=ch["values2"], name=ch.get("name2", ""), marker_color="#2ca02c"))
            fig.add_hline(y=0.1, line_dash="dash", line_color="black")
        else:
            fig.add_trace(go.Bar(x=ch["labels"], y=ch["values"], marker_color="#1f77b4", text=ch["values"], textposition="outside"))
    fig.update_layout(title=ch["title"], height=330, margin=dict(l=10, r=10, t=40, b=10), showlegend=ch.get("values2") is not None)
    st.plotly_chart(fig, width="stretch", key=key)


def render():
    p = out_dir("metrics") / "flows.json"
    if not p.exists():
        st.warning("flows.json missing. Run `python scripts/run_pipeline.py --stage flows`.")
        return
    data = json.load(open(p, encoding="utf-8"))
    flows, overall = data["flows"], data["overall"]
    st.subheader("End-to-end flows: from raw logs to a campaign decision")
    st.caption("Pipeline: " + "  →  ".join(PIPE) + ".  Four flow types × two examples; every stage shows its own metrics and the programme-level metrics sit at the bottom.")
    with st.expander("All flows at a glance", expanded=False):
        rows = [{"flow": f["short"], "type": TYPE_NAMES[f["type"]], **f["summary"]} for f in flows]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    names = [f["short"] for f in flows]
    choice = st.radio("Choose a flow", names, horizontal=True, key="flow_choice")
    f = flows[names.index(choice)]
    st.markdown(f"### {f['title']}")
    st.info(f["story"])
    n = len(f["stages"])
    st.progress(1.0, text=f"{n} stages · " + " → ".join(s["name"].split(" · ")[1] for s in f["stages"]))
    for si, s in enumerate(f["stages"]):
        with st.container(border=True):
            st.markdown(f"#### {BADGE.get(s['status'], '')} {s['name']}")
            items = list(s["metrics"].items())
            for i in range(0, len(items), 4):
                cols = st.columns(4)
                for c, (k, v) in zip(cols, items[i:i + 4]):
                    c.metric(k, "–" if v is None else v)
            st.caption(s["reading"])
            if s.get("chart") or s.get("table"):
                c1, c2 = st.columns([3, 2]) if (s.get("chart") and s.get("table")) else (st.container(), st.container())
                if s.get("chart"):
                    with c1:
                        _chart(s["chart"], f"{f['id']}_{si}")
                if s.get("table"):
                    with c2:
                        st.dataframe(pd.DataFrame(s["table"]).astype(str), width="stretch", hide_index=True)
    st.markdown("### Programme-level metrics (all flows)")
    items = list(overall.items())
    for i in range(0, len(items), 4):
        cols = st.columns(4)
        for c, (k, v) in zip(cols, items[i:i + 4]):
            c.metric(k, v)
