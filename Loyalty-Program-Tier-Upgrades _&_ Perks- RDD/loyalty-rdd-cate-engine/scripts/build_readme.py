"""Fill the generated sections of README.md from outputs/metrics/flows_report.json and outputs/screenshots/.

    python scripts/build_readme.py

Replaces everything between <!-- FLOWS:START --> and <!-- FLOWS:END --> so the README always shows the numbers the
dashboard shows (both read the same JSON written by `python -m src.flows`).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SHOTS = "outputs/screenshots/flows"
STATUS = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL", "INFO": "info"}


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def flow_block(f: dict, shots_dir: Path) -> str:
    out = [f"#### {f['short']}: {f['summary']['verdict']}", "", f"*{f['title']}.* {f['story']}", "",
           f"**Verdict: {f['summary']['verdict']}.** {f['summary']['reason']}", "",
           "| # | Stage | Gate | Metrics at this stage |", "|---|---|---|---|"]
    for i, s in enumerate(f["stages"], 1):
        mets = "; ".join(f"{m['label']} **{m['value']}**" for m in s["metrics"][:4])
        out.append(f"| {i} | {_cell(s['title'].split(':')[0])} | {STATUS[s['status']]} | {_cell(mets)} |")
    out += ["", "Flow-level metrics: " + "; ".join(f"{m['label']} **{m['value']}**" for m in f["summary"]["metrics"]) + ".", ""]
    imgs = [(s["title"], f"{SHOTS}/{f['id']}_{s['key']}.png") for s in f["stages"] if (shots_dir / f"{f['id']}_{s['key']}.png").exists()]
    if (shots_dir / f"{f['id']}_summary.png").exists():
        out += [f"![{f['id']} summary]({SHOTS}/{f['id']}_summary.png)", ""]
    if imgs:
        out += ["<details><summary>Screenshots of every stage</summary>", ""]
        for title, path in imgs:
            out += [f"![{f['id']} {title}]({path})", ""]
        out += ["</details>", ""]
    return "\n".join(out)


def flows_markdown(rep: dict, shots_dir: Path) -> str:
    a = rep["assumptions"]
    md = [
        "Open the dashboard's last tab, **End-to-end flows**, pick a flow, and read it top to bottom. Four questions an analyst actually faces, "
        "two worked examples each; every stage has its own metrics and a PASS / WARN / FAIL gate, every flow ends in a verdict, and a programme "
        "scorecard rolls up all flows and all 12 models. `python -m src.flows` (or `make flows`) writes `outputs/metrics/flows_report.json`; the "
        "tab, the screenshots and this section all read that one file, and `scripts/capture_screenshots.py` checks that the numbers on screen "
        "equal the API's numbers.", "",
        f"Money assumptions (editable under `flows:` in `configs/config.yaml`): **{a['contribution_margin']:.0%} contribution margin** on incremental "
        f"revenue and **${a['perk_cost_per_member']:.0f} perk cost** per Gold member per year. Revenue is room nights x the guest's ADR plus ancillary spend; "
        "direct-booking gains are not monetised.", "",
        "| Flow | Question | Stages |", "|---|---|---|"]
    for t, ft in rep["flow_types"].items():
        ex = [f for f in rep["flows"] if f["type"] == t]
        md.append(f"| **{t}. {ft['title']}** | {ft['question']} | {' / '.join(str(len(f['stages'])) for f in ex)} per example |")
    md += ["", "```mermaid", "flowchart LR",
           "    A[A. Gold effect read-out<br/>setup, density, balance,<br/>estimates, robustness, economics] --> V1[verdict]",
           "    B[B. Validity and negative control<br/>same stages plus replication] --> V2[verdict]",
           "    C[C. Guest what-if<br/>profile, context, models,<br/>evidence, economics] --> V3[verdict]",
           "    D[D. Threshold policy<br/>today, candidate, who, impact,<br/>net value, sensitivity] --> V4[verdict]",
           "    V1 --> S[programme scorecard]", "    V2 --> S", "    V3 --> S", "    V4 --> S", "```", ""]
    for t, ft in rep["flow_types"].items():
        md += [f"### {t}. {ft['title']}", "", f"*{ft['question']}*", ""]
        for f in (x for x in rep["flows"] if x["type"] == t):
            md.append(flow_block(f, shots_dir))
    prog = rep["programme"]
    md += ["### Programme scorecard (all flows, all models)", ""]
    if (shots_dir / "overall.png").exists():
        md += [f"![programme scorecard]({SHOTS}/overall.png)", ""]
    md += ["| Flow | Verdict | Headline |", "|---|---|---|"]
    md += [f"| {r['Flow']} | **{r['Verdict']}** | {_cell(r['Headline'])} |" for r in prog["compare"]]
    md.append("")
    for g in prog["groups"]:
        md += [f"**{g['title']}:** " + "; ".join(f"{m['label']} **{m['value']}**" for m in g["metrics"]) + ".", ""]
    md += [
        "How to read the two examples that look odd: **B1** is a programme whose perks do nothing, and with the default seed it produces one "
        "of the roughly 1-in-20 draws that looks significant. The flow reports that honestly as a `FALSE POSITIVE` and its replication stage shows the "
        "method is calibrated (about 5% over re-draws, standard errors match the spread of estimates). **D1** and **D2** both say ADOPT because they sit "
        "on opposite sides of the same break-even perk cost (about $220 per member): lowering the threshold pays below it, raising the threshold pays above it.", ""]
    return "\n".join(md)


def main() -> None:
    rep_path = ROOT / "outputs" / "metrics" / "flows_report.json"
    if not rep_path.exists():
        sys.exit("run `python -m src.flows` first")
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    block = flows_markdown(rep, ROOT / SHOTS)
    text = README.read_text(encoding="utf-8")
    pat = re.compile(r"<!-- FLOWS:START -->.*?<!-- FLOWS:END -->", re.S)
    if not pat.search(text):
        sys.exit("README.md has no FLOWS markers")
    README.write_text(pat.sub(lambda m: f"<!-- FLOWS:START -->\n{block}\n<!-- FLOWS:END -->", text), encoding="utf-8")
    print(f"README flows section rebuilt ({len(block.splitlines())} lines)")


if __name__ == "__main__":
    main()
