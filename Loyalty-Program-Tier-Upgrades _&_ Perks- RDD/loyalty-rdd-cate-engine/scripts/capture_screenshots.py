"""Launch FastAPI + Streamlit, drive them with Playwright and save PNGs to outputs/screenshots/.

Usage:
    python scripts/capture_screenshots.py            # API docs, every dashboard tab, and all end-to-end flows
    python scripts/capture_screenshots.py flows      # only the end-to-end flow screenshots (outputs/screenshots/flows/)
Requires:  python -m playwright install chromium

Outputs: 01-04 API, 10-16 dashboard tabs, flows/<ID>_<stage>.png, flows/<ID>_summary.png, flows/<ID>_full.png,
flows/overall.png, and capture_report.json (including a check that the numbers on screen equal the API's numbers).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE_VIEW = {"width": 1500, "height": 1000}


def wait_for(url: str, timeout: float = 120.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status < 500:
                    return
        except Exception:
            time.sleep(1.0)
    raise TimeoutError(f"{url} did not come up in {timeout}s")


def api_post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


class Ui:
    """Small helpers around a Playwright page showing the Streamlit app."""

    def __init__(self, page, url: str):
        self.page, self.url = page, url

    def idle(self, timeout: int = 240000) -> None:
        """Streamlit disables the tab buttons while a script run is in progress."""
        self.page.wait_for_timeout(1500)
        self.page.wait_for_function("() => !document.querySelector('[role=tab][disabled]')", timeout=timeout)
        self.page.wait_for_timeout(800)

    def open(self, query: str = "") -> None:
        self.page.set_viewport_size(BASE_VIEW)
        self.page.goto(self.url + query)
        self.page.wait_for_selector("text=Does reaching Gold change how guests behave?", timeout=120000)
        self.idle()

    def tab(self, label: str) -> None:
        self.idle()
        self.page.get_by_role("tab", name=label).click()
        self.idle()
        try:
            self.page.wait_for_selector(".js-plotly-plot", timeout=4000)
        except Exception:
            pass
        self.page.wait_for_timeout(1200)

    def fit(self, cap: int = 12000) -> int:
        """Grow the viewport to the full content height (Streamlit scrolls inside <section.main>, so full_page is not enough)."""
        h = self.page.evaluate("() => document.querySelector('section.main').scrollHeight")
        h = int(min(max(h + 120, BASE_VIEW["height"]), cap))
        self.page.set_viewport_size({"width": BASE_VIEW["width"], "height": h})
        self.page.wait_for_timeout(1500)
        return h

    def shot(self, path: Path) -> None:
        self.fit()
        self.page.screenshot(path=str(path))
        self.page.set_viewport_size(BASE_VIEW)

    def crop(self, start_id: str, end_id: str, path: Path, pad: int = 6) -> bool:
        """Screenshot the slice of the page between two anchor divs (the viewport must already be tall enough)."""
        box = self.page.evaluate(
            """([a, b]) => {
                const A = document.getElementById(a), B = document.getElementById(b);
                const bc = document.querySelector('section.main .block-container') || document.body;
                if (!A) return null;
                const ra = A.getBoundingClientRect(), rb = B ? B.getBoundingClientRect() : null, rc = bc.getBoundingClientRect();
                const top = ra.top + window.scrollY;
                const bottom = rb ? rb.top + window.scrollY : top + 800;
                return {x: Math.max(rc.left - 8, 0), y: Math.max(top - 4, 0), width: rc.width + 16, height: bottom - top + 4};
            }""", [start_id, end_id])
        if not box or box["height"] < 20:
            return False
        self.page.screenshot(path=str(path), clip=box)
        return True


def capture_flows(ui: Ui, shots: Path, api: str, flows_report: dict) -> dict:
    """One page load per flow: crop every stage, the summary, the whole flow; then the programme scorecard once."""
    from src import flows as fl
    fdir = shots / "flows"
    fdir.mkdir(parents=True, exist_ok=True)
    check = {}
    for f in flows_report["flows"]:
        fid = f["id"]
        ui.open(f"?flow={fid}")
        ui.tab("End-to-end flows")
        ui.fit()
        ids = [f"flowstep-{fid}-{s['key']}" for s in f["stages"]] + [f"flowsummary-{fid}", "flow-overall"]
        saved = []
        for s, nxt in zip(f["stages"], ids[1:]):
            if ui.crop(f"flowstep-{fid}-{s['key']}", nxt, fdir / f"{fid}_{s['key']}.png"):
                saved.append(s["key"])
        ui.crop(f"flowsummary-{fid}", "flow-overall", fdir / f"{fid}_summary.png")
        ui.crop("flow-top", "flow-overall", fdir / f"{fid}_full.png")
        if fid == "A1":
            ui.crop("flow-overall", "flow-end", fdir / "overall.png")
        text = ui.page.inner_text("body")
        check[fid] = {"stages_saved": saved, "verdict_on_screen": f["summary"]["verdict"] in text}
        # ---- the numbers on screen must be the API's numbers
        if f["kind"] == "cate":
            out = api_post(api, "/predict/cate", {"guest": fl.PERSONAS[f["persona"]], "outcome": "room_nights", "compare_all": False})
            check[fid]["api_value"] = f"{out['predicted_effect']:+.2f} nights"
            check[fid]["ui_matches_api"] = check[fid]["api_value"] in text and f"{out['percentile_among_guests']:.0f}th" in text
        elif f["kind"] == "policy":
            nc = int(f["new_cutoff"]) if "new_cutoff" in f else int(next(x for x in fl.FLOWS if x["id"] == fid)["new_cutoff"])
            perk = float(next(x for x in fl.FLOWS if x["id"] == fid)["perk_cost"])
            out = api_post(api, "/simulate/threshold", {"new_cutoff": nc, "perk_cost_per_member": perk})
            net = f"-${abs(out['net_value_usd']):,.0f}" if out["net_value_usd"] < 0 else f"${out['net_value_usd']:,.0f}"
            check[fid]["api_value"] = f"{out['guests_affected']:,} guests, net {net}"
            check[fid]["ui_matches_api"] = f"{out['guests_affected']:,}" in text and net in text
        elif fid == "A1":
            out = api_post(api, "/predict/late", {"outcome": "next_year_room_nights"})
            check[fid]["api_value"] = f"{out['estimate']:+.2f} nights"
            check[fid]["ui_matches_api"] = check[fid]["api_value"].split()[0] in text
        print(f"  flow {fid}: {len(saved)} stage shots, verdict shown={check[fid]['verdict_on_screen']}, "
              f"ui==api={check[fid].get('ui_matches_api', 'n/a')}", flush=True)
    return check


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="all", choices=["all", "flows"])
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--ui-port", type=int, default=8501)
    args = ap.parse_args()
    out = Path(args.outputs)
    shots = out / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)

    report_path = out / "metrics" / "flows_report.json"
    if not report_path.exists():
        from src import flows as fl
        fl.build_and_save(out)
    flows_report = json.loads(report_path.read_text(encoding="utf-8"))

    env = {**os.environ, "LOYALTY_OUTPUTS": str(out), "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.main:app", "--port", str(args.api_port)],
                           cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ui_proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "src/ui/app.py", "--server.port", str(args.ui_port),
                                "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                               cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report = {"mode": args.mode}
    t0 = time.time()
    try:
        api_url = f"http://127.0.0.1:{args.api_port}"
        wait_for(api_url + "/health")
        wait_for(f"http://127.0.0.1:{args.ui_port}/")
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport=BASE_VIEW, device_scale_factor=1.25)

            if args.mode == "all":
                # ---------------- FastAPI / Swagger UI
                page.goto(f"{api_url}/docs")
                page.wait_for_selector(".opblock-summary")
                page.screenshot(path=str(shots / "01_api_swagger_overview.png"), full_page=True)

                def try_endpoint(path: str, fname: str) -> None:
                    page.goto(f"{api_url}/docs")
                    page.wait_for_selector(".opblock-summary")
                    page.locator(".opblock-summary", has_text=path).first.click()
                    page.get_by_role("button", name="Try it out").click()
                    page.get_by_role("button", name="Execute").click()
                    page.wait_for_selector(".live-responses-table", timeout=60000)
                    page.wait_for_timeout(800)
                    page.locator(".opblock.is-open").first.screenshot(path=str(shots / fname))

                try_endpoint("/predict/late", "02_api_predict_late.png")
                try_endpoint("/predict/cate", "03_api_predict_cate.png")
                try_endpoint("/simulate/threshold", "04_api_simulate_threshold.png")

                # ---------------- Streamlit tabs (viewport grown to the content so nothing is cut off)
                ui = Ui(page, f"http://127.0.0.1:{args.ui_port}/")
                ui.open()
                tabs = [("Gold effect (RDD)", "10_ui_rdd_effect.png"), ("Validity checks", "11_ui_validity_checks.png"),
                        ("Guest explorer (CATE)", "12_ui_guest_explorer.png"), ("Threshold simulator", "14_ui_threshold_simulator.png"),
                        ("Model leaderboard", "15_ui_model_leaderboard.png")]
                for label, fname in tabs:
                    ui.tab(label)
                    ui.shot(shots / fname)
                    if label == "Guest explorer (CATE)":
                        try:    # second flow: switch persona to the corporate road warrior
                            page.locator('div[data-baseweb="select"]').first.click()
                            page.get_by_role("option", name="Corporate road warrior").click()
                            ui.idle()
                            ui.shot(shots / "13_ui_guest_explorer_corporate.png")
                        except Exception as exc:
                            print("persona switch skipped:", exc)
                ui.tab("End-to-end flows")
                ui.shot(shots / "16_ui_end_to_end_flows.png")

            # ---------------- end-to-end flows (per-stage crops)
            ui = Ui(page, f"http://127.0.0.1:{args.ui_port}/")
            report["flows"] = capture_flows(ui, shots, api_url, flows_report)
            browser.close()
        report["all_ui_matches_api"] = all(v.get("ui_matches_api", True) for v in report["flows"].values())
        report["all_verdicts_shown"] = all(v["verdict_on_screen"] for v in report["flows"].values())
        report["seconds"] = round(time.time() - t0)
        (shots / "capture_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("saved:", *sorted(str(q.relative_to(shots)) for q in shots.rglob("*.png")), sep="\n  ")
        print(f"UI == API for every flow: {report['all_ui_matches_api']} | verdicts on screen: {report['all_verdicts_shown']}")
    finally:
        for proc in (api, ui_proc):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
