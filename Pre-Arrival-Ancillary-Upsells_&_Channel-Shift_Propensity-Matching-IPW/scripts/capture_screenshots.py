"""Start FastAPI + Streamlit in the background and capture screenshots with headless Chromium.

Writes: docs/screenshots/*.png (API + 4 dashboard tabs), outputs/screenshots/*.png (all tabs) and
outputs/screenshots/flows/<id>.png (one full-page shot per end-to-end flow). Also checks UI flow == API prediction.
Usage: python scripts/capture_screenshots.py [flows]   # "flows" re-shoots only the flows tab
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.config import out_dir, shot_dir  # noqa: E402
from src.serving import PERSONAS  # noqa: E402

API, UI = "http://127.0.0.1:8000", "http://127.0.0.1:8501"


def wait(url, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(1.5)
    raise RuntimeError(f"timeout waiting for {url}")


def api_post(path, body):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def fit_viewport(pg, width=1500):
    """Streamlit scrolls an inner container, so size the viewport to the rendered content height."""
    pg.set_viewport_size({"width": width, "height": 1000})
    pg.wait_for_timeout(600)
    h = pg.evaluate("() => Math.max(...[...document.querySelectorAll('[data-testid=stMain], section.main, [data-testid=stAppViewContainer]')].map(e => e.scrollHeight))")
    pg.set_viewport_size({"width": width, "height": int(h) + 60})
    pg.wait_for_timeout(1200)


def capture_flows(pg, fdir):
    flows = json.load(open(out_dir("metrics") / "flows.json", encoding="utf-8"))["flows"]
    pg.get_by_role("tab", name="5 · End-to-end flows").click()
    pg.wait_for_selector("text=Choose a flow", timeout=90000)
    checks = {}
    for f in flows:
        pg.get_by_text(f["short"], exact=True).first.click()
        pg.wait_for_selector(f"h3:has-text('{f['title']}')", timeout=90000)
        pg.wait_for_selector("text=Programme-level metrics", timeout=90000)
        pg.wait_for_timeout(3000)     # let Streamlit finish re-rendering (avoid faded stale frame)
        fit_viewport(pg)
        pg.screenshot(path=str(fdir / f"{f['id']}.png"), full_page=True)
        if f["type"] == "A":
            r = api_post("/predict/cate", {"guests": [f["api_payload"]]})["predictions"][0]
            checks[f["id"]] = abs(r["tau_ancillary"] - f["expect"]["tau_ancillary"]) < 1e-2 and abs(r["tau_direct"] - f["expect"]["tau_direct"]) < 1e-3
        print("flow", f["id"], (fdir / f"{f['id']}.png").stat().st_size)
    print("UI==API for guest flows:", checks)
    return checks


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    from playwright.sync_api import sync_playwright
    docs, outs = shot_dir(), out_dir("screenshots")
    fdir = outs / "flows"; fdir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    procs = [
        subprocess.Popen([py, "-m", "uvicorn", "src.api.main:app", "--port", "8000", "--log-level", "warning"], cwd=ROOT),
        subprocess.Popen([py, "-m", "streamlit", "run", "src/app/streamlit_app.py", "--server.port", "8501", "--server.headless", "true",
                          "--browser.gatherUsageStats", "false"], cwd=ROOT),
    ]
    try:
        wait(API + "/health"); wait(UI + "/_stcore/health")
        with sync_playwright() as p:
            br = p.chromium.launch()
            pg = br.new_page(viewport={"width": 1500, "height": 1000})
            if only != "flows":
                pg.goto(API + "/docs"); pg.wait_for_selector(".opblock-summary")
                pg.wait_for_timeout(800)
                pg.screenshot(path=str(docs / "api_swagger_docs.png"), full_page=True)
                pg.click(".opblock-summary:has-text('/predict/cate')")
                pg.click("button:has-text('Try it out')")
                body = json.dumps({"guests": [PERSONAS["Expedia family, Suite, 60-day lead (iOS)"]]}, indent=2)
                pg.locator(".opblock-body textarea").first.fill(body)
                pg.click("button.execute")
                pg.wait_for_selector(".live-responses-table .microlight", timeout=60000)
                pg.wait_for_timeout(800)
                pg.locator(".opblock.is-open").first.screenshot(path=str(docs / "api_cate_response.png"))
            pg.goto(UI); pg.wait_for_selector("text=Observational bias", timeout=90000)
            pg.wait_for_selector("text=Naive over-statement", timeout=90000)
            pg.wait_for_timeout(2500)
            if only != "flows":
                fit_viewport(pg)
                pg.screenshot(path=str(docs / "streamlit_tab1_ate_balance.png"), full_page=True)
                for tab, text, fname in (("2 · CATE leaderboard", "CATE model leaderboard", "streamlit_tab2_cate_uplift.png"),
                                         ("3 · Guest simulator", "Predicted segment", "streamlit_tab3_guest_simulator.png"),
                                         ("4 · Campaign optimiser", "Guests contacted", "streamlit_tab4_campaign_optimizer.png")):
                    pg.get_by_role("tab", name=tab).click()
                    pg.wait_for_selector(f"text={text}", timeout=90000)
                    pg.wait_for_timeout(3000)
                    fit_viewport(pg)
                    pg.screenshot(path=str(docs / fname), full_page=True)
                for f in docs.glob("streamlit_tab*.png"):     # same shots under outputs/screenshots
                    shutil.copy(f, outs / f.name)
            checks = capture_flows(pg, fdir)
            br.close()
        assert all(checks.values()), f"UI/API mismatch: {checks}"
    finally:
        for pr in procs:
            pr.terminate()
        for pr in procs:
            try:
                pr.wait(timeout=10)
            except Exception:
                pr.kill()


if __name__ == "__main__":
    main()
