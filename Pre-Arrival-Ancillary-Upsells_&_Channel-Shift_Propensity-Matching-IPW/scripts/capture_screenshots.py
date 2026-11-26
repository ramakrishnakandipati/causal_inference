"""Start FastAPI + Streamlit in the background and capture screenshots with headless Chromium."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.config import shot_dir  # noqa: E402
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


def main():
    from playwright.sync_api import sync_playwright
    out = shot_dir()
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
            # ---- Swagger docs
            pg.goto(API + "/docs"); pg.wait_for_selector(".opblock-summary")
            pg.wait_for_timeout(800)
            pg.screenshot(path=str(out / "api_swagger_docs.png"), full_page=True)
            # ---- /predict/cate executed through Swagger
            pg.click(".opblock-summary:has-text('/predict/cate')")
            pg.click("button:has-text('Try it out')")
            body = json.dumps({"guests": [PERSONAS["Expedia family, Suite, 60-day lead (iOS)"]]}, indent=2)
            ta = pg.locator(".opblock-body textarea").first
            ta.fill(body)
            pg.click("button.execute")
            pg.wait_for_selector(".live-responses-table .microlight", timeout=60000)
            pg.wait_for_timeout(800)
            pg.locator(".opblock.is-open").first.screenshot(path=str(out / "api_cate_response.png"))
            # ---- Streamlit tabs
            pg.goto(UI); pg.wait_for_selector("text=Observational bias", timeout=90000)
            pg.wait_for_selector("text=Naive over-statement", timeout=90000)
            pg.wait_for_timeout(2500)
            pg.screenshot(path=str(out / "streamlit_tab1_ate_balance.png"), full_page=True)
            for tab, text, fname in (("2 · CATE leaderboard", "CATE model leaderboard", "streamlit_tab2_cate_uplift.png"),
                                     ("3 · Guest simulator", "Predicted segment", "streamlit_tab3_guest_simulator.png"),
                                     ("4 · Campaign optimiser", "Guests contacted", "streamlit_tab4_campaign_optimizer.png")):
                pg.get_by_role("tab", name=tab).click()
                pg.wait_for_selector(f"text={text}", timeout=90000)
                pg.wait_for_timeout(3500)     # let Streamlit finish re-render (avoid faded stale frame)
                pg.screenshot(path=str(out / fname), full_page=True)
            br.close()
        for f in sorted(out.glob("*.png")):
            print(f.name, f.stat().st_size)
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
