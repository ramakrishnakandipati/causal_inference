"""End-to-end check of FastAPI + Streamlit with Playwright; saves screenshots to outputs/screenshots.

    python tests/verify_e2e_playwright.py

Starts uvicorn (:8000) and streamlit (:8501) itself, asserts that the dashboard numbers equal the API
numbers for the three presets, and always shuts both servers down.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.api.presets import PRESETS  # noqa: E402

SHOTS = ROOT / "outputs" / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)
API, UI = "http://127.0.0.1:8000", "http://127.0.0.1:8501"
SIM_KEYS = ["baseline_direct_bookings_day", "ota_to_direct_ratio", "lastclick_share", "cannibalization_rate",
            "ota_capture_rate", "diversion_rate", "adr", "commission_rate", "cpc"]


def wait_for(url, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"timeout waiting for {url}")


def settle(page):
    page.wait_for_function("() => document.querySelectorAll('[data-stale=\"true\"]').length === 0", timeout=60000)
    page.wait_for_timeout(600)


def main():
    py = sys.executable
    procs = [subprocess.Popen([py, "-m", "uvicorn", "src.api.main:app", "--port", "8000"], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
             subprocess.Popen([py, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", "8501",
                               "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    failures = []
    try:
        wait_for(f"{API}/health")
        wait_for(f"{UI}/_stcore/health")
        # ---------------- API checks
        h = requests.get(f"{API}/health").json()
        assert h["status"] == "ok", h
        print("health:", h["hardware"])
        ms = requests.get(f"{API}/metrics/summary").json()
        assert ms["cate_leaderboard"]["direct"] and ms["macro_geolift"]["direct_bookings"]["SDID"]
        cate = requests.post(f"{API}/predict/dma-cate", json={"tier": 2}).json()
        assert "decision" in cate and len(cate["per_model"]["direct"]) >= 9, cate
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            br = pw.chromium.launch()
            ctx = br.new_context(viewport={"width": 1500, "height": 1000})
            page = ctx.new_page()
            # ---- Swagger
            page.goto(f"{API}/docs")
            page.wait_for_selector(".opblock-summary")
            page.screenshot(path=str(SHOTS / "01_fastapi_swagger.png"), full_page=True)
            page.locator('.opblock-summary-path', has_text='/simulate/holdout-policy').click()
            page.click("button:has-text('Try it out')")
            page.click("button:has-text('Execute')")
            page.wait_for_selector("text=Server response")
            page.screenshot(path=str(SHOTS / "02_fastapi_simulate_executed.png"), full_page=True)
            # ---- Streamlit
            page.goto(UI)
            page.wait_for_selector("text=Paid Search Cannibalization vs. Incrementality", timeout=60000)
            settle(page)
            page.screenshot(path=str(SHOTS / "03_streamlit_tab1_geolift.png"), full_page=True)
            page.get_by_role("tab", name=re.compile("2\\. CATE")).click()
            page.wait_for_selector("text=Out-of-fold, DMA-grouped"); settle(page)
            page.screenshot(path=str(SHOTS / "04_streamlit_tab2_leaderboard.png"), full_page=True)
            page.get_by_role("tab", name=re.compile("3\\. Policy")).click()
            for i, (k, pre) in enumerate(PRESETS.items()):
                page.get_by_role("button", name=re.compile(f"Preset {k}:")).click()
                page.wait_for_selector(f"text=Scenario: {pre['label']}"); settle(page)
                api = requests.post(f"{API}/simulate/holdout-policy", json={x: pre[x] for x in SIM_KEYS}).json()
                body = page.inner_text("body")
                ui_dec = "Pause" if "Policy decision: Pause" in body else "Keep"
                api_dec = "Pause" if api["decision"].startswith("Pause") else "Keep"
                metric = page.locator('[data-testid="stMetric"]', has_text="Net margin change").inner_text()
                ui_val = float(re.search(r"\$\s*(-?[\d,]+)", metric.replace("−", "-")).group(1).replace(",", ""))
                ok = ui_dec == api_dec and abs(ui_val - api["net_margin_change_if_paused_usd"]) <= 1.0
                print(f"preset {k}: UI {ui_dec} {ui_val:,.0f} | API {api_dec} {api['net_margin_change_if_paused_usd']:,.0f} -> {'OK' if ok else 'MISMATCH'}")
                if not ok:
                    failures.append(f"preset {k}")
                page.screenshot(path=str(SHOTS / f"0{5 + i}_streamlit_tab3_preset_{k}.png"), full_page=True)
            page.get_by_role("tab", name=re.compile("4\\. Refutation")).click()
            page.wait_for_selector("text=Optimal bidding policy by DMA"); settle(page)
            page.screenshot(path=str(SHOTS / "08_streamlit_tab4_policy_refutation.png"), full_page=True)
            br.close()
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
    bad = [p.name for p in SHOTS.glob("*.png") if p.stat().st_size == 0]
    if failures or bad:
        print("FAILED:", failures, bad)
        sys.exit(1)
    print("E2E OK;", len(list(SHOTS.glob('*.png'))), "screenshots in", SHOTS)


if __name__ == "__main__":
    main()
