"""Launch FastAPI + Streamlit, drive them with Playwright and save PNGs to outputs/screenshots/.

Usage:  python scripts/capture_screenshots.py [--outputs outputs] [--api-port 8000] [--ui-port 8501]
Requires:  python -m playwright install chromium
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wait_for(url: str, timeout: float = 90.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status < 500:
                    return
        except Exception:
            time.sleep(1.0)
    raise TimeoutError(f"{url} did not come up in {timeout}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--ui-port", type=int, default=8501)
    args = ap.parse_args()
    out = Path(args.outputs)
    shots = out / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "LOYALTY_OUTPUTS": str(out), "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.main:app", "--port", str(args.api_port)],
                           cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "src/ui/app.py", "--server.port", str(args.ui_port),
                           "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                          cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for(f"http://127.0.0.1:{args.api_port}/health")
        wait_for(f"http://127.0.0.1:{args.ui_port}/")
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=1.5)

            # ---------------- FastAPI / Swagger UI
            base = f"http://127.0.0.1:{args.api_port}"
            page.goto(f"{base}/docs")
            page.wait_for_selector(".opblock-summary")
            page.screenshot(path=str(shots / "01_api_swagger_overview.png"), full_page=True)

            def try_endpoint(path: str, fname: str) -> None:
                page.goto(f"{base}/docs")
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

            # ---------------- Streamlit
            page.goto(f"http://127.0.0.1:{args.ui_port}/")
            page.wait_for_selector("text=Does reaching Gold change how guests behave?", timeout=90000)
            tabs = [("Gold effect (RDD)", "10_ui_rdd_effect.png"), ("Validity checks", "11_ui_validity_checks.png"),
                    ("Guest explorer (CATE)", "12_ui_guest_explorer.png"),
                    ("Threshold simulator", "14_ui_threshold_simulator.png"), ("Model leaderboard", "15_ui_model_leaderboard.png")]
            def idle(timeout: int = 240000) -> None:
                """Streamlit disables the tab buttons while the script is running."""
                page.wait_for_timeout(1500)          # let the rerun start before we test for "idle"
                page.wait_for_function("() => !document.querySelector('[role=tab][disabled]')", timeout=timeout)
                page.wait_for_timeout(800)

            idle()
            for label, fname in tabs:
                idle()
                page.get_by_role("tab", name=label).click()
                idle()
                if label == "Guest explorer (CATE)":
                    page.screenshot(path=str(shots / fname), full_page=True)
                    # second flow: switch persona to the corporate road warrior
                    try:
                        page.locator('div[data-baseweb="select"]').first.click()
                        page.get_by_role("option", name="Corporate road warrior").click()
                        idle()
                        page.screenshot(path=str(shots / "13_ui_guest_explorer_corporate.png"), full_page=True)
                    except Exception as exc:  # persona switching is cosmetic; keep the first shot
                        print("persona switch skipped:", exc)
                else:
                    page.screenshot(path=str(shots / fname), full_page=True)
            browser.close()
        print("saved:", *sorted(p.name for p in shots.glob("*.png")), sep="\n  ")
    finally:
        for proc in (api, ui):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
