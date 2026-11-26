"""Launch FastAPI + Streamlit, drive them with headless Chromium, save screenshots to outputs/screenshots/.

Run: python scripts/capture_screenshots.py      (needs: playwright install chromium)
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402
from playwright.async_api import async_playwright  # noqa: E402

from src.utils.logger import get_logger  # noqa: E402

log = get_logger("playwright", "playwright.log")
API, UI = "http://127.0.0.1:8000", "http://127.0.0.1:8501"
SHOTS = ROOT / "outputs" / "screenshots"
LOGS = ROOT / "outputs" / "logs"
VIEWPORT = {"width": 1440, "height": 1080}


def start(cmd: list[str], logfile: str) -> subprocess.Popen:
    LOGS.mkdir(parents=True, exist_ok=True)
    fh = open(LOGS / logfile, "ab")
    log.info("Launching: %s", " ".join(cmd))
    return subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONWARNINGS": "ignore"})


def wait_http(urls: list[str], timeout: float = 180.0) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout:
        for u in urls:
            try:
                if requests.get(u, timeout=3).status_code == 200:
                    log.info("Ready: %s (%.1fs)", u, time.time() - t0)
                    return u
            except Exception:  # noqa: BLE001
                pass
        time.sleep(1.5)
    raise TimeoutError(f"service not ready: {urls}")


async def swagger_flow(browser) -> None:
    (SHOTS / "api").mkdir(parents=True, exist_ok=True)
    page = await browser.new_page(viewport=VIEWPORT)
    await page.goto(f"{API}/docs", wait_until="networkidle")
    await page.wait_for_selector(".opblock-summary")
    await page.screenshot(path=str(SHOTS / "api" / "01_api_swagger_overview.png"), full_page=True)
    log.info("saved 01_api_swagger_overview.png")

    async def run(path_text: str, fname: str) -> None:
        block = page.locator(f".opblock:has([data-path='{path_text}'])").first
        await block.locator(".opblock-summary").click()
        await block.locator("button.try-out__btn").click()
        await block.locator("button.execute").click()
        await block.locator("td.response-col_status >> text='200'").first.wait_for(timeout=120000)
        await block.scroll_into_view_if_needed()
        await page.wait_for_timeout(800)
        await page.screenshot(path=str(SHOTS / "api" / fname), full_page=True)
        log.info("saved %s (HTTP 200 observed)", fname)
        await block.locator(".opblock-summary").click()   # collapse

    await run("/api/v1/optimize-price", "02_api_optimize_price_execution.png")
    await run("/api/v1/cate-benchmark", "03_api_cate_benchmark_execution.png")
    await page.close()


async def st_wait_idle(page, extra: int = 1500) -> None:
    await page.wait_for_load_state("networkidle")
    try:
        await page.wait_for_selector("[data-testid='stStatusWidget']", state="detached", timeout=60000)
    except Exception:  # noqa: BLE001
        pass
    await page.wait_for_timeout(extra)


async def streamlit_flow(browser) -> None:
    (SHOTS / "streamlit").mkdir(parents=True, exist_ok=True)
    # Streamlit scrolls an inner container, so use a tall viewport to capture the whole tab
    page = await browser.new_page(viewport={"width": 1440, "height": 2400})
    await page.goto(UI, wait_until="networkidle")
    await page.wait_for_selector("[data-testid='stSidebar']", timeout=120000)
    await st_wait_idle(page, 3000)

    async def scenario(letter: str) -> None:
        await page.locator("[data-testid='stSidebar'] label", has_text=f"{letter}:").first.click()
        await st_wait_idle(page, 2500)

    async def tab(idx: int, fname: str, charts: bool = True) -> None:
        await page.locator("button[role='tab']").nth(idx).click()
        await st_wait_idle(page, 2500)
        if charts:
            try:
                await page.wait_for_selector(".js-plotly-plot, [data-testid='stImage'] img, img", timeout=30000)
                # every <img> in the main area must be fully decoded (a broken/unloaded image has naturalWidth 0)
                await page.wait_for_function(
                    "() => Array.from(document.querySelectorAll('[data-testid=\"stImage\"] img'))"
                    ".every(i => i.complete && i.naturalWidth > 0)", timeout=45000)
            except Exception:  # noqa: BLE001
                log.warning("charts/images not fully loaded on tab %d", idx)
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(SHOTS / "streamlit" / fname), full_page=True)
        log.info("saved %s", fname)

    await scenario("A")
    await tab(0, "01_streamlit_eda_and_iv_diagnostics.png")
    await tab(1, "02_streamlit_dynamic_price_optimizer.png")
    await scenario("B")
    await tab(1, "02b_streamlit_scenario_b_segment_contrast.png")
    await tab(2, "03_streamlit_cate_metalearner_arena.png")
    await scenario("C")
    await tab(3, "04_streamlit_batch_crm_targeting.png")
    await scenario("C")
    await tab(1, "02c_streamlit_scenario_c_airline_cut.png")
    await scenario("D")
    await tab(1, "02d_streamlit_scenario_d_custom.png")
    await scenario("A")
    await tab(4, "05_streamlit_end_to_end_flows_elasticity.png", charts=False)
    for letter, fname in [("Flow 2", "05b_streamlit_flow_price_optimisation.png"), ("Flow 3", "05c_streamlit_flow_promo_cate.png"),
                          ("Flow 4", "05d_streamlit_flow_batch_crm.png")]:
        await page.locator("label", has_text=letter).first.click()
        await st_wait_idle(page, 2500)
        await page.screenshot(path=str(SHOTS / "streamlit" / fname), full_page=True)
        log.info("saved %s", fname)
    exc = await page.locator("[data-testid='stException']").count()
    log.info("Streamlit exception blocks on final page: %d", exc)
    await page.close()


async def main() -> None:
    api = ui = None
    try:
        api = start([sys.executable, "-m", "uvicorn", "src.api.main:app", "--port", "8000", "--log-level", "warning"], "api_server_process.log")
        ui = start([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", "8501",
                    "--server.headless", "true", "--browser.gatherUsageStats", "false"], "streamlit_process.log")
        wait_http([f"{API}/health"])
        wait_http([f"{UI}/_stcore/health", f"{UI}/healthz"])
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                await swagger_flow(browser)
                await streamlit_flow(browser)
            finally:
                await browser.close()
        log.info("All screenshots written to %s", SHOTS)
    finally:
        for name, proc in (("streamlit", ui), ("uvicorn", api)):
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                log.info("stopped %s", name)


if __name__ == "__main__":
    asyncio.run(main())
