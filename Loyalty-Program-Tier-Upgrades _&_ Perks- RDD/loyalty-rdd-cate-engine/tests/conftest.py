"""pytest configuration.

Known environment quirk: on this stack (xgboost 2.x CUDA + econml/joblib, Windows) the process can abort
natively (0xC0000409) while unloading DLLs *after* Python has finished, which swallows the pytest summary and
turns a green run into a failing exit code. All tests complete before that point, so after the summary is
printed we flush and terminate with pytest's real exit status, skipping the faulty DLL teardown.
Set LOYALTY_CLEAN_EXIT=0 to disable.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    # Objects that own CUDA/XGBoost handles are freed only at process end (and we skip that teardown below);
    # a mid-session collection of cyclic garbage holding such handles can hit the same native abort.
    if os.environ.get("LOYALTY_CLEAN_EXIT", "1") != "0":
        import gc
        gc.disable()


def pytest_sessionfinish(session, exitstatus):
    session.config._loyalty_exitstatus = int(exitstatus)


def pytest_unconfigure(config):
    if os.environ.get("LOYALTY_CLEAN_EXIT", "1") == "0":
        return
    status = getattr(config, "_loyalty_exitstatus", None)
    if status is None:          # collection-only or aborted runs: leave pytest alone
        return
    from src.utils.device import hard_exit
    hard_exit(status)
