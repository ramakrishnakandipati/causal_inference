import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def lat(cfg):
    from src.data.generate_panel import build_latent

    return build_latent(cfg, seed=11)


@pytest.fixture(scope="session")
def panel(cfg, lat):
    from src.data.generate_panel import assemble_panel, scenario_specs

    return assemble_panel(lat, scenario_specs(cfg)["mia_flagship"], 11)
