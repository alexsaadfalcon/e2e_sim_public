"""Fixtures for the browser-driven UI journey tests.

All tests in this directory are marked ``browser`` (see the module-level marker
below) and therefore skip unless ``RUN_BROWSER=1`` (gate lives in tests/conftest.py,
alongside sionna/slow/gui). On top of that gate:
  * if Playwright or its browser binary is unavailable, the fixtures skip with a
    clear "run: playwright install chromium" message rather than erroring;
  * the pipeline-Run journey additionally skips if torch or the munich frames are
    absent, so the lighter journeys still run on a torch-free CI box.

Isolation: a fresh browser context (its own cookies/storage) per test, so journeys
never leak state into one another.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import pytest

# Make the sibling helper modules (_ui_server, _ui_screenplay, _ui_app) importable
# as top-level modules from the test file without a package install.
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from _ui_server import LiveDashServer  # noqa: E402
from _ui_screenplay import Actor  # noqa: E402

# NOTE: test modules in this directory carry a module-level ``pytestmark =
# pytest.mark.browser`` so the RUN_BROWSER gate in tests/conftest.py skips them by
# default. A module-level pytestmark binds during collection (before the gate's
# modifyitems hook runs); a pytestmark in *this* conftest would NOT apply to tests.


@pytest.fixture(scope="session")
def _playwright():
    pw_spec = importlib.util.find_spec("playwright")
    if pw_spec is None:
        pytest.skip("playwright not installed (pip install playwright)")
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    try:
        yield pw
    finally:
        pw.stop()


@pytest.fixture(scope="session")
def browser(_playwright):
    try:
        b = _playwright.chromium.launch(headless=True)
    except Exception as e:  # browser binary not installed
        pytest.skip(f"could not launch headless chromium ({e}); "
                    "run: playwright install chromium")
    try:
        yield b
    finally:
        b.close()


@pytest.fixture(scope="session")
def dash_server():
    """The live Dash app, served once for the whole browser session."""
    with LiveDashServer() as server:
        yield server


@pytest.fixture
def page(browser):
    """A fresh, isolated page (own context) per test."""
    context = browser.new_context()
    pg = context.new_page()
    pg.set_default_timeout(15000)
    try:
        yield pg
    finally:
        context.close()


@pytest.fixture
def actor(page):
    """The user who performs the journey."""
    return Actor("A user", page)


@pytest.fixture
def base_url(dash_server):
    return dash_server.url


@pytest.fixture
def run_capable():
    """True if the heavy pipeline-Run journey can actually execute here (needs torch
    and the precomputed munich frames). Lighter journeys don't need either."""
    if importlib.util.find_spec("torch") is None:
        return False
    frames = (pathlib.Path(__file__).resolve().parents[2]
              / "e2e" / "environment" / "sionna_sims" / "munich.pkl")
    return frames.exists()
