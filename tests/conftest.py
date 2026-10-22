import os
import sys
from pathlib import Path

os.environ["TG_LLM"] = "mock"          # deterministic offline mode for tests
os.environ["TG_TODAY"] = "2026-10-19"
os.environ["TG_NOW"] = "2026-10-19T12:00:00"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from truthguard import config  # noqa: E402
from truthguard.ingest import pipeline  # noqa: E402
from truthguard.store import Store  # noqa: E402


@pytest.fixture(autouse=True)
def _clock(monkeypatch):
    monkeypatch.setenv("TG_LLM", "mock")
    monkeypatch.setenv("TG_TODAY", "2026-10-19")
    monkeypatch.setenv("TG_NOW", "2026-10-19T12:00:00")


@pytest.fixture()
def store():
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture()
def seeded(store, monkeypatch):
    """Store with the seed corpus ingested (rule-based mock extraction)."""
    pipeline.ingest_seed(store)
    return store


@pytest.fixture()
def after_resc(seeded, monkeypatch):
    """Seed + RESC recorded on 2026-03-01T10:00."""
    monkeypatch.setenv("TG_NOW", "2026-03-01T10:00:00")
    res = pipeline.ingest_document(seeded, config.INCOMING_DIR / "RESC.md")
    monkeypatch.setenv("TG_NOW", "2026-10-19T12:00:00")
    seeded._resc_result = res
    return seeded
