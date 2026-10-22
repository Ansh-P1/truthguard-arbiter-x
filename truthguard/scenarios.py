"""Reusable corpus states for tests, eval and the demo (built in-memory from corpus/)."""
from __future__ import annotations

import contextlib
import json
import os
from typing import Iterator, Optional

from . import attest, config
from .ingest import pipeline
from .store import Store

RESC_RECORDED = "2026-03-01T10:00:00"
ATTEST_AT = "2026-10-19T12:00:00"
STATES = ("base", "after_resc", "after_vendor", "attested")


@contextlib.contextmanager
def clock_at(now_iso: Optional[str]) -> Iterator[None]:
    """Temporarily set ``TG_NOW`` (None = leave unchanged)."""
    old = os.environ.get("TG_NOW")
    if now_iso:
        os.environ["TG_NOW"] = now_iso
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TG_NOW", None)
        else:
            os.environ["TG_NOW"] = old


def build_state(name: str, store: Optional[Store] = None) -> Store:
    """Build one of the named states in a fresh in-memory store.

    * ``base``         seed corpus only
    * ``after_resc``   + RESC recorded 2026-03-01T10:00
    * ``after_vendor`` + VENDOR upload attempt (quarantined)
    * ``attested``     + "24 hours" attestation by the MNP Ops Lead (expires in 90 days)
    """
    if name not in STATES:
        raise ValueError(name)
    s = store or Store(":memory:")
    pipeline.ingest_seed(s)
    if name in ("after_resc", "after_vendor", "attested"):
        with clock_at(RESC_RECORDED):
            pipeline.ingest_document(s, config.INCOMING_DIR / "RESC.md")
    if name == "after_vendor":
        with clock_at("2026-10-19T11:00:00"):
            pipeline.ingest_document(s, config.INCOMING_DIR / "VENDOR.md")
    if name == "attested":
        with clock_at(ATTEST_AT):
            attest.create_attestation(s, "mnp.portout_cancel_window", "24",
                                      "A port-out request can be cancelled within 24 hours of submission.",
                                      "MNP Ops Lead", "MNP Ops Lead", expiry_days=90)
    return s


def load_questions() -> list[dict]:
    return json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))
