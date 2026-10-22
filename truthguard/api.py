"""Thin service layer shared by the Streamlit UI, the MCP server and the eval harness."""
from __future__ import annotations

import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from . import attest as attest_mod
from . import config, health
from .agents.graph import Flags, run_question
from .ingest import pipeline
from .store import Store

_store: Optional[Store] = None
_lock = threading.Lock()


def get_store(seed_if_empty: bool = True) -> Store:
    """Process-wide store at ``TG_DB`` (default ``data/ledger.duckdb``); seeds the corpus on first use."""
    global _store
    with _lock:
        if _store is None:
            _store = Store()
        if seed_if_empty and not _store.list_documents():
            pipeline.ingest_seed(_store)
        return _store


def set_store(store: Optional[Store]) -> None:
    """Inject a store (tests, eval)."""
    global _store
    _store = store


def _d(v) -> Optional[date]:
    return date.fromisoformat(v[:10]) if isinstance(v, str) and v else (v or None)


def _dt(v) -> Optional[datetime]:
    return datetime.fromisoformat(v) if isinstance(v, str) and v else (v or None)


def verify(question: str, scope: str = "general", as_of: Optional[str] = None, known_as_of: Optional[str] = None,
           flags: Optional[Flags] = None) -> dict:
    """Answer a question; returns the ConflictCard as a JSON-able dict."""
    card = run_question(get_store(), question, [scope] if scope else None, _d(as_of), _dt(known_as_of), flags)
    return card.model_dump(mode="json")


def ingest(path: str) -> dict:
    """Ingest one markdown file; returns ``{commit, quarantined, reason, truth_diff, ...}``."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = config.INCOMING_DIR / p
    res = pipeline.ingest_document(get_store(), p)
    return {"doc_id": res["doc_id"], "commit": res["commit"], "quarantined": res["quarantined"],
            "reason": res["reason"], "claims": res["claims"], "truth_diff": res["truth_diff"],
            "attestation_changes": res["attestation_changes"], "skipped": res["skipped"]}


def attest(slot: str, value: str, statement: str, attester: str, role: str, expiry_days: int = 90,
           scope: str = "general", quorum: int = 1) -> dict:
    a = attest_mod.create_attestation(get_store(), slot, value, statement, attester, role, expiry_days, [scope],
                                      quorum)
    return a.model_dump(mode="json")


def health_report() -> dict:
    return health.health_report(get_store())
