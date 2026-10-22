"""Retriever: hybrid similarity search + slot expansion. Tracks where each claim came from."""
from __future__ import annotations

from datetime import date, datetime

from .. import retrieval
from ..store import Store


def retrieve(store: Store, question: str, slot: str, q_scope: list[str], as_of: date, known_as_of: datetime,
             expansion: bool = True, k: int = 6) -> list[dict]:
    """Return evidence entries ``{claim_id, origin: [...], score}``."""
    ev: dict[str, dict] = {}
    for c, s in retrieval.hybrid_search(store, question, as_of, known_as_of, q_scope, k=k):
        ev[c.claim_id] = {"claim_id": c.claim_id, "origin": ["similarity"], "score": round(s, 4)}
    if expansion and slot != "none":
        for c in retrieval.expand_slot(store, slot, as_of, known_as_of, q_scope):
            e = ev.setdefault(c.claim_id, {"claim_id": c.claim_id, "origin": [], "score": 0.0})
            if "expansion" not in e["origin"]:
                e["origin"].append("expansion")
    if slot != "none":
        for aid in retrieval.attestation_ids(store, slot):
            ev.setdefault(aid, {"claim_id": aid, "origin": ["expansion"], "score": 0.0})
    return sorted(ev.values(), key=lambda e: (-e["score"], e["claim_id"]))
