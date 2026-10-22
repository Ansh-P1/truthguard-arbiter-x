"""Truth Diff: recompute the affected slots, diff against the verdict cache, update the cache."""
from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Optional

from . import clock, config
from .models import Claim, Verdict
from .store import Store
from .verdicts import resolve_slot


def affected_slots(store: Store, claims: Iterable[Claim]) -> list[str]:
    """Slots of the new claims, plus every slot of any withdrawn doc."""
    slots: list[str] = []
    for c in claims:
        if c.slot not in slots:
            slots.append(c.slot)
        if c.kind == "withdrawal" and c.target_doc:
            for tc in store.get_claims_raw(c.target_doc):
                if tc.slot not in slots:
                    slots.append(tc.slot)
    return slots


def _deciding_rule(before: Optional[dict], after: Verdict) -> Optional[str]:
    if before and before.get("winners"):
        for d in after.defeated:
            if d.claim in before["winners"] and d.claim not in after.winners:
                return d.rule
    if after.defeated:
        r1 = [d for d in after.defeated if d.rule.startswith("R1")]
        return (r1 or after.defeated)[0].rule
    return None


def _before_without(store: Store, slot: str, scope: str, as_of: date, known_as_of: datetime, doc_id: str) -> dict:
    """Verdict as it stood before ``doc_id`` arrived (used when the cache has no entry for this date)."""
    from . import engine
    claims = [c for c in store.engine_claims(as_of, known_as_of) if c.doc_id != doc_id]
    return engine.resolve(claims, slot, {scope}, as_of, known_as_of).model_dump()


def compute_diff(store: Store, slots: Iterable[str], as_of: Optional[date] = None,
                 known_as_of: Optional[datetime] = None, exclude_doc: Optional[str] = None) -> list[dict]:
    """Recompute verdicts for every scope key of ``slots`` and diff against the cache.

    On a cache miss (e.g. the corpus was last touched on another day) ``exclude_doc`` lets us rebuild the
    "before" verdict by resolving without the newly ingested document.
    """
    as_of = as_of or clock.today()
    known_as_of = known_as_of or clock.now()
    rows = []
    for slot in slots:
        for scope in config.SCOPE_KEYS:
            after = resolve_slot(store, slot, [scope], as_of, known_as_of)
            prev = store.cache_get(slot, scope, as_of)
            if prev is None and exclude_doc:
                prev = _before_without(store, slot, scope, as_of, known_as_of, exclude_doc)
                if prev["status"] == "UNSUPPORTED" and not prev["winners"]:
                    prev = None
            b = {"status": prev["status"], "value": prev.get("answer_value")} if prev else None
            a = {"status": after.status, "value": after.answer_value}
            changed = (b != a) if b else (after.status != "UNSUPPORTED")
            rows.append({"slot": slot, "scope": scope, "before": b, "after": a, "changed": changed,
                         "deciding_rule": _deciding_rule(prev, after) if changed else None})
            store.cache_put(slot, scope, as_of, after.status, after.model_dump())
    return rows
