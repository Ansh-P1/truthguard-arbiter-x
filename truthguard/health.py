"""Corpus health report: contested slots, quarantined docs, doc debt, attestations nearing expiry."""
from __future__ import annotations

from datetime import timedelta

from . import attest, clock, config
from .ingest import slots as slotmod
from .store import Store
from .verdicts import resolve_slot


def health_report(store: Store, expiry_warning_days: int = 14) -> dict:
    """Snapshot of corpus health at the current clock."""
    contested, unsupported = [], []
    for slot in slotmod.slot_ids():
        for scope in config.SCOPE_KEYS:
            v = resolve_slot(store, slot, [scope])
            if v.status == "CONTESTED":
                contested.append({"slot": slot, "scope": scope, "sides": v.contested})
            elif v.status == "UNSUPPORTED" and scope == "general":
                unsupported.append({"slot": slot, "scope": scope})
    now = clock.now()
    atts = attest.list_attestations(store)
    expiring = [{"att_id": a.att_id, "slot": a.slot, "expires_at": a.expires_at.isoformat(),
                 "days_left": (a.expires_at - now).days}
                for a in atts if a.status == "active" and a.expires_at - now <= timedelta(days=expiry_warning_days)]
    docs = store.list_documents()
    return {
        "as_of": str(clock.today()), "documents": len(docs),
        "claims": len(store.all_claims()),
        "contested_slots": contested,
        "uncovered_slots": unsupported,
        "quarantined_docs": [{"doc_id": d.doc_id, "reason": d.quarantine_reason} for d in docs if d.quarantined],
        "open_doc_debt": store.tickets("open"),
        "attestations": [{"att_id": a.att_id, "slot": a.slot, "value": a.value, "status": a.status,
                          "expires_at": a.expires_at.isoformat()} for a in atts],
        "attestations_expiring_soon": expiring,
    }
