"""Attestation lifecycle: an expert vouches for an undocumented fact, with an expiry date."""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

from . import clock
from .engine import C
from .ingest import slots as slotmod
from .models import Attestation
from .store import Store


def effective_status(a: Attestation) -> str:
    """Stored status, with expiry computed against the current clock (reversible: nothing is persisted)."""
    if a.status == "active" and a.expires_at <= clock.now():
        return "expired"
    return a.status


def list_attestations(store: Store) -> list[Attestation]:
    out = []
    for a in store.list_attestations():
        a.status = effective_status(a)
        out.append(a)
    return out


def create_attestation(store: Store, slot: str, value: str, statement: str, attester: str, role: str,
                       expiry_days: int = 90, scope: Optional[list[str]] = None, quorum: int = 1,
                       unit: Optional[str] = None) -> Attestation:
    """Create an attestation (tier 0). With ``quorum`` > 1 it stays ``pending`` until enough approvals."""
    if slot not in slotmod.SLOTS:
        raise ValueError(f"unknown slot {slot!r}")
    now = clock.now()
    n = len(store.list_attestations()) + 1
    a = Attestation(
        att_id=f"ATT-{n:04d}", slot=slot, scope=scope or ["general"], value=str(value),
        unit=unit if unit is not None else slotmod.slot_unit(slot),
        statement=statement or f"{role} {attester} confirms {slot} = {value}.",
        attester=attester, role=role, created_at=now, expires_at=now + timedelta(days=int(expiry_days)),
        status="active" if quorum <= 1 else "pending", quorum_required=quorum, approvals=[attester])
    store.save_attestation(a)
    recheck(store, [slot])
    return _get(store, a.att_id)


def _get(store: Store, att_id: str) -> Attestation:
    for a in list_attestations(store):
        if a.att_id == att_id:
            return a
    raise KeyError(att_id)


def approve(store: Store, att_id: str, approver: str) -> Attestation:
    """Add an approval; a ``pending`` attestation becomes ``active`` once the quorum is met."""
    a = _get(store, att_id)
    if approver not in a.approvals:
        a.approvals.append(approver)
    if a.status == "pending" and len(a.approvals) >= a.quorum_required:
        a.status = "active"
    store.save_attestation(a)
    return _get(store, att_id)


def dispute(store: Store, att_id: str, by: str) -> Attestation:
    a = _get(store, att_id)
    a.status = "disputed"
    a.statement += f" [disputed by {by}]"
    store.save_attestation(a)
    return _get(store, att_id)


def engine_views(store: Store) -> list[C]:
    """Active, unexpired attestations as engine claims (tier 0, ``is_attestation``)."""
    out = []
    for a in store.list_attestations():
        if effective_status(a) != "active":
            continue
        out.append(C(id=a.att_id, doc_id=a.att_id, slot=a.slot, kind="value", value=a.value,
                     scope=frozenset(a.scope), tier=0, eff=a.created_at.date(), rec=a.created_at,
                     is_attestation=True))
    return out


def recheck(store: Store, slots: list[str]) -> list[dict]:
    """After ingest: if a document claim now echoes or defeats an attestation, update its status."""
    from .verdicts import resolve_slot  # local import: avoid cycle

    changes = []
    for a in store.list_attestations():
        if a.slot not in slots or a.status != "active" or effective_status(a) != "active":
            continue
        v = resolve_slot(store, a.slot, a.scope)
        echoed = any(a.att_id in ids for ids in v.echoes.values())
        if echoed:
            new = "documented"
        elif v.labels.get(a.att_id) == "OUT":
            new = "contradicted"
        else:
            continue
        a.status = new
        store.save_attestation(a)
        changes.append({"att_id": a.att_id, "slot": a.slot, "status": new})
    return changes
