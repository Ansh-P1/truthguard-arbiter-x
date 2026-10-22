"""Resolution: runs the deterministic engine and builds the ConflictCard.

The final answer text is templated from the verdict; an LLM may only polish wording (numbers must survive).
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from .. import attest, checker, engine, llm
from ..ingest import slots as slotmod
from ..ingest.extract import find_span
from ..models import ConflictCard, ProvenanceRow, Verdict
from ..store import Store
from .synthesizer import fmt_value

NOT_COVERED = "The documents do not cover this. No answer given."


class _Item:
    """Uniform view of a claim or an attestation for rendering."""

    def __init__(self, id_, doc_id, title, version, tier, kind, value, unit, scope, eff, rec, quote, target=None,
                 att=None):
        self.id, self.doc_id, self.title, self.version, self.tier, self.kind = id_, doc_id, title, version, tier, kind
        self.value, self.unit, self.scope, self.eff, self.rec, self.quote = value, unit, scope, eff, rec, quote
        self.target, self.att = target, att


def _item(store: Store, cid: str, atts: dict) -> Optional[_Item]:
    if cid in atts:
        a = atts[cid]
        return _Item(cid, cid, f"Attestation by {a.attester} ({a.role})", "-", 0, "value", a.value, a.unit, a.scope,
                     a.created_at.date(), a.created_at, a.statement, att=a)
    c = store.get_claim(cid)
    if not c:
        return None
    d = store.get_document(c.doc_id)
    return _Item(cid, c.doc_id, d.title if d else c.doc_id, d.version if d else "?", c.tier, c.kind, c.value, c.unit,
                 c.scope, c.effective_from, c.recorded_at, c.quote, c.target_doc)


def _polish(answer: str) -> str:
    """Optional LLM wording polish; discarded unless every number survives."""
    if llm.is_mock():
        return answer
    try:
        out = llm.chat("Rewrite for clarity in at most the same length. Keep every number, date and document name "
                       "exactly. Output only the text.\n\n" + answer).strip()
        if set(re.findall(r"\d+", answer)) <= set(re.findall(r"\d+", out)):
            return out
    except Exception:
        pass
    return answer


def build_card(store: Store, question: str, slot: str, q_scope: list[str], as_of: date, known_as_of: datetime,
               evidence: list[dict], draft: list[dict], attacks: list[dict], redrafts: int, trace: list[dict],
               rule_order=engine.FIXED_ORDER) -> ConflictCard:
    atts = {a.att_id: a for a in attest.list_attestations(store)}
    ev_ids = {e["claim_id"] for e in evidence}
    origin = {e["claim_id"]: e.get("origin", []) for e in evidence}
    notes: list[str] = []

    if slot == "none" or slot not in slotmod.SLOTS:
        verdict = Verdict(slot=slot, status="UNSUPPORTED")
    else:
        claims = [c for c in store.engine_claims(as_of, known_as_of, include_attestations=True)
                  if (c.id in ev_ids and c.slot == slot) or (c.kind == "withdrawal" and c.id in ev_ids)]
        verdict = engine.resolve(claims, slot, set(q_scope), as_of, known_as_of, rule_order)

    rows: list[ProvenanceRow] = []
    rule_trace: list[str] = []
    defeat_of = {d.claim: d for d in verdict.defeated}
    echo_of = {e: rep for rep, es in verdict.echoes.items() for e in es}

    def row(cid: str, role: str) -> Optional[ProvenanceRow]:
        it = _item(store, cid, atts)
        if not it:
            return None
        d = defeat_of.get(cid)
        by_doc = None
        if d:
            by_it = _item(store, d.by, atts)
            by_doc = by_it.doc_id if by_it else d.by
        return ProvenanceRow(
            claim_id=cid, doc_id=it.doc_id, title=it.title, version=it.version, tier=it.tier, kind=it.kind,
            value=it.value, unit=it.unit, scope=list(it.scope), effective_from=str(it.eff),
            recorded_at=str(it.rec), quote=it.quote, label=verdict.labels.get(cid, "UNDEC"), role=role,
            defeated_by=by_doc, rule=d.rule if d else None, echo_of=echo_of.get(cid),
            origin=origin.get(cid, []))

    order = []
    for cid in verdict.labels:
        it_kind = _item(store, cid, atts)
        role = ("echo" if cid in echo_of else "winner" if cid in verdict.winners else
                "contested" if cid in verdict.contested else
                "withdrawal" if it_kind and it_kind.kind == "withdrawal" else
                "defeated" if cid in defeat_of else "other")
        order.append((role, cid))
    rank = {"winner": 0, "contested": 1, "withdrawal": 2, "defeated": 3, "echo": 4, "other": 5}
    for role, cid in sorted(order, key=lambda t: (rank[t[0]], t[1])):
        r = row(cid, role)
        if r:
            rows.append(r)
    for d in verdict.defeated:
        loser, winner = _item(store, d.claim, atts), _item(store, d.by, atts)
        if loser and winner:
            rule_trace.append(f"{loser.doc_id} OUT: {d.rule} — {winner.doc_id}"
                              + (f" ({fmt_value(winner.value, winner.unit)})" if winner.kind == "value" else ""))
    if verdict.status == "CONTESTED":
        rule_trace.append("R5 unresolved tie: same authority tier, same scope, same effective date")

    ticket = None
    needs_att = False
    value = verdict.answer_value
    win = _item(store, verdict.winners[0], atts) if verdict.winners else None
    unit = win.unit if win else None

    if verdict.status == "SETTLED":
        answer = f'{fmt_value(value, unit)}. Source: {win.title} v{win.version}, effective {win.eff}: "{win.quote}"'
    elif verdict.status == "SUPERSEDED":
        lines = [f"{fmt_value(value, unit)} ({win.title} v{win.version})."]
        for d in verdict.defeated:
            loser = _item(store, d.claim, atts)
            by = _item(store, d.by, atts)
            if loser and loser.kind == "value":
                lines.append(f"Superseded: {loser.title} said {fmt_value(loser.value, loser.unit)} — {d.rule}"
                             + (f" ({by.title})." if by else "."))
        for rep, es in verdict.echoes.items():
            names = ", ".join(sorted({(_item(store, e, atts) or _Item(e, e, e, "", 0, "", None, None, [], None, None, "")).title
                                      for e in es}))
            r_it = _item(store, rep, atts)
            if names and r_it:
                lines.append(f"Also stated by: {names} (same value as {r_it.title}).")
        answer = "\n".join(lines)
    elif verdict.status == "CONTESTED":
        sides = [_item(store, i, atts) for i in verdict.contested]
        txt = "; ".join(f"{s.title} says {fmt_value(s.value, s.unit)}" for s in sides if s)
        ech = []
        for rep in verdict.contested:
            for e in verdict.echoes.get(rep, []):
                ei, ri = _item(store, e, atts), _item(store, rep, atts)
                if ei and ri:
                    ech.append(f"{ei.title} (agrees with {ri.title})")
        answer = f"Cannot resolve: {txt}; same authority and date."
        if ech:
            answer += f" Also stated by: {', '.join(ech)}."
        answer += " Escalated to policy owner."
        ticket = store.open_ticket(slot, f"Unresolvable conflict: {txt}")
    elif verdict.status == "ATTESTED":
        a = win.att
        answer = (f"{fmt_value(value, unit)} — confirmed by {a.role} ({a.attester}), not yet documented. "
                  f"Expires {a.expires_at.date()}.")
    else:
        answer = NOT_COVERED
        needs_att = slot in slotmod.SLOTS
        ticket = store.open_ticket(slot if slot in slotmod.SLOTS else "none",
                                   f"Unanswerable question: {question[:120]}")

    # proof certificate for the proposed answer
    certs: list[dict] = []
    if verdict.status in ("SETTLED", "SUPERSEDED", "ATTESTED") and win:
        prop = f"{slotmod.slot_description(slot)}: {fmt_value(value, unit)}"
        if win.att:
            quote_found = bool(win.quote)
        else:
            d = store.get_document(win.doc_id)
            quote_found = bool(d and find_span(d.body, win.quote))
        sup = checker.supported(prop, [win.quote])
        no_attack = verdict.labels.get(win.id) == "IN"
        certs.append({"sentence": prop, "claim_ids": [win.id], "quote_found": quote_found, "supported": sup,
                      "no_open_attack": no_attack})
        if not (quote_found and sup and no_attack):
            notes.append("proof certificate failed; the proposed answer is withheld in favour of the conflict view")
            answer = "Could not verify the proposed answer against its source; see the provenance table."
    if verdict.status in ("CONTESTED", "ATTESTED"):   # single-line answers only; numbers must survive
        answer = _polish(answer)

    return ConflictCard(
        question=question, slot=slot, scope=list(q_scope), as_of=str(as_of), known_as_of=known_as_of.isoformat(),
        status=verdict.status, answer=answer, value=value, unit=unit, provenance=rows, rule_trace=rule_trace,
        echoes=verdict.echoes, attacks=verdict.attacks, certificates=certs, ticket_id=ticket,
        needs_attestation=needs_att, draft=draft, critic_attacks=attacks, redrafts=redrafts, trace=trace,
        notes=notes)
