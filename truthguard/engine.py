"""Deterministic resolution engine. Pure Python: no I/O, no LLM.

Precedence cascade (authority before specificity) feeds an abstract argumentation
framework; the grounded extension labels each claim IN / OUT / UNDEC.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from .models import Defeat, Verdict

RULE_NAMES = {
    "R2": "R2 lex superior (higher authority)",
    "R3": "R3 lex specialis (narrower scope, same tier)",
    "R4": "R4 lex posterior (later date)",
}
FIXED_ORDER = ("R2", "R3", "R4")
OLD_ORDER = ("R3", "R2", "R4")  # the original deck's (wrong) order; kept for the ablation


@dataclass(frozen=True)
class C:
    """Engine-internal claim view."""
    id: str
    doc_id: str
    slot: str
    kind: str  # "value" | "withdrawal"
    value: Optional[str]
    scope: frozenset
    tier: int
    eff: date
    rec: datetime
    replaces: frozenset = frozenset()
    target_doc: Optional[str] = None
    is_attestation: bool = False


def in_play(c: C, q_scope: frozenset, as_of: date, known_as_of: datetime) -> bool:
    """Bitemporal + scope filter."""
    if c.eff > as_of or c.rec > known_as_of:
        return False
    if c.kind == "value" and "general" not in c.scope and not (c.scope & q_scope):
        return False
    return True


_NUM = re.compile(r"^(\d+(?:\.\d+)?)\s*(?:days?|d|hours?|hrs?|h)?$")


def norm(v) -> str:
    """Normalise a value: ``"45 days" == "45"``; otherwise lowercase + collapsed whitespace."""
    s = " ".join(str(v).strip().lower().split())
    m = _NUM.match(s)
    if m:
        n = m.group(1)
        return n[:-2] if n.endswith(".0") else n
    return s


def specific(c: C) -> bool:
    return "general" not in c.scope


def collapse_echoes(vals):
    """Group by (value, scope); representative = highest (tier, eff). Others are echoes."""
    groups: dict = {}
    for c in vals:
        groups.setdefault((norm(c.value), c.scope), []).append(c)
    reps, echoes_of = [], {}
    for members in groups.values():
        rep = max(members, key=lambda m: (m.tier, m.eff, m.id))
        reps.append(rep)
        echoes_of[rep.id] = [m for m in members if m.id != rep.id]
    return reps, echoes_of


def _pair_rule(a: C, b: C, order):
    """First applicable rule in ``order`` -> (winner, loser, rule text) or None for a tie."""
    for r in order:
        if r == "R2" and a.tier != b.tier:
            w, lo = (a, b) if a.tier > b.tier else (b, a)
            return w, lo, RULE_NAMES["R2"]
        if r == "R3" and specific(a) != specific(b):
            w, lo = (a, b) if specific(a) else (b, a)
            return w, lo, RULE_NAMES["R3"]
        if r == "R4" and a.eff != b.eff:
            w, lo = (a, b) if a.eff > b.eff else (b, a)
            return w, lo, RULE_NAMES["R4"]
    return None


def derive_attacks(play, order=FIXED_ORDER):
    by_doc: dict = {}
    for c in play:
        by_doc.setdefault(c.doc_id, []).append(c)
    vals = [c for c in play if c.kind == "value"]
    reps, echoes_of = collapse_echoes(vals)
    nodes = reps + [c for c in play if c.kind == "withdrawal"]
    ids = {c.id for c in nodes}
    attacks: set = set()
    why: dict = {}

    def add(a, b, rule):
        if a.id in ids and b.id in ids and a.id != b.id and (a.id, b.id) not in attacks:
            attacks.add((a.id, b.id))
            why[(a.id, b.id)] = rule

    # R1 declared
    for c in nodes:
        if c.kind == "withdrawal":
            for t in by_doc.get(c.target_doc, []):
                add(c, t, "R1 withdrawal (declared)")
        for d in c.replaces:
            for t in by_doc.get(d, []):
                if t.kind == "value" and t.slot == c.slot:
                    add(c, t, "R1 explicit supersession (declared)")
    # R2-R5 among representatives with different values
    for i, a in enumerate(reps):
        for b in reps[i + 1:]:
            if a.slot != b.slot or norm(a.value) == norm(b.value):
                continue
            if (a.id, b.id) in attacks or (b.id, a.id) in attacks:
                continue
            res = _pair_rule(a, b, order)
            if res:
                add(res[0], res[1], res[2])
            else:
                add(a, b, "R5 unresolved tie")
                add(b, a, "R5 unresolved tie")
    return nodes, attacks, why, echoes_of


def grounded(node_ids, attacks):
    """Grounded extension: returns (IN, OUT, UNDEC)."""
    inn: set = set()
    out: set = set()
    changed = True
    while changed:
        changed = False
        for n in node_ids:
            if n in inn or n in out:
                continue
            if all(a in out for (a, b) in attacks if b == n):
                inn.add(n)
                changed = True
        for n in node_ids:
            if n not in out and any(a in inn for (a, b) in attacks if b == n):
                out.add(n)
                changed = True
    return inn, out, set(node_ids) - inn - out


def resolve(claims, slot: str, q_scope, as_of: date, known_as_of: datetime,
            rule_order=FIXED_ORDER) -> Verdict:
    """Resolve a slot. ``claims`` are engine ``C`` views (any slots; filtered here)."""
    q_scope = frozenset(q_scope)
    relevant = [c for c in claims if c.slot == slot or c.kind == "withdrawal"]
    play = sorted((c for c in relevant if in_play(c, q_scope, as_of, known_as_of)), key=lambda c: c.id)
    value_claims = [c for c in play if c.kind == "value" and c.slot == slot]
    if not value_claims:
        return Verdict(slot=slot, status="UNSUPPORTED")
    # withdrawals of other slots only matter through their target doc; keep them (they attack by doc)
    nodes, attacks, why, echoes_of = derive_attacks(play, rule_order)
    node_ids = [c.id for c in nodes]
    inn, out, undec = grounded(node_ids, attacks)
    by_id = {c.id: c for c in nodes}

    labels: dict = {}
    for n in node_ids:
        labels[n] = "IN" if n in inn else "OUT" if n in out else "UNDEC"
    for rep_id, ech in echoes_of.items():
        for e in ech:
            labels[e.id] = labels[rep_id]

    defeated = []
    for n in node_ids:
        if n in out:
            att = sorted(a for (a, b) in attacks if b == n and a in inn)
            if att:
                defeated.append(Defeat(claim=n, by=att[0], rule=why[(att[0], n)]))

    in_vals = [n for n in node_ids if n in inn and by_id[n].kind == "value"]
    undec_vals = [n for n in node_ids if n in undec and by_id[n].kind == "value"]
    out_vals = [n for n in node_ids if n in out and by_id[n].kind == "value"]

    if in_vals:
        winner = by_id[in_vals[0]]
        if winner.is_attestation:
            status = "ATTESTED"
        elif out_vals:
            status = "SUPERSEDED"
        else:
            status = "SETTLED"
        answer = winner.value
    elif len(undec_vals) >= 2:
        status, answer = "CONTESTED", None
    else:
        status, answer = "UNSUPPORTED", None

    return Verdict(
        slot=slot, status=status, answer_value=answer,
        winners=in_vals, defeated=defeated,
        contested=undec_vals if status == "CONTESTED" else [],
        echoes={k: [e.id for e in v] for k, v in echoes_of.items() if v},
        attacks=[{"src": a, "dst": b, "rule": why[(a, b)]} for (a, b) in sorted(attacks)],
        labels=labels,
    )
