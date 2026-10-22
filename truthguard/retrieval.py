"""Hybrid retrieval (0.6 cosine + 0.4 BM25) and slot expansion."""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Iterable, Optional

from rank_bm25 import BM25Okapi

from . import attest, llm
from .ingest import slots as slotmod
from .models import Claim
from .store import Store

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOK.findall(text.lower())


def claim_text(c: Claim) -> str:
    return f"{c.quote} {slotmod.slot_description(c.slot)}"


def scoped(c: Claim, q_scope: Iterable[str]) -> bool:
    """Same scope test as the engine's ``in_play``: withdrawals always, values if general or scope overlaps."""
    return c.kind == "withdrawal" or "general" in c.scope or bool(set(c.scope) & set(q_scope))


def in_play_claims(store: Store, as_of: date, known_as_of: datetime, q_scope: Iterable[str]) -> list[Claim]:
    return [c for c in store.get_claims(as_of, known_as_of) if scoped(c, q_scope)]


def hybrid_search(store: Store, query: str, as_of: date, known_as_of: datetime, q_scope: Iterable[str],
                  k: int = 6, only_slot: Optional[str] = None) -> list[tuple[Claim, float]]:
    """Top-``k`` claims by 0.6*cosine + 0.4*BM25 (BM25 max-normalised)."""
    claims = in_play_claims(store, as_of, known_as_of, q_scope)
    if only_slot:
        claims = [c for c in claims if c.slot == only_slot or c.kind == "withdrawal"]
    if not claims:
        return []
    qv = llm.embed([query])[0]
    vecs = []
    missing = [i for i, c in enumerate(claims) if not c.embedding or len(c.embedding) != len(qv)]
    if missing:
        re_emb = llm.embed([claim_text(claims[i]) for i in missing])
        fix = dict(zip(missing, re_emb))
    else:
        fix = {}
    for i, c in enumerate(claims):
        vecs.append(fix.get(i) or c.embedding)
    cos = [max(0.0, llm.cosine(qv, v)) for v in vecs]
    bm = BM25Okapi([tokenize(claim_text(c)) for c in claims])
    raw = list(bm.get_scores(tokenize(query)))
    mx = max(raw) if raw and max(raw) > 0 else 1.0
    scored = [(c, 0.6 * cs + 0.4 * (max(0.0, b) / mx)) for c, cs, b in zip(claims, cos, raw)]
    scored.sort(key=lambda t: (-t[1], t[0].claim_id))
    return scored[:k]


def expand_slot(store: Store, slot: str, as_of: date, known_as_of: datetime,
                q_scope: Iterable[str]) -> list[Claim]:
    """Every in-play claim of the slot + 1-hop SUPERSEDES/WITHDRAWS neighbours (incl. withdrawal claims)."""
    play = in_play_claims(store, as_of, known_as_of, q_scope)
    by_id = {c.claim_id: c for c in play}
    slot_docs = {c.doc_id for c in play if c.slot == slot}
    picked: dict[str, Claim] = {c.claim_id: c for c in play if c.slot == slot}
    for c in play:                                   # withdrawal claims of the slot or aimed at the slot's docs
        if c.kind == "withdrawal" and (c.slot == slot or c.target_doc in slot_docs):
            picked[c.claim_id] = c
    for e in store.get_edges():
        if e.type == "SUPERSEDES":
            for cid in (e.src, e.dst):
                if cid in by_id and (e.src in picked or e.dst in picked):
                    picked[cid] = by_id[cid]
        elif e.type == "WITHDRAWS" and e.src in picked:
            for c in play:
                if c.doc_id == e.dst and c.kind == "value" and c.slot == slot:
                    picked[c.claim_id] = c
    return sorted(picked.values(), key=lambda c: c.claim_id)


def attestation_ids(store: Store, slot: str) -> list[str]:
    return [c.id for c in attest.engine_views(store) if c.slot == slot]
