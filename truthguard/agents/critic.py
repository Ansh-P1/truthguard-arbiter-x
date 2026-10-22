"""Critic: three attackers (contradiction, staleness, unsupported) + falsification retrieval.

The critic never edits the draft; it only emits attacks (and extra evidence found by falsification).
"""
from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from .. import checker, llm, retrieval
from ..engine import norm
from ..ingest import slots as slotmod
from ..store import Store


class Queries(BaseModel):
    queries: list[str] = Field(default_factory=list)


def falsification_queries(question: str, draft_text: str, slot: str) -> list[str]:
    desc = slotmod.slot_description(slot).lower() if slot in slotmod.SLOTS else question
    prompt = ("Write 3 short search queries that would DISPROVE this draft answer (e.g. was the source withdrawn, "
              "is there a newer version, is there an exception).\n"
              f"QUESTION: {question}\nDRAFT: {draft_text}\nReturn JSON {{\"queries\": [..3 strings..]}}")
    mock = {"queries": [f"{desc} withdrawn rescinded notice", f"newer {desc} amendment update",
                        f"{desc} replaced superseded"]}
    try:
        q = llm.chat_json(prompt, Queries, mock=lambda: mock, fallback=Queries(**mock)).queries[:3]
    except Exception:
        q = mock["queries"]
    return q or mock["queries"]


def critique(store: Store, question: str, slot: str, q_scope: list[str], as_of: date, known_as_of: datetime,
             draft: list[dict], evidence: list[dict]) -> dict:
    """Return ``{attacks, new_evidence, queries, needs_redraft}``."""
    attacks: list[dict] = []
    play = {c.claim_id: c for c in retrieval.in_play_claims(store, as_of, known_as_of, q_scope)}
    sup = [e for e in store.get_edges("SUPERSEDES") if e.src in play]
    wd = [e for e in store.get_edges("WITHDRAWS") if e.src in play]
    ev_ids = {e["claim_id"] for e in evidence}

    # 4. falsification first, so the other attackers see the enlarged evidence
    draft_text = " ".join(s["text"] for s in draft)
    queries = falsification_queries(question, draft_text, slot) if slot != "none" else []
    new_evidence = []
    for qtext in queries:
        for c, score in retrieval.hybrid_search(store, qtext, as_of, known_as_of, q_scope, k=3):
            if c.claim_id not in ev_ids and (c.slot == slot or c.kind == "withdrawal"):
                ev_ids.add(c.claim_id)
                new_evidence.append({"claim_id": c.claim_id, "origin": ["falsification"],
                                     "score": round(score, 4), "found_by_falsification": qtext})

    # 1. contradiction: claim pairs with different values in the slot
    slot_claims = [play[i] for i in ev_ids if i in play and play[i].slot == slot and play[i].kind == "value"]
    for i, a in enumerate(slot_claims):
        for b in slot_claims[i + 1:]:
            if norm(a.value) != norm(b.value):
                attacks.append({"type": "CONTRADICTS", "claim_ids": [a.claim_id, b.claim_id],
                                "detail": f"{a.doc_id} says {a.value}; {b.doc_id} says {b.value}"})

    # 2/3. staleness + support, per draft sentence
    for idx, s in enumerate(draft):
        cited = [store.get_claim(i) for i in s["claim_ids"]]
        cited = [c for c in cited if c]
        for c in cited:
            why = None
            for e in sup:
                if e.dst == c.claim_id:
                    why = f"superseded by {play[e.src].doc_id}"
            for e in wd:
                if e.dst == c.doc_id:
                    why = f"{c.doc_id} withdrawn by {play[e.src].doc_id}"
            if not why:
                for o in play.values():
                    if (o.slot == c.slot and o.kind == "value" and o.tier == c.tier and o.effective_from > c.effective_from
                            and norm(o.value) != norm(c.value) and (("general" in o.scope) == ("general" in c.scope))):
                        why = f"newer same-tier claim in {o.doc_id}"
            if why:
                attacks.append({"type": "STALE", "claim_ids": [c.claim_id], "sentence": idx, "detail": why})
        if not cited:
            attacks.append({"type": "UNSUPPORTED", "claim_ids": [], "sentence": idx, "detail": "no citations"})
        elif not checker.supported(s["text"], [c.quote for c in cited]):
            attacks.append({"type": "UNSUPPORTED", "claim_ids": [c.claim_id for c in cited], "sentence": idx,
                            "detail": "sentence not supported by its cited quotes"})
    needs = any(a["type"] in ("UNSUPPORTED", "STALE") for a in attacks)
    return {"attacks": attacks, "new_evidence": new_evidence, "queries": queries, "needs_redraft": needs}
