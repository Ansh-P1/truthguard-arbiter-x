"""Baseline: naive top-k RAG over raw document chunks (no claims, no critic, no rules, no guard)."""
from __future__ import annotations

import re
from typing import Optional

from . import llm
from .ingest import extract
from .ingest import slots as slotmod
from .store import Store

CHUNK_TOKENS = 300
TOP_K = 4
MOCK_SIM_FLOOR = 0.12   # mock "LLM" abstains when nothing in the context resembles the question


def chunk(body: str, size: int = CHUNK_TOKENS) -> list[str]:
    words = body.split()
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words), 1), size)]


def _mock_answer(question: str, top: list[tuple[str, str, float]]) -> str:
    """Stand-in for 'answer from the context': a naive reader trusts the best-matching chunk."""
    if not top or top[0][2] < MOCK_SIM_FLOOR:
        return "The context does not say."
    slot = slotmod.assign_slot(question)
    for doc_id, text, _ in top:
        for s in extract._sentences(text):
            if slotmod.assign_slot(s) == slot or slot is None:
                m = re.search(r"(\d+)\s*(days|hours)", s, re.I)
                if m:
                    return f"{m.group(1)} {m.group(2)}"
                m = re.search(r"available\s+(24x7 on \d+)", s, re.I)
                if m:
                    return m.group(1)
    return "The context does not say."


def answer(store: Store, question: str) -> dict:
    """Answer ``question`` from raw bodies of every document in the store (quarantined ones too)."""
    chunks: list[tuple[str, str]] = []
    for d in store.list_documents():
        chunks += [(d.doc_id, c) for c in chunk(d.body)]
    vecs = llm.embed([question] + [c for _, c in chunks])
    scored = sorted(((did, text, llm.cosine(vecs[0], v)) for (did, text), v in zip(chunks, vecs[1:])),
                    key=lambda t: -t[2])[:TOP_K]
    if llm.is_mock():
        text = _mock_answer(question, scored)
    else:
        ctx = "\n\n".join(f"[{did}] {t}" for did, t, _ in scored)
        try:
            text = llm.chat(f"Answer the question from the context.\n\nCONTEXT:\n{ctx}\n\nQUESTION: {question}")
        except Exception as e:
            text = f"(model error: {e})"
    return {"answer": text, "value": parse_value(text), "context_docs": [d for d, _, _ in scored]}


def parse_value(text: str) -> Optional[str]:
    """Number (or ``24x7 on 198``-style string) from a free-text answer; ``None`` if none found."""
    m = re.search(r"24x7 on \d+", text)
    if m:
        return m.group(0)
    m = re.search(r"\b(\d+)\b", text)
    return m.group(1) if m else None
