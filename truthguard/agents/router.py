"""Router: question -> {slot, scope, as_of} (structured output, cosine fallback)."""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

from .. import llm
from ..ingest import slots as slotmod

COSINE_THRESHOLD = 0.45


class RouterOut(BaseModel):
    slot: str = "none"
    scope: list[str] = Field(default_factory=lambda: ["general"])
    as_of: Optional[date] = None


def _scope_from_text(q: str) -> list[str]:
    t = q.lower()
    if re.search(r"\b(enterprise|corporate|business account)\b", t):
        return ["enterprise"]
    if re.search(r"\b(prepaid|pre-paid|pay[- ]as[- ]you[- ]go)\b", t):
        return ["prepaid"]
    return ["general"]


def _date_from_text(q: str) -> Optional[date]:
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", q)
    return date.fromisoformat(m.group(1)) if m else None


def _mock(question: str) -> dict:
    sc = slotmod.keyword_scores(question)
    best = max(sc, key=lambda k: sc[k])
    return {"slot": best if sc[best] > 0 else "none", "scope": _scope_from_text(question),
            "as_of": _date_from_text(question)}


def cosine_slot(question: str) -> str:
    """Fallback: cosine of the question against slot descriptions (threshold 0.45, else ``none``)."""
    ids = slotmod.slot_ids()
    vecs = llm.embed([question] + [slotmod.slot_description(s) for s in ids])
    scores = {s: llm.cosine(vecs[0], v) for s, v in zip(ids, vecs[1:])}
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] >= COSINE_THRESHOLD else "none"


def route(question: str, scope: Optional[list[str]] = None, as_of: Optional[date] = None) -> dict:
    """Route a question. UI-supplied ``scope``/``as_of`` override the model's guess."""
    prompt = ("Route the customer question to ONE slot id from this catalog (or \"none\" if it is not covered), "
              "give the customer scope (['general'], ['enterprise'] or ['prepaid']) and an ISO as_of date only if "
              "the question names a date, else null.\n" + slotmod.catalog_text() +
              f"\n\nQUESTION: {question}\nReturn JSON {{\"slot\":..., \"scope\":[...], \"as_of\":...}}")
    method = "llm"
    try:
        out = llm.chat_json(prompt, RouterOut, mock=lambda: _mock(question))
    except Exception:
        out = RouterOut()
    if llm.is_mock():
        method = "keywords (mock)"
    if out.slot not in slotmod.SLOTS:
        out.slot = cosine_slot(question)
        method += " + cosine fallback"
    return {"slot": out.slot, "scope": scope or out.scope or ["general"], "as_of": as_of or out.as_of,
            "method": method}
