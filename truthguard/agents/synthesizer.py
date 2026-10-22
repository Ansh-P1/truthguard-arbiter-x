"""Synthesizer: drafts an answer from claim fields only (never raw document bodies). Allowed to be naive."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .. import llm
from ..ingest import slots as slotmod
from ..models import Claim
from ..store import Store


class Sentence(BaseModel):
    text: str
    claim_ids: list[str] = Field(default_factory=list)


class Draft(BaseModel):
    sentences: list[Sentence] = Field(default_factory=list)


def fmt_value(value: Optional[str], unit: Optional[str]) -> str:
    if value is None:
        return ""
    return f"{value} {unit}" if unit and unit not in str(value) else str(value)


def _line(c: Claim) -> str:
    return (f"[{c.claim_id}] slot={c.slot} value={fmt_value(c.value, c.unit)} scope={','.join(c.scope)} "
            f"tier={c.tier} effective={c.effective_from} quote=\"{c.quote}\"")


def synthesize(store: Store, question: str, slot: str, evidence: list[dict], stale_ids: set[str],
               redraft: int) -> list[dict]:
    """Return ``[{text, claim_ids}]``. ``stale_ids`` are claims the critic already flagged."""
    claims = {e["claim_id"]: store.get_claim(e["claim_id"]) for e in evidence}
    claims = {k: v for k, v in claims.items() if v is not None and v.kind == "value"}
    prompt = ("Answer the question using ONLY these claims. Cite claim ids for every sentence. If sources "
              "disagree, say so; do not average or blend.\n"
              + ("These claims were flagged as stale or unsupported; avoid relying on them: "
                 + ", ".join(sorted(stale_ids)) + "\n" if stale_ids else "")
              + "\nCLAIMS:\n" + "\n".join(_line(c) for c in claims.values())
              + f"\n\nQUESTION: {question}\nReturn JSON {{\"sentences\": [{{\"text\":..., \"claim_ids\": [...]}}]}}")

    def mock() -> dict:
        # naive: trust the single best-ranked claim that has not been flagged
        for e in evidence:
            c = claims.get(e["claim_id"])
            if c and c.slot == slot and c.claim_id not in stale_ids:
                desc = slotmod.slot_description(slot).lower() if slot in slotmod.SLOTS else "answer"
                return {"sentences": [{"text": f"{desc}: {fmt_value(c.value, c.unit)}.", "claim_ids": [c.claim_id]}]}
        return {"sentences": [{"text": "No reliable source found.", "claim_ids": []}]}

    try:
        d = llm.chat_json(prompt, Draft, mock=mock, fallback=Draft(**mock()))
    except Exception:
        d = Draft(**mock())
    out = []
    for s in d.sentences:
        ids = [i for i in s.claim_ids if i in claims]          # drop hallucinated ids
        out.append({"text": s.text, "claim_ids": ids})
    return out
