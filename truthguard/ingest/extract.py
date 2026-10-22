"""Claim extraction (LLM structured output) with verbatim-quote check and a gold fallback."""
from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field

from .. import config, llm
from ..models import Claim, Document
from . import slots as slotmod

log = logging.getLogger("truthguard.extract")


class ExtractedClaim(BaseModel):
    slot: str
    kind: str = "value"
    value: Optional[str] = None
    unit: Optional[str] = None
    scope: list[str] = Field(default_factory=list)
    quote: str
    target_doc: Optional[str] = None


class ExtractionResult(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list)


# ------------------------------------------------------------------ helpers
def find_span(body: str, quote: str) -> Optional[tuple[int, int]]:
    """Locate ``quote`` in ``body`` ignoring whitespace differences. ``None`` if absent."""
    toks = quote.split()
    if not toks:
        return None
    rx = re.compile(r"\s+".join(re.escape(t) for t in toks))
    m = rx.search(body)
    return (m.start(), m.end()) if m else None


@lru_cache(maxsize=1)
def _gold_all() -> tuple:
    try:
        return tuple(json.loads(config.GOLD_PATH.read_text(encoding="utf-8")))
    except Exception:
        return tuple()


def gold_for(doc_id: str) -> list[dict]:
    return [g for g in _gold_all() if g["doc_id"] == doc_id]


def _build(doc: Document, raw: dict, idx: int) -> Claim:
    span = find_span(doc.body, raw["quote"]) or (-1, -1)
    return Claim(
        claim_id=f"{doc.doc_id}#{idx}", doc_id=doc.doc_id, slot=raw["slot"], kind=raw.get("kind") or "value",
        value=None if raw.get("value") in (None, "") else str(raw["value"]), unit=raw.get("unit"),
        scope=list(raw.get("scope") or doc.scope), tier=doc.tier, effective_from=doc.effective_from,
        recorded_at=doc.recorded_at, target_doc=raw.get("target_doc"), quote=raw["quote"],
        span_start=span[0], span_end=span[1])


def from_gold(doc: Document) -> list[Claim]:
    """Gold claims for a doc (used for tests, the fallback, and ``TG_USE_GOLD=1``)."""
    return [_build(doc, g, i) for i, g in enumerate(gold_for(doc.doc_id), 1)]


# ------------------------------------------------------- mock (rule-based)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _sentences(body: str) -> list[str]:
    out = []
    for line in body.split("\n"):
        line = re.sub(r"^\s*(?:[QA]:|[-*])\s*", "", line.strip())
        if line:
            out += [s.strip() for s in _SENT.split(line) if s.strip()]
    return out


def heuristic_extract(doc: Document) -> ExtractionResult:
    """Rule-based extractor used as the offline stand-in for the LLM (``TG_LLM=mock``)."""
    claims = []
    for s in _sentences(doc.body):
        slot = slotmod.assign_slot(s)
        if not slot:
            continue
        if slot == "support.hours":
            m = re.search(r"\bavailable\s+(.+?)\.?$", s, re.I)
            if m and re.search(r"24\s*x\s*7|\d\s*(?:am|pm)\b|\d:\d\d|round[- ]the[- ]clock|\bmon", m.group(1), re.I) \
                    and re.search(r"support|helpline|care", s, re.I):
                claims.append(ExtractedClaim(slot=slot, value=m.group(1).strip(), scope=doc.scope, quote=s))
            continue
        unit = slotmod.slot_unit(slot) or "days"
        m = re.search(r"(\d+(?:\.\d+)?)\s*" + unit.rstrip("s") + "s?\\b", s, re.I)
        if m:
            claims.append(ExtractedClaim(slot=slot, value=m.group(1), unit=unit, scope=doc.scope, quote=s))
    return ExtractionResult(claims=claims)


# --------------------------------------------------------------------- main
def _prompt(doc: Document) -> str:
    return ("Extract atomic factual CLAIMS from the document, one per statement that sets a value for one of these "
            "slots:\n" + slotmod.catalog_text() + "\n\nFor each claim return slot, kind ('value'), value (number only "
            "when the slot has a unit), unit, scope (list, e.g. ['general'], ['enterprise'], ['prepaid']) and quote: "
            "the EXACT sentence copied verbatim from the document. If the document announces that another document "
            "is withdrawn/rescinded, return kind 'withdrawal' with target_doc set to its doc_id. Only use the slots "
            f"listed. Return JSON {{\"claims\": [...]}}.\n\nDOC_ID: {doc.doc_id}\nTITLE: {doc.title}\n"
            f"DEFAULT SCOPE: {doc.scope}\n\nDOCUMENT:\n{doc.body}")


def _withdrawal_quote(body: str) -> str:
    for s in _sentences(body):
        if re.search(r"withdrawn|rescind|revoked|cancelled|repealed", s, re.I):
            return s
    return _sentences(body)[0] if _sentences(body) else body[:120]


def extract_claims(doc: Document, known_slots_of: Optional[dict[str, list[str]]] = None,
                   allow_gold: bool = True, prefer_gold: bool = False) -> tuple[list[Claim], dict]:
    """Extract claims for ``doc`` -> ``(claims, info)``.

    ``known_slots_of`` maps doc_id -> slots of its stored claims (used for withdrawal claims).
    Order of precedence: ``TG_USE_GOLD=1`` -> LLM/heuristic (2 tries) -> gold fallback
    (``allow_gold=False`` disables both gold paths; used to measure raw extraction accuracy).
    """
    info: dict = {"source": None, "dropped": [], "warnings": []}
    raws: Optional[list[dict]] = None
    if allow_gold and (prefer_gold or os.environ.get("TG_USE_GOLD") == "1") and gold_for(doc.doc_id):
        info["source"] = "gold"
        raws = [g for g in gold_for(doc.doc_id)]
    else:
        err = None
        for attempt in range(2):
            try:
                res = llm.chat_json(_prompt(doc), ExtractionResult, mock=lambda: heuristic_extract(doc), retries=0)
                kept: list[dict] = []
                info["dropped"] = []
                for c in res.claims:
                    if c.slot not in slotmod.SLOTS:
                        info["dropped"].append({"reason": "slot not in catalog", "quote": c.quote})
                        log.info("dropped claim: slot %s not in catalog", c.slot)
                    elif find_span(doc.body, c.quote) is None:
                        info["dropped"].append({"reason": "quote not verbatim", "quote": c.quote})
                        log.warning("dropped claim: quote not verbatim in %s: %r", doc.doc_id, c.quote)
                    else:
                        kept.append(c.model_dump())
                raws = kept
                info["source"] = "heuristic" if llm.is_mock() else "llm"
                break
            except Exception as e:
                err = e
                log.warning("extraction attempt %d failed for %s: %s", attempt + 1, doc.doc_id, e)
        if raws is None or (not raws and gold_for(doc.doc_id) and not doc.withdraws):
            if allow_gold and gold_for(doc.doc_id):
                msg = f"extraction failed/empty for {doc.doc_id} ({err}); using gold claims"
                log.warning(msg)
                info["warnings"].append(msg)
                info["source"] = "gold"
                raws = list(gold_for(doc.doc_id))
            elif raws is None:
                raws = []
    claims: list[Claim] = []
    for raw in raws:
        claims.append(_build(doc, raw, len(claims) + 1))
    _ensure_withdrawals(doc, claims, known_slots_of or {})
    return claims, info


def _ensure_withdrawals(doc: Document, claims: list[Claim], slots_of: dict[str, list[str]]) -> None:
    """Make sure every doc_id in ``doc.withdraws`` has a withdrawal claim (create from metadata)."""
    for c in claims:
        if c.kind == "withdrawal" and not c.target_doc and len(doc.withdraws) == 1:
            c.target_doc = doc.withdraws[0]
    for target in doc.withdraws:
        if any(c.kind == "withdrawal" and c.target_doc == target for c in claims):
            continue
        quote = _withdrawal_quote(doc.body)
        span = find_span(doc.body, quote) or (-1, -1)
        for slot in (slots_of.get(target) or [next(iter(slotmod.SLOTS))]):
            claims.append(Claim(
                claim_id=f"{doc.doc_id}#{len(claims) + 1}", doc_id=doc.doc_id, slot=slot, kind="withdrawal",
                value=None, unit=None, scope=list(doc.scope), tier=doc.tier, effective_from=doc.effective_from,
                recorded_at=doc.recorded_at, target_doc=target, quote=quote, span_start=span[0], span_end=span[1]))
