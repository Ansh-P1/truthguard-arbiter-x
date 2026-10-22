"""Slot catalog and slot assignment."""
from __future__ import annotations

import re
from typing import Optional

SLOTS: dict[str, dict] = {
    "billing.dispute_window": {
        "description": "Days a customer has to raise a billing dispute",
        "unit": "days",
        "keywords": ["dispute", "disputes", "disputed", "disputing", "billing dispute", "contest a bill", "challenge a bill",
                     "wrong bill", "incorrect charge", "bill is wrong", "overcharged", "raise a complaint about a bill"],
    },
    "data.cdr_retention": {
        "description": "How long call detail records are retained",
        "unit": "days",
        "keywords": ["call detail", "call record", "call records", "cdr", "call history", "records retained",
                     "retention", "retained", "keep my call", "store my call", "kept"],
    },
    "support.hours": {
        "description": "Customer support availability",
        "unit": None,
        "keywords": ["support", "helpline", "customer care", "available", "availability", "reach", "contact",
                     "open hours", "24x7", "call centre", "call center"],
    },
    "mnp.portout_cancel_window": {
        "description": "Time allowed to cancel a port-out (MNP) request",
        "unit": "hours",
        "keywords": ["port-out", "port out", "portout", "mnp", "porting", "port request", "number portability",
                     "cancel a port"],
    },
}


def slot_ids() -> list[str]:
    return list(SLOTS)


def slot_description(slot: str) -> str:
    return SLOTS[slot]["description"]


def slot_unit(slot: str) -> Optional[str]:
    return SLOTS[slot]["unit"]


def catalog_text() -> str:
    return "\n".join(f"- {k}: {v['description']} (unit: {v['unit'] or 'free text'})" for k, v in SLOTS.items())


def keyword_scores(text: str) -> dict[str, float]:
    """Cheap keyword scoring used by the mock router and as a router fallback."""
    t = text.lower()
    out = {}
    for sid, info in SLOTS.items():
        hits = 0.0
        for kw in info["keywords"]:
            if re.search(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])", t):
                hits += 1.0 + 0.5 * kw.count(" ")
        out[sid] = hits
    return out


def assign_slot(text: str) -> Optional[str]:
    """Best slot for ``text`` by keyword, or ``None``."""
    sc = keyword_scores(text)
    best = max(sc, key=lambda k: sc[k])
    return best if sc[best] > 0 else None
