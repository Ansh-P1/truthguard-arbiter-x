"""Support checker: is a sentence entailed by its cited quotes?

Order: MiniCheck via Ollama -> HF NLI cross-encoder -> offline lexical check (always available).
"""
from __future__ import annotations

import logging
import re

from . import config, llm

log = logging.getLogger("truthguard.checker")
_NUM = re.compile(r"\d+(?:\.\d+)?(?:x\d+)?", re.I)
_W = re.compile(r"[a-z]{4,}")
_STOP = {"that", "this", "with", "from", "have", "will", "your", "they", "their", "which", "within", "what",
         "when", "does", "must", "should", "were", "been", "than", "into", "such", "also", "each"}
_nli = None


def lexical_supported(sentence: str, quotes: list[str]) -> tuple[bool, float]:
    """All numbers in the sentence occur in a quote, and enough content words overlap."""
    if not quotes:
        return False, 0.0
    qtext = " ".join(quotes).lower()
    nums = _NUM.findall(sentence.lower())
    if any(n not in qtext for n in nums):
        return False, 0.0
    words = {w for w in _W.findall(sentence.lower()) if w not in _STOP}
    if not words:
        return bool(nums), 1.0 if nums else 0.0
    qwords = set(_W.findall(qtext))
    overlap = len(words & qwords) / len(words)
    return (overlap >= 0.2 or (bool(nums) and overlap >= 0.1)), overlap


def _minicheck(sentence: str, quotes: list[str]) -> bool:
    out = llm.chat(f"Document: {' '.join(quotes)}\nClaim: {sentence}", model=config.CHECK_MODEL)
    return out.strip().lower().startswith("yes")


def _hf_nli(sentence: str, quotes: list[str]) -> bool:
    global _nli
    from transformers import pipeline as hf  # optional dependency

    if _nli is None:
        _nli = hf("text-classification", model="cross-encoder/nli-deberta-v3-base", top_k=None)
    scores = _nli({"text": " ".join(quotes), "text_pair": sentence})
    scores = scores[0] if isinstance(scores[0], list) else scores
    ent = next((s["score"] for s in scores if s["label"].lower().startswith("entail")), 0.0)
    return ent > 0.5


def supported(sentence: str, quotes: list[str]) -> bool:
    """True if ``sentence`` is supported by ``quotes``. Never raises."""
    if not quotes or not sentence.strip():
        return False
    ok_lex, _ = lexical_supported(sentence, quotes)
    if llm.mode() == "ollama":
        try:
            return _minicheck(sentence, quotes)
        except Exception as e:
            log.info("minicheck unavailable (%s)", e)
        try:
            return _hf_nli(sentence, quotes)
        except Exception as e:
            log.info("HF NLI unavailable (%s)", e)
    return ok_lex
