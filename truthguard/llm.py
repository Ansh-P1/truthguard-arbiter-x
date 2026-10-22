"""All model calls go through here: chat(), chat_json(schema), embed().

Backends: Ollama (default if reachable), any OpenAI-compatible endpoint
(``TG_OPENAI_BASE_URL`` + ``TG_OPENAI_API_KEY``) or a deterministic offline ``mock``
(``TG_LLM=mock``; also the automatic fallback when nothing is reachable).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.request
from typing import Callable, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from . import config

log = logging.getLogger("truthguard.llm")
T = TypeVar("T", bound=BaseModel)
MOCK_DIM = 256
_ollama_ok: Optional[bool] = None


class LLMError(RuntimeError):
    pass


def _http(url: str, payload: Optional[dict] = None, headers: Optional[dict] = None, timeout: float = 120.0,
          retries: int = 4) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (local/user-configured endpoint)
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries:       # free tiers rate-limit: back off and retry
                wait = min(60.0, 5.0 * 2 ** attempt)
                log.warning("HTTP %s from %s; retrying in %.0fs", e.code, url.split("/")[2], wait)
                time.sleep(wait)
                continue
            raise
    raise LLMError("unreachable")


def ollama_available() -> bool:
    """True if a local Ollama answers (cached for the process)."""
    global _ollama_ok
    if _ollama_ok is None:
        try:
            _http(config.OLLAMA_URL + "/api/tags", timeout=0.7)
            _ollama_ok = True
        except Exception:
            _ollama_ok = False
    return _ollama_ok


def mode() -> str:
    """``mock`` | ``ollama`` | ``openai`` (``TG_LLM`` forces; otherwise auto-detect)."""
    forced = os.environ.get("TG_LLM", "").lower()
    if forced in ("mock", "ollama", "openai"):
        return forced
    if os.environ.get("TG_OPENAI_BASE_URL"):
        return "openai"
    return "ollama" if ollama_available() else "mock"


def is_mock() -> bool:
    return mode() == "mock"


# --------------------------------------------------------------------- chat
def chat(prompt: str, system: str = "", schema: Optional[dict] = None, model: Optional[str] = None) -> str:
    """One chat completion; ``schema`` requests JSON-schema structured output."""
    m = mode()
    if m == "mock":
        raise LLMError("chat() is unavailable in mock mode; supply a mock= fallback to chat_json")
    model = model or config.CHAT_MODEL
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    try:
        if m == "ollama":
            body: dict = {"model": model, "messages": msgs, "stream": False, "options": {"temperature": 0}}
            if schema:
                body["format"] = schema
            return _http(config.OLLAMA_URL + "/api/chat", body)["message"]["content"]
        base = os.environ["TG_OPENAI_BASE_URL"].rstrip("/")
        hdr = {"Authorization": "Bearer " + os.environ.get("TG_OPENAI_API_KEY", "")}
        body = {"model": os.environ.get("TG_OPENAI_MODEL", model), "messages": msgs, "temperature": 0}
        if schema:
            body["response_format"] = {"type": "json_object"}
        return _http(base + "/chat/completions", body, hdr, timeout=60, retries=2)["choices"][0]["message"]["content"]
    except Exception as e:  # network / server errors
        raise LLMError(f"chat failed: {e}") from e


def _extract_json(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", s, re.S)
        if m:
            return json.loads(m.group(1))
        raise


def chat_json(prompt: str, model: Type[T], system: str = "", mock: Optional[Callable[[], object]] = None,
              fallback: Optional[T] = None, retries: int = 1) -> T:
    """Structured call validated with pydantic: one retry, then ``fallback`` (or raise LLMError).

    In mock mode ``mock()`` supplies the answer (a model instance or a dict).
    """
    if is_mock():
        if mock is None:
            if fallback is not None:
                return fallback
            raise LLMError("no mock supplied")
        r = mock()
        return r if isinstance(r, model) else model.model_validate(r)
    schema = model.model_json_schema()
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            raw = chat(prompt + ("" if attempt == 0 else "\n\nReturn ONLY valid JSON matching the schema."),
                       system=system, schema=schema)
            return model.model_validate(_extract_json(raw))
        except (LLMError, ValidationError, ValueError, KeyError) as e:
            last = e
            log.warning("chat_json attempt %d failed: %s", attempt + 1, e)
    if fallback is not None:
        log.warning("chat_json falling back after failures: %s", last)
        return fallback
    raise LLMError(f"structured call failed: {last}")


# -------------------------------------------------------------------- embed
_TOK = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "to", "is", "are", "in", "on", "and", "or", "for", "i", "you", "my", "do", "how",
         "can", "be", "it", "that", "this", "we", "have", "has", "with", "by", "at", "as", "your"}


def mock_embed(text: str) -> list[float]:
    """Deterministic hashing bag-of-words (+bigrams) embedding. Offline stand-in for nomic-embed-text."""
    v = [0.0] * MOCK_DIM
    toks = [t for t in _TOK.findall(text.lower()) if t not in _STOP]
    grams = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
    for g in grams:
        h = int(hashlib.md5(g.encode()).hexdigest(), 16)
        v[h % MOCK_DIM] += 1.0 if "_" not in g else 0.5
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


_embed_down_until = 0.0


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts with the configured backend. On failure fall back to mock embeddings, and stop calling the
    remote endpoint for 5 minutes (circuit breaker) so a throttled free tier cannot stall the app."""
    global _embed_down_until
    m = mode()
    if m == "mock" or time.time() < _embed_down_until:
        return [mock_embed(t) for t in texts]
    try:
        if m == "ollama":
            r = _http(config.OLLAMA_URL + "/api/embed", {"model": config.EMBED_MODEL, "input": texts}, timeout=30,
                      retries=1)
            return r["embeddings"]
        base = os.environ["TG_OPENAI_BASE_URL"].rstrip("/")
        hdr = {"Authorization": "Bearer " + os.environ.get("TG_OPENAI_API_KEY", "")}
        r = _http(base + "/embeddings", {"model": os.environ.get("TG_OPENAI_EMBED_MODEL", config.EMBED_MODEL),
                                          "input": texts}, hdr, timeout=20, retries=1)
        return [d["embedding"] for d in r["data"]]
    except Exception as e:
        log.warning("embed failed (%s); using mock embeddings for 5 minutes", e)
        _embed_down_until = time.time() + 300
        return [mock_embed(t) for t in texts]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)
