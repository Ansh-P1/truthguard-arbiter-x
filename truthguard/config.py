"""Configuration: env vars, authority tiers, paths."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = ROOT / "corpus"
SEED_DIR = CORPUS_DIR / "seed"
INCOMING_DIR = CORPUS_DIR / "incoming"
GOLD_PATH = CORPUS_DIR / "gold_claims.json"
QUESTIONS_PATH = CORPUS_DIR / "questions.json"


def db_path() -> str:
    """DuckDB file path (``TG_DB`` overrides; ``:memory:`` allowed)."""
    return os.environ.get("TG_DB", str(DATA_DIR / "ledger.duckdb"))


CHAT_MODEL = os.environ.get("TG_CHAT_MODEL", "qwen2.5:7b-instruct")
EMBED_MODEL = os.environ.get("TG_EMBED_MODEL", "nomic-embed-text")
CHECK_MODEL = os.environ.get("TG_CHECK_MODEL", "bespoke-minicheck")
OLLAMA_URL = os.environ.get("TG_OLLAMA_URL", "http://localhost:11434")

# Authority tiers (editable). Higher = more authoritative.
TIERS: dict[str, int] = {
    "policy": 3, "amendment": 3, "notice": 3, "terms": 3,
    "memo": 2, "faq": 1, "vendor": 1, "attestation": 0,
}

SCOPE_KEYS = ["general", "enterprise", "prepaid"]
