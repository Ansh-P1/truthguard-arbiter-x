"""Hash-chained commit ledger. Every ingest is a commit; the chain is verifiable."""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Optional

from .models import Commit
from .store import Store

GENESIS = "0" * 64
_COLS = ["seq", "doc_id", "doc_sha256", "recorded_at", "parent_hash", "hash"]


def compute_hash(parent_hash: str, doc_id: str, doc_sha256: str, recorded_at: datetime) -> str:
    """``sha256(parent_hash + doc_id + doc_sha256 + recorded_at.isoformat())``."""
    return hashlib.sha256((parent_hash + doc_id + doc_sha256 + recorded_at.isoformat()).encode()).hexdigest()


def head(store: Store) -> Optional[Commit]:
    r = store.rows(f"SELECT {','.join(_COLS)} FROM commits ORDER BY seq DESC LIMIT 1", _COLS)
    return Commit(**r[0]) if r else None


def commit(store: Store, doc_id: str, doc_sha256: str, recorded_at: datetime) -> Commit:
    """Append a commit to the chain."""
    h = head(store)
    seq = (h.seq + 1) if h else 1
    parent = h.hash if h else GENESIS
    c = Commit(seq=seq, doc_id=doc_id, doc_sha256=doc_sha256, recorded_at=recorded_at,
               parent_hash=parent, hash=compute_hash(parent, doc_id, doc_sha256, recorded_at))
    store.x("INSERT INTO commits VALUES (?,?,?,?,?,?)", [getattr(c, k) for k in _COLS])
    return c


def list_commits(store: Store) -> list[Commit]:
    return [Commit(**r) for r in store.rows(f"SELECT {','.join(_COLS)} FROM commits ORDER BY seq", _COLS)]


def verify_chain(store: Store) -> tuple[bool, Optional[int]]:
    """Recompute every hash. Returns ``(ok, first_bad_seq)``."""
    parent = GENESIS
    for c in list_commits(store):
        if c.parent_hash != parent or c.hash != compute_hash(c.parent_hash, c.doc_id, c.doc_sha256, c.recorded_at):
            return False, c.seq
        parent = c.hash
    return True, None
