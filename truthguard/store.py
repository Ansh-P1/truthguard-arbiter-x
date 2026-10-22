"""DuckDB store: schema, CRUD, bitemporal loading."""
from __future__ import annotations

import json
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import duckdb

from . import clock, config
from .engine import C
from .models import Attestation, Claim, Document, Edge

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY, title TEXT, doc_type TEXT, tier INT, version TEXT,
  effective_from DATE, recorded_at TIMESTAMP, scope TEXT[], replaces TEXT[], withdraws TEXT[],
  body TEXT, sha256 TEXT, quarantined BOOLEAN DEFAULT FALSE, quarantine_reason TEXT
);
CREATE TABLE IF NOT EXISTS claims (
  claim_id TEXT PRIMARY KEY, doc_id TEXT, slot TEXT, kind TEXT,
  value TEXT, unit TEXT, scope TEXT[], tier INT,
  effective_from DATE, recorded_at TIMESTAMP,
  target_doc TEXT,
  quote TEXT, span_start INT, span_end INT, embedding FLOAT[]
);
CREATE TABLE IF NOT EXISTS edges (src TEXT, dst TEXT, type TEXT, source TEXT, rule TEXT);
CREATE TABLE IF NOT EXISTS commits (
  seq INT PRIMARY KEY, doc_id TEXT, doc_sha256 TEXT, recorded_at TIMESTAMP,
  parent_hash TEXT, hash TEXT
);
CREATE TABLE IF NOT EXISTS attestations (
  att_id TEXT PRIMARY KEY, slot TEXT, scope TEXT[], value TEXT, unit TEXT, statement TEXT,
  attester TEXT, role TEXT, created_at TIMESTAMP, expires_at TIMESTAMP,
  status TEXT, quorum_required INT DEFAULT 1, approvals TEXT[]
);
CREATE TABLE IF NOT EXISTS verdict_cache (slot TEXT, scope_key TEXT, as_of DATE, status TEXT, payload JSON,
  PRIMARY KEY (slot, scope_key, as_of));
CREATE TABLE IF NOT EXISTS doc_debt (ticket_id TEXT PRIMARY KEY, slot TEXT, reason TEXT, created_at TIMESTAMP, status TEXT);
"""

DOC_COLS = ["doc_id", "title", "doc_type", "tier", "version", "effective_from", "recorded_at", "scope",
            "replaces", "withdraws", "body", "sha256", "quarantined", "quarantine_reason"]
CLAIM_COLS = ["claim_id", "doc_id", "slot", "kind", "value", "unit", "scope", "tier", "effective_from",
              "recorded_at", "target_doc", "quote", "span_start", "span_end", "embedding"]
ATT_COLS = ["att_id", "slot", "scope", "value", "unit", "statement", "attester", "role", "created_at",
            "expires_at", "status", "quorum_required", "approvals"]


class Store:
    """Thin thread-safe wrapper over one DuckDB connection."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or config.db_path()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(self.path)
        self._lock = threading.RLock()
        with self._lock:
            self.con.execute(SCHEMA)

    # -- low level --------------------------------------------------------
    def q(self, sql: str, params=()):
        with self._lock:
            return self.con.execute(sql, list(params)).fetchall()

    def x(self, sql: str, params=()):
        with self._lock:
            self.con.execute(sql, list(params))

    def rows(self, sql: str, cols, params=()):
        return [dict(zip(cols, r)) for r in self.q(sql, params)]

    def close(self):
        with self._lock:
            self.con.close()

    # -- documents --------------------------------------------------------
    def has_doc(self, doc_id: str) -> bool:
        return bool(self.q("SELECT 1 FROM documents WHERE doc_id=?", [doc_id]))

    def upsert_document(self, d: Document):
        self.x("DELETE FROM documents WHERE doc_id=?", [d.doc_id])
        self.x(f"INSERT INTO documents VALUES ({','.join('?' * len(DOC_COLS))})",
               [getattr(d, c) for c in DOC_COLS])

    def get_document(self, doc_id: str) -> Optional[Document]:
        r = self.rows(f"SELECT {','.join(DOC_COLS)} FROM documents WHERE doc_id=?", DOC_COLS, [doc_id])
        return Document(**r[0]) if r else None

    def list_documents(self) -> list[Document]:
        return [Document(**r) for r in
                self.rows(f"SELECT {','.join(DOC_COLS)} FROM documents ORDER BY effective_from, doc_id", DOC_COLS)]

    # -- claims -----------------------------------------------------------
    def insert_claim(self, c: Claim):
        self.x("DELETE FROM claims WHERE claim_id=?", [c.claim_id])
        self.x(f"INSERT INTO claims VALUES ({','.join('?' * len(CLAIM_COLS))})",
               [getattr(c, k) for k in CLAIM_COLS])

    def delete_claims_of(self, doc_id: str):
        ids = [r[0] for r in self.q("SELECT claim_id FROM claims WHERE doc_id=?", [doc_id])]
        for i in ids:
            self.x("DELETE FROM edges WHERE src=? OR dst=?", [i, i])
        self.x("DELETE FROM claims WHERE doc_id=?", [doc_id])

    def get_claims(self, as_of: Optional[date] = None, known_as_of: Optional[datetime] = None,
                   slot: Optional[str] = None, doc_id: Optional[str] = None) -> list[Claim]:
        """Bitemporal load: effective_from <= as_of AND recorded_at <= known_as_of AND doc not quarantined."""
        as_of = as_of or clock.today()
        known_as_of = known_as_of or clock.now()
        sql = (f"SELECT {','.join('c.' + k for k in CLAIM_COLS)} FROM claims c JOIN documents d USING (doc_id) "
               "WHERE c.effective_from <= ? AND c.recorded_at <= ? AND NOT d.quarantined")
        params: list = [as_of, known_as_of]
        if slot:
            sql += " AND c.slot = ?"
            params.append(slot)
        if doc_id:
            sql += " AND c.doc_id = ?"
            params.append(doc_id)
        sql += " ORDER BY c.claim_id"
        return [Claim(**r) for r in self.rows(sql, CLAIM_COLS, params)]

    def get_claims_raw(self, doc_id: Optional[str] = None) -> list[Claim]:
        """All stored claims of non-quarantined docs, no temporal filter (graph building)."""
        sql = (f"SELECT {','.join('c.' + k for k in CLAIM_COLS)} FROM claims c JOIN documents d USING (doc_id) "
               "WHERE NOT d.quarantined")
        params: list = []
        if doc_id:
            sql += " AND c.doc_id = ?"
            params.append(doc_id)
        return [Claim(**r) for r in self.rows(sql + " ORDER BY c.claim_id", CLAIM_COLS, params)]

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        r = self.rows(f"SELECT {','.join(CLAIM_COLS)} FROM claims WHERE claim_id=?", CLAIM_COLS, [claim_id])
        return Claim(**r[0]) if r else None

    def all_claims(self) -> list[Claim]:
        return [Claim(**r) for r in self.rows(f"SELECT {','.join(CLAIM_COLS)} FROM claims ORDER BY claim_id", CLAIM_COLS)]

    # -- edges ------------------------------------------------------------
    def add_edge(self, e: Edge):
        if not self.q("SELECT 1 FROM edges WHERE src=? AND dst=? AND type=?", [e.src, e.dst, e.type]):
            self.x("INSERT INTO edges VALUES (?,?,?,?,?)", [e.src, e.dst, e.type, e.source, e.rule])

    def get_edges(self, type_: Optional[str] = None) -> list[Edge]:
        sql = "SELECT src,dst,type,source,rule FROM edges"
        params: list = []
        if type_:
            sql += " WHERE type=?"
            params.append(type_)
        return [Edge(**r) for r in self.rows(sql, ["src", "dst", "type", "source", "rule"], params)]

    # -- engine view ------------------------------------------------------
    def engine_claims(self, as_of: Optional[date] = None, known_as_of: Optional[datetime] = None,
                      slot: Optional[str] = None, include_attestations: bool = True) -> list[C]:
        """Claims (+ active attestations) as engine views, bitemporally filtered."""
        from . import attest  # local import: avoid cycle

        as_of = as_of or clock.today()
        known_as_of = known_as_of or clock.now()
        claims = self.get_claims(as_of, known_as_of)
        claim_doc = {c.claim_id: c.doc_id for c in claims}
        replaces: dict[str, set] = {}
        for e in self.get_edges("SUPERSEDES"):
            if e.dst in claim_doc:
                replaces.setdefault(e.src, set()).add(claim_doc[e.dst])
            else:  # dst claim may be outside the window; resolve its doc directly
                r = self.q("SELECT doc_id FROM claims WHERE claim_id=?", [e.dst])
                if r:
                    replaces.setdefault(e.src, set()).add(r[0][0])
        out = [C(id=c.claim_id, doc_id=c.doc_id, slot=c.slot, kind=c.kind, value=c.value,
                 scope=frozenset(c.scope), tier=c.tier, eff=c.effective_from, rec=c.recorded_at,
                 replaces=frozenset(replaces.get(c.claim_id, ())), target_doc=c.target_doc)
               for c in claims]
        if include_attestations:
            out += attest.engine_views(self)
        if slot:
            out = [c for c in out if c.slot == slot or c.kind == "withdrawal"]
        return out

    # -- verdict cache ----------------------------------------------------
    def cache_get(self, slot: str, scope_key: str, as_of: date) -> Optional[dict]:
        r = self.q("SELECT payload FROM verdict_cache WHERE slot=? AND scope_key=? AND as_of=?",
                   [slot, scope_key, as_of])
        return json.loads(r[0][0]) if r else None

    def cache_put(self, slot: str, scope_key: str, as_of: date, status: str, payload: dict):
        self.x("DELETE FROM verdict_cache WHERE slot=? AND scope_key=? AND as_of=?", [slot, scope_key, as_of])
        self.x("INSERT INTO verdict_cache VALUES (?,?,?,?,?)",
               [slot, scope_key, as_of, status, json.dumps(payload)])

    # -- doc debt ---------------------------------------------------------
    def open_ticket(self, slot: str, reason: str) -> str:
        """Create (or reuse) an open doc-debt ticket for (slot, reason)."""
        r = self.q("SELECT ticket_id FROM doc_debt WHERE slot=? AND reason=? AND status='open'", [slot, reason])
        if r:
            return r[0][0]
        n = self.q("SELECT count(*) FROM doc_debt")[0][0] + 1
        tid = f"DEBT-{n:04d}"
        self.x("INSERT INTO doc_debt VALUES (?,?,?,?,?)", [tid, slot, reason, clock.now(), "open"])
        return tid

    def tickets(self, status: Optional[str] = None) -> list[dict]:
        sql = "SELECT ticket_id, slot, reason, created_at, status FROM doc_debt"
        params: list = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        return self.rows(sql + " ORDER BY ticket_id", ["ticket_id", "slot", "reason", "created_at", "status"], params)

    # -- attestations -----------------------------------------------------
    def save_attestation(self, a: Attestation):
        self.x("DELETE FROM attestations WHERE att_id=?", [a.att_id])
        self.x(f"INSERT INTO attestations VALUES ({','.join('?' * len(ATT_COLS))})", [getattr(a, k) for k in ATT_COLS])

    def list_attestations(self) -> list[Attestation]:
        return [Attestation(**r) for r in
                self.rows(f"SELECT {','.join(ATT_COLS)} FROM attestations ORDER BY created_at, att_id", ATT_COLS)]
