"""Pydantic models shared across the system."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    title: str
    doc_type: str
    tier: int
    version: str = "1.0"
    effective_from: date
    recorded_at: datetime
    scope: list[str] = Field(default_factory=lambda: ["general"])
    replaces: list[str] = Field(default_factory=list)
    withdraws: list[str] = Field(default_factory=list)
    body: str = ""
    sha256: str = ""
    quarantined: bool = False
    quarantine_reason: Optional[str] = None


class Claim(BaseModel):
    claim_id: str
    doc_id: str
    slot: str
    kind: str = "value"  # value | withdrawal
    value: Optional[str] = None
    unit: Optional[str] = None
    scope: list[str] = Field(default_factory=lambda: ["general"])
    tier: int = 0
    effective_from: date
    recorded_at: datetime
    target_doc: Optional[str] = None
    quote: str = ""
    span_start: int = -1
    span_end: int = -1
    embedding: Optional[list[float]] = None


class Edge(BaseModel):
    src: str
    dst: str
    type: str  # SUPERSEDES | WITHDRAWS | CONTRADICTS | REAFFIRMS | EXCEPTION_OF
    source: str = "declared"  # declared | derived
    rule: Optional[str] = None


class Commit(BaseModel):
    seq: int
    doc_id: str
    doc_sha256: str
    recorded_at: datetime
    parent_hash: str
    hash: str


class Attestation(BaseModel):
    att_id: str
    slot: str
    scope: list[str] = Field(default_factory=lambda: ["general"])
    value: str
    unit: Optional[str] = None
    statement: str = ""
    attester: str
    role: str
    created_at: datetime
    expires_at: datetime
    status: str = "active"  # active | pending | expired | disputed | contradicted | documented
    quorum_required: int = 1
    approvals: list[str] = Field(default_factory=list)


class Defeat(BaseModel):
    claim: str
    by: str
    rule: str


class Verdict(BaseModel):
    """Output of the deterministic engine."""
    slot: str
    status: str  # SETTLED | SUPERSEDED | CONTESTED | ATTESTED | UNSUPPORTED
    answer_value: Optional[str] = None
    winners: list[str] = Field(default_factory=list)
    defeated: list[Defeat] = Field(default_factory=list)
    contested: list[str] = Field(default_factory=list)
    echoes: dict[str, list[str]] = Field(default_factory=dict)
    attacks: list[dict[str, str]] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)


class ProvenanceRow(BaseModel):
    claim_id: str
    doc_id: str
    title: str
    version: str
    tier: int
    kind: str
    value: Optional[str] = None
    unit: Optional[str] = None
    scope: list[str] = Field(default_factory=list)
    effective_from: str
    recorded_at: str
    quote: str
    label: str  # IN | OUT | UNDEC | ECHO
    role: str  # winner | defeated | contested | echo | withdrawal
    defeated_by: Optional[str] = None
    rule: Optional[str] = None
    echo_of: Optional[str] = None
    origin: list[str] = Field(default_factory=list)  # similarity | expansion | falsification


class ConflictCard(BaseModel):
    question: str
    slot: str
    scope: list[str]
    as_of: str
    known_as_of: str
    status: str
    answer: str
    value: Optional[str] = None
    unit: Optional[str] = None
    provenance: list[ProvenanceRow] = Field(default_factory=list)
    rule_trace: list[str] = Field(default_factory=list)
    echoes: dict[str, list[str]] = Field(default_factory=dict)
    attacks: list[dict[str, str]] = Field(default_factory=list)
    certificates: list[dict[str, Any]] = Field(default_factory=list)
    ticket_id: Optional[str] = None
    needs_attestation: bool = False
    draft: list[dict[str, Any]] = Field(default_factory=list)
    critic_attacks: list[dict[str, Any]] = Field(default_factory=list)
    redrafts: int = 0
    trace: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
