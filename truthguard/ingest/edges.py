"""Edges: declared SUPERSEDES / WITHDRAWS from metadata, regex backup, derived relation edges."""
from __future__ import annotations

import re

from ..engine import norm
from ..models import Claim, Document, Edge
from ..store import Store

_VERB = re.compile(r"\b(replaced|replaces|supersedes|superseded|withdrawn|rescinded|rescinds|withdraws)\b", re.I)
_WITHDRAW_VERB = re.compile(r"withdrawn|rescind|withdraws", re.I)


def _sentences(body: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", body) if s.strip()]


def build_edges(store: Store, doc: Document, claims: list[Claim]) -> list[Edge]:
    """Create and store all edges for the claims of ``doc``. Returns the edges added."""
    added: list[Edge] = []

    def put(e: Edge):
        store.add_edge(e)
        added.append(e)

    def claims_of(doc_id: str, slot: str):
        return [c for c in store.get_claims_raw(doc_id) if c.slot == slot and c.kind == "value"]

    # declared: this doc replaces / withdraws others
    for c in claims:
        if c.kind == "value":
            for x in doc.replaces:
                for t in claims_of(x, c.slot):
                    put(Edge(src=c.claim_id, dst=t.claim_id, type="SUPERSEDES", source="declared",
                             rule="R1 explicit supersession (declared)"))
        elif c.kind == "withdrawal" and c.target_doc:
            put(Edge(src=c.claim_id, dst=c.target_doc, type="WITHDRAWS", source="declared",
                     rule="R1 withdrawal (declared)"))
    # declared, other direction: already-stored docs that declared they replace this one
    for other in store.list_documents():
        if other.doc_id != doc.doc_id and doc.doc_id in other.replaces and not other.quarantined:
            for oc in store.get_claims_raw(other.doc_id):
                if oc.kind == "value":
                    for t in claims_of(doc.doc_id, oc.slot):
                        put(Edge(src=oc.claim_id, dst=t.claim_id, type="SUPERSEDES", source="declared",
                                 rule="R1 explicit supersession (declared)"))
    # regex backup: a sentence with a replace/withdraw verb that names another known doc
    others = [d for d in store.list_documents() if d.doc_id != doc.doc_id and not d.quarantined]
    for s in _sentences(doc.body):
        if not _VERB.search(s):
            continue
        low = s.lower()
        for x in others:
            if x.doc_id.lower() in low or x.title.lower() in low:
                if _WITHDRAW_VERB.search(s):
                    for c in claims:
                        if c.kind == "withdrawal" and (c.target_doc in (None, x.doc_id)):
                            put(Edge(src=c.claim_id, dst=x.doc_id, type="WITHDRAWS", source="derived",
                                     rule="R1 withdrawal (regex)"))
                else:
                    for c in claims:
                        if c.kind == "value":
                            for t in claims_of(x.doc_id, c.slot):
                                put(Edge(src=c.claim_id, dst=t.claim_id, type="SUPERSEDES", source="derived",
                                         rule="R1 explicit supersession (regex)"))
    # derived relation edges against already stored claims of the same slot
    for c in claims:
        if c.kind != "value":
            continue
        for o in store.get_claims_raw():
            if o.doc_id == doc.doc_id or o.kind != "value" or o.slot != c.slot:
                continue
            if norm(o.value) == norm(c.value) and set(o.scope) == set(c.scope):
                put(Edge(src=c.claim_id, dst=o.claim_id, type="REAFFIRMS", source="derived"))
            elif ("general" in c.scope) != ("general" in o.scope):
                scoped, general = (c, o) if "general" not in c.scope else (o, c)
                put(Edge(src=scoped.claim_id, dst=general.claim_id, type="EXCEPTION_OF", source="derived"))
            else:
                put(Edge(src=c.claim_id, dst=o.claim_id, type="CONTRADICTS", source="derived"))
    return added
