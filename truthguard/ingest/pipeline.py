"""Ingestion pipeline: parse -> guard -> extract -> embed -> edges -> commit -> Truth Diff."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .. import attest, config, ledger, llm, truthdiff
from ..models import Document
from ..store import Store
from . import edges as edgemod
from . import extract, guard, parse
from . import slots as slotmod

log = logging.getLogger("truthguard.ingest")


def claim_embed_text(quote: str, slot: str) -> str:
    return f"{quote} ({slotmod.slot_description(slot)})"


def ingest_document(store: Store, source: str | Path | Document, recorded_at: Optional[datetime] = None,
                    trusted: bool = False) -> dict:
    """Ingest one document (path or parsed ``Document``). Returns a result dict.

    ``trusted=True`` (the shipped seed corpus) uses the regex-only guard and the gold claims, so seeding needs no
    chat-model calls (only embeddings).

    Result keys: ``doc_id, commit, quarantined, reason, claims, edges, truth_diff, attestation_changes,
    extraction, skipped``.
    """
    doc = source if isinstance(source, Document) else parse.parse_file(source, recorded_at)
    if isinstance(source, Document) and recorded_at:
        doc.recorded_at = recorded_at
    existing = store.get_document(doc.doc_id)
    if existing and existing.sha256 == doc.sha256 and existing.recorded_at == doc.recorded_at:
        return {"doc_id": doc.doc_id, "skipped": True, "commit": None, "quarantined": existing.quarantined,
                "reason": existing.quarantine_reason, "claims": [], "edges": [], "truth_diff": [],
                "attestation_changes": [], "extraction": {}}
    if existing:
        store.delete_claims_of(doc.doc_id)

    quarantine, reason = guard.check(doc.body, use_llm=not trusted)
    doc.quarantined, doc.quarantine_reason = quarantine, (reason or None)
    store.upsert_document(doc)

    claims, info, new_edges, diff, att_changes = [], {}, [], [], []
    if quarantine:
        log.warning("QUARANTINED %s: %s", doc.doc_id, reason)
        store.open_ticket("*", f"Document {doc.doc_id} quarantined: {reason}")
    else:
        slots_of = {}
        for c in store.get_claims_raw():
            slots_of.setdefault(c.doc_id, [])
            if c.slot not in slots_of[c.doc_id]:
                slots_of[c.doc_id].append(c.slot)
        claims, info = extract.extract_claims(doc, slots_of, prefer_gold=trusted)
        vecs = llm.embed([claim_embed_text(c.quote, c.slot) for c in claims]) if claims else []
        for c, v in zip(claims, vecs):
            c.embedding = v
            store.insert_claim(c)
        new_edges = edgemod.build_edges(store, doc, claims)

    commit = ledger.commit(store, doc.doc_id, doc.sha256, doc.recorded_at)

    if claims:
        slots = truthdiff.affected_slots(store, claims)
        diff = truthdiff.compute_diff(store, slots, exclude_doc=doc.doc_id)
        att_changes = attest.recheck(store, slots)
    return {"doc_id": doc.doc_id, "skipped": False, "commit": commit.model_dump(mode="json"),
            "quarantined": quarantine, "reason": reason or None, "claims": [c.claim_id for c in claims],
            "edges": [e.model_dump() for e in new_edges], "truth_diff": diff,
            "attestation_changes": att_changes, "extraction": info}


def ingest_seed(store: Store, directory: Path = config.SEED_DIR, trusted: bool = False) -> list[dict]:
    """Ingest a directory of markdown docs in ``effective_from`` order."""
    docs = [parse.parse_file(p) for p in sorted(Path(directory).glob("*.md"))]
    docs.sort(key=lambda d: (d.effective_from, d.doc_id))
    return [ingest_document(store, d, trusted=trusted) for d in docs]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="TruthGuard ingestion")
    ap.add_argument("--seed", action="store_true", help="ingest corpus/seed/")
    ap.add_argument("paths", nargs="*", help="markdown files to ingest")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    store = Store()
    results = ingest_seed(store) if a.seed else [ingest_document(store, p) for p in a.paths]
    for r in results:
        flag = "QUARANTINED" if r["quarantined"] else ("skipped" if r["skipped"] else "ok")
        print(f"{r['doc_id']:10s} {flag:12s} claims={len(r['claims'])}")


if __name__ == "__main__":
    main()
