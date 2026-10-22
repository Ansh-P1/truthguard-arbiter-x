"""M5: Truth Diff, attestation lifecycle, audit replay, health."""
from datetime import date, datetime

from truthguard import attest, config, ledger
from truthguard.agents.graph import run_question
from truthguard.health import health_report
from truthguard.ingest import pipeline
from truthguard.scenarios import build_state, clock_at

DQ = "How long do I have to dispute my bill?"


def test_resc_truth_diff_flips_general_but_not_enterprise(seeded):
    with clock_at("2026-03-01T10:00:00"):
        res = pipeline.ingest_document(seeded, config.INCOMING_DIR / "RESC.md")
    rows = {(r["slot"], r["scope"]): r for r in res["truth_diff"]}
    gen = rows[("billing.dispute_window", "general")]
    assert gen["changed"] and gen["before"]["value"] == "45" and gen["after"]["value"] == "30"
    assert gen["deciding_rule"].startswith("R1") and "withdrawal" in gen["deciding_rule"]
    ent = rows[("billing.dispute_window", "enterprise")]
    assert not ent["changed"] and ent["after"]["value"] == "60"
    pre = rows[("billing.dispute_window", "prepaid")]
    assert pre["before"]["value"] == "45" and pre["after"]["value"] == "30"   # prepaid also flips (R2 vs PFAQ)


def test_vendor_upload_returns_quarantine_and_empty_diff(seeded):
    res = pipeline.ingest_document(seeded, config.INCOMING_DIR / "VENDOR.md")
    assert res["quarantined"] and res["truth_diff"] == [] and "injection" in res["reason"]


def test_attestation_flow_q8_q8b_q8c():
    s = build_state("base")
    q = "Can I cancel a port-out request after submitting it?"
    assert run_question(s, q).status == "UNSUPPORTED"
    with clock_at("2026-10-19T12:00:00"):
        a = attest.create_attestation(s, "mnp.portout_cancel_window", "24", "Cancellable within 24 hours.",
                                      "Priya", "MNP Ops Lead", expiry_days=90)
        card = run_question(s, q)
    assert a.status == "active" and (card.status, card.value) == ("ATTESTED", "24")
    assert "MNP Ops Lead" in card.answer and "not yet documented" in card.answer
    with clock_at("2027-01-18T12:00:00"):                      # +91 days
        assert run_question(s, q).status == "UNSUPPORTED"
        assert attest.list_attestations(s)[0].status == "expired"
    with clock_at("2026-12-01T12:00:00"):                      # inside the window again
        assert run_question(s, q).status == "ATTESTED"


def test_quorum_requires_second_approval():
    s = build_state("base")
    a = attest.create_attestation(s, "mnp.portout_cancel_window", "24", "x", "A", "Lead", quorum=2)
    assert a.status == "pending"
    assert run_question(s, "Can I cancel a port-out request?").status == "UNSUPPORTED"
    assert attest.approve(s, a.att_id, "B").status == "active"
    assert run_question(s, "Can I cancel a port-out request?").status == "ATTESTED"


def test_document_arrival_marks_attestation_documented_or_contradicted():
    s = build_state("attested")
    from truthguard.ingest.parse import parse_text
    base = ("---\ndoc_id: {id}\ntitle: MNP Policy {id}\ndoc_type: policy\ntier: 3\nversion: '1'\n"
            "effective_from: 2026-10-01\nscope: [general]\nreplaces: []\nwithdraws: []\n---\n\n"
            "A port-out request may be cancelled within {n} hours of submission.\n")
    # same value -> attestation becomes documented
    res = pipeline.ingest_document(s, parse_text(base.format(id="MNP-A", n=24)))
    assert res["attestation_changes"] == [{"att_id": "ATT-0001", "slot": "mnp.portout_cancel_window",
                                           "status": "documented"}]
    # different value in a second world -> contradicted
    s2 = build_state("attested")
    res2 = pipeline.ingest_document(s2, parse_text(base.format(id="MNP-B", n=48)))
    assert res2["attestation_changes"][0]["status"] == "contradicted"
    card = run_question(s2, "Can I cancel a port-out request?")
    assert card.value == "48" and card.status == "SETTLED"   # contradicted attestation drops out of play


def test_audit_replay_q10():
    s = build_state("after_resc")
    q = "How long did customers have to dispute a bill?"
    then = run_question(s, q, as_of=date(2026, 2, 1), known_as_of=datetime(2026, 2, 1))
    now = run_question(s, q, as_of=date(2026, 2, 1))
    assert (then.value, now.value) == ("45", "30")
    resc = next(r for r in now.provenance if r.doc_id == "RESC")
    assert resc.recorded_at.startswith("2026-03-01") and resc.effective_from == "2026-01-15"


def test_health_report():
    s = build_state("after_vendor")
    h = health_report(s)
    assert [c["slot"] for c in h["contested_slots"]] == ["data.cdr_retention"] * 3
    assert h["quarantined_docs"][0]["doc_id"] == "VENDOR"
    assert any(u["slot"] == "mnp.portout_cancel_window" for u in h["uncovered_slots"])
    assert h["open_doc_debt"]
    s2 = build_state("attested")
    with clock_at("2027-01-10T12:00:00"):                       # 9 days left
        assert [a["att_id"] for a in health_report(s2)["attestations_expiring_soon"]] == ["ATT-0001"]


def test_chain_verifies_after_everything_and_tamper_in_copy_fails(tmp_path):
    from truthguard.store import Store
    s = build_state("after_vendor")
    assert ledger.verify_chain(s) == (True, None)
    # tamper-test on a temp copy: file-based store
    p = str(tmp_path / "copy.duckdb")
    c = Store(p)
    pipeline.ingest_seed(c)
    c.x("UPDATE commits SET doc_sha256='tampered' WHERE seq=3")
    assert ledger.verify_chain(c) == (False, 3)
    c.close()


def test_truth_diff_before_is_rebuilt_when_cache_is_for_another_day(monkeypatch):
    """Seed on one day, upload RESC the next: 'before' must still be 45, not blank."""
    from truthguard.store import Store
    monkeypatch.setenv("TG_TODAY", "2026-10-18")
    s = Store(":memory:")
    pipeline.ingest_seed(s)
    monkeypatch.setenv("TG_TODAY", "2026-10-19")
    res = pipeline.ingest_document(s, config.INCOMING_DIR / "RESC.md")
    gen = next(r for r in res["truth_diff"] if r["scope"] == "general")
    assert gen["before"]["value"] == "45" and gen["after"]["value"] == "30" and gen["changed"]
