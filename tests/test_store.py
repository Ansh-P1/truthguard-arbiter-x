from datetime import date, datetime

from truthguard.models import Claim, Document


def _doc(i, quarantined=False):
    return Document(doc_id=f"D{i}", title=f"Doc {i}", doc_type="policy", tier=3, effective_from=date(2025, 1, i),
                    recorded_at=datetime(2025, 1, i, 9), body="b", sha256="s", quarantined=quarantined)


def _claim(i, eff, rec):
    return Claim(claim_id=f"D{i}#1", doc_id=f"D{i}", slot="billing.dispute_window", value=str(i), unit="days",
                 tier=3, effective_from=eff, recorded_at=rec, quote="q")


def test_bitemporal_filter(store):
    for i in (1, 2, 3):
        store.upsert_document(_doc(i, quarantined=(i == 3)))
    store.insert_claim(_claim(1, date(2025, 1, 1), datetime(2025, 1, 1, 9)))
    # valid from Feb but only *recorded* in April (late-arriving knowledge)
    store.insert_claim(_claim(2, date(2025, 2, 1), datetime(2025, 4, 1, 9)))
    store.insert_claim(_claim(3, date(2025, 1, 1), datetime(2025, 1, 1, 9)))  # quarantined doc

    ids = lambda a, k: [c.claim_id for c in store.get_claims(a, k)]  # noqa: E731
    assert ids(date(2025, 3, 1), datetime(2025, 3, 1)) == ["D1#1"]            # D2 not yet recorded
    assert ids(date(2025, 3, 1), datetime(2025, 5, 1)) == ["D1#1", "D2#1"]     # known now
    assert ids(date(2025, 1, 15), datetime(2025, 5, 1)) == ["D1#1"]            # not yet effective
    assert "D3#1" not in ids(date(2025, 12, 1), datetime(2026, 1, 1))          # quarantined never loads


def test_roundtrip_arrays(store):
    store.upsert_document(_doc(1))
    c = _claim(1, date(2025, 1, 1), datetime(2025, 1, 1, 9))
    c.scope = ["enterprise", "prepaid"]
    c.embedding = [0.5, 0.25]
    store.insert_claim(c)
    got = store.get_claim("D1#1")
    assert got.scope == ["enterprise", "prepaid"] and got.embedding == [0.5, 0.25]
