from datetime import datetime

from truthguard import ledger


def _chain(store, n=4):
    for i in range(n):
        ledger.commit(store, f"D{i}", f"sha{i}", datetime(2026, 1, 1 + i, 9, 0, 0))


def test_chain_verifies(store):
    assert ledger.verify_chain(store) == (True, None)
    _chain(store)
    assert ledger.verify_chain(store) == (True, None)
    cs = ledger.list_commits(store)
    assert cs[0].parent_hash == "0" * 64
    assert cs[1].parent_hash == cs[0].hash


def test_hash_formula(store):
    c = ledger.commit(store, "X", "abc", datetime(2026, 1, 1, 9, 0, 0))
    assert c.hash == ledger.compute_hash("0" * 64, "X", "abc", datetime(2026, 1, 1, 9, 0, 0))


def test_tamper_detected(store):
    _chain(store)
    store.x("UPDATE commits SET doc_sha256='evil' WHERE seq=2")
    assert ledger.verify_chain(store) == (False, 2)


def test_deleted_link_detected(store):
    _chain(store)
    store.x("DELETE FROM commits WHERE seq=2")
    ok, bad = ledger.verify_chain(store)
    assert not ok and bad == 3
