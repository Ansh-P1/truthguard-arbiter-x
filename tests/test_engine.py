"""Engine tests: pure claims in, verdict out. No LLM, no DB."""
from datetime import date, datetime

from truthguard import engine
from truthguard.engine import C, resolve

D = date.fromisoformat
T = datetime.fromisoformat
SLOT = "billing.dispute_window"
TODAY = D("2026-10-19")
NOW = T("2026-10-19T12:00:00")


def mk(doc, value, tier, eff, scope=("general",), slot=SLOT, replaces=(), rec=None, kind="value", target=None,
       att=False, cid=None):
    return C(id=cid or f"{doc}#1", doc_id=doc, slot=slot, kind=kind, value=value, scope=frozenset(scope), tier=tier,
             eff=D(eff), rec=T(rec or eff + "T09:00:00"), replaces=frozenset(replaces), target_doc=target,
             is_attestation=att)


BASE = [
    mk("BP-V1", "14", 3, "2023-01-10"),
    mk("FAQ-2023", "14", 1, "2023-03-05"),
    mk("BP-V2", "30", 3, "2024-07-01", replaces=["BP-V1"]),
    mk("AMD-3", "45", 3, "2025-06-01", replaces=["BP-V2"]),
    mk("EST", "60", 3, "2025-09-01", scope=("enterprise",)),
    mk("PFAQ", "7", 1, "2025-10-01", scope=("prepaid",)),
    mk("MEMO", "21", 2, "2025-11-10"),
]
RESC = mk("RESC", None, 3, "2026-01-15", kind="withdrawal", target="AMD-3", rec="2026-03-01T10:00:00")
CDR = [mk("SEC", "90", 3, "2025-09-01", slot="data.cdr_retention"),
       mk("CMP", "180", 3, "2025-09-01", slot="data.cdr_retention"),
       mk("FAQ-2025", "90", 1, "2025-12-01", slot="data.cdr_retention")]


def run(claims, scope="general", as_of=TODAY, known=NOW, slot=SLOT, **kw):
    return resolve(claims, slot, {scope}, as_of, known, **kw)


def defeat(v, doc):
    return next(d for d in v.defeated if d.claim == f"{doc}#1")


def test_q1_superseded_45():
    v = run(BASE)
    assert (v.status, v.answer_value, v.winners) == ("SUPERSEDED", "45", ["AMD-3#1"])
    assert defeat(v, "BP-V2").rule.startswith("R1") and defeat(v, "BP-V2").by == "AMD-3#1"
    assert "BP-V1#1" in {d.claim for d in v.defeated}
    assert defeat(v, "MEMO").rule.startswith("R2")
    assert v.echoes == {"BP-V1#1": ["FAQ-2023#1"]}            # FAQ-2023 is an echo, not a side
    assert v.labels["FAQ-2023#1"] == "OUT"                     # echo inherits representative's label
    assert "EST#1" not in v.labels and "PFAQ#1" not in v.labels  # out of scope


def test_q2_after_rescission_30():
    v = run(BASE + [RESC])
    assert (v.status, v.answer_value) == ("SUPERSEDED", "30")
    d = defeat(v, "AMD-3")
    assert d.by == "RESC#1" and "withdrawal" in d.rule and d.rule.startswith("R1")
    assert v.winners == ["BP-V2#1"]                            # BP-V2 reinstated


def test_q3_rescission_not_yet_effective():
    v = run(BASE + [RESC], as_of=D("2025-12-01"))
    assert (v.status, v.answer_value) == ("SUPERSEDED", "45")


def test_q4_enterprise_lex_specialis():
    v = run(BASE + [RESC], scope="enterprise")
    assert (v.status, v.answer_value) == ("SUPERSEDED", "60")
    assert defeat(v, "BP-V2").rule.startswith("R3")


def test_q5_prepaid_authority_before_specificity():
    v = run(BASE + [RESC], scope="prepaid")
    assert (v.status, v.answer_value) == ("SUPERSEDED", "30")
    assert defeat(v, "PFAQ").rule.startswith("R2")


def test_q5_old_rule_order_gives_7_fixed_gives_30():
    old = run(BASE + [RESC], scope="prepaid", rule_order=engine.OLD_ORDER)
    new = run(BASE + [RESC], scope="prepaid", rule_order=engine.FIXED_ORDER)
    assert old.answer_value == "7"
    assert new.answer_value == "30"


def test_q6_contested_with_echo():
    v = run(CDR, slot="data.cdr_retention")
    assert v.status == "CONTESTED" and v.answer_value is None
    assert sorted(v.contested) == ["CMP#1", "SEC#1"]
    assert v.echoes == {"SEC#1": ["FAQ-2025#1"]}
    assert "FAQ-2025#1" not in v.contested


def test_q7_settled():
    v = run([mk("FAQ-2025", "24x7 on 198", 1, "2025-12-01", slot="support.hours")], slot="support.hours")
    assert (v.status, v.answer_value) == ("SETTLED", "24x7 on 198")


def test_q8_unsupported():
    assert run(BASE, slot="mnp.portout_cancel_window").status == "UNSUPPORTED"


def test_q8b_attested_and_q8c_expired():
    att = mk("ATT-1", "24", 0, "2026-10-19", slot="mnp.portout_cancel_window", rec="2026-10-19T12:00:00", att=True)
    v = run(BASE + [att], slot="mnp.portout_cancel_window")
    assert (v.status, v.answer_value) == ("ATTESTED", "24")
    # expiry is handled by the caller (attestation not passed in once expired)
    assert run(BASE, slot="mnp.portout_cancel_window").status == "UNSUPPORTED"


def test_document_defeats_attestation():
    att = mk("ATT-1", "99", 0, "2026-10-19", slot="data.cdr_retention", rec="2026-10-19T12:00:00", att=True)
    v = run([CDR[0], att], slot="data.cdr_retention")
    assert v.status == "SUPERSEDED" and v.answer_value == "90" and v.labels["ATT-1#1"] == "OUT"


def test_q10_audit_replay():
    feb1 = D("2026-02-01")
    then = run(BASE + [RESC], as_of=feb1, known=T("2026-02-01T00:00:00"))
    now = run(BASE + [RESC], as_of=feb1, known=NOW)
    assert (then.status, then.answer_value) == ("SUPERSEDED", "45")     # RESC not yet recorded
    assert (now.status, now.answer_value) == ("SUPERSEDED", "30")       # same valid time, newer knowledge


def test_norm_units():
    assert engine.norm("45 days") == engine.norm("45") == "45"
    assert engine.norm(" 24 Hours ") == "24"
    assert engine.norm("24x7 on 198") == "24x7 on 198"


def test_unit_variants_are_echoes_not_conflicts():
    cs = [mk("A", "30 days", 3, "2024-01-01"), mk("B", "30", 1, "2024-02-01")]
    v = run(cs)
    assert v.status == "SETTLED" and v.echoes == {"A#1": ["B#1"]}


def test_equal_tier_equal_date_is_tie_and_deterministic():
    cs = [mk("A", "1", 3, "2024-01-01"), mk("B", "2", 3, "2024-01-01")]
    assert run(cs).status == "CONTESTED"
    assert run(list(reversed(cs))).model_dump() == run(cs).model_dump()


def test_withdrawing_only_claim_gives_unsupported():
    cs = [mk("A", "5", 3, "2024-01-01"), mk("W", None, 3, "2025-01-01", kind="withdrawal", target="A")]
    assert run(cs).status == "UNSUPPORTED"
