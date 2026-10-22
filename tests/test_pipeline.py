"""End-to-end agent graph tests (TG_LLM=mock): every labelled question in corpus/questions.json."""
from datetime import date, datetime

import pytest

from truthguard.agents.graph import Flags, run_question
from truthguard.scenarios import build_state, clock_at, load_questions

QUESTIONS = load_questions()
_STATES: dict = {}


def state(name):
    if name not in _STATES:
        _STATES[name] = build_state(name)
    return _STATES[name]


def ask(q, flags=None):
    with clock_at(q.get("now")):
        return run_question(
            state(q["state"]), q["question"], scope=[q["scope"]] if q.get("scope") else None,
            as_of=date.fromisoformat(q["as_of"]) if q.get("as_of") else None,
            known_as_of=datetime.fromisoformat(q["known_as_of"]) if q.get("known_as_of") else None, flags=flags)


@pytest.mark.parametrize("q", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_labelled_question(q):
    card = ask(q)
    assert card.status == q["expected_status"], card.answer
    if q["expected_value"] is not None:
        assert card.value == q["expected_value"]
    docs = {r.doc_id for r in card.provenance}
    for d in q["must_show_docs"]:
        assert d in docs, f"{d} missing from provenance {docs}"
    for bad in q["must_not_contain"]:
        blob = card.model_dump_json()
        assert bad not in blob, f"{bad} leaked into the card"


def test_at_least_25_questions():
    assert len(QUESTIONS) >= 25


def test_q1_trace_shows_amd3_only_via_expansion():
    card = ask(next(q for q in QUESTIONS if q["id"] == "Q1"))
    amd = next(r for r in card.provenance if r.doc_id == "AMD-3")
    assert amd.origin == ["expansion"]


def test_q2_resc_only_via_expansion_or_falsification():
    card = ask(next(q for q in QUESTIONS if q["id"] == "Q2"))
    resc = next(r for r in card.provenance if r.doc_id == "RESC")
    assert "similarity" not in resc.origin
    amd = next(r for r in card.provenance if r.doc_id == "AMD-3")
    assert amd.rule.startswith("R1") and "withdrawal" in amd.rule and amd.defeated_by == "RESC"


def test_q5_rule_trace_names_r2():
    card = ask(next(q for q in QUESTIONS if q["id"] == "Q5"))
    pf = next(r for r in card.provenance if r.doc_id == "PFAQ")
    assert pf.rule.startswith("R2") and pf.label == "OUT"


def test_q6_card_lists_two_sides_and_echo_and_ticket():
    card = ask(next(q for q in QUESTIONS if q["id"] == "Q6"))
    contested = [r for r in card.provenance if r.role == "contested"]
    assert {r.doc_id for r in contested} == {"SEC", "CMP"}
    faq = next(r for r in card.provenance if r.doc_id == "FAQ-2025")
    assert faq.role == "echo" and faq.echo_of == "SEC#1"
    assert card.ticket_id and "Cannot resolve" in card.answer


def test_q8_creates_ticket_and_offers_attestation():
    card = ask(next(q for q in QUESTIONS if q["id"] == "Q8"))
    assert card.ticket_id and card.needs_attestation
    assert any(t["slot"] == "mnp.portout_cancel_window" for t in state("base").tickets("open"))


def test_q8c_attestation_status_expired():
    from truthguard import attest
    q = next(q for q in QUESTIONS if q["id"] == "Q8c")
    ask(q)
    with clock_at(q["now"]):
        assert [a.status for a in attest.list_attestations(state("attested"))] == ["expired"]
    assert [a.status for a in attest.list_attestations(state("attested"))] == ["active"]  # reversible clock


def test_q9_vendor_quarantined_and_365_absent():
    s = state("after_vendor")
    assert s.get_document("VENDOR").quarantined
    card = ask(next(q for q in QUESTIONS if q["id"] == "Q9"))
    assert "365" not in card.model_dump_json()


def test_q10_audit_replay_changes_answer():
    then = ask(next(q for q in QUESTIONS if q["id"] == "Q10"))
    now = ask(next(q for q in QUESTIONS if q["id"] == "Q10b"))
    assert (then.value, now.value) == ("45", "30")
    resc = next(r for r in now.provenance if r.doc_id == "RESC")
    assert resc.recorded_at.startswith("2026-03-01")           # the cause shown to the user


def test_critic_flags_stale_draft_and_redraft_is_bounded():
    card = ask(next(q for q in QUESTIONS if q["id"] == "Q2"))
    assert card.redrafts <= 2
    assert any(a["type"] == "STALE" for a in card.critic_attacks) or card.redrafts == 0
    agents = [t["agent"] for t in card.trace]
    assert agents[0] == "router" and agents[-1] == "resolution" and "critic" in agents


def test_synthesizer_never_sees_raw_bodies(monkeypatch):
    """Raw document text must never reach downstream prompts: only claim fields."""
    from truthguard import llm
    seen = []
    orig = llm.chat_json
    monkeypatch.setattr(llm, "chat_json", lambda prompt, *a, **k: (seen.append(prompt), orig(prompt, *a, **k))[1])
    ask(next(q for q in QUESTIONS if q["id"] == "Q1"))
    bodies = [d.body for d in state("base").list_documents()]
    sentences = [s for b in bodies for s in b.split("\n") if len(s) > 60 and "days" not in s]
    assert seen and not any(s in p for p in seen for s in sentences)


def test_ablation_no_expansion_hurts_q1():
    """AMD-3 ranks 8th by similarity, so without slot expansion (and without falsification) Q1 goes wrong."""
    q1 = next(q for q in QUESTIONS if q["id"] == "Q1")
    assert ask(q1).value == "45"
    assert ask(q1, Flags(expansion=False, critic=False)).value != "45"


def test_old_rule_order_flag_breaks_q5():
    from truthguard import engine
    q5 = next(q for q in QUESTIONS if q["id"] == "Q5")
    assert ask(q5, Flags(rule_order=engine.OLD_ORDER)).value == "7"
