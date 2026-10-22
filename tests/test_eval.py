"""M7: ARBITER beats the baseline; ablations show measurable drops."""
import pytest

from truthguard import engine
from truthguard import eval as ev
from truthguard.agents.graph import Flags


@pytest.fixture(scope="module")
def result():
    cfgs = [("full", Flags()), ("no-expansion + no-critic", Flags(expansion=False, critic=False)),
            ("old-rule-order", Flags(rule_order=engine.OLD_ORDER))]
    return ev.run_eval(cfgs)


def test_arbiter_beats_baseline(result):
    a, b = result["systems"]["full"], result["systems"]["Baseline RAG"]
    assert a["answer_accuracy"] > b["answer_accuracy"]
    assert a["stale_answer_rate"] < b["stale_answer_rate"]
    assert a["correct_abstention"] > b["correct_abstention"]
    assert a["conflict_recall"] > b["conflict_recall"]
    assert a["injection_resistance"] >= b["injection_resistance"]


def test_arbiter_is_perfect_on_labelled_set_in_mock_mode(result):
    a = result["systems"]["full"]
    assert a["answer_accuracy"] == 1.0 and a["status_match"] == 1.0 and a["stale_answer_rate"] == 0.0


def test_ablations_drop(result):
    full = result["systems"]["full"]
    assert result["systems"]["no-expansion + no-critic"]["answer_accuracy"] < full["answer_accuracy"]
    assert "Q1" in result["systems"]["no-expansion + no-critic"]["wrong"]
    assert "Q5" in result["systems"]["old-rule-order"]["wrong"]


def test_extraction_f1(result):
    assert result["extraction"]["f1"] == 1.0


def test_table_renders(result):
    t = ev.table(result)
    assert "| Answer accuracy" in t and "Baseline RAG" in t
