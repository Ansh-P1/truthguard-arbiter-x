"""Eval harness: ARBITER vs baseline RAG on the labelled questions (+ ablations).

    python -m truthguard.eval                 # metrics table (ARBITER vs baseline) + ablation table
    python -m truthguard.eval --no-expansion --no-critic --old-rule-order   # one custom ARBITER config
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime
from typing import Optional

from . import baseline, clock, config, engine, llm
from .agents.graph import Flags, run_question
from .engine import norm
from .ingest import extract, guard, parse
from .scenarios import STATES, build_state, clock_at, load_questions

RESULTS_PATH = config.ROOT / "eval_results.json"


def _scope(q):
    return [q["scope"]] if q.get("scope") else None


def run_arbiter(questions, flags: Flags, stores: dict) -> list[dict]:
    out = []
    for q in questions:
        with clock_at(q.get("now")):
            t0 = time.perf_counter()
            card = run_question(stores[q["state"]], q["question"], _scope(q),
                                date.fromisoformat(q["as_of"]) if q.get("as_of") else None,
                                datetime.fromisoformat(q["known_as_of"]) if q.get("known_as_of") else None, flags)
            dt = time.perf_counter() - t0
        out.append({"id": q["id"], "status": card.status, "value": card.value, "answer": card.answer,
                    "docs": sorted({r.doc_id for r in card.provenance}), "latency": dt,
                    "blob": card.model_dump_json()})
    return out


def run_baseline(questions, stores: dict) -> list[dict]:
    out = []
    for q in questions:
        t0 = time.perf_counter()
        r = baseline.answer(stores[q["state"]], q["question"])
        dt = time.perf_counter() - t0
        out.append({"id": q["id"], "status": None, "value": r["value"], "answer": r["answer"],
                    "docs": [], "latency": dt, "blob": r["answer"]})
    return out


def _same(a: Optional[str], b: Optional[str]) -> bool:
    return a is not None and b is not None and norm(a) == norm(b)


def score(questions, results: list[dict]) -> dict:
    by = {r["id"]: r for r in results}
    val_q = [q for q in questions if q["expected_value"] is not None]
    abs_q = [q for q in questions if q["expected_value"] is None]
    stale_q = [q for q in questions if q["stale_values"]]
    show_q = [q for q in questions if q["must_show_docs"]]
    inj_q = [q for q in questions if "injection" in q["tags"]]
    correct = {q["id"]: _same(by[q["id"]]["value"], q["expected_value"]) for q in val_q}
    abstained = {q["id"]: by[q["id"]]["value"] is None for q in abs_q}
    stale = {q["id"]: (by[q["id"]]["value"] is not None
                       and any(_same(by[q["id"]]["value"], s) for s in q["stale_values"])
                       and not _same(by[q["id"]]["value"], q["expected_value"])) for q in stale_q}
    recall = []
    for q in show_q:
        shown = set(by[q["id"]]["docs"]) | {d for d in q["must_show_docs"]
                                              if d.lower() in by[q["id"]]["answer"].lower()}
        recall.append(len(shown & set(q["must_show_docs"])) / len(q["must_show_docs"]))
    inj = {q["id"]: (_same(by[q["id"]]["value"], q["expected_value"])
                     and not any(b in by[q["id"]]["blob"] for b in q["must_not_contain"])) for q in inj_q}
    pct = lambda d: (sum(d.values()) / len(d)) if d else float("nan")  # noqa: E731
    return {
        "answer_accuracy": pct(correct), "correct_abstention": pct(abstained), "stale_answer_rate": pct(stale),
        "conflict_recall": (sum(recall) / len(recall)) if recall else float("nan"),
        "injection_resistance": pct(inj),
        "status_match": (sum(by[q["id"]]["status"] == q["expected_status"] for q in questions) / len(questions))
        if results and results[0]["status"] is not None else None,
        "mean_latency_s": sum(r["latency"] for r in results) / len(results),
        "wrong": [q["id"] for q in questions
                  if (q["expected_value"] is not None and not correct[q["id"]])
                  or (q["expected_value"] is None and not abstained[q["id"]])],
    }


def extraction_accuracy() -> dict:
    """Precision/recall/F1 of raw extraction (gold paths disabled) vs corpus/gold_claims.json."""
    gold = json.loads(config.GOLD_PATH.read_text(encoding="utf-8"))
    slots_of: dict = {}
    for g in gold:
        slots_of.setdefault(g["doc_id"], [])
        if g["slot"] not in slots_of[g["doc_id"]]:
            slots_of[g["doc_id"]].append(g["slot"])
    key = lambda d, s, k, v, t: (d, s, k, norm(v) if v else "", t or "")  # noqa: E731
    gold_keys = {key(g["doc_id"], g["slot"], g["kind"], g["value"], g.get("target_doc")) for g in gold}
    got = set()
    for p in sorted(list(config.SEED_DIR.glob("*.md")) + list(config.INCOMING_DIR.glob("*.md"))):
        doc = parse.parse_file(p)
        if guard.check(doc.body)[0]:   # quarantined docs never reach extraction
            continue
        claims, _ = extract.extract_claims(doc, slots_of, allow_gold=False)
        got |= {key(c.doc_id, c.slot, c.kind, c.value, c.target_doc) for c in claims}
    tp = len(got & gold_keys)
    prec = tp / len(got) if got else 0.0
    rec = tp / len(gold_keys) if gold_keys else 0.0
    return {"precision": prec, "recall": rec, "f1": (2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
            "extractor": "rule-based (mock)" if llm.is_mock() else os.environ.get("TG_OPENAI_MODEL") or config.CHAT_MODEL}


def _fmt(v, kind="pct"):
    if v is None or v != v:
        return "n/a"
    return f"{v:.1f}s" if kind == "s" else f"{v * 100:.0f}%"


def run_eval(configs: Optional[list[tuple[str, Flags]]] = None, with_baseline: bool = True) -> dict:
    questions = load_questions()
    stores = {name: build_state(name) for name in STATES}
    configs = configs or [("ARBITER (full)", Flags())]
    result: dict = {"mode": llm.mode(), "n_questions": len(questions), "systems": {}}
    if with_baseline:
        res = run_baseline(questions, stores)
        result["systems"]["Baseline RAG"] = score(questions, res)
    for label, flags in configs:
        res = run_arbiter(questions, flags, stores)
        result["systems"][label] = score(questions, res)
    result["extraction"] = extraction_accuracy()
    return result


def table(result: dict) -> str:
    systems = result["systems"]
    cols = list(systems)
    rows = [("Answer accuracy (value questions)", "answer_accuracy", "pct"),
            ("Stale-answer rate (lower is better)", "stale_answer_rate", "pct"),
            ("Correct-abstention rate", "correct_abstention", "pct"),
            ("Conflict recall", "conflict_recall", "pct"),
            ("Injection resistance", "injection_resistance", "pct"),
            ("Verdict-status match", "status_match", "pct"),
            ("Mean latency / question", "mean_latency_s", "s")]
    out = ["| Metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, k, kind in rows:
        out.append(f"| {name} | " + " | ".join(_fmt(systems[c][k], kind) for c in cols) + " |")
    e = result["extraction"]
    out.append("")
    out.append(f"Extraction accuracy ({e['extractor']}, gold paths disabled): precision {_fmt(e['precision'])}, "
               f"recall {_fmt(e['recall'])}, F1 {_fmt(e['f1'])}")
    out.append("")
    out.append("Questions answered wrongly: " + "; ".join(f"{c}: {', '.join(systems[c]['wrong']) or 'none'}" for c in cols))
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="TruthGuard eval: ARBITER vs baseline")
    ap.add_argument("--no-expansion", action="store_true")
    ap.add_argument("--no-critic", action="store_true")
    ap.add_argument("--old-rule-order", action="store_true")
    ap.add_argument("--no-ablations", action="store_true", help="skip the standard ablation sweep")
    a = ap.parse_args(argv)
    custom = a.no_expansion or a.no_critic or a.old_rule_order
    if custom:
        label = "ARBITER (" + ", ".join(n for n, f in [("no-expansion", a.no_expansion), ("no-critic", a.no_critic),
                                                       ("old-rule-order", a.old_rule_order)] if f) + ")"
        cfgs = [("ARBITER (full)", Flags()),
                (label, Flags(expansion=not a.no_expansion, critic=not a.no_critic,
                              rule_order=engine.OLD_ORDER if a.old_rule_order else engine.FIXED_ORDER))]
    elif a.no_ablations:
        cfgs = [("ARBITER (full)", Flags())]
    else:
        cfgs = [("ARBITER (full)", Flags()),
                ("no-expansion", Flags(expansion=False)),
                ("no-critic", Flags(critic=False)),
                ("no-expansion + no-critic", Flags(expansion=False, critic=False)),
                ("old-rule-order", Flags(rule_order=engine.OLD_ORDER))]
    res = run_eval(cfgs)
    print(f"LLM mode: {res['mode']}  |  questions: {res['n_questions']}  |  today={clock.today()}\n")
    print(table(res))
    RESULTS_PATH.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {RESULTS_PATH.name}")


if __name__ == "__main__":
    os.environ.setdefault("TG_TODAY", "2026-10-19")
    main()
