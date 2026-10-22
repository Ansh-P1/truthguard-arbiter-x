"""LangGraph wiring: router -> retriever -> synthesizer -> critic -> (redraft?) -> resolution."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .. import clock, engine
from ..models import ConflictCard
from ..store import Store
from . import critic as critic_mod
from . import resolution, retriever, router, synthesizer

MAX_REDRAFTS = 2


@dataclass
class Flags:
    """Ablation switches."""
    expansion: bool = True
    critic: bool = True
    rule_order: tuple = engine.FIXED_ORDER


class State(TypedDict, total=False):
    question: str
    ui_scope: Optional[list]
    ui_as_of: Optional[date]
    q_scope: list
    as_of: date
    known_as_of: datetime
    slot: str
    evidence: list
    draft: list
    attacks: list
    stale_ids: list
    redrafts: int
    needs_redraft: bool
    card: Any
    trace: list


def _t(state: State, agent: str, t0: float, **summary) -> list:
    return list(state.get("trace", [])) + [{"agent": agent, "ms": round((time.perf_counter() - t0) * 1000, 1),
                                            **summary}]


def build_graph(store: Store, flags: Optional[Flags] = None):
    """Compile the agent graph for ``store``."""
    flags = flags or Flags()

    def n_router(s: State):
        t0 = time.perf_counter()
        r = router.route(s["question"], s.get("ui_scope"), s.get("ui_as_of"))
        return {"slot": r["slot"], "q_scope": r["scope"], "as_of": r["as_of"] or s["as_of"],
                "trace": _t(s, "router", t0, slot=r["slot"], scope=r["scope"], method=r["method"])}

    def n_retriever(s: State):
        t0 = time.perf_counter()
        ev = retriever.retrieve(store, s["question"], s["slot"], s["q_scope"], s["as_of"], s["known_as_of"],
                                expansion=flags.expansion)
        return {"evidence": ev, "trace": _t(
            s, "retriever", t0, evidence=[{"claim_id": e["claim_id"], "origin": e["origin"]} for e in ev])}

    def n_synth(s: State):
        t0 = time.perf_counter()
        draft = synthesizer.synthesize(store, s["question"], s["slot"], s["evidence"], set(s.get("stale_ids", [])),
                                       s.get("redrafts", 0))
        return {"draft": draft, "trace": _t(s, "synthesizer", t0, redraft=s.get("redrafts", 0), draft=draft)}

    def n_critic(s: State):
        t0 = time.perf_counter()
        r = critic_mod.critique(store, s["question"], s["slot"], s["q_scope"], s["as_of"], s["known_as_of"],
                                s["draft"], s["evidence"])
        known = {e["claim_id"] for e in s["evidence"]}
        evidence = list(s["evidence"]) + [e for e in r["new_evidence"] if e["claim_id"] not in known]
        stale = set(s.get("stale_ids", [])) | {i for a in r["attacks"] if a["type"] == "STALE" for i in a["claim_ids"]}
        redrafts = s.get("redrafts", 0)
        again = r["needs_redraft"] and redrafts < MAX_REDRAFTS
        return {"attacks": r["attacks"], "evidence": evidence, "stale_ids": sorted(stale),
                "needs_redraft": again, "redrafts": redrafts + (1 if again else 0),
                "trace": _t(s, "critic", t0, attacks=[{k: a[k] for k in ("type", "claim_ids", "detail")}
                                                      for a in r["attacks"]],
                            falsification_queries=r["queries"],
                            found_by_falsification=[e["claim_id"] for e in r["new_evidence"]],
                            needs_redraft=again)}

    def n_resolution(s: State):
        t0 = time.perf_counter()
        trace = _t(s, "resolution", t0)
        card = resolution.build_card(store, s["question"], s["slot"], s["q_scope"], s["as_of"], s["known_as_of"],
                                     s["evidence"], s.get("draft", []), s.get("attacks", []), s.get("redrafts", 0),
                                     trace, flags.rule_order)
        card.trace[-1]["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        card.trace[-1].update(status=card.status, value=card.value)
        return {"card": card, "trace": card.trace}

    g = StateGraph(State)
    g.add_node("router", n_router)
    g.add_node("retriever", n_retriever)
    g.add_node("resolution", n_resolution)
    g.set_entry_point("router")
    g.add_edge("router", "retriever")
    if flags.critic:
        g.add_node("synthesizer", n_synth)
        g.add_node("critic", n_critic)
        g.add_edge("retriever", "synthesizer")
        g.add_edge("synthesizer", "critic")
        g.add_conditional_edges("critic", lambda s: "synthesizer" if s.get("needs_redraft") else "resolution",
                                {"synthesizer": "synthesizer", "resolution": "resolution"})
    else:
        g.add_edge("retriever", "resolution")
    g.add_edge("resolution", END)
    return g.compile()


def run_question(store: Store, question: str, scope: Optional[list[str]] = None, as_of: Optional[date] = None,
                 known_as_of: Optional[datetime] = None, flags: Optional[Flags] = None) -> ConflictCard:
    """Answer ``question`` and return its ConflictCard. UI-supplied scope/as_of override the router."""
    graph = build_graph(store, flags)
    out = graph.invoke({"question": question, "ui_scope": scope, "ui_as_of": as_of,
                        "as_of": as_of or clock.today(), "known_as_of": known_as_of or clock.now(),
                        "redrafts": 0, "trace": [], "stale_ids": []},
                       {"recursion_limit": 25})
    return out["card"]


def main(argv=None) -> None:
    from ..ingest import pipeline

    q = " ".join(argv or sys.argv[1:]) or "How long do I have to dispute my bill?"
    store = Store()
    if not store.list_documents():
        pipeline.ingest_seed(store)
    print(run_question(store, q).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
