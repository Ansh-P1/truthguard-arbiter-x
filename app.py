"""TruthGuard ARBITER-X Lite: Streamlit UI.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from truthguard import api, attest, clock, config, ledger, llm  # noqa: E402
from truthguard import eval as tg_eval  # noqa: E402
from truthguard.agents.graph import run_question  # noqa: E402
from truthguard.health import health_report  # noqa: E402
from truthguard.ingest import parse, pipeline  # noqa: E402
from truthguard.ingest import slots as slotmod  # noqa: E402
from truthguard.store import Store  # noqa: E402

# Streamlit Cloud: bridge secrets -> env vars (e.g. TG_OPENAI_BASE_URL / TG_OPENAI_API_KEY / TG_LLM)
try:
    for _k in ("TG_LLM", "TG_OPENAI_BASE_URL", "TG_OPENAI_API_KEY", "TG_OPENAI_MODEL", "TG_OPENAI_EMBED_MODEL"):
        if _k in st.secrets and _k not in os.environ:
            os.environ[_k] = str(st.secrets[_k])
except Exception:  # no secrets file: run in auto/mock mode
    pass

st.set_page_config(page_title="TruthGuard ARBITER-X", page_icon="🛡️", layout="wide")

BADGE = {"SETTLED": ("#1b5e20", "#e8f5e9"), "SUPERSEDED": ("#0d47a1", "#e3f2fd"), "CONTESTED": ("#e65100", "#fff3e0"),
         "ATTESTED": ("#4a148c", "#f3e5f5"), "UNSUPPORTED": ("#b71c1c", "#ffebee")}
LABEL_COLOR = {"IN": "#c8e6c9", "OUT": "#ffcdd2", "UNDEC": "#ffe0b2"}


@st.cache_resource
def store() -> Store:
    return api.get_store()


def badge(status: str) -> str:
    fg, bg = BADGE.get(status, ("#333", "#eee"))
    return (f"<span style='background:{bg};color:{fg};border:1px solid {fg};padding:4px 12px;border-radius:14px;"
            f"font-weight:700;letter-spacing:.5px'>{status}</span>")


def fmt_val(v, u):
    return "" if v is None else (f"{v} {u}" if u and u not in str(v) else str(v))


# ------------------------------------------------------------------ sidebar
def sidebar() -> str:
    st.sidebar.title("🛡️ TruthGuard")
    st.sidebar.caption("ARBITER-X Lite · Truth-as-Code")
    if "goto" in st.session_state:                      # set before the radio is instantiated
        st.session_state["page"] = st.session_state.pop("goto")
    page = st.sidebar.radio("Page", ["Ask", "Upload / Truth Diff", "Attestations", "Corpus Health", "Audit", "Eval"],
                            key="page")
    st.sidebar.divider()
    st.sidebar.markdown(f"**LLM mode:** `{llm.mode()}`")
    if llm.is_mock():
        st.sidebar.caption("Offline mock: rule-based extraction/router + hashing embeddings. Start Ollama for real models.")
    with st.sidebar.expander("🕒 Clock (demo)", expanded=False):
        st.caption(f"now = `{clock.now().isoformat()}`  ·  today = `{clock.today()}`")
        d = st.date_input("Set 'now' date", value=clock.now().date(), key="clk_d")
        t = st.time_input("time", value=clock.now().time().replace(microsecond=0), key="clk_t")
        pin = st.checkbox("Also pin 'today' (valid time) to this date", value=False,
                          help="Off: today follows 'now'. On: sets TG_TODAY too.")
        c1, c2 = st.columns(2)
        if c1.button("Apply"):
            os.environ["TG_NOW"] = datetime.combine(d, t).isoformat()
            if pin:
                os.environ["TG_TODAY"] = d.isoformat()
            else:
                os.environ.pop("TG_TODAY", None)
            st.rerun()
        if c2.button("Reset"):
            os.environ.pop("TG_NOW", None)
            os.environ.pop("TG_TODAY", None)
            st.rerun()
        st.caption("Process-wide (single-user demo). Use it to record RESC 'in March' or expire an attestation.")
    return page


# ---------------------------------------------------------------- graph view
def graph_dot(card) -> str:
    lines = ["digraph G { rankdir=LR; node [shape=box, style=filled, fontname=Helvetica, fontsize=10];"]
    for r in card.provenance:
        if r.role == "echo":
            fill, style = "#eeeeee", "dashed,filled"
        else:
            fill, style = LABEL_COLOR.get(r.label, "#ffffff"), "filled"
        txt = f"{r.doc_id}\\n{fmt_val(r.value, r.unit) or r.kind}\\n[{r.label}]"
        lines.append(f'"{r.claim_id}" [label="{txt}", fillcolor="{fill}", style="{style}"];')
        if r.echo_of:
            lines.append(f'"{r.claim_id}" -> "{r.echo_of}" [style=dotted, arrowhead=none, label="echo"];')
    for a in card.attacks:
        rule = a["rule"].split(" ")[0]
        lines.append(f'"{a["src"]}" -> "{a["dst"]}" [label="{rule}", color="#b71c1c", fontsize=9];')
    lines.append("}")
    return "\n".join(lines)


def render_card(card, replay_note: str | None = None):
    st.markdown(badge(card.status), unsafe_allow_html=True)
    body = card.answer.replace("\n", "  \n")   # keep one line per superseded source
    if card.status == "UNSUPPORTED":
        st.error(body)
    elif card.status == "CONTESTED":
        st.warning(body)
    else:
        st.success(body)
    for n in card.notes:
        st.warning(n)
    if replay_note:
        st.info(replay_note)
    cap = f"slot `{card.slot}` · scope `{', '.join(card.scope)}` · as_of `{card.as_of}` · knowledge as of `{card.known_as_of}`"
    if card.ticket_id:
        cap += f" · 🎫 doc-debt ticket `{card.ticket_id}`"
    st.caption(cap)

    if card.provenance:
        st.subheader("Provenance")
        df = pd.DataFrame([{
            "label": r.label, "role": r.role, "doc": f"{r.doc_id} v{r.version}", "title": r.title, "tier": r.tier,
            "value": fmt_val(r.value, r.unit) or r.kind, "effective": r.effective_from, "recorded": r.recorded_at[:16],
            "defeated by": r.defeated_by or "", "rule": r.rule or "", "also stated by / echo of": r.echo_of or "",
            "found via": ", ".join(r.origin), "quote": r.quote} for r in card.provenance])
        colors = {"IN": "background-color:#c8e6c9", "OUT": "background-color:#ffcdd2",
                  "UNDEC": "background-color:#ffe0b2"}
        st.dataframe(df.style.map(lambda v: colors.get(v, ""), subset=["label"]), use_container_width=True,
                     hide_index=True)
    if card.rule_trace:
        st.subheader("Rule trace")
        for line in card.rule_trace:
            st.markdown(f"- {line}")
    if card.echoes:
        st.caption("Echoes (same value, no extra weight): " + "; ".join(
            f"{k} ← {', '.join(v)}" for k, v in card.echoes.items()))
    if card.provenance:
        st.subheader("Argument graph")
        st.caption("green = IN · red = OUT · amber = UNDEC · grey dashed = echo · red arrows = attacks (rule)")
        st.graphviz_chart(graph_dot(card), use_container_width=True)
    if card.certificates:
        with st.expander("Proof certificate"):
            st.json(card.certificates)

    with st.expander("Agent trace"):
        for t in card.trace:
            st.markdown(f"**{t['agent']}** · {t['ms']} ms")
            if t["agent"] == "router":
                st.json({k: t[k] for k in ("slot", "scope", "method")})
            elif t["agent"] == "retriever":
                st.dataframe(pd.DataFrame([{"claim": e["claim_id"], "found via": ", ".join(e["origin"])}
                                           for e in t["evidence"]]), hide_index=True)
                only_exp = [e["claim_id"] for e in t["evidence"] if e["origin"] == ["expansion"]]
                if only_exp:
                    st.caption("Found ONLY via slot expansion (similarity search missed them): " + ", ".join(only_exp))
            elif t["agent"] == "synthesizer":
                st.caption(f"Naive draft #{t['redraft'] + 1}")
                for s in t["draft"]:
                    st.markdown(f"> {s['text']}  _{', '.join(s['claim_ids']) or 'no citation'}_")
            elif t["agent"] == "critic":
                st.dataframe(pd.DataFrame(t["attacks"] or [{"type": "-", "claim_ids": [], "detail": "no attacks"}]
                                          ).astype(str), hide_index=True)
                st.caption("Falsification queries: " + " | ".join(t["falsification_queries"]))
                if t["found_by_falsification"]:
                    st.caption("Found by falsification: " + ", ".join(t["found_by_falsification"]))
            elif t["agent"] == "resolution":
                st.json({"status": t.get("status"), "value": t.get("value")})
        st.caption(f"Redrafts: {card.redrafts}")


# --------------------------------------------------------------------- pages
def page_ask():
    st.title("Ask")
    st.caption("Conflicting policies in, one auditable answer out.")
    SAMPLES = ["How long do I have to dispute my bill?", "How long are call records kept?",
               "When is customer support available?", "Can I cancel a port-out request after submitting it?"]
    pick = st.selectbox("Sample questions", [""] + SAMPLES)
    q = st.text_input("Your question", value=pick or st.session_state.get("last_q", ""), key="q_input")
    c1, c2, c3 = st.columns(3)
    scope = c1.selectbox("Customer scope", config.SCOPE_KEYS)
    as_of = c2.date_input("Valid as of", value=clock.today())
    replay = c3.checkbox("Audit replay: use knowledge as of an earlier date")
    known = None
    if replay:
        kd = c3.date_input("Replay with knowledge as of", value=date(2026, 2, 1))
        known = datetime.combine(kd, time(23, 59, 59))
    if st.button("Ask", type="primary") and q.strip():
        st.session_state["last_q"] = q
        with st.spinner("Retrieving, drafting, attacking, resolving…"):
            card = run_question(store(), q, [scope], as_of, known)
            note = None
            if known is not None:
                now_card = run_question(store(), q, [scope], as_of, None)
                if now_card.value != card.value or now_card.status != card.status:
                    late = [r for r in now_card.provenance
                            if r.recorded_at > known.isoformat(sep=" ") and r.role in ("winner", "withdrawal")]
                    cause = f" because {late[0].doc_id} was recorded on {late[0].recorded_at[:10]}" if late else ""
                    note = (f"On {known.date()} we said **{fmt_val(card.value, card.unit) or card.status}**. With what "
                            f"we know now, the right answer for {as_of} is **{fmt_val(now_card.value, now_card.unit) or now_card.status}** — "
                            f"answer changed{cause}.")
        st.session_state["card"], st.session_state["note"] = card, note
    card = st.session_state.get("card")
    if card:
        render_card(card, st.session_state.get("note"))
        if card.needs_attestation and st.button("✋ Attest this (go to Attestations)"):
            st.session_state["att_slot"] = card.slot
            st.session_state["goto"] = "Attestations"
            st.rerun()


def show_diff(rows):
    if not rows:
        return
    df = pd.DataFrame([{
        "slot": r["slot"], "scope": r["scope"], "changed": "🔁 yes" if r["changed"] else "no",
        "before": "-" if not r["before"] else f"{r['before']['status']} {r['before']['value'] or ''}",
        "after": f"{r['after']['status']} {r['after']['value'] or ''}", "deciding rule": r["deciding_rule"] or ""}
        for r in rows])
    st.dataframe(df, hide_index=True, use_container_width=True)


def page_upload():
    st.title("Upload / Truth Diff")
    st.caption("Every upload is a hash-chained commit; affected slots are recomputed and flipped answers shown.")
    incoming = sorted(p.name for p in config.INCOMING_DIR.glob("*.md"))
    choice = st.selectbox("From corpus/incoming", [""] + incoming)
    up = st.file_uploader("…or upload a .md file", type=["md"])
    if st.button("Ingest", type="primary") and (choice or up):
        try:
            if up is not None:
                doc = parse.parse_text(up.getvalue().decode("utf-8"))
            else:
                doc = parse.parse_file(config.INCOMING_DIR / choice)
            res = pipeline.ingest_document(store(), doc)
        except Exception as e:
            st.error(f"Could not ingest: {e}")
            return
        if res["skipped"]:
            st.info(f"{res['doc_id']} was already ingested (identical content); nothing changed.")
        elif res["quarantined"]:
            st.error(f"🚫 {res['doc_id']} QUARANTINED — {res['reason']}\n\nIt was committed for the audit trail but "
                     f"contributes no claims. A doc-debt ticket was opened.")
        else:
            st.success(f"{res['doc_id']} ingested · commit #{res['commit']['seq']} · hash {res['commit']['hash'][:12]}… "
                       f"· claims: {', '.join(res['claims']) or 'none'}")
            if res["extraction"].get("warnings"):
                st.warning("; ".join(res["extraction"]["warnings"]))
            st.subheader("Truth Diff")
            changed = [r for r in res["truth_diff"] if r["changed"]]
            if changed:
                st.markdown("**Answers that flipped:**")
            show_diff(changed or res["truth_diff"])
            if res["attestation_changes"]:
                st.info(f"Attestation status changes: {res['attestation_changes']}")


def page_attest():
    st.title("Attestations")
    st.caption("An expert vouches for an undocumented fact. Tier 0, expires, and any document claim defeats it.")
    undocumented = list(slotmod.SLOTS)
    default = st.session_state.get("att_slot", "mnp.portout_cancel_window")
    with st.form("att"):
        slot = st.selectbox("Slot", undocumented, index=undocumented.index(default) if default in undocumented else 0)
        c1, c2 = st.columns(2)
        scope = c1.selectbox("Scope", config.SCOPE_KEYS)
        value = c2.text_input("Value (e.g. 24)", "24")
        statement = st.text_area("Statement", "A port-out request can be cancelled within 24 hours of submission.")
        c3, c4 = st.columns(2)
        attester = c3.text_input("Attester", "Priya N.")
        role = c4.text_input("Role", "MNP Ops Lead")
        c5, c6 = st.columns(2)
        days = c5.number_input("Expiry (days)", 1, 3650, 90)
        quorum = c6.number_input("Quorum required", 1, 5, 1)
        if st.form_submit_button("Create attestation"):
            try:
                a = attest.create_attestation(store(), slot, value, statement, attester, role, int(days), [scope],
                                              int(quorum))
                st.success(f"{a.att_id} created ({a.status}); expires {a.expires_at:%Y-%m-%d}")
            except Exception as e:
                st.error(str(e))
    atts = attest.list_attestations(store())
    st.subheader("All attestations")
    if atts:
        st.dataframe(pd.DataFrame([{
            "id": a.att_id, "slot": a.slot, "scope": ", ".join(a.scope), "value": a.value, "status": a.status,
            "attester": f"{a.attester} ({a.role})", "created": f"{a.created_at:%Y-%m-%d %H:%M}",
            "expires": f"{a.expires_at:%Y-%m-%d}", "approvals": f"{len(a.approvals)}/{a.quorum_required}"}
            for a in atts]), hide_index=True, use_container_width=True)
        pend = [a for a in atts if a.status == "pending"]
        if pend:
            c1, c2, c3 = st.columns([2, 2, 1])
            sel = c1.selectbox("Pending attestation", [a.att_id for a in pend])
            who = c2.text_input("Approver")
            if c3.button("Approve") and who:
                attest.approve(store(), sel, who)
                st.rerun()
    else:
        st.info("None yet.")
    st.caption("Tip: use the sidebar clock to jump 91 days ahead and watch the attestation expire.")


def page_health():
    st.title("Corpus Health")
    h = health_report(store())
    c = st.columns(4)
    c[0].metric("Documents", h["documents"])
    c[1].metric("Claims", h["claims"])
    c[2].metric("Contested slots", len(h["contested_slots"]))
    c[3].metric("Open doc-debt", len(h["open_doc_debt"]))
    st.subheader("Contested slots")
    st.dataframe(pd.DataFrame(h["contested_slots"]).astype(str) if h["contested_slots"] else pd.DataFrame(
        [{"info": "none"}]), hide_index=True)
    st.subheader("Slots with no documentation")
    st.dataframe(pd.DataFrame(h["uncovered_slots"]) if h["uncovered_slots"] else pd.DataFrame([{"info": "none"}]),
                 hide_index=True)
    st.subheader("Quarantined documents")
    for d in h["quarantined_docs"]:
        st.error(f"{d['doc_id']}: {d['reason']}")
    if not h["quarantined_docs"]:
        st.caption("none")
    st.subheader("Doc-debt tickets")
    st.dataframe(pd.DataFrame(h["open_doc_debt"]).astype(str) if h["open_doc_debt"] else pd.DataFrame(
        [{"info": "none"}]), hide_index=True, use_container_width=True)
    st.subheader("Attestations nearing expiry (14 days)")
    st.dataframe(pd.DataFrame(h["attestations_expiring_soon"]) if h["attestations_expiring_soon"] else pd.DataFrame(
        [{"info": "none"}]), hide_index=True)


def page_audit():
    st.title("Audit")
    st.caption("Every ingest is a commit: hash = sha256(parent_hash + doc_id + doc_sha256 + recorded_at).")
    commits = ledger.list_commits(store())
    st.dataframe(pd.DataFrame([{"seq": c.seq, "doc": c.doc_id, "recorded_at": c.recorded_at.isoformat(),
                                "doc_sha256": c.doc_sha256[:12] + "…", "parent": c.parent_hash[:12] + "…",
                                "hash": c.hash[:12] + "…"} for c in commits]), hide_index=True,
                 use_container_width=True)
    c1, c2 = st.columns(2)
    if c1.button("Verify chain", type="primary"):
        ok, bad = ledger.verify_chain(store())
        (st.success("✅ Chain intact" if ok else f"❌ Chain broken at commit #{bad}"))
    if c2.button("Tamper test (on a temp copy)"):
        tmp = Store(":memory:")
        try:
            for c in commits:
                tmp.x("INSERT INTO commits VALUES (?,?,?,?,?,?)",
                      [c.seq, c.doc_id, c.doc_sha256, c.recorded_at, c.parent_hash, c.hash])
            target = max(1, len(commits) // 2)
            tmp.x("UPDATE commits SET doc_sha256='tampered' WHERE seq=?", [target])
            ok, bad = ledger.verify_chain(tmp)
            st.error(f"Edited commit #{target} in a temp copy → verification {'passed?!' if ok else 'FAILED'} "
                     f"(first bad commit: #{bad}). The real ledger is untouched.")
        finally:
            tmp.close()


def page_eval():
    st.title("Eval")
    st.caption("ARBITER vs a naive top-k RAG baseline on the labelled questions, plus ablations.")
    if llm.is_mock():
        st.info("Running in offline mock mode: numbers are indicative of the architecture, not of a real LLM.")
    full = st.checkbox("Include ablations", value=True)
    if st.button("Run eval", type="primary"):
        from truthguard import engine
        from truthguard.agents.graph import Flags
        cfgs = [("ARBITER (full)", Flags())]
        if full:
            cfgs += [("no-expansion", Flags(expansion=False)), ("no-critic", Flags(critic=False)),
                     ("no-expansion + no-critic", Flags(expansion=False, critic=False)),
                     ("old-rule-order", Flags(rule_order=engine.OLD_ORDER))]
        with st.spinner("Running…"):
            res = tg_eval.run_eval(cfgs)
        st.markdown(tg_eval.table(res))
        tg_eval.RESULTS_PATH.write_text(__import__("json").dumps(res, indent=2, default=str), encoding="utf-8")


PAGES = {"Ask": page_ask, "Upload / Truth Diff": page_upload, "Attestations": page_attest,
         "Corpus Health": page_health, "Audit": page_audit, "Eval": page_eval}

PAGES[sidebar()]()
