# TruthGuard ARBITER-X Lite

Multi-agent document Q&A for **Virtusa Jatayu S6, Use Case 2 (TruthGuard)**. The corpus is deliberately
inconsistent (superseded policies, contradicting FAQs, withdrawn amendments). The system never silently picks a
side: every answer comes with a **Conflict Card** (provenance, rule trace, argument graph), and conflicts are
decided by a **deterministic rule engine plus argumentation semantics, not an LLM's opinion** ("Truth-as-Code").

Pipeline: `Router → Retriever (similarity + slot expansion) → Synthesizer → Critic (contradiction / staleness /
unsupported / falsification, bounded redraft) → Resolution (pure-Python engine → templated answer)`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the diagram and rule table, and [DEMO_SCRIPT.md](DEMO_SCRIPT.md) for a
6-minute walkthrough. The build spec is [TRUTHGUARD_BUILD_SPEC.md](TRUTHGUARD_BUILD_SPEC.md).

## Setup (Windows / macOS / Linux)

```bash
uv venv .venv --python 3.12             # 3.11+ works
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # Linux/macOS: .venv/bin/python
.venv/Scripts/python.exe -m pytest -q                                   # 100+ tests, all offline
.venv/Scripts/python.exe -m streamlit run app.py                        # UI on http://localhost:8501
```

(No `uv`? `python -m venv .venv && .venv/Scripts/pip install -r requirements.txt` works too.)

### LLM backends (`truthguard/llm.py`)

| Mode | How it is chosen | Notes |
|---|---|---|
| `ollama` | auto if `http://localhost:11434` answers | `ollama pull qwen2.5:7b-instruct nomic-embed-text bespoke-minicheck`. Override with `TG_CHAT_MODEL`, `TG_EMBED_MODEL`, `TG_CHECK_MODEL`. |
| `openai` | `TG_OPENAI_BASE_URL` + `TG_OPENAI_API_KEY` set | any OpenAI-compatible endpoint (`TG_OPENAI_MODEL`, `TG_OPENAI_EMBED_MODEL`) |
| `mock` | `TG_LLM=mock`, or automatic when nothing is reachable | deterministic and fully offline: rule-based claim extraction and router, hashing-trick embeddings, lexical support checker |

**Important:** the build machine had no Ollama, so everything was verified in `mock` mode. The Ollama / OpenAI
code paths (structured output, MiniCheck, NLI) are implemented to spec but **have not been run against a real model
yet**. Every structured call has a pydantic-validated retry and a logged fallback, so a bad model response degrades
to the gold claims / `"none"` instead of crashing the UI.

Other env vars: `TG_DB` (default `data/ledger.duckdb`, `:memory:` allowed), `TG_USE_GOLD=1` (use
`corpus/gold_claims.json` instead of extraction), `TG_NOW` / `TG_TODAY` (clock overrides, see `clock.py`).

### Reset the demo
Delete `data/ledger.duckdb`. It is re-seeded from `corpus/seed/` on next start.

## Commands

```bash
python -m truthguard.ingest.pipeline --seed             # ingest corpus/seed
python -m truthguard.ingest.pipeline corpus/incoming/RESC.md
python -m truthguard.agents.graph "How long do I have to dispute my bill?"   # prints a Conflict Card
python -m truthguard.eval                               # ARBITER vs baseline + ablations -> eval_results.json
python -m truthguard.eval --no-expansion --no-critic --old-rule-order        # one custom ablation
python mcp_server.py                                    # MCP server (stdio)
```

## MCP server

Tools: `verify(question, scope, as_of, known_as_of)`, `ingest(path)`, `attest(slot, value, statement, attester,
role, expiry_days, scope)`, `health_report()`. Works with the `mcp` 1.x (`FastMCP`) and 2.x (`MCPServer`) SDKs.

Claude Desktop / Claude Code config:

```json
{
  "mcpServers": {
    "truthguard": {
      "command": "C:/Users/ansha/CODING_AND_PROJECTS_ETC/omnitrix-jatayu/.venv/Scripts/python.exe",
      "args": ["C:/Users/ansha/CODING_AND_PROJECTS_ETC/omnitrix-jatayu/mcp_server.py"],
      "env": { "TG_LLM": "mock" }
    }
  }
}
```

(Claude Code: `claude mcp add truthguard -- <python> <path>/mcp_server.py`. Drop `TG_LLM` once Ollama is running.)

## Results (mock mode, 29 labelled questions)

| Metric | Baseline RAG | ARBITER (full) | no-expansion | no-critic | no-exp + no-critic | old-rule-order |
|---|---|---|---|---|---|---|
| Answer accuracy (value questions) | 39% | 100% | 96% | 100% | 70% | 91% |
| Stale-answer rate (lower is better) | 56% | 0% | 6% | 0% | 39% | 11% |
| Correct-abstention rate | 50% | 100% | 100% | 100% | 100% | 100% |
| Conflict recall | 0% | 100% | 93% | 100% | 76% | 100% |
| Injection resistance | 33% | 100% | 100% | 100% | 67% | 100% |

Extraction (rule-based mock extractor vs gold): P/R/F1 = 100%.

How to read this honestly:

* In mock mode the "LLM" parts are rule-based. The 100% for ARBITER mostly shows that the **deterministic engine,
  slot expansion, bitemporal store and guard are correct** on this corpus. The paraphrases were written by the same
  author as the mock router's keyword list, so they are not an independent test of language understanding. Re-run
  `python -m truthguard.eval` with Ollama to measure the real thing.
* The baseline is a mock "naive reader" that trusts the best-matching chunk. It is intentionally simple, but its
  failure modes (stale values, no abstention on conflicts, following the poisoned doc) are the real ones.
* `no-critic` alone shows **no drop**: the final answer comes from the engine over the evidence set, and with slot
  expansion on the critic adds nothing. The critic's falsification retrieval matters only when expansion is off
  (`no-expansion` loses only Q4, `no-expansion + no-critic` loses seven questions). The critic's other job
  (drafts, attacks, redraft trace, proof certificates) is explanatory rather than accuracy-moving.

## Manual UI checklist

1. **Ask** `How long do I have to dispute my bill?` → SUPERSEDED, 45 days; provenance table; argument graph;
   agent trace shows AMD-3 found *only via slot expansion*.
2. **Upload / Truth Diff** → pick `VENDOR.md` → red quarantine banner; pick `RESC.md` → flip table
   `billing.dispute_window / general: 45 → 30 (R1 withdrawal)`; ask again → 30.
3. Scope = `enterprise` → 60 (R3); scope = `prepaid` → 30 (PFAQ loses by R2, not R3).
4. `How long are call records kept?` → CONTESTED, FAQ-2025 shown as an echo, a ticket is opened.
5. **Ask** with *Audit replay* on, date 2026-02-01 → "answer changed because RESC was recorded on …".
6. **Attestations**: ask the port-out question (UNSUPPORTED) → create a "24" attestation → ATTESTED → sidebar
   Clock, set 'now' 91 days ahead → UNSUPPORTED (status `expired`), Reset → ATTESTED again.
7. **Audit** → Verify chain (green) → Tamper test (red, on a temp copy). **Corpus Health**, **Eval** pages.

## Layout

`truthguard/` (package), `corpus/` (seed, incoming, `gold_claims.json`, `questions.json`), `tests/`, `app.py`,
`mcp_server.py`. Small additions beyond the spec's layout: `clock.py`, `verdicts.py` (store → engine glue),
`api.py` (shared by UI/MCP), `scenarios.py` (named corpus states for tests/eval).

## Known deviations from the spec

* Python 3.12 venv (spec says 3.11; `requires-python >= 3.11`).
* Attestations awaiting quorum use an extra status `pending`; expiry is computed from the clock at read time
  (status `expired` is displayed, not persisted) so that moving the demo clock back restores the attestation.
* The checker falls back to a lexical number/word-overlap test when neither MiniCheck nor the HF NLI model
  (needs `transformers`, not installed) is available.
* Out of scope per spec and not built: conformal calibration, Dawid-Skene, Z3, clingo, OCR, auth, Docker.
