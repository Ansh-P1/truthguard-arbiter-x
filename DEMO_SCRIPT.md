# Demo script (about 6 minutes)

Setup: delete `data/ledger.duckdb` for a clean slate, then `streamlit run app.py`. The seed corpus loads on first
start. (Optional: sidebar → Clock, to jump time for step 8.)

1. **Plain question** (Ask → "When is customer support available?") → **SETTLED**, `24x7 on 198`.
   *"Not paranoid: when there is no conflict it just answers, with the source quote."*
2. **Supersession** (Ask → "How long do I have to dispute my bill?") → **45 days (SUPERSEDED)**. Open *Agent trace*:
   the naive draft cites a stale/outranked claim, the critic raises attacks, and AMD-3 appears under "found ONLY via
   slot expansion". Show the argument graph: AMD-3 green, BP-V2 / BP-V1 / MEMO red, FAQ-2023 grey (echo).
3. **Truth Diff**: Upload / Truth Diff → `RESC.md` → Ingest. Flip table: `billing.dispute_window / general:
   SUPERSEDED 45 → SUPERSEDED 30 (R1 withdrawal)`; enterprise unchanged at 60. Ask again → **30**, AMD-3 OUT by the
   withdrawal, BP-V2 reinstated.
4. **Scope + rule order**: scope `enterprise` → **60** (BP-V2 loses by R3 lex specialis). Scope `prepaid` → **30**:
   the Prepaid FAQ (7) loses by **R2**, because a tier-1 FAQ cannot outrank a tier-3 policy.
5. **Honest tie**: "How long are call records kept?" → **CONTESTED**: SEC (90) vs CMP (180), same authority and date.
   FAQ-2025 is shown as an *echo* of SEC, not a third side. A doc-debt ticket is created (see Corpus Health).
6. **Time machine**: Ask the dispute question with *Audit replay* on, date `2026-02-01` → "On 2026-02-01 we said 45;
   with what we know now, the right answer for that date is 30, because RESC was recorded on <upload date>."
   Then **Audit** → Verify chain (green) → Tamper test (red: verification fails on the edited copy).
7. **Poisoned doc**: Upload `VENDOR.md` → red **QUARANTINED** banner (injection pattern). Ask the dispute question:
   still 30, and "365" appears nowhere.
8. **Undocumented truth**: "Can I cancel a port-out request after submitting it?" → **UNSUPPORTED** ("Attest this"
   button, ticket opened). Attestations → create `24` by "MNP Ops Lead", 90 days → ask again → **ATTESTED**. Sidebar
   Clock → set 'now' 91 days ahead → Apply → ask again → UNSUPPORTED (status `expired`). Reset the clock.
9. **MCP**: from Claude Code or Claude Desktop call `verify("How long do I have to dispute my bill?")` → the same Conflict
   Card as JSON (see README for the config).
10. **Eval**: Eval page → Run eval: ARBITER vs baseline plus ablations.

Fallback if anything misbehaves live: set `TG_USE_GOLD=1` so ingestion uses the gold claims; the demo never depends
on a model's extraction being right.
