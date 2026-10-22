# System Permissions Log

Everything the build session did that touched the system beyond plain file edits in this project folder.
Rule I followed: nothing uploaded anywhere, nothing outside the project folder modified, nothing irreversible.

| # | Action | Scope | Reversible? |
|---|--------|-------|-------------|
| 1 | Read `TRUTHGUARD_BUILD_SPEC.md` | project folder | n/a (read-only) |
| 2 | Ran `python --version`, `py --list`, `uv --version`, `pip list` | read-only queries | n/a |
| 3 | Checked for `ollama` (not installed, so the app runs in offline mock-LLM mode) | read-only | n/a |
| 4 | Created virtual env `.venv/` with `uv venv` (Python 3.12) | inside project folder only | Yes: delete `.venv/` |
| 5 | `uv pip install` of duckdb, rank_bm25, mcp, pyvis, pytest, pydantic, pyyaml, numpy, langgraph, streamlit, ruff, httpx into `.venv` | network download from PyPI, installed only into `.venv`; global Python untouched | Yes: delete `.venv/` |
| 6 | Wrote project source, corpus, tests, docs | inside project folder only | Yes |
| 7 | Ran `pytest`, `python -m truthguard.eval`, smoke scripts using `.venv\Scripts\python.exe` | local processes, in-memory DuckDB or temp files | Yes |
| 8 | Spawned `mcp_server.py` as a local stdio subprocess to check a real MCP round-trip (no network port opened) | local process, exited when done; its temp DB in my scratchpad was deleted | Yes |
| 9 | Wrote generator scripts + temp DB files only to the session scratchpad dir (`%TEMP%\claude\...\scratchpad`) | outside project but session-private temp dir | Yes (temp) |
| 10 | pip package note: installed `mcp` resolved to 2.x, where `FastMCP` was renamed `MCPServer`; `mcp_server.py` supports both | code change only | Yes |

| 11 | Started the Streamlit dev server on `localhost:8765` through the built-in browser preview (reads `.claude/launch.json`, which I created in the project), clicked through the UI, then stopped the server | local-only port, loopback; no external exposure | Yes (server stopped; delete `.claude/launch.json` if unwanted) |
| 12 | The UI run created `data/ledger.duckdb` inside the project; I deleted `data/` afterwards (it is regenerated on next start) | project folder only | Yes |
| 13 | `ruff check --fix` auto-fixed import order/unused imports in my own source files | project files only | Yes |
| 14 | Created `.gitignore`, `pyproject.toml`, `requirements.txt` | project folder only | Yes |

## Things I deliberately did NOT do
* No git init/commit/push, no uploads, no cloud calls (only the PyPI download in #5).
* No changes outside the project folder (other than session-private temp files in the scratchpad).
* No global installs, no system/registry/PATH/security settings changes, no admin elevation.
* No deletions other than my own generated files (`data/`, scratch DBs, `.pytest_cache`).
* Ollama was not installed (a multi-GB download). The app runs in mock mode until you install it.
* No credentials or personal data entered anywhere.
