"""TruthGuard MCP server (stdio). Tools: verify, ingest, attest, health_report.

Run:  python mcp_server.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:  # mcp 2.x renamed FastMCP -> MCPServer
    from mcp.server.mcpserver import MCPServer as FastMCP  # noqa: E402
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP  # noqa: E402

from truthguard import api  # noqa: E402

mcp = FastMCP("truthguard")


@mcp.tool()
def verify(question: str, scope: str = "general", as_of: Optional[str] = None,
           known_as_of: Optional[str] = None) -> dict:
    """Answer a policy question with a Conflict Card (verdict, provenance, rule trace).

    scope: general | enterprise | prepaid. as_of / known_as_of: ISO dates for audit replay.
    """
    return api.verify(question, scope, as_of, known_as_of)


@mcp.tool()
def ingest(path: str) -> dict:
    """Ingest a markdown document (path, or a file name inside corpus/incoming). Returns commit + Truth Diff."""
    return api.ingest(path)


@mcp.tool()
def attest(slot: str, value: str, statement: str, attester: str, role: str, expiry_days: int = 90,
           scope: str = "general") -> dict:
    """Record an expert attestation for an undocumented fact; it expires after ``expiry_days``."""
    return api.attest(slot, value, statement, attester, role, expiry_days, scope)


@mcp.tool()
def health_report() -> dict:
    """Corpus health: contested slots, quarantined docs, doc-debt tickets, attestations nearing expiry."""
    return api.health_report()


if __name__ == "__main__":
    mcp.run()
