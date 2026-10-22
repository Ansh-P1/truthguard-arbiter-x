"""M6: the MCP tool functions work when called directly (and the server object lists them)."""
import asyncio
import importlib
import sys
from pathlib import Path

import pytest

from truthguard import api
from truthguard.scenarios import build_state

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def server():
    api.set_store(build_state("base"))
    mod = importlib.import_module("mcp_server")
    yield mod
    api.set_store(None)


def test_tools_registered(server):
    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert names == {"verify", "ingest", "attest", "health_report"}


def test_verify_returns_conflict_card(server):
    card = server.verify("How long do I have to dispute my bill?")
    assert card["status"] == "SUPERSEDED" and card["value"] == "45"
    assert {r["doc_id"] for r in card["provenance"]} >= {"AMD-3", "BP-V2", "BP-V1"}


def test_ingest_then_verify_flips(server, monkeypatch):
    monkeypatch.setenv("TG_NOW", "2026-03-01T10:00:00")
    res = server.ingest("RESC.md")
    monkeypatch.setenv("TG_NOW", "2026-10-19T12:00:00")
    gen = next(r for r in res["truth_diff"] if r["scope"] == "general")
    assert gen["before"]["value"] == "45" and gen["after"]["value"] == "30"
    assert server.verify("How long do I have to dispute my bill?")["value"] == "30"


def test_vendor_ingest_quarantined(server):
    assert server.ingest("VENDOR.md")["quarantined"] is True


def test_attest_and_health(server):
    a = server.attest("mnp.portout_cancel_window", "24", "ok", "Priya", "MNP Ops Lead", 90)
    assert a["status"] == "active"
    assert server.verify("Can I cancel a port-out request?")["status"] == "ATTESTED"
    assert server.health_report()["attestations"][0]["status"] == "active"
