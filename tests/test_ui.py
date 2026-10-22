"""M6: every Streamlit page renders and the main flows work (Streamlit AppTest, mock LLM, in-memory DB)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["Ask", "Upload / Truth Diff", "Attestations", "Corpus Health", "Audit", "Eval"]


@pytest.fixture()
def app(monkeypatch):
    from truthguard import api
    monkeypatch.setenv("TG_DB", ":memory:")
    api.set_store(None)
    at = AppTest.from_file(APP, default_timeout=120).run()
    yield at
    api.set_store(None)


def goto(at, page):
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(app, page):
    goto(app, page)
    assert app.title[0].value


def test_ask_flow_shows_card(app):
    app.text_input(key="q_input").set_value("How long do I have to dispute my bill?")
    app.button[0].click().run()
    assert not app.exception
    html = " ".join(m.value for m in app.markdown)
    assert "SUPERSEDED" in html
    assert any("45 days" in s.value for s in app.success)


def test_upload_vendor_shows_quarantine_and_resc_shows_diff(app):
    goto(app, "Upload / Truth Diff")
    app.selectbox[0].set_value("VENDOR.md").run()
    app.button[0].click().run()
    assert not app.exception and any("QUARANTINED" in e.value for e in app.error)
    app.selectbox[0].set_value("RESC.md").run()
    app.button[0].click().run()
    assert not app.exception and any("ingested" in s.value for s in app.success)
    assert app.dataframe


def test_audit_verify_and_tamper(app):
    goto(app, "Audit")
    app.button[0].click().run()
    assert any("intact" in s.value for s in app.success)
    app.button[1].click().run()
    assert any("FAILED" in e.value for e in app.error)


def test_attestation_creation(app):
    goto(app, "Attestations")
    app.button[0].click().run()   # form submit
    assert not app.exception and any("ATT-0001" in s.value for s in app.success)
