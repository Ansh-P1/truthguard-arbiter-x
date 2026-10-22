"""Read markdown + YAML front matter."""
from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import yaml

from .. import clock, config
from ..models import Document

_FM = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.S)


def _as_date(v) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v)[:10])


def _as_dt(v) -> Optional[datetime]:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, date):
        return datetime(v.year, v.month, v.day, 9, 0, 0)
    return datetime.fromisoformat(str(v))


def sha256_text(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def parse_text(text: str, recorded_at: Optional[datetime] = None) -> Document:
    """Parse a markdown string. ``recorded_at`` (arg) > front matter > ``clock.now()``."""
    m = _FM.match(text.replace("\r\n", "\n"))
    if not m:
        raise ValueError("document has no YAML front matter")
    meta = yaml.safe_load(m.group(1)) or {}
    body = m.group(2).strip()
    for k in ("doc_id", "title", "doc_type", "effective_from"):
        if k not in meta:
            raise ValueError(f"front matter missing '{k}'")
    doc_type = str(meta["doc_type"])
    scope = meta.get("scope") or ["general"]
    if isinstance(scope, str):
        scope = [scope]
    return Document(
        doc_id=str(meta["doc_id"]), title=str(meta["title"]), doc_type=doc_type,
        tier=int(meta.get("tier", config.TIERS.get(doc_type, 1))), version=str(meta.get("version", "1.0")),
        effective_from=_as_date(meta["effective_from"]),
        recorded_at=recorded_at or _as_dt(meta.get("recorded_at")) or clock.now(),
        scope=[str(s) for s in scope],
        replaces=[str(x) for x in (meta.get("replaces") or [])],
        withdraws=[str(x) for x in (meta.get("withdraws") or [])],
        body=body, sha256=sha256_text(body),
    )


def parse_file(path: str | Path, recorded_at: Optional[datetime] = None) -> Document:
    return parse_text(Path(path).read_text(encoding="utf-8"), recorded_at)
