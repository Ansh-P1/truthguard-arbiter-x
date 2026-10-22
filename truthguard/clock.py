"""Single source of time. ``TG_NOW`` / ``TG_TODAY`` override the clock everywhere."""
from __future__ import annotations

import os
from datetime import date, datetime


def now() -> datetime:
    """Current knowledge time (naive). ``TG_NOW`` (ISO) overrides."""
    v = os.environ.get("TG_NOW")
    if v:
        return datetime.fromisoformat(v)
    return datetime.now().replace(microsecond=0)


def today() -> date:
    """Current valid-time date. ``TG_TODAY`` overrides, else the date of ``now()``."""
    v = os.environ.get("TG_TODAY")
    if v:
        return date.fromisoformat(v)
    if os.environ.get("TG_NOW"):
        return now().date()
    return date.today()
