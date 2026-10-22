"""Glue: load bitemporal claims from the store and run the pure engine."""
from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Optional

from . import clock, engine
from .models import Verdict
from .store import Store


def scope_set(scope: Iterable[str] | str) -> frozenset:
    if isinstance(scope, str):
        scope = [scope]
    return frozenset(scope)


def resolve_slot(store: Store, slot: str, scope: Iterable[str] | str = "general",
                 as_of: Optional[date] = None, known_as_of: Optional[datetime] = None,
                 rule_order=engine.FIXED_ORDER) -> Verdict:
    """Resolve ``slot`` from everything the store knew at ``known_as_of`` for valid-time ``as_of``."""
    as_of = as_of or clock.today()
    known_as_of = known_as_of or clock.now()
    claims = store.engine_claims(as_of, known_as_of)
    return engine.resolve(claims, slot, scope_set(scope), as_of, known_as_of, rule_order)
