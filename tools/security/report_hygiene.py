#!/usr/bin/env python3
"""Predicates every rail's REPORT path needs, in one place.

Two of them, both hoisted out of a rail that already had them right while
its siblings did not — the campaign's most-repeated defect shape (L14):

* ``strict_loads`` — JSON that refuses an ambiguous document. ``json.loads``
  keeps the LAST value for a repeated key, so a document declaring
  ``"runtime": ["hidden-package==1"]`` and later ``"runtime": []`` parses
  clean and audits clean while saying two different things. A gate cannot
  attribute a contradictory declaration, so it refuses it (L4).
* ``mapped`` — a value function applied through the shapes a report is
  actually made of, so redaction reaches a string nested in a list in a
  dict rather than only a top-level field (L7).
"""

from __future__ import annotations

import json
from typing import Any, Callable

__all__ = ["DuplicateKeys", "mapped", "reject_duplicate_keys", "strict_loads"]


class DuplicateKeys(ValueError):
    """A JSON object repeated a key, so the document says two things."""


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``object_pairs_hook`` that refuses a repeated key."""
    # Single pass: a per-key list scan is quadratic, and one duplicate among
    # thousands of keys is enough to refuse the whole document.
    seen: set[str] = set()
    duplicates: set[str] = set()
    for key, _value in pairs:
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        raise DuplicateKeys(f"duplicate keys: {sorted(duplicates)}")
    return dict(pairs)


def strict_loads(text: str) -> Any:
    """``json.loads`` that refuses duplicate keys."""
    return json.loads(text, object_pairs_hook=reject_duplicate_keys)


def mapped(value: Any, fn: Callable[[str], str]) -> Any:
    """``fn`` applied to every string reachable through a report's shapes.

    Non-strings pass through unchanged, so booleans stay booleans and a
    verdict is not stringified on its way through a redaction pass.
    """
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, list):
        return [mapped(item, fn) for item in value]
    if isinstance(value, tuple):
        return tuple(mapped(item, fn) for item in value)
    if isinstance(value, dict):
        return {key: mapped(item, fn) for key, item in value.items()}
    return value
