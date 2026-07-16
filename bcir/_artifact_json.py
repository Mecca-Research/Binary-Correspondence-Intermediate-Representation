"""Small fail-closed JSON helpers for installable BCIR artifacts.

These parsers deliberately reject duplicate keys and JSON's non-standard NaN/Infinity
constants.  Schema-specific validation remains with each artifact type.
"""

from __future__ import annotations

import json

DEFAULT_MAX_ARTIFACT_BYTES = 1 << 20


def _duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _bad_constant(value: str):
    raise ValueError(f"non-finite JSON constant {value!r}")


def strict_json_loads(text: str, label: str, *, max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES):
    if not isinstance(text, str):
        raise ValueError(f"{label} JSON must be text")
    try:
        size = len(text.encode("utf-8"))
    except UnicodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8 text") from exc
    if size > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes} bytes")
    try:
        return json.loads(text, object_pairs_hook=_duplicate_keys, parse_constant=_bad_constant)
    except (json.JSONDecodeError, TypeError, RecursionError) as exc:
        raise ValueError(f"invalid {label} JSON: {exc}") from exc


def read_bounded_text(path: str, label: str, *, max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES) -> str:
    try:
        with open(path, "rb") as stream:
            raw = stream.read(max_bytes + 1)
    except OSError as exc:
        raise ValueError(f"cannot read {label} {path!r}: {exc}") from exc
    if len(raw) > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes} bytes")
    try:
        return raw.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
