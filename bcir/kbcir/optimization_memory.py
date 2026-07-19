"""Bounded dual-rail optimization memory.

The fuzzy rail is an exact Q15 squared-distance reference over immutable embeddings.
The hard rail is a canonical set of hardware/program facts.  A similar historical
pattern is returned only when every required fact is present; similarity can rank an
already-admissible shard, never relax a BCIR law or fabricate a hardware capability.

This is deliberately not a production vector database or a Code Property Graph store.
It defines the portable contract and oracle that a future FAISS/ScaNN/graph adapter must
match before being admitted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .._artifact_json import strict_json_loads

_SCHEMA = "bcir.optimization_memory.v1"
_MAX_JSON = 64 * 1024 * 1024
_MAX_PATTERNS = 65536
_MAX_FACTS = 262144
_MAX_DIMENSION = 4096
_Q15_MIN = -(1 << 15)
_Q15_MAX = (1 << 15) - 1


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _digest(value, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def _text(value, field: str) -> str:
    if (not isinstance(value, str) or not value or len(value.encode("utf-8")) > 4096
            or any(ord(character) < 0x20 for character in value)):
        raise ValueError(f"{field} must be a bounded, control-free string")
    return value


@dataclass(frozen=True, order=True)
class OptimizationFact:
    subject: str
    predicate: str
    value: str

    def __post_init__(self) -> None:
        _text(self.subject, "optimization fact subject")
        _text(self.predicate, "optimization fact predicate")
        _text(self.value, "optimization fact value")

    @property
    def digest(self) -> str:
        return _sha(asdict(self))


@dataclass(frozen=True)
class OptimizationPattern:
    pattern_sha256: str
    embedding_q15: tuple[int, ...]
    context_shard_sha256: str
    evidence_sha256: str
    required_facts: tuple[OptimizationFact, ...] = ()

    def __post_init__(self) -> None:
        _digest(self.pattern_sha256, "optimization pattern_sha256")
        _digest(self.context_shard_sha256, "optimization context_shard_sha256")
        _digest(self.evidence_sha256, "optimization evidence_sha256")
        if (not isinstance(self.embedding_q15, tuple) or not self.embedding_q15
                or len(self.embedding_q15) > _MAX_DIMENSION
                or any(type(value) is not int or not _Q15_MIN <= value <= _Q15_MAX
                       for value in self.embedding_q15)):
            raise ValueError("optimization embedding must be a bounded Q15 integer tuple")
        if (not isinstance(self.required_facts, tuple)
                or len(self.required_facts) > _MAX_FACTS
                or any(not isinstance(row, OptimizationFact) for row in self.required_facts)
                or tuple(sorted(self.required_facts)) != self.required_facts
                or len(set(self.required_facts)) != len(self.required_facts)):
            raise ValueError("optimization required facts must be unique and canonical")


@dataclass(frozen=True)
class OptimizationMemory:
    model_revision_sha256: str
    embedding_model_sha256: str
    patterns: tuple[OptimizationPattern, ...]
    facts: tuple[OptimizationFact, ...]
    schema: str = _SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA:
            raise ValueError("unsupported optimization-memory schema")
        _digest(self.model_revision_sha256, "optimization model_revision_sha256")
        _digest(self.embedding_model_sha256, "optimization embedding_model_sha256")
        if (not isinstance(self.patterns, tuple) or not self.patterns
                or len(self.patterns) > _MAX_PATTERNS
                or any(not isinstance(row, OptimizationPattern) for row in self.patterns)):
            raise ValueError("optimization memory needs a bounded typed pattern inventory")
        if (not isinstance(self.facts, tuple) or len(self.facts) > _MAX_FACTS
                or any(not isinstance(row, OptimizationFact) for row in self.facts)
                or tuple(sorted(self.facts)) != self.facts
                or len(set(self.facts)) != len(self.facts)):
            raise ValueError("optimization memory facts must be unique and canonical")
        if tuple(sorted(self.patterns, key=lambda row: row.pattern_sha256)) != self.patterns \
                or len({row.pattern_sha256 for row in self.patterns}) != len(self.patterns):
            raise ValueError("optimization patterns must have unique canonical identities")
        dimensions = {len(row.embedding_q15) for row in self.patterns}
        if len(dimensions) != 1:
            raise ValueError("optimization pattern embedding dimensions differ")
        known = set(self.facts)
        if any(not set(row.required_facts) <= known for row in self.patterns):
            raise ValueError("optimization pattern requires a fact absent from hard memory")

    @property
    def dimension(self) -> int:
        return len(self.patterns[0].embedding_q15)

    def to_dict(self) -> dict:
        return {"schema": self.schema, "model_revision_sha256": self.model_revision_sha256,
                "embedding_model_sha256": self.embedding_model_sha256,
                "patterns": [{"pattern_sha256": row.pattern_sha256,
                              "embedding_q15": list(row.embedding_q15),
                              "context_shard_sha256": row.context_shard_sha256,
                              "evidence_sha256": row.evidence_sha256,
                              "required_facts": [asdict(fact) for fact in row.required_facts]}
                             for row in self.patterns],
                "facts": [asdict(row) for row in self.facts]}

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @property
    def digest(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "OptimizationMemory":
        doc = strict_json_loads(text, "optimization memory", max_bytes=_MAX_JSON)
        fields = {"schema", "model_revision_sha256", "embedding_model_sha256",
                  "patterns", "facts"}
        if (not isinstance(doc, dict) or set(doc) != fields
                or not isinstance(doc["patterns"], list) or not doc["patterns"]
                or len(doc["patterns"]) > _MAX_PATTERNS
                or not isinstance(doc["facts"], list)
                or len(doc["facts"]) > _MAX_FACTS):
            raise ValueError("optimization memory has missing, unknown, or unbounded fields")
        try:
            facts = tuple(OptimizationFact(**row) for row in doc.pop("facts"))
            patterns = []
            for row in doc.pop("patterns"):
                if not isinstance(row, dict):
                    raise ValueError("optimization pattern entries must be objects")
                value = dict(row)
                value["embedding_q15"] = tuple(value["embedding_q15"])
                value["required_facts"] = tuple(
                    OptimizationFact(**fact) for fact in value["required_facts"])
                patterns.append(OptimizationPattern(**value))
            return cls(**doc, patterns=tuple(patterns), facts=facts)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed optimization memory: {exc}") from exc


@dataclass(frozen=True)
class OptimizationMatch:
    pattern_sha256: str
    context_shard_sha256: str
    squared_distance: int

    def __post_init__(self) -> None:
        _digest(self.pattern_sha256, "optimization match pattern_sha256")
        _digest(self.context_shard_sha256, "optimization match context_shard_sha256")
        if type(self.squared_distance) is not int or self.squared_distance < 0:
            raise ValueError("optimization match distance must be a non-negative integer")


def query_optimization_memory(memory: OptimizationMemory, embedding_q15,
                              *, top_k: int = 1,
                              active_facts=None) -> tuple[OptimizationMatch, ...]:
    """Return exact nearest admissible patterns with lowest-digest tie breaking."""
    values, eligible = _prepare_optimization_query(
        memory, embedding_q15, top_k=top_k, active_facts=active_facts)
    matches = []
    for pattern, admitted in zip(memory.patterns, eligible):
        if not admitted:
            continue
        distance = sum((left - right) * (left - right)
                       for left, right in zip(values, pattern.embedding_q15))
        matches.append(OptimizationMatch(pattern.pattern_sha256,
                                         pattern.context_shard_sha256, distance))
    matches.sort(key=lambda row: (row.squared_distance, row.pattern_sha256))
    return tuple(matches[:top_k])


def _prepare_optimization_query(memory: OptimizationMemory, embedding_q15, *,
                                top_k: int, active_facts) -> tuple[tuple[int, ...],
                                                                  tuple[bool, ...]]:
    """Shared strict boundary for the Python oracle and optional C data plane."""
    if not isinstance(memory, OptimizationMemory):
        raise ValueError("optimization query requires an OptimizationMemory")
    if not isinstance(embedding_q15, tuple | list) \
            or len(embedding_q15) != memory.dimension \
            or any(type(value) is not int or not _Q15_MIN <= value <= _Q15_MAX
                   for value in embedding_q15):
        raise ValueError("optimization query embedding has the wrong Q15 shape")
    if type(top_k) is not int or not 1 <= top_k <= min(1024, len(memory.patterns)):
        raise ValueError("optimization query top_k is outside the bounded pattern inventory")
    if active_facts is None:
        admitted = set(memory.facts)
    elif (not isinstance(active_facts, tuple | list | set | frozenset)
          or any(not isinstance(row, OptimizationFact) for row in active_facts)):
        raise ValueError("optimization query active_facts must be typed facts")
    else:
        admitted = set(active_facts)
        if not admitted <= set(memory.facts):
            raise ValueError("optimization query supplied a fact outside hard memory")
    return tuple(embedding_q15), tuple(set(pattern.required_facts) <= admitted
                                      for pattern in memory.patterns)


__all__ = ["OptimizationFact", "OptimizationMatch", "OptimizationMemory",
           "OptimizationPattern", "query_optimization_memory"]
