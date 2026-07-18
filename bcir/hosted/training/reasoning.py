"""Bounded, verifier-driven test-time compute for explicit reasoning candidates."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable

from .contracts import canonical_json, require_int, token_tuple


@dataclass(frozen=True)
class SearchBudget:
    rounds: int
    candidates_per_round: int
    max_candidate_tokens: int
    max_total_tokens: int

    def __post_init__(self) -> None:
        for field in ("rounds", "candidates_per_round", "max_candidate_tokens",
                      "max_total_tokens"):
            require_int(getattr(self, field), field, minimum=1, maximum=1_000_000)
        if self.max_total_tokens < self.max_candidate_tokens:
            raise ValueError("max_total_tokens must cover at least one candidate")


@dataclass(frozen=True)
class ReasoningCandidate:
    trace_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]
    producer: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_ids", token_tuple(
            self.trace_ids, "trace_ids", allow_empty=True))
        object.__setattr__(self, "answer_ids", token_tuple(self.answer_ids, "answer_ids"))
        if not isinstance(self.producer, str) or not self.producer:
            raise ValueError("candidate producer must be nonempty")

    @property
    def digest(self) -> str:
        raw = canonical_json({"trace_ids": self.trace_ids, "answer_ids": self.answer_ids,
                              "producer": self.producer})
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @property
    def token_count(self) -> int:
        return len(self.trace_ids) + len(self.answer_ids)


@dataclass(frozen=True)
class Verification:
    accepted: bool
    score: float
    verifier: str

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool or not isinstance(self.verifier, str) or not self.verifier:
            raise ValueError("verification has invalid fields")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)) \
                or not math.isfinite(float(self.score)):
            raise ValueError("verification score must be finite")


@dataclass(frozen=True)
class SearchResult:
    selected: ReasoningCandidate
    verification: Verification
    proposed: int
    verified: int
    rejected_by_budget: int
    duplicate_candidates: int
    total_tokens: int
    exhausted: bool


def bounded_reasoning_search(prompt_ids, budget: SearchBudget,
                             propose: Callable[[tuple[int, ...], int, int], list[ReasoningCandidate]],
                             verify: Callable[[tuple[int, ...], ReasoningCandidate], Verification]
                             ) -> SearchResult:
    prompt = token_tuple(prompt_ids, "prompt_ids")
    if not isinstance(budget, SearchBudget) or not callable(propose) or not callable(verify):
        raise ValueError("budget, proposer, and verifier are required")
    seen = set()
    scored = []
    proposed = verified = rejected = duplicates = total = 0
    exhausted = False
    for round_index in range(budget.rounds):
        candidates = propose(prompt, round_index, budget.candidates_per_round)
        if not isinstance(candidates, list) or len(candidates) > budget.candidates_per_round \
                or any(not isinstance(candidate, ReasoningCandidate) for candidate in candidates):
            raise ValueError("proposer violated the bounded candidate contract")
        proposed += len(candidates)
        for candidate in candidates:
            if candidate.digest in seen:
                duplicates += 1
                continue
            seen.add(candidate.digest)
            if candidate.token_count > budget.max_candidate_tokens \
                    or total + candidate.token_count > budget.max_total_tokens:
                rejected += 1
                exhausted = total + candidate.token_count > budget.max_total_tokens
                continue
            total += candidate.token_count
            verdict = verify(prompt, candidate)
            if not isinstance(verdict, Verification):
                raise ValueError("verifier returned an invalid result")
            verified += 1
            scored.append((candidate, verdict))
        if exhausted:
            break
    if not scored:
        raise RuntimeError("reasoning search produced no candidate within budget")
    # Accepted candidates always outrank rejected ones; then score, shorter work, and
    # content digest provide a stable replay tie-break.
    candidate, verdict = min(scored, key=lambda item: (
        not item[1].accepted, -float(item[1].score), item[0].token_count, item[0].digest))
    return SearchResult(candidate, verdict, proposed, verified, rejected, duplicates,
                        total, exhausted)
