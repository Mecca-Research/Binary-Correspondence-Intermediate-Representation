"""Verified contracts for sequence interfaces, tokenizer adaptation, and model growth.

This dependency-free module is the semantic rail for a bounded set of tokenizer and
progressive-training experiments.  It deliberately separates exact byte/string identity,
cost accounting, and trainable-stage ownership from hosted tensor-framework experiments.
Learned scores can rank research candidates; they never change BCIR legality.

The admitted contracts are intentionally narrower than the papers that motivated them:

* minimal synchronized byte chunks and probability-conserving cross-tokenizer projection;
* continued-BPE expansion with an exact decomposition into the source vocabulary;
* bounded unigram segmentation and conservative UTF-8 candidate admission;
* multi-objective sequence-interface evidence and Pareto selection;
* causal prefix-stable time-series coding with finite scalar quantization (FSQ);
* deterministic low-rank binary token interfaces; and
* constructive depth growth under an explicit active-parameter budget.

Large tokenizer construction, glyph rendering/PCA, adversarial training, LoRA execution,
and accelerator kernels remain hosted or hardware-gated work.  The final growth-stage
lowering does pass through the normal R-law verifier and StreamPack pipeline.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass
from typing import Sequence

from ..abi.streampack_abi import encode
from ..gem.streampack import StreamPack, hydrate_pipelined
from ..model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from ..verify import verify_all, verify_smart_lowering
from .cost import TargetProfile, Theta
from .provenance import hash_module
from .realize import RealizationResult, optimize

_MAX_U63 = (1 << 63) - 1
_MAX_SEQUENCE_TOKENS = 1 << 20
_MAX_SEQUENCE_BYTES = 64 << 20
_MAX_SEGMENT_BYTES = 1 << 20
_MAX_DECOMPOSITION = 1 << 16
_MAX_SEGMENT_TRANSITIONS = 1 << 22
_MAX_EXPANSION_ELEMENTS = 1 << 22
_MAX_EVIDENCE_TRIALS = 4096
_Q20 = 1 << 20


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _integer(value, field: str, *, minimum: int = 0, maximum: int = _MAX_U63) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def _finite(
    value, field: str, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if (
        not math.isfinite(result)
        or minimum is not None
        and result < minimum
        or maximum is not None
        and result > maximum
    ):
        raise ValueError(f"{field} is outside its valid range")
    return result


def _sha256(value, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _checked_add(*values: int) -> int:
    total = 0
    for value in values:
        _integer(value, "checked-add operand")
        total += value
        if total > _MAX_U63:
            raise ValueError("sequence-interface integer addition exceeds u63")
    return total


def _checked_mul(*values: int) -> int:
    product = 1
    for value in values:
        _integer(value, "checked-multiply operand")
        if value and product > _MAX_U63 // value:
            raise ValueError("sequence-interface integer multiplication exceeds u63")
        product *= value
    return product


@dataclass(frozen=True)
class DecodedToken:
    """One token occurrence and its exact nonempty decoded byte payload."""

    token_id: int
    payload: bytes

    def __post_init__(self) -> None:
        _integer(self.token_id, "token_id", maximum=(1 << 31) - 1)
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("decoded token payload must be nonempty bytes")
        if len(self.payload) > _MAX_SEQUENCE_BYTES:
            raise ValueError("decoded token payload exceeds the bounded byte limit")


@dataclass(frozen=True)
class AlignedTokenChunk:
    """A minimal common decoded prefix, with half-open token ranges."""

    student_start: int
    student_end: int
    teacher_start: int
    teacher_end: int
    byte_length: int
    payload_sha256: str

    def __post_init__(self) -> None:
        for field in ("student_start", "teacher_start"):
            _integer(getattr(self, field), field, maximum=_MAX_SEQUENCE_TOKENS)
        _integer(
            self.student_end,
            "student_end",
            minimum=self.student_start + 1,
            maximum=_MAX_SEQUENCE_TOKENS,
        )
        _integer(
            self.teacher_end,
            "teacher_end",
            minimum=self.teacher_start + 1,
            maximum=_MAX_SEQUENCE_TOKENS,
        )
        _integer(self.byte_length, "byte_length", minimum=1, maximum=_MAX_SEQUENCE_BYTES)
        _sha256(self.payload_sha256, "payload_sha256")


def _decoded_tokens(value, field: str) -> tuple[DecodedToken, ...]:
    if (
        not isinstance(value, (tuple, list))
        or not value
        or len(value) > _MAX_SEQUENCE_TOKENS
        or any(not isinstance(item, DecodedToken) for item in value)
    ):
        raise ValueError(f"{field} must be a bounded nonempty DecodedToken sequence")
    return tuple(value)


def align_decoded_tokens(
    student, teacher, *, max_bytes: int = _MAX_SEQUENCE_BYTES
) -> tuple[AlignedTokenChunk, ...]:
    """Find every minimal synchronization chunk using exact decoded bytes.

    This is the dual-pointer catch-up algorithm.  Empty token decodings are rejected,
    equal-length unequal prefixes fail immediately, and both streams must represent the
    same complete byte string.  Operating on bytes avoids implicit Unicode normalization.
    """
    student = _decoded_tokens(student, "student")
    teacher = _decoded_tokens(teacher, "teacher")
    _integer(max_bytes, "max_bytes", minimum=1, maximum=_MAX_SEQUENCE_BYTES)
    student_total = sum(len(token.payload) for token in student)
    teacher_total = sum(len(token.payload) for token in teacher)
    if student_total > max_bytes or teacher_total > max_bytes:
        raise ValueError("decoded token stream exceeds max_bytes")
    if student_total != teacher_total:
        raise ValueError("student and teacher decoded streams have different lengths")

    chunks = []
    student_index = teacher_index = 0
    while student_index < len(student) and teacher_index < len(teacher):
        student_start, teacher_start = student_index, teacher_index
        student_payload = bytearray(student[student_index].payload)
        teacher_payload = bytearray(teacher[teacher_index].payload)
        student_index += 1
        teacher_index += 1
        while student_payload != teacher_payload:
            if len(student_payload) < len(teacher_payload):
                if student_index >= len(student):
                    raise ValueError("student stream cannot reach the next synchronization point")
                student_payload.extend(student[student_index].payload)
                student_index += 1
            elif len(teacher_payload) < len(student_payload):
                if teacher_index >= len(teacher):
                    raise ValueError("teacher stream cannot reach the next synchronization point")
                teacher_payload.extend(teacher[teacher_index].payload)
                teacher_index += 1
            else:
                raise ValueError("decoded streams disagree before a synchronization point")
        chunks.append(
            AlignedTokenChunk(
                student_start,
                student_index,
                teacher_start,
                teacher_index,
                len(student_payload),
                hashlib.sha256(student_payload).hexdigest(),
            )
        )
    if student_index != len(student) or teacher_index != len(teacher):
        raise ValueError("decoded streams do not terminate at a shared synchronization point")
    return tuple(chunks)


@dataclass(frozen=True)
class CrossTokenizerProjection:
    """Teacher chunk likelihoods projected onto student token occurrences."""

    projected_log_probabilities: tuple[float, ...]
    advantages: tuple[float, ...]
    student_chunk_log_probabilities: tuple[float, ...]
    teacher_chunk_log_probabilities: tuple[float, ...]
    projection_sha256: str

    def __post_init__(self) -> None:
        if not self.projected_log_probabilities or len(self.projected_log_probabilities) != len(
            self.advantages
        ):
            raise ValueError("cross-tokenizer projection arrays are malformed")
        for field in (
            "projected_log_probabilities",
            "advantages",
            "student_chunk_log_probabilities",
            "teacher_chunk_log_probabilities",
        ):
            values = getattr(self, field)
            if not isinstance(values, tuple) or any(not math.isfinite(value) for value in values):
                raise ValueError(f"{field} must contain finite values")
        _sha256(self.projection_sha256, "projection_sha256")


def project_aligned_log_probabilities(
    chunks: Sequence[AlignedTokenChunk],
    student_log_probabilities: Sequence[float],
    teacher_log_probabilities: Sequence[float],
) -> CrossTokenizerProjection:
    """Apply proportional chunk credit assignment while conserving teacher log mass."""
    if (
        not isinstance(chunks, (tuple, list))
        or not chunks
        or len(chunks) > _MAX_SEQUENCE_TOKENS
        or any(not isinstance(chunk, AlignedTokenChunk) for chunk in chunks)
    ):
        raise ValueError("chunks must be a bounded nonempty aligned-chunk sequence")
    if (
        not isinstance(student_log_probabilities, (tuple, list))
        or not student_log_probabilities
        or len(student_log_probabilities) > _MAX_SEQUENCE_TOKENS
    ):
        raise ValueError("student log probabilities must be a bounded nonempty sequence")
    if (
        not isinstance(teacher_log_probabilities, (tuple, list))
        or not teacher_log_probabilities
        or len(teacher_log_probabilities) > _MAX_SEQUENCE_TOKENS
    ):
        raise ValueError("teacher log probabilities must be a bounded nonempty sequence")
    student_logs = tuple(
        _finite(value, "student log probability", maximum=0.0)
        for value in student_log_probabilities
    )
    teacher_logs = tuple(
        _finite(value, "teacher log probability", maximum=0.0)
        for value in teacher_log_probabilities
    )
    if (
        chunks[0].student_start != 0
        or chunks[0].teacher_start != 0
        or chunks[-1].student_end != len(student_logs)
        or chunks[-1].teacher_end != len(teacher_logs)
    ):
        raise ValueError("aligned chunks do not cover both log-probability arrays")
    previous_student = previous_teacher = 0
    projected = [0.0] * len(student_logs)
    advantages = [0.0] * len(student_logs)
    student_sums = []
    teacher_sums = []
    for chunk in chunks:
        if chunk.student_start != previous_student or chunk.teacher_start != previous_teacher:
            raise ValueError("aligned chunks must be contiguous and ordered")
        student_sum = math.fsum(student_logs[chunk.student_start : chunk.student_end])
        teacher_sum = math.fsum(teacher_logs[chunk.teacher_start : chunk.teacher_end])
        if student_sum == 0.0:
            if teacher_sum != 0.0:
                raise ValueError("zero student chunk log mass cannot absorb teacher mass")
            scale = 1.0
        else:
            scale = teacher_sum / student_sum
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("cross-tokenizer projection produced an invalid scale")
        for index in range(chunk.student_start, chunk.student_end):
            projected[index] = scale * student_logs[index]
            advantages[index] = projected[index] - student_logs[index]
        if not math.isclose(
            math.fsum(projected[chunk.student_start : chunk.student_end]),
            teacher_sum,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("cross-tokenizer projection did not conserve chunk log mass")
        student_sums.append(student_sum)
        teacher_sums.append(teacher_sum)
        previous_student, previous_teacher = chunk.student_end, chunk.teacher_end
    identity = {
        "chunks": [asdict(chunk) for chunk in chunks],
        "projected": [value.hex() for value in projected],
        "advantages": [value.hex() for value in advantages],
    }
    return CrossTokenizerProjection(
        tuple(projected),
        tuple(advantages),
        tuple(student_sums),
        tuple(teacher_sums),
        _digest(identity),
    )


def canonical_substring_candidate(payload: bytes) -> bytes:
    """Conservatively admit a Thunder-Tok-style substring candidate.

    Valid UTF-8 is retained.  One invalid byte is admitted as an atomic byte-fallback
    candidate, while mixed or multi-byte invalid fragments are rejected.  A candidate
    spanning whitespace is truncated after its final whitespace so a suffix-array sample
    cannot retain an unfinished trailing word.
    """
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_SEGMENT_BYTES:
        raise ValueError("substring candidate must be bounded nonempty bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        if len(payload) == 1:
            return payload
        raise ValueError("candidate contains a mixed or incomplete UTF-8 fragment") from exc
    whitespace = [index for index, character in enumerate(text) if character.isspace()]
    if whitespace and whitespace[-1] + 1 < len(text):
        prefix = text[: whitespace[-1] + 1]
        if prefix.strip():
            return prefix.encode("utf-8")
    return payload


@dataclass(frozen=True)
class UnigramPiece:
    token_id: int
    payload: bytes
    log_probability: float

    def __post_init__(self) -> None:
        _integer(self.token_id, "unigram token_id", maximum=(1 << 31) - 1)
        canonical = canonical_substring_candidate(self.payload)
        if canonical != self.payload:
            raise ValueError("unigram piece retains an unfinished trailing word fragment")
        _finite(self.log_probability, "unigram log_probability", maximum=0.0)


def thunder_candidate_score(delta_log_likelihood: float, delta_branching_entropy: float) -> float:
    """Approximate uniform Jensen pruning score; final segmentation stays unigram-only."""
    likelihood = _finite(delta_log_likelihood, "delta_log_likelihood")
    branching = _finite(delta_branching_entropy, "delta_branching_entropy", minimum=0.0)
    return likelihood - branching


def segment_unigram(
    payload: bytes, pieces: Sequence[UnigramPiece], *, max_pieces: int = 65535
) -> tuple[int, ...]:
    """Find the highest-likelihood bounded exact byte segmentation by dynamic programming."""
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_SEGMENT_BYTES:
        raise ValueError("segmentation payload must be bounded nonempty bytes")
    _integer(max_pieces, "max_pieces", minimum=1, maximum=_MAX_SEQUENCE_TOKENS)
    if (
        not isinstance(pieces, (tuple, list))
        or not pieces
        or len(pieces) > max_pieces
        or any(not isinstance(piece, UnigramPiece) for piece in pieces)
    ):
        raise ValueError("pieces must be a bounded nonempty UnigramPiece sequence")
    token_ids = [piece.token_id for piece in pieces]
    if len(set(token_ids)) != len(token_ids) or len({piece.payload for piece in pieces}) != len(
        pieces
    ):
        raise ValueError("unigram pieces must have unique IDs and payloads")
    by_first: dict[int, list[UnigramPiece]] = {}
    for piece in pieces:
        by_first.setdefault(piece.payload[0], []).append(piece)
    for rows in by_first.values():
        rows.sort(key=lambda piece: (piece.token_id, piece.payload))
    # Backpointers keep the Viterbi pass linear in the admitted transitions.  Equal
    # likelihood prefers fewer pieces, then a stable final-token/predecessor key; no
    # progressively copied path tuple can turn a long byte stream into quadratic work.
    scores: list[float | None] = [None] * (len(payload) + 1)
    counts = [0] * (len(payload) + 1)
    parents = [-1] * (len(payload) + 1)
    final_tokens = [-1] * (len(payload) + 1)
    scores[0] = 0.0
    transitions = 0
    for start in range(len(payload)):
        if scores[start] is None:
            continue
        for piece in by_first.get(payload[start], ()):
            transitions += 1
            if transitions > _MAX_SEGMENT_TRANSITIONS:
                raise ValueError("unigram segmentation exceeds the bounded transition budget")
            end = start + len(piece.payload)
            if end > len(payload) or payload[start:end] != piece.payload:
                continue
            candidate_count = counts[start] + 1
            if candidate_count > _MAX_DECOMPOSITION:
                continue
            candidate_score = scores[start] + piece.log_probability
            previous_score = scores[end]
            previous_key = (final_tokens[end], parents[end])
            candidate_key = (piece.token_id, start)
            if (
                previous_score is None
                or candidate_score > previous_score
                or candidate_score == previous_score
                and candidate_count < counts[end]
                or candidate_score == previous_score
                and candidate_count == counts[end]
                and candidate_key < previous_key
            ):
                scores[end] = candidate_score
                counts[end] = candidate_count
                parents[end] = start
                final_tokens[end] = piece.token_id
    if scores[-1] is None:
        raise ValueError(
            "unigram vocabulary cannot exactly segment the byte payload within the token bound"
        )
    result = []
    cursor = len(payload)
    while cursor:
        result.append(final_tokens[cursor])
        cursor = parents[cursor]
        if cursor < 0:
            raise RuntimeError("unigram segmentation backpointer is malformed")
    result.reverse()
    return tuple(result)


def _merge_rows(value, field: str) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(value, (tuple, list)) or len(value) > 65535:
        raise ValueError(f"{field} must be a bounded merge sequence")
    rows = []
    for index, row in enumerate(value):
        if (
            not isinstance(row, (tuple, list))
            or len(row) != 3
            or any(type(item) is not int or item < 0 for item in row)
        ):
            raise ValueError(f"{field}[{index}] must contain three non-negative integers")
        rows.append(tuple(row))
    return tuple(rows)


@dataclass(frozen=True)
class ExpansionRow:
    target_token_id: int
    source_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _integer(self.target_token_id, "target_token_id", maximum=(1 << 31) - 1)
        if (
            not isinstance(self.source_token_ids, tuple)
            or not self.source_token_ids
            or len(self.source_token_ids) > _MAX_DECOMPOSITION
            or any(type(token) is not int or token < 0 for token in self.source_token_ids)
        ):
            raise ValueError("source_token_ids must be a bounded nonempty integer tuple")


@dataclass(frozen=True)
class ContinuedBPEExpansionPlan:
    """Proof that an expanded merge table preserves every source token ID."""

    base_vocab_size: int
    source_vocab_size: int
    target_vocab_size: int
    special_token_ids: tuple[int, ...]
    new_rows: tuple[ExpansionRow, ...]
    source_tokenizer_sha256: str
    target_tokenizer_sha256: str
    plan_sha256: str

    def __post_init__(self) -> None:
        _integer(self.base_vocab_size, "base_vocab_size", minimum=1, maximum=(1 << 31) - 1)
        _integer(
            self.source_vocab_size,
            "source_vocab_size",
            minimum=self.base_vocab_size,
            maximum=(1 << 31) - 1,
        )
        _integer(
            self.target_vocab_size,
            "target_vocab_size",
            minimum=self.source_vocab_size + 1,
            maximum=(1 << 31) - 1,
        )
        if (
            not isinstance(self.special_token_ids, tuple)
            or any(
                type(token) is not int or not 0 <= token < self.base_vocab_size
                for token in self.special_token_ids
            )
            or len(set(self.special_token_ids)) != len(self.special_token_ids)
        ):
            raise ValueError("special_token_ids must be unique source-base IDs")
        if (
            not isinstance(self.new_rows, tuple)
            or len(self.new_rows) != self.target_vocab_size - self.source_vocab_size
            or any(not isinstance(row, ExpansionRow) for row in self.new_rows)
        ):
            raise ValueError("new_rows do not cover the expanded vocabulary")
        if tuple(row.target_token_id for row in self.new_rows) != tuple(
            range(self.source_vocab_size, self.target_vocab_size)
        ):
            raise ValueError("expansion rows are not canonical and contiguous")
        if any(
            any(token >= self.source_vocab_size for token in row.source_token_ids)
            for row in self.new_rows
        ):
            raise ValueError("new token decomposition must terminate in the source vocabulary")
        decomposition_elements = 0
        for row in self.new_rows:
            decomposition_elements = _checked_add(decomposition_elements, len(row.source_token_ids))
            if decomposition_elements > _MAX_EXPANSION_ELEMENTS:
                raise ValueError("continued-BPE plan exceeds the aggregate decomposition budget")
        for field in ("source_tokenizer_sha256", "target_tokenizer_sha256", "plan_sha256"):
            _sha256(getattr(self, field), field)
        expected = _digest(
            {
                "source": self.source_tokenizer_sha256,
                "target": self.target_tokenizer_sha256,
                "rows": [asdict(row) for row in self.new_rows],
            }
        )
        if self.plan_sha256 != expected:
            raise ValueError("continued-BPE plan digest does not match its rows")


def plan_continued_bpe_expansion(
    source_merges, target_merges, *, base_vocab_size: int, special_token_ids: Sequence[int] = ()
) -> ContinuedBPEExpansionPlan:
    """Prove prefix-preserving BPE continuation and derive each new row's source mean."""
    _integer(base_vocab_size, "base_vocab_size", minimum=1, maximum=(1 << 31) - 1)
    source = _merge_rows(source_merges, "source_merges")
    target = _merge_rows(target_merges, "target_merges")
    if len(target) <= len(source) or target[: len(source)] != source:
        raise ValueError("target merges must be a strict continuation of source merges")
    special = tuple(special_token_ids)
    if any(type(token) is not int or not 0 <= token < base_vocab_size for token in special) or len(
        set(special)
    ) != len(special):
        raise ValueError("special_token_ids must be unique source-base IDs")

    def validate(rows: tuple[tuple[int, int, int], ...], field: str) -> None:
        known = set(range(base_vocab_size))
        for index, (left, right, token) in enumerate(rows):
            if left not in known or right not in known or token != base_vocab_size + index:
                raise ValueError(f"{field}[{index}] is non-canonical or references a future token")
            known.add(token)

    validate(source, "source_merges")
    validate(target, "target_merges")
    source_vocab_size = base_vocab_size + len(source)
    decomposition: dict[int, tuple[int, ...]] = {
        token: (token,) for token in range(source_vocab_size)
    }
    rows = []
    decomposition_elements = 0
    for left, right, token in target[len(source) :]:
        expanded_length = _checked_add(len(decomposition[left]), len(decomposition[right]))
        if expanded_length > _MAX_DECOMPOSITION:
            raise ValueError("expanded token decomposition exceeds the bounded limit")
        decomposition_elements = _checked_add(decomposition_elements, expanded_length)
        if decomposition_elements > _MAX_EXPANSION_ELEMENTS:
            raise ValueError("continued-BPE plan exceeds the aggregate decomposition budget")
        expanded = decomposition[left] + decomposition[right]
        decomposition[token] = expanded
        rows.append(ExpansionRow(token, expanded))
    source_sha = _digest(
        {"base_vocab_size": base_vocab_size, "merges": source, "special_token_ids": special}
    )
    target_sha = _digest(
        {"base_vocab_size": base_vocab_size, "merges": target, "special_token_ids": special}
    )
    identity = {"source": source_sha, "target": target_sha, "rows": [asdict(row) for row in rows]}
    return ContinuedBPEExpansionPlan(
        base_vocab_size,
        source_vocab_size,
        base_vocab_size + len(target),
        special,
        tuple(rows),
        source_sha,
        target_sha,
        _digest(identity),
    )


@dataclass(frozen=True)
class TokenizerExpansionCost:
    source_vocab_size: int
    target_vocab_size: int
    added_parameter_bytes: int
    additional_head_macs_per_token: int
    source_tokens_per_unit_q20: int
    target_tokens_per_unit_q20: int
    source_head_macs_per_unit: int
    target_head_macs_per_unit: int
    decode_compute_improves: bool
    cost_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "source_vocab_size",
            "target_vocab_size",
            "added_parameter_bytes",
            "additional_head_macs_per_token",
            "source_tokens_per_unit_q20",
            "target_tokens_per_unit_q20",
            "source_head_macs_per_unit",
            "target_head_macs_per_unit",
        ):
            _integer(getattr(self, field), field)
        if type(self.decode_compute_improves) is not bool:
            raise ValueError("decode_compute_improves must be bool")
        _sha256(self.cost_sha256, "cost_sha256")


def assess_tokenizer_expansion(
    plan: ContinuedBPEExpansionPlan,
    *,
    hidden_width: int,
    bytes_per_element: int,
    tied_head: bool,
    source_tokens_per_unit_q20: int,
    target_tokens_per_unit_q20: int,
) -> TokenizerExpansionCost:
    """Price vocabulary storage and output projection per source unit, not fertility alone."""
    if not isinstance(plan, ContinuedBPEExpansionPlan):
        raise ValueError("plan must be ContinuedBPEExpansionPlan")
    _integer(hidden_width, "hidden_width", minimum=1, maximum=1 << 20)
    _integer(bytes_per_element, "bytes_per_element", minimum=1, maximum=16)
    if type(tied_head) is not bool:
        raise ValueError("tied_head must be bool")
    _integer(source_tokens_per_unit_q20, "source_tokens_per_unit_q20", minimum=1)
    _integer(target_tokens_per_unit_q20, "target_tokens_per_unit_q20", minimum=1)
    added = plan.target_vocab_size - plan.source_vocab_size
    matrices = 1 if tied_head else 2
    parameter_bytes = _checked_mul(added, hidden_width, bytes_per_element, matrices)
    additional_head_macs = _checked_mul(added, hidden_width)
    source_macs = (
        _checked_mul(plan.source_vocab_size, hidden_width, source_tokens_per_unit_q20) // _Q20
    )
    target_macs = (
        _checked_mul(plan.target_vocab_size, hidden_width, target_tokens_per_unit_q20) // _Q20
    )
    identity = {
        "plan": plan.plan_sha256,
        "hidden_width": hidden_width,
        "bytes_per_element": bytes_per_element,
        "tied_head": tied_head,
        "source_tokens_per_unit_q20": source_tokens_per_unit_q20,
        "target_tokens_per_unit_q20": target_tokens_per_unit_q20,
        "added_parameter_bytes": parameter_bytes,
        "source_head_macs_per_unit": source_macs,
        "target_head_macs_per_unit": target_macs,
    }
    return TokenizerExpansionCost(
        plan.source_vocab_size,
        plan.target_vocab_size,
        parameter_bytes,
        additional_head_macs,
        source_tokens_per_unit_q20,
        target_tokens_per_unit_q20,
        source_macs,
        target_macs,
        target_macs < source_macs,
        _digest(identity),
    )


@dataclass(frozen=True)
class SequenceInterfaceEvidence:
    """One multi-objective tokenizer/codec trial; no aggregate score is invented."""

    name: str
    interface_sha256: str
    corpus_sha256: str
    source_units: int
    token_count: int
    artifact_bytes: int
    encode_ns: int
    decode_ns: int
    reconstruction_error_q20: int
    linear_probe_q20: int
    nonlinear_probe_q20: int
    domain_probe_q20: int
    exact_roundtrip: bool

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("sequence-interface evidence name must be nonempty")
        _sha256(self.interface_sha256, "interface_sha256")
        _sha256(self.corpus_sha256, "corpus_sha256")
        for field in ("source_units", "token_count"):
            _integer(getattr(self, field), field, minimum=1)
        for field in ("artifact_bytes", "encode_ns", "decode_ns", "reconstruction_error_q20"):
            _integer(getattr(self, field), field)
        for field in ("linear_probe_q20", "nonlinear_probe_q20", "domain_probe_q20"):
            _integer(getattr(self, field), field, maximum=_Q20)
        if type(self.exact_roundtrip) is not bool:
            raise ValueError("exact_roundtrip must be bool")

    @property
    def tokens_per_unit_q20(self) -> int:
        return self.token_count * _Q20 // self.source_units

    @property
    def digest(self) -> str:
        return _digest(asdict(self))


def pareto_sequence_interfaces(
    evidence: Sequence[SequenceInterfaceEvidence],
) -> tuple[SequenceInterfaceEvidence, ...]:
    """Return the deterministic non-dominated frontier across all recorded objectives."""
    if (
        not isinstance(evidence, (tuple, list))
        or not evidence
        or len(evidence) > _MAX_EVIDENCE_TRIALS
        or any(not isinstance(row, SequenceInterfaceEvidence) for row in evidence)
    ):
        raise ValueError("evidence must be a bounded nonempty SequenceInterfaceEvidence sequence")
    rows = tuple(evidence)
    if len({row.digest for row in rows}) != len(rows):
        raise ValueError("sequence-interface evidence contains duplicate trials")

    def dominates(left: SequenceInterfaceEvidence, right: SequenceInterfaceEvidence) -> bool:
        left_min = (
            left.tokens_per_unit_q20,
            left.artifact_bytes,
            left.encode_ns,
            left.decode_ns,
            left.reconstruction_error_q20,
        )
        right_min = (
            right.tokens_per_unit_q20,
            right.artifact_bytes,
            right.encode_ns,
            right.decode_ns,
            right.reconstruction_error_q20,
        )
        left_max = (
            left.linear_probe_q20,
            left.nonlinear_probe_q20,
            left.domain_probe_q20,
            int(left.exact_roundtrip),
        )
        right_max = (
            right.linear_probe_q20,
            right.nonlinear_probe_q20,
            right.domain_probe_q20,
            int(right.exact_roundtrip),
        )
        no_worse = all(a <= b for a, b in zip(left_min, right_min)) and all(
            a >= b for a, b in zip(left_max, right_max)
        )
        strictly = any(a < b for a, b in zip(left_min, right_min)) or any(
            a > b for a, b in zip(left_max, right_max)
        )
        return no_worse and strictly

    frontier = [
        row for row in rows if not any(other is not row and dominates(other, row) for other in rows)
    ]
    return tuple(sorted(frontier, key=lambda row: (row.name, row.digest)))


@dataclass(frozen=True)
class FiniteScalarQuantizerSpec:
    """Cartesian-product FSQ levels with no learned codebook."""

    levels: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.levels, tuple) or not self.levels or len(self.levels) > 16:
            raise ValueError("FSQ levels must be a nonempty tuple with at most 16 dimensions")
        codebook = 1
        for index, level in enumerate(self.levels):
            _integer(level, f"levels[{index}]", minimum=3, maximum=255)
            if level % 2 == 0:
                raise ValueError("FSQ levels must be odd so zero is exactly representable")
            codebook = _checked_mul(codebook, level)
            if codebook > (1 << 31) - 1:
                raise ValueError("FSQ Cartesian codebook exceeds the bounded limit")

    @property
    def codebook_size(self) -> int:
        return math.prod(self.levels)

    def encode(self, values: Sequence[float]) -> int:
        if not isinstance(values, (tuple, list)) or len(values) != len(self.levels):
            raise ValueError("FSQ input must match the configured dimensionality")
        code = 0
        multiplier = 1
        for index, (value, levels) in enumerate(zip(values, self.levels)):
            value = max(-1.0, min(1.0, _finite(value, f"FSQ value {index}")))
            raw = (value + 1.0) * 0.5 * (levels - 1)
            digit = min(levels - 1, int(math.floor(raw + 0.5)))
            code += digit * multiplier
            multiplier *= levels
        return code

    def decode(self, code: int) -> tuple[float, ...]:
        _integer(code, "FSQ code", maximum=self.codebook_size - 1)
        values = []
        for levels in self.levels:
            digit = code % levels
            code //= levels
            values.append(2.0 * digit / (levels - 1) - 1.0)
        return tuple(values)

    @property
    def digest(self) -> str:
        return _digest({"schema": "bcir.fsq.v1", "levels": self.levels})


@dataclass(frozen=True)
class CausalSeriesCodecSpec:
    """Incremental univariate codec with exact float32 anchors and causal FSQ residuals."""

    quantizer: FiniteScalarQuantizerSpec
    warmup_values: int = 2
    normalized_clip: float = 4.0
    scale_floor: float = 1.0e-6
    max_values: int = 1 << 20

    def __post_init__(self) -> None:
        if (
            not isinstance(self.quantizer, FiniteScalarQuantizerSpec)
            or len(self.quantizer.levels) != 1
        ):
            raise ValueError("causal series codec requires a one-dimensional FSQ")
        _integer(self.warmup_values, "warmup_values", minimum=2, maximum=1024)
        _finite(self.normalized_clip, "normalized_clip", minimum=1.0e-12)
        _finite(self.scale_floor, "scale_floor", minimum=1.0e-30)
        _integer(
            self.max_values, "max_values", minimum=self.warmup_values, maximum=_MAX_SEQUENCE_TOKENS
        )

    @property
    def vocabulary_size(self) -> int:
        # 0=exact tag, 1=FSQ tag, 2..257=raw byte, 258..=FSQ code.
        return 258 + self.quantizer.codebook_size

    def prefix_token_count(self, values: int) -> int:
        _integer(values, "values", maximum=self.max_values)
        exact = min(values, self.warmup_values)
        return exact * 5 + max(0, values - exact) * 2

    @property
    def digest(self) -> str:
        return _digest(
            {
                "schema": "bcir.causal_series_codec.v1",
                "quantizer": self.quantizer.digest,
                "warmup_values": self.warmup_values,
                "normalized_clip": self.normalized_clip.hex(),
                "scale_floor": self.scale_floor.hex(),
                "max_values": self.max_values,
            }
        )


@dataclass(frozen=True)
class CausalSeriesEncoding:
    tokens: tuple[int, ...]
    token_ends: tuple[int, ...]
    reconstructed_values: tuple[float, ...]
    maximum_absolute_error: float
    encoding_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.tokens, tuple)
            or not isinstance(self.token_ends, tuple)
            or not isinstance(self.reconstructed_values, tuple)
            or not self.tokens
            or not self.token_ends
            or not self.reconstructed_values
            or len(self.tokens) > _MAX_SEQUENCE_TOKENS * 5
            or len(self.token_ends) != len(self.reconstructed_values)
        ):
            raise ValueError("causal series encoding arrays are malformed")
        if any(type(token) is not int or token < 0 for token in self.tokens):
            raise ValueError("causal series tokens must be non-negative integers")
        if self.token_ends[-1] != len(self.tokens) or any(
            left >= right for left, right in zip((0,) + self.token_ends, self.token_ends)
        ):
            raise ValueError("causal series token ends must be strictly increasing")
        if any(not math.isfinite(value) for value in self.reconstructed_values):
            raise ValueError("causal series reconstruction must be finite")
        _finite(self.maximum_absolute_error, "maximum_absolute_error", minimum=0.0)
        _sha256(self.encoding_sha256, "encoding_sha256")


def _float32(value, field: str) -> float:
    value = _finite(value, field)
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as exc:
        raise ValueError(f"{field} is outside finite float32 range") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} is outside finite float32 range")
    return result


class _CausalMoments:
    """Online population moments; avoids rescanning every historical prefix."""

    __slots__ = ("count", "mean", "m2")

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def location_scale(self, floor: float) -> tuple[float, float]:
        if self.count < 1:
            raise RuntimeError("causal moments require a reconstructed prefix")
        variance = max(0.0, self.m2 / self.count)
        return self.mean, max(math.sqrt(variance), floor)


def encode_causal_series(
    values: Sequence[float], spec: CausalSeriesCodecSpec
) -> CausalSeriesEncoding:
    """Encode a finite prefix; extending input never changes previously emitted tokens."""
    if not isinstance(spec, CausalSeriesCodecSpec):
        raise ValueError("spec must be CausalSeriesCodecSpec")
    if not isinstance(values, (tuple, list)) or not values or len(values) > spec.max_values:
        raise ValueError("values must be a bounded nonempty sequence")
    source = tuple(_float32(value, f"values[{index}]") for index, value in enumerate(values))
    tokens: list[int] = []
    ends: list[int] = []
    reconstructed: list[float] = []
    errors = []
    moments = _CausalMoments()
    for index, value in enumerate(source):
        if index < spec.warmup_values:
            raw = struct.pack("<f", value)
            tokens.extend((0, *(byte + 2 for byte in raw)))
            reconstructed_value = struct.unpack("<f", raw)[0]
        else:
            mean, scale = moments.location_scale(spec.scale_floor)
            normalized = (value - mean) / (scale * spec.normalized_clip)
            code = spec.quantizer.encode((normalized,))
            tokens.extend((1, 258 + code))
            decoded = spec.quantizer.decode(code)[0]
            reconstructed_value = _float32(
                mean + decoded * scale * spec.normalized_clip, f"reconstructed_values[{index}]"
            )
        reconstructed.append(reconstructed_value)
        moments.add(reconstructed_value)
        errors.append(abs(value - reconstructed_value))
        ends.append(len(tokens))
    identity = {
        "spec": spec.digest,
        "tokens": tokens,
        "token_ends": ends,
        "reconstructed": [value.hex() for value in reconstructed],
    }
    return CausalSeriesEncoding(
        tuple(tokens), tuple(ends), tuple(reconstructed), max(errors), _digest(identity)
    )


def decode_causal_series(tokens: Sequence[int], spec: CausalSeriesCodecSpec) -> tuple[float, ...]:
    if not isinstance(spec, CausalSeriesCodecSpec):
        raise ValueError("spec must be CausalSeriesCodecSpec")
    if (
        not isinstance(tokens, (tuple, list))
        or not tokens
        or len(tokens) > spec.max_values * 5
        or any(type(token) is not int or token < 0 for token in tokens)
    ):
        raise ValueError("tokens must be a bounded nonempty integer sequence")
    result = []
    moments = _CausalMoments()
    index = 0
    while index < len(tokens):
        if len(result) >= spec.max_values:
            raise ValueError("causal series token stream exceeds max_values")
        tag = tokens[index]
        if len(result) < spec.warmup_values:
            if tag != 0 or index + 5 > len(tokens):
                raise ValueError("causal series exact anchor is truncated or mistagged")
            raw_tokens = tokens[index + 1 : index + 5]
            if any(not 2 <= token <= 257 for token in raw_tokens):
                raise ValueError("causal series exact anchor contains an invalid byte token")
            value = struct.unpack("<f", bytes(token - 2 for token in raw_tokens))[0]
            if not math.isfinite(value):
                raise ValueError("causal series exact anchor is non-finite")
            index += 5
        else:
            if tag != 1 or index + 2 > len(tokens):
                raise ValueError("causal series FSQ value is truncated or mistagged")
            code = tokens[index + 1] - 258
            _integer(code, "causal series FSQ token", maximum=spec.quantizer.codebook_size - 1)
            mean, scale = moments.location_scale(spec.scale_floor)
            value = _float32(
                mean + spec.quantizer.decode(code)[0] * scale * spec.normalized_clip,
                f"decoded_values[{len(result)}]",
            )
            index += 2
        result.append(value)
        moments.add(value)
    return tuple(result)


@dataclass(frozen=True)
class FrozenTokenCodeSpec:
    """Canonical token-ID bits repeated into a fixed model-width interface."""

    vocab_size: int
    code_bits: int
    model_width: int

    def __post_init__(self) -> None:
        _integer(self.vocab_size, "vocab_size", minimum=2, maximum=1 << 20)
        _integer(self.code_bits, "code_bits", minimum=1, maximum=64)
        _integer(self.model_width, "model_width", minimum=self.code_bits, maximum=1 << 16)
        if self.vocab_size > 1 << self.code_bits:
            raise ValueError("code_bits cannot uniquely represent every token ID")

    def code(self, token_id: int) -> tuple[float, ...]:
        _integer(token_id, "token_id", maximum=self.vocab_size - 1)
        bits = tuple(1.0 if token_id & (1 << bit) else -1.0 for bit in range(self.code_bits))
        return tuple(bits[index % self.code_bits] for index in range(self.model_width))

    @property
    def rank_upper_bound(self) -> int:
        return min(self.vocab_size, self.code_bits, self.model_width)

    @property
    def digest(self) -> str:
        return _digest({"schema": "bcir.frozen_token_code.v1", **asdict(self)})


@dataclass(frozen=True)
class GrowthStage:
    stage_index: int
    kind: str
    total_layers: int
    new_block_start: int
    new_block_end: int
    trainable_blocks: tuple[int, ...]
    lora_layers: tuple[int, ...]
    lora_rank: int
    active_parameter_elements: int
    optimizer_state_bytes: int

    def __post_init__(self) -> None:
        _integer(self.stage_index, "stage_index", maximum=1024)
        if self.kind not in ("dense_growth", "lora_readjustment"):
            raise ValueError("growth stage kind must be dense_growth or lora_readjustment")
        _integer(self.total_layers, "total_layers", minimum=1, maximum=1024)
        _integer(self.new_block_start, "new_block_start", maximum=self.total_layers)
        _integer(
            self.new_block_end,
            "new_block_end",
            minimum=self.new_block_start,
            maximum=self.total_layers,
        )
        for field in ("trainable_blocks", "lora_layers"):
            values = getattr(self, field)
            if (
                not isinstance(values, tuple)
                or any(
                    type(layer) is not int or not 0 <= layer < self.total_layers for layer in values
                )
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"{field} must contain unique in-range layer IDs")
        _integer(self.lora_rank, "lora_rank", maximum=1 << 16)
        _integer(self.active_parameter_elements, "active_parameter_elements", minimum=1)
        _integer(self.optimizer_state_bytes, "optimizer_state_bytes")
        if self.kind == "dense_growth":
            if (
                self.new_block_start == self.new_block_end
                or self.trainable_blocks != tuple(range(self.new_block_start, self.new_block_end))
                or self.lora_layers
                or self.lora_rank
            ):
                raise ValueError("dense growth must train exactly its newly added blocks")
        elif (
            self.trainable_blocks
            or self.lora_layers != tuple(range(self.total_layers))
            or self.lora_rank < 1
            or self.new_block_start != self.new_block_end
        ):
            raise ValueError("LoRA stage must adapt every active layer without adding blocks")


@dataclass(frozen=True)
class ProgressiveGrowthSpec:
    initial_layers: int
    final_layers: int
    growth_layers: int
    block_parameter_elements: int
    head_parameter_elements: int
    fixed_interface_elements: int
    interface_width: int
    active_parameter_budget: int
    optimizer_state_bytes_per_parameter: int = 8
    lora_rank: int = 0
    lora_elements_per_rank_per_layer: int = 0

    def __post_init__(self) -> None:
        _integer(self.initial_layers, "initial_layers", minimum=1, maximum=1024)
        _integer(self.final_layers, "final_layers", minimum=self.initial_layers, maximum=1024)
        _integer(self.growth_layers, "growth_layers", minimum=1, maximum=1024)
        for field in (
            "block_parameter_elements",
            "head_parameter_elements",
            "fixed_interface_elements",
            "interface_width",
            "active_parameter_budget",
        ):
            _integer(getattr(self, field), field, minimum=1)
        if self.fixed_interface_elements % self.interface_width:
            raise ValueError("fixed_interface_elements must contain complete interface rows")
        _integer(
            self.optimizer_state_bytes_per_parameter,
            "optimizer_state_bytes_per_parameter",
            minimum=1,
            maximum=64,
        )
        _integer(self.lora_rank, "lora_rank", maximum=1 << 16)
        _integer(self.lora_elements_per_rank_per_layer, "lora_elements_per_rank_per_layer")
        if bool(self.lora_rank) != bool(self.lora_elements_per_rank_per_layer):
            raise ValueError("LoRA rank and per-rank elements must be enabled together")
        # Construct now so invalid budgets fail before any hosted allocation or training.
        self.stages()

    def stages(self) -> tuple[GrowthStage, ...]:
        rows = []
        current = 0
        stage_index = 0
        while current < self.final_layers:
            added = (
                self.initial_layers
                if current == 0
                else min(self.growth_layers, self.final_layers - current)
            )
            end = current + added
            active = _checked_add(
                _checked_mul(added, self.block_parameter_elements), self.head_parameter_elements
            )
            if active > self.active_parameter_budget:
                raise ValueError("dense growth stage exceeds active_parameter_budget")
            rows.append(
                GrowthStage(
                    stage_index,
                    "dense_growth",
                    end,
                    current,
                    end,
                    tuple(range(current, end)),
                    (),
                    0,
                    active,
                    _checked_mul(active, self.optimizer_state_bytes_per_parameter),
                )
            )
            stage_index += 1
            current = end
            if self.lora_rank and current > self.initial_layers:
                lora_active = _checked_mul(
                    current, self.lora_rank, self.lora_elements_per_rank_per_layer
                )
                if lora_active > self.active_parameter_budget:
                    raise ValueError("LoRA readjustment stage exceeds active_parameter_budget")
                rows.append(
                    GrowthStage(
                        stage_index,
                        "lora_readjustment",
                        current,
                        current,
                        current,
                        (),
                        tuple(range(current)),
                        self.lora_rank,
                        lora_active,
                        _checked_mul(lora_active, self.optimizer_state_bytes_per_parameter),
                    )
                )
                stage_index += 1
        return tuple(rows)

    @property
    def inference_elements(self) -> int:
        return _checked_add(
            self.fixed_interface_elements,
            _checked_mul(self.final_layers, self.block_parameter_elements),
            self.head_parameter_elements,
        )

    @property
    def digest(self) -> str:
        return _digest(
            {
                "schema": "bcir.progressive_growth.v1",
                **asdict(self),
                "stages": [asdict(stage) for stage in self.stages()],
            }
        )


@dataclass(frozen=True)
class LoweredGrowthStage:
    spec: ProgressiveGrowthSpec
    stage: GrowthStage
    module: Module
    realization: RealizationResult
    streampack: StreamPack
    streampack_data: bytes
    artifact_sha256: str


def lower_growth_stage(
    spec: ProgressiveGrowthSpec,
    stage_index: int,
    *,
    sequence_length: int,
    batch: int = 1,
    target: TargetProfile | None = None,
) -> LoweredGrowthStage:
    """Lower one immutable train stage through verified claims and StreamPack."""
    if not isinstance(spec, ProgressiveGrowthSpec):
        raise ValueError("spec must be ProgressiveGrowthSpec")
    stages = spec.stages()
    _integer(stage_index, "stage_index", maximum=len(stages) - 1)
    _integer(sequence_length, "sequence_length", minimum=1, maximum=1 << 20)
    _integer(batch, "batch", minimum=1, maximum=1 << 16)
    stage = stages[stage_index]
    rows = _checked_mul(batch, sequence_length)
    base_elements = _checked_add(
        _checked_mul(stage.total_layers, spec.block_parameter_elements),
        spec.head_parameter_elements,
    )
    # Dense stages update the new blocks and head, so only the prior blocks are frozen.
    # LoRA adapters are additional trainable tensors: every base-model element remains
    # frozen and must not be subtracted from the frozen resource inventory.
    frozen = (
        base_elements
        if stage.kind == "lora_readjustment"
        else max(1, base_elements - stage.active_parameter_elements)
    )
    active = stage.active_parameter_elements
    module = Module(name=f"progressive-growth:{spec.digest[:16]}:{stage_index}")
    module.add_resource(Resource(1, Domain.RAM, 4, (rows,), name="token_ids"))
    module.add_resource(
        Resource(2, Domain.RAM, 4, (spec.fixed_interface_elements,), name="fixed_interface")
    )
    module.add_resource(Resource(3, Domain.RAM, 4, (frozen,), name="frozen_weights"))
    module.add_resource(Resource(4, Domain.RAM, 4, (active,), name="active_weights"))
    module.add_resource(
        Resource(5, Domain.RAM, 1, (max(1, stage.optimizer_state_bytes),), name="optimizer_state")
    )
    module.add_resource(
        Resource(
            6,
            Domain.RAM,
            4,
            (stage.total_layers + 1, rows, spec.interface_width),
            name="activations",
        )
    )
    vocabulary = spec.fixed_interface_elements // spec.interface_width
    module.add_resource(Resource(7, Domain.RAM, 4, (rows, vocabulary), name="loss_gradient"))

    phase_id = claim_id = 0

    def emit(
        opcode: Opcode,
        lane: Lane,
        stride: StrideClass,
        *,
        count: int,
        reads: tuple[int, ...],
        writes: tuple[int, ...],
        op: str,
        cost_class: str,
    ) -> None:
        nonlocal phase_id, claim_id
        phase_id += 1
        claim_id += 1
        claim = Claim(
            claim_id,
            opcode,
            lane,
            stride,
            count=max(1, count),
            rd=reads,
            wr=writes,
            hazard="barriered",
            bounds="assumed_safe",
            domain=Domain.RAM,
            op=op,
            cost_class=cost_class,
        )
        module.add_phase(Phase(phase_id, (phase_id - 1,) if phase_id > 1 else (), [claim]))

    emit(
        Opcode.LOAD,
        Lane.GGG,
        StrideClass.RANDOM,
        count=_checked_mul(rows, spec.interface_width),
        reads=(1, 2),
        writes=(6,),
        op="model.interface.fixed_lookup",
        cost_class="bandwidth",
    )
    if frozen > 1:
        emit(
            Opcode.T_MACC,
            Lane.T,
            StrideClass.TILE,
            count=_checked_mul(rows, frozen),
            reads=(3, 6),
            writes=(6,),
            op="model.growth.forward.frozen",
            cost_class="compute",
        )
    stage_op = (
        "model.growth.forward.new_blocks"
        if stage.kind == "dense_growth"
        else "model.growth.forward.lora_adapters"
    )
    emit(
        Opcode.T_MACC,
        Lane.T,
        StrideClass.TILE,
        count=_checked_mul(rows, active),
        reads=(4, 6),
        writes=(6,),
        op=stage_op,
        cost_class="compute",
    )
    emit(
        Opcode.T_MACC,
        Lane.T,
        StrideClass.TILE,
        count=_checked_mul(rows, active),
        reads=(1, 3, 4, 6),
        writes=(7,),
        op="model.growth.backward.active_only",
        cost_class="compute",
    )
    emit(
        Opcode.ADD,
        Lane.U,
        StrideClass.UNIT,
        count=active,
        reads=(4, 5, 7),
        writes=(4, 5),
        op="model.growth.optimizer.update",
        cost_class="compute",
    )
    emit(
        Opcode.BARRIER,
        Lane.H,
        StrideClass.SCALAR,
        count=1,
        reads=(4, 5),
        writes=(),
        op="model.growth.stage.freeze_boundary",
        cost_class="latency",
    )

    chosen_target = target or TargetProfile.for_host()
    realization = optimize(module, chosen_target, Theta.cool())
    pack = hydrate_pipelined(
        module, realization, plan=f"growth:{spec.digest[:16]}:{stage_index}", depth=2
    )
    diagnostics = verify_all(
        module, result=realization, pack=pack, h=chosen_target, theta=Theta.cool()
    )
    diagnostics += verify_smart_lowering(module, pack=pack)
    if diagnostics:
        message = "; ".join(f"{row.law}: {row.message}" for row in diagnostics)
        raise ValueError("progressive-growth lowering failed verification: " + message)
    data = encode(pack)
    artifact = _digest(
        {
            "spec": spec.digest,
            "stage": asdict(stage),
            "module": hash_module(module),
            "streampack": hashlib.sha256(data).hexdigest(),
        }
    )
    return LoweredGrowthStage(spec, stage, module, realization, pack, data, artifact)


__all__ = [
    "AlignedTokenChunk",
    "CausalSeriesCodecSpec",
    "CausalSeriesEncoding",
    "ContinuedBPEExpansionPlan",
    "CrossTokenizerProjection",
    "DecodedToken",
    "ExpansionRow",
    "FiniteScalarQuantizerSpec",
    "FrozenTokenCodeSpec",
    "GrowthStage",
    "LoweredGrowthStage",
    "ProgressiveGrowthSpec",
    "SequenceInterfaceEvidence",
    "TokenizerExpansionCost",
    "UnigramPiece",
    "align_decoded_tokens",
    "assess_tokenizer_expansion",
    "canonical_substring_candidate",
    "decode_causal_series",
    "encode_causal_series",
    "lower_growth_stage",
    "pareto_sequence_interfaces",
    "plan_continued_bpe_expansion",
    "project_aligned_log_probabilities",
    "segment_unigram",
    "thunder_candidate_score",
]
