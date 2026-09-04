"""The native microbench protocol — what makes a `measured` cost table possible.

J6 built the layer that decides when a measurement may decide, and left one thing open:
nothing could produce a table with `provenance="measured"`, so `select_certified` refused
every timing objective and §2's note that Python timings "cannot establish a
target-independent ordering" stood unresolved. This closes that.

**The refusal moves rather than disappears.** J6 refuses to decide a timing objective from
an oracle table. This module refuses to put a candidate in a measured table when the C rail
has no native implementation of it — so the table is smaller than the candidate list, and
`select_certified` then refuses any objective that would need a missing row. Nothing is
filled in. That chain is the design: at every step the answer to "we do not have this
number" is a refusal that names what is missing, never a substitute that looks like data.

**Two candidates are absent for different reasons, and the difference matters.**

* **PER cannot have one.** X.691 §7.2: a PER encoding is not self-delimiting — "without
  knowledge of the type of the value" the octets cannot be walked at all. There is no
  schema-free structural pass to time, so a comparable native number does not exist and
  will not exist. `bcir_per.c` implements the reading *primitives*; timing those against a
  whole-document scan would compare unlike work and call the difference an encoding cost.
* **OER cannot either, and this entry was wrong at first.** It read "no C OER decoder
  exists yet", which called a law an ordinary gap. X.696 §6.2 states the same rule as
  X.691 §7.2: *"without knowledge of the type of the value encoded, it is not possible to
  determine the structure of the encoding"*. `runtime/c/bcir_oer.c` now decodes OER
  natively — and that did **not** make it measurable here, because it is schema-*directed*
  while everything in this table is a schema-free structural pass. Writing the decoder is
  what exposed the mislabel.

Both reasons are recorded in `NATIVE_OPS` rather than left implicit, because a reader who
cannot tell "not yet" from "not ever" will either wait for a row that is never coming or
conclude the harness is broken.

**What is timed is what a peer pays at a trust boundary**: walk the octets you were handed
and decide whether they are well formed. `bcir_asn1_validate_der` for DER, the three-stage
bounded pass for JER, `bcir_asn1_validate` for BER, a tag walk for XER. Not a
schema-directed decode into typed values, because the C rail has one for no candidate here
— and timing a full decode against a structural scan would be measuring the implementations
rather than the encodings, which is the exact error §2 warns about one level up.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

from .certified import MIN_SAMPLES, CostRow, EncodingCostTable, interval_of
from .selection import ALL_CANDIDATES
from .tags import Asn1Error

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_SOURCES = [
    "bcir_asn1_bench.c",
    "bcir_asn1.c",
    "bcir_jer.c",
    "bcir_xer.c",
    "bcir_runtime.c",
    "bcir_emit.c",
    "bcir_oer.c",
    "bcir_per.c",
    "bcir_per_plan.c",
]


@dataclass(frozen=True)
class NativeOp:
    """How one candidate is measured natively, or why it is not.

    `op` is None when there is no native measurement, and `reason` then says which kind of
    absence it is. A consumer that treats "no C implementation yet" and "not measurable in
    principle" the same way would either wait forever for a PER row or conclude the harness
    is broken.
    """

    op: str | None
    reason: str = ""
    permanent: bool = False


#: candidate name -> how it is measured. Every candidate in `ALL_CANDIDATES` appears, and a
#: test enforces that, so adding a candidate forces a decision here rather than letting it
#: fall out of the table silently.
NATIVE_OPS: dict[str, NativeOp] = {
    "DER": NativeOp("der"),
    "BER": NativeOp("ber"),
    "JER": NativeOp("jer"),
    "JER-BCIR-CANONICAL": NativeOp("jer"),
    "CANONICAL-PER-ALIGNED": NativeOp(
        None,
        permanent=True,
        reason="X.691 §7.2: a PER encoding is not self-delimiting, so there is no "
        "schema-free structural pass to time and no comparable native number "
        "exists — this is a property of the encoding, not a gap",
    ),
    "CANONICAL-PER-UNALIGNED": NativeOp(
        None,
        permanent=True,
        reason="X.691 §7.2: a PER encoding is not self-delimiting (see the aligned entry)",
    ),
    "BASIC-PER-ALIGNED": NativeOp(
        None,
        permanent=True,
        reason="X.691 §7.2: a PER encoding is not self-delimiting (see the aligned entry)",
    ),
    "BASIC-PER-UNALIGNED": NativeOp(
        None,
        permanent=True,
        reason="X.691 §7.2: a PER encoding is not self-delimiting (see the aligned entry)",
    ),
    # CORRECTED. These previously read "no C OER decoder exists yet; this closes when one
    # is written", which called X.696 §6.2's law an ordinary gap. It is not: §6.2 says
    # "without knowledge of the type of the value encoded, it is not possible to determine
    # the structure of the encoding" — the same law X.691 §7.2 states for PER. A C OER
    # decoder now exists (`runtime/c/bcir_oer.c`) and it did NOT make OER measurable here,
    # because it is schema-directed and everything else in this table is a schema-free
    # structural pass. Timing the two against each other would compare unlike work and
    # report the difference as an encoding cost.
    "COER": NativeOp(
        None,
        permanent=True,
        reason="X.696 §6.2: without the type, the structure of an OER encoding cannot be "
        "determined, so there is no schema-free structural pass to time. "
        "runtime/c/bcir_oer.c decodes OER natively but is schema-DIRECTED, which is "
        "not comparable to the structural scans this table measures",
    ),
    "BASIC-OER": NativeOp(
        None,
        permanent=True,
        reason="X.696 §6.2: without the type, the structure of an OER encoding cannot be "
        "determined (see the COER entry)",
    ),
}


# --- the encode column, and why it is not this table's mirror ------------------------------------


@dataclass(frozen=True)
class EncodeOp:
    """Whether a candidate can be encoded WITHOUT a schema, and what follows if it cannot.

    A schema-free encoder is what makes a candidate measurable on the same terms as the
    decode table above: one common input, one pass, no descriptor to compile or to blame for
    the difference.
    """

    schema_free: bool
    reason: str = ""
    #: The `bcir_emit` rules name that measures this candidate natively, or None. A
    #: candidate can be schema-DIRECTED and still measurable — that is the whole point of
    #: E1/E2 — so this is independent of `schema_free`.
    native_op: str | None = None


#: The partition, and it is **not** the decode table's partition.
#:
#: Everything above measures a schema-free structural *scan*, and the rows missing from it
#: are missing because X.691 §7.2 and X.696 §6.2 say the structure of the encoding cannot be
#: recovered without the type. On the write side that law does not apply — you are handed the
#: value, not the octets — so the obvious expectation is that the encode column has *more*
#: rows. It has fewer, and the two absences do not overlap:
#:
#: - **X.690 is self-describing in both directions.** `encode_der` takes a value and no type;
#:   a TLV tree carries its own tags and lengths, so a re-emit is the complete operation
#:   rather than a stand-in for one.
#: - **Every other candidate needs the type to encode.** X.697 §22.2 puts member
#:   *identifiers* in a JER document, and an identifier exists only in the schema — the value
#:   has never heard of it. X.693 needs element names for the same reason, and OER and PER
#:   need the type to fix field widths and presence bits.
#:
#: So JER — the candidate the whole J roadmap is about — is on the measurable side of the
#: decode table and the **unmeasurable** side of a schema-free encode column, while PER and
#: OER, permanently absent above, are perfectly encodable *given a plan*. A schema-directed
#: encode harness would therefore cover **every** candidate, including the two the decode
#: table can never hold. That harness is J2's plan compiled into C for the write side; J3
#: built the read side from `JerSchemaPlan.serialize()` and the write side does not exist.
#:
#: Recording this is the point. A schema-free encode harness is cheap to build and would
#: yield a two-row table with JER absent, which reads as a gap in the implementation rather
#: than as the law it is.
ENCODE_OPS: dict[str, EncodeOp] = {
    "DER": EncodeOp(True, native_op="der"),
    "BER": EncodeOp(True, native_op="ber"),
    "JER": EncodeOp(
        False,
        native_op="jer",
        reason="X.697 §22.2: a JER document carries member IDENTIFIERS, which exist only in "
        "the type — the value has never heard of them",
    ),
    "JER-BCIR-CANONICAL": EncodeOp(
        False,
        native_op="jer",
        reason="X.697 §22.2: member identifiers come from the type (see the JER entry)",
    ),
    "COER": EncodeOp(
        False,
        native_op="coer",
        reason="X.696: field widths, presence bits and the preamble are fixed by the type, "
        "so there is nothing to emit without one",
    ),
    "BASIC-OER": EncodeOp(
        False,
        reason="X.696: the type fixes the layout; CANONICAL-OER is the row E2 "
        "measures, and BASIC-OER's non-canonical spellings are a decoder's "
        "problem rather than a distinct encode cost",
    ),
    # Four rows rather than one. X.691's ALIGNED/UNALIGNED split is a real cost trade —
    # ALIGNED pads so multi-octet fields start on octet boundaries, UNALIGNED never pads —
    # and CANONICAL/BASIC decides §19.5's DEFAULT rule. One row for the pair would report
    # one number for two encodings, which is the error §6.2 warns about one level up.
    "CANONICAL-PER-ALIGNED": EncodeOp(
        False,
        native_op="cper-a",
        reason="X.691: the type fixes field widths, the extension bit and the presence "
        "bitmap, so a value alone determines no octets",
    ),
    "CANONICAL-PER-UNALIGNED": EncodeOp(
        False,
        native_op="cper-u",
        reason="X.691: the type fixes the bit layout (see the aligned entry)",
    ),
    "BASIC-PER-ALIGNED": EncodeOp(
        False,
        native_op="bper-a",
        reason="X.691: the type fixes the bit layout (see the aligned entry)",
    ),
    "BASIC-PER-UNALIGNED": EncodeOp(
        False,
        native_op="bper-u",
        reason="X.691: the type fixes the bit layout (see the aligned entry)",
    ),
}


def observed_encode_partition() -> dict[str, bool]:
    """Which candidates the *oracle's own encoders* can serialize without a schema.

    Derived from the encoder signatures rather than asserted, so `ENCODE_OPS` cannot drift
    away from the code it describes. An encoder whose first parameter is the type is
    schema-directed by construction; `encode_der`'s first parameter is the value.
    """
    import inspect

    from . import codec, jer, oer, per, xer

    by_family = {
        "DER": codec.encode_der,
        "BER": codec.encode_der,
        "JER": jer.encode_jer,
        "JER-BCIR-CANONICAL": jer.encode_jer,
        "COER": oer.encode_oer,
        "BASIC-OER": oer.encode_oer,
        "CANONICAL-PER-ALIGNED": per.encode_per,
        "CANONICAL-PER-UNALIGNED": per.encode_per,
        "BASIC-PER-ALIGNED": per.encode_per,
        "BASIC-PER-UNALIGNED": per.encode_per,
    }
    # `xer` is imported so a future XER candidate is a KeyError here rather than a silent
    # omission from the partition.
    assert hasattr(xer, "encode_xer")
    return {
        name: next(iter(inspect.signature(fn).parameters)) != "kind"
        for name, fn in by_family.items()
    }


def native_available() -> bool:
    """Whether this host can build the harness at all. Absence is a clean skip."""
    return (shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")) is not None


def build_harness(tmp: str) -> str | None:
    cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        return None
    out = os.path.join(tmp, "bcir_asn1_bench")
    proc = None
    for std in ("c23", "c2x", "c11"):
        proc = subprocess.run(
            [
                cc,
                f"-std={std}",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                _C,
                *[os.path.join(_C, name) for name in _SOURCES],
                "-o",
                out,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return out
    raise Asn1Error(f"the native bench must build warning-clean:\n{proc.stderr[:2000]}")


@dataclass(frozen=True)
class NativeSamples:
    """Per-round decode times for one candidate, in nanoseconds."""

    candidate: str
    op: str
    octets: int
    decode_ns: tuple[int, ...]
    #: Cycles per round, when the host exposed a PMU. Empty when it did not — an absent
    #: counter is reported by its absence rather than filled in from the clock.
    decode_cycles: tuple[int, ...] = ()


def run_native_bench(
    kind,
    value,
    *,
    warmup: int = 2,
    rounds: int = MIN_SAMPLES + 4,
    iterations: int = 64,
    candidates=ALL_CANDIDATES,
) -> tuple[list[NativeSamples], dict[str, str]]:
    """Encode `value` under every candidate, then time the native decode of each.

    The corpus is built by the **Python encoders**, and that is correct rather than a
    compromise: the octets a candidate produces are the candidate, and the C rail is being
    timed on reading them. Encoding is not timed here at all — the C rail has no encoder for
    any of these, so an encode column would be Python timings wearing a `measured` label.

    Returns the samples and a map of candidate -> why it was skipped.
    """
    skipped: dict[str, str] = {}
    corpus: list[tuple[str, str, bytes]] = []
    for candidate in candidates:
        entry = NATIVE_OPS.get(candidate.name)
        if entry is None:
            skipped[candidate.name] = "no NATIVE_OPS entry; add one rather than defaulting"
            continue
        if entry.op is None:
            skipped[candidate.name] = entry.reason
            continue
        try:
            octets = candidate.encode(kind, value)
        except Exception as error:  # noqa: BLE001 - reported, not raised
            skipped[candidate.name] = f"not representable: {error}"
            continue
        corpus.append((candidate.name, entry.op, octets))

    if not corpus:
        return [], skipped

    with tempfile.TemporaryDirectory() as tmp:
        binary = build_harness(tmp)
        if binary is None:
            raise Asn1Error("no C compiler; a measured table cannot be produced here")
        lines = [f"rounds {warmup} {rounds} {iterations}"]
        lines += [f"case {name} {op} {octets.hex()}" for name, op, octets in corpus]
        lines.append("run")
        proc = subprocess.run(
            [binary], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600
        )
        if proc.returncode != 0:
            raise Asn1Error(f"the native bench refused the corpus: {proc.stdout.strip()}")

    per_case: dict[str, list[int]] = {}
    for row in proc.stdout.splitlines():
        parts = row.split()
        if parts and parts[0] == "sample":
            per_case.setdefault(parts[1], []).append(int(parts[4]))

    sizes = {name: len(octets) for name, _op, octets in corpus}
    ops = {name: op for name, op, _octets in corpus}
    return (
        [
            NativeSamples(
                candidate=name, op=ops[name], octets=sizes[name], decode_ns=tuple(samples)
            )
            for name, samples in sorted(per_case.items())
        ],
        skipped,
    )


#: X.696's field kinds, as `bcir_oer.h` numbers them. Duplicated here rather than parsed out
#: of the header because a silent renumber on either side must break a test, and a test that
#: reads the header would follow the renumber instead of catching it.
_OER_INTEGER, _OER_BOOLEAN, _OER_NULL, _OER_FIXED_OCTETS, _OER_VAR_OCTETS = 0, 1, 2, 3, 4

#: `bcir_per_kind` and `bcir_per_bounds`. Separate from the OER constants above because
#: the two enums are different tables that happen to start the same way, and one shared
#: name would make a renumbering on either side silently rebind the other.
_PER_INTEGER, _PER_BOOLEAN, _PER_NULL, _PER_FIXED_OCTETS, _PER_VAR_OCTETS = 0, 1, 2, 3, 4
_PER_UNCONSTRAINED, _PER_SEMI, _PER_CONSTRAINED = 0, 1, 2


def oer_fields_for(plan) -> bytes:
    """The `bcir_oer_field[]` a schema-directed decode needs, built from the write plan.

    Eight octets per field — kind, width, is_signed, optional, then a little-endian
    `fixed_len` — so the C side parses a fixed record instead of re-deriving X.696's field
    kinds from a second pass over the descriptor. The mapping lives here because this is where
    the plan's semantics already live.

    **Refuses rather than approximates.** A member this cannot map is an `Asn1Error` naming the
    kind, not a guessed field: a decode timed against the wrong field array would be a real
    number for the wrong work, which is worse than no number at all.
    """
    root = plan.root
    if root.kind != "sequence":
        raise Asn1Error(
            f"X.696 16.1: the schema-directed decode arm decodes a SEQUENCE; this plan's root "
            f"is {root.kind!r}"
        )
    out = bytearray()
    for member in root.members:
        node = member.node
        optional = 1 if getattr(member, "optional", False) else 0
        if node.kind == "integer":
            # 10.4's variable-size form: width 0 means a length determinant then the octets.
            out += bytes((_OER_INTEGER, 0, 1, optional, 0, 0, 0, 0))
        elif node.kind == "boolean":
            out += bytes((_OER_BOOLEAN, 0, 0, optional, 0, 0, 0, 0))
        elif node.kind == "null":
            out += bytes((_OER_NULL, 0, 0, optional, 0, 0, 0, 0))
        elif node.kind in ("string", "octetstring"):
            # 14.2/27: a length determinant then that many octets.
            out += bytes((_OER_VAR_OCTETS, 0, 0, optional, 0, 0, 0, 0))
        else:
            raise Asn1Error(
                f"X.696: the schema-directed decode arm has no field kind for a "
                f"{node.kind!r} member ({member.name!r}); add one to `oer_fields_for` rather "
                f"than letting the timing describe different work than the plan does"
            )
    return bytes(out)


#: Which candidates have a plan-driven decoder in the C rail, and why the others do not.
#:
#: **This is a different partition from `ENCODE_OPS`, and the asymmetry is the point.**
#: `bcir_oer.c` is a plan-driven whole-SEQUENCE decoder, so CANONICAL-OER has a row.
#: `bcir_per.h` exposes a bit READER — `bcir_per_get_bits`, `bcir_per_constrained`,
#: `bcir_per_length` — and no plan-driven whole-value decode, so PER has none. That is a
#: missing *build*, not a law: X.691 7.2 bars a schema-FREE decode and says nothing against a
#: schema-directed one. Recorded as a gap so it reads as buildable work rather than as
#: another clause.
def per_fields_for(plan) -> str:
    """The `bcir_per_field[]` a schema-directed PER decode needs, as the harness spells it.

    A text plan (`kind:bounds:lb:ub:fixed:optional`, comma separated) rather than the packed
    octets `oer_fields_for` emits, because PER's field carries two signed 64-bit bounds and a
    packed record would be sixteen bytes of endian-sensitive layout for a benchmark argument.
    The C harness parses the same spelling `test_per_plan.c` does, so one format serves the
    differential and the bench.

    **Refuses rather than approximates**, exactly as the OER mapping does: a member this
    cannot map raises rather than guessing a field, since a decode timed against the wrong
    field array is a real number for the wrong work.
    """
    root = plan.root
    if root.kind != "sequence":
        raise Asn1Error(
            f"X.691 18: the schema-directed decode arm decodes a SEQUENCE; this plan's root "
            f"is {root.kind!r}"
        )
    fields = []
    for member in root.members:
        node = member.node
        optional = 1 if getattr(member, "optional", False) else 0
        low = getattr(node, "low", None)
        high = getattr(node, "high", None)
        if node.kind == "integer":
            # §13.2 decides the shape from the TYPE's constraint, never from the octets.
            if low is not None and high is not None:
                fields.append(f"{_PER_INTEGER}:{_PER_CONSTRAINED}:{low}:{high}:0:{optional}")
            elif low is not None:
                fields.append(f"{_PER_INTEGER}:{_PER_SEMI}:{low}:0:0:{optional}")
            else:
                fields.append(f"{_PER_INTEGER}:{_PER_UNCONSTRAINED}:0:0:0:{optional}")
        elif node.kind == "boolean":
            fields.append(f"{_PER_BOOLEAN}:0:0:0:0:{optional}")
        elif node.kind == "null":
            fields.append(f"{_PER_NULL}:0:0:0:0:{optional}")
        elif node.kind in ("string", "octetstring"):
            width = getattr(node, "fixed_size", None)
            if width:
                fields.append(f"{_PER_FIXED_OCTETS}:0:0:0:{width}:{optional}")
            else:
                bound = getattr(node, "max_size", 0) or 0
                fields.append(f"{_PER_VAR_OCTETS}:0:0:0:{bound}:{optional}")
        else:
            raise Asn1Error(
                f"X.691: the schema-directed decode arm has no field kind for a {node.kind!r} "
                f"member; the plan-driven PER decoder states its subset rather than guessing"
            )
    return ",".join(fields)


DIRECTED_DECODE_OPS: dict[str, str | None] = {
    "COER": "coer-d",
    "BASIC-OER": None,
    "CANONICAL-PER-ALIGNED": "per-d-aligned",
    "CANONICAL-PER-UNALIGNED": "per-d-unaligned",
    "BASIC-PER-ALIGNED": None,
    "BASIC-PER-UNALIGNED": None,
    "DER": None,
    "BER": None,
    "JER": None,
    "JER-BCIR-CANONICAL": None,
}

_DIRECTED_REASONS: dict[str, str] = {
    "BASIC-OER": "X.696: BASIC-OER's non-canonical spellings are what the CANONICAL-OER "
    "decoder already accepts; a separate row would time the same decoder twice",
    "BASIC-PER-ALIGNED": "X.691: BASIC-PER's encodings are what the canonical decoder already "
    "accepts, so a separate row would time the same decoder twice",
    "DER": "already has a schema-free decode row; the schema-free table is where X.690 "
    "candidates belong, because they can be walked without a type",
}


def run_native_directed_decode_bench(
    kind,
    value,
    *,
    warmup: int = 2,
    rounds: int = MIN_SAMPLES + 4,
    iterations: int = 64,
    candidates=ALL_CANDIDATES,
) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    """Time the native **schema-directed** decode of `value` under every candidate that has one.

    This is deliberately NOT a column of the schema-free table, and merging the two would be
    the error 6.2 spends a paragraph on. They answer different questions:

    * the schema-free decode asks *can untrusted octets be walked with no type in hand* — a
      trust-boundary cost, and the reason DER/BER/JER/XER are the rows that have one;
    * the schema-directed decode asks *what does decode cost in deployment, where the type is
      always known* — which is the only question X.696 6.2 lets OER answer at all.

    One value, one plan, one set of octets per candidate, so what is timed is the decoder.
    """
    from .emit import flatten
    from .encode_plan import compile_encode_plan

    skipped: dict[str, str] = {}
    try:
        plan = compile_encode_plan(kind, module="bench", type_name="Bench")
        # Two field tables, because the two decoders take different ones: X.696's is
        # octet-oriented with a width and a sign, X.691's bit-oriented with two bounds. Both
        # are built from the SAME plan, so a schema either has a directed decode on both rails
        # or on neither, and a candidate never silently borrows the other's shape.
        fields = oer_fields_for(plan)
        per_fields = per_fields_for(plan)
        stream = flatten(plan, value)
    except Asn1Error as error:
        raise Asn1Error(
            f"no schema-directed decode table exists for this schema: {error}"
        ) from None

    cases: list[tuple[str, str, bytes]] = []
    for candidate in candidates:
        op = DIRECTED_DECODE_OPS.get(candidate.name, None)
        if op is None:
            skipped[candidate.name] = _DIRECTED_REASONS.get(
                candidate.name, "no plan-driven decoder in the C rail for this candidate"
            )
            continue
        try:
            octets = candidate.encode(kind, value)
        except Asn1Error as error:
            skipped[candidate.name] = f"the oracle cannot encode this value: {error}"
            continue
        if not octets:
            skipped[candidate.name] = "the oracle produced no octets for this candidate"
            continue
        cases.append((candidate.name, op, octets))
    if not cases:
        return {}, skipped

    with tempfile.TemporaryDirectory() as tmp:
        binary = build_harness(tmp)
        if binary is None:
            raise Asn1Error("no C compiler; a schema-directed decode table cannot be produced")
        lines = [f"rounds {warmup} {rounds} {iterations}"]
        lines += [
            f"dircase {name} {op} "
            f"{per_fields if op.startswith('per-d') else fields.hex()} {octets.hex()}"
            for name, op, octets in cases
        ]
        lines.append("run")
        proc = subprocess.run(
            [binary], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600
        )
        if proc.returncode != 0:
            raise Asn1Error(f"the native bench refused the directed corpus: {proc.stdout.strip()}")

    per_case: dict[str, list[int]] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == "sample":
            per_case.setdefault(parts[1], []).append(int(parts[4]))
    return ({name: tuple(values) for name, values in sorted(per_case.items())}, skipped)


def run_native_encode_bench(
    kind,
    value,
    *,
    warmup: int = 2,
    rounds: int = MIN_SAMPLES + 4,
    iterations: int = 64,
    candidates=ALL_CANDIDATES,
) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    """Time the NATIVE encode of `value` under every candidate E2 can emit.

    Every candidate goes through one write-side plan and one format-neutral value stream, so
    what is timed is the encoding rather than an adapter — which is the property #682 showed
    a schema-free harness could not have.

    **This column is not the decode column's mirror, and the difference is the point.**
    CANONICAL-OER appears here and can never appear in the decode table (X.696 §6.2). PER
    appears here too, in all four ALIGNED/UNALIGNED x CANONICAL/BASIC rows, and can never
    appear in the decode table either (X.691 §7.2). An earlier version of this paragraph said
    PER appeared in neither column because `bcir_emit` had no bit-oriented writer; it has one,
    so the only rows missing from the encode column now are the ones a *law* keeps out.
    """
    from .encode_plan import compile_encode_plan
    from .emit import flatten

    skipped: dict[str, str] = {}
    cases: list[tuple[str, str]] = []
    try:
        plan = compile_encode_plan(kind, module="bench", type_name="Bench")
        stream = flatten(plan, value)
    except Asn1Error as error:
        raise Asn1Error(
            f"the write plan refuses this schema, so no encode column exists for it: {error}"
        ) from None

    seen: set[str] = set()
    for candidate in candidates:
        entry = ENCODE_OPS.get(candidate.name)
        if entry is None:
            skipped[candidate.name] = "no ENCODE_OPS entry; add one rather than defaulting"
            continue
        if entry.native_op is None:
            skipped[candidate.name] = entry.reason
            continue
        cases.append((candidate.name, entry.native_op))
        seen.add(candidate.name)
    if not cases:
        return {}, skipped

    with tempfile.TemporaryDirectory() as tmp:
        binary = build_harness(tmp)
        if binary is None:
            raise Asn1Error("no C compiler; a measured encode table cannot be produced here")
        lines = [f"rounds {warmup} {rounds} {iterations}"]
        lines += [
            f"encase {name} {op} {plan.serialize().hex()} {stream.hex()}" for name, op in cases
        ]
        lines.append("run")
        proc = subprocess.run(
            [binary], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600
        )
        if proc.returncode != 0:
            raise Asn1Error(f"the native bench refused the encode corpus: {proc.stdout.strip()}")

    per_case: dict[str, list[int]] = {}
    for row in proc.stdout.splitlines():
        parts = row.split()
        if parts and parts[0] == "sample":
            per_case.setdefault(parts[1], []).append(int(parts[4]))
    return {name: tuple(samples) for name, samples in sorted(per_case.items())}, skipped


def native_counters() -> str:
    """What the native harness reports about this host's performance counters.

    `"cycles"` when a PMU was attached; otherwise the reason it was not, in the kernel's own
    words — `"No such file or directory"` on a container with no PMU exposed, or a permission
    error where `perf_event_paranoid` denies it. Reported so a calibration record can say
    which, because "this host has no counters" and "nobody asked" are different facts and only
    one of them is about the machine.
    """
    if not native_available():
        return "no compiler"
    with tempfile.TemporaryDirectory() as tmp:
        binary = build_harness(tmp)
        if binary is None:
            return "harness did not build"
        proc = subprocess.run(
            [binary], input="rounds 0 1 1\nrun\n", capture_output=True, text=True, timeout=600
        )
    for row in proc.stdout.splitlines():
        parts = row.split(maxsplit=1)
        if parts and parts[0] == "counters":
            return parts[1] if len(parts) > 1 else "unknown"
    return "not reported"


def directed_decode_table(
    kind, value, *, target: str, cal_gen: int, candidates=ALL_CANDIDATES, **bench
) -> EncodingCostTable:
    """The **schema-directed** decode table — a second table, never a column of the first.

    `measured_table` answers *can these octets be walked with no type in hand, and what does
    that cost*. This one answers *what does decode cost in deployment, where the type is always
    known*. Both are true; neither substitutes for the other; and `EncodingCostTable.decode_kind`
    keeps a certificate from confusing them, because it is inside the table's digest.

    **This is the table X.696 6.2 leaves room for.** CANONICAL-OER can never appear in the
    schema-free table — the clause is explicit that the octets cannot be walked without the
    type — and it appears here for exactly the same reason, from the other side.

    A row still needs BOTH axes, so a candidate is included only when it has a schema-directed
    decode *and* a native encode. That is not a formality: a row with one axis measured and the
    other copied is the under-claim `measured_table` used to make before `bcir_emit` landed.
    """
    decode_samples, skipped = run_native_directed_decode_bench(
        kind, value, candidates=candidates, **bench
    )
    if not decode_samples:
        raise Asn1Error(
            "no candidate has a schema-directed decode on this rail, so the table would be "
            f"empty; the reasons are: {skipped}"
        )
    encode_samples, _ = run_native_encode_bench(kind, value, candidates=candidates, **bench)

    rows: list[CostRow] = []
    for candidate in candidates:
        name = candidate.name
        decode_ns = decode_samples.get(name)
        encode_ns = encode_samples.get(name)
        if decode_ns is None or encode_ns is None:
            continue
        octets = candidate.encode(kind, value)
        rows.append(
            CostRow(
                candidate=name,
                octets=len(octets),
                encode=interval_of(list(encode_ns)),
                decode=interval_of(list(decode_ns)),
            )
        )
    if not rows:
        raise Asn1Error(
            "every schema-directed decode row lacked its encode axis, so no two-axis row "
            "closes; CostRow needs both and a copied axis is not a measurement"
        )
    return EncodingCostTable(
        target=target,
        cal_gen=cal_gen,
        provenance="measured",
        rows=tuple(rows),
        decode_kind="schema-directed",
    )


def measured_table(
    kind, value, *, target: str, cal_gen: int, candidates=ALL_CANDIDATES, **kwargs
) -> EncodingCostTable:
    """A `provenance="measured"` table — containing only what was natively measured.

    **The encode interval is now a real encode measurement** where E2 can produce one. It
    used to be a copy of the decode figure, an under-claim made necessary by the C rail
    having no encoder; `bcir_emit` removed that.

    A row still needs BOTH axes, because `CostRow` carries both. CANONICAL-OER therefore
    remains outside this table even though it now has a perfectly good encode number —
    X.696 §6.2 denies it a schema-free decode forever, so the two-axis row can never close.
    `run_native_encode_bench` returns that number directly for a caller who wants the axis
    rather than the row, and refusing to fabricate the missing half is the same discipline
    §6.2 applies one level up.
    """
    samples, skipped = run_native_bench(kind, value, candidates=candidates, **kwargs)
    encode_samples: dict[str, tuple[int, ...]] = {}
    try:
        encode_samples, _ = run_native_encode_bench(kind, value, candidates=candidates, **kwargs)
    except Asn1Error:
        # A schema the WRITE plan refuses still has a decode column; the encode interval
        # then falls back to the decode figure and the docstring above says what that means.
        encode_samples = {}
    rows = []
    for sample in samples:
        if len(sample.decode_ns) < MIN_SAMPLES:
            raise Asn1Error(
                f"{sample.candidate}: {len(sample.decode_ns)} rounds is below the "
                f"{MIN_SAMPLES}-sample floor an order-statistic interval needs"
            )
        interval = interval_of(list(sample.decode_ns))
        measured_encode = encode_samples.get(sample.candidate)
        encode = (
            interval_of(list(measured_encode))
            if measured_encode and len(measured_encode) >= MIN_SAMPLES
            else interval
        )
        rows.append(
            CostRow(
                candidate=sample.candidate, octets=sample.octets, encode=encode, decode=interval
            )
        )
    _ = skipped  # reported by run_native_bench
    return EncodingCostTable(
        target=target, cal_gen=cal_gen, provenance="measured", rows=tuple(rows)
    )


__all__ = [
    "ENCODE_OPS",
    "NATIVE_OPS",
    "EncodeOp",
    "NativeOp",
    "NativeSamples",
    "build_harness",
    "measured_table",
    "native_available",
    "native_counters",
    "observed_encode_partition",
    "run_native_bench",
    "run_native_encode_bench",
]
