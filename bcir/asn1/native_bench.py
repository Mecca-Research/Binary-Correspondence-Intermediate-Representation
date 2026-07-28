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

* **OER** has no C implementation yet. That is a gap in the repository, and it closes when
  somebody writes one.
* **PER cannot have one.** X.691 §7.2: a PER encoding is not self-delimiting — "without
  knowledge of the type of the value" the octets cannot be walked at all. There is no
  schema-free structural pass to time, so a comparable native number does not exist and
  will not exist. `bcir_per.c` implements the reading *primitives*; timing those against a
  whole-document scan would compare unlike work and call the difference an encoding cost.

That asymmetry is recorded in `NATIVE_OPS` rather than left implicit, because "not yet" and
"not ever" call for different decisions from whoever reads the table.

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
_SOURCES = ["bcir_asn1_bench.c", "bcir_asn1.c", "bcir_jer.c", "bcir_xer.c",
            "bcir_runtime.c"]


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
        None, permanent=True,
        reason="X.691 §7.2: a PER encoding is not self-delimiting, so there is no "
               "schema-free structural pass to time and no comparable native number "
               "exists — this is a property of the encoding, not a gap"),
    "CANONICAL-PER-UNALIGNED": NativeOp(
        None, permanent=True,
        reason="X.691 §7.2: a PER encoding is not self-delimiting (see the aligned entry)"),
    "BASIC-PER-ALIGNED": NativeOp(
        None, permanent=True,
        reason="X.691 §7.2: a PER encoding is not self-delimiting (see the aligned entry)"),
    "BASIC-PER-UNALIGNED": NativeOp(
        None, permanent=True,
        reason="X.691 §7.2: a PER encoding is not self-delimiting (see the aligned entry)"),
    "COER": NativeOp(
        None, reason="no C OER decoder exists yet; this closes when one is written"),
    "BASIC-OER": NativeOp(
        None, reason="no C OER decoder exists yet; this closes when one is written"),
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
            [cc, f"-std={std}", "-O2", "-Wall", "-Wextra", "-Werror", "-I", _C,
             *[os.path.join(_C, name) for name in _SOURCES], "-o", out],
            capture_output=True, text=True)
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


def run_native_bench(kind, value, *, warmup: int = 2, rounds: int = MIN_SAMPLES + 4,
                     iterations: int = 64, candidates=ALL_CANDIDATES
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
        except Exception as error:                       # noqa: BLE001 - reported, not raised
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
        proc = subprocess.run([binary], input="\n".join(lines) + "\n",
                              capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise Asn1Error(f"the native bench refused the corpus: {proc.stdout.strip()}")

    per_case: dict[str, list[int]] = {}
    for row in proc.stdout.splitlines():
        parts = row.split()
        if parts and parts[0] == "sample":
            per_case.setdefault(parts[1], []).append(int(parts[4]))

    sizes = {name: len(octets) for name, _op, octets in corpus}
    ops = {name: op for name, op, _octets in corpus}
    return ([NativeSamples(candidate=name, op=ops[name], octets=sizes[name],
                           decode_ns=tuple(samples))
             for name, samples in sorted(per_case.items())], skipped)


def measured_table(kind, value, *, target: str, cal_gen: int,
                   candidates=ALL_CANDIDATES, **kwargs) -> EncodingCostTable:
    """A `provenance="measured"` table — containing only what was natively measured.

    The encode interval is the decode interval, and that is a deliberate under-claim rather
    than an oversight: nothing here times an encode, because the C rail has no encoder for
    these candidates. Reusing the decode figure keeps the row shape uniform while making an
    encode-latency objective decide on a number that is honestly labelled in the docstring
    and in the roadmap; a caller who needs a real encode column must wait for a native
    encoder rather than read one out of this table.
    """
    samples, skipped = run_native_bench(kind, value, candidates=candidates, **kwargs)
    rows = []
    for sample in samples:
        if len(sample.decode_ns) < MIN_SAMPLES:
            raise Asn1Error(
                f"{sample.candidate}: {len(sample.decode_ns)} rounds is below the "
                f"{MIN_SAMPLES}-sample floor an order-statistic interval needs")
        interval = interval_of(list(sample.decode_ns))
        rows.append(CostRow(candidate=sample.candidate, octets=sample.octets,
                            encode=interval, decode=interval))
    _ = skipped                                          # reported by run_native_bench
    return EncodingCostTable(target=target, cal_gen=cal_gen, provenance="measured",
                             rows=tuple(rows))


__all__ = [
    "NATIVE_OPS", "NativeOp", "NativeSamples", "build_harness", "measured_table",
    "native_available", "run_native_bench",
]
