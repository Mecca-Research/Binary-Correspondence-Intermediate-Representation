"""LLVM's poison, imported as proved facts: `nsw` / `nuw` on an integer claim's lowering (G29).

The 2026-10-06 audit (item 13b, Missing) found no poison or `freeze` semantics in any rail.
LLVM's `add nsw` is a promise -- the signed result does not wrap, else it is poison -- that the
optimizer then spends: it widens, reassociates and folds on the strength of it. BCIR may
therefore emit such a flag only where it is a theorem, never a hope.

**The facts.** A resource may declare the inclusive range of its integer element values
(`model.graph.Resource.value_range`): the contract every element read keeps. For a
2-read/1-write integer claim at the element width (i32), `prove_no_wrap` evaluates the
operation over the two read ranges by exact interval arithmetic and proves:

* `nsw` when the exact result range fits the signed range of the width;
* `nuw` when, reading every operand bit pattern as unsigned -- a negative range is an
  interval near `2^width`, a range across zero two intervals -- the exact unsigned result of
  every pair of those intervals fits `[0, 2^width - 1]`.

A claim with an undeclared read range proves nothing, and nothing is emitted. A declared range
that does not fit the width, or is empty, is refused (`PoisonError`), never trusted.

**The lowering** (`lower.llvm`, `elem="i32"`) carries exactly the proved flags on every
compute instruction, and each read's declared range as `!range` metadata on its loads -- the
contract made visible to LLVM, where a value outside it is poison as the contract says. **R12**
(`verify.poison`) reads both back from the kernel's text and holds them to the proof: a flag
the ranges do not prove is a forged fact, a dropped one or a range that differs from the
declaration is a finding.

**LLVM judges** (`judge`): the kernel with its `!range` loads and *no* flags, through LLVM's
own range reasoning (`correlated-propagation`, then `instcombine`), must infer every flag BCIR
exports -- an independent derivation of each fact from the same contract, sharing no code
with the proof here. (`instcombine` alone misses `nsw` on a product its known-bits test cannot
bound; the lazy-value-info ranges `correlated-propagation` reads do bound it.)

Not built: the law rail carries no range attribute and the C emitters no flags (the oracle's
lowering only); `freeze`; facts on any width but i32; poison-generating `exact`, `inbounds`
proofs beyond the loop's own (`getelementptr inbounds` over `[0, n)` is the kernel contract).
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass

from ..model import Module, Opcode

#: The integer element width the facts are proved for.
WIDTH = 32
SIGNED = (-(1 << (WIDTH - 1)), (1 << (WIDTH - 1)) - 1)
UNSIGNED_MAX = (1 << WIDTH) - 1
FLAGS = ("nuw", "nsw")  # LLVM's printed order


class PoisonError(ValueError):
    """A declared value range that cannot be a contract for the element width."""


@dataclass(frozen=True)
class NoWrapProof:
    """The facts proved for one claim: the read ranges, the exact result range and the flags."""

    reads: tuple[tuple[int, int] | None, tuple[int, int] | None]
    result: tuple[int, int] | None
    flags: frozenset

    def __str__(self) -> str:
        return " ".join(f for f in FLAGS if f in self.flags) or "none"


def declared_range(module: Module, rid: int) -> tuple[int, int] | None:
    """A resource's declared value range, checked against the width; None when undeclared."""
    rng = module.resources[rid].value_range
    if rng is None:
        return None
    if (
        not isinstance(rng, tuple)
        or len(rng) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in rng)
    ):
        raise PoisonError(f"resource {rid} declares a value range that is not (lo, hi) integers")
    lo, hi = rng
    if lo > hi:
        raise PoisonError(f"resource {rid} declares an empty value range [{lo}, {hi}]")
    if lo < SIGNED[0] or hi > SIGNED[1]:
        raise PoisonError(
            f"resource {rid} declares [{lo}, {hi}], outside the i{WIDTH} range {list(SIGNED)}"
        )
    return lo, hi


def _exact(opcode, a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    if opcode == Opcode.ADD:
        return a[0] + b[0], a[1] + b[1]
    if opcode == Opcode.SUB:
        return a[0] - b[1], a[1] - b[0]
    corners = [x * y for x in a for y in b]
    return min(corners), max(corners)


def _unsigned(rng: tuple[int, int]) -> list[tuple[int, int]]:
    """The unsigned readings of a signed range's bit patterns, as one or two intervals."""
    lo, hi = rng
    span = 1 << WIDTH
    if lo >= 0:
        return [(lo, hi)]
    if hi < 0:
        return [(lo + span, hi + span)]
    return [(0, hi), (lo + span, span - 1)]


def prove_no_wrap(module: Module, claim) -> NoWrapProof:
    """The no-wrap facts the declared read ranges prove for `claim` (module docstring)."""
    reads = (declared_range(module, claim.rd[0]), declared_range(module, claim.rd[1]))
    if None in reads or claim.opcode not in (Opcode.ADD, Opcode.SUB, Opcode.MUL):
        return NoWrapProof(reads, None, frozenset())
    lo, hi = _exact(claim.opcode, reads[0], reads[1])
    flags = set()
    if SIGNED[0] <= lo and hi <= SIGNED[1]:
        flags.add("nsw")
    if all(
        0 <= ulo and uhi <= UNSIGNED_MAX
        for ua in _unsigned(reads[0])
        for ub in _unsigned(reads[1])
        for ulo, uhi in (_exact(claim.opcode, ua, ub),)
    ):
        flags.add("nuw")
    return NoWrapProof(reads, (lo, hi), frozenset(flags))


def range_metadata(rng: tuple[int, int]) -> str | None:
    """LLVM's `!range` operands for an inclusive range: half-open `[lo, hi + 1)` in i32,
    the upper end wrapping when `hi` is the largest value; None for the full set, which
    `!range` cannot spell (and which says nothing)."""
    lo, hi = rng
    if (lo, hi) == SIGNED:
        return None
    end = hi + 1
    if end > SIGNED[1]:
        end -= 1 << WIDTH
    return f"i32 {lo}, i32 {end}"


_COMPUTE = re.compile(r"=\s*(add|sub|mul)((?:\s+(?:nuw|nsw))*)\s+i32\s")


#: LLVM's range reasoning, in the order the judge runs it.
JUDGE_PASSES = "correlated-propagation,instcombine"


def judge(module: Module, result) -> tuple[str, frozenset | None, str]:
    """LLVM's own verdict on the facts (module docstring): the scalar i32 kernel with its
    `!range` loads and its flags stripped, through `opt -passes=JUDGE_PASSES`. Returns
    (verdict, the flags LLVM inferred, output): `ok`; `folded` when LLVM removed the compute
    instruction (a range that makes it an identity -- nothing is left to judge); `failed`; or
    `skip:<reason>` without a coherent LLVM toolchain -- a skip, never a pass."""
    from ..toolchain import resolve_llvm_tools
    from .llvm import emit_kernel_ll

    llvm = resolve_llvm_tools("opt", pipeline="AOT")
    if not llvm.ok:
        return f"skip:{llvm.message}", None, ""
    text = emit_kernel_ll(module, result, "bcir_judge", elem="i32", width_override=1)
    stripped = re.sub(r"\b(add|sub|mul)((?:\s+(?:nuw|nsw))+)(\s+i32\s)", r"\1\3", text)
    with tempfile.TemporaryDirectory(prefix="bcir-poison-") as work:
        path = os.path.join(work, "kernel.ll")
        with open(path, "w", newline="\n") as f:
            f.write(stripped)
        run = subprocess.run(
            [llvm.paths["opt"], "-S", f"-passes={JUDGE_PASSES}", path],
            capture_output=True,
            text=True,
        )
    if run.returncode != 0:
        return "failed", None, run.stdout + run.stderr
    inferred, seen = set(), False
    for line in run.stdout.splitlines():
        m = _COMPUTE.search(line)
        if m:
            seen = True
            inferred |= set(m.group(2).split())
    return ("ok" if seen else "folded"), frozenset(inferred), run.stdout


__all__ = [
    "FLAGS",
    "JUDGE_PASSES",
    "NoWrapProof",
    "PoisonError",
    "SIGNED",
    "UNSIGNED_MAX",
    "WIDTH",
    "declared_range",
    "judge",
    "prove_no_wrap",
    "range_metadata",
]
