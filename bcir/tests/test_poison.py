"""G29: LLVM's poison imported as proved facts -- no-wrap flags under R12, judged by LLVM.

The parent had no poison semantics in any rail (the audit's item 13b): an i32 kernel carried
no `nsw`/`nuw`, and R12 read none, so a kernel claiming one passed whatever it promised.
These witnesses hold the proof to wrapped arithmetic at every corner and sampled point, hold
R12 to every forgery (a flag the ranges do not prove, a dropped one, a range widened or
removed, a range on a float kernel), pin hand-derived proofs, and let LLVM judge each
exported fact from the same declared ranges where an LLVM is installed.
"""

from __future__ import annotations

from bcir.asn1.program import jer_to_module, module_to_jer
from bcir.kbcir.provenance import hash_module
from bcir.lower.llvm import emit_kernel_ll
from bcir.lower.poison import PoisonError, prove_no_wrap, range_metadata
from bcir.model import Opcode
from bcir.tests.poison_fixtures import (
    CASES,
    _plan,
    case,
    forged_accepted,
    forged_kernels,
    judged,
    module_of,
    unsound,
)
from bcir.verify import verify_lowering

BIG = (1 << 31) - 1


def _flags(m):
    return set(prove_no_wrap(m, m.phases[0].claims[0]).flags)


def test_hand_derived_proofs():
    assert _flags(module_of(Opcode.ADD, (-1000, 999), (-1000, 999))) == {"nsw"}
    assert _flags(module_of(Opcode.ADD, (0, 1000), (0, 1000))) == {"nsw", "nuw"}
    # one past the signed edge: the unsigned sum still fits, the signed one does not
    assert _flags(module_of(Opcode.ADD, (0, BIG), (0, 1))) == {"nuw"}
    # a difference that can go negative wraps unsigned, never signed here
    assert _flags(module_of(Opcode.SUB, (0, 10), (0, 20))) == {"nsw"}
    # read unsigned, a negative range sits just below 2^32: a small non-negative addend never
    # carries out of it, so the sum is nuw although an operand is negative
    assert _flags(module_of(Opcode.ADD, (0, 100), (-900, -800))) == {"nsw", "nuw"}
    # a range across zero is two unsigned intervals, and the upper one does carry out
    assert _flags(module_of(Opcode.ADD, (0, 1000), (-1, 1))) == {"nsw"}
    assert _flags(module_of(Opcode.SUB, (20, 30), (0, 20))) == {"nsw", "nuw"}
    # 46340^2 fits i32, 46341^2 does not; unsigned room remains for both
    assert _flags(module_of(Opcode.MUL, (0, 46340), (0, 46340))) == {"nsw", "nuw"}
    assert _flags(module_of(Opcode.MUL, (0, 46341), (0, 46341))) == {"nuw"}
    assert _flags(module_of(Opcode.MUL, (-3, 3), (0, 65536))) == {"nsw"}
    # an undeclared read proves nothing, and nothing is emitted
    assert _flags(module_of(Opcode.ADD, None, (0, 1))) == set()
    assert range_metadata((0, BIG)) == f"i32 0, i32 {-(1 << 31)}"  # the end wraps
    assert range_metadata((-(1 << 31), BIG)) is None  # the full set says nothing
    for bad in ((5, 4), (0, 1 << 31), (-(1 << 31) - 1, 0), (0.0, 1.0), (True, 2)):
        m = module_of(Opcode.ADD, bad, (0, 1))
        try:
            prove_no_wrap(m, m.phases[0].claims[0])
        except PoisonError:
            continue
        raise AssertionError(f"the declared range {bad!r} was trusted")


def test_every_proved_flag_survives_wrapped_arithmetic():
    """Over 120 generated claims every proved flag holds at every corner and sampled point
    of the read ranges, evaluated in exact and wrapped i32 arithmetic."""
    flagged = 0
    for seed in range(CASES):
        assert unsound(seed) == [], seed
        flagged += bool(_flags(case(seed)))
    assert flagged >= 40, flagged


def test_the_kernel_carries_exactly_the_proof_and_r12_reads_it_back():
    """Every generated kernel, at the selected width and scalar, verifies clean under R12;
    it is the parent's kernel exactly where no read range is declared."""
    for seed in range(CASES):
        m = case(seed)
        plan = _plan(m)
        for width in (None, 1):
            text = emit_kernel_ll(m, plan, elem="i32", width_override=width)
            assert verify_lowering(m, plan, text, elem="i32", width_override=width) == [], seed
    plain = module_of(Opcode.ADD, None, None)
    text = emit_kernel_ll(plain, _plan(plain), elem="i32")
    assert "nsw" not in text.split("%inext")[0] and "!range" not in text


def test_r12_refuses_every_forgery():
    """Each forged flag, dropped flag, widened or removed range and float-kernel range is
    refused, with the finding that names it."""
    accepted, total = forged_accepted()
    assert accepted == 0 and total >= 100, (accepted, total)
    names = {
        "forged nuw",
        "forged nsw",
        "dropped nuw",
        "dropped nsw",
        "widened range",
        "dropped range",
        "range on a float kernel",
    }
    seen = {what for _m, _p, _t, what in forged_kernels()}
    assert seen == names, seen
    m, plan, text, _what = next(k for k in forged_kernels() if k[3] == "forged nsw")
    messages = [d.message for d in verify_lowering(m, plan, text, elem="i32")]
    assert any(msg.startswith("forged no-wrap fact") for msg in messages), messages


def test_the_range_travels_with_the_program_and_moves_no_digest():
    """The declared range is carried by the ASN.1 program projection (absent and declared
    kept apart) and is outside the R13 digest, like a claim's timing and stream."""
    m = module_of(Opcode.ADD, (-5, 7), None)
    back = jer_to_module(module_to_jer(m))
    assert back.resources[10].value_range == (-5, 7) and back.resources[11].value_range is None
    assert hash_module(m) == hash_module(module_of(Opcode.ADD, None, None))


def test_llvm_judges_every_exported_fact():
    """With its loads' ranges and no flags, LLVM's own range reasoning re-derives every flag
    BCIR exports, and derives none BCIR does not: the proof is LLVM's, fact for fact. A skip
    without an LLVM toolchain, never a pass."""
    verdict = judged()
    if verdict is None:
        return
    missed, unjudged, folded, seen = verdict
    assert (missed, unjudged) == (0, 0), verdict
    assert seen >= 100 and folded <= 5, verdict
