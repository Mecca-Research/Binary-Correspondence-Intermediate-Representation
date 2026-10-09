"""G29 fixtures, shared by `test_poison` and the GEM+ harness so both grade the same way:
generated integer claims over declared value ranges, a soundness check that evaluates every
corner and sampled point in wrapped i32 arithmetic, LLVM's own verdict on each fact, and the
forged kernels R12 must refuse."""

from __future__ import annotations

import random
import re

from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta
from bcir.lower.llvm import emit_kernel_ll
from bcir.lower.poison import judge, prove_no_wrap
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.verify import verify_lowering

CASES = 120


def module_of(opcode, ra, rb, wr=12) -> Module:
    m = Module(name="poison")
    m.add_resource(Resource(rid=10, domain=Domain.RAM, shape=(256,), value_range=ra))
    m.add_resource(Resource(rid=11, domain=Domain.RAM, shape=(256,), value_range=rb))
    m.add_resource(Resource(rid=12, domain=Domain.RAM, shape=(256,)))
    claim = Claim(
        id=1,
        opcode=opcode,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=256,
        rd=(10, 11),
        wr=(wr,),
        op="vector.int",
    )
    m.add_phase(Phase(phase_id=0, claims=[claim]))
    return m


def _range(r: random.Random):
    kind = r.random()
    if kind < 0.08:
        return None
    if kind < 0.35:
        lo = r.randint(-2000, 0)
        return lo, lo + r.randint(1, 4000)
    if kind < 0.6:
        hi = r.choice((1000, 46340, 46341, 65535, 1 << 30, (1 << 31) - 2, (1 << 31) - 1))
        lo = r.randint(0, hi - 1) if r.random() < 0.3 else 0
        return lo, hi
    if kind < 0.8:
        lo = r.randint(-(1 << 31), -(1 << 20))
        return lo, lo + r.randint(1, 1 << 20)
    lo = r.randint(-100, 100)
    return lo, r.randint(lo + 1, (1 << 31) - 1)


def case(seed: int) -> Module:
    """One generated integer claim: add, sub or mul over two read ranges (some undeclared),
    each at least two values wide so no operand is a constant LLVM would fold away."""
    r = random.Random(seed)
    return module_of(r.choice((Opcode.ADD, Opcode.SUB, Opcode.MUL)), _range(r), _range(r))


def _wrap(value: int) -> int:
    return (value + (1 << 31)) % (1 << 32) - (1 << 31)


def unsound(seed: int) -> list[str]:
    """The proved flags a point of the read ranges breaks -- every corner and 64 sampled
    points evaluated in exact and in wrapped i32 arithmetic, sharing no code with the proof."""
    m = case(seed)
    claim = m.phases[0].claims[0]
    proof = prove_no_wrap(m, claim)
    if not proof.flags:
        return []
    (alo, ahi), (blo, bhi) = proof.reads
    r = random.Random(seed + 7919)
    points = [(a, b) for a in (alo, ahi) for b in (blo, bhi)]
    points += [(r.randint(alo, ahi), r.randint(blo, bhi)) for _ in range(64)]
    broken = set()
    for a, b in points:
        exact = {Opcode.ADD: a + b, Opcode.SUB: a - b, Opcode.MUL: a * b}[claim.opcode]
        if "nsw" in proof.flags and _wrap(exact) != exact:
            broken.add("nsw")
        if "nuw" in proof.flags:  # the operands' bit patterns read unsigned
            ua, ub = a % (1 << 32), b % (1 << 32)
            uexact = {Opcode.ADD: ua + ub, Opcode.SUB: ua - ub, Opcode.MUL: ua * ub}[claim.opcode]
            if not 0 <= uexact <= (1 << 32) - 1:
                broken.add("nuw")
    return sorted(broken)


def _plan(m: Module):
    return optimize(m, TARGETS["x86_avx512"], Theta.cool())


def judged(count: int = CASES):
    """LLVM's verdict over the generated cases: (missed, unjudged, folded, judged), where
    `missed` counts the flags LLVM derives that BCIR does not export and `unjudged` the flags
    BCIR exports that LLVM does not derive. None without a coherent LLVM."""
    missed = unjudged = folded = seen = 0
    for seed in range(count):
        m = case(seed)
        claim = m.phases[0].claims[0]
        proof = prove_no_wrap(m, claim)
        verdict, inferred, out = judge(m, _plan(m))
        if verdict.startswith("skip:"):
            return None
        if verdict == "failed":
            raise AssertionError(f"LLVM refused the judged kernel of case {seed}:\n{out}")
        if verdict == "folded":
            folded += 1
            continue
        seen += 1
        missed += len(inferred - proof.flags)
        unjudged += len(proof.flags - inferred)
    return missed, unjudged, folded, seen


def forged_kernels(count: int = 40):
    """(module, plan, text, what) for every forgery of the first `count` cases' kernels: each
    flag the ranges do not prove added, each proved flag dropped, each load range widened or
    removed, and a range attached to a float kernel."""
    out = []
    for seed in range(count):
        m = case(seed)
        claim = m.phases[0].claims[0]
        plan = _plan(m)
        text = emit_kernel_ll(m, plan, elem="i32")
        proof = prove_no_wrap(m, claim)
        op = {Opcode.ADD: "add", Opcode.SUB: "sub", Opcode.MUL: "mul"}[claim.opcode]
        for flag in ("nuw", "nsw"):
            if flag not in proof.flags:
                forged = re.sub(
                    rf"= {op}((?: nuw| nsw)*) (<\d+ x i32>|i32) ", rf"= {op}\1 {flag} \2 ", text
                )
                out.append((m, plan, forged, f"forged {flag}"))
            else:
                dropped = re.sub(rf"(= {op}(?: nuw| nsw)*?) {flag}", r"\1", text)
                out.append((m, plan, dropped, f"dropped {flag}"))
        ranges = re.findall(r"^(!\d+) = !\{i32 (-?\d+), i32 (-?\d+)\}$", text, re.M)
        for node, lo, end in ranges:
            wider = text.replace(
                f"{node} = !{{i32 {lo}, i32 {end}}}", f"{node} = !{{i32 {int(lo) - 1}, i32 {end}}}"
            )
            out.append((m, plan, wider, "widened range"))
        if ranges:
            out.append((m, plan, re.sub(r", !range !\d+", "", text), "dropped range"))
        if seed % 4 == 0:
            ftext = emit_kernel_ll(m, plan)
            node = len(re.findall(r"^!\d+ = ", ftext, re.M))
            fforged = ftext.replace(", !tbaa", f", !range !{node}, !tbaa", 1) + (
                f"!{node} = !{{i32 0, i32 10}}\n"
            )
            out.append((m, plan, fforged, "range on a float kernel"))
    return out


def forged_accepted(count: int = 40) -> tuple[int, int]:
    """(forgeries R12 accepts, forgeries in all)."""
    kernels = forged_kernels(count)
    accepted = 0
    for m, plan, text, what in kernels:
        elem = "f32" if what == "range on a float kernel" else "i32"
        accepted += not verify_lowering(m, plan, text, elem=elem)
    return accepted, len(kernels)


def measure() -> dict[str, float]:
    """The G29 harness rows; the LLVM-judged ones only where LLVM is."""
    accepted, _total = forged_accepted()
    out = {
        "poison.forged.accepted": float(accepted),
        "poison.facts.unsound": float(sum(bool(unsound(s)) for s in range(CASES))),
    }
    verdict = judged()
    if verdict is not None:
        missed, unjudged, _folded, _seen = verdict
        out["poison.facts.missed"] = float(missed)
        out["poison.facts.unjudged"] = float(unjudged)
    return out
