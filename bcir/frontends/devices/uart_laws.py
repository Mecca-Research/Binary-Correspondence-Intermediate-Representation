"""The UART protocol laws (blueprint slice U1), on the oracle rail.

`docs/kernel/BCIR_UART_DRIVER_BLUEPRINT.md` section 5 (U1). A C unit that includes a U0 header embeds
the `uart_regmap_contract` literal; `uart_unit_diagnostics` finds it, attests it against the
registry (`uart_model.verify_regmap_contract`) and runs `verify_uart_protocol` for its placement. A
unit without the literal is no UART unit and gets nothing: the laws are vacuous off-UART by
construction, and `bcir/tests/test_uart_laws.py` measures that over the whole cfront corpus. Each
finding is a message; the pipeline wraps them as advisory R3 diagnostics (`CompileResult
.uart_diagnostics`), as it wraps the call-ABI law's -- this module imports no verifier.

A register access is an MMIO-domain `c.load` / `c.store` whose byte offset from the device base can
be read off the claim graph, named by the placement's layout, its direction and the DLAB state
(`register_at`): the slot-0 and slot-1 accesses are DLL/DLM while a DLAB=1 bracket is open and
RBR/THR/IER otherwise, which is how the hardware itself decides. The laws (message shapes are the
blueprint's; each is `<function>: R3-uart: <message>`):

  1. DLAB bracket -- a bracket opens at an LCR store whose bit 7 is 1 and closes at the next whose
     bit 7 is 0. Only DLL, DLM and LCR may be touched inside it (`<REG> access inside a DLAB=1
     bracket`, hazard H-B); it must be straight-line, closed before any branch, loop, switch, call,
     jump, label or return and by the function's end (`DLAB bracket must be straight-line`); and
     every LCR store's bit 7 must be decidable (`LCR store with an undecidable DLAB bit`), or no
     bracket can be read at all.
  2. Divisor sanity -- `divisor 0 is forbidden` when a bracket closes over constant DLL and DLM
     stores of 0; and, when the unit declares its target rate as `uart_target_baud`, `baud error
     4.3% exceeds 3.0%` for a constant divisor whose rate misses it by more than
     `uart_model.BAUD_ERROR_LIMIT`, from the placement's reference clock.
  3. FCR gating -- on a placement with `fcr_gated`, an FCR store whose bit 0 is provably clear and
     whose other bits are not provably clear `programs nothing on <placement>` (H-C). A store of
     0, the 16450-mode write, programs what it says.
  4. Reconfigure only at idle -- an LCR, DLL or DLM store that some path reaches with no LSR read
     since its last THR store (or since the function's entry): `line reconfiguration without a
     dominating LSR (TEMT) check`, once per run.
  5. Read effects -- every register read must be a volatile access (`non-volatile access to UART
     register <REG>`, the clause the blueprint states for the MLIR rail), and since an RBR, IIR, LSR
     or MSR read changes device state, the plan must realize every one of them (`two <REG> reads
     merged`). The planner's CSE already refuses volatile and MMIO-domain claims categorically
     (`kbcir.realize.cse_eligible`); this law is the witness that it did.

Carried forward for the IRQ slice (blueprint section 3.2), and checked by no law yet: transmit never
depends on a THRE interrupt edge. The polled driver reads LSR.THRE before every burst; an interrupt
driver takes a THRE interrupt as a hint and re-checks LSR, because clones fail to re-assert it or
assert it spuriously.

Declared scope (docs/security/laws.md L18), and what the law does not do:
  * every MMIO access with a resolvable offset in a UART unit is a UART register access; a unit
    driving a second MMIO device beside the UART is out of scope (one device per unit);
  * offsets are read for `base[k]` with a constant index, a member or a dereference, a member array
    element `s.arr[k]`, and a base derived by `c.bin.add` of a constant, `c.copy` or a pointer cast;
    any other access is unidentified -- illegal inside a bracket, not policed outside one;
  * a constant is a literal, or integer arithmetic (`+ - * / % & | ^ << >>`), copies and casts over
    literals, folded exactly in the result's width, through values written once; beyond that, known
    bits flow through `|`, `&`, copies and casts (`fmt | 0x80` has bit 7 set whatever `fmt` is);
  * every function starts and must end with DLAB=0 (law 1 holds each function to it), so a callee
    is checked on its own and a call inside a bracket is a straight-line violation;
  * the walk follows every path through the structured body: branch arms meet (idle only if every
    arm is), a loop's head is walked again until it covers the way back from the loop's end,
    `break` and `continue` carry their state to their targets, a switch enters at every case and
    falls through, and a constant loop condition has only the exit it takes. A label assumes
    nothing, since a `goto` may reach it from anywhere: a reconfiguration after one needs an LSR
    read after it. A call to a function the unit defines leaves law 4's check as the callee
    returns it (a recursive call, as none); a call out of the unit is transparent to it;
  * the blueprint's two intent-keyed messages -- `DLL/DLM access outside a DLAB=1 bracket` and
    `<RBR|THR|IER> access inside a DLAB=1 bracket` -- need the register a driver *meant*, which no
    offset-addressed access carries once the preprocessor has made UART_RBR_OFF and UART_DLL_OFF
    the same 0; the hardware's own reading (above) is what this law enforces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from fractions import Fraction

from ...model import Claim, Domain
from . import uart_model as um

#: Registers whose read changes device state: LSR clears OE/PE/FE/BI (PC16550D p.21), IIR clears a
#: THRE indication (Table 5, p.21), MSR clears the deltas (p.24), RBR pops the FIFO and the
#: character timeout (p.16).
READ_CLEAR = ("RBR", "IIR", "LSR", "MSR")
#: The only registers a DLAB=1 bracket may touch (hazard H-B: clone EFRs behind DLAB).
BRACKET_REGS = ("DLL", "DLM", "LCR")
#: The stores that reconfigure the line.
RECONFIG_REGS = ("LCR", "DLL", "DLM")

_DLAB = 0x80
_MASK = 0xFFFF_FFFF
#: The integer operations a constant index or offset is folded through, exactly; None refuses.
_FOLD = {
    "c.bin.add": lambda a, b: a + b,
    "c.bin.sub": lambda a, b: a - b,
    "c.bin.mul": lambda a, b: a * b,
    "c.bin.div": lambda a, b: a // b if b else None,
    "c.bin.mod": lambda a, b: a % b if b else None,
    "c.bin.and": lambda a, b: a & b,
    "c.bin.or": lambda a, b: a | b,
    "c.bin.xor": lambda a, b: a ^ b,
    "c.bin.shl": lambda a, b: a << b if b < 64 else None,
    "c.bin.shr": lambda a, b: a >> b if b < 64 else None,
}
_CONTRACT = "uart_regmap_contract"
_TARGET = "uart_target_baud"
_LITERAL = re.compile(r'\{"([a-z][a-z0-9_]*)", ([0-9]+), ([0-9]+), ([0-9]+), ([0-9]+)\}', re.ASCII)
_DECIMAL = re.compile(r"[0-9]+", re.ASCII)


def register_at(placement, offset: int, write: bool, dlab: int | None) -> str | None:
    """The register an access at byte `offset` reaches, by direction and the DLAB state; None when
    no register answers it (or the DLAB state that would decide it is unknown)."""
    for reg in um.UART16550:
        if um.reg_offset(placement, reg) != offset:
            continue
        if reg.dlab is not None and reg.dlab != dlab:
            continue
        if ("w" if write else "r") in reg.access:
            return reg.name
    return None


def _slot_name(placement, offset: int, write: bool, dlab: int | None) -> str | None:
    """The register an access names: its own direction's, else the slot's other one (a write to
    the read-only LSR still lands on LSR)."""
    return register_at(placement, offset, write, dlab) or register_at(
        placement, offset, not write, dlab
    )


# --- what the claim graph says about values and addresses ---------------------------------------------


class _Facts:
    """One function's constants, single-write definitions and register-access offsets."""

    def __init__(self, lf):
        self.lf = lf
        writers: dict[int, int] = {}
        for claim in lf.claims:
            for rid in claim.wr:
                writers[rid] = writers.get(rid, 0) + 1
        self.defs = {rid: claim for claim in lf.claims for rid in claim.wr if writers.get(rid) == 1}
        self.consts = {
            claim.wr[0]: claim.imm[0]
            for claim in lf.claims
            if claim.op == "c.const"
            and claim.imm
            and type(claim.imm[0]) is int  # a floating constant has no bits here
            and writers.get(claim.wr[0]) == 1
        }
        self.derived: dict[int, tuple[int, int | None]] = {}
        self.offsets: dict[int, int | None] = {}
        for claim in lf.claims:
            self._derive(claim)
            if claim.op in ("c.load", "c.store") and claim.domain == Domain.MMIO:
                self.offsets[claim.id] = self._offset(claim)

    def _pointee(self, rid: int) -> int | None:
        ct = self.lf.rid_types.get(rid)
        if ct is None or ct.kind not in ("pointer", "array") or ct.of is None:
            return None
        return max(1, ct.of.size)

    def _root(self, rid: int) -> tuple[int, int | None]:
        return self.derived.get(rid, (rid, 0))

    def _derive(self, claim) -> None:
        """A pointer the claim makes from another: `p + k`, a copy, a cast."""
        if len(claim.wr) != 1 or self.defs.get(claim.wr[0]) is not claim:
            return
        out = claim.wr[0]
        if self._pointee(out) is None:
            return
        if claim.op == "c.copy" or claim.op.startswith("c.cast:"):
            self.derived[out] = self._root(claim.rd[0]) if claim.rd else (out, None)
        elif claim.op == "c.bin.add" and len(claim.rd) == 2:
            base, k = claim.rd
            root, off = self._root(base)
            size = self._pointee(base)
            step = self.value(k)
            if step is not None and size is not None and off is not None:
                self.derived[out] = (root, off + step * size)
            else:
                self.derived[out] = (root, None)

    def _offset(self, claim) -> int | None:
        load = claim.op == "c.load"
        rd, imm = tuple(claim.rd), tuple(claim.imm)
        if not rd:
            return None
        _root, off = self._root(rd[0])
        if off is None:
            return None
        indexed = len(rd) == (2 if load else 3)
        if not indexed:
            return off + (imm[0] if imm else 0)
        k = self.value(rd[1])
        if k is None:
            return None
        if not imm:
            size = self._pointee(rd[0])
            return None if size is None else off + k * size
        if len(imm) == 2:  # a member array element: (member offset, element size)
            return off + imm[0] + k * imm[1]
        return None  # a strided or flagged member array: outside the declared scope

    def _width(self, rid: int) -> int | None:
        """The value mask of an integer scalar `rid`; None for anything else (a pointer)."""
        ct = self.lf.rid_types.get(rid)
        if ct is None or ct.kind != "scalar" or ct.size not in (1, 2, 4, 8):
            return None
        return (1 << (8 * ct.size)) - 1

    def value(self, rid: int, depth: int = 0) -> int | None:
        """`rid`'s value when the claim graph fixes it: a constant, or integer arithmetic, copies
        and casts over constants -- exact, in the result's width; never a negative value."""
        if rid in self.consts:
            return self.consts[rid] if self.consts[rid] >= 0 else None
        claim = self.defs.get(rid)
        width = self._width(rid)
        if claim is None or width is None or depth > 32 or not claim.rd:
            return None
        if claim.op in _FOLD and len(claim.rd) == 2:
            a = self.value(claim.rd[0], depth + 1)
            b = self.value(claim.rd[1], depth + 1)
            if a is None or b is None:
                return None
            result = _FOLD[claim.op](a, b)
            return None if result is None or result < 0 else result & width
        if claim.op == "c.copy" or claim.op.startswith("c.cast:"):
            inner = self.value(claim.rd[0], depth + 1)
            return None if inner is None else inner & width
        return None

    def known(self, rid: int, depth: int = 0) -> tuple[int, int]:
        """(ones, zeros): the bits of `rid`'s value known to be 1 and known to be 0."""
        fixed = self.value(rid)
        if fixed is not None:
            fixed &= _MASK
            return fixed, ~fixed & _MASK
        claim = self.defs.get(rid)
        if claim is None or depth > 32 or not claim.rd:
            return 0, 0
        if claim.op in ("c.bin.or", "c.bin.and") and len(claim.rd) == 2:
            a1, a0 = self.known(claim.rd[0], depth + 1)
            b1, b0 = self.known(claim.rd[1], depth + 1)
            if claim.op == "c.bin.or":
                return a1 | b1, a0 & b0
            return a1 & b1, a0 | b0
        if claim.op == "c.copy" or claim.op.startswith("c.cast:"):
            ones, zeros = self.known(claim.rd[0], depth + 1)
            out = self.lf.rid_types.get(rid)
            if out is not None and out.kind == "scalar" and 0 < out.size < 4:
                width = (1 << (8 * out.size)) - 1  # a narrowing cast clears what it drops
                return ones & width, zeros | (~width & _MASK)
            return ones, zeros
        return 0, 0

    def constant(self, rid: int) -> int | None:
        ones, zeros = self.known(rid)
        return ones if ones | zeros == _MASK else None


# --- the walk -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _State:
    dlab: int | None = 0  # 0 at entry; None once undecidable
    idle: bool = False  # an LSR read dominates since the last THR store
    divisor: tuple | None = None  # (DLL, DLM) stored in the open bracket, None for not or unknown


#: The state at a label: a jump may arrive from anywhere, so nothing is known there.
_ANYWHERE = _State(dlab=None, idle=False)
#: A loop's head state only climbs (DLAB known to unknown, idle to not), so it settles within three
#: rounds; the bound is a guard, and a loop that reached it would be walked from `_ANYWHERE`.
_ROUNDS = 8
#: The calls whose callee the unit defines (`c.call:<name>`, `c.call.void:<name>`).
_UNIT_CALLS = ("c.call", "c.call.void")


def _join(*states: _State | None) -> _State | None:
    """The state where control paths meet: the DLAB value they agree on, idle only if every path
    is; None is a path that never arrives (after a return, a break, a jump)."""
    live = [s for s in states if s is not None]
    if len(live) < 2:
        return live[0] if live else None
    return _State(
        dlab=live[0].dlab if all(s.dlab == live[0].dlab for s in live) else None,
        idle=all(s.idle for s in live),
    )


class _Unit:
    """What a function's walk needs from its unit: the placement, the declared rate, each function's
    facts, and what a call to one of the unit's own functions leaves of the LSR check."""

    def __init__(self, lowered, placement, target_baud: int | None):
        self.lowered, self.p, self.target = lowered, placement, target_baud
        self._facts: dict[str, _Facts] = {}
        self._idle: dict[str, tuple | None] = {}

    def facts(self, fn: str) -> _Facts:
        if fn not in self._facts:
            self._facts[fn] = _Facts(self.lowered.functions[fn])
        return self._facts[fn]

    def idle_after(self, fn: str, idle: bool) -> bool | None:
        """Whether a call to `fn` returns with an LSR read standing, entered with one standing
        (`idle`) or not; None when it never returns. A recursive call returns with none."""
        if fn not in self._idle:
            self._idle[fn] = None  # what a recursive call meets while the summary is computed
            self._idle[fn] = tuple(_Function(fn, self).exit_idle(entry) for entry in (True, False))
        summary = self._idle[fn]
        return False if summary is None else summary[0 if idle else 1]


class _Function:
    """The laws over one function's structured body, along every path through it."""

    def __init__(self, fn: str, unit: _Unit):
        self.fn, self.unit, self.p = fn, unit, unit.p
        self.target = unit.target
        self.lf = unit.lowered.functions[fn]
        self.facts = unit.facts(fn)
        self.msgs: list[str] = []
        self.reads: dict[str, list[int]] = {}  # read-clear register -> its read claims
        # per enclosing loop or switch, innermost last: (its breaks, its continues -- None for a switch)
        self.exits: list[tuple[list, list | None]] = []
        self.returns: list[_State | None] = []  # the states the function returns with

    def say(self, message: str, report: bool) -> None:
        if report:
            self.msgs.append(f"{self.fn}: R3-uart: {message}")

    def run(self) -> list[str]:
        self.closed(self.block(self.lf.body, _State(), True), True)
        return self.msgs

    def exit_idle(self, idle: bool) -> bool | None:
        """Whether the function returns with an LSR read standing, entered with one (or not)."""
        end = _join(self.block(self.lf.body, _State(idle=idle), False), *self.returns)
        return None if end is None else end.idle

    def closed(self, state: _State | None, report: bool) -> _State | None:
        """`state` where a region ends -- a branch arm, a trip round a loop, a case, the function: a
        bracket opened in it must have closed in it."""
        if state is None or state.dlab != 1:
            return state
        self.say("DLAB bracket must be straight-line", report)
        return replace(state, dlab=None, divisor=None)

    def block(self, nodes, state: _State | None, report: bool) -> _State | None:
        """The state after `nodes` entered with `state`; None when no path falls out of them."""
        for node in nodes:
            if isinstance(node, Claim):
                state = state if state is None else self.claim(node, state, report)
                continue
            kind = type(node).__name__
            if kind in ("ScopeOpen", "ScopeClose"):
                continue
            if state is not None and state.dlab == 1:  # control flow of any kind inside a bracket
                self.say("DLAB bracket must be straight-line", report)
                state = replace(state, dlab=None, divisor=None)
            if kind == "IfNode":
                then = self.closed(self.block(node.then, state, report), report)
                state = _join(then, self.closed(self.block(node.els, state, report), report))
            elif kind == "WhileNode":
                state = self.loop(node, state, report)
            elif kind == "SwitchNode":
                state = self.switch(node, state, report)
            elif kind == "BreakNode":
                if self.exits:
                    self.exits[-1][0].append(state)
                state = None
            elif kind == "ContinueNode":
                loops = [continues for _breaks, continues in self.exits if continues is not None]
                if loops:
                    loops[-1].append(state)
                state = None
            elif kind == "ReturnNode":
                self.returns.append(state)
                state = None
            elif kind in ("GotoNode", "ComputedGotoNode"):
                state = None
            else:  # a label, where a jump may land, or a construct this walk does not know
                state = _ANYWHERE
        return state

    def loop(self, node, entry: _State | None, report: bool) -> _State | None:
        """A loop's head is entered from before the loop and from the end of every trip round it,
        so the walk climbs to the head state that covers both, then reports from there once."""
        head = entry
        for _ in range(_ROUNDS):
            back, _out = self.trip(node, head, False)
            settled = _join(head, back)
            if settled == head:
                break
            head = settled
        else:
            head = _ANYWHERE
        return self.trip(node, head, report)[1]

    def trip(self, node, head: _State | None, report: bool) -> tuple:
        """One trip round a loop from its head: (the state it comes back to the head with, the state
        it leaves with). A while/for loop tests at its head, a do/while after its body; a `continue`
        goes on to the step, a `break` leaves; a constant condition has only the way it takes."""
        test = self.facts.value(node.cond)
        always, never = test not in (None, 0), test == 0
        breaks: list = []
        continues: list = []
        self.exits.append((breaks, continues))
        if node.test_at_end:
            body = self.block(node.body, head, report)
            stepped = self.block(node.step, _join(body, *continues), report)
            tested = self.closed(self.block(node.cond_block, stepped, report), report)
            back = None if never else tested
        else:
            tested = self.block(node.cond_block, head, report)
            body = self.block(node.body, None if never else tested, report)
            back = self.closed(self.block(node.step, _join(body, *continues), report), report)
        self.exits.pop()
        return back, _join(None if always else tested, *breaks)

    def switch(self, node, entry: _State | None, report: bool) -> _State | None:
        """A switch enters at the case its discriminant picks and falls through from one case into
        the next; it leaves at a `break` -- and, without a `default`, straight from its head."""
        breaks: list = []
        self.exits.append((breaks, None))
        state, default = None, False
        for item in node.body:
            kind = type(item).__name__
            if kind in ("CaseLabel", "DefaultLabel"):
                default = default or kind == "DefaultLabel"
                state = _join(self.closed(state, report), entry)
            else:
                state = self.block([item], state, report)
        self.exits.pop()
        return _join(self.closed(state, report), *breaks, None if default else entry)

    def claim(self, c, state: _State, report: bool) -> _State | None:
        if c.op.startswith("c.call"):
            if state.dlab == 1:
                self.say("DLAB bracket must be straight-line", report)
                state = replace(state, dlab=None, divisor=None)
            kind, _sep, callee = c.op.partition(":")
            if kind in _UNIT_CALLS and callee in self.unit.lowered.functions:
                idle = self.unit.idle_after(callee, state.idle)
                return None if idle is None else replace(state, idle=idle)
            return state
        if c.op not in ("c.load", "c.store") or c.domain != Domain.MMIO:
            return state
        write = c.op == "c.store"
        offset = self.facts.offsets.get(c.id)
        reg = None if offset is None else register_at(self.p, offset, write, state.dlab)
        if not write and reg is not None:
            if not c.volatile:
                self.say(f"non-volatile access to UART register {reg}", report)
            if reg in READ_CLEAR and report:
                self.reads.setdefault(reg, []).append(c.id)
        if state.dlab == 1 and reg not in BRACKET_REGS:
            name = None if offset is None else _slot_name(self.p, offset, write, state.dlab)
            self.say(f"{name or 'unidentified register'} access inside a DLAB=1 bracket", report)
        if write and reg in RECONFIG_REGS and not state.idle:
            self.say("line reconfiguration without a dominating LSR (TEMT) check", report)
            state = replace(state, idle=True)  # once per run, not once per store
        if not write and reg == "LSR":
            state = replace(state, idle=True)
        if write and reg == "THR":
            state = replace(state, idle=False)
        if not write:
            return state
        value = c.rd[-1]
        if reg == "FCR" and self.p.fcr_gated:
            ones, zeros = self.facts.known(value)
            rest = self.p.fcr_impl_mask & ~0x01 & 0xFF
            if zeros & 0x01 and zeros & rest != rest:
                self.say(
                    f"FCR write with FIFO-enable clear programs nothing on {self.p.name}", report
                )
        if reg in ("DLL", "DLM") and state.dlab == 1:
            byte = self.facts.constant(value)
            low, high = state.divisor or (None, None)
            byte = None if byte is None else byte & 0xFF
            state = replace(state, divisor=(byte, high) if reg == "DLL" else (low, byte))
        if reg == "LCR":
            ones, zeros = self.facts.known(value)
            if ones & _DLAB:
                if state.dlab != 1:
                    state = replace(state, dlab=1, divisor=None)
            elif zeros & _DLAB:
                if state.dlab == 1:
                    self.divisor_law(state.divisor, report)
                state = replace(state, dlab=0, divisor=None)
            else:
                self.say("LCR store with an undecidable DLAB bit", report)
                state = replace(state, dlab=None, divisor=None)
        return state

    def divisor_law(self, divisor, report: bool) -> None:
        if divisor is None or None in divisor:
            return  # a latch the bracket left alone, or a value the claim graph does not fix
        low, high = divisor
        value = (high << 8) | low
        if value == 0:
            self.say("divisor 0 is forbidden", report)
            return
        if self.target is None:
            return
        error = abs(um.baud_error(self.p.ref_clk_hz, value, self.target))
        if error > um.BAUD_ERROR_LIMIT:
            self.say(
                f"baud error {_percent(error)}% exceeds {_percent(um.BAUD_ERROR_LIMIT)}%", report
            )


def _percent(fraction: Fraction) -> str:
    """A ratio as a percentage with one decimal, rounded half up, exactly."""
    tenths = int(fraction * 1000 + Fraction(1, 2))
    return f"{tenths // 10}.{tenths % 10}"


def verify_uart_protocol(
    lowered, placement, plans=None, target_baud: int | None = None
) -> list[str]:
    """The five UART protocol laws over a lowered unit laid out for `placement`: one message per
    violation, `<function>: R3-uart: <message>` ([] == clean). `plans` (function -> its K_BCIR
    realization) arms law 5's merged-read check; `target_baud` law 2's rate bound."""
    problems = um.placement_problems(placement)
    if problems:
        return [f"R3-uart: malformed placement: {'; '.join(problems)}"]
    unit = _Unit(lowered, placement, target_baud)
    msgs: list[str] = []
    for fn in lowered.functions:
        walk = _Function(fn, unit)
        msgs += walk.run()
        plan = (plans or {}).get(fn)
        if plan is None:
            continue
        realized = {step.claim_id for step in plan.steps}
        for reg, ids in walk.reads.items():
            if sum(1 for i in ids if i in realized) < len(ids):
                msgs.append(f"{fn}: R3-uart: two {reg} reads merged")
    return msgs


# --- the unit: find the contract literal, attest it, run the laws ---------------------------------------


def _global_init(lowered, name: str) -> str | None:
    for entry in lowered.globals_decl:
        if entry[0] == name and entry[2] and isinstance(entry[2][0], str):
            return entry[2][0]
    return None


def uart_unit_diagnostics(lowered, plans=None) -> tuple:
    """(the RegMapContract the unit attests, its messages) -- (None, []) for a unit without the U0
    header's `uart_regmap_contract`, which is no UART unit."""
    literal = _global_init(lowered, _CONTRACT)
    if literal is None:
        return None, []
    match = _LITERAL.fullmatch(literal)
    if match is None:
        return None, [f"R3-uart: the {_CONTRACT} literal {literal!r} is not a U0 header's"]
    name = match.group(1)
    stride, base, fcr_reset, ref_clk_hz = (int(g) for g in match.groups()[1:])
    placement = um.PLACEMENTS.get(name)
    laid_out = um.regmap_contract_for(placement).regs if placement is not None else ()
    contract = um.RegMapContract(placement=name, stride=stride, base_offset=base, regs=laid_out)
    check = placement if placement is not None else um.PLACEMENTS[um.DEFAULT_PLACEMENT]
    msgs = [f"R3-uart: regmap contract: {m}" for m in um.verify_regmap_contract(contract, check)]
    if placement is None:
        return None, msgs
    if fcr_reset != placement.fcr_reset:
        msgs.append(
            f"R3-uart: regmap contract: fcr_reset {fcr_reset:#04x} != placement "
            f"{placement.fcr_reset:#04x}"
        )
    if ref_clk_hz != placement.ref_clk_hz:
        msgs.append(
            f"R3-uart: regmap contract: ref_clk_hz {ref_clk_hz} != placement {placement.ref_clk_hz}"
        )
    target_text = _global_init(lowered, _TARGET)
    target = None
    if target_text is not None:
        if _DECIMAL.fullmatch(target_text) and int(target_text) > 0:
            target = int(target_text)
        else:
            msgs.append(f"R3-uart: {_TARGET} {target_text!r} is not a positive integer constant")
    msgs += verify_uart_protocol(lowered, placement, plans=plans, target_baud=target)
    return contract, msgs
