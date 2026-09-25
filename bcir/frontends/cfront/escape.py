"""Escape analysis, indirect-call narrowing and the effect footprint of a lowered C unit (GEM+ G10).

One analysis answers three questions the call graph could not:

  * **which memory a function touches** -- its read and write FOOTPRINT, the input of
    `CompileResult.commute`. The footprint used to take a write from a claim's `wr` alone, and a
    cfront store is `c.store rd=(base, [index,] value) wr=()`: every store was recorded as a READ of
    its base, and every write through a pointer (a parameter, a global pointer) and every static
    local was invisible, so two functions that race on one array were reported to commute;
  * **which local memory never leaves its function** -- ESCAPE: a local array whose address never
    crosses a call boundary and never reaches a global, a static, unknown code or a return value is
    private to its activation (the roadmap's "RID never crosses a call boundary"), which licenses a
    frame slot instead of a registry resource;
  * **which functions an indirect call can reach** -- NARROWING: a `c.call.indirect` or
    `c.call.imember` whose function-pointer value can only hold known functions of this unit is a
    known edge, and one with exactly one is resolved; its footprint is its targets', not everything.

The analysis is Andersen's: inclusion-based, flow-insensitive (the order of claims does not
matter -- the two cfront rails evaluate sibling expressions in different orders), context- and
field-insensitive, and OPEN WORLD -- a translation unit is not the whole program, so a non-static
function can be called from another unit with anything, and a global can be written from one:

  * an OBJECT is a global (`G`, by name), a static local (`S`), an automatic local (`L`), a
    function (`F`), the string literals (`STR`, read-only) or unknown memory (`TOP`);
  * every rid has a storage node; `pt(node)` is the set of objects the pointer it HOLDS may point
    to (for an aggregate, any pointer stored anywhere in it). What a node holds also includes
    unknown pointers when unknown code can have written it: a global, a static, or an object that
    escaped;
  * the rid's C type decides how it is used: an ARRAY used as a value is its own address (the
    decay), a struct, pointer or scalar is what it holds; a load or store through an array or a
    struct touches that object, through a pointer (or a pointer-valued temporary) what it holds;
  * a call binds actuals to formals and the callee's return values to its result; an unknown
    callee -- another translation unit, `extern`, inline assembly, an unresolved indirect call --
    receives its actuals into unknown memory (they ESCAPE) and returns unknown pointers;
  * the parameters of a function another unit can call hold unknown pointers: a non-static
    function, or one whose address is taken -- by a claim, or by a file-scope initializer (an ops
    table `struct ops t = { handler };` hands `handler` to whoever reads the table);
  * memory no declaration names is still memory: an allocator's result is the calling function's
    HEAP object (one per function, every allocation site merged), and a pointer made from an
    integer (`(T *)0x40000000u`, `(T *)addr`) points to unknown memory -- only a pointer variable,
    an array, an address (`&x`), another pointer cast, a function, a string literal or the null
    constant is known not to be one;
  * a volatile (MMIO-domain) access is an observable side effect on state the unit cannot name:
    it reads and writes unknown memory, so two device accesses never commute. A claim is one when
    it is MMIO-domain, or when it is a load or store whose BASE resource is (`_device`): the base
    decides, not how a lowering spelled the claim.

What is not claimed: flow sensitivity (`f = add1; f(x); f = dbl; f(x)` narrows both sites to
{add1, dbl}), field sensitivity (a struct of two function pointers holds both), global
initializers' pointer contents (a pointer loaded from a global is unknown), a pointer forged by
type punning (an integer stored into memory and reloaded as a pointer through a union), a volatile
access neither frontend carries the qualifier to -- a volatile member of a non-volatile struct, a
volatile file-scope variable, a file-scope or member pointer to volatile: each reads as an ordinary
access -- and a unit with a call carrying more operands than a C-twin claim holds -- both rails
REFUSE it (every footprint is `*`, no verdict, no narrowed site), because the C twin never sees the
dropped operands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bcir.model.lanes import Domain

from .lower import (
    _IO_PORT_RID,
    ComputedGotoNode,
    IfNode,
    ReturnNode,
    SwitchNode,
    WhileNode,
)

#: The object for unknown memory, and the footprint name every conflict test treats as "anything".
TOP = ("TOP",)
UNKNOWN = "*"
#: The band the lowering allocates file-scope globals from, and the string-literal / function-value
#: band above it (`lower.py`: `900000 + gi`, `970000 + idx`).
_SHARED_RID = 900000
_CONST_RID = 970000
#: The operand capacity of a C-twin claim (`BCIR_CLAIM_MAX_RD`): a call with more reads than this
#: cannot be represented there, so both rails refuse to analyze such a unit (see `analyze`).
CLAIM_MAX_RD = 6
#: The allocators: each returns fresh memory, the calling function's heap object.
_ALLOCATORS = frozenset(
    f"c.call.libm:{name}" for name in ("malloc", "calloc", "realloc", "aligned_alloc")
)

# Ops that move no pointer value and touch no memory.
_NO_FLOW_PREFIXES = ("c.const", "c.fconst:", "c.cconst:", "c.labeladdr:", "c.sizeof.vla", "c.fence")
_ATOMIC_PREFIXES = ("c.atomic.", "c.c11atom.", "c.cmpxchg.")
_EXTERNAL_PREFIXES = ("c.call.tu:", "c.call.extern:", "c.asm:", "c.asm.volatile:")
_LIBRARY_PREFIXES = ("c.call.libm:", "c.call.libm.void:", "c.call.builtin:", "c.call.vabuiltin:")
_DIRECT_PREFIXES = ("c.call:", "c.call.void:")

#: The verdicts the escape report gives each named automatic local memory object.
VERDICTS = ("nonescaping", "lent", "escaping")


@dataclass(frozen=True)
class Footprint:
    """A function's read and write footprint, by NAME: a global's name, a static local's
    `function.name`, and `*` for memory the unit cannot name (reached through an unknown pointer,
    or a local another function can reach). A function's own private locals are not in it."""

    reads: frozenset
    writes: frozenset

    def conflicts(self, other: "Footprint") -> bool:
        """RAW / WAR / WAW: one side writes what the other reads or writes. `*` may be anything,
        so it conflicts with every name, itself included."""
        return _overlap(self.writes, other.reads | other.writes) or _overlap(
            other.writes, self.reads
        )


def _overlap(a: frozenset, b: frozenset) -> bool:
    if not a or not b:
        return False
    return UNKNOWN in a or UNKNOWN in b or bool(a & b)


@dataclass(frozen=True)
class IndirectSite:
    """One indirect call: its function, claim id, op, and the functions it can reach. `targets`
    is None when the pointer may hold an unknown function (the site is an external edge)."""

    function: str
    claim: int
    op: str
    targets: tuple | None

    @property
    def resolved(self) -> bool:
        """Narrowed to exactly one function of this unit."""
        return self.targets is not None and len(self.targets) == 1

    def report(self) -> str:
        return UNKNOWN if self.targets is None else ",".join(self.targets)


@dataclass
class EscapeResult:
    """Everything `analyze` derives: the footprint of every function, the escape verdict of every
    named automatic local memory object, the narrowed indirect calls, and the call graph they
    extend (`edges[f]`: the defined functions `f` calls, directly or through a known pointer)."""

    footprints: dict = field(default_factory=dict)  # fn -> Footprint
    objects: dict = field(default_factory=dict)  # fn -> {local name: verdict} (named, automatic)
    candidates: dict = field(default_factory=dict)  # fn -> {name: verdict} (declared_extent)
    sites: list = field(default_factory=list)  # IndirectSite, in unit order
    edges: dict = field(default_factory=dict)  # fn -> frozenset of defined callees
    truncated: bool = False  # a call carries more operands than a C-twin claim can

    def commute(self, a: str, b: str) -> bool:
        return not self.footprints[a].conflicts(self.footprints[b])

    def counts(self) -> dict:
        """The G10 rows' raw counts over this unit."""
        cand = [v for per in self.candidates.values() for v in per.values()]
        return {
            "candidates": len(cand),
            "nonescaping": sum(v == "nonescaping" for v in cand),
            "lent": sum(v == "lent" for v in cand),
            "escaping": sum(v == "escaping" for v in cand),
            "indirect": len(self.sites),
            "resolved": sum(s.resolved for s in self.sites),
            "known": sum(s.targets is not None for s in self.sites),
        }


# --- the unit, as the analysis reads it --------------------------------------------------------


def _kind(ct) -> str:
    """How a rid's C type makes it behave: `array` (a value is its address), `struct` (an object
    touched in place, a value is its contents), `pointer`, or `scalar`. An untyped rid is
    `unknown`, and gets both readings (its address and what it holds)."""
    if ct is None:
        return "unknown"
    if ct.kind == "array":
        return "array"
    if ct.kind in ("struct", "union"):
        return "struct"
    if ct.kind in ("pointer", "funcptr"):
        return "pointer"
    return "scalar"


class _Unit:
    """The lowered unit's facts the rules need, per function: the category and kind of every rid,
    the return and condition rids the body tree names without a claim, and the parameters."""

    def __init__(self, lowered) -> None:
        self.lowered = lowered
        self.functions = lowered.functions
        self.static_rids = {
            name: {rid: sname for rid, sname, _ct, *_ in lf.statics}
            for name, lf in self.functions.items()
        }
        self.returns = {name: _return_rids(lf.body) for name, lf in self.functions.items()}
        self.conditions = {name: _condition_rids(lf.body) for name, lf in self.functions.items()}
        self.local_names = {
            name: {rid: lname for rid, lname, _ct in (*lf.locals, *lf.vla_locals)}
            for name, lf in self.functions.items()
        }
        self.params = {
            name: [rid for _n, rid, _ct in lf.params] for name, lf in self.functions.items()
        }
        # the DECLARED variables (parameters, locals, statics) -- as opposed to temporaries -- and
        # the claim that first writes each rid (a temporary is written once)
        self.named = {
            name: set(self.params[name])
            | {rid for rid, *_ in (*lf.locals, *lf.vla_locals, *lf.statics)}
            for name, lf in self.functions.items()
        }
        self.defs = {}
        for name, lf in self.functions.items():
            first: dict = {}
            for c in lf.claims:
                for w in c.wr:
                    first.setdefault(w, c)
            self.defs[name] = first
        # the identifiers file-scope initializers name: a function among them is address-taken
        self.init_refs = frozenset(getattr(lowered, "init_refs", ()))
        # MEMORY rids: those whose storage the program addresses -- written by no claim as a
        # result (an array, a parameter, a global stored to), or a load/store/address-of/dispatch
        # base, or a declared VLA. Only these are reported as local objects.
        self.memory = {}
        for name, lf in self.functions.items():
            written = {w for c in lf.claims for w in c.wr}
            bases = {
                c.rd[0]
                for c in lf.claims
                if c.rd
                and (c.op in ("c.load", "c.store", "c.addrof") or c.op.startswith("c.call.imember"))
            }
            vlas = {c.wr[0] for c in lf.claims if c.op == "c.vladecl" and c.wr}
            used = {r for c in lf.claims for r in (*c.rd, *c.wr)}
            used |= self.returns[name] | self.conditions[name] | set(self.params[name])
            self.memory[name] = {r for r in used if r not in written} | bases | vlas

    def node(self, fn: str, rid: int):
        """The storage node of `rid` in `fn`, or a constant object for a string literal / function
        value (which have no storage the program can write)."""
        lf = self.functions[fn]
        statics = self.static_rids[fn]
        if rid in statics:
            return ("S", f"{fn}.{statics[rid]}")
        if rid >= _SHARED_RID or rid == _IO_PORT_RID:
            name = lf.globals_used.get(rid) or getattr(self.lowered.resources.get(rid), "name", "")
            if rid >= _CONST_RID and rid != _IO_PORT_RID:
                ct = lf.rid_types.get(rid)
                if ct is not None and ct.kind == "funcptr":
                    return ("F", name)
                return ("STR",)
            return ("G", name)
        return ("L", fn, rid)

    def kind(self, fn: str, rid: int) -> str:
        return _kind(self.functions[fn].rid_types.get(rid))

    def declared(self, fn: str, rid: int) -> bool:
        """A variable the source declares (a global, a static, a parameter, a local), not a
        temporary the lowering made."""
        return rid in self.named[fn] or rid >= _SHARED_RID or rid == _IO_PORT_RID

    def callers_unknown(self, fn: str, address_taken: set) -> bool:
        """A function another translation unit can call with anything: not `static`, or its
        address is taken -- by a claim, or by a file-scope initializer (the pointer may reach
        code this analysis does not see)."""
        lf = self.functions[fn]
        return not getattr(lf, "static_fn", False) or fn in address_taken or fn in self.init_refs


def _return_rids(body: list) -> set:
    out: set = set()

    def walk(nodes):
        for n in nodes:
            if isinstance(n, ReturnNode):
                if n.rid is not None:
                    out.add(n.rid)
            elif isinstance(n, IfNode):
                walk(n.then)
                walk(n.els)
            elif isinstance(n, WhileNode):
                walk(n.cond_block)
                walk(n.body)
                walk(n.step)
            elif isinstance(n, SwitchNode):
                walk(n.body)

    walk(body)
    return out


def _condition_rids(body: list) -> set:
    out: set = set()

    def walk(nodes):
        for n in nodes:
            if isinstance(n, IfNode):
                out.add(n.cond)
                walk(n.then)
                walk(n.els)
            elif isinstance(n, WhileNode):
                out.add(n.cond)
                walk(n.cond_block)
                walk(n.body)
                walk(n.step)
            elif isinstance(n, SwitchNode):
                out.add(n.disc)
                walk(n.body)
            elif isinstance(n, ComputedGotoNode):
                out.add(n.target)

    walk(body)
    return out


def _is(op: str, prefixes) -> bool:
    return any(op.startswith(p) for p in prefixes)


def _is_indirect(op: str) -> bool:
    return op == "c.call.indirect" or op.startswith("c.call.imember")


def _is_pointer_cast(op: str) -> bool:
    """`(T *)v` -- both rails spell a cast to a pointer type `c.cast:<pointee> *`."""
    return op.startswith("c.cast:") and op.endswith("*")


# --- the solver ------------------------------------------------------------------------------


class _Solver:
    def __init__(self, unit: _Unit) -> None:
        self.u = unit
        self.pt: dict = {}  # node -> set of objects it holds pointers to
        self.escaped: set = set()  # objects stored into unknown memory
        self.address_taken: set = set()  # functions whose value is used
        self.changed = False

    # what a node holds, what a rid is as a value, and what a base addresses

    def held(self, node) -> set:
        """The pointers a storage node holds: what the program stored, plus unknown pointers when
        unknown code can have written it -- a global, a static, an escaped object."""
        out = set(self.pt.get(node, ()))
        if node[0] in ("G", "S") or node in self.escaped:
            out.add(TOP)
        return out

    def val(self, fn: str, rid: int) -> set:
        """The objects `rid`'s VALUE may point to."""
        node = self.u.node(fn, rid)
        if node[0] == "F":
            if node[1] not in self.address_taken:  # a change: its parameters may now be unknown
                self.address_taken.add(node[1])
                self.changed = True
            return {node}
        if node[0] == "STR":
            return {node}
        kind = self.u.kind(fn, rid)
        if kind == "array":
            return {node}  # the decay: the array's own address
        if kind == "unknown":
            return self.held(node) | {node}
        return self.held(node)

    def deref(self, fn: str, rid: int) -> set:
        """The objects a load or store with base `rid` touches: an array or struct in place, a
        scalar VARIABLE in place (`a++` on a parameter stores to `a`), what a pointer or a
        temporary holding an address points to. A string literal is read-only memory."""
        node = self.u.node(fn, rid)
        if node[0] in ("F", "STR"):
            return {node}
        kind = self.u.kind(fn, rid)
        if kind in ("array", "struct"):
            return {node}
        if kind == "unknown":
            return self.held(node) | {node}  # an untyped base: both readings
        if kind == "scalar" and self.u.declared(fn, rid):
            return {node}
        return self.held(node)

    def address(self, fn: str, rid: int) -> set:
        """What `c.addrof` of base `rid` points to: the object itself (`&x`, `&a[i]`, `&s.f`), or,
        for a pointer or a temporary holding an address, into what it holds (`&p->f`, `&p[i]`) as
        well as the variable (`&p`)."""
        node = self.u.node(fn, rid)
        if node[0] in ("F", "STR"):
            return {node}
        kind = self.u.kind(fn, rid)
        if kind in ("array", "struct") or (kind == "scalar" and self.u.declared(fn, rid)):
            return {node}
        return {node} | self.held(node)

    def forged(self, fn: str, rid: int) -> bool:
        """Whether `(T *)rid` may make a pointer out of an integer: unless the operand is provably
        a pointer -- a pointer or array variable, an address (`&x`), another pointer cast, a
        function, a string literal -- or the null constant, the result points to unknown memory."""
        node = self.u.node(fn, rid)
        if node[0] in ("F", "STR"):
            return False
        if self.u.declared(fn, rid):
            return self.u.kind(fn, rid) not in ("pointer", "array")
        c = self.u.defs[fn].get(rid)
        if c is None:
            return True
        if c.op == "c.addrof" or _is_pointer_cast(c.op):
            return False
        return not (c.op == "c.const" and tuple(c.imm[:1]) == (0,))

    def contents(self, objs) -> set:
        out: set = set()
        for o in objs:
            if o == TOP:
                out.add(TOP)
            elif o[0] not in ("F", "STR"):
                out |= self.held(o)
        return out

    def add(self, node, objs) -> None:
        if node == TOP:
            new = {o for o in objs if o != TOP} - self.escaped
            if new:
                self.escaped |= new
                self.changed = True
            return
        if node[0] in ("F", "STR"):
            return  # code and literals: nothing the program can store into
        s = self.pt.get(node)
        if s is None:
            s = self.pt[node] = set()
        before = len(s)
        s |= objs
        if len(s) != before:
            self.changed = True

    def call_values(self, fn: str, c) -> set:
        """What an indirect call's function pointer may hold."""
        if c.op == "c.call.indirect":
            node = self.u.node(fn, c.rd[0])
            return {node} if node[0] == "F" else self.held(node)
        return self.contents(self.deref(fn, c.rd[0]))  # c.call.imember: the member of the base

    def targets(self, fn: str, c) -> tuple | None:
        """The defined functions an indirect call can reach, or None if it may reach unknown
        code: the pointer may hold an unknown value, a function of another unit, or nothing known
        (a data address in the set is not a callable target -- calling one is undefined)."""
        values = self.call_values(fn, c)
        if TOP in values:
            return None
        names = sorted({o[1] for o in values if o[0] == "F"})
        if not names or any(n not in self.u.functions for n in names):
            return None
        return tuple(names)

    # the rules

    def bind(self, fn: str, callee: str, actuals, results) -> None:
        params = self.u.params[callee]
        for k, a in enumerate(actuals):
            v = self.val(fn, a)
            if k < len(params):
                self.add(self.u.node(callee, params[k]), v)
            else:  # a variadic extra: the callee reads it through va_arg as an unknown value
                self.add(TOP, v)
        if results:
            ret: set = set()
            for r in self.u.returns[callee]:
                ret |= self.val(callee, r)
            for w in results:
                self.add(self.u.node(fn, w), ret)

    def external(self, fn: str, actuals, results) -> None:
        for a in actuals:
            self.add(TOP, self.val(fn, a))
        for w in results:
            self.add(self.u.node(fn, w), {TOP})

    def claim(self, fn: str, c) -> None:
        op = c.op
        if _is(op, _NO_FLOW_PREFIXES) or op == "c.vladecl":
            return
        if op == "c.load":
            got = self.contents(self.deref(fn, c.rd[0]))
            for w in c.wr:
                self.add(self.u.node(fn, w), got)
            return
        if op == "c.store":
            value = self.val(fn, c.rd[-1]) if len(c.rd) > 1 else set()
            for o in self.deref(fn, c.rd[0]):
                self.add(o, value)
            return
        if op == "c.addrof":
            got = self.address(fn, c.rd[0])
            for w in c.wr:
                self.add(self.u.node(fn, w), got)
            return
        if _is(op, _ATOMIC_PREFIXES):
            objs: set = set()
            vals: set = set()
            for r in c.rd:
                objs |= self.deref(fn, r)
                vals |= self.val(fn, r)
            for o in objs:
                self.add(o, vals)
            got = self.contents(objs)
            for w in c.wr:
                self.add(self.u.node(fn, w), got)
            return
        if _is(op, _DIRECT_PREFIXES):
            callee = op.split(":", 1)[1]
            if callee in self.u.functions:
                self.bind(fn, callee, c.rd, c.wr)
            else:
                self.external(fn, c.rd, c.wr)
            return
        if _is_indirect(op):
            # Monotone in what the pointer holds, so the fixpoint does not depend on the order the
            # claims are visited in (the two rails order sibling expressions differently): bind every
            # known target; go external as soon as the pointer may hold something unknown; an empty
            # set binds nothing yet (a call through a pointer that holds nothing is undefined).
            actuals = c.rd[1:]
            self.val(fn, c.rd[0])  # the pointer / dispatch base is used (address-taken bookkeeping)
            values = self.call_values(fn, c)
            names = {o[1] for o in values if o[0] == "F"}
            for callee in sorted(n for n in names if n in self.u.functions):
                self.bind(fn, callee, actuals, c.wr)
            if TOP in values or any(n not in self.u.functions for n in names):
                self.external(fn, actuals, c.wr)
            return
        if op.startswith("c.call.vaarg"):  # the C twin spells it `c.call.vaarg:<type>`
            for w in c.wr:
                self.add(self.u.node(fn, w), {TOP})
            return
        if _is(op, _EXTERNAL_PREFIXES):
            self.external(fn, c.rd, c.wr)
            return
        if _is(op, _LIBRARY_PREFIXES):
            # A library routine keeps no pointer and returns only what it was given (memcpy-like
            # copying between its operands is folded in, so a pointer it moves stays visible).
            vals: set = set()
            for r in c.rd:
                vals |= self.val(fn, r)
            moved = self.contents(vals)
            for o in vals:
                self.add(o, moved)
            for w in c.wr:
                got = set(vals)
                if op in _ALLOCATORS:
                    got.add(("H", fn))  # fresh memory: this function's heap object
                elif self.u.kind(fn, w) in ("pointer", "unknown"):
                    got.add(TOP)  # a pointer a library routine made: memory no one declared
                self.add(self.u.node(fn, w), got)
            return
        if _is_pointer_cast(op):
            got = self.val(fn, c.rd[0]) if c.rd else set()
            if not c.rd or self.forged(fn, c.rd[0]):
                got = got | {TOP}
            for w in c.wr:
                self.add(self.u.node(fn, w), got)
            return
        vals = set()  # every other op: a value computed from its operands
        for r in c.rd:
            vals |= self.val(fn, r)
        for w in c.wr:
            self.add(self.u.node(fn, w), vals)

    def solve(self) -> None:
        while True:
            self.changed = False
            for fn in self.u.functions:
                if self.u.callers_unknown(fn, self.address_taken):
                    for rid in self.u.params[fn]:
                        # an unknown caller passes unknown pointers; a scalar can only become one
                        # through a cast, which `forged` already makes unknown
                        if self.u.kind(fn, rid) != "scalar":
                            self.add(self.u.node(fn, rid), {TOP})
            for fn, lf in self.u.functions.items():
                for c in lf.claims:
                    self.claim(fn, c)
                for r in self.u.returns[fn] | self.u.conditions[fn]:
                    self.val(fn, r)  # a returned function value is address-taken
            if not self.changed:
                return


# --- the answers -----------------------------------------------------------------------------


def _closure(solver: _Solver, roots) -> set:
    seen: set = set()
    work = [o for o in roots if o != TOP]
    while work:
        o = work.pop()
        if o in seen or o[0] in ("F", "STR"):
            continue
        seen.add(o)
        work.extend(x for x in solver.pt.get(o, ()) if x != TOP and x not in seen)
    return seen


def _device(lf, c) -> bool:
    """Whether claim `c` of function `lf` touches a device: it is MMIO-domain, or it is a load or
    store whose base resource is. The base decides, not the claim's spelling -- this lowering marks
    every such access, but the C twin lowers `p[i]` through a `volatile T *` as an ordinary load,
    and one predicate over the resource keeps the two rails' answers the same."""
    if c.domain == Domain.MMIO:
        return True
    if c.op in ("c.load", "c.store") and c.rd:
        res = lf.resources.get(c.rd[0])
        return res is not None and res.domain == Domain.MMIO
    return False


def _own_access(solver: _Solver, u: _Unit, fn: str) -> tuple[set, set]:
    """The objects `fn`'s own claims, conditions and returns read and write."""
    reads: set = set()
    writes: set = set()
    lf = u.functions[fn]

    def operand(rid, into):
        node = u.node(fn, rid)
        if node[0] in ("G", "S", "L"):
            into.add(node)

    for c in lf.claims:
        op = c.op
        for r in c.rd:
            operand(r, reads)
        for w in c.wr:
            operand(w, writes)
        if _device(lf, c):  # a device access: an observable effect on unnamed state
            reads.add(TOP)
            writes.add(TOP)
        if op == "c.load":
            reads |= solver.deref(fn, c.rd[0])
        elif op == "c.store":
            writes |= solver.deref(fn, c.rd[0])
        elif _is(op, _ATOMIC_PREFIXES):
            objs: set = set()
            for r in c.rd:
                objs |= solver.deref(fn, r)
            reads |= objs
            writes |= objs
        elif _is(op, _LIBRARY_PREFIXES):
            objs = set()
            for r in c.rd:
                objs |= solver.val(fn, r)
            reads |= objs
            writes |= objs
        elif _is(op, _EXTERNAL_PREFIXES) or (
            _is(op, _DIRECT_PREFIXES) and op.split(":", 1)[1] not in u.functions
        ):
            reads.add(TOP)
            writes.add(TOP)
        elif _is_indirect(op):
            if op.startswith("c.call.imember"):
                reads |= solver.deref(fn, c.rd[0])  # the member is loaded from the base
            if solver.targets(fn, c) is None:
                reads.add(TOP)
                writes.add(TOP)
        elif op.startswith("c.call.vaarg"):
            reads.add(TOP)
    for r in u.returns[fn] | u.conditions[fn]:
        operand(r, reads)
    return reads, writes


def analyze(lowered) -> EscapeResult:
    """Run the analysis over a lowered unit (see the module docstring)."""
    u = _Unit(lowered)
    result = EscapeResult()
    result.truncated = any(
        len(c.rd) > CLAIM_MAX_RD for lf in u.functions.values() for c in lf.claims
    )
    if result.truncated:
        return _refused(u, result)
    solver = _Solver(u)
    solver.solve()

    # the call graph, direct and narrowed
    for fn, lf in u.functions.items():
        callees: set = set()
        for c in lf.claims:
            if _is(c.op, _DIRECT_PREFIXES):
                callee = c.op.split(":", 1)[1]
                if callee in u.functions:
                    callees.add(callee)
            elif _is_indirect(c.op):
                found = solver.targets(fn, c)
                result.sites.append(IndirectSite(fn, c.id, c.op, found))
                if found is not None:
                    callees |= set(found)
        result.edges[fn] = frozenset(callees)

    # escape: what unknown code, a global or a static can reach, and what a function returns
    roots = set(solver.escaped)
    for node, objs in solver.pt.items():
        if node[0] in ("G", "S"):
            roots |= objs
    escaped = _closure(solver, roots)
    lent_roots: set = set()
    for fn, lf in u.functions.items():
        for c in lf.claims:
            if c.op.startswith("c.call") or _is(c.op, _EXTERNAL_PREFIXES):
                for a in c.rd[1:] if _is_indirect(c.op) else c.rd:
                    lent_roots |= solver.val(fn, a)
    lent = _closure(solver, lent_roots)
    frame_escaping: dict = {}
    for fn in u.functions:
        returned: set = set()
        for r in u.returns[fn]:
            returned |= solver.val(fn, r)
        frame_escaping[fn] = escaped | _closure(solver, returned)

    def verdict(fn, node):
        if node in frame_escaping[fn]:
            return "escaping"
        return "lent" if node in lent else "nonescaping"

    severity = {v: k for k, v in enumerate(VERDICTS)}
    for fn, lf in u.functions.items():
        names = u.local_names[fn]
        mem = u.memory[fn]
        per: dict = {}
        for rid, lname in names.items():
            if rid in mem:  # one name in two scopes: the more severe verdict stands for both
                got = verdict(fn, ("L", fn, rid))
                if lname not in per or severity[got] > severity[per[lname]]:
                    per[lname] = got
        result.objects[fn] = per
        cands = {
            c.rd[0]
            for c in lf.claims
            if c.bounds_provenance == "declared_extent" and c.rd and c.rd[0] in names
        }
        result.candidates[fn] = {names[rid]: verdict(fn, ("L", fn, rid)) for rid in sorted(cands)}

    # footprints: a function's own accesses and everything it can call, named
    own = {fn: _own_access(solver, u, fn) for fn in u.functions}
    for fn in u.functions:
        reach = {fn}
        work = [fn]
        while work:
            h = work.pop()
            for g in result.edges.get(h, ()):
                if g not in reach:
                    reach.add(g)
                    work.append(g)
        reads: set = set()
        writes: set = set()
        for h in reach:
            reads |= own[h][0]
            writes |= own[h][1]
        result.footprints[fn] = Footprint(
            _named(reads, reach, frame_escaping), _named(writes, reach, frame_escaping)
        )
    return result


def _refused(u: _Unit, result: EscapeResult) -> EscapeResult:
    """A unit the C twin cannot hold whole (a call with more operands than a claim carries): every
    footprint is `*` (nothing commutes), no local is proved private and no indirect call is
    narrowed -- the conservative answer both rails can give without the dropped operands."""
    everything = Footprint(frozenset({UNKNOWN}), frozenset({UNKNOWN}))
    for fn, lf in u.functions.items():
        names = u.local_names[fn]
        result.footprints[fn] = everything
        result.objects[fn] = {}
        cands = {
            c.rd[0]
            for c in lf.claims
            if c.bounds_provenance == "declared_extent" and c.rd and c.rd[0] in names
        }
        result.candidates[fn] = {names[rid]: "escaping" for rid in sorted(cands)}
        result.sites.extend(
            IndirectSite(fn, c.id, c.op, None) for c in lf.claims if _is_indirect(c.op)
        )
        result.edges[fn] = frozenset(
            c.op.split(":", 1)[1]
            for c in lf.claims
            if _is(c.op, _DIRECT_PREFIXES) and c.op.split(":", 1)[1] in u.functions
        )
    return result


def _named(objs: set, reach: set, frame_escaping: dict) -> frozenset:
    """A footprint's objects by name, for a function whose calls reach `reach`: a global by its
    name, a static by `function.name`, unknown memory as `*`; a local of one of those activations
    is private unless it can be reached from outside (then it is `*`, as is any other local);
    code and string literals are not memory the program writes."""
    out = set()
    for o in objs:
        if o == TOP:
            out.add(UNKNOWN)
        elif o[0] in ("G", "S"):
            out.add(o[1])
        elif o[0] in ("L", "H"):  # a local or heap object, owned by the function o[1]
            if o[1] not in reach or o in frame_escaping[o[1]]:
                out.add(UNKNOWN)
    return frozenset(out)


def effects_report(lowered, result: EscapeResult) -> str:
    """The footprint report both rails print (`bcir-cc --emit-effects`): one line per function in
    unit order, then the commute matrix over every pair."""
    fns = list(lowered.functions)
    out = []
    for n in fns:
        fp = result.footprints[n]
        reads = ",".join(sorted(fp.reads)) or "-"
        writes = ",".join(sorted(fp.writes)) or "-"
        out.append(f"fn={n} reads={reads} writes={writes}")
    for i, a in enumerate(fns):
        for b in fns[i + 1 :]:
            out.append(f"commute {a} {b} = {1 if result.commute(a, b) else 0}")
    return "\n".join(out) + "\n"


def escape_report(lowered, result: EscapeResult) -> str:
    """The escape and narrowing report both rails print (`bcir-cc --emit-escape`): each function's
    named automatic local memory objects by verdict, then its indirect calls' target sets (sorted,
    so the order in which a rail evaluates sibling expressions cannot show)."""
    if result.truncated:  # refused: the C twin cannot see every operand (see `_refused`)
        return "truncated=1\n" + "".join(f"fn={fn} refused\n" for fn in lowered.functions)
    out = []
    for fn in lowered.functions:
        per = result.objects.get(fn, {})
        cols = []
        for v in VERDICTS:
            names = sorted(n for n, got in per.items() if got == v)
            cols.append(f"{v}={','.join(names) or '-'}")
        sites = sorted(s.report() for s in result.sites if s.function == fn)
        out.append(f"fn={fn} {' '.join(cols)} icalls={';'.join(sites) or '-'}")
    return "\n".join(out) + "\n"
