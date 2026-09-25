"""The G9 fixtures and grader (S5-A): the declared alias facts carried the rest of the way to LLVM,
held fact by fact to the declaration they come from.

A claim DECLARES its read and write RIDs, its hazard and its volatility, and every resource declares
its element size; nothing downstream has to infer them. The elementwise claim `C = A op B` reaches
LLVM through the textual emitter (`lower.llvm`, which the AOT, JIT and WASM paths share) and through
five C emitters clang lowers -- the kernel (`c_kernel.emit_kernel_c`, which the library facade and
the native rows use), the Q-fixed kernel, the gather form the native rows compare against, the
hot-shape specialist, and the published ABI header. One function, `measure`, grades what each
carries; the tests, the harness (`tools/perf/gemplus_baseline.py --group alias`) and
`tools/perf/check_alias.py` all call it:

    alias.noalias.mismatch      pointer parameters whose no-alias assertion -- LLVM `noalias`, C
                                `restrict`, on every emitter -- is not the RID partition's:
                                asserted on a position whose resource another position names (a
                                false fact) or missing on one whose resource no other position
                                names (a dropped one)
    alias.scope.mismatch        LLVM memory accesses whose `!alias.scope` / `!noalias` scopes do
                                not encode the partition: one scope per resource, all in one
                                domain, the access's own resource in `!alias.scope` and every
                                other one in `!noalias`
    alias.tbaa.mismatch         LLVM memory accesses without the TBAA access tag of the declared
                                element type -- the tag clang gives the same C type
    alias.volatile.mismatch     LLVM memory accesses, and C pointer parameters, whose volatility
                                is not the claim's
    alias.fence.mismatch        kernels whose fences are not the hazard's: a sequentially
                                consistent fence first and last for a barriered claim, none for a
                                unique one
    alias.refusal.accepted      (module, emitter) pairs the elementwise subset must refuse and
                                lowered instead: an atomic or unknown hazard (the subset emits no
                                atomic element operation), an operand resource declaring an
                                element size the kernel does not address, or not declared at all
    alias.differential.silent   (module, module', emitter) triples differing in exactly one
                                declared fact -- the RID partition, volatility, the hazard, an
                                operand's element size, the element type -- whose emitted facts
                                are identical
    alias.harness.unaliased     self-checks (the LLVM AOT/JIT harness, the C and Q-fixed
                                self-checks, the WASM node harness) that do not bind one buffer
                                per declared resource
    alias.r12.rejected          (kernel, backend) pairs the emitter produced and its own lowering
                                law (R12) rejects
    alias.r12.forgery.accepted  forged kernels -- one fact of an honest kernel dropped, added or
                                contradicted (`LL_FORGERIES`, `C_FORGERIES`) -- to which R12
                                raises no new finding naming the forged fact (`FORGERY_FINDS`: a
                                false fact must read as one, a dropped fact as one)

Where a coherent LLVM toolset (clang, llvm-link, opt) is present, `measure(llvm=True)` adds the
rows LLVM itself judges -- the consumer the facts are for, and the one judge that shares no code
with the emitters:

    alias.llvm.false_noalias        access pairs (one at least a store) through two positions that
                                    name one resource, which LLVM's default alias analysis proves
                                    NoAlias: a fact the declaration contradicts
    alias.llvm.scope_facts.missing  access pairs (one at least a store) through positions naming
                                    distinct resources, which the scoped-noalias analysis ALONE
                                    does not prove NoAlias: scopes absent, or present and inert
    alias.backends.disagree         (kernel, fact family) pairs where clang's IR for the C kernel
                                    carries other facts than the LLVM kernel: the noalias
                                    parameters, the TBAA types, the volatility, the fences
    alias.llvm.caller_memory        accesses to a C caller's own `long` data, around the kernel
                                    LLVM inlined into it, that contradict the declared facts:
                                    a store or reload the kernel's accesses cannot touch kept
                                    across a unique kernel (its TBAA proves them disjoint), or
                                    one dropped across a barriered kernel (its fences order the
                                    caller's memory too)
    alias.exec.failed               (kernel, runner) pairs whose self-check fails when compiled
                                    and run with the declared aliasing bound: LLVM AOT and JIT,
                                    the C and Q-fixed kernels, WASM under node -- each runner
                                    whose tools the host has (the others are recorded, not run)

The corpus: every RID partition of the three operand positions (and two relabellings, so no check
can key on RID values or their order), each operation, both element types, the planner's width
and three forced ones, and the three lawful contracts (unique; barriered; barriered and volatile)
-- plus the modules the subset must refuse. Nothing here needs the S5-A mechanisms: the declared
facts are derived here, the emitted ones parsed here, the bindings read from the harness text,
all through entry points the parent tree has (where the parent spells an entry point without the
plan -- the header, the node harness -- it can only state one contract for every claim, and is
graded on that). On the parent every mechanism is absent, so every row it owns fails (RED,
docs/security/laws.md L1, L25).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

ROWS = (
    "alias.noalias.mismatch",
    "alias.scope.mismatch",
    "alias.tbaa.mismatch",
    "alias.volatile.mismatch",
    "alias.fence.mismatch",
    "alias.refusal.accepted",
    "alias.differential.silent",
    "alias.harness.unaliased",
    "alias.r12.rejected",
    "alias.r12.forgery.accepted",
)
LLVM_ROWS = (
    "alias.llvm.false_noalias",
    "alias.llvm.scope_facts.missing",
    "alias.backends.disagree",
    "alias.llvm.caller_memory",
    "alias.exec.failed",
)

POSITIONS = ("A", "B", "C")
# The RIDs behind (A, B, C) = (rd[0], rd[1], wr[0]): the five partitions of three positions, and
# two relabellings of them (a permuted all-distinct one, a shared class whose RID sorts last).
PARTITIONS = ((1, 2, 3), (1, 2, 1), (1, 1, 2), (1, 2, 2), (1, 1, 1), (3, 1, 2), (7, 7, 4))
OPS = ("ADD", "SUB", "MUL")
ELEMS = ("f32", "i32")
WIDTHS = (None, 1, 4, 16)  # None: the planner's width
CONTRACTS = (("unique", False), ("barriered", False), ("barriered", True))
EXEC_CONTRACTS = (("unique", False), ("barriered", True))
COUNT = 64
FRESH = 99  # a RID no partition uses
# What the subset must refuse, as (hazard, volatile, {position: element size}, positions whose
# resource the module does not declare).
REFUSALS = (
    ("atomic", False, {}, ()),
    ("atomic", True, {}, ()),
    ("wild", False, {}, ()),
    ("unique", False, {0: 8}, ()),
    ("unique", False, {2: 2}, ()),
    ("barriered", True, {1: 8}, ()),
    ("unique", False, {}, (1,)),
    ("barriered", True, {}, (2,)),
)
# The element type each lowering contract declares: (LLVM type, TBAA scalar type, C type).
ELEM_TYPES = {"f32": ("float", "float", "float"), "i32": ("i32", "int", "int32_t")}
TBAA_ROOT = "Simple C/C++ TBAA"
TBAA_CHAR = "omnipotent char"
C_FENCE = "atomic_thread_fence(memory_order_seq_cst)"


# --- the corpus -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    rids: tuple  # the RIDs behind (A, B, C)
    op: str = "ADD"
    elem: str = "f32"
    width: int | None = None
    hazard: str = "unique"
    volatile: bool = False
    sizes: tuple = ()  # ((position, element size), ...) overriding the declared 4
    undeclared: tuple = ()  # positions whose resource the module does not declare


def module_of(case: Case):
    """The one-claim module a case declares: a resource per RID (element size 4 unless the case
    overrides a position's), and `C = A op B` over `COUNT` elements."""
    from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass

    sizes = {case.rids[p]: s for p, s in case.sizes}
    missing = {case.rids[p] for p in case.undeclared}
    m = Module(name="alias")
    for rid in sorted(set(case.rids) - missing):
        m.add_resource(Resource(rid=rid, shape=(COUNT,), elem_bytes=sizes.get(rid, 4)))
    claim = Claim(
        id=1000,
        opcode=Opcode[case.op],
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=COUNT,
        rd=(case.rids[0], case.rids[1]),
        wr=(case.rids[2],),
        op=f"vector.{case.op.lower()}",
        hazard=case.hazard,
        volatile=case.volatile,
    )
    m.add_phase(Phase(phase_id=0, claims=[claim]))
    return m


_PLANS: dict = {}


def planned(case: Case):
    """The module and the planner's result for it (cached: the plan does not depend on the
    lowering's width or element type)."""
    from bcir.kbcir import TARGETS, optimize
    from bcir.kbcir.cost import Theta
    from bcir.kbcir.weights import PERF

    key = (case.rids, case.op, case.hazard, case.volatile, case.sizes, case.undeclared)
    if key not in _PLANS:
        m = module_of(case)
        _PLANS[key] = (m, optimize(m, TARGETS["x86_avx2"], Theta.cool(), PERF))
    return _PLANS[key]


def valid_cases() -> list[Case]:
    return [
        Case(rids, op, elem, width, hazard, volatile)
        for rids in PARTITIONS
        for op in OPS
        for elem in ELEMS
        for width in WIDTHS
        for hazard, volatile in CONTRACTS
    ]


def refusal_cases() -> list[Case]:
    return [
        Case(rids, "ADD", elem, None, hazard, volatile, tuple(sorted(sizes.items())), undeclared)
        for rids in PARTITIONS
        for elem in ELEMS
        for hazard, volatile, sizes, undeclared in REFUSALS
    ]


def core_cases() -> list[Case]:
    """The cases LLVM and R12's forgeries judge: each partition, both element types, the
    planner's width and scalar, every lawful contract."""
    return [
        Case(rids, "ADD", elem, width, hazard, volatile)
        for rids in PARTITIONS
        for elem in ELEMS
        for width in (None, 1)
        for hazard, volatile in CONTRACTS
    ]


def exec_cases() -> list[Case]:
    """Each partition under each executed contract, the operation rotating so every one runs
    beside several partitions (a self-check once ran only `+`)."""
    return [
        Case(rids, OPS[(i + j) % len(OPS)], "f32", None, hazard, volatile)
        for i, rids in enumerate(PARTITIONS)
        for j, (hazard, volatile) in enumerate(EXEC_CONTRACTS)
    ]


# --- what the module declares (this module's own derivation) ----------------------------------


@dataclass(frozen=True)
class Declared:
    rids: tuple
    exclusive: tuple  # per position: no other position names its resource
    volatile: bool
    fence: str | None  # the fence ordering a barriered claim needs; None for a unique one
    ll_type: str
    tbaa: str
    ctype: str


def declared(case: Case) -> Declared:
    rids = case.rids
    exclusive = tuple(sum(r == other for other in rids) == 1 for r in rids)
    ll_type, tbaa, ctype = ELEM_TYPES[case.elem]
    fence = "seq_cst" if case.hazard == "barriered" else None
    return Declared(rids, exclusive, case.volatile, fence, ll_type, tbaa, ctype)


def canonical_partition(rids) -> tuple:
    first: dict = {}
    return tuple(first.setdefault(r, len(first)) for r in rids)


# --- what an LLVM kernel carries (parsed here) -------------------------------------------------

_LL_DEFINE = re.compile(r"^define\b[^@]*@([\w.$]+)\((.*)\)[^{]*\{\s*$")
_LL_GEP = re.compile(r"^\s*(%[\w.]+)\s*=\s*getelementptr\b.*?,\s*ptr\s+(%[\w.]+)\s*,")
# Every `load` / `store`, in any spelling LLVM's parser accepts: the alignment is optional (the
# parser fills it in), an atomic access names its ordering after the pointer.
_LL_ACCESS = re.compile(r"^\s*(?:%[\w.]+\s*=\s*)?(load|store)\b(.*)$")
_LL_OPERANDS = re.compile(r"^(.*?),\s*ptr\s+(%[\w.]+)(?=[\s,]|$)(.*)$")
_LL_MD = re.compile(r"^!(\d+)\s*=\s*(distinct\s+)?!\{(.*)\}\s*$")
_LL_ATTACH = re.compile(r",\s*!([\w.]+)\s+!(\d+)")
_LL_FENCE = re.compile(r'^\s*fence\s+(?:syncscope\("([^"]*)"\)\s+)?(\w+)\s*(?:,.*)?$')


@dataclass(frozen=True)
class Access:
    position: int | None  # 0/1/2 for A/B/C, None when the pointer is none of them
    kind: str
    volatile: bool
    scope: int | None
    noalias: int | None
    tbaa: int | None


@dataclass
class LLKernel:
    noalias: frozenset
    accesses: list
    metadata: dict  # id -> (distinct, operand tokens)
    entry_fence: str | None
    exit_fences: tuple  # the fence (or None) before each `ret`
    fences: tuple  # every fence's ordering, in order
    functions: int


def _split_params(params: str) -> list[str]:
    """A parameter list split at its top-level commas (clang writes `captures(address, ...)`)."""
    out, depth, cur = [], 0, []
    for ch in params:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def _md_tokens(body: str) -> list[str]:
    return [tok.strip() for tok in body.split(",")] if body.strip() else []


def _ref(tok: str) -> int | None:
    m = re.fullmatch(r"!(\d+)", tok)
    return int(m.group(1)) if m else None


def _string(tok: str) -> str | None:
    m = re.fullmatch(r'!"(.*)"', tok)
    return m.group(1) if m else None


def _ll_ordering(ins: str) -> str | None:
    """A fence's ordering -- scope-qualified unless it is the system scope (no `syncscope`, or
    `syncscope("")`): a narrowed fence is not the barrier -- or None when `ins` is no fence."""
    if ins.split()[:1] != ["fence"]:
        return None
    m = _LL_FENCE.match(ins)
    if m is None:
        return "unreadable"
    return m.group(2) if not m.group(1) else f"{m.group(2)}@{m.group(1)}"


def parse_ll(text: str, fn: str | None = None) -> LLKernel:
    """The facts one function of an LLVM module carries (the first, or the one named `fn`), the
    module's metadata kept. Total over any text: what cannot be read is absent, and an absent
    fact fails every row that asks for it."""
    noalias: set = set()
    accesses: list = []
    metadata: dict = {}
    pointers: dict = {}
    functions = 0
    body: list = []
    inside = False
    for raw in text.splitlines():
        md = _LL_MD.match(raw.strip())
        if md:
            metadata[int(md.group(1))] = (bool(md.group(2)), _md_tokens(md.group(3)))
            continue
        line = raw.split(";", 1)[0].rstrip()
        d = _LL_DEFINE.match(line)
        if d:
            functions += 1
            inside = d.group(1) == fn if fn is not None else functions == 1
            if inside:
                for param in _split_params(d.group(2)):
                    toks = param.split()
                    if toks and toks[-1] in ("%A", "%B", "%C") and "noalias" in toks:
                        noalias.add(POSITIONS.index(toks[-1][1:]))
            continue
        if not inside:
            continue
        if line.strip() == "}":
            inside = False
            continue
        body.append(line)
        g = _LL_GEP.match(line)
        if g:
            if g.group(2) in ("%A", "%B", "%C"):
                pointers[g.group(1)] = POSITIONS.index(g.group(2)[1:])
            elif g.group(2) in pointers:
                pointers[g.group(1)] = pointers[g.group(2)]
        a = _LL_ACCESS.match(line)
        if a:
            o = _LL_OPERANDS.match(a.group(2))
            attach = {k: int(v) for k, v in _LL_ATTACH.findall(o.group(3))} if o else {}
            ptr = o.group(2) if o else None
            position = pointers.get(ptr)
            if position is None and ptr in ("%A", "%B", "%C"):
                position = POSITIONS.index(ptr[1:])
            accesses.append(
                Access(
                    position,
                    a.group(1),
                    bool(o) and "volatile" in o.group(1).split(),
                    attach.get("alias.scope"),
                    attach.get("noalias"),
                    attach.get("tbaa"),
                )
            )
    instrs = [ln.strip() for ln in body if ln.strip() and not ln.strip().endswith(":")]
    orderings = [_ll_ordering(ins) for ins in instrs]
    exits = tuple(
        orderings[i - 1] if i else None for i, ln in enumerate(instrs) if ln.startswith("ret")
    )
    return LLKernel(
        frozenset(noalias),
        accesses,
        metadata,
        orderings[0] if orderings else None,
        exits,
        tuple(o for o in orderings if o is not None),
        functions,
    )


def _list(k: LLKernel, node: int | None) -> frozenset | None:
    """The node ids a metadata list names, or None when it is not a list of references."""
    if node is None or node not in k.metadata:
        return None
    refs = [_ref(tok) for tok in k.metadata[node][1]]
    return None if any(r is None for r in refs) else frozenset(refs)


def _scope_domain(k: LLKernel, s: int) -> int | None:
    """A well-formed scope: `distinct !{!s, !domain, ...}` over `distinct !{!domain, ...}`."""
    entry = k.metadata.get(s)
    if entry is None or not entry[0] or len(entry[1]) < 2 or _ref(entry[1][0]) != s:
        return None
    d = _ref(entry[1][1])
    dom = k.metadata.get(d) if d is not None else None
    if dom is None or not dom[0] or not dom[1] or _ref(dom[1][0]) != d:
        return None
    return d


def scope_mismatches(k: LLKernel, decl: Declared) -> int:
    """Accesses whose scopes do not encode the RID partition (module docstring)."""
    own: dict = {}  # rid -> the scope set its first access carries
    for acc in k.accesses:
        s = _list(k, acc.scope) if acc.position is not None else None
        if s is not None:
            own.setdefault(decl.rids[acc.position], s)
    scopes = {rid: next(iter(s)) for rid, s in own.items() if len(s) == 1}
    domains = {_scope_domain(k, s) for s in scopes.values()}
    healthy = (
        len(scopes) == len(set(decl.rids))
        and len(set(scopes.values())) == len(scopes)
        and len(domains) == 1
        and None not in domains
    )
    bad = 0
    for acc in k.accesses:
        if acc.position is None or not healthy:
            bad += 1
            continue
        rid = decl.rids[acc.position]
        want_noalias = frozenset(s for r, s in scopes.items() if r != rid)
        got_noalias = _list(k, acc.noalias) if acc.noalias is not None else frozenset()
        bad += not (_list(k, acc.scope) == frozenset({scopes[rid]}) and got_noalias == want_noalias)
    return bad


def tbaa_name(k: LLKernel, tag: int | None) -> str | None:
    """The scalar type an access tag names under clang's C/C++ root, or None."""
    entry = k.metadata.get(tag) if tag is not None else None
    if entry is None or entry[0] or len(entry[1]) != 3 or entry[1][2] != "i64 0":
        return None
    base, access = _ref(entry[1][0]), _ref(entry[1][1])
    if base is None or base != access:
        return None
    scalar = k.metadata.get(base)
    if scalar is None or scalar[0] or len(scalar[1]) != 3 or scalar[1][2] != "i64 0":
        return None
    char = k.metadata.get(_ref(scalar[1][1]))
    if char is None or len(char[1]) != 3 or _string(char[1][0]) != TBAA_CHAR:
        return None
    root = k.metadata.get(_ref(char[1][1]))
    if root is None or len(root[1]) != 1 or _string(root[1][0]) != TBAA_ROOT:
        return None
    return _string(scalar[1][0])


def ll_fence_ok(k: LLKernel, decl: Declared) -> bool:
    if decl.fence is None:
        return not k.fences
    return (
        k.entry_fence == decl.fence
        and bool(k.exit_fences)
        and all(f == decl.fence for f in k.exit_fences)
        and len(k.fences) == 1 + len(k.exit_fences)
    )


def ll_facts(k: LLKernel) -> tuple:
    """The kernel's facts with the metadata numbering factored out: the differential compares
    these, so two kernels differ only where a fact does."""
    label: dict = {}
    for acc in sorted(k.accesses, key=lambda a: (a.position is None, a.position or 0)):
        for s in sorted(_list(k, acc.scope) or ()):
            label.setdefault(s, len(label))

    def names(node):
        refs = _list(k, node)
        return None if refs is None else tuple(sorted(label.get(r, -1) for r in refs))

    accesses = sorted(
        (
            -1 if a.position is None else a.position,
            a.kind,
            a.volatile,
            str(names(a.scope)),
            str(names(a.noalias)),
            str(tbaa_name(k, a.tbaa)),
        )
        for a in k.accesses
    )
    return (k.noalias, tuple(accesses), k.entry_fence, k.exit_fences, k.fences)


def side_facts(k: LLKernel) -> tuple:
    """The four fact families the two backends are compared on: the noalias parameters, the
    accesses' volatility and TBAA types, the fences."""
    return (
        k.noalias,
        frozenset(a.volatile for a in k.accesses),
        frozenset(str(tbaa_name(k, a.tbaa)) for a in k.accesses),
        tuple(sorted(k.fences)),
    )


# --- what a C kernel or prototype carries (parsed here) ----------------------------------------

_C_SIG = re.compile(r"\bvoid\s+(\w+)\s*\(([^)]*)\)\s*(\{|;)", re.S)


@dataclass
class CKernel:
    restrict: frozenset
    volatile: frozenset
    types: tuple  # the element type token of each pointer
    entry_fence: bool
    exit_fence: bool
    fences: int
    found: bool


_C_TYPES = ("float", "int32_t", "q_lane_t")


def parse_c(text: str, fn_name: str) -> CKernel:
    """The facts the named C function (a definition or a prototype) carries."""
    for m in _C_SIG.finditer(text):
        if m.group(1) != fn_name:
            continue
        restrict, volatile, types = set(), set(), []
        for i, param in enumerate(m.group(2).replace("*", " * ").split(",")[:3]):
            toks = param.split()
            types.append(next((t for t in toks if t in _C_TYPES), ""))
            if "restrict" in toks:
                restrict.add(i)
            if "volatile" in toks:
                volatile.add(i)
        entry = exit_ = False
        fences = 0
        if m.group(3) == "{":
            depth, j = 1, m.end()
            while j < len(text) and depth:
                depth += {"{": 1, "}": -1}.get(text[j], 0)
                j += 1
            stmts = [
                ln.strip()
                for ln in text[m.end() : j - 1].splitlines()
                if ln.strip() and not ln.strip().startswith(("/*", "//"))
            ]
            fences = sum(C_FENCE in s for s in stmts)
            entry = bool(stmts) and stmts[0].startswith(C_FENCE)
            exit_ = bool(stmts) and stmts[-1].startswith(C_FENCE)
        return CKernel(
            frozenset(restrict), frozenset(volatile), tuple(types), entry, exit_, fences, True
        )
    return CKernel(frozenset(), frozenset(), (), False, False, 0, False)


def c_fence_ok(k: CKernel, decl: Declared) -> bool:
    if decl.fence is None:
        return k.fences == 0
    return k.entry_fence and k.exit_fence and k.fences == 2


def c_facts(k: CKernel) -> tuple:
    return (k.restrict, k.volatile, k.types, k.entry_fence, k.exit_fence, k.fences, k.found)


_CALL = r"\b{fn}\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*,\s*\w+\s*\)\s*;"


def bound_like(bufs, decl: Declared) -> bool:
    """One buffer per declared resource: two positions share a buffer iff they share a RID."""
    return all(
        (bufs[p] == bufs[q]) == (decl.rids[p] == decl.rids[q])
        for p in range(3)
        for q in range(p + 1, 3)
    )


def bound_like_declared(harness: str, fn_name: str, decl: Declared) -> bool:
    """Every call of the kernel in a harness passes one buffer per declared resource."""
    calls = re.findall(_CALL.format(fn=re.escape(fn_name)), harness)
    return bool(calls) and all(bound_like(bufs, decl) for bufs in calls)


# --- emitting, through the entry points both trees have ----------------------------------------


class Refused(Exception):
    pass


def _refusing(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except NotImplementedError as exc:
        raise Refused(str(exc)) from exc


def emit_ll(case: Case, fn_name: str = "bcir_kernel") -> str:
    from bcir.lower.llvm import emit_kernel_ll

    m, r = planned(case)
    return _refusing(emit_kernel_ll, m, r, fn_name, case.elem, width_override=case.width)


def emit_c(case: Case, fn_name: str = "bcir_kernel") -> str:
    from bcir.lower.c_kernel import emit_kernel_c

    m, r = planned(case)
    return _refusing(emit_kernel_c, m, r, fn_name, case.elem, width_override=case.width)


def emit_header(case: Case) -> tuple[str, str]:
    """The published C ABI header. The parent's takes no plan, so it states one contract for
    every claim; the plan is passed where the signature takes it."""
    import inspect

    from bcir.lower.c_kernel import emit_header_c

    m, r = planned(case)
    if "module" in inspect.signature(emit_header_c).parameters:
        return _refusing(emit_header_c, "bcir_kernel", case.elem, module=m, result=r), "bcir_kernel"
    return emit_header_c("bcir_kernel", case.elem), "bcir_kernel"


def emit_gather(case: Case) -> tuple[str, str]:
    from bcir.lower.c_kernel import emit_gather_kernel_c

    m, r = planned(case)
    return _refusing(emit_gather_kernel_c, m, r, "bcir_gather", case.elem), "bcir_gather"


def emit_specialist(case: Case) -> tuple[str, str]:
    """The hot-shape specialist (`synthesize` with a threshold of one: hot on the first call)."""
    from bcir.lower.specialist import ShapeLedger, synthesize

    m, r = planned(case)
    res = _refusing(synthesize, m, r, ShapeLedger(), case.elem, threshold=1)
    return res.kernel_c, res.fn_name


def emit_qfixed(case: Case) -> tuple[str, str]:
    from bcir.lower.c_kernel import emit_qfixed_kernel_c

    m, r = planned(case)
    return _refusing(emit_qfixed_kernel_c, m, r, "bcir_qfixed", 16, 8), "bcir_qfixed"


def emit_c_kernel(case: Case) -> tuple[str, str]:
    return emit_c(case), "bcir_kernel"


# The C emitters: (label, emit, has a body, per element type, reads the resources' declaration).
# The kernel depends on the width; the others are graded once per plan (and the Q-fixed kernel,
# whose lanes are its own representation and read no declared element, once per plan).
C_RAILS = (
    ("c.kernel", emit_c_kernel, True, True, True),
    ("c.gather", emit_gather, True, True, True),
    ("c.specialist", emit_specialist, True, True, True),
    ("c.qfixed", emit_qfixed, True, False, False),
    ("c.header", emit_header, False, True, True),
)


def wasm_binding(case: Case) -> tuple:
    """The buffer the node harness binds each operand to. The parent's harness had no binding:
    three private regions, whatever the claim declared."""
    from bcir.lower import wasm

    if not hasattr(wasm, "harness_binding"):
        return (0, 1, 2)
    m, r = planned(case)
    return tuple(wasm.harness_binding(m, r))


def harnesses(case: Case) -> list[tuple[str, object]]:
    """(label, check(decl) -> bound as declared) for each self-check of the case's plan."""
    from bcir.lower.c_kernel import emit_qfixed_selfcheck_c, emit_selfcheck_c
    from bcir.lower.llvm import emit_harness_c

    m, r = planned(case)
    out = [
        (
            "c.selfcheck",
            lambda d: bound_like_declared(
                emit_selfcheck_c(m, r, "bcir_kernel", case.elem), "bcir_kernel", d
            ),
        )
    ]
    if case.elem == "f32":
        out += [
            (
                "ll.harness",
                lambda d: bound_like_declared(
                    emit_harness_c(m, r, "bcir_kernel"), "bcir_kernel", d
                ),
            ),
            (
                "qfixed.selfcheck",
                lambda d: bound_like_declared(
                    emit_qfixed_selfcheck_c(m, r, "bcir_qfixed"), "bcir_qfixed", d
                ),
            ),
            ("wasm.node", lambda d: bound_like(wasm_binding(case), d)),
        ]
    return out


# --- the grader --------------------------------------------------------------------------------


def _built(build, rows, out: dict[str, float]) -> list:
    """A corpus, or none with every row it feeds failed: a corpus that cannot be built is a
    finding in a named row, never a traceback (L1)."""
    try:
        return build()
    except Exception:  # noqa: BLE001 -- an oracle that cannot build its corpus decided nothing
        for row in rows:
            out[row] += 1
        return []


def _note(seen: dict | None, key: str, value=1) -> None:
    if seen is not None:
        if isinstance(value, set):
            seen.setdefault(key, set()).update(value)
        else:
            seen[key] = seen.get(key, 0) + value


def _once_per_plan(case: Case, per_elem: bool) -> bool:
    return case.width is None and case.op == "ADD" and (per_elem or case.elem == "f32")


def grade_ll(case: Case, decl: Declared, out: dict, seen) -> None:
    from bcir.verify import verify_lowering

    m, r = planned(case)
    try:
        ll = emit_ll(case)
        k = parse_ll(ll)
        out["alias.noalias.mismatch"] += sum(
            (p in k.noalias) != decl.exclusive[p] for p in range(3)
        )
        out["alias.scope.mismatch"] += scope_mismatches(k, decl)
        if not k.accesses or any(a.position is None for a in k.accesses):
            out["alias.scope.mismatch"] += 1  # an access no operand explains, or none at all
        out["alias.tbaa.mismatch"] += sum(tbaa_name(k, a.tbaa) != decl.tbaa for a in k.accesses)
        out["alias.volatile.mismatch"] += sum(a.volatile != decl.volatile for a in k.accesses)
        out["alias.fence.mismatch"] += not ll_fence_ok(k, decl)
        diags = verify_lowering(m, r, ll, case.elem, width_override=case.width)
        out["alias.r12.rejected"] += any(d.law == "R12" for d in diags)
        _note(seen, "ll.kernels")
        _note(seen, "ll.accesses", len(k.accesses))
    except Exception:  # noqa: BLE001 -- a lawful case that did not lower fails what it feeds
        for row in ROWS[:5] + ("alias.r12.rejected",):
            out[row] += 1


def grade_c(case: Case, decl: Declared, out: dict, seen) -> None:
    from bcir.verify import verify_c_lowering

    m, r = planned(case)
    want_vol = frozenset(range(3)) if decl.volatile else frozenset()
    for label, emit, body, per_elem, _sizes in C_RAILS:
        if label != "c.kernel" and not _once_per_plan(case, per_elem):
            continue
        try:
            text, fn = emit(case)
            k = parse_c(text, fn)
            if not k.found:
                raise ValueError(f"{label}: no {fn}")
            out["alias.noalias.mismatch"] += sum(
                (p in k.restrict) != decl.exclusive[p] for p in range(3)
            )
            out["alias.volatile.mismatch"] += len(k.volatile ^ want_vol)
            if body:
                out["alias.fence.mismatch"] += not c_fence_ok(k, decl)
            if label == "c.kernel":
                diags = verify_c_lowering(m, r, text, case.elem, width_override=case.width)
                out["alias.r12.rejected"] += any(d.law == "R12" for d in diags)
            _note(seen, f"{label}.kernels")
        except Exception:  # noqa: BLE001
            out["alias.noalias.mismatch"] += 3
            out["alias.volatile.mismatch"] += 3
            if body:
                out["alias.fence.mismatch"] += 1
            if label == "c.kernel":
                out["alias.r12.rejected"] += 1


def grade_case(case: Case, out: dict[str, float], seen: dict | None = None) -> None:
    """One lawful case on every emitter, and its plan's self-checks."""
    decl = declared(case)
    grade_ll(case, decl, out, seen)
    grade_c(case, decl, out, seen)
    if _once_per_plan(case, True):
        for label, check in harnesses(case):
            try:
                out["alias.harness.unaliased"] += not check(decl)
            except Exception:  # noqa: BLE001
                out["alias.harness.unaliased"] += 1
            _note(seen, f"{label}.harnesses")


def grade_refusals(out: dict[str, float], seen: dict | None = None) -> None:
    rails = [("ll", lambda c: emit_ll(c), True)] + [
        (label, emit, declaration) for label, emit, _body, _per_elem, declaration in C_RAILS
    ]
    for case in _built(refusal_cases, ("alias.refusal.accepted",), out):
        for label, emit, declaration in rails:
            if (case.sizes or case.undeclared) and not declaration:
                continue  # the Q-fixed lanes are their own representation, not the resources'
            try:
                emit(case)
                out["alias.refusal.accepted"] += 1
            except Refused:
                _note(seen, "refused")
            except Exception:  # noqa: BLE001 -- a traceback is not a refusal (L1)
                out["alias.refusal.accepted"] += 1


def flips(case: Case) -> list[tuple[str, Case]]:
    """Every case differing from `case` in exactly one declared fact, with the fact's kind."""
    out: list = []
    base = canonical_partition(case.rids)
    seen = set()
    for p in range(3):
        for rid in tuple(case.rids[q] for q in range(3) if q != p) + (FRESH,):
            rids = tuple(rid if q == p else case.rids[q] for q in range(3))
            shape = canonical_partition(rids)
            if shape != base and shape not in seen:
                seen.add(shape)
                out.append(
                    (
                        "partition",
                        Case(rids, case.op, case.elem, case.width, case.hazard, case.volatile),
                    )
                )
    if case.hazard == "barriered":
        out.append(
            (
                "volatile",
                Case(case.rids, case.op, case.elem, case.width, "barriered", not case.volatile),
            )
        )
    if not case.volatile:
        other = "barriered" if case.hazard == "unique" else "unique"
        out.append(("hazard", Case(case.rids, case.op, case.elem, case.width, other, False)))
    out.append(
        (
            "size",
            Case(case.rids, case.op, case.elem, case.width, case.hazard, case.volatile, ((0, 8),)),
        )
    )
    other_elem = "i32" if case.elem == "f32" else "f32"
    out.append(
        ("elem", Case(case.rids, case.op, other_elem, case.width, case.hazard, case.volatile))
    )
    return out


def _diff_rails() -> tuple:
    """(label, facts(case) or Refused, the flip kinds its text can carry)."""
    every = frozenset({"partition", "volatile", "hazard", "size", "elem"})

    def c_rail(emit):
        def facts(case):
            text, fn = emit(case)
            return c_facts(parse_c(text, fn))

        return facts

    return (
        ("ll", lambda c: ll_facts(parse_ll(emit_ll(c))), every),
        ("c.kernel", c_rail(emit_c_kernel), every),
        ("c.gather", c_rail(emit_gather), every),
        ("c.specialist", c_rail(emit_specialist), every),
        ("c.header", c_rail(emit_header), every - {"hazard"}),  # a prototype carries no fence
        ("c.qfixed", c_rail(emit_qfixed), frozenset({"partition", "volatile", "hazard"})),
    )


def grade_differential(out: dict[str, float], seen: dict | None = None) -> None:
    bases = [
        Case(rids, "ADD", elem, None, hazard, volatile)
        for rids in PARTITIONS
        for elem in ELEMS
        for hazard, volatile in CONTRACTS
    ]
    for base in bases:
        for kind, flipped in flips(base):
            for label, facts, kinds in _diff_rails():
                if kind not in kinds or (label == "c.qfixed" and base.elem != "f32"):
                    continue
                try:
                    a = _facts_or_refusal(facts, base)
                    b = _facts_or_refusal(facts, flipped)
                    out["alias.differential.silent"] += a[0] == "facts" and a == b
                except Exception:  # noqa: BLE001
                    out["alias.differential.silent"] += 1
                _note(seen, f"{label}.flips")


def _facts_or_refusal(facts, case: Case):
    try:
        return ("facts", facts(case))
    except Refused:
        return ("refused",)


# --- R12 against forged kernels ----------------------------------------------------------------

LL_FORGERIES = (
    "noalias.add",  # noalias on every pointer: false on a shared resource
    "noalias.drop",  # noalias on none: dropped on an exclusive one
    "volatile.drop",
    "volatile.add",
    "scope.drop",  # every !alias.scope / !noalias removed
    "scope.own",  # every access's !noalias names every scope, its own among them
    "tbaa.drop",
    "tbaa.type",  # the scalar type renamed to another C type
    "fence.drop",  # the exit fence removed
    "fence.add",  # a fence on a claim that declares none
    "fence.narrow",  # the barrier's fences narrowed to one thread (`syncscope("singlethread")`)
    "volatile.bare",  # one access of a volatile claim respelled bare: no volatile, no alignment
)
C_FORGERIES = (
    "c.restrict.add",
    "c.restrict.drop",
    "c.volatile.drop",
    "c.fence.drop",
    "c.fence.add",
    "c.fence.narrow",  # the barrier's fences narrowed to a signal fence
    "c.fence.return",  # an early `return` between the barrier's fences
    "c.volatile.cast",  # one read of a volatile operand through a cast that sheds `volatile`
    "c.volatile.address",  # one read through its address, cast to a plain pointer
)
# What R12 must say about each forgery: a new finding naming the forged fact. A false fact and a
# dropped one in the same family must read apart (`noalias.*`, `scope.*`); elsewhere the family's
# name suffices -- which also keeps the parent's own R12 findings (its fence check, its check of a
# disjoint claim's `restrict`) credited where they fired.
FORGERY_FINDS = {
    "noalias.add": "false alias fact",
    "noalias.drop": "alias fact dropped",
    "volatile.drop": "volatile",
    "volatile.add": "volatile",
    "scope.drop": "!alias.scope",
    "scope.own": "own resource's scope",
    "tbaa.drop": "TBAA",
    "tbaa.type": "TBAA",
    "fence.drop": "fence",
    "fence.add": "ordering",
    "fence.narrow": "fence",
    "volatile.bare": "volatile",
    "c.restrict.add": "restrict",
    "c.restrict.drop": "restrict",
    "c.volatile.drop": "volatile",
    "c.fence.drop": "atomic_thread_fence",
    "c.fence.add": "ordering",
    "c.fence.narrow": "atomic_thread_fence",
    "c.fence.return": "return",
    "c.volatile.cast": "other than as a subscript",
    "c.volatile.address": "takes an address",
}


def forge_ll(kind: str, text: str, decl: Declared) -> str | None:
    """`text` with one fact forged, or None when the case has no such fact to forge."""
    shared = not all(decl.exclusive)
    if kind == "noalias.add":
        return re.sub(r"ptr (?:noalias )?%([ABC])\b", r"ptr noalias %\1", text) if shared else None
    if kind == "noalias.drop":
        return text.replace("ptr noalias %", "ptr %") if any(decl.exclusive) else None
    if kind == "volatile.drop":
        return re.sub(r"\b(load|store) volatile\b", r"\1", text) if decl.volatile else None
    if kind == "volatile.add":
        return (
            None if decl.volatile else re.sub(r"\b(load|store) (?!volatile)", r"\1 volatile ", text)
        )
    if kind == "scope.drop":
        return re.sub(r",\s*!(?:alias\.scope|noalias)\s+!\d+", "", text)
    if kind == "scope.own":
        k = parse_ll(text)
        scopes = sorted(s for s in k.metadata if _scope_domain(k, s) is not None)
        if not scopes:
            return text
        node = max(k.metadata) + 1
        lines = []
        for ln in text.splitlines():
            if _LL_ACCESS.match(ln.split(";", 1)[0]):
                ln = re.sub(r",\s*!noalias\s+!\d+", "", ln) + f", !noalias !{node}"
            lines.append(ln)
        lines.append(f"!{node} = !{{{', '.join(f'!{s}' for s in scopes)}}}")
        return "\n".join(lines) + "\n"
    if kind == "tbaa.drop":
        return re.sub(r",\s*!tbaa\s+!\d+", "", text)
    if kind == "tbaa.type":
        other = "int" if decl.tbaa == "float" else "float"
        return text.replace(f'!{{!"{decl.tbaa}", ', f'!{{!"{other}", ')
    if kind == "fence.drop":
        if decl.fence is None:
            return None
        lines = text.splitlines()
        kept = [
            ln
            for i, ln in enumerate(lines)
            if not (
                _LL_FENCE.match(ln)
                and i + 1 < len(lines)
                and lines[i + 1].strip().startswith("ret")
            )
        ]
        return "\n".join(kept) + "\n"
    if kind == "fence.add":
        return (
            None
            if decl.fence is not None
            else text.replace("\nentry:\n", "\nentry:\n  fence seq_cst\n", 1)
        )
    if kind == "fence.narrow":
        if decl.fence is None:
            return None
        return re.sub(r"\bfence (?=seq_cst\b)", 'fence syncscope("singlethread") ', text)
    if kind == "volatile.bare":
        if not decl.volatile:
            return None
        lines = text.splitlines()
        for i, ln in enumerate(lines):
            m = re.match(
                r"^(\s*%[\w.]+\s*=\s*load)\s+(?:volatile\s+)?([^,]+),\s*ptr\s+(%[\w.]+)", ln
            )
            if m:
                lines[i] = f"{m.group(1)} {m.group(2)}, ptr {m.group(3)}"
                return "\n".join(lines) + "\n"
        return text
    raise ValueError(kind)


def forge_c(kind: str, text: str, decl: Declared) -> str | None:
    m = _C_SIG.search(text)
    if m is None:
        return text
    sig = m.group(0)
    if kind == "c.restrict.add":
        if all(decl.exclusive):
            return None
        return text.replace(sig, re.sub(r"\*\s*(?:restrict\s+)?([ABC])\b", r"* restrict \1", sig))
    if kind == "c.restrict.drop":
        return (
            text.replace(sig, re.sub(r"\s*\brestrict\b", "", sig)) if any(decl.exclusive) else None
        )
    if kind == "c.volatile.drop":
        return text.replace(sig, re.sub(r"\bvolatile\s+", "", sig)) if decl.volatile else None
    if kind == "c.fence.drop":
        if decl.fence is None:
            return None
        return "\n".join(ln for ln in text.splitlines() if C_FENCE not in ln) + "\n"
    if kind == "c.fence.add":
        return None if decl.fence is not None else text.replace(sig, sig + f"\n  {C_FENCE};", 1)
    if kind == "c.fence.narrow":
        return (
            None
            if decl.fence is None
            else text.replace("atomic_thread_fence", "atomic_signal_fence")
        )
    if kind == "c.fence.return":
        if decl.fence is None:
            return None
        head, tail = text[: m.end()], text[m.end() :]
        k = tail.find(C_FENCE)
        at = tail.index("\n", k) if k >= 0 else 0  # after the entry fence (or the brace)
        return head + tail[:at] + "\n  if (n == 0) return;" + tail[at:]
    if kind == "c.volatile.cast":
        if not decl.volatile:
            return None
        head, tail = text[: m.end()], text[m.end() :]
        return head + tail.replace("A[", f"((const {decl.ctype} *)A)[", 1)
    if kind == "c.volatile.address":
        if not decl.volatile:
            return None
        head, tail = text[: m.end()], text[m.end() :]
        return head + tail.replace("A[", f"*(const {decl.ctype} *)&A[", 1)
    raise ValueError(kind)


def grade_forgeries(out: dict[str, float], seen: dict | None = None) -> None:
    from bcir.verify import verify_c_lowering, verify_lowering

    for case in _built(core_cases, ("alias.r12.forgery.accepted",), out):
        decl = declared(case)
        m, r = planned(case)
        rails = (
            (emit_ll, forge_ll, LL_FORGERIES, verify_lowering),
            (emit_c, forge_c, C_FORGERIES, verify_c_lowering),
        )
        for emit, forge, kinds, law in rails:
            try:
                honest = emit(case)
                base = {
                    d.message
                    for d in law(m, r, honest, case.elem, width_override=case.width)
                    if d.law == "R12"
                }
            except Exception:  # noqa: BLE001
                out["alias.r12.forgery.accepted"] += len(kinds)
                continue
            for kind in kinds:
                try:
                    forged = forge(kind, honest, decl)
                    if forged is None:
                        continue
                    new = {
                        d.message
                        for d in law(m, r, forged, case.elem, width_override=case.width)
                        if d.law == "R12"
                    } - base
                    out["alias.r12.forgery.accepted"] += not any(
                        FORGERY_FINDS[kind] in message for message in new
                    )
                    _note(seen, "forgeries", {kind})
                except Exception:  # noqa: BLE001 -- a law that raised on a forgery judged nothing
                    out["alias.r12.forgery.accepted"] += 1


def measure(stride: int = 1, seen: dict | None = None, llvm: bool = False) -> dict[str, float]:
    """The G9 rows (module docstring) over every `stride`-th lawful case, every refusal, the
    differential and the forgeries; with `llvm`, the rows LLVM judges, where its tools are."""
    out = {row: 0.0 for row in ROWS}
    for case in _built(valid_cases, ROWS[:5], out)[::stride]:
        grade_case(case, out, seen)
    grade_refusals(out, seen)
    grade_differential(out, seen)
    grade_forgeries(out, seen)
    if llvm:
        out.update(measure_llvm(seen) or {})
    return out


# --- the rows LLVM judges ----------------------------------------------------------------------


def llvm_tools() -> dict | None:
    """clang, llvm-link and opt from one coherent LLVM, or None."""
    from bcir.toolchain import resolve_llvm_tools

    tools = resolve_llvm_tools("clang", "llvm-link", "opt", pipeline="alias facts")
    return dict(tools.paths) if tools.ok else None


_AA_PAIR = re.compile(r"^\s*(NoAlias|MayAlias|MustAlias|PartialAlias):\s+(.*<->.*)$")
_AA_FUNC = re.compile(r"^Function:\s+([\w.$]+):")


def aa_verdicts(opt: str, module_path: str, pipeline: str) -> dict:
    """{function: [(verdict, pointer, pointer), ...]} over the load/store pairs `aa-eval` judges
    with the access metadata (`-evaluate-aa-metadata`)."""
    run = subprocess.run(
        [
            opt,
            "-passes=aa-eval",
            f"-aa-pipeline={pipeline}",
            "-evaluate-aa-metadata",
            "-print-all-alias-modref-info",
            "-disable-output",
            module_path,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if run.returncode != 0:
        raise RuntimeError(f"opt aa-eval failed: {run.stderr[-400:]}")
    out: dict = {}
    fn = None
    for line in (run.stdout + run.stderr).splitlines():
        f = _AA_FUNC.match(line)
        if f:
            fn = f.group(1)
            continue
        p = _AA_PAIR.match(line)
        if p and fn is not None:
            ptrs = re.findall(r"\bptr\s+(%[\w.]+)\s*,\s*align", p.group(2))
            if len(ptrs) == 2:
                out.setdefault(fn, []).append((p.group(1), ptrs[0], ptrs[1]))
    return out


def _gep_table(text: str) -> dict:
    table = {}
    for raw in text.splitlines():
        g = _LL_GEP.match(raw)
        if g and g.group(2) in ("%A", "%B", "%C"):
            table[g.group(1)] = POSITIONS.index(g.group(2)[1:])
    return table


def measure_llvm(seen: dict | None = None) -> dict[str, float] | None:
    """The rows LLVM judges (module docstring), or None without a coherent LLVM toolset."""
    tools = llvm_tools()
    if tools is None:
        return None
    out = {row: 0.0 for row in LLVM_ROWS}
    tmp = tempfile.mkdtemp(prefix="bcir-alias-")
    try:
        cases = core_cases()
        texts, c_parts, paths = {}, [], []
        for i, case in enumerate(cases):
            try:
                texts[i] = emit_ll(case, f"k{i}")
                c_parts.append(emit_c(case, f"c{i}"))
            except Exception:  # noqa: BLE001 -- a lawful case that did not lower
                texts.pop(i, None)
                for row in LLVM_ROWS[:3]:
                    out[row] += 1
                continue
            path = os.path.join(tmp, f"k{i}.ll")
            with open(path, "w", newline="\n") as f:
                f.write(texts[i])
            paths.append(path)
        linked = os.path.join(tmp, "all.ll")
        link = subprocess.run(
            [tools["llvm-link"], "-S", *paths, "-o", linked],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if link.returncode != 0:
            raise RuntimeError(f"llvm-link failed: {link.stderr[-400:]}")
        default = aa_verdicts(tools["opt"], linked, "default")
        scoped = aa_verdicts(tools["opt"], linked, "scoped-noalias-aa")
        for i, case in enumerate(cases):
            if i not in texts:
                continue
            table = _gep_table(texts[i])
            for verdict, p1, p2 in default.get(f"k{i}", ()):
                a, b = table.get(p1), table.get(p2)
                if a is not None and b is not None and a != b:
                    out["alias.llvm.false_noalias"] += (
                        verdict == "NoAlias" and case.rids[a] == case.rids[b]
                    )
                    _note(seen, "aa.default.pairs")
            for verdict, p1, p2 in scoped.get(f"k{i}", ()):
                a, b = table.get(p1), table.get(p2)
                if a is not None and b is not None and a != b and case.rids[a] != case.rids[b]:
                    out["alias.llvm.scope_facts.missing"] += verdict != "NoAlias"
                    _note(seen, "aa.scoped.pairs")
        # clang's IR for the C kernels against the LLVM kernels.
        csrc, cll = os.path.join(tmp, "all.c"), os.path.join(tmp, "all_c.ll")
        with open(csrc, "w", newline="\n") as f:
            f.write("\n".join(c_parts))
        build = subprocess.run(
            [
                tools["clang"],
                "-std=c2x",
                "-O1",
                "-S",
                "-emit-llvm",
                "-fno-discard-value-names",
                csrc,
                "-o",
                cll,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if build.returncode != 0:
            raise RuntimeError(f"clang failed: {build.stderr[-400:]}")
        with open(cll) as f:
            ctext = f.read()
        for i in texts:
            theirs = parse_ll(ctext, f"c{i}")
            if theirs.functions == 0 or not theirs.accesses:
                out["alias.backends.disagree"] += 4
                continue
            ours = side_facts(parse_ll(texts[i]))
            out["alias.backends.disagree"] += sum(x != y for x, y in zip(ours, side_facts(theirs)))
            _note(seen, "clang.kernels")
        out["alias.llvm.caller_memory"] = float(
            caller_memory(tools, tmp, linked, cases, texts, seen)
        )
        out["alias.exec.failed"] = float(exec_failures(tmp, seen))
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_CALLER = """
void caller{i}(const {t} *A, const {t} *B, {t} *C, long n, long *count) {{
  *count = 7;          /* a store to the caller's own data, before the kernel */
  k{i}(A, B, C, n);    /* inlined: the kernel's accesses */
  *count += n;         /* forwarded from the store above, or reloaded */
}}
"""


def caller_memory(tools: dict, tmp: str, linked: str, cases: list, texts: dict, seen) -> int:
    """`alias.llvm.caller_memory` (module docstring): each kernel inlined into a C caller keeping a
    `long` in memory, the caller compiled by clang (its own TBAA) and the module optimized by opt
    -O2, the inliner told to inline. A unique kernel's accesses carry the element type's TBAA
    tag, disjoint from `long`, so the store before it is dead and the value after it known: any
    `long` access beyond the final store is one the facts should have removed. A barriered
    kernel's fences order the caller's memory as well, so both must survive it."""
    parts = []
    for i in texts:
        t = ELEM_TYPES[cases[i].elem][2]
        parts.append(f"extern void k{i}(const {t} *A, const {t} *B, {t} *C, long n);")
        parts.append(_CALLER.format(i=i, t=t))
    src, cll, both, opt_out = (
        os.path.join(tmp, n) for n in ("callers.c", "callers.ll", "lto.ll", "lto2.ll")
    )
    with open(src, "w", newline="\n") as f:
        f.write("#include <stdint.h>\n" + "\n".join(parts))
    for cmd in (
        [tools["clang"], "-std=c2x", "-O2", "-S", "-emit-llvm", src, "-o", cll],
        [tools["llvm-link"], "-S", cll, linked, "-o", both],
        [tools["opt"], "-O2", "-inline-threshold=100000", "-S", both, "-o", opt_out],
    ):
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if run.returncode != 0:
            raise RuntimeError(f"{os.path.basename(cmd[0])} failed: {run.stderr[-400:]}")
    with open(opt_out) as f:
        text = f.read()
    bad = 0
    for i in texts:
        body = re.search(rf"^define[^@]*@caller{i}\(.*?^}}", text, re.S | re.M)
        if body is None or re.search(rf"\bcall\b[^\n]*@k{i}\(", body.group(0)):
            bad += 3  # not inlined: the facts were never tried
            continue
        stores = len(re.findall(r"\bstore i64\b", body.group(0)))
        loads = len(re.findall(r"\bload i64\b", body.group(0)))
        if cases[i].hazard == "barriered":
            bad += max(0, 2 - stores) + max(0, 1 - loads)
        else:
            bad += max(0, stores - 1) + loads
        _note(seen, "lto.callers")
    return bad


def exec_runners() -> list[tuple[str, object]]:
    """The self-check runners whose tools this host has: (label, run(module, result, workdir))."""
    from bcir.lower.c_kernel import compile_and_run_c, compile_and_run_qfixed_c
    from bcir.lower.jit import jit_run
    from bcir.lower.llvm import compile_and_run
    from bcir.lower.wasm import run_wasm_node
    from bcir.toolchain import resolve_llvm_tools

    runners = []
    if resolve_llvm_tools("clang", pipeline="alias exec").ok:
        runners += [
            ("ll.aot", lambda m, r, d: compile_and_run(m, r, workdir=d)),
            ("c.aot", lambda m, r, d: compile_and_run_c(m, r, workdir=d)),
            ("qfixed.aot", lambda m, r, d: compile_and_run_qfixed_c(m, r, workdir=d)),
        ]
    if resolve_llvm_tools("clang", "llvm-link", "lli", pipeline="alias exec").ok:
        runners.append(("ll.jit", lambda m, r, d: jit_run(m, r, workdir=d)))
    if shutil.which("node") and resolve_llvm_tools("clang", "wasm-ld", pipeline="alias exec").ok:
        runners.append(("wasm.node", lambda m, r, _d: run_wasm_node(m, r)))
    return runners


def exec_failures(tmp: str, seen: dict | None = None) -> int:
    """(kernel, runner) pairs whose compiled self-check fails with the declared aliasing bound."""
    failed = 0
    runners = exec_runners()
    _note(seen, "exec.runners", {label for label, _ in runners})
    for case in exec_cases():
        m, r = planned(case)
        for label, run in runners:
            try:
                ok, _ = run(m, r, tempfile.mkdtemp(dir=tmp))
            except Exception:  # noqa: BLE001
                ok = False
            failed += not ok
            _note(seen, f"{label}.runs")
    return failed
