"""Emit C back out of a lowered function's claim graph — the *verified C output* seam. Walking the
claims in order reproduces the source's integer semantics exactly (the emitter maps each claim's `op`
back to its C operator / memory access), so compiling this alongside the original fixture and
diffing the results on the same inputs is the behaviour-equivalence check the ladder requires.

This is the per-pattern-free, claim-graph-driven path the roadmap's C.2 generalizes toward: it emits
an *arbitrary* straight-line scalar claim graph, not a fixed kernel template.
"""

from __future__ import annotations

import re

from ...model import Claim
from .ctype_model import CType, unqualified
from .lower import (
    AsmInfo,
    BreakNode,
    CaseLabel,
    CLowerError,
    ComputedGotoNode,
    ContinueNode,
    DefaultLabel,
    GotoNode,
    IfNode,
    LabelNode,
    LoweredFunc,
    ReturnNode,
    SwitchNode,
    WhileNode,
    _STRING_MEM,
)

# ASM2 -- the x86 targets that have port-mapped I/O (`in`/`out` instructions). ARM/RISC-V have no port I/O
# (only MMIO), so their realization is an honest unsupported diagnostic -> the LLVM fallback. ASM3 reuses the
# same target families to key the per-ISA memory-fence (barrier) instruction; unlike port I/O, EVERY ISA has
# fences, so a target outside these three families falls back to the portable __atomic_thread_fence emit.
_X86_TARGETS = frozenset({"x86_64-linux", "i386-linux", "x86_64-windows"})
_ARM_TARGETS = frozenset({"aarch64-linux"})
_RISCV_TARGETS = frozenset({"riscv64-linux"})

# The x86 `in`/`out` operand templates per access width, in the standard GCC/Linux <asm/io.h> form. The
# accumulator is `%b0`/`%w0`/`%k0` (al / ax / eax) and the port is `%w1` (an immediate-or-dx port, the `Nd`
# constraint). A READ writes the accumulator (`"=a"`); a WRITE reads it (`"a"`). The port rides `"Nd"`.
_PORTIO_IN_ASM = {"b": '"inb %w1, %b0"', "w": '"inw %w1, %w0"', "l": '"inl %w1, %k0"'}
_PORTIO_OUT_ASM = {"b": '"outb %b0, %w1"', "w": '"outw %w0, %w1"', "l": '"outl %k0, %w1"'}


def _portio_stmt(lf: LoweredFunc, c: Claim, ref) -> str:
    """Re-emit one port-I/O claim (ASM2) as the real x86 `in`/`out` instruction, behind a GNU
    `__asm__ __volatile__` (reusing the ASM1 trusted-edge + `barriered` machinery), keyed off the function's
    target. The width + direction live in the op suffix (`c.portio.in.b:` etc.); the port/value RIDs ride the
    claim's rd. A NON-x86 target raises an honest `CLowerError` -- port-mapped I/O is an x86-only facility
    (ARM/RISC-V have no port I/O, only MMIO), so the unit routes to the LLVM fallback (the established
    honest-depth pattern). The emitted instruction is REAL x86 the toolchain assembles; EXECUTING it is
    privileged (ring-0 / iopl), so the runtime probe + tests ASSEMBLE the asm (gcc/clang -c), never run it."""
    head, name = c.op.split(":", 1) if ":" in c.op else (c.op, "")
    parts = head.split(".")  # ["c", "portio", "in"|"out", "b"|"w"|"l"]
    direction, width = parts[2], parts[3]
    if lf.target not in _X86_TARGETS:  # ARM / RISC-V: no port I/O -> honest diagnostic
        raise CLowerError(
            f"port-mapped I/O ({name}) requires an x86 target; {lf.target} has no port I/O -- use MMIO"
        )
    if direction == "in":  # inb/inw/inl: port is rd[0]; result temp is wr[0]
        et = _load_ctype(lf, c.wr[0])
        templ = _PORTIO_IN_ASM[width]
        return (
            f"{et} {ref(c.wr[0])}; "
            f'__asm__ __volatile__ ({templ} : "=a" ({ref(c.wr[0])}) : "Nd" ({ref(c.rd[0])}));'
        )
    # outb/outw/outl: value is rd[0], port is rd[1] (the Linux out*(value, port) order)
    templ = _PORTIO_OUT_ASM[width]
    return f'__asm__ __volatile__ ({templ} :  : "a" ({ref(c.rd[0])}), "Nd" ({ref(c.rd[1])}));'


# ASM3 -- the native memory-fence (hardware barrier) mnemonic per (kind, ISA family). The kind rides the op
# suffix (`c.fence` full / `c.fence.acquire` load / `c.fence.release` store); the `"memory"` clobber is the
# REQUIRED compiler-barrier half of the fence (it stops the compiler reordering memory accesses across it).
# x86: mfence (full) / lfence (load) / sfence (store). aarch64: `dmb ish` (full) / `dmb ishld` (load) /
# `dmb ishst` (store), the inner-shareable data-memory barriers. riscv64: `fence rw,rw` / `fence r,rw` /
# `fence rw,w`, the RISC-V predecessor,successor fences. Every ISA HAS a fence, so a target outside these
# three families uses the portable __atomic_thread_fence default (an honest seam, NOT a fallback).
_FENCE_ASM = {
    "x86": {"full": "mfence", "acquire": "lfence", "release": "sfence"},
    "arm": {"full": "dmb ish", "acquire": "dmb ishld", "release": "dmb ishst"},
    "riscv": {"full": "fence rw,rw", "acquire": "fence r,rw", "release": "fence rw,w"},
}


def _fence_stmt(lf: LoweredFunc, c: Claim) -> str:
    """Re-emit one memory-fence claim (ASM3) as the REAL per-ISA hardware barrier instruction behind a GNU
    `__asm__ __volatile__ (... ::: "memory")`, keyed off the function's target (mirroring ASM2's `_portio_stmt`).
    The KIND rides the op suffix: `c.fence` (full / seq_cst) / `c.fence.acquire` (load fence) / `c.fence.release`
    (store fence). Unlike port I/O, every ISA HAS a fence -- so a target outside the x86/aarch64/riscv64
    families keeps the portable `__atomic_thread_fence(__ATOMIC_SEQ_CST);` as an honest default (all five
    shipping ABIs are covered by the three families, so this is just a safety net, NOT an unsupported
    diagnostic). The `"memory"` clobber is the compiler-barrier half of the fence and is always present.

    When NO ISA was explicitly chosen (a host-default target), the portable __atomic_thread_fence is emitted
    instead -- so the default emit compiles on ANY host (a cross-arch native compile never sees foreign asm,
    e.g. x86 `mfence` would not assemble on an aarch64 host); native per-ISA asm is opt-in via `--target`."""
    if not getattr(lf, "target_explicit", False):
        return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"  # no explicit --target: portable, host-agnostic
    parts = c.op.split(".")  # ["c", "fence"] | ["c", "fence", "acquire"|"release"]
    kind = parts[2] if len(parts) > 2 else "full"  # the bare `c.fence` is the FULL (seq_cst) fence
    if lf.target in _X86_TARGETS:
        family = "x86"
    elif lf.target in _ARM_TARGETS:
        family = "arm"
    elif lf.target in _RISCV_TARGETS:
        family = "riscv"
    else:  # an ISA outside the three families: portable default
        return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"
    mnem = _FENCE_ASM[family][kind]
    return f'__asm__ __volatile__ ("{mnem}" ::: "memory");'


# op-suffix -> C operator.
_BINOP = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "mod": "%",
    "and": "&",
    "or": "|",
    "xor": "^",
    "shl": "<<",
    "shr": ">>",
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "gt": ">",
    "le": "<=",
    "ge": ">=",
    "land": "&&",
    "lor": "||",
}
_UNOP = {"neg": "-", "bnot": "~", "lnot": "!"}


def _cname(ct: CType) -> str:
    if ct.kind == "pointer":
        return _cname(ct.of) + " *"
    if ct.kind == "array":
        return _cname(ct.of)  # decays in a parameter position
    vol = "volatile " if ct.volatile else ""  # a volatile object / pointee keeps its qualifier:
    if ct.is_aggregate:  # every access through the declaration stays one
        return f"{vol}{ct.kind} {ct.name}"
    return vol + ("_Atomic " if ct.atomic else "") + ct.name


def _volatile_ptr(t: str) -> str:
    """A pointer to a volatile slot of C type `t`: the conventional `volatile T *`, or `T volatile *`
    when `t` is itself a pointer type (a qualifier in front would qualify its pointee, not the slot).
    The twin spells it byte-identically (`bcir_cfront.c`, `vol_ptr`)."""
    return f"{t} volatile *" if t.rstrip().endswith("*") else f"volatile {t} *"


def _atomic_ptr(t: str, vol: bool = False) -> str:
    """A pointer to an `_Atomic` object of C type `t` (CF-ATOMIC): `_Atomic T *`, `volatile` too for a
    device access, and `T _Atomic *` when `t` is itself a pointer type (a qualifier in front would
    qualify its pointee). The twin spells it the same (`bcir_cfront.c`, `atomic_ptr`)."""
    q = "volatile _Atomic" if vol else "_Atomic"
    return f"{t} {q} *" if t.rstrip().endswith("*") else f"{q} {t} *"


# An increment or decrement of an `_Atomic` object (`c.c11atom.rmw:<kind>`), as the emitted operator.
_RMW_INCDEC = {"preinc": "++{}", "predec": "--{}", "postinc": "{}++", "postdec": "{}--"}


def _atomic_object(lf: LoweredFunc, c: Claim, ref, t: str, naddr: int, *, stride_at: int) -> str:
    """The `_Atomic` lvalue an atomic access performs (CF-ATOMIC), `(*(_Atomic T *)ADDR)`, at the
    address the claim's first `naddr` reads and its imm name -- exactly where the plain access lands: a
    typed element `base[idx]` keeps its bounds guard (a write site's, since an atomic access may write);
    a member-array element or an array-of-structs field lands at `base + off + idx*stride` (the stride at
    `imm[stride_at]`, else the element size); a member, a dereference or a named object at `base + off`.
    Never a byte copy: a `memcpy` of an atomic object is not an atomic access, and it tears."""
    ptr = _atomic_ptr(t, c.volatile)
    if naddr == 2 and not c.imm:
        return f"(*({ptr})&{ref(c.rd[0])}[{_idx(lf, c, ref, write=c.op != 'c.load')}])"
    bp = _base_ptr(lf, c.rd[0], ref)
    off = c.imm[0] if c.imm else 0
    if naddr == 2:
        es = c.imm[1] if len(c.imm) > 1 else 4
        stride = c.imm[stride_at] if len(c.imm) > stride_at else es
        return f"(*({ptr})((char *){bp} + {off} + (size_t){ref(c.rd[1])} * {stride}))"
    return f"(*({ptr})((char *){bp} + {off}))"


def _elem_ctype(lf: LoweredFunc, rid: int) -> str:
    """The C type of an element of the pointer or array `rid`, unqualified: the slot an atomic store
    into a typed element `base[idx]` writes."""
    ct = lf.rid_types.get(rid)
    el = ct.of if ct is not None and ct.of is not None else None
    return _cname(unqualified(el)) if el is not None else "uint32_t"


def _funcptr_decl(ct: CType, name: str) -> str:
    """Render an inline function-pointer declarator `RET (*name)(PARAMS)`. Used in param + local
    position, where (unlike a typedef alias) there is no spelling to print and the full signature
    must be reconstructed from the carried return + parameter types — `int (*g)(int)`."""
    ret = _cname(ct.of) if ct.of is not None else "void"
    plist = ", ".join(_cname(p) for p in ct.params) or "void"
    return f"{ret} (*{name})({plist})"


_ANON_REF = re.compile(r"(?<![A-Za-z0-9_$])(?:struct|union) (\$anon\d+)(?![A-Za-z0-9_$])")
_LITERAL_OR_COMMENT = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|/\*.*?\*/|//[^\n]*', re.S)


def respell_anon(text: str, spelling: dict) -> str:
    """Every `struct $anonN` / `union $anonN` of emitted C -- outside a string or character literal and a
    comment -- spelled as C names it (`Unit.anon_spelling`: a typedef's name, or `__typeof__` of the member
    whose type it is): the synthesized tag is no compiler's, so an emit using a typedef'd anonymous struct
    did not compile (CF-ANON). One C cannot name stays as it was. The twin rewrites its emit alike
    (`bcir_cfront.c`, `respell_anon`)."""
    if not spelling:
        return text

    def _code(part: str) -> str:
        return _ANON_REF.sub(lambda m: spelling.get(m.group(1), m.group(0)), part)

    out, at = [], 0
    for m in _LITERAL_OR_COMMENT.finditer(text):
        out += [_code(text[at : m.start()]), m.group(0)]
        at = m.end()
    out.append(_code(text[at:]))
    return "".join(out)


def _object_declarator(ct: CType, name: str) -> str:
    """An object's declaration without its initializer: `T name`, an array's every dimension after its name
    (`T name[2][3]`, the element type the innermost; an unsized one `[]`)."""
    dims = []
    while ct.kind == "array":
        dims.append(f"[{ct.count}]" if ct.count else "[]")
        ct = ct.of
    return f"{_cname(ct)} {name}{''.join(dims)}"


def emit_linkable(lowered, emitted: dict) -> str:
    """The LINKABLE artifact (Phase 3 linking): the unit's emitted functions re-rendered with
    EXTERNAL linkage -- definitions non-static under their REAL names, in-unit calls unprefixed
    -- plus the unit's file-scope globals (a definition with its constant initializer: integers,
    signed integers, float and string spellings; `extern T g;` stays a declaration), so two
    emitted TUs link TO EACH OTHER (and to any host object). SOURCE LINKAGE IS HONORED: a
    source-`static` function or global keeps `static` (internal linkage, real name -- two TUs
    may each carry a same-named static helper); everything else exports. The default emit stays
    `static bcir_`-prefixed (self-contained beside the source); this mode is opt-in
    (`--linkable`). A non-renderable initializer (&x, arithmetic, sizeof) raises rather than
    mislowers."""
    names = list(lowered.functions)
    parts: list[str] = ["#include <stdint.h>"]  # the emitted temps are int32_t/uint32_t/...
    if any("BCIR_CHK" in emitted[n] for n in names):  # a masked (bounds-promoted) access references
        parts.append('#include "bcir_quarantine.h"')  # the quarantine ABI -- link bcir_quarantine.c
    ops = [c.op for lf in lowered.functions.values() for c in lf.claims]
    callees = {
        op.split(":", 1)[1]
        for op in ops
        if op.startswith(("c.call.libm:", "c.call.libm.void:", "c.call.extern:"))
    }
    libc = {"malloc", "calloc", "realloc", "aligned_alloc", "free"}
    if callees & libc:
        parts.append("#include <stdlib.h>")
    if callees & _STRING_MEM:
        parts.append("#include <string.h>")  # the memcpy/memmove/memset edges
    if any(op.startswith("c.call.extern:") for op in ops):
        parts.append("#include <stdio.h>")  # the printf/scanf-family edges
    if callees - libc - _STRING_MEM and any(
        op.startswith(("c.call.libm:", "c.call.libm.void:")) for op in ops
    ):
        parts.append("#include <math.h>")  # the remaining libm edges
    threads = getattr(lowered, "thread_globals", frozenset())
    for gname, ct, vals, is_extern, is_static in lowered.globals_decl:
        tls = "_Thread_local " if gname in threads else ""  # each thread's own object (CF-TLS)
        if is_extern:
            parts.append(f"extern {tls}{_object_declarator(ct, gname)};")
            continue
        if vals is None:
            raise ValueError(
                f"linkable emit: global {gname!r} has a non-renderable constant "
                f"initializer (unsupported in this slice)"
            )
        kw = "static " if is_static else ""  # source `static` stays file-local
        # the initializer as the source spells it, re-spelled (`lower._file_scope_rendering`); none: a
        # tentative definition (zero-init)
        init = f" = {vals[0]}" if vals else ""
        parts.append(f"{kw}{tls}{_object_declarator(ct, gname)}{init};")
    for name in names:  # forward-declare every kept-static
        lf = lowered.functions[name]  # function: a static callee may be
        if not lf.static_fn:  # DEFINED after its caller (the C
            continue  # static-forward-declaration idiom)
        ps = ", ".join(_cname(p[2]) for p in lf.params) or "void"
        if lf.variadic:
            ps = (ps + ", ...") if ps != "void" else "..."
        parts.append(f"static {_cname(lf.ret_type)} {name}({ps});")
    for name in names:
        text = emitted[name]
        for fn in names:  # unprefix every in-unit call site
            text = text.replace(f"bcir_{fn}(", f"{fn}(")
        if lowered.functions[name].static_fn:  # source `static` honored: keep the
            parts.append(text)  # definition's internal linkage
            continue
        lines = [
            ln[len("static ") :] if ln.startswith("static ") else ln  # the definition line is the
            for ln in text.splitlines()
        ]  # only UNINDENTED `static `
        parts.append("\n".join(lines))
    return "\n".join(parts) + "\n"


class _Names:
    """The C identifier each rid emits as (`ref`): a parameter, local, static or global its declared name
    (`named`), an intermediate `t<rid>`. A temporary's name must be free of every declared one (`used`): a
    re-parsed emit names its locals `t<rid>` after the temporaries they were (and a source may name a
    parameter so), so a temporary whose `t<rid>` a declared name spells takes the first free `t<rid>_<k>`
    -- never a second declaration of the name (CF-RTVOL). Whether a rid is a temporary is `named`
    membership, never its spelling: a local re-parsed from `t103` may be rid 103 again."""

    def __init__(self, named: dict, used: set):
        self.named, self.used, self.temps = named, used, {}

    def __call__(self, rid: int) -> str:
        if rid in self.named:
            return self.named[rid]
        if rid not in self.temps:
            name, k = f"t{rid}", 2
            while name in self.used:
                name, k = f"t{rid}_{k}", k + 1
            self.temps[rid] = name
        return self.temps[rid]

    def is_temp(self, rid: int) -> bool:
        """An intermediate, declared where it is written, not a named object declared up front."""
        return rid not in self.named


def _signature(lf: LoweredFunc) -> str:
    """The function's signature as its definition spells it, `static RET bcir_NAME(PARAMS)`: the
    definition's head, and the forward declaration of it that a caller's emit makes."""
    parts = [
        _funcptr_decl(ct, pname) if ct.kind == "funcptr" else f"{_cname(ct)} {pname}"
        for pname, _rid, ct in lf.params
    ]
    if lf.variadic:  # a trailing `...` after the named params
        parts.append("...")
    return f"static {_cname(lf.ret_type)} bcir_{lf.name}({', '.join(parts) or 'void'})"


def _proto_param(ct: CType, const_pointee: bool) -> str:
    """A prototyped callee's parameter in its `extern` declaration: a function pointer as its declarator
    with no name, `RET (*)(PARAMS)` (its name alone does not compile), and a pointer to `const` keeps the
    qualifier -- a parameter of `const T *` is another type than one of `T *`, and the two declarations of
    the callee would conflict. The twin spells both alike (`bcir_cfront.c`, the prototype's `tudefs`)."""
    if ct.kind == "funcptr":
        return _funcptr_decl(ct, "")
    return ("const " if const_pointee and ct.kind == "pointer" else "") + _cname(ct)


def emit_function(lf: LoweredFunc, unit: dict | None = None) -> str:
    """The lowered function as standalone C, named `bcir_<name>` (so it can sit beside the original).
    Walks the structured body tree, so `if`/`while`/`return` emit real C control flow; mutable named
    locals are declared up front and assigned (so branch merges + loop accumulators reproduce the
    source); intermediate expression results stay single-assignment temporaries. `unit` -- the unit's
    lowered functions by name -- declares each one the function calls before it: a callee defined after
    its caller is otherwise undeclared at the call, which does not compile."""
    nm: dict[int, str] = {rid: pname for pname, rid, _ct in lf.params}
    # Each local needs a *unique* C identifier: the lowering flattens scopes, so two source locals that
    # shared a name in disjoint scopes (e.g. `i` in two separate `for` loops, or a block local shadowing
    # a param) become distinct rids with the same name. Declaring both at function scope is a C
    # redefinition. Disambiguate the second-and-later occurrences (`i`, `i_2`, ...) -- a fresh variable
    # preserves the source's separate-scope semantics; naive name-sharing would corrupt a shadowed value.
    used: set[str] = set(nm.values())
    used.update(lf.globals_used.values())
    local_name: dict[int, str] = {}

    def _uniq(name: str) -> str:
        uniq, k = name, 2
        while uniq in used:
            uniq, k = f"{name}_{k}", k + 1
        used.add(uniq)
        return uniq

    # a static too: two scopes' `static n` are two objects, both declared at function scope
    for rid, name, _ct, _init in lf.statics:
        nm[rid] = _uniq(name)
    for rid, name, _ct in lf.locals + lf.vla_locals:
        # a VLA is NAMED here (so its accesses resolve) but declared IN-BODY (the c.vladecl claim below),
        # never in the up-front `decls` -- its runtime size isn't known until execution reaches the decl
        local_name[rid] = nm[rid] = _uniq(name)
    nm.update(lf.globals_used)  # file-scope globals (defined in the source)
    ref = _Names(nm, used)

    def _local_decl(rid, name, ct):
        # the zero baseline as the empty initializer: `= {0}` re-lowers as the baseline AND a store of 0 to the
        # first scalar, which the next emit spells as a store -- one more each round (CF-RTWIDE)
        zi = " = {}" if rid in lf.zero_init_locals else ""
        if ct.kind == "funcptr":  # `RET (*name)(PARAMS)` (no typedef alias)
            return f"    {_funcptr_decl(ct, name)}{zi};"
        if ct.kind == "array":  # `T name[N]` (the dims follow the name)
            return f"    {_cname(ct.of)} {name}[{ct.count}]{zi};"
        return f"    {_cname(ct)} {name}{zi};"

    def _static_decl(name, ct, init, thread):
        # static storage: a once-only constant initializer in the declaration -- the image the lowering
        # folded and rendered (CF-STATICTAB), or zero: `{0}` for an array or aggregate, `0u` for the rest.
        # An array keeps its shape (a multi-dimensional one flat, as its image is rendered); thread storage
        # its `_Thread_local` (CF-TLS)
        sc = "static _Thread_local" if thread else "static"
        if ct.kind == "array":
            return f"    {sc} {_cname(ct.of)} {name}[{ct.count}] = {init or '{0}'};"
        if ct.kind in ("struct", "union"):
            return f"    {sc} {_cname(ct)} {name} = {init or '{0}'};"
        if ct.kind == "funcptr":  # `RET (*name)(PARAMS)`, as a local's
            return f"    {sc} {_funcptr_decl(ct, name)} = {init or '0u'};"
        return f"    {sc} {_cname(ct)} {name} = {init or '0u'};"

    decls = [_local_decl(rid, local_name[rid], ct) for rid, _name, ct in lf.locals]
    decls += [
        _static_decl(nm[rid], ct, init, rid in lf.thread_statics)
        for rid, _name, ct, init in lf.statics
    ]
    body = _walk(lf, lf.body, ref, 1, cont=_cont_labels(lf))
    # Phase 3 linking: declare every PROTOTYPED cross-TU callee this function calls, so the
    # emitted TU compiles standalone and the host LINKER resolves the symbol from a sibling object.
    tu_decls = [
        f"extern {_cname(rct)} {callee}("
        + (", ".join(_proto_param(p, k) for p, k in zip(pcts, consts)) or "void")
        + ");"
        for callee, (rct, pcts, consts) in sorted(lf.tu_protos.items())
    ]
    # ... and every function of the unit it calls, in the order of the first call (not itself)
    callees = dict.fromkeys(c for c, _a in lf.calls if c != lf.name and c in (unit or {}))
    fwd = [_signature(unit[c]) + ";" for c in callees]
    head = "\n".join(tu_decls + fwd) + "\n" if tu_decls or fwd else ""
    return head + _signature(lf) + "\n{\n" + "\n".join(decls + body) + "\n}"


def _labels(block: list, out: set) -> set:
    """Every label the structured body `block` defines, at any depth."""
    for node in block:
        if isinstance(node, LabelNode):
            out.add(node.name)
        elif isinstance(node, IfNode):
            _labels(node.then, out)
            _labels(node.els, out)
        elif isinstance(node, WhileNode):
            for part in (node.cond_block, node.body, node.step):
                _labels(part, out)
        elif isinstance(node, SwitchNode):
            _labels([i for i in node.body if not isinstance(i, (CaseLabel, DefaultLabel))], out)
    return out


def _loop_ids(block: list, out: list) -> list:
    """Every loop of the structured body `block`, outer before inner, in source order."""
    for node in block:
        if isinstance(node, WhileNode):
            out.append(node.loop_id)
            for part in (node.cond_block, node.body, node.step):
                _loop_ids(part, out)
        elif isinstance(node, IfNode):
            _loop_ids(node.then, out)
            _loop_ids(node.els, out)
        elif isinstance(node, SwitchNode):
            _loop_ids([i for i in node.body if not isinstance(i, (CaseLabel, DefaultLabel))], out)
    return out


def _cont_labels(lf: LoweredFunc) -> dict:
    """Each loop's continue label: `__cont_<id>`, or `__cont_<id>_<k>` where a label of the function's own
    spells that already. A re-parsed emit keeps the labels the emit before it placed, as labels of its own, and a
    function that defines one label twice does not compile (CF-RTWIDE)."""
    taken = _labels(lf.body, set())
    names = {}
    for loop_id in dict.fromkeys(_loop_ids(lf.body, [])):
        name, k = f"__cont_{loop_id}", 2
        while name in taken:
            name, k = f"__cont_{loop_id}_{k}", k + 1
        taken.add(name)
        names[loop_id] = name
    return names


def _const_true(node: WhileNode) -> bool:
    """The loop's condition is the constant 1 -- `while (1)`, `for (;;)`, the emit's own loop scaffold re-parsed
    -- which ends no iteration, so its emit tests nothing. A break test the emit wrote re-lowered as a branch of
    the loop's body, one more each round (CF-RTWIDE). Only 1 is taken: it is nonzero in every integer type."""
    return any(
        isinstance(c, Claim)
        and c.op == "c.const"
        and c.wr[:1] == (node.cond,)
        and tuple(c.imm) == (1,)
        for c in node.cond_block
    )


def _walk(
    lf: LoweredFunc,
    block: list,
    ref,
    depth: int,
    loops: list | None = None,
    cont: dict | None = None,
) -> list:
    ind = "    " * depth
    loops = loops if loops is not None else []
    cont = cont if cont is not None else {}
    out: list = []
    for node in block:
        if isinstance(node, IfNode):
            out.append(f"{ind}if ({ref(node.cond)}) {{")
            out += _walk(lf, node.then, ref, depth + 1, loops, cont)
            if node.els:
                out.append(f"{ind}}} else {{")
                out += _walk(lf, node.els, ref, depth + 1, loops, cont)
            out.append(f"{ind}}}")
        elif isinstance(node, WhileNode):
            loops.append(node.loop_id)
            label = cont.get(node.loop_id, f"__cont_{node.loop_id}")
            out.append(f"{ind}while (1) {{")
            if node.test_at_end:  # do/while: body, [continue:], recompute + test
                out += _walk(lf, node.body, ref, depth + 1, loops, cont)
                out.append(f"{ind}    {label}: ;")
                out += _walk(lf, node.cond_block, ref, depth + 1, loops, cont)
                if not _const_true(node):
                    out.append(f"{ind}    if (!{ref(node.cond)}) break;")
            else:  # while/for: test, body, [continue:], step
                out += _walk(lf, node.cond_block, ref, depth + 1, loops, cont)
                if not _const_true(node):
                    out.append(f"{ind}    if (!{ref(node.cond)}) break;")
                out += _walk(lf, node.body, ref, depth + 1, loops, cont)
                out.append(f"{ind}    {label}: ;")
                out += _walk(lf, node.step, ref, depth + 1, loops, cont)
            out.append(f"{ind}}}")
            loops.pop()
        elif isinstance(node, SwitchNode):  # a real C switch (fallthrough preserved)
            out.append(f"{ind}switch ({ref(node.disc)}) {{")
            for item in node.body:
                if isinstance(item, CaseLabel):
                    out.append(f"{ind}case {item.value}:")
                elif isinstance(item, DefaultLabel):
                    out.append(f"{ind}default:")
                else:
                    out += _walk(lf, [item], ref, depth + 1, loops, cont)
            out.append(f"{ind}}}")
        elif isinstance(node, ReturnNode):
            out.append(f"{ind}return {ref(node.rid)};" if node.rid is not None else f"{ind}return;")
        elif isinstance(node, BreakNode):
            out.append(f"{ind}break;")
        elif isinstance(node, ContinueNode):
            out.append(f"{ind}goto {cont.get(loops[-1], f'__cont_{loops[-1]}')};")
        elif isinstance(node, GotoNode):
            out.append(f"{ind}goto {node.label};")
        elif isinstance(node, ComputedGotoNode):
            out.append(
                f"{ind}goto *{ref(node.target)};"
            )  # an indirect jump to a label address (GNU)
        elif isinstance(node, LabelNode):
            out.append(f"{node.name}:;")  # a jump target (function-body scope)
        elif isinstance(node, Claim):
            out.append(ind + _claim_stmt(lf, node, ref))
    return out


def _asm_stmt(lf: LoweredFunc, c: Claim, ref) -> str:
    """Re-emit one inline-asm claim (ASM1) as a VERBATIM GNU `__asm__` statement. The template +
    constraints + clobbers pass through unchanged (ISA-neutral); each operand renders as
    `[name] "constraint" (ref(rid))` using the same `ref(...)` operand printer the other trusted edges use.
    The reserved spellings `__asm__` / `__volatile__` are used so the emitted C is valid even under
    `-std=c11 -pedantic`. The BASIC source form (`info.is_basic`: no colon sections) renders as
    `__asm__ [__volatile__] ("template");` so re-parsing it round-trips; an EXTENDED asm (incl. one with all
    sections empty, e.g. `asm("x" : :)`) always emits all three `:` sections, so it round-trips as the
    NON-volatile extended form rather than drifting to the implicitly-volatile basic form."""
    info: AsmInfo = lf.asm_meta[c.id]
    vol = " __volatile__" if info.is_volatile else ""

    def operand(name, constraint, rid) -> str:
        sym = f"[{name}] " if name else ""
        return f'{sym}"{constraint}" ({ref(rid)})'

    # `info.template` is the SOURCE spelling (quotes intact, adjacent-literal concatenation joined) -- emitted
    # unchanged (ISA-neutral verbatim pass-through), so no quotes are re-added around it.
    # the BASIC source form: emit the bare 1-colon form. (Gated on `is_basic`, NOT "all sections empty", so an
    # all-empty EXTENDED `asm("x" : :)` does NOT collapse to the implicitly-volatile basic form on re-emit.)
    if info.is_basic:
        return f"__asm__{vol} ({info.template});"
    out_s = ", ".join(
        operand(n, ct, r) for n, ct, r in zip(info.out_names, info.out_constraints, info.out_rids)
    )
    in_s = ", ".join(
        operand(n, ct, r) for n, ct, r in zip(info.in_names, info.in_constraints, info.in_rids)
    )
    clob_s = ", ".join(f'"{cl}"' for cl in info.clobbers)
    return f"__asm__{vol} ({info.template} : {out_s} : {in_s} : {clob_s});"


def _claim_stmt(lf: LoweredFunc, c: Claim, ref) -> str:
    suf = c.op.split(".", 2)[-1] if "." in c.op else c.op

    def deftmp(rid: int, expr: str, ty: str | None = None) -> str:
        if ty is None and getattr(lf.rid_types.get(rid), "kind", None) == "funcptr":
            # a null function pointer (CF-NULLPTR): `RET (*t)(PARAMS) = 0u;`
            return f"{_funcptr_decl(lf.rid_types[rid], ref(rid))} = {expr};"
        if ty is None:  # a temp renders its true C type (`_load_ctype`): float/double, the
            ty = _load_ctype(lf, rid)  # (width, signedness) integer, `T *`, a struct/union
        return f"{ty} {ref(rid)} = {expr};"

    if c.op == "c.copy":  # write a mutable local (no new decl)
        if ref.is_temp(c.wr[0]):  # into a temp: the value read from a volatile object
            return deftmp(c.wr[0], ref(c.rd[0]))
        return f"{ref(c.wr[0])} = {ref(c.rd[0])};"
    if c.op == "c.vladecl":  # an in-body stack VLA decl: `T a[__ext];`
        act = lf.rid_types.get(c.wr[0])  # the VLA array CType (count 0, of=element type)
        return f"{_cname(act.of)} {ref(c.wr[0])}[{ref(c.rd[0])}];"
    if c.op == "c.ptradd":  # pointer p += n (C scales by element size)
        return f"{ref(c.wr[0])} += {ref(c.rd[1])};"
    if c.op == "c.ptrsub":  # pointer p -= n
        return f"{ref(c.wr[0])} -= {ref(c.rd[1])};"
    if c.op == "c.const":
        return deftmp(c.wr[0], f"{c.imm[0]}u")
    if c.op == "c.sizeof.vla":  # runtime `sizeof a` of a VLA: extent × sizeof(elem)
        return deftmp(c.wr[0], f"(size_t)((size_t){ref(c.rd[0])} * {c.imm[0]})")
    if c.op.startswith("c.labeladdr:"):  # `&&L` -- a label's address as a `void *` (GNU)
        return deftmp(c.wr[0], "&&" + c.op.split(":", 1)[1])
    if c.op.startswith("c.fconst:"):  # a floating constant -> its literal spelling
        return deftmp(c.wr[0], c.op.split(":", 1)[1])
    if c.op.startswith("c.cconst:"):  # <complex.h> imaginary unit -> verbatim token
        return deftmp(c.wr[0], c.op.split(":", 1)[1])
    if c.op.startswith("c.bin."):
        return deftmp(c.wr[0], f"{ref(c.rd[0])} {_BINOP[suf]} {ref(c.rd[1])}")
    if c.op == "c.un.creal":  # GNU __real__ z (complex part extraction)
        return deftmp(c.wr[0], f"__real__ {ref(c.rd[0])}")
    if c.op == "c.un.cimag":
        return deftmp(c.wr[0], f"__imag__ {ref(c.rd[0])}")
    if c.op.startswith("c.un."):
        return deftmp(c.wr[0], f"({_UNOP[suf]}{ref(c.rd[0])})")
    if c.op.startswith("c.cast:"):  # (type)operand — width cast / reinterpret
        return deftmp(c.wr[0], f"({c.op.split(':', 1)[1]}){ref(c.rd[0])}")
    if c.op == "c.addrof":  # &lvalue -> a pointer value (T *t = &x;)
        rt = lf.rid_types.get(c.wr[0])
        ty = _cname(rt) if rt is not None else None
        castp = f"({ty})" if ty else ""
        if len(c.rd) == 2:  # &base[idx] -> (T *)((char *)bp + off + idx*stride)
            bp = _base_ptr(lf, c.rd[0], ref)  # bp decays a pointer/array base, addresses a value
            return deftmp(
                c.wr[0],
                f"{castp}((char *){bp} + {c.imm[0]} + (size_t){ref(c.rd[1])} * {c.imm[1]})",
                ty,
            )
        if c.imm:  # &base.member -> (T *)((char *)bp + off); bp is
            bp = _base_ptr(lf, c.rd[0], ref)  # `s` for `s->m` (pointer), `&s` for `s.m` (value)
            return deftmp(c.wr[0], f"{castp}((char *){bp} + {c.imm[0]})", ty)
        return deftmp(c.wr[0], f"&{ref(c.rd[0])}", ty)  # &name / &(compound literal)
    if c.op == "c.select":  # ternary: cond ? then : els
        return deftmp(c.wr[0], f"({ref(c.rd[0])} ? {ref(c.rd[1])} : {ref(c.rd[2])})")
    if c.op == "c.load":
        et = _load_ctype(lf, c.wr[0])
        if c.hazard == "atomic":  # a read of an `_Atomic` object (CF-ATOMIC): one atomic load
            return deftmp(c.wr[0], _atomic_object(lf, c, ref, et, len(c.rd), stride_at=2), et)
        off = c.imm[0] if c.imm else 0
        t = ref(c.wr[0])
        if len(c.rd) == 2:  # base[index]
            if c.imm:  # s.arr[i]: member offset + element-scaled
                es = (
                    c.imm[1] if len(c.imm) > 1 else 4
                )  # index -> &base + off + i*stride, copy `es` bytes
                stride = (
                    c.imm[2] if len(c.imm) > 2 else es
                )  # array-of-structs `arr[i].field`: stride sizeof(elem)
                bp = _base_ptr(lf, c.rd[0], ref)  # != the field copy size `es`
                if c.volatile:  # a volatile element: one access of exactly its type
                    return deftmp(
                        c.wr[0],
                        f"*({_volatile_ptr(et)})((const volatile char *){bp} + {off} + "
                        f"(size_t){ref(c.rd[1])} * {stride})",
                        et,
                    )
                return (
                    f"{et} {t}; memcpy(&{t}, (const char *){bp} + {off} + "
                    f"(size_t){ref(c.rd[1])} * {stride}, {es});"
                )
            return deftmp(
                c.wr[0], f"{ref(c.rd[0])}[{_idx(lf, c, ref)}]", et
            )  # typed array (masked -> guarded)
        ptr = _base_ptr(lf, c.rd[0], ref)
        if c.volatile:
            # a volatile member / dereference: one ordered access of exactly the accessed type (a
            # bitfield: its storage unit's width)
            at = _unit_ctype(c.imm[1]) if len(c.imm) > 1 else et
            return deftmp(
                c.wr[0], f"*({_volatile_ptr(at)})((const volatile char *){ptr} + {off})", et
            )
        if len(c.imm) > 1:  # a (non-MMIO) BITFIELD unit: read only `imm[1]`
            return f"{et} {t} = 0; memcpy(&{t}, (const char *){ptr} + {off}, {c.imm[1]});"  # spanned bytes (zeroed)
        # plain RAM member/deref: memcpy is alignment-safe (handles packed) — Clang folds it to a load.
        return f"{et} {t}; memcpy(&{t}, (const char *){ptr} + {off}, sizeof {t});"
    if c.op == "c.store":
        # a write of an `_Atomic` object (CF-ATOMIC): one atomic store of exactly its slot's type (a
        # typed element: the element's own type)
        if c.hazard == "atomic":
            naddr = len(c.rd) - 1
            if naddr == 2 and not c.imm:
                slot = _elem_ctype(lf, c.rd[0])
            else:
                slot = _slot_ctype(
                    lf.rid_types.get(c.rd[-1]),
                    c.imm[1] if len(c.imm) > 1 else 4,
                    len(c.imm) > 2 and bool(c.imm[2]),
                )
            return f"{_atomic_object(lf, c, ref, slot, naddr, stride_at=3)} = {ref(c.rd[-1])};"
        off = c.imm[0] if c.imm else 0
        if len(c.rd) == 3:  # base[index] = value
            if c.imm:  # s.arr[i] = v: &base + off + i*stride, copy `es` bytes
                es = c.imm[1] if len(c.imm) > 1 else 4
                stride = (
                    c.imm[3] if len(c.imm) > 3 else es
                )  # array-of-structs `arr[i].field=v`: stride sizeof(elem)
                bp = _base_ptr(lf, c.rd[0], ref)
                if c.volatile:  # a volatile element: one store of exactly its slot type
                    slot = _slot_ctype(
                        lf.rid_types.get(c.rd[2]), es, len(c.imm) > 2 and bool(c.imm[2])
                    )
                    return (
                        f"*({_volatile_ptr(slot)})((volatile char *){bp} + {off} + "
                        f"(size_t){ref(c.rd[1])} * {stride}) = {ref(c.rd[2])};"
                    )
                dst = f"(char *){bp} + {off} + (size_t){ref(c.rd[1])} * {stride}"
                conv = (
                    "_Bool"
                    if (len(c.imm) > 2 and c.imm[2])
                    else _store_conv(lf.rid_types.get(c.rd[2]), es)
                )
                if conv:  # convert the source to the element type first (a
                    return f"{{ {conv} _sv = {ref(c.rd[2])}; memcpy({dst}, &_sv, {es}); }}"  # _Bool normalizes), else
                return f"memcpy({dst}, &{ref(c.rd[2])}, {es});"  # `es` bytes of a narrower/float source corrupts it
            return f"{ref(c.rd[0])}[{_idx(lf, c, ref, write=True)}] = {ref(c.rd[2])};"  # typed array (masked -> WRITE-guarded)
        ptr = _base_ptr(lf, c.rd[0], ref)
        size = c.imm[1] if len(c.imm) > 1 else 4
        if c.volatile:
            # a volatile member / dereference: one ordered store of exactly its slot type -- never a
            # 32-bit register by assumption
            vt = lf.rid_types.get(c.rd[1])
            if vt is not None and vt.kind == "funcptr":
                return (
                    f"*(void (* volatile *)(void))((volatile char *){ptr} + {off}) = "
                    f"(void (*)(void)){ref(c.rd[1])};"
                )
            slot = _slot_ctype(vt, size, len(c.imm) > 2 and bool(c.imm[2]))
            return f"*({_volatile_ptr(slot)})((volatile char *){ptr} + {off}) = {ref(c.rd[1])};"
        # plain RAM member/deref: memcpy `size` bytes (correct truncation on little-endian, packed-safe). A
        # source whose type does not match the slot is CONVERTED through a slot-typed temp first -- a narrower
        # int (`*p = (short)b`) must widen, and a `float` stored into a `double` member must convert (not
        # memcpy 4 bytes into 8, nor copy float bits into a double slot).
        vt = lf.rid_types.get(c.rd[1])
        if vt is not None and vt.kind == "funcptr":  # a funcptr member set from a funcptr value / a
            # function NAME: store through a GENERIC funcptr lvalue so the name decays to its address (a plain
            # `memcpy(&g_func,8)` copies the function's CODE; `void *` can't hold a funcptr). The call site reads
            # the member's real type, and function pointers round-trip through the cast.
            return f"*(void (**)(void))((char *){ptr} + {off}) = (void (*)(void)){ref(c.rd[1])};"
        conv = (
            "_Bool"
            if (len(c.imm) > 2 and c.imm[2])
            else _store_conv(lf.rid_types.get(c.rd[1]), size)
        )
        if conv:  # a _Bool slot normalizes the stored value to 0/1
            return (
                f"{{ {conv} _sv = {ref(c.rd[1])}; memcpy((char *){ptr} + {off}, &_sv, {size}); }}"
            )
        return f"memcpy((char *){ptr} + {off}, &{ref(c.rd[1])}, {size});"
    if c.op == "c.bf.get":  # (unit >> bit_off) & mask (sign-extended if signed)
        bit_off, width = c.imm[0], c.imm[1]
        rt = lf.rid_types.get(c.wr[0])  # a WIDE bitfield (> 32b) needs 64-bit literals/cast
        wide = rt is not None and rt.size > 4
        sfx = "ull" if wide else "u"
        mask = (1 << width) - 1
        if len(c.imm) > 2 and c.imm[2]:  # a signed bitfield: sign-extend from bit width-1
            sbit, cast = 1 << (width - 1), ("int64_t" if wide else "int32_t")
            return deftmp(
                c.wr[0],
                f"({cast})(((({ref(c.rd[0])} >> {bit_off}) & {mask}{sfx}) ^ {sbit}{sfx}) - {sbit}{sfx})",
            )
        return deftmp(c.wr[0], f"({ref(c.rd[0])} >> {bit_off}) & {mask}{sfx}")
    if c.op == "c.bf.set":  # (old & ~(mask<<off)) | ((v&mask)<<off)
        bit_off, width = c.imm
        rt = lf.rid_types.get(c.wr[0])
        wide = rt is not None and rt.size > 4  # 64-bit unit: a 64-bit clear mask, not a 32-bit one
        sfx = "ull" if wide else "u"
        mask = (1 << width) - 1
        clear = ~(mask << bit_off) & ((1 << 64) - 1 if wide else 0xFFFFFFFF)
        return deftmp(
            c.wr[0],
            f"({ref(c.rd[0])} & {clear}{sfx}) | (({ref(c.rd[1])} & {mask}{sfx}) << {bit_off})",
        )
    if c.op.startswith("c.call.libm:"):  # a <math.h> / <stdlib.h> call -> the real function
        callee = c.op.split(":", 1)[1]  # (no bcir_ twin; the harness links -lm/libc)
        rt = lf.rid_types.get(c.wr[0])  # declare at the true result width: a long
        ty = _cname(rt) if rt is not None else None  # return (lround) is not narrowed to uint32
        return deftmp(c.wr[0], f"{callee}({', '.join(ref(r) for r in c.rd)})", ty)
    if c.op.startswith("c.call.libm.void:"):  # a void external (free) -- verbatim statement, opaque
        callee = c.op.split(":", 1)[1]
        return f"{callee}({', '.join(ref(r) for r in c.rd)});"
    if (
        c.op == "c.asm:" or c.op == "c.asm.volatile:"
    ):  # inline assembly (ASM1): a TRUSTED OPAQUE EFFECT
        return _asm_stmt(lf, c, ref)  # EDGE -- re-emit the GNU __asm__ statement VERBATIM
    if c.op.startswith("c.portio."):  # port-mapped I/O (ASM2): the real x86 in/out
        return _portio_stmt(lf, c, ref)  # instruction, keyed off the target (non-x86 raises)
    if c.op.startswith("c.call.extern:"):  # a printf/scanf-family external variadic call
        callee = c.op.split(":", 1)[1]  # emitted verbatim against <stdio.h>, returns int
        return deftmp(c.wr[0], f"{callee}({', '.join(ref(r) for r in c.rd)})", "int")
    if c.op.startswith("c.call.tu:"):  # a PROTOTYPED cross-TU callee (Phase 3 linking):
        callee = c.op.split(":", 1)[1]  # verbatim, external linkage -- the emitted TU
        if not c.wr:  # declares it; the host LINKER resolves it
            return f"{callee}({_args(lf, ref, c.rd)});"
        rt = lf.rid_types.get(c.wr[0])
        return deftmp(
            c.wr[0],
            f"{callee}({_args(lf, ref, c.rd)})",
            _cname(rt) if rt is not None else None,
        )
    if c.op.startswith("c.call.builtin:"):  # a GCC/Clang integer builtin -> verbatim
        callee = "__builtin_" + c.op.split(":", 1)[1]  # the op stores the suffix; re-add the prefix
        rt = lf.rid_types.get(c.wr[0])
        return deftmp(
            c.wr[0],
            f"{callee}({', '.join(ref(r) for r in c.rd)})",
            _cname(rt) if rt is not None else None,
        )
    if c.op == "c.call.vaarg":  # va_arg(ap, T) -- T is the result temp's type
        rt = lf.rid_types.get(c.wr[0])
        ty = _cname(rt) if rt is not None else "int"
        return deftmp(c.wr[0], f"va_arg({ref(c.rd[0])}, {ty})", ty)
    if c.op.startswith("c.call.vabuiltin:"):  # va_start / va_end / va_copy -- verbatim, void
        callee = c.op.split(":", 1)[1]
        return f"{callee}({', '.join(ref(r) for r in c.rd)});"
    if c.op.startswith("c.call.void:"):  # a void callee -> a bare call statement
        callee = c.op.split(":", 1)[1]
        return f"bcir_{callee}({_args(lf, ref, c.rd)});"
    if c.op.startswith("c.call:"):
        callee = c.op.split(":", 1)[1]
        rt = lf.rid_types.get(c.wr[0])  # a wide (8-byte) int OR an aggregate return declares
        ty = (
            _cname(rt)
            if (
                rt is not None
                and (
                    rt.is_aggregate  # at its true type (`struct P t =
                    or (rt.is_integer and rt.size > 4)
                )
            )
            else None
        )  # mk(x);`), not uint32
        return deftmp(c.wr[0], f"bcir_{callee}({_args(lf, ref, c.rd)})", ty)
    if c.op == "c.call.indirect":  # rd[0] is the function pointer; rd[1:] args
        call = f"{ref(c.rd[0])}({_args(lf, ref, c.rd[1:])})"
        return deftmp(c.wr[0], call) if c.wr else f"{call};"  # a void function: a bare call
    if c.op.startswith("c.call.imember:"):  # o->fn(args): funcptr struct member
        field = c.op.split(":", 1)[1]
        sep = "->" if c.imm and c.imm[0] else "."
        call = f"{ref(c.rd[0])}{sep}{field}({_args(lf, ref, c.rd[1:])})"
        return deftmp(c.wr[0], call) if c.wr else f"{call};"
    if c.op.startswith("c.atomic."):  # atomic RMW -> the matching builtin (§5.8)
        return deftmp(
            c.wr[0],
            f"__atomic_fetch_{c.op.split('.')[-1]}("
            f"{ref(c.rd[0])}, {ref(c.rd[1])}, __ATOMIC_SEQ_CST)",
        )
    if c.op.startswith("c.cmpxchg."):  # compare-and-swap -> the __sync CAS builtin
        return deftmp(
            c.wr[0],
            f"__sync_{c.op.split('.')[-1]}_compare_and_swap("
            f"{ref(c.rd[0])}, {ref(c.rd[1])}, {ref(c.rd[2])})",
        )
    if c.op == "c.fence" or c.op.startswith(
        "c.fence."
    ):  # memory fence (ASM3): the real per-ISA hardware
        return _fence_stmt(lf, c)  # barrier behind --target ("memory" compiler barrier)
    # a compound assignment / inc / dec of an `_Atomic` object (CF-ATOMIC): one atomic
    # read-modify-write through an `_Atomic` lvalue, its value the expression's
    if c.op.startswith("c.c11atom.rmw:"):
        kind = c.op.split(":", 1)[1]
        et = _load_ctype(lf, c.wr[0])
        incdec = _RMW_INCDEC.get(kind)
        obj = _atomic_object(lf, c, ref, et, len(c.rd) - (0 if incdec else 1), stride_at=3)
        expr = incdec.format(obj) if incdec else f"{obj} {_BINOP[kind]}= {ref(c.rd[-1])}"
        return deftmp(c.wr[0], f"({expr})", et)
    if c.op.startswith("c.c11atom."):  # C11 <stdatomic.h> generics on _Atomic objects
        fn = c.op.split(".")[-1]  # fetch_add / fetch_sub / fetch_xor / load / store
        if fn == "load":
            return deftmp(c.wr[0], f"atomic_load({ref(c.rd[0])})")
        if fn == "store":
            return f"atomic_store({ref(c.rd[0])}, {ref(c.rd[1])});"
        if fn.startswith("cas_"):  # cas_strong/weak -> bool compare_exchange (obj, &exp, des)
            return deftmp(
                c.wr[0],
                f"atomic_compare_exchange_{fn[4:]}({ref(c.rd[0])}, {ref(c.rd[1])}, {ref(c.rd[2])})",
            )
        return deftmp(c.wr[0], f"atomic_{fn}({ref(c.rd[0])}, {ref(c.rd[1])})")
    raise ValueError(f"emit: unhandled claim op {c.op!r}")


def _args(lf: LoweredFunc, ref, rids) -> str:
    """A call's arguments as the emit spells them. A file-scope multi-dimensional array is declared as the source
    declares it, nested, so its name decays to a pointer to its first row (`T (*)[N]`); an emitted parameter
    that takes an array is the flat `T *` (a local array is declared flat already). Such an argument is spelled
    as its first element's address, `&m[0][0]` -- the same address, of the parameter's type. The twin spells
    it the same (`bcir_cfront.c`, `emit_arg`)."""
    out = []
    for r in rids:
        ct = lf.rid_types.get(r)
        dims = 0
        while ct is not None and ct.kind == "array":
            dims, ct = dims + 1, ct.of
        out.append(f"&{ref(r)}" + "[0]" * dims if r in lf.globals_used and dims > 1 else ref(r))
    return ", ".join(out)


def _unit_ctype(size: int) -> str:
    """The unsigned type of a `size`-byte storage unit (a bitfield's), `uint32_t` if not a width C has."""
    return f"uint{size * 8}_t" if size in (1, 2, 4, 8) else "uint32_t"


def _slot_ctype(vt, size: int, is_bool: bool) -> str:
    """The C type of the `size`-byte slot a store of a `vt` value writes, decided the way the plain
    store decides it (`_store_conv`): a `_Bool` slot, a float of the slot's width, a pointer or an
    aggregate as itself, else the unsigned integer of the slot's width. The C twin decides the same
    (`bcir_cfront.c`, `vol_slot_ty`). A volatile store writes exactly this type -- one access of the slot's
    width, never a 32-bit register by assumption."""
    if is_bool:
        return "_Bool"
    if vt is not None and vt.is_complex:
        return (
            "float _Complex"
            if size == 8
            else ("long double _Complex" if size > 16 else "double _Complex")
        )
    if vt is not None and vt.is_float:
        return "float" if size == 4 else ("long double" if size > 8 else "double")
    if vt is not None and (vt.kind == "pointer" or vt.is_aggregate):
        return _cname(unqualified(vt))
    return _unit_ctype(size)


def _store_conv(vt, size: int):
    """The C type to convert a store source through (a slot-typed temp) before memcpy'ing `size` bytes into
    the slot -- or None when the source already matches the slot width/kind (a direct memcpy is correct). A
    `float`/`double` source of a different width must convert (float<->double, not a byte copy), a complex one
    to the complex type of the slot's width; a narrower integer source must widen/sign-extend to the slot
    width. A source of another arithmetic class was already converted by the lowering (`c.cast`). An ARRAY
    source decays to its address (C11 6.3.2.1p3): the slot takes the pointer, staged in a pointer object --
    `&arr` would copy the array's first bytes instead (CF-MEMDECAY)."""
    if vt is not None and vt.kind == "array":
        return "const volatile void *"
    if vt is not None and vt.is_complex and vt.size != size:
        return (
            "float _Complex"
            if size == 8
            else ("long double _Complex" if size > 16 else "double _Complex")
        )
    if vt is not None and vt.is_float and vt.size != size:
        return "float" if size == 4 else ("long double" if size > 8 else "double")
    if vt is not None and vt.is_integer and vt.size < size:
        return f"uint{size * 8}_t"
    return None


def _load_ctype(lf: LoweredFunc, rid: int) -> str:
    ct = lf.rid_types.get(rid)
    # a pointer member/deref load carries its real `T *` type, so `sizeof t` reads pointer_size bytes
    # (not 4) and the loaded value is a usable pointer -- a uint32 would truncate it. A float/double load
    # likewise keeps its real type: a `uint32_t` temp would reinterpret-truncate the value (and drop 4 of
    # a double's 8 bytes on the memcpy `sizeof t`), so a `float[]`/double member/deref reads as itself.
    # A struct or union value -- an element of an array of structs, a struct member, `*p`, a select of two
    # structs -- is a temp of the aggregate itself, copied whole (CF-STRUCTVAL): `uint32_t t = ps[i];` did
    # not compile. The twin spells it the same (`bcir_cfront.c`, `tty`).
    return (
        _cname(ct)
        if ct and (ct.is_integer or ct.is_float or ct.kind == "pointer" or ct.is_aggregate)
        else "uint32_t"
    )


def _base_ptr(lf: LoweredFunc, rid: int, ref) -> str:
    """A pointer to the base resource: the name decays if it's a pointer/array, else address-of."""
    ct = lf.rid_types.get(rid)
    name = ref(rid)
    if ct and ct.kind in ("pointer", "array"):
        return name
    return f"&{name}"


def _idx(lf: LoweredFunc, c, ref, *, write: bool = False) -> str:
    """The index expression for a `base[idx]` access. A `masked` access (§5.12 bounds-promotion) into a
    known-extent local/static array is wrapped in a bounds guard: in-bounds returns idx (transparent ->
    behaviour-identical to the raw `a[i]`), out-of-bounds calls the bounds-quarantine handler. The numeric
    `rid` is the access provenance and `"<func>:<array>"` is the source-site handle the debugger / ML-layer
    reads (a site->source table realized inline). Any other access -> the bare index.

    READ vs WRITE matters (§5.12): a READ index site uses `BCIR_CHK` (the handler MAY clamp an OOB read to a
    valid element -- a load mutates nothing); a WRITE index site uses `BCIR_CHK_W`, whose handler is
    `noreturn` and NEVER clamps -- a clamped OOB store would silently redirect the write onto a valid element
    (a[extent-1]) and corrupt it, so an OOB store always fails-fast. `write=True` for c.store index sites."""
    chk = "BCIR_CHK_W" if write else "BCIR_CHK"
    idx = ref(c.rd[1])
    if c.bounds == "masked":
        ext = lf.ptr_extent.get(c.rd[0])
        if ext is not None:  # a naked pointer with a RECOVERED runtime extent
            return f'{chk}({c.rd[0]}, {idx}, {ref(ext)}, "{lf.name}:{ref(c.rd[0])}")'
        rt = lf.rid_types.get(c.rd[0])
        n = getattr(rt, "count", 0) if rt is not None else 0
        if n:  # a known-extent local/static array (constant N)
            return f'{chk}({c.rd[0]}, {idx}, {n}u, "{lf.name}:{ref(c.rd[0])}")'
    return idx
