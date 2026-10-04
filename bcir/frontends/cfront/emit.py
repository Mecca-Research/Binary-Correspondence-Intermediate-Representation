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
from .ctype_model import CType, pointer, unqualified
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
    _const_spelling,
    _kspell,
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
    Never a byte copy: a `memcpy` of an atomic object is not an atomic access, and it tears.

    The indexed form reaches the element through its own pointer type, `(*((_Atomic T *)((char *)base +
    off) + (size_t)idx * stride/es))`, not through byte arithmetic cast at the end (TC23): Clang 23 gives
    a pointer computed from a *defined* object by `char` arithmetic with a runtime index the alignment of
    `char`, and an atomic access it cannot prove aligned becomes a libatomic call (`__atomic_load`,
    `__atomic_store`; undefined in a freestanding link) where Clang 18 inlined one instruction. Typed
    arithmetic from the constant-offset base keeps the element's alignment on every Clang and GCC; the
    byte form remains for a stride the element size does not divide (no lock-free atomic has one). The
    twin spells it the same (`bcir_cfront.c`, `atomic_object`)."""
    ptr = _atomic_ptr(t, c.volatile)
    if naddr == 2 and not c.imm:
        return (
            f"(*({ptr})&{_elem_base(lf, c.rd[0], ref)}[{_idx(lf, c, ref, write=c.op != 'c.load')}])"
        )
    bp = _base_ptr(lf, c.rd[0], ref)
    off = c.imm[0] if c.imm else 0
    if naddr == 2:
        es = c.imm[1] if len(c.imm) > 1 else 4
        stride = c.imm[stride_at] if len(c.imm) > stride_at else es
        if es > 0 and stride % es == 0:
            # the element's pointer from the aligned base, then arithmetic in its own units (TC23)
            return f"(*(({ptr})((char *){bp} + {off}) + (size_t){ref(c.rd[1])} * {stride // es}))"
        return f"(*({ptr})((char *){bp} + {off} + (size_t){ref(c.rd[1])} * {stride}))"
    return f"(*({ptr})((char *){bp} + {off}))"


def _elem_ctype(lf: LoweredFunc, rid: int) -> str:
    """The C type of an element of the pointer or array `rid`, unqualified: the slot an atomic store
    into a typed element `base[idx]` writes."""
    ct = lf.rid_types.get(rid)
    el = ct.of if ct is not None and ct.of is not None else None
    return _cname(unqualified(el)) if el is not None else "uint32_t"


def _has_fp(ct: "CType | None") -> bool:
    """Whether a function pointer is inside `ct` -- it is one, or a pointer to or an array of one at any
    depth (CF-FPTAB) -- so that only a declarator around a name spells it (`_funcptr_decl`), never a type
    name before it."""
    while ct is not None and ct.kind in ("pointer", "array"):
        ct = ct.of
    return ct is not None and ct.kind == "funcptr"


def _qual_word(quals: tuple) -> str:
    """The qualifiers of one level of a declarator, as written after its `*` (`const `), or nothing."""
    return " ".join(quals) + " " if quals else ""


def _funcptr_decl(ct: CType, name: str, sig: tuple = ()) -> str:
    """Render an inline function-pointer declarator `RET (*name)(PARAMS)` -- and around it the pointers and
    dimensions of a type that holds one (CF-FPTAB): an array of them `RET (*name[N])(PARAMS)`, a pointer to
    one `RET (**name)(PARAMS)`. Used in param, local and temp position, where (unlike a typedef alias)
    there is no spelling to print and the full signature must be reconstructed from the carried return +
    parameter types -- `int (*g)(int)`. With no name it is the type's own spelling (a prototype's). Its
    return and parameters keep their qualifiers (the function type's `fquals`), and `sig` gives the
    qualifiers of the function pointer and of the `*`s around it below their top level, `RET (**const
    *pp)(PARAMS)` (`lower._qual_sig`, CF-QUALS)."""
    depth, at = 0, ct
    while at.kind in ("pointer", "array"):
        depth += at.kind == "pointer"
        at = at.of
    inner, ptr_last, level = name, False, depth
    while ct.kind in ("pointer", "array"):
        if ct.kind == "pointer":
            # the outermost `*` is the object's own level; each one inside keeps its qualifiers
            quals = sig[level] if level < len(sig) else ()
            inner, ptr_last, level = f"*{_qual_word(quals)}{inner}", True, level - 1
        else:  # a dimension binds tighter than a `*`: a pointer to an array keeps its parentheses
            inner, ptr_last = (
                (f"({inner})[{ct.count}]" if ptr_last else f"{inner}[{ct.count}]"),
                False,
            )
        ct = ct.of
    rsig, psigs = ct.fquals or ((), ())
    if ct.of is None:
        ret = "void"
    else:  # a function pointer it returns is spelled by its typedef (CF-FPRET)
        ret = _cname(ct.of) if _has_fp(ct.of) else _qual_type(ct.of, rsig)
    plist = [_proto_param(p, psigs[k] if k < len(psigs) else ()) for k, p in enumerate(ct.params)]
    params = ", ".join(plist + (["..."] if ct.variadic else [])) or "void"
    # the function pointer's own qualifiers, below a `*` around it
    own = sig[0] if sig and depth else ()
    return f"{ret} (*{_qual_word(own)}{inner})({params})"


def _fp_store(lf, dst: str, rid: int, ref) -> str:
    """A function pointer stored at `dst`, a byte address: through a pointer to the value's own function-pointer
    type, `*(RET (**)(PARAMS))(dst) = v;` -- the member's, so the store and a read of the member through its
    declared type access one object as one type (C11 6.5p7). A store through a generic `void (**)(void)` slot,
    read back as the member's `op_t`, is undefined, and GCC at -O2 dropped it: the emitted function called a null
    pointer (CF-RTFP). A function's name decays to its address, as in any assignment, and is typed by its whole
    function type (`LoweredFunc.fn_types`); one whose parameters are not yet typed keeps the generic slot."""
    vt = lf.fn_types[rid] if rid in lf.fn_types else lf.rid_types.get(rid)
    if vt is None:
        return f"*(void (**)(void))({dst}) = (void (*)(void)){ref(rid)};"
    return f"*({_funcptr_decl(pointer(vt), '')})({dst}) = {ref(rid)};"


def _qual_type(ct: CType, sig: tuple) -> str:
    """The type `ct` as a function type spells a parameter or return of it: with the qualifiers `sig` keeps at
    each level below its top (`lower._qual_sig`) -- `const char *const *` -- where the emit's own objects
    are spelled without them (`_cname`) and meet it through a cast (`_qual_arg`, `_qual_result`). A function
    pointer's own `RET (*)(PARAMS)`, its parameters qualified as its type says (CF-QUALS)."""
    if _has_fp(ct):
        return _funcptr_decl(ct, "", sig)
    if not sig:
        return _cname(ct)
    stars, at = 0, ct
    while at.kind == "pointer":
        stars, at = stars + 1, at.of
    out = ("const " if "const" in sig[0] else "") + _cname(at)
    for level in range(1, stars + 1):
        out += " *" + (" " + " ".join(sig[level]) if level < len(sig) and sig[level] else "")
    return out


_ANON_REF = re.compile(r"(?<![A-Za-z0-9_$])(?:struct|union) (\$anon\d+)(?![A-Za-z0-9_$])")
# The headers declaring what the linkable emit's own text names -- read outside literals and comments
# (`_code_only`) -- where no claim names it as a callee: a `size_t` temp (CF-SIZEOF) and a `wchar_t` element
# (CF-STRELEM), the copies the emit makes, a variadic function's `va_list`, the C11 atomics and the complex
# functions and `_Complex_I` it spells (CF-LINKEMIT)
_STDDEF_NAME = re.compile(r"\b(?:size_t|ptrdiff_t|wchar_t|max_align_t)\b")
_MEM_CALL = re.compile(r"\b(?:memcpy|memmove|memset)\s*\(")
_OWN_HEADERS = (
    ("<stdarg.h>", re.compile(r"\b(?:va_list|va_start|va_arg|va_end|va_copy)\b")),
    ("<stdatomic.h>", re.compile(r"\b(?:atomic_[a-z_]+\s*\(|memory_order_[a-z_]+\b)")),
    (
        "<complex.h>",
        re.compile(
            r"\b(?:_Complex_I\b|(?:c(?:abs|arg|imag|real|proj|exp|log|pow|sqrt|a?sinh?|a?cosh?|a?tanh?)|conj)[fl]?\s*\()"
        ),
    ),
)
# The names an emitted function spells for itself, whatever its source names: the <string.h> and <stdlib.h> routines
# it calls or copies through (`memcpy` spells every plain member or element store), the C11 atomics it calls, the
# twin's store helper `_v`, and the standard type names it declares objects with. No parameter or local the emit
# declares takes one (`_spelled_names`; CF-TYPEDEFSCOPE). The twin's `emit_spelled` (`bcir_cfront.c`) holds the same
# names -- a test reads both lists out of their sources.
_EMIT_SPELLED = frozenset(
    (
        "_v",
        "aligned_alloc",
        "atomic_compare_exchange_strong",
        "atomic_compare_exchange_weak",
        "atomic_exchange",
        "atomic_fetch_add",
        "atomic_fetch_sub",
        "atomic_fetch_xor",
        "atomic_load",
        "atomic_store",
        "atomic_thread_fence",
        "calloc",
        "char16_t",
        "char32_t",
        "free",
        "int16_t",
        "int32_t",
        "int64_t",
        "int8_t",
        "intmax_t",
        "intptr_t",
        "malloc",
        "memcpy",
        "memmove",
        "memset",
        "ptrdiff_t",
        "realloc",
        "size_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "uint8_t",
        "uintmax_t",
        "uintptr_t",
        "va_list",
        "wchar_t",
    )
)
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


def _code_only(text: str) -> str:
    """`text` without its string and character literals and its comments: what it names as code."""
    return _LITERAL_OR_COMMENT.sub(" ", text)


def _object_declarator(ct: CType, name: str, quals: tuple = ()) -> str:
    """An object's declaration without its initializer: `T name`, an array's every dimension after its name
    (`T name[2][3]`, the element type the innermost; an unsized one `[]`). `quals` are the qualifiers of its type
    level by level from the base type out (`lower._object_quals`), which the linkable emit's globals keep:
    `const char *const names[2]` (CF-LINKEMIT)."""
    # a function pointer, a table of them: the declarator around the name (CF-FPTAB)
    if _has_fp(ct):
        return _funcptr_decl(ct, name)
    dims = []
    while ct.kind == "array":
        dims.append(f"[{ct.count}]" if ct.count else "[]")
        ct = ct.of
    return f"{_qual_type(ct, quals)} {name}{''.join(dims)}"


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
    defs = list(getattr(lowered, "type_defs", ()))
    spelling = getattr(lowered, "anon_spelling", {})
    # what the artifact itself names
    code = _code_only("\n".join([emitted[n] for n in names] + defs))
    parts: list[str] = ["#include <stdint.h>"]  # the emitted temps are int32_t/uint32_t/...
    # a `size_t` temp (CF-SIZEOF), a `wchar_t` literal element (CF-STRELEM)
    if _STDDEF_NAME.search(code):
        parts.append("#include <stddef.h>")
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
    # the memcpy/memmove/memset edges, and the emit's own copies
    if callees & _STRING_MEM or _MEM_CALL.search(code):
        parts.append("#include <string.h>")
    parts += [f"#include {h}" for h, names_re in _OWN_HEADERS if names_re.search(code)]
    # `I`, which the lowering read as <complex.h>'s imaginary unit
    if any(op.startswith("c.cconst:") for op in ops) and "#include <complex.h>" not in parts:
        parts.append("#include <complex.h>")
    if any(op.startswith("c.call.extern:") for op in ops):
        parts.append("#include <stdio.h>")  # the printf/scanf-family edges
    if callees - libc - _STRING_MEM and any(
        op.startswith(("c.call.libm:", "c.call.libm.void:")) for op in ops
    ):
        parts.append("#include <math.h>")  # the remaining libm edges
    # the types the artifact names, as the source defines them, in its order (CF-LINKEMIT)
    parts += defs
    # every function declared before the globals: a static one may be DEFINED after its caller (the C
    # static-forward-declaration idiom), and a global's initializer may name any (an ops table, CF-LINKEMIT); a
    # function pointer parameter is its declarator with no name
    for name in names:
        lf = lowered.functions[name]
        ps = ", ".join(_proto_param(p[2]) for p in lf.params) or "void"
        if lf.variadic:
            ps = (ps + ", ...") if ps != "void" else "..."
        kw = "static " if lf.static_fn else ""
        parts.append(respell_anon(f"{kw}{_cname(lf.ret_type)} {name}({ps});", spelling))
    threads = getattr(lowered, "thread_globals", frozenset())
    quals = getattr(lowered, "global_quals", {})
    for gname, ct, vals, is_extern, is_static in lowered.globals_decl:
        tls = "_Thread_local " if gname in threads else ""  # each thread's own object (CF-TLS)
        decl = respell_anon(_object_declarator(ct, gname, quals.get(gname, ())), spelling)
        if _ANON_REF.search(decl):  # an untagged aggregate no typedef names: C names no type of it
            raise ValueError(
                f"linkable emit: global {gname!r} has a type C cannot name (an untagged aggregate)"
            )
        if is_extern:
            parts.append(f"extern {tls}{decl};")
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
        parts.append(f"{kw}{tls}{decl}{init};")
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

    def __init__(self, named: dict, used: set, labels: dict | None = None):
        self.named, self.used, self.temps = named, used, {}
        self.labels = labels or {}  # rid -> the declared name where `named` spells an expression

    def __call__(self, rid: int) -> str:
        if rid in self.named:
            return self.named[rid]
        if rid not in self.temps:
            name, k = f"t{rid}", 2
            while name in self.used:
                name, k = f"t{rid}_{k}", k + 1
            self.temps[rid] = name
        return self.temps[rid]

    def label(self, rid: int) -> str:
        """The rid's name as a source-site handle names it: a global's own, where the code reaches it through an
        expression (`_unqualified_global`)."""
        return self.labels.get(rid) or self(rid)

    def is_temp(self, rid: int) -> bool:
        """An intermediate, declared where it is written, not a named object declared up front."""
        return rid not in self.named


def _signature(lf: LoweredFunc, names: dict | None = None) -> str:
    """The function's signature as its definition spells it, `static RET bcir_NAME(PARAMS)`: the
    definition's head, and the forward declaration of it that a caller's emit makes. `names` -- rid -> the
    name the definition gives a parameter (`emit_function`), where it is not the source's."""
    names = names or {}
    parts = []
    for pname, rid, ct in lf.params:
        name = names.get(rid, pname)
        parts.append(_funcptr_decl(ct, name) if _has_fp(ct) else f"{_cname(ct)} {name}")
    if lf.variadic:  # a trailing `...` after the named params
        parts.append("...")
    return f"static {_cname(lf.ret_type)} bcir_{lf.name}({', '.join(parts) or 'void'})"


def _proto_param(ct: CType, sig: tuple = ()) -> str:
    """A parameter of a function type as it spells it: a function pointer as its declarator with no name,
    `RET (*)(PARAMS)` (its name alone does not compile), and every qualifier below its top level kept
    (`_qual_type`) -- a parameter of `const T *` or `T *const *` is another type than one of `T *` or `T **`,
    and two declarations of the callee would conflict (CF-QUALS). The twin spells it alike (`bcir_cfront.c`,
    `ctype_qstr`)."""
    return _qual_type(ct, sig)


def _extern_decl(callee: str, rct: CType, pcts: tuple, quals: tuple, va: bool) -> str:
    """The `extern` declaration of a function another unit defines, as its prototype declares it: its
    return and every parameter with the qualifiers each keeps below its top level (CF-QUALS), a variadic
    one with its `...` (CF-EXTDESIG). A function pointer it returns is spelled by its typedef (CF-FPRET)."""
    rsig, psigs = quals or ((), ())
    params = [_proto_param(p, psigs[k] if k < len(psigs) else ()) for k, p in enumerate(pcts)]
    ret = _cname(rct) if _has_fp(rct) else _qual_type(rct, rsig)
    return (
        f"extern {ret} {callee}(" + (", ".join(params + (["..."] if va else [])) or "void") + ");"
    )


def _below_top(ct: "CType | None", sig: tuple, top: int) -> bool:
    """Whether the type `ct` of a function type's parameter or return keeps a qualifier the emit's own
    objects do not spell, below its level `top`: an argument passed to it, or a result taken from it, meets
    the emit's unqualified type through a cast (`_qual_arg`, `_qual_result`). A function pointer's type is
    spelled whole wherever it is, so it never needs one."""
    if ct is None or not sig or _has_fp(ct):
        return False
    return any(sig[k] for k in range(min(top, len(sig))))


def _qual_args(lf: LoweredFunc, ref, rids, params: tuple, fquals: tuple) -> str:
    """A call's arguments (`_args`), each passed to a parameter whose type keeps a qualifier more than one level
    below its top -- `const char **`, `const char *const *`, which C converts no `char **` to (6.5.16.1p1) --
    cast to that parameter's type: the emit's own objects are spelled without qualifiers (CF-QUALS). One level
    down C adds them itself (`const T *` from a `T *`)."""
    psigs = fquals[1] if fquals else ()
    out = _arg_list(lf, ref, rids)
    for k, arg in enumerate(out):
        if k < len(params) and k < len(psigs):
            pct, sig = params[k], psigs[k]
            depth = 0
            at = pct
            while at is not None and at.kind == "pointer":
                depth, at = depth + 1, at.of
            if _below_top(pct, sig, depth - 1):
                out[k] = f"({_qual_type(pct, sig)}){arg}"
    return ", ".join(out)


def _qual_result(call: str, ret: "CType | None", fquals: tuple) -> str:
    """A call whose function type returns a pointer to a qualified type, `const T *`, taken by the emit's own
    unqualified temp through a cast (CF-QUALS)."""
    rsig = fquals[0] if fquals else ()
    if ret is None or not rsig or _has_fp(ret) or ret.kind != "pointer":
        return call
    depth, at = 0, ret
    while at.kind == "pointer":
        depth, at = depth + 1, at.of
    return f"({_cname(ret)}){call}" if _below_top(ret, rsig, depth) else call


def _unqualified_global(name: str, ct: CType) -> str:
    """A global whose type is qualified at some level -- `const uint32_t k`, `const char *const names[2]` -- as
    the emit reaches it: an lvalue of its type with no qualifier, `(*(char *(*)[2])&names)`. The emit spells its
    own objects without qualifiers (`_cname`), so a pointer read out of a `const char *` table into its temp, or
    `&k` into a pointer, discarded a qualifier C diagnoses (6.5.16.1p1); through the cast it meets the object as
    a call meets a qualified parameter (`_qual_args`). A volatile or `_Atomic` object keeps its own name: each
    access of it must stay one access of its own type (CF-LINKEMIT)."""
    at, dims = ct, []
    while at.kind == "array":
        dims.append(f"[{at.count}]" if at.count else "[]")
        at = at.of
    if at.volatile or at.atomic or ct.volatile or _has_fp(at):
        return name
    ptr = f"{_cname(at)} (*){''.join(dims)}" if dims else f"{_cname(at)} *"
    return f"(*({ptr})&{name})"


def _spelled_names(lf: LoweredFunc) -> set:
    """Every name the emitted function spells other than as one of its own parameters and locals: `_EMIT_SPELLED`, a
    global or function it reads (`globals_used`), the unit's typedef names, and each function it calls as the emit
    calls it -- `bcir_X` for one the unit defines, `X` for any other (a libc routine, a function another unit defines).
    No parameter or local the emit declares may take one: declared up front for the whole function, where the source
    declared it for its block -- or, for a name the source never spelled there, at all -- it would capture the emit's
    own reference (CF-TYPEDEFSCOPE). The twin's `emit_name_taken` decides alike."""
    spelled = set(_EMIT_SPELLED) | set(lf.globals_used.values()) | set(lf.typedef_names)
    callees = {callee for callee, _args in lf.calls}
    for c in lf.claims:  # a libc routine, a function another unit defines: `c.call<kind>:X`
        if c.op.startswith("c.call") and ":" in c.op:
            callees.add(c.op.split(":", 1)[1])
    for callee in callees:
        spelled.update((callee, f"bcir_{callee}"))
    return spelled


def emit_function(lf: LoweredFunc, unit: dict | None = None) -> str:
    """The lowered function as standalone C, named `bcir_<name>` (so it can sit beside the original).
    Walks the structured body tree, so `if`/`while`/`return` emit real C control flow; mutable named
    locals are declared up front and assigned (so branch merges + loop accumulators reproduce the
    source); intermediate expression results stay single-assignment temporaries. `unit` -- the unit's
    lowered functions by name -- declares each one the function calls before it: a callee defined after
    its caller is otherwise undeclared at the call, which does not compile."""
    nm: dict[int, str] = {}
    # Each local needs a *unique* C identifier: the lowering flattens scopes, so two source locals that
    # shared a name in disjoint scopes (e.g. `i` in two separate `for` loops, or a block local shadowing
    # a param) become distinct rids with the same name. Declaring both at function scope is a C
    # redefinition. Disambiguate the second-and-later occurrences (`i`, `i_2`, ...) -- a fresh variable
    # preserves the source's separate-scope semantics; naive name-sharing would corrupt a shadowed value.
    # ... nor a name the function spells other than as one of its own objects (`_spelled_names`): the declarations up
    # front would hide it for the whole function, where the source's local hid it for its block (CF-TYPEDEFSCOPE:
    # `{ uint32_t S = s; } S v;` became `uint32_t S; S v;`, which does not compile) -- and a parameter or a local
    # named as a name the emit alone spells (`memcpy`, `bcir_g`) would capture it everywhere
    used: set[str] = _spelled_names(lf)
    local_name: dict[int, str] = {}

    def _uniq(name: str) -> str:
        uniq, k = name, 2
        while uniq in used:
            uniq, k = f"{name}_{k}", k + 1
        used.add(uniq)
        return uniq

    # a parameter first: its source name, unless the emit spells that otherwise
    for pname, rid, _ct in lf.params:
        nm[rid] = _uniq(pname)
    params = dict(nm)
    # a static too: two scopes' `static n` are two objects, both declared at function scope
    for rid, name, _ct, _init in lf.statics:
        nm[rid] = _uniq(name)
    for rid, name, _ct in lf.locals + lf.vla_locals:
        # a VLA is NAMED here (so its accesses resolve) but declared IN-BODY (the c.vladecl claim below),
        # never in the up-front `decls` -- its runtime size isn't known until execution reaches the decl
        local_name[rid] = nm[rid] = _uniq(name)
    nm.update(lf.globals_used)  # file-scope globals (defined in the source)
    for rid, quals in lf.global_quals.items():  # a qualified global: an unqualified lvalue
        if (ct := lf.rid_types.get(rid)) is not None and quals:
            nm[rid] = _unqualified_global(nm[rid], ct)
    ref = _Names(nm, used, labels=dict(lf.globals_used))

    def _local_decl(rid, name, ct):
        # the zero baseline as the empty initializer: `= {0}` re-lowers as the baseline AND a store of 0 to the
        # first scalar, which the next emit spells as a store -- one more each round (CF-RTWIDE)
        zi = " = {}" if rid in lf.zero_init_locals else ""
        if _has_fp(
            ct
        ):  # `RET (*name)(PARAMS)` (no typedef alias), a table `RET (*name[N])(PARAMS)`
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
        if _has_fp(ct):  # `RET (*name)(PARAMS)`, as a local's -- a table's `RET (*name[N])(PARAMS)`
            zero = "{0}" if ct.kind == "array" else "0u"
            return f"    {sc} {_funcptr_decl(ct, name)} = {init or zero};"
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
        _extern_decl(callee, rct, pcts, quals, va)
        for callee, (rct, pcts, quals, va) in sorted(lf.tu_protos.items())
    ]
    # ... and every function of the unit it calls, in the order of the first call (not itself)
    callees = dict.fromkeys(c for c, _a in lf.calls if c != lf.name and c in (unit or {}))
    fwd = [_signature(unit[c]) + ";" for c in callees]
    head = "\n".join(tu_decls + fwd) + "\n" if tu_decls or fwd else ""
    return head + _signature(lf, params) + "\n{\n" + "\n".join(decls + body) + "\n}"


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
                if isinstance(item, CaseLabel):  # exact: `Nu` past LLONG_MAX (CF-ENUMFOLD)
                    out.append(f"{ind}case {_const_spelling(item.value)}:")
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
        if _has_fp(lf.rid_types.get(rid)):
            # a function pointer, a pointer to one (`&fp`, CF-FPTAB) or a null one (CF-NULLPTR) is declared by its
            # declarator, whatever spelling the caller made of its type: `RET (*t)(PARAMS) = 0u;`
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
    if c.op == "c.const":  # a negative one signed, not `-Nu` (CF-ENUMFOLD)
        return deftmp(c.wr[0], _kspell(c.imm[0]))
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
        rt = lf.rid_types.get(c.wr[0])
        # a function pointer, a pointer to one: spelled from the temp's type, which carries it whole (CF-RTFP)
        to = _qual_type(rt, ()) if _has_fp(rt) else c.op.split(":", 1)[1]
        return deftmp(c.wr[0], f"({to}){ref(c.rd[0])}")
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
        # the temp a load declares: a function pointer read from a table or a member is declared as one
        # (CF-FPTAB) -- a `uint32_t` of it did not compile
        decl = (
            _funcptr_decl(lf.rid_types[c.wr[0]], ref(c.wr[0]))
            if _has_fp(lf.rid_types.get(c.wr[0]))
            else f"{et} {ref(c.wr[0])}"
        )
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
                    f"{decl}; memcpy(&{t}, (const char *){bp} + {off} + "
                    f"(size_t){ref(c.rd[1])} * {stride}, {es});"
                )
            return deftmp(
                c.wr[0],
                f"{_elem_base(lf, c.rd[0], ref)}[{_idx(lf, c, ref)}]",
                None if _has_fp(lf.rid_types.get(c.wr[0])) else et,
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
        return f"{decl}; memcpy(&{t}, (const char *){ptr} + {off}, sizeof {t});"
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
                vt = lf.rid_types.get(c.rd[2])
                if (
                    vt is not None and vt.kind == "funcptr"
                ):  # into a member table (CF-FPTAB): through a pointer to
                    # its own type, as a member's store below (a memcpy of a designator copies code)
                    return _fp_store(lf, dst, c.rd[2], ref)
                conv = (
                    "_Bool"
                    if (len(c.imm) > 2 and c.imm[2])
                    else _store_conv(lf.rid_types.get(c.rd[2]), es)
                )
                if conv:  # convert the source to the element type first (a
                    return f"{{ {conv} _sv = {ref(c.rd[2])}; memcpy({dst}, &_sv, {es}); }}"  # _Bool normalizes), else
                return f"memcpy({dst}, &{ref(c.rd[2])}, {es});"  # `es` bytes of a narrower/float source corrupts it
            return f"{_elem_base(lf, c.rd[0], ref)}[{_idx(lf, c, ref, write=True)}] = {ref(c.rd[2])};"  # typed array (masked -> WRITE-guarded)
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
            # function NAME: store through a pointer to its own type so the name decays to its address (a plain
            # `memcpy(&g_func,8)` copies the function's CODE; `void *` can't hold a funcptr), and the call site
            # reads the member as the type it was stored as (`_fp_store`)
            return _fp_store(lf, f"(char *){ptr} + {off}", c.rd[1], ref)
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
        # a `_BitInt` value converts to the unit's type first: `v & 31u` of an `unsigned _BitInt(12)` is an
        # `unsigned int` in C23, `_BitInt` arithmetic whose result is a standard type, which neither rail lowers --
        # the emit re-parsed was refused (CF-RTFP; C converts it so anyway, so the twin's emit, never re-parsed, does
        # without)
        vt = lf.rid_types.get(c.rd[1])
        val = (
            f"({'uint64_t' if wide else 'uint32_t'}){ref(c.rd[1])}"
            if vt is not None and vt.is_bitint
            else ref(c.rd[1])
        )
        return deftmp(
            c.wr[0],
            f"({ref(c.rd[0])} & {clear}{sfx}) | (({val} & {mask}{sfx}) << {bit_off})",
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
        # declares it; the host LINKER resolves it. Its qualified parameters and return meet the emit's
        # unqualified objects through casts (CF-QUALS)
        pret, pcts, quals, _va = lf.tu_protos.get(callee, (None, (), (), False))
        call = f"{callee}({_qual_args(lf, ref, c.rd, pcts, quals)})"
        if not c.wr:
            return f"{call};"
        rt = lf.rid_types.get(c.wr[0])
        return deftmp(
            c.wr[0],
            _qual_result(call, pret, quals),
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
        # its qualified parameters and return meet the emit's unqualified objects through casts (CF-QUALS)
        fct = lf.rid_types.get(c.rd[0])
        if fct is None or fct.kind != "funcptr":
            fct = None
        params, quals = (fct.params, fct.fquals) if fct is not None else ((), ())
        call = f"{ref(c.rd[0])}({_qual_args(lf, ref, c.rd[1:], params, quals)})"
        if not c.wr:  # a void function: a bare call
            return f"{call};"
        return deftmp(c.wr[0], _qual_result(call, fct.of if fct is not None else None, quals))
    if c.op.startswith("c.call.imember:"):  # o->fn(args): funcptr struct member
        field = c.op.split(":", 1)[1]
        sep = "->" if c.imm and c.imm[0] else "."
        # its qualified parameters and return meet the emit's unqualified objects through casts (CF-QUALS)
        fct = lf.member_calls.get(c.id)
        params, quals = (fct.params, fct.fquals) if fct is not None else ((), ())
        call = f"{ref(c.rd[0])}{sep}{field}({_qual_args(lf, ref, c.rd[1:], params, quals)})"
        if not c.wr:
            return f"{call};"
        return deftmp(c.wr[0], _qual_result(call, fct.of if fct is not None else None, quals))
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
    return ", ".join(_arg_list(lf, ref, rids))


def _arg_list(lf: LoweredFunc, ref, rids) -> list:
    """`_args`, one argument per entry."""
    out = []
    for r in rids:
        ct = lf.rid_types.get(r)
        dims = 0
        while ct is not None and ct.kind == "array":
            dims, ct = dims + 1, ct.of
        out.append(f"&{ref(r)}" + "[0]" * dims if r in lf.globals_used and dims > 1 else ref(r))
    return out


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


def _array_dims(ct) -> list:
    """The dimensions of a nested array type, outermost first (`T g[A][B]` -> [A, B]); [] for any other."""
    dims = []
    while ct is not None and ct.kind == "array" and not ct.shape:
        dims.append(ct.count)
        ct = ct.of
    return dims


def _elem_base(lf: LoweredFunc, rid: int, ref) -> str:
    """The base an element access `B[i]` indexes. A file-scope multi-dimensional array is declared nested, as its
    source declares it, so `g[i]` would be a row; its flat index (Horner-flattened, bounded by the whole array)
    indexes its first element, `(&g[0][0])[i]`, as a local array declared flat is indexed (CF-AOS2D). The twin's
    `elem_base`."""
    dims = _array_dims(lf.rid_types.get(rid))
    if rid in lf.globals_used and len(dims) > 1:
        return f"(&{ref(rid)}" + "[0]" * len(dims) + ")"
    return ref(rid)


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
            return f'{chk}({c.rd[0]}, {idx}, {ref(ext)}, "{lf.name}:{ref.label(c.rd[0])}")'
        rt = lf.rid_types.get(c.rd[0])
        n = getattr(rt, "count", 0) if rt is not None else 0
        dims = _array_dims(rt)
        if len(dims) > 1:  # a nested (file-scope) array: the whole array bounds its flat index
            n = 1
            for d in dims:
                n *= d
        if n:  # a known-extent local/static array (constant N)
            return f'{chk}({c.rd[0]}, {idx}, {n}u, "{lf.name}:{ref.label(c.rd[0])}")'
    return idx
