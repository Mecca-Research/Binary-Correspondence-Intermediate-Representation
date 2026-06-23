"""Emit C back out of a lowered function's claim graph — the *verified C output* seam. Walking the
claims in order reproduces the source's integer semantics exactly (the emitter maps each claim's `op`
back to its C operator / memory access), so compiling this alongside the original fixture and
diffing the results on the same inputs is the behaviour-equivalence check the ladder requires.

This is the per-pattern-free, claim-graph-driven path the roadmap's C.2 generalizes toward: it emits
an *arbitrary* straight-line scalar claim graph, not a fixed kernel template.
"""
from __future__ import annotations

from ...model import Claim
from .ctype_model import CType
from .lower import (
    BreakNode,
    CaseLabel,
    ContinueNode,
    DefaultLabel,
    GotoNode,
    IfNode,
    LabelNode,
    LoweredFunc,
    ReturnNode,
    SwitchNode,
    WhileNode,
)

# op-suffix -> C operator.
_BINOP = {"add": "+", "sub": "-", "mul": "*", "div": "/", "mod": "%", "and": "&", "or": "|",
          "xor": "^", "shl": "<<", "shr": ">>", "eq": "==", "ne": "!=", "lt": "<", "gt": ">",
          "le": "<=", "ge": ">=", "land": "&&", "lor": "||"}
_UNOP = {"neg": "-", "bnot": "~", "lnot": "!"}


def _cname(ct: CType) -> str:
    if ct.kind == "pointer":
        return _cname(ct.of) + " *"
    if ct.kind == "array":
        return _cname(ct.of)            # decays in a parameter position
    if ct.is_aggregate:
        return f"{ct.kind} {ct.name}"
    return ("_Atomic " if ct.atomic else "") + ct.name


def _funcptr_decl(ct: CType, name: str) -> str:
    """Render an inline function-pointer declarator `RET (*name)(PARAMS)`. Used in param + local
    position, where (unlike a typedef alias) there is no spelling to print and the full signature
    must be reconstructed from the carried return + parameter types — `int (*g)(int)`."""
    ret = _cname(ct.of) if ct.of is not None else "void"
    plist = ", ".join(_cname(p) for p in ct.params) or "void"
    return f"{ret} (*{name})({plist})"


def emit_function(lf: LoweredFunc) -> str:
    """The lowered function as standalone C, named `bcir_<name>` (so it can sit beside the original).
    Walks the structured body tree, so `if`/`while`/`return` emit real C control flow; mutable named
    locals are declared up front and assigned (so branch merges + loop accumulators reproduce the
    source); intermediate expression results stay single-assignment temporaries."""
    nm: dict[int, str] = {rid: pname for pname, rid, _ct in lf.params}
    # Each local needs a *unique* C identifier: the lowering flattens scopes, so two source locals that
    # shared a name in disjoint scopes (e.g. `i` in two separate `for` loops, or a block local shadowing
    # a param) become distinct rids with the same name. Declaring both at function scope is a C
    # redefinition. Disambiguate the second-and-later occurrences (`i`, `i_2`, ...) -- a fresh variable
    # preserves the source's separate-scope semantics; naive name-sharing would corrupt a shadowed value.
    used: set[str] = set(nm.values())
    used.update(name for _rid, name, _ct, _init in lf.statics)
    used.update(lf.globals_used.values())
    local_name: dict[int, str] = {}
    for rid, name, _ct in lf.locals:
        uniq = name
        if uniq in used:
            k = 2
            while f"{name}_{k}" in used:
                k += 1
            uniq = f"{name}_{k}"
        used.add(uniq)
        local_name[rid] = uniq
        nm[rid] = uniq
    for rid, name, _ct, _init in lf.statics:
        nm[rid] = name
    nm.update(lf.globals_used)                            # file-scope globals (defined in the source)

    def ref(rid: int) -> str:
        return nm.get(rid, f"t{rid}")

    def _local_decl(rid, name, ct):
        zi = " = {0}" if rid in lf.zero_init_locals else ""
        if ct.kind == "funcptr":                             # `RET (*name)(PARAMS)` (no typedef alias)
            return f"    {_funcptr_decl(ct, name)}{zi};"
        if ct.kind == "array":                               # `T name[N]` (the dims follow the name)
            return f"    {_cname(ct.of)} {name}[{ct.count}]{zi};"
        return f"    {_cname(ct)} {name}{zi};"
    decls = [_local_decl(rid, local_name[rid], ct) for rid, _name, ct in lf.locals]
    decls += [f"    static {_cname(ct)} {name} = {init}u;"      # static storage: once-only const init
              for _rid, name, ct, init in lf.statics]
    body = _walk(lf, lf.body, ref, 1)
    parts = [_funcptr_decl(ct, pname) if ct.kind == "funcptr" else f"{_cname(ct)} {pname}"
             for pname, _rid, ct in lf.params]
    if lf.variadic:                                      # a trailing `...` after the named params
        parts.append("...")
    sig_params = ", ".join(parts) or "void"
    ret = _cname(lf.ret_type)
    return (f"static {ret} bcir_{lf.name}({sig_params})\n{{\n"
            + "\n".join(decls + body) + "\n}")


def _walk(lf: LoweredFunc, block: list, ref, depth: int, loops: list | None = None) -> list:
    ind = "    " * depth
    loops = loops if loops is not None else []
    out: list = []
    for node in block:
        if isinstance(node, IfNode):
            out.append(f"{ind}if ({ref(node.cond)}) {{")
            out += _walk(lf, node.then, ref, depth + 1, loops)
            if node.els:
                out.append(f"{ind}}} else {{")
                out += _walk(lf, node.els, ref, depth + 1, loops)
            out.append(f"{ind}}}")
        elif isinstance(node, WhileNode):
            loops.append(node.loop_id)
            out.append(f"{ind}while (1) {{")
            if node.test_at_end:                       # do/while: body, [continue:], recompute + test
                out += _walk(lf, node.body, ref, depth + 1, loops)
                out.append(f"{ind}    __cont_{node.loop_id}: ;")
                out += _walk(lf, node.cond_block, ref, depth + 1, loops)
                out.append(f"{ind}    if (!{ref(node.cond)}) break;")
            else:                                      # while/for: test, body, [continue:], step
                out += _walk(lf, node.cond_block, ref, depth + 1, loops)
                out.append(f"{ind}    if (!{ref(node.cond)}) break;")
                out += _walk(lf, node.body, ref, depth + 1, loops)
                out.append(f"{ind}    __cont_{node.loop_id}: ;")
                out += _walk(lf, node.step, ref, depth + 1, loops)
            out.append(f"{ind}}}")
            loops.pop()
        elif isinstance(node, SwitchNode):                 # a real C switch (fallthrough preserved)
            out.append(f"{ind}switch ({ref(node.disc)}) {{")
            for item in node.body:
                if isinstance(item, CaseLabel):
                    out.append(f"{ind}case {item.value}:")
                elif isinstance(item, DefaultLabel):
                    out.append(f"{ind}default:")
                else:
                    out += _walk(lf, [item], ref, depth + 1, loops)
            out.append(f"{ind}}}")
        elif isinstance(node, ReturnNode):
            out.append(f"{ind}return {ref(node.rid)};" if node.rid is not None else f"{ind}return;")
        elif isinstance(node, BreakNode):
            out.append(f"{ind}break;")
        elif isinstance(node, ContinueNode):
            out.append(f"{ind}goto __cont_{loops[-1]};")
        elif isinstance(node, GotoNode):
            out.append(f"{ind}goto {node.label};")
        elif isinstance(node, LabelNode):
            out.append(f"{node.name}:;")               # a jump target (function-body scope)
        elif isinstance(node, Claim):
            out.append(ind + _claim_stmt(lf, node, ref))
    return out


def _claim_stmt(lf: LoweredFunc, c: Claim, ref) -> str:
    suf = c.op.split(".", 2)[-1] if "." in c.op else c.op

    def deftmp(rid: int, expr: str, ty: str | None = None) -> str:
        if ty is None:                                       # a temp renders its true C type: float/double
            ct = lf.rid_types.get(rid)                        # for a float, the (width, signedness) integer;
            ty = _cname(ct) if (ct is not None and (ct.is_float or ct.is_integer                # a pointer
                                                    or ct.kind == "pointer")) else "uint32_t"   # value -> `T *`

        return f"{ty} {ref(rid)} = {expr};"

    if c.op == "c.copy":                                     # write a mutable local (no new decl)
        return f"{ref(c.wr[0])} = {ref(c.rd[0])};"
    if c.op == "c.ptradd":                                   # pointer p += n (C scales by element size)
        return f"{ref(c.wr[0])} += {ref(c.rd[1])};"
    if c.op == "c.ptrsub":                                   # pointer p -= n
        return f"{ref(c.wr[0])} -= {ref(c.rd[1])};"
    if c.op == "c.const":
        return deftmp(c.wr[0], f"{c.imm[0]}u")
    if c.op.startswith("c.fconst:"):                         # a floating constant -> its literal spelling
        return deftmp(c.wr[0], c.op.split(":", 1)[1])
    if c.op.startswith("c.cconst:"):                         # <complex.h> imaginary unit -> verbatim token
        return deftmp(c.wr[0], c.op.split(":", 1)[1])
    if c.op.startswith("c.bin."):
        return deftmp(c.wr[0], f"{ref(c.rd[0])} {_BINOP[suf]} {ref(c.rd[1])}")
    if c.op == "c.un.creal":                                 # GNU __real__ z (complex part extraction)
        return deftmp(c.wr[0], f"__real__ {ref(c.rd[0])}")
    if c.op == "c.un.cimag":
        return deftmp(c.wr[0], f"__imag__ {ref(c.rd[0])}")
    if c.op.startswith("c.un."):
        return deftmp(c.wr[0], f"({_UNOP[suf]}{ref(c.rd[0])})")
    if c.op.startswith("c.cast:"):                           # (type)operand — width cast / reinterpret
        return deftmp(c.wr[0], f"({c.op.split(':', 1)[1]}){ref(c.rd[0])}")
    if c.op == "c.addrof":                                   # &lvalue -> a pointer value (T *t = &x;)
        rt = lf.rid_types.get(c.wr[0])
        ty = _cname(rt) if rt is not None else None
        castp = f"({ty})" if ty else ""
        if len(c.rd) == 2:                                   # &base[idx] -> (T *)((char *)bp + off + idx*stride)
            bp = _base_ptr(lf, c.rd[0], ref)                 # bp decays a pointer/array base, addresses a value
            return deftmp(c.wr[0], f"{castp}((char *){bp} + {c.imm[0]} + "
                                   f"(size_t){ref(c.rd[1])} * {c.imm[1]})", ty)
        if c.imm:                                            # &base.member -> (T *)((char *)bp + off); bp is
            bp = _base_ptr(lf, c.rd[0], ref)                 # `s` for `s->m` (pointer), `&s` for `s.m` (value)
            return deftmp(c.wr[0], f"{castp}((char *){bp} + {c.imm[0]})", ty)
        return deftmp(c.wr[0], f"&{ref(c.rd[0])}", ty)       # &name / &(compound literal)
    if c.op == "c.select":                                   # ternary: cond ? then : els
        return deftmp(c.wr[0], f"({ref(c.rd[0])} ? {ref(c.rd[1])} : {ref(c.rd[2])})")
    if c.op == "c.load":
        et = _load_ctype(lf, c.wr[0])
        off = c.imm[0] if c.imm else 0
        t = ref(c.wr[0])
        if len(c.rd) == 2:                                   # base[index]
            if c.imm:                                        # s.arr[i]: member offset + element-scaled
                es = c.imm[1] if len(c.imm) > 1 else 4        # index -> &base + off + i*stride, copy `es` bytes
                stride = c.imm[2] if len(c.imm) > 2 else es   # array-of-structs `arr[i].field`: stride sizeof(elem)
                bp = _base_ptr(lf, c.rd[0], ref)              # != the field copy size `es`
                return (f"{et} {t}; memcpy(&{t}, (const char *){bp} + {off} + "
                        f"(size_t){ref(c.rd[1])} * {stride}, {es});")
            return deftmp(c.wr[0], f"{ref(c.rd[0])}[{_idx(lf, c, ref)}]", et)   # typed array (masked -> guarded)
        ptr = _base_ptr(lf, c.rd[0], ref)
        if c.domain.name == "MMIO":                          # device register: ordered volatile load (its
            return deftmp(c.wr[0], f"*(volatile {et} *)((const volatile char *){ptr} + {off})", et)  # natural width)
        if len(c.imm) > 1:                                    # a (non-MMIO) BITFIELD unit: read only `imm[1]`
            return f"{et} {t} = 0; memcpy(&{t}, (const char *){ptr} + {off}, {c.imm[1]});"   # spanned bytes (zeroed)
        # plain RAM member/deref: memcpy is alignment-safe (handles packed) — Clang folds it to a load.
        return f"{et} {t}; memcpy(&{t}, (const char *){ptr} + {off}, sizeof {t});"
    if c.op == "c.store":
        off = c.imm[0] if c.imm else 0
        if len(c.rd) == 3:                                   # base[index] = value
            if c.imm:                                        # s.arr[i] = v: &base + off + i*stride, copy `es` bytes
                es = c.imm[1] if len(c.imm) > 1 else 4
                stride = c.imm[3] if len(c.imm) > 3 else es   # array-of-structs `arr[i].field=v`: stride sizeof(elem)
                bp = _base_ptr(lf, c.rd[0], ref)
                dst = f"(char *){bp} + {off} + (size_t){ref(c.rd[1])} * {stride}"
                conv = "_Bool" if (len(c.imm) > 2 and c.imm[2]) else _store_conv(lf.rid_types.get(c.rd[2]), es)
                if conv:                                     # convert the source to the element type first (a
                    return f"{{ {conv} _sv = {ref(c.rd[2])}; memcpy({dst}, &_sv, {es}); }}"   # _Bool normalizes), else
                return f"memcpy({dst}, &{ref(c.rd[2])}, {es});"   # `es` bytes of a narrower/float source corrupts it
            return f"{ref(c.rd[0])}[{_idx(lf, c, ref)}] = {ref(c.rd[2])};"   # typed array (masked -> guarded)
        ptr = _base_ptr(lf, c.rd[0], ref)
        size = c.imm[1] if len(c.imm) > 1 else 4
        if c.domain.name == "MMIO":                          # device register: ordered volatile store
            return f"*(volatile uint32_t *)((volatile char *){ptr} + {off}) = {ref(c.rd[1])};"
        # plain RAM member/deref: memcpy `size` bytes (correct truncation on little-endian, packed-safe). A
        # source whose type does not match the slot is CONVERTED through a slot-typed temp first -- a narrower
        # int (`*p = (short)b`) must widen, and a `float` stored into a `double` member must convert (not
        # memcpy 4 bytes into 8, nor copy float bits into a double slot).
        vt = lf.rid_types.get(c.rd[1])
        if vt is not None and vt.kind == "funcptr":          # a funcptr member set from a funcptr value / a
            # function NAME: store through a GENERIC funcptr lvalue so the name decays to its address (a plain
            # `memcpy(&g_func,8)` copies the function's CODE; `void *` can't hold a funcptr). The call site reads
            # the member's real type, and function pointers round-trip through the cast.
            return f"*(void (**)(void))((char *){ptr} + {off}) = (void (*)(void)){ref(c.rd[1])};"
        conv = "_Bool" if (len(c.imm) > 2 and c.imm[2]) else _store_conv(lf.rid_types.get(c.rd[1]), size)
        if conv:                                             # a _Bool slot normalizes the stored value to 0/1
            return f"{{ {conv} _sv = {ref(c.rd[1])}; memcpy((char *){ptr} + {off}, &_sv, {size}); }}"
        return f"memcpy((char *){ptr} + {off}, &{ref(c.rd[1])}, {size});"
    if c.op == "c.bf.get":                                   # (unit >> bit_off) & mask (sign-extended if signed)
        bit_off, width = c.imm[0], c.imm[1]
        rt = lf.rid_types.get(c.wr[0])                       # a WIDE bitfield (> 32b) needs 64-bit literals/cast
        wide = rt is not None and rt.size > 4
        sfx = "ull" if wide else "u"
        mask = (1 << width) - 1
        if len(c.imm) > 2 and c.imm[2]:                      # a signed bitfield: sign-extend from bit width-1
            sbit, cast = 1 << (width - 1), ("int64_t" if wide else "int32_t")
            return deftmp(c.wr[0], f"({cast})(((({ref(c.rd[0])} >> {bit_off}) & {mask}{sfx}) ^ {sbit}{sfx}) - {sbit}{sfx})")
        return deftmp(c.wr[0], f"({ref(c.rd[0])} >> {bit_off}) & {mask}{sfx}")
    if c.op == "c.bf.set":                                   # (old & ~(mask<<off)) | ((v&mask)<<off)
        bit_off, width = c.imm
        rt = lf.rid_types.get(c.wr[0])
        wide = rt is not None and rt.size > 4                # 64-bit unit: a 64-bit clear mask, not a 32-bit one
        sfx = "ull" if wide else "u"
        mask = (1 << width) - 1
        clear = ~(mask << bit_off) & ((1 << 64) - 1 if wide else 0xFFFFFFFF)
        return deftmp(c.wr[0], f"({ref(c.rd[0])} & {clear}{sfx}) | "
                               f"(({ref(c.rd[1])} & {mask}{sfx}) << {bit_off})")
    if c.op.startswith("c.call.libm:"):                      # a <math.h> / <stdlib.h> call -> the real function
        callee = c.op.split(":", 1)[1]                       # (no bcir_ twin; the harness links -lm/libc)
        rt = lf.rid_types.get(c.wr[0])                       # declare at the true result width: a long
        ty = _cname(rt) if rt is not None else None          # return (lround) is not narrowed to uint32
        return deftmp(c.wr[0], f"{callee}({', '.join(ref(r) for r in c.rd)})", ty)
    if c.op.startswith("c.call.libm.void:"):                 # a void external (free) -- verbatim statement, opaque
        callee = c.op.split(":", 1)[1]
        return f"{callee}({', '.join(ref(r) for r in c.rd)});"
    if c.op.startswith("c.call.extern:"):                    # a printf/scanf-family external variadic call
        callee = c.op.split(":", 1)[1]                       # emitted verbatim against <stdio.h>, returns int
        return deftmp(c.wr[0], f"{callee}({', '.join(ref(r) for r in c.rd)})", "int")
    if c.op.startswith("c.call.builtin:"):                   # a GCC/Clang integer builtin -> verbatim
        callee = "__builtin_" + c.op.split(":", 1)[1]        # the op stores the suffix; re-add the prefix
        rt = lf.rid_types.get(c.wr[0])
        return deftmp(c.wr[0], f"{callee}({', '.join(ref(r) for r in c.rd)})",
                      _cname(rt) if rt is not None else None)
    if c.op == "c.call.vaarg":                               # va_arg(ap, T) -- T is the result temp's type
        rt = lf.rid_types.get(c.wr[0])
        ty = _cname(rt) if rt is not None else "int"
        return deftmp(c.wr[0], f"va_arg({ref(c.rd[0])}, {ty})", ty)
    if c.op.startswith("c.call.vabuiltin:"):                 # va_start / va_end / va_copy -- verbatim, void
        callee = c.op.split(":", 1)[1]
        return f"{callee}({', '.join(ref(r) for r in c.rd)});"
    if c.op.startswith("c.call.void:"):                      # a void callee -> a bare call statement
        callee = c.op.split(":", 1)[1]
        return f"bcir_{callee}({', '.join(ref(r) for r in c.rd)});"
    if c.op.startswith("c.call:"):
        callee = c.op.split(":", 1)[1]
        rt = lf.rid_types.get(c.wr[0])                       # a wide (8-byte) int OR an aggregate return declares
        ty = (_cname(rt) if (rt is not None and (rt.is_aggregate                # at its true type (`struct P t =
              or (rt.is_integer and rt.size > 4))) else None)                   # mk(x);`), not uint32
        return deftmp(c.wr[0], f"bcir_{callee}({', '.join(ref(r) for r in c.rd)})", ty)
    if c.op == "c.call.indirect":                            # rd[0] is the function pointer; rd[1:] args
        return deftmp(c.wr[0], f"{ref(c.rd[0])}({', '.join(ref(r) for r in c.rd[1:])})")
    if c.op.startswith("c.call.imember:"):                   # o->fn(args): funcptr struct member
        field = c.op.split(":", 1)[1]
        sep = "->" if c.imm and c.imm[0] else "."
        return deftmp(c.wr[0], f"{ref(c.rd[0])}{sep}{field}({', '.join(ref(r) for r in c.rd[1:])})")
    if c.op.startswith("c.atomic."):              # atomic RMW -> the matching builtin (§5.8)
        return deftmp(c.wr[0], f"__atomic_fetch_{c.op.split('.')[-1]}("
                               f"{ref(c.rd[0])}, {ref(c.rd[1])}, __ATOMIC_SEQ_CST)")
    if c.op.startswith("c.cmpxchg."):             # compare-and-swap -> the __sync CAS builtin
        return deftmp(c.wr[0], f"__sync_{c.op.split('.')[-1]}_compare_and_swap("
                               f"{ref(c.rd[0])}, {ref(c.rd[1])}, {ref(c.rd[2])})")
    if c.op == "c.fence":
        return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"
    if c.op.startswith("c.c11atom."):             # C11 <stdatomic.h> generics on _Atomic objects
        fn = c.op.split(".")[-1]                   # fetch_add / fetch_sub / fetch_xor / load / store
        if fn == "load":
            return deftmp(c.wr[0], f"atomic_load({ref(c.rd[0])})")
        if fn == "store":
            return f"atomic_store({ref(c.rd[0])}, {ref(c.rd[1])});"
        if fn.startswith("cas_"):                 # cas_strong/weak -> bool compare_exchange (obj, &exp, des)
            return deftmp(c.wr[0], f"atomic_compare_exchange_{fn[4:]}("
                                   f"{ref(c.rd[0])}, {ref(c.rd[1])}, {ref(c.rd[2])})")
        return deftmp(c.wr[0], f"atomic_{fn}({ref(c.rd[0])}, {ref(c.rd[1])})")
    raise ValueError(f"emit: unhandled claim op {c.op!r}")


def _store_conv(vt, size: int):
    """The C type to convert a store source through (a slot-typed temp) before memcpy'ing `size` bytes into
    the slot -- or None when the source already matches the slot width/kind (a direct memcpy is correct). A
    `float`/`double` source of a different width must convert (float<->double, not a byte copy); a narrower
    integer source must widen/sign-extend to the slot width."""
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
    return _cname(ct) if ct and (ct.is_integer or ct.is_float or ct.kind == "pointer") else "uint32_t"


def _base_ptr(lf: LoweredFunc, rid: int, ref) -> str:
    """A pointer to the base resource: the name decays if it's a pointer/array, else address-of."""
    ct = lf.rid_types.get(rid)
    name = ref(rid)
    if ct and ct.kind in ("pointer", "array"):
        return name
    return f"&{name}"


def _idx(lf: LoweredFunc, c, ref) -> str:
    """The index expression for a `base[idx]` access. A `masked` access (§5.12 bounds-promotion) into a
    known-extent local/static array is wrapped in `BCIR_CHK(rid, idx, N)`: in-bounds returns idx
    (transparent -> behaviour-identical to the raw `a[i]`), out-of-bounds calls the bounds-quarantine
    handler. The numeric `rid` is the access provenance for the debugger. Any other access -> the bare index."""
    idx = ref(c.rd[1])
    if c.bounds == "masked":
        rt = lf.rid_types.get(c.rd[0])
        n = getattr(rt, "count", 0) if rt is not None else 0
        if n:
            return f"BCIR_CHK({c.rd[0]}, {idx}, {n}u)"
    return idx
