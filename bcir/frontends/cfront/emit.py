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
        if ct.kind == "array":                               # `T name[N]` (the dims follow the name)
            return f"    {_cname(ct.of)} {name}[{ct.count}]{zi};"
        return f"    {_cname(ct)} {name}{zi};"
    decls = [_local_decl(rid, local_name[rid], ct) for rid, _name, ct in lf.locals]
    decls += [f"    static {_cname(ct)} {name} = {init}u;"      # static storage: once-only const init
              for _rid, name, ct, init in lf.statics]
    body = _walk(lf, lf.body, ref, 1)
    sig_params = ", ".join(f"{_cname(ct)} {pname}" for pname, _rid, ct in lf.params) or "void"
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
            ct = lf.rid_types.get(rid)                        # for a float, the (width, signedness) integer
            ty = _cname(ct) if (ct is not None and (ct.is_float or ct.is_integer)) else "uint32_t"
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
    if c.op.startswith("c.bin."):
        return deftmp(c.wr[0], f"{ref(c.rd[0])} {_BINOP[suf]} {ref(c.rd[1])}")
    if c.op.startswith("c.un."):
        return deftmp(c.wr[0], f"({_UNOP[suf]}{ref(c.rd[0])})")
    if c.op.startswith("c.cast:"):                           # (type)operand — width cast / reinterpret
        return deftmp(c.wr[0], f"({c.op.split(':', 1)[1]}){ref(c.rd[0])}")
    if c.op == "c.addrof":                                   # &lvalue -> a pointer value (T *t = &x;)
        rt = lf.rid_types.get(c.wr[0])
        return deftmp(c.wr[0], f"&{ref(c.rd[0])}", _cname(rt) if rt is not None else None)
    if c.op == "c.select":                                   # ternary: cond ? then : els
        return deftmp(c.wr[0], f"({ref(c.rd[0])} ? {ref(c.rd[1])} : {ref(c.rd[2])})")
    if c.op == "c.load":
        et = _load_ctype(lf, c.wr[0])
        off = c.imm[0] if c.imm else 0
        t = ref(c.wr[0])
        if len(c.rd) == 2:                                   # base[index]
            if c.imm:                                        # s.arr[i]: member offset + element-scaled
                es = c.imm[1] if len(c.imm) > 1 else 4        # index -> &base + off + i*elem_size
                bp = _base_ptr(lf, c.rd[0], ref)
                return (f"{et} {t}; memcpy(&{t}, (const char *){bp} + {off} + "
                        f"(size_t){ref(c.rd[1])} * {es}, {es});")
            return deftmp(c.wr[0], f"{ref(c.rd[0])}[{ref(c.rd[1])}]", et)   # typed array — aligned
        ptr = _base_ptr(lf, c.rd[0], ref)
        if c.domain.name == "MMIO":                          # device register: ordered volatile load
            return deftmp(c.wr[0], f"*(volatile {et} *)((const volatile char *){ptr} + {off})", et)
        # plain RAM member/deref: memcpy is alignment-safe (handles packed) — Clang folds it to a load.
        return f"{et} {t}; memcpy(&{t}, (const char *){ptr} + {off}, sizeof {t});"
    if c.op == "c.store":
        off = c.imm[0] if c.imm else 0
        if len(c.rd) == 3:                                   # base[index] = value
            if c.imm:                                        # s.arr[i] = v: &base + off + i*elem_size
                es = c.imm[1] if len(c.imm) > 1 else 4
                bp = _base_ptr(lf, c.rd[0], ref)
                return (f"memcpy((char *){bp} + {off} + (size_t){ref(c.rd[1])} * {es}, "
                        f"&{ref(c.rd[2])}, {es});")
            return f"{ref(c.rd[0])}[{ref(c.rd[1])}] = {ref(c.rd[2])};"   # typed array
        ptr = _base_ptr(lf, c.rd[0], ref)
        size = c.imm[1] if len(c.imm) > 1 else 4
        if c.domain.name == "MMIO":                          # device register: ordered volatile store
            return f"*(volatile uint32_t *)((volatile char *){ptr} + {off}) = {ref(c.rd[1])};"
        # plain RAM member/deref: memcpy `size` bytes (correct truncation on little-endian, packed-safe).
        return f"memcpy((char *){ptr} + {off}, &{ref(c.rd[1])}, {size});"
    if c.op == "c.bf.get":                                   # (unit >> bit_off) & mask
        bit_off, width = c.imm
        return deftmp(c.wr[0], f"({ref(c.rd[0])} >> {bit_off}) & {(1 << width) - 1}u")
    if c.op == "c.bf.set":                                   # (old & ~(mask<<off)) | ((v&mask)<<off)
        bit_off, width = c.imm
        mask = (1 << width) - 1
        clear = ~(mask << bit_off) & 0xFFFFFFFF
        return deftmp(c.wr[0], f"({ref(c.rd[0])} & {clear}u) | "
                               f"(({ref(c.rd[1])} & {mask}u) << {bit_off})")
    if c.op.startswith("c.call.libm:"):                      # a <math.h> call -> the real libm function
        callee = c.op.split(":", 1)[1]                       # (no bcir_ twin; the harness links -lm)
        rt = lf.rid_types.get(c.wr[0])                       # declare at the true result width: a long
        ty = _cname(rt) if rt is not None else None          # return (lround) is not narrowed to uint32
        return deftmp(c.wr[0], f"{callee}({', '.join(ref(r) for r in c.rd)})", ty)
    if c.op.startswith("c.call.void:"):                      # a void callee -> a bare call statement
        callee = c.op.split(":", 1)[1]
        return f"bcir_{callee}({', '.join(ref(r) for r in c.rd)});"
    if c.op.startswith("c.call:"):
        callee = c.op.split(":", 1)[1]
        rt = lf.rid_types.get(c.wr[0])                       # a wide (8-byte) return declares at its true
        ty = _cname(rt) if (rt is not None and rt.is_integer and rt.size > 4) else None   # width, not uint32
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
        return deftmp(c.wr[0], f"atomic_{fn}({ref(c.rd[0])}, {ref(c.rd[1])})")
    raise ValueError(f"emit: unhandled claim op {c.op!r}")


def _load_ctype(lf: LoweredFunc, rid: int) -> str:
    ct = lf.rid_types.get(rid)
    return _cname(ct) if ct and ct.is_integer else "uint32_t"


def _base_ptr(lf: LoweredFunc, rid: int, ref) -> str:
    """A pointer to the base resource: the name decays if it's a pointer/array, else address-of."""
    ct = lf.rid_types.get(rid)
    name = ref(rid)
    if ct and ct.kind in ("pointer", "array"):
        return name
    return f"&{name}"
