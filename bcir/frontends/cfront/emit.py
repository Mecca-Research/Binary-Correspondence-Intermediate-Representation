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
from .lower import IfNode, LoweredFunc, ReturnNode, WhileNode

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
    return ct.name


def emit_function(lf: LoweredFunc) -> str:
    """The lowered function as standalone C, named `bcir_<name>` (so it can sit beside the original).
    Walks the structured body tree, so `if`/`while`/`return` emit real C control flow; mutable named
    locals are declared up front and assigned (so branch merges + loop accumulators reproduce the
    source); intermediate expression results stay single-assignment temporaries."""
    nm: dict[int, str] = {rid: pname for pname, rid, _ct in lf.params}
    for rid, name, _ct in lf.locals:
        nm[rid] = name
    nm.update(lf.globals_used)                            # file-scope globals (defined in the source)

    def ref(rid: int) -> str:
        return nm.get(rid, f"t{rid}")

    decls = [f"    {_cname(ct)} {name};" for _rid, name, ct in lf.locals]
    body = _walk(lf, lf.body, ref, 1)
    sig_params = ", ".join(f"{_cname(ct)} {pname}" for pname, _rid, ct in lf.params) or "void"
    ret = _cname(lf.ret_type)
    return (f"static {ret} bcir_{lf.name}({sig_params})\n{{\n"
            + "\n".join(decls + body) + "\n}")


def _walk(lf: LoweredFunc, block: list, ref, depth: int) -> list:
    ind = "    " * depth
    out: list = []
    for node in block:
        if isinstance(node, IfNode):
            out.append(f"{ind}if ({ref(node.cond)}) {{")
            out += _walk(lf, node.then, ref, depth + 1)
            if node.els:
                out.append(f"{ind}}} else {{")
                out += _walk(lf, node.els, ref, depth + 1)
            out.append(f"{ind}}}")
        elif isinstance(node, WhileNode):
            out.append(f"{ind}while (1) {{")
            out += _walk(lf, node.cond_block, ref, depth + 1)
            out.append(f"{ind}    if (!{ref(node.cond)}) break;")
            out += _walk(lf, node.body, ref, depth + 1)
            out.append(f"{ind}}}")
        elif isinstance(node, ReturnNode):
            out.append(f"{ind}return {ref(node.rid)};" if node.rid is not None else f"{ind}return;")
        elif isinstance(node, Claim):
            out.append(ind + _claim_stmt(lf, node, ref))
    return out


def _claim_stmt(lf: LoweredFunc, c: Claim, ref) -> str:
    suf = c.op.split(".", 2)[-1] if "." in c.op else c.op

    def deftmp(rid: int, expr: str, ty: str = "uint32_t") -> str:
        return f"{ty} {ref(rid)} = {expr};"

    if c.op == "c.copy":                                     # write a mutable local (no new decl)
        return f"{ref(c.wr[0])} = {ref(c.rd[0])};"
    if c.op == "c.const":
        return deftmp(c.wr[0], f"{c.imm[0]}u")
    if c.op.startswith("c.bin."):
        return deftmp(c.wr[0], f"{ref(c.rd[0])} {_BINOP[suf]} {ref(c.rd[1])}")
    if c.op.startswith("c.un."):
        return deftmp(c.wr[0], f"({_UNOP[suf]}{ref(c.rd[0])})")
    if c.op == "c.load":
        et = _load_ctype(lf, c.wr[0])
        off = c.imm[0] if c.imm else 0
        t = ref(c.wr[0])
        if len(c.rd) == 2:                                   # base[index] (typed array — aligned)
            return deftmp(c.wr[0], f"{ref(c.rd[0])}[{ref(c.rd[1])}]", et)
        ptr = _base_ptr(lf, c.rd[0], ref)
        if c.domain.name == "MMIO":                          # device register: ordered volatile load
            return deftmp(c.wr[0], f"*(volatile {et} *)((const volatile char *){ptr} + {off})", et)
        # plain RAM member/deref: memcpy is alignment-safe (handles packed) — Clang folds it to a load.
        return f"{et} {t}; memcpy(&{t}, (const char *){ptr} + {off}, sizeof {t});"
    if c.op == "c.store":
        off = c.imm[0] if c.imm else 0
        if len(c.rd) == 3:                                   # base[index] = value (typed array)
            return f"{ref(c.rd[0])}[{ref(c.rd[1])}] = {ref(c.rd[2])};"
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
    if c.op.startswith("c.call:"):
        callee = c.op.split(":", 1)[1]
        return deftmp(c.wr[0], f"bcir_{callee}({', '.join(ref(r) for r in c.rd)})")
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
