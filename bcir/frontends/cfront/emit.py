"""Emit C back out of a lowered function's claim graph — the *verified C output* seam. Walking the
claims in order reproduces the source's integer semantics exactly (the emitter maps each claim's `op`
back to its C operator / memory access), so compiling this alongside the original fixture and
diffing the results on the same inputs is the behaviour-equivalence check the ladder requires.

This is the per-pattern-free, claim-graph-driven path the roadmap's C.2 generalizes toward: it emits
an *arbitrary* straight-line scalar claim graph, not a fixed kernel template.
"""
from __future__ import annotations

from .ctype_model import CType
from .lower import LoweredFunc

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
    """The lowered function as standalone C, named `bcir_<name>` (so it can sit beside the original)."""
    nm: dict[int, str] = {}                                  # rid -> C expression
    for pname, rid, _ct in lf.params:
        nm[rid] = pname

    def ref(rid: int) -> str:
        return nm.get(rid, f"t{rid}")

    lines: list[str] = []
    for c in lf.claims:
        suf = c.op.split(".", 2)[-1] if "." in c.op else c.op
        if c.op == "c.const":
            t = c.wr[0]
            lines.append(f"    uint32_t {ref(t)} = {c.imm[0]}u;")
        elif c.op.startswith("c.bin."):
            t = c.wr[0]
            lines.append(f"    uint32_t {ref(t)} = {ref(c.rd[0])} {_BINOP[suf]} {ref(c.rd[1])};")
        elif c.op.startswith("c.un."):
            t = c.wr[0]
            lines.append(f"    uint32_t {ref(t)} = ({_UNOP[suf]}{ref(c.rd[0])});")
        elif c.op == "c.load":
            t = c.wr[0]
            base = c.rd[0]
            et = _load_ctype(lf, t)
            off = c.imm[0] if c.imm else 0
            if len(c.rd) == 2:                               # base[index]
                lines.append(f"    {et} {ref(t)} = {ref(base)}[{ref(c.rd[1])}];")
            else:                                            # member / deref at byte offset
                ptr = _base_ptr(lf, base)
                lines.append(f"    {et} {ref(t)} = *({et} *)((const char *){ptr} + {off});")
        elif c.op == "c.store":
            base = c.rd[0]
            off = c.imm[0] if c.imm else 0
            if len(c.rd) == 3:                               # base[index] = value
                lines.append(f"    {ref(base)}[{ref(c.rd[1])}] = {ref(c.rd[2])};")
            else:                                            # member / deref = value
                ptr = _base_ptr(lf, base)
                val = c.rd[1]
                lines.append(f"    *(uint32_t *)((char *){ptr} + {off}) = {ref(val)};")
        elif c.op.startswith("c.call:"):
            t = c.wr[0]
            callee = c.op.split(":", 1)[1]
            args = ", ".join(ref(r) for r in c.rd)
            lines.append(f"    uint32_t {ref(t)} = bcir_{callee}({args});")
        else:
            raise ValueError(f"emit: unhandled claim op {c.op!r}")

    sig_params = ", ".join(f"{_cname(ct)} {pname}" for pname, _rid, ct in lf.params) or "void"
    ret = _cname(lf.ret_type)
    body = "\n".join(lines)
    retv = f"    return {ref(lf.return_rid)};" if lf.return_rid is not None else ""
    return f"static {ret} bcir_{lf.name}({sig_params})\n{{\n{body}\n{retv}\n}}"


def _load_ctype(lf: LoweredFunc, rid: int) -> str:
    ct = lf.rid_types.get(rid)
    return _cname(ct) if ct and ct.is_integer else "uint32_t"


def _base_ptr(lf: LoweredFunc, rid: int) -> str:
    """A pointer to the base resource: the name decays if it's a pointer/array, else address-of."""
    ct = lf.rid_types.get(rid)
    name = next((p for p, r, _ in lf.params if r == rid), f"t{rid}")
    if ct and ct.kind in ("pointer", "array"):
        return name
    return f"&{name}"
