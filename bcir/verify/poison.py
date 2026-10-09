"""R12's reading of the poison facts an emitted elementwise kernel carries (G29).

The proved side is `lower.poison.prove_no_wrap` over the resources' declared value ranges. The
emitted side is read here from the kernel's text and nothing else -- the flags on every i32
compute instruction (vector or scalar) and the `!range` on every load -- with the same kernel
reader the alias facts use (`verify.alias._Kernel`). Three kinds of finding, named apart:

  * a FORGED fact -- `nsw` / `nuw` the declared ranges do not prove, a `!range` no declaration
    backs or that differs from it: LLVM folds on the strength of these, so each turns a value
    the contract allows into poison;
  * a DROPPED fact -- a proved flag or a declared range the kernel does not carry;
  * an UNREADABLE kernel -- no compute instruction, a `!range` that is not two i32 bounds, a
    range on a store: a finding, never a traceback (L1).

A float kernel carries no poison facts at all: any flag or range there is forged.
"""

from __future__ import annotations

import re

_COMPUTE = re.compile(r"=\s*(add|sub|mul)((?:\s+(?:nuw|nsw))*)\s+(<\d+\s+x\s+i32>|i32)\s")
_ACCESS_KIND = re.compile(r"^(?:%[\w.]+\s*=\s*)?(load|store)\b")


def ll_poison_diagnostics(module, claim, text: str, elem: str) -> list[str]:
    """Every way the kernel's no-wrap flags and load ranges differ from what the declared
    ranges prove for `claim`."""
    from ..lower.poison import FLAGS, PoisonError, prove_no_wrap, range_metadata
    from .alias import _Kernel

    k = _Kernel(text)
    out: list[str] = []
    try:
        proof = prove_no_wrap(module, claim) if elem == "i32" else None
    except PoisonError as exc:
        return [f"poison facts unprovable: {exc}"]
    want = proof.flags if proof is not None else frozenset()
    computes = [m for ins in k.instrs if (m := _COMPUTE.search(ins))]
    if elem == "i32" and not computes:
        out.append("poison facts unreadable: no i32 compute instruction in the kernel")
    for m in computes:
        got = frozenset(m.group(2).split())
        where = f"`{m.group(1)}` on {m.group(3)}"
        for flag in FLAGS:
            if flag in got and flag not in want:
                out.append(
                    f"forged no-wrap fact: {where} carries `{flag}`, which the declared value "
                    f"ranges do not prove (proved: {proof if proof is not None else 'none'})"
                )
            elif flag in want and flag not in got:
                out.append(f"dropped no-wrap fact: {where} lacks the proved `{flag}`")
    kinds = []
    for ins in k.instrs:
        a = _ACCESS_KIND.match(ins)
        if a:
            kinds.append(a.group(1))
    for (position, _ptr, _volatile, attach), kind in zip(k.accesses, kinds):
        node = attach.get("range")
        if kind == "store":
            if node is not None:
                out.append("poison facts unreadable: a store carries !range")
            continue
        declared = None
        if proof is not None and position in (0, 1) and proof.reads[position] is not None:
            declared = range_metadata(proof.reads[position])
        got = None
        if node is not None:
            entry = k.metadata.get(node)
            if (
                entry is None
                or entry[0]
                or len(entry[1]) != 2
                or not all(re.fullmatch(r"i32 -?\d+", tok) for tok in entry[1])
            ):
                out.append(f"poison facts unreadable: !range !{node} is not two i32 bounds")
                continue
            got = ", ".join(entry[1])
        if got is not None and got != declared:
            out.append(
                f"forged range fact: a load of operand {position} carries !range [{got}], "
                f"which no declared value range backs (declared: {declared or 'none'})"
            )
        elif got is None and declared is not None:
            out.append(
                f"dropped range fact: a load of operand {position} lacks the declared "
                f"!range [{declared}]"
            )
    return out


__all__ = ["ll_poison_diagnostics"]
