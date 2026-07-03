"""The cross-platform target-ABI matrix for the C frontend (Phase 4).

`ctype_model` computed `sizeof`/`alignof`/struct layout against a single hard-coded host data model
(LP64: `long` and pointers 8 bytes, `long double` 16). A *replacement* compiler must lay types out
for the *selected* target, so this module factors the size-varying part of the C data model into a
`TargetABI` and a table of the targets the roadmap names (x86-64 / AArch64 / RISC-V) plus the two that
actually diverge in layout (Windows LLP64, 32-bit ILP32).

What varies across these targets is the data model -- `sizeof(long)`, `sizeof(void *)` (and the
pointer-tracking types `size_t`/`intptr_t`/`ptrdiff_t`), and `long double`'s size+alignment. The
fixed-width (`<stdint.h>`) types, `int`, `short`, `char`, and `long long` are invariant. Float/double
IEEE math and the calling convention stay delegated to the resident backend (the two-truth line);
this models only the layout the frontend itself decides.
"""
from __future__ import annotations

from dataclasses import dataclass

# scalar names whose size tracks the pointer width (the data model's `intptr`-class types).
_PTR_TRACKING = frozenset({"size_t", "intptr_t", "uintptr_t", "ptrdiff_t"})


@dataclass(frozen=True)
class TargetABI:
    """The size-varying part of a target's C data model. `long_size`/`pointer_size`/`long_double_*`
    are the axes that move across the matrix; everything else (`int`, fixed-width, `long long`) is
    fixed by C / the common ABIs and lives in `ctype_model._SCALAR`."""
    name: str                       # short id, e.g. "x86_64-linux"
    triple: str                     # the Clang target triple (for `-target` / provenance)
    data_model: str                 # "LP64" | "LLP64" | "ILP32"
    long_size: int                  # sizeof(long) == sizeof(unsigned long)
    pointer_size: int               # sizeof(void *); also size_t / intptr_t / ptrdiff_t
    long_double_size: int
    long_double_align: int
    endian: str = "little"

    def scalar_size(self, name: str, default: int) -> int:
        """The size of an integer scalar under this ABI: `long` and the pointer-tracking types follow
        the data model; every other name keeps its fixed `default`."""
        if name in ("long", "unsigned long"):
            return self.long_size
        if name in _PTR_TRACKING:
            return self.pointer_size
        return default


_LP64 = dict(data_model="LP64", long_size=8, pointer_size=8,
             long_double_size=16, long_double_align=16)

# The named matrix. x86-64 / AArch64 / RISC-V are all LP64, so their *layouts* coincide (they differ
# in long-double format + calling convention, both delegated to the backend); Windows x64 is LLP64
# (`long` is 4) and 32-bit x86 is ILP32 (pointers are 4) -- the cases that change what the frontend
# lays out.
TARGETS: dict[str, TargetABI] = {
    "x86_64-linux":   TargetABI("x86_64-linux",   "x86_64-unknown-linux-gnu",   **_LP64),
    "aarch64-linux":  TargetABI("aarch64-linux",  "aarch64-unknown-linux-gnu",  **_LP64),
    "riscv64-linux":  TargetABI("riscv64-linux",  "riscv64-unknown-linux-gnu",  **_LP64),
    "x86_64-windows": TargetABI("x86_64-windows", "x86_64-pc-windows-msvc",
                                data_model="LLP64", long_size=4, pointer_size=8,
                                long_double_size=8, long_double_align=8),
    "i386-linux":     TargetABI("i386-linux",     "i386-unknown-linux-gnu",
                                data_model="ILP32", long_size=4, pointer_size=4,
                                long_double_size=12, long_double_align=4),
}

# The default target -- byte-identical to the constants ctype_model used before this module existed
# (LP64; an x86-64 / AArch64 Linux host). `scalar(name)` with no abi resolves against it.
HOST = TARGETS["x86_64-linux"]


def target(name: str) -> TargetABI:
    """Look up a target ABI by short name, with a clear error listing the matrix."""
    if name not in TARGETS:
        raise KeyError(f"unknown target {name!r}; choose from {', '.join(sorted(TARGETS))}")
    return TARGETS[name]


# --- §5.14 Phase 2 (the last area): the per-function CALL-ABI CONTRACT ---------------------------
#
# The call ABI was backend-delegated: abi.py computes sizes/offsets, the emitter materializes the
# frame, and nothing RECORDS the layout facts a cross-target object path would have to verify.
# The contract is that record -- the R12 lowering-contract pattern extended to the call ABI --
# and `verify_abi_contract` is its law: the recorded facts must agree with the target's data
# model (a pointer-kind parameter is pointer_size; a `long` scalar is long_size; the laid-out
# CType facts match the record). Advisory on the compile path (CompileResult.abi_diagnostics,
# like the lifetime/lowering laws); the MLIR rail carries the same record as bcir.abi_contract
# with the matching -bcir-verify R12 clause.

@dataclass(frozen=True)
class AbiContract:
    """One function's call-ABI record: (name, size, align) per parameter in C order, and the
    return slot's (size, align) -- (0, 0) for void. `target` names the TargetABI it was laid
    out for (the normative 5-entry matrix)."""

    fn: str
    target: str
    params: tuple                    # ((name, size, align), ...) in C order
    ret: tuple                       # (size, align); (0, 0) for void


def abi_contract_for(lf, abi: TargetABI) -> AbiContract:
    """Record the contract from an already-lowered function (the laid-out CTypes)."""
    params = tuple((nm, ct.size, ct.align) for nm, _rid, ct in lf.params)
    ret = (0, 0) if lf.ret_type is None or lf.ret_type.size == 0 else         (lf.ret_type.size, lf.ret_type.align)
    return AbiContract(fn=lf.name, target=abi.name, params=params, ret=ret)


def verify_abi_contract(contract: AbiContract, lf, abi: TargetABI) -> list[str]:
    """The R12-tagged call-ABI law: the recorded contract must agree with BOTH the laid-out
    CTypes and the target's data model. Returns message strings (the pipeline wraps them)."""
    msgs: list[str] = []
    if contract.target != abi.name:
        msgs.append(f"contract target {contract.target!r} != the unit target {abi.name!r}")
        return msgs
    if len(contract.params) != len(lf.params):
        msgs.append(f"contract arity {len(contract.params)} != {len(lf.params)} parameters")
        return msgs
    for (cn, csz, cal), (nm, _rid, ct) in zip(contract.params, lf.params):
        if cn != nm or csz != ct.size or cal != ct.align:
            msgs.append(f"parameter {nm!r}: contract ({cn!r}, {csz}, {cal}) != laid-out "
                        f"({nm!r}, {ct.size}, {ct.align})")
        if ct.kind in ("pointer", "funcptr") and csz != abi.pointer_size:
            msgs.append(f"parameter {nm!r}: a pointer-kind parameter must be the target's "
                        f"pointer size {abi.pointer_size}, contract says {csz}")
        if ct.kind == "scalar" and ct.name in ("long", "unsigned long") and csz != abi.long_size:
            msgs.append(f"parameter {nm!r}: a `long` parameter must be the target's long size "
                        f"{abi.long_size}, contract says {csz}")
    want_ret = (0, 0) if lf.ret_type is None or lf.ret_type.size == 0 else         (lf.ret_type.size, lf.ret_type.align)
    if contract.ret != want_ret:
        msgs.append(f"return slot: contract {contract.ret} != laid-out {want_ret}")
    return msgs
