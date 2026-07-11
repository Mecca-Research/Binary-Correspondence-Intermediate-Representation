"""Textual LLVM IR lowering for BCIR's single-claim elementwise subset.

This realizes LangRef Milestones 5-7 "in miniature": the K_BCIR-selected lane
width becomes a real per-lane LLVM kernel (`<W x float>` vector loads/adds/stores
for U/UX lanes, scalar for width 1), which clang compiles and runs on the host
today. This is deliberately a partial LLVM AOT/JIT backend: it accepts exactly
one selected, executable 2-read/1-write add/sub/mul claim and rejects arbitrary
graphs instead of silently truncating them. The emitter only produces standard
SSA instructions -- no constant-expression-over-SSA, no invented opcodes.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

from ..model import Module, Opcode
from ..kbcir.realize import Candidate, RealizationResult
from ..toolchain import resolve_llvm_tools

_FOP = {Opcode.ADD: ("fadd", "+"), Opcode.SUB: ("fsub", "-"), Opcode.MUL: ("fmul", "*")}
_IOP = {Opcode.ADD: "add", Opcode.SUB: "sub", Opcode.MUL: "mul"}  # integer (e.g. eBPF)
_NON_EXECUTABLE = {Opcode.NOP, Opcode.PHASE_ENTER, Opcode.PHASE_LEAVE, Opcode.PROV_NOTE}


def find_elementwise(module: Module, result: RealizationResult) -> tuple:
    """Return the one supported selected executable claim and its candidate.

    Public: `bcir.verify.verify_lowering` (law R12) uses the same selection
    contract to check the emitted kernel against the K_BCIR-chosen realization.
    Bookkeeping-only claims are ignored; every executable graph claim counts,
    including unsupported operations and barriers. This prevents an arbitrary
    graph from being misrepresented by lowering only its first selected convenient
    node when a malformed or partial realization omits another claim.
    """
    by_claim = result.by_claim()
    executable = []
    for ph in module.phases:
        for claim in ph.claims:
            cand = by_claim.get(claim.id)
            if claim.opcode not in _NON_EXECUTABLE:
                executable.append((claim, cand))

    if len(executable) != 1:
        raise NotImplementedError(
            "the single-claim elementwise LLVM AOT/JIT subset requires exactly one "
            f"selected executable claim; found {len(executable)}"
        )

    claim, cand = executable[0]
    if cand is None:
        raise NotImplementedError(
            "the single-claim elementwise LLVM AOT/JIT subset requires its "
            f"executable claim {claim.id} to have a selected realization"
        )
    if claim.opcode not in _FOP or len(claim.rd) != 2 or len(claim.wr) != 1:
        raise NotImplementedError(
            "the single-claim elementwise LLVM AOT/JIT subset supports only a "
            "2-read/1-write add, sub, or mul claim; "
            f"claim {claim.id} is {claim.opcode.name.lower()} with "
            f"{len(claim.rd)} reads and {len(claim.wr)} writes"
        )
    return claim, cand


# Back-compat alias (pre-R12 internal name).
_find_elementwise = find_elementwise


def emit_kernel_ll(module: Module, result: RealizationResult, fn_name: str = "bcir_kernel",
                   elem: str = "f32", width_override: int | None = None) -> str:
    """Emit a legal LLVM IR kernel. `elem` is "f32" (float) or "i32" (integer, e.g.
    for FP-less targets like eBPF); `width_override` forces a vector/scalar width."""
    claim, cand = _find_elementwise(module, result)
    n = max(1, claim.count)
    base_w = width_override if width_override else cand.width
    w = base_w if (base_w >= 1 and n % base_w == 0) else 1
    if elem == "i32":
        ety = "i32"
        op_ll = _IOP[claim.opcode]
    else:
        ety = "float"
        op_ll = _FOP[claim.opcode][0]

    head = (
        f"; BCIR -> LLVM IR (legal-IR-only). op={claim.op or op_ll} "
        f"lane={cand.lane.name} width={w} elem={ety} (K_BCIR-selected; candidate={cand.name})\n"
        f"source_filename = \"bcir.{module.name}.ll\"\n\n"
    )

    if w == 1:
        body = f"""define void @{fn_name}(ptr noalias %A, ptr noalias %B, ptr noalias %C, i64 %n) {{
entry:
  %empty = icmp sle i64 %n, 0
  br i1 %empty, label %exit, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %pa = getelementptr inbounds {ety}, ptr %A, i64 %i
  %pb = getelementptr inbounds {ety}, ptr %B, i64 %i
  %pc = getelementptr inbounds {ety}, ptr %C, i64 %i
  %a = load {ety}, ptr %pa, align 4
  %b = load {ety}, ptr %pb, align 4
  %c = {op_ll} {ety} %a, %b
  store {ety} %c, ptr %pc, align 4
  %inext = add nuw nsw i64 %i, 1
  %done = icmp sge i64 %inext, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}}
"""
    else:
        vty = f"<{w} x {ety}>"
        body = f"""define void @{fn_name}(ptr noalias %A, ptr noalias %B, ptr noalias %C, i64 %n) {{
entry:
  %empty = icmp sle i64 %n, 0
  br i1 %empty, label %exit, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %pa = getelementptr inbounds {ety}, ptr %A, i64 %i
  %pb = getelementptr inbounds {ety}, ptr %B, i64 %i
  %pc = getelementptr inbounds {ety}, ptr %C, i64 %i
  %va = load {vty}, ptr %pa, align 4
  %vb = load {vty}, ptr %pb, align 4
  %vc = {op_ll} {vty} %va, %vb
  store {vty} %vc, ptr %pc, align 4
  %inext = add nuw nsw i64 %i, {w}
  %done = icmp sge i64 %inext, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}}
"""
    return head + body


def emit_harness_c(module: Module, result: RealizationResult, fn_name: str = "bcir_kernel") -> str:
    claim, _ = _find_elementwise(module, result)
    n = max(1, claim.count)
    _, op_c = _FOP[claim.opcode]
    return f"""#include <stdio.h>
#include <stdlib.h>

extern void {fn_name}(const float *A, const float *B, float *C, long n);

int main(void) {{
  long n = {n};
  float *A = (float *)malloc((size_t)n * sizeof(float));
  float *B = (float *)malloc((size_t)n * sizeof(float));
  float *C = (float *)malloc((size_t)n * sizeof(float));
  if (!A || !B || !C) return 2;
  for (long i = 0; i < n; i++) {{ A[i] = (float)i; B[i] = 2.0f * (float)i; C[i] = -1.0f; }}
  {fn_name}(A, B, C, n);
  for (long i = 0; i < n; i++) {{
    float want = A[i] {op_c} B[i];
    if (C[i] != want) {{ printf("FAIL at %ld: got %f want %f\\n", i, C[i], want); return 1; }}
  }}
  printf("OK {fn_name} n=%ld\\n", n);
  return 0;
}}
"""


def compile_and_run(
    module: Module,
    result: RealizationResult,
    fn_name: str = "bcir_kernel",
    workdir: str | None = None,
) -> tuple[bool, str]:
    """Emit kernel.ll + harness.c, compile with clang, run, and self-check.

    Returns (ok, combined_output). ok is True only if clang built the program and
    it printed OK with exit code 0.
    """
    llvm = resolve_llvm_tools("clang", pipeline="AOT")
    if not llvm.ok:
        return False, llvm.message
    clang = llvm.paths["clang"]

    created = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="bcir-lower-")
    try:
        ll = os.path.join(workdir, "kernel.ll")
        cc = os.path.join(workdir, "harness.c")
        exe = os.path.join(workdir, "prog")
        with open(ll, "w") as f:
            f.write(emit_kernel_ll(module, result, fn_name))
        with open(cc, "w") as f:
            f.write(emit_harness_c(module, result, fn_name))
        build = subprocess.run([clang, "-O2", cc, ll, "-o", exe],
                               capture_output=True, text=True)
        if build.returncode != 0:
            return False, "clang build failed:\n" + build.stdout + build.stderr
        run = subprocess.run([exe], capture_output=True, text=True)
        ok = run.returncode == 0 and "OK" in run.stdout
        return ok, run.stdout + run.stderr
    finally:
        if created:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
