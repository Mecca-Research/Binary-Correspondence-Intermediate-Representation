"""Textual LLVM IR lowering for elementwise BCIR claims.

This realizes LangRef Milestones 5-7 "in miniature": the K_BCIR-selected lane
width becomes a real per-lane LLVM kernel (`<W x float>` vector loads/adds/stores
for U/UX lanes, scalar for width 1), which clang compiles and runs on the host
today. The emitter only produces standard SSA instructions -- no
constant-expression-over-SSA, no invented opcodes.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

from ..model import Module, Opcode
from ..kbcir.realize import Candidate, RealizationResult

_FOP = {Opcode.ADD: ("fadd", "+"), Opcode.SUB: ("fsub", "-"), Opcode.MUL: ("fmul", "*")}


def _find_elementwise(module: Module, result: RealizationResult) -> tuple:
    """Return (claim, candidate) for the single 2-read/1-write elementwise claim."""
    by_claim = result.by_claim()
    for ph in module.phases:
        for claim in ph.claims:
            if claim.opcode in _FOP and len(claim.rd) == 2 and len(claim.wr) == 1:
                cand = by_claim.get(claim.id)
                if cand is not None:
                    return claim, cand
    raise NotImplementedError(
        "LLVM lowering currently supports a single 2-read elementwise binary claim "
        "(e.g. vector_add); this module has none selected."
    )


def emit_kernel_ll(module: Module, result: RealizationResult, fn_name: str = "bcir_kernel") -> str:
    claim, cand = _find_elementwise(module, result)
    n = max(1, claim.count)
    w = cand.width if (cand.width >= 1 and n % cand.width == 0) else 1
    op_ll, _ = _FOP[claim.opcode]

    head = (
        f"; BCIR -> LLVM IR (legal-IR-only). op={claim.op or op_ll} "
        f"lane={cand.lane.name} width={w} (K_BCIR-selected; candidate={cand.name})\n"
        f"source_filename = \"bcir.{module.name}.ll\"\n\n"
    )

    if w == 1:
        body = f"""define void @{fn_name}(ptr noalias %A, ptr noalias %B, ptr noalias %C, i64 %n) {{
entry:
  %empty = icmp sle i64 %n, 0
  br i1 %empty, label %exit, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %pa = getelementptr inbounds float, ptr %A, i64 %i
  %pb = getelementptr inbounds float, ptr %B, i64 %i
  %pc = getelementptr inbounds float, ptr %C, i64 %i
  %a = load float, ptr %pa, align 4
  %b = load float, ptr %pb, align 4
  %c = {op_ll} float %a, %b
  store float %c, ptr %pc, align 4
  %inext = add nuw nsw i64 %i, 1
  %done = icmp sge i64 %inext, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}}
"""
    else:
        vty = f"<{w} x float>"
        body = f"""define void @{fn_name}(ptr noalias %A, ptr noalias %B, ptr noalias %C, i64 %n) {{
entry:
  %empty = icmp sle i64 %n, 0
  br i1 %empty, label %exit, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %pa = getelementptr inbounds float, ptr %A, i64 %i
  %pb = getelementptr inbounds float, ptr %B, i64 %i
  %pc = getelementptr inbounds float, ptr %C, i64 %i
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
    clang = _which("clang")
    if clang is None:
        return False, "clang not found on PATH"

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


def _which(name: str) -> str | None:
    from shutil import which
    return which(name)
