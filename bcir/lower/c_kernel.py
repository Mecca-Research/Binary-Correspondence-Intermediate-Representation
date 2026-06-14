"""Portable C23 lowering for elementwise BCIR claims (the C kernel backend).

The K_BCIR-selected realization in the GEM StreamPack becomes a clean, portable
**C23** kernel: the chosen lane width drives the loop structure, the elementwise
op is emitted on the contracted element type, pointers are `restrict`-qualified
(the non-aliasing contract), and a scalar tail keeps the kernel bounds-safe for
any trip count. C is the universal kernel lingua franca (CUDA C / HIP / OpenCL C
/ ISPC / plain CPU C); this is the lowering that a driver-resident toolchain can
finish without BCIR-native instruction selection.

Library-first: `emit_kernel_c` is a pure `(plan) -> C string` function, reusable
both AOT (compiled + self-checked here by `compile_and_run_c`) and driver-embedded
(emit, hand to the resident compiler). `emit_selfcheck_c` wraps the kernel with a
self-checking `main` for the AOT path.

C23 conformance (ISO/IEC 9899:2023): `restrict` on the pointer parameters
(6.7.3.1, the aliasing guarantee), file-scope `static_assert` on the element size
(6.7.11), `#pragma STDC FP_CONTRACT OFF` for reproducible float results (7.12.2 --
no fused-multiply-add contraction, so the self-check's exact compare holds),
`<stddef.h>` `size_t` trip counts, and `<stdint.h>` `int32_t` for the FP-less
(eBPF-style) element type. Emit with `-std=c23` (clang) or `-std=c2x` (gcc). The
R12 lowering contract -- lane geometry, bounds, precision -- is checked by
`verify.verify_c_lowering`.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

from ..gem import hydrate
from ..model import Module, Opcode
from ..kbcir.realize import RealizationResult
from .llvm import find_elementwise

# Opcode -> C operator (the elementwise binary ops the lowering supports).
C_OP = {Opcode.ADD: "+", Opcode.SUB: "-", Opcode.MUL: "*"}


def _ctype(elem: str) -> str:
    return "int32_t" if elem == "i32" else "float"


def _selected_segment(module: Module, result: RealizationResult, claim):
    """The hydrated StreamPack lane segment for `claim` (the GEM artifact the C
    kernel lowers from), or None if the pack has no matching segment."""
    pack = hydrate(module, result)
    for seg in pack.segments:
        if seg.claim_id == claim.id:
            return seg
    return None


def emit_kernel_c(module: Module, result: RealizationResult, fn_name: str = "bcir_kernel",
                  elem: str = "f32", width_override: int | None = None) -> str:
    """Emit a portable C23 kernel for the selected elementwise realization, driven
    by the GEM StreamPack segment (lane width -> loop structure, op, resources).
    `elem` is "f32" (float) or "i32" (int32_t for FP-less targets); `width_override`
    forces a width. Pure: no I/O, reusable AOT or driver-embedded."""
    claim, cand = find_elementwise(module, result)
    seg = _selected_segment(module, result, claim)
    w = int(width_override) if width_override else int(seg.width if seg else cand.width)
    w = w if w >= 1 else 1
    op = C_OP[claim.opcode]
    ctype = _ctype(elem)

    # restrict is sound only when the read and write resources are disjoint (no
    # aliasing). The elementwise contract (2 reads, 1 write) makes this the norm.
    reads = tuple(seg.reads) if seg else tuple(claim.rd)
    writes = tuple(seg.writes) if seg else tuple(claim.wr)
    rqual = " restrict" if not (set(reads) & set(writes)) else ""

    includes = "#include <stddef.h>\n"
    fp_pragma = ""
    if elem == "i32":
        includes += "#include <stdint.h>\n"
    else:
        # reproducible float: disallow contraction so a + b is a single rounding.
        fp_pragma = "#pragma STDC FP_CONTRACT OFF\n"

    head = (
        f"/* BCIR -> portable C23 kernel (lower_contract). op={claim.op or op} "
        f"lane={cand.lane.name} width={w} elem={ctype} "
        f"(K_BCIR-selected; StreamPack {seg.name if seg else '-'}). */\n"
        f"{includes}"
        f'static_assert(sizeof({ctype}) == 4, "BCIR {ctype} kernel needs a 4-byte element");\n'
        f"{fp_pragma}\n"
    )
    sig = (f"void {fn_name}(const {ctype} *{rqual} A, const {ctype} *{rqual} B,\n"
           f"             {ctype} *{rqual} C, size_t n)")

    if w == 1:
        body = (f"{sig} {{\n"
                f"  for (size_t i = 0; i < n; ++i)\n"
                f"    C[i] = A[i] {op} B[i];\n"
                f"}}\n")
    else:
        body = (
            f"{sig} {{\n"
            f"  size_t i = 0;\n"
            f"  /* width-{w} lane: a fixed-trip inner loop the compiler vectorizes "
            f"(the K_BCIR-selected width) */\n"
            f"  for (; i + {w}u <= n; i += {w}u)\n"
            f"    for (size_t j = 0; j < {w}u; ++j)\n"
            f"      C[i + j] = A[i + j] {op} B[i + j];\n"
            f"  /* scalar tail: bounds-safe for any trip count (the strict bounds contract) */\n"
            f"  for (; i < n; ++i)\n"
            f"    C[i] = A[i] {op} B[i];\n"
            f"}}\n"
        )
    return head + body


def emit_selfcheck_c(module: Module, result: RealizationResult, fn_name: str = "bcir_kernel",
                     elem: str = "f32") -> str:
    """Wrap the kernel with a self-checking C23 `main` (the AOT path). Checks the
    selected count and a non-divisible size (count + 7) to exercise the tail."""
    claim, _ = find_elementwise(module, result)
    n = max(1, claim.count)
    op = C_OP[claim.opcode]
    ctype = _ctype(elem)
    kernel = emit_kernel_c(module, result, fn_name, elem)
    if elem == "i32":
        init = "A[i] = (int32_t)i; B[i] = (int32_t)(2u * i); C[i] = -1;"
        fmt = "%d"
    else:
        init = "A[i] = (float)i; B[i] = 2.0f * (float)i; C[i] = -1.0f;"
        fmt = "%g"

    return (
        "#include <stddef.h>\n#include <stdio.h>\n#include <stdlib.h>\n\n"
        + kernel
        + f"""
static int check(size_t n) {{
  {ctype} *A = malloc(n * sizeof *A), *B = malloc(n * sizeof *B), *C = malloc(n * sizeof *C);
  if (!A || !B || !C) {{ free(A); free(B); free(C); return 2; }}
  for (size_t i = 0; i < n; ++i) {{ {init} }}
  {fn_name}(A, B, C, n);
  int rc = 0;
  for (size_t i = 0; i < n; ++i) {{
    {ctype} want = A[i] {op} B[i];
    if (C[i] != want) {{ printf("FAIL n=%zu i=%zu got {fmt} want {fmt}\\n", n, i, C[i], want); rc = 1; break; }}
  }}
  free(A); free(B); free(C);
  return rc;
}}

int main(void) {{
  if (check({n}u)) return 1;        /* the selected count */
  if (check({n}u + 7u)) return 1;   /* a non-divisible size: exercises the scalar tail */
  puts("OK {fn_name}");
  return 0;
}}
"""
    )


def compile_and_run_c(module: Module, result: RealizationResult, fn_name: str = "bcir_kernel",
                      elem: str = "f32", workdir: str | None = None) -> tuple[bool, str]:
    """Emit the self-checking C, compile it (C23: clang -std=c23, else gcc -std=c2x),
    run, and check it printed OK. Returns (ok, combined_output)."""
    cc = _which("clang") or _which("cc") or _which("gcc")
    if cc is None:
        return False, "no C compiler on PATH"

    created = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="bcir-ckernel-")
    try:
        src = os.path.join(workdir, "kernel_check.c")
        exe = os.path.join(workdir, "prog")
        with open(src, "w") as f:
            f.write(emit_selfcheck_c(module, result, fn_name, elem))
        build = None
        for std in ("-std=c23", "-std=c2x"):
            build = subprocess.run([cc, std, "-O2", "-Wall", "-Wextra", src, "-o", exe],
                                   capture_output=True, text=True)
            if build.returncode == 0:
                break
        if build is None or build.returncode != 0:
            return False, "C build failed:\n" + (build.stdout + build.stderr if build else "")
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
