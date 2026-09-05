"""Textual LLVM IR lowering for BCIR's single-claim elementwise subset.

This realizes LangRef Milestones 5-7 "in miniature": the K_BCIR-selected lane
width becomes a real per-lane LLVM kernel (`<W x float>` vector loads/adds/stores
for U/UX lanes, scalar for width 1), which clang compiles and runs on the host
today. This is deliberately a partial LLVM AOT/JIT backend: it accepts exactly
one selected, executable 2-read/1-write add/sub/mul claim and rejects arbitrary
graphs instead of silently truncating them. The emitter only produces standard
SSA instructions -- no constant-expression-over-SSA, no invented opcodes.

The runtime-`n` tail contract (S0-8). The kernel takes its trip count `n` at RUNTIME,
so the claim's `count` is only the expected one. A vector kernel therefore runs its
`<W x ...>` loop over the largest multiple of W not above `n` (`n & -W`, W a power of
two) and finishes the remainder in a scalar epilogue: bounds-safe for any `n`, the
selected width kept whatever the count. The parent emitter legalized a non-divisible
compile-time count to scalar and otherwise stepped the vector loop to `n` -- a runtime
`n` that was not a multiple of W read and wrote past the buffers (the harness now calls
every kernel with `count + 7`, a sub-width count and 0 behind canaries, so that miscompile
cannot come back). R12 (`verify.verify_lowering`) holds the contract: the declared width
is the selected one and a vector kernel carries the mask and the scalar epilogue.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

from ..model import Module, Opcode, StrideClass
from ..kbcir.realize import Candidate, RealizationResult
from ..toolchain import resolve_llvm_tools

_FOP = {Opcode.ADD: ("fadd", "+"), Opcode.SUB: ("fsub", "-"), Opcode.MUL: ("fmul", "*")}
_IOP = {Opcode.ADD: "add", Opcode.SUB: "sub", Opcode.MUL: "mul"}  # integer (e.g. eBPF)
_NON_EXECUTABLE = {Opcode.NOP, Opcode.PHASE_ENTER, Opcode.PHASE_LEAVE, Opcode.PROV_NOTE}


#: The access patterns whose element order the emitted `for (i...) C[i] = A[i] op B[i]`
#: body actually walks. Anything else needs addressing the emitters do not generate.
_UNIT_STRIDE_CLASSES = frozenset({StrideClass.UNIT, StrideClass.SCALAR})


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
    # Both emitters address the operands as `A[i]`, `B[i]`, `C[i]` unconditionally. A claim
    # declaring `offset=8` or `stride_k=4` was ACCEPTED and then lowered to that same body,
    # and R12 -- which compares the emitted text against the same subset contract -- called
    # it clean, so a kernel computing a different function than the claim declares carried a
    # verified attestation. Refusing is the honest boundary until the addressing is actually
    # emitted: a subset that says what it does not cover is a subset; one that silently
    # widens is a miscompile.
    if claim.offset != 0:
        raise NotImplementedError(
            f"the elementwise lowering subset addresses operands from index 0; claim "
            f"{claim.id} declares offset={claim.offset}, which the emitted body does not "
            f"apply"
        )
    if claim.stride_k != 1 or claim.stride_class not in _UNIT_STRIDE_CLASSES:
        raise NotImplementedError(
            f"the elementwise lowering subset emits a unit-stride walk; claim "
            f"{claim.id} declares stride_k={claim.stride_k} and "
            f"stride_class={claim.stride_class.name}, which the emitted body does not "
            f"apply"
        )
    return claim, cand


# Back-compat alias (pre-R12 internal name).
_find_elementwise = find_elementwise


def _alias_params(claim) -> str:
    """The parameter list, with `noalias` only where the RIDs prove it.

    LLVM's `noalias` is an ASSERTION the caller must honour, not a hint: the optimizer
    reorders loads and stores across it, and violating it is undefined behaviour. This
    emitter used to write it on all three pointers unconditionally, which is a FALSE
    assertion the moment a claim writes a resource it also reads --

        Claim(rd=(1, 2), wr=(1,))   ->   A and C are both resource 1

    -- and `A[i] = A[i] + B[i]` is an ordinary in-place graph, not an exotic one. The
    subset gate accepts it (two reads, one write) and said A, B and C never alias.

    BCIR does not have to infer any of this. A claim DECLARES its read and write RIDs, so
    disjointness is a fact here rather than an analysis result, which is strictly better
    information than an alias analysis could recover downstream. Emitting it accurately is
    also what lets LLVM keep the win: a pointer that really is unaliased still gets the
    attribute and still gets reordered.
    """
    reads = tuple(claim.rd)
    writes = tuple(claim.wr)
    rids = (reads[0], reads[1], writes[0])
    names = ("A", "B", "C")
    parts = []
    for index, (name, rid) in enumerate(zip(names, rids)):
        shared = any(other == rid for position, other in enumerate(rids) if position != index)
        parts.append(f"ptr {'' if shared else 'noalias '}%{name}")
    return ", ".join(parts) + ", i64 %n"


def emit_kernel_ll(
    module: Module,
    result: RealizationResult,
    fn_name: str = "bcir_kernel",
    elem: str = "f32",
    width_override: int | None = None,
) -> str:
    """Emit a legal LLVM IR kernel. `elem` is "f32" (float) or "i32" (integer, e.g.
    for FP-less targets like eBPF); `width_override` forces a vector/scalar width."""
    claim, cand = _find_elementwise(module, result)
    base_w = width_override if width_override else cand.width
    if base_w < 1 or (base_w & (base_w - 1)):
        raise NotImplementedError(
            f"the elementwise lowering subset emits a power-of-two lane width; candidate "
            f"{cand.name} selects {base_w}"
        )
    # The selected width is realized whatever the count: a runtime `n` that is not a
    # multiple of the width finishes in the scalar epilogue (the tail contract).
    w = base_w
    if elem == "i32":
        ety = "i32"
        op_ll = _IOP[claim.opcode]
    else:
        ety = "float"
        op_ll = _FOP[claim.opcode][0]

    params = _alias_params(claim)
    head = (
        f"; BCIR -> LLVM IR (legal-IR-only). op={claim.op or op_ll} "
        f"lane={cand.lane.name} width={w} elem={ety} "
        f"epilogue={'scalar' if w > 1 else 'none'} "
        f"(K_BCIR-selected; candidate={cand.name})\n"
        f'source_filename = "bcir.{module.name}.ll"\n\n'
    )

    if w == 1:
        body = f"""define void @{fn_name}({params}) {{
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
        # The vector loop runs over nvec = n & -W (the largest multiple of W not above n);
        # the scalar epilogue finishes [nvec, n). Both guards compare against %n, so a
        # trip count of any value -- including one below W, or zero -- is bounds-safe.
        vty = f"<{w} x {ety}>"
        body = f"""define void @{fn_name}({params}) {{
entry:
  %empty = icmp sle i64 %n, 0
  br i1 %empty, label %exit, label %vec.check
vec.check:
  %nvec = and i64 %n, -{w}
  %novec = icmp eq i64 %nvec, 0
  br i1 %novec, label %tail.check, label %vec
vec:
  %i = phi i64 [ 0, %vec.check ], [ %inext, %vec ]
  %pa = getelementptr inbounds {ety}, ptr %A, i64 %i
  %pb = getelementptr inbounds {ety}, ptr %B, i64 %i
  %pc = getelementptr inbounds {ety}, ptr %C, i64 %i
  %va = load {vty}, ptr %pa, align 4
  %vb = load {vty}, ptr %pb, align 4
  %vc = {op_ll} {vty} %va, %vb
  store {vty} %vc, ptr %pc, align 4
  %inext = add nuw nsw i64 %i, {w}
  %vdone = icmp sge i64 %inext, %nvec
  br i1 %vdone, label %tail.check, label %vec
tail.check:
  %tdone0 = icmp sge i64 %nvec, %n
  br i1 %tdone0, label %exit, label %tail
tail:
  %j = phi i64 [ %nvec, %tail.check ], [ %jnext, %tail ]
  %ta = getelementptr inbounds {ety}, ptr %A, i64 %j
  %tb = getelementptr inbounds {ety}, ptr %B, i64 %j
  %tc = getelementptr inbounds {ety}, ptr %C, i64 %j
  %a = load {ety}, ptr %ta, align 4
  %b = load {ety}, ptr %tb, align 4
  %c = {op_ll} {ety} %a, %b
  store {ety} %c, ptr %tc, align 4
  %jnext = add nuw nsw i64 %j, 1
  %tdone = icmp sge i64 %jnext, %n
  br i1 %tdone, label %exit, label %tail
exit:
  ret void
}}
"""
    return head + body


def harness_trip_counts(module: Module, result: RealizationResult) -> tuple:
    """The runtime trip counts the self-check harness drives the kernel with: the
    planned count, a non-divisible one (`count + 7`), a count below one vector, and
    zero. A kernel that steps its vector loop to `n` writes past the buffers on the
    second and third; the harness's canaries turn that into a FAIL, not silence."""
    claim, cand = _find_elementwise(module, result)
    n = max(1, claim.count)
    return (n, n + 7, max(1, cand.width - 1), 0)


def emit_harness_c(module: Module, result: RealizationResult, fn_name: str = "bcir_kernel") -> str:
    """The C self-check harness: every trip count in `harness_trip_counts`, each behind
    a canary region past `n` that the kernel must leave untouched (the tail contract)."""
    claim, _ = _find_elementwise(module, result)
    trips = ", ".join(str(t) for t in harness_trip_counts(module, result))
    _, op_c = _FOP[claim.opcode]
    return f"""#include <stdio.h>
#include <stdlib.h>

extern void {fn_name}(const float *A, const float *B, float *C, long n);

#define CANARY 64
static const float SENTINEL = -7777777.0f;

/* Run the kernel at trip count n with CANARY elements of headroom past n on every
 * array; a write past n is an unmasked tail, and it fails here rather than corrupting
 * the heap silently. */
static int check(long n) {{
  long total = n + CANARY;
  float *A = (float *)malloc((size_t)total * sizeof(float));
  float *B = (float *)malloc((size_t)total * sizeof(float));
  float *C = (float *)malloc((size_t)total * sizeof(float));
  if (!A || !B || !C) return 2;
  for (long i = 0; i < total; i++) {{ A[i] = (float)i; B[i] = 2.0f * (float)i; C[i] = SENTINEL; }}
  {fn_name}(A, B, C, n);
  for (long i = 0; i < n; i++) {{
    float want = A[i] {op_c} B[i];
    if (C[i] != want) {{ printf("FAIL n=%ld at %ld: got %f want %f\\n", n, i, C[i], want); return 1; }}
  }}
  for (long i = n; i < total; i++) {{
    if (C[i] != SENTINEL) {{ printf("FAIL n=%ld: wrote past n at %ld (an unmasked tail)\\n", n, i); return 1; }}
  }}
  free(A); free(B); free(C);
  return 0;
}}

int main(void) {{
  long trips[] = {{ {trips} }};
  for (unsigned t = 0; t < sizeof trips / sizeof trips[0]; t++) {{
    int rc = check(trips[t]);
    if (rc) return rc;
  }}
  printf("OK {fn_name} trips={trips}\\n");
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
        build = subprocess.run([clang, "-O2", cc, ll, "-o", exe], capture_output=True, text=True)
        if build.returncode != 0:
            return False, "clang build failed:\n" + build.stdout + build.stderr
        run = subprocess.run([exe], capture_output=True, text=True)
        ok = run.returncode == 0 and "OK" in run.stdout
        return ok, run.stdout + run.stderr
    finally:
        if created:
            import shutil

            shutil.rmtree(workdir, ignore_errors=True)
