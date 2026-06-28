"""G3: the SoA<->AoS layout pivot lowering -- a kernel that HONORS the chosen layout, with proven-identical
results under either layout.

``kbcir.layout`` prices SoA vs AoS for a multi-field resource and chooses the cheaper one (a ``LayoutPlan``).
This module is where that choice flows into emitted C: ``emit_layout_kernel_c`` lowers a concrete multi-field
computation under the CHOSEN layout's addressing, and ``emit_layout_selfcheck_c`` wraps it in a self-check
that runs the SAME computation under BOTH layouts and asserts the results are BIT-IDENTICAL -- the
reference-invariance backbone of G3 (a layout change moves addresses, never values).

THE INVARIANCE, made concrete. The example computation over an F-field record array of N records is, per
record r:

    out[r] = sum over fields f in [0,F) of  data[r, f] * weight[f]      (a whole-record reduction)

The ONLY thing the layout changes is how ``data[r, f]`` is addressed in the flat backing array:

    SoA index:  soa_index(r, f) = f * N + r      (field f's own contiguous array, record r within it)
    AoS index:  aos_index(r, f) = r * F + f      (record r's contiguous block, field f within it)

Both index maps are bijections over the same N*F elements -- a PERMUTATION of the storage -- so as long as
the backing array is filled CONSISTENTLY with its layout (the self-check fills both), the value
``data[r, f]`` read at ``out[r]``'s accumulation is identical, hence ``out`` is bit-identical. (The sum is
over a fixed per-record order f = 0..F-1 in BOTH layouts, so even for floating point there is no
reassociation -- the additions happen in the same order; the result is bit-exact, not merely close.)

This is the demonstrated boundary the task allows: a full claim-graph address rewrite for AoS across every
op is a larger change; here we price + certify the selection (``kbcir.layout``), flow the chosen layout into
the emit (the addressing macro below + the ``Resource.layout`` field + the MLIR emitter), and PROVE the two
layouts compute identical values on a concrete multi-field kernel. The address map is the single seam an
MLIR ``#bcir.layout`` pass must mirror (``soa_index`` vs ``aos_index``).
"""

from __future__ import annotations

import os
import subprocess
import tempfile

from ..kbcir.layout import AOS, SOA, LayoutPlan


def _ctype(elem: str) -> str:
    return "int32_t" if elem == "i32" else "float"


def _index_expr(layout: str, fields: int, records: int) -> tuple[str, str]:
    """The C address expression for ``data[r, f]`` under ``layout`` + a human label. The two are a
    permutation of the same N*F storage -- only the addressing differs, never the value read."""
    if layout == AOS:
        return f"(r) * {fields}u + (f)", f"AoS: data[r*F + f] (F={fields}, record-contiguous)"
    return f"(f) * {records}u + (r)", f"SoA: data[f*N + r] (N={records}, field-contiguous)"


def emit_layout_kernel_c(plan: LayoutPlan, fn_name: str = "bcir_layout", elem: str = "f32",
                         records: int | None = None) -> str:
    """Emit the multi-field reduction kernel ``out[r] = sum_f data[r,f]*weight[f]`` lowered under the layout
    the cost model CHOSE (``plan.layout``) -- the emit HONORS the priced choice instead of a hard-coded
    ``soa`` constant. The only thing the chosen layout changes is the ``DATA(r,f)`` addressing macro
    (``f*N+r`` for SoA, ``r*F+f`` for AoS); the computed ``out`` is layout-independent (the invariance the
    self-check proves). ``records`` defaults to the workload size carried by the plan's accesses."""
    fields = plan.fields
    if fields < 1:
        raise ValueError(f"layout kernel needs fields >= 1; got {fields}")
    n = records if records is not None else (max((a.records for a in plan.accesses), default=1))
    n = max(1, n)
    ctype = _ctype(elem)
    idx, label = _index_expr(plan.layout, fields, n)
    includes = "#include <stddef.h>\n" + ("#include <stdint.h>\n" if elem == "i32" else "")
    fp = "" if elem == "i32" else "#pragma STDC FP_CONTRACT OFF\n"
    return (
        f"/* BCIR -> G3 SoA<->AoS layout pivot kernel (cost-model-CHOSEN layout={plan.layout}; "
        f"kbcir.layout). out[r] = sum_f data[r,f]*weight[f] over N={n} records, F={fields} fields. "
        f"The chosen layout sets ONLY the addressing ({label}); the computed out is layout-independent "
        f"-- a layout change never changes values (the G3 invariance). */\n"
        f"{includes}{fp}"
        f"#define BCIR_F {fields}u\n"
        f"#define BCIR_N {n}u\n"
        f"/* the chosen-layout address of field f of record r in the flat backing array */\n"
        f"#define DATA(r, f) ({idx})\n"
        f"void {fn_name}(const {ctype} *restrict data, const {ctype} *restrict weight,\n"
        f"             {ctype} *restrict out, size_t n) {{\n"
        f"  for (size_t r = 0; r < n; ++r) {{\n"
        f"    {ctype} acc = 0;\n"
        f"    for (size_t f = 0; f < BCIR_F; ++f)\n"
        f"      acc += data[DATA(r, f)] * weight[f];\n"
        f"    out[r] = acc;\n"
        f"  }}\n"
        f"}}\n"
        f"#undef DATA\n"
        f"#undef BCIR_F\n"
        f"#undef BCIR_N\n"
    )


def emit_layout_selfcheck_c(fields: int, records: int, elem: str = "f32") -> str:
    """Emit a self-checking ``main`` that proves REFERENCE-INVARIANCE: it fills a SoA backing array and an
    AoS backing array with the SAME logical ``data[r,f]`` values (each in its own layout's order), runs the
    SoA-addressed kernel and the AoS-addressed kernel, and asserts the two ``out`` arrays are BIT-IDENTICAL
    -- the hard G3 invariant that a layout change never changes a computed value. (The per-record sum is over
    the same fixed field order f=0..F-1 in both, so even for f32 there is no reassociation: bit-exact.)

    Distinct kernels are emitted for the two layouts (``bcir_layout_soa`` / ``bcir_layout_aos``), each via
    ``emit_layout_kernel_c`` with the corresponding ``LayoutPlan`` layout -- so the self-check exercises the
    REAL emit path, not a hand-written addressing."""
    if fields < 1 or records < 1:
        raise ValueError(f"layout self-check needs fields>=1, records>=1; got F={fields} N={records}")
    ctype = _ctype(elem)
    soa_plan = LayoutPlan(rid=0, fields=fields, layout=SOA, soa_cost=0, aos_cost=0, accesses=())
    aos_plan = LayoutPlan(rid=0, fields=fields, layout=AOS, soa_cost=0, aos_cost=0, accesses=())
    soa_kernel = emit_layout_kernel_c(soa_plan, "bcir_layout_soa", elem, records)
    aos_kernel = emit_layout_kernel_c(aos_plan, "bcir_layout_aos", elem, records)
    if elem == "i32":
        val = "(int32_t)((r * 7u + f * 3u) % 17u) - 8"
        wexpr = "(int32_t)((f * 5u + 1u) % 9u) - 4"
        cmp = "soa_out[r] != aos_out[r]"
        fmt = "%d"
    else:
        val = "(float)((int)((r * 7u + f * 3u) % 17u) - 8)"
        wexpr = "(float)((int)((f * 5u + 1u) % 9u) - 4)"
        cmp = "soa_out[r] != aos_out[r]"   # integer-valued floats -> exact, so == is sound
        fmt = "%g"
    return (
        "#include <stddef.h>\n#include <stdio.h>\n#include <stdlib.h>\n"
        + ("#include <stdint.h>\n" if elem == "i32" else "")
        + "\n" + soa_kernel + "\n" + aos_kernel
        + f"""
int main(void) {{
  const size_t F = {fields}u, N = {records}u;
  {ctype} *soa = malloc(N * F * sizeof *soa), *aos = malloc(N * F * sizeof *aos);
  {ctype} *weight = malloc(F * sizeof *weight);
  {ctype} *soa_out = malloc(N * sizeof *soa_out), *aos_out = malloc(N * sizeof *aos_out);
  if (!soa || !aos || !weight || !soa_out || !aos_out) {{
    free(soa); free(aos); free(weight); free(soa_out); free(aos_out); return 2;
  }}
  /* fill BOTH backing arrays with the SAME logical data[r,f], each in its layout's order */
  for (size_t r = 0; r < N; ++r)
    for (size_t f = 0; f < F; ++f) {{
      {ctype} v = {val};
      soa[f * N + r] = v;   /* SoA address */
      aos[r * F + f] = v;   /* AoS address -- same value v */
    }}
  for (size_t f = 0; f < F; ++f) weight[f] = {wexpr};
  bcir_layout_soa(soa, weight, soa_out, N);
  bcir_layout_aos(aos, weight, aos_out, N);
  int rc = 0;
  for (size_t r = 0; r < N; ++r)
    if ({cmp}) {{
      printf("FAIL r=%zu soa={fmt} aos={fmt}\\n", r, soa_out[r], aos_out[r]);
      rc = 1; break;
    }}
  free(soa); free(aos); free(weight); free(soa_out); free(aos_out);
  if (rc == 0) puts("OK layout invariance: SoA out == AoS out (bit-exact)");
  return rc;
}}
"""
    )


def reference_layout_out(fields: int, records: int, data_soa, data_aos, weight) -> tuple[list, list]:
    """The pure-Python oracle of the two layouts' output -- ``out[r] = sum_f data[r,f]*weight[f]`` evaluated
    by addressing ``data_soa`` as SoA (``f*N+r``) and ``data_aos`` as AoS (``r*F+f``). When both backing
    arrays hold the SAME logical ``data[r,f]``, the two outputs are IDENTICAL -- the invariance, proven in
    Python with no compiler (so the gate holds even when the toolchain is hidden)."""
    soa_out, aos_out = [], []
    for r in range(records):
        s_soa = s_aos = 0
        for f in range(fields):
            s_soa += data_soa[f * records + r] * weight[f]
            s_aos += data_aos[r * fields + f] * weight[f]
        soa_out.append(s_soa)
        aos_out.append(s_aos)
    return soa_out, aos_out


def compile_and_run_layout_c(fields: int, records: int, elem: str = "f32",
                             std: str = "c23", workdir: str | None = None) -> tuple[bool, str]:
    """Emit the layout invariance self-check, compile under ``-std=<std>``, run, and confirm it printed OK
    (SoA out == AoS out, bit-exact). Returns (ok, combined_output). Self-skips cleanly (False, message) when
    no C compiler is visible, exactly like ``c_kernel.compile_and_run_c``."""
    from shutil import which
    cc = which("clang") or which("cc") or which("gcc")
    if cc is None:
        return False, "no C compiler on PATH"
    created = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="bcir-layout-")
    try:
        src = os.path.join(workdir, "layout_check.c")
        exe = os.path.join(workdir, "lprog")
        with open(src, "w") as f:
            f.write(emit_layout_selfcheck_c(fields, records, elem))
        build = subprocess.run([cc, f"-std={std}", "-O2", "-Wall", "-Wextra", src, "-o", exe],
                               capture_output=True, text=True)
        if build.returncode != 0:
            return False, "layout build failed:\n" + build.stdout + build.stderr
        run = subprocess.run([exe], capture_output=True, text=True)
        ok = run.returncode == 0 and "OK" in run.stdout
        return ok, run.stdout + run.stderr
    finally:
        if created:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
