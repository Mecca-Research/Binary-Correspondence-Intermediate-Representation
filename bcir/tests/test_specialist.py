"""JIT specialist synthesis: bake + unroll a kernel for a hot, small shape; keep the
generic runtime-n kernel for cold or large shapes (gains-only). The specialist
computes exactly the generic result."""

import os
import subprocess
import tempfile
from shutil import which

from bcir.examples import vector_add
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta
from bcir.lower.specialist import (
    DEFAULT_HOT_THRESHOLD,
    ShapeLedger,
    emit_specialist_c,
    specializable,
    synthesize,
)

AVX = TARGETS["x86_avx512"]


def _plan(n=1024):
    m = vector_add(n)
    return m, optimize(m, AVX, Theta.cool())


def _cc():
    return which("clang") or which("cc") or which("gcc")


# --- the hot-shape detector ------------------------------------------------------

def test_ledger_counts_shapes_and_flags_hot_ones():
    led = ShapeLedger()
    for _ in range(DEFAULT_HOT_THRESHOLD):
        k = led.observe("vector.add", 1024)
    assert led.frequency(k) == DEFAULT_HOT_THRESHOLD and led.is_hot(k)
    assert not led.is_hot(led.observe("vector.add", 99))   # seen once


# --- gating: specialize only hot AND small ---------------------------------------

def test_cold_shape_keeps_the_generic_kernel():
    m, r = _plan(1024)
    led = ShapeLedger()
    res = synthesize(m, r, led)               # first sight: not hot
    assert not res.specialized and res.fn_name == "bcir_kernel"
    assert "n =" not in res.kernel_c or "baked" not in res.kernel_c


def test_hot_small_shape_synthesizes_a_baked_unrolled_specialist():
    m, r = _plan(1024)
    led = ShapeLedger()
    for _ in range(DEFAULT_HOT_THRESHOLD - 1):
        synthesize(m, r, led)                 # warm it up to the threshold
    res = synthesize(m, r, led)
    assert res.specialized and res.count == 1024
    assert "shape specialist: n = 1024 baked" in res.kernel_c
    assert "size_t n" not in res.kernel_c     # the count is baked, not a parameter
    assert "+ 7u]" in res.kernel_c or "+ 1u]" in res.kernel_c   # unrolled body


def test_large_shape_is_never_specialized():
    big = 1 << 20
    assert not specializable(big)
    m, r = _plan(big)
    led = ShapeLedger()
    for _ in range(DEFAULT_HOT_THRESHOLD + 2):
        res = synthesize(m, r, led)           # hot, but too large to bake
    assert not res.specialized and "large shape" in res.reason


def test_synthesis_is_deterministic():
    a = emit_specialist_c("vector.add", "+", 130, "f32")
    b = emit_specialist_c("vector.add", "+", 130, "f32")
    assert a == b and "C[128u] =" in a and "C[129u] =" in a   # baked remainder (130 % 8)


# --- correctness: the specialist computes the same result as the generic ----------

def test_specialist_compiles_and_is_correct():
    cc = _cc()
    if cc is None:
        return  # no compiler: skip cleanly
    n = 1024
    kernel = emit_specialist_c("vector.add", "+", n, "f32", "spec")
    main = kernel + f"""
#include <stdlib.h>
#include <stdio.h>
int main(void) {{
  float *A=malloc({n}*4),*B=malloc({n}*4),*C=malloc({n}*4);
  for (size_t i=0;i<{n}u;++i){{A[i]=(float)i;B[i]=2.0f*(float)i;C[i]=-1.0f;}}
  spec(A,B,C);
  for (size_t i=0;i<{n}u;++i) if (C[i]!=A[i]+B[i]) {{ printf("FAIL %zu\\n",i); return 1; }}
  puts("OK"); return 0;
}}
"""
    with tempfile.TemporaryDirectory() as d:
        src, exe = os.path.join(d, "s.c"), os.path.join(d, "s")
        open(src, "w").write(main)
        build = None
        for std in ("-std=c23", "-std=c2x"):
            build = subprocess.run([cc, std, "-O2", src, "-o", exe], capture_output=True, text=True)
            if build.returncode == 0:
                break
        assert build is not None and build.returncode == 0, build.stderr
        run = subprocess.run([exe], capture_output=True, text=True)
        assert run.returncode == 0 and "OK" in run.stdout, run.stdout + run.stderr
