#!/usr/bin/env bash
# layernorm: E3 Transformer layernorm (emit_layernorm_c): per-row normalize to mean~0/var~1 (#layernorm)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-layernorm` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. The binary is the
# manifest kernel kernel_layernorm: C the Python oracle emits (bcir.lower.c_kernel) with its driver
# appended by tools/build/emit_kernel.py, the writer the gate and the CMake build share.
#
#   usage: layernorm.sh <kernel_layernorm>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: layernorm.sh <kernel_layernorm>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
KERNEL="$(abs "$1")"; [ -x "${KERNEL}" ] || { echo "  FAIL: KERNEL: ${KERNEL} is not an executable kernel"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# E3 (ML-breadth) full Transformer block's ONE new numeric primitive (#layernorm): the LAYERNORM kernel
# (emit_layernorm_c). Unlike E1/E2 (each a wrap of one external LAPACK kernel), the Transformer block is a
# COMPOSITION -- its per-head matmuls + scale + softmax are the EXISTING emit_attention_c C twin, the
# projections/feed-forward are the EXISTING matmul emitters, so the ONLY net-new C kernel is the per-row
# layernorm: out = gamma*(x-mean)/sqrtf(var+eps) + beta (POPULATION /dim variance). Its only transcendental is
# the 1/sqrtf(var+eps), which rides the c.call.libm:sqrtf edge (-lm, ALREADY mapped -- no linkflags change,
# confirmed by #linkflags-* probes: sqrtf rides the libm rule). This probe compiles + runs the kernel (no
# external library needed -- only libm) on a known matrix with gamma=1/beta=0 and checks each output row is
# normalized to mean~0 / var~1 (the property layernorm guarantees), the C twin of kbcir.transformer.layernorm_
# reference + its layernorm_stats independent verifier.
ln_out="$("${KERNEL}")"; ln_rc=$?    # rc=0 IS the check: each output row has mean~0 and var~1 (in the driver)
{ [ "${ln_rc}" = "0" ] && printf '%s' "${ln_out}" | grep -q "^r0 mean=.* var=.*"; } \
  && echo "  PASS layernorm: rows normalize to mean~0/var~1 (${ln_out//$'\n'/ }; sqrtf -> -lm via the libm rule, no linkflags change)" \
  || { echo "  FAIL: layernorm did not normalize rows (rc=${ln_rc}: ${ln_out})"; exit 1; }
