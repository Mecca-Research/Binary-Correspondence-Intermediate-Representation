#!/usr/bin/env bash
# Validate the bcir-opt passes: -bcir-verify (laws R1-R12),
# -bcir-promote-lanes (GGG->UX opt-law), -convert-bcir-to-llvm (LLVM lowering).
# Run tools/wsl/build_mlir.sh first (or set BCIR_OPT).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BO="${BCIR_OPT:-$(find "${ROOT}/build/mlir-build" -name bcir-opt -type f 2>/dev/null | head -1)}"
if [ -z "${BO}" ]; then
  echo "bcir-opt not built; run tools/wsl/build_mlir.sh first (skipping)." >&2
  exit 0
fi
# Resolve FileCheck version-agnostically: the highest /usr/lib/llvm-*/bin/FileCheck (finds the
# installed major -- 22 now), then fall back to versioned / unversioned names on PATH.
FC="$(ls /usr/lib/llvm-*/bin/FileCheck 2>/dev/null | sort -V | tail -1)"
[ -n "${FC}" ] || FC="$(command -v FileCheck-22 || command -v FileCheck-19 || command -v FileCheck-18 || command -v FileCheck || true)"
T="${ROOT}/mlir/test/passes"
fail=0

echo "[passes] -bcir-verify negative cases (-verify-diagnostics)"
"${BO}" -bcir-verify -verify-diagnostics -split-input-file "${T}/verify_laws.mlir" \
  && echo "  PASS verify_laws (R1-R7)" || { echo "  FAIL verify_laws"; fail=1; }
"${BO}" -bcir-verify -verify-diagnostics -split-input-file "${T}/verify_laws_deep.mlir" \
  && echo "  PASS verify_laws_deep (R8-R16)" || { echo "  FAIL verify_laws_deep"; fail=1; }
"${BO}" -bcir-verify -verify-diagnostics -split-input-file "${T}/verify_accuracy.mlir" \
  && echo "  PASS verify_accuracy (R17 accuracy contract)" || { echo "  FAIL verify_accuracy"; fail=1; }
"${BO}" -bcir-verify -verify-diagnostics -split-input-file "${T}/verify_provenance.mlir" \
  && echo "  PASS verify_provenance (R13 digest recompute + m_theta cross-check)" || { echo "  FAIL verify_provenance"; fail=1; }
"${BO}" -bcir-verify -verify-diagnostics -split-input-file "${T}/verify_callgraph.mlir" \
  && echo "  PASS verify_callgraph (R18 callee resolution + recursion)" || { echo "  FAIL verify_callgraph"; fail=1; }

echo "[passes] parse-time op verifiers (hasVerifier structural well-formedness)"
"${BO}" -verify-diagnostics -split-input-file "${T}/verify_ops.mlir" \
  && echo "  PASS verify_ops (resource align/shape)" || { echo "  FAIL verify_ops"; fail=1; }

echo "[passes] -bcir-verify on the pretty corpus (must be clean)"
for f in "${ROOT}"/mlir/examples/*.mlir; do
  "${BO}" -bcir-verify "${f}" >/dev/null 2>/tmp/pe \
    && echo "  PASS verify $(basename "${f}")" || { echo "  FAIL verify $(basename "${f}")"; cat /tmp/pe; fail=1; }
done

run_fc() { # pass-flag, test-file
  if [ -n "${FC}" ]; then
    "${BO}" "$1" "$2" 2>/tmp/pe | "${FC}" "$2" \
      && echo "  PASS $(basename "$2")" || { echo "  FAIL $(basename "$2")"; cat /tmp/pe; fail=1; }
  else
    "${BO}" "$1" "$2" >/dev/null 2>/tmp/pe \
      && echo "  RUN-ONLY $(basename "$2")" || { echo "  FAIL $(basename "$2")"; cat /tmp/pe; fail=1; }
  fi
}
echo "[passes] -bcir-promote-lanes"
run_fc -bcir-promote-lanes "${T}/promote_lanes.mlir"
echo "[passes] -convert-bcir-to-llvm"
run_fc -convert-bcir-to-llvm "${T}/convert_to_llvm.mlir"

echo "[passes] async tokens (parse/roundtrip)"
if [ -n "${FC}" ]; then
  "${BO}" "${T}/async_tokens.mlir" 2>/tmp/pe | "${FC}" "${T}/async_tokens.mlir" \
    && echo "  PASS async_tokens.mlir" || { echo "  FAIL async_tokens.mlir"; cat /tmp/pe; fail=1; }
else
  "${BO}" "${T}/async_tokens.mlir" >/dev/null 2>/tmp/pe && echo "  RUN-ONLY async_tokens.mlir" || { echo "  FAIL"; cat /tmp/pe; fail=1; }
fi
echo "[passes] memory ordering (barrier -> llvm.fence)"
run_fc -convert-bcir-to-llvm "${T}/memory_ordering.mlir"

echo "[passes] GEM pipeline (classify/select/batch/schedule/lower)"
GEM="-bcir-classify-lanes -bcir-select-realization -bcir-batch -bcir-schedule -bcir-lower-to-llvm"
if [ -n "${FC}" ]; then
  "${BO}" ${GEM} "${T}/gem_passes.mlir" 2>/tmp/pe | "${FC}" "${T}/gem_passes.mlir" \
    && echo "  PASS gem_passes.mlir" || { echo "  FAIL gem_passes.mlir"; cat /tmp/pe; fail=1; }
else
  "${BO}" ${GEM} "${T}/gem_passes.mlir" >/dev/null 2>/tmp/pe \
    && echo "  RUN-ONLY gem_passes.mlir" || { echo "  FAIL gem_passes.mlir"; cat /tmp/pe; fail=1; }
fi
echo "[passes] GEM pipeline on the widened corpus (matmul/scan/histogram, generated)"
if [ -n "${FC}" ]; then
  "${BO}" ${GEM} "${T}/gem_corpus.mlir" 2>/tmp/pe | "${FC}" "${T}/gem_corpus.mlir" \
    && echo "  PASS gem_corpus.mlir" || { echo "  FAIL gem_corpus.mlir"; cat /tmp/pe; fail=1; }
else
  "${BO}" ${GEM} "${T}/gem_corpus.mlir" >/dev/null 2>/tmp/pe \
    && echo "  RUN-ONLY gem_corpus.mlir" || { echo "  FAIL gem_corpus.mlir"; cat /tmp/pe; fail=1; }
fi
echo "[passes] gem-matmul-cost (B1: recompute the matmul roofline (cost_of) parity-check + annotate; informs-only, never gates legality)"
run_fc -bcir-gem-matmul-cost "${T}/gem_matmul_cost.mlir"
echo "[passes] gem-matmul-cost analytic-parity negatives (declared compute/mem != the recomputed roofline)"
"${BO}" -bcir-gem-matmul-cost -verify-diagnostics -split-input-file "${T}/gem_matmul_cost_neg.mlir" \
  && echo "  PASS gem_matmul_cost_neg.mlir" || { echo "  FAIL gem_matmul_cost_neg.mlir"; fail=1; }
echo "[passes] gem-activation-cost (G1: recompute the activation roofline (cost_of) parity-check + annotate; informs-only, never gates legality)"
run_fc -bcir-gem-activation-cost "${T}/gem_activation_cost.mlir"
echo "[passes] gem-activation-cost analytic-parity negatives (declared compute/mem != the recomputed roofline)"
"${BO}" -bcir-gem-activation-cost -verify-diagnostics -split-input-file "${T}/gem_activation_cost_neg.mlir" \
  && echo "  PASS gem_activation_cost_neg.mlir" || { echo "  FAIL gem_activation_cost_neg.mlir"; fail=1; }
echo "[passes] gem-conv-cost (G7: recompute the conv roofline (cost_of, a structured matmul priced through matmul.cost_of) parity-check + annotate; informs-only, never gates legality)"
run_fc -bcir-gem-conv-cost "${T}/gem_conv_cost.mlir"
echo "[passes] gem-conv-cost analytic-parity negatives (declared compute/mem != the recomputed im2col-gemm roofline)"
"${BO}" -bcir-gem-conv-cost -verify-diagnostics -split-input-file "${T}/gem_conv_cost_neg.mlir" \
  && echo "  PASS gem_conv_cost_neg.mlir" || { echo "  FAIL gem_conv_cost_neg.mlir"; fail=1; }
echo "[passes] gem-attention-cost (G7: recompute the attention roofline (cost_of, two summed matmuls priced through matmul.cost_of) parity-check + annotate; informs-only, never gates legality)"
run_fc -bcir-gem-attention-cost "${T}/gem_attention_cost.mlir"
echo "[passes] gem-attention-cost analytic-parity negatives (declared per-gemm/summed compute/mem != the recomputed two-matmul roofline)"
"${BO}" -bcir-gem-attention-cost -verify-diagnostics -split-input-file "${T}/gem_attention_cost_neg.mlir" \
  && echo "  PASS gem_attention_cost_neg.mlir" || { echo "  FAIL gem_attention_cost_neg.mlir"; fail=1; }
echo "[passes] lower-gem-matmul (gem.matmul plan -> concrete tiled gem.block sequence)"
run_fc -bcir-lower-gem-matmul "${T}/lower_gem_matmul.mlir"
echo "[passes] lower-gem-matmul-buffer (gem.matmul_buffer -> tiled scf.for nest, C += A*B)"
if [ -n "${FC}" ]; then
  "${BO}" -bcir-lower-gem-matmul-buffer -split-input-file "${T}/lower_gem_matmul_buffer.mlir" 2>/tmp/pe | "${FC}" "${T}/lower_gem_matmul_buffer.mlir" \
    && echo "  PASS lower_gem_matmul_buffer.mlir" || { echo "  FAIL lower_gem_matmul_buffer.mlir"; cat /tmp/pe; fail=1; }
else
  "${BO}" -bcir-lower-gem-matmul-buffer -split-input-file "${T}/lower_gem_matmul_buffer.mlir" >/dev/null 2>/tmp/pe \
    && echo "  RUN-ONLY lower_gem_matmul_buffer.mlir" || { echo "  FAIL lower_gem_matmul_buffer.mlir"; cat /tmp/pe; fail=1; }
fi
echo "[passes] lower-gem-matmul-buffer op verifier negatives (-verify-diagnostics)"
"${BO}" -verify-diagnostics -split-input-file "${T}/lower_gem_matmul_buffer_neg.mlir" \
  && echo "  PASS lower_gem_matmul_buffer_neg.mlir" || { echo "  FAIL lower_gem_matmul_buffer_neg.mlir"; fail=1; }
echo "[passes] lower-gem-activation (gem.activation plan -> lane-width gem.block stripes; G1 dual-rail parity)"
run_fc -bcir-lower-gem-activation "${T}/lower_gem_activation.mlir"
echo "[passes] lower-gem-activation op verifier negatives (the quarantine rule + shape/dtype/axis laws)"
"${BO}" -verify-diagnostics -split-input-file "${T}/lower_gem_activation_neg.mlir" \
  && echo "  PASS lower_gem_activation_neg.mlir" || { echo "  FAIL lower_gem_activation_neg.mlir"; fail=1; }
echo "[passes] lower-gem-conv (gem.conv plan -> im2col-gemm tiled gem.block sequence; G7 dual-rail parity)"
run_fc -bcir-lower-gem-conv "${T}/lower_gem_conv.mlir"
echo "[passes] lower-gem-conv op verifier negatives (derived out dims / im2col gemm dims / strategy/tile / bottleneck / R17)"
"${BO}" -verify-diagnostics -split-input-file "${T}/lower_gem_conv_neg.mlir" \
  && echo "  PASS lower_gem_conv_neg.mlir" || { echo "  FAIL lower_gem_conv_neg.mlir"; fail=1; }
echo "[passes] lower-gem-attention (gem.attention plan -> two gem.matmul tile seqs + softmax; G7 dual-rail parity)"
run_fc -bcir-lower-gem-attention "${T}/lower_gem_attention.mlir"
echo "[passes] lower-gem-attention op verifier negatives (quarantine dtype / scores+context gemm dims / tile / summed cost / bottleneck / R17)"
"${BO}" -verify-diagnostics -split-input-file "${T}/lower_gem_attention_neg.mlir" \
  && echo "  PASS lower_gem_attention_neg.mlir" || { echo "  FAIL lower_gem_attention_neg.mlir"; fail=1; }
echo "[passes] fuse-matmul-activation (G2: sole-consumer gem.matmul -> gem.activation -> fused epilogue; deforestation-priced)"
run_fc -bcir-fuse-matmul-activation "${T}/fuse_matmul_activation.mlir"
echo "[passes] fuse-matmul-activation op verifier negatives (softmax scope-out + the quarantine rule + the strict-win invariant)"
"${BO}" -verify-diagnostics -split-input-file "${T}/fuse_matmul_activation_neg.mlir" \
  && echo "  PASS fuse_matmul_activation_neg.mlir" || { echo "  FAIL fuse_matmul_activation_neg.mlir"; fail=1; }
echo "[passes] gem.autodiff round-trip (B3: the closed-set forward autodiff DAG serializes/parses/prints identically)"
if [ -n "${FC}" ]; then
  "${BO}" "${T}/gem_autodiff_roundtrip.mlir" 2>/tmp/pe | "${BO}" | "${FC}" "${T}/gem_autodiff_roundtrip.mlir" \
    && echo "  PASS gem_autodiff_roundtrip.mlir" || { echo "  FAIL gem_autodiff_roundtrip.mlir"; cat /tmp/pe; fail=1; }
else
  "${BO}" "${T}/gem_autodiff_roundtrip.mlir" >/dev/null 2>/tmp/pe \
    && echo "  RUN-ONLY gem_autodiff_roundtrip.mlir" || { echo "  FAIL gem_autodiff_roundtrip.mlir"; cat /tmp/pe; fail=1; }
fi
echo "[passes] gem.autodiff op verifier negatives (the closed-set law: foreign opcode / wrong arity / forward index / var slot / output / payload)"
"${BO}" -verify-diagnostics -split-input-file "${T}/gem_autodiff_verify_neg.mlir" \
  && echo "  PASS gem_autodiff_verify_neg.mlir" || { echo "  FAIL gem_autodiff_verify_neg.mlir"; fail=1; }
echo "[passes] bcir.asm round-trip (ASM1: verbatim inline asm parses/prints identically -- 0-output fence + 1-out/1-in form)"
if [ -n "${FC}" ]; then
  "${BO}" "${T}/inline_asm_roundtrip.mlir" 2>/tmp/pe | "${BO}" | "${FC}" "${T}/inline_asm_roundtrip.mlir" \
    && echo "  PASS inline_asm_roundtrip.mlir" || { echo "  FAIL inline_asm_roundtrip.mlir"; cat /tmp/pe; fail=1; }
else
  "${BO}" "${T}/inline_asm_roundtrip.mlir" >/dev/null 2>/tmp/pe \
    && echo "  RUN-ONLY inline_asm_roundtrip.mlir" || { echo "  FAIL inline_asm_roundtrip.mlir"; cat /tmp/pe; fail=1; }
fi
echo "[passes] bcir.asm -> llvm.inline_asm (ASM1 lowering: out/in/clobber constraint string, ~{...} clobbers, side-effecting)"
run_fc -convert-bcir-to-llvm "${T}/inline_asm.mlir"
echo "[passes] bcir.asm op verifier negatives (arg/constraint count + result/out count)"
"${BO}" -verify-diagnostics -split-input-file "${T}/inline_asm_verify_neg.mlir" \
  && echo "  PASS inline_asm_verify_neg.mlir" || { echo "  FAIL inline_asm_verify_neg.mlir"; fail=1; }
echo "[passes] bcir.asm -> llvm LOWERING negatives (the '+' read-write output reject + multi-output + exotic-GCC-syntax rejects)"
"${BO}" -convert-bcir-to-llvm -verify-diagnostics -split-input-file "${T}/inline_asm_lower_neg.mlir" \
  && echo "  PASS inline_asm_lower_neg.mlir" || { echo "  FAIL inline_asm_lower_neg.mlir"; fail=1; }
echo "[passes] bcir.portio round-trip (ASM2: x86 port-I/O edge parses/prints identically -- in.{b,l} read + out.{b,l} void write)"
if [ -n "${FC}" ]; then
  "${BO}" "${T}/portio_roundtrip.mlir" 2>/tmp/pe | "${BO}" | "${FC}" "${T}/portio_roundtrip.mlir" \
    && echo "  PASS portio_roundtrip.mlir" || { echo "  FAIL portio_roundtrip.mlir"; cat /tmp/pe; fail=1; }
else
  "${BO}" "${T}/portio_roundtrip.mlir" >/dev/null 2>/tmp/pe \
    && echo "  RUN-ONLY portio_roundtrip.mlir" || { echo "  FAIL portio_roundtrip.mlir"; cat /tmp/pe; fail=1; }
fi
echo "[passes] bcir.portio -> llvm.inline_asm (ASM2 lowering: x86 in/out \${N:mod} template + ={ax},N{dx} / {ax},N{dx} constraints, side-effecting; LLVM-IR-correct, assembles via llc)"
run_fc -convert-bcir-to-llvm "${T}/portio.mlir"
echo "[passes] bcir.portio op verifier negatives (width {8,16,32} + in/out operand/result arity + value width)"
"${BO}" -verify-diagnostics -split-input-file "${T}/portio_verify_neg.mlir" \
  && echo "  PASS portio_verify_neg.mlir" || { echo "  FAIL portio_verify_neg.mlir"; fail=1; }
echo "[passes] bcir.volatile_load/store round-trip (first-class MMIO: ordered volatile device-register access parses/prints identically)"
if [ -n "${FC}" ]; then
  "${BO}" "${T}/volatile_mmio_roundtrip.mlir" 2>/tmp/pe | "${BO}" | "${FC}" "${T}/volatile_mmio_roundtrip.mlir" \
    && echo "  PASS volatile_mmio_roundtrip.mlir" || { echo "  FAIL volatile_mmio_roundtrip.mlir"; cat /tmp/pe; fail=1; }
else
  "${BO}" "${T}/volatile_mmio_roundtrip.mlir" >/dev/null 2>/tmp/pe \
    && echo "  RUN-ONLY volatile_mmio_roundtrip.mlir" || { echo "  FAIL volatile_mmio_roundtrip.mlir"; cat /tmp/pe; fail=1; }
fi
echo "[passes] bcir.volatile_load/store -> llvm.inttoptr + volatile llvm.load/store (MMIO lowering; mirrors the cfront *(volatile T*)(intaddr) emit)"
run_fc -convert-bcir-to-llvm "${T}/volatile_mmio.mlir"
echo "[passes] bcir.volatile_load/store op negatives (the device-register address must be a signless integer)"
"${BO}" -verify-diagnostics -split-input-file "${T}/volatile_mmio_verify_neg.mlir" \
  && echo "  PASS volatile_mmio_verify_neg.mlir" || { echo "  FAIL volatile_mmio_verify_neg.mlir"; fail=1; }
echo "[passes] bcir.atomic_rmw/atomic_cas round-trip (§5.14 Phase 2: first-class atomics -- kind, #bcir.mem_ordering, weak all survive print->parse)"
if [ -n "${FC}" ]; then
  "${BO}" "${T}/atomic_ops_roundtrip.mlir" 2>/tmp/pe | "${BO}" | "${FC}" "${T}/atomic_ops_roundtrip.mlir" \
    && echo "  PASS atomic_ops_roundtrip.mlir" || { echo "  FAIL atomic_ops_roundtrip.mlir"; cat /tmp/pe; fail=1; }
else
  "${BO}" "${T}/atomic_ops_roundtrip.mlir" >/dev/null 2>/tmp/pe \
    && echo "  RUN-ONLY atomic_ops_roundtrip.mlir" || { echo "  FAIL atomic_ops_roundtrip.mlir"; cat /tmp/pe; fail=1; }
fi
echo "[passes] bcir.atomic_rmw/atomic_cas -> llvm.atomicrmw/cmpxchg (§5.14 Phase 2 lowering: inttoptr + mapped ordering, seq_cst default, derived CAS failure ordering, weak flag)"
run_fc -convert-bcir-to-llvm "${T}/atomic_ops.mlir"
echo "[passes] bcir.atomic_rmw/atomic_cas op negatives (kind set, integer discipline, old-value result typing, >=32-bit address)"
"${BO}" -verify-diagnostics -split-input-file "${T}/atomic_ops_verify_neg.mlir" \
  && echo "  PASS atomic_ops_verify_neg.mlir" || { echo "  FAIL atomic_ops_verify_neg.mlir"; fail=1; }
echo "[passes] R5 volatile-claim law (§5.14 Phase 2: is_volatile on the claim rail requires an ordered hazard; positive + negative)"
"${BO}" -bcir-verify -verify-diagnostics -split-input-file "${T}/verify_volatile.mlir" \
  && echo "  PASS verify_volatile.mlir" || { echo "  FAIL verify_volatile.mlir"; fail=1; }
echo "[passes] R12 call-ABI contract (SS5.14 Phase 2 last area: named target matrix + truthful pointer/long sizes; negatives + a legal record)"
"${BO}" -bcir-verify -verify-diagnostics -split-input-file "${T}/verify_abi_contract.mlir" \
  && echo "  PASS verify_abi_contract.mlir" || { echo "  FAIL verify_abi_contract.mlir"; fail=1; }
echo "[passes] R18 indirect-callee signature (SS5.14 Phase 2: a carried callee_sig must be well-formed; vacuous when absent)"
"${BO}" -bcir-verify -verify-diagnostics -split-input-file "${T}/verify_callee_sig.mlir" \
  && echo "  PASS verify_callee_sig.mlir" || { echo "  FAIL verify_callee_sig.mlir"; fail=1; }
echo "[passes] R7 masked clause + extent-provenance (SS5.14 Phase 2: masked needs the bounds verify contract, provenance surfaced; dual-railed from the oracle)"
"${BO}" -bcir-verify -verify-diagnostics -split-input-file "${T}/verify_bounds_provenance.mlir" \
  && echo "  PASS verify_bounds_provenance.mlir" || { echo "  FAIL verify_bounds_provenance.mlir"; fail=1; }
echo "[passes] R22/R23 gem shape/dtype seam laws (D2 promotion: matmul->activation extent + conv/attention->activation dtype; negatives + a legal seam)"
"${BO}" -bcir-verify -verify-diagnostics -split-input-file "${T}/verify_shape_dtype.mlir" \
  && echo "  PASS verify_shape_dtype.mlir" || { echo "  FAIL verify_shape_dtype.mlir"; fail=1; }
echo "[passes] bcir.creg_read/write round-trip (D1.3: x86-64 control-register access parses/prints identically)"
if [ -n "${FC}" ]; then
  "${BO}" "${T}/creg_roundtrip.mlir" 2>/tmp/pe | "${BO}" | "${FC}" "${T}/creg_roundtrip.mlir" \
    && echo "  PASS creg_roundtrip.mlir" || { echo "  FAIL creg_roundtrip.mlir"; cat /tmp/pe; fail=1; }
else
  "${BO}" "${T}/creg_roundtrip.mlir" >/dev/null 2>/tmp/pe \
    && echo "  RUN-ONLY creg_roundtrip.mlir" || { echo "  FAIL creg_roundtrip.mlir"; cat /tmp/pe; fail=1; }
fi
echo "[passes] bcir.creg_read/write -> llvm.inline_asm (D1.3 lowering: mov %crN,\$0 / mov \$0,%crN; LLVM-IR asm syntax, side-effecting)"
run_fc -convert-bcir-to-llvm "${T}/creg.mlir"
echo "[passes] bcir.creg_read/write op negatives (the control-register value must be an i64)"
"${BO}" -verify-diagnostics -split-input-file "${T}/creg_verify_neg.mlir" \
  && echo "  PASS creg_verify_neg.mlir" || { echo "  FAIL creg_verify_neg.mlir"; fail=1; }
echo "[passes] bcir.msr_read/write round-trip (D1.4: x86-64 model-specific-register access parses/prints identically)"
if [ -n "${FC}" ]; then
  "${BO}" "${T}/msr_roundtrip.mlir" 2>/tmp/pe | "${BO}" | "${FC}" "${T}/msr_roundtrip.mlir" \
    && echo "  PASS msr_roundtrip.mlir" || { echo "  FAIL msr_roundtrip.mlir"; cat /tmp/pe; fail=1; }
else
  "${BO}" "${T}/msr_roundtrip.mlir" >/dev/null 2>/tmp/pe \
    && echo "  RUN-ONLY msr_roundtrip.mlir" || { echo "  FAIL msr_roundtrip.mlir"; cat /tmp/pe; fail=1; }
fi
echo "[passes] bcir.msr_read/write -> llvm.inline_asm (D1.4 lowering: rdmsr/wrmsr; index->ECX, i64 reassembled/split across EDX:EAX, side-effecting + ~{memory})"
run_fc -convert-bcir-to-llvm "${T}/msr.mlir"
echo "[passes] ASSEMBLE-SMOKE-TEST (anti-masking: portio/asm/creg/volatile/msr lowerings piped through mlir-translate-20 | llc-20 -- a real .o + the expected instruction, not just FileCheck text)"
# Source the harness so it shares ${BO}; it returns nonzero on any assemble failure (and skips cleanly
# when llc-20/mlir-translate-20 are absent, like the FileCheck guard). Fold its result into ${fail}.
if . "${ROOT}/tools/wsl/check_asm_lowering.sh"; then :; else fail=1; fi
echo "[passes] cache-contention (G4: recompute the cache-line/bank-conflict CONTENTION signal; informs-only, never gates legality)"
run_fc -bcir-cache-contention "${T}/cache_contention.mlir"
echo "[passes] cache-contention op verifier negatives (the frozen analytic model: waste/conflict/q8-sum/cost consistency)"
"${BO}" -verify-diagnostics -split-input-file "${T}/cache_contention_neg.mlir" \
  && echo "  PASS cache_contention_neg.mlir" || { echo "  FAIL cache_contention_neg.mlir"; fail=1; }
echo "[passes] layout-pivot (G3: recompute the SoA<->AoS pricing through the stride-penalty terms; informs-only, the priced choice)"
run_fc -bcir-layout-pivot "${T}/layout_pivot.mlir"
echo "[passes] layout-pivot op verifier negatives (priced SoA/AoS cost / the min-cost layout choice / the gain)"
"${BO}" -verify-diagnostics -split-input-file "${T}/layout_pivot_neg.mlir" \
  && echo "  PASS layout_pivot_neg.mlir" || { echo "  FAIL layout_pivot_neg.mlir"; fail=1; }
echo "[passes] cost model (the K_BCIR cost algebra recomputed from claim + capability)"
run_fc -bcir-cost-model "${T}/cost_model.mlir"
echo "[passes] cost-model fusion (intra-phase deforestation + CSE)"
run_fc -bcir-cost-model "${T}/cost_model_fusion.mlir"
echo "[passes] cost-model verify dimension (exact/hash discharge cost)"
run_fc -bcir-cost-model "${T}/cost_model_verify.mlir"
echo "[passes] -bcir-cost-model cross-check on the pretty corpus (reproduces 7808)"
"${BO}" -bcir-cost-model "${ROOT}/mlir/examples/full_vec_add_ct1.mlir" >/dev/null 2>/tmp/pe \
  && echo "  PASS cost-model on full_vec_add_ct1.mlir" || { echo "  FAIL cost-model on full_vec_add"; cat /tmp/pe; fail=1; }
echo "[passes] plan: the layered min-plus shortest path (the full realize.optimize in C++)"
run_fc -bcir-plan "${T}/plan.mlir"
echo "[passes] -bcir-plan reproduces the oracle's coupled scores on the widened corpus"
plan_out="$("${BO}" -bcir-plan "${T}/gem_corpus.mlir" 2>/tmp/pe)"
if grep -q "kbcir.plan_score = 1015808" <<<"${plan_out}" \
   && grep -q "kbcir.plan_score = 101888" <<<"${plan_out}" \
   && grep -q "kbcir.plan_score = 1595520" <<<"${plan_out}"; then
  echo "  PASS plan on gem_corpus (matmul 1015808 / scan 101888 / histogram 1595520)"
else
  echo "  FAIL plan on gem_corpus"; cat /tmp/pe; fail=1
fi
echo "[passes] overlap: the (max,+) scheduled price M(pi,Theta) (gem/overlap.py in C++)"
run_fc -bcir-overlap "${T}/overlap.mlir"
echo "[passes] overlap-optimize: the makespan re-selection sweep (optimize_scheduled)"
run_fc -bcir-overlap-optimize "${T}/overlap_optimize.mlir"
echo "[passes] sense: the regret-driven telemetry resolution gate (kbcir/sensing.py)"
run_fc -bcir-sense "${T}/sense.mlir"
oo_out="$("${BO}" -bcir-overlap-optimize "${T}/gem_corpus.mlir" 2>/tmp/pe)"
if grep -q "kbcir.overlap_opt_makespan = 253952" <<<"${oo_out}"; then
  echo "  PASS overlap-optimize on gem_corpus (sweep stable: matmul makespan 253952)"
else
  echo "  FAIL overlap-optimize on gem_corpus"; cat /tmp/pe; fail=1
fi
echo "[passes] -bcir-overlap reproduces the oracle's makespan on the widened corpus"
ov_out="$("${BO}" -bcir-overlap "${T}/gem_corpus.mlir" 2>/tmp/pe)"
if grep -q "kbcir.overlap_gain = 761856" <<<"${ov_out}" \
   && grep -q "kbcir.overlap_makespan = 253952" <<<"${ov_out}"; then
  echo "  PASS overlap on gem_corpus (matmul makespan 253952, gain 761856)"
else
  echo "  FAIL overlap on gem_corpus"; cat /tmp/pe; fail=1
fi
echo "[passes] RCSP / Pareto (the deterministic optimizer core ported to C++)"
run_fc -bcir-rcsp "${T}/rcsp.mlir"
echo "[passes] RCSP plan-level (accumulated-budget label-DP across the plan)"
run_fc -bcir-rcsp-plan "${T}/rcsp_plan.mlir"
echo "[passes] bundle detection (multi-claim joint bundles, the kbcir.bundle analysis)"
run_fc -bcir-bundle "${T}/bundle.mlir"
echo "[passes] bundle joint-reorder (reorder the cost columns + re-price -> kbcir.bundle_gain)"
run_fc -bcir-bundle "${T}/bundle_reorder.mlir"
echo "[passes] explain (proof-carrying decision record as IR annotations, the proof.explain port)"
run_fc -bcir-explain "${T}/explain.mlir"
echo "[passes] replay (recheck the declared explain_* record vs a fresh plan, the proof.replay port)"
run_fc -bcir-replay "${T}/replay.mlir"
echo "[passes] compose func/if op family (round-trip: kbcir.func / kbcir.call / kbcir.cond)"
if [ -n "${FC}" ]; then
  "${BO}" "${T}/compose_ops.mlir" 2>/tmp/pe | "${FC}" "${T}/compose_ops.mlir" \
    && echo "  PASS compose_ops.mlir" || { echo "  FAIL compose_ops.mlir"; cat /tmp/pe; fail=1; }
else
  "${BO}" "${T}/compose_ops.mlir" >/dev/null 2>/tmp/pe \
    && echo "  RUN-ONLY compose_ops.mlir" || { echo "  FAIL compose_ops.mlir"; cat /tmp/pe; fail=1; }
fi
echo "[passes] compose cost (compositional plan over func/if: Seq sum / Cond max+expected / Call)"
run_fc -bcir-compose "${T}/compose_cost.mlir"
echo "[passes] compose summary (inter-procedural: plan once, reuse compatible calls, re-price HBM)"
run_fc -bcir-compose "${T}/compose_summary.mlir"
echo "[passes] compose budget (RCSP-constrained: thermal cap re-prices vec16->vec8 / infeasible)"
run_fc -bcir-compose "${T}/compose_budget.mlir"
echo "[passes] compose effect/independence + dynamic shapes (footprint / commutes / bound)"
run_fc -bcir-compose "${T}/compose_effect.mlir"
echo "[passes] cim (recompute CIM/PIM dispatch decision: gem.cim.cim_decision)"
run_fc -bcir-cim "${T}/cim.mlir"
echo "[passes] dvfs (recompute phase-aware clock: gem.dvfs classify + clock_for)"
run_fc -bcir-dvfs "${T}/dvfs.mlir"
echo "[passes] schedule-eft (duration-aware EFT waves: gem.schedule.schedule_eft)"
run_fc -bcir-schedule-eft "${T}/schedule_eft.mlir"
echo "[passes] async (fork/await plan + pipelined cross-phase schedule: execute_tokens)"
run_fc -bcir-async "${T}/async.mlir"
echo "[passes] power-rail (per-slot DVFS over the EFT timeline: schedule_power_rail)"
run_fc -bcir-power-rail "${T}/power_rail.mlir"
echo "[passes] alloc-pool (liveness-based memory pooling: allocator.pool_plan)"
run_fc -bcir-alloc-pool "${T}/alloc_pool.mlir"
echo "[passes] hot-Theta plan parity (the kbcir.theta context op)"
run_fc -bcir-plan "${T}/theta_hot.mlir"
echo "[passes] six-target capability matrix (-bcir-plan/-overlap/-rcsp-plan per target)"
MATRIX="-bcir-plan -bcir-overlap -bcir-rcsp-plan"
if [ -n "${FC}" ]; then
  "${BO}" ${MATRIX} "${T}/target_matrix.mlir" 2>/tmp/pe | "${FC}" "${T}/target_matrix.mlir" \
    && echo "  PASS target_matrix.mlir (FileCheck)" || { echo "  FAIL target_matrix.mlir"; cat /tmp/pe; fail=1; }
fi
# Robust per-target spot checks (run even without FileCheck), distinctive per-target
# scores the C++ plan must recompute from the capability seeds alone: the ARM-NEON
# (width 4) and PTX (width 32) vector_add plans, the GPU's cheaper coalesced gather
# (penalty 16 -> 266240 vs the CPUs' 528384), and a plan-level RCSP constrained optimum.
mx_out="$("${BO}" ${MATRIX} "${T}/target_matrix.mlir" 2>/tmp/pe)"
if grep -q "kbcir.plan_score = 12800" <<<"${mx_out}" \
   && grep -q "kbcir.plan_score = 6976" <<<"${mx_out}" \
   && grep -q "kbcir.plan_score = 266240" <<<"${mx_out}" \
   && grep -q "kbcir.rcsp_plan_score = 17280" <<<"${mx_out}"; then
  echo "  PASS target_matrix per-target spot checks (neon 12800 / ptx 6976 / ptx gather 266240 / rvv-class rcsp 17280)"
else
  echo "  FAIL target_matrix per-target spot checks"; cat /tmp/pe; fail=1
fi
echo "[passes] -bcir-rcsp cross-check on the widened corpus (argmin reproduces the oracle)"
"${BO}" -bcir-rcsp "${T}/gem_corpus.mlir" >/dev/null 2>/tmp/pe \
  && echo "  PASS rcsp on gem_corpus.mlir" || { echo "  FAIL rcsp on gem_corpus.mlir"; cat /tmp/pe; fail=1; }

echo "[passes] named pipelines (bcir-audit / -optimize / -hydrate / -lower-llvm / -aot)"
EX="${ROOT}/mlir/examples/full_vec_add_ct1.mlir"
audit_out="$("${BO}" -bcir-audit "${EX}" 2>/tmp/pe)"
if grep -q "kbcir.plan_score = 7808" <<<"${audit_out}" \
   && grep -q "kbcir.overlap_makespan = 7808" <<<"${audit_out}" \
   && grep -q "kbcir.cm_min_score = 7808" <<<"${audit_out}"; then
  echo "  PASS bcir-audit (verify + cost/plan/overlap = 7808)"
else echo "  FAIL bcir-audit"; cat /tmp/pe; fail=1; fi
aot_out="$("${BO}" -bcir-aot "${EX}" 2>/tmp/pe)"
if grep -q "llvm.fadd" <<<"${aot_out}" && grep -q "kbcir.lowered = true" <<<"${aot_out}"; then
  echo "  PASS bcir-aot (verify -> hydrate -> LLVM: llvm.fadd + lowered)"
else echo "  FAIL bcir-aot"; cat /tmp/pe; fail=1; fi
for pl in bcir-optimize bcir-hydrate bcir-lower-llvm; do
  "${BO}" -${pl} "${EX}" >/dev/null 2>/tmp/pe \
    && echo "  PASS ${pl}" || { echo "  FAIL ${pl}"; cat /tmp/pe; fail=1; }
done

echo "[passes] GEM cross-checks against the oracle (-verify-diagnostics)"
"${BO}" -bcir-select-realization -bcir-lower-to-llvm -verify-diagnostics -split-input-file \
  "${T}/gem_passes_neg.mlir" \
  && echo "  PASS gem_passes_neg.mlir" || { echo "  FAIL gem_passes_neg.mlir"; fail=1; }

[ "${fail}" -eq 0 ] && echo "[passes] all passes validate" || echo "[passes] FAILURES"
exit "${fail}"
