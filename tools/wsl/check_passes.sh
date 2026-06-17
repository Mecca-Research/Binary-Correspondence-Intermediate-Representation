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
FC="$(command -v FileCheck-19 || command -v FileCheck-18 || command -v FileCheck-17 || command -v FileCheck || true)"
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
