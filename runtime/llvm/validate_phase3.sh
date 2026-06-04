#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
source runtime/llvm/validate_common.sh
require_llvm_tool llvm-as
require_llvm_tool llvm-link
require_llvm_tool opt
require_llvm_tool llvm-dis


mkdir -p build

llvm-link \
  runtime/llvm/bcir_claim_schema.ll \
  runtime/llvm/bcir_claim_accessors.ll \
  runtime/llvm/bcir_registry_schema.ll \
  runtime/llvm/bcir_ops.ll \
  runtime/llvm/bcir_phase_epoch.ll \
  runtime/llvm/bcir_claim_verify.ll \
  runtime/llvm/bcir_gem_seed.ll \
  runtime/llvm/bcir_worklist.ll \
  runtime/llvm/bcir_schedule_schema.ll \
  runtime/llvm/bcir_schedule_accessors.ll \
  runtime/llvm/bcir_prefetch_profiles.ll \
  runtime/llvm/bcir_lane_classifier.ll \
  runtime/llvm/bcir_batch_executor.ll \
  runtime/llvm/bcir_batch_verify.ll \
  runtime/llvm/bcir_stream_pack.ll \
  runtime/llvm/bcir_kbcost.ll \
  runtime/llvm/bcir_examples_phase3.ll \
  -S -o build/bcir_phase3_all.ll

llvm-as build/bcir_phase3_all.ll -o build/bcir_phase3_all.bc
opt -passes=verify build/bcir_phase3_all.bc -o /dev/null
llvm-dis build/bcir_phase3_all.bc -o build/bcir_phase3_all.dis.ll

grep -q "%bcir.batch" build/bcir_phase3_all.dis.ll
grep -q "%bcir.phase.range" build/bcir_phase3_all.dis.ll
grep -q "llvm.prefetch" build/bcir_phase3_all.dis.ll
grep -q "@bcir.gem.execute_batch" build/bcir_phase3_all.dis.ll
grep -q "@bcir.classify.memory_lane" build/bcir_phase3_all.dis.ll

grep -q "%bcir.stream.pack" build/bcir_phase3_all.dis.ll
grep -q "@bcir.gem.execute_stream_pack" build/bcir_phase3_all.dis.ll
grep -q "@bcir.kbdi.score.batch.q16" build/bcir_phase3_all.dis.ll
grep -q "!bcir.kbdi.phase3" build/bcir_phase3_all.dis.ll
