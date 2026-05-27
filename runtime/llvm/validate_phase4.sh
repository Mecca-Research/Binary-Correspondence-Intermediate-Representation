#!/usr/bin/env bash
set -euo pipefail

command -v llvm-as >/dev/null || { echo "missing llvm-as"; exit 127; }
command -v llvm-link >/dev/null || { echo "missing llvm-link"; exit 127; }
command -v opt >/dev/null || { echo "missing opt"; exit 127; }

mkdir -p build

tools/bcir-as/bcir-as examples/vector_add.bcir -o build/vector_add.generated.ll

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
  runtime/llvm/bcir_blob_schema.ll \
  runtime/llvm/bcir_blob_verify.ll \
  runtime/llvm/bcir_telemetry.ll \
  runtime/llvm/bcir_rehydrate.ll \
  build/vector_add.generated.ll \
  -S -o build/bcir_phase4_vector_add.ll

llvm-as build/bcir_phase4_vector_add.ll -o build/bcir_phase4_vector_add.bc
opt -passes=verify build/bcir_phase4_vector_add.bc -o /dev/null

grep -q "@bcir.generated.claims" build/bcir_phase4_vector_add.ll
grep -q "%bcir.blob.header" build/bcir_phase4_vector_add.ll
grep -q "@bcir.rehydrate.decide" build/bcir_phase4_vector_add.ll
grep -q "%bcir.telemetry" build/bcir_phase4_vector_add.ll
