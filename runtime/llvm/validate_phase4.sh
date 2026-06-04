#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=runtime/llvm/validate_common.sh
source "$SCRIPT_DIR/validate_common.sh"

require_tools llvm-as llvm-link opt
require_executable tools/bcir-as/bcir-as
init_build_dir
log_toolchain llvm-as llvm-link opt
log "bcir-as: $REPO_ROOT/tools/bcir-as/bcir-as"

run_step "generate vector_add LLVM IR" tools/bcir-as/bcir-as examples/vector_add.bcir -o build/vector_add.generated.ll

run_step "link phase4 vector_add runtime modules" llvm-link \
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

run_step "assemble phase4 vector_add module" llvm-as build/bcir_phase4_vector_add.ll -o build/bcir_phase4_vector_add.bc
run_step "verify phase4 vector_add module" opt -passes=verify build/bcir_phase4_vector_add.bc -o /dev/null

require_pattern '@bcir.generated.claims' build/bcir_phase4_vector_add.ll 'generated claims global'
require_pattern '%bcir.blob.header' build/bcir_phase4_vector_add.ll 'blob header type'
require_pattern '@bcir.rehydrate.decide' build/bcir_phase4_vector_add.ll 'rehydrate decision symbol'
require_pattern '@bcir.telemetry.zero' build/bcir_phase4_vector_add.ll 'telemetry zeroing symbol'

log "validate_phase4.sh completed successfully"
