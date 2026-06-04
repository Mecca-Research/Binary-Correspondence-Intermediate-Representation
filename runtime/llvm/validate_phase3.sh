#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=runtime/llvm/validate_common.sh
source "$SCRIPT_DIR/validate_common.sh"

require_tools llvm-as llvm-link opt llvm-dis
init_build_dir
log_toolchain llvm-as llvm-link opt llvm-dis

run_step "link phase3 runtime modules" llvm-link \
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

run_step "assemble phase3 linked module" llvm-as build/bcir_phase3_all.ll -o build/bcir_phase3_all.bc
run_step "verify phase3 linked module" opt -passes=verify build/bcir_phase3_all.bc -o /dev/null
run_step "disassemble phase3 linked module" llvm-dis build/bcir_phase3_all.bc -o build/bcir_phase3_all.dis.ll

require_pattern '%bcir.batch' build/bcir_phase3_all.dis.ll 'BCIR batch type'
require_pattern '%bcir.phase.range' build/bcir_phase3_all.dis.ll 'phase range type'
require_pattern 'llvm.prefetch' build/bcir_phase3_all.dis.ll 'LLVM prefetch intrinsic'
require_pattern '@bcir.gem.execute_batch' build/bcir_phase3_all.dis.ll 'batch executor symbol'
require_pattern '@bcir.classify.memory_lane' build/bcir_phase3_all.dis.ll 'lane classifier symbol'
require_pattern '%bcir.stream.pack' build/bcir_phase3_all.dis.ll 'stream pack type'
require_pattern '@bcir.gem.execute_stream_pack' build/bcir_phase3_all.dis.ll 'stream pack executor symbol'
require_pattern '@bcir.kbdi.score.batch.q16' build/bcir_phase3_all.dis.ll 'KBDI score symbol'
require_pattern '!bcir.kbdi.phase3' build/bcir_phase3_all.dis.ll 'KBDI phase3 metadata'

log "validate_phase3.sh completed successfully"
