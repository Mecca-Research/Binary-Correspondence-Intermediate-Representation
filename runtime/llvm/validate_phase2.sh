#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=runtime/llvm/validate_common.sh
source "$SCRIPT_DIR/validate_common.sh"

require_tools llvm-as llvm-link opt llvm-dis
init_build_dir
log_toolchain llvm-as llvm-link opt llvm-dis

require_file runtime/llvm/bcir_master_reference_v2.ll
run_step "assemble master reference" llvm-as runtime/llvm/bcir_master_reference_v2.ll -o build/bcir_master_reference_v2.bc
run_step "verify master reference" opt -passes=verify build/bcir_master_reference_v2.bc -o /dev/null

run_step "link phase2 runtime modules" llvm-link \
  runtime/llvm/bcir_claim_schema.ll \
  runtime/llvm/bcir_claim_accessors.ll \
  runtime/llvm/bcir_registry_schema.ll \
  runtime/llvm/bcir_ops.ll \
  runtime/llvm/bcir_phase_epoch.ll \
  runtime/llvm/bcir_claim_verify.ll \
  runtime/llvm/bcir_gem_seed.ll \
  runtime/llvm/bcir_phase_worklist.ll \
  runtime/llvm/bcir_kbcost.ll \
  -S -o build/bcir_phase2_all.ll

run_step "assemble phase2 linked module" llvm-as build/bcir_phase2_all.ll -o build/bcir_phase2_all.bc
run_step "verify phase2 linked module" opt -passes=verify build/bcir_phase2_all.bc -o /dev/null
run_step "disassemble phase2 linked module" llvm-dis build/bcir_phase2_all.bc -o build/bcir_phase2_all.dis.ll

log "validate_phase2.sh completed successfully"
