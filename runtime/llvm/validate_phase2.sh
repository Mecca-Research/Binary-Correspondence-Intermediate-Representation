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

llvm-as runtime/llvm/bcir_master_reference_v2.ll -o build/bcir_master_reference_v2.bc
opt -passes=verify build/bcir_master_reference_v2.bc -o /dev/null

llvm-link \
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

llvm-as build/bcir_phase2_all.ll -o build/bcir_phase2_all.bc
opt -passes=verify build/bcir_phase2_all.bc -o /dev/null
llvm-dis build/bcir_phase2_all.bc -o build/bcir_phase2_all.dis.ll
