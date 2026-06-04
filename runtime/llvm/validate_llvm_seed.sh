#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=runtime/llvm/validate_common.sh
source "$SCRIPT_DIR/validate_common.sh"

require_tools llvm-as opt llvm-dis
init_build_dir
log_toolchain llvm-as opt llvm-dis

require_file runtime/llvm/bcir_master_reference_v2.ll
run_step "assemble master reference" llvm-as runtime/llvm/bcir_master_reference_v2.ll -o build/bcir_master_reference_v2.bc
run_step "verify master reference" opt -passes=verify build/bcir_master_reference_v2.bc -o /dev/null
run_step "disassemble master reference" llvm-dis build/bcir_master_reference_v2.bc -o build/bcir_master_reference_v2.dis.ll

require_pattern '%bcir.claim = type' runtime/llvm/bcir_master_reference_v2.ll 'BCIR claim type declaration'
require_pattern 'declare i32 @bcir.op.atomic.add.i32' build/bcir_master_reference_v2.dis.ll 'atomic add interface declaration'
require_pattern 'declare { i32, i1 } @bcir.op.cmpxchg.i32' build/bcir_master_reference_v2.dis.ll 'cmpxchg interface declaration'
require_pattern 'declare void @bcir.op.barrier' build/bcir_master_reference_v2.dis.ll 'barrier interface declaration'
require_pattern 'declare <8 x i32> @bcir.op.load.v8i32' build/bcir_master_reference_v2.dis.ll 'vector load interface declaration'
require_pattern 'declare void @bcir.op.store.v8i32' build/bcir_master_reference_v2.dis.ll 'vector store interface declaration'
require_pattern 'declare void @bcir.op.phase.enter' build/bcir_master_reference_v2.dis.ll 'phase-enter interface declaration'
require_pattern 'declare void @bcir.gem.execute_claim' build/bcir_master_reference_v2.dis.ll 'GEM execute-claim interface declaration'
require_pattern 'declare void @bcir.gem.execute_worklist' build/bcir_master_reference_v2.dis.ll 'GEM execute-worklist interface declaration'
require_pattern '!bcir.claim.layout' build/bcir_master_reference_v2.dis.ll 'claim-layout metadata'
require_pattern '!bcir.lanes' build/bcir_master_reference_v2.dis.ll 'lane metadata'

log "validate_llvm_seed.sh completed successfully"
