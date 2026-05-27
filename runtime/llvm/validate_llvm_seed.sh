#!/usr/bin/env bash
set -euo pipefail

mkdir -p build

llvm-as runtime/llvm/bcir_master_reference_v2.ll -o build/bcir_master_reference_v2.bc
opt -passes=verify build/bcir_master_reference_v2.bc -o /dev/null
llvm-dis build/bcir_master_reference_v2.bc -o build/bcir_master_reference_v2.dis.ll

grep -q "%bcir.claim = type { i64, \[4 x i32\], \[4 x i32\], i64, \[2 x i64\] }" build/bcir_master_reference_v2.dis.ll
grep -q "atomicrmw add" build/bcir_master_reference_v2.dis.ll
grep -q "cmpxchg" build/bcir_master_reference_v2.dis.ll
grep -q "fence seq_cst" build/bcir_master_reference_v2.dis.ll
grep -q "load <8 x i32>" build/bcir_master_reference_v2.dis.ll
grep -q "store <8 x i32>" build/bcir_master_reference_v2.dis.ll
grep -q "@bcir.op.phase.enter" build/bcir_master_reference_v2.dis.ll
grep -q "@bcir.op.barrier" build/bcir_master_reference_v2.dis.ll
grep -q "@bcir.gem.execute_claim" build/bcir_master_reference_v2.dis.ll
grep -q "@bcir.gem.execute_worklist" build/bcir_master_reference_v2.dis.ll
grep -q "!bcir.claim.layout" build/bcir_master_reference_v2.dis.ll
grep -q "!bcir.lanes" build/bcir_master_reference_v2.dis.ll
