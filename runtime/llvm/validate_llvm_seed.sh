#!/usr/bin/env bash
set -euo pipefail

command -v llvm-as >/dev/null || { echo "missing llvm-as"; exit 127; }
command -v opt >/dev/null || { echo "missing opt"; exit 127; }
command -v llvm-dis >/dev/null || { echo "missing llvm-dis"; exit 127; }

mkdir -p build

llvm-as runtime/llvm/bcir_master_reference_v2.ll -o build/bcir_master_reference_v2.bc
opt -passes=verify build/bcir_master_reference_v2.bc -o /dev/null
llvm-dis build/bcir_master_reference_v2.bc -o build/bcir_master_reference_v2.dis.ll

grep -q "%bcir.claim = type" build/bcir_master_reference_v2.dis.ll
grep -q "declare i32 @bcir.op.atomic.add.i32" build/bcir_master_reference_v2.dis.ll
grep -q "declare { i32, i1 } @bcir.op.cmpxchg.i32" build/bcir_master_reference_v2.dis.ll
grep -q "declare void @bcir.op.barrier" build/bcir_master_reference_v2.dis.ll
grep -q "declare <8 x i32> @bcir.op.load.v8i32" build/bcir_master_reference_v2.dis.ll
grep -q "declare void @bcir.op.store.v8i32" build/bcir_master_reference_v2.dis.ll
grep -q "declare void @bcir.op.phase.enter" build/bcir_master_reference_v2.dis.ll
grep -q "declare void @bcir.gem.execute_claim" build/bcir_master_reference_v2.dis.ll
grep -q "declare void @bcir.gem.execute_worklist" build/bcir_master_reference_v2.dis.ll
grep -q "!bcir.claim.layout" build/bcir_master_reference_v2.dis.ll
grep -q "!bcir.lanes" build/bcir_master_reference_v2.dis.ll
