source_filename = "bcir_master_reference_v2.ll"
target triple = "unknown-unknown-unknown"
target datalayout = ""

; Interface-only master reference for standalone validation.
; Canonical implementations live in:
; - bcir_claim_accessors.ll
; - bcir_ops.ll
; - bcir_gem_seed.ll

%bcir.claim = type { i64, [4 x i32], [4 x i32], i64, [2 x i64] }
%bcir.execctx = type { ptr, i32, i32, i32, i64, ptr }
%bcir.batch = type { i32, i32, i32, i32, i32, i32, i64, i64, i64 }

declare void @llvm.trap() cold noreturn

declare i64 @bcir.claim.opstride(ptr)
declare i8 @bcir.claim.opcode(ptr)
declare i8 @bcir.claim.lane(ptr)
declare i32 @bcir.claim.phase(ptr)
declare i32 @bcir.claim.epoch(ptr)
declare i32 @bcir.claim.rd(ptr, i32)
declare i32 @bcir.claim.wr(ptr, i32)
declare i64 @bcir.claim.imm(ptr, i32)

declare void @bcir.op.phase.enter(ptr, i32, i32)
declare void @bcir.op.phase.leave(ptr, i32, i32)
declare void @bcir.op.barrier(ptr, i32)
declare i32 @bcir.op.load.i32(ptr, i64)
declare void @bcir.op.store.i32(ptr, i64, i32)
declare <8 x i32> @bcir.op.load.v8i32(ptr, i64)
declare void @bcir.op.store.v8i32(ptr, i64, <8 x i32>)
declare i32 @bcir.op.atomic.add.i32(ptr, i64, i32)
declare i32 @bcir.op.atomic.sub.i32(ptr, i64, i32)
declare i32 @bcir.op.atomic.xor.i32(ptr, i64, i32)
declare { i32, i1 } @bcir.op.cmpxchg.i32(ptr, i64, i32, i32)

declare ptr @bcir.registry.lookup(ptr, i32)
declare void @bcir.gem.execute_claim(ptr, ptr, ptr)
declare void @bcir.gem.execute_worklist(ptr, ptr, i64, ptr)

!bcir.opcodes = !{!100}
!100 = !{!"NOP", i32 0, !"LOAD", i32 1, !"STORE", i32 2, !"ATOMIC_ADD", i32 32, !"CMPXCHG", i32 35, !"BARRIER", i32 48, !"PHASE_ENTER", i32 49, !"PHASE_LEAVE", i32 50}
!bcir.lanes = !{!101}
!101 = !{!"U", i32 0, !"UX", i32 1, !"T", i32 2, !"GGG", i32 3, !"A", i32 4, !"H", i32 5}
!bcir.domains = !{!102}
!102 = !{!"RAM", i32 0, !"VRAM", i32 1, !"NVM", i32 2, !"MMIO", i32 3, !"CXL", i32 4, !"HBM", i32 5}
!bcir.claim.layout = !{!103}
!103 = !{!"BCIR_ClaimV2", !"size_bytes", i32 64, !"control", i32 0, i32 8, !"rd_rids", i32 8, i32 16, !"wr_rids", i32 24, i32 16, !"hazard_domain", i32 40, i32 8, !"immediates", i32 48, i32 16}

!bcir.schedule = !{!300}
!300 = !{!"sort_order", !"epoch,phase,lane,opcode,type,hazard_domain", !"batch_key_bits", i32 64}

!bcir.master.phase3 = !{!400}
!400 = !{"phase3_seed", "stream-pack and batch execution path enabled"}
