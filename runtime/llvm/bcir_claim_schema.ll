;===----------------------------------------------------------------------===;
; BCIR Claim Schema v2
;===----------------------------------------------------------------------===;

source_filename = "bcir_claim_schema.ll"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

%bcir.claim = type {
  i64,
  [4 x i32],
  [4 x i32],
  i64,
  [2 x i64]
}

%bcir.execctx = type {
  ptr,
  i32,
  i32,
  i32,
  i64,
  ptr
}

%bcir.costvec.q16 = type {
  i32,
  i32,
  i32,
  i32,
  i32,
  i32,
  i32,
  i32,
  i32,
  i32,
  i32
}

!bcir.opcodes = !{!100}
!100 = !{!"NOP", i32 0, !"LOAD", i32 1, !"STORE", i32 2, !"ADD", i32 3, !"SUB", i32 4, !"MUL", i32 5, !"ATOMIC_ADD", i32 32, !"ATOMIC_SUB", i32 33, !"ATOMIC_XOR", i32 34, !"CMPXCHG", i32 35, !"BARRIER", i32 48, !"PHASE_ENTER", i32 49, !"PHASE_LEAVE", i32 50, !"PROV_NOTE", i32 51, !"GGG_LOAD", i32 64, !"GGG_STORE", i32 65, !"T_LOAD", i32 80, !"T_STORE", i32 81, !"T_MACC", i32 82, !"GEM_DISPATCH", i32 96}

!bcir.lanes = !{!101}
!101 = !{!"U", i32 0, !"UX", i32 1, !"T", i32 2, !"GGG", i32 3, !"A", i32 4, !"H", i32 5}

!bcir.domains = !{!102}
!102 = !{!"RAM", i32 0, !"VRAM", i32 1, !"NVM", i32 2, !"MMIO", i32 3, !"CXL", i32 4, !"HBM", i32 5}

!bcir.claim.layout = !{!103}
!103 = !{!"BCIR_ClaimV2", !"size_bytes", i32 64, !"control", i32 0, i32 8, !"rd_rids", i32 8, i32 16, !"wr_rids", i32 24, i32 16, !"hazard_domain", i32 40, i32 8, !"immediates", i32 48, i32 16}


!bcir.phase3 = !{!104}
!104 = !{!"BCIR_Phase3", !"hot_stream", !"claims -> phase ranges -> lane/opcode/type batches"}
