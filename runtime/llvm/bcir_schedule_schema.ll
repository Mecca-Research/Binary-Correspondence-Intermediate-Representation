source_filename = "bcir_schedule_schema.ll"
target triple = "unknown-unknown-unknown"
target datalayout = ""

%bcir.phase.range = type {
  i32,
  i32,
  i64,
  i64
}

%bcir.batch = type {
  i32,
  i32,
  i32,
  i32,
  i32,
  i32,
  i64,
  i64,
  i64
}

%bcir.layout.profile = type {
  i32,
  i32,
  i32,
  i32,
  i64
}

%bcir.prefetch.profile = type {
  i32,
  i32,
  i32,
  i32,
  i32,
  i32,
  i64
}

%bcir.tile.profile = type {
  i32,
  i32,
  i32,
  i32,
  i64
}

%bcir.stream.pack = type {
  ptr,
  i64,
  ptr,
  i64,
  ptr,
  i64,
  ptr,
  i64
}

!bcir.schedule.schema = !{!300}
!300 = !{
  !"BCIR Phase 3 Schedule Schema",
  !"sort_order", !"epoch,phase,lane,opcode,type,hazard_domain",
  !"execution", !"phase_ranges -> batches -> tight kernels"
}

!bcir.batch.layout = !{!301}
!301 = !{!"BCIR_BatchV1", !"size_bytes", i32 48, !"epoch", i32 0, i32 4, !"phase", i32 4, i32 4, !"lane_class", i32 8, i32 4, !"opcode", i32 12, i32 4, !"flags", i32 16, i32 4, !"first_claim_index", i32 24, i32 8, !"claim_count", i32 32, i32 8, !"prefetch_profile_index", i32 40, i32 8}
