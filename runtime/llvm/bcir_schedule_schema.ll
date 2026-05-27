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
