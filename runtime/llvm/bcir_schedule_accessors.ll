source_filename = "bcir_schedule_accessors.ll"
target triple = "unknown-unknown-unknown"
target datalayout = ""

; Phase 3 schedule accessors for global-indexed batch windows.
%bcir.batch = type { i32, i32, i32, i32, i32, i32, i64, i64, i64 }
%bcir.stream.pack = type { ptr, i64, ptr, i64, ptr, i64, ptr, i64 }

define i32 @bcir.batch.epoch(ptr %batch) alwaysinline {
entry:
  %p = getelementptr inbounds %bcir.batch, ptr %batch, i32 0, i32 0
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

define i32 @bcir.batch.phase(ptr %batch) alwaysinline {
entry:
  %p = getelementptr inbounds %bcir.batch, ptr %batch, i32 0, i32 1
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

define i32 @bcir.batch.lane_class(ptr %batch) alwaysinline {
entry:
  %p = getelementptr inbounds %bcir.batch, ptr %batch, i32 0, i32 2
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

define i32 @bcir.batch.opcode(ptr %batch) alwaysinline {
entry:
  %p = getelementptr inbounds %bcir.batch, ptr %batch, i32 0, i32 3
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

define i64 @bcir.batch.first_claim_index(ptr %batch) alwaysinline {
entry:
  %p = getelementptr inbounds %bcir.batch, ptr %batch, i32 0, i32 6
  %v = load i64, ptr %p, align 8
  ret i64 %v
}

define i64 @bcir.batch.claim_count(ptr %batch) alwaysinline {
entry:
  %p = getelementptr inbounds %bcir.batch, ptr %batch, i32 0, i32 7
  %v = load i64, ptr %p, align 8
  ret i64 %v
}

define ptr @bcir.stream.pack.batches(ptr %pack) alwaysinline {
entry:
  %p = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 2
  %v = load ptr, ptr %p, align 8
  ret ptr %v
}

define i64 @bcir.stream.pack.batch_count(ptr %pack) alwaysinline {
entry:
  %p = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 3
  %v = load i64, ptr %p, align 8
  ret i64 %v
}
