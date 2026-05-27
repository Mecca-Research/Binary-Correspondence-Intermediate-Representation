source_filename = "bcir_stream_pack.ll"
target triple = "unknown-unknown-unknown"
target datalayout = ""

%bcir.stream.pack = type { ptr, i64, ptr, i64, ptr, i64, ptr, i64 }

define ptr @bcir.stream.pack.claims(ptr %pack) alwaysinline {
entry:
  %p = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 0
  %v = load ptr, ptr %p, align 8
  ret ptr %v
}

define i64 @bcir.stream.pack.claim_count(ptr %pack) alwaysinline {
entry:
  %p = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 1
  %v = load i64, ptr %p, align 8
  ret i64 %v
}


define void @bcir.stream.pack.init(ptr %pack, ptr %claims, i64 %claim_count, ptr %batches, i64 %batch_count, ptr %phase_ranges, i64 %phase_count, ptr %prefetch_profiles, i64 %prefetch_count) {
entry:
  %c0 = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 0
  %c1 = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 1
  %b0 = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 2
  %b1 = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 3
  %p0 = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 4
  %p1 = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 5
  %pf0 = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 6
  %pf1 = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 7
  store ptr %claims, ptr %c0, align 8
  store i64 %claim_count, ptr %c1, align 8
  store ptr %batches, ptr %b0, align 8
  store i64 %batch_count, ptr %b1, align 8
  store ptr %phase_ranges, ptr %p0, align 8
  store i64 %phase_count, ptr %p1, align 8
  store ptr %prefetch_profiles, ptr %pf0, align 8
  store i64 %prefetch_count, ptr %pf1, align 8
  ret void
}
