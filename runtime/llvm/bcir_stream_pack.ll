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
