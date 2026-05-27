source_filename = "bcir_blob_verify.ll"
target triple = "unknown-unknown-unknown"
target datalayout = ""

%bcir.blob.header = type { i32, i16, i16, i64, i64, i64, i64, i64 }
%bcir.blob.view = type { ptr, ptr, ptr, ptr, ptr, ptr, i64, ptr, i64 }

define i1 @bcir.blob.verify.header(ptr %h) {
entry:
  %magic_p = getelementptr inbounds %bcir.blob.header, ptr %h, i32 0, i32 0
  %magic = load i32, ptr %magic_p, align 4
  %ok = icmp eq i32 %magic, 1111903554
  ret i1 %ok
}

define i1 @bcir.blob.verify.view(ptr %v) {
entry:
  %hptr_p = getelementptr inbounds %bcir.blob.view, ptr %v, i32 0, i32 0
  %hptr = load ptr, ptr %hptr_p, align 8
  %nonnull = icmp ne ptr %hptr, null
  br i1 %nonnull, label %check, label %bad
check:
  %okh = call i1 @bcir.blob.verify.header(ptr %hptr)
  ret i1 %okh
bad:
  ret i1 false
}
