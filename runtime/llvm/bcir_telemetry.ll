source_filename = "bcir_telemetry.ll"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

%bcir.telemetry = type {
  i64,
  i64,
  i64,
  i64,
  i64,
  i64,
  i64,
  i64
}

define void @bcir.telemetry.zero(ptr %t) {
entry:
  %p0 = getelementptr inbounds i64, ptr %t, i64 0
  %p1 = getelementptr inbounds i64, ptr %t, i64 1
  %p2 = getelementptr inbounds i64, ptr %t, i64 2
  %p3 = getelementptr inbounds i64, ptr %t, i64 3
  %p4 = getelementptr inbounds i64, ptr %t, i64 4
  %p5 = getelementptr inbounds i64, ptr %t, i64 5
  %p6 = getelementptr inbounds i64, ptr %t, i64 6
  %p7 = getelementptr inbounds i64, ptr %t, i64 7
  store i64 0, ptr %p0, align 8
  store i64 0, ptr %p1, align 8
  store i64 0, ptr %p2, align 8
  store i64 0, ptr %p3, align 8
  store i64 0, ptr %p4, align 8
  store i64 0, ptr %p5, align 8
  store i64 0, ptr %p6, align 8
  store i64 0, ptr %p7, align 8
  ret void
}
