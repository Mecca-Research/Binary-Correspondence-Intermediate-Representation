source_filename = "bcir_registry_schema.ll"
target triple = "unknown-unknown-unknown"
target datalayout = ""

%bcir.cap = type {
  i32,
  i32,
  i64,
  i64
}

%bcir.res = type {
  i32,
  i32,
  ptr,
  i64,
  i64,
  i64,
  i64
}

%bcir.exe = type {
  i32,
  i32,
  i64,
  i64,
  i64,
  i64
}

%bcir.wl = type {
  i32,
  i32,
  i32,
  ptr,
  i64,
  i64,
  i64,
  i64
}

%bcir.blob.header = type {
  i32,
  i16,
  i16,
  i64,
  i64,
  i64,
  i64,
  i64
}

%bcir.blob.view = type {
  ptr,
  ptr,
  ptr,
  ptr,
  ptr,
  ptr,
  i64,
  ptr,
  i64
}
