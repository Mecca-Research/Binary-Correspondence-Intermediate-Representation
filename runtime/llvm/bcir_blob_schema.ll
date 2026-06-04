source_filename = "bcir_blob_schema.ll"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

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
