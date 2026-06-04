; Lowering of claim-resource-lookup.bcir.txt.
; Claim read/write IDs become registry GEPs and resource base-pointer loads.

source_filename = "claim-resource-lookup.bcir.txt"
target triple = "unknown-unknown-unknown"
target datalayout = ""

%bcir.map.claim = type { [4 x i32], [4 x i32], i32 }
%bcir.map.resource = type { i32, i32, ptr, i64 }

define ptr @bcir.map.lookup_read0_base(ptr %claim, ptr %registry) {
entry:
  %rid_ptr = getelementptr inbounds %bcir.map.claim, ptr %claim, i32 0, i32 0, i64 0
  %rid32 = load i32, ptr %rid_ptr, align 4
  %rid64 = zext i32 %rid32 to i64
  %res_ptr = getelementptr inbounds %bcir.map.resource, ptr %registry, i64 %rid64
  %base_ptr = getelementptr inbounds %bcir.map.resource, ptr %res_ptr, i32 0, i32 2
  %base = load ptr, ptr %base_ptr, align 8
  ret ptr %base
}

define void @bcir.map.copy_read0_to_write0(ptr %claim, ptr %registry) {
entry:
  %rd_ptr = getelementptr inbounds %bcir.map.claim, ptr %claim, i32 0, i32 0, i64 0
  %wr_ptr = getelementptr inbounds %bcir.map.claim, ptr %claim, i32 0, i32 1, i64 0
  %rd32 = load i32, ptr %rd_ptr, align 4
  %wr32 = load i32, ptr %wr_ptr, align 4
  %rd64 = zext i32 %rd32 to i64
  %wr64 = zext i32 %wr32 to i64
  %rd_res = getelementptr inbounds %bcir.map.resource, ptr %registry, i64 %rd64
  %wr_res = getelementptr inbounds %bcir.map.resource, ptr %registry, i64 %wr64
  %rd_base_ptr = getelementptr inbounds %bcir.map.resource, ptr %rd_res, i32 0, i32 2
  %wr_base_ptr = getelementptr inbounds %bcir.map.resource, ptr %wr_res, i32 0, i32 2
  %rd_base = load ptr, ptr %rd_base_ptr, align 8
  %wr_base = load ptr, ptr %wr_base_ptr, align 8
  %value = load i32, ptr %rd_base, align 4
  store i32 %value, ptr %wr_base, align 4
  ret void
}
