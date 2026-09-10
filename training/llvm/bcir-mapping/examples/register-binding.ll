; BCIR register/resource binding sketch.
; Standalone example: a compact claim binds read/write resource IDs to slots.

source_filename = "register-binding.ll"
target triple = "unknown-unknown-unknown"
target datalayout = ""

%bcir.binding.claim = type { [4 x i32], [4 x i32] }
%bcir.binding.resource = type { ptr, i64 }

define i32 @bcir.example.load_bound_read0(ptr %claim, ptr %resource_table) {
entry:
  %rd0_ptr = getelementptr inbounds %bcir.binding.claim, ptr %claim, i32 0, i32 0, i64 0
  %rid32 = load i32, ptr %rd0_ptr, align 4
  %rid64 = zext i32 %rid32 to i64
  %res_ptr = getelementptr inbounds %bcir.binding.resource, ptr %resource_table, i64 %rid64
  %base_ptr = getelementptr inbounds %bcir.binding.resource, ptr %res_ptr, i32 0, i32 0
  %base = load ptr, ptr %base_ptr, align 8
  %value = load i32, ptr %base, align 4
  ret i32 %value
}

define void @bcir.example.copy_bound_read0_to_write0(ptr %claim, ptr %resource_table) {
entry:
  %rd0_ptr = getelementptr inbounds %bcir.binding.claim, ptr %claim, i32 0, i32 0, i64 0
  %wr0_ptr = getelementptr inbounds %bcir.binding.claim, ptr %claim, i32 0, i32 1, i64 0
  %rd32 = load i32, ptr %rd0_ptr, align 4
  %wr32 = load i32, ptr %wr0_ptr, align 4
  %rd64 = zext i32 %rd32 to i64
  %wr64 = zext i32 %wr32 to i64
  %read_res = getelementptr inbounds %bcir.binding.resource, ptr %resource_table, i64 %rd64
  %write_res = getelementptr inbounds %bcir.binding.resource, ptr %resource_table, i64 %wr64
  %read_base_ptr = getelementptr inbounds %bcir.binding.resource, ptr %read_res, i32 0, i32 0
  %write_base_ptr = getelementptr inbounds %bcir.binding.resource, ptr %write_res, i32 0, i32 0
  %read_base = load ptr, ptr %read_base_ptr, align 8
  %write_base = load ptr, ptr %write_base_ptr, align 8
  %value = load i32, ptr %read_base, align 4
  store i32 %value, ptr %write_base, align 4
  ret void
}
