; opaque-pointer-after.ll — modern opaque-pointer IR
;
; Verify: opt -passes=verify -disable-output opaque-pointer-after.ll
; Assemble: llvm-as opaque-pointer-after.ll -o /dev/null

source_filename = "opaque-pointer-after.ll"

%Pair = type { i32, i32 }

declare void @take_ptr(ptr)
declare void @take_pair(ptr)

; Modern load/store: the access type is on load/store, not the pointer type.
define i32 @load_store_i32(ptr %src, ptr %dst) {
entry:
  %v = load i32, ptr %src, align 4
  store i32 %v, ptr %dst, align 4
  ret i32 %v
}

; Modern GEP: %Pair is explicit in the instruction and the result is ptr.
define i32 @load_second_field(ptr %p) {
entry:
  %field = getelementptr inbounds %Pair, ptr %p, i32 0, i32 1
  %v = load i32, ptr %field, align 4
  ret i32 %v
}

; Modern equivalent of the old bitcast chain: no pointer-to-pointer bitcasts.
define i32 @no_bitcast_chain(ptr %p) {
entry:
  call void @take_ptr(ptr %p)
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

; Byte-wise addressing still names the byte element type on GEP.
define i8 @load_byte_at(ptr %p, i64 %offset) {
entry:
  %byte.ptr = getelementptr i8, ptr %p, i64 %offset
  %b = load i8, ptr %byte.ptr, align 1
  ret i8 %b
}

; Address-space information remains part of the pointer type.
define i32 @addrspace_load(ptr addrspace(1) %p) {
entry:
  %v = load i32, ptr addrspace(1) %p, align 4
  ret i32 %v
}

; Function signatures use ptr, while uses and callees define semantics.
define void @call_examples(ptr %p) {
entry:
  call void @take_pair(ptr %p)
  call void @take_ptr(ptr %p)
  ret void
}
