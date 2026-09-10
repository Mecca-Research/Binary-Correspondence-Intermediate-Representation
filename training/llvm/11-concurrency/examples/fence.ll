; Fence examples.
; Assemble/check with: llvm-as fence.ll -o /dev/null

@payload = global i32 0, align 4
@ready = global i8 0, align 1

; Fence-based publication: the fence carries release ordering, while the
; flag store is monotonic.
define void @publish_with_fence(i32 %value) {
entry:
  store i32 %value, ptr @payload, align 4
  fence release
  store atomic i8 1, ptr @ready monotonic, align 1
  ret void
}

; Matching acquire fence after seeing the monotonic flag.
define i32 @consume_with_fence() {
entry:
  %seen_byte = load atomic i8, ptr @ready monotonic, align 1
  %seen = icmp ne i8 %seen_byte, 0
  br i1 %seen, label %ready_block, label %not_ready

ready_block:
  fence acquire
  %value = load i32, ptr @payload, align 4
  ret i32 %value

not_ready:
  ret i32 -1
}

; A sequentially consistent fence participates in the global seq_cst order.
define void @global_barrier() {
entry:
  fence seq_cst
  ret void
}
