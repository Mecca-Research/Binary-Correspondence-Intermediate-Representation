; Atomic counter examples.
; Assemble/check with: llvm-as atomic-counter.ll -o /dev/null

@counter = global i64 0, align 8
@next_id = global i64 1000, align 8

; Simple statistic: atomicity matters, but no other data is published.
define i64 @increment_counter() {
entry:
  %old = atomicrmw add ptr @counter, i64 1 monotonic, align 8
  %new = add i64 %old, 1
  ret i64 %new
}

; Unique ID allocation: fetch-and-add returns the old value.
define i64 @allocate_id() {
entry:
  %id = atomicrmw add ptr @next_id, i64 1 monotonic, align 8
  ret i64 %id
}

; Reset can be atomic too. Use stronger ordering only if reset publishes
; or consumes other memory in the surrounding protocol.
define void @reset_counter() {
entry:
  store atomic i64 0, ptr @counter monotonic, align 8
  ret void
}
