; Compare-exchange loop examples.
; Assemble/check with: llvm-as cmpxchg-loop.ll -o /dev/null

@max_value = global i32 0, align 4
@payload = global i32 0, align 4
@ready = global i8 0, align 1

; Atomically raise @max_value to %candidate if %candidate is larger.
define i32 @raise_max(i32 %candidate) {
entry:
  %initial = load atomic i32, ptr @max_value monotonic, align 4
  br label %loop

loop:
  %expected = phi i32 [ %initial, %entry ], [ %old, %retry ]
  %should_update = icmp slt i32 %expected, %candidate
  br i1 %should_update, label %try, label %done

try:
  %pair = cmpxchg ptr @max_value, i32 %expected, i32 %candidate monotonic monotonic, align 4
  %old = extractvalue { i32, i1 } %pair, 0
  %ok = extractvalue { i32, i1 } %pair, 1
  br i1 %ok, label %done, label %retry

retry:
  br label %loop

done:
  %result = phi i32 [ %expected, %loop ], [ %candidate, %try ]
  ret i32 %result
}

; Publish/consume flag pattern using release/acquire.
define void @publish_payload(i32 %value) {
entry:
  store i32 %value, ptr @payload, align 4
  store atomic i8 1, ptr @ready release, align 1
  ret void
}

define i32 @try_consume_payload() {
entry:
  %ready_byte = load atomic i8, ptr @ready acquire, align 1
  %is_ready = icmp ne i8 %ready_byte, 0
  br i1 %is_ready, label %ready_block, label %not_ready

ready_block:
  %value = load i32, ptr @payload, align 4
  ret i32 %value

not_ready:
  ret i32 -1
}
