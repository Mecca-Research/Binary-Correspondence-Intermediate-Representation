; Atomic ordering pairs: release store synchronizes with acquire load externally.
source_filename = "atomic-ordering-pairs.c"

define void @publish(ptr %flag, ptr %data, i32 %value) {
entry:
  store i32 %value, ptr %data, align 4
  store atomic i32 1, ptr %flag release, align 4
  ret void
}

define i32 @consume(ptr %flag, ptr %data) {
entry:
  %ready = load atomic i32, ptr %flag acquire, align 4
  %is.ready = icmp eq i32 %ready, 1
  br i1 %is.ready, label %load, label %not_ready

load:
  %value = load i32, ptr %data, align 4
  ret i32 %value

not_ready:
  ret i32 -1
}
