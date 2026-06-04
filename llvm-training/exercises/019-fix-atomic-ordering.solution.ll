source_filename = "019-fix-atomic-ordering.solution.ll"

define void @publish(ptr %p, i32 %v) {
entry:
  store atomic i32 %v, ptr %p release, align 4
  ret void
}
