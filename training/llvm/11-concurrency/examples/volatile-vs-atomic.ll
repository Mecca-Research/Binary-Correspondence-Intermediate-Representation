; volatile preserves side-effecting accesses; atomic participates in the memory model.
source_filename = "volatile-vs-atomic.c"

define i32 @volatile_load_twice(ptr %mmio) {
entry:
  %a = load volatile i32, ptr %mmio, align 4
  %b = load volatile i32, ptr %mmio, align 4
  %sum = add i32 %a, %b
  ret i32 %sum
}

define i32 @atomic_load_twice(ptr %shared) {
entry:
  %a = load atomic i32, ptr %shared acquire, align 4
  %b = load atomic i32, ptr %shared acquire, align 4
  %sum = add i32 %a, %b
  ret i32 %sum
}
