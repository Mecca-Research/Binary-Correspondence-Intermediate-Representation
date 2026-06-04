; A JIT may resolve declarations to absolute process symbols at runtime.
source_filename = "jit-absolute-symbol.c"

declare void @host_log(i32)

define void @jit_entry(i32 %code) {
entry:
  call void @host_log(i32 %code)
  ret void
}
