; noreturn callees are commonly followed by unreachable.
source_filename = "unreachable-error-block.c"

declare void @trap_handler() noreturn

define i32 @checked_divisor(i32 %den) {
entry:
  %bad = icmp eq i32 %den, 0
  br i1 %bad, label %error, label %ok

ok:
  ret i32 %den

error:
  call void @trap_handler()
  unreachable
}
