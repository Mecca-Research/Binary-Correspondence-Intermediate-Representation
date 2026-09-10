; Verifier-safe ASan-like instrumentation shape.
; This is a training example, not a complete sanitizer runtime integration.
source_filename = "sanitizer-instrumentation.c"
target triple = "x86_64-unknown-linux-gnu"

@__asan_shadow_memory_dynamic_address = external local_unnamed_addr global i64

declare void @__asan_report_load4(i64)

define i32 @asan_like_checked_load(ptr %p) {
entry:
  ; Instrumentation computes the shadow byte for the real program address.
  %addr = ptrtoint ptr %p to i64
  %shadow_base = load i64, ptr @__asan_shadow_memory_dynamic_address, align 8, !nosanitize !0
  %shadow_index = lshr i64 %addr, 3
  %shadow_addr = add i64 %shadow_index, %shadow_base
  %shadow_ptr = inttoptr i64 %shadow_addr to ptr
  %shadow = load i8, ptr %shadow_ptr, align 1, !nosanitize !0
  %poisoned = icmp ne i8 %shadow, 0
  br i1 %poisoned, label %slowpath, label %ok

slowpath:
  ; Partial-granule redzone check for a 4-byte load.
  %last_byte = and i64 %addr, 7
  %access_end = add i64 %last_byte, 3
  %shadow_as_i64 = zext i8 %shadow to i64
  %overlaps_poison = icmp sge i64 %access_end, %shadow_as_i64
  br i1 %overlaps_poison, label %report, label %ok

report:
  call void @__asan_report_load4(i64 %addr)
  unreachable

ok:
  ; The original program access stays here.  The preceding shadow access is not
  ; dead code: it decides whether this load is safe to execute.
  %value = load i32, ptr %p, align 4
  ret i32 %value
}

!0 = !{}
