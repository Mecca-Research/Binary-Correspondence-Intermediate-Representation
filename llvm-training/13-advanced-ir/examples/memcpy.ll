target triple = "x86_64-unknown-linux-gnu"

; Common memory intrinsics.
; Assemble/check with: llvm-as memcpy.ll -o /dev/null

@src = internal global [16 x i8] c"BCIR intrinsic!\00", align 16
@dst = internal global [16 x i8] zeroinitializer, align 16
@buf = internal global [32 x i8] zeroinitializer, align 16

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
declare void @llvm.prefetch(ptr, i32 immarg, i32 immarg, i32 immarg)

define void @copy_global() {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr @dst, ptr @src, i64 16, i1 false)
  ret void
}

define void @shift_buffer_left() {
entry:
  %from = getelementptr i8, ptr @buf, i64 1
  call void @llvm.memmove.p0.p0.i64(ptr @buf, ptr %from, i64 31, i1 false)
  ret void
}

define void @clear_and_prefetch(ptr %p) {
entry:
  call void @llvm.lifetime.start.p0(i64 32, ptr @buf)
  call void @llvm.memset.p0.i64(ptr @buf, i8 0, i64 32, i1 false)
  call void @llvm.prefetch(ptr %p, i32 0, i32 3, i32 1)
  call void @llvm.lifetime.end.p0(i64 32, ptr @buf)
  ret void
}
