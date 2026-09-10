; Normalized `clang -O0 -S -emit-llvm -target x86_64-unknown-linux-gnu` output for abi-boundary.c.
; Produced by clang 18. A host on that major must
; reproduce this file byte for byte; a different major reports its
; delta instead, because lowering legitimately moves between releases.
; Regenerate with:
;   python3 training/llvm/tools/verify-frontend-lowering.py --update
; Attribute groups, module flags, the ident string, and parameter
; attributes newer than LLVM 15 are stripped: they are build
; configuration rather than lowering, and their spellings move between
; releases faster than the corpus's LLVM >= 15 baseline allows.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Small = type { i32, i32 }
%struct.Medium = type { i32, i32, i32 }
%struct.Large = type { [8 x double] }

define dso_local i32 @take_small(i64 %0) {
  %2 = alloca %struct.Small, align 4
  store i64 %0, ptr %2, align 4
  %3 = getelementptr inbounds %struct.Small, ptr %2, i32 0, i32 0
  %4 = load i32, ptr %3, align 4
  %5 = getelementptr inbounds %struct.Small, ptr %2, i32 0, i32 1
  %6 = load i32, ptr %5, align 4
  %7 = add nsw i32 %4, %6
  ret i32 %7
}

define dso_local i32 @take_medium(i64 %0, i32 %1) {
  %3 = alloca %struct.Medium, align 4
  %4 = alloca { i64, i32 }, align 4
  %5 = getelementptr inbounds { i64, i32 }, ptr %4, i32 0, i32 0
  store i64 %0, ptr %5, align 4
  %6 = getelementptr inbounds { i64, i32 }, ptr %4, i32 0, i32 1
  store i32 %1, ptr %6, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %3, ptr align 4 %4, i64 12, i1 false)
  %7 = getelementptr inbounds %struct.Medium, ptr %3, i32 0, i32 0
  %8 = load i32, ptr %7, align 4
  %9 = getelementptr inbounds %struct.Medium, ptr %3, i32 0, i32 1
  %10 = load i32, ptr %9, align 4
  %11 = add nsw i32 %8, %10
  %12 = getelementptr inbounds %struct.Medium, ptr %3, i32 0, i32 2
  %13 = load i32, ptr %12, align 4
  %14 = add nsw i32 %11, %13
  ret i32 %14
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)

define dso_local double @take_large(ptr noundef byval(%struct.Large) align 8 %0) {
  %2 = getelementptr inbounds %struct.Large, ptr %0, i32 0, i32 0
  %3 = getelementptr inbounds [8 x double], ptr %2, i64 0, i64 0
  %4 = load double, ptr %3, align 8
  %5 = getelementptr inbounds %struct.Large, ptr %0, i32 0, i32 0
  %6 = getelementptr inbounds [8 x double], ptr %5, i64 0, i64 7
  %7 = load double, ptr %6, align 8
  %8 = fadd double %4, %7
  ret double %8
}

define dso_local i64 @return_small() {
  %1 = alloca %struct.Small, align 4
  %2 = call i64 @make_small(i32 noundef 1, i32 noundef 2)
  store i64 %2, ptr %1, align 4
  %3 = load i64, ptr %1, align 4
  ret i64 %3
}

declare i64 @make_small(i32 noundef, i32 noundef)

define dso_local void @return_large(ptr noalias sret(%struct.Large) align 8 %0) {
  call void @make_large(ptr sret(%struct.Large) align 8 %0)
  ret void
}

declare void @make_large(ptr sret(%struct.Large) align 8)

define dso_local signext i8 @narrow(i8 noundef signext %0, i16 noundef signext %1, i1 noundef zeroext %2) {
  %4 = alloca i8, align 1
  %5 = alloca i16, align 2
  %6 = alloca i8, align 1
  store i8 %0, ptr %4, align 1
  store i16 %1, ptr %5, align 2
  %7 = zext i1 %2 to i8
  store i8 %7, ptr %6, align 1
  %8 = load i8, ptr %4, align 1
  %9 = sext i8 %8 to i32
  %10 = load i16, ptr %5, align 2
  %11 = sext i16 %10 to i32
  %12 = add nsw i32 %9, %11
  %13 = load i8, ptr %6, align 1
  %14 = trunc i8 %13 to i1
  %15 = zext i1 %14 to i32
  %16 = add nsw i32 %12, %15
  %17 = trunc i32 %16 to i8
  ret i8 %17
}

define dso_local i32 @call_varargs() {
  %1 = call i32 (i32, ...) @sum_varargs(i32 noundef 3, i32 noundef 1, i32 noundef 2, i32 noundef 3)
  ret i32 %1
}

declare i32 @sum_varargs(i32 noundef, ...)
