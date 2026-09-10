; Normalized `clang -O0 -S -emit-llvm -target x86_64-unknown-linux-gnu` output for struct-layout.c.
; Regenerate with:
;   python3 llvm-training/tools/verify-frontend-lowering.py --update
; Attribute groups, module flags, and the ident string are stripped:
; they are build configuration, not lowering, and their spellings move
; between releases faster than the corpus's LLVM >= 15 baseline allows.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Padded = type { i8, i32, i8 }
%struct.Packed = type <{ i8, i32, i8 }>
%union.Word = type { i32 }

define dso_local i32 @padded_value(ptr noundef %0) {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %struct.Padded, ptr %3, i32 0, i32 1
  %5 = load i32, ptr %4, align 4
  ret i32 %5
}

define dso_local i32 @packed_value(ptr noundef %0) {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %struct.Packed, ptr %3, i32 0, i32 1
  %5 = load i32, ptr %4, align 1
  ret i32 %5
}

define dso_local i32 @bits_count(ptr noundef %0) {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = load i32, ptr %3, align 4
  %5 = lshr i32 %4, 3
  %6 = and i32 %5, 4095
  ret i32 %6
}

define dso_local void @bits_set_count(ptr noundef %0, i32 noundef %1) {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  store ptr %0, ptr %3, align 8
  store i32 %1, ptr %4, align 4
  %5 = load i32, ptr %4, align 4
  %6 = load ptr, ptr %3, align 8
  %7 = load i32, ptr %6, align 4
  %8 = and i32 %5, 4095
  %9 = shl i32 %8, 3
  %10 = and i32 %7, -32761
  %11 = or i32 %10, %9
  store i32 %11, ptr %6, align 4
  ret void
}

define dso_local i32 @union_bits_of_float(float noundef %0) {
  %2 = alloca float, align 4
  %3 = alloca %union.Word, align 4
  store float %0, ptr %2, align 4
  %4 = load float, ptr %2, align 4
  store float %4, ptr %3, align 4
  %5 = load i32, ptr %3, align 4
  ret i32 %5
}

define dso_local i32 @array_element(ptr noundef align 4 dereferenceable(32) %0, i32 noundef %1) {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  store ptr %0, ptr %3, align 8
  store i32 %1, ptr %4, align 4
  %5 = load ptr, ptr %3, align 8
  %6 = load i32, ptr %4, align 4
  %7 = sext i32 %6 to i64
  %8 = getelementptr inbounds i32, ptr %5, i64 %7
  %9 = load i32, ptr %8, align 4
  ret i32 %9
}

define dso_local i64 @padded_size() {
  ret i64 12
}

define dso_local i64 @padded_offset() {
  ret i64 4
}

define dso_local i64 @packed_size() {
  ret i64 6
}
