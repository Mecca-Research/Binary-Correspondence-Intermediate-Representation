; Normalized `clang -O0 -S -emit-llvm -target x86_64-unknown-linux-gnu` output for cxx-object-model.cpp.
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

%struct.Guard = type { i32 }

$_Z5twiceIiET_S0_ = comdat any

define dso_local noundef i32 @_Z10total_areaRK5Shape(ptr noundef nonnull align 8 dereferenceable(8) %s) {
entry:
  %s.addr = alloca ptr, align 8
  store ptr %s, ptr %s.addr, align 8
  %0 = load ptr, ptr %s.addr, align 8
  %vtable = load ptr, ptr %0, align 8
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 2
  %1 = load ptr, ptr %vfn, align 8
  %call = call noundef i32 %1(ptr noundef nonnull align 8 dereferenceable(8) %0)
  ret i32 %call
}

define dso_local noundef i32 @_Z11square_areaRK6Square(ptr noundef nonnull align 8 dereferenceable(12) %s) {
entry:
  %s.addr = alloca ptr, align 8
  store ptr %s, ptr %s.addr, align 8
  %0 = load ptr, ptr %s.addr, align 8
  %vtable = load ptr, ptr %0, align 8
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 2
  %1 = load ptr, ptr %vfn, align 8
  %call = call noundef i32 %1(ptr noundef nonnull align 8 dereferenceable(12) %0)
  ret i32 %call
}

define dso_local noundef i32 @_Z5scalei(i32 noundef %v) {
entry:
  %v.addr = alloca i32, align 4
  store i32 %v, ptr %v.addr, align 4
  %0 = load i32, ptr %v.addr, align 4
  %mul = mul nsw i32 %0, 2
  ret i32 %mul
}

define dso_local noundef double @_Z5scaled(double noundef %v) {
entry:
  %v.addr = alloca double, align 8
  store double %v, ptr %v.addr, align 8
  %0 = load double, ptr %v.addr, align 8
  %mul = fmul double %0, 2.000000e+00
  ret double %mul
}

define dso_local noundef i32 @_Z12use_templatev() {
entry:
  %call = call noundef i32 @_Z5twiceIiET_S0_(i32 noundef 21)
  ret i32 %call
}

define linkonce_odr dso_local noundef i32 @_Z5twiceIiET_S0_(i32 noundef %v) comdat {
entry:
  %v.addr = alloca i32, align 4
  store i32 %v, ptr %v.addr, align 4
  %0 = load i32, ptr %v.addr, align 4
  %1 = load i32, ptr %v.addr, align 4
  %add = add nsw i32 %0, %1
  ret i32 %add
}

define dso_local noundef i32 @_Z7guardedv() {
entry:
  %ref.tmp = alloca %struct.Guard, align 4
  call void @_ZN5GuardC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %ref.tmp)
  %n = getelementptr inbounds %struct.Guard, ptr %ref.tmp, i32 0, i32 0
  %0 = load i32, ptr %n, align 4
  call void @_ZN5GuardD1Ev(ptr noundef nonnull align 4 dereferenceable(4) %ref.tmp)
  ret i32 %0
}

declare void @_ZN5GuardC1Ev(ptr noundef nonnull align 4 dereferenceable(4)) unnamed_addr

declare void @_ZN5GuardD1Ev(ptr noundef nonnull align 4 dereferenceable(4)) unnamed_addr
