; types-cookbook.ll — one-stop tour of LLVM IR types
;
; Assemble: llvm-as types-cookbook.ll -o /dev/null

source_filename = "types-cookbook.ll"

; ---- Named struct ----
%Point  = type { i32, i32 }
%Person = type { i32, float, ptr }

; ---- Opaque type (forward declaration; body not specified here) ----
%Opaque = type opaque

; ---- Globals using composite types ----
@origin    = global %Point { i32 0, i32 0 }
@triangle  = global [3 x %Point] [
  %Point { i32 0, i32 0 },
  %Point { i32 1, i32 0 },
  %Point { i32 0, i32 1 }
]
@msg       = private unnamed_addr constant [6 x i8] c"hello\00"

; ---- Function: integers of various widths ----
define i64 @widen_demo(i32 %x) {
entry:
  %a = zext i32 %x to i64           ; unsigned widen
  %b = sext i32 %x to i64           ; signed   widen
  %c = trunc i64 %a to i16          ; narrow
  %d = zext i16 %c to i64
  %sum = add i64 %a, %d
  ret i64 %sum
}

; ---- Function: vectors and element access ----
define <4 x i32> @vec_demo(<4 x i32> %v) {
entry:
  %first = extractelement <4 x i32> %v, i32 0
  %new   = insertelement <4 x i32> %v, i32 42, i32 1
  %shuf  = shufflevector <4 x i32> %new, <4 x i32> %v,
                         <4 x i32> <i32 0, i32 5, i32 1, i32 6>
  ret <4 x i32> %shuf
}

; ---- Function: struct field access via GEP ----
define i32 @get_x(ptr %p) {
entry:
  %x_p = getelementptr inbounds %Point, ptr %p, i32 0, i32 0
  %x   = load i32, ptr %x_p, align 4
  ret i32 %x
}

define void @set_y(ptr %p, i32 %v) {
entry:
  %y_p = getelementptr inbounds %Point, ptr %p, i32 0, i32 1
  store i32 %v, ptr %y_p, align 4
  ret void
}

; ---- Function: array element access ----
define i32 @get_array_elem(ptr %arr, i32 %i) {
entry:
  %p = getelementptr inbounds [10 x i32], ptr %arr, i32 0, i32 %i
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

; ---- Function: SSA struct via insertvalue/extractvalue ----
define i32 @ssa_struct_demo(i32 %a, i32 %b) {
entry:
  %p0 = insertvalue { i32, i32 } undef, i32 %a, 0
  %p1 = insertvalue { i32, i32 } %p0, i32 %b, 1
  %x  = extractvalue { i32, i32 } %p1, 0
  %y  = extractvalue { i32, i32 } %p1, 1
  %r  = add i32 %x, %y
  ret i32 %r
}

; ---- Function: opaque pointer (no pointee type) ----
define void @opaque_ptr_demo(ptr %p) {
entry:
  ; The same `ptr` is interpreted differently per operation
  %v32   = load i32, ptr %p, align 4
  store i64 0, ptr %p, align 8
  %gep32 = getelementptr i32, ptr %p, i32 4    ; stride by 16 bytes
  %gepi8 = getelementptr i8, ptr %p, i32 4     ; stride by 4 bytes
  ret void
}
