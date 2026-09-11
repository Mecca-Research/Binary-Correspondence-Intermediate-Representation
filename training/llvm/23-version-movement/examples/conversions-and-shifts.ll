; The cast and shift opcodes the corpus lists but never showed running.
;
; Every result asserted in the comments here was produced by LLVM 23's own constant
; folder (`opt -O2 -S`), not from recollection. These assemble on the corpus baseline
; too -- nothing here is new syntax; what was missing was a demonstration.

; ---------------------------------------------------------------------------
; ashr vs lshr: the only difference is what fills the vacated high bits.
; ---------------------------------------------------------------------------

; ashr shifts in copies of the sign bit, so a negative stays negative.
; opt folds this to  -4.
define i32 @arithmetic_shift_right() {
  %r = ashr i32 -8, 1
  ret i32 %r
}

; lshr shifts in zeros, so the sign bit becomes an ordinary value bit and the
; same input becomes a large positive number.
; opt folds this to  2147483644.
define i32 @logical_shift_right() {
  %r = lshr i32 -8, 1
  ret i32 %r
}

; ---------------------------------------------------------------------------
; Float -> int is POISON out of range. Not saturation, not a trap, not zero.
; ---------------------------------------------------------------------------

; 2^32 is exactly representable as a float and is one bit too wide for i32. opt folds
; this to  poison  -- and poison propagates, so a single out-of-range conversion can
; make a whole computation meaningless without any diagnostic at all.
;
; The constant is written 4294967296.0 rather than something like 1.0e30 on purpose:
; LLVM 15 and 18 reject a decimal float literal that is not exactly representable in
; its type ("floating point constant invalid for type"), while LLVM 23 accepts it. A
; power of two is exact everywhere, so this file stays baseline-assemblable.
define i32 @float_to_signed_out_of_range() {
  %r = fptosi float 4294967296.0 to i32
  ret i32 %r
}

; The same rule catches sign mismatches: -1.0 is perfectly representable as a float
; and is simply not in the range of an UNSIGNED i32. opt folds this to  poison.
define i32 @negative_float_to_unsigned() {
  %r = fptoui float -1.0 to i32
  ret i32 %r
}

; The saturating intrinsics exist precisely because the instructions are poison.
; This clamps instead: the result is i32's maximum, and it is defined for every input.
define i32 @float_to_signed_saturating(float %f) {
  %r = call i32 @llvm.fptosi.sat.i32.f32(float %f)
  ret i32 %r
}

; ---------------------------------------------------------------------------
; int -> float loses precision silently once the integer outruns the mantissa.
; ---------------------------------------------------------------------------

; A double has 53 significand bits, so 2^53 + 1 = 9007199254740993 is not
; representable. opt folds this to the double 0x4340000000000000, which is
; 9007199254740992 -- the input minus one, with nothing to signal the loss.
define double @signed_to_float_precision_loss() {
  %r = sitofp i64 9007199254740993 to double
  ret double %r
}

; uitofp differs from sitofp only in how it reads the source bits. For a value
; that is positive in both readings the two agree; for one that is not, they do not.
define double @unsigned_to_float(i64 %u) {
  %r = uitofp i64 %u to double
  ret double %r
}

; ---------------------------------------------------------------------------
; Float width changes, and negation that is not subtraction.
; ---------------------------------------------------------------------------

; fpext is exact: every float is a double. fptrunc is a rounding operation and may
; be inexact, which is why it is a separate opcode rather than a bitcast.
define double @widen(float %f) {
  %r = fpext float %f to double
  ret double %r
}

define float @narrow(double %d) {
  %r = fptrunc double %d to float
  ret float %r
}

; fneg flips the sign bit and nothing else. `fsub float -0.0, %f` is the older
; spelling and is NOT equivalent: it is a subtraction, so it follows subtraction's
; rules for NaN payloads and for signed zero. Use fneg when you mean negation.
define float @negate(float %f) {
  %r = fneg float %f
  ret float %r
}

declare i32 @llvm.fptosi.sat.i32.f32(float)
