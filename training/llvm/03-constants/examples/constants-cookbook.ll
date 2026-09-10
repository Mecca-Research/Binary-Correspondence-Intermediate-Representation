; constants-cookbook.ll — integer, floating-point, string, aggregate, and global constants.
; Assemble/check with: llvm-as constants-cookbook.ll -o /dev/null

source_filename = "constants-cookbook.ll"

%Pair = type { i32, i32 }

; Integer constants can be written in decimal or hexadecimal form.
@answer = constant i32 42, align 4
@mask   = constant i64 255, align 8

; Floating-point constants may use decimal spelling or exact hexadecimal bits.
@half = constant float 5.000000e-01, align 4
@pi   = constant double 0x400921FB54442D18, align 8

; Strings are arrays of bytes. The trailing \00 is an explicit NUL byte.
@message = private unnamed_addr constant [13 x i8] c"hello, LLVM!\00", align 1

; Aggregate constants initialize arrays and structs in place.
@origin = constant %Pair { i32 0, i32 0 }, align 4
@small_primes = constant [4 x i32] [i32 2, i32 3, i32 5, i32 7], align 16
@record = constant { i32, double, ptr } {
  i32 7,
  double 0x4005BF0A8B145769,
  ptr @message
}, align 8

; Constant expressions can compute addresses inside global constants.
@message_body = private constant ptr getelementptr inbounds ([13 x i8], ptr @message, i64 0, i64 0), align 8

define i32 @integer_constant_demo() {
entry:
  %value = add i32 40, 2
  ret i32 %value
}

define double @floating_constant_demo() {
entry:
  %tau = fadd double 0x400921FB54442D18, 0x400921FB54442D18
  ret double %tau
}

define ptr @string_constant_demo() {
entry:
  %body = load ptr, ptr @message_body, align 8
  ret ptr %body
}

define i32 @aggregate_constant_demo(i32 %index) {
entry:
  %slot = getelementptr inbounds [4 x i32], ptr @small_primes, i64 0, i32 %index
  %prime = load i32, ptr %slot, align 4
  ret i32 %prime
}
