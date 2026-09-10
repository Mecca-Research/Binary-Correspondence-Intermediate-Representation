; Comparisons and select: signed vs unsigned predicates, ordered vs unordered
; float comparisons, pointer equality, vector masks, and poison-safe selection.
;
; Companion chapter: training/llvm/05-control-flow/06-comparisons-and-select.md
;
;   llvm-as training/llvm/05-control-flow/examples/comparisons-and-select.ll -o /dev/null
;   opt -passes=verify -S training/llvm/05-control-flow/examples/comparisons-and-select.ll -o /dev/null
;
; Assembly-only: every function takes caller-provided arguments.

declare i1 @llvm.vector.reduce.or.v4i1(<4 x i1>)

; --- the same bits, two answers ---------------------------------------------
; @signed_lt and @unsigned_lt differ only in the predicate and disagree on every
; input whose sign bits differ. Nothing in the type system distinguishes them.

define i1 @signed_lt(i32 %a, i32 %b) {
entry:
  %c = icmp slt i32 %a, %b
  ret i1 %c
}

define i1 @unsigned_lt(i32 %a, i32 %b) {
entry:
  %c = icmp ult i32 %a, %b
  ret i1 %c
}

; --- ordered vs unordered ----------------------------------------------------
; "o" predicates are false when either operand is NaN; "u" predicates are true.
; @is_not_nan and @is_nan are the canonical self-comparison idioms.

define i1 @is_not_nan(double %x) {
entry:
  %c = fcmp oeq double %x, %x
  ret i1 %c
}

define i1 @is_nan(double %x) {
entry:
  %c = fcmp uno double %x, %x
  ret i1 %c
}

; C's `x != y` is the unordered form.
define i1 @c_not_equal(double %x, double %y) {
entry:
  %c = fcmp une double %x, %y
  ret i1 %c
}

; --- pointer comparison ------------------------------------------------------
; Equality is always meaningful; relational comparison only within one
; allocation. Both operands must share an address space.

define i1 @same_object(ptr %p, ptr %q) {
entry:
  %c = icmp eq ptr %p, %q
  ret i1 %c
}

; --- vector masks feed select, not br ---------------------------------------

define <4 x i32> @lanewise_max(<4 x i32> %v, <4 x i32> %w) {
entry:
  %mask = icmp sgt <4 x i32> %v, %w
  %r = select <4 x i1> %mask, <4 x i32> %v, <4 x i32> %w
  ret <4 x i32> %r
}

; A <4 x i1> cannot be a branch condition: reduce it to a scalar i1 first.
define i32 @any_lane_positive(<4 x i32> %v) {
entry:
  %zero = insertelement <4 x i32> poison, i32 0, i32 0
  %zeros = shufflevector <4 x i32> %zero, <4 x i32> poison, <4 x i32> zeroinitializer
  %mask = icmp sgt <4 x i32> %v, %zeros
  %any = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> %mask)
  br i1 %any, label %some, label %none

some:
  ret i32 1

none:
  ret i32 0
}

; --- select evaluates both arms ---------------------------------------------
; The not-taken arm still contributes its value. If it can be poison, freeze it
; before the select; a branch would never have evaluated it at all.

define i32 @clamp_low(i32 %x) {
entry:
  %negative = icmp slt i32 %x, 0
  %r = select i1 %negative, i32 0, i32 %x
  ret i32 %r
}

define i32 @poison_safe_select(i1 %c, i32 %a, i32 %shift_amount) {
entry:
  ; %maybe_poison is poison when %shift_amount >= 32.
  %maybe_poison = shl nuw i32 %a, %shift_amount
  %frozen = freeze i32 %maybe_poison
  %r = select i1 %c, i32 %frozen, i32 0
  ret i32 %r
}

; --- predicate idioms --------------------------------------------------------

define i32 @three_way(i32 %a, i32 %b) {
entry:
  %gt = icmp sgt i32 %a, %b
  %lt = icmp slt i32 %a, %b
  %hi = select i1 %gt, i32 1, i32 0
  %r = select i1 %lt, i32 -1, i32 %hi
  ret i32 %r
}

define i32 @count_if(i32 %acc, i32 %x, i32 %limit) {
entry:
  %c = icmp ult i32 %x, %limit
  %inc = zext i1 %c to i32
  %r = add i32 %acc, %inc
  ret i32 %r
}

define i32 @mask_from_predicate(i32 %x) {
entry:
  %c = icmp sgt i32 %x, 0
  %m = sext i1 %c to i32
  ret i32 %m
}

define i1 @logical_not(i1 %c) {
entry:
  %n = xor i1 %c, true
  ret i1 %n
}
