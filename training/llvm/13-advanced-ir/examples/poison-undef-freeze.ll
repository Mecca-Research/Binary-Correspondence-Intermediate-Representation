target triple = "x86_64-unknown-linux-gnu"

; Examples for 05-poison-undef-freeze.md.
; These functions are verifier-valid. Comments mark which patterns are unsafe
; for some dynamic inputs even though the IR shape is accepted.

declare i32 @abi_requires_noundef(i32 noundef)

; Safe use of freeze: choose one arbitrary i32 for this execution, then use that
; same chosen value twice.
define i32 @freeze_undef_once() {
entry:
  %x = freeze i32 undef
  %twice = add i32 %x, %x
  ret i32 %twice
}

; Unsafe for %x == 2147483647: add nsw creates poison, the comparison propagates
; it, and branching on the poison i1 is undefined behavior.
define i32 @branch_on_poison_capable_add(i32 %x) {
entry:
  %sum = add nsw i32 %x, 1
  %take = icmp sgt i32 %sum, 0
  br i1 %take, label %positive, label %nonpositive

positive:
  ret i32 %sum

nonpositive:
  ret i32 0
}

; Safer speculation pattern: freeze the poison-capable value before feeding a
; comparison whose result controls a branch. This does not recover source
; semantics; it only prevents poison from reaching the branch condition.
define i32 @freeze_before_branch(i32 %x) {
entry:
  %sum = add nsw i32 %x, 1
  %stable = freeze i32 %sum
  %take = icmp sgt i32 %stable, 0
  br i1 %take, label %positive, label %nonpositive

positive:
  ret i32 %stable

nonpositive:
  ret i32 0
}

; Vector poison is lane-sensitive. If one lane overflows under nsw, extracting
; and using only that lane can be unsafe while other lanes may remain defined.
define i32 @extract_poison_capable_lane(<4 x i32> %v) {
entry:
  %inc = add nsw <4 x i32> %v, <i32 1, i32 1, i32 1, i32 1>
  %lane2 = extractelement <4 x i32> %inc, i32 2
  ret i32 %lane2
}

; Unsafe ABI boundary: the call is verifier-valid, but the declaration promises
; a noundef argument. Passing undef violates that semantic contract.
define i32 @unsafe_undef_to_noundef() {
entry:
  %r = call i32 @abi_requires_noundef(i32 undef)
  ret i32 %r
}
