; adversarial-class: assemble-valid-semantically-risky
; Risk: `add nsw` may produce poison, and branching on the derived predicate is
; immediate undefined behavior. Assembly/verifier success is not a safety proof.

define i32 @poison_branch_valid_risk(i32 %x) {
entry:
  %next = add nsw i32 %x, 1
  %wrapped = icmp slt i32 %next, %x
  br i1 %wrapped, label %overflow_path, label %normal_path

overflow_path:
  ret i32 -1

normal_path:
  ret i32 %next
}
