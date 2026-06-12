; adversarial-class: assemble-valid-semantically-risky
; Risk: a transform that rebuilds this call but omits the deopt bundle changes
; deoptimization state even if the ordinary arguments and return value match.

declare i32 @bundle_target(i32)

define i32 @operand_bundle_loss_risk(i32 %x, i32 %state) {
entry:
  %result = call i32 @bundle_target(i32 %x) [ "deopt"(i32 %state) ]
  ret i32 %result
}
