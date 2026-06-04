# Solution 038: MLIR diagnostic plan review

Dropping an unsupported HAM hint can be a remark only if the hint is explicitly
optional and correctness does not depend on it. The remark should name the policy
and target reason, for example: `remark: dropping optional HAM prefetch hint;
target has no configured prefetch lowering for policy=prefetch distance=2`.

Dropping an unsupported required register binding must be an error. Required
bindings are ABI or backend constraints, not optimization hints. A useful error
would say: `error: cannot lower required bcir.bind_register reg_class=fpr
preference=xmm0 for target generic; no inline-asm, intrinsic, calling-convention,
or backend lowering is configured`.

Runtime ABI declarations should be centralized or checked against one ABI table.
Emitting declarations independently at each call site risks signature drift. The
pass should error when an emitted declaration disagrees with the configured ABI
version, and the diagnostic should include the callee name, expected signature,
actual signature, and schema/ABI version.
