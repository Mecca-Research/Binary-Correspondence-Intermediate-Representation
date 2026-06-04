# Solution 027: Fast-math risk review

`fast` is a bundle of aggressive floating-point assumptions. It can permit
reassociation, ignore NaNs and infinities, disregard signed-zero distinctions,
and enable transformations that are invalid for strict IEEE behavior. The more
specific flags are narrower: `reassoc` permits regrouping operations, `nnan`
promises no NaN values are relevant, and `ninf` promises infinities are not
relevant.

The candidate should not blindly use `fast`. If the original BCIR or MLIR op has
strict arithmetic semantics, `%sum = fadd fast` is an over-promise. Even the
narrower `reassoc nnan ninf` on the multiply must be justified by a source-level
contract, a dialect attribute, or a documented runtime invariant. A safe lowering
should preserve strict operations by default and add only proven flags, with a
comment or metadata trail identifying the source of the relaxed math contract.
