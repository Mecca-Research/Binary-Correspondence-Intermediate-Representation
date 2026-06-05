# 06 — Transform dialect for BCIR lowering

The transform dialect can describe how to apply lowering strategy without making
that strategy part of runtime semantics. Use it for repeatable experiments and
for documenting expected pass order.

## What transform dialect should control

- selecting BCIR graph regions to lower through affine or vector staging;
- applying canonicalization before legality conversion;
- choosing tile or vector sizes for regular graph fragments;
- sequencing partial conversions before the final full conversion;
- running checks that all claim IDs and diagnostics were attached.

## What transform dialect should not become

Transform operations should not be required at runtime and should not be the only
place where semantic facts live. If a BCIR claim must survive, attach it to the
program IR or descriptor before transform handles are consumed.

## BCIR strategy sketch

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @lower_bcir(%root: !transform.any_op) {
    %graphs = transform.structured.match ops{["bcir.graph"]} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %graphs {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
}
```

Treat this as a strategy sketch. A repository that wants verifier-complete
transform dialect tests should pin the available MLIR version and registered
extensions.

## Review checklist

- Does the transform sequence leave the IR legal for the next conversion phase?
- Are transform-only decisions mirrored in attributes, descriptors, metadata, or
  pass options that the lowering patterns can observe?
- Is there a final full-conversion or verifier step outside the transform script?
