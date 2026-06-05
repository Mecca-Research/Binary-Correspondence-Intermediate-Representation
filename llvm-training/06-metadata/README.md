# Metadata: Debug Info, Profiles, and Optimization Hints

## Key takeaways

- Metadata annotates IR without changing core value types, but consumers may use it for debug info, optimization, and diagnostics.
- Debug locations should be preserved only when they still identify the transformed instruction honestly.
- Profile weights, loop metadata, and TBAA are contracts with optimizers; stale or fabricated metadata can mislead transforms.
- When lowering BCIR-like facts, keep semantic metadata small, named, and easy to verify through pass pipelines.

## Chapter dispatcher

| Need | Read |
| --- | --- |
| Metadata nodes, attachments, named metadata, and syntax basics | [`01-metadata-basics.md`](01-metadata-basics.md) |
| Debug info structure, locations, and preservation rules | [`02-debug-info.md`](02-debug-info.md) |
| Branch weights, loop hints, TBAA, and optimization metadata | [`03-profile-and-optimization-metadata.md`](03-profile-and-optimization-metadata.md) |

## Examples

Open the `examples/` directory in this chapter for standalone artifacts and small fixtures that accompany the lessons. Files ending in `.ll` are intended to assemble unless the lesson or filename says they are intentionally invalid.
