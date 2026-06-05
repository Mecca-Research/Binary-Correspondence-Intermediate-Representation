# Constants: Literals, Aggregates, and Globals

## Key takeaways

- Constants are typed IR values; integer width, floating syntax, and aggregate shape must match the consuming operation exactly.
- String globals are arrays of bytes, usually referenced through opaque `ptr` plus explicit access or GEP source types.
- Global initializers are constant-only; local computation belongs in instructions inside a function body.
- Prefer explicit `null`, `zeroinitializer`, `undef`, `poison`, and `freeze` semantics over guesswork when repairing constants.

## Chapter dispatcher

| Need | Read |
| --- | --- |
| Integer constants, signedness interpretation, and width pitfalls | [`01-integer.md`](01-integer.md) |
| Floating-point constants and bit-pattern spelling | [`02-floating-point.md`](02-floating-point.md) |
| String constants, byte arrays, and GEP access | [`03-strings.md`](03-strings.md) |
| Global initializers versus local runtime values | [`04-global-vs-local.md`](04-global-vs-local.md) |

## Examples

Open the `examples/` directory in this chapter for standalone artifacts and small fixtures that accompany the lessons. Files ending in `.ll` are intended to assemble unless the lesson or filename says they are intentionally invalid.
