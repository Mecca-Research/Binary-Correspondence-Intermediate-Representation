# Memory: Allocas, Loads, Stores, Globals, and Address Spaces

## Key takeaways

- Memory objects are reached through `ptr`, but operation types describe what is loaded, stored, or indexed.
- Use `alloca` for stack storage, globals for module-level storage, and explicit alignment when the ABI or transform needs it.
- `getelementptr` computes addresses using a source element type; it does not load memory by itself.
- Address spaces are part of pointer type identity, so do not silently cast or drop them during lowering.

## Chapter dispatcher

| Need | Read |
| --- | --- |
| Stack slots, lifetime, and promotion-friendly alloca patterns | [`01-alloca.md`](01-alloca.md) |
| Load/store syntax, alignment, volatility, and opaque-pointer access types | [`02-load-store.md`](02-load-store.md) |
| Global variables, linkage, initializers, and symbol visibility | [`03-global-variables.md`](03-global-variables.md) |
| Address spaces and cross-space pointer mistakes | [`04-address-spaces.md`](04-address-spaces.md) |

## Examples

Open the `examples/` directory in this chapter for standalone artifacts and small fixtures that accompany the lessons. Files ending in `.ll` are intended to assemble unless the lesson or filename says they are intentionally invalid.
