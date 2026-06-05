# Exception Handling in LLVM IR

Exception handling (EH) is where LLVM IR stops looking like simple structured
control flow. A throwing call has two successors, EH pads have strict placement
rules, and different target ABIs use different IR families.

This chapter teaches the two major shapes you will see in real modules:

- **Itanium-style EH** (`invoke`, `landingpad`, `resume`) used by many Unix-like
  C++ targets.
- **Windows EH funclets** (`catchswitch`, `catchpad`, `cleanuppad`, `catchret`,
  `cleanupret`) used by WinEH personalities and represented with token-valued
  pads.

## Key takeaways

- Use `invoke` instead of `call` when the IR must model an exceptional edge.
- A `landingpad` is tied to an `invoke` unwind destination and produces the
  exception package that `resume` can continue propagating.
- WinEH funclets are explicit subgraphs. `catchswitch` chooses handlers,
  `catchpad`/`cleanuppad` begin funclets, and `catchret`/`cleanupret` leave
  them.
- Calls inside a funclet normally carry a `"funclet"` operand bundle naming the
  active pad token so transforms and codegen keep ownership intact.
- EH rewrites are CFG rewrites: update `phi` nodes, preserve pad placement rules,
  and do not drop operand bundles while cloning calls.

## Lessons

1. [`01-eh-overview.md`](01-eh-overview.md) — terminology, personality
   functions, unwind edges, and why EH is target ABI shaped.
2. [`02-itanium-landingpad.md`](02-itanium-landingpad.md) — Itanium-style
   `invoke`, `landingpad`, catch/filter/cleanup clauses, and `resume`.
3. [`03-wineh-funclets.md`](03-wineh-funclets.md) — Windows EH funclet tokens,
   `catchswitch`, `catchpad`, `cleanuppad`, `catchret`, and `cleanupret`.
4. [`04-cleanups-and-resume.md`](04-cleanups-and-resume.md) — cleanup-only
   handlers, rethrowing with `resume`, and safe cleanup call placement.

## Examples

- [`examples/invoke-landingpad.ll`](examples/invoke-landingpad.ll) — minimal
  Itanium-style exceptional edge with a `landingpad`.
- [`examples/cleanup-resume.ll`](examples/cleanup-resume.ll) — cleanup work that
  resumes exception propagation.
- [`examples/catchswitch-funclet.ll`](examples/catchswitch-funclet.ll) — minimal
  WinEH catch funclet using a `"funclet"` operand bundle.

## Where this chapter fits

Read [`05-control-flow/README.md`](../05-control-flow/README.md) first so normal
terminator and CFG rules are familiar. Read
[`13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md)
for `token` and other special types, then
[`13-advanced-ir/07-operand-bundles.md`](../13-advanced-ir/07-operand-bundles.md)
for call-site operand bundles, including the `"funclet"` bundle used by WinEH.
