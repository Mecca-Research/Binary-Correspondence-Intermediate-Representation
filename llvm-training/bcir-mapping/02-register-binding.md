# Register and Resource Binding

Register binding maps symbolic BCIR operands to concrete runtime resources or
slots. LLVM IR should make this binding explicit as table lookups, struct field
loads, and ordinary pointer arithmetic rather than as implicit register names.

## BCIR-level meaning

- A BCIR read or write operand names a resource ID, virtual register, lane slot,
  or external storage object.
- Binding resolves that symbolic ID against a runtime table before execution.
- Claims carry up to four read resource IDs and four write resource IDs.
- The same binding model supports graph attributes, worklists, executable
  descriptors, and memory resource records.

## Likely LLVM IR representation

- Keep the claim pointer and registry/register table pointer as explicit
  function arguments.
- Load a resource ID from `%bcir.claim` or a compact training struct, then use it
  as a GEP index into a table of resource records.
- Load the resource base pointer or scalar slot from the selected table entry.
- Avoid representing hardware registers as LLVM SSA names unless the value is
  genuinely local to the function; ABI-visible binding belongs in memory or
  explicit call arguments.
- See [`examples/register-binding.ll`](examples/register-binding.ll) for a small
  copy from a bound read resource to a bound write resource.

## Relevant runtime ABI structs/functions

- [`%bcir.claim`](../../runtime/llvm/bcir_claim_schema.ll) includes fixed read
  and write resource-ID arrays.
- [`@bcir.claim.rd`](../../runtime/llvm/bcir_claim_accessors.ll) and
  [`@bcir.claim.wr`](../../runtime/llvm/bcir_claim_accessors.ll) are the canonical
  accessors for those arrays.
- [`%bcir.res`](../../runtime/llvm/bcir_registry_schema.ll) is the resource table
  record used after binding.
- [`%bcir.execctx`](../../runtime/llvm/bcir_claim_schema.ll) carries execution
  state that should stay separate from resource table contents.
- [`@bcir.registry.lookup`](../../runtime/llvm/bcir_gem_seed.ll) shows the table
  indexing pattern, while [`@bcir.gem.execute_claim`](../../runtime/llvm/bcir_gem_seed.ll)
  shows opcode-based dispatch after binding.
- Existing examples: [`runtime/llvm/bcir_master_reference_v2.ll`](../../runtime/llvm/bcir_master_reference_v2.ll),
  [`runtime/llvm/bcir_gem_seed.ll`](../../runtime/llvm/bcir_gem_seed.ll), and
  [`llvm-training/exercises/011-register-binding-pattern.prompt.md`](../exercises/011-register-binding-pattern.prompt.md).

## Verifier risks

- Resource table and claim struct layouts must be identical in every linked
  module or LLVM will reject duplicate named type redefinitions or silently leave
  incompatible literal structs in prose examples.
- GEP indexes must match the loaded integer width; explicitly `zext` or `sext`
  IDs to `i64` when indexing tables.
- Bounds checks that branch around a binding need valid PHI predecessor lists.
- Link multiple generated binding modules with declarations for shared helpers,
  not duplicate definitions.

## Optimization risks

- Optimizers may reorder non-atomic loads/stores through resource pointers unless
  aliasing, ordering, or phase/barrier semantics are represented.
- Overusing `volatile` to protect bindings does not create synchronization.
- If a resource ID is known constant, inlining and constant propagation may erase
  table-lookup shape; preserve diagnostics with metadata or side records when
  needed.
- Mismatched address spaces for GPU, MMIO, or CXL-like resources can make a
  binding look valid in generic tests but fail for target-specific lowering.

## Pitfall links

- [`02-phi-predecessor-mismatch.md`](../08-pitfalls/02-phi-predecessor-mismatch.md)
- [`04-duplicate-symbols.md`](../08-pitfalls/04-duplicate-symbols.md)
- [`05-type-schema-drift.md`](../08-pitfalls/05-type-schema-drift.md)
- [`10-volatile-is-not-atomic.md`](../08-pitfalls/10-volatile-is-not-atomic.md)
- [`11-address-space-confusion.md`](../08-pitfalls/11-address-space-confusion.md)
- [`12-vectorization-blocked-by-aliasing.md`](../08-pitfalls/12-vectorization-blocked-by-aliasing.md)
