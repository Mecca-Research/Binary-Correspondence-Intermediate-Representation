# BCIR 0.3b draft — the freestanding-C23-driver compiler + the law catch-up

*Unreleased draft. The package remains version `0.2.0`; no `v0.3b` release tag is claimed.*

BCIR's release ladder gates a rung on **laws landed dual-rail**, not on feature checklists.
0.3b is the rung where the C frontend stops being a research rail and becomes a compiler a
driver project can actually run: C source in, a verified claim graph out. The current MLIR law
rail is R1–R23; the C frontend's explicitly scoped verifier remains R1–R18. The Python oracle and
C production rail must agree byte-for-byte wherever both speak.

## What this release is

**The law catch-up — R19/R20/R21 are first-class verifier laws.** Synchronous timing,
clock-domain crossing, and lifetime (use-after-free / double-free) verdicts moved from
advisory checkers to numbered laws with the full six-artifact treatment (LangRef entry,
oracle law, MLIR verifier clause, negative FileCheck case, C twin, generated status). The
generated status inventories **R1–R23** on the MLIR law rail. Python covers each applicable
oracle surface; the C frontend remains explicitly scoped to R1–R18 rather than claiming
false law-for-law symmetry beyond its subset.

**§5.14 Phase 2 complete — the law-bearing C semantics have MLIR representation.** Every C
construct the frontend lowers whose *meaning* is a law now exists on the law rail with its
verifier clause and negative cases:

- **volatile** load/store qualifiers (R3-adjacent MMIO ordering discipline),
- **atomic** RMW / compare-and-swap ops (§5.8 ordering legality),
- **indirect-call** callee signature + conservative effects (R18 over function pointers),
- **pointer extent-provenance** (the R7 dual rail),
- the **call-ABI contract** (`bcir.abi_contract`, R12 on the call ABI): every compile records
  the layout facts each function's frame was materialized from — target, per-parameter
  (name, size, align), the return slot — and the law checks the record against both the
  laid-out types and the normative five-target data-model matrix (LP64 x86-64 / aarch64 /
  riscv64, LLP64 x86-64-windows, ILP32 i386). A tampered contract is caught on both rails.

**The multi-file driver project through `bcir-cc`.** The cc-style C driver
(`runtime/c/bcir_cc.c`, no Python) now orchestrates a project the way the oracle driver
does: multiple files per invocation, per-file `--fallback` routing to the LLVM backend for
constructs outside the supported subset, the R21 lifetime policy (`--r21
advisory|fallback|reject`), and the per-project verdict line —

    project: CLEAN (N files)
    project: PARTIAL-FALLBACK (k/N routed to the LLVM backend)
    project: DIRTY (k/N failed[, f fell back])

— **byte-identical** to `bcir-cfront`, with the same exit-code discipline (a hard error `1`
dominates a fallback `2`). Parity is gated in `tools/c/check_runtime.sh` (`#project`).
Compile-database mode (`-p`, `compile_commands.json`) and `-M` dependency rules live on the
oracle driver.

**The closed no-Python loop.** `C → bcir_cpp → bcir_cfront → bcir_plan → bcir_hydrate →
bcir_exec`: preprocess, compile, verify, plan, hydrate to the binary StreamPack, and execute
deterministically — freestanding C end to end, fuzzed (libFuzzer + ASan/UBSan) and
byte-identity-gated against the oracle.

**Supporting surfaces** shipped along the way: the Clang-grade diagnostics engine, the
`--target` data-model matrix, module-scope effect/commutation analysis (`--emit-effects`),
derived linker flags (`--emit-link-flags`), the self-contained verified-C emit
(`--emit-c`, bounds-quarantine runtime included when a masked access needs it), the
naked-pointer policy documented user-facing, and the `c-runtime` CI tier.

## What this draft release is NOT (remaining breadth, stated plainly)

No tag has been created. Three breadth areas remain open and are tracked in
the master roadmap's Phase 3 section — none of them changes a law:

- **Linking** — `bcir-cc` emits verified C and derived link flags per unit; driving the
  emitted objects through a host linker into one runnable image is the next Phase 3 item.
- **Real fixtures** — Linux UAPI / CMSIS-style headers and PCIe/NVMe/ACPI register-map
  ingestion: the breadth that exercises Phase 2's volatile/atomic/ABI laws on real driver
  headers, beyond the synthetic and UART fixtures gated today.
- **Native-object artifacts** — `.o` emission stays behind the explicit gate
  (`BCIR_NATIVE_OBJECT_GATE.md`); the WASM byte-encoder and stackify JVM/CIL paths carry
  their own execution-validation status honestly (see `docs/STATUS.md`).

Measurement / real-silicon replan (§5.4) remains **deferred and host-side** until a
bare-metal rig with PMU + RAPL runs the runbook (`HARDWARE_VALIDATION.md`).

## Verifying the release candidate

```
python -m bcir.tests.run_all --tier c-runtime   # oracle + C byte-identity tiers
tools/c/check_runtime.sh                        # the C-rail parity/fuzz sweep (#project, #fallback, #r21policy ...)
tools/wsl/check_passes.sh                       # the MLIR law rail (R1-R23 negative cases)
```

The generated inventory (test counts, op counts, pass counts, law table) lives in
[`docs/STATUS.md`](STATUS.md); the ladder itself in
[`docs/BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md) §7.
