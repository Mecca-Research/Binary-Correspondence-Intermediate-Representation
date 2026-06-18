# BCIR Repository Current State Audit

> Audited 2026-06-16 against the `bcir/` (oracle) + `mlir/` (law) tree, after the
> **full deterministic optimizer-core C++ port**, the six-target capability matrix, the
> C23 `_BitInt`/`#embed` kernels + ETL-binary C decoder, and the native-object decision
> gate. The normative status lives in [`BCIR_LANGREF.md`](BCIR_LANGREF.md) §16; the
> forward plan in [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md) (the single,
> consolidated roadmap); this file is the honest snapshot + changelog. Earlier revisions
> described the retired C++ `ir/` skeleton (removed 2026-06-07) and the pre-Phase-13 tree
> (audited 2026-06-12).

## Snapshot

- Two trees implement BCIR in lockstep ([`PARITY.md`](PARITY.md)):
  - **`bcir/`** — the executable conformance oracle (pure Python, no third-party
    deps), ~11.8K LOC: model, K_BCIR optimizer (min-plus + RCSP/Pareto +
    (max,+) overlap), GEM hydration/scheduling/execution, ROP/MAP front-ends, M5
    ETL, telemetry/calibration, StreamPack ABI, the R1–R18 verifier, lowering
    (clang AOT / lli JIT / WASM / stackify / per-target llc / **portable C23
    kernel**), **and** the
    Phase 13–26 organs: calibration (microbench + Bayesian/conformal), policy
    portfolio + replay gate, MoE gate, search accelerator, soft optimizer, regret
    ledger, provenance manifest, e-graph + memory-module fixpoints, the two-truth
    quarantine, modular mapping functions, the enriched-operad memory
    interface, and the closed calibration loop (`calibloop`: measure → freeze →
    replan → certified win). Suite: `python -m bcir.tests.run_all` (live count + coverage
    in [`STATUS.md`](STATUS.md), generated from the tree — see that file rather than a
    hard-coded number here, which is what the 580/615/631 drift came from).
  - **`mlir/`** — the law: the ODS/TableGen dialect family (op count in [`STATUS.md`](STATUS.md)),
    the compiled `bcir-opt` with `-bcir-verify` (R1–R18), `-bcir-promote-lanes`,
    `-convert-bcir-to-llvm`, the **GEM pipeline passes** (`-bcir-classify-lanes
    / -select-realization / -batch / -schedule / -lower-to-llvm`), the **full
    deterministic optimizer core in C++23** (`-bcir-cost-model` cost+fusion/CSE →
    `-bcir-plan` coupled min-plus → `-bcir-overlap` (max,+) → `-bcir-rcsp` /
    `-bcir-rcsp-plan` constrained search), a `bcir.kbcir.theta` context op, named pass
    pipelines (`bcir-audit`/`-optimize`/`-hydrate`/`-lower-llvm`/`-aot`), and the IRDL
    projection for stock `mlir-opt`. Validated in CI on the latest LLVM/MLIR release —
    LLVM 22, gating (`mlir-rail-validate`).
- **`runtime/c/`** — the freestanding (no-libc) C runtime: the StreamPack ABI v1 decoder
  (Python-encode ↔ C-decode parity + libFuzzer), the ETL binary-record decoder
  (`bcir_binrec.c`, parity + libFuzzer), the C23 `#embed` frozen Q8 tier table, and the
  microbench. The C23 kernel emitter adds exact-width `_BitInt(N)` Q-fixed lanes.

## Confirmed strengths

1. The oracle runs the whole correspondence chain end to end, deterministically
   (integer/Q-fixed), with worked-example parity pinned (`vector_add` AVX-512
   cool Θ → vec16, score **7808**; under a 700 thermal/power cap → vec8, **9472**).
2. Verifier laws **R1–R13** run on both rails and are negative-tested per law:
   `bcir/verify` and the MLIR `-bcir-verify` pass.
3. The **GEM pipeline is now MLIR-native and cross-checked against the oracle**:
   `-bcir-select-realization` recomputes the min-plus score from `cost · weights`
   and reproduces 7808/9472; the other four stages classify/batch/schedule/lower
   with positive (FileCheck) and negative (`-verify-diagnostics`) tests.
4. **Hot/cold separation is verified and locked** (`bcir/tests/test_hot_cold.py`):
   the executor and ABI codec import no learned organ or planner; no
   planning→execution→telemetry runtime recursion.
5. The StreamPack ABI v1 is frozen, CRC-gated, and decoded by a freestanding C
   runtime; cross-language parity is CI-gated.
6. CI gates every push: the oracle suite, the C runtime, the LLVM-training
   validators, and the full MLIR rail (tblgen, IRDL round-trip, `bcir-opt` build,
   ODS corpus, pass tests including the GEM pipeline).
7. **First measured win on real silicon.** The evidence rail (`bcir.bench`) shows
   BCIR's gather-avoidance (picking the direct realization over GGG) is **~6–7×
   faster** than the gather form (random indices) — the bare-metal-calibrated
   `gather_penalty` realized. And budget feasibility (`rcsp.feasible`,
   `api.build_artifact(budget=…)`) is a **correctness** win: BCIR emits the
   feasible vec8 where the naive max-width vec16 violates a 700 thermal/power cap.
   The library façade (`bcir.api`) packages a plan as a deployable, R12-attested
   artifact (AOT or driver-embedded).

## Confirmed limitations

1. **No BCIR-native instruction selection** (by design — emit C/LLVM and reuse the
   resident backend; the explicit decision gate is `BCIR_NATIVE_OBJECT_GATE.md`). A
   portable **C23 kernel backend** (`lower.c_kernel`) emits restrict-qualified,
   bounds-safe, width-driven C (now with exact-width `_BitInt(N)` Q-fixed lanes) from
   the selected StreamPack; LLVM/llc/lli/wasm are the other machine-code paths. **One
   target end-to-end is closed** (`codegen.codegen_object_c`: emitted C → resident clang
   → real eBPF / x86-64 ELF object, ELF-verified) — without hand-rolled isel. Register
   allocation/linking stay the resident toolchain's job; `bcir.target.lower_contract` is
   the seam. **Remaining:** GPU-C dialect gather variants.
2. **The calibration loop is fully closed.** Bare-metal cost constants
   (`runtime/c/bcir_microbench.c`, `calibrate_native`, `--calibrate --native`,
   real cache latency: gather_penalty ≈ 5), R13-certified replan (`kbcir.calibloop`),
   **a trained calibrator** (`calibrate.train_calibrator` → `FrozenCalibrator`, an
   online model frozen to deterministic Q8) and **a live broker**
   (`telemetry.Broker`, pub/sub fan-out). The frozen calibrator drives the loop end
   to end. *(Production hardening — a real Kafka deployment — remains operational.)*
3. **The example corpus is real, not toy.** Beyond vector_add: `saxpy_strided`
   (strided gather-avoidance, ~1.4× measured), `gather_reduce` (reduction
   gather-avoidance, ~16×), `fused_chain` (multi-claim overlap + the fusion
   discount), `scan_chain` (a dependency chain that serializes), **and the widened
   `examples.CORPUS`** — `matmul_tiled` (real blocked matmul, register-resident
   K-accumulation → deforestation), `scan` (multi-stage prefix pipeline),
   `multi_histogram` (map/reduce multi-claim gather). Python↔MLIR parity is now
   **generated and adversarial** (`bcir.kbcir.differential`), with per-target parity
   across the six TARGETS for the whole corpus (`mlir/test/passes/gem_corpus.mlir`).
   Joint multi-claim *bundle* optimization now ships (`kbcir.bundle`, a 12% matmul gain).
4. **Intelligence ahead of substrate.** Phases 13–26 added a rich learned/
   categorical optimization stack over a backend that cannot yet codegen and
   tables that are not yet measured; the ROI is unproven until §1–2 close.
5. **LLVM rail tracks the latest release (LLVM 22, gating).** The
   `mlir-rail-validate` CI job builds + validates `bcir-opt` against **LLVM/MLIR 22**
   (the latest release), installed from `apt.llvm.org` since Ubuntu's default repos
   top out near 18/19. The toolchain is resolved version-agnostically (the highest
   `/usr/lib/llvm-*` for cmake; `FileCheck` / `mlir-opt` / `mlir-tblgen` the same way),
   so a major bump is a one-line matrix change. The Symbol-container ops (`registry` /
   `kbcir.plan` / `gem.stream_pack` / `parse.grammar` / `fsm.machine` / `binary.format`)
   carry the `SymbolTable` trait, satisfying the "symbol's parent must have the
   SymbolTable trait" verifier that has only tightened since LLVM 19.

## Recommended next milestones (see [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md) §5–6 for detail)

The deterministic optimizer core is **fully ported to C++/MLIR**; the six-target
capability matrix, C23 `_BitInt`/`#embed`, the ETL-binary C decoder fuzz, the
native-object gate, the **C StreamPack executor** (`runtime/c/bcir_exec.c`), and
**R14/R15/R16 as first-class `-bcir-verify` laws** all landed. The next build steps:

1. **The C StreamPack encoder** (`runtime/c/`, mirroring `streampack_abi.encode`,
   CRC-gated) — completes a full C round-trip, and the **`verification` cost axis** a
   real producer (today modeled as 0).
2. **Measured real-silicon calibration** — the software path (`kbcir.calibloop`,
   `bcir.silicon`) is closed + certified on host; the one deferred result is a *measured*
   (not synthetic) replan win on a bare-metal rig (the rig spec is in
   `HARDWARE_VALIDATION.md`). *The single most valuable next result.*
3. **Multi-claim bundle (joint) optimization** — the first genuinely new optimizer
   capability, where the combinatorial (min,+) formulation earns its keep beyond the
   pairwise coupling that ships today.

## Changelog

- 2026-06-18: **C compiler (Phase 2) — `do/while` + `break`.** Two more control-flow forms,
  dual-rail, reusing the loop markers (no new claim ops). `do body while(cond);` puts the cond test
  at the loop-body *bottom* (body runs at least once): the oracle adds a `WhileNode.test_at_end` flag
  (the flatten/region walks treat it identically; only the emitter branches), the C frontend emits
  `c.loop → body → cond → c.loop.test → c.endloop`. `break;` is a `BreakNode` (oracle) / `c.break`
  NOP marker (C) rendered as `break;` -- correct in every loop form (it exits the `while(1)`). New
  fixture `cfront_dowhile.c` (a bounded do/while sum + a counter, both with an early `break`) matches
  on both rails (`funcs=2 claims=17 const=6 binop=6 ok=1`) and is Clang-behaviour-equivalent. Wired
  into `_CONTROL` + `tools/c/check_runtime.sh`; warning-clean; 693 tests pass. (`continue` is
  deferred -- in a `for` it must run the step, which the `while(1)` desugar would skip.)
- 2026-06-18: **C compiler (Phase 2) — `for` loops.** The next control-flow construct, dual-rail,
  desugared onto the existing `while` machinery: `for(init; cond; step) body` ≡
  `init; while(cond){ body; step }` (exact while there is no `break`/`continue`). The oracle adds a
  `cast.For` node lowered to a `WhileNode` with the step appended to the body block; the C frontend
  inlines it in `p_stmt` with a token-rewind that re-lowers the step (recorded between the 2nd `;`
  and `)`) at the loop-body end (a new `p_simple` parses the `name = expr` step with no trailing
  `;`). No new IR/lowering/emit ops -- it reuses the `c.loop`/`c.loop.test`/`c.endloop` markers, so
  both rails agree on the structural summary. New fixture `cfront_for.c` (a masked-count sum + a
  polynomial) matches on both rails (`funcs=2 claims=14 const=4 binop=5 ok=1`) and is
  Clang-behaviour-equivalent. Wired into `_CONTROL` + `tools/c/check_runtime.sh`; warning-clean;
  693 tests pass.
- 2026-06-18: **C compiler (Phase 2) — the ternary operator `?:`.** The first Phase 2 language gap,
  dual-rail. The oracle lexer gained `?` in its punct set (the "not yet lexed" gap; the C lexer's
  single-char fallback already tokenized it); both parsers layer a conditional expression
  `cond ? then : els` (right-associative) over the binary grammar (oracle `_ternary` between
  `_assign` and `_binary`; C `p_expr` over `p_binexpr`). It lowers to a scalar `c.select` claim
  (both arms evaluated -- the straight-line subset has no branches -- then chosen), and the emitter
  renders the real C `(cond ? a : b)`, so it is **behaviour-exact under Clang** for the pure scalar
  arms a driver uses (no side effects to double-run). New fixture `cfront_ternary.c` (clamp + a
  nested right-associative pick) matches on both rails (`funcs=1 claims=13 const=1 binop=5 ok=1`,
  4 selects), is Clang-equivalent, and runs the execute loop (R9/R10-R11 clean). Wired into
  `_STRAIGHTLINE` + `tools/c/check_runtime.sh`; warning-clean. 693 tests pass.
- 2026-06-18: **C compiler — `bcir-cc`, the production compiler driver (Phase 1) + the C
  preprocessor's `-I`/`-D` extension.** A cc-like driver binary (`runtime/c/bcir_cc.c`) over the
  full C rail — `bcir_cpp_run_ex` (preprocess) → `bcir_cfront` (lower + R1-R18 verify + C.2 attest)
  → `bcir_plan` → `bcir_hydrate` — so a driver with sibling headers builds via a normal compile
  command, no Python and no test-harness include map: `bcir-cc runtime/c/cfront_driver_uart.c` →
  `funcs=3 claims=22 … ok=1`. Flags: `-I` (multi-dir) / `-D name[=val]` / `-U` / `-std=` / `-E` /
  `-o`, and artifact emission `--emit-c` (verified C + the C.2 attestation) / `--emit-claimgraph`
  (summary + per-function claim dump) / `--emit-pack` (the entry's hydrated StreamPack, `BSPK`
  magic). To feed it, the **C preprocessor gained the dual-rail twin of the oracle's
  search-path/defines** (PR #276): `bcir_cpp_run_ex(dirs[], defines[])` resolves `#include`/`#embed`
  across multiple `-I` dirs (the source dir first) + seeds `-D` predefines, and the include
  recursion was refactored so **macros persist across nested includes** (matching `cpp.py`; the old
  per-include `NM=0` reset was a latent divergence). The existing `#include` fixtures lower
  identically (no regression); warning-clean (C11 + C23). Gated by
  `test_bcir_cc_driver_compiles_and_emits_artifacts` (sibling + `-I`/`-D` compile, artifacts, oracle
  parity on the `-D`-selected branch) + a `bcir-cc` section in `tools/c/check_runtime.sh`. 693 tests
  pass.
- 2026-06-18: **Plug-in C compiler — the road to a C23 replacement compiler (§5.9) + the CLI
  include-path fix (Phase 1).** Recorded the four-phase plan to take the frontend from a
  driver-subset to a full C23 *replacement* compiler (roadmap §5.9): Phase 1 productionizes the
  existing subset as a `cc`-like driver, Phase 2 is the freestanding embedded/kernel C23 subset,
  Phase 3 a hosted candidate, Phase 4 a general distribution — each landed in PR-sized chunks. The
  first chunk fixes the **CLI include-path gap**: `python -m bcir.frontends.cfront file.c` used to
  call `compile_unit(f.read())` with no include context, so a driver with a sibling
  `#include "uart_regs.h"` failed from the CLI even though the tests + `check_runtime.sh` supplied
  the map (a real productization gap, not a compiler-core bug). The preprocessor now resolves headers
  from **disk via a search path** (`Preprocessor(search_paths=…, defines=…)`; `_resolve` checks the
  in-memory mount, then the `-I` dirs / source dir), and the CLI is a `cc`-compatible driver:
  `-I` / `-D` / `-U` / `-std=` / `-E` / `-o`, with the source file's own directory always on the
  search path. Gated by `test_cli_resolves_sibling_and_search_path_headers` (sibling + `-I` headers
  compile, `-E` preprocesses, a missing header is a clean diagnostic + non-zero exit, `-D` seeds a
  `#if`). 692 tests pass.
- 2026-06-18: **Phase D (the write + control-flow half) — a UART driver end-to-end.** A second,
  complementary register-map driver (`runtime/c/uart_regs.h` + `cfront_driver_uart.c`) closes the
  patterns the DMA driver does not exercise: MMIO register **writes** (`u->BRR/CR/DR =`, ordered
  volatile stores) and a **bounded status-poll loop** (L6 `while`+`if` re-reading `SR` each
  iteration) — the bread-and-butter of real device drivers — alongside the same typedef/enum/union/
  bitfield ABI and `#include`+object-macro preprocessing. Both rails agree on `funcs=3 claims=22
  mmio=0 bf=4 const=4 binop=6 call=0 ok=1`; every function (incl. the MMIO + control-flow
  `uart_send`) is Clang-behaviour-equivalent; the straight-line entry `uart_configure` plans /
  hydrates / executes R9/R10-R11 clean with the loop's provenance digest matching the C.2
  attestation. Gated by `test_phase_d_uart_driver_write_and_poll_path` + `tools/c/check_runtime.sh`.
- 2026-06-18: **Phase D — a real register-map header driven end-to-end through the plug-in C
  compiler.** A vendor-style memory-mapped DMA-channel register map (`runtime/c/cfront_driver.h`) +
  driver (`cfront_driver.c`) is ingested with **no hand-written claim graph**, exercising the whole
  accumulated stack at once: `#include` + field-extract macros (L7), `typedef volatile`/enum/union/
  bitfields (the type model), volatile MMIO loads (L5), struct pointers (L3), and a 3-function call
  graph (L4 / R18). The full loop runs in C — `C -> bcir_cpp -> bcir_cfront -> bcir_plan ->
  bcir_hydrate -> bcir_exec` — both rails agreeing on `funcs=3 claims=30 mmio=1 bf=3 const=5
  binop=10 call=2 ok=1`, R9/R10-R11 clean, and the emitted verified-C (volatile MMIO load + bitfield
  decode + the call graph) is **Clang behaviour-equivalent** on a seeded register block
  (`test_phase_d_real_header_driver_end_to_end`; gated in `tools/c/check_runtime.sh`). **Evidence
  surfaced:** the frontend does not yet lex the **ternary operator** `?:` (worked around with a
  `0/1`-comparison multiply); the next type-model item remains **multi-dimensional arrays**. This is
  the demonstration the L1-L8 + verifier/type/atomics/channel work was built toward: a real header,
  compiled and verified, with no Python.
- 2026-06-18: **Plug-in C compiler — §5.8 follow-ons: `cmpxchg`, type-model breadth (typedef /
  enum / union), and the multi-channel lowering decision in C.** Three dual-rail, parity-gated
  increments on the C compiler. **`cmpxchg`:** `__sync_val/bool_compare_and_swap` → the `CMPXCHG`
  opcode (a 3-read claim: ptr, expected, desired) on lane A, emitted back as the matching `__sync`
  builtin, behaviour-equivalent under Clang on independent copies of the same seeded cell
  (`cfront_cmpxchg.c`). **Type-model breadth:** `typedef` (scalar/pointer/aggregate aliases incl.
  `typedef struct {...} N;`, resolved at parse time), `enum` (enumerators folded to their integer
  values), and full `union` layout (members overlap at offset 0, size = the widest member; the C
  rail previously laid unions out as structs — a latent bug — now tracked via `sdef`/`bcir_ctype.
  is_union` and emitted as `union tag`). New fixtures `cfront_{typedef,enum,union}.c` match the
  oracle's structural summary on both rails and are Clang-behaviour-equivalent; typedef + enum also
  run the full compile→execute loop. **Multi-channel lowering decision** (`runtime/c/bcir_channel.
  {h,c}`): the C twin of `bcir/channels`' routing seam — a `channel.json` reader + `bcir_claim_
  required_caps` / `bcir_channel_suits` / `bcir_channel_route` (the cost-free plan-time backend
  pick, mirroring the new `bcir/channels.route_claim`), so a driver routes a claim to its backend
  with no Python. Python↔C parity-gated (`test_c_channel.py`; `channels/example_{cpu,tpu,pim}.
  channel.json` exercise the plugin/universal/legacy paths). All warning-clean (C11 + C23); gated in
  `tools/c/check_runtime.sh`. Remaining type-model breadth (multi-dimensional arrays, function
  pointers) tracked in roadmap §5.8.
- 2026-06-18: **Plug-in C compiler — C-side infra (§5.8): the R1–R18 verifier, atomics/fences, and
  the C.2 attestation in C.** A new `runtime/c/bcir_verify.c` (the C twin of `bcir/verify`) lands the
  runnable LangRef laws over the claim graph + plan + StreamPack: R1–R8 (incl. **R6** lane↔stride
  legality), **R9** plan legality (`bcir_verify_plan`), **R10–R11** StreamPack well-formedness
  (`bcir_verify_pack`, validating magic/version/CRC + the segment count), **R12** lowering-contract,
  **R13** provenance digest (`bcir_provenance_digest`, FNV-1a over the claim graph), R14–R16 vacuous
  for the scalar subset, **R17** accuracy, **R18** call-graph integrity. `bcir_cfront` now verifies via
  `bcir_verify_unit` and the loop (`test_cfront_loop.c`) checks the R9/R10–R11 verdicts. **Atomics/
  fences:** `__atomic_fetch_add/sub/xor` → `ATOMIC_ADD/SUB/XOR` and `__atomic_thread_fence`/
  `__sync_synchronize` → `BARRIER`, lowered on **lane A** — the lane law (R6) is widened so lane A is
  legal for a SCALAR atomic counter (a single-location RMW, not on the decoupled GGG/scatter tail) as
  well as a RANDOM scatter-atomic, and the atomic/barriered hazard discharges R5. They emit back as the
  matching seq-cst builtins; the new fixture `cfront_atomic.c` matches the oracle's structural summary
  (`claims=9 const=3 binop=1 ok=1`) and is **behaviour-equivalent under Clang** on independent copies
  of the same seeded counter (a side-effect-aware harness, since atomics mutate their pointee). **C.2
  attestation:** the emitted verified-C carries a header naming the discharged laws + the **R13
  provenance digest** — the *same* digest the compile→execute loop reports (reproducible across the two
  C entry points). Gated by `bcir/tests/test_c_cfront.py` (the §5.8 dual-rail test + the C.2 digest
  test) and `tools/c/check_runtime.sh` (atomics in the parity + execute loops). Warning-clean
  (C11 + C23). *Still to port in §5.8:* `cmpxchg`, dynamic shapes, multi-channel lowering.
- 2026-06-18: **Plug-in C compiler — L8 ABI in C → the full L1–L8 C ladder is ported.** Struct
  return-by-value (a function returning a struct: a struct local built via member stores, then
  `return r`), struct member stores (`r.field = expr` → `c.store` → `memcpy` at the field offset),
  `__attribute__((packed))` / `aligned(N)` / `alignas` struct layout (no-padding packed; forced
  alignment), and a field-size-correct member load (`memcpy` the field's exact byte width into a
  zeroed temp — fixes packed `uint8_t`/`uint16_t` reads). Fixtures `cfront_structret.c` (vec2 swap,
  struct param + struct return) and `cfront_packed.c` (a packed wire header) match the oracle's
  structural summary exactly and are Clang-behaviour-equivalent; the C frontend's packed offsets are
  cross-checked against Clang's `sizeof`/`offsetof` (`size=7 addr@1 len@5`). With L8, the **entire C
  frontend ladder (L1–L8) now lives in `runtime/c/`** — Python↔C parity + six-artifact gated across
  all nine fixtures (`tools/c/check_runtime.sh`). The remaining `runtime/c/` ports are infra, not
  ladder stages: the verifier R9–R17, C.2 attestation, atomics/fences, dynamic shapes, multi-channel
  lowering (§5.8). Warning-clean (C11 + C23).
- 2026-06-18: **Plug-in C compiler — L7 a real C preprocessor in C (`bcir_cpp.c`).** A C
  preprocessor that runs before `bcir_cfront`'s lexer: object- and function-like `#define` macros
  (with `#` stringize + `##` paste + expand-until-stable rescanning), `#undef`, conditional
  compilation (`#if`/`#ifdef`/`#ifndef`/`#elif`/`#elifdef`/`#elifndef`/`#else`/`#endif` with an
  integer constant-expression evaluator + `defined`), `#include` of project headers (resolved
  relative to the source's directory), and **C23 `#embed`** (→ a byte list). So a real vendor
  register-map header — with its `#define REG_BASE 0x4000_0000` macros — now ingests through the C
  compiler. The harnesses preprocess then compile; fixtures `cfront_macros.c` (macros + `#if`) and
  `cfront_ppinc.c` (`#include cfront_ppdefs.h`) match the oracle's structural summary exactly
  (`claims=17 const=8 binop=9`; `claims=4 const=2 binop=2`) and are Clang-behaviour-equivalent.
  `bcir/tests/test_c_cfront.py` adds a preprocessor unit test (macros / `#elifndef` / `#embed`);
  `tools/c/check_runtime.sh` parity now spans all L1–L7 fixtures. Warning-clean (C11 + C23). Remaining
  C port: L8 full ABI (§5.8).
- 2026-06-18: **Plug-in C compiler — L6 control flow in C (if/else/while).** Ported the L6 ladder
  stage to `runtime/c/bcir_cfront.c`: `if`/`else` and bounded `while` lower through control-flow
  *marker claims* (NOP-opcode `c.if`/`c.else`/`c.endif`/`c.loop`/`c.loop.test`/`c.endloop`/`c.return`)
  that the faithful emitter renders as real C braces (with mutable named locals, so branch merges +
  loop accumulators reproduce the source), while the realizable claims stay a flat list for
  verify/plan/hydrate (markers are emit-only — excluded from the summary + skipped by the hydrator).
  Early returns inside branches are preserved. Fixtures `cfront_branch.c` (clamp) + `cfront_while.c`
  (weighted sum) match the oracle's structural summary exactly and are Clang-behaviour-equivalent
  (`test_c_cfront.py`); the parity loop in `check_runtime.sh` covers all L1–L6 fixtures. Remaining C
  ports: L7 preprocessor, L8 full ABI (§5.8).
- 2026-06-18: **Plug-in C compiler — the C compile→execute loop closes (planner + StreamPack
  hydrator), no Python.** Added `runtime/c/bcir_plan.c` (a compact freestanding K_BCIR planner:
  per-claim realization width + an integer cost + the plan total) and `runtime/c/bcir_hydrate.c` (the
  `gem.hydrate` twin: serializes the planned claim graph into the frozen StreamPack ABI, one segment
  per claim, freestanding + bounds-checked). The whole loop now runs in C with no Python —
  `C source → bcir_cfront → bcir_plan → bcir_hydrate → bcir_exec` — `test_cfront_loop.c` compiles a C
  file, plans it, hydrates a StreamPack the existing bounds-checked decoder validates, and executes
  every claim in deterministic lowering order (claims=23 plan_cost=55 pack_bytes=1419 executed=23 for
  the register map). Gated by `bcir/tests/test_c_cfront.py` (executed count == oracle claim count) and
  a new `tools/c/check_runtime.sh` section. `bcir_plan.c`/`bcir_hydrate.c` are freestanding-clean
  (C11 + C23). Roadmap §5.8 marks the planner + hydrator landed (the two biggest infra gaps); the
  remaining C ports are L6/L7/L8, R9–R17, C.2 attestation, atomics, dynamic shapes, multi-channel.
- 2026-06-18: **Plug-in C compiler — L3/L4 ports + a faithful C emitter + the full six-artifact gate
  in C.** Extended `runtime/c/bcir_cfront.c` to **multi-function translation units**: **L3**
  pointers/arrays (GEP-equivalent `base[i]` loads), **L4** functions + a call graph with an **R18**
  check in C (rejects recursion + undefined callees), on top of the existing L1/L2/L5. Replaced the
  trace emitter with a **faithful C emitter** (params, scalar arithmetic, member/MMIO/bitfield loads
  via `memcpy`/`volatile`, calls, returns) so the C frontend's output is now **Clang
  behaviour-equivalent** to the source — `bcir/tests/test_c_cfront.py` builds the emitted C beside the
  original and diffs them on seeded-random inputs. Each shared fixture (`cfront_regmap.c`,
  `cfront_array.c`, `cfront_callgraph.c`) now passes the full six-artifact gate on the C rail: claim
  graph (Python↔C parity), R1–R8 + R18 verifier (`ok=1`), faithful C output, and Clang
  behaviour-equivalence. `tools/c/check_runtime.sh` runs the parity for all three. Warning-clean
  (`-Wall -Wextra -Werror`, C11 + C23). The roadmap adds **§5.8** — the researched list of components
  still missing from `runtime/c/` to close `C → claim graph → K_BCIR plan → verified C → Clang check`:
  the verifier R9–R17, a C planner (or MLIR bridge — the biggest gap), a claim-graph→StreamPack
  hydrator (to feed `bcir_exec.c`), C.2 attestation, atomics/fences, dynamic shapes, multi-channel
  lowering, and type-model breadth (typedef/enum/unions).
- 2026-06-18: **Plug-in C compiler — porting the C frontend from the oracle prototype to production
  C (`runtime/c/`).** Correction of course: the L1–L8 ladder built in `bcir/frontends/cfront/` was the
  **oracle prototype**; per the dual-rail discipline (prototype → STOP → port to the production rail,
  exactly as the optimizer core went oracle→MLIR and GEM hydrate/execute went oracle→C), the real
  plug-in C compiler must live in `runtime/c/`. Added: `runtime/c/bcir_cir.h` (the freestanding BCIR
  claim-graph IR — resources/claims/phases, enum values matching `bcir/model`), and
  `runtime/c/bcir_cfront.c` (the host compiler tool, the C twin of `bcir/frontends/cfront/`): a
  recursive-descent C lexer/parser + struct/bitfield layout + lowering for the **register-map slice**
  (fixed-width integer expressions, struct member access, bitfields, volatile/MMIO register reads) →
  the claim graph, with an R1–R8 verifier and an emitter, all in C and warning-clean (`-Wall -Wextra
  -Werror`, C11 + C23). It is **Python↔C parity-gated**: both rails lower the same register-map
  fixture (`runtime/c/cfront_regmap.c`) to the identical structural summary `claims=23 mmio=1 bf=3
  const=4 binop=8 ok=1` (`bcir/tests/test_c_cfront.py` + a new section in `tools/c/check_runtime.sh`,
  alongside the existing decoder/executor/encoder twins). The roadmap §3 now states the
  prototype-then-port discipline explicitly and tracks the C-frontend port (◑) in the placement map.
  Next: port arrays/calls, control flow, the preprocessor, and full ABI to C, stage by stage.
- 2026-06-18: **C frontend L7 (preprocessor) + L8 (ABI) — the C ladder is complete.** L7: a real C
  preprocessor (`cpp.py`) runs before the parser — object/function `#define` macros (with `#`
  stringize + `##` paste, nested expansion with a hideset), `#undef`, conditional compilation
  (`#if`/`#ifdef`/`#ifndef`/`#elif`/`#elifdef`/`#elifndef`/`#else`/`#endif`) with an integer
  constant-expression evaluator (`defined`, `__has_include`, `__has_embed`), `#include` of project
  headers (unmapped `<system>` headers are intrinsic no-ops), and **C23 `#embed`** (expands to a byte
  list, usable via new file-scope `const` array globals that lower to read-only resources). The
  preprocessed text feeds both the parser and the Clang harness, so equivalence validates the
  *preprocessed* program. L8: **struct return-by-value** (the harness diffs returns via `memcpy`),
  `__attribute__((packed))` / `aligned(N)` / `alignas` layout, and member access switched to
  `memcpy` (alignment-safe / packed-safe; Clang folds it to a load) — the frontend's `sizeof`/field
  offsets are cross-checked against Clang's ABI in the tests. A vendor register-map header now
  ingests end to end (L7 → L5 → an R1–R18-clean plan + `bcir-explain`, behaviour-equivalent to
  Clang). Fixtures L7_macros/L7_include/L7_embed + L8_struct_return/L8_packed added; suite: live count
  in [`STATUS.md`](STATUS.md). Phase C effectively complete; next is Phase D (the first real driver).
- 2026-06-18: **C frontend L5–L6 (the register-map/MMIO MVP) + the C.2 generalized backend.**
  L5: a `volatile`-qualified register-map struct (accessed through a `volatile struct *`) lowers to
  `Domain.MMIO` resources with ordered (`barriered`) load/store hazards — R3 (MMIO domain + write
  hazard) and R5 clean — and bitfields lower to explicit LSB-first mask/shift extract + read-modify-
  write claims (`c.bf.get`/`c.bf.set`), Clang-layout-compatible. L6: control flow — `if`/`else`
  lowers through a structured body tree to real emitted C (and `compose.Cond` for the cost model),
  bounded `while` loops to mutable named-local accumulators; the lowering switched from straight-line
  SSA to a body tree + mutable storage so branch merges and loop accumulators reproduce the source.
  C.2: every emitted function now carries a **verified-C attestation** (R1–R9 module/plan status,
  R12 lowering contract — attested by Clang behaviour-equivalence, an R13-style claim-graph provenance
  digest, R17 integer-exactness, R18 call-graph integrity, the plan score), and `emit_selfcheck`
  produces a standalone self-checking C program (`--selfcheck`). The register-map/MMIO success
  criterion is met: a device-register C file lowers to an R1–R18-clean plan with `bcir-explain` and is
  byte-for-byte behaviour-equivalent to Clang. Fixtures L5/L6 added; suite: live count in
  [`STATUS.md`](STATUS.md). Next: L7 (preprocessor subset) + L8 (full ABI).
- 2026-06-18: **C frontend MVP — ladder stages L1–L4 (the keystone's first milestone).** Built
  `bcir/frontends/cfront/`: a recursive-descent C lexer + parser for the driver/kernel-relevant
  subset, a type model with Clang-compatible struct/union layout, and a lowering to the **same**
  claim-graph model the oracle reasons over (`Resource`/`Claim`/`Phase`) — the dual-rail invariant, so
  R1–R18 + the K_BCIR cost model apply unchanged. `compile_unit` runs the full six-artifact gate per
  function: parse → claim graph → K_BCIR plan → emitted C (an *arbitrary* scalar claim-graph C
  emitter, not a per-pattern kernel) → R1–R18 checkpoint (R18 via the real `plan_composite`
  call-graph machinery: recursion + undefined-callee rejected) → a seeded-random Clang
  behaviour-equivalence harness (toolchain-gated; skips in the quick tier). Stages: L1 fixed-width
  integer expressions, L2 struct/union member access at correct byte offsets, L3 pointer/array
  GEP-equivalent indexing, L4 functions + a call graph. `python -m bcir.frontends.cfront <file.c>`
  prints every artifact. Next: L5–L6 (`volatile`/MMIO + bitfields + control flow) to reach the full
  register-map/MMIO MVP. Suite: live count + coverage in [`STATUS.md`](STATUS.md).
- 2026-06-18: **Channel plugin boundary, Clang-comparison perf budgets, C23 + C-frontend roadmap.**
  Turned hardware channels into a real plugin boundary: `bcir/channel_plugin.py` defines a stable,
  versioned `channel.json` manifest (the seven sections — identity, target-profile schema, codegen
  identity, runtime signal-provider contract, StreamPack-execution capability, calibration artifact,
  simulator/model/provenance flag); `register_from_manifest` / `discover_plugins` admit an FPGA/NVMe/
  HBM-PIM/accelerator extension **without editing the core**, and routing became data-driven
  (declared `capabilities`) while the built-ins keep their exact legacy per-kind routing (pinned).
  All nine built-ins round-trip through the format; a worked external example ships at
  `channels/example_tpu.channel.json`. Wired the five **Clang-comparison budgets** from
  `CLANG_COMPARISON.md` into the non-flaky gate (`bcir.perf_budget` gained a bare-metal-only
  `match_band` for the dense 0.98×/1.00× *matches* + a `reference_milli` so the documented
  6.0×/14.1×/1.33× *wins* are tracked in the JSONL trend log, never asserted in CI). Relearned C23
  from ISO/IEC 9899:2024 and folded the findings into roadmap §5.7 Phase C: a Tier-1/Tier-2 C23
  feature-adoption table (`_BitInt(N)`, `<stdckdint.h>`, `<stdbit.h>`+endian, `enum : T`, `constexpr`,
  `typeof`, `static_assert`, `nullptr`, `[[attributes]]`, `unreachable()`, `#embed` …) mapped to the
  oracle components each enables, the "move C beyond kernels" build-later tracks (a C graph runtime,
  fixed-point AI processes, X-macro/`typeof`/`constexpr` metaprogramming), the staged C-frontend
  conformance ladder (L1–L8, each gated by the six artifacts), the register-map/MMIO MVP criteria,
  and the generalized-C-output backend closing `C input → claim graph → K_BCIR plan → verified C
  output → Clang behaviour check`. Suite: live count + coverage in [`STATUS.md`](STATUS.md).
- 2026-06-18: **Test tiers, perf budgets, import quarantine, packaging/governance.** Named test
  tiers — `python -m bcir.tests.run_all --tier {quick,c-runtime,silicon-degrade,thorough}` — an
  escalating capability ladder (`quick` ⊂ `c-runtime` ⊂ `silicon-degrade` ⊂ `thorough`) layered on
  the suite's existing self-gating (a tier sets which host tools are visible + whether the full
  campaigns run; `BCIR_THOROUGH` still maps to `thorough`, so CI is unchanged). A non-flaky
  **perf-budget gate** (`bcir.perf_budget` + `tools/perf/check_budgets.py`): STRICT on correctness +
  measurement validity everywhere, perf floors enforced only under `BCIR_BAREMETAL` (waived on shared
  CI), with a provenance-tagged JSONL trend log. A **hot/cold import-quarantine** dependency-graph
  tool (`tools/perf/import_graph.py --check`) + test that pins the 27 research organs off the simple
  path (complements the path-level `test_perf.py` / `test_hot_cold.py`). Packaging/governance:
  `pyproject.toml` (PEP 621 + a green-on-the-tree ruff config), `.clang-format` / `.clang-tidy`,
  `mlir/CMakePresets.json` (mirrors `build_mlir.sh`), `SECURITY.md`, `.github/CODEOWNERS`,
  `.pre-commit-config.yaml` (ruff + clang-format + the doc/quarantine gates; quick chain on push). CI
  gains the perf-budget step (oracle job) and the import-quarantine gate (docs-governance job). Suite:
  650 oracle tests (live count in [`STATUS.md`](STATUS.md)).
- 2026-06-18: **Docs governance — generated status, link/retired-path CI, drift reconciliation.**
  Added `tools/docs/gen_status.py` → [`docs/STATUS.md`](STATUS.md), the single source of truth for
  counts + coverage (Python tests, ODS ops, registered passes, MLIR FileCheck tests, runtime-C
  components, the R1–R18 law-coverage matrix computed from `mlir/test/passes/verify*.mlir`, and the
  hardware-channel matrix) — so prose links here instead of carrying a hand-typed number (the root of
  the 580/615/631 + R17/R18 drift). Added two CI gates: `tools/docs/check_links.py` (code-aware
  broken relative-link checker — strips fenced + inline code so C++/IR snippets aren't mistaken for
  links) and `tools/docs/check_retired_paths.py` (an *active-doc* references a retired tree — e.g. the
  removed `runtime/llvm/` LLVM-IR-schema runtime — only with an explicit `<!-- allow-retired-paths -->`
  historical marker). Quarantined the 86 dead `runtime/llvm/` links across the `llvm-training/`
  substrate: retired guides now carry a deprecation banner pointing at the MLIR dialect + C runtime,
  and dead links became plain code mentions (display text preserved). Reconciled the live snapshots:
  the audit + roadmap now defer counts to `STATUS.md`, current-state verifier descriptions read
  **R1–R18** (R18 = compositional call-graph integrity, with its own MLIR negative cases), the MLIR
  rail is noted as LLVM 22, and the release ladder splits **0.4** into **0.4a** (proof-carrying
  *mechanism* — explain/replay/reduce, ✅) and **0.4b** (proof-carrying *contract* — stable cert
  schema + external replay-CLI contract + upgrade tests, ☐). Dated changelog entries below are left as
  historical snapshots (they captured the count/law set accurate *at that date*).
- 2026-06-18: **Roadmap expanded — the dependency-ordered plug-in-compiler plan (§5.7).** Replaced
  the "long-term frontends" stub with a sequenced, rationale-backed phased roadmap that elevates a
  **solid C frontend + backend to the immediate next major milestone** (it gates everything below):
  **Phase C** (C.1 a usable Clang-compatible C *frontend* → claim graph — the input seam that does
  not exist yet; C.2 generalize the per-pattern `lower.c_kernel` into a self-verifying C
  lowering/codegen for an arbitrary claim graph; C.3 full C — preprocessor, complete ABI, stdlib);
  **Phase M** (selective ML ops — tensor/attention/quantization — in parallel but throttled,
  oracle-first then ported to the MLIR law); **Phase D** (drivers + opcode tables + the Hardware
  Description Layer — import Linux kernel tables/register maps/PCIe/ACPI once C is verifiable, build
  BCIR-native ISA/opcode/registry, and a semi-separate `drivers/` JIT kernel generator that gives
  each hardware **channel** its real driver, closing the heterogeneous-tower loop); **Phase F** (C++
  deferred — templates/exceptions/RAII/STL/ABI; Python as a transpiler→full frontend); **Phase L**
  (the ML library, a *compressed extraction* — same "take only what we need" strategy as the Linux C
  files — from GCC/TensorFlow/PyTorch/pandas/NumPy/scikit-learn/XGBoost/JAX/Keras/XLA/SPIR-V/ONNX/
  Cassandra/SQL-NoSQL/vector-DBs onto the claim-graph + channel model). Through-line: frontends
  produce claim graphs, drivers populate channels, the ML library composes them — all decomposing to
  the same K_BCIR plan + GEM execution; isel stays gated. §6 carries the C-frontend keystone as the
  prominent next build step. Doc-only.

- 2026-06-18: **Unified heterogeneous runtime — hardware channels (`bcir/channels.py`).** The
  recurring x86/ARM friction was a symptom: per-architecture rules (the `perf_event_open` syscall
  number, the energy/thermal sensor paths, the codegen triple, the realizable lane widths) were
  *scattered* as inline `platform.machine()` checks, so the architectures collided. Introduced the
  **`HardwareChannel`** abstraction that isolates *everything* hardware-specific about one backend
  (the K_BCIR cost `profile`, the real `llvm_triple`/`e_machine`, the `runtime` hooks, the hardware
  `kind`) behind one uniform interface, so two channels coexist in one tower without their rules
  colliding. A registry of the six real arch backends (x86/ARM/RISC-V/GPU, host-ELF + GPU) plus three
  **modeled** future backends (`fpga_systolic`, `nvme_stream`, `hbm_pim` — the FPGA/NVMe/HBM-PIM
  extension points). **`orchestrate(module, tower, theta)`** plans a module across a heterogeneous
  tower — for each claim, the cheapest *suitable* channel by K_BCIR cost — so a reduction lands on
  HBM/PIM, a tiled matmul on the FPGA/GPU, a gather on the GPU, control on the CPU, all in one plan,
  every placement the **same K_BCIR arithmetic** on that channel's profile. It decomposes to a single
  GEM StreamPack: `LaneSegment.channel` + `apply_channels` tag **one binary graph** that runs across
  the whole tower. The silicon layer now reads its perf syscall from the host channel (one source of
  truth — the #255 collision is structural now). New `bcir/tests/test_channels.py` (13 tests:
  isolation, unified core, orchestration, the extension point); full suite 631 green; pure-Python (no
  MLIR rail change). Architecture + extension path documented in `docs/HETEROGENEOUS_CHANNELS.md`.

- 2026-06-18: **Test-suite consolidation (2b) — a 5.8× faster quick chain (~28s → 4.8s).** A
  measured audit corrected the premise: the bottleneck is NOT the test count or the generative
  campaigns (differential 1.4s, fuzz 1.8s — cheap), it is the **C-compiler subprocesses +
  wall-clock timing loops** (~77% of wall-time). `run_all.py` is now the **quick chain by default**:
  it gates the C/LLVM toolchain (one `shutil.which` monkeypatch, before any test import, covering
  both `shutil.which(...)` and `from shutil import which` bindings) so the ~130 compile / native-
  execute tests early-return through their existing guards. `BCIR_THOROUGH=1` (which **CI now
  sets** on both the x86 and aarch64 oracle jobs) runs the full compile + C-byte-identity tier.
  Also made the in-suite campaign/timing sizes mode-aware (differential 1500→400, verifier
  2000→400, fuzz 1500→400, the ring micro-benchmark 200k→20k by default; CI's separate
  `differential -n 8000` ×2 / `fuzz -n 4000` ×2 remain the deep proof, and `BCIR_THOROUGH` restores
  the larger in-suite N). **All 618 tests still run, 0 failures, in both modes.** Coverage preserved
  in the quick chain (verified): every law R1–R18, all six targets, the StreamPack/ABI *logic*, and
  the budget/RCSP correctness property are pure-Python and still execute; only the C *byte-identity*
  and native-*execute* proofs defer to CI (which runs them anyway). The parity assertions (no bugs,
  no coupling gaps, ok == checked) are unchanged at any N — the crown-jewel differential proof is
  not weakened, only sampled less in-suite.

- 2026-06-18: **Hardware-agnosticism / ARM (Raspberry Pi 5) compliance.** BCIR was meant to be
  hardware-agnostic but had subtle x86 wiring (the build + runtime were already portable: no x86
  intrinsics, no `-march` flags). A full audit found and fixed the blockers so the first real driver
  target — ARM/aarch64 on a Pi 5 — is first-class:
  - **B2 (parity-critical):** the tile lane width was hardcoded to 16 (AVX-512 f32) in *both* rails
    (`realize.py`, `BCIRCostModel.h`). Now `min(16, max_lane_width)` — 16 on AVX-512/SVE/RVV/PTX
    (pinned scores unchanged), but a realizable **4 on NEON / 8 on AVX2** (an unrealizable width was
    previously priced). Updated the `matmul_tiled` per-target pins + regenerated `target_matrix.mlir`;
    `tiled_matmul_arm64_neon` now scores 851968 in both rails (the regression net for B2).
  - **B1:** `perf_event_open` syscall number was hardcoded to x86's 298 — now arch-dispatched
    (aarch64/riscv64 = 241, armv7 = 364) so the ARM PMU actually works on a Pi 5 (`silicon.py`).
  - **B3:** added the `aarch64` native-object target (EM_AARCH64 = 183) to the codegen object gate.
  - **B4:** the conda MLIR-22 bootstrap now arch-selects (`micromamba-linux-aarch64`,
    `gxx_linux-aarch64`, `aarch64-conda-linux-gnu`) so the rail builds natively on a Pi.
  - **D1:** added `TargetProfile.for_host()` + `default_target_name()` (reads `platform.machine()`
    + `/proc/cpuinfo` features) and made the CLI / `api.build_artifact` / `bench.compare*` default to
    the **host** architecture instead of a hardwired `x86_avx512`; pinned the x86-asserting `test_api`
    cases to `x86_avx512` so the suite is deterministic on any host (incl. ARM).
  - **D3/D4:** the regenerated matrix now exercises the NEON tile path; added a native **aarch64 CI
    job** (`ubuntu-24.04-arm`) running the oracle + differential + C runtime + the perf-syscall path.
  Honest remainder (documented, not faked): ARM energy (RAPL is Intel-only; the Pi exposes no
  drop-in Joules — thermal + cpufreq + PMU work, energy stays dark until a shunt monitor is wired);
  the telemetry-ring endianness is LE-correct on the Pi (latent only on a big-endian host). Validated
  on x86 (full rail green, 617 oracle); the aarch64 CI job proves the ARM half on real hardware.

- 2026-06-18: **Closed the three remaining law-rail gaps — the deterministic spine has no known
  buildable gaps left.** All three built and dual-rail-verified against the oracle on true MLIR 22.
  **(1) `-bcir-overlap-optimize`** ports `gem/overlap.py::optimize_scheduled` — the makespan-driven
  re-selection sweep (from the serial optimum, adopt the per-claim alternative that strictly lowers
  the scheduled makespan; re-price serially for R9). `computeMakespan` was extracted from
  `-bcir-overlap` and shared. Matches the oracle's `(makespan, serial)` on all 11 corpus programs,
  including the real gains (fused_chain 7808<13696, matmul 253952<1015808); no-op where the serial
  optimum is already makespan-optimal (`overlap_optimize.mlir`). **(2) MOPC R12 support-preservation**
  — `bcir.target.lower_contract` gains optional `source_support`/`target_support`/`discharges`, and
  `-bcir-verify` R12 enforces `f(Supp(J)) ⊆ Supp(J')` unless discharged, reproducing
  `mapping.py::dropped` (`verify_laws_deep.mlir`). The commuting-square `Λ∘Ψ=Φ` is a runtime
  path-equivalence (already covered by the provenance digest + parity campaign), not a static law.
  **(3) `-bcir-sense`** ports `kbcir/sensing.py::RegretSensor.sense` — per-segment `cv_milli` over
  the `trace.data_dna` cycles (population variance, floor-isqrt), ranked `(-cv, segment)`, gated to
  `high`/`low`/`off`; matches the oracle exactly (`sense.mlir`). Now **85 ODS ops, 25 passes**, full
  rail green on true 22, 617 oracle tests unchanged. Roadmap §5.1 marks the gaps closed.

- 2026-06-17: **MLIR-22 completion follow-ups + dual-rail completeness scan.** Closed the
  remaining feature-adoption follow-ups and acted on an independent oracle↔law scan. **(1)
  `-std=c++2c`:** the C++ standard moves to C++26 — CMake's GNU `CXX_STANDARD=26` flag mapping
  only landed in CMake ~3.30 (ubuntu-latest + conda ship 3.28), so the base standard stays 23
  (mapped everywhere) and `-std=c++2c` is appended when `check_cxx_compiler_flag` accepts it (the
  later `-std` wins → the actual compile is C++26 on clang-22 / gcc-15; older toolchains stay at
  23). **(2) `hasVerifier` on more ops:** `bcir.claim` (count non-negative — `getCount()` is
  `uint64_t`, so it is reinterpreted signed; stride_k positive) and `bcir.target.capability`
  (cacheline a positive power of two; lane widths positive) join `resource`/`gem.lane_segment`;
  10 negative cases in `verify_ops.mlir`. **(3) Built the one op PARITY claimed but did not
  exist — `bcir.kbcir.memory_module`** (the resolution-fixpoint organ `a = Lim(Res(U))`) with a
  first-class `-bcir-verify` **R13** admissibility law (`saturated ∧ generation ≥ 1`), a
  `verify_laws_deep.mlir` negative case, and an IRDL projection entry. **(4) Corrected the stale
  `PARITY.md` rows the scan found:** `bundle` (was "roadmap" — `-bcir-bundle` shipped),
  `verifier differential` (R1–R18 are first-class in `-bcir-verify`), `two_truth` (no op — it is
  correctly *quarantined off-rail by design*, now framed like the operad row), and `target.lower_contract`
  (the support-containment / commuting-square check is a tracked R12 refinement, not yet enforced).
  **(5) Roadmap:** §5.1 records the MLIR-22 completion + the three narrow remaining law-rail gaps
  (overlap re-selection sweep, the MOPC R12 refinement, the sensing telemetry gate); new §5.7 sets
  the long-term language-frontend direction (full C / Python / C++ frontends — explicitly *not*
  near-term). The scan's verdict: the law rail mirrors the oracle's **entire deterministic spine**
  (84 ops, 23 passes, R1–R18); the interpretive/learned/two-truth layer is correctly off-rail. All
  validated on **true MLIR 22.1.7**; 617 oracle tests unchanged. The deeper IRDL attribute/type
  constraints stay structurally bounded (pure IRDL cannot reference local types via `irdl.base`
  cleanly, nor express the cost-vector/R-law predicates without `irdl.c_pred` + compiled C++).

- 2026-06-17: **Goal 4 — IRDL projection fidelity (op coverage).** The pure-IRDL portability
  projection (`mlir/irdl/bcir.irdl.mlir`, validated on stock `mlir-opt`) was missing 4 ODS ops
  and carried 1 orphan. Added the 4 K_BCIR ops it omitted — **`kbcir.theta`** (the runtime-state
  context op), **`kbcir.func`** / **`kbcir.call`** / **`kbcir.cond`** (the compose.Function/Call/
  Cond family, regions presence-only per the loose rail) — and removed the orphan
  **`@verify_target_capability`** (there are 12 `bcir.verify.*` ODS ops; the IRDL carried 13, the
  extra having no ODS counterpart — the `verify_*` IRDL family now matches the ODS family exactly).
  New `mlir/test/irdl/compose_theta_generic.mlir` round-trips the four ops through the projection.
  Verified on **true MLIR 22.1.7** (conda `mlir-opt`): the projection loads and the whole IRDL
  corpus round-trips. (Deeper attribute/enum/type constraint fidelity stays a structurally-bounded
  follow-up — pure IRDL on 22 cannot express the cost-vector/R-law predicates without `irdl.c_pred`
  + compiled C++, which is the stated rationale for keeping the projection a loose structural rail.)

- 2026-06-17: **Goal 3 — parse-time op verifiers (`hasVerifier`).** The dialect had no per-op
  verifiers; all checking lived in the monolithic `-bcir-verify` pass (the cross-op semantic
  R-laws), which runs only when invoked. Added `hasVerifier = 1` + a `verify()` to the ops with
  genuinely per-op structural invariants so a malformed op is rejected **at parse/build**, not
  just under the pass: **`bcir.resource`** (`align` is a positive power of two; `shape` extents
  are positive) and **`bcir.gem.lane_segment`** (`width` is a positive power of two — the SIMD
  lane width; `stride_k` is positive). New `mlir/test/passes/verify_ops.mlir` (6 negative
  `-verify-diagnostics` cases) + a `check_passes.sh` gate. The whole positive corpus already
  satisfies these (every `align = 64`, every shape/width positive pow2, every `stride_k = 1`),
  so nothing was rejected; verified on **true MLIR 22.1.7** with the full rail green and 617
  oracle tests unchanged. The pattern extends to further ops as a follow-up.

- 2026-06-17: **Goal 2 complete — all optimizer passes share the plan analysis.** Migrated the
  remaining consumers onto `cm::PlanAnalysis` (following the validated cost/plan/overlap
  pattern): `-bcir-rcsp-plan` (cols/θ/w for its constrained label-DP), `-bcir-cim` (needs only
  the capability `h`), `-bcir-dvfs` (reuses `pa.chosen`; `firstThetaThermal` == the first theta
  op's thermal, verified; keeps the theta walk for `power`), `-bcir-explain` + its `freshRecord`
  helper + `-bcir-replay`, the `-bcir-schedule-eft` / `-bcir-async` / `-bcir-power-rail` trio
  (via `planInfos`, which now takes `pa` and reads `memChannels`/`affinityDomains` off the
  carried capability handle), and `-bcir-bundle` (a mutable copy of `pa.cols` for its permuted
  re-prices). Each `markAnalysesPreserved<PlanAnalysis>()`. Verified on **true MLIR 22.1.7**: the
  whole rail is green (tblgen, R1–R18, GEM pipeline, ODS, bytecode, IRDL) with every pinned score
  byte-identical, and a multi-pass pipeline now builds the plan **once** instead of per pass.
  (Non-`PlanAnalysis` passes and the GEM transforms correctly invalidate it, so an interleaved
  transform re-plans — sharing happens within annotation-pass groups, e.g. `bcir-audit`.)

- 2026-06-17: **Goal 2 (foundation) — cross-pass plan sharing via an MLIR Analysis.** Every
  optimizer pass independently re-walked the module (`cm::fusedColumns`) and re-ran the coupled
  min-plus shortest path (`cm::planChosen`) for itself — the same plan recomputed up to ~9×
  across a pipeline. Added `cm::PlanAnalysis` (BCIRCostModel.h): a per-`bcir.module` MLIR
  analysis that computes the fused columns + chosen plan once, requested via
  `getChildAnalysis<PlanAnalysis>(mod)` and **shared** across passes that
  `markAnalysesPreserved<PlanAnalysis>()` (correct because they only add `kbcir.*` annotation
  attrs, never plan inputs). It splits `hasCap` (cost-model rail: columns only) from `valid`
  (plan rail: + weighted shortest path) and carries the capability handle for per-pass scalars
  (affinity domains, mem channels). Migrated the **bcir-audit core — `-bcir-cost-model`,
  `-bcir-plan`, `-bcir-overlap`** — verified on **true MLIR 22.1.7** (conda toolchain): the
  whole `cost+plan+overlap` pipeline now builds `PlanAnalysis` **once** (was 3×), every pinned
  score is byte-identical (7808 / corpus 1015808·101888·1595520 / overlap 253952·761856), and
  the full rail (tblgen, R1–R18, ODS, bytecode, IRDL) is green. The remaining consumers
  (`-bcir-rcsp-plan`, `-bcir-cim`/`-dvfs`, `-bcir-explain`/`-replay`, `-bcir-schedule-eft`
  `planInfos` trio, `-bcir-bundle`) still self-compute (correct) and follow the same pattern —
  to be migrated next.

- 2026-06-17: **Goal 1 — verifier fault-injection hardening (R2–R9 generative).** The
  oracle's law-for-law fault-injection campaign (`run_verifier_campaign`) exercised only 5 of
  the laws (R2/R3/R5/R6/R7). Extended it to the full **module/claim rail R2–R8** (added R4 =
  phase self-cycle, R8 = unknown cost class) and added a new **plan rail R9** (`gen_illegal_plan`
  / `check_plan_verifier`: a clean `optimize()` result corrupted to break the score-sum and
  total-coverage invariants, flagged through `verify_plan`). **R1 (RID uniqueness) is documented
  as enforced by construction** — `Module.resources` is a dict keyed by RID and `add_resource`
  rejects dups, so verify()'s R1 loop is unreachable and cannot be fault-injected (the campaign
  states the invariant rather than faking it). New `test_plan_injectors_fire_R9`; the
  isolation test now pins R2–R8; 900-iteration campaign clean; 616 oracle tests green. The
  artifact laws R10–R18 (pack / lowering / smart-lowering / provenance / accuracy / call-graph)
  remain covered by hand-written negative tests on both rails plus the R1–R18 toolchain coverage
  gate (Phase 1); extending the *generative* campaign to them needs valid random artifact
  generators (a tracked follow-up).

- 2026-06-17: **True MLIR-22 local validation (conda-forge), solving the apt.llvm.org block.**
  The web sandbox's network policy denies `apt.llvm.org` (`403 host_not_allowed`) — the usual
  MLIR-22 source — and the stock Ubuntu archive tops out at MLIR 18, so prior sessions could
  only build the rail against 18 locally (validating logic, not the 22-only rules) and leaned
  on the single LLVM-22 CI job. Probing the egress allowlist showed **conda-forge is reachable**
  and ships real `mlir=22.1.7` dev libs + `llvmdev` + an ABI-matched compiler. New
  `tools/local/{setup_mlir22.sh,env_mlir22.sh,check_rail22.sh}` install micromamba + a conda
  `m22` env and build `bcir-opt` against **true MLIR 22.1.7** (the system compiler segfaults on
  conda's MLIR via duplicated `TypeID` statics — fixed by building with conda's
  `gxx_linux-64`). The **whole rail is now green locally on real 22**: tblgen, R1–R18 verify,
  GEM pipeline, cost/plan/overlap/RCSP, compose, ODS examples, bytecode round-trip, and — the
  check an 18 build cannot do — the **IRDL named-operand corpus**. CI (apt.llvm.org) stays the
  authoritative gate; the clean alternative is to allow `apt.llvm.org` in the environment's
  network policy. (The 18 path, `tools/wsl/build_mlir.sh`, still works as a lighter fast loop.)

- 2026-06-17: **Phase 2a — MLIR-22 feature adoption + dead-code prune.** Two empirical
  findings reshaped the planned "Properties migration": (1) the BCIR dialect **already uses
  inherent Properties storage** for every op's attributes — MLIR's `usePropertiesForAttributes`
  defaults on, so `let arguments = (ins …Attr…)` is stored as properties without explicit
  `Prop<>`/`IntProp<>` syntax (the generic dump prints `"bcir.…"() <{…}>`, the properties
  form). The "zero Properties adoption" reading was a false negative; the further explicit
  typed-property migration is low-ROI/high-churn and deferred. (2) **MLIR bytecode** (a stable
  versioned format the rail never exercised) round-trips the whole positive corpus
  byte-identically — `text -> --emit-bytecode -> text` is a no-op and passes run on bytecode
  input — which also proves the Properties storage serializes correctly. Added
  `tools/wsl/check_bytecode.sh` (14 modules: examples + plan/gem/overlap/rcsp/compose/async
  corpora) and a CI gate. Removed the **dead, stale `mlir/passes/GEMPasses.td`** (generated
  into `GEMPasses.h.inc` but never `#include`d; declared only 5 of 23 passes, mis-pinned to
  `ModuleOp`) + its CMake/tblgen-check wiring; clean rebuild + full local rail (R1–R18,
  GEM pipeline, ODS examples, bytecode) green. Pass-registration modernization (td-generated
  `GEN_PASS_DEF` + stable cross-DSO TypeID) was assessed as **low practical ROI** for a single
  `bcir-opt` binary (the internal-inline TypeID instability only bites across shared-library
  boundaries, which the rail does not have) and deferred.

- 2026-06-17: **Phase 1 — functional hardening + performance (post-LLVM-22 follow-up).**
  Three-axis audit (MLIR-22 compliance / performance / correctness) drove a first batch of
  safe, locally-validated fixes. **Test correctness:** the verifier coverage gate
  (`test_verify_differential.py`) was a string-grep that enforced only R1..R17 and omitted
  `verify_callgraph.mlir` — extended to **R1..R18** so R18 (call-graph integrity) can no
  longer silently lose its toolchain-rail negative case. **Performance (measured):** the
  full oracle suite dropped **~39.6s → ~27.6s (-30%)** from two changes — (P1) the MLIR
  emitter recomputed the optimizer + `fused_candidates` 4–5× per fuzz iteration; `optimize`
  now carries its deforested `cand_map` on `RealizationResult` and `_fuzz_mlir` plans once
  and threads the `result` through both emissions + `plan_view`; (P4) `DataDNA.to_dict`
  swapped the recursive deep-copying `dataclasses.asdict` for an explicit flat dict (it is 9
  scalar fields). Emitted corpus stays byte-identical (drift gates hold); differential parity
  clean (0 bugs / 0 verifier misses); 615 oracle tests green. **CI (the ~9-min rail is
  build-dominated):** `build_mlir.sh` now uses `ccache` as the compiler launcher when present
  (no-op locally) + explicit `--parallel nproc`; CI installs+caches `ccache` (warm relink
  instead of full rebuild), adds `concurrency: cancel-in-progress` (superseded pushes no
  longer queue 9-min jobs), and runs the two differential + two fuzz seeds in parallel.
  **Local-build unlock:** `bcir-opt` now builds against the stock Ubuntu MLIR 18 dev libs in
  this environment, so the full pass rail (R1–R18 verifier, GEM pipeline, cost/plan/overlap/
  RCSP, compose) — previously gated only by the single LLVM-22 CI job — is now runnable
  locally as a fast pre-CI gate (CI stays the authoritative MLIR-22 gate; IRDL's 22-only
  named syntax stays CI-validated).

- 2026-06-17: **MLIR rail moved to the latest LLVM/MLIR (22).** The `mlir-rail-validate` CI matrix
  was `["18","19"]` (both from Ubuntu's default repos); it is now `["22"]`, installed from
  apt.llvm.org (the `llvm.sh` helper adds the signed repo) since the default repos top out near
  18/19. The toolchain resolvers were made version-agnostic (FileCheck / mlir-opt / mlir-tblgen now
  resolve the highest `/usr/lib/llvm-*/bin/<tool>` instead of a hardcoded `-18`/`-19` fallback, which
  would have silently degraded `check_passes.sh` to parse-only on 22). build_mlir.sh already
  auto-resolves the highest installed `/usr/lib/llvm-*`. Compatibility against the 22 compiler +
  verifier is validated in CI (the sandbox cannot build MLIR); this is the foundational change for
  the LLVM-22 feature-adoption work. C++ standard stays C++23 for now (a `-std=c++2c` bump is a
  candidate follow-up on the clang-22 toolchain). The hand-written C++ pass library compiles clean
  on MLIR 22 (only benign C++20 `operator==` ambiguity warnings from MLIR's own headers).
  Three upstream MLIR-22 verifier/parse tightenings were adapted: (1) IRDL's value-list ops
  (`irdl.operands/results/regions`) became NAMED (`name: %value`, variadicity ahead of the value);
  (2) IRDL forbids dots in op names (`isValidName`), so the pure-IRDL portability projection +
  corpus were flattened to underscores (`bcir.target_capability`), the compiled dotted ODS dialect
  staying the source of truth; (3) a `Symbol` op may no longer produce an SSA result, so
  `bcir.resource` / `bcir.gem.stream_pack` / `bcir.kbcir.path` (symbols that also returned a now-
  vestigial, never-consumed handle) became pure symbols -- results dropped from the ODS, the ~30
  hand-written test `.mlir`, the Python emitter, and the regenerated drift-gated corpus
  (gem_corpus / target_matrix / theta_hot); the C++ only ever reads these by symbol, so it is
  unaffected. `bcir.kbcir.select` is not a symbol and keeps its `!bcir.path` result. 615 oracle
  tests + the Python<->MLIR differential stay green.
- 2026-06-17: **proof.replay on the IR (`-bcir-replay`).** Ports bcir/kbcir/proof.replay: a
  recheck of the proof-carrying record a deployed plan carries. BCIRExplainPass.cpp gained a
  shared freshRecord() helper (recompute the plan -> per-claim chosen width + coupled edge score +
  module total) used by the new -bcir-replay pass, which diffs the fresh decision against the
  declared kbcir.explain_* record (module kbcir.explain_total + per-claim explain_chosen/
  explain_score) and annotates kbcir.replay_reproduced (bool) + replay_mismatches (the per-field
  divergence list, mirroring ReplayResult.mismatches). replay.mlir: @good carries the faithful
  7808/w16/7808 record -> reproduced=true; @tampered declares a wrong edge score (9999) -> the
  exact "claim 1: replay (w16/7808) != recorded (w16/9999)" divergence, reproduced=false. The
  R13 provenance-digest gate of proof.replay is already covered by -bcir-verify's first-principles
  provenance recheck; this pass adds the decision-record half. Pinned by test_proof.py. +1 test
  (615). -bcir-explain is unchanged (explain.mlir validates the freshRecord factoring).
- 2026-06-17: **Per-slot power rail (`-bcir-power-rail`).** Ports
  gem.schedule.schedule_power_rail: a per-slot DVFS overlay on the EFT *placed timeline* -- the join
  of -bcir-schedule-eft and -bcir-dvfs. BCIRScheduleEftPass.cpp gained a shared placeBarriered() free
  function (the phase-barriered EFT loop, now reused by -bcir-schedule-eft and the rail) plus base
  compute/memory on each Info. The rail classifies each scheduled slot by its base compute:memory mix
  (gem.dvfs.classify/clock_for) and sets a per-slot Q8 clock for its real [start,finish) interval --
  memory-bound slots downclock to 192 (power saved, bandwidth-bound throughput unaffected), keying
  off the slot interval rather than -bcir-dvfs's per-phase totals. Annotates per claim
  kbcir.rail_domain/start/finish + rail_class/rail_clock, per module rail_makespan/rail_knee and
  rail_energy_saved (the modeled energy avoided, sum of (nominal-clock) x interval). power_rail.mlir:
  two memory-bound slots on the 7808/5888 timeline both downclock, energy_saved 3424000. Pinned by
  test_schedule.py. +1 test (614). The EFT pass's behavior is unchanged (its FileCheck validates the
  placeBarriered refactor). Also carries the buildInfos const-accessor fix from the async refactor.
- 2026-06-17: **Async token plan + pipelined schedule (`-bcir-async`).** Ports
  gem.async_tokens.async_plan + schedule.execute_tokens: the !bcir.token fork/await DAG drives a
  SINGLE cross-phase EFT dispatch (no phase barriers), so an independent claim of a later phase
  overlaps an earlier one -- software pipelining falls out of the dependency structure.
  BCIRScheduleEftPass.cpp was refactored to share the placement machinery (buildInfos / eftDispatch
  / topoPhases as free functions, the dispatch annotating kbcir.<prefix>_*) between -bcir-schedule-eft
  (phase-barriered) and -bcir-async (pipelined). The async pass annotates kbcir.async_awaits (the
  awaited claim ids), async_domain/start/finish, and async_makespan. async.mlir: a phase-1
  independent claim overlaps phase 0 (start 0), a dependent one awaits c1 (start 7808), pipelined
  makespan 15616 vs 2*7808 phase-barriered. Pinned by test_schedule.py. +1 test (613). The EFT
  pass's behavior is unchanged (its FileCheck validates the refactor).
- 2026-06-17: **Tier-2 (3/3): allocator pool-plan on the law rail -- Tier-2 complete.**
  `-bcir-alloc-pool` (`BCIRAllocPoolPass.cpp`) ports `kbcir.allocator.pool_plan`: liveness-based
  memory pooling. It computes each touched resource's [first_phase, last_phase] live range (over
  the phase declaration order), then greedy left-edge interval partitioning -- each resource, in
  (start, rid) order, reuses the first arena whose last member has died (`last_phase < this start`),
  else opens a new one. Resources with disjoint live ranges share an arena, so the peak footprint
  (sum of arena sizes) drops below the naive sum. Annotates kbcir.pool_id per resource +
  kbcir.pool_naive_bytes / pool_peak_bytes / pool_saved. `alloc_pool.mlir` (A/D + B/E share arenas,
  C its own -> peak 12288 vs naive 20480, saved 8192); `test_persistent_oracles.py` pins it. +1
  test (612). **All three Tier-2 "recompute-don't-trust" ports are now on the law rail.**
- 2026-06-17: **Tier-2 (2/3): EFT duration-aware schedule on the law rail.** `-bcir-schedule-eft`
  (`BCIRScheduleEftPass.cpp`) ports `gem.schedule.schedule_eft` -- the HEFT-lite refinement of CT2
  wave formation. It plans the module for per-claim durations (the chosen edge costs), then per
  phase (topo order) runs an event-driven list scheduler: LPT priority (longest duration first),
  earliest-finish-time domain placement, locality tie-breaks (the deduped operand-RID overlap with
  each domain's resident set), and the bandwidth-knee clamp (bandwidth-class claims contend for
  min(affinity_domains, mem_channels) slots, compute for all); the GGG/random tail runs decoupled
  on its own stream; phases compose serially. Annotates kbcir.sched_domain/start/finish per claim +
  sched_makespan/knee per module. `schedule_eft.mlir` (two shared-read compute claims -> domain 0
  @7808 / domain 1 @5888, makespan 7808, knee 4); `test_schedule.py` pins the constants. +1 test (611).
- 2026-06-17: **Tier-2 (1/3): CIM/PIM dispatch + DVFS clock DECISION recompute.** Two new
  passes recompute the GEM scheduling decisions from the IR -- the way `-bcir-cost-model`
  recomputes cost -- instead of R14/R15 only *verifying* a declared attr. (1) **`-bcir-cim`**
  (`BCIRCimDvfsPass.cpp`) ports `gem.cim.cim_decision`: for a reduction, model core_cost
  (`count*(elem_bytes+mem_unit)`) vs pim_cost (in-memory compute x1.5 + a 4096 dispatch setup +
  a 1-element result) and annotate `kbcir.cim_offload`/`cim_core_cost`/`cim_pim_cost`; offload
  iff PIM strictly wins (large reductions). (2) **`-bcir-dvfs`** ports `gem.dvfs`: plan the
  module, sum each phase's (compute, memory), classify by intensity, and set a Q8 clock
  (downclock memory-bound, overclock compute-bound unless Theta-capped, else nominal) ->
  `kbcir.dvfs_class`/`dvfs_clock`. `cim.mlir` (offload at 4096 / not at 1024) + `dvfs.mlir`
  (vector_add -> memory -> 192) pin the law rail; `test_cim.py` pins the constants. +1 test
  (610). R14/R15 still verify legality (defense in depth); these derive the decision.
- 2026-06-17: **Tier-1 compose remainder: alias/effect + independence + dynamic shapes on the
  law rail.** `-bcir-compose` now also ports `compose.effect`/`independent` and dynamic shapes:
  (1) **`kbcir.effect_reads`/`effect_writes`** -- each func's read/write footprint, folded
  through its calls' argument substitution (`regionEffect`/`opEffect`). (2)
  **`kbcir.commutes_with_prev`** on each call with a preceding sibling -- true iff their
  footprints are disjoint (no RAW/WAR/WAW, `effectsConflict`), so the two reorder/overlap (the
  cross-call alias test the pairwise plan cannot see). (3) **`kbcir.compose_dynamic`** -- the
  claim op gained a `dynamic` attr, and a dynamic-shape leaf makes the func's cost a worst-case
  bound that holds for any actual size <= the count (`compose.plan_holds_for`). `compose_effect.mlir`
  pins all three. MLIR-only (the oracle effect/independent/dynamic are already tested); 609
  oracle tests pass. **Compositional semantics is now complete on both rails.**
- 2026-06-17: **RCSP-constrained compose + a compose-rail differential.** (1) `plan_composite`
  gained a `budget` (oracle-first): each Leaf is priced by `rcsp.optimize_constrained`, so the
  region-tree plan respects the central equation `min M(pi,Theta) s.t. R(pi,Theta) <= B` -- a
  thermal cap makes wide SIMD infeasible per parallel block (re-prices to vec8) or raises
  `Infeasible`; unbounded == the old pinned scores. (2) Ported to `-bcir-compose` via a reusable
  `cm::planConstrained` (the accumulated-budget label DP, score-only twin of `-bcir-rcsp-plan`):
  with a `kbcir.budget` present each Leaf is priced constrained and the func is annotated
  `kbcir.compose_feasible` (thermal<=800 -> vec8 9472 feasible; <=400 -> infeasible;
  `compose_budget.mlir`). (3) A generated **compose differential** (`test_compose_differential.py`)
  fuzzes the metamorphic laws -- determinism, worst>=expected, unbounded-budget degeneracy, RCSP
  monotonicity (a feasible cap never lowers the cost), summary consistency -- over randomized
  region trees. +6 tests (609 total). Compositional semantics is now complete on both rails.
- 2026-06-17: **Inter-procedural summary costs in `-bcir-compose`.** `kbcir.call` is no longer
  a plain inline: a `kbcir.func` is planned **once** over its formals (memoized -- the
  summary), and a call whose actuals are **cost-compatible** with the formals (same
  `compose._cost_key`: domain / element-count / access -- `costKeyEq`) reuses that summary;
  an incompatible call (e.g. an HBM actual where the formal was DRAM) re-prices the body with
  the actuals substituted (`fusedColumnsFromClaims` gained a `subst` remap). It annotates a new
  `kbcir.compose_reused` (leaf plans saved) alongside `compose_worst`/`expected`. Reproduces
  the oracle's `plan_composite(summaries=...)`: a 7808 summary reused for a DRAM call + re-priced
  to 2816 for an HBM call -> 10624 / reused 1 (`compose_summary.mlir`, pinned by
  `test_compose.py`). +1 test (603 total). With this, compositional semantics is fully dual-rail.
- 2026-06-17: **Full provenance component cross-checks + the compositional plan on the law
  rail.** Two follow-ups completed. (1) The IR now carries every field the provenance
  `hash_*` consume: the claim `opcode`, the capability `target_name` + `scalable`, the policy
  unfolded `base_weights`, and the module `cacheline`/`align` (all defaulted/optional ->
  back-compatible). R13 in `-bcir-verify` recomputes **all four** component hashes
  (`hash_module`/`hash_target`/`hash_theta`/`hash_policy`) from the manifest's enclosing
  `bcir.module` -- byte-identical to the oracle (validated leaf-by-leaf against a real
  vector_add manifest) -- so a manifest can be re-pointed at neither a different goal graph,
  target, runtime state, nor policy (`verify_provenance.mlir`: a full real module + opcode/
  capability tamper negatives; `test_provenance.py` pins the byte-exact algorithm). (2)
  **`-bcir-compose`** (`BCIRComposePass.cpp`) computes the compositional cost over the
  `kbcir.func`/`call`/`cond` region tree -- the law-rail `compose.plan_composite`: a region's
  direct `bcir.claim` leaves are priced by the shared cost model (`fusedColumnsFromClaims` +
  `planChosen`), `Seq` sums, `Cond` is worst-case max + probability-weighted expected, `Call`
  inlines; it annotates `kbcir.compose_worst`/`compose_expected`, reproducing the oracle's
  7808 leaf and 23432/18747 Seq/Cond/Call program (`compose_cost.mlir`). +1 test (602 total).
- 2026-06-17: **Law-rail deepening: R18 call-graph law + R13 m_theta cross-check.** Two
  follow-ups from the paradigm audit. (1) **R18 (compositional call-graph integrity)** in
  `-bcir-verify`: every `kbcir.call` must resolve to a `kbcir.func` and the call graph must
  be acyclic (DFS back-edge = recursion) -- the law-rail form of `compose.plan_composite`'s
  undefined-callee + recursion rejections (`verify_callgraph.mlir`: resolve / undefined /
  self-recursion / mutual-recursion-through-cond). (2) **R13 now cross-checks `m_theta`
  against the IR**: `kbcir.theta` carries all eight pressures (added noise/wear/utilization/
  voltage), and R13 recomputes `hash_theta` byte-identically to `provenance.hash_theta` and
  confirms it equals the manifest's declared `m_theta`, so a manifest can't be re-pointed at
  a different runtime state (`verify_provenance.mlir`). The module/target/policy component
  cross-checks remain follow-ups (the IR must first carry the claim `opcode` / full
  capability fields / unfolded base weights). LangRef R1–R18; 601 oracle tests pass
  (MLIR-only change; the oracle recursion contract is already pinned by `test_compose.py`).
- 2026-06-17: **Full-oracle paradigm audit + the one gap closed: R13 provenance digest
  recompute.** Swept all ~55 oracle modules against the two-truth placement rule and
  confirmed every deterministic-integer decision/execution-path component is on its correct
  rail (MLIR/C++ optimizer core + verifier R1–R17 + GEM passes; C runtime decode/encode/
  execute/binrec/telemetry) and every float/learned/train-time/generator module is correctly
  Python. The single finding: `-bcir-verify` **trusted** the declared provenance digest rather
  than recomputing it. Closed it — when the `kbcir.provenance_manifest` op carries the four
  component hashes (`m_module`/`m_target`/`m_theta`/`m_policy`) + the in-force artifacts, R13
  now recomputes the digest with an FNV-1a chain **byte-identical to `provenance._digest`**
  and rejects a tampered/stale one (`verify_provenance.mlir`, real vector_add hashes; pinned
  by `test_provenance.py`). Additive + back-compatible (a manifest omitting the components is
  range/reproduced-checked as before). +2 tests (601 total).
- 2026-06-17: **Compositional deepening + the MLIR ports finished (bundle joint-reorder,
  proof-carrying explain, func/if ops) + the rig contract made crisp.** (1) `kbcir.compose`
  deepens: **alias/effect modeling** (`Effect`/`effect`/`independent` -- the read/write
  footprint folded through calls + the RAW/WAR/WAW test that decides whether two calls
  commute) and **inter-procedural summary costs** (`summarize`/`FunctionSummary` -- a
  function is planned **once** over its formals and every cost-compatible call reuses that
  cost instead of re-planning the body; sound because reuse is gated on the actuals matching
  the formals' cost-keys, else it falls back to inline; bounds compile time to
  O(functions + call-sites)). (2) `-bcir-bundle` gains the **joint-reorder transformation**:
  it reorders the cost-model columns so an input-sharing bundle is contiguous, re-runs the
  min-plus shortest path for every legal intra-bundle order, and annotates the re-priced
  `kbcir.bundle_gain` / `bundle_order` (`bundle_reorder.mlir`) -- a re-price, never an IR
  mutation. (3) `-bcir-explain` (`BCIRExplainPass.cpp`) ports `proof.explain`: the
  proof-carrying decision record as IR annotations -- per claim the candidates weighed
  (widths + scalarized costs), the chosen width/score, and any fusion credit; per module the
  plan total (reproduces 7808 on `vector_add`; `explain.mlir`). (4) The `kbcir.func` /
  `kbcir.call` / `kbcir.cond` op family gives `compose.py`'s region tree first-class MLIR
  form and round-trips (`compose_ops.mlir`). (5) The CT4 runbook probe now **enumerates the
  three gating signals** (PMU + RAPL + cpufreq userspace governor) and prints an explicit
  `rig-ready` verdict; `--require-real` fires exactly when all three are present, so the
  measured win lights up the moment a bare-metal host runs the runbook
  (`test_silicon_runbook.py`). +7 tests (599 total).
- 2026-06-16: **Compositional semantics (first slice) + the CT4 runbook + bundle-detection
  on the law rail.** (1) `kbcir.compose` extends planning past straight-line kernels along
  the central equation's series-parallel grain: a region tree of `Seq` (sum), `Cond`
  (control flow -- worst-case max + probability-weighted expected), `Call`/`Function`
  (reuse via inline argument substitution; recursion rejected), and `dynamic` claims (count
  as a static upper bound, worst-case priced -- the plan holds for any actual <= the bound).
  Reuses `optimize` for leaves, so `Leaf([vector_add])` prices to exactly 7808.
  (2) `tools/silicon/measure_replan.sh` makes the rig-gated measured replan **push-button**
  and CI-exercises it in degrade mode (synthetic, no fabricated number; `--require-real`
  fails without a rig) -- the measured win still needs the bare-metal rig
  (HARDWARE_VALIDATION.md). (3) `-bcir-bundle` (`BCIRBundlePass.cpp`) ports the bundle
  *analysis* to the law rail: it annotates the input-sharing bundles (`kbcir.bundle` /
  `bundle_shared` / `bundle_count`); `bundle.mlir`. The joint-reorder transformation +
  proof-carrying MLIR are the next increment. +13 tests (592 total).
- 2026-06-16: **Compensated precision + R17 accuracy law, bundle (joint) optimization,
  proof-carrying records, verifier law-for-law differential.** Four layers:
  (1) `lower.c_kernel.emit_compensated_reduce_c` lowers the residual-carry Q8 reduction
  (`precision="compensated"`), bit-identical to the int64-exact result (the naive form
  drifts up to n ULP); the **R17 accuracy contract** is now dual-rail
  (`verify.verify_accuracy` + the MLIR `-bcir-verify` R17 law on `#bcir.precision<…,exact,
  tol>`): a tight tolerance on a long reduction forces the compensated realization.
  (2) **Multi-claim bundle (joint) optimization** (`kbcir.bundle.optimize_bundled`) -- the
  first genuinely new optimizer capability: jointly reorder input-sharing claims (bounded,
  exhaustive, dependency-preserving) to recover the fusion discount the pairwise shortest
  path misses; a real **12% gain on tiled matmul**, emitting a search certificate per gain,
  corpus otherwise pinned. (3) **Proof-carrying records** (`kbcir.proof`; CLI `bcir.run
  --explain/--replay/--reduce`): a replayable `DecisionRecord` (R13 digest + per-claim
  rationale + rewrite certificates) that reproduces bit-for-bit or diffs. (4) The **C++
  `-bcir-verify` law-for-law differential** is complete -- every law R1-R17 has a
  toolchain-rail negative case + a coverage gate. Verifier laws are now **R1-R17**. +19
  tests (580 total).
- 2026-06-16: **C StreamPack encoder + the verification cost dimension.** (1)
  `runtime/c/bcir_encode.c` -- the write-side twin of `bcir/abi/streampack_abi.py`:
  `bcir_sp_reencode` parses a pack and re-serializes it through value-based write
  primitives, **byte-identical** to the Python encoder across the corpus and both ABI
  versions (v1 + v2 pipeline/double-buffer tails). With the decoder + executor, the
  StreamPack is a full no-Python, no-libc C round-trip. Parity gate `test_c_encoder.py`
  + `check_runtime.sh`; libFuzzer + ASan/UBSan `fuzz_encode.c`. (2) The 12th cost axis
  (`verification`), modeled as 0 until now, gains a real producer on both rails
  (`realize._verify_cost` / `BCIRCostModel.h::verifyCostFor`, `cost_model_verify.mlir`):
  the verify-contract discharge cost -- `none`/`bounds` free (the bounds check fuses into
  the priced access, so every existing pinned score is unchanged), `exact`/`hash` an O(n)
  cost. Width-independent (never perturbs selection), a tradeable RCSP resource. exact/hash
  were unused, so purely additive. +9 tests (553 total).

- 2026-06-16: **The C StreamPack executor + R14/R15/R16 as first-class `-bcir-verify`
  laws.** (1) `runtime/c/bcir_exec.{h,c}` is the C twin of `bcir/gem/execute.py`: a
  freestanding `bcir_sp_execute` that decodes a pack and dispatches its claims in GEM
  order -- topological phase order (first appearance in the pack), then ascending
  claim_id within a phase -- invoking an optional per-claim kernel callback and
  collecting per-phase telemetry, with no libc and caller-owned scratch/phases buffers.
  Python<->C dispatch-order + telemetry parity (`bcir/tests/test_c_executor.py`,
  `tools/c/check_runtime.sh`) and a libFuzzer + ASan/UBSan harness (`fuzz_exec.c`, wired
  into `fuzz_streampack.sh`) -- the StreamPack is now a no-Python hot artifact a driver
  runs end to end. (2) `-bcir-verify` (`BCIRVerifyPass.cpp`) gains **R14** (CIM/PIM
  dispatch: pim only for reduce.*), **R15** (DVFS clock in [64,512]; pim must not
  overclock), and **R16** (allocator placement: L1 <= 64 KiB, L2 <= 4 MiB) as
  first-class verifier laws -- previously enforced only at the `-bcir-lower-to-llvm`
  checkpoint -- so the dedicated verifier now checks all R1-R16 (dual-rail with
  `verify.{verify_cim,verify_dvfs,verify_allocator}`); positive/negative cases in
  `mlir/test/passes/verify_laws_deep.mlir`. +4 tests (544 total).
- 2026-06-16: **Doc consolidation -- one master roadmap.** Reviewed all 15 `docs/`
  files; consolidated the strategy/blueprint/plan notes into a single, current
  [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md) (positioning + measured state + the
  two-truth MLIR/C/C++ placement map + what's done + the forward roadmap + next build
  steps) and **removed** the now-redundant `BCIR_STRATEGY_AND_ROADMAP.md`,
  `BCIR_LOWERING_PLAN.md`, `BCIR_BLUEPRINT.md`, `BCIR_Codex_Blueprint.md`,
  `BCIR_Full_LLVM_Build_Blueprint.md`, and `BCIR_LLVM_IR.md` (the last three were
  work-orders for the retired C++ `ir/` skeleton). Kept the normative/evidence/governance
  references (`BCIR_LANGREF`, `BCIR_STREAMPACK_ABI`, `PARITY`, `BCIR_NATIVE_OBJECT_GATE`,
  `HARDWARE_VALIDATION`, `CLANG_COMPARISON`, `BCIR_Repo_Structure`). Re-pointed the
  references in README / mlir-README / bench / test docstrings. The roadmap names the
  remaining ports: Oracle→MLIR (R14/R15 into `-bcir-verify`, the verify-cost dimension);
  Oracle→C (the StreamPack executor + encoder, the compensated-reduction kernel); and the
  new deterministic features (multi-claim bundle optimization, proof-carrying records).
- 2026-06-16: **More C23 where it pays + the native-object decision gate** (PR #224).
  (1) `_BitInt(N)` exact-width Q-fixed lane kernels (`lower.c_kernel.emit_qfixed_kernel_c`)
  -- a `_BitInt(N)` lane + `_BitInt(2N)` product accumulator, where a standard int
  promotes and a 12/20/24-bit lane has no type at all, with a preprocessor-selected C11
  fallback so the same source is bit-identical under `-std=c23`/`-std=c11`. (2) `#embed`
  frozen Q8 tables (`bcir.abi.q8_tables` -> `runtime/c/{q8_tiers.bin,bcir_q8_tables.h}`,
  nested `__has_embed` guard + fallback array, drift-gated). (3) An ETL binary-record C
  decoder (`runtime/c/bcir_binrec.c`, the `etl.binary` twin) with a libFuzzer + ASan/UBSan
  harness and a Python<->C decode-parity gate; `kbcir.fuzz` gains the `etl.binary`
  boundary. (4) The native-object decision gate (`docs/BCIR_NATIVE_OBJECT_GATE.md`, GO
  G1-G4 / STOP S1-S4) + the warranted slice: `codegen.codegen_object_c` compiles the
  emitted kernel to a real eBPF (EM_BPF) / x86-64 (EM_X86_64) object via the resident
  compiler. +19 tests (540 total).
- 2026-06-16: **The six-target capability matrix** (PR #223). All six TARGETS are now
  emitted with their own `bcir.target.capability` seeds and cross-checked on the MLIR/C++
  rail (`mlir/test/passes/target_matrix.mlir`, generated by
  `differential.emit_target_matrix` / `--emit-matrix`, drift-gated): `-bcir-plan`/
  `-bcir-overlap`/`-bcir-rcsp-plan` recompute the oracle's per-target plan / makespan+gain
  / constrained optimum **from the capability alone** -- vector_add avx512/sve/rvv 16->7808,
  avx2 8->9472, neon 4->12800, ptx 32->6976; the GPU's coalesced gather halves histogram
  (266240 vs 528384). The emitter now writes a full descriptor (`warp` + `mem_channels`
  added). The six-target parity is proven on both rails, not just Python. +11 tests.
- 2026-06-16: **Post-optimizer lowering batch -- named pipelines, the Theta context op,
  C-runtime hardening.** (1) `registerBCIRPipelines` adds named, verifier-checkpointed
  pipelines: `bcir-audit` (verify -> cost/plan/overlap), `bcir-optimize` (claims+H ->
  coupled plan), `bcir-hydrate` (plan -> StreamPack), `bcir-lower-llvm`, `bcir-aot`
  (verify -> hydrate -> LLVM). (2) New `bcir.kbcir.theta` op carries the runtime state
  into the IR; `-bcir-plan`/`-bcir-overlap` now apply the multiplicative thermal
  coupling, so the C++ plan matches the oracle under hot Theta (matmul hot 1159168,
  `theta_hot.mlir`) -- the cool-regime restriction is lifted; the cool corpus is
  unchanged (theta 0). (3) The StreamPack C decoder (a trust boundary) gains `restrict`
  + `[[nodiscard]]` + a frozen-ABI `static_assert` -- which caught the header struct
  being 60 bytes vs the declared 64 (reserved[22]->[26]); builds clean under C11 + C23
  and is fuzzed under libFuzzer + ASan/UBSan (`runtime/c/fuzz_streampack.c`,
  `tools/c/fuzz_streampack.sh`, 500k runs in CI) with a sanitizer smoke over a real
  pack + byte mutations. +1 test (drift gate; 510 total). Deferred: real-silicon
  calibration (needs a rig).
- 2026-06-16: **Optimizer-core C++ port COMPLETE -- step 5, plan-level RCSP.** New
  `-bcir-rcsp-plan` (`BCIRRcspPlanPass.cpp`) ports `rcsp.optimize_constrained`: the
  accumulated-budget label DP over the fused candidate columns (labels carry score +
  per-tracked-dim totals; dominance pruning + infeasible-extension cuts). A plan-wide
  cap bounds the plan's accumulated thermal/power -- it narrows one claim where a
  per-claim cap cannot: two vec16 claims (thermal 2176) under thermal<=2000 ->
  {16,8} @ 17280, <=1500 -> {8,8} @ 18944, matching optimize_constrained.
  `mlir/test/passes/rcsp_plan.mlir`. **With this the whole deterministic optimizer
  core is on the MLIR rail (C++23):** cost+fusion (`-bcir-cost-model`) -> coupled
  shortest path (`-bcir-plan`) -> overlap (`-bcir-overlap`) -> per-claim + plan-level
  constrained search (`-bcir-rcsp`, `-bcir-rcsp-plan`), all bit-exact vs the oracle.
  Next (BCIR_LOWERING_PLAN.md): named pass pipelines, a Theta context op, C-runtime
  hardening.
- 2026-06-16: **Optimizer-core C++ port, step 4 -- overlap (max,+) M(pi,Theta).** New
  `-bcir-overlap` (`BCIROverlapPass.cpp`) ports `gem/overlap.py`'s
  `price_scheduled`/`_makespan`: over the coupled plan it does the wave assignment by
  conflict, round-robin affinity bins, per-bin re-coupling against the in-bin
  predecessor, max over bins/tail, series over phases. Reproduces the oracle's
  scheduled price bit-for-bit: matmul makespan 253952 / gain 761856 (4 tile chains
  fan out over 8 domains), scan & histogram gain 0, the shared-input chain gain 5888.
  The emitter now also emits `affinity_domains`; `gem_corpus.mlir` regenerated.
  `mlir/test/passes/overlap.mlir` + a corpus cross-check (check_passes.sh). Next:
  step 5, plan-level multi-claim RCSP -- the last optimizer-core piece.
- 2026-06-16: **Optimizer-core C++ port, step 3 -- the layered min-plus shortest path
  (the full optimize() in C++).** New `-bcir-plan` (`BCIRPlanPass.cpp`) runs the coupled
  tropical shortest path over the fused candidate columns (shared cost/fusion logic now
  in `BCIRCostModel.h`), each edge coupling `_context_factor`'s path-based shared-input
  fusion. It reproduces the oracle's `optimize` bit-for-bit on every module: 7808
  (vector_add), 13696 (shared-input chain, `plan.mlir`), and the corpus -- matmul
  1015808 / scan 101888 / histogram 1595520. The emitter (`to_mlir`) now emits a
  registry + capability so the law plans from first principles; passes are scoped per
  bcir.module. The per-claim argmin `-bcir-select-realization` is now subsumed by the
  coupled `-bcir-plan` for multi-claim. Next: step 4 overlap (max,+), step 5
  plan-level RCSP. See `BCIR_LOWERING_PLAN.md`.
- 2026-06-16: **Optimizer-core C++ port, step 2 -- fusion / deforestation / CSE.**
  `-bcir-cost-model` now processes claims in (phase, declared) order with
  value-numbering + a produced-rid set and applies the two intra-phase redundancy
  credits -- producer->consumer deforestation (x0.75 memory) and CSE (compute zeroed,
  copy-priced memory) -- matching the oracle's `fused_candidates` bit-for-bit
  (7808 / 5888 / 5100) and annotating `kbcir.cm_fusion`.
  `mlir/test/passes/cost_model_fusion.mlir`. Next: step 3, the layered min-plus
  shortest path with `_context_factor` (the full `optimize` in C++).
- 2026-06-16: **The keystone C++ port -- `-bcir-cost-model` (the K_BCIR cost algebra).**
  `mlir/lib/passes/BCIRCostModel.cpp` recomputes each claim's candidate set + 12-d cost
  vectors from `bcir.claim` + `bcir.target.capability` (a faithful C++23 port of
  `cost.py::_cost` / `realize.candidates_for` / `_stride_penalty`, constexpr tier table +
  seeded constants read off the capability, which gained
  mem_unit/base_overhead/thermal_density/power_density/per_op_heat/elem_bytes defaulted
  to the CPU seeds). Reproduces the oracle bit-for-bit -- vec16 @ 7808 (compute 64,
  memory 3840), gather @ 528384, tile @ 126976 -- from the claim graph alone, so the
  law no longer trusts emitter-baked path costs. `mlir/test/passes/cost_model.mlir` +
  a cross-check on `full_vec_add_ct1.mlir`, gated by `check_passes.sh`. Next:
  fusion/CSE (step 2) then the layered min-plus shortest path (step 3) -- see
  `BCIR_LOWERING_PLAN.md`.
- 2026-06-16: **De-monolithed the C++ pass library + the reformulated lowering plan.**
  The 1.5k-line `mlir/lib/BCIRPasses.cpp` is split into one TU per pass group under
  `mlir/lib/passes/` (`BCIRVerifyPass`, `BCIRPromotePass`, `BCIRConvertToLLVM`,
  `BCIRGEMPasses`, `BCIRSelectPass`, `BCIRRcspPass`) sharing `BCIRPassSupport.h`;
  `BCIRPasses.cpp` is now registration-only (factory-callback `registerPass`). Builds
  clean at C++23, all passes/ODS/IRDL validate, every flag still registered -- a far
  better CI module than a single script. New `docs/BCIR_LOWERING_PLAN.md` reanalyzes
  the oracle vs the MLIR/C/C++ rails, sets the two-truth placement, the C23/C++23/26
  modernization map, and the ordered port plan -- the immediate next build step is
  `-bcir-cost-model` (the K_BCIR cost algebra on the MLIR rail, the keystone that lets
  the law recompute costs instead of trusting emitter-baked path costs).
- 2026-06-16: **MLIR toolchain in place + the optimizer core starts porting to C++23.**
  The dev toolchain (mlir-18-tools / libmlir-18-dev / llvm-18-dev) builds `bcir-opt`
  locally; every pass validates, including the generated `gem_corpus.mlir` (the C++
  `-bcir-select-realization` recomputes the oracle's per-claim scores for the widened
  corpus -- the prior Python work confirmed against the real law). The dialect now
  builds at **C++23** (renamed the `bcir.opt.*` `$requires` attr -> `$require`, a C++20+
  keyword that blocked the standard bump). First optimizer-core port: **`-bcir-rcsp`**
  (`BCIRPasses.cpp`) ports `kbcir.rcsp` -- the budget-feasible label-DP argmin + the
  Pareto front over (score, thermal, power); reproduces 9472 under the 700 cap and the
  size-2 {vec16, vec8} front (`mlir/test/passes/rcsp.mlir` + a `gem_corpus` cross-check,
  gated by `check_passes.sh`). Remaining C++ ports: the cost model on the MLIR rail
  (enables overlap (max,+) + fusion/CSE recomputation). PMU/RAPL/DVFS real-silicon
  calibration is explicitly deferred ("do later").
- 2026-06-16: **Next-steps phase 2 -- real-silicon calibration path, R14-R16
  verifier + verifier differential, trust-boundary fuzzing, the overlap conformance
  net.** (1) `bcir.silicon` reads real RAPL package energy + on-die thermal and
  `kbcir.calibloop.measured_replan` (`MeasuredReplanCertificate`, CLI
  `bcir.run --silicon`) closes CT4's software path -- measured telemetry trains a
  frozen `LinearCalibrator`, replans, certifies the win, provenance-tagged
  real-vs-synthetic (honest degrade in a sandbox; lights up on a bare-metal rig).
  (2) `verify.{verify_cim,verify_dvfs,verify_allocator,verify_smart_lowering}` add
  R14-R16 to the Python verifier (dual-rail with `-bcir-lower-to-llvm`), plus a
  verifier differential (`gen_illegal_module` + `run_verifier_campaign`). (3)
  `kbcir.fuzz` fuzzes the trust boundaries seeded by `gen_module`. (4)
  `kbcir.differential.check_overlap` nets the (max,+) scheduled-price law for the
  pending C++ optimizer-core port. +26 tests (509 total). The C++ port of
  RCSP/Pareto/overlap/fusion/CSE and a measured replan on real silicon remain the
  toolchain/hardware-gated next steps.
- 2026-06-16: **Generated, adversarial Python↔MLIR differential testing + the
  widened corpus.** `bcir.kbcir.differential` turns the parity contract into a
  proof: a structured/adversarial `gen_module`, an independent `law_select`
  (per-claim min-plus argmin, mirroring `-bcir-select-realization`), a `check_module`
  diff (selection / per-claim+total score / RCSP budget feasibility / schedule
  order), a `shrink`er, and `run_campaign` over the six targets × Θ × policy.
  `lower.mlir.to_mlir` emits the GEM-pipeline IR from any oracle plan (the
  Python→law bridge), frozen for the corpus in `mlir/test/passes/gem_corpus.mlir`
  (drift-gated; recomputed by real `bcir-opt` when present). The corpus is widened
  past toy kernels: `examples.{matmul_tiled, scan, multi_histogram}` (real blocked
  matmul / multi-stage scan / map-reduce histogram), with per-target parity pinned
  across all six TARGETS. Closes quoted roadmap #2 (symmetric cross-validation) and
  #3 (widen the op surface). +15 tests (483 total).
- 2026-06-07: Reorganized into `bcir/` (oracle) + `mlir/` (law); retired the
  legacy C++ `ir/` tree (`docs/BCIR_Repo_Structure.md`).
- 2026-06-12: Rewrote against the post-reorg tree; verifier R1–R12 completed on
  both rails.
- 2026-06-14: Refreshed for Phases 13–26 (learning/intelligence organs), R13, the
  oracle optimization pass (recursive-planning overhead removed; hot/cold locked),
  and the MLIR-native GEM pipeline passes cross-checked against the oracle. Added
  `docs/BCIR_STRATEGY_AND_ROADMAP.md`. Closed the calibration loop
  (`kbcir.calibloop`, R13) and added the portable **C23 kernel backend**
  (`lower.c_kernel`, R12: `verify.verify_c_lowering`).
- 2026-06-15: MLIR-side R7 reduction-write parity (+ `gather_reduce_ct1.mlir` and
  the reduction test pair); the strided gather-avoidance (`saxpy_strided`,
  ~1.4×); `scan_chain` (serialization) + the fusion discount + per-target parity;
  the trained calibrator (`FrozenCalibrator`) + live `Broker` (Kafka bridge); and
  the multi-version LLVM CI matrix (LLVM 18 + 19, both gating — the `SymbolTable`
  forward-compat sweep on the six container ops landed, so 19 is green not
  informational; the training MLIR grader now grades against the matrix's
  `LLVM_SUFFIX` major instead of a hard-pinned 18, and the 18-calibrated corpus
  grades clean on 19).
- 2026-06-15: **Performance audit** (strategy doc §6). Found the simplest-process
  tax was fixed *import* overhead — planning a kernel eagerly loaded the whole
  Phase-13..26 research stack + GEM executor. Made `bcir.kbcir` / `bcir.lower` /
  `bcir.gem` import lazily (PEP 562; public API unchanged): `bcir.api` cold import
  −33%, `bcir.kbcir` −49%. Added structural perf guards (`bcir/tests/test_perf.py`)
  so the heavy stack stays unloaded on the plan→emit path. Documented the
  elementwise loop-form finding (bandwidth-bound, measured-neutral; the width cap
  is a load-bearing thermal throttle) and corrected the `bench.py` narrative; the
  width-aware C codegen + R12 refinement is a tracked follow-up.
- 2026-06-15: **Width-aware C lowering + R12 refinement** (strategy §6.2). The C
  backend used to cap at the selected width *unconditionally* — even at the full
  hardware lane, which let the planner override the compiler's isel. Now the width
  is a *floor* at the full lane (idiomatic loop, `emit_kernel_c(hw_width=…)`) and a
  *ceiling* when sub-maximal (a hard cap that honors a `Theta.hot` thermal
  throttle). R12 (`verify_c_lowering`) refined to match: no sub-lane cap on a
  full-lane kernel, a mandatory cap on a throttled one. Threaded `h.vector_width`
  through api/bench/CLI/self-check. Rigorous re-measurement (separate-process,
  alternated, median-of-N) showed the prior “~12% blocked penalty” was a
  measurement artifact — loop form is measured-neutral — so this lands as a
  correctness/semantics fix, not a speedup. Dual-rail-safe (MLIR R12 checks
  StreamPack lane-segment preservation, not C structure; segment width unchanged).
  +5 tests (372 total).
- 2026-06-15: **Adaptive "smart" layer** (strategy §7) — 8 intent-aware
  capabilities, each deterministic, opt-in (never on the default plan/emit/import
  path), and gains-only (tested no-op fallback): RL allocator / smart malloc
  (`kbcir.allocator`), compute-in-memory dispatch (`gem.cim` + `LaneSegment.dispatch`),
  persistent e-graph with telemetry pivot (`kbcir.egraph.ResidentEGraph`), JIT shape
  specialist (`lower.specialist`), active uncertainty-gated telemetry
  (`kbcir.sensing`), zero-copy telemetry ring (`telemetry.TelemetryRing`), fuzzy/
  continuous MoE routing (`kbcir.moegate.route_fuzzy`/`harden` + `FrozenGate.distribution`),
  and phase-aware DVFS (`gem.dvfs`). New organs are lazy (the `test_perf` guard
  asserts they stay unloaded on the simple path — `bcir.api` cold import unchanged).
  MLIR-law parity for these is future work. +48 tests (420 total).
- 2026-06-15: **Measured wiring to real signals + first MLIR parity law for the
  smart layer.** New `bcir.silicon` probes read the real machine (read-only,
  honest): `/sys` cache topology (real L1/L2/L3 tier map for the allocator),
  cpufreq table + nominal (DVFS anchor; actuation gated on a `userspace` governor +
  privilege — reported, never faked), and `getrusage`/timers feeding the telemetry
  ring (the guest exposes no hardware PMU — `perf_event_open` ENOENT — so OS
  counters are used and that fact is reported). **Re-validated on this Xeon:** the
  zero-copy ring is **~31× faster** than JSON serialization; cache-resident access
  is **~166× lower latency** than DRAM (pointer-chase), justifying hot→SRAM. **MLIR
  parity:** the `bcir.gem.lane_segment` op gains an append-only `dispatch` attr and
  a new law **R14** (`-bcir-lower-to-llvm`: `dispatch = "pim"` legal only on a
  `reduce.*` op), mirroring `gem.cim`; built + validated on LLVM **18 and 19**, with
  positive+negative `.mlir` cases. DVFS/allocator MLIR laws follow the same pattern
  (tracked). +9 oracle tests (429 total) + the R14 MLIR law (positive+negative).
- 2026-06-15: **Privileged/bare-metal paths (attempt + degrade) + R15/R16 MLIR
  parity.** `bcir.silicon.read_hw_counters` opens real hardware PMU counters
  (cycles/instructions/cache-misses via `perf_event_open`, user-space) and feeds the
  ring with them when present; this guest exposes no PMU (ENOENT) so it degrades to
  OS counters — reported, not faked. `gem.dvfs.actuate` **attempts** to set the real
  CPU clock (`scaling_setspeed`, read back via `scaling_cur_freq`) and returns a
  dry-run `ActuationResult` naming the missing capability when there is no
  `userspace` governor / privilege (the sandbox case) — a safe no-op. **MLIR
  parity:** `bcir.gem.lane_segment.clock_q8` (append-only) + **R15**
  (`-bcir-lower-to-llvm`: clock ∈ [64,512]; a `pim` memory-bound segment must not
  overclock) mirroring `gem.dvfs`; and `bcir.resource.placement` (append-only
  `BCIR_MemTier`) + **R16** (an L1 placement ≤ 64 KiB, L2 ≤ 4 MiB; static
  `product(shape)*4`) mirroring `kbcir.allocator`. Built + validated on LLVM **18
  and 19** (positive + negative `.mlir`). New `docs/HARDWARE_VALIDATION.md` states
  the sandbox limits and the exact bare-metal rig (PMU + `intel_pstate=passive`
  userspace governor + root + RAPL) needed to measure the DVFS power-savings claim —
  which we do **not** assert until measured. +5 oracle tests (434 total).
- 2026-06-15: **Second pass over the adaptive layer — audited 8 proposed
  refinements, shipped the ones that hold gains, excluded the ones that don't.**
  Verdicts: (1) Persistent EGraph + pivot — **already DONE** (`ResidentEGraph`), no
  change. (2) Uncertainty-gated sensing — **shipped the delta**:
  `accel.FrozenRanker.confidence` (top-2 z-margin) + `sensing.sense_by_ranker`
  (a-priori gating: instrument only columns the ranker can't resolve). (3)
  Continuous routing — **excluded**: a soft distribution already exists
  (`moegate.route_fuzzy`); an execution-layer blend is redundant compute (recomputes
  the answer N×) — exploration value only, **no steady-state gain**, not shipped. (4)
  Predictive allocator — **shipped** `allocator.pool_plan`/`live_intervals`
  (liveness interval-partitioning: disjoint-lifetime tensors share an arena ⇒
  peak ≤ naive; gains-only modeled footprint win). (5) CIM/PIM partitioning —
  **shipped** `c_kernel.optimize_spatial` + `is_pim_target` (a `pim` ISA-feature
  target binds reductions to memory, modeled transport-saved; reuses the R14 law;
  real PIM emitter is next-phase). (6) JIT specialist — **measured, no gain**:
  specialist-vs-generic = generic/spec ≈ 1.0 on bandwidth-bound elementwise (same
  finding as the loop-form audit), so it is **not** wired as a perf path — kept only
  as a correctness-preserving option (`test_specialist_is_correctness_preserving_only`).
  (7) Zero-copy ring — **shipped the C side**: `memory_model.emit_ring_header_c`
  (atomic release-store producer via `hazard_to_ordering`) + `telemetry.parse_shared_ring`;
  measured C-writes/Python-reads the same mmap, no syscall/serialization. (8)
  Phase-aware DVFS — **shipped** `gem.schedule.schedule_power_rail` (per-Slot clock
  over the placed timeline; energy figure modeled — no RAPL in-sandbox). All
  deterministic, opt-in, off the simple path (test_perf guard). +13 tests (447
  total). No new MLIR laws needed (these are planning/runtime passes; the PIM
  binding is covered by R14).
- 2026-06-16: **Mined two prior-project (BDI) research notes — MPAT + AEDACI — for
  precision/accuracy ideas; shipped the deterministic, integer subset, skipped the
  bloat.** New `kbcir.precision` (opt-in, off the default plan path — pinned scores
  unchanged): the Q8-ULP error unit (`ulp_distance`); integer interval error bounds
  (`Interval`, `accuracy_bound`, `reduction_error_bound`) that give the `accuracy`
  cost dim a real producer + `meets_tolerance` (a checkable accuracy contract);
  a **compensated Q8 reduction** (`compensated_reduce_q8`, residual-carry MAC) that
  is bit-identical to the int64-exact result vs the naive accumulator's `count`-ULP
  drift — the measured numerical win; and stability diagnostics (`cancellation`,
  `condition_milli`) emitted as two-truth `Graded` signals (inform, never legislate).
  **Skipped as bloat/wrong-substrate:** the ECC/coding-theory catalogue, ML decoders
  and the novel-algorithm zoo, arbitrary-precision bignum, decimal, CORDIC,
  root-finder/optimizer/quadrature libraries, affine arithmetic, the Newton-Raphson
  condition search, and the runtime hot-patch/kernel-daemon machinery. +10 tests
  (457 total). Next-step roadmap (the MLIR/C/C++ lowering pass): a `precision=
  "compensated"` C-kernel variant + the accuracy-contract verifier law on both rails
  (strategy §8).
- 2026-06-16: **Final BDI-notes mining pass for K_BCIR/GEM (5 docs) — verdict +
  one shipped gain.** Screened all five uploads against the two engines with a hard
  filter (deterministic integer, IR-level/plan-time, gains-only). Result: **four
  yield nothing** — `paradigms_and_concepts` (philosophy/survey), the `BDI SRS`
  (requirements prose), the `AI-Trainer` (a float toy-VM + symbolic curriculum), and
  `PrimeDivisor_Tools` (symbolic-AI; its divisor-lattice tiling idea tests **neutral**
  — the ceil cost model already prefers wider-with-remainder, so divisor-alignment
  gives no gain). `execution_and_compiler_research` is ~95% runtime/backend; its
  "strength reduction" is exactly what BCIR delegates to LLVM. BCIR already has the
  rigorous versions (min-plus, RCSP, Pareto, (max,+) overlap, EFT/HEFT, affinity,
  bandwidth-knee). **The one real gap it named** — *deforestation / producer→consumer
  fusion* — is now shipped: `realize.fused_candidates` bakes a memory discount into a
  consumer claim that reads an operand a prior same-phase claim produced (the
  intermediate never round-trips), applied **uniformly** across optimize / RCSP /
  overlap / accel / softdp so `makespan ≤ serial` and the unbounded-RCSP == optimize
  invariants hold. Measured **~12% lower plan score** on a producer→consumer chain,
  no width churn, pinned 7808/9472 intact. Oracle-only (no `mlir/` change; the MLIR
  select reproduces 7808 for single-claim vector_add unchanged). +3 tests (460 total).
- 2026-06-16: **Full optimizer-completeness audit (10 classic data-flow opts) + CSE
  shipped.** Audited K_BCIR/GEM for every classic optimization the cost model should
  capture but doesn't (the producer→consumer-fusion miss suggested others). Verdict:
  **CSE / duplicate-claim elimination was MISSING and is now fixed** — the egraph
  already detected the liked-pair (`module_exprs`/`shared_blocks`) but it never
  reached the plan cost. `realize.fused_candidates` now value-numbers each claim
  `(op, operand-versions)`; a repeat within a phase is priced as a *copy* (compute
  zeroed, memory scaled to the copy fraction), and a write between duplicates bumps
  the operand version and soundly withdraws the credit. ~15% cheaper on a duplicate,
  applied uniformly across all five rails, 7808/9472 + makespan≤serial intact, +4
  tests (464 total). The rest, assessed and **not** shipped (honest): **DCE** —
  blocked (no `Resource.is_output` flag, so "written but never read" is the program's
  result, e.g. C in vector_add); **in-place/WAR** — no gain in the traffic-based
  memory cost (`x=x+y` streams the same 3 operands as `z=x+y`; the saving is
  allocation, already covered by `allocator.pool_plan` liveness); **residency /
  multi-hop shared-input** — order-dependent, can't be priced uniformly without
  breaking makespan≤serial / overlap_gain==0 (or it over-credits past cache
  eviction); **reduction-tree depth** — doesn't fit the atomic-claim granularity
  (intra-claim critical path isn't modeled); **algebraic-identity / const-fold /
  remat** — N/A or speculative on array kernels. Oracle-only (no `mlir/`; the MLIR
  select reproduces 7808 for single-claim vector_add unchanged).
- 2026-06-16: **Fair BCIR-vs-Clang comparative analysis** (`bcir/clang_compare.py`,
  `docs/CLANG_COMPARISON.md`). Held the compiler constant (Clang 18) and compared
  BCIR's planned realization vs the naive one, separate binaries, alternated,
  median-of-N. Verdict (Xeon @ 2.80 GHz): **MATCH** on simple dense kernels
  (elementwise 0.98x stream / 1.00x L1 -- BCIR's go-fast loop regresses nothing);
  **WIN** wherever BCIR exploits program intent Clang's backend lacks -- gather
  avoidance **6.0x**, reduction order **14.1x**, strided-vs-gather **1.33x**; plus
  budget feasibility (a correctness win). **LOSE** only where a planning layer is
  expected to: it cannot out-codegen the backend it delegates to (MATCH is the
  ceiling on pure compute) and carries a one-time ~58 ms Python import (per-kernel
  plan+emit is 0.31 ms, ~1% of the 34 ms compile that happens anyway). +4 gated
  tests (468 total). Confirms the design contract: a cost-governed access-pattern /
  feasibility planner on top of LLVM, not a faster LLVM.
- 2026-06-16: **Master lowering roadmap reconciliation** (`docs/BCIR_MASTER_ROADMAP.md`).
  Reconciled the two pasted planning notes (the original ODS/build-layer bootstrap and
  the Phase A-H post-roadmap audit) against the current tree: their headline task (the
  5 C++ GEM passes) and "next law" (R13) are **done**, validation realism is now
  **measured** (Clang comparison + LLVM 18/19 matrix + 468 tests), so the quoted
  six-item roadmap stands at ~1 done / 4 partial / 1 deferred. The doc gives the
  definitive **MLIR / C / C++ placement map** (deterministic core -> C++/MLIR + C
  runtime; graded organs stay Python and freeze to Q8 -- the two-truth line), a
  lowering audit (what emits where + gaps: compensated-C variant, real matmul/scan,
  GPU-C gather, one target end-to-end), a testing audit (gaps: generated differential
  Python<->MLIR parity, property/metamorphic, fuzzing, R14-R16 as bcir.verify fns), and
  the reconciled 0.2->1.0 ladder. Center of gravity now: generated adversarial
  cross-rail equivalence, closing calibration on real silicon, and widening the corpus.
