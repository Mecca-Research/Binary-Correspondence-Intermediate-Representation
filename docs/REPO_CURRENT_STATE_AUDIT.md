# BCIR Repository Current State Audit

> Audited 2026-06-12 against the post-reorg tree (`bcir/` + `mlir/`). Earlier
> revisions of this document described the retired C++ `ir/`/`dialect/` skeleton
> (removed in the 2026-06-07 reorg, commit `7f42b81`); see the changelog at the
> bottom. The normative status tables live in
> [`BCIR_LANGREF.md`](BCIR_LANGREF.md) §15 and
> [`BCIR_BLUEPRINT.md`](BCIR_BLUEPRINT.md); this file is the honest snapshot.

## Snapshot

- Two trees implement BCIR in lockstep ([`PARITY.md`](PARITY.md)):
  - **`bcir/`** — the executable conformance oracle (pure Python, no third-party
    deps): model, K_BCIR optimizer, GEM hydration/concurrency, ROP/MAP
    front-ends, ETL, telemetry/calibration, StreamPack ABI, verifier (R1–R12),
    and lowering (LLVM AOT via clang, JIT via lli, WASM via clang+node, stackify,
    per-target codegen via llc). Conformance suite: `python -m bcir.tests.run_all`.
  - **`mlir/`** — the law: the ODS/TableGen dialect family (~104 ops), the
    compiled `bcir-opt` with `-bcir-verify` (R1–R12), `-bcir-promote-lanes`, and
    `-convert-bcir-to-llvm` (compute/barrier), plus the pure-data IRDL
    projection for stock `mlir-opt` (the portability rail). Validated on LLVM 18
    in CI (`mlir-rail-validate`).
- **`runtime/c/`** — the freestanding (no-libc) C StreamPack runtime for the
  frozen ABI v1, with a Python-encode ↔ C-decode parity gate.
- **`llvm-training/`** — a separate agent-readable LLVM/MLIR curriculum. It is
  not part of the IR and the IR does not depend on it (`AGENTS.md`).

## Confirmed strengths

1. The Python oracle runs the whole correspondence chain end to end today
   (plan → schedule → hydrate → encode → lower → run), deterministically
   (integer/Q-fixed only), with worked-example parity pinned in tests
   (e.g. `vector_add` on AVX-512 cool Θ → vec16, score 7808).
2. Verifier laws R1–R12 are runnable on both rails and negative-tested per law:
   `bcir/verify` (`verify`/`verify_plan`/`verify_pack`/`verify_lowering`) and
   the MLIR `-bcir-verify` pass (`mlir/test/passes/verify_laws*.mlir`).
3. The StreamPack ABI v1 is frozen, CRC-gated, and decoded by a freestanding C
   runtime; cross-language parity is CI-gated.
4. The dual-rail MLIR architecture holds: the compiled ODS dialect and the
   C++-free IRDL projection both validate on stock LLVM 18 tooling.
5. CI gates every push: the oracle suite, the C runtime, the LLVM-training
   corpus validators, and the full MLIR rail (tblgen, IRDL round-trip, bcir-opt
   build, ODS corpus, pass tests).

## Confirmed limitations

1. **No BCIR-native code generation.** All machine-code paths exit through the
   LLVM toolchain as subprocesses (`clang`/`llc`/`lli`/wasm-ld/node). Instruction
   selection, register allocation, and linking are not implemented; the
   `bcir.target.lower_contract` descriptors are the designated seam.
2. **The C++ rail is verifier-first, not a full compiler.** Five GEM passes are
   declared in `mlir/passes/GEMPasses.td` (classify/select/batch/schedule/lower)
   with no C++ implementation yet; the Python oracle is the reference for those
   stages. `-convert-bcir-to-llvm` lowers compute/barrier only.
3. **Cost constants are modeled, not measured.** Target profiles ship seeded
   constants; the `LinearCalibrator` learns from telemetry but no trained model
   or live broker deployment is wired into CI.
4. **The example corpus is small** (elementwise, strided, gather, tile-MACC
   skeletons). Multi-claim fusion and joint optimization are future work.
5. **LLVM version pinning is loose**: validated on LLVM 18; CMake declares no
   version constraint and no multi-version CI matrix exists.

## Recommended next milestones

1. C++ ports of the five declared GEM passes, cross-checked against the
   oracle's pinned scores (symmetric oracle ↔ law validation in CI).
2. Widen the op/example corpus (reductions, real tiled matmul, scan) and the
   per-target worked-example parity matrix.
3. Close the CT4 loop on real hardware (trained calibrator, live broker,
   measured replan wins).
4. BCIR-native instruction selection behind `bcir.target.lower_contract`,
   starting with one target end to end.

## Changelog

- 2026-05-26: (historical) pure textual LLVM backend milestone scaffolding in
  the original C++ skeleton.
- 2026-06-07: Reorganized the IR into `bcir/` (oracle) + `mlir/` (law); retired
  the legacy C++ `ir/`/`dialect/` tree and the hand-written `runtime/llvm/*.ll`
  seed. See `docs/BCIR_Repo_Structure.md`.
- 2026-06-12: Rewrote this audit against the current tree (the previous text
  described the retired C++ skeleton). Verifier laws R1–R12 completed on both
  rails; MemTier enum parity restored in the oracle (`kbcir.cost.MemTier`).
