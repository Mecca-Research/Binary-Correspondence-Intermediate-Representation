# Python ↔ MLIR parity contract

The Python package `bcir/` is the **executable conformance oracle**; the MLIR
dialect family under `mlir/` is the **law**. They must agree. This file is the
cross-map and the invariants that keep them in lockstep.

## Enum value parity (normative)

Integer enum values are identical in `bcir/model/lanes.py` /
`bcir/kbcir/cost.py` and `mlir/include/BCIR/BCIRAttrs.td`:

| Enum | Values |
|---|---|
| Lane | U=0, UX=1, T=2, GGG=3, A=4, H=5 |
| StrideClass | Scalar=0, Unit=1, Strided=2, Cacheline=3, Tile=4, Random=5 |
| Domain | RAM=0, VRAM=1, NVM=2, MMIO=3, CXL=4, HBM=5 |
| HazardMode | Unique=0, Atomic=1, Barriered=2 |
| Verify | None=0, Bounds=1, Exact=2, Hash=3 |
| Bounds | Strict=0, Masked=1, AssumedSafe=2 |
| MemTier | L1=0, L2=1, L3=2, DRAM=3, HBM=4, CXL=5, SSD=6 |
| Access | Flat=0, HAM=1 |

The K_BCIR cost vector is **12-d** in both, same dimension order:
`compute, memory, fabric, sync, compile, thermal, power, reliability, security,
accuracy, contention, verification`.

## Concept parity

| Concept | Oracle (`bcir/`) | Law (`mlir/`) |
|---|---|---|
| target descriptor H | `kbcir.cost.TargetProfile` (alias `HProfile`) | `bcir.target.capability` |
| memory hierarchy | `kbcir.cost.Tier` / `MemoryHierarchy` | `bcir.mem.tier` / `bcir.mem.ham` / `bcir.mem.cxl_swap` |
| resource | `model.graph.Resource` (`access`, `priority`) | `bcir.resource` |
| claim | `model.graph.Claim` | `bcir.claim` |
| phase DAG | `model.graph.Phase` | `bcir.phase` |
| cost vector | `kbcir.cost.CostVector` (12-d) | `#bcir.costvec<...>` |
| policy / weights | `kbcir.weights.Policy` | `bcir.kbcir.policy` |
| candidate path | `kbcir.realize.Candidate` | `bcir.kbcir.path` |
| min-plus select | `kbcir.realize.optimize` + `semiring` | `bcir.kbcir.select` (`#bcir.semiring<min_plus>`) |
| StreamPack | `gem.streampack.StreamPack` | `bcir.gem.stream_pack` |
| lane segment | `gem.streampack.LaneSegment` | `bcir.gem.lane_segment` |
| verifier R1–R12 | `verify.verify` | `bcir.verify.*` |
| lowering (AOT) | `lower.llvm` (clang) | `bcir.target.lower_contract` |
| concurrency/affinity (CT2) | `gem.schedule_concurrent` | `bcir.gem.lane_segment` `affinity`/`unroll` |
| ROP/MAP front-ends (CT3) | `frontends.{rop,map}` | `bcir.parse.*` / `bcir.binary.*` |
| data-DNA telemetry (CT4) | `telemetry.DataDNA` + `kbcir.calibrate` | `bcir.trace.data_dna` |
| JIT (CT5) | `lower.jit` (lli) | per-target `bcir.target.lower_contract` |
| StreamPack ABI (Phase 7) | `abi.streampack_abi` (v1) | `runtime/c/bcir_streampack.h` |
| WASM (Phase 7) | `lower.wasm` (clang→wasm + node) | per-target `bcir.target.lower_contract` |
| stackify (Phase 7) | `lower.stackify` (→ wasm/jvm/cil) | foundation for `bcir.target.lower_contract` encoders |
| C runtime (Phase 8) | `runtime/c/bcir_runtime.{h,c}` decodes `abi.streampack_abi` | `runtime/c/bcir_streampack.h` |
| async tokens (Phase 8) | `gem.async_tokens` (fork/await plan) | `bcir.async.fork` / `bcir.async.await` (`!bcir.token`) |
| memory model (Phase 8) | `lower.memory_model` (hazard→ordering) | `BCIR_MemOrdering` + barrier `ordering` → `llvm.fence` |
| per-target codegen (Phase 9) | `codegen.*` (llc → ARM/RISC-V/PTX/eBPF/C) | `bcir.target.lower_contract` (one per target) |

## Worked-example parity

`vector_add` (n=1024) on the AVX-512 profile under cool Θ selects the `vec16`
realization with K_BCIR **score = 7808** in both:

- Oracle: `python -m bcir.run vector_add --target x86_avx512` (and
  `bcir/tests/test_kbcir.py::test_vector_add_cool_selects_vec16_score_7808`).
- Law: `mlir/examples/full_vec_add_ct1.mlir` `bcir.kbcir.select ... score = 7808`.

A hot Θ replans both to `vec8` (AVX-512 downclock). Per-target π* differs by
lane width: x86_avx512→16, x86_avx2→8, arm64_neon→4, nvidia_ptx→32.

## How parity is enforced today

`bcir/tests/` pins the exact scores and per-target widths (17 checks, runnable
with `python -m bcir.tests.run_all`, no third-party deps). When the MLIR
toolchain is available, the `mlir/examples` + `mlir/test/irdl` corpus round-trips
through `bcir-opt` / stock `mlir-opt` and must carry the same constants.
