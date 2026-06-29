# Unified heterogeneous runtime — hardware channels

BCIR is built to orchestrate **any mix of hardware** in one system — an x86 host, an ARM (aarch64)
host, a RISC-V core, a GPU, an FPGA, an NVMe near-storage engine, an HBM/PIM module — and have them
**work together**, while decomposing every backend to the **same K_BCIR plan** and the **same GEM
binary-graph (StreamPack) execution**. You should be able to throw mixed hardware into a single tower
and BCIR orchestrates it.

This document describes the **hardware-channel** architecture that makes that work, and how to add a
new backend.

## The problem it solves

Per-architecture specifics — the `perf_event_open` syscall number (298 on x86_64, 241 on aarch64),
the energy/thermal sensor paths (Intel RAPL vs ARM hwmon), the codegen triple, the realizable lane
widths (vec16 on AVX-512 vs vec4 on NEON) — used to be **scattered** as inline `platform.machine()`
checks across the cost model, the silicon layer, and codegen. That is why x86 and ARM rules kept
**colliding** (e.g. the x86 perf syscall silently invoked on ARM; an unrealizable 16-wide tile priced
on NEON). The fix is to **isolate** each backend's rules behind one uniform interface.

## The abstraction (`bcir/channels.py`)

A **`HardwareChannel`** bundles *everything* hardware-specific about one backend, isolated so two
channels never collide:

| Field | What it isolates |
|---|---|
| `profile` | the **K_BCIR cost model** (lane widths, ISA features, penalties) — the *unified* input the optimizer reasons over (the only part the planner sees) |
| `kind` | the hardware class: `cpu` / `gpu` / `fpga` / `accelerator` / `storage` / `memory` — lets the orchestrator route suitable work |
| `llvm_triple`, `e_machine` | the **real** codegen identity (a genuine LLVM triple + ELF machine), for the AOT / native-object path |
| `runtime` | the host hooks: the `perf_event_open` syscall number, how Joules are read (`rapl`/`hwmon`/`none`), which thermal zones name the backend |

A **`RuntimeChannel`** holds the host-runtime specifics; the silicon layer reads its perf syscall
from the host channel — **one source of truth**, so the #255-class collision is now structural.

### The registry

```
CHANNELS = {
  # real arch backends (host-ELF CPUs + the GPU), wrapping the pinned K_BCIR target profiles:
  x86_avx512, x86_avx2, arm64_neon, arm64_sve, riscv_rvv, nvidia_ptx,
  # modeled future backends (no resident driver yet — the architecture makes room for them):
  fpga_systolic (kind=fpga), nvme_stream (kind=storage), hbm_pim (kind=memory),
}
```

`host_channel()` picks the channel matching `platform.machine()` + CPU features (the arch-neutral
default for a single-arch run). `register_channel(c)` adds a backend — the **single extension point**.

## The unified core — every channel plans the same way

Regardless of hardware origin, a module decomposes to a K_BCIR plan via the **one** optimizer on the
channel's `profile`:

```python
optimize(module, channel.profile, theta)   # the same min-plus shortest path, any channel
```

Realizable widths are per channel (NEON tops out at vec4; AVX-512 at vec16) — isolated in the
profile, so no unrealizable width leaks across architectures.

## Heterogeneous orchestration — one binary graph across the tower

`orchestrate(module, tower, theta)` plans a module across a **list of channels** (the tower). For each
claim, among the channels whose *kind* suits it, it picks the one whose K_BCIR cost is lowest — so a
large reduction lands on the **HBM/PIM** module, a tiled matmul on the **FPGA** or **GPU**, a gather on
the **GPU**, and control on the **CPU**, all in one plan. Every placement is the **same K_BCIR
arithmetic** on that channel's profile.

```python
from bcir.channels import CHANNELS, orchestrate, apply_channels
from bcir.gem import hydrate
tower = [CHANNELS[n] for n in ("x86_avx512", "nvidia_ptx", "fpga_systolic", "hbm_pim", "nvme_stream")]
plan = orchestrate(module, tower, theta)          # per-claim (channel, realization, cost)
pack = apply_channels(hydrate(module, result), plan)   # ONE StreamPack, segments tagged by channel
```

The plan decomposes to a **single GEM StreamPack** whose `LaneSegment.channel` records the backend for
each segment. The executor dispatches per segment, so **one unified binary graph runs across the whole
tower** — the unified data-structure execution BCIR is built for.

## Cross-device placement cost (fabric/sync)

Picking the cheapest channel **per claim in isolation** is wrong: it charges nothing for *moving data*
when a claim lands on a different device than the claim that produced its inputs. `orchestrate` therefore
runs as a **greedy forward pass in topological claim order** and charges a deterministic, integer
**cross-device transfer** on the K_BCIR `fabric` and `sync` cost dims (this is the first real producer of
the `fabric` axis). As each claim is placed, the channel its outputs now live on is recorded keyed by the
resources it writes. For the claim being placed on a candidate channel **C**, each input resource **R** it
reads is checked against that map:

- **Cross-channel edge** (a prior claim wrote `R` on a *different* channel): charge a transfer for `R` —
  `FABRIC = (bytes(R) · 64) >> 8` where `bytes(R) = R.count · elem_bytes` (the `×0.25` Q8 factor makes the
  term **one memory-stream-equivalent** of `R` on the cost model's scale, so it is the same order as a real
  memory pass over the operand, not an invented scale), plus `SYNC = 16` (one cross-device barrier, the same
  magnitude as the `BARRIER` `sync=16` in the realizer).
- **Same-channel edge** (the producer landed on **C**): **zero** — no move is needed. This is what keeps a
  single-channel / CPU-only tower at *exactly* the oracle's single-channel score (the invariant the
  `test_a_cpu_only_tower_*` test pins).
- **Live-in** (no claim in the module wrote `R`): **zero** in v1 — the operand is assumed pre-resident on
  whatever channel first consumes it. Modelling the cost of *staging* a live-in onto a device is a scoped
  follow-up.

The per-input transfer vectors are summed, scalarized under the **same policy weights** the planner scores
compute with (so the transfer term is in the same integer units as the per-claim cost), and added to the
candidate's compute cost **before** the `min()` — tie-broken by `(burdened cost, channel name)` exactly as
before. Because a claim's transfer cost depends on where its producers already landed, the greedy forward
placement *is* the model.

This is what makes offload a **real cost-governed decision**: a claim only moves to an off-host channel
(e.g. the GPU / `sycl_spirv` or the HBM/PIM module) when its compute is enough cheaper there to pay the
host↔device transfer. A small operand feeding a much-cheaper-off-host consumer offloads; the *same* chain
with a large operand keeps the consumer on the host because the transfer outweighs the saving. The
placement is recorded on each `ChannelPlacement.transfer_cost`, with the plan-level `transfer_total` /
burdened `total_cost` (= `compute_cost + transfer_total`) exposed for inspection.

## Adding a backend (the extension path)

A new architecture is **one channel**; the optimizer, the executor, and the other channels are
untouched:

```python
from bcir.channels import HardwareChannel, RuntimeChannel, register_channel
from bcir.kbcir.cost import TargetProfile
register_channel(HardwareChannel(
    name="my_accel", kind="accelerator",
    profile=TargetProfile(name="my-accel", triple="my-accel", lane_widths=(1, 32), ...),
    llvm_triple="...", e_machine=0,                    # 0 == off-host (no ELF), e.g. a bitstream
    runtime=RuntimeChannel(perf_syscall_nr=241, energy_source="hwmon", thermal_zone_types=("soc",)),
    arch_match=("...",), modeled=True))                # modeled until a resident driver calibrates it
```

Then it participates in `orchestrate()` automatically (route suitable work via `_claim_suits_channel`,
refine as the real driver lands), and the silicon/codegen layers read its runtime/triple from the
channel. The GPU/FPGA/NVMe/HBM channels above are exactly such modeled extension points — calibrating
each (and wiring a real driver) is that channel's own future task and touches nothing else.

## Status

- ✅ The channel abstraction + registry + isolation (x86/ARM/RISC-V/GPU real; FPGA/NVMe/HBM modeled).
- ✅ The unified core: every channel plans through the one K_BCIR optimizer; realizable widths per channel.
- ✅ Heterogeneous orchestration → a tower routes claims to the cheapest suitable backend.
- ✅ Cross-device placement cost: a greedy forward pass charges `fabric` (∝ bytes moved) + `sync` (a
  barrier) for cross-channel producer→consumer edges, so offload is a real cost-governed decision (a
  claim moves off-host only when its compute savings beat the transfer). Same-channel = free; live-ins
  assumed resident in v1.
- ✅ The unified binary graph: `LaneSegment.channel` + `apply_channels` tag one StreamPack across the tower.
- ✅ Single source of truth for the runtime ABI (the perf syscall now flows from the host channel).
- ⏭ Per-channel resident drivers (FPGA bitstream, NVMe engine, HBM/PIM) + their measured calibration
  tables; the executor's real per-channel dispatch routing. Each is an isolated per-channel task.

Tests: `bcir/tests/test_channels.py` (isolation, unified core, orchestration, the extension point).
