# Hardware-channel plugins

A backend (FPGA, NVMe near-storage, HBM-PIM, a new accelerator) joins the BCIR tower as a
**plugin** — a declarative `*.channel.json` manifest — **without editing the core**. The optimizer,
the GEM executor, and the other channels are untouched; the plan and the StreamPack are identical
regardless of which hardware a claim lands on. The manifest is the stable boundary.

See [`bcir/channel_plugin.py`](../bcir/channel_plugin.py) for the schema (`ChannelManifest`) and the
loader; [`bcir/channels.py`](../bcir/channels.py) for `HardwareChannel` itself. The built-in
channels round-trip through this exact format (pinned by `bcir/tests/test_channel_plugin.py`), so it
is known complete.

## The seven sections of a `channel.json`

| Section | Field(s) | Meaning |
|---|---|---|
| **identity** | `name`, `kind` | `kind` ∈ `cpu / gpu / fpga / accelerator / storage / memory`. |
| **target profile** | `profile` | The full K_BCIR cost model H — the *only* part the optimizer reasons over (lane widths, gather penalty, memory hierarchy, ISA features). |
| **codegen identity** | `codegen.llvm_triple`, `codegen.e_machine` | The real LLVM triple + ELF machine. `e_machine` is set only on a `cpu` (host ELF); off-host backends use `0`. |
| **runtime signals** | `runtime.perf_syscall_nr`, `runtime.energy_source`, `runtime.thermal_zone_types` | The host-runtime signal-provider contract (how real perf / Joules / temperature are read). |
| **execution capability** | `capabilities` | Which GEM work the channel runs (`matmul`, `tile`, `reduce`, `gather`, `data_parallel`, `stream_unit`, `scalar_stream`, or `universal`) — the data-driven routing contract. |
| **calibration artifact** | `calibration.ref`, `.digest`, `.cal_gen`, `.provenance` | A reference + content digest for the calibration data behind the cost profile (auditable). |
| **provenance flag** | `provenance`, `modeled` | `real` (measured silicon) / `modeled` / `simulated`, so the tower never mistakes a simulator's numbers for measured hardware. |

## Registering a plugin

```python
from bcir.channel_plugin import register_from_manifest, discover_plugins

register_from_manifest("channels/example_tpu.channel.json")   # one manifest (validates first)
discover_plugins()                                            # all channels/*.channel.json + $BCIR_CHANNEL_PATH
```

Discovery also picks up Python entry points in the `bcir.channels` group when the package is
installed. Invalid manifests are rejected (`register_from_manifest` raises `ValueError`); a broken
plugin never silently joins the tower.

[`example_tpu.channel.json`](example_tpu.channel.json) is a worked example: a modeled systolic
matrix accelerator the core has never heard of, declared entirely in JSON.

[`sycl.channel.json`](sycl.channel.json) is a modeled SPIR-V/SYCL GPU channel; SYCL also serves as a
*differential oracle* (a SAXPY `parallel_for` checked against BCIR's own reference) — see
[`docs/kernel/SYCL_INTEROP.md`](../docs/kernel/SYCL_INTEROP.md). SYCL is a C++ compiler mode (`-fsycl`), **not** a
`c.call.libm:` link edge (no `-l<lib>` rule).
