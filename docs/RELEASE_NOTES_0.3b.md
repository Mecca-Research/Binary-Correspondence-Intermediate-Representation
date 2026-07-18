# BCIR 0.3b draft — driver-oriented C and verified runtime candidate

> **Unreleased draft.** The package and LangRef remain version `0.2.0`. There is no
> `v0.3b` tag and this file does not declare support beyond checked-in interfaces.

The 0.3b candidate is the first release rung intended to support construction of a
resident driver package on top of BCIR’s verified C/runtime substrate. Most compiler,
portability, and memory foundations are present; the release remains blocked on proving
the first direct driver lifecycle and completing the publication gate.

## Candidate baseline already landed

### Law and compiler rails

- The MLIR law rail carries R1–R23 with negative fixtures; the C frontend remains
  explicitly scoped to the laws and constructs it implements rather than claiming false
  full-law symmetry.
- Python and C C-front rails cover a broad driver-oriented C23 subset, target ABI
  selection, volatile/atomic/indirect-call/extent/call-ABI contracts, project verdicts,
  dependency output on the Python rail, fallback routing, and externally linkable
  emitted C.
- Two emitted translation units are compiled and linked by the resident compiler in the
  C runtime gate. This is not a BCIR-native linker or general ISO C23 implementation.
- `bcir-aot` remains partial MLIR preparation; Python LLVM AOT/JIT accepts one supported
  elementwise claim and rejects arbitrary graphs rather than truncating them.

### Runtime, portability, and memory

- StreamPack v1 is frozen; v2/v3 extensions are append-only and Python/C codec parity is
  adversarially tested.
- Native Windows host behavior includes canonical line endings, guarded resource
  counters, spawn-compatible test workers, platform-aware `-lm` handling, and coherent
  version-aware LLVM discovery.
- Runtime C is classified as freestanding core, hosted tools, or driver adapters.
  Hosted allocation has checked arithmetic/growth, allocator injection, idempotent
  destruction, and fail-every-allocation regressions.
- Direct RuntimeChannel v1 hooks and a bounded loopback exist. There is no Linux IPC
  adapter, stable UAPI, or resident hardware driver yet.

### Correctness and model gate

- C statement-expression bitfield typing, `TrainStepSpec` execution, final partial
  batches, optimizer state, and tied/untied decoder-head quantization behavior have
  deterministic regressions.
- BCIRQ8 v1 is normative in [`BCIR_LANGREF.md`](BCIR_LANGREF.md) §16. A pinned licensed
  TinyLlama checkpoint/tokenizer can be verified, converted to a compact Q8 artifact,
  run through Python and standalone C greedy inference, and compared in a deterministic
  parity report. Source assets and derived weights are not committed.
- The optional hosted rail includes deterministic corpus/BPE preparation and bounded
  pretraining, SFT, reward, DPO, PPO, verified-reasoning, embedding, MLP, GRU, and encoder
  confirmation gates. Provider-neutral recorded-teacher and offline-compute contracts are
  present; live API adapters and large-model training are not release claims.
- BCIRQ4T/SmoothQuant/AVX2, measured schedule artifacts, expanded closed-set AD, and
  workload-scoped numerical evidence are bounded research/reference slices. They do not
  declare whole-model Q4, multi-target performance, or hardware qualification.
- `bcir-model-assess` can inventory Safetensors headers without reading payloads, account for
  format/KV/training/per-bank memory, enumerate resident/layer-stream/host-device candidates,
  and emit a verified content-addressed claim/StreamPack plan. Its bounded reference timings are
  evidence records, not a production GPU-performance or executable large-model claim.
- The ordinary x86-64 long-mode C entry, descriptor/segment loads, and ordinary interrupt
  trampoline are typed/lowered and object/disassembly-gated. Reset transition, paranoid
  exception entry, and production feature policy remain outside this release.

## Release blockers

0.3b may be tagged only when all of the following are true:

1. **Direct driver proof:** the first UART package reaches the direct in-process
   RuntimeChannel lifecycle with simulator parity, cancellation/teardown, telemetry,
   replay evidence, and deterministic faults. Compiler-shaped UART/GPIO fixtures do not
   satisfy this requirement.
2. **Contract agreement:** the documented C subset, fallback behavior, project/link
   surface, target ABI matrix, StreamPack/telemetry contracts, and ownership annotations
   agree across their applicable rails.
3. **Complete validation:** every required oracle, C/C++, sanitizer, fuzz, MLIR/IRDL,
   training, docs, Windows, Ubuntu, and native-ARM CI job is green. Hardware-gated work
   must be labeled as such, not simulated locally without a bounded job.
4. **Release hygiene:** package metadata, LangRef, current-state audit, generated status,
   model provenance/license notices, and release notes contain no version or support
   drift; no caches, checkpoints, tokenizers, binaries, or generated Q8 weights are in
   the diff.

## Explicit non-goals

- complete ISO C23/C++ replacement;
- arbitrary-graph LLVM lowering or a BCIR-native CPU instruction selector/linker;
- stable Linux UAPI, Linux fork, native kernel, or native IPC;
- resident AMD/NVIDIA/other device drivers;
- production model serving or unqualified ARM/board performance claims.

Those programs are governed by [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md),
[`BCIR_DRIVER_KERNEL_ROADMAP.md`](kernel/BCIR_DRIVER_KERNEL_ROADMAP.md), and the
[`BCIR_NATIVE_OBJECT_GATE.md`](BCIR_NATIVE_OBJECT_GATE.md).

## Candidate validation

```bash
python -m bcir.tests.run_all --tier quick -j 2
python -m bcir.tests.run_all --tier thorough -j 2
bash tools/c/check_runtime.sh
bash tools/cpp/check_handoff.sh
bash tools/wsl/check_passes.sh
python tools/docs/gen_status.py --check
python tools/docs/check_links.py
git diff --check
```

The generated static inventory is [`STATUS.md`](STATUS.md). The complete publication
policy is [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md) §7.
