# BCIR Whole-Model Reference (WMR) — rung-8 implementation status (2026-07-17)

**Current status**: the greedy Q8 portion of BCIR's **rung-8 capstone is implemented**. A pinned
trained TinyLlama checkpoint and tokenizer are verified, ingested, exported as BCIRQ8, loaded by
a portable standalone C decoder, and compared with the Python Q8 oracle in an always-on parity
gate. The source checkpoint, tokenizer, and generated weights remain cache-only; CI publishes
only the deterministic parity report. See [`BCIR_LANGREF.md` §16](../BCIR_LANGREF.md#16-bcirq8-v1-decoder-artifact-contract) and
[`THIRD_PARTY_MODELS.md`](THIRD_PARTY_MODELS.md).

The optional hosted model laboratory now closes the complementary training side with a bounded
from-random-weights micro gate. A PyTorch Llama reference trains for 64 CPU steps, publishes a
pickle-free exact-resume checkpoint, exports standard Llama Safetensors, re-enters BCIR through
the strict ingest rail, and reaches the same BCIRQ8/standalone-C runtime. This proves composition,
not useful language-model scale; the pinned TinyLlama gate remains the real trained-model gate.

The payload-free assessment rung now precedes either runtime path. `TensorInventory` and
`bcir-model-assess` account for source/BCIRQ8 bytes, KV and training state, enumerate resident,
layer-streamed, and host/device split placements, and lower a selected candidate to verified
claims plus StreamPack without reading a weight payload. Its empirical prefill/decode intervals
are target-evidence records, not performance guarantees. It plans execution; it does not replace
the strict source hashes, create Q8 weights, run a large model, or lower training updates (training
state is capacity-accounted only in this rung). Prefill and decode are separate ordered templates,
and a decode repeat executes the complete template once per generated token.

This document began as the design note for importing the useful whole-model shape from
[`karpathy/llama2.c`](https://github.com/karpathy/llama2.c). It now records what landed and what
remains. Seeded temperature/top-p/top-k sampling and raw-text tokenization in the standalone C
executable remain follow-on work; the landed executable intentionally consumes verified token IDs.

> **The honest reframe (read first).** llama2.c's C side is **inference-only** — the training is
> PyTorch (`train.py`/`model.py` on TinyStories); the C files (`run.c` fp32, `runq.c` int8) only
> *infer*, reading a custom `.bin` checkpoint that `export.py` writes from the PyTorch weights. So
> the value for BCIR is **not** a "C trainer" — it is the compact, dependency-free, whole-model
> **inference + export + quantize + tokenize** toolchain and, above all, the discipline of
> composing a whole model into **one readable, self-contained artifact** that closes the
> train→serve loop end-to-end. BCIR already owned the numerics; the landed greedy Q8 capstone adds
> the export/runtime/parity composition.

---

## 1. The overlap — BCIR already owns the reference numerics

Most of `run.c`'s forward pass is **already in BCIR, with bit-exact C kernel twins**. Importing
it would re-derive what exists — so the numerics are a **non-target**:

| llama2.c piece | BCIR equivalent (already covered) | Anchor |
|---|---|---|
| RMSNorm / RoPE / embedding / GQA attention / SwiGLU FF | `decoder_layer_reference` + the C kernel twins | `frontends/models/decode.py:253`; `runtime/c/bcir_decode.c:7-92` |
| KV-cache incremental decode | `KVCache` / `decode_with_kv_cache` (bit-for-bit vs naive) | `decode.py:310,368` |
| SentencePiece BPE tokenizer | `SpTokenizer` (dep-free `tokenizer.model` reader, score-BPE) | `frontends/models/spm.py:88,104,149` |
| Per-group integer quantization *math* | `quantize_group` / `quantized_dot` (any bit width) | `kbcir/quantize.py:73,150` |
| Greedy argmax | `_argmax` | `decode.py:241` |

---

## 2. What landed and what remains

The original analysis identified four modules. Three are now present in the greedy Q8 capstone:

1. **Whole-model C inference — landed.** `runtime/c/bcir_llama.c` composes RMSNorm, RoPE,
   GQA/KV-cache attention, projection matmuls, residuals, SwiGLU, tied/untied heads, and
   lowest-ID greedy argmax. `bcir-llama` loads with portable stdio/heap APIs and accepts verified
   prompt IDs.
2. **Compact weight export — landed.** `frontends/models/weights_io.py` deterministically writes
   and reads BCIRQ8 v1. The fixed header and tensor directory carry dimensions, token IDs,
   RoPE/RMSNorm parameters, source/config/tokenizer SHA-256 values, and layered CRC protection.
3. **End-to-end int8 runtime — landed.** Group-32 signed int8 weights use one signed `int16`
   power-of-two exponent per group. Python and C retain quantized weights and use deterministic
   double accumulation for the standalone parity rail.
4. **Seeded probabilistic samplers — open.** Temperature, top-p, and top-k sampling with a shared
   replayable Python/C RNG remain WMR-3. Greedy generation is the supported capstone boundary.
5. **Hosted train-to-C bridge — landed.** `bcir.hosted.models` independently implements the
   `DecoderSpec` Llama family in opt-in PyTorch. Its micro gate trains from random weights and
   traverses safe Safetensors checkpoint/export, strict BCIR ingest, BCIRQ8, and standalone C.

**The BCIR value-add on import** (the same "migrate the idea, wrap it in BCIR's discipline"
pattern as the Triton and AMD analyses — nothing lands as a raw copy):

- The checkpoint carries **source/config/tokenizer hashes plus header, body, and per-tensor CRCs**
  (the StreamPack integrity discipline applied to weights, not a bare blob). Bounds, alignment,
  overlap, ordering, truncation, and corruption violations refuse to load.
- The whole-model C artifact is a **bit-exact twin gated against the Python oracle** (two-truth:
  Python `reference_decode`/`decode_with_kv_cache` is the oracle, the C artifact is the port) —
  the same prototype-then-port + `bcir_decode.c`-style parity discipline, lifted from per-kernel
  to per-model.
- Quantization drift and NLL delta are measured against the float oracle and emitted in the
  deterministic report; the gate never disguises format loss as C/Python implementation drift.
- The planned sampler extension must be **seeded and replayable** with a shared Python/C RNG
  stream. Until that lands, the supported lowest-ID greedy path is deterministic without RNG.

llama2.c ships readable-but-unverified C; BCIR imports the *shape* and makes it proof-carrying.

---

## 3. What NOT to import

- The **hardcoded Llama-2 architecture** — `DecoderSpec` is already parameterized (Llama/Gemma/Qwen
  shape knobs, GQA, tied/untied head, SwiGLU) (`decode.py:36`).
- The **upstream PyTorch trainer or checkpoint** — BCIR's hosted trainer is an independent,
  `DecoderSpec`-driven implementation with strict safe-checkpoint and ingest contracts. It does
  not copy llama2.c/CUDA-LLM training code or claim provenance for an external model.
- The **OpenMP matmul** — the channel + resident-compiler + K_BCIR cost model own parallelism and
  placement; a hand-pragma'd matmul is the wrong layer.
- **fp32-only** — BCIR is multi-precision; the C twins are `double` today (a nuance: bit-parity
  against *upstream* `stories15M.bin` would need a `float32` mode; BCIR's own Python↔C parity does
  not, and internal parity is the discipline that matters).

---

## 4. Where each piece lands (anchored homes)

The natural home is **`runtime/c/`** (the C-twin rail) + **`bcir/frontends/models/`** (the ingest/
export rail) — **not** the `c_kernel.py` emitter (a whole-model reference is static hand-written C
built by `bcir-cc`/`cc`, like `test_decode.c` already is, not an emitted string):

| Module | Home | New vs covered |
|---|---|---|
| Whole-model C inference driver | `runtime/c/bcir_llama.c`, `bcir_llama_cli.c` | **Landed** for greedy token-ID inference |
| Binary checkpoint format + export | `bcir/frontends/models/weights_io.py`, `runtime/c/bcir_q8_model.c` | **Landed** as BCIRQ8 v1 |
| Samplers (temperature/top-p/top-k) | Future shared Python/C sampler at the greedy argmax boundary | **Open**; greedy remains deterministic |
| Int8 end-to-end | `weights_io.py`, `bcir_q8_model.c`, `bcir_llama.c` | **Landed** with group-32 power-of-two scaling |

---

## 5. Build-slice status (WMR-1 … WMR-4)

Highest-value, most-self-contained first. Each is one PR-sized commit under the Part-IX-style
process (oracle first, C twin second, measured-then-pinned parity gate, registry-registered test):

- **WMR-1 — landed:** the portable whole-model C twin and CLI cover projection matmuls, SwiGLU,
  tied/untied logits, layer/KV-cache loops, and greedy generation. Toy tied/untied tests compare
  two generated tokens and final logits against Python Q8.
- **WMR-2 — landed:** BCIRQ8 v1 is deterministic, self-describing, compact, atomically exported,
  and validated by both Python and C readers. It uses `fopen`/`fread`/`malloc`, not `mmap`.
- **WMR-3 — seeded samplers** (`serve.py`/`decode.py`). A `Sampler` (temperature / top-p nucleus /
  top-k) over a deterministic, replayable RNG stream, composing with the `TokenDFA` mask; sampled
  ids carried in the `DataDNA` provenance so a generation is reproducible from `(seed, plan)`.
  Gate: `temperature=0` ≡ greedy `_argmax` bit-for-bit; a fixed seed replays identically; the
  same sampler ports to WMR-1's C driver with matching ids.
- **WMR-4 — landed:** persisted BCIRQ8 and standalone C Q8 inference are gated on the immutable
  Maykeye/TinyLLama-v0 revision. The gate requires token-ID parity and C-vs-Python-Q8 logit error
  at most `1e-9`, while recording float-vs-Q8 drift and NLL delta without treating them as hidden
  implementation details.
- **WMR-5 — bounded adaptive architecture laboratory:** tied-depth LoopDeepNorm, fixed-residual
  variable widths, reference-plus-sliding attention, H0/H1 ExoFormer mixing, and multi-patch image
  processing have independent dependency-free contracts and tiny hosted training probes. Their
  work lowers to verified claims/StreamPack, but they are **not** part of `DecoderSpec`, BCIRQ8, or
  the standalone C decoder. Native export waits for measured value, a format decision, and parity.

---

## 6. The larger implication — closing the train → export → serve loop

BCIR now has four complementary bounded gates. The real-model gate starts from a pinned trained
checkpoint and verifies the deployment half:

```
   source               export              run standalone           verify
 pinned checkpoint  →  BCIRQ8 export    →  bcir_llama.c          →  parity report
 + tokenizer IDs       (hashes + CRCs)     (portable C, no deps)     (Python Q8 oracle)
```

The hosted micro gate proves the owned training path without conflating it with the external model:

```
 random weights → 64-step hosted train → safe Safetensors → strict BCIR ingest
                → BCIRQ8 group-32 → Python Q8 + standalone C → parity report
```

The offline staged-training gate proves the machinery above pretraining without claiming a
large-model run:

```
 generated corpus → byte BPE → tiny safe pretrain → SFT → RM → DPO → bounded PPO
                                      ├────────────→ verified reasoning SFT
                                      └────────────→ relational embedding distillation
 generated tensors ────────────────────────────────→ MLP / GRU / encoder confirmation
```

It uses provider-neutral contracts and generated fixtures, runs twice on one CPU thread, restores
process-global RNG/determinism state around each stage, and requires an identical content-addressed
pipeline report. No live teacher API, remote executor, pretrained weight, or generated checkpoint is
committed.

The hardware-policy gate closes the first model→hardware-plan learning loop without executing a
large model or granting a network authority over legality:

```
 model header + hardware envelope → finite placement candidates → generated telemetry episodes
     → tiny GNN/Transformer reward+DPO+PPO → bounded PUCT → verified claims + StreamPack
     → exact per-bank static addresses → simulated evidence withheld from live promotion
```

It also runs twice on one CPU thread. The policy learns metric preferences over real candidate
shapes, while the fixture outcomes remain explicitly simulated. A live generation may be promoted
only from measured selected/baseline outcomes at a quiescent boundary after plan, pack, bank-move,
and static-memory re-verification.

The adaptive architecture gate explores model structure without changing the deployment claim:

```
 explicit architecture → exact sizes + analytic lower bound → tiny hosted forward/backward
        → verified claims + StreamPack → deterministic report (no weights/checkpoint committed)
```

It proves implementation consistency, not useful model quality or production speed. Variable-width
carry preserves inactive coordinates in a fixed global residual stream; R-SWA bounds the continuation
cache; LoopDeepNorm prices virtual depth; and ExoFormer anchors are normalized and gated as specified
by [arXiv:2601.08131v4](https://arxiv.org/abs/2601.08131). None may silently enter the stable Llama
train-to-C path.

CI does not claim that BCIR trained the pinned TinyLlama checkpoint, and none of the micro gates is
evidence of useful language-model quality. Together the gates prove deployment, safe owned
pretraining, staged-alignment composition, and a bounded learned plan-selection seam while keeping
their provenance and claims separate.

---

## 7. Coherence with the rest of the system

The WMR is a **composition capstone**, so it deliberately reuses — and ties together — machinery
from the other roadmaps, importing no new subsystem:

- **Open-weight ladder** — the WMR is rung 8's standalone greedy-Q8 capstone: it directly composes
  pinned-manifest ingestion, tokenization, reference decode, quantized export, and portable C
  inference. It does not claim to fold serving or scale-out machinery into the executable.
- **D1 and hosted training rails** — planned/streamed claim-graph training remains the semantic
  oracle; optional hosted PyTorch supplies scalable tensor execution. Hosted exports must pass
  strict ingestion before they can use the same artifact writer. The real-model gate starts from
  immutable upstream weights and does not claim a TinyLlama trainer.
- **Resident-compiler gate** — `bcir_llama.c` is static C built by `bcir-cc`/`cc`; the WMR never
  hand-rolls codegen. It is the *reference* artifact, and later a GPU rail (AMD roadmap) can host
  the same whole-model shape with real Matrix-Core kernels behind the gate.
- **AMD roadmap + Triton analysis** — BCIRQ8 supplies a concrete portable weight-only format and
  reference rail. MXFP/NF4 remain distinct future formats rather than aliases for BCIRQ8.
- **Two-truth + provenance** — the C artifact is the *port* and the Python decoder is the
  *oracle*. The BCIRQ8 header carries immutable source/config/tokenizer hashes and layered CRCs;
  the deterministic parity report records the exact outcome. Seeded sampler provenance remains
  part of the open WMR-3 slice.
- **Hardware-policy rail** — model placement supplies the finite action set; telemetry and exact
  K_BCIR metrics supply training evidence; verification and static memory remain the authority.
  The policy cannot turn model layers into independent agents, bypass autoregressive ordering, or
  certify a simulated speedup.
- **Adaptive architecture rail** — shape and cache contracts are dependency-free and
  content-addressed; optional PyTorch performs only bounded tensor validation. K_BCIR sees explicit
  width/carry/attention/MLP/barrier work, while unsupported combinations refuse. C/C++ lowering is
  deferred until per-shape and cache measurements demonstrate an inference benefit.
