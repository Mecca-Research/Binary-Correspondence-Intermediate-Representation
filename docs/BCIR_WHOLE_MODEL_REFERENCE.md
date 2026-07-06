# BCIR Whole-Model Reference (WMR) — the open-weight ladder's rung-8 capstone (2026-07-06)

**What this is**: the design note + build plan for BCIR's **rung-8 capstone** — a single,
dependency-free, whole-model artifact that composes the already-built ladder rungs (1–7) into one
`train → export → run-standalone → verify-bit-exact` loop. It is the machinery worth importing
from [`karpathy/llama2.c`](https://github.com/karpathy/llama2.c), reshaped to BCIR's
proof-carrying disciplines. This doc is the reference for the ML/AI roadmap's §7.4 rung 8; the
per-slice work lands under the process in that roadmap.

> **The honest reframe (read first).** llama2.c's C side is **inference-only** — the training is
> PyTorch (`train.py`/`model.py` on TinyStories); the C files (`run.c` fp32, `runq.c` int8) only
> *infer*, reading a custom `.bin` checkpoint that `export.py` writes from the PyTorch weights. So
> the value for BCIR is **not** a "C trainer" — it is the compact, dependency-free, whole-model
> **inference + export + quantize + tokenize** toolchain and, above all, the discipline of
> composing a whole model into **one readable, self-contained artifact** that closes the
> train→serve loop end-to-end. BCIR already owns the numerics; what it lacks is the *composition*.

---

## 1. The overlap — BCIR already owns the reference numerics

Most of `run.c`'s forward pass is **already in BCIR, with bit-exact C kernel twins**. Importing
it would re-derive what exists — so the numerics are a **non-target**:

| llama2.c piece | BCIR equivalent (already covered) | Anchor |
|---|---|---|
| RMSNorm / RoPE / embedding / GQA attention / SwiGLU FF | `decoder_layer_reference` + the C kernel twins | `frontends/models/decode.py:253`; `runtime/c/bcir_decode.c:7-92` |
| KV-cache incremental decode | `KVCache` / `decode_with_kv_cache` (bit-for-bit vs naive) | `decode.py:310,368` |
| SentencePiece BPE tokenizer | `SpTokenizer` (dep-free `tokenizer.model` reader, score-BPE) | `frontends/models/spm.py:88,104,149` |
| Per-group int8 / Q8_0 quant *math* | `quantize_group` / `quantized_dot` (any bit width) | `kbcir/quantize.py:73,150` |
| Greedy argmax | `_argmax` | `decode.py:241` |

---

## 2. What is genuinely worth importing (four modules)

BCIR's C side is **kernels only** — no C *driver*, no weight-*write* path anywhere, no persisted
quantized runtime. Those are the real gaps llama2.c fills:

1. **The complete self-contained C inference driver** (`run.c`'s driver half). `bcir_decode.c`
   has the kernels but no projection matmuls, no FF-in-C, no logits head, no layer loop, no
   weight struct, no sampler, no CLI. The biggest gap: a *whole-model* C artifact, not a bag of
   kernels.
2. **A custom binary checkpoint format** (f32 legacy + the v2 int8). BCIR ingests safetensors
   **read-only** (`safetensors_io.py:53`) and has **no module that writes weights at all**.
   llama2.c's header-of-hyperparams + raw-blob, `mmap`-able format is a clean template for BCIR's
   first weight-export path.
3. **The end-to-end int8 runtime** (`runq.c` Q8_0: group-64 int8 weights + f32 scales, dynamic
   activation quant). BCIR has the quant *math* but `quantized.py:33` only round-trips in memory
   for drift measurement — no persisted int8 checkpoint, no quantized C inference path.
4. **Samplers** (temperature / top-p nucleus / top-k). BCIR does greedy argmax + the `TokenDFA`
   schema constraint only — probabilistic sampling exists nowhere. Small, portable, slots into
   the exact `_argmax` site.

**The BCIR value-add on import** (the same "migrate the idea, wrap it in BCIR's discipline"
pattern as the Triton and AMD analyses — nothing lands as a raw copy):

- The checkpoint format carries a **`ModelManifest` + generation tags + CRC/sha + provenance**
  (the StreamPack discipline applied to *weights*, not a bare blob) — a stale or tampered
  checkpoint refuses to load, exactly like a stale StreamPack.
- The whole-model C artifact is a **bit-exact twin gated against the Python oracle** (two-truth:
  Python `reference_decode`/`decode_with_kv_cache` is the oracle, the C artifact is the port) —
  the same prototype-then-port + `bcir_decode.c`-style parity discipline, lifted from per-kernel
  to per-model.
- The int8 export is bounded by the **R17 accuracy law** (`precision.quantization_error_bound`).
- The sampler is **seeded and replayable** (a deterministic, provenance-carried RNG stream), so
  generation stays reproducible and the sampled ids flow into the `DataDNA` frames — BCIR needs
  determinism where llama2.c uses wall-clock `rng_seed`.

llama2.c ships readable-but-unverified C; BCIR imports the *shape* and makes it proof-carrying.

---

## 3. What NOT to import

- The **hardcoded Llama-2 architecture** — `DecoderSpec` is already parameterized (Llama/Gemma/Qwen
  shape knobs, GQA, tied/untied head, SwiGLU) (`decode.py:36`).
- The **PyTorch trainer** — D1's planned-training graph (`train_graph.py`) + the AMD-roadmap
  supplement boundary (delegate real training to PyTorch/JAX) already cover training; BCIR is not
  importing a second trainer.
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
| Whole-model C inference driver | **`runtime/c/bcir_llama.c`** (new), composing `bcir_decode.c`'s kernels + the driver half | Kernels **covered**; the driver (matmuls, FF, head, loop, weight struct, sampler, CLI) is **new** |
| Binary checkpoint format + export | **`bcir/frontends/models/weights_io.py`** (new, beside `safetensors_io.py`) + a C reader in `runtime/c/` | **New** — the first weight-*write* path in the repo |
| Samplers (temperature/top-p/top-k) | **`serve.py`/`decode.py`** at the `_argmax` site, composing with `TokenDFA` | **New** — only greedy + schema-mask exist |
| Int8 end-to-end (persisted Q8_0 + int8 C mode) | **`quantized.py`** + `weights_io.py` (format) + a `runtime/c/` int8 mode, reusing `quantize.py` math + `emit_quantized_dot_c` (`c_kernel.py:240`) | Math **covered**; the persisted format + C runtime are **new** |

---

## 5. The build slices (WMR-1 … WMR-4) — ordered, each independently gated

Highest-value, most-self-contained first. Each is one PR-sized commit under the Part-IX-style
process (oracle first, C twin second, measured-then-pinned parity gate, registry-registered test):

- **WMR-1 — the whole-model C twin** (`runtime/c/bcir_llama.c` + `test_bcir_llama.c`). Compose the
  existing `bcir_decode.c` kernels + add the driver half (Q/K/V/O projection matmuls, SwiGLU FF,
  tied/untied logits head, the layer loop, a `DecoderWeights`-shaped struct, greedy argmax, a
  CLI). Gate: bit-exact (≤1e-12, the `bcir_decode` twin tolerance) against `decode.py`'s
  `reference_decode` / `decode_with_kv_cache` over a toy spec, driven like `test_decode.c`. **The
  biggest gap; needs no new formats.**
- **WMR-2 — the weight-export path** (`frontends/models/weights_io.py` + a C reader). Write
  `DecoderWeights` to a compact self-describing BCIR checkpoint (a header of `DecoderSpec`
  hyperparams + raw row-major tensors) wrapped in a `ModelManifest` + a generation tag + a
  CRC/sha; a freestanding C reader `mmap`s/loads it for WMR-1. Refusals mirror the StreamPack
  envelope (wrong-magic / newer-schema / stale-gen / tampered-digest). Optional: a
  `stories15M.bin` legacy-format *importer* for real-model interop. Gate: round-trip
  `DecoderWeights → bytes → DecoderWeights` bit-identical, and WMR-1 fed the exported checkpoint
  emits ids bit-exact vs the in-memory path.
- **WMR-3 — seeded samplers** (`serve.py`/`decode.py`). A `Sampler` (temperature / top-p nucleus /
  top-k) over a deterministic, replayable RNG stream, composing with the `TokenDFA` mask; sampled
  ids carried in the `DataDNA` provenance so a generation is reproducible from `(seed, plan)`.
  Gate: `temperature=0` ≡ greedy `_argmax` bit-for-bit; a fixed seed replays identically; the
  same sampler ports to WMR-1's C driver with matching ids.
- **WMR-4 — int8 end-to-end** (`quantized.py` + `weights_io.py` + a `runtime/c/` int8 mode). The
  persisted Q8_0-style format (group-quant int8 weights + f32 scales) + an int8 C inference path
  reusing `quantize.py`'s group math and the exact-integer dot; R17-bounded drift. Gate: the int8
  C path's logits within the R17 accuracy bound of the f32 path; the drift matches
  `quantized.py`'s `DriftRecord`. **Feeds the AMD roadmap's MXFP/NF4 migration and the Triton
  analysis's MXFP migrate-idea — one quant format, many consumers.**

---

## 6. The larger implication — closing the train → export → serve loop

BCIR already has every *piece* of a whole-model pipeline — D1 training graphs (`train_graph.py`),
the decode rail (rungs 3–6), the quant rail, the tokenizer rail, the manifest rail, StreamPack —
but has **never composed them into a single, dependency-free, whole-model artifact with an
end-to-end round-trip gate.** It proves parity *per kernel*; llama2.c proves it *per model*. The
WMR capstone gives BCIR the loop it is missing:

```
   train                export              run standalone           verify
 (D1 planned graph   →  weights_io       →  bcir_llama.c          →  bit-exact vs
  OR a delegated         (BCIR ckpt +        (whole-model C,          decode.py oracle
  PyTorch trainer)       manifest/prov)      no deps, mmap ckpt)      (two-truth port)
```

You cannot credibly claim a "training universe" until you can **train, export, and run the
*result* standalone and prove it matches** — that closed loop is exactly what llama2.c
demonstrates in 700 lines, and it is the one thing the ladder is missing. The WMR is where D1
(training) and the decode/serve rail (inference) finally *meet* in a single verifiable deliverable.

---

## 7. Coherence with the rest of the system

The WMR is a **composition capstone**, so it deliberately reuses — and ties together — machinery
from the other roadmaps, importing no new subsystem:

- **Open-weight ladder (rungs 1–7)** — the WMR *is* rung 8: it composes manifest ingestion (1),
  the tokenizer (2), reference decode (3), the quant artifact (4), the C/MLIR law rail (5), the
  serving endpoint (6), and paged-KV/scale-out (7) into one artifact. Rung 9 (fine-tune/adapt)
  builds *on* it.
- **D1 training rail** — the WMR consumes D1's (or a delegated trainer's) weights through the new
  `weights_io` export path; it does not add a trainer.
- **Resident-compiler gate** — `bcir_llama.c` is static C built by `bcir-cc`/`cc`; the WMR never
  hand-rolls codegen. It is the *reference* artifact, and later a GPU rail (AMD roadmap) can host
  the same whole-model shape with real Matrix-Core kernels behind the gate.
- **AMD roadmap + Triton analysis** — WMR-4's persisted int8/Q8_0 format is the same quant-format
  family as the AMD roadmap's MXFP4/6 + NF4 migration and the Triton analysis's MXFP migrate-idea;
  building it once here gives all three a shared, R17-bounded format.
- **Two-truth + provenance** — the C artifact is the *port*, the Python decoder is the *oracle*;
  the checkpoint carries a `ModelManifest` + generation tags; the sampler is seeded/replayable —
  every imported piece lands proof-carrying, never as an unverified copy.
