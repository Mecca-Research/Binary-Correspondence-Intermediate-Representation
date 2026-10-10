# Qualification on a concrete target

One model, built from this repository, run through the whole BCIR stack on real hardware, with
every check gated and every number traceable to a commit, a recipe and an artifact digest. This
is the QUAL ladder's result: QUAL-0 mapped the pipeline, QUAL-1 made a decoder a BCIR program,
QUAL-2 executed it through GEM, QUAL-M trained the model, QUAL-LIN made GEM's executor linear in
the program, and QUAL-3 is the harness and the CI job.

The claim is deliberately narrow. It is not that the model is good -- it is small and lightly
trained -- but that the clean-slate path from a model definition to tokens on a CPU is closed:
the model is defined, trained and quantized by BCIR's own code, compiled into a verified BCIR
program, planned by K_BCIR, hydrated by GEM into a StreamPack, carried as JSON and as ASN.1, and
executed by GEM's C executor, generating bit for bit what a conventional monolithic runner
generates from the same weights, with the memory traffic it planned.

## The model: BCIR-Docs-860K

Nothing is downloaded. [`tools/models/build_lab_model.py`](../tools/models/build_lab_model.py)
reads one recipe, [`configs/bcir-docs-860k.json`](../tools/models/configs/bcir-docs-860k.json)
(schema `bcir.lab_model.v1`, every field checked before any work), and writes a model directory
whose `model_card.json` ties each artifact to the others by SHA-256.

| Stage | BCIR component | This model |
|---|---|---|
| Corpus | `hosted.training.data` (license filter, exact dedup, salted split) | the tracked `docs/**/*.md` at the commit, one document per `## ` section: 75 files, 782 train / 41 validation sections, 1,165,559 train tokens |
| Tokenizer | `hosted.training.bpe.BytePairTokenizer` (byte fallback, NFC) | 1,024 tokens, trained on the train split's first 787,915 octets (the recipe asks for 768 KiB, in whole sections); the file is the tokenizer's canonical JSON exactly |
| Architecture | `bcir.hosted_train.v1` decoder | 4 layers, width 128, 4 query / 2 KV heads (GQA), SiLU-gated FFN 344, RoPE, RMSNorm, tied embeddings: 857,216 parameters |
| Training | `hosted.models.native` (C forward/backward, AdamW) | 1,200 steps of 8 x 256 tokens (2.46 M tokens, 22.5 min on this host), 60 warmup steps then cosine 3e-3 -> 3e-4, seed 1729; validation loss 6.946 -> 4.144 (perplexity 63.0) |
| Export | `weights_io.write_q8_decoder` | BCIRQ8, 912,840 octets, naming the checkpoint, recipe and tokenizer digests it came from |

At this size and training budget the model's text is repetitive ("The StreamPack repression is
a resident ..."). That is a property of the model, which the qualification does not grade; the
same harness takes any BCIRQ8 decoder with a card, and BCIR-TinyStories-32M
([ML roadmap](machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md) §1.5) is the next one meant
for it.

## The pipeline and its checks

[`tools/models/run_qualification.py`](../tools/models/run_qualification.py) takes the model
directory and a prompt (the card's, by default) and runs:

1. **Program.** `decoder_program` turns the greedy generation into a BCIR module: one claim per
   decoder operation per token (embed, RMSNorm, the projections, RoPE, the KV append, attention,
   the SwiGLU block, the head, argmax), each naming its resources by the decoder RID ABI. The
   module is verified under R1-R25 (`verify_all`).
2. **Plan and pack.** K_BCIR plans it for the host's target profile and GEM hydrates the plan
   into a pipelined StreamPack (double-buffered prefetches).
3. **Projections: JSON <-> ASN.1 <-> GEM.** The module as JER; the plan as its binary ABI, DER,
   OER and JER; the pack on the native wire and as DER, OER and JER. Each is decoded back, must
   equal its source and must re-encode to the same octets.
4. **Execution.** `bcir-qualify` (C, `runtime/c/bcir_qualify_cli.c`) runs the program through
   GEM's executor (`bcir_sp_execute`) with the decoder kernels (`bcir_decoder_gem.c`), and the
   same request through the monolithic C runner -- once from the native pack and once from the
   DER, converted to the native wire in C. The oracle runs the decoded DER pack through its own
   interpreter (`run_decoder_pack`) and the monolithic oracle decode, and the float checkpoint
   through the same decode.
5. **The JSON chain.** A shorter request of the same model is carried as JER, decoded, re-encoded
   as DER, converted in C and executed by GEM. Greedy decoding is prefix-consistent, so it must
   generate the headline run's first tokens.

| Gated check | What it rules out |
|---|---|
| the model artifact is the card's; the tokenizer is the one the card and the BCIRQ8 file name | qualifying a different model than the one described |
| the program verifies (R1-R25) | executing a program the laws refuse |
| every projection round-trips (or is refused by the bounded JER reader's maxima, recorded) | a lossy or ambiguous carrier |
| every Q8 rail generates the same tokens | a rail that decodes differently |
| C GEM == monolithic C, bit for bit, from the native pack and from the DER | GEM computing something else than the runner it replaces |
| every C GEM run identical | state leaking from one run to the next |
| the C rail executes the oracle's native octets | the DER path executing a different pack |
| oracle GEM == oracle monolithic, bit for bit; C within 1e-9 of the oracle | a rail drifting from the conformance oracle |
| the JSON-carried program runs bit for bit and reproduces the headline's prefix | a JSON chain that only round-trips |
| GEM within 1.5x of the monolithic runner (the ratio row, a wide band) | a dispatch cost that grows with the program |
| counted traffic == planned traffic, op for op | kernels moving other octets than the plan priced |
| simulated memory traffic present, one profile part per rail read by callgrind's grammar (`--require-callgrind`, CI) | a skipped measurement reading as a pass; a rail never measured read as zero, or two runs read as one |

The float reference's agreement is reported, not gated: it measures quantization, not the stack.

## Results

Measured on 2026-10-10 on a virtualized Intel Xeon at 2.1 GHz, 4 vCPUs, clang 18, CPython 3.11,
prompt "The StreamPack" (5 ids), 48 new tokens. The report is `bcir.qualification_report.v1`;
`gem`, `oracle` and `traffic` rows below are its fields.

**The program.** 2,900 claims in 2,276 phases over 52 positions: 1,040 `matvec`, 464 `rmsnorm`,
416 `matvec_add`, 208 each of `rope`, `kv_append`, `attention` and `swiglu`, 52 `embed`, 48
`head`, 48 `argmax`. Zero R-law diagnostics.

**The carriers** (octets; every one round-trips):

| | binary / native | DER | OER | JER |
|---|---|---|---|---|
| plan | 149,947 | 111,102 | 81,291 | 356,000 |
| pack | 588,331 | 398,374 | 313,782 | 913,010 (refused) |

The module itself is 2,803,187 octets of JER. The pack's JER is refused by the bounded JER
reader: it needs 100,001 nodes, and X.697's §4.3 profile (`jer_bounded.py`) lets a caller tighten
the maxima, never raise them. The JSON chain therefore qualifies a 12-token request of the same
model: 884 claims, 273,777 JER octets -> 118,495 DER octets -> GEM, bit for bit, and the
headline run's first 12 tokens.

**Correctness** (exact). The same 48 tokens on five rails (C GEM, C monolithic, C GEM from DER,
oracle GEM, oracle monolithic). C GEM's and C monolithic's logits share one SHA-256; the C and
oracle logits differ by 0. The float decoder generates the same 48 tokens; its logits differ from
BCIRQ8's by at most 0.064.

**Latency** (wall clock: INDICATIVE, never a gate; medians of 7 interleaved runs).

| | ms |
|---|---|
| GEM, whole request | 57.06 (min 54.84) |
| monolithic, whole request | 54.07 (min 48.85) |
| GEM / monolithic | 1.055 (gated at 1.5) |
| of which the trust boundary per GEM run (pack CRC and R10 walk) | 3.28 |
| time to first token (prompt of 5) | 8.89 |
| per generated token (median) | 0.93 |
| tokens per second over the instrumented run | 918 |
| model load / DER -> native in C | 7.23 / 7.86 |

The kernels' time is the decoder's: `matvec` 26.1 ms, `matvec_add` 12.8 ms, `head` 5.7 ms, the
rest under 1 ms each. The ratio is a property of a real model: GEM re-verifies the pack (its CRC
and the R10 walk) on every run, so on a toy decoder whose request is microseconds of kernel work
-- the one-layer, 16-wide model of `test_qualification.py` -- the same trust boundary makes GEM
about 4x the monolithic runner. The band is applied to the real model. Before QUAL-LIN, GEM's executor and the R10 walk were quadratic in the
program on hydrated packs and the same request ran 5-14x the monolithic runner; both are now
linear with identical verdicts and dispatch order
(`test_the_r10_walk_and_the_dispatch_cost_linear_in_the_claims`).

**Memory traffic.** Planned (`program_traffic`) and counted (the C kernels' counters) agree op for
op: 57,539,312 octets read and 4,236,480 written per request. Simulated with callgrind's cache
model (I1/D1 32 KiB 8-way, LL 8 MiB 16-way, 64-octet lines, fixed so reports compare across
hosts), per request: 243.6 M data references for GEM against 237.4 M for the monolithic runner
(+2.6%), 1.09 M / 1.07 M D1 misses, and almost no last-level misses -- the 0.9 MB of weights fit
the modelled 8 MiB LL. Each row is one run of one rail: bcir-qualify's callgrind build
(`-DBCIR_QUALIFY_CALLGRIND`) brackets each rail's first run with callgrind client requests, which
work alike on every architecture valgrind runs on, and the harness reads each rail's profile part
in callgrind's own format, refusing a missing, repeated or malformed part rather than reading it
as zero. No hardware counter is read: this host has no PMU
([target access](BCIR_TARGET_ACCESS.md)), and a simulated row is labelled simulated.

## Reproduce

```
python3 tools/models/build_lab_model.py tools/models/configs/bcir-docs-860k.json \
    --output-dir build/lab-model
python3 tools/models/run_qualification.py --model-dir build/lab-model \
    --output build/qualification/report.json --require-callgrind
```

The model is reproducible from the commit and the recipe: the corpus is the tracked files at the
commit (the card records `corpus_tree_dirty`), and the report records `tree_dirty` for the
checkout it ran from. CI's `qualification` job runs both on x86-64 and native AArch64 with the
training cut to 200 steps -- the gate is the pipeline, not the perplexity -- and uploads the
report and the card. `bcir/tests/test_qualification.py` holds the builder and the harness to their
contracts on a tiny recipe, refusals included.

## What is not claimed

* **One host.** The numbers above are one virtualized x86-64 host; CI adds native AArch64. A
  second physical target with hardware counters is the next qualification, not this one.
* **No silicon counters.** Cache traffic is simulated; wall times are indicative.
* **No optimality.** The plan is TMSAO-4: K_BCIR's choice, with no lower bound behind it
  ([GEM+ roadmap](research/BCIR_GEMPLUS_ROADMAP.md) §2).
* **No model quality.** Perplexity 63 on held-out docs is what 860K parameters and 1,200 steps
  buy; the harness does not grade text.
