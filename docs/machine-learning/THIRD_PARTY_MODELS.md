# Third-party model fixtures

BCIR's real-model parity gate downloads, but does not redistribute, the following
public model assets. Checkpoints, tokenizer files, and derived BCIRQ8 artifacts are
cache/build products and must not be committed to this repository.

## Maykeye/TinyLLama-v0

- Repository: `Maykeye/TinyLLama-v0`
- Immutable revision: `298338802ab94432b917bcce11382aa151aee50f`
- Declared license: Apache License 2.0
- Upstream architecture: Llama causal language model
- Training provenance stated upstream: TinyStories data
- Tokenizer provenance stated upstream: OpenLLaMA SentencePiece tokenizer

The gate pins `config.json`, `model.safetensors`, and `tokenizer.model` by exact
byte length and SHA-256 in `tools/models/model_pins.json`. It downloads only those
three files. The source checkpoint remains in the user's model cache; the derived
BCIRQ8 file remains under the ignored `build/` directory. CI may publish only the
deterministic `parity-report.json`, which contains hashes and measurements but no
model weights or tokenizer content.

Anyone redistributing the upstream files or a derived weight artifact is responsible
for including the upstream copyright notice and satisfying the Apache-2.0 license
terms. This repository's test metadata is not a substitute for those obligations.

## CUDA-LLM comparison boundary

BCIR reviewed [`MagicCoding2006/CUDA-LLM` at
`7813ea500098b7a49871492ef2e4ec1fef6dfeab`](https://github.com/MagicCoding2006/CUDA-LLM/tree/7813ea500098b7a49871492ef2e4ec1fef6dfeab)
as engineering research. The repository demonstrates from-scratch transformer training,
custom CUDA attention and low-bit kernels, static KV/CUDA-graph decode, serving, RAG, and
metrics. On 2026-07-17 its public model-health endpoint reported deployed commit
`632e83514205ad17904593e7becbf984665b4ae2` and six FP16/INT8/INT4 eager/fast variants.

The assessed source repository declares no license through GitHub and contains no `LICENSE`
file. BCIR therefore copies or vendors neither its source nor its weights. The hosted model
laboratory is an independent implementation of generally known Llama/PyTorch techniques; the
adopt/adapt/reject decision and technical boundaries are recorded in
[`BCIR_ML_AI_INTEGRATION_ROADMAP.md`](BCIR_ML_AI_INTEGRATION_ROADMAP.md#12-cuda-llm-findings-and-the-bcir-owned-32m-program).

## TinyStories dataset planned for BCIR-TinyStories-32M

- Repository: [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories/tree/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64)
- Immutable revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`
- Dataset-card license: CDLA-Sharing-1.0
- Pinned inventory: four ordered train Parquet shards and one separate validation shard in
  `tools/models/tinystories_pins.json`

The five source files total 1,000,775,442 bytes and are pinned by exact path, length, and LFS
SHA-256. The always-on gate does not download or redistribute them, and BCIR has not yet trained
the 32M model. A later
training run must retain the train/validation split, dataset notice, tokenizer provenance, and
source hashes in every published model artifact.

## Adaptive-architecture research boundary

The following repositories were cloned into a disposable external audit directory and tested only
with bounded CPU/offline probes. They are not submodules, dependencies, model fixtures, or sources
for copied code or weights.

| Project | Audited commit | Declared repository license | Bounded evidence |
|---|---|---|---|
| [`lszshu/DeepLoop`](https://github.com/lszshu/DeepLoop) | `9d86da3367214b1e760c4713dc8612d2ae518430` | Apache-2.0 | Tiny 2-physical-layer/3-repeat CPU forward and backward were finite; the repository declares Python 3.12, so a Python 3.10-only evaluation-script parse failure was recorded as an environment mismatch |
| [`ZhaofengWu/variable-width-transformers`](https://github.com/ZhaofengWu/variable-width-transformers) | `69dde8143d9a7912de353b57093d36d8788070d4` | **NOASSERTION** at repository root | Pure schedule, residual-resize, Sinkhorn, and shape helpers were exercised; CUDA/FlashAttention/lm-engine training was not built |
| [`quandao10/MPDiT`](https://github.com/quandao10/MPDiT) | `258ebda7a2e15dc99f3f3948520e3098f348dd9f` | **NOASSERTION**: source refers to a root license file that is absent | A tiny 13,224-parameter model ran finite forward/backward with faithful lightweight `timm` interfaces; token upsampling produced the expected coarse-to-fine shape |
| [`baidu/Unlimited-OCR`](https://github.com/baidu/Unlimited-OCR) | `1ab6b46b989ebf26328a968d87ce583a9650ab90` | MIT | Image/job/stream helpers were tested offline; a valid stream object with an empty `choices` array raises an uncaught `IndexError`, so BCIR does not inherit that parser behavior |

The Variable-Width and MPDiT roots do not provide a usable license grant at the audited commits, so
BCIR uses only independently implemented paper-level ideas. Unlimited-OCR's public launcher relies
on remote model code and large GPU infrastructure; neither is admitted. DeepLoop's Apache grant
would permit reuse, but BCIR still implements the small tied-depth/normalization contract
independently to keep one coherent oracle and avoid its training-stack dependencies.

The associated papers are [arXiv:2607.13491](https://arxiv.org/abs/2607.13491) (DeepLoop),
[arXiv:2606.18246v1](https://arxiv.org/abs/2606.18246) (Variable-Width Transformers),
[arXiv:2603.26357](https://arxiv.org/abs/2603.26357) (MPDiT),
[arXiv:2606.23050](https://arxiv.org/abs/2606.23050) (Unlimited-OCR), and
[arXiv:2601.08131v4](https://arxiv.org/abs/2601.08131) (ExoFormer). No paper result is
restated as BCIR-measured performance.

## Byte-native architecture research boundary

The byte-native laboratory is an independent implementation from the papers below. The associated
repositories were source-audited at immutable commits only; they are not submodules, dependencies,
training inputs, or sources of copied code, weights, tokenizer assets, or datasets.

| Research source | Audited source | License at audited source | BCIR use |
|---|---|---|---|
| [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) | [`facebookresearch/blt`](https://github.com/facebookresearch/blt/tree/9774ed4fcc78313f9f218295f3d7e4decdadf2ae) `9774ed4fcc78313f9f218295f3d7e4decdadf2ae` | CC-BY-NC-4.0 | Independently implemented local/global/local shape, causal entropy patches, and hash n-gram idea; no source, weights, or data imported |
| [Fast Byte Latent Transformer](https://arxiv.org/abs/2605.08044) | Paper specification | Paper-derived independent implementation | Joint AR/block-denoising objective, diffusion selection, local self-draft, and exact full-model verification contracts |
| [MambaByte](https://arxiv.org/abs/2401.13660) | [`jxiw/MambaByte`](https://github.com/jxiw/MambaByte/tree/5e16f780bc331d1dc9430f4258dfb247d73aff44) `5e16f780bc331d1dc9430f4258dfb247d73aff44` | Apache-2.0 | Independently implemented readable selective diagonal SSM reference; no fused scan or upstream trainer imported |
| [End-to-end learned tokenization](https://arxiv.org/abs/2602.13940) | [`SamD770/bitter-lesson-tokenization`](https://github.com/SamD770/bitter-lesson-tokenization/tree/8992678ece92f33eb572d6e5fd213ac7510a04c0) `8992678ece92f33eb572d6e5fd213ac7510a04c0` | MIT | Paper-derived score-function boundary objective and target-rate contract only |
| [GPUTOK](https://arxiv.org/abs/2603.02597) | [`venugopalkadamba/gpu-tokenizer`](https://github.com/venugopalkadamba/gpu-tokenizer/tree/d08e0bf8135bd25553a5887b81176f2347013464) `d08e0bf8135bd25553a5887b81176f2347013464` | Apache-2.0 | Measured provider/crossover/pool contract only; BCIR does not import its CUDA BPE implementation or describe it as a GPU Unicode pipeline |

The five reviewed PDF byte identities are recorded in the deterministic hosted gate report. They
are research provenance, not redistributed artifacts. In particular, the CC-BY-NC BLT source is
not mixed into BCIR's implementation. Any future source reuse requires a separate license review;
any future pretrained “byteification” requires the source model's own license, architecture,
tensor provenance, and evaluation record in addition to the contracts above.

## Sequence-interface and progressive-growth research boundary

The sequence-interface laboratory is independently implemented from seven supplied research
papers. PDF hashes below identify the exact reviewed revisions; the PDFs are not redistributed or
used as runtime/training inputs.

| Paper | Reviewed revision | Reviewed PDF SHA-256 | BCIR boundary |
|---|---|---|---|
| [Emergent Semantics Beyond Token Embeddings](https://arxiv.org/abs/2507.04886) | `2507.04886v4` | `9676a04d619f09f35dace23906a7196c876153bffb447ad37a39a98923e74310` | Fixed binary-ID interface landed; font rendering, glyph rasterization, PCA basis, and visual-code evidence remain provenance-pinned hosted work |
| [The Galaxy's Guide to the Tokenizer](https://arxiv.org/abs/2606.25610) | `2606.25610v1` | `dccc20680ac120a8916cb1091433fb2ed02fda4eaaecc6d572a54ab577bfe87a` | Multi-objective evidence/Pareto contract only; no imported tokenizer, weights, data, or claim of a universal best interface |
| [Time Series as Language](https://arxiv.org/abs/2606.09861) | `2606.09861v1` | `1777fcd9f81fce593a884e65d7b7bb7239777ccf410fadd69dc5598ba9f91143` | Independent causal prefix/float32/FSQ oracle; no UniTok model, checkpoint, GAN, corpus, or result imported |
| [Less Is More](https://arxiv.org/abs/2506.15138) | `2506.15138v2` | `a395f6aad4219a1447a32ef2edf6440f1d8b3d4d395f2deb2500a73ca2fa74e4` | Conservative candidate/scoring/segmentation contract only; no suffix-array trainer or vocabulary imported |
| [Breaking the Tokenizer Barrier](https://arxiv.org/abs/2606.09456) | `2606.09456v1` | `d9d361add05fbeb9fd338966ecd0f99282033a3160eb7f4d35e6940402d2e18f` | Exact-byte DPCA/projection and tiny hosted clipped objective; no upstream distillation code, teacher, student, rollout, or dataset imported |
| [In-Place Tokenizer Expansion](https://arxiv.org/abs/2607.15232) | `2607.15232v1` | `63f689a974302bbfc025b74b897d0b7aceb24a06be750e7117acc3cfa81eb918` | Continued-BPE proof, copy/mean rows, freeze boundary, and cost accounting only; no expanded tokenizer, model, weights, or multilingual corpus imported |
| [Growing Transformers](https://arxiv.org/abs/2507.07129) | `2507.07129v3` | `b0aeb9afd97ad498e9f23a1c5970e1b493a5f442227c5956c4eae0590692e192` | Independent active-budget dense-growth contract and tiny fixture; LoRA is cost/schedule-only and no model, checkpoint, data, or result is imported |

Two associated repositories were cloned into a disposable external directory and audited at
immutable commits. They are not dependencies, submodules, fixtures, or sources of copied code:

| Project | Audited commit | Declared license | Source-backed audit result |
|---|---|---|---|
| [`AVBochkov/PGT`](https://github.com/AVBochkov/PGT) | `4ed95fd272e126281352aea321d175cd00b9b381` | Apache-2.0 | Current tree contains tokenizer builders, n-gram data, benchmark helpers, and notebooks, including two publishing notebooks; it has no automated test suite, dependency lock, or executable progressive-growth trainer |
| [`AVBochkov/Embeddings`](https://github.com/AVBochkov/Embeddings) | `4161521d9f88a9d7fe7aa146de673bdac3f4a1a5` | Apache-2.0 | Current implementation tree is otherwise materially the same tokenizer/benchmark research surface; its Python helper compiles and notebooks parse, but no recorded outputs establish reproducible training evidence |

The repository license would permit reuse, but BCIR intentionally keeps one independent oracle and
imports no source, model, tokenizer, n-gram table, font, result, or notebook. Any later reuse needs
a new source/license/dependency review and must not silently replace the content-addressed BCIR
contracts or their differential gates.
