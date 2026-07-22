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
