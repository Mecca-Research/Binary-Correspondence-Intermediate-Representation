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
