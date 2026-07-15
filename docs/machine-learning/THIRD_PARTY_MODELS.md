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
