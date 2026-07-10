# BCIRQ8 v1 compact decoder format

BCIRQ8 is BCIR's deterministic, weight-only signed-int8 format for the reference
Llama/SwiGLU decoder. Python writes and validates it through
`bcir.frontends.models.weights_io`; the portable C loader is
`runtime/c/bcir_q8_model.{h,c}`. The standalone `bcir-llama` executable consumes
verified token IDs, not raw text.

## Wire contract

- Little-endian; version 1; group size 32 in the real-model gate.
- Signed int8 codes in the canonical symmetric range `[-127, 127]`.
- One signed int16 power-of-two exponent per group; a value is reconstructed as
  `ldexp(code, exponent)` and accumulated in deterministic double-precision loops.
- Fixed 224-byte header with model/tokenizer geometry, RoPE base, RMSNorm epsilon,
  directory/payload/file offsets, CRCs, and checkpoint/config/tokenizer SHA-256 values.
- Fixed 48-byte tensor directory entries with tensor ID, layer, rank, two dimensions,
  element/group counts, tensor CRC, and exponent/code offsets.
- Canonical order: embedding; nine tensors per decoder layer; final norm; and the
  optional untied LM head.
- Every payload starts on an eight-byte boundary. Readers reject unknown, missing,
  duplicated, reordered, overlapping, unaligned, truncated, or CRC-invalid data.
- RoPE inverse-frequency buffers are validated during safetensors ingest and then
  reconstructed from `rope_base`; they are not persisted.

The Python writer uses an atomic replace and produces byte-identical output for the
same inputs. The format stores only weights; KV state remains runtime memory.

## Interfaces and gate

```python
write_q8_decoder(path, spec, weights, group_size=32,
                 source_hashes=hashes, tokenizer_ids=ids)
spec, weights, metadata = read_q8_decoder(path)
```

```c
int bcir_q8_model_load(const char *, bcir_q8_model *, char *, size_t);
void bcir_q8_model_free(bcir_q8_model *);
int bcir_llama_generate_greedy(const bcir_q8_model *, const int32_t *, size_t,
                               size_t, int32_t *, double *);
```

Run `python tools/models/run_real_model_gate.py`; add `--offline` to require a
previously verified cache. The gate publishes only `build/model-gate/parity-report.json`.
Checkpoint, tokenizer, logits, executable, and derived BCIRQ8 files remain local build/cache
products and must not be committed. See [`THIRD_PARTY_MODELS.md`](THIRD_PARTY_MODELS.md).
