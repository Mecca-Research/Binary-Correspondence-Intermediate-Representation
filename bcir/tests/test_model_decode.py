"""Rung 3 of the open-weight ladder (§7.4): the REFERENCE DECODER (`frontends/models/decode.py`).

The dep-light dense-decoder reference composed from the existing oracle pieces (embedding
lookup, RMSNorm/RoPE, causal attention, feed-forward, tied-embedding logits, greedy argmax).
Pins: (1) the CAUSALITY law -- a later token can never move an earlier row, bit-for-bit;
(2) the KV-cache twin -- incremental decode emits the SAME ids as naive full recompute,
the reference-vs-realization (E3) pattern; (3) the LADDER TIE -- the same synthetic model
decodes a mini-tokenizer prompt while its shard census (rung 1) and tokenizer digest
(rung 2) pin the manifest; (4) op-level well-formedness reports lying weight shapes."""

import random
import tempfile

from bcir.frontends.models.decode import (DecoderSpec, DecoderWeights, LayerWeights,
                                          check_decoder_weights, decode_with_kv_cache,
                                          decoder_forward_reference, decoder_param_count,
                                          reference_decode)
from bcir.kbcir.unsupervised import EmbeddingTable

SPEC = DecoderSpec(vocab_size=264, d_model=8, n_heads=2, n_layers=2, d_ff=16)


def _toy_weights(spec: DecoderSpec, seed: int = 0x7E57) -> DecoderWeights:
    rng = random.Random(seed)

    def t(n):
        return tuple(rng.uniform(-0.3, 0.3) for _ in range(n))

    d, f = spec.d_model, spec.d_ff
    layers = tuple(
        LayerWeights(g_attn=t(d), w_q=t(d * d), w_k=t(d * d), w_v=t(d * d), w_o=t(d * d),
                     g_ff=t(d), w1=t(d * f), b1=t(f), w2=t(f * d), b2=t(d))
        for _ in range(spec.n_layers))
    emb = EmbeddingTable(table=t(spec.vocab_size * d), n_vocab=spec.vocab_size, dim=d)
    return DecoderWeights(embedding=emb, layers=layers, g_final=t(d))


def test_causality_a_later_token_never_moves_an_earlier_row():
    """Every stage is row-independent except attention, and attention is causally masked --
    so swapping the LAST prompt token leaves every earlier hidden row bit-identical."""
    w = _toy_weights(SPEC)
    a = decoder_forward_reference([1, 2, 3, 4], SPEC, w)
    b = decoder_forward_reference([1, 2, 3, 250], SPEC, w)
    d = SPEC.d_model
    assert a[:3 * d] == b[:3 * d]                        # rows 0..2: bit-identical
    assert a[3 * d:] != b[3 * d:]                        # the changed row itself moves


def test_kv_cached_decode_is_the_naive_decode_bit_for_bit():
    """The rung's KV primitive: incremental decode == full recompute, exactly (both paths
    run the same row arithmetic; a masked naive score is an exact trailing zero)."""
    w = _toy_weights(SPEC)
    for prompt in ([7], [1, 2, 3], [260, 259, 104, 33]):
        naive = reference_decode(prompt, SPEC, w, max_new=6)
        cached = decode_with_kv_cache(prompt, SPEC, w, max_new=6)
        assert len(naive) == 6
        assert naive == cached, (prompt, naive, cached)


def test_greedy_decode_is_deterministic_and_eos_stops_it():
    w = _toy_weights(SPEC)
    first = reference_decode([1, 2, 3], SPEC, w, max_new=5)
    assert first == reference_decode([1, 2, 3], SPEC, w, max_new=5)
    assert all(0 <= t < SPEC.vocab_size for t in first)
    # force the first emitted token to be the eos: decode must stop after emitting it.
    stopped = reference_decode([1, 2, 3], SPEC, w, max_new=5, eos_id=first[0])
    assert stopped == [first[0]]
    assert decode_with_kv_cache([1, 2, 3], SPEC, w, max_new=5, eos_id=first[0]) == stopped


def test_the_ladder_ties_manifest_tokenizer_and_decode_together():
    """Rung 1 + rung 2 + rung 3 over ONE synthetic model: the shard census's param_count
    equals the decoder's, the manifest carries the mini tokenizer's digest, and the decoder
    greedily extends a prompt that tokenizer encoded."""
    from bcir.frontends.models import build_manifest, load_tokenizer
    from bcir.tests.test_model_manifest import _shard
    from bcir.tests.test_model_tokenizer import _mini_tokenizer
    spec, w = SPEC, _toy_weights(SPEC)
    d, f = spec.d_model, spec.d_ff
    tensors = {"model.embed_tokens.weight": ("F32", (spec.vocab_size, d), spec.vocab_size * d * 4)}
    per_layer = {"input_layernorm.weight": (d,), "self_attn.q_proj.weight": (d, d),
                 "self_attn.k_proj.weight": (d, d), "self_attn.v_proj.weight": (d, d),
                 "self_attn.o_proj.weight": (d, d), "post_attention_layernorm.weight": (d,),
                 "mlp.up_proj.weight": (d, f), "mlp.up_proj.bias": (f,),
                 "mlp.down_proj.weight": (f, d), "mlp.down_proj.bias": (d,)}
    for li in range(spec.n_layers):
        for name, shape in per_layer.items():
            n = 1
            for x in shape:
                n *= x
            tensors[f"model.layers.{li}.{name}"] = ("F32", shape, n * 4)
    tensors["model.norm.weight"] = ("F32", (d,), d * 4)
    with tempfile.TemporaryDirectory() as tmp:
        tok_path = _mini_tokenizer(tmp)
        shard = _shard(tmp, "model-00001-of-00001.safetensors", tensors)
        man = build_manifest([shard], {"model_type": "toy-dense", "vocab_size": spec.vocab_size},
                             tokenizer_ref="mini", tokenizer_path=tok_path)
        tok = load_tokenizer(tok_path)
    assert man.param_count == decoder_param_count(spec)          # rung 1 census == rung 3 spec
    assert man.tokenizer_digest == tok.digest != ""              # rung 2 tie
    assert check_decoder_weights(spec, w) == []                  # the weights fit the spec
    prompt = tok.encode("<bos>hello")                            # [260, 259]
    assert prompt == [260, 259]
    new = reference_decode(prompt, spec, w, max_new=4)
    assert len(new) == 4 and all(0 <= t < spec.vocab_size for t in new)
    assert new == decode_with_kv_cache(prompt, spec, w, max_new=4)


def test_lying_weight_shapes_are_reported():
    w = _toy_weights(SPEC)
    bad_layer = LayerWeights(g_attn=w.layers[0].g_attn, w_q=w.layers[0].w_q[:-1],
                             w_k=w.layers[0].w_k, w_v=w.layers[0].w_v, w_o=w.layers[0].w_o,
                             g_ff=w.layers[0].g_ff, w1=w.layers[0].w1, b1=w.layers[0].b1,
                             w2=w.layers[0].w2, b2=w.layers[0].b2)
    bad = DecoderWeights(embedding=w.embedding, layers=(bad_layer,) + w.layers[1:],
                         g_final=w.g_final)
    msgs = check_decoder_weights(SPEC, bad)
    assert msgs and any("w_q" in m for m in msgs), msgs
    # and the spec itself validates: odd d_k (RoPE needs channel pairs) is rejected.
    try:
        DecoderSpec(vocab_size=8, d_model=6, n_heads=2, n_layers=1, d_ff=4)
        raise AssertionError("odd d_k must be rejected")
    except ValueError as e:
        assert "even" in str(e)


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    tests = sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))
    passed = failed = 0
    for name in tests:
        try:
            getattr(mod, name)()
            passed += 1
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
