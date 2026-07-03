"""Rung 4 of the open-weight ladder (§7.4): the QUANTIZED inference artifact
(`frontends/models/quantized.py`).

The Q8<->float32 bridge wrapped around the trusted rung-3 decoder (the attention_via_bridge
pattern): the quantized artifact is the reference decoder over round-tripped weights, so the
sole certified error is the R17-bounded weight quantization. Pins: the bridge preserves
shapes and is exactly the per-tensor round-trip; drift records over deterministic prompt
fixtures are deterministic and R17-stamped; MORE BITS = LESS DRIFT (Q8 ~20x tighter than Q4
on the fixture, gates set ~3x above measurement); and the record HONESTLY captures a greedy
flip -- at Q4 the fixture's argmax flips and `ids_match` says so, it is never hidden."""

from bcir.frontends.models.decode import check_decoder_weights, reference_decode
from bcir.frontends.models.quantized import decode_drift_record, quantize_decoder_weights
from bcir.kbcir.precision import quantization_error_bound
from bcir.kbcir.quantize import dequantize, quantize_per_group
from bcir.tests.test_model_decode import SPEC, _toy_weights

W = _toy_weights(SPEC)


def test_the_bridge_is_the_per_tensor_roundtrip_and_preserves_shapes():
    wq = quantize_decoder_weights(W, group_size=8, bits=8)
    assert check_decoder_weights(SPEC, wq) == []                 # shapes intact -> still a decoder
    want = tuple(dequantize(quantize_per_group(list(W.layers[0].w_q), 8, 8)))
    assert wq.layers[0].w_q == want                              # exactly the round-trip, no extra math
    assert wq.embedding.n_vocab == W.embedding.n_vocab and wq.embedding.dim == W.embedding.dim
    # the artifact IS the reference decoder over the round-tripped weights (no new kernel).
    assert reference_decode([1, 2, 3], SPEC, wq, max_new=3) == \
        reference_decode([1, 2, 3], SPEC, quantize_decoder_weights(W, 8, 8), max_new=3)


def test_drift_records_are_deterministic_and_r17_stamped():
    a = decode_drift_record([260, 259], SPEC, W, group_size=8, bits=8, max_new=4)
    b = decode_drift_record([260, 259], SPEC, W, group_size=8, bits=8, max_new=4)
    assert a == b                                                # same weights+prompt -> same record
    assert a.ulp_bound == quantization_error_bound() == 1        # the certified per-value bound
    assert len(a.ids_f32) == len(a.ids_q) == 4
    assert a.nll_f32 > 0 and a.nll_q > 0


def test_more_bits_less_drift_with_measured_gates():
    """Q8 vs Q4 over both deterministic fixtures: measured Q8 drift ~2.7e-3 (gate 1e-2),
    Q4 ~5.5e-2 to 6.5e-2 (gate 2e-1), monotone ~20x apart; NLL drift gated likewise."""
    for prompt in ([260, 259], [1, 2, 3]):
        q8 = decode_drift_record(prompt, SPEC, W, group_size=8, bits=8, max_new=4)
        q4 = decode_drift_record(prompt, SPEC, W, group_size=8, bits=4, max_new=4)
        assert 0 < q8.max_logit_drift < 1e-2, q8.max_logit_drift
        assert q8.max_logit_drift < q4.max_logit_drift < 2e-1, (q8, q4)
        assert abs(q8.nll_q - q8.nll_f32) < 1e-2
        assert abs(q4.nll_q - q4.nll_f32) < 5e-2


def test_a_greedy_flip_is_recorded_never_hidden():
    """The near-flat toy logits make argmax sensitive: at Q4 the fixture's greedy path
    flips and the record SAYS so; at Q8 the [1,2,3] fixture holds. Both pinned."""
    q4 = decode_drift_record([260, 259], SPEC, W, group_size=8, bits=4, max_new=4)
    assert not q4.ids_match and q4.ids_f32 != q4.ids_q
    q8 = decode_drift_record([1, 2, 3], SPEC, W, group_size=8, bits=8, max_new=4)
    assert q8.ids_match and q8.ids_f32 == q8.ids_q


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
