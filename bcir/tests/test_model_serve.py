"""Rung 6, first slice (`frontends/models/serve.py`): the serving seam.

`generate()` `generate()` emits `decode_with_kv_cache`'s ids BIT-FOR-BIT while
recording the flight -- one valid DataDNA frame per token through a Broker into a
DurableLog that round-trips, a live `gem.kv_cache`-shaped record obeying the MLIR op's
paging law, ONE R13 manifest per generation (replayable; distinct prompts -> distinct
digests), and the prefill/decode SPLIT CERTIFICATE (batched prefill priced ≤ the
token-by-token arm -- measured 3x, gated ≥2x). The session module is R-law clean and
hydrates like any program. SentencePiece: a synthetic `tokenizer.model` written byte-
for-byte in the protobuf wire format loads dep-free; score-based merges, byte fallback
and control pieces pin the llama.cpp semantics; decode(encode(x)) == x for arbitrary
UTF-8. The closer ties rungs 2+4+6: TEXT in -> SP encode -> an ingested HF-layout
checkpoint generates -> SP decode -> text out, proof-carrying."""

import json
import os
import struct
import tempfile

from bcir.frontends.models.decode import DecoderSpec, decode_with_kv_cache
from bcir.frontends.models.serve import (SessionCertificate, certify_session,
                                         decode_session_module, generate)
from bcir.frontends.models.spm import load_sentencepiece
from bcir.tests.test_model_spm import _write_sp_model
from bcir.kbcir import TARGETS
from bcir.kbcir.cost import Theta
from bcir.tests.test_model_decode import _toy_weights

AVX = TARGETS["x86_avx512"]
COOL = Theta.cool()
SPEC = DecoderSpec(vocab_size=64, d_model=8, n_heads=2, n_layers=2, d_ff=16)


# --- the serving seam ------------------------------------------------------------------------

def test_generate_is_the_cached_decode_with_a_flight_recorder():
    """The ids are decode_with_kv_cache's BIT-FOR-BIT; every frame is schema-valid; the
    kv record obeys the gem.kv_cache paging law (pos <= capacity) and lands at
    prompt+generated; the certificate admits the prefill/decode split at the measured
    ~3x (gated >= 2x); eos stops the emission."""
    w = _toy_weights(SPEC)
    for prompt in ([3, 9, 1], [60, 7]):
        r = generate(prompt, SPEC, w, 5, h=AVX, theta=COOL)
        assert r.ids == decode_with_kv_cache(prompt, SPEC, w, max_new=5)
        assert len(r.frames) == 5
        assert all(f.violations() == [] for f in r.frames)
        assert r.kv_record["pos"] == len(prompt) + 5 == r.kv_record["capacity"]
        assert r.kv_record["pos"] <= r.kv_record["capacity"]          # the paging law
        assert r.kv_record["n_kv_heads"] == SPEC.kv_heads
        assert r.certificate.admitted
        assert r.certificate.prefill_batched * 2 <= r.certificate.prefill_sequential
    first = generate([3, 9, 1], SPEC, w, 5, h=AVX, theta=COOL).ids[0]
    stopped = generate([3, 9, 1], SPEC, w, 5, h=AVX, theta=COOL, eos_id=first)
    assert stopped.ids == [first]
    assert stopped.kv_record["pos"] == 3                              # the eos id is emitted,
    assert len(stopped.frames) == 1                                   # never fed back


def test_the_session_module_is_law_clean_and_the_log_replays():
    """The session module passes R1-R8 + the advisory laws and hydrates to an
    R10/R11-clean StreamPack (a generation is a PROGRAM); the DurableLog round-trips the
    frames bit-for-bit; the R13 manifest replays, and distinct prompts give distinct
    manifests (the ids digest is an artifact)."""
    from bcir.gem.streampack import hydrate
    from bcir.kbcir.provenance import replay
    from bcir.kbcir.realize import optimize
    from bcir.telemetry import load_durable_log
    from bcir.verify import verify, verify_pack, verify_smart_lowering
    m = decode_session_module(SPEC, 3, 4)
    assert verify(m) == [], verify(m)
    assert verify_smart_lowering(m) == []
    pack = hydrate(m, optimize(m, AVX, COOL), plan=m.name)
    assert pack.provenance_ok()
    assert verify_pack(m, pack) == []
    seq = decode_session_module(SPEC, 3, 4, batched_prefill=False)
    assert verify(seq) == []                                          # both arms lawful
    w = _toy_weights(SPEC)
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "gen.jsonl")
        r = generate([3, 9, 1], SPEC, w, 4, h=AVX, theta=COOL, log_path=log)
        back = load_durable_log(log)
        assert list(back) == r.frames                                 # bit-for-bit replay
    rep = replay(r.manifest, decode_session_module(SPEC, 3, 4), AVX, COOL,
                 artifacts=r.manifest.artifacts)
    assert rep.score > 0
    r2 = generate([5, 2, 8], SPEC, w, 4, h=AVX, theta=COOL)
    assert r2.manifest.digest != r.manifest.digest                    # prompt-distinct


def test_text_in_text_out_ties_rungs_2_4_and_6():
    """THE WAVE-10 CLOSER: TEXT -> SentencePiece encode -> an ingested HF-layout
    checkpoint (rung 4) -> generate (rung 6) -> SentencePiece decode -> TEXT, with the
    ids equal to the naive reference decode over the same ingested weights."""
    from bcir.frontends.models.decode import reference_decode
    from bcir.frontends.models.hf_ingest import ingest_checkpoint
    from bcir.tests.test_model_ingest import _CONFIG, _hf_tensors, _write_shard
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
            json.dump(_CONFIG, f)
        _write_shard(os.path.join(tmp, "model.safetensors"), _hf_tensors(_CONFIG))
        spec, w = ingest_checkpoint(tmp)
        sp_path = os.path.join(tmp, "tokenizer.model")
        _write_sp_model(sp_path)
        tok = load_sentencepiece(sp_path)
    prompt_ids = [i % spec.vocab_size for i in tok.encode("hello world")]
    r = generate(prompt_ids, spec, w, 6, h=AVX, theta=COOL)
    assert r.ids == reference_decode(prompt_ids, spec, w, max_new=6)  # naive == served
    text = tok.decode([i for i in r.ids if i < tok.n_pieces])
    assert isinstance(text, str)                                      # decodable output
    assert r.certificate.admitted and r.manifest.digest


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
