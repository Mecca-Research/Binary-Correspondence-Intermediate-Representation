"""Rung 6 (`frontends/models/serve.py`): the serving seam.

SLICE 1: `generate()` emits `decode_with_kv_cache`'s ids BIT-FOR-BIT while
recording the flight -- one valid DataDNA frame per token through a Broker into a
DurableLog that round-trips, a live `gem.kv_cache`-shaped record obeying the MLIR op's
paging law, ONE R13 manifest per generation (replayable; distinct prompts -> distinct
digests), and the prefill/decode SPLIT CERTIFICATE (batched prefill priced ≤ the
token-by-token arm -- measured 3x, gated ≥2x). The session module is R-law clean and
hydrates like any program. SentencePiece: a synthetic `tokenizer.model` written byte-
for-byte in the protobuf wire format loads dep-free; score-based merges, byte fallback
and control pieces pin the llama.cpp semantics; decode(encode(x)) == x for arbitrary
UTF-8. The closer ties rungs 2+4+6: TEXT in -> SP encode -> an ingested HF-layout
checkpoint generates -> SP decode -> text out, proof-carrying.

SLICE 2: `generate_stream` yields the SAME generation token-by-token ("token" events,
then one terminal "done" carrying the batch result -- `generate` is the stream drained,
so equivalence is by construction and pinned here anyway); a `TokenDFA` schema masks
the argmax (every constrained id is admitted by the walked state; the mask genuinely
bites; the identity DFA changes nothing; deadlock and malformed DFAs refuse loudly)."""

import json
import os
import struct
import tempfile

from bcir.frontends.models.decode import DecoderSpec, decode_with_kv_cache
from bcir.frontends.models.serve import (SessionCertificate, TokenDFA, certify_session,
                                         check_token_dfa, decode_session_module,
                                         generate, generate_stream)
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


def test_the_stream_is_generate_token_by_token():
    """SLICE 2, streaming: the "token" events carry generate()'s ids/frames one-by-one
    in order; the terminal "done" event IS the batch result (same ids, same frames, the
    SAME manifest digest -- one code path, zero drift); eos truncates the stream to one
    token event, with the pinned eos law (emitted, never fed back) intact."""
    w = _toy_weights(SPEC)
    batch = generate([3, 9, 1], SPEC, w, 5, h=AVX, theta=COOL)
    events = list(generate_stream([3, 9, 1], SPEC, w, 5, h=AVX, theta=COOL))
    toks, done = events[:-1], events[-1]
    assert [e.kind for e in toks] == ["token"] * 5 and done.kind == "done"
    assert [e.token for e in toks] == batch.ids
    assert [e.index for e in toks] == list(range(5)) and done.index == 5
    assert [e.frame for e in toks] == batch.frames == done.result.frames
    assert done.result.ids == batch.ids
    assert done.result.manifest.digest == batch.manifest.digest
    assert done.result.kv_record == batch.kv_record
    ev = list(generate_stream([3, 9, 1], SPEC, w, 5, h=AVX, theta=COOL,
                              eos_id=batch.ids[0]))
    assert [e.kind for e in ev] == ["token", "done"]                  # eos stops the stream
    assert ev[0].token == batch.ids[0]
    assert ev[-1].result.kv_record["pos"] == 3                        # eos never fed back
    try:
        generate_stream([], SPEC, w, 5, h=AVX, theta=COOL)
        raise AssertionError("expected the empty-prompt refusal AT THE CALL")
    except ValueError as e:
        assert "non-empty" in str(e)                                  # eager, not lazy


def test_schema_constrained_emission_obeys_the_dfa_and_refuses_deadlock():
    """SLICE 2, schema constraint: the identity DFA (whole vocab, one state) changes
    NOTHING (the mask is a mask, not a sampler); a biting DFA forces every emitted id
    into the walked state's allowed set and genuinely diverges from the free ids; the
    manifest gains ("constrained", 1); a deadlock state refuses mid-walk; malformed
    DFAs (edge-less allowed token, ghost target, off-vocab id) refuse AT THE CALL."""
    w = _toy_weights(SPEC)
    free = generate([3, 9, 1], SPEC, w, 4, h=AVX, theta=COOL)
    every = tuple(range(SPEC.vocab_size))
    identity = TokenDFA(allowed=(every,), edges=({t: 0 for t in every},))
    assert check_token_dfa(identity, SPEC.vocab_size) == []
    same = generate([3, 9, 1], SPEC, w, 4, h=AVX, theta=COOL, schema=identity)
    assert same.ids == free.ids                                       # the mask never bit
    assert dict(same.manifest.artifacts)["constrained"] == 1
    assert "constrained" not in dict(free.manifest.artifacts)         # slice-1 pins intact
    # a 2-state alternator that EXCLUDES the free path's first pick: the mask must bite.
    a, b = sorted(t for t in (0, 1, 2) if t != free.ids[0])[:2]
    biting = TokenDFA(allowed=((a, b), (b,)), edges=({a: 1, b: 1}, {b: 0}))
    assert check_token_dfa(biting, SPEC.vocab_size) == []
    forced = generate([3, 9, 1], SPEC, w, 4, h=AVX, theta=COOL, schema=biting)
    assert forced.ids != free.ids and forced.ids[0] in (a, b)
    state = biting.start                                              # re-walk: every id admitted
    for tid in forced.ids:
        assert tid in biting.allowed[state], (tid, state)
        state = biting.edges[state][tid]
    dead = TokenDFA(allowed=((a,), ()), edges=({a: 1}, {}))
    assert check_token_dfa(dead, SPEC.vocab_size) == []               # well-formed, yet...
    try:
        generate([3, 9, 1], SPEC, w, 4, h=AVX, theta=COOL, schema=dead)
        raise AssertionError("expected the deadlock refusal")
    except ValueError as e:
        assert "deadlock" in str(e) and "refuses" in str(e)
    for bad, needle in ((TokenDFA(allowed=((a,),), edges=({},)), "no edge"),
                        (TokenDFA(allowed=((a,),), edges=({a: 7},)), "does not exist"),
                        (TokenDFA(allowed=((10 ** 6,),), edges=({10 ** 6: 0},)), "vocab"),
                        (TokenDFA(allowed=((a,), (b,)), edges=({a: 0},)), "edge maps")):
        assert check_token_dfa(bad, SPEC.vocab_size), needle
        assert any(needle in m for m in check_token_dfa(bad, SPEC.vocab_size)), needle
        try:
            generate([3, 9, 1], SPEC, w, 2, h=AVX, theta=COOL, schema=bad)
            raise AssertionError(f"expected the {needle!r} refusal")
        except ValueError as e:
            assert "TokenDFA rejected" in str(e)


def test_schema_validation_is_total_and_generation_owns_its_snapshot():
    """Malformed schemas report diagnostics, and post-validation caller mutation cannot
    redirect or crash an in-flight walk."""
    malformed = (TokenDFA(allowed=(), edges=()),
                 TokenDFA(allowed=(([1],),), edges=({1: 0},)),
                 TokenDFA(allowed=((1, 1),), edges=({1: 0},)),
                 TokenDFA(allowed=((1,),), edges=({1: True},)),
                 TokenDFA(allowed=((1,),), edges=({"1": 0},)))
    for dfa in malformed:
        msgs = check_token_dfa(dfa, SPEC.vocab_size)
        assert msgs, dfa
    edges = {0: 0, 1: 0}
    schema = TokenDFA(allowed=((0, 1),), edges=(edges,))
    stream = generate_stream([3, 9, 1], SPEC, _toy_weights(SPEC), 3,
                             h=AVX, theta=COOL, schema=schema)
    edges.clear()                         # mutate the caller-owned table after validation
    events = list(stream)
    assert [event.kind for event in events] == ["token", "token", "token", "done"]


def test_long_sequential_prefill_has_a_unique_claim_namespace():
    """The historical decode-id band starts at 100; a 100-token sequential prefill
    must move its own IDs rather than alias the first generated claim."""
    from bcir.verify import verify
    module = decode_session_module(SPEC, 100, 2, batched_prefill=False)
    claim_ids = [claim.id for phase in module.phases for claim in phase.claims]
    assert len(claim_ids) == len(set(claim_ids)) == 102
    assert not any(diag.law == "R1.1" for diag in verify(module))


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
