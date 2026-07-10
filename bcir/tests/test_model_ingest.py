"""Rung 4, second half (§7.4): REAL-WEIGHT INGESTION (`safetensors_io.py` + `hf_ingest.py`).

The tensor reader decodes F64/F32/F16/BF16 exactly (dep-free struct; lying spans refuse);
the HF-Llama layout mapping (Linear transpose, the rotate_half -> interleaved RoPE
out-channel permutation, gated-SiLU MLP, untied head) is gated by an INDEPENDENT
HF-semantics decoder written in this file straight from the paper conventions (x @ W^T,
half-split rotation) -- the ingested weights through the rung-3 reference decoder must
emit the same greedy ids. The census ties (param_count == the shard's element count), the
KV-cache twin gate holds on ingested weights, and the Q8 bridge quantizes them. A REAL
released checkpoint runs through the same path when BCIR_HF_MODEL_DIR is set (asset-gated,
self-skips in CI)."""

import json
import math
import os
import struct
import tempfile

from bcir.frontends.models.decode import (check_decoder_weights, decode_with_kv_cache,
                                          decoder_param_count, reference_decode)
from bcir.frontends.models.hf_ingest import (ingest_checkpoint, ingest_checkpoint_with_report,
                                             rope_interleave_order, spec_from_config,
                                             weights_from_tensors)
from bcir.frontends.models.safetensors_io import decode_tensor, load_tensors

_CONFIG = {"vocab_size": 64, "hidden_size": 8, "num_attention_heads": 4,
           "num_key_value_heads": 2, "num_hidden_layers": 2, "intermediate_size": 16,
           "rope_theta": 10000.0, "hidden_act": "silu", "tie_word_embeddings": False}


def _hf_tensors(config: dict, seed: int = 0x4EA1) -> dict:
    """A synthetic checkpoint in the HF layout: every Linear [out, in], rotate_half RoPE
    convention, gate/up/down MLP, untied lm_head. Values BF16-REPRESENTABLE (rounded
    through the truncation) so the F32 and BF16 shard encodings carry identical floats."""
    import random
    rng = random.Random(seed)

    def bf16(v):
        (bits,) = struct.unpack("<I", struct.pack("<f", v))
        (out,) = struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))
        return out

    def t(*shape):
        n = 1
        for x in shape:
            n *= x
        return (tuple(shape), [bf16(rng.uniform(-0.3, 0.3)) for _ in range(n)])

    d, f = config["hidden_size"], config["intermediate_size"]
    v, kvd = config["vocab_size"], (config["num_key_value_heads"] * d
                                    // config["num_attention_heads"])
    out = {"model.embed_tokens.weight": t(v, d), "model.norm.weight": t(d),
           "lm_head.weight": t(v, d)}
    for li in range(config["num_hidden_layers"]):
        p = f"model.layers.{li}."
        out[p + "input_layernorm.weight"] = t(d)
        out[p + "self_attn.q_proj.weight"] = t(d, d)
        out[p + "self_attn.k_proj.weight"] = t(kvd, d)
        out[p + "self_attn.v_proj.weight"] = t(kvd, d)
        out[p + "self_attn.o_proj.weight"] = t(d, d)
        out[p + "post_attention_layernorm.weight"] = t(d)
        out[p + "mlp.gate_proj.weight"] = t(f, d)
        out[p + "mlp.up_proj.weight"] = t(f, d)
        out[p + "mlp.down_proj.weight"] = t(d, f)
    return out


def _write_shard(path: str, tensors: dict, dtype: str = "F32") -> None:
    """The tensors as a real safetensors file (little-endian; BF16 truncates)."""
    header, blobs, off = {}, [], 0
    for name in sorted(tensors):
        shape, vals = tensors[name]
        if dtype == "F32":
            raw = struct.pack(f"<{len(vals)}f", *vals)
        else:                                          # BF16: the top 16 bits of the F32
            bits = struct.unpack(f"<{len(vals)}I", struct.pack(f"<{len(vals)}f", *vals))
            raw = struct.pack(f"<{len(vals)}H", *(b >> 16 for b in bits))
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hb = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))


def _hf_decode(config: dict, tensors: dict, prompt: list, max_new: int) -> list:
    """An INDEPENDENT greedy decoder written from the HF Llama conventions -- x @ W^T
    Linears, rotate_half RoPE (half-split pairs), gated-SiLU MLP, untied head, eps=1e-6
    RMSNorm -- sharing NO code with decode.py. The ids it emits are the ground truth the
    ingest mapping must reproduce."""
    d = config["hidden_size"]
    nh, nkv = config["num_attention_heads"], config["num_key_value_heads"]
    dk, group, base = d // nh, nh // nkv, config["rope_theta"]
    norm_eps = float(config.get("rms_norm_eps", 1e-6))

    def lin(x_rows, wname):                            # y = x @ W^T, W is [out, in]
        (out_d, in_d), w = tensors[wname]
        return [[sum(row[i] * w[o * in_d + i] for i in range(in_d)) for o in range(out_d)]
                for row in x_rows]

    def rms(x_rows, gname):
        (_,), g = (tensors[gname][0], tensors[gname][1])
        out = []
        for row in x_rows:
            r = math.sqrt(sum(v * v for v in row) / len(row) + norm_eps)
            out.append([row[i] / r * g[i] for i in range(len(row))])
        return out

    def rope_half(vec, pos):                           # rotate_half over one head's dk chans
        out = [0.0] * dk
        for c in range(dk // 2):
            th = pos * base ** (-2.0 * c / dk)
            co, si = math.cos(th), math.sin(th)
            out[c] = vec[c] * co - vec[c + dk // 2] * si
            out[c + dk // 2] = vec[c + dk // 2] * co + vec[c] * si
        return out

    def forward(ids):
        (v_d, _), emb = tensors["model.embed_tokens.weight"]
        x = [list(emb[t * d:(t + 1) * d]) for t in ids]
        n = len(x)
        for li in range(config["num_hidden_layers"]):
            p = f"model.layers.{li}."
            h = rms(x, p + "input_layernorm.weight")
            q = lin(h, p + "self_attn.q_proj.weight")
            k = lin(h, p + "self_attn.k_proj.weight")
            vv = lin(h, p + "self_attn.v_proj.weight")
            qh = [[rope_half(q[i][hd * dk:(hd + 1) * dk], i) for hd in range(nh)]
                  for i in range(n)]
            kh = [[rope_half(k[i][g * dk:(g + 1) * dk], i) for g in range(nkv)]
                  for i in range(n)]
            ctx = [[None] * nh for _ in range(n)]
            for hd in range(nh):
                g = hd // group
                for i in range(n):
                    scores = []
                    for j in range(i + 1):
                        scores.append(sum(qh[i][hd][c] * kh[j][g][c] for c in range(dk))
                                      / math.sqrt(dk))
                    m = max(scores)
                    ex = [math.exp(s - m) for s in scores]
                    tot = sum(ex)
                    a = [e / tot for e in ex]
                    ctx[i][hd] = [sum(a[j] * vv[j][g * dk + c] for j in range(i + 1))
                                  for c in range(dk)]
            cat = [[c for hd in range(nh) for c in ctx[i][hd]] for i in range(n)]
            att = lin(cat, p + "self_attn.o_proj.weight")
            x = [[x[i][c] + att[i][c] for c in range(d)] for i in range(n)]
            h2 = rms(x, p + "post_attention_layernorm.weight")
            gt = lin(h2, p + "mlp.gate_proj.weight")
            up = lin(h2, p + "mlp.up_proj.weight")
            act = [[gt[i][c] / (1.0 + math.exp(-gt[i][c])) * up[i][c]
                    for c in range(len(gt[i]))] for i in range(n)]
            dn = lin(act, p + "mlp.down_proj.weight")
            x = [[x[i][c] + dn[i][c] for c in range(d)] for i in range(n)]
        return rms(x, "model.norm.weight")

    ids = list(prompt)
    out = []
    for _ in range(max_new):
        hfin = forward(ids)[-1]
        (v_d, _), head = tensors["lm_head.weight"]
        logits = [sum(hfin[c] * head[t * d + c] for c in range(d)) for t in range(v_d)]
        best = 0
        for t in range(1, v_d):
            if logits[t] > logits[best]:
                best = t
        ids.append(best)
        out.append(best)
    return out


def test_the_ingested_checkpoint_matches_independent_hf_semantics():
    """THE RUNG-4 CLOSER: an HF-layout checkpoint (Linear transpose + rotate_half RoPE +
    gated MLP + untied head) written byte-for-byte as safetensors, ingested through
    config -> spec -> tensors -> weights, decodes the SAME greedy ids as an independent
    HF-semantics decoder -- and the KV-cache twin gate holds on the real-layout weights."""
    tensors = _hf_tensors(_CONFIG)
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
            json.dump(_CONFIG, f)
        _write_shard(os.path.join(tmp, "model.safetensors"), tensors)
        spec, w = ingest_checkpoint(tmp)
    assert spec.n_kv_heads == 2 and spec.activation == "silu_gate"
    assert not spec.tied_embeddings
    assert check_decoder_weights(spec, w) == []
    # the census tie: the spec's param count == the shard's element census.
    census = sum(len(v) for _s, v in tensors.values())
    assert decoder_param_count(spec) == census
    for prompt in ([3], [1, 2, 3], [60, 7, 33]):
        want = _hf_decode(_CONFIG, tensors, prompt, 5)
        got = reference_decode(prompt, spec, w, max_new=5)
        assert got == want, (prompt, got, want)
        assert decode_with_kv_cache(prompt, spec, w, max_new=5) == got   # the twin gate


def test_bf16_and_f32_shards_ingest_identically():
    """The values are BF16-representable by construction, so the BF16-encoded shard and
    the F32 shard must ingest to the SAME floats and the same ids (the dtype decoder is
    exact, not approximate)."""
    tensors = _hf_tensors(_CONFIG)
    with tempfile.TemporaryDirectory() as tmp:
        for enc in ("F32", "BF16"):
            os.mkdir(os.path.join(tmp, enc))
            with open(os.path.join(tmp, enc, "config.json"), "w", encoding="utf-8") as f:
                json.dump(_CONFIG, f)
            _write_shard(os.path.join(tmp, enc, "model.safetensors"), tensors, dtype=enc)
        s32, w32 = ingest_checkpoint(os.path.join(tmp, "F32"))
        s16, w16 = ingest_checkpoint(os.path.join(tmp, "BF16"))
    assert w32 == w16
    assert reference_decode([5, 9], s32, w32, max_new=4) == \
        reference_decode([5, 9], s16, w16, max_new=4)


def test_lying_shards_refuse_loudly():
    """A span past the data section, a shape/byte mismatch, and an unknown dtype each
    refuse with the tensor's name -- never a silent mis-read."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.safetensors")
        raw = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        for header, msg in (
                ({"t": {"dtype": "F32", "shape": [4], "data_offsets": [0, 32]}}, "escapes"),
                ({"t": {"dtype": "F32", "shape": [3], "data_offsets": [0, 16]}}, "bytes"),
                ({"t": {"dtype": "I64", "shape": [2], "data_offsets": [0, 16]}}, "dtype")):
            hb = json.dumps(header).encode("utf-8")
            with open(path, "wb") as f:
                f.write(struct.pack("<Q", len(hb)) + hb + raw)
            try:
                load_tensors(path)
                raise AssertionError(f"expected a refusal ({msg})")
            except ValueError as e:
                assert "t" in str(e)
    # and the quantized bridge rides the ingested (gated, untied) weights unchanged.
    tensors = _hf_tensors(_CONFIG)
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
            json.dump(_CONFIG, f)
        _write_shard(os.path.join(tmp, "model.safetensors"), tensors)
        spec, w = ingest_checkpoint(tmp)
    from bcir.frontends.models.quantized import quantize_decoder_weights
    qw = quantize_decoder_weights(w, group_size=32, bits=8)
    assert check_decoder_weights(spec, qw) == []       # shapes survive the Q8 round-trip


def test_a_real_released_checkpoint_ingests_when_present():
    """ASSET-GATED (self-skips): point BCIR_HF_MODEL_DIR at a downloaded Llama-family
    checkpoint directory (config.json + model.safetensors -- e.g. Maykeye/TinyLLama-v0)
    and the full rung-4 path runs on REAL trained weights: ingest, weight-shape law,
    census tie, greedy decode with the naive-vs-cached bit-for-bit twin gate, and finite
    logits end to end."""
    model_dir = os.environ.get("BCIR_HF_MODEL_DIR", "")
    if not model_dir or not os.path.isfile(os.path.join(model_dir, "model.safetensors")):
        return                                          # the asset gate (like the rig gates)
    spec, w, report = ingest_checkpoint_with_report(model_dir)
    assert check_decoder_weights(spec, w) == []
    from bcir.frontends.models.manifest import build_manifest
    man = build_manifest([os.path.join(model_dir, "model.safetensors")], {})
    assert report.decoder_element_count == decoder_param_count(spec)
    assert report.decoder_element_count + report.auxiliary_element_count == \
        report.shard_element_count == man.param_count            # the census tie, REAL
    prompt = [1, 306, 4658]                                      # ids only (no tokenizer)
    naive = reference_decode(prompt, spec, w, max_new=4)
    assert decode_with_kv_cache(prompt, spec, w, max_new=4) == naive
    assert all(0 <= t < spec.vocab_size for t in naive)
    from bcir.frontends.models.decode import next_token_logits
    logits = next_token_logits(prompt, spec, w)
    assert all(math.isfinite(v) for v in logits)


def test_ingest_report_allowlists_and_validates_rotary_auxiliaries():
    config = dict(_CONFIG, rms_norm_eps=1e-5)
    tensors = _hf_tensors(config)
    dk = config["hidden_size"] // config["num_attention_heads"]
    expected = [config["rope_theta"] ** (-(2.0 * i) / dk) for i in range(dk // 2)]
    aux_names = []
    for li in range(config["num_hidden_layers"]):
        name = f"model.layers.{li}.self_attn.rotary_emb.inv_freq"
        tensors[name] = ((dk // 2,), list(expected))
        aux_names.append(name)
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)
        _write_shard(os.path.join(tmp, "model.safetensors"), tensors)
        spec, w, report = ingest_checkpoint_with_report(tmp)
    assert spec.rms_norm_eps == 1e-5
    assert check_decoder_weights(spec, w) == []
    assert report.auxiliary_tensors == tuple(aux_names)
    assert report.decoder_element_count == decoder_param_count(spec)
    assert report.auxiliary_element_count == config["num_hidden_layers"] * (dk // 2)
    assert report.decoder_element_count + report.auxiliary_element_count == report.shard_element_count
    assert reference_decode([1, 2, 3], spec, w, max_new=3) == \
        _hf_decode(config, tensors, [1, 2, 3], 3)       # non-default epsilon, independent twin


def test_ingest_apis_reject_unknown_or_lying_auxiliary_tensors():
    def write_and_ingest(tensors, *, with_report=True):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
                json.dump(_CONFIG, f)
            _write_shard(os.path.join(tmp, "model.safetensors"), tensors)
            return (ingest_checkpoint_with_report(tmp) if with_report
                    else ingest_checkpoint(tmp))

    unknown = _hf_tensors(_CONFIG)
    unknown["model.unexpected.weight"] = ((1,), [0.0])
    try:
        write_and_ingest(unknown)
        raise AssertionError("unknown unused tensors must not disappear from a strict census")
    except ValueError as exc:
        assert "not allowlisted" in str(exc) and "unexpected" in str(exc)
    try:
        write_and_ingest(unknown, with_report=False)
        raise AssertionError("the compatibility tuple API must reject unknown tensors too")
    except ValueError as exc:
        assert "not allowlisted" in str(exc) and "unexpected" in str(exc)

    bad_rotary = _hf_tensors(_CONFIG)
    bad_rotary["model.layers.0.self_attn.rotary_emb.inv_freq"] = ((1,), [0.5])
    try:
        write_and_ingest(bad_rotary)
        raise AssertionError("a rotary auxiliary inconsistent with rope_base must fail")
    except ValueError as exc:
        assert "rope_base" in str(exc)


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
