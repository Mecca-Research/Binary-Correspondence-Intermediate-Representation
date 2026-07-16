"""Rung 1 of the open-weight ladder (§7.4): ModelManifest-ONLY ingestion, weights never loaded.

Synthetic Gemma-shaped fixtures: two tiny .safetensors shards (a real 8-byte LE header length +
JSON header + zero payload) and a config dict. The manifest carries the architecture, dtype
census, parameter count, shard hashes, context/vocab, tokenizer ref -- and is deterministic
(its canonical-JSON sha256 is the identity later rungs pin against). Malformed shards are
rejected loudly; the parser reads the header only (the rung-1 contract)."""

import json
import os
import struct
import tempfile

from bcir.frontends.models import (build_manifest, manifest_from_json,
                                   parse_safetensors_header, shard_digest)


def _shard(d, fname, tensors):
    """Write a minimal valid .safetensors shard: header length + JSON + zeroed payload."""
    offset = 0
    header = {}
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header).encode()
    path = os.path.join(d, fname)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob + b"\x00" * offset)
    return path


_CFG = {"architectures": ["GemmaForCausalLM"], "model_type": "gemma",
        "max_position_embeddings": 8192, "vocab_size": 256000}


def _two_shards(d):
    a = _shard(d, "model-00001-of-00002.safetensors",
               {"model.embed_tokens.weight": ("F32", (16, 8), 16 * 8 * 4),
                "model.layers.0.mlp.up.weight": ("F32", (8, 8), 8 * 8 * 4)})
    b = _shard(d, "model-00002-of-00002.safetensors",
               {"model.layers.0.attn.q.weight": ("BF16", (8, 8), 8 * 8 * 2),
                "lm_head.weight": ("F32", (16, 8), 16 * 8 * 4)})
    return [a, b]


def test_header_parse_reads_dtype_shape_census_only():
    with tempfile.TemporaryDirectory() as d:
        (a, _b) = _two_shards(d)
        tensors, meta = parse_safetensors_header(a)
        assert tensors["model.embed_tokens.weight"] == {"dtype": "F32", "shape": [16, 8]}
        assert meta == {"format": "pt"}


def test_manifest_census_params_dtypes_shards():
    with tempfile.TemporaryDirectory() as d:
        m = build_manifest(_two_shards(d), _CFG, tokenizer_ref="tokenizer.model",
                           license="gemma", source=d)
        assert m.architecture == "GemmaForCausalLM"
        assert m.param_count == 16 * 8 + 8 * 8 + 8 * 8 + 16 * 8 and m.tensor_count == 4
        assert m.dtypes == (("BF16", 1), ("F32", 3))
        assert [s.file for s in m.shards] == ["model-00001-of-00002.safetensors",
                                              "model-00002-of-00002.safetensors"]
        assert m.context_length == 8192 and m.vocab_size == 256000
        assert all(len(s.sha256) == 64 for s in m.shards)


def test_manifest_is_deterministic_and_round_trips():
    with tempfile.TemporaryDirectory() as d:
        shards = _two_shards(d)
        m1 = build_manifest(shards, _CFG, tokenizer_ref="t", source="s")
        m2 = build_manifest(list(reversed(shards)), _CFG, tokenizer_ref="t", source="s")
        assert m1 == m2 and m1.digest == m2.digest        # ingestion order never matters
        assert manifest_from_json(m1.to_json()) == m1     # the record survives serialization


def test_persisted_manifest_rejects_ambiguous_or_inconsistent_state():
    with tempfile.TemporaryDirectory() as d:
        manifest = build_manifest(_two_shards(d), _CFG)
        base = json.loads(manifest.to_json())
        variants = []
        wrong_total = json.loads(json.dumps(base)); wrong_total["param_count"] += 1
        variants.append(wrong_total)
        bad_hash = json.loads(json.dumps(base)); bad_hash["shards"][0]["sha256"] = "x" * 64
        variants.append(bad_hash)
        duplicate_file = json.loads(json.dumps(base))
        duplicate_file["shards"][1]["file"] = duplicate_file["shards"][0]["file"]
        variants.append(duplicate_file)
        for doc in variants:
            try:
                manifest_from_json(json.dumps(doc))
                raise AssertionError("forged manifest must be rejected")
            except ValueError:
                pass
        try:
            manifest_from_json('{"name":"a","name":"b"}')
            raise AssertionError("duplicate manifest keys must be rejected")
        except ValueError:
            pass


def test_manifest_rejects_duplicate_tensor_ownership_across_shards():
    with tempfile.TemporaryDirectory() as d:
        first = _shard(d, "a.safetensors", {"same": ("F32", (1,), 4)})
        second = _shard(d, "b.safetensors", {"same": ("F32", (1,), 4)})
        try:
            build_manifest([first, second], _CFG)
            raise AssertionError("one tensor cannot be owned by two shards")
        except ValueError as exc:
            assert "multiple shards" in str(exc)


def test_a_flipped_weight_byte_changes_the_identity():
    # integrity: the manifest digest pins the shard BYTES (hashed, never interpreted).
    with tempfile.TemporaryDirectory() as d:
        shards = _two_shards(d)
        before = build_manifest(shards, _CFG).digest
        with open(shards[1], "r+b") as f:
            f.seek(-1, os.SEEK_END)
            f.write(b"\x01")
        assert build_manifest(shards, _CFG).digest != before


def test_malformed_shards_are_rejected_loudly():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "short.safetensors")
        with open(p, "wb") as f:
            f.write(b"\x01\x02")                          # short header length
        for bad in (p,):
            try:
                parse_safetensors_header(bad)
                assert False, "short file must be rejected"
            except ValueError:
                pass
        q = os.path.join(d, "lying.safetensors")
        with open(q, "wb") as f:
            f.write(struct.pack("<Q", 1 << 40) + b"{}")   # implausible header length
        try:
            parse_safetensors_header(q)
            assert False, "implausible header length must be rejected"
        except ValueError:
            pass


def test_header_rejects_aliases_holes_negative_shapes_and_duplicate_keys():
    """The header is a wire trust boundary: every payload byte has exactly one owner
    and duplicate JSON keys cannot select different tensors in different parsers."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ambiguous.safetensors")
        cases = (
            ({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
              "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}},
             b"\0" * 12, "overlap"),
            ({"a": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}},
             b"\0" * 8, "hole"),
            ({"a": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 0]}},
             b"", "shape"),
        )
        for header, payload, needle in cases:
            blob = json.dumps(header).encode()
            with open(path, "wb") as f:
                f.write(struct.pack("<Q", len(blob)) + blob + payload)
            try:
                parse_safetensors_header(path)
                raise AssertionError(f"expected {needle} refusal")
            except ValueError as exc:
                assert needle in str(exc), str(exc)

        duplicate = (b'{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
                     b'"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}')
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(duplicate)) + duplicate + b"\0" * 4)
        try:
            parse_safetensors_header(path)
            raise AssertionError("expected duplicate-key refusal")
        except ValueError as exc:
            assert "duplicate" in str(exc)
        r = os.path.join(d, "notjson.safetensors")
        blob = b"not json at all"
        with open(r, "wb") as f:
            f.write(struct.pack("<Q", len(blob)) + blob)
        try:
            parse_safetensors_header(r)
            assert False, "non-JSON header must be rejected"
        except ValueError:
            pass


def test_shard_digest_is_the_streamed_sha256():
    import hashlib
    with tempfile.TemporaryDirectory() as d:
        (a, _b) = _two_shards(d)
        with open(a, "rb") as f:
            assert shard_digest(a) == hashlib.sha256(f.read()).hexdigest()


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
