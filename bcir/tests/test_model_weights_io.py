"""BCIRQ8 v1 persistence and standalone-C Llama parity gates."""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

from bcir.frontends.models.decode import next_token_logits, reference_decode
from bcir.frontends.models.hf_ingest import spec_from_config, weights_from_tensors
from bcir.frontends.models.quantized import quantize_decoder_weights
from bcir.frontends.models.weights_io import (HEADER_SIZE, read_q8_decoder,
                                               write_q8_decoder)
from bcir.tests.test_model_ingest import _CONFIG, _hf_tensors

_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME = _ROOT / "runtime" / "c"
_HASHES = {"model": "00" * 32, "config": "11" * 32, "tokenizer": "22" * 32}
_TOKENS = {"bos": 1, "eos": 2, "pad": 0, "context_length": 128}


def _model(*, tied: bool = False):
    config = dict(_CONFIG, tie_word_embeddings=tied)
    tensors = _hf_tensors(config)
    prepared = {name: ("F32", shape, values) for name, (shape, values) in tensors.items()}
    spec = spec_from_config(config)
    return spec, weights_from_tensors(spec, prepared)


def _write(path: Path, *, tied: bool = False):
    spec, weights = _model(tied=tied)
    metadata = write_q8_decoder(path, spec, weights, source_hashes=_HASHES,
                                tokenizer_ids=_TOKENS)
    return spec, weights, metadata


def test_bcirq8_export_is_deterministic_and_exactly_the_q8_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a.bcirq8", Path(tmp) / "b.bcirq8"
        spec, weights, ma = _write(a)
        _, _, mb = _write(b)
        decoded_spec, decoded, mr = read_q8_decoder(a)
        expected = quantize_decoder_weights(weights, group_size=32, bits=8)
        assert a.read_bytes() == b.read_bytes()
        assert ma == mb == mr
        assert decoded_spec == spec
        assert decoded == expected
        assert ma.tensor_count == 2 + 9 * spec.n_layers + 1
        assert ma.element_count == sum(len(t) for t in (
            weights.embedding.table, weights.g_final, weights.lm_head)) + sum(
                sum(len(getattr(layer, name)) for name in
                    ("g_attn", "w_q", "w_k", "w_v", "w_o", "g_ff",
                     "w_gate", "w1", "w2")) for layer in weights.layers)
        assert ma.artifact_sha256 and ma.body_crc32


def _rewrite_crcs(raw: bytearray) -> None:
    struct.pack_into("<I", raw, 112, zlib.crc32(raw[HEADER_SIZE:]) & 0xFFFFFFFF)
    struct.pack_into("<I", raw, 116, 0)
    struct.pack_into("<I", raw, 116, zlib.crc32(raw[:HEADER_SIZE]) & 0xFFFFFFFF)


def test_bcirq8_rejects_truncation_corruption_and_invalid_spans():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "model.bcirq8"
        _write(source)
        original = source.read_bytes()
        variants = {
            "truncated": original[:-1],
            "magic": b"X" + original[1:],
            "version": original[:8] + b"\x02\x00" + original[10:],
            "body": original[:-1] + bytes([original[-1] ^ 1]),
        }
        invalid_span = bytearray(original)
        payload_offset = struct.unpack_from("<Q", invalid_span, 96)[0]
        struct.pack_into("<Q", invalid_span, HEADER_SIZE + 32, payload_offset - 8)
        _rewrite_crcs(invalid_span)
        variants["invalid-span"] = bytes(invalid_span)
        noncanonical = bytearray(original)
        first = noncanonical[HEADER_SIZE:HEADER_SIZE + 48]
        second = noncanonical[HEADER_SIZE + 48:HEADER_SIZE + 96]
        noncanonical[HEADER_SIZE:HEADER_SIZE + 48] = second
        noncanonical[HEADER_SIZE + 48:HEADER_SIZE + 96] = first
        _rewrite_crcs(noncanonical)
        variants["noncanonical-order"] = bytes(noncanonical)
        for name, data in variants.items():
            path = Path(tmp) / f"{name}.bcirq8"
            path.write_bytes(data)
            try:
                read_q8_decoder(path)
                raise AssertionError(f"{name} artifact must be rejected")
            except ValueError:
                pass


def _link_args() -> list[str]:
    try:
        from bcir.toolchain import host_link_args
    except ImportError:
        return [] if os.name == "nt" else ["-lm"]
    return list(host_link_args(["-lm"]))


def _compiler() -> str | None:
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _build_cli(directory: Path) -> Path:
    compiler = _compiler()
    assert compiler is not None
    executable = directory / ("bcir-llama.exe" if os.name == "nt" else "bcir-llama")
    command = [compiler, "-std=c11", "-O2", "-ffp-contract=off", "-Wall", "-Wextra",
               "-I", str(_RUNTIME), str(_RUNTIME / "bcir_q8_model.c"),
               str(_RUNTIME / "bcir_decode.c"), str(_RUNTIME / "bcir_llama.c"),
               str(_RUNTIME / "bcir_llama_cli.c"), "-o", str(executable), *_link_args()]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return executable


def test_standalone_c_q8_decoder_matches_python_for_tied_and_untied_heads():
    if _compiler() is None:
        return
    prompt = [1, 2, 3]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        executable = _build_cli(directory)
        for tied in (False, True):
            artifact = directory / f"model-{tied}.bcirq8"
            _write(artifact, tied=tied)
            spec, weights, _ = read_q8_decoder(artifact)
            expected_ids = reference_decode(prompt, spec, weights, 2)
            python_logits = next_token_logits(prompt + expected_ids[:1], spec, weights)
            logits_path = directory / f"logits-{tied}.f64"
            result = subprocess.run(
                [str(executable), "--model", str(artifact), "--prompt-ids", "1,2,3",
                 "--max-new", "2", "--logits-out", str(logits_path)],
                capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            assert json.loads(result.stdout) == {"generated_ids": expected_ids}
            raw = logits_path.read_bytes()
            c_logits = struct.unpack(f"<{spec.vocab_size}d", raw)
            assert max(abs(a - b) for a, b in zip(python_logits, c_logits)) <= 1e-9
        corrupt = bytearray((directory / "model-False.bcirq8").read_bytes())
        corrupt[-1] ^= 1
        corrupt_path = directory / "corrupt.bcirq8"
        corrupt_path.write_bytes(corrupt)
        result = subprocess.run(
            [str(executable), "--model", str(corrupt_path), "--prompt-ids", "1",
             "--max-new", "1"], capture_output=True, text=True)
        assert result.returncode != 0 and "CRC" in result.stderr


if __name__ == "__main__":
    import sys
    module = sys.modules[__name__]
    names = sorted(name for name in dir(module)
                   if name.startswith("test_") and callable(getattr(module, name)))
    failed = 0
    for name in names:
        try:
            getattr(module, name)()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc!r}")
    raise SystemExit(1 if failed else 0)
