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
from bcir.frontends.models.weights_io import (
    DEFAULT_MAX_ARTIFACT_BYTES,
    HEADER_SIZE,
    read_q8_decoder,
    write_q8_decoder,
)
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
    metadata = write_q8_decoder(path, spec, weights, source_hashes=_HASHES, tokenizer_ids=_TOKENS)
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
        assert ma.element_count == sum(
            len(t) for t in (weights.embedding.table, weights.g_final, weights.lm_head)
        ) + sum(
            sum(
                len(getattr(layer, name))
                for name in ("g_attn", "w_q", "w_k", "w_v", "w_o", "g_ff", "w_gate", "w1", "w2")
            )
            for layer in weights.layers
        )
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
        first = noncanonical[HEADER_SIZE : HEADER_SIZE + 48]
        second = noncanonical[HEADER_SIZE + 48 : HEADER_SIZE + 96]
        noncanonical[HEADER_SIZE : HEADER_SIZE + 48] = second
        noncanonical[HEADER_SIZE + 48 : HEADER_SIZE + 96] = first
        _rewrite_crcs(noncanonical)
        variants["noncanonical-order"] = bytes(noncanonical)
        too_many_layers = bytearray(original)
        struct.pack_into("<I", too_many_layers, 40, 0x8000)
        _rewrite_crcs(too_many_layers)
        variants["layer-field-overflow"] = bytes(too_many_layers)
        for name, data in variants.items():
            path = Path(tmp) / f"{name}.bcirq8"
            path.write_bytes(data)
            try:
                read_q8_decoder(path)
                raise AssertionError(f"{name} artifact must be rejected")
            except ValueError:
                pass


def test_bcirq8_limits_and_metadata_are_strict_before_allocation_or_write():
    spec, weights = _model()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        target = directory / "bad.bcirq8"
        bad_calls = (
            lambda: write_q8_decoder(
                target, spec, weights, group_size=True, source_hashes=_HASHES, tokenizer_ids=_TOKENS
            ),
            lambda: write_q8_decoder(
                target, spec, weights, source_hashes=_HASHES, tokenizer_ids=dict(_TOKENS, bos=1.5)
            ),
            lambda: write_q8_decoder(
                target,
                spec,
                weights,
                source_hashes=_HASHES,
                tokenizer_ids=dict(_TOKENS, bos=spec.vocab_size),
            ),
            lambda: write_q8_decoder(
                target,
                spec,
                weights,
                source_hashes=dict(_HASHES, source_model_sha256=_HASHES["model"]),
                tokenizer_ids=_TOKENS,
            ),
            lambda: write_q8_decoder(
                target,
                spec,
                weights,
                source_hashes=_HASHES,
                tokenizer_ids=_TOKENS,
                context_length="128",
            ),
            lambda: write_q8_decoder(
                target, spec, weights, source_hashes=_HASHES, tokenizer_ids=_TOKENS, max_elements=1
            ),
            lambda: write_q8_decoder(
                target,
                spec,
                weights,
                source_hashes=_HASHES,
                tokenizer_ids=_TOKENS,
                max_bytes=HEADER_SIZE,
            ),
        )
        for call in bad_calls:
            try:
                call()
                raise AssertionError("malformed BCIRQ8 metadata was accepted")
            except ValueError:
                pass
            assert not target.exists()

        artifact = directory / "model.bcirq8"
        _write(artifact)
        try:
            read_q8_decoder(artifact, max_elements=1)
            raise AssertionError("decoded-element ceiling was ignored")
        except ValueError as exc:
            assert "elements" in str(exc) and "limit" in str(exc)

        sparse = directory / "oversized.bcirq8"
        with sparse.open("wb") as stream:
            stream.seek(DEFAULT_MAX_ARTIFACT_BYTES)
            stream.write(b"\0")
        try:
            read_q8_decoder(sparse)
            raise AssertionError("oversized sparse BCIRQ8 artifact was accepted")
        except ValueError as exc:
            assert "bytes" in str(exc) and "limit" in str(exc)


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
    command = [
        compiler,
        "-std=c11",
        "-O2",
        "-ffp-contract=off",
        "-Wall",
        "-Wextra",
        "-I",
        str(_RUNTIME),
        str(_RUNTIME / "bcir_q8_model.c"),
        str(_RUNTIME / "bcir_decode.c"),
        str(_RUNTIME / "bcir_ai_kernels.c"),
        str(_RUNTIME / "bcir_llama.c"),
        str(_RUNTIME / "bcir_llama_cli.c"),
        "-o",
        str(executable),
        *_link_args(),
    ]
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
                [
                    str(executable),
                    "--model",
                    str(artifact),
                    "--prompt-ids",
                    "1,2,3",
                    "--max-new",
                    "2",
                    "--logits-out",
                    str(logits_path),
                ],
                capture_output=True,
                text=True,
            )
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
            [str(executable), "--model", str(corrupt_path), "--prompt-ids", "1", "--max-new", "1"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0 and "CRC" in result.stderr
        sparse = directory / "oversized-sparse.bcirq8"
        with sparse.open("wb") as stream:
            stream.seek(DEFAULT_MAX_ARTIFACT_BYTES)
            stream.write(b"\0")
        result = subprocess.run(
            [str(executable), "--model", str(sparse), "--prompt-ids", "1", "--max-new", "1"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0 and "limit" in result.stderr
        result = subprocess.run(
            [
                str(executable),
                "--model",
                str(directory / "model-False.bcirq8"),
                "--prompt-ids",
                "1",
                "--max-new",
                str((1 << 16) + 1),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0 and "context limit" in result.stderr


_ROPE_WITNESS = r"""
#include "bcir_decode.h"
#include "bcir_llama_ops.h"
#include "bcir_q8_model.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The decoder's rotation (bcir_llama_op_rope: a table filled once a position, every head applied from
 * it) against bcir_rope over each head's row, byte for byte, at positions in the orders a GEM program
 * may take them: repeated, backward, after a gap, the first again. */
int main(int argc, char **argv) {
  static const size_t pos[] = {0, 0, 1, 2, 2, 7, 3, 63, 0, 5, 5, 62, 61, 1};
  bcir_q8_model m;
  bcir_llama_ws w;
  char err[256];
  uint64_t state = 0x2545F4914F6CDD1Du;
  size_t i, h, n = sizeof pos / sizeof pos[0];
  double *ref;
  uint32_t dk, kvd;
  bcir_q8_model_init(&m);
  if (argc != 2 || bcir_q8_model_load(argv[1], &m, err, sizeof err)) return 2;
  if (!bcir_llama_model_ready(&m) || bcir_llama_ws_init(&w, &m, 64, NULL)) return 3;
  dk = m.d_model / m.n_heads;
  kvd = m.n_kv_heads * dk;
  ref = malloc((size_t)dk * sizeof *ref);
  if (!ref) return 4;
  for (i = 0; i < n; i++) {
    for (h = 0; h < m.d_model + kvd; h++) {
      state ^= state << 13, state ^= state >> 7, state ^= state << 17;
      *(h < m.d_model ? &w.q[h] : &w.k[h - m.d_model]) =
          (double)(int64_t)(state >> 11) / 4503599627370496.0 - 1.0;
    }
    bcir_llama_op_rope(&m, &w, pos[i]);
    for (h = 0; h < m.n_heads + m.n_kv_heads; h++) {
      const double *x = h < m.n_heads ? w.q + h * dk : w.k + (h - m.n_heads) * dk;
      const double *got = h < m.n_heads ? w.q_rope + h * dk : w.k_rope + (h - m.n_heads) * dk;
      bcir_rope(x, 1, (int)dk, m.rope_base, (int)pos[i], ref);
      if (memcmp(ref, got, (size_t)dk * sizeof *ref)) {
        printf("rope: position %zu (call %zu) head %zu differs\n", pos[i], i, h);
        return 1;
      }
    }
  }
  printf("rope: %zu calls, every head bcir_rope's row\n", n);
  free(ref);
  bcir_llama_ws_free(&w);
  bcir_q8_model_free(&m);
  return 0;
}
"""


def test_the_decoders_rotation_is_bcir_ropes_rows_bit_for_bit():
    """DECODE-ROW: bcir_llama_op_rope fills the rotation's cosines and sines once a position and turns
    every query and key head by them. Each head's row must be bcir_rope's, byte for byte -- the
    standalone decoder's 1e-9 parity cannot see a table one position off, since attention reads only
    the positions' differences -- at positions taken repeated, backward and after a gap, as a GEM
    program may take them."""
    if _compiler() is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        artifact = directory / "model.bcirq8"
        config = dict(
            _CONFIG, hidden_size=64, intermediate_size=32
        )  # heads of 16: eight pairs each
        tensors = _hf_tensors(config)
        spec = spec_from_config(config)
        prepared = {name: ("F32", shape, values) for name, (shape, values) in tensors.items()}
        write_q8_decoder(
            artifact,
            spec,
            weights_from_tensors(spec, prepared),
            source_hashes=_HASHES,
            tokenizer_ids=_TOKENS,
        )
        assert spec.d_model // spec.n_heads == 16 and spec.n_kv_heads == 2
        source = directory / "rope_witness.c"
        source.write_text(_ROPE_WITNESS, encoding="utf-8")
        executable = directory / ("rope_witness.exe" if os.name == "nt" else "rope_witness")
        command = [
            _compiler(),
            "-std=c11",
            "-O2",
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(_RUNTIME),
            str(source),
            str(_RUNTIME / "bcir_q8_model.c"),
            str(_RUNTIME / "bcir_decode.c"),
            str(_RUNTIME / "bcir_ai_kernels.c"),
            str(_RUNTIME / "bcir_llama.c"),
            "-o",
            str(executable),
            *_link_args(),
        ]
        build = subprocess.run(command, capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        run = subprocess.run([str(executable), str(artifact)], capture_output=True, text=True)
        assert run.returncode == 0, (run.returncode, run.stdout, run.stderr)
        assert run.stdout == "rope: 14 calls, every head bcir_rope's row\n", run.stdout


if __name__ == "__main__":
    import sys

    module = sys.modules[__name__]
    names = sorted(
        name for name in dir(module) if name.startswith("test_") and callable(getattr(module, name))
    )
    failed = 0
    for name in names:
        try:
            getattr(module, name)()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc!r}")
    raise SystemExit(1 if failed else 0)
