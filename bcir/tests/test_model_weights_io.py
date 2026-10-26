"""BCIRQ8 v1 persistence and standalone-C Llama parity gates."""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
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


def _model(*, tied: bool = False, **dimensions):
    config = dict(_CONFIG, tie_word_embeddings=tied, **dimensions)
    tensors = _hf_tensors(config)
    prepared = {name: ("F32", shape, values) for name, (shape, values) in tensors.items()}
    spec = spec_from_config(config)
    return spec, weights_from_tensors(spec, prepared)


def _write(path: Path, *, tied: bool = False, **dimensions):
    """The toy decoder (_CONFIG, its dimensions overridden by `dimensions`) as a BCIRQ8 artifact. _WIDE is a toy
    whose rows are a whole number of groups (64 = 2 x 32), so the C loader packs its head (HEAD-PACK)."""
    spec, weights = _model(tied=tied, **dimensions)
    metadata = write_q8_decoder(path, spec, weights, source_hashes=_HASHES, tokenizer_ids=_TOKENS)
    return spec, weights, metadata


_WIDE = {"hidden_size": 64, "intermediate_size": 32}


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
        # DECODE-THREADS: the deterministic pool bcir-llama --threads lends the decoder (pthreads here, Win32 threads
        # with no flag on Windows)
        str(_RUNTIME / "bcir_tensor.c"),
        *([] if sys.platform == "win32" else ["-pthread"]),
        "-o",
        str(executable),
        *_link_args(),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return executable


def test_standalone_c_q8_decoder_matches_python_for_tied_and_untied_heads():
    """Each head tied and untied, in the toy whose rows straddle its groups (no pack) and in _WIDE (HEAD-PACK: the
    head packed at load); and with --threads 3 (DECODE-THREADS), every operation split and by the defaults, the same
    tokens and logits byte for byte."""
    if _compiler() is None:
        return
    prompt = [1, 2, 3]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        executable = _build_cli(directory)
        for tied, wide in ((False, False), (True, False), (False, True), (True, True)):
            artifact = directory / f"model-{tied}{'-wide' if wide else ''}.bcirq8"
            _write(artifact, tied=tied, **(_WIDE if wide else {}))
            spec, weights, _ = read_q8_decoder(artifact)
            expected_ids = reference_decode(prompt, spec, weights, 2)
            python_logits = next_token_logits(prompt + expected_ids[:1], spec, weights)
            logits_path = directory / f"logits-{tied}-{wide}.f64"
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
            # DECODE-THREADS: three threads with every operation split, then the default thresholds -- the same
            # tokens and the same logits, byte for byte
            for split in (["--split-min", "0"], []):
                threaded_path = directory / f"logits-{tied}-{wide}-threads.f64"
                threaded = subprocess.run(
                    [
                        str(executable),
                        "--model",
                        str(artifact),
                        "--prompt-ids",
                        "1,2,3",
                        "--max-new",
                        "2",
                        "--logits-out",
                        str(threaded_path),
                        "--threads",
                        "3",
                        *split,
                    ],
                    capture_output=True,
                    text=True,
                )
                assert threaded.returncode == 0, threaded.stderr
                assert threaded.stdout == result.stdout and threaded_path.read_bytes() == raw, split
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
        # DECODE-THREADS: a thread count outside 1 to 64, or a threshold that is not a count, is a usage error
        for bad in (
            ["--threads", "0"],
            ["--threads", "65"],
            ["--threads", "x"],
            ["--split-min", "-1"],
        ):
            result = subprocess.run(
                [
                    str(executable),
                    "--model",
                    str(directory / "model-False.bcirq8"),
                    "--prompt-ids",
                    "1",
                    "--max-new",
                    "1",
                    *bad,
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 2 and "usage" in result.stderr, (bad, result.stderr)


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


_HEAD_WITNESS = r"""
#include "bcir_ai_kernels.h"
#include "bcir_llama_ops.h"
#include "bcir_q8_model.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int bcir_ai_q8_pack_probe(size_t *passes, size_t *blocks);

/* an allocator that counts its live allocations and refuses the fail_at-th */
typedef struct { size_t attempt, fail_at, live; } faults;
static void *fault_allocate(void *context, size_t size) {
  faults *f = context;
  void *p;
  if (++f->attempt == f->fail_at) return NULL;
  p = malloc(size);
  f->live += p != NULL;
  return p;
}
static void *fault_reallocate(void *context, void *allocation, size_t size) {
  (void)context;
  return realloc(allocation, size);
}
static void fault_deallocate(void *context, void *allocation) {
  faults *f = context;
  if (allocation) --f->live;
  free(allocation);
}

/* HEAD-PACK: the loader under an allocator refusing each of its allocations in turn -- every refusal fails the
 * load with nothing live and no pack, and a whole load's allocations, the pack's among them, all freed by
 * bcir_q8_model_free; then the decoder's head (bcir_llama_op_head, which both dispatchers call) against the row
 * dots over the head's rows as they are, byte for byte, at rows of either sign and of two magnitudes, and whether
 * it read the pack the loader built, as the kernel unit's probe saw each call. */
int main(int argc, char **argv) {
  bcir_q8_model m;
  char err[256];
  uint64_t state = 0x9E3779B97F4A7C15u;
  const bcir_q8_tensor *head;
  double *row, *got, *ref;
  size_t passes = 0, blocks = 0, i, k, on_pack = 0, allocations;
  int path = 0;
  faults f = {0, 0, 0};
  bcir_host_allocator counted = {&f, fault_allocate, fault_reallocate, fault_deallocate};
  if (argc != 2 || bcir_q8_model_load_with_allocator(argv[1], &m, err, sizeof err, &counted)) return 2;
  allocations = f.attempt;
  bcir_q8_model_free(&m);
  if (f.live != 0) {
    printf("a whole load left %zu allocations live\n", f.live);
    return 1;
  }
  for (i = 1; i <= allocations; i++) {
    f.attempt = 0, f.fail_at = i, f.live = 0;
    if (!bcir_q8_model_load_with_allocator(argv[1], &m, err, sizeof err, &counted) || f.live != 0 ||
        m.head_pack != NULL || m.storage != NULL) {
      printf("allocation %zu refused: the load did not fail clean\n", i);
      return 1;
    }
  }
  printf("loads: %zu allocations, each refused in turn, nothing left live\n", allocations);
  bcir_q8_model_init(&m);
  if (bcir_q8_model_load(argv[1], &m, err, sizeof err)) return 2;
  if (!bcir_llama_model_ready(&m)) return 3;
  head = bcir_llama_head_tensor(&m);
  row = malloc(m.d_model * sizeof *row);
  got = malloc(m.vocab_size * sizeof *got);
  ref = malloc(m.vocab_size * sizeof *ref);
  if (!row || !got || !ref) return 4;
  for (i = 0; i < 12; i++) {
    for (k = 0; k < m.d_model; k++) {
      state ^= state << 13, state ^= state >> 7, state ^= state << 17;
      row[k] = ((double)(int64_t)(state >> 11) / 4503599627370496.0 - 1.0) * (i % 3 == 2 ? 1e-3 : 4.0);
    }
    (void)bcir_ai_q8_pack_probe(NULL, NULL);
    if (bcir_llama_op_head(&m, NULL, row, got)) return 5;
    path = bcir_ai_q8_pack_probe(&passes, &blocks);
    on_pack += path != 0;
    if (bcir_ai_q8_rows_dot_f64_prevalidated(row, m.d_model, head->codes, head->element_count, head->exponents,
                                             head->group_count, m.vocab_size, m.group_size, ref))
      return 6;
    if (memcmp(got, ref, m.vocab_size * sizeof *ref)) {
      printf("row %zu: a logit differs\n", i);
      return 1;
    }
  }
  printf("instances %d\nhead: 12 rows, every logit the rows' own; %s, %zu of 12 calls on it (path %d, %zu passes, "
         "%zu blocks)\n",
         bcir_ai_q8_instances(), m.head_pack ? "packed" : "no pack", on_pack, path, passes, blocks);
  free(row);
  free(got);
  free(ref);
  bcir_q8_model_free(&m);
  return 0;
}
"""


def test_the_decoders_head_reads_the_pack_and_is_the_rows_own_bit_for_bit():
    """HEAD-PACK: the loader's pack is one more allocation where it packs the head -- each refused in turn fails
    the load with nothing left live, and bcir_q8_model_free releases it with the rest -- and bcir_llama_op_head, the
    head both dispatchers call, gives every logit of the row dots over the head's rows as they are, byte for byte,
    tied and untied. The loader packs a head whose rows are a whole number
    of groups (_WIDE: 64 = 2 x 32), and the head reads that pack, all 64 rows in two 32-row passes on a vector
    instance (the portable loop over the pack on a host without one); the toy whose rows straddle its groups has
    no pack, and the head reads its rows as they are."""
    if _compiler() is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        source = directory / "head_witness.c"
        source.write_text(_HEAD_WITNESS, encoding="utf-8")
        executable = directory / ("head_witness.exe" if os.name == "nt" else "head_witness")
        command = [
            _compiler(),
            "-std=c11",
            "-O2",
            "-ffp-contract=off",
            "-DBCIR_AI_INSTANCE_PROBE",
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
        for tied in (False, True):
            for wide in (False, True):
                artifact = directory / f"model-{tied}-{wide}.bcirq8"
                _write(artifact, tied=tied, **(_WIDE if wide else {}))
                run = subprocess.run(
                    [str(executable), str(artifact)], capture_output=True, text=True
                )
                assert run.returncode == 0, (tied, wide, run.returncode, run.stdout, run.stderr)
                loads, _, rest = run.stdout.partition("\n")
                # storage, the tensors and the spans' scratch, and the pack where the loader packs the head
                allocations = 4 if wide else 3
                assert (
                    loads
                    == f"loads: {allocations} allocations, each refused in turn, nothing left live"
                ), loads
                first, _, rest = rest.partition("\n")
                instances = int(first.removeprefix("instances "))
                assert instances & 1 and not instances & ~7, run.stdout
                if not wide:
                    tail = "no pack, 0 of 12 calls on it (path 0, 0 passes, 0 blocks)"
                elif instances & 6:  # a vector instance: the 64 rows two passes of 32
                    tail = "packed, 12 of 12 calls on it (path 1, 2 passes, 0 blocks)"
                else:  # the portable loop over the pack
                    tail = "packed, 12 of 12 calls on it (path 2, 0 passes, 0 blocks)"
                assert rest == f"head: 12 rows, every logit the rows' own; {tail}\n", run.stdout


_THREADS_WITNESS = r"""
#include "bcir_llama.h"
#include "bcir_llama_ops.h"
#include "bcir_tensor.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* a runner of its own: every part on the caller's thread, the last first -- the parts' order is no part of their
 * outputs -- each call and part counted */
typedef struct { size_t calls, parts; } counted;
static void reverse_run(void *self, bcir_llama_task_fn task, void *ctx, size_t tasks) {
  counted *c = self;
  size_t t;
  c->calls++;
  c->parts += tasks;
  for (t = tasks; t-- > 0;) task(ctx, t);
}
static void pool_run(void *self, bcir_llama_task_fn task, void *ctx, size_t tasks) {
  bcir_tensor_pool_run((bcir_tensor_pool *)self, task, ctx, tasks);
}
/* a runner that leaves an operation's last part unrun */
static void lazy_run(void *self, bcir_llama_task_fn task, void *ctx, size_t tasks) {
  size_t t;
  (void)self;
  for (t = 0; t + 1u < tasks; t++) task(ctx, t);
}

/* DECODE-THREADS: a greedy decode (prompt 1, 2, 3; six new tokens) lent a runner at every worker count with every
 * operation split, then the trainer's pool at two to four threads -- every token and every final logit the decode's
 * without one, byte for byte; the default thresholds, which split none of a toy's operations; the runners a decode
 * and a workspace refuse; and a runner leaving a part unrun, which fails the decode rather than let it go on. */
int main(int argc, char **argv) {
  bcir_q8_model m;
  bcir_llama_ws w;
  char err[256];
  int32_t prompt[3] = {1, 2, 3}, ref_ids[6], ids[6];
  double *ref_logits, *logits;
  bcir_host_allocator alloc = bcir_host_allocator_default();
  counted c;
  unsigned workers;
  if (argc != 2 || bcir_q8_model_load(argv[1], &m, err, sizeof err)) return 2;
  ref_logits = malloc(m.vocab_size * sizeof *ref_logits);
  logits = malloc(m.vocab_size * sizeof *logits);
  if (!ref_logits || !logits) return 2;
  if (bcir_llama_generate_greedy_with_runner(&m, prompt, 3, 6, ref_ids, ref_logits, &alloc, NULL)) return 3;
  for (workers = 1; workers <= 5; workers++) {
    bcir_llama_runner r = {reverse_run, NULL, 0u, 0u, 0u};
    r.self = &c, r.workers = workers;
    memset(&c, 0, sizeof c);
    if (bcir_llama_generate_greedy_with_runner(&m, prompt, 3, 6, ids, logits, &alloc, &r)) return 4;
    if (memcmp(ids, ref_ids, sizeof ids) || memcmp(logits, ref_logits, m.vocab_size * sizeof *logits)) {
      printf("%u workers: a token or a logit differs\n", workers);
      return 1;
    }
    printf("%u workers: %zu splits, %zu parts\n", workers, c.calls, c.parts);
  }
  {
    bcir_llama_runner r = {reverse_run, NULL, 4u, BCIR_LLAMA_SPLIT_MIN, BCIR_LLAMA_HEAD_SPLIT_MIN};
    r.self = &c;
    memset(&c, 0, sizeof c);
    if (bcir_llama_generate_greedy_with_runner(&m, prompt, 3, 6, ids, logits, &alloc, &r)) return 4;
    if (memcmp(ids, ref_ids, sizeof ids) || memcmp(logits, ref_logits, m.vocab_size * sizeof *logits)) return 1;
    printf("defaults: %zu splits\n", c.calls);
  }
  for (workers = 2; workers <= 4; workers++) {
    size_t bytes = bcir_tensor_pool_size(workers);
    void *memory = bytes ? malloc(bytes) : NULL;
    bcir_tensor_pool *pool = memory ? bcir_tensor_pool_init(memory, bytes, workers) : NULL;
    bcir_llama_runner r = {pool_run, NULL, 0u, 0u, 0u};
    int rc;
    if (pool == NULL) {
      printf("pool of %u: none on this host\n", workers);
      free(memory);
      continue;
    }
    r.self = pool, r.workers = workers;
    rc = bcir_llama_generate_greedy_with_runner(&m, prompt, 3, 6, ids, logits, &alloc, &r);
    bcir_tensor_pool_stop(pool);
    free(memory);
    if (rc) return 4;
    if (memcmp(ids, ref_ids, sizeof ids) || memcmp(logits, ref_logits, m.vocab_size * sizeof *logits)) {
      printf("pool of %u: a token or a logit differs\n", workers);
      return 1;
    }
    printf("pool of %u: every token and logit the same\n", workers);
  }
  {
    /* refused before any allocation: no function for more than one worker, more workers than parts; a workspace
     * keeps the runner it had */
    bcir_llama_runner none = {NULL, NULL, 2u, 0u, 0u}, many = {reverse_run, NULL, BCIR_LLAMA_MAX_PARTS + 1u, 0u, 0u};
    bcir_llama_runner one = {NULL, NULL, 1u, 0u, 0u}, good = {reverse_run, NULL, 3u, 7u, 9u};
    if (bcir_llama_generate_greedy_with_runner(&m, prompt, 3, 6, ids, logits, &alloc, &none) != -1 ||
        bcir_llama_generate_greedy_with_runner(&m, prompt, 3, 6, ids, logits, &alloc, &many) != -1)
      return 5;
    if (bcir_llama_generate_greedy_with_runner(&m, prompt, 3, 6, ids, logits, &alloc, &one) ||
        memcmp(ids, ref_ids, sizeof ids))
      return 5;
    if (bcir_llama_ws_init(&w, &m, 16u, &alloc)) return 2;
    if (w.runner.run != NULL || w.runner.workers != 0u || bcir_llama_ws_set_runner(&w, &good) ||
        bcir_llama_ws_set_runner(&w, &none) != -1 || bcir_llama_ws_set_runner(&w, &many) != -1 ||
        w.runner.run != reverse_run || w.runner.workers != 3u || w.runner.split_min != 7u ||
        w.runner.head_split_min != 9u || bcir_llama_ws_set_runner(&w, NULL) || w.runner.run != NULL ||
        w.runner.workers != 0u)
      return 6;
    bcir_llama_ws_free(&w);
  }
  printf("refusals: a decode and a workspace refuse what they must\n");
  {
    bcir_llama_runner lazy = {lazy_run, NULL, 3u, 0u, 0u};
    if (bcir_llama_generate_greedy_with_runner(&m, prompt, 3, 6, ids, logits, &alloc, &lazy) != -1) return 7;
  }
  printf("a part left unrun: the decode fails\n");
  free(ref_logits);
  free(logits);
  bcir_q8_model_free(&m);
  return 0;
}
"""


def _split_parts(outputs: int, unit: int, workers: int) -> int:
    """DECODE-THREADS' partition, stated here apart from the C: as many parts as workers, at most one a unit of
    outputs, the chunk the outputs over the parts rounded up to a unit -- the parts the runner runs, 0 unsplit."""
    parts = min(workers, outputs // unit)
    if workers < 2 or parts < 2:
        return 0
    share = (outputs + parts - 1) // parts
    chunk = (share + unit - 1) // unit * unit
    return (outputs + chunk - 1) // chunk


def test_the_decoder_lent_a_runner_is_the_decoder_bit_for_bit():
    """DECODE-THREADS: a decode lent a runner -- one of the test's own that runs the parts last first, at one to
    five workers with every operation split, and the trainer's pool at two to four threads -- gives every token and
    every final logit of the decode without one, byte for byte, tied and untied, with the head packed (_WIDE) and
    not. The parts are exactly the partition stated above, matvec by matvec and head by head: eight steps of two
    layers' seven matvecs, and six heads where the loader packed one (a head over its rows as they are is never
    split). The defaults split none of a toy's operations; a runner of no function, or of too many workers, fails
    the decode before any allocation, and a workspace keeps its runner when it refuses another; a runner that leaves
    a part unrun fails the decode."""
    if _compiler() is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        source = directory / "threads_witness.c"
        source.write_text(_THREADS_WITNESS, encoding="utf-8")
        executable = directory / ("threads_witness.exe" if os.name == "nt" else "threads_witness")
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
            str(_RUNTIME / "bcir_tensor.c"),
            *([] if sys.platform == "win32" else ["-pthread"]),
            "-o",
            str(executable),
            *_link_args(),
        ]
        build = subprocess.run(command, capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        for tied in (False, True):
            for wide in (False, True):
                artifact = directory / f"model-{tied}-{wide}.bcirq8"
                _write(artifact, tied=tied, **(_WIDE if wide else {}))
                spec, _, _ = read_q8_decoder(artifact)
                run = subprocess.run(
                    [str(executable), str(artifact)], capture_output=True, text=True
                )
                assert run.returncode == 0, (tied, wide, run.returncode, run.stdout, run.stderr)
                d, ff = spec.d_model, spec.d_ff
                kvd = spec.n_kv_heads * (d // spec.n_heads)
                outs = (d, kvd, kvd, d, ff, ff, d)  # each layer's seven matvecs
                packed = d % 32 == 0
                lines = run.stdout.splitlines()
                for workers in range(1, 6):
                    parts = [_split_parts(out, 8, workers) for out in outs]
                    head = _split_parts(spec.vocab_size, 32, workers) if packed else 0
                    splits = 8 * spec.n_layers * sum(1 for n in parts if n) + 6 * (head > 0)
                    total = 8 * spec.n_layers * sum(parts) + 6 * head
                    assert (
                        lines[workers - 1] == f"{workers} workers: {splits} splits, {total} parts"
                    ), (
                        tied,
                        wide,
                        lines,
                    )
                assert lines[5] == "defaults: 0 splits", lines
                pools = lines[6:9]
                assert all(
                    line.endswith(": every token and logit the same")
                    or line.endswith(": none on this host")
                    for line in pools
                ), pools
                if sys.platform != "win32":  # pthreads here: every pool runs
                    assert all(
                        line.endswith(": every token and logit the same") for line in pools
                    ), pools
                assert lines[9:] == [
                    "refusals: a decode and a workspace refuse what they must",
                    "a part left unrun: the decode fails",
                ], lines


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
