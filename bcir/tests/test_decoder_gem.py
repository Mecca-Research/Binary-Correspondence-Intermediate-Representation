"""QUAL-2, the C half: a decoder program's StreamPack executed through GEM in C.

`bcir-qualify` (runtime/c/bcir_qualify_cli.c) loads a BCIRQ8 model and a decoder program's pack
-- the native wire, or the ASN.1 DER projection converted to it in C -- runs the program through
GEM's executor with the decoder kernels (bcir_decoder_gem.c) and the same request through the
monolithic runner (bcir_llama_generate_greedy), and exits 0 only when the two agree bit for
bit. These tests hold that verdict, and hold the C interpreter to the oracle's
(`test_decoder_program.py`): the same tokens, logits within the cross-rail bound of the BCIRQ8
parity gate, the same refusal for every pack the oracle refuses, and kernel-counted traffic
equal to the program's planned traffic, op for op.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path

from bcir.abi import encode
from bcir.asn1 import streampack as pack_asn1
from bcir.frontends.models.decoder_gem import DecoderProgramError, run_decoder_pack
from bcir.frontends.models.decoder_program import (
    OP_ARGMAX,
    OP_ATTENTION,
    OP_EMBED,
    OP_HEAD,
    OP_KV_APPEND,
    OP_MATVEC,
    OP_RMSNORM,
    RID_G_FINAL,
    RID_H,
    RID_X,
    layer_rid,
    program_traffic,
)
from bcir.tests import decoder_fixtures as fx

_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME = _ROOT / "runtime" / "c"

#: The host C compiler, or None (the tests return early); a capability for check_tests.py.
_CC = __import__("bcir.toolchain", fromlist=["host_c_compiler"]).host_c_compiler()

#: The C refusals (bcir_decoder_gem.h) the oracle's kinds mirror.
_C_REFUSAL = {"pack": -2, "program": -3, "order": -6}


def _link_args() -> list[str]:
    from bcir.toolchain import host_link_args

    return list(host_link_args(["-lm"]))


def _build(directory: Path) -> Path:
    executable = directory / ("bcir-qualify.exe" if os.name == "nt" else "bcir-qualify")
    sources = (
        "bcir_qualify_cli.c",
        "bcir_decoder_gem.c",
        "bcir_llama.c",
        "bcir_q8_model.c",
        "bcir_q4_kernel.c",
        "bcir_ai_kernels.c",
        "bcir_decode.c",
        "bcir_exec.c",
        "bcir_runtime.c",
        "bcir_asn1_streampack.c",
        "bcir_asn1.c",
        "bcir_sha256.c",
    )
    command = [_CC, "-std=c11", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"]
    command += ["-I", str(_RUNTIME), *(str(_RUNTIME / s) for s in sources)]
    command += ["-o", str(executable), *_link_args()]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return executable


def _qualify(exe: Path, model: Path, pack: Path, *, der: bool, reps: int = 2, logits=None):
    """(exit code, report or None, stderr) of one bcir-qualify run."""
    argv = [str(exe), str(model), str(pack), "--der" if der else "--native"]
    argv += ["--prompt", ",".join(map(str, fx.PROMPT)), "--reps", str(reps)]
    if logits is not None:
        argv += ["--logits-out", str(logits)]
    r = subprocess.run(argv, capture_output=True, text=True)
    report = json.loads(r.stdout) if r.stdout.strip() else None
    return r.returncode, report, r.stderr


def _refusal(stderr: str) -> int:
    """The interpreter's refusal code from bcir-qualify's message, e.g. `(... (-3)`."""
    tail = stderr.rsplit("(", 1)[-1]
    return int(tail.split(",")[0].split(")")[0])


def test_the_c_interpreter_is_the_monolithic_runner_bit_for_bit():
    """Tied and untied heads, the pack on the native wire and as DER converted in C: the GEM
    run and the monolithic run agree to the last bit (exit 0, every run identical), the C
    conversion of the DER is the oracle's native encoding octet for octet, and the tokens and
    logits are the oracle's interpreter's -- the four rails of one program agreeing."""
    if _CC is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        exe = _build(d)
        for tied in (False, True):
            spec, weights, model_bytes = fx.model(tied=tied)
            model = d / f"model-{tied}.bcirq8"
            model.write_bytes(model_bytes)
            pack, _ = fx.planned_pack(fx.program(spec))
            native = encode(pack)
            (d / "pack.bin").write_bytes(native)
            (d / "pack.der").write_bytes(pack_asn1.encode_pack(pack))
            oracle = run_decoder_pack(pack, spec, weights, fx.PROMPT)
            for der, path in ((False, d / "pack.bin"), (True, d / "pack.der")):
                logits_path = d / "logits.f64"
                rc, report, err = _qualify(exe, model, path, der=der, reps=3, logits=logits_path)
                assert rc == 0 and report["status"] == "PASS", (tied, der, err, report)
                assert report["logits"]["bitwise_identical"] is True
                assert report["logits"]["gem_runs_identical"] is True
                assert report["tokens"]["gem"] == report["tokens"]["monolithic"] == oracle.tokens
                assert report["program"]["claims"] == len(pack.segments)
                assert report["program"]["native_sha256"] == hashlib.sha256(native).hexdigest()
                raw = logits_path.read_bytes()
                c_logits = struct.unpack(f"<{spec.vocab_size}d", raw)
                assert max(abs(a - b) for a, b in zip(c_logits, oracle.logits)) <= 1e-9


def test_the_c_interpreter_counts_the_traffic_the_program_plans():
    """The kernels' counted octets (read and written, per op, one instrumented run) are the
    program's planned traffic (decoder_program.program_traffic), and every claim of every op
    ran once."""
    if _CC is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        exe = _build(d)
        spec, _, model_bytes = fx.model(tied=False)
        (d / "model.bcirq8").write_bytes(model_bytes)
        prog = fx.program(spec)
        pack, _ = fx.planned_pack(prog)
        (d / "pack.bin").write_bytes(encode(pack))
        rc, report, err = _qualify(exe, d / "model.bcirq8", d / "pack.bin", der=False)
        assert rc == 0, err
        planned = program_traffic(prog)
        claims = {}
        for ph in prog.module.phases:
            for c in ph.claims:
                claims[c.op] = claims.get(c.op, 0) + 1
        for op, row in report["operations"].items():
            assert row["claims"] == claims.get(op, 0), op
            if row["claims"]:
                assert (row["bytes_read"], row["bytes_written"]) == planned[op], op
        total = report["traffic"]
        assert (total["bytes_read"], total["bytes_written"]) == planned["total"]


def test_the_c_interpreter_refuses_what_the_oracle_refuses():
    """The oracle's refusal suite (test_decoder_program.py) on the C rail: each pack is refused
    by both interpreters, as the same kind -- a claim that is not this model's operation when
    the program loads, a claim outside its data-flow order when it runs."""
    if _CC is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        exe = _build(d)
        spec, weights, model_bytes = fx.model(tied=False)
        (d / "model.bcirq8").write_bytes(model_bytes)
        pack, _ = fx.planned_pack(fx.program(spec))
        q = fx.claim_at(pack, OP_MATVEC)
        attention = fx.claim_at(pack, OP_ATTENTION, 0)
        append = fx.claim_at(pack, OP_KV_APPEND, 0)
        e0, e1 = fx.claim_at(pack, OP_EMBED, 0), fx.claim_at(pack, OP_EMBED, 1)
        edits = (
            fx.edit_claim(pack, q, writes=(RID_X,)),
            fx.edit_claim(pack, q, block={"count": 1}),
            fx.edit_claim(pack, q, reads=(RID_H, layer_rid(spec.n_layers, 1))),
            fx.edit_claim(pack, fx.claim_at(pack, OP_RMSNORM), opcode="dec.softmax"),
            fx.edit_claim(pack, attention, block={"base": 1 << 40}),
            fx.drop_claim(pack, fx.claim_at(pack, OP_ARGMAX, 4)),
            fx.edit_claim(pack, e1, block={"base": 99}),
            fx.swap_positions(pack, append, attention),
            fx.swap_positions(pack, e0, e1),
            fx.drop_claim(pack, fx.claim_at(pack, OP_HEAD)),
            fx.drop_claim(pack, append),
        )
        skipped = fx.drop_claim(pack, append)
        skipped = fx.drop_claim(skipped, fx.claim_at(skipped, OP_ATTENTION, 0))
        edits += (skipped,)  # position 1's append would leave row 0 of the cache unwritten
        bare = fx.layerless(pack, RID_G_FINAL)
        edits += (
            fx.swap_positions(bare, fx.claim_at(bare, OP_EMBED, 0), fx.claim_at(bare, OP_EMBED, 1)),
        )
        kinds = []
        for edited in edits:
            try:
                run_decoder_pack(edited, spec, weights, fx.PROMPT)
            except DecoderProgramError as exc:
                kind = exc.kind
            else:
                raise AssertionError("the oracle ran an edited pack")
            (d / "edited.bin").write_bytes(encode(edited))
            rc, _, err = _qualify(exe, d / "model.bcirq8", d / "edited.bin", der=False)
            assert rc == 1 and _refusal(err) == _C_REFUSAL[kind], (kind, rc, err)
            kinds.append(kind)
        assert set(kinds) == {"program", "order"}


def test_a_c_run_reads_what_it_wrote_or_zeros():
    """A program that reads an activation before writing it (its first attention norm dropped)
    runs on the C rail as on the oracle -- to the same tokens -- and every repetition of it
    agrees: a run never sees what an earlier run left. (It is not the decoder, so it does not
    agree with the monolithic runner: exit 3.)"""
    if _CC is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        exe = _build(d)
        spec, weights, model_bytes = fx.model(tied=False)
        (d / "model.bcirq8").write_bytes(model_bytes)
        pack, _ = fx.planned_pack(fx.program(spec))
        holed = fx.drop_claim(pack, fx.claim_at(pack, OP_RMSNORM, 0))
        (d / "holed.bin").write_bytes(encode(holed))
        logits_path = d / "logits.f64"
        rc, report, err = _qualify(
            exe, d / "model.bcirq8", d / "holed.bin", der=False, reps=4, logits=logits_path
        )
        assert rc == 3 and report["status"] == "FAIL", (rc, err)
        assert report["logits"]["gem_runs_identical"] is True
        oracle = run_decoder_pack(holed, spec, weights, fx.PROMPT)
        assert report["tokens"]["gem"] == oracle.tokens
        c_logits = struct.unpack(f"<{spec.vocab_size}d", logits_path.read_bytes())
        assert max(abs(a - b) for a, b in zip(c_logits, oracle.logits)) <= 1e-9


def _build_harness(directory: Path) -> Path:
    executable = directory / ("test_decoder_gem.exe" if os.name == "nt" else "test_decoder_gem")
    sources = (
        "test_decoder_gem.c",
        "bcir_decoder_gem.c",
        "bcir_llama.c",
        "bcir_q8_model.c",
        "bcir_q4_kernel.c",
        "bcir_ai_kernels.c",
        "bcir_decode.c",
        "bcir_exec.c",
        "bcir_runtime.c",
    )
    command = [_CC, "-std=c11", "-O1", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"]
    command += ["-I", str(_RUNTIME), *(str(_RUNTIME / s) for s in sources)]
    command += ["-o", str(executable), *_link_args()]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return executable


def _load(harness: Path, model: Path, pack: Path) -> dict[str, int]:
    r = subprocess.run([str(harness), str(model), str(pack)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return {k: int(v) for k, v in (field.split("=") for field in r.stdout.split())}


def test_a_count_the_body_does_not_carry_is_refused_before_anything_is_allocated():
    """A header counting more segments than the body carries (its CRC resealed) is a malformed
    pack, refused at load by the semantic walk -- with nothing allocated: the interpreter sizes
    no table from a count it has not read. A load that allocates, accepted or refused later,
    gives everything back to its allocator on free."""
    if _CC is None:
        return
    import zlib

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        harness = _build_harness(d)
        spec, _, model_bytes = fx.model(tied=False)
        model = d / "model.bcirq8"
        model.write_bytes(model_bytes)
        pack, _ = fx.planned_pack(fx.program(spec))
        good = encode(pack)
        data = bytearray(good)
        struct.pack_into("<I", data, 20, 1 << 20)  # n_segments
        body = bytes(data[:-4])
        (d / "lying.bin").write_bytes(body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF))
        assert _load(harness, model, d / "lying.bin") == {
            "status": _C_REFUSAL["pack"], "allocations": 0, "live_after_free": 0
        }  # fmt: skip
        (d / "good.bin").write_bytes(good)
        loaded = _load(harness, model, d / "good.bin")
        assert loaded["status"] == 0 and loaded["allocations"] > 0, loaded
        assert loaded["live_after_free"] == 0
        wrong = fx.edit_claim(pack, fx.claim_at(pack, OP_MATVEC), writes=(RID_X,))
        (d / "wrong.bin").write_bytes(encode(wrong))
        refused = _load(harness, model, d / "wrong.bin")
        assert refused["status"] == _C_REFUSAL["program"] and refused["live_after_free"] == 0


def test_final_is_a_resource_of_its_own_on_the_c_rail():
    """The program with H written between the final norm and the head (test_decoder_program's
    witness) is, on the C rail too, the decoder: bit for bit the monolithic runner, whose own
    final row lives in h -- so the interpreter must not keep FINAL there."""
    if _CC is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        exe = _build(d)
        spec, _, model_bytes = fx.model(tied=False)
        (d / "model.bcirq8").write_bytes(model_bytes)
        pack, _ = fx.planned_pack(fx.program(spec))
        final = next(
            s.claim_id
            for s in pack.segments
            if s.opcode == OP_RMSNORM and s.reads[1] == RID_G_FINAL
        )
        extra = fx.insert_copy(pack, fx.claim_at(pack, OP_RMSNORM), after=final)
        (d / "extra.bin").write_bytes(encode(extra))
        rc, report, err = _qualify(exe, d / "model.bcirq8", d / "extra.bin", der=False)
        assert rc == 0 and report["logits"]["bitwise_identical"] is True, (rc, err)
