#!/usr/bin/env python3
"""Bounded malformed-input campaign over StreamPack, BCAB, BCIRQ8, and C decoders.

Python surfaces always run. The C sanitizer/libFuzzer rail is invoked when clang
and compiler-rt are available; otherwise it is recorded as UNAVAILABLE/SKIPPED.
A campaign that hits fewer than the required Python surfaces is INVALID.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import zlib
from pathlib import Path
from typing import Any, Callable

try:
    from tools.security.proc_bounds import put_down_group
except ModuleNotFoundError:  # script execution: sys.path[0] is tools/security
    from proc_bounds import put_down_group

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_PYTHON = ("streampack", "bcab", "bcirq8")
DECODE_TIMEOUT = 10.0
# Each surface's DELIBERATE rejection type, and only that. A blanket tuple
# (IndexError, KeyError, struct.error, OverflowError, zlib.error...) counted
# unchecked indexing and unpacking as "graceful rejection", so a decoder
# could regress to raising IndexError on every mutation and the campaign
# would still report PASS as long as the seed decoded. Implementation
# exceptions are ungraceful findings, which is what they always were.
# Measured against the live decoders: 400 mutations per surface raise the
# declared type and nothing else, so this narrowing hides no real rejection.
_DECLARED_REJECTIONS = {
    "streampack": ("bcir.abi.streampack_abi", "AbiError"),
    "bcab": ("bcir.abi.artifact_bundle", "BundleError"),
    # weights_io raises ValueError for every content rejection; its OSError
    # is re-raised as RuntimeError by _decode_q8_bytes so an environmental
    # failure can never pass as a decode verdict.
    "bcirq8": ("builtins", "ValueError"),
}


class _DecodeHang(Exception):
    """Raised by the watchdog when a single decode exceeds its wall bound."""


def _bounded_decode(decode: Callable[[bytes], Any], blob: bytes, seconds: float) -> Any:
    """Run one decode under a wall-clock bound. A decoder that stops
    terminating is the exact regression this campaign hunts; it must become
    a structured finding, not a hung required job. Where SIGALRM is
    unavailable (Windows, a non-main thread) the call runs unbounded — the
    watchdog is a POSIX rail, like the C campaign."""
    if (
        os.name == "nt"
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        return decode(blob)

    def _expired(signum: int, frame: Any) -> None:
        raise _DecodeHang(f"decode exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return decode(blob)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _graceful_for(surface: str) -> tuple[type[BaseException], ...]:
    """The exception classes this surface rejects malformed input WITH.

    An unknown surface has no declared rejection, so every exception it
    raises is ungraceful — a new surface must declare its contract to be
    counted, never inherit a blanket one."""
    import importlib
    entry = _DECLARED_REJECTIONS.get(surface)
    if entry is None:
        return ()
    module, name = entry
    return (getattr(importlib.import_module(module), name),)


def _mutate(data: bytes, rng: random.Random) -> bytes:
    if not data:
        return rng.randbytes(rng.randint(1, 16))
    blob = bytearray(data)
    mode = rng.randint(0, 2)
    if mode == 0:
        for _ in range(rng.randint(1, max(1, len(blob) // 8))):
            blob[rng.randrange(len(blob))] ^= 1 << rng.randint(0, 7)
        return bytes(blob)
    if mode == 1:
        return bytes(blob[:rng.randint(0, len(blob))])
    blob.extend(rng.randbytes(rng.randint(1, 24)))
    return bytes(blob)


def _probe(name: str, decode: Callable[[bytes], Any], seed: bytes,
           rng: random.Random, mutations: int,
           timeout: float = DECODE_TIMEOUT) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    accepted = 0
    rejected = 0
    graceful = _graceful_for(name)
    try:
        _bounded_decode(decode, seed, timeout)
        accepted += 1
    except _DecodeHang:
        findings.append({"kind": "decoder-hang"})
    except graceful:
        findings.append({"kind": "seed-rejected"})
        rejected += 1
    except Exception as exc:  # noqa: BLE001 - campaign records unexpected classes
        findings.append({"kind": "ungraceful-seed", "type": type(exc).__name__})
    for _ in range(mutations):
        try:
            _bounded_decode(decode, _mutate(seed, rng), timeout)
            accepted += 1
        except _DecodeHang:
            findings.append({"kind": "decoder-hang"})
        except graceful:
            rejected += 1
        except Exception as exc:  # noqa: BLE001
            findings.append({"kind": "ungraceful", "type": type(exc).__name__})
    # A decoder that regresses to accepting EVERYTHING must not be greener for
    # it: one deterministic format-invalid probe per surface, whose acceptance
    # is a finding rather than another accepted count.
    try:
        _bounded_decode(decode, b"\x00", timeout)
        findings.append({"kind": "invalid-accepted"})
    except _DecodeHang:
        findings.append({"kind": "decoder-hang"})
    except graceful:
        rejected += 1
    except Exception as exc:  # noqa: BLE001
        findings.append({"kind": "ungraceful-invalid", "type": type(exc).__name__})
    return {
        "surface": name,
        "state": "FAIL" if findings else "PASS",
        "accepted": accepted,
        "rejected": rejected,
        "mutations": mutations,
        "findings": findings,
    }


def _streampack_seed() -> bytes:
    from bcir.abi import encode
    from bcir.examples import vector_add
    from bcir.gem import hydrate
    from bcir.kbcir import optimize
    from bcir.kbcir.cost import TargetProfile, Theta
    module = vector_add(32)
    pack = hydrate(module, optimize(module, TargetProfile.x86_avx512(), Theta.cool()))
    return encode(pack)


def _bcab_seed(pack: bytes) -> bytes:
    from bcir.abi.artifact_bundle import (
        ArtifactBundle, ArtifactFormat, ArtifactKind, ArtifactVariant, encode_bundle,
    )
    bundle = ArtifactBundle((
        ArtifactVariant(
            "00-root", ArtifactKind.STREAM_PACK, ArtifactFormat.STREAM_PACK,
            pack, channel="host", portable=True,
        ),
    ), "00-root", "00-root", 7, 3)
    return encode_bundle(bundle)


def _q8_seed() -> bytes:
    from bcir.frontends.models.hf_ingest import spec_from_config, weights_from_tensors
    from bcir.frontends.models.weights_io import write_q8_decoder
    from bcir.tests.test_model_ingest import _CONFIG, _hf_tensors
    spec = spec_from_config(_CONFIG)
    prepared = {
        name: ("F32", shape, values)
        for name, (shape, values) in _hf_tensors(_CONFIG).items()
    }
    weights = weights_from_tensors(spec, prepared)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.bcirq8"
        write_q8_decoder(
            path, spec, weights,
            source_hashes={"model": "00" * 32, "config": "11" * 32, "tokenizer": "22" * 32},
            tokenizer_ids={"bos": 1, "eos": 2, "pad": 0, "context_length": 32},
        )
        return path.read_bytes()


def _decode_q8_bytes(data: bytes) -> Any:
    from bcir.frontends.models.weights_io import read_q8_decoder
    # Campaign I/O is environmental, not a decode verdict: a disk-full or
    # permission failure in this plumbing must surface as a finding
    # (RuntimeError sits outside the graceful set), never count as a
    # graceful rejection that keeps the surface green.
    try:
        tmp = tempfile.TemporaryDirectory()
    except OSError as exc:
        raise RuntimeError(f"campaign I/O failed: {exc}") from exc
    with tmp:
        path = Path(tmp.name) / "mut.bcirq8"
        try:
            path.write_bytes(data)
        except OSError as exc:
            raise RuntimeError(f"campaign I/O failed: {exc}") from exc
        try:
            return read_q8_decoder(path)
        except OSError as exc:
            # The decoder's content rejections are ValueError by contract; an
            # OSError out of its open/fstat/read is the filesystem failing
            # mid-campaign, not a decode verdict.
            raise RuntimeError(f"campaign I/O failed: {exc}") from exc


def run_python_campaign(mutations: int, seed: int) -> list[dict[str, Any]]:
    from bcir.abi.artifact_bundle import decode_bundle
    from bcir.abi.streampack_abi import decode as decode_pack
    rng = random.Random(seed)
    pack = _streampack_seed()
    return [
        _probe("streampack", decode_pack, pack, rng, mutations),
        _probe("bcab", decode_bundle, _bcab_seed(pack), rng, mutations),
        _probe("bcirq8", _decode_q8_bytes, _q8_seed(), rng, mutations),
    ]


CAMPAIGN_OUTPUT_CAP = 1 << 20  # per stream; fuzzers log diagnostics, not payload


def _put_down_group(proc: Any) -> None:
    """Kill the wrapper's whole process tree — its per-target subshells hold
    the pipes. One predicate for every rail (POSIX session, Windows tree)."""
    put_down_group(proc)


def _drain_capped(stream: Any, chunks: list[bytes], state: dict[str, Any]) -> None:
    """Read one campaign pipe under a byte budget; on overflow the campaign
    group is put down and the pipe kept draining unretained so no writer
    can block on it."""
    received = 0
    while True:
        chunk = stream.read(65536)
        if not chunk:
            return
        if state["overflow"]:
            continue
        received += len(chunk)
        if received > CAMPAIGN_OUTPUT_CAP:
            state["overflow"] = True
            _put_down_group(state["proc"])
            continue
        chunks.append(chunk)


def run_c_campaign(root: Path, runs: int, seconds: int) -> dict[str, Any]:
    script = root / "tools" / "c" / "fuzz_streampack.sh"
    bash = shutil.which("bash")
    # The wrapper honors CLANG, so this preflight must resolve the same
    # toolchain: checking only an unversioned `clang` on PATH skipped hosts
    # that had configured a versioned or custom one, and --require-c then
    # failed a campaign that would have run.
    configured = os.environ.get("CLANG")
    clang = shutil.which(configured) if configured else shutil.which("clang")
    if os.name == "nt":
        return {
            "state": "UNAVAILABLE/SKIPPED",
            "reason": "native Windows host; C fuzzer is a POSIX rail",
        }
    if not bash or not script.is_file():
        return {"state": "UNAVAILABLE/SKIPPED", "reason": "bash or fuzz_streampack.sh missing"}
    if not clang:
        missing = configured or "clang"
        return {"state": "UNAVAILABLE/SKIPPED", "reason": f"{missing} missing"}
    env = dict(os.environ)
    env.update({
        "FUZZ_RUNS": str(runs),
        "FUZZ_MAX_TOTAL_TIME": str(seconds),
        "FUZZ_JOBS": "2",
    })
    # ~15 targets on 2 workers can each reach the per-target time bound, plus
    # compile time — a timeout sized to one target aborts healthy campaigns.
    timeout = 180 + seconds * 8
    # Its own session: the wrapper backgrounds per-target subshells, and
    # only a process-group kill enforces the wall bound on the whole tree.
    # Bytes, not text=True: sanitizer and fuzzer binaries write raw bytes,
    # and a strict decode would crash the campaign instead of recording it.
    proc = subprocess.Popen(
        [bash, str(script)],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name != "nt",
    )
    # Bounded drains instead of communicate(): a concurrent fuzzer spraying
    # output would otherwise accumulate in memory until the wall bound.
    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []
    state: dict[str, Any] = {"overflow": False, "proc": proc}
    drains = [
        threading.Thread(
            target=_drain_capped, args=(proc.stdout, out_chunks, state), daemon=True,
        ),
        threading.Thread(
            target=_drain_capped, args=(proc.stderr, err_chunks, state), daemon=True,
        ),
    ]
    for drain in drains:
        drain.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _put_down_group(proc)
        proc.wait()
    for drain in drains:
        drain.join(5.0)
    stdout = b"".join(out_chunks).decode("utf-8", "replace")
    stderr = b"".join(err_chunks).decode("utf-8", "replace")
    if any(drain.is_alive() for drain in drains):
        # A pipe still open after the group was reaped (or exited) means an
        # escaped descendant holds it; the campaign cannot vouch for output
        # it never saw the end of.
        _put_down_group(proc)
        return {
            "state": "FAIL",
            "reason": "descendant processes still hold the campaign's pipes",
            "stdout_tail": stdout[-800:],
            "stderr_tail": stderr[-800:],
        }
    if timed_out:
        # The captured output is the only evidence of which target hung.
        return {
            "state": "FAIL",
            "reason": f"C campaign timed out after {timeout}s",
            "stdout_tail": stdout[-800:],
            "stderr_tail": stderr[-800:],
        }
    if state["overflow"]:
        return {
            "state": "FAIL",
            "reason": (
                f"C campaign output exceeded {CAMPAIGN_OUTPUT_CAP} bytes per stream"
            ),
            "stdout_tail": stdout[-800:],
            "stderr_tail": stderr[-800:],
        }
    combined = (stdout + stderr).lower()
    if "skipping" in combined and proc.returncode == 0:
        return {
            "state": "UNAVAILABLE/SKIPPED",
            "reason": "clang has no libFuzzer/compiler-rt",
            "returncode": proc.returncode,
        }
    if "fuzzing unseeded" in combined and proc.returncode == 0:
        # The script exits 0 after "SKIP BCIRQ8 seed corpus"; random mutations
        # cannot pass BCIRQ8's checksum layers, so the target was not really
        # exercised — record unavailability (fatal under --require-c).
        return {
            "state": "UNAVAILABLE/SKIPPED",
            "reason": "BCIRQ8 seed corpus unavailable; campaign ran unseeded",
            "returncode": proc.returncode,
        }
    return {
        "state": "PASS" if proc.returncode == 0 else "FAIL",
        "returncode": proc.returncode,
        "stdout_tail": stdout[-800:],
        "stderr_tail": stderr[-800:],
    }


def run_campaign(
    root: Path, mutations: int, seed: int, fuzz_runs: int, fuzz_seconds: int,
    require_c: bool = False,
) -> dict[str, Any]:
    if mutations < 1 or fuzz_runs < 1 or fuzz_seconds < 1:
        # A zero iteration budget on any rail turns the malformed-input
        # campaign into a seed-only smoke test that can only ever pass;
        # refuse the configuration outright.
        return {
            "state": "INVALID/VACUOUS",
            "error": (
                "mutations/fuzz-runs/fuzz-seconds must all be positive, got "
                f"{mutations}/{fuzz_runs}/{fuzz_seconds}"
            ),
            "python": [],
            "c_decoder": {"state": "UNAVAILABLE/SKIPPED", "reason": "vacuous configuration"},
        }
    python = run_python_campaign(mutations, seed)
    names = {item["surface"] for item in python}
    report: dict[str, Any] = {
        "state": "PASS",
        "python": python,
        "c_decoder": run_c_campaign(root, fuzz_runs, fuzz_seconds),
    }
    if names != set(REQUIRED_PYTHON):
        report["state"] = "INVALID/VACUOUS"
        report["error"] = f"missing python surfaces: {sorted(set(REQUIRED_PYTHON) - names)}"
        return report
    if any(item["findings"] for item in python):
        report["state"] = "FAIL"
    if report["c_decoder"]["state"] == "FAIL":
        report["state"] = "FAIL"
    if require_c and report["c_decoder"]["state"] != "PASS":
        # CI installs clang/compiler-rt specifically for this campaign; there,
        # a silently unavailable C rail must fail the job, not stay green.
        report["state"] = "FAIL"
        report.setdefault(
            "error",
            f"C campaign required but {report['c_decoder']['state']}: "
            f"{report['c_decoder'].get('reason', 'no reason recorded')}",
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_decoder_campaign")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--mutations", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--fuzz-runs", type=int, default=200)
    parser.add_argument("--fuzz-seconds", type=int, default=8)
    parser.add_argument("--require-c", action="store_true",
                        help="treat an unavailable C campaign as failure (CI)")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if str(args.root) not in sys.path:
        sys.path.insert(0, str(args.root))
    report = run_campaign(
        args.root, args.mutations, args.seed, args.fuzz_runs, args.fuzz_seconds,
        require_c=args.require_c,
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    surfaces = ",".join(f"{item['surface']}:{item['state']}" for item in report["python"])
    print(f"decoder_campaign: {report['state']} python={surfaces} c={report['c_decoder']['state']}")
    if report.get("error"):
        print(f"  {report['error']}")
    c_decoder = report["c_decoder"]
    if c_decoder["state"] not in {"PASS", "UNAVAILABLE/SKIPPED"}:
        # The captured tails are the only retained diagnostics once the
        # temporary build directory is gone; print them on failure.
        for key in ("reason", "stdout_tail", "stderr_tail"):
            if c_decoder.get(key):
                print(f"  c_decoder.{key}: {c_decoder[key]}")
    return 0 if report["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
