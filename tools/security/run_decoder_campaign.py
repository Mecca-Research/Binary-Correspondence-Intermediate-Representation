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
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_PYTHON = ("streampack", "bcab", "bcirq8")
_BASE_GRACEFUL = (
    ValueError, KeyError, IndexError, struct.error, EOFError,
    UnicodeDecodeError, OverflowError, OSError, zlib.error,
)


def _graceful() -> tuple[type[BaseException], ...]:
    from bcir.abi.streampack_abi import AbiError
    extras: list[type[BaseException]] = [AbiError]
    try:
        from bcir.abi.artifact_bundle import BundleError
        extras.append(BundleError)
    except ImportError:
        pass
    return _BASE_GRACEFUL + tuple(extras)


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
           rng: random.Random, mutations: int) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    accepted = 0
    rejected = 0
    graceful = _graceful()
    try:
        decode(seed)
        accepted += 1
    except graceful:
        rejected += 1
        findings.append({"kind": "seed-rejected", "type": "graceful"})
    except Exception as exc:  # noqa: BLE001 - campaign records unexpected classes
        findings.append({"kind": "ungraceful-seed", "type": type(exc).__name__})
    for _ in range(mutations):
        try:
            decode(_mutate(seed, rng))
            accepted += 1
        except graceful:
            rejected += 1
        except Exception as exc:  # noqa: BLE001
            findings.append({"kind": "ungraceful", "type": type(exc).__name__})
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
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mut.bcirq8"
        path.write_bytes(data)
        return read_q8_decoder(path)


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


def run_c_campaign(root: Path, runs: int, seconds: int) -> dict[str, Any]:
    script = root / "tools" / "c" / "fuzz_streampack.sh"
    bash = shutil.which("bash")
    clang = shutil.which("clang")
    if os.name == "nt":
        return {
            "state": "UNAVAILABLE/SKIPPED",
            "reason": "native Windows host; C fuzzer is a POSIX rail",
        }
    if not bash or not script.is_file():
        return {"state": "UNAVAILABLE/SKIPPED", "reason": "bash or fuzz_streampack.sh missing"}
    if not clang:
        return {"state": "UNAVAILABLE/SKIPPED", "reason": "clang missing"}
    env = dict(os.environ)
    env.update({
        "FUZZ_RUNS": str(runs),
        "FUZZ_MAX_TOTAL_TIME": str(seconds),
        "FUZZ_JOBS": "2",
    })
    result = subprocess.run(
        [bash, str(script)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=max(60, seconds * 4),
    )
    skipped = "skipping" in (result.stdout + result.stderr).lower()
    if skipped and result.returncode == 0:
        return {
            "state": "UNAVAILABLE/SKIPPED",
            "reason": "clang has no libFuzzer/compiler-rt",
            "returncode": result.returncode,
        }
    return {
        "state": "PASS" if result.returncode == 0 else "FAIL",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-800:],
        "stderr_tail": result.stderr[-800:],
    }


def run_campaign(
    root: Path, mutations: int, seed: int, fuzz_runs: int, fuzz_seconds: int,
    require_c: bool = False,
) -> dict[str, Any]:
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
        report["state"] = "FAIL"
        report["error"] = (
            "C decoder campaign required but was "
            f"{report['c_decoder']['state']}: {report['c_decoder'].get('reason', '')}"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_decoder_campaign")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--mutations", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--fuzz-runs", type=int, default=200)
    parser.add_argument("--fuzz-seconds", type=int, default=8)
    parser.add_argument("--require-c", action="store_true")
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
    return 0 if report["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
