"""Record this host's libraries/source hashes and optional post-training controls."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

BEFORE = "379ee9b611864dc1b58094f3f5098686541e053d"
SOURCES = (
    "bcir/asn1/cgen.py", "bcir/gem/streampack.py", "bcir/gem/delta_chain.py",
    "bcir/gem/async_tokens.py", "bcir/gem/concurrency.py", "bcir/kbcir/realize.py",
    "bcir/abi/streampack_abi.py", "runtime/c/bcir_kplan.c", "runtime/c/bcir_runtime.c",
    "bcir/hosted/training/bpe.py", "runtime/c/bcir_tensor.c", "runtime/c/bcir_decoder_train.c",
)  # fmt: skip


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--controls", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    source = repo / "docs/research/measurements/2026-10-11/train_comparison.py"
    spec = importlib.util.spec_from_file_location("pr816_metadata_source", source)
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    sys.path.insert(0, str(repo))
    hashes = {}
    for name in SOURCES:
        old = subprocess.check_output(["git", "-C", str(repo), "show", f"{BEFORE}:{name}"])
        hashes[name] = {
            "before": hashlib.sha256(old).hexdigest(),
            "after": hashlib.sha256((repo / name).read_bytes()).hexdigest(),
        }
    write(args.root / "source-diff.json", hashes)
    environment = {
        **tc._environment(), **tc._libraries(), "gcc": tc._compiler("gcc"),
        "affinity": sorted(os.sched_getaffinity(0)),
        "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        "memory_limit": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
        "openblas_library_sha256": hashlib.sha256(tc._openblas()[0].read_bytes()).hexdigest(),
    }  # fmt: skip
    if args.controls:
        from bcir.hosted.models.native import NativeDecoder
        from bcir.tests import native_decoder_fixtures as nf

        library = args.root / "data/lib/gcc/bcir_decoder_train.so"
        lib = ctypes.CDLL(str(library.resolve()))
        swept, mismatch, ulp = nf.exp_sweep(lib)
        assert swept > 8_000_000 and mismatch == 0 and ulp <= 1
        environment["provider_mismatches_one_thread"] = nf.provider_mismatch(lib)
        resumed = []
        for task in ("lm", "copy"):
            directory = args.root / "full" / f"{task}-bcir"
            [record] = [json.loads(p.read_text()) for p in (directory / "runs").glob("*.json")]
            decoder = NativeDecoder.resume(
                library.resolve(), directory / "checkpoint.bin", threads=1, max_memory_bytes=4 << 30
            )
            state = tc._state_digest_native(decoder)
            assert state == record["state_sha256"] and decoder.state.step == 1000
            resumed.append(
                {
                    "task": task,
                    "state_sha256": state,
                    "step": decoder.state.step,
                    "contract": decoder.contract,
                    "resume_threads": 1,
                    "full_state_and_counters_sha256": nf.state_digest(decoder),
                }  # fmt: skip
            )
            decoder.close()
        (args.root / "verification").mkdir(exist_ok=True)
        write(
            args.root / "verification/native-controls.json",
            {
                "exp_sweep": {"inputs": swept, "mismatches": mismatch, "max_ulp": ulp},
                "resumed_checkpoints": resumed,
            },
        )
    write(args.root / "environment.json", environment)
    print(json.dumps({"source_files": len(hashes), "controls": args.controls}), flush=True)


if __name__ == "__main__":
    main()
