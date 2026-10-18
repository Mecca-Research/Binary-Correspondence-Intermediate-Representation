"""Before/after BPE encoding with one pinned tokenizer and payload corpus."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path


def digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.after))
    from bcir.hosted.training.bpe import BytePairTokenizer

    old_path = args.before / "bcir/hosted/training/bpe.py"
    spec = importlib.util.spec_from_file_location("bcir.hosted.training.bpe_before", old_path)
    before = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = before
    spec.loader.exec_module(before)
    commit = "75d50d4f6e280e35a25032024edffd92f25c52f2"
    paths = subprocess.check_output(
        ["git", "-C", str(args.after), "ls-tree", "-r", "--name-only", commit], text=True
    ).splitlines()
    chunks = []
    for path in sorted(
        p for p in paths if p.startswith(("docs/", "training/")) and p.endswith(".md")
    )[:16]:
        chunks.append(
            subprocess.check_output(["git", "-C", str(args.after), "show", f"{commit}:{path}"])
        )
    raw = b"\n\n".join(chunks)
    text = raw.decode("utf-8")
    training_text = raw[:32768].decode("utf-8", errors="ignore")
    started = time.perf_counter()
    model = BytePairTokenizer.train([training_text], vocab_size=1024, min_frequency=2)
    training_seconds = time.perf_counter() - started
    old = before.BytePairTokenizer.from_json(model.to_json())
    assert old.merges == model.merges
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tokenizer.json").write_text(model.to_json())
    rows = []
    for size in (1024, 8192, 65536, 262144):
        repeated = raw * (size // len(raw) + 1)
        value = repeated[:size].decode("utf-8", errors="ignore")
        assert old.encode(value) == model.encode(value)
        timings = {"before": [], "after": []}
        for round_ in range(7):
            order = ("before", "after") if round_ % 2 else ("after", "before")
            for name in order:
                fn = old.encode if name == "before" else model.encode
                gc.collect()
                start = time.perf_counter_ns()
                tokens = fn(value)
                timings[name].append((time.perf_counter_ns() - start) / 1e6)
                assert tokens == model.encode(value)
        peaks = {}
        for name in timings:
            gc.collect()
            tracemalloc.start()
            tokens = (old if name == "before" else model).encode(value)
            peaks[name] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        rows.append(
            {
                "requested_bytes": size,
                "actual_utf8_bytes": len(value.encode()),
                "input_sha256": hashlib.sha256(value.encode()).hexdigest(),
                "tokens": len(tokens),
                "encoded_ids_sha256": digest(tokens),
                "samples_ms": timings,
                "median_ms": {name: statistics.median(v) for name, v in timings.items()},
                "peak_allocated_bytes": peaks,
            }
        )
        print(json.dumps({"bpe_bytes": size, "done": True}), flush=True)
    rng = random.Random(31)
    checks = 0
    for _ in range(100):
        value = "".join(rng.choice("abcd é\n\rα🤖") for _ in range(rng.randrange(1, 1000)))
        for bos, eos in ((False, False), (True, True)):
            assert old.encode(value, add_bos=bos, add_eos=eos) == model.encode(
                value, add_bos=bos, add_eos=eos
            )
            checks += 1
    duplicate_merges = ((101, 102, 260), (101, 102, 261))
    old_edge = before.BytePairTokenizer(duplicate_merges)
    new_edge = BytePairTokenizer.from_json(BytePairTokenizer(duplicate_merges).to_json())
    witness = {
        "model": json.loads(new_edge.to_json()),
        "text": "ab",
        "before_tokens": old_edge.encode("ab"),
        "after_tokens": new_edge.encode("ab"),
        "same": old_edge.encode("ab") == new_edge.encode("ab"),
    }
    result = {
        "corpus_commit": commit,
        "source_paths": paths[:0]
        + sorted(p for p in paths if p.startswith(("docs/", "training/")) and p.endswith(".md"))[
            :16
        ],
        "tokenizer_vocab": model.vocab_size,
        "merges": len(model.merges),
        "tokenizer_sha256": model.digest,
        "training_prefix_bytes": len(training_text.encode()),
        "tokenizer_training_seconds": training_seconds,
        "parity_cases": checks,
        "rows": rows,
        "duplicate_pair_witness": witness,
        "scope": "Identical trained tokenizer, pinned document text, seven alternating calls, normalization and token-list allocation included; tokenizer training excluded and unchanged. GC enabled inside encode, explicit collection before timer. Peak tracemalloc measured separately.",
    }
    (args.output / "bpe.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
