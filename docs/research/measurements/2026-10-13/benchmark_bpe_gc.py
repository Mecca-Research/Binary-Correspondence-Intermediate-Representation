"""Measure the new BPE encoder's GC cost on the pinned large-document fixtures.

This is a control experiment, not a change to production GC policy. Run alone.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--bpe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from bcir.hosted.training.bpe import BytePairTokenizer

    original = json.loads((args.bpe / "bpe.json").read_text())
    tokenizer = BytePairTokenizer.from_json((args.bpe / "tokenizer.json").read_text())
    raw = b"\n\n".join(
        subprocess.check_output(
            ["git", "-C", str(args.repo), "show", f"{original['corpus_commit']}:{path}"]
        )
        for path in original["source_paths"]
    )
    rows = []
    for size in (65536, 262144):
        text = (raw * (size // len(raw) + 1))[:size].decode("utf-8", errors="ignore")
        [base] = [r for r in original["rows"] if r["actual_utf8_bytes"] == len(text.encode())]
        assert hashlib.sha256(text.encode()).hexdigest() == base["input_sha256"]
        expected = tokenizer.encode(text)
        timings = {"enabled": [], "paused": []}
        garbage, collections = {"enabled": [], "paused": []}, {"enabled": [], "paused": []}
        for rep in range(7):
            for mode in ("enabled", "paused") if rep % 2 else ("paused", "enabled"):
                gc.collect()
                before = [r["collections"] for r in gc.get_stats()]
                if mode == "paused":
                    gc.disable()
                try:
                    started = time.perf_counter_ns()
                    tokens = tokenizer.encode(text)
                    dt = (time.perf_counter_ns() - started) / 1e6
                finally:
                    gc.enable()
                after = [r["collections"] for r in gc.get_stats()]
                collections[mode].append([b - a for a, b in zip(before, after)])
                timings[mode].append(dt)
                assert tokens == expected
                garbage[mode].append(gc.collect())
        rows.append(
            {
                "utf8_bytes": len(text.encode()),
                "samples_ms": timings,
                "median_ms": {name: statistics.median(v) for name, v in timings.items()},
                "collections_by_generation": collections,
                "collected_after_encode": garbage,
                "output_identical": True,
            }
        )
    args.output.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
