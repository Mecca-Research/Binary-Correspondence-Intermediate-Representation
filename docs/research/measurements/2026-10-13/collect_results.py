"""Collect the PR816 observations without discarding raw per-call timings."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    structures = {}
    for path in sorted((root / "structures").glob("r*/*/*/*.json")):
        if path.stem == path.parent.name:
            structures[str(path.relative_to(root / "structures"))] = read(path)
    assert len(structures) == 42
    progress = read(root / "timing/progress.json")
    assert len(progress) == 30
    screen = root / "eight-thread-screen/progress.json"
    if screen.exists():
        extra = read(screen)
        assert len(extra) == 9
        progress += extra
    runs = {
        f"r{row['repetition']}/{row['name']}": read(Path(row["record"]))
        for row in progress
        if row["exit_code"] == 0
    }
    matrix = {}
    for row in progress:
        entry = matrix.setdefault(row["name"], {"medians_ms": [], "failures": []})
        if row["exit_code"] == 0:
            entry["medians_ms"].append(row["median_step_ms"])
        else:
            entry["failures"].append(row["failure_tail"])
    for entry in matrix.values():
        values = entry["medians_ms"]
        if values:
            entry.update(
                median_ms=statistics.median(values),
                min_ms=min(values),
                max_ms=max(values),
                tokens_per_second=1024 / (statistics.median(values) / 1000),
            )
    # The default arithmetic state is the same before/after and at every pool width.
    v1_names = ("before-native-v1-t1", "after-native-v1-t1", "after-native-v1-t4")
    v1_states = {
        record["state_sha256"] for key, record in runs.items() if key.split("/")[1] in v1_names
    }
    assert len(v1_states) == 1
    kblock_states = {record["state_sha256"] for key, record in runs.items() if "kblock-exp" in key}
    assert len(kblock_states) == 1
    assert v1_states != kblock_states
    full = {
        str(path.relative_to(root / "full")): read(path)
        for path in sorted((root / "full").glob("*/runs/*.json"))
    }
    assert len(full) == 4 and all(record["steps"] == 1000 for record in full.values())
    quality = {}
    for task in ("lm", "copy"):
        pair = {record["backend"]: record for record in full.values() if record["task"] == task}
        left, right = pair["bcir"], pair["torch-compiled"]
        assert left["requested_compute_threads"] == right["actual_torch_threads"]
        quality[task] = {}
        for split in left["final"]:
            a, b = left["final"][split], right["final"][split]
            delta = abs(a["cross_entropy"] - b["cross_entropy"])
            quality[task][split] = {
                "cross_entropy_abs_difference": delta,
                "within_0_001_nats": delta <= 0.001,
                "accuracy_abs_difference": abs(a["accuracy"] - b["accuracy"]),
                "answer_accuracy_abs_difference": abs(a["answer_accuracy"] - b["answer_accuracy"]),
                "exact_match_abs_difference": abs(a["exact_match"] - b["exact_match"]),
            }
    identity = read(root / "data/identity.json")
    assert identity["identical"]
    result = {
        "before_commit": "379ee9b611864dc1b58094f3f5098686541e053d",
        "after_commit": "be6e377a2eb417cb4bfe3de5243ca09961ba1833",
        "source_sha256": read(root / "source-diff.json"),
        "environment": read(root / "environment.json"),
        "preparation": read(root / "data/prepare.json"),
        "structures": structures,
        "bpe": read(root / "bpe/bpe.json"),
        "bpe_gc_control": read(root / "bpe/gc-control.json"),
        "timing_progress": progress,
        "timing_runs": runs,
        "timing_summary": matrix,
        "default_state_identical_before_after_threads": True,
        "blocked_state_identical_threads": True,
        "provider_training_state_checked": any("openblas" in key for key in runs),
        "identity": identity,
        "full_training": full,
        "quality_progress": read(root / "quality-progress.json"),
        "quality_comparison": quality,
        "native_controls": read(root / "verification/native-controls.json"),
        "breakdowns": {
            path.name: read(path) for path in sorted((root / "breakdowns").glob("*.json"))
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":"), sort_keys=True) + "\n")
    print(json.dumps(matrix, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
