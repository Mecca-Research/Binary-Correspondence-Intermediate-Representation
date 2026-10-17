#!/usr/bin/env python3
"""NDT-PAGES' step record: two native trainers in one process -- the comparison's 9.84M decoder,
its weights and its first batch -- one with the state on ctypes arrays (`huge_pages=False`, 4 KB
pages), one on the mapping advised for 2 MB transparent huge pages (the default), stepped in
alternation; their states are held bit-identical after every pair.

    python3 pages_ab.py --library LIB --data DATA --threads 4 --pairs 20 --out OUT.json

`LIB` is the trainer `NativeDecoder.build` makes (CC=gcc for the records), `DATA` the directory
`train_comparison.py prepare` writes. Each step is timed alone after a 30 ms rest, so the other
trainer's pool is asleep, and the order alternates pair by pair.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
HARNESS = ROOT / "docs" / "research" / "measurements" / "2026-10-11" / "train_comparison.py"
THP = Path("/sys/kernel/mm/transparent_hugepage/enabled")


def _huge_kb(trainer) -> int:
    """AnonHugePages, in kB, of the mappings holding the trainer's state (/proc/self/smaps)."""
    spans = [(ct.addressof(b), ct.addressof(b) + ct.sizeof(b)) for b in trainer.buffers.values()]
    lo, hi = min(s for s, _ in spans), max(e for _, e in spans)
    total, inside = 0, False
    for line in Path("/proc/self/smaps").read_text().splitlines():
        head = line.split()
        if not head:
            continue
        if not head[0].endswith(":"):
            start, end = (int(x, 16) for x in head[0].split("-"))
            inside = start < hi and lo < end
        elif inside and head[0] == "AnonHugePages:":
            total += int(head[1])
    return total


def _digest(trainer) -> str:
    digest = hashlib.sha256()
    for name in ("weights", "moment1", "moment2"):
        digest.update(ct.string_at(trainer.buffers[name], trainer.state.plan.parameters * 4))
    return digest.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--pairs", type=int, default=20)
    p.add_argument("--contract", default="kblock+exp+reduce")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("train_comparison", HARNESS)
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    from bcir.hosted.models.native import NativeAdamW, NativeDecoder, parameter_layout

    flat = tc._load(a.data, "weights.f32", "f")
    layout = parameter_layout(tc._spec())

    def trainer(huge_pages):
        nd = NativeDecoder(a.library, tc._spec(), batch_size=tc.BATCH, context_length=tc.CONTEXT,
                           max_memory_bytes=4 << 30, threads=a.threads, contract=a.contract,
                           huge_pages=huge_pages)  # fmt: skip
        nd.load_state_dict({name: flat[off : off + n] for name, _s, off, n in layout})
        return nd

    trainers = {"4k": trainer(False), "2m": trainer(True)}
    stream = tc._load(a.data, "lm-train.u16", "H")
    offsets = tc._load(a.data, "lm-schedule.u32", "I")
    o = tc.OPT
    opt = NativeAdamW(lr=o["peak"] * 0.25, beta1=o["beta1"], beta2=o["beta2"],
                      epsilon=o["eps"], weight_decay=o["decay"], grad_clip=o["clip"])  # fmt: skip
    xs, ys = tc._batch(stream, offsets, 0, "lm")
    for nd in trainers.values():
        for _ in range(2):
            nd.run(xs, ys, steps=1, optimizer=opt)
    ms = {name: [] for name in trainers}
    identical = True
    for i in range(a.pairs):
        for name in ("4k", "2m") if i % 2 == 0 else ("2m", "4k"):
            time.sleep(0.03)
            t0 = time.perf_counter()
            trainers[name].run(xs, ys, steps=1, optimizer=opt)
            ms[name].append(1000 * (time.perf_counter() - t0))
        identical &= _digest(trainers["4k"]) == _digest(trainers["2m"])
    diff = [x - y for x, y in zip(ms["4k"], ms["2m"])]
    record = {
        "schema": "bcir.ndt_pages_ab.v1",
        "threads": a.threads,
        "contract": a.contract,
        "pairs": a.pairs,
        "python": platform.python_version(),
        "thp_enabled": THP.read_text().strip() if THP.is_file() else None,
        "advice_taken": {name: nd.huge_pages for name, nd in trainers.items()},
        "anon_huge_kb": {name: _huge_kb(nd) for name, nd in trainers.items()},
        "identical_after_every_pair": identical,
        "state_sha256": _digest(trainers["2m"]),
        "step_ms_median": {k: round(statistics.median(v), 1) for k, v in ms.items()},
        "step_ms_min": {k: round(min(v), 1) for k, v in ms.items()},
        "paired_diff_ms_median": round(statistics.median(diff), 1),
        "pairs_2m_faster": sum(d > 0 for d in diff),
        "step_ms": {k: [round(x, 1) for x in v] for k, v in ms.items()},
    }
    a.out.write_text(json.dumps(record, indent=1) + "\n")
    print({k: v for k, v in record.items() if k != "step_ms"})
    for nd in trainers.values():
        nd.close()
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
