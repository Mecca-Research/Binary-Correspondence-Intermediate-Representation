"""ATT-BOUND: the GEO step's GEMM claims at one context -- each claim's calls (orientation, m x n x k) and FLOPs
from the trainer's own step program, its rate from a step_parts.py reading of the same step (../bwd-keep/).
usage: gemm_rates.py PARTS.json OUT.json [T] [CONTRACT]"""

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]  # the repository
sys.path.insert(0, str(ROOT))
from bcir.frontends.models.decode import DecoderSpec  # noqa: E402
from bcir.hosted.models import native_launch as nl  # noqa: E402
from bcir.hosted.models.native import NativeDecoder  # noqa: E402
from bcir.hosted.models.native_pack import GEMM_ORIENTS, op_name  # noqa: E402

PARTS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
T = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
CONTRACT = sys.argv[4] if len(sys.argv) > 4 else "kblock+exp+reduce"
parts = {p["claim"]: p for p in json.loads(PARTS.read_text())["parts"]}
spec = DecoderSpec(
    vocab_size=260,
    d_model=128,
    n_heads=4,
    n_layers=3,
    d_ff=256,
    activation="silu_gate",
    n_kv_heads=4,
    tied_embeddings=True,
    rms_norm_eps=1e-5,
)
with tempfile.TemporaryDirectory() as tmp:
    lib = NativeDecoder.build(Path(tmp) / "lib")
    tr = NativeDecoder(
        lib,
        spec,
        batch_size=4,
        context_length=T,
        threads=4,
        contract=CONTRACT,
        max_memory_bytes=4 << 30,
    )
    program = nl.step_program(spec, 4, T, **tr.program_decisions(steps=1))
flops, calls = defaultdict(int), defaultdict(list)
for launch in program:
    name = op_name(launch.records[0])
    if not name.startswith("train.gemm"):
        continue
    for r in launch.records:
        if r.kind == nl.GEMM:
            ta, tb, m, n, k = r.dims[:5]
            flops[name] += 2 * m * n * k
            calls[name].append(f"{GEMM_ORIENTS[ta, tb]} {m}x{n}x{k}")
rows = []
for name in sorted(flops, key=lambda s: -parts[s]["ms_median"]):
    ms = parts[name]["ms_median"]
    rows.append(
        {
            "claim": name,
            "ms_median": ms,
            "gflop": flops[name] / 1e9,
            "gflop_per_s": flops[name] / ms / 1e6,
            "calls": len(calls[name]),
            "shapes": sorted(set(calls[name])),
        }
    )
total_ms, total_f = sum(r["ms_median"] for r in rows), sum(flops.values())
doc = {
    "T": T,
    "contract": CONTRACT,
    "threads": 4,
    "parts": PARTS.name,
    "claims": rows,
    "all": {"ms": total_ms, "gflop": total_f / 1e9, "gflop_per_s": total_f / total_ms / 1e6},
}
OUT.write_text(json.dumps(doc, indent=1) + "\n")
for r in rows:
    print(
        f"{r['claim']:24s} {r['ms_median']:6.2f} ms {r['gflop']:6.2f} GFLOP {r['gflop_per_s']:6.1f} GFLOP/s  {r['calls']} calls"
    )
print(
    f"all GEMMs {total_ms:.1f} ms {total_f / 1e9:.2f} GFLOP {total_f / total_ms / 1e6:.1f} GFLOP/s"
)
