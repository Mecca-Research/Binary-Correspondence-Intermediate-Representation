#!/usr/bin/env python3
"""In-place attribution of one training step of the native trainer (NDT-SCAN's instrument).

The trainer's own breakdown (`train_comparison.py bcir-breakdown`, 2026-10-11) times each kernel
family alone on synthetic operands and reports the rest as a difference. This times every
statement of the step where it runs: a copy of the two C units (`--src`) with a monotonic clock
read after every statement of `forward()`, `bcir_decoder_loss_backward()` and
`bcir_decoder_update()`, each statement charged to a category by what it calls; the copy is built
with the trainer's flags and run on the comparison's 9.84M decoder, weights and first batch.
Timers are read by the caller's thread only (never inside a pool task), about 30 ns each.

    python3 attribution.py --src runtime/c --data DATA --threads 4 --steps 5 --out OUT.json

`--src` is any tree's `runtime/c` (a worktree of another commit for a before/after pair). The
record is the per-category milliseconds per step, the step's own wall time beside it.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import importlib.util
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
HARNESS = ROOT / "docs" / "research" / "measurements" / "2026-10-11" / "train_comparison.py"
FLAGS = ["-std=c11", "-O3", "-ffp-contract=off", "-fno-math-errno", "-Wall", "-Wextra",
         "-Wpedantic", "-Werror", "-DBCIR_DT_BUILD_SHARED", "-shared", "-fPIC", "-pthread"]  # fmt: skip

PROF = r"""
#include <time.h>
#define BCIR_PROF_MAX 64
static double bcir_prof_ms[BCIR_PROF_MAX]; static const char *bcir_prof_names[BCIR_PROF_MAX];
static unsigned bcir_prof_count; static struct timespec bcir_prof_t;
BCIR_DT_API void bcir_prof_mark(void) { clock_gettime(CLOCK_MONOTONIC,&bcir_prof_t); }
BCIR_DT_API void bcir_prof_add(const char *name) {
  struct timespec t; unsigned k;
  clock_gettime(CLOCK_MONOTONIC,&t);
  for (k=0;k<bcir_prof_count && strcmp(bcir_prof_names[k],name);k++) {}
  if (k==bcir_prof_count && k<BCIR_PROF_MAX) { bcir_prof_names[k]=name; bcir_prof_count++; }
  if (k<BCIR_PROF_MAX)
    bcir_prof_ms[k]+=(double)(t.tv_sec-bcir_prof_t.tv_sec)*1e3+(double)(t.tv_nsec-bcir_prof_t.tv_nsec)*1e-6;
  bcir_prof_t=t;
}
BCIR_DT_API unsigned bcir_prof_dump(double *ms,const char **names,unsigned cap) {
  unsigned k; for (k=0;k<bcir_prof_count && k<cap;k++) { ms[k]=bcir_prof_ms[k]; names[k]=bcir_prof_names[k]; }
  return bcir_prof_count;
}
BCIR_DT_API void bcir_prof_reset(void) { unsigned k; for (k=0;k<BCIR_PROF_MAX;k++) bcir_prof_ms[k]=0.0; }
"""

#: A statement's category: the first pattern its text matches (the backward's rope before the
#: forward's, which shares its call).
CATEGORIES = [
    (r"finite_(span|pool)\(s,?\s*s->weights|finite_span\(s->weights", "finite.weights"),
    (r"finite_(span|pool)\(s,?\s*s->activations|finite_span\(s->activations", "finite.activations"),
    (r"finite_(span|pool)\(s,?\s*s->gradients|finite_span\(s->gradients", "finite.gradients"),
    (r"memset\(s->gradients|zero_pool\(s,s->gradients", "zero.gradients"),
    (r"memcpy\(xs\+i\*g\.d", "embed.gather"),
    (r"loss_task", "loss.rows"),
    (r"ce\+=term", "loss.sum"),
    (r"linears?_backward\(", "gemm.backward"),
    (r"rms_backward\(", "rms.backward"),
    (r"swiglu_backward\(", "swiglu.backward"),
    (r"attention_backward", "attention.backward"),
    (r"bcir_tensor_rope_pool\(s->pool,g\.b,g\.t,g\.k?h,g\.hd,s->plan", "rope.backward"),
    (r"bcir_tensor_rope_pool", "rope.forward"),
    (r"scatter_task", "embed.scatter"),
    (r"bcir_tensor_rms_pool", "rms.forward"),
    (r"\blinears?\(", "gemm.forward"),
    (r"\battention\(s", "attention.forward"),
    (r"bcir_tensor_add_pool", "residual.add"),
    (r"\bsilu\(s", "silu"),
    (r"bcir_tensor_mul_pool", "swiglu.mul"),
    (r"first_pass\(", "update.first_pass"),
    (r"preflight_task", "update.preflight"),
    (r"adamw_commit_pool", "update.commit"),
    (r"\bforward\(s,tokens,g\)", "forward.call"),
]
DECLARATION = re.compile(
    r"^(geometry|size_t|double|float|int|loss_job|scatter_job|layer|cache|bcir_decoder_adamw|"
    r"pass_part|pass_result|const|adamw_terms|preflight_job|unsigned)\b"
)
FUNCTIONS = (
    "static void forward(bcir_decoder_state *s,const uint32_t *tokens,geometry g) {",
    "int bcir_decoder_loss_backward(bcir_decoder_state *s,const uint32_t *tokens,",
    "int bcir_decoder_update(bcir_decoder_state *s,const bcir_decoder_adamw *o,",
)


def category(statement: str) -> str:
    for pattern, name in CATEGORIES:
        if re.search(pattern, statement):
            return name
    return "other"


def instrument(text: str, header: str) -> str:
    """A timer mark before the body's first statement and a charge after every statement."""
    start = text.index(header)
    body_open = text.index("{", start + len(header) - 1)
    depth, i = 0, body_open
    while True:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            break
        i += 1
    lines, pending, marked = [], "", False
    for line in text[body_open + 1 : i].split("\n"):
        lines.append(line)
        stripped = line.strip()
        if not stripped or stripped.startswith(("/*", "*", "//")):
            continue
        pending += " " + stripped
        if stripped.endswith(";") and not stripped.startswith("return"):
            if DECLARATION.match(stripped) and not marked:
                pending = ""
                continue
            if not marked:
                lines.insert(len(lines) - 1, "  bcir_prof_mark();")
                marked = True
            lines.append('  bcir_prof_add("%s");' % category(pending))
            pending = ""
        elif stripped.endswith("}"):
            pending = ""
    return text[: body_open + 1] + "\n".join(lines) + text[i:]


#: The attention's zero fills where a tree still runs them on the caller before the pool (before
#: NDT-SCAN): each gets its own row; a tree that fills in the tasks has neither anchor.
FILLS = (
    (
        "  memset(y,0,b*t*nh*d*sizeof(float)); memset(p,0,b*nh*t*t*sizeof(float));\n",
        "attention.forward.zero",
    ),
    (
        "  memset(dv,0,b*t*nk*d*sizeof(float));\n"
        "  a.b=b; a.t=t; a.nh=nh; a.nk=nk; a.d=d; a.q=q; a.k=k; a.v=v; a.dy=dy; a.p=p; a.y=NULL;\n",
        "attention.backward.zero",
    ),
)


def instrumented_copy(src: Path, out: Path) -> None:
    for name in ("bcir_tensor.h", "bcir_decoder_train.h"):
        (out / name).write_text((src / name).read_text(encoding="utf-8"), encoding="utf-8")
    tensor = (src / "bcir_tensor.c").read_text(encoding="utf-8")
    for anchor, name in FILLS:
        if tensor.count(anchor) == 1:
            timer = '  { extern void bcir_prof_add(const char *); bcir_prof_add("%s"); }\n' % name
            tensor = tensor.replace(anchor, anchor + timer)
    (out / "bcir_tensor.c").write_text(tensor, encoding="utf-8")
    train = (src / "bcir_decoder_train.c").read_text(encoding="utf-8")
    train = "#define _POSIX_C_SOURCE 199309L\n" + train
    anchor = "#include <float.h>\n"
    if train.count(anchor) != 1:
        raise SystemExit("attribution: the trainer's includes moved; update the instrument")
    train = train.replace(anchor, anchor + PROF, 1)
    for header in FUNCTIONS:
        if train.count(header) != 1:
            raise SystemExit(f"attribution: {header!r} not found once; update the instrument")
        train = instrument(train, header)
    (out / "bcir_decoder_train.c").write_text(train, encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", type=Path, required=True, help="a tree's runtime/c")
    p.add_argument("--data", type=Path, required=True, help="the comparison's data directory")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--contract", default="kblock+exp+reduce")
    p.add_argument("--cc", default="gcc")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("train_comparison", HARNESS)
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    from bcir.hosted.models.native import NativeAdamW

    with tempfile.TemporaryDirectory(prefix="bcir-attribution-") as tmp:
        work = Path(tmp)
        instrumented_copy(a.src, work)
        lib = work / "bcir_decoder_train.so"
        units = [str(work / "bcir_tensor.c"), str(work / "bcir_decoder_train.c")]
        subprocess.run([a.cc, *FLAGS, "-I", str(work), *units, "-o", str(lib), "-lm"], check=True)
        nd, _info = tc._native(a.data, lib, threads=a.threads, contract=a.contract)
        nd.lib.bcir_prof_dump.argtypes = [
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_char_p),
            ct.c_uint,
        ]
        nd.lib.bcir_prof_dump.restype = ct.c_uint
        stream = tc._load(a.data, "lm-train.u16", "H")
        offsets = tc._load(a.data, "lm-schedule.u32", "I")
        o = tc.OPT
        opt = NativeAdamW(lr=o["peak"] * 0.25, beta1=o["beta1"], beta2=o["beta2"],
                          epsilon=o["eps"], weight_decay=o["decay"], grad_clip=o["clip"])  # fmt: skip
        xs, ys = tc._batch(stream, offsets, 0, "lm")
        for _ in range(2):
            nd.run(xs, ys, steps=1, optimizer=opt)
        nd.lib.bcir_prof_reset()
        walls = []
        for _ in range(a.steps):
            t0 = time.perf_counter()
            nd.run(xs, ys, steps=1, optimizer=opt)
            walls.append(1000 * (time.perf_counter() - t0))
        ms, names = (ct.c_double * 64)(), (ct.c_char_p * 64)()
        count = nd.lib.bcir_prof_dump(ms, names, 64)
        rows = {names[k].decode(): ms[k] / a.steps for k in range(count)}
    record = {
        "schema": "bcir.ndt_attribution.v1",
        "contract": a.contract,
        "threads": a.threads,
        "steps": a.steps,
        "compiler": a.cc,
        "step_wall_ms_median": statistics.median(walls),
        "step_wall_ms": walls,
        "timed_ms": sum(rows.values()),
        "rows_ms": dict(sorted(rows.items(), key=lambda kv: -kv[1])),
    }
    a.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"step {record['step_wall_ms_median']:.1f} ms, timed {record['timed_ms']:.1f} ms")
    for name, v in record["rows_ms"].items():
        print(f"  {name:24s} {v:8.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
