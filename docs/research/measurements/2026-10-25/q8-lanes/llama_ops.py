"""Q8-LANES' operation timer: a copy of TREE's runtime/c/bcir_llama.c with each operation of the monolithic
decoder's step timed (CLOCK_MONOTONIC around each call: embed, rmsnorm, matvec, rope, kv_append, attention,
residual, swiglu, head, argmax) and the totals printed at the generation's end, built with TREE's other units
(-std=gnu11 -O2 -ffp-contract=off) into OUT_DIR/llama-timed and run RUNS times: 200 new tokens from the 5-token
prompt of MODEL. Prints one JSON object: each operation's median microseconds over the runs and its calls, and the
operations' total.
usage: llama_ops.py TREE MODEL OUT_DIR CC RUNS"""

import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

tree, model, out_dir, cc, runs = (
    Path(sys.argv[1]),
    sys.argv[2],
    Path(sys.argv[3]),
    sys.argv[4],
    int(sys.argv[5]),
)
c = tree / "runtime" / "c"
src = (c / "bcir_llama.c").read_text()
names = [
    "embed",
    "rmsnorm",
    "matvec",
    "rope",
    "kv_append",
    "attention",
    "residual",
    "swiglu",
    "head",
    "argmax",
]
prelude = (
    "#include <time.h>\n#include <stdio.h>\n"
    "static double op_ns[16]; static long op_n[16];\n"
    "static double nowns(void){struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec;}\n"
    "#define TIMED(k, expr) ({ double _t0 = nowns(); __typeof__(expr) _r = (expr); op_ns[k] += nowns() - _t0; op_n[k]++; _r; })\n"
    "#define TIMEDV(k, stmt) do { double _t0 = nowns(); stmt; op_ns[k] += nowns() - _t0; op_n[k]++; } while (0)\n"
    "void llama_op_report(void); void llama_op_report(void){ static const char *nm[]={"
    + ",".join(f'"{n}"' for n in names)
    + '}; printf("{"); for(int i=0;i<10;i++) printf("%s\\"%s\\":[%.1f,%ld]", i?",":"", nm[i], op_ns[i]/1e3, op_n[i]); printf("}\\n");}\n'
)
src = src.replace("#include <string.h>\n", "#include <string.h>\n" + prelude, 1)
start, end = (
    src.index("static int step_token("),
    src.index("int bcir_llama_generate_greedy_with_allocator("),
)
body = src[start:end]
body = body.replace(
    "if (bcir_llama_op_embed(m, w, token)) return -1;",
    "if (TIMED(0, bcir_llama_op_embed(m, w, token))) return -1;",
)
body = re.sub(
    r"bcir_llama_op_rmsnorm\((.*?)\);", r"TIMEDV(1, bcir_llama_op_rmsnorm(\1));", body, flags=re.S
)
body = re.sub(
    r"matvec\(m, (.*?)\)(\s*\|\||\))",
    lambda mm: f"TIMED(2, matvec(m, {mm.group(1)})){mm.group(2)}",
    body,
    flags=re.S,
)
for old, new in (
    ("bcir_llama_op_rope(m, w, pos);", "TIMEDV(3, bcir_llama_op_rope(m, w, pos));"),
    (
        "bcir_llama_op_kv_append(m, w, layer, pos);",
        "TIMEDV(4, bcir_llama_op_kv_append(m, w, layer, pos));",
    ),
    (
        "if (bcir_llama_op_attention(m, w, layer, pos)) return -1;",
        "if (TIMED(5, bcir_llama_op_attention(m, w, layer, pos))) return -1;",
    ),
    ("bcir_llama_op_residual(m, w);", "TIMEDV(6, bcir_llama_op_residual(m, w));"),
    ("bcir_llama_op_swiglu(m, w);", "TIMEDV(7, bcir_llama_op_swiglu(m, w));"),
    (
        "if (!best_out || bcir_llama_op_head(m, row, logits)) return -1;",
        "if (!best_out || TIMED(8, bcir_llama_op_head(m, row, logits))) return -1;",
    ),
    (
        "*best_out = (int)bcir_llama_op_argmax(m, logits);",
        "*best_out = (int)TIMED(9, bcir_llama_op_argmax(m, logits));",
    ),
):
    assert old in body, old
    body = body.replace(old, new)
src = src[:start] + body + src[end:]
src = src.replace(
    "  bcir_llama_ws_free(&w);\n  return 0;\n}",
    "  bcir_llama_ws_free(&w);\n  llama_op_report();\n  return 0;\n}",
    1,
)
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "llama_timed.c").write_text(src)
exe = out_dir / "llama-timed"
subprocess.run([cc, "-std=gnu11", "-O2", "-ffp-contract=off", "-I", str(c), str(c / "bcir_ai_kernels.c"),
                str(c / "bcir_q8_model.c"), str(c / "bcir_decode.c"), str(out_dir / "llama_timed.c"),
                str(c / "bcir_llama_cli.c"), "-o", str(exe), "-lm"], check=True)  # fmt: skip
samples = []
for _ in range(runs):
    done = subprocess.run([str(exe), "--model", model, "--prompt-ids", "1,100,200,300,400", "--max-new", "200",
                           "--logits-out", str(out_dir / "logits.bin")], check=True, capture_output=True, text=True)  # fmt: skip
    samples.append(json.loads(done.stdout.strip().splitlines()[0]))
ops = {
    k: {"us": round(statistics.median(s[k][0] for s in samples), 1), "calls": samples[0][k][1]}
    for k in names
}
ops["total"] = {"us": round(sum(v["us"] for v in ops.values()), 1)}
print(json.dumps({"runs": runs, "operations": ops}))
