"""GP-LANES' faults: each injected into a copy of the overlay (its kernels or its tasks), the overlay applied to a copy
of a work tree once_level3.py edited, and both witnesses run -- gp_lanes_check.c (the kernels against the fork's,
byte for byte) and gp_tasks_check.c (the forks' entries, the build before against the faulty build). A fault is
caught where a witness reports a mismatched output. The equivalent mutations beside them change no output on any
input, and are recorded with the reason: they show what the witnesses cannot see by construction.
usage: red_lanes.py TOOLS ONCE_WORK GP_SCALAR_INC TASKS_CHECK_BIN OUT_DIR OUT.json"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS, ONCE, SCALAR, TASKS, OUT, REPORT = (Path(a).resolve() for a in sys.argv[1:7])
FAULTS = {
    "LN1": (
        "the forward normalises in float (the scalar kernel rounds o*inv from double)",
        "gp_lanes_pair.inc",
        "for(c=0;c<GP_P;c++)ST(O[c],gpd_float(gpd_mul(gpd_of(LD(O[c])),inv)));",
        "{__m512 fi=gpd_float(inv); for(c=0;c<GP_P;c++)ST(O[c],_mm512_mul_ps(LD(O[c]),fi));}",
    ),
    "LN2": (
        "a row the scalar kernel sends to its FP64 fallback kept as the lanes computed it",
        "gp_lanes_overlay.py",
        "  for(l=0;l<GPL;l++) if(ill[l]) gp_forward_task((void *)j,r0+l);\n",
        "  (void)ill;\n",
    ),
    "LN3": (
        "the backward's rows past the last group of sixteen start one row late",
        "gp_lanes_overlay.py",
        "  if(r0+GPL<=n) gpl_backward_group(j,r0); else for(r=r0;r<n;r++) gp_backward_task(ctx,r);",
        "  if(r0+GPL<=n) gpl_backward_group(j,r0); else for(r=r0+1;r<n;r++) gp_backward_task(ctx,r);",
    ),
    "LN4": (
        "the query contraction's output a rounded float sum with a term the scalar kernel lacks",
        "gp_lanes_query.inc",
        "for(i=0;i<GP_RANK;i++)ST(C[i],gpd_float(acc[i]));",
        "for(i=0;i<GP_RANK;i++)ST(C[i],_mm512_add_ps(gpd_float(acc[i]),_mm512_set1_ps(1e-7f)));",
    ),
    "LN5": (
        "one adjoint's first term scaled by 1 + 2^-23 after it lands",
        "gp_lanes_pair.inc",
        "ST(DV[i],_mm512_sub_ps(LD(DV[i]),_mm512_mul_ps(q,LD(U[k]))));",
        "ST(DV[i],_mm512_sub_ps(LD(DV[i]),_mm512_mul_ps(q,LD(U[k]))));"
        "if(c==0)ST(DV[i],_mm512_mul_ps(LD(DV[i]),_mm512_set1_ps(1.0000001f)));",
    ),
}
EQUIVALENT = {
    "LE1": (
        "the norm's square and sum fused: od*od of a widened float is exact in double (48 significant bits), "
        "so one rounding of od*od+norm is the two roundings' value",
        "gp_lanes_pair.inc",
        "ST(O[c],o); od=gpd_of(o); norm=gpd_add(norm,gpd_mul(od,od));",
        "ST(O[c],o); od=gpd_of(o); norm.lo=_mm512_fmadd_pd(od.lo,od.lo,norm.lo); norm.hi=_mm512_fmadd_pd(od.hi,od.hi,norm.hi);",
    ),
    "LE2": (
        "the query pair's two updates swapped: acc[k] and acc[i] are distinct accumulators (i < k)",
        "gp_lanes_query.inc",
        "acc[k]=gpd_add(acc[k],gpd_mul(qi,pa));acc[i]=gpd_sub(acc[i],gpd_mul(gpd_of(LD(Q[k])),pa));",
        "acc[i]=gpd_sub(acc[i],gpd_mul(gpd_of(LD(Q[k])),pa));acc[k]=gpd_add(acc[k],gpd_mul(qi,pa));",
    ),
    "LE3": (
        "the vjp's dot accumulated on every lane: a lane whose norm is at most the epsilon never reads it",
        "gp_lanes_query.inc",
        "    dot=gpd_pick(big,gpd_add(dot,t),dot);\n",
        "    dot=gpd_add(dot,t);\n",
    ),
}
CC = ["gcc", "-std=c11", "-O3", "-ffp-contract=off", "-fno-math-errno", "-Wno-unused-function"]


def run(name, desc, target, old, new):
    d = OUT / name
    shutil.rmtree(d, ignore_errors=True)
    (d / "tools").mkdir(parents=True)
    for f in ("gp_lanes_overlay.py", "gp_lanes_pair.inc", "gp_lanes_query.inc", "gp_lanes_check.c"):
        shutil.copyfile(TOOLS / f, d / "tools" / f)
    shutil.copyfile(SCALAR, d / "tools/gp_scalar.inc")
    text = (d / "tools" / target).read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"{name}: anchor found {text.count(old)} times")
    (d / "tools" / target).write_text(text.replace(old, new), encoding="utf-8")
    shutil.copytree(ONCE, d / "work", symlinks=True)
    r = subprocess.run(
        [
            sys.executable,
            "-I",
            str(d / "tools/gp_lanes_overlay.py"),
            str(d / "work"),
            "guard_compact",
            "query_guard_compact",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode:
        raise SystemExit(f"{name}: overlay failed\n{r.stderr[-800:]}")
    r = subprocess.run(
        CC
        + [
            "-I",
            str(d / "tools"),
            str(d / "tools/gp_lanes_check.c"),
            "-o",
            str(d / "check"),
            "-lm",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode:
        raise SystemExit(f"{name}: check build failed\n{r.stderr[-800:]}")
    k = subprocess.run([str(d / "check"), "170", "1"], capture_output=True, text=True)
    kernels = json.loads(k.stdout)["mismatched_outputs"] if k.stdout.startswith("{") else None
    tasks = {}
    for lib, q in (("guard_compact", "0"), ("query_guard_compact", "1")):
        t = subprocess.run(
            [str(TASKS), str(ONCE / "build" / f"{lib}.so"), str(d / "work/build" / f"{lib}.so"), q],
            capture_output=True,
            text=True,
        )
        tasks[lib] = (
            json.loads(t.stdout)["mismatched_outputs"] if t.stdout.startswith("{") else None
        )
    shutil.rmtree(d / "work")
    return {
        "what": desc,
        "in": target,
        "kernel_check_mismatches": kernels,
        "task_check_mismatches": tasks,
    }


faults = {n: run(n, *spec) for n, spec in FAULTS.items()}
equivalent = {n: run(n, *spec) for n, spec in EQUIVALENT.items()}
for n, f in faults.items():
    f["caught"] = bool(f["kernel_check_mismatches"]) or any(
        v for v in f["task_check_mismatches"].values()
    )
for n, f in equivalent.items():
    f["observable"] = bool(f["kernel_check_mismatches"]) or any(
        v for v in f["task_check_mismatches"].values()
    )
REPORT.write_text(
    json.dumps({"faults": faults, "equivalent": equivalent}, indent=1) + "\n", encoding="utf-8"
)
print(
    f"faults caught {sum(f['caught'] for f in faults.values())}/{len(faults)}; "
    f"equivalent mutations observable {sum(f['observable'] for f in equivalent.values())}/{len(equivalent)}"
)
sys.exit(0 if all(f["caught"] for f in faults.values()) else 1)
