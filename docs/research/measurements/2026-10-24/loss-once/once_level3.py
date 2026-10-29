#!/usr/bin/env python3
"""LOSS-ONCE on the GEO study's models: the study's three native libraries are forks of BCIR's trainer, and each fork's
loss task calls libm's exp twice a logit -- for the row's sum and again for its adjoint -- as BCIR's trainer did before
LOSS-ONCE. This makes the same edit to each fork named (attention, guard_compact, query_guard_compact): a row keeps
its first 1,024 exponentials in double on the task's stack and its adjoints divide the kept values, a wider row
evaluating its tail again, the fork's ignored-target rows (UINT32_MAX) as before. In the geometric heads
(guard_compact, query_guard_compact) it also gives the gate's forward its exponential a task's 512 elements at a
time through the tensor unit's span entry, bcir_tensor_exp (the widest instance, exp1's bits on every input), where
the fork calls bcir_tensor_expf an element. Every value is the fork's: the same exponential of the same argument, and
the operations after it in their order. Each library is rebuilt by its build record's own commands, the tensor
unit's object as the record left it.

usage: python3 -I once_level3.py OUT_WORK LIB [LIB ...]
  OUT_WORK a work tree as build_amx.py (level 2: <lib>-build-amx.json) or bands_level3.py (the attention model at
  level 3: attention-build-level3.json) left it, or a copy of one: the recorded commands are pointed at OUT_WORK's
  build/ before they run, so a copy never rebuilds its original. Rewrites OUT_WORK/build/<lib>.c, .o and .so, and
  writes OUT_WORK/build/<lib>-build-once.json.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(sys.argv[1]).resolve()
LIBS = sys.argv[2:]
build = OUT / "build"
LOSS = (
    """    for (j=1;j<v;j++) if (row[j]>maxv) maxv=row[j];
    for (j=0;j<v;j++) sum+=exp((double)row[j]-maxv);
    term=log(sum)+(double)maxv-row[jb->targets[i]];
    for (j=0;j<v;j++) d[j]=(float)((exp((double)row[j]-maxv)/sum-
        (j==jb->targets[i] ? 1.0 : 0.0))/(double)jb->denom);
""",
    """    for (j=1;j<v;j++) if (row[j]>maxv) maxv=row[j];
    {  /* LOSS-ONCE: each exponential once, the adjoints from the kept values (once_level3.py) */
      size_t kept=v<1024 ? v : 1024,t=jb->targets[i]; double e[1024];
      for (j=0;j<kept;j++) { e[j]=exp((double)row[j]-maxv); sum+=e[j]; }
      for (;j<v;j++) sum+=exp((double)row[j]-maxv);
      term=log(sum)+(double)maxv-row[t];
      for (j=0;j<kept;j++) d[j]=(float)((e[j]/sum-(j==t ? 1.0 : 0.0))/(double)jb->denom);
      for (;j<v;j++) d[j]=(float)((exp((double)row[j]-maxv)/sum-(j==t ? 1.0 : 0.0))/(double)jb->denom);
    }
""",
)
GATE = (
    """static void wedge_gate_fwd_task(void *ctx,size_t task) {
  wedge_gate_job *j=ctx;size_t i,end=(task+1)*512;if(end>j->n)end=j->n;
  for(i=task*512;i<end;i++) {
    float e=bcir_tensor_expf(-(j->a[i]+j->x[i]));
    j->a[i]=1.0f/(1.0f+e);j->x[i]=j->a[i]*j->h[i]+(1.0f-j->a[i])*j->g[i];
  }
}""",
    """static void wedge_gate_fwd_task(void *ctx,size_t task) {
  /* GATE-SPAN: the task's 512 exponentials through bcir_tensor_exp, exp1's bits (once_level3.py) */
  wedge_gate_job *j=ctx;size_t i,i0=task*512,end=(task+1)*512;float z[512],e[512];if(end>j->n)end=j->n;
  for(i=i0;i<end;i++) z[i-i0]=-(j->a[i]+j->x[i]);
  bcir_tensor_exp(end-i0,z,e);
  for(i=i0;i<end;i++) {
    j->a[i]=1.0f/(1.0f+e[i-i0]);j->x[i]=j->a[i]*j->h[i]+(1.0f-j->a[i])*j->g[i];
  }
}""",
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def record_of(lib):
    """The library's build record and the commands that build its overlay and shared object (after the tensor
    unit's object), pointed at OUT's build/."""
    for name in (f"{lib}-build-level3.json", f"{lib}-build-amx.json"):
        p = build / name
        if p.is_file():
            rec = json.loads(p.read_text(encoding="utf-8"))
            cmds = [c for c in rec["commands"] if not any(x.endswith("bcir_tensor.c") for x in c)]
            link = cmds[-1]
            made = Path(link[link.index("-o") + 1])
            if made.name != f"{lib}.so":
                raise SystemExit(f"{name}: the last command makes {made.name}, not {lib}.so")
            there = str(made.parent)
            cmds = [[x.replace(there, str(build)) for x in c] for c in cmds]
            if any(str(build) not in " ".join(c) for c in cmds):
                raise SystemExit(f"{name}: a command does not name {build}")
            return name, cmds
    raise SystemExit(f"{lib}: no level-2 or level-3 build record in {build}")


for lib in LIBS:
    source_path = build / f"{lib}.c"
    source = source_path.read_text(encoding="utf-8")
    before = sha(source_path)
    edits = [LOSS]
    if lib in ("guard_compact", "query_guard_compact"):
        edits.append(GATE)
    for old, _ in edits:
        if source.count(old) != 1:
            raise SystemExit(f"{lib}: edit anchor found {source.count(old)} times: {old[:70]!r}")
    for old, new in edits:
        source = source.replace(old, new)
    source_path.write_text(source, encoding="utf-8")
    name, commands = record_of(lib)
    warnings = []
    for cmd in commands:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"{' '.join(cmd)}\n{r.stderr}")
        warnings.append(r.stderr)
    out = {
        "record": name,
        "commands": commands,
        "compiler_stderr": warnings,
        "edits": ["LOSS-ONCE"] + (["GATE-SPAN"] if len(edits) > 1 else []),
        "overlay_before_sha256": before,
        "overlay_sha256": sha(source_path),
        "native_sha256": sha(build / f"{lib}.so"),
    }
    (build / f"{lib}-build-once.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )
    print("once", build / f"{lib}.so", ", ".join(out["edits"]))
