"""GP-LANES on the GEO study's geometric heads: the pair kernels (gp_pair_forward, gp_pair_backward)
and the query frame's (gp_query_contract_pair, gp_query_contract_vjp) sixteen rows at a time, a row
a vector lane (gp_lanes_pair.inc, gp_lanes_query.inc beside this script): each lane runs its row's
IEEE operations in the scalar kernel's order, so every feature and adjoint is the fork's bit for
bit, and a row the scalar kernel sends to its FP64 fallback is recomputed by the scalar kernel. Each
group prefetches the next group's rows (a hint to the caches, no value changes).
The forks' jobs take a task a group of sixteen rows where the CPU has AVX-512F, the rows past the
last whole group by the scalar task; elsewhere the fork's tasks as they were. Applied to
guard_compact (the pair kernels) and query_guard_compact (and its query frame) in a work tree as
once_level3.py left it, each rebuilt by its build record's own commands, the tensor unit's
object as the record left it.

usage: python3 -I gp_lanes_overlay.py OUT_WORK LIB [LIB ...]
  OUT_WORK a level-2 work tree (<lib>-build-amx.json) once_level3.py edited, or a copy of one: the
  recorded commands are pointed at OUT_WORK's build/ before they run. Rewrites
  OUT_WORK/build/<lib>.c, .o and .so, and writes OUT_WORK/build/<lib>-build-lanes.json.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(sys.argv[1]).resolve()
LIBS = sys.argv[2:]
build = OUT / "build"
HERE = Path(__file__).resolve().parent
PAIR = (HERE / "gp_lanes_pair.inc").read_text(encoding="utf-8")
QUERY = (HERE / "gp_lanes_query.inc").read_text(encoding="utf-8")
PAIR_TASKS = """/* GP-LANES: a task a group of sixteen rows; the rows past the last whole group, and a row the
 * scalar kernel sends to its fallback, by the fork's scalar task */
GPL_TARGET static void gpl_forward_group(const gp_job *j,size_t r0) {
  size_t n=j->batch*j->time,l; unsigned char ill[GPL];
  _Alignas(64) float U[GP_RANK][GPL],V[GP_RANK][GPL],O[GP_P][GPL];
  if(r0+2*GPL<=n){size_t q=r0+GPL;  /* the next group's rows */
    gpl_pf_r(j->z+q*GP_RANK,GPL*GP_RANK*4);gpl_pf_r(j->z+(n+q)*GP_RANK,GPL*GP_RANK*4);gpl_pf_w(j->p+q*GP_P,GPL*GP_P*4);}
  gpl_in(j->z+r0*GP_RANK,GP_RANK,U); gpl_in(j->z+(n+r0)*GP_RANK,GP_RANK,V);
  gp_pair_forward_lanes(U,V,O,ill); gpl_out(O,GP_P,j->p+r0*GP_P,ill);
  for(l=0;l<GPL;l++) if(ill[l]) gp_forward_task((void *)j,r0+l);
}
static void gpl_forward_task(void *ctx,size_t g) {
  gp_job *j=ctx; size_t n=j->batch*j->time,r0=g*GPL,r;
  if(r0+GPL<=n) gpl_forward_group(j,r0); else for(r=r0;r<n;r++) gp_forward_task(ctx,r);
}
GPL_TARGET static void gpl_backward_group(const gp_job *j,size_t r0) {
  size_t n=j->batch*j->time,l; unsigned char ill[GPL];
  _Alignas(64) float U[GP_RANK][GPL],V[GP_RANK][GPL],DP[GP_P][GPL],DU[GP_RANK][GPL],DV[GP_RANK][GPL];
  if(r0+2*GPL<=n){size_t q=r0+GPL;
    gpl_pf_r(j->z+q*GP_RANK,GPL*GP_RANK*4);gpl_pf_r(j->z+(n+q)*GP_RANK,GPL*GP_RANK*4);gpl_pf_r(j->dp+q*GP_P,GPL*GP_P*4);
    gpl_pf_w(j->dz+q*GP_RANK,GPL*GP_RANK*4);gpl_pf_w(j->dz+(n+q)*GP_RANK,GPL*GP_RANK*4);}
  gpl_in(j->z+r0*GP_RANK,GP_RANK,U); gpl_in(j->z+(n+r0)*GP_RANK,GP_RANK,V); gpl_in(j->dp+r0*GP_P,GP_P,DP);
  gp_pair_backward_lanes(U,V,DP,DU,DV,ill);
  gpl_out(DU,GP_RANK,j->dz+r0*GP_RANK,ill); gpl_out(DV,GP_RANK,j->dz+(n+r0)*GP_RANK,ill);
  for(l=0;l<GPL;l++) if(ill[l]) gp_backward_task((void *)j,r0+l);
}
static void gpl_backward_task(void *ctx,size_t g) {
  gp_job *j=ctx; size_t n=j->batch*j->time,r0=g*GPL,r;
  if(r0+GPL<=n) gpl_backward_group(j,r0); else for(r=r0;r<n;r++) gp_backward_task(ctx,r);
}
"""
QUERY_TASKS = """static void gp_query_backward_task(void *ctx,size_t row);
GPL_TARGET static void gpl_query_forward_group(const gp_query_job *j,size_t r0) {
  _Alignas(64) float U[GP_RANK][GPL],P[GP_P][GPL],C[GP_RANK][GPL];
  if(r0+2*GPL<=j->n){size_t q=r0+GPL;
    gpl_pf_r(j->z+q*GP_RANK,GPL*GP_RANK*4);gpl_pf_r(j->p+q*GP_P,GPL*GP_P*4);gpl_pf_w(j->c+q*GP_RANK,GPL*GP_RANK*4);}
  gpl_in(j->z+r0*GP_RANK,GP_RANK,U); gpl_in(j->p+r0*GP_P,GP_P,P);
  gp_query_contract_pair_lanes(U,P,C); gpl_out(C,GP_RANK,j->c+r0*GP_RANK,NULL);
}
static void gpl_query_forward_task(void *ctx,size_t g) {
  gp_query_job *j=ctx; size_t r0=g*GPL,r;
  if(r0+GPL<=j->n) gpl_query_forward_group(j,r0); else for(r=r0;r<j->n;r++) gp_query_forward_task(ctx,r);
}
/* gp_query_backward_task a group at a time: the features rebuilt, the query frame's adjoint into
 * the local dp, the pair adjoint, the direct path added -- the scalar task's sequence */
GPL_TARGET static void gpl_query_backward_group(const gp_query_job *j,size_t r0) {
  size_t i,l; unsigned char ill[GPL],ill2[GPL];
  _Alignas(64) float U[GP_RANK][GPL],V[GP_RANK][GPL],P[GP_P][GPL],DP[GP_P][GPL],DC[GP_RANK][GPL];
  _Alignas(64) float DIRECT[GP_RANK][GPL],DU[GP_RANK][GPL],DV[GP_RANK][GPL];
  if(r0+2*GPL<=j->n){size_t q=r0+GPL;
    gpl_pf_r(j->z+q*GP_RANK,GPL*GP_RANK*4);gpl_pf_r(j->z+(j->n+q)*GP_RANK,GPL*GP_RANK*4);
    gpl_pf_r(j->dp+q*GP_P,GPL*GP_P*4);gpl_pf_r(j->dc+q*GP_RANK,GPL*GP_RANK*4);
    gpl_pf_w(j->dz+q*GP_RANK,GPL*GP_RANK*4);gpl_pf_w(j->dz+(j->n+q)*GP_RANK,GPL*GP_RANK*4);}
  gpl_in(j->z+r0*GP_RANK,GP_RANK,U); gpl_in(j->z+(j->n+r0)*GP_RANK,GP_RANK,V);
  gp_pair_forward_lanes(U,V,P,ill);
  gpl_in(j->dp+r0*GP_P,GP_P,DP); gpl_in(j->dc+r0*GP_RANK,GP_RANK,DC);
  gp_query_contract_vjp_lanes(U,P,DC,DP,DIRECT);
  gp_pair_backward_lanes(U,V,DP,DU,DV,ill2);
  for(i=0;i<GP_RANK;i++) ST(DU[i],_mm512_add_ps(LD(DU[i]),LD(DIRECT[i])));
  for(l=0;l<GPL;l++) ill[l]|=ill2[l];
  gpl_out(DU,GP_RANK,j->dz+r0*GP_RANK,ill); gpl_out(DV,GP_RANK,j->dz+(j->n+r0)*GP_RANK,ill);
  for(l=0;l<GPL;l++) if(ill[l]) gp_query_backward_task((void *)j,r0+l);
}
static void gpl_query_backward_task(void *ctx,size_t g) {
  gp_query_job *j=ctx; size_t r0=g*GPL,r;
  if(r0+GPL<=j->n) gpl_query_backward_group(j,r0); else for(r=r0;r<j->n;r++) gp_query_backward_task(ctx,r);
}
"""
DISPATCH = (
    (
        "  bcir_tensor_pool_run(pool,gp_forward_task,&j,batch*time);\n",
        "  if(gpl_have()) bcir_tensor_pool_run(pool,gpl_forward_task,&j,(batch*time+GPL-1)/GPL);  /* GP-LANES */\n"
        "  else bcir_tensor_pool_run(pool,gp_forward_task,&j,batch*time);\n",
    ),
    (
        "  bcir_tensor_pool_run(pool,gp_backward_task,&j,batch*time);\n",
        "  if(gpl_have()) bcir_tensor_pool_run(pool,gpl_backward_task,&j,(batch*time+GPL-1)/GPL);  /* GP-LANES */\n"
        "  else bcir_tensor_pool_run(pool,gp_backward_task,&j,batch*time);\n",
    ),
)
QUERY_DISPATCH = (
    (
        "  bcir_tensor_pool_run(pool,gp_query_forward_task,&j,n);\n",
        "  if(gpl_have()) bcir_tensor_pool_run(pool,gpl_query_forward_task,&j,(n+GPL-1)/GPL);  /* GP-LANES */\n"
        "  else bcir_tensor_pool_run(pool,gp_query_forward_task,&j,n);\n",
    ),
    (
        "  bcir_tensor_pool_run(pool,gp_query_backward_task,&j,batch*time);\n",
        "  if(gpl_have()) bcir_tensor_pool_run(pool,gpl_query_backward_task,&j,(batch*time+GPL-1)/GPL);  /* GP-LANES */\n"
        "  else bcir_tensor_pool_run(pool,gp_query_backward_task,&j,batch*time);\n",
    ),
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def once(text, old, new, lib):
    if text.count(old) != 1:
        raise SystemExit(f"{lib}: anchor found {text.count(old)} times: {old[:70]!r}")
    return text.replace(old, new)


def record_of(lib):
    """The library's level-2 build record and its commands after the tensor unit's object, pointed
    at OUT's build/ (once_level3.py's rule)."""
    name = f"{lib}-build-amx.json"
    rec = json.loads((build / name).read_text(encoding="utf-8"))
    cmds = [c for c in rec["commands"] if not any(x.endswith("bcir_tensor.c") for x in c)]
    made = Path(cmds[-1][cmds[-1].index("-o") + 1])
    if made.name != f"{lib}.so":
        raise SystemExit(f"{name}: the last command makes {made.name}, not {lib}.so")
    there = str(made.parent)
    cmds = [[x.replace(there, str(build)) for x in c] for c in cmds]
    if any(str(build) not in " ".join(c) for c in cmds):
        raise SystemExit(f"{name}: a command does not name {build}")
    return name, cmds


for lib in LIBS:
    if lib not in ("guard_compact", "query_guard_compact"):
        raise SystemExit(f"{lib}: GP-LANES edits the geometric heads only")
    path = build / f"{lib}.c"
    src = path.read_text(encoding="utf-8")
    before = sha(path)
    if "GATE-SPAN" not in src or "LOSS-ONCE" not in src:
        raise SystemExit(f"{lib}: not once_level3.py's overlay")
    src = once(
        src,
        "static void gp_forward_task(void *ctx,size_t row) {\n",
        PAIR + "static void gp_forward_task(void *ctx,size_t row) {\n",
        lib,
    )
    src = once(
        src,
        "void gp_wedge_forward(bcir_tensor_pool *pool,",
        PAIR_TASKS + "void gp_wedge_forward(bcir_tensor_pool *pool,",
        lib,
    )
    for old, new in DISPATCH:
        src = once(src, old, new, lib)
    edits = ["GP-LANES pair"]
    if lib == "query_guard_compact":
        src = once(
            src,
            "void gp_query_contract_forward(",
            QUERY + QUERY_TASKS + "void gp_query_contract_forward(",
            lib,
        )
        for old, new in QUERY_DISPATCH:
            src = once(src, old, new, lib)
        edits.append("GP-LANES query")
    path.write_text(src, encoding="utf-8")
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
        "edits": edits,
        "overlay_before_sha256": before,
        "overlay_sha256": sha(path),
        "native_sha256": sha(build / f"{lib}.so"),
    }
    (build / f"{lib}-build-lanes.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )
    print("lanes", build / f"{lib}.so", ", ".join(edits))
