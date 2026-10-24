"""GEO-2b level 2: Codex's research overlays ported to current main's training rail.

The same graft Codex's build_upgrade.py made onto merged PR818 (2376511), made onto origin/main
(490f5be0; its runtime is 44b16ff7's): from each saved ABI-3 research overlay (foundation/*.c) the
blocks build_upgrade.py replaces are replaced by main's -- the state checks (valid_with, REMAT's
recompute-aware arena, WS-PLAN's floor), the bf16 workspace functions (the attention model takes
main's exact need; the research heads keep their own group bound, which main's ws_need cannot
describe), the GEMM dispatch (WS-PLAN's refusal returns), the attention entry (BF16-ATTN), the
SwiGLU block's forward and backward (BF16-PANELS under WS-PLAN's gate), and the optimizer and run
tail (FACTS-RUN, run_args; not the PX-2 program runner, which runs the transformer's launch program,
a step the research heads do not have) -- and compiled with main's tensor unit and
headers (ABI 5). Geometry, compact layouts, reconstruction tiers, causal prefix/VJP and the masked
loss stay the research implementation, as in build_upgrade.py.

Main's helpers return whether a product ran (WS-PLAN); the research steps were written when they
returned nothing. Every refusal is therefore counted where it happens -- in gemm(), in
gemm_group_with() and at each direct tensor entry of the SwiGLU blocks -- and
bcir_geo_refused() returns the count, which the harness requires to be 0 after every run: no
product is ever skipped silently.

usage: python3 -I build_main.py WORK_DIR MAIN_SRC_DIR
  WORK_DIR has foundation/*.c (the bundle's); MAIN_SRC_DIR has main's bcir_tensor.c/.h and
  bcir_decoder_train.c/.h. Writes WORK_DIR/build/<name>.c, .so and <name>-build-main.json.
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

WORK, SRC = Path(sys.argv[1]), Path(sys.argv[2])
NATIVE = (SRC / "bcir_decoder_train.c").read_text()
NAMES = (
    "guard_compact",
    "guard_frame_ff_compact",
    "guard_layer_frame_ff",
    "query_guard_compact",
    "query_guard_frame_ff_compact",
    "query_guard_layer_frame_ff",
    "guard_hybrid_compact",
    "attention",
)


def once(s, a, b):
    assert s.count(a) == 1, (a[:100], s.count(a))
    return s.replace(a, b)


def block(s, a, b):
    return s[s.index(a) : s.index(b, s.index(a))]


def counted(text):
    """main's `ran&=X;` statements in a grafted block, each refusal counted where it happens: a
    direct tensor entry's refusal here, a GEMM helper's inside gemm()/gemm_group_with()."""

    def sub(m):
        expr = m.group(1)
        if expr.lstrip().startswith("bcir_tensor_"):
            return f"dt_refused+=!({expr});"
        return f"(void)({expr});"

    out = re.sub(r"ran&=(.*?);", sub, text, flags=re.S)
    assert "ran&=" not in out
    return out


PREAMBLE = block(NATIVE, "_Static_assert(sizeof(float)==4", "uint32_t bcir_decoder_abi_version")
COUNTER = """/* GEO-2b: main's products return whether they ran (WS-PLAN); a research step written before that
 * ignores the return, so every refusal is counted here and the harness requires the count 0. */
static unsigned long dt_refused;
unsigned long bcir_geo_refused(void);
unsigned long bcir_geo_refused(void) { return dt_refused; }
"""
#: main's workspace and REMAT region; the research heads replace its workspace_bytes with their own
WS_REGION = block(NATIVE, "static size_t ws_most(", "typedef struct layer ")
MAIN_WS_BYTES = (
    "size_t bcir_decoder_workspace_bytes(const bcir_decoder_plan *plan,uint32_t contract) {\n"
    "  return bcir_decoder_workspace_need(plan,contract,0);\n}\n"
)
RESEARCH_WS = """static size_t ws_sum(size_t a,size_t b){return a>SIZE_MAX-b?SIZE_MAX:a+b;}
size_t bcir_decoder_workspace_bytes(const bcir_decoder_plan *plan,uint32_t contract){
  bcir_decoder_plan p;geometry g;size_t q,c,most=0;
  if(!(contract&BCIR_DT_CONTRACT_BF16))return 0;
  if(!plan||bcir_decoder_make_plan(&plan->spec,&p))return SIZE_MAX;g=geom(&p.spec);
  {const size_t n=g.n,d=g.d,f=g.f,v=g.v,r=GP_RANK,a=GP_P;
   /* Upper bound of each actual research-head group, including the query side channel. */
   const size_t groups[][4][3]={
    {{n,r,d},{n,d,d}},{{n,d,a}},{{n,d,r}},{{n,d,d}},
    {{n,f,d},{n,f,d}},{{n,d,f}},{{n,v,d}},
    {{n,d,v},{v,d,n}},{{n,f,d},{d,f,n}},
    {{n,d,f},{n,d,f},{f,d,n},{f,d,n}},
    {{n,d,d},{d,d,n}},{{n,a,d},{d,a,n}},
    {{n,r,d},{d,r,n}},{{n,d,r},{r,d,n}}
   };
   for(q=0;q<sizeof(groups)/sizeof(groups[0]);q++){
    size_t sum=0;for(c=0;c<4&&groups[q][c][0];c++)sum=ws_sum(sum,bcir_tensor_mm_bf16_workspace(groups[q][c][0],groups[q][c][1],groups[q][c][2]));
    if(sum>most)most=sum;
   }
  }return most;
}
"""


def workspace(global_head):
    if not global_head:
        return WS_REGION
    return once(WS_REGION, MAIN_WS_BYTES, RESEARCH_WS)


def dispatch():
    """main's GEMM dispatch block, every bf16 GEMM refusal counted at its source."""
    s = block(NATIVE, "static int gemm_contract(", "/* y = silu(")
    s = once(
        s,
        "    return bcir_tensor_mm_bf16_group_panels_pool(s->pool,chain,nchain,y,ny,pre,s->workspace,\n"
        "        s->workspace_bytes);",
        "    { int ok=bcir_tensor_mm_bf16_group_panels_pool(s->pool,chain,nchain,y,ny,pre,s->workspace,\n"
        "        s->workspace_bytes); dt_refused+=!ok; return ok; }",
    )
    s = once(
        s,
        "    return bcir_tensor_mm_bf16_pool(s->pool,ta,tb,m,n,k,alpha,a,b,beta,c,s->workspace,s->workspace_bytes);",
        "    { int ok=bcir_tensor_mm_bf16_pool(s->pool,ta,tb,m,n,k,alpha,a,b,beta,c,s->workspace,s->workspace_bytes);\n"
        "      dt_refused+=!ok; return ok; }",
    )
    return s


def ff_forward():
    """main's SwiGLU forward (inside layer_forward, REMAT) as the research forward's statement list:
    `full` is 1 there (the research heads recompute through their own tiers), `bad` a local."""
    s = block(
        NATIVE,
        "  if (panels(s,g,a.hidden,a.hidden)) {",
        "  if (full) bcir_tensor_add_pool(s->pool,g.nd,a.x2,y);",
    )
    s = s.replace("if (full) ", "")
    s = s.replace(",bad);", ",&bad);")
    return counted(s).rstrip()


def upgraded(s, name):
    s = once(
        s,
        "uint32_t bcir_decoder_abi_version(void) { return 3; }",
        PREAMBLE + COUNTER + "uint32_t bcir_decoder_abi_version(void) { return 5; }",
    )
    s = once(
        s,
        block(s, "static int overlap(", "static int ids("),
        block(NATIVE, "static int overlap(", "static int ids("),
    )
    s = once(s, "typedef struct layer ", workspace(name != "attention") + "typedef struct layer ")
    s = once(s, block(s, "static int gemm_contract(", "/* y = silu("), dispatch())
    s = once(
        s,
        "static void swiglu_backward(",
        "static int swiglu_v2(const bcir_decoder_state *s){return (s->contract&BCIR_DT_CONTRACT_EXP)!=0;}\n"
        "static void swiglu_backward(",
    )
    if name == "attention":
        s = once(
            s,
            block(s, "static void attention(", "/* The forward,"),
            block(NATIVE, "static unsigned attention_contract(", "/* The forward,"),
        )
        s = once(
            s,
            "bcir_tensor_attention_backward_inplace_pool(s->pool,g.b,g.t,g.h,g.kh,g.hd,a.q,a.k,a.v,a.p,\n        dc,dq,dk,dv);",
            "dt_refused+=!attention_backward(s,g,a.q,a.k,a.v,a.p,dc,dq,dk,dv);",
        )
    oldff = "    silu_mul(s,g.nf,a.gate,a.up,a.hidden,&bad);\n    linear(s,g.n,g.f,g.d,a.hidden,p.down,y);"
    s = once(s, oldff, ff_forward())
    # Reconstruct FF gates before the block; its down VJP writes only dh, apart (build_upgrade.py).
    start = s.index("    linear_backward(s,g.n,g.f,g.d,a.hidden,p.down,cur,dh,grad.down,0.0f,wb);")
    end = s.index("    /* NDT-RMS-ACC:", start)
    old = s[start:end]
    remat = ""
    anchor = "    swiglu_backward(s,g.nf,a.gate,a.up,dh,dg,du);"
    if "frame_ff" in name or "layer_frame" in name:
        remat = old[old.index("\n") + 1 : old.index(anchor)]
        assert "linears(s,g.n,g.d,a.n2,2,out,w,yy);" in remat or "forward_layer" in remat
        # Panel writers use nonlocal tiled stores; compact FF's cross aliases are only qualified
        # for binary32 elementwise writers. Keep the packed path off here (build_upgrade.py).
        s = "#define BCIR_DT_NO_PANELS 1\n" + s
        start = s.index(
            "    linear_backward(s,g.n,g.f,g.d,a.hidden,p.down,cur,dh,grad.down,0.0f,wb);"
        )
        end = s.index("    /* NDT-RMS-ACC:", start)
    newback = counted(
        block(NATIVE, "    /* the hidden activation is a panel exactly when", "    /* NDT-RMS-ACC:")
    )
    s = s[:start] + remat + newback + s[end:]
    # The tail up to the PX-2 program runner: it runs the transformer's launch program (its records
    # name the transformer's embed, attention, loss rows and scatter), a step the research heads do
    # not have and the study never calls; the trace probe after it is a build option off here.
    tail = NATIVE[
        NATIVE.index("typedef struct adamw_terms ") : NATIVE.index(
            "/* PX-2 (PACK): the run from a launch program"
        )
    ]
    # Preserve research ignored-target support; production dense next-token loss has none.
    tail = once(tail, "!ids_in(s,placed,targets,total)", "!target_ids(s,targets,total)")
    s = (
        s[: s.index("typedef struct adamw_terms ")]
        + "#ifndef DT_COMMITTED\n#define DT_COMMITTED 4u\n#endif\n"
        + tail
    )
    return s


def compile_one(source, name, extra=()):
    path = WORK / "build" / f"{name}.c"
    path.write_text(source)
    cmd = [
        "gcc",
        "-std=c11",
        "-O3",
        "-fPIC",
        "-shared",
        "-ffp-contract=off",
        "-fno-math-errno",
        "-pthread",
        "-I",
        str(SRC),
        "-DGP_RANK=22",
        "-DGP_PROJECTOR=0",
        "-DGP_SYMMETRIC=0",
        "-DGP_HEADS=1",
        *extra,
        str(SRC / "bcir_tensor.c"),
        str(path),
        "-lm",
        "-o",
        str(path.with_suffix(".so")),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode:
        raise SystemExit(f"{name}: build failed\n{r.stderr[-6000:]}")
    (WORK / "build" / f"{name}-build-main.json").write_text(
        json.dumps(
            {
                "command": cmd,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "native_sha256": hashlib.sha256(NATIVE.encode()).hexdigest(),
                "tensor_sha256": hashlib.sha256((SRC / "bcir_tensor.c").read_bytes()).hexdigest(),
                "stderr": r.stderr,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        name,
        "built",
        len([ln for ln in r.stderr.splitlines() if "warning" in ln]),
        "warning(s)",
        flush=True,
    )


if __name__ == "__main__":
    only = sys.argv[3:]
    for name in NAMES:
        if only and name not in only:
            continue
        compile_one(upgraded((WORK / "foundation" / f"{name}.c").read_text(), name), name)
