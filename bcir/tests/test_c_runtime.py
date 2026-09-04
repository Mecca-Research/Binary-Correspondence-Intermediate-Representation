"""Freestanding C StreamPack runtime parity (Phase 8): Python encodes, C decodes."""

import os
import subprocess
import tempfile
from shutil import which

from bcir.abi import encode
from bcir.examples import vector_add
from bcir.gem import hydrate
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta

_C_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "runtime", "c")


def _pack():
    m = vector_add(1024)
    return hydrate(m, optimize(m, TargetProfile.x86_avx512(), Theta.cool()))


def test_runtime_is_freestanding():
    # The runtime itself must compile with no libc / freestanding.
    clang = which("clang")
    if clang is None:
        return
    r = subprocess.run(
        [
            clang,
            "-ffreestanding",
            "-nostdlib",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-c",
            os.path.join(_C_DIR, "bcir_runtime.c"),
            "-o",
            os.devnull,
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr


def test_python_encodes_c_decodes():
    clang = which("clang")
    if clang is None:
        return  # skip cleanly without a C compiler
    pack = _pack()
    seg = pack.segments[0]
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "test_runtime")
        build = subprocess.run(
            [
                clang,
                "-std=c11",
                "-O2",
                os.path.join(_C_DIR, "bcir_runtime.c"),
                os.path.join(_C_DIR, "test_runtime.c"),
                "-I",
                _C_DIR,
                "-o",
                exe,
            ],
            capture_output=True,
            text=True,
        )
        assert build.returncode == 0, build.stderr

        blob = os.path.join(d, "pack.bin")
        with open(blob, "wb") as f:
            f.write(encode(pack))
        run = subprocess.run([exe, blob], capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
        out = dict(line.split("=", 1) for line in run.stdout.splitlines() if "=" in line)

        # Cross-language ABI fidelity: the C decode matches the Python encode.
        assert "OK" in run.stdout
        assert int(out["version"]) == 1
        assert int(out["data_gen"]) == pack.data_gen
        assert int(out["n_segments"]) == len(pack.segments)
        assert int(out["walked"]) == len(pack.segments)
        assert int(out["seg0.claim_id"]) == seg.claim_id
        assert int(out["seg0.lane"]) == int(seg.lane)
        assert int(out["seg0.width"]) == seg.width
        assert out["seg0.opcode"] == seg.opcode
        assert int(out["seg0.read0"]) == seg.reads[0]
        assert int(out["seg0.write0"]) == seg.writes[0]
        assert int(out["pipeline_depth"]) == 1  # v1 pack: single phase in flight


def test_python_encodes_v2_c_decodes():
    # StreamPack v2 (append-only): the C runtime accepts the new version and
    # reads the pipeline contract; the segment stream stays v1-shaped.
    clang = which("clang")
    if clang is None:
        return
    from bcir.gem import hydrate_pipelined
    from bcir.kbcir import optimize as _opt

    m = vector_add(1024)
    pack = hydrate_pipelined(m, _opt(m, TargetProfile.x86_avx512(), Theta.cool()), depth=2)
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "test_runtime")
        build = subprocess.run(
            [
                clang,
                "-std=c11",
                "-O2",
                os.path.join(_C_DIR, "bcir_runtime.c"),
                os.path.join(_C_DIR, "test_runtime.c"),
                "-I",
                _C_DIR,
                "-o",
                exe,
            ],
            capture_output=True,
            text=True,
        )
        assert build.returncode == 0, build.stderr
        blob = os.path.join(d, "pack.bin")
        with open(blob, "wb") as f:
            f.write(encode(pack))
        run = subprocess.run([exe, blob], capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
        out = dict(line.split("=", 1) for line in run.stdout.splitlines() if "=" in line)
        assert int(out["version"]) == 2
        assert int(out["pipeline_depth"]) == 2
        assert int(out["walked"]) == len(pack.segments)


def _build_runtime_harness(
    clang, d, name="test_runtime", srcs=("bcir_runtime.c", "test_runtime.c")
):
    exe = os.path.join(d, name)
    build = subprocess.run(
        [
            clang,
            "-std=c23",
            "-O2",
            "-Wall",
            "-Wextra",
            *[os.path.join(_C_DIR, s) for s in srcs],
            "-I",
            _C_DIR,
            "-o",
            exe,
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    return exe


def _v3_pack():
    from bcir.gem.streampack import LaneSegment, Prefetch, StreamPack, TraceNote
    from bcir.model import Lane

    p = StreamPack(source_plan="plan0", topo_gen=1, map_gen=7, data_gen=19)
    p.prefetches.append(Prefetch("pf0", 4, (10, 11)))
    p.segments.append(
        LaneSegment(
            name="seg0",
            claim_id=1000,
            phase_id=0,
            lane=Lane.GGG,
            width=16,
            opcode="reduce.add",
            reads=(10, 11),
            writes=(12,),
            prefetch="pf0",
            dispatch="pim",
            channel="nvidia_ptx",
        )
    )
    p.trace_notes.append(TraceNote(claim_id=1000))
    return p


def test_python_encodes_v3_c_decodes_dispatch_channel():
    # StreamPack v3 (append-only): the C runtime accepts v3 and reads the on-wire
    # segment dispatch (pim) + channel (nvidia_ptx) -- the routing fields are now
    # inside the CRC, so a mutation is no longer invisible to byte-identity.
    clang = which("clang")
    if clang is None:
        return
    pack = _v3_pack()
    with tempfile.TemporaryDirectory() as d:
        exe = _build_runtime_harness(clang, d)
        blob = os.path.join(d, "v3.bin")
        with open(blob, "wb") as f:
            f.write(encode(pack))
        run = subprocess.run([exe, blob], capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
        out = dict(line.split("=", 1) for line in run.stdout.splitlines() if "=" in line)
        assert int(out["version"]) == 3
        assert int(out["seg0.dispatch"]) == 1  # pim
        assert out["seg0.channel"] == "nvidia_ptx"
        assert out["seg0.prefetch"] == "pf0"


def test_c_decode_equals_python_decode_differential():
    # C-decode == Python-decode on every field, over v1 / v2 / v3 packs (the
    # lane-asymmetry class: a value the Python decode normalizes/raises on must
    # decode identically on the C rail).
    clang = which("clang")
    if clang is None:
        return
    from bcir.abi import decode as py_decode
    from bcir.gem import hydrate_pipelined

    m = vector_add(1024)
    packs = {
        1: _pack(),
        2: hydrate_pipelined(m, optimize(m, TargetProfile.x86_avx512(), Theta.cool()), depth=2),
        3: _v3_pack(),
    }
    with tempfile.TemporaryDirectory() as d:
        exe = _build_runtime_harness(clang, d)
        for want_ver, pack in packs.items():
            blob = os.path.join(d, f"p{want_ver}.bin")
            with open(blob, "wb") as f:
                f.write(encode(pack))
            run = subprocess.run([exe, blob], capture_output=True, text=True)
            assert run.returncode == 0, run.stdout + run.stderr
            c = dict(line.split("=", 1) for line in run.stdout.splitlines() if "=" in line)
            py = py_decode(encode(pack))
            s = py.segments[0]
            assert int(c["version"]) == want_ver
            assert int(c["pipeline_depth"]) == py.pipeline_depth
            assert int(c["map_gen"]) == py.map_gen
            assert int(c["data_gen"]) == py.data_gen
            assert int(c["n_segments"]) == len(py.segments)
            assert int(c["seg0.claim_id"]) == s.claim_id
            assert int(c["seg0.lane"]) == int(s.lane)
            assert int(c["seg0.width"]) == s.width
            assert c["seg0.opcode"] == s.opcode
            assert int(c["seg0.read0"]) == s.reads[0]
            assert int(c["seg0.write0"]) == s.writes[0]
            assert int(c["seg0.dispatch"]) == {"core": 0, "pim": 1}[s.dispatch]
            assert c["seg0.channel"] == s.channel  # "host" on v1/v2, set on v3
            assert c["seg0.prefetch"] == (s.prefetch or "")


def test_c_rejects_crc_valid_semantically_corrupt_packs():
    # The freestanding trust boundary (R10/R11 + lane/width/dispatch range): each CRC-VALID
    # but semantically-corrupt pack is REJECTED by bcir_sp_verify_semantic + the checked
    # executor, where the bare decoder used to accept it silently. Mirrors verify_pack.
    clang = which("clang")
    if clang is None:
        return
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    mutator = os.path.join(repo, "tools", "c", "streampack_corrupt.py")
    want = {
        "dangling_claim": "BCIR_ERR_PROVENANCE",
        "swapped_rid": "BCIR_ERR_PROVENANCE",
        "out_of_range_lane": "BCIR_ERR_LANE",
        "out_of_range_width": "BCIR_ERR_WIDTH",
        "unresolved_prefetch": "BCIR_ERR_PROVENANCE",
        "invalid_pipeline": "BCIR_ERR_PROVENANCE",
        "invalid_buffers": "BCIR_ERR_PROVENANCE",
        "stale_generation": "BCIR_ERR_STALE",
        "tampered_dispatch": "BCIR_ERR_DISPATCH",
        "trailing_bytes": "BCIR_ERR_TRAILING",
        "duplicate_segment": "BCIR_ERR_PROVENANCE",
        "duplicate_prefetch": "BCIR_ERR_PROVENANCE",
        "duplicate_trace": "BCIR_ERR_PROVENANCE",
    }
    env = dict(os.environ, PYTHONPATH=repo + os.pathsep + os.environ.get("PYTHONPATH", ""))
    with tempfile.TemporaryDirectory() as d:
        exe = _build_runtime_harness(
            clang,
            d,
            name="test_sem",
            srcs=("bcir_exec.c", "bcir_runtime.c", "bcir_verify.c", "test_streampack_semantic.c"),
        )
        for kind, code in want.items():
            bin_path = os.path.join(d, f"{kind}.bin")
            meta = subprocess.run(
                [sys.executable, mutator, kind, bin_path], capture_output=True, text=True, env=env
            )
            assert meta.returncode == 0, meta.stderr
            _k, _n, mapg, datag = meta.stdout.split()
            argv = [exe, bin_path]
            if kind == "stale_generation":
                argv += [mapg, datag]  # tell the C rail the live generation
            run = subprocess.run(argv, capture_output=True, text=True)
            # harness prints:  crc=<name>\nstatus=<n> name=<NAME>\nexec=<NAME>
            fields = {}
            for line in run.stdout.splitlines():
                for tok in line.split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        fields[k] = v
            # the pack still passes CRC (it is a CRC-fixed corruption) ...
            assert fields.get("crc") == "BCIR_OK", (kind, run.stdout)
            # ... but the semantic verifier AND the checked executor both reject it.
            assert fields.get("name") == code, (kind, run.stdout)
            assert fields.get("exec") == code, (kind, run.stdout)
            # The graph-verifier API has no live registry argument, so only the
            # stale-generation case remains intrinsically well-formed there.
            assert fields.get("verify_pack") == (
                "BCIR_OK" if kind == "stale_generation" else "BCIR_ERR"
            ), (kind, run.stdout)
            assert run.returncode == 1, (kind, run.stdout)


def test_c_planner_rejects_unrepresentable_cost_instead_of_wrapping():
    clang = which("clang")
    if clang is None:
        return
    source = (
        '#include <stdint.h>\n#include <string.h>\n#include "bcir_plan.h"\n'
        "int main(void){bcir_claim c;bcir_func f;bcir_plan_step s;bcir_plan p;"
        "memset(&c,0,sizeof c);memset(&f,0,sizeof f);memset(&s,0xa5,sizeof s);"
        "memset(&p,0xa5,sizeof p);"
        "c.opcode=BCIR_OP_ATOMIC_ADD;c.domain=BCIR_DOM_MMIO;c.count=UINT32_MAX;"
        "f.claims=&c;f.n_claims=1;"
        "if(bcir_plan_func(&f,&s,1,&p)!=BCIR_ERR_OVERFLOW)return 1;"
        "for(size_t i=0;i<sizeof s;i++)if(((unsigned char*)&s)[i]!=0xa5)return 2;"
        "if(p.steps||p.n||p.total_cost)return 3;return 0;}\n"
    )
    with tempfile.TemporaryDirectory() as d:
        driver = os.path.join(d, "plan_overflow.c")
        with open(driver, "w", encoding="utf-8") as f:
            f.write(source)
        exe = os.path.join(d, "plan_overflow")
        build = subprocess.run(
            [
                clang,
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                _C_DIR,
                os.path.join(_C_DIR, "bcir_plan.c"),
                driver,
                "-o",
                exe,
            ],
            capture_output=True,
            text=True,
        )
        assert build.returncode == 0, build.stderr
        assert subprocess.run([exe], capture_output=True).returncode == 0


def test_c_hydrator_preflights_before_writing_an_artifact():
    clang = which("clang")
    if clang is None:
        return
    source = r"""
#include <stdint.h>
#include <string.h>
#include "bcir_hydrate.h"
static int filled(const uint8_t *p,size_t n){for(size_t i=0;i<n;i++)if(p[i]!=0xa5)return 0;return 1;}
int main(void){
  bcir_claim c; bcir_func f; bcir_plan_step s; bcir_plan p; uint8_t out[512]; size_t n=99;
  memset(&c,0,sizeof c); memset(&f,0,sizeof f); memset(out,0xa5,sizeof out);
  c.id=7; c.opcode=BCIR_OP_ADD; c.lane=BCIR_LANE_U; c.count=8; strcpy(c.op,"c.bin.add");
  f.claims=&c; f.n_claims=1; s.claim_id=7; s.width=8; s.cost=8;
  p.steps=&s; p.n=1; p.total_cost=8;
  if(bcir_hydrate(&f,&p,out,1,&n)!=BCIR_ERR_NOSPACE||n||!filled(out,sizeof out))return 1;
  n=99; s.claim_id=8;
  if(bcir_hydrate(&f,&p,out,sizeof out,&n)!=BCIR_ERR_PROVENANCE||n||!filled(out,sizeof out))return 2;
  n=99; s.claim_id=7; s.width=3;
  if(bcir_hydrate(&f,&p,out,sizeof out,&n)!=BCIR_ERR_WIDTH||n||!filled(out,sizeof out))return 3;
  s.width=8;
  if(bcir_hydrate(&f,&p,out,sizeof out,&n)!=BCIR_OK||!n)return 4;
  if(bcir_sp_verify_semantic(out,n,UINT32_MAX,UINT32_MAX)!=BCIR_OK)return 5;
  return 0;
}
"""
    with tempfile.TemporaryDirectory() as d:
        driver = os.path.join(d, "hydrate_preflight.c")
        with open(driver, "w", encoding="utf-8") as f:
            f.write(source)
        exe = os.path.join(d, "hydrate_preflight")
        build = subprocess.run(
            [
                clang,
                "-std=c11",
                "-O1",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                _C_DIR,
                os.path.join(_C_DIR, "bcir_runtime.c"),
                os.path.join(_C_DIR, "bcir_hydrate.c"),
                driver,
                "-o",
                exe,
            ],
            capture_output=True,
            text=True,
        )
        assert build.returncode == 0, build.stderr
        run = subprocess.run([exe], capture_output=True, text=True)
        assert run.returncode == 0, (run.returncode, run.stdout, run.stderr)


def test_c_lifetime_verifier_tracks_more_than_256_freed_resources():
    """R21 must not silently forget a freed RID when a function crosses the old fixed cap."""
    clang = which("clang")
    if clang is None:
        return
    source = r"""
#include <string.h>
#include "bcir_verify.h"
static int reports;
static void report(const char *fn,const char *kind,void *ctx){
  (void)fn;(void)ctx;if(!strcmp(kind,"use-after-free"))reports++;
}
int main(void){
  static bcir_claim claims[258]; bcir_func f; bcir_unit u;
  memset(&f,0,sizeof f);memset(&u,0,sizeof u);strcpy(f.name,"many_frees");
  for(int i=0;i<257;i++){
    claims[i].id=(unsigned)i+1u;claims[i].lifetime=2;claims[i].n_rd=1;
    claims[i].rd[0]=(unsigned)i+1u;strcpy(claims[i].op,"c.free");
  }
  claims[257].id=258;claims[257].n_rd=1;claims[257].rd[0]=257;
  strcpy(claims[257].op,"c.load");f.claims=claims;f.n_claims=258;
  u.funcs=&f;u.n_funcs=1;bcir_verify_lifetime(&u,report,0);
  bcir_verify_lifetime(0,report,0);bcir_verify_lifetime(&u,0,0);
  return reports==1?0:1;
}
"""
    with tempfile.TemporaryDirectory() as d:
        driver = os.path.join(d, "lifetime_many.c")
        with open(driver, "w", encoding="utf-8") as f:
            f.write(source)
        exe = os.path.join(d, "lifetime_many")
        build = subprocess.run(
            [
                clang,
                "-std=c11",
                "-O1",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                _C_DIR,
                os.path.join(_C_DIR, "bcir_runtime.c"),
                os.path.join(_C_DIR, "bcir_verify.c"),
                driver,
                "-o",
                exe,
            ],
            capture_output=True,
            text=True,
        )
        assert build.returncode == 0, build.stderr
        run = subprocess.run([exe], capture_output=True, text=True)
        assert run.returncode == 0, (run.stdout, run.stderr)


def test_c_planner_width_contract_and_r9_rederives_costs():
    """The scalar planner used to write a claim's ELEMENT COUNT into the step's lane width, so
    a claim over 3 elements planned at "width 3" -- a width the hydrator rightly refuses -- and
    the C R9 trusted whatever cost a step carried. Now: the planner emits width 1 and
    cost = base * n; R9 re-derives every cost from the claim and requires a power-of-two
    width; the planner and the hydrator compose on any count."""
    clang = which("clang")
    if clang is None:
        return
    source = r"""
#include <stdint.h>
#include <string.h>
#include "bcir_plan.h"
#include "bcir_verify.h"
#include "bcir_hydrate.h"
int main(void){
  bcir_claim c; bcir_func f; bcir_plan_step s; bcir_plan p; char d[160]; uint8_t out[512]; size_t n=0;
  memset(&c,0,sizeof c); memset(&f,0,sizeof f);
  c.id=7; c.opcode=BCIR_OP_MUL; c.lane=BCIR_LANE_U; c.count=3; strcpy(c.op,"c.bin.mul");
  f.claims=&c; f.n_claims=1;
  if(bcir_plan_func(&f,&s,1,&p)!=BCIR_OK)return 1;
  if(s.width!=1u||s.cost!=bcir_plan_base_cost(&c)*3u||p.total_cost!=s.cost)return 2;   /* the contract */
  if(!bcir_verify_plan(&f,&p,d,sizeof d))return 3;                                       /* R9 admits it */
  if(bcir_hydrate(&f,&p,out,sizeof out,&n)!=BCIR_OK||!n)return 4;                      /* and it hydrates */
  s.cost+=1; p.total_cost+=1;                                                            /* forged cost, sum intact */
  if(bcir_verify_plan(&f,&p,d,sizeof d)||!strstr(d,"does not re-derive"))return 5;
  s.cost-=1; p.total_cost-=1; s.width=3;                                                 /* a width no lane issues */
  if(bcir_verify_plan(&f,&p,d,sizeof d)||!strstr(d,"power of two"))return 6;
  s.width=4; s.cost=bcir_plan_base_cost(&c); p.total_cost=s.cost;                        /* 3 elements in one 4-lane issue */
  if(!bcir_verify_plan(&f,&p,d,sizeof d))return 7;
  n=0; if(bcir_hydrate(&f,&p,out,sizeof out,&n)!=BCIR_OK||!n)return 8;
  n=0; if(bcir_hydrate(&f,NULL,out,sizeof out,&n)!=BCIR_OK||!n)return 9;                /* no plan: scalar, any count */
  return 0;
}
"""
    with tempfile.TemporaryDirectory() as d:
        driver = os.path.join(d, "plan_contract.c")
        with open(driver, "w", encoding="utf-8") as fh:
            fh.write(source)
        exe = os.path.join(d, "plan_contract")
        build = subprocess.run(
            [
                clang,
                "-std=c11",
                "-O1",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                _C_DIR,
                os.path.join(_C_DIR, "bcir_runtime.c"),
                os.path.join(_C_DIR, "bcir_plan.c"),
                os.path.join(_C_DIR, "bcir_verify.c"),
                os.path.join(_C_DIR, "bcir_hydrate.c"),
                driver,
                "-o",
                exe,
            ],
            capture_output=True,
            text=True,
        )
        assert build.returncode == 0, build.stderr
        run = subprocess.run([exe], capture_output=True, text=True)
        assert run.returncode == 0, (run.returncode, run.stdout, run.stderr)
