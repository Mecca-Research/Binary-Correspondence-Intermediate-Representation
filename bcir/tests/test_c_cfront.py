"""Python<->C dual-rail parity + behaviour-equivalence for the plug-in C compiler
(`runtime/c/bcir_cfront.c`).

The C frontend is the production port of the Python prototype (`bcir/frontends/cfront/`). For each
shared fixture this gate checks the C rail against the six artifacts:
  * the lowered claim graph — its RID-independent structural summary equals the oracle's (parity);
  * the R1-R8 + R18 verifier checkpoint (`ok=1`, and R18 rejects recursion / undefined callees);
  * the faithful emitted C — compiled beside the original source and run on seeded-random inputs, it
    is behaviour-equivalent under Clang.
Toolchain-gated (builds `bcir_cfront.c`): self-skips in the quick tier, runs under c-runtime/thorough.
"""

import os
import re
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront import compile_unit
from bcir.model import Domain

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_CC = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
_FIXTURES = ["cfront_regmap.c", "cfront_array.c", "cfront_callgraph.c"]


def _oracle(src: str):
    r = compile_unit(src, check_clang=False)
    funcs = r.lowered.functions
    entry = funcs[next(reversed(funcs))]
    cl = entry.claims
    mmio = sum(1 for c in cl if c.op == "c.load" and c.domain == Domain.MMIO)
    bf = sum(1 for c in cl if c.op == "c.bf.get")
    kn = sum(1 for c in cl if c.op == "c.const")
    bo = sum(1 for c in cl if c.op.startswith("c.bin."))
    ca = sum(1 for c in cl if c.op.startswith("c.call:"))
    summary = (f"funcs={len(funcs)} claims={len(cl)} mmio={mmio} bf={bf} const={kn} "
               f"binop={bo} call={ca} ok={1 if r.is_clean else 0}")
    return summary, r, entry


def _build_frontend(d: str) -> str:
    exe = os.path.join(d, "tcf")
    for std in ("c23", "c11"):
        b = subprocess.run([_CC, f"-std={std}", "-O2", "-I", _C,
                            os.path.join(_C, "bcir_cfront.c"), os.path.join(_C, "test_cfront.c"),
                            "-o", exe], capture_output=True, text=True)
        if b.returncode == 0:
            return exe
    raise AssertionError(f"C frontend build failed:\n{b.stderr}")


def _c_run(exe: str, fixture_path: str):
    out = subprocess.run([exe, fixture_path], capture_output=True, text=True).stdout
    summary, _, emit = out.partition("----EMIT----\n")
    return summary.strip().splitlines()[0], emit


def _cname(ct) -> str:
    if ct.kind == "pointer":
        return _cname(ct.of) + " *"
    if ct.kind == "array":
        return _cname(ct.of)
    if ct.is_aggregate:
        return f"{ct.kind} {ct.name}"
    return ct.name


def _equiv(source: str, c_emitted: str, entry) -> str:
    """Compile the original source beside the C-frontend's emitted bcir_* and diff outputs."""
    has_ptr = any(ct.kind in ("pointer", "array") for _n, _r, ct in entry.params)
    decls, setup, args = [], [], []
    for i, (_pn, _rid, ct) in enumerate(entry.params):
        if ct.kind in ("pointer", "array"):
            decls.append(f"  static {_cname(ct.of)} buf{i}[256];")
            setup.append(f"    for(unsigned k=0;k<sizeof buf{i}/4;k++) ((uint32_t*)buf{i})[k]=rng();")
            args.append(f"buf{i}")
        elif ct.is_aggregate:
            decls.append(f"  {ct.kind} {ct.name} a{i};")
            inits = "".join(f"    a{i}.{fn}=(rng()&{(1 << bw) - 1}u);\n" if bw
                            else f"    a{i}.{fn}=({_cname(ft)})rng();\n"
                            for fn, ft, _bo, _bf, bw in ct.fields)
            setup.append(inits.rstrip("\n"))
            args.append(f"a{i}")
        else:
            decls.append(f"  {_cname(ct)} s{i};")
            setup.append(f"    s{i}=({_cname(ct)})(rng()%{200 if has_ptr else 4000000000});")
            args.append(f"s{i}")
    call = ", ".join(args)
    rt = _cname(entry.ret_type)
    if entry.ret_type.is_aggregate:
        cmp = (f"    {rt} ra={entry.name}({call}), rb=bcir_{entry.name}({call});\n"
               f"    if(memcmp(&ra,&rb,sizeof ra)){{printf(\"MISMATCH@%d\\n\",i);return 1;}}")
    else:
        cmp = (f"    if({entry.name}({call})!=bcir_{entry.name}({call}))"
               f"{{printf(\"MISMATCH@%d\\n\",i);return 1;}}")
    harness = f"""#include <stdint.h>
#include <stdio.h>
#include <string.h>
{source}

{c_emitted}
static uint64_t S=0x9E3779B97F4A7C15u;
static uint32_t rng(void){{S=S*6364136223846793005u+1442695040888963407u;return (uint32_t)(S>>32);}}
int main(void){{
{chr(10).join(decls)}
  for(int i=0;i<256;i++){{
{chr(10).join(setup)}
{cmp}
  }}
  printf("MATCH\\n");return 0;}}"""
    with tempfile.TemporaryDirectory() as d:
        c, e = os.path.join(d, "e.c"), os.path.join(d, "e")
        open(c, "w").write(harness)
        for std in ("c23", "c2x", "c17"):
            b = subprocess.run([_CC, f"-std={std}", "-O2", c, "-o", e], capture_output=True, text=True)
            if b.returncode == 0:
                break
        else:
            return f"build-failed:{b.stderr.strip().splitlines()[-1] if b.stderr else '?'}"
        return subprocess.run([e], capture_output=True, text=True).stdout.strip()


def test_python_c_parity_and_equivalence_across_fixtures():
    if not _CC:
        # quick tier: still validate the oracle side computes the summaries.
        for fx in _FIXTURES:
            s, _, _ = _oracle(open(os.path.join(_C, fx), encoding="utf-8").read())
            assert "ok=1" in s
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        for fx in _FIXTURES:
            path = os.path.join(_C, fx)
            src = open(path, encoding="utf-8").read()
            oracle_summary, _r, entry = _oracle(src)
            c_summary, c_emit = _c_run(exe, path)
            assert c_summary == oracle_summary, f"{fx}: parity diverged\n C: {c_summary}\nPY: {oracle_summary}"
            assert _equiv(src, c_emit, entry) == "MATCH", f"{fx}: emitted C not behaviour-equivalent"


def _build_loop(d: str) -> str:
    exe = os.path.join(d, "loop")
    srcs = [os.path.join(_C, s) for s in ("bcir_cfront.c", "bcir_plan.c", "bcir_hydrate.c",
                                          "bcir_exec.c", "bcir_runtime.c", "test_cfront_loop.c")]
    for std in ("c23", "c11"):
        b = subprocess.run([_CC, f"-std={std}", "-O2", "-I", _C, *srcs, "-o", exe],
                           capture_output=True, text=True)
        if b.returncode == 0:
            return exe
    raise AssertionError(f"loop build failed:\n{b.stderr}")


def test_full_compile_execute_loop_in_c():
    """C source -> bcir_cfront -> bcir_plan -> bcir_hydrate -> bcir_exec, entirely in C: the
    hydrated StreamPack is valid and the executor runs every claim in lowering order."""
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        loop = _build_loop(d)
        for fx in _FIXTURES:
            path = os.path.join(_C, fx)
            _summary, _r, entry = _oracle(open(path, encoding="utf-8").read())
            out = subprocess.run([loop, path], capture_output=True, text=True).stdout.strip()
            assert out.startswith("loop:"), out
            m = dict(re.findall(r"(\w+)=([0-9]+)", out))
            # the loop executes exactly the entry's claims (parity-identical to the oracle's count)...
            assert int(m["executed"]) == int(m["claims"]) == len(entry.claims), f"{fx}: {out}"
            assert int(m["plan_cost"]) > 0 and int(m["pack_bytes"]) > 64                # a real plan + a real pack
            order = out.split("order=")[1].split(",")
            assert order == sorted(order, key=int)                       # deterministic lowering order


def test_c_frontend_builds_warning_clean():
    if not _CC:
        return
    for std in ("c23", "c11"):
        b = subprocess.run([_CC, f"-std={std}", "-Wall", "-Wextra", "-Werror", "-I", _C, "-c",
                            os.path.join(_C, "bcir_cfront.c"), "-o", os.devnull],
                           capture_output=True, text=True)
        if b.returncode == 0:
            return
    raise AssertionError(f"C frontend has warnings:\n{b.stderr}")


def test_c_frontend_R18_rejects_recursion_and_undefined_callee():
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        for src, needle in [
                ("uint32_t f(uint32_t n){ return f(n-1); }\nuint32_t g(uint32_t n){ return f(n); }\n",
                 "recursive"),
                ("uint32_t g(uint32_t a){ return missing(a); }\n", "undefined")]:
            fx = os.path.join(d, "bad.c")
            open(fx, "w").write(src)
            out = subprocess.run([exe, fx], capture_output=True, text=True).stdout
            assert "ok=0" in out and "R18" in out and needle in out, out
