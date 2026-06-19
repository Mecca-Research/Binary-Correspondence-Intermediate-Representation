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
import sys
import tempfile

from bcir.frontends.cfront import compile_unit
from bcir.model import Domain

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_CC = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
# straight-line fixtures run the full execute loop; control-flow fixtures get parity + emit + Clang ≡
# (control flow is not a flat StreamPack segment stream, so the loop runs the straight-line set).
_STRAIGHTLINE = ["cfront_regmap.c", "cfront_array.c", "cfront_array2d.c", "cfront_deref.c",
                 "cfront_callgraph.c", "cfront_typedef.c", "cfront_enum.c", "cfront_ternary.c",
                 "cfront_sizeof.c", "cfront_strsizeof.c", "cfront_cast.c", "cfront_alignof.c", "cfront_static.c",
                 "cfront_global.c", "cfront_compound.c", "cfront_logic.c"]   # + const LUT + compound assignment
_CONTROL = ["cfront_branch.c", "cfront_while.c", "cfront_for.c", "cfront_dowhile.c",
            "cfront_continue.c", "cfront_switch.c", "cfront_goto.c", "cfront_incdec.c"]
_PREPROC = ["cfront_macros.c", "cfront_ppinc.c"]      # L7: exercise the preprocessor
_ABI = ["cfront_structret.c", "cfront_packed.c",      # L8: struct return-by-value + packed layout
        "cfront_union.c",                             # + full union (members overlap at offset 0)
        "cfront_interleave.c",                        # + enum/struct defined *between* two functions
        "cfront_funcptr.c",                           # + funcptr param + indirect call (HAL dispatch)
        "cfront_rmw.c",                               # + MMIO register read-modify-write (d->reg |= bits)
        "cfront_bitfield.c",                          # + MMIO bitfield write (r->field = v, c.bf.set)
        "cfront_bfcompound.c"]                         # + bitfield compound-assign (r->field |= bits)
_FIXTURES = _STRAIGHTLINE + _CONTROL + _PREPROC + _ABI
# §5.8 atomics/fences/CAS run their own gate: their memory side effects make the generic
# pure-function equivalence harness invalid (it would call the original first and observe
# the mutated cell), so they get a side-effect-aware behaviour check below.
_ATOMIC = ["cfront_atomic.c", "cfront_cmpxchg.c", "cfront_atomic11.c", "cfront_atomic_xchg.c"]  # + C11 stdatomic + atomic_exchange


def _includes_for(fx: str) -> dict:
    """The `#include "..."` header map the oracle needs (the C frontend reads the sibling files
    directly; the oracle is given their contents). Auto-resolved from the source, so a new fixture
    that includes a header in runtime/c/ needs no per-file case."""
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    inc = {}
    for h in re.findall(r'#include\s+"([^"]+)"', src):
        p = os.path.join(_C, h)
        if os.path.exists(p):
            inc[h] = open(p, encoding="utf-8").read()
    return inc


def _oracle(src: str, includes=None):
    r = compile_unit(src, check_clang=False, includes=includes)
    funcs = r.lowered.functions
    entry = funcs[next(reversed(funcs))]
    cl = entry.claims
    mmio = sum(1 for c in cl if c.op == "c.load" and c.domain == Domain.MMIO)
    bf = sum(1 for c in cl if c.op == "c.bf.get")
    kn = sum(1 for c in cl if c.op == "c.const")
    bo = sum(1 for c in cl if c.op.startswith("c.bin."))
    ca = sum(1 for c in cl if c.op.startswith("c.call"))   # c.call:NAME (direct) + c.call.indirect
    summary = (f"funcs={len(funcs)} claims={len(cl)} mmio={mmio} bf={bf} const={kn} "
               f"binop={bo} call={ca} ok={1 if r.is_clean else 0}")
    return summary, r, entry


def _build_frontend(d: str) -> str:
    exe = os.path.join(d, "tcf")
    # bcir_cfront verifies (bcir_verify.c -> bcir_verify_unit / bcir_provenance_digest) and the
    # pack law reaches into bcir_runtime.c (bcir_sp_validate), so both are linked in.
    srcs = [os.path.join(_C, s) for s in ("bcir_cfront.c", "bcir_cpp.c", "bcir_verify.c",
            "bcir_runtime.c", "test_cfront.c")]
    for std in ("c23", "c11"):
        b = subprocess.run([_CC, f"-std={std}", "-O2", "-I", _C, *srcs, "-o", exe],
                           capture_output=True, text=True)
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
    return ("_Atomic " if getattr(ct, "atomic", False) else "") + ct.name


def _equiv(source: str, c_emitted: str, entry) -> str:
    """Compile the original source beside the C-frontend's emitted bcir_* and diff outputs."""
    has_ptr = any(ct.kind in ("pointer", "array") for _n, _r, ct in entry.params)
    decls, setup, args, prelude = [], [], [], []
    for i, (_pn, _rid, ct) in enumerate(entry.params):
        if ct.kind == "funcptr":                          # pass a real (deterministic) target fn
            rety = _cname(ct.of) if ct.of else "uint32_t"
            plist = ", ".join(f"{_cname(pt)} p{j}" for j, pt in enumerate(ct.params)) or "void"
            comb = " + ".join(f"(p{j} * {2 * j + 1}u)" for j in range(len(ct.params))) or "1u"
            prelude.append(f"static {rety} _fp{i}({plist}){{ return ({rety})({comb}); }}")
            args.append(f"_fp{i}")
            continue
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
#include <stdatomic.h>
{source}

{c_emitted}
{chr(10).join(prelude)}
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


def _equiv_atomic(source: str, c_emitted: str, entry) -> str:
    """Side-effect-aware behaviour equivalence for atomics. The generic `_equiv` calls the
    original then the emitted bcir_* on the *same* buffer -- invalid here, since an atomic RMW
    mutates its pointee, so the second call would start from a counter the first already moved.
    Instead this runs each on an independent copy of the *same* seeded state and compares both the
    return value and the final memory state (an atomic counter is a single location, not an array)."""
    # Seed from a small range so a compare-and-swap's expected value collides with the cell
    # often enough to exercise the swap-taken path (not just the no-op path); equivalence holds
    # for any inputs, but this makes the behaviour check meaningful for CAS.
    decls, setup, args_a, args_b, cell_cmp = [], [], [], [], []
    for i, (_pn, _rid, ct) in enumerate(entry.params):
        if ct.kind in ("pointer", "array"):
            base = _cname(ct.of)                            # may be `_Atomic uint32_t` (a C11 cell)
            plain = base.replace("_Atomic ", "")            # the seed casts to the non-atomic type
            decls += [f"  {base} ca{i};", f"  {base} cb{i};"]
            setup.append(f"    ca{i}=cb{i}=({plain})(rng()%16);")
            args_a.append(f"&ca{i}")
            args_b.append(f"&cb{i}")
            cell_cmp.append(f"ca{i}!=cb{i}")
        else:
            decls.append(f"  {_cname(ct)} s{i};")
            setup.append(f"    s{i}=({_cname(ct)})(rng()%16);")
            args_a.append(f"s{i}")
            args_b.append(f"s{i}")
    rt = _cname(entry.ret_type)
    cells = (" || " + " || ".join(cell_cmp)) if cell_cmp else ""
    harness = f"""#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdatomic.h>
{source}

{c_emitted}
static uint64_t S=0x9E3779B97F4A7C15u;
static uint32_t rng(void){{S=S*6364136223846793005u+1442695040888963407u;return (uint32_t)(S>>32);}}
int main(void){{
{chr(10).join(decls)}
  for(int i=0;i<256;i++){{
{chr(10).join(setup)}
    {rt} ra={entry.name}({", ".join(args_a)});
    {rt} rb=bcir_{entry.name}({", ".join(args_b)});
    if(ra!=rb{cells}){{printf("MISMATCH@%d\\n",i);return 1;}}
  }}
  printf("MATCH\\n");return 0;}}"""
    with tempfile.TemporaryDirectory() as d:
        c, e = os.path.join(d, "a.c"), os.path.join(d, "a")
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
            s, _, _ = _oracle(open(os.path.join(_C, fx), encoding="utf-8").read(), _includes_for(fx))
            assert "ok=1" in s
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        for fx in _FIXTURES:
            path = os.path.join(_C, fx)
            src = open(path, encoding="utf-8").read()
            oracle_summary, r, entry = _oracle(src, _includes_for(fx))
            c_summary, c_emit = _c_run(exe, path)
            assert c_summary == oracle_summary, f"{fx}: parity diverged\n C: {c_summary}\nPY: {oracle_summary}"
            # equivalence uses the PREPROCESSED source (r.source) so Clang needs no #include.
            assert _equiv(r.source, c_emit, entry) == "MATCH", f"{fx}: emitted C not behaviour-equivalent"


def _build_loop(d: str) -> str:
    exe = os.path.join(d, "loop")
    srcs = [os.path.join(_C, s) for s in ("bcir_cfront.c", "bcir_cpp.c", "bcir_plan.c",
            "bcir_hydrate.c", "bcir_exec.c", "bcir_runtime.c", "bcir_verify.c",
            "test_cfront_loop.c")]
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
        for fx in _STRAIGHTLINE:
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


# the atomic builtins each fixture must emit back (the faithful-emit artifact, not a scalar fallback).
_ATOMIC_EMITS = {
    "cfront_atomic.c": ["__atomic_fetch_", "__atomic_thread_fence", "__ATOMIC_SEQ_CST"],
    "cfront_cmpxchg.c": ["__sync_val_compare_and_swap", "__sync_bool_compare_and_swap"],
    "cfront_atomic11.c": ["_Atomic uint32_t *", "atomic_fetch_add", "atomic_fetch_xor", "atomic_load"],
    "cfront_atomic_xchg.c": ["_Atomic uint32_t *", "atomic_exchange", "atomic_load"],
}


def test_atomic_fence_dual_rail_parity_and_behaviour():
    """§5.8 atomics/fences/CAS: __atomic_fetch_add/sub/xor -> ATOMIC_ADD/SUB/XOR,
    __atomic_thread_fence/__sync_synchronize -> BARRIER, and __sync_{val,bool}_compare_and_swap ->
    CMPXCHG (a 3-read claim: ptr, expected, desired) all lower on lane A (R6 admits lane A for a
    scalar atomic; R5 demands the atomic/barriered hazard), pass R1-R8 + R18, emit the matching
    builtins, and -- run on independent copies of the same seeded cell -- are behaviour-equivalent
    under Clang. The full C compile->execute loop hydrates and executes every atomic claim with
    R9/R10-R11 clean."""
    for fx in _ATOMIC:                       # quick tier: the oracle accepts the atomic fixtures.
        s, _, _ = _oracle(open(os.path.join(_C, fx), encoding="utf-8").read())
        assert "ok=1" in s, f"{fx}: oracle rejects atomics: {s}"
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        loop = _build_loop(d)
        for fx in _ATOMIC:
            path = os.path.join(_C, fx)
            src = open(path, encoding="utf-8").read()
            oracle_summary, r, entry = _oracle(src)
            c_summary, c_emit = _c_run(exe, path)
            assert c_summary == oracle_summary, f"{fx}: parity diverged\n C: {c_summary}\nPY: {oracle_summary}"
            assert "ok=1" in c_summary, c_summary
            # the emitted C carries the real atomic builtins, not a scalar fallback.
            for needle in _ATOMIC_EMITS[fx]:
                assert needle in c_emit, f"{fx}: emit missing {needle}\n{c_emit}"
            assert _equiv_atomic(r.source, c_emit, entry) == "MATCH", f"{fx}: not behaviour-equivalent"
            # the full C compile->execute loop: every atomic claim hydrates + executes, R9/R10-R11 clean.
            out = subprocess.run([loop, path], capture_output=True, text=True).stdout.strip()
            assert out.startswith("loop:"), out
            m = dict(re.findall(r"(\w+)=([0-9]+)", out))
            assert int(m["executed"]) == int(m["claims"]) == len(entry.claims), out
            assert m["r9"] == "1" and m["r10r11"] == "1", out


def test_funcptr_member_dispatch_table():
    """Function-pointer struct members (HAL dispatch table): `o->fn(args)` fuses into one
    `c.call.imember:<field>` claim (reads: the struct base, then the actuals), emitted verbatim as
    `o->fn(args)` -- so no 8-byte function-pointer value rides in the 4-byte value model -- and R18
    leaves it an opaque external edge. Oracle<->C parity + a bespoke behaviour harness: the generic
    `_equiv` fills a pointee with seeded rng, which for a struct of function pointers would be
    invalid call targets, so this builds the struct with one real (deterministic) target per member
    and checks `run(&o,...)` == `bcir_run(&o,...)` over seeded inputs."""
    fx = "cfront_dispatch.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary and "call=2" in oracle_summary, oracle_summary
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        assert "o->add2(" in c_emit and "o->mul2(" in c_emit, c_emit       # faithful member-call emit
        struct_ct = entry.params[0][2].of                                  # the pointed-to ops struct
        helpers, inits = [], []
        for fi, (fname, ftype, *_rest) in enumerate(struct_ct.fields):
            rety = _cname(ftype.of) if ftype.of else "uint32_t"
            plist = ", ".join(f"{_cname(pt)} p{j}" for j, pt in enumerate(ftype.params)) or "void"
            comb = " + ".join(f"(p{j} * {2 * j + 3}u)" for j in range(len(ftype.params))) or "1u"
            helpers.append(f"static {rety} _hm{fi}({plist}){{ return ({rety})({comb}); }}")
            inits.append(f"  obj.{fname} = _hm{fi};")
        scalars = [f"s{i}" for i in range(1, len(entry.params))]
        call = ", ".join(["&obj", *scalars])
        harness = f"""#include <stdint.h>
#include <stdio.h>
{r.source}

{c_emit}
{chr(10).join(helpers)}
static uint64_t S=0x9E3779B97F4A7C15u;
static uint32_t rng(void){{S=S*6364136223846793005u+1442695040888963407u;return (uint32_t)(S>>32);}}
int main(void){{
  {struct_ct.kind} {struct_ct.name} obj;
{chr(10).join(inits)}
  for(int i=0;i<256;i++){{
    {''.join(f'uint32_t {s}=rng(); ' for s in scalars)}
    if({entry.name}({call})!=bcir_{entry.name}({call})){{printf("MISMATCH@%d",i);return 1;}}
  }}
  printf("MATCH");return 0;}}"""
        c, e = os.path.join(d, "disp.c"), os.path.join(d, "disp")
        open(c, "w").write(harness)
        for std in ("c23", "c2x", "c17"):
            b = subprocess.run([_CC, f"-std={std}", "-O2", c, "-o", e], capture_output=True, text=True)
            if b.returncode == 0:
                break
        else:
            raise AssertionError(f"dispatch harness build failed:\n{b.stderr}")
        out = subprocess.run([e], capture_output=True, text=True).stdout.strip()
        assert out == "MATCH", f"{fx}: dispatch-table emit not behaviour-equivalent ({out})"


def test_integration_driver_composes_phase2_surface():
    """Integration: a realistic multi-feature driver (`cfront_integration.c`) ingested with no
    hand-written claim graph, exercising the Phase-2 surface *together* -- typedef + enum + an MMIO
    register-map struct (L5 volatile), a `switch` over an enum status, a `static` fault counter, a
    `goto` cleanup path, integer casts, a 2D bank lookup, and an inter-procedural call graph (L4 /
    R18). The two rails agree on the entry's structural summary, and *every* function is
    Clang-behaviour-equivalent -- the proof the features compose, not just pass in isolation."""
    fx = "cfront_integration.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    assert len(r.lowered.functions) == 3                      # decode_state, bank_lookup, sensor_read
    # the entry reads two MMIO registers (status + sample) and makes one resolved call (decode_state).
    assert "mmio=2" in oracle_summary and "call=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        # the emit carries each composed feature in one verified-C unit.
        for needle in ("volatile uint32_t *", "goto done", "static uint32_t faults",
                       "(uint16_t)", "bcir_decode_state(", "BCIR verified-C attestation"):
            assert needle in c_emit, f"{fx}: emit missing {needle!r}"
        # every function (the switch decode, the 2D+cast lookup, the MMIO/static/goto entry) is
        # behaviour-equivalent to the original under Clang.
        for name, lf in r.lowered.functions.items():
            assert _equiv(r.source, c_emit, lf) == "MATCH", f"{fx}:{name} not behaviour-equivalent"


def test_register_driver_composes_register_map_surface():
    """Register-map composition checkpoint: a realistic device driver (`cfront_regdriver.c`) ingested
    with no hand-written claim graph, exercising the whole register surface *together* -- a `switch`
    over a status field, a multi-bit bitfield write (`dev->mode = ...`), a bitfield read
    (`dev->prio`), a register read-modify-write (`dev->ctrl |= ...`), a file-scope lookup table, an
    `enum`, and a `static` persistent counter. The two rails agree on the structural summary and the
    emit is Clang-behaviour-equivalent -- the proof the register-map features (PRs #294-#297) compose,
    not just pass in isolation. The function mutates its device, but every mutation is idempotent (a
    bitfield set to the same value, an `|=` of the same bit, a `static` evolving in lockstep) and the
    branched-on status field is never written, so the generic shared-buffer harness stays valid."""
    fx = "cfront_regdriver.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    # a real register driver: a volatile MMIO read + a bitfield read, no hand-written claim graph.
    assert "mmio=5" in oracle_summary and "bf=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        # the emit carries each composed register feature in one verified-C unit.
        for needle in ("volatile uint32_t *", "QUANTA[", "static uint32_t halts",
                       "} else {", "BCIR verified-C attestation"):
            assert needle in c_emit, f"{fx}: emit missing {needle!r}"
        assert _equiv(r.source, c_emit, entry) == "MATCH", f"{fx}: emit not behaviour-equivalent"


def test_c2_attestation_in_emitted_c():
    """C.2: the emitted verified-C carries the attestation header naming the discharged laws and
    the R13 provenance digest -- and that digest is the same one the compile->execute loop reports
    (the manifest is reproducible across the two C entry points)."""
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        loop = _build_loop(d)
        path = os.path.join(_C, "cfront_regmap.c")
        _c_summary, c_emit = _c_run(exe, path)
        assert "BCIR verified-C attestation (C.2)" in c_emit
        assert "R1-R8 + R18" in c_emit and "R13 provenance digest" in c_emit
        m = re.search(r"R13 provenance digest\s+([0-9a-f]{16})", c_emit)
        assert m, c_emit
        out = subprocess.run([loop, path], capture_output=True, text=True).stdout.strip()
        prov = re.search(r"prov=([0-9a-f]{16})", out)
        assert prov and prov.group(1) == m.group(1), f"digest mismatch: emit={m.group(1)} loop={prov}"


def test_phase_d_real_header_driver_end_to_end():
    """Phase D: a real vendor-style register-map header (`cfront_driver.h`) + driver
    (`cfront_driver.c`) ingested END-TO-END by the plug-in C compiler with NO hand-written claim
    graph -- `#include` + field macros (L7), typedef/enum/union/bitfields (the type model), volatile
    MMIO loads (L5), struct pointers (L3), and the call graph (L4 / R18) in one driver. The six
    artifacts on the C rail: oracle<->C structural parity, the R1-R18 verdict, the faithful (Clang
    behaviour-equivalent) emit, and the full `C -> bcir_cpp -> bcir_cfront -> bcir_plan ->
    bcir_hydrate -> bcir_exec` loop with R9/R10-R11 clean."""
    inc = {"cfront_driver.h": open(os.path.join(_C, "cfront_driver.h"), encoding="utf-8").read()}
    src = open(os.path.join(_C, "cfront_driver.c"), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src, inc)
    assert "ok=1" in oracle_summary, oracle_summary
    # a real driver, not a toy: a memory-mapped read + bitfield decode + a call graph.
    assert "mmio=1" in oracle_summary and "bf=3" in oracle_summary and "call=2" in oracle_summary, oracle_summary
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        loop = _build_loop(d)
        path = os.path.join(_C, "cfront_driver.c")
        c_summary, c_emit = _c_run(exe, path)
        assert c_summary == oracle_summary, f"Phase D parity diverged\n C: {c_summary}\nPY: {oracle_summary}"
        # the emitted verified-C carries the device semantics + the C.2 attestation.
        assert "volatile uint32_t *" in c_emit and "BCIR verified-C attestation" in c_emit, c_emit
        # behaviour-equivalent against the original header+driver under Clang (r.source is preprocessed).
        assert _equiv(r.source, c_emit, entry) == "MATCH", "Phase D emit not behaviour-equivalent"
        # the full C compile->execute loop runs the real driver, every claim, R9/R10-R11 clean.
        out = subprocess.run([loop, path], capture_output=True, text=True).stdout.strip()
        assert out.startswith("loop:"), out
        m = dict(re.findall(r"(\w+)=([0-9]+)", out))
        assert int(m["executed"]) == int(m["claims"]) == len(entry.claims), out
        assert m["r9"] == "1" and m["r10r11"] == "1", out


def _cli(args, cwd=None):
    """Invoke the bcir-cfront driver CLI; return (rc, stdout, stderr)."""
    p = subprocess.run([sys.executable, "-m", "bcir.frontends.cfront", *args],
                       capture_output=True, text=True, cwd=cwd or _ROOT)
    return p.returncode, p.stdout, p.stderr


def test_cli_resolves_sibling_and_search_path_headers():
    """The frontend CLI must compile a file with sibling/`-I` headers DIRECTLY -- the productization
    gap a replacement compiler can't have. (`compile_unit(f.read())` used to pass no include context,
    so `#include "uart_regs.h"` failed even though the tests/check_runtime.sh supplied the map.)"""
    # (1) sibling header: the file's own directory is on the search path automatically.
    rc, out, err = _cli(["runtime/c/cfront_driver_uart.c"])
    assert rc == 0, f"sibling-header compile failed: {err}\n{out}"
    assert "R1-R18: CLEAN" in out and "uart_configure" in out, out
    # (2) -E preprocesses the #include + object macros (the struct + a bit position survive).
    rc, out, _ = _cli(["-E", "runtime/c/cfront_driver_uart.c"])
    assert rc == 0 and "struct uart_regs" in out and "uart_regs_t" in out, out
    with tempfile.TemporaryDirectory() as d:
        # (3) -I <dir>: a header outside the source directory resolves via the search path.
        incd = os.path.join(d, "inc"); os.makedirs(incd)
        with open(os.path.join(incd, "regs.h"), "w") as f:
            f.write("typedef volatile unsigned int reg32;\nstruct dev { reg32 r; };\n")
        src = os.path.join(d, "drv.c")
        with open(src, "w") as f:
            f.write('#include "regs.h"\nunsigned int rd(volatile struct dev *p){ return p->r; }\n')
        rc, out, err = _cli(["-I", incd, src])
        assert rc == 0, f"-I compile failed: {err}\n{out}"
        assert "R1-R18: CLEAN" in out, out
        # (4) a missing header is a clean diagnostic + non-zero exit (not a crash).
        rc, out, err = _cli([src])                     # no -I -> regs.h unresolved
        assert rc != 0 and "not found" in (out + err), (out, err)
    # (5) -D predefines a macro used in a #if.
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "g.c")
        with open(src, "w") as f:
            f.write("#if defined(WIDE)\nunsigned int w(unsigned int x){return x+1;}\n"
                    "#else\nunsigned int w(unsigned int x){return x;}\n#endif\n")
        rc, out, _ = _cli(["-E", "-D", "WIDE", src])
        assert rc == 0 and "x+1" in out.replace(" ", ""), out


def _build_bcir_cc(d: str) -> str:
    exe = os.path.join(d, "bcir-cc")
    srcs = [os.path.join(_C, s) for s in ("bcir_cc.c", "bcir_cpp.c", "bcir_cfront.c", "bcir_verify.c",
            "bcir_runtime.c", "bcir_plan.c", "bcir_hydrate.c")]
    for std in ("c23", "c11"):
        b = subprocess.run([_CC, f"-std={std}", "-O2", "-I", _C, *srcs, "-o", exe],
                           capture_output=True, text=True)
        if b.returncode == 0:
            return exe
    raise AssertionError(f"bcir-cc build failed:\n{b.stderr}")


def test_file_macro_real_path_dual_rail():
    """__FILE__ reflects the actual source path the driver was given (not the "<source>" default),
    byte-identically on both rails: the C `bcir-cc -E` driver and the Python `-m bcir.frontends.cfront
    -E` CLI each thread argv into the preprocessor's file name, matching `preprocess(name=path)`."""
    if not _CC:
        return
    from bcir.frontends.cfront.cpp import preprocess as _py_pp  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as d:
        exe = _build_bcir_cc(d)
        src = os.path.join(d, "unit.c")
        text = "a __FILE__ __LINE__\nb __FILE__\n"
        with open(src, "w") as f:
            f.write(text)
        # C rail: bcir-cc -E <path> emits the raw preprocessed text.
        cr = subprocess.run([exe, "-E", src], capture_output=True, text=True)
        assert cr.returncode == 0, cr.stderr
        assert f'"{src}"' in cr.stdout, cr.stdout                     # __FILE__ == the given path
        assert cr.stdout == _py_pp(text, name=src)                   # byte-identical to the oracle
        # Python CLI rail: -E <path> (the driver appends one trailing newline).
        rc, pyo, err = _cli(["-E", src])
        assert rc == 0, err
        assert f'"{src}"' in pyo, pyo
        assert pyo.rstrip("\n") == cr.stdout.rstrip("\n")            # same content across the rails
        # the default (no driver-supplied name) stays "<source>".
        assert _py_pp(text).startswith('a"<source>"')


def test_bcir_cc_driver_compiles_and_emits_artifacts():
    """`bcir-cc` -- the production C compiler driver -- compiles a driver with sibling/`-I` headers
    via a normal compile command (no test-harness include map), honours `-D` in a `#if`, and emits
    the verified C / claim graph / StreamPack artifacts. The C preprocessor's `-I` + `-D` path is
    dual-rail with the oracle (`bcir_cpp_run_ex` ~ cpp.py search_paths/defines)."""
    if not _CC:
        return
    from bcir.frontends.cfront import compile_unit  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as d:
        cc = _build_bcir_cc(d)
        uart = os.path.join(_C, "cfront_driver_uart.c")
        # (1) the sibling header resolves; the default output is the dual-rail structural summary.
        p = subprocess.run([cc, uart], capture_output=True, text=True)
        assert p.returncode == 0 and "ok=1" in p.stdout, (p.stdout, p.stderr)
        # (2) --emit-c emits the verified C + the C.2 attestation.
        p = subprocess.run([cc, "--emit-c", uart], capture_output=True, text=True)
        assert "bcir_uart_configure" in p.stdout and "attestation (C.2)" in p.stdout, p.stdout
        # (3) --emit-pack writes a valid StreamPack (the BSPK magic).
        pack = os.path.join(d, "u.pack")
        assert subprocess.run([cc, "--emit-pack", "-o", pack, uart]).returncode == 0
        with open(pack, "rb") as f:
            assert f.read(4) == b"BSPK"
        # (4) -I <dir> + -D macro: a header outside the source dir + a -D-selected #if branch, and
        #     the C rail agrees with the oracle given the same search path + define.
        incd = os.path.join(d, "inc"); os.makedirs(incd)
        with open(os.path.join(incd, "r.h"), "w") as f:
            f.write("typedef volatile unsigned int reg32;\nstruct dev { reg32 s; };\n")
        src = os.path.join(d, "m.c")
        with open(src, "w") as f:
            f.write('#include "r.h"\n#if defined(FAST)\n'
                    'unsigned int g(volatile struct dev *p){ return p->s + 1u; }\n'
                    '#else\nunsigned int g(volatile struct dev *p){ return p->s; }\n#endif\n')
        c_sum = subprocess.run([cc, "-I", incd, "-D", "FAST", src],
                               capture_output=True, text=True).stdout.strip()
        assert "binop=1" in c_sum and "ok=1" in c_sum, c_sum     # the FAST branch (+1u) -> one binop
        r = compile_unit(open(src, encoding="utf-8").read(), check_clang=False,
                         search_paths=[incd], defines={"FAST": "1"})
        entry = r.lowered.functions[next(reversed(r.lowered.functions))]
        assert sum(1 for c in entry.claims if c.op.startswith("c.bin.")) == 1   # oracle agrees
        # without -D FAST the other branch (no +1) is taken -> zero binops, on both rails.
        c_sum0 = subprocess.run([cc, "-I", incd, src], capture_output=True, text=True).stdout.strip()
        assert "binop=0" in c_sum0, c_sum0


def test_phase_d_uart_driver_write_and_poll_path():
    """Phase D (the write + control-flow half): a vendor-style UART register-map header
    (uart_regs.h) + driver (cfront_driver_uart.c) driven END TO END through the plug-in C compiler
    with no Python. Complements the DMA driver (test_phase_d_real_header_driver_end_to_end, which is
    read-only + a call graph) with the patterns a real driver lives on: MMIO register *writes*
    (u->BRR/CR/DR =) and a *bounded status-poll loop* (L6 while + if). bcir_cpp expands the #include
    + object macros; bcir_cfront lowers + R1-R18 verifies + C.2 attests; the closed loop plans /
    hydrates / executes the straight-line entry. Also exercises typedef / enum / union, L5 volatile
    MMIO, L2 bitfields, the L8 by-value ABI. The two rails agree on the structural summary, every
    driver function is Clang-behaviour-equivalent, and the entry executes R9/R10-R11 clean."""
    fx = "cfront_driver_uart.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src, _includes_for(fx))
    assert "ok=1" in oracle_summary, oracle_summary
    assert len(r.lowered.functions) == 3                      # cfg_low, send, configure
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        loop = _build_loop(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity diverged\n C: {c_summary}\nPY: {oracle_summary}"
        assert "ok=1" in c_summary, c_summary
        # the emitted C carries real volatile MMIO accesses + the union view + the C.2 attestation.
        assert "volatile uint32_t *" in c_emit and "union uart_cfg" in c_emit, c_emit
        assert "BCIR verified-C attestation (C.2)" in c_emit and "R13 provenance digest" in c_emit
        # every driver function (MMIO loads/stores, control flow, the bitfield + union ABI) is
        # behaviour-equivalent to the original under Clang.
        for name, lf in r.lowered.functions.items():
            assert _equiv(r.source, c_emit, lf) == "MATCH", f"{fx}:{name} not behaviour-equivalent"
        # the straight-line entry (uart_configure) plans/hydrates/executes, R9/R10-R11 clean, and
        # the loop's provenance digest is the one stamped in the emitted attestation.
        out = subprocess.run([loop, os.path.join(_C, fx)], capture_output=True, text=True).stdout.strip()
        assert out.startswith("loop:"), out
        m = dict(re.findall(r"(\w+)=([0-9]+)", out))
        assert int(m["executed"]) == int(m["claims"]) == len(entry.claims), out
        assert m["r9"] == "1" and m["r10r11"] == "1", out
        prov = re.search(r"prov=([0-9a-f]{16})", out)
        emit_prov = re.search(r"R13 provenance digest\s+([0-9a-f]{16})", c_emit)
        assert prov and emit_prov and prov.group(1) == emit_prov.group(1), "digest mismatch loop vs emit"


def test_L8_packed_layout_matches_clang():
    """The C frontend's packed struct offsets must equal Clang's sizeof/offsetof (the ABI)."""
    src = open(os.path.join(_C, "cfront_packed.c"), encoding="utf-8").read()
    hdr = compile_unit(src, check_clang=False).lowered.aggregates["wire_hdr"]
    assert hdr.size == 7 and hdr.field("addr")[1] == 1 and hdr.field("len")[1] == 5
    if not _CC:
        return
    probe = ("#include <stdint.h>\n#include <stddef.h>\n#include <stdio.h>\n"
             "struct __attribute__((packed)) wire_hdr { uint8_t cmd; uint32_t addr; uint16_t len; };\n"
             'int main(void){printf("%zu %zu %zu %zu", sizeof(struct wire_hdr),'
             " offsetof(struct wire_hdr,cmd), offsetof(struct wire_hdr,addr),"
             " offsetof(struct wire_hdr,len)); return 0;}")
    with tempfile.TemporaryDirectory() as d:
        c, e = os.path.join(d, "p.c"), os.path.join(d, "p")
        open(c, "w").write(probe)
        if subprocess.run([_CC, "-std=c11", c, "-o", e], capture_output=True).returncode == 0:
            nums = [int(x) for x in subprocess.run([e], capture_output=True, text=True).stdout.split()]
            assert nums == [hdr.size, hdr.field("cmd")[1], hdr.field("addr")[1], hdr.field("len")[1]]


def test_c_frontend_builds_warning_clean():
    if not _CC:
        return
    for unit in ("bcir_cfront.c", "bcir_cpp.c", "bcir_plan.c", "bcir_hydrate.c", "bcir_verify.c",
                 "bcir_cc.c"):
        ok = False
        for std in ("c23", "c11"):
            b = subprocess.run([_CC, f"-std={std}", "-Wall", "-Wextra", "-Werror", "-I", _C, "-c",
                                os.path.join(_C, unit), "-o", os.devnull], capture_output=True, text=True)
            if b.returncode == 0:
                ok = True
                break
        assert ok, f"{unit} has warnings:\n{b.stderr}"


def test_c_preprocessor_macros_conditionals_and_embed():
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        drv = os.path.join(d, "drv.c")
        open(drv, "w").write(
            '#include <stdio.h>\n#include "bcir_cpp.h"\n'
            'int main(int c,char**v){static char o[65536],e[256],s[65536];'
            'size_t n=fread(s,1,sizeof s-1,stdin);s[n]=0;'
            'if(bcir_cpp_run(s,c>1?v[1]:"",o,sizeof o,e,sizeof e)){printf("ERR %s",e);return 1;}'
            'fputs(o,stdout);return 0;}\n')
        exe = os.path.join(d, "drv")
        b = subprocess.run([_CC, "-std=c11", "-O1", "-I", _C, os.path.join(_C, "bcir_cpp.c"), drv,
                            "-o", exe], capture_output=True, text=True)
        assert b.returncode == 0, b.stderr

        def pp(src, basedir=""):
            return subprocess.run([exe, basedir], input=src, capture_output=True, text=True).stdout

        # function macro + object macro + rescanning
        out = pp("#define A 2\n#define SQ(x) ((x)*(x))\nint v = SQ(A+1);\n").replace(" ", "")
        assert "((2+1)*(2+1))" in out
        # #if arithmetic + #elifndef (C23)
        assert pp("#if 1+1==2\nyes\n#else\nno\n#endif\n").split() == ["yes"]
        assert pp("#ifdef X\na\n#elifndef Y\nb\n#endif\n").split() == ["b"]
        # C23 #embed -> the byte list
        open(os.path.join(d, "blob.bin"), "wb").write(bytes([10, 20, 30]))
        emb = pp('x\n#embed "blob.bin"\ny\n', d)
        assert "10, 20, 30" in emb

        # predefined macros: __LINE__ (per-line), __FILE__ (the "<source>" default), __STDC_HOSTED__.
        assert pp("a __LINE__\nb __LINE__\nc __LINE__\n").split() == ["a", "1", "b", "2", "c", "3"]
        assert pp("x __FILE__\n").strip() == 'x"<source>"'
        assert pp("#ifdef __LINE__\nyes\n#endif\n").split() == ["yes"]
        assert pp("#if defined(__FILE__) && __LINE__ == 1\nok\n#endif\n").split() == ["ok"]
        assert pp("#if __STDC_HOSTED__\nhosted\n#endif\n").split() == ["hosted"]

        # #line: resets the presumed line of the next line (and __FILE__ when named).
        assert pp("a __LINE__\n#line 100\nb __LINE__\n").split() == ["a", "1", "b", "100"]
        assert pp('#line 50 "foo.c"\nx __LINE__ __FILE__\n').strip() == 'x 50"foo.c"'

        # dual-rail gate: the C twin's output is byte-identical to cpp.py over the same probes,
        # including __LINE__ through a function macro (the invocation line), #line, and across #include.
        from bcir.frontends.cfront.cpp import preprocess as _py_pp  # noqa: PLC0415
        open(os.path.join(d, "ph.h"), "w").write("in __LINE__ __FILE__")
        probes = [
            "a __LINE__\nb __LINE__\nc __LINE__\n",
            "x __FILE__\n",
            "#define ID(x) x\nint b = ID(__LINE__);\n",
            "#define L __LINE__\nq\nint c = L;\n",
            "#ifndef __FILE__\nno\n#else\nyes\n#endif\n",
            "#if defined __LINE__\nok\n#endif\n",
            "aa \\\nbb\n__LINE__\n",
            "__STDC__ __STDC_VERSION__ __STDC_HOSTED__\n",
            "a __LINE__\n#line 100\nb __LINE__\nc __LINE__\n",
            "#define N 200\n#line N\nq __LINE__\n",
            '#line 30 "a\\"b.c"\nz __FILE__\n',                  # an escaped quote in the name
            "p __LINE__\n#line\nq __LINE__\n",                   # malformed -> ignored
            "#if 0\n#line 999\n#endif\nr __LINE__\n",            # inactive branch -> skipped
            'int x; _Pragma("once") int y;\n',                   # _Pragma operator: a no-op
            'p _Pragma("a(b)c") q\n',                            # balanced parens consumed
            "#define DO(x) _Pragma(#x)\nDO(message hi)\nz\n",    # _Pragma produced by a macro
            "#if __has_attribute(packed)\nP\n#else\nn\n#endif\n",        # feature-test: supported
            "#if __has_attribute(__aligned__)\nA\n#endif\n",            # GCC __x__ spelling
            "#if __has_attribute(deprecated)\nd\n#else\nU\n#endif\n",   # unsupported attribute
            "#if __has_builtin(__builtin_expect)\nb\n#else\nU\n#endif\n",
            "#if __has_c_attribute(nodiscard)\nc\n#else\nU\n#endif\n",
            "#ifdef __has_attribute\nDEF\n#endif\n",                    # reported as `defined`
            "#if defined(__has_builtin) && !__has_builtin(x)\nG\n#endif\n",
            "#define V(...) f(__VA_ARGS__)\nV(1,2,3)\n",                 # __VA_ARGS__ flattens all args
            "#define L(a, ...) g(a, __VA_ARGS__)\nL(x,1,2)\nL(z)\n",    # named + variadic, incl. empty
            "#define S(...) #__VA_ARGS__\nS(1, 2, 3)\nS()\n",           # stringize __VA_ARGS__
            "#define P(...) x ## __VA_ARGS__\nP(1,2)\nP()\n",           # paste __VA_ARGS__
            "#define LOG(f, ...) p(f __VA_OPT__(,) __VA_ARGS__)\nLOG(z)\nLOG(z,1,2)\n",  # __VA_OPT__
            "#define W(x, ...) [x __VA_OPT__(/ __VA_ARGS__)]\nW(p)\nW(p,q,r)\n",         # nested VA
            "#define E(...) z __VA_OPT__(Y)\nE()\nE(,)\nE(q)\n",        # emptiness incl. a lone comma
        ]
        for s in probes:
            assert pp(s) == _py_pp(s), f"twin divergence on {s!r}\n C: {pp(s)!r}\nPY: {_py_pp(s)!r}"
        # the #include-boundary case (header numbered from 1, __FILE__ restored on return), and the
        # same with a #line-set name that must survive the include and restore afterwards.
        inc = 't __LINE__ __FILE__\n#include "ph.h"\nu __LINE__ __FILE__\n'
        assert pp(inc, d) == _py_pp(inc, search_paths=[d])
        linc = '#line 7 "a.h"\nt __LINE__ __FILE__\n#include "ph.h"\nu __LINE__ __FILE__\n'
        assert pp(linc, d) == _py_pp(linc, search_paths=[d])

        # __has_include resolves against the search path on both rails (the C eval_if gained this);
        # ph.h exists in `d`, so it probes true; a missing header probes false. Both <...> and "...".
        for s in ('#if __has_include("ph.h")\nY\n#else\nN\n#endif\n',
                  '#if __has_include(<ph.h>)\nY\n#else\nN\n#endif\n',
                  '#if __has_include("nope.h")\nY\n#else\nN\n#endif\n',
                  '#if defined(__has_include) && __has_include("ph.h")\nOK\n#endif\n',
                  '#if !__has_include("nope.h")\nNEG\n#endif\n'):
            assert pp(s, d) == _py_pp(s, search_paths=[d]), \
                f"__has_include divergence on {s!r}\n C:{pp(s, d)!r}\nPY:{_py_pp(s, search_paths=[d])!r}"

        # __DATE__/__TIME__: SOURCE_DATE_EPOCH (UTC) freezes both twins to the same string.
        def ppe(src, epoch):
            env = dict(os.environ); env["SOURCE_DATE_EPOCH"] = epoch
            return subprocess.run([exe, ""], input=src, capture_output=True, text=True, env=env).stdout
        old_epoch = os.environ.get("SOURCE_DATE_EPOCH")
        try:
            for epoch, want in (("1234567890", '"Feb 13 2009""23:31:30"'),     # 2009-02-13 23:31:30Z
                                ("1577836800", '"Jan  1 2020""00:00:00"')):    # padded single-digit day
                os.environ["SOURCE_DATE_EPOCH"] = epoch                        # _py_pp reads it here
                assert ppe("__DATE__ __TIME__\n", epoch).strip() == want
                assert ppe("__DATE__ __TIME__\n", epoch) == _py_pp("__DATE__ __TIME__\n")
        finally:
            if old_epoch is None:
                os.environ.pop("SOURCE_DATE_EPOCH", None)
            else:
                os.environ["SOURCE_DATE_EPOCH"] = old_epoch


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
