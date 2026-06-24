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

import atexit
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

# Bounds-quarantine support (§5.12): the emit of a `masked` array access uses BCIR_CHK(rid, idx, N, "site"),
# which calls bcir_bounds_quarantine on an out-of-bounds index. Inlined here (matching the ABI in
# runtime/c/bcir_quarantine.h) so the equivalence harness is self-contained; for the in-bounds seeds the
# handler is never reached, so a guarded access is behaviour-identical to the raw `a[i]`.
_BOUNDS_GUARD = (
    "#include <stdlib.h>\n#include <stddef.h>\n"
    "static size_t bcir_bounds_quarantine(uint64_t r,uint64_t i,uint64_t e,const char*s)"
    "{(void)r;(void)i;(void)e;(void)s;abort();return 0;}\n"
    "#define BCIR_CHK(rid,idx,n,site) ((uint64_t)(idx)<(uint64_t)(n)?(size_t)(idx):"
    "bcir_bounds_quarantine((uint64_t)(rid),(uint64_t)(idx),(uint64_t)(n),(site)))")
# straight-line fixtures run the full execute loop; control-flow fixtures get parity + emit + Clang ≡
# (control flow is not a flat StreamPack segment stream, so the loop runs the straight-line set).
_STRAIGHTLINE = ["cfront_regmap.c", "cfront_array.c", "cfront_array2d.c", "cfront_widerow.c", "cfront_deref.c",
                 "cfront_callgraph.c", "cfront_typedef.c", "cfront_enum.c", "cfront_enumtype.c", "cfront_ternary.c",
                 "cfront_sizeof.c", "cfront_strsizeof.c", "cfront_strval.c", "cfront_charlit.c",
                 "cfront_strtab.c", "cfront_strconcat.c", "cfront_widelit.c", "cfront_cast.c", "cfront_alignof.c", "cfront_static.c",
                 "cfront_global.c", "cfront_compound.c", "cfront_logic.c", "cfront_abi.c", "cfront_signed.c", "cfront_signedcmp.c", "cfront_signedbare.c", "cfront_longunary.c", "cfront_boolnorm.c", "cfront_unarypromote.c", "cfront_floatsigncast.c", "cfront_intsigncast.c", "cfront_boolcast.c", "cfront_boolmember.c"]   # + char consts + str table/dedup + const LUT + ABI sizeof model + bool normalization + unary integer-promotion/float + float->signed + int->signed cast + bool cast + _Bool member/element store-normalization
_CONTROL = ["cfront_branch.c", "cfront_while.c", "cfront_for.c", "cfront_dowhile.c",
            "cfront_continue.c", "cfront_switch.c", "cfront_switchfall.c", "cfront_goto.c", "cfront_incdec.c",
            "cfront_multidecl.c", "cfront_commastep.c", "cfront_emptystmt.c", "cfront_loopreuse.c", "cfront_loopscope.c", "cfront_blockscope.c", "cfront_localmd.c", "cfront_ptrlocal.c"]
            # + multi-declarator locals (T a=x, b, c=z), comma-operator for-step (i++, j--), empty stmts
_PREPROC = ["cfront_macros.c", "cfront_ppinc.c", "cfront_comments.c"]      # L7: exercise the preprocessor
_ABI = ["cfront_structret.c", "cfront_structcall.c",  # L8: struct return-by-value (+ using a call RESULT)
        "cfront_packed.c",                            # + packed layout
        "cfront_union.c",                             # + full union (members overlap at offset 0)
        "cfront_interleave.c",                        # + enum/struct defined *between* two functions
        "cfront_funcptr.c",                           # + funcptr param + indirect call (HAL dispatch)
        "cfront_fnptrparam.c",                         # + DIRECT inline funcptr params int (*g)(int) (no typedef)
        "cfront_rmw.c",                               # + MMIO register read-modify-write (d->reg |= bits)
        "cfront_bitfield.c",                          # + MMIO bitfield write (r->field = v, c.bf.set)
        "cfront_bfcompound.c",                         # + bitfield compound-assign (r->field |= bits)
        "cfront_signedbf.c",                           # + signed bitfield read sign-extension (int x:N)
        "cfront_widebf.c",                             # + WIDE bitfields in a 64-bit unit (long long x:N, N>32)
        "cfront_packedbf.c",                           # + PACKED bitfields (bit-by-bit, byte/word-straddling)
        "cfront_alignasmember.c",                      # + over-aligned members (_Alignas/alignas/aligned(N))
        "cfront_anonmember.c",                          # + anonymous struct/union members (promoted leaves)
        "cfront_unnamedbf.c",                           # + unnamed / zero-width bitfields (layout-only padding)
        "cfront_charmember.c",                          # + plain `char` members read as `char` not int8_t (ARM)
        "cfront_aostruct.c",                            # + ARRAY-OF-STRUCTS members p->arr[i].field (strided)
        "cfront_fnptrmember.c",                         # + funcptr members set from NAMED functions (dispatch)
        "cfront_assignexpr.c",                          # + assignment as an EXPRESSION (a=b=c, if((x=f()))...)
        "cfront_memassignexpr.c",                       # + member-lvalue assignment as a value ((p->x=v)+1)
        "cfront_signedload.c",                         # + signed sub-int member/array read sign-extension
        "cfront_restrict.c",                           # + restrict/__restrict pointer params (consumed hint)
        "cfront_shiftassign.c",                        # + <<= / >>= shift compound-assign (scalar/member/array)
        "cfront_ptrarith.c",                           # + pointer mutation p++ / p += n (buffer-walk cursor)
        "cfront_structmulti.c",                        # + multi-declarator struct members (unsigned x,y,z;)
        "cfront_nestmember.c",                         # + nested member access (o.pos.lo / dev->ctrl.bf)
        "cfront_memberarray.c",                        # + native 1-D struct member arrays (s.arr[i])
        "cfront_neststruct.c"]                          # + nested struct members + nested-brace init `{ {..}, .. }`
_FLOAT = ["cfront_float.c", "cfront_floatcast.c", "cfront_hexfloat.c", "cfront_mathh.c",
          "cfront_mathh_mixed.c", "cfront_mathh_long.c", "cfront_mathh_ptr.c",
          "cfront_calltyped.c", "cfront_complex.c", "cfront_complexdiv.c",   # + C99 _Complex (#complex) + complex `/`
          "cfront_complextrans.c",                                           # + complex transcendentals (#complextrans)
          "cfront_imagunit.c",                                               # + <complex.h> imaginary unit `I` (#imagunit)
          "cfront_complexlong.c",                                           # + long-double complex (#complexlong)
          "cfront_complexmember.c"]                                          # + complex struct members (#complexmember)
#   float/double: parity + emit + Clang ≡ (the
#   integer StreamPack executor doesn't compute float; the math is delegated to the resident backend)
_INIT = ["cfront_dispatch_table.c",   # designated initializers ([i]=v) for a file-scope dispatch table
         "cfront_agginit.c",          # local struct/union aggregate init ({.field=v}) -> = {0} + stores
         "cfront_localarray.c",       # local array decl T a[N] + array aggregate init (positional + [i]=)
         "cfront_nestinit.c"]         # NESTED-brace init `{ m, {e0..}, n }` for a struct's array member
#   (a local decl, a compound literal, and a struct return BY VALUE) -- offset-based element stores
#   parity + emit + Clang ≡ (the table is referenced by name, defined in the source -- not re-hydrated)
_PTRVALUE = ["cfront_ptrvalue.c",   # pointer VALUES across non-address contexts (#ptrvalue): pointer
#   arithmetic `p + i` as an rvalue returned by value -- the temp carries the pointee type (a real
#   `T *t = p + i`), not a truncating uint32. Parity + emit + Clang ≡ (returns a pointer, not executed).
             "cfront_ptrfield.c",   # + a pointer stored into / loaded from a struct field (#ptrfield):
             "cfront_addrof.c",   # general address-of `&`
             "cfront_addrofarr.c",   # member-array element address
             "cfront_addrofaos.c",   # array-of-structs element field address
             "cfront_extentsnap.c",  # §5.12 recoverable-extent SNAPSHOT: expression counts (#extentsnap)
             "cfront_comma.c",       # the comma operator in a primary parenthesized expr (#comma)
             "cfront_vla.c",         # native 1-D stack VLAs `T a[n]` -- in-body decl + masked bounds (#vla)
             "cfront_vlasizeof.c",   # runtime `sizeof a` of a VLA -> extent * sizeof(elem) (#vlasizeof)
             "cfront_vlaparam.c",    # VLA function parameters `T a[n]` -> masked param bounds vs n (#vlaparam)
             "cfront_vlamd.c",       # multi-dimensional VLAs `T a[m][n]` -> flat m*n extent + Horner (#vlamd)
             "cfront_lvassignexpr.c",# array/deref/nested lvalue assignment used as a value (#lvassignexpr)
             "cfront_narrowcompound.c", # a narrow-target compound assignment AS A VALUE re-reads (#narrowcompound)
             "cfront_bfassignexpr.c",# a BITFIELD member assignment used as a value (#bfassignexpr)
             "cfront_aosassignexpr.c",# an array-of-structs field / member-array element as a value (#aosassignexpr)
             "cfront_signedfnptr.c", # a SIGNED function-pointer return reads back signed (#signedfnptr)
             "cfront_stdlibmem.c"]   # + <stdlib.h> malloc/calloc/realloc/free as external libc edges (#stdlibmem)   # + address-of an array-of-structs element field in a member (#addrofaos)   # + address-of a member-array element (#addrofarr): &s.arr[i] / &s.m[i][j]   # + general address-of `&` of an lvalue (#addrof): &s->m / &*p / &arr[i]   # + a pointer stored into / loaded from a struct field (#ptrfield):
#   the member occupies pointer_size (8) bytes -- a correct layout (an adjacent field no longer overlaps
#   the high half of the pointer) and an untruncated 8-byte store/load that carries the real `T *` type.
_FIXTURES = _STRAIGHTLINE + _CONTROL + _PREPROC + _ABI + _FLOAT + _INIT + _PTRVALUE
# §5.8 atomics/fences/CAS run their own gate: their memory side effects make the generic
# pure-function equivalence harness invalid (it would call the original first and observe
# the mutated cell), so they get a side-effect-aware behaviour check below.
_ATOMIC = ["cfront_atomic.c", "cfront_cmpxchg.c", "cfront_atomic11.c", "cfront_atomic_xchg.c", "cfront_cmpxchg11.c"]  # + C11 stdatomic + atomic_exchange + compare_exchange


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


_BUILD_DIR = None
_BUILD_CACHE: dict = {}


def _session_build_dir() -> str:
    global _BUILD_DIR
    if _BUILD_DIR is None:
        _BUILD_DIR = tempfile.mkdtemp(prefix="bcir_cc_build_")
        atexit.register(shutil.rmtree, _BUILD_DIR, ignore_errors=True)
    return _BUILD_DIR


def _compile_once(key: str, out_name: str, src_names: tuple, label: str) -> str:
    """Compile the runtime/c `src_names` to `out_name` ONCE per session (memoized by `key`) and reuse
    the binary across every test that needs it. These binaries (the ~2100-line `bcir_cfront.c` + its
    siblings, at -O2) are identical from test to test, so the old per-test rebuild was the suite's
    single dominant wall-cost -- ~20 redundant front-end/loop/driver compiles. The cache lives for the
    process (run_all + pytest both run in one process), keyed by the source set so a different binary
    still builds its own."""
    cached = _BUILD_CACHE.get(key)
    if cached:
        return cached
    exe = os.path.join(_session_build_dir(), out_name)
    srcs = [os.path.join(_C, s) for s in src_names]
    for std in ("c23", "c11"):
        b = subprocess.run([_CC, f"-std={std}", "-O2", "-I", _C, *srcs, "-o", exe],
                           capture_output=True, text=True)
        if b.returncode == 0:
            _BUILD_CACHE[key] = exe
            return exe
    raise AssertionError(f"{label} build failed:\n{b.stderr}")


def _build_frontend(d: str) -> str:
    # `d` is retained for call-site compatibility; the binary is cached session-wide (see _compile_once).
    # bcir_cfront verifies (bcir_verify.c) and the pack law reaches into bcir_runtime.c, so both link in.
    return _compile_once("frontend", "tcf",
                         ("bcir_cfront.c", "bcir_cpp.c", "bcir_verify.c", "bcir_runtime.c", "test_cfront.c"),
                         "C frontend")


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
        elif ct.is_complex:                          # a _Complex param: seed BOTH axes (finite, in range)
            decls.append(f"  {_cname(ct)} s{i};")
            el = "float" if ct.size == 8 else ("long double" if ct.size > 16 else "double")
            setup.append(f"    s{i}=({el})(rng()%1000) + ({el})(rng()%1000)*I;")
            args.append(f"s{i}")
        else:
            decls.append(f"  {_cname(ct)} s{i};")
            # a float param gets an in-range value (so a float->int cast stays defined, not UB);
            # an integer scalar stays below 2**31 so it is non-negative as `int` -- the value model
            # is unsigned, so an int->float cast must agree in sign (wrapping arithmetic is unaffected).
            mod = 1000 if ct.is_float else (200 if has_ptr else 2000000000)
            setup.append(f"    s{i}=({_cname(ct)})(rng()%{mod});")
            args.append(f"s{i}")
    call = ", ".join(args)
    rt = _cname(entry.ret_type)
    if entry.ret_type.is_complex:
        # a _Complex result is compared element-wise with a nan-aware equality: value-based (creall/cimagl,
        # narrower complex widening exactly), so it is immune to `long double _Complex`'s indeterminate x87
        # padding bytes -- which memcmp would wrongly flag -- AND nan-safe, which a complex division by a
        # near-zero divisor needs (`==` is false for a nan, so the isnan&&isnan arm catches it). Both rails
        # run the identical native op, so equal-value already implies bit-equal (incl. signed zero / inf).
        cmp = (f"    {rt} ra={entry.name}({call}), rb=bcir_{entry.name}({call});\n"
               f"    if(!((creall(ra)==creall(rb)||(isnan(creall(ra))&&isnan(creall(rb))))"
               f"&&(cimagl(ra)==cimagl(rb)||(isnan(cimagl(ra))&&isnan(cimagl(rb))))))"
               f"{{printf(\"MISMATCH@%d\\n\",i);return 1;}}")
    elif entry.ret_type.is_aggregate:
        # an aggregate result is compared BIT-exactly (memcmp): both rails run the identical stores.
        cmp = (f"    {rt} ra={entry.name}({call}), rb=bcir_{entry.name}({call});\n"
               f"    if(memcmp(&ra,&rb,sizeof ra)){{printf(\"MISMATCH@%d\\n\",i);return 1;}}")
    else:
        cmp = (f"    if({entry.name}({call})!=bcir_{entry.name}({call}))"
               f"{{printf(\"MISMATCH@%d\\n\",i);return 1;}}")
    harness = f"""#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <math.h>
#include <complex.h>
{_BOUNDS_GUARD}
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
            b = subprocess.run([_CC, f"-std={std}", "-O2", c, "-o", e, "-lm"],   # -lm: <math.h> links
                               capture_output=True, text=True)
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
{_BOUNDS_GUARD}
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
            b = subprocess.run([_CC, f"-std={std}", "-O2", c, "-o", e, "-lm"],   # -lm: <math.h> links
                               capture_output=True, text=True)
            if b.returncode == 0:
                break
        else:
            return f"build-failed:{b.stderr.strip().splitlines()[-1] if b.stderr else '?'}"
        return subprocess.run([e], capture_output=True, text=True).stdout.strip()


def _parity_check_fixture(args):
    """Parity + dual-emit behaviour-equivalence for ONE fixture. Returns (fx, None) on pass or (fx, msg)
    on failure. Module-level + (exe, fx) string args so it is dispatchable to a process pool: the ~90
    fixtures each run two Clang compile+run cycles (the twin's emit AND the oracle's own emit), which
    dominate this module's wall time and are independent (each its own tempdir), so they fan out across
    the runner's cores. (Threads do not help -- the oracle lowering + harness build are GIL-bound.)"""
    exe, fx = args
    path = os.path.join(_C, fx)
    src = open(path, encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src, _includes_for(fx))
    c_summary, c_emit = _c_run(exe, path)
    if c_summary != oracle_summary:
        return (fx, f"parity diverged\n C: {c_summary}\nPY: {oracle_summary}")
    # equivalence uses the PREPROCESSED source (r.source) so Clang needs no #include.
    if _equiv(r.source, c_emit, entry) != "MATCH":
        return (fx, "emitted C not behaviour-equivalent")
    # ALSO compile + run the ORACLE's own emitted C (the check above uses the C twin's emit, so the oracle
    # emitter was unguarded across the corpus -- the general form of the #387 fix, which caught a member
    # array at offset 0 emitting an invalid `struct[idx]`).
    oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    if _equiv(r.source, oracle_emit, entry) != "MATCH":
        return (fx, "oracle's own emitted C not behaviour-equivalent")
    # §5.12 bounds-promotion parity: both rails must promote the SAME accesses to `masked` (the R13 digest
    # includes `bounds`), so they emit the same number of `BCIR_CHK` guards -- a local/static array OR a
    # malloc/calloc'd pointer with a recovered extent. A divergence here means one rail promoted and the
    # other did not (a silent two-rail split the claim-summary parity does not catch).
    if oracle_emit.count("BCIR_CHK") != c_emit.count("BCIR_CHK"):
        return (fx, f"bounds-guard parity: oracle={oracle_emit.count('BCIR_CHK')} "
                    f"twin={c_emit.count('BCIR_CHK')} BCIR_CHK guards")
    return (fx, None)


def test_python_c_parity_and_equivalence_across_fixtures():
    if not _CC:
        # quick tier: still validate the oracle side computes the summaries.
        for fx in _FIXTURES:
            s, _, _ = _oracle(open(os.path.join(_C, fx), encoding="utf-8").read(), _includes_for(fx))
            assert "ok=1" in s
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        workers = min(len(_FIXTURES), os.cpu_count() or 1)
        if workers > 1:
            import concurrent.futures  # noqa: PLC0415 -- only when the toolchain tier actually runs
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_parity_check_fixture, [(exe, fx) for fx in _FIXTURES]))
        else:
            results = [_parity_check_fixture((exe, fx)) for fx in _FIXTURES]
    fails = [(fx, msg) for fx, msg in results if msg]
    assert not fails, "C-frontend parity/equivalence failures:\n" + "\n".join(
        f"  {fx}: {msg}" for fx, msg in fails)


def test_pointer_to_pointer_dual_rail():
    """Pointer-to-pointer (#ptr2ptr): `int **pp`, the double dereference `**pp`, `*pp = q` (the
    output-parameter idiom), `**pp = v` / `**pp += d`, and `int **pp = &p` built by address-of-a-pointer.
    The type model gained a pointer indirection DEPTH on both rails -- `*pp` on a `T**` loads a `T*`
    (pointer_size bytes), `**pp` derefs that to a `T`, and `&p` of a `T*` yields a `T**`. Both rails
    modeled `int **` as a single `int *` before (so `*pp` read the base width, `**pp` fell back, and a
    store truncated). Parity + a bespoke behaviour harness: the generic equivalence harness fills a
    pointee with random bytes -- invalid to dereference for a double pointer -- so this builds real
    x / &x / &&x chains and checks BOTH the twin's and the oracle's emit == Clang."""
    fx = "cfront_ptr2ptr.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["p2_read", "p2_get", "p2_set", "p2_store_through", "p2_rmw", "p2_local"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""int main(void){
  for(int i=-50;i<3000;i+=7){
    int x=i*3-7; int *px=&x; int **ppx=&px;
    if(p2_read_s(ppx)!=bcir_p2_read(ppx)){printf("read@%d\n",i);return 1;}
    if(p2_get_s(ppx)!=bcir_p2_get(ppx)){printf("get@%d\n",i);return 1;}
    int y=i+9; int *a1=px,*a2=px; int **b1=&a1,**b2=&a2;
    p2_set_s(b1,&y); bcir_p2_set(b2,&y);
    if(*b1!=*b2){printf("set@%d\n",i);return 1;}
    int s1=x,s2=x; int *p1=&s1,*p2=&s2; int **q1=&p1,**q2=&p2;
    if(p2_store_through_s(q1,i)!=bcir_p2_store_through(q2,i)||s1!=s2){printf("store@%d\n",i);return 1;}
    int u1=x,u2=x; int *r1=&u1,*r2=&u2; int **w1=&r1,**w2=&r2;
    if(p2_rmw_s(w1,i)!=bcir_p2_rmw(w2,i)||u1!=u2){printf("rmw@%d\n",i);return 1;}
    if(p2_local_s(i)!=bcir_p2_local(i)){printf("local@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}"
            cpath, epath = os.path.join(d, f"{label}.c"), os.path.join(d, label)
            open(cpath, "w").write(harness)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} harness build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} emit not behaviour-equivalent ({out})"


def test_field_deref_dual_rail():
    """Deref-through a loaded pointer field (#fieldderef): `*(s->p)`, the chain `s->mid->k` (member
    access through a loaded pointer-to-struct field, and the two-hop `s->mid->leaf->x`), and the
    subscript `s->p[i]` -- reads, writes, and compound RMW. Both rails resolved a member used as a base
    to the enclosing struct's address + the field type (so a deref read the struct's own bytes); now a
    pointer-valued field used as a base is loaded and the loaded pointer becomes the new base. Slice 2b
    of the pointer-value model. Parity + a bespoke behaviour harness (the generic one fills a pointee
    with random bytes -- an invalid deref target -- so this builds real Box->Mid->Leaf chains and checks
    BOTH the twin's and the oracle's emit == Clang."""
    fx = "cfront_fieldderef.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["fd_read", "fd_write", "fd_qread", "fd_index", "fd_index_set", "fd_rmw", "fd_chain1",
             "fd_chain1_set", "fd_chain1_rmw", "fd_chain2", "fd_chain2_long", "fd_chain2_set"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""int main(void){
  for(int i=-40;i<2000;i+=7){
    int buf[8]; for(int k=0;k<8;k++) buf[k]=i*k-3;
    long lq=(long)i*1000003L-7;
    struct Box b={0,&buf[0],i,&lq};
    if(fd_read_s(&b)!=bcir_fd_read(&b)){printf("read@%d\n",i);return 1;}
    if(fd_qread_s(&b)!=bcir_fd_qread(&b)){printf("qread@%d\n",i);return 1;}
    if(fd_index_s(&b,5)!=bcir_fd_index(&b,5)){printf("index@%d\n",i);return 1;}
    int w1[4]={0},w2[4]={0};
    struct Box c1={0,&w1[0],0,&lq},c2={0,&w2[0],0,&lq};
    fd_write_s(&c1,i); bcir_fd_write(&c2,i);
    if(w1[0]!=w2[0]){printf("write@%d\n",i);return 1;}
    fd_index_set_s(&c1,3,i); bcir_fd_index_set(&c2,3,i);
    if(w1[3]!=w2[3]){printf("iset@%d\n",i);return 1;}
    if(fd_rmw_s(&c1,i)!=bcir_fd_rmw(&c2,i)||w1[0]!=w2[0]){printf("rmw@%d\n",i);return 1;}
    struct Leaf lf1={i+1,(long)i*7+2},lf2={i+1,(long)i*7+2};
    struct Mid m1={&lf1,i+5},m2={&lf2,i+5};
    struct Box d1={&m1,&buf[0],0,&lq},d2={&m2,&buf[0],0,&lq};
    if(fd_chain1_s(&d1)!=bcir_fd_chain1(&d2)){printf("chain1@%d\n",i);return 1;}
    if(fd_chain2_s(&d1)!=bcir_fd_chain2(&d2)){printf("chain2@%d\n",i);return 1;}
    if(fd_chain2_long_s(&d1)!=bcir_fd_chain2_long(&d2)){printf("chain2l@%d\n",i);return 1;}
    fd_chain1_set_s(&d1,i*3); bcir_fd_chain1_set(&d2,i*3);
    if(m1.k!=m2.k){printf("c1set@%d\n",i);return 1;}
    if(fd_chain1_rmw_s(&d1,i)!=bcir_fd_chain1_rmw(&d2,i)||m1.k!=m2.k){printf("c1rmw@%d\n",i);return 1;}
    fd_chain2_set_s(&d1,i*2); bcir_fd_chain2_set(&d2,i*2);
    if(lf1.x!=lf2.x){printf("c2set@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}"
            cpath, epath = os.path.join(d, f"{label}.c"), os.path.join(d, label)
            open(cpath, "w").write(harness)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} harness build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} emit not behaviour-equivalent ({out})"


def test_pointer_element_signedness_dual_rail():
    """Pointer-element signedness (#ptrsign): a load / store / subscript through a pointer carries the
    POINTEE's signedness, not just its width -- so a deref of a signed sub-int pointer sign-extends (a
    negative byte/short reads back negative), an unsigned one zero-extends, and the loaded value drives
    signed-vs-unsigned divide / remainder / shift / comparison / the usual arithmetic conversions. Both
    rails thread the pointee sign through every pointer-resource path (param, local, struct field, and a
    pointer-arithmetic result). Parity + a bespoke behaviour harness over negative + boundary pointee
    values: a width-only model would zero-extend a negative pointee and pick the wrong arithmetic sign."""
    fx = "cfront_ptrsign.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["ps_s8", "ps_u8", "ps_s16", "ps_u16", "ps_s8_divrem", "ps_u8_div", "ps_s8_shr", "ps_u8_shr",
             "ps_s8_cmp", "ps_s64_div", "ps_u64_div", "ps_arith", "ps_uac", "ps_field", "ps_w8"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""int main(void){
  for(long i=-400;i<400;i++){
    int8_t sb=(int8_t)(i*7-3); uint8_t ub=(uint8_t)(i*5+1);
    int16_t ha[4]={(int16_t)(i*3),(int16_t)(-i),(int16_t)(i+9),(int16_t)(i*7)};
    uint16_t ua[4]={(uint16_t)(i*3),(uint16_t)(i),(uint16_t)(i+9),(uint16_t)(i*7)};
    long lv=i*1000000007L-7; unsigned long ul=(unsigned long)(i*2654435761UL+9);
    int8_t a[4]={(int8_t)i,(int8_t)(i-1),(int8_t)(i+2),(int8_t)(-i)};
    if(ps_s8_s(&sb)!=bcir_ps_s8(&sb)){printf("s8@%ld\n",i);return 1;}
    if(ps_u8_s(&ub)!=bcir_ps_u8(&ub)){printf("u8@%ld\n",i);return 1;}
    if(ps_s16_s(ha,2)!=bcir_ps_s16(ha,2)){printf("s16@%ld\n",i);return 1;}
    if(ps_u16_s(ua,2)!=bcir_ps_u16(ua,2)){printf("u16@%ld\n",i);return 1;}
    if(ps_s8_divrem_s(&sb)!=bcir_ps_s8_divrem(&sb)){printf("divrem@%ld\n",i);return 1;}
    if(ps_u8_div_s(&ub)!=bcir_ps_u8_div(&ub)){printf("udiv@%ld\n",i);return 1;}
    if(ps_s8_shr_s(&sb)!=bcir_ps_s8_shr(&sb)){printf("sshr@%ld\n",i);return 1;}
    if(ps_u8_shr_s(&ub)!=bcir_ps_u8_shr(&ub)){printf("ushr@%ld\n",i);return 1;}
    if(ps_s8_cmp_s(&sb)!=bcir_ps_s8_cmp(&sb)){printf("cmp@%ld\n",i);return 1;}
    if(ps_s64_div_s(&lv)!=bcir_ps_s64_div(&lv)){printf("s64@%ld\n",i);return 1;}
    if(ps_u64_div_s(&ul)!=bcir_ps_u64_div(&ul)){printf("u64@%ld\n",i);return 1;}
    if(ps_arith_s(a,2)!=bcir_ps_arith(a,2)){printf("arith@%ld\n",i);return 1;}
    if(ps_uac_s(&ub,(int)i)!=bcir_ps_uac(&ub,(int)i)){printf("uac@%ld\n",i);return 1;}
    struct Buf bb={&sb,&ub}; if(ps_field_s(&bb)!=bcir_ps_field(&bb)){printf("field@%ld\n",i);return 1;}
    signed char w1=0,w2=0; ps_w8_s(&w1,(int)i); bcir_ps_w8(&w2,(int)i);
    if(w1!=w2){printf("w8@%ld\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}"
            cpath, epath = os.path.join(d, f"{label}.c"), os.path.join(d, label)
            open(cpath, "w").write(harness)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} harness build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} emit not behaviour-equivalent ({out})"


def test_funcptr_dispatch_through_loaded_pointer_dual_rail():
    """Funcptr dispatch through a loaded pointer (#fnptrchain): calling a function-pointer struct member
    reached THROUGH a loaded pointer-to-struct field -- `d->ops->fn(args)` and the two-hop
    `s->dev->ops->fn(args)`. The direct `o->fn(args)` already fused member access + call into one
    `c.call.imember` claim; the postfix pointer chain (from #fieldderef) now recognizes a `(` after a
    member as that fused indirect call on the loaded pointer base, emitting `ptr->fn(args)`. The oracle
    already lowered this; this brings the C twin into agreement. Parity + a bespoke behaviour harness
    that wires real operation tables (each call is an R18-opaque indirect dispatch)."""
    fx = "cfront_fnptrchain.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["fc_add", "fc_combo", "fc_twohop"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""static int real_add(int a,int b){return a+b;}
static int real_sub(int a,int b){return a-b;}
static int real_mul(int a,int b){return a*b;}
int main(void){
  struct Ops ops={real_add,real_sub,real_mul};
  struct Dev dev={&ops,42};
  struct Sys sys={&dev,7};
  for(int i=-200;i<200;i++){
    int a=i*3-1,b=7-i;
    if(fc_add_s(&dev,a,b)!=bcir_fc_add(&dev,a,b)){printf("add@%d\n",i);return 1;}
    if(fc_combo_s(&dev,a,b)!=bcir_fc_combo(&dev,a,b)){printf("combo@%d\n",i);return 1;}
    if(fc_twohop_s(&sys,a,b)!=bcir_fc_twohop(&sys,a,b)){printf("twohop@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}"
            cpath, epath = os.path.join(d, f"{label}.c"), os.path.join(d, label)
            open(cpath, "w").write(harness)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} harness build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} emit not behaviour-equivalent ({out})"


def test_multi_declarator_pointer_dual_rail():
    """Per-declarator pointer/array shape in a multi-declarator declaration (#multiptr): in `int *p, q;`
    the `*` binds to the DECLARATOR, not the type-specifier -- p is `int*`, q is `int`; `int *p, *q;`
    types both as pointers; a per-declarator array no longer leaks dims onto the next declarator. The
    oracle typed each declarator individually; the twin folded `*` into the shared specifier, so
    `int *p, q;` mis-typed q as an 8-byte pointer and `int *p, *q;` was rejected. The twin now parses the
    base specifier once and applies each declarator's own `*`/`[]` on a fresh copy (locals + struct
    members). Parity + a bespoke differential that uses each trailing declarator AS a scalar (an 8-byte
    store would clobber an adjacent field). Also pins a wide-scalar member store (`long m`) moving 8
    bytes, not 4."""
    fx = "cfront_multiptr.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["md_local_mixed", "md_local_two_ptr", "md_local_ptr_arr", "md_struct"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""int main(void){
  for(int i=-300;i<300;i++){
    int a=i*3-1,b=7-i;
    if(md_local_mixed_s(i)!=bcir_md_local_mixed(i)){printf("mixed@%d\n",i);return 1;}
    if(md_local_two_ptr_s(a,b)!=bcir_md_local_two_ptr(a,b)){printf("twoptr@%d\n",i);return 1;}
    if(md_local_ptr_arr_s(i)!=bcir_md_local_ptr_arr(i)){printf("ptrarr@%d\n",i);return 1;}
    struct Mix m1,m2;
    if(md_struct_s(&m1,i)!=bcir_md_struct(&m2,i)){printf("struct@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}"
            cpath, epath = os.path.join(d, f"{label}.c"), os.path.join(d, label)
            open(cpath, "w").write(harness)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} harness build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} emit not behaviour-equivalent ({out})"


def test_faithful_char_types_dual_rail():
    """Faithful char types (#chartypes): C's three distinct one-byte char types are emitted faithfully,
    so the output is behaviour-equivalent on every target -- plain `char` -> `char` (implementation-
    defined signedness), `signed char` -> always signed, `unsigned char` -> always unsigned. The oracle
    collapsed `signed char` -> `char` (zero-extending a negative on ARM); the twin emitted int8_t for
    plain `char` (sign-extending on ARM). The harness is built under BOTH -fsigned-char AND
    -funsigned-char, so plain char's platform sign is exercised both ways -- the case the old emit got
    wrong (it would pass under one and fail the other)."""
    fx = "cfront_chartypes.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["ct_plain_deref", "ct_signed_deref", "ct_unsigned_deref", "ct_plain_cmp", "ct_signed_cmp",
             "ct_plain_div", "ct_signed_div", "ct_unsigned_div", "ct_roundtrip", "ct_plain_widen"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""int main(void){
  for(int i=-200;i<200;i++){
    char pc=(char)(i*7-3); signed char sc=(signed char)(i*5+1); unsigned char uc=(unsigned char)(i*3+2);
    if(ct_plain_deref_s(&pc)!=bcir_ct_plain_deref(&pc)){printf("pd@%d\n",i);return 1;}
    if(ct_signed_deref_s(&sc)!=bcir_ct_signed_deref(&sc)){printf("sd@%d\n",i);return 1;}
    if(ct_unsigned_deref_s(&uc)!=bcir_ct_unsigned_deref(&uc)){printf("ud@%d\n",i);return 1;}
    if(ct_plain_cmp_s(pc)!=bcir_ct_plain_cmp(pc)){printf("pc@%d\n",i);return 1;}
    if(ct_signed_cmp_s(sc)!=bcir_ct_signed_cmp(sc)){printf("sc@%d\n",i);return 1;}
    if(ct_plain_div_s(&pc)!=bcir_ct_plain_div(&pc)){printf("pdv@%d\n",i);return 1;}
    if(ct_signed_div_s(&sc)!=bcir_ct_signed_div(&sc)){printf("sdv@%d\n",i);return 1;}
    if(ct_unsigned_div_s(&uc)!=bcir_ct_unsigned_div(&uc)){printf("udv@%d\n",i);return 1;}
    if(ct_roundtrip_s(&pc)!=bcir_ct_roundtrip(&pc)){printf("rt@%d\n",i);return 1;}
    if(ct_plain_widen_s(&pc)!=bcir_ct_plain_widen(&pc)){printf("pw@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}"
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            for charmode in ("-fsigned-char", "-funsigned-char"):       # exercise plain char both ways
                epath = os.path.join(d, f"{label}{charmode}")
                for std in ("c23", "c2x", "c17"):
                    b = subprocess.run([_CC, f"-std={std}", "-O2", charmode, cpath, "-o", epath],
                                       capture_output=True, text=True)
                    if b.returncode == 0:
                        break
                else:
                    raise AssertionError(f"{fx}: {label} {charmode} build failed:\n{b.stderr}")
                out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
                assert out == "MATCH", f"{fx}: {label} {charmode} not behaviour-equivalent ({out})"


def test_compound_literals_dual_rail():
    """Compound literals (#complit): `( type-name ){ init }` is an anonymous object materialized as a
    nameless local and yielded in rvalue position (a by-value struct argument, a scalar value, a member
    initializer), under `&` (a pointer to the temporary), or with direct postfix on the literal
    (`(struct P){...}.field`, incl. a designated/partial init and a wide `long` field) -- struct
    designators (any order) + partial init zero-fill included. Differential == Clang on both rails;
    oracle/twin claim-count parity."""
    fx = "cfront_complit.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["cl_byval", "cl_designated", "cl_partial", "cl_scalar",
             "cl_addr_scalar", "cl_addr_struct", "cl_nested",
             "cl_dot", "cl_dot_desig", "cl_dot_part", "cl_dot_wide"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""int main(void){
  for(int a=-40;a<40;a++) for(int b=-7;b<7;b++){
    if(cl_byval_s(a,b)!=bcir_cl_byval(a,b)){printf("byval@%d,%d\n",a,b);return 1;}
    if(cl_designated_s(a,b)!=bcir_cl_designated(a,b)){printf("desig@%d,%d\n",a,b);return 1;}
    if(cl_partial_s(a)!=bcir_cl_partial(a)){printf("partial@%d\n",a);return 1;}
    if(cl_scalar_s(a)!=bcir_cl_scalar(a)){printf("scalar@%d\n",a);return 1;}
    if(cl_addr_scalar_s(a)!=bcir_cl_addr_scalar(a)){printf("as@%d\n",a);return 1;}
    if(cl_addr_struct_s(a,b)!=bcir_cl_addr_struct(a,b)){printf("ast@%d,%d\n",a,b);return 1;}
    if(cl_nested_s(a)!=bcir_cl_nested(a)){printf("nested@%d\n",a);return 1;}
    if(cl_dot_s(a,b)!=bcir_cl_dot(a,b)){printf("dot@%d,%d\n",a,b);return 1;}
    if(cl_dot_desig_s(a,b)!=bcir_cl_dot_desig(a,b)){printf("dotdes@%d,%d\n",a,b);return 1;}
    if(cl_dot_part_s(a)!=bcir_cl_dot_part(a)){printf("dotpart@%d\n",a);return 1;}
    if(cl_dot_wide_s(a)!=bcir_cl_dot_wide(a)){printf("dotwide@%d\n",a);return 1;}
  }
  printf("MATCH\n");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}"
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath],
                                   capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} not behaviour-equivalent ({out})"


def test_typeof_dual_rail():
    """typeof (#typeof): C23 `typeof(type-name)` / `typeof(variable)` / `typeof(expression)` (+ GNU
    `__typeof__`) as a type-specifier, resolving to the operand's type. Both rails resolve a type-name
    operand (incl. `typeof(int*)`), a bare in-scope variable, and a general expression operand
    (`typeof(a+b)`, `typeof((short)x)`, `typeof(*p)`, `typeof(s.f)`, `typeof(arr[i])`) -- the oracle by
    static type inference, the twin by speculatively lowering then rolling the emission back. Each case
    is built so the WRONG type (int vs long, signed vs unsigned, a missing short truncation) would
    diverge. Differential == Clang on both rails."""
    fx = "cfront_typeof.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["to_width", "to_sign", "to_typename", "to_ptr", "to_struct", "to_unqual",
             "to_ebinop", "to_ebinsign", "to_ecast", "to_ederef", "to_emember", "to_eindex"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""int main(void){
  for(long a=-50;a<50;a++){
    if(to_width_s(a)!=bcir_to_width(a)){printf("width@%ld\n",a);return 1;}
    if(to_sign_s((unsigned)a)!=bcir_to_sign((unsigned)a)){printf("sign@%ld\n",a);return 1;}
    if(to_typename_s((int)a)!=bcir_to_typename((int)a)){printf("tn@%ld\n",a);return 1;}
    if(to_ptr_s((int)a)!=bcir_to_ptr((int)a)){printf("ptr@%ld\n",a);return 1;}
    if(to_struct_s((int)a)!=bcir_to_struct((int)a)){printf("struct@%ld\n",a);return 1;}
    if(to_unqual_s(a)!=bcir_to_unqual(a)){printf("unq@%ld\n",a);return 1;}
    if(to_ebinop_s(a)!=bcir_to_ebinop(a)){printf("ebinop@%ld\n",a);return 1;}
    if(to_ebinsign_s((unsigned)a)!=bcir_to_ebinsign((unsigned)a)){printf("ebinsign@%ld\n",a);return 1;}
    if(to_ecast_s((int)a)!=bcir_to_ecast((int)a)){printf("ecast@%ld\n",a);return 1;}
    if(to_ederef_s(a)!=bcir_to_ederef(a)){printf("ederef@%ld\n",a);return 1;}
    if(to_emember_s((int)a)!=bcir_to_emember((int)a)){printf("emember@%ld\n",a);return 1;}
    if(to_eindex_s(a)!=bcir_to_eindex(a)){printf("eindex@%ld\n",a);return 1;}
  }
  printf("MATCH\n");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}"
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath],
                                   capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} not behaviour-equivalent ({out})"


def test_struct_member_init_dual_rail():
    """Struct-valued member set (#structinit): a struct/union-typed member assigned a whole struct value
    -- in an aggregate initializer (`{ inner, ... }` / `{ (struct Pt){...}, ... }`) or by a direct
    member assignment (`o.p = q`). The member store copies the whole object (a memcpy of the member's
    size); a scalar `uintN _v = <struct>` is a type error and under-reads a wide member (the twin's emit
    previously failed to compile here). Differential == Clang on both rails; oracle/twin claim parity."""
    fx = "cfront_structinit.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["si_var", "si_lit", "si_wide", "si_assign"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""int main(void){
  for(int a=-40;a<40;a++) for(int b=-40;b<40;b++){
    if(si_var_s(a,b)!=bcir_si_var(a,b)){printf("var@%d,%d\n",a,b);return 1;}
    if(si_lit_s(a,b)!=bcir_si_lit(a,b)){printf("lit@%d,%d\n",a,b);return 1;}
    if(si_wide_s(a,b)!=bcir_si_wide(a,b)){printf("wide@%d,%d\n",a,b);return 1;}
    if(si_assign_s(a,b)!=bcir_si_assign(a,b)){printf("assign@%d,%d\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}"
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath],
                                   capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} not behaviour-equivalent ({out})"


def test_array_compound_literals_dual_rail():
    """Array compound literals (#arraylit): an anonymous array `(T[N]){...}` / `(T[]){...}` subscripted at
    the use site -- the inline lookup-table idiom `(int[]){...}[i]`. An inferred `[]` size comes from the
    initializer (max index + 1); the typed element store converts to any scalar element (int / char / long);
    `[i]=` designators + positional entries mix (gaps zero-fill). Differential == Clang on BOTH the twin's
    and the oracle's emit, with oracle/twin claim-count parity."""
    fx = "cfront_arraylit.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["weekday", "sized", "charlut", "desig", "flut", "widelut"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    cmps = "\n".join(
        f'    if({f}_s(i)!=bcir_{f}(i)){{printf("{f}@%u\\n",i);return 1;}}' for f in funcs)
    driver = ("int main(void){\n"
              "  for(unsigned i=0;i<60u;i++){\n"
              f"{cmps}\n"
              "  }\n"
              '  printf("MATCH\\n");return 0;}')
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}"
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath],
                                   capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} not behaviour-equivalent ({out})"


def test_compound_wide_dual_rail():
    """Wide / floating compound assignment (#compoundwide): an `OP=` (or ++/--) on a long/double lvalue
    keeps its operand width/float-ness in the result instead of truncating to a 4-byte uint32 -- across a
    local, a struct member, an array element, and a pointer deref -- and the float/wide-int variadic
    accumulation (`s += va_arg(ap, double)` / `va_arg(ap, long)`) it unblocks. Twin-only fix (the oracle
    was already correct); differential == Clang on BOTH emits over values that overflow 32 bits."""
    fx = "cfront_compoundwide.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["l_local", "d_local", "l_inc", "l_member", "l_array", "l_ptr", "d_vararg", "l_vararg", "driver"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""int main(void){
  long V[]={0,1,-1,1000000000L,-1000000000L,5000000000L,-5000000000L,99999999999L};
  int n=(int)(sizeof V/sizeof V[0]);
  for(int i=0;i<n;i++) for(int j=0;j<n;j++){ long x=V[i],y=V[j];
    if(driver_s(x,y)!=bcir_driver(x,y)){printf("driver@%ld,%ld\n",x,y);return 1;}
    if(l_local_s(x,y)!=bcir_l_local(x,y)){printf("l_local@%ld,%ld\n",x,y);return 1;}
    if(l_member_s(x,y)!=bcir_l_member(x,y)){printf("l_member@%ld,%ld\n",x,y);return 1;}
    if(l_ptr_s(x,y)!=bcir_l_ptr(x,y)){printf("l_ptr@%ld,%ld\n",x,y);return 1;}
    if(d_local_s((double)x,(double)y)!=bcir_d_local((double)x,(double)y)){printf("d_local@%ld,%ld\n",x,y);return 1;}
    if(d_vararg_s(3,(double)x,(double)y,1.5)!=bcir_d_vararg(3,(double)x,(double)y,1.5)){printf("d_vararg@%ld,%ld\n",x,y);return 1;}
    if(l_vararg_s(3,x,y,x+y)!=bcir_l_vararg(3,x,y,x+y)){printf("l_vararg@%ld,%ld\n",x,y);return 1;}
  }
  printf("MATCH\n");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = (f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
                       f"#include <stdarg.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}")
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath],
                                   capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} not behaviour-equivalent ({out})"


def test_stmtexpr_dual_rail():
    """GCC statement expressions (#stmtexpr): `({ s1; ...; e; })` -- a compound statement in its own scope
    whose value is the last (expression) statement. The prefix statements lower inline; the result is the
    last expression's value. The twin (no AST) lowers the prefix in place, then rolls the last statement
    back (the typeof speculative undo) and re-parses it as the value. Covers the temporary idiom, the
    safe-max macro, embedding in a larger expression, a loop inside, nesting, and scope shadowing.
    Differential == Clang on BOTH emits."""
    fx = "cfront_stmtexpr.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["se_simple", "se_max", "se_embed", "se_loop", "se_nest", "se_scope", "se_void"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""int main(void){
  for(int a=-60;a<60;a++) for(int b=-25;b<25;b++){
    if(se_simple_s(a)!=bcir_se_simple(a)){puts("simple");return 1;}
    if(se_max_s(a,b)!=bcir_se_max(a,b)){puts("max");return 1;}
    if(se_embed_s(a)!=bcir_se_embed(a)){puts("embed");return 1;}
    if(se_loop_s(b)!=bcir_se_loop(b)){puts("loop");return 1;}
    if(se_nest_s(a)!=bcir_se_nest(a)){puts("nest");return 1;}
    if(se_scope_s(a)!=bcir_se_scope(a)){puts("scope");return 1;}
    if(se_void_s(a)!=bcir_se_void(a)){puts("void");return 1;}
  }
  puts("MATCH");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}"
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath],
                                   capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} not behaviour-equivalent ({out})"


def test_builtins_dual_rail():
    """GCC/Clang integer builtins (#builtins): __builtin_popcount/clz/ctz/ffs/parity/bswap/abs and their
    l/ll variants -- emitted verbatim (the libm-call mold: opaque to R18, no bcir_ twin) with a fixed
    result type, instead of a synthesized `bcir___builtin_popcount` that tripped R18. Both rails emit the
    same builtin; differential == Clang on BOTH emits (clz/ctz operands forced non-zero; abs avoids INT_MIN)."""
    fx = "cfront_builtins.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["bi_pop", "bi_clz", "bi_ffs", "bi_bswap", "bi_bswap64", "bi_abs"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""int main(void){
  for(long i=-40000;i<40000;i+=3){ unsigned x=(unsigned)(i*131071); int xi=(int)i;
    unsigned long long w=(unsigned long long)x * 2654435761ULL + (unsigned)xi;
    if(bi_pop_s(x)!=bcir_bi_pop(x)){puts("pop");return 1;}
    if(bi_clz_s(x)!=bcir_bi_clz(x)){puts("clz");return 1;}
    if(bi_ffs_s(xi)!=bcir_bi_ffs(xi)){puts("ffs");return 1;}
    if(bi_bswap_s(x)!=bcir_bi_bswap(x)){puts("bswap");return 1;}
    if(bi_bswap64_s(w)!=bcir_bi_bswap64(w)){puts("bswap64");return 1;}
    if(bi_abs_s(xi)!=bcir_bi_abs(xi)){puts("abs");return 1;}
  }
  puts("MATCH");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}"
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath],
                                   capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} not behaviour-equivalent ({out})"


def test_atomic_local_dual_rail():
    """_Atomic local objects (#atomiclocal): `_Atomic int a;` (qualifier) and `_Atomic(int) a;` (type
    specifier), `const _Atomic`, and a pointer-to-atomic. A LOCAL was rejected (the statement-level
    decl detector omitted `_Atomic`, and the `_Atomic(T)` paren spelling was unparsed); the global form
    already worked. An automatic-storage _Atomic local is unshared, so its single-threaded semantics equal
    the plain type -- both rails lower the arithmetic identically; differential == Clang on BOTH emits."""
    fx = "cfront_atomiclocal.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["a_qual", "a_paren", "a_long", "a_const", "a_ptr"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""int main(void){
  for(int x=-300;x<300;x++){ long b=(long)x*100000;
    if(a_qual_s(x)!=bcir_a_qual(x)){puts("qual");return 1;}
    if(a_paren_s(x)!=bcir_a_paren(x)){puts("paren");return 1;}
    if(a_long_s(b)!=bcir_a_long(b)){puts("long");return 1;}
    if(a_const_s(x)!=bcir_a_const(x)){puts("const");return 1;}
    if(a_ptr_s(x)!=bcir_a_ptr(x)){puts("ptr");return 1;}
  }
  puts("MATCH");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = (f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
                       f"#include <stdatomic.h>\n{renamed}\n{emit}\n{driver}")
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath],
                                   capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} not behaviour-equivalent ({out})"


def test_addrmember_dual_rail():
    """Address-of a (nested) struct member (#addrmember): `&s.field`, `&t.q.a`, `&t.q`. The twin couldn't
    parse `&member` at all; the oracle emitted the enclosing struct's address (right only for a first
    member at offset 0). Now `&member` resolves to a typed `(T *)((char *)&base + off)` -- used through a
    pointer (read/write/compound-assign) and passed to a helper. Differential == Clang on BOTH emits."""
    fx = "cfront_addrmember.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["addone", "am_first", "am_nested", "am_struct", "am_arg"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""int main(void){
  for(int x=-200;x<200;x++){
    if(am_first_s(x)!=bcir_am_first(x)){puts("first");return 1;}
    if(am_nested_s(x)!=bcir_am_nested(x)){puts("nested");return 1;}
    if(am_struct_s(x)!=bcir_am_struct(x)){puts("struct");return 1;}
    if(am_arg_s(x)!=bcir_am_arg(x)){puts("arg");return 1;}
  }
  puts("MATCH");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}"
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath],
                                   capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} not behaviour-equivalent ({out})"


def test_nestoffset_dual_rail():
    """Nested member access at a NON-FIRST offset (#nestoffset): `t.q.a` where the enclosing member `q`
    is not the struct's first member -- two bugs the #designate follow-on surfaced. The oracle's `_addr`
    dropped the enclosing member's byte offset (so `t.p`/`t.q` aliased at 0 on read AND write); the twin
    over-aligned a nested value-struct member to its SIZE (`struct{int;struct Big t;}` placed t at
    sizeof(Big), not its alignment), shifting every later offset. cfront_nestmember only nested through a
    first member, hiding both. Differential == Clang on BOTH emits over read/write, a member array, a
    deeper chain, and a non-first designated initializer (the #designate read-back)."""
    fx = "cfront_nestoffset.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["no_rw", "no_memarr", "no_deep", "no_desig"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""int main(void){
  for(int x=-300;x<300;x++){
    if(no_rw_s(x)!=bcir_no_rw(x)){printf("rw@%d\n",x);return 1;}
    if(no_memarr_s(x)!=bcir_no_memarr(x)){printf("memarr@%d\n",x);return 1;}
    if(no_deep_s(x)!=bcir_no_deep(x)){printf("deep@%d\n",x);return 1;}
    if(no_desig_s(x)!=bcir_no_desig(x)){printf("desig@%d\n",x);return 1;}
  }
  puts("MATCH");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}"
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath],
                                   capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} not behaviour-equivalent ({out})"


def test_designate_dual_rail():
    """Nested / chained designated initializers (#designate): a designator LIST `.a.b`, `.v[i]`,
    `.m[i][j]` (and deeper) in an aggregate initializer, resolving to a cumulative byte offset -- `.field`
    descends a (nested value-)struct/union member, `[i]` folds a constant index into a member array.
    Both rails walk the layout identically; differential == Clang on BOTH emits (read-back here uses only
    first-member nesting -- a non-first nested read needs the member-access offset fix, a follow-on)."""
    fx = "cfront_designate.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["desig_chain", "desig_memarr", "desig_md", "desig_mix", "desig_deep"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""int main(void){
  for(int x=-300;x<300;x++){
    if(desig_chain_s(x)!=bcir_desig_chain(x)){printf("chain@%d\n",x);return 1;}
    if(desig_memarr_s(x)!=bcir_desig_memarr(x)){printf("memarr@%d\n",x);return 1;}
    if(desig_md_s(x)!=bcir_desig_md(x)){printf("md@%d\n",x);return 1;}
    if(desig_mix_s(x)!=bcir_desig_mix(x)){printf("mix@%d\n",x);return 1;}
    if(desig_deep_s(x)!=bcir_desig_deep(x)){printf("deep@%d\n",x);return 1;}
  }
  puts("MATCH");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}"
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath],
                                   capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} not behaviour-equivalent ({out})"


def test_generic_dual_rail():
    """_Generic (#generic): C11 generic selection on the static type of the UNEVALUATED controlling
    expression -- the first matching type-name (int/int32_t collapse by width+sign; plain char distinct
    from signed/unsigned char; floats key on width; pointer on pointee) wins, else `default`, and only the
    chosen arm is lowered. Both rails read the controlling type the same way and pick the same arm;
    differential == Clang on BOTH emits over int/long/unsigned/double/float/char/pointer controls."""
    fx = "cfront_generic.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["g_int", "g_long", "g_uint", "g_double", "g_float", "g_char", "g_ptr",
             "g_exprtype", "g_default", "g_compute"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""int main(void){
  for(int i=-200;i<200;i++){ long b=(long)i*7777; int x=i;
    if(g_int_s(i)!=bcir_g_int(i)){puts("g_int");return 1;}
    if(g_long_s(b)!=bcir_g_long(b)){puts("g_long");return 1;}
    if(g_uint_s((unsigned)i)!=bcir_g_uint((unsigned)i)){puts("g_uint");return 1;}
    if(g_double_s((double)i)!=bcir_g_double((double)i)){puts("g_double");return 1;}
    if(g_float_s((float)i)!=bcir_g_float((float)i)){puts("g_float");return 1;}
    if(g_char_s((char)i)!=bcir_g_char((char)i)){puts("g_char");return 1;}
    if(g_ptr_s(&x)!=bcir_g_ptr(&x)){puts("g_ptr");return 1;}
    if(g_exprtype_s(i,b)!=bcir_g_exprtype(i,b)){puts("g_exprtype");return 1;}
    if(g_default_s((double)i)!=bcir_g_default((double)i)){puts("g_default");return 1;}
    if(g_compute_s(b)!=bcir_g_compute(b)){puts("g_compute");return 1;}
  }
  puts("MATCH");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}"
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath],
                                   capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} not behaviour-equivalent ({out})"


def test_long_double_dual_rail():
    """long double (#longdouble): the extended floating type (80-bit / ABI-sized). The twin emits real
    `long double` C -- like float/double, it lets the backend do the arithmetic -- closing a parity gap
    (the twin previously could not parse `long double`; the oracle already supported it). Covers
    arithmetic, an `L` constant, conversions to/from double and int, a `long double *`, the `+l` libm
    variants (sqrtl/fabsl), and `+=` accumulation. Differential == Clang on BOTH emits; oracle/twin parity."""
    fx = "cfront_longdouble.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["ld_arith", "ld_promote", "ld_to_int", "ld_narrow", "ld_libm", "ld_ptr", "ld_acc"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""int main(void){
  for(int i=-300;i<300;i++){ long double a=(long double)i*0.3L, b=(long double)(i+5)*0.13L;
    if(ld_arith_s(a,b)!=bcir_ld_arith(a,b)){printf("arith@%d\n",i);return 1;}
    if(ld_promote_s((double)i*0.5,i)!=bcir_ld_promote((double)i*0.5,i)){printf("promote@%d\n",i);return 1;}
    if(ld_to_int_s(a,b)!=bcir_ld_to_int(a,b)){printf("toint@%d\n",i);return 1;}
    if(ld_narrow_s(a)!=bcir_ld_narrow(a)){printf("narrow@%d\n",i);return 1;}
    if(i>=0 && ld_libm_s(a)!=bcir_ld_libm(a)){printf("libm@%d\n",i);return 1;}
    if(ld_ptr_s(a)!=bcir_ld_ptr(a)){printf("ptr@%d\n",i);return 1;}
    if(ld_acc_s(i%17,a)!=bcir_ld_acc(i%17,a)){printf("acc@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = (f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
                       f"#include <math.h>\n{renamed}\n{emit}\n{driver}")
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath, "-lm"],
                                   capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} not behaviour-equivalent ({out})"


def test_extern_variadic_dual_rail():
    """External variadic calls (#extvariadic): the printf/scanf-family <stdio.h> variadics (snprintf /
    vsnprintf) emit verbatim and stay opaque to the R18 call graph (no bcir_ twin), returning int; the
    read-only format string passes through as an argument; a vsnprintf-forwarding wrapper hands its own
    va_list cursor to the external. The differential compares the formatted BUFFER and the returned count
    == Clang on BOTH the twin's and the oracle's emit (the real libc produces identical bytes)."""
    fx = "cfront_extvariadic.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["ev_int", "ev_mix", "ev_width", "ev_fwd", "ev_call"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""int main(void){
  for(int x=-3000;x<3000;x++){ long y=(long)x*1234567L; char b1[64],b2[64]; int r1,r2;
    r1=ev_int_s(b1,x);   r2=bcir_ev_int(b2,x);   if(r1!=r2||strcmp(b1,b2)){printf("int@%d\n",x);return 1;}
    r1=ev_mix_s(b1,x,y); r2=bcir_ev_mix(b2,x,y); if(r1!=r2||strcmp(b1,b2)){printf("mix@%d\n",x);return 1;}
    r1=ev_width_s(b1,x); r2=bcir_ev_width(b2,x); if(r1!=r2||strcmp(b1,b2)){printf("width@%d\n",x);return 1;}
    r1=ev_call_s(b1,x,(int)(y&0xff)); r2=bcir_ev_call(b2,x,(int)(y&0xff));
    if(r1!=r2||strcmp(b1,b2)){printf("call@%d\n",x);return 1;}
  }
  printf("MATCH\n");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = (f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
                       f"#include <stdarg.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}")
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath],
                                   capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} not behaviour-equivalent ({out})"


def test_variadic_dual_rail():
    """Variadic functions (#variadic): `f(T last, ...)` with <stdarg.h> -- a `va_list` cursor walked by
    va_start/va_arg/va_end, va_copy (two passes), a `va_list` parameter (vprintf-style forwarding), and a
    same-unit variadic call passing args past the fixed params (default promotions ride the real call).
    va_start/va_arg/va_end/va_copy lower as opaque builtins emitted verbatim; `va_arg(ap, T)` carries
    type T. Differential == Clang on BOTH the twin's and the oracle's emit, with oracle/twin parity."""
    fx = "cfront_variadic.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = ["isum", "twice", "vsumv", "forward", "nth", "caller"]
    renamed = src
    for f in funcs:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""int main(void){
  for(int a=-40;a<40;a++) for(int b=-40;b<40;b++){ int c=a-2*b;
    if(caller_s(a,b,c)!=bcir_caller(a,b,c)){printf("caller@%d,%d\n",a,b);return 1;}
    if(isum_s(4,a,b,c,a+b)!=bcir_isum(4,a,b,c,a+b)){printf("isum@%d,%d\n",a,b);return 1;}
    if(twice_s(3,a,b,c)!=bcir_twice(3,a,b,c)){printf("twice@%d,%d\n",a,b);return 1;}
    if(forward_s(3,a,b,c)!=bcir_forward(3,a,b,c)){printf("forward@%d,%d\n",a,b);return 1;}
    if(nth_s(2,(double)a,(double)b,(double)c)!=bcir_nth(2,(double)a,(double)b,(double)c)){printf("nth@%d,%d\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = (f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
                       f"#include <stdarg.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}")
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run([_CC, f"-std={std}", "-O2", cpath, "-o", epath],
                                   capture_output=True, text=True)
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} not behaviour-equivalent ({out})"


def _build_loop(d: str) -> str:
    return _compile_once("loop", "loop",
                         ("bcir_cfront.c", "bcir_cpp.c", "bcir_plan.c", "bcir_hydrate.c", "bcir_exec.c",
                          "bcir_runtime.c", "bcir_verify.c", "test_cfront_loop.c"), "loop")


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
    "cfront_cmpxchg11.c": ["atomic_compare_exchange_strong", "atomic_compare_exchange_weak", "_Atomic"],
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
{_BOUNDS_GUARD}
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
            b = subprocess.run([_CC, f"-std={std}", "-O2", c, "-o", e, "-lm"],   # -lm: <math.h> links
                               capture_output=True, text=True)
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
    # (mmio=4: the real `switch` lowers the MMIO status discriminant once, where the old if/else-if
    # desugar re-read it per case label.)
    assert "mmio=4" in oracle_summary and "bf=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        # the emit carries each composed register feature in one verified-C unit (the status `switch`
        # now renders as a real C `switch`, not an if/else-if desugar).
        for needle in ("volatile uint32_t *", "QUANTA[", "static uint32_t halts",
                       "switch (", "case ", "BCIR verified-C attestation"):
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
    return _compile_once("bcir_cc", "bcir-cc",
                         ("bcir_cc.c", "bcir_cpp.c", "bcir_cfront.c", "bcir_verify.c", "bcir_runtime.c",
                          "bcir_plan.c", "bcir_hydrate.c"), "bcir-cc")


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
                 "bcir_cc.c", "bcir_diag.c"):
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


_ABI_TARGETS = ["x86_64-linux", "aarch64-linux", "riscv64-linux", "x86_64-windows", "i386-linux"]


def _abi_const_vec_oracle(src: str, target: str):
    """The ordered sizeof immediates the oracle folds under `target` (the data-model vector)."""
    r = compile_unit(src, check_clang=False, target=target)
    lf = r.lowered.functions[next(reversed(r.lowered.functions))]
    return [c.imm[0] for c in lf.claims if c.op == "c.const"]


def _abi_const_vec_twin(exe: str, path: str, target: str):
    """The same vector read off the C twin's emitted C (each sizeof const is a `= Nu;` literal)."""
    out = subprocess.run([exe, "--target", target, path], capture_output=True, text=True).stdout
    _summary, _, emit = out.partition("----EMIT----\n")
    return [int(m) for m in re.findall(r"=\s*(\d+)u;", emit)]


def test_abi_target_matrix_dual_rail():
    """The cross-platform target-ABI matrix (#abi, the C twin of frontends/cfront/abi.py): the C
    frontend's `--target` data model lays `long` / the pointer / the `size_t`-class types out exactly
    like the oracle's `TargetABI`, for every named target. The structural summary is target-invariant,
    so this compares the FOLDED `sizeof` constants (which carry long_size / pointer_size): the two
    rails agree per target, and the LP64 / LLP64 / ILP32 vectors are distinct so the gate has teeth.
    `cfront_abi.c` folds [sizeof long, void*, size_t, int, long long]."""
    path = os.path.join(_C, "cfront_abi.c")
    src = open(path, encoding="utf-8").read()
    # oracle side always runs (quick tier too): the matrix spans the three data models.
    vecs = {t: _abi_const_vec_oracle(src, t) for t in _ABI_TARGETS}
    assert vecs["x86_64-linux"] == [8, 8, 8, 4, 8]                       # LP64
    assert vecs["aarch64-linux"] == vecs["riscv64-linux"] == [8, 8, 8, 4, 8]
    assert vecs["x86_64-windows"] == [4, 8, 8, 4, 8]                     # LLP64: long is 4
    assert vecs["i386-linux"] == [4, 4, 4, 4, 8]                         # ILP32: pointers are 4 too
    assert len({tuple(v) for v in vecs.values()}) == 3                   # three distinct data models
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        for t in _ABI_TARGETS:
            assert _abi_const_vec_twin(exe, path, t) == vecs[t], \
                f"ABI {t}: twin {_abi_const_vec_twin(exe, path, t)} != oracle {vecs[t]}"
        # an unknown target is a clean diagnostic + nonzero exit on the C rail (not a crash).
        bad = subprocess.run([exe, "--target", "sparc-solaris", path], capture_output=True, text=True)
        assert bad.returncode != 0 and "unknown target" in bad.stdout, bad.stdout


# (source, the expected total-compile outcome): "clean" compiles+verifies, "dirty" compiles but the
# verifier rejects it (R18 recursion), "fallback" is outside the supported subset (route to LLVM).
_FALLBACK_PROBES = [
    ("unsigned f(unsigned x){ return x*2u + 1u; }", "clean"),
    ("unsigned f(unsigned n){ return f(n-1u); }", "dirty"),              # R18 recursion -> DIRTY, not fallback
    ("unsigned f(unsigned x){ return x + ; }", "fallback"),              # malformed -> parse reject
    ("unsigned f(void){ _Complex double z; (void)z; return 0u; }", "clean"),  # _Complex: now in the subset
    ("unsigned f(void){ _Imaginary double z; return 0u; }", "fallback"),  # _Imaginary: still outside the subset
    ("unsigned f(unsigned n){ unsigned a[n][n][n][n]; return a[0][0][0][0]; }", "fallback"),   # a >3-D VLA:
                     # 1-D / 2-D / 3-D stack VLAs are now natively lowered (#vla / #vlamd -- snapshot each
                     # runtime dim, flatten to m*n, mask the Horner index), but >3 dims defers to fallback
                     # on both rails (the dim table caps at 3).
    ("unsigned f(unsigned x){ return ({ unsigned y=x; y+1u; }); }", "clean"),  # statement-expr (#stmtexpr): now native
    ("unsigned f(unsigned a){ a = a*3u + 1u; return a; }", "clean"),       # assigning a PARAMETER: a bare
                     # `a = ..;` in the emit, never a `uint32_t a = ..;` redeclaration (twin-emit regression).
    ("unsigned f(unsigned a){ return ({ a++; }); }", "fallback"),          # `i++` as a stmt-expr VALUE: the
                     # post/pre distinction was discarded in the desugar, so it routes away on both rails.
    ("unsigned f(unsigned a){ return ({ a = a+1u; }); }", "fallback"),     # an assignment as a stmt-expr value
                     # (the twin's value-expression grammar has none) -> fallback on both rails, in lockstep.
    ("unsigned f(unsigned x){ void *p=&&L; goto *p; L: return x; }", "fallback"),  # computed goto
    ("struct Q{unsigned*p; unsigned n;}; unsigned f(struct Q q,unsigned i){ return q.p[i&3u]+q.n; }",
     "clean"),      # a *pointer* member indexed (`q.p[i]` == `*(q.p + i)`): both rails now load the full
                     # pointer field and subscript the loaded pointer (#fieldderef, pointer-value slice 2b).
                     # (1-D..3-D member *arrays* are native -- #memberarray; deref-through is now native too.)
    ("unsigned f(unsigned i){ unsigned m[2][2][2][2]; m[0][0][0][0]=1u; return m[i&1u][0][0][0]; }",
     "fallback"),   # a >3-dimensional local array: 1-D..3-D locals are now natively lowered (a flat
                     # resource of the product of dims + the per-dim flatten shape), but the dim table
                     # holds 3, so 4-D+ defers to fallback on both rails (the subset stays pinned).
]
_FALLBACK_RC = {"clean": 0, "dirty": 1, "fallback": 2}


def _oracle_fallback_rc(src: str) -> int:
    """The oracle's total-compile outcome as an exit code: needs_fallback=2 / not-clean=1 / clean=0
    (the contract `bcir-cc --fallback` mirrors)."""
    from bcir.frontends.cfront.pipeline import compile_with_fallback  # noqa: PLC0415
    r = compile_with_fallback(src, check_clang=False)
    return 2 if r.needs_fallback else (0 if r.is_clean else 1)


def test_fallback_contract_dual_rail():
    """The total-compilation / fallback contract (#fallback): `bcir-cc --fallback` is the C twin of
    `pipeline.compile_with_fallback` -- a **total** entry point that never crashes on a construct
    outside the supported subset, instead exiting 2 ("fallback to LLVM backend") so a driver can route
    the unit to the resident backend. A unit that compiles + verifies exits 0; one that compiles but
    the verifier rejects (R18 recursion) exits 1 (DIRTY, NOT a fallback). The three-way outcome agrees
    with the oracle across the probe set -- which pins the two rails' supported subset to coincide
    (a construct one rail silently accepted while the other routed away would diverge here)."""
    # oracle side always runs (quick tier too); confirm the probe set actually spans all three.
    oracle = {src: _oracle_fallback_rc(src) for src, _ in _FALLBACK_PROBES}
    for src, want in _FALLBACK_PROBES:
        assert oracle[src] == _FALLBACK_RC[want], f"oracle {oracle[src]} != {want} for {src!r}"
    assert set(oracle.values()) == {0, 1, 2}                            # spans clean / dirty / fallback
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        cc = _build_bcir_cc(d)
        for src, want in _FALLBACK_PROBES:
            p = os.path.join(d, "u.c")
            with open(p, "w") as f:
                f.write(src + "\n")
            got = subprocess.run([cc, "--fallback", p], capture_output=True, text=True)
            assert got.returncode == _FALLBACK_RC[want], \
                f"twin rc={got.returncode} != {want}({_FALLBACK_RC[want]}) for {src!r}\n{got.stderr}"
            if want == "fallback":
                assert "fallback to LLVM backend" in got.stderr, got.stderr


def test_member_array_oracle_emit_is_clang_equivalent():
    """The *oracle's own* emitted C for a 1-D struct member array must compile and be Clang-equivalent --
    the per-fixture differential compiles the C *twin's* emit, so the oracle emitter was unguarded here.
    Critically includes a member array at offset 0 (the first member): the access must still carry the
    (member offset, element size) imm, not collapse to an invalid `struct[idx]` -- the regression this
    pins. `compile_unit(check_clang=True)` builds and diffs the oracle's emit against the source."""
    if not _CC:
        return
    for src in (
        # member array at offset 0 (the first member) -- the regression
        "struct B{unsigned a[4]; unsigned n;}; unsigned f(unsigned i,unsigned v){ struct B b; b.n=v;"
        " for(unsigned t=0u;t<4u;t++) b.a[t]=v+t; return b.a[i&3u]+b.n; }",
        # a uint8 element array at offset 0 (a narrowing store, byte stride)
        "struct C{unsigned char d[6]; unsigned k;}; unsigned f(unsigned i,unsigned v){ struct C c; c.k=v;"
        " for(unsigned t=0u;t<6u;t++) c.d[t]=(unsigned char)(v*t); return (unsigned)c.d[i%6u]+c.k; }",
        # a member array at a non-zero offset
        "struct D{unsigned n; unsigned a[4];}; unsigned f(unsigned i,unsigned v){ struct D d; d.n=v;"
        " for(unsigned t=0u;t<4u;t++) d.a[t]=v+t; return d.a[i&3u]+d.n; }",
    ):
        r = compile_unit(src, check_clang=True)
        assert r.is_clean, f"oracle not clean: {src!r}"
        assert r.equivalence == "match", f"oracle emit not Clang-equivalent ({r.equivalence}): {src!r}"


def _build_diag(d: str) -> str:
    """Build the bcir_diag renderer harness (test_diag.c + bcir_diag.c)."""
    exe = os.path.join(d, "tdiag")
    srcs = [os.path.join(_C, s) for s in ("bcir_diag.c", "test_diag.c")]
    for std in ("c23", "c11"):
        b = subprocess.run([_CC, f"-std={std}", "-O2", "-I", _C, *srcs, "-o", exe],
                           capture_output=True, text=True)
        if b.returncode == 0:
            return exe
    raise AssertionError(f"bcir_diag build failed:\n{b.stderr}")


def _diag_spec(primary, notes):
    """The tab-separated spec the C harness reads (start == end == -1 -> a spanless banner; a leading
    "-" marks a note, since a primary's severity may itself be "note")."""
    sev, (s, e), msg = primary
    lines = [f"{sev}\t{s}\t{e}\t{msg}"]
    for (a, b), m in notes:
        lines.append(f"-\t{a}\t{b}\t{m}")
    return "\n".join(lines) + "\n"


def test_diagnostic_renderer_dual_rail():
    """Clang-grade diagnostics (#diag): the C source-location model + caret renderer (bcir_diag.c) is
    the C twin of cfront/diagnostics.py. Fed the SAME synthetic diagnostic (severity / message / byte
    span, plus notes) over the same source, the C renderer's Clang-layout output -- the
    `file:line:col: severity: message` banner, the source line, and the `^~~~` underline (leading tabs
    reproduced so the caret aligns) -- is byte-identical to `diagnostics.render()`. The two rails thus
    share one diagnostic format, independent of which parser produced the error (the messages are not
    shared; the LAYOUT is). Covers spanned / spanless / zero-width / past-EOF spans, multi-line
    sources, tab-indented lines, multi-column underlines, and attached notes."""
    from bcir.frontends.cfront.diagnostics import (  # noqa: PLC0415
        SourceDiagnostic, Span, Note, render)
    src_a = "unsigned f(unsigned x){ return x + ; }\n"
    src_b = "int main(void)\n{\n\treturn foo(1, 2);\n}\n"     # a tab-indented line
    src_c = "a\nbb\nccc\n"
    cases = [
        (src_a, "u.c", ("error", (34, 35), "expected ';'"), []),
        (src_a, "u.c", ("error", (-1, -1), "file-level problem"), []),       # spanless banner
        (src_a, "u.c", ("warning", (9, 10), "odd parameter name"), []),
        (src_a, "u.c", ("error", (34, 34), "zero-width insertion point"), []),
        (src_a, "u.c", ("error", (40, 41), "past end of file"), []),
        (src_b, "m.c", ("error", (19, 22), "implicit declaration of 'foo'"),  # tab line + a note
         [((4, 8), "expanded from macro here")]),
        (src_c, "t.c", ("error", (4, 7), "underline runs to end of line"), []),  # line 2, multi-col
        (src_c, "t.c", ("note", (8, 9), "on the last line"), []),
        (src_b, "m.c", ("error", (-1, -1), "no primary span"),               # spanless + mixed notes
         [((19, 20), "see here"), ((-1, -1), "and a spanless note")]),
    ]

    def py_render(src, fn, primary, notes):
        sev, (s, e), msg = primary
        span = None if (s == -1 and e == -1) else Span(s, e)
        nlist = [Note(m, None if (a == -1 and b == -1) else Span(a, b)) for (a, b), m in notes]
        return render(SourceDiagnostic(sev, msg, span=span, notes=nlist), src, fn)

    # the oracle side always runs (quick tier too): the renderer is pure-Python and deterministic.
    for src, fn, primary, notes in cases:
        assert isinstance(py_render(src, fn, primary, notes), str)
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_diag(d)
        for src, fn, primary, notes in cases:
            sp = os.path.join(d, "s.c")
            with open(sp, "w") as f:
                f.write(src)
            c_out = subprocess.run([exe, sp, fn], input=_diag_spec(primary, notes),
                                   capture_output=True, text=True).stdout
            assert c_out == py_render(src, fn, primary, notes), \
                f"diag layout diverged for {fn} {primary}\n C: {c_out!r}\nPY: {py_render(src, fn, primary, notes)!r}"


def test_diagnostic_json_dual_rail():
    """The machine-readable (JSON) diagnostics feed (#diag): `bcir_diag_to_json` is the C twin of
    DiagnosticReport.to_json() (a `-fdiagnostics-format=json`-style feed). Over the same diagnostics
    its output is byte-identical to Python's `json.dumps(indent=2)` -- the same 2-space indentation,
    member order (severity / message / phase / file:line:column / range / notes), nested range and
    note objects, JSON string escaping, and the spanless-vs-spanned location shape. Covers a single
    spanned diagnostic with a note, a spanless banner, escaped characters in the message, a
    multi-element array, and the empty array."""
    from bcir.frontends.cfront.diagnostics import (  # noqa: PLC0415
        SourceDiagnostic, Span, Note, DiagnosticReport)
    src_a = "unsigned f(unsigned x){ return x + ; }\n"
    src_b = "int main(void)\n{\n\treturn foo(1, 2);\n}\n"
    cases = [
        (src_a, "u.c", [("error", (34, 35), "expected ';'", [])]),
        (src_a, "u.c", [("warning", (-1, -1), "file-level problem", [])]),
        (src_b, "m.c", [("error", (19, 22), "implicit declaration of 'foo'",
                         [((4, 8), "expanded from macro here")])]),
        (src_a, "u.c", [("error", (0, 3), 'quote" and back\\slash and a\ttab', [])]),  # JSON escapes
        (src_b, "m.c", [("warning", (-1, -1), "first", []),
                        ("error", (19, 20), "second", [((-1, -1), "spanless note")])]),  # 2-element array
        (src_a, "u.c", []),                                                  # the empty array -> "[]"
    ]

    def spec(diags):
        lines = []
        for sev, (s, e), msg, notes in diags:
            lines.append(f"{sev}\t{s}\t{e}\t{msg}")
            for (a, b), m in notes:
                lines.append(f"-\t{a}\t{b}\t{m}")
        return ("\n".join(lines) + "\n") if lines else ""

    def py_json(src, fn, diags):
        dl = []
        for sev, (s, e), msg, notes in diags:
            span = None if (s == -1 and e == -1) else Span(s, e)
            nl = [Note(m, None if (a == -1 and b == -1) else Span(a, b)) for (a, b), m in notes]
            dl.append(SourceDiagnostic(sev, msg, span=span, notes=nl, phase="parse"))
        return DiagnosticReport(dl, src, fn).to_json()

    for src, fn, diags in cases:
        assert py_json(src, fn, diags).startswith("[")                      # oracle side always runs
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_diag(d)
        for src, fn, diags in cases:
            sp = os.path.join(d, "s.c")
            with open(sp, "w") as f:
                f.write(src)
            c_out = subprocess.run([exe, "--json", sp, fn], input=spec(diags),
                                   capture_output=True, text=True).stdout
            assert c_out == py_json(src, fn, diags), \
                f"JSON diverged for {fn}\n C: {c_out!r}\nPY: {py_json(src, fn, diags)!r}"


def test_diagnostic_fixits_dual_rail():
    """Fix-it hints (#diag): the C renderer derives the verb (remove / insert / replace with) from each
    fix-it's span + replacement and prints the replacement with Python `repr()` in text and JSON-escaped
    in the feed, byte-identical to diagnostics.render() / to_json(). The fixits object array sits
    between the location and the notes (the diagnostic_to_dict member order). Covers all three verbs and
    `repr`'s quote selection (a `'` in the replacement switches it to double quotes)."""
    from bcir.frontends.cfront.diagnostics import (  # noqa: PLC0415
        SourceDiagnostic, Span, FixIt, Note, DiagnosticReport, render)
    src = "unsigned f(unsigned x){ return x + ; }\n"
    # (fixit spans+replacements, notes) on a fixed primary error @34:35.
    fixit_sets = [
        [((34, 35), ";")],                                  # replace with ';'
        [((34, 34), ")")],                                  # insert ')'
        [((34, 36), "")],                                   # remove ''
        [((10, 11), "x'y")],                                # repr -> double quotes ("x'y")
        [((34, 35), ";"), ((34, 34), ")"), ((34, 36), "")],  # several fix-its in order
    ]
    note_sets = [[], [((9, 10), "macro here")]]

    def spec(fixits, notes):
        lines = ["error\t34\t35\texpected token"]
        for (a, b), r in fixits:
            lines.append(f"+\t{a}\t{b}\t{r}")
        for (a, b), m in notes:
            lines.append(f"-\t{a}\t{b}\t{m}")
        return "\n".join(lines) + "\n"

    def build(fixits, notes):
        fx = [FixIt(Span(a, b), r) for (a, b), r in fixits]
        nt = [Note(m, Span(a, b)) for (a, b), m in notes]
        return SourceDiagnostic("error", "expected token", span=Span(34, 35),
                                fixits=fx, notes=nt, phase="parse")

    # the oracle side runs in the quick tier too.
    for fixits in fixit_sets:
        assert "fix-it:" in render(build(fixits, []), src, "u.c")
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_diag(d)
        sp = os.path.join(d, "s.c")
        with open(sp, "w") as f:
            f.write(src)
        for fixits in fixit_sets:
            for notes in note_sets:
                diag = build(fixits, notes)
                for flag, want in ((None, render(diag, src, "u.c")),
                                   ("--json", DiagnosticReport([diag], src, "u.c").to_json())):
                    args = [exe] + ([flag] if flag else []) + [sp, "u.c"]
                    out = subprocess.run(args, input=spec(fixits, notes),
                                         capture_output=True, text=True).stdout
                    assert out == want, f"fix-it {flag} diverged for {fixits}\n C: {out!r}\nPY: {want!r}"


def test_diagnostic_include_stack_origin_dual_rail():
    """Include / line-map origin (#diag): a diagnostic relocated to its origin file:line, with the
    #include chain printed as Clang "In file included from <file>:<line>:" frames (text) and an
    "includedFrom" array (JSON), byte-identical to diagnostics.render() / diagnostic_to_dict() given an
    `origin`. The primary banner moves to (origin_file, origin_line) but the column + source snippet
    still come from the (preprocessed) source; notes are NOT relocated. Covers the render-vs-JSON
    asymmetry for a spanless diagnostic (render still shows the frames + origin file; JSON ignores the
    origin and keeps the default file), an empty include stack, and origin alongside a fix-it + note."""
    from bcir.frontends.cfront.diagnostics import (  # noqa: PLC0415
        SourceDiagnostic, Span, Note, FixIt, render, diagnostic_to_dict)
    import json as _json  # noqa: PLC0415
    src = "line0\nline1 has the token X here\nline2\n"
    off = src.index("X")

    def build(span, msg, fixits, notes):
        return SourceDiagnostic("error" if span else "warning", msg, span=span,
                                fixits=[FixIt(Span(a, b), r) for (a, b), r in fixits],
                                notes=[Note(m, Span(a, b)) for (a, b), m in notes], phase="parse")

    # (diag, origin, spec) tuples.
    span = Span(off, off + 1)
    cases = [
        # spanned + origin + 2 include frames
        (build(span, "undeclared 'X'", [], []), ("inc/b.h", 42, [("main.c", 10), ("inc/a.h", 3)]),
         f"error\t{off}\t{off + 1}\tundeclared 'X'\n@\t42\t0\tinc/b.h\n^\t10\t0\tmain.c\n^\t3\t0\tinc/a.h\n"),
        # spanned + origin + empty include stack (no "includedFrom" in JSON)
        (build(span, "undeclared 'X'", [], []), ("inc/b.h", 42, []),
         f"error\t{off}\t{off + 1}\tundeclared 'X'\n@\t42\t0\tinc/b.h\n"),
        # spanless + origin: render shows frames + origin file; JSON ignores origin (keeps default file)
        (build(None, "no span", [], []), ("inc/b.h", 42, [("main.c", 10)]),
         "warning\t-1\t-1\tno span\n@\t42\t0\tinc/b.h\n^\t10\t0\tmain.c\n"),
        # origin alongside a fix-it + a note (only the primary is relocated)
        (build(span, "bad", [((off, off + 1), "Y")], [((0, 3), "see")]), ("hdr.h", 7, [("top.c", 2)]),
         f"error\t{off}\t{off + 1}\tbad\n@\t7\t0\thdr.h\n^\t2\t0\ttop.c\n+\t{off}\t{off + 1}\tY\n-\t0\t3\tsee\n"),
    ]

    def py_text(diag, origin):
        return render(diag, src, "u.c", origin=origin)

    def py_json(diag, origin):
        return _json.dumps([diagnostic_to_dict(diag, src, "u.c", origin=origin)], indent=2)

    for diag, origin, _spec in cases:                                      # oracle side always runs
        assert "In file included from" in py_text(diag, origin) or origin[2] == []
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_diag(d)
        sp = os.path.join(d, "s.c")
        with open(sp, "w") as f:
            f.write(src)
        for diag, origin, spec in cases:
            for flag, want in ((None, py_text(diag, origin)), ("--json", py_json(diag, origin))):
                args = [exe] + ([flag] if flag else []) + [sp, "u.c"]
                out = subprocess.run(args, input=spec, capture_output=True, text=True).stdout
                assert out == want, f"origin {flag} diverged\n C: {out!r}\nPY: {want!r}"


def _report_spec(rep):
    """Build the test_diag spec (one diagnostic per primary, with its fix-its and notes) from a real
    DiagnosticReport -- the multi-diagnostic output of a panic-mode parser-recovery run."""
    lines = []
    for d in rep.diagnostics:
        s, e = (d.span.start, d.span.end) if d.span else (-1, -1)
        lines.append(f"{d.severity}\t{s}\t{e}\t{d.message}")
        for fx in d.fixits:
            lines.append(f"+\t{fx.span.start}\t{fx.span.end}\t{fx.replacement}")
        for nt in d.notes:
            ns, ne = (nt.span.start, nt.span.end) if nt.span else (-1, -1)
            lines.append(f"-\t{ns}\t{ne}\t{nt.message}")
    return ("\n".join(lines) + "\n") if lines else ""


def test_diagnostic_error_recovery_report_dual_rail():
    """Parser error recovery (#diag): a panic-mode run reports EVERY error it resynchronizes past, not
    just the first -- a multi-diagnostic DiagnosticReport. The oracle's `diagnose()` produces that
    report (over the preprocessed source); the C report renderer (`bcir_diag_report_render` and
    `bcir_diag_to_json` over the array) formats the IDENTICAL report, text + JSON, byte-for-byte. This
    drives the C engine with real recovery output (multiple errors, some carrying a fix-it) -- not a
    synthetic battery -- and also covers the clean source (the empty report -> "" / "[]")."""
    from bcir.frontends.cfront.pipeline import diagnose  # noqa: PLC0415
    sources = [
        ("unsigned f(unsigned x) { return x + ; }\n"
         "unsigned g(unsigned y) { return y 7; }\n", "multi.c", 2),     # >=2 errors, one with a fix-it
        ("unsigned ok(unsigned x){ return x*2u + 1u; }\n", "clean.c", 0),  # the empty report
    ]
    reports = [(diagnose(src, filename=fn), fn, lo) for src, fn, lo in sources]
    for rep, _fn, lo in reports:
        assert len(rep.diagnostics) >= lo                              # the recovery actually fired
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_diag(d)
        for rep, fn, _lo in reports:
            sp = os.path.join(d, "pp.c")
            with open(sp, "w") as f:
                f.write(rep.source)                                    # the preprocessed source the spans index
            spec = _report_spec(rep)
            for flag, want in ((None, rep.render()), ("--json", rep.to_json())):
                args = [exe] + ([flag] if flag else []) + [sp, fn]
                out = subprocess.run(args, input=spec, capture_output=True, text=True).stdout
                assert out == want, f"recovery report {flag} diverged for {fn}\n C: {out!r}\nPY: {want!r}"


def _oracle_effects_report(src: str) -> str:
    """The oracle's per-function effect footprints + commute matrix in the bcir-cc --emit-effects
    text format (the C twin of pipeline.effects / commute)."""
    r = compile_unit(src, check_clang=False)
    fns = list(r.lowered.functions)

    def names(rids):
        return sorted(r.lowered.resources[x].name for x in rids if x in r.lowered.resources)

    out = []
    for n in fns:
        e = r.effects[n]
        out.append(f"fn={n} reads={','.join(names(e.reads)) or '-'} writes={','.join(names(e.writes)) or '-'}")
    for i, a in enumerate(fns):
        for b in fns[i + 1:]:
            out.append(f"commute {a} {b} = {1 if r.commute(a, b) else 0}")
    return "\n".join(out) + "\n"


def test_effect_commutation_analysis_dual_rail():
    """Module-scope effect / commutation analysis (#effects): bcir-cc --emit-effects is the C twin of
    pipeline.own_footprint + commute. For each function it reports the file-scope globals it reads and
    writes -- callee effects folded in transitively (the call graph is a DAG under R18) -- then the
    pairwise commute matrix: two functions commute iff their footprints don't conflict (two readers of
    a global commute; a writer conflicts with any reader/writer of it). The whole report is
    byte-identical to the oracle's pipeline.effects / commute, and the gate spans a commuting pair
    (read_a/read_b over disjoint ga/gb) and conflicts (the writer write_a, and the folded via_a)."""
    fixtures = ["cfront_effects.c", "cfront_global_rw.c"]
    reports = {}
    for fx in fixtures:
        src = open(os.path.join(_C, fx), encoding="utf-8").read()
        reports[fx] = _oracle_effects_report(src)
    # the analysis has teeth: cfront_effects spans a commute=1 pair and a conflict=0 pair.
    assert "commute read_a read_b = 1" in reports["cfront_effects.c"]
    assert "commute read_a write_a = 0" in reports["cfront_effects.c"]
    assert "fn=via_a reads=ga writes=ga" in reports["cfront_effects.c"]      # folded callee effects
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        cc = _build_bcir_cc(d)
        for fx in fixtures:
            out = subprocess.run([cc, "--emit-effects", os.path.join(_C, fx)],
                                 capture_output=True, text=True).stdout
            assert out == reports[fx], f"{fx}: effects diverged\n C:\n{out}\nPY:\n{reports[fx]}"


def _scale_unit_src() -> str:
    """A translation unit that busts every *old* fixed IR ceiling: 40 leaf functions + a
    12-parameter function + a 40-call aggregator + a 7500-claim function."""
    fns = [f"unsigned g{k}(void){{ return {k}u; }}" for k in range(40)]          # > old BCIR_MAX_FUNCS 16
    ps = [chr(ord("a") + i) for i in range(12)]
    fns.append("unsigned many12(" + ",".join(f"unsigned {p}" for p in ps) +
               "){ return " + "+".join(ps) + "; }")                              # > old BCIR_MAX_PARAMS 8
    fns.append("unsigned agg(void){ return " + "+".join(f"g{k}()" for k in range(40)) + "; }")  # > old BCIR_MAX_CALLS 32
    body = "\n".join("  acc = acc + 1u;" for _ in range(2500))
    fns.append("unsigned big(unsigned acc){\n" + body + "\n  return acc;\n}")    # > old 4096-claim per-fn cap
    return "\n".join(fns) + "\n"


def test_scalable_ir_no_fixed_ceilings():
    """Scalable IR (no fixed `BCIR_MAX_*`): a unit that busts every old ceiling -- 43 functions (>
    the old `BCIR_MAX_FUNCS` 16), a 12-parameter function (> 8), a 40-call aggregator (> 32), and a
    7500-claim function (> the old 4096 per-function cap) -- compiles clean on the C twin (the IR
    grows geometrically) and matches the oracle's structural counts, which are uncapped by design."""
    src = _scale_unit_src()
    r = compile_unit(src, check_clang=False)                 # oracle: Python lists, no caps
    assert len(r.lowered.functions) == 43
    assert len(r.lowered.functions["big"].claims) == 7500
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        cc = _build_bcir_cc(d)
        p = os.path.join(d, "scale.c")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(src)
        out = subprocess.run([cc, "--emit-claimgraph", p], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        m = re.search(r"funcs=(\d+) claims=(\d+).*ok=(\d)", out.stdout)
        assert m and (m.group(1), m.group(2), m.group(3)) == ("43", "7500", "1"), out.stdout[:200]


def _pstress_unit_src() -> str:
    """A unit busting every old *parser-state* cap: 20 struct defs (> s[16]), 25 file-scope globals
    (> gv[16]), 20 typedefs, and a function with 300 locals (> env[256])."""
    L = [f"struct S{k} {{ unsigned m0; unsigned m1; }};" for k in range(20)]
    L += [f"typedef unsigned U{k};" for k in range(20)]
    L += [f"static const unsigned G{k}[2] = {{ {k}u, {k + 1}u }};" for k in range(25)]
    L.append("unsigned big(void){\n" + "\n".join(f"  unsigned v{i} = {i}u;" for i in range(300)) +
             "\n  return " + "+".join(f"v{i}" for i in range(300)) + "; }")
    L.append("unsigned useg(unsigned i){ return G0[i%2u] + G19[i%2u] + G24[i%2u]; }")
    return "\n".join(L) + "\n"


def test_scalable_parser_state_no_fixed_caps():
    """Scalable parser state: the twin's parser-state record arrays (struct defs / globals / typedefs /
    enum constants / locals) grow geometrically -- the old fixed `s[16]` / `gv[16]` / `env[256]` caps
    are gone, so a real header (20 structs, 25 globals, 300 locals) lowers and matches the oracle, which
    is uncapped by design."""
    src = _pstress_unit_src()
    r = compile_unit(src, check_clang=False)                 # oracle: Python dicts/lists, no caps
    assert len(r.lowered.functions) == 2 and r.is_clean
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        cc = _build_bcir_cc(d)
        p = os.path.join(d, "pstress.c")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(src)
        out = subprocess.run([cc, "--emit-claimgraph", p], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert "ok=1" in out.stdout, out.stdout[:200]          # all structs/globals/locals resolved


_INTPROMOTE_SRC = (
    "int sdiv(int a, int b){ return b ? a / b : 0; }\n"          # signed division (truncates toward 0)
    "int smod(int a, int b){ return b ? a % b : 0; }\n"          # signed remainder
    "int sshr(int a){ return a >> 3; }\n"                        # arithmetic right shift (sign-extends)
    "int scmp(int a, int b){ return (a < b) + 2*(a <= b) + 4*(a > b) + 8*(a >= b); }\n"  # signed compares
    "unsigned udiv(unsigned a, unsigned b){ return b ? a / b : 0u; }\n"   # unsigned division (control)
    "long wide(int a, long b){ return a + b; }\n"                # int + long -> 64-bit UAC
    "long umix(unsigned a, long b){ return a * b; }\n"          # unsigned*long -> long (mixed width/sign)
)


def test_integer_promotions_and_uac_oracle():
    """Integer promotions + usual arithmetic conversions (§6.3.1.1 / §6.3.1.8), oracle prototype: the
    lowering now types every temp by its true (width, signedness), so a signed `int` divide / remainder
    / right-shift / comparison emits signed C (not the old flat `uint32_t`), and `int + long` widens to
    64-bit. The emitted C is therefore behaviour-equivalent to the source over the FULL signed range
    (negatives included) -- exactly the case the old unsigned-32 value model got wrong. (The C twin
    port is the next segment; here the oracle prototype is validated against real Clang.)"""
    r = compile_unit(_INTPROMOTE_SRC, check_clang=False)
    emit = "\n".join(r.emitted.values())
    # teeth: result temps carry their real integer types -- the old model rendered them all uint32_t.
    assert "int32_t" in emit, emit
    assert "int64_t" in emit, emit          # int + long widened to 64-bit
    assert "uint32_t" in emit               # unsigned stays unsigned
    if not _CC:
        return
    harness = f"""#include <stdint.h>
#include <stdio.h>
{_BOUNDS_GUARD}
{_INTPROMOTE_SRC}
{emit}
static uint64_t S=0x9E3779B97F4A7C15u;
static uint64_t nx(void){{S=S*6364136223846793005u+1442695040888963407u;return S>>32;}}
int main(void){{
  for(int i=0;i<300000;i++){{
    int a=(int)nx(), b=(int)nx(); long lb=(long)nx()<<3 | (long)nx();
    if(sdiv(a,b)!=bcir_sdiv(a,b)){{printf("sdiv@%d a=%d b=%d\\n",i,a,b);return 1;}}
    if(smod(a,b)!=bcir_smod(a,b)){{printf("smod@%d\\n",i);return 1;}}
    if(sshr(a)!=bcir_sshr(a)){{printf("sshr@%d a=%d\\n",i,a);return 1;}}
    if(scmp(a,b)!=bcir_scmp(a,b)){{printf("scmp@%d a=%d b=%d\\n",i,a,b);return 1;}}
    if(udiv((unsigned)a,(unsigned)b)!=bcir_udiv((unsigned)a,(unsigned)b)){{printf("udiv@%d\\n",i);return 1;}}
    if(wide(a,lb)!=bcir_wide(a,lb)){{printf("wide@%d\\n",i);return 1;}}
    if(umix((unsigned)a,lb)!=bcir_umix((unsigned)a,lb)){{printf("umix@%d\\n",i);return 1;}}
  }}
  printf("MATCH\\n");return 0;}}"""
    with tempfile.TemporaryDirectory() as d:
        c, e = os.path.join(d, "e.c"), os.path.join(d, "e")
        with open(c, "w", encoding="utf-8") as fh:
            fh.write(harness)
        for std in ("c23", "c2x", "c17"):
            b = subprocess.run([_CC, f"-std={std}", "-O2", c, "-o", e], capture_output=True, text=True)
            if b.returncode == 0:
                break
        else:
            raise AssertionError(f"harness build failed: {b.stderr[-400:]}")
        assert subprocess.run([e], capture_output=True, text=True).stdout.strip() == "MATCH"


_AGGINIT_SRC = (
    "struct P { unsigned a; unsigned b; unsigned c; };\n"
    "union U { unsigned w; unsigned h; };\n"
    "unsigned spos(unsigned x){ struct P p = {x, x+1u, x+2u}; return p.a*100u+p.b*10u+p.c; }\n"
    "unsigned sdes(unsigned x){ struct P p = {.c=x, .a=x+5u}; return p.a*100u+p.b*10u+p.c; }\n"   # .b gap -> 0
    "unsigned apos(unsigned x){ unsigned a[4] = {x, x+1u, x+2u}; return a[0]+a[1]+a[2]+a[3]; }\n"  # a[3] gap -> 0
    "unsigned ades(unsigned x){ unsigned a[4] = {[3]=x, [0]=x+9u}; return a[0]*10u+a[3]; }\n"      # gaps -> 0
    "unsigned udes(unsigned x){ union U u = {.h=x}; return u.w; }\n"                               # overlap @0
)


def test_local_aggregate_initializers_oracle():
    """Local aggregate initializers (§6.7.10), oracle prototype: a braced `struct P p = {…}` /
    `union U u = {…}` / `T a[N] = {…}` (positional + `.field=` / `[i]=` designators) lowers to a
    `= {0}` zero baseline plus a store per initialized member/element (reusing the member/array store
    path), so uninitialized members zero-fill. Behaviour-equivalent to Clang across struct/union/array
    and positional/designated. (The C-twin port is the next segment.)"""
    r = compile_unit(_AGGINIT_SRC, check_clang=False)
    emit = "\n".join(r.emitted.values())
    assert "= {0}" in emit                  # the zero baseline (uninitialized members zero-fill)
    assert "[4]" in emit                     # a local array is declared with its dimension
    if not _CC:
        return
    harness = f"""#include <stdint.h>
#include <stdio.h>
#include <string.h>
{_BOUNDS_GUARD}
{_AGGINIT_SRC}
{emit}
int main(void){{
  for(unsigned x=0; x<5000u; x++){{
    if(spos(x)!=bcir_spos(x)||sdes(x)!=bcir_sdes(x)||apos(x)!=bcir_apos(x)
     ||ades(x)!=bcir_ades(x)||udes(x)!=bcir_udes(x)){{printf("MISMATCH x=%u\\n",x);return 1;}}
  }}
  printf("MATCH\\n");return 0;}}"""
    with tempfile.TemporaryDirectory() as d:
        c, e = os.path.join(d, "e.c"), os.path.join(d, "e")
        with open(c, "w", encoding="utf-8") as fh:
            fh.write(harness)
        for std in ("c23", "c2x", "c17"):
            b = subprocess.run([_CC, f"-std={std}", "-O2", c, "-o", e], capture_output=True, text=True)
            if b.returncode == 0:
                break
        else:
            raise AssertionError(f"harness build failed: {b.stderr[-400:]}")
        assert subprocess.run([e], capture_output=True, text=True).stdout.strip() == "MATCH"


def test_scalar_globals_read_write_dual_rail():
    """Scalar file-scope globals (#globals): the oracle models a scalar global as a plain resource --
    a read references it directly (no c.load), a write is a c.copy to the global rid. The C twin now
    lowers global writes (`acc = v`, `acc += b`) identically, and emits the global by NAME (a bare
    `acc = t;` assignment, not a `uint32_t acc = t;` declaration -- the storage is external). The two
    rails agree on the structural summary; behaviour-equivalence is checked with a side-effect-aware
    harness (the functions mutate module-scope state, so the global is reset between the original and
    the emitted twin, and both the return value AND the final global are compared)."""
    fx = "cfront_global_rw.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, _entry = _oracle(src)
    assert "ok=1" in oracle_summary and "binop=2" in oracle_summary, oracle_summary
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        # the emit references the global by name: a bare `acc = ...;`, never `uint32_t acc = ...;`.
        assert "acc = " in c_emit and "uint32_t acc" not in c_emit and "return acc;" in c_emit, c_emit
        # side-effect-aware behaviour: reset `acc` between the original and the emitted twin per call.
        harness = f"""#include <stdint.h>
#include <stdio.h>
{_BOUNDS_GUARD}
{r.source}

{c_emit}
static uint64_t S=0x9E3779B97F4A7C15u;
static uint32_t rng(void){{S=S*6364136223846793005u+1442695040888963407u;return (uint32_t)(S>>32);}}
int main(void){{
  for(int i=0;i<256;i++){{
    unsigned a=rng(), b=rng(), sd=rng();
    acc=sd; unsigned r1=accumulate(a,b); unsigned acc1=acc;
    acc=sd; unsigned r2=bcir_accumulate(a,b); unsigned acc2=acc;
    if(r1!=r2||acc1!=acc2){{printf("MISMATCH accumulate@%d\\n",i);return 1;}}
    acc=sd; seed(a); unsigned s1=acc;
    acc=sd; bcir_seed(a); unsigned s2=acc;
    if(s1!=s2){{printf("MISMATCH seed@%d\\n",i);return 1;}}
    acc=sd; unsigned p1=peek(); acc=sd; unsigned p2=bcir_peek();
    if(p1!=p2){{printf("MISMATCH peek@%d\\n",i);return 1;}}
  }}
  printf("MATCH\\n");return 0;}}"""
        cf, ef = os.path.join(d, "g.c"), os.path.join(d, "g")
        open(cf, "w").write(harness)
        for std in ("c23", "c2x", "c17"):
            b = subprocess.run([_CC, f"-std={std}", "-O2", cf, "-o", ef], capture_output=True, text=True)
            if b.returncode == 0:
                break
        else:
            raise AssertionError(f"global r/w harness build failed:\n{b.stderr}")
        out = subprocess.run([ef], capture_output=True, text=True).stdout.strip()
        assert out == "MATCH", f"{fx}: scalar-global r/w not behaviour-equivalent ({out})"


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


def test_cfront_differential_fuzz():
    """A seeded differential fuzzer over the shared cfront subset (`tools/c/fuzz_cfront.py`): random but
    well-defined programs -- struct/union type definitions, an optional helper prelude, then an entry `f`,
    with `char`/`short`/`int`/`long`/`unsigned`/`unsigned long`/`float`/`double` and mixed
    scalar + `_Bool` + bitfield struct / union-by-value parameters/locals (a struct may carry a NESTED struct
    member `struct S0 in;` read/written via `s.in.x` / `s->in.x` -- as a by-value param, a local, a return,
    OR a pointer param), a struct-BY-VALUE return, AND `struct T *`
    parameters read+written through the pointer (members `s.m` / `s->m`, a union's single active member --
    which may itself be a bitfield (union-of-bitfields) -- a
    bitfield `m:W` (incl. in an `__attribute__((packed))` struct, where bitfields pack bit-by-bit and may
    straddle byte/word boundaries -- the `sizeof`/`offsetof` LAYOUT differential validates it), a
    dynamic-indexed array member `s.arr[e & 3u]` -- an array-bearing struct now also as a
    LOCAL and a RETURN via a NESTED-brace init `{ m, {e0,e1,..}, n }`; a struct return / a struct-pointer's
    backing struct is compared member-and-element-by-value after the call), plus up to two possibly-aliasing
    writable `unsigned *`, drawing from the mixed-width usual arithmetic conversions / floating-point
    arithmetic / bitwise / bounded shifts / comparisons / ternary / if / bounded for / statement expressions /
    inc-dec / mutable-local-and-member assignment / same-unit calls / pointer reads AND writes -- are run
    through BOTH rails and Clang. The two rails must agree on the total-compile OUTCOME (clean/dirty/fallback);
    a mutually-clean unit must additionally have an identical structural claim SUMMARY (parity), a struct/union
    LAYOUT (`sizeof` + each member's `offsetof`) equal to Clang's (a `_Static_assert` differential -- the
    behaviour check is size-blind), and emitted C that is behaviour-equivalent to Clang on both rails --
    integer results compared exactly, float results ULP-tolerantly (nan/inf-aware), and every pointer's
    backing array compared after the call (same alias pattern). This is the regression guard for the dual-rail bugs this fuzzer flushed -- the twin's
    parameter-write redeclaration, the oracle's assignment/`i++`-as-stmt-expr-value, the ternary / call
    result types losing their sign OR float type (a logical shift on a signed select / a signed
    char/short/int/long call result / a `double` select truncated to int), the twin rejecting a pointer
    subscript OR a struct member as a statement-expression value, the oracle re-evaluating a compound store's
    index, the twin loading a `float`/`double` struct member as integer bits, the oracle memcpy'ing a
    mismatched-width / narrower-integer / float store source into a slot, BOTH rails reading an unsigned
    sub-int bitfield as `unsigned` instead of promoting it to `int` (a wrongly-unsigned compare), the
    twin storing a `float` member-array element as a `uint32_t` reinterpret instead of converting, and BOTH
    rails laying out a bitfield that FOLLOWS a sub-word member (`short m0; unsigned m1:1;`) in a fresh
    type-aligned storage unit instead of packing it into the current bit cursor (the Itanium/Clang rule),
    giving a wrong struct size + member offsets vs Clang, and BOTH rails storing into a `_Bool` MEMBER /
    `_Bool[]` element as a raw byte copy instead of NORMALIZING any nonzero to 1 (§6.3.1.2) -- `s.flag = 2`
    read back as 2. The seeds are fixed (deterministic)."""
    import random as _random
    import sys as _sys
    tools_c = os.path.join(_ROOT, "tools", "c")
    if tools_c not in _sys.path:
        _sys.path.insert(0, tools_c)
    import fuzz_cfront

    if not _CC:                                         # no compiler -> can't build the twin; at least pin
        rng = _random.Random(1234)                     # that generation terminates and the oracle never
        for _ in range(60):                            # crashes on an in-subset program.
            fuzz_cfront._oracle_outcome(fuzz_cfront.Gen(rng).program().source)
        return
    with tempfile.TemporaryDirectory() as d:
        twin = _build_frontend(d)
        for seed in (1234, 5678, 4242):
            divergence, stats = fuzz_cfront.run_seed(twin, _CC, count=40, seed=seed, d=d)
            assert divergence is None, divergence
            assert stats["clean"] >= 1 and stats["checked"] == stats["clean"], stats


def test_bounds_promotion_local_static_arrays_to_masked():
    """§5.12 bounds-promotion: an indexed access into a known-extent LOCAL/STATIC array OBJECT promotes
    from `assumed_safe` (trusted) to `masked` (runtime-bounds-checked -- the extent is recoverable from
    the resource shape, the contract the quarantine handler discharges). A POINTER base (extent unknown),
    a struct MEMBER array (a follow-on), and an MMIO register stay `assumed_safe`. Metadata only -- no
    emit/behaviour change (every cfront fixture still passes + is Clang-equivalent), and the twin promotes
    identically (the differential fuzzer is clean), so parity holds."""
    from bcir.frontends.cfront import compile_unit

    def bounds_of(src):
        r = compile_unit(src, check_clang=False)
        b = set()
        for n in r.lowered.functions:
            for ph in r.lowered.functions[n].module.phases:
                for c in ph.claims:
                    if c.op in ("c.load", "c.store"):
                        b.add(c.bounds)
        return r.is_clean, b

    clean, b = bounds_of("unsigned f(unsigned i){ unsigned a[8]; a[i&7u]=3u; return a[i&7u]; }")
    assert clean and b == {"masked"}                                  # a local array -> runtime-checked
    clean, b = bounds_of("unsigned f(unsigned i){ static unsigned a[4]; a[i&3u]=2u; return a[i&3u]; }")
    assert clean and b == {"masked"}                                  # a static array -> runtime-checked
    clean, b = bounds_of("unsigned f(unsigned *p,unsigned i){ p[i&7u]=3u; return p[i&7u]; }")
    assert clean and b == {"assumed_safe"}                            # a pointer (extent unknown) stays trusted


def test_bounds_quarantine_traps_out_of_bounds():
    """§5.12 quarantine handler: the emitted guard on a `masked` local-array access is transparent for an
    in-bounds index (behaviour-identical to the raw `a[i]`) and, on an out-of-bounds index, calls the WEAK
    `bcir_bounds_quarantine` runtime handler -- which records the provenance (including the `<func>:<array>`
    source site) and aborts (fail-fast). Linked against the real runtime/c/bcir_quarantine.c."""
    if not _CC:
        return
    src = "unsigned g(unsigned i){ unsigned a[8]; for(unsigned k=0u;k<8u;k++) a[k]=k*2u; return a[i]; }"
    from bcir.frontends.cfront import compile_unit
    r = compile_unit(src, check_clang=False)
    name = next(reversed(r.lowered.functions))
    body = r.emitted[name].split("*/\n", 1)[-1]
    assert 'BCIR_CHK(' in body and '"g:a"' in body, body          # the guard threads the <func>:<array> site
    with tempfile.TemporaryDirectory() as d:
        prog = (f'#include <stdint.h>\n#include <stdlib.h>\n#include <stdio.h>\n#include "bcir_quarantine.h"\n'
                f'{r.source}\n\n{body}\n'
                f'int main(int c, char **v){{ (void)c; printf("%u\\n", bcir_g((unsigned)atoi(v[1]))); return 0; }}\n')
        cpath, epath = os.path.join(d, "e.c"), os.path.join(d, "e")
        open(cpath, "w").write(prog)
        b = subprocess.run([_CC, "-std=c23", "-O2", "-I", _C, cpath,
                            os.path.join(_C, "bcir_quarantine.c"), "-o", epath], capture_output=True, text=True)
        assert b.returncode == 0, b.stderr
        inb = subprocess.run([epath, "3"], capture_output=True, text=True)   # in-bounds: a[3] = 6, exits 0
        assert inb.returncode == 0 and inb.stdout.strip() == "6", (inb.returncode, inb.stdout)
        oob = subprocess.run([epath, "99"], capture_output=True, text=True)  # OOB: the handler aborts
        assert oob.returncode != 0 and "bounds-quarantine" in oob.stderr, (oob.returncode, oob.stderr)
        assert "g:a" in oob.stderr, oob.stderr                    # the source site is in the fail-fast message


def test_recovered_extent_quarantines_out_of_bounds():
    """§5.12 recoverable extents end-to-end: a NAKED pointer from `malloc(n*sizeof(T))` recovers its element
    count `n`, so `p[i]` is guarded against the RUNTIME extent `BCIR_CHK(rid, i, n, "mpick:p")`. In-bounds
    (`i < n`) is transparent (the raw value); out-of-bounds calls the weak handler, which records the
    provenance naming the `<func>:<pointer>` site and aborts. Linked against the real runtime -- this proves
    the recovered runtime extent (not a constant) actually bounds-checks the heap buffer."""
    if not _CC:
        return
    src = ("unsigned mpick(unsigned n, unsigned i){ unsigned *p = malloc(n*sizeof(unsigned)); "
           "for(unsigned k=0u;k<n;k++) p[k]=k*2u; return p[i]; }")
    from bcir.frontends.cfront import compile_unit
    r = compile_unit("#include <stdlib.h>\n" + src, check_clang=False)
    body = r.emitted["mpick"].split("*/\n", 1)[-1]
    assert 'BCIR_CHK(' in body and ', n, "mpick:p")' in body, body   # the extent is the runtime count `n`
    with tempfile.TemporaryDirectory() as d:
        prog = (f'#include <stdint.h>\n#include <stdlib.h>\n#include <stdio.h>\n#include "bcir_quarantine.h"\n'
                f'{body}\n'
                f'int main(int c, char **v){{ (void)c; printf("%u\\n", bcir_mpick(8u, (unsigned)atoi(v[1]))); '
                f'return 0; }}\n')
        cpath, epath = os.path.join(d, "e.c"), os.path.join(d, "e")
        open(cpath, "w").write(prog)
        b = subprocess.run([_CC, "-std=c23", "-O2", "-I", _C, cpath,
                            os.path.join(_C, "bcir_quarantine.c"), "-o", epath], capture_output=True, text=True)
        assert b.returncode == 0, b.stderr
        inb = subprocess.run([epath, "3"], capture_output=True, text=True)   # in-bounds: p[3] = 6
        assert inb.returncode == 0 and inb.stdout.strip() == "6", (inb.returncode, inb.stdout)
        oob = subprocess.run([epath, "99"], capture_output=True, text=True)  # OOB of the 8-element buffer
        assert oob.returncode != 0 and "bounds-quarantine" in oob.stderr and "mpick:p" in oob.stderr, \
            (oob.returncode, oob.stderr)


def _r21(src):
    """Compile a heap snippet and return (is_clean, [R21 lifetime messages])."""
    from bcir.frontends.cfront import compile_unit
    r = compile_unit("#include <stdlib.h>\n" + src, check_clang=False)
    return r.is_clean, [d.message for d in r.lifetime_diagnostics]


def test_r21_lifetime_is_load_bearing_for_c_heap():
    """§5.12 R21 made load-bearing for the C frontend: the malloc/free `claim.lifetime` annotations
    (ALLOC on the allocator result, FREE on `free(p)`) feed the pointer-lifetime law, so a use-after-free
    or double-free a C program would have left UB is now CAUGHT -- as an ADVISORY diagnostic, never folded
    into the frontend pass/fail (`is_clean` stays True), exactly like the R19/R20 timing laws."""
    # a dangling READ (load), a dangling WRITE (store), and a dangling deref `*p` are all use-after-free
    # (each reads the freed pointer to form the address); free-of-freed is a double-free.
    for src in ("unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); return p[0]; }",
                "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); p[0]=1u; return n; }",
                "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); return *p; }"):
        clean, diags = _r21(src)
        assert clean and any("use-after-free" in d for d in diags), (src, diags)
    clean, diags = _r21("unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); free(p); return n; }")
    assert clean and any("double-free" in d for d in diags), diags
    # well-formed heap use is silent: access BEFORE free, and free-then-reallocate-then-use (the write
    # re-validates the pointer), both produce no lifetime diagnostic.
    for src in ("unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); unsigned r=p[0]; free(p); return r; }",
                "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); "
                "p=malloc(n*sizeof(unsigned)); unsigned r=p[0]; free(p); return r; }"):
        clean, diags = _r21(src)
        assert clean and diags == [], (src, diags)


def test_r21_does_not_disturb_the_corpus():
    """Non-disturbance: R21 is advisory, so it never flips a fixture's clean verdict, and no well-formed
    fixture (the whole corpus -- only `cfront_stdlibmem.c` even allocates, and it frees correctly) emits a
    spurious lifetime diagnostic."""
    import glob
    from bcir.frontends.cfront import compile_unit
    for path in sorted(glob.glob(os.path.join(_C, "cfront_*.c"))):
        fx = os.path.basename(path)
        r = compile_unit(open(path, encoding="utf-8").read(), check_clang=False, includes=_includes_for(fx))
        if r.fallback:
            continue
        assert r.lifetime_diagnostics == [], (fx, [d.message for d in r.lifetime_diagnostics])


def _r21_kinds(messages):
    """(func, kind) pairs from R21 diagnostic strings -- `f: claim N: use-after-free of RID M ...` (oracle)
    or `R21 f: use-after-free` (twin). RID/claim numbers differ across rails; the FUNC + KIND must agree."""
    out = []
    for m in messages:
        kind = "double-free" if "double-free" in m else ("use-after-free" if "use-after-free" in m else None)
        if kind is None:
            continue
        m = m[len("R21 "):] if m.startswith("R21 ") else m
        out.append((m.split(":", 1)[0].strip(), kind))
    return sorted(out)


def test_r21_dual_rail_parity():
    """§5.12 R21 dual-rail: the C twin verifier reports the SAME use-after-free / double-free events as the
    Python oracle for heap C. The twin prints `R21 <func>: <kind>` lines (kind ∈ {use-after-free,
    double-free}) ahead of its `----EMIT----` marker; the oracle's `lifetime_diagnostics` carry the same.
    RID/claim numbering differs across rails, so parity is on the (function, kind) multiset."""
    if not _CC:
        return
    from bcir.frontends.cfront import compile_unit
    cases = [
        "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); return p[0]; }",      # UAF
        "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); p[0]=1u; return n; }", # UAF store
        "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); free(p); return n; }", # double-free
        "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); unsigned r=p[0]; free(p); return r; }",  # clean
        "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); "
        "p=malloc(n*sizeof(unsigned)); unsigned r=p[0]; free(p); return r; }",                           # reuse
    ]
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        for i, src in enumerate(cases):
            full = "#include <stdlib.h>\n" + src
            oracle = _r21_kinds(d.message for d in compile_unit(full, check_clang=False).lifetime_diagnostics)
            cpath = os.path.join(d, f"u{i}.c")
            open(cpath, "w").write(full)
            out = subprocess.run([exe, cpath], capture_output=True, text=True).stdout
            summary = out.partition("----EMIT----")[0]
            twin = _r21_kinds(ln.strip() for ln in summary.splitlines() if ln.startswith("R21 "))
            assert twin == oracle, f"case {i}: twin={twin} oracle={oracle}\n{src}"


def test_extent_count_mutation_is_not_promoted():
    """§5.12 soundness: a recovered count must be STABLE from the allocation onward. A count that is
    re-assigned AFTER the alloc (`n = n - 1`, `n--`) -- so its single assignment is an ordinary BODY write,
    not a decl-init -- must NOT bind, or the re-emitted runtime extent would disagree with the allocation
    and FALSE-TRAP a valid access. Both rails leave it unmanaged (no `BCIR_CHK`); a decl-init count
    (`unsigned m = ...`, before the alloc) still promotes. Guards the gate that distinguishes the two."""
    from bcir.frontends.cfront import compile_unit
    # the access p[n] (after n=n-1) reads p[original-1], the LAST valid element -- it must not be guarded
    # against the mutated extent (which would reject original-1 < original-1).
    src = ("unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); "
           "for(unsigned k=0u;k<n;k++) p[k]=k; n=n-1u; return p[n]; }")
    r = compile_unit("#include <stdlib.h>\n" + src, check_clang=False)
    assert "BCIR_CHK" not in r.emitted["f"], r.emitted["f"]              # oracle: unmanaged (sound)
    if _CC:
        with tempfile.TemporaryDirectory() as d:
            # twin agrees (no BCIR_CHK), and the emit RUNS without a false trap on the valid p[original-1].
            exe = _build_frontend(d)
            cpath = os.path.join(d, "f.c")
            open(cpath, "w").write("#include <stdlib.h>\n" + src)
            out = subprocess.run([exe, cpath], capture_output=True, text=True).stdout
            assert "BCIR_CHK" not in out.partition("----EMIT----")[2], out   # twin: unmanaged too (parity)
            body = r.emitted["f"].split("*/\n", 1)[-1]
            prog = (f'#include <stdint.h>\n#include <stdlib.h>\n#include <stdio.h>\n#include "bcir_quarantine.h"\n'
                    f'{body}\nint main(void){{ printf("%u\\n", bcir_f(5u)); return 0; }}\n')  # f(5)=p[4]=4
            ep = os.path.join(d, "e")
            open(os.path.join(d, "e.c"), "w").write(prog)
            b = subprocess.run([_CC, "-std=c23", "-O2", "-I", _C, os.path.join(d, "e.c"),
                                os.path.join(_C, "bcir_quarantine.c"), "-o", ep], capture_output=True, text=True)
            assert b.returncode == 0, b.stderr
            run = subprocess.run([ep], capture_output=True, text=True)     # must NOT abort (no false trap)
            assert run.returncode == 0 and run.stdout.strip() == "4", (run.returncode, run.stdout, run.stderr)


def test_masked_claims_are_discharged_by_a_runtime_guard():
    """§5.12 lowering faithfulness (item 4): the emit must HONOR the `masked` bounds metadata -- every
    masked load/store claim is discharged by exactly one `BCIR_CHK` runtime guard in the emitted C, and
    every masked claim carries the `bounds` verify contract (so R7 validates it). Across the whole corpus,
    a masked claim never silently loses its guard, and a guard is never emitted without a masked claim."""
    import glob
    from bcir.frontends.cfront import compile_unit
    seen_masked = 0
    for path in sorted(glob.glob(os.path.join(_C, "cfront_*.c"))):
        fx = os.path.basename(path)
        r = compile_unit(open(path, encoding="utf-8").read(), check_clang=False, includes=_includes_for(fx))
        if r.fallback:
            continue
        for name, lf in r.lowered.functions.items():
            masked = [c for c in lf.claims if c.op in ("c.load", "c.store") and c.bounds == "masked"]
            seen_masked += len(masked)
            assert all(c.verify == "bounds" for c in masked), (fx, name)        # R7 contract
            assert len(masked) == r.emitted[name].count("BCIR_CHK"), \
                (fx, name, "masked claims", len(masked), "BCIR_CHK guards", r.emitted[name].count("BCIR_CHK"))
    assert seen_masked > 0                                                       # the corpus exercises the path


def test_cfront_lowering_faithfulness_is_a_self_check():
    """§5.12 item 4: the cfront pipeline SELF-VERIFIES that its emit honors the masked bounds metadata
    (`verify_cfront_lowering`, surfaced in `CompileResult.lowering_diagnostics`). A real compile is faithful
    (no diagnostic); a doctored emit that DROPS a masked claim's guard is flagged R12 -- the law catches a
    backend that would silently lose a bounds check, on any compile (not just the corpus)."""
    from bcir.frontends.cfront import compile_unit
    from bcir.frontends.cfront.pipeline import verify_cfront_lowering
    r = compile_unit(open(os.path.join(_C, "cfront_stdlibmem.c"), encoding="utf-8").read(), check_clang=False)
    assert r.lowering_diagnostics == [], [d.message for d in r.lowering_diagnostics]   # the real emit is faithful
    lf = r.lowered.functions["msum"]                                                   # 2 masked claims
    assert verify_cfront_lowering(lf, r.emitted["msum"]) == []                         # faithful
    dropped = verify_cfront_lowering(lf, r.emitted["msum"].replace("BCIR_CHK", "no_guard", 1))
    assert dropped and dropped[0].law == "R12", dropped                                # one guard dropped -> flagged


def test_native_vla_lowering_and_unsupported_forms():
    """§5.9 native VLAs: a 1-D stack VLA `T a[n]` (runtime size) is lowered FAITHFULLY -- the size is evaluated
    once and snapshotted, the array is declared IN-BODY (`T a[__ext];`, a real stack array -- no heap, no leak)
    and `a[i]` is bounds-masked against the snapshot (§5.12). Behaviour-equivalent to Clang on both rails. The
    genuinely unsupported forms (a VLA with an initializer, or multi-dimensional) route cleanly to `--fallback`,
    and a plain integer-literal dim still compiles a static array exactly as before."""
    from bcir.frontends.cfront import compile_unit
    from bcir.frontends.cfront.lower import CLowerError
    from bcir.frontends.cfront.cparse import CParseError
    # a native VLA compiles and is Clang-equivalent (in-bounds for every n -> the rails + Clang agree)
    for src in ("unsigned f(unsigned n){ unsigned m=(n&7u)+1u; unsigned a[m]; unsigned s=0u;"
                "  for(unsigned i=0u;i<m;i++){a[i]=i+n;s+=a[i];} return s; }",
                "int g(int n){ int m=(n&7)+1; int a[m]; int s=0;"
                "  for(int i=0;i<m;i++){a[i]=i*2-n;s+=a[i];} return s; }",
                "unsigned h(unsigned n){ unsigned a[(n&3u)+2u]; unsigned k=(n&3u)+2u; unsigned s=0u;"
                "  for(unsigned i=0u;i<k;i++){a[i]=i^n;s+=a[i];} return s; }"):
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)
    # a single-integer-literal dim still compiles a static array (unchanged)
    r = compile_unit("unsigned f(unsigned i){ unsigned a[8]={0}; a[i & 7u]=i; return a[i & 7u]; }", check_clang=True)
    assert r.equivalence == "match" and r.is_clean, r.equivalence
    # a VLA with an initializer (illegal C) routes to fallback
    try:
        compile_unit("unsigned f(unsigned n){ unsigned a[n]={0}; return a[0]; }", check_clang=False)
        assert False, "an initialized VLA should route to fallback"
    except CLowerError as e:
        assert "VLA" in str(e) or "variable-length" in str(e), e
    # a 2-D / 3-D VLA is now natively lowered (see test_multidim_vla_lowering); only a >3-D VLA falls back
    try:
        compile_unit("unsigned f(unsigned n){ unsigned a[n][n][n][n]; return a[0][0][0][0]; }", check_clang=False)
        assert False, "a >3-D VLA should route to fallback"
    except (CParseError, CLowerError):
        pass


def test_vla_sizeof_is_runtime():
    """§5.9 (#vlasizeof): `sizeof a` of a 1-D stack VLA is a RUNTIME value -- the snapshot extent times the
    element size, emitted as `(size_t)((size_t)__bcir_extK * sizeof(elem))`. `sizeof a[0]` (an element) stays
    the STATIC element size, and `sizeof` of a non-VLA (a static array, a pointer) is unchanged -- so no
    cross-rail divergence and Clang-equivalent on both rails."""
    from bcir.frontends.cfront import compile_unit, cparse, lower, emit
    # the runtime sizeof compiles + is Clang-equivalent, and emits the runtime form (NOT a stale `= 0u`)
    src = ("unsigned f(unsigned n){ unsigned m=(n&7u)+1u; unsigned a[m]; unsigned b=(unsigned)sizeof a;"
           " unsigned e=(unsigned)sizeof a[0]; unsigned c=b/e; unsigned s=0u;"
           " for(unsigned i=0u;i<c;i++){a[i]=i+n;s+=a[i];} return s+b+e; }")
    r = compile_unit(src, check_clang=True)
    assert r.equivalence == "match" and r.is_clean, r.equivalence
    body = emit.emit_function(lower.lower_unit(cparse.parse_unit(src), None).functions["f"])
    assert "(size_t)((size_t)__bcir_ext0 * 4)" in body, body         # the RUNTIME extent*size form
    assert "size_t" in body                                          # the temp is size_t (matches the twin)
    # sizeof of a NON-VLA stays a static fold (no extent read) -- no regression
    for src2, want in [("unsigned f(unsigned i){ unsigned a[8]={0}; a[i&7u]=i; return (unsigned)sizeof a + a[0]; }", "match"),
                       ("unsigned g(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); unsigned z=(unsigned)sizeof p; free(p); return z; }", "match")]:
        r2 = compile_unit(src2, check_clang=True)
        assert r2.equivalence == want and r2.is_clean, (src2, r2.equivalence)


def test_vla_function_parameters_recover_masked_bounds():
    """§5.9 (#vlaparam): a VLA function parameter `T a[n]` decays to a pointer (C), but the runtime extent `n`
    (a prior in-scope integer parameter) is RECOVERED and bound via ptr_extent so the param's `a[i]` promotes
    to masked -- `a[BCIR_CHK(rid, i, n, "fn:a")]` -- the count re-emitted by name. Behaviour-equivalent to
    Clang on both rails. Binding is gated on `n` being a stable (unmutated, non-address-taken) integer param,
    so a mutated size or a non-VLA param stays unchanged (no cross-rail divergence)."""
    from bcir.frontends.cfront import compile_unit, cparse, lower, emit

    def _body(src):
        lu = lower.lower_unit(cparse.parse_unit(src), None)
        return emit.emit_function(lu.functions[next(iter(lu.functions))])
    # a VLA param read masks against n; Clang-equivalent
    src = "unsigned f(unsigned n, unsigned a[n]){ unsigned s=0u; for(unsigned i=0u;i<n;i++) s+=a[i]; return s; }"
    r = compile_unit(src, check_clang=True)
    assert r.equivalence == "match" and r.is_clean, r.equivalence
    assert 'BCIR_CHK(' in _body(src) and ', n, ' in _body(src), _body(src)   # masked vs n, by name
    # a regular (static-dim) array param is unchanged -- NOT masked
    assert "BCIR_CHK" not in _body("unsigned f(unsigned a[5], unsigned i){ return a[i % 5u]; }")
    # a MUTATED size param is not bound (assumed_safe) -- the stability gate; still Clang-equivalent
    src2 = "unsigned f(unsigned n, unsigned a[n]){ n=n+1u; unsigned s=0u; for(unsigned i=0u;i<3u;i++) s+=a[i]; return s; }"
    assert "BCIR_CHK" not in _body(src2)
    assert compile_unit(src2, check_clang=True).equivalence == "match"


def test_multidim_vla_lowering():
    """§5.9 (#vlamd): a multi-dimensional stack VLA `T a[m][n]` (2-D + 3-D) is lowered FAITHFULLY -- each dim
    is snapshotted once, the array is declared IN-BODY as a flat `T a[__ext_total];` sized by the runtime
    product, and the row-major Horner index `i*n + j` (the inner-dim runtime stride, NOT a const) is
    bounds-masked against the total. Behaviour-equivalent to Clang on both rails. A >3-D VLA routes to
    fallback (the dim table caps at 3); the static multi-dim local + the 1-D VLA paths are unchanged."""
    from bcir.frontends.cfront import compile_unit, cparse, lower, emit
    from bcir.frontends.cfront.lower import CLowerError
    from bcir.frontends.cfront.cparse import CParseError
    src = ("unsigned f(unsigned p, unsigned q){ unsigned m=(p&3u)+1u; unsigned n=(q&3u)+1u; unsigned a[m][n];"
           " unsigned s=0u; for(unsigned i=0u;i<m;i++) for(unsigned j=0u;j<n;j++){a[i][j]=i*n+j+p;s+=a[i][j];}"
           " return s; }")
    r = compile_unit(src, check_clang=True)
    assert r.equivalence == "match" and r.is_clean, r.equivalence
    body = emit.emit_function(lower.lower_unit(cparse.parse_unit(src), None).functions["f"])
    assert "a[__bcir_ext2]" in body                         # flat in-body decl sized by the product m*n
    assert "i * __bcir_ext1" in body                        # Horner uses the RUNTIME inner-dim stride (not a const)
    assert "* __bcir_ext1" in body and "__bcir_ext0 * __bcir_ext1" in body   # total = m*n
    # a >3-D VLA falls back (dim table caps at 3)
    try:
        compile_unit("unsigned f(unsigned n){ unsigned a[n][n][n][n]; return a[0][0][0][0]; }", check_clang=False)
        assert False, "a >3-D VLA should route to fallback"
    except (CParseError, CLowerError):
        pass


def test_lvalue_assignment_as_value_extended_forms():
    """§5.9 (#lvassignexpr): an assignment whose target is an ARRAY ELEMENT `a[i]`, a pointer DEREF `*p`, or a
    NESTED member `o.in.x` -- used as a VALUE (`(a[i]=v)+1`, `(*p=v)*2`, chained `a[0]=b[0]=v`) -- yields the
    stored/converted value (the once-resolved lvalue re-read). Extends the single-level-scalar-member case
    (#memassignexpr). Behaviour-equivalent to Clang on both rails; a VOLATILE/MMIO lvalue stays a fallback
    (the re-read would be an extra observable access), and a bitfield / array-of-structs target stays a
    follow-on."""
    from bcir.frontends.cfront import compile_unit
    from bcir.frontends.cfront.lower import CLowerError
    for src in ("unsigned f(unsigned i, unsigned v){ unsigned a[8]={0}; unsigned r=(a[i&7u]=v)+1u; return r+a[i&7u]; }",
                "unsigned f(unsigned v){ unsigned y=0u; unsigned *p=&y; unsigned r=(*p=v)*2u; return r+y; }",
                "unsigned f(unsigned v){ unsigned a[4]={0},b[4]={0}; unsigned r=(a[0]=b[0]=v)+7u; return r+a[0]+b[0]; }",
                "unsigned f(unsigned i, unsigned v){ unsigned a[8]={0}; a[i&7u]=v; unsigned r=(a[i&7u]+=5u)*2u; return r+a[i&7u]; }"):
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)
    # the single-level scalar member case (#memassignexpr) still compiles
    assert compile_unit("struct S{unsigned a;}; unsigned f(unsigned v){ struct S s; return (s.a=v)+1u; }",
                        check_clang=True).equivalence == "match"
    # a VOLATILE lvalue as a value falls back (the re-read would be an extra MMIO access)
    try:
        compile_unit("struct R{volatile unsigned reg;}; unsigned f(volatile struct R *r, unsigned v){ return (r->reg=v)+1u; }",
                     check_clang=False)
        assert False, "a volatile lvalue-as-value should fall back"
    except CLowerError:
        pass


def test_narrow_compound_assignment_as_value_is_the_stored_value():
    """§5.9 (#narrowcompound): the value of a COMPOUND assignment `lv OP= rhs` used as a value is the STORED
    (narrowed) value, not the raw binop result. For a sub-int target (`unsigned char`/`unsigned short` member,
    array element, deref) the store truncates, so the value must be a re-read -- returning the un-narrowed sum
    was a both-rails SILENT MISCOMPILE (clean, wrong, untriggered by the fuzzer). A full-width target needs no
    re-read and is byte-unchanged."""
    from bcir.frontends.cfront import compile_unit
    for src in ("struct N{unsigned char c;}; unsigned f(unsigned v){ struct N s; s.c=200u; return (s.c += v)*3u + s.c; }",
                "unsigned f(unsigned i, unsigned v){ unsigned short a[4]={0}; a[i&3u]=60000u; return (a[i&3u] += v)+7u + a[i&3u]; }",
                "unsigned f(unsigned v){ unsigned char y=250u; unsigned char *p=&y; return (*p += v)*2u + y; }"):
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)
    # the NARROW target adds exactly ONE re-read vs the otherwise-identical FULL-width target (which is
    # unchanged -- its value is the binop result, no spurious re-read)
    from bcir.frontends.cfront import cparse, lower, emit
    def _chk(src):
        return emit.emit_function(lower.lower_unit(cparse.parse_unit(src), None).functions["f"]).count("BCIR_CHK")
    full = _chk("unsigned f(unsigned i, unsigned v){ unsigned a[4]={0}; a[i&3u]=1000u; return (a[i&3u] += v)*2u; }")
    narrow = _chk("unsigned f(unsigned i, unsigned v){ unsigned short a[4]={0}; a[i&3u]=1000u; return (a[i&3u] += v)*2u; }")
    assert narrow == full + 1, (full, narrow)         # the narrow target re-reads the stored (truncated) value


def test_bitfield_assignment_as_value():
    """§5.9 (#bfassignexpr): a BITFIELD member assignment used as a VALUE -- `(s.bits = v) + 1`, compound
    `(s.bits += v) * 2`, signed `(s.c = v)` -- yields the masked / sign-extended STORED field (a re-read via
    bf.get; the compound path re-reads because a bitfield narrows to its bit width). Extends the lvalue-as-value
    forms to bitfield targets. Behaviour-equivalent to Clang on both rails."""
    from bcir.frontends.cfront import compile_unit
    for src in ("struct B{unsigned f:5;}; unsigned g(unsigned v){ struct B s; s.f=0u; return (s.f=v)+1u + s.f; }",
                "struct B{unsigned f:5;}; unsigned g(unsigned v){ struct B s; s.f=10u; return (s.f+=v)*2u + s.f; }",
                "struct B{int c:6;}; int g(int v){ struct B s; s.c=0; return (s.c=v)-1 + s.c; }"):
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)


def test_array_of_structs_field_assignment_as_value():
    """§5.9 (#aosassignexpr): the LAST lvalue-as-value form -- an array-of-structs element field `(a[i].f = v)`
    or a member-array element `(s.arr[i] = v)` used as a value. The lvalue is a STRIDED member (index + element
    stride + field offset), resolved ONCE, stored, then re-read (the value). Combines with the narrow re-read
    for a sub-int field. Behaviour-equivalent to Clang on both rails."""
    from bcir.frontends.cfront import compile_unit
    for src in ("struct P{unsigned x,y;}; unsigned f(unsigned i, unsigned v){ struct P a[4]; a[i&3u].x=0u; return (a[i&3u].x = v)+1u + a[i&3u].x; }",
                "struct P{unsigned x,y;}; unsigned f(unsigned i, unsigned v){ struct P a[4]; a[i&3u].y=10u; return (a[i&3u].y += v)*2u + a[i&3u].y; }",
                "struct S{unsigned arr[4];}; unsigned f(unsigned i, unsigned v){ struct S s; s.arr[i&3u]=0u; return (s.arr[i&3u] = v)+3u + s.arr[i&3u]; }",
                "struct N{unsigned char c; unsigned x;}; unsigned f(unsigned i, unsigned v){ struct N a[4]; a[i&3u].c=0u; return (a[i&3u].c += v)*2u + a[i&3u].c; }"):
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)


def test_signed_function_pointer_return():
    """§5.9 (#signedfnptr): a call through a function-pointer (a funcptr struct member / dispatch, or a funcptr
    param) whose target returns a SIGNED type now types the call RESULT by the return type -- a signed sub-int
    return promotes to `int`, a wide return keeps its width -- so a downstream arithmetic `>>` / `< 0` /
    `(long)`-widen sign-extends. The indirect/member call results were hardcoded uint32 (a both-rails
    miscompile). Behaviour-equivalent to Clang on both rails; an unresolved funcptr return stays uint32."""
    from bcir.frontends.cfront import compile_unit
    for src in ("static int neg(int x){return -x-1;} struct D{int(*op)(int);}; int f(int x){ struct D d; d.op=neg; return d.op(x) >> 1; }",
                "static int neg(int x){return -x-1;} struct D{int(*op)(int);}; int f(int x){ struct D d; d.op=neg; int r=d.op(x); return r<0?-r:r; }",
                "static long lw(int x){return -(long)x-1;} struct D{long(*op)(int);}; long f(int x){ struct D d; d.op=lw; return d.op(x)-1; }"):
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)


def test_quarantine_report_is_the_debugger_trace_surface():
    """§5.12 debugger trace surface: a STRONG override of `bcir_bounds_quarantine` (the ML-layer / debugger
    seam) records each OOB event into the ring without aborting, and `bcir_quarantine_report` reads the ring
    back -- the running total plus each retained event with its `<func>:<array>` site, index, and extent.
    The override path is the only way the program survives multiple OOB accesses; the reader is pure
    observation (it never decides legality). Linked against the real runtime/c/bcir_quarantine.c."""
    if not _CC:
        return
    src = "unsigned g(unsigned i){ unsigned a[8]; for(unsigned k=0u;k<8u;k++) a[k]=k*2u; return a[i]; }"
    from bcir.frontends.cfront import compile_unit
    r = compile_unit(src, check_clang=False)
    name = next(reversed(r.lowered.functions))
    body = r.emitted[name].split("*/\n", 1)[-1]
    with tempfile.TemporaryDirectory() as d:
        # A strong (non-weak) override records the event but does NOT abort, so several OOB accesses survive.
        prog = (f'#include <stdint.h>\n#include <stdlib.h>\n#include <stdio.h>\n#include "bcir_quarantine.h"\n'
                f'{r.source}\n\n{body}\n'
                f'size_t bcir_bounds_quarantine(uint64_t rid,uint64_t index,uint64_t extent,const char *site)\n'
                f'{{ bcir_oob_record_event(rid,index,extent,site); return 0; }}\n'
                f'int main(void){{ (void)bcir_g(8u); (void)bcir_g(40u); bcir_quarantine_report(stdout); return 0; }}\n')
        cpath, epath = os.path.join(d, "e.c"), os.path.join(d, "e")
        open(cpath, "w").write(prog)
        b = subprocess.run([_CC, "-std=c23", "-O2", "-I", _C, cpath,
                            os.path.join(_C, "bcir_quarantine.c"), "-o", epath], capture_output=True, text=True)
        assert b.returncode == 0, b.stderr
        run = subprocess.run([epath], capture_output=True, text=True)
        assert run.returncode == 0, (run.returncode, run.stderr)         # survived: the override did not abort
        out = run.stdout
        assert "2 out-of-bounds event(s)" in out, out                    # the running total
        assert "g:a" in out and "index 8" in out and "index 40" in out, out   # both sites + indices, in order
        assert "out of [0, 8)" in out, out                               # the extent the report resolves


def test_quarantine_recover_is_the_two_truth_crossing():
    """§5.12 the ML-layer / debugger recovery override (the two-truth crossing). The reference override
    (`bcir_quarantine_recover.c`) turns an out-of-bounds access into a CLASSICAL action -- abort or clamp --
    through a RECORDED `decide`: a frozen per-site policy proposes `(action, confidence)`, the crossing
    collapses it at a frozen threshold, and the decision is appended to the audit ring (LANGREF §14: graded
    truth may inform but never silently BECOME the access). A clamp lets the program survive on a valid
    element; an under-confident proposal is rejected and fail-fasts -- exactly as the frozen threshold dictates."""
    if not _CC:
        return
    src = "unsigned g(unsigned i){ unsigned a[8]; for(unsigned k=0u;k<8u;k++) a[k]=k*2u; return a[i]; }"
    from bcir.frontends.cfront import compile_unit
    r = compile_unit(src, check_clang=False)
    name = next(reversed(r.lowered.functions))
    body = r.emitted[name].split("*/\n", 1)[-1]
    with tempfile.TemporaryDirectory() as d:
        prog = (
            '#include <stdint.h>\n#include <stdlib.h>\n#include <stdio.h>\n#include "bcir_quarantine_recover.h"\n'
            f'{r.source}\n\n{body}\n'
            'int main(int argc, char **argv){\n'
            '  static const bcir_recover_rule confident[] = {{"g:a", BCIR_RECOVER_CLAMP, 900}};\n'
            '  static const bcir_recover_rule underconf[] = {{"g:a", BCIR_RECOVER_CLAMP, 300}};\n'
            '  int abort_mode = argc > 1 && argv[1][0] == \'1\';\n'
            '  if (abort_mode) bcir_recover_set_policy(underconf, 1, 500);  /* 300 < 500 -> rejected */\n'
            '  else            bcir_recover_set_policy(confident, 1, 500);  /* 900 >= 500 -> admitted clamp */\n'
            '  unsigned v = bcir_g(40u);                 /* index 40 is out of [0,8) -> the handler decides */\n'
            '  printf("recovered=%u\\n", v);\n'
            '  bcir_decide_report(stdout);\n'
            '  return 0;\n}\n')
        cpath, epath = os.path.join(d, "e.c"), os.path.join(d, "e")
        open(cpath, "w").write(prog)
        b = subprocess.run([_CC, "-std=c23", "-O2", "-I", _C, cpath,
                            os.path.join(_C, "bcir_quarantine.c"), os.path.join(_C, "bcir_quarantine_recover.c"),
                            "-o", epath], capture_output=True, text=True)
        assert b.returncode == 0, b.stderr
        # admitted clamp: confidence 900 >= threshold 500 -> the access lands on a[7] (= 14), program survives,
        # and the recorded decide witnesses the crossing.
        ok = subprocess.run([epath, "0"], capture_output=True, text=True)
        assert ok.returncode == 0 and "recovered=14" in ok.stdout, (ok.returncode, ok.stdout, ok.stderr)
        assert "1 recovery crossing(s)" in ok.stdout, ok.stdout
        assert "g:a" in ok.stdout and "confidence 900/1000 vs threshold 500/1000" in ok.stdout, ok.stdout
        assert "admitted, clamp to index 7" in ok.stdout, ok.stdout
        # rejected: confidence 300 < threshold 500 -> not confident enough to recover -> fail-fast.
        no = subprocess.run([epath, "1"], capture_output=True, text=True)
        assert no.returncode != 0 and "recovery rejected" in no.stderr, (no.returncode, no.stderr)
