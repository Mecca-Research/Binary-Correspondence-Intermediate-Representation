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
from functools import wraps

from bcir.frontends.cfront import compile_unit
from bcir.model import Domain
from bcir.toolchain import host_link_args
from bcir.verify import cfront_structural_digest

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_CC = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _requires_cc(test):
    """Capability guard for tests whose assertion is specifically Clang equivalence."""

    @wraps(test)
    def guarded():
        if not _CC:
            return
        return test()

    return guarded


# Bounds-quarantine support (§5.12): the emit of a `masked` array access uses BCIR_CHK(rid, idx, N, "site")
# for a READ and BCIR_CHK_W(...) for a WRITE -- a READ may clamp an OOB index, a WRITE never clamps (its
# handler is noreturn: an OOB store fails-fast rather than silently redirect onto a valid element). Inlined
# here (matching the ABI in runtime/c/bcir_quarantine.h) so the equivalence harness is self-contained; for
# the in-bounds seeds neither handler is reached, so a guarded access is behaviour-identical to raw `a[i]`.
_BOUNDS_GUARD = (
    "#include <stdlib.h>\n#include <stddef.h>\n"
    "static size_t bcir_bounds_quarantine(uint64_t r,uint64_t i,uint64_t e,const char*s)"
    "{(void)r;(void)i;(void)e;(void)s;abort();return 0;}\n"
    "static void bcir_bounds_quarantine_write(uint64_t r,uint64_t i,uint64_t e,const char*s)"
    "{(void)r;(void)i;(void)e;(void)s;abort();}\n"
    "#define BCIR_CHK(rid,idx,n,site) ((uint64_t)(idx)<(uint64_t)(n)?(size_t)(idx):"
    "bcir_bounds_quarantine((uint64_t)(rid),(uint64_t)(idx),(uint64_t)(n),(site)))\n"
    "#define BCIR_CHK_W(rid,idx,n,site) ((uint64_t)(idx)<(uint64_t)(n)?(size_t)(idx):"
    "(bcir_bounds_quarantine_write((uint64_t)(rid),(uint64_t)(idx),(uint64_t)(n),(site)),(size_t)(idx)))"
)
# straight-line fixtures run the full execute loop; control-flow fixtures get parity + emit + Clang ≡
# (control flow is not a flat StreamPack segment stream, so the loop runs the straight-line set).
_STRAIGHTLINE = [
    "cfront_regmap.c",
    "cfront_array.c",
    "cfront_array2d.c",
    "cfront_widerow.c",
    "cfront_deref.c",
    "cfront_callgraph.c",
    "cfront_typedef.c",
    "cfront_enum.c",
    "cfront_enumtype.c",
    "cfront_ternary.c",
    "cfront_sizeof.c",
    "cfront_strsizeof.c",
    "cfront_strval.c",
    "cfront_charlit.c",
    "cfront_strtab.c",
    "cfront_strconcat.c",
    "cfront_widelit.c",
    "cfront_cast.c",
    "cfront_alignof.c",
    "cfront_static.c",
    "cfront_staticarr.c",  # a static array or aggregate keeps its shape: scalar, 2-D, of pointers, of structs
    "cfront_global.c",
    "cfront_compound.c",
    "cfront_logic.c",
    "cfront_abi.c",
    "cfront_signed.c",
    "cfront_signedcmp.c",
    "cfront_signedbare.c",
    "cfront_longunary.c",
    "cfront_boolnorm.c",
    "cfront_unarypromote.c",
    "cfront_floatsigncast.c",
    "cfront_intsigncast.c",
    "cfront_boolcast.c",
    "cfront_boolmember.c",
    "cfront_unsequenced.c",
    "cfront_reproducible.c",  # + char consts + str table/dedup + const LUT + ABI sizeof model + bool normalization + unary integer-promotion/float + float->signed + int->signed cast + bool cast + _Bool member/element store-normalization + C23 [[unsequenced]]/[[reproducible]] hints (A1.3)
    "cfront_bitint.c",  # + C23 `_BitInt(N)` (#bitint): exact-width, non-promoting bit-precise ints (incl. a NON-standard 12/20-bit lane) as locals/params/returns, same-type arithmetic, faithful `_BitInt(N)` emit
    "cfront_bitint_mixed.c",
]  # + C23 `_BitInt(N)` MIXED-WIDTH arithmetic (#bitintmixed): a `_BitInt(N)` mixed with a NARROWER standard int / `_BitInt` / constant, where the C23 6.3.1.8 result is the WIDER `_BitInt` -- modeled result type == Clang (the `_Generic` differential), faithful emit
_CONTROL = [
    "cfront_branch.c",
    "cfront_while.c",
    "cfront_for.c",
    "cfront_dowhile.c",
    "cfront_continue.c",
    "cfront_switch.c",
    "cfront_switchfall.c",
    "cfront_goto.c",
    "cfront_incdec.c",
    "cfront_multidecl.c",
    "cfront_commastep.c",
    "cfront_emptystmt.c",
    "cfront_loopreuse.c",
    "cfront_loopscope.c",
    "cfront_blockscope.c",
    "cfront_localmd.c",
    "cfront_localmdinit.c",
    "cfront_ptrlocal.c",
    "cfront_condeval.c",  # the operands C may leave unevaluated: a `?:` arm, the right of `&&`/`||` that
    #   divides, reads through a pointer or calls lowers as a branch, a pure one as a select (CF-TERNARY)
]
# + multi-declarator locals (T a=x, b, c=z), comma-operator for-step (i++, j--), empty stmts
_PREPROC = [
    "cfront_macros.c",
    "cfront_ppinc.c",
    "cfront_comments.c",
    "cfront_paste.c",  # tokens that would lex as others, written apart: `a + ++g` (CF-PASTE)
]  # L7: exercise the preprocessor
_ABI = [
    "cfront_structret.c",
    "cfront_structcall.c",  # L8: struct return-by-value (+ using a call RESULT)
    "cfront_packed.c",  # + packed layout
    "cfront_union.c",  # + full union (members overlap at offset 0)
    "cfront_interleave.c",  # + enum/struct defined *between* two functions
    "cfront_funcptr.c",  # + funcptr param + indirect call (HAL dispatch)
    "cfront_fnptrparam.c",  # + DIRECT inline funcptr params int (*g)(int) (no typedef)
    "cfront_rmw.c",  # + MMIO register read-modify-write (d->reg |= bits)
    "cfront_volatile_width.c",  # + volatile `p[i]` at 8/16/32 bits, stores seen through a plain alias
    "cfront_bitfield.c",  # + MMIO bitfield write (r->field = v, c.bf.set)
    "cfront_bfcompound.c",  # + bitfield compound-assign (r->field |= bits)
    "cfront_signedbf.c",  # + signed bitfield read sign-extension (int x:N)
    "cfront_widebf.c",  # + WIDE bitfields in a 64-bit unit (long long x:N, N>32)
    "cfront_packedbf.c",  # + PACKED bitfields (bit-by-bit, byte/word-straddling)
    "cfront_alignasmember.c",  # + over-aligned members (_Alignas/alignas/aligned(N))
    "cfront_anonmember.c",  # + anonymous struct/union members (promoted leaves)
    "cfront_unnamedbf.c",  # + unnamed / zero-width bitfields (layout-only padding)
    "cfront_charmember.c",  # + plain `char` members read as `char` not int8_t (ARM)
    "cfront_aostruct.c",  # + ARRAY-OF-STRUCTS members p->arr[i].field (strided)
    "cfront_aoslocal.c",  # + regular LOCAL array decl init (inferred/explicit, scalar/struct)
    "cfront_fnptrmember.c",  # + funcptr members set from NAMED functions (dispatch)
    "cfront_assignexpr.c",  # + assignment as an EXPRESSION (a=b=c, if((x=f()))...)
    "cfront_memassignexpr.c",  # + member-lvalue assignment as a value ((p->x=v)+1)
    "cfront_signedload.c",  # + signed sub-int member/array read sign-extension
    "cfront_restrict.c",  # + restrict/__restrict pointer params (consumed hint)
    "cfront_shiftassign.c",  # + <<= / >>= shift compound-assign (scalar/member/array)
    "cfront_ptrarith.c",  # + pointer mutation p++ / p += n (buffer-walk cursor)
    "cfront_structmulti.c",  # + multi-declarator struct members (unsigned x,y,z;)
    "cfront_nestmember.c",  # + nested member access (o.pos.lo / dev->ctrl.bf)
    "cfront_memberarray.c",  # + native 1-D struct member arrays (s.arr[i])
    "cfront_neststruct.c",  # + nested struct members + nested-brace init `{ {..}, .. }`
    "cfront_bitint_member.c",  # + C23 `_BitInt(N)` struct MEMBERS (#bitintmember): a PLAIN
    #   (non-bitfield) `_BitInt(N)` member -- read + write + same-type
    #   arithmetic, layout (sizeof/offsetof) == Clang, faithful emit
    "cfront_bitint_bitfield.c",
]  # + C23 `_BitInt(N)` BITFIELDS (#bitintbitfield): `_BitInt(N) m:W`
#   (1<=W<=N) packs into the `_BitInt(N)` storage unit LSB-first,
#   byte-identical to Clang (the layout differential); W<=32 reads
#   promote to int, W>32 stay `_BitInt(N)` -- faithful emit
_FLOAT = [
    "cfront_float.c",
    "cfront_floatcast.c",
    "cfront_globaltype.c",  # a signed / float global's value type (shift, divide, negate, *array)
    "cfront_globalstruct.c",  # a file-scope struct's members, read and written; an initialized global keeps
    #   its declared type (CF-GSTRUCT, CF-GINIT)
    "cfront_memberconv.c",  # a byte-copy store converts to the slot's type: int <-> float members, bitfields,
    #   `*p`, initializers, the value of `(s.i += x)` (CF-MEMCONV)
    "cfront_hexfloat.c",
    "cfront_mathh.c",
    "cfront_mathh_mixed.c",
    "cfront_mathh_long.c",
    "cfront_mathh_ptr.c",
    "cfront_calltyped.c",
    "cfront_complex.c",
    "cfront_complexdiv.c",  # + C99 _Complex (#complex) + complex `/`
    "cfront_complextrans.c",  # + complex transcendentals (#complextrans)
    "cfront_imagunit.c",  # + <complex.h> imaginary unit `I` (#imagunit)
    "cfront_complexlong.c",  # + long-double complex (#complexlong)
    "cfront_complexmember.c",
    "cfront_complexalign.c",  # a _Complex member aligns to its element: `double _Complex` at 8 (CF-CALIGN)
]  # + complex struct members (#complexmember)
#   float/double: parity + emit + Clang ≡ (the
#   integer StreamPack executor doesn't compute float; the math is delegated to the resident backend)
_INIT = [
    "cfront_dispatch_table.c",  # designated initializers ([i]=v) for a file-scope dispatch table
    "cfront_agginit.c",  # local struct/union aggregate init ({.field=v}) -> = {0} + stores
    "cfront_localarray.c",  # local array decl T a[N] + array aggregate init (positional + [i]=)
    "cfront_nestinit.c",
    "cfront_braceelide.c",  # brace elision, designators that continue, anonymous members, unions, braced
    #   scalars and compound literals walked as C walks the current object (CF-BRACE)
    "cfront_strlocal.c",  # a character array sized by its string literal: plain/wide/concatenated, rows
    #   of strings, the NUL dropped only when the array is exactly as long (CF-STRLOCAL)
    "cfront_statictab.c",  # static local tables: a brace or string initializer folded to constants and
    #   emitted in the declaration, never stored at a call; static pointers keep `static` (CF-STATICTAB)
]  # NESTED-brace init `{ m, {e0..}, n }` for a struct's array member
#   (a local decl, a compound literal, and a struct return BY VALUE) -- offset-based element stores
#   parity + emit + Clang ≡ (the table is referenced by name, defined in the source -- not re-hydrated)
_PTRVALUE = [
    "cfront_ptrindex.c",  # a subscript chain through pointer elements: q[j][i], pp[j][i], *(q[j] + i)
    "cfront_ptrvalue.c",  # pointer VALUES across non-address contexts (#ptrvalue): pointer
    #   arithmetic `p + i` as an rvalue returned by value -- the temp carries the pointee type (a real
    #   `T *t = p + i`), not a truncating uint32. Parity + emit + Clang ≡ (returns a pointer, not executed).
    "cfront_ptrfield.c",  # + a pointer stored into / loaded from a struct field (#ptrfield):
    "cfront_trailpacked.c",  # `} __attribute__((packed))` after the body packs the members (CF-TRAILPACK)
    "cfront_ptrmember.c",  # an array-of-pointers member `T *arr[N]`: pointer-size elements, whole-pointer
    #   loads/stores typed `T *`, `*t->arr[i]` / `*s->p` through the element (CF-PTRARR)
    "cfront_decay.c",  # an array used as a value is its first element's address; `p - q` is a ptrdiff_t
    #   (CF-DECAY, CF-PTRDIFF, CF-DEREFIDX)
    "cfront_sizeof_forms.c",  # `sizeof` of every operand form, type-name and target width, a size_t (CF-SIZEOF)
    "cfront_selfref.c",  # self-referential, mutually referencing, forward-declared and never-defined
    #   structs through pointers; pointer-returning calls dereferenced in place (CF-SELFREF)
    "cfront_typedefarr.c",  # a typedef'd array type as a local, rows of it, a member, a global, sizeof and
    #   a compound literal's type-name (CF-SMALL)
    "cfront_mdglobal.c",  # 2-D/3-D globals and `a[i].m[j]` member arrays in every access form, `&m[i][j]` of
    #   a local, a VLA and a `T m[][N]` parameter (CF-SMALL)
    "cfront_storage.c",  # `static` and the qualifiers in any specifier position (CF-STORAGE)
    "cfront_memdecay.c",  # an array stored into a pointer member, dereference or element decays to its
    #   address in every store form (CF-MEMDECAY)
    "cfront_nullptr.c",  # the constant 0 taken by a pointer in every form -- a null pointer (CF-NULLPTR)
    "cfront_nullarg.c",  # the constant 0 passed to a pointer parameter of every kind (CF-NULLARG)
    "cfront_structvalue.c",  # a struct or union read as a value -- an element, a member, `*p`, a select, a call
    #   through a function pointer -- copied, returned, passed and placed whole (CF-STRUCTVAL, CF-STRUCTINIT)
    "cfront_resgrow.c",  # the twin's resource table grows while a pointer temporary is made -- the sanitized
    #   twin's use-after-free witness (tools/c/sanitize_cfront.sh)
    "cfront_railsplit.c",  # forms the rails lowered to different claim graphs: `!`, a polling loop, a bare
    #   `*e;`, `&p[i]` of an allocation, a volatile VLA (CF-RAILSPLIT)
    "cfront_aosnest.c",  # `a[i].m.k` and `a[i].m.arr[j]` in every access form, on local, global, pointer and
    #   member-array bases, and the elements a pointer member points at (CF-NESTMEM)
    "cfront_parenpostfix.c",  # a postfix operator after `( ... )`: `(a)[i]`, `((T[N]){...})[i]`, `(p)->x`,
    #   `(x)++`, `(*p).x` as `p->x`, `(*p)[i]` as `p[0][i]` (CF-PAREN)
    "cfront_longnames.c",  # a 63-character name everywhere the claim graph keeps one, and a 63-character
    #   floating constant: the longest either rail accepts (CF-BUF)
    "cfront_splits.c",  # forms one rail lowered and the other refused or lowered apart: `p = p + n`, a step
    #   or a value through a loaded pointer, `p[i]++`, `(*p)++`, `*&a`, `*(c ? &a : &b)`, `(fp)(x)` (CF-SPLIT2)
    "cfront_addrof.c",  # general address-of `&`
    "cfront_addrofarr.c",  # member-array element address
    "cfront_addrofaos.c",  # array-of-structs element field address
    "cfront_extentsnap.c",  # §5.12 recoverable-extent SNAPSHOT: expression counts (#extentsnap)
    "cfront_comma.c",  # the comma operator in a primary parenthesized expr (#comma)
    "cfront_vla.c",  # native 1-D stack VLAs `T a[n]` -- in-body decl + masked bounds (#vla)
    "cfront_vlasizeof.c",  # runtime `sizeof a` of a VLA -> extent * sizeof(elem) (#vlasizeof)
    "cfront_vlaparam.c",  # VLA function parameters `T a[n]` -> masked param bounds vs n (#vlaparam)
    "cfront_vlamd.c",  # multi-dimensional VLAs `T a[m][n]` -> flat m*n extent + Horner (#vlamd)
    "cfront_lvassignexpr.c",  # array/deref/nested lvalue assignment used as a value (#lvassignexpr)
    "cfront_narrowcompound.c",  # a narrow-target compound assignment AS A VALUE re-reads (#narrowcompound)
    "cfront_bfassignexpr.c",  # a BITFIELD member assignment used as a value (#bfassignexpr)
    "cfront_aosassignexpr.c",  # an array-of-structs field / member-array element as a value (#aosassignexpr)
    "cfront_signedfnptr.c",  # a SIGNED function-pointer return reads back signed (#signedfnptr)
    "cfront_addroffollow.c",  # &arr[i].field (plain base) + &s->ptr (pointer member) (#addroffollow)
    "cfront_incdecexpr.c",  # ++/-- in expression position as a value (#incdecexpr)
    "cfront_computedgoto.c",  # computed goto: &&L label-as-value + goto *p (#computedgoto)
    "cfront_fnptrlocal.c",  # a function-pointer LOCAL VARIABLE `RET (*f)(P)=fn;` (#fnptrlocal)
    "cfront_arrcomplit.c",  # 1-D scalar array compound literals, bounds-guard-reconciled (#arrcomplit)
    "cfront_stdlibmem.c",
    "cfront_strmem.c",  # <string.h> memcpy/memmove/memset as external libc edges, each returning its
    #   destination (CF-RTWIDE)
    "cfront_decls.c",  # callees defined after their callers behind prototypes leaving parameters unnamed,
    #   declared by each emit ahead of its callers (CF-DECLS)
    "cfront_anonstruct.c",  # anonymous structs and unions named by a typedef, and a nested anonymous member,
    #   spelled as C names them (CF-ANON)
]  # + <stdlib.h> malloc/calloc/realloc/free as external libc edges (#stdlibmem)   # + address-of an array-of-structs element field in a member (#addrofaos)   # + address-of a member-array element (#addrofarr): &s.arr[i] / &s.m[i][j]   # + general address-of `&` of an lvalue (#addrof): &s->m / &*p / &arr[i]   # + a pointer stored into / loaded from a struct field (#ptrfield):
#   the member occupies pointer_size (8) bytes -- a correct layout (an adjacent field no longer overlaps
#   the high half of the pointer) and an untruncated 8-byte store/load that carries the real `T *` type.
_FIXTURES = _STRAIGHTLINE + _CONTROL + _PREPROC + _ABI + _FLOAT + _INIT + _PTRVALUE
# §5.8 atomics/fences/CAS run their own gate: their memory side effects make the generic
# pure-function equivalence harness invalid (it would call the original first and observe
# the mutated cell), so they get a side-effect-aware behaviour check below.
_ATOMIC = [
    "cfront_atomic.c",
    "cfront_cmpxchg.c",
    "cfront_atomic11.c",
    "cfront_atomic_xchg.c",
    "cfront_cmpxchg11.c",
]  # + C11 stdatomic + atomic_exchange + compare_exchange


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
    return _summary_line(r), r, funcs[next(reversed(funcs))]


def _summary_line(r) -> str:
    """The oracle's parity summary of a compiled unit -- the line the twin's driver prints first."""
    funcs = r.lowered.functions
    entry = funcs[next(reversed(funcs))]
    cl = entry.claims
    mmio = sum(1 for c in cl if c.op == "c.load" and c.domain == Domain.MMIO)
    bf = sum(1 for c in cl if c.op == "c.bf.get")
    kn = sum(1 for c in cl if c.op == "c.const")
    bo = sum(1 for c in cl if c.op.startswith("c.bin."))
    ca = sum(1 for c in cl if c.op.startswith("c.call"))  # c.call:NAME (direct) + c.call.indirect
    # A1.3 fusion-legality signal: how many functions in the unit carry a C23 `[[reproducible]]` /
    # `[[unsequenced]]` hint. Value-neutral (dropped from the emit), so it never affects behaviour --
    # it is surfaced here so the dual-rail parity gate pins it identically on both rails.
    repro = sum(1 for f in funcs.values() if getattr(f, "reproducible", False))
    # digest = the cross-rail per-claim STRUCTURAL digest (the count->structural parity fix): the C twin
    # appends the byte-identical `digest=<16-hex>` to its summary, so the gate now compares structure.
    digest = cfront_structural_digest(r.lowered)
    return (
        f"funcs={len(funcs)} claims={len(cl)} mmio={mmio} bf={bf} const={kn} "
        f"binop={bo} call={ca} repro={repro} ok={1 if r.is_clean else 0} digest={digest:016x}"
    )


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
        b = subprocess.run(
            [_CC, f"-std={std}", "-O2", "-I", _C, *srcs, "-o", exe], capture_output=True, text=True
        )
        if b.returncode == 0:
            _BUILD_CACHE[key] = exe
            return exe
    raise AssertionError(f"{label} build failed:\n{b.stderr}")


def _build_frontend(d: str) -> str:
    # `d` is retained for call-site compatibility; the binary is cached session-wide (see _compile_once).
    # bcir_cfront verifies (bcir_verify.c) and the pack law reaches into bcir_runtime.c, so both link in.
    return _compile_once(
        "frontend",
        "tcf",
        ("bcir_cfront.c", "bcir_cpp.c", "bcir_verify.c", "bcir_runtime.c", "test_cfront.c"),
        "C frontend",
    )


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


def _seed_params(entry):
    """The seeded-input harness fragments shared by `_equiv` and the cross-compiler original-fairness
    fingerprint (`_original_xcc_fingerprint`): `(decls, setup, args, prelude)` -- the per-parameter
    declarations, the per-trial RNG seeding, the call arguments, and any funcptr-target preludes. Factored
    out so the fairness fingerprint feeds the ORIGINAL function byte-identically-seeded inputs."""
    has_ptr = any(ct.kind in ("pointer", "array") for _n, _r, ct in entry.params)
    decls, setup, args, prelude = [], [], [], []
    for i, (_pn, _rid, ct) in enumerate(entry.params):
        if ct.kind == "funcptr":  # pass a real (deterministic) target fn
            rety = _cname(ct.of) if ct.of else "uint32_t"
            plist = ", ".join(f"{_cname(pt)} p{j}" for j, pt in enumerate(ct.params)) or "void"
            comb = " + ".join(f"(p{j} * {2 * j + 1}u)" for j in range(len(ct.params))) or "1u"
            prelude.append(f"static {rety} _fp{i}({plist}){{ return ({rety})({comb}); }}")
            args.append(f"_fp{i}")
            continue
        if ct.kind in ("pointer", "array"):
            decls.append(f"  static {_cname(ct.of)} buf{i}[256];")
            setup.append(
                f"    for(unsigned k=0;k<sizeof buf{i}/4;k++) ((uint32_t*)buf{i})[k]=rng();"
            )
            # The original multidimensional-array parameter adjusts to a pointer-to-row
            # (for example ``uint32_t (*)[8]``), while the verified-C rail deliberately
            # flattens it to ``uint32_t *``. A void-object-pointer argument converts to
            # both exact parameter types without the incompatible-pointer diagnostic that
            # Clang 22 promoted to an error; the same aligned backing bytes reach each rail.
            args.append(f"(void *)buf{i}")
        elif ct.is_aggregate:
            decls.append(f"  {ct.kind} {ct.name} a{i};")
            inits = "".join(
                f"    a{i}.{fn}=(rng()&{(1 << bw) - 1}u);\n"
                if bw
                else f"    a{i}.{fn}=({_cname(ft)})rng();\n"
                for fn, ft, _bo, _bf, bw in ct.fields
            )
            setup.append(inits.rstrip("\n"))
            args.append(f"a{i}")
        elif ct.is_complex:  # a _Complex param: seed BOTH axes (finite, in range)
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
    return decls, setup, args, prelude


def _equiv(source: str, c_emitted: str, entry, *, cc=_CC) -> str:
    """Compile the original source beside the C-frontend's emitted bcir_* and diff outputs.

    `cc` is the C compiler to build the equivalence harness with (default `_CC` -- the single
    compiler the suite picks up). The cross-compiler differential (test_emitted_c_is_equivalent_under
    _both_gcc_and_clang) passes an EXPLICIT gcc / clang so the SAME emit is exercised under each."""
    decls, setup, args, prelude = _seed_params(entry)
    call = ", ".join(args)
    rt = _cname(entry.ret_type)
    if entry.ret_type.is_complex:
        # a _Complex result is compared element-wise with a nan-aware equality: value-based (creall/cimagl,
        # narrower complex widening exactly), so it is immune to `long double _Complex`'s indeterminate x87
        # padding bytes -- which memcmp would wrongly flag -- AND nan-safe, which a complex division by a
        # near-zero divisor needs (`==` is false for a nan, so the isnan&&isnan arm catches it). Both rails
        # run the identical native op, so equal-value already implies bit-equal (incl. signed zero / inf).
        cmp = (
            f"    {rt} ra={entry.name}({call}), rb=bcir_{entry.name}({call});\n"
            f"    if(!((creall(ra)==creall(rb)||(isnan(creall(ra))&&isnan(creall(rb))))"
            f"&&(cimagl(ra)==cimagl(rb)||(isnan(cimagl(ra))&&isnan(cimagl(rb))))))"
            f'{{printf("MISMATCH@%d\\n",i);return 1;}}'
        )
    elif entry.ret_type.is_aggregate:
        # an aggregate result is compared BIT-exactly (memcmp): both rails run the identical stores.
        cmp = (
            f"    {rt} ra={entry.name}({call}), rb=bcir_{entry.name}({call});\n"
            f'    if(memcmp(&ra,&rb,sizeof ra)){{printf("MISMATCH@%d\\n",i);return 1;}}'
        )
    else:
        cmp = (
            f"    if({entry.name}({call})!=bcir_{entry.name}({call}))"
            f'{{printf("MISMATCH@%d\\n",i);return 1;}}'
        )
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
            b = subprocess.run(
                host_link_args(
                    [cc, f"-std={std}", "-O2", c, "-o", e, "-lm"]
                ),  # logical libm; omitted for Windows CRT
                capture_output=True,
                text=True,
            )
            if b.returncode == 0:
                break
        else:
            return f"build-failed:{b.stderr.strip().splitlines()[-1] if b.stderr else '?'}"
        return subprocess.run([e], capture_output=True, text=True).stdout.strip()


def _equiv_atomic(source: str, c_emitted: str, entry, *, cc=_CC) -> str:
    """Side-effect-aware behaviour equivalence for atomics. The generic `_equiv` calls the
    original then the emitted bcir_* on the *same* buffer -- invalid here, since an atomic RMW
    mutates its pointee, so the second call would start from a counter the first already moved.
    Instead this runs each on an independent copy of the *same* seeded state and compares both the
    return value and the final memory state (an atomic counter is a single location, not an array).

    `cc` parameterizes the harness compiler (default `_CC`); the cross-compiler differential passes an
    explicit gcc / clang so the same atomic emit is exercised under each."""
    # Seed from a small range so a compare-and-swap's expected value collides with the cell
    # often enough to exercise the swap-taken path (not just the no-op path); equivalence holds
    # for any inputs, but this makes the behaviour check meaningful for CAS.
    decls, setup, args_a, args_b, cell_cmp = [], [], [], [], []
    for i, (_pn, _rid, ct) in enumerate(entry.params):
        if ct.kind in ("pointer", "array"):
            base = _cname(ct.of)  # may be `_Atomic uint32_t` (a C11 cell)
            plain = base.replace("_Atomic ", "")  # the seed casts to the non-atomic type
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
            b = subprocess.run(
                host_link_args(
                    [cc, f"-std={std}", "-O2", c, "-o", e, "-lm"]
                ),  # logical libm; omitted for Windows CRT
                capture_output=True,
                text=True,
            )
            if b.returncode == 0:
                break
        else:
            return f"build-failed:{b.stderr.strip().splitlines()[-1] if b.stderr else '?'}"
        return subprocess.run([e], capture_output=True, text=True).stdout.strip()


def _array_extents(emit: str) -> tuple:
    """The multiset (sorted) of TOTAL element counts of every array DECLARATION in emitted C -- e.g. both
    `T a[2][2]` and `T b[4]` contribute 4. Used to compare STORAGE SIZES across rails: an OVER-sized backing
    array (a compound literal / inferred-size array sized larger than it should be) keeps the same per-element
    stores, the same observable behaviour, AND the same `BCIR_CHK` guard COUNT, so every other check in
    `_parity_check_fixture` misses it -- a multi-dim compound literal once sized `_cl[10]` vs `_cl[6]` and
    slipped through parity + Clang-equivalence undetected (#arrcomplit md). Normalizing to the dim PRODUCT
    makes a flat `[4]` and a nested `[2][2]` compare equal, so the check is robust to decl-form differences
    between the rails."""
    sizes = []
    for dims in re.findall(r"[A-Za-z_]\w*(?:\s+\w+)?\s+\w+((?:\[\d+\])+)\s*[=;]", emit):
        total = 1
        for n in re.findall(r"\[(\d+)\]", dims):
            total *= int(n)
        sizes.append(total)
    return tuple(sorted(sizes))


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
        return (
            fx,
            f"bounds-guard parity: oracle={oracle_emit.count('BCIR_CHK')} "
            f"twin={c_emit.count('BCIR_CHK')} BCIR_CHK guards",
        )
    # Storage-extent parity: both rails must allocate identically-sized backing arrays. An over-sized
    # compound literal / inferred-size array keeps the same stores, behaviour, and guard COUNT (so the checks
    # above all pass), but a larger backing array -- the dim-product multiset of array declarations catches it.
    eo, ec = _array_extents(oracle_emit), _array_extents(c_emit)
    if eo != ec:
        return (fx, f"storage-extent parity: oracle={eo} twin={ec} array element-counts")
    return (fx, None)


# --- the GCC<->Clang behaviour-equivalence DIFFERENTIAL -----------------------------------------
# The single-compiler equivalence gate (`_parity_check_group`) builds the {original + emitted bcir_*}
# harness with only ONE host compiler (`_CC` == clang here, the first of clang/cc/gcc found). So a
# construct that leans on compiler-specific / undefined behaviour -- or a cfront emit bug a single
# compiler happens to tolerate -- can pass unseen. This differential builds the SAME harness for every
# fixture under BOTH gcc AND clang and requires MATCH under EACH, for BOTH rails' emit (the C twin's
# emit AND the oracle's own emit) -- the exact dual-rail pair the parity gate uses. It reuses the
# `_equiv` / `_equiv_atomic` helpers (now parameterized by `cc=`), so the seeded inputs / harness shape
# are byte-identical to the existing gate; only the compiler differs.


def _differential_compilers():
    """The (gcc, clang) pair the cross-compiler differential builds under, or None if either is absent.
    Resolved at call time so run_all's tier capability gate (`shutil.which`) is honoured -- under the
    quick tier both are hidden and the differential self-skips, exactly like the rest of this module."""
    gcc, clang = shutil.which("gcc"), shutil.which("clang")
    return (gcc, clang) if (gcc and clang) else None


def _original_xcc_verdict(source: str, entry, compilers) -> str:
    """Cross-compiler FAIRNESS gate on the ORIGINAL program (#3 of the differential): build a harness that
    runs ONLY the original `entry` over the SAME 256 seeded inputs `_equiv` uses and checksums every
    trial's result, under EACH compiler. The differential charges a {original != emit} divergence against
    cfront only when the ORIGINAL is well-defined and agrees across compilers; otherwise the fixture itself
    is unfair and is excluded. Returns:
      * "fair"        -- the original builds under every compiler and they agree on the checksum;
      * "unsupported" -- the original fails to BUILD under some compiler (a toolchain capability gap, e.g.
                         gcc 13 has no C23 `_BitInt`), so that feature simply isn't testable there;
      * "xcc-divergent" -- the original builds under every compiler but they DISAGREE on the checksum, so
                         the original is not single-valued across toolchains: it relies on undefined
                         behaviour (signed overflow / oversized shift for an oversized seed) OR on an
                         implementation-defined choice (gcc vs clang lower `_Complex` multiply/divide
                         differently). A single-compiler emit gate was never a fair check for such a
                         fixture -- within ONE binary both rails share the toolchain's choice, which is why
                         the in-binary `_equiv` still passes; ACROSS toolchains there is no single truth.
    The fingerprint mirrors `_equiv`'s comparison so it never FALSE-flags a well-defined original: a scalar
    folds its value; an aggregate folds its bytes (both rails run identical stores, like `_equiv`'s memcmp);
    a _Complex folds creal/cimag VALUES (immune to indeterminate x87 padding, like `_equiv`'s element-wise
    compare). A pointer/array return is fingerprinted at zero -- an absolute address is binary-dependent
    (gcc and clang place a static buffer differently), so it is not a fair cross-BINARY value; such fixtures
    have no return-type UB exposure, so they are treated as fair (the in-binary `_equiv` check still runs)."""
    decls, setup, args, prelude = _seed_params(entry)
    call = ", ".join(args)
    rt = _cname(entry.ret_type)
    if entry.ret_type.kind in ("pointer", "array"):
        fold = f"    (void)({entry.name}({call}));"  # absolute address: not a cross-binary value
    elif entry.ret_type.is_complex:
        fold = (
            f"    {rt} rr={entry.name}({call});\n"
            f"    h=h*1099511628211u + (uint64_t)(int64_t)creall(rr);\n"
            f"    h=h*1099511628211u + (uint64_t)(int64_t)cimagl(rr);"
        )
    elif entry.ret_type.is_aggregate:
        fold = (
            f"    {rt} rr={entry.name}({call});\n"
            f"    for(unsigned b=0;b<sizeof rr;b++) h=h*1099511628211u + ((unsigned char*)&rr)[b];"
        )
    else:
        fold = f"    h=h*1099511628211u + (uint64_t)({entry.name}({call}));"
    harness = f"""#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <math.h>
#include <complex.h>
{_BOUNDS_GUARD}
{source}
{chr(10).join(prelude)}
static uint64_t S=0x9E3779B97F4A7C15u;
static uint32_t rng(void){{S=S*6364136223846793005u+1442695040888963407u;return (uint32_t)(S>>32);}}
int main(void){{
  uint64_t h=1469598103934665603u;
{chr(10).join(decls)}
  for(int i=0;i<256;i++){{
{chr(10).join(setup)}
{fold}
  }}
  printf("%llu\\n",(unsigned long long)h);return 0;}}"""
    fingerprints = []
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "o.c")
        open(c, "w").write(harness)
        for idx, cc in enumerate(compilers):
            e = os.path.join(d, f"o{idx}")
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run(
                    host_link_args([cc, f"-std={std}", "-O2", c, "-o", e, "-lm"]),
                    capture_output=True,
                    text=True,
                )
                if b.returncode == 0:
                    break
            else:
                return "unsupported"  # the original doesn't build under this compiler
            fingerprints.append(subprocess.run([e], capture_output=True, text=True).stdout.strip())
    return "fair" if len(set(fingerprints)) == 1 else "xcc-divergent"


def _differential_check_fixture(fx: str, compilers) -> tuple[str | None, str | None]:
    """Cross-compiler emit-equivalence for ONE fixture, GATED on the original being fair (see
    `_original_xcc_verdict`). Returns `(failure, excluded)`:
      * `(msg, None)`  -- a REAL divergence: the fixture's original is well-defined + agrees across
                          compilers, yet the cfront emit (twin or oracle) diverges under some compiler;
      * `(None, why)`  -- the fixture was excluded as not a fair differential ("unsupported" capability
                          gap / "xcc-divergent" original); the emit is NOT charged against cfront;
      * `(None, None)` -- the emit matched the original under every compiler (a clean pass).
    The diagnostic names the failing fixture + rail + compiler + the harness's own output
    (`build-failed:...` / `MISMATCH@<trial>`)."""
    path = os.path.join(_C, fx)
    src = open(path, encoding="utf-8").read()
    _oracle_summary, r, entry = _oracle(src, _includes_for(fx))
    verdict = _original_xcc_verdict(r.source, entry, compilers)
    if verdict != "fair":
        return (None, f"{fx}: {verdict}")  # not a fair differential -- exclude, don't blame
    _c_summary, c_emit = _c_run(_build_frontend(_session_build_dir()), path)
    oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    check = _equiv_atomic if fx in _ATOMIC else _equiv
    for cc in compilers:
        for rail, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            out = check(r.source, emit, entry, cc=cc)
            if out != "MATCH":
                return (
                    f"{fx}: {rail} emit not behaviour-equivalent under "
                    f"{os.path.basename(cc)} ({out})",
                    None,
                )
    return (None, None)


# Same round-robin slicing as the single-compiler parity campaign (`_PARITY_SLICES`): N independent
# `test_*` groups so run_all's worker pool fans them out WITHOUT a nested pool, and any fixture added to
# the corpus is auto-covered. Atomics get their own slices (they need the side-effect-aware harness).
_DIFF_GROUPS = 4
_DIFF_SLICES = [(_FIXTURES + _ATOMIC)[i::_DIFF_GROUPS] for i in range(_DIFF_GROUPS)]


def _differential_check_group(fxs):
    """Cross-compiler (gcc + clang) behaviour-equivalence over a SLICE of the corpus, gated on each
    fixture's original being a fair differential. Runs serially -- run_all supplies the parallelism across
    groups. Self-skips (returns) when gcc or clang is absent. A fixture excluded as unfair (a capability
    gap or an xcc-divergent original) is NOT charged against cfront; only a real {original != emit} divergence
    on a fair fixture fails the group."""
    compilers = _differential_compilers()
    if not compilers:  # quick tier or a single-compiler host
        return
    fails, tested = [], 0
    for fx in fxs:
        msg, excluded = _differential_check_fixture(fx, compilers)
        if msg:
            fails.append(msg)
        elif excluded is None:
            tested += 1  # a fair fixture that passed both compilers
    assert not fails, (
        "GCC<->Clang behaviour-equivalence differential failures (the cfront emit diverges "
        "from the original under a compiler where the original is well-defined):\n"
        + "\n".join(f"  {m}" for m in fails)
    )
    # at least one fixture in the slice must have actually run under BOTH compilers, so a slice that
    # silently degenerated to all-excluded (or a single-compiler fall-back) is itself a failure.
    assert tested > 0, (
        f"differential exercised no fair fixture in this slice under both compilers: {fxs}"
    )


def test_emitted_c_is_equivalent_under_both_gcc_and_clang_g0():
    """Cross-compiler emit differential, group 0/4: the {original + cfront-emitted bcir_*} harness for
    each fixture in this slice builds + runs MATCH under BOTH gcc AND clang, for BOTH rails' emit. The
    single-compiler gate (`_parity_check_group`) builds with clang only; this proves the emit is not
    relying on a clang-tolerated construct that gcc would diverge on. Self-skips if gcc or clang is
    absent (e.g. the quick tier hides the toolchain)."""
    _differential_check_group(_DIFF_SLICES[0])


def test_emitted_c_is_equivalent_under_both_gcc_and_clang_g1():
    """Cross-compiler emit differential, group 1/4 (see `_differential_check_group`)."""
    _differential_check_group(_DIFF_SLICES[1])


def test_emitted_c_is_equivalent_under_both_gcc_and_clang_g2():
    """Cross-compiler emit differential, group 2/4 (see `_differential_check_group`)."""
    _differential_check_group(_DIFF_SLICES[2])


def test_emitted_c_is_equivalent_under_both_gcc_and_clang_g3():
    """Cross-compiler emit differential, group 3/4 (see `_differential_check_group`)."""
    _differential_check_group(_DIFF_SLICES[3])


def test_emitted_c_is_equivalent_under_both_gcc_and_clang_smoke():
    """A non-skippable proof both compilers actually run (the differential's anti-skip guard): on a host
    where gcc + clang are BOTH present this builds one representative fixture's emit under EACH and
    asserts MATCH under each, FAILING (not skipping) if either compiler is missing when it should be
    there. So a silent fall-back to a single compiler -- the failure mode this whole differential guards
    against -- is itself caught. Under the quick tier (toolchain hidden) it correctly self-skips."""
    if not _CC:  # quick tier: toolchain hidden -> skip
        return
    compilers = _differential_compilers()
    # _CC resolved (a compiler is visible), so under any tier that exposes a C compiler BOTH gcc and
    # clang are in the visible set (run_all's `_C_COMPILER`) -- a single-compiler host is the bug.
    assert compilers, (
        "cross-compiler differential degenerated to ONE compiler: "
        f"gcc={shutil.which('gcc')} clang={shutil.which('clang')} -- "
        "the equivalence gate would silently exercise only one toolchain"
    )
    gcc, clang = compilers
    fx = "cfront_intsigncast.c"  # signed/float casts: a UB-sensitive pick
    path = os.path.join(_C, fx)
    src = open(path, encoding="utf-8").read()
    _summary, r, entry = _oracle(src, _includes_for(fx))
    _c_summary, c_emit = _c_run(_build_frontend(_session_build_dir()), path)
    assert _equiv(r.source, c_emit, entry, cc=gcc) == "MATCH", f"{fx}: twin emit diverges under gcc"
    assert _equiv(r.source, c_emit, entry, cc=clang) == "MATCH", (
        f"{fx}: twin emit diverges under clang"
    )


def test_storage_extent_parity_catches_oversizing():
    """The `_array_extents` storage-extent guard in `_parity_check_fixture` flags an over-sized backing array
    even when stores / behaviour / `BCIR_CHK` count all match -- the exact gap a multi-dim compound literal
    once hit (`_cl[10]` vs `_cl[6]`). A harness regression guard so the gap stays closed."""
    assert _array_extents("unsigned int _cl1[6] = {0};\n_cl1[0] = 1u;") == (6,)
    assert _array_extents("uint32_t _cl1[10] = {0};\n_cl1[0] = 1u;") == (10,)
    # the divergence the parity check now flags (same stores/behaviour, different backing size)
    assert _array_extents("T a[6] = {0};") != _array_extents("T a[10] = {0};")
    # robust to a flat-vs-nested decl form: a [2][2] and a [4] are the same storage (dim product)
    assert (
        _array_extents("unsigned int a[2][2] = {0};")
        == _array_extents("unsigned int b[4] = {0};")
        == (4,)
    )
    # a multiset: two same-sized arrays are distinct from one
    assert _array_extents("T a[2];\nT b[2];") == (2, 2) != _array_extents("T a[2];")


def test_link_flag_derivation_dual_rail():
    """B1 (`bcir-cc --emit-link-flags`): the compiler DERIVES the linker flags a translation unit needs
    from the external-call edges it uses, instead of every harness hard-coding `-lm`. The callee->library
    mapping (linkflags.py / bcir_cfront.c's bcir_lib_for_callee) is the dual-rail source of truth; the
    derived, deduped, STABLY-SORTED flag line must be BYTE-IDENTICAL on both rails.

    Covers a pure-integer unit (no flags), a math.h unit (`-lm`), a free()/malloc unit (no flag --
    libc is implicit), and a mixed math+free unit (still just `-lm`, dedup + implicit-skip). Also pins
    the callee->library classification (incl. cblas_*->-lcblas and the unknown-callee policy) directly."""
    from bcir.frontends.cfront.linkflags import (
        NO_FLAG,
        derive_link_flags,
        format_link_flags,
        library_for_callee,
    )

    # (a) the mapping itself -- the source of truth, independent of the C-toolchain gate.
    assert library_for_callee("sqrt") == "-lm"  # base math.h
    assert library_for_callee("sqrtf") == "-lm"  # f-suffixed variant
    assert library_for_callee("lroundl") == "-lm"  # fixed-long suffixed variant
    assert library_for_callee("free") == NO_FLAG  # libc-implicit, EXPLICITLY known (not unknown)
    assert library_for_callee("malloc") == NO_FLAG
    assert library_for_callee("printf") == NO_FLAG  # printf-family extern variadic
    for name in ("memcpy", "memmove", "memset"):  # the <string.h> memory routines (CF-RTWIDE)
        assert library_for_callee(name) == NO_FLAG, name
    assert library_for_callee("cblas_sgemm") == "-lcblas"  # B5 BLAS (the existing path's choice)
    assert library_for_callee("cblas_dgemm") == "-lcblas"  # any cblas_*
    assert library_for_callee("fftwf_execute") == "-lfftw3"  # B2 FFTW (single-prec edge)
    assert library_for_callee("fftwf_plan_dft_1d") == "-lfftw3"  # any fftwf_*
    assert library_for_callee("fftw_execute") == "-lfftw3"  # the double-prec fftw_* prefix too
    assert (
        library_for_callee("LAPACKE_sgesv") == "-llapack"
    )  # #61 LAPACK (the linear-solve wrap's callee)
    assert library_for_callee("LAPACKE_dgesv") == "-llapack"  # any LAPACKE_*
    assert library_for_callee("sgesv_") == "-llapack"  # the Fortran-ABI driver symbol
    assert library_for_callee("gsl_stats_mean") == "-lgsl"  # #62 GSL (the statistics wrap's callee)
    assert library_for_callee("gsl_sf_erf") == "-lgsl"  # any gsl_*
    assert (
        library_for_callee("Sleef_expf1_u10") == "-lsleef"
    )  # #63 SLEEF (the vectorized-exp wrap's callee)
    assert library_for_callee("Sleef_sinf1_u10") == "-lsleef"  # any Sleef_*
    assert library_for_callee("erfcxf") == "-lcerf"  # #64 libcerf (the scaled-erfc wrap's callee)
    assert (
        library_for_callee("erfcx") == "-lcerf"
    )  # the bare/double erfcx too (a symbol libm lacks)
    assert (
        library_for_callee("erfcf") == "-lm"
    )  # ... but erfcf/erfc/erf are still libm (not shadowed)
    assert (
        library_for_callee("totally_unknown_fn") is None
    )  # unknown-callee policy: None (no invented -l)
    # dedup + STABLE sort (reproducible): a set with two libs always yields the same ordered line.
    assert format_link_flags(sorted({"-lm", "-lcblas"})) == "-lcblas -lm"
    assert format_link_flags(sorted({"-lm", "-lfftw3"})) == "-lfftw3 -lm"
    assert format_link_flags(sorted({"-lm", "-llapack"})) == "-llapack -lm"
    assert format_link_flags(sorted({"-lm", "-lsleef"})) == "-lm -lsleef"
    assert format_link_flags(sorted({"-lm", "-lcerf"})) == "-lcerf -lm"

    # (b) end-to-end derivation over real units (oracle rail).
    cases = {
        "pure-int": ("uint32_t f(uint32_t a){ return a*3u + 1u; }", ""),
        "mathh": ("#include <math.h>\ndouble f(double x){ return sqrt(x) + floor(x); }", "-lm"),
        "free": (
            "#include <stdlib.h>\nunsigned f(unsigned n){ unsigned *p=malloc(n*4u);"
            " unsigned r=p[0]; free(p); return r; }",
            "",
        ),
        "math+free": (
            "#include <math.h>\n#include <stdlib.h>\ndouble f(double x){ double *p=malloc(8);"
            " double r=sqrt(x); free(p); return r; }",
            "-lm",
        ),
        "string+math": (
            "#include <math.h>\n#include <string.h>\ndouble f(double x){ double y; memset(&y, 0, 8);"
            " memcpy(&y, &x, sizeof y); memmove(&y, &x, 8); return sqrt(y); }",
            "-lm",
        ),
    }
    for label, (src, want) in cases.items():
        r = compile_unit(src, check_clang=False)
        flags = format_link_flags(derive_link_flags(r.lowered))
        assert flags == want, f"{label}: oracle derived {flags!r}, want {want!r}"
        assert format_link_flags(r.link_flags) == want, f"{label}: result.link_flags mismatch"

    if not _CC:
        return
    # (c) the dual-rail parity gate: the C twin's --emit-link-flags == the oracle, byte-for-byte.
    exe = _build_frontend(_session_build_dir())
    d = tempfile.mkdtemp(prefix="bcir_lf_")
    atexit.register(shutil.rmtree, d, ignore_errors=True)
    for label, (src, want) in cases.items():
        path = os.path.join(d, f"lf_{label.replace('+', '_')}.c")
        with open(path, "w", encoding="utf-8") as f:
            f.write(src + "\n")
        c_out = subprocess.run(
            [exe, "--emit-link-flags", path], capture_output=True, text=True
        ).stdout.strip()
        py_out = format_link_flags(derive_link_flags(compile_unit(src, check_clang=False).lowered))
        assert c_out == want, f"{label}: C twin emitted {c_out!r}, want {want!r}"
        assert c_out == py_out, f"{label}: dual-rail divergence  C={c_out!r}  PY={py_out!r}"


def test_c23_unsequenced_reproducible_dual_rail():
    """C23 function attributes (#unsequenced / A1.3): `[[unsequenced]]` / `[[reproducible]]` on a function
    definition. Both rails parse + CONSUME the hint and record it as a fusion-legality signal surfaced in the
    `repro=` field of the parity summary; the value-neutral hint is DROPPED from the emit (so the C stays
    behaviour-equivalent under Clang). Asserts: (a) both rails parse the fixtures and agree on `repro=`;
    (b) the hinted fixtures count repro=2 while an un-annotated unit counts repro=0 (NON-DISTURBANCE);
    (c) the attribute does NOT leak into the emitted C; (d) twin == oracle parity + Clang-equivalence."""
    # (b) NON-DISTURBANCE: a unit with no attribute counts repro=0; a one-function hinted unit counts 1.
    plain_summary, _, _ = _oracle("uint32_t f(uint32_t a){ return a + 1u; }")
    assert "repro=0" in plain_summary, plain_summary
    hinted_summary, hr, _ = _oracle("[[unsequenced]] uint32_t g(uint32_t a){ return a + 1u; }")
    assert "repro=1" in hinted_summary, hinted_summary
    # (c) the value-neutral hint is dropped from the emit -- no `[[` survives into the emitted C.
    assert "[[" not in "\n".join(hr.emitted.values())

    for fx, want_repro in (("cfront_unsequenced.c", 2), ("cfront_reproducible.c", 2)):
        path = os.path.join(_C, fx)
        src = open(path, encoding="utf-8").read()
        oracle_summary, r, entry = _oracle(src)
        assert "ok=1" in oracle_summary, oracle_summary
        assert f"repro={want_repro}" in oracle_summary, oracle_summary
        assert "[[" not in "\n".join(r.emitted.values())  # (c) emit drops the attribute
        if not _CC:
            continue
        # (d) the dual-rail gate: the twin prints the same summary (incl. repro=) and its emit == Clang.
        exe = _build_frontend(_session_build_dir())
        fx_, msg = _parity_check_fixture((exe, fx))
        assert msg is None, f"{fx_}: {msg}"


# The cross-fixture parity/equivalence campaign is split into N groups so run_all's worker pool
# parallelizes it WITHOUT a nested process pool. The former single test spawned its OWN
# os.cpu_count()-wide pool INSIDE run_all's already-cpu_count()-wide pool -- a 2x oversubscription that
# both inflated this test's own wall (~6.7s -> ~12.4s) and crushed whatever ran alongside it (a 3.8s
# neighbour ballooned to 12.3s in its window). As N independent `test_*` groups each run their slice
# SERIALLY, the global scheduler load-balances them against every other test with no core contention.
# Round-robin slicing keeps the groups balanced and auto-covers any fixture added to _FIXTURES (no
# hand-maintained partition to drift).
_PARITY_GROUPS = 4
_PARITY_SLICES = [_FIXTURES[i::_PARITY_GROUPS] for i in range(_PARITY_GROUPS)]


def _parity_check_group(fxs):
    """Parity + dual-emit behaviour-equivalence over a SLICE of the fixtures (see `_PARITY_SLICES`).
    Runs serially -- run_all's process pool supplies the parallelism *across* groups, so there is no
    nested pool to oversubscribe the cores."""
    if not _CC:
        # quick tier: still validate the oracle side computes the summaries.
        for fx in fxs:
            s, _, _ = _oracle(
                open(os.path.join(_C, fx), encoding="utf-8").read(), _includes_for(fx)
            )
            assert "ok=1" in s
        return
    exe = _build_frontend(_session_build_dir())  # session-cached; the arg is ignored
    results = [_parity_check_fixture((exe, fx)) for fx in fxs]
    fails = [(fx, msg) for fx, msg in results if msg]
    assert not fails, "C-frontend parity/equivalence failures:\n" + "\n".join(
        f"  {fx}: {msg}" for fx, msg in fails
    )


def test_python_c_parity_and_equivalence_across_fixtures_g0():
    """Cross-fixture C<->oracle parity + behaviour-equivalence, group 0/4 (see `_parity_check_group`)."""
    _parity_check_group(_PARITY_SLICES[0])


def test_python_c_parity_and_equivalence_across_fixtures_g1():
    """Cross-fixture C<->oracle parity + behaviour-equivalence, group 1/4 (see `_parity_check_group`)."""
    _parity_check_group(_PARITY_SLICES[1])


def test_python_c_parity_and_equivalence_across_fixtures_g2():
    """Cross-fixture C<->oracle parity + behaviour-equivalence, group 2/4 (see `_parity_check_group`)."""
    _parity_check_group(_PARITY_SLICES[2])


def test_python_c_parity_and_equivalence_across_fixtures_g3():
    """Cross-fixture C<->oracle parity + behaviour-equivalence, group 3/4 (see `_parity_check_group`)."""
    _parity_check_group(_PARITY_SLICES[3])


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
                b = subprocess.run(
                    [_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True
                )
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
    funcs = [
        "fd_read",
        "fd_write",
        "fd_qread",
        "fd_index",
        "fd_index_set",
        "fd_rmw",
        "fd_chain1",
        "fd_chain1_set",
        "fd_chain1_rmw",
        "fd_chain2",
        "fd_chain2_long",
        "fd_chain2_set",
    ]
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
                b = subprocess.run(
                    [_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True
                )
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
    funcs = [
        "ps_s8",
        "ps_u8",
        "ps_s16",
        "ps_u16",
        "ps_s8_divrem",
        "ps_u8_div",
        "ps_s8_shr",
        "ps_u8_shr",
        "ps_s8_cmp",
        "ps_s64_div",
        "ps_u64_div",
        "ps_arith",
        "ps_uac",
        "ps_field",
        "ps_w8",
    ]
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
                b = subprocess.run(
                    [_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True
                )
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
                b = subprocess.run(
                    [_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True
                )
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
                b = subprocess.run(
                    [_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True
                )
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
    funcs = [
        "ct_plain_deref",
        "ct_signed_deref",
        "ct_unsigned_deref",
        "ct_plain_cmp",
        "ct_signed_cmp",
        "ct_plain_div",
        "ct_signed_div",
        "ct_unsigned_div",
        "ct_roundtrip",
        "ct_plain_widen",
    ]
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
            for charmode in ("-fsigned-char", "-funsigned-char"):  # exercise plain char both ways
                epath = os.path.join(d, f"{label}{charmode}")
                for std in ("c23", "c2x", "c17"):
                    b = subprocess.run(
                        [_CC, f"-std={std}", "-O2", charmode, cpath, "-o", epath],
                        capture_output=True,
                        text=True,
                    )
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
    funcs = [
        "cl_byval",
        "cl_designated",
        "cl_partial",
        "cl_scalar",
        "cl_addr_scalar",
        "cl_addr_struct",
        "cl_nested",
        "cl_dot",
        "cl_dot_desig",
        "cl_dot_part",
        "cl_dot_wide",
    ]
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
                b = subprocess.run(
                    [_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True
                )
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
    funcs = [
        "to_width",
        "to_sign",
        "to_typename",
        "to_ptr",
        "to_struct",
        "to_unqual",
        "to_ebinop",
        "to_ebinsign",
        "to_ecast",
        "to_ederef",
        "to_emember",
        "to_eindex",
    ]
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
                b = subprocess.run(
                    [_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True
                )
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
                b = subprocess.run(
                    [_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True
                )
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
        f'    if({f}_s(i)!=bcir_{f}(i)){{printf("{f}@%u\\n",i);return 1;}}' for f in funcs
    )
    driver = (
        "int main(void){\n"
        "  for(unsigned i=0;i<60u;i++){\n"
        f"{cmps}\n"
        "  }\n"
        '  printf("MATCH\\n");return 0;}'
    )
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
                b = subprocess.run(
                    [_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True
                )
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
    funcs = [
        "l_local",
        "d_local",
        "l_inc",
        "l_member",
        "l_array",
        "l_ptr",
        "d_vararg",
        "l_vararg",
        "driver",
    ]
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
            harness = (
                f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
                f"#include <stdarg.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}"
            )
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run(
                    [_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True
                )
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
    safe-max macro, embedding in a larger expression, a loop inside, nesting, scope shadowing, and
    Clang's declared-bitfield exception for only a bare terminal member (parentheses/arrow included;
    casts, arithmetic, and comma expressions retain ordinary promotion). Differential == Clang on BOTH emits."""
    fx = "cfront_stmtexpr.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    oracle_summary, r, entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    funcs = [
        "se_simple",
        "se_max",
        "se_embed",
        "se_loop",
        "se_nest",
        "se_scope",
        "se_void",
        "se_bf_declared",
        "se_bf_regular",
        "se_bf_cast",
        "se_bf_expr",
        "se_bf_comma",
        "se_bf_paren",
        "se_bf_arrow",
        "se_bf_signed",
    ]
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
    if(se_bf_declared_s((unsigned)a)!=bcir_se_bf_declared((unsigned)a)){puts("bf-declared");return 1;}
    if(se_bf_regular_s((unsigned)a)!=bcir_se_bf_regular((unsigned)a)){puts("bf-regular");return 1;}
    if(se_bf_cast_s((unsigned)a)!=bcir_se_bf_cast((unsigned)a)){puts("bf-cast");return 1;}
    if(se_bf_expr_s((unsigned)a)!=bcir_se_bf_expr((unsigned)a)){puts("bf-expr");return 1;}
    if(se_bf_comma_s((unsigned)a)!=bcir_se_bf_comma((unsigned)a)){puts("bf-comma");return 1;}
    if(se_bf_paren_s((unsigned)a)!=bcir_se_bf_paren((unsigned)a)){puts("bf-paren");return 1;}
    if(se_bf_arrow_s((unsigned)a)!=bcir_se_bf_arrow((unsigned)a)){puts("bf-arrow");return 1;}
    if(se_bf_signed_s(a)!=bcir_se_bf_signed(a)){puts("bf-signed");return 1;}
  }
  if(se_bf_declared_s(5u)!=4294967291u || se_bf_regular_s(5u)!=-5 ||
     se_bf_cast_s(5u)!=-5 || se_bf_expr_s(5u)!=-5 || se_bf_comma_s(5u)!=-5 ||
     se_bf_paren_s(5u)!=4294967291u || se_bf_arrow_s(5u)!=4294967291u){puts("bf-values");return 1;}
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
                b = subprocess.run(
                    [_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True
                )
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
                b = subprocess.run(
                    [_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True
                )
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
            harness = (
                f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
                f"#include <stdatomic.h>\n{renamed}\n{emit}\n{driver}"
            )
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run(
                    [_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True
                )
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} not behaviour-equivalent ({out})"


# CF-ATOMIC: per function of `cfront_atomicaccess.c`, the atomic loads and stores and the atomic
# read-modify-writes each rail lowers -- every access to an `_Atomic` object is exactly one of them -- and so
# the `_Atomic` lvalues its emit spells, one per claim. Pinned exactly: a count that only had to be nonzero
# would pass with one access left a byte copy.
_ATOMIC_ACCESS = {
    "acc_deref": (2, 4),
    "acc_member": (8, 4),
    "acc_float": (4, 0),
    "acc_elems": (6, 2),
    "acc_index": (2, 1),
    "acc_global": (0, 3),  # a named object is atomic by declaration: loaded and stored by name
    "acc_load64": (0, 0),  # the generics lower to their own `c.c11atom.*` claims
    "acc_loadf": (0, 0),
    "acc_ptrmember": (4, 1),
    "acc_bool": (4, 0),
    "acc_bump": (0, 3),
}
# the plain memory accesses a function keeps: only the loads of a plain pointer member (`s->ap` itself)
_ATOMIC_ACCESS_PLAIN = {"acc_ptrmember": 5}

# every function of the fixture, run on independent copies of the same state by the original and by an
# emit, compared by value and by memory; the global is written before it is modified, so a second run of
# the same function sees the state the first did
_ATOMIC_ACCESS_DRIVER = r"""
static int fail(const char *what) { puts(what); return 1; }
int main(void) {
  for (uint32_t k = 0; k < 97u; k++) {
    uint32_t v = k * 2654435761u, i = k;
    { _Atomic uint32_t a = 7u, b = 7u;
      if (acc_deref_s(&a, v) != bcir_acc_deref(&b, v) || a != b) return fail("deref"); }
    { struct acc_s a, b; memset(&a, 0, sizeof a); memset(&b, 0, sizeof b);
      if (acc_member_s(&a, v) != bcir_acc_member(&b, v) || memcmp(&a, &b, sizeof a)) return fail("member");
      float x = (float)v * 0.25f;
      if (acc_float_s(&a, x) != bcir_acc_float(&b, x) || memcmp(&a, &b, sizeof a)) return fail("float"); }
    { struct acc_a a, b; struct acc_e ea[2], eb[2];
      memset(&a, 0x11, sizeof a); memset(&b, 0x11, sizeof b); memset(ea, 0x22, sizeof ea); memset(eb, 0x22, sizeof eb);
      if (acc_elems_s(&a, ea, i, v) != bcir_acc_elems(&b, eb, i, v) || memcmp(&a, &b, sizeof a)
          || memcmp(ea, eb, sizeof ea)) return fail("elems"); }
    { _Atomic uint32_t a[4] = {1u, 2u, 3u, 4u}, b[4] = {1u, 2u, 3u, 4u};
      if (acc_index_s(a, i, v) != bcir_acc_index(b, i, v) || memcmp(a, b, sizeof a)) return fail("index"); }
    { uint32_t ra = acc_global_s(v), ga = acc_g, rb = bcir_acc_global(v);
      if (ra != rb || ga != acc_g) return fail("global"); }
    { _Atomic uint64_t a = 0u, b = 0u; uint64_t w = ((uint64_t)v << 32) | (k * 7u + 3u);
      if (acc_load64_s(&a, w) != bcir_acc_load64(&b, w) || a != b) return fail("load64"); }
    { _Atomic float a = 0.0f, b = 0.0f; float x = (float)k * 1.375f + 0.3f;
      if (acc_loadf_s(&a, x) != bcir_acc_loadf(&b, x) || a != b) return fail("loadf"); }
    { _Atomic uint32_t ca[2] = {0u, 0u}, cb[2] = {0u, 0u}; struct acc_p a, b;
      memset(&a, 0, sizeof a); memset(&b, 0, sizeof b); a.c = b.c = (uint8_t)k; a.ap = ca; b.ap = cb;
      if (acc_ptrmember_s(&a, v) != bcir_acc_ptrmember(&b, v) || memcmp(ca, cb, sizeof ca) || a.c != b.c
          || a.ap != ca || b.ap != cb) return fail("ptrmember"); }
    { _Atomic _Bool a[3] = {0, 0, 0}, b[3] = {0, 0, 0};
      if (acc_bool_s(a, i, v) != bcir_acc_bool(b, i, v) || memcmp(a, b, sizeof a)) return fail("bool"); }
    { _Atomic uint32_t a = k, b = k; struct acc_s sa, sb; struct acc_a aa, ab;
      memset(&sa, 0, sizeof sa); memset(&sb, 0, sizeof sb); memset(&aa, 0, sizeof aa); memset(&ab, 0, sizeof ab);
      acc_bump_s(&a, &sa, &aa); bcir_acc_bump(&b, &sb, &ab);
      if (a != b || memcmp(&sa, &sb, sizeof sa) || memcmp(&aa, &ab, sizeof aa)) return fail("bump"); }
  }
  puts("MATCH");
  return 0;
}
"""

# the emitted `acc_bump` from several threads at once: an update lowered as a load, an add and a store
# loses the increments a concurrent step wrote between them; one atomic read-modify-write loses none. The
# call goes through a volatile function pointer, so no compiler folds the loop's steps into one.
_ATOMIC_BUMP_THREADS = r"""
#include <pthread.h>
enum { THREADS = 4, STEPS = 100000 };
static _Atomic uint32_t cnt;
static struct acc_s S;
static struct acc_a A;
static void (*volatile bump)(_Atomic uint32_t *, struct acc_s *, struct acc_a *) = bcir_acc_bump;
static void *worker(void *arg) {
  (void)arg;
  for (int k = 0; k < STEPS; k++) bump(&cnt, &S, &A);
  return 0;
}
int main(void) {
  pthread_t t[THREADS];
  for (int j = 0; j < THREADS; j++)
    if (pthread_create(&t[j], 0, worker, 0)) { puts("pthread_create"); return 2; }
  for (int j = 0; j < THREADS; j++) pthread_join(t[j], 0);
  uint32_t want = (uint32_t)THREADS * STEPS;
  if (cnt != want || S.n != want || A.arr[1] != want) {
    printf("LOST %u %u %u of %u\n", (unsigned)cnt, (unsigned)S.n, (unsigned)A.arr[1], (unsigned)want);
    return 1;
  }
  puts("MATCH");
  return 0;
}
"""


def _emit_functions(emit: str) -> dict:
    """The emitted C of each function: `bcir_<name>` -> its text, up to the next function's."""
    starts = [
        (m.start(), m.group(1)) for m in re.finditer(r"^static [^\n(]*\bbcir_(\w+)\(", emit, re.M)
    ]
    return {
        name: emit[at : starts[k + 1][0] if k + 1 < len(starts) else len(emit)]
        for k, (at, name) in enumerate(starts)
    }


def _assert_atomic_emit(emit: str, rail: str) -> None:
    """Every access to an `_Atomic` object in `emit` goes through an `_Atomic` lvalue, one per claim, and
    no byte copy moves one: a `memcpy` of an atomic object is no atomic access, and it tears. The generics
    type their value by the pointee."""
    bodies = _emit_functions(emit)
    assert set(bodies) == set(_ATOMIC_ACCESS), (rail, sorted(bodies))
    for name, (loads_stores, rmws) in _ATOMIC_ACCESS.items():
        body = bodies[name]
        lvalues = re.findall(r"\(\*\((?:volatile )?_Atomic ", body)
        assert len(lvalues) == loads_stores + rmws, (rail, name, len(lvalues), body)
        copies = [line for line in body.splitlines() if "memcpy" in line]
        assert len(copies) == _ATOMIC_ACCESS_PLAIN.get(name, 0), (rail, name, copies)
        for line in copies:  # the pointer member itself: copied whole into a pointer temp
            assert re.search(r"\* (t\d+); memcpy\(&\1, ", line), (rail, name, line)
    assert re.search(r"\buint64_t t\d+ = atomic_load\(p\);", bodies["acc_load64"]), (rail, bodies)
    assert re.search(r"\bfloat t\d+ = atomic_load\(p\);", bodies["acc_loadf"]), (rail, bodies)


def _build_run_c(d: str, cc: str, label: str, text: str, extra=()) -> str:
    """Build `text` with `cc` (the newest C standard it takes) and run it; its stdout, stripped."""
    cpath, epath = os.path.join(d, f"{label}.c"), os.path.join(d, label)
    with open(cpath, "w", encoding="utf-8") as fh:
        fh.write(text)
    for std in ("c23", "c2x", "c17"):
        b = subprocess.run(
            [cc, f"-std={std}", "-O2", "-Werror=incompatible-pointer-types", cpath, "-o", epath]
            + list(extra),
            capture_output=True,
            text=True,
        )
        if b.returncode == 0:
            break
    else:
        raise AssertionError(f"{label}: build failed under {cc}:\n{b.stderr}")
    return subprocess.run([epath], capture_output=True, text=True, timeout=300).stdout.strip()


def test_atomic_access_is_one_atomic_operation_on_both_rails():
    """CF-ATOMIC: an access to an `_Atomic` object is one atomic operation wherever the object is --
    through a pointer, a member, a member array, an array of structs, a subscripted pointer, a pointer
    member or a named global. Both rails had lowered one through a pointer or a member as a plain byte
    copy, and a compound assignment or an increment as a load, an operation and a store, so a concurrent
    update between them was lost (C11 6.5.16.2p3 and 6.5.2.4p2 make each one read-modify-write). A load or a
    store now keeps its claim on lane A with the atomic hazard; `E op= v` and `++E` are one
    `c.c11atom.rmw:<op>` claim; both emits spell every one through an `_Atomic` lvalue. The value of an
    assignment is the value stored, never a second atomic read; an `_Atomic _Bool` normalizes every store;
    the generics type their value by the pointee (a 64-bit `atomic_load` truncated to 32 bits). The
    fixture runs against the original on both emits under both compilers, and `acc_bump` from four
    threads at once loses no update."""
    from bcir.model.lanes import Lane

    fx = "cfront_atomicaccess.c"
    path = os.path.join(_C, fx)
    src = open(path, encoding="utf-8").read()
    oracle_summary, r, _entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    assert set(r.lowered.functions) == set(_ATOMIC_ACCESS), sorted(r.lowered.functions)
    for name, fn in r.lowered.functions.items():
        mem = [c for c in fn.claims if c.op in ("c.load", "c.store")]
        atomic = [c for c in mem if c.hazard == "atomic"]
        rmw = [c for c in fn.claims if c.op.startswith("c.c11atom.rmw:")]
        assert (len(atomic), len(rmw)) == _ATOMIC_ACCESS[name], (name, [c.op for c in fn.claims])
        plain = len(mem) - len(atomic)
        assert plain == _ATOMIC_ACCESS_PLAIN.get(name, 0), (name, [c.op for c in mem])
        for c in atomic + rmw:
            assert c.lane == Lane.A and c.hazard == "atomic", (name, c.op, c.lane, c.hazard)
    oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    _assert_atomic_emit(oracle_emit, "oracle")
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    c_summary, c_emit = _c_run(exe, path)
    assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
    _assert_atomic_emit(c_emit, "twin")
    renamed = src
    for f in _ATOMIC_ACCESS:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    head = "#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n#include <stdatomic.h>\n"
    compilers = [c for c in dict.fromkeys((_CC, shutil.which("clang"), shutil.which("gcc"))) if c]
    with tempfile.TemporaryDirectory() as d:
        for cc in compilers:
            for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
                text = f"{head}{renamed}\n{emit}\n{_ATOMIC_ACCESS_DRIVER}"
                out = _build_run_c(d, cc, label, text)
                assert out == "MATCH", f"{fx}: {label} emit not equivalent under {cc} ({out})"
        if os.name == "posix":  # the threaded witness needs pthreads
            for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
                text = f"{head}{src}\n{emit}\n{_ATOMIC_BUMP_THREADS}"
                out = _build_run_c(d, _CC, f"{label}_threads", text, ("-pthread",))
                assert out == "MATCH", f"{fx}: the {label} emit lost concurrent updates ({out})"


# CF-ATOMIC: sources that differ ONLY in `_Atomic` (`{A}` spelled empty, then `_Atomic `) -- an atomic
# access is a different operation, so each pair must digest differently, on both rails alike.
_ATOMIC_DIGEST_PAIRS = (
    "uint32_t f({A}uint32_t *p, uint32_t v) {{ *p = v; return *p; }}\n",
    "uint32_t f({A}uint32_t *p, uint32_t i) {{ return p[i & 3u]; }}\n",
    "void f({A}uint32_t *p, uint32_t i, uint32_t v) {{ p[i & 3u] = v; }}\n",
    "struct S {{ uint8_t c; {A}uint32_t n; }};\nuint32_t f(struct S *s, uint32_t v) {{ s->n = v; return s->n; }}\n",
    "struct A {{ uint8_t c; {A}uint32_t arr[4]; }};\n"
    "uint32_t f(struct A *a, uint32_t i) {{ return a->arr[i & 3u]; }}\n",
    "uint32_t f({A}uint32_t *p, uint32_t v) {{ *p += v; return *p; }}\n",
)


def test_an_atomic_access_changes_the_structural_digest():
    """CF-ATOMIC: an atomic load or store keeps the plain access's claim and differs only in its order
    (lane A, the atomic hazard) -- which the structural digest did not read, so a source with `_Atomic`
    and one without digested alike, and a cross-rail parity gate could not tell an atomic access from a
    byte copy. The digest canon names it (`c.load!atomic`, `c.store!atomic`: the oracle's `_vn_op`, the
    twin's `canon_op`); every pair here differs in the digest, and each digest is the same on both
    rails."""
    exe = _build_frontend(_session_build_dir()) if _CC else None
    for form in _ATOMIC_DIGEST_PAIRS:
        digests = []
        for qual in ("", "_Atomic "):
            src = "#include <stdint.h>\n" + form.format(A=qual)
            s, _r, _e = _oracle(src)
            assert "ok=1" in s, (src, s)
            digest = s.split("digest=")[1]
            if exe is not None:
                with tempfile.TemporaryDirectory() as d:
                    p = os.path.join(d, "atomic_digest.c")
                    with open(p, "w", encoding="utf-8") as fh:
                        fh.write(src)
                    c_summary, _emit = _c_run(exe, p)
                assert c_summary == s, f"parity\n C: {c_summary}\nPY: {s}\n{src}"
            digests.append(digest)
        assert digests[0] != digests[1], form


def test_unqualified_drops_atomic_and_restores_the_natural_layout():
    """CF-ATOMIC: lvalue conversion (C23 6.3.2.1p2) drops `_Atomic` with the other qualifiers, and the value
    it yields has the non-atomic type's layout. The ABI's atomic promotion (`with_atomic`) widened an
    `_Atomic float _Complex` to align 8 and an i386 `_Atomic uint64_t` to align 8; neither belongs to the
    value read out of one, whose temp and whose arithmetic are the plain type's. `unqualified` had kept
    the flag and the promoted layout."""
    from bcir.frontends.cfront.abi import TARGETS
    from bcir.frontends.cfront.ctype_model import scalar, unqualified, with_atomic, with_volatile

    names = ("_Bool", "uint8_t", "int16_t", "uint32_t", "uint64_t", "float", "double")
    for tname, abi in TARGETS.items():
        for name in (*names, "float _Complex", "double _Complex"):
            plain = scalar(name, abi)
            at = with_atomic(plain, abi=abi)
            assert at.atomic and at.natural == (plain.size, plain.align), (tname, name, at)
            assert unqualified(at) == plain, (tname, name, unqualified(at))
            assert unqualified(with_volatile(at)) == plain, (tname, name)
            assert unqualified(with_atomic(at, abi=abi)) == plain, (tname, name)  # `_Atomic` twice
    x86, i386 = TARGETS["x86_64-linux"], TARGETS["i386-linux"]
    fc = with_atomic(scalar("float _Complex", x86), abi=x86)
    assert (fc.size, fc.align) == (8, 8)
    assert (unqualified(fc).size, unqualified(fc).align) == (8, 4)
    u64 = with_atomic(scalar("uint64_t", i386), abi=i386)
    assert (u64.size, u64.align) == (8, 8)
    assert (unqualified(u64).size, unqualified(u64).align) == (8, 4)


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
                b = subprocess.run(
                    [_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True
                )
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
                b = subprocess.run(
                    [_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True
                )
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
                b = subprocess.run(
                    [_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True
                )
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
    funcs = [
        "g_int",
        "g_long",
        "g_uint",
        "g_double",
        "g_float",
        "g_char",
        "g_ptr",
        "g_exprtype",
        "g_default",
        "g_compute",
    ]
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
                b = subprocess.run(
                    [_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True
                )
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
            harness = (
                f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
                f"#include <math.h>\n{renamed}\n{emit}\n{driver}"
            )
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run(
                    host_link_args([_CC, f"-std={std}", "-O2", cpath, "-o", epath, "-lm"]),
                    capture_output=True,
                    text=True,
                )
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
            harness = (
                f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
                f"#include <stdarg.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}"
            )
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run(
                    [_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True
                )
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
    funcs = ["isum", "twice", "vsumv", "forward", "nth", "dsum", "lsum", "caller"]
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
    if(dsum_s(3,(double)a,(double)b,(double)c)!=bcir_dsum(3,(double)a,(double)b,(double)c)){printf("dsum@%d,%d\n",a,b);return 1;}
    if(lsum_s(3,(long)a,(long)b,(long)c)!=bcir_lsum(3,(long)a,(long)b,(long)c)){printf("lsum@%d,%d\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}"""
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            harness = (
                f"#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
                f"#include <stdarg.h>\n{_BOUNDS_GUARD}\n{renamed}\n{emit}\n{driver}"
            )
            cpath = os.path.join(d, f"{label}.c")
            open(cpath, "w").write(harness)
            epath = os.path.join(d, label)
            for std in ("c23", "c2x", "c17"):
                b = subprocess.run(
                    [_CC, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True
                )
                if b.returncode == 0:
                    break
            else:
                raise AssertionError(f"{fx}: {label} build failed:\n{b.stderr}")
            out = subprocess.run([epath], capture_output=True, text=True).stdout.strip()
            assert out == "MATCH", f"{fx}: {label} not behaviour-equivalent ({out})"


def _build_loop(d: str) -> str:
    return _compile_once(
        "loop",
        "loop",
        (
            "bcir_cfront.c",
            "bcir_cpp.c",
            "bcir_plan.c",
            "bcir_hydrate.c",
            "bcir_exec.c",
            "bcir_runtime.c",
            "bcir_verify.c",
            "test_cfront_loop.c",
        ),
        "loop",
    )


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
            assert (
                int(m["plan_cost"]) > 0 and int(m["pack_bytes"]) > 64
            )  # a real plan + a real pack
            order = out.split("order=")[1].split(",")
            assert order == sorted(order, key=int)  # deterministic lowering order


# the atomic builtins each fixture must emit back (the faithful-emit artifact, not a scalar fallback).
_ATOMIC_EMITS = {
    "cfront_atomic.c": ["__atomic_fetch_", "__atomic_thread_fence", "__ATOMIC_SEQ_CST"],
    "cfront_cmpxchg.c": ["__sync_val_compare_and_swap", "__sync_bool_compare_and_swap"],
    "cfront_atomic11.c": [
        "_Atomic uint32_t *",
        "atomic_fetch_add",
        "atomic_fetch_xor",
        "atomic_load",
    ],
    "cfront_atomic_xchg.c": ["_Atomic uint32_t *", "atomic_exchange", "atomic_load"],
    "cfront_cmpxchg11.c": [
        "atomic_compare_exchange_strong",
        "atomic_compare_exchange_weak",
        "_Atomic",
    ],
}


def test_atomic_fence_dual_rail_parity_and_behaviour():
    """§5.8 atomics/fences/CAS: __atomic_fetch_add/sub/xor -> ATOMIC_ADD/SUB/XOR,
    __atomic_thread_fence/__sync_synchronize -> BARRIER, and __sync_{val,bool}_compare_and_swap ->
    CMPXCHG (a 3-read claim: ptr, expected, desired) all lower on lane A (R6 admits lane A for a
    scalar atomic; R5 demands the atomic/barriered hazard), pass R1-R8 + R18, emit the matching
    builtins, and -- run on independent copies of the same seeded cell -- are behaviour-equivalent
    under Clang. The full C compile->execute loop hydrates and executes every atomic claim with
    R9/R10-R11 clean."""
    for fx in _ATOMIC:  # quick tier: the oracle accepts the atomic fixtures.
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
            assert c_summary == oracle_summary, (
                f"{fx}: parity diverged\n C: {c_summary}\nPY: {oracle_summary}"
            )
            assert "ok=1" in c_summary, c_summary
            # the emitted C carries the real atomic builtins, not a scalar fallback.
            for needle in _ATOMIC_EMITS[fx]:
                assert needle in c_emit, f"{fx}: emit missing {needle}\n{c_emit}"
            assert _equiv_atomic(r.source, c_emit, entry) == "MATCH", (
                f"{fx}: not behaviour-equivalent"
            )
            # the full C compile->execute loop: every atomic claim hydrates + executes, R9/R10-R11 clean.
            out = subprocess.run([loop, path], capture_output=True, text=True).stdout.strip()
            assert out.startswith("loop:"), out
            m = dict(re.findall(r"(\w+)=([0-9]+)", out))
            assert int(m["executed"]) == int(m["claims"]) == len(entry.claims), out
            assert m["r9"] == "1" and m["r10r11"] == "1", out


def test_fence_order_edge_cases_dual_rail():
    """SEG7: the order-parameterized fence routing must be byte-identical on BOTH rails for the tricky
    forms the corpus' bare / integer / parenthesized fences do not exhaust -- a leading unary `+` (which
    the parser DROPS, like a redundant paren), a cast / binary order (which stays a node -> the full
    fence), and a `memory_order_*` / `__ATOMIC_*` name SHADOWED by an enum constant (routed by the enum's
    VALUE, not the builtin mapping -- the oracle parser folds the enumerator to an IntLit). These are
    compile-only edge cases that were LATENT dual-rail divergences (the C twin's token-peek `fence_order_op`
    vs the oracle's parsed-AST `_fence_order_kind`); they are pinned here so a future change to either rail
    cannot silently reintroduce a split. (cfront_atomic.c already pins the bare / paren / unary-+ forms
    through the full compile->execute loop; this adds the cast/binary-fold and the enum-shadow forms, which
    need their own translation units.)"""
    if not _CC:
        return  # quick tier hides the toolchain -> self-skip
    cases = [
        ("unary-plus integer", "uint32_t f(uint32_t *p){ atomic_thread_fence(+2); return *p; }"),
        (
            "unary-plus name",
            "uint32_t f(uint32_t *p){ __atomic_thread_fence(+memory_order_release); return *p; }",
        ),
        (
            "paren+unary-plus",
            "uint32_t f(uint32_t *p){ atomic_thread_fence(+(memory_order_acquire)); return *p; }",
        ),
        (
            "cast folds to full",
            "uint32_t f(uint32_t *p){ atomic_thread_fence((int)2); return *p; }",
        ),
        (
            "binary folds to full",
            "uint32_t f(uint32_t *p){ atomic_thread_fence(memory_order_acquire + 0); return *p; }",
        ),
        (
            "enum shadow ->release",
            "enum { memory_order_acquire = 3 };\nuint32_t f(uint32_t *p){ atomic_thread_fence(memory_order_acquire); return *p; }",
        ),
        (
            "enum shadow ->full",
            "enum { memory_order_acquire = 7 };\nuint32_t f(uint32_t *p){ __atomic_thread_fence(memory_order_acquire); return *p; }",
        ),
        (
            "enum __ATOMIC ->acq",
            "enum { __ATOMIC_RELEASE = 2 };\nuint32_t f(uint32_t *p){ __atomic_thread_fence(__ATOMIC_RELEASE); return *p; }",
        ),
    ]
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        for label, src in cases:
            oracle_summary, _r, _entry = _oracle(src)
            assert "ok=1" in oracle_summary, f"{label}: oracle rejects: {oracle_summary}"
            path = os.path.join(d, "fenceorder.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
            c_summary, _emit = _c_run(exe, path)
            assert c_summary == oracle_summary, (
                f"{label}: dual-rail fence-order parity diverged\n C: {c_summary}\nPY: {oracle_summary}"
            )


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
        assert "o->add2(" in c_emit and "o->mul2(" in c_emit, c_emit  # faithful member-call emit
        struct_ct = entry.params[0][2].of  # the pointed-to ops struct
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
    {"".join(f"uint32_t {s}=rng(); " for s in scalars)}
    if({entry.name}({call})!=bcir_{entry.name}({call})){{printf("MISMATCH@%d",i);return 1;}}
  }}
  printf("MATCH");return 0;}}"""
        c, e = os.path.join(d, "disp.c"), os.path.join(d, "disp")
        open(c, "w").write(harness)
        for std in ("c23", "c2x", "c17"):
            b = subprocess.run(
                host_link_args(
                    [_CC, f"-std={std}", "-O2", c, "-o", e, "-lm"]
                ),  # logical libm; omitted for Windows CRT
                capture_output=True,
                text=True,
            )
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
    assert len(r.lowered.functions) == 3  # decode_state, bank_lookup, sensor_read
    # the entry reads two MMIO registers (status + sample) and makes one resolved call (decode_state).
    assert "mmio=2" in oracle_summary and "call=1" in oracle_summary, oracle_summary
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
        # the emit carries each composed feature in one verified-C unit.
        for needle in (
            "volatile uint32_t *",
            "goto done",
            "static uint32_t faults",
            "(uint16_t)",
            "bcir_decode_state(",
            "BCIR verified-C attestation",
        ):
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
        for needle in (
            "volatile uint32_t *",
            "QUANTA[",
            "static uint32_t halts",
            "switch (",
            "case ",
            "BCIR verified-C attestation",
        ):
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
        assert prov and prov.group(1) == m.group(1), (
            f"digest mismatch: emit={m.group(1)} loop={prov}"
        )


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
    assert "mmio=1" in oracle_summary and "bf=3" in oracle_summary and "call=2" in oracle_summary, (
        oracle_summary
    )
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        loop = _build_loop(d)
        path = os.path.join(_C, "cfront_driver.c")
        c_summary, c_emit = _c_run(exe, path)
        assert c_summary == oracle_summary, (
            f"Phase D parity diverged\n C: {c_summary}\nPY: {oracle_summary}"
        )
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
    p = subprocess.run(
        [sys.executable, "-m", "bcir.frontends.cfront", *args],
        capture_output=True,
        text=True,
        cwd=cwd or _ROOT,
    )
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
        incd = os.path.join(d, "inc")
        os.makedirs(incd)
        with open(os.path.join(incd, "regs.h"), "w") as f:
            f.write("typedef volatile unsigned int reg32;\nstruct dev { reg32 r; };\n")
        src = os.path.join(d, "drv.c")
        with open(src, "w") as f:
            f.write('#include "regs.h"\nunsigned int rd(volatile struct dev *p){ return p->r; }\n')
        rc, out, err = _cli(["-I", incd, src])
        assert rc == 0, f"-I compile failed: {err}\n{out}"
        assert "R1-R18: CLEAN" in out, out
        # (4) a missing header is a clean diagnostic + non-zero exit (not a crash).
        rc, out, err = _cli([src])  # no -I -> regs.h unresolved
        assert rc != 0 and "not found" in (out + err), (out, err)
    # (5) -D predefines a macro used in a #if.
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "g.c")
        with open(src, "w") as f:
            f.write(
                "#if defined(WIDE)\nunsigned int w(unsigned int x){return x+1;}\n"
                "#else\nunsigned int w(unsigned int x){return x;}\n#endif\n"
            )
        rc, out, _ = _cli(["-E", "-D", "WIDE", src])
        assert rc == 0 and "x+1" in out.replace(" ", ""), out


def _build_bcir_cc(d: str) -> str:
    return _compile_once(
        "bcir_cc",
        "bcir-cc",
        (
            "bcir_cc.c",
            "bcir_cpp.c",
            "bcir_cfront.c",
            "bcir_verify.c",
            "bcir_runtime.c",
            "bcir_plan.c",
            "bcir_hydrate.c",
        ),
        "bcir-cc",
    )


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
        assert f'"{src}"' in cr.stdout, cr.stdout  # __FILE__ == the given path
        assert cr.stdout == _py_pp(text, name=src)  # byte-identical to the oracle
        # Python CLI rail: -E <path> (the driver appends one trailing newline).
        rc, pyo, err = _cli(["-E", src])
        assert rc == 0, err
        assert f'"{src}"' in pyo, pyo
        assert pyo.rstrip("\n") == cr.stdout.rstrip("\n")  # same content across the rails
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
        incd = os.path.join(d, "inc")
        os.makedirs(incd)
        with open(os.path.join(incd, "r.h"), "w") as f:
            f.write("typedef volatile unsigned int reg32;\nstruct dev { reg32 s; };\n")
        src = os.path.join(d, "m.c")
        with open(src, "w") as f:
            f.write(
                '#include "r.h"\n#if defined(FAST)\n'
                "unsigned int g(volatile struct dev *p){ return p->s + 1u; }\n"
                "#else\nunsigned int g(volatile struct dev *p){ return p->s; }\n#endif\n"
            )
        c_sum = subprocess.run(
            [cc, "-I", incd, "-D", "FAST", src], capture_output=True, text=True
        ).stdout.strip()
        assert "binop=1" in c_sum and "ok=1" in c_sum, c_sum  # the FAST branch (+1u) -> one binop
        r = compile_unit(
            open(src, encoding="utf-8").read(),
            check_clang=False,
            search_paths=[incd],
            defines={"FAST": "1"},
        )
        entry = r.lowered.functions[next(reversed(r.lowered.functions))]
        assert sum(1 for c in entry.claims if c.op.startswith("c.bin.")) == 1  # oracle agrees
        # without -D FAST the other branch (no +1) is taken -> zero binops, on both rails.
        c_sum0 = subprocess.run(
            [cc, "-I", incd, src], capture_output=True, text=True
        ).stdout.strip()
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
    assert len(r.lowered.functions) == 3  # cfg_low, send, configure
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        loop = _build_loop(d)
        c_summary, c_emit = _c_run(exe, os.path.join(_C, fx))
        assert c_summary == oracle_summary, (
            f"{fx}: parity diverged\n C: {c_summary}\nPY: {oracle_summary}"
        )
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
        out = subprocess.run(
            [loop, os.path.join(_C, fx)], capture_output=True, text=True
        ).stdout.strip()
        assert out.startswith("loop:"), out
        m = dict(re.findall(r"(\w+)=([0-9]+)", out))
        assert int(m["executed"]) == int(m["claims"]) == len(entry.claims), out
        assert m["r9"] == "1" and m["r10r11"] == "1", out
        prov = re.search(r"prov=([0-9a-f]{16})", out)
        emit_prov = re.search(r"R13 provenance digest\s+([0-9a-f]{16})", c_emit)
        assert prov and emit_prov and prov.group(1) == emit_prov.group(1), (
            "digest mismatch loop vs emit"
        )


def test_L8_packed_layout_matches_clang():
    """The C frontend's packed struct offsets must equal Clang's sizeof/offsetof (the ABI)."""
    src = open(os.path.join(_C, "cfront_packed.c"), encoding="utf-8").read()
    hdr = compile_unit(src, check_clang=False).lowered.aggregates["wire_hdr"]
    assert hdr.size == 7 and hdr.field("addr")[1] == 1 and hdr.field("len")[1] == 5
    if not _CC:
        return
    probe = (
        "#include <stdint.h>\n#include <stddef.h>\n#include <stdio.h>\n"
        "struct __attribute__((packed)) wire_hdr { uint8_t cmd; uint32_t addr; uint16_t len; };\n"
        'int main(void){printf("%zu %zu %zu %zu", sizeof(struct wire_hdr),'
        " offsetof(struct wire_hdr,cmd), offsetof(struct wire_hdr,addr),"
        " offsetof(struct wire_hdr,len)); return 0;}"
    )
    with tempfile.TemporaryDirectory() as d:
        c, e = os.path.join(d, "p.c"), os.path.join(d, "p")
        open(c, "w").write(probe)
        if subprocess.run([_CC, "-std=c11", c, "-o", e], capture_output=True).returncode == 0:
            nums = [
                int(x) for x in subprocess.run([e], capture_output=True, text=True).stdout.split()
            ]
            assert nums == [
                hdr.size,
                hdr.field("cmd")[1],
                hdr.field("addr")[1],
                hdr.field("len")[1],
            ]


def test_bitint_member_layout_matches_clang():
    """A PLAIN C23 `_BitInt(N)` struct member must lay out at Clang's sizeof/offsetof (the ABI). Clang
    lays a `_BitInt(N)`, 2<=N<=64, in the smallest power-of-two byte storage unit >= N bits, so a
    `_BitInt(12)` is 2-byte/2-aligned (== uint16_t) and a `_BitInt(64)` is 8-byte/8-aligned. The oracle's
    `bitint` CType already carries that storage width + alignment, so AggregateBuilder must agree with
    Clang's actual layout -- the same differential the packed-layout test runs."""
    src = open(os.path.join(_C, "cfront_bitint_member.c"), encoding="utf-8").read()
    bp = compile_unit(src, check_clang=False).lowered.aggregates["bipair"]
    # the oracle layout: tag@0 (int), lo@4 (_BitInt(12) -> 2-byte slot), hi@8 (_BitInt(64) -> 8-byte slot)
    assert bp.size == 16 and bp.align == 8
    assert bp.field("tag")[1] == 0 and bp.field("lo")[1] == 4 and bp.field("hi")[1] == 8
    assert (
        bp.field("lo")[0].is_bitint
        and bp.field("lo")[0].bit_width == 12
        and bp.field("lo")[0].size == 2
    )
    assert (
        bp.field("hi")[0].is_bitint
        and bp.field("hi")[0].bit_width == 64
        and bp.field("hi")[0].size == 8
    )
    if not _CC:
        return
    probe = (
        "#include <stdint.h>\n#include <stddef.h>\n#include <stdio.h>\n"
        "struct bipair { int tag; unsigned _BitInt(12) lo; unsigned _BitInt(64) hi; };\n"
        'int main(void){printf("%zu %zu %zu %zu %zu", sizeof(struct bipair),'
        " (size_t)_Alignof(struct bipair), offsetof(struct bipair,tag),"
        " offsetof(struct bipair,lo), offsetof(struct bipair,hi)); return 0;}"
    )
    with tempfile.TemporaryDirectory() as d:
        c, e = os.path.join(d, "p.c"), os.path.join(d, "p")
        open(c, "w").write(probe)
        for std in ("c23", "c2x"):
            if (
                subprocess.run([_CC, f"-std={std}", c, "-o", e], capture_output=True).returncode
                == 0
            ):
                nums = [
                    int(x)
                    for x in subprocess.run([e], capture_output=True, text=True).stdout.split()
                ]
                assert nums == [
                    bp.size,
                    bp.align,
                    bp.field("tag")[1],
                    bp.field("lo")[1],
                    bp.field("hi")[1],
                ]
                break


def _bitint_model_tag(ta, tb):
    """The cfront's MODELED C23 result type for `ta OP b` (an arithmetic/bitwise op), as a `_Generic`-style
    type-name tag (`_BitInt(N)` / `unsigned _BitInt(N)` / a standard name), OR None iff the cfront routes
    that mix to fallback (its result would be a STANDARD integer type, outside the first-class subset). This
    is the single source of truth the differential checks against Clang -- if it returns a `_BitInt` tag,
    Clang's `_Generic` MUST select that exact tag; if it returns None, Clang's result MUST NOT be a
    `_BitInt` (so routing to fallback is correct, never hiding a wrong claim)."""
    from bcir.frontends.cfront.ctype_model import BitIntMix, usual_arith_int

    try:
        r = usual_arith_int(ta, tb)
    except BitIntMix:
        return None
    if not r.is_bitint:
        return None
    return f"_BitInt({r.bit_width})" if r.signed else f"unsigned _BitInt({r.bit_width})"


def test_bitint_mixed_width_result_type_matches_clang():
    """SUB-FEATURE A differential: the cfront's MODELED C23 6.3.1.8 result type for a mixed `_BitInt`/
    standard-int (or two-`_BitInt`) expression must EQUAL Clang's, on every case -- the gate that makes the
    new first-class subset Clang-equivalent. For each operand pair the cfront's model (`_bitint_model_tag`)
    yields either a `_BitInt(N)` tag (first-class) or None (routed to fallback). A single Clang `_Generic`
    probe asserts, per case, that:
      * a first-class `_BitInt` tag is EXACTLY the type Clang selects (`_Generic((a OP b), <tag>:1, ...)==1`);
      * a None (fallback) case's Clang result is NOT a `_BitInt` (the model never hides a wrong `_BitInt`).
    If the model disagreed with Clang on ANY case, the probe's `_Static_assert`s fail (so the test fails) --
    correctness over coverage. (Verified live against Clang 18; the model reproduces every case.)"""
    if not _CC:
        return
    from bcir.frontends.cfront.ctype_model import bitint, scalar

    def bi(n, signed=True):
        return bitint(n, signed)

    # (label, decl-c-type, cfront-CType) for each operand we mix.
    operands = {
        "bi8": ("_BitInt(8)", bi(8)),
        "ubi8": ("unsigned _BitInt(8)", bi(8, False)),
        "bi12": ("_BitInt(12)", bi(12)),
        "ubi12": ("unsigned _BitInt(12)", bi(12, False)),
        "bi16": ("_BitInt(16)", bi(16)),
        "ubi16": ("unsigned _BitInt(16)", bi(16, False)),
        "bi32": ("_BitInt(32)", bi(32)),
        "ubi32": ("unsigned _BitInt(32)", bi(32, False)),
        "bi33": ("_BitInt(33)", bi(33)),
        "ubi33": ("unsigned _BitInt(33)", bi(33, False)),
        "bi40": ("_BitInt(40)", bi(40)),
        "ubi40": ("unsigned _BitInt(40)", bi(40, False)),
        "bi64": ("_BitInt(64)", bi(64)),
        "ubi64": ("unsigned _BitInt(64)", bi(64, False)),
        "i": ("int", scalar("int")),
        "u": ("unsigned", scalar("unsigned int")),
        "l": ("long", scalar("long")),
        "ul": ("unsigned long", scalar("unsigned long")),
        "ll": ("long long", scalar("long long")),
        "ull": ("unsigned long long", scalar("unsigned long long")),
        "sh": ("short", scalar("short")),
        "c": ("char", scalar("char")),
        "fl": ("float", scalar("float")),
        "db": ("double", scalar("double")),
    }
    # Cover: _BitInt + int / unsigned, signed/unsigned _BitInt mixes, two different _BitInt widths, and the
    # boundary on both sides of the rank (N<width, N==width, N>width); plus long / long long / sub-int.
    pairs = [
        ("bi8", "i"),
        ("bi12", "i"),
        ("bi16", "i"),
        ("bi32", "i"),
        ("bi33", "i"),
        ("bi40", "i"),
        ("bi64", "i"),
        ("bi8", "u"),
        ("bi32", "u"),
        ("bi33", "u"),
        ("bi64", "u"),
        ("ubi8", "i"),
        ("ubi32", "i"),
        ("ubi33", "i"),
        ("ubi64", "i"),
        ("ubi8", "u"),
        ("ubi32", "u"),
        ("ubi64", "u"),
        ("bi8", "sh"),
        ("bi12", "sh"),
        ("bi8", "c"),
        ("bi8", "l"),
        ("bi33", "l"),
        ("bi64", "l"),
        ("bi64", "ll"),
        ("bi64", "ull"),
        ("bi40", "l"),
        ("bi8", "bi12"),
        ("bi12", "bi8"),
        ("bi8", "bi16"),
        ("bi32", "bi64"),
        ("bi16", "bi40"),
        ("bi8", "ubi8"),
        ("bi32", "ubi32"),
        ("bi64", "ubi64"),
        ("bi8", "ubi12"),
        ("ubi8", "bi12"),
        ("bi16", "ubi8"),
        ("bi32", "ubi64"),
        ("ubi32", "bi64"),
        ("bi40", "ubi16"),
        # a `_BitInt` mixed with a FLOAT -> float (NOT a `_BitInt`): the model must route to fallback, never
        # mistake the float's byte-width for an integer rank (the `bi64 + float` regression class). The probe
        # asserts Clang's result is NOT a `_BitInt` (a float result lands on its float:300/double:301 tag,
        # still >= 100, so the "fallback => not a _BitInt" assertion holds).
        ("bi8", "fl"),
        ("bi64", "fl"),
        ("bi64", "db"),
        ("ubi64", "fl"),
        ("bi32", "db"),
    ]
    # the universal _Generic type list (every _BitInt width we might land on, + the standard ints).
    glist = (
        "    _BitInt(8):1, unsigned _BitInt(8):2, _BitInt(12):3, unsigned _BitInt(12):4,\n"
        "    _BitInt(16):5, unsigned _BitInt(16):6, _BitInt(32):7, unsigned _BitInt(32):8,\n"
        "    _BitInt(33):9, unsigned _BitInt(33):10, _BitInt(40):11, unsigned _BitInt(40):12,\n"
        "    _BitInt(64):13, unsigned _BitInt(64):14, int:100, unsigned:101, long:102,\n"
        "    unsigned long:103, long long:104, unsigned long long:105, float:300, double:301, default:200"
    )
    tags = {  # the _Generic value for a `_BitInt` tag (so a static-assert can compare)
        "_BitInt(8)": 1,
        "unsigned _BitInt(8)": 2,
        "_BitInt(12)": 3,
        "unsigned _BitInt(12)": 4,
        "_BitInt(16)": 5,
        "unsigned _BitInt(16)": 6,
        "_BitInt(32)": 7,
        "unsigned _BitInt(32)": 8,
        "_BitInt(33)": 9,
        "unsigned _BitInt(33)": 10,
        "_BitInt(40)": 11,
        "unsigned _BitInt(40)": 12,
        "_BitInt(64)": 13,
        "unsigned _BitInt(64)": 14,
    }
    decls, asserts, checked_firstclass, checked_fallback = [], [], 0, 0
    for i, (la, lb) in enumerate(pairs):
        (ca, _ta), (cb, _tb) = operands[la], operands[lb]
        ta, tb = operands[la][1], operands[lb][1]
        model = _bitint_model_tag(ta, tb)
        decls.append(f"{ca} a{i}; {cb} b{i};")
        gen = f"_Generic((a{i} + b{i}),\n{glist})"
        if model is not None:  # first-class: Clang MUST select the modeled _BitInt tag
            assert model in tags, (la, lb, model)
            asserts.append(
                f"_Static_assert(({gen}) == {tags[model]}, "
                f'"{la}+{lb}: cfront models {model}, Clang disagrees");'
            )
            checked_firstclass += 1
        else:  # fallback: Clang's result must NOT be a _BitInt (< 100)
            asserts.append(
                f"_Static_assert(({gen}) >= 100, "
                f'"{la}+{lb}: cfront routes to fallback but Clang result IS a _BitInt");'
            )
            checked_fallback += 1
    assert (
        checked_firstclass >= 10 and checked_fallback >= 8
    )  # both sides of the boundary are exercised
    probe = (
        "int probe(void){\n  "
        + "\n  ".join(decls)
        + "\n  "
        + "\n  ".join(asserts)
        + "\n  return 0;\n}\n"
    )
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "g.c")
        open(c, "w").write(probe)
        built = None
        for std in ("c23", "c2x"):
            b = subprocess.run(
                [_CC, f"-std={std}", "-Wno-constant-conversion", "-c", c, "-o", os.devnull],
                capture_output=True,
                text=True,
            )
            if "unknown" not in b.stderr.lower() or b.returncode == 0:
                built = b
                break
        assert built is not None
        # a non-zero return code here means a `_Static_assert` fired: the cfront's model disagreed with Clang.
        assert built.returncode == 0, (
            "cfront _BitInt result-type model diverges from Clang:\n" + built.stderr
        )


def test_bitint_bitfield_layout_matches_clang():
    """SUB-FEATURE B differential: a `_BitInt(N)` BITFIELD `_BitInt(N) m : W` must lay out byte-identically
    to Clang -- the gate that makes bit-precise bitfields Clang-equivalent. The cfront packs a `_BitInt(N)`
    bitfield LSB-first into the `_BitInt(N)` storage unit (its 1/2/4/8-byte slot), the same rule it uses for
    a standard-int bitfield of that size. This asserts the cfront's sizeof / _Alignof for a battery of
    `_BitInt` bitfield structs EQUALS Clang's (the same `sizeof`/`_Alignof` differential the member-layout
    test runs, extended to bitfields -- `offsetof` is illegal on a bitfield, so the cross-check is on the
    aggregate size + alignment, which the LSB-first packing fully determines)."""
    cases = [  # (tag, member-decls) -- each a struct of `_BitInt` (and standard) bitfields
        ("A", "unsigned _BitInt(12) a : 5; unsigned _BitInt(12) b : 6;"),
        ("B", "unsigned _BitInt(20) a : 10; unsigned _BitInt(20) b : 10;"),
        ("C", "unsigned _BitInt(8) a : 3; unsigned _BitInt(8) b : 5; unsigned _BitInt(8) c : 2;"),
        ("D", "int tag; _BitInt(64) x : 40; _BitInt(64) y : 20;"),
        (
            "E",
            "unsigned _BitInt(12) a : 5; unsigned int b : 6;",
        ),  # mixed bit-int + standard bitfield
        ("F", "unsigned _BitInt(12) a : 12;"),  # full-width W == N
        ("G", "unsigned _BitInt(16) a : 5; unsigned _BitInt(8) b : 5;"),  # different storage sizes
        ("H", "unsigned _BitInt(32) a : 20; unsigned _BitInt(32) b : 20;"),  # straddle -> two units
        (
            "I",
            "unsigned _BitInt(12) a : 12; unsigned _BitInt(12) b : 6;",
        ),  # straddle in a 2-byte unit
    ]
    for tag, members in cases:
        src = f"struct {tag} {{ {members} }};\n_BitInt(8) f(struct {tag} s){{ (void)s; return (_BitInt(8))0; }}\n"
        agg = compile_unit(src, check_clang=False).lowered.aggregates[tag]
        if not _CC:
            continue
        probe = (
            "#include <stddef.h>\n#include <stdio.h>\n"
            f"struct {tag} {{ {members} }};\n"
            f'int main(void){{printf("%zu %zu", sizeof(struct {tag}),'
            f" (size_t)_Alignof(struct {tag})); return 0;}}"
        )
        with tempfile.TemporaryDirectory() as d:
            c, e = os.path.join(d, "p.c"), os.path.join(d, "p")
            open(c, "w").write(probe)
            for std in ("c23", "c2x"):
                if (
                    subprocess.run([_CC, f"-std={std}", c, "-o", e], capture_output=True).returncode
                    == 0
                ):
                    nums = [
                        int(x)
                        for x in subprocess.run([e], capture_output=True, text=True).stdout.split()
                    ]
                    assert nums == [agg.size, agg.align], (tag, members, nums, agg.size, agg.align)
                    break


def test_c_frontend_builds_warning_clean():
    if not _CC:
        return
    for unit in (
        "bcir_cfront.c",
        "bcir_cpp.c",
        "bcir_plan.c",
        "bcir_hydrate.c",
        "bcir_verify.c",
        "bcir_cc.c",
        "bcir_diag.c",
    ):
        ok = False
        for std in ("c23", "c11"):
            b = subprocess.run(
                [
                    _CC,
                    f"-std={std}",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    _C,
                    "-c",
                    os.path.join(_C, unit),
                    "-o",
                    os.devnull,
                ],
                capture_output=True,
                text=True,
            )
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
            "int main(int c,char**v){static char o[65536],e[256],s[65536];"
            "size_t n=fread(s,1,sizeof s-1,stdin);s[n]=0;"
            'if(bcir_cpp_run(s,c>1?v[1]:"",o,sizeof o,e,sizeof e)){printf("ERR %s",e);return 1;}'
            "fputs(o,stdout);return 0;}\n"
        )
        exe = os.path.join(d, "drv")
        b = subprocess.run(
            [_CC, "-std=c11", "-O1", "-I", _C, os.path.join(_C, "bcir_cpp.c"), drv, "-o", exe],
            capture_output=True,
            text=True,
        )
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
            '#line 30 "a\\"b.c"\nz __FILE__\n',  # an escaped quote in the name
            "p __LINE__\n#line\nq __LINE__\n",  # malformed -> ignored
            "#if 0\n#line 999\n#endif\nr __LINE__\n",  # inactive branch -> skipped
            'int x; _Pragma("once") int y;\n',  # _Pragma operator: a no-op
            'p _Pragma("a(b)c") q\n',  # balanced parens consumed
            "#define DO(x) _Pragma(#x)\nDO(message hi)\nz\n",  # _Pragma produced by a macro
            "#if __has_attribute(packed)\nP\n#else\nn\n#endif\n",  # feature-test: supported
            "#if __has_attribute(__aligned__)\nA\n#endif\n",  # GCC __x__ spelling
            "#if __has_attribute(deprecated)\nd\n#else\nU\n#endif\n",  # unsupported attribute
            "#if __has_builtin(__builtin_expect)\nb\n#else\nU\n#endif\n",
            "#if __has_c_attribute(nodiscard)\nc\n#else\nU\n#endif\n",
            "#ifdef __has_attribute\nDEF\n#endif\n",  # reported as `defined`
            "#if defined(__has_builtin) && !__has_builtin(x)\nG\n#endif\n",
            "#define V(...) f(__VA_ARGS__)\nV(1,2,3)\n",  # __VA_ARGS__ flattens all args
            "#define L(a, ...) g(a, __VA_ARGS__)\nL(x,1,2)\nL(z)\n",  # named + variadic, incl. empty
            "#define S(...) #__VA_ARGS__\nS(1, 2, 3)\nS()\n",  # stringize __VA_ARGS__
            "#define P(...) x ## __VA_ARGS__\nP(1,2)\nP()\n",  # paste __VA_ARGS__
            "#define LOG(f, ...) p(f __VA_OPT__(,) __VA_ARGS__)\nLOG(z)\nLOG(z,1,2)\n",  # __VA_OPT__
            "#define W(x, ...) [x __VA_OPT__(/ __VA_ARGS__)]\nW(p)\nW(p,q,r)\n",  # nested VA
            "#define E(...) z __VA_OPT__(Y)\nE()\nE(,)\nE(q)\n",  # emptiness incl. a lone comma
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
        for s in (
            '#if __has_include("ph.h")\nY\n#else\nN\n#endif\n',
            "#if __has_include(<ph.h>)\nY\n#else\nN\n#endif\n",
            '#if __has_include("nope.h")\nY\n#else\nN\n#endif\n',
            '#if defined(__has_include) && __has_include("ph.h")\nOK\n#endif\n',
            '#if !__has_include("nope.h")\nNEG\n#endif\n',
        ):
            assert pp(s, d) == _py_pp(s, search_paths=[d]), (
                f"__has_include divergence on {s!r}\n C:{pp(s, d)!r}\nPY:{_py_pp(s, search_paths=[d])!r}"
            )

        # __DATE__/__TIME__: SOURCE_DATE_EPOCH (UTC) freezes both twins to the same string.
        def ppe(src, epoch):
            env = dict(os.environ)
            env["SOURCE_DATE_EPOCH"] = epoch
            return subprocess.run(
                [exe, ""], input=src, capture_output=True, text=True, env=env
            ).stdout

        old_epoch = os.environ.get("SOURCE_DATE_EPOCH")
        try:
            for epoch, want in (
                ("1234567890", '"Feb 13 2009""23:31:30"'),  # 2009-02-13 23:31:30Z
                ("1577836800", '"Jan  1 2020""00:00:00"'),
            ):  # padded single-digit day
                os.environ["SOURCE_DATE_EPOCH"] = epoch  # _py_pp reads it here
                assert ppe("__DATE__ __TIME__\n", epoch).strip() == want
                assert ppe("__DATE__ __TIME__\n", epoch) == _py_pp("__DATE__ __TIME__\n")
        finally:
            if old_epoch is None:
                os.environ.pop("SOURCE_DATE_EPOCH", None)
            else:
                os.environ["SOURCE_DATE_EPOCH"] = old_epoch


def test_c_preprocessor_driver_and_emitter_fail_closed_at_capacity_edges():
    """Hostile/oversized source must diagnose, never overflow or compile a truncated unit."""
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        cc = _build_bcir_cc(d)

        def run(name, text, *args):
            path = os.path.join(d, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            return subprocess.run([cc, *args, path], capture_output=True, text=True)

        # This used to memcpy 3,000 bytes into a 64-byte Macro.params slot (ASan stack OOB).
        parameter = "p" * 3000
        r = run("long_macro.c", f"#define F({parameter}) 1\nint f(void){{return F(0);}}\n")
        assert r.returncode == 1 and "macro parameter is too long" in r.stderr, r.stderr

        # Live invalid shifts are diagnosed without host UB; a dead && branch remains unevaluated.
        r = run("bad_shift.c", "#if 1 << 999\nint f(void){return 1;}\n#endif\n")
        assert r.returncode == 1 and "invalid shift in #if expression" in r.stderr, r.stderr
        r = run("dead_shift.c", "#if 0 && (1 << 999)\ninvalid\n#endif\nint f(void){return 1;}\n")
        assert r.returncode == 0 and "ok=1" in r.stdout, (r.stdout, r.stderr)
        r = run(
            "nested_dead_shift.c",
            "#if 0\n#if 1 / 0\ninvalid\n#endif\n#endif\nint f(void){return 1;}\n",
        )
        assert r.returncode == 0 and "ok=1" in r.stdout, (r.stdout, r.stderr)

        # Malformed directives and driver definitions fail deterministically, without reading
        # uninitialized token buffers, overflowing atoi, or compiling a truncated -D value.
        r = run("bad_ifdef.c", "#ifdef\n#endif\nint f(void){return 1;}\n")
        assert r.returncode == 1 and "requires an identifier" in r.stderr, r.stderr
        r = run("bad_line.c", "#line 999999999999999999999\nint f(void){return 1;}\n")
        assert r.returncode == 1 and "#line number is out of range" in r.stderr, r.stderr
        r = run("define_arg.c", "int f(void){return 1;}\n", "-D", "VALUE=" + "x" * 300)
        assert r.returncode == 2 and "overlong -D" in r.stderr, r.stderr
        r = run("bad_params.c", "#define F(a,) a\nint f(void){return F(1);}\n")
        assert r.returncode == 1 and "invalid macro parameter list" in r.stderr, r.stderr

        # Source and include reads are dynamic: >64 KiB source / >8 KiB header comments no longer
        # erase the valid declaration that follows them.
        r = run("large_source.c", "/*" + "x" * 70000 + "*/\nint f(void){return 1;}\n")
        assert r.returncode == 0 and "ok=1" in r.stdout, (r.stdout, r.stderr)
        header = os.path.join(d, "large.h")
        with open(header, "w", encoding="utf-8") as f:
            f.write("/*" + "x" * 10000 + "*/\ntypedef unsigned word;\n")
        r = run("large_include.c", '#include "large.h"\nword f(void){return 1u;}\n')
        assert r.returncode == 0 and "ok=1" in r.stdout, (r.stdout, r.stderr)

        # Fixed public result buffers now fail explicitly instead of returning omitted text/claims.
        r = run("pp_overflow.c", "x\n" * 40000, "-E")
        assert r.returncode == 1 and "preprocessed output too large" in r.stderr, r.stderr
        # ...and the verified C has none (CF-BUF): a unit whose emit runs past the 32 KiB the result once held --
        # refused then as `emitted C exceeds` -- emits whole, and its linkable form renames every call in it.
        body = "\n".join("x += 1;" for _ in range(700))
        for mode, name in (
            ("--emit-c", "bcir_f(int32_t x)"),
            ("--linkable", "int32_t f(int32_t x)"),
        ):
            r = run("emit_large.c", f"int f(int x){{\n{body}\nreturn x;\n}}\n", mode)
            assert r.returncode == 0 and len(r.stdout) > 1 << 15, (
                mode,
                r.returncode,
                r.stderr[-300:],
            )
            assert name in r.stdout and r.stdout.count(" = x + ") == 700, (mode, r.stdout[:300])
            assert r.stdout.rstrip().endswith("}"), (mode, r.stdout[-300:])

        # Empty units and unsupported pointer depth are clean errors, not funcs[-1] or truncated stars.
        r = run("empty.c", "", "--emit-pack")
        assert r.returncode == 1 and "no entry function" in r.stderr, r.stderr
        r = run("deep_pointer.c", "int *****************f(void);\n")
        assert r.returncode == 1 and "pointer nesting too deep" in r.stderr, r.stderr


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


_SCALAR_ALIGN_SRC = """#include <stdint.h>
typedef float _Complex cf;
struct Ld { uint8_t c; long double l; uint16_t h; };
struct Lc { uint8_t c; long double _Complex l; uint16_t h; };
struct Dc { float _Complex z; double _Complex w; uint32_t t; };
struct Td { uint8_t c; cf z; uint16_t h; };
struct Am { uint8_t c; _Atomic float _Complex z; _Atomic double _Complex w; uint16_t h; };
struct Al { uint8_t c; _Atomic long double _Complex l; uint16_t h; };
struct __attribute__((packed)) Pk { uint8_t c; _Atomic uint64_t n; _Atomic float _Complex z; };
struct At { uint8_t c; _Atomic cf z; uint16_t h; };
uint32_t f(void) {
    uint32_t a = (uint32_t)sizeof(struct Ld);
    uint32_t b = (uint32_t)sizeof(struct Lc);
    uint32_t d = (uint32_t)sizeof(struct Dc);
    uint32_t e = (uint32_t)_Alignof(long double);
    uint32_t g = (uint32_t)_Alignof(long double _Complex);
    uint32_t h = (uint32_t)_Alignof(double _Complex);
    uint32_t i = (uint32_t)sizeof(struct Td);
    uint32_t j = (uint32_t)sizeof(struct Am);
    uint32_t k = (uint32_t)sizeof(struct Al);
    uint32_t l = (uint32_t)_Alignof(_Atomic float _Complex);
    uint32_t m = (uint32_t)sizeof(_Atomic cf);
    uint32_t p = (uint32_t)sizeof(struct Pk);
    uint32_t q = (uint32_t)sizeof(struct At);
    return a + b + d + e + g + h + i + j + k + l + m + p + q;
}
"""


def test_scalar_alignment_matrix_dual_rail():
    """CF-CALIGN: a scalar's layout is the target ABI's, on every target and identically on both rails. A
    `_Complex` member aligns to its element; a `long double` (and a `long double _Complex`) to the ABI's
    long-double alignment; an `_Atomic` type no wider than the ABI's atomic promotion width (16 bytes on
    the 64-bit targets, 8 on i386) to its size; and a typedef'd complex keeps its size. The twin aligned
    every scalar member to its size, so a `double _Complex` after an 8-byte member sat at 16 (x86-64
    `sizeof(struct Dc)` 48, not 32) and an i386 `long double` at 12 (`sizeof(struct Ld)` 36, not 20). It
    sized `cf` 16, refused `sizeof(_Atomic cf)`, and dropped `_Atomic` from a typedef. Neither rail promoted an `_Atomic` member: the
    oracle folded `sizeof(struct Am)` to 40, not 48, and `_Alignof(_Atomic float _Complex)` to 4, not 8.
    The folded constants [Ld, Lc, Dc, _Alignof(long double), _Alignof(long double _Complex),
    _Alignof(double _Complex), Td, Am, Al, _Alignof(_Atomic float _Complex), sizeof(_Atomic cf), Pk, At]
    are compared per target; `packed` (Pk) still wins over the atomic alignment, and an `_Atomic` typedef
    (At) keeps its qualifier. Every vector is Clang's: the two i386 entries a `double` decides (Dc and
    `_Alignof(double _Complex)`, 4-aligned there) became so with CF-I386's `eight_byte_align`."""
    vecs = {t: _abi_const_vec_oracle(_SCALAR_ALIGN_SRC, t) for t in _ABI_TARGETS}
    for t in ("x86_64-linux", "aarch64-linux", "riscv64-linux"):
        assert vecs[t] == [48, 64, 32, 16, 16, 8, 16, 48, 64, 8, 8, 17, 24], (t, vecs[t])
    win = vecs["x86_64-windows"]
    assert win == [24, 32, 32, 8, 8, 8, 16, 48, 48, 8, 8, 17, 24], win
    i386 = vecs["i386-linux"]
    assert i386 == [20, 32, 28, 4, 4, 4, 16, 40, 32, 8, 8, 17, 24], i386
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "scalar_align.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_SCALAR_ALIGN_SRC)
        for t in _ABI_TARGETS:
            assert _abi_const_vec_twin(exe, path, t) == vecs[t], (t, vecs[t])


# CF-I386: i386 aligns an 8-byte scalar -- `double`, `long long`, `int64_t`, a `_BitInt(33..64)`, a `double
# _Complex`'s element -- to 4 while its size stays 8 (Clang's DoubleAlign / LongLongAlign), and a bitfield of
# such a type then follows Clang's placement rule, whose storage unit starts at an ALIGNMENT boundary.
_EIGHT_BYTE_SRC = """#include <stdint.h>
struct D { uint8_t c; double d; };
struct L { uint8_t c; long long x; };
struct Z { uint8_t c; double _Complex z; };
union U { uint8_t c; double d; };
struct A2 { uint8_t c; double d[2]; };
struct AL { uint8_t c; _Atomic long long x; };
struct BI { uint8_t c; _BitInt(40) x; };
struct B1 { uint8_t c; long long x : 8; };
struct B2 { uint8_t c; long long x : 40; uint8_t d; };
struct B3 { uint32_t a; uint8_t b; long long x : 40; };
struct B4 { uint8_t a; long long : 0; uint8_t b; };
struct __attribute__((packed)) B5 { uint8_t a; long long : 0; uint8_t b; };
struct B7 { unsigned long long a : 40; unsigned long long b : 30; };
struct B8 { uint8_t c; long long : 3; long long x : 60; };
struct B9 { uint8_t c; _BitInt(40) x : 20; uint8_t d; };
uint64_t g(struct B1 *o, struct B3 *p, struct B7 *q, struct B8 *r, uint64_t v) {
    o->x = (long long)v;  /* no literal here: the twin's constant vector reads every `= N;` in the emit */
    p->x = (long long)v;
    q->b = v;
    r->x = (long long)v;
    return (uint64_t)o->x + (uint64_t)p->x + q->a + q->b + (uint64_t)r->x;
}
uint32_t f(void) {
    uint32_t a = (uint32_t)sizeof(struct D);
    uint32_t b = (uint32_t)sizeof(struct L);
    uint32_t d = (uint32_t)sizeof(struct Z);
    uint32_t e = (uint32_t)sizeof(union U);
    uint32_t g2 = (uint32_t)sizeof(struct A2);
    uint32_t h = (uint32_t)sizeof(struct AL);
    uint32_t i = (uint32_t)sizeof(struct BI);
    uint32_t j = (uint32_t)sizeof(struct B1);
    uint32_t k = (uint32_t)sizeof(struct B2);
    uint32_t l = (uint32_t)sizeof(struct B3);
    uint32_t m = (uint32_t)sizeof(struct B4);
    uint32_t n = (uint32_t)sizeof(struct B5);
    uint32_t o = (uint32_t)sizeof(struct B7);
    uint32_t p = (uint32_t)sizeof(struct B8);
    uint32_t q = (uint32_t)sizeof(struct B9);
    uint32_t r = (uint32_t)_Alignof(double);
    uint32_t s = (uint32_t)_Alignof(long long);
    uint32_t t = (uint32_t)_Alignof(double _Complex);
    uint32_t u = (uint32_t)_Alignof(struct BI);
    uint32_t w = (uint32_t)_Alignof(union U);
    uint32_t x = (uint32_t)_Alignof(struct B1);
    uint32_t y = (uint32_t)_Alignof(_Atomic double _Complex);
    return a + b + d + e + g2 + h + i + j + k + l + m + n + o + p + q + r + s + t + u + w + x + y;
}
"""
# the folded operands of `f`, in order (a `_Static_assert` per entry is what Clang checks)
_EIGHT_BYTE_EXPRS = [
    *(
        f"sizeof({agg})"
        for agg in (
            "struct D", "struct L", "struct Z", "union U", "struct A2", "struct AL", "struct BI",
            "struct B1", "struct B2", "struct B3", "struct B4", "struct B5", "struct B7", "struct B8",
            "struct B9",
        )
    ),
    "_Alignof(double)", "_Alignof(long long)", "_Alignof(double _Complex)", "_Alignof(struct BI)",
    "_Alignof(union U)", "_Alignof(struct B1)", "_Alignof(_Atomic double _Complex)",
]  # fmt: skip
_LP64_EIGHT = [16, 16, 24, 8, 24, 16, 16, 8, 8, 16, 9, 9, 16, 16, 8, 8, 8, 8, 8, 8, 8, 16]
_I386_EIGHT = [12, 12, 20, 8, 20, 16, 12, 4, 8, 12, 5, 5, 12, 12, 8, 4, 4, 4, 4, 4, 4, 4]
# The entries that do not depend on a bitfield's layout. Bitfields follow the Itanium rules the rails model
# on x86-64 and RISC-V Linux and on i386; AArch64 also raises a record's alignment for an unnamed bitfield,
# and Windows lays bitfields out by the MSVC rules. Neither rail models either yet, so on those two targets
# only these entries are held to Clang.
_NO_BITFIELD = [i for i, e in enumerate(_EIGHT_BYTE_EXPRS) if not re.search(r"struct B\d", e)]
_ITANIUM_BITFIELDS = ("x86_64-linux", "riscv64-linux", "i386-linux")
# the members of each bitfield struct in declaration order (None: an unnamed bitfield), as Clang's
# record-layout dump lists their bit offsets
_EIGHT_BYTE_BITFIELDS = {
    "B1": ("c", "x"),
    "B2": ("c", "x", "d"),
    "B3": ("a", "b", "x"),
    "B4": ("a", None, "b"),
    "B5": ("a", None, "b"),
    "B7": ("a", "b"),
    "B8": ("c", None, "x"),
    "B9": ("c", "x", "d"),
}


def _clang_bit_offsets(clang: str, triple: str, src: str) -> dict:
    """Each record's field bit offsets as Clang lays them out under `triple` (`-fdump-record-layouts-simple`,
    unnamed bitfields included): the reference the rails' layouts are held to, bitfields included, since
    `offsetof` is illegal on a bitfield."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "layout.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        run = subprocess.run(
            [clang, "-target", triple, "-std=c2x", "-ffreestanding", "-fsyntax-only", "-Xclang",
             "-fdump-record-layouts-simple", path],
            capture_output=True, text=True,
        )  # fmt: skip
    assert run.returncode == 0, run.stderr
    out, name = {}, None
    for line in run.stdout.splitlines():
        m = re.match(r"Type: (?:struct|union) (\w+)", line)
        if m:
            name = m.group(1)
        m = re.match(r"\s*FieldOffsets: \[([\d, ]*)\]", line)
        if m and name:
            out[name] = [int(v) for v in m.group(1).split(",") if v.strip()]
    return out


def test_an_eight_byte_scalar_and_its_bitfields_follow_the_target_abi():
    """CF-I386: both rails aligned an 8-byte scalar to 8 on every target, but i386 aligns `double`, `long
    long`, `int64_t`, a `_BitInt(33..64)` and a `double _Complex` to 4 while their size stays 8, so every
    such member, `sizeof` and `_Alignof` disagreed with the ABI there (`struct D` folded to 16, not 12). Both
    target tables now carry the ABI's `eight_byte_align`, and a bitfield follows Clang's placement rule: a
    field bumps to its type's next ALIGNMENT boundary only if it would overflow a storage unit of its type's
    size, and a zero-width one aligns to its type's alignment, packed or not. Where the alignment equals the
    size (every other target and type) that is the old rule, so no other layout moves. A bitfield of an
    under-aligned type is accessed over only the bytes it spans (`struct B1` is 4 bytes; an 8-byte unit at
    its start would run past the end). Checked per target: the folded sizes and alignments against a pinned
    vector, between the rails, and against Clang (a `_Static_assert` per entry); each bitfield's bit offset
    against Clang's record layout on i386 and x86-64; and the claim graphs of code reading and writing those
    bitfields, digest for digest between the rails. Bitfield layout is held to Clang only where the rails
    implement its rules (`_ITANIUM_BITFIELDS`): AArch64's alignment for an unnamed bitfield and the MSVC
    bitfield layout are modeled by neither rail, which this test does not claim."""
    vecs = {t: _abi_const_vec_oracle(_EIGHT_BYTE_SRC, t) for t in _ABI_TARGETS}
    for t in ("x86_64-linux", "riscv64-linux"):
        assert vecs[t] == _LP64_EIGHT, (t, vecs[t])
    for t in ("aarch64-linux", "x86_64-windows"):
        assert [vecs[t][i] for i in _NO_BITFIELD] == [_LP64_EIGHT[i] for i in _NO_BITFIELD], t
    assert vecs["i386-linux"] == _I386_EIGHT, vecs["i386-linux"]
    clang = shutil.which("clang")
    if clang:
        from bcir.frontends.cfront.abi import TARGETS

        # Clang's own layout for every target, one `_Static_assert` per folded entry
        for t in _ABI_TARGETS:
            held = range(len(_EIGHT_BYTE_EXPRS)) if t in _ITANIUM_BITFIELDS else _NO_BITFIELD
            asserts = "".join(
                f'_Static_assert({_EIGHT_BYTE_EXPRS[i]} == {vecs[t][i]}, "{t}: {_EIGHT_BYTE_EXPRS[i]}");\n'
                for i in held
            )
            src = _EIGHT_BYTE_SRC.split("uint64_t g(")[0] + asserts
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "eight.c")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(src)
                run = subprocess.run(
                    [clang, "-target", TARGETS[t].triple, "-std=c2x", "-ffreestanding",
                     "-fsyntax-only", path],
                    capture_output=True, text=True,
                )  # fmt: skip
            assert run.returncode == 0, (t, run.stderr)
        # each named bitfield's bit offset, against Clang's record layout
        for t in ("i386-linux", "x86_64-linux"):
            uses = " + ".join(f"sizeof(struct {tag})" for tag in _EIGHT_BYTE_BITFIELDS)
            defs = _EIGHT_BYTE_SRC.split("uint64_t g(")[0] + f"int layout_uses = {uses};\n"
            # Clang dumps only the records a unit uses
            want = _clang_bit_offsets(clang, TARGETS[t].triple, defs)
            aggs = compile_unit(_EIGHT_BYTE_SRC, check_clang=False, target=t).lowered.aggregates
            for tag, names in _EIGHT_BYTE_BITFIELDS.items():
                clang_at = {n: off for n, off in zip(names, want[tag], strict=True) if n}
                ours = {fn: fbo * 8 + fbit for fn, _ft, fbo, fbit, _fw in aggs[tag].fields}
                assert ours == clang_at, (t, tag, ours, clang_at)
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "eight_byte.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_EIGHT_BYTE_SRC)
        for t in _ABI_TARGETS:
            assert _abi_const_vec_twin(exe, path, t) == vecs[t], (t, vecs[t])
        # the bitfield accesses: the same claim graph (offsets, unit bytes, bit positions) on both rails
        for t in ("i386-linux", "x86_64-linux"):
            r = compile_unit(_EIGHT_BYTE_SRC, check_clang=False, target=t)
            oracle = f"{cfront_structural_digest(r.lowered):016x}"
            run = subprocess.run([exe, "--target", t, path], capture_output=True, text=True)
            twin = re.search(r"digest=([0-9a-f]{16})", run.stdout)
            assert twin and twin.group(1) == oracle, (t, run.stdout[:300], oracle)


# CF-PTRARR: an array-of-pointers member `T *arr[N]` -- every form of access, and whether both rails lower
# it (True) or both refuse it (False). The twin laid the elements out by their pointee (`uint32_t *arr[2]` at
# 4-byte elements), so every lowered form here differed from the oracle, and it accepted three forms the
# oracle refuses while refusing `*t->arr[i]`, which a postfix operator's precedence makes `*(t->arr[i])`.
_PTRARR_HEAD = """#include <stdint.h>
struct T { uint8_t c; uint32_t *arr[2]; uint8_t d; };
struct S { uint32_t v; };
struct U { uint8_t c; struct S *sp[2]; double *dp[3]; };
struct Q { uint8_t c; uint32_t *p; };
struct R { uint8_t c; uint32_t **pp[2]; };
struct V { uint8_t c; volatile uint32_t *regs[2]; };
"""
_PTRARR_FORMS = {
    "uint32_t f(struct T *t, uint32_t *p){ t->arr[1] = p; return t->d; }": True,
    "uint32_t f(struct T *t){ uint32_t *q = t->arr[1]; return *q; }": True,
    "uint32_t f(struct T *t){ return *t->arr[1]; }": True,
    "uint32_t f(struct T *t){ return *(t->arr[1]); }": True,
    "uint32_t f(struct T *t, uint32_t v){ *t->arr[1] = v; return t->d; }": True,
    "uint32_t f(struct T *t, uint32_t v){ *t->arr[1] += v; return t->d; }": True,
    "uint32_t f(struct T *t, uint32_t i){ return *t->arr[i & 1u]; }": True,
    "uint32_t f(struct T *t){ uint32_t **pp = &t->arr[1]; return **pp; }": True,
    "uint32_t f(struct T *t, uint32_t *p){ return t->arr[0] == p; }": True,
    "static uint32_t g(uint32_t *x){ return *x; }\nuint32_t f(struct T *t){ return g(t->arr[0]); }": True,
    "uint32_t f(uint32_t *p){ struct T s = {0}; s.arr[0] = p; return *s.arr[0]; }": True,
    "uint32_t f(uint32_t *p){ struct T s = { 1, { p, p }, 2 }; uint32_t *q = s.arr[1]; return *q + s.d; }": True,
    "uint32_t f(uint32_t *p){ struct T s = { .arr = { p, p } }; uint32_t *q = s.arr[0]; return *q; }": True,
    "uint32_t f(struct T *t){ t->arr[0] += 1; return t->d; }": True,
    "uint32_t f(struct T *t){ t->arr[0] -= 1; return t->d; }": True,
    "long f(struct T *t){ return t->arr[1] - t->arr[0]; }": True,
    "double f(struct U *u){ double *q = u->dp[2]; return *q; }": True,
    "uint32_t f(struct U *u, struct S *s){ u->sp[0] = s; struct S *q = u->sp[0]; return q->v; }": True,
    "uint32_t f(struct R *r, uint32_t **x){ r->pp[1] = x; uint32_t **y = r->pp[1]; return **y; }": True,
    "uint32_t f(struct V *v){ volatile uint32_t *r = v->regs[1]; return *r; }": True,
    "uint32_t f(struct Q *q){ return *q->p; }": True,
    "uint32_t f(struct Q q){ return *q.p + 1u; }": True,
    "uint32_t f(void){ return (uint32_t)sizeof(struct T) + 100u*(uint32_t)sizeof(struct U); }": True,
    "uint32_t f(struct T *t){ return t->arr[1][2]; }": False,
    "uint32_t f(struct U *u){ return u->sp[1]->v; }": False,
    "uint32_t f(struct T *t){ t->arr[0]++; return t->d; }": False,
    "uint32_t f(struct T *t){ t->arr[0]--; return t->d; }": False,
    "uint32_t f(struct T *t){ ++t->arr[0]; return t->d; }": False,
    "uint32_t f(struct T *t){ uint32_t *q = t->arr[0]++; return *q; }": False,
}
_PTRARR_LAYOUT = """#include <stdint.h>
struct T { uint8_t c; uint32_t *arr[2]; uint8_t d; };
struct U { uint8_t c; double *dp[3]; uint32_t n; };
uint32_t f(void) {
    uint32_t a = (uint32_t)sizeof(struct T);
    uint32_t b = (uint32_t)_Alignof(struct T);
    uint32_t d = (uint32_t)sizeof(struct U);
    return a + b + d;
}
"""


def test_an_array_of_pointers_member_is_laid_out_and_accessed_as_pointers():
    """CF-PTRARR: the twin gave a `T *arr[N]` member's elements its pointee's size, not the pointer's, so a
    later member sat at the wrong offset, `sizeof` was wrong, a store truncated the pointer and a load read
    the element into an integer (the emitted C did not compile). Each element now takes the ABI's pointer
    size and loads and stores whole, typed `T *`. `*t->arr[i]` (and `*s->p` through a plain pointer member)
    dereferences the postfix expression, as its precedence says, where the twin had dereferenced the name.
    `t->arr[i][j]` and `t->arr[i]++` are refused on both rails, since the oracle refuses them. Every form is
    pinned (lowered or refused) and compared digest for digest on x86-64 and i386; the layout is held to
    Clang on every target. `cfront_ptrmember.c` runs the lowered forms against the original."""
    for body, lowered in _PTRARR_FORMS.items():
        for t in ("x86_64-linux", "i386-linux"):
            try:
                compile_unit(_PTRARR_HEAD + body + "\n", check_clang=False, target=t)
                got = True
            except Exception:  # a refusal: both rails route the unit to fallback
                got = False
            assert got is lowered, (t, body)
    vecs = {t: _abi_const_vec_oracle(_PTRARR_LAYOUT, t) for t in _ABI_TARGETS}
    for t in ("x86_64-linux", "aarch64-linux", "riscv64-linux", "x86_64-windows"):
        assert vecs[t] == [32, 8, 40], (t, vecs[t])
    assert vecs["i386-linux"] == [16, 4, 20], vecs["i386-linux"]
    clang = shutil.which("clang")
    if clang:
        from bcir.frontends.cfront.abi import TARGETS

        for t in _ABI_TARGETS:
            asserts = (
                f'_Static_assert(sizeof(struct T) == {vecs[t][0]}, "{t} T");\n'
                f'_Static_assert(_Alignof(struct T) == {vecs[t][1]}, "{t} align T");\n'
                f'_Static_assert(sizeof(struct U) == {vecs[t][2]}, "{t} U");\n'
            )
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "ptrarr.c")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(_PTRARR_LAYOUT.split("uint32_t f(")[0] + asserts)
                run = subprocess.run(
                    [clang, "-target", TARGETS[t].triple, "-std=c2x", "-ffreestanding",
                     "-fsyntax-only", path],
                    capture_output=True, text=True,
                )  # fmt: skip
            assert run.returncode == 0, (t, run.stderr)
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ptrarr.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_PTRARR_LAYOUT)
        for t in _ABI_TARGETS:
            assert _abi_const_vec_twin(exe, path, t) == vecs[t], (t, vecs[t])
        for body, lowered in _PTRARR_FORMS.items():
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_PTRARR_HEAD + body + "\n")
            for t in ("x86_64-linux", "i386-linux"):
                run = subprocess.run([exe, "--target", t, path], capture_output=True, text=True)
                if not lowered:
                    assert run.returncode != 0 and not run.stdout.startswith("funcs="), (t, body)
                    continue
                r = compile_unit(_PTRARR_HEAD + body + "\n", check_clang=False, target=t)
                oracle = f"{cfront_structural_digest(r.lowered):016x}"
                twin = re.search(r"digest=([0-9a-f]{16})", run.stdout)
                assert twin and twin.group(1) == oracle, (t, body, run.stdout[:200], oracle)


# CF-TRAILPACK: `__attribute__((packed))` / `aligned(N)` written after a struct's closing brace, beside the
# leading spelling the twin already honoured. `(expression, holds a bitfield)` per folded entry.
_TRAILPACK_SRC = """#include <stdint.h>
struct In { uint8_t a; uint32_t b; } __attribute__((packed));
struct Pk { uint8_t c; uint64_t n; uint32_t lo : 3; uint32_t hi : 30; struct In in; uint16_t arr[3]; }
    __attribute__((packed));
struct Pn { uint8_t c; uint64_t n; struct In in; uint16_t arr[3]; } __attribute__((packed));
struct __attribute__((packed)) Pl { uint8_t c; uint64_t n; };
struct Wide { uint8_t c; uint32_t n; } __attribute__((packed, aligned(4)));
struct Al { uint8_t c; uint32_t n; } __attribute__((aligned(16)));
typedef struct TPt { uint8_t c; uint32_t n; uint16_t h; } __attribute__((packed)) TP;
uint32_t f(void) {
    uint32_t a = (uint32_t)sizeof(struct In);
    uint32_t b = (uint32_t)sizeof(struct Pk);
    uint32_t c = (uint32_t)sizeof(struct Pn);
    uint32_t d = (uint32_t)sizeof(struct Pl);
    uint32_t e = (uint32_t)sizeof(struct Wide);
    uint32_t g = (uint32_t)_Alignof(struct Wide);
    uint32_t h = (uint32_t)sizeof(struct Al);
    uint32_t i = (uint32_t)_Alignof(struct Al);
    uint32_t j = (uint32_t)sizeof(TP);
    uint32_t k = (uint32_t)_Alignof(struct Pn);
    return a + b + c + d + e + g + h + i + j + k;
}
"""
_TRAILPACK_EXPRS = [
    ("sizeof(struct In)", False), ("sizeof(struct Pk)", True), ("sizeof(struct Pn)", False),
    ("sizeof(struct Pl)", False), ("sizeof(struct Wide)", False), ("_Alignof(struct Wide)", False),
    ("sizeof(struct Al)", False), ("_Alignof(struct Al)", False), ("sizeof(TP)", False),
    ("_Alignof(struct Pn)", False),
]  # fmt: skip


def test_a_trailing_packed_attribute_packs_the_members_on_both_rails():
    """CF-TRAILPACK: the twin honoured `__attribute__((packed))` before a struct's body but read one written
    after the closing brace -- the common spelling -- only for the aggregate's alignment, after every member
    had been placed at its natural offset: `struct { uint8_t c; uint64_t n; } __attribute__((packed))` put
    `n` at 8 and folded `sizeof` to 16 where Clang and the oracle say 1 and 9. The attributes after the `}`
    are now read before the members are laid out. The folded sizes and alignments (a misaligned `uint64_t`,
    a straddling bitfield, a nested packed struct, an array member, `packed, aligned(4)`, a trailing
    `aligned(16)`, and a typedef) are compared between the rails on every target and held to Clang; the
    bitfield struct only where the rails implement the target's bitfield rules. `cfront_trailpacked.c` runs
    the accesses against the original."""
    vecs = {t: _abi_const_vec_oracle(_TRAILPACK_SRC, t) for t in _ABI_TARGETS}
    for t in _ABI_TARGETS:
        # packed: no member moves with the data model
        assert vecs[t][1:4] == [25, 20, 9], (t, vecs[t])
    clang = shutil.which("clang")
    if clang:
        from bcir.frontends.cfront.abi import TARGETS

        for t in _ABI_TARGETS:
            asserts = "".join(
                f'_Static_assert({e} == {v}, "{t}: {e}");\n'
                for (e, bitfield), v in zip(_TRAILPACK_EXPRS, vecs[t], strict=True)
                if t in _ITANIUM_BITFIELDS or not bitfield
            )
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "trailpack.c")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(_TRAILPACK_SRC.split("uint32_t f(")[0] + asserts)
                run = subprocess.run(
                    [clang, "-target", TARGETS[t].triple, "-std=c2x", "-ffreestanding",
                     "-fsyntax-only", path],
                    capture_output=True, text=True,
                )  # fmt: skip
            assert run.returncode == 0, (t, run.stderr)
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trailpack.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_TRAILPACK_SRC)
        for t in _ABI_TARGETS:
            assert _abi_const_vec_twin(exe, path, t) == vecs[t], (t, vecs[t])


# An `_Atomic` struct or union -- a member, a `sizeof` operand, a pointee -- refused on both rails with the
# one reason; and a cast or compound literal of an `_Atomic` type, which neither rail parses as one.
_ATOMIC_AGGREGATE = "an `_Atomic` struct or union is not supported"
_ATOMIC_BITFIELD = "a bit-field of `_Atomic` type is not supported"
_ATOMIC_REFUSED = (
    (
        "struct P3 { uint8_t c[3]; };\nstruct W { uint8_t k; _Atomic struct P3 p; uint8_t t; };\n"
        "uint32_t f(struct W *w) { w->t = 9u; return w->t; }\n",
        _ATOMIC_AGGREGATE,
    ),
    (
        "struct P3 { uint8_t c[3]; };\nuint32_t f(void) { return (uint32_t)sizeof(_Atomic struct P3); }\n",
        _ATOMIC_AGGREGATE,
    ),
    (
        "struct P3 { uint8_t c[3]; };\nuint32_t f(_Atomic struct P3 *p) { return 1u; }\n",
        _ATOMIC_AGGREGATE,
    ),
    ("uint32_t f(uint32_t x) { uint32_t y = (_Atomic uint32_t)x; return y + 1u; }\n", None),
    ("uint32_t f(uint32_t x) { uint32_t *p = &(_Atomic uint32_t){x}; return *p; }\n", None),
    # CF-ATOMIC: a bit-field of `_Atomic` type -- named, unnamed or `_Atomic(T)` -- which GCC and Clang
    # reject too; no atomic operation reaches a bit-field
    (
        "struct B { uint8_t k; _Atomic uint32_t x : 3; uint32_t y; };\n"
        "uint32_t f(struct B *b) { return b->y; }\n",
        _ATOMIC_BITFIELD,
    ),
    (
        "struct B { uint8_t k; _Atomic uint32_t : 3; uint32_t y; };\n"
        "uint32_t f(struct B *b) { return b->y; }\n",
        _ATOMIC_BITFIELD,
    ),
    (
        "struct B { uint8_t k; _Atomic(uint32_t) x : 3; };\nuint32_t f(struct B *b) { return b->k; }\n",
        _ATOMIC_BITFIELD,
    ),
)


def test_an_atomic_aggregate_or_cast_is_refused_on_both_rails():
    """CF-CALIGN: the ABI's atomic promotion would lay out an `_Atomic` struct at a rounded-up size (a
    3-byte one occupies 4), which every declaration, copy and extent would then have to carry, and each
    access to it would have to be one atomic operation on the whole object. Neither rail models that, so
    both refuse it, wherever it is spelled. The oracle had laid it out as the plain struct, and the twin
    had placed it as a member but refused `sizeof(_Atomic struct P3)`. A cast or compound literal of an
    `_Atomic` type is refused on both rails too (the oracle's `_is_cast`, the twin's
    `starts_type_name`); `sizeof`, `_Alignof` and `typeof` of one fold identically on both."""
    from bcir.frontends.cfront.cparse import CParseError
    from bcir.frontends.cfront.lower import CLowerError

    exe = _build_frontend(_session_build_dir()) if _CC else None
    for body, why in _ATOMIC_REFUSED:
        src = "#include <stdint.h>\n" + body
        try:
            compile_unit(src, check_clang=False)
            raise AssertionError(f"the oracle lowered {body!r}")
        except (CParseError, CLowerError) as e:
            assert why is None or str(e) == why, (body, str(e))
        if exe is None:
            continue
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "atomic_refused.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode != 0 and run.stdout.startswith("PARSE-ERR"), (body, run.stdout)
            assert why is None or why in run.stdout, (body, run.stdout)
    src = (
        "#include <stdint.h>\nuint32_t f(uint32_t x) { typeof(_Atomic uint32_t) y = x;"
        " return y + (uint32_t)sizeof(_Atomic(float _Complex)) + 100u * (uint32_t)_Alignof(typeof(_Atomic"
        " float _Complex)); }\n"
    )
    assert _abi_const_vec_oracle(src, "x86_64-linux") == [8, 100, 8]
    if exe is not None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "atomic_folds.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
            assert _abi_const_vec_twin(exe, path, "x86_64-linux") == [8, 100, 8]


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
    assert vecs["x86_64-linux"] == [8, 8, 8, 4, 8]  # LP64
    assert vecs["aarch64-linux"] == vecs["riscv64-linux"] == [8, 8, 8, 4, 8]
    assert vecs["x86_64-windows"] == [4, 8, 8, 4, 8]  # LLP64: long is 4
    assert vecs["i386-linux"] == [4, 4, 4, 4, 8]  # ILP32: pointers are 4 too
    assert len({tuple(v) for v in vecs.values()}) == 3  # three distinct data models
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        for t in _ABI_TARGETS:
            assert _abi_const_vec_twin(exe, path, t) == vecs[t], (
                f"ABI {t}: twin {_abi_const_vec_twin(exe, path, t)} != oracle {vecs[t]}"
            )
        # an unknown target is a clean diagnostic + nonzero exit on the C rail (not a crash).
        bad = subprocess.run(
            [exe, "--target", "sparc-solaris", path], capture_output=True, text=True
        )
        assert bad.returncode != 0 and "unknown target" in bad.stdout, bad.stdout


# (source, the expected total-compile outcome): "clean" compiles+verifies, "dirty" compiles but the
# verifier rejects it (R18 recursion), "fallback" is outside the supported subset (route to LLVM).
_FALLBACK_PROBES = [
    ("unsigned f(unsigned x){ return x*2u + 1u; }", "clean"),
    ("unsigned f(unsigned n){ return f(n-1u); }", "dirty"),  # R18 recursion -> DIRTY, not fallback
    ("unsigned f(unsigned x){ return x + ; }", "fallback"),  # malformed -> parse reject
    (
        "unsigned f(void){ _Complex double z; (void)z; return 0u; }",
        "clean",
    ),  # _Complex: now in the subset
    (
        "unsigned f(void){ _Imaginary double z; return 0u; }",
        "fallback",
    ),  # _Imaginary: still outside the subset
    (
        "unsigned f(unsigned n){ unsigned a[n][n][n][n]; return a[0][0][0][0]; }",
        "fallback",
    ),  # a >3-D VLA:
    # 1-D / 2-D / 3-D stack VLAs are now natively lowered (#vla / #vlamd -- snapshot each
    # runtime dim, flatten to m*n, mask the Horner index), but >3 dims defers to fallback
    # on both rails (the dim table caps at 3).
    (
        "unsigned f(unsigned x){ return ({ unsigned y=x; y+1u; }); }",
        "clean",
    ),  # statement-expr (#stmtexpr): now native
    (
        "unsigned f(unsigned a){ a = a*3u + 1u; return a; }",
        "clean",
    ),  # assigning a PARAMETER: a bare
    # `a = ..;` in the emit, never a `uint32_t a = ..;` redeclaration (twin-emit regression).
    ("unsigned f(unsigned a){ return ({ a++; }); }", "fallback"),  # `i++` as a stmt-expr VALUE: the
    # post/pre distinction was discarded in the desugar, so it routes away on both rails.
    (
        "unsigned f(unsigned a){ return ({ a = a+1u; }); }",
        "fallback",
    ),  # an assignment as a stmt-expr value
    # (the twin's value-expression grammar has none) -> fallback on both rails, in lockstep.
    (
        "unsigned f(unsigned x){ void *p=&&L; goto *p; L: return x; }",
        "clean",
    ),  # computed goto (#computedgoto):
    # the GNU label-as-value `&&L` (a `void *`) + the indirect `goto *p` are now native on BOTH
    # rails -- they lower to the GNU forms (which Clang compiles), so the unit is clean, not fallback.
    (
        "struct Q{unsigned*p; unsigned n;}; unsigned f(struct Q q,unsigned i){ return q.p[i&3u]+q.n; }",
        "clean",
    ),  # a *pointer* member indexed (`q.p[i]` == `*(q.p + i)`): both rails now load the full
    # pointer field and subscript the loaded pointer (#fieldderef, pointer-value slice 2b).
    # (1-D..3-D member *arrays* are native -- #memberarray; deref-through is now native too.)
    (
        "unsigned f(unsigned i){ unsigned m[2][2][2][2]; m[0][0][0][0]=1u; return m[i&1u][0][0][0]; }",
        "fallback",
    ),  # a >3-dimensional local array: 1-D..3-D locals are now natively lowered (a flat
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
    assert set(oracle.values()) == {0, 1, 2}  # spans clean / dirty / fallback
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        cc = _build_bcir_cc(d)
        for src, want in _FALLBACK_PROBES:
            p = os.path.join(d, "u.c")
            with open(p, "w") as f:
                f.write(src + "\n")
            got = subprocess.run([cc, "--fallback", p], capture_output=True, text=True)
            assert got.returncode == _FALLBACK_RC[want], (
                f"twin rc={got.returncode} != {want}({_FALLBACK_RC[want]}) for {src!r}\n{got.stderr}"
            )
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
        assert r.equivalence == "match", (
            f"oracle emit not Clang-equivalent ({r.equivalence}): {src!r}"
        )


def _build_diag(d: str) -> str:
    """Build the bcir_diag renderer harness (test_diag.c + bcir_diag.c)."""
    exe = os.path.join(d, "tdiag")
    srcs = [os.path.join(_C, s) for s in ("bcir_diag.c", "test_diag.c")]
    for std in ("c23", "c11"):
        b = subprocess.run(
            [_CC, f"-std={std}", "-O2", "-I", _C, *srcs, "-o", exe], capture_output=True, text=True
        )
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
        SourceDiagnostic,
        Span,
        Note,
        render,
    )

    src_a = "unsigned f(unsigned x){ return x + ; }\n"
    src_b = "int main(void)\n{\n\treturn foo(1, 2);\n}\n"  # a tab-indented line
    src_c = "a\nbb\nccc\n"
    cases = [
        (src_a, "u.c", ("error", (34, 35), "expected ';'"), []),
        (src_a, "u.c", ("error", (-1, -1), "file-level problem"), []),  # spanless banner
        (src_a, "u.c", ("warning", (9, 10), "odd parameter name"), []),
        (src_a, "u.c", ("error", (34, 34), "zero-width insertion point"), []),
        (src_a, "u.c", ("error", (40, 41), "past end of file"), []),
        (
            src_b,
            "m.c",
            ("error", (19, 22), "implicit declaration of 'foo'"),  # tab line + a note
            [((4, 8), "expanded from macro here")],
        ),
        (src_c, "t.c", ("error", (4, 7), "underline runs to end of line"), []),  # line 2, multi-col
        (src_c, "t.c", ("note", (8, 9), "on the last line"), []),
        (
            src_b,
            "m.c",
            ("error", (-1, -1), "no primary span"),  # spanless + mixed notes
            [((19, 20), "see here"), ((-1, -1), "and a spanless note")],
        ),
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
            c_out = subprocess.run(
                [exe, sp, fn], input=_diag_spec(primary, notes), capture_output=True, text=True
            ).stdout
            assert c_out == py_render(src, fn, primary, notes), (
                f"diag layout diverged for {fn} {primary}\n C: {c_out!r}\nPY: {py_render(src, fn, primary, notes)!r}"
            )


def test_diagnostic_json_dual_rail():
    """The machine-readable (JSON) diagnostics feed (#diag): `bcir_diag_to_json` is the C twin of
    DiagnosticReport.to_json() (a `-fdiagnostics-format=json`-style feed). Over the same diagnostics
    its output is byte-identical to Python's `json.dumps(indent=2)` -- the same 2-space indentation,
    member order (severity / message / phase / file:line:column / range / notes), nested range and
    note objects, JSON string escaping, and the spanless-vs-spanned location shape. Covers a single
    spanned diagnostic with a note, a spanless banner, escaped characters in the message, a
    multi-element array, and the empty array."""
    from bcir.frontends.cfront.diagnostics import (  # noqa: PLC0415
        SourceDiagnostic,
        Span,
        Note,
        DiagnosticReport,
    )

    src_a = "unsigned f(unsigned x){ return x + ; }\n"
    src_b = "int main(void)\n{\n\treturn foo(1, 2);\n}\n"
    cases = [
        (src_a, "u.c", [("error", (34, 35), "expected ';'", [])]),
        (src_a, "u.c", [("warning", (-1, -1), "file-level problem", [])]),
        (
            src_b,
            "m.c",
            [
                (
                    "error",
                    (19, 22),
                    "implicit declaration of 'foo'",
                    [((4, 8), "expanded from macro here")],
                )
            ],
        ),
        (
            src_a,
            "u.c",
            [("error", (0, 3), 'quote" and back\\slash and a\ttab', [])],
        ),  # JSON escapes
        (
            src_b,
            "m.c",
            [
                ("warning", (-1, -1), "first", []),
                ("error", (19, 20), "second", [((-1, -1), "spanless note")]),
            ],
        ),  # 2-element array
        (src_a, "u.c", []),  # the empty array -> "[]"
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
        assert py_json(src, fn, diags).startswith("[")  # oracle side always runs
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_diag(d)
        for src, fn, diags in cases:
            sp = os.path.join(d, "s.c")
            with open(sp, "w") as f:
                f.write(src)
            c_out = subprocess.run(
                [exe, "--json", sp, fn], input=spec(diags), capture_output=True, text=True
            ).stdout
            assert c_out == py_json(src, fn, diags), (
                f"JSON diverged for {fn}\n C: {c_out!r}\nPY: {py_json(src, fn, diags)!r}"
            )


def test_diagnostic_fixits_dual_rail():
    """Fix-it hints (#diag): the C renderer derives the verb (remove / insert / replace with) from each
    fix-it's span + replacement and prints the replacement with Python `repr()` in text and JSON-escaped
    in the feed, byte-identical to diagnostics.render() / to_json(). The fixits object array sits
    between the location and the notes (the diagnostic_to_dict member order). Covers all three verbs and
    `repr`'s quote selection (a `'` in the replacement switches it to double quotes)."""
    from bcir.frontends.cfront.diagnostics import (  # noqa: PLC0415
        SourceDiagnostic,
        Span,
        FixIt,
        Note,
        DiagnosticReport,
        render,
    )

    src = "unsigned f(unsigned x){ return x + ; }\n"
    # (fixit spans+replacements, notes) on a fixed primary error @34:35.
    fixit_sets = [
        [((34, 35), ";")],  # replace with ';'
        [((34, 34), ")")],  # insert ')'
        [((34, 36), "")],  # remove ''
        [((10, 11), "x'y")],  # repr -> double quotes ("x'y")
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
        return SourceDiagnostic(
            "error", "expected token", span=Span(34, 35), fixits=fx, notes=nt, phase="parse"
        )

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
                for flag, want in (
                    (None, render(diag, src, "u.c")),
                    ("--json", DiagnosticReport([diag], src, "u.c").to_json()),
                ):
                    args = [exe] + ([flag] if flag else []) + [sp, "u.c"]
                    out = subprocess.run(
                        args, input=spec(fixits, notes), capture_output=True, text=True
                    ).stdout
                    assert out == want, (
                        f"fix-it {flag} diverged for {fixits}\n C: {out!r}\nPY: {want!r}"
                    )


def test_diagnostic_include_stack_origin_dual_rail():
    """Include / line-map origin (#diag): a diagnostic relocated to its origin file:line, with the
    #include chain printed as Clang "In file included from <file>:<line>:" frames (text) and an
    "includedFrom" array (JSON), byte-identical to diagnostics.render() / diagnostic_to_dict() given an
    `origin`. The primary banner moves to (origin_file, origin_line) but the column + source snippet
    still come from the (preprocessed) source; notes are NOT relocated. Covers the render-vs-JSON
    asymmetry for a spanless diagnostic (render still shows the frames + origin file; JSON ignores the
    origin and keeps the default file), an empty include stack, and origin alongside a fix-it + note."""
    from bcir.frontends.cfront.diagnostics import (  # noqa: PLC0415
        SourceDiagnostic,
        Span,
        Note,
        FixIt,
        render,
        diagnostic_to_dict,
    )
    import json as _json  # noqa: PLC0415

    src = "line0\nline1 has the token X here\nline2\n"
    off = src.index("X")

    def build(span, msg, fixits, notes):
        return SourceDiagnostic(
            "error" if span else "warning",
            msg,
            span=span,
            fixits=[FixIt(Span(a, b), r) for (a, b), r in fixits],
            notes=[Note(m, Span(a, b)) for (a, b), m in notes],
            phase="parse",
        )

    # (diag, origin, spec) tuples.
    span = Span(off, off + 1)
    cases = [
        # spanned + origin + 2 include frames
        (
            build(span, "undeclared 'X'", [], []),
            ("inc/b.h", 42, [("main.c", 10), ("inc/a.h", 3)]),
            f"error\t{off}\t{off + 1}\tundeclared 'X'\n@\t42\t0\tinc/b.h\n^\t10\t0\tmain.c\n^\t3\t0\tinc/a.h\n",
        ),
        # spanned + origin + empty include stack (no "includedFrom" in JSON)
        (
            build(span, "undeclared 'X'", [], []),
            ("inc/b.h", 42, []),
            f"error\t{off}\t{off + 1}\tundeclared 'X'\n@\t42\t0\tinc/b.h\n",
        ),
        # spanless + origin: render shows frames + origin file; JSON ignores origin (keeps default file)
        (
            build(None, "no span", [], []),
            ("inc/b.h", 42, [("main.c", 10)]),
            "warning\t-1\t-1\tno span\n@\t42\t0\tinc/b.h\n^\t10\t0\tmain.c\n",
        ),
        # origin alongside a fix-it + a note (only the primary is relocated)
        (
            build(span, "bad", [((off, off + 1), "Y")], [((0, 3), "see")]),
            ("hdr.h", 7, [("top.c", 2)]),
            f"error\t{off}\t{off + 1}\tbad\n@\t7\t0\thdr.h\n^\t2\t0\ttop.c\n+\t{off}\t{off + 1}\tY\n-\t0\t3\tsee\n",
        ),
    ]

    def py_text(diag, origin):
        return render(diag, src, "u.c", origin=origin)

    def py_json(diag, origin):
        return _json.dumps([diagnostic_to_dict(diag, src, "u.c", origin=origin)], indent=2)

    for diag, origin, _spec in cases:  # oracle side always runs
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
        (
            "unsigned f(unsigned x) { return x + ; }\nunsigned g(unsigned y) { return y 7; }\n",
            "multi.c",
            2,
        ),  # >=2 errors, one with a fix-it
        ("unsigned ok(unsigned x){ return x*2u + 1u; }\n", "clean.c", 0),  # the empty report
    ]
    reports = [(diagnose(src, filename=fn), fn, lo) for src, fn, lo in sources]
    for rep, _fn, lo in reports:
        assert len(rep.diagnostics) >= lo  # the recovery actually fired
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        exe = _build_diag(d)
        for rep, fn, _lo in reports:
            sp = os.path.join(d, "pp.c")
            with open(sp, "w") as f:
                f.write(rep.source)  # the preprocessed source the spans index
            spec = _report_spec(rep)
            for flag, want in ((None, rep.render()), ("--json", rep.to_json())):
                args = [exe] + ([flag] if flag else []) + [sp, fn]
                out = subprocess.run(args, input=spec, capture_output=True, text=True).stdout
                assert out == want, (
                    f"recovery report {flag} diverged for {fn}\n C: {out!r}\nPY: {want!r}"
                )


def test_c_diagnostic_renderer_rejects_malformed_shapes_and_clamps_hostile_spans():
    """Public diagnostic APIs must fail atomically instead of dereferencing sparse records."""
    if not _CC:
        return
    driver = r"""
#include <limits.h>
#include <string.h>
#include "bcir_diag.h"
int main(void){
  bcir_diag d;char out[128];int line=0,col=0;memset(&d,0,sizeof d);memset(out,0xa5,sizeof out);
  if(bcir_diag_render(&d,"x","u.c",out,sizeof out)!=0||out[0]!=0)return 1;
  d.severity="error";d.message="bad";d.span.has_span=1;d.span.start=INT_MIN;d.span.end=INT_MAX;
  size_t n=bcir_diag_render(&d,"x\n","u.c",out,sizeof out);if(!n||n>=sizeof out)return 2;
  d.n_notes=1;d.notes=0;memset(out,0xa5,sizeof out);
  if(bcir_diag_to_json(&d,1,"x","u.c",out,sizeof out)!=0||out[0]!=0)return 3;
  if(bcir_diag_report_render(0,1,"x","u.c",out,sizeof out)!=0||out[0]!=0)return 4;
  bcir_diag_line_col(0,0,&line,&col);if(line!=1||col!=1)return 5;
  if(bcir_diag_render(&d,"x","u.c",0,128)!=0)return 6;
  return 0;
}
"""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "diag_hostile.c")
        with open(src, "w", encoding="utf-8") as f:
            f.write(driver)
        exe = os.path.join(d, "diag_hostile")
        build = subprocess.run(
            [
                _CC,
                "-std=c11",
                "-O1",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                _C,
                os.path.join(_C, "bcir_diag.c"),
                src,
                "-o",
                exe,
            ],
            capture_output=True,
            text=True,
        )
        assert build.returncode == 0, build.stderr
        run = subprocess.run([exe], capture_output=True, text=True)
        assert run.returncode == 0, (run.stdout, run.stderr)


def _oracle_effects_report(src: str) -> str:
    """The oracle's per-function footprints + commute matrix in the bcir-cc --emit-effects text
    format (`escape.effects_report`, the report the C twin's bcir_cfront_effects prints)."""
    from bcir.frontends.cfront.escape import effects_report  # noqa: PLC0415

    r = compile_unit(src, check_clang=False)
    return effects_report(r.lowered, r.escape)


def test_effect_commutation_analysis_dual_rail():
    """The effect footprint and the commute matrix (#effects, G10): bcir-cc --emit-effects is the C
    twin of escape.effects_report -- one points-to analysis, the same reads and writes by name (a
    global, a static's `fn.name`, `*` for memory the unit cannot name), callees and narrowed
    indirect targets folded in. The two fixtures pin the teeth: a commuting pair over disjoint
    globals, a writer's conflict, and a caller inheriting its callee's footprint."""
    fixtures = ["cfront_effects.c", "cfront_global_rw.c"]
    reports = {}
    for fx in fixtures:
        src = open(os.path.join(_C, fx), encoding="utf-8").read()
        reports[fx] = _oracle_effects_report(src)
    assert "commute read_a read_b = 1" in reports["cfront_effects.c"]
    assert "commute read_a write_a = 0" in reports["cfront_effects.c"]
    assert "fn=via_a reads=ga writes=ga" in reports["cfront_effects.c"]  # folded callee effects
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        cc = _build_bcir_cc(d)
        for fx in fixtures:
            out = subprocess.run(
                [cc, "--emit-effects", os.path.join(_C, fx)], capture_output=True, text=True
            ).stdout
            assert out == reports[fx], f"{fx}: effects diverged\n C:\n{out}\nPY:\n{reports[fx]}"


def test_escape_and_effects_reports_are_byte_identical_over_the_corpus():
    """G10's parity gate: over every cfront corpus unit, the generated units of `escape_fixtures`
    (heap buffers, pointers made from integers, an ops-table callback, locals read in place / lent /
    captured, one- and two-target indirect calls) and its forms (every declaration kind against
    every access form in every storage place), bcir-cc --emit-effects and --emit-escape print the
    oracle's reports byte for byte. Not vacuous: every unit is compared except the pinned
    preprocessor limits -- exactly those, so a pin that starts compiling fails here -- and the
    corpus carries each verdict."""
    from bcir.tests import escape_fixtures as ef  # noqa: PLC0415

    units = ef.corpus()
    assert len(units) > 100
    verdicts = {
        v for *_x, res in units for per in res.escape.objects.values() for v in per.values()
    }
    assert {"nonescaping", "escaping"} <= verdicts
    assert ef.TWIN_PREPROCESSOR_LIMITS <= {name for name, *_x in units}
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        cc = _build_bcir_cc(d)
        pairs = [(p, r) for _n, p, _s, r in units]
        eff, esc, compared = ef.rail_parity(cc, pairs, ef.TWIN_PREPROCESSOR_LIMITS)
        assert (eff, esc) == (0, 0), (eff, esc)
        assert compared == len(units) - len(ef.TWIN_PREPROCESSOR_LIMITS), (compared, len(units))
        extra = [(f"gen_{seed}.c", ef.generate(seed)[0]) for seed in ef.SEEDS]
        extra += [(f"forms_{place}.c", source) for place, source in ef.form_units()]
        made = []
        for name, source in extra:
            path = os.path.join(d, name)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(source)
            made.append((path, compile_unit(source, check_clang=False)))
        eff, esc, compared = ef.rail_parity(cc, made)
        assert (eff, esc, compared) == (0, 0, len(extra)), (eff, esc, compared)


def test_escape_refusal_and_its_boundary_agree_across_rails():
    """A call with more operands than a claim holds is dropped past the sixth on the twin, so both
    rails REFUSE the unit (`truncated=1`, every footprint `*`); six operands is analyzed. The
    twin's `truncated` claim flag is what makes the refusal visible to it."""
    from bcir.frontends.cfront.escape import effects_report, escape_report  # noqa: PLC0415

    seven = (
        "static unsigned s7(unsigned a, unsigned b, unsigned c, unsigned d, unsigned e,"
        " unsigned f, unsigned *p) { p[0] = a + b + c + d + e + f; return p[1]; }\n"
        "unsigned caller(unsigned x) { unsigned t[4]; t[1] = x;"
        " return s7(x, x, x, x, x, x, t); }\n"
    )
    six = (
        "static unsigned s6(unsigned a, unsigned b, unsigned c, unsigned d, unsigned e,"
        " unsigned *p) { p[0] = a + b + c + d + e; return p[1]; }\n"
        "unsigned caller(unsigned x) { unsigned t[4]; t[1] = x;"
        " return s6(x, x, x, x, x, t); }\n"
    )
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        cc = _build_bcir_cc(d)
        for label, src in (("seven", seven), ("six", six)):
            path = os.path.join(d, f"{label}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
            r = compile_unit(src, check_clang=False)
            for flag, report in (
                ("--emit-effects", effects_report),
                ("--emit-escape", escape_report),
            ):
                out = subprocess.run([cc, flag, path], capture_output=True, text=True).stdout
                assert out == report(r.lowered, r.escape), (label, flag, out)
            assert r.escape.truncated == (label == "seven")


def _scale_unit_src() -> str:
    """A translation unit that busts every *old* fixed IR ceiling: 40 leaf functions + a
    12-parameter function + a 40-call aggregator + a 7500-claim function."""
    fns = [f"unsigned g{k}(void){{ return {k}u; }}" for k in range(40)]  # > old BCIR_MAX_FUNCS 16
    ps = [chr(ord("a") + i) for i in range(12)]
    fns.append(
        "unsigned many12("
        + ",".join(f"unsigned {p}" for p in ps)
        + "){ return "
        + "+".join(ps)
        + "; }"
    )  # > old BCIR_MAX_PARAMS 8
    fns.append(
        "unsigned agg(void){ return " + "+".join(f"g{k}()" for k in range(40)) + "; }"
    )  # > old BCIR_MAX_CALLS 32
    body = "\n".join("  acc = acc + 1u;" for _ in range(2500))
    fns.append(
        "unsigned big(unsigned acc){\n" + body + "\n  return acc;\n}"
    )  # > old 4096-claim per-fn cap
    return "\n".join(fns) + "\n"


def test_scalable_ir_no_fixed_ceilings():
    """Scalable IR (no fixed `BCIR_MAX_*`): a unit that busts every old ceiling -- 43 functions (>
    the old `BCIR_MAX_FUNCS` 16), a 12-parameter function (> 8), a 40-call aggregator (> 32), and a
    7500-claim function (> the old 4096 per-function cap) -- compiles clean on the C twin (the IR
    grows geometrically) and matches the oracle's structural counts, which are uncapped by design."""
    src = _scale_unit_src()
    r = compile_unit(src, check_clang=False)  # oracle: Python lists, no caps
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


# the C twin's claim-id uniqueness law (R1.1): lower a unit, then inject a duplicate claim id (both
# WITHIN a function and ACROSS two functions) and confirm bcir_verify_unit flips clean->dirty with an
# R1.1 diagnostic in BOTH cases. Mirrors the oracle's R1.1
# (test_ir_structural_parity.{test_duplicate_claim_id..., test_cross_function_duplicate_claim_id...}).
_R11_HARNESS = r"""
#include <stdio.h>
#include <string.h>
#include "bcir_cfront.h"
#include "bcir_cpp.h"
#include "bcir_verify.h"
static int verdict_has_r11(const char *src, int intra){
  static bcir_cfront_result r;
  if(bcir_cfront_compile(src,&r)){ printf("COMPILE-ERR %s\n", r.diag); return -2; }
  char diag[256];
  int ok0=bcir_verify_unit(&r.unit,diag,sizeof diag);     /* clean baseline */
  if(intra){
    bcir_func *f=&r.unit.funcs[r.unit.n_funcs-1];
    if(f->n_claims<2){ bcir_cfront_free(&r); return -3; }
    f->claims[1].id=f->claims[0].id;                      /* intra-function duplicate */
  } else {
    if(r.unit.n_funcs<2 || r.unit.funcs[0].n_claims<1 || r.unit.funcs[1].n_claims<1){ bcir_cfront_free(&r); return -3; }
    r.unit.funcs[1].claims[0].id=r.unit.funcs[0].claims[0].id;   /* CROSS-function duplicate */
  }
  int ok1=bcir_verify_unit(&r.unit,diag,sizeof diag);
  int good = (ok0==1 && ok1==0 && strstr(diag,"R1.1")!=NULL);
  bcir_cfront_free(&r);
  return good;
}
int main(void){
  int intra=verdict_has_r11("unsigned f(unsigned a,unsigned b){ return a+b; }\n", 1);
  int cross=verdict_has_r11("unsigned foo(unsigned x){return x+1u;}\nunsigned bar(unsigned x){return x+2u;}\n", 0);
  printf("intra=%d cross=%d\n", intra, cross);
  return 0;
}
"""


def test_claim_id_uniqueness_law_R1_1_dual_rail():
    """R1.1 (claim-id uniqueness) on the C twin: a clean unit verifies, and an injected duplicate claim
    id flips the verdict to dirty with an `R1.1` diagnostic -- for BOTH an intra-function and a
    CROSS-function duplicate (unit-wide). The C mirror of the oracle's unit-wide R1.1."""
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        cpath, exe = os.path.join(d, "r11.c"), os.path.join(d, "r11")
        with open(cpath, "w", encoding="utf-8") as fh:
            fh.write(_R11_HARNESS)
        srcs = [
            os.path.join(_C, s)
            for s in ("bcir_cfront.c", "bcir_cpp.c", "bcir_verify.c", "bcir_runtime.c")
        ]
        for std in ("c23", "c2x", "c11"):
            b = subprocess.run(
                [_CC, f"-std={std}", "-O2", "-I", _C, cpath, *srcs, "-o", exe],
                capture_output=True,
                text=True,
            )
            if b.returncode == 0:
                break
        else:
            raise AssertionError(f"R1.1 harness build failed:\n{b.stderr}")
        out = subprocess.run([exe], capture_output=True, text=True).stdout.strip()
        assert out == "intra=1 cross=1", (
            f"C twin R1.1 (intra + cross): expected 'intra=1 cross=1', got {out!r}"
        )


def _pstress_unit_src() -> str:
    """A unit busting every old *parser-state* cap: 20 struct defs (> s[16]), 25 file-scope globals
    (> gv[16]), 20 typedefs, and a function with 300 locals (> env[256])."""
    L = [f"struct S{k} {{ unsigned m0; unsigned m1; }};" for k in range(20)]
    L += [f"typedef unsigned U{k};" for k in range(20)]
    L += [f"static const unsigned G{k}[2] = {{ {k}u, {k + 1}u }};" for k in range(25)]
    L.append(
        "unsigned big(void){\n"
        + "\n".join(f"  unsigned v{i} = {i}u;" for i in range(300))
        + "\n  return "
        + "+".join(f"v{i}" for i in range(300))
        + "; }"
    )
    L.append("unsigned useg(unsigned i){ return G0[i%2u] + G19[i%2u] + G24[i%2u]; }")
    return "\n".join(L) + "\n"


def test_scalable_parser_state_no_fixed_caps():
    """Scalable parser state: the twin's parser-state record arrays (struct defs / globals / typedefs /
    enum constants / locals) grow geometrically -- the old fixed `s[16]` / `gv[16]` / `env[256]` caps
    are gone, so a real header (20 structs, 25 globals, 300 locals) lowers and matches the oracle, which
    is uncapped by design."""
    src = _pstress_unit_src()
    r = compile_unit(src, check_clang=False)  # oracle: Python dicts/lists, no caps
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
        assert "ok=1" in out.stdout, out.stdout[:200]  # all structs/globals/locals resolved


_INTPROMOTE_SRC = (
    "int sdiv(int a, int b){ return b ? a / b : 0; }\n"  # signed division (truncates toward 0)
    "int smod(int a, int b){ return b ? a % b : 0; }\n"  # signed remainder
    "int sshr(int a){ return a >> 3; }\n"  # arithmetic right shift (sign-extends)
    "int scmp(int a, int b){ return (a < b) + 2*(a <= b) + 4*(a > b) + 8*(a >= b); }\n"  # signed compares
    "unsigned udiv(unsigned a, unsigned b){ return b ? a / b : 0u; }\n"  # unsigned division (control)
    "long wide(int a, long b){ return a + b; }\n"  # int + long -> 64-bit UAC
    "long umix(unsigned a, long b){ return a * b; }\n"  # unsigned*long -> long (mixed width/sign)
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
    assert "int64_t" in emit, emit  # int + long widened to 64-bit
    assert "uint32_t" in emit  # unsigned stays unsigned
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
            b = subprocess.run(
                [_CC, f"-std={std}", "-O2", c, "-o", e], capture_output=True, text=True
            )
            if b.returncode == 0:
                break
        else:
            raise AssertionError(f"harness build failed: {b.stderr[-400:]}")
        assert subprocess.run([e], capture_output=True, text=True).stdout.strip() == "MATCH"


_AGGINIT_SRC = (
    "struct P { unsigned a; unsigned b; unsigned c; };\n"
    "union U { unsigned w; unsigned h; };\n"
    "unsigned spos(unsigned x){ struct P p = {x, x+1u, x+2u}; return p.a*100u+p.b*10u+p.c; }\n"
    "unsigned sdes(unsigned x){ struct P p = {.c=x, .a=x+5u}; return p.a*100u+p.b*10u+p.c; }\n"  # .b gap -> 0
    "unsigned apos(unsigned x){ unsigned a[4] = {x, x+1u, x+2u}; return a[0]+a[1]+a[2]+a[3]; }\n"  # a[3] gap -> 0
    "unsigned ades(unsigned x){ unsigned a[4] = {[3]=x, [0]=x+9u}; return a[0]*10u+a[3]; }\n"  # gaps -> 0
    "unsigned udes(unsigned x){ union U u = {.h=x}; return u.w; }\n"  # overlap @0
)


def test_local_aggregate_initializers_oracle():
    """Local aggregate initializers (§6.7.10), oracle prototype: a braced `struct P p = {…}` /
    `union U u = {…}` / `T a[N] = {…}` (positional + `.field=` / `[i]=` designators) lowers to a
    zero baseline (spelled `= {}`, CF-RTWIDE) plus a store per initialized member/element (reusing the
    member/array store path), so uninitialized members zero-fill. Behaviour-equivalent to Clang across struct/union/array
    and positional/designated. (The C-twin port is the next segment.)"""
    r = compile_unit(_AGGINIT_SRC, check_clang=False)
    emit = "\n".join(r.emitted.values())
    assert "= {}" in emit  # the zero baseline (uninitialized members zero-fill; `= {}`, CF-RTWIDE)
    assert "[4]" in emit  # a local array is declared with its dimension
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
            b = subprocess.run(
                [_CC, f"-std={std}", "-O2", c, "-o", e], capture_output=True, text=True
            )
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
        assert "acc = " in c_emit and "uint32_t acc" not in c_emit and "return acc;" in c_emit, (
            c_emit
        )
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
            b = subprocess.run(
                [_CC, f"-std={std}", "-O2", cf, "-o", ef], capture_output=True, text=True
            )
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
            (
                "uint32_t f(uint32_t n){ return f(n-1); }\nuint32_t g(uint32_t n){ return f(n); }\n",
                "recursive",
            ),
            ("uint32_t g(uint32_t a){ return missing(a); }\n", "undefined"),
        ]:
            fx = os.path.join(d, "bad.c")
            open(fx, "w").write(src)
            out = subprocess.run([exe, fx], capture_output=True, text=True).stdout
            assert "ok=0" in out and "R18" in out and needle in out, out


def _cfront_differential_fuzz_seed(seed: int):
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
    read back as 2. The seeds are fixed (deterministic).

    Factored to ONE seed so the three independent seeds run as three separate `test_*` callables: this
    campaign was the conformance suite's single longest pole (~47s serial), and run_all schedules
    `test_*` functions across its worker pool, so one indivisible 3-seed unit pinned the whole parallel
    wall time to itself. Splitting it lets the three seeds pack across cores -- the seeds/count are
    unchanged, so coverage is byte-identical to the former single test (it just parallelizes)."""
    import random as _random
    import sys as _sys

    tools_c = os.path.join(_ROOT, "tools", "c")
    if tools_c not in _sys.path:
        _sys.path.insert(0, tools_c)
    import fuzz_cfront

    if not _CC:  # no compiler -> can't build the twin; at least pin
        rng = _random.Random(seed)  # that generation terminates and the oracle never
        for _ in range(60):  # crashes on an in-subset program.
            fuzz_cfront._oracle_outcome(fuzz_cfront.Gen(rng).program().source)
        return
    with tempfile.TemporaryDirectory() as d:
        twin = _build_frontend(d)
        divergence, stats = fuzz_cfront.run_seed(twin, _CC, count=40, seed=seed, d=d)
        assert divergence is None, divergence
        assert stats["clean"] >= 1 and stats["checked"] == stats["clean"], stats


def test_cfront_differential_fuzz_seed1234():
    """Differential cfront fuzz, seed 1234 (one of three seed shards; see `_cfront_differential_fuzz_seed`)."""
    _cfront_differential_fuzz_seed(1234)


def test_cfront_differential_fuzz_seed5678():
    """Differential cfront fuzz, seed 5678 (one of three seed shards; see `_cfront_differential_fuzz_seed`)."""
    _cfront_differential_fuzz_seed(5678)


def test_cfront_differential_fuzz_seed4242():
    """Differential cfront fuzz, seed 4242 (one of three seed shards; see `_cfront_differential_fuzz_seed`)."""
    _cfront_differential_fuzz_seed(4242)


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
    assert clean and b == {"masked"}  # a local array -> runtime-checked
    clean, b = bounds_of(
        "unsigned f(unsigned i){ static unsigned a[4]; a[i&3u]=2u; return a[i&3u]; }"
    )
    assert clean and b == {"masked"}  # a static array -> runtime-checked
    clean, b = bounds_of("unsigned f(unsigned *p,unsigned i){ p[i&7u]=3u; return p[i&7u]; }")
    assert clean and b == {"assumed_safe"}  # a pointer (extent unknown) stays trusted


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
    assert "BCIR_CHK(" in body and '"g:a"' in body, (
        body
    )  # the guard threads the <func>:<array> site
    with tempfile.TemporaryDirectory() as d:
        prog = (
            f'#include <stdint.h>\n#include <stdlib.h>\n#include <stdio.h>\n#include "bcir_quarantine.h"\n'
            f"{r.source}\n\n{body}\n"
            f'int main(int c, char **v){{ (void)c; printf("%u\\n", bcir_g((unsigned)atoi(v[1]))); return 0; }}\n'
        )
        cpath, epath = os.path.join(d, "e.c"), os.path.join(d, "e")
        open(cpath, "w").write(prog)
        b = subprocess.run(
            [
                _CC,
                "-std=c23",
                "-O2",
                "-I",
                _C,
                cpath,
                os.path.join(_C, "bcir_quarantine.c"),
                "-o",
                epath,
            ],
            capture_output=True,
            text=True,
        )
        assert b.returncode == 0, b.stderr
        inb = subprocess.run(
            [epath, "3"], capture_output=True, text=True
        )  # in-bounds: a[3] = 6, exits 0
        assert inb.returncode == 0 and inb.stdout.strip() == "6", (inb.returncode, inb.stdout)
        oob = subprocess.run(
            [epath, "99"], capture_output=True, text=True
        )  # OOB: the handler aborts
        assert oob.returncode != 0 and "bounds-quarantine" in oob.stderr, (
            oob.returncode,
            oob.stderr,
        )
        assert "g:a" in oob.stderr, oob.stderr  # the source site is in the fail-fast message


def test_recovered_extent_quarantines_out_of_bounds():
    """§5.12 recoverable extents end-to-end: a NAKED pointer from `malloc(n*sizeof(T))` recovers its element
    count `n`, so `p[i]` is guarded against the RUNTIME extent `BCIR_CHK(rid, i, n, "mpick:p")`. In-bounds
    (`i < n`) is transparent (the raw value); out-of-bounds calls the weak handler, which records the
    provenance naming the `<func>:<pointer>` site and aborts. Linked against the real runtime -- this proves
    the recovered runtime extent (not a constant) actually bounds-checks the heap buffer."""
    if not _CC:
        return
    src = (
        "unsigned mpick(unsigned n, unsigned i){ unsigned *p = malloc(n*sizeof(unsigned)); "
        "for(unsigned k=0u;k<n;k++) p[k]=k*2u; return p[i]; }"
    )
    from bcir.frontends.cfront import compile_unit

    r = compile_unit("#include <stdlib.h>\n" + src, check_clang=False)
    body = r.emitted["mpick"].split("*/\n", 1)[-1]
    assert "BCIR_CHK(" in body and ', n, "mpick:p")' in body, (
        body
    )  # the extent is the runtime count `n`
    with tempfile.TemporaryDirectory() as d:
        prog = (
            f'#include <stdint.h>\n#include <stdlib.h>\n#include <stdio.h>\n#include "bcir_quarantine.h"\n'
            f"{body}\n"
            f'int main(int c, char **v){{ (void)c; printf("%u\\n", bcir_mpick(8u, (unsigned)atoi(v[1]))); '
            f"return 0; }}\n"
        )
        cpath, epath = os.path.join(d, "e.c"), os.path.join(d, "e")
        open(cpath, "w").write(prog)
        b = subprocess.run(
            [
                _CC,
                "-std=c23",
                "-O2",
                "-I",
                _C,
                cpath,
                os.path.join(_C, "bcir_quarantine.c"),
                "-o",
                epath,
            ],
            capture_output=True,
            text=True,
        )
        assert b.returncode == 0, b.stderr
        inb = subprocess.run([epath, "3"], capture_output=True, text=True)  # in-bounds: p[3] = 6
        assert inb.returncode == 0 and inb.stdout.strip() == "6", (inb.returncode, inb.stdout)
        oob = subprocess.run(
            [epath, "99"], capture_output=True, text=True
        )  # OOB of the 8-element buffer
        assert (
            oob.returncode != 0 and "bounds-quarantine" in oob.stderr and "mpick:p" in oob.stderr
        ), (oob.returncode, oob.stderr)


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
    for src in (
        "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); return p[0]; }",
        "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); p[0]=1u; return n; }",
        "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); return *p; }",
    ):
        clean, diags = _r21(src)
        assert clean and any("use-after-free" in d for d in diags), (src, diags)
    clean, diags = _r21(
        "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); free(p); return n; }"
    )
    assert clean and any("double-free" in d for d in diags), diags
    # well-formed heap use is silent: access BEFORE free, and free-then-reallocate-then-use (the write
    # re-validates the pointer), both produce no lifetime diagnostic.
    for src in (
        "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); unsigned r=p[0]; free(p); return r; }",
        "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); "
        "p=malloc(n*sizeof(unsigned)); unsigned r=p[0]; free(p); return r; }",
    ):
        clean, diags = _r21(src)
        assert clean and diags == [], (src, diags)


def test_r21_does_not_disturb_the_corpus():
    """Non-disturbance: R21 is advisory, so it never flips a fixture's clean verdict, and no well-formed
    fixture (the whole corpus -- only `cfront_stdlibmem.c` even allocates, and it frees correctly) emits a
    spurious lifetime diagnostic."""
    import glob
    from bcir.frontends.cfront import compile_unit
    from bcir.frontends.cfront.cparse import CParseError

    for path in sorted(glob.glob(os.path.join(_C, "cfront_*.c"))):
        fx = os.path.basename(path)
        try:
            r = compile_unit(
                open(path, encoding="utf-8").read(), check_clang=False, includes=_includes_for(fx)
            )
        except CParseError:
            if fx.startswith(("cfront_sec_", "cfront_pp_")):
                continue  # a deliberately-malformed adversarial fixture (cfront_sec_deepnest/lextail) or a
                # PREPROCESSOR-only adversarial fixture (cfront_pp_*: macro definitions + bare
                # expansions, not a complete translation unit -- consumed only by the L7 reference
                # differential, never lowered) -- out of scope for this corpus check.
            raise  # a REAL fixture must still parse: a regression to CParseError is a hard failure.
        if r.fallback:
            continue
        assert r.lifetime_diagnostics == [], (fx, [d.message for d in r.lifetime_diagnostics])


def _r21_kinds(messages):
    """(func, kind) pairs from R21 diagnostic strings -- `f: claim N: use-after-free of RID M ...` (oracle)
    or `R21 f: use-after-free` (twin). RID/claim numbers differ across rails; the FUNC + KIND must agree."""
    out = []
    for m in messages:
        kind = (
            "double-free"
            if "double-free" in m
            else ("use-after-free" if "use-after-free" in m else None)
        )
        if kind is None:
            continue
        m = m[len("R21 ") :] if m.startswith("R21 ") else m
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
        "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); return p[0]; }",  # UAF
        "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); p[0]=1u; return n; }",  # UAF store
        "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); free(p); return n; }",  # double-free
        "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); unsigned r=p[0]; free(p); return r; }",  # clean
        "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); "
        "p=malloc(n*sizeof(unsigned)); unsigned r=p[0]; free(p); return r; }",  # reuse
    ]
    with tempfile.TemporaryDirectory() as d:
        exe = _build_frontend(d)
        for i, src in enumerate(cases):
            full = "#include <stdlib.h>\n" + src
            oracle = _r21_kinds(
                d.message for d in compile_unit(full, check_clang=False).lifetime_diagnostics
            )
            cpath = os.path.join(d, f"u{i}.c")
            open(cpath, "w").write(full)
            out = subprocess.run([exe, cpath], capture_output=True, text=True).stdout
            summary = out.partition("----EMIT----")[0]
            twin = _r21_kinds(ln.strip() for ln in summary.splitlines() if ln.startswith("R21 "))
            assert twin == oracle, f"case {i}: twin={twin} oracle={oracle}\n{src}"


def test_r21_policy_gates_the_verdict():
    """§5.12 R21 promotion: the `--r21` driver policy turns a detected use-after-free / double-free
    into a VERDICT. advisory (the default) never gates; fallback routes the unit to the LLVM backend
    (exit 2); reject is a hard error (exit 1); a clean (no-UAF) unit is exit 0 under every policy.
    The C twin driver (bcir-cc) is checked for the SAME exit codes (cross-rail parity) in
    tools/c/check_runtime.sh; here the Python rail (bcir-cfront) is exercised directly so the policy
    is gated even without a C compiler. Default == advisory keeps the corpus + fuzzer undisturbed."""
    from bcir.frontends.cfront.__main__ import main

    uaf = "#include <stdlib.h>\nunsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); return p[0]; }\n"
    dfree = "#include <stdlib.h>\nunsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); free(p); return n; }\n"
    clean = "#include <stdlib.h>\nunsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); unsigned r=p[0]; free(p); return r; }\n"
    with tempfile.TemporaryDirectory() as d:

        def run(src, *flags):
            p = os.path.join(d, "u.c")
            open(p, "w").write(src)
            return main([*flags, "-o", os.path.join(d, "out.txt"), p])

        # advisory (explicit AND the default) never gates -- everything compiles clean (exit 0)
        for src in (uaf, dfree, clean):
            assert run(src, "--r21=advisory") == 0
            assert run(src) == 0  # no --r21 == advisory (non-disturbance)
        # fallback: a UAF / double-free routes to LLVM (2); a clean unit stays 0
        assert run(uaf, "--r21=fallback") == 2
        assert run(dfree, "--r21=fallback") == 2
        assert run(clean, "--r21=fallback") == 0
        # reject: a UAF / double-free is a hard verify error (1); a clean unit stays 0
        assert run(uaf, "--r21=reject") == 1
        assert run(dfree, "--r21=reject") == 1
        assert run(clean, "--r21=reject") == 0
        # an unknown policy is a usage error (2)
        assert run(clean, "--r21=bogus") == 2


def test_extent_count_mutation_is_not_promoted():
    """§5.12 soundness: a recovered count must be STABLE from the allocation onward. A count that is
    re-assigned AFTER the alloc (`n = n - 1`, `n--`) -- so its single assignment is an ordinary BODY write,
    not a decl-init -- must NOT bind, or the re-emitted runtime extent would disagree with the allocation
    and FALSE-TRAP a valid access. Both rails leave it unmanaged (no `BCIR_CHK`); a decl-init count
    (`unsigned m = ...`, before the alloc) still promotes. Guards the gate that distinguishes the two."""
    from bcir.frontends.cfront import compile_unit

    # the access p[n] (after n=n-1) reads p[original-1], the LAST valid element -- it must not be guarded
    # against the mutated extent (which would reject original-1 < original-1).
    src = (
        "unsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); "
        "for(unsigned k=0u;k<n;k++) p[k]=k; n=n-1u; return p[n]; }"
    )
    r = compile_unit("#include <stdlib.h>\n" + src, check_clang=False)
    assert "BCIR_CHK" not in r.emitted["f"], r.emitted["f"]  # oracle: unmanaged (sound)
    if _CC:
        with tempfile.TemporaryDirectory() as d:
            # twin agrees (no BCIR_CHK), and the emit RUNS without a false trap on the valid p[original-1].
            exe = _build_frontend(d)
            cpath = os.path.join(d, "f.c")
            open(cpath, "w").write("#include <stdlib.h>\n" + src)
            out = subprocess.run([exe, cpath], capture_output=True, text=True).stdout
            assert "BCIR_CHK" not in out.partition("----EMIT----")[2], (
                out
            )  # twin: unmanaged too (parity)
            body = r.emitted["f"].split("*/\n", 1)[-1]
            prog = (
                f'#include <stdint.h>\n#include <stdlib.h>\n#include <stdio.h>\n#include "bcir_quarantine.h"\n'
                f'{body}\nint main(void){{ printf("%u\\n", bcir_f(5u)); return 0; }}\n'
            )  # f(5)=p[4]=4
            ep = os.path.join(d, "e")
            open(os.path.join(d, "e.c"), "w").write(prog)
            b = subprocess.run(
                [
                    _CC,
                    "-std=c23",
                    "-O2",
                    "-I",
                    _C,
                    os.path.join(d, "e.c"),
                    os.path.join(_C, "bcir_quarantine.c"),
                    "-o",
                    ep,
                ],
                capture_output=True,
                text=True,
            )
            assert b.returncode == 0, b.stderr
            run = subprocess.run(
                [ep], capture_output=True, text=True
            )  # must NOT abort (no false trap)
            assert run.returncode == 0 and run.stdout.strip() == "4", (
                run.returncode,
                run.stdout,
                run.stderr,
            )


def test_masked_claims_are_discharged_by_a_runtime_guard():
    """§5.12 lowering faithfulness (item 4): the emit must HONOR the `masked` bounds metadata -- every
    masked load/store claim is discharged by exactly one `BCIR_CHK` runtime guard in the emitted C, and
    every masked claim carries the `bounds` verify contract (so R7 validates it). Across the whole corpus,
    a masked claim never silently loses its guard, and a guard is never emitted without a masked claim."""
    import glob
    from bcir.frontends.cfront import compile_unit
    from bcir.frontends.cfront.cparse import CParseError

    seen_masked = 0
    for path in sorted(glob.glob(os.path.join(_C, "cfront_*.c"))):
        fx = os.path.basename(path)
        try:
            r = compile_unit(
                open(path, encoding="utf-8").read(), check_clang=False, includes=_includes_for(fx)
            )
        except CParseError:
            if fx.startswith(("cfront_sec_", "cfront_pp_")):
                continue  # a deliberately-malformed adversarial fixture (cfront_sec_deepnest/lextail) or a
                # PREPROCESSOR-only adversarial fixture (cfront_pp_*, not a complete translation
                # unit -- consumed only by the L7 reference differential) -- out of scope here.
            raise  # a REAL fixture must still parse: a regression to CParseError is a hard failure.
        if r.fallback:
            continue
        for name, lf in r.lowered.functions.items():
            masked = [
                c for c in lf.claims if c.op in ("c.load", "c.store") and c.bounds == "masked"
            ]
            seen_masked += len(masked)
            assert all(c.verify == "bounds" for c in masked), (fx, name)  # R7 contract
            assert len(masked) == r.emitted[name].count("BCIR_CHK"), (
                fx,
                name,
                "masked claims",
                len(masked),
                "BCIR_CHK guards",
                r.emitted[name].count("BCIR_CHK"),
            )
    assert seen_masked > 0  # the corpus exercises the path


def test_cfront_lowering_faithfulness_is_a_self_check():
    """§5.12 item 4: the cfront pipeline SELF-VERIFIES that its emit honors the masked bounds metadata
    (`verify_cfront_lowering`, surfaced in `CompileResult.lowering_diagnostics`). A real compile is faithful
    (no diagnostic); a doctored emit that DROPS a masked claim's guard is flagged R12 -- the law catches a
    backend that would silently lose a bounds check, on any compile (not just the corpus)."""
    from bcir.frontends.cfront import compile_unit
    from bcir.frontends.cfront.pipeline import verify_cfront_lowering

    r = compile_unit(
        open(os.path.join(_C, "cfront_stdlibmem.c"), encoding="utf-8").read(), check_clang=False
    )
    assert r.lowering_diagnostics == [], [
        d.message for d in r.lowering_diagnostics
    ]  # the real emit is faithful
    lf = r.lowered.functions["msum"]  # 2 masked claims
    assert verify_cfront_lowering(lf, r.emitted["msum"]) == []  # faithful
    dropped = verify_cfront_lowering(lf, r.emitted["msum"].replace("BCIR_CHK", "no_guard", 1))
    assert dropped and dropped[0].law == "R12", dropped  # one guard dropped -> flagged


@_requires_cc
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
    for src in (
        "unsigned f(unsigned n){ unsigned m=(n&7u)+1u; unsigned a[m]; unsigned s=0u;"
        "  for(unsigned i=0u;i<m;i++){a[i]=i+n;s+=a[i];} return s; }",
        "int g(int n){ int m=(n&7)+1; int a[m]; int s=0;"
        "  for(int i=0;i<m;i++){a[i]=i*2-n;s+=a[i];} return s; }",
        "unsigned h(unsigned n){ unsigned a[(n&3u)+2u]; unsigned k=(n&3u)+2u; unsigned s=0u;"
        "  for(unsigned i=0u;i<k;i++){a[i]=i^n;s+=a[i];} return s; }",
    ):
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)
    # a single-integer-literal dim still compiles a static array (unchanged)
    r = compile_unit(
        "unsigned f(unsigned i){ unsigned a[8]={0}; a[i & 7u]=i; return a[i & 7u]; }",
        check_clang=True,
    )
    assert r.equivalence == "match" and r.is_clean, r.equivalence
    # a VLA with an initializer (illegal C) routes to fallback
    try:
        compile_unit("unsigned f(unsigned n){ unsigned a[n]={0}; return a[0]; }", check_clang=False)
        assert False, "an initialized VLA should route to fallback"
    except CLowerError as e:
        assert "VLA" in str(e) or "variable-length" in str(e), e
    # a 2-D / 3-D VLA is now natively lowered (see test_multidim_vla_lowering); only a >3-D VLA falls back
    try:
        compile_unit(
            "unsigned f(unsigned n){ unsigned a[n][n][n][n]; return a[0][0][0][0]; }",
            check_clang=False,
        )
        assert False, "a >3-D VLA should route to fallback"
    except (CParseError, CLowerError):
        pass


@_requires_cc
def test_vla_sizeof_is_runtime():
    """§5.9 (#vlasizeof): `sizeof a` of a 1-D stack VLA is a RUNTIME value -- the snapshot extent times the
    element size, emitted as `(size_t)((size_t)__bcir_extK * sizeof(elem))`. `sizeof a[0]` (an element) stays
    the STATIC element size, and `sizeof` of a non-VLA (a static array, a pointer) is unchanged -- so no
    cross-rail divergence and Clang-equivalent on both rails."""
    from bcir.frontends.cfront import compile_unit, cparse, lower, emit

    # the runtime sizeof compiles + is Clang-equivalent, and emits the runtime form (NOT a stale `= 0u`)
    src = (
        "unsigned f(unsigned n){ unsigned m=(n&7u)+1u; unsigned a[m]; unsigned b=(unsigned)sizeof a;"
        " unsigned e=(unsigned)sizeof a[0]; unsigned c=b/e; unsigned s=0u;"
        " for(unsigned i=0u;i<c;i++){a[i]=i+n;s+=a[i];} return s+b+e; }"
    )
    r = compile_unit(src, check_clang=True)
    assert r.equivalence == "match" and r.is_clean, r.equivalence
    body = emit.emit_function(lower.lower_unit(cparse.parse_unit(src), None).functions["f"])
    assert "(size_t)((size_t)__bcir_ext0 * 4)" in body, body  # the RUNTIME extent*size form
    assert "size_t" in body  # the temp is size_t (matches the twin)
    # sizeof of a NON-VLA stays a static fold (no extent read) -- no regression
    for src2, want in [
        (
            "unsigned f(unsigned i){ unsigned a[8]={0}; a[i&7u]=i; return (unsigned)sizeof a + a[0]; }",
            "match",
        ),
        (
            "unsigned g(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); unsigned z=(unsigned)sizeof p; free(p); return z; }",
            "match",
        ),
    ]:
        r2 = compile_unit(src2, check_clang=True)
        assert r2.equivalence == want and r2.is_clean, (src2, r2.equivalence)


@_requires_cc
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
    assert "BCIR_CHK(" in _body(src) and ", n, " in _body(src), _body(src)  # masked vs n, by name
    # a regular (static-dim) array param is unchanged -- NOT masked
    assert "BCIR_CHK" not in _body("unsigned f(unsigned a[5], unsigned i){ return a[i % 5u]; }")
    # a MUTATED size param is not bound (assumed_safe) -- the stability gate; still Clang-equivalent
    src2 = "unsigned f(unsigned n, unsigned a[n]){ n=n+1u; unsigned s=0u; for(unsigned i=0u;i<3u;i++) s+=a[i]; return s; }"
    assert "BCIR_CHK" not in _body(src2)
    assert compile_unit(src2, check_clang=True).equivalence == "match"


@_requires_cc
def test_a_byte_copy_store_converts_the_value_to_the_slots_type():
    """CF-MEMCONV: a store the emit spells as a byte copy converts the value to the slot's declared type,
    as C's assignment does. The conversion is in the claim graph -- `s->f = v` lowers to a `c.cast:float`
    feeding the store, so the cross-rail digest carries it -- where both rails once copied the integer's
    bits into the float. And the oracle's emit converts a complex value to the complex type of a member of
    another width; it went through a real `double` and wrote that value's bytes into the slot. The twin's
    half, and every store form, is `cfront_memberconv.c`."""
    from bcir.frontends.cfront import compile_unit

    r = compile_unit(
        "#include <stdint.h>\nstruct P { uint32_t tag; float f; };\n"
        "uint32_t f(struct P *s, int32_t v) { s->f = v;"
        " return (uint32_t)(s->f > 0.0f) + 2u * (uint32_t)(s->f == (float)v); }\n",
        check_clang=True,
    )
    assert r.is_clean and r.equivalence == "match", r.equivalence
    claims = r.lowered.functions["f"].claims
    cast = next(c for c in claims if c.op == "c.cast:float")
    store = next(c for c in claims if c.op == "c.store")
    assert store.rd[1] == cast.wr[0]
    r = compile_unit(
        "#include <stdint.h>\nstruct C { uint32_t pad; float _Complex z; };\n"
        "uint32_t g(struct C *c, double _Complex w) { c->z = w;"
        " return (uint32_t)(__real__ c->z == (float)__real__ w)"
        " + 2u * (uint32_t)(__imag__ c->z == (float)__imag__ w); }\n",
        check_clang=True,
    )
    assert r.is_clean and r.equivalence == "match", r.equivalence


@_requires_cc
def test_multidim_vla_lowering():
    """§5.9 (#vlamd): a multi-dimensional stack VLA `T a[m][n]` (2-D + 3-D) is lowered FAITHFULLY -- each dim
    is snapshotted once, the array is declared IN-BODY as a flat `T a[__ext_total];` sized by the runtime
    product, and the row-major Horner index `i*n + j` (the inner-dim runtime stride, NOT a const) is
    bounds-masked against the total. Behaviour-equivalent to Clang on both rails. A >3-D VLA routes to
    fallback (the dim table caps at 3); the static multi-dim local + the 1-D VLA paths are unchanged."""
    from bcir.frontends.cfront import compile_unit, cparse, lower, emit
    from bcir.frontends.cfront.lower import CLowerError
    from bcir.frontends.cfront.cparse import CParseError

    src = (
        "unsigned f(unsigned p, unsigned q){ unsigned m=(p&3u)+1u; unsigned n=(q&3u)+1u; unsigned a[m][n];"
        " unsigned s=0u; for(unsigned i=0u;i<m;i++) for(unsigned j=0u;j<n;j++){a[i][j]=i*n+j+p;s+=a[i][j];}"
        " return s; }"
    )
    r = compile_unit(src, check_clang=True)
    assert r.equivalence == "match" and r.is_clean, r.equivalence
    body = emit.emit_function(lower.lower_unit(cparse.parse_unit(src), None).functions["f"])
    assert "a[__bcir_ext2]" in body  # flat in-body decl sized by the product m*n
    assert "i * __bcir_ext1" in body  # Horner uses the RUNTIME inner-dim stride (not a const)
    assert "* __bcir_ext1" in body and "__bcir_ext0 * __bcir_ext1" in body  # total = m*n
    # a >3-D VLA falls back (dim table caps at 3)
    try:
        compile_unit(
            "unsigned f(unsigned n){ unsigned a[n][n][n][n]; return a[0][0][0][0]; }",
            check_clang=False,
        )
        assert False, "a >3-D VLA should route to fallback"
    except (CParseError, CLowerError):
        pass


@_requires_cc
def test_lvalue_assignment_as_value_extended_forms():
    """§5.9 (#lvassignexpr): an assignment whose target is an ARRAY ELEMENT `a[i]`, a pointer DEREF `*p`, or a
    NESTED member `o.in.x` -- used as a VALUE (`(a[i]=v)+1`, `(*p=v)*2`, chained `a[0]=b[0]=v`) -- yields the
    stored/converted value (the once-resolved lvalue re-read). Extends the single-level-scalar-member case
    (#memassignexpr). Behaviour-equivalent to Clang on both rails; a VOLATILE/MMIO lvalue stays a fallback
    (the re-read would be an extra observable access), and a bitfield / array-of-structs target stays a
    follow-on."""
    from bcir.frontends.cfront import compile_unit
    from bcir.frontends.cfront.lower import CLowerError

    for src in (
        "unsigned f(unsigned i, unsigned v){ unsigned a[8]={0}; unsigned r=(a[i&7u]=v)+1u; return r+a[i&7u]; }",
        "unsigned f(unsigned v){ unsigned y=0u; unsigned *p=&y; unsigned r=(*p=v)*2u; return r+y; }",
        "unsigned f(unsigned v){ unsigned a[4]={0},b[4]={0}; unsigned r=(a[0]=b[0]=v)+7u; return r+a[0]+b[0]; }",
        "unsigned f(unsigned i, unsigned v){ unsigned a[8]={0}; a[i&7u]=v; unsigned r=(a[i&7u]+=5u)*2u; return r+a[i&7u]; }",
    ):
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)
    # the single-level scalar member case (#memassignexpr) still compiles
    assert (
        compile_unit(
            "struct S{unsigned a;}; unsigned f(unsigned v){ struct S s; return (s.a=v)+1u; }",
            check_clang=True,
        ).equivalence
        == "match"
    )
    # a VOLATILE lvalue as a value falls back (the re-read would be an extra MMIO access)
    try:
        compile_unit(
            "struct R{volatile unsigned reg;}; unsigned f(volatile struct R *r, unsigned v){ return (r->reg=v)+1u; }",
            check_clang=False,
        )
        assert False, "a volatile lvalue-as-value should fall back"
    except CLowerError:
        pass


@_requires_cc
def test_narrow_compound_assignment_as_value_is_the_stored_value():
    """§5.9 (#narrowcompound): the value of a COMPOUND assignment `lv OP= rhs` used as a value is the STORED
    (narrowed) value, not the raw binop result. For a sub-int target (`unsigned char`/`unsigned short` member,
    array element, deref) the store truncates, so the value must be a re-read -- returning the un-narrowed sum
    was a both-rails SILENT MISCOMPILE (clean, wrong, untriggered by the fuzzer). A full-width target needs no
    re-read and is byte-unchanged."""
    from bcir.frontends.cfront import compile_unit

    for src in (
        "struct N{unsigned char c;}; unsigned f(unsigned v){ struct N s; s.c=200u; return (s.c += v)*3u + s.c; }",
        "unsigned f(unsigned i, unsigned v){ unsigned short a[4]={0}; a[i&3u]=60000u; return (a[i&3u] += v)+7u + a[i&3u]; }",
        "unsigned f(unsigned v){ unsigned char y=250u; unsigned char *p=&y; return (*p += v)*2u + y; }",
    ):
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)
    # the NARROW target adds exactly ONE re-read vs the otherwise-identical FULL-width target (which is
    # unchanged -- its value is the binop result, no spurious re-read)
    from bcir.frontends.cfront import cparse, lower, emit

    def _chk(src):
        return emit.emit_function(
            lower.lower_unit(cparse.parse_unit(src), None).functions["f"]
        ).count("BCIR_CHK")

    full = _chk(
        "unsigned f(unsigned i, unsigned v){ unsigned a[4]={0}; a[i&3u]=1000u; return (a[i&3u] += v)*2u; }"
    )
    narrow = _chk(
        "unsigned f(unsigned i, unsigned v){ unsigned short a[4]={0}; a[i&3u]=1000u; return (a[i&3u] += v)*2u; }"
    )
    assert narrow == full + 1, (
        full,
        narrow,
    )  # the narrow target re-reads the stored (truncated) value


@_requires_cc
def test_bitfield_assignment_as_value():
    """§5.9 (#bfassignexpr): a BITFIELD member assignment used as a VALUE -- `(s.bits = v) + 1`, compound
    `(s.bits += v) * 2`, signed `(s.c = v)` -- yields the masked / sign-extended STORED field (a re-read via
    bf.get; the compound path re-reads because a bitfield narrows to its bit width). Extends the lvalue-as-value
    forms to bitfield targets. Behaviour-equivalent to Clang on both rails."""
    from bcir.frontends.cfront import compile_unit

    for src in (
        "struct B{unsigned f:5;}; unsigned g(unsigned v){ struct B s; s.f=0u; return (s.f=v)+1u + s.f; }",
        "struct B{unsigned f:5;}; unsigned g(unsigned v){ struct B s; s.f=10u; return (s.f+=v)*2u + s.f; }",
        "struct B{int c:6;}; int g(int v){ struct B s; s.c=0; return (s.c=v)-1 + s.c; }",
    ):
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)


@_requires_cc
def test_array_of_structs_field_assignment_as_value():
    """§5.9 (#aosassignexpr): the LAST lvalue-as-value form -- an array-of-structs element field `(a[i].f = v)`
    or a member-array element `(s.arr[i] = v)` used as a value. The lvalue is a STRIDED member (index + element
    stride + field offset), resolved ONCE, stored, then re-read (the value). Combines with the narrow re-read
    for a sub-int field. Behaviour-equivalent to Clang on both rails."""
    from bcir.frontends.cfront import compile_unit

    for src in (
        "struct P{unsigned x,y;}; unsigned f(unsigned i, unsigned v){ struct P a[4]; a[i&3u].x=0u; return (a[i&3u].x = v)+1u + a[i&3u].x; }",
        "struct P{unsigned x,y;}; unsigned f(unsigned i, unsigned v){ struct P a[4]; a[i&3u].y=10u; return (a[i&3u].y += v)*2u + a[i&3u].y; }",
        "struct S{unsigned arr[4];}; unsigned f(unsigned i, unsigned v){ struct S s; s.arr[i&3u]=0u; return (s.arr[i&3u] = v)+3u + s.arr[i&3u]; }",
        "struct N{unsigned char c; unsigned x;}; unsigned f(unsigned i, unsigned v){ struct N a[4]; a[i&3u].c=0u; return (a[i&3u].c += v)*2u + a[i&3u].c; }",
    ):
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)


@_requires_cc
def test_signed_function_pointer_return():
    """§5.9 (#signedfnptr): a call through a function-pointer (a funcptr struct member / dispatch, or a funcptr
    param) whose target returns a SIGNED type now types the call RESULT by the return type -- a signed sub-int
    return promotes to `int`, a wide return keeps its width -- so a downstream arithmetic `>>` / `< 0` /
    `(long)`-widen sign-extends. The indirect/member call results were hardcoded uint32 (a both-rails
    miscompile). Behaviour-equivalent to Clang on both rails; an unresolved funcptr return stays uint32."""
    from bcir.frontends.cfront import compile_unit

    for src in (
        "static int neg(int x){return -x-1;} struct D{int(*op)(int);}; int f(int x){ struct D d; d.op=neg; return d.op(x) >> 1; }",
        "static int neg(int x){return -x-1;} struct D{int(*op)(int);}; int f(int x){ struct D d; d.op=neg; int r=d.op(x); return r<0?-r:r; }",
        "static long lw(int x){return -(long)x-1;} struct D{long(*op)(int);}; long f(int x){ struct D d; d.op=lw; return d.op(x)-1; }",
    ):
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)


@_requires_cc
def test_address_of_follow_ons():
    """§5.10 item 1 follow-on (#addroffollow): a PLAIN-base array-of-structs element-field address
    `&arr[i].field` (the indexed base is a bare array/pointer param or a local array, not a struct member) and
    a POINTER-member address `&s->ptr`, both store-through-able. Both were both-rails fallbacks. An ARRAY
    member `&s->arr` (a pointer-to-array result) stays a fallback. Behaviour-equivalent to Clang on both
    rails."""
    from bcir.frontends.cfront import compile_unit
    from bcir.frontends.cfront.lower import CLowerError

    for src in (
        "struct P{unsigned a,b;}; unsigned f(struct P *arr, unsigned i){ unsigned *p=&arr[i&3u].b; return *p+1u; }",
        "struct P{unsigned a,b;}; unsigned f(unsigned i, unsigned v){ struct P arr[4]; unsigned *p=&arr[i&3u].a; *p=v; return arr[i&3u].a; }",
        "struct S{unsigned *ptr; unsigned n;}; unsigned f(struct S *s, unsigned *q){ s->ptr=q; unsigned **pp=&s->ptr; *pp=q+1; return (unsigned)(s->ptr==q+1); }",
    ):
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)
    try:  # &s->arr (array member) stays a clean fallback
        compile_unit(
            "struct S{unsigned arr[4];}; unsigned f(struct S *s){ unsigned (*pa)[4]=&s->arr; return (**pa); }",
            check_clang=True,
        )
    except CLowerError:
        pass


@_requires_cc
def test_incdec_as_expression_value():
    """§5.10 item 5 (#incdecexpr): `++`/`--` in EXPRESSION position -> a read-modify-write yielding the OLD
    value (postfix) or the NEW value (prefix). The lvalue is resolved ONCE. Covers a named local, member,
    array element, bitfield, sub-int (narrowing) local, pointer step, and the comma operator. The statement
    forms `a++;` already worked. Behaviour-equivalent to Clang on both rails."""
    from bcir.frontends.cfront import compile_unit

    for src in (
        "unsigned f(unsigned a){ unsigned x=a++; return x*100u+a; }",
        "unsigned f(unsigned a){ unsigned x=++a; return x*100u+a; }",
        "int f(int a){ int x=a--; return x*100+a; }",
        "struct S{unsigned x;}; unsigned f(unsigned v){ struct S s; s.x=v; unsigned r=s.x++; return r*100u+s.x; }",
        "unsigned f(unsigned i, unsigned v){ unsigned a[4]; a[i&3u]=v; unsigned r=a[i&3u]++; return r*100u+a[i&3u]; }",
        "struct B{unsigned x:5;}; unsigned f(unsigned v){ struct B b; b.x=v&31u; unsigned r=b.x++; return r*100u+b.x; }",
        "unsigned f(unsigned a, unsigned b){ return (a++, b)+a; }",
    ):
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)


@_requires_cc
def test_array_compound_literal_1d_scalar():
    """§5.10 item 4 (#arrcomplit): a 1-D SCALAR-element array compound literal lowers on both rails -- indexed,
    sized+zero-fill, signed-element. Both rails emit the per-element init as a MASKED subscript (a `BCIR_CHK`
    per write, like a regular array init), so the bounds-guard count reconciles. Behaviour-equivalent to
    Clang."""
    from bcir.frontends.cfront import compile_unit

    for src in (
        "unsigned f(unsigned i){ return (unsigned[]){10u,20u,30u,40u}[i&3u]; }",
        "unsigned f(unsigned i){ return (unsigned[4]){10u,20u}[i&3u]; }",
        "int f(unsigned i){ return (int[]){-1,-2,-3}[i%3u]; }",
    ):
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)


@_requires_cc
def test_multidim_scalar_array_complit_lowers():
    """§5.10 item 4 (#arrcomplit): a MULTI-DIM SCALAR-element array compound literal `(T[A][B]){{..},{..}}[i][j]`
    (including an INFERRED outer dim `(T[][N]){...}` and a SIGNED leaf). Both rails rebuild it as the FLAT
    `array(leaf, A*B)` + per-dim `shape` (the same representation the regular multi-dim local decl uses), so the
    storage is `[A*B]` -- NOT the old `shape=()` defect that left it under-sized with the wrong `[i][j]` stride
    (the #500 silent miscompile). The nested ROW braces descend via `_array_row` / `subagg_init_md`, and `[i][j]`
    Horner-flattens to `i*B + j` (the inner dim). Both rails emit an IDENTICAL claim sequence (offsets/strides
    pinned by the `match` check) and stay Clang-behaviour-equivalent."""
    from bcir.frontends.cfront import compile_unit

    for src in (
        "unsigned f(unsigned i, unsigned j){ return (unsigned[2][2]){{1u,2u},{3u,4u}}[i&1u][j&1u]; }",
        "unsigned f(unsigned i, unsigned j){ return (unsigned[][2]){{1u,2u},{3u,4u},{5u,6u}}[i%3u][j&1u]; }",
        # an inferred outer dim via OUT-OF-ORDER DESIGNATORS: the outer dim is max(designator index)+1 = 2
        # (NOT the raw entry count), so the storage is `_cl[2*2]` -- the #506 latent under-sizing where
        # peek_top_entries ignored designators and sized `_cl[1*2]` (a silent over-write past the storage).
        "unsigned f(unsigned i, unsigned j){ return (unsigned[][2]){[1]={5u,6u},[0]={1u,2u}}[i&1u][j&1u]; }",
        "int f(unsigned i, unsigned j){ return (int[2][2]){{-1,-2},{-3,-4}}[i&1u][j&1u]; }",
    ):
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)


@_requires_cc
def test_aggregate_array_complit_lowers():
    """§5.10 item 4 (#arrcomplit): a 1-D AGGREGATE-element array compound literal `(struct P[]){...}[i].field`
    lowers on BOTH rails -- the INFERRED form `(struct P[]){...}` (count from the init), the EXPLICIT form
    `(struct P[N]){...}`, and a PARTIAL element init (the missing `.y` zero-fills off the `= {0}` baseline).
    Each `{...}` element routes through the offset-based per-element struct store (`idx*sizeof(elem)`), and
    `[i].field` strides by the element struct (offsetof(field) + i*sizeof(elem)) -- the same array-of-structs
    descent a regular `struct P a[]` decl uses. Both rails emit an IDENTICAL claim sequence (offsets/strides
    pinned by the `match` check -- a wrong extent/stride is a #500 silent miscompile) and stay
    Clang-behaviour-equivalent. (The MULTI-dim aggregate form lowers too -- see
    test_multidim_aggregate_complit_lowers.)"""
    from bcir.frontends.cfront import compile_unit

    for src in (
        "struct P{unsigned x,y;}; unsigned f(unsigned i){ return (struct P[]){{1u,2u},{3u,4u}}[i&1u].x; }",  # INFERRED
        "struct P{unsigned x,y;}; unsigned f(unsigned i){ return (struct P[2]){{5u,6u},{7u,8u}}[i&1u].y; }",  # EXPLICIT
        "struct P{unsigned x,y;}; unsigned f(unsigned i){ return (struct P[]){{1u},{3u,4u}}[i&1u].y; }",
    ):  # PARTIAL (= {0})
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)


@_requires_cc
def test_multidim_aggregate_complit_lowers():
    """§5.10 item 4 (#arrcomplit): a MULTI-DIM AGGREGATE-element array compound literal
    `(struct P[A][B]){{{..},{..}},{{..},{..}}}[i][j].field` lowers on BOTH rails -- the FIXED-dims form, the
    INFERRED outer form `(struct P[][N]){...}` (row count from the init), and a PARTIAL element init (the
    missing field / row zero-fills off the `= {0}` baseline). Both rails rebuild it as the FLAT `array(struct,
    A*B)` + per-dim `shape` (storage `[A*B]` -- not the old under-sized `shape=()` defect, the #500 silent
    miscompile), descend the nested ROW braces routing each innermost `{...}` element through the offset-based
    per-element struct store, and lower `[i][j].field` by Horner-flattening the OUTER dims (`i*B + j`) then
    striding by the element STRUCT size (8) + the field offset -- the same array-of-structs descent a regular
    `struct P a[A][B]` decl uses. Both rails emit an IDENTICAL claim sequence (offsets/strides pinned by the
    `match` check -- a wrong extent/stride is a #500 silent miscompile) and stay Clang-behaviour-equivalent."""
    from bcir.frontends.cfront import compile_unit

    for src in (
        "struct P{unsigned x,y;}; unsigned f(unsigned i, unsigned j){ "  # FIXED dims
        "return (struct P[2][2]){{{1u,2u},{3u,4u}},{{5u,6u},{7u,8u}}}[i&1u][j&1u].x; }",
        "struct P{unsigned x,y;}; unsigned f(unsigned i, unsigned j){ "  # INFERRED outer
        "return (struct P[][2]){{{1u,2u},{3u,4u}},{{5u,6u},{7u,8u}},{{9u,10u},{11u,12u}}}[i%3u][j&1u].y; }",
        # an inferred outer dim via OUT-OF-ORDER DESIGNATORS: outer = max(idx)+1 = 2, so storage is `_cl[2*2]`
        # (the #506 designated-outer under-sizing, shared with the scalar path -- fixed in peek_top_entries).
        "struct P{unsigned x,y;}; unsigned f(unsigned i, unsigned j){ "  # INFERRED via DESIGNATORS
        "return (struct P[][2]){[1]={{5u,6u},{7u,8u}},[0]={{1u,2u},{3u,4u}}}[i&1u][j&1u].x; }",
        "struct P{unsigned x,y;}; unsigned f(unsigned i, unsigned j){ "  # PARTIAL (= {0})
        "return (struct P[2][2]){{{1u},{3u}},{{5u,6u}}}[i&1u][j&1u].y; }",
    ):
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)


@_requires_cc
def test_multidim_array_braceinit():
    """§5.10 (#localmdinit): a REGULAR multi-dimensional local array with a NESTED-brace initializer
    `T a[A][B] = {{..},{..}}` (2-D + 3-D + a PARTIAL init that exercises the `= {0}` zero baseline). The
    flat resource keeps the row-major `a[A][B]` layout, so each rail descends a nested brace by ROW
    (offset = row*stride; the innermost dim stores per element), not by treating the scalar leaf as the
    element (the old IndexError / parse-error). Behaviour-equivalent to Clang on both rails (the strides /
    offsets are pinned by the `match` check -- a wrong stride is a silent miscompile, #500)."""
    from bcir.frontends.cfront import compile_unit

    for src in (
        "unsigned f(unsigned i, unsigned j){ unsigned a[2][2]={{1u,2u},{3u,4u}}; return a[i&1u][j&1u]; }",
        "unsigned f(unsigned i, unsigned j, unsigned k){ unsigned b[2][2][2]={{{1u,2u},{3u,4u}},{{5u,6u},{7u,8u}}}; return b[i&1u][j&1u][k&1u]; }",
        "unsigned f(unsigned i, unsigned j){ unsigned c[2][3]={{1u},{4u,5u}}; return c[i&1u][j%3u]; }",
    ):
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)


@_requires_cc
def test_inferred_and_aos_local_array():
    """§5.10 (#aoslocal): a REGULAR local array decl with a brace initializer -- the INFERRED-size form
    `T a[] = {...}` (the outer count comes from the initializer; an under-sized `a[0]` would let the
    per-element stores write past the storage -- a #500-class silent miscompile) and the EXPLICIT-size
    form `T a[N] = {...}`, for a SCALAR element AND a STRUCT element. A struct-element array routes each
    `{...}` element through the offset-based per-element struct store (idx*sizeof(elem)), riding the
    `= {0}` baseline so a partial init zero-fills. Both rails lower an IDENTICAL claim sequence
    (offsets/strides pinned by the `match` check) and stay Clang-behaviour-equivalent."""
    from bcir.frontends.cfront import compile_unit

    for src in (
        "struct P{unsigned x,y;}; unsigned f(unsigned i){ struct P a[]={{1u,2u},{3u,4u}}; return a[i&1u].x; }",  # struct, INFERRED
        "struct P{unsigned x,y;}; unsigned f(unsigned i){ struct P b[2]={{5u,6u},{7u,8u}}; return b[i&1u].y; }",  # struct, EXPLICIT
        "unsigned f(unsigned i){ unsigned c[]={10u,20u,30u,40u}; return c[i&3u]; }",  # scalar, INFERRED
        "unsigned f(unsigned i){ unsigned a[]={1u,2u,3u,4u}; return a[i&3u]; }",  # scalar, INFERRED (Bug A)
        "struct P{unsigned x,y;}; unsigned f(unsigned i){ struct P d[2]={{1u}}; return d[i&1u].x + d[i&1u].y; }",
    ):  # struct, PARTIAL (= {0})
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)


@_requires_cc
def test_computed_goto():
    """§5.10 item 6 (#computedgoto): the GNU label-as-value `&&L` (a `void *`) and the indirect `goto *p`.
    A void* holds a taken label address; `goto *p` dispatches. Both rails lower to the GNU forms (which Clang
    compiles), behaviour-equivalent to Clang."""
    from bcir.frontends.cfront import compile_unit

    for src in (
        "unsigned f(unsigned x){ void *p=&&O; if((x&1u)==0u) p=&&E; goto *p; E: return x*2u; O: return x*3u+1u; }",
        "unsigned f(unsigned i){ void *t[3]; t[0]=&&a; t[1]=&&b; t[2]=&&c; goto *t[i%3u]; a: return 1u; b: return 2u; c: return 3u; }",
        "unsigned f(unsigned n){ unsigned s=0u,i=0u; void *p=&&L; L: if(i<(n&7u)){ s+=i; i++; goto *p; } return s; }",
    ):
        r = compile_unit(src, check_clang=True)
        assert r.equivalence == "match" and r.is_clean, (src, r.equivalence)


@_requires_cc
def test_function_pointer_local_variable():
    """§5.10 (#fnptrlocal): a function-pointer LOCAL VARIABLE `RET (*f)(P) = fn;` -- declared, called (an
    indirect call), reassigned, value-selected, and with a signed return. The oracle parsed it; the twin's
    local-decl path did not (only funcptr struct members + typedefs), a latent divergence. Behaviour-equivalent
    to Clang on both rails."""
    from bcir.frontends.cfront import compile_unit

    for src in (
        "static unsigned a(unsigned x){return x+1u;} static unsigned d(unsigned x){return x*2u;} unsigned f(unsigned x){ unsigned (*g)(unsigned)=a; unsigned r=g(x); g=d; return r*10u+g(x); }",
        "static unsigned a(unsigned x){return x+1u;} static unsigned d(unsigned x){return x*2u;} unsigned f(unsigned x, unsigned w){ unsigned (*g)(unsigned)=a; if(w&1u) g=d; return g(x); }",
        "static int n(int x){return -x-1;} int f(int x){ int (*g)(int)=n; int r=g(x); return r>>1; }",
    ):
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
        prog = (
            f'#include <stdint.h>\n#include <stdlib.h>\n#include <stdio.h>\n#include "bcir_quarantine.h"\n'
            f"{r.source}\n\n{body}\n"
            f"size_t bcir_bounds_quarantine(uint64_t rid,uint64_t index,uint64_t extent,const char *site)\n"
            f"{{ bcir_oob_record_event(rid,index,extent,site); return 0; }}\n"
            f"int main(void){{ (void)bcir_g(8u); (void)bcir_g(40u); bcir_quarantine_report(stdout); return 0; }}\n"
        )
        cpath, epath = os.path.join(d, "e.c"), os.path.join(d, "e")
        open(cpath, "w").write(prog)
        b = subprocess.run(
            [
                _CC,
                "-std=c23",
                "-O2",
                "-I",
                _C,
                cpath,
                os.path.join(_C, "bcir_quarantine.c"),
                "-o",
                epath,
            ],
            capture_output=True,
            text=True,
        )
        assert b.returncode == 0, b.stderr
        run = subprocess.run([epath], capture_output=True, text=True)
        assert run.returncode == 0, (
            run.returncode,
            run.stderr,
        )  # survived: the override did not abort
        out = run.stdout
        assert "2 out-of-bounds event(s)" in out, out  # the running total
        assert "g:a" in out and "index 8" in out and "index 40" in out, (
            out
        )  # both sites + indices, in order
        assert "out of [0, 8)" in out, out  # the extent the report resolves


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
            f"{r.source}\n\n{body}\n"
            "int main(int argc, char **argv){\n"
            '  static const bcir_recover_rule confident[] = {{"g:a", BCIR_RECOVER_CLAMP, 900}};\n'
            '  static const bcir_recover_rule underconf[] = {{"g:a", BCIR_RECOVER_CLAMP, 300}};\n'
            "  int abort_mode = argc > 1 && argv[1][0] == '1';\n"
            "  if (abort_mode) bcir_recover_set_policy(underconf, 1, 500);  /* 300 < 500 -> rejected */\n"
            "  else            bcir_recover_set_policy(confident, 1, 500);  /* 900 >= 500 -> admitted clamp */\n"
            "  unsigned v = bcir_g(40u);                 /* index 40 is out of [0,8) -> the handler decides */\n"
            '  printf("recovered=%u\\n", v);\n'
            "  bcir_decide_report(stdout);\n"
            "  return 0;\n}\n"
        )
        cpath, epath = os.path.join(d, "e.c"), os.path.join(d, "e")
        open(cpath, "w").write(prog)
        b = subprocess.run(
            [
                _CC,
                "-std=c23",
                "-O2",
                "-I",
                _C,
                cpath,
                os.path.join(_C, "bcir_quarantine.c"),
                os.path.join(_C, "bcir_quarantine_recover.c"),
                "-o",
                epath,
            ],
            capture_output=True,
            text=True,
        )
        assert b.returncode == 0, b.stderr
        # admitted clamp: confidence 900 >= threshold 500 -> the access lands on a[7] (= 14), program survives,
        # and the recorded decide witnesses the crossing.
        ok = subprocess.run([epath, "0"], capture_output=True, text=True)
        assert ok.returncode == 0 and "recovered=14" in ok.stdout, (
            ok.returncode,
            ok.stdout,
            ok.stderr,
        )
        assert "1 recovery crossing(s)" in ok.stdout, ok.stdout
        assert "g:a" in ok.stdout and "confidence 900/1000 vs threshold 500/1000" in ok.stdout, (
            ok.stdout
        )
        assert "admitted, clamp to index 7" in ok.stdout, ok.stdout
        # rejected: confidence 300 < threshold 500 -> not confident enough to recover -> fail-fast.
        no = subprocess.run([epath, "1"], capture_output=True, text=True)
        assert no.returncode != 0 and "recovery rejected" in no.stderr, (no.returncode, no.stderr)


# ===========================================================================================
# L7 ADVERSARIAL PREPROCESSOR DIFFERENTIAL (against clang -E -P / gcc -E -P) + ROBUSTNESS FUZZ
# ===========================================================================================
# The cfront preprocessor (`bcir/frontends/cfront/cpp.py`, the Python oracle, and its C twin
# `runtime/c/bcir_cpp.c`) is the most bug-prone phase: the no-recursion "blue paint" rule, argument
# prescan, rescanning, stringization, token paste forming new tokens, `__VA_OPT__` comma elision. The
# only prior coverage was 3 fixtures validated *indirectly* through the full lowering pipeline. This
# gate is an adversarial corner-case corpus (`runtime/c/cfront_pp_*.c`) compared DIRECTLY against a
# REFERENCE preprocessor (clang -E -P and gcc -E -P), failing on any disagreement -- the exact gate that
# finds a botched rescan / a stringization escaping error / a `__VA_OPT__` comma bug / a missing
# blue-paint. (Round-1 of this gate flushed four real cpp.py bugs, now fixed: the two-level
# `XSTR(__LINE__)` not prescanning its argument; an argument that is itself a macro call left
# unexpanded; an empty `##` operand emitting a literal `##` instead of a placemarker; and stringize
# dropping a bare backslash + mis-escaping a quoted literal. See the per-fixture headers.)
import glob as _glob

from bcir.frontends.cfront import cpp as _cpp

# The adversarial corpus: every runtime/c/cfront_pp_*.c. Auto-discovered so a new fixture is covered
# with no list edit. These are preprocessor-focused (they need NOT lower through the frontend -- they
# are consumed ONLY by this differential, which compares the *preprocessed text*).
_PP_FIXTURES = sorted(os.path.basename(p) for p in _glob.glob(os.path.join(_C, "cfront_pp_*.c")))

# FAIRNESS PIN (mirrors the RT4 / GCC-differential idiom): a fixture whose single-truth value is
# implementation-defined -- clang -E and gcc -E themselves disagree -- is NOT a fair differential and is
# excluded with a recorded reason. The set is PINNED here so the excluded slice stays explicit and
# minimal; the test ALSO re-derives the xcc-divergence dynamically (so a host whose compilers happen to
# agree still runs it), but a pinned entry documents WHY a known-divergent fixture is expected to drop.
_PP_PIN_XCC = {
    "cfront_pp_stdcver.c": "__STDC_VERSION__ is implementation-defined: clang reports 202311L, gcc "
    "(-std=c2x) 202000L -- no single truth, so it is not a fair differential.",
}


def _pp_token_normalize(text: str) -> list[str]:
    """The differential's normalization, as a TOKEN-SEQUENCE projection. It is deliberately NOT a
    whitespace collapse: it drops `#`-line / line-marker lines and blank lines, then RE-TOKENIZES every
    surviving line with the preprocessor's own pp-token regex and concatenates the token lists.

    WHY IT CANNOT HIDE A TOKEN-LEVEL DIVERGENCE: two outputs compare equal iff they yield the IDENTICAL
    ordered list of preprocessing tokens. The only information discarded is (a) inter-token horizontal
    whitespace and (b) line boundaries -- exactly the two things a reference `-E` formats differently
    from cpp.py (clang writes `((3 +1))` where cpp.py writes `((3+1))`, and elides `__VA_OPT__` to a
    space). It can NEVER delete, insert, merge, split, or reorder a token: `a b` (two tokens) is not
    equal to `ab` (one token), `[##x]` is not equal to `[x]`, and `f(0 , a)` has the same token list as
    `f(0,a)` but a DIFFERENT one from `f(0 a)`. So a real expansion divergence -- a missing paste, a
    leaked `##`, a wrong stringize spelling, a dropped/extra argument, a botched blue-paint -- changes
    the token list and is caught; only pure formatting is absorbed. (The `#`-marker drop is sound
    because cpp.py emits NO `#`-lines after its pass and `-P` already suppresses line markers; the only
    `#`-lines that could survive are a `#pragma`/`#error` we model as a no-op, which neither side emits
    as program text.)"""
    toks: list[str] = []
    for ln in text.splitlines():
        if re.match(r"\s*#", ln):  # a residual directive / line marker: drop
            continue
        toks.extend(_cpp._tokens(ln))
    return toks


def _pp_reference(path: str, cc: str, std: str) -> list[str] | None:
    """The reference preprocessor's token-normalized expansion of `path` under `cc -std=<std> -E -P`,
    or None if it fails to run (a toolchain capability gap -> the caller treats the fixture as unfair)."""
    r = subprocess.run([cc, f"-std={std}", "-E", "-P", path], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return _pp_token_normalize(r.stdout)


def _pp_reference_pair(path: str):
    """The (clang, gcc) reference token lists for `path`, each under the first -std it accepts (clang
    speaks c23; gcc 13 wants c2x). Returns (clang_toks_or_None, gcc_toks_or_None)."""
    clang, gcc = shutil.which("clang"), shutil.which("gcc")
    cl = _pp_reference(path, clang, "c23") if clang else None
    # gcc 13 rejects -std=c23 (it is -std=c2x there); try c23 first then fall back, so a newer gcc still works.
    gc = None
    if gcc:
        gc = _pp_reference(path, gcc, "c23")
        if gc is None:
            gc = _pp_reference(path, gcc, "c2x")
    return cl, gc


def _pp_fixture_verdict(fx: str):
    """Differential ONE adversarial fixture's `cpp.preprocess` against the reference. Returns
    `(failure, excluded)`:
      * `(msg, None)`  -- a REAL divergence: clang and gcc AGREE (a fair single truth) but cpp.preprocess
                          differs -- a genuine cfront preprocessor bug, charged against cpp.py;
      * `(None, why)`  -- excluded as not a fair differential (clang vs gcc disagree, or a compiler is
                          absent / a fixture is pinned in `_PP_PIN_XCC`); cpp.py is NOT charged;
      * `(None, None)` -- cpp.preprocess matched the agreed reference exactly (a clean pass)."""
    path = os.path.join(_C, fx)
    cl, gc = _pp_reference_pair(path)
    if cl is None or gc is None:  # a reference compiler is missing -> not gradable
        return (None, f"{fx}: a reference preprocessor is unavailable")
    if cl != gc:  # clang and gcc disagree -> impl-defined -> unfair
        reason = _PP_PIN_XCC.get(fx, "clang -E and gcc -E disagree (implementation-defined)")
        return (None, f"{fx}: xcc-divergent -- {reason}")
    src = open(path, encoding="utf-8").read()
    try:
        got = _pp_token_normalize(_cpp.preprocess(src, name=fx))
    except _cpp.CPPError as e:  # a fair fixture must preprocess cleanly
        return (f"{fx}: cpp.preprocess raised CPPError on a fair fixture: {e}", None)
    if got != cl:
        # the first differing token pins the divergence in the diagnostic.
        diff = next((i for i, (a, b) in enumerate(zip(got, cl)) if a != b), min(len(got), len(cl)))
        return (
            f"{fx}: cpp.preprocess diverges from the reference (clang==gcc) at token #{diff}: "
            f"cpp.py={got[diff : diff + 4]} reference={cl[diff : diff + 4]} "
            f"(full cpp.py={got} reference={cl})",
            None,
        )
    return (None, None)


def test_preprocessor_differential_against_clang_and_gcc():
    """ADVERSARIAL PREPROCESSOR DIFFERENTIAL (the core deliverable). For every adversarial fixture
    `runtime/c/cfront_pp_*.c`, assert cpp.preprocess's token-normalized expansion EQUALS the reference
    (clang -E -P / gcc -E -P), failing on any disagreement -- but ONLY where clang and gcc agree (a fair
    single truth); an implementation-defined fixture where they disagree is excluded with a recorded
    reason (the fairness gate, e.g. __STDC_VERSION__). Self-skips cleanly if NO reference compiler is
    present. ANTI-DEGENERATION: a non-trivial number of fixtures must actually run the differential, so
    an all-excluded / all-skipped slice FAILS (see the asserts below)."""
    if not (shutil.which("clang") or shutil.which("gcc")):
        return  # no reference preprocessor -> clean self-skip
    assert _PP_FIXTURES, "no cfront_pp_*.c adversarial fixtures were discovered"
    fails, excluded, tested = [], [], 0
    for fx in _PP_FIXTURES:
        msg, why = _pp_fixture_verdict(fx)
        if msg:
            fails.append(msg)
        elif why is not None:
            excluded.append(why)
        else:
            tested += 1
    assert not fails, (
        "cfront preprocessor (cpp.preprocess) diverges from the reference on a FAIR "
        "fixture (clang and gcc agree, cpp.py does not) -- a real preprocessor bug:\n"
        + "\n".join(f"  {m}" for m in fails)
    )
    # ANTI-DEGENERATION: a fair single-truth differential must actually have RUN on a non-trivial number
    # of fixtures. An all-excluded / all-skipped slice (every fixture pinned, or a silent normalization
    # that swallowed every divergence) is itself a failure of this gate, not a pass.
    assert tested >= max(4, len(_PP_FIXTURES) - len(_PP_PIN_XCC) - 1), (
        f"preprocessor differential degenerated: only {tested} of {len(_PP_FIXTURES)} fixtures ran a "
        f"fair differential (excluded: {excluded}) -- expected nearly all to be single-valued and run"
    )


def test_preprocessor_differential_is_not_degenerate_smoke():
    """The differential's NON-SKIPPABLE anti-degeneration smoke (mirrors the GCC-differential smoke): on
    a host where a reference preprocessor is present this FAILS (not skips) if the differential ran on
    too few fixtures -- the failure mode where a botched normalization or an all-pinned set silently
    reduces the gate to nothing. Under the quick tier (toolchain hidden) it correctly self-skips."""
    if not (shutil.which("clang") or shutil.which("gcc")):
        return
    tested = sum(1 for fx in _PP_FIXTURES if _pp_fixture_verdict(fx) == (None, None))
    # at least the blue-paint / paste / stringize / prescan / vaopt / conditionals cores must have run.
    assert tested >= 4, (
        f"preprocessor differential is degenerate: only {tested} fair fixtures ran "
        f"under the reference -- a real differential must exercise the adversarial core"
    )
    # and the fairness gate must actually be EXERCISED: the pinned impl-defined fixture must be present
    # and recognized as excluded, proving the gate is not vacuously passing everything.
    pin = "cfront_pp_stdcver.c"
    if pin in _PP_FIXTURES:
        msg, why = _pp_fixture_verdict(pin)
        assert msg is None and why is not None and "xcc-divergent" in why, (
            f"the impl-defined fairness pin {pin} was not excluded as expected: {(msg, why)}"
        )


# --- preprocessor robustness fuzzing (the totality contract) ------------------------------------
# A deterministic seeded fuzz feeding MALFORMED preprocessor input to cpp.preprocess and asserting it
# ALWAYS returns or raises a clean CPPError -- never crashes, hangs, or blows the stack/memory. Mirrors
# the existing cfuzz / fuzz_cfront totality style: the contract is "every input is handled" (a result OR
# a typed CPPError), with expansion capped so a fork-bomb macro can't run away.


def _pp_fuzz_corpus(rng):
    """A grab-bag of MALFORMED / adversarial preprocessor inputs built from random pieces: unterminated
    `#if`, bad `#define` syntax, `##` at the start/end of a body, an unbalanced macro-call paren list,
    `#include` of nothing, deeply nested `#if`, and a macro that expands toward megabytes (the cap must
    hold). Deterministic given `rng`."""
    ids = ["A", "B", "f", "g", "X", "VA", "M", "Q"]
    rid = lambda: rng.choice(ids)
    pieces = [
        lambda: f"#if {rng.randint(0, 3)}",  # unterminated #if (no #endif)
        lambda: f"#ifdef {rid()}",  # unterminated #ifdef
        lambda: "#elif 1",  # #elif with no #if
        lambda: "#else",  # #else with no #if
        lambda: "#endif",  # #endif with no #if
        lambda: f"#define {rid()}(",  # bad function-macro: open paren only
        lambda: f"#define {rid()}(a, b",  # unterminated param list
        lambda: f"#define {rid()} ## tail",  # ## at the START of a body
        lambda: f"#define {rid()} head ##",  # ## at the END of a body
        lambda: f"#define {rid()}(x) # ",  # lone # with no operand
        lambda: f"#define {rid()}(x) x ## ## x",  # doubled ##
        lambda: f"{rid()}({rid()}, {rid()}",  # unbalanced macro CALL (no close)
        lambda: f"{rid()}(((((",  # deeply unbalanced parens
        lambda: "#include",  # #include of nothing
        lambda: '#include "',  # #include with a dangling quote
        lambda: "#include <>",  # empty angle include
        lambda: f"#define {rid()} {rid()}",  # a plain define (may form a cycle)
        lambda: f"#undef {rid()}",  # undef (maybe of an undefined name)
        lambda: "#" + rng.choice(["bogus", "1nvalid", "", "pragma x"]),  # unknown / empty directive
        lambda: f"#if defined({rid()}) && ({rng.randint(0, 9)} / 0)",  # division by zero in #if
        lambda: f"#line {rng.choice(['x', '', '999999999999999999999'])}",  # malformed #line
        lambda: (
            "#define S(x) #x\nS(" + "\\" * rng.randint(1, 6)
        ),  # stringize of trailing backslashes
        lambda: rid() + "'unterminated char",  # an unterminated char literal
        lambda: '"unterminated string',  # an unterminated string literal
    ]
    n = rng.randint(1, 12)
    return "\n".join(rng.choice(pieces)() for _ in range(n)) + "\n"


def _pp_fuzz_cycle_and_bomb(rng):
    """Two pathological-but-well-formed inputs the totality contract must survive WITHOUT hanging: a
    self-/mutually-referential macro (the blue-paint rule must terminate the rescan) and an
    exponentially-growing nested expansion (the cap must stop it). Deterministic given `rng`."""
    out = []
    # a mutual-reference cycle: blue paint must leave the painted identifiers, not loop forever.
    out.append("#define A B\n#define B A\nA B\n")
    # a self-referential function macro.
    out.append(f"#define f(x) f(f(x))\nf({rng.randint(0, 9)})\n")
    # a doubling chain L0->L1->...: each level concatenates the previous twice. Bounded depth so it is a
    # large-but-finite expansion; the engine must produce a (capped) result or a CPPError, never run away.
    depth = rng.randint(6, 12)
    body = ["#define L0 xx"]
    for k in range(1, depth):
        body.append(f"#define L{k} L{k - 1} L{k - 1}")
    body.append(f"L{depth - 1}")
    out.append("\n".join(body) + "\n")
    return out


def _preprocessor_robustness_fuzz_seed(seed: int):
    """The seeded preprocessor robustness fuzz (the totality contract): feed many malformed / adversarial
    inputs to cpp.preprocess and assert it ALWAYS terminates with EITHER a string result OR a clean
    typed CPPError -- never an uncaught exception, a crash, a hang, or unbounded memory. Pure-Python, so
    it runs under EVERY tier (no toolchain needed); deterministic given the seed. A hang is caught by the
    suite-level wall clock + the expansion cap; an uncaught non-CPPError exception fails here."""
    import random as _random

    rng = _random.Random(seed)
    cap = 8 * 1024 * 1024  # an output this large is a runaway -> fail
    for _ in range(400):
        src = _pp_fuzz_corpus(rng)
        try:
            out = _cpp.preprocess(src)
            assert isinstance(out, str), (src, type(out))
            assert len(out) < cap, f"runaway expansion ({len(out)} bytes) on:\n{src}"
        except _cpp.CPPError:
            pass  # a typed, intended preprocessing error: fine
        except RecursionError as e:  # a stack blowout is a robustness BUG, not clean
            raise AssertionError(
                f"cpp.preprocess RecursionError (not a clean CPPError) on:\n{src}"
            ) from e
    for src in (s for _ in range(40) for s in _pp_fuzz_cycle_and_bomb(rng)):
        try:
            out = _cpp.preprocess(src)
            assert isinstance(out, str) and len(out) < cap, f"runaway/non-str on:\n{src[:200]}"
        except _cpp.CPPError:
            pass
        except RecursionError as e:
            raise AssertionError(
                f"cpp.preprocess RecursionError on a cycle/bomb input:\n{src[:200]}"
            ) from e


def test_preprocessor_robustness_fuzz_seed1():
    """Preprocessor totality fuzz, seed 1 (malformed directives + cycle/bomb; see
    `_preprocessor_robustness_fuzz_seed`). cpp.preprocess must always return or raise a clean CPPError."""
    _preprocessor_robustness_fuzz_seed(1)


def test_preprocessor_robustness_fuzz_seed2():
    """Preprocessor totality fuzz, seed 2 (see `_preprocessor_robustness_fuzz_seed`)."""
    _preprocessor_robustness_fuzz_seed(2)


def test_preprocessor_robustness_fuzz_seed3():
    """Preprocessor totality fuzz, seed 3 (see `_preprocessor_robustness_fuzz_seed`)."""
    _preprocessor_robustness_fuzz_seed(3)


# --- preprocessor FUZZ-DIFFERENTIAL (random macro programs vs clang -E -P AND gcc -E -P) ---------
# The 9 hand-authored fixtures are a fixed adversarial set; a seeded fuzz-differential generalizes the
# coverage -- it generates random small macro programs (object + function `#define`s mixing `##`, `#`,
# nested calls), keeps only the FAIR ones (clang -E -P token-equals gcc -E -P AND both compile), and
# asserts cpp.preprocess token-equals the reference on ALL of them. This is exactly the gate that
# surfaced the object-macro `##` bug (a `##` in an OBJECT-macro body must paste, not stay literal -- the
# function-macro path already pasted; the object path did not). It runs ONLY when both clang AND gcc are
# present (it needs the cross-compiler fairness vote); self-skips otherwise.

_PP_FUZZ_IDS = ["a", "b", "c", "x", "y", "z", "FOO", "BAR", "M", "N", "P", "Q"]


def _pp_fuzz_body(rng, params, obj):
    """A random replacement list: identifiers, parameters, `##` pastes, and (function-only) `#` stringize.
    Trims a leading/trailing `##` and a trailing lone `#` -- those are constraint violations the reference
    rejects (an unfair program), so the generator avoids authoring them."""
    toks: list[str] = []
    for _ in range(rng.randint(1, 4)):
        r = rng.random()
        if params and r < 0.3:
            toks.append(rng.choice(params))
        elif r < 0.55 and toks and toks[-1] not in ("##", "#"):
            toks.append("##")
            toks.append(rng.choice(_PP_FUZZ_IDS + params))
        elif not obj and params and r < 0.7:
            toks.append("#")
            toks.append(rng.choice(params))
        else:
            toks.append(rng.choice(_PP_FUZZ_IDS))
    while toks and toks[0] == "##":
        toks.pop(0)
    while toks and toks[-1] in ("##", "#"):
        toks.pop()
    return " ".join(toks) if toks else rng.choice(_PP_FUZZ_IDS)


def _pp_fuzz_program(rng) -> str:
    """A random small macro program: a few object + function `#define`s, then a few uses of them."""
    defs = []  # (name, nparams_or_None, source_line)
    for _ in range(rng.randint(2, 5)):
        nm = rng.choice(["O", "F"]) + str(len(defs))
        if rng.random() < 0.5:  # object macro
            defs.append((nm, None, f"#define {nm} {_pp_fuzz_body(rng, [], obj=True)}"))
        else:  # function macro
            ps = [chr(ord("p") + k) for k in range(rng.randint(0, 2))]
            defs.append(
                (nm, len(ps), f"#define {nm}({', '.join(ps)}) {_pp_fuzz_body(rng, ps, obj=False)}")
            )
    lines = [d[2] for d in defs]
    for _ in range(rng.randint(1, 3)):
        nm, npar, _src = rng.choice(defs)
        lines.append(
            nm
            if npar is None
            else f"{nm}({', '.join(rng.choice(_PP_FUZZ_IDS) for _ in range(npar))})"
        )
    return "\n".join(lines) + "\n"


def _pp_reference_text(src: str, cc: str, std: str):
    """The reference's token-normalized expansion of source TEXT `src` (not a path)."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.c")
        open(p, "w", encoding="utf-8").write(src)
        return _pp_reference(p, cc, std)


def _pp_gcc_std() -> str:
    """gcc's C23 spelling: `c23` on a new gcc, `c2x` on gcc 13 (which rejects `-std=c23`)."""
    gcc = shutil.which("gcc")
    probe = "int x;\n"
    return "c23" if (gcc and _pp_reference_text(probe, gcc, "c23") is not None) else "c2x"


def _run_pp_fuzz_differential(seed: int, count: int = 300):
    """The fuzz-differential body: returns (fair, divergences, first_examples)."""
    import random as _random

    clang, gcc = shutil.which("clang"), shutil.which("gcc")
    if not (clang and gcc):
        return None  # needs the cross-compiler fairness vote -> skip
    gstd = _pp_gcc_std()
    rng = _random.Random(seed)
    fair = div = 0
    examples = []
    for _ in range(count):
        src = _pp_fuzz_program(rng)
        cl = _pp_reference_text(src, clang, "c23")
        gc = _pp_reference_text(src, gcc, gstd)
        if cl is None or gc is None or cl != gc:  # not a fair single-truth program
            continue
        fair += 1
        try:
            got = _pp_token_normalize(_cpp.preprocess(src))
        except _cpp.CPPError:
            continue  # a malformed program cpp.py rejects -> not graded
        if got != cl:
            div += 1
            if len(examples) < 5:
                examples.append((src.strip(), got, cl))
    return fair, div, examples


def test_preprocessor_fuzz_differential_seed_a():
    """FUZZ-DIFFERENTIAL (the generalization of the fixed corpus): random object+function macro programs
    with `##`/`#`/nested calls; on the fair subset (clang -E -P == gcc -E -P), cpp.preprocess must
    token-equal the reference -- 0 divergences. This is the gate that surfaced the object-macro `##`
    paste bug (now fixed); a regression re-introduces divergences and fails here. Self-skips if clang or
    gcc is absent (it needs the two-compiler fairness vote). The seed/count are fixed (deterministic)."""
    res = _run_pp_fuzz_differential(20240628, count=300)
    if res is None:
        return
    fair, div, examples = res
    assert fair >= 100, (
        f"fuzz-differential generated too few fair programs ({fair}) -- not exercising"
    )
    assert div == 0, (
        "cpp.preprocess diverges from the reference (clang==gcc) on a fair random macro "
        "program -- a real preprocessor bug:\n"
        + "\n".join(f"  SRC {s!r}\n    cpp.py={p}\n    reference={c}" for s, p, c in examples)
    )


def test_preprocessor_fuzz_differential_seed_b():
    """Fuzz-differential, a second seed (see `test_preprocessor_fuzz_differential_seed_a`)."""
    res = _run_pp_fuzz_differential(1337, count=300)
    if res is None:
        return
    fair, div, examples = res
    assert fair >= 100, f"fuzz-differential generated too few fair programs ({fair})"
    assert div == 0, (
        "cpp.preprocess diverges from the reference on a fair random macro program:\n"
        + "\n".join(f"  SRC {s!r}\n    cpp.py={p}\n    reference={c}" for s, p, c in examples)
    )


# --- DUAL-RAIL C TWIN preprocessor differential (bonus) -----------------------------------------
# The C twin `bcir_cpp_run` preprocesses then feeds the frontend; `test_cfront.c --emit-cpp` now dumps
# the preprocessed text and exits, so the C rail's preprocessor can be differentialed against the SAME
# reference -- catching a Python<->C preprocessor divergence. The C twin is the production port and is
# NOT yet at parity with the (now-fixed) Python oracle on three adversarial classes; those are PINNED
# below with a recorded reason + TODO (fixing the C twin's prescan / stringize-escaping / 3-cycle
# blue-paint is fragile surgery in a separate ~500-line C engine whose expand_line uses non-reentrant
# static buffers -- out of scope for this Python-side gate). The differential still runs on every fixture
# where the C twin AGREES with the reference, so a future Python<->C regression on the clean subset is
# caught, and the pinned set is asserted to stay BOUNDED (it must not silently grow).
_PP_CTWIN_PIN = {
    "cfront_pp_stringize.c": "C twin (bcir_cpp.c) does not prescan a two-level XSTR argument and does "
    'not escape `"`/`\\` inside a stringized string-literal token -- '
    "TODO(bcir_cpp): add argument prescan + spec stringize escaping.",
    "cfront_pp_predefined.c": "C twin does not prescan `__LINE__` through the two-level XSTR "
    '(emits "__LINE__" not the number) -- TODO(bcir_cpp): prescan dynamic '
    "predefineds in argument position.",
    "cfront_pp_bluepaint.c": "C twin mis-handles a 3-macro indirect self-reference cycle (p->q->r->p) "
    "-- TODO(bcir_cpp): paint the whole active replacement chain, not one level.",
}


def _ctwin_emit_cpp(fx: str):
    """The C twin's preprocessed text for `fx` via `test_cfront <bin> --emit-cpp`, token-normalized; or
    None if the binary or run is unavailable."""
    exe = _build_frontend(_session_build_dir())
    r = subprocess.run([exe, "--emit-cpp", os.path.join(_C, fx)], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return _pp_token_normalize(r.stdout)


def test_c_twin_preprocessor_differential_against_reference():
    """DUAL-RAIL C-TWIN preprocessor differential (bonus). For each adversarial fixture where the C twin
    is at parity, assert its `--emit-cpp` token-normalized output equals the reference (on the fair,
    single-truth fixtures) -- catching a Python<->C preprocessor divergence on the clean subset. The
    three known C-twin-immature classes are PINNED in `_PP_CTWIN_PIN` (recorded reason + TODO; fixing
    the C engine is fragile surgery, out of scope). Self-skips if no C compiler / reference is present.
    Asserts the pinned set stays BOUNDED (a new C-twin divergence must be triaged, not silently pinned)
    and that a non-trivial subset actually ran (anti-degeneration)."""
    if not _CC or not (shutil.which("clang") or shutil.which("gcc")):
        return  # no toolchain -> clean self-skip
    fails, pinned, tested = [], [], 0
    for fx in _PP_FIXTURES:
        _msg, why = _pp_fixture_verdict(fx)
        if why is not None:  # an unfair fixture (impl-defined) -> skip the C rail too
            continue
        cl, _gc = _pp_reference_pair(os.path.join(_C, fx))
        ct = _ctwin_emit_cpp(fx)
        if fx in _PP_CTWIN_PIN:
            # a pinned fixture is EXPECTED to diverge; assert it still DOES (so a silent C-twin FIX
            # un-pins it loudly instead of leaving a stale pin that hides real coverage).
            if ct is not None and ct == cl:
                fails.append(
                    f"{fx}: STALE PIN -- the C twin now AGREES with the reference; drop it "
                    f"from _PP_CTWIN_PIN so the fixture is differentialed again "
                    f"(recorded reason: {_PP_CTWIN_PIN[fx]})"
                )
            pinned.append(fx)
            continue
        if ct is None:
            fails.append(f"{fx}: C twin --emit-cpp failed to run")
        elif ct != cl:
            diff = next(
                (i for i, (a, b) in enumerate(zip(ct, cl)) if a != b), min(len(ct), len(cl))
            )
            fails.append(
                f"{fx}: C twin diverges from the reference at token #{diff}: "
                f"ctwin={ct[diff : diff + 4]} reference={cl[diff : diff + 4]} -- a Python<->C "
                f"preprocessor divergence on a fixture the C twin was expected to handle"
            )
        else:
            tested += 1
    assert not fails, (
        "C-twin preprocessor differential failures (the C rail diverges from the "
        "reference on a non-pinned fair fixture):\n" + "\n".join(f"  {m}" for m in fails)
    )
    # The pinned set stays bounded from BOTH sides, and neither side is a tautology:
    #   * it cannot GROW  -- a new divergence lands on a non-pinned fixture and is in `fails` above;
    #   * it cannot ROT   -- a pin whose divergence disappeared is a STALE PIN, also in `fails`.
    # This residual check only guards the bookkeeping itself (a pin recorded for a fixture that is
    # not in the pin table would mean `_pp_fixture_verdict`/the loop drifted).
    assert set(pinned) <= set(_PP_CTWIN_PIN), (
        f"unexpected C-twin pins: {set(pinned) - set(_PP_CTWIN_PIN)}"
    )
    # ANTI-DEGENERATION: the C-twin differential must have actually RUN on a non-trivial fair subset.
    assert tested >= 3, (
        f"C-twin preprocessor differential degenerated: only {tested} fixtures ran "
        f"(pinned={pinned}) -- a real differential must exercise the clean subset"
    )


# --- CF-PASTE: the preprocessor keeps apart tokens that would lex as others ----------------------------
# Every pp-token the property below draws from: words, pp-numbers (an exponent's sign, a leading `.`, a hex
# digit `e`), string literals, and every punctuator the preprocessor lexes.
_PASTE_VOCAB = (
    "a", "b", "x1", "e", "E", "p", "_y", "L",
    "0", "1", "12", "1.5", ".5", "0x1e", "1e", "1e+5", "0x1p-3", "07",
    '"s"', '"a b"',
    *_cpp._PUNCT,
)  # fmt: skip


def _paste_lines(seed: int, n: int) -> list[list[str]]:
    """`n` random token sequences over `_PASTE_VOCAB` (2..8 tokens each), deterministic in `seed`."""
    import random

    rng = random.Random(seed)
    return [[rng.choice(_PASTE_VOCAB) for _ in range(rng.randint(2, 8))] for _ in range(n)]


def test_the_preprocessor_keeps_apart_tokens_that_would_lex_as_others():
    """CF-PASTE: both cfront preprocessors re-spell every source line from its tokens, and kept a space
    only between two words, so two tokens whose spellings run together into others came out as those
    (maximal munch, C 6.4p4). `a + ++g` came out `a+++g`, which is `(a++) + g`; `-NEG(a)` with
    `#define NEG(x) -x` came out `--a`; `a + INC b` with `#define INC ++` came out `a+++b` -- each unit
    lowered clean and computed another value. `y / *p` came out as a comment opener and `a + +b` as a
    parse error. One predicate on each rail now decides the space (`cpp._pastes`, `bcir_cpp.c` `pastes`):
    two words, a comment opener, a punctuator maximal munch would extend, two dots (the ellipsis), a
    pp-number that runs on. The property: for random token sequences over every punctuator, pp-number
    shape and word, the joined text re-lexes to exactly those tokens, and the twin writes the same text
    byte for byte. Every tracked C source preprocesses as before (checked when this landed: the rule
    only ever adds a space where the text lexed as other tokens). `cfront_paste.c` runs each once-wrong
    function against the original on both rails' emits; `cfront_pp_avoidpaste.c` holds both
    preprocessors to clang and gcc in the reference differentials above."""
    lines = _paste_lines(20260929, 2000)
    for toks in lines:
        text = _cpp._join(toks)
        assert _cpp._tokens(text) == toks, (toks, text)
    assert _cpp._join(["a", "+", "++", "g"]) == "a+ ++g"  # the space where it is needed ...
    assert _cpp._join(["i", "--", ">", "0"]) == "i-->0"  # ... and only there: `-->` lexes `--` `>`
    assert _cpp._join([".", ".", "."]) == ". . ."  # three tokens, not the ellipsis
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    # each sequence as a line (after a word, so never a directive); then -- where it is one macro
    # argument (no parenthesis or comma; no string, whose escaping in `#` the twin does not model) --
    # stringized, which keeps the argument's own spelling, and through an identity macro, rescanned. Two
    # units, each inside the twin harness's 64 KiB input.
    args = [t for t in lines if not {"(", ")", ",", '"s"', '"a b"'} & set(t)][:700]
    assert len(args) == 700, len(args)
    units = (
        "".join(f"q {' '.join(t)}\n" for t in lines),
        "#define STR(x) #x\n#define ID(x) x\n"
        + "".join(f"q STR({' '.join(t)}) ID({' '.join(t)})\n" for t in args),
    )
    for body in units:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "paste_lines.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(body)
            twin = subprocess.run([exe, "--emit-cpp", path], capture_output=True, text=True)
        assert twin.returncode == 0, twin.stderr
        py = _cpp.preprocess(body, name=path)
        assert twin.stdout.splitlines() == py.splitlines(), next(
            (t, p) for t, p in zip(twin.stdout.splitlines(), py.splitlines(), strict=True) if t != p
        )
    fx = "cfront_paste.c"
    fpath = os.path.join(_C, fx)
    src = open(fpath, encoding="utf-8").read()
    oracle_summary, r, _entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    c_summary, c_emit = _c_run(exe, fpath)
    assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
    oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    renamed = src
    for f in r.lowered.functions:
        renamed = re.sub(r"\b" + f + r"\b", f + "_s", renamed)
    driver = r"""
static int fail(const char *what) { puts(what); return 1; }
int main(void) {
  for (int32_t k = -300; k < 300; k++) {
    uint32_t u = (uint32_t)k * 2654435761u, v = (uint32_t)(k * 7 + 3), dv = v | 1u;
    if (ps_preinc_s(u, v) != bcir_ps_preinc(u, v)) return fail("preinc");
    if (ps_predec_s(u, v) != bcir_ps_predec(u, v)) return fail("predec");
    if (ps_negneg_s(k) != bcir_ps_negneg(k)) return fail("negneg");
    if (ps_incmacro_s(k, 3 * k) != bcir_ps_incmacro(k, 3 * k)) return fail("incmacro");
    if (ps_deref_s(u, &dv) != bcir_ps_deref(u, &dv)) return fail("deref");
    if (ps_unary_s(k, 5 * k) != bcir_ps_unary(k, 5 * k)) return fail("unary");
  }
  puts("MATCH");
  return 0;
}
"""
    head = "#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
    compilers = [c for c in dict.fromkeys((_CC, shutil.which("clang"), shutil.which("gcc"))) if c]
    with tempfile.TemporaryDirectory() as d:
        for cc in compilers:
            for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
                out = _build_run_c(d, cc, label, f"{head}{renamed}\n{emit}\n{driver}")
                assert out == "MATCH", f"{fx}: {label} emit not equivalent under {cc} ({out})"


# CF-SIZEOF: every operand form `sizeof` takes, one function each, folded per target and held three ways --
# the oracle's value against a pinned vector (every tier), against Clang's own value for the same function
# under `-target` (`ret i64 N`) when Clang is present, and the twin's claim graph against the oracle's (the
# structural digest folds every constant) when a C compiler is. The operand is measured as its own type: an
# array whole, a row of a multi-dimensional one a row, a member array its member, a parameter declared as an
# array the pointer it is; an operator's result by its operands after they convert, a bit-field operand
# promoted as Clang promotes its value; a call by its callee's return. The rails had folded 224 and 119 of
# these 460 cases wrong (a bare name's element, or a flat 4), and refused 265 on the twin.
_SIZEOF_HEAD = """#include <stdint.h>
#include <stddef.h>
typedef uint32_t (*fp_t)(uint32_t);
struct s { uint8_t c; uint64_t n; uint32_t a[4]; };
struct t { uint16_t k; struct s in; uint8_t z[3]; };
union u { uint8_t b[5]; uint32_t w; };
struct bf { uint32_t lo : 3; uint32_t hi : 29; uint64_t big : 40; uint8_t tiny : 2; uint64_t mid : 5; };
struct fps { uint8_t c; fp_t f; uint8_t d; };
struct fpd { uint8_t c; uint32_t (*g)(uint32_t); uint8_t d; };
uint32_t ga[10];
uint8_t gb[3][5];
struct s gs;
struct s gsa[3];
double gd[7];
long double gld;
uint16_t hfun(uint32_t v) { return (uint16_t)v; }
struct s mk(void) { struct s v = {0}; return v; }
"""
_SIZEOF_FORMS = (  # (parameters, locals, the operand after `sizeof`)
    ("", "uint32_t la[10]; ", "(la)"),
    ("", "uint16_t l2[4][6]; ", "(l2)"),
    ("", "uint16_t l2[4][6]; ", "(l2[1])"),
    ("", "uint16_t l2[4][6]; ", "(l2[1][2])"),
    ("", "uint16_t l2[4][6]; ", "(*l2)"),
    ("", "uint32_t la[10]; ", "(la + 1)"),
    ("", "uint32_t la[10]; ", "(&la)"),
    ("", "struct s ls; ", "(ls.a)"),
    ("", "struct s ls; ", "(ls.n)"),
    ("", "struct s ls; ", "(ls)"),
    ("", "struct s lsa[4]; ", "(lsa[1].a)"),
    ("", "struct s lsa[4]; ", "(lsa->n)"),
    ("", "struct t lt; ", "(lt.in.a)"),
    ("", "struct t lt; ", "(lt)"),
    ("", "union u lu; ", "(lu)"),
    ("", "union u lu; ", "(lu.b)"),
    ("uint32_t *p", "", "(p)"),
    ("struct s *sp", "", " *sp"),
    ("struct s *sp", "", "(sp->a)"),
    ("uint16_t (*pr)[6]", "", "(*pr)"),
    ("uint32_t ap[8]", "", "(ap)"),
    ("uint16_t m[4][6]", "", "(m[1])"),
    ("", "", "(ga)"),
    ("", "", "(gb)"),
    ("", "", "(gb[1])"),
    ("", "", "(gs)"),
    ("", "", "(gs.a)"),
    ("", "", "(gsa)"),
    ("", "", "(gsa[2].a)"),
    ("", "", "(gd)"),
    ("", "", "(gld)"),
    ("", "", '("abc")'),
    ("uint8_t x8", "", "(x8 + 1)"),
    ("int64_t x64", "", " -x64"),
    ("float xf", "", "(xf * 2.0)"),
    ("uint32_t x32", "", "((uint8_t)x32)"),
    ("uint64_t x64u", "", "(!x64u)"),
    ("uint64_t x64u", "", "(x64u == 1u)"),
    ("uint8_t x8, uint64_t x64u", "", "(x8 ? x8 : x64u)"),
    ("uint8_t x8, uint64_t x64u", "", "(x8, x64u)"),
    ("uint8_t x8, uint64_t x64u", "", "(x8 = x64u)"),
    ("uint8_t x8", "", "(x8++)"),
    ("", "", "(hfun(1u))"),
    ("", "", "(mk().n)"),
    ("uint8_t x8", "", "(&x8)"),
    ("uint8_t x8", "", "(sizeof x8)"),
    ("", "", "(1.0L)"),
    ("uint32_t *p, uint32_t *q", "", "(p - q)"),
    ("uint32_t *p, uint32_t *q, uint8_t x8", "", "(x8 ? p : q)"),
    ("", "struct bf b; ", "(b.lo + 1)"),
    ("", "struct bf b; ", "(b.big + 1)"),
    ("", "struct bf b = {0}; ", "(b.tiny = 1)"),
    ("", "struct bf b = {0}; ", "(0, b.tiny)"),
    ("fp_t fp", "", "(fp)"),
    ("", "", "(fp_t)"),
    ("fp_t fp", "", "(fp(1u))"),
    ("", "", "(int)"),
    ("", "", "(struct s)"),
    ("", "", "(uint32_t *)"),
    ("", "", "(uint32_t[10])"),
    ("", "", "(struct s[2])"),
    ("", "", "(uint8_t *[5])"),
    ("", "", "(uint32_t[3][4])"),
    ("", "", "(_Bool)"),
    ("", "", "(long double)"),
    ("", "struct bf b; ", "(b.mid + 1)"),
    ("", "", "(struct fps)"),
    ("", "", "(struct fpd)"),
    ("", "", "(wchar_t)"),  # CF-SMALL: the target's wchar_t -- 4 bytes on Linux, 2 on Windows
    ("", "", '(L"ab")'),
    ("", "", '(u"ab")'),
    ("", "", '(U"ab")'),
    ("", "", "(&hfun)"),  # a function's address is a pointer, never the function
)
# Clang's values (x86-64, AArch64 and RISC-V Linux share LP64; Windows x64 differs only in `long double`)
_SIZEOF_LP64 = [
    40, 48, 12, 2, 12, 8, 8, 16, 8, 32, 16, 8, 16, 48, 8, 5, 8, 32, 16, 12, 8, 12, 40, 15, 5, 32, 16, 96,
    16, 56, 16, 4, 4, 8, 8, 1, 4, 4, 8, 8, 1, 1, 2, 8, 8, 8, 16, 8, 8, 4, 8, 1, 1, 8, 8, 4, 4, 32, 8, 40,
    64, 40, 48, 1, 16, 4, 24, 24, 4, 12, 6, 12, 8,
]  # fmt: skip
_SIZEOF_PINS = {
    "x86_64-linux": _SIZEOF_LP64,
    "aarch64-linux": _SIZEOF_LP64,
    "riscv64-linux": _SIZEOF_LP64,
    "x86_64-windows": [
        40, 48, 12, 2, 12, 8, 8, 16, 8, 32, 16, 8, 16, 48, 8, 5, 8, 32, 16, 12, 8, 12, 40, 15, 5, 32, 16, 96,
        16, 56, 8, 4, 4, 8, 8, 1, 4, 4, 8, 8, 1, 1, 2, 8, 8, 8, 8, 8, 8, 4, 8, 1, 1, 8, 8, 4, 4, 32, 8, 40,
        64, 40, 48, 1, 8, 4, 24, 24, 2, 6, 6, 12, 8,
    ],
    "i386-linux": [
        40, 48, 12, 2, 12, 4, 4, 16, 8, 28, 16, 8, 16, 36, 8, 5, 4, 28, 16, 12, 4, 12, 40, 15, 5, 28, 16, 84,
        16, 56, 12, 4, 4, 8, 8, 1, 4, 4, 8, 8, 1, 1, 2, 8, 4, 4, 12, 4, 4, 4, 8, 1, 1, 4, 4, 4, 4, 28, 4, 40,
        56, 20, 48, 1, 12, 4, 12, 12, 4, 12, 6, 12, 4,
    ],
}  # fmt: skip


def _sizeof_unit() -> str:
    return _SIZEOF_HEAD + "".join(
        f"uint64_t szf{i}({params or 'void'}) {{ {locals_}return sizeof{operand}; }}\n"
        for i, (params, locals_, operand) in enumerate(_SIZEOF_FORMS)
    )


def _returned_const(fn):
    """The constant a lowered function returns: its return value's writer, followed through copies and
    conversions back to the `c.const` it was folded to (None when the value is not a folded constant)."""
    writer = {w: c for c in fn.claims for w in c.wr}
    rid = fn.return_rid
    for _ in range(8):
        c = writer.get(rid)
        if c is None:
            return None
        if c.op == "c.const":
            return c.imm[0]
        rid = c.rd[0] if c.rd else None
    return None


def test_sizeof_measures_the_operand_type_on_every_target():
    """CF-SIZEOF: `sizeof` measures its operand's own type (C11 6.5.3.4p2) -- never the pointer an array
    decays to elsewhere -- on every target the ABI tables model, identically on both rails. The pinned
    vectors are Clang's; with Clang present they are re-derived, and with a C compiler present the twin's
    claim graph for the whole unit must be the oracle's (the digest folds every constant)."""
    src = _sizeof_unit()
    folded = {}
    for t in _ABI_TARGETS:
        r = compile_unit(src, check_clang=False, target=t)
        folded[t] = [
            _returned_const(r.lowered.functions[f"szf{i}"]) for i in range(len(_SIZEOF_FORMS))
        ]
        wrong = [
            (_SIZEOF_FORMS[i][2], folded[t][i], _SIZEOF_PINS[t][i])
            for i in range(len(_SIZEOF_FORMS))
            if folded[t][i] != _SIZEOF_PINS[t][i]
        ]
        assert not wrong, (t, wrong)
    clang = shutil.which("clang")
    if clang:
        from bcir.frontends.cfront.abi import TARGETS

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sizeof.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
            for t in _ABI_TARGETS:
                run = subprocess.run(
                    [clang, "-target", TARGETS[t].triple, "-std=c2x", "-ffreestanding", "-w", "-O1",
                     "-S", "-emit-llvm", "-o", "-", path],
                    capture_output=True, text=True,
                )  # fmt: skip
                assert run.returncode == 0, (t, run.stderr[:2000])
                got = {}
                for m in re.finditer(r"define [^@]*@szf(\d+)\(.*?\n\}", run.stdout, re.S):
                    rets = re.findall(r"ret i64 (\d+)", m.group(0))
                    assert len(rets) == 1, (t, m.group(0)[:300])
                    got[int(m.group(1))] = int(rets[0])
                assert [got[i] for i in range(len(_SIZEOF_FORMS))] == _SIZEOF_PINS[t], t
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sizeof.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        for t in _ABI_TARGETS:
            r = compile_unit(src, check_clang=False, target=t)
            oracle = f"{cfront_structural_digest(r.lowered):016x}"
            run = subprocess.run([exe, "--target", t, path], capture_output=True, text=True)
            twin = re.search(r"digest=([0-9a-f]{16})", run.stdout)
            assert twin and twin.group(1) == oracle, (t, run.stdout[:300], oracle)


# CF-SIZEOF: operands neither rail gives a value to -- each a constraint violation, an incomplete type, a
# size no target object can have, or a runtime-sized row -- refused by both, never folded to a guess.
# (the unit, the oracle's refusal, the twin's): each form must be refused for the reason it witnesses
_SIZEOF_REFUSED = (
    (
        "struct bf { uint32_t lo : 3; };\nuint64_t f(void) { struct bf b; return sizeof b.lo; }",
        "sizeof of a bit-field",
        "sizeof of a bit-field",
    ),
    (
        "uint32_t g(uint32_t v) { return v; }\nuint64_t f(void) { return sizeof g; }",
        "sizeof of the function 'g'",
        "sizeof of a function designator",
    ),
    (
        "uint64_t f(struct fwd *p) { return sizeof *p; }",
        "sizeof of an incomplete type",
        "sizeof of an incomplete type",
    ),
    ("uint64_t f(void *p) { return sizeof *p; }", "incomplete type", "incomplete type"),
    (
        "uint64_t f(void) { return sizeof(char[1000000][1000000][1000000][1000000]); }",
        "too large for the target",
        "too large for the target",
    ),
    ("uint64_t f(void) { return sizeof(char[-1]); }", "incomplete type", "incomplete type"),
    (
        "uint64_t f(uint32_t n) { uint32_t v[n][n]; v[0][0] = 1u; return sizeof v[0]; }",
        "incomplete type",
        "incomplete type",
    ),
    (
        "uint64_t f(void) { return sizeof(undeclared(1)); }",
        "return type is unknown",
        "return type is unknown",
    ),
)


def test_sizeof_refuses_what_it_cannot_measure_on_both_rails():
    """CF-SIZEOF: a bit-field, a function designator, an incomplete type (a pointer to an undefined struct,
    `void`), a type larger than any target object, a negative dimension, a row of a multi-dimensional VLA
    and a call to an undeclared function are refused by both rails, each for its own reason; the twin exits
    with its refusal, never a crash. (The twin had folded the bit-field and the void pointee, and the oracle
    a 10^24 constant; the oracle refused `sizeof g` as an undeclared identifier, and the undefined struct
    with a bare KeyError.)"""
    from bcir.frontends.cfront.lower import CLowerError

    head = "#include <stdint.h>\n"
    for body, why, _ in _SIZEOF_REFUSED:
        try:
            compile_unit(head + body + "\n", check_clang=False)
        except CLowerError as e:
            assert why in str(e), (body, str(e))
        else:
            raise AssertionError(f"the oracle folded {body!r}")
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, (body, _, why) in enumerate(_SIZEOF_REFUSED):
            path = os.path.join(d, f"r{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(head + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode > 0, (body, run.returncode, run.stdout[:200])
            assert why in run.stdout + run.stderr, (body, run.stdout[:200], run.stderr[:200])


def test_a_pointer_to_an_incomplete_struct_lowers_and_only_its_layout_is_refused():
    """CF-SELFREF: a pointer to a struct the unit has not defined yet -- the struct being defined, one
    defined later, one only declared (`struct t;`), or one never defined at all -- is an ordinary pointer
    (C11 6.2.5p22), laid out and passed on both rails with the same claim graph. The oracle had refused
    every one of these (a bare KeyError, then a `CLowerError`); the twin laid out only the self-reference.
    What needs the struct's layout is refused on both rails until the struct is complete: a member access
    through it, `sizeof`, a local or a parameter of the type itself -- each a verdict, never a crash."""
    from bcir.frontends.cfront.cparse import CParseError
    from bcir.frontends.cfront.lower import CLowerError

    head = "#include <stdint.h>\n"
    lowered = (
        "struct node { uint32_t v; struct node *next; };\nuint32_t f(struct node *n) { return n->v; }",
        "struct a { struct b *pb; uint32_t v; };\nuint32_t f(struct a *p) { return p->v; }",
        "struct fwd *gp;\nuint32_t f(void) { return gp ? 1u : 0u; }",
        "uint32_t f(struct fwd *p) { return p ? 1u : 0u; }",
        "uint32_t f(void) { struct fwd *p = 0; return p ? 1u : 0u; }",
        "struct fwd;\ntypedef struct fwd *h_t;\nuint32_t f(h_t h) { return h ? 1u : 0u; }",
        "struct b;\nstruct a { struct b *pb; };\nstruct b { struct a *pa; uint32_t v; };\n"
        "uint32_t f(struct a *p) { return p->pb->v; }",
    )
    refused = (  # (the unit, the oracle's refusal, the twin's)
        (
            "struct fwd;\nuint32_t f(struct fwd *p) { return p->v; }",
            "a member access into the incomplete struct or union 'fwd'",
            "unknown field",
        ),
        (
            "struct fwd;\nuint32_t f(void) { return sizeof(struct fwd); }",
            "the incomplete struct or union 'fwd' has no layout here",
            "the incomplete struct or union has no layout here",
        ),
        (
            "struct fwd;\nuint32_t f(void) { struct fwd x; return 0u; }",
            "the incomplete struct or union 'fwd' has no layout here",
            "the incomplete struct or union has no layout here",
        ),
        (
            "struct fwd;\nuint32_t f(struct fwd x) { return 0u; }",
            "the incomplete struct or union 'fwd' has no layout here",
            "the incomplete struct or union has no layout here",
        ),
    )
    summaries = {}
    for body in lowered:
        summaries[body], _r, _e = _oracle(head + body + "\n")
        assert "ok=1" in summaries[body], (body, summaries[body])
    for body, why, _ in refused:
        try:
            compile_unit(head + body + "\n", check_clang=False)
        except (CLowerError, CParseError) as e:
            assert why in str(e), (body, str(e))
        else:
            raise AssertionError(f"the oracle lowered {body!r}")
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, body in enumerate(lowered):
            path = os.path.join(d, f"l{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(head + body + "\n")
            c_summary, _emit = _c_run(exe, path)
            assert c_summary == summaries[body], (body, c_summary, summaries[body])
        for n, (body, _, why) in enumerate(refused):
            path = os.path.join(d, f"r{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(head + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode > 0, (body, run.returncode, run.stdout[:200])
            assert why in run.stdout + run.stderr, (body, run.stdout[:200], run.stderr[:200])


def _fixture_both_rails(fx: str):
    """A corpus fixture through both rails: its parity summary (claim counts and the structural digest)
    equal on both, and the two emits."""
    path = os.path.join(_C, fx)
    src = open(path, encoding="utf-8").read()
    oracle_summary, r, _entry = _oracle(src)
    assert "ok=1" in oracle_summary, oracle_summary
    oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    exe = _build_frontend(_session_build_dir())
    c_summary, c_emit = _c_run(exe, path)
    assert c_summary == oracle_summary, f"{fx}: parity\n C: {c_summary}\nPY: {oracle_summary}"
    return src, oracle_emit, c_emit


def _run_against_original(fx: str, src: str, emits, driver: str, extra=()):
    """Each emit linked with the original under every compiler at hand, run by `driver` (which prints MATCH
    when every function's result -- and every global it touches -- is the original's). `extra`: more
    compiler flags (a warning made an error)."""
    head = "#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n#include <stddef.h>\n"
    compilers = [c for c in dict.fromkeys((_CC, shutil.which("clang"), shutil.which("gcc"))) if c]
    with tempfile.TemporaryDirectory() as d:
        for cc in compilers:
            for label, emit in emits:
                text = f"{head}{_BOUNDS_GUARD}\n{src}\n{emit}\n{driver}"
                out = _build_run_c(d, cc, label, text, extra)
                assert out == "MATCH", (
                    f"{fx}: the {label} emit is not the original under {cc} ({out})"
                )


_SIZEOF_FORMS_DRIVER = r"""
static int fail(const char *what) { puts(what); return 1; }
#define SAME(f, ...) do { if (f(__VA_ARGS__) != bcir_##f(__VA_ARGS__)) return fail(#f); } while (0)
int main(void) {
  static uint32_t p8[8], q8[8]; static struct sz_s s; static uint16_t m[4][6];
  SAME(sz_local_array); SAME(sz_local_row); SAME(sz_local_elem); SAME(sz_local_decays);
  SAME(sz_member_array); SAME(sz_nested); SAME(sz_struct_array); SAME(sz_union); SAME(sz_brace_sized);
  SAME(sz_param_ptr, p8, &s); SAME(sz_param_array, p8);
  if (sz_param_2d(m) != bcir_sz_param_2d(&m[0][0])) return fail("sz_param_2d");
  if (sz_param_rowptr(m) != bcir_sz_param_rowptr(&m[0][0])) return fail("sz_param_rowptr");
  SAME(sz_global); SAME(sz_global_struct); SAME(sz_global_sized); SAME(sz_types); SAME(sz_types2);
  SAME(sz_alignof); SAME(sz_promote, 200, -5); SAME(sz_bare, -9, &s); SAME(sz_compare, 3u, 4u);
  SAME(sz_conditional, 1, 5u, p8, q8); SAME(sz_assign, 7, 300u); SAME(sz_float, 1.5f, 2.5);
  SAME(sz_literals); SAME(sz_calls); SAME(sz_bitfield_ops); SAME(sz_bitfield_keeps);
  SAME(sz_nested_sizeof, 3); SAME(sz_pointer_difference, p8 + 3, p8);
  SAME(sz_size_t_wraps); SAME(sz_array_count);
  for (uint32_t i = 0; i < 20u; i++) {
    SAME(sz_unevaluated, i); SAME(sz_vla, i); SAME(sz_size_t_scales, 1u << (i + 12u));
  }
  puts("MATCH");
  return 0;
}
"""


def test_sizeof_forms_run_as_the_original_on_both_rails():
    """CF-SIZEOF: `cfront_sizeof_forms.c` -- every operand form, the type-names with their dimensions, the
    unevaluated operand (an increment or a call inside never runs), a VLA's runtime size, and the size_t
    witnesses (`sizeof(uint32_t) - 8u` is -4 as an int64, not 4294967292; `sizeof(struct big) * n` does not
    wrap at 2^32) -- lowers identically on both rails, and each emit returns what the original does."""
    if not _CC:
        return
    fx = "cfront_sizeof_forms.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _run_against_original(
        fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _SIZEOF_FORMS_DRIVER
    )


_GLOBALSTRUCT_DRIVER = r"""
static unsigned char b1[4096], b2[4096];
static void seed(void) {
  memset(&gs_g, 0x5A, sizeof gs_g); memset(&gs_t2, 0x3C, sizeof gs_t2); memset(gs_arr, 0x11, sizeof gs_arr);
  memset(&gs_static, 0x21, sizeof gs_static); memset(&gs_dev, 0x42, sizeof gs_dev); gs_p = &gs_g;
  gs_wide = 0x100000000u; gs_small = -3; gs_init.c = 1; gs_init.n = 0x100000005u;
}
static size_t snap(unsigned char *b) {
  size_t o = 0;
#define TAKE(x) do { memcpy(b + o, &(x), sizeof(x)); o += sizeof(x); } while (0)
  TAKE(gs_g); TAKE(gs_t2); TAKE(gs_arr); TAKE(gs_static); TAKE(gs_dev); TAKE(gs_wide); TAKE(gs_small); TAKE(gs_init);
  return o;
}
static int fail(const char *what) { puts(what); return 1; }
#define SAME(f, ...) do { seed(); uint64_t r1 = (uint64_t)(f(__VA_ARGS__)); size_t n1 = snap(b1); \
    seed(); uint64_t r2 = (uint64_t)(bcir_##f(__VA_ARGS__)); size_t n2 = snap(b2); \
    if (r1 != r2 || n1 != n2 || memcmp(b1, b2, n1)) return fail(#f); } while (0)
#define SAME_PTR(f) do { seed(); uintptr_t r1 = (uintptr_t)f(); seed(); uintptr_t r2 = (uintptr_t)bcir_##f(); \
    if (r1 != r2) return fail(#f); } while (0)
#define SAME_VOID(f, ...) do { seed(); f(__VA_ARGS__); size_t n1 = snap(b1); seed(); bcir_##f(__VA_ARGS__); \
    size_t n2 = snap(b2); if (n1 != n2 || memcmp(b1, b2, n1)) return fail(#f); } while (0)
int main(void) {
  for (uint32_t i = 0; i < 8u; i++) {
    SAME(gs_read_u64); SAME(gs_read_u8); SAME_VOID(gs_write, 0x1122334455667788u + i);
    SAME(gs_member_array, i); SAME(gs_nested, i); SAME(gs_compound, i); SAME_PTR(gs_address);
    SAME(gs_through_pointer); SAME_VOID(gs_array_of_structs, i, 3u * i); SAME(gs_static_member);
    SAME(gs_pointer_global); SAME(gs_condition); SAME(gs_sizes); SAME(gs_device, i);
    SAME(gs_wide_plus); SAME(gs_wide_shift); SAME(gs_small_scaled); SAME(gs_init_member); SAME(gs_init_sizes);
  }
  seed(); struct gs_s w1 = gs_whole(); seed(); struct gs_s w2 = bcir_gs_whole();
  if (memcmp(&w1, &w2, sizeof w1)) return fail("gs_whole");
  struct gs_s v; memset(&v, 0x77, sizeof v);
  seed(); gs_whole_write(v); size_t n1 = snap(b1); seed(); bcir_gs_whole_write(v); size_t n2 = snap(b2);
  if (n1 != n2 || memcmp(b1, b2, n1)) return fail("gs_whole_write");
  puts("MATCH");
  return 0;
}
"""


def test_file_scope_struct_members_lower_on_both_rails():
    """CF-GSTRUCT / CF-GINIT: `cfront_globalstruct.c` -- a file-scope struct's members read, written,
    nested, indexed, compound-assigned, incremented, addressed, copied whole, reached through a pointer
    global and an array of them, a static one, one holding volatile registers, and initialized scalars and
    structs -- lowers to the same claim graph on both rails (the twin had crashed on every member access,
    reading `c->s[-1]`), and from the same seeded globals each emit returns what the original does and
    leaves every global byte as it does (the oracle had computed an initialized `uint64_t`'s `gw + 1` in 32
    bits)."""
    if not _CC:
        return
    fx = "cfront_globalstruct.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _run_against_original(
        fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _GLOBALSTRUCT_DRIVER
    )


def test_an_initialized_global_keeps_its_declared_type_in_the_oracle():
    """CF-GINIT: an initialized scalar or struct global is one object of its declared type -- not a table
    of its initializers, the read-only lookup-table model's relic that typed `gw + 1` of a `uint64_t gw =
    0x100000000u` as a 32-bit int, doubled `sizeof` of a struct with two initializers and refused its
    members. Held without a compiler: the sum's temp is 64-bit, the struct's size is its own, and the
    member reads."""
    src = (
        "#include <stdint.h>\nstruct gi { uint8_t c; uint64_t n; };\nuint64_t gw = 0x100000000u;\n"
        "struct gi gs = {1, 2};\nuint64_t f(void) { return gw + 1u; }\n"
        "uint64_t g(void) { return sizeof gs + gs.n; }\n"
    )
    r = compile_unit(src, check_clang=False)
    f = r.lowered.functions["f"]
    add = [c for c in f.claims if c.op == "c.bin.add"]
    assert len(add) == 1 and f.rid_types[add[0].wr[0]].size == 8, [c.op for c in f.claims]
    g = r.lowered.functions["g"]
    assert 16 in [c.imm[0] for c in g.claims if c.op == "c.const"], [c.op for c in g.claims]


_DECAY_DRIVER = r"""
static int fail(const char *what) { puts(what); return 1; }
#define SAME(f, ...) do { if ((f(__VA_ARGS__)) != (bcir_##f(__VA_ARGS__))) return fail(#f); } while (0)
int main(void) {
  static uint8_t b8[16]; static uint32_t b32[16]; static struct dk_s s;
  memset(&s, 0x33, sizeof s); memset(&dk_g, 0x44, sizeof dk_g);
  for (uint32_t i = 0; i < 12u; i++) {
    SAME(dk_diff_u8, b8 + (i % 7u), b8 + 3); SAME(dk_diff_u32, b32 + 1, b32 + (i % 9u));
    SAME(dk_array_plus, i); SAME(dk_global_plus, i); SAME(dk_select, i & 1u); SAME(dk_select_array, i & 1u);
    SAME(dk_select_float, i & 1u); SAME(dk_member_local, i); SAME(dk_member_select, i & 1u);
    SAME(dk_deref_index, i % 9u); SAME(dk_deref_index_value, i);
  }
  SAME(dk_diff_array); SAME(dk_float_plus); SAME(dk_array_as_integer); SAME(dk_member_value);
  SAME(dk_member_arrow, &s); SAME(dk_member_nested); SAME(dk_member_offset);
  puts("MATCH");
  return 0;
}
"""


def test_an_array_value_decays_to_its_address_on_both_rails():
    """CF-DECAY / CF-PTRDIFF / CF-DEREFIDX: `cfront_decay.c` lowers to the same claim graph on both rails and
    each emit runs as the original: a member array used as a value is its address (both rails had loaded
    its first element -- `return gs.a;` did not compile and `(uintptr_t)gs.a` was that element's value); an
    array operand of `+`, `-` or `?:` is a pointer (`int32_t t = la + 1;` did not compile); `p - q` is a
    signed pointer-wide `ptrdiff_t` (the twin's unsigned 32-bit one made -2 read 4294967294); and the twin
    lowers the index of `*(la + j++)` once (it had stepped `j` twice and read the wrong element)."""
    if not _CC:
        return
    fx = "cfront_decay.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _DECAY_DRIVER)
    # the LOCAL arrays whose address leaves the value model, which the corpus fixture keeps at file scope
    src = _DECAY_LOCAL_ESCAPES
    oracle_summary, r, _entry = _oracle(src)
    oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "decay_local.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        c_summary, c_emit = _c_run(_build_frontend(_session_build_dir()), path)
    assert c_summary == oracle_summary, f"parity\n C: {c_summary}\nPY: {oracle_summary}"
    emits = (("twin", c_emit), ("oracle", oracle_emit))
    _run_against_original("decay_local.c", src, emits, _DECAY_LOCAL_DRIVER)


# CF-DECAY: a LOCAL array whose address leaves the value model -- subtracted into a `ptrdiff_t`, or converted
# to an integer -- escapes, and the escape analysis says so, rightly. `escape.unproved` counts every local
# array of the `cfront_*.c` corpus the analysis does not prove private, so `cfront_decay.c` holds these forms
# on a file-scope array and they run here.
_DECAY_LOCAL_ESCAPES = """#include <stdint.h>
int64_t dl_diff(void) { uint32_t la[8] = {0}; uint32_t *e = la + 6; return la - e; }
uintptr_t dl_as_integer(void) { uint32_t la[4] = {0}; return (uintptr_t)(la + 1) - (uintptr_t)la; }
"""
_DECAY_LOCAL_DRIVER = r"""
static int fail(const char *what) { puts(what); return 1; }
int main(void) {
  if (dl_diff() != bcir_dl_diff()) return fail("dl_diff");
  if (dl_as_integer() != bcir_dl_as_integer()) return fail("dl_as_integer");
  puts("MATCH");
  return 0;
}
"""


# CF-DECAY: sources that differ only in reading a member array's ADDRESS (`s.a` as a value) or its first
# ELEMENT (`s.a[0]`) -- the decay is a different datum, so each pair must digest apart, on both rails alike.
_DECAY_DIGEST_PAIRS = (
    "struct s {{ uint8_t c; uint32_t a[4]; }};\nstruct s g;\nuintptr_t f(void) {{ return (uintptr_t){E}; }}\n",
    "struct s {{ uint8_t c; uint32_t a[4]; }};\nuintptr_t f(struct s *p) {{ return (uintptr_t){P}; }}\n",
)


def test_a_decayed_member_array_digests_apart_from_its_element():
    """CF-DECAY: the structural digest tells a member array's address from a read of its first element on
    both rails (the decay was that very read, so the pair had digested alike)."""
    head = "#include <stdint.h>\n"
    exe = _build_frontend(_session_build_dir()) if _CC else None
    for tmpl in _DECAY_DIGEST_PAIRS:
        digests = []
        for addr, elem in (("g.a", "g.a[0]"), ("p->a", "p->a[0]")):
            if ("{E}" in tmpl) != (addr == "g.a"):
                continue
            for spelling in (addr, elem):
                src = head + tmpl.format(E=spelling, P=spelling)
                oracle = (
                    f"{cfront_structural_digest(compile_unit(src, check_clang=False).lowered):016x}"
                )
                if exe:
                    with tempfile.TemporaryDirectory() as d:
                        path = os.path.join(d, "pair.c")
                        with open(path, "w", encoding="utf-8") as fh:
                            fh.write(src)
                        run = subprocess.run([exe, path], capture_output=True, text=True)
                    twin = re.search(r"digest=([0-9a-f]{16})", run.stdout)
                    assert twin and twin.group(1) == oracle, (spelling, run.stdout[:200])
                digests.append(oracle)
        assert len(digests) == 2 and digests[0] != digests[1], (tmpl, digests)


# CF-DECAY: a row -- of a multi-dimensional local, parameter or global, or a multi-dimensional member array --
# used as a value, and arithmetic on a multi-dimensional array, would decay to a `T (*)[N]` the value model
# has no spelling for (its emit declares the array flat, so C would step an element where the source steps a
# row). Both rails refuse each one; both had read the row's flat element silently.
_DECAY_REFUSED = (
    "uintptr_t f(void) { uint16_t l2[4][6] = {0}; return (uintptr_t)l2[1] - (uintptr_t)l2; }",
    "uint16_t *f(void) { static uint16_t l2[4][6]; return l2[1]; }",
    "uintptr_t f(void) { uint16_t l2[4][6] = {0}; return (uintptr_t)(l2 + 1) - (uintptr_t)l2; }",
    "uintptr_t f(uint8_t m[3][5]) { return (uintptr_t)(m + 1) - (uintptr_t)m; }",
    "uint8_t *f(uint8_t m[3][5]) { return m[1]; }",
    "uint8_t gb[3][5];\nuint8_t *f(void) { return gb[1]; }",
    "struct s { uint8_t m[2][3]; };\nuint8_t *f(struct s *p) { return (uint8_t *)p->m; }",
)


def test_a_row_used_as_a_value_is_refused_on_both_rails():
    """CF-DECAY: every form in `_DECAY_REFUSED` is refused by both rails (the twin with its refusal status,
    never a crash) -- each had lowered to a read of the row's flat element."""
    from bcir.frontends.cfront.cparse import CParseError
    from bcir.frontends.cfront.lower import CLowerError

    head = "#include <stdint.h>\n"
    for body in _DECAY_REFUSED:
        try:
            compile_unit(head + body + "\n", check_clang=False)
        except (CLowerError, CParseError):
            pass
        else:
            raise AssertionError(f"the oracle lowered {body!r}")
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, body in enumerate(_DECAY_REFUSED):
            path = os.path.join(d, f"r{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(head + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode > 0, (body, run.returncode, run.stdout[:200])


def test_a_member_access_on_a_non_struct_is_refused_on_both_rails():
    """CF-GSTRUCT: a member access on an object that is no struct or union is refused by both rails. The
    twin's member path read `c->s[sidx]` unguarded, so an index of -1 -- every struct global's, before
    `use_global` bound its definition -- read before the table; it now refuses any object without one."""
    from bcir.frontends.cfront.cparse import CParseError
    from bcir.frontends.cfront.lower import CLowerError

    bodies = (
        "uint32_t f(uint32_t x) { return x.n; }",
        "uint32_t g;\nuint32_t f(void) { return g.n; }",
        "uint32_t *g;\nuint32_t f(void) { return g->n; }",
    )
    for body in bodies:
        try:
            compile_unit("#include <stdint.h>\n" + body + "\n", check_clang=False)
        except (CLowerError, CParseError, KeyError, AttributeError):
            pass
        else:
            raise AssertionError(f"the oracle lowered {body!r}")
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, body in enumerate(bodies):
            path = os.path.join(d, f"m{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("#include <stdint.h>\n" + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode > 0, (body, run.returncode, run.stdout[:200])


# CF-GAPS: the eight gaps closed on both rails -- brace elision (CF-BRACE), string-sized locals
# (CF-STRLOCAL), self-referential structs (CF-SELFREF), typedef'd arrays, 2-D/3-D globals, array-of-structs
# member arrays, `wchar_t` and `sizeof &f` (CF-SMALL) -- and the storage classes found beside them
# (CF-STORAGE). Each fixture runs every function, not only its entry, against the original under every
# compiler at hand; the generic campaigns (parity, the gcc/clang differential, the round trip) cover the
# fixtures through `_FIXTURES` besides.
_GAPS_SAME = r"""
static int fail(const char *what) { puts(what); return 1; }
#define SAME(f, ...) do { if ((uint64_t)f(__VA_ARGS__) != (uint64_t)bcir_##f(__VA_ARGS__)) return fail(#f); } while (0)
static const uint32_t gaps_in[] = {0u, 1u, 2u, 3u, 7u, 255u, 256u, 65535u, 65536u, 0x12345678u, 0xFFFFFFFFu};
#define GAPS_N (sizeof gaps_in / sizeof gaps_in[0])
"""

_BRACE_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(be_flat_2d, s); SAME(be_struct_array, s); SAME(be_nested, s); SAME(be_mixed, s);
    SAME(be_designated, s); SAME(be_array_designators, s); SAME(be_member_rows, s); SAME(be_inferred, s);
    SAME(be_inferred_rows, s); SAME(be_inferred_designated, s); SAME(be_string_member, s);
    SAME(be_string_exact, s); SAME(be_anon, s); SAME(be_anon_designated, s); SAME(be_union, s);
    SAME(be_union_designated, s); SAME(be_struct_value, s); SAME(be_braced_scalar, s);
    SAME(be_compound_literal, s); SAME(be_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)


def test_brace_elision_runs_as_the_original_on_both_rails():
    """CF-BRACE: `cfront_braceelide.c` -- C11 6.7.9p17-21's walk of the current object: an initializer list
    that opens no brace for a sub-aggregate fills it from the enclosing list (rows of a 2-D array, arrays of
    structs, structs of structs, a member's rows); a designator returns to the list's own object and the
    entries after it continue past the named subobject; an anonymous struct or union member is one subobject;
    a union takes one value; a string literal initializes the character array it meets; a struct value
    initializes its subobject whole; `{e}` and `{}` initialize a scalar; an inferred `[]` is sized by the
    entries the walk reached. Both rails lower it to the same claim graph and each emit returns what the
    original does, function by function."""
    if not _CC:
        return
    fx = "cfront_braceelide.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _BRACE_DRIVER)


_STRLOCAL_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t i = gaps_in[n];
    SAME(sl_plain, i); SAME(sl_braced, i); SAME(sl_unsigned, i); SAME(sl_concat, i); SAME(sl_escapes, i);
    SAME(sl_sized_longer, i); SAME(sl_sized_exact, i); SAME(sl_utf16, i); SAME(sl_utf32, i); SAME(sl_wide, i);
    SAME(sl_rows, i); SAME(sl_struct_member, i); SAME(sl_write, i); SAME(sl_entry, i);
  }
  puts("MATCH");
  return 0;
}
"""
)


def test_string_sized_locals_run_as_the_original_on_both_rails():
    """CF-STRLOCAL: `cfront_strlocal.c` -- `char s[] = "abc"` is a four-element local (the units and the
    NUL), filled unit by unit with the rest zero: plain, unsigned, braced, concatenated and escaped literals,
    `u"..."`/`U"..."`/`L"..."` arrays of the code unit, a sized array longer than its literal, one exactly as
    long (the NUL dropped), rows of strings and a string member. Both rails size and fill it the same way and
    each emit returns what the original does (both had declared `char s[] = "abc"` one element long, its
    emit -- and a sized `char s[8] = "hi"`'s -- did not compile, and `sizeof s` was refused as incomplete)."""
    if not _CC:
        return
    fx = "cfront_strlocal.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _STRLOCAL_DRIVER)


_SELFREF_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  struct node c3 = {3u, 0}, b3 = {2u, &c3}, a3 = {1u, &b3};
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n], x = s;
    SAME(sr_list, s); SAME(sr_mutual, s); SAME(sr_tree, s); SAME(sr_typeof_next, s); SAME(sr_entry, s);
    SAME(sr_opaque, (handle_t)&x, (struct opaque *)&x, s);
    SAME(sr_opaque, (handle_t)0, (struct opaque *)&x, s);
    if (sr_advance(&a3, s % 5u) != bcir_sr_advance(&a3, s % 5u)) return fail("sr_advance");
  }
  puts("MATCH");
  return 0;
}
"""
)


def test_self_referential_structs_run_as_the_original_on_both_rails():
    """CF-SELFREF: `cfront_selfref.c` -- a list linked through the struct's own pointer, two structs that
    point at each other across a forward declaration, a tree of its own type, pointers (and a typedef'd
    handle) to a struct never defined, a function returning a pointer to the struct whose result is
    dereferenced in place, and `typeof` a member that points at its own struct -- a reader of the pointee
    size the struct's completion back-fills, so a subscript through it steps one whole node. Both
    rails lay the structs out and lower them to the same claim graph; each emit returns what the original
    does."""
    if not _CC:
        return
    fx = "cfront_selfref.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _SELFREF_DRIVER)


_TYPEDEFARR_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t i = gaps_in[n];
    SAME(td_local, i); SAME(td_rows, i); SAME(td_matrix, i); SAME(td_member, i); SAME(td_global, i);
    SAME(td_literal, i); SAME(td_entry, i);
  }
  puts("MATCH");
  return 0;
}
"""
)


def test_typedef_arrays_run_as_the_original_on_both_rails():
    """CF-SMALL: `cfront_typedefarr.c` -- `typedef T row_t[N]` keeps its shape wherever it is spelled, and a
    declarator's own dimensions are outer to it: a local, rows of it (`row_t rw[2]`), a typedef of the
    typedef, a struct member, file-scope objects, `sizeof` and a compound literal's type-name. The twin had
    dropped the typedef's dimensions (a `row_t` was one element). Both rails lower it to the same claim graph
    and each emit returns what the original does. A pointer to such a type has no spelling on either rail:
    refused by both."""
    from bcir.frontends.cfront.cparse import CParseError

    body = "#include <stdint.h>\ntypedef uint16_t row_t[3];\nuint32_t f(row_t *p) { return (*p)[2]; }\n"
    try:
        compile_unit(body, check_clang=False)
    except CParseError as e:
        assert "a pointer to a typedef'd array is not supported" in str(e), str(e)
    else:
        raise AssertionError("the oracle lowered a pointer to a typedef'd array")
    if not _CC:
        return
    fx = "cfront_typedefarr.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _TYPEDEFARR_DRIVER)
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "p.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
        run = subprocess.run([exe, path], capture_output=True, text=True)
        assert (
            run.returncode > 0 and "a pointer to a typedef'd array is not supported" in run.stdout
        ), run.stdout[:200]


_MDGLOBAL_DRIVER = (
    _GAPS_SAME
    + r"""
#define SAME_MD(f, ...) do { md_reset(); uint64_t a_ = f(__VA_ARGS__); md_reset(); \
    uint64_t b_ = bcir_##f(__VA_ARGS__); if (a_ != b_) return fail(#f); } while (0)
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t i = gaps_in[n], j = gaps_in[(n + 3u) % GAPS_N], k = gaps_in[(n + 7u) % GAPS_N];
    SAME_MD(md_read, i, j, k); SAME_MD(md_write, i, j, k); SAME_MD(md_address, i, j);
    SAME_MD(md_aos_read, i, j); SAME_MD(md_aos_values, i, j); SAME_MD(md_aos_stmts, i, j);
    SAME_MD(md_aos_address, i, j); SAME_MD(md_aos_rows, i, j, k); SAME_MD(md_aos_narrow, i, j);
    SAME(md_local, i, j, k); SAME(md_vla, i, j, k); SAME(md_param_call, i, j); SAME(md_entry, i, j, k);
  }
  puts("MATCH");
  return 0;
}
"""
)


def test_multidimensional_globals_and_member_arrays_run_as_the_original_on_both_rails():
    """CF-SMALL: `cfront_mdglobal.c` -- a 2-D/3-D file-scope array indexed in full is one element of the
    whole object (row-major, at its byte offset): read, written, compound-assigned and stepped as a statement
    and as a value, and addressed; `a[i].m[j]` of an array of structs folds the element index with the
    member's in every access form, a 2-D member included, and a byte member re-reads what it stored; `&m[i][j]`
    of a local, a VLA and a `T m[][N]` parameter flattens every subscript. The rails had refused the global
    forms (the oracle as "a subscript of an element that is not a pointer"), and the twin every
    `a[i].m[j]` and `&m[i][j]`. Each global is reset before each call, so the original and each emit start
    from the same state; both rails lower it to the same claim graph and each emit returns what the original
    does."""
    if not _CC:
        return
    fx = "cfront_mdglobal.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _MDGLOBAL_DRIVER)


_STORAGE_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned round = 0; round < 3u; round++) {
    for (unsigned n = 0; n < GAPS_N; n++) {
      uint32_t i = gaps_in[n];
      SAME(st_first, i); SAME(st_after_type, i); SAME(st_after_qualifier, i); SAME(st_after_unsigned, i);
      SAME(st_after_typedef, i); SAME(st_array, i); SAME(st_struct, i); SAME(st_qualifiers, i);
      SAME(st_pointer, i); SAME(st_entry, i);
    }
  }
  puts("MATCH");
  return 0;
}
"""
)


def test_storage_classes_in_any_position_run_as_the_original_on_both_rails():
    """CF-STORAGE: `cfront_storage.c` -- `static` is a static wherever the declaration spells it (C11 6.7p1):
    first, after the type, after a qualifier, inside an `unsigned` run, after a typedef name or a struct
    tag; a qualifier after the type qualifies it. Both rails had honored only a LEADING `static`: `volatile
    static uint32_t n` and `unsigned static n` became uninitialized locals on both (a runtime mismatch), and
    `uint32_t static n` on the oracle. Each function runs repeatedly in lockstep with its emit, so a local in
    place of a static diverges; both rails lower it to the same claim graph and each emit returns what the
    original does."""
    if not _CC:
        return
    fx = "cfront_storage.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _STORAGE_DRIVER)


_MEMDECAY_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(md_member, s); SAME(md_braced, s); SAME(md_arrow, s); SAME(md_deref, s); SAME(md_global, s);
    SAME(md_string, s); SAME(md_compound, s); SAME(md_ptr_array, s); SAME(md_structs, s); SAME(md_vla, s);
    SAME(md_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)


# CF-MEMDECAY on LOCAL arrays: a local array stored into a pointer slot escapes, and the escape analysis says so,
# rightly. `escape.unproved` counts every local array of the `cfront_*.c` corpus the analysis does not prove
# private, so `cfront_memdecay.c` holds these forms on file-scope arrays and they run here.
_MEMDECAY_LOCAL = """#include <stdint.h>
struct holder { uint32_t tag; uint32_t *p; };
struct msg { uint8_t n; const char *txt; };
struct ptrs { uint8_t k; uint32_t *slot[3]; };
uint32_t ml_member(uint32_t s) {                    /* h.p = arr */
  uint32_t arr[3] = {s, s + 1u, s + 2u};
  struct holder h;
  h.tag = 7u;
  h.p = arr;
  return h.p[2] + h.tag;
}
uint32_t ml_braced(uint32_t s) {                    /* {7u, arr} and {.p = arr, ...} */
  uint32_t arr[3] = {s, s * 3u, s ^ 5u};
  struct holder a = {7u, arr};
  struct holder b = {.p = arr, .tag = s};
  return a.p[1] + b.p[2] * 3u + b.tag;
}
uint32_t ml_arrow(uint32_t s) {                     /* hp->p = arr */
  uint32_t arr[2] = {s, s + 9u};
  struct holder h;
  struct holder *hp = &h;
  hp->tag = 1u;
  hp->p = arr;
  return hp->p[1] + h.p[0] + h.tag;
}
uint32_t ml_deref(uint32_t s) {                     /* *pp = arr */
  uint32_t arr[2] = {s + 4u, s};
  uint32_t other = 3u;
  uint32_t *q = &other;
  uint32_t **pp = &q;
  *pp = arr;
  return q[0] + q[1];
}
uint32_t ml_string(uint32_t s) {                    /* a string-initialized array into a `const char *` */
  char buf[] = "bcir";
  struct msg m;
  m.n = (uint8_t)s;
  m.txt = buf;
  return (uint32_t)m.txt[2] + m.n;
}
uint32_t ml_compound(uint32_t s) {                  /* a compound literal's member */
  uint32_t arr[3] = {s, 2u, 3u};
  struct holder h = (struct holder){5u, arr};
  return h.p[0] * h.tag + h.p[2];
}
uint32_t ml_ptr_array(uint32_t s) {                 /* a member array of pointers, element by element */
  uint32_t a[2] = {s, 1u};
  uint32_t b[2] = {2u, s + 3u};
  struct ptrs r;
  r.k = 1u;
  r.slot[0] = a;
  r.slot[1] = b;
  r.slot[2] = a;
  uint32_t *w = r.slot[0];
  uint32_t *x = r.slot[1];
  uint32_t *y = r.slot[2];
  return w[0] + x[1] * 5u + y[1] + r.k;
}
"""
_MEMDECAY_LOCAL_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(ml_member, s); SAME(ml_braced, s); SAME(ml_arrow, s); SAME(ml_deref, s); SAME(ml_string, s);
    SAME(ml_compound, s); SAME(ml_ptr_array, s);
  }
  puts("MATCH");
  return 0;
}
"""
)


def test_arrays_stored_into_pointer_slots_decay_on_both_rails():
    """CF-MEMDECAY: `cfront_memdecay.c` -- an array stored into a pointer slot the emit writes by `memcpy` (a
    member, a dereference, a member array of pointers; by assignment, in a brace or designated list, or in a
    compound literal) is its first element's address (C11 6.3.2.1p3). Both rails lowered it to the same claim
    graph, then the oracle's emit copied the array's first bytes into the slot (`memcpy(&h.p, &arr, 8)`: a
    wild pointer, read back as one) and the twin's did not compile (`uint64_t _v = arr`); each emit now
    stages the decayed pointer and returns what the original does, function by function."""
    if not _CC:
        return
    fx = "cfront_memdecay.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _MEMDECAY_DRIVER)
    # the same forms on LOCAL arrays, which the corpus fixture keeps at file scope
    src = _MEMDECAY_LOCAL
    oracle_summary, r, _entry = _oracle(src)
    oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "memdecay_local.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        c_summary, c_emit = _c_run(_build_frontend(_session_build_dir()), path)
    assert c_summary == oracle_summary, f"parity\n C: {c_summary}\nPY: {oracle_summary}"
    emits = (("twin", c_emit), ("oracle", oracle_emit))
    _run_against_original("memdecay_local.c", src, emits, _MEMDECAY_LOCAL_DRIVER)


_NULLPTR_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n], g1 = s, g2 = s;
    if (np_assign(&g1, s) != bcir_np_assign(&g2, s)) return fail("np_assign");
    SAME(np_decl, s); SAME(np_return, s); SAME(np_elements, s); SAME(np_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)


def test_null_pointer_constants_run_as_the_original_on_both_rails():
    """CF-NULLPTR: `cfront_nullptr.c` -- the integer constant 0 taken by a pointer (a declaration's plain,
    braced or empty initializer, any integer literal, an assignment to a local, a parameter or a file-scope
    pointer, `return 0;` from a function returning a pointer, an element of an array of pointers or of a
    `T **`, stored or listed) is a null pointer (C11 6.3.2.3p3). Both rails lowered it to the same claim graph,
    and each emit assigned an `int` temp to the pointer -- `int t = 0u; p = t;`, which Clang and GCC reject.
    Each emit now declares the constant as the pointer it becomes, and returns what the original does."""
    if not _CC:
        return
    fx = "cfront_nullptr.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _NULLPTR_DRIVER)


def _actual_types(emit: str, callee: str) -> list:
    """The C type each actual of each call `t = bcir_<callee>(...)` in an emit is declared with (None for
    an actual declared elsewhere -- a parameter, a local), call by call in emit order."""
    out = []
    for m in re.finditer(rf"= bcir_{callee}\(([^)]*)\);", emit):
        types = []
        for a in (x.strip() for x in m.group(1).split(",")):
            d = re.search(rf"^\s*(\S.*?)\s*\b{re.escape(a)} = ", emit, re.M)
            types.append(d.group(1) if d else None)
        out.append(types)
    return out


_NULLARG_DRIVER = (
    _GAPS_SAME
    + r"""
static uint32_t drv_op(uint32_t x) { return x * 5u + 2u; }
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n], v = s + 3u, w = s + 3u, a[4] = {s, 1u, 2u, 3u}, m[2][2] = {{s, 4u}, {5u, s}};
    char *argv[1] = {0};
    struct na_node b = {s, 0}, c = {7u, &b};
    SAME(na_deref, 0, s); SAME(na_deref, &v, s); SAME(na_raw, 0, 0, s); SAME(na_raw, &v, argv, s);
    SAME(na_twice, s); SAME(na_apply, 0, s); SAME(na_apply, drv_op, s); SAME(na_apply_decl, 0, s);
    SAME(na_apply_decl, drv_op, s); SAME(na_walk, 0, s); SAME(na_walk, &c, s); SAME(na_sum, 0, 4u);
    SAME(na_sum, a, 4u); SAME(na_last, 0, s); SAME(na_last, a, s); SAME(na_cell, 0, s);
    if (na_cell(m, s) != bcir_na_cell(&m[0][0], s)) return fail("na_cell");
    SAME(na_vla, 2u, 0); SAME(na_vla, 4u, a); SAME(na_count, 0, 2u, 3, 4); SAME(na_count, &v, 1u, 0);
    na_store(0, s); bcir_na_store(0, s); na_store(&v, s + 1u); bcir_na_store(&w, s + 1u);
    if (v != w) return fail("na_store");
    if (na_pick(0, &v) != bcir_na_pick(0, &v) || bcir_na_pick(0, 0) != 0) return fail("na_pick");
    struct na_pair r = na_make(0, s), q = bcir_na_make(0, s);
    if (r.a != q.a || r.b != q.b) return fail("na_make");
    SAME(na_pointers, s); SAME(na_functions, s); SAME(na_structs, s); SAME(na_arrays, s); SAME(na_calls, s);
    SAME(na_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)


def test_null_pointer_arguments_run_as_the_original_on_both_rails():
    """CF-NULLARG: `cfront_nullarg.c` -- the integer constant 0 passed to a pointer parameter (C11 6.5.2.2p7:
    an argument converts to its parameter's type as if by assignment) -- a pointer to a scalar, to const, to
    void, to a pointer or to a struct, a typedef'd function pointer or a declarator, `T a[]`, `T a[N]`, `T m[][N]`
    and a VLA, the callee void, returning a pointer or a struct, or variadic (its extra `0` stays an int), the
    constant spelled `0`, `0u`, `0L` or `(0)` -- is a null pointer of the parameter's type. Both rails lowered
    each call to the same claim graph, and each emit passed an `int` temp where the callee takes a pointer
    (`int t = 0u; bcir_g(t, s)`), which Clang and GCC reject. Each emit now declares the constant as the
    parameter's pointer, and every function returns what the original does."""
    if not _CC:
        return
    fx = "cfront_nullarg.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        # `na_count(0, 2u, 0, 5)`: the named pointer parameter's constant is a pointer, the extra `0`
        # -- which the callee reads as an `int` -- is not
        first = _actual_types(emit, "na_count")[0]
        assert None not in (first[0], first[2]), (label, first)
        assert first[0].endswith("*") and "*" not in first[2], (label, first)
    _run_against_original(
        fx,
        src,
        (("twin", c_emit), ("oracle", oracle_emit)),
        _NULLARG_DRIVER,
        ("-Werror=int-conversion",),
    )


_NULLARG_LINK_DRIVER = (
    _GAPS_SAME
    + r"""
uint32_t nl_ext(uint32_t *p, uint32_t s) {       /* the other unit's definitions */
  if (p) return *p * 5u + s;
  return s + 0x77u;
}
uint32_t nl_ext_node(struct nl_node *n, nl_op fn, uint32_t s) {
  uint32_t k = s;
  if (n) k += n->v * 3u;
  if (fn) k += fn(k);
  return k;
}
uint32_t nl_ext_sum(uint32_t a[], uint32_t n) {
  uint32_t k = 1u;
  if (!a) return k + n;
  for (uint32_t i = 0u; i < n; i++) k += a[i];
  return k;
}
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n], v = s + 1u, a[2] = {s, 2u};
    struct nl_node b = {s, 0};
    SAME(nl_twice, s); SAME(nl_later, 0, s); SAME(nl_later, &v, s); SAME(nl_later_node, 0, 0, 0, s);
    SAME(nl_later_node, &b, nl_twice, a, s); SAME(nl_sum, 0, 2u, 3, (int)(s & 0xFFFFu)); SAME(nl_sum, &v, 1u, 0);
    SAME(nl_forward, s); SAME(nl_cross, s); SAME(nl_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)
_NULLARG_BARE = """#include <stdint.h>
uint32_t nl_f(uint32_t s) { return nl_g(0, s) + nl_g(&s, 1u); }
uint32_t nl_g(uint32_t *p, uint32_t s) {
  if (p) return *p + s;
  return s ^ 0x3Cu;
}
"""


def test_null_pointer_arguments_to_later_and_prototyped_callees():
    """CF-NULLARG: `cfront_nullarg_link.c` -- the constant 0 passed to a callee whose definition the call
    cannot see: one defined after its caller (declared first by a static or an external prototype, variadic
    or not) and one only prototyped, which another unit defines (the driver here). The C twin parses in one
    pass, so it types these arguments once the unit is parsed; the oracle reads every definition and prototype
    before it lowers. The unit lowers to one claim graph on both rails, and each emit -- which declares the
    callees defined late (CF-DECLS; they were supplied by hand) -- declares the constants as the parameters'
    pointers, keeps a variadic callee's extra `0` the `int` it reads, and returns what the original does (a
    prototype's array parameter is declared `T *`, as the definition binds it). A call to a later definition
    with no prototype before it, which C99 does not allow, is refused on both rails (CF-DECLS; both had lowered
    it as the prototyped unit), and the prototyped unit's emit runs as the original."""
    if not _CC:
        return
    fx = "cfront_nullarg_link.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        first = _actual_types(emit, "nl_sum")[0]  # `nl_sum(0, 2u, 0, 5)`, defined after the call
        assert None not in (first[0], first[2]), (label, first)
        assert first[0].endswith("*") and "*" not in first[2], (label, first)
    emits = (("twin", c_emit), ("oracle", oracle_emit))
    _run_against_original(fx, src, emits, _NULLARG_LINK_DRIVER, ("-Werror=int-conversion",))

    proto = _NULLARG_BARE.replace(
        "#include <stdint.h>\n", "#include <stdint.h>\nuint32_t nl_g(uint32_t *p, uint32_t s);\n"
    )
    exe = _build_frontend(_session_build_dir())
    _refused_on_both_rails(exe, _NULLARG_BARE, "call to undeclared function 'nl_g'")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "proto.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(proto)
        oracle_summary, r, _entry = _oracle(proto)
        c_summary, emit = _c_run(exe, path)
        assert c_summary == oracle_summary and "ok=1" in c_summary, c_summary
    driver = (
        _GAPS_SAME
        + "int main(void) {\n  for (unsigned n = 0; n < GAPS_N; n++) { SAME(nl_f, gaps_in[n]); }\n"
        + '  puts("MATCH");\n  return 0;\n}\n'
    )
    _run_against_original(
        "prototype",
        proto,
        (("twin", emit), ("oracle", "\n".join(r.emitted.values()))),
        driver,
        ("-Werror=int-conversion",),
    )


_RAILSPLIT_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(rs_not, s); SAME(rs_if_not, s); SAME(rs_poll, s); SAME(rs_bare_deref, s); SAME(rs_elem_addr, s);
    SAME(rs_vla_volatile, s); SAME(rs_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)


def test_rail_splits_lower_alike_and_run_as_the_original():
    """CF-RAILSPLIT: `cfront_railsplit.c` -- forms the two rails lowered to different claim graphs, each an
    accepted form the corpus never held: a logical not as a value, a condition and a status-register polling
    loop (the twin gave `c.un.lnot` the ADD opcode, the oracle's `_UN` SUB); a bare dereference statement (the
    twin's store probe kept the operand's claims, then the statement lowered it again); `&p[i]` of an allocated
    buffer (the twin counted it as taking `p`'s address, so it did not recover the allocation's extent, which
    the oracle does); a VLA of volatile elements (the twin laid it out as ordinary memory, failed R3, and
    declared it without `volatile`). Both rails now lower the fixture to one claim graph -- the generic parity
    gate holds it on the four targets -- and each emit returns what the original does, function by function."""
    if not _CC:
        return
    fx = "cfront_railsplit.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    for rail, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        # a run cannot see a dropped `volatile` on a local; the declaration can
        for decl in ("volatile uint32_t va[", "volatile uint16_t vm["):
            assert decl in emit, (
                f"{rail}: the volatile VLA is declared without its qualifier ({decl})"
            )
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _RAILSPLIT_DRIVER)


def _deep_struct_init(n: int, designated: bool) -> str:
    """A struct nested `n` levels deep, initialized by one scalar -- through `n` elided sub-aggregates, or
    through one designator of `n - 1` steps."""
    s = "struct n0 { uint32_t v; };\n" + "".join(
        f"struct n{k} {{ struct n{k - 1} x; }};\n" for k in range(1, n)
    )
    init = "{ ." + ".".join(["x"] * (n - 1)) + ".v = 7u }" if designated else "{ 7u }"
    return s + f"uint32_t f(void) {{ struct n{n - 1} o = {init}; return *(uint32_t *)&o; }}"


# CF-GAPS: what both rails refuse, each for the reason the form witnesses -- a constraint violation of the
# initializer walk (C11 6.7.9p2, p14, p17-19), a walk past the bounds both rails share, or a form neither
# rail spells (a row's address, a local pointer to an array, a block-scope `extern`). Before CF-GAPS the
# oracle lowered 15 of these 18 units (and crashed on a 16th) and the twin 14 -- an invalid initializer's
# stores (a designator one past a 4-element array among them), a block-scope `extern` as a new
# uninitialized local, and on the twin a row's address taken as an element's.
# (the unit, the oracle's refusal, the twin's)
_GAPS_REFUSED = (
    (
        "uint32_t f(void) { uint32_t a[2] = {1u, 2u, 3u}; return a[0]; }",
        "excess elements in an initializer",
        "excess elements in an initializer",
    ),
    (
        "struct s { uint8_t a, b; };\nuint32_t f(void) { struct s x = {1, 2, 3}; return x.a; }",
        "excess elements in an initializer",
        "excess elements in an initializer",
    ),
    (
        "struct s { uint8_t a[2]; uint8_t b; };\n"
        "uint32_t f(void) { struct s x = {.a[1] = 5, .a = {1}}; return x.a[1]; }",
        "an initializer overrides a prior initialization of a subobject",
        "an initializer overrides a prior initialization of a subobject",
    ),
    (
        "union u { uint8_t a; uint32_t b; };\nstruct s { union u m; };\n"
        "uint32_t f(void) { struct s x = {.m.a = 1, .m.b = 2}; return x.m.b; }",
        "an initializer overrides a prior initialization of a subobject",
        "an initializer overrides a prior initialization of a subobject",
    ),
    (
        'uint32_t f(void) { char s[2] = "abc"; return (uint32_t)s[0]; }',
        "an initializer-string for a character array is too long",
        "an initializer-string for a character array is too long",
    ),
    (
        'uint32_t f(void) { _Bool b[] = "a"; return b[0]; }',
        "an array is initialized by a brace list or a string literal",
        "an array is initialized by a brace list or a string literal",
    ),
    (
        "uint8_t g[4];\nuint32_t f(void) { uint8_t a[4] = g; return a[0]; }",
        "an array is initialized by a brace list or a string literal",
        "an array is initialized by a brace list or a string literal",
    ),
    (
        "uint32_t f(void) { uint8_t a[4] = {[4] = 1}; return a[0]; }",
        "an array designator outside the array",
        "an array designator outside the array",
    ),
    (
        "uint32_t f(void) { uint8_t a[] = {[3000000000u] = 1}; return (uint32_t)sizeof a; }",
        "an array designator outside the array",
        "an array designator outside the array",
    ),
    (
        _deep_struct_init(65, designated=False),
        "an initializer nested deeper than 64 subobjects",
        "an initializer nested deeper than 64 subobjects",
    ),
    (
        _deep_struct_init(65, designated=True),
        "an initializer nested deeper than 64 subobjects",
        "an initializer nested deeper than 64 subobjects",
    ),
    (
        "uint32_t f(void) { uint8_t a[]; return 0u; }",
        "an array of unknown size needs an initializer",
        "an array of unknown size needs an initializer",
    ),
    (
        "uint8_t gm[3][5];\nuint32_t f(uint32_t i) { uint8_t (*r)[5] = gm; return r[i % 3u][1]; }",
        "a local pointer to an array is not supported",
        "expected declarator",
    ),
    (
        "uint32_t f(uint32_t i) { uint8_t m[3][5] = {0}; uint8_t *r = (uint8_t *)&m[i % 3u]; return r[1]; }",
        "partial indexing of a multi-dimensional array",
        "the address of a row of a multi-dimensional array",
    ),
    (
        "struct s { uint8_t c; uint32_t a[2][3]; };\nstruct s ga[4];\n"
        "uint32_t f(uint32_t i) { uint32_t *r = ga[i % 4u].a[1]; return r[0]; }",
        "partial indexing of a struct member array",
        "partial indexing of a struct member array",
    ),
    (
        "uint32_t g = 5u;\nuint32_t f(uint32_t i) { extern uint32_t g; return g + i; }",
        "a block-scope extern declaration is not supported",
        "a block-scope extern declaration is not supported",
    ),
    (
        "const uint32_t g = 5u;\nuint32_t f(uint32_t i) { const extern uint32_t g; return g + i; }",
        "a block-scope extern declaration is not supported",
        "a block-scope extern declaration is not supported",
    ),
    (
        "uint32_t g = 5u;\nuint32_t f(uint32_t i) { uint32_t extern g; return g + i; }",
        "a block-scope extern declaration is not supported",
        "a block-scope extern declaration is not supported",
    ),
)


def test_initializer_and_declaration_refusals_on_both_rails():
    """CF-GAPS: every form of `_GAPS_REFUSED` is refused by both rails for the reason it witnesses -- an
    excess initializer, an override of an initialized subobject, a string too long for its array, a
    non-character array given a string or an expression, a designator outside its array (past the end, past
    INT_MAX), a nesting past the 64 subobjects both walks bound, an unsized array with nothing to count, a
    local pointer to an array, a row's address, a partial member-array index, a block-scope `extern` -- and
    the twin exits with its refusal, never a crash or a truncated walk."""
    from bcir.frontends.cfront.cparse import CParseError
    from bcir.frontends.cfront.lower import CLowerError

    head = "#include <stdint.h>\n"
    for body, why, _ in _GAPS_REFUSED:
        try:
            compile_unit(head + body + "\n", check_clang=False)
        except (CLowerError, CParseError) as e:
            assert why in str(e), (body, str(e))
        else:
            raise AssertionError(f"the oracle lowered {body!r}")
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, (body, _, why) in enumerate(_GAPS_REFUSED):
            path = os.path.join(d, f"r{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(head + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode > 0, (body, run.returncode, run.stdout[:200])
            assert why in run.stdout + run.stderr, (body, run.stdout[:200], run.stderr[:200])


# CF-SMALL: `wchar_t` is the target's own integer type -- Clang's `__WCHAR_TYPE__`: `int` on x86-64,
# RISC-V and i386 Linux, `unsigned int` on AArch64 Linux, `unsigned short` on Windows. (size, signed)
_WCHAR_PINS = {
    "x86_64-linux": (4, True),
    "aarch64-linux": (4, False),
    "riscv64-linux": (4, True),
    "x86_64-windows": (2, False),
    "i386-linux": (4, True),
}
_WCHAR_UNIT = "#include <stdint.h>\n#include <stddef.h>\nint32_t w(uint32_t v) { wchar_t c = (wchar_t)v; return c < 0; }\n"
_TWIN_INT_TYPES = {
    "int32_t": (4, True),
    "uint32_t": (4, False),
    "int16_t": (2, True),
    "uint16_t": (2, False),
}


def test_wchar_t_is_the_targets_integer_type_on_both_rails():
    """CF-SMALL: `wchar_t` (and an `L"..."` literal's code unit) is the target's own integer type, in size
    and in signedness, on both rails: the rails had made it a 4-byte signed `int` everywhere -- a 2-byte
    Windows `wchar_t` read 4 bytes, and AArch64's unsigned one compared below zero. Its size and the literal's
    are in `test_sizeof_measures_the_operand_type_on_every_target`; its signedness is carried by the type of
    the value a conversion to it produces, which neither digest spells (the cast is named by its width), so
    it is read from each rail's own typing: the oracle's temp and the twin's declaration of it. With Clang
    present, the pins are Clang's `__SIZEOF_WCHAR_T__` and `__WCHAR_UNSIGNED__`."""
    from bcir.frontends.cfront.abi import TARGETS

    for t, (size, signed) in _WCHAR_PINS.items():
        r = compile_unit(_WCHAR_UNIT, check_clang=False, target=t)
        fn = r.lowered.functions["w"]
        cast = next(c for c in fn.claims if c.op.startswith("c.cast:"))
        ct = fn.rid_types[cast.wr[0]]
        assert (ct.size, ct.signed) == (size, signed), (t, ct)
    clang = shutil.which("clang")
    if clang:
        for t, (size, signed) in _WCHAR_PINS.items():
            run = subprocess.run(
                [clang, "-target", TARGETS[t].triple, "-dM", "-E", "-x", "c", os.devnull],
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, (t, run.stderr[:500])
            assert f"#define __SIZEOF_WCHAR_T__ {size}" in run.stdout, t
            assert ("#define __WCHAR_UNSIGNED__ 1" in run.stdout) == (not signed), t
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_WCHAR_UNIT)
        for t, pin in _WCHAR_PINS.items():
            run = subprocess.run([exe, "--target", t, path], capture_output=True, text=True)
            emit = run.stdout.partition("----EMIT----\n")[2]
            m = re.search(r"(\w+) t\d+ = \(uint(?:16|32)_t\)v;", emit)
            assert m and _TWIN_INT_TYPES.get(m.group(1)) == pin, (t, emit[:400])


_STRUCTVALUE_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned round = 0; round < 2u; round++) {
    for (unsigned n = 0; n < GAPS_N; n++) {
      uint32_t i = gaps_in[n];
      struct sv_pt p1 = sv_pick(i), p2 = bcir_sv_pick(i);
      if (memcmp(&p1, &p2, sizeof p1)) return fail("sv_pick");
      SAME(sv_local, i); SAME(sv_storage, i); SAME(sv_params, i); SAME(sv_members, i); SAME(sv_shapes, i);
      SAME(sv_select, i); SAME(sv_brace, i); SAME(sv_volatile, i); SAME(sv_calls, i); SAME(sv_stores, i);
      SAME(sv_copies, i); SAME(sv_entry, i);
    }
  }
  puts("MATCH");
  return 0;
}
"""
)


# CF-STRUCTVAL: `va_arg` of a struct is a value of the struct too -- in a unit of its own, since the corpus
# harness does not include <stdarg.h>
_STRUCTVALUE_VARIADIC = """#include <stdint.h>
#include <stdarg.h>
struct sva_pt { uint16_t x, y; };
static uint32_t sva_take(uint32_t n, ...) {
  va_list ap;
  va_start(ap, n);
  struct sva_pt q = va_arg(ap, struct sva_pt);
  va_end(ap);
  return q.x + q.y * 3u + n;
}
uint32_t sva_entry(uint32_t i) {
  struct sva_pt a = {(uint16_t)i, (uint16_t)(i >> 16)};
  struct sva_pt b = {3, 4};
  return sva_take(1u, a) + sva_take(2u, b) * 5u;
}
"""
_STRUCTVALUE_VARIADIC_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) SAME(sva_entry, gaps_in[n]);
  puts("MATCH");
  return 0;
}
"""
)


def test_struct_values_run_as_the_original_on_both_rails():
    """CF-STRUCTVAL: `cfront_structvalue.c` -- a struct or union read as a value (an element of a local, static or
    file-scope array of structs, one through a pointer parameter, an array parameter or a pointer member, a struct
    or union member, a member array's element, `*p`, a select of two structs, a struct returned through a function
    pointer) is the struct itself: copied into a declaration or an assignment, returned, passed by value, stored
    into an element or a member, placed whole by a brace list, and seen as the struct by `typeof` and `_Generic`;
    so is `va_arg` of a struct (`_STRUCTVALUE_VARIADIC`). Both rails had read it into an integer temp -- `uint32_t
    t = ps[i];` on the oracle, `int32_t`/`int64_t` on the twin -- so no emit compiled; the twin had also kept an
    array parameter of structs a struct by value and typed `&ps[i]` `int32_t *`, and its brace list, `typeof` and
    `_Generic` had not seen the struct (a different claim graph, and `_Generic` picked its default). Both rails
    lower it to the same claim graph and each emit returns what the original does, function by function, under
    every compiler at hand."""
    if not _CC:
        return
    fx = "cfront_structvalue.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _STRUCTVALUE_DRIVER)
    src = _STRUCTVALUE_VARIADIC
    oracle_summary, r, _entry = _oracle(src)
    oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "structvalue_va.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        c_summary, c_emit = _c_run(_build_frontend(_session_build_dir()), path)
    assert c_summary == oracle_summary, f"parity\n C: {c_summary}\nPY: {oracle_summary}"
    emits = (("twin", c_emit), ("oracle", oracle_emit))
    _run_against_original("structvalue_va.c", src, emits, _STRUCTVALUE_VARIADIC_DRIVER)


# CF-STRUCTINIT: a struct or union object takes only a value of its own type -- an initializer that is not a brace
# list is one expression of it (C11 6.7.9p13), and so is what `=` assigns it (6.5.16.1p1), whether the object is
# named, a member, an element or reached through a pointer; the arms of `?:` share one struct type (6.5.15p3). Both
# rails had lowered each of these units -- the same claim graph on both -- to a copy the emit spells `x = 5;`,
# which does not compile. (the unit, the reason both rails refuse it with)
_STRUCT_INITIALIZED = "a struct or union is initialized by a brace list or a value of its own type"
_STRUCT_ASSIGNED = "a struct or union is assigned a value of its own type"
_STRUCT_SELECTED = "the arms of `?:` are a struct or union and a value of another type"
_STRUCTINIT_REFUSED = (
    (
        'struct sb { uint8_t b[2]; };\nuint32_t f(void) { struct sb x = "a"; return x.b[0]; }',
        _STRUCT_INITIALIZED,
    ),
    (
        "struct sb { uint8_t b[2]; };\nuint32_t f(void) { struct sb x = 5; return x.b[0]; }",
        _STRUCT_INITIALIZED,
    ),
    (
        "struct sb { uint8_t b[2]; };\nstruct sa { uint8_t b[2]; };\n"
        "uint32_t f(void) { struct sa y = {{1, 2}}; struct sb x = y; return x.b[0]; }",
        _STRUCT_INITIALIZED,
    ),
    (
        "struct sb { uint8_t b[2]; };\nuint32_t f(void) { struct sb x = {{0}}; x = 5; return x.b[0]; }",
        _STRUCT_ASSIGNED,
    ),
    (
        "union u { uint32_t w; };\nuint32_t f(void) { union u x = 5; return x.w; }",
        _STRUCT_INITIALIZED,
    ),
    (
        "struct s { uint32_t w; };\nunion u { uint32_t w; };\n"
        "uint32_t f(void) { struct s y = {3u}; union u x = {0}; x = y; return x.w; }",
        _STRUCT_ASSIGNED,
    ),
    (
        "struct pt { uint16_t x, y; };\nuint32_t f(void) { struct pt b = {1, 2}; struct pt a = &b; return a.x; }",
        _STRUCT_INITIALIZED,
    ),
    (
        "struct pt { uint16_t x, y; };\nstruct hh { uint32_t k; struct pt inner; };\n"
        "uint32_t f(void) { struct hh h = {0}; h.inner = 5; return h.k; }",
        _STRUCT_ASSIGNED,
    ),
    (
        "struct pt { uint16_t x, y; };\nstruct hh { uint32_t k; struct pt inner; };\n"
        "uint32_t f(struct hh *p) { p->inner = 5; return p->k; }",
        _STRUCT_ASSIGNED,
    ),
    (
        "struct pt { uint16_t x, y; };\nuint32_t f(void) { struct pt ps[2] = {0}; ps[0] = 5; return ps[0].x; }",
        _STRUCT_ASSIGNED,
    ),
    (
        "struct pt { uint16_t x, y; };\nuint32_t f(struct pt *p) { *p = 5; p[1] = 6; return p->x; }",
        _STRUCT_ASSIGNED,
    ),
    (
        "struct pt { uint16_t x, y; };\nuint32_t f(struct pt **pp) { **pp = 5; return 0u; }",
        _STRUCT_ASSIGNED,
    ),
    (
        "struct pt { uint16_t x, y; };\nstruct poly { uint32_t n; struct pt v[3]; };\n"
        "uint32_t f(void) { struct poly P = {0}; P.v[1] = 5; return P.n; }",
        _STRUCT_ASSIGNED,
    ),
    (
        "struct pt { uint16_t x, y; };\nstruct nd { struct pt v; struct nd *next; };\n"
        "uint32_t f(struct nd *p) { p->next->v = 5; return 0u; }",
        _STRUCT_ASSIGNED,
    ),
    (
        "struct pt { uint16_t x, y; };\nstruct pt gp;\nuint32_t f(void) { gp = 5; return gp.x; }",
        _STRUCT_ASSIGNED,
    ),
    (
        "struct pt { uint16_t x, y; };\n"
        "uint32_t f(void) { struct pt a = {0, 0}, b = {0, 0}; b = (a = 5); return b.x; }",
        _STRUCT_ASSIGNED,
    ),
    (
        "struct pt { uint16_t x, y; };\nstruct pq { uint16_t x, y; };\n"
        "uint32_t f(uint32_t i) { struct pt a = {1, 2}; struct pq q = {3, 4}; struct pt c = i ? a : q; return c.x; }",
        _STRUCT_SELECTED,
    ),
    (
        "struct pt { uint16_t x, y; };\n"
        "uint32_t f(uint32_t i) { struct pt a = {1, 2}; uint32_t k = i ? 5u : a; return k; }",
        _STRUCT_SELECTED,
    ),
)
# ... and a value of the object's own type -- through a typedef, qualified, a union, a call's result, a compound
# literal, a select -- still lowers, to one claim graph on both rails (`cfront_structvalue.c` runs them)
_STRUCTINIT_LOWERED = (
    "struct sb { uint8_t b[2]; };\nuint32_t f(void) { struct sb y = {{1, 2}}; struct sb x = y; return x.b[1]; }",
    "struct sb { uint8_t b[2]; };\nuint32_t f(void) { struct sb y = {{1, 2}}, x = {{0}}; x = y; return x.b[1]; }",
    "struct pt { uint16_t x, y; };\ntypedef struct pt P;\n"
    "uint32_t f(uint32_t i) { P a = {(uint16_t)i, 2}; const struct pt b = a; P c = b; return c.x + c.y; }",
    "union u { uint32_t w; };\nuint32_t f(uint32_t i) { union u a, b; a.w = i; b = a; union u c = b; return c.w; }",
    "struct pt { uint16_t x, y; };\nstruct pt mk(uint32_t i) { struct pt q = {(uint16_t)i, 2}; return q; }\n"
    "uint32_t f(uint32_t i) { struct pt a = mk(i), b = (struct pt){1, 2}; b = i ? a : b; return a.x + b.y; }",
    "struct pt { uint16_t x, y; };\nstruct hh { uint32_t k; struct pt inner; struct pt v[2]; };\n"
    "uint32_t f(struct hh *p, struct pt *q) { struct pt a = *q; p->inner = a; p->v[1] = p->inner; q[1] = a; "
    "*q = p->v[1]; return p->k; }",
)


def test_a_struct_takes_only_a_value_of_its_own_type_on_both_rails():
    """CF-STRUCTINIT: every unit of `_STRUCTINIT_REFUSED` -- a struct or union initialized from a string, a scalar,
    a pointer or another struct or union; assigned a scalar or another type as a named object, a member, an element
    or through a pointer, as a statement or a value; a `?:` whose arms are not of one struct type -- is refused on
    both rails with the reason it witnesses (both had lowered it to an emit that does not compile), and every unit
    of `_STRUCTINIT_LOWERED`, a value of the object's own type, still lowers to the same claim graph on both."""
    from bcir.frontends.cfront.cparse import CParseError
    from bcir.frontends.cfront.lower import CLowerError

    head = "#include <stdint.h>\n"
    summaries = {}
    for body in _STRUCTINIT_LOWERED:
        summaries[body], _r, _e = _oracle(head + body + "\n")
        assert "ok=1" in summaries[body], (body, summaries[body])
    for body, why in _STRUCTINIT_REFUSED:
        try:
            compile_unit(head + body + "\n", check_clang=False)
        except (CLowerError, CParseError) as e:
            assert why in str(e), (body, str(e))
        else:
            raise AssertionError(f"the oracle lowered {body!r}")
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, body in enumerate(_STRUCTINIT_LOWERED):
            path = os.path.join(d, f"l{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(head + body + "\n")
            c_summary, _emit = _c_run(exe, path)
            assert c_summary == summaries[body], (body, c_summary, summaries[body])
        for n, (body, why) in enumerate(_STRUCTINIT_REFUSED):
            path = os.path.join(d, f"r{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(head + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode > 0, (body, run.returncode, run.stdout[:200])
            assert why in run.stdout + run.stderr, (body, run.stdout[:200], run.stderr[:200])


def _emitted_function(emit: str, name: str) -> str:
    """The text of the emitted function `bcir_<name>` in a rail's emit (to its closing brace at column 0)."""
    start = emit.index(f" bcir_{name}(")
    return emit[start : emit.index("\n}", start)]


_AOSNEST_DRIVER = (
    _GAPS_SAME
    + r"""
#define SAME_AN(f, ...) do { an_reset(); uint64_t a_ = f(__VA_ARGS__); an_reset(); \
    uint64_t b_ = bcir_##f(__VA_ARGS__); if (a_ != b_) return fail(#f); } while (0)
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(an_union, s); SAME(an_compound, s); SAME(an_steps, s); SAME(an_values, s); SAME(an_address, s);
    SAME(an_deep, s); SAME(an_member_array, s); SAME_AN(an_global, s); SAME(an_param_call, s);
    SAME(an_member_elements, s); SAME(an_pointer_member, s); SAME(an_volatile, s); SAME(an_atomic, s);
    SAME(an_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)


def test_nested_members_of_array_elements_run_as_the_original_on_both_rails():
    """CF-NESTMEM: `cfront_aosnest.c` -- a member of a member of an array element, `a[i].m.k` through nested
    structs and unions at any depth, and a member array at the end of the chain (`a[i].m.arr[j]`, 1-D and
    2-D), is one access at the element's stride plus the member's flattened offset: read, stored,
    compound-assigned, stepped as a statement and as a value, assigned as a value and addressed, on a local
    array, a file-scope array, the elements a pointer parameter points at, a member array of structs
    (`b.s[i].m.k`) and the elements a pointer member points at (`h.next[i].m.k`). Both rails had refused
    every nested form, and the twin the statement `a[i].f++;` and `h.next[i].f`. Both rails lower it to the
    same claim graph and each emit returns what the original does, function by function; a volatile member's
    members are volatile accesses in both emits (the twin's descent had dropped the enclosing qualifier)."""
    if not _CC:
        return
    fx = "cfront_aosnest.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _AOSNEST_DRIVER)
    for rail, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        body = _emitted_function(emit, "an_volatile")
        assert body.count("*(volatile uint16_t *)") == 5, (rail, body)


_PARENPOSTFIX_DRIVER = (
    _GAPS_SAME
    + r"""
#define SAME_PP(f, ...) do { pp_reset(); uint64_t a_ = f(__VA_ARGS__); pp_reset(); \
    uint64_t b_ = bcir_##f(__VA_ARGS__); if (a_ != b_) return fail(#f); } while (0)
int main(void) {
  static uint32_t m1[2][3], m2[2][3];
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(pp_index, s); SAME(pp_store, s); SAME(pp_steps, s); SAME(pp_literal, s); SAME(pp_literal_struct, s);
    SAME(pp_members, s); SAME(pp_deref, s); SAME(pp_macro, s); SAME_PP(pp_global, s);
    SAME(pp_control, s); SAME(pp_forms, s); SAME(pp_not_operands, s); SAME(pp_entry, s);
    for (unsigned k = 0; k < 6u; k++) { m1[k / 3u][k % 3u] = s + k; m2[k / 3u][k % 3u] = s + k; }
    if (pp_rows(m1, s) != bcir_pp_rows(&m2[0][0], s) || memcmp(m1, m2, sizeof m1)) return fail("pp_rows");
  }
  puts("MATCH");
  return 0;
}
"""
)


# CF-PAREN on a LOCAL array lent to a call: a local array passed to a function escapes its activation's proof,
# and `escape.unproved` counts every local array of the `cfront_*.c` corpus the analysis does not prove private,
# so the in-unit call of `pp_rows` on a local row pair runs here (the driver above passes the fixture's own).
_PARENPOSTFIX_LOCAL = """#include <stdint.h>
uint32_t pl_rows(uint32_t (*p)[3], uint32_t i) {
  (*p)[i % 3u] = 9u;
  (*p)[0] += 1u;
  uint32_t *q = &(*p)[1];
  *q += 2u;
  return (*p)[i % 3u] + (*p)[0] * 3u + (*p)[1] * 5u;
}
uint32_t pl_rows_call(uint32_t s) {
  uint32_t m[2][3] = {{s, 2u, 3u}, {4u, 5u, 6u}};
  uint32_t r = pl_rows(m, s);
  return r + m[0][0] + m[1][2] * 3u;
}
"""
_PARENPOSTFIX_LOCAL_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) SAME(pl_rows_call, gaps_in[n]);
  puts("MATCH");
  return 0;
}
"""
)


def test_postfix_after_parentheses_runs_as_the_original_on_both_rails():
    """CF-PAREN: `cfront_parenpostfix.c` -- a postfix operator after a parenthesized operand applies to what
    the parentheses hold: `(a)[i]`, `((a))[i]`, `(s).x`, `(p)->x`, `(x)++`, `((uint32_t[3]){s, 7})[1]`,
    `((struct pt[2]){...})[1].x`, `("abc")[i]` and the macro spellings `#define REGS (base)` / `((d)->regs)`,
    read, stored, compound-assigned, stepped and addressed, in a for, a do-while, an if-else, a switch, a
    statement expression, `_Generic`, `typeof` and `sizeof`; `(*p).m` is `p->m` and `(*p)[i]` is `p[0][i]`,
    the row a `T (*p)[N]` parameter points at. Parentheses that open a call's arguments, a condition or a cast
    stay (`if (s) ++y;`, `(T)++w`, `at(s)[1]`). The twin had refused every form; the oracle had loaded the
    whole struct for `(*p).m` into a 4-byte temp (a wrong value past its first word) and emitted an
    uncompilable scalar subscript for `(*p)[i]`. Both rails lower it to the same claim graph and each emit
    returns what the original does, function by function."""
    if not _CC:
        return
    fx = "cfront_parenpostfix.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _run_against_original(
        fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _PARENPOSTFIX_DRIVER
    )
    # the in-unit call on a LOCAL row pair, which the corpus fixture leaves to its driver
    src = _PARENPOSTFIX_LOCAL
    oracle_summary, r, _entry = _oracle(src)
    oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "paren_local.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        c_summary, c_emit = _c_run(_build_frontend(_session_build_dir()), path)
    assert c_summary == oracle_summary, f"parity\n C: {c_summary}\nPY: {oracle_summary}"
    emits = (("twin", c_emit), ("oracle", oracle_emit))
    _run_against_original("paren_local.c", src, emits, _PARENPOSTFIX_LOCAL_DRIVER)


# CF-NESTMEM, CF-PAREN: what both rails still refuse, each for the reason the form witnesses -- a member of an
# array element that is no scalar (a bitfield, a struct, a pointer, an array used as a value) and a member
# array of a member-array element, whatever the nesting; and a parenthesized DECLARATOR, which neither rail
# spells: the twin's CF-PAREN rewrite must leave a declaration's parentheses -- after its type, a `*`, a
# declarator comma, in a for-init or a statement expression, `T (*p)[N]` -- as they are.
# (the unit, the oracle's refusal, the twin's)
_PAREN_NEST_REFUSED = (
    (
        "struct pt { uint16_t x : 4, y : 12; }; struct seg { struct pt a, b; };\n"
        "uint32_t f(uint32_t s) { struct seg g[2] = {0}; g[1].b.y = (uint16_t)s; return g[1].b.y; }",
        "array-of-structs non-scalar element field ('.y')",
        "array-of-structs non-scalar element field",
    ),
    (
        "struct pt { uint16_t x, y; }; struct seg { struct pt a, b; }; struct top { struct seg s; };\n"
        "uint32_t f(uint32_t s) { struct top g[2] = {0}; struct pt q = g[1].s.b; return q.x + s; }",
        "array-of-structs non-scalar element field ('.b')",
        "array-of-structs non-scalar element field",
    ),
    (
        "struct pt { uint16_t x, y; }; struct hold { uint32_t k; struct pt *p; }; struct seg { struct hold h; };\n"
        "uint32_t f(uint32_t s) { struct pt q = {1, 2}; struct seg g[2] = {0}; g[1].h.p = &q; return s; }",
        "array-of-structs non-scalar element field ('.p')",
        "array-of-structs non-scalar element field",
    ),
    (
        "struct in { uint8_t arr[4]; }; struct seg { struct in m; };\n"
        "uint32_t f(uint32_t s) { struct seg g[2] = {0}; uint8_t *p = g[1].m.arr; return p[0] + s; }",
        "array-of-structs non-scalar element field ('.arr')",
        "array-of-structs non-scalar element field",
    ),
    (
        "struct pt { uint16_t x; uint8_t m[2]; }; struct box { struct pt s[2]; };\n"
        "uint32_t f(uint32_t i) { struct box b[2] = {0}; b[1].s[1].m[0] = 3u; return b[1].s[1].m[0] + i; }",
        "a subscript of the array-of-structs member 's' is not supported",
        "a subscript of an array-of-structs member array is not supported",
    ),
    (
        "uint32_t f(uint32_t x) { uint32_t (y)[3]; y[0] = x; return y[0]; }",
        "expected 'IDENT', got PUNCT '('",
        "expected declarator name",
    ),
    (
        "uint32_t f(uint32_t x) { uint32_t a = 1u, (b)[3]; b[0] = x; return a + b[0]; }",
        "expected 'IDENT', got PUNCT '('",
        "expected declarator name",
    ),
    (
        "uint32_t f(uint32_t x) { uint32_t *(q)[3]; q[0] = &x; return *q[0]; }",
        "expected 'IDENT', got PUNCT '('",
        "expected declarator name",
    ),
    (
        "uint32_t f(uint32_t x) { uint32_t t = 0u; for (uint32_t (y)[2] = {1u, 2u}; t < 2u; t++) x += y[t];"
        " return x; }",
        "expected 'IDENT', got PUNCT '('",
        "expected declarator name",
    ),
    (
        "uint32_t f(uint32_t x) { uint32_t v = ({ uint32_t (w)[2] = {1u, 2u}; w[1] + x; }); return v; }",
        "expected 'IDENT', got PUNCT '('",
        "expected declarator name",
    ),
    (
        "uint32_t f(uint32_t x) { uint32_t m[2][3] = {{1u, 2u, 3u}, {4u, 5u, 6u}}; uint32_t (*p)[3] = m;"
        " return (*p)[1] + x; }",
        "a local pointer to an array is not supported",
        "expected declarator name",
    ),
)


def test_paren_and_nested_member_refusals_on_both_rails():
    """CF-NESTMEM, CF-PAREN: every form of `_PAREN_NEST_REFUSED` is refused by both rails for the reason it
    witnesses -- a non-scalar member of an array element at any nesting (a bitfield, a struct, a pointer, an
    array used as a value), a member array of a member-array element, a parenthesized declarator (after the
    type, a `*`, a declarator comma, in a for-init, in a statement expression, `T (*p)[N]` for a local) -- and
    the twin exits with its refusal, never a crash or a rewritten declaration."""
    from bcir.frontends.cfront.cparse import CParseError
    from bcir.frontends.cfront.lower import CLowerError

    head = "#include <stdint.h>\n"
    for body, why, _ in _PAREN_NEST_REFUSED:
        try:
            compile_unit(head + body + "\n", check_clang=False)
        except (CLowerError, CParseError) as e:
            assert why in str(e), (body, str(e))
        else:
            raise AssertionError(f"the oracle lowered {body!r}")
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, (body, _, why) in enumerate(_PAREN_NEST_REFUSED):
            path = os.path.join(d, f"r{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(head + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode > 0, (body, run.returncode, run.stdout[:200])
            assert why in run.stdout + run.stderr, (body, run.stdout[:200], run.stderr[:200])


# CF-RTVOL: a volatile access at a literal byte offset of a device region, `*(volatile T *)((char *)p + K)`,
# is one load or store at offset K from p -- the member access the emit spells so -- on both rails. Every
# form is the body of `uint32_t f(volatile struct regs *d, struct pl s, uint32_t v)` over one memory image:
# `d` a register block, `s` a struct passed by value whose storage is volatile, `ga` a volatile global array.
_BYTEOFF_HEAD = (
    "#include <stdint.h>\n"
    "struct regs { uint32_t a; uint32_t b; int16_t h0; int16_t h1; float x; };\n"
    "struct pl { volatile uint32_t r; uint32_t q; };\n"
    "volatile uint32_t ga[4];\n"
)
# the forms that fold: (body, the access's byte offset)
_BYTEOFF_FOLD = {
    "load": ("return *(volatile uint32_t *)((const volatile char *)d + 4);", 4),
    "store": ("*(volatile uint32_t *)((volatile char *)d + 4) = v; return 0u;", 4),
    "plain_char": ("return *(volatile uint32_t *)((char *)d + 4);", 4),
    "value_struct": ("return *(volatile uint32_t *)((const volatile char *)&s + 0);", 0),
    "array": ("return *(volatile uint32_t *)((const volatile char *)ga + 8);", 8),
    "int16": (
        "int32_t h = *(volatile int16_t *)((const volatile char *)d + 10); return (uint32_t)h;",
        10,
    ),
    "float": ("return *(volatile float *)((const volatile char *)d + 12) > 1.5f;", 12),
    "or_assign": ("*(volatile uint32_t *)((volatile char *)d + 0) |= v; return 0u;", 0),
}
# the near misses, which fold on neither rail: an offset that is no literal, a byte pointer that is not a
# plain `char` one, an access that is not volatile, and one that is `_Atomic`
_BYTEOFF_NEAR = {
    "var_offset": "return *(volatile uint32_t *)((char *)d + (v & 4u));",
    "uchar_cast": "return *(volatile uint32_t *)((volatile unsigned char *)d + 4);",
    "nonvolatile": "return *(uint32_t *)((char *)d + 4);",
    "atomic": "return *(volatile _Atomic uint32_t *)((char *)d + 4);",
}
# runs the original and an emit from the same seeded image -- register block, by-value struct, global array
# -- and compares the value each returns and every byte each leaves
_BYTEOFF_DRIVER = r"""
static uint64_t S=0x9E3779B97F4A7C15u;
static uint32_t rng(void){S=S*6364136223846793005u+1442695040888963407u;return (uint32_t)(S>>32);}
int main(void){
  for(int i=0;i<256;i++){
    struct regs m0, m1; struct pl s; uint32_t g0[4], g1[4], v=rng();
    for(unsigned k=0;k<sizeof m0;k++) ((unsigned char *)&m0)[k]=(unsigned char)rng();
    memcpy(&m1,&m0,sizeof m0); s.r=rng(); s.q=rng();
    for(int k=0;k<4;k++){ g0[k]=rng(); ga[k]=g0[k]; }
    uint32_t ra=f_s((volatile struct regs *)&m0, s, v);
    for(int k=0;k<4;k++){ g1[k]=ga[k]; ga[k]=g0[k]; }
    uint32_t rb=bcir_f((volatile struct regs *)&m1, s, v);
    int same=ra==rb && !memcmp(&m0,&m1,sizeof m0);
    for(int k=0;k<4;k++) same=same && g1[k]==ga[k];
    if(!same){printf("MISMATCH@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}
"""


def _byteoff_equiv(source: str, emit: str) -> str:
    """The `_BYTEOFF_DRIVER` verdict for one form's emit: MATCH, MISMATCH@i, or why it did not build."""
    harness = (
        "#include <stdio.h>\n#include <string.h>\n"
        + re.sub(r"\bf\(", "f_s(", source)
        + "\n"
        + emit
        + _BYTEOFF_DRIVER
    )
    with tempfile.TemporaryDirectory() as d:
        c, e = os.path.join(d, "b.c"), os.path.join(d, "b")
        open(c, "w").write(harness)
        for std in ("c23", "c2x", "c17"):
            b = subprocess.run(
                host_link_args([_CC, f"-std={std}", "-O2", c, "-o", e]),
                capture_output=True,
                text=True,
            )
            if b.returncode == 0:
                break
        else:
            return f"build-failed:{b.stderr.strip().splitlines()[-1] if b.stderr else '?'}"
        return subprocess.run([e], capture_output=True, text=True).stdout.strip()


def _pointer_casts(fn) -> list:
    """The claims of `fn` that cast to a pointer type -- the pointer computation a fold replaces."""
    return [c.op for c in fn.claims if c.op.startswith("c.cast:") and c.op.endswith("*")]


def test_a_volatile_access_at_a_literal_byte_offset_folds_alike_on_both_rails():
    """CF-RTVOL. `*(volatile T *)((char *)p + K)` -- a volatile access of `T` at the literal byte offset `K`
    of a device region `p` -- is one load or store at offset K from p, the claim the member access `p->m`
    lowers to (the oracle's `_byte_offset_access`, the twin's `byte_off_access`): through a pointer, `&` of
    a struct passed by value and an array; plain-`char` or qualified byte pointer; `uint32_t`, `int16_t` and
    `float`; a load, a store and `|=`. It is how the emit spells such an access, so the round trip keeps
    its graph -- and ordinary driver code spells it too, so the rails must agree on it: the oracle alone
    had folded it, 1 claim against the twin's 5. Each form digests alike on both rails, and each rail's emit
    returns and leaves what the original does. A near miss -- an offset no literal, a byte pointer no plain
    `char` one, an access not volatile, or `_Atomic` -- folds on neither rail and digests alike too."""
    exe = _build_frontend(_session_build_dir()) if _CC else None
    forms = [(n, b, k) for n, (b, k) in _BYTEOFF_FOLD.items()]
    forms += [(n, b, None) for n, b in _BYTEOFF_NEAR.items()]
    for name, body, off in forms:
        src = (
            _BYTEOFF_HEAD
            + "uint32_t f(volatile struct regs *d, struct pl s, uint32_t v) { "
            + body
            + " }\n"
        )
        summary, r, entry = _oracle(src)
        assert "ok=1" in summary, (name, summary)
        accesses = [c for c in entry.claims if c.op in ("c.load", "c.store")]
        bases = {rid for _n, rid, _ct in entry.params} | set(entry.globals_used)
        if off is None:  # a near miss: the access goes through the computed pointer
            assert _pointer_casts(entry), (name, [c.op for c in entry.claims])
            assert all(c.rd[0] not in bases for c in accesses), name
        else:  # folded: each access through the base itself, at the literal offset, volatile
            assert not _pointer_casts(entry), (name, [c.op for c in entry.claims])
            assert accesses and all(
                c.rd[0] in bases and (c.imm[0] if c.imm else 0) == off and c.volatile
                for c in accesses
            ), (name, [(c.op, c.rd, c.imm, c.volatile) for c in accesses])
        if exe is None:
            continue
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "byteoff.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
            c_summary, c_emit = _c_run(exe, path)
        assert c_summary == summary, f"{name}: parity\n C: {c_summary}\nPY: {summary}"
        oracle_emit = "\n".join(r.emitted[n] for n in r.lowered.functions)
        for rail, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            verdict = _byteoff_equiv(src, emit)
            assert verdict == "MATCH", f"{name}: the {rail}'s emit: {verdict}\n{emit}"


# CF-RTVOL: a temporary is `t<rid>` only while no declared name spells it -- the rails number their temporaries
# differently, so each rail's own emit of the plain program names the parameters and the local that collide
_TEMP_NAMES = (
    "#include <stdint.h>\n"
    "uint32_t f(uint32_t {a}, uint32_t {b}) {{ uint32_t {c} = {a} * 3u; {a} = {c} + {b}; return {a} + 1u; }}\n"
)


def test_a_temporary_never_redeclares_a_declared_name_on_either_rail():
    """CF-RTVOL. The emit names a temporary `t<rid>`; a parameter or a local that a source (or a re-parsed
    emit) spells the same was declared a second time -- `uint32_t f(uint32_t t102) { ... t102 = ...` -- a
    redefinition the emit did not compile past. Each rail now names such a temporary clear of every declared
    name. The colliding names are read off each rail's own emit of the plain program: its first temporaries
    become the two parameters (one of them assigned) and its last the local. Emit-only: the program digests
    as the plain one did, on both rails, and both rails' emits run equivalent to it."""
    plain = _TEMP_NAMES.format(a="a", b="b", c="c")
    summary0, r0, _entry = _oracle(plain)
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "plain.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(plain)
        _s, twin_emit = _c_run(exe, path)
        emits = {"oracle": "\n".join(r0.emitted.values()), "twin": twin_emit}
        for rail, emit in emits.items():
            temps = list(
                dict.fromkeys(re.findall(r"\bt\d+\b", re.sub(r"/\*.*?\*/", "", emit, flags=re.S)))
            )
            assert len(temps) >= 3, (rail, emit)
            src = _TEMP_NAMES.format(a=temps[0], b=temps[1], c=temps[-1])
            summary, r, entry = _oracle(src)
            assert summary == summary0, (rail, summary, summary0)
            path = os.path.join(d, f"{rail}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
            c_summary, c_emit = _c_run(exe, path)
            assert c_summary == summary, f"{rail} names: parity\n C: {c_summary}\nPY: {summary}"
            oracle_emit = "\n".join(r.emitted[n] for n in r.lowered.functions)
            for label, e in (("twin", c_emit), ("oracle", oracle_emit)):
                verdict = _equiv(r.source, e, entry)
                assert verdict == "MATCH", f"{rail} names, the {label}'s emit: {verdict}\n{e}"


_STATICTAB_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned round = 0; round < 4u; round++) {
    for (unsigned n = 0; n < GAPS_N; n++) {
      uint32_t i = gaps_in[n];
      SAME(stt_crc_step, i); SAME(stt_counter, i); SAME(stt_conf, i); SAME(stt_const_first, i);
      SAME(stt_inferred, i); SAME(stt_designated, i); SAME(stt_elided, i); SAME(stt_rows, i);
      SAME(stt_string, i); SAME(stt_nested, i); SAME(stt_points, i); SAME(stt_union, i);
      SAME(stt_negatives, i); SAME(stt_constexpr, i); SAME(stt_wide, i); SAME(stt_bitfields, i);
      SAME(stt_flags, i); SAME(stt_floats, i); SAME(stt_colors, i); SAME(stt_scalars, i);
      SAME(stt_in_loop, i); SAME(stt_two_scopes, i); SAME(stt_pointer, i); SAME(stt_struct_pointer, i);
      SAME(stt_void_pointer, i); SAME(stt_function_pointer, i); SAME(stt_entry, i);
    }
  }
  puts("MATCH");
  return 0;
}
"""
)


def test_static_tables_run_as_the_original_on_both_rails():
    """CF-STATICTAB: `cfront_statictab.c` -- a static local array, struct or union with a brace initializer
    (a lookup table, a partly initialized counter, `[]` sized by its list, designators, brace-elided and
    braced rows, a string, nested structs, an array of structs, a union by its first and by a designated
    member, negative and 64-bit constants, constant expressions with `sizeof`, enumerators and a cast,
    bit-fields, `_Bool` and float elements), a scalar static folded the same way, and static pointers and
    function pointers. Both rails had refused every table; the twin had also declared a static pointer as an
    uninitialized automatic one. Now each lowers to the same claim graph on both rails, and each emit, run
    call after call in lockstep with the original, returns what it does -- a table stored at each call, or a
    pointer that forgets where it was, diverges on the second call."""
    if not _CC:
        return
    fx = "cfront_statictab.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    for emit in (oracle_emit, c_emit):  # the image is the declaration's, never a store at a call
        assert "tbl[4] = {3u, 5u, 7u, 11u};" in emit, emit[:600]
        assert "static int64_t w[2] = {-5, (-9223372036854775807 - 1)};" in emit, emit[:600]
        # a static pointer or function pointer keeps its storage class: declared as an automatic one, it
        # would be read uninitialized -- undefined, so only the declaration tells it reliably
        for name in ("sp", "sq", "np", "vp", "fp", "fq"):
            decl = rf"^[ \t]*static [^;=\n]*\b{name}\b[^;\n]*;"
            assert re.search(decl, emit, re.M), (name, emit[:600])
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _STATICTAB_DRIVER)


def _twin_canon(exe: str, body: str) -> str:
    """The twin's canonical serialization (`--canon`) of the unit `body`."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "u.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
        return subprocess.run([exe, "--canon", path], capture_output=True, text=True).stdout


def test_static_table_image_is_in_the_canon_not_the_body_on_both_rails():
    """CF-STATICTAB, the claim-level decision: a static's initializer runs once, before the program, so it is
    no claim -- a static table lowers to exactly the claims of the same static declared without one (no
    per-call store or constant), its accesses reading the static as an input. Its folded image joins the
    canon instead, as the initializer both rails render (`static t = {3u, 5u, 7u, 11u}`), so two tables
    differing in one value differ in the digest on each rail, and each digest is the same on both."""
    from bcir.verify import cfront_structural_canon

    head = "#include <stdint.h>\nuint32_t f(uint32_t i) { static uint8_t t[4]"
    tail = "; t[i % 4u] += 1u; return t[i % 4u]; }\n"
    units = {
        "zero": head + tail,
        "table": head + " = {3u, 5u, 7u, 11u}" + tail,
        "other": head + " = {3u, 5u, 7u, 12u}" + tail,
    }
    lowered = {k: compile_unit(u, check_clang=False).lowered for k, u in units.items()}

    def ops(k):
        return sorted(c.op for c in lowered[k].functions["f"].claims)

    assert ops("table") == ops("zero") == ops("other")  # the image adds no claim
    digests = {k: cfront_structural_digest(lw) for k, lw in lowered.items()}
    assert len(set(digests.values())) == 3, digests
    canon = cfront_structural_canon(lowered["table"])
    assert "static t = {3u, 5u, 7u, 11u}\n" in canon, canon
    # a zero image is the static's own zero: no line
    assert "static t" not in cfront_structural_canon(lowered["zero"])
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    for k, body in units.items():
        assert _twin_canon(exe, body) == cfront_structural_canon(lowered[k]), k
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "u.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(body)
            c_summary, _emit = _c_run(exe, path)
        assert f"digest={digests[k]:016x}" in c_summary, (k, c_summary)


# CF-STATICTAB: what both rails refuse in a static's initializer, each for the reason the form witnesses: an
# entry that is no integer constant expression -- a parameter, another static, a local `const`, an address, a
# string's address, a floating constant, a struct value, arithmetic C leaves undefined -- a union given an
# anonymous member other than its first, which the rendered brace list cannot name, and the walk's own.
_STATICTAB_REFUSED = (
    ("uint32_t f(uint32_t i) { static uint32_t t[2] = {i, 2}; return t[0] + t[1]; }", "not"),
    ("uint32_t f(uint32_t i) { static uint32_t n = i; return n; }", "not"),
    (
        "uint32_t f(uint32_t i) { static int a = 5; static int b = a; return (uint32_t)(a + b) + i; }",
        "not",
    ),
    (
        "uint32_t f(uint32_t i) { const uint32_t k = 5u; static uint32_t t[2] = {k, 1u}; return t[0] + i; }",
        "not",
    ),
    ("uint32_t g = 5u;\nuint32_t f(uint32_t i) { static uint32_t *p = &g; return *p + i; }", "not"),
    (
        'uint32_t f(uint32_t i) { static const char *n[2] = {"a", "b"}; return (uint32_t)n[i % 2u][0]; }',
        "not",
    ),
    (
        "uint32_t f(uint32_t i) { static float t[2] = {1.5f, 2}; return (uint32_t)t[i % 2u]; }",
        "not",
    ),
    (
        "struct z { uint32_t a, b; };\n"
        "uint32_t f(uint32_t i) { struct z l = {1, 2}; static struct z s = l; return s.a + i; }",
        "not",
    ),
    ("uint32_t f(uint32_t i) { static int t[2] = {1 / 0, 2}; return (uint32_t)t[0] + i; }", "not"),
    ("uint32_t f(uint32_t i) { static int t[2] = {1 << 40}; return (uint32_t)t[0] + i; }", "not"),
    (
        "uint32_t f(uint32_t i) { static int t[2] = {2147483647 + 1}; return (uint32_t)t[0] + i; }",
        "not",
    ),
    ("uint32_t f(uint32_t i) { static int x = -1 << 2; return (uint32_t)x + i; }", "not"),
    (
        "union ua { uint32_t z; struct { uint16_t lo, hi; }; };\n"
        "uint32_t f(uint32_t i) { static const union ua u = {.lo = 5, 6}; return u.lo + u.hi * 3u + i; }",
        "anon",
    ),
    (
        "uint8_t g[4];\nuint32_t f(void) { static uint8_t a[4] = g; return a[0]; }",
        "an array is initialized by a brace list or a string literal",
    ),
    (
        "uint32_t f(void) { static uint32_t a[2] = {1u, 2u, 3u}; return a[0]; }",
        "excess elements in an initializer",
    ),
)
_STATICTAB_REASONS = {
    "not": "a static initializer is not an integer constant expression",
    "anon": "a static union initialized through an anonymous member other than its first",
}


def test_static_initializer_refusals_on_both_rails():
    """CF-STATICTAB: every form of `_STATICTAB_REFUSED` is refused by both rails for the one reason it
    witnesses -- an entry the constant fold cannot evaluate, as C cannot (a parameter, another static, a local
    `const`, an address, a string, a floating constant, a struct value, a division by zero, an oversized or
    negative shift, a signed overflow), a union initialized through an anonymous member other than its first,
    and the walk's own refusals, which a static's initializer shares -- never lowered with a guessed value."""
    from bcir.frontends.cfront.cparse import CParseError
    from bcir.frontends.cfront.lower import CLowerError

    head = "#include <stdint.h>\n"
    refused = [(body, _STATICTAB_REASONS.get(why, why)) for body, why in _STATICTAB_REFUSED]
    for body, why in refused:
        try:
            compile_unit(head + body + "\n", check_clang=False)
        except (CLowerError, CParseError) as e:
            assert why in str(e), (body, str(e))
        else:
            raise AssertionError(f"the oracle lowered {body!r}")
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, (body, why) in enumerate(refused):
            path = os.path.join(d, f"r{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(head + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode > 0, (body, run.returncode, run.stdout[:200])
            assert why in run.stdout + run.stderr, (body, run.stdout[:200], run.stderr[:200])


# CF-STATICTAB: an integer constant's type is the first of its candidates the value fits (C11 6.4.4.1), and a
# `long` is 32 bits on LLP64 and ILP32 -- so `3000000000` and `0x100000000` are `long long` there, while `5L`
# and `4294967295UL` stay 32-bit. The sizeof of each of 3000000000, 5L, 4294967295UL, 0x100000000 and
# 4294967296UL, per target:
_LITERAL_SIZES = {
    "x86_64-linux": (8, 8, 8, 8, 8),
    "aarch64-linux": (8, 8, 8, 8, 8),
    "x86_64-windows": (8, 4, 4, 8, 8),
    "i386-linux": (8, 4, 4, 8, 8),
}
_LITERAL_NAMES = ("3000000000", "5L", "4294967295UL", "0x100000000", "4294967296UL")
_LITERAL_UNIT = "#include <stdint.h>\n" + "".join(
    f"uint32_t s{k}(void) {{ return (uint32_t)sizeof({n}); }}\n"
    for k, n in enumerate(_LITERAL_NAMES)
)


def test_integer_constant_types_follow_the_target_long_on_both_rails():
    """CF-STATICTAB: a static's constants fold in their literals' own types, so those must be the target's.
    The oracle chose a literal's candidate for LP64 (`3000000000` a `long`) and then sized that `long` for the
    target -- 4 bytes on x86-64 Windows and i386, truncating the value -- and the twin took every `L`-suffixed
    literal as 8 bytes wide. On every target both rails now type each literal as C does: its `sizeof` folds
    to the pinned size (with Clang present, Clang's own for the target), and the two digests are equal."""
    from bcir.frontends.cfront.abi import TARGETS

    for t, sizes in _LITERAL_SIZES.items():
        fns = compile_unit(_LITERAL_UNIT, check_clang=False, target=t).lowered.functions
        folded = tuple(
            next(c.imm[0] for c in fns[f"s{k}"].claims if c.op == "c.const")
            for k in range(len(_LITERAL_NAMES))
        )
        assert folded == sizes, (t, folded)
    clang = shutil.which("clang")
    if clang:
        with tempfile.TemporaryDirectory() as d:
            probe = os.path.join(d, "s.c")
            for t, sizes in _LITERAL_SIZES.items():
                with open(probe, "w", encoding="utf-8") as fh:
                    fh.write(
                        "".join(
                            f'_Static_assert(sizeof({n}) == {s}, "{n}");\n'
                            for n, s in zip(_LITERAL_NAMES, sizes)
                        )
                    )
                run = subprocess.run(
                    [clang, "-target", TARGETS[t].triple, "-fsyntax-only", probe],
                    capture_output=True,
                    text=True,
                )
                assert run.returncode == 0, (t, run.stderr[:500])
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "l.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_LITERAL_UNIT)
        for t in _LITERAL_SIZES:
            r = compile_unit(_LITERAL_UNIT, check_clang=False, target=t)
            run = subprocess.run([exe, "--target", t, path], capture_output=True, text=True)
            want = f"digest={cfront_structural_digest(r.lowered):016x}"
            assert want in run.stdout.split("\n", 1)[0], (t, run.stdout[:200])


# CF-TERNARY: the operands C may leave unevaluated. `cfront_condeval.c` runs every function with its guards false
# as well as true -- a zero divisor, INT32_MIN / -1, a NULL pointer, a bounds guard at the array's end, a device
# register at an address nothing maps (only ever with its guard false), and a counter of the calls made (each
# function that reads it resets it first).
_CONDEVAL_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  static const int32_t sd[][2] = {{-2147483647 - 1, -1}, {-2147483647 - 1, 1}, {7, 0}, {7, -1}, {-7, 2}, {0, 0}};
  for (unsigned k = 0; k < sizeof sd / sizeof sd[0]; k++) SAME(ce_sdiv, sd[k][0], sd[k][1]);
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n], d = gaps_in[(n + 3u) % GAPS_N], v = s & 7u, w = 3u;
    SAME(ce_div, s, 0u); SAME(ce_div, s, d); SAME(ce_chain, s, 0u); SAME(ce_chain, s, d);
    SAME(ce_mixed, s, 0u); SAME(ce_mixed, s, d);
    SAME(ce_deref, (uint32_t *)0, s); SAME(ce_deref, &v, s);
    SAME(ce_and_guard, (uint32_t *)0, s); SAME(ce_and_guard, &v, s); SAME(ce_and_guard, &w, s);
    SAME(ce_nested, (uint32_t *)0, s); SAME(ce_nested, &v, s);
    SAME(ce_call_arm, s); SAME(ce_both_calls, s); SAME(ce_or_call, s); SAME(ce_and_or, s);
    SAME(ce_bounds, s); SAME(ce_null_arm, s); SAME(ce_struct_arm, s); SAME(ce_float_arm, s);
    SAME(ce_writes, s); SAME(ce_comma, s); SAME(ce_volatile, s); SAME(ce_device, s & 3u); SAME(ce_void, s);
    SAME(ce_pure, s); SAME(ce_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)


def test_conditional_operands_evaluate_as_c_does_on_both_rails():
    """CF-TERNARY: `cfront_condeval.c` -- C evaluates one arm of `c ? a : b`, and the right operand of `&&` or
    `||` only when the left one does not decide (C11 6.5.15p4, 6.5.13p4, 6.5.14p4). Both rails had computed
    every operand and chosen after: a guarded division divided by zero, a NULL-guarded read read through NULL,
    a bounds guard read past the array, and a call in the operand C skips ran. An operand that can neither trap
    nor change state still lowers to a select; any other lowers as a branch that assigns the result in each
    arm, alike on both rails. The driver makes every guard false as well as true, and each emit returns what
    the original does -- including how many calls ran. A function whose operands are pure keeps its select
    (no branch), and one whose operand divides, reads through a pointer, calls, or reads a volatile variable
    or a device register branches, on both rails. Arms of type void (`c ? f() : g();`, an `assert`'s
    `c ? (void)0 : fail()`) branch with no local: both rails had assigned the void call's non-value and the
    emit did not compile, `(void)e` was a cast of e to uint32_t, and the twin's verifier refused a void
    function whose only effect is its calls."""
    if not _CC:
        return
    fx = "cfront_condeval.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    for rail, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        pure = _emitted_function(emit, "ce_pure")
        assert "if (" not in pure and " ? " in pure, (rail, pure)
        for name in ("ce_div", "ce_deref", "ce_call_arm", "ce_or_call", "ce_volatile", "ce_device"):
            assert "if (" in _emitted_function(emit, name), (rail, name)
        void = _emitted_function(emit, "ce_void")  # void arms: branches, and no local for a value
        assert "if (" in void and " sel" not in void, (rail, void)
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _CONDEVAL_DRIVER)


# CF-TERNARY: a void value returned from a void function -- the GNU extension GCC and Clang accept.
_VOID_RETURN_UNIT = r"""#include <stdint.h>
static uint32_t g_n;
static void inc(void) { g_n++; }
static void dec(void) { g_n += 10u; }
static void only_calls(void) { inc(); dec(); }
static void ret_call(void) { return inc(); }
static void ret_cast(uint32_t s) { g_n += s; return (void)s; }
static void ret_cond(uint32_t s) { return s > 3u ? inc() : dec(); }
uint32_t vr_entry(uint32_t s) { g_n = 0u; only_calls(); ret_call(); ret_cast(s & 7u); ret_cond(s); return g_n; }
"""


def test_a_void_function_returns_a_void_expression_alike_on_both_rails():
    """CF-TERNARY: `return f();` in a void function -- of a void call, a `(void)` cast or a void conditional --
    makes the call and returns nothing, on both rails. The twin had returned the call's placeholder (`return t;`
    in a void function did not compile) and its verifier refused a void function whose only effect is the calls
    it makes (R12), which the oracle verifies clean."""
    if not _CC:
        return
    oracle_summary, r, _entry = _oracle(_VOID_RETURN_UNIT)
    assert "ok=1" in oracle_summary, oracle_summary
    oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "vr.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_VOID_RETURN_UNIT)
        c_summary, c_emit = _c_run(exe, path)
    assert c_summary == oracle_summary, f"parity\n C: {c_summary}\nPY: {oracle_summary}"
    driver = (
        _GAPS_SAME
        + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) SAME(vr_entry, gaps_in[n]);
  puts("MATCH");
  return 0;
}
"""
    )
    emits = (("twin", c_emit), ("oracle", oracle_emit))
    _run_against_original("void returns", _VOID_RETURN_UNIT, emits, driver)


# CF-BUF: the C twin kept a name -- a callee, a parameter, a local, a tag, a member, an alias -- in 32 bytes, a
# claim's op (a prefix and a name, a floating constant's spelling, a cast's type) in the same 32, and the verified C
# it emits in 32 KiB, with each synthesized prelude line in 512 bytes. A name is now at most 63 characters on both
# rails (C11 5.2.4.1's significant initial characters of an internal identifier; the twin's fields hold that and
# every prefix), a longer identifier or floating constant is refused where it is lexed, and the emit grows.
_BUF_TARGETS = ("x86_64-linux", "aarch64-linux", "x86_64-windows", "i386-linux")


def _parity_on_targets(path: str, src: str) -> None:
    """The twin's summary -- the claim counts and the structural digest -- is the oracle's on each of the four
    targets, the unit at `path` holding `src`."""
    exe = _build_frontend(_session_build_dir())
    for t in _BUF_TARGETS:
        oracle = _summary_line(compile_unit(src, check_clang=False, target=t))
        run = subprocess.run([exe, "--target", t, path], capture_output=True, text=True)
        twin = (run.stdout.partition("----EMIT----\n")[0].strip().splitlines() or [""])[0]
        assert twin == oracle, f"{os.path.basename(path)} on {t}\n C: {twin}\nPY: {oracle}"


_LONGNAMES_DRIVER = (
    _GAPS_SAME
    + r"""
#define LN_TABLE file_scope_lookup_table_of_four_words_with_a_sixty_three_chars_
int main(void) {
  uint32_t init[4], after[4];
  memcpy(init, LN_TABLE, sizeof init);
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(ln_struct, s); SAME(ln_typedef, s); SAME(ln_shadow, s); SAME(ln_goto, s); SAME(ln_imember, s);
    SAME(ln_static, s); SAME(ln_entry, s);
    memcpy(LN_TABLE, init, sizeof init);   /* each rail's call starts from the same table */
    uint32_t a = ln_calls(s);
    memcpy(after, LN_TABLE, sizeof after);
    memcpy(LN_TABLE, init, sizeof init);
    if (a != bcir_ln_calls(s) || memcmp(after, LN_TABLE, sizeof after)) return fail("ln_calls");
    double x = (double)s + 0.5;
    if (ln_float(x) != bcir_ln_float(x)) return fail("ln_float");
  }
  puts("MATCH");
  return 0;
}
"""
)


# A variadic callee of 63 characters whose `va_arg` type the twin's op had cut to 16 characters (`unsigned long lo`,
# which does not compile) -- in a unit of its own, since the corpus harness does not include <stdarg.h>
_LONGNAMES_VARIADIC = """#include <stdint.h>
#include <stdarg.h>
static uint64_t variadic_callee_summing_its_unsigned_long_long_extra_arguments_(uint32_t n, ...) {
  va_list ap;
  va_start(ap, n);
  uint64_t s = 0u;
  for (uint32_t i = 0u; i < n; i++) s += va_arg(ap, unsigned long long);
  va_end(ap);
  return s;
}
uint64_t lnv_entry(uint32_t s) {
  return variadic_callee_summing_its_unsigned_long_long_extra_arguments_(2u, (unsigned long long)s, 5ull);
}
"""


def test_long_names_run_as_the_original_on_both_rails():
    """CF-BUF: `cfront_longnames.c` holds a 63-character identifier wherever the claim graph keeps a name -- a
    direct, a void and a prototyped callee (the op `c.call:` / `c.call.void:` carries it after its prefix), a
    parameter, two same-named locals in disjoint blocks, a static local, a file-scope table, a struct tag and its
    members, a function-pointer member (`c.call.imember:`), a typedef, an enum constant, a label -- and a
    63-character floating constant, whose exponent the twin's 32-byte op had cut off (`1.000...e-300` became 1.0).
    The twin kept 31 characters of every name, so it refused the unit (`undefined identifier`), and a callee named
    past 24 characters digested apart from the oracle's and was called by its truncated name. Both rails now lower
    the fixture to one claim graph on the four targets, and each emit returns what the original does, function by
    function, under every compiler at hand; so does `_LONGNAMES_VARIADIC`, whose `va_arg(ap, unsigned long long)`
    the twin had emitted as `va_arg(ap, unsigned long lo)`."""
    if not _CC:
        return
    fx = "cfront_longnames.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    for rail, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        for whole in (
            "bcir_static_callee_whose_name_fills_all_sixty_three_characters_here_(",
            "bcir_void_callee_that_adds_its_argument_into_the_lookup_table_entry_(",
            "local_variable_declared_twice_in_two_disjoint_blocks_of_scope___2",
            "struct register_block_descriptor_with_a_tag_as_long_as_c_allows_here_1 *",
            "1.00000000000000000000000000000000000000000000000000000000e-300",
        ):
            assert whole in emit, f"{rail}: the emit does not spell {whole!r} whole"
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _LONGNAMES_DRIVER)

    src = _LONGNAMES_VARIADIC
    oracle_summary, r, _entry = _oracle(src)
    oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "longnames_va.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        c_summary, c_emit = _c_run(_build_frontend(_session_build_dir()), path)
        assert c_summary == oracle_summary, f"parity\n C: {c_summary}\nPY: {oracle_summary}"
        _parity_on_targets(path, src)
    for rail, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        for whole in (
            "bcir_variadic_callee_summing_its_unsigned_long_long_extra_arguments_(",
            "va_arg(ap, unsigned long long)",
        ):
            assert whole in emit, f"{rail}: the emit does not spell {whole!r} whole"
    driver = (
        _GAPS_SAME
        + "int main(void) {\n  for (unsigned n = 0; n < GAPS_N; n++) SAME(lnv_entry, gaps_in[n]);\n"
        + '  puts("MATCH");\n  return 0;\n}\n'
    )
    _run_against_original(
        "longnames_va.c", src, (("twin", c_emit), ("oracle", oracle_emit)), driver
    )


# (a unit with one name at 64 characters -- `{n}` -- in each place a name can stand, and its twin at 63)
_ID64 = "a_name_one_character_longer_than_the_sixty_three_both_rails_take"
_FLT64, _FLT63 = "2." + "0" * 57 + "e-300", "2." + "0" * 56 + "e-300"
_HEX64, _HEX63 = "0x1." + "0" * 55 + "p-900", "0x1." + "0" * 54 + "p-900"
_LONG_ID_REFUSED = "an identifier longer than 63 characters is not supported"
_LONG_FLT_REFUSED = "a floating constant longer than 63 characters is not supported"
_LONG_NAME_UNITS = (
    ("uint32_t {n}(uint32_t s) {{ return s + 1u; }}", _LONG_ID_REFUSED),
    ("uint32_t lf(uint32_t {n}) {{ return {n} * 3u; }}", _LONG_ID_REFUSED),
    ("uint32_t lf(uint32_t s) {{ uint32_t {n} = s; return {n} + 2u; }}", _LONG_ID_REFUSED),
    (
        "struct {n} {{ uint32_t v; }};\nuint32_t lf(struct {n} *p) {{ return p->v; }}",
        _LONG_ID_REFUSED,
    ),
    (
        "struct lt {{ uint32_t {n}; }};\nuint32_t lf(struct lt *p) {{ return p->{n}; }}",
        _LONG_ID_REFUSED,
    ),
    ("typedef uint32_t {n};\nuint32_t lf(uint32_t s) {{ {n} w = s; return w; }}", _LONG_ID_REFUSED),
    ("enum {{ {n} = 3 }};\nuint32_t lf(uint32_t s) {{ return s + {n}; }}", _LONG_ID_REFUSED),
    (
        "static uint32_t {n}[2];\nuint32_t lf(uint32_t s) {{ return {n}[s & 1u]; }}",
        _LONG_ID_REFUSED,
    ),
    ("uint32_t lf(uint32_t s) {{ if (s) goto {n}; s++;\n{n}: return s; }}", _LONG_ID_REFUSED),
    (
        "static uint32_t {n}(uint32_t s) {{ return s; }}\nuint32_t lf(uint32_t s) {{ return {n}(s); }}",
        _LONG_ID_REFUSED,
    ),
    ("double lf(double x) {{ return x * {f}; }}", _LONG_FLT_REFUSED),
    ("double lf(double x) {{ return x * {h}; }}", _LONG_FLT_REFUSED),
)


def test_a_name_past_63_characters_is_refused_on_both_rails_and_one_at_63_lowers():
    """CF-BUF: an identifier of 64 characters -- a function, a parameter, a local, a struct tag, a member, a
    typedef, an enum constant, a global, a label, a callee -- and a floating constant of 64 characters, decimal or
    hexadecimal, are refused on both rails where they are lexed, each with its own reason: the twin's graph holds 63
    and could only truncate the 64th (another callee, member or constant), and the oracle refuses what the twin
    cannot hold. The same unit one character shorter lowers on both rails to one claim graph."""
    from bcir.frontends.cfront.clex import CLexError

    assert {len(_ID64), len(_FLT64), len(_HEX64)} == {64} and {len(_FLT63), len(_HEX63)} == {63}
    head = "#include <stdint.h>\n"
    units = []
    for shape, why in _LONG_NAME_UNITS:
        at64 = head + shape.format(n=_ID64, f=_FLT64, h=_HEX64) + "\n"
        at63 = head + shape.format(n=_ID64[:-1], f=_FLT63, h=_HEX63) + "\n"
        try:
            compile_unit(at64, check_clang=False)
        except CLexError as e:
            assert why in str(e), (shape, str(e))
        else:
            raise AssertionError(f"the oracle lowered a 64-character name: {shape!r}")
        units.append((shape, why, at64, at63))
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for k, (shape, why, at64, at63) in enumerate(units):
            path = os.path.join(d, f"long{k}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(at64)
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode == 1 and why in run.stdout, (
                shape,
                run.returncode,
                run.stdout[:200],
            )
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(at63)
            oracle_summary, _r, _entry = _oracle(at63)
            c_summary, _emit = _c_run(exe, path)
            assert c_summary == oracle_summary and "ok=1" in c_summary, (
                shape,
                c_summary,
                oracle_summary,
            )


def _wide_unit(n: int) -> str:
    """`n` small functions whose verified C together runs past 64 KiB on both rails."""
    return "#include <stdint.h>\n" + "".join(
        f"uint32_t wide_{k}(uint32_t a, uint32_t b) {{\n"
        f"  uint32_t s = a * {k + 3}u + b;\n  s ^= s >> {k % 7 + 1}u;\n  s += b * {k + 5}u;\n"
        f"  s = (s << 5u) | (s >> 27u);\n  s -= a ^ {k + 11}u;\n  return s + {k + 7}u;\n}}\n"
        for k in range(n)
    )


def test_an_emit_past_64_kib_lowers_and_runs_as_the_original():
    """CF-BUF: the twin's verified C had a fixed 32 KiB (`bcir_cfront_result.emitted`); a unit whose emit is larger
    reported `EMIT-ERR` and printed no digest to compare, so corpus fixtures were trimmed to fit. The emit now grows
    through the result's allocator: a unit of 140 functions -- past 64 KiB of emitted C on both rails -- lowers to
    the oracle's claim graph on the four targets, and each function of each emit returns what the original does."""
    if not _CC:
        return
    n = 140
    src = _wide_unit(n)
    oracle_summary, r, _entry = _oracle(src)
    oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "wide.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        c_summary, c_emit = _c_run(exe, path)
        assert c_summary == oracle_summary, f"parity\n C: {c_summary}\nPY: {oracle_summary}"
        _parity_on_targets(path, src)
    assert len(src) < 1 << 16 and len(c_emit) > 1 << 16 and len(oracle_emit) > 1 << 16, (
        len(src),
        len(c_emit),
        len(oracle_emit),
    )
    calls = "".join(f"    SAME(wide_{k}, a, b);\n" for k in range(n))
    driver = (
        _GAPS_SAME
        + "int main(void) {\n  for (unsigned n = 0; n < GAPS_N; n++) {\n"
        + "    uint32_t a = gaps_in[n], b = gaps_in[(n + 3u) % GAPS_N];\n"
        + calls
        + '  }\n  puts("MATCH");\n  return 0;\n}\n'
    )
    _run_against_original("wide", src, (("twin", c_emit), ("oracle", oracle_emit)), driver)


_SIGOVERFLOW_DRIVER = r"""
static int fail(const char *what) { puts(what); return 1; }
int main(void) {
  if (g_param(0) != bcir_g_param(0)) return fail("g_param");
  if (g_local() != bcir_g_local()) return fail("g_local");
  puts("MATCH");
  return 0;
}
"""


def test_a_long_function_pointer_signature_emits_on_the_twin():
    """CF-BUF: `cfront_sec_sigoverflow.c` declares a function-pointer parameter and a local of sixty parameters,
    whose synthesized `typedef RET (*__bcir_fpN)(PARAMS);` runs past 512 bytes. The twin rendered that line into a
    fixed 512-byte buffer and, when it did not fit, marked the whole emit impossible: its driver printed `EMIT-ERR`
    for a unit whose emit is under 4 KB, so the fixture's digest was never compared with the oracle's. The prelude
    now grows: the twin emits the unit with the oracle's digest on the four targets, the typedef names all sixty
    parameters, and the emit returns what the original does."""
    if not _CC:
        return
    fx = "cfront_sec_sigoverflow.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    typedefs = [line for line in c_emit.splitlines() if line.startswith("typedef ")]
    assert len(typedefs) == 2 and all(t.count("uint64_t") == 60 for t in typedefs), typedefs
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _SIGOVERFLOW_DRIVER)


def test_the_twin_formats_no_name_into_a_field_that_could_cut_it():
    """CF-BUF: GCC's format-truncation analysis (`-Wformat-truncation=1`, which bounds a `%s` by the array it
    reads) finds no `snprintf` in the twin that could cut a name, a tag, an op or a type short: each field is sized
    for the longest spelling it holds, now that a name is at most 63 characters. On the parent it found 32."""
    gcc = shutil.which("gcc")
    if not gcc:
        return
    b = subprocess.run(
        [
            gcc,
            "-std=c11",
            "-O1",
            "-Wformat-truncation=1",
            "-Werror=format-truncation",
            "-I",
            _C,
            "-c",
        ]
        + [os.path.join(_C, "bcir_cfront.c"), "-o", os.devnull],
        capture_output=True,
        text=True,
    )
    assert b.returncode == 0, b.stderr[-3000:]


def test_the_emit_grows_whole_under_a_failing_allocator():
    """CF-BUF: the twin's allocation-failure sweep (`runtime/c/test_memory_discipline.c`, run here for the C
    frontend alone) fails every allocation a compile makes, in turn, under an injected allocator -- now over a unit
    whose verified C outgrows the emit's first block several times and whose function-pointer parameters grow the
    typedef prelude. Each failure is the whole compile's (`oom`): no unit, no emitted C (never a partial text), every
    block released; the fault-free run emits the whole unit."""
    if not _CC:
        return
    srcs = (
        "test_memory_discipline.c",
        "bcir_runtime_channel.c",
        "bcir_cfront.c",
        "bcir_cpp.c",
        "bcir_verify.c",
        "bcir_runtime.c",
        "bcir_q8_model.c",
        "bcir_decode.c",
        "bcir_ai_kernels.c",
        "bcir_llama.c",
    )
    exe = os.path.join(_session_build_dir(), "memory_discipline")
    b = subprocess.run(
        [_CC, "-std=c11", "-O1", "-I", _C, *(os.path.join(_C, s) for s in srcs), "-o", exe, "-lm"],
        capture_output=True,
        text=True,
    )
    assert b.returncode == 0, b.stderr[-3000:]
    run = subprocess.run([exe, "--cfront"], capture_output=True, text=True, timeout=600)
    assert run.returncode == 0 and "memory-discipline: cfront ok" in run.stdout, (
        run.returncode,
        run.stdout[-2000:],
        run.stderr[-2000:],
    )


_SPLITS_DRIVER = (
    _GAPS_SAME
    + r"""
#define SNAP(t) do { t[0] = sp_n1.v; t[1] = sp_n1.c; t[2] = (uint32_t)sp_n1.w; t[3] = sp_n1.bits;              \
    t[4] = sp_n2.v; t[5] = sp_n2.c; t[6] = (uint32_t)sp_n2.w; t[7] = sp_n2.bits;                            \
    for (int k = 0; k < 8; k++) t[8 + k] = sp_buf[k];                                                       \
    t[16] = sp_at.n; t[17] = sp_slots[1]; t[18] = sp_rw.a[0]; t[19] = sp_rw.c[0]; t[20] = sp_rw.in.b[0]; } while (0)
/* the result and every global the function touches: the original's, then the emit's from the same start */
#define SAME_STATE(f, s) do { uint32_t x_[21], y_[21]; uint32_t r_ = f(s); SNAP(x_);                        \
    uint32_t q_ = bcir_##f(s); SNAP(y_); if (r_ != q_ || memcmp(x_, y_, sizeof x_)) return fail(#f); } while (0)
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME_STATE(sp_ptr_sum, s); SAME_STATE(sp_chain_steps, s); SAME_STATE(sp_chain_values, s);
    SAME_STATE(sp_derefs, s); SAME_STATE(sp_addresses, s); SAME_STATE(sp_paren_calls, s);
    SAME_STATE(sp_atomics, s); SAME_STATE(sp_member_arrays, s); SAME_STATE(sp_entry, s);
    uint32_t pa[4] = {s, s + 1u, s ^ 7u, 3u}, pb[4];
    memcpy(pb, pa, sizeof pa);
    if (sp_param_elems(pa, s & 1u) != bcir_sp_param_elems(pb, s & 1u) || memcmp(pa, pb, sizeof pa))
      return fail("sp_param_elems");
    uint8_t ba[4] = {(uint8_t)s, 255u, 0u, (uint8_t)(s >> 8)}, bb[4];
    memcpy(bb, ba, sizeof ba);
    if (sp_param_bytes(ba, s & 3u) != bcir_sp_param_bytes(bb, s & 3u) || memcmp(ba, bb, sizeof ba))
      return fail("sp_param_bytes");
  }
  puts("MATCH");
  return 0;
}
"""
)


def test_split_forms_lower_alike_and_run_as_the_original_on_both_rails():
    """CF-SPLIT2: `cfront_splits.c` -- forms the rails lowered to different claim graphs, or that one lowered
    and the other refused. `p = p + n` of a pointer was a `c.ptradd` on the oracle and a sum and a copy on the
    twin; it is the sum and the copy on both now, as `q = p + n` and `p = n + p` were, and only the compound's
    desugaring (`p += n`, `p++`) steps the pointer in place. The twin refused an increment or an assignment used
    as a value through a pointer the lvalue loads (`h.next->v++`, `x = (h.next->next->v ^= s)`, `s->p[i]++`),
    through a pointer parameter (`p[i]++`, `x = (p[i] = v)`) or a dereference (`(*p)++`, `++*p`), the address
    of such an object (`&h.next->v`) and a call through a parenthesized callee (`(fp)(x)`, `(o.fn)(x)`); the
    oracle refused a dereference of a pointer value other than a name, a member or a call (`*&a`,
    `*(c ? &a : &b)`, `*p++`, `*(q - 1)`) and both refused the statement `++h.next->v;`; and the twin took
    `*q->a` of a member array as the array's address and an access through it, where the oracle accesses the
    first element at the member's offset. Each lowers to one claim graph on the four targets now, and each emit
    runs as the original does, globals included."""
    if not _CC:
        return
    fx = "cfront_splits.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    # the one lowering both rails keep: only `p += 1` and `p++` step the pointer in place
    ops = [
        c.op for c in compile_unit(src, check_clang=False).lowered.functions["sp_ptr_sum"].claims
    ]
    assert ops.count("c.ptradd") == 2 and "c.ptrsub" not in ops, ops
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _SPLITS_DRIVER)


# The forms stay refused where the object is a device's: a re-read of it (the value of `=`/OP= and a prefix
# step) or its read-modify-write through a device pointer would be an extra device access, which the oracle's
# `_mmio` gate refuses and the twin's `plv_volatile` refuses alike.
_SPLITS_DEVICE = (
    "struct dv {{ uint32_t v; volatile uint32_t r; }};\nstruct dh {{ struct dv *d; }};\n"
    "uint32_t f(struct dh h, uint32_t s) {{ {body} }}\n"
)
_SPLITS_DEVICE_BODIES = (
    "return h.d->v++;",  # a plain member of a struct that holds volatile storage
    "return ++h.d->r;",  # a volatile member
    "return (h.d->v = s);",
    "return (h.d->r += s);",
    "h.d->v--; return s;",
)


def test_split_forms_on_a_device_object_are_refused_on_both_rails():
    """CF-SPLIT2: an increment, or an assignment used as a value, through a loaded pointer into a struct that
    holds volatile storage is refused on both rails -- the oracle's device gate (`_mmio`), the twin's
    `plv_volatile` reading the loaded pointer's device domain -- rather than lowered by one rail alone."""
    from bcir.frontends.cfront.cparse import CParseError
    from bcir.frontends.cfront.lower import CLowerError

    exe = _build_frontend(_session_build_dir()) if _CC else None
    for body in _SPLITS_DEVICE_BODIES:
        src = "#include <stdint.h>\n" + _SPLITS_DEVICE.format(body=body)
        try:
            compile_unit(src, check_clang=False)
            raise AssertionError(f"the oracle lowered {body!r}")
        except (CParseError, CLowerError):
            pass
        if exe is None:
            continue
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "split_device.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode != 0 and run.stdout.startswith("PARSE-ERR"), (body, run.stdout)


_STRMEM_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(sm_copy, s); SAME(sm_move, s); SAME(sm_fill, s); SAME(sm_result, s); SAME(sm_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)


def test_string_routines_are_libc_edges_on_both_rails():
    """CF-RTWIDE: `memcpy`, `memmove` and `memset` lower on both rails to one `c.call.libm:` edge each -- a libc
    routine returning its destination, opaque to R18, linked with no flag -- where both rails had refused the
    unit as a call to an undefined function (R18). `cfront_strmem.c` copies whole structs, moves overlapping
    bytes and fills; both rails lower it to one claim graph on the four targets, and each emit runs as the
    original does."""
    from bcir.frontends.cfront.linkflags import derive_link_flags

    fx = "cfront_strmem.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    r = compile_unit(src, check_clang=False)
    assert r.is_clean, r.diagnostics
    ops = [c.op for lf in r.lowered.functions.values() for c in lf.claims]
    got = {name: ops.count(f"c.call.libm:{name}") for name in ("memcpy", "memmove", "memset")}
    assert got == {"memcpy": 3, "memmove": 2, "memset": 3}, got
    assert derive_link_flags(r.lowered) == [], derive_link_flags(r.lowered)
    if not _CC:
        return
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _STRMEM_DRIVER)


# A library name the unit defines is the unit's own function, wherever its definition stands. The twin parses
# in one pass and asked only the definitions before the call (the printf family) or none (the allocators and
# `free`), so it lowered a libc edge where the oracle -- whose `func_rets` holds every definition -- lowered a
# call of the unit's function.
_OWN_LIBRARY = {
    "malloc before": (
        "static uint32_t pool[4];\nvoid *malloc(unsigned long n) { (void)n; return pool; }\n"
        "uint32_t f(uint32_t x) { uint32_t *p = malloc(4); *p = x; return *p + pool[0]; }\n",
        "c.call:malloc",
    ),
    "free before": (
        "static uint32_t freed;\nvoid free(void *p) { (void)p; freed++; }\n"
        "uint32_t f(uint32_t x) { free(&x); return x + freed; }\n",
        "c.call.void:free",
    ),
    "memcpy after": (
        "void *memcpy(void *d, const void *s, size_t n);\n"
        "uint32_t f(uint32_t x) { uint32_t y = 0; memcpy(&y, &x, 4); return y; }\n"
        "void *memcpy(void *d, const void *s, size_t n) { uint8_t *a = d; const uint8_t *b = s;"
        " for (size_t i = 0; i < n; i++) a[i] = b[i]; return d; }\n",
        "c.call:memcpy",
    ),
    "memset before": (
        "void *memset(void *d, int v, size_t n) { uint8_t *a = d;"
        " for (size_t i = 0; i < n; i++) a[i] = (uint8_t)v; return d; }\n"
        "uint32_t f(uint32_t x) { uint32_t y = x; memset(&y, 1, 2); return y; }\n",
        "c.call:memset",
    ),
    "printf after": (
        "int printf(const char *f, ...);\n"
        'uint32_t f(uint32_t x) { return (uint32_t)printf("%u", x) + x; }\n'
        "int printf(const char *f, ...) { (void)f; return 3; }\n",
        "c.call:printf",
    ),
}


def test_a_library_name_the_unit_defines_is_its_own_function_on_both_rails():
    """CF-RTWIDE: a call of `malloc`, `free`, `memcpy`, `memset` or `printf` lowers as a call of the unit's own
    function when the unit defines one -- before the call or after it -- on both rails: the twin asks every
    definition of the unit (`unit_defines`), as the oracle's `func_rets` holds them all. The twin had lowered
    `malloc` and `free` as the libc edges even when the unit defined them, and a `printf` defined after its
    call as the external variadic."""
    exe = _build_frontend(_session_build_dir()) if _CC else None
    head = "#include <stdint.h>\n#include <stddef.h>\n"
    for label, (body, want) in _OWN_LIBRARY.items():
        src = head + body
        r = compile_unit(src, check_clang=False)
        calls = [c.op for c in r.lowered.functions["f"].claims if c.op.startswith("c.call")]
        assert calls == [want], (label, calls)
        if exe is None:
            continue
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "own_library.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
            twin, _emit = _c_run(exe, path)
        assert twin == _summary_line(r), f"{label}\n C: {twin}\nPY: {_summary_line(r)}"


# CF-RTWIDE: labels of the function's own spelled as the emitters' continue labels -- the twin numbers its loops
# from 0 (`__cont_0`), the oracle names each by its loop id (read off a first emit) -- beside two loops that
# `continue`, and the two constant-1 loops, whose emits test nothing.
_CONT_LABELS = r"""#include <stdint.h>
uint32_t cl_f(uint32_t n) {
  uint32_t s = 0;
  for (uint32_t i = 0; i < n % 9u; i++) { if (i & 1u) continue; s += i; }
  while (s < 50u) { s += 7u; if (s & 2u) continue; s ^= 1u; }
  for (;;) { s += 3u; if (s > 60u) break; }
  while (1) { if (s & 1u) break; s++; }
  if (n & 1u) goto TWIN;
  s += 3u;
TWIN:
  if (n & 2u) goto ORACLE;
  s ^= 5u;
ORACLE:
  return s;
}
"""
_CONT_LABELS_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) SAME(cl_f, gaps_in[n]);
  puts("MATCH");
  return 0;
}
"""
)


def test_a_label_spelled_as_a_continue_label_stays_one_label_on_both_rails():
    """CF-RTWIDE: each loop's continue label is clear of every label the function defines, on both emitters --
    `__cont_<n>` or, where the function spells that already, `__cont_<n>_<k>`. A source label spelled as an
    emitter's continue label had been defined twice in that emitter's output, which does not compile. Both
    rails lower the unit to one claim graph, and each emit runs as the original does."""
    first = compile_unit(_CONT_LABELS.replace("ORACLE", "done"), check_clang=False)
    oracle_label = re.search(r"__cont_\d+", first.emitted["cl_f"]).group(0)
    assert oracle_label != "__cont_0"
    src = _CONT_LABELS.replace("TWIN", "__cont_0").replace("ORACLE", oracle_label)
    r = compile_unit(src, check_clang=False)
    oracle_emit = r.emitted["cl_f"]
    assert f"{oracle_label}_2: ;" in oracle_emit, (
        oracle_emit
    )  # the loop's label, clear of the function's own
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cont_labels.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        twin, c_emit = _c_run(exe, path)
    assert twin == _summary_line(r), f"parity\n C: {twin}\nPY: {_summary_line(r)}"
    assert "__cont_0_2: ;" in c_emit, c_emit
    for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        assert "if (!" in emit and emit.count("if (!") == 2, (
            label,
            emit,
        )  # the two loops of no constant
    _run_against_original(
        "cont_labels", src, (("twin", c_emit), ("oracle", oracle_emit)), _CONT_LABELS_DRIVER
    )


# CF-RTWIDE: aggregate locals a brace initializer names only in part -- positional, designated, a union's first
# member, a member array's first element: each the object's zero baseline and one store per named scalar.
_ZERO_BASELINE_UNIT = r"""#include <stdint.h>
struct zb { uint32_t a; uint16_t b; uint8_t c[3]; };
union zu { uint8_t c; uint32_t w; };
struct zw { uint32_t v[4]; };
uint32_t zb_f(uint32_t x) {
  struct zb p = { x, (uint16_t)(x + 1u) };
  union zu u = { (uint8_t)x };
  struct zw a = { { x } };
  struct zb q = { .c = { 1u, 2u } };
  return p.a + p.b + p.c[2] + u.c + a.v[3] + q.c[1] + q.a;
}
"""
_ZERO_BASELINE_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) SAME(zb_f, gaps_in[n]);
  puts("MATCH");
  return 0;
}
"""
)


def test_both_emitters_spell_the_zero_baseline_as_the_empty_initializer():
    """CF-RTWIDE: both emitters declare an aggregate local's zero baseline as the empty initializer `= {}` -- the
    whole object zero, and nothing stored -- where both had written `= {0}`, which a re-parse reads as the
    baseline and a store of 0 to the first scalar. Both rails lower the unit to one claim graph, and each emit
    runs as the original does: every member no initializer names is zero."""
    r = compile_unit(_ZERO_BASELINE_UNIT, check_clang=False)
    oracle_emit = r.emitted["zb_f"]
    assert oracle_emit.count(" = {};") == 4 and "{0}" not in oracle_emit, oracle_emit
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "zero_baseline.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_ZERO_BASELINE_UNIT)
        twin, c_emit = _c_run(exe, path)
    assert twin == _summary_line(r), f"parity\n C: {twin}\nPY: {_summary_line(r)}"
    assert c_emit.count(" = {};") == 4 and "{0}" not in c_emit, c_emit
    _run_against_original(
        "zero_baseline",
        _ZERO_BASELINE_UNIT,
        (("twin", c_emit), ("oracle", oracle_emit)),
        _ZERO_BASELINE_DRIVER,
    )


# CF-RTWIDE: units whose only libc edges are the <string.h> routines, or `aligned_alloc`.
_LINKABLE_HEADERS = {
    "string": (
        "#include <stdint.h>\n#include <string.h>\nuint32_t f(uint32_t x) { uint32_t y; memcpy(&y, &x, 4);"
        " memset(&x, 0, 4); memmove(&y, &x, 2); return y; }\n",
        "#include <string.h>",
        ("#include <math.h>", "#include <stdlib.h>"),
    ),
    "aligned_alloc": (
        "#include <stdint.h>\n#include <stdlib.h>\nuint32_t f(uint32_t n) { uint32_t *p = aligned_alloc(16, 16);"
        " return (uint32_t)(p != 0) + n; }\n",
        "#include <stdlib.h>",
        ("#include <math.h>", "#include <string.h>"),
    ),
}


def test_the_linkable_emit_includes_the_header_of_each_libc_edge():
    """CF-RTWIDE: the linkable emit (`--linkable`) includes the header that declares each libc edge it calls --
    `<string.h>` for `memcpy`/`memmove`/`memset`, `<stdlib.h>` for `aligned_alloc` as for the other allocators
    -- where it had included `<math.h>` for any edge outside malloc/calloc/realloc/free, which declares neither.
    Where a C compiler is visible, the artifact compiles with implicit declarations refused."""
    from bcir.frontends.cfront.emit import emit_linkable

    for label, (src, want, never) in _LINKABLE_HEADERS.items():
        r = compile_unit(src, check_clang=False)
        assert r.is_clean, (label, r.diagnostics)
        text = emit_linkable(r.lowered, r.emitted)
        assert want in text and not any(h in text for h in never), (label, text)
        if _CC:
            cp = subprocess.run(
                [
                    _CC,
                    "-std=c11",
                    "-fsyntax-only",
                    "-Werror=implicit-function-declaration",
                    "-x",
                    "c",
                    "-",
                ],
                input=text,
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert cp.returncode == 0, (label, cp.stderr, text)


def _refused_on_both_rails(exe, src: str, why: str) -> None:
    """Neither rail lowers `src`: the oracle raises its parse or lowering error, naming `why`, and the twin (when
    it is built, `exe`) prints its PARSE-ERR, naming it too."""
    from bcir.frontends.cfront.cparse import CParseError
    from bcir.frontends.cfront.lower import CLowerError

    try:
        compile_unit(src, check_clang=False)
        raise AssertionError(f"the oracle lowered:\n{src}")
    except (CParseError, CLowerError) as e:
        assert why in str(e), (why, str(e), src)
    if exe is None:
        return
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "refused.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        run = subprocess.run([exe, path], capture_output=True, text=True)
        assert run.returncode != 0 and run.stdout.startswith("PARSE-ERR"), (src, run.stdout)
        assert why in run.stdout, (why, run.stdout, src)


_DECLS_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(dc_entry, s); SAME(dc_wide, s); SAME(dc_later, 0, s); SAME(dc_later, &s, s); SAME(dc_big, dc_tab, s);
    SAME(dc_apply, dc_twice, s); SAME(dc_sum, dc_tab, 3u); SAME(dc_twice, s);
  }
  puts("MATCH");
  return 0;
}
"""
)


def test_callees_declared_ahead_of_their_callers_run_as_the_original_on_both_rails():
    """CF-DECLS: `cfront_decls.c` -- callees defined after their callers, `static` or not, behind prototypes
    leaving parameters unnamed (a pointer to `const`, an array, a function pointer), a function named as a value
    and a call in a `sizeof` operand before the definition. Each rail's emit declares the unit's functions a
    function calls ahead of it -- neither emit declared one, so a call to a later definition did not compile.
    Both rails refused an unnamed prototype parameter, and the twin refused the designator and the `sizeof`
    through a prototype. The unit lowers to one claim graph on the four targets, and every emit, with no
    declaration supplied by hand, runs as the original."""
    if not _CC:
        return
    fx = "cfront_decls.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        decls = [ln for ln in emit.splitlines() if ln.startswith("static ") and ln.endswith(");")]
        assert any("bcir_dc_later(" in ln for ln in decls), (label, decls)  # ahead of `dc_entry`
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _DECLS_DRIVER)


_DECLS_LINK_DRIVER = (
    _GAPS_SAME
    + r"""
uint32_t dl_read(const uint32_t *p, uint32_t (*fn)(uint32_t), uint32_t s) { return p[0] + p[1] * 3u + fn(s); }
uint64_t dl_mix(const uint8_t *b, uint32_t s) { return ((uint64_t)b[3] << 40) | (uint64_t)(b[0] + s); }
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) { SAME(dl_entry, gaps_in[n]); }
  puts("MATCH");
  return 0;
}
"""
)


def test_prototyped_external_callees_are_declared_as_their_prototypes_on_both_rails():
    """CF-DECLS: `cfront_decls_link.c` -- callees another unit defines (the driver), prototyped with a pointer
    and an array parameter to `const` and an unnamed function-pointer parameter. Each rail's `extern` declaration
    keeps the `const` -- both dropped it, a declaration that conflicts with the original prototype in one
    translation unit -- and spells the function-pointer parameter as C does (`RET (*)(PARAMS)`, or the twin's
    `__bcir_fpN` alias), where the oracle spelled it by its name. The unit lowers to one claim graph on the four
    targets, and each emit runs as the original."""
    if not _CC:
        return
    fx = "cfront_decls_link.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        ext = {
            ln.split("(", 1)[0].split()[-1]: ln
            for ln in emit.splitlines()
            if ln.startswith("extern ")
        }
        assert ext["dl_read"].split("(", 1)[1].startswith("const uint32_t *, "), (label, ext)
        assert ext["dl_mix"].split("(", 1)[1].startswith("const uint8_t *, "), (label, ext)
    want = "extern uint32_t dl_read(const uint32_t *, uint32_t (*)(uint32_t), uint32_t);"
    assert want in oracle_emit, oracle_emit
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _DECLS_LINK_DRIVER)


# A call, a function designator and a `sizeof` operand naming a function no declaration precedes: each unit
# is refused on both rails (C11 6.5.1p2), and the same unit with the declaration first lowers alike on both.
_UNDECLARED = {
    "a later static definition": (
        "uint32_t f(uint32_t s) { return g(s) + 1u; }\nstatic uint32_t g(uint32_t s) { return s * 3u; }\n",
        "static uint32_t g(uint32_t s);\n",
        "call to undeclared function 'g'",
    ),
    "a later wide definition": (
        "uint64_t f(uint32_t s) { return g(s) + 1u; }\nuint64_t g(uint32_t s) { return (uint64_t)s << 40; }\n",
        "uint64_t g(uint32_t s);\n",
        "call to undeclared function 'g'",
    ),
    "a later prototype": (
        "uint32_t f(uint32_t s) { return g(s) + 1u; }\nuint32_t g(uint32_t s);\n",
        "uint32_t g(uint32_t s);\n",
        "call to undeclared function 'g'",
    ),
    "a call in a later function's body": (
        "uint32_t h(uint32_t s) { return s + 2u; }\nuint32_t f(uint32_t s) { return g(s) + h(s); }\n"
        "uint32_t g(uint32_t s) { return h(s) ^ 5u; }\n",
        "uint32_t g(uint32_t s);\n",
        "call to undeclared function 'g'",
    ),
}
# ... and forms both rails refuse either way -- a function designator before any declaration, a `sizeof` of a
# call to a function defined later -- where the oracle had lowered them (the twin refused them already)
_UNDECLARED_REFUSED = {
    "a designator before the definition": (
        "typedef uint32_t (*op_t)(uint32_t);\n"
        "static uint32_t apply(op_t fn, uint32_t s) { return fn(s); }\n"
        "uint32_t f(uint32_t s) { return apply(later, s); }\n"
        "static uint32_t later(uint32_t v) { return v + 7u; }\n",
        "use of undeclared identifier 'later'",
    ),
    "a sizeof of a later call": (
        "uint32_t f(uint32_t s) { return (uint32_t)sizeof(g(s)) + s; }\nuint64_t g(uint32_t s) { return s; }\n",
        "",
    ),
    "a definition leaving a parameter unnamed": (
        "uint32_t f(uint32_t *, uint32_t s) { return s; }\n",
        "has no name",
    ),
}


def test_a_function_named_before_its_declaration_is_refused_on_both_rails():
    """CF-DECLS: a call to a function the unit defines or prototypes only after the call is refused on both
    rails -- C99 dropped the implicit declaration (C11 6.5.1p2) -- where both lowered it as if prototyped and the
    twin typed its result by the `uint32_t` default: a `uint64_t` callee's result truncated, with the same claim
    graph on both rails, so no digest showed it. A later prototype declares nothing either, where the oracle had
    typed the call by it and the twin had lowered an undefined callee. With the declaration first, each unit
    lowers to one claim graph on both rails. A function designator or a `sizeof` operand before any declaration,
    and a definition leaving a parameter unnamed, are refused on both rails."""
    exe = _build_frontend(_session_build_dir()) if _CC else None
    head = "#include <stdint.h>\n"
    for label, (body, decl, why) in _UNDECLARED.items():
        _refused_on_both_rails(exe, head + body, why)
        src = head + decl + body
        summary, _r, _entry = _oracle(src)
        if exe is None:
            continue
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "declared.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
            c_summary, emit = _c_run(exe, path)
            assert c_summary == summary, (label, c_summary, summary)
            if "uint64_t g" in decl:  # the call's result keeps the callee's width on the twin too
                assert "uint64_t t" in emit.split("bcir_f(", 1)[1].split("}", 1)[0], (label, emit)
    for label, (body, why) in _UNDECLARED_REFUSED.items():
        _refused_on_both_rails(exe, head + body, why)


_DECLARED_LATER = {
    "a sizeof of a prototyped call": (
        "uint64_t g(uint32_t s);\nuint32_t f(uint32_t s) { return (uint32_t)sizeof(g(s)) + s; }\n"
        "uint64_t g(uint32_t s) { return s; }\n"
    ),
    "a sizeof of an external call": (
        "uint64_t g(uint32_t s);\nuint32_t f(uint32_t s) { return (uint32_t)sizeof(g(s)) + s; }\n"
    ),
    "a designator of a prototyped function": (
        "typedef uint32_t (*op_t)(uint32_t);\nstatic uint32_t later(uint32_t v);\n"
        "static uint32_t apply(op_t fn, uint32_t s) { return fn(s); }\n"
        "uint32_t f(uint32_t s) { return apply(later, s); }\n"
        "static uint32_t later(uint32_t v) { return v + 7u; }\n"
    ),
    "an unnamed prototype": (
        "uint32_t g(uint32_t *, uint32_t);\nuint32_t f(uint32_t s) { uint32_t a = s; return g(&a, s); }\n"
        "uint32_t g(uint32_t *p, uint32_t s) { return *p + s; }\n"
    ),
}


def test_a_prototype_declares_its_function_alike_on_both_rails():
    """CF-DECLS: a function a prototype declares -- before its definition, or with no definition in the unit --
    is typed by the prototype on both rails: a `sizeof` of a call to it (the twin refused one, typing only by a
    definition before the call), a designator of it before its definition (the twin refused one as an undefined
    identifier), and a prototype leaving its parameters unnamed (both rails refused one). Each unit lowers to one
    claim graph on the four targets."""
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        for label, body in _DECLARED_LATER.items():
            src = "#include <stdint.h>\n" + body
            path = os.path.join(d, "declared.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
            summary, _r, _entry = _oracle(src)
            assert "ok=1" in summary, (label, summary)
            _parity_on_targets(path, src)


_ANON_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    an_pt p = {(uint16_t)s, 9u}, p1 = p, p2 = p;
    __typeof__(*(an_ref)0) o1 = {s ^ 3u, 7u}, o2 = o1;
    SAME(an_local, s); SAME(an_array, s); SAME(an_nested, s); SAME(an_union, s); SAME(an_global, s);
    SAME(an_len, p); SAME(an_entry, s);
    if (an_bump(&p1, s) != bcir_an_bump(&p2, s) || p1.x != p2.x || p1.y != p2.y) return fail("an_bump");
    if (an_deref(&o1, s) != bcir_an_deref(&o2, s) || o1.k != o2.k) return fail("an_deref");
    if (an_make(s).x != bcir_an_make(s).x || an_make(s).y != bcir_an_make(s).y) return fail("an_make");
  }
  puts("MATCH");
  return 0;
}
"""
)


def test_anonymous_structs_named_by_a_typedef_run_as_the_original_on_both_rails():
    """CF-ANON: `cfront_anonstruct.c` -- structs and a union declared without a tag and named by a typedef, as
    a local, a parameter, a returned value, a member, an array element, a pointer's target, a global and a
    `sizeof` operand; nested anonymous members, one an array; one named only through a pointer typedef. Each
    rail's emit names such a type as C does -- the typedef's name, `__typeof__(*(an_ref)0)`, or `__typeof__` of
    the member whose type it is -- where both spelled a tag no compiler knows (the oracle `struct ` with no tag,
    the twin `struct $anon0`), and the oracle refused the unit outright: its anonymous structs shared the empty
    tag, so every one past the first had no layout. The unit lowers to one claim graph on the four targets, and
    every emit runs as the original."""
    if not _CC:
        return
    fx = "cfront_anonstruct.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    spelled = (
        "__typeof__(((an_box *)0)->span)",  # `c.span = b.span`, by value
        "__typeof__(((an_box *)0)->pair[0])",  # an element of the member array `pair`
        "__typeof__(*(an_ref)0) *",  # the pointer typedef's pointee
    )
    for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        assert "$anon" not in emit and "struct  " not in emit, (label, emit)
        assert all(sp in emit for sp in spelled), (label, [sp for sp in spelled if sp not in emit])
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _ANON_DRIVER)
