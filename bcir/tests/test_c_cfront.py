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
import time
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
    "cfront_intconst.c",  # every integer constant base and suffix at its type's edges, and file-scope
    #   initializers folded in C's types (CF-INTCONST)
    "cfront_garray.c",  # multi-dimensional globals passed to row-pointer parameters, character tables sized by
    #   their strings, declarations listing several objects (CF-GARRAY)
    "cfront_gbrace.c",  # file-scope initializers with nested braces, elision and designators (CF-GBRACE), and
    #   thread-local globals and statics (CF-TLS)
    "cfront_voidcallback.c",  # a call through a pointer to a void function -- a local, a parameter, a member
    #   by `.`/`->`/a pointer chain, an ops table, in a `?:` arm, `return cb();` -- has no result (CF-VOIDCB)
    "cfront_fnselect.c",  # the arms of `?:` that are function designators or function-pointer objects: the
    #   select is a pointer to their function type, a null pointer arm that pointer (CF-FNSEL)
    "cfront_nullconst.c",  # the constant 0 compared with a pointer, passed through a function pointer, given to
    #   `free` and `realloc`: a null pointer; `p -= 1` of a pointer to a struct on the twin (CF-NULLCALL)
    "cfront_enumfold.c",  # enumerators, case labels and array dimensions folded in C's types and the target's
    #   data model; a dimension that is an integer constant expression a fixed array (CF-ENUMFOLD)
    "cfront_enumscope.c",  # a local, a parameter and a loop's declaration hide an enumerator of their name to
    #   the end of their block; file scope after a function reads the enumerator again (CF-ENUMSCOPE)
    "cfront_constexpr2.c",  # case labels in the promoted type, prefixed character constants, block enumerations,
    #   constant widths and dimensions, sizeof and float casts in enumerators, zero-length members (CF-CONSTEXPR2)
    "cfront_typedefscope.c",  # a local, a parameter, a loop's declaration and a block's enumerator hide a typedef
    #   name to the end of their block: no declaration, cast or sizeof type-name there (CF-TYPEDEFSCOPE)
    "cfront_fptab.c",  # tables of function pointers -- typedef'd and inline, local, file-scope, 2-D, members,
    #   pointers to them, parameters of them -- and calls through `(*fp)`, `(**fp)` and `(*f)` (CF-FPTAB)
    "cfront_fpret.c",  # function pointers calls return; pointers, structs and function pointers returned through a
    #   function pointer; pointers to variadic functions (CF-FPRET)
    "cfront_idxarrow.c",  # elements that are pointers: `pp[i][j].f` as a value, `pp + 1` and `arr + 1` of `T **`,
    #   `&arr[i]`, a file-scope array of pointers decayed (CF-IDXARROW)
    "cfront_rtfp.c",  # casts to function-pointer and `_Atomic` types and the emit's stores of them at a byte offset; a
    #   typedef'd table of function pointers, a compound literal of one, `sizeof` of one, `( E ) = v;` (CF-RTFP)
    "cfront_unaryops.c",  # `+a` promoted, `-z` of a complex, `__real__` a part, bit-field and `&x` operand types, an
    #   array compared with 0, `void *` of malloc, `++(x)`, void values where C reads none (CF-UNARY, CF-STRUCTCOND)
    "cfront_filescope.c",  # plain `char` and string literal elements, `gm[i][j].x`, 2-D pointer tables, `*(p + i -
    #   j)`, `char s[] = ("abc");` and `const` globals read into the emit's temps (CF-CHARELEM, CF-AOS2D, CF-LINKEMIT)
    "cfront_enumobj.c",  # objects of an enumerated type -- locals, globals, parameters, returns, members, a bit-field,
    #   typedef names -- typed as the enumeration's compatible type: `unsigned int` with no negative enumerator on System
    #   V, `int` otherwise and on MSVC (CF-ENUMOBJ)
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


# The corpus harnesses' seeded generator. They set the original source and its emits beside their own file-scope names,
# so those are named as no fixture spells one: `cfront_typedefscope.c`'s `typedef struct { ... } S;` met a generator
# state named `S` and the harness did not build (CF-TYPEDEFSCOPE).
_HARNESS_RNG = (
    "static uint64_t bcir_h_seed=0x9E3779B97F4A7C15u;\n"
    "static uint32_t bcir_h_rng(void){bcir_h_seed=bcir_h_seed*6364136223846793005u+1442695040888963407u;"
    "return (uint32_t)(bcir_h_seed>>32);}"
)


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
                f"    for(unsigned k=0;k<sizeof buf{i}/4;k++) ((uint32_t*)buf{i})[k]=bcir_h_rng();"
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
                f"    a{i}.{fn}=(bcir_h_rng()&{(1 << bw) - 1}u);\n"
                if bw
                else f"    a{i}.{fn}=({_cname(ft)})bcir_h_rng();\n"
                for fn, ft, _bo, _bf, bw in ct.fields
            )
            setup.append(inits.rstrip("\n"))
            args.append(f"a{i}")
        elif ct.is_complex:  # a _Complex param: seed BOTH axes (finite, in range)
            decls.append(f"  {_cname(ct)} s{i};")
            el = "float" if ct.size == 8 else ("long double" if ct.size > 16 else "double")
            setup.append(f"    s{i}=({el})(bcir_h_rng()%1000) + ({el})(bcir_h_rng()%1000)*I;")
            args.append(f"s{i}")
        else:
            decls.append(f"  {_cname(ct)} s{i};")
            # a float param gets an in-range value (so a float->int cast stays defined, not UB);
            # an integer scalar stays below 2**31 so it is non-negative as `int` -- the value model
            # is unsigned, so an int->float cast must agree in sign (wrapping arithmetic is unaffected).
            mod = 1000 if ct.is_float else (200 if has_ptr else 2000000000)
            setup.append(f"    s{i}=({_cname(ct)})(bcir_h_rng()%{mod});")
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
    # (`source` is preprocessed, its `#include`s gone: what a fixture includes, the harness does -- <stdarg.h> for a
    # variadic function's `va_list`, CF-FPRET)
    harness = f"""#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <math.h>
#include <complex.h>
{_BOUNDS_GUARD}
{source}

{c_emitted}
{chr(10).join(prelude)}
{_HARNESS_RNG}
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
            setup.append(f"    ca{i}=cb{i}=({plain})(bcir_h_rng()%16);")
            args_a.append(f"&ca{i}")
            args_b.append(f"&cb{i}")
            cell_cmp.append(f"ca{i}!=cb{i}")
        else:
            decls.append(f"  {_cname(ct)} s{i};")
            setup.append(f"    s{i}=({_cname(ct)})(bcir_h_rng()%16);")
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
{_HARNESS_RNG}
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
    between the rails -- a table of function pointers spelled inline (`RET (*t[N])(P) = ...;`, the oracle's) as one
    spelled by its alias (`op_t t[N] = ...;`, the twin's; CF-FPTAB)."""
    sizes = []
    decls = re.findall(r"[A-Za-z_]\w*(?:\s+\w+)?\s+\w+((?:\[\d+\])+)\s*[=;]", emit)
    decls += re.findall(r"\(\*+\s*\w+((?:\[\d+\])+)\)\s*\([^;={]*\)\s*[=;]", emit)
    for dims in decls:
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
            f"    h=h*1099511628211u + (uint64_t)xcc_fp_i64(creall(rr));\n"
            f"    h=h*1099511628211u + (uint64_t)xcc_fp_i64(cimagl(rr));"
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
/* a complex part's fingerprint: truncated toward zero, saturated past int64_t and 0 for a NaN -- a conversion out
   of range is undefined (C11 6.3.1.4p1), and the part of an overflowing complex result is often one (CF-UBGATE) */
static int64_t xcc_fp_i64(long double v){{
  return v != v ? 0 : v >= 9223372036854775807.0L ? INT64_MAX : v <= -9223372036854775808.0L ? INT64_MIN : (int64_t)v;}}
{_HARNESS_RNG}
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
        # `(*(_Atomic T *)A)`, and the indexed `(*((_Atomic T *)B + i))` (TC23)
        lvalues = re.findall(r"\(\*\(\(?(?:volatile )?_Atomic ", body)
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
            "&QUANTA)[",  # the `const` table, read through the unqualified lvalue the emit names it by (CF-LINKEMIT)
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
        # a `#line` of no decimal digit sequence is refused alike (CF-PPSPLITS; it was ignored on both)
        bare = "p __LINE__\n#line\nq __LINE__\n"
        assert pp(bare) == "ERR #line number is not a decimal digit sequence", pp(bare)
        assert _oracle_refusal(bare) == "#line number is not a decimal digit sequence"
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

        # The preprocessed text is held whole, in a block grown as it needs (CF-PPLIMITS): an 80 KB `-E` output,
        # refused once past the 64 KiB the driver held, is written whole.
        r = run("pp_large.c", "x\n" * 40000, "-E")
        assert r.returncode == 0 and r.stdout.split() == ["x"] * 40000, (r.returncode, r.stderr)
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
# one reason; and a cast to or compound literal of an `_Atomic` type, each for one reason of its own.
_ATOMIC_AGGREGATE = "an `_Atomic` struct or union is not supported"
_ATOMIC_BITFIELD = "a bit-field of `_Atomic` type is not supported"
_ATOMIC_LITERAL = "a compound literal of `_Atomic` type is not supported"
_CAST_ATOMIC = "a cast to an `_Atomic` type is not supported"
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
    # CF-RTFP: a compound literal of an `_Atomic` type, an `_Atomic` object -- under `&`, of an array, under `sizeof`
    (
        "uint32_t f(uint32_t x) { uint32_t *p = &(_Atomic uint32_t){x}; return *p; }\n",
        _ATOMIC_LITERAL,
    ),
    (
        "uint32_t f(uint32_t x) { _Atomic uint32_t *q = (_Atomic uint32_t[2]){1u, x}; return q[1]; }\n",
        _ATOMIC_LITERAL,
    ),
    (
        "uint32_t f(uint32_t x) { return (uint32_t)sizeof((_Atomic uint32_t){x}); }\n",
        _ATOMIC_LITERAL,
    ),
    (
        "uint32_t f(uint32_t x) { return (uint32_t)sizeof (_Atomic uint32_t){x}; }\n",
        _ATOMIC_LITERAL,
    ),
    (
        "typedef _Atomic uint32_t au32_t;\nuint32_t f(uint32_t x) { uint32_t *p = &(au32_t){x}; return *p; }\n",
        _ATOMIC_LITERAL,
    ),
    # CF-RTFP: a cast to an `_Atomic` type -- spelled, `_Atomic(T)`, qualified, a typedef of one, a `?:` arm, under
    # `sizeof` and `typeof` -- which Clang types `_Atomic` and rejects as an operand (where C17 6.5.4p5 and GCC drop
    # the qualifier); a cast to a pointer to an `_Atomic` object lowers
    ("uint32_t f(uint32_t x) { uint32_t y = (_Atomic uint32_t)x; return y + 1u; }\n", _CAST_ATOMIC),
    (
        "uint32_t f(uint32_t x) { return (_Atomic(uint32_t))x + (const _Atomic uint32_t)x; }\n",
        _CAST_ATOMIC,
    ),
    (
        "typedef _Atomic uint32_t au32_t;\nuint32_t f(uint32_t x) { return (au32_t)x; }\n",
        _CAST_ATOMIC,
    ),
    ("uint32_t f(uint32_t x) { return x ? (_Atomic uint32_t)x : 2u; }\n", _CAST_ATOMIC),
    (
        "uint32_t f(uint32_t x) { double d = (_Atomic double)x; return (uint32_t)d; }\n",
        _CAST_ATOMIC,
    ),
    ("uint32_t f(uint32_t x) { return (uint32_t)sizeof((_Atomic uint32_t)x); }\n", _CAST_ATOMIC),
    (
        "typedef _Atomic uint32_t au32_t;\nuint32_t f(uint32_t x) { return (uint32_t)sizeof((au32_t)x) + x; }\n",
        _CAST_ATOMIC,
    ),
    ("uint32_t f(uint32_t x) { __typeof__((_Atomic uint32_t)x) y = x; return y; }\n", _CAST_ATOMIC),
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


def test_an_atomic_aggregate_cast_or_literal_is_refused_on_both_rails():
    """CF-CALIGN: the ABI's atomic promotion would lay out an `_Atomic` struct at a rounded-up size (a
    3-byte one occupies 4), which every declaration, copy and extent would then have to carry, and each
    access to it would have to be one atomic operation on the whole object. Neither rail models that, so
    both refuse it, wherever it is spelled. The oracle had laid it out as the plain struct, and the twin
    had placed it as a member but refused `sizeof(_Atomic struct P3)`. A compound literal of an `_Atomic`
    type, an `_Atomic` object too, is refused on both rails for one reason, and so is a cast to an `_Atomic` type:
    C17 6.5.4p5 gives the cast the unqualified type, as GCC does, but Clang types it `_Atomic` and rejects it as an
    operand -- initialized from, assigned, returned, cast again or added to another -- so no emit of it could be
    held to the original (CF-RTFP; refused as parse errors until both read `_Atomic` as a cast's type name, which a
    cast to a pointer to an `_Atomic` object, the emit's own spelling, needs); `sizeof`, `_Alignof` and `typeof` of
    an `_Atomic` type fold identically on both."""
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
    """A C-twin claim holds a call's sixteen arguments and its callee value (CF-CALLARGS; it held six and dropped
    the rest, so both rails refused a unit with a call of seven as `truncated=1`, every footprint `*`). Sixteen
    arguments is the boundary now: a call of seven and one of sixteen are analyzed on both rails, their effects
    and escape reports byte-identical, and a call of seventeen is refused at lowering by both in the same words,
    so no unit reaches either rail's analysis truncated."""
    from bcir.frontends.cfront.escape import effects_report, escape_report  # noqa: PLC0415
    from bcir.frontends.cfront.lower import MAX_CALL_ARGS  # noqa: PLC0415

    def unit(n: int) -> str:
        params = ", ".join(f"unsigned a{i}" for i in range(n - 1)) + ", unsigned *p"
        total = " + ".join(f"a{i}" for i in range(n - 1))
        args = ", ".join(["x"] * (n - 1)) + ", t"
        return (
            f"static unsigned s{n}({params}) {{ p[0] = {total}; return p[1]; }}\n"
            f"unsigned caller(unsigned x) {{ unsigned t[4]; t[1] = x; return s{n}({args}); }}\n"
        )

    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        cc = _build_bcir_cc(d)
        for label, n in (("seven", 7), ("sixteen", MAX_CALL_ARGS)):
            src = unit(n)
            path = os.path.join(d, f"{label}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
            r = compile_unit(src, check_clang=False)
            assert not r.escape.truncated and r.escape.objects["caller"] == {"t": "lent"}, label
            for flag, report in (
                ("--emit-effects", effects_report),
                ("--emit-escape", escape_report),
            ):
                out = subprocess.run([cc, flag, path], capture_output=True, text=True).stdout
                assert out == report(r.lowered, r.escape), (label, flag, out)
        why = f"a call of more than {MAX_CALL_ARGS} arguments is not supported"
        assert _oracle_refusal(unit(MAX_CALL_ARGS + 1)) == why
        path = os.path.join(d, "seventeen.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(unit(MAX_CALL_ARGS + 1))
        run = subprocess.run([cc, "--emit-escape", path], capture_output=True, text=True)
        assert run.returncode != 0 and (run.stdout + run.stderr).strip().endswith(
            f": parse error: {why}"
        ), (run.returncode, run.stdout, run.stderr)


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
    int a=(int)nx(), b=(int)nx(); long lb=(long)(nx()>>2);  /* 30 bits: umix's a*lb is a long (CF-UBGATE) */
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
    from bcir.frontends.cfront.cpp import CPPError

    for path in sorted(glob.glob(os.path.join(_C, "cfront_*.c"))):
        fx = os.path.basename(path)
        try:
            r = compile_unit(
                open(path, encoding="utf-8").read(), check_clang=False, includes=_includes_for(fx)
            )
        except (CParseError, CPPError):
            if fx.startswith(("cfront_sec_", "cfront_pp_")):
                continue  # a deliberately-malformed adversarial fixture (cfront_sec_deepnest/lextail, and
                # cfront_sec_cppmacro's overlong macro parameter both preprocessors refuse) or a
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
    from bcir.frontends.cfront.cpp import CPPError

    seen_masked = 0
    for path in sorted(glob.glob(os.path.join(_C, "cfront_*.c"))):
        fx = os.path.basename(path)
        try:
            r = compile_unit(
                open(path, encoding="utf-8").read(), check_clang=False, includes=_includes_for(fx)
            )
        except (CParseError, CPPError):
            if fx.startswith(("cfront_sec_", "cfront_pp_")):
                continue  # a deliberately-malformed adversarial fixture (cfront_sec_deepnest/lextail, and
                # cfront_sec_cppmacro's overlong macro parameter both preprocessors refuse) or a
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
        "int g(int n){ int m=(n&7)+1; int a[m]; int s=0; n%=100000;"  # its sums are ints (CF-UBGATE)
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
        "int f(int a){ a %= 1000; int x=a--; return x*100+a; }",  # in range: x*100 is an int (CF-UBGATE)
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
        "sizeof of a function designator",  # the twin's reason on both rails (CF-FPTAB)
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

    from bcir.frontends.cfront.lower import ARROW_NOT, DOT_NOT

    bodies = (  # each refused for its operator's reason on both rails (CF-FPTAB)
        ("uint32_t f(uint32_t x) { return x.n; }", DOT_NOT),
        ("uint32_t g;\nuint32_t f(void) { return g.n; }", DOT_NOT),
        ("uint32_t *g;\nuint32_t f(void) { return g->n; }", ARROW_NOT),
        # `.` of a pointer to a struct: the twin read it as `->` (CF-FPTAB)
        ("struct S { uint32_t n; };\nuint32_t f(struct S *p) { return p.n; }", DOT_NOT),
        ("struct S { uint32_t n; };\nstruct S *g;\nuint32_t f(void) { return g.n; }", DOT_NOT),
    )
    for body, why in bodies:
        try:
            compile_unit("#include <stdint.h>\n" + body + "\n", check_clang=False)
        except (CLowerError, CParseError) as e:
            assert str(e) == why, (body, str(e))
        else:
            raise AssertionError(f"the oracle lowered {body!r}")
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, (body, why) in enumerate(bodies):
            path = os.path.join(d, f"m{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("#include <stdint.h>\n" + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode == 1 and run.stdout.strip() == f"PARSE-ERR {why}", (
                body,
                run.returncode,
                run.stdout[:200],
            )


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
    # an `_Atomic` one too, one atomic access: the emit's own spelling of an `_Atomic` member (CF-RTFP)
    "atomic": ("return *(volatile _Atomic uint32_t *)((char *)d + 4);", 4),
}
# the near misses, which fold on neither rail: an offset that is no literal, a byte pointer that is not a
# plain `char` one, an access that is neither volatile nor `_Atomic`
_BYTEOFF_NEAR = {
    "var_offset": "return *(volatile uint32_t *)((char *)d + (v & 4u));",
    "uchar_cast": "return *(volatile uint32_t *)((volatile unsigned char *)d + 4);",
    "nonvolatile": "return *(uint32_t *)((char *)d + 4);",
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
    `char` one, an access neither volatile nor `_Atomic` -- folds on neither rail and digests alike too. CF-RTFP:
    a volatile `_Atomic` access folds as one atomic access, as the emit spells an `_Atomic` member."""
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


_RANK_NAMES = (
    "#include <stdint.h>\n"
    "uint32_t x_2(uint32_t v) { return v + 1u; }\n"
    "uint32_t f(uint32_t x, uint32_t y) {\n"
    "  uint32_t z = 0u;\n"
    "  { uint32_t x = y * 3u; z = x_2(x); }\n"
    "  { uint32_t x = y ^ 5u; z += x; }\n"
    "  return x + z;\n"
    "}\n"
)


def test_a_local_of_a_parameters_name_ranks_after_it_and_past_the_global_it_prefixes_on_both_rails():
    """CF-NAMECACHE. A named object emits under its rank among the objects of its source name -- the parameters
    first, then the locals in order -- skipping a suffixed candidate the emit spells otherwise. The parameter `x`
    keeps its name; the first block's `x` ranks second, and `x_2` being a function it calls (spelled `bcir_x_2`), takes
    `x_3`; the second block's takes `x_4`. The oracle and the twin's walks read the names so; the twin's table (`names_rank`) must give the
    walks' own: each object declared once, no local under the global's name, and both emits run as the original."""
    summary, r, entry = _oracle(_RANK_NAMES)
    oracle_emit = "\n".join(r.emitted[n] for n in r.lowered.functions)

    def family(emit):
        return set(re.findall(r"\bx(?:_\d+)?\b", re.sub(r"/\*.*?\*/", "", emit, flags=re.S)))

    assert family(oracle_emit) == {"x", "x_3", "x_4"}, oracle_emit  # the callee: `bcir_x_2`
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rank.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_RANK_NAMES)
        c_summary, c_emit = _c_run(exe, path)
    assert c_summary == summary, f"parity\n C: {c_summary}\nPY: {summary}"
    assert family(c_emit) == {"x", "x_3", "x_4"}, c_emit
    for label, e in (("twin", c_emit), ("oracle", oracle_emit)):
        verdict = _equiv(_RANK_NAMES, e, entry)
        assert verdict == "MATCH", f"the {label}'s emit: {verdict}\n{e}"


_GBLOCK_UNITS = {
    "block": "#include <stdint.h>\nuint32_t g = 7u;\nuint32_t f(uint32_t y) {\n  { g = y; }\n  return g;\n}\n",
    "if": (
        "#include <stdint.h>\nuint32_t g = 7u;\nuint32_t f(uint32_t y) {\n"
        "  if (y & 1u) { g = y; } else { g = y + 1u; }\n  return g;\n}\n"
    ),
    "loop": (
        "#include <stdint.h>\nuint32_t g = 7u;\nuint32_t f(uint32_t y) {\n"
        "  { g = 0u; }\n  for (uint32_t i = 0u; i < (y & 15u); i++) { g += i; }\n  return g;\n}\n"
    ),
    "shadowed": (
        "#include <stdint.h>\nuint32_t g = 7u;\nuint32_t f(uint32_t y) {\n"
        "  g = y;\n  { uint32_t g = 1u; y += g; }\n  g += y;\n  return g;\n}\n"
    ),
    "function scope first": (
        "#include <stdint.h>\nuint32_t g = 7u;\nuint32_t f(uint32_t y) {\n  g = 1u;\n  { g = y; }\n  return g;\n}\n"
    ),
}


def test_a_global_first_named_inside_a_block_is_one_resource_on_both_rails():
    """CF-GBLOCK. The twin bound a global's resource with the block that first named it and dropped the binding
    with the block, so the next reference -- `{ g = y; } return g;`, an `if` body, a loop body -- made a second
    resource for the same object: the canon read the written value back as an input and the digest left the
    oracle's (the emits still ran as the original, both naming the object `g`). A global is one object, so it is
    one resource per function on both rails (the oracle's `gres`; the twin's `global_res_of_name`). Each unit's
    summary, digest included, is the oracle's on the twin; a block-scope `g` still hides the global for its block
    (`shadowed`); the twin's emits run as the originals. Every unit assigns `g` on every path before reading it,
    so the equivalence harness, whose original and emit share one `g`, reads no call's leftovers."""
    oracle = {name: _oracle(src) for name, src in _GBLOCK_UNITS.items()}
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for name, src in _GBLOCK_UNITS.items():
            summary, _r, entry = oracle[name]
            path = os.path.join(d, name.replace(" ", "_") + ".c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
            c_summary, c_emit = _c_run(exe, path)
            assert c_summary == summary, f"{name}: parity\n C: {c_summary}\nPY: {summary}"
            verdict = _equiv(src, c_emit, entry)
            assert verdict == "MATCH", f"{name}: the twin's emit: {verdict}\n{c_emit}"


def _call_of(n: int, *, indirect: bool = False) -> str:
    """`f` calls a function of `n` parameters with `x + i` for its i-th argument -- directly, or through a function
    pointer it holds -- and the callee folds its arguments with distinct weights, so a dropped or reordered one
    changes the result."""
    params = ", ".join(f"uint32_t a{i}" for i in range(n))
    body = " + ".join(f"{i + 1}u * a{i}" for i in range(n))
    args = ", ".join(f"x + {i}u" for i in range(n))
    callee = f"#include <stdint.h>\nuint32_t h({params}) {{ return {body}; }}\n"
    if indirect:
        sig = ", ".join(["uint32_t"] * n)
        return f"{callee}uint32_t f(uint32_t x) {{ uint32_t (*p)({sig}) = h; return p({args}); }}\n"
    return f"{callee}uint32_t f(uint32_t x) {{ return h({args}); }}\n"


def test_a_call_carries_sixteen_arguments_and_a_seventeenth_is_refused_on_both_rails():
    """CF-CALLARGS. The twin's claim held six reads, so `p_call` kept a call's first six arguments and dropped the
    rest: a call of seven lowered `ok=1` with a digest the oracle's no longer, and its emit passed six arguments to
    a function of seven, which C refuses. Both rails now carry sixteen arguments (`BCIR_CALL_MAX_ARGS`,
    `lower.MAX_CALL_ARGS`; a claim's reads are one more, for a callee value or an object base) and refuse a
    seventeenth alike: calls of seven and of sixteen, direct and through a function pointer, are digest-equal on
    both rails and the twin's emits run as the originals."""
    units = {
        "seven": _call_of(7),
        "sixteen": _call_of(16),
        "sixteen through a pointer": _call_of(16, indirect=True),
    }
    oracle = {name: _oracle(src) for name, src in units.items()}
    exe = _build_frontend(_session_build_dir()) if _CC else None
    for src in (_call_of(17), _call_of(17, indirect=True)):
        _refused_on_both_rails(exe, src, "a call of more than 16 arguments is not supported")
    if exe is None:
        return
    with tempfile.TemporaryDirectory() as d:
        for name, src in units.items():
            summary, _r, entry = oracle[name]
            path = os.path.join(d, name.replace(" ", "_") + ".c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
            c_summary, c_emit = _c_run(exe, path)
            assert c_summary == summary, f"{name}: parity\n C: {c_summary}\nPY: {summary}"
            verdict = _equiv(src, c_emit, entry)
            assert verdict == "MATCH", f"{name}: the twin's emit: {verdict}\n{c_emit}"


def _lowers_alike(exe, src: str) -> str:
    """Both rails lower `src`: the oracle's summary, which the twin (when built) prints verbatim."""
    summary, _r, _entry = _oracle(src)
    if exe is not None:
        assert _twin_line(exe, src) == (0, summary), src
    return summary


def _refused_alike(exe, src: str, why: str, *, phase: str = "CPP") -> None:
    """Neither rail lowers `src`, each naming `why` exactly: the oracle's refusal is `why`, the twin's line is
    `<phase>-ERR why` (`CPP` for the preprocessor, `PARSE` for the lexer and parser)."""
    assert _oracle_refusal(src) == why, (src, _oracle_refusal(src))
    if exe is not None:
        assert _twin_line(exe, src) == (1, f"{phase}-ERR {why}"), (src, _twin_line(exe, src))


def test_form_feed_and_vertical_tab_are_white_space_on_both_rails():
    """CF-PPSPLITS. C's white space is a space, a tab, a new-line, a vertical tab and a form feed (6.4p3). The twin's
    preprocessor and lexer read a space and a tab alone, so `\f` in code was an expression's end (`expected
    expression`) and `\f#define` no directive, where the oracle lowered both. One predicate on the twin
    (`pp_space`), the oracle's lexer widened to match its preprocessor: the unit lowers digest-equal."""
    exe = _build_frontend(_session_build_dir()) if _CC else None
    src = (
        "#include <stdint.h>\nuint32_t f(uint32_t x) {\f return x\v + 1u; }\n"
        "\f#define K 2u\n\v#\tdefine L 3u\nuint32_t g(uint32_t x) {\treturn\fx + K\v+ L; }\n"
    )
    summary = _lowers_alike(exe, src)
    assert "funcs=2" in summary, summary
    # each preprocessor spells its output with single spaces, so the lexers are witnessed at their own entries: the
    # oracle's here, the twin's through `bcir_cfront_compile` (`test_c_the_frontend_reads_c_white_space_at_its_own_entry`)
    from bcir.frontends.cfront.clex import tokenize

    assert [(t.kind, t.text) for t in tokenize("x\f+\v1u\f")] == [
        (t.kind, t.text) for t in tokenize("x + 1u")
    ]
    # the white space between a directive's `#` and its name, and after the name: the oracle read a space and a tab
    # there (`_DIRECTIVE_SPACE`), so `#\finclude <stdint.h>` was a null directive with its header unread and
    # `#include\f<stdint.h>` named no header
    for directive in ("#\finclude <stdint.h>", "#include\f<stdint.h>", "#\vinclude\v<stdint.h>"):
        summary = _lowers_alike(exe, directive + "\nuint32_t f(uint32_t x) { return x; }\n")
        assert "funcs=1" in summary, summary
    # a form feed ends no line: `#define K 1u` after one on a code line is text, not a directive, on both rails
    # (each lexer skips a `#` line), so `K` is undeclared alike
    _refused_alike(
        exe,
        "uint32_t f(uint32_t x) { return x; }\f#define K 1u\nuint32_t g(uint32_t x) { return x + K; }\n",
        "use of undeclared identifier 'K'",
        phase="PARSE",
    )


def test_an_error_directive_and_an_unknown_one_are_refused_alike_on_both_rails():
    """CF-PPSPLITS. `#error text` is the compile's failure (C11 6.10.5), its text macro-expanded, and a non-directive
    (`#foo`) is undefined behaviour the oracle refused (6.10p9); the twin ignored both, with `#warning` and
    `#pragma`, and lowered on. Both refuse alike now, spelling the oracle's reason; `#warning`, `#pragma` and the
    null directive `#` lower on both, and a group not taken keeps its `#error` and `#foo` unread (6.10.1p6)."""
    exe = _build_frontend(_session_build_dir()) if _CC else None
    unit = "uint32_t f(uint32_t x) { return x; }\n"
    _refused_alike(exe, "#define WHY stop here\n#error WHY now\n" + unit, "#error stop here now")
    _refused_alike(exe, "#foo bar\n" + unit, "unknown directive #foo")
    _refused_alike(exe, "#include_next <x.h>\n" + unit, "unknown directive #include_next")
    summary = _lowers_alike(
        exe, "#pragma once\n#warning not an error\n#\n#if 0\n#error unread\n#foo\n#endif\n" + unit
    )
    assert "funcs=1" in summary, summary


def test_the_preprocessors_hold_one_set_of_limits_and_one_arity_on_both_rails():
    """CF-PPSPLITS. The twin's preprocessor holds a logical line to 8 190 bytes, a macro's replacement to 1 023 and an
    invocation to 16 arguments; the oracle's held none of these, and neither rail held a function-like macro to as
    many arguments as parameters (C11 6.10.3p4): the twin took `M(x, 1u, 2u)` for `M(a, b)`, the oracle that and
    seventeen arguments for sixteen parameters. Both now hold the same bounds in the same words -- the last byte
    within each lowers digest-equal, the first past it is refused alike -- and the same arity: more arguments than
    parameters `too many macro arguments`, fewer `too few macro arguments`, a variadic macro's `...` free to be empty,
    and `M()` of a macro of no parameters passing none."""
    exe = _build_frontend(_session_build_dir()) if _CC else None
    unit = "#include <stdint.h>\nuint32_t f(uint32_t x) {{ return {v}; }}\n"
    # a macro replacement: 1 023 bytes lowers, 1 024 is refused
    body = "x" + "+1u" * 340 + "+2u"  # 1 + 1020 + 3 = 1 024 bytes
    assert len(body.encode()) == 1024
    _lowers_alike(exe, "#define B " + body[:-1] + "\n" + unit.format(v="B"))
    _refused_alike(
        exe, "#define B " + body + "\n" + unit.format(v="B"), "macro replacement is too large"
    )
    # a logical line: 8 190 bytes lowers (spliced across a continuation), 8 191 is refused
    head = "uint32_t g(uint32_t x) { return x"
    line = head + "+1u" * ((8190 - len(head) - 5) // 3)
    line += " " * (8190 - len(line) - 5) + "+1u;}"
    assert len(line.encode()) == 8190
    _lowers_alike(
        exe,
        "#include <stdint.h>\n" + line[:4000] + "\\\n" + line[4000:] + "\n" + unit.format(v="g(x)"),
    )
    _refused_alike(
        exe,
        "#include <stdint.h>\n" + line + " " + "\n" + unit.format(v="g(x)"),
        "preprocessor line too long",
    )
    # the bound is read after the comments are gone, as the twin's reader reads it: a comment of 9 000 bytes on
    # one line lowers on both (CI's thorough tier caught the oracle refusing a 70 KB comment line)
    _lowers_alike(
        exe, "#include <stdint.h>\n/*" + "c" * 9000 + "*/ " + head + "; }\n" + unit.format(v="g(x)")
    )
    # an invocation: 16 arguments lower, 17 are refused -- and as many as the parameters, or refused
    va = "#define V(...) (0u __VA_OPT__(+) __VA_ARGS__)\n"
    _lowers_alike(exe, va + unit.format(v="V(" + ", ".join(["x"] * 16) + ")"))
    _refused_alike(
        exe, va + unit.format(v="V(" + ", ".join(["x"] * 17) + ")"), "too many macro arguments"
    )
    two = "#define M(a, b) ((a) + (b))\n"
    _lowers_alike(exe, two + unit.format(v="M(x, 1u)"))
    _refused_alike(exe, two + unit.format(v="M(x, 1u, 2u)"), "too many macro arguments")
    _refused_alike(exe, two + unit.format(v="M(x)"), "too few macro arguments")
    _lowers_alike(
        exe, "#define N() 3u\n#define O(a) (3u a)\n" + unit.format(v="x + N() + O() + O(+1u)")
    )
    _refused_alike(exe, "#define N() 3u\n" + unit.format(v="x + N(1u)"), "too many macro arguments")
    va2 = "#define W(a, b, ...) ((a) + (b) __VA_OPT__(+) __VA_ARGS__)\n"
    _lowers_alike(exe, va2 + unit.format(v="W(x, 1u) + W(x, 1u, 2u, 3u)"))
    _refused_alike(exe, va2 + unit.format(v="W(x)"), "too few macro arguments")
    _lowers_alike(
        exe,
        "#define E(a, ...) (0u a __VA_OPT__(+) __VA_ARGS__)\n" + unit.format(v="E() + E(+x, 1u)"),
    )


def test_a_line_directive_takes_a_decimal_digit_sequence_in_range_on_both_rails():
    """CF-PPSPLITS. `#line` takes a decimal digit sequence (C11 6.10.4p3) in 1..2147483647, then an optional file
    name. `#line 12abc` read 12 on the oracle and was `out of range` on the twin; `#line abc` was ignored on both.
    Both refuse a number that is no decimal digit sequence, and one out of range, in the same words; the directive
    that conforms sets `__LINE__` and `__FILE__` alike."""
    exe = _build_frontend(_session_build_dir()) if _CC else None
    unit = "#include <stdint.h>\nuint32_t f(uint32_t x) {{ return {v}; }}\n"
    for bad in ("12abc", "abc", "0x10", "", '"a.c"'):
        _refused_alike(
            exe,
            f"#line {bad}\n" + unit.format(v="x"),
            "#line number is not a decimal digit sequence",
        )
    for bad in ("0", "2147483648", "999999999999999999999"):
        _refused_alike(exe, f"#line {bad}\n" + unit.format(v="x"), "#line number is out of range")
    summary = _lowers_alike(exe, '#define N 70\n#line N "a.c"\n' + unit.format(v="x + __LINE__"))
    assert "const=1" in summary, summary


def test_a_stray_character_outside_a_literal_is_refused_alike_on_both_rails():
    """CF-PPSPLITS. No C token starts with `@`, `$`, a backquote or a stray backslash: the oracle's lexer refused
    them as `unexpected character '@'`, the twin's made each a one-character punctuator for the parser to stumble
    on (`parse error: ;`). The twin refuses them where the oracle does, in the oracle's words -- a character past
    ASCII stays `NONASCII` on both (CF-PPLIMITS)."""
    exe = _build_frontend(_session_build_dir()) if _CC else None
    unit = "#include <stdint.h>\nuint32_t f(uint32_t x) {{ return {v}; }}\n"
    for stray in ("@", "$", "`", "\\"):
        _refused_alike(
            exe, unit.format(v=f"x {stray} 1u"), f"unexpected character {stray!r}", phase="PARSE"
        )
    _lowers_alike(exe, unit.format(v='x + sizeof("@$`")'))


def test_a_line_ends_at_a_new_line_alone_on_both_rails():
    """CF-PPSPLITS. A logical line ends at a new-line, and a CRLF end of line is one (translation phase 1). The twin
    read a CRLF file's `\\r` into the line, so `#if K == 5u\\r` was `malformed` there and lowered on the oracle, and a
    `\\\\\\r\\n` splice was none; the oracle ended a line where `str.splitlines` does -- at a form feed, a `\\x1c`, a NEL,
    a U+2028 -- so `return x\\x1c + 1u;` lowered there with the character gone and was refused here, and a NEL in code
    vanished before the reader that refuses a character past ASCII saw it. One rule on each rail now: a CRLF unit
    lowers to its LF twin's digest on both, the twin's longest line included; a `\\x1c` or `\\x1e` in code, at a
    line's end too, is `unexpected character` alike; a NEL or a U+2028 in code is `NONASCII` alike; and a lone
    carriage return is white space alike, in a directive's operand too."""
    from bcir.frontends.cfront.clex import NONASCII

    exe = _build_frontend(_session_build_dir()) if _CC else None
    lf = "#define K 2u + \\\n 3u\n#if K == 5u\nuint32_t f(uint32_t x) { return x + K; }\n#endif\n"
    assert _lowers_alike(exe, lf.replace("\n", "\r\n")) == _lowers_alike(exe, lf)
    head = "uint32_t g(uint32_t x) { return x"
    line = head + "+1u" * ((8190 - len(head) - 5) // 3)
    line += (
        " " * (8190 - len(line) - 5) + "+1u;}"
    )  # the twin's longest line, its CRLF end not counted
    assert len(line.encode()) == 8190
    unit = "#include <stdint.h>\nuint32_t f(uint32_t x) {{ return {v}; }}\n"
    longest = "#include <stdint.h>\n" + line + "\n" + unit.format(v="g(x)")
    assert _lowers_alike(exe, longest.replace("\n", "\r\n")) == _lowers_alike(exe, longest)
    for stray in ("\x1c", "\x1e"):
        _refused_alike(
            exe, unit.format(v=f"x{stray} + 1u"), f"unexpected character {stray!r}", phase="PARSE"
        )
    _refused_alike(
        exe, unit.format(v="x") + "\x1c\n", "unexpected character '\\x1c'", phase="PARSE"
    )
    for past in ("\u0085", "\u2028"):
        _refused_alike(exe, unit.format(v=f"x{past} + 1u"), NONASCII, phase="PARSE")
    summary = _lowers_alike(
        exe, "#define K 2u\n#if K ==\r2u\n#define L 3u\n#endif\n" + unit.format(v="x + L")
    )
    assert "const=1" in summary, summary


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


# CF-LIMITS: the bound CF-BUF set where the lexers read a name, held where the text enters before them -- the
# preprocessors and the twin's driver. A macro name or a macro parameter is at most 63 characters on both rails
# (`IDENT_MAX`, C11 5.2.4.1) and a longer one is refused for one reason wherever a directive reads one; the twin's
# canonical serialization returns its whole length, so its driver prints all of it; and the driver reads the whole
# source, as `bcir-cc` does.
_MACRO_NAME_REFUSED = "macro name is too long"
_MACRO_PARAM_REFUSED = "macro parameter is too long"
_UNIT_F = "int f(int a) {{ return a; }}\n"
_LONG_MACRO_UNITS = (
    # (where the name stands, the unit with `{n}` there, the reason past 63 characters, whether the name is written
    # in the directive itself -- one a macro's replacement list holds is read only at 64: the twin's preprocessor
    # refuses a replacement-list token of 256 characters or more for its length before any directive reads it)
    ("#define", "#define {n} 7\nint f(void) {{ return {n}; }}\n", _MACRO_NAME_REFUSED, True),
    (
        "#define (",
        "#define {n}(x) ((x) + 1)\nint f(int a) {{ return {n}(a); }}\n",
        _MACRO_NAME_REFUSED,
        True,
    ),
    (
        "a parameter",
        "#define F({n}) ({n} + 1)\nint f(int a) {{ return F(a); }}\n",
        _MACRO_PARAM_REFUSED,
        True,
    ),
    (
        "a parameter before ...",
        "#define F({n}, ...) ({n})\nint f(int a) {{ return F(a, 2); }}\n",
        _MACRO_PARAM_REFUSED,
        True,
    ),
    ("#undef", "#undef {n}\n" + _UNIT_F, _MACRO_NAME_REFUSED, True),
    (
        "#ifdef",
        "#ifdef {n}\nint g(void) {{ return 0; }}\n#endif\n" + _UNIT_F,
        _MACRO_NAME_REFUSED,
        True,
    ),
    ("#ifndef", "#ifndef {n}\n" + _UNIT_F + "#endif\n", _MACRO_NAME_REFUSED, True),
    ("#elifdef", "#if 0\n#elifdef {n}\n#else\n" + _UNIT_F + "#endif\n", _MACRO_NAME_REFUSED, True),
    ("#elifndef", "#if 0\n#elifndef {n}\n" + _UNIT_F + "#endif\n", _MACRO_NAME_REFUSED, True),
    ("defined", "#if defined {n}\n#else\n" + _UNIT_F + "#endif\n", _MACRO_NAME_REFUSED, True),
    ("defined (", "#if !defined({n})\n" + _UNIT_F + "#endif\n", _MACRO_NAME_REFUSED, True),
    (
        "#elif defined (",
        "#if 0\n#elif defined({n})\n#else\n" + _UNIT_F + "#endif\n",
        _MACRO_NAME_REFUSED,
        True,
    ),
    ("a name #if looks up", "#if {n} == 0\n" + _UNIT_F + "#endif\n", _MACRO_NAME_REFUSED, True),
    (
        "an argument #if drops",
        "#define K(x) 0\n#if K({n}) == 0\n" + _UNIT_F + "#endif\n",
        _MACRO_NAME_REFUSED,
        True,
    ),
    (
        "a name #if expands to",
        "#define M {n}\n#if M == 0\n" + _UNIT_F + "#endif\n",
        _MACRO_NAME_REFUSED,
        False,
    ),
)


def _twin_line(exe: str, src: str, *args: str) -> tuple:
    """The twin driver's exit status and first line of output for the unit `src` (written as bytes, so a NUL in it
    reaches the file)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "u.c")
        with open(path, "wb") as fh:
            fh.write(src.encode("utf-8"))
        run = subprocess.run([exe, *args, path], capture_output=True, text=True)
    return run.returncode, (run.stdout.splitlines() or [""])[0]


def test_a_macro_name_past_63_characters_is_refused_on_both_rails_and_one_at_63_expands():
    """CF-LIMITS: CF-BUF bounded an identifier at 63 characters where the lexers read one, but a macro name never
    reaches a lexer -- the preprocessor replaces it. The twin's preprocessor kept a macro name in 64 bytes and refused
    a longer one at `#define` and `-D` (`macro name is too long`); at `#undef`, `#ifdef`, `#ifndef`, `#elifdef`,
    `#elifndef` and in a `#if` it refused one for the length of a token buffer (`preprocessor token too long`), and
    as the operand of `defined` only past 255 characters, for the same reason. The oracle's took every one, so a unit
    naming a 64-character macro lowered on the oracle and was refused on the twin. Now a macro name -- the one
    `#define` or `-D` defines, `#undef` removes, `#ifdef`, `#ifndef`, `#elifdef`, `#elifndef` or `defined` tests, or
    any name an evaluated `#if` or `#elif` looks up -- past 63 characters is refused on both rails with `macro name is
    too long`, at any length, and a parameter with `macro parameter is too long`. At 63 characters each unit lowers
    to one claim graph on both rails."""
    from bcir.frontends.cfront.cpp import CPPError

    n63, n64, n300 = "q" * 63, "q" * 64, "q" * 300
    exe = _build_frontend(_session_build_dir()) if _CC else None
    for where, shape, why, written in _LONG_MACRO_UNITS:
        for name in (n64, n300) if written else (n64,):
            src = shape.format(n=name)
            try:
                compile_unit(src, check_clang=False)
            except CPPError as e:
                assert str(e) == why, (where, len(name), str(e))
            else:
                raise AssertionError(f"the oracle took a {len(name)}-character name at {where}")
            if exe:
                got = _twin_line(exe, src)
                assert got == (1, f"CPP-ERR {why}"), (where, len(name), got)
        oracle_summary, _r, _entry = _oracle(shape.format(n=n63))
        assert "ok=1" in oracle_summary, (where, oracle_summary)
        if exe:
            assert _twin_line(exe, shape.format(n=n63)) == (0, oracle_summary), where
    # the corpus's adversarial unit, a parameter of 3 000 characters: the oracle had lowered it, and the escape
    # tests pinned it as a limit of the twin's alone (`escape_fixtures.TWIN_PREPROCESSOR_LIMITS`, now empty)
    fx = os.path.join(_C, "cfront_sec_cppmacro.c")
    try:
        compile_unit(open(fx, encoding="utf-8").read(), check_clang=False)
    except CPPError as e:
        assert str(e) == _MACRO_PARAM_REFUSED, str(e)
    else:
        raise AssertionError("the oracle lowered cfront_sec_cppmacro.c")
    if exe:
        run = subprocess.run([exe, fx], capture_output=True, text=True)
        assert (run.returncode, run.stdout.strip()) == (1, f"CPP-ERR {_MACRO_PARAM_REFUSED}"), (
            run.stdout
        )
    # a definition a driver seeds (`-D`): the oracle's `defines`, the twin's `bcir-cc -D` (whose own spec buffer
    # refuses a definition past 255 characters before the preprocessor sees it, so 64 is the length read here)
    unit = "int f(void) {{ return {n}; }}\n"
    try:
        compile_unit(unit.format(n=n64), check_clang=False, defines={n64: "7"})
    except CPPError as e:
        assert str(e) == _MACRO_NAME_REFUSED, str(e)
    else:
        raise AssertionError("the oracle took a 64-character -D name")
    oracle_summary, _r, _entry = _oracle_defines(unit.format(n=n63), {n63: "7"})
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        cc = _build_bcir_cc(d)
        for name, ok in ((n64, False), (n63, True)):
            path = os.path.join(d, "d.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(unit.format(n=name))
            run = subprocess.run([cc, "-D", f"{name}=7", path], capture_output=True, text=True)
            if ok:
                assert run.returncode == 0 and oracle_summary in run.stdout, (
                    run.stdout,
                    run.stderr,
                )
            else:
                assert run.returncode == 1, (run.returncode, run.stderr)
                assert f"preprocessor error: {_MACRO_NAME_REFUSED}" in run.stderr, run.stderr


def _oracle_defines(src: str, defines: dict):
    r = compile_unit(src, check_clang=False, defines=defines)
    return _summary_line(r), r, r.lowered.functions[next(reversed(r.lowered.functions))]


_SKIPPED_DIRECTIVES = (
    "#if 0\n#ifdef {n}\n#endif\n#endif\n",
    "#if 0\n#ifndef {n}\n#endif\n#endif\n",
    "#if 1\n#elifdef {n}\n#endif\n",
    "#if 1\n#elifndef {n}\n#endif\n",
    "#if 0\n#if defined({n}) || {n}\n#endif\n#endif\n",
    "#if 1\n#elif defined {n}\n#endif\n",
    "#if 0\n#define {n} 1\n#undef {n}\n#endif\n",
    # no name where one would be read, and an expression C never evaluates
    "#if 0\n#ifdef\n#elifndef 9\n#endif\n#if 0x\n#endif\n#endif\n",
)


def test_a_skipped_group_reads_no_macro_name_on_either_rail():
    """CF-LIMITS: a directive in a skipped group is processed only through its name (C11 6.10.1p6), as is one
    after a group was taken, so nothing past the name is read. The twin read the operand of `#ifdef`, `#ifndef`,
    `#elifdef` and `#elifndef` there all the same and refused one it could not hold (`preprocessor token too long`,
    or `conditional directive requires an identifier` for none), while the oracle evaluated a skipped `#if` (`#if
    0x` raised `ValueError`) and indexed a skipped `#ifdef`'s missing operand (`IndexError`). Neither rail reads a
    skipped directive's operand now, so a name of any length, or none, in a skipped group lowers the unit to one
    claim graph on both rails."""
    exe = _build_frontend(_session_build_dir()) if _CC else None
    for shape in _SKIPPED_DIRECTIVES:
        for name in ("q" * 64, "q" * 300):
            src = shape.format(n=name) + _UNIT_F.format()
            oracle_summary, _r, _entry = _oracle(src)
            assert "ok=1" in oracle_summary, (shape, oracle_summary)
            if exe:
                assert _twin_line(exe, src) == (0, oracle_summary), (shape, len(name))


_NAMELESS_DIRECTIVES = (
    # (a directive with no macro name where it reads one, the reason both rails give)
    ("#define\n", "macro name must be an identifier"),
    ("#define 9x 1\n", "macro name must be an identifier"),
    ("#undef\n", "#undef requires an identifier"),
    ("#undef 9x\n", "#undef requires an identifier"),
    ("#ifdef\n#endif\n", "conditional directive requires an identifier"),
    ("#ifndef (X)\n#endif\n", "conditional directive requires an identifier"),
    ("#if 0\n#elifdef\n#endif\n", "conditional directive requires an identifier"),
    ("#if 0\n#elifndef 9\n#endif\n", "conditional directive requires an identifier"),
    ("#if defined 9x\n#endif\n", "malformed defined operator"),
    ("#if defined(9x)\n#endif\n", "malformed defined operator"),
)


def test_a_directive_that_reads_no_macro_name_is_refused_on_both_rails():
    """CF-LIMITS: each rail reads a directive's macro name through one predicate (the twin's `macro_name`, the
    oracle's `_macro_name`), the one that bounds its length -- so a directive with no name where it reads one is
    refused on both rails, for the twin's reasons. The oracle had defined a macro named `9x`, ignored a bare
    `#define` and `#undef`, taken `defined 9x` as 0, and raised `IndexError` on a bare `#ifdef`."""
    from bcir.frontends.cfront.cpp import CPPError

    exe = _build_frontend(_session_build_dir()) if _CC else None
    for directive, why in _NAMELESS_DIRECTIVES:
        src = directive + _UNIT_F.format()
        try:
            compile_unit(src, check_clang=False)
        except CPPError as e:
            assert str(e) == why, (directive, str(e))
        else:
            raise AssertionError(f"the oracle took {directive!r}")
        if exe:
            assert _twin_line(exe, src) == (1, f"CPP-ERR {why}"), directive


def _fnv1a(data: bytes) -> int:
    h = 1469598103934665603
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def test_the_twin_driver_prints_a_canon_past_128_kib_whole():
    """CF-LIMITS: `bcir_cfront_canon` wrote the canonical serialization into its caller's buffer and stopped at its
    end without saying so, and the twin's driver held 128 KiB: the 140-function unit's canon (239 KB) came back as
    its first 131071 bytes, exit 0. The canon now returns its whole length (snprintf semantics; `SIZE_MAX` when an
    allocation failed), and the driver measures it, holds it and prints all of it -- byte for byte the oracle's
    `cfront_structural_canon`, whose FNV-1a is the digest both rails' summaries print: the digest hashes the canon as
    it is produced, and is unmoved. Its value numbers numbered (CF-PPLIMITS), the 140-function unit's canon is 116 KB,
    a 170-function unit's 141 KB."""
    if not _CC:
        return
    from bcir.verify import cfront_structural_canon

    src = _wide_unit(170)
    r = compile_unit(src, check_clang=False)
    canon = cfront_structural_canon(r.lowered)
    digest = cfront_structural_digest(r.lowered)
    assert len(canon) > 1 << 17 and _fnv1a(canon.encode()) == digest, len(canon)
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "wide.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        run = subprocess.run([exe, "--canon", path], capture_output=True, text=True)
        assert run.returncode == 0 and len(run.stdout) == len(canon), (
            run.returncode,
            len(run.stdout),
        )
        assert run.stdout == canon
        c_summary, _emit = _c_run(exe, path)
    assert c_summary == _summary_line(r) and f"digest={digest:016x}" in c_summary, c_summary


def test_the_twin_driver_reads_a_source_past_64_kib_whole():
    """CF-LIMITS: the twin's driver (`runtime/c/test_cfront.c`) read the first 64 KiB of a source and lowered that
    prefix without saying so: a function after 70 000 bytes of comment was dropped (`funcs=1`, exit 0, against the
    oracle's two), and so was one after a NUL, where the text stopped. The driver now reads the whole file through
    the host allocator, as `bcir-cc` does: the unit lowers to the oracle's claim graph. A file it cannot hand on
    whole -- one holding a NUL -- is refused; and a unit whose preprocessed text runs past 64 KiB, refused there
    once, lowers whole (CF-PPLIMITS: the text is held in a block grown as it needs)."""
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    src = (
        "int f(int a) { return a + 1; }\n/*" + "x" * 70000 + "*/\nint g(int a) { return a * 3; }\n"
    )
    oracle_summary, r, _entry = _oracle(src)
    assert len(src) > 1 << 16 and list(r.lowered.functions) == ["f", "g"]
    got = _twin_line(exe, src)
    assert got == (0, oracle_summary), (got, oracle_summary)
    nul = "int f(int a) { return a + 1; }\n\0int g(int a) { return a * 3; }\n"
    got = _twin_line(exe, nul)
    assert got == (1, "READ-ERR the source holds a NUL byte"), got
    wide = (
        "".join(f"int v{k} = {k};\n" for k in range(6000)) + "int g(int a) { return a + v5999; }\n"
    )
    assert len(wide) > 1 << 16
    got = _twin_line(exe, wide)
    assert got == (0, _oracle(wide)[0]), got


#: A unit after a directive under test (CF-PPLIMITS).
_PPLIMITS_UNIT = "#include <stdint.h>\nuint32_t f(uint32_t x) {{ return x + {v}; }}\n"


def _twin_cpp(exe: str, src: str) -> tuple:
    """The twin driver's exit status and its `--emit-cpp` text for the unit `src`."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "u.c")
        with open(path, "wb") as fh:
            fh.write(src.encode("utf-8"))
        run = subprocess.run([exe, "--emit-cpp", path], capture_output=True, text=True)
    return run.returncode, run.stdout


def _oracle_refusal(src: str, includes=None) -> str:
    """The reason the oracle refuses the unit `src` for, from any phase ("" when it lowers)."""
    from bcir.frontends.cfront.clex import CLexError
    from bcir.frontends.cfront.cparse import CParseError
    from bcir.frontends.cfront.cpp import CPPError
    from bcir.frontends.cfront.lower import CLowerError

    try:
        compile_unit(src, check_clang=False, includes=includes)
    except (CPPError, CLexError, CParseError, CLowerError) as e:
        return str(e)
    return ""


def test_the_twin_drivers_hold_a_unit_of_any_length_to_the_token_bound_both_rails_share():
    """CF-PPLIMITS: the loop driver (`test_cfront_loop.c`) read the first 64 KiB of a source and lowered that prefix,
    silently -- a 70 KB unit ran its first function as the entry -- and every twin driver held the preprocessed text
    in 64 KiB, so a unit of 6 000 globals was refused there that the oracle lowered. The three drivers read through
    one reader (`bcir_cpp_read_source`) and preprocess into a block grown as the text needs (`bcir_cpp_run_alloc`).
    Past that, the twin's compiler held 16 384 tokens in a fixed array and refused more as `input too large`, where
    the oracle had no cap at all (its comment said the rails agreed): the array grows now, to a bound both lexers
    hold -- `MAX_TOKENS - 1` tokens lower on both rails, one more is refused on both."""
    from bcir.frontends.cfront.clex import INPUT_TOO_LARGE, MAX_TOKENS

    wide = "#include <stdint.h>\n" + "".join(f"uint32_t g{k} = {k}u;\n" for k in range(6000))
    wide += "uint32_t f(uint32_t x) { return x + g5999; }\n"
    oracle_summary, _r, _entry = _oracle(wide)
    exe = None
    if _CC:
        exe = _build_frontend(_session_build_dir())
        assert _twin_line(exe, wide) == (0, oracle_summary)
        loop = _build_loop(_session_build_dir())
        short = "#include <stdint.h>\nuint32_t first(uint32_t x) { return x + 1u; }\n"
        entry = "uint32_t entry(uint32_t x) { return x * 3u + 7u; }\n"
        outs = []
        with tempfile.TemporaryDirectory() as d:
            for name, src in (
                ("short.c", short + entry),
                ("long.c", short + "/*" + "x" * 70000 + "*/\n" + entry),
                ("nul.c", short + "\0" + entry),
            ):
                path = os.path.join(d, name)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(src)
                run = subprocess.run([loop, path], capture_output=True, text=True)
                outs.append((run.returncode, run.stdout.strip()))
        assert (
            outs[0][0] == 0 and outs[0][1].startswith("loop: claims=4 ") and outs[1] == outs[0]
        ), outs
        assert outs[2] == (1, "READ-ERR the source holds a NUL byte"), outs[2]

    # k empty statements, a hundred to a line (a line is at most 8 KiB on the twin)
    def body(k: int) -> str:
        return "\n".join(";" * 100 for _ in range(k // 100)) + "\n" + ";" * (k % 100) + "\n"

    # `int f(void){` ... `return 0;}` is 10 tokens: with `k` more, the unit is `MAX_TOKENS - 1` tokens, or `MAX_TOKENS`
    for k, lowers in ((MAX_TOKENS - 11, True), (MAX_TOKENS - 10, False)):
        src = "int f(void){\n" + body(k) + "return 0;}\n"
        if lowers:
            expect = (0, _oracle(src)[0])
        else:
            assert _oracle_refusal(src) == INPUT_TOO_LARGE, k
            expect = (1, f"PARSE-ERR {INPUT_TOO_LARGE}")
        if exe:
            assert _twin_line(exe, src) == expect, (k, expect)


def test_a_token_of_any_length_within_a_line_is_read_whole_on_both_rails():
    """CF-PPLIMITS: the twin's preprocessor held a token in 256 bytes, a macro argument in 1 KiB and a substitution in
    2 KiB, and refused a longer one (`preprocessor token too long`) where the oracle took it: a 300-character string
    literal, a stringized argument, a `__has_attribute` operand, an argument of 2 000 bytes. Each buffer holds a line
    now, so no token is refused for its own length. And the spacing token it judged a run-on by kept the first 255
    bytes of an argument, so after a longer one it read the wrong last byte: `F(...+b)` with the body `t y` wrote
    `...+by` -- another identifier, which a parameter `by` made a program C refuses lower on the twin alone. It
    writes `...+b y` now, as the oracle does, and both refuse it."""
    arg = "aa" + "+a" * 200 + "+b"
    units = (
        '#include <stdint.h>\nstatic const char *s = "' + "A" * 300 + '";\n'
        "uint32_t f(uint32_t x) { return x + (uint32_t)s[299]; }\n",
        "#include <stdint.h>\n#define S(t) #t\n"
        "uint32_t f(uint32_t x) { return x + (uint32_t)sizeof(S(" + "b" * 300 + ")); }\n",
        "#include <stdint.h>\n#if __has_attribute(" + "z" * 300 + ")\n#error\n#endif\n"
        "uint32_t f(uint32_t x) { return x + 2u; }\n",
        "#include <stdint.h>\n#define ID(t) t\nuint32_t f(uint32_t a) { return ID("
        + "a+" * 1000
        + "a); }\n",
    )
    exe = _build_frontend(_session_build_dir()) if _CC else None
    for k, src in enumerate(units):
        summary, _r, _entry = _oracle(src)
        if exe:
            assert _twin_line(exe, src) == (0, summary), k
    run_on = (
        "#include <stdint.h>\n#define F(t) t y\n"
        "uint32_t f(uint32_t a, uint32_t b, uint32_t y, uint32_t by) { return F(" + arg + "); }\n"
    )
    assert _oracle_refusal(run_on), "the oracle lowered `b y`"
    if exe:
        rc, text = _twin_cpp(exe, run_on)
        assert rc == 0 and "+b y" in text and "+by" not in text, text[-120:]
        assert _twin_line(exe, run_on)[0] == 1


_BAD_DEFINED = (
    "defined(X",
    "defined +",
    "defined",
    "defined()",
    "defined(X Y)",
    "defined(1)",
    "1 || defined",
)


def test_a_malformed_defined_and_a_nul_are_refused_alike_on_both_rails():
    """CF-PPLIMITS: the oracle read `defined` with two regular expressions, which matched a well-formed one only and
    left a malformed one standing, to be refused as a malformed expression; the twin refused it as `malformed defined
    operator`. Both rails read a `#if` token by token now, the oracle as the twin does, and give the twin's reason.
    The oracle's tokenizer also dropped every character no alternative matched -- a NUL between tokens, `@`, a
    character past ASCII -- and lowered what surrounded it (`x + caf€` as `x + caf`): it keeps each as a token now,
    and a NUL in the source or a header is refused before, in the twin's words."""
    exe = _build_frontend(_session_build_dir()) if _CC else None
    for operand in _BAD_DEFINED:
        src = f"#if {operand}\n#endif\n" + _PPLIMITS_UNIT.format(v="1u")
        assert _oracle_refusal(src) == "malformed defined operator", operand
        if exe:
            assert _twin_line(exe, src) == (1, "CPP-ERR malformed defined operator"), operand
    good = (
        "#define X\n#if defined X && defined(X) && defined ( X ) && !defined(Y)\n"
        + _PPLIMITS_UNIT.format(v="7u")
        + "#endif\n"
    )
    summary, _r, _entry = _oracle(good)
    assert "binop=1" in summary, summary
    if exe:
        assert _twin_line(exe, good) == (0, summary)
    nul = _PPLIMITS_UNIT.format(v="1u").replace("x + 1u", "x \0+ 1u")
    assert _oracle_refusal(nul) == "the source holds a NUL byte"
    if exe:
        assert _twin_line(exe, nul) == (1, "READ-ERR the source holds a NUL byte")
    header = '#include "h.h"\n' + _PPLIMITS_UNIT.format(v="1u")
    assert (
        _oracle_refusal(header, includes={"h.h": "#define H 1\0\n"}) == "#include h.h contains NUL"
    )
    if exe:
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "h.h"), "wb") as fh:
                fh.write(b"#define H 1\0\n")
            with open(os.path.join(d, "u.c"), "w", encoding="utf-8") as fh:
                fh.write(header)
            run = subprocess.run([exe, os.path.join(d, "u.c")], capture_output=True, text=True)
        assert (run.returncode, run.stdout.strip()) == (1, "CPP-ERR #include h.h contains NUL"), (
            run.stdout
        )
    for stray in ("@", "`"):
        src = _PPLIMITS_UNIT.format(v="1u " + stray + " 2u")
        assert _oracle_refusal(src) == f"unexpected character {stray!r}", stray
        if exe:
            assert _twin_line(exe, src)[0] == 1, stray


def test_bcir_cc_undefines_a_macro_its_command_line_defines():
    """CF-PPLIMITS: `bcir-cc -DXQ=1 -UXQ` failed every compile (`macro name must be an identifier`): the `-U` turned
    the `-D` into an empty definition, which the preprocessor refused. A `-U` drops every `-D` of its name, wherever it
    stands, as the oracle's CLI pops it, and the branch taken is the oracle's."""
    if not _CC:
        return
    src = "#ifdef XQ\nint f(void){return 11;}\n#else\nint f(void){return 22;}\n#endif\n"
    with tempfile.TemporaryDirectory() as d:
        cc = _build_bcir_cc(d)
        path = os.path.join(d, "u.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        for flags, taken in (
            (["-DXQ=1", "-UXQ"], "22"),
            (["-UXQ", "-DXQ=1"], "22"),
            (["-DXQ"], "11"),
            (["-D", "XQ=2", "-U", "XQ", "-DYQ"], "22"),
        ):
            run = subprocess.run([cc, "-E", *flags, path], capture_output=True, text=True)
            assert run.returncode == 0 and f"return {taken}" in run.stdout, (flags, run.stderr)
            rc, out, err = _cli(["-E", *flags, path])
            assert rc == 0 and f"return {taken}" in out, (flags, err)


_NONASCII_REFUSED = (
    "#define caf 5\n#ifdef café\n#endif\n",
    "#define caf 5\n#ifndef café\n#endif\n",
    "#define caf 5\n#if 0\n#elifdef café\n#endif\n",
    "#define caf 5\n#if defined(café)\n#endif\n",
    "#define caf 5\n#if defined café\n#endif\n",
    "#define caf 5\n#if café\n#endif\n",
    "#define caf 5\n#if é\n#endif\n",
    "#undef café\n",
    "#define café 5\n",
    "#define F(café) 1\n",
    "#define F(é) 1\n",
    "#define F(a, bé) 1\n",
    "#if 1\u00a0\n#endif\n",
    "#café x\n",
    "#\u00a0define K 1\n",
)


def test_a_name_that_runs_into_a_non_ascii_character_is_refused_alike_on_both_rails():
    """CF-PPLIMITS: the twin read the ASCII start of a name and the oracle a whole Unicode word, so beside `#define
    caf 5`, `#ifdef café` kept its group on the twin and skipped it on the oracle -- digest-different -- and the
    oracle's lexer took `int café;` (`str.isalpha`), where the twin's refused it. An identifier is ASCII on both
    rails now: a name a directive reads, a macro parameter, an identifier or a number the lexer reads, refused with
    one reason where a character past ASCII runs on from it or stands outside a literal. One inside a string or a
    comment is the literal's or the comment's, and lowers on both. A directive line is read by the twin's grammar on
    both: its `#` and name across spaces and tabs only, the name an ASCII identifier -- the oracle had stripped every
    Unicode space (`\u00a0#define K 3u` defined `K` there) and ended a name only at a space (`#define\tK 3u` was an
    unknown directive there)."""
    from bcir.frontends.cfront.clex import NONASCII

    exe = _build_frontend(_session_build_dir()) if _CC else None
    for directive in _NONASCII_REFUSED:
        src = directive + _PPLIMITS_UNIT.format(v="1u")
        assert _oracle_refusal(src) == NONASCII, directive
        if exe:
            assert _twin_line(exe, src) == (1, f"CPP-ERR {NONASCII}"), directive
    for code in (
        "#include <stdint.h>\nuint32_t f(uint32_t café) { return café + 1u; }\n",
        "#include <stdint.h>\nuint32_t f(uint32_t x) { return x + ٣; }\n",
        "#include <stdint.h>\n#define caf 5u\nuint32_t f(uint32_t x) { return x + caf€; }\n",
        "#include <stdint.h>\n\u00a0#define K 3u\nuint32_t f(uint32_t x) { return x + 1u; }\n",
        "#include <stdint.h>\n#define K 3u\u00a0\nuint32_t f(uint32_t x) { return x + K; }\n",
    ):
        assert _oracle_refusal(code) == NONASCII, code
        if exe:
            assert _twin_line(exe, code) == (1, f"PARSE-ERR {NONASCII}"), code
    literal = (
        '#include <stdint.h>\n/* café */\nstatic const char *s = "café";\n'
        "uint32_t f(uint32_t x) { return x + (uint32_t)s[1]; }\n"
    )
    # `#line` reads a number of ASCII digits: the oracle's `\d` had read `\u0663` as 3, numbering the next line 3
    # where the twin, which read no number there, left it 2 -- and since CF-PPSPLITS a `#line` whose operand is no
    # decimal digit sequence is refused alike, so the digit past ASCII is refused on both rails as no digit
    line = "#line \u0663\n#include <stdint.h>\nuint32_t f(uint32_t x) { return x + __LINE__; }\n"
    assert _oracle_refusal(line) == "#line number is not a decimal digit sequence"
    if exe:
        assert _twin_line(exe, line) == (1, "CPP-ERR #line number is not a decimal digit sequence")
    spaced = [
        f"#include <stdint.h>\n{d}\nuint32_t f(uint32_t x) {{ return x + K; }}\n"
        for d in (
            "#define\tK 3u",
            "#\tdefine K 3u",
            "#if\t1\n#define K 3u\n#endif",
            "#if(1)\n#define K 3u\n#endif",
        )
    ]
    for src in (literal, *spaced):
        summary, _r, _entry = _oracle(src)
        if exe:
            assert _twin_line(exe, src) == (0, summary), src


_PARAM_LISTS = (
    ("#define F(a b) a\n", "invalid macro parameter list"),
    ("#define F(a b c) a\n", "invalid macro parameter list"),
    ("#define F(1) 1\n", "invalid macro parameter"),
    ("#define F(a, a) a\n", "duplicate macro parameter"),
    ("#define F(a\n", "invalid macro parameter list"),
    ("#define F(a,) a\n", "invalid macro parameter list"),
    ("#define F(..., a) 1\n", "variadic macro parameter must be last"),
    ("#define F(" + ",".join(f"p{k}" for k in range(17)) + ") 1\n", "too many macro parameters"),
)


def test_a_macro_parameter_list_is_read_by_one_grammar_on_both_rails():
    """CF-PPLIMITS: the oracle split a parameter list at its commas and took whatever lay between two as a parameter,
    so `F(a b)`, `F(1)`, `F(a, a)`, an unclosed list and seventeen parameters each defined a macro there that the twin
    refused. The oracle reads a list by the twin's grammar now (`cpp._params`), for the twin's reasons; well-formed
    lists -- spaced, nullary, variadic -- expand alike."""
    exe = _build_frontend(_session_build_dir()) if _CC else None
    for directive, why in _PARAM_LISTS:
        src = directive + _PPLIMITS_UNIT.format(v="1u")
        assert _oracle_refusal(src) == why, directive
        if exe:
            assert _twin_line(exe, src) == (1, f"CPP-ERR {why}"), directive
    good = (
        "#include <stdint.h>\n#define A( a , b ) ((a) - (b))\n#define Z() 3u\n#define V(x, ...) (x + __VA_ARGS__)\n"
        "uint32_t f(uint32_t x) { return A(x, 1u) + Z() + V(x, 2u); }\n"
    )
    summary, _r, _entry = _oracle(good)
    if exe:
        assert _twin_line(exe, good) == (0, summary)


def _canon_chain(n: int, name: str = "f") -> str:
    """A function of n locals, each the one before plus one: a chain of n dependent values (CF-PPLIMITS)."""
    body = "".join(f" uint32_t v{k} = v{k - 1} + 1u;\n" for k in range(1, n))
    return f"uint32_t {name}(uint32_t x, uint32_t *p) {{\n uint32_t v0 = x;\n{body} return v{n - 1};\n}}\n"


def _canon_doubling(n: int, stores: int = 0) -> str:
    """A function of n locals, each the one before added to itself, the last stored through `p` `stores` times: a
    value read twice at every step, whose tree spells 2^n leaves (CF-PPLIMITS)."""
    body = "".join(f" uint32_t x{k} = x{k - 1} + x{k - 1};\n" for k in range(1, n + 1))
    sink = f" *p = x{n};\n" * stores
    return f"uint32_t f(uint32_t x0, uint32_t *p) {{\n{body}{sink} return x{n};\n}}\n"


def test_the_canon_numbers_each_value_once_and_is_linear_on_both_rails():
    """CF-PPLIMITS: the structural canon -- the digest every summary line carries, and `bcir-cc` prints one for each
    file it compiles -- spelled a value number as its value's whole dataflow tree and kept the string, so a chain of
    n dependent values cost it O(n^2) bytes and a value read twice at each of n steps 2^n: 2 000 chained locals took
    the twin 551 MB and 11 s, 8 000 more than 4 GiB, and 18 doublings -- a unit of 500 bytes -- 32 MB of canon, a few
    more any memory there is. A value number is now the FNV-1a (64-bit) of its one-level spelling, its reads' own
    numbers in it, in 16 hex digits, on both rails (`_vn_number`, `vn_claim`): read here out of a hash this test
    computes, `(a - b) - a` numbers its inner value `fnv("c.bin.sub(in:p0,in:p1)")` and returns
    `fnv("c.bin.sub(<that>,in:p0)")`. A record is then its claim's op, imm and reads' numbers, and the canon linear
    in the unit: 1 200 chained locals and 64 doublings stored five times lower digest-equal, the twin's canon byte for
    byte the oracle's and at most 64 bytes a claim. The chain runs first: spelled, its canon is 42 MB and fails here
    in a moment, where the doublings' would take any memory there is."""
    from bcir.verify import cfront_structural_canon

    def num(spelling: str) -> str:
        return f"{_fnv1a(spelling.encode()):016x}"

    exe = _build_frontend(_session_build_dir()) if _CC else None
    head = "#include <stdint.h>\n"
    pin = head + "uint32_t f(uint32_t a, uint32_t b) { return (a - b) - a; }\n"
    inner = num("c.bin.sub(in:p0,in:p1)")
    canon = cfront_structural_canon(compile_unit(pin, check_clang=False).lowered)
    assert canon == (
        f"c.bin.sub|4|{inner},in:p0||0\nc.bin.sub|4|in:p0,in:p1||0\n"
        f"ret={num(f'c.bin.sub({inner},in:p0)')}|stores=\n@\n"
    ), canon
    if exe:
        assert _twin_canon(exe, pin) == canon
    for what, src in (
        ("1 200 chained locals", head + _canon_chain(1200)),
        ("64 doublings stored five times", head + _canon_doubling(64, stores=5)),
    ):
        summary, r, _entry = _oracle(src)
        canon = cfront_structural_canon(r.lowered)
        claims = sum(len(lf.claims) for lf in r.lowered.functions.values())
        assert r.is_clean and len(canon) <= 64 * claims, (what, claims, len(canon))
        if exe:
            assert _twin_line(exe, src) == (0, summary), what
            assert _twin_canon(exe, src) == canon, what


def test_the_twin_names_each_object_once_so_a_run_of_statements_emits_in_linear_time():
    """CF-NAMECACHE: the twin's emit named each object again at every reference, walking every resource and every
    name the emit spells (`uniq_local`, `declared_name`, `spelled_scan`), and R21 rescanned every prior claim for
    every read, so a run of statements on one local cost the square of its length: 1 800 updates of one local
    emitted in 285 ms where CF-PPLIMITS's parent took 97. The names are computed once per function now
    (`names_build`, `names_rank`), R21 reads back to a rid's last event, and the twin's compile of 4 800 updates
    costs under 24 times its compile of 600 (8 times the input; linear at about 8, the walks at 35 and more). The
    time is the child's CPU time (`RUSAGE_CHILDREN`, the min of three runs), which a loaded runner does not inflate
    as it does wall time; where `resource` is absent the wall time stands in. The 600-update unit's summary is the
    oracle's."""
    exe = _build_frontend(_session_build_dir()) if _CC else None
    if not exe:
        return
    head = "#include <stdint.h>\n"

    def updates(n: int) -> str:
        steps = "".join(f" x = x + {k}u;\n" for k in range(1, n + 1))
        return f"{head}uint32_t f(uint32_t x) {{\n{steps} return x;\n}}\n"

    try:
        import resource

        def children_cpu() -> float:
            ru = resource.getrusage(resource.RUSAGE_CHILDREN)
            return ru.ru_utime + ru.ru_stime
    except ImportError:  # no `resource` on this host: the wall clock stands in
        children_cpu = time.perf_counter

    def seconds(src: str) -> float:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "u.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
            best = float("inf")
            for _ in range(3):
                start = children_cpu()
                run = subprocess.run([exe, path], capture_output=True, text=True)
                best = min(best, children_cpu() - start)
                assert run.returncode == 0 and " ok=1 " in run.stdout, run.stdout + run.stderr
        return best

    small = updates(600)
    summary, _r, _entry = _oracle(small)
    assert _twin_line(exe, small) == (0, summary)
    short, long = seconds(small), seconds(updates(4800))
    assert long < 24 * short, (short, long)


def test_a_value_number_past_the_depth_cap_is_cyc_on_both_rails():
    """CF-PPLIMITS: a value number's walk stops past depth 96 (`cyc`) on both rails, but the twin read its memo before
    the depth and the oracle the depth first, so a value an earlier walk numbered whole, reached again one step past
    the cap, was its number on the twin and `cyc` on the oracle: `*p = t + 1u + ... + 1u` of 97 terms beside `return
    t` -- the anchor numbers `t` first -- lowered digest-different. The twin reads the depth first, as the oracle. The
    oracle's canon with its cap lifted shows each witness reaches the cap from 97 terms and not at 96."""
    import bcir.verify as verify

    exe = _build_frontend(_session_build_dir()) if _CC else None
    for terms in (96, 97, 98):
        src = (
            "#include <stdint.h>\nuint32_t f(uint32_t x, uint32_t *p) {\n uint32_t t = x + 1u + 2u + 3u;\n"
            f" *p = t{' + 1u' * terms};\n return t;\n}}\n"
        )
        summary, r, _entry = _oracle(src)
        capped = verify.cfront_structural_canon(r.lowered)
        cap = verify._VN_MAXDEPTH
        verify._VN_MAXDEPTH = 1 << 10
        try:
            whole = verify.cfront_structural_canon(r.lowered)
        finally:
            verify._VN_MAXDEPTH = cap
        assert (capped != whole) == (terms >= 97), terms
        if exe:
            assert _twin_line(exe, src) == (0, summary), terms


def test_claim_ids_stay_unique_past_a_function_of_1000_claims():
    """CF-PPLIMITS: the twin numbered a function's claims from 1000 + 1000 * its index, so a function of more than
    1000 claims ran into the next one's ids, and R1.1 refused a valid unit (`duplicate claim id 2000`, ok=0) the
    oracle -- one sequence over the unit -- lowered clean, digest-equal but for that. Each function's ids start past
    the last one taken; a function of 1000 claims or fewer keeps the ids it had."""
    src = "#include <stdint.h>\n" + _canon_chain(600, "g") + _canon_chain(600)
    summary, r, _entry = _oracle(src)
    ids = [c.id for lf in r.lowered.functions.values() for c in lf.claims]
    assert r.is_clean and " ok=1 " in summary and len(ids) == len(set(ids)) > 3000, summary
    exe = _build_frontend(_session_build_dir()) if _CC else None
    if exe:
        assert _twin_line(exe, src) == (0, summary)


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


# CF-INTCONST: an integer constant is its exact value in its C11 6.4.4.1 type on both rails. The C twin had kept a
# constant past LLONG_MAX as LLONG_MAX and read an octal constant as decimal; the oracle had folded a file-scope
# initializer with unbounded integers, where C evaluates each operation in its operands' types.
_INTCONST_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(ic_hex, s); SAME(ic_octal, s); SAME(ic_binary, s); SAME(ic_decimal, s); SAME(ic_types, s);
    SAME(ic_suffix, s); SAME(ic_edges, s); SAME(ic_edges, ((uint64_t)s << 32) | (uint32_t)~s);
    SAME(ic_dims, s); SAME(ic_globals, s); SAME(ic_entry, s);
  }
  for (uint32_t i = 0; i < 40u; i++) SAME(ic_cases, i);
  SAME(ic_edges, 0xFFFFFFFFFFFFFFFFu); SAME(ic_edges, 0x8000000000000000u); SAME(ic_edges, 0x7FFFFFFFFFFFFFFFu);
  puts("MATCH");
  return 0;
}
"""
)


_PINNED_GLOBALS = {
    "cfront_intconst.c": (
        ("ic_gww", "18446744073709551615u"),
        ("ic_gdiv", "6148914691236517205"),
        ("ic_gmin", "(-9223372036854775807 - 1)"),
        ("ic_gq", "-3"),
        ("ic_gw", "4294967295"),
    ),
}


def _global_values_driver(lowered) -> str:
    """A `main` printing each file-scope object the unit defines -- an array element by element -- as a signed
    and an unsigned 64-bit value, read out of the oracle's own record of the unit's globals."""
    lines = []
    for name, ct, _vals, is_extern, _static in lowered.globals_decl:
        if is_extern:
            continue
        if ct.kind == "array":
            lines += [f"P({name}[{i}]);" for i in range(ct.count)]
        else:
            lines.append(f"P({name});")
    return (
        '#define P(g) printf(#g " %lld %llu\\n", (long long)(g), (unsigned long long)(g))\n'
        "int main(void) {\n  " + "\n  ".join(lines) + "\n  return 0;\n}\n"
    )


def _linkable_globals_are_the_originals(fx: str, src: str, r) -> None:
    """The oracle's linkable emit defines the unit's globals with the initializers it folded: built beside
    `_global_values_driver` (and the quarantine its bounds guards call), under every compiler at hand, each
    global holds what the original's does."""
    from bcir.frontends.cfront.emit import emit_linkable

    head = "#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
    driver = _global_values_driver(r.lowered)
    linkable = emit_linkable(r.lowered, r.emitted)
    # a value past LLONG_MAX is spelled unsigned and LLONG_MIN as an expression: a decimal constant past
    # LLONG_MAX without `u` has no type both compilers agree on, and neither rail takes one back
    for name, value in _PINNED_GLOBALS.get(fx, ()):
        assert re.search(rf"\b{name}(\[\d+\])? = {re.escape(value)};", linkable), (fx, name, value)
    quarantine = ("-I", _C, os.path.join(_C, "bcir_quarantine.c"))
    compilers = [c for c in dict.fromkeys((_CC, shutil.which("clang"), shutil.which("gcc"))) if c]
    with tempfile.TemporaryDirectory() as d:
        for cc in compilers:
            want = _build_run_c(d, cc, "original", f"{head}{src}\n{driver}")
            got = _build_run_c(d, cc, "linkable", f"{head}{linkable}\n{driver}", quarantine)
            assert want and got == want, (
                f"{fx}: the linkable emit's globals are not the original's under {cc}\n"
                + "\n".join(
                    f"  {w!r} != {g!r}"
                    for w, g in zip(want.splitlines(), got.splitlines())
                    if w != g
                )
            )


def test_integer_constants_are_exact_on_both_rails():
    """CF-INTCONST: `cfront_intconst.c` -- integer constants in every base (hexadecimal, octal, binary, decimal)
    and with every suffix, at the edges of their types (`0x8000000000000000`, 64 binary digits, `2^64 - 1` in
    octal, the decimal constants that turn `long`), their types read through a comparison with -1 and `sizeof`,
    octal case labels and array dimensions -- lowers to one claim graph on the four targets, and each function
    of each emit returns what the original does. The twin had saturated a constant past LLONG_MAX
    (`0xFFFFFFFFFFFFFFFFu` was 9223372036854775807) and read `017` as 17. The file-scope initializers -- a
    signed division and remainder, shifts, `~` of an unsigned int, a cast, `sizeof`, a select -- fold in C's own
    types through the fold a static's initializer takes: the oracle's linkable emit defines each global with
    the value the original's holds (`-7 / 2` had rendered -4, `~0u` -1)."""
    if not _CC:
        return
    fx = "cfront_intconst.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    for rail, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        for whole in ("18446744073709551615u", "9223372036854775808u", "9223372036854775809u"):
            assert whole in emit, f"{rail}: the emit does not spell the constant {whole}"
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _INTCONST_DRIVER)
    _linkable_globals_are_the_originals(fx, src, compile_unit(src, check_clang=False))


# The integer constants no type in their list can hold (C11 6.4.4p2), and malformed ones: refused on both rails
# where they are lexed. A decimal constant with no `u` past LLONG_MAX has a type C leaves to the implementation
# -- GCC's `__int128`, Clang's `unsigned long long` -- so it is refused too.
_INT_TOO_LARGE = "an integer constant too large for every type its base and suffix allow"
_INT_INVALID = "invalid integer literal"
_INTLIT_REFUSED = (
    ("18446744073709551616u", _INT_TOO_LARGE),
    ("0x10000000000000000", _INT_TOO_LARGE),
    ("02000000000000000000000", _INT_TOO_LARGE),
    ("0b1" + "0" * 64, _INT_TOO_LARGE),
    ("9223372036854775808", _INT_TOO_LARGE),
    ("18446744073709551615LL", _INT_TOO_LARGE),
    ("08", _INT_INVALID),
    ("0b102", _INT_INVALID),
    ("0o17", _INT_INVALID),
    ("0x", _INT_INVALID),
    ("12ab", _INT_INVALID),
)
# ... and the largest constant of each form, which both rails take
_INTLIT_LIMITS = (
    "18446744073709551615u",
    "0xFFFFFFFFFFFFFFFF",
    "01777777777777777777777",
    "0b" + "1" * 64,
    "9223372036854775807",
    "9223372036854775807LL",
    "0777",
    "0b101",
)


def test_an_integer_constant_no_type_holds_is_refused_on_both_rails():
    """CF-INTCONST: a constant past ULLONG_MAX in any base, a decimal constant without `u` past LLONG_MAX, and a
    malformed constant (a digit its base lacks, `0o17`, which Python's `int` read as octal, a bare `0x`) are
    refused on both rails where they are lexed, each for its reason; the twin had saturated the large ones and
    read `08` as 8. The largest constant of each form lowers to one claim graph on both rails. A call in a
    file-scope initializer is refused on both too: the twin skips a global's initializer, and no longer reads
    it with the enum evaluator, which had refused a cast or `sizeof` there."""
    from bcir.frontends.cfront.clex import CLexError
    from bcir.frontends.cfront.cparse import CParseError

    head = "#include <stdint.h>\n"
    shape = "uint64_t f(uint64_t s) {{ return s + {lit}; }}\n"
    for lit, why in _INTLIT_REFUSED:
        try:
            compile_unit(head + shape.format(lit=lit), check_clang=False)
        except CLexError as e:
            assert why in str(e), (lit, str(e))
        else:
            raise AssertionError(f"the oracle lowered the constant {lit}")
    call = "static uint32_t k(void) { return 3u; }\nuint32_t g = k();\nuint32_t f(void) { return g; }\n"
    try:
        compile_unit(head + call, check_clang=False)
    except CParseError as e:
        assert "calls a function" in str(e), str(e)
    else:
        raise AssertionError("the oracle lowered a call in a file-scope initializer")
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lit.c")
        for lit, why in _INTLIT_REFUSED + (("", "calls a function"),):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(head + (shape.format(lit=lit) if lit else call))
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode == 1 and why in run.stdout, (
                lit,
                run.returncode,
                run.stdout[:200],
            )
        for lit in _INTLIT_LIMITS:
            text = head + shape.format(lit=lit)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            oracle_summary, _r, _entry = _oracle(text)
            c_summary, _emit = _c_run(exe, path)
            assert c_summary == oracle_summary and "ok=1" in c_summary, (
                lit,
                c_summary,
                oracle_summary,
            )


# CF-GARRAY: file-scope arrays and character tables pass through cfront as local ones do. Each function writes the
# globals, so the driver runs the original and the emit from one saved state and compares what each returns and
# leaves behind.
_GARRAY_DRIVER = (
    _GAPS_SAME
    + r"""
#define GA_STATE(X) X(ga_m) X(ga_w) X(ga_name) X(ga_pad) X(ga_bytes) X(ga_a) X(ga_b) X(ga_x) X(ga_y) X(ga_p) \
  X(ga_z) X(ga_s1) X(ga_s2) X(ga_u) X(ga_v) X(ga_arr) X(ga_c1) X(ga_c2)
static unsigned char st_before[1024], st_orig[1024], st_emit[1024];
static size_t ga_save(unsigned char *b) {
  size_t n = 0;
#define GA_SAVE(g) memcpy(b + n, &(g), sizeof(g)); n += sizeof(g);
  GA_STATE(GA_SAVE)
  return n;
}
static void ga_load(const unsigned char *b) {
  size_t n = 0;
#define GA_LOAD(g) memcpy(&(g), b + n, sizeof(g)); n += sizeof(g);
  GA_STATE(GA_LOAD)
}
#define SAME_STATE(f, ...) do { size_t n_ = ga_save(st_before); uint64_t r1_ = (uint64_t)f(__VA_ARGS__); \
    ga_save(st_orig); ga_load(st_before); uint64_t r2_ = (uint64_t)bcir_##f(__VA_ARGS__); ga_save(st_emit); \
    if (r1_ != r2_ || memcmp(st_orig, st_emit, n_)) return fail(#f); } while (0)
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME_STATE(ga_pass, s); SAME_STATE(ga_chars, s); SAME_STATE(ga_lists, s); SAME_STATE(ga_structs, s);
    SAME_STATE(ga_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)


def test_file_scope_arrays_and_character_tables_run_as_the_original_on_both_rails():
    """CF-GARRAY: `cfront_garray.c` -- a 2-D and a 3-D global passed to row-pointer parameters (`T (*p)[N]`,
    `T m[][N]`), character tables sized by their string literals (concatenated, with escapes, and in a larger
    array), and declarations that list several objects: arrays, scalars, a pointer, initialized ones, a `static`
    list, structs, and the objects of a struct the declaration itself defines. Both emits passed a
    multi-dimensional global by name, whose type is still `T[A][N]`, where the emitted parameter is the flat `T *`,
    which Clang and GCC reject; each now passes its first element's address. The twin refused a character array
    initialized by a string (`non-constant enum initializer`) and bounded an unsized global array's accesses by one
    element; both rails refused a declaration's second declarator. The unit lowers to one claim graph on the four
    targets, and each function of each emit returns -- and leaves the globals -- as the original does."""
    if not _CC:
        return
    fx = "cfront_garray.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    for rail, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        for whole in ("(&ga_m[0][0], ", "(&ga_w[0][0][0], ", '5u, "ga_chars:ga_name")'):
            assert whole in emit, f"{rail}: the emit does not spell {whole!r}"
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _GARRAY_DRIVER)


# CF-GBRACE, CF-TLS: a file-scope initializer is walked as C walks it, and thread storage is kept. `gb_tls_bump`
# writes the thread-local global both builds share, so the original and the emit run it from one saved value. The
# threaded run repeats the round in a new thread, where every thread-local object starts at its initial value: an
# emit that declared a thread-local static a plain `static` carries the first thread's value into it.
_GBRACE_ROUND = (
    _GAPS_SAME
    + r"""
static int gb_round(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n], t0 = gb_tls, r1 = gb_tls_bump(s), t1 = gb_tls;
    gb_tls = t0;
    if (bcir_gb_tls_bump(s) != r1 || gb_tls != t1) return fail("gb_tls_bump");
    SAME(gb_pt_at, s); SAME(gb_counts); SAME(gb_row_at, s, s >> 2); SAME(gb_name_at, s, s >> 1);
    SAME(gb_rec_at, s, s >> 3); SAME(gb_seg_at, s); SAME(gb_cube_at, s); SAME(gb_scalars); SAME(gb_entry, s);
    SAME(gb_tls_count, s); SAME(gb_tls_table, s); SAME(gb_tls_point, s); SAME(gb_tls_zero, s);
  }
  return 0;
}
"""
)
_GBRACE_DRIVER = (
    _GBRACE_ROUND
    + r"""
int main(void) {
  if (gb_round()) return 1;
  puts("MATCH");
  return 0;
}
"""
)
_GBRACE_THREADS = (
    _GBRACE_ROUND
    + r"""
#include <pthread.h>
static void *gb_thread(void *arg) { (void)arg; return gb_round() ? (void *)1 : (void *)0; }
int main(void) {
  if (gb_round()) return 1;   /* this thread's objects move on */
  pthread_t t;
  void *bad = (void *)1;
  if (pthread_create(&t, 0, gb_thread, 0) || pthread_join(t, &bad)) { puts("pthread"); return 2; }
  if (bad) return 1;
  puts("MATCH");
  return 0;
}
"""
)


def _linkable_bytes_are_the_originals(fx: str, src: str, r) -> None:
    """The oracle's linkable emit defines each of the unit's globals as the original does: built beside a `main`
    that dumps every global's bytes -- a static-storage object's padding is zero too (C11 6.7.9p10) -- under every
    compiler at hand, each holds the original's bytes. The linkable emit defines the unit's structs and unions as
    the source does (CF-LINKEMIT), so its build takes nothing of the source. Where pthreads are, `main` then
    overwrites each thread-local global and a new thread dumps it: it holds its initial bytes again, as the
    original's does."""
    from bcir.frontends.cfront.emit import emit_linkable

    linkable = emit_linkable(r.lowered, r.emitted)
    names = [g[0] for g in r.lowered.globals_decl if not g[3]]
    tls = sorted(r.lowered.thread_globals) if os.name == "posix" else []
    driver = (
        "static void dump(const char *name, const void *p, size_t n) {\n"
        "  const unsigned char *b = p;\n"
        '  printf("%s", name);\n'
        '  for (size_t i = 0; i < n; i++) printf(" %02x", b[i]);\n'
        "  putchar('\\n');\n"
        "}\n"
        "#define D(g) dump(#g, &(g), sizeof(g))\n"
    )
    if tls:
        driver += (
            "#include <pthread.h>\n"
            "static void *tls_dump(void *arg) {\n  (void)arg;\n  "
            + " ".join(f"D({n});" for n in tls)
            + "\n  return 0;\n}\n"
        )
    driver += "int main(void) {\n  " + " ".join(f"D({n});" for n in names) + "\n"
    if tls:
        driver += (
            "  " + " ".join(f"memset(&{n}, 0xA5, sizeof {n});" for n in tls) + "\n"
            "  pthread_t t;\n"
            '  if (pthread_create(&t, 0, tls_dump, 0) || pthread_join(t, 0)) puts("pthread");\n'
        )
    driver += "  return 0;\n}\n"
    head = "#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n#include <stddef.h>\n"
    threads = ("-pthread",) if tls else ()
    quarantine = ("-I", _C, os.path.join(_C, "bcir_quarantine.c"))
    compilers = [c for c in dict.fromkeys((_CC, shutil.which("clang"), shutil.which("gcc"))) if c]
    with tempfile.TemporaryDirectory() as d:
        for cc in compilers:
            want = _build_run_c(d, cc, "original", f"{head}{src}\n{driver}", threads)
            got = _build_run_c(
                d, cc, "linkable", f"{head}{linkable}\n{driver}", quarantine + threads
            )
            assert want and got == want, (
                f"{fx}: the linkable emit's globals are not the original's under {cc}\n"
                + "\n".join(
                    f"  {w!r} != {g!r}"
                    for w, g in zip(want.splitlines(), got.splitlines())
                    if w != g
                )
            )


def test_file_scope_initializers_and_thread_storage_run_as_the_original_on_both_rails():
    """CF-GBRACE, CF-TLS: `cfront_gbrace.c` -- arrays of structs, rows, a 3-D array, character tables, a nested
    struct, a union and a braced scalar initialized at file scope with nested braces, brace elision and designators,
    and a thread-local global and four thread-local statics (a scalar, an array, a struct and a zero one, the
    storage class before and after `static` and after the type). The oracle refused a nested brace in a global's
    initializer, and both rails sized an unsized global by its top-level entries; each global is now walked as a
    local is, for its shape, and both rails dropped `_Thread_local` from a static local, which both emits now keep.
    The unit lowers to one claim graph on the four targets; each function of each emit returns -- and leaves the
    thread-local global -- as the original does, in a second thread too; and the oracle's linkable emit renders
    each global's braces and designators, holding the original's bytes."""
    if not _CC:
        return
    fx = "cfront_gbrace.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    for rail, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        for whole in (
            "static _Thread_local uint32_t n = 3u;",
            "static _Thread_local uint32_t t[3] = {1u, 2u};",
            "static _Thread_local struct gb_pt p = {0u, 4u};",
            "static _Thread_local uint64_t z = 0u;",
        ):
            assert whole in re.sub(r"\s+", " ", emit), (
                f"{rail}: the emit does not declare {whole!r}"
            )
    emits = (("twin", c_emit), ("oracle", oracle_emit))
    _run_against_original(fx, src, emits, _GBRACE_DRIVER)
    if os.name == "posix":  # the threaded witness needs pthreads
        _run_against_original(fx, src, emits, _GBRACE_THREADS, ("-pthread",))
    _linkable_bytes_are_the_originals(fx, src, compile_unit(src, check_clang=False))


# A file-scope initializer C refuses, or the walk does not take, is refused on both rails for one reason: each
# global's initializer is walked as a local's is (C11 6.7.9p2, p11, p14, p17-19). A block-scope `_Thread_local`
# object needs `static` too (C11 6.7.1p3).
_GBRACE_REFUSED = (
    ("uint32_t g[2] = {1u, 2u, 3u};", "excess elements in an initializer"),
    ("struct pt g = {1u, 2u, 3u};", "excess elements in an initializer"),
    (
        "struct pt g[1] = {[0].x = 3u, [0] = {1u, 2u}};",
        "an initializer overrides a prior initialization of a subobject",
    ),
    ('char g[2] = "abc";', "an initializer-string for a character array is too long"),
    ("uint32_t g[2] = {[5] = 1u};", "an array designator outside the array"),
    (
        "uint8_t g[2][2][2][2] = {0};",
        "an initialized file-scope array of more than 3 dimensions is not supported",
    ),
    ("uint32_t g[2] = 5u;", "an array is initialized by a brace list or a string literal"),
    ('char g[2][4] = "abc";', "an array is initialized by a brace list or a string literal"),
    ("uint32_t g = {5u, 6u};", "a braced scalar initializer holds one expression"),
    (
        "uint32_t k(void);\nuint32_t g[1] = {1u, k()};",
        "calls a function (not a constant expression)",
    ),
    ("uint32_t k(void);\nuint32_t g = (k)();", "calls a function (not a constant expression)"),
    (
        "uint32_t f2(uint32_t s) { _Thread_local uint32_t n = 1u; n += s; return n; }",
        "a block-scope `_Thread_local` object that is not `static`",
    ),
)


def test_file_scope_initializers_the_walk_refuses_are_refused_on_both_rails():
    """CF-GBRACE, CF-TLS: an excess entry, an override of an initialized subobject, a string too long, a designator
    outside its array, an initialized array of more than three dimensions, an array initialized by an expression or
    a 2-D one by a string, two entries for a scalar, and a call -- by name, or through a parenthesized callee, which
    the twin's scan had taken -- are each refused on both rails, for the same reason, the call before the walk, as
    the oracle's parser refuses it. A block-scope `_Thread_local` object that is not `static` is refused on both,
    which the twin had read as an expression. A static differing only in its storage duration lowers to another
    claim graph, on both rails alike."""
    from bcir.frontends.cfront.cparse import CParseError
    from bcir.frontends.cfront.lower import CLowerError

    head = "#include <stdint.h>\nstruct pt { uint32_t x, y; };\n"
    cases = [
        (f"{head}{decl}\nuint32_t f(void) {{ return 1u; }}\n", why) for decl, why in _GBRACE_REFUSED
    ]
    for text, why in cases:
        try:
            compile_unit(text, check_clang=False)
        except (CParseError, CLowerError) as e:
            assert why in str(e), (text, str(e))
        else:
            raise AssertionError(f"the oracle lowered {text!r}")
    pair = (
        "uint32_t f(uint32_t s) { static uint32_t n = 3u; n += s; return n; }\n",
        "uint32_t f(uint32_t s) { static _Thread_local uint32_t n = 3u; n += s; return n; }\n",
    )
    oracle = [_summary_line(compile_unit(head + p, check_clang=False)) for p in pair]
    assert oracle[0] != oracle[1], oracle
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "g.c")
        for text, why in cases:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode == 1 and why in run.stdout, (
                text,
                run.returncode,
                run.stdout[:200],
            )
        for p, want in zip(pair, oracle):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(head + p)
            assert _c_run(exe, path)[0] == want, (p, want)


# CF-VOIDCB: a call through a pointer to a function returning void. Both rails lowered it as a claim writing a
# result temp, and each emit declared `uint32_t t = fp(s);`, which Clang and GCC reject -- so no void callback
# compiled. Each function of `cfront_voidcallback.c` starts the callbacks' counter at 0 and returns it.
_VOIDCB_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(vc_local, s); SAME(vc_param, s); SAME(vc_member, s); SAME(vc_arrow, s); SAME(vc_cond, s);
    SAME(vc_return, s); SAME(vc_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)
# ... and the forms whose calls the G10 escape rows count as not resolved to one function (the analysis keeps one
# set per pointer and per struct) or not narrowed at all (a loaded pointer chain, a file-scope table), which a
# corpus fixture keeps out: a pointer given two functions, a struct holding several, a member through a loaded
# pointer chain, a file-scope ops table
_VOIDCB_OPEN = """#include <stdint.h>
typedef void (*vo_fn)(uint32_t);
struct vo_ops { void (*step)(uint32_t); uint32_t k; vo_fn reset; void (*tick)(void); };
struct vo_dev { struct vo_ops *ops; uint32_t id; };
static uint32_t vo_n;
static void vo_bump(uint32_t x) { vo_n += x; }
static void vo_twice(uint32_t x) { vo_n += 2u * x + 1u; }
static void vo_tick(void) { vo_n += 100u; }
static const struct vo_ops vo_table = {vo_bump, 7u, vo_twice, vo_tick};
static void vo_apply(vo_fn f, uint32_t s) { f(s); }
uint32_t vo_calls(uint32_t s) {
  struct vo_ops o = {vo_bump, 2u, vo_twice, vo_tick};
  struct vo_dev d = {&o, 9u};
  struct vo_dev *p = &d;
  void (*fp)(uint32_t) = vo_bump;
  vo_n = 0u;
  fp(s);
  fp = vo_twice;
  fp(s + 1u);
  vo_apply(vo_bump, s);
  vo_apply(vo_twice, s ^ 5u);
  o.step(s);
  o.step = vo_twice;
  o.step(s);
  o.tick();
  p->ops->step(s);
  p->ops->reset(s + p->id);
  p->ops->tick();
  vo_table.step(s);
  vo_table.reset(s + vo_table.k);
  vo_table.tick();
  return vo_n;
}
"""


def test_void_callbacks_run_as_the_original_on_both_rails():
    """CF-VOIDCB: `cfront_voidcallback.c` -- a call through a pointer to a void function held in a local, a
    typedef'd local, a parameter (a typedef or a declarator), a struct member (declared or typedef'd) reached by
    `.` or `->`, one taking no arguments, one in an arm of `?:`, one cast to void, one returned from a void
    function -- and `_VOIDCB_OPEN`: a pointer given two functions, a struct holding several, a member through a
    loaded pointer chain, a file-scope ops table. Both rails lowered each call as a claim writing a result temp
    and emitted `uint32_t t = fp(s);`, which does not compile. Such a call now writes nothing -- the void value,
    as a direct void call's -- on both rails, which lower each unit to one claim graph on the four targets; each
    emit spells a bare call and returns what the original does, function by function."""
    if not _CC:
        return
    fx = "cfront_voidcallback.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    for unit in (src, _VOIDCB_OPEN):
        calls = [
            c
            for lf in compile_unit(unit, check_clang=False).lowered.functions.values()
            for c in lf.claims
            if c.op == "c.call.indirect" or c.op.startswith("c.call.imember:")
        ]
        assert calls and all(not c.wr for c in calls), [(c.op, c.wr) for c in calls]
    for rail, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        for call in ("fp(s);", "t.tick();", "o->step(s);", "r->reset(s);"):
            assert re.search(rf"^\s+{re.escape(call)}$", emit, re.M), (rail, call)
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _VOIDCB_DRIVER)

    oracle_summary, r, _entry = _oracle(_VOIDCB_OPEN)
    assert "ok=1" in oracle_summary, oracle_summary
    oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "voidcb_open.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_VOIDCB_OPEN)
        c_summary, c_emit = _c_run(_build_frontend(_session_build_dir()), path)
        assert c_summary == oracle_summary, f"parity\n C: {c_summary}\nPY: {oracle_summary}"
        _parity_on_targets(path, _VOIDCB_OPEN)
    driver = (
        _GAPS_SAME
        + "int main(void) {\n  for (unsigned n = 0; n < GAPS_N; n++) SAME(vo_calls, gaps_in[n]);\n"
        + '  puts("MATCH");\n  return 0;\n}\n'
    )
    emits = (("twin", c_emit), ("oracle", oracle_emit))
    _run_against_original("voidcb_open.c", _VOIDCB_OPEN, emits, driver)


# CF-FNSEL: the arms of `?:` that are function designators decay to function pointers (C11 6.3.2.1p4, 6.5.15p6).
# Both rails typed such a select `uint32_t`, so `uint32_t t = (c ? f : g);` did not compile. The pointers
# `cfront_fnselect.c` returns are compared with the original's and called here.
_FNSELECT_DRIVER = (
    _GAPS_SAME
    + r"""
#define SAME_PICK(f) do { if (f(s) != bcir_##f(s) || f(s)(s) != bcir_##f(s)(s)) return fail(#f); } while (0)
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME_PICK(fs_pick); SAME_PICK(fs_pick_decl); SAME_PICK(fs_pick_assign); SAME_PICK(fs_pick_nested);
    fs_n = 0u;
    fs_op fa = fs_pick_branch(s);
    uint32_t na = fs_n;
    fs_n = 0u;
    if (fa != bcir_fs_pick_branch(s) || na != fs_n) return fail("fs_pick_branch");
    fs_n = s; fs_pick_act(s)(s + 1u); na = fs_n;
    fs_n = s; bcir_fs_pick_act(s)(s + 1u);
    if (fs_pick_act(s) != bcir_fs_pick_act(s) || na != fs_n) return fail("fs_pick_act");
    struct fs_pair p = fs_pick_mk(s)(s), q = bcir_fs_pick_mk(s)(s);
    if (fs_pick_mk(s) != bcir_fs_pick_mk(s) || p.a != q.a || p.b != q.b) return fail("fs_pick_mk");
    SAME(fs_pass, s); SAME(fs_object, s); SAME(fs_null, s); SAME(fs_compare, s); SAME(fs_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)
# ... and a call through a select in place: a pointer the escape rows count as resolved to no one function, which a
# corpus fixture keeps out (a declarator, a typedef'd local, a member, an argument, a branch's local), and designators
# of functions whose parameter a `typeof` or a `va_list` types
_FNSEL_CALLS = """#include <stdint.h>
#include <stdarg.h>
typedef uint32_t (*op_t)(uint32_t);
struct ops { op_t run; uint32_t k; };
static uint32_t tw(uint32_t x) { return x * 2u + 1u; }
static uint32_t th(uint32_t x) { return x * 3u + 5u; }
static uint32_t apply(op_t f, uint32_t s) { return f(s) + 1u; }
static uint32_t tyo(uint32_t a, typeof(a) n) { return a * 5u + n; }
static uint32_t tyt(uint32_t a, uint32_t n) { return a * 7u + n; }
static uint32_t vfirst(uint32_t n, va_list ap) { return n + va_arg(ap, uint32_t); }
static uint32_t vsecond(uint32_t n, va_list ap) { return n * 3u + va_arg(ap, uint32_t); }
static uint32_t vrun(uint32_t s, uint32_t n, ...) {
  va_list ap;
  va_start(ap, n);
  uint32_t (*vf)(uint32_t, va_list) = s & 8u ? vfirst : vsecond;
  uint32_t r = vf(n, ap);
  va_end(ap);
  return r;
}
uint32_t fsc_calls(uint32_t s) {
  uint32_t (*fp)(uint32_t) = s > 3u ? tw : th;
  op_t f = s & 1u ? th : tw;
  struct ops o = {tw, 4u};
  o.run = s & 2u ? th : tw;
  uint32_t n = 0u;
  op_t g = s > 9u ? (n++, tw) : th;
  uint32_t (*q)(uint32_t, uint32_t) = s & 4u ? tyo : tyt;
  return fp(s) + f(s) * 3u + apply(s > 100u ? tw : th, s) * 5u + o.run(s + o.k) * 7u + g(s) * 11u + n
         + q(s, 3u) * 13u + vrun(s, s, 9u) * 17u;
}
"""
# ... and arms that point to functions of two different types are refused on both rails (6.5.15p3), where Clang and
# GCC only warn and give the select `void *`: a designator or a function-pointer object, whose return type, parameter
# count or a parameter's type differs -- a parameter a `typeof` types too, and a `va_list` beside an integer of its size
_FN_SELECTED = "the arms of `?:` point to functions of different types"
_FNSEL_HEAD = (
    "#include <stdint.h>\n#include <stdarg.h>\ntypedef uint32_t (*op_t)(uint32_t);\n"
    "static uint32_t tw(uint32_t x) { return x * 2u; }\n"
    "static uint32_t two(uint32_t x, uint32_t y) { return x + y; }\n"
    "static uint64_t wide(uint32_t x) { return x; }\n"
    "static uint32_t narrow(uint16_t x) { return x; }\n"
    "static uint32_t flt(float x) { return (uint32_t)x; }\n"
    "static void none(uint32_t x) { (void)x; }\n"
    "static uint32_t tyw(uint64_t a, typeof(a) n) { return (uint32_t)(a + n); }\n"
    "static uint32_t vl(uint32_t n, va_list ap) { return n + va_arg(ap, uint32_t); }\n"
    "static uint32_t vw(uint32_t n, uint64_t k) { return n + (uint32_t)k; }\n"
)
_FNSEL_REFUSED = (
    "uint32_t f(uint32_t s) { op_t g = s ? tw : two; return g(s); }",
    "uint32_t f(uint32_t s) { op_t g = s ? wide : tw; return g(s); }",
    "uint32_t f(uint32_t s) { op_t g = s ? tw : narrow; return g(s); }",
    "uint32_t f(uint32_t s) { op_t g = s ? flt : tw; return g(s); }",
    "uint32_t f(uint32_t s) { op_t g = tw; g = s > 2u ? g : none; return g(s); }",
    "uint32_t f(uint32_t s) { op_t h = tw; uint32_t (*k)(uint32_t, uint32_t) = two; op_t g = s ? h : k; return g(s); }",
    "uint32_t f(uint32_t s) { op_t h = tw; op_t g = s ? (s > 1u ? h : tw) : wide; return g(s); }",
    "uint32_t f(uint32_t s) { return (s ? two : tyw) != 0; }",
    "uint32_t f(uint32_t s) { return (s ? vl : vw) != 0; }",
    # a function pointer read from a member, typed as one since CF-FPTAB -- an integer on both rails before it
    "struct ops { op_t run; };\n"
    "uint32_t f(uint32_t s) { struct ops o = {tw}; return (s ? o.run : wide) != 0; }",
)
# ... while the same function type spelled another way, a designator beside a null pointer, and a member read beside a
# function of its own type, read as it is or through `*` (CF-FPTAB), lower alike
_FNSEL_LOWERED = (
    "static uint32_t same(unsigned int x) { return x + 1u; }\n"
    "uint32_t f(uint32_t s) { op_t h = same; op_t g = s ? tw : h; return g(s); }",
    "uint32_t f(uint32_t s) { op_t g = s ? 0 : tw; return g ? g(s) : 1u; }",
    "struct ops { op_t run; };\n"
    "uint32_t f(uint32_t s) { struct ops o = {tw}; op_t g = s ? o.run : tw; return g(s); }",
    "struct ops { op_t run; };\n"
    "uint32_t f(uint32_t s) { struct ops o = {tw}; op_t g = s ? *o.run : tw; return g(s); }",
)


def test_function_designator_arms_select_a_function_pointer_on_both_rails():
    """CF-FNSEL: `cfront_fnselect.c` -- the arms of `?:` that are function designators, or a designator and a
    function-pointer object, or two objects, or one of them and a null pointer constant (`c ? f : 0`), nested, in a
    branch whose arm has an effect, of void and of struct-returning functions -- selected, assigned, passed, called
    and compared. Both rails typed the select `uint32_t` and emitted `uint32_t t = (c ? f : g);`, which does not
    compile. The select is now a pointer to the arms' function type on both rails -- and a null pointer arm that
    pointer -- which lower the fixture and `_FNSEL_CALLS` (calls through a select in place) to one claim graph on the
    four targets; each emit returns what the original does, and the pointers it returns are the original's. Arms
    that point to functions of different types (`_FNSEL_REFUSED`) are refused on both rails -- a member read among them
    since CF-FPTAB typed it; the same type spelled another way, a null pointer arm and a member read of the arms' own
    type (`_FNSEL_LOWERED`) lower alike on both."""
    from bcir.frontends.cfront.lower import CLowerError

    for body in _FNSEL_REFUSED:
        try:
            compile_unit(_FNSEL_HEAD + body + "\n", check_clang=False)
        except CLowerError as e:
            assert _FN_SELECTED in str(e), (body, str(e))
        else:
            raise AssertionError(f"the oracle lowered {body!r}")
    if not _CC:
        return
    fx = "cfront_fnselect.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    for rail, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        assert not re.search(r"\bu?int\d+_t t\d+ = \(t\d+ \? fs_\w+ : ", emit), (
            rail,
            "an integer select",
        )
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _FNSELECT_DRIVER)
    exe = _build_frontend(_session_build_dir())
    oracle_summary, r, _entry = _oracle(_FNSEL_CALLS)
    oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fnsel_calls.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_FNSEL_CALLS)
        c_summary, c_emit = _c_run(exe, path)
        assert c_summary == oracle_summary and "ok=1" in c_summary, (c_summary, oracle_summary)
        _parity_on_targets(path, _FNSEL_CALLS)
    driver = (
        _GAPS_SAME
        + "int main(void) {\n  for (unsigned n = 0; n < GAPS_N; n++) SAME(fsc_calls, gaps_in[n]);\n"
        + '  puts("MATCH");\n  return 0;\n}\n'
    )
    _run_against_original(
        "fnsel_calls.c", _FNSEL_CALLS, (("twin", c_emit), ("oracle", oracle_emit)), driver
    )
    with tempfile.TemporaryDirectory() as d:
        for n, body in enumerate(_FNSEL_REFUSED):
            path = os.path.join(d, f"r{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_FNSEL_HEAD + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode > 0 and _FN_SELECTED in run.stdout, (body, run.stdout[:200])
        for n, body in enumerate(_FNSEL_LOWERED):
            path = os.path.join(d, f"l{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_FNSEL_HEAD + body + "\n")
            oracle_summary, _r, _entry = _oracle(_FNSEL_HEAD + body + "\n")
            c_summary, _emit = _c_run(exe, path)
            assert c_summary == oracle_summary and "ok=1" in c_summary, (
                body,
                c_summary,
                oracle_summary,
            )


# CF-NULLCALL: the null pointer constants CF-NULLPTR and CF-NULLARG left an `int` -- compared with a pointer, passed
# through a function pointer, given to `free` and `realloc`, the arm of `?:` beside a pointer. Each emit is built
# with the pointer/integer mixes made errors: Clang's `-Wint-conversion` and `-Wpointer-integer-compare`; GCC, which
# names no option for a pointer compared with an integer (a default warning), with every warning an error.
_NULLCONST_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n], v = s + 1u;
    SAME(nc_compare, &v, s); SAME(nc_compare, 0, s); SAME(nc_member, s); SAME(nc_fnptr, s); SAME(nc_calls, s);
    SAME(nc_libc, s); SAME(nc_step, s); SAME(nc_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)
_NULLCONST_WERROR = {
    "clang": ("-Werror=int-conversion", "-Werror=pointer-integer-compare"),
    "gcc": ("-Werror=int-conversion", "-Werror"),
}


def _run_against_original_werror(fx: str, src: str, emits, driver: str, werror: dict):
    """`_run_against_original` under Clang and GCC, each with its own flags (`werror`: the compiler's name -> its
    flags) -- a warning one compiler names and the other does not made an error under both."""
    head = "#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n#include <stddef.h>\n"
    ran = 0
    with tempfile.TemporaryDirectory() as d:
        for name, flags in werror.items():
            cc = shutil.which(name)
            if not cc:
                continue
            for label, emit in emits:
                text = f"{head}{_BOUNDS_GUARD}\n{src}\n{emit}\n{driver}"
                out = _build_run_c(d, cc, label, text, flags)
                assert out == "MATCH", (
                    f"{fx}: the {label} emit is not the original under {cc} ({out})"
                )
            ran += 1
    assert ran, "no compiler of the pair"


def test_null_pointer_constants_compared_passed_and_freed_run_as_the_original():
    """CF-NULLCALL: `cfront_nullconst.c` -- the constant 0 compared with a pointer by `==` or `!=` (C11 6.5.9p5), on
    either side of a parameter, a local, a global, a loaded member or a member chain, or a function pointer; passed
    through a function pointer to a pointer parameter (a declarator, a typedef'd local, a parameter, a member by `.`
    and `->`); given to `free` and as `realloc`'s pointer; the arm of `?:` beside a pointer. Both rails lowered each
    to the same claim graph, and each emit declared the constant an `int` temp -- compared with a pointer, a
    constraint violation Clang and GCC only warn about, or passed where the callee takes a pointer, which they
    reject. Each is now declared as the pointer it converts to, on both rails, which lower the fixture to one claim
    graph on the four targets; each emit, built with those mixes made errors, returns what the original does. The
    twin now also lowers `p -= 1` for a pointer to a struct, which it had read as `p->` and refused."""
    if not _CC:
        return
    fx = "cfront_nullconst.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    _run_against_original_werror(
        fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _NULLCONST_DRIVER, _NULLCONST_WERROR
    )


# CF-STRUCTARITH: a struct or union is no operand of an operator that takes a scalar, and converts to no scalar type.
# Both rails had lowered each of these units -- to the same claim graph -- to an emit Clang rejects. (the unit, the
# reason both rails refuse it with)
_STRUCT_OPERAND = (
    "a struct or union is an operand of an arithmetic, bitwise, logical or comparison operator"
)
_STRUCT_CONVERTED = "a struct or union is converted to a scalar type"
_STRUCTARITH_HEAD = (
    "#include <stdint.h>\nstruct s { uint32_t x, y; };\nunion u { uint32_t w; uint8_t b; };\n"
    "struct h { struct s in; uint32_t z; };\nstruct s g_s;\nstruct s g_a[2];\n"
)
_STRUCTARITH_REFUSED = (
    # a compound assignment, of every operator class, of a named struct, a union, a global, an element, a member,
    # `*p`, `p[i]`; a struct value into a scalar's compound assignment; as a value
    ("uint32_t f(uint32_t v) { struct s a = {v, 2u}; a += 5; return a.x; }", _STRUCT_OPERAND),
    ("uint32_t f(uint32_t v) { struct s a = {v, 2u}; a -= 1u; return a.x; }", _STRUCT_OPERAND),
    ("uint32_t f(uint32_t v) { struct s a = {v, 2u}; a *= 2u; return a.x; }", _STRUCT_OPERAND),
    ("uint32_t f(uint32_t v) { struct s a = {v, 2u}; a |= 1u; return a.x; }", _STRUCT_OPERAND),
    ("uint32_t f(uint32_t v) { struct s a = {v, 2u}; a <<= 1u; return a.x; }", _STRUCT_OPERAND),
    ("uint32_t f(uint32_t v) { union u a = {v}; a += 5; return a.w; }", _STRUCT_OPERAND),
    ("uint32_t f(uint32_t v) { g_s += v; return g_s.x; }", _STRUCT_OPERAND),
    ("uint32_t f(uint32_t v) { g_a[1] += v; return g_a[1].x; }", _STRUCT_OPERAND),
    ("uint32_t f(struct h *p) { p->in += 1u; return p->z; }", _STRUCT_OPERAND),
    ("uint32_t f(struct h h) { h.in ^= 1u; return h.z; }", _STRUCT_OPERAND),
    ("uint32_t f(struct s *p) { *p += 1u; return p->x; }", _STRUCT_OPERAND),
    ("uint32_t f(struct s *p) { p[0] -= 1u; return p->x; }", _STRUCT_OPERAND),
    (
        "uint32_t f(uint32_t v) { struct s a = {v, 2u}; uint32_t k = v; k += a; return k; }",
        _STRUCT_OPERAND,
    ),
    (
        "uint32_t f(uint32_t v) { struct s a = {v, 2u}; uint32_t k = (a += 1u); return k; }",
        _STRUCT_OPERAND,
    ),
    # an increment or a decrement, prefix and postfix, statement and value, of a local, a union, a global
    ("uint32_t f(uint32_t v) { struct s a = {v, 2u}; a++; return a.x; }", _STRUCT_OPERAND),
    ("uint32_t f(uint32_t v) { struct s a = {v, 2u}; --a; return a.x; }", _STRUCT_OPERAND),
    (
        "uint32_t f(uint32_t v) { struct s a = {v, 2u}; uint32_t k = a++; return k; }",
        _STRUCT_OPERAND,
    ),
    ("uint32_t f(uint32_t v) { union u a = {v}; a--; return a.w; }", _STRUCT_OPERAND),
    ("uint32_t f(uint32_t v) { ++g_s; return g_s.x + v; }", _STRUCT_OPERAND),
    # an operand of an arithmetic, bitwise, shift, relational, equality, logical or unary operator
    (
        "uint32_t f(uint32_t v) { struct s a = {v, 2u}; uint32_t k = a + 1u; return k; }",
        _STRUCT_OPERAND,
    ),
    ("uint32_t f(uint32_t v) { struct s a = {v, 2u}; return v * a; }", _STRUCT_OPERAND),
    ("uint32_t f(uint32_t v) { struct s a = {v, 2u}; return a << 1; }", _STRUCT_OPERAND),
    ("uint32_t f(uint32_t v) { struct s a = {v, 2u}; return a < v; }", _STRUCT_OPERAND),
    (
        "uint32_t f(uint32_t v) { struct s a = {v, 2u}, b = {v, 3u}; return a == b; }",
        _STRUCT_OPERAND,
    ),
    ("uint32_t f(uint32_t v) { struct s a = {v, 2u}; return a && v; }", _STRUCT_OPERAND),
    ("uint32_t f(uint32_t v) { struct s a = {v, 2u}; return v || a; }", _STRUCT_OPERAND),
    ("uint32_t f(uint32_t v) { struct s a = {v, 2u}; return -a; }", _STRUCT_OPERAND),
    ("uint32_t f(uint32_t v) { struct s a = {v, 2u}; return ~a; }", _STRUCT_OPERAND),
    ("uint32_t f(uint32_t v) { struct s a = {v, 2u}; return !a; }", _STRUCT_OPERAND),
    # converted to an integer, a float or a pointer: an initializer (braced too), an assignment to a local, a
    # member, an element or `*q`, a list element, a `return`, a cast, an argument
    (
        "uint32_t f(uint32_t v) { struct s a = {v, 2u}; uint32_t k = a; return k; }",
        _STRUCT_CONVERTED,
    ),
    (
        "uint32_t f(uint32_t v) { struct s a = {v, 2u}; uint64_t k = {a}; return (uint32_t)k; }",
        _STRUCT_CONVERTED,
    ),
    (
        "uint32_t f(uint32_t v) { union u a = {v}; double k = a; return (uint32_t)k; }",
        _STRUCT_CONVERTED,
    ),
    (
        "uint32_t f(uint32_t v) { struct s a = {v, 2u}; uint32_t *k = a; return *k; }",
        _STRUCT_CONVERTED,
    ),
    ("uint32_t f(struct s *p) { uint32_t k = *p; return k; }", _STRUCT_CONVERTED),
    ("uint32_t f(struct h *p) { uint32_t k = p->in; return k; }", _STRUCT_CONVERTED),
    ("uint32_t f(uint32_t v) { uint32_t k = g_a[1]; return k + v; }", _STRUCT_CONVERTED),
    (
        "uint32_t f(uint32_t v) { struct s a = {v, 2u}; uint32_t k = 0u; k = a; return k; }",
        _STRUCT_CONVERTED,
    ),
    (
        "struct w { uint32_t m; };\n"
        "uint32_t f(uint32_t v) { struct s a = {v, 2u}; struct w b; b.m = a; return b.m; }",
        _STRUCT_CONVERTED,
    ),
    (
        "uint32_t f(uint32_t *q, uint32_t v) { struct s a = {v, 2u}; *q = a; return *q; }",
        _STRUCT_CONVERTED,
    ),
    (
        "uint32_t f(uint32_t v) { struct s a = {v, 2u}; uint32_t k[2]; k[0] = a; return k[0]; }",
        _STRUCT_CONVERTED,
    ),
    (
        "uint32_t f(uint32_t v) { struct s a = {v, 2u}; uint32_t k[2] = {1u, a}; return k[0]; }",
        _STRUCT_CONVERTED,
    ),
    ("uint32_t f(uint32_t v) { struct s a = {v, 2u}; return a; }", _STRUCT_CONVERTED),
    ("uint64_t *f(uint32_t v) { union u a = {v}; return a; }", _STRUCT_CONVERTED),
    ("uint32_t f(uint32_t v) { struct s a = {v, 2u}; return (uint32_t)a; }", _STRUCT_CONVERTED),
    (
        "static uint32_t g(uint32_t x) { return x; }\n"
        "uint32_t f(uint32_t v) { struct s a = {v, 2u}; return g(a); }",
        _STRUCT_CONVERTED,
    ),
    (
        "uint32_t h2(uint32_t x);\nuint32_t f(uint32_t v) { struct s a = {v, 2u}; return h2(a); }\n"
        "uint32_t h2(uint32_t x) { return x; }",
        _STRUCT_CONVERTED,
    ),
    (
        "typedef uint32_t (*op_t)(uint32_t);\nstatic uint32_t g(uint32_t x) { return x; }\n"
        "uint32_t f(uint32_t v) { struct s a = {v, 2u}; op_t p = g; return p(a); }",
        _STRUCT_CONVERTED,
    ),
)
# ... while a struct copied whole, its members in arithmetic, a pointer to structs stepped and compared, and an array
# of structs decayed still lower to one claim graph on both rails
_STRUCTARITH_LOWERED = (
    "uint32_t f(uint32_t v) { struct s a = {v, 2u}, b = a; b.x += a.y; b.y++; return b.x + b.y; }",
    "uint32_t f(struct s *p, uint32_t v) { struct s *q = p + 1; q -= 1; p->x = v; return q == p; }",
    "uint32_t f(uint32_t v) { struct s *p = g_a; p += 1; return (p - g_a) + v + (g_a != 0); }",
    "static struct s mk(uint32_t v) { struct s r = {v, 1u}; return r; }\n"
    "uint32_t f(uint32_t v) { struct s a = mk(v); return a.x * 3u + mk(v + 1u).y; }",
)


def test_struct_arithmetic_and_conversion_are_refused_on_both_rails():
    """CF-STRUCTARITH: every unit of `_STRUCTARITH_REFUSED` -- a compound assignment of a struct or union (named, a
    global, an element, a member, `*p`, `p[i]`) or of a struct into a scalar, an increment or decrement of one, a
    struct operand of an arithmetic, bitwise, shift, relational, equality, logical or unary operator, a struct
    converted to an integer, a float or a pointer (an initializer, an assignment, a list element, a `return`, a cast,
    an argument to a function defined before or after its caller or called through a pointer) -- is refused on both
    rails with the reason it witnesses: both had lowered it to an emit Clang rejects. Every unit of
    `_STRUCTARITH_LOWERED` still lowers to the same claim graph on both."""
    from bcir.frontends.cfront.cparse import CParseError
    from bcir.frontends.cfront.lower import CLowerError

    summaries = {}
    for body in _STRUCTARITH_LOWERED:
        summaries[body], _r, _e = _oracle(_STRUCTARITH_HEAD + body + "\n")
        assert "ok=1" in summaries[body], (body, summaries[body])
    for body, why in _STRUCTARITH_REFUSED:
        try:
            compile_unit(_STRUCTARITH_HEAD + body + "\n", check_clang=False)
        except (CLowerError, CParseError) as e:
            assert why in str(e), (body, str(e))
        else:
            raise AssertionError(f"the oracle lowered {body!r}")
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, body in enumerate(_STRUCTARITH_LOWERED):
            path = os.path.join(d, f"l{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_STRUCTARITH_HEAD + body + "\n")
            c_summary, _emit = _c_run(exe, path)
            assert c_summary == summaries[body], (body, c_summary, summaries[body])
        for n, (body, why) in enumerate(_STRUCTARITH_REFUSED):
            path = os.path.join(d, f"r{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_STRUCTARITH_HEAD + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode > 0 and why in run.stdout, (
                body,
                run.returncode,
                run.stdout[:200],
            )


# CF-ENUMFOLD: an enumerator, a case label and an array dimension are integer constant expressions, folded where they
# are parsed, in C's own types and the target's data model, by the predicate a static's initializer folds with. The
# oracle had folded them with unbounded integers (`-7 / 2` was -4, `-7 % 2` 1) and the twin in `long long`, and
# neither compared an unsigned int operand as unsigned (`~0u > 5` was 0). The twin had spelled a negative enumerator
# as its 64-bit two's complement (`int32_t t = 18446744073709551613;`) and the oracle as an unsigned int (`-3u`); the
# oracle had spelled a case label past LLONG_MAX without `u`, which Clang and GCC read back with a warning.
_ENUMFOLD_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(ef_values, s); SAME(ef_signed, (int32_t)s); SAME(ef_uswitch, s); SAME(ef_dims, s); SAME(ef_entry, s);
    SAME(ef_wswitch, (uint64_t)s << 31); SAME(ef_wswitch, (uint64_t)s * 0x9E3779B97F4A7C15u);
  }
  for (int32_t v = -80; v < 80; v++) SAME(ef_switch, v);
  SAME(ef_switch, INT32_MIN); SAME(ef_switch, INT32_MAX); SAME(ef_signed, INT32_MIN); SAME(ef_signed, INT32_MAX);
  SAME(ef_uswitch, 0xFFFFFFFFu); SAME(ef_uswitch, 0xFFFFFFFDu); SAME(ef_uswitch, 2147483644u);
  SAME(ef_wswitch, 0xFFFFFFFFFFFFFFFFu); SAME(ef_wswitch, 0x8000000000000000u);
  SAME(ef_wswitch, 0xFFFFFFFFFFFFFFFEu); SAME(ef_wswitch, 0xFFFFFFFF80000000u);
  puts("MATCH");
  return 0;
}
"""
)
# a constant C converts back, which Clang and GCC warn about, made an error: an integer constant past LLONG_MAX
# spelled without `u` (the twin's negative enumerator, the oracle's case label)
_ENUMFOLD_WERROR = {"clang": ("-Werror=implicitly-unsigned-literal",), "gcc": ("-Werror",)}


def _assert_signed_spellings(rail: str, emit: str) -> None:
    """A negative constant is spelled as one, signed, and a case label past LLONG_MAX with `u`."""
    assert re.search(r" = -3;", emit) and re.search(r" = -2147483648;", emit), rail
    assert re.search(r"\bcase -3:", emit) and "case 18446744073709551615u:" in emit, rail
    assert not re.search(r" = -\d+u;", emit), (rail, re.findall(r".* = -\d+u;", emit)[:3])
    assert not re.search(r"[ (]1844674407370955\d{4}[;:]", emit), (
        rail,
        "a 64-bit two's complement",
    )


def test_enumerators_case_labels_and_dimensions_fold_in_c_types_on_both_rails():
    """CF-ENUMFOLD: `cfront_enumfold.c` -- enumerators that divide and take the remainder of negatives, compare an
    unsigned int operand, a long long and a type narrower than int, shift in every direction and width, cast to
    every width and to `_Bool`, select through nested `?:` in the arms' common type and leave an operand C does not
    evaluate unfolded (`0 && 1 / 0`), count on from INT_MIN and to INT_MAX; case labels of a signed, an unsigned and
    a 64-bit switch from the same expressions; enumerators as the dimensions of a local, a member, a typedef and a
    global, and as designators -- lowers to one claim graph on the four targets, and each function of each emit,
    built with a constant C converts back made an error, returns what the original does under Clang and GCC. Both
    emits spell a negative enumerator as a signed constant and a case label past LLONG_MAX with `u`."""
    if not _CC:
        return
    fx = "cfront_enumfold.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    for rail, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        _assert_signed_spellings(rail, emit)
        assert "__bcir_ext" not in emit, f"{rail}: a dimension that is a constant made a VLA"
    _run_against_original_werror(
        fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _ENUMFOLD_DRIVER, _ENUMFOLD_WERROR
    )


# The unit the CF-GLOBALS triage found: the oracle folded its enumerators to -4, 1 and 0 and its case label to -4, the
# twin to -3, -1 and 0, and C gives -3, -1 and 1.
_ENUMFOLD_FOUND = """#include <stdint.h>
enum { ED_Q = -7 / 2, ED_R = -7 % 2, ED_N = ~0u > 5 };
int32_t ed_found(int32_t v)
{
    switch (v) {
    case -7 / 2:
        return 1000;
    default:
        return ED_Q * 100 + ED_R * 10 + ED_N + v;
    }
}
"""


def test_the_enumerators_the_triage_found_fold_as_c_does_on_both_rails():
    """CF-ENUMFOLD: `enum { ED_Q = -7 / 2, ED_R = -7 % 2, ED_N = ~0u > 5 };` and `case -7 / 2:` -- -3, -1, 1 and -3
    in C -- lower to one claim graph on the four targets, both emits spell -3 signed, and each returns what the
    original does for every value the switch tells apart."""
    oracle_summary, r, _entry = _oracle(_ENUMFOLD_FOUND)
    oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    assert re.search(r"\bcase -3:", oracle_emit) and re.search(r" = -3;", oracle_emit), oracle_emit
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "found.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_ENUMFOLD_FOUND)
        c_summary, c_emit = _c_run(exe, path)
        assert c_summary == oracle_summary and "ok=1" in c_summary, (c_summary, oracle_summary)
        _parity_on_targets(path, _ENUMFOLD_FOUND)
    assert re.search(r"\bcase -3:", c_emit) and re.search(r" = -3;", c_emit), c_emit
    driver = (
        _GAPS_SAME
        + "int main(void) {\n  for (int32_t v = -9; v < 9; v++) SAME(ed_found, v);\n"
        + '  puts("MATCH");\n  return 0;\n}\n'
    )
    _run_against_original_werror(
        "found.c",
        _ENUMFOLD_FOUND,
        (("twin", c_emit), ("oracle", oracle_emit)),
        driver,
        _ENUMFOLD_WERROR,
    )


def _enumerator_names(src: str) -> list:
    """Every enumerator the `enum` definitions of `src` declare, in order (its comments dropped first)."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    names = []
    for body in re.findall(r"\benum\b[^{;]*\{(.*?)\}", src, re.S):
        for part in body.split(","):
            m = re.match(r"\s*([A-Za-z_]\w*)", part)
            if m:
                names.append(m.group(1))
    return names


# the four targets, by their Clang triples
_ENUMFOLD_TRIPLES = {
    "x86_64-linux": "x86_64-unknown-linux-gnu",
    "aarch64-linux": "aarch64-unknown-linux-gnu",
    "x86_64-windows": "x86_64-pc-windows-msvc",
    "i386-linux": "i386-unknown-linux-gnu",
}


def test_every_enumerator_folds_to_clangs_value_on_each_target():
    """CF-ENUMFOLD: each enumerator of `cfront_enumfold.c`, read back through a function returning it, is the
    constant the oracle folds on each of the four targets, which Clang, compiling the unit for that target, holds
    it to (`_Static_assert`) -- a comparison of `-1L` with `1u`, a `long` of `0xFFFFFFFFL` and a cast to `unsigned
    long` are 1 where `long` is 64 bits and 0 where it is 32, while `2147483648` and `0xFFFFFFFFL`, a `long` or
    the next type of their lists, are positive on all four -- and the twin folds to the same claim graph."""
    clang = shutil.which("clang")
    if not clang:
        return
    src = open(os.path.join(_C, "cfront_enumfold.c"), encoding="utf-8").read()
    names = _enumerator_names(src)
    assert len(names) == len(set(names)) == 51, names
    unit = src + "".join(f"int64_t ev_{n}(void) {{ return {n}; }}\n" for n in names)
    with tempfile.TemporaryDirectory() as d:
        for target, triple in _ENUMFOLD_TRIPLES.items():
            r = compile_unit(unit, check_clang=False, target=target)
            values = {}
            for n in names:
                (k,) = [c for c in r.lowered.functions[f"ev_{n}"].claims if c.op == "c.const"]
                values[n] = int(k.imm[0])
            checks = "".join(f'_Static_assert({n} == {v}LL, "{n}");\n' for n, v in values.items())
            path = os.path.join(d, f"values_{target}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src + checks)
            run = subprocess.run(
                [clang, f"--target={triple}", "-ffreestanding", "-std=c11", "-fsyntax-only", path],
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, (
                f"{target}: the oracle's values are not Clang's\n{run.stderr[:2000]}"
            )
        if not _CC:
            return
        path = os.path.join(d, "getters.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(unit)
        _parity_on_targets(path, unit)


# Where C requires a diagnostic -- an enumerator value no int holds (C11 6.7.2.2p2), a division or a remainder by
# zero, a signed overflow, a shift past the width or of a negative value (6.5.7p4, 6.6p4), an array dimension that
# is negative (6.7.6.2p1) -- or the expression is not an integer constant expression (6.6p6), both rails refuse for
# one reason, never picking a value.
_ICE_NOT = "not an integer constant expression"
_ENUM_NOT_INT = "an enumerator value not representable as int"
_DIM_RANGE = "an array dimension outside 0..INT_MAX"
_ENUMFOLD_HEAD = "#include <stdint.h>\nuint32_t g_x;\n"
_ENUMFOLD_REFUSED = (
    ("enum { N = 0x80000000 };", _ENUM_NOT_INT),
    ("enum { N = 0x100000000 };", _ENUM_NOT_INT),
    ("enum { N = 0xFFFFFFFFFFFFFFFFu };", _ENUM_NOT_INT),
    ("enum { N = -2147483649LL };", _ENUM_NOT_INT),
    ("enum { N = 2147483647u + 1u };", _ENUM_NOT_INT),
    ("enum { A = 2147483647, B };", _ENUM_NOT_INT),
    ("enum { N = 1 / 0 };", _ICE_NOT),
    ("enum { N = 1 % 0 };", _ICE_NOT),
    ("enum { N = 2147483647 + 1 };", _ICE_NOT),
    ("enum { N = 65536 * 65536 };", _ICE_NOT),
    ("enum { N = -(-2147483647 - 1) };", _ICE_NOT),
    ("enum { N = (-2147483647 - 1) / -1 };", _ICE_NOT),
    ("enum { N = (-2147483647 - 1) % -1 };", _ICE_NOT),
    ("enum { N = 1 << 31 };", _ICE_NOT),
    ("enum { N = 1 << 32 };", _ICE_NOT),
    ("enum { N = 1u << 32 };", _ICE_NOT),
    ("enum { N = -1 << 1 };", _ICE_NOT),
    ("enum { N = 8 >> -1 };", _ICE_NOT),
    ("enum { N = g_x };", _ICE_NOT),
    ("enum { N = 0 && g_x };", _ICE_NOT),
    ("enum { N = sizeof g_x };", _ICE_NOT),
    ("enum { N = (1, 2) };", _ICE_NOT),
    ("enum { N = (int)1.5L };", _ICE_NOT),
    ("enum { N = (uint8_t *)0 == 0 };", _ICE_NOT),
    ("int32_t f(int32_t v) { switch (v) { case 1 / 0: return 1; default: return 0; } }", _ICE_NOT),
    (
        "int32_t f(int32_t v) { switch (v) { case 2147483647 + 1: return 1; default: return 0; } }",
        _ICE_NOT,
    ),
    (
        "int32_t f(int32_t v) { switch (v) { case 1 << 31: return 1; default: return 0; } }",
        _ICE_NOT,
    ),
    ("int32_t f(int32_t v) { switch (v) { case v: return 1; default: return 0; } }", _ICE_NOT),
    ("uint32_t f(uint32_t s) { uint32_t a[4] = {[1 / 0] = s}; return a[0]; }", _ICE_NOT),
    ("uint32_t f(uint32_t s) { uint32_t a[-1]; a[0] = s; return a[0]; }", _DIM_RANGE),
    ("struct r { uint8_t b[1 - 3]; uint32_t k; };", _DIM_RANGE),
    ("uint8_t g_big[0x80000000];", _DIM_RANGE),
)
# ... and a dimension that is an integer constant expression makes a fixed array (C11 6.7.6.2p4) on both rails --
# no runtime extent -- while one that is not makes a VLA, as before
_ENUMFOLD_DIMS = (
    (
        "uint32_t f(uint32_t s) { uint32_t a[2 + 1]; a[2] = s; return a[2] + (uint32_t)sizeof a; }",
        False,
    ),
    (
        "enum { N = 3 };\nuint32_t f(uint32_t s) { uint8_t a[N * 2]; a[5] = (uint8_t)s; return a[5] + "
        "(uint32_t)sizeof a; }",
        False,
    ),
    (
        "enum { N = 3 };\nstruct r { uint8_t b[N + 1]; uint32_t k; };\nuint32_t g[N << 1];\n"
        "uint32_t f(uint32_t s) { struct r v; v.b[N] = (uint8_t)s; v.k = s; g[5] = s; "
        "return v.b[N] + v.k + g[5] + (uint32_t)sizeof v + (uint32_t)sizeof g; }",
        False,
    ),
    ("enum { N = 3 };\nuint32_t f(uint32_t a[N], uint32_t s) { return a[N - 1] + s; }", False),
    (
        "uint32_t f(uint32_t s) { uint32_t n = (s & 3u) + 1u; uint32_t a[n + 1u]; a[0] = s; "
        "return a[0] + (uint32_t)sizeof a; }",
        True,
    ),
)


def test_constant_expressions_c_refuses_are_refused_and_constant_dimensions_fixed_on_both_rails():
    """CF-ENUMFOLD: every unit of `_ENUMFOLD_REFUSED` -- an enumerator past int's range (stated, an unsigned one
    past LLONG_MAX among them, or counted on from INT_MAX), a division or a remainder by zero, a signed overflow
    (`+`, `*`, `-`, `/` and `%` of INT_MIN by -1), a shift by the width or more, by a negative count or of a
    negative value, an object, `sizeof` of an expression, the comma operator, a `long double` constant and a pointer
    in an enumerator, the same in a case label, a designator that divides by zero, and the constant dimension of a local, a member or a
    global outside 0..INT_MAX -- is refused on both rails for the one reason it witnesses, where both had picked a
    value or refused for reasons of their own. Every unit of `_ENUMFOLD_DIMS` lowers to one claim graph on both rails, a
    dimension that is an integer constant expression -- `2 + 1`, `N * 2` of a local, `N + 1` of a member, `N << 1`
    of a global, `N` of a parameter -- a fixed array, a runtime one a VLA."""
    from bcir.frontends.cfront.cparse import CParseError

    for body, why in _ENUMFOLD_REFUSED:
        try:
            compile_unit(_ENUMFOLD_HEAD + body + "\n", check_clang=False)
        except CParseError as e:
            assert str(e) == why, (body, str(e))
        else:
            raise AssertionError(f"the oracle lowered {body!r}")
    summaries = {}
    for body, vla in _ENUMFOLD_DIMS:
        summaries[body], r, _e = _oracle(_ENUMFOLD_HEAD + body + "\n")
        emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
        assert ("__bcir_ext" in emit) == vla, (body, emit)
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, (body, why) in enumerate(_ENUMFOLD_REFUSED):
            path = os.path.join(d, f"r{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_ENUMFOLD_HEAD + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode == 1 and run.stdout.strip() == f"PARSE-ERR {why}", (
                body,
                run.returncode,
                run.stdout[:200],
            )
        for n, (body, vla) in enumerate(_ENUMFOLD_DIMS):
            path = os.path.join(d, f"d{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_ENUMFOLD_HEAD + body + "\n")
            c_summary, c_emit = _c_run(exe, path)
            assert c_summary == summaries[body] and "ok=1" in c_summary, (
                body,
                c_summary,
                summaries[body],
            )
            assert ("__bcir_ext" in c_emit) == vla, (body, c_emit)


# CF-ENUMSCOPE: a name a block declares -- a local, a parameter, a loop's own declaration -- hides a file-scope
# enumerator of its name (C11 6.2.1p4) from the end of its declarator (6.2.1p7) to the end of its block. Both rails
# read an enumerator first, wherever its name stood: `uint32_t N = (s & 1u) + 1u; uint32_t a[N];` beside an
# `enum { N = 4 }` became `a[4]` and `+ 4` -- a silent miscompile, digest-equal on both rails. Each rail now reads an
# enumerator through one visibility rule (the oracle's `_enumerator`, the twin's `visible_enum`).
_ENUMSCOPE_DRIVER = r"""
static int fail(const char *what) { puts(what); return 1; }
#define SAME(f, ...) do { if (f(__VA_ARGS__) != bcir_##f(__VA_ARGS__)) return fail(#f); } while (0)
int main(void) {
  for (uint32_t i = 0; i < 40u; i++) {
    SAME(es_local, i); SAME(es_param, i, i * 3u + 1u); SAME(es_block, i); SAME(es_loop, i); SAME(es_self, i);
    SAME(es_sizeof, i); SAME(es_pn, i); SAME(es_after, i); SAME(enumscope, i, i ^ 9u);
  }
  /* the values C gives -- reading the enumerator in place of the object gives 21, 21, 6, 4 and 12 */
  if (bcir_es_local(1u) != 11u || bcir_es_block(0u) != 13u || bcir_es_loop(3u) != 6u || bcir_es_self(0u) != 8u ||
      bcir_es_sizeof(0u) != 11u || bcir_es_after(4u) != 12u)
    return fail("values");
  puts("MATCH");
  return 0;
}
"""

# A case label naming a local that hides an enumerator is no integer constant expression (6.8.4.2p3, 6.6p6).
_ENUMSCOPE_CASE = (
    "#include <stdint.h>\nenum { N = 4 };\n"
    "uint32_t f(uint32_t s) { uint32_t N = s; switch (s) { case N: return 1u; default: return 0u; } }\n"
)
# The two readers of a name besides an expression's, each a unit whose claim graph names what it read: a fence's
# order (an acquire enumerator would make an acquire fence; the local makes a full one) and a volatile access at a
# byte offset (an enumerator offset is folded into the access; the local's is an address computed at run time).
_ENUMSCOPE_READERS = (
    "#include <stdint.h>\n#include <stdatomic.h>\nenum { MO = 2 };\n"
    "uint32_t f(uint32_t *p, uint32_t s) { int MO = (int)(s & 1u) * 2; *p = s; atomic_thread_fence(MO); "
    "return *p; }\n",
    "#include <stdint.h>\nstruct dev { uint32_t a, b, c; };\nenum { K = 4 };\n"
    "uint32_t f(volatile struct dev *d, uint32_t s) { uint32_t K = (s & 1u) * 8u; "
    "return *(volatile uint32_t *)((volatile char *)d + K); }\n",
)


def test_a_block_scope_name_hides_an_enumerator_on_both_rails():
    """CF-ENUMSCOPE: `cfront_enumscope.c` -- a local hiding an enumerator as a VLA's extent and its `sizeof`, a
    parameter, an inner block's local and a loop's declaration whose scope ends, a local's own initializer
    (`uint64_t N = sizeof N;` is 8), a local array under `sizeof`, and file scope reading the enumerator again after a
    function whose parameter hid it (an enumerator's value, a table's dimension) -- lowers to one claim graph on both
    rails, and each emit returns what the original does. On the parent both rails refused the unit, and the one
    function they took, `es_local`, both lowered with `a[4]` and `+ 4`. A case label naming the local is refused on
    both rails as no integer constant expression."""
    from bcir.frontends.cfront.cparse import CParseError
    from bcir.frontends.cfront.lower import ICE_NOT

    try:
        compile_unit(_ENUMSCOPE_CASE, check_clang=False)
    except CParseError as e:
        assert str(e) == ICE_NOT, str(e)
    else:
        raise AssertionError("the oracle took a case label naming a local")
    if not _CC:
        return
    fx = "cfront_enumscope.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _ENUMSCOPE_DRIVER)
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "case.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_ENUMSCOPE_CASE)
        run = subprocess.run([exe, path], capture_output=True, text=True)
        out = run.stdout.strip()
        assert run.returncode == 1 and out == f"PARSE-ERR {ICE_NOT}", out[:200]
        for n, unit in enumerate(_ENUMSCOPE_READERS):
            path = os.path.join(d, f"reader{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(unit)
            _parity_on_targets(path, unit)


# CF-CONSTEXPR2: the constant expressions CF-ENUMFOLD left. `cfront_constexpr2.c` holds what the 64-bit Linux hosts
# give alike; what the target's character types and data model decide is `_CONSTEXPR2_TARGET`'s, held to Clang's
# value on each target.
_CONSTEXPR2_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(ce_switch8, (uint8_t)s); SAME(ce_switch8, (uint8_t)(s + 'a')); SAME(ce_switch8, (uint8_t)'b');
    SAME(ce_switch32, s); SAME(ce_switch32, 0xFFFFFFFFu); SAME(ce_switch32, 0xFFFFFFFEu);
    SAME(ce_switch32, (uint32_t)sizeof(struct ce_pair)); SAME(ce_switch32, 7u);
    SAME(ce_chars, s); SAME(ce_block, s); SAME(ce_bitfields, s); SAME(ce_literals, s); SAME(constexpr2, s);
    /* each header in storage of its own with room for two elements, written through the member itself */
    struct ce_zhdr *z = malloc(sizeof *z + 2 * sizeof(uint32_t));
    struct ce_fhdr *f = malloc(sizeof *f + 2 * sizeof(uint64_t));
    if (!z || !f) return fail("malloc");
    z->n = s & 255u; z->tail[0] = s >> 8; z->tail[1] = s ^ 0x55u;
    f->c = (uint8_t)(s & 0x7Fu); f->tail[0] = (uint64_t)s * 3u; f->tail[1] = (uint64_t)s + 11u;
    uint32_t got = ce_zero(z, f, s), want = bcir_ce_zero(z, f, s);
    free(z); free(f);
    if (got != want) return fail("ce_zero");
  }
  /* the values C gives: a zero-length or flexible member occupies no bytes, each width truncates its field */
  if (sizeof(struct ce_zhdr) != 4u || sizeof(struct ce_fhdr) != 8u || bcir_ce_switch8(255u) != 1u ||
      bcir_ce_switch32(0xFFFFFFFEu) != 20u || bcir_ce_bitfields(0u) != (uint32_t)sizeof(struct ce_bits) * 100000u)
    return fail("values");
  puts("MATCH");
  return 0;
}
"""
)
_CONSTEXPR2_WERROR = {"clang": ("-Wno-multichar",), "gcc": ("-Wno-multichar",)}


def test_constant_expressions_c_takes_lower_and_run_as_the_original_on_both_rails():
    """CF-CONSTEXPR2: `cfront_constexpr2.c` -- switches whose labels differ only in the promoted controlling type, prefixed
    and multi-character constants in their own types, an enumeration declared in a block (its constants and tag scoped
    to it, hiding and hidden in turn), enumerators and `sizeof` as bit-field widths, a compound literal's and a row
    pointer's dimension, `sizeof`/`_Alignof` of type-names and floating constants cast to integer types in
    enumerators, zero-length and flexible array members -- lowers to one claim graph on the four targets, and each
    function of each emit returns what the original does under Clang and GCC. On the parent both rails refused the
    unit, and the twin laid a `uint32_t t[0]` member out as one element (`sizeof` 8 for Clang's 4) and an enumerator
    width as a member as wide as its type."""
    if not _CC:
        return
    fx = "cfront_constexpr2.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    for rail, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        assert "__bcir_ext" not in emit, f"{rail}: a dimension that is a constant made a VLA"
        assert re.search(r"\b(?:uint8_t|unsigned char) \w+ = 255u;", emit), (
            f"{rail}: u8'\\xff' no unsigned char"
        )
        assert re.search(r"\b(?:uint16_t|unsigned short) \w+ = 98u;", emit), (
            f"{rail}: u'b' is no char16_t"
        )
    _run_against_original_werror(
        fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _CONSTEXPR2_DRIVER, _CONSTEXPR2_WERROR
    )


# Each enumerator of `_CONSTEXPR2_TARGET` is the value Clang gives it on each target: a plain character constant by the
# target's `char` (`'\xff'` 255 on AArch64, -1 elsewhere), an `L` one by its `wchar_t` (unsigned on AArch64 and
# Windows, 16 bits on Windows), the sizes and alignments of its types, a `_BitInt` an enumerator sizes.
_CONSTEXPR2_TARGET = r"""#include <stdint.h>
#include <stddef.h>
enum { W = 12 };
struct ct_z { uint8_t c; uint64_t t[0]; };
struct ct_f { uint16_t n; double t[]; };
enum ct_char { CT_HI = '\xff', CT_OCT = '\200', CT_CMP = 'a' - 98 < 0, CT_LCMP = L'a' - 98 < 0, CT_LHI = L'\xffff',
               CT_SZL = sizeof(L'a'), CT_SZU8 = sizeof(u8'a'), CT_SZU = sizeof(u'a'), CT_SZU32 = sizeof(U'a'),
               CT_SZC = sizeof('a'), CT_MULTI = '\377a', CT_PLUS = '\xff' + 1, CT_SH = '\x80' >> 1 };
enum ct_size { CT_LD = sizeof(long double), CT_ALD = _Alignof(long double), CT_L = sizeof(long),
               CT_AD = _Alignof(double), CT_ALL = _Alignof(long long), CT_Z = sizeof(struct ct_z),
               CT_F = sizeof(struct ct_f), CT_AZ = _Alignof(struct ct_z), CT_BI = sizeof(_BitInt(W)),
               CT_BI2 = _Alignof(unsigned _BitInt(W * 4)), CT_PTR = sizeof(void *), CT_WCHAR = sizeof(wchar_t) };
enum ct_float { CT_FL = (int)16777217.0f, CT_DB = (int)16777217.0, CT_HX = (int)0x1.fffffep23f, CT_TINY = (_Bool)1e-50f,
                CT_TINYD = (_Bool)1e-50, CT_ROUND = (int)0.99999999999999999999, CT_BIG = (unsigned)4294967295.5 == 4294967295u,
                CT_SZD = sizeof 1.0L, CT_SZF = sizeof(1.5f), CT_DEAD = 0 && (int)1e10, CT_DEAD2 = 1 || (unsigned char)300.5,
                CT_DEAD3 = 1 ? 2 : (int)1e10 };
"""


def test_character_constants_sizes_and_float_casts_fold_to_clangs_values_on_each_target():
    """CF-CONSTEXPR2: each enumerator of `_CONSTEXPR2_TARGET` -- character constants plain, octal, multi-character and
    prefixed, their `sizeof`, `sizeof`/`_Alignof` of `long double`, `long`, a zero-length and a flexible member's struct,
    a `_BitInt` an enumerator sizes, and floating constants cast to integer types (`(int)16777217.0f` 16777216 after the
    `float` rounds it, `(_Bool)1e-50f` 0 where `1e-50` is 1; one C does not evaluate, `0 && (int)1e10`, no refusal) --
    is the constant the oracle folds on each of the four
    targets, which Clang, compiling the unit for that target, holds it to (`_Static_assert`), and the twin folds the
    same claim graph. Both rails had read every character constant as an `int` of signed bytes, `'\\xff'` -1 on
    AArch64, and refused `sizeof` and a cast of a floating constant in an enumerator."""
    clang = shutil.which("clang")
    if not clang:
        return
    names = _enumerator_names(_CONSTEXPR2_TARGET)
    assert len(names) == len(set(names)) == 38, names
    unit = _CONSTEXPR2_TARGET + "".join(f"int64_t ev_{n}(void) {{ return {n}; }}\n" for n in names)
    seen = set()
    with tempfile.TemporaryDirectory() as d:
        for target, triple in _ENUMFOLD_TRIPLES.items():
            r = compile_unit(unit, check_clang=False, target=target)
            values = {}
            for n in names:
                (k,) = [c for c in r.lowered.functions[f"ev_{n}"].claims if c.op == "c.const"]
                values[n] = int(k.imm[0])
            seen.add((values["CT_HI"], values["CT_LCMP"], values["CT_SZL"]))
            checks = "".join(f'_Static_assert({n} == {v}LL, "{n}");\n' for n, v in values.items())
            path = os.path.join(d, f"values_{target}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_CONSTEXPR2_TARGET + checks)
            run = subprocess.run(
                [
                    clang,
                    f"--target={triple}",
                    "-ffreestanding",
                    "-std=c2x",
                    "-Wno-multichar",
                    "-fsyntax-only",
                    path,
                ],
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, (
                f"{target}: the oracle's values are not Clang's\n{run.stderr[:2000]}"
            )
        assert len(seen) == 3, seen  # the targets tell `char` and `wchar_t` apart
        if not _CC:
            return
        path = os.path.join(d, "getters.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(unit)
        _parity_on_targets(path, unit)


# A wide character constant is a code unit of its target's `wchar_t`: `L'\xffffffff'` is -1 where that is a signed
# 32-bit type (x86-64 and i386 Linux), 4294967295 where it is unsigned (AArch64), and no constant where it holds 16 bits
# (Windows), whose code unit the escape is past -- refused there on both rails, as Clang refuses it. It is read so in
# an enumerator, which wraps it to its type, and as a value, which only the reader's sign gives (`cw_val`, `cw_half`).
_CONSTEXPR2_WIDE = (
    "#include <stdint.h>\nenum { CW_NEG = L'\\xffffffff' < 0, CW_TOP = L'\\xffffffff' > 0x7fffffff };\n"
    "int64_t cw_neg(void) { return CW_NEG; }\nint64_t cw_top(void) { return CW_TOP; }\n"
    "int64_t cw_val(void) { return L'\\xffffffff'; }\nint64_t cw_half(void) { return (int64_t)L'\\xfffffffe' / 2; }\n"
)
#: Each function of `_CONSTEXPR2_WIDE` whose first constant is the one Clang must hold, and the expression it holds.
_CONSTEXPR2_WIDE_HELD = {
    "cw_neg": "CW_NEG",
    "cw_top": "CW_TOP",
    "cw_val": "(int64_t)L'\\xffffffff'",
    "cw_half": "(int64_t)L'\\xfffffffe'",
}


def test_a_wide_character_constant_is_a_code_unit_of_its_targets_wchar_t_on_both_rails():
    """CF-CONSTEXPR2: `_CONSTEXPR2_WIDE` folds, on each of the four targets, to what Clang holds it to compiling the unit
    for that target (`_Static_assert`) and to one claim graph on both rails -- or, on Windows, is refused by Clang and
    by both rails for one reason. The parent read `L'\\xffffffff'` as an `int` of signed bytes on every target."""
    from bcir.frontends.cfront.clex import CLexError

    clang = shutil.which("clang")
    exe = _build_frontend(_session_build_dir()) if _CC else None
    refused = set()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "wide.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_CONSTEXPR2_WIDE)
        for target, triple in _ENUMFOLD_TRIPLES.items():
            try:
                r = compile_unit(_CONSTEXPR2_WIDE, check_clang=False, target=target)
            except CLexError as e:
                assert str(e) == _CHAR_BAD, (target, str(e))
                refused.add(target)
                oracle = f"PARSE-ERR {e}"
                checks = ""
            else:
                checks = ""
                for fn, held in _CONSTEXPR2_WIDE_HELD.items():
                    k = next(c for c in r.lowered.functions[fn].claims if c.op == "c.const")
                    checks += f'_Static_assert({held} == {int(k.imm[0])}, "{fn}");\n'
                oracle = _summary_line(r)
            if clang:
                cpath = os.path.join(d, f"wide_{target}.c")
                with open(cpath, "w", encoding="utf-8") as fh:
                    fh.write(_CONSTEXPR2_WIDE + checks)
                run = subprocess.run(
                    [
                        clang,
                        f"--target={triple}",
                        "-ffreestanding",
                        "-std=c2x",
                        "-fsyntax-only",
                        cpath,
                    ],
                    capture_output=True,
                    text=True,
                )
                assert (run.returncode != 0) == (target in refused), (target, run.stderr[:2000])
            if exe:
                run = subprocess.run(
                    [exe, "--target", target, path], capture_output=True, text=True
                )
                twin = (run.stdout.partition("----EMIT----\n")[0].strip().splitlines() or [""])[0]
                assert twin == oracle, f"on {target}\n C: {twin}\nPY: {oracle}"
    assert refused == {"x86_64-windows"}, refused


# What C requires a diagnostic for, or what is no integer constant expression where one is required, refused on both
# rails for one reason: two case labels equal in the switch's promoted type (C11 6.8.4.2p3), a second `default:`, a
# bit-field's width negative, named and zero, or wider than its type (6.7.2.1p4), a floating constant cast where C
# leaves the conversion undefined (6.3.1.4p1), one that is no cast's immediate operand (`(int)-1.5`, 6.6p6), a `long
# double` one (its format the target's), `sizeof` of an expression, `sizeof` of an incomplete type, an enumerator read
# after the block or the function body that declared it, and a character constant C does not define or the C front
# does not read (no character, an escape past its code unit or C does not define, a universal character name, two
# behind a prefix) -- each had been read as an `int` of whatever bytes it held.
_DUP_CASE = "duplicate case value"
_DUP_DEFAULT = "multiple default labels in one switch"
_BF_WIDTH = "invalid bit-field width"
_CHAR_BAD = "unsupported character constant"
_CONSTEXPR2_REFUSED = (
    (
        "uint32_t f(uint32_t v) { switch (v) { case -1: return 1u; case 4294967295u: return 2u; } return 0u; }",
        _DUP_CASE,
    ),
    (
        "int32_t f(int32_t v) { switch (v) { case 'a': return 1; case 97: return 2; } return 0; }",
        _DUP_CASE,
    ),
    (
        "int32_t f(int32_t v) { switch (v) { case u8'a': return 1; case 97u: return 2; } return 0; }",
        _DUP_CASE,
    ),
    (
        "int64_t f(int64_t v) { switch (v) { case -1: return 1; case 18446744073709551615u: return 2; } return 0; }",
        _DUP_CASE,
    ),
    (
        "uint8_t f(uint8_t v) { switch (v) { case 3: return 1u; case 1 + 2: return 2u; } return 0u; }",
        _DUP_CASE,
    ),
    (  # a `_BitInt(N)` is not promoted (C23 6.3.1.1p2): -1 is its 4095
        "uint32_t f(unsigned _BitInt(12) v) { switch (v) { case 4095: return 1u; case -1: return 2u; } return 0u; }",
        _DUP_CASE,
    ),
    (
        "int32_t f(int32_t v) { switch (v) { default: return 1; case 1: return 2; default: return 3; } }",
        _DUP_DEFAULT,
    ),
    (
        "struct s { uint8_t a : 9; };\nuint32_t f(uint32_t x) { struct s v; v.a = x; return v.a; }",
        _BF_WIDTH,
    ),
    (
        "struct s { uint32_t a : 0; };\nuint32_t f(uint32_t x) { struct s v; v.a = x; return v.a; }",
        _BF_WIDTH,
    ),
    (
        "struct s { uint32_t a : -1; };\nuint32_t f(uint32_t x) { struct s v; v.a = x; return v.a; }",
        _BF_WIDTH,
    ),
    (
        "struct s { _Bool a : 2; };\nuint32_t f(uint32_t x) { struct s v; v.a = x; return v.a; }",
        _BF_WIDTH,
    ),
    (
        "struct s { uint32_t : 33; uint32_t b; };\nuint32_t f(uint32_t x) { struct s v; v.b = x; return v.b; }",
        _BF_WIDTH,
    ),
    ("enum { N = (int)1e10 };", _ICE_NOT),
    ("enum { N = (unsigned char)300.7 };", _ICE_NOT),
    ("enum { N = (unsigned)-0.5 };", _ICE_NOT),
    ("enum { N = (int)-1.5 };", _ICE_NOT),
    ("enum { N = (int)1.5L };", _ICE_NOT),
    ("enum { N = sizeof g_x };", _ICE_NOT),
    ("enum { N = sizeof(void) };", _ICE_NOT),
    ("enum { N = sizeof(struct nope) };", _ICE_NOT),
    ("struct half;\nenum { N = _Alignof(struct half) };", _ICE_NOT),
    (
        "uint32_t f(uint32_t x) { { enum { Q = 3 }; x += Q; } switch (x) { case Q: return 1u; } return 0u; }",
        _ICE_NOT,
    ),
    (
        "uint32_t g(uint32_t x) { enum { Q = 3 }; return x + Q; }\n"
        "uint32_t f(uint32_t x) { switch (x) { case Q: return 1u; } return 0u; }",
        _ICE_NOT,
    ),
    # an enumeration's tag ends with the block that declares it (6.2.1p4): after it, `enum e` names no definition
    (
        "uint32_t f(uint32_t s) { { enum e { A = 1 }; s += A; } enum e x = 0; return s + x; }",
        "an enumerated type with no definition",
    ),
    (
        "uint32_t f(uint32_t s) { { enum e { A = 1 }; s += A; } return s + (uint32_t)sizeof(enum e); }",
        "an enumerated type with no definition",
    ),
    ("uint32_t f(uint32_t s) { return s + '\\q'; }", _CHAR_BAD),
    ("uint32_t f(uint32_t s) { return s + ''; }", _CHAR_BAD),
    ("uint32_t f(uint32_t s) { return s + '\\x100'; }", _CHAR_BAD),
    ("uint32_t f(uint32_t s) { return s + '\\u0041'; }", _CHAR_BAD),
    ("uint32_t f(uint32_t s) { return s + u8'ab'; }", _CHAR_BAD),
    ("uint32_t f(uint32_t s) { return s + u'\\x10000'; }", _CHAR_BAD),
)


def test_case_labels_widths_and_constants_c_rejects_are_refused_alike_on_both_rails():
    """CF-CONSTEXPR2: every unit of `_CONSTEXPR2_REFUSED` is refused on both rails for the one reason it witnesses. On
    the parent both rails lowered the duplicate case labels and the second `default:` -- an emit no compiler takes --
    laid out every one of the bit-fields, which Clang refuses, and read each character constant as an `int` of its
    bytes."""
    from bcir.frontends.cfront.clex import CLexError
    from bcir.frontends.cfront.cparse import CParseError
    from bcir.frontends.cfront.lower import CLowerError

    for body, why in _CONSTEXPR2_REFUSED:
        try:
            compile_unit(_ENUMFOLD_HEAD + body + "\n", check_clang=False)
        except (CLexError, CParseError, CLowerError) as e:
            assert str(e) == why, (body, str(e))
        else:
            raise AssertionError(f"the oracle lowered {body!r}")
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, (body, why) in enumerate(_CONSTEXPR2_REFUSED):
            path = os.path.join(d, f"r{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_ENUMFOLD_HEAD + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode == 1 and run.stdout.strip() == f"PARSE-ERR {why}", (
                body,
                run.stdout[:200],
            )


# ... while what C takes where it requires an integer constant expression lowers alike: an enumerator in a `_BitInt`'s
# width, `aligned(N)` and `alignas(N)`, a nested switch reusing its enclosing one's labels, a `uint8_t` switch's 255 and
# -1, `typedef T row[0];` and its member, a block enumeration hiding a file-scope one of its name -- its tag too, so
# `enum t` after the block is the outer one again (an `unsigned int` on the System V targets, an `int` on Windows).
_CONSTEXPR2_TAKEN = (
    "enum { W = 12 };\nuint32_t f(uint32_t s) { _BitInt(W) x = (_BitInt(W))(s & 0x7FFu); return (uint32_t)x + 1u; }",
    "enum { A = 8 };\nstruct s { uint8_t c; __attribute__((aligned(A * 2))) uint32_t v; };\n"
    "uint32_t f(uint32_t x) { struct s v; v.v = x; return v.v + (uint32_t)sizeof(struct s); }",
    "struct s { uint8_t c; _Alignas(sizeof(double) * 2) uint32_t v; };\n"
    "uint32_t f(uint32_t x) { struct s v; v.v = x; return v.v + (uint32_t)_Alignof(struct s); }",
    "uint32_t f(uint32_t v, uint32_t w) { switch (v) { case 1: switch (w) { case 1: return 5u; case 2: return 6u; } "
    "case 2: return 7u; } return 0u; }",
    "typedef uint32_t row[0];\nstruct hdr { uint32_t n; row tail; };\n"
    "uint32_t f(const struct hdr *h, uint32_t i) { return h->n + h->tail[i & 1u] + (uint32_t)sizeof(struct hdr); }",
    "enum { N = 4 };\nuint32_t f(uint32_t s) { enum { N = 2 }; uint32_t a[N]; a[1] = s; return a[1] + N + "
    "(uint32_t)sizeof a; }",
    "enum t { TA = 1 };\nuint32_t f(uint32_t s) { { enum t { TB = -1 }; enum t y = TB; s += (uint32_t)y; } "
    "enum t x = TA; return s + (uint32_t)(x - 2 > 0); }",
)


def test_constant_widths_dimensions_and_block_enumerations_lower_alike_on_both_rails():
    """CF-CONSTEXPR2: every unit of `_CONSTEXPR2_TAKEN` lowers to one claim graph on the four targets. On the parent the
    oracle refused an enumerator as `_BitInt(N)`'s width, `aligned(N)`'s and `alignas(N)`'s, and both rails a block
    enumeration; the twin refused the zero-length typedef."""
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        for n, body in enumerate(_CONSTEXPR2_TAKEN):
            unit = _ENUMFOLD_HEAD + body + "\n"
            path = os.path.join(d, f"t{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(unit)
            _parity_on_targets(path, unit)


def _chain(term: str, op: str, n: int) -> str:
    """`term op term op ...`, `n` terms, one to a line (the twin's preprocessor bounds a line)."""
    return f" {op}\n".join([term] * n)


def test_a_3000_term_chain_lowers_and_runs_as_the_original_on_both_rails():
    """CF-CONSTEXPR2: a left-associative chain of 3000 terms -- an enumerator's `1 + 1 + ...`, a case label's, a static's
    initializer, a return's `x + x + ...`, `&&`, `||` and comma chains, a pointer's `p + 0 + ...` and `sizeof` of a
    chain and an allocation's element count -- lowers to one claim graph on both rails, and each function returns what
    the original does. The oracle had raised RecursionError folding one (`_kfold_node`), walking it
    (`_scan_mutations`), lowering it (`_rvalue`), typing it (`_type_of`, `_sizeof_type`) and judging it pure
    (`_is_pure`); the twin lowers it. Each unit stays under the twin's preprocessed
    text bound."""
    n = 3000
    units = {
        "ch_enum": (
            f"enum {{ CH_N = {_chain('1', '+', n)} }};\n"
            "uint32_t ch_enum(uint32_t x) { return x + (uint32_t)CH_N; }\n"
        ),
        "ch_static": (
            f"static const int32_t CH_K = {_chain('1', '+', n)};\n"
            "uint32_t ch_static(uint32_t x) { return x + (uint32_t)CH_K; }\n"
        ),
        "ch_case": (
            "uint32_t ch_case(uint32_t v) {\n"
            f"  switch (v) {{ case {_chain('1', '+', n)}: return 1u; default: return 2u; }}\n}}\n"
        ),
        "ch_sum": f"uint32_t ch_sum(uint32_t x) {{ return {_chain('x', '+', n)}; }}\n",
        "ch_and": f"uint32_t ch_and(uint32_t x) {{ return {_chain('x', '&&', n)}; }}\n",
        "ch_or": f"uint32_t ch_or(uint32_t x) {{ return {_chain('x', '||', n)}; }}\n",
        "ch_comma": f"uint32_t ch_comma(uint32_t x) {{ return ({_chain('x', ',', n)}); }}\n",
        "ch_ptr": (
            "uint32_t ch_ptr(uint32_t x) { uint32_t a[2] = {x, x + 1u}; "
            f"return *(a + {_chain('0', '+', n)}) + (uint32_t)sizeof({_chain('x', '+', n // 2)}); }}\n"
        ),
        "ch_alloc": (  # an allocation's element count, judged pure (`_is_pure`) to bound the pointer's accesses
            "uint32_t ch_alloc(uint32_t x) {\n"
            f"  uint32_t *p = malloc((x * 0u + {_chain('0u', '+', n)} + 2u) * sizeof(uint32_t));\n"
            "  if (!p) return 0u;\n  p[0] = x; p[1] = x + 1u;\n  uint32_t r = p[0] + p[1];\n  free(p);\n  return r;\n}\n"
        ),
    }
    if not _CC:
        for body in units.values():
            assert "ok=1" in _oracle("#include <stdint.h>\n#include <stdlib.h>\n" + body)[0]
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for name, body in units.items():
            unit = "#include <stdint.h>\n#include <stdlib.h>\n" + body
            oracle_summary, r, _entry = _oracle(unit)
            assert "ok=1" in oracle_summary, (name, oracle_summary)
            oracle_emit = "\n".join(r.emitted[f] for f in r.lowered.functions)
            path = os.path.join(d, f"{name}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(unit)
            c_summary, c_emit = _c_run(exe, path)
            assert c_summary == oracle_summary, (name, c_summary, oracle_summary)
            driver = (
                _GAPS_SAME
                + "int main(void) {\n  for (unsigned n = 0; n < GAPS_N; n++) {\n    uint32_t s = gaps_in[n];\n"
                + f"    SAME({name}, s);\n"
                + ("    SAME(ch_case, 3000u);\n" if name == "ch_case" else "")
                + '  }\n  puts("MATCH");\n  return 0;\n}\n'
            )
            _run_against_original(
                f"{name}.c", unit, (("twin", c_emit), ("oracle", oracle_emit)), driver
            )


def test_a_character_constant_after_an_identifier_is_no_digit_separator():
    """CF-CONSTEXPR2: both preprocessors copy an identifier and a pp-number whole when they strip comments (C11 6.4.8),
    so the `'` of `u8'a'` or `case'a'` opens a character constant and a comment after it is stripped, while a C23 digit
    separator (`1'000`, `0xca'fe`, `1e+5'0`, `1e+'0` after an exponent's sign) stays inside its number. Both had taken a `'` between two hex digits for a
    separator: the closing quote of `u8'a'` then opened a literal that ran past the next comment, which survived as the
    tokens `/ *`."""
    from bcir.frontends.cfront.cpp import preprocess

    unit = (
        "#include <stdint.h>\n"
        "uint32_t f(uint32_t v) { switch (v) { case'a': return u8'a'; /* a comment's quote */ } return 1'000u; }\n"
        "/* it's stripped */ uint32_t g(void) { return 0xca'feu; }\n"
        "#if 0\ndouble h(void) { return 1e+5'0; } /* neither rail's lexer reads this C23 constant */\n#endif\n"
        "#if 0\nint k = 1e+'0;\n#endif\n"  # one pp-number: a separator may follow an exponent's sign (6.4.8)
        "/* stripped after the separator */ uint32_t m(void) { return 'm'; }\n"
    )
    out = preprocess(unit)
    assert (
        "/" not in out and "comment" not in out and "stripped" not in out and "lexer" not in out
    ), out
    assert "1'000u" in out and "0xca'feu" in out, out
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sep.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(unit)
        run = subprocess.run([exe, "--emit-cpp", path], capture_output=True, text=True)
        assert (
            run.returncode == 0 and "comment" not in run.stdout and "stripped" not in run.stdout
        ), run.stdout[-400:]
        _parity_on_targets(path, unit)


# CF-TYPEDEFSCOPE: a block-scope name hides a typedef name of its own (C11 6.2.1p4). `cfront_typedefscope.c`'s
# functions are each run against the original.
_TYPEDEFSCOPE_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(td_assign, s); SAME(td_operands, s); SAME(td_param, s, 3u); SAME(td_param, 7u, s); SAME(td_block, s);
    SAME(td_loop, s); SAME(td_enum, s); SAME(td_init, s); SAME(td_array, s); SAME(td_pointer, s); SAME(td_call, s);
    SAME(td_struct, s); SAME(td_next, s); SAME(td_shadow, s); SAME(td_typeof, s); SAME(td_selfname, s);
    SAME(td_selflocal, s); SAME(td_member, s); SAME(td_spelled, s); SAME(td_helper, s, s ^ 0x55u);
    SAME(td_suffix, s); SAME(typedefscope, s);
    /* the global `td_write` stores after its block, not the block's local of its name */
    td_w = 5u; uint32_t w = td_write(s), gw = td_w; td_w = 5u;
    if (bcir_td_write(s) != w || td_w != gw) return fail("td_write");
  }
  /* the values C gives: the object, not the typedef, where the object hides it */
  if (bcir_td_init(0u) != 4u || bcir_td_operands(0u) != 1u + 800u + 8u + 3u || bcir_td_enum(2u) != 15u ||
      bcir_td_selfname(1u) != 9u || bcir_td_member(4u) != 5u || bcir_td_helper(2u, 5u) != 11u)
    return fail("values");
  puts("MATCH");
  return 0;
}
"""
)


def test_a_block_scope_name_hides_a_typedef_on_both_rails():
    """CF-TYPEDEFSCOPE: `cfront_typedefscope.c` -- a local, a parameter, a loop's own declaration and a block's
    enumerator each hiding a typedef name to the end of its block, in statements that start with it (`T = T * 3u;`,
    `T * x;`, `T[1] ^= 4u;`, `*T += 2u;`, `F(s)`), a cast's place (`(T) - s`, `(T) * x`, `(T)[1]`, `(F)(s)`) and `sizeof`
    and `typeof` operands (`sizeof(B)` of a `uint64_t` local beside `typedef uint8_t B`, 8, its own initializer's
    `sizeof(B)` 4), a parameter and a local declared with the typedef they name (`T T`), a member named as it, and the
    typedef declaring and casting again after the block, the loop and the function -- lowers to one claim graph on the
    four targets, and each emit returns what the original does: its locals, declared up front, take no name the
    function spells at file scope (a typedef, a global it reads or writes after the block, a function it calls). On the
    parent both rails refused the statements that start with the hidden name, and both lowered `(T) - s` as a cast of
    `-s` and `sizeof(B)` as 1, digest-equal: silent miscompiles; the twin's emit read a block's `td_g` for the global
    after the block."""
    if not _CC:
        return
    fx = "cfront_typedefscope.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    _run_against_original(
        fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _TYPEDEFSCOPE_DRIVER
    )


def test_both_emits_keep_their_objects_off_the_names_they_spell_for_themselves():
    """CF-TYPEDEFSCOPE: the names an emit spells for itself -- the libc routines it calls or copies through (`memcpy`
    spells every member store), the C11 atomics, the twin's store helper `_v`, the standard type names -- are one list
    on both rails (`emit._EMIT_SPELLED`; `bcir_cfront.c`'s `emit_spelled`), read here out of each rail's own source:
    neither emit names a parameter or a local as the other spells something else. The fixture's `td_spelled`,
    `td_helper` and `td_suffix` hold the rule on both rails' emits against the original."""
    from bcir.frontends.cfront import emit

    with open(os.path.join(_C, "bcir_cfront.c"), encoding="utf-8") as fh:
        src = fh.read()
    m = re.search(r"static const char \*const emit_spelled\[\]=\{(.*?)\};", src, re.S)
    assert m, "bcir_cfront.c holds no emit_spelled list"
    twin = re.findall(r'"([^"]*)"', m.group(1))
    assert len(twin) == len(set(twin)), "a name twice in emit_spelled"
    assert {"memcpy", "_v", "size_t"} <= set(twin), twin
    assert set(twin) == set(emit._EMIT_SPELLED), sorted(set(twin) ^ set(emit._EMIT_SPELLED))


# ... and where the hidden name is used as the type it no longer names, C has a syntax error, as Clang reports: a
# declaration, a cast, a compound literal, `sizeof` of a type built on it, after a local, a parameter or a block's
# enumerator hides it. Each parser refuses it in its own syntax-error words.
_TYPEDEFSCOPE_REFUSED = (
    "typedef uint32_t T;\nuint32_t f(uint32_t s) { uint32_t T = s; T y = 3u; return y; }",
    "typedef uint32_t T;\nuint32_t f(uint32_t s) { uint32_t T = s; const T y = 3u; return y; }",
    "typedef uint32_t T;\nuint32_t f(uint32_t s) { uint32_t T = s; return (T)s; }",
    "typedef uint32_t T;\nuint32_t f(uint32_t s) { uint32_t T = s; return (uint32_t)sizeof(T *); }",
    "typedef uint32_t T;\nuint32_t f(uint32_t s) { uint32_t T = s; return (T){s}; }",
    "typedef uint32_t T;\nuint32_t f(uint32_t T) { T y = 1u; return y; }",
    "typedef uint32_t T;\nuint32_t f(uint32_t s) { enum { T = 5 }; T y = s; return y; }",
    "typedef uint32_t T;\nuint32_t f(uint32_t s) { for (uint32_t T = 0u; T < s; T++) { T y = T; s += y; } return s; }",
    # a parameter hides the typedef for the parameters after it, in a definition's list and in a prototype's
    "typedef uint32_t T;\nuint32_t f(uint32_t T, T x) { return T + x; }",
    "typedef uint32_t T;\nuint32_t g(uint32_t T, T x);\nuint32_t f(uint32_t s) { return s; }",
    "typedef uint32_t T;\nuint32_t g(uint32_t T, uint32_t (*p)(T));\nuint32_t f(uint32_t s) { return s; }",
)


def test_a_hidden_typedef_name_starts_no_declaration_or_cast_on_both_rails():
    """CF-TYPEDEFSCOPE: every unit of `_TYPEDEFSCOPE_REFUSED` -- the hidden name used as a type -- is refused by Clang,
    by the oracle's parser and by the twin's, so neither rail reads the typedef through the object that hides it."""
    from bcir.frontends.cfront.cparse import CParseError

    clang = shutil.which("clang")
    exe = _build_frontend(_session_build_dir()) if _CC else None
    with tempfile.TemporaryDirectory() as d:
        for n, body in enumerate(_TYPEDEFSCOPE_REFUSED):
            unit = "#include <stdint.h>\n" + body + "\n"
            try:
                compile_unit(unit, check_clang=False)
            except CParseError:
                pass
            else:
                raise AssertionError(f"the oracle lowered {body!r}")
            path = os.path.join(d, f"h{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(unit)
            if clang:
                run = subprocess.run(
                    [clang, "-std=c2x", "-fsyntax-only", path], capture_output=True, text=True
                )
                assert run.returncode != 0, f"Clang took {body!r}"
            if exe:
                run = subprocess.run([exe, path], capture_output=True, text=True)
                assert run.returncode == 1 and run.stdout.startswith("PARSE-ERR "), (
                    body,
                    run.stdout[:200],
                )


# CF-FPTAB: a call through any postfix expression whose value is a function pointer (C11 6.5.2.2p1) -- `(*fp)(x)`,
# `(**fp)(x)` and `(*f)(x)` of a function, whose `*` names it again (6.5.3.2p4, 6.3.2.1p4), `t[i](x)` and
# `(*t[i])(x)` of a table -- and the tables themselves. `cfront_fptab.c` returns the elements it picks, and the
# functions `*fp` names as a value; they are compared with the original's and called here.
_FPTAB_DRIVER = (
    _GAPS_SAME
    + r"""
#define SAME_PICK(f) do { if (f(s) != bcir_##f(s) || f(s)(s) != bcir_##f(s)(s)) return fail(#f); } while (0)
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME_PICK(ft_pick_global); SAME_PICK(ft_pick_raw); SAME_PICK(ft_pick_local); SAME_PICK(ft_pick_grid);
    SAME_PICK(ft_pick_member); SAME_PICK(ft_pick_ptr); SAME_PICK(ft_pick_deref);
    SAME(ft_params, s); SAME(ft_call_deref, s); SAME(ft_call_one, s); SAME(ft_compare, s); SAME(ft_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)
# ... and a call through a table in place: a pointer the escape rows count as resolved to no one function, which a
# corpus fixture keeps out (local, file-scope and 2-D tables by a runtime index, through a pointer, a parameter of
# every spelling and a member, under `*` and `**`, a generic selection's and a select's function)
_FPTAB_CALLS = """#include <stdint.h>
typedef uint32_t (*op_t)(uint32_t);
struct dev { uint32_t id; op_t fn[2]; uint32_t (*raw[2])(uint32_t); op_t one; };
static uint32_t tw(uint32_t x) { return x * 2u + 1u; }
static uint32_t th(uint32_t x) { return x * 3u + 5u; }
static uint32_t ng(uint32_t x) { return 0u - x; }
static uint32_t mx(uint32_t x) { return (x ^ 0x5Au) * 7u; }
static op_t g_ops[4] = {tw, th, ng, mx};
static uint32_t (*g_raw[2])(uint32_t) = {mx, tw};
static op_t g_grid[2][2] = {{ng, tw}, {mx, th}};
static struct dev g_dev = {3u, {th, ng}, {mx, tw}, th};
static uint32_t via_ptr(op_t *p, uint32_t s) { return (*p)(s) + p[1](s) * 3u + (**p)(s + 1u) * 5u + (*(p + 2))(s) * 7u; }
static uint32_t via_arr(op_t t[], uint32_t s) { return t[s & 3u](s) + (*t[(s >> 1) & 3u])(s) * 3u; }
static uint32_t via_grid(uint32_t (*t[2][2])(uint32_t), uint32_t s) { return t[s & 1u][(s >> 1) & 1u](s) + (*t[1][s & 1u])(s); }
static uint32_t via_pp(uint32_t (**pp)(uint32_t), uint32_t s) { return (**pp)(s) + (*pp)(s + 2u) * 3u; }
static uint32_t via_dev(struct dev *d, uint32_t s) {
  return d->fn[s & 1u](s) + (*d->fn[(s >> 1) & 1u])(s) * 3u + d->raw[s & 1u](s) * 5u + (*d->one)(s) * 7u + (**d->raw)(s) * 11u;
}
uint32_t fpt_calls(uint32_t s) {
  op_t t[4] = {mx, ng, th, tw};
  uint32_t (*u[2])(uint32_t) = {th, mx};
  uint32_t (*h[2][2])(uint32_t) = {{tw, th}, {ng, mx}};
  op_t fp = s & 1u ? tw : ng;
  op_t *p = t;
  uint32_t (**q)(uint32_t) = u;
  struct dev d = {1u, {ng, mx}, {th, tw}, mx};
  uint32_t k = t[s & 3u](s) + (t[(s >> 2) & 3u])(s) * 3u + (*t[(s + 1u) & 3u])(s) * 5u + u[s & 1u](s) * 7u;
  k += h[s & 1u][(s >> 1) & 1u](s) * 11u + (*h[(s >> 1) & 1u][s & 1u])(s) * 13u;
  k += g_ops[s & 3u](s) * 17u + (*g_raw[s & 1u])(s) * 19u + g_grid[1][s & 1u](s) * 23u + (**g_ops)(s) * 29u;
  k += (*fp)(s) * 31u + (**fp)(s) * 37u + p[(s >> 1) & 3u](s) * 41u + (*q)(s) * 43u + q[1](s) * 47u;
  k += d.fn[s & 1u](s) * 53u + (*d.raw[(s >> 1) & 1u])(s) * 59u + (*d.one)(s) * 61u;
  k += g_dev.fn[(s >> 2) & 1u](s) * 67u + g_dev.raw[s & 1u](s) * 71u;
  k += via_ptr(g_ops, s) * 73u + via_arr(t, s) * 79u + via_grid(h, s) * 83u + via_pp(&fp, s) * 89u + via_dev(&d, s) * 97u;
  k += _Generic(s, uint32_t: tw, default: th)(s) * 101u + (s & 1u ? tw : th)(s) * 103u + (*(s & 2u ? ng : mx))(s) * 107u;
  return k;
}
"""
# The operand kinds C requires of a call, `*`, `[]`, `.` and `->` (C11 6.5.2.2p1, 6.5.3.2p2, 6.5.2.1p1,
# 6.5.2.3p1-2) and of what `sizeof` measures (6.5.3.4p1): each wrong operand refused on both rails for one reason.
# On the parent the oracle read `*s`, `(*s)++` and `*s += 1u` of an integer through memory at `s` as a `uint32_t`;
# both rails lowered `*s = 1u`, `*(s + 1u)`, `&*s`, `s[1]`, `&s[1]` and `s[1]++`, the twin `s->x` of a struct as
# `s.x`, `p.x` of a pointer as `p->x`, `o[1].v` and `fp[0](s)`, and both `*fp = 1u`; the rest were refused for
# reasons that differed between the rails.
_FPTAB_HEAD = (
    "#include <stdint.h>\ntypedef uint32_t (*op_t)(uint32_t);\nstruct S { uint32_t x; };\n"
    "static uint32_t inc(uint32_t x) { return x + 1u; }\nuint32_t g;\n"
)
_FPTAB_REFUSED = (
    ("uint32_t f(uint32_t s) { uint32_t x = s; return (*x)(1u); }", "dereference of a non-pointer"),
    (
        "uint32_t f(uint32_t *p) { return (*p)(1u); }",
        "called object is not a function or function pointer",
    ),
    (
        "uint32_t f(uint32_t s) { uint32_t a[2] = {s, s}; return a[0](1u); }",
        "called object is not a function or function pointer",
    ),
    (
        "uint32_t f(uint32_t s) { return inc(s)(1u); }",
        "called object is not a function or function pointer",
    ),
    (
        "uint32_t f(uint32_t s) { (void)s; return 1(2); }",
        "called object is not a function or function pointer",
    ),
    (
        'uint32_t f(uint32_t s) { (void)s; return "ab"(1u); }',
        "called object is not a function or function pointer",
    ),
    ("uint32_t f(uint32_t s) { return *s; }", "dereference of a non-pointer"),
    ("uint32_t f(uint32_t s) { *s = 1u; return s; }", "dereference of a non-pointer"),
    ("uint32_t f(uint32_t s) { return *(s + 1u); }", "dereference of a non-pointer"),
    ("uint32_t f(uint32_t s) { *g = s; return g; }", "dereference of a non-pointer"),
    ("uint32_t f(uint32_t s) { (*s)++; return s; }", "dereference of a non-pointer"),
    ("uint32_t f(uint32_t s) { ++*s; return s; }", "dereference of a non-pointer"),
    ("uint32_t f(uint32_t s) { *s += 1u; return s; }", "dereference of a non-pointer"),
    ("uint32_t f(uint32_t s) { uint32_t *p = &*s; return *p; }", "dereference of a non-pointer"),
    (
        "uint32_t f(uint32_t s) { uint32_t x = (*s = 3u); return x; }",
        "dereference of a non-pointer",
    ),
    ("uint32_t f(struct S s) { struct S t = *s; return t.x; }", "dereference of a non-pointer"),
    ("uint32_t f(uint32_t s) { return (uint32_t)sizeof *s; }", "dereference of a non-pointer"),
    (
        "uint32_t f(uint32_t s) { uint32_t a[2] = {s, s}; return *a[0]; }",
        "dereference of a non-pointer",
    ),
    ("uint32_t f(uint32_t s) { return *(uint32_t)s; }", "dereference of a non-pointer"),
    (
        "uint32_t f(uint32_t s) { return s[1]; }",
        "subscripted value is not an array or a pointer to an object",
    ),
    (
        "uint32_t f(uint32_t s) { s[1] = 2u; return s; }",
        "subscripted value is not an array or a pointer to an object",
    ),
    (
        "uint32_t f(uint32_t s) { s[1]++; return s; }",
        "subscripted value is not an array or a pointer to an object",
    ),
    (
        "uint32_t f(uint32_t s) { uint32_t *p = &s[1]; return *p; }",
        "subscripted value is not an array or a pointer to an object",
    ),
    (
        "uint32_t f(uint32_t s) { return (uint32_t)sizeof s[1]; }",
        "subscripted value is not an array or a pointer to an object",
    ),
    (
        "uint32_t f(uint32_t s) { op_t fp = inc; return fp[0](s); }",
        "subscripted value is not an array or a pointer to an object",
    ),
    (
        "struct V { uint32_t v; };\nuint32_t f(struct V *o) { return o->v[1]; }",
        "subscripted value is not an array or a pointer to an object",
    ),
    (
        "uint32_t f(struct S o) { return o[1].x; }",
        "subscripted value is not an array or a pointer to an object",
    ),
    (
        "uint32_t f(uint32_t *p) { return 1[p]; }",
        "a subscript of an integer constant is not supported",
    ),
    (
        "uint32_t f(struct S s) { return s->x; }",
        "member access `->` through a value that is not a pointer to a struct or union",
    ),
    (
        "uint32_t f(struct S s) { s->x = 1u; return s.x; }",
        "member access `->` through a value that is not a pointer to a struct or union",
    ),
    (
        "uint32_t f(struct S s) { return (*s).x; }",
        "member access `->` through a value that is not a pointer to a struct or union",
    ),
    (
        "uint32_t f(uint32_t s) { return s->x; }",
        "member access `->` through a value that is not a pointer to a struct or union",
    ),
    (
        "uint32_t f(struct S *p) { return p.x; }",
        "member access on an object that is not a struct or union",
    ),
    (
        "uint32_t f(struct S *p) { p.x = 1u; return p->x; }",
        "member access on an object that is not a struct or union",
    ),
    (
        "uint32_t f(uint32_t s) { return s.x; }",
        "member access on an object that is not a struct or union",
    ),
    (
        "uint32_t f(uint32_t s) { op_t fp = inc; *fp = 1u; return s; }",
        "a function designator is not an lvalue",
    ),
    (
        "uint32_t f(uint32_t s) { op_t fp = inc; (*fp)++; return s; }",
        "a function designator is not an lvalue",
    ),
    (
        "uint32_t f(uint32_t s) { op_t fp = inc; return (uint32_t)sizeof *fp + s; }",
        "sizeof of a function designator",
    ),
    (
        "uint32_t f(uint32_t s) { op_t t[s]; (void)t; return s; }",
        "a variable-length array of function pointers is not supported",
    ),
    (
        "uint32_t f(uint32_t s) { uint32_t (*t[s])(uint32_t); (void)t; return s; }",
        "a variable-length array of function pointers is not supported",
    ),
    (
        "uint32_t f(uint32_t s) { uint32_t (*t[1][1][1][1])(uint32_t); (void)t; return s; }",
        "an array of function pointers of more than 3 dimensions is not supported",
    ),
)
# ... while the valid forms beside them lower alike: `*(1 + p)`, a table's `*t` and `**t`, an element stored, a
# table's `sizeof`, `&*fp`, a pointer to a pointer to a function pointer, a select of member reads compared with a
# function of their type, a 3-D table
_FPTAB_LOWERED = (
    "uint32_t f(uint32_t *p) { return *(1 + p); }",
    "uint32_t f(uint32_t *p) { *(1 + p) = 3u; return p[1]; }",
    "uint32_t f(uint32_t s) { op_t t[2] = {inc, inc}; return (*t)(s) + (**t)(s) + (*(t + 1))(s); }",
    "static op_t gt[2] = {inc, inc};\nuint32_t f(uint32_t s) { return (*gt)(s) + (**gt)(s) + (*(gt + 1))(s); }",
    "uint32_t f(uint32_t s) { op_t t[2] = {inc, inc}; *t = inc; t[s & 1u] = inc; return t[0](s) + (uint32_t)sizeof *t; }",
    "uint32_t f(uint32_t s) { op_t fp = inc; op_t q = &*fp; return q(s) + (**inc)(s); }",
    "uint32_t f(uint32_t s) { op_t fp = inc; op_t *pp = &fp; op_t **ppp = &pp; return (***ppp)(s) + (*pp)(s); }",
    "struct ops { op_t run; };\nuint32_t f(uint32_t s) { struct ops o = {inc}; op_t g2 = s ? o.run : inc; return g2(s); }",
    "uint32_t f(uint32_t s) { op_t t[2][2][2] = {{{inc, inc}, {inc, inc}}, {{inc, inc}, {inc, inc}}}; return t[1][s & 1u][1](s); }",
)


def _fptab_refusal(src: str) -> str:
    """The oracle's refusal of `src`, or "" when it lowers."""
    from bcir.frontends.cfront.cparse import CParseError
    from bcir.frontends.cfront.lower import CLowerError

    try:
        compile_unit(src, check_clang=False)
    except (CParseError, CLowerError) as e:
        return str(e)
    return ""


def test_tables_of_function_pointers_and_calls_through_them_run_as_the_original():
    """CF-FPTAB: `cfront_fptab.c` -- tables of function pointers, typedef'd and spelled inline (`uint32_t
    (*t[N])(uint32_t)`), local and file-scope, of two dimensions, struct members, pointers to them (`op_t *`,
    `uint32_t (**p)(uint32_t)`) and parameters of them; their elements read through `t[i]`, `*(t + i)` and `*p` and
    compared; calls through `(*fp)(x)`, `(**fp)(x)` and `(*f)(x)` of a function, and `*fp` as a value -- held,
    selected and returned (`ft_pick_deref`). Both rails refused `(*fp)(x)` and a call through an element, the twin
    every inline declarator and a typedef'd table parameter (declared as one function pointer), and both typed a
    member read as an integer. The fixture and `_FPTAB_CALLS` -- every call in place: through local, file-scope and
    2-D tables by a runtime index, a pointer, a parameter of every spelling, a member, under `*` and `**`, a generic
    selection and a select -- lower to one claim graph on the four targets, and each emit returns what the original
    does and the pointers the original's."""
    if not _CC:
        return
    fx = "cfront_fptab.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _FPTAB_DRIVER)
    exe = _build_frontend(_session_build_dir())
    oracle_summary, r, _entry = _oracle(_FPTAB_CALLS)
    oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fptab_calls.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_FPTAB_CALLS)
        c_summary, c_emit = _c_run(exe, path)
        assert c_summary == oracle_summary and "ok=1" in c_summary, (c_summary, oracle_summary)
        _parity_on_targets(path, _FPTAB_CALLS)
    driver = (
        _GAPS_SAME
        + "int main(void) {\n  for (unsigned n = 0; n < GAPS_N; n++) SAME(fpt_calls, gaps_in[n]);\n"
        + '  puts("MATCH");\n  return 0;\n}\n'
    )
    _run_against_original(
        "fptab_calls.c", _FPTAB_CALLS, (("twin", c_emit), ("oracle", oracle_emit)), driver
    )


def test_operands_of_call_star_subscript_and_member_are_refused_alike_on_both_rails():
    """CF-FPTAB: the operand kinds C requires of a call, `*`, `[]`, `.` and `->` and of what `sizeof` measures --
    a call of a value that is no function pointer, a dereference of a non-pointer, a subscript of a value that is
    neither an array nor a pointer to an object (a function pointer included), `.` of a pointer and `->` of a
    struct, a function designator stored to or stepped, `sizeof` of a function, a table of variable length or of
    more than three dimensions (`_FPTAB_REFUSED`) -- each refused on both rails for its one reason. On the parent
    the oracle read `*s` of an integer through memory at `s`, both rails lowered `*s = 1u`, `s[1]`, `&*s` and
    `*fp = 1u`, and the twin `s->x` of a struct and `p.x` of a pointer. The valid forms beside them
    (`_FPTAB_LOWERED`) lower to one claim graph on the four targets."""
    for body, why in _FPTAB_REFUSED:
        got = _fptab_refusal(_FPTAB_HEAD + body + "\n")
        assert got == why, (body, got, why)
    for body in _FPTAB_LOWERED:
        got = _fptab_refusal(_FPTAB_HEAD + body + "\n")
        assert got == "", (body, got)
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, (body, why) in enumerate(_FPTAB_REFUSED):
            path = os.path.join(d, f"r{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_FPTAB_HEAD + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode == 1 and run.stdout.strip() == f"PARSE-ERR {why}", (
                body,
                run.returncode,
                run.stdout[:200],
            )
        for n, body in enumerate(_FPTAB_LOWERED):
            path = os.path.join(d, f"l{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_FPTAB_HEAD + body + "\n")
            _parity_on_targets(path, _FPTAB_HEAD + body + "\n")


# CF-FPRET: a function pointer a call returns -- held, compared, selected, returned and called through -- and what a
# call through a function pointer returns: a pointer, a struct whose member is read, a function pointer; and pointers
# to variadic functions, declared, selected and called. `cfront_fpret.c` returns the pointers it picks; they are
# compared with the original's and called here.
_FPRET_DRIVER = (
    _GAPS_SAME
    + r"""
#define SAME_PICK(f) do { if (f(s) != bcir_##f(s) || f(s)(s) != bcir_##f(s)(s)) return fail(#f); } while (0)
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME_PICK(fr_pick); SAME_PICK(fr_pick_held); SAME_PICK(fr_pick_late);
    if (fr_pick_va(s) != bcir_fr_pick_va(s) || fr_pick_va(s)(s, 3u) != bcir_fr_pick_va(s)(s, 3u)) return fail("va");
    SAME(fr_compare, s); SAME(fr_ptr_ret, s); SAME(fr_struct_ret, s); SAME(fr_variadic, s); SAME(fr_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)
# ... and a call through a returned pointer, or through a pointer to a variadic function, in place: a pointer the
# escape rows count as resolved to no one function, which a corpus fixture keeps out -- through a direct call's
# result, under `*`, through a pointer to a function returning one and a member of that type, a struct and a pointer
# returned through a member, a select of variadic functions
_FPRET_CALLS = """#include <stdint.h>
#include <stdarg.h>
typedef uint32_t (*op_t)(uint32_t);
typedef op_t (*mk_t)(uint32_t);
typedef uint32_t (*vop_t)(uint32_t, ...);
struct pair { uint32_t a, b; };
struct ops { mk_t mk; struct pair (*mp)(uint32_t); uint32_t *(*get)(uint32_t *); vop_t v; };
struct vs { uint32_t (*f)(uint32_t, ...); };
static uint32_t tw(uint32_t x) { return x * 2u + 1u; }
static uint32_t th(uint32_t x) { return x * 3u + 5u; }
static op_t pick(uint32_t s) { return s & 1u ? tw : th; }
static op_t pick2(uint32_t s) { return pick(s + 1u); }
static struct pair mkp(uint32_t s) { struct pair p = {s + 1u, s + 2u}; return p; }
static struct pair mkq(uint32_t s) { struct pair p = {s * 3u, s * 5u}; return p; }
static uint32_t *id(uint32_t *p) { return p; }
static uint32_t va(uint32_t n, ...) { va_list ap; va_start(ap, n); uint32_t r = n + va_arg(ap, uint32_t); va_end(ap); return r; }
static uint32_t vb(uint32_t n, ...) { va_list ap; va_start(ap, n); uint32_t r = n ^ va_arg(ap, uint32_t); va_end(ap); return r; }
uint32_t fpr_calls(uint32_t s) {
  mk_t m = s & 2u ? pick : pick2;
  struct ops o = {pick, mkp, id, va};
  struct pair (*mp)(uint32_t) = s & 4u ? mkp : mkq;
  vop_t g = s & 8u ? va : vb;
  uint32_t v = s;
  uint32_t k = pick(s)(s) + (*pick(s))(s) * 3u + m(s)(s) * 5u + (*m)(s)(s) * 7u + (*m(s))(s) * 11u;
  k += o.mk(s)(s) * 13u + mp(s).a * 17u + (*mp)(s).b * 19u + o.mp(s).b * 23u + *o.get(&v) * 29u;
  k += (s & 1u ? va : vb)(s, 3u) * 31u + o.v(s, 4u) * 37u + g(s, 6u) * 41u + pick2(s)(s + 1u) * 43u;
  struct vs o2 = {vb};
  uint32_t (*q)(uint32_t, ...) = o2.f;
  struct pair (*pt[2])(uint32_t) = {mkp, mkq};
  k += q(s, 2u) * 47u + (*(*m)(s))(s) * 53u + pt[s & 1u](s).a * 59u;
  return k;
}
"""
_FPRET_HEAD = (
    "#include <stdint.h>\n#include <stdarg.h>\ntypedef uint32_t (*op_t)(uint32_t);\n"
    "struct P { uint32_t a, b; };\n"
    "static uint32_t inc(uint32_t v) { return v + 1u; }\n"
    "static uint32_t *id(uint32_t *p) { return p; }\n"
    "static uint32_t va(uint32_t n, ...) { return n; }\n"
    "static uint32_t vb(uint32_t n, ...) { return n + 1u; }\n"
    "static uint32_t vc(uint64_t n, ...) { return (uint32_t)n; }\n"
    "static struct P gp = {3u, 4u};\n"
    "static struct P *getp(uint32_t s) { (void)s; return &gp; }\n"
)
_FN_RET_FP = "a function returning a function pointer is not supported without a typedef"
# A function declared to return a function pointer by a nested declarator, which neither rail parses, is refused for
# one reason (the same function with a typedef for its return type lowers); arms of `?:` that are functions of two
# types, a variadic one among them, are refused as CF-FNSEL refuses any (C11 6.5.15p3).
_FPRET_REFUSED = (
    ("uint32_t (*pk(uint32_t s))(uint32_t) { (void)s; return inc; }", _FN_RET_FP),
    ("uint32_t (*pk(uint32_t s))(uint32_t);\nuint32_t f(uint32_t s) { return s; }", _FN_RET_FP),
    ("static uint32_t (**pk2(void))(uint32_t);\nuint32_t f(uint32_t s) { return s; }", _FN_RET_FP),
    ("uint32_t f(uint32_t s) { return (s ? va : inc) != 0; }", _FN_SELECTED),
    ("uint32_t f(uint32_t s) { return (s ? va : vc) != 0; }", _FN_SELECTED),
)
# ... while a pointer return spelled in every declarator -- a typedef, a local, a member, a parameter, a pointer to a
# struct -- a function-pointer return declared by a prototype, and a table of pointers to variadic functions lower
# alike
_FPRET_LOWERED = (
    "typedef uint32_t *(*pf)(uint32_t *);\nuint32_t f(uint32_t s) { pf h = id; uint32_t v = s; return *h(&v); }",
    "uint32_t f(uint32_t s) { uint32_t *(*h)(uint32_t *) = id; uint32_t v = s; return *h(&v) + 1u; }",
    "struct G { uint32_t *(*get)(uint32_t *); };\n"
    "uint32_t f(uint32_t s) { struct G g = {id}; uint32_t v = s; return *g.get(&v); }",
    "static uint32_t ap2(uint32_t *(*h)(uint32_t *), uint32_t s) { uint32_t v = s; return *h(&v); }\n"
    "uint32_t f(uint32_t s) { return ap2(id, s); }",
    "uint32_t f(uint32_t s) { struct P *(*h)(uint32_t) = getp; return h(s)->a + h(s)->b; }",
    "op_t pickp(uint32_t s);\nuint32_t f(uint32_t s) { op_t g = pickp(s); return g(s); }\n"
    "op_t pickp(uint32_t s) { (void)s; return inc; }",
    "uint32_t f(uint32_t s) { uint32_t (*t[2])(uint32_t, ...) = {va, vb}; return t[s & 1u](s, 1u) + (*t)(s, 0); }",
)


def test_function_pointers_returned_by_calls_run_as_the_original():
    """CF-FPRET: `cfront_fpret.c` -- a function pointer a call returns, held, compared and returned; one returned
    through a pointer to a function that returns one; a pointer and a struct returned through a function pointer, the
    struct's member read; a pointer to a variadic function, declared inline and by a typedef, selected among variadic
    functions and called. On the parent both rails typed a call's function pointer `uint32_t` (no emit compiled, and a
    comparison read a pointer cut to 32 bits), the oracle refused `typedef T *(*pf)(T *)` and the twin typed what it
    returns as an integer, the twin refused `m(s).a`, and both refused a variadic function-pointer declarator. The
    fixture and `_FPRET_CALLS` -- every call through a returned pointer in place -- lower to one claim graph on the
    four targets, and each emit returns what the original does and the pointers the original's."""
    if not _CC:
        return
    fx = "cfront_fpret.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    _run_against_original(fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _FPRET_DRIVER)
    exe = _build_frontend(_session_build_dir())
    oracle_summary, r, _entry = _oracle(_FPRET_CALLS)
    oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fpret_calls.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_FPRET_CALLS)
        c_summary, c_emit = _c_run(exe, path)
        assert c_summary == oracle_summary and "ok=1" in c_summary, (c_summary, oracle_summary)
        _parity_on_targets(path, _FPRET_CALLS)
    driver = (
        _GAPS_SAME
        + "int main(void) {\n  for (unsigned n = 0; n < GAPS_N; n++) SAME(fpr_calls, gaps_in[n]);\n"
        + '  puts("MATCH");\n  return 0;\n}\n'
    )
    _run_against_original(
        "fpret_calls.c", _FPRET_CALLS, (("twin", c_emit), ("oracle", oracle_emit)), driver
    )


def test_function_pointer_returns_refused_and_lowered_alike_on_both_rails():
    """CF-FPRET: a function declared to return a function pointer by a nested declarator -- defined, prototyped,
    returning a pointer to one -- is refused on both rails for one reason, where each refused it with a parse error of
    its own; arms of `?:` that are a variadic function and a function of another type are refused as CF-FNSEL refuses
    any (`_FPRET_REFUSED`). A pointer return spelled in every declarator, a function-pointer return declared by a
    prototype and a table of pointers to variadic functions (`_FPRET_LOWERED`) lower to one claim graph on the four
    targets."""
    for body, why in _FPRET_REFUSED:
        got = _fptab_refusal(_FPRET_HEAD + body + "\n")
        assert got == why, (body, got, why)
    for body in _FPRET_LOWERED:
        got = _fptab_refusal(_FPRET_HEAD + body + "\n")
        assert got == "", (body, got)
    # the claim of a call through a pointer to a variadic function declares its signature with its `...`
    src = (
        _FPRET_HEAD
        + "uint32_t f(uint32_t s) { uint32_t (*g)(uint32_t, ...) = va; return g(s, 1u); }\n"
    )
    claims = compile_unit(src, check_clang=False).lowered.functions["f"].claims
    sigs = {c.callee_sig for c in claims if c.op == "c.call.indirect"}
    assert sigs == {"uint32_t(uint32_t, ...)"}, sigs
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, (body, why) in enumerate(_FPRET_REFUSED):
            path = os.path.join(d, f"r{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_FPRET_HEAD + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode == 1 and run.stdout.strip() == f"PARSE-ERR {why}", (
                body,
                run.returncode,
                run.stdout[:200],
            )
        for n, body in enumerate(_FPRET_LOWERED):
            path = os.path.join(d, f"l{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_FPRET_HEAD + body + "\n")
            _parity_on_targets(path, _FPRET_HEAD + body + "\n")


# CF-EXTDESIG: a function the unit only prototypes -- another unit defines it -- named as a value, its address taken, and
# called through every spelling. `cfront_extdesig_link.c` returns the pointers it makes; the driver defines the
# external functions, compares the pointers with the original's and calls them.
_EXTDESIG_DEFS = r"""
#include <stdarg.h>
uint32_t ed_ext(uint32_t v) { return v * 7u + 3u; }
uint32_t ed_ext2(uint32_t v) { return v ^ 0x5Au; }
uint32_t ed_vext(uint32_t n, ...) {
  va_list ap;
  va_start(ap, n);
  uint32_t r = n + va_arg(ap, uint32_t) * 3u;
  va_end(ap);
  return r;
}
uint32_t ed_apply(ed_op f, uint32_t s) { return f(s) + 1u; }
"""
_EXTDESIG_DRIVER = (
    _GAPS_SAME
    + _EXTDESIG_DEFS
    + r"""
#define SAME_PICK(f) do { if (f(s) != bcir_##f(s) || f(s)(s) != bcir_##f(s)(s)) return fail(#f); } while (0)
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME_PICK(ed_pick); SAME_PICK(ed_held); SAME_PICK(ed_member); SAME_PICK(ed_table);
    if (ed_pick_va(s) != bcir_ed_pick_va(s)) return fail("ed_pick_va");
    if (ed_pick_va(s) && ed_pick_va(s)(s, 4u) != bcir_ed_pick_va(s)(s, 4u)) return fail("ed_pick_va call");
    SAME(ed_compare, s); SAME(ed_pass, s); SAME(ed_calls, s); SAME(ed_sizes, s); SAME(ed_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)
# The emits built with a pointer compared with an integer, or converted to one, made an error under both compilers --
# a `0` compared with a designator is a null pointer of its type
_EXTDESIG_WERROR = {
    "clang": (
        "-Werror=int-conversion",
        "-Werror=pointer-integer-compare",
        "-Werror=incompatible-pointer-types",
    ),
    "gcc": ("-Werror=int-conversion", "-Werror"),
}
# ... and calls in place through a pointer that may hold a function another unit defines: through a local, a select,
# a member, a table, a pointer to a variadic one and the result of a call -- each an external edge, which the G10
# rows count as reaching unknown code, so the corpus fixture keeps them out
_EXTDESIG_CALLS = """#include <stdint.h>
typedef uint32_t (*ed_op)(uint32_t);
typedef uint32_t (*ed_vop)(uint32_t, ...);
struct ed_ops { ed_op fn; ed_op alt; };
uint32_t ed_ext(uint32_t v);
uint32_t ed_ext2(uint32_t v);
uint32_t ed_vext(uint32_t n, ...);
uint32_t ed_apply(ed_op f, uint32_t s);
static uint32_t ed_inc(uint32_t v) { return v + 1u; }
static ed_op ed_choose(uint32_t s) { return s & 1u ? ed_ext : &ed_ext2; }
uint32_t edc_calls(uint32_t s) {
  ed_op g = ed_ext;
  ed_op h = s & 2u ? &ed_ext2 : ed_inc;
  struct ed_ops o = {ed_ext2, &ed_ext};
  ed_op t[2] = {ed_ext, ed_inc};
  ed_vop v = ed_vext;
  uint32_t k = g(s) + h(s) * 3u + o.fn(s) * 5u + o.alt(s) * 7u + t[s & 1u](s) * 11u;
  k += v(s, 2u) * 13u + (s & 4u ? ed_ext : ed_ext2)(s) * 17u + ed_choose(s)(s) * 19u;
  return k;
}
"""
#: the indirect calls `_EXTDESIG_CALLS` makes, each an external edge
_EXTDESIG_SITES = 8


def test_designators_of_functions_another_unit_defines_run_as_the_original():
    """CF-EXTDESIG: `cfront_extdesig_link.c` -- functions the unit only prototypes, named as values (passed to a
    function of this unit and to one of another, held, selected, stored in a member and a table, compared with a
    pointer, another designator and 0), their addresses taken (`&f`), called directly, through `*f` and `(&f)`, and a
    variadic one called with more arguments than it names. Both rails refused every such designator, and `&f` of any
    function; the oracle declared a variadic prototype without its `...`, so its emit did not compile. Each emit now
    declares every function it names as its prototype does, lowers to one claim graph on the four targets, and runs as
    the original with the driver defining the functions. `_EXTDESIG_CALLS` calls through those pointers in place: it
    lowers alike and runs as the original, each of its indirect calls is an external edge on both rails, and both
    rails report its effects and escapes byte for byte."""
    if not _CC:
        return
    fx = "cfront_extdesig_link.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        ext = {
            ln.split("(", 1)[0].split()[-1]: ln
            for ln in emit.splitlines()
            if ln.startswith("extern ")
        }
        assert {"ed_ext", "ed_ext2", "ed_vext", "ed_apply"} <= set(ext), (label, ext)
        assert ext["ed_vext"] == "extern uint32_t ed_vext(uint32_t, ...);", (label, ext)
    # ... each function the oracle emits declares the functions it names, not only those it calls: it compiles
    # standalone (the twin's prelude declares every prototype)
    _s, r_fx, _e = _oracle(src)
    for name in ("ed_pick", "ed_held", "ed_member", "ed_table", "ed_compare", "ed_pass"):
        decls = {ln for ln in r_fx.emitted[name].splitlines() if ln.startswith("extern ")}
        assert "extern uint32_t ed_ext(uint32_t);" in decls, (name, decls)
    _run_against_original_werror(
        fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _EXTDESIG_DRIVER, _EXTDESIG_WERROR
    )
    exe = _build_frontend(_session_build_dir())
    oracle_summary, r, _entry = _oracle(_EXTDESIG_CALLS)
    counts = r.escape.counts()
    assert (counts["indirect"], counts["known"]) == (_EXTDESIG_SITES, 0), counts
    calls_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    from bcir.tests import escape_fixtures as ef  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "extdesig_calls.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_EXTDESIG_CALLS)
        c_summary, c_calls_emit = _c_run(exe, path)
        assert c_summary == oracle_summary and "ok=1" in c_summary, (c_summary, oracle_summary)
        _parity_on_targets(path, _EXTDESIG_CALLS)
        assert ef.rail_parity(_build_bcir_cc(d), [(path, r)]) == (0, 0, 1)
    driver = (
        _GAPS_SAME
        + _EXTDESIG_DEFS
        + "int main(void) {\n  for (unsigned n = 0; n < GAPS_N; n++) SAME(edc_calls, gaps_in[n]);\n"
        + '  puts("MATCH");\n  return 0;\n}\n'
    )
    _run_against_original_werror(
        "extdesig_calls.c",
        _EXTDESIG_CALLS,
        (("twin", c_calls_emit), ("oracle", calls_emit)),
        driver,
        _EXTDESIG_WERROR,
    )


_FN_NOT_LVALUE = "a function designator is not an lvalue"
_SIZEOF_FN = "sizeof of a function designator"
_EXTDESIG_HEAD = (
    "#include <stdint.h>\ntypedef uint32_t (*op_t)(uint32_t);\n"
    "uint32_t ext(uint32_t v);\n"
    "uint32_t ext3(uint32_t a, uint32_t b);\n"
    "uint32_t vext(uint32_t n, ...);\n"
    "static uint32_t inc(uint32_t v) { return v + 1u; }\n"
)
# A function no object can stand for, refused as `*f = v` is: stored to, stepped, measured; a designator before any
# declaration of it; arms of `?:` that point to functions of two types, a prototyped one among them
_EXTDESIG_REFUSED = (
    ("uint32_t f(uint32_t s) { ext = inc; return s; }", _FN_NOT_LVALUE),
    ("uint32_t f(uint32_t s) { ext++; return s; }", _FN_NOT_LVALUE),
    ("uint32_t f(uint32_t s) { --ext; return s; }", _FN_NOT_LVALUE),
    ("uint32_t f(uint32_t s) { ext += 1; return s; }", _FN_NOT_LVALUE),
    ("uint32_t f(uint32_t s) { inc = ext; return s; }", _FN_NOT_LVALUE),
    ("uint32_t f(uint32_t s) { ++inc; return s; }", _FN_NOT_LVALUE),
    ("uint32_t f(uint32_t s) { return (uint32_t)sizeof ext + s; }", _SIZEOF_FN),
    ("uint32_t f(uint32_t s) { return (uint32_t)sizeof(*&ext) + s; }", _SIZEOF_FN),
    ("uint32_t f(uint32_t s) { return (uint32_t)sizeof *ext + s; }", _SIZEOF_FN),
    (
        "uint32_t f(uint32_t s) { op_t g = late; return g(s); }\nuint32_t late(uint32_t v);",
        "use of undeclared identifier 'late'",
    ),
    ("uint32_t f(uint32_t s) { return (s ? ext3 : ext) != 0; }", _FN_SELECTED),
    ("uint32_t f(uint32_t s) { return (s ? vext : ext) != 0; }", _FN_SELECTED),
    ("uint32_t f(uint32_t s) { return (s ? &ext3 : inc) != 0; }", _FN_SELECTED),
)
# ... while every other spelling of a prototyped function's value and address lowers alike
_EXTDESIG_LOWERED = (
    "uint32_t f(uint32_t s) { op_t g = &ext; op_t h = &inc; return (g == h) + (&ext == ext) + s; }",
    "uint32_t f(uint32_t s) { return (uint32_t)sizeof(&ext) + (uint32_t)sizeof(&inc) + s; }",
    "uint32_t f(uint32_t s) { return (*ext)(s) + (&ext)(s) + (&inc)(s); }",
    "uint32_t f(uint32_t s) { op_t p = inc; return (*&ext)(s) + (*&p)(s) + (**&ext)(s) + (&*inc)(s); }",
    "uint32_t f(uint32_t s) { op_t g = *&ext; op_t h = &*inc; return (g == h) + (*&ext == &*ext) + s; }",
    "uint32_t f(uint32_t s) { uint32_t (*g)(uint32_t, ...) = s ? vext : 0; return g != 0; }",
    "static op_t pick(uint32_t s) { return s ? ext : &inc; }\nuint32_t f(uint32_t s) { return pick(s) == ext; }",
    "uint32_t later(uint32_t v);\nuint32_t f(uint32_t s) { op_t g = &later; return g == later; }\n"
    "uint32_t later(uint32_t v) { return v * 3u; }",
)


def test_designators_of_prototyped_functions_refused_and_lowered_alike_on_both_rails():
    """CF-EXTDESIG: a function -- defined or only prototyped -- is no object, so storing to it, stepping it and `sizeof`
    of it are refused on both rails for one reason each, where the oracle had refused `f = g` and `f++` as an
    undeclared identifier and the twin as a parse error; a designator before any declaration of its function is
    undeclared; and arms of `?:` of two function types, a prototyped function's among them, are refused as CF-FNSEL
    refuses any (`_EXTDESIG_REFUSED`). Every other spelling -- `&f` compared, measured and called, `*&f` and `&*f`
    as values and callees, a variadic one beside a null pointer, a prototyped function returned, a function
    prototyped before its definition -- lowers to one claim graph on the four targets (`_EXTDESIG_LOWERED`)."""
    for body, why in _EXTDESIG_REFUSED:
        got = _fptab_refusal(_EXTDESIG_HEAD + body + "\n")
        assert got == why, (body, got, why)
    for body in _EXTDESIG_LOWERED:
        got = _fptab_refusal(_EXTDESIG_HEAD + body + "\n")
        assert got == "", (body, got)
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, (body, why) in enumerate(_EXTDESIG_REFUSED):
            path = os.path.join(d, f"r{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_EXTDESIG_HEAD + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode == 1 and run.stdout.strip() == f"PARSE-ERR {why}", (
                body,
                run.returncode,
                run.stdout[:200],
            )
        for n, body in enumerate(_EXTDESIG_LOWERED):
            path = os.path.join(d, f"l{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_EXTDESIG_HEAD + body + "\n")
            _parity_on_targets(path, _EXTDESIG_HEAD + body + "\n")


# CF-QUALS: the qualifiers below a type's top level -- what a pointer points to, each `*` under the outermost, a function
# pointer's own parameters' and return's -- kept in every function type both rails spell. `cfront_quals_link.c` names
# functions another unit defines, prototyped with them; the driver defines those functions.
_QUALS_DEFS = r"""
uint32_t ql_count(const char *const *v, uint32_t n) {
  uint32_t k = 0u;
  for (uint32_t i = 0u; i < n; i++) k += (uint32_t)v[i][0] * (i + 1u);
  return k;
}
uint32_t ql_first(char *const *v) { return (uint32_t)v[0][0]; }
uint32_t ql_cpp(const char **v) { return (uint32_t)v[0][0] * 2u; }
uint32_t ql_pp(const uint32_t *const *const pp, uint32_t n) { return *pp[0] + *pp[n - 1u] * 3u; }
uint32_t ql_set(char *restrict *pp, uint32_t n) { return (uint32_t)pp[0][0] + n; }
uint32_t ql_apply(uint32_t (*fn)(const uint32_t *), const uint32_t *p) { return fn(p) * 2u + 1u; }
static const uint32_t ql_ext_t[2] = {9u, 4u};
const uint32_t *ql_tab(uint32_t i) { return &ql_ext_t[i & 1u]; }
static const char *const ql_ext_v[2] = {"hi", "yo"};
const char *const *ql_names(uint32_t i) { return &ql_ext_v[i & 1u]; }
uint32_t ql_strs(char *const *v) { return (uint32_t)v[0][0] * 3u; }
uint32_t ql_ops(uint32_t (*const *pp)(uint32_t), uint32_t n) { return pp[0](n) + pp[n - 1u](n * 2u); }
"""
_QUALS_DRIVER = (
    _GAPS_SAME
    + _QUALS_DEFS
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(ql_externs, s); SAME(ql_results, s); SAME(ql_fnptrs, s); SAME(ql_typedefs, s); SAME(ql_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)
# A declaration whose type differs from its prototype's is an error under both compilers; a qualifier dropped from an
# argument, a result or a function pointer is one too under these flags
_QUALS_WERROR = {
    "clang": (
        "-Werror=int-conversion",
        "-Werror=incompatible-pointer-types",
        "-Werror=incompatible-function-pointer-types",
        "-Werror=pointer-integer-compare",
    ),
    "gcc": ("-Werror=int-conversion", "-Werror"),
}
# ... and the forms whose objects the G10 escape rows would count, the test's own unit: locals lent to the functions
# another unit defines, a call through a member by `.`, `->` and a chain, through a call's result and through a pointer
# to a variadic function
_QUALS_CALLS = """#include <stdint.h>
typedef uint32_t (*ql_cnt)(const char *const *, uint32_t);
struct ql_ops { uint32_t (*cnt)(const char *const *v, uint32_t n); const uint32_t *(*get)(uint32_t i); };
struct ql_dev { struct ql_ops *ops; };
uint32_t ql_count(const char *const *v, uint32_t n);
uint32_t ql_first(char *const *v);
uint32_t ql_cpp(const char **v);
uint32_t ql_pp(const uint32_t *const *const pp, uint32_t n);
uint32_t ql_set(char *restrict *pp, uint32_t n);
static const uint32_t qc_k[2] = {5u, 7u};
static uint32_t qc_cnt(const char *const *v, uint32_t n) {
  uint32_t k = 0u;
  for (uint32_t i = 0u; i < n; i++) k += (uint32_t)v[i][0];
  return k;
}
static const uint32_t *qc_pick(uint32_t i) { return &qc_k[i & 1u]; }
static ql_cnt qc_choose(uint32_t s) { return s & 1u ? qc_cnt : &qc_cnt; }
static uint32_t qc_va(const char *const *v, uint32_t n, ...) { return (uint32_t)v[0][0] + n; }
uint32_t qc_calls(uint32_t s) {
  const char *names[2] = {"ab", "c"};
  char a[2] = {'x', 0};
  char *v[1] = {a};
  const char *w[1] = {"r"};
  uint32_t b[2] = {s, 1u};
  const uint32_t *p[2] = {b, b + 1};
  struct ql_ops o = {qc_cnt, qc_pick};
  struct ql_ops *po = &o;
  struct ql_dev d = {&o};
  struct ql_dev *pd = &d;
  uint32_t (*vf)(const char *const *, uint32_t, ...) = qc_va;
  uint32_t k = ql_count(names, 2u) + ql_first(v) * 3u + ql_cpp(w) * 5u + ql_pp(p, 2u) * 7u + ql_set(v, 1u) * 11u;
  k += o.cnt(names, 2u) * 13u + po->cnt(names, 1u) * 17u + pd->ops->cnt(names, 2u) * 19u;
  k += *o.get(s) * 23u + *po->get(s + 1u) * 29u + qc_choose(s)(names, 1u) * 31u + vf(names, 1u, s) * 37u;
  return k;
}
"""
#: what each emit's `extern` declarations of `cfront_quals_link.c`'s callees spell, spaces aside
_QUALS_EXTERNS = (
    "externuint32_tql_count(constchar*const*,uint32_t);",
    "externuint32_tql_first(char*const*);",
    "externuint32_tql_cpp(constchar**);",
    "externuint32_tql_pp(constuint32_t*const*,uint32_t);",
    "externuint32_tql_set(char*restrict*,uint32_t);",
    "externconstuint32_t*ql_tab(uint32_t);",
    "externconstchar*const*ql_names(uint32_t);",
    "externuint32_tql_strs(char*const*);",
)


def test_qualifiers_below_the_top_level_run_as_the_original():
    """CF-QUALS: `cfront_quals_link.c` -- functions another unit defines, prototyped with qualifiers below each
    parameter's and return's top level (`const char *const *`, `char *restrict *`, `const uint32_t *const *const`, a
    qualified result), and function pointers -- typedef'd, inline, a table, a `const` one -- whose parameters keep
    theirs. Both rails had declared each callee with every qualifier past the first level dropped -- a type that
    conflicts with its prototype, so no emit compiled beside the original -- and the oracle refused a qualifier after
    a `*` in a function pointer's parameter list. Each emit now declares every callee as its prototype does (the same
    tokens on both rails), lowers to one claim graph on the four targets, and runs as the original under Clang and
    GCC with a qualifier mismatch an error; an argument C converts to no qualified parameter by itself, and a
    qualified result, are cast at the call. `_QUALS_CALLS` passes locals and calls through members, a call's result
    and a variadic pointer: it lowers alike, runs as the original, and both rails report its effects and escapes
    byte for byte."""
    if not _CC:
        return
    fx = "cfront_quals_link.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    for label, emit in (("twin", c_emit), ("oracle", oracle_emit)):
        ext = {ln.replace(" ", "") for ln in emit.splitlines() if ln.startswith("extern ")}
        missing = [d for d in _QUALS_EXTERNS if d not in ext]
        assert not missing, (label, missing, sorted(ext))
    _run_against_original_werror(
        fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _QUALS_DRIVER, _QUALS_WERROR
    )
    exe = _build_frontend(_session_build_dir())
    oracle_summary, r, _entry = _oracle(_QUALS_CALLS)
    calls_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    from bcir.tests import escape_fixtures as ef  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "quals_calls.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_QUALS_CALLS)
        c_summary, c_calls_emit = _c_run(exe, path)
        assert c_summary == oracle_summary and "ok=1" in c_summary, (c_summary, oracle_summary)
        _parity_on_targets(path, _QUALS_CALLS)
        assert ef.rail_parity(_build_bcir_cc(d), [(path, r)]) == (0, 0, 1)
    driver = (
        _GAPS_SAME
        + _QUALS_DEFS
        + "int main(void) {\n  for (unsigned n = 0; n < GAPS_N; n++) SAME(qc_calls, gaps_in[n]);\n"
        + '  puts("MATCH");\n  return 0;\n}\n'
    )
    _run_against_original_werror(
        "quals_calls.c",
        _QUALS_CALLS,
        (("twin", c_calls_emit), ("oracle", calls_emit)),
        driver,
        _QUALS_WERROR,
    )


_VOLATILE_PTR = "a volatile-qualified pointer is not supported"
_GENERIC_QUALIFIED = "a `_Generic` association of a qualified type is not supported"
_QUAL_DEEP = "a qualified pointer nested more than 8 deep is not supported"
_INDEX_BASE = "unsupported base expression Index"
# The qualified pointers neither rail lowers, each refused for one reason on both: an object that is itself a
# volatile pointer, whose accesses C performs as written and neither rail does; a `_Generic` association of a
# qualified type, which the type of no controlling expression has (an rvalue drops its qualifiers, C11 6.5.1.1p2) and
# which both rails had matched as the unqualified type -- `p` of `char *` selected `const char *:`, a silent
# miscompile; arms of `?:` of function types that differ in a parameter's qualifier; and a qualified `*` past the
# eighth from the base, which the twin's per-level masks have no bit for
_QUALS_HEAD = (
    "#include <stdint.h>\n#include <stdarg.h>\ntypedef char *q_str;\ntypedef uint32_t (*q_op)(uint32_t);\n"
    "typedef uint32_t (*q_cop)(const uint32_t *);\n"
    "static uint32_t q_inc(uint32_t v) { return v + 1u; }\n"
    "static uint32_t q_sum_c(const uint32_t *p) { return p[0] + 1u; }\n"
    "static uint32_t q_sum_m(uint32_t *p) { return p[0] + 2u; }\n"
)
_QUALS_REFUSED = (
    (
        "uint32_t f(uint32_t s) { uint32_t a = s; uint32_t *volatile p = &a; return *p; }",
        _VOLATILE_PTR,
    ),
    ("uint32_t f(uint32_t *volatile *pp) { return **pp; }", _VOLATILE_PTR),
    (
        "uint32_t f(uint32_t s) { char c = (char)s; volatile q_str p = &c; return (uint32_t)*p; }",
        _VOLATILE_PTR,
    ),
    ("uint32_t f(uint32_t s) { q_str volatile p = 0; return p ? 1u : s; }", _VOLATILE_PTR),
    ("uint32_t f(uint32_t s) { volatile q_op g = q_inc; return g(s); }", _VOLATILE_PTR),
    ("uint32_t g(uint32_t *volatile p);\nuint32_t f(uint32_t s) { return s; }", _VOLATILE_PTR),
    ("uint32_t f(uint32_t s) { uint32_t a = s; return *(uint32_t *volatile)&a; }", _VOLATILE_PTR),
    (
        "uint32_t f(uint32_t s) { uint32_t (*volatile g)(uint32_t) = q_inc; return g(s); }",
        _VOLATILE_PTR,
    ),
    (
        "uint32_t f(uint32_t s) { char c = (char)s; char *p = &c;\n"
        "  return (uint32_t)_Generic(p, const char *: 1u, char *: 2u, default: 3u) + s; }",
        _GENERIC_QUALIFIED,
    ),
    (
        "uint32_t f(uint32_t s) { char c = (char)s; const char *p = &c;\n"
        "  return (uint32_t)_Generic(p, char *: 1u, const char *: 2u, default: 3u) + s; }",
        _GENERIC_QUALIFIED,
    ),
    (
        "uint32_t f(uint32_t s) { char c = (char)s; char *p = &c;\n"
        "  return (uint32_t)_Generic(p, char *const: 1u, default: 3u) + s; }",
        _GENERIC_QUALIFIED,
    ),
    (
        "uint32_t f(uint32_t s) { uint32_t x = s;\n"
        "  return (uint32_t)_Generic(x, const uint32_t: 1u, default: 3u) + s; }",
        _GENERIC_QUALIFIED,
    ),
    (
        "uint32_t f(uint32_t s) { uint32_t a[1] = {s}; return (s ? q_sum_c : q_sum_m)(a); }",
        _FN_SELECTED,
    ),
    ("uint32_t f(uint32_t s) { uint32_t ********* const p9 = 0; return p9 ? 1u : s; }", _QUAL_DEEP),
    (
        "typedef uint32_t *********q_p9;\nuint32_t f(uint32_t s) { const q_p9 p = 0; return p ? 1u : s; }",
        _QUAL_DEEP,
    ),
    (
        "uint32_t f(uint32_t s) { uint32_t (**********const fp)(uint32_t) = 0; return fp ? 1u : s; }",
        _QUAL_DEEP,
    ),
    (
        "uint32_t g(uint32_t ********* restrict *v);\nuint32_t f(uint32_t s) { return s; }",
        _QUAL_DEEP,
    ),
)
# ... while every other qualified pointer lowers to one claim graph on the four targets
_QUALS_LOWERED = (
    "uint32_t f(uint32_t s) { char c = (char)s; char *p = &c; char *const *q = &p; return (uint32_t)**q; }",
    "uint32_t f(uint32_t s) { char c = (char)s; char *p = &c; const q_str *q = &p; return (uint32_t)**q; }",
    "uint32_t f(uint32_t s) { char c = (char)s; char *p = &c; return (uint32_t)**(char *const *)&p; }",
    "uint32_t f(uint32_t s) { char c = (char)s; char *p = &c; return (uint32_t)**(const char *const *)&p; }",
    "uint32_t f(uint32_t s) { uint32_t a[1] = {s}; q_cop g = q_sum_c;\n"
    "  uint32_t (*h)(const uint32_t *const) = q_sum_c; return g(a) + h(a); }",
    "uint32_t f(int *restrict *pp) { return (uint32_t)**pp; }",
    "uint32_t f(uint32_t s) { q_op const g = q_inc; uint32_t (*const h)(uint32_t) = q_inc; return g(s) + h(s); }",
    "uint32_t f(uint32_t s) { uint32_t a[1] = {s}; return (s ? q_sum_c : q_sum_c)(a) + (s ? q_sum_m : q_sum_m)(a); }",
    "uint32_t f(uint32_t s) { uint32_t ******** const p8 = 0; return p8 ? 1u : s; }",
    "uint32_t f(uint32_t s) { uint32_t (*********const fp)(uint32_t) = 0; return fp ? 1u : s; }",
    "uint32_t f(uint32_t s) { return (uint32_t)sizeof(const char *const *) + (uint32_t)sizeof(q_str const) + s; }",
    "uint32_t f(uint32_t s) { char c = (char)s; char *p = &c; typeof(char *const *) q = &p; return (uint32_t)**q; }",
    "static uint32_t q_va(uint32_t n, ...) { va_list ap; va_start(ap, n);\n"
    "  const char *const *v = va_arg(ap, const char *const *); va_end(ap); return (uint32_t)v[0][0] + n; }\n"
    'uint32_t f(uint32_t s) { const char *names[1] = {"z"}; return q_va(s, names); }',
    "uint32_t f(uint32_t s) { char c = (char)s; char *p = &c;\n"
    "  return (uint32_t)_Generic(p, char *: 2u, default: 3u) + s; }",
    "static uint32_t q_cnt(const char *const *v) { return (uint32_t)v[0][0]; }\n"
    'uint32_t f(uint32_t s) { const char *names[1] = {"q"}; return q_cnt(names) + s; }',
)
# CF-IDXARROW: a member access straight through an element that is a pointer -- of an array of pointers, through a
# `T **`, a typedef of one, a file-scope array of them, a member array of them, read, stored, stepped, compounded,
# addressed, parenthesized, used as a value -- is refused on both rails for the oracle's reason (it has no subscript
# base). The twin had read the pointer's own slot as the struct it points to, a silent miscompile, and had taken
# `arr[i].m` of such an element alike
_IDX_HEAD = (
    "#include <stdint.h>\nstruct ix { uint32_t v; uint32_t w; };\ntypedef struct ix *ix_p;\n"
    "struct ixh { struct ix *p[2]; struct ix **pp; };\n"
    "static struct ix ix_a = {1u, 2u}, ix_b = {3u, 4u};\n"
    "static struct ix *ix_g[2] = {&ix_a, &ix_b};\n"
)
_IDXARROW_REFUSED = (
    "uint32_t f(uint32_t s) { struct ix *arr[2] = {&ix_a, &ix_b}; return arr[1]->w + s; }",
    "uint32_t f(uint32_t s) { struct ix *arr[2] = {&ix_a, &ix_b}; return arr[s & 1u]->v; }",
    "uint32_t f(struct ix **pp) { return pp[1]->w; }",
    "uint32_t f(uint32_t s) { return ix_g[s & 1u]->w; }",
    "uint32_t f(uint32_t s) { ix_p arr[2] = {&ix_a, &ix_b}; return arr[1]->w + s; }",
    "uint32_t f(ix_p *pp, uint32_t i) { return pp[i]->w + pp[i]->v; }",
    "uint32_t f(uint32_t s) { struct ix *arr[2] = {&ix_a, &ix_b}; arr[1]->w = s; return ix_b.w; }",
    "void f(struct ix **pp, uint32_t x) { pp[1]->v = x; }",
    "uint32_t f(uint32_t s) { struct ix *arr[2] = {&ix_a, &ix_b}; arr[1]->w++; return ix_b.w + s; }",
    "uint32_t f(uint32_t s) { struct ix *arr[2] = {&ix_a, &ix_b}; arr[1]->w += s; return ix_b.w; }",
    "uint32_t f(uint32_t s) { struct ix *arr[2] = {&ix_a, &ix_b}; uint32_t *q = &arr[1]->w; return *q + s; }",
    "uint32_t f(uint32_t s) { struct ix *arr[2] = {&ix_a, &ix_b}; return (arr[1])->w + s; }",
    "uint32_t f(uint32_t s) { struct ix *arr[2] = {&ix_a, &ix_b}; uint32_t r = (arr[1]->w = s); return r; }",
    "uint32_t f(uint32_t s) { struct ix *arr[2] = {&ix_a, &ix_b}; return arr[1].w + s; }",
    "uint32_t f(struct ixh *h) { return h->pp[1]->w; }",
    "uint32_t f(uint32_t s) { struct ixh h = {{&ix_a, &ix_b}, ix_g}; return h.p[1]->w + s; }",
    "uint32_t f(uint32_t s) { struct ixh h = {{&ix_a, &ix_b}, ix_g}; struct ixh *hp = &h; hp->pp[1]->w = s; return ix_b.w; }",
)


def test_qualified_pointers_refused_and_lowered_alike_on_both_rails():
    """CF-QUALS: the qualified pointers neither rail lowers are refused on both for one reason each
    (`_QUALS_REFUSED`): a volatile pointer object -- a declarator, a parameter, a prototype's, a typedef'd pointer or
    function pointer, a cast -- both rails had lowered with its qualifier dropped; a `_Generic` association of a
    qualified type, which both rails had matched as the unqualified one, so `p` of `char *` selected `const char *:`;
    arms of `?:` of function types that differ only in a parameter's qualifier, which both rails had selected
    between; a qualified `*` past the eighth. Every other qualified pointer lowers to one claim graph on the four
    targets (`_QUALS_LOWERED`): a pointer to `const` pointers, a `const` pointer typedef, casts, `typeof`, `sizeof`
    and `va_arg` of qualified types, `restrict` below the top, `const` function pointers, the eighth level."""
    for body, why in _QUALS_REFUSED:
        got = _fptab_refusal(_QUALS_HEAD + body + "\n")
        assert got == why, (body, got, why)
    for body in _QUALS_LOWERED:
        got = _fptab_refusal(_QUALS_HEAD + body + "\n")
        assert got == "", (body, got)
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, (body, why) in enumerate(_QUALS_REFUSED):
            path = os.path.join(d, f"r{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_QUALS_HEAD + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode == 1 and run.stdout.strip() == f"PARSE-ERR {why}", (
                body,
                run.returncode,
                run.stdout[:200],
            )
        for n, body in enumerate(_QUALS_LOWERED):
            path = os.path.join(d, f"l{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_QUALS_HEAD + body + "\n")
            _parity_on_targets(path, _QUALS_HEAD + body + "\n")


_IDXARROW_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(ix_chain_values, s); SAME(ix_steps, s); SAME(ix_addresses, s); SAME(ix_kinds, s); SAME(ix_typedefs, s);
    SAME(ix_globals, s); SAME(ix_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)
# ... and its forms over locals, whose stored addresses the G10 escape rows would count: the test's own unit
_IDXARROW_LOCALS = """#include <stdint.h>
struct ixl { uint32_t v; uint32_t w; };
uint32_t ixl_addresses(uint32_t s) {
  int32_t x = (int32_t)s, y = 2;
  int32_t *arr[2] = {&x, &y};
  void *va[2] = {&x, &y};
  int32_t **q = &arr[1];
  void **vq = &va[s & 1u];
  int32_t **pp = arr;
  int32_t **r = pp + 1;
  return (uint32_t)**q + (q == arr + 1) * 3u + (vq == &va[s & 1u]) * 5u + (uint32_t)*(int32_t *)*vq * 7u
         + (uint32_t)(r - pp) * 11u;
}
uint32_t ixl_chain(uint32_t s) {
  struct ixl a = {s, 1u}, b = {s + 7u, 2u};
  struct ixl *arr[2] = {&a, &b};
  struct ixl **pp = arr;
  uint32_t r = (pp[1][0].w = s + 9u);
  r += (pp[0][0].v += 3u);
  r += pp[1][0].w++;
  r += ++pp[0][0].v;
  return r + a.v + b.w;
}
"""


def test_elements_that_are_pointers_run_as_the_original():
    """CF-IDXARROW: `cfront_idxarrow.c` -- elements that are pointers: `pp[i][j].f` assigned, compounded and stepped as
    a value through a `T **`; `pp + 1`, `pp++`, `arr + 1` and a select of `T **`; `&arr[i]` of an array of pointers; a
    `char **`, a `double **`, a pointer to a struct-pointer typedef; a file-scope array of pointers read through. The
    twin had read `pp[1][0].f` at the flat index `1*1+0` of `pp`, typed `pp + 1` and `arr + 1` as one pointer level
    less and `&arr[i]` as an element of the pointee's width, and read a file-scope array of pointers as integers;
    both rails now lower the unit to one claim graph on the four targets and each emit returns what the original
    does, function by function. `_IDXARROW_LOCALS` does the same over locals. A member access straight through such
    an element is refused on both rails for one reason (`_IDXARROW_REFUSED`)."""
    for body in _IDXARROW_REFUSED:
        got = _fptab_refusal(_IDX_HEAD + body + "\n")
        assert got == _INDEX_BASE, (body, got)
    if not _CC:
        return
    fx = "cfront_idxarrow.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    _run_against_original_werror(
        fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _IDXARROW_DRIVER, _QUALS_WERROR
    )
    exe = _build_frontend(_session_build_dir())
    oracle_summary, r, _entry = _oracle(_IDXARROW_LOCALS)
    locals_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "idxarrow_locals.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_IDXARROW_LOCALS)
        c_summary, c_locals_emit = _c_run(exe, path)
        assert c_summary == oracle_summary and "ok=1" in c_summary, (c_summary, oracle_summary)
        _parity_on_targets(path, _IDXARROW_LOCALS)
        for n, body in enumerate(_IDXARROW_REFUSED):
            rpath = os.path.join(d, f"r{n}.c")
            with open(rpath, "w", encoding="utf-8") as fh:
                fh.write(_IDX_HEAD + body + "\n")
            run = subprocess.run([exe, rpath], capture_output=True, text=True)
            assert run.returncode == 1 and run.stdout.strip() == f"PARSE-ERR {_INDEX_BASE}", (
                body,
                run.returncode,
                run.stdout[:200],
            )
    driver = (
        _GAPS_SAME
        + "int main(void) {\n  for (unsigned n = 0; n < GAPS_N; n++) {\n"
        + "    SAME(ixl_addresses, gaps_in[n]); SAME(ixl_chain, gaps_in[n]);\n  }\n"
        + '  puts("MATCH");\n  return 0;\n}\n'
    )
    _run_against_original_werror(
        "idxarrow_locals.c",
        _IDXARROW_LOCALS,
        (("twin", c_locals_emit), ("oracle", locals_emit)),
        driver,
        _QUALS_WERROR,
    )


# CF-RTFP: casts to function-pointer and `_Atomic` types, and the forms the emit writes for them -- a function pointer
# stored through a generic slot at a byte offset, an `_Atomic` member reached at one; a typedef'd table of function
# pointers, a compound literal of one, `sizeof` of an array compound literal, `( E ) = v;`, a braced function-pointer
# initializer.
_RTFP_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(rf_casts, s); SAME(rf_slots, s); SAME(rf_tables, s); SAME(rf_sizes, s); SAME(rf_braced, s);
    SAME(rf_parens, s); SAME(rf_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)
# ... and its calls through a table or a select of two functions, through a function pointer read from memory and a
# null one guarded, which the G10 rows would count: the test's own unit
_RTFP_CALLS = """#include <stdint.h>
typedef uint32_t (*rc_op)(uint32_t);
typedef uint32_t (*rc_tab[2])(uint32_t);
struct rc_slot { rc_op fn; uint32_t tag; };
static uint32_t rc_inc(uint32_t v) { return v + 1u; }
static uint32_t rc_dbl(uint32_t v) { return v * 2u; }
uint32_t rc_tables(uint32_t s) {
  rc_tab t = {rc_inc, rc_dbl};
  __typeof__(t[0]) g = t[1];
  uint32_t r = t[s & 1u](s) + g(s) * 3u;
  r += (rc_op[2]){rc_dbl, rc_inc}[s & 1u](s) * 5u;
  r += (uint32_t (*[2])(uint32_t)){rc_inc, rc_dbl}[(s >> 1) & 1u](s) * 7u;
  rc_op *q = (rc_op[2]){rc_inc, rc_dbl};
  return r + q[s & 1u](s) * 11u;
}
uint32_t rc_selects(uint32_t s) {
  rc_op k = (s & 1u) ? (rc_op)rc_inc : (uint32_t (*)(uint32_t))rc_dbl;
  uint32_t (*f)(uint32_t) = {rc_inc};
  rc_op g = {rc_dbl,};
  rc_op w = (s & 2u) ? f : g;
  uint32_t (*z)(uint32_t) = 0;
  return k(s) + w(s) * 3u + (z ? z(s) : 5u) * 5u;
}
uint32_t rc_slots(uint32_t s) {
  struct rc_slot o = {rc_inc, 1u};
  *(void (**)(void))((char *)&o + 0) = (void (*)(void))rc_dbl;
  rc_op f = o.fn;
  uint32_t r = f(s) + o.fn(s) * 3u;
  struct rc_slot *p = &o;
  *(void (**)(void))((char *)p + 0) = (void (*)(void))rc_inc;
  return r + p->fn(s) * 5u + o.tag;
}
"""
# The shared head of the units the two rails refuse alike, and of those they lower alike.
_RTFP_HEAD = """#include <stdint.h>
static uint32_t inc(uint32_t v) { return v + 1u; }
static uint32_t dbl(uint32_t v) { return v * 2u; }
typedef uint32_t (*op_t)(uint32_t);
typedef uint32_t row_t[2];
typedef uint32_t (*tab_t[2])(uint32_t);
struct am { uint32_t k; _Atomic uint32_t v; };
static row_t grow = {3u, 4u};
"""
_TYPE_NAME_NAMED = "a type name names an identifier"
_CAST_ARRAY = "a cast to an array type"
_BRACED_SCALAR = "a braced scalar initializer holds one expression"
_SIZEOF_UNSIZED = "sizeof of an incomplete type"
_NOT_CALLABLE = "called object is not a function or function pointer"
# Refused on both rails for one reason each: a type name that names an identifier (C11 6.7.7p1); a cast to an array
# type (6.5.4p2), which the oracle had lowered as the pointer the array decays to and the twin had refused as an
# undeclared name, and both had lowered for an array of function pointers, to two claim graphs; a braced
# function-pointer initializer of more than one expression (6.7.9p11); `sizeof` of a compound literal its initializer
# sizes, which the twin had given the size of a pointer; a call through an element that is no function pointer.
_RTFP_REFUSED = (
    (
        "uint32_t f(uint32_t s) { op_t g = (uint32_t (*p)(uint32_t))inc; return g(s); }",
        _TYPE_NAME_NAMED,
    ),
    (
        "uint32_t f(uint32_t s) { return (uint32_t)sizeof(uint32_t (*p)(uint32_t)) + s; }",
        _TYPE_NAME_NAMED,
    ),
    ("uint32_t f(uint32_t s) { uint32_t *p = (uint32_t[2])s; return p ? s : 0u; }", _CAST_ARRAY),
    ("uint32_t f(uint32_t s) { uint32_t *p = (row_t)grow; return p[0] + s; }", _CAST_ARRAY),
    (
        "uint32_t f(uint32_t s) { op_t g = (uint32_t (*[2])(uint32_t))inc; return g(s); }",
        _CAST_ARRAY,
    ),
    ("uint32_t f(uint32_t s) { op_t g = (tab_t)inc; return g(s); }", _CAST_ARRAY),
    ("uint32_t f(uint32_t s) { uint32_t **p = (uint32_t *[2])0; return p ? s : 0u; }", _CAST_ARRAY),
    ("uint32_t f(uint32_t s) { op_t g = {{inc}}; return g(s); }", _BRACED_SCALAR),
    (
        "uint32_t f(uint32_t s) { uint32_t (*g)(uint32_t) = {inc, dbl}; return g(s); }",
        _BRACED_SCALAR,
    ),
    (
        "uint32_t f(uint32_t s) { return (uint32_t)sizeof((uint32_t[]){1u, 2u}) + s; }",
        _SIZEOF_UNSIZED,
    ),
    (
        "uint32_t f(uint32_t s) { return (uint32_t)sizeof (uint32_t[]){1u, 2u} + s; }",
        _SIZEOF_UNSIZED,
    ),
    ("uint32_t f(uint32_t s) { return (uint32_t[2]){1u, 2u}[s & 1u](s); }", _NOT_CALLABLE),
)
# Lowered alike on the four targets, each emit the original where `f` takes a `uint32_t`: casts to function-pointer
# types and to pointers to `_Atomic` objects; braced and empty function-pointer initializers; typedefs of a table of function pointers -- a
# local, a global, a parameter, a member, two dimensions -- and of a pointer to one; compound literals of function
# pointers; `typeof` of an element and a member that are function pointers; `sizeof` of arrays of function pointers
# and of compound literals; `( E ) = v;`; a unit that declares a standard type itself.
_RTFP_LOWERED = (
    "uint32_t f(uint32_t s) { op_t g = (op_t)inc; op_t z = (op_t)0; return g(s) + (z == 0); }",
    "uint32_t f(uint32_t s) { void (*g)(void) = (void (*)(void))dbl; op_t p = (uint32_t (*)(uint32_t))g; return p(s); }",
    "static op_t gp = inc;\n"
    "uint32_t f(uint32_t s) { void *v = &gp; uint32_t (**pp)(uint32_t) = (uint32_t (**)(uint32_t))v; return (*pp)(s); }",
    "uint32_t f(uint32_t s) {\n"
    "  uint32_t *q = (uint32_t *)0; _Atomic uint32_t *r = (_Atomic uint32_t *)0; return s + (q == 0) + (r == 0);\n}",
    "uint32_t f(uint32_t s) { struct am a; a.k = s; *(_Atomic(uint32_t) *)((char *)&a + 4) = s + 2u; return a.v + a.k; }",
    "uint32_t f(uint32_t s) { uint32_t (*fp)(uint32_t) = {}; op_t g = {}; return (fp ? fp(s) : 1u) + (g ? g(s) : 2u); }",
    "uint32_t f(uint32_t s) { uint32_t k = {s}; uint32_t *p = {0}; return k + (p == 0); }",
    "uint32_t f(uint32_t s) { tab_t t = {inc, dbl}; return t[s & 1u](s) + (uint32_t)sizeof t; }",
    "static tab_t gt = {inc, dbl};\nuint32_t f(uint32_t s) { return gt[s & 1u](s) + (uint32_t)sizeof gt; }",
    "static uint32_t ap(tab_t t, uint32_t s) { return t[s & 1u](s); }\n"
    "uint32_t f(uint32_t s) { tab_t t = {inc, dbl}; return ap(t, s); }",
    "struct ops2 { tab_t t; uint32_t k; };\n"
    "uint32_t f(uint32_t s) { struct ops2 o = {{inc, dbl}, 3u}; return o.t[s & 1u](s) + o.k; }",
    "typedef uint32_t (**pp_t)(uint32_t);\nuint32_t f(uint32_t s) { op_t g = inc; pp_t p = &g; return (*p)(s); }",
    "typedef uint32_t (*tab2_t[2][2])(uint32_t);\n"
    "uint32_t f(uint32_t s) { tab2_t t = {{inc, dbl}, {dbl, inc}}; return t[s & 1u][1](s); }",
    "uint32_t f(uint32_t s) { return (tab_t){inc, dbl}[s & 1u](s); }",
    "uint32_t f(uint32_t s) { return (op_t[2][2]){{inc, dbl}, {dbl, inc}}[s & 1u][1](s); }",
    "uint32_t f(uint32_t s) { return (op_t[]){inc, dbl, inc}[s % 3u](s); }",
    "uint32_t f(uint32_t s) { op_t t[2] = {inc, dbl}; __typeof__(t[0]) g = t[1]; return g(s); }",
    "struct ops3 { op_t fn; uint32_t k; };\n"
    "uint32_t f(uint32_t s) { struct ops3 o = {inc, 1u}; __typeof__(o.fn) g = o.fn; return g(s) + o.k; }",
    "uint32_t f(uint32_t s) {\n"
    "  return (uint32_t)sizeof(uint32_t (*[2][3])(uint32_t)) + (uint32_t)_Alignof(uint32_t (*[3])(uint32_t)) + s;\n}",
    "uint32_t f(uint32_t s) {\n"
    "  return (uint32_t)sizeof((op_t[3]){inc, dbl, inc}) + (uint32_t)sizeof (op_t[2]){inc, dbl} + s;\n}",
    "uint32_t f(uint32_t s) {\n"
    "  return (uint32_t)sizeof(0, (uint32_t[]){1u, 2u, 3u}) + (uint32_t)sizeof((uint32_t[]){1u, 2u}[1]) + s;\n}",
    "uint32_t f(uint32_t s) { return (uint32_t)sizeof((struct am[2]){{1u, 2u}}[1].v) + (uint32_t)sizeof grow + s; }",
    "uint32_t f(uint32_t *q, uint32_t s) { (*q++) = s; ((*q)) += s; (q[-1]) <<= 1u; return *q; }",
    "uint32_t f(struct am *p, uint32_t s) {\n"
    "  (*(_Atomic uint32_t *)((char *)p + 4)) = s; (*(_Atomic uint32_t *)((char *)p + 4)) += s; return p->k;\n}",
    "typedef unsigned int uint32_t;\nuint32_t f(uint32_t s) { return s + 1u; }",
    # a function pointer stored to a member -- an initializer's, a statement's, an element of a member table's --
    # then read back through the member: each emit stores it through a pointer to its own type, which a read of the
    # member's type may alias (C11 6.5p7); through `void (**)(void)`, GCC at -O2 dropped the store and the emit
    # called a null pointer, and the twin's store into a member table did not compile
    "struct fsl { op_t fn; uint32_t k; };\n"
    "uint32_t f(uint32_t s) {\n"
    "  struct fsl a[2] = {{inc, 1u}, {dbl, 2u}}; struct fsl *e = &a[s & 1u]; return e->fn(s) + a[1].k;\n}",
    "struct fsl { op_t fn; uint32_t k; };\n"
    "uint32_t f(uint32_t s) {\n"
    "  struct fsl o; o.k = 1u; o.fn = inc; op_t g = dbl; struct fsl *p = &o;\n"
    "  if (s & 1u) p->fn = g;\n  return p->fn(s) + o.k;\n}",
    "struct tb { op_t t[2]; uint32_t k; };\n"
    "uint32_t f(uint32_t s) { struct tb o = {{inc, inc}, 1u}; o.t[1] = dbl; return o.t[s & 1u](s) + o.k; }",
    # a compound literal of a pointer or function-pointer type: a pointer of its type -- the twin's had been a
    # `uint32_t` the pointer was cut into, so `*(T *){p}` read no pointer and `&(T *){p}` was a `T *` -- and `0` or
    # `{}` in one a null pointer, which both rails had held in an `int`
    "static uint32_t gv = 7u;\n"
    "uint32_t f(uint32_t s) {\n"
    "  uint32_t *q = (uint32_t *){&gv}; uint32_t **pp = &(uint32_t *){&gv}; return *q + **pp + *(uint32_t *){&gv} + s;\n}",
    "uint32_t f(uint32_t s) {\n"
    "  _Atomic uint32_t *q = (_Atomic uint32_t *){0}; uint32_t *p = (uint32_t *){}; op_t z = (op_t){0};\n"
    "  return (q == 0) + (p == 0) + (z == 0) + s;\n}",
    "uint32_t f(uint32_t s) { op_t h = (op_t){dbl}; return h(s) + (op_t){inc}(s); }",
)
# ... and units that declare `size_t` themselves, which the oracle had refused (it read `size_t` as a keyword of the
# specifier before it, `unsigned long size_t`)
_RTFP_UNITS = (
    "typedef unsigned long size_t;\ntypedef unsigned int uint32_t;\n"
    "uint32_t f(uint32_t s) { size_t n = s; return (uint32_t)(n + sizeof n); }\n",
    "typedef unsigned long size_t;\nsize_t f(size_t n) { return n * 2u; }\n",
)


def _rtfp_run_unit(name: str, unit: str, exe: str, d: str) -> None:
    """A unit both rails lower alike on the four targets; where its `f` takes a `uint32_t`, each emit returns what
    the original does under Clang and GCC, with the mixes `_QUALS_WERROR` names made errors."""
    path = os.path.join(d, f"{name}.c")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(unit)
    _parity_on_targets(path, unit)
    if not re.search(r"^uint32_t f\(uint32_t s\)", unit, re.M):
        return
    oracle_summary, r, _entry = _oracle(unit)
    c_summary, c_emit = _c_run(exe, path)
    assert c_summary == oracle_summary and "ok=1" in c_summary, (name, c_summary, oracle_summary)
    oracle_emit = "\n".join(r.emitted[n] for n in r.lowered.functions)
    driver = (
        _GAPS_SAME
        + "int main(void) {\n  for (unsigned n = 0; n < GAPS_N; n++) SAME(f, gaps_in[n]);\n"
        + '  puts("MATCH");\n  return 0;\n}\n'
    )
    _run_against_original_werror(
        name, unit, (("twin", c_emit), ("oracle", oracle_emit)), driver, _QUALS_WERROR
    )


def test_casts_to_function_pointer_and_atomic_types_run_as_the_original():
    """CF-RTFP: `cfront_rtfp.c` -- casts to a function-pointer type, `(R (*)(P))f`, `(op_t)f`, `(op_t)0`, in a select;
    the emit's own forms for a function-pointer member and an `_Atomic` one, a store through a generic slot at a byte
    offset, `*(void (**)(void))((char *)p + 0) = (void (*)(void))f;`, and `(*(_Atomic uint32_t *)((char *)a + 4))`
    stored, compounded, stepped and read; a typedef of a table of function pointers, its `sizeof`, a compound literal
    of one and `typeof` its element; `sizeof` of array compound literals; braced and null function-pointer
    initializers; `( E ) = v;` and `( E ) OP= v;`. Both rails had refused the casts and the slot store; the twin had
    refused the typedef, the parenthesized targets and the braced initializer, given `sizeof` of an array compound
    literal the size of a pointer, and emitted `op_t t = 0` as an integer. Both rails lower the unit to one claim graph
    on the four targets and each emit, built with Clang's and GCC's mixes made errors, returns what the original does,
    function by function. `_RTFP_CALLS` does the same for calls through tables and selects of two functions and
    through a function pointer read from memory."""
    if not _CC:
        return
    fx = "cfront_rtfp.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    _run_against_original_werror(
        fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _RTFP_DRIVER, _QUALS_WERROR
    )
    exe = _build_frontend(_session_build_dir())
    oracle_summary, r, _entry = _oracle(_RTFP_CALLS)
    calls_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rtfp_calls.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_RTFP_CALLS)
        c_summary, c_calls_emit = _c_run(exe, path)
        assert c_summary == oracle_summary and "ok=1" in c_summary, (c_summary, oracle_summary)
        _parity_on_targets(path, _RTFP_CALLS)
    driver = (
        _GAPS_SAME
        + "int main(void) {\n  for (unsigned n = 0; n < GAPS_N; n++) {\n"
        + "    SAME(rc_tables, gaps_in[n]); SAME(rc_selects, gaps_in[n]); SAME(rc_slots, gaps_in[n]);\n  }\n"
        + '  puts("MATCH");\n  return 0;\n}\n'
    )
    _run_against_original_werror(
        "rtfp_calls.c",
        _RTFP_CALLS,
        (("twin", c_calls_emit), ("oracle", calls_emit)),
        driver,
        _QUALS_WERROR,
    )


def test_function_pointer_types_refused_and_lowered_alike_on_both_rails():
    """CF-RTFP: the forms neither rail lowers are refused on both for one reason each (`_RTFP_REFUSED`): a type name
    that names an identifier, a cast to an array type, a braced function-pointer initializer of two expressions,
    `sizeof` of a compound literal its initializer sizes, a call through an element that is no function pointer. Every
    other form of the slice lowers to one claim graph on the four targets, each emit the original where it runs
    (`_RTFP_LOWERED`, `_RTFP_UNITS`)."""
    for body, why in _RTFP_REFUSED:
        got = _fptab_refusal(_RTFP_HEAD + body + "\n")
        assert got == why, (body, got, why)
    for body in _RTFP_LOWERED:
        got = _fptab_refusal(_RTFP_HEAD + body + "\n")
        assert got == "", (body, got)
    for unit in _RTFP_UNITS:
        assert _fptab_refusal(unit) == "", unit
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, (body, why) in enumerate(_RTFP_REFUSED):
            path = os.path.join(d, f"r{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_RTFP_HEAD + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode == 1 and run.stdout.strip() == f"PARSE-ERR {why}", (
                body,
                run.returncode,
                run.stdout[:200],
            )
        for n, body in enumerate(_RTFP_LOWERED):
            _rtfp_run_unit(f"l{n}", _RTFP_HEAD + body + "\n", exe, d)
        for n, unit in enumerate(_RTFP_UNITS):
            _rtfp_run_unit(f"u{n}", unit, exe, d)


# CF-VOIDVAL, CF-STRUCTCOND, CF-UNARY (card 5): a void expression's value used, a struct or union where C requires a
# scalar -- a controlling expression, the operand of `+`, `__real__`, `++` -- and the operands C gives each unary
# operator. Both rails had lowered each refused form below to one claim graph and an emit no compiler takes, or refused
# it for reasons of their own; the lowered forms had split the rails, or lowered alike to an emit that ran wrong.
_VOID_VALUE = "the value of a void expression is used"
_UNARY_NOT = "invalid argument type to unary expression"
_SWITCH_NOT = "statement requires expression of integer type"
_INCDEC_FOLLOW = "inc/dec of this lvalue form is a follow-on"
_CARD5_HEAD = """#include <stdint.h>
#include <stdlib.h>
#include <complex.h>
struct s { uint32_t in; uint32_t o; };
struct h { struct s in; uint32_t k; };
union u { uint32_t w; float f; };
struct bf { unsigned a : 3; signed b : 4; };
typedef void (*vcb_t)(uint32_t);
typedef uint32_t (*op_t)(uint32_t);
static uint32_t g_n;
static uint32_t gv = 5u;
static void vd(uint32_t x) { g_n += x; }
static uint32_t inc(uint32_t v) { return v + 1u; }
void vext(uint32_t x);
static struct s g_a[2];
static uint32_t garr[4];
"""
_UNARYOPS_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(uo_plus, s); SAME(uo_parts, s); SAME(uo_bitfields, s); SAME(uo_addr, s); SAME(uo_null, s);
    SAME(uo_conds, s); SAME(uo_gconds, s); SAME(uo_steps, s); SAME(uo_voids, s); SAME(uo_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)
# Refused on both rails for the reason each names (the unit, the reason)
_CARD5_REFUSED = (
    # a void expression's value used (C11 6.3.2.2): an initializer, an assignment, an operand, a cast, a condition, an
    # argument, an index, the value of `,`, of `?:` and of a statement expression, a `return` from a function that
    # returns one -- of a void function called directly, through a pointer or prototyped
    ("uint32_t f(uint32_t s) { uint32_t k = vd(s); return k; }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { uint32_t k = 1u; k = vd(s); return k; }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { return vd(s) + 1u; }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { return !vd(s); }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { return -vd(s); }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { return (uint32_t)vd(s); }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { vcb_t p = vd; uint32_t k = p(s); return k; }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { vcb_t p = vd; return (*p)(s) + 1u; }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { return vext(s) * 2u; }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { if (vd(s)) return 1u; return 0u; }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { while (vd(s)) return 1u; return 0u; }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { for (; vd(s);) return 1u; return 0u; }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { do { s++; } while (vd(s)); return s; }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { switch (vd(s)) { default: return 1u; } }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { return vd(s) ? 1u : 2u; }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { uint32_t k = s ? vd(s) : vd(1u); return k; }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { return s && vd(s); }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { return vd(s) || s; }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { return vd(s); }", _VOID_VALUE),
    (
        "static uint32_t id(uint32_t v) { return v; }\nuint32_t f(uint32_t s) { return id(vd(s)); }",
        _VOID_VALUE,
    ),
    ("uint32_t f(uint32_t s) { return garr[vd(s)]; }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { uint32_t k = (s++, vd(s)); return k; }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { return vd(s) == 0; }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { uint32_t k = 1u; k += vd(s); return k; }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { uint32_t k = ({ vd(s); }); return k; }", _VOID_VALUE),
    ("uint32_t f(uint32_t s) { return (uint32_t)(void)s; }", _VOID_VALUE),
    # a struct or union as a controlling expression (6.8.4.1p1, 6.8.5p2, 6.5.15p2) or the operand of `+`, `__real__`,
    # `__imag__`, `++` and `--` (6.5.3.3p1, 6.5.2.4p1) -- in memory too, and under `sizeof`, `typeof` and `_Generic`
    ("uint32_t f(struct s a) { if (a) return 1u; return 0u; }", _STRUCT_OPERAND),
    ("uint32_t f(struct s a) { while (a) return 1u; return 0u; }", _STRUCT_OPERAND),
    ("uint32_t f(struct s a) { for (; a;) return 1u; return 0u; }", _STRUCT_OPERAND),
    ("uint32_t f(struct s a) { do { a.in++; } while (a); return a.in; }", _STRUCT_OPERAND),
    ("uint32_t f(struct s a) { return a ? 1u : 0u; }", _STRUCT_OPERAND),
    ("uint32_t f(struct s a) { switch (a) { default: return 1u; } }", _STRUCT_OPERAND),
    ("uint32_t f(union u a) { if (a) return 1u; return 0u; }", _STRUCT_OPERAND),
    ("uint32_t f(struct h a) { if (a.in) return 1u; return 0u; }", _STRUCT_OPERAND),
    ("uint32_t f(struct s *p) { if (*p) return 1u; return 0u; }", _STRUCT_OPERAND),
    ("uint32_t f(struct s a) { struct s b = +a; return b.in; }", _STRUCT_OPERAND),
    ("uint32_t f(struct s a) { return (uint32_t)__real__ a; }", _STRUCT_OPERAND),
    ("uint32_t f(struct s a) { return (uint32_t)__imag__ a; }", _STRUCT_OPERAND),
    ("void f(struct s *p) { (*p)++; }", _STRUCT_OPERAND),
    ("void f(struct s *p) { p[0]++; }", _STRUCT_OPERAND),
    ("void f(struct h *p) { p->in++; }", _STRUCT_OPERAND),
    ("void f(struct h *q) { struct h h = *q; h.in++; *q = h; }", _STRUCT_OPERAND),
    ("void f(void) { g_a[1]++; }", _STRUCT_OPERAND),
    ("void f(struct s *p) { ++*p; }", _STRUCT_OPERAND),
    ("void f(struct s *p) { p[1]--; }", _STRUCT_OPERAND),
    ("uint32_t f(struct s *p) { return (uint32_t)sizeof((*p)++); }", _STRUCT_OPERAND),
    (
        "uint32_t f(uint32_t s) { struct s a = {s, 1u}; return (uint32_t)sizeof(-a); }",
        _STRUCT_OPERAND,
    ),
    (
        "uint32_t f(uint32_t s) { struct s a = {s, 1u}; return (uint32_t)sizeof(!a); }",
        _STRUCT_OPERAND,
    ),
    (
        "uint32_t f(uint32_t s) { struct s a = {s, 1u}; return _Generic(+a, int: 1u, default: 2u); }",
        _STRUCT_OPERAND,
    ),
    (
        "uint32_t f(uint32_t s) { struct s a = {s, 1u}; __typeof__(-a) k = 0; return (uint32_t)k; }",
        _STRUCT_OPERAND,
    ),
    # ... and `!`, `__real__` and `__imag__`, whose `typeof` and `_Generic` typing asks the same predicate
    (
        "uint32_t f(uint32_t s) { struct s a = {s, 1u}; return _Generic(!a, int: 1u, default: 2u); }",
        _STRUCT_OPERAND,
    ),
    (
        "uint32_t f(uint32_t s) { struct s a = {s, 1u}; return _Generic(__imag__ a, int: 1u, default: 2u); }",
        _STRUCT_OPERAND,
    ),
    (
        "uint32_t f(uint32_t s) { uint32_t *p = &gv; __typeof__(__real__ p) k = 0; return (uint32_t)k + s; }",
        _UNARY_NOT,
    ),
    (
        "uint32_t f(uint32_t s) { struct s a = {s, 1u}; return (uint32_t)sizeof(__real__ a); }",
        _STRUCT_OPERAND,
    ),
    # a pointer, a function or an array as the operand of `+`, `-`, `~`, `__real__`; a real floating value under `~`
    # (6.5.3.3p1) -- under `sizeof`, `typeof` and `_Generic` too
    ("uint32_t f(uint32_t s) { uint32_t *p = &gv; uint32_t *q = +p; return *q + s; }", _UNARY_NOT),
    (
        "uint32_t f(uint32_t s) { uint32_t *p = &gv; return (uint32_t)(uintptr_t)-p + s; }",
        _UNARY_NOT,
    ),
    (
        "uint32_t f(uint32_t s) { uint32_t *p = &gv; return (uint32_t)(uintptr_t)~p + s; }",
        _UNARY_NOT,
    ),
    ("uint32_t f(uint32_t s) { return (uint32_t)(uintptr_t)+inc + s; }", _UNARY_NOT),
    ("uint32_t f(uint32_t s) { return (uint32_t)(uintptr_t)-inc + s; }", _UNARY_NOT),
    ("uint32_t f(uint32_t s) { return (uint32_t)(uintptr_t)-garr + s; }", _UNARY_NOT),
    ("uint32_t f(uint32_t s) { double d = s; return (uint32_t)~d; }", _UNARY_NOT),
    (
        "uint32_t f(uint32_t s) { uint32_t *p = &gv; double r = __real__ p; return (uint32_t)r + s; }",
        _UNARY_NOT,
    ),
    ("uint32_t f(uint32_t s) { uint32_t *p = &gv; return (uint32_t)sizeof(-p) + s; }", _UNARY_NOT),
    ("uint32_t f(uint32_t s) { double d = s; return (uint32_t)sizeof(~d) + s; }", _UNARY_NOT),
    (
        "uint32_t f(uint32_t s) { uint32_t *p = &gv; return _Generic(+p, int: 1u, default: 2u) + s; }",
        _UNARY_NOT,
    ),
    (
        "uint32_t f(uint32_t s) { double d = s; __typeof__(~d) k = 0; return (uint32_t)k + s; }",
        _UNARY_NOT,
    ),
    # a `switch` of no integer: a pointer, a floating value, an array, a function (6.8.4.2p1)
    (
        "uint32_t f(uint32_t s) { uint32_t *p = &gv; switch (p) { default: return s; } }",
        _SWITCH_NOT,
    ),
    (
        "uint32_t f(uint32_t s) { double d = s; switch (d) { case 1: return 2u; default: return 1u; } }",
        _SWITCH_NOT,
    ),
    ("uint32_t f(uint32_t s) { switch (garr) { default: return s; } }", _SWITCH_NOT),
    ("uint32_t f(uint32_t s) { switch (inc) { default: return s; } }", _SWITCH_NOT),
    # a step of an element of a table of function pointers (6.5.6p2), which the rails had refused for reasons of their
    # own (CF-FPTAB): the twin, as the oracle, refuses a step it does not take
    ("uint32_t f(uint32_t s) { op_t t[2] = {inc, inc}; (*t)++; return t[0](s); }", _INCDEC_FOLLOW),
    ("uint32_t f(uint32_t s) { op_t t[2] = {inc, inc}; t[0]++; return t[0](s); }", _INCDEC_FOLLOW),
)
# Lowered alike on the four targets, each emit the original
_CARD5_LOWERED = (
    # a void expression where C reads no value: a statement, `(void)e`, the left of `,`, the arms of a statement
    # `?:`, a `for`'s initializer and step, a statement expression's last statement, `return f();` of a void function,
    # a call through a pointer
    "uint32_t f(uint32_t s) { g_n = 0u; vd(s); (void)vd(1u); uint32_t k = (vd(2u), s + 1u); return k + g_n; }",
    "uint32_t f(uint32_t s) { g_n = 0u; s ? vd(s) : vd(1u); s ? vd(3u) : (void)0; return g_n; }",
    "uint32_t f(uint32_t s) { uint32_t i = 0u; g_n = 0u; for (vd(s); i < 2u; i++) { vd(i); } return g_n; }",
    "uint32_t f(uint32_t s) { uint32_t i; g_n = 0u; for (i = s & 1u; i < 3u; vd(i)) { i++; } return g_n + i; }",
    "static void w(uint32_t s) { return vd(s); }\n"
    "uint32_t f(uint32_t s) { g_n = 0u; w(s); ({ vd(1u); }); return g_n; }",
    "uint32_t f(uint32_t s) { vcb_t p = vd; g_n = 0u; p(s); (*p)(s); return g_n; }",
    # `+a`: the promoted operand, which `sizeof`, `_Generic` and `typeof` read
    "uint32_t f(uint32_t s) { uint8_t c = (uint8_t)s; return (uint32_t)sizeof(+c) + (uint32_t)(+c << 23 >> 23) + (uint32_t)(+c - 256 < 0); }",
    "uint32_t f(uint32_t s) { _Bool b = s & 1u; return (uint32_t)sizeof(+b) * 7u + (uint32_t)+b; }",
    "uint32_t f(uint32_t s) { uint16_t h = (uint16_t)s; return _Generic(+h, int: 1u, unsigned: 2u, default: 3u); }",
    "uint32_t f(uint32_t s) { uint16_t h = (uint16_t)s; __typeof__(+h) k = -1; return (uint32_t)(k < 0) + s; }",
    "uint32_t f(uint32_t s) {\n"
    "  char c = (char)s; return (uint32_t)sizeof(+c) + (uint32_t)(+c) + (uint32_t)sizeof(+ +c);\n}",
    "uint32_t f(uint32_t s) {\n"
    "  uint64_t v = s; double d = s; float x = (float)s;\n"
    "  return (uint32_t)(+v >> 1) + (uint32_t)(+d * 0.5) + (uint32_t)sizeof(+x);\n}",
    "uint32_t f(uint32_t s) { __atomic_thread_fence(+(__ATOMIC_ACQUIRE)); return +5u + s; }",
    # `-z` and `~z` of a complex are complex; `__real__` and `__imag__` are a part (of a byte of `s`: a part converted
    # to `uint32_t` stays in its range, where a negative one is undefined -- AArch64 saturates it, x86-64 wraps it)
    "uint32_t f(uint32_t s) {\n"
    "  uint32_t t = s & 255u; double _Complex z = t + (2.0 * t) * I; z = -z; return (uint32_t)(__imag__ z + 1000.0);\n}",
    "uint32_t f(uint32_t s) {\n"
    "  uint32_t t = s & 255u; float _Complex z = t + (3.0f * t) * I; z = ~z;\n"
    "  return (uint32_t)(__imag__ z + 1000.0f) + (uint32_t)sizeof(-z);\n}",
    "uint32_t f(uint32_t s) {\n"
    "  double _Complex z = s; float _Complex fz = s;\n"
    "  return (uint32_t)sizeof(__real__ z) * 3u + (uint32_t)sizeof(__imag__ fz) + s;\n}",
    "uint32_t f(uint32_t s) { double _Complex z = s; return _Generic(__real__ z, double: 1u, default: 2u) + s; }",
    "uint32_t f(uint32_t s) {\n"
    "  uint8_t c = (uint8_t)s;\n"
    "  return (uint32_t)sizeof(__real__ c) + (uint32_t)(__real__ c) + (uint32_t)(__imag__ c)\n"
    "         + _Generic(__real__ c, int: 1u, uint8_t: 2u, default: 3u);\n}",
    # a bit-field operand has the type of its value
    "uint32_t f(uint32_t s) {\n"
    "  struct bf x = {5u, -3}; x.a = s;\n"
    "  return _Generic(x.a - 1, int: 1u, unsigned: 2u, default: 3u) + _Generic(-x.a, int: 4u, unsigned: 8u, default: 16u);\n}",
    "uint32_t f(uint32_t s) {\n"
    "  struct bf x = {5u, -3}; x.a = s;\n"
    "  return _Generic(x.a << 1, int: 1u, unsigned: 2u, default: 3u) + (uint32_t)sizeof(+x.a) + (uint32_t)(+x.b);\n}",
    # `&x` is a pointer to x
    "uint32_t f(uint32_t s) {\n"
    "  struct s a = {s, 1u};\n"
    "  return _Generic(&gv, uint32_t *: 1u, default: 2u) + _Generic(&a, struct s *: 4u, default: 8u)\n"
    "         + _Generic(&a.o, uint32_t *: 16u, default: 32u) + _Generic(&garr[1], uint32_t *: 64u, default: 128u);\n}",
    "uint32_t f(uint32_t s) { __typeof__(&gv) p = &gv; struct s a = {s, 2u}; __typeof__(&a) q = &a; return *p + q->o; }",
    # scalar controlling expressions; a `switch` of `_Bool` and of `char`
    "uint32_t f(uint32_t s) {\n"
    "  uint32_t *p = &gv; double d = s; uint32_t r = 0u;\n"
    "  if (p) r += 1u; if (d) r += 2u; if (garr) r += 8u; return r + (d ? 16u : 32u);\n}",
    "uint32_t f(uint32_t s) {\n"
    "  _Bool b = s & 1u; char c = (char)(s & 7u); uint32_t r = 0u;\n"
    "  switch (b) { case 1: r += 1u; break; default: r += 2u; }\n"
    "  switch (c) { case 3: return r + 4u; default: return r; }\n}",
    # `++(x)`: a parenthesized lvalue steps as itself
    "uint32_t f(uint32_t s) {\n"
    "  uint32_t x = s; struct s a = {s, 2u};\n"
    "  ++(x); --(a.o); ++((x)); uint32_t y = ++(x) * 2u; return x + y + a.o;\n}",
    "uint32_t f(uint32_t s) { uint32_t a[2] = {1u, s}; uint32_t *p = &gv; ++(a[1]); ++(p[0]); --(*p); return a[1] + *p; }",
    # an array compared with 0; `malloc` into a `void *`
    "uint32_t f(uint32_t s) {\n"
    "  uint32_t arr[2] = {1u, 2u}; uint32_t m[2][2] = {{1u}, {2u}};\n"
    "  return (garr == 0) + (garr != 0) * 2u + (arr != 0) * 4u + (0 == arr) * 8u + (m != 0) * 16u + s;\n}",
    "uint32_t f(uint32_t s) { uint32_t v[(s & 3u) + 1u]; v[0] = s; return (v != 0) + v[0]; }",
    "uint32_t f(uint32_t s) { void *vp = malloc(4u); uint32_t r = vp != 0; free(vp); return r + s; }",
    # a file-scope object read only by a control node -- an early `return`, a `switch`, a `while`, a `do`, a `for` --
    # touched by no claim, named in the emit all the same (CF-GCOND); a void `_Generic` association as a statement
    "uint32_t f(uint32_t s) { if (s & 1u) return gv; return 0u; }",
    "uint32_t f(uint32_t s) { switch (gv) { case 5: return s; default: return 0u; } }",
    "uint32_t f(uint32_t s) { uint32_t r = s; while (gv) { r++; break; } return r; }",
    "uint32_t f(uint32_t s) { uint32_t r = 0u; do { r++; if (r > 2u) break; } while (gv); return r + s; }",
    "uint32_t f(uint32_t s) { uint32_t r = 0u; for (; gv;) { r += s; break; } return r; }",
    "uint32_t f(uint32_t s) { g_n = 0u; _Generic(s, uint32_t: vd(s), default: vd(1u)); return g_n; }",
    # `+4` is no literal byte offset: the access is no member access at 4, on either rail (the twin's reader had
    # skipped the `+` the oracle's parser dropped)
    "struct ua { uint32_t k; _Atomic uint32_t v; };\n"
    "static struct ua g_ua = {1u, 2u};\n"
    "uint32_t f(uint32_t s) { struct ua *a = &g_ua; *(_Atomic uint32_t *)((char *)a + +4) = s; return a->v; }",
)
# ... and a `_BitInt` keeps its type under `+`, `-` and `~` (C23 6.3.1.1p2), the original built by Clang alone
_CARD5_BITINT = (
    "uint32_t f(uint32_t s) {\n"
    "  _BitInt(7) b = (_BitInt(7))(s & 31u);\n"
    "  return (uint32_t)(-b < 0) + (uint32_t)sizeof(-b) * 3u + (uint32_t)sizeof(+b) * 5u;\n}",
    "uint32_t f(uint32_t s) { unsigned _BitInt(12) b = (unsigned _BitInt(12))s; return (uint32_t)(~b) + (uint32_t)sizeof(~b); }",
)


def test_unary_operators_controlling_expressions_and_void_values_run_as_the_original():
    """CF-UNARY, CF-STRUCTCOND, CF-VOIDVAL: `cfront_unaryops.c` -- `+a` promotes its operand under `sizeof`, `_Generic`
    and `typeof`; `-z` and `~z` of a complex are complex, `__real__` and `__imag__` a part, of a complex its element
    type and of an integer the integer; a bit-field operand has the type of its value; `&x` is a pointer to x; an array
    compared with 0 is the pointer it decays to; `void *vp = malloc(4u)`; scalar controlling expressions and a
    `switch` of `_Bool` and `char`; `++(x)`; a void expression where C reads no value. Both parsers had dropped the `+`
    and each rail mistyped some of the rest -- silent miscompiles, digest-equal where both rails agreed. Both lower the
    unit to one claim graph on the four targets and each emit, with the mixes `_QUALS_WERROR` names made errors,
    returns what the original does under Clang and GCC, function by function."""
    if not _CC:
        return
    fx = "cfront_unaryops.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    _run_against_original_werror(
        fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _UNARYOPS_DRIVER, _QUALS_WERROR
    )


def test_void_values_struct_conditions_and_unary_operands_refused_and_lowered_alike():
    """CF-VOIDVAL, CF-STRUCTCOND, CF-UNARY: every unit of `_CARD5_REFUSED` is refused on both rails for the reason it
    names -- a void expression's value used (C11 6.3.2.2), a struct or union as a controlling expression or as the
    operand of `+`, `__real__`, `__imag__`, `++` or `--` (in memory too), a pointer, a function or an array as the
    operand of a unary arithmetic operator and a real floating value under `~` (6.5.3.3p1), under `sizeof`, `typeof`
    and `_Generic` as well, a `switch` of no integer (6.8.4.2p1), a step of a table's function pointer. Both rails
    had lowered most of them to an emit no compiler takes, or refused them for reasons of their own. Every unit of
    `_CARD5_LOWERED` lowers to one claim graph on the four targets, each emit the original under Clang and GCC; those
    of `_CARD5_BITINT` under Clang, which alone builds a `_BitInt` here."""
    for body, why in _CARD5_REFUSED:
        got = _fptab_refusal(_CARD5_HEAD + body + "\n")
        assert got == why, (body, got, why)
    for body in _CARD5_LOWERED + _CARD5_BITINT:
        got = _fptab_refusal(_CARD5_HEAD + body + "\n")
        assert got == "", (body, got)
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, (body, why) in enumerate(_CARD5_REFUSED):
            path = os.path.join(d, f"r{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_CARD5_HEAD + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert run.returncode == 1 and run.stdout.strip() == f"PARSE-ERR {why}", (
                body,
                run.returncode,
                run.stdout[:200],
            )
        for n, body in enumerate(_CARD5_LOWERED):
            _rtfp_run_unit(f"l{n}", _CARD5_HEAD + body + "\n", exe, d)
        clang = {"clang": _QUALS_WERROR["clang"]} if shutil.which("clang") else {}
        for n, body in enumerate(_CARD5_BITINT):
            unit = _CARD5_HEAD + body + "\n"
            path = os.path.join(d, f"b{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(unit)
            _parity_on_targets(path, unit)
            if not clang:
                continue
            oracle_summary, r, _entry = _oracle(unit)
            c_summary, c_emit = _c_run(exe, path)
            assert c_summary == oracle_summary and "ok=1" in c_summary, (
                body,
                c_summary,
                oracle_summary,
            )
            oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
            driver = (
                _GAPS_SAME
                + "int main(void) {\n  for (unsigned n = 0; n < GAPS_N; n++) SAME(f, gaps_in[n]);\n"
                + '  puts("MATCH");\n  return 0;\n}\n'
            )
            _run_against_original_werror(
                f"b{n}", unit, (("twin", c_emit), ("oracle", oracle_emit)), driver, clang
            )


# CF-CHARELEM, CF-STRELEM, CF-AOS2D, CF-DEREFSUM, CF-PARENSTR, CF-LINKEMIT (card 4): file-scope objects as C types them
# and the emits name them. Each emit runs with AArch64's `char` too, which is unsigned: an `int8_t` temp of a `char`
# element reads 200 as -56 there.
_FILESCOPE_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(fs_chars, s); SAME(fs_strelem, s); SAME(fs_aos2d, s); SAME(fs_aos3d, s); SAME(fs_aoscopy, s);
    SAME(fs_ptr2d, s); SAME(fs_derefsum, s); SAME(fs_parenstr, s); SAME(fs_consts, s); SAME(fs_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)
_FILESCOPE_CALLS = (
    "fs_chars",
    "fs_strelem",
    "fs_aos2d",
    "fs_aos3d",
    "fs_aoscopy",
    "fs_ptr2d",
    "fs_derefsum",
    "fs_parenstr",
    "fs_consts",
    "fs_entry",
)
_UNSIGNED_CHAR_WERROR = {cc: (*flags, "-funsigned-char") for cc, flags in _QUALS_WERROR.items()}


def test_file_scope_objects_run_as_the_original_with_either_char():
    """CF-CHARELEM, CF-STRELEM, CF-AOS2D, CF-DEREFSUM, CF-PARENSTR, CF-LINKEMIT: `cfront_filescope.c` -- plain `char`
    elements and casts; string literal elements under `*`, `+` and `_Generic`; `gm[i][j].x` of 2-D and 3-D file-scope
    arrays of structs and unions read, stored, compounded, stepped, copied, addressed and passed; 2-D tables of
    pointers; `*(p + i - j)`; `char s[] = ("abc");` at file and block scope and static; `const` globals of every level
    read into the emit's own temps. The twin had typed a `char` element `int8_t` and a string literal's element `int`;
    both rails refused `gm[i][j].x`; the twin read `*(p + i - j)` as `p[i - j]` with the index computed unsigned -- a
    silent miscompile -- and refused the parenthesized string; both emits took a `const` global into an unqualified
    temp, which Clang refuses. Both rails lower the fixture to one claim graph on the four targets, and each emit, with
    the mixes `_QUALS_WERROR` names made errors, returns what the original does under Clang and GCC with a signed and
    with an unsigned `char`, function by function."""
    if not _CC:
        return
    fx = "cfront_filescope.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    emits = (("twin", c_emit), ("oracle", oracle_emit))
    _run_against_original_werror(fx, src, emits, _FILESCOPE_DRIVER, _QUALS_WERROR)
    _run_against_original_werror(fx, src, emits, _FILESCOPE_DRIVER, _UNSIGNED_CHAR_WERROR)


def test_file_scope_fixture_linkable_emit_builds_alone_and_runs_as_the_original():
    """CF-LINKEMIT: the oracle's linkable emit of `cfront_filescope.c` stands alone -- its struct, union and typedef
    definitions as the source spells them, its `const` globals `const`, its tables of string pointers and its address
    constants (C11 6.6p9) rendered, `<string.h>` for the memcpy it calls -- and builds under `-Wall -Werror` with Clang
    and GCC (Clang's `-Wstring-plus-int` aside: it names the source's own `"ab" + 1`); linked with a driver that calls
    each function, it prints what the original prints. It had named each struct undefined, dropped `const`, refused
    `const char *tab[] = {"a"}` and called memcpy undeclared."""
    from bcir.frontends.cfront.emit import emit_linkable

    if not _CC:
        return
    fx = "cfront_filescope.c"
    src = open(os.path.join(_C, fx), encoding="utf-8").read()
    r = compile_unit(src, check_clang=False)
    linkable = emit_linkable(r.lowered, r.emitted)
    printed = " ".join(f'printf(" %u", (unsigned){f}(s));' for f in _FILESCOPE_CALLS)
    driver = (
        "#include <stdint.h>\n#include <stdio.h>\n"
        + "".join(f"uint32_t {f}(uint32_t s);\n" for f in _FILESCOPE_CALLS)
        + "static const uint32_t gaps_in[] = {0u, 1u, 2u, 3u, 7u, 255u, 256u, 65535u, 65536u, 0x12345678u,"
        + " 0xFFFFFFFFu};\n"
        + "int main(void) {\n  for (unsigned n = 0; n < sizeof gaps_in / sizeof gaps_in[0]; n++) {\n"
        + f"    uint32_t s = gaps_in[n];\n    {printed}\n    putchar('\\n');\n  }}\n  return 0;\n}}\n"
    )
    ran = 0
    with tempfile.TemporaryDirectory() as d:
        paths = {}
        for name, text in (("linkable", linkable), ("original", src), ("driver", driver)):
            paths[name] = os.path.join(d, f"{name}.c")
            with open(paths[name], "w", encoding="utf-8") as fh:
                fh.write(text)
        for cc in ("clang", "gcc"):
            exe = shutil.which(cc)
            if not exe:
                continue
            quiet = ("-Wno-string-plus-int",) if cc == "clang" else ()
            alone = subprocess.run(
                [exe, "-std=c11", "-Wall", "-Werror", *quiet, "-I", _C, "-c", paths["linkable"]]
                + ["-o", os.path.join(d, "linkable.o")],
                capture_output=True,
                text=True,
            )
            assert alone.returncode == 0, (
                f"{cc}: the linkable emit does not build alone\n{alone.stderr}"
            )
            outs = []
            for label, parts in (
                ("original", [paths["original"]]),
                ("linkable", ["-I", _C, paths["linkable"], os.path.join(_C, "bcir_quarantine.c")]),
            ):
                prog = os.path.join(d, f"{label}_{cc}")
                b = subprocess.run(
                    host_link_args([exe, "-std=c11", "-O2", *parts, paths["driver"], "-o", prog]),
                    capture_output=True,
                    text=True,
                )
                assert b.returncode == 0, f"{cc}: the {label} program does not build\n{b.stderr}"
                outs.append(
                    subprocess.run([prog], capture_output=True, text=True, timeout=300).stdout
                )
            assert outs[0] and outs[1] == outs[0], f"{cc}: the linkable emit is not the original"
            ran += 1
    assert ran, "no compiler of the pair"


# CF-ENUMOBJ: an object of an enumerated type has the type the enumeration is compatible with (C11 6.7.2.2p4): on the
# System V targets `unsigned int` when no enumerator is negative and `int` otherwise, on MSVC `int`. Both rails had typed
# every one `int` -- the same claim graph, so only an emit run against the original showed it.
_ENUMOBJ_DRIVER = (
    _GAPS_SAME
    + r"""
int main(void) {
  for (unsigned n = 0; n < GAPS_N; n++) {
    uint32_t s = gaps_in[n];
    SAME(eo_local, s); SAME(eo_global, s); SAME(eo_calls, s); SAME(eo_member, s); SAME(eo_typedef, s);
    SAME(eo_conv, s); SAME(eo_compound, s); SAME(eo_switch, s); SAME(eo_signed, s); SAME(eo_names, s);
    SAME(eo_entry, s);
  }
  puts("MATCH");
  return 0;
}
"""
)


def test_enum_objects_take_their_compatible_type_on_both_rails():
    """CF-ENUMOBJ: `cfront_enumobj.c` -- objects of an enumerated type as locals, a static and an external global,
    a parameter, a return, members, a 3-bit bit-field and typedef names of a tagged and an untagged enum, read where
    `int` and `unsigned int` part: `(c - 5) < 0`, a widening conversion, `_Generic`, a division, a shift, `c -= 1` of
    `RED`, `c--`, a switch. Both rails had typed each `int`, so the comparison was 1 and `_Generic` chose `int` where
    Clang and GCC, making an enumeration with no negative enumerator `unsigned int`, give 0 and `unsigned int`. Both
    rails lower the fixture to one claim graph on the four targets, and each emit returns what the original does under
    Clang and GCC, function by function; the enumeration with a negative enumerator stays `int`."""
    if not _CC:
        return
    fx = "cfront_enumobj.c"
    src, oracle_emit, c_emit = _fixture_both_rails(fx)
    _parity_on_targets(os.path.join(_C, fx), src)
    _run_against_original_werror(
        fx, src, (("twin", c_emit), ("oracle", oracle_emit)), _ENUMOBJ_DRIVER, _QUALS_WERROR
    )


# Units whose `f` folds to a constant the enumeration's type decides, held to Clang's own fold on each target (the
# MSVC one makes every enumeration `int`), and the enumeration constants that stay `int` everywhere.
_ENUMOBJ_TARGET_UNITS = (
    "enum col { RED, GREEN = 4 };\nuint32_t f(void) { enum col c = RED; return _Generic(c, unsigned int: 1u, "
    "int: 2u, default: 3u); }",
    "enum col { RED, GREEN = 4 };\nuint32_t f(void) { enum col c = RED; return (uint32_t)((int64_t)(c - 5) < 0); }",
    "enum sg { NEG = -1, POS };\nuint32_t f(void) { enum sg g = POS; return _Generic(g, unsigned int: 1u, "
    "int: 2u, default: 3u); }",
    "typedef enum { A, B } e_t;\nuint32_t f(void) { e_t x = B; return (uint32_t)((int64_t)(x - 2) >> 40); }",
    "enum col { RED, GREEN = 4 };\nuint32_t f(void) { return _Generic(GREEN, unsigned int: 1u, int: 2u, "
    "default: 3u); }",
    "enum col { RED, GREEN = 4 };\nstruct h { enum col b : 3; };\nuint32_t f(void) { struct h v = {GREEN}; "
    "return (uint32_t)((int64_t)v.b < 0); }",
    "enum col { RED, GREEN = 4 };\nstatic const enum col g = RED;\nuint32_t f(void) { return (uint32_t)"
    "((int64_t)(g - 1) >> 40); }",
    "enum col { RED, GREEN = 4 };\nenum col pick(void) { return GREEN; }\nuint32_t f(void) { return "
    "(uint32_t)((int64_t)(pick() - 5) < 0); }",
)


def _folded_return(ir: str, name: str):
    """The constant `name` returns in LLVM IR `ir` that folded it to one, else None."""
    m = re.search(r"^define [^\n]*@" + re.escape(name) + r"\(.*?^\}", ir, re.M | re.S)
    if not m:
        return None
    rets = re.findall(r"ret i32 (-?\d+)", m.group(0))
    return rets[0] if len(rets) == 1 else None


def test_enum_objects_fold_as_clang_folds_them_on_each_target():
    """CF-ENUMOBJ: each unit of `_ENUMOBJ_TARGET_UNITS`, lowered by both rails for each of the four targets, folds
    under Clang for that target (`-O1`) to the constant the original folds to -- `unsigned int` for an enumeration with
    no negative enumerator on the System V targets, `int` on MSVC's -- so the MSVC rule holds where nothing here can run
    it. Both rails had folded each as `int` on every target."""
    clang = shutil.which("clang")
    if not clang:
        return
    from bcir.frontends.cfront import abi as abi_mod

    exe = _build_frontend(_session_build_dir())
    # freestanding, so no target's libc headers are needed: an emit's own `memcpy` is the builtin
    head = "#include <stdint.h>\n#include <stddef.h>\n#define memcpy __builtin_memcpy\n"
    with tempfile.TemporaryDirectory() as d:
        for k, unit in enumerate(_ENUMOBJ_TARGET_UNITS):
            unit = "#include <stdint.h>\n" + unit + "\n"
            path = os.path.join(d, f"u{k}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(unit)
            for t in _BUF_TARGETS:
                r = compile_unit(unit, check_clang=False, target=t)
                oracle_emit = "\n".join(r.emitted[n] for n in r.lowered.functions)
                run = subprocess.run([exe, "--target", t, path], capture_output=True, text=True)
                summary, _, twin_emit = run.stdout.partition("----EMIT----\n")
                assert _summary_line(r) == (summary.strip().splitlines() or [""])[0], (k, t)
                for label, emit in (("oracle", oracle_emit), ("twin", twin_emit)):
                    # the emit's `bcir_f` is static: each is read through a function that keeps it
                    keep = "uint32_t f_kept(void) { return f(); }\nuint32_t bcir_f_kept(void) { return bcir_f(); }\n"
                    text = head + unit + emit + keep
                    c = os.path.join(d, f"u{k}_{t}_{label}.c")
                    with open(c, "w", encoding="utf-8") as fh:
                        fh.write(text)
                    b = subprocess.run(
                        [
                            clang,
                            "-target",
                            abi_mod.target(t).triple,
                            "-std=c11",
                            "-ffreestanding",
                            "-O1",
                        ]
                        + ["-S", "-emit-llvm", "-o", "-", c],
                        capture_output=True,
                        text=True,
                    )
                    assert b.returncode == 0, (k, t, label, b.stderr[-2000:])
                    want = _folded_return(b.stdout, "f_kept")
                    got = _folded_return(b.stdout, "bcir_f_kept")
                    assert want is not None and got == want, (k, t, label, want, got)


def test_linkable_emit_spells_an_enum_global_by_its_compatible_type():
    """CF-ENUMOBJ: the oracle's linkable emit of `cfront_enumobj.c` declares its enum-typed globals by the integer type
    each enumeration is compatible with, so another unit's `extern enum col gcol2;` and `extern enum sg gsg;` -- here
    in the same translation unit, where Clang and GCC refuse two incompatible declarations of one object (C11 6.2.7p2)
    -- name the same object. It had declared each `int`: `conflicting types` for the `enum col` one."""
    if not _CC:
        return
    from bcir.frontends.cfront.emit import emit_linkable

    src = open(os.path.join(_C, "cfront_enumobj.c"), encoding="utf-8").read()
    # the fixture's static global made external, beside the one already external
    src = src.replace("static enum col gcol = GREEN;", "enum col gcol = GREEN;")
    r = compile_unit(src, check_clang=False)
    linkable = emit_linkable(r.lowered, r.emitted)
    assert "int gsg" in linkable and "gcol" in linkable, linkable
    redeclared = linkable + "\nextern enum col gcol;\nextern enum sg gsg;\n"
    ran = 0
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "linkable.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(redeclared)
        for cc in ("clang", "gcc"):
            exe = shutil.which(cc)
            if not exe:
                continue
            b = subprocess.run(
                [exe, "-std=c11", "-I", _C, "-c", path, "-o", os.devnull],
                capture_output=True,
                text=True,
            )
            assert b.returncode == 0, (
                f"{cc}: an enum global's declaration conflicts\n{b.stderr[-2000:]}"
            )
            ran += 1
    assert ran, "no compiler of the pair"


# CF-ENUMOBJ: an enumerated type is known only after its enumerator list (C11 6.7.2.3p3) -- before it, or with none,
# which integer type it is compatible with is not known; both rails refuse an object, a cast, a `sizeof` or a typedef of
# one for one reason (they had read it as `int`). And forms that read an enum object where `int` and `unsigned int`
# part -- through a pointer, an element, a division, a shift, a remainder, `?:`, a compound literal -- lowered alike, each
# emit the original.
_ENUMOBJ_HEAD = "#include <stdint.h>\nenum col { RED, GREEN = 4 };\n"
_ENUMOBJ_INCOMPLETE = "an enumerated type with no definition"
_ENUMOBJ_REFUSED = (
    "uint32_t f(uint32_t s) { enum nope c = s; return c; }",
    "enum nope gn;\nuint32_t f(uint32_t s) { return s; }",
    "typedef enum later later_t;\nenum later { L0, L1 };\nuint32_t f(uint32_t s) { later_t x = L1; return x + s; }",
    "uint32_t f(uint32_t s) { return (uint32_t)sizeof(enum nope) + s; }",
    "uint32_t f(uint32_t s) { return (uint32_t)(enum nope)s; }",
    "uint32_t f(enum nope p) { return 1u; }",
)
# ... and a tag names no object (6.2.3p1): `col`, the tag of `enum col`, read as an identifier at file scope or in the
# block that declares it, is refused on both rails as Clang refuses it -- never folded to the enumeration's record.
_ENUMOBJ_TAG_ALONE = (
    ("uint32_t f(uint32_t s) { return s + col; }", "col"),
    ("uint32_t f(uint32_t s) { enum blk { B0, B1 = 4 }; return s + blk + B1; }", "blk"),
)
_ENUMOBJ_LOWERED = (
    "uint32_t f(uint32_t s) { enum col c = (enum col)s; enum col *p = &c; *p -= 1u; "
    "return (uint32_t)((int64_t)*p >> 40); }",
    "uint32_t f(uint32_t s) { enum col a[2] = {RED, GREEN}; a[0] -= 1; "
    "return (uint32_t)((int64_t)a[s & 1u] >> 40) + (uint32_t)a[1]; }",
    "uint32_t f(uint32_t s) { enum col a = (enum col)(0u - (s & 1u)); return (uint32_t)(a / 3); }",
    "uint32_t f(uint32_t s) { enum col a = (enum col)(0u - (s & 1u)); return (uint32_t)(a >> 1); }",
    "uint32_t f(uint32_t s) { enum col a = (enum col)(0u - 7u * (s & 1u)); return (uint32_t)(a % 3); }",
    "uint32_t f(uint32_t s) { enum col a = RED, b = GREEN; "
    "return (uint32_t)((int64_t)((s & 1u) ? a - 1 : b - 1) >> 40); }",
    "uint32_t f(uint32_t s) { return (uint32_t)((int64_t)((enum col){RED} - 1) >> 40) + s; }",
    "uint32_t f(uint32_t s) { enum col a = (enum col)s, b = GREEN; return (uint32_t)(a - b > 0) + "
    "(uint32_t)(a < b); }",
)


def test_an_incomplete_enum_type_is_refused_and_enum_forms_lowered_alike():
    """CF-ENUMOBJ: each unit of `_ENUMOBJ_REFUSED` -- an object, a file-scope object, a typedef, a `sizeof`, a cast and
    a parameter of an enumerated type before or without its definition -- is refused on both rails as `an enumerated
    type with no definition` (C11 6.7.2.3p3); both had read each as `int`. Each of `_ENUMOBJ_TAG_ALONE`, a tag read as
    an identifier, is refused on both as an undeclared identifier. Each unit of `_ENUMOBJ_LOWERED` lowers to one
    claim graph on the four targets, each emit the original under Clang and GCC."""
    for body in _ENUMOBJ_REFUSED:
        got = _card4_refusal(_ENUMOBJ_HEAD + body + "\n")
        assert got == _ENUMOBJ_INCOMPLETE, (body, got)
    for body, tag in _ENUMOBJ_TAG_ALONE:
        got = _card4_refusal(_ENUMOBJ_HEAD + body + "\n")
        assert got == f"use of undeclared identifier {tag!r}", (body, got)
    for body in _ENUMOBJ_LOWERED:
        got = _card4_refusal(_ENUMOBJ_HEAD + body + "\n")
        assert got == "", (body, got)
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, body in enumerate(_ENUMOBJ_REFUSED):
            path = os.path.join(d, f"r{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_ENUMOBJ_HEAD + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            assert (
                run.returncode == 1 and run.stdout.strip() == f"PARSE-ERR {_ENUMOBJ_INCOMPLETE}"
            ), (
                body,
                run.returncode,
                run.stdout[:200],
            )
        for n, (body, tag) in enumerate(_ENUMOBJ_TAG_ALONE):
            path = os.path.join(d, f"t{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_ENUMOBJ_HEAD + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            want = f"PARSE-ERR use of undeclared identifier {tag!r}"
            assert run.returncode == 1 and run.stdout.strip() == want, (body, run.stdout[:200])
        for n, body in enumerate(_ENUMOBJ_LOWERED):
            _rtfp_run_unit(f"l{n}", _ENUMOBJ_HEAD + body + "\n", exe, d)


# CF-PPARITH: a `#if`/`#elif` expression in C's integer arithmetic (6.10.1p4) on both rails. Each form takes the branch
# Clang takes for the target -- read from `clang -target T -std=c2x -E` -- on both rails, on the four targets: every
# operand an `intmax_t` or a `uintmax_t`, an unsigned operand making the arithmetic unsigned, a character constant by
# the target's character types (plain `char` unsigned on AArch64, so `'a' - 98 < 0` is false there; `wchar_t` unsigned
# on AArch64 and Windows), `true` 1, the operand C leaves unevaluated never refused. Both rails had evaluated in a
# signed host integer (Python's unbounded `int`, the twin's `long`), read a character constant as 0, taken `?:` whole
# (the twin) and ignored an unclosed `(` or a trailing token.
_PPARITH_FORMS = (
    # unsigned operands and the usual arithmetic conversions
    "-1 > 0u",
    "-1 < 0",
    "0u - 1 == 18446744073709551615u",
    "0xFFFFFFFFFFFFFFFF == -1",
    "0x8000000000000000 > 0",
    "-9223372036854775807 - 1 < 0",
    "-1 / 2u == 9223372036854775807",
    "-1 % 3u == 0",
    "-7 / 2 == -3 && -7 % 2 == -1",
    "1u << 63 > 0",
    "-1 >> 1 == -1",
    "(0u - 1) >> 63 == 1",
    "~0u == 18446744073709551615u",
    "~0 == -1",
    "-0u == 0",
    "!0u - 2 < 0",
    "(1 ? -1 : 0u) > 0",
    "(0 ? 1u : -1) > 0",
    "(1 ? -1 : 0) < 0",
    "1 ? 2 : 3 ? 4 : 5",
    "defined(__STDC__) + 0u - 2 > 0",
    "NOT_A_MACRO == 0",
    "true && !false",
    "1 + 2 * 3 == 7 && (1 + 2) * 3 == 9",
    # operands C leaves unevaluated refuse for nothing
    "1 ? 2 : 1 / 0",
    "0 ? 1 / 0 : 3",
    "0 && 1 / 0",
    "1 || 1 / 0",
    "(2 || 1 / 0) == 1",
    "0 && (1, 2)",
    "0 && (1 << 64)",
    "1 || -9223372036854775807 - 2",
    # character constants
    "'a' == 97",
    "'a' - 98 < 0",
    "'\\377' < 0",
    "'\\xff' == -1",
    "'\\0' == 0 && '\\n' == 10 && '\\'' == 39",
    "'ab' == 24930",
    "'\\xff\\xff\\xff\\xff' < 0",
    "L'a' - 98 < 0",
    "L'\\xff' == 255",
    "u'\\xffff' > 0",
    "U'\\xffffffff' > 0",
    "u8'\\x80' > 0",
    "u8'a' - 98 < 0",
    # C23's digit separators, which the twin's tokenizer had cut a number at
    "1'000 == 1000 && 0x1'F == 31",
)
# GCC reads a multi-character constant in `#if` as a signed `int` whatever plain `char` is (libcpp: "multichar constants
# are of type int and therefore signed"); Clang reads it by `char`'s signedness, as a single one. Both rails read
# Clang's, so GCC judges these only with a signed `char`.
_PPARITH_GCC_APART = frozenset({"'\\xff\\xff\\xff\\xff' < 0"})
# the cfront target each machine host GCC preprocesses for is (`platform.machine()`)
_PPARITH_HOSTS = {
    "x86_64": "x86_64-linux",
    "amd64": "x86_64-linux",
    "aarch64": "aarch64-linux",
    "arm64": "aarch64-linux",
}
_PPARITH_MALFORMED = "malformed #if expression"
_PPARITH_OVERFLOW = "integer overflow in #if expression"
_PPARITH_SHIFT = "invalid shift in #if expression"
_PPARITH_DIVZERO = "division by zero in #if expression"
_PPARITH_CHAR = "unsupported character constant"
# ... and the forms both rails refuse, for one reason each, on every target: C's undefined behaviour or constraint
# violations in an evaluated operand, a form the grammar does not admit, a constant no type holds, a character
# constant the C front does not read, and the twin's bounds held on both rails
_PPARITH_REFUSED = (
    ("(1, 2)", "comma operator in #if expression"),
    ("1 2", _PPARITH_MALFORMED),
    ("", _PPARITH_MALFORMED),
    ("(1", _PPARITH_MALFORMED),
    ("1 ? 2", _PPARITH_MALFORMED),
    ('"a"', _PPARITH_MALFORMED),
    ("1 = 1", _PPARITH_MALFORMED),
    ("L 'a'", _PPARITH_MALFORMED),
    ("1.0", "invalid integer literal '1.0'"),
    ("1 / 0", _PPARITH_DIVZERO),
    ("1 % (2 - 2)", _PPARITH_DIVZERO),
    ("9223372036854775807 + 1", _PPARITH_OVERFLOW),
    ("-9223372036854775807 - 2", _PPARITH_OVERFLOW),
    ("-(-9223372036854775807 - 1)", _PPARITH_OVERFLOW),
    ("(-9223372036854775807 - 1) / -1", _PPARITH_OVERFLOW),
    ("(-9223372036854775807 - 1) % -1", _PPARITH_OVERFLOW),
    ("4611686018427387904 * 2", _PPARITH_OVERFLOW),
    ("1 << 63", _PPARITH_OVERFLOW),
    ("-1 << 1", _PPARITH_OVERFLOW),
    ("-1 << 0", _PPARITH_OVERFLOW),  # a negative left operand, whatever the count (6.5.7p4)
    ("1 << 64", _PPARITH_SHIFT),
    ("1 >> -1", _PPARITH_SHIFT),
    ("1u << 64u", _PPARITH_SHIFT),
    (
        "18446744073709551616",
        "an integer constant too large for every type its base and suffix allow",
    ),
    (
        "9223372036854775808",
        "an integer constant too large for every type its base and suffix allow",
    ),
    (
        "0 && 18446744073709551616",
        "an integer constant too large for every type its base and suffix allow",
    ),
    ("''", _PPARITH_CHAR),
    ("'\\x100'", _PPARITH_CHAR),
    ("'\\400'", _PPARITH_CHAR),
    ("'\\q'", _PPARITH_CHAR),
    ("'\\u0041'", _PPARITH_CHAR),
    ("L'ab'", _PPARITH_CHAR),
    ("u8'\\x100'", _PPARITH_CHAR),
    ("u'\\x10000'", _PPARITH_CHAR),
    ("0 && '\\q'", _PPARITH_CHAR),
    ("(" * 64 + "1" + ")" * 64, "#if expression nested too deeply"),
    ("1" + " + 1" * 256, "too many tokens in #if expression"),
)


def _pparith_unit(expr: str) -> str:
    return f"#if {expr}\nint pp_taken;\n#else\nint pp_skipped;\n#endif\n"


def _pparith_branch(text: str) -> str:
    """Which branch a preprocessed `_pparith_unit` kept."""
    taken, skipped = "pp_taken" in text, "pp_skipped" in text
    assert taken != skipped, text
    return "taken" if taken else "skipped"


def test_if_expressions_take_the_branch_clang_takes_on_each_target():
    """CF-PPARITH: each `_PPARITH_FORMS` expression, preprocessed by both rails for each of the four targets, keeps the
    branch Clang keeps for that target (`clang -target T -std=c2x -E`) -- and GCC on the host, for the host's own ABI
    with a signed `char` (`-fsigned-char`) and with an unsigned one (`-funsigned-char`)."""
    if not _CC:
        return
    import dataclasses
    import platform

    from bcir.frontends.cfront import abi as abi_mod
    from bcir.frontends.cfront.cpp import preprocess

    clang = shutil.which("clang")
    gcc = shutil.which("gcc")
    host = _PPARITH_HOSTS.get(platform.machine().lower())  # the machine host GCC preprocesses for
    exe = _build_frontend(_session_build_dir())
    judged = 0
    with tempfile.TemporaryDirectory() as d:
        for k, expr in enumerate(_PPARITH_FORMS):
            unit = _pparith_unit(expr)
            path = os.path.join(d, f"pp{k}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(unit)
            for t in _BUF_TARGETS:
                abi = abi_mod.target(t)
                oracle = _pparith_branch(preprocess(unit, abi=abi))
                run = subprocess.run(
                    [exe, "--target", t, "--emit-cpp", path], capture_output=True, text=True
                )
                assert run.returncode == 0, (expr, t, run.stdout)
                assert _pparith_branch(run.stdout) == oracle, f"#if {expr} on {t}: the rails part"
                if clang:
                    ref = subprocess.run(
                        [clang, "-target", abi.triple, "-std=c2x", "-E", "-P", path],
                        capture_output=True,
                        text=True,
                    )
                    assert ref.returncode == 0, (expr, t, ref.stderr)
                    assert _pparith_branch(ref.stdout) == oracle, (
                        f"#if {expr} on {t}: not Clang's branch"
                    )
                    judged += 1
            if gcc and host:
                # GCC preprocesses for the machine it runs on: its own `wchar_t`, and the `char` a flag gives it --
                # held to the oracle for that machine's ABI with that `char` (on an AArch64 runner the x86-64 ABI's
                # signed `wchar_t` is not GCC's, CF-PPARITH.1)
                for flag in ("-fsigned-char", "-funsigned-char"):
                    signed = flag == "-fsigned-char"
                    if not signed and expr in _PPARITH_GCC_APART:
                        continue  # GCC's own reading of a multi-character constant
                    ref = subprocess.run(
                        [gcc, flag, "-std=c2x", "-E", "-P", path], capture_output=True, text=True
                    )
                    assert ref.returncode == 0, (expr, flag, ref.stderr)
                    habi = dataclasses.replace(abi_mod.target(host), char_signed=signed)
                    want = _pparith_branch(preprocess(unit, abi=habi))
                    assert _pparith_branch(ref.stdout) == want, (
                        f"#if {expr} {flag} on {host}: not GCC's branch"
                    )
    assert judged or not clang


def test_if_expressions_refused_for_one_reason_on_both_rails():
    """CF-PPARITH: each `_PPARITH_REFUSED` expression is refused by both rails, on the four targets, in the same words
    -- none raises a bare Python exception (`0xFFFFFFFFFFFFFFFF == -1` had raised `ValueError: negative shift count`
    on the oracle) -- and a nesting one level short of the bound, and a token count at it, are read."""
    if not _CC:
        return
    from bcir.frontends.cfront import abi as abi_mod
    from bcir.frontends.cfront.cpp import CPPError, preprocess

    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for k, (expr, why) in enumerate(_PPARITH_REFUSED):
            unit = _pparith_unit(expr)
            path = os.path.join(d, f"pr{k}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(unit)
            for t in _BUF_TARGETS:
                try:
                    preprocess(unit, abi=abi_mod.target(t))
                    raise AssertionError(f"oracle read #if {expr!r} on {t}")
                except CPPError as e:
                    assert str(e) == why, (expr, t, str(e))
                run = subprocess.run(
                    [exe, "--target", t, "--emit-cpp", path], capture_output=True, text=True
                )
                assert run.stdout.strip() == f"CPP-ERR {why}", (expr, t, run.stdout)
        for expr in ("(" * 63 + "1" + ")" * 63, "1" + " + 1" * 255, "1 ? 1 : " * 62 + "1"):
            unit = _pparith_unit(expr)
            path = os.path.join(d, "edge.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(unit)
            assert _pparith_branch(preprocess(unit)) == "taken", expr[:40]
            run = subprocess.run([exe, "--emit-cpp", path], capture_output=True, text=True)
            assert _pparith_branch(run.stdout) == "taken", (expr[:40], run.stdout[:200])


# CF-PPARITH: an encoding prefix and the character constant it begins are one preprocessing token (C11 6.4.4.4), so
# `L` there is no macro name; a prefix a macro expands to stays apart from a constant after it; a C23 digit separator
# stays inside its number. The preprocessed text of each rail (the oracle's `preprocess`, the twin's `--emit-cpp`).
_PPARITH_TOKENS = (
    "#define P L\n#define L 7\n#define Q u8\n"
    "int x = P'a';\nint y = L'a';\nint z = L;\nint w = 1'000;\nint v = Q'a';\n"
)
_PPARITH_TOKENS_TEXT = "int x=7'a';\nint y=L'a';\nint z=7;\nint w=1'000;\nint v=u8 'a';\n"


def test_a_prefixed_character_constant_is_one_token_on_both_rails():
    """CF-PPARITH: `_PPARITH_TOKENS` preprocesses to `_PPARITH_TOKENS_TEXT` on both rails. Both had read `L'a'` as the
    identifier `L` and a plain constant -- with `L` a macro, `7'a'` -- and the twin had ended `1'000` at its separator
    and read a character constant from it, so `#if 1'000 == 1000` took no branch of C's."""
    if not _CC:
        return
    from bcir.frontends.cfront.cpp import preprocess

    assert preprocess(_PPARITH_TOKENS) == _PPARITH_TOKENS_TEXT
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tokens.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_PPARITH_TOKENS)
        run = subprocess.run([exe, "--emit-cpp", path], capture_output=True, text=True)
        assert run.stdout == _PPARITH_TOKENS_TEXT, run.stdout


# ... and units whose `#if` picks a function body, lowered by both rails for x86-64 Linux and for AArch64 Linux and run
# against the original built with that target's plain `char` (`-fsigned-char`, `-funsigned-char`) under Clang and GCC:
# the task's three reproductions, a character constant read by each `char`, and `#if`/`#elif` chains in unsigned and
# signed arithmetic with operands C leaves unevaluated
_PPARITH_RUN_EXPRS = (
    "-1 > 0u",
    "'a' == 97",
    "0xFFFFFFFFFFFFFFFF == -1",
    "'\\xff' < 0",
    "'a' - 98 < 0",
    "-1 / 2u == 9223372036854775807 && -7 / 2 == -3 && 1u << 63 > 0 && ~0 == -1",
    "(1 ? -1 : 0u) > 0 && true && !false",
    "defined(__STDC__) + 0u - 2 > 0 && (0 && 1 / 0) == 0",
)
_PPARITH_RUN_ELIF = """#include <stdint.h>
#if 0 && 1 / 0
uint32_t f(uint32_t s) { return s; }
#elif 1 || 1 / 0
#if (0u - 1) >> 63 == 1 && -1 >> 63 == -1
uint32_t f(uint32_t s) { return s * 3u + 1u; }
#else
uint32_t f(uint32_t s) { return s * 5u; }
#endif
#else
uint32_t f(uint32_t s) { return s + 9u; }
#endif
"""


def _pparith_run_unit(expr: str) -> str:
    return (
        f"#include <stdint.h>\n#if {expr}\nuint32_t f(uint32_t s) {{ return s + 1u; }}\n"
        "#else\nuint32_t f(uint32_t s) { return s ^ 7u; }\n#endif\n"
    )


def test_if_branches_lowered_by_both_rails_run_as_the_original():
    """CF-PPARITH: each `_PPARITH_RUN_EXPRS` unit and `_PPARITH_RUN_ELIF`, lowered by both rails for x86-64 Linux and
    for AArch64 Linux, gives one claim graph, and each emit returns what the original does built with that target's
    plain `char` under Clang and GCC -- the original's `#if` evaluated by the compiler itself. On the parent both
    rails took `#else` for `-1 > 0u` and `'a' == 97`, and the oracle raised a bare `ValueError` on
    `0xFFFFFFFFFFFFFFFF == -1`."""
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    driver = (
        _GAPS_SAME
        + "int main(void) {\n  for (unsigned n = 0; n < GAPS_N; n++) SAME(f, gaps_in[n]);\n"
        + '  puts("MATCH");\n  return 0;\n}\n'
    )
    units = [_pparith_run_unit(e) for e in _PPARITH_RUN_EXPRS] + [_PPARITH_RUN_ELIF]
    with tempfile.TemporaryDirectory() as d:
        for k, unit in enumerate(units):
            path = os.path.join(d, f"pa{k}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(unit)
            for t, flag in (
                ("x86_64-linux", "-fsigned-char"),
                ("aarch64-linux", "-funsigned-char"),
            ):
                r = compile_unit(unit, check_clang=False, target=t)
                run = subprocess.run([exe, "--target", t, path], capture_output=True, text=True)
                twin_summary, _, twin_emit = run.stdout.partition("----EMIT----\n")
                assert twin_summary.strip().splitlines()[0] == _summary_line(r), (
                    unit,
                    t,
                    run.stdout[:300],
                )
                oracle_emit = "\n".join(r.emitted[n] for n in r.lowered.functions)
                _run_against_original_werror(
                    f"pa{k}-{t}",
                    unit,
                    (("twin", twin_emit), ("oracle", oracle_emit)),
                    driver,
                    {"clang": (flag,), "gcc": (flag,)},
                )


def _card4_refusal(src: str) -> str:
    """The oracle's refusal of `src` -- its lexer's and its `#if`'s too -- or "" when it lowers."""
    from bcir.frontends.cfront.clex import CLexError
    from bcir.frontends.cfront.cparse import CParseError
    from bcir.frontends.cfront.cpp import CPPError
    from bcir.frontends.cfront.lower import CLowerError

    try:
        compile_unit(src, check_clang=False)
    except (CLexError, CPPError, CParseError, CLowerError) as e:
        return str(e)
    return ""


# CF-SUFFIX, CF-STRELEM (card 4): an integer constant whose suffix C does not spell, a designator naming no member and
# the pieces of a string literal of two encodings -- refused on both rails in the same words, Clang's own for a suffix.
_SUFFIX = "invalid suffix '{}' on integer constant"
_STR_PREFIXES = "string literals with different encoding prefixes are concatenated"
_CARD4_HEAD = """#include <stdint.h>
struct q { uint32_t x; uint32_t y; };
struct pt { uint32_t x, y; };
static struct pt gm[2][3] = {{{1u, 2u}, {3u, 4u}}, {{5u, 6u}}};
static uint32_t ga[8] = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
"""
# Refused on both rails for the reason each names (the unit, the reason)
_CARD4_REFUSED = (
    # a suffix C does not spell (C11 6.4.4.1p1) -- `lL`, `uu`, `lul`, three `l`s -- in an expression, an initializer
    # and a `#if`, of a decimal, hexadecimal and octal constant
    ("uint32_t f(uint32_t s) { return (uint32_t)1lL + s; }", _SUFFIX.format("lL")),
    ("uint32_t f(uint32_t s) { return (uint32_t)1Ll + s; }", _SUFFIX.format("Ll")),
    ("uint32_t f(uint32_t s) { return (uint32_t)1uu + s; }", _SUFFIX.format("uu")),
    ("uint32_t f(uint32_t s) { return (uint32_t)1lul + s; }", _SUFFIX.format("lul")),
    ("uint32_t f(uint32_t s) { return (uint32_t)1lll + s; }", _SUFFIX.format("lll")),
    ("uint32_t f(uint32_t s) { return (uint32_t)1LLL + s; }", _SUFFIX.format("LLL")),
    ("uint32_t f(uint32_t s) { return (uint32_t)0x1lL + s; }", _SUFFIX.format("lL")),
    ("uint32_t f(uint32_t s) { return (uint32_t)017lLu + s; }", _SUFFIX.format("lLu")),
    ("uint32_t f(uint32_t s) { return (uint32_t)1uLl + s; }", _SUFFIX.format("uLl")),
    (
        "static const uint32_t g = 1lL;\nuint32_t f(uint32_t s) { return g + s; }",
        _SUFFIX.format("lL"),
    ),
    ("#if 1lL\nuint32_t f(uint32_t s) { return s; }\n#endif", _SUFFIX.format("lL")),
    ("#if 0x1uu\nuint32_t f(uint32_t s) { return s; }\n#endif", _SUFFIX.format("uu")),
    # a designator naming no member of its type, named as the oracle names it
    (
        "uint32_t f(uint32_t s) { struct q v = { .z = 1u }; return v.x + s; }",
        "no member named 'z' to designate",
    ),
    (
        "static struct q g = { .y = 1u, .k = 2u };\nuint32_t f(uint32_t s) { return g.x + s; }",
        "no member named 'k' to designate",
    ),
    # two pieces of one string literal with different encoding prefixes (Clang: a non-standard concatenation)
    ('uint32_t f(uint32_t s) { return (uint32_t)sizeof(u"a" U"b") + s; }', _STR_PREFIXES),
    ('uint32_t f(uint32_t s) { return (uint32_t)(L"a" u8"b")[s & 1u] + s; }', _STR_PREFIXES),
)
# Lowered alike on the four targets, each emit the original
_CARD4_LOWERED = (
    # every suffix C spells, in an expression and in a `#if`; octal and hexadecimal constants in a `#if`
    "uint32_t f(uint32_t s) { return (uint32_t)(1ull + 2LLu + 3uLL + 4lu + 5Ul + 6LU + 7llu + 8ULL + 9uL + 10Lu) + s; }",
    "#if 1ull && 2LLu && 0x3uLL && 07lu\nuint32_t f(uint32_t s) { return s + 1u; }\n#else\n"
    "uint32_t f(uint32_t s) { return s; }\n#endif",
    "#if 0777 == 511 && 0x1F == 31\nuint32_t f(uint32_t s) { return s + 1u; }\n#else\n"
    "uint32_t f(uint32_t s) { return s; }\n#endif",
    # designators in any order; pieces of one string literal, one of them prefixed
    "uint32_t f(uint32_t s) { struct q v = { .y = 1u, .x = 2u }; return v.x + v.y * 3u + s; }",
    'uint32_t f(uint32_t s) { return (uint32_t)sizeof(u"a" "b") + (uint32_t)sizeof("a" U"b") * 3u'
    ' + (uint32_t)("a" L"b")[s & 1u] * 5u; }',
    # `&gm[i][j].y` held and written through; a volatile and an `_Atomic` member of a 2-D array of structs; a loop over
    # every element
    "uint32_t f(uint32_t s) { uint32_t *p = &gm[1][2].y; uint32_t old = *p; *p = s; uint32_t r = gm[1][2].y; *p = old;"
    " return r; }",
    "static volatile struct pt gv[2][2];\n"
    "uint32_t f(uint32_t s) { gv[s & 1u][1].x = s; gv[1][0].y = 2u; return gv[s & 1u][1].x + gv[1][0].y; }",
    "struct at { _Atomic uint32_t x; uint32_t y; };\nstatic struct at gt[2][2];\n"
    "uint32_t f(uint32_t s) { gt[s & 1u][1].x = s; gt[s & 1u][1].x += 2u; return gt[s & 1u][1].x + gt[1][0].y; }",
    "uint32_t f(uint32_t s) { uint32_t r = 0u; for (uint32_t i = 0u; i < 2u; i++) for (uint32_t j = 0u; j < 3u; j++)"
    " { gm[i][j].x += s; r += gm[i][j].x; gm[i][j].x -= s; } return r; }",
    # a `const` global under `_Generic` and `typeof`, which the twin lowers and rolls back: the rid it took is a
    # temp's next, which its cast must not name
    "static const uint32_t gk[2] = {1u, 2u};\n"
    "uint32_t f(uint32_t s) { uint32_t r = _Generic(gk[s & 1u], uint32_t: 3u, default: 5u) + s;"
    " return r + gk[1] + (uint32_t)sizeof(__typeof__(gk[0])) * 7u; }",
    # `*(p + i - j)` through a table of pointers; a parenthesized string's element; a parenthesized string initializer
    "uint32_t f(uint32_t s) { uint32_t *q[2] = {&ga[2], &ga[5]};"
    " return *(q[s & 1u] + 1u - 1u) + *(q[1] + (s & 1u) - 1u) * 3u; }",
    'uint32_t f(uint32_t s) { return (uint32_t)(int)*("abc" + 1 + (s & 1u))'
    ' + (uint32_t)(int)*(("ab\\xf1") + (s & 2u)) * 3u; }',
    'uint32_t f(uint32_t s) { char t[] = ("xyz"); static char u[] = ("pq");'
    " return (uint32_t)t[s % 3u] + (uint32_t)u[s & 1u] * 3u + (uint32_t)sizeof t * 5u; }",
)


def test_malformed_constants_refused_and_file_scope_forms_lowered_alike():
    """CF-SUFFIX, CF-STRELEM, CF-AOS2D, CF-DEREFSUM, CF-PARENSTR: every unit of `_CARD4_REFUSED` is refused on both
    rails in the same words -- an integer constant whose suffix C does not spell (C11 6.4.4.1p1), in an expression, an
    initializer and a `#if`, in Clang's words; a designator naming no member, which the twin now names; two pieces of a
    string literal with different encoding prefixes. Both rails had taken every malformed suffix -- the oracle raised a
    bare `KeyError` on `1lll` -- and the twin's `#if` had read `0777` as 777. Every unit of `_CARD4_LOWERED` lowers to
    one claim graph on the four targets, each emit the original under Clang and GCC."""
    from bcir.frontends.cfront.clex import STR_PREFIXES

    assert _STR_PREFIXES == STR_PREFIXES
    for body, why in _CARD4_REFUSED:
        got = _card4_refusal(_CARD4_HEAD + body + "\n")
        assert got == why, (body, got, why)
    for body in _CARD4_LOWERED:
        got = _card4_refusal(_CARD4_HEAD + body + "\n")
        assert got == "", (body, got)
    if not _CC:
        return
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        for n, (body, why) in enumerate(_CARD4_REFUSED):
            path = os.path.join(d, f"r{n}.c")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_CARD4_HEAD + body + "\n")
            run = subprocess.run([exe, path], capture_output=True, text=True)
            stage = "CPP-ERR" if body.startswith("#") else "PARSE-ERR"
            assert run.returncode == 1 and run.stdout.strip() == f"{stage} {why}", (
                body,
                run.returncode,
                run.stdout[:200],
            )
        for n, body in enumerate(_CARD4_LOWERED):
            _rtfp_run_unit(f"l{n}", _CARD4_HEAD + body + "\n", exe, d)


# TC23: an `_Atomic` member of a file-scope array of structs and a member array of atomics, each reached at
# a runtime index -- the two indexed forms `atomic_object` emits through the element's own pointer type.
_TC23_INDEXED_ATOMIC_UNIT = """#include <stdint.h>
#include <stddef.h>
#include <string.h>
struct at { _Atomic uint32_t x; uint32_t y; };
static struct at gt[2][2];
struct ma { uint32_t k; _Atomic uint32_t arr[4]; };
static struct ma gm;
uint32_t f(uint32_t s) { gt[s & 1u][1].x = s; gt[s & 1u][1].x += 2u; return gt[s & 1u][1].x + gt[1][0].y; }
uint32_t g(uint32_t s) { gm.arr[s & 3u] = s; gm.arr[(s >> 1) & 3u]++; return gm.arr[s & 3u] + gm.k; }
"""


def test_an_indexed_atomic_member_of_a_defined_object_is_a_lock_free_atomic_under_clang():
    """TC23: an `_Atomic` member of a file-scope array of structs, or of a member array, reached at a runtime
    index is one lock-free atomic instruction in both emits under the host Clang -- never a libatomic call.
    Clang 23 gives a pointer computed from a *defined* object by `char` arithmetic with a runtime index the
    alignment of `char`, and lowers an atomic access it cannot prove aligned to `__atomic_load` /
    `__atomic_store` / `__atomic_compare_exchange`, which no freestanding link defines; Clang 18 inlined the
    same access. Both rails now reach the element through its own pointer type from the constant-offset
    base, `(*((_Atomic T *)((char *)base + off) + (size_t)idx * K))`, so the alignment is the element's on
    either Clang. The judge is Clang's own IR at -O2: the emitted `f` and `g` carry their atomic
    instructions and no `@__atomic_*` call. Under `BCIR_REQUIRE_LLVM` (both CI oracle jobs) the absence of
    Clang is a failure, not a skip (L2)."""
    clang = shutil.which("clang")
    if not clang:
        assert not os.environ.get("BCIR_REQUIRE_LLVM"), (
            "BCIR_REQUIRE_LLVM is set and no Clang judges the atomic lowering"
        )
        return
    unit = _TC23_INDEXED_ATOMIC_UNIT
    oracle_summary, r, _entry = _oracle(unit)
    assert "ok=1" in oracle_summary, oracle_summary
    oracle_emit = "\n".join(r.emitted[name] for name in r.lowered.functions)
    exe = _build_frontend(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "unit.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(unit)
        c_summary, c_emit = _c_run(exe, path)
        assert c_summary == oracle_summary, (c_summary, oracle_summary)
        for rail, emit in (("twin", c_emit), ("oracle", oracle_emit)):
            # the indexed forms, through the element's pointer type: the array of structs (stride 8 over a
            # 4-byte element, K = 2) and the member array (stride == element size, K = 1)
            assert "(*((_Atomic uint32_t *)((char *)gt + 0) + (size_t)" in emit, (rail, emit)
            assert re.search(r"\* 2\)\)", emit), (rail, emit)
            assert "(*((_Atomic uint32_t *)((char *)&gm + 4) + (size_t)" in emit, (rail, emit)
            assert re.search(r"\* 1\)\)", emit), (rail, emit)
            cpath = os.path.join(d, f"{rail}.c")
            keep = "\nvoid *bcir_tc23_keep[] = {(void *)bcir_f, (void *)bcir_g};\n"  # static: keep
            with open(cpath, "w", encoding="utf-8") as fh:
                fh.write(unit + "\n" + emit + keep)
            run = subprocess.run(
                [clang, "-std=c2x", "-O2", "-w", "-S", "-emit-llvm", "-o", "-", cpath],
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, (rail, run.stderr[-2000:])
            for fn in ("bcir_f", "bcir_g"):
                m = re.search(rf"define [^@\n]*@{fn}\(.*?\n\}}", run.stdout, re.S)
                assert m, (rail, fn, run.stdout[:400])
                body = m.group(0)
                atomics = re.findall(r"\b(?:atomicrmw|load atomic|store atomic|cmpxchg)\b", body)
                assert len(atomics) >= 3, (rail, fn, atomics, body)
                libcalls = re.findall(r"call [^\n]*@__atomic_\w+", body)
                assert not libcalls, (rail, fn, libcalls)
