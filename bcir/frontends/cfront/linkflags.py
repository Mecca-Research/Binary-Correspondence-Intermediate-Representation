"""Derive the linker flags a translation unit actually needs from the external-call edges it uses
(roadmap B1) — the dual-rail SOURCE OF TRUTH for the callee->library mapping.

Today the C frontend emits external calls through the claim-graph edges `c.call.libm:<name>`,
`c.call.libm.void:<name>`, and `c.call.extern:<name>` (the math.h / libc / printf-family seams; the
B5 `cblas_sgemm` BLAS wrap rides `c.call.libm:` too). The compiler emits the call but never told the
build system WHICH libraries to link, so every harness hard-codes `-lm` and BLAS hard-codes `-lcblas`.
This module closes that gap: it maps each external callee to the library flag it needs, scans a lowered
unit's claims, and produces a DEDUPED, STABLY-SORTED flag list (reproducibility is a hard BCIR
requirement — the same unit always yields byte-identical flags).

EXTENSIBILITY (the B2 hook): the classification is a single ordered table of `(matcher, flag)` rules,
`_LIBRARY_RULES`. B2 (wrapping FFTW / LAPACK / GSL / SLEEF) adds one rule per library here (and the
byte-identical twin in `runtime/c/bcir_cfront.c`'s `bcir_lib_for_callee`), e.g. a `fftwf_*`-prefix rule
-> `-lfftw3f`. Nothing else changes: the derivation, the `--emit-link-flags` surface, and the parity
gate all read this table.

The C twin (`bcir_cfront.c`: `bcir_lib_for_callee` + `bcir_cfront_link_flags`) mirrors this table and
the sort byte-for-byte, and `bcir/tests/test_c_cfront.py` + `tools/c/check_runtime.sh` gate the parity.
"""

from __future__ import annotations

from .lower import _EXTERN_VARIADIC, _LIBM, _LIBM_INT, _STDLIB_ALLOC, _STRING_MEM

# The external-call claim-op prefixes whose `:<callee>` suffix names a real (non-bcir_) symbol the
# linker must resolve. A unit's link flags are derived purely from the callees behind these edges.
#   c.call.libm:NAME       -- a <math.h> / <complex.h> value call, or the B5 cblas_sgemm BLAS wrap
#   c.call.libm.void:NAME  -- a void external (free)
#   c.call.extern:NAME     -- a printf/scanf-family <stdio.h> variadic
# (c.call.builtin: is a compiler intrinsic with no library; c.call:/c.call.void: are in-unit bcir_
# twins; c.call.indirect/imember go through a pointer -- none name an external library symbol.)
_EXTERN_CALL_PREFIXES = ("c.call.libm:", "c.call.libm.void:", "c.call.extern:")

# A flag string meaning "this callee resolves with NO extra link flag". This is an EXPLICIT
# classification (libc is linked implicitly by every toolchain), NOT an omission: malloc/free/printf
# and friends are real external symbols, but they need no `-l`. Distinguishing "known, needs no flag"
# from "unknown" keeps the unknown-callee policy honest (see `library_for_callee`).
NO_FLAG = ""


def _is_libm(callee: str) -> bool:
    """True if `callee` is a <math.h> function (mirrors lower._libm_type's name recognition): a base
    name, an f/l-suffixed real variant, or a fixed-integer one (ilogb / lround...). Resolves to -lm.
    The full name is tested before stripping a trailing f/l so a base that ends in f/l (`erf`, `modf`,
    the `*l` long-double forms) is classified by the table, never as a phantom suffix of a shorter base."""
    if callee in _LIBM or callee in _LIBM_INT:
        return True
    base = callee[:-1]
    return callee[-1:] in "fl" and (base in _LIBM or base in _LIBM_INT)


def _is_libc_implicit(callee: str) -> bool:
    """True if `callee` is a libc symbol linked implicitly (no `-l`): the <stdlib.h> allocators +
    `free`, the <string.h> memory routines, and the printf/scanf-family <stdio.h> variadics. These are
    real external edges but need no flag. Classified explicitly (NO_FLAG) rather than left to the unknown
    default, so the mapping is a complete statement of what BCIR knows about its own emitted external
    seams."""
    return (
        callee == "free"
        or callee in _STDLIB_ALLOC
        or callee in _STRING_MEM
        or callee in _EXTERN_VARIADIC
    )


# B-breadth (#61) LAPACK: the Fortran-ABI LU/solve driver base names (the trailing-underscore form a C
# caller links against, e.g. `sgesv_`). The LAPACKE C interface (`LAPACKE_*`) is matched by prefix; this
# set covers a direct Fortran-symbol edge. Kept tiny and explicit -- exactly the LU/solve family the wrap
# touches, so an unrelated `foo_` callee is NOT swept into -llapack. The C twin lists the identical names.
_LAPACK_FORTRAN = frozenset({"sgesv", "dgesv", "sgetrf", "dgetrf", "sgetrs", "dgetrs"})

# FFTW 3 builds one library per precision, each prefixing its symbols (FFTW-PRECISION): double `fftw_`
# (libfftw3), single `fftwf_` (libfftw3f), long double `fftwl_` (libfftw3l), quad `fftwq_` (libfftw3q), and
# the MPI interface `<prefix>mpi_` in `libfftw3<p>_mpi` (no quad one). Read from the libraries' own dynamic
# symbol tables (FFTW 3.3.10): libfftw3 defines no `fftwf_` symbol. The threading entry points are defined
# alike by the POSIX-threads library (`libfftw3<p>_threads`) and the OpenMP one (`libfftw3<p>_omp`), so they
# name no one library: unknown, the build's choice (the unknown-callee policy of `library_for_callee`).
_FFTW_PRECISIONS = (
    ("fftw_", "-lfftw3", "-lfftw3_mpi"),
    ("fftwf_", "-lfftw3f", "-lfftw3f_mpi"),
    ("fftwl_", "-lfftw3l", "-lfftw3l_mpi"),
    ("fftwq_", "-lfftw3q", None),
)
_FFTW_THREADS = frozenset(
    {
        "init_threads",
        "cleanup_threads",
        "plan_with_nthreads",
        "planner_nthreads",
        "make_planner_thread_safe",
        "threads_set_callback",
    }
)


def _is_fftw(callee: str) -> bool:
    return callee.startswith(tuple(prefix for prefix, _lib, _mpi in _FFTW_PRECISIONS))


def _fftw_library(callee: str) -> str | None:
    """The FFTW library a `fftw*_` callee resolves in: its precision's, or that precision's MPI library for
    an `mpi_` entry point; None (unknown) for a threading entry point, which two libraries define alike."""
    for prefix, lib, mpi in _FFTW_PRECISIONS:
        if callee.startswith(prefix):
            rest = callee[len(prefix) :]
            if rest in _FFTW_THREADS:
                return None
            return mpi if rest.startswith("mpi_") else lib
    return None


# The callee -> library classification, an ORDERED list of `(matcher, flag)` rules. The first matching
# rule wins; `flag` is the `-l...` string, `NO_FLAG` ("") for a known-but-implicit symbol, or a function of the
# callee giving one of those or None (FFTW's, whose library depends on the rest of the name). This is
# the single EXTENSION POINT: B2 adds one rule per newly-wrapped library here (and its byte-identical
# twin in bcir_cfront.c). Keep both rails' tables in the SAME ORDER -- the first-match semantics make
# order significant, and the parity gate compares the derived flags, not the rule list.
_LIBRARY_RULES: tuple[tuple, ...] = (
    # <math.h> (incl. <complex.h>, which links libm too) -> -lm. The math seam is the dominant case.
    (_is_libm, "-lm"),
    # libc-implicit (malloc/free/realloc/calloc/aligned_alloc, memcpy/memmove/memset + the printf/scanf
    # family) -> no flag.
    (_is_libc_implicit, NO_FLAG),
    # B5 BLAS: cblas_sgemm and any cblas_* (CBLAS) -> -lcblas. Matches the existing B5 path's choice
    # (bcir/lower/c_kernel.py emit_blas_gemm_c links `-lcblas`); stay consistent so a BLAS unit links
    # with one flag regardless of which emitter produced the call.
    (lambda c: c.startswith("cblas_"), "-lcblas"),
    # B2 FFTW: each precision's library (`_fftw_library`): fftwf_* -> -lfftw3f, the single-precision
    # library the B2 wrap (bcir/lower/c_kernel.py emit_fftw_fft_c) calls -- fftwf_plan_dft_1d /
    # fftwf_execute / fftwf_destroy_plan -- and the one the dependency index probes first
    # (`bcir.toolchain`'s FFTW3F); fftw_* -> -lfftw3, fftwl_* -> -lfftw3l, fftwq_* -> -lfftw3q, and an
    # `mpi_` entry point its precision's MPI library. Until FFTW-PRECISION both fftw_* and fftwf_* mapped to
    # -lfftw3, which defines no fftwf_ symbol: a unit with a single-precision edge did not link.
    (_is_fftw, _fftw_library),
    # B-breadth (#61) LAPACK: the LAPACKE C interface (LAPACKE_sgesv et al.) and the Fortran-ABI driver
    # symbols (sgesv_/dgesv_/sgetrf_/...) -> -llapack. The linear-solve wrap (bcir/lower/c_kernel.py
    # emit_lapack_solve_c) calls `LAPACKE_sgesv` and links `-llapacke -llapack`; libllapacke depends on
    # liblapack, so -llapack is the load-bearing flag a unit with a LAPACK edge needs (the LAPACKE_ prefix
    # rule covers the wrapper's actual callee). Matches the C twin's branch in the SAME order (first match
    # wins). The trailing-underscore Fortran forms are matched too so a direct sgesv_ edge resolves.
    (
        lambda c: c.startswith("LAPACKE_") or (c.endswith("_") and c[:-1] in _LAPACK_FORTRAN),
        "-llapack",
    ),
    # Area-B breadth (#62) GSL: any gsl_* (the GNU Scientific Library -- special functions / statistics) ->
    # -lgsl. The statistics wrap (bcir/lower/c_kernel.py emit_gsl_stats_c) calls gsl_stats_mean/variance/sd
    # and links `-lgsl -lgslcblas` (gsl depends on a cblas); -lgsl is the load-bearing dependency a unit
    # with a GSL edge needs (mirrors how LAPACK emits the load-bearing -llapack though the call is
    # LAPACKE_*). Matches the C twin's branch in the SAME order (first match wins).
    (lambda c: c.startswith("gsl_"), "-lgsl"),
    # Area-B breadth (#63) SLEEF: any Sleef_* (the SIMD-oriented vectorized math library -- a fast,
    # vectorized libm) -> -lsleef. The vectorized-exp wrap (bcir/lower/c_kernel.py emit_sleef_exp_c) calls
    # Sleef_expf1_u10 and links `-lsleef`; this rule makes a unit with a SLEEF edge link it automatically.
    # SLEEF's vectorized transcendentals are the vectorized alternative to the libm expf the G1 activation
    # kernels use (so a Sleef_* edge is a trusted external just like the cblas/fftw/lapack/gsl edges).
    # Matches the C twin's branch in the SAME order (first match wins).
    (lambda c: c.startswith("Sleef_"), "-lsleef"),
    # Area-B breadth (SEG2) libcerf: erfcx / erfcxf (the SCALED COMPLEMENTARY ERROR FUNCTION
    # erfcx(x) = e^{x^2}*erfc(x)) -> -lcerf. The erfcx wrap (bcir/lower/c_kernel.py emit_cerf_erfcx_c) calls
    # erfcxf and links `-lcerf`; this rule makes a unit with a libcerf edge link it automatically. erfcx is a
    # NUMERICALLY-ROBUST SPECIAL FUNCTION libm does NOT provide (the naive expf(x*x)*erfcf(x) overflows for
    # moderately large x; libcerf's erfcx stays finite across the full real line) -- so this adds a capability,
    # not just a faster path (unlike SLEEF's exp, which duplicates libm expf). Note erfcx/erfcxf are NOT in
    # _LIBM/_LIBM_INT (libm has erf/erfc but not erfcx), so this rule is not shadowed by the _is_libm rule
    # above. Matches the C twin's branch in the SAME order (first match wins).
    (lambda c: c.startswith("erfcx"), "-lcerf"),
    # --- EXTENSION POINT (roadmap): add one rule per newly-wrapped trusted library here.
    # Each new rule's twin goes in bcir_cfront.c's bcir_lib_for_callee in the SAME order.
)


def library_for_callee(callee: str) -> str | None:
    """The link flag a single external `callee` needs: a `-l...` string, `NO_FLAG` ("") for a
    known-but-implicit libc symbol, or `None` if the callee is UNKNOWN to the mapping.

    Unknown-callee policy (deterministic): an unrecognized external callee returns `None` and
    contributes NO flag to the derived list. BCIR does not invent a `-l` it cannot justify -- an
    unknown symbol is the build system's to resolve (exactly today's behaviour, where the compiler
    emits nothing). `None` (unknown) is kept DISTINCT from `NO_FLAG` (known, needs no flag) so a caller
    that wants to surface unknown external edges can (e.g. a future diagnostic); the flag derivation
    treats both as "adds no flag", so the result stays deterministic and never silently wrong."""
    for matcher, flag in _LIBRARY_RULES:
        if matcher(callee):
            return flag(callee) if callable(flag) else flag
    return None


def _callee_of(op: str) -> str | None:
    """The external callee named by a claim op, or None if the op is not an external-call edge."""
    for prefix in _EXTERN_CALL_PREFIXES:
        if op.startswith(prefix):
            return op[len(prefix) :]
    return None


def link_flags_for_callees(callees) -> list[str]:
    """The deduped, stably-sorted link flags a set/iterable of external callees needs. Sorted so the
    output is REPRODUCIBLE (a BCIR hard requirement): the same callees always yield byte-identical
    flags, independent of claim order or set iteration order."""
    flags = set()
    for callee in callees:
        flag = library_for_callee(callee)
        if flag:  # skip NO_FLAG ("") and unknown (None)
            flags.add(flag)
    return sorted(flags)


def derive_link_flags(lowered) -> list[str]:
    """The link flags a whole lowered translation unit needs: scan every function's claim ops for the
    external-call edges, map each callee to its library, dedup, and return a STABLY-SORTED flag list.
    Empty for a pure-integer unit (no external edge). Deterministic by construction."""
    callees = set()
    for lf in lowered.functions.values():
        for c in lf.claims:
            callee = _callee_of(c.op)
            if callee is not None:
                callees.add(callee)
    return link_flags_for_callees(callees)


def format_link_flags(flags) -> str:
    """The one-line, space-separated rendering of a derived flag list (the `--emit-link-flags` body and
    the value side of the `--emit-c` `link_flags` header). Empty list -> empty string."""
    return " ".join(flags)
