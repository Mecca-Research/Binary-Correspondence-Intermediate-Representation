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
byte-identical twin in `runtime/c/bcir_cfront.c`'s `bcir_lib_for_callee`), e.g. a `fftw_*`-prefix rule
-> `-lfftw3`. Nothing else changes: the derivation, the `--emit-link-flags` surface, and the parity
gate all read this table.

The C twin (`bcir_cfront.c`: `bcir_lib_for_callee` + `bcir_cfront_link_flags`) mirrors this table and
the sort byte-for-byte, and `bcir/tests/test_c_cfront.py` + `tools/c/check_runtime.sh` gate the parity.
"""
from __future__ import annotations

from .lower import _EXTERN_VARIADIC, _LIBM, _LIBM_INT, _STDLIB_ALLOC

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
    `free`, and the printf/scanf-family <stdio.h> variadics. These are real external edges but need no
    flag. Classified explicitly (NO_FLAG) rather than left to the unknown default, so the mapping is a
    complete statement of what BCIR knows about its own emitted external seams."""
    return callee == "free" or callee in _STDLIB_ALLOC or callee in _EXTERN_VARIADIC


# The callee -> library classification, an ORDERED list of `(matcher, flag)` rules. The first matching
# rule wins; `flag` is the `-l...` string, or `NO_FLAG` ("") for a known-but-implicit symbol. This is
# the single EXTENSION POINT: B2 adds one rule per newly-wrapped library here (and its byte-identical
# twin in bcir_cfront.c). Keep both rails' tables in the SAME ORDER -- the first-match semantics make
# order significant, and the parity gate compares the derived flags, not the rule list.
_LIBRARY_RULES: tuple[tuple, ...] = (
    # <math.h> (incl. <complex.h>, which links libm too) -> -lm. The math seam is the dominant case.
    (_is_libm, "-lm"),
    # libc-implicit (malloc/free/realloc/calloc/aligned_alloc + the printf/scanf family) -> no flag.
    (_is_libc_implicit, NO_FLAG),
    # B5 BLAS: cblas_sgemm and any cblas_* (CBLAS) -> -lcblas. Matches the existing B5 path's choice
    # (bcir/lower/c_kernel.py emit_blas_gemm_c links `-lcblas`); stay consistent so a BLAS unit links
    # with one flag regardless of which emitter produced the call.
    (lambda c: c.startswith("cblas_"), "-lcblas"),
    # B2 FFTW: fftwf_* (single-precision) and fftw_* (double) -> -lfftw3. The B2 wrap
    # (bcir/lower/c_kernel.py emit_fftw_fft_c) links `-lfftw3` for the trusted fftwf_plan_dft_1d /
    # fftwf_execute / fftwf_destroy_plan edge; this rule makes a unit with an FFTW edge link it
    # automatically. (libfftw3 is the single-prec build too -- fftwf_* lives in -lfftw3, not -lfftw3f.)
    (lambda c: c.startswith("fftw_") or c.startswith("fftwf_"), "-lfftw3"),
    # --- B2 EXTENSION POINT (roadmap): add one rule per newly-wrapped trusted library, e.g.
    #   (lambda c: c.startswith("LAPACKE_") or c.startswith("dge"), "-llapack"), # LAPACK
    #   (lambda c: c.startswith("gsl_"), "-lgsl"),                                # GSL
    #   (lambda c: c.startswith("Sleef_"), "-lsleef"),                            # SLEEF
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
            return flag
    return None


def _callee_of(op: str) -> str | None:
    """The external callee named by a claim op, or None if the op is not an external-call edge."""
    for prefix in _EXTERN_CALL_PREFIXES:
        if op.startswith(prefix):
            return op[len(prefix):]
    return None


def link_flags_for_callees(callees) -> list[str]:
    """The deduped, stably-sorted link flags a set/iterable of external callees needs. Sorted so the
    output is REPRODUCIBLE (a BCIR hard requirement): the same callees always yield byte-identical
    flags, independent of claim order or set iteration order."""
    flags = set()
    for callee in callees:
        flag = library_for_callee(callee)
        if flag:                                          # skip NO_FLAG ("") and unknown (None)
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
