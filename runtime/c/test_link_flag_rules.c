/* The C twin's link-flag rules (bcir_cfront_link_flags), one rule per argument: the five probes the
 * runtime gate wrote out one file each (#linkflags-fftw, -lapack, -gsl, -sleef, -cerf), as one harness
 * the gate and the CMake build compile alike (BUILD-2h, docs/BCIR_BUILD_ROADMAP.md). The trusted-library
 * edges (cblas_*, fftwf_*, LAPACKE_*, gsl_*, Sleef_*, erfcx*) are minted by the kernel EMITTERS, never
 * reached from a cfront source -- an unknown callee lowers to an in-unit `c.call:` edge -- so each table
 * drives bcir_cfront_link_flags over FABRICATED one-claim units carrying a `c.call.libm:` edge: its own
 * rule, then the rules before it and libm and an unknown callee, which must not regress. The oracle
 * (linkflags.library_for_callee) is pinned in bcir/tests/test_c_cfront.py.
 *
 *   usage: test_link_flag_rules fftw|lapack|gsl|sleef|cerf
 *
 * Exit 0 with `OK linkflags-<rule>` when every edge of the table derives its flag, 1 with a FAIL line
 * per edge that does not, 2 on a usage error. */
#include <stdio.h>
#include <string.h>
#include "bcir_cir.h"
#include "bcir_cfront.h"
/* Build a one-function unit whose single claim is the given external-call edge op, then derive its flags. */
static const char *derive(const char *op) {
  static char buf[128];
  bcir_claim cl; memset(&cl, 0, sizeof cl);
  snprintf(cl.op, sizeof cl.op, "%s", op);
  bcir_func f; memset(&f, 0, sizeof f);
  f.claims = &cl; f.n_claims = 1;
  bcir_unit u; memset(&u, 0, sizeof u);
  u.funcs = &f; u.n_funcs = 1;
  bcir_cfront_link_flags(&u, buf, sizeof buf);
  return buf;
}
static int eq(const char *op, const char *want) {
  const char *got = derive(op);
  if (strcmp(got, want)) { printf("FAIL %s -> '%s' want '%s'\n", op, got, want); return 0; }
  return 1;
}

struct edge { const char *op, *want; };

static const struct edge fftw_edges[] = {
  {"c.call.libm:fftwf_execute", "-lfftw3"},      /* the B2 rule */
  {"c.call.libm:fftwf_plan_dft_1d", "-lfftw3"},  /* any fftwf_* */
  {"c.call.libm:fftw_execute", "-lfftw3"},       /* the double-prec fftw_* prefix */
  {"c.call.libm:cblas_sgemm", "-lcblas"},        /* B5 (no regression) */
  {"c.call.libm:sqrt", "-lm"},                   /* libm (no regression) */
  {"c.call.libm:totally_unknown_fn", ""},        /* unknown -> no flag (no regression) */
};

static const struct edge lapack_edges[] = {
  {"c.call.libm:LAPACKE_sgesv", "-llapack"},     /* the LAPACK rule (the wrapper's actual callee) */
  {"c.call.libm:LAPACKE_dgesv", "-llapack"},     /* any LAPACKE_* */
  {"c.call.libm:LAPACKE_sgels", "-llapack"},     /* E1 OLS: LAPACKE_sgels rides the SAME LAPACKE_* rule */
  {"c.call.libm:LAPACKE_ssyev", "-llapack"},     /* E2 PCA: LAPACKE_ssyev rides the SAME LAPACKE_* rule */
  {"c.call.libm:sgesv_", "-llapack"},            /* the Fortran-ABI driver symbol */
  {"c.call.libm:fftwf_execute", "-lfftw3"},      /* B2 (no regression) */
  {"c.call.libm:cblas_sgemm", "-lcblas"},        /* B5 (no regression) */
  {"c.call.libm:sqrt", "-lm"},                   /* libm (no regression) */
  {"c.call.libm:totally_unknown_fn", ""},        /* unknown -> no flag (no regression) */
};

static const struct edge gsl_edges[] = {
  {"c.call.libm:gsl_stats_mean", "-lgsl"},       /* the GSL rule (the wrapper's actual callee) */
  {"c.call.libm:gsl_stats_variance", "-lgsl"},   /* any gsl_* */
  {"c.call.libm:gsl_sf_erf", "-lgsl"},           /* a special-function gsl_* too */
  {"c.call.libm:LAPACKE_sgesv", "-llapack"},     /* #61 LAPACK (no regression) */
  {"c.call.libm:fftwf_execute", "-lfftw3"},      /* B2 (no regression) */
  {"c.call.libm:cblas_sgemm", "-lcblas"},        /* B5 (no regression) */
  {"c.call.libm:sqrt", "-lm"},                   /* libm (no regression) */
  {"c.call.libm:totally_unknown_fn", ""},        /* unknown -> no flag (no regression) */
};

static const struct edge sleef_edges[] = {
  {"c.call.libm:Sleef_expf1_u10", "-lsleef"},    /* the SLEEF rule (the wrapper's actual callee) */
  {"c.call.libm:Sleef_sinf1_u10", "-lsleef"},    /* any Sleef_* */
  {"c.call.libm:gsl_stats_mean", "-lgsl"},       /* #62 GSL (no regression) */
  {"c.call.libm:LAPACKE_sgesv", "-llapack"},     /* #61 LAPACK (no regression) */
  {"c.call.libm:fftwf_execute", "-lfftw3"},      /* B2 (no regression) */
  {"c.call.libm:cblas_sgemm", "-lcblas"},        /* B5 (no regression) */
  {"c.call.libm:expf", "-lm"},                   /* libm (no regression -- the SLEEF fallback's twin) */
  {"c.call.libm:totally_unknown_fn", ""},        /* unknown -> no flag (no regression) */
};

static const struct edge cerf_edges[] = {
  {"c.call.libm:erfcxf", "-lcerf"},              /* the libcerf rule (the wrapper's actual callee) */
  {"c.call.libm:erfcx", "-lcerf"},               /* the bare/double erfcx too */
  {"c.call.libm:erfcf", "-lm"},                  /* erfcf/erfc are still libm (NOT shadowed by -lcerf) */
  {"c.call.libm:Sleef_expf1_u10", "-lsleef"},    /* #63 SLEEF (no regression) */
  {"c.call.libm:gsl_stats_mean", "-lgsl"},       /* #62 GSL (no regression) */
  {"c.call.libm:LAPACKE_sgesv", "-llapack"},     /* #61 LAPACK (no regression) */
  {"c.call.libm:fftwf_execute", "-lfftw3"},      /* B2 (no regression) */
  {"c.call.libm:cblas_sgemm", "-lcblas"},        /* B5 (no regression) */
  {"c.call.libm:expf", "-lm"},                   /* libm (no regression -- the erfcx fallback's twin) */
  {"c.call.libm:totally_unknown_fn", ""},        /* unknown -> no flag (no regression) */
};

static const struct rule { const char *name; const struct edge *edges; size_t n; } rules[] = {
  {"fftw", fftw_edges, sizeof fftw_edges / sizeof fftw_edges[0]},
  {"lapack", lapack_edges, sizeof lapack_edges / sizeof lapack_edges[0]},
  {"gsl", gsl_edges, sizeof gsl_edges / sizeof gsl_edges[0]},
  {"sleef", sleef_edges, sizeof sleef_edges / sizeof sleef_edges[0]},
  {"cerf", cerf_edges, sizeof cerf_edges / sizeof cerf_edges[0]},
};

int main(int argc, char **argv) {
  if (argc != 2) { fputs("usage: test_link_flag_rules fftw|lapack|gsl|sleef|cerf\n", stderr); return 2; }
  for (size_t r = 0; r < sizeof rules / sizeof rules[0]; r++) {
    if (strcmp(argv[1], rules[r].name)) continue;
    int ok = 1;
    for (size_t i = 0; i < rules[r].n; i++) ok &= eq(rules[r].edges[i].op, rules[r].edges[i].want);
    if (ok) printf("OK linkflags-%s\n", rules[r].name);
    return ok ? 0 : 1;
  }
  fprintf(stderr, "test_link_flag_rules: no rule '%s' (fftw, lapack, gsl, sleef, cerf)\n", argv[1]);
  return 2;
}
