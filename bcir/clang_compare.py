"""bcir.clang_compare -- a fair, rigorous BCIR-vs-Clang comparison harness.

BCIR is a *planning* layer that delegates instruction selection / register
allocation to LLVM (it emits C; Clang compiles it). So the honest comparison holds
the **compiler constant** (one Clang, one flag set) and pits BCIR's *planned*
realization against the *naive/default* one a developer would write -- isolating
BCIR's planning contribution from codegen. Two questions:

  1. Where BCIR cannot beat Clang (simple, already-optimal kernels), does it at least
     MATCH -- i.e. does its emitted code regress nothing?  (It emits the idiomatic
     loop Clang vectorizes, so it should tie.)
  2. Where BCIR makes a *semantic* choice Clang's backend cannot (avoid a gather,
     reduce in a cache-friendly order, stride instead of gather), does it WIN, and by
     how much, measured on silicon?

Methodology (this matters -- cache-resident kernels are noise-sensitive): each
variant is compiled into its **own** binary, each binary self-times best-of-R, and
the harness runs the two binaries **alternated** for N trials and reports the
**median** ns/call -- cancelling machine drift and the in-process warm-cache order
bias that a single-binary A/B suffers. A ratio in [0.95, 1.05] is reported as MATCH.

Offline tooling; never the hot path. Skips cleanly without clang/cc/gcc.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from shutil import which

from .examples import saxpy_strided, vector_add
from .kbcir import TARGETS, optimize
from .kbcir.cost import Theta
from .lower.c_kernel import (
    emit_gather_kernel_c,
    emit_kernel_c,
    emit_reduce_c,
    emit_strided_c,
    find_reduce,
    find_strided,
)

_HARNESS = """
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <stdint.h>
static long now_ns(void){ struct timespec t; timespec_get(&t, TIME_UTC);
  return (long)t.tv_sec*1000000000L + t.tv_nsec; }
static uint64_t g=0xBC12u; static uint64_t lcg(void){ g=g*6364136223846793005ULL+1442695040888963407ULL; return g; }
"""


def clang_available() -> bool:
    return which("clang") is not None or which("cc") is not None or which("gcc") is not None


def _cc():
    return which("clang") or which("cc") or which("gcc")


@dataclass(frozen=True)
class Verdict:
    name: str
    bcir_ns: int
    naive_ns: int
    opt: str
    n: int

    @property
    def speedup(self) -> float:
        return self.naive_ns / self.bcir_ns if self.bcir_ns else 0.0

    @property
    def category(self) -> str:
        s = self.speedup
        return "WIN" if s >= 1.05 else "LOSE" if s <= 0.95 else "MATCH"


def _build(src: str, opt: str, d: str, tag: str) -> str | None:
    cc = _cc()
    path = os.path.join(d, f"{tag}.c")
    exe = os.path.join(d, tag)
    with open(path, "w") as f:
        f.write(src)
    for std in ("-std=gnu23", "-std=gnu2x"):
        b = subprocess.run([cc, std, opt, path, "-o", exe], capture_output=True, text=True)
        if b.returncode == 0:
            return exe
    return None


def _median_ns(exe: str, args: list[str]) -> int:
    out = subprocess.run([exe, *args], capture_output=True, text=True)
    return int(out.stdout.strip())


def _ab(bcir_src: str, naive_src: str, *, opt: str, args: list[str], trials: int = 15) -> tuple[int, int]:
    """Compile both to separate binaries; run alternated; return (median_bcir, median_naive)."""
    with tempfile.TemporaryDirectory() as d:
        be = _build(bcir_src, opt, d, "bcir")
        ne = _build(naive_src, opt, d, "naive")
        if not be or not ne:
            return 0, 0
        bs, ns = [], []
        for _ in range(trials):
            bs.append(_median_ns(be, args))
            ns.append(_median_ns(ne, args))
        return int(statistics.median(bs)), int(statistics.median(ns))


# --- the kernels: BCIR-planned vs the naive/default realization -------------------

def _elementwise_srcs(n: int):
    """Simple C = A + B. BCIR emits the go-fast idiomatic loop; naive is the loop a
    developer writes. Same compiler -> this measures whether BCIR REGRESSES a simple
    kernel (it should MATCH)."""
    m = vector_add(n)
    h = TARGETS["x86_avx512"]
    bcir_k = emit_kernel_c(m, optimize(m, h, Theta.cool()), "k", "f32", hw_width=h.vector_width)
    naive_k = ("#include <stddef.h>\nvoid k(const float*restrict A,const float*restrict B,"
               "float*restrict C,size_t n){for(size_t i=0;i<n;++i)C[i]=A[i]+B[i];}\n")
    main = (_HARNESS + f"""
int main(int c,char**v){{ size_t n={n}u; int reps=atoi(v[1]);
  float*A=malloc(n*4),*B=malloc(n*4),*C=malloc(n*4);
  for(size_t i=0;i<n;++i){{A[i]=(float)(i%1000);B[i]=(float)(i%7);C[i]=0;}}
  k(A,B,C,n); long best=LONG_MAX;
  for(int r=0;r<reps;++r){{long t0=now_ns();k(A,B,C,n);long dt=now_ns()-t0;if(dt>0&&dt<best)best=dt;}}
  volatile float s=C[n-1];(void)s; printf("%ld\\n",best); return 0; }}
""")
    return bcir_k + main, naive_k + main


def _gather_srcs(n: int):
    """BCIR picks the contiguous realization; the naive code gathers `A[idx[i]]` with
    a shuffled permutation. BCIR WINS by avoiding the gather (a choice Clang's backend
    cannot make -- it does not know the gather is avoidable)."""
    m = vector_add(n)
    h = TARGETS["x86_avx512"]
    r = optimize(m, h, Theta.cool())
    bcir_k = emit_kernel_c(m, r, "kd", "f32", hw_width=h.vector_width)            # contiguous
    gather_k = emit_gather_kernel_c(m, r, "kg", "f32")                            # A[idx[i]]+B[i]
    main_b = (_HARNESS + f"""
int main(int c,char**v){{ size_t n={n}u; int reps=atoi(v[1]);
  float*A=malloc(n*4),*B=malloc(n*4),*C=malloc(n*4);
  for(size_t i=0;i<n;++i){{A[i]=(float)(i%1000);B[i]=(float)(i%7);C[i]=0;}}
  kd(A,B,C,n); long best=LONG_MAX;
  for(int r=0;r<reps;++r){{long t0=now_ns();kd(A,B,C,n);long dt=now_ns()-t0;if(dt>0&&dt<best)best=dt;}}
  volatile float s=C[n-1];(void)s; printf("%ld\\n",best); return 0; }}
""")
    main_n = (_HARNESS + f"""
int main(int c,char**v){{ size_t n={n}u; int reps=atoi(v[1]);
  float*A=malloc(n*4),*B=malloc(n*4),*C=malloc(n*4); long*idx=malloc(n*sizeof*idx);
  for(size_t i=0;i<n;++i){{A[i]=(float)(i%1000);B[i]=(float)(i%7);C[i]=0;idx[i]=(long)i;}}
  for(size_t i=n-1;i>0;--i){{size_t j=(size_t)(lcg()%(i+1));long t=idx[i];idx[i]=idx[j];idx[j]=t;}}
  kg(A,B,C,idx,n); long best=LONG_MAX;
  for(int r=0;r<reps;++r){{long t0=now_ns();kg(A,B,C,idx,n);long dt=now_ns()-t0;if(dt>0&&dt<best)best=dt;}}
  volatile float s=C[n-1];(void)s; printf("%ld\\n",best); return 0; }}
""")
    return bcir_k + main_b, gather_k + main_n


def _reduce_srcs(n: int):
    """A reducible permutation: BCIR reduces in blocked (sequential) order; naive
    gathers `T[idx[i]]`. Integer sum is associative -> identical result, so the
    gather is pure waste. BCIR WINS."""
    m = __import__("bcir.examples", fromlist=["gather_reduce"]).gather_reduce(n)
    h = TARGETS["x86_avx512"]
    r = optimize(m, h, Theta.cool())
    blk = emit_reduce_c(m, r, "kb", "i32", gather=False)
    grd = emit_reduce_c(m, r, "kr", "i32", gather=True)
    main_b = (_HARNESS + f"""
int main(int c,char**v){{ size_t n={n}u; int reps=atoi(v[1]);
  int32_t*T=malloc(n*4); for(size_t i=0;i<n;++i)T[i]=(int32_t)(i%1000);
  volatile int32_t s=kb(T,n); long best=LONG_MAX;
  for(int r=0;r<reps;++r){{long t0=now_ns();s=kb(T,n);long dt=now_ns()-t0;if(dt>0&&dt<best)best=dt;}}
  (void)s; printf("%ld\\n",best); return 0; }}
""")
    main_n = (_HARNESS + f"""
int main(int c,char**v){{ size_t n={n}u; int reps=atoi(v[1]);
  int32_t*T=malloc(n*4); long*idx=malloc(n*sizeof*idx);
  for(size_t i=0;i<n;++i){{T[i]=(int32_t)(i%1000);idx[i]=(long)i;}}
  for(size_t i=n-1;i>0;--i){{size_t j=(size_t)(lcg()%(i+1));long t=idx[i];idx[i]=idx[j];idx[j]=t;}}
  volatile int32_t s=kr(T,idx,n); long best=LONG_MAX;
  for(int r=0;r<reps;++r){{long t0=now_ns();s=kr(T,idx,n);long dt=now_ns()-t0;if(dt>0&&dt<best)best=dt;}}
  (void)s; printf("%ld\\n",best); return 0; }}
""")
    return blk + main_b, grd + main_n


def _strided_srcs(n: int):
    """Y[i] = X[i*k]: BCIR emits direct strided addressing; naive gathers X[idx[i]]
    with idx[i]=i*k. Same elements, but BCIR avoids the gather instruction. WIN."""
    m = saxpy_strided(n)
    h = TARGETS["x86_avx512"]
    r = optimize(m, h, Theta.cool())
    claim, _ = find_strided(m, r)
    k = max(1, claim.stride_k)
    direct = emit_strided_c(m, r, "ks", "f32", gather=False)
    gath = emit_strided_c(m, r, "kg", "f32", gather=True)
    main_b = (_HARNESS + f"""
int main(int c,char**v){{ size_t n={n}u,k={k}u; int reps=atoi(v[1]);
  float*X=malloc(n*k*4),*Y=malloc(n*4); for(size_t i=0;i<n*k;++i)X[i]=(float)(i%1000);
  ks(X,Y,n,k); long best=LONG_MAX;
  for(int r=0;r<reps;++r){{long t0=now_ns();ks(X,Y,n,k);long dt=now_ns()-t0;if(dt>0&&dt<best)best=dt;}}
  volatile float s=Y[n-1];(void)s; printf("%ld\\n",best); return 0; }}
""")
    main_n = (_HARNESS + f"""
int main(int c,char**v){{ size_t n={n}u,k={k}u; int reps=atoi(v[1]);
  float*X=malloc(n*k*4),*Y=malloc(n*4); long*idx=malloc(n*sizeof*idx);
  for(size_t i=0;i<n*k;++i)X[i]=(float)(i%1000); for(size_t i=0;i<n;++i)idx[i]=(long)(i*k);
  kg(X,idx,Y,n); long best=LONG_MAX;
  for(int r=0;r<reps;++r){{long t0=now_ns();kg(X,idx,Y,n);long dt=now_ns()-t0;if(dt>0&&dt<best)best=dt;}}
  volatile float s=Y[n-1];(void)s; printf("%ld\\n",best); return 0; }}
""")
    return direct + main_b, gath + main_n


# --- the suite -------------------------------------------------------------------

def run_comparison(trials: int = 15) -> list[Verdict]:
    """The full BCIR-planned vs naive comparison, same Clang, separate binaries,
    alternated, median-of-`trials`. Returns one Verdict per category."""
    if not clang_available():
        return []
    out: list[Verdict] = []
    cases = [
        ("elementwise stream (N=4M, -O3)", *_elementwise_srcs(1 << 22), "-O3", 1 << 22, "200"),
        ("elementwise L1 (N=4K, -O3)",     *_elementwise_srcs(1 << 12), "-O3", 1 << 12, "4000"),
        ("gather avoidance (shuffled, -O2)", *_gather_srcs(1 << 20), "-O2", 1 << 20, "60"),
        ("reduction blocked-vs-gather (-O2)", *_reduce_srcs(1 << 20), "-O2", 1 << 20, "60"),
        ("strided-vs-gather (-O2)",        *_strided_srcs(1 << 22), "-O2", 1 << 22, "60"),
    ]
    for name, bsrc, nsrc, opt, n, reps in cases:
        bn, nn = _ab(bsrc, nsrc, opt=opt, args=[reps], trials=trials)
        if bn and nn:
            out.append(Verdict(name=name, bcir_ns=bn, naive_ns=nn, opt=opt, n=n))
    return out


def main() -> int:
    rows = run_comparison()
    if not rows:
        print("no C compiler available; skipping BCIR-vs-Clang comparison")
        return 0
    print(f"{'category':36} {'BCIR ns':>10} {'naive ns':>10} {'speedup':>8}  verdict")
    for v in rows:
        print(f"{v.name:36} {v.bcir_ns:>10} {v.naive_ns:>10} {v.speedup:>7.2f}x  {v.category}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
