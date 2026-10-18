"""oneDNN's bf16 matmul (through PyTorch) on the 9.84M decoder's 13 GEMM shapes at THREADS threads:
bf16 operands already converted (the GEMM alone), and from fp32 operands converted per call (what
BCIR's entry does: pack + compute). Median ms of REPS calls; GFLOP/s."""

import sys
import time

import torch

threads = int(sys.argv[1]) if len(sys.argv) > 1 else 4
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 31
torch.set_num_threads(threads)
S = [
    ("fwd q", 0, 1, 1024, 384, 384),
    ("fwd k", 0, 1, 1024, 192, 384),
    ("fwd gate", 0, 1, 1024, 1024, 384),
    ("fwd down", 0, 1, 1024, 384, 1024),
    ("fwd head", 0, 1, 1024, 260, 384),
    ("dx q", 0, 0, 1024, 384, 384),
    ("dx k", 0, 0, 1024, 384, 192),
    ("dx gate", 0, 0, 1024, 384, 1024),
    ("dx down", 0, 0, 1024, 1024, 384),
    ("dW q", 1, 0, 384, 384, 1024),
    ("dW k", 1, 0, 192, 384, 1024),
    ("dW gate", 1, 0, 1024, 384, 1024),
    ("dW down", 1, 0, 384, 1024, 1024),
]
print(
    f"torch {torch.__version__} threads {threads} mkldnn bf16 {torch.backends.mkldnn.is_available()}"
)
tb = tf = 0.0
for name, ta, tbf, m, n, k in S:
    a32 = torch.randn(k, m) if ta else torch.randn(m, k)
    b32 = torch.randn(n, k) if tbf else torch.randn(k, n)
    a16, b16 = a32.bfloat16(), b32.bfloat16()

    def mm16(a16=a16, b16=b16, ta=ta, tbf=tbf):
        a = a16.t() if ta else a16
        b = b16.t() if tbf else b16
        return a @ b

    def mm32(a32=a32, b32=b32, ta=ta, tbf=tbf):
        a = (a32.t() if ta else a32).bfloat16()
        b = (b32.t() if tbf else b32).bfloat16()
        return a @ b

    res = []
    for f in (mm16, mm32):
        for _ in range(3):
            f()
        t = []
        for _ in range(reps):
            t0 = time.perf_counter()
            f()
            t.append(time.perf_counter() - t0)
        t.sort()
        res.append(t[reps // 2])
    fl = 2.0 * m * n * k
    tb += res[0]
    tf += res[1]
    print(
        f"{name:9s} {m:5d} {n:5d} {k:5d}  bf16 operands {res[0] * 1e3:7.3f} ms {fl / res[0] / 1e9:6.0f} GF/s | "
        f"from fp32 {res[1] * 1e3:7.3f} ms {fl / res[1] / 1e9:6.0f} GF/s"
    )
print(f"sum: bf16 operands {tb * 1e3:.3f} ms, from fp32 {tf * 1e3:.3f} ms")
