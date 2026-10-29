"""KERNEL-ALIGN's GEMM placement check: runtime/c/bcir_tensor.c as a shared library for ab_gemm, its .text
PLACEMENT bytes past a 64-byte boundary of the library's image (a pad linked before it, solved by relinking).
usage: build_so.py TENSOR_C CC OUT PLACEMENT [FLAG...]  -> OUT, one line: its text's start mod 64 and the pad"""

import subprocess
import sys
from pathlib import Path


def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def main():
    src, cc, out, place = (
        Path(sys.argv[1]).resolve(),
        sys.argv[2],
        Path(sys.argv[3]).resolve(),
        int(sys.argv[4]),
    )
    flags = sys.argv[5:]
    work = out.parent / (out.name + ".d")
    work.mkdir(parents=True, exist_ok=True)
    obj = work / "t.o"
    run(
        cc,
        "-std=c11",
        "-O3",
        "-ffp-contract=off",
        "-fno-math-errno",
        "-pthread",
        "-fPIC",
        "-Wall",
        "-Wextra",
        "-Werror",
        *flags,
        "-I",
        str(src.parent),
        "-c",
        str(src),
        "-o",
        str(obj),
    )
    first = min(
        (int(f[0], 16), f[2])
        for f in (ln.split() for ln in run("nm", "--defined-only", "-g", str(obj)).splitlines())
        if len(f) == 3 and f[1] == "T"
    )
    pad = 0
    for _ in range(8):
        s = work / "pad.s"
        s.write_text(
            '.section .note.GNU-stack,"",@progbits\n'
            + (f".text\n.p2align 4\n.skip {pad}\n" if pad else "")
        )
        run(cc, "-c", str(s), "-o", str(work / "pad.o"))
        run(cc, "-shared", "-pthread", str(work / "pad.o"), str(obj), "-o", str(out), "-lm")
        at = [
            int(f[0], 16)
            for f in (ln.split() for ln in run("nm", "--defined-only", str(out)).splitlines())
            if len(f) == 3 and f[2] == first[1]
        ][0] - first[0]
        if at % 64 == place:
            print(f"{out.name}: text@{at % 64} pad {pad} {' '.join(flags) or '(default)'}")
            return
        pad = (pad + place - at % 64) % 64
    raise SystemExit(f"{out.name}: the pad did not converge")


main()
