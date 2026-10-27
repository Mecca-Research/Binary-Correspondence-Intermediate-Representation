"""KERNEL-ALIGN's kernel placement build: align_ab linked from copies of builds of one kernel unit (-std=c11 -O2
-ffp-contract=off -Wall -Wextra -Werror and the arm's flags), each copy's exported symbols renamed LABEL_ and its
.text PLACEMENT bytes past a 64-byte boundary, with the pads before the copies solved by relinking until every copy
starts where it should (a copy whose build aligns its text more coarsely than its placement is refused: no link can
place it so).
usage: build_align.py KERNELS_C CC OUT ARM...   ARM = LABEL:PLACEMENT[:FLAG[ FLAG...]]
-> OUT, and one JSON line a copy: its label, flags, text alignment and start mod 64, its hot kernels' starts mod 64"""

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HOT = (
    "q8_matvec_pair512_xs",
    "q8_matvec_pair512",
    "q8_matvec_pair256_xs",
    "q8_matvec_pair256",
    "q8_rows_pack512x4",
    "q8_rows_pack256x4",
)


def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def text_align(obj):
    for line in run("readelf", "-SW", obj).splitlines():
        f = line.replace("[ ", "[").split()
        if len(f) > 2 and f[1] == ".text":
            return int(f[-1])
    raise SystemExit(f"{obj}: no .text")


def symbols(path, globals_only=False):
    args = ["nm", "--defined-only"] + (["-g"] if globals_only else []) + [path]
    out = {}
    for line in run(*args).splitlines():
        f = line.split()
        if len(f) == 3:
            out.setdefault(f[2], []).append((int(f[0], 16), f[1]))
    return out


def main():
    src, cc, out = Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3]).resolve()
    arms = []
    for spec in sys.argv[4:]:
        label, place, *rest = spec.split(":", 2)
        arms.append((label, int(place), shlex.split(rest[0]) if rest else []))
    here = Path(__file__).resolve().parent
    work = out.parent / (out.name + ".d")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    builds = {}
    for label, place, flags in arms:
        key = " ".join(flags)
        if key not in builds:
            obj = work / f"v{len(builds)}.o"
            run(
                cc,
                "-std=c11",
                "-O2",
                "-ffp-contract=off",
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
            g = symbols(str(obj), globals_only=True)
            first = min(((a[0][0], n) for n, a in g.items() if a[0][1] == "T"))
            builds[key] = (obj, text_align(str(obj)), first, sorted(g))
        obj, align, first, names = builds[key]
        if place % align:
            raise SystemExit(f"{label}: placement {place} under a {align}-byte text alignment")
        mp = work / f"{label}.map"
        mp.write_text("".join(f"{n} {label}_{n}\n" for n in names))
        run("objcopy", f"--redefine-syms={mp}", str(obj), str(work / f"{label}.o"))
    (work / "arms.h").write_text("".join(f"ARM({label})\n" for label, _, _ in arms))
    pads = [0] * len(arms)

    def link():
        objs = []
        for k, (label, _, _) in enumerate(arms):
            s = work / f"pad{k}.s"
            body = '.section .note.GNU-stack,"",@progbits\n'
            if pads[k]:
                body += f".text\n.p2align 4\n.skip {pads[k]}\n"
            s.write_text(body)
            run(cc, "-c", str(s), "-o", str(work / f"pad{k}.o"))
            objs += [str(work / f"pad{k}.o"), str(work / f"{label}.o")]
        run(
            cc,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(work),
            str(here / "align_ab.c"),
            *objs,
            "-o",
            str(out),
            "-lm",
        )
        syms = symbols(str(out))
        starts = []
        for label, _, flags in arms:
            off, name = builds[" ".join(flags)][2]
            starts.append(syms[f"{label}_{name}"][0][0] - off)
        return starts

    for _ in range(4 * len(arms) + 4):
        starts = link()
        wrong = [k for k, (_, place, _) in enumerate(arms) if starts[k] % 64 != place]
        if not wrong:
            break
        k = wrong[0]
        pads[k] = (pads[k] + arms[k][1] - starts[k] % 64) % 64
    else:
        raise SystemExit("the pads did not converge")
    # each copy's hot kernels: local symbols, one a copy, in link order
    syms = symbols(str(out))
    for k, (label, place, flags) in enumerate(arms):
        lo = starts[k]
        hi = starts[k + 1] if k + 1 < len(arms) else float("inf")
        hot = {}
        for h in HOT:
            at = sorted(
                a
                for n, v in syms.items()
                if n == h or n.startswith(h + ".")
                for a, _ in v
                if lo <= a < hi
            )
            hot[h] = at[0] % 64 if at else None
        print(
            json.dumps(
                {
                    "arm": label,
                    "flags": flags,
                    "text_alignment": builds[" ".join(flags)][1],
                    "placement": starts[k] % 64,
                    "pad": pads[k],
                    "hot_kernels_mod_64": hot,
                }
            )
        )


main()
