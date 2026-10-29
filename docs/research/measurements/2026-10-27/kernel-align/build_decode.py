"""KERNEL-ALIGN's decode placement build: decode_place linked from copies of builds of the C decoder's units
(bcir_ai_kernels.c first, then bcir_q8_model.c, bcir_llama.c and bcir_decode.c, partially linked into one object a
build, every global renamed LABEL_), each copy's text PLACEMENT bytes past a 64-byte boundary -- its kernel unit's
text first -- with the pads before the copies solved by relinking. Each unit compiles -std=c11 -O2 -Wall -Wextra
-Werror with the options its tree's runtime/manifest.json gives it under CC, read by the tree after's
bcir.toolchain.manifest_options (the tree before names no scoped option, so it reads the same under either): a
build is `before` (the tree before's units), `after` (the tree after's) or `kernels` (the tree after's kernel unit
with its options, the other units with the tree before's) -- an arm a build at a placement.
usage: build_decode.py BEFORE AFTER CC OUT ARM...   ARM = LABEL:PLACEMENT:BUILD  -> OUT, one line a copy"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

UNITS = ("bcir_ai_kernels.c", "bcir_q8_model.c", "bcir_llama.c", "bcir_decode.c")


def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def text_align(obj):
    for line in run("readelf", "-SW", obj).splitlines():
        f = line.replace("[ ", "[").split()
        if len(f) > 2 and f[1] == ".text":
            return int(f[-1])
    raise SystemExit(f"{obj}: no .text")


def unit_flags(tree, source, family, machine, toolchain):
    """The options the tree's manifest gives the unit that owns `source`, resolved."""
    manifest = json.loads((tree / "runtime" / "manifest.json").read_text(encoding="utf-8"))
    for units in manifest.values():
        if isinstance(units, dict):
            for unit in units.values():
                if isinstance(unit, dict) and source in unit.get("sources", []):
                    return toolchain.manifest_options(unit.get("options", []), family, machine)
    raise SystemExit(f"{tree}: no unit compiles {source}")


def main():
    before, after = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    cc, out = sys.argv[3], Path(sys.argv[4]).resolve()
    arms = [(lab, int(p), b) for lab, p, b in (s.split(":") for s in sys.argv[5:])]
    sys.path.insert(0, str(after))
    import bcir.toolchain as toolchain

    family, machine = toolchain.compiler_identity(cc)
    here = Path(__file__).resolve().parent
    work = out.parent / (out.name + ".d")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    builds = {}
    for label, place, build in arms:
        if build not in builds:
            if build not in ("before", "after", "kernels"):
                raise SystemExit(f"{label}: build {build!r} is none of before, after, kernels")
            objs, flags = [], {}
            for u in UNITS:
                tree = (
                    after if build == "after" or (build == "kernels" and u == UNITS[0]) else before
                )
                flags[u] = unit_flags(tree, u, family, machine, toolchain)
                o = work / f"{build}-{u[:-2]}.o"
                c = tree / "runtime" / "c"
                run(
                    cc,
                    "-std=c11",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    *flags[u],
                    "-I",
                    str(c),
                    "-c",
                    str(c / u),
                    "-o",
                    str(o),
                )
                objs.append(str(o))
            whole = work / f"{build}.o"
            run("ld", "-r", "-o", str(whole), *objs)
            g = [ln.split() for ln in run("nm", "--defined-only", "-g", str(whole)).splitlines()]
            first = min((int(f[0], 16), f[2]) for f in g if len(f) == 3 and f[1] == "T")
            builds[build] = (
                whole,
                text_align(str(whole)),
                first,
                sorted(f[2] for f in g if len(f) == 3),
                flags,
            )
        whole, align, first, names, _ = builds[build]
        if place % align:
            raise SystemExit(f"{label}: placement {place} under a {align}-byte text alignment")
        mp = work / f"{label}.map"
        mp.write_text("".join(f"{n} {label}_{n}\n" for n in names))
        run("objcopy", f"--redefine-syms={mp}", str(whole), str(work / f"{label}.o"))
    (work / "arms.h").write_text("".join(f"ARM({label})\n" for label, _, _ in arms))
    pads = [0] * len(arms)

    def link():
        objs = []
        for k, (label, _, _) in enumerate(arms):
            s = work / f"pad{k}.s"
            s.write_text(
                '.section .note.GNU-stack,"",@progbits\n'
                + (f".text\n.p2align 4\n.skip {pads[k]}\n" if pads[k] else "")
            )
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
            str(after / "runtime" / "c"),
            "-I",
            str(work),
            str(here / "decode_place.c"),
            *objs,
            "-o",
            str(out),
            "-lm",
        )
        syms = {}
        for ln in run("nm", "--defined-only", str(out)).splitlines():
            f = ln.split()
            if len(f) == 3:
                syms.setdefault(f[2], int(f[0], 16))
        return [syms[f"{label}_{builds[b][2][1]}"] - builds[b][2][0] for label, _, b in arms]

    for _ in range(4 * len(arms) + 4):
        starts = link()
        wrong = [k for k, (_, place, _) in enumerate(arms) if starts[k] % 64 != place]
        if not wrong:
            break
        k = wrong[0]
        pads[k] = (pads[k] + arms[k][1] - starts[k] % 64) % 64
    else:
        raise SystemExit("the pads did not converge")
    for k, (label, place, build) in enumerate(arms):
        flags = builds[build][4]
        print(
            json.dumps(
                {
                    "arm": label,
                    "build": build,
                    "placement": starts[k] % 64,
                    "pad": pads[k],
                    "text_alignment": builds[build][1],
                    "compiler": family,
                    "machine": machine,
                    "flags": {u: flags[u] for u in UNITS},
                }
            )
        )


main()
