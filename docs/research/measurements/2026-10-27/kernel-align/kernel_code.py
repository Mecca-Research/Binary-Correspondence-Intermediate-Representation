"""KERNEL-ALIGN's code record. `objects`: each source of the decoder's units (the bcir_models library) compiled as
the tree before's manifest and the tree after's give it under CC (-std=c11 -O2 -Wall -Wextra -Werror and the unit's
options, read by the tree after's bcir.toolchain.manifest_options) -- its .text size and alignment -- and in the
kernel unit each hot kernel's instructions, NOPs and padding prefixes, all and inside its loops. `layout`: each copy of
each hot kernel in a binary build_align.py linked (copies in link order, the arms named in that order), where it lies
mod 64, and each of its loops -- every backward branch's [target, branch] -- with its branches and how many of them
cross or end on a 32-byte boundary, a macro-fused compare and branch counted whole (the condition Intel's JCC
erratum names). One JSON line a row.
usage: kernel_code.py objects CC BEFORE AFTER WORK   |   kernel_code.py layout BINARY LABEL..."""

import json
import re
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
    "q8_run_avx512_xs",
    "q8_run_avx2_xs",
)
FUSERS = ("cmp", "test", "add", "sub", "and", "inc", "dec")
LINE = re.compile(r"^\s*([0-9a-f]+):\t([0-9a-f ]+)\t(.*)$")
HEAD = re.compile(r"^([0-9a-f]+) <([^>]+)>:$")


def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def functions(path):
    """{name: [(address, length, text)]}, every instance of a name in address order."""
    found, cur = [], None
    for line in run("objdump", "-d", "-w", str(path)).splitlines():
        h = HEAD.match(line)
        if h:
            cur = (h.group(2), int(h.group(1), 16), [])
            found.append(cur)
            continue
        m = LINE.match(line)
        if m and cur is not None:
            cur[2].append((int(m.group(1), 16), len(m.group(2).split()), m.group(3).strip()))
    out = {}
    for name, _, insns in found:
        base = name.split(".", 1)[0]
        if insns:
            out.setdefault(base, []).append(insns)
    for v in out.values():
        v.sort(key=lambda insns: insns[0][0])
    return out


def is_nop(text):
    return bool(re.match(r"^(?:(?:data16|cs|ds|rex\.\w*)\s+)*(?:nop\w*|xchg\s+%ax,%ax)\b", text))


def is_prefixed(text):
    return bool(re.match(r"^(?:cs|ds|ss|es)\s+(?!nop)", text))


def loops(insns):
    out = []
    for a, _, text in insns:
        t = re.search(r"\bj\w+\s+([0-9a-f]+) <", text)
        if t and int(t.group(1), 16) <= a:
            out.append((int(t.group(1), 16), a))
    return out


def branch_hits(insns):
    """(address, kind, hit) a branch -- a jump (jcc, jmp, a fused compare and jcc) or a call or return -- and
    whether it, with the compare it fuses with, crosses or ends on 32 bytes."""
    res = []
    for n, (a, ln, text) in enumerate(insns):
        mn = text.split()[0] if text else ""
        if not (mn.startswith("j") or mn.startswith("call") or mn.startswith("ret")):
            continue
        start = a
        if mn.startswith("j") and not mn.startswith("jmp") and n > 0:
            pa, pl, pt = insns[n - 1]
            pm = pt.split()[0] if pt else ""
            if pm.startswith(FUSERS) and pa + pl == a and not ("$" in pt and "(" in pt):
                start = pa
        end = a + ln
        kind = "jump" if mn.startswith("j") else "call" if mn.startswith("call") else "ret"
        res.append((a, kind, start // 32 != (end - 1) // 32 or end % 32 == 0))
    return res


def objects(cc, before, after, work):
    sys.path.insert(0, str(after))
    import bcir.toolchain as toolchain

    family, machine = toolchain.compiler_identity(cc)
    work.mkdir(parents=True, exist_ok=True)
    for tree, side in ((before, "before"), (after, "after")):
        manifest = json.loads((tree / "runtime" / "manifest.json").read_text(encoding="utf-8"))
        unit = manifest["libraries"]["bcir_models"]
        flags = toolchain.manifest_options(unit.get("options", []), family, machine)
        for source in unit["sources"]:
            obj = work / f"{side}-{source[:-2]}.o"
            c = tree / "runtime" / "c"
            run(
                cc,
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                *flags,
                "-I",
                str(c),
                "-c",
                str(c / source),
                "-o",
                str(obj),
            )
            size = align = None
            for line in run("readelf", "-SW", str(obj)).splitlines():
                f = line.replace("[ ", "[").split()
                if len(f) > 6 and f[1] == ".text":
                    size, align = int(f[5], 16), int(f[-1])
            row = {
                "compiler": family,
                "machine": machine,
                "tree": side,
                "source": source,
                "flags": flags,
                "text_bytes": size,
                "text_alignment": align,
            }
            if source == "bcir_ai_kernels.c":
                funcs = functions(obj)
                row["hot_kernels"] = {}
                for h in HOT:
                    insns = funcs.get(h, [[]])[0]
                    spans = loops(insns)
                    inside = [x for x in insns if any(lo <= x[0] <= hi for lo, hi in spans)]
                    row["hot_kernels"][h] = {
                        "instructions": len(insns),
                        "nops": sum(is_nop(t) for _, _, t in insns),
                        "prefixed": sum(is_prefixed(t) for _, _, t in insns),
                        "in_loops": len(inside),
                        "nops_in_loops": sum(is_nop(t) for _, _, t in inside),
                        "prefixed_in_loops": sum(is_prefixed(t) for _, _, t in inside),
                    }
            print(json.dumps(row))


def layout(binary, labels):
    funcs = functions(binary)
    for h in HOT:
        copies = funcs.get(h, [])
        if len(copies) != len(labels):
            raise SystemExit(f"{h}: {len(copies)} copies for {len(labels)} arms")
        for label, insns in zip(labels, copies):
            branches = branch_hits(insns)
            spans = sorted(loops(insns), key=lambda x: x[1] - x[0])
            rows = []
            for lo, hi in spans:
                inside = [(kind, hit) for a, kind, hit in branches if lo <= a <= hi]
                rows.append(
                    {
                        "offset": lo - insns[0][0],
                        "bytes": hi - lo,
                        "head_mod_32": lo % 32,
                        "branches": len(inside),
                        "jump_hits": sum(h for k, h in inside if k == "jump"),
                        "call_hits": sum(h for k, h in inside if k != "jump"),
                    }
                )
            looped = [
                (kind, hit) for a, kind, hit in branches if any(lo <= a <= hi for lo, hi in spans)
            ]
            print(
                json.dumps(
                    {
                        "arm": label,
                        "kernel": h,
                        "start_mod_64": insns[0][0] % 64,
                        "loops": rows,
                        "branches_in_loops": len(looped),
                        "jump_hits_in_loops": sum(h for k, h in looped if k == "jump"),
                        "call_hits_in_loops": sum(h for k, h in looped if k != "jump"),
                    }
                )
            )


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "objects":
        objects(
            sys.argv[2],
            Path(sys.argv[3]).resolve(),
            Path(sys.argv[4]).resolve(),
            Path(sys.argv[5]).resolve(),
        )
    elif len(sys.argv) > 3 and sys.argv[1] == "layout":
        layout(Path(sys.argv[2]), sys.argv[3:])
    else:
        sys.exit(__doc__.rsplit("usage: ", 1)[1])
