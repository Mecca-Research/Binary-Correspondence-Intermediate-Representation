"""DECODE-THREADS' kernel code: the whole-call vector kernels of Q8-PAIR (q8_matvec_pair512, q8_matvec_pair512_xs,
q8_matvec_pair256, q8_matvec_pair256_xs) as three builds of runtime/c/bcir_ai_kernels.c compile them -- the tree before
DECODE-THREADS (BEFORE), DECODE-THREADS as first committed (FIRST: one kernel for the whole call and a range, its column
range a variable) and the tree after (AFTER: the kernel's whole-call instance) -- from objdump's listing, call targets
and RIP-relative displacements masked. For each kernel and build: its instructions; the instructions FIRST's and
AFTER's differ from BEFORE's by (difflib, lines removed plus lines added) as they are -- 0 where a build's kernel is
BEFORE's instruction for instruction, its registers and stack slots included -- and with every register and stack
offset masked; and its stack accesses in its vector loops (each backward branch's span holding at least four vector
instructions; the memory operands on the stack in it, summed over those loops).
usage: kernel_code.py OBJDUMP BEFORE.o FIRST.o AFTER.o  -- one JSON object"""

import difflib
import json
import re
import subprocess
import sys

KERNELS = ("q8_matvec_pair512", "q8_matvec_pair512_xs", "q8_matvec_pair256", "q8_matvec_pair256_xs")
GPR = r"\b(r(?:[a-d]x|[sd]i|[sb]p|8|9|1[0-5])[dwb]?|e(?:[a-d]x|[sd]i|[sb]p)|[a-d][lh]|[sd]il|[sb]pl)\b"


def listing(objdump, obj, name):
    """the kernel's (address, instruction) pairs, or None where the build has no such function"""
    text = subprocess.run(
        [objdump, "-d", "--no-show-raw-insn", "-M", "intel", f"--disassemble={name}", obj],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    lines = text.splitlines()
    start = [i for i, line in enumerate(lines) if line.endswith(f"<{name}>:")]
    if len(start) != 1:
        return None
    rows = []
    for line in lines[start[0] + 1 :]:
        found = re.match(r"\s*([0-9a-f]+):\s+(\S.*)$", line)
        if not found:
            break
        rows.append((int(found.group(1), 16), re.sub(r"\s+", " ", found.group(2)).strip()))
    return rows


def plain(ins):
    ins = re.sub(r"[0-9a-f]+ <[^>]*>", "<target>", ins)
    return re.sub(r"\[rip\+0x[0-9a-f]+\]", "[rip+disp]", ins)


def masked(ins):
    ins = re.sub(r"\b[xyz]mm\d+\b", "v", plain(ins))
    ins = re.sub(r"\bk[0-7]\b", "k", ins)
    return re.sub(r"\[r\+0x[0-9a-f]+\]", "[r+off]", re.sub(GPR, "r", ins))


def loop_stack_accesses(rows):
    addresses = [a for a, _ in rows]
    total = 0
    for i, (address, ins) in enumerate(rows):
        branch = re.match(r"j\w+ ([0-9a-f]+) <", ins)
        if (
            not branch
            or int(branch.group(1), 16) > address
            or int(branch.group(1), 16) not in addresses
        ):
            continue
        span = [x for _, x in rows[addresses.index(int(branch.group(1), 16)) : i + 1]]
        if sum(1 for x in span if re.search(r"\b[xyz]mm\d", x)) >= 4:
            total += sum(1 for x in span if "[rsp" in x or "[rbp" in x)
    return total


def differ(a, b):
    return sum(
        1
        for line in difflib.unified_diff(a, b, lineterm="", n=0)
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    )


def main():
    objdump, objects = sys.argv[1], dict(zip(("before", "first", "after"), sys.argv[2:5]))
    rows = {}
    for name in KERNELS:
        built = {build: listing(objdump, obj, name) for build, obj in objects.items()}
        if any(v is None for v in built.values()):
            rows[name] = None
            continue
        text = {build: [plain(x) for _, x in v] for build, v in built.items()}
        shape = {build: [masked(x) for _, x in v] for build, v in built.items()}
        rows[name] = {
            "instructions": {build: len(v) for build, v in text.items()},
            "first_differs_by": differ(text["before"], text["first"]),
            "after_differs_by": differ(text["before"], text["after"]),
            "first_differs_by_masked": differ(shape["before"], shape["first"]),
            "after_differs_by_masked": differ(shape["before"], shape["after"]),
            "vector_loop_stack_accesses": {b: loop_stack_accesses(v) for b, v in built.items()},
        }
    print(json.dumps({"kernels": rows}))


main()
