"""SOFTMAX-T's witness on its own: each fault of softmax-t.json applied to a copy of the tensor unit, the trainer's C
harness built twice -- whole (its main as committed) and with exp_scalef alone in main -- and each run; records the
assertion each stops at (the case that caught the defect), or that it passed. Without -Werror (the trimmed main
leaves the other cases unused); C11, -O2 -ffp-contract=off, as the gate's C11 build. A control run without a fault
first; both builds must pass it.
usage: case_alone.py REPO WORK OUT.json"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO, WORK, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
C = REPO / "runtime/c"
table = json.loads((REPO / "tools/testing/faults/softmax-t.json").read_text())
WORK.mkdir(parents=True, exist_ok=True)
src = WORK / "src"
if src.exists():
    shutil.rmtree(src)
src.mkdir()
for name in (
    "bcir_tensor.c",
    "bcir_tensor.h",
    "bcir_decoder_train.c",
    "bcir_decoder_train.h",
    "test_decoder_train.c",
):
    shutil.copyfile(C / name, src / name)
unit = (src / "bcir_tensor.c").read_text()
harness = (src / "test_decoder_train.c").read_text()
main_at = harness.index("int main(void) {")
alone = harness[:main_at] + "int main(void) {\n  exp_scalef();\n  return 0;\n}\n"
(src / "alone.c").write_text(alone)
ASSERT = re.compile(r"(\w+\.c):(\d+): (?:[\w ]*?\b)?(\w+)\([^)]*\): Assertion")


def build(tag, harness_file, unit_file):
    exe = WORK / f"h_{tag}"
    cmd = [
        "clang",
        "-std=c11",
        "-O2",
        "-ffp-contract=off",
        "-w",
        "-I",
        str(src),
        str(harness_file),
        str(unit_file),
        str(src / "bcir_decoder_train.c"),
        "-o",
        str(exe),
        "-lm",
        "-lpthread",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode:
        return None, r.stderr[-400:]
    return exe, ""


def run(exe):
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
    m = ASSERT.search(r.stderr)
    return {
        "exit": r.returncode,
        "passes": r.stdout.count("PASS"),
        "stopped_in": m.group(3) if m else None,
        "line": int(m.group(2)) if m else None,
        "stderr_tail": r.stderr[-300:] if r.returncode and not m else "",
    }


results = []
for label, unit_text in [("control (no fault)", unit)] + [
    (f["label"], None) for f in table["faults"]
]:
    if unit_text is None:
        f = next(f for f in table["faults"] if f["label"] == label)
        assert unit.count(f["old"]) == 1, label
        unit_text = unit.replace(f["old"], f["new"], 1)
    uf = WORK / "unit.c"
    uf.write_text(unit_text)
    row = {"label": label}
    for tag, hf in (("whole", src / "test_decoder_train.c"), ("alone", src / "alone.c")):
        exe, err = build(tag, hf, uf)
        row[tag] = run(exe) if exe else {"build_failed": err}
    results.append(row)
    print(json.dumps(row), flush=True)
assert results[0]["whole"]["exit"] == 0 and results[0]["alone"]["exit"] == 0, (
    "the control must pass"
)
OUT.write_text(json.dumps({"what": __doc__, "results": results}, indent=1) + "\n")
