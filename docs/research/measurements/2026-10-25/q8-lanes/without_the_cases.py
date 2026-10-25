"""Q8-LANES' vacuity control: turn a worktree of the Q8-LANES tree into the witnesses there were before it -- the
kernel harness's cases without q8_lanes_test (runtime/c/test_ai_kernels.c), built once as tools/c/check_runtime.sh
builds it (bcir/tests/test_native_ai.py's harness test, its C11 build alone, the Q8-LANES line not asked for), and
the matrix kernels' oracle test -- and write the fault table q8-lanes.json with that command, so a sweep reads
which of its faults those witnesses catch. Every edit is exact text that occurs once.
usage: without_the_cases.py WORKTREE OUT_TABLE.json"""

import json
import sys
from pathlib import Path

tree, out = Path(sys.argv[1]), Path(sys.argv[2])


def edit(path, old, new):
    p = tree / path
    text = p.read_text()
    if text.count(old) != 1:
        sys.exit(f"{path}: the anchor occurs {text.count(old)} times")
    p.write_text(text.replace(old, new))


edit("runtime/c/test_ai_kernels.c", "  CHECK(q8_lanes_test() == 0);\n", "  (void)q8_lanes_test;\n")
edit(
    "bcir/tests/test_native_ai.py",
    """    builds = (
        ("c11", [], None),
        ("c2x", [], None),
        ("c2x", ["-DBCIR_AI_MAX_INSTANCE=2"], 3),
        ("c2x", ["-DBCIR_AI_NO_DISPATCH"], 1),
    )
""",
    """    builds = (("c11", [], None),)
""",
)
edit(
    "bcir/tests/test_native_ai.py",
    """            found = re.search(
                r"Q8-LANES: instances (\\d+), (\\d+) vector comparisons equal", result.stdout
            )
            assert found, result.stdout
            admitted, compared = int(found.group(1)), int(found.group(2))
            assert admitted & 1 and not admitted & ~7
            if cap is not None:
                assert not admitted & ~cap, (options, admitted)
            # 22 shapes, four exponent ranges, two kernels: 176 comparisons an admitted vector instance
            assert compared == 176 * bin(admitted & 6).count("1"), (options, admitted, compared)
            seen.add(admitted)
    native = _native()
    if native is not None:
        assert native.q8_instances() in seen  # the default builds admit what the library does
""",
    "",
)
table = json.loads((tree / "tools/testing/faults/q8-lanes.json").read_text())
command = table["command"]
dropped = (
    "bcir.tests.test_native_ai:test_native_q8_instances_match_the_increasing_coordinate_oracle"
)
if command.count(dropped) != 1:
    sys.exit("the table's command does not name the instance test once")
command.remove(dropped)
out.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
