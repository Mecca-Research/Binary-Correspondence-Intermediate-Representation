"""NEXT-9's copies: the benches' kernels are the measured tree's and the prototype's, byte for byte. hoist_bench.c's
exponent_le, q8_scale and q8_terms512 are TREE's runtime/c/bcir_ai_kernels.c's, its ref512 that unit's
q8_matvec_block512 renamed, and its q8_xs_exact and q8_matvec_block512_xs the lines q8_hoist.patch adds; rows_bench.c
holds the unit's exponent_le and q8_scale and its row-dot blocks (from the codes' transpose to the AVX-512 block,
the AVX2 block left out). Prints one JSON object; exits 1 if any copy differs.
usage: copies.py TREE HERE"""

import json
import sys
from pathlib import Path

tree, here = Path(sys.argv[1]), Path(sys.argv[2])
unit = (tree / "runtime" / "c" / "bcir_ai_kernels.c").read_text()
bench = (here / "hoist_bench.c").read_text()
rows = (here / "rows_bench.c").read_text()
patch = (here / "q8_hoist.patch").read_text()
added = (
    "\n".join(
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    + "\n"
)


def function(text, signature):
    """The function whose definition starts with SIGNATURE, through its closing brace at the line's start."""
    start = text.index(signature)
    return text[start : text.index("\n}\n", start) + 3]


EXPONENT, SCALE = "static int exponent_le(", "static double q8_scale("
TERMS = '__attribute__((target("avx512f"))) static inline __m512d q8_terms512('
BLOCK = '__attribute__((target("avx512f"))) static void q8_matvec_block512('
EXACT = "static int q8_xs_exact("
PROTO = '__attribute__((target("avx512f"))) static int q8_matvec_block512_xs('
checks = {
    "hoist_bench.c exponent_le": function(bench, EXPONENT) == function(unit, EXPONENT),
    "hoist_bench.c q8_scale": function(bench, SCALE) == function(unit, SCALE),
    "hoist_bench.c q8_terms512": function(bench, TERMS) == function(unit, TERMS),
    "hoist_bench.c ref512": function(bench, BLOCK.replace("q8_matvec_block512(", "ref512("))
    == function(unit, BLOCK).replace("q8_matvec_block512(", "ref512(", 1),
    "hoist_bench.c q8_xs_exact": function(bench, EXACT) == function(added, EXACT),
    "hoist_bench.c q8_matvec_block512_xs": function(bench, PROTO) == function(added, PROTO),
    "rows_bench.c exponent_le": function(rows, EXPONENT) == function(unit, EXPONENT),
    "rows_bench.c q8_scale": function(rows, SCALE) == function(unit, SCALE),
}
start = unit.index("/* 8 rows x 8 coordinates of codes, the rows `ld` apart, transposed")
end = unit.index("/* every row eight at a time, each block's totals checked in row order", start)
avx2 = unit.index(
    "/* Rows row0 .. row0+rows-1 (at most 8) of the row dots, a row's chain a lane: four lanes a",
    start,
)
avx512 = unit.index("/* The same block, eight lanes a register. */", avx2)
checks["rows_bench.c the row-dot blocks"] = unit[start:avx2] + unit[avx512:end] in rows
print(json.dumps({"copies": checks, "every_copy_the_same_bytes": all(checks.values())}))
sys.exit(0 if all(checks.values()) else 1)
