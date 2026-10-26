"""NEXT-9's switched kernel unit: the tree after's runtime/c/bcir_ai_kernels.c (q8_hoist.patch applied) with the
block matvec's path chosen at run time by the global bcir_ai_q8_hoist_off: 0 the prototype as patched; 1 the
committed kernels as before it; 2 the committed block kernel's copy (q8_matvec_block512_copy: its text under another
name, an empty asm statement first so that GCC's identical-code folding keeps it apart), the committed arithmetic at
addresses of its own -- the placement control; 3 the prototype's block kernel with its checks taken out
(q8_matvec_block512_ceiling), the hoist's ceiling, which is exact only where every x*s is (as on the lab model's
activations, which a run's byte comparison shows) and is a measurement, never a kernel. Every other byte is the
patched unit's, so one binary holds the four paths at fixed addresses and switch_ab.c and clock_probe.c time them
against each other with nothing else moved.
usage: switch.py PATCHED_UNIT OUT"""

import sys
from pathlib import Path

args = sys.argv[1:]
if len(args) != 2:
    sys.exit(__doc__.rsplit("usage: ", 1)[1])
src = Path(args[0]).read_text()
SIGNATURE = '__attribute__((target("avx512f"))) static void q8_matvec_block512('
start = src.index(SIGNATURE)
block = src[start : src.index("\n}\n", start) + 3]
copy = block.replace("q8_matvec_block512(", "q8_matvec_block512_copy(", 1)
# an empty asm statement first, so that GCC's identical-code folding keeps the copy apart from the original
head = "    double *output) {\n"
assert copy.count(head) == 1
copy = copy.replace(
    head, head + '  __asm__ volatile("" ::: "memory"); /* NEXT-9: not folded into the original */\n'
)
PROTO = '__attribute__((target("avx512f"))) static int q8_matvec_block512_xs('
start = src.index(PROTO)
proto = src[start : src.index("\n}\n", start) + 3]
ceiling = proto.replace("q8_matvec_block512_xs(", "q8_matvec_block512_ceiling(", 1).replace(
    "Q8_HOIST_VECTOR", "Q8_CEILING_VECTOR"
)
for line in (
    "  int exact = 1;\n",
    "      exact &= q8_xs_exact(x, xs);\n",
    "      exact &= q8_xs_exact(x, xs);                                                                       \\\n",
    "    if (!exact) return 0;\n",
):
    assert ceiling.count(line) == 1, line
    ceiling = ceiling.replace(line, "")
assert "exact" not in ceiling, "a check is left in the ceiling"
edits = (
    (
        "/* Q8-HOIST first; a call that formed an inexact x * s computed again term by term, as before it */\n"
        "static void q8_matvec_lanes(",
        "/* NEXT-9's placement control: the committed block kernel's text under another name */\n"
        + copy
        + "/* NEXT-9's ceiling: the prototype's block kernel with its checks taken out (a measurement, not a kernel) */\n"
        + ceiling
        + "/* Q8-HOIST first; a call that formed an inexact x * s computed again term by term, as before it */\n"
        "extern int bcir_ai_q8_hoist_off;\n"
        "int bcir_ai_q8_hoist_off; /* NEXT-9's switch: 0 the prototype, 1 committed, 2 the copy, 3 the ceiling */\n"
        "static void q8_matvec_lanes(",
    ),
    (
        "  int block = instance == 4 && output_count % 8u == 0u && group_size >= 8u && !(group_size & (group_size - 1u));\n",
        "  int block = instance == 4 && output_count % 8u == 0u && group_size >= 8u && !(group_size & (group_size - 1u));\n"
        "  if (bcir_ai_q8_hoist_off == 2 && block) {\n"
        "    q8_matvec_block512_copy(input, input_count, codes, exponents_le, output_count, group_size, output);\n"
        "    return;\n"
        "  }\n"
        "  if (bcir_ai_q8_hoist_off == 3 && block) {\n"
        "    (void)q8_matvec_block512_ceiling(input, input_count, codes, exponents_le, output_count, group_size, output);\n"
        "    return;\n"
        "  }\n",
    ),
    (
        "  if (block ? q8_matvec_block512_xs(input, input_count, codes, exponents_le, output_count, group_size, output)\n"
        "            : q8_matvec_runs_xs(instance, input, input_count, codes, exponents_le, output_count, group_size,\n"
        "                                output)) {\n",
        "  if (!bcir_ai_q8_hoist_off &&\n"
        "      (block ? q8_matvec_block512_xs(input, input_count, codes, exponents_le, output_count, group_size, output)\n"
        "             : q8_matvec_runs_xs(instance, input, input_count, codes, exponents_le, output_count, group_size,\n"
        "                                 output))) {\n",
    ),
)
for old, new in edits:
    if src.count(old) != 1:
        sys.exit(f"switch.py: anchor occurs {src.count(old)} times: {old[:60]!r}")
    src = src.replace(old, new)
Path(args[1]).write_text(src)
