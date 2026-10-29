"""Q8-PAIR's switched kernel unit: the tree after's runtime/c/bcir_ai_kernels.c with Q8-PAIR's choice behind a run-time
switch, the global bcir_ai_q8_pair_off -- 0 the unit as committed, any other value every call Q8-PAIR takes running
the tree before's kernel instead (the AVX-512 block, the tree before's q8_matvec_block512 and q8_matvec_block512_xs
copied here byte for byte from its unit; at AVX2 the runs, which the tree after still holds) -- and those calls counted
in bcir_ai_q8_pair_calls (0 the pair, 1 the tree before's kernel). Every other byte is the unit's: one binary holds both
kernels at fixed addresses, and pair_kernels.c and pair_decode.c time one against the other with nothing else moved.
usage: pair_switch.py UNIT_AFTER UNIT_BEFORE OUT"""

import sys
from pathlib import Path

if len(sys.argv) != 4:
    sys.exit(__doc__.rsplit("usage: ", 1)[1])
src = Path(sys.argv[1]).read_text()
before = Path(sys.argv[2]).read_text()


def between(text, start, end):
    """the text from `start` up to `end`, each occurring once"""
    for anchor in (start, end):
        if text.count(anchor) != 1:
            sys.exit(f"pair_switch.py: anchor occurs {text.count(anchor)} times: {anchor[:60]!r}")
    i = text.index(start)
    return text[i : text.index(end, i)]


block = between(
    before,
    "/* The matvec 64 columns at a time, their sums in registers over every input row",
    "/* Q8-BOUND's block: eight columns' terms, xs the row's input times their group's scale",
)
block_xs = between(
    before,
    "/* the block with x * s formed once a row and group */",
    "/* Q8-BOUND's ranges, read once a call:",
)
edits = (
    (
        "/* Q8-BOUND's ranges, read once a call:",
        "/* the tree before's AVX-512 block (pair_switch.py), byte for byte */\n"
        + block
        + block_xs
        + "/* Q8-BOUND's ranges, read once a call:",
    ),
    (
        "static void q8_matvec_lanes(int instance,",
        "extern int bcir_ai_q8_pair_off;\n"
        "extern unsigned long bcir_ai_q8_pair_calls[2];\n"
        "int bcir_ai_q8_pair_off; /* Q8-PAIR's switch: nonzero, the pair's calls run the tree before's kernels */\n"
        "unsigned long bcir_ai_q8_pair_calls[2]; /* the pair's calls: 0 run as committed, 1 on the tree before's */\n"
        "static void q8_matvec_lanes(int instance,",
    ),
    (
        "  Q8_PAIR_SAW(pair ? 1 : 2);\n",
        "  int before = pair && bcir_ai_q8_pair_off;\n"
        "  Q8_PAIR_SAW(pair ? 1 : 2);\n"
        "  if (pair) ++bcir_ai_q8_pair_calls[before];\n",
    ),
    (
        "    if (!pair)\n      q8_matvec_runs_xs(",
        "    if (before && instance == 4)\n"
        "      q8_matvec_block512_xs(input, input_count, codes, exponents_le, output_count, group_size, output);\n"
        "    else if (!pair || before)\n      q8_matvec_runs_xs(",
    ),
    (
        "  if (!pair)\n    q8_matvec_runs(",
        "  if (before && instance == 4)\n"
        "    q8_matvec_block512(input, input_count, codes, exponents_le, output_count, group_size, output);\n"
        "  else if (!pair || before)\n    q8_matvec_runs(",
    ),
)
for old, new in edits:
    if src.count(old) != 1:
        sys.exit(f"pair_switch.py: anchor occurs {src.count(old)} times: {old[:60]!r}")
    src = src.replace(old, new)
Path(sys.argv[3]).write_text(src)
