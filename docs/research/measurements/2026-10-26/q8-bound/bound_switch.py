"""Q8-BOUND's switched kernel unit: the tree after's runtime/c/bcir_ai_kernels.c with its once-a-call choice behind a
run-time switch, the global bcir_ai_q8_bound_off -- 0 the unit as committed, any other value every call taking the
terms as before Q8-BOUND (the committed kernels it falls back to) -- and the calls each path took counted in
bcir_ai_q8_bound_calls (0 hoisted, 1 the terms as before), so a timing tool can show every call of an arm took its
arm's path. Every other byte is the unit's: one binary holds both paths at fixed addresses, and bound_kernels.c,
bound_ab.c and bound_clock.c time one against the other with nothing else moved.
usage: bound_switch.py UNIT OUT"""

import sys
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit(__doc__.rsplit("usage: ", 1)[1])
src = Path(sys.argv[1]).read_text()
edits = (
    (
        "/* Q8-BOUND where the call's ranges prove every x * s exact, the terms as before elsewhere */\n"
        "static void q8_matvec_lanes(",
        "/* Q8-BOUND where the call's ranges prove every x * s exact, the terms as before elsewhere */\n"
        "extern int bcir_ai_q8_bound_off;\n"
        "extern unsigned long bcir_ai_q8_bound_calls[2];\n"
        "int bcir_ai_q8_bound_off; /* Q8-BOUND's switch: nonzero, every call takes the terms as before */\n"
        "unsigned long bcir_ai_q8_bound_calls[2]; /* the calls each path took: 0 hoisted, 1 the terms as before */\n"
        "static void q8_matvec_lanes(",
    ),
    (
        "  if (q8_call_exact(input, input_count, exponents_le, 1u + (input_count * output_count - 1u) / group_size)) {\n",
        "  if (!bcir_ai_q8_bound_off &&\n"
        "      q8_call_exact(input, input_count, exponents_le, 1u + (input_count * output_count - 1u) / group_size)) {\n",
    ),
    ("    Q8_BOUND_SAW(1);\n", "    Q8_BOUND_SAW(1);\n    ++bcir_ai_q8_bound_calls[0];\n"),
    ("  Q8_BOUND_SAW(2);\n", "  Q8_BOUND_SAW(2);\n  ++bcir_ai_q8_bound_calls[1];\n"),
)
for old, new in edits:
    if src.count(old) != 1:
        sys.exit(f"bound_switch.py: anchor occurs {src.count(old)} times: {old[:60]!r}")
    src = src.replace(old, new)
Path(sys.argv[2]).write_text(src)
