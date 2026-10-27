"""DECODE-PASSES' vacuity control: the fault table the old-witness sweep ran (faults-old-witnesses.json).

usage: vacuity_table.py DECODE_PASSES_JSON OUT_JSON

The same fifteen faults as tools/testing/faults/decode-passes.json, each expecting any finding (''), under the
witnesses there were before DECODE-PASSES: the sixteen tests of the tree before that build or run the units it changes
(bcir_q8_model.c, bcir_llama.c) -- the GEM interpreter's tests, the BCIRQ8 export, loader and decoder tests, the C
runtime's CRC test and the C front end's allocation-failure sweep, which runs the memory-discipline harness. The sweep
ran on a copy of the tree after whose three witness files (test_memory_discipline.c, fuzz_q8_model.c,
test_model_weights_io.py) were the tree before's."""

import json
import sys

OLD_WITNESSES = (
    "bcir.tests.test_decoder_gem:test_the_c_interpreter_is_the_monolithic_runner_bit_for_bit",
    "bcir.tests.test_decoder_gem:test_the_c_interpreter_counts_the_traffic_the_program_plans",
    "bcir.tests.test_decoder_gem:test_the_c_interpreter_refuses_what_the_oracle_refuses",
    "bcir.tests.test_decoder_gem:test_a_c_run_reads_what_it_wrote_or_zeros",
    "bcir.tests.test_decoder_gem:test_a_count_the_body_does_not_carry_is_refused_before_anything_is_allocated",
    "bcir.tests.test_decoder_gem:test_the_c_interpreter_lent_a_runner_is_itself_bit_for_bit",
    "bcir.tests.test_decoder_gem:test_final_is_a_resource_of_its_own_on_the_c_rail",
    "bcir.tests.test_model_weights_io:test_bcirq8_export_is_deterministic_and_exactly_the_q8_roundtrip",
    "bcir.tests.test_model_weights_io:test_bcirq8_rejects_truncation_corruption_and_invalid_spans",
    "bcir.tests.test_model_weights_io:test_bcirq8_limits_and_metadata_are_strict_before_allocation_or_write",
    "bcir.tests.test_model_weights_io:test_standalone_c_q8_decoder_matches_python_for_tied_and_untied_heads",
    "bcir.tests.test_model_weights_io:test_the_decoders_rotation_is_bcir_ropes_rows_bit_for_bit",
    "bcir.tests.test_model_weights_io:test_the_decoders_head_reads_the_pack_and_is_the_rows_own_bit_for_bit",
    "bcir.tests.test_model_weights_io:test_the_decoder_lent_a_runner_is_the_decoder_bit_for_bit",
    "bcir.tests.test_c_runtime:test_the_runtime_crc_is_zlibs",
    "bcir.tests.test_c_cfront:test_the_emit_grows_whole_under_a_failing_allocator",
)

if len(sys.argv) != 3:
    sys.exit(__doc__.rsplit("usage: ", 1)[1].split("\n", 1)[0])
src = json.load(open(sys.argv[1], encoding="utf-8"))
table = {
    "command": [
        "@python",
        "tools/testing/check_tests.py",
        "--require",
        "bcir.tests.test_native_ai:_CC",
        *OLD_WITNESSES,
    ],
    "faults": [dict(f, expects="") for f in src["faults"]],
}
with open(sys.argv[2], "w", encoding="utf-8") as out:
    out.write(json.dumps(table, indent=1) + "\n")
