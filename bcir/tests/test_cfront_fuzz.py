"""Phase 4 — segment 4 (security/fuzzing): the C-frontend fuzzing campaigns.

Valid random programs must compile clean (no crash / fallback / mismatch); corrupted programs must
never crash the front end (the totality contract). Both are seeded and reproducible. This campaign
already caught a real bug: a malformed numeric literal (`9au`) raised a bare ValueError that escaped
`diagnose` -- now a clean CLexError (regression-tested below).
"""

from __future__ import annotations

from bcir.frontends.cfront.cfuzz import fuzz_malformed, fuzz_valid


def test_valid_programs_compile_clean():
    s = fuzz_valid(seed=1234, trials=150)
    assert s["crash"] == 0 and s["fallback"] == 0 and s["mismatch"] == 0


def test_valid_programs_are_clang_equivalent():
    # clang-gated: with a C compiler each generated program must be behaviour-equivalent; with none,
    # the equivalence is "skip" and mismatch stays 0 either way. The generator now spans control flow +
    # name scoping (loops that reuse a counter, bare blocks, and a loop/block that shadows a param read
    # after the construct), so this campaign is a standing guard for the emit-compilation / scope-leak
    # class of miscompiles -- a regression there surfaces here as a MISMATCH (or a build failure).
    s = fuzz_valid(seed=99, trials=60, check_clang=True)
    assert s["crash"] == 0 and s["mismatch"] == 0 and s["fallback"] == 0


def test_malformed_input_never_crashes_the_frontend():
    # the totality contract: compile_with_fallback + diagnose always return on garbage input.
    for seed in (0, 1, 2):
        assert fuzz_malformed(seed=seed, trials=600)["crash"] == 0


def test_campaign_is_deterministic():
    assert fuzz_valid(seed=7, trials=80) == fuzz_valid(seed=7, trials=80)
    assert fuzz_malformed(seed=7, trials=200) == fuzz_malformed(seed=7, trials=200)


def test_malformed_integer_literal_is_a_clean_lex_error():
    # the bug the fuzzer found: `9au` tokenizes as INT but is not a valid integer -> a CLexError,
    # not a bare ValueError, so diagnose reports it instead of crashing.
    from bcir.frontends.cfront import diagnose
    from bcir.frontends.cfront.clex import CLexError, parse_int_literal

    try:
        parse_int_literal("9au")
        raise AssertionError("expected CLexError for a malformed integer literal")
    except CLexError:
        pass
    rep = diagnose("unsigned f(unsigned a){ return a + 9au; }\n", filename="t.c")
    assert not rep.ok  # reported, did not crash
