"""`bcir.asn1.cgen`: ASN.1 compiled to straight-line C, held to the Python rails.

Generation-time behaviour (what is refused, how names are spelled) is pure Python and runs in
every tier. Everything else compiles the generated codecs for the ITS envelope study
(`bcir.asn1.its_native`) with the host compiler and self-skips without one, as every C-twin
test here does:

* PARITY -- every generated encoder writes the oracle's octets for every envelope, and every
  generated decoder reads them back to the value the encoder started from;
* the DIFFERENTIAL -- truncated and mutated encodings through every generated decoder: where
  the decoder accepts, the oracle accepts and both re-encode to exactly the input (the codecs
  are canonical-only); where it refuses, the oracle refuses too, or the input is a
  non-canonical spelling only the oracle's BASIC-tolerant length reader admits;
* TOTALITY -- the differential under AddressSanitizer and UndefinedBehaviorSanitizer when the
  thorough tier asks for it, and a cyclic value refused by every encoder's depth bound;
* MEMORY -- no allocator symbol in any generated object: the heap claim is checked, not stated.
"""

from __future__ import annotations

import os
import random
import tempfile

from bcir.asn1.cgen import COER, UPER, ByteRule, generate
from bcir.asn1.tags import Asn1Error
from bcir.frontends.asn1 import compile_module

_SEED = 20261009
#: The host C compiler the compiled tests build with, or None: `tools/testing/faults/its.json`
#: requires it, since without one those tests return before examining anything.
_CC = __import__("bcir.toolchain", fromlist=["host_c_compiler"]).host_c_compiler()


def _types(text: str) -> dict:
    return compile_module(text, "t.asn1").module.types


def _refusal(source: str, root: str, rules=(COER,)) -> str:
    try:
        generate(_types(source), [root], prefix="t_", rules=list(rules))
    except Asn1Error as error:
        return str(error)
    raise AssertionError(f"{root} was compiled")


# --- generation time ----------------------------------------------------------------------------


def test_a_construct_the_compiler_does_not_build_is_refused_by_name():
    """A codec that silently skipped a construct would disagree with the oracle on exactly the
    values that use it, and the parity test would report an unexplained byte."""
    head = "T DEFINITIONS AUTOMATIC TAGS ::= BEGIN\n"
    cases = {
        "DEFAULT": "S ::= SEQUENCE { a INTEGER (0..7) DEFAULT 3 }",
        "extension addition": "S ::= SEQUENCE { a INTEGER (0..7), ..., b INTEGER (0..7) }",
        "extension alternative": "S ::= CHOICE { a INTEGER (0..7), ..., b NULL }",
        "SET": "S ::= SET { a INTEGER (0..7) }",
        "UTF8String": "S ::= SEQUENCE { a UTF8String }",
        "SEQUENCE OF NULL": "S ::= SEQUENCE OF NULL",
        "0..127": "S ::= SEQUENCE { e ENUMERATED { a(0), b(200) } }",
    }
    for name, body in cases.items():
        assert name in _refusal(head + body + "\nEND", "S"), name


def test_a_byte_rule_choice_without_type_codes_is_refused():
    source = "T DEFINITIONS AUTOMATIC TAGS ::= BEGIN\nS ::= CHOICE { a NULL, b NULL }\nEND"
    rule = ByteRule(name="b", type_codes={})
    assert "no type codes" in _refusal(source, "S", rules=(rule,))


def test_keywords_and_hyphens_become_c_identifiers():
    """`Payload ::= CHOICE { signed ... }` is TS 103 097's own spelling, and `signed` is C11's."""
    source = (
        "T DEFINITIONS AUTOMATIC TAGS ::= BEGIN\n"
        "S ::= CHOICE { signed OCTET STRING, time-end INTEGER (0..7) }\nEND"
    )
    out = generate(_types(source), ["S"], prefix="t_", rules=[COER, UPER])
    assert "t_octets signed_;" in out.header and "time_end;" in out.header
    assert "#define T_S_signed_ 0u" in out.header


def test_the_generated_files_name_only_the_helpers_they_use():
    """A file with no UPER codec carries no bit reader -- which is also what keeps it clean
    under -Wunused-function without suppressing the warning."""
    source = "T DEFINITIONS AUTOMATIC TAGS ::= BEGIN\nS ::= SEQUENCE { a INTEGER (0..255) }\nEND"
    coer_only = generate(_types(source), ["S"], prefix="t_", rules=[COER]).source
    assert "t_get_octets" not in coer_only and "t_pw_finish" not in coer_only
    both = generate(_types(source), ["S"], prefix="t_", rules=[COER, UPER]).source
    assert "t_pw_finish" in both


def test_nothing_a_generated_file_defines_escapes_its_prefix():
    """Two generated files link into one program, and a caller includes their headers beside
    its own code, so every name a file defines carries the file's prefix -- macros too. The
    byte-swap words and the native-word switch once leaked as `P_BSWAP16` and `P_NATIVE_WORDS`."""
    import re

    head = "T DEFINITIONS AUTOMATIC TAGS ::= BEGIN\nS ::= SEQUENCE { a INTEGER (0..65535), "
    rich = generate(
        _types(head + "b OCTET STRING, c INTEGER, d BOOLEAN }\nEND"),
        ["S"],
        prefix="t_",
        rules=[COER, UPER],
    )
    byte = generate(
        _types(head + "b OCTET STRING }\nEND"),
        ["S"],
        prefix="t_",
        rules=[ByteRule(name="b", type_codes={})],
    )
    for text in (rich.header, rich.source, byte.source):
        assert not re.search(r"\bP_[A-Za-z]", text), sorted(set(re.findall(r"\bP_\w+", text)))
        defined = (
            re.findall(r"^#\s*define\s+(\w+)", text, re.M)
            + re.findall(r"^typedef\b.*?(\w+);\s*$", text, re.M)
            + re.findall(r"^struct (\w+) \{", text, re.M)
            + re.findall(r"^(?!typedef|struct)[A-Za-z][^(;=]*?\b(\w+)\(", text, re.M)
        )
        assert len(defined) >= 10, defined
        assert [n for n in defined if not n.startswith(("t_", "T_"))] == []


def test_a_file_compiles_alone_however_few_helpers_it_needs():
    """The helper selection keeps a preprocessor conditional whole or not at all, and keeps
    the cursor types whenever the code names them. Before it did, a codec of fixed octets
    alone kept the byte-order `#if` without its `#endif` and lost the cursor typedef -- two
    reasons it did not compile. Every minimal file is checked for balance in every tier and
    compiled with warnings as errors where a C compiler exists (an unused helper would warn)."""
    import re
    import subprocess

    from bcir.toolchain import host_c_compiler

    head = "T DEFINITIONS AUTOMATIC TAGS ::= BEGIN\nS ::= SEQUENCE { "
    byte = ByteRule(name="b", type_codes={})  # V1.1.1 has no BOOLEAN and no unbounded INTEGER
    cases = [
        (body, rules)
        for body in ("a OCTET STRING (SIZE(4))", "a INTEGER (0..255)", "a OCTET STRING")
        for rules in ([COER], [UPER], [byte])
    ]
    cases += [(body, rules) for body in ("a BOOLEAN", "a INTEGER") for rules in ([COER], [UPER])]
    cc = host_c_compiler()
    with tempfile.TemporaryDirectory() as tmp:
        for body, rules in cases:
            out = generate(_types(head + body + " }\nEND"), ["S"], prefix="t_", rules=rules)
            opened = len(re.findall(r"^\s*#\s*if", out.source, re.M))
            closed = len(re.findall(r"^\s*#\s*endif", out.source, re.M))
            assert opened == closed, (body, rules, opened, closed)
            if cc is None:
                continue
            with open(os.path.join(tmp, out.header_name), "w") as f:
                f.write(out.header)
            path = os.path.join(tmp, "t_codec.c")
            with open(path, "w") as f:
                f.write(out.source)
            proc = subprocess.run(
                [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", "-I", tmp, path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert proc.returncode == 0, (body, rules, proc.stderr[-1500:])


# --- compiled: parity, differential, totality, memory ---------------------------------------------


def _harness(tmp: str, flags=("-O2",), cc=None):
    from bcir.asn1 import its_native as nat

    if not nat.native_available():
        return None
    return nat.build(tmp, flags=flags, cc=cc)


def test_every_generated_codec_writes_and_reads_the_oracles_octets():
    from bcir.asn1 import its_native as nat

    with tempfile.TemporaryDirectory() as tmp:
        binary = _harness(tmp)
        if binary is None:
            return
        want = nat.oracle_octets()
        keys = [(c, i) for c, _m, _r in nat.CODECS for i in range(4)]
        enc = nat.run(binary, [f"enc {c} {i}" for c, i in keys])
        dec = nat.run(binary, [f"dec {c} {want[(c, i)].hex()}" for c, i in keys])
    for (codec, i), got in zip(keys, enc):
        assert got == "OK " + want[(codec, i)].hex(), (codec, i, got[:60])
    for (codec, i), got in zip(keys, dec):
        parts = got.split()
        # the decoded value equals envelope i, and re-encodes to the very octets it came from
        assert parts[:2] == ["OK", str(i)] and parts[3] == want[(codec, i)].hex(), (codec, i)


def _oracle_reencode(codec: str, data: bytes):
    from bcir.asn1 import its_security as its
    from bcir.asn1.oer import OerRules, decode_oer, encode_oer
    from bcir.asn1.per import decode_per, encode_per

    if codec in ("v111", "coer", "uper"):
        kind = its.transcription_type()
    else:  # each TMSAO codec runs on the instance of the schema shaped for its rule
        kind = its.tmsao_type("SecuredMessage" if codec == "tmsao-coer" else "SecuredMessagePer")
    try:
        if codec == "v111":
            return its.encode_binary(its.decode_binary(data))
        if codec.endswith("coer"):
            return encode_oer(kind, decode_oer(kind, data, rules=OerRules.CANONICAL))
        return encode_per(kind, decode_per(data, kind))
    except Asn1Error as error:
        return error


def _differential(binary: str, per_encoding: int) -> dict:
    from bcir.asn1 import its_native as nat

    want = nat.oracle_octets()
    rng = random.Random(_SEED)
    cases = []
    for codec, _m, _r in nat.CODECS:
        for i in range(4):
            raw = want[(codec, i)]
            cases += [(codec, raw[:cut]) for cut in range(len(raw))]
            for _ in range(per_encoding):
                b = bytearray(raw)
                for _k in range(rng.randint(1, 3)):
                    b[rng.randrange(len(b))] = rng.randrange(256)
                cases.append((codec, bytes(b)))
    out = nat.run(binary, [f"dec {c} {d.hex() or '-'}" for c, d in cases], timeout=3600)
    seen = {"both accept": 0, "both refuse": 0, "non-canonical": 0}
    for (codec, data), line in zip(cases, out):
        oracle = _oracle_reencode(codec, data)
        parts = line.split()
        if parts[0] == "OK":
            again = bytes.fromhex(parts[3]) if len(parts) > 3 else b""
            assert not isinstance(oracle, Exception), (codec, data.hex(), "C accepts", oracle)
            assert again == data == oracle, (codec, data.hex(), "re-encodings differ")
            seen["both accept"] += 1
        else:
            assert parts[0] == "ERR", (codec, line)
            if isinstance(oracle, Exception):
                seen["both refuse"] += 1
            else:
                assert oracle != data, (codec, data.hex(), "C refuses a canonical encoding")
                seen["non-canonical"] += 1
    return seen


def test_the_generated_decoders_agree_with_the_oracle_on_damaged_input():
    with tempfile.TemporaryDirectory() as tmp:
        binary = _harness(tmp)
        if binary is None:
            return
        seen = _differential(binary, per_encoding=40)
    # Anti-vacuity: both verdicts must actually have been exercised, on every codec's inputs.
    assert seen["both accept"] >= 100 and seen["both refuse"] >= 1000, seen


def test_a_cut_encoding_is_refused_as_truncated_by_every_decoder():
    """A proper prefix of a canonical encoding ends inside a value, so the first thing every
    decoder runs out of is input, and the refusal says TRUNC. The status is the point: a bit
    reader that over-reads past its input is still refused by the final length check, but as
    MALFORMED, so the differential's accept/refuse verdict cannot see it. Every cut of every
    envelope, including the empty input, on all five codecs."""
    from bcir.asn1 import its_native as nat

    with tempfile.TemporaryDirectory() as tmp:
        binary = _harness(tmp)
        if binary is None:
            return
        want = nat.oracle_octets()
        cases = [
            (codec, i, cut)
            for codec, _m, _r in nat.CODECS
            for i in range(4)
            for cut in range(len(want[(codec, i)]))
        ]
        out = nat.run(binary, [f"dec {c} {want[(c, i)][:cut].hex() or '-'}" for c, i, cut in cases])
    trunc = [k for k, v in nat.STATUS.items() if v == "TRUNC"]
    assert len(out) == len(cases) > 3000
    for case, line in zip(cases, out):
        assert line.split() == ["ERR", str(trunc[0])], (case, line)


def test_the_differential_holds_under_the_sanitizers():
    """ASan and UBSan over the same campaign, when the thorough tier runs the C twins."""
    if os.environ.get("BCIR_THOROUGH") != "1":
        return
    import shutil

    cc = shutil.which("clang")
    if cc is None:
        return
    flags = (
        "-O1",
        "-g",
        "-fsanitize=address,undefined",
        "-fno-sanitize-recover=all",
        "-fno-omit-frame-pointer",
    )
    with tempfile.TemporaryDirectory() as tmp:
        binary = _harness(tmp, flags=flags, cc=cc)
        if binary is None:
            return
        seen = _differential(binary, per_encoding=150)
    assert seen["both accept"] >= 300, seen


def test_the_bench_refuses_to_time_a_decoder_that_refuses():
    """A refused decode returns early, so timing it would report the refusal as the codec's
    speed. The bench decodes and re-encodes every case once before any clock starts and stops
    on the first failure. Witness: the TMSAO UPER entry point made to refuse what it reads."""
    import subprocess

    from bcir.asn1 import its_native as nat
    from bcir.toolchain import host_c_compiler

    cc = host_c_compiler()
    if cc is None or not nat.native_available():
        return
    good = "  return ITST_OK;\n}\n"
    with tempfile.TemporaryDirectory() as tmp:
        files = nat.sources()
        text = files["itst_codec.c"]
        end = text.index(good, text.index("int itst_uper_decode_SecuredMessagePer("))
        files["itst_codec.c"] = (
            text[:end] + "  return ITST_E_MALFORMED;\n}\n" + text[end + len(good) :]
        )
        for name, body in files.items():
            with open(os.path.join(tmp, name), "w") as f:
                f.write(body)
        binary = os.path.join(tmp, "its_harness")
        subprocess.run(
            [
                cc,
                "-std=c11",
                "-O0",
                "-I",
                tmp,
                *(os.path.join(tmp, n) for n in ("its_harness.c", "itsv_codec.c", "itst_codec.c")),
                "-o",
                binary,
                "-pthread",
            ],
            check=True,
            capture_output=True,
            timeout=600,
        )
        try:
            nat.bench(binary, rounds=4, iters=1)
        except Asn1Error as error:
            assert "bench-decode tmsao-uper 0" in str(error), error
        else:
            raise AssertionError("the bench timed a decoder that refuses its own encoding")


def test_a_cyclic_value_is_refused_by_every_encoder():
    """A caller can build a certificate whose signer points back at itself; an encoder that
    followed it would recurse until the stack ran out. Every one refuses with the bound."""
    from bcir.asn1 import its_native as nat

    with tempfile.TemporaryDirectory() as tmp:
        binary = _harness(tmp)
        if binary is None:
            return
        (line,) = nat.run(binary, ["cycle"])
    assert line.split() == ["OK"] + ["5"] * len(nat.CODECS), line  # 5 = E_LIMIT


def test_no_generated_codec_references_an_allocator():
    from bcir.asn1 import its_native as nat

    with tempfile.TemporaryDirectory() as tmp:
        if _harness(tmp) is None:
            return
        found = nat.allocator_symbols(tmp)
    if not found:
        return  # no nm on this host
    assert found == {"itsv_codec.c": set(), "itst_codec.c": set()}, found


def test_the_codecs_memory_is_measured_and_small():
    """Arena octets are exact; stack is a measured high-water bound. The study's claim is the
    order of magnitude against the paper's 12-20 KB of stack and 0.2-2.2 KB of heap, so the
    gate is deliberately loose: every codec under 4 KB of stack, the TMSAO COER codec's
    arena empty for the CAM envelope without a certificate."""
    from bcir.asn1 import its_native as nat

    with tempfile.TemporaryDirectory() as tmp:
        binary = _harness(tmp)
        if binary is None:
            return
        rows = nat.memory(binary)
    assert len(rows) == 4 * len(nat.CODECS)
    for row in rows:
        assert 0 < row.encode_stack < 4096 and 0 < row.decode_stack < 4096, row
        assert row.heap_octets == 0
    by = {(r.codec, r.envelope): r for r in rows}
    assert by[("tmsao-coer", 0)].arena_octets == 0


def test_libfuzzer_finds_no_crash_and_no_second_spelling():
    """Every generated decoder is a trust-boundary decoder, so it is fuzzed like one: the
    thorough tier runs a bounded libFuzzer campaign over all five under ASan and UBSan, and
    the target aborts when an accepted input does not re-encode to itself -- a second
    spelling of one value is reported exactly like a crash."""
    if os.environ.get("BCIR_THOROUGH") != "1":
        return
    import shutil
    import subprocess

    from bcir.asn1 import its_native as nat

    cc = shutil.which("clang")
    if cc is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in nat.fuzz_sources().items():
            with open(os.path.join(tmp, name), "w") as f:
                f.write(text)
        corpus = os.path.join(tmp, "corpus")
        os.mkdir(corpus)
        for i, seed in enumerate(nat.seeds()):
            with open(os.path.join(corpus, f"seed{i:02d}"), "wb") as f:
                f.write(seed)
        binary = os.path.join(tmp, "fuzz_its")
        build = subprocess.run(
            [
                cc,
                "-std=c11",
                "-g",
                "-O1",
                "-fsanitize=fuzzer,address,undefined",
                "-fno-sanitize-recover=all",
                "-I",
                tmp,
                os.path.join(tmp, "fuzz_its_codecs.c"),
                os.path.join(tmp, "itsv_codec.c"),
                os.path.join(tmp, "itst_codec.c"),
                "-o",
                binary,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if build.returncode != 0 and "fuzzer" in build.stderr:
            return  # this clang has no libFuzzer runtime
        assert build.returncode == 0, build.stderr[-2000:]
        run = subprocess.run(
            [binary, "-runs=200000", "-max_len=600", corpus],
            capture_output=True,
            text=True,
            timeout=900,
        )
    assert run.returncode == 0, run.stderr[-3000:]
    assert "Done 200000 runs" in run.stderr, run.stderr[-500:]
