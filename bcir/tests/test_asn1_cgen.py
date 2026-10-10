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


def test_a_fixed_size_of_64k_or_more_is_refused_for_uper_in_both_directions():
    """X.691 §17.5-§17.7 write a SIZE-fixed OCTET STRING bare only below 64K octets; from 64K
    §17.8 gives it a length determinant, which at that size is fragmented (§11.9.3.8). The
    generated UPER codec wrote and read the 65536 octets bare -- a wire format the oracle
    neither writes nor reads -- and now refuses the type by name, encoder and decoder alike,
    at exactly the size where the oracle's encoding changes form. COER never prefixes a fixed
    size (X.696 §14.1), so it still builds the type."""
    from bcir.asn1.cgen import CModel, CType, _Emitter
    from bcir.asn1.per import encode_per

    def module(size: int) -> str:
        return _HEAD + f"S ::= SEQUENCE {{ a OCTET STRING (SIZE({size})), b BOOLEAN }}\nEND"

    for size in (65536, 70000):
        assert "17.8" in _refusal(module(size), "S", rules=(UPER,))
        generate(_types(module(size)), ["S"], prefix="t_", rules=[COER])
        emitter = _Emitter(CModel(_types(module(size)), ["S"], "t_"), UPER)
        big = CType("octfix", None, cname=f"oct{size}", size=size)
        for direction in (emitter._enc_prim_uper, emitter._dec_prim_uper):
            try:
                direction(big, "v")
            except Asn1Error as error:
                assert "17.8" in str(error), direction.__name__
            else:
                raise AssertionError(f"{direction.__name__} built SIZE({size}) bare")
    generate(_types(module(65535)), ["S"], prefix="t_", rules=[UPER, COER])
    # The oracle's form changes at the same size: bare below 64K, four 16K fragments (`c4`) at it.
    below = encode_per(_types(module(65535))["S"], {"a": bytes(65535), "b": True})
    at = encode_per(_types(module(65536))["S"], {"a": bytes(65536), "b": True})
    assert len(below) == 65536 and at[:1] == b"\xc4" and len(at) == 65539


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


# --- the generator's contract on small schemas, compiled -------------------------------------------
#
# Each witness below builds one small schema with a short driver and holds the generated codec
# to the oracle or to its own encoder. Every one failed on the generator before its fix.


def _schema_codec(text: str, root: str, rules, values=()):
    """`root` of the module `text`, generated for `rules` and compiled with a driver: a function
    running driver lines, or None without a C compiler. Lines are `enc <rule> <i>` -- the i-th
    of `values` -> `OK <hex>` | `ERR <status>` -- and `dec <rule> <cap> <hex>` -- a decode into
    a `cap`-octet arena -> `OK <arena octets> <hex of the re-encoding>` | `ERR <status>`. The
    arena's memory is 64-aligned; `decat <rule> <off> <cap> <hex>` starts it `off` octets in."""
    import subprocess

    from bcir.asn1.cgen import CModel, ValueWriter, _rule_name

    if _CC is None:
        return None
    types = _types(text)
    out = generate(types, [root], prefix="t_", rules=list(rules))
    model = CModel(types, [root], "t_")
    cname = model.ctype_of(types[root], root).cname
    writer = ValueWriter(model)
    decls = ""
    for i, value in enumerate(values):
        decls = writer.define(f"t_value{i}", root, value)
    names = [_rule_name(r) for r in rules]
    refs = ", ".join(f"&t_value{i}" for i in range(len(values))) or "0"
    enc = "".join(
        f'  if (!strcmp(rule, "{n}")) return t_{n}_encode_{cname}(v, out, cap, len);\n'
        for n in names
    )
    dec = "".join(
        f'  if (!strcmp(rule, "{n}")) return t_{n}_decode_{cname}(in, len, v, a);\n' for n in names
    )
    driver = f"""#include <stdio.h>
#include <string.h>
#include "t_codec.h"
{decls}
static const t_{cname} *const VALUES[] = {{{refs}}};
#define N_VALUES {len(values)}
static int nibble(char c) {{
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}}
static int enc(const char *rule, const t_{cname} *v, uint8_t *out, size_t cap, size_t *len) {{
{enc}  return -1;
}}
static int dec(const char *rule, const uint8_t *in, size_t len, t_{cname} *v, t_arena *a) {{
{dec}  return -1;
}}
static _Alignas(64) uint8_t arena_mem[1 << 20];
int main(void) {{
  static char line[1 << 16], arg[1 << 15];
  static uint8_t in[1 << 15], out[1 << 15];
  while (fgets(line, sizeof line, stdin)) {{
    char op[8], rule[32];
    size_t n = 0, len = 0, k;
    long i = 0;
    int st;
    if (sscanf(line, "%7s %31s", op, rule) != 2) {{ puts("ERR usage"); continue; }}
    if (!strcmp(op, "enc")) {{
      if (sscanf(line, "%*s %*s %ld", &i) != 1 || i < 0 || i >= N_VALUES) {{ puts("ERR usage"); continue; }}
      st = enc(rule, VALUES[i], out, sizeof out, &len);
    }} else {{
      t_{cname} v;
      t_arena a;
      long cap = 0, off = 0;
      int got;
      arg[0] = 0;
      if (!strcmp(op, "decat"))
        got = sscanf(line, "%*s %*s %ld %ld %32767s", &off, &cap, arg) - 1;
      else
        got = sscanf(line, "%*s %*s %ld %32767s", &cap, arg);
      if (got != 2 || off < 0 || off > 63 || cap < 0 || cap > (long)sizeof arena_mem - off) {{
        puts("ERR usage");
        continue;
      }}
      if (!strcmp(arg, "-")) arg[0] = 0;
      for (k = 0; arg[k] && arg[k + 1]; k += 2) {{
        int hi = nibble(arg[k]), lo = nibble(arg[k + 1]);
        if (hi < 0 || lo < 0) break;
        in[n++] = (uint8_t)(hi << 4 | lo);
      }}
      if (arg[k]) {{ puts("ERR hex"); continue; }}
      a.base = arena_mem + off; a.cap = (size_t)cap; a.used = 0;
      memset(&v, 0, sizeof v);
      st = dec(rule, in, n, &v, &a);
      if (st == 0) {{
        size_t used = a.used;
        st = enc(rule, &v, out, sizeof out, &len);
        if (st != 0) {{ printf("REENCODE %d\\n", st); continue; }}
        printf("OK %zu ", used);
        for (k = 0; k < len; k++) printf("%02x", out[k]);
        puts("");
        continue;
      }}
    }}
    if (st != 0) {{ printf("ERR %d\\n", st); continue; }}
    printf("OK ");
    for (k = 0; k < len; k++) printf("%02x", out[k]);
    puts("");
  }}
  return 0;
}}
"""
    tmp = tempfile.mkdtemp()
    for name, body in (
        (out.header_name, out.header),
        ("t_codec.c", out.source),
        ("main.c", driver),
    ):
        with open(os.path.join(tmp, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
    binary = os.path.join(tmp, "codec")
    proc = subprocess.run(
        [_CC, "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-I", tmp,
         os.path.join(tmp, "main.c"), os.path.join(tmp, "t_codec.c"), "-o", binary],
        capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-3000:]

    def run(lines: list[str]) -> list[str]:
        done = subprocess.run(
            [binary], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600
        )
        assert done.returncode == 0, done.stderr[-2000:]
        return done.stdout.splitlines()

    return run


_HEAD = "T DEFINITIONS AUTOMATIC TAGS ::= BEGIN\n"
_ARENA = 1 << 20


def test_a_decoder_refuses_every_integer_its_encoder_refuses():
    """The value constraint of a variable-size INTEGER -- COER's length-prefixed forms, UPER's
    semi-constrained and unconstrained ones (`(MIN..10)` is unconstrained in PER) -- was checked
    by the encoders alone, so a decoder handed back values its own encoder then refused to
    write: decode-then-re-encode, the fuzz target's invariant, failed on them."""
    from bcir.asn1.oer import encode_oer
    from bcir.asn1.per import encode_per

    text = _HEAD + "S ::= SEQUENCE { a INTEGER (5..MAX), b INTEGER (MIN..10) }\nEND"
    loose = _types(_HEAD + "S ::= SEQUENCE { a INTEGER (0..MAX), b INTEGER }\nEND")["S"]
    semi = _types(_HEAD + "S ::= SEQUENCE { a INTEGER (5..MAX), b INTEGER }\nEND")["S"]
    good, low, high = {"a": 5, "b": 10}, {"a": 3, "b": 0}, {"a": 5, "b": 20}
    run = _schema_codec(text, "S", [COER, UPER], [good, low, high])
    if run is None:
        return
    kind = _types(text)["S"]
    got = run([f"enc {r} {i}" for r in ("coer", "uper") for i in range(3)])
    assert got == [
        "OK " + encode_oer(kind, good).hex(), "ERR 4", "ERR 4",
        "OK " + encode_per(kind, good).hex(), "ERR 4", "ERR 4",
    ]  # fmt: skip
    got = run(
        [f"dec coer {_ARENA} {encode_oer(loose, v).hex()}" for v in (good, low, high)]
        + [f"dec uper {_ARENA} {encode_per(semi, v).hex()}" for v in (good, high)]
    )
    assert [line.split()[0] for line in got] == ["OK", "ERR", "ERR", "OK", "ERR"], got
    assert got[1:3] == ["ERR 3", "ERR 3"] and got[4] == "ERR 3", got


def test_a_preamble_wider_than_one_bit_read_is_written_in_pieces():
    """64 OPTIONAL components make a 64-bit preamble, wider than the 57 bits one P_put or P_get
    carries; after five bits of another field it was one shift by a negative count. The
    encoding is held to the oracle's and decodes back to itself."""
    from bcir.asn1.per import encode_per

    fields = ", ".join(f"c{i} BOOLEAN OPTIONAL" for i in range(64))
    text = (
        _HEAD + f"R ::= SEQUENCE {{ x INTEGER (0..31), s S }}\nS ::= SEQUENCE {{ {fields} }}\nEND"
    )
    value = {"x": 17, "s": {f"c{i}": i % 2 == 0 for i in range(64) if i % 3 != 1}}
    run = _schema_codec(text, "R", [UPER], [value])
    if run is None:
        return
    want = encode_per(_types(text)["R"], value).hex()
    (enc,) = run(["enc uper 0"])
    assert enc == "OK " + want, (enc, want)
    (dec,) = run([f"dec uper {_ARENA} {want}"])
    assert dec.split()[0] == "OK" and dec.split()[2] == want, dec


def test_an_extensible_integer_with_an_open_root_bound_compiles_and_round_trips():
    """`(0..MAX, ...)` has no upper root bound and `(MIN..5, ...)` no lower one; the root test
    printed the absent bound as `None`, so the generated codec did not compile."""
    from bcir.asn1.per import encode_per

    text = _HEAD + "S ::= SEQUENCE { a INTEGER (0..MAX, ...), b INTEGER (MIN..5, ...) }\nEND"
    values = [{"a": 7, "b": -3}, {"a": -1, "b": 9}]  # inside both roots, then outside both
    run = _schema_codec(text, "S", [UPER], values)
    if run is None:
        return
    kind = _types(text)["S"]
    want = [encode_per(kind, v).hex() for v in values]
    assert run([f"enc uper {i}" for i in range(2)]) == ["OK " + w for w in want]
    for w, line in zip(want, run([f"dec uper {_ARENA} {w}" for w in want])):
        assert line.split()[0] == "OK" and line.split()[2] == w, line


def test_an_extensible_integers_additions_move_no_generated_octet():
    """The relay posture on the generated codecs (docs/BCIR_LANGREF.md §17.3): `INTEGER (0..7,
    ..., 8..255)` -- whose additional set the model now carries -- is still an unbounded
    `int64_t` with no `_int_check`, and under UPER and COER the generated codecs write and read a
    value in the root (3), in the additions (9, 255) and beyond both (300, -1) as the oracle does,
    and as they do for `(0..7, ...)`, whose constraint does not write the additions."""
    from bcir.asn1.oer import encode_oer
    from bcir.asn1.per import encode_per

    values = [{"v": v} for v in (3, 9, 255, 300, -1)]
    written = {}
    for constraint in ("(0..7, ..., 8..255)", "(0..7, ...)"):
        text = _HEAD + f"S ::= SEQUENCE {{ v INTEGER {constraint} }}\nEND"
        run = _schema_codec(text, "S", [UPER, COER], values)
        if run is None:
            return
        kind = _types(text)["S"]
        for rule, oracle in ((UPER, encode_per), (COER, encode_oer)):
            want = [oracle(kind, v).hex() for v in values]
            got = run([f"enc {rule} {i}" for i in range(len(values))])
            assert got == ["OK " + w for w in want], (constraint, rule, got)
            for w, line in zip(want, run([f"dec {rule} {_ARENA} {w}" for w in want])):
                assert line.split()[0] == "OK" and line.split()[2] == w, (constraint, line)
            written.setdefault(rule, []).append(want)
    assert all(a == b for a, b in written.values()), written


def test_a_count_of_empty_elements_is_bounded_by_its_size_not_the_input():
    """A one-enumerator ENUMERATED and a single-value INTEGER encode in no bits under UPER, so
    ten of them take only the count's four bits. The decoder bounded a count by the input left
    as if each element took a bit, and refused these valid encodings as truncated; an empty
    element's count is bounded by its SIZE instead, and without one the type is refused."""
    from bcir.asn1.per import encode_per

    text = _HEAD + (
        "S ::= SEQUENCE { l SEQUENCE (SIZE(0..10)) OF ENUMERATED { y }, "
        "m SEQUENCE (SIZE(0..10)) OF INTEGER (3..3) }\nEND"
    )
    value = {"l": [0] * 10, "m": [3] * 9}
    run = _schema_codec(text, "S", [UPER], [value])
    if run is None:
        return
    want = encode_per(_types(text)["S"], value).hex()
    assert run(["enc uper 0"]) == ["OK " + want]
    (dec,) = run([f"dec uper {_ARENA} {want}"])
    assert dec.split()[0] == "OK" and dec.split()[2] == want, dec
    unbounded = _HEAD + "S ::= SEQUENCE { l SEQUENCE OF ENUMERATED { y } }\nEND"
    assert "zero bits" in _refusal(unbounded, "S", rules=(UPER,))


def test_the_byte_rule_holds_a_strings_size_as_coer_does():
    """A variable OCTET STRING's SIZE bounds were checked by the COER and UPER codecs and not by
    the byte rule's, so one C value was valid under one rule and invalid under another."""
    byte = ByteRule(name="b", type_codes={})
    text = _HEAD + "S ::= SEQUENCE { a OCTET STRING (SIZE(1..32)) }\nEND"
    values = [{"a": b""}, {"a": b"x" * 40}, {"a": b"abcde"}]
    run = _schema_codec(text, "S", [COER, byte], values)
    if run is None:
        return
    got = run([f"enc {r} {i}" for r in ("coer", "b") for i in range(3)])
    assert [g.split()[0] for g in got] == ["ERR", "ERR", "OK"] * 2, got
    assert got[:2] == got[3:5] == ["ERR 4", "ERR 4"], got
    long_ = "28" + "78" * 40  # IntX 40, then forty octets
    assert run([f"dec b {_ARENA} {long_}", f"dec b {_ARENA} 00"]) == ["ERR 3", "ERR 3"]


def test_a_vector_grows_in_place_in_the_arena():
    """V1.1.1 gives a vector's length in octets, so the decoder grows its array as elements
    arrive. It moved the array at every doubling and abandoned each copy: thirty-three
    elements took 124 slots. Grown in place while it is the arena's last allocation it takes
    64, the doubling's own bound -- and a vector of one-size elements is counted by division
    and takes exactly what COER's counted vector takes."""
    byte = ByteRule(name="b", type_codes={})
    items = [{"a": i, "b": 1000 * i} for i in range(33)]
    for element, slots in (("b INTEGER (0..MAX)", 64), ("b INTEGER (0..65535)", 33)):
        text = _HEAD + (
            f"L ::= SEQUENCE {{ items SEQUENCE OF E }}\n"
            f"E ::= SEQUENCE {{ a INTEGER (0..255), {element} }}\nEND"
        )
        run = _schema_codec(text, "L", [COER, byte], [{"items": items}])
        if run is None:
            return
        enc = [line.split()[1] for line in run(["enc coer 0", "enc b 0"])]
        dec = run([f"dec coer {_ARENA} {enc[0]}", f"dec b {_ARENA} {enc[1]}"])
        used = [int(line.split()[1]) for line in dec]
        slot = used[0] // 33  # COER counts first, so its array is exactly 33 elements
        assert used == [33 * slot, slots * slot], (element, used, slot)


def test_the_arena_aligns_an_address_not_an_offset():
    """The arena is caller memory, which may start at any octet. Its allocator rounded the fill
    OFFSET up to an element's alignment, so an arena starting one octet past an 8-octet boundary
    handed out arrays one octet past one too: undefined behaviour, and a bus error on a core
    that traps misaligned loads. Each array now starts at the next aligned ADDRESS, so starting
    the arena `off` octets into aligned memory costs exactly the padding back to a boundary."""
    byte = ByteRule(name="b", type_codes={})
    text = _HEAD + "L ::= SEQUENCE { items SEQUENCE OF INTEGER (0..MAX) }\nEND"
    run = _schema_codec(text, "L", [COER, UPER, byte], [{"items": [1, 2, 3, 4, 5]}])
    if run is None:
        return
    for rule in ("coer", "uper", "b"):
        (enc,) = run([f"enc {rule} 0"])
        want = enc.split()[1]
        got = run([f"decat {rule} {off} {_ARENA - 64} {want}" for off in range(9)])
        assert all(line.split()[0] == "OK" and line.split()[2] == want for line in got), got
        used = [int(line.split()[1]) for line in got]
        # The elements are 8-aligned uint64_t: an arena `off` octets past a boundary pads
        # (8 - off) % 8 octets before its first array, and the rest is laid out as at 0.
        assert used == [used[0] + (8 - off) % 8 for off in range(9)], (rule, used)


def test_the_harness_refuses_what_is_not_hex():
    """The harness read hex with `sscanf("%2x")`, which takes one digit, a sign or a `0x`
    prefix as an octet -- so `0g` decoded as `00` and `+f` as `0f`, and an odd digit count
    stepped past the end of the string. Each is refused as what it is, before any decoder."""
    from bcir.asn1 import its_native as nat

    with tempfile.TemporaryDirectory() as tmp:
        binary = _harness(tmp)
        if binary is None:
            return
        out = nat.run(binary, [f"dec coer {bad}" for bad in ("abc", "0g", "+f", "0x12", "f")])
    assert out == ["ERR hex"] * 5, out
