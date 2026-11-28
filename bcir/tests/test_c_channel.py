"""Python<->C parity for the multi-channel lowering decision (`runtime/c/bcir_channel.c`).

The C router consumes the same `channel.json` manifests and, for each claim, derives the same
required capabilities, the same per-channel eligibility, and the same routed backend as the oracle
(`bcir/channels`: `claim_required_caps` / `channel_suits` / `route_claim`). Toolchain-gated: the
quick tier still checks the oracle routing is computable; the C byte-for-byte parity runs under
c-runtime/thorough.
"""

import os
import shutil
import subprocess
import tempfile

from bcir.channel_plugin import load_manifest
from bcir.channels import channel_suits, claim_required_caps, route_claim
from bcir.model import Claim, Opcode, StrideClass

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_CHAN = os.path.join(_ROOT, "channels")
_CC = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")

# cpu (universal plugin), tpu (a capability plugin), pim (no caps -> the legacy per-kind rule).
_CHANNEL_FILES = [
    "example_cpu.channel.json",
    "example_tpu.channel.json",
    "example_pim.channel.json",
]
# claims spanning the routing space: op-derived caps (reduce/gather/matmul) + every stride class.
_CLAIMS = [
    ("reduce.sum", StrideClass.UNIT),
    ("matmul.acc", StrideClass.TILE),
    ("gather.load", StrideClass.RANDOM),
    ("vector.add", StrideClass.UNIT),
    ("scalar.mov", StrideClass.SCALAR),
    ("c.bin.add", StrideClass.SCALAR),
    ("strided.load", StrideClass.STRIDED),
    ("tile.mac", StrideClass.TILE),
    ("cacheline.load", StrideClass.CACHELINE),
]


def _claim(op: str, sc: StrideClass) -> Claim:
    return Claim(id=1, opcode=Opcode.LOAD, op=op, stride_class=sc)


def _py_line(op: str, sc: StrideClass, mans) -> str:
    cl = _claim(op, sc)
    caps = ",".join(sorted(claim_required_caps(cl)))
    chosen = route_claim(cl, mans)
    route = chosen.name if chosen else "NONE"
    suits = ";".join(f"{m.name}={1 if channel_suits(cl, m) else 0}" for m in mans)
    return f"{op}|{int(sc)}|caps={caps}|route={route}|suits={suits}"


def _build(d: str) -> str:
    exe = os.path.join(d, "tch")
    for std in ("c23", "c11"):
        b = subprocess.run(
            [
                _CC,
                f"-std={std}",
                "-O2",
                "-Wall",
                "-Wextra",
                "-I",
                _C,
                os.path.join(_C, "bcir_channel.c"),
                os.path.join(_C, "test_channel.c"),
                "-o",
                exe,
            ],
            capture_output=True,
            text=True,
        )
        if b.returncode == 0:
            return exe
    raise AssertionError(f"channel router build failed:\n{b.stderr}")


def test_python_c_channel_routing_parity():
    mans = [load_manifest(os.path.join(_CHAN, f)) for f in _CHANNEL_FILES]
    if not _CC:  # quick tier: the oracle routing is deterministic.
        for op, sc in _CLAIMS:
            assert "route=" in _py_line(op, sc, mans)
        return
    paths = [os.path.join(_CHAN, f) for f in _CHANNEL_FILES]
    stdin = "".join(f"{op} {int(sc)}\n" for op, sc in _CLAIMS)
    with tempfile.TemporaryDirectory() as d:
        exe = _build(d)
        out = subprocess.run([exe, *paths], input=stdin, capture_output=True, text=True).stdout
    c_lines = [ln for ln in out.strip().splitlines() if ln]
    assert len(c_lines) == len(_CLAIMS), out
    for (op, sc), c_line in zip(_CLAIMS, c_lines):
        assert c_line == _py_line(op, sc, mans), (
            f"{op}: parity diverged\n C: {c_line}\nPY: {_py_line(op, sc, mans)}"
        )


def test_c_channel_parser_rejects_malformed():
    """The C channel.json reader fails cleanly on a missing name/kind and a non-object."""
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        drv = os.path.join(d, "drv.c")
        open(drv, "w").write(
            '#include <stdio.h>\n#include <string.h>\n#include "bcir_channel.h"\n'
            "int main(int c,char**v){(void)c;bcir_channel ch;char dg[128];"
            'if(!bcir_channel_parse("{}",BCIR_CHANNEL_JSON_MAX_BYTES+1u,&ch,dg,sizeof dg))return 4;'
            "const char*j=v[1];memset(&ch,0xa5,sizeof ch);"
            "int r=bcir_channel_parse(j,strlen(j),&ch,dg,sizeof dg);"
            "if(!r)return 1;for(size_t i=0;i<sizeof ch;i++)if(((unsigned char*)&ch)[i])return 2;"
            "if(bcir_claim_required_caps(NULL)!=0||bcir_channel_suits(NULL,NULL)!=0||"
            "bcir_channel_route(NULL,NULL,1)!=-1)return 3;return 0;}\n"
        )
        exe = os.path.join(d, "drv")
        b = subprocess.run(
            [_CC, "-std=c11", "-O1", "-I", _C, os.path.join(_C, "bcir_channel.c"), drv, "-o", exe],
            capture_output=True,
            text=True,
        )
        assert b.returncode == 0, b.stderr
        for bad in (
            "not json",
            '{"kind":"cpu"}',
            '{"name":"x"}',
            '{"name":"x" "kind":"cpu"}',  # missing comma
            '{"name":"x","kind":"cpu",}',  # trailing comma
            '{"name":"x","kind":"cpu","modeled":"true"}',  # wrong value type
            '{"name":"x","name":"y","kind":"cpu"}',  # duplicate key
            '{"name":"x","kind":"cpu","capabilities":["telepathy"]}',
            '{"name":"x","kind":"cpu","capabilities":["reduce","reduce"]}',
            '{"name":"x","kind":"quantum"}',
            '{"name":"x","kind":"cpu","provenance":"real","modeled":true}',
            '{"name":"x\\nroot","kind":"cpu"}',  # escaped log/control injection
            '{"name":"x","kind":"cpu","profile":{"x":[1,{bad}]}}',
            '{"name":"x","kind":"cpu"} trailing',
            '{"name":"' + ("x" * 80) + '","kind":"cpu"}',
        ):
            assert subprocess.run([exe, bad]).returncode == 0, f"should reject: {bad}"
        # a well-formed minimal channel parses (returns non-zero from the probe == parse ok).
        assert (
            subprocess.run(
                [exe, '{"name":"x","kind":"cpu","capabilities":["universal"]}']
            ).returncode
            == 1
        )


# The channel reader's JSON skip, as its grammar reads: whitespace is space, tab, newline and return; a string is ASCII
# from 0x20 up to 0x7f with the escapes `\" \\ \/ \b \f \n \r \t`; `true`, `false` and `null` match as a prefix; a number
# is JSON's; and a value nested in more than 64 open containers is refused. A reference walk, written from that grammar
# and recursive, as the C one was until it took an explicit stack for R18 (`bcir_channel.c`'s `jskip`).
_JSKIP_DEPTH = 64


def _jws(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t\n\r":
        i += 1
    return i


def _jstr(s: str, i: int):
    if i >= len(s) or s[i] != '"':
        return None
    i += 1
    while i < len(s) and s[i] != '"':
        c = s[i]
        i += 1
        if not 0x20 <= ord(c) < 0x80:
            return None
        if c == "\\":
            if i >= len(s) or s[i] not in '"\\/bfnrt':
                return None
            i += 1
    return i + 1 if i < len(s) else None


def _jnum(s: str, i: int):
    def digits(j):
        k = j
        while k < len(s) and s[k].isascii() and s[k].isdigit():
            k += 1
        return k

    if i < len(s) and s[i] == "-":
        i += 1
    if i >= len(s):
        return None
    if s[i] == "0":
        i += 1
    elif "1" <= s[i] <= "9":
        i = digits(i)
    else:
        return None
    if i < len(s) and s[i] == ".":
        if digits(i + 1) == i + 1:
            return None
        i = digits(i + 1)
    if i < len(s) and s[i] in "eE":
        i += 1
        if i < len(s) and s[i] in "+-":
            i += 1
        if digits(i) == i:
            return None
        i = digits(i)
    return i


def _jskip(s: str, i: int, depth: int = 0):
    i = _jws(s, i)
    if i >= len(s) or depth > _JSKIP_DEPTH:
        return None
    c = s[i]
    if c == '"':
        return _jstr(s, i)
    if c in "{[":
        close = "}" if c == "{" else "]"
        i = _jws(s, i + 1)
        if i < len(s) and s[i] == close:
            return i + 1
        while True:
            if c == "{":
                i = _jstr(s, i)
                if i is None:
                    return None
                i = _jws(s, i)
                if i >= len(s) or s[i] != ":":
                    return None
                i += 1
            i = _jskip(s, i, depth + 1)
            if i is None:
                return None
            i = _jws(s, i)
            if i < len(s) and s[i] == close:
                return i + 1
            if i >= len(s) or s[i] != ",":
                return None
            i = _jws(s, i + 1)
            if i >= len(s) or s[i] == close:
                return None
    for word in ("true", "false", "null"):
        if s.startswith(word, i):
            return i + len(word)
    return _jnum(s, i)


def _profile_value(rnd, depth: int) -> str:
    r = rnd.random()
    if depth > 70 or r < 0.35:
        return rnd.choice(["1", "-0", "0.5", "1e9", "-1.2E-3", "true", "false", "null", '"s"', '"a\\nb"', "01", "1.",
                           "-", "tru", '"x', "nul", "1e", ".5", "truex", '"\\u0041"'])  # fmt: skip
    sep = rnd.choice([",", ", ", " ,", " "])  # ... and no comma at all
    if r < 0.68:
        body = sep.join(_profile_value(rnd, depth + 1) for _ in range(rnd.randint(0, 3)))
        return (
            rnd.choice(["[", "[ "])
            + body
            + ("," if rnd.random() < 0.05 else "")
            + rnd.choice(["]", " ]"])
        )
    members = [
        rnd.choice(['"k"', '"a b"', "k"])
        + rnd.choice([":", " : ", " "])
        + _profile_value(rnd, depth + 1)
        for _ in range(rnd.randint(0, 3))
    ]
    return "{" + sep.join(members) + ("," if rnd.random() < 0.05 else "") + "}"


def test_the_channel_readers_json_skip_is_its_grammar_at_every_depth():
    """O1. The channel reader skips the value of a key it does not know by its grammar, at most 64 containers
    deep, on an explicit stack: the recursive walk it replaced, which R18 refuses by design, is the reference. Each of
    a seeded corpus of documents, and nestings of 60 to 69 arrays, objects and empty containers across the bound, is
    accepted by the C reader exactly where the reference walk accepts its profile -- both verdicts drawn, each many
    times."""
    import random

    if not _CC:
        return
    rnd = random.Random(20261128)
    profiles = [_profile_value(rnd, 0) for _ in range(3000)]
    for k in range(60, 70):
        profiles += ["[" * k + "1" + "]" * k, '{"a":' * k + "1" + "}" * k, "[" * k + "]" * k]
    docs = [
        '{"name":"x","kind":"cpu","meta":' + v + "}" for v in profiles
    ]  # a key the reader does not know
    want = [_jskip(v, 0) == len(v) for v in profiles]
    assert sum(want) > 500 and len(want) - sum(want) > 500, sum(want)
    assert want[-30:-27] == [True, True, True] and want[-3:] == [
        False,
        False,
        False,
    ]  # 60 deep, and 69
    with tempfile.TemporaryDirectory() as d:
        drv = os.path.join(d, "drv.c")
        with open(drv, "w", encoding="utf-8") as fh:
            fh.write(
                '#include <stdio.h>\n#include <string.h>\n#include "bcir_channel.h"\n'
                "int main(void){ static char line[1 << 16]; bcir_channel ch; char dg[256];\n"
                "  while (fgets(line, sizeof line, stdin)) { size_t n = strlen(line);\n"
                "    if (n && line[n - 1] == '\\n') line[--n] = 0;\n"
                '    printf("%d\\n", bcir_channel_parse(line, n, &ch, dg, sizeof dg)); }\n  return 0; }\n'
            )
        exe = os.path.join(d, "drv")
        b = subprocess.run(
            [_CC, "-std=c11", "-O1", "-I", _C, os.path.join(_C, "bcir_channel.c"), drv, "-o", exe],
            capture_output=True,
            text=True,
        )
        assert b.returncode == 0, b.stderr
        run = subprocess.run([exe], input="\n".join(docs) + "\n", capture_output=True, text=True)
        got = [
            line == "0" for line in run.stdout.split()
        ]  # bcir_channel_parse returns 0 on a document it accepts
    assert len(got) == len(docs), (len(got), len(docs))
    wrong = [(docs[k][:120], want[k]) for k in range(len(docs)) if got[k] != want[k]]
    assert not wrong, wrong[:4]
