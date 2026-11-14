"""The C SentencePiece tokenizer (runtime/c/bcir_spm.c, PORT-2b): the oracle's ids and octets.

`bcir-tokenize --format spm` loads what `bcir.frontends.models.spm.load_sentencepiece` loads,
refuses what it refuses, and encodes (with and without the dummy prefix) and decodes as
`SpTokenizer.encode` and `SpTokenizer.decode` do. The corpus holds:
- the hand-built model of test_model_spm;
- a model trained here, whose scores tie in threes, so the leftmost-on-a-tie rule decides;
- a model with no byte alphabet (unknown symbols become `<unk>`);
- a model of every piece type: user-defined and unused pieces merge, and a control piece never
  does; a normal piece spelled like a byte piece renders as its text; the empty piece; NUL,
  U+2581 inside a piece, and scores of -0.0, subnormals and ties;
- the mini model written in odd but valid wire form: non-minimal varints, fields out of order,
  unknown fields of every wire type, a field 1 with another wire type, enormous field numbers,
  and pieces with no score or type.

Decode is compared over random ids and over every short run of byte pieces drawn from the octets
UTF-8 is decided by, so Python's errors='replace' is matched subpart for subpart. Load is compared
over documents that each break one wire or model rule (and some that keep them), and over random
byte mutations.

`check_tool` is the one comparison: this module runs it against a binary it builds from the manifest
closure, and tools/c/sections/tokenizer.sh runs it against the binary BCIR Make and CMake build.
"""

from __future__ import annotations

import functools
import itertools
import os
import random
import struct
import subprocess
import tempfile
from collections import Counter

from bcir.frontends.models.spm import load_sentencepiece

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RUNTIME_C = os.path.join(_ROOT, "runtime", "c")
_SOURCES = (
    "bcir_tokenize_cli.c",
    "bcir_bpe.c",
    "bcir_spm.c",
    "bcir_jer.c",
    "bcir_sha256.c",
    "bcir_runtime.c",
)
_SPACE = "\u2581"
NORMAL, UNKNOWN, CONTROL, USER, UNUSED, BYTE = 1, 2, 3, 4, 5, 6

# --- the protobuf wire, written by hand -----------------------------------------------------------


def _varint(v: int, pad: int = 0) -> bytes:
    """`v` as a varint, `pad` octets longer than it needs (a non-minimal spelling)."""
    out = bytearray()
    while True:
        out.append(v & 0x7F)
        v >>= 7
        if not v:
            break
        out[-1] |= 0x80
    if pad:
        out[-1] |= 0x80
        out += b"\x80" * (pad - 1) + b"\x00"
    assert len(out) <= 10
    return bytes(out)


def _tag(field: int, wire: int, pad: int = 0) -> bytes:
    return _varint(field << 3 | wire, pad)


def _ld(field: int, payload: bytes, pad: int = 0) -> bytes:
    return _tag(field, 2) + _varint(len(payload), pad) + payload


def _piece(text: str, score: float, ptype: int) -> bytes:
    msg = (
        _ld(1, text.encode("utf-8"))
        + _tag(2, 5)
        + struct.pack("<f", score)
        + _tag(3, 0)
        + _varint(ptype)
    )
    return _ld(1, msg)


def _model(pieces) -> bytes:
    return b"".join(_piece(t, s, k) for t, s, k in pieces)


def _base(bytes_too: bool = True) -> list:
    out = [("<unk>", 0.0, UNKNOWN), ("<s>", 0.0, CONTROL), ("</s>", 0.0, CONTROL)]
    if bytes_too:
        out += [(f"<0x{b:02X}>", 0.0, BYTE) for b in range(256)]
    return out


_MINI_SCORED = (
    (_SPACE, -1.0), (_SPACE + "h", -1.9), (_SPACE + "he", -1.8), (_SPACE + "hel", -1.7),
    (_SPACE + "hell", -1.6), (_SPACE + "hello", -0.5), (_SPACE + "w", -1.95), (_SPACE + "wo", -1.85),
    (_SPACE + "wor", -1.75), (_SPACE + "worl", -1.65), (_SPACE + "world", -0.6), ("he", -2.0),
    ("ll", -2.1), ("o", -3.0), ("l", -3.1), ("h", -3.2), ("e", -3.3), ("w", -3.4), ("r", -3.5),
    ("d", -3.6), ("el", -2.2),
)  # fmt: skip


def _mini() -> bytes:
    from bcir.tests.test_model_spm import _sp_piece

    parts = [_sp_piece(t, s, k) for t, s, k in _base()]
    parts += [_sp_piece(t, s, NORMAL) for t, s in _MINI_SCORED]
    parts.append(b"\x12" + _varint(3) + b"abc")  # a trainer_spec-shaped field the reader skips
    return b"".join(parts)


def _words(seed: int, count: int) -> str:
    rng = random.Random(seed)
    syllables = ("ka", "ri", "to", "men", "sha", "lo", "vi", "dra", "que", "th", "ing", "er", "st",
                 "ou", "\xe9", "\xfc", "\xe7a", "\u65e5", "\u672c", "\u8a9e", "\U0001f600")  # fmt: skip
    return " ".join(
        "".join(rng.choice(syllables) for _ in range(rng.randrange(1, 5))) for _ in range(count)
    )


_TRAINING_TEXT = (
    ("the quick brown fox jumps over the lazy dog " * 3)
    + "aaaa aaaaaaaa abab baba "
    + _words(11, 400)
)


@functools.cache
def _trained() -> bytes:
    """Character pieces, then merges by pair frequency over the ▁-prefixed words; each merge's
    score falls by one every third merge, so scores tie in threes."""
    words = Counter(tuple(_SPACE + w) for w in _TRAINING_TEXT.split())
    chars = sorted({c for w in words for c in w})
    pieces = {c: -100.0 - i * 0.5 for i, c in enumerate(chars)}
    for k in range(300):
        pairs: Counter = Counter()
        for w, n in words.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += n
        if not pairs:
            break
        a, b = min(pairs, key=lambda p: (-pairs[p], p))
        merged = a + b
        pieces.setdefault(merged, -float(k // 3))
        joined: Counter = Counter()
        for w, n in words.items():
            i, out = 0, []
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == (a, b):
                    out.append(merged)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            joined[tuple(out)] += n
        words = joined
    return _model(_base() + [(t, s, NORMAL) for t, s in pieces.items()])


def _no_bytes() -> bytes:
    return _model(_base(bytes_too=False) + [(t, s, NORMAL) for t, s in _MINI_SCORED])


def _types() -> bytes:
    pieces = _base() + [
        (_SPACE, -1.0, NORMAL),
        (_SPACE + "a", -0.0, USER),  # user-defined pieces merge like normal ones
        (_SPACE + "ab", 0.0, UNUSED),  # so do unused ones
        ("ab", 0.0, NORMAL),  # a tie with ▁ab: the leftmost pair wins
        ("bc", 0.0, NORMAL),
        ("abc", 1e-40, NORMAL),  # a subnormal score
        ("cd", 1e-45, NORMAL),
        (_SPACE + "he", 5.0, CONTROL),  # a control piece never merges, however high its score
        ("<0xZZ>", -2.0, NORMAL),  # spelled like a byte piece: renders as its text
        ("", -3.0, NORMAL),  # the empty piece
        ("\u0000", -3.0, NORMAL),
        ("a" + _SPACE + "b", -0.5, NORMAL),  # U+2581 inside a piece: a space when decoded
        ("a" + _SPACE, -0.25, NORMAL),
        ("\u65e5\u672c", -0.75, NORMAL),
        ("\U0001f600", -4.0, NORMAL),
        ("x", -9.0, NORMAL),
        ("xx", -1.0, NORMAL),
        ("xxx", -1.0, NORMAL),
        ("xxxx", -1.0, NORMAL),
        ("y", -1.0, NORMAL),
        ("z", -1.0, NORMAL),
        ("yz", 9.0, CONTROL),  # a control piece two symbols could make: it is never made
        ("q", 0.0, CONTROL),  # a control piece one symbol is: the symbol falls back to bytes
    ]
    return _model(pieces)


def _odd_wire() -> bytes:
    """The mini model's pieces in odd but valid wire form."""
    out = [_tag(2, 0) + _varint(2**64 - 1)]  # a varint field at the top, as large as one gets
    out.append(_tag(2**61 - 1, 1) + bytes(8))  # an enormous field number, 64-bit wire
    out.append(_tag(7, 5, pad=3) + bytes(4))  # a padded tag, 32-bit wire
    out.append(_tag(1, 0) + _varint(5))  # field 1 with the varint wire: not a piece, skipped
    rows = _base() + [(t, s, NORMAL) for t, s in _MINI_SCORED]
    for k, (text, score, ptype) in enumerate(rows):
        pb = text.encode("utf-8")
        fields = [
            _tag(1, 2) + _varint(len(pb), pad=k % 3) + pb,
            _tag(2, 5) + struct.pack("<f", score),
            _tag(3, 0, pad=k % 2) + _varint(ptype, pad=k % 4),
        ]
        if k % 5 == 0:
            fields.reverse()  # fields out of order
        if k % 7 == 0:
            fields.insert(
                1,
                _tag(9, 0)
                + _varint(123)
                + _tag(10, 1)
                + bytes(8)
                + _ld(11, b"xyz")
                + _tag(12, 5)
                + bytes(4),
            )
        if k % 11 == 0:
            fields.insert(
                0, _tag(1, 0) + _varint(9)
            )  # field 1 with the varint wire inside a piece: skipped
        if ptype == BYTE and k % 13 == 0:
            fields = [f for f in fields if not f.startswith(_tag(2, 5))]  # no score: 0.0
        msg = b"".join(fields)
        out.append(_tag(1, 2) + _varint(len(msg), pad=k % 2) + msg)
    out.append(_ld(1, _ld(1, b"zz")))  # a piece with text alone: score 0.0, a normal piece
    return b"".join(out)


def models() -> list[tuple[str, bytes]]:
    return [
        ("mini", _mini()),
        ("trained", _trained()),
        ("no-bytes", _no_bytes()),
        ("types", _types()),
        ("odd-wire", _odd_wire()),
    ]


# --- the texts ------------------------------------------------------------------------------------

_POOL = (
    " ", " ", " ", "a", "b", "c", "d", "x", "h", "e", "l", "o", "w", "r", "hello", "world", "the",
    "\t", "\n", "\u0000", "\xe9", "\xfc", "\xe7", "\u65e5", "\u672c", "\u8a9e", "\U0001f600", "\U0001f44d\U0001f3fd", _SPACE, "<s>", "<0x41>",
    "\xa0", "\U0010ffff", "abc", "ab",
)  # fmt: skip


def texts(rng: random.Random, count: int) -> list[str]:
    fixed = [
        "", " ", "  ", "hello world", "hell", " hello", "hello ", "  two  spaces ", "h\xe9llo w\xf6rld",
        "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8", "emoji \U0001f680 mixed", "tabs\tand\nnewlines", _SPACE + "x", _SPACE, "<s>hello</s>",
        "abc abcd ab bc", "xxxxxxxxxxx", "a" + _SPACE + "b a b", "\u0000\u0000", "<0x41>A",
        _TRAINING_TEXT[:300], "aaaaaaaaaaaaaaaaaaaaaaaaa", "abababababab", "yz q yzq", "q",
    ]  # fmt: skip
    rand = ["".join(rng.choice(_POOL) for _ in range(rng.randrange(1, 30))) for _ in range(count)]
    return fixed + rand


# --- the documents a loader must judge --------------------------------------------------------------


def law_documents() -> list[tuple[str, bytes]]:
    good = _model(_base())
    unk = _piece("<unk>", 0.0, UNKNOWN)
    out = [
        ("empty", b""),
        ("only-other-fields", _ld(2, b"abc") + _tag(3, 0) + _varint(1)),
        ("unknown-only", unk),
        ("no-unknown", _piece("a", 0.0, NORMAL)),
        ("two-unknown", unk + _piece("<unk2>", 0.0, UNKNOWN)),
        ("varint-truncated", b"\x80"),
        ("varint-11", b"\x80" * 10 + b"\x01"),
        ("varint-10th-2", b"\xff" * 9 + b"\x02"),
        # refused though the field it holds is one the reader skips
        ("varint-10th-2-skipped", unk + _tag(2, 0) + b"\xff" * 9 + b"\x02"),
        ("varint-10th-1-length", _tag(2, 2) + b"\xff" * 9 + b"\x01"),
        ("varint-10th-0", unk + _tag(2, 0) + b"\x80" * 9 + b"\x00"),
        ("field-zero", unk + b"\x02\x00"),
        ("field-zero-in-piece", _ld(1, _ld(1, b"<unk>") + _tag(3, 0) + b"\x02" + b"\x00\x00")),
        ("wire-3", unk + _tag(5, 3)),
        ("wire-4", unk + _tag(5, 4)),
        ("wire-6", unk + _tag(5, 6)),
        ("wire-7", unk + _tag(5, 7)),
        ("wire-3-in-piece", _ld(1, _ld(1, b"<unk>") + _tag(3, 0) + b"\x02" + _tag(9, 3))),
        ("length-past-end", unk + _tag(1, 2) + _varint(100) + b"x"),
        ("length-past-message", _ld(1, _tag(1, 2) + _varint(9) + b"<unk>")),
        ("torn-fixed32-score", _ld(1, _ld(1, b"<unk>") + _tag(2, 5) + b"\x00\x00")),
        ("torn-fixed64-skip", unk + _tag(9, 1) + bytes(7)),
        ("torn-fixed32-skip", unk + _tag(9, 5) + bytes(3)),
        ("piece-without-text", unk + _ld(1, _tag(2, 5) + bytes(4))),
        ("piece-text-twice", _ld(1, _ld(1, b"<unk>") + _ld(1, b"<unk>") + _tag(3, 0) + b"\x02")),
        (
            "piece-score-twice",
            _ld(1, _ld(1, b"<unk>") + (_tag(2, 5) + bytes(4)) * 2 + _tag(3, 0) + b"\x02"),
        ),
        ("piece-type-twice", _ld(1, _ld(1, b"<unk>") + (_tag(3, 0) + b"\x02") * 2)),
        (
            "piece-text-wire-0-and-2",
            _ld(1, _tag(1, 0) + b"\x01" + _ld(1, b"<unk>") + _tag(3, 0) + b"\x02"),
        ),
        ("text-overlong", unk + _ld(1, _ld(1, b"\xc0\xaf"))),
        ("text-surrogate", unk + _ld(1, _ld(1, b"\xed\xa0\x80"))),
        ("text-torn", unk + _ld(1, _ld(1, b"\xe6\x97"))),
        ("text-past-u10ffff", unk + _ld(1, _ld(1, b"\xf4\x90\x80\x80"))),
        ("text-nul", unk + _piece("\u0000", 0.0, NORMAL)),
        ("text-empty", unk + _piece("", 0.0, NORMAL)),
        ("score-nan", unk + _ld(1, _ld(1, b"a") + _tag(2, 5) + struct.pack("<I", 0x7FC00000))),
        ("score-inf", unk + _ld(1, _ld(1, b"a") + _tag(2, 5) + struct.pack("<I", 0x7F800000))),
        ("score-neg-inf", unk + _ld(1, _ld(1, b"a") + _tag(2, 5) + struct.pack("<I", 0xFF800000))),
        ("score-max", unk + _ld(1, _ld(1, b"a") + _tag(2, 5) + struct.pack("<I", 0x7F7FFFFF))),
        ("type-0", unk + _piece("a", 0.0, 0)),
        ("type-7", unk + _piece("a", 0.0, 7)),
        ("type-huge", unk + _ld(1, _ld(1, b"a") + _tag(3, 0) + _varint(2**64 - 1))),
        ("type-5", unk + _piece("a", 0.0, UNUSED)),
        ("duplicate-text", unk + _piece("a", 0.0, NORMAL) + _piece("a", -1.0, NORMAL)),
        ("duplicate-text-types", unk + _piece("<unk>", 0.0, CONTROL)),
        ("byte-alphabet", good),
        ("byte-alphabet-partial", _model(_base()[:-1])),
        ("byte-alphabet-one-normal", _model(_base()[:-1] + [("<0xFF>", 0.0, NORMAL)])),
    ]
    # each spelling stands in for one byte of an otherwise complete alphabet, so only the spelling
    # decides; `int(x, 16)` read the first four as bytes 0x01, 0x01, 0x12 and 0x1F
    for name, byte, text in (
        ("space", 0x01, "<0x 1>"),
        ("sign", 0x01, "<0x+1>"),
        ("arabic", 0x12, "<0x\u0661\u0662>"),
        ("lowercase", 0x1F, "<0x1f>"),
        ("short", 0x01, "<0x1>"),
        ("long", 0x41, "<0x041>"),
        ("bracket", 0x41, "[0x41]"),
        ("x-upper", 0x41, "<0X41>"),
    ):
        out.append((f"byte-spelled-{name}", _alphabet_with(byte, text)))
    out.append(("byte-spelled-ok", unk + _piece("<0x41>", 0.0, NORMAL)))
    return out


def _alphabet_with(byte: int, text: str) -> bytes:
    """The base model with byte `byte`'s piece spelled `text`."""
    return _model([(text if t == f"<0x{byte:02X}>" else t, s, k) for t, s, k in _base()])


def mutants(blob: bytes, rng: random.Random, count: int) -> list[bytes]:
    out = []
    for _ in range(count):
        b = bytearray(blob)
        for _ in range(rng.randrange(1, 3)):
            at = rng.randrange(len(b))
            op = rng.randrange(4)
            if op == 0:
                b[at] = rng.randrange(256)
            elif op == 1:
                del b[at]
            elif op == 2:
                b.insert(at, rng.randrange(256))
            else:
                b[at] ^= 1 << rng.randrange(8)
        out.append(bytes(b))
    return out


# --- the comparison -------------------------------------------------------------------------------

#: The octets UTF-8 is decided by: ASCII, the space, continuations at both ends of their range, the
#: leads with narrowed second octets, and the octets no character starts with.
_DECIDING = (0x41, 0x20, 0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF, 0xC0, 0xC2, 0xDF, 0xE0, 0xE2, 0xED,
             0xEF, 0xF0, 0xF4, 0xF5, 0xFF)  # fmt: skip


def _gives_a_reason(stderr: bytes) -> bool:
    lines = stderr.splitlines()
    return bool(lines) and lines[-1].startswith(b"bcir-tokenize: ")


def _refused(rc: int, out: bytes, err: bytes) -> bool:
    return rc == 2 and out == b"" and _gives_a_reason(err)


def _oracle_load(path: str, **kw):
    try:
        return load_sentencepiece(path, **kw)
    except ValueError:
        return None


def _info(tok) -> bytes:
    return (
        f"sha256 {tok.digest}\npieces {tok.n_pieces}\nunk {tok.unk_id}\nbytes {len(tok.byte_ids)}\n"
        f"specials {len(tok.specials)}\n"
    ).encode()


class _Checker:
    def __init__(self, tool: str, work: str) -> None:
        self.tool, self.work, self.failures, self.serial = tool, work, [], 0

    def file(self, data: bytes) -> str:
        self.serial += 1
        path = os.path.join(self.work, f"m{self.serial}")
        with open(path, "wb") as f:
            f.write(data)
        return path

    def run(self, args: list[str], stdin: bytes = b"") -> tuple[int, bytes, bytes]:
        p = subprocess.run(
            [self.tool, "--format", "spm", *args], input=stdin, capture_output=True, timeout=120
        )
        return p.returncode, p.stdout, p.stderr

    def fail(self, line: str) -> None:
        self.failures.append(line)

    def load(self, name: str, blob: bytes, encode_texts=()) -> object:
        path = self.file(blob)
        tok = _oracle_load(path)
        rc, out, err = self.run(["info", path])
        if tok is None:
            if not _refused(rc, out, err):
                self.fail(
                    f"{name}: the oracle refuses, C answers rc={rc} {out[:80]!r} {err[-120:]!r}"
                )
            return None
        if rc != 0 or out != _info(tok):
            self.fail(f"{name}: the oracle loads, C answers rc={rc} {out[:200]!r} {err[-200:]!r}")
            return None
        for text in encode_texts:
            for prefix in (True, False):
                self.encode(name, path, tok, text, prefix)
        return tok

    def encode(self, name: str, path: str, tok, text: str, prefix: bool) -> None:
        want = (" ".join(map(str, tok.encode(text, add_dummy_prefix=prefix))) + "\n").encode()
        args = ["encode", path] + ([] if prefix else ["--no-dummy-prefix"])
        rc, out, err = self.run(args, text.encode("utf-8"))
        if rc != 0 or out != want:
            self.fail(
                f"{name}: encode {text[:40]!r} prefix={prefix}: want {want[:100]!r}, C rc={rc} {out[:100]!r} {err[-100:]!r}"
            )

    def decode(self, name: str, path: str, tok, ids: list[int]) -> None:
        try:
            want = tok.decode(ids).encode("utf-8")
        except ValueError:
            want = None
        rc, out, err = self.run(["decode", path], " ".join(map(str, ids)).encode())
        if want is None:
            if not _refused(rc, out, err):
                self.fail(f"{name}: decode {ids[:12]}: the oracle refuses, C rc={rc} {out[:80]!r}")
        elif rc != 0 or out != want:
            self.fail(
                f"{name}: decode {ids[:12]}: want {want[:80]!r}, C rc={rc} {out[:80]!r} {err[-100:]!r}"
            )


def check_tool(tool: str, work: str, *, mutations: int = 24, seed: int = 20261110) -> list[str]:
    """Every difference between `tool --format spm` and the oracle over the corpus; [] when none."""
    rng = random.Random(seed)
    c = _Checker(tool, work)
    corpus = models()
    for name, blob in corpus:
        tok = c.load(name, blob, encode_texts=texts(rng, 30))
        if tok is None:
            c.fail(f"{name}: a corpus model must load")
            continue
        path = c.file(blob)
        for text in texts(rng, 0):  # round trips through both rails
            c.decode(name, path, tok, tok.encode(text))
        n = tok.n_pieces
        for _ in range(30):
            c.decode(name, path, tok, [rng.randrange(n) for _ in range(rng.randrange(0, 12))])
        for bad in ([n], [0, n + 5], [2**32 - 1], [n - 1, 2**31]):
            c.decode(name, path, tok, bad)
        if tok.byte_ids:  # every short run of deciding octets, through the byte pieces
            runs = [list(r) for k in (1, 2) for r in itertools.product(_DECIDING, repeat=k)]
            runs += [
                [rng.choice(_DECIDING) for _ in range(rng.randrange(3, 7))] for _ in range(120)
            ]
            for run in runs[:: 1 if name == "mini" else 7]:
                c.decode(name, path, tok, [tok.byte_ids[b] for b in run])
            word = tok.pieces.get(_SPACE + "hello", tok.pieces.get(_SPACE, (0,)))[0]
            c.decode(name, path, tok, [tok.byte_ids[0xC3], word, tok.byte_ids[0xA9]])
            c.decode(name, path, tok, [tok.byte_ids[0x20], tok.byte_ids[0x20], word])
        rc, out, err = c.run(["encode", path], b"\xc3\x28 not UTF-8")
        if not _refused(rc, out, err):
            c.fail(f"{name}: encode of text that is not UTF-8: C rc={rc} {out[:80]!r}")
    for name, blob in law_documents():
        c.load(f"law {name}", blob, encode_texts=("hello <s>a  b", ""))
    for name, blob in corpus[:1] + corpus[2:]:
        for k, m in enumerate(mutants(blob, rng, mutations)):
            c.load(f"{name} mutant {k}", m, encode_texts=("hello world", ""))
    blob = corpus[0][1]
    path = c.file(blob)
    for bound in (len(blob) - 1, len(blob), len(blob) + 1):
        want_ok = _oracle_load(path, max_bytes=bound) is not None
        rc, out, err = c.run(["--max-bytes", str(bound), "info", path])
        if want_ok != (rc == 0) or (not want_ok and not _refused(rc, out, err)):
            c.fail(f"--max-bytes {bound} over {len(blob)} octets: oracle {want_ok}, C rc={rc}")
    for args in (
        ["--format", "sp", "info", path],
        ["--format"],
        ["info", path, "extra"],
        ["decode", path, "-x"],
    ):
        p = subprocess.run([tool, *args], capture_output=True, timeout=60)
        if not _refused(p.returncode, p.stdout, p.stderr):
            c.fail(f"{args}: a refusal is rc 2, a reason and no output; C rc={p.returncode}")
    p = subprocess.run([tool, "encode", path, "--no-dummy-prefix"], capture_output=True, timeout=60)
    if not _refused(p.returncode, p.stdout, p.stderr):
        c.fail("--no-dummy-prefix is SentencePiece's: the BPE format refuses it")
    return c.failures


def _build(tmp: str) -> str | None:
    """The tool, built from the checkout's sources; None where there are none to build (an
    installed package ships no runtime/c) or no compiler (the quick tier hides it)."""
    from bcir.tests.run_all import _is_source_checkout
    from bcir.toolchain import host_c_compiler

    cc = host_c_compiler()
    if cc is None or not _is_source_checkout():
        return None
    exe = os.path.join(tmp, "bcir-tokenize.exe" if os.name == "nt" else "bcir-tokenize")
    sources = [os.path.join(_RUNTIME_C, s) for s in _SOURCES]
    last = None
    for std in ("-std=c23", "-std=c2x"):  # GCC 13 knows C23 only as c2x
        last = subprocess.run(
            [cc, std, "-O2", "-Wall", "-Wextra", "-I", _RUNTIME_C, *sources, "-o", exe],
            capture_output=True,
            text=True,
        )
        if last.returncode == 0:
            return exe
    raise AssertionError(f"bcir-tokenize did not build: {last.stderr if last else ''}")


def test_the_fixture_builds_what_the_gate_links():
    from bcir.tests.gate_links import assert_fixture_links

    assert_fixture_links("tools", "bcir-tokenize", _SOURCES)


def test_the_corpus_reaches_what_the_encoder_and_the_laws_decide():
    loaded, verdicts = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m")
        for name, blob in models():
            with open(path, "wb") as f:
                f.write(blob)
            loaded[name] = load_sentencepiece(path)
        for name, blob in law_documents():
            with open(path, "wb") as f:
                f.write(blob)
            verdicts[name] = _oracle_load(path) is not None
    trained = loaded["trained"]
    scores = Counter(s for _, s in trained.pieces.values() if s > -100.0)
    assert len(trained.pieces) > 300 and max(scores.values()) >= 3, "merges whose scores tie"
    assert len(trained.encode(_TRAINING_TEXT[:200])) < 120, "the trained merges apply"
    assert loaded["no-bytes"].encode("zz")[-2:] == [0, 0], "unknown symbols are <unk> without bytes"
    types = loaded["types"]
    assert types.encode("ab") == [types.pieces[_SPACE + "ab"][0]], (
        "unused pieces merge; ties go left"
    )
    assert types.decode([types.pieces["<0xZZ>"][0]]) == "<0xZZ>", (
        "a normal piece renders as its text"
    )
    assert types.encode("he") != [types.pieces.get(_SPACE + "he", (None,))[0]], (
        "a control piece never merges"
    )
    mini = load_sentencepiece_bytes(_mini()).ids_to_pieces
    assert loaded["odd-wire"].ids_to_pieces == {**mini, len(mini): "zz"}, (
        "the odd wire reads as the mini"
    )
    for keep in (
        "varint-10th-0",
        "text-nul",
        "text-empty",
        "score-max",
        "type-5",
        "byte-alphabet",
        "byte-spelled-ok",
        "piece-text-wire-0-and-2",
    ):
        assert verdicts[keep], keep
    for refuse in (
        "byte-spelled-lowercase",
        "byte-spelled-space",
        "byte-spelled-sign",
        "byte-spelled-arabic",
        "score-nan",
        "text-surrogate",
        "varint-10th-2",
        "wire-4",
        "field-zero",
    ):
        assert not verdicts[refuse], refuse


def load_sentencepiece_bytes(blob: bytes):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m")
        with open(path, "wb") as f:
            f.write(blob)
        return load_sentencepiece(path)


def test_the_oracle_reads_byte_pieces_by_their_one_spelling_and_decodes_by_type():
    """PORT-2b's oracle fixes: `int(x, 16)` read `<0x 1>`, `<0x+1>` and Arabic-Indic digits as byte
    pieces; decode chose bytes by spelling, so a normal piece spelled `<0xZZ>` crashed it; and an
    id that names no piece was dropped silently."""
    unk = _piece("<unk>", 0.0, UNKNOWN)
    for byte, text in (
        (0x01, "<0x 1>"),
        (0x01, "<0x+1>"),
        (0x12, "<0x\u0661\u0662>"),
        (0x1F, "<0x1f>"),
    ):
        try:
            load_sentencepiece_bytes(_alphabet_with(byte, text))
        except ValueError:
            continue
        raise AssertionError(f"accepted the byte piece {text!r}")
    assert load_sentencepiece_bytes(_alphabet_with(0x1F, "<0x1F>")).byte_ids[0x1F]
    tok = load_sentencepiece_bytes(unk + _piece("<0xZZ>", 0.0, NORMAL))
    assert tok.decode([1]) == "<0xZZ>"
    for bad in ([2], [-1], [True], ["1"]):
        try:
            tok.decode(bad)
        except ValueError:
            continue
        raise AssertionError(f"decoded {bad!r}")


def test_the_c_tool_tokenizes_as_the_oracle_does():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return  # no compiler or no sources here; the checkout's c-runtime tier and the gate run it
        failures = check_tool(exe, tmp, mutations=96 if os.environ.get("BCIR_THOROUGH") else 24)
        assert not failures, "\n".join(failures[:20])
