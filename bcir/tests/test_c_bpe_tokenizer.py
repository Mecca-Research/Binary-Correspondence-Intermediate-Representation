"""The C byte-level BPE tokenizer (runtime/c/bcir_bpe.c, PORT-2): the oracle's ids, id for id.

`bcir-tokenize` in C loads what `bcir.frontends.models.tokenizer.load_tokenizer` loads, refuses what
it refuses, and encodes and decodes as `Tokenizer.encode` and `Tokenizer.decode` do. The corpus holds
the hand-computed mini tokenizer, a tokenizer trained here on text with long repeated runs (so a word
takes hundreds of merges, ties included, and two merges can make one token), the same merges in the
[a, b] pair form under a permuted, decorated member order, and a tokenizer whose specials overlap,
share a vocab token's text and id, live outside the byte alphabet, and include the space itself.
Texts cover runs of spaces, every UTF-8 width, NUL, specials back to back and random strings over all
of those. Decode is compared over random id sequences, which split UTF-8 and name unknown ids. Load
is compared over a list of documents, each breaking one law or one reader limit (and a few keeping
them), and over random byte mutations of real files.

`check_tool` is the one comparison: this module runs it against a binary it builds from the manifest
closure (held there by `assert_fixture_links`), and tools/c/sections/tokenizer.sh runs it against the
binary BCIR Make and CMake build.
"""

from __future__ import annotations

import functools
import json
import os
import random
import subprocess
import tempfile
from collections import Counter

from bcir.frontends.models.tokenizer import (
    TOKENIZER_LIMITS,
    _pretokenize,
    bytes_to_unicode,
    load_tokenizer,
)

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RUNTIME_C = os.path.join(_ROOT, "runtime", "c")
_SOURCES = (
    "bcir_tokenize_cli.c",
    "bcir_bpe.c",
    "bcir_jer.c",
    "bcir_sha256.c",
    "bcir_runtime.c",
)
_B2U = bytes_to_unicode()

# --- the tokenizers -------------------------------------------------------------------------------


def _base_vocab() -> dict[str, int]:
    return {_B2U[b]: b for b in range(256)}


def _alphabet(text: str) -> str:
    return "".join(_B2U[b] for b in text.encode("utf-8"))


def _train(text: str, merges: int) -> tuple[dict[str, int], list[tuple[str, str]]]:
    """A deterministic byte-level BPE trainer: merge the most frequent adjacent pair (ties: the
    smallest pair) over the pre-tokenized words, `merges` times. A merge can make a token another
    merge already made; that is legal, and the corpus wants it."""
    words = Counter(tuple(_alphabet(w)) for w in _pretokenize(text))
    vocab = _base_vocab()
    out: list[tuple[str, str]] = []
    for _ in range(merges):
        pairs: Counter = Counter()
        for w, n in words.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += n
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        out.append(best)
        merged = best[0] + best[1]
        vocab.setdefault(merged, len(vocab))
        joined: Counter = Counter()
        for w, n in words.items():
            k, nw = 0, []
            while k < len(w):
                if k + 1 < len(w) and (w[k], w[k + 1]) == best:
                    nw.append(merged)
                    k += 2
                else:
                    nw.append(w[k])
                    k += 1
            joined[tuple(nw)] += n
        words = joined
    return vocab, out


def _words(seed: int, count: int) -> str:
    """Pseudo-words from syllables of every UTF-8 width, so training finds hundreds of pairs."""
    rng = random.Random(seed)
    syllables = (
        "ka",
        "ri",
        "to",
        "men",
        "sha",
        "lo",
        "vi",
        "dra",
        "que",
        "nus",
        "th",
        "ing",
        "tion",
        "er",
        "st",
        "ou",
        "é",
        "ü",
        "ça",
        "日",
        "本",
        "語",
        "の",
        "😀",
        "👍🏽",
    )
    words = (
        "".join(rng.choice(syllables) for _ in range(rng.randrange(1, 5))) for _ in range(count)
    )
    return "".join(w + rng.choice((" ", " ", " ", "  ", "\n")) for w in words)


_TRAINING_TEXT = (
    "the quick brown fox jumps over the lazy dog and the dog sleeps; "
    "aaaa aaaaaaaa aaaaaaaaaaaa abababab abab baba bbbb "
    "héllo wörld, ça va très bien; naïve façade "
    "日本語のテキストと日本の言葉 "
    "emoji 😀😀 👍🏽 done "
) * 4 + _words(7, 600)


def _mini() -> dict:
    vocab = _base_vocab()
    for tok, i in (("he", 256), ("ll", 257), ("hell", 258), ("hello", 259)):
        vocab[tok] = i
    return {
        "model": {"type": "BPE", "vocab": vocab, "merges": ["h e", "l l", "he ll", "hell o"]},
        "added_tokens": [
            {"content": "<bos>", "id": 260},
            {"content": "<eos>", "id": 261},
            {"content": "<start_of_turn>", "id": 262},
            {"content": "<end_of_turn>", "id": 263},
        ],
    }


@functools.cache
def _trained_json() -> str:
    vocab, merges = _train(_TRAINING_TEXT, 320)
    # sparse ids: the merged tokens move far apart, and one sits at the top of the id range
    moved = {}
    for tok, i in sorted(vocab.items(), key=lambda kv: kv[1]):
        moved[tok] = i if i < 256 else 1000 + 7 * (i - 256)
    top = max(moved, key=lambda t: moved[t])
    moved[top] = 0x7FFFFFFF
    # Last, at the lowest ranks: a second merge for tokens the trained merges already make, as
    # (a + b[0], b[1:]) beside (a, b), with (a, b[0]) to make its left part.
    have = set(merges)
    for a, b in list(merges[:40]):
        if len(b) < 2 or b[1:] not in moved:
            continue
        left = a + b[0]
        moved.setdefault(left, 1_000_000 + len(moved))
        for pair in ((a, b[0]), (left, b[1:])):
            if pair not in have:
                have.add(pair)
                merges.append(pair)
    return json.dumps(
        {
            "model": {"type": "BPE", "vocab": moved, "merges": [f"{a} {b}" for a, b in merges]},
            "added_tokens": [{"content": "<|endoftext|>", "id": 900, "special": True}],
        }
    )


def _trained() -> dict:
    return json.loads(_trained_json())


def _decorated() -> dict:
    """The trained tokenizer in the [a, b] pair form, members permuted and decorated the way real
    files are -- every extra member is read, held to the grammar and not kept."""
    base = _trained()
    model = base["model"]
    return {
        "added_tokens": [
            {
                "id": 900,
                "content": "<|endoftext|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {"special": False, "content": "<|pad|>", "id": 901, "extra": [1.5, -0.0, 1e10, None]},
        ],
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "normalizer": {"type": "NFC", "nested": [{"a": [[], {}, ""]}]},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "use_regex": True},
        "model": {
            "merges": [[a, b] for a, b in (m.split(" ") for m in model["merges"])],
            "ignore_merges": False,
            "dropout": None,
            "unk_token": None,
            "byte_fallback": False,
            "vocab": dict(reversed(list(model["vocab"].items()))),
            "type": "BPE",
        },
        "decoder": {"type": "ByteLevel", "trim_offsets": True, "ratio": 0.25e-3},
    }


def _specials() -> dict:
    """Specials that overlap (longest wins), that equal a vocab token's text under its id (so the
    id renders as text, not bytes), that lie outside the byte alphabet, and the space itself."""
    vocab = _base_vocab()
    vocab[_alphabet(" hello")] = 300
    vocab[_alphabet(" h")] = 301
    vocab[_alphabet("ello")] = 302
    vocab[_alphabet("el")] = 303
    vocab[_alphabet("ll")] = 304
    vocab[_alphabet("llo")] = 305
    vocab[_alphabet("ell")] = 306
    vocab[_alphabet("  ")] = 307
    merges = [
        f"{_alphabet(' ')} {_alphabet(' ')}",  # the only merge across spaces: the pre-tokenizer decides
        f"{_alphabet('l')} {_alphabet('l')}",
        [_alphabet("e"), _alphabet("ll")],
        f"{_alphabet('ll')} {_alphabet('o')}",
        f"{_alphabet('e')} {_alphabet('llo')}",
        f"{_alphabet('e')} {_alphabet('l')}",
        f"{_alphabet(' ')} {_alphabet('h')}",
        f"{_alphabet(' h')} {_alphabet('ello')}",
    ]
    added = [
        {"content": "<", "id": ord("<")},  # a byte's own alphabet text: its base id, now a special
        {"content": "<<", "id": 401},
        {"content": "<a>", "id": 402},
        {"content": "<a>b", "id": 403},
        {"content": "é", "id": 0xE9},  # the alphabet text of byte 0xE9, under its own id
        {"content": _alphabet(" hello"), "id": 300},  # a merged vocab token's text, its id
        {"content": "日本", "id": 404},
        {"content": "\u0000", "id": 405},
        {"content": "😀", "id": 406},
        {"content": "\t\t", "id": 407},
    ]
    return {"model": {"type": "BPE", "vocab": vocab, "merges": merges}, "added_tokens": added}


def _spaced() -> dict:
    """The space as a special: the pre-tokenizer never sees one."""
    doc = _mini()
    doc["added_tokens"] = [{"content": " ", "id": 500}, {"content": "  ", "id": 501}]
    return doc


def _bare() -> dict:
    """Base bytes only: no merges member, no added tokens."""
    return {"model": {"type": "BPE", "vocab": _base_vocab()}}


def tokenizers() -> list[tuple[str, bytes]]:
    return [
        ("mini", json.dumps(_mini()).encode()),
        ("mini-utf8", json.dumps(_mini(), ensure_ascii=False, indent=1).encode("utf-8")),
        ("trained", json.dumps(_trained()).encode()),
        ("decorated", json.dumps(_decorated(), ensure_ascii=False).encode("utf-8")),
        ("specials", json.dumps(_specials()).encode()),
        ("spaced", json.dumps(_spaced()).encode()),
        ("bare", json.dumps(_bare(), separators=(",", ":")).encode()),
    ]


# --- the texts ------------------------------------------------------------------------------------

_POOL = (
    " ",
    " ",
    " ",
    "a",
    "b",
    "h",
    "e",
    "l",
    "o",
    "hello",
    "the",
    "dog",
    "\t",
    "\n",
    "\r",
    "\u0000",
    "é",
    "ü",
    "ç",
    "日",
    "本",
    "語",
    "😀",
    "👍🏽",
    " ",
    " ",
    "<",
    "<<",
    "<a>",
    "<a>b",
    ">",
    "<|endoftext|>",
    "<bos>",
    "<eos>",
    "aaaa",
    "abab",
    "Ā",
    "Ń",
    "\U0010ffff",
)


def texts(rng: random.Random, count: int) -> list[str]:
    fixed = [
        "",
        "hello",
        " hello",
        "  hello",
        "hello ",
        "hello  ",
        "   ",
        " ",
        "a b  c   d    e",
        "\thello\nworld\r\n",
        "héllo wörld",
        "日本語のテキスト",
        "emoji 😀👍🏽 mixed",
        "<bos>hello<eos>",
        "<<a>b<a><<<",
        "<a>b<a>b<a>",
        "\u0000nul\u0000",
        " hello hello  hello",
        "é é  é",
        "日本日本語",
        "\t\t\t\t\t",
        "the quick brown fox jumps over the lazy dog " * 3,
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "ab" * 150,
        "abab" * 40 + " " + "ba" * 70,
        _TRAINING_TEXT[:400],
        "\U0010ffffĀŃ",
        "<|endoftext|>the<|endoftext|>",
    ]
    rand = ["".join(rng.choice(_POOL) for _ in range(rng.randrange(1, 40))) for _ in range(count)]
    return fixed + rand


# --- the documents a loader must judge --------------------------------------------------------------


def _dump(doc) -> bytes:
    return json.dumps(doc).encode()


def law_documents() -> list[tuple[str, bytes]]:
    """Documents one law or one reader limit away from the mini tokenizer, and a few that keep
    them all. Raw text where json.dumps cannot say it (repeated names, NaN, bad UTF-8...)."""
    out: list[tuple[str, bytes]] = []

    def edit(name, fn):
        doc = _mini()
        fn(doc)
        out.append((name, _dump(doc)))

    def setm(key, value):
        return lambda d: d["model"].__setitem__(key, value)

    vocab_key = _B2U[ord("a")]
    out += [("root-array", b"[]"), ("root-string", b'"BPE"'), ("root-empty", b"{}"), ("empty", b"")]
    edit("model-null", lambda d: d.__setitem__("model", None))
    edit("model-array", lambda d: d.__setitem__("model", []))
    edit("model-missing", lambda d: d.pop("model"))
    edit("type-missing", lambda d: d["model"].pop("type"))
    edit("type-wordpiece", setm("type", "WordPiece"))
    edit("type-lower", setm("type", "bpe"))
    edit("type-number", setm("type", 1))
    edit("vocab-array", setm("vocab", []))
    edit("vocab-null", setm("vocab", None))
    edit("vocab-missing", lambda d: d["model"].pop("vocab"))
    edit("id-float", lambda d: d["model"]["vocab"].__setitem__(vocab_key, 97.0))
    edit("id-negative", lambda d: d["model"]["vocab"].__setitem__(vocab_key, -1))
    edit("id-top", lambda d: d["model"]["vocab"].__setitem__(vocab_key, 0x7FFFFFFF))
    edit("id-past-top", lambda d: d["model"]["vocab"].__setitem__(vocab_key, 0x80000000))
    edit("id-true", lambda d: d["model"]["vocab"].__setitem__(vocab_key, True))
    edit("id-string", lambda d: d["model"]["vocab"].__setitem__(vocab_key, "97"))
    edit("id-null", lambda d: d["model"]["vocab"].__setitem__(vocab_key, None))
    edit("id-duplicate", lambda d: d["model"]["vocab"].__setitem__(vocab_key, 98))
    edit("base-byte-missing", lambda d: d["model"]["vocab"].pop(_B2U[0]))
    edit("token-outside-alphabet", lambda d: d["model"]["vocab"].__setitem__(" ", 900))
    edit("token-cjk", lambda d: d["model"]["vocab"].__setitem__("日", 900))
    edit("token-empty", lambda d: d["model"]["vocab"].__setitem__("", 900))
    edit("token-extra", lambda d: d["model"]["vocab"].__setitem__("Ġhello", 900))
    edit("merges-object", setm("merges", {}))
    edit("merges-null", setm("merges", None))
    edit("merges-empty", setm("merges", []))
    edit("merges-missing", lambda d: d["model"].pop("merges"))
    for i, bad in enumerate(
        (
            "h  e",
            " h e",
            "h e ",
            "he",
            "h e l",
            "",
            " ",
            "x y",
            ["h"],
            ["h", "e", "l"],
            [],
            ["h", ""],
            ["", "e"],
            [1, 2],
            ["h", None],
            [["h"], "e"],
            5,
            None,
            {"h": "e"},
        )
    ):
        edit(f"merge-bad-{i}", lambda d, b=bad: d["model"]["merges"].append(b))
    edit("merge-pair-form", lambda d: d["model"]["merges"].__setitem__(0, ["h", "e"]))
    edit("merge-duplicate", lambda d: d["model"]["merges"].append("l l"))
    edit("merge-duplicate-forms", lambda d: d["model"]["merges"].append(["h", "e"]))
    edit("merge-missing-part", lambda d: d["model"]["merges"].append("h qĀ"))
    edit("merge-missing-concat", lambda d: d["model"]["merges"].append("o h"))
    edit("merge-reordered", lambda d: d["model"]["merges"].reverse())
    edit("added-object", lambda d: d.__setitem__("added_tokens", {}))
    edit("added-null", lambda d: d.__setitem__("added_tokens", None))
    edit("added-missing", lambda d: d.pop("added_tokens"))
    edit("added-item-array", lambda d: d["added_tokens"].append([]))
    edit("added-item-string", lambda d: d["added_tokens"].append("<x>"))
    edit("added-no-content", lambda d: d["added_tokens"].append({"id": 600}))
    edit("added-empty-content", lambda d: d["added_tokens"].append({"content": "", "id": 600}))
    edit("added-number-content", lambda d: d["added_tokens"].append({"content": 5, "id": 600}))
    edit("added-no-id", lambda d: d["added_tokens"].append({"content": "<x>"}))
    edit("added-float-id", lambda d: d["added_tokens"].append({"content": "<x>", "id": 600.0}))
    edit("added-bool-id", lambda d: d["added_tokens"].append({"content": "<x>", "id": False}))
    edit("added-duplicate", lambda d: d["added_tokens"].append({"content": "<bos>", "id": 600}))
    edit("added-dup-same-id", lambda d: d["added_tokens"].append({"content": "<bos>", "id": 260}))
    edit(
        "added-vocab-other-id", lambda d: d["added_tokens"].append({"content": "hello", "id": 600})
    )
    edit("added-vocab-same-id", lambda d: d["added_tokens"].append({"content": "hello", "id": 259}))
    edit("added-id-of-token", lambda d: d["added_tokens"].append({"content": "<x>", "id": 259}))
    edit("added-id-of-special", lambda d: d["added_tokens"].append({"content": "<x>", "id": 260}))
    edit("added-id-of-byte", lambda d: d["added_tokens"].append({"content": "a", "id": 97}))
    edit("added-ignored-extras", lambda d: d["added_tokens"][0].update(special=True, lstrip=None))
    good = _dump(_mini())
    body = good[1:-1]
    out += [
        ("dup-root-key", b'{"model":{"type":"BPE"},' + body + b"}"),
        ("dup-model-key", good.replace(b'"type": "BPE"', b'"type": "BPE", "type": "BPE"', 1)),
        ("dup-vocab-key", good.replace(b'"vocab": {', b'"vocab": {"a": 97, ', 1)),
        ("dup-ignored-key", b'{"x":{"k":1,"k":2},' + body + b"}"),
        ("dup-in-added-item", good.replace(b'"id": 260', b'"id": 260, "id": 260', 1)),
        ("same-key-two-objects", b'{"x":{"k":1},"y":{"k":2},' + body + b"}"),
        ("nan-ignored", b'{"x":NaN,' + body + b"}"),
        ("infinity-ignored", b'{"x":-Infinity,' + body + b"}"),
        ("lone-surrogate-ignored", b'{"x":"\\ud800",' + body + b"}"),
        ("surrogate-pair-ignored", b'{"x":"\\ud83d\\ude00",' + body + b"}"),
        ("lone-surrogate-content", good.replace(b'"<bos>"', b'"<\\udc00>"', 1)),
        ("escaped-content", good.replace(b'"<bos>"', b'"\\u003cbos\\u003e"', 1)),
        ("bad-utf8-ignored", b'{"x":"\xc3\x28",' + body + b"}"),
        ("overlong-utf8", b'{"x":"\xc0\xaf",' + body + b"}"),
        ("bom", b"\xef\xbb\xbf" + good),
        ("trailing-garbage", good + b" x"),
        ("trailing-space", good + b" \n\t\r"),
        ("two-documents", good + good),
        ("deep-64", b'{"x":' + b"[" * 63 + b"]" * 63 + b"," + body + b"}"),
        ("deep-65", b'{"x":' + b"[" * 64 + b"]" * 64 + b"," + body + b"}"),
        ("digits-64", b'{"x":' + b"9" * 64 + b"," + body + b"}"),
        ("digits-65", b'{"x":' + b"9" * 65 + b"," + body + b"}"),
        ("exponent-4096", b'{"x":1e4096,' + body + b"}"),
        ("exponent-4097", b'{"x":1e4097,' + body + b"}"),
        ("leading-zero", b'{"x":01,' + body + b"}"),
        ("id-minus-zero", good.replace(b'"\\u0100": 0,', b'"\\u0100": -0,', 1)),
        ("id-exponent", good.replace(b'"id": 260', b'"id": 26e1', 1)),
        ("id-fraction", good.replace(b'"id": 260', b'"id": 260.0', 1)),
        ("id-huge", good.replace(b'"id": 260', b'"id": ' + b"9" * 40, 1)),
        ("comment", b"/*c*/" + good),
        ("single-quotes", b"{'x':1," + body + b"}"),
        ("control-in-string", b'{"x":"a\tb",' + body + b"}"),
        ("escaped-control", b'{"x":"a\\tb\\u0000",' + body + b"}"),
        ("long-string", b'{"x":"' + b"a" * (1 << 20) + b'",' + body + b"}"),
        ("too-long-string", b'{"x":"' + b"a" * ((1 << 20) + 1) + b'",' + body + b"}"),
    ]
    return out


def limit_documents() -> list[tuple[str, bytes]]:
    """Documents at and past the profile's per-container ceilings, which no real file is near and
    the law documents do not reach: what tells TOKENIZER_LIMITS from the reader's defaults (10 000
    members, 100 000 elements). Megabytes each, so only the C comparison reads them."""
    body = _dump(_mini())[1:-1]
    elements = TOKENIZER_LIMITS["elements"]
    members = TOKENIZER_LIMITS["members"]
    return [
        ("elements-at-limit", b'{"x":[' + b"0," * (elements - 1) + b"0]," + body + b"}"),
        ("elements-past-limit", b'{"x":[' + b"0," * elements + b"0]," + body + b"}"),
        (
            "members-past-limit",
            b'{"x":{' + b",".join(b'"%d":0' % i for i in range(members + 1)) + b"}," + body + b"}",
        ),
    ]


def mutants(blob: bytes, rng: random.Random, count: int) -> list[bytes]:
    """Byte-level edits of a real file: most break the grammar, some change a token, an id or a
    merge and reach the laws."""
    out = []
    for _ in range(count):
        b = bytearray(blob)
        for _ in range(rng.randrange(1, 3)):
            at = rng.randrange(len(b))
            op = rng.randrange(5)
            if op == 0:
                b[at] = (
                    rng.choice(b'0123456789 "[]{},:-.eE\\u')
                    if rng.random() < 0.6
                    else rng.randrange(256)
                )
            elif op == 1:
                del b[at]
            elif op == 2:
                b.insert(at, rng.choice(b'0123456789 "[]{},:'))
            elif op == 3:
                n = rng.randrange(1, 12)
                b[at:at] = b[at : at + n]
            else:  # one digit of a number, or one character of a string, moved by one
                b[at] = (b[at] + rng.choice((1, -1))) & 0xFF
        out.append(bytes(b))
    return out


# --- the comparison -------------------------------------------------------------------------------


def _run(tool: str, args: list[str], stdin: bytes = b"") -> tuple[int, bytes, bytes]:
    p = subprocess.run([tool, *args], input=stdin, capture_output=True, timeout=120)
    return p.returncode, p.stdout, p.stderr


def _gives_a_reason(stderr: bytes) -> bool:
    lines = stderr.splitlines()
    return bool(lines) and lines[-1].startswith(b"bcir-tokenize: ")


def _oracle_load(path: str, **kw):
    try:
        return load_tokenizer(path, **kw)
    except ValueError:
        return None


def _info(tok) -> bytes:
    return (
        f"sha256 {tok.digest}\nvocab {len(tok.vocab)}\nmerges {len(tok.merges)}\n"
        f"specials {len(tok.specials)}\n"
    ).encode()


def _refused(rc: int, out: bytes, err: bytes) -> bool:
    return rc == 2 and out == b"" and _gives_a_reason(err)


class _Checker:
    def __init__(self, tool: str, work: str) -> None:
        self.tool, self.work, self.failures, self.calls = tool, work, [], 0
        self.serial = 0

    def file(self, data: bytes) -> str:
        self.serial += 1
        path = os.path.join(self.work, f"in{self.serial}")
        with open(path, "wb") as f:
            f.write(data)
        return path

    def run(self, args: list[str], stdin: bytes = b"") -> tuple[int, bytes, bytes]:
        self.calls += 1
        return _run(self.tool, args, stdin)

    def fail(self, line: str) -> None:
        self.failures.append(line)

    def load(
        self, name: str, blob: bytes, *, deep: bool, rng: random.Random, encode_texts=()
    ) -> object:
        """Both rails judge `blob`; when both accept, the info and some encodings agree."""
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
            self.encode(name, path, tok, text)
        if deep:
            self.decodes(name, path, tok, rng)
        return tok

    def encode(self, name: str, path: str, tok, text: str) -> None:
        want = (" ".join(map(str, tok.encode(text))) + "\n").encode()
        rc, out, err = self.run(["encode", path], text.encode("utf-8"))
        if rc != 0 or out != want:
            self.fail(
                f"{name}: encode {text[:40]!r}: want {want[:120]!r}, C rc={rc} {out[:120]!r} {err[-120:]!r}"
            )

    def decode(self, name: str, path: str, tok, ids: list[int], raw: bytes | None = None) -> None:
        try:
            want = tok.decode(ids).encode("utf-8")
        except ValueError:  # an unknown id, or a run that is not UTF-8 (UnicodeDecodeError)
            want = None
        rc, out, err = self.run(
            ["decode", path], raw if raw is not None else " ".join(map(str, ids)).encode()
        )
        if want is None:
            if not _refused(rc, out, err):
                self.fail(f"{name}: decode {ids[:12]}: the oracle refuses, C rc={rc} {out[:80]!r}")
        elif rc != 0 or out != want:
            self.fail(
                f"{name}: decode {ids[:12]}: want {want[:80]!r}, C rc={rc} {out[:80]!r} {err[-120:]!r}"
            )

    def decodes(self, name: str, path: str, tok, rng: random.Random) -> None:
        byte = {b: tok.vocab[_B2U[b]] for b in (0xC3, 0xA9, 0xE6, 0x97, 0xA5, 0xF0)}
        for special in list(tok.specials.values())[:2]:  # UTF-8 is judged per run between specials
            self.decode(name, path, tok, [byte[0xC3], special, byte[0xA9]])
            self.decode(name, path, tok, [byte[0xE6], byte[0x97], special])
            self.decode(name, path, tok, [special, byte[0xA9]])
        for ids in (
            [byte[0xC3], byte[0xA9]],
            [byte[0xC3]],
            [byte[0xA9]],
            [byte[0xE6], byte[0x97]],
            [byte[0xE6], byte[0x97], byte[0xA5]],
            [byte[0xF0]],
        ):
            self.decode(name, path, tok, ids)
        known = sorted(tok.ids_to_tokens)
        unknown = [
            max(known) + 1 if max(known) < 0x7FFFFFFF else 0x7FFFFFFE,
            0x80000000,
            0xFFFFFFFF,
        ]
        for _ in range(24):
            n = rng.randrange(0, 10)
            ids = [
                rng.choice(known) if rng.random() < 0.97 else rng.choice(unknown) for _ in range(n)
            ]
            self.decode(name, path, tok, ids)
        self.decode(name, path, tok, [], b"")
        self.decode(
            name, path, tok, [known[0], known[-1]], f" \t{known[0]}\r\n{known[-1]}\n".encode()
        )
        rc, out, err = self.run(["decode", path], b"1 x")
        if not _refused(rc, out, err):
            self.fail(f"{name}: decode of a non-number: C rc={rc} {out[:80]!r}")
        rc, out, err = self.run(["decode", path], b"9" * 30)
        if not _refused(rc, out, err):
            self.fail(f"{name}: decode of a 30-digit id: C rc={rc} {out[:80]!r}")


def check_tool(tool: str, work: str, *, mutations: int = 24, seed: int = 20261110) -> list[str]:
    """Every difference between `tool` and the oracle over the corpus, as lines; [] when none."""
    rng = random.Random(seed)
    c = _Checker(tool, work)
    rc, out, _ = c.run(["limits"])
    want = "".join(f"{k} {v}\n" for k, v in TOKENIZER_LIMITS.items()).encode()
    if rc != 0 or out != want:
        c.fail(f"limits: want {want!r}, C rc={rc} {out!r}")
    corpus = tokenizers()
    for name, blob in corpus:
        tok = c.load(name, blob, deep=True, rng=rng, encode_texts=texts(rng, 40))
        if tok is None:
            c.fail(f"{name}: a corpus tokenizer must load")
            continue
        path = c.file(blob)
        for text in texts(rng, 0)[:12]:  # round trips through both rails
            ids = tok.encode(text)
            c.decode(name, path, tok, ids)
        rc, out, err = c.run(["encode", path], b"\xc3\x28 not UTF-8")
        if not _refused(rc, out, err):
            c.fail(f"{name}: encode of text that is not UTF-8: C rc={rc} {out[:80]!r}")
    for name, blob in law_documents():
        c.load(
            f"law {name}", blob, deep=False, rng=rng, encode_texts=("hello <bos>a  b", "\u0000x")
        )
    for name, blob in limit_documents():
        c.load(f"limit {name}", blob, deep=False, rng=rng)
    for name, blob in corpus[:3] + corpus[4:5]:
        for k, m in enumerate(mutants(blob, rng, mutations)):
            c.load(
                f"{name} mutant {k}",
                m,
                deep=False,
                rng=rng,
                encode_texts=("hello world", "<bos> é"),
            )
    # --max-bytes is load_tokenizer's max_bytes
    blob = corpus[0][1]
    path = c.file(blob)
    for bound in (len(blob) - 1, len(blob), len(blob) + 1):
        want_ok = _oracle_load(path, max_bytes=bound) is not None
        rc, out, err = c.run(["--max-bytes", str(bound), "info", path])
        if want_ok != (rc == 0) or (not want_ok and not _refused(rc, out, err)):
            c.fail(f"--max-bytes {bound} over {len(blob)} octets: oracle {want_ok}, C rc={rc}")
    for args in (
        ["info"],
        [],
        ["frobnicate", path],
        ["info", path, "extra"],
        ["--max-bytes", "0", "info", path],
        ["--max-bytes", "268435457", "info", path],
        ["--max-bytes", "1e3", "info", path],
        ["info", os.path.join(work, "no-such-file")],
        ["limits", "x"],
        ["encode", path, "-q"],
    ):
        rc, out, err = c.run(args)
        if not _refused(rc, out, err):
            c.fail(
                f"{args}: a refusal is rc 2, a reason and no output; C rc={rc} {out[:80]!r} {err[-80:]!r}"
            )
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
    loaded = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, blob in tokenizers():
            path = os.path.join(tmp, name)
            with open(path, "wb") as f:
                f.write(blob)
            loaded[name] = load_tokenizer(path)
        verdicts = {}
        for name, blob in law_documents():
            path = os.path.join(tmp, "law")
            with open(path, "wb") as f:
                f.write(blob)
            verdicts[name] = _oracle_load(path) is not None
    trained = loaded["trained"]
    assert len(trained.merges) >= 300
    merged = Counter(a + b for a, b in trained.merges)
    assert any(n > 1 for n in merged.values()), "two merges that make one token"
    assert len(trained.encode("ab" * 150)) <= 100, "one word of 300 octets takes 200 merges"
    assert len(trained.encode("a" * 41)) <= 8, "a run of one byte: rank ties at every step"
    assert 0x7FFFFFFF in trained.ids_to_tokens
    specials = loaded["specials"]
    assert specials.encode("<a>b") == [403] and specials.encode("<<a>") == [401, 97, 62]
    assert specials.encode("<x") == [ord("<"), ord("x")]
    assert specials.encode("é") == [0xE9] and specials.decode([0xE9]) == "é"
    assert loaded["spaced"].encode(" hello") == [500, 259]
    assert specials.encode("    x") == [307, 32, 32, ord("x")], (
        "spaces merge only inside their own word"
    )
    # the law list holds documents both rails must accept, and some that only a reader refuses
    for keep in (
        "merges-empty",
        "merges-missing",
        "added-missing",
        "trailing-space",
        "deep-64",
        "digits-64",
        "exponent-4096",
        "id-minus-zero",
        "surrogate-pair-ignored",
        "escaped-content",
        "merge-pair-form",
        "added-vocab-same-id",
        "added-id-of-byte",
        "long-string",
        "same-key-two-objects",
        "escaped-control",
        "id-top",
    ):
        assert verdicts[keep], keep
    for refuse in (
        "nan-ignored",
        "lone-surrogate-ignored",
        "lone-surrogate-content",
        "deep-65",
        "digits-65",
        "exponent-4097",
        "too-long-string",
        "dup-ignored-key",
        "bom",
    ):
        assert not verdicts[refuse], refuse
    assert sum(verdicts.values()) < len(verdicts) - 60


def test_the_loader_reads_through_the_bounded_reader():
    """What json.loads alone accepts and the shared reader refuses (PORT-2's tightening)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.json")
        body = json.dumps(_mini())[1:-1].encode()
        for doc in (
            b'{"x":NaN,' + body + b"}",
            b'{"x":"\\udfff",' + body + b"}",
            b'{"x":' + b"[" * 64 + b"]" * 64 + b"," + body + b"}",
        ):
            with open(path, "wb") as f:
                f.write(doc)
            json.loads(doc)  # the grammar alone admits it
            try:
                load_tokenizer(path)
            except ValueError:
                continue
            raise AssertionError(f"admitted {doc[:40]!r}")


def test_the_c_tool_tokenizes_as_the_oracle_does():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return  # no compiler or no sources here; the checkout's c-runtime tier and the gate run it
        failures = check_tool(exe, tmp, mutations=96 if os.environ.get("BCIR_THOROUGH") else 24)
        assert not failures, "\n".join(failures[:20])
