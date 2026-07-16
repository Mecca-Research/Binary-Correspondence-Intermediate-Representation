"""Rung 2's SentencePiece half (`frontends/models/spm.py`): `tokenizer.model` dep-free.

A synthetic `tokenizer.model` written byte-for-byte in the protobuf wire format loads
through the minimal wire reader; score-based merges walk the EXISTING-intermediate-piece
chains (the real SP-BPE law -- a trained model carries the full merge chain); byte
fallback carries arbitrary UTF-8 (`decode(encode(x)) == x`); control pieces never leak
into decode output; unknown-field skipping is exercised by a trainer_spec-shaped blob.
The file digest ties to `ModelManifest.tokenizer_ref` (rung 1)."""

import os
import struct
import tempfile

from bcir.frontends.models.spm import SpTokenizer, load_sentencepiece

def _sp_varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _sp_piece(piece: str, score: float, ptype: int) -> bytes:
    pb = piece.encode("utf-8")
    msg = (b"\x0a" + _sp_varint(len(pb)) + pb          # field 1 (piece), wire 2
           + b"\x15" + struct.pack("<f", score)        # field 2 (score), wire 5
           + b"\x18" + _sp_varint(ptype))              # field 3 (type), wire 0
    return b"\x0a" + _sp_varint(len(msg)) + msg        # ModelProto field 1, wire 2


def _write_sp_model(path: str) -> None:
    """A REAL-format tokenizer.model: unk + two controls, the byte alphabet, and scored
    word/subword pieces (higher score merges first -- '▁hello' beats 'he'+'ll'+'o')."""
    parts = [_sp_piece("<unk>", 0.0, 2), _sp_piece("<s>", 0.0, 3), _sp_piece("</s>", 0.0, 3)]
    for b in range(256):                               # ids 3..258: the byte alphabet
        parts.append(_sp_piece(f"<0x{b:02X}>", 0.0, 6))
    # SP-BPE merges only through EXISTING intermediate pieces (the real law: a trained
    # model carries the full merge chain) -- so the chains ▁h..▁hello and ▁w..▁world are
    # all present, scores ascending toward the full word.
    scored = [("▁", -1.0), ("▁h", -1.9), ("▁he", -1.8), ("▁hel", -1.7), ("▁hell", -1.6),
              ("▁hello", -0.5), ("▁w", -1.95), ("▁wo", -1.85), ("▁wor", -1.75),
              ("▁worl", -1.65), ("▁world", -0.6), ("he", -2.0), ("ll", -2.1),
              ("o", -3.0), ("l", -3.1), ("h", -3.2), ("e", -3.3), ("w", -3.4), ("r", -3.5),
              ("d", -3.6), ("el", -2.2)]
    for piece, score in scored:
        parts.append(_sp_piece(piece, score, 1))
    # a trainer_spec-shaped OTHER field the reader must SKIP (field 2, wire 2)
    parts.append(b"\x12" + _sp_varint(3) + b"abc")
    with open(path, "wb") as f:
        f.write(b"".join(parts))


def test_sentencepiece_model_loads_and_merges_by_score():
    """The dep-free protobuf reader + the score-merge law: '▁hello' (score -0.5) wins over
    any subword split; control pieces are specials; the digest is the file sha."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tokenizer.model")
        _write_sp_model(path)
        tok = load_sentencepiece(path)
    assert tok.n_pieces == 3 + 256 + 21
    assert tok.unk_id == 0 and tok.specials["<s>"] == 1 and tok.specials["</s>"] == 2
    assert len(tok.byte_ids) == 256 and tok.digest
    ids = tok.encode("hello world")
    assert ids == [tok.pieces["▁hello"][0], tok.pieces["▁world"][0]], ids
    assert tok.decode(ids) == "hello world"
    # merge order is by SCORE, not length: '▁h' + 'el' + 'l' + 'o' assembles 'hell' pieces
    # for a word the vocab lacks, and byte fallback never fires while pieces cover it.
    ids2 = tok.encode("hell")
    assert all(i >= 3 + 256 for i in ids2)             # scored pieces, no bytes, no unk
    assert tok.decode(ids2) == "hell"


def test_sentencepiece_round_trips_arbitrary_utf8_via_byte_fallback():
    """decode(encode(x)) == x for text far outside the scored vocab -- the <0xXX> byte
    alphabet carries anything (the llama.cpp byte_fallback contract)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tokenizer.model")
        _write_sp_model(path)
        tok = load_sentencepiece(path)
    for text in ("héllo wörld", "日本語のテキスト", "emoji 🚀 mixed", "tabs\tand\nnewlines"):
        assert tok.decode(tok.encode(text)) == text, text
    # and controls never leak into decode output.
    assert tok.decode([1, tok.pieces["▁hello"][0], 2]) == "hello"


def test_malformed_protobuf_is_always_a_clean_value_error():
    """Truncated/overlong varints and torn length/fixed32 fields never escape as an
    IndexError or struct.error from the tokenizer trust boundary."""
    malformed = (
        b"\x80",                              # unterminated outer tag varint
        b"\x80" * 11,                         # varint longer than uint64
        b"\x0a\x05\x0a\x01x",                # outer message length escapes file
        b"\x0a\x03\x15\x00\x00",             # torn fixed32 score
        b"\x0a\x03\x0a\x05x",                # piece string length escapes message
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.model")
        for raw in malformed:
            with open(path, "wb") as f:
                f.write(raw)
            try:
                load_sentencepiece(path)
                raise AssertionError(f"malformed protobuf was accepted: {raw!r}")
            except ValueError as exc:
                assert "tokenizer.model" in str(exc)


def test_partial_byte_alphabet_and_duplicate_piece_text_are_rejected():
    """The encoder assumes byte fallback is either absent or complete; a partial table
    would otherwise turn ordinary UTF-8 into a mid-request KeyError."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.model")
        variants = (
            _sp_piece("<unk>", 0.0, 2) + _sp_piece("<0x00>", 0.0, 6),
            _sp_piece("<unk>", 0.0, 2) + _sp_piece("dup", 0.0, 1)
            + _sp_piece("dup", -1.0, 1),
        )
        for raw in variants:
            with open(path, "wb") as f:
                f.write(raw)
            try:
                load_sentencepiece(path)
                raise AssertionError("ambiguous SentencePiece model was accepted")
            except ValueError:
                pass


def test_a_real_released_tokenizer_loads_when_present():
    """ASSET-GATED (self-skips): point BCIR_HF_MODEL_DIR at a checkpoint directory whose
    `tokenizer.model` is a real SentencePiece file (e.g. Maykeye/TinyLLama-v0) -- the
    wire reader loads it, encode/decode round-trips real text, and every id is in range."""
    model_dir = os.environ.get("BCIR_HF_MODEL_DIR", "")
    path = os.path.join(model_dir, "tokenizer.model") if model_dir else ""
    if not path or not os.path.isfile(path):
        return                                          # the asset gate (the rig pattern)
    tok = load_sentencepiece(path)
    assert tok.n_pieces > 256 and tok.byte_ids          # a real model carries the alphabet
    for text in ("Once upon a time", "hello world", "byte-fallback: 日本語 🚀"):
        ids = tok.encode(text)
        assert ids and all(0 <= i < tok.n_pieces for i in ids)
        assert tok.decode(ids) == text, text


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    tests = sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))
    passed = failed = 0
    for name in tests:
        try:
            getattr(mod, name)()
            passed += 1
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
