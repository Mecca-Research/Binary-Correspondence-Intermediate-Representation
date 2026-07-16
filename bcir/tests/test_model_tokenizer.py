"""Rung 2 of the open-weight ladder: TOKENIZER PARITY fixtures (`frontends/models/tokenizer.py`).

The dep-free byte-level BPE reference: golden ids over a hand-computed mini `tokenizer.json`,
the LOSSLESS round-trip property over arbitrary unicode (guaranteed by the 256-byte base
alphabet), specials never split, the Gemma-style chat-template fixture with a pinned golden
rendering, and the digest tie into `ModelManifest.tokenizer_digest` (rung-1 integrity)."""

import json
import os
import random
import tempfile

from bcir.frontends.models import bytes_to_unicode, load_tokenizer, render_chat
from bcir.frontends.models.tokenizer import _pretokenize

_B2U = bytes_to_unicode()


def _mini_tokenizer(d):
    """A hand-computable tokenizer.json: the full 256-char byte alphabet (id == byte), four
    merges building "hello", and four specials."""
    vocab = {_B2U[b]: b for b in range(256)}
    for tok, i in (("he", 256), ("ll", 257), ("hell", 258), ("hello", 259)):
        vocab[tok] = i
    doc = {"model": {"type": "BPE", "vocab": vocab,
                     "merges": ["h e", "l l", "he ll", "hell o"]},
           "added_tokens": [{"content": "<bos>", "id": 260}, {"content": "<eos>", "id": 261},
                            {"content": "<start_of_turn>", "id": 262},
                            {"content": "<end_of_turn>", "id": 263}]}
    path = os.path.join(d, "tokenizer.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return path


def test_golden_ids_over_the_mini_fixture():
    with tempfile.TemporaryDirectory() as d:
        tok = load_tokenizer(_mini_tokenizer(d))
        assert tok.encode("hello") == [259]                        # the full merge chain
        assert tok.encode(" hello") == [32, 259]                   # leading space = byte token 32
        assert tok.encode("hell") == [258]
        assert tok.encode("helo") == [256, 108, 111]               # he + l + o (no ll pair)
        assert tok.encode("<bos>hello<eos>") == [260, 259, 261]    # specials never split


def test_round_trip_is_lossless_over_arbitrary_unicode():
    rng = random.Random(0x70C1)
    pool = "hello world Größe 東京 🚀 tab\t nl\n <bos> mixed"
    with tempfile.TemporaryDirectory() as d:
        tok = load_tokenizer(_mini_tokenizer(d))
        for _ in range(50):
            s = "".join(rng.choice(pool) for _ in range(rng.randint(0, 40)))
            assert tok.decode(tok.encode(s)) == s, repr(s)


def test_pretokenizer_partition_reproduces_the_text():
    for s in ("a b  c", " lead", "trail ", "", "   ", "no-spaces"):
        assert "".join(_pretokenize(s)) == s, repr(s)


def test_chat_template_golden_and_round_trip():
    turns = [("user", "What is BCIR?"), ("model", "A correspondence IR."), ("user", "Thanks")]
    got = render_chat(turns)
    want = ("<bos><start_of_turn>user\nWhat is BCIR?<end_of_turn>\n"
            "<start_of_turn>model\nA correspondence IR.<end_of_turn>\n"
            "<start_of_turn>user\nThanks<end_of_turn>\n"
            "<start_of_turn>model\n")
    assert got == want                                             # the pinned fixture contract
    with tempfile.TemporaryDirectory() as d:
        tok = load_tokenizer(_mini_tokenizer(d))
        ids = tok.encode(got)
        assert tok.decode(ids) == got                              # template survives the rail
        assert ids.count(262) == 4 and ids[0] == 260               # turn markers ride as specials


def test_tokenizer_digest_ties_into_the_manifest():
    import struct
    from bcir.frontends.models import build_manifest
    with tempfile.TemporaryDirectory() as d:
        tok_path = _mini_tokenizer(d)
        tok = load_tokenizer(tok_path)
        shard = os.path.join(d, "model.safetensors")
        blob = json.dumps({"w": {"dtype": "F32", "shape": [2, 2],
                                 "data_offsets": [0, 16]}}).encode()
        with open(shard, "wb") as f:
            f.write(struct.pack("<Q", len(blob)) + blob + b"\x00" * 16)
        m = build_manifest([shard], {"model_type": "toy", "vocab_size": 264},
                           tokenizer_ref="tokenizer.json", tokenizer_path=tok_path)
        assert m.tokenizer_digest == tok.digest and len(tok.digest) == 64
        # rung-2 parity check: a tokenizer whose bytes do not hash to the manifest's record
        # is the WRONG tokenizer for this model.
        with open(tok_path, "ab") as f:
            f.write(b" ")
        assert load_tokenizer(tok_path).digest != m.tokenizer_digest


def test_malformed_tokenizer_structures_fail_closed():
    with tempfile.TemporaryDirectory() as d:
        valid_path = _mini_tokenizer(d)
        with open(valid_path, encoding="utf-8") as f:
            valid = json.load(f)

        variants = []
        missing_byte = json.loads(json.dumps(valid))
        del missing_byte["model"]["vocab"][_B2U[0]]
        variants.append(missing_byte)
        duplicate_id = json.loads(json.dumps(valid))
        duplicate_id["model"]["vocab"][_B2U[1]] = duplicate_id["model"]["vocab"][_B2U[0]]
        variants.append(duplicate_id)
        empty_special = json.loads(json.dumps(valid))
        empty_special["added_tokens"].append({"content": "", "id": 300})
        variants.append(empty_special)
        duplicate_merge = json.loads(json.dumps(valid))
        duplicate_merge["model"]["merges"].append("h e")
        variants.append(duplicate_merge)

        for index, doc in enumerate(variants):
            path = os.path.join(d, f"invalid-{index}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f)
            try:
                load_tokenizer(path)
                raise AssertionError(f"invalid tokenizer {index} must be rejected")
            except ValueError:
                pass

        duplicate_json = os.path.join(d, "duplicate-key.json")
        with open(duplicate_json, "w", encoding="utf-8") as f:
            f.write('{"model":{"type":"BPE","type":"BPE","vocab":{},"merges":[]}}')
        try:
            load_tokenizer(duplicate_json)
            raise AssertionError("duplicate JSON keys must be rejected")
        except ValueError:
            pass

        tok = load_tokenizer(valid_path)
        try:
            tok.decode([999999])
            raise AssertionError("unknown ids must be rejected cleanly")
        except ValueError:
            pass


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
