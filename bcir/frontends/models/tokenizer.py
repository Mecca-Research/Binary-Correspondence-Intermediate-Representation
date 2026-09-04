"""Rung 2 of the open-weight ladder (ML/AI roadmap §7.4): TOKENIZER PARITY, before weights.

A dependency-free (stdlib-only) reference implementation of the byte-level BPE tokenizer family
(the GPT-2/Qwen `tokenizer.json` shape): the byte<->unicode alphabet, rank-based merge encoding,
special tokens (never split), and lossless decode -- so `decode(encode(text)) == text` holds for
ARBITRARY UTF-8 by construction (every byte has a base token). Parity is pinned two ways:
golden-fixture ids over a hand-computed mini `tokenizer.json`, and the round-trip property over
random unicode. The chat-template fixture renders the Gemma-style turn format as a FIXED named
template (a fixture contract, not a Jinja engine -- rendering drift breaks the pinned goldens).
The tokenizer's sha256 digest ties it to `ModelManifest.tokenizer_ref` (rung-1 integrity).

Scope honesty: this is the REFERENCE rail for round-trip + fixture parity. Byte-for-byte parity
against a specific released model's tokenizer (its exact pre-tokenizer regex + normalizer) is
validated when that model's real `tokenizer.json` is ingested -- the loader accepts the real
file shape; the pre-tokenizer here is the simple leading-space word splitter."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field


_DEFAULT_MAX_TOKENIZER_BYTES = 256 * 1024 * 1024


def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte<->unicode alphabet: every byte 0..255 maps to a PRINTABLE unicode char
    (visible ranges pass through; the rest shift above U+0100). Deterministic, self-inverse
    via `unicode_to_bytes`."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_B2U = bytes_to_unicode()
_U2B = {u: b for b, u in _B2U.items()}


@dataclass(frozen=True)
class Tokenizer:
    """A loaded byte-level BPE tokenizer: vocab, merge ranks, specials, and its file digest."""

    vocab: dict  # token string (byte-alphabet space) -> id
    merges: dict  # (left, right) -> rank
    specials: dict  # special token text -> id (never split)
    digest: str = ""  # sha256 of the tokenizer.json bytes (the manifest tie)
    ids_to_tokens: dict = field(default_factory=dict)

    def encode(self, text: str) -> list[int]:
        """Byte-level BPE encode. Specials are matched greedily first (longest wins) and never
        split; everything else maps to the byte alphabet, then merges by ascending rank."""
        out: list[int] = []
        i = 0
        spec = sorted(self.specials, key=len, reverse=True)
        while i < len(text):
            for s in spec:
                if text.startswith(s, i):
                    out.append(self.specials[s])
                    i += len(s)
                    break
            else:
                j = i
                while j < len(text) and not any(text.startswith(s, j) for s in spec):
                    j += 1
                out.extend(self._encode_span(text[i:j]))
                i = j
        return out

    def _encode_span(self, span: str) -> list[int]:
        ids: list[int] = []
        for word in _pretokenize(span):
            sym = [_B2U[b] for b in word.encode("utf-8")]
            while len(sym) > 1:
                best, at = None, -1
                for k in range(len(sym) - 1):
                    r = self.merges.get((sym[k], sym[k + 1]))
                    if r is not None and (best is None or r < best):
                        best, at = r, k
                if best is None:
                    break
                sym[at : at + 2] = [sym[at] + sym[at + 1]]
            ids.extend(self.vocab[s] for s in sym)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Lossless inverse: specials render as their text; every other token's chars map back
        through the byte alphabet and the byte stream decodes as UTF-8."""
        parts: list[str] = []
        buf = bytearray()
        for t in ids:
            if isinstance(t, bool) or not isinstance(t, int) or t not in self.ids_to_tokens:
                raise ValueError(f"unknown tokenizer id {t!r}")
            tok = self.ids_to_tokens[t]
            if tok in self.specials:
                parts.append(bytes(buf).decode("utf-8"))
                buf = bytearray()
                parts.append(tok)
            else:
                buf.extend(_U2B[ch] for ch in tok)
        parts.append(bytes(buf).decode("utf-8"))
        return "".join(parts)


def _pretokenize(text: str) -> list[str]:
    """The reference pre-tokenizer: split into words that keep their LEADING space attached
    (the byte-level BPE convention), so concatenating the words reproduces the text exactly."""
    words: list[str] = []
    cur = ""
    for ch in text:
        if ch == " ":
            if cur and not cur.endswith(" "):
                words.append(cur)
                cur = ""
            cur += ch
        else:
            if cur.endswith(" ") and len(cur) > 1:
                words.append(cur[:-1])
                cur = " "
            cur += ch
    if cur:
        words.append(cur)
    return words


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _token_id(value, *, path: str, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0x7FFFFFFF:
        raise ValueError(f"{path}: {where} must be an integer token id in [0, 2^31-1]")
    return value


def load_tokenizer(path: str, *, max_bytes: int = _DEFAULT_MAX_TOKENIZER_BYTES) -> Tokenizer:
    """Load a HF-shaped `tokenizer.json` (model.type == "BPE"): vocab, merges (either the
    "a b" string form or the [a, b] pair form), and added special tokens."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    try:
        with open(path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            if size > max_bytes:
                raise ValueError(f"{path}: tokenizer.json exceeds {max_bytes} bytes")
            raw = f.read(max_bytes + 1)
    except OSError as exc:
        raise ValueError(f"{path}: cannot read tokenizer.json: {exc}") from exc
    if len(raw) > max_bytes:
        raise ValueError(f"{path}: tokenizer.json exceeds {max_bytes} bytes")
    try:
        d = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{path}: malformed tokenizer.json: {exc}") from exc
    if not isinstance(d, dict):
        raise ValueError(f"{path}: tokenizer.json root must be an object")
    model = d.get("model", {})
    if not isinstance(model, dict):
        raise ValueError(f"{path}: tokenizer model must be an object")
    if model.get("type") != "BPE":
        raise ValueError(f"{path}: unsupported tokenizer model {model.get('type')!r} (BPE only)")
    raw_vocab = model.get("vocab", {})
    if not isinstance(raw_vocab, dict):
        raise ValueError(f"{path}: tokenizer vocab must be an object")
    vocab: dict[str, int] = {}
    ids: dict[int, str] = {}
    for token, raw_id in raw_vocab.items():
        if not isinstance(token, str) or not token:
            raise ValueError(f"{path}: vocab tokens must be nonempty strings")
        token_id = _token_id(raw_id, path=path, where=f"vocab id for {token!r}")
        if token_id in ids:
            raise ValueError(
                f"{path}: duplicate token id {token_id} for {ids[token_id]!r} and {token!r}"
            )
        if any(ch not in _U2B for ch in token):
            raise ValueError(f"{path}: vocab token {token!r} is outside the byte alphabet")
        vocab[token] = token_id
        ids[token_id] = token
    missing_bytes = [byte for byte, token in _B2U.items() if token not in vocab]
    if missing_bytes:
        raise ValueError(
            f"{path}: byte-level BPE vocab is missing {len(missing_bytes)} base byte tokens"
        )

    raw_merges = model.get("merges", [])
    if not isinstance(raw_merges, list):
        raise ValueError(f"{path}: tokenizer merges must be an array")
    merges: dict = {}
    for rank, mg in enumerate(raw_merges):
        pair = (
            tuple(mg.split(" "))
            if isinstance(mg, str)
            else tuple(mg)
            if isinstance(mg, list)
            else ()
        )
        if len(pair) != 2 or not all(isinstance(part, str) and part for part in pair):
            raise ValueError(f"{path}: malformed merge {mg!r}")
        if pair in merges:
            raise ValueError(f"{path}: duplicate merge {pair!r}")
        if pair[0] not in vocab or pair[1] not in vocab or pair[0] + pair[1] not in vocab:
            raise ValueError(f"{path}: merge {pair!r} references a missing vocab symbol")
        merges[pair] = rank

    raw_specials = d.get("added_tokens", [])
    if not isinstance(raw_specials, list):
        raise ValueError(f"{path}: added_tokens must be an array")
    specials: dict[str, int] = {}
    for index, item in enumerate(raw_specials):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("content"), str)
            or not item["content"]
        ):
            raise ValueError(f"{path}: added token {index} must have nonempty string content")
        content = item["content"]
        token_id = _token_id(item.get("id"), path=path, where=f"added token {index} id")
        if content in specials:
            raise ValueError(f"{path}: duplicate added token {content!r}")
        if content in vocab and vocab[content] != token_id:
            raise ValueError(f"{path}: added token {content!r} conflicts with its vocab id")
        if token_id in ids and ids[token_id] != content:
            raise ValueError(f"{path}: added token id {token_id} conflicts with {ids[token_id]!r}")
        specials[content] = token_id
        ids[token_id] = content
    return Tokenizer(
        vocab=vocab,
        merges=merges,
        specials=specials,
        digest=hashlib.sha256(raw).hexdigest(),
        ids_to_tokens=ids,
    )


# --- the chat-template fixture (a fixed named contract, not a template engine) ------------------

_GEMMA_TURN = "<start_of_turn>{role}\n{text}<end_of_turn>\n"


def render_chat(
    turns: list[tuple[str, str]], template: str = "gemma", add_generation_prompt: bool = True
) -> str:
    """Render a conversation under a NAMED chat template. `turns` is [(role, text), ...].
    "gemma": <start_of_turn>{role}\\n{text}<end_of_turn>\\n per turn, then the generation
    prompt opens a model turn. A fixture contract: the goldens in `test_model_tokenizer.py`
    pin the exact rendering -- drift breaks them loudly."""
    if template != "gemma":
        raise ValueError(f"unknown chat template {template!r}")
    out = "<bos>" + "".join(_GEMMA_TURN.format(role=r, text=t) for r, t in turns)
    if add_generation_prompt:
        out += "<start_of_turn>model\n"
    return out
