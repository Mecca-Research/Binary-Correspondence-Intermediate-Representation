"""Rung 2 of the open-weight ladder, the SENTENCEPIECE half: `tokenizer.model` decoded
dep-free (a minimal protobuf wire reader -- the fields SentencePiece actually writes),
and the score-based BPE encoder the Llama family ships.

The GPT-2/Qwen byte-level BPE (`tokenizer.py`) merges by RANK; SentencePiece-BPE merges
by SCORE: split the normalized text into unicode characters, then repeatedly merge the
adjacent pair whose concatenation is a vocabulary piece with the HIGHEST score (leftmost
wins ties) until no merge applies -- the llama.cpp `llm_tokenizer_spm` algorithm.
Normalization is the SentencePiece default the Llama family uses: spaces become the
LOW LINE block `▁`, with one dummy prefix. A symbol that never merged into a known
piece falls back to its UTF-8 bytes through the `<0xXX>` BYTE pieces (type 6), so
`decode(encode(text)) == text` holds for arbitrary UTF-8 whenever the model carries the
byte alphabet; without byte pieces the unknown maps to `<unk>` (type 2) -- recorded, not
hidden. The file digest ties the tokenizer to `ModelManifest.tokenizer_ref` (rung 1).

Protobuf scope honesty: `_read_pieces` walks the ModelProto wire format for field 1
(the repeated `SentencePiece {piece=1: string, score=2: float, type=3: enum}` messages)
and SKIPS every other field by wire type -- enough to load any real `tokenizer.model`,
not a general protobuf library. Dep-free stdlib; cost-side (imports no verifier)."""

from __future__ import annotations

import hashlib
import math
import os
import struct
from dataclasses import dataclass, field

_SPACE = "▁"  # the SentencePiece whitespace piece
_TYPE_NORMAL, _TYPE_UNKNOWN, _TYPE_CONTROL, _TYPE_BYTE = 1, 2, 3, 6
_MAX_MODEL_BYTES = 256 * 1024 * 1024


def _read_varint(buf: bytes, i: int, end: int | None = None) -> tuple[int, int]:
    """Read one bounded protobuf uint64 varint with clean truncation/overflow errors."""
    end = len(buf) if end is None else end
    value = 0
    for byte_index in range(10):
        if i >= end:
            raise ValueError("tokenizer.model: truncated protobuf varint")
        b = buf[i]
        i += 1
        if byte_index == 9 and b > 1:
            raise ValueError("tokenizer.model: protobuf varint exceeds uint64")
        value |= (b & 0x7F) << (7 * byte_index)
        if not b & 0x80:
            return value, i
    raise ValueError("tokenizer.model: overlong protobuf varint")


def _checked_end(buf: bytes, i: int, n: int, end: int) -> int:
    if n < 0 or i < 0 or i > end or n > end - i or end > len(buf):
        raise ValueError("tokenizer.model: truncated protobuf field")
    return i + n


def _skip(buf: bytes, i: int, wire: int, end: int | None = None) -> int:
    """Skip one field's payload by wire type (varint / 64-bit / length-delimited / 32-bit)."""
    end = len(buf) if end is None else end
    if wire == 0:
        return _read_varint(buf, i, end)[1]
    if wire == 1:
        return _checked_end(buf, i, 8, end)
    if wire == 2:
        n, i = _read_varint(buf, i, end)
        return _checked_end(buf, i, n, end)
    if wire == 5:
        return _checked_end(buf, i, 4, end)
    raise ValueError(f"tokenizer.model: unsupported protobuf wire type {wire}")


def _read_pieces(buf: bytes) -> list[tuple[str, float, int]]:
    """Every `pieces` entry of a ModelProto: (piece, score, type)."""
    out: list[tuple[str, float, int]] = []
    i = 0
    while i < len(buf):
        tag, i = _read_varint(buf, i, len(buf))
        fieldno, wire = tag >> 3, tag & 7
        if fieldno == 0:
            raise ValueError("tokenizer.model: protobuf field number zero")
        if fieldno != 1 or wire != 2:  # not a `pieces` message: skip it
            i = _skip(buf, i, wire, len(buf))
            continue
        n, i = _read_varint(buf, i, len(buf))
        msg_end = _checked_end(buf, i, n, len(buf))
        msg, i = buf[i:msg_end], msg_end
        piece, score, ptype = "", 0.0, _TYPE_NORMAL
        seen: set[int] = set()
        j = 0
        while j < len(msg):
            t, j = _read_varint(msg, j, len(msg))
            f, w = t >> 3, t & 7
            if f == 0:
                raise ValueError("tokenizer.model: protobuf field number zero in piece")
            if f == 1 and w == 2:
                if f in seen:
                    raise ValueError("tokenizer.model: duplicate piece text field")
                ln, j = _read_varint(msg, j, len(msg))
                field_end = _checked_end(msg, j, ln, len(msg))
                try:
                    piece = msg[j:field_end].decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError("tokenizer.model: piece text is not UTF-8") from exc
                j = field_end
                seen.add(f)
            elif f == 2 and w == 5:
                if f in seen:
                    raise ValueError("tokenizer.model: duplicate piece score field")
                field_end = _checked_end(msg, j, 4, len(msg))
                (score,) = struct.unpack("<f", msg[j:field_end])
                j = field_end
                seen.add(f)
            elif f == 3 and w == 0:
                if f in seen:
                    raise ValueError("tokenizer.model: duplicate piece type field")
                ptype, j = _read_varint(msg, j, len(msg))
                seen.add(f)
            else:
                j = _skip(msg, j, w, len(msg))
        if 1 not in seen:
            raise ValueError("tokenizer.model: piece entry has no text field")
        if not math.isfinite(score):
            raise ValueError(f"tokenizer.model: piece {piece!r} has a non-finite score")
        if not 1 <= ptype <= 6:
            raise ValueError(f"tokenizer.model: piece {piece!r} has unknown type {ptype}")
        out.append((piece, score, ptype))
    return out


@dataclass(frozen=True)
class SpTokenizer:
    """A loaded SentencePiece-BPE model: piece -> (id, score), the byte alphabet, the
    specials (control/unknown pieces -- never produced by merging), and the file digest."""

    pieces: dict  # piece string -> (id, score)
    byte_ids: dict  # byte value 0..255 -> id (the <0xXX> pieces; may be empty)
    unk_id: int
    specials: dict  # control-piece text -> id (e.g. <s>, </s>)
    digest: str = ""
    ids_to_pieces: dict = field(default_factory=dict)

    @property
    def n_pieces(self) -> int:
        return len(self.ids_to_pieces)

    def encode(self, text: str, *, add_dummy_prefix: bool = True) -> list[int]:
        """Score-based BPE over the normalized text (spaces -> `▁`, one dummy prefix):
        merge the best-scoring adjacent pair until fixpoint, then byte-fall-back any symbol
        that is not a piece. Deterministic (leftmost wins a score tie)."""
        text = text.replace(" ", _SPACE)
        if add_dummy_prefix and not text.startswith(_SPACE):
            text = _SPACE + text
        syms = list(text)
        while len(syms) > 1:  # the llama.cpp spm merge loop
            best_i, best_score = -1, None
            for i in range(len(syms) - 1):
                cand = self.pieces.get(syms[i] + syms[i + 1])
                if cand is not None and (best_score is None or cand[1] > best_score):
                    best_i, best_score = i, cand[1]
            if best_i < 0:
                break
            syms[best_i : best_i + 2] = [syms[best_i] + syms[best_i + 1]]
        out: list[int] = []
        for s in syms:
            hit = self.pieces.get(s)
            if hit is not None:
                out.append(hit[0])
            elif self.byte_ids:  # byte fallback: the UTF-8 bytes
                out.extend(self.byte_ids[b] for b in s.encode("utf-8"))
            else:
                out.append(self.unk_id)  # no byte alphabet: honest <unk>
        return out

    def decode(self, ids: list) -> str:
        """Pieces concatenated; `▁` -> space; `<0xXX>` byte pieces -> raw UTF-8 bytes
        (errors='replace' -- a torn byte sequence never crashes the decode); the dummy
        prefix space is stripped."""
        buf: list[bytes] = []
        for t in ids:
            piece = self.ids_to_pieces.get(int(t), "")
            if piece in self.specials:
                continue  # control pieces render as nothing
            if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
                buf.append(bytes([int(piece[3:5], 16)]))
            else:
                buf.append(piece.replace(_SPACE, " ").encode("utf-8"))
        text = b"".join(buf).decode("utf-8", errors="replace")
        return text[1:] if text.startswith(" ") else text


def load_sentencepiece(path: str, *, max_bytes: int = _MAX_MODEL_BYTES) -> SpTokenizer:
    """`tokenizer.model` -> SpTokenizer. Ids are the piece order (the SentencePiece id
    law); the digest is the file's sha256 (the rung-1 manifest tie)."""
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        if size > max_bytes:
            raise ValueError(f"{path}: tokenizer.model is {size} bytes; limit is {max_bytes}")
        raw = f.read(max_bytes + 1)
        if len(raw) != size:
            raise ValueError(f"{path}: tokenizer.model changed or truncated while reading")
    entries = _read_pieces(raw)
    if not entries:
        raise ValueError(f"{path}: no SentencePiece pieces (not a tokenizer.model?)")
    pieces: dict = {}
    byte_ids: dict = {}
    specials: dict = {}
    ids_to_pieces: dict = {}
    unk_id: int | None = None
    seen_text: set[str] = set()
    for pid, (piece, score, ptype) in enumerate(entries):
        if piece in seen_text:
            raise ValueError(f"{path}: duplicate SentencePiece text {piece!r}")
        seen_text.add(piece)
        ids_to_pieces[pid] = piece
        if ptype == _TYPE_BYTE:
            if not (len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">")):
                raise ValueError(f"{path}: malformed byte piece {piece!r}")
            try:
                byte = int(piece[3:5], 16)
            except ValueError as exc:
                raise ValueError(f"{path}: malformed byte piece {piece!r}") from exc
            if byte in byte_ids:
                raise ValueError(f"{path}: duplicate byte fallback piece 0x{byte:02X}")
            byte_ids[byte] = pid
        elif ptype == _TYPE_UNKNOWN:
            if unk_id is not None:
                raise ValueError(f"{path}: multiple unknown pieces")
            unk_id = pid
            specials[piece] = pid
        elif ptype == _TYPE_CONTROL:
            specials[piece] = pid
        else:
            pieces[piece] = (pid, score)
    if unk_id is None:
        raise ValueError(f"{path}: tokenizer.model has no unknown piece")
    if byte_ids and set(byte_ids) != set(range(256)):
        raise ValueError(f"{path}: partial byte fallback alphabet ({len(byte_ids)}/256)")
    return SpTokenizer(
        pieces=pieces,
        byte_ids=byte_ids,
        unk_id=unk_id,
        specials=specials,
        digest=hashlib.sha256(raw).hexdigest(),
        ids_to_pieces=ids_to_pieces,
    )
