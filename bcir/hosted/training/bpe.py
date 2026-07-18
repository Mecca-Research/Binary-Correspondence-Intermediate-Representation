"""A deterministic byte-fallback BPE reference for bounded BCIR corpus tests.

Production tokenizer training may use pinned SentencePiece, but its exported model must
match this dependency-free normalization/byte-fallback contract.  This implementation is
intentionally compact and exact; it is suitable for micro-models, fixtures, and parity
checks rather than pretending to be a trillion-token tokenizer trainer.
"""
from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass

from ..._artifact_json import strict_json_loads
from ..models.spec import CorpusManifest, SequenceTokenSource
from .contracts import canonical_json, require_int, sha256_text
from .data import PreparedCorpus


_SPECIALS = ("<pad>", "<bos>", "<eos>", "<unk>")
_BASE_VOCAB = len(_SPECIALS) + 256


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def _merge_sequence(sequence: list[int], pair: tuple[int, int], token: int) -> list[int]:
    output = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
            output.append(token); index += 2
        else:
            output.append(sequence[index]); index += 1
    return output


@dataclass(frozen=True)
class BytePairTokenizer:
    merges: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        expected = _BASE_VOCAB
        known = set(range(_BASE_VOCAB))
        for index, row in enumerate(self.merges):
            if not isinstance(row, (tuple, list)) or len(row) != 3 \
                    or any(type(value) is not int for value in row):
                raise ValueError(f"merge {index} must contain three integer token IDs")
            left, right, token = row
            if left not in known or right not in known or token != expected:
                raise ValueError(f"merge {index} is not canonical or references a future token")
            known.add(token); expected += 1

    @property
    def vocab_size(self) -> int:
        return _BASE_VOCAB + len(self.merges)

    @property
    def bos_token_id(self) -> int:
        return 1

    @property
    def eos_token_id(self) -> int:
        return 2

    @property
    def pad_token_id(self) -> int:
        return 0

    def _token_bytes(self) -> dict[int, bytes]:
        table = {index + len(_SPECIALS): bytes([index]) for index in range(256)}
        for left, right, token in self.merges:
            table[token] = table[left] + table[right]
        return table

    @classmethod
    def train(cls, texts, *, vocab_size: int, min_frequency: int = 2) -> "BytePairTokenizer":
        require_int(vocab_size, "vocab_size", minimum=_BASE_VOCAB, maximum=65535)
        require_int(min_frequency, "min_frequency", minimum=1)
        if not isinstance(texts, (tuple, list)) or not texts \
                or any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("texts must be a nonempty sequence of nonempty strings")
        sequences = [[byte + len(_SPECIALS) for byte in _normalize(text).encode("utf-8")]
                     for text in texts]
        token_bytes = {index + len(_SPECIALS): bytes([index]) for index in range(256)}
        merges = []
        while _BASE_VOCAB + len(merges) < vocab_size:
            counts: dict[tuple[int, int], int] = {}
            for sequence in sequences:
                for pair in zip(sequence, sequence[1:]):
                    counts[pair] = counts.get(pair, 0) + 1
            eligible = [pair for pair, count in counts.items() if count >= min_frequency]
            if not eligible:
                break
            pair = min(eligible, key=lambda item: (
                -counts[item], token_bytes[item[0]] + token_bytes[item[1]], item))
            token = _BASE_VOCAB + len(merges)
            token_bytes[token] = token_bytes[pair[0]] + token_bytes[pair[1]]
            merges.append((pair[0], pair[1], token))
            sequences = [_merge_sequence(sequence, pair, token) for sequence in sequences]
        return cls(tuple(merges))

    def encode(self, text: str, *, add_bos: bool = False,
               add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        sequence = [byte + len(_SPECIALS) for byte in _normalize(text).encode("utf-8")]
        for left, right, token in self.merges:
            sequence = _merge_sequence(sequence, (left, right), token)
        if add_bos:
            sequence.insert(0, self.bos_token_id)
        if add_eos:
            sequence.append(self.eos_token_id)
        return sequence

    def decode(self, token_ids) -> str:
        if not isinstance(token_ids, (tuple, list)) \
                or any(type(token) is not int or not 0 <= token < self.vocab_size
                       for token in token_ids):
            raise ValueError("token_ids must contain in-vocabulary integers")
        table = self._token_bytes()
        raw = bytearray()
        for token in token_ids:
            if token in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
                continue
            if token == 3:
                raise ValueError("the byte-fallback tokenizer never emits <unk>")
            raw.extend(table[token])
        return bytes(raw).decode("utf-8")

    def to_dict(self) -> dict:
        return {"schema": "bcir.byte_bpe.v1", "normalization": "NFC+LF",
                "special_tokens": list(_SPECIALS),
                "merges": [list(row) for row in self.merges]}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        return sha256_text(self.to_json())

    @classmethod
    def from_json(cls, text: str) -> "BytePairTokenizer":
        value = strict_json_loads(
            text, "byte BPE tokenizer", max_bytes=16 * 1024 * 1024)
        if not isinstance(value, dict) or set(value) != {
                "schema", "normalization", "special_tokens", "merges"} \
                or value["schema"] != "bcir.byte_bpe.v1" \
                or value["normalization"] != "NFC+LF" \
                or value["special_tokens"] != list(_SPECIALS):
            raise ValueError("unsupported or malformed tokenizer contract")
        tokenizer = cls(tuple(tuple(row) if isinstance(row, list) else row
                              for row in value["merges"]))
        if tokenizer.to_json() != text:
            raise ValueError("tokenizer JSON is not canonical")
        return tokenizer


def token_source_from_corpus(corpus: PreparedCorpus, tokenizer: BytePairTokenizer, *,
                             split: str = "train") -> SequenceTokenSource:
    if not isinstance(corpus, PreparedCorpus) or not isinstance(tokenizer, BytePairTokenizer):
        raise ValueError("corpus and tokenizer have invalid types")
    documents = corpus.split(split)
    if not documents:
        raise ValueError(f"prepared corpus has no {split} documents")
    token_ids = []
    for document in documents:
        token_ids.extend(tokenizer.encode(document.text, add_bos=True, add_eos=True))
    tokenizer_sha = hashlib.sha256((tokenizer.to_json() + "\n").encode("utf-8")).hexdigest()
    manifest = CorpusManifest(
        name="bcir-prepared-corpus", revision=corpus.digest, split=split,
        license=" AND ".join(sorted({document.license for document in documents})),
        source_sha256=corpus.digest, tokenizer_sha256=tokenizer_sha,
        token_count=len(token_ids), vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id)
    return SequenceTokenSource(token_ids, manifest)
