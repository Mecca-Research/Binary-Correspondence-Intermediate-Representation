"""Token shards and the batches over them: the native data loader's oracle.

The C subset audit's second native addition (`docs/research/BCIR_C_SUBSET_AUDIT_2026-11-10.md`): the trainer takes
token-id arrays from its caller (`bcir_decoder_run`'s `tokens` and `targets`); this is what fills them, without
Python, from shards on disk. `runtime/c/bcir_tokload.c` is its freestanding twin.

A **shard** (BTKS, version 0) is one token sequence, addressed by the SHA-256 of its payload -- its generation:

    header(64)  magic "BTKS" | version u16 = 0 | width u16 (2 or 4) | vocab u32 | reserved u32 = 0 |
                count u64 | reserved[8] = 0 | sha256[32] of the payload
    payload     count token ids, `width` bytes each, little-endian, every one below vocab

little-endian throughout, the length exactly 64 + count * width, a width of 2 only with a vocab of at most 65,536.
Opening a shard checks all of it, the digest included, before a token is read.

A **loader** over shards s_0 .. s_{k-1} of one vocabulary, with context T, has windows of T + 1 tokens: shard s's
window w starts at token w * T, for w < (len(s) - 1) // T, so a window's inputs are its first T tokens and its
targets its last T, and neighbours share one token, as `SequenceTokenSource` rows do. The windows are numbered in
shard order, N of them. Sample p is window perm_e[p mod N] of epoch e = p // N, where perm_e is the RNG's
Fisher-Yates permutation of [0, N) under the loader's key, stream e, from element 0 (`rng.permutation`) -- or the
identity, unshuffled. Batch b of size B is samples [b * B, (b + 1) * B): a pure function of the index, so a run
resumes exactly.
"""

from __future__ import annotations

import hashlib
import struct

from bcir.frontends.models import rng

MAGIC = b"BTKS"
HEADER = 64
WIDTHS = (2, 4)
_U32 = 0xFFFFFFFF
_U64 = (1 << 64) - 1

#: the refusal codes the C twin returns, by name
OK, ERR_ARG, ERR_FORMAT, ERR_TOKEN, ERR_DIGEST, ERR_EMPTY, ERR_SPACE, ERR_VOCAB = range(8)


class ShardError(ValueError):
    """A refusal, with the code `bcir_tokload.c` returns for it."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def _int(value, what: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ShardError(ERR_ARG, f"{what} must be an integer in [{lo}, {hi}], got {value!r}")
    return value


def encode(tokens, vocab: int, width: int = 4) -> bytes:
    """The shard of `tokens` under `vocab` at `width` bytes a token."""
    if width not in WIDTHS:
        raise ShardError(ERR_ARG, "a token is 2 or 4 bytes wide")
    _int(vocab, "a vocabulary", 1, 1 << 16 if width == 2 else _U32)
    ids = [_int(t, "a token id", 0, vocab - 1) for t in tokens]
    payload = b"".join(t.to_bytes(width, "little") for t in ids)
    head = MAGIC + struct.pack("<HHII", 0, width, vocab, 0) + struct.pack("<Q", len(ids)) + bytes(8)
    return head + hashlib.sha256(payload).digest() + payload


class Shard:
    """An opened shard: its vocabulary, its width, its generation and its tokens."""

    def __init__(self, vocab: int, width: int, digest: bytes, tokens: tuple[int, ...]):
        self.vocab, self.width, self.digest, self.tokens = vocab, width, digest, tokens


def open_shard(data: bytes) -> Shard:
    """Check a shard whole -- header, length, every token, the digest -- or refuse it."""
    if not isinstance(data, (bytes, bytearray)):
        raise ShardError(ERR_ARG, "a shard is bytes")
    if len(data) < HEADER or data[:4] != MAGIC:
        raise ShardError(ERR_FORMAT, "not a BTKS shard")
    version, width, vocab, reserved = struct.unpack_from("<HHII", data, 4)
    (count,) = struct.unpack_from("<Q", data, 16)
    if version != 0 or width not in WIDTHS or reserved or any(data[24:32]):
        raise ShardError(ERR_FORMAT, "an unknown version, width or reserved field")
    if vocab == 0 or (width == 2 and vocab > 1 << 16):
        raise ShardError(ERR_FORMAT, "a vocabulary its width cannot hold")
    if count > (len(data) - HEADER) // width or HEADER + count * width != len(data):
        raise ShardError(ERR_FORMAT, "a length that is not the header's count")
    payload = bytes(data[HEADER:])
    tokens = tuple(
        int.from_bytes(payload[i : i + width], "little") for i in range(0, len(payload), width)
    )
    if any(t >= vocab for t in tokens):
        raise ShardError(ERR_TOKEN, "a token id past the vocabulary")
    digest = bytes(data[32:64])
    if hashlib.sha256(payload).digest() != digest:
        raise ShardError(ERR_DIGEST, "a payload that is not its digest")
    return Shard(vocab, width, digest, tokens)


def shard_windows(count: int, context: int) -> int:
    """The windows of T + 1 tokens a shard of `count` tokens has."""
    return (count - 1) // context if count > context else 0


class Loader:
    """Index-addressed batches over opened shards (see the module docstring)."""

    def __init__(self, shards, context: int, key, shuffle: bool = True):
        shards = list(shards)
        if not shards or not all(isinstance(s, Shard) for s in shards):
            raise ShardError(ERR_ARG, "a loader takes one or more opened shards")
        _int(context, "a context", 1, _U32 - 1)
        if len({s.vocab for s in shards}) != 1:
            raise ShardError(ERR_VOCAB, "shards of different vocabularies")
        self.key = (_int(key[0], "a key word", 0, _U32), _int(key[1], "a key word", 0, _U32))
        self.shards, self.context, self.shuffle = shards, context, bool(shuffle)
        self.starts = []  # (shard, first token) of every window, in shard order
        for s in shards:
            self.starts += [(s, w * context) for w in range(shard_windows(len(s.tokens), context))]
        if not self.starts:
            raise ShardError(ERR_EMPTY, "no shard holds a window")
        if len(self.starts) > _U32:
            raise ShardError(ERR_SPACE, "more windows than a permutation holds")
        self._perm: dict[int, list[int]] = {}

    @property
    def windows(self) -> int:
        return len(self.starts)

    def _order(self, epoch: int) -> list[int]:
        if not self.shuffle:
            return list(range(self.windows))
        if epoch not in self._perm:
            self._perm = {epoch: rng.permutation(self.key, epoch, 0, self.windows)[0]}
        return self._perm[epoch]

    def batch(self, index: int, size: int) -> tuple[list[list[int]], list[list[int]]]:
        """(inputs, targets) of batch `index`: `size` rows of T tokens each."""
        _int(index, "a batch index", 0, _U64)
        _int(size, "a batch size", 1, _U32)
        if (index + 1) * size - 1 > _U64:
            raise ShardError(ERR_ARG, "a batch past the last sample")
        inputs, targets = [], []
        for p in range(index * size, (index + 1) * size):
            epoch, j = divmod(p, self.windows)
            shard, start = self.starts[self._order(epoch)[j]]
            row = shard.tokens[start : start + self.context + 1]
            inputs.append(list(row[:-1]))
            targets.append(list(row[1:]))
        return inputs, targets


__all__ = [
    "ERR_ARG",
    "ERR_DIGEST",
    "ERR_EMPTY",
    "ERR_FORMAT",
    "ERR_SPACE",
    "ERR_TOKEN",
    "ERR_VOCAB",
    "HEADER",
    "Loader",
    "MAGIC",
    "OK",
    "Shard",
    "ShardError",
    "WIDTHS",
    "encode",
    "open_shard",
    "shard_windows",
]
