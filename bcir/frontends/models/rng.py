"""A deterministic counter-based random-number generator: Philox4x32-10 and the streams over it.

The C subset audit's first native addition (`docs/research/BCIR_C_SUBSET_AUDIT_2026-11-10.md`): a key plus a
counter and no hidden state, so element `i` of a stream is one number whatever thread, batch or order asks for it.
This is the oracle; `runtime/c/bcir_rng.c` is its freestanding twin, and both are held to the generator's published
known-answer vectors (`KNOWN_ANSWERS`, from Random123: J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw,
"Parallel Random Numbers: As Easy as 1, 2, 3", SC11) and to each other.

A stream is a key (two 32-bit words) and a 64-bit stream id. Element `index` of it is lane `index & 3` of the
Philox block whose counter is `(lo32(index >> 2), hi32(index >> 2), lo32(stream), hi32(stream))`.
"""

from __future__ import annotations

_M0, _M1 = 0xD2511F53, 0xCD9E8D57
_W0, _W1 = 0x9E3779B9, 0xBB67AE85
_U32 = 0xFFFFFFFF
_U64 = (1 << 64) - 1

#: (counter, key, output) of Philox4x32-10, from Random123's known-answer vectors.
KNOWN_ANSWERS = (
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((_U32, _U32, _U32, _U32), (_U32, _U32), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
)


def _word(value, what: str, mask: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= mask:
        raise ValueError(f"{what} must be an integer in [0, {mask}], got {value!r}")
    return value


def _key(key) -> tuple[int, int]:
    if not isinstance(key, (tuple, list)) or len(key) != 2:
        raise ValueError("a Philox key is two 32-bit words")
    return _word(key[0], "a key word", _U32), _word(key[1], "a key word", _U32)


def philox4x32_10(counter, key) -> tuple[int, int, int, int]:
    """One Philox4x32-10 block: the bijection of `counter` (four words) under `key` (two words)."""
    if not isinstance(counter, (tuple, list)) or len(counter) != 4:
        raise ValueError("a Philox counter is four 32-bit words")
    c0, c1, c2, c3 = (_word(c, "a counter word", _U32) for c in counter)
    k0, k1 = _key(key)
    for rnd in range(10):
        if rnd:
            k0 = (k0 + _W0) & _U32
            k1 = (k1 + _W1) & _U32
        p0, p1 = _M0 * c0, _M1 * c2
        c0, c1, c2, c3 = (
            ((p1 >> 32) ^ c1 ^ k0) & _U32,
            p1 & _U32,
            ((p0 >> 32) ^ c3 ^ k1) & _U32,
            p0 & _U32,
        )
    return c0, c1, c2, c3


def _stream(key, stream, index) -> None:
    """Refuse a stream request that is not (two key words, a 64-bit stream id, a 64-bit index): every entry point
    checks before it draws, so a request drawing nothing (a fill of 0, a bound of 0) is refused alike."""
    _key(key)
    _word(stream, "a stream id", _U64)
    _word(index, "an element index", _U64)


def _block(key, stream: int, number: int) -> tuple[int, int, int, int]:
    return philox4x32_10((number & _U32, number >> 32, stream & _U32, stream >> 32), key)


def u32(key, stream: int, index: int) -> int:
    """Element `index` of the stream (`key`, `stream`)."""
    _stream(key, stream, index)
    return _block(key, stream, index >> 2)[index & 3]


def fill_u32(key, stream: int, first: int, n: int) -> list[int]:
    """Elements `[first, first + n)` of the stream."""
    _stream(key, stream, first)
    if isinstance(n, bool) or not isinstance(n, int) or n < 0 or first + n > _U64 + 1:
        raise ValueError("a count must take the stream no past its last element")
    return [u32(key, stream, first + k) for k in range(n)]


def unit_f32_bits(bits: int) -> float:
    """A float in [0, 1) from one element: its top 24 bits times 2^-24 (exact in binary32)."""
    return (_word(bits, "an element", _U32) >> 8) * 2.0**-24


def unit_f64(hi: int, lo: int) -> float:
    """A double in [0, 1) from two elements, `hi` the earlier: the top 53 of their 64 bits times 2^-53."""
    bits = (_word(hi, "an element", _U32) << 32) | _word(lo, "an element", _U32)
    return (bits >> 11) * 2.0**-53


def below(key, stream: int, index: int, bound: int) -> tuple[int, int]:
    """(a uniform integer in [0, bound), the next index): multiply-and-shift with rejection (D. Lemire, "Fast Random
    Integer Generation in an Interval", ACM TOMACS 29(1), 2019), taking elements from `index` on. A bound of 0
    gives 0 and takes none."""
    _stream(key, stream, index)
    _word(bound, "a bound", _U32)
    if bound == 0:
        return 0, index
    m = u32(key, stream, index) * bound
    index += 1
    low = m & _U32
    if low < bound:
        threshold = ((1 << 32) - bound) % bound
        while low < threshold:
            m = u32(key, stream, index) * bound
            index += 1
            low = m & _U32
    return m >> 32, index


def permutation(key, stream: int, index: int, n: int) -> tuple[list[int], int]:
    """(a permutation of [0, n), the next index): Fisher-Yates from element `index` on -- for i from n - 1 down to
    1, j = below(i + 1), then swap the two."""
    _stream(key, stream, index)
    _word(n, "a permutation's length", _U32)
    out = list(range(n))
    for i in range(n - 1, 0, -1):
        j, index = below(key, stream, index, i + 1)
        out[i], out[j] = out[j], out[i]
    return out, index


__all__ = [
    "KNOWN_ANSWERS",
    "below",
    "fill_u32",
    "permutation",
    "philox4x32_10",
    "u32",
    "unit_f32_bits",
    "unit_f64",
]
