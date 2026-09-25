"""SP-ENC: the StreamPack encoder's record layouts, compiled.

`bcir.abi.streampack_abi` builds a plain record in one `struct` call through the layout of its
shape, and any other record field by field; the encode contract takes an exact fast path per
record. None of it may change a byte or a refusal: `encode_fixtures.encode_reference` is the
encoder before, kept verbatim, and every test here holds the compiled encoder to it -- over honest
packs in every wire version and spelling, and over every field of every record forged.
"""

from __future__ import annotations

import dataclasses
import functools
import struct

from bcir.tests import encode_fixtures as ef


@functools.cache
def _base():
    return ef.honest_packs()[1]  # a pipelined pack: every record kind, wire v4


def test_the_compiled_encoder_is_the_reference_byte_for_byte_and_refusal_for_refusal():
    seen: dict = {}
    assert ef.measure(seen) == {"streampack.encode.parity": 0.0}
    # Not a vacuous zero: every wire version was encoded, every field of every record kind was
    # forged, and the corpus both packs and refuses, through encode and the records directly.
    assert seen["versions"] == {1, 2, 3, 4}, seen["versions"]
    expected = {f"header.{field}" for field in ef.HEADER_FIELDS}
    expected |= {f"{kind}.{field}" for kind, fields in ef.RECORDS.items() for field in fields}
    assert seen["fields"] == expected, expected ^ seen["fields"]
    assert seen["refused"] >= 500 and seen["accepted"] >= 500, seen
    assert seen["records"] >= 1000 and seen["packs"] >= 900, seen


def test_struct_packs_what_the_wire_refuses_so_a_layout_sees_only_exact_types():
    """`struct` is not the contract: it packs `True` and an object with `__index__`, and the wire
    refuses both. Each reaches the field path, and the field path's refusal is `_Writer`'s."""
    from bcir.abi.streampack_abi import AbiError, _block_bytes, _segment_bytes

    assert struct.pack("<Q", True) == struct.pack("<Q", 1)
    assert struct.pack("<Q", ef._Index(1)) == struct.pack("<Q", 1)
    seg = _base().segments[0]
    blk = _base().blocks[0]
    for forged, emit in (
        (dataclasses.replace(seg, claim_id=True), lambda r: _segment_bytes(r, 4)),
        (dataclasses.replace(seg, claim_id=ef._Index(1)), lambda r: _segment_bytes(r, 4)),
        (dataclasses.replace(seg, reads=(True,)), lambda r: _segment_bytes(r, 4)),
        (dataclasses.replace(blk, strides=(ef._Index(1),)), _block_bytes),
    ):
        try:
            emit(forged)
        except AbiError as exc:
            assert "must be an unsigned" in str(exc), exc
        else:
            raise AssertionError(f"{forged!r} was packed")
    # An `int` subclass in range is the contract's, and packs to the reference's bytes.
    accepted = dataclasses.replace(seg, claim_id=ef._Int(seg.claim_id))
    assert _segment_bytes(accepted, 4) == ef._ref_record(ef._ref_write_segment, accepted, 4)


def test_a_plain_record_takes_one_layout_and_any_other_record_the_field_path():
    from bcir.abi import streampack_abi as sp

    seg = _base().segments[0]
    pf = _base().prefetches[0]
    assert sp._plain_segment(seg, 4) == sp._segment_fields(seg, 4)
    assert sp._plain_segment(seg, 2) == sp._segment_fields(seg, 2)
    assert sp._plain_prefetch(pf, 4) == sp._prefetch_fields(pf, 4)
    assert sp._plain_prefetch(pf, 1) == sp._prefetch_fields(pf, 1)
    for field, value in (
        ("fence_before", ("acq",)),  # fence names take the field path
        ("fence_after", None),  # a falsy non-array is no empty array (the first layout's bug)
        ("fence_before", 0),
        ("reads", (True,)),
        ("name", ef._Str(seg.name)),
        ("lane", int(seg.lane)),
        ("channel", ef._Str("host")),
    ):
        assert sp._plain_segment(dataclasses.replace(seg, **{field: value}), 4) is None, field
    for field, value in (("targets", [1, False]), ("hint", b"T0"), ("buffers", 2.0)):
        assert sp._plain_prefetch(dataclasses.replace(pf, **{field: value}), 4) is None, field


def test_an_array_is_walked_once_as_the_writer_walks_it():
    """`_Writer` walks an array once and writes its `len()` as the count. The field path walks a
    tuple or a list twice (checked, then packed), so anything else is walked once: a sized
    iterable that yields its items once packs as their tuple does, a generator is refused at
    `len()`, and a `len()` short of the items is written as `_Writer` writes it."""
    from bcir.abi import streampack_abi as sp

    seg, pf, blk = _base().segments[0], _base().prefetches[0], _base().blocks[0]
    for record, field, items, emit, write, version in (
        (seg, "reads", (1, 2), sp._segment_bytes, ef._ref_write_segment, (4,)),
        (seg, "writes", (1, 2), sp._segment_bytes, ef._ref_write_segment, (1,)),
        (seg, "fence_after", ("a",), sp._segment_bytes, ef._ref_write_segment, (4,)),
        (pf, "targets", (1, 2), sp._prefetch_bytes, ef._ref_write_prefetch, (2,)),
        (blk, "strides", (1, 2), sp._block_bytes, ef._ref_write_block, ()),
    ):
        outcomes = []
        for spent in ef._spent(items):
            new = ef.outcome(emit, dataclasses.replace(record, **{field: spent.make()}), *version)
            forged = dataclasses.replace(record, **{field: spent.make()})
            assert new == ef.outcome(ef._ref_record, write, forged, *version), (field, spent)
            outcomes.append(new)
        generator, once, short = outcomes
        assert generator[:2] == ("raise", "TypeError"), (field, generator)
        assert once == ("bytes", emit(dataclasses.replace(record, **{field: items}), *version))
        assert short[0] == "bytes" and short != once, (field, short)


def test_the_layout_caches_are_bounded():
    """A layout per shape, but never more than `_SHAPES_MAX` of them per record kind: a stream of
    distinct shapes (here: a name of every length) costs a compile each, not unbounded memory."""
    from bcir.abi import streampack_abi as sp

    seg = _base().segments[0]
    for n in range(sp._SHAPES_MAX + 40):
        forged = dataclasses.replace(seg, name="n" * n)
        assert sp._segment_bytes(forged, 4) == ef._ref_record(ef._ref_write_segment, forged, 4)
    assert len(sp._SEGMENT_V3_SHAPES) <= sp._SHAPES_MAX


def test_the_inspector_partitions_every_spelling_through_the_compiled_records():
    """The MC1 inspector recovers each record's span through the same record functions `encode`
    uses; it raises when they do not partition the blob exactly."""
    from bcir.abi.streampack_abi import encode, inspect_stream_pack

    for label, pack in ef.spellings(_base()):
        data = encode(pack)
        inspection = inspect_stream_pack(data)
        assert inspection.length == len(data), label
        assert sum(span.length for span in inspection.spans) == len(data), label
        assert all(a.end == b.offset for a, b in zip(inspection.spans, inspection.spans[1:])), label


def test_encode_makes_a_fraction_of_the_reference_calls():
    """The rows that move: calls are deterministic for one interpreter, so the two encoders are
    compared in one process (the harness records the scale-8 count)."""
    new, ref = ef.encode_calls(scale=2)
    assert new * ef.CALLS_FRACTION < ref, (new, ref)
    assert ef.calls_over(scale=1) == 0
