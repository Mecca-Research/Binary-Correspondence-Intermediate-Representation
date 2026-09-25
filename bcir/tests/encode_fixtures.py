"""The StreamPack encoder's compiled record layouts, held to the encoder they replace.

`bcir.abi.streampack_abi` builds each record in one function -- a plain record (every field of
its exact type) in ONE `struct` call through the cached layout of its shape, any other record
field by field -- and checks the encode contract with an exact fast path per record. The rule is
that none of it may change a byte or a refusal. `encode_reference` below is the encoder as it was
before: the `_Writer` rail, one method call, one range check and one `struct.pack` per field, and
the contract checked record by record. It is kept verbatim, as the planner keeps
`realize_reference`, and is read only by this grader. (It shares the contract functions this
slice did not change -- `_validate_header_contract`, `_validate_segment`, `_validate_prefetch`,
`_encode_header`, `_wire_version`, `_segment_needs_v3` and the `_Writer` class -- and carries its
own copy of each one it did.)

One function, `measure`, grades the row the tests, the harness (`tools/perf/gemplus_baseline.py
--group encode`) and `tools/perf/check_encode.py` share:

    streampack.encode.parity   (item, rail) pairs where the compiled encoder's outcome -- its
                               bytes, or its refusal: the exception's type and message -- differs
                               from the reference's. The items: every honest pack of the corpus
                               in 12 spellings (`spellings`: wire v1 through v4, each v3 field
                               alone, and eight more a fast path could get wrong); every field of
                               every record kind forged, one at a time and two at a time (the
                               FIRST finding must be the same); and every forged record handed to
                               the record functions directly, without the contract in front of
                               them, at every wire version -- the delta StreamPack and the MC1
                               inspector call them that way.

The forged values are the ones a fast path gets wrong: `True` and `1.0` where the wire wants an
integer (`struct` packs a `bool`), an object with `__index__` that is not an `int` (`struct`
packs that too, the contract refuses it), an `int` subclass in range (the contract accepts it),
values one past each field's width, a lone surrogate, strings and arrays one element or one byte
past the u16 length, `str` and `bytes` where the other belongs. And in place of an array: a
generator (it has no `len()`), a sized iterable that yields its items once, and one whose `len()`
is one short of them. `_Writer` walks an array once and writes its `len()` as the count; a path
that walks an array twice (checked, then packed) or believes an empty `len()` does not. Each of
the three is spent by use, so every rail call gets its own (`_Fresh`).

On the parent tree the compiled encoder is the reference itself, so the row reads 0 there too: it
is a guard, and the rows that move are the calls and the time (`encode_calls`, `encode_ratio`).
"""

from __future__ import annotations

import dataclasses
import struct
import zlib

ROWS = ("streampack.encode.parity",)
RATIO_SCALE = 4
CALLS_SCALE = 8
# The floor the compiled encoder holds: one encode of the audit pack makes fewer than a sixth of the
# reference's calls (0.130 measured at scales 1, 2 and 8; without the plain segment layouts 0.215,
# without the plain prefetch layouts 0.177 -- each over this floor).
CALLS_FRACTION = 6
TARGETS = ("x86_avx2", "arm64_sve")


# --- the reference: the encoder before SP-ENC, verbatim -----------------------------------------


def _ref_write_segment(w, seg, version: int) -> None:
    from bcir.abi.streampack_abi import _DISPATCH_WIRE

    w.s(seg.name)
    w.u64(seg.claim_id)
    w.u32(seg.phase_id)
    w.u8(int(seg.lane))
    w.u32(seg.width)
    w.u32(0)  # stride_k reserved on the segment record (carried per-claim)
    w.s(seg.opcode)
    w.u32_array(seg.reads)
    w.u32_array(seg.writes)
    w.s(seg.prefetch or "")
    w.s_array(seg.fence_before)
    w.s_array(seg.fence_after)
    if version >= 3:
        w.u8(_DISPATCH_WIRE[seg.dispatch])
        w.s(seg.channel)


def _ref_write_prefetch(w, pf, version: int) -> None:
    w.s(pf.name)
    w.u32(pf.distance)
    w.u32_array(pf.targets)
    w.s(pf.hint)
    w.s(pf.pattern)
    if version >= 2:
        w.u8(pf.buffers)


def _ref_write_block(w, blk) -> None:
    w.u64(blk.base)
    w.u64(blk.count)
    w.u64_array(blk.strides)


def _ref_write_trace(w, note) -> None:
    w.u64(note.claim_id)
    w.u64(note.src_hash)
    w.u64(note.trace_hash)


def _ref_write_generation(w, g) -> None:
    w.u32(g.rid)
    w.u32(g.map_gen)
    w.u32(g.data_gen)


def _ref_validate_generation_vector(pack) -> None:
    from bcir.abi.streampack_abi import AbiError, _checked_uint

    gens = list(pack.generations)
    _checked_uint("n_gens", len(gens), 32)
    previous = -1
    for index, g in enumerate(gens):
        rid = _checked_uint(f"generation[{index}].rid", g.rid, 32)
        _checked_uint(f"generation[{index}].map_gen", g.map_gen, 32)
        _checked_uint(f"generation[{index}].data_gen", g.data_gen, 32)
        if rid <= previous:
            raise AbiError(
                f"generation vector RIDs must be strictly ascending (generation[{index}] "
                f"rid {rid} after {previous})"
            )
        previous = rid
    if gens:
        vec_map = max(g.map_gen for g in gens)
        vec_data = max(g.data_gen for g in gens)
        if pack.map_gen != vec_map or pack.data_gen != vec_data:
            raise AbiError(
                f"header map_gen/data_gen ({pack.map_gen}, {pack.data_gen}) must be the "
                f"generation vector's maxima ({vec_map}, {vec_data})"
            )


def _ref_validate_encode_contract(pack) -> None:
    from bcir.abi.streampack_abi import (
        _validate_header_contract,
        _validate_prefetch,
        _validate_segment,
    )

    _validate_header_contract(pack)
    for index, seg in enumerate(pack.segments):
        _validate_segment(index, seg)
    for index, pf in enumerate(pack.prefetches):
        _validate_prefetch(index, pf)
    _ref_validate_generation_vector(pack)


def encode_reference(pack) -> bytes:
    """The StreamPack encoder before SP-ENC (the `_Writer` rail), verbatim."""
    from bcir.abi.streampack_abi import _encode_header, _segment_needs_v3, _wire_version, _Writer

    _ref_validate_encode_contract(pack)
    needs_v2 = pack.pipeline_depth > 1 or any(pf.buffers != 1 for pf in pack.prefetches)
    needs_v3 = any(_segment_needs_v3(seg) for seg in pack.segments)
    version = _wire_version(needs_v2, needs_v3, bool(pack.generations))
    header = _encode_header(pack, version)

    w = _Writer()
    w.s(pack.source_plan)
    for seg in pack.segments:
        _ref_write_segment(w, seg, version)
    for pf in pack.prefetches:
        _ref_write_prefetch(w, pf, version)
    for blk in pack.blocks:
        _ref_write_block(w, blk)
    for t in pack.trace_notes:
        _ref_write_trace(w, t)
    if version >= 4:
        for g in pack.generations:
            _ref_write_generation(w, g)

    body = header + bytes(w.buf)
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def _ref_record(write, record, *version) -> bytes:
    from bcir.abi.streampack_abi import _Writer

    w = _Writer()
    write(w, record, *version)
    return bytes(w.buf)


# --- the corpus -------------------------------------------------------------------------------


class _Int(int):
    """An `int` subclass: the contract accepts it (`isinstance(v, int)`, not a `bool`)."""


class _Str(str):
    """A `str` subclass: the contract accepts it."""


class _Index:
    """Not an `int`, but `struct` packs it (`__index__`): the contract refuses it."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return f"_Index({self.value})"


class _Once:
    """A sized iterable that yields its items once, and whose `len()` may be other than their
    number: `_Writer` walks an array once and writes its `len()` as the count, so it packs one."""

    def __init__(self, items: tuple, length: int | None = None) -> None:
        self.items, self.length, self.spent = items, length, False

    def __len__(self) -> int:
        return len(self.items) if self.length is None else self.length

    def __iter__(self):
        if self.spent:
            return iter(())
        self.spent = True
        return iter(self.items)

    def __repr__(self) -> str:
        return f"_Once({self.items!r}, len={len(self)})"


class _Fresh:
    """A forged value spent by use: `_forge` makes a new one for every forged pack, so no rail
    sees what another rail left of it."""

    def __init__(self, make, label: str) -> None:
        self.make, self.label = make, label

    def __repr__(self) -> str:
        return self.label


def _spent(items: tuple) -> list:
    """In place of an array holding `items`: a generator (no `len()`: refused), a sized iterable
    that yields them once, and one whose `len()` is one short of them."""
    return [
        _Fresh(lambda: (x for x in items), f"a generator of {items!r}"),
        _Fresh(lambda: _Once(items), f"_Once({items!r})"),
        _Fresh(lambda: _Once(items, len(items) - 1), f"_Once({items!r}, len short by one)"),
    ]


def _ints(bits: int) -> list:
    return [
        -1,
        (1 << bits) - 1,
        1 << bits,
        1 << 64,
        True,
        False,
        1.0,
        "1",
        None,
        _Int(1),
        _Int(1 << bits),
        _Index(1),
    ]


_STRS = [
    b"x",
    1,
    None,
    "\ud800",  # a lone surrogate: UTF-8 refuses it
    "x" * 65535,  # the longest string the u16 length carries
    "x" * 65536,  # one byte past it
    "é" * 32768,  # 32,768 characters, 65,536 bytes
    _Str("ok"),
    "",
    "café ✓",
]
_U32S = [
    (),
    [1, 2],
    (1, -1),
    (1 << 32,),
    ((1 << 32) - 1,),
    (True,),
    (1.0,),
    ("1",),
    (_Int(3),),
    (_Index(1),),
    (0,) * 15,
    (0,) * 16,  # the first count past the precompiled layouts
    (0,) * 65535,
    (0,) * 65536,
    "ab",
    None,
    *_spent((1, 2)),
]
_U64S = [x if not (isinstance(x, tuple) and x == (1 << 32,)) else (1 << 64,) for x in _U32S]
_STR_ARRAYS = [
    ("a",),
    ("a", 1),
    ["x", "y"],
    ("\ud800",),
    ("",) * 65536,
    (b"x",),
    None,
    (),
    *_spent(("a",)),  # the one short of its item: `len()` 0
]
_LANES = ["U", None, True, 99, 256, -1, 1, _Int(1), 1.5]
_DISPATCHES = ["core", "pim", "gpu", 1, None, ["core"]]
_BUFFERS = [0, 1, 2, 3, True, 2.0, _Int(2), None]

# Every field of every record kind, and the values it is forged with.
SEGMENT_FIELDS = {
    "name": _STRS,
    "claim_id": _ints(64),
    "phase_id": _ints(32),
    "lane": _LANES,
    "width": _ints(32) + [3, 0, 8],
    "opcode": _STRS,
    "reads": _U32S,
    "writes": _U32S,
    "prefetch": _STRS,
    "fence_before": _STR_ARRAYS,
    "fence_after": _STR_ARRAYS,
    "dispatch": _DISPATCHES,
    "channel": _STRS,
}
PREFETCH_FIELDS = {
    "name": _STRS,
    "distance": _ints(32),
    "targets": _U32S,
    "hint": _STRS,
    "pattern": _STRS,
    "buffers": _BUFFERS,
}
BLOCK_FIELDS = {"base": _ints(64), "count": _ints(64), "strides": _U64S}
TRACE_FIELDS = {"claim_id": _ints(64), "src_hash": _ints(64), "trace_hash": _ints(64)}
GENERATION_FIELDS = {"rid": _ints(32), "map_gen": _ints(32), "data_gen": _ints(32)}
HEADER_FIELDS = {
    "source_plan": _STRS,
    "topo_gen": _ints(32),
    "map_gen": _ints(32),
    "data_gen": _ints(32),
    "pipeline_depth": _ints(16) + [0, 2],
}
RECORDS = {
    "segments": SEGMENT_FIELDS,
    "prefetches": PREFETCH_FIELDS,
    "blocks": BLOCK_FIELDS,
    "trace_notes": TRACE_FIELDS,
    "generations": GENERATION_FIELDS,
}


def honest_packs() -> list:
    """Every program under two targets, hydrated plain and pipelined, and the audit fixture."""
    from bcir.examples import PROGRAMS
    from bcir.gem.streampack import hydrate, hydrate_pipelined
    from bcir.kbcir import TARGETS as PROFILES
    from bcir.kbcir import optimize
    from bcir.kbcir.cost import Theta
    from bcir.kbcir.weights import PERF
    from bcir.tests.planner_fixtures import audit_fixture

    packs = []
    for name in sorted(PROGRAMS):
        module = PROGRAMS[name]() if callable(PROGRAMS[name]) else PROGRAMS[name]
        for target in TARGETS:
            result = optimize(module, PROFILES[target], Theta.cool(), PERF)
            packs.append(hydrate(module, result))
            packs.append(hydrate_pipelined(module, result, "plan0", 2))
    module, h, theta = audit_fixture(1)
    packs.append(hydrate_pipelined(module, optimize(module, h, theta, PERF), "plan0", 2))
    return packs


def spellings(pack) -> list:
    """`pack` in every wire version and in the spellings a plain record does not take."""
    replace = dataclasses.replace
    base = replace(pack, generations=[], pipeline_depth=1)
    v1 = replace(base, prefetches=[replace(pf, buffers=1) for pf in pack.prefetches])
    out = [("v4", pack), ("v1", v1), ("v2", replace(v1, pipeline_depth=2))]
    if pack.segments:
        first = pack.segments[0]
        rest = pack.segments[1:]
        out += [
            (
                "v3",
                replace(v1, segments=[replace(first, dispatch="pim", channel="hbm_pim"), *rest]),
            ),
            # each v3 field raises the version on its own
            ("v3-dispatch", replace(v1, segments=[replace(first, dispatch="pim"), *rest])),
            ("v3-channel", replace(v1, segments=[replace(first, channel="hbm_pim"), *rest])),
            ("fences", replace(pack, segments=[replace(first, fence_before=("acq",)), *rest])),
            ("lists", replace(pack, segments=[replace(first, reads=list(first.reads)), *rest])),
            ("subclasses", replace(pack, segments=[replace(first, name=_Str(first.name)), *rest])),
            ("int-lane", replace(pack, segments=[replace(first, lane=int(first.lane)), *rest])),
        ]
    if len(pack.generations) > 1:
        # the vector's maxima in its LAST entry, so a maximum taken from any other entry is wrong
        *head, last = pack.generations
        late = replace(last, map_gen=last.map_gen + 5, data_gen=last.data_gen + 7)
        top_map = max(g.map_gen for g in (*head, late))
        top_data = max(g.data_gen for g in (*head, late))
        late_max = replace(pack, generations=[*head, late], map_gen=top_map, data_gen=top_data)
        out.append(("late-max", late_max))
    if pack.blocks:
        out.append(
            ("long-strides", replace(pack, blocks=[replace(pack.blocks[0], strides=(1,) * 20)]))
        )
    return out


def outcome(fn, *args) -> tuple:
    """What `fn(*args)` does: ("bytes", its bytes) or ("raise", the exception's type, message)."""
    try:
        return ("bytes", fn(*args))
    except Exception as exc:  # noqa: BLE001 -- the refusal is the outcome being compared
        return ("raise", type(exc).__name__, str(exc))


def _forge(pack, kind: str, fields: dict) -> object:
    """`pack` with the first record of `kind` given `fields`, or the pack's header fields. A value
    spent by use is made anew each time (`_Fresh`)."""
    fields = {name: v.make() if isinstance(v, _Fresh) else v for name, v in fields.items()}
    if kind == "header":
        return dataclasses.replace(pack, **fields)
    records = list(getattr(pack, kind))
    records[0] = dataclasses.replace(records[0], **fields)
    return dataclasses.replace(pack, **{kind: records})


def _first_refused(values: list):
    """The first value of a family the contract refuses (for the two-field forgeries)."""
    for value in values:
        if value is None or isinstance(value, (bytes, float, _Index)) or value == -1:
            return value
    return values[0]


def _record_rails() -> dict:
    from bcir.abi import streampack_abi as sp

    return {
        "segments": (sp._segment_bytes, _ref_write_segment, True),
        "prefetches": (sp._prefetch_bytes, _ref_write_prefetch, True),
        "blocks": (sp._block_bytes, _ref_write_block, False),
        "trace_notes": (sp._trace_bytes, _ref_write_trace, False),
        "generations": (sp._generation_bytes, _ref_write_generation, False),
    }


def measure(seen: dict | None = None) -> dict[str, float]:
    """The row (module docstring), over the whole corpus."""
    from bcir.abi.streampack_abi import encode

    mismatches = 0
    stats = {"packs": 0, "records": 0, "refused": 0, "accepted": 0, "versions": set()}
    stats["fields"] = set()

    def judge(new: tuple, ref: tuple) -> None:
        nonlocal mismatches
        mismatches += new != ref
        stats["refused" if ref[0] == "raise" else "accepted"] += 1

    packs = honest_packs()
    for pack in packs:
        for label, spelled in spellings(pack):
            ref = outcome(encode_reference, spelled)
            judge(outcome(encode, spelled), ref)
            stats["packs"] += 1
            if ref[0] == "bytes":
                stats["versions"].add(int.from_bytes(ref[1][4:6], "little"))
    base = packs[1]  # a pipelined pack with every record kind
    rails = _record_rails()

    def compare(kind: str, fields: dict, versions: tuple) -> None:
        """One forgery through `encode`, and a record's through the record functions at each of
        `versions`. Every call forges its own pack: a value may be spent by use (`_Fresh`)."""
        judge(
            outcome(encode, _forge(base, kind, fields)),
            outcome(encode_reference, _forge(base, kind, fields)),
        )
        stats["packs"] += 1
        if kind == "header":
            return
        compiled, reference, versioned = rails[kind]
        for version in versions if versioned else (None,):
            tail = (version,) if versioned else ()
            new = outcome(compiled, getattr(_forge(base, kind, fields), kind)[0], *tail)
            record = getattr(_forge(base, kind, fields), kind)[0]
            judge(new, outcome(_ref_record, reference, record, *tail))
            stats["records"] += 1

    for kind, fields in (("header", HEADER_FIELDS), *RECORDS.items()):
        if kind != "header" and not getattr(base, kind):
            continue
        for field, values in fields.items():
            stats["fields"].add(f"{kind}.{field}")
            for value in values:
                compare(kind, {field: value}, (1, 2, 3, 4))
        # two fields forged at once: the first finding in wire order must be the same one
        names = list(fields)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                two = {first: _first_refused(fields[first]), second: _first_refused(fields[second])}
                compare(kind, two, (1, 4))
    if seen is not None:
        seen.update(stats)
    return {"streampack.encode.parity": float(mismatches)}


# --- the rows that move: calls and time -------------------------------------------------------


def audit_pack(scale: int):
    """The audit fixture's pipelined pack at `scale` (the chain's StreamPack)."""
    from bcir.gem.streampack import hydrate_pipelined
    from bcir.kbcir import optimize
    from bcir.kbcir.weights import PERF
    from bcir.tests.planner_fixtures import audit_fixture

    module, h, theta = audit_fixture(scale)
    return hydrate_pipelined(module, optimize(module, h, theta, PERF), "plan0", 2)


def encode_calls(scale: int = CALLS_SCALE) -> tuple[int, int]:
    """(calls of `encode`, calls of `encode_reference`) on the audit pack at `scale` (cProfile's
    total, builtins included). Deterministic for one interpreter, so a gate compares the two in
    one process."""
    from bcir.abi.streampack_abi import encode
    from bcir.tests.delta_fixtures import _profiled_calls

    pack = audit_pack(scale)
    if encode(pack) != encode_reference(pack):  # warm, and never time two different answers
        raise AssertionError("encode and encode_reference disagree on the audit pack")
    return _profiled_calls(encode, pack), _profiled_calls(encode_reference, pack)


def encode_ratio(scale: int = RATIO_SCALE, rounds: int = 9) -> float:
    """Median time of `encode` / median time of `encode_reference` on the audit pack at `scale`,
    interleaved in one process (a same-host ratio, the `ratio` band)."""
    import time

    from bcir.abi.streampack_abi import encode

    pack = audit_pack(scale)
    new, ref = [], []
    for _ in range(rounds):
        start = time.perf_counter()
        encode(pack)
        new.append(time.perf_counter() - start)
        start = time.perf_counter()
        encode_reference(pack)
        ref.append(time.perf_counter() - start)
    new.sort()
    ref.sort()
    return new[rounds // 2] / ref[rounds // 2]


def calls_over(scale: int = 2) -> int:
    """1 when one encode of the audit pack at `scale` makes a sixth or more of the reference
    encoder's calls (`CALLS_FRACTION`), else 0: the saving itself, as a row that can fire."""
    new, ref = encode_calls(scale)
    return int(new * CALLS_FRACTION >= ref)
