"""GEM+ G18 (S4-B): the delta StreamPack -- a pack re-emitted from the steps a delta changed.

`PackState.build(module, result, plan, depth)` hydrates the plan (`hydrate_pipelined`) and keeps
the pack as records with their encodings, one byte string per record, joined in chunks of
`_CHUNK` records. `apply` takes the next plan -- the `IncrementalPlan` result a delta produced,
with the step columns it changed -- and re-emits only:

  * the records of the changed steps (`streampack.step_records`, the function `hydrate` builds
    them with), each kept as it was when the new record equals the old one;
  * the double-buffer prefetch of a transition into a phase whose reads the delta changed
    (`streampack.double_buffer`);
  * the generation records of the replaced resources, and the header maxima over them;

and splices them into the old encoding: the header re-derived for the new section counts, the
chunks holding a re-emitted record re-joined, the body joined from the chunks, one CRC-32 over the
whole. The bytes are `abi.streampack_abi.encode(hydrate_pipelined(module, result, plan, depth))`,
byte for byte, and the object equals the pack hydration builds (`pack.delta.identity`).

A pack the wire cannot carry is refused with the error `encode` raises first: the re-emitted
records are checked with the encoder's own contract functions and written with its own writers,
section by section, in its order, and every kept record was accepted when it was first emitted.
After a refusal the state is behind its plan, so the next call re-emits in full.

Declared boundary (v0): the pack's version follows from facts a delta keeps or counts (the
generation vector's presence, the pipelining, the segments that carry a dispatch or channel of
their own, the prefetches that double-buffer); a delta that would change it is re-emitted in
full (`PackState.full`).
"""

from __future__ import annotations

import struct
import zlib

from ..abi.streampack_abi import (
    _encode_header,
    _prefetch_buffers_ok,
    _record_bytes,
    _segment_needs_v3,
    _validate_generation_vector,
    _validate_header_contract,
    _validate_prefetch,
    _validate_segment,
    _wire_version,
    _write_block,
    _write_generation,
    _write_prefetch,
    _write_segment,
    _write_trace,
    encode,
)
from ..model import Module, topological_phase_ids
from .streampack import (
    StreamPack,
    double_buffer,
    generation_vector,
    hydrate_pipelined,
    step_records,
)

__all__ = ["PackState"]

_CHUNK = 256  # records per joined chunk: a delta re-joins the chunks it touched, not a section


class _Section:
    """One record family's encodings, per record (None: the record is absent -- a step with no
    prefetch) and joined per chunk of `_CHUNK` records."""

    __slots__ = ("records", "chunks")

    def __init__(self, records: list):
        self.records = records
        self.chunks = [
            b"".join(filter(None, records[c : c + _CHUNK])) for c in range(0, len(records), _CHUNK)
        ]

    def replace(self, encoded: dict) -> None:
        """Replace records (index -> encoding or None) and re-join the chunks they sit in."""
        records = self.records
        touched = set()
        for i, data in encoded.items():
            records[i] = data
            touched.add(i // _CHUNK)
        for c in touched:
            lo = c * _CHUNK
            self.chunks[c] = b"".join(filter(None, records[lo : lo + _CHUNK]))


class PackState:
    """A hydrated StreamPack kept as records and their encodings, advanced by the steps a delta
    changed. `pack` and `data` are always the pack of the current plan and its bytes."""

    __slots__ = (
        "plan",
        "depth",
        "pack",
        "data",
        "version",
        "_segments",
        "_step_pf",
        "_blocks",
        "_notes",
        "_transitions",
        "_into",
        "_tr_pf",
        "_gens",
        "_gen_at",
        "_maxima",
        "_source",
        "_seg_b",
        "_pf_b",
        "_blk_b",
        "_note_b",
        "_tr_b",
        "_gen_b",
        "_v3",
        "_buffered",
        "_stale",
    )

    @classmethod
    def build(cls, module: Module, result, plan: str = "plan0", depth: int = 2) -> "PackState":
        """Hydrate `result` and keep its encoding per record."""
        self = cls.__new__(cls)
        self.plan, self.depth = plan, depth
        self.full(module, result)
        return self

    def full(self, module: Module, result) -> tuple[StreamPack, bytes]:
        """Hydrate and encode from scratch (the reference path, and the fallback)."""
        pack = hydrate_pipelined(module, result, self.plan, self.depth)
        data = encode(pack)  # the encoder's own contract and writers, once
        by_name = {pf.name: pf for pf in pack.prefetches}
        self._segments = list(pack.segments)
        self._blocks = list(pack.blocks)
        self._notes = list(pack.trace_notes)
        self._step_pf = [
            by_name[seg.prefetch] if seg.prefetch is not None else None for seg in pack.segments
        ]
        order = topological_phase_ids(module)
        pmap = module.phase_map()
        self._transitions = list(zip(order, order[1:])) if pack.pipeline_depth > 1 else []
        self._into = {nxt: t for t, (_, nxt) in enumerate(self._transitions)}
        self._tr_pf = [double_buffer(a, b, pmap[b]) for a, b in self._transitions]
        self._gens = list(pack.generations)
        self._gen_at = {g.rid: i for i, g in enumerate(self._gens)}
        self._maxima = (pack.map_gen, pack.data_gen)
        self._v3 = sum(1 for seg in pack.segments if _segment_needs_v3(seg))
        self._buffered = sum(1 for pf in pack.prefetches if _buffered(pf))
        self.version = v = data[4] | (data[5] << 8)
        self._source = _record_bytes(_write_source, self.plan)
        self._seg_b = _Section([_record_bytes(_write_segment, s, v) for s in self._segments])
        self._pf_b = _Section(
            [None if p is None else _record_bytes(_write_prefetch, p, v) for p in self._step_pf]
        )
        self._blk_b = _Section([_record_bytes(_write_block, b) for b in self._blocks])
        self._note_b = _Section([_record_bytes(_write_trace, t) for t in self._notes])
        self._tr_b = [
            None if p is None else _record_bytes(_write_prefetch, p, v) for p in self._tr_pf
        ]
        self._gen_b = [_record_bytes(_write_generation, g) for g in self._gens]
        self.pack, self.data = pack, data
        self._stale = False
        return pack, data

    def apply(
        self, module: Module, result, claims, changed, edited, resources=()
    ) -> tuple[StreamPack, bytes]:
        """Re-emit the pack of `result` (the plan of `module`) from the old one. `claims[n]` is
        the claim step n realizes; `changed` names the step columns the delta changed, `edited`
        the columns whose claim it replaced, `resources` the RIDs it replaced.

        A pack the wire refuses raises what `encode` raises and leaves the state behind its
        plan; the next call then re-emits in full, so a later delta cannot splice onto records of
        a pack that was never emitted."""
        if self._stale:
            return self.full(module, result)
        try:
            return self._apply(module, result, claims, changed, edited, resources)
        except Exception:
            self._stale = True
            raise

    def _apply(self, module, result, claims, changed, edited, resources):
        segments, step_pf, blocks = self._segments, self._step_pf, self._blocks
        steps = result.steps
        seg_new: dict = {}
        pf_new: dict = {}
        blk_new: dict = {}
        for n in sorted(changed):
            seg, pf, blk, _note = step_records(n, steps[n], claims[n])
            if seg != segments[n]:
                seg_new[n] = seg
            if pf != step_pf[n]:
                pf_new[n] = pf
            if blk != blocks[n]:
                blk_new[n] = blk
        # A transition's double buffer reads the next phase's claims: re-derive it for each
        # phase an edited claim's reads changed in.
        tr_new: dict = {}
        if self._transitions:
            pmap = None
            for n in edited:
                t = self._into.get(steps[n].phase_id)
                if t is None or t in tr_new:
                    continue
                if pmap is None:
                    pmap = module.phase_map()
                a, b = self._transitions[t]
                pf = double_buffer(a, b, pmap[b])
                if pf != self._tr_pf[t]:
                    tr_new[t] = pf
        gen_new: dict = {}
        if resources:
            live = {g.rid: g for g in generation_vector(module)}
            for rid in resources:
                i = self._gen_at.get(rid)
                if i is not None and live[rid] != self._gens[i]:
                    gen_new[i] = live[rid]
        # The version facts, kept as counts so a delta reads them without a scan.
        v3 = self._v3 + sum(
            _segment_needs_v3(seg) - _segment_needs_v3(segments[n]) for n, seg in seg_new.items()
        )
        buffered = self._buffered
        buffered += sum(_buffered(p) - _buffered(step_pf[n]) for n, p in pf_new.items())
        buffered += sum(_buffered(p) - _buffered(self._tr_pf[t]) for t, p in tr_new.items())
        version = _wire_version(self.depth > 1 or buffered > 0, v3 > 0, bool(self._gens))
        if version != self.version:
            return self.full(module, result)  # every record's encoding depends on the version
        return self._splice(seg_new, pf_new, blk_new, tr_new, gen_new, v3, buffered)

    def _splice(self, seg_new, pf_new, blk_new, tr_new, gen_new, v3, buffered):
        step_pf = self._step_pf
        if pf_new:  # the new step prefetches, a list of their own until the commit below
            step_pf = step_pf.copy()
            for n, pf in pf_new.items():
                step_pf[n] = pf
        tr_pf = self._tr_pf
        if tr_new:
            tr_pf = list(tr_pf)
            for t, pf in tr_new.items():
                tr_pf[t] = pf
        gens, maxima = self._gens, self._maxima
        if gen_new:
            gens = list(gens)
            for i, g in gen_new.items():
                gens[i] = g
            maxima = (
                max((g.map_gen for g in gens), default=0),
                max((g.data_gen for g in gens), default=0),
            )
        # The new pack's prefetch section, built here: its length is the header's count, so the
        # count is the section's own (no counter kept beside it to go stale).
        stepped = list(filter(None, step_pf))
        prefetches = stepped + list(filter(None, tr_pf))
        header = StreamPack(
            source_plan=self.plan,
            topo_gen=1,
            map_gen=maxima[0],
            data_gen=maxima[1],
            pipeline_depth=max(1, self.depth),
            segments=_Sized(len(self._segments)),
            prefetches=_Sized(len(prefetches)),
            blocks=_Sized(len(self._blocks)),
            trace_notes=_Sized(len(self._notes)),
            generations=gens,
        )
        # The encoder's contract, in its order, over what is re-emitted: the header, the
        # segments, the prefetches, the generation vector (every kept record passed it once).
        _validate_header_contract(header)
        for n in sorted(seg_new):
            _validate_segment(n, seg_new[n])
        for n in sorted(pf_new):
            pf = pf_new[n]
            if pf is not None and not _prefetch_buffers_ok(pf):
                # its place among the new pack's prefetches (only a refusal needs it)
                _validate_prefetch(sum(1 for p in step_pf[:n] if p is not None), pf)
        for t in sorted(tr_new):
            pf = tr_new[t]
            if pf is not None and not _prefetch_buffers_ok(pf):
                _validate_prefetch(len(stepped) + sum(1 for p in tr_pf[:t] if p is not None), pf)
        if gen_new:
            _validate_generation_vector(header)
        # The encoder's writers, section by section in its order (a refusal from here on leaves
        # the state stale, and the next call re-emits in full).
        v = self.version
        seg_b = {n: _record_bytes(_write_segment, seg_new[n], v) for n in sorted(seg_new)}
        pf_b = {
            n: None if pf_new[n] is None else _record_bytes(_write_prefetch, pf_new[n], v)
            for n in sorted(pf_new)
        }
        tr_b = {
            t: None if tr_new[t] is None else _record_bytes(_write_prefetch, tr_new[t], v)
            for t in sorted(tr_new)
        }
        blk_b = {n: _record_bytes(_write_block, blk_new[n]) for n in sorted(blk_new)}
        gen_b = {i: _record_bytes(_write_generation, gen_new[i]) for i in sorted(gen_new)}
        # Commit, in place: the pack handed out below owns fresh lists.
        for n, seg in seg_new.items():
            self._segments[n] = seg
        self._step_pf = step_pf
        for n, blk in blk_new.items():
            self._blocks[n] = blk
        self._seg_b.replace(seg_b)
        self._pf_b.replace(pf_b)
        self._blk_b.replace(blk_b)
        for t, b in tr_b.items():
            self._tr_b[t] = b
        for i, b in gen_b.items():
            self._gen_b[i] = b
        self._tr_pf, self._gens, self._maxima = tr_pf, gens, maxima
        self._v3, self._buffered = v3, buffered
        pack = StreamPack(
            source_plan=self.plan,
            topo_gen=1,
            map_gen=maxima[0],
            data_gen=maxima[1],
            pipeline_depth=header.pipeline_depth,
            segments=list(self._segments),
            prefetches=prefetches,
            blocks=list(self._blocks),
            trace_notes=list(self._notes),
            generations=list(gens),
        )
        body = b"".join(  # the header joined with the sections: the bytes are copied once
            [
                _encode_header(pack, v),
                self._source,
                *self._seg_b.chunks,
                *self._pf_b.chunks,
                *filter(None, self._tr_b),
                *self._blk_b.chunks,
                *self._note_b.chunks,
                *(self._gen_b if v >= 4 else ()),
            ]
        )
        data = body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)
        self.pack, self.data = pack, data
        return pack, data


class _Sized:
    """A stand-in list for the header contract, which reads only a section's length."""

    __slots__ = ("n",)

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n


def _buffered(pf) -> bool:
    """A prefetch that needs v2 (a buffer count other than 1)."""
    return pf is not None and pf.buffers != 1


def _write_source(w, plan: str) -> None:
    w.s(plan)
