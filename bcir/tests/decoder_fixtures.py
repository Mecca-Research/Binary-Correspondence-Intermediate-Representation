"""Fixtures for the decoder program and its two interpreters (QUAL-1, QUAL-2).

One BCIRQ8 model per head kind (the GQA fixture of `test_model_ingest`: vocab 64, d_model 8,
four query heads over two KV heads, two layers), the decoder program a prompt and a length
make of it, hydrated and planned as a pipeline would, and the edits that turn a program into
one an interpreter must refuse. Shared by `test_decoder_program.py` (the oracle), by
`test_decoder_gem.py` (the C interpreter) and by the fuzz campaign's seed corpus
(`tools/c/fuzz_streampack.sh`), so the three hold the same programs.
"""

from __future__ import annotations

import dataclasses
import struct
import tempfile
from pathlib import Path

from bcir.frontends.models.decoder_program import DecoderProgram, decoder_program
from bcir.frontends.models.hf_ingest import spec_from_config, weights_from_tensors
from bcir.frontends.models.weights_io import read_q8_decoder, write_q8_decoder
from bcir.gem.streampack import hydrate_pipelined
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.tests.test_model_ingest import _CONFIG, _hf_tensors

HASHES = {"model": "00" * 32, "config": "11" * 32, "tokenizer": "22" * 32}
TOKENS = {"bos": 1, "eos": 2, "pad": 0, "context_length": 128}
PROMPT = [1, 2, 3]
MAX_NEW = 4


#: A wider decoder of the same layout. At the fixture's width every feed-forward contribution
#: is far below the residual's last bit, so a kernel that rounds differently (a product
#: reassociated) can still reproduce the logits bit for bit; at this width it cannot.
WIDE = {"hidden_size": 16, "intermediate_size": 48}


def write_model(path: Path, *, tied: bool, wide: bool = False) -> None:
    """The fixture decoder as a BCIRQ8 artifact (tied or untied head; `wide`: WIDE)."""
    config = dict(_CONFIG, tie_word_embeddings=tied, **(WIDE if wide else {}))
    spec = spec_from_config(config)
    prepared = {name: ("F32", s, v) for name, (s, v) in _hf_tensors(config).items()}
    write_q8_decoder(
        path, spec, weights_from_tensors(spec, prepared), source_hashes=HASHES, tokenizer_ids=TOKENS
    )


def model(*, tied: bool, wide: bool = False):
    """(spec, dequantized weights, artifact bytes): what both interpreters execute."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.bcirq8"
        write_model(path, tied=tied, wide=wide)
        spec, weights, _ = read_q8_decoder(path)
        return spec, weights, path.read_bytes()


def program(spec, prompt_len: int = len(PROMPT), max_new: int = MAX_NEW) -> DecoderProgram:
    return decoder_program(spec, prompt_len, max_new)


def planned_pack(prog: DecoderProgram, *, depth: int = 2):
    """The program planned by K_BCIR and hydrated as a pipeline (double-buffer prefetches)."""
    result = optimize(prog.module, TargetProfile.x86_avx512(), Theta.cool())
    return hydrate_pipelined(prog.module, result, depth=depth), result


def claim_at(pack, op: str, offset: int | None = None, nth: int = 0) -> int:
    """The claim id of the nth segment running `op` (at block offset `offset`, if given)."""
    hits = [
        seg.claim_id
        for seg, block in zip(pack.segments, pack.blocks)
        if seg.opcode == op and (offset is None or block.base == offset)
    ]
    return hits[nth]


def _copy(pack):
    return dataclasses.replace(
        pack,
        segments=list(pack.segments),
        prefetches=list(pack.prefetches),
        blocks=list(pack.blocks),
        trace_notes=list(pack.trace_notes),
    )


def edit_claim(pack, claim_id: int, *, block: dict | None = None, **segment):
    """A copy of `pack` with one claim's segment fields (and block fields) replaced."""
    p = _copy(pack)
    i = next(k for k, seg in enumerate(p.segments) if seg.claim_id == claim_id)
    p.segments[i] = dataclasses.replace(p.segments[i], **segment)
    if block:
        p.blocks[i] = dataclasses.replace(p.blocks[i], **block)
    return p


def keep_claims(pack, keep):
    """A copy of `pack` with only the segments `keep(segment)` accepts: the others' segments,
    blocks and trace notes gone, the kept claims renumbered densely in order (the
    interpreters' claim table is indexed by id)."""
    p = _copy(pack)
    kept = [k for k, seg in enumerate(p.segments) if keep(seg)]
    remap = {old: new for new, old in enumerate(sorted(p.segments[k].claim_id for k in kept))}
    p.segments = [
        dataclasses.replace(p.segments[k], claim_id=remap[p.segments[k].claim_id]) for k in kept
    ]
    p.blocks = [p.blocks[k] for k in kept]
    p.trace_notes = [
        dataclasses.replace(t, claim_id=remap[t.claim_id])
        for t in p.trace_notes
        if t.claim_id in remap
    ]
    return p


def drop_claim(pack, claim_id: int):
    """A copy of `pack` without one claim (keep_claims' renumbering)."""
    return keep_claims(pack, lambda seg: seg.claim_id != claim_id)


def layerless(pack, final_gain_rid: int):
    """The program with every layer's claims gone: embeddings, final norms, heads, argmaxes.
    Still a program the claim checks accept -- and one whose embeddings only the position
    order check keeps in order, since no RoPE, cache or attention claim names a position."""
    return keep_claims(
        pack,
        lambda seg: (
            seg.opcode in ("dec.embed", "dec.head", "dec.argmax")
            or (seg.opcode == "dec.rmsnorm" and seg.reads[1] == final_gain_rid)
        ),
    )


def swap_positions(pack, a: int, b: int):
    """A copy of `pack` in which claims `a` and `b` trade places in the segment stream (their
    blocks with them). GEM ranks phases by first appearance in the pack, so this -- not a
    trade of phase labels -- is what changes the dispatch order."""
    p = _copy(pack)
    i = next(k for k, seg in enumerate(p.segments) if seg.claim_id == a)
    j = next(k for k, seg in enumerate(p.segments) if seg.claim_id == b)
    p.segments[i], p.segments[j] = p.segments[j], p.segments[i]
    p.blocks[i], p.blocks[j] = p.blocks[j], p.blocks[i]
    return p


def insert_copy(pack, claim_id: int, after: int):
    """A copy of `pack` with a duplicate of claim `claim_id` -- a new claim id after the last,
    a phase of its own, a trace note -- placed in the segment stream right after claim
    `after`, so GEM dispatches it between `after` and whatever followed it."""
    p = _copy(pack)
    i = next(k for k, seg in enumerate(p.segments) if seg.claim_id == claim_id)
    j = next(k for k, seg in enumerate(p.segments) if seg.claim_id == after)
    new_id = len(p.segments)
    phase = max(seg.phase_id for seg in p.segments) + 1
    segment = dataclasses.replace(
        p.segments[i], name=f"seg{new_id}", claim_id=new_id, phase_id=phase
    )
    p.segments.insert(j + 1, segment)
    p.blocks.insert(j + 1, p.blocks[i])
    p.trace_notes.append(dataclasses.replace(p.trace_notes[0], claim_id=new_id))
    return p


def fuzz_seed(model_bytes: bytes, pack_bytes: bytes) -> bytes:
    """One input of runtime/c/fuzz_decoder_gem.c: `[u32 LE model length][model][pack]`."""
    return struct.pack("<I", len(model_bytes)) + model_bytes + pack_bytes
