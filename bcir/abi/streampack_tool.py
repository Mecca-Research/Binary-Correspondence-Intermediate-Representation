"""Validated StreamPack disassembly/listing and annotated hexadecimal dump (MC1).

The tool never interprets unverified bytes: `inspect_stream_pack` first applies the frozen codec's
magic/version/CRC/range/trailing-byte checks. Record boundaries come from the codec's own writers,
so append-only ABI changes cannot silently desynchronize a duplicate parser.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .streampack_abi import AbiError, StreamPackInspection, WireSpan, inspect_stream_pack

DEFAULT_MAX_BYTES = 64 * 1024 * 1024


def _span(info: StreamPackInspection, kind: str, index: int | None = None) -> WireSpan:
    for span in info.spans:
        if span.kind == kind and span.index == index:
            return span
    raise AssertionError(f"missing {kind}[{index}] wire span")


def _rids(values: tuple[int, ...]) -> str:
    return "[" + ",".join(f"r{rid}" for rid in values) + "]"


def format_listing(data: bytes) -> str:
    """Return a deterministic, provenance-aware StreamPack assembly listing."""
    info = inspect_stream_pack(data)
    pack = info.pack
    lines = [
        f"BSPK v{info.version} bytes={info.length} flags=0x{info.flags:04x}",
        f"generations topo={pack.topo_gen} map={pack.map_gen} data={pack.data_gen} "
        f"pipeline_depth={pack.pipeline_depth}",
        f"records segments={len(pack.segments)} prefetches={len(pack.prefetches)} "
        f"blocks={len(pack.blocks)} trace={len(pack.trace_notes)}",
        f"source_plan {pack.source_plan!r}",
    ]

    traced = {note.claim_id: note for note in pack.trace_notes}
    lines.append("segments:")
    for index, seg in enumerate(pack.segments):
        span = _span(info, "segment", index)
        trace = traced.get(seg.claim_id)
        provenance = (
            "untraced"
            if trace is None
            else (f"src=0x{trace.src_hash:016x} trace=0x{trace.trace_hash:016x}")
        )
        lines.append(
            f"  @{span.offset:08x}+{span.length:<4} {seg.name}: {seg.opcode} "
            f"claim={seg.claim_id} phase={seg.phase_id} lane={seg.lane.name.lower()} "
            f"width={seg.width} rd={_rids(seg.reads)} wr={_rids(seg.writes)} "
            f"dispatch={seg.dispatch} channel={seg.channel} prefetch={seg.prefetch or '-'} "
            f"{provenance}"
        )
        if seg.fence_before or seg.fence_after:
            lines.append(
                f"    fences before={list(seg.fence_before)!r} after={list(seg.fence_after)!r}"
            )

    lines.append("prefetches:")
    for index, pf in enumerate(pack.prefetches):
        span = _span(info, "prefetch", index)
        lines.append(
            f"  @{span.offset:08x}+{span.length:<4} {pf.name}: distance={pf.distance} "
            f"targets={_rids(pf.targets)} hint={pf.hint} pattern={pf.pattern} "
            f"buffers={pf.buffers}"
        )

    lines.append("blocks:")
    for index, block in enumerate(pack.blocks):
        span = _span(info, "block", index)
        lines.append(
            f"  @{span.offset:08x}+{span.length:<4} block{index}: base={block.base} "
            f"count={block.count} strides={list(block.strides)!r}"
        )

    crc_span = _span(info, "crc32")
    lines.append(f"crc32 @{crc_span.offset:08x} = 0x{info.crc32:08x} valid")
    return "\n".join(lines) + "\n"


def format_hexdump(data: bytes, *, width: int = 16) -> str:
    """Return a record-delimited hexadecimal/ASCII dump of a validated StreamPack."""
    if width < 1 or width > 64:
        raise ValueError("hexdump width must be in [1, 64]")
    info = inspect_stream_pack(data)
    lines: list[str] = []
    for span in info.spans:
        index = "" if span.index is None else f"[{span.index}]"
        lines.append(
            f"# {span.kind}{index} {span.name!r} offset=0x{span.offset:08x} length={span.length}"
        )
        for offset in range(span.offset, span.end, width):
            chunk = data[offset : min(offset + width, span.end)]
            hex_bytes = " ".join(f"{byte:02x}" for byte in chunk)
            ascii_bytes = "".join(chr(byte) if 32 <= byte < 127 else "." for byte in chunk)
            lines.append(f"{offset:08x}  {hex_bytes:<{width * 3 - 1}}  |{ascii_bytes}|")
    return "\n".join(lines) + "\n"


def _read_bounded(path: Path, maximum: int) -> bytes:
    if maximum < 68:
        raise ValueError("--max-bytes must be at least 68")
    size = path.stat().st_size
    if size > maximum:
        raise AbiError(f"input is {size} bytes, exceeds --max-bytes={maximum}")
    # The file may grow between stat() and open(). Read at most one byte beyond the
    # declared ceiling so the refusal itself cannot allocate an unbounded replacement.
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise AbiError(
            f"input grew to {len(data)} bytes while reading, exceeds --max-bytes={maximum}"
        )
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bcir-pack", description="validated StreamPack listing and hexadecimal dump"
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help=f"refuse larger inputs (default: {DEFAULT_MAX_BYTES})",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    dis = sub.add_parser("dis", help="print an annotated StreamPack assembly listing")
    dis.add_argument("file", type=Path)
    hexdump = sub.add_parser("hexdump", help="print a record-delimited hex/ASCII dump")
    hexdump.add_argument("file", type=Path)
    hexdump.add_argument("--width", type=int, default=16)
    args = parser.parse_args(argv)

    try:
        data = _read_bounded(args.file, args.max_bytes)
        output = (
            format_listing(data)
            if args.command == "dis"
            else format_hexdump(data, width=args.width)
        )
    except (AbiError, OSError, ValueError) as exc:
        print(f"bcir-pack: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
