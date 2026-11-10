"""MC1 StreamPack disassembler/listing/hexdump tests."""

from contextlib import redirect_stderr, redirect_stdout
import io
import struct
import tempfile
from pathlib import Path
import zlib

from bcir.abi import AbiError, decode, encode, inspect_stream_pack
from bcir.abi.streampack_tool import _read_bounded, format_hexdump, format_listing, main
from bcir.examples import vector_add
from bcir.gem import hydrate
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta


def _blob() -> bytes:
    module = vector_add(32)
    pack = hydrate(module, optimize(module, TargetProfile.x86_avx512(), Theta.cool()))
    return encode(pack)


def test_wire_spans_partition_the_validated_artifact_exactly():
    blob = _blob()
    info = inspect_stream_pack(blob)
    assert info.spans[0].kind == "header" and info.spans[-1].kind == "crc32"
    assert info.spans[0].offset == 0 and info.spans[-1].end == len(blob)
    assert all(left.end == right.offset for left, right in zip(info.spans, info.spans[1:]))
    assert sum(span.length for span in info.spans) == len(blob)


def test_listing_is_deterministic_and_contains_execution_and_provenance_fields():
    blob = _blob()
    listing = format_listing(blob)
    assert listing == format_listing(blob)
    assert listing.startswith("BSPK v4 bytes=")  # a hydrated pack carries the v4 vector
    assert "segments:" in listing and "lane=u width=16" in listing
    assert "rd=[r10,r11] wr=[r12]" in listing
    assert "crc32 @" in listing and listing.endswith(" valid\n")


def test_listing_shows_the_per_resource_generation_vector():
    # v4 (R11 per resource): one line per declared resource in RID order, at the record's
    # wire span; a vector-less (v1-v3) pack lists an empty section, never a fabricated one.
    blob = _blob()
    listing = format_listing(blob)
    assert "generations=3" in listing
    vector = listing.split("generation vector:\n", 1)[1].split("crc32 @", 1)[0].splitlines()
    assert [line.split()[1] for line in vector] == ["r10:", "r11:", "r12:"]
    assert all(line.split()[2:] == ["map=0", "data=0"] for line in vector)
    assert all("+12 " in line for line in vector)  # 12 bytes per record
    assert "# generation[0] 'rid10'" in format_hexdump(blob)

    from bcir.gem.streampack import LaneSegment, StreamPack, TraceNote
    from bcir.model import Lane

    v1 = StreamPack(source_plan="plan0", map_gen=3, data_gen=2)
    v1.segments.append(LaneSegment("seg0", 1000, 0, Lane.U, 16, "f32.add", (10, 11), (12,), ""))
    v1.trace_notes.append(TraceNote(claim_id=1000))
    v1_listing = format_listing(encode(v1))
    assert v1_listing.startswith("BSPK v1 bytes=")
    assert "generations=0" in v1_listing
    assert "generation vector:\ncrc32 @" in v1_listing


def test_hexdump_is_record_delimited_and_covers_exact_bytes():
    blob = _blob()
    dump = format_hexdump(blob, width=8)
    assert "# header 'header' offset=0x00000000 length=64" in dump
    assert "# segment[0] 'seg0'" in dump
    assert "# crc32 'crc32'" in dump
    byte_tokens = []
    for line in dump.splitlines():
        if not line or line.startswith("#"):
            continue
        byte_tokens.extend(line.split("  ", 2)[1].strip().split())
    assert bytes(int(token, 16) for token in byte_tokens) == blob


def test_decoder_rejects_crc_valid_trailing_body_bytes():
    blob = _blob()
    body = blob[:-4] + b"\x00"
    malformed = body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)
    try:
        decode(malformed)
        assert False, "expected a trailing-byte refusal"
    except AbiError as exc:
        assert "trailing body bytes" in str(exc)


def test_cli_disassembles_and_bounds_input_size():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.bspk"
        path.write_bytes(_blob())
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            assert main(["dis", str(path)]) == 0
        assert "segments:" in stdout.getvalue()

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            assert main(["--max-bytes", "68", "dis", str(path)]) == 2
        assert "exceeds --max-bytes" in stderr.getvalue()


def test_bounded_reader_rechecks_actual_bytes_after_stat():
    class BoundedReader:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, limit):
            assert limit == 69
            return bytes(69)

    class GrowingInput:
        class Stat:
            st_size = 68

        def stat(self):
            return self.Stat()

        def open(self, mode):
            assert mode == "rb"
            return BoundedReader()

    try:
        _read_bounded(GrowingInput(), 68)  # type: ignore[arg-type]
        assert False, "expected concurrent-growth refusal"
    except AbiError as exc:
        assert "grew to 69 bytes" in str(exc)


def test_quoted_strings_do_not_depend_on_the_interpreters_unicode_database():
    # repr keeps a printable non-ASCII character literal, and which are printable is the
    # interpreter's Unicode database: U+1FAE8 (Unicode 15) printed as itself under Python 3.15 and
    # escaped under 3.11. ascii() escapes every code point past U+007E, and agrees with repr on
    # every ASCII string, so the listing is one text on every interpreter.
    from bcir.gem.streampack import LaneSegment, StreamPack, TraceNote
    from bcir.model import Lane

    pack = StreamPack(source_plan="\U0001fae8͸\xe9 it's")
    pack.segments.append(LaneSegment("sĀ", 1, 0, Lane.U, 4, "op", (), (), fence_before=("͸",)))
    pack.trace_notes.append(TraceNote(1))
    blob = encode(pack)
    listing = format_listing(blob)
    assert 'source_plan "\\U0001fae8\\u0378\\xe9 it\'s"\n' in listing
    assert "    fences before=['\\u0378'] after=[]\n" in listing
    assert "  sĀ: op " in listing  # unquoted fields stay the record's own text
    assert "# segment[0] 's\\u0100' offset=" in format_hexdump(blob)
    assert (
        format_listing(encode(StreamPack(source_plan="plan0"))).count("source_plan 'plan0'\n") == 1
    )


def test_cli_writes_utf8_octets_whatever_the_console():
    from bcir.gem.streampack import LaneSegment, StreamPack
    from bcir.model import Lane
    import sys

    pack = StreamPack(source_plan="p")
    pack.segments.append(LaneSegment("s\xe9g", 1, 0, Lane.U, 4, "Ā", (), ()))
    blob = encode(pack)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.bspk"
        path.write_bytes(blob)
        raw = io.BytesIO()
        console = io.TextIOWrapper(raw, encoding="ascii", newline="\r\n")
        saved, sys.stdout = sys.stdout, console
        try:
            assert main(["dis", str(path)]) == 0
        finally:
            sys.stdout = saved
        assert raw.getvalue() == format_listing(blob).encode("utf-8")  # no \r, no encode error
