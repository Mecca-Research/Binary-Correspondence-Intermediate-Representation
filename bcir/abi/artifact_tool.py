"""Validated listing, selection, extraction, and disassembly for BCAB artifacts.

The tool validates the complete bundle before displaying or extracting a byte.
Payload disassembly remains delegated to the resident format toolchain; BCAB is
a container and compatibility contract, not a replacement object-file parser.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from .artifact_bundle import (
    HEADER_SIZE,
    MAX_BUNDLE_BYTES,
    ArtifactBundleInspection,
    ArtifactFormat,
    ArtifactKind,
    BundleError,
    CompatibilityEnvelope,
    Endianness,
    inspect_bundle,
    select_variant,
)
from .streampack_tool import format_listing as format_streampack_listing


DEFAULT_MAX_BYTES = MAX_BUNDLE_BYTES
DEFAULT_TOOL_OUTPUT_BYTES = 8 * 1024 * 1024


def _flags(variant) -> str:
    values = []
    if variant.executable:
        values.append("exec")
    if variant.r12_attested:
        values.append("r12")
    if variant.portable:
        values.append("portable")
    if variant.debug:
        values.append("debug")
    return ",".join(values) or "-"


def format_listing(data: bytes) -> str:
    """Return a deterministic directory listing for a fully validated bundle."""
    info = inspect_bundle(data)
    bundle = info.bundle
    lines = [
        f"BCAB v1 bytes={info.length} entries={len(bundle.variants)}",
        f"sha256={info.artifact_sha256} embedded_sha256={info.embedded_sha256}",
        f"crc32 body=0x{info.body_crc32:08x} header=0x{info.header_crc32:08x}",
        f"provenance=0x{bundle.provenance_digest:016x} generation={bundle.generation} "
        f"root={bundle.root_variant_id or '-'} default={bundle.default_variant_id or '-'}",
        "variants:",
    ]
    payload_spans = {span.index: span for span in info.spans if span.kind == "payload"}
    for index, variant in enumerate(bundle.variants):
        span = payload_spans[index]
        features = ",".join(variant.required_features) or "-"
        prohibited = ",".join(variant.prohibited_features) or "-"
        lines.append(
            f"  [{index}] {variant.variant_id} kind={variant.kind.name.lower()} "
            f"format={variant.format.name.lower()} @{span.offset:08x}+{span.length} "
            f"priority={variant.priority} flags={_flags(variant)}"
        )
        lines.append(
            f"      target triple={variant.triple or '-'} arch={variant.architecture or '-'} "
            f"os_abi={variant.os_abi or '-'} channel={variant.channel or '-'} "
            f"endian={variant.endianness.name.lower()} bits={variant.pointer_bits or '-'} "
            f"machine={variant.e_machine or '-'}"
        )
        lines.append(
            f"      features require={features} prohibit={prohibited} "
            f"entry={variant.entry_symbol or '-'} cal_gen={variant.cal_gen}"
        )
        lines.append(
            f"      crc32=0x{variant.payload_crc32:08x} sha256={variant.payload_sha256} "
            f"manifest={variant.target_manifest_sha256 or '-'}"
        )
    return "\n".join(lines) + "\n"


def format_hexdump(data: bytes, *, width: int = 16) -> str:
    """Return a wire-span-delimited dump of a fully validated bundle."""
    if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= 64:
        raise BundleError("hexdump width must be in [1, 64]")
    info = inspect_bundle(data)
    lines: list[str] = []
    for span in sorted(info.spans, key=lambda item: item.offset):
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


def _text_listing(payload: bytes) -> str:
    return payload.decode("utf-8") + ("" if payload.endswith(b"\n") else "\n")


def _find_tool(*names: str) -> str | None:
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def _run_tool(command: list[str], *, maximum: int = DEFAULT_TOOL_OUTPUT_BYTES) -> str:
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise BundleError("disassembler output limit must be a positive integer")
    process = None
    try:
        with tempfile.TemporaryFile() as captured:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=captured,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 30.0
            while process.poll() is None:
                if os.fstat(captured.fileno()).st_size > maximum:
                    raise BundleError(f"disassembler output exceeds {maximum} bytes")
                if time.monotonic() >= deadline:
                    raise BundleError("disassembler timed out after 30 seconds")
                time.sleep(0.02)
            size = os.fstat(captured.fileno()).st_size
            if size > maximum:
                raise BundleError(f"disassembler output exceeds {maximum} bytes")
            captured.seek(0)
            output = captured.read(maximum + 1)
            returncode = process.returncode
    except BundleError:
        raise
    except OSError as exc:
        raise BundleError(f"disassembler failed: {exc}") from exc
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired as exc:
                raise BundleError("disassembler could not be terminated") from exc
    text = output.decode("utf-8", errors="replace")
    if returncode:
        raise BundleError(f"disassembler exited {returncode}: {text.strip()}")
    return text + ("" if text.endswith("\n") else "\n")


def disassemble_variant(variant) -> str:
    """Disassemble one validated payload, or fail honestly if no tool exists."""
    if variant.kind == ArtifactKind.STREAM_PACK:
        return format_streampack_listing(variant.payload)
    if variant.format == ArtifactFormat.TEXT:
        return _text_listing(variant.payload)
    suffix = {
        ArtifactFormat.ELF: ".o",
        ArtifactFormat.COFF: ".obj",
        ArtifactFormat.MACHO: ".o",
        ArtifactFormat.ARCHIVE: ".a",
        ArtifactFormat.WASM: ".wasm",
        ArtifactFormat.LLVM_BITCODE: ".bc",
        ArtifactFormat.SPIRV: ".spv",
        ArtifactFormat.JVM_CLASS: ".class",
        ArtifactFormat.PE: ".exe",
        ArtifactFormat.RAW: ".bin",
    }.get(variant.format, ".bin")
    with tempfile.TemporaryDirectory(prefix="bcir-bundle-dis-") as temporary:
        path = os.path.join(temporary, "artifact" + suffix)
        with open(path, "wb") as stream:
            stream.write(variant.payload)
        if variant.format in (
            ArtifactFormat.ELF,
            ArtifactFormat.COFF,
            ArtifactFormat.MACHO,
            ArtifactFormat.WASM,
            ArtifactFormat.ARCHIVE,
            ArtifactFormat.PE,
        ):
            tool = _find_tool("llvm-objdump", "objdump")
            if not tool:
                raise BundleError("no llvm-objdump/objdump is available for this payload")
            if variant.format == ArtifactFormat.ARCHIVE:
                return _run_tool([tool, "-a", "-f", path])
            return _run_tool([tool, "-d", "-r", "-h", path])
        if variant.format == ArtifactFormat.LLVM_BITCODE:
            tool = _find_tool("llvm-dis")
            if not tool:
                raise BundleError("llvm-dis is not available for LLVM bitcode")
            return _run_tool([tool, "-o", "-", path])
        if variant.format == ArtifactFormat.SPIRV:
            tool = _find_tool("spirv-dis")
            if not tool:
                raise BundleError("spirv-dis is not available for SPIR-V")
            return _run_tool([tool, path, "-o", "-"])
        if variant.format == ArtifactFormat.JVM_CLASS:
            tool = _find_tool("javap")
            if not tool:
                raise BundleError("javap is not available for JVM bytecode")
            return _run_tool([tool, "-c", "-verbose", path])
    raise BundleError(f"no disassembler is defined for {variant.format.name}")


def format_mnemonics(variant) -> str:
    """Return a compact mnemonic view without claiming to normalize backend ISAs."""
    if variant.kind == ArtifactKind.STREAM_PACK:
        from .streampack_abi import decode

        return "\n".join(segment.opcode for segment in decode(variant.payload).segments) + "\n"
    if variant.format == ArtifactFormat.TEXT:
        mnemonics = []
        for raw in _text_listing(variant.payload).splitlines():
            line = raw.strip()
            if not line or line.startswith(("#", ";", "//", ".")):
                continue
            mnemonics.append(line.split(None, 1)[0])
        return "\n".join(mnemonics) + ("\n" if mnemonics else "")
    return disassemble_variant(variant)


def _read_bounded(
    path: Path,
    maximum: int,
    *,
    minimum: int = HEADER_SIZE,
    hard_limit: int = MAX_BUNDLE_BYTES,
) -> bytes:
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or not isinstance(minimum, int)
        or not isinstance(hard_limit, int)
        or minimum < 1
        or hard_limit < minimum
        or maximum < minimum
        or maximum > hard_limit
    ):
        raise BundleError(f"--max-bytes must be in [{minimum}, {hard_limit}]")
    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if size > maximum:
                raise BundleError(f"input is {size} bytes, exceeds --max-bytes={maximum}")
            data = stream.read(maximum + 1)
    except OSError as exc:
        raise BundleError(f"cannot read {path}: {exc}") from exc
    if len(data) > maximum or len(data) != size:
        raise BundleError("input changed, grew, or was truncated while reading")
    return data


def _atomic_write(path: Path, payload: bytes) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _enum_set(values, enum_type):
    try:
        return frozenset(enum_type[value.upper().replace("-", "_")] for value in values)
    except KeyError as exc:
        raise BundleError(f"unknown {enum_type.__name__} {exc.args[0]!r}") from exc


def _envelope(args) -> CompatibilityEnvelope:
    endian = {"neutral": Endianness.NEUTRAL, "little": Endianness.LITTLE, "big": Endianness.BIG}[
        args.endianness
    ]
    return CompatibilityEnvelope(
        triple=args.triple,
        architecture=args.architecture,
        os_abi=args.os_abi,
        channel=args.channel,
        features=frozenset(args.feature),
        accepted_kinds=_enum_set(args.kind, ArtifactKind),
        accepted_formats=_enum_set(args.format, ArtifactFormat),
        endianness=endian,
        pointer_bits=args.pointer_bits,
        e_machine=args.machine,
        target_manifest_sha256=args.manifest_sha256,
        cal_gen=args.cal_gen,
        require_r12=not args.allow_unattested,
        allow_debug=args.allow_debug,
    )


def _add_selector_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--id", default="", help="select this exact variant ID")
    parser.add_argument("--triple", default="")
    parser.add_argument("--architecture", default="")
    parser.add_argument("--os-abi", default="")
    parser.add_argument("--channel", default="")
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--kind", action="append", default=[])
    parser.add_argument("--format", action="append", default=[])
    parser.add_argument("--endianness", choices=("neutral", "little", "big"), default="neutral")
    parser.add_argument("--pointer-bits", type=int, default=0)
    parser.add_argument("--machine", type=int, default=0)
    parser.add_argument("--manifest-sha256", default="")
    parser.add_argument("--cal-gen", type=int)
    parser.add_argument("--allow-unattested", action="store_true")
    parser.add_argument("--allow-debug", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bcir-bundle", description="validate, inspect, and select BCIR artifact bundles"
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="refuse larger inputs (default: the native or ASN.1 format limit)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("list", "print the validated bundle directory"),
        ("hexdump", "print a wire-span-delimited hexadecimal dump"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("file", type=Path)
        if name == "hexdump":
            command.add_argument("--width", type=int, default=16)
    for name, help_text in (
        ("extract", "extract one validated payload"),
        ("select", "select a compatible payload and report its ID"),
        ("dis", "disassemble one validated payload"),
        ("mnemonics", "print one payload's mnemonic view"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("file", type=Path)
        _add_selector_options(command)
        if name == "extract":
            command.add_argument("output", type=Path)
    for name, help_text in (
        ("to-der", "project a native BCAB to canonical ASN.1 DER"),
        ("from-der", "reconstruct canonical native BCAB from ASN.1 DER/BER"),
        ("to-oer", "project a native BCAB to CANONICAL-OER"),
        ("from-oer", "reconstruct canonical native BCAB from BASIC/CANONICAL-OER"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("file", type=Path)
        command.add_argument("output", type=Path)
        if name == "from-der":
            command.add_argument(
                "--ber",
                action="store_true",
                help="admit BER sender choices; output is still canonical native BCAB",
            )
        if name == "from-oer":
            command.add_argument(
                "--canonical",
                action="store_true",
                help="require CANONICAL-OER instead of accepting BASIC-OER",
            )
    args = parser.parse_args(argv)
    try:
        projected_input = args.command in ("from-der", "from-oer")
        hard_limit = MAX_BUNDLE_BYTES
        if projected_input:
            # Lazy import preserves the rule that ordinary BCAB tooling does not pull
            # the ASN.1 implementation onto the dependency-free import path.
            from bcir.asn1.artifact_bundle import MAX_PROJECTION_BYTES

            hard_limit = MAX_PROJECTION_BYTES
        maximum = hard_limit if args.max_bytes is None else args.max_bytes
        data = _read_bounded(
            args.file,
            maximum,
            minimum=1 if projected_input else HEADER_SIZE,
            hard_limit=hard_limit,
        )
        if args.command == "list":
            output = format_listing(data)
        elif args.command == "hexdump":
            output = format_hexdump(data, width=args.width)
        elif args.command in ("to-der", "from-der", "to-oer", "from-oer"):
            from bcir.asn1.artifact_bundle import (
                der_to_native,
                native_to_der,
                native_to_oer,
                oer_to_native,
            )
            from bcir.asn1.codec import Strictness

            if args.command == "to-der":
                payload = native_to_der(data)
            elif args.command == "from-der":
                payload = der_to_native(
                    data,
                    strictness=Strictness.BER if args.ber else Strictness.DER,
                )
            elif args.command == "to-oer":
                payload = native_to_oer(data)
            else:
                payload = oer_to_native(data, canonical=args.canonical)
            _atomic_write(args.output, payload)
            output = f"{args.command} {len(payload)} bytes\n"
        else:
            info: ArtifactBundleInspection = inspect_bundle(data)
            envelope = None if args.id else _envelope(args)
            variant = select_variant(info.bundle, envelope, variant_id=args.id)
            if args.command == "extract":
                _atomic_write(args.output, variant.payload)
                output = (
                    f"{variant.variant_id} {len(variant.payload)} bytes {variant.payload_sha256}\n"
                )
            elif args.command == "select":
                output = f"{variant.variant_id}\n"
            elif args.command == "dis":
                output = disassemble_variant(variant)
            else:
                output = format_mnemonics(variant)
    except (BundleError, OSError, ValueError) as exc:
        print(f"bcir-bundle: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
