"""BCAB v1 wire, selector, backend-adapter, CLI, MLIR, and C parity tests."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch
import zlib

from bcir.abi import (
    ARTIFACT_ENTRY_SIZE,
    ARTIFACT_HEADER_SIZE,
    ARTIFACT_MAGIC,
    ARTIFACT_VERSION,
    ArtifactBundle,
    ArtifactFormat,
    ArtifactKind,
    ArtifactVariant,
    BundleError,
    CompatibilityEnvelope,
    Endianness,
    compatibility_sha256,
    decode,
    decode_bundle,
    encode,
    encode_bundle,
    host_envelope,
    inspect_bundle,
    inspect_stream_pack,
    is_compatible,
    read_bundle,
    select_variant,
    write_bundle,
)
from bcir.abi.artifact_tool import _run_tool, format_hexdump, format_listing, main
from bcir.codegen.artifact_bundle import ArtifactBundleBuilder
from bcir.codegen.codegen import CodegenResult
from bcir.examples import vector_add
from bcir.gem import hydrate
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.lower.artifact_bundle import bundle_to_mlir, selection_to_mlir
from bcir.lower.jvm_class import JVMClassError, build_jvm_class
from bcir.lower.stackify import BinOp, Const, stackify


def _pack_bytes() -> bytes:
    module = vector_add(8)
    return encode(hydrate(module, optimize(module, TargetProfile.x86_avx512(), Theta.cool())))


def _elf(
    machine: int = 62, *, bits: int = 64, endian: Endianness = Endianness.LITTLE, elf_type: int = 1
) -> bytes:
    payload = bytearray(20)
    payload[:4] = b"\x7fELF"
    payload[4] = 1 if bits == 32 else 2
    payload[5] = 1 if endian == Endianness.LITTLE else 2
    payload[6] = 1
    struct.pack_into("<H" if endian == Endianness.LITTLE else ">H", payload, 16, elf_type)
    struct.pack_into("<H" if endian == Endianness.LITTLE else ">H", payload, 18, machine)
    return bytes(payload)


def _three_bundle() -> ArtifactBundle:
    return ArtifactBundle(
        (
            ArtifactVariant(
                "00-root",
                ArtifactKind.STREAM_PACK,
                ArtifactFormat.STREAM_PACK,
                _pack_bytes(),
                channel="host",
                portable=True,
            ),
            ArtifactVariant(
                "portable-c",
                ArtifactKind.C_SOURCE,
                ArtifactFormat.TEXT,
                b"int bcir_kernel(void){return 0;}\n",
                portable=True,
            ),
            ArtifactVariant(
                "x86-avx2",
                ArtifactKind.ELF_OBJECT,
                ArtifactFormat.ELF,
                _elf(),
                triple="x86_64-unknown-linux-gnu",
                architecture="x86_64",
                os_abi="linux-gnu",
                channel="host",
                entry_symbol="bcir_kernel",
                required_features=("avx2",),
                endianness=Endianness.LITTLE,
                pointer_bits=64,
                e_machine=62,
                priority=9,
                r12_attested=True,
                executable=True,
            ),
        ),
        "00-root",
        "portable-c",
        123,
        7,
    )


def _host_envelope(**changes) -> CompatibilityEnvelope:
    values = dict(
        triple="x86_64-unknown-linux-gnu",
        architecture="x86_64",
        os_abi="linux-gnu",
        channel="host",
        features=frozenset(("avx2",)),
        accepted_kinds=frozenset((ArtifactKind.ELF_OBJECT,)),
        accepted_formats=frozenset((ArtifactFormat.ELF,)),
        endianness=Endianness.LITTLE,
        pointer_bits=64,
        e_machine=62,
    )
    values.update(changes)
    return CompatibilityEnvelope(**values)


def _reseal(blob: bytearray) -> bytes:
    struct.pack_into("<I", blob, 76, zlib.crc32(blob[128:]) & 0xFFFFFFFF)
    struct.pack_into("<I", blob, 80, 0)
    blob[84:116] = bytes(32)
    blob[84:116] = hashlib.sha256(blob).digest()
    struct.pack_into("<I", blob, 80, zlib.crc32(blob[:128]) & 0xFFFFFFFF)
    return bytes(blob)


def _must_reject(blob: bytes, needle: str = "") -> None:
    try:
        decode_bundle(blob)
        assert False, "expected malformed BCAB refusal"
    except BundleError as exc:
        if needle:
            assert needle in str(exc), (needle, str(exc))


def _semantic_streampack_spoof() -> bytes:
    """Return a fully re-sealed BCAB whose root has a dangling trace binding."""
    original = bytearray(encode_bundle(_three_bundle()))
    root_span = next(
        span
        for span in inspect_bundle(bytes(original)).spans
        if span.kind == "payload" and span.name == "00-root"
    )
    pack = bytearray(original[root_span.offset : root_span.end])
    trace = next(span for span in inspect_stream_pack(bytes(pack)).spans if span.kind == "trace")
    struct.pack_into("<Q", pack, trace.offset, 0xFFFFFFFFFFFFFFFF)
    struct.pack_into("<I", pack, len(pack) - 4, zlib.crc32(pack[:-4]) & 0xFFFFFFFF)
    original[root_span.offset : root_span.end] = pack
    entry = ARTIFACT_HEADER_SIZE
    struct.pack_into("<I", original, entry + 48, zlib.crc32(pack) & 0xFFFFFFFF)
    original[entry + 56 : entry + 88] = hashlib.sha256(pack).digest()
    return _reseal(original)


def test_bcab_constants_deterministic_roundtrip_and_exact_span_partition():
    bundle = _three_bundle()
    first = encode_bundle(bundle)
    second = encode_bundle(bundle)
    assert first == second
    assert first[:4] == ARTIFACT_MAGIC == b"BCAB"
    assert struct.unpack_from("<H", first, 4)[0] == ARTIFACT_VERSION == 1
    assert ARTIFACT_HEADER_SIZE == 128 and ARTIFACT_ENTRY_SIZE == 448
    assert decode_bundle(first) == bundle
    assert encode_bundle(decode_bundle(first)) == first
    info = inspect_bundle(first)
    ordered = sorted(info.spans, key=lambda span: span.offset)
    assert ordered[0].offset == 0 and ordered[-1].end == len(first)
    assert all(left.end == right.offset for left, right in zip(ordered, ordered[1:]))
    assert sum(span.length for span in ordered) == len(first)
    assert info.artifact_sha256 == hashlib.sha256(first).hexdigest()


def test_constructor_rejects_noncanonical_metadata_and_payload_identity():
    base = dict(
        variant_id="x",
        kind=ArtifactKind.C_SOURCE,
        format=ArtifactFormat.TEXT,
        payload=b"int x;\n",
    )
    cases = [
        {**base, "variant_id": " x"},
        {**base, "required_features": ("z", "a")},
        {**base, "required_features": ("a",), "prohibited_features": ("a",)},
        {**base, "payload": b"bad\x00text"},
        {**base, "target_manifest_sha256": "0" * 64},
        {**base, "kind": ArtifactKind.WASM, "format": ArtifactFormat.WASM},
        {**base, "pointer_bits": 16},
        {**base, "target_manifest_sha256": "A" * 64},
        {
            "variant_id": "native-no-machine",
            "kind": ArtifactKind.ELF_OBJECT,
            "format": ArtifactFormat.ELF,
            "payload": _elf(),
            "endianness": Endianness.LITTLE,
            "pointer_bits": 64,
        },
        {
            "variant_id": "unmarked-executable",
            "kind": ArtifactKind.ELF_EXECUTABLE,
            "format": ArtifactFormat.ELF,
            "payload": _elf(elf_type=2),
            "endianness": Endianness.LITTLE,
            "pointer_bits": 64,
            "e_machine": 62,
            "r12_attested": True,
        },
    ]
    for values in cases:
        try:
            ArtifactVariant(**values)
            assert False, values
        except BundleError:
            pass
    variants = _three_bundle().variants
    try:
        ArtifactBundle(tuple(reversed(variants)))
        assert False, "unsorted directory must fail"
    except BundleError:
        pass
    try:
        ArtifactBundle(
            (
                ArtifactVariant(
                    "run",
                    ArtifactKind.RAW_BINARY,
                    ArtifactFormat.RAW,
                    b"run",
                    architecture="test",
                    channel="device",
                    executable=True,
                ),
            ),
            default_variant_id="run",
        )
        assert False, "an executable default without R12 attestation must fail"
    except BundleError:
        pass


def test_encode_preflights_wire_limit_before_materializing_directory_or_payload_copies():
    bundle = ArtifactBundle(
        (
            ArtifactVariant(
                "raw",
                ArtifactKind.RAW_BINARY,
                ArtifactFormat.RAW,
                b"x" * 32,
            ),
        )
    )
    with (
        patch("bcir.abi.artifact_bundle.MAX_BUNDLE_BYTES", ARTIFACT_HEADER_SIZE),
        patch(
            "bcir.abi.artifact_bundle._pack_entry",
            side_effect=AssertionError("must not materialize"),
        ),
    ):
        try:
            encode_bundle(bundle)
            assert False, "oversized projected wire must fail during preflight"
        except BundleError as exc:
            assert "wire limit" in str(exc)


def test_compatibility_inputs_and_explicit_ids_are_strictly_bounded():
    for values in (
        {"features": 7},
        {"features": frozenset(("avx2", 7))},
        {"target_manifest_sha256": "0" * 64},
    ):
        try:
            CompatibilityEnvelope(**values)
            assert False, values
        except BundleError:
            pass
    for variant_id in ("x" * 48, b"x86-avx2"):
        try:
            select_variant(_three_bundle(), _host_envelope(), variant_id=variant_id)
            assert False, variant_id
        except BundleError:
            pass


def test_payload_identity_validation_covers_standard_backend_formats():
    jvm = build_jvm_class("BcirBundleTest", ("ldc 2.0f",))
    coff = bytearray(20)
    struct.pack_into("<H", coff, 0, 0x8664)
    macho = bytearray(28)
    macho[:4] = b"\xcf\xfa\xed\xfe"
    struct.pack_into("<I", macho, 4, 0x01000007)
    struct.pack_into("<I", macho, 12, 1)
    pe = bytearray(0x80)
    pe[:2] = b"MZ"
    struct.pack_into("<I", pe, 0x3C, 0x40)
    pe[0x40:0x44] = b"PE\0\0"
    struct.pack_into("<H", pe, 0x44, 0x8664)
    struct.pack_into("<H", pe, 0x54, 2)
    struct.pack_into("<H", pe, 0x58, 0x20B)
    payloads = (
        ArtifactVariant("archive", ArtifactKind.ARCHIVE, ArtifactFormat.ARCHIVE, b"!<arch>\n"),
        ArtifactVariant(
            "bitcode", ArtifactKind.LLVM_BITCODE, ArtifactFormat.LLVM_BITCODE, b"BC\xc0\xde"
        ),
        ArtifactVariant(
            "coff",
            ArtifactKind.COFF_OBJECT,
            ArtifactFormat.COFF,
            bytes(coff),
            endianness=Endianness.LITTLE,
            pointer_bits=64,
            e_machine=0x8664,
        ),
        ArtifactVariant(
            "jvm", ArtifactKind.JVM_CLASS, ArtifactFormat.JVM_CLASS, jvm, portable=True
        ),
        ArtifactVariant(
            "macho",
            ArtifactKind.MACHO_OBJECT,
            ArtifactFormat.MACHO,
            bytes(macho),
            endianness=Endianness.LITTLE,
            pointer_bits=64,
            e_machine=0x01000007,
        ),
        ArtifactVariant(
            "ptx",
            ArtifactKind.PTX,
            ArtifactFormat.TEXT,
            b".version 7.0\n.target sm_50\nmov.u32 %r1, %r2;\n",
        ),
        ArtifactVariant("raw", ArtifactKind.RAW_BINARY, ArtifactFormat.RAW, b"\x01\x02\x03"),
        ArtifactVariant(
            "pe",
            ArtifactKind.PE_EXECUTABLE,
            ArtifactFormat.PE,
            bytes(pe),
            endianness=Endianness.LITTLE,
            pointer_bits=64,
            e_machine=0x8664,
            r12_attested=True,
            executable=True,
        ),
        ArtifactVariant(
            "spirv", ArtifactKind.SPIRV, ArtifactFormat.SPIRV, b"\x03\x02\x23\x07" + bytes(16)
        ),
        ArtifactVariant(
            "wasm",
            ArtifactKind.WASM,
            ArtifactFormat.WASM,
            b"\x00asm\x01\x00\x00\x00",
            endianness=Endianness.LITTLE,
            pointer_bits=32,
            portable=True,
        ),
    )
    bundle = ArtifactBundle(tuple(sorted(payloads, key=lambda variant: variant.variant_id)))
    assert decode_bundle(encode_bundle(bundle)) == bundle


def test_selector_is_fail_closed_and_deterministic():
    bundle = _three_bundle()
    envelope = _host_envelope()
    selected = select_variant(bundle, envelope)
    assert selected.variant_id == "x86-avx2"
    assert is_compatible(selected, envelope)
    assert select_variant(bundle).variant_id == "portable-c"
    assert compatibility_sha256(envelope) == compatibility_sha256(envelope)
    rejects = (
        _host_envelope(features=frozenset()),
        _host_envelope(triple=""),
        _host_envelope(pointer_bits=32),
        _host_envelope(e_machine=183),
        _host_envelope(channel="sycl_spirv"),
        _host_envelope(
            require_r12=True, allow_debug=False, accepted_kinds=frozenset((ArtifactKind.WASM,))
        ),
    )
    for incompatible in rejects:
        try:
            select_variant(bundle, incompatible)
            assert False, incompatible
        except BundleError:
            pass
    try:
        select_variant(bundle, envelope, variant_id="portable-c")
        assert False, "explicit incompatible IDs must not bypass the envelope"
    except BundleError:
        pass


def test_host_envelope_admits_both_windows_native_container_formats():
    with (
        patch("bcir.abi.artifact_bundle.platform.machine", return_value="AMD64"),
        patch("bcir.abi.artifact_bundle.platform.system", return_value="Windows"),
    ):
        envelope = host_envelope()
    assert envelope.triple == "x86_64-pc-windows-msvc"
    assert envelope.e_machine == 0x8664
    assert ArtifactFormat.COFF in envelope.accepted_formats
    assert ArtifactFormat.PE in envelope.accepted_formats
    with (
        patch("bcir.abi.artifact_bundle.platform.machine", return_value="arm64"),
        patch("bcir.abi.artifact_bundle.platform.system", return_value="Darwin"),
    ):
        darwin = host_envelope()
    assert darwin.e_machine == 0x0100000C
    assert darwin.accepted_formats == frozenset(
        (
            ArtifactFormat.MACHO,
            ArtifactFormat.WASM,
            ArtifactFormat.TEXT,
            ArtifactFormat.STREAM_PACK,
            ArtifactFormat.LLVM_BITCODE,
            ArtifactFormat.JVM_CLASS,
            ArtifactFormat.SPIRV,
        )
    )


def test_selector_gates_manifest_generation_debug_and_r12_then_tiebreaks_by_id():
    manifest = "11" * 32
    common = dict(
        kind=ArtifactKind.WASM,
        format=ArtifactFormat.WASM,
        payload=b"\x00asm\x01\x00\x00\x00",
        architecture="wasm32",
        channel="wasm",
        required_features=("simd128",),
        endianness=Endianness.LITTLE,
        pointer_bits=32,
        target_manifest_sha256=manifest,
        cal_gen=4,
        priority=3,
        executable=True,
        portable=True,
    )
    bundle = ArtifactBundle(
        (
            ArtifactVariant("a", r12_attested=True, **common),
            ArtifactVariant("b", r12_attested=True, **common),
            ArtifactVariant(
                "debug",
                r12_attested=True,
                debug=True,
                priority=99,
                **{k: v for k, v in common.items() if k != "priority"},
            ),
            ArtifactVariant(
                "unattested",
                r12_attested=False,
                priority=100,
                **{k: v for k, v in common.items() if k != "priority"},
            ),
        )
    )
    envelope = CompatibilityEnvelope(
        architecture="wasm32",
        channel="wasm",
        features=frozenset(("simd128",)),
        accepted_kinds=frozenset((ArtifactKind.WASM,)),
        endianness=Endianness.LITTLE,
        pointer_bits=32,
        target_manifest_sha256=manifest,
        cal_gen=4,
    )
    assert select_variant(bundle, envelope).variant_id == "a"
    for changed in (
        {"target_manifest_sha256": "22" * 32},
        {"cal_gen": 5},
    ):
        values = {**envelope.__dict__, **changed}
        try:
            select_variant(bundle, CompatibilityEnvelope(**values))
            assert False, changed
        except BundleError:
            pass


def test_wire_rejects_header_body_digest_directory_geometry_and_payload_corruption():
    original = encode_bundle(_three_bundle())
    mutations = []
    blob = bytearray(original)
    blob[80] ^= 1
    mutations.append((bytes(blob), "header CRC"))
    blob = bytearray(original)
    blob[-1] ^= 1
    mutations.append((bytes(blob), "body CRC"))
    blob = bytearray(original)
    blob[84] ^= 1
    struct.pack_into("<I", blob, 80, 0)
    struct.pack_into("<I", blob, 80, zlib.crc32(blob[:128]) & 0xFFFFFFFF)
    mutations.append((bytes(blob), "embedded SHA"))
    blob = bytearray(original)
    blob[116] = 1
    mutations.append((_reseal(blob), "reserved"))
    blob = bytearray(original)
    struct.pack_into("<I", blob, 128 + 52, 1)
    mutations.append((_reseal(blob), "reserved"))
    blob = bytearray(original)
    struct.pack_into("<Q", blob, 128 + 16, len(original) - 8)
    mutations.append((_reseal(blob), "geometry"))
    first_payload = inspect_bundle(original).spans[-3]
    payload_spans = [span for span in inspect_bundle(original).spans if span.kind == "payload"]
    assert payload_spans
    blob = bytearray(original)
    blob[payload_spans[0].offset] ^= 1
    mutations.append((_reseal(blob), "payload CRC"))
    blob = bytearray(original)
    blob[128 + 56] ^= 1
    mutations.append((_reseal(blob), "payload SHA"))
    for malformed, needle in mutations:
        _must_reject(malformed, needle)
    _must_reject(original[:-1], "file_size")
    assert first_payload.length > 0  # ensure inspection remained usable after negative cases


def test_crc_valid_payload_spoof_and_nonzero_padding_are_rejected():
    original = encode_bundle(_three_bundle())
    info = inspect_bundle(original)
    spans = sorted(info.spans, key=lambda span: span.offset)
    padding = next(span for span in spans if span.kind == "padding")
    blob = bytearray(original)
    blob[padding.offset] = 1
    _must_reject(_reseal(blob), "padding")

    payload = next(span for span in spans if span.kind == "payload" and span.name == "x86-avx2")
    blob = bytearray(original)
    blob[payload.offset : payload.offset + 4] = b"NOPE"
    entry_index = 2
    entry = 128 + entry_index * 448
    changed = bytes(blob[payload.offset : payload.end])
    struct.pack_into("<I", blob, entry + 48, zlib.crc32(changed) & 0xFFFFFFFF)
    blob[entry + 56 : entry + 88] = hashlib.sha256(changed).digest()
    _must_reject(_reseal(blob), "ELF")

    blob = bytearray(original)
    struct.pack_into("<I", blob, entry + 8, 0)
    _must_reject(_reseal(blob), "machine")

    _must_reject(_semantic_streampack_spoof(), "matching trace note")


def test_atomic_file_io_listing_hexdump_selection_and_extraction_cli():
    bundle = _three_bundle()
    data = encode_bundle(bundle)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.bcab"
        write_bundle(path, bundle)
        assert read_bundle(path) == bundle and not path.with_name(path.name + ".part").exists()
        listing = format_listing(data)
        assert listing.startswith("BCAB v1") and "x86-avx2 kind=elf_object" in listing
        dump = format_hexdump(data, width=8)
        tokens = []
        for line in dump.splitlines():
            if line and not line.startswith("#"):
                tokens.extend(line.split("  ", 2)[1].strip().split())
        assert bytes(int(token, 16) for token in tokens) == data

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            assert main(["list", str(path)]) == 0
        assert "embedded_sha256=" in stdout.getvalue()
        output = Path(directory) / "kernel.c"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            assert main(["extract", str(path), "--id", "portable-c", str(output)]) == 0
        assert output.read_bytes().startswith(b"int bcir_kernel")
        victim = Path(directory) / "victim"
        legacy_part = output.with_name(output.name + ".part")
        victim.write_bytes(b"unchanged")
        try:
            legacy_part.symlink_to(victim)
        except OSError:
            pass  # Windows may deny unprivileged symlink creation.
        else:
            with redirect_stdout(io.StringIO()):
                assert main(["extract", str(path), "--id", "portable-c", str(output)]) == 0
            assert victim.read_bytes() == b"unchanged"
            assert legacy_part.is_symlink()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            assert main(["select", str(path), "--id", "missing"]) == 2
        assert "no variant" in stderr.getvalue()


def test_delegated_tool_output_is_bounded_before_it_reaches_memory():
    try:
        _run_tool(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'x' * 4096)",
            ],
            maximum=32,
        )
        assert False, "oversized delegated-tool output must be rejected"
    except BundleError as exc:
        assert "exceeds 32 bytes" in str(exc)


def test_builder_wraps_real_backend_types_and_preserves_explicit_skips():
    builder = ArtifactBundleBuilder(provenance_digest=9, generation=3)
    builder.add_c_source("c-source", "int x;\n")
    builder.add_sycl_source("sycl-source", "void kernel() {}\n", entry_symbol="kernel")
    builder.add_codegen_result(CodegenResult(True, "x86_64", _elf(), "ok"))
    builder.add_codegen_result(CodegenResult(False, "spirv64", None, "backend unavailable"))
    builder.add_wasm("wasm", b"\x00asm\x01\x00\x00\x00")
    builder.add_archive("archive", b"!<arch>\n")
    builder.add_raw_binary("firmware", b"\x01\x02", architecture="test-isa", channel="device")
    builder.add_stack_program("stack", stackify(BinOp("add", Const(2), Const(3))))
    report = builder.finish(default_variant_id="c-source")
    assert report.skipped == (("target-spirv64", "backend unavailable"),)
    assert {variant.kind for variant in report.bundle.variants}.issuperset(
        {
            ArtifactKind.ELF_OBJECT,
            ArtifactKind.WASM,
            ArtifactKind.JVM_CLASS,
            ArtifactKind.CIL,
            ArtifactKind.SYCL_SOURCE,
            ArtifactKind.ARCHIVE,
            ArtifactKind.RAW_BINARY,
        }
    )
    assert decode_bundle(encode_bundle(report.bundle)) == report.bundle
    try:
        builder.add_linked_image("bad-shared", _elf(elf_type=2), shared=True)
        assert False, "ET_EXEC must not be labeled as a shared image"
    except BundleError:
        pass


def test_native_object_builder_recognizes_big_endian_macho64():
    payload = bytearray(28)
    payload[:4] = b"\xfe\xed\xfa\xcf"
    struct.pack_into(">I", payload, 4, 0x01000007)
    struct.pack_into(">I", payload, 12, 1)
    builder = ArtifactBundleBuilder(provenance_digest=0x1234)
    builder.add_native_object("macho-be64", bytes(payload))
    bundle = builder.finish(default_variant_id="macho-be64").bundle
    variant = bundle.variant("macho-be64")
    assert variant.kind == ArtifactKind.MACHO_OBJECT
    assert variant.format == ArtifactFormat.MACHO
    assert variant.endianness == Endianness.BIG
    assert variant.pointer_bits == 64
    assert variant.e_machine == 0x01000007
    assert decode_bundle(encode_bundle(bundle)) == bundle


def test_jvm_class_backend_rejects_bad_stack_and_produces_real_class_shape():
    payload = build_jvm_class("BcirKernel", ("ldc 1.0f", "ldc 2.0f", "fadd"))
    assert payload[:4] == b"\xca\xfe\xba\xbe" and len(payload) > 100
    for name, program in (("bad/name", ("ldc 1.0f",)), ("Good", ("fadd",)), ("Good", ("unknown",))):
        try:
            build_jvm_class(name, program)
            assert False, (name, program)
        except JVMClassError:
            pass
    try:
        build_jvm_class("Good", ("ldc 1.0f" for _ in range(16_385)))
        assert False, "the public assembler must bound iterable consumption"
    except JVMClassError as exc:
        assert "16384 instructions" in str(exc)


def test_mlir_lowering_carries_verified_directory_and_content_address():
    bundle = _three_bundle()
    envelope = _host_envelope()
    text = bundle_to_mlir(bundle, symbol_name="release_bundle")
    assert "bcir.artifact.bundle @release_bundle" in text
    assert 'bcir.artifact.variant @"00-root"' in text
    assert 'kind = "elf_object", format = "elf"' in text
    assert 'provenance_digest = "0000000000000000"' in text
    assert hashlib.sha256(encode_bundle(bundle)).hexdigest() in text
    selection = selection_to_mlir(bundle, envelope, bundle_symbol="release_bundle")
    assert "bcir.artifact.selection @artifact_selection" in selection
    assert 'variant = "x86-avx2"' in selection
    assert compatibility_sha256(envelope) in selection


def test_c_freestanding_reader_and_selector_match_python():
    compiler = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not compiler:
        return
    root = Path(__file__).resolve().parents[2]
    c = root / "runtime" / "c"
    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory) / "bundle.bcab"
        write_bundle(bundle, _three_bundle())
        executable = Path(directory) / ("test.exe" if compiler.lower().endswith(".exe") else "test")
        command = [
            compiler,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(c),
            str(c / "bcir_artifact_bundle.c"),
            str(c / "bcir_runtime.c"),
            str(c / "test_artifact_bundle.c"),
            "-o",
            str(executable),
        ]
        build = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert build.returncode == 0, build.stderr
        run = subprocess.run(
            [str(executable), str(bundle)], capture_output=True, text=True, timeout=30
        )
        assert run.returncode == 0, run.stderr
        assert run.stdout.startswith("OK entries=3")
        malformed = bytearray(bundle.read_bytes())
        struct.pack_into("<I", malformed, 128 + 2 * 448 + 8, 0)
        malformed_path = Path(directory) / "malformed.bcab"
        malformed_path.write_bytes(_reseal(malformed))
        rejected = subprocess.run(
            [str(executable), str(malformed_path), "reject"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert rejected.returncode == 0, rejected.stderr
        assert rejected.stdout.startswith("REJECT metadata")
        semantic_path = Path(directory) / "semantic-spoof.bcab"
        semantic_path.write_bytes(_semantic_streampack_spoof())
        rejected = subprocess.run(
            [str(executable), str(semantic_path), "reject"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert rejected.returncode == 0, rejected.stderr
        assert rejected.stdout.startswith("REJECT payload")


def test_resident_compiler_and_linker_products_roundtrip_when_available():
    compiler = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not compiler:
        return
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "kernel.c"
        image = Path(directory) / ("kernel.exe" if compiler.lower().endswith(".exe") else "kernel")
        source.write_text(
            "int bcir_kernel(void){return 7;} int main(void){return bcir_kernel()!=7;}\n"
        )
        build = subprocess.run(
            [compiler, str(source), "-o", str(image)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if build.returncode:
            return  # cross/minimal compilers can compile objects but lack a host linker
        payload = image.read_bytes()
        builder = ArtifactBundleBuilder(provenance_digest=11)
        builder.add_linked_image(
            "host-linked",
            payload,
            triple="x86_64-unknown-linux-gnu",
            entry_symbol="bcir_kernel",
        )
        bundle = builder.finish(default_variant_id="host-linked").bundle
        assert decode_bundle(encode_bundle(bundle)) == bundle
        extracted = decode_bundle(encode_bundle(bundle)).variant("host-linked").payload
        assert extracted == payload
