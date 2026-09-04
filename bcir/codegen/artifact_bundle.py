"""Backend adapters for constructing strict BCIR Artifact Bundles.

Adapters wrap real backend output without changing its standard representation.
Failures remain explicit in ``BundleBuildReport.skipped`` and never become dummy
variants, preserving BCIR's honest-degrade contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

from ..abi.artifact_bundle import (
    ArtifactBundle,
    ArtifactFormat,
    ArtifactKind,
    ArtifactVariant,
    BundleError,
    Endianness,
)
from ..abi.streampack_abi import encode as encode_stream_pack
from ..lower.jvm_class import build_jvm_class
from ..lower.stackify import StackOp, to_cil, to_jvm, to_wasm
from .codegen import CodegenResult, codegen, codegen_c, emit_c_source
from .targets import CODEGEN_TARGETS


_ELF_MACHINE_ARCH = {
    3: "i386",
    40: "arm",
    62: "x86_64",
    183: "aarch64",
    243: "riscv64",
    247: "bpf",
}
_COFF_MACHINE_ARCH = {0x014C: "i386", 0x8664: "x86_64", 0xAA64: "aarch64"}


@dataclass(frozen=True)
class BundleBuildReport:
    bundle: ArtifactBundle
    included: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]


def _native_identity(payload: bytes) -> tuple[ArtifactFormat, Endianness, int, int, str]:
    if len(payload) >= 20 and payload[:4] == b"\x7fELF":
        cls, data = payload[4], payload[5]
        if cls not in (1, 2) or data not in (1, 2):
            raise BundleError("native object has malformed ELF identity")
        endian = Endianness.LITTLE if data == 1 else Endianness.BIG
        machine = struct.unpack_from("<H" if data == 1 else ">H", payload, 18)[0]
        return (
            ArtifactFormat.ELF,
            endian,
            32 if cls == 1 else 64,
            machine,
            _ELF_MACHINE_ARCH.get(machine, ""),
        )
    macho = {
        b"\xce\xfa\xed\xfe": (Endianness.LITTLE, 32, "<"),
        b"\xcf\xfa\xed\xfe": (Endianness.LITTLE, 64, "<"),
        b"\xfe\xed\xfa\xce": (Endianness.BIG, 32, ">"),
        b"\xfe\xed\xfa\xcf": (Endianness.BIG, 64, ">"),
    }
    if len(payload) >= 28 and payload[:4] in macho:
        endian, bits, prefix = macho[payload[:4]]
        machine = struct.unpack_from(prefix + "I", payload, 4)[0]
        arch = {
            7: "i386",
            0x01000007: "x86_64",
            12: "arm",
            0x0100000C: "aarch64",
        }.get(machine, "")
        return ArtifactFormat.MACHO, endian, bits, machine, arch
    if len(payload) >= 20:
        machine = struct.unpack_from("<H", payload, 0)[0]
        if machine in _COFF_MACHINE_ARCH:
            bits = 64 if machine in (0x8664, 0xAA64) else 32
            return (
                ArtifactFormat.COFF,
                Endianness.LITTLE,
                bits,
                machine,
                _COFF_MACHINE_ARCH[machine],
            )
    raise BundleError("compiler output is not a recognized ELF, COFF, or Mach-O object")


class ArtifactBundleBuilder:
    """Mutable hosted builder whose ``finish`` result is immutable and canonical."""

    def __init__(self, *, provenance_digest: int = 0, generation: int = 0) -> None:
        self.provenance_digest = provenance_digest
        self.generation = generation
        self._variants: dict[str, ArtifactVariant] = {}
        self._skipped: list[tuple[str, str]] = []

    def add(self, variant: ArtifactVariant) -> None:
        if not isinstance(variant, ArtifactVariant):
            raise BundleError("builder.add expects an ArtifactVariant")
        if variant.variant_id in self._variants:
            raise BundleError(f"duplicate variant ID {variant.variant_id!r}")
        self._variants[variant.variant_id] = variant

    def skip(self, variant_id: str, reason: str) -> None:
        if not variant_id or not isinstance(reason, str) or not reason:
            raise BundleError("skipped variants require an ID and reason")
        self._skipped.append((variant_id, reason))

    def add_stream_pack(
        self, variant_id: str, pack, *, channel: str = "host", priority: int = 0
    ) -> None:
        self.add(
            ArtifactVariant(
                variant_id,
                ArtifactKind.STREAM_PACK,
                ArtifactFormat.STREAM_PACK,
                encode_stream_pack(pack),
                channel=channel,
                priority=priority,
                provenance_digest=self.provenance_digest,
                portable=True,
            )
        )

    def add_c_source(
        self, variant_id: str, source: str, *, entry_symbol: str = "bcir_kernel"
    ) -> None:
        if not isinstance(source, str):
            raise BundleError("C source must be text")
        self.add(
            ArtifactVariant(
                variant_id,
                ArtifactKind.C_SOURCE,
                ArtifactFormat.TEXT,
                source.encode("utf-8"),
                entry_symbol=entry_symbol,
                provenance_digest=self.provenance_digest,
                portable=True,
            )
        )

    def add_sycl_source(
        self,
        variant_id: str,
        source: str,
        *,
        entry_symbol: str,
        required_features: tuple[str, ...] = ("sycl",),
    ) -> None:
        self.add(
            ArtifactVariant(
                variant_id,
                ArtifactKind.SYCL_SOURCE,
                ArtifactFormat.TEXT,
                source.encode("utf-8"),
                channel="sycl_spirv",
                entry_symbol=entry_symbol,
                required_features=required_features,
                provenance_digest=self.provenance_digest,
                portable=True,
            )
        )

    def add_native_object(
        self,
        variant_id: str,
        payload: bytes,
        *,
        triple: str = "",
        channel: str = "host",
        priority: int = 0,
        target_manifest_sha256: str = "",
        cal_gen: int = 0,
        r12_attested: bool = True,
    ) -> None:
        fmt, endian, bits, machine, architecture = _native_identity(payload)
        kind = {
            ArtifactFormat.ELF: ArtifactKind.ELF_OBJECT,
            ArtifactFormat.COFF: ArtifactKind.COFF_OBJECT,
            ArtifactFormat.MACHO: ArtifactKind.MACHO_OBJECT,
        }[fmt]
        self.add(
            ArtifactVariant(
                variant_id,
                kind,
                fmt,
                payload,
                triple=triple,
                architecture=architecture,
                channel=channel,
                entry_symbol="bcir_kernel",
                endianness=endian,
                pointer_bits=bits,
                e_machine=machine,
                priority=priority,
                provenance_digest=self.provenance_digest,
                target_manifest_sha256=target_manifest_sha256,
                cal_gen=cal_gen,
                r12_attested=r12_attested,
                executable=True,
            )
        )

    def add_linked_image(
        self,
        variant_id: str,
        payload: bytes,
        *,
        triple: str = "",
        channel: str = "host",
        shared: bool = False,
        entry_symbol: str = "",
        priority: int = 0,
        r12_attested: bool = True,
    ) -> None:
        """Add a real ELF/PE/Mach-O linker product with header-derived metadata."""
        if payload.startswith(b"\x7fELF"):
            fmt, endian, bits, machine, architecture = _native_identity(payload)
            prefix = "<" if endian == Endianness.LITTLE else ">"
            elf_type = struct.unpack_from(prefix + "H", payload, 16)[0]
            if elf_type == 2:
                if shared:
                    raise BundleError("linked ELF ET_EXEC disagrees with shared=True")
                kind = ArtifactKind.ELF_EXECUTABLE
            elif elf_type == 3:
                kind = ArtifactKind.ELF_SHARED if shared else ArtifactKind.ELF_EXECUTABLE
            else:
                raise BundleError("linked ELF image must be ET_EXEC or ET_DYN")
        elif payload.startswith(b"MZ"):
            if len(payload) < 0x40:
                raise BundleError("linked PE image is truncated")
            pe_offset = struct.unpack_from("<I", payload, 0x3C)[0]
            if pe_offset > len(payload) - 24 or payload[pe_offset : pe_offset + 4] != b"PE\0\0":
                raise BundleError("linked PE image has no bounded PE header")
            machine = struct.unpack_from("<H", payload, pe_offset + 4)[0]
            optional_size = struct.unpack_from("<H", payload, pe_offset + 20)[0]
            characteristics = struct.unpack_from("<H", payload, pe_offset + 22)[0]
            if optional_size < 2 or pe_offset + 24 + optional_size > len(payload):
                raise BundleError("linked PE optional header is truncated")
            magic = struct.unpack_from("<H", payload, pe_offset + 24)[0]
            if magic not in (0x10B, 0x20B):
                raise BundleError("linked PE optional header has an unknown class")
            bits, endian, fmt = (32 if magic == 0x10B else 64), Endianness.LITTLE, ArtifactFormat.PE
            architecture = _COFF_MACHINE_ARCH.get(machine, "")
            is_dll = bool(characteristics & 0x2000)
            if is_dll != shared:
                raise BundleError("linked PE DLL flag disagrees with shared=")
            kind = ArtifactKind.PE_SHARED if shared else ArtifactKind.PE_EXECUTABLE
        else:
            fmt, endian, bits, machine, architecture = _native_identity(payload)
            if fmt != ArtifactFormat.MACHO:
                raise BundleError("linked image is not ELF, PE, or Mach-O")
            prefix = "<" if endian == Endianness.LITTLE else ">"
            file_type = struct.unpack_from(prefix + "I", payload, 12)[0]
            expected = 6 if shared else 2
            if file_type != expected:
                raise BundleError("linked Mach-O file type disagrees with shared=")
            kind = ArtifactKind.MACHO_SHARED if shared else ArtifactKind.MACHO_EXECUTABLE
        self.add(
            ArtifactVariant(
                variant_id,
                kind,
                fmt,
                payload,
                triple=triple,
                architecture=architecture,
                channel=channel,
                entry_symbol=entry_symbol,
                endianness=endian,
                pointer_bits=bits,
                e_machine=machine,
                priority=priority,
                provenance_digest=self.provenance_digest,
                r12_attested=r12_attested,
                executable=True,
            )
        )

    def add_archive(self, variant_id: str, payload: bytes, *, priority: int = 0) -> None:
        self.add(
            ArtifactVariant(
                variant_id,
                ArtifactKind.ARCHIVE,
                ArtifactFormat.ARCHIVE,
                payload,
                priority=priority,
                provenance_digest=self.provenance_digest,
            )
        )

    def add_raw_binary(
        self,
        variant_id: str,
        payload: bytes,
        *,
        architecture: str,
        channel: str,
        entry_symbol: str = "",
        priority: int = 0,
        r12_attested: bool = True,
    ) -> None:
        self.add(
            ArtifactVariant(
                variant_id,
                ArtifactKind.RAW_BINARY,
                ArtifactFormat.RAW,
                payload,
                architecture=architecture,
                channel=channel,
                entry_symbol=entry_symbol,
                priority=priority,
                provenance_digest=self.provenance_digest,
                r12_attested=r12_attested,
                executable=True,
            )
        )

    def add_codegen_result(self, result: CodegenResult, *, priority: int = 0) -> None:
        variant_id = f"target-{result.target}"
        if not result.ok or result.artifact is None:
            self.skip(variant_id, result.message)
            return
        target = CODEGEN_TARGETS.get(result.target)
        if isinstance(result.artifact, bytes):
            self.add_native_object(
                variant_id,
                result.artifact,
                triple=target.triple if target else "",
                priority=priority,
            )
            return
        payload = result.artifact.encode("utf-8")
        if result.target == "nvptx64":
            kind, channel = ArtifactKind.PTX, "nvidia_ptx"
        else:
            kind, channel = ArtifactKind.ASSEMBLY, result.target
        self.add(
            ArtifactVariant(
                variant_id,
                kind,
                ArtifactFormat.TEXT,
                payload,
                triple=target.triple if target else "",
                architecture=result.target,
                channel=channel,
                entry_symbol="bcir_kernel",
                priority=priority,
                provenance_digest=self.provenance_digest,
                r12_attested=True,
                executable=True,
            )
        )

    def add_wasm(
        self, variant_id: str, payload: bytes, *, priority: int = 0, r12_attested: bool = True
    ) -> None:
        self.add(
            ArtifactVariant(
                variant_id,
                ArtifactKind.WASM,
                ArtifactFormat.WASM,
                payload,
                triple="wasm32-unknown-unknown",
                architecture="wasm32",
                channel="wasm",
                entry_symbol="bcir_kernel",
                endianness=Endianness.LITTLE,
                pointer_bits=32,
                priority=priority,
                provenance_digest=self.provenance_digest,
                r12_attested=r12_attested,
                executable=True,
                portable=True,
            )
        )

    def add_stack_program(
        self, variant_prefix: str, ops: list[StackOp], *, class_name: str = "BcirStackKernel"
    ) -> None:
        jvm_text = tuple(to_jvm(ops))
        self.add(
            ArtifactVariant(
                f"{variant_prefix}-jvm",
                ArtifactKind.JVM_CLASS,
                ArtifactFormat.JVM_CLASS,
                build_jvm_class(class_name, jvm_text),
                triple="jvm-unknown-java",
                architecture="jvm",
                channel="jvm",
                entry_symbol="run",
                provenance_digest=self.provenance_digest,
                r12_attested=True,
                executable=True,
                portable=True,
            )
        )
        self.add(
            ArtifactVariant(
                f"{variant_prefix}-cil",
                ArtifactKind.CIL,
                ArtifactFormat.TEXT,
                ("\n".join(to_cil(ops)) + "\n").encode(),
                triple="cil-unknown-dotnet",
                architecture="cil",
                channel="cil",
                entry_symbol="run",
                provenance_digest=self.provenance_digest,
                r12_attested=True,
                executable=True,
                portable=True,
            )
        )
        self.add(
            ArtifactVariant(
                f"{variant_prefix}-wasm-text",
                ArtifactKind.ASSEMBLY,
                ArtifactFormat.TEXT,
                ("\n".join(to_wasm(ops)) + "\n").encode(),
                triple="wasm32-unknown-unknown",
                architecture="wasm32",
                channel="wasm",
                entry_symbol="run",
                provenance_digest=self.provenance_digest,
                r12_attested=True,
                executable=True,
                portable=True,
            )
        )

    def finish(
        self, *, root_variant_id: str = "", default_variant_id: str = ""
    ) -> BundleBuildReport:
        if not self._variants:
            raise BundleError("cannot finish an empty artifact bundle")
        variants = tuple(self._variants[key] for key in sorted(self._variants))
        bundle = ArtifactBundle(
            variants,
            root_variant_id,
            default_variant_id,
            self.provenance_digest,
            self.generation,
        )
        return BundleBuildReport(
            bundle,
            tuple(v.variant_id for v in variants),
            tuple(self._skipped),
        )


def build_codegen_bundle(
    module,
    realization,
    stream_pack,
    *,
    include_targets: bool = True,
    include_c_object: bool = True,
    include_c_source: bool = True,
    provenance_digest: int = 0,
    generation: int = 0,
) -> BundleBuildReport:
    """Build all available host/cross artifacts without inventing missing outputs."""
    builder = ArtifactBundleBuilder(
        provenance_digest=provenance_digest,
        generation=generation,
    )
    builder.add_stream_pack("00-streampack", stream_pack, priority=100)
    if include_c_source:
        builder.add_c_source("source-c", emit_c_source(module, realization))
    if include_c_object:
        builder.add_codegen_result(codegen_c(module, realization), priority=50)
    if include_targets:
        for target_name in sorted(CODEGEN_TARGETS):
            builder.add_codegen_result(codegen(module, realization, target_name), priority=40)
    return builder.finish(
        root_variant_id="00-streampack",
        default_variant_id="00-streampack",
    )


__all__ = ["ArtifactBundleBuilder", "BundleBuildReport", "build_codegen_bundle"]
