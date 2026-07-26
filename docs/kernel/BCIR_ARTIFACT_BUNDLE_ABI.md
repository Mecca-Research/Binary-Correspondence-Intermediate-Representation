# BCIR Artifact Bundle ABI — BCAB v1 (normative)

BCAB is BCIR's deterministic **multi-backend artifact envelope**. It binds one verified
StreamPack and any number of standard target images—ELF, COFF, PE, Mach-O, archives,
WASM, LLVM IR/bitcode, PTX/CUBIN, SPIR-V, JVM class files, CIL text, C/C++/SYCL source,
assembly, or explicitly typed raw device binaries—to one compatibility and provenance
directory.

BCAB is not a new object format, linker, OS loader, code signer, or instruction set. Every
payload remains byte-for-byte in its native format and is assembled, linked, loaded, and
disassembled by its owning toolchain. BCAB answers the missing fat-binary question: which
standard image is compatible with this target envelope, and which exact bytes were selected?

The reference codec is
[`bcir/abi/artifact_bundle.py`](../../bcir/abi/artifact_bundle.py), the allocation-free C
reader is [`runtime/c/bcir_artifact_bundle.h`](../../runtime/c/bcir_artifact_bundle.h), and
the C++ borrowed-view wrapper is
[`runtime/cpp/bcir_artifact_bundle.hpp`](../../runtime/cpp/bcir_artifact_bundle.hpp).

## 1. Wire conventions and limits

- Little-endian integer fields.
- Fixed 128-byte header and fixed 448-byte directory entries.
- One through 1,024 variants; total file size at most 1 GiB.
- Directory entries and payloads are sorted by canonical `variant_id`.
- Payload starts and the final file size are aligned to eight bytes. Every alignment byte is
  zero; gaps, overlaps, undeclared tails, and alternate layouts are invalid.
- Fixed strings are printable ASCII, NUL-terminated inside their field, zero after the first
  NUL, and have no leading or trailing space. Feature names use `[A-Za-z0-9_.+:-]+`; feature
  CSV fields are sorted, unique, and disjoint.
- Reserved fields and unknown flag bits are zero. Readers reject unknown versions, kinds,
  formats, and enum values.

## 2. Header (128 bytes)

| Offset | Bytes | Field | Contract |
|---:|---:|---|---|
| 0 | 4 | `magic` | ASCII `BCAB` |
| 4 | 2 | `version` | `1` |
| 6 | 2 | `header_size` | `128` |
| 8 | 4 | `flags` | zero in v1 |
| 12 | 4 | `entry_count` | `1..1024` |
| 16 | 4 | `entry_size` | `448` |
| 20 | 4 | `root_index` | StreamPack directory index or `0xffffffff` |
| 24 | 4 | `default_index` | default directory index or `0xffffffff` |
| 28 | 8 | `directory_offset` | exactly `128` |
| 36 | 8 | `directory_size` | exactly `entry_count * 448` |
| 44 | 8 | `payload_offset` | aligned end of the directory |
| 52 | 8 | `file_size` | exact input length |
| 60 | 8 | `provenance_digest` | bundle-level R13/source provenance key |
| 68 | 8 | `generation` | immutable activation generation |
| 76 | 4 | `body_crc32` | zlib CRC-32 of bytes `[128,file_size)` |
| 80 | 4 | `header_crc32` | CRC-32 of the header with this field zero |
| 84 | 32 | `embedded_sha256` | SHA-256 described below |
| 116 | 12 | reserved | zero |

The embedded SHA-256 covers the complete artifact with both `header_crc32` and
`embedded_sha256` zeroed. Encoding order is therefore: build directory/payload body, compute
body CRC, compute embedded SHA with the two excluded fields zero, insert the SHA, then compute
the header CRC with only its own field zero. `inspect_bundle()` also reports the ordinary
SHA-256 of the final file; that external content address is intentionally distinct from the
self-embedded digest.

`root_index`, when present, must name a StreamPack. `default_index` may name any payload, but
an executable default must carry the R12-attested flag. A root identifies the portable
execution plan; it does not override target compatibility selection.

## 3. Directory entry (448 bytes)

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 2 | `kind` |
| 2 | 2 | `format` |
| 4 | 1 | endianness: `0 neutral`, `1 little`, `2 big` |
| 5 | 1 | pointer bits: `0`, `32`, or `64` |
| 6 | 2 | flags |
| 8 | 4 | native machine/CPU identifier (`e_machine`, COFF machine, or Mach CPU type) |
| 12 | 4 | signed selection priority |
| 16 | 8 | aligned payload offset |
| 24 | 8 | nonzero payload size |
| 32 | 8 | variant provenance digest |
| 40 | 8 | calibration generation (`0` means unpinned) |
| 48 | 4 | payload CRC-32 |
| 52 | 4 | reserved zero |
| 56 | 32 | payload SHA-256 |
| 88 | 32 | target-manifest SHA-256; all zero means unpinned |
| 120 | 48 | variant ID |
| 168 | 48 | target triple |
| 216 | 24 | architecture |
| 240 | 24 | OS ABI |
| 264 | 24 | RuntimeChannel/backend channel |
| 288 | 32 | entry symbol |
| 320 | 64 | required-feature CSV |
| 384 | 64 | prohibited-feature CSV |

Flags are `0x1 R12-attested`, `0x2 executable`, `0x4 portable`, and `0x8 debug`.
Every ELF, COFF, Mach-O, or PE entry has a nonzero machine identifier that exactly matches
the payload header. Kinds explicitly named `*_executable` must set the executable flag; they
cannot bypass R12 admission by presenting themselves as data.
Integrity is checked before payload identity or compatibility. CRC provides fast corruption
detection; SHA-256 pins exact bytes. Neither is an authenticity signature.

## 4. Kinds and payload formats

| Kind IDs | Meaning | Required format |
|---|---|---|
| 1 | StreamPack | StreamPack |
| 2–3, 19 | ELF relocatable, shared object, executable/PIE | ELF |
| 4 | COFF relocatable | COFF |
| 5, 22–23 | Mach-O relocatable, executable, shared image | Mach-O |
| 6 | deterministic/system archive | archive |
| 7 | WebAssembly module | WASM |
| 8–9 | LLVM bitcode, LLVM IR text | LLVM bitcode, text |
| 10–11 | PTX text, CUBIN | text, ELF |
| 12 | SPIR-V module | SPIR-V |
| 13–14 | JVM class, CIL text | JVM class, text |
| 15–18 | C, C++, SYCL, assembly source | text |
| 20–21 | PE executable, PE DLL | PE |
| 24 | explicitly typed device/firmware image | raw |

Format IDs are `0 none`, `1 StreamPack`, `2 ELF`, `3 COFF`, `4 Mach-O`, `5 archive`,
`6 WASM`, `7 LLVM bitcode`, `8 text`, `9 SPIR-V`, `10 JVM class`, `11 PE`, and
`12 raw`.

Readers perform bounded identity checks before admission: complete StreamPack semantic
verification; ELF class/byte order/machine and applicable `e_type`; COFF machine; Mach-O
class/byte order/CPU/file type; archive, WASM, bitcode, SPIR-V, JVM, and PE signatures and
header bounds; valid NUL-free UTF-8 text; and PTX `.version`/`.target`. Raw images have no
universal internal header, so their exact digest plus explicit architecture/channel is the
contract. Full relocation, verifier, W^X, signature, errata, and OS-loader policy remains the
responsibility of the standard loader and MC14.

## 5. Compatibility and deterministic selection

A `CompatibilityEnvelope` declares target triple, architecture, OS ABI, channel, available
features, accepted kinds/formats, byte order, pointer width, native machine ID, target-manifest
digest, calibration generation, R12 requirement, and debug policy. A zero kind or format mask
accepts all values. An absent envelope identity field does not satisfy a variant that constrains
that field; nonempty variant constraints must match exactly. C callers provide NUL-terminated
selector strings within the wire-equivalent bounds (`47` bytes for variant IDs/triples, `23`
for architecture/OS ABI/channel, and `63` for feature CSV); unknown mask bits and non-Boolean
policy bytes are rejected before matching.

A variant is compatible only when:

1. kind/format masks admit it;
2. every declared target identity field matches;
3. endianness, pointer width, and machine match when constrained;
4. required features are a subset and prohibited features are disjoint;
5. a pinned target manifest and nonzero calibration generation match exactly;
6. executable variants satisfy the requested R12 policy; and
7. debug variants are explicitly allowed.

Compatible candidates are ranked by descending priority, descending metadata specificity,
descending required-feature count, then ascending variant ID. This ordering is identical in
Python and C. An explicit ID still undergoes envelope checks. Selecting without an envelope is
limited to the declared default; extraction by ID is not an execution admission decision.
`compatibility_sha256()` content-addresses the canonical envelope, and
`bcir.artifact.selection` records the selected ID, envelope digest, generation, and
exact/quantized/approximate classification in MLIR.

## 6. Interfaces and backend boundary

Python:

```python
encode_bundle(bundle) -> bytes
decode_bundle(data) -> ArtifactBundle
inspect_bundle(data) -> ArtifactBundleInspection
write_bundle(path, bundle)          # fsync + atomic replace
read_bundle(path, max_bytes=...)
select_variant(bundle, envelope) -> ArtifactVariant
```

`ArtifactBundleBuilder` wraps real codegen results and records unavailable targets as explicit
skips. It supports StreamPack, C/SYCL source, resident GCC/Clang/LLVM objects and linked images,
WASM, PTX/assembly, archives, raw device images, and the shared stackified JVM/CIL/WASM rail.
The bounded JVM assembler emits a real Java 8 class for the supported float subset.

The C API (`bcir_ab_open/get/select`) allocates nothing, copies no payload, zeroes outputs on
failure, and returns borrowed spans. The C++ wrapper gives the same borrowed lifetime an RAII
error surface; it does not take ownership of the bytes. The MLIR operations
`bcir.artifact.bundle`, `bcir.artifact.variant`, and `bcir.artifact.selection` carry validated
metadata, not payload blobs.

`bcir-bundle list|hexdump|extract|select|dis|mnemonics` validates the entire BCAB before exposing
one byte. StreamPack disassembly uses `bcir-pack`; native/WASM/bitcode/SPIR-V/JVM payloads are
delegated to `llvm-objdump`/`objdump`, `llvm-dis`, `spirv-dis`, or `javap`. Missing tools fail
honestly. Standard assembler/linker output is extracted byte-identically; BCAB never rewrites
relocations, sections, symbols, or instruction encodings.

## 7. Additive ASN.1 transfer syntax

BCAB v1 remains the native artifact and its bytes above do not change. The
[`BCIR-ArtifactBundle`](../../bcir/asn1/BCIR-ArtifactBundle.asn1) X.680 module assigns
OID `{ 1 3 6 1 4 1 62596 2 }` and supplies a second transfer syntax for the same
abstract bundle:

- `encode_bundle_der` emits DER; `decode_bundle_der` requires DER unless the caller
  explicitly selects `Strictness.BER`;
- `encode_bundle_oer` emits CANONICAL-OER; `decode_bundle_oer` accepts BASIC-OER or
  requires canonical form on request;
- `native_to_der`/`der_to_native` and `native_to_oer`/`oer_to_native` validate both
  sides and reconstruct canonical native BCAB bytes; and
- `bcir-bundle to-der|from-der|to-oer|from-oer` exposes those bounded transcodes with
  atomic destination replacement.

The projection names every semantic header/directory field and carries payloads as
OCTET STRING values. It deliberately omits native offsets, alignment padding, CRCs,
and SHA-256 values because they are derived from the reconstructed canonical BCAB.
The compatibility law is byte identity:

```text
der_to_native(native_to_der(native)) == native
oer_to_native(native_to_oer(native), canonical=True) == native
```

It is not an ASN.1 blob hidden in a BCAB raw-image slot and it allocates no competing
BCAB kind/format IDs. The native enum values are projected as ASN.1 ENUMERATED values,
so all selectors and C/C++ readers continue to consume the original v1 directory.

`bcir.asn1.projection` records `native = "artifact_bundle"` and `additive` under R24.
The generic freestanding C X.690 parser validates the DER tree; schema-aware
DER/OER→BCAB reconstruction currently lives in the dependency-free Python oracle,
after which the allocation-free C BCAB reader validates the reconstructed native
artifact. No schema-specific C reconstruction path is claimed.

## 8. Binary compatibility strategy

- BCIR semantics and R1–R24 are source/IR contracts.
- StreamPack is the target-neutral executable plan and remains independently versioned.
- BCAB is the target-image selection envelope and remains independently versioned.
- ELF/COFF/PE/Mach-O/WASM/JVM/SPIR-V and platform calling conventions remain owned by their
  standards and resident toolchains.
- Linux consumes extracted ELF/StreamPack payloads through existing file-descriptor, loader, and
  future RuntimeChannel paths. BCAB adds no Linux kernel-internal ABI and does not replace POSIX.
- Windows uses the same BCAB parser and selector with COFF/PE target metadata; no POSIX library or
  `mmap` is required by the C reader.
- SYCL variants carry source or produced SPIR-V/native images and the `sycl_spirv` channel; SYCL
  remains a backend/data path, never a legality authority.

BCAB v1 is append-only only through a future explicit version. Reserved bytes, new enum values,
trailing data, and alternate geometry are not extension points. A future signature directory,
loader policy, or pack-level symbol/relocation section requires a versioned contract; it must not
silently overload the v1 checksums or metadata.

## 9. Conformance

Required evidence includes deterministic encode/decode, exact span partitioning, corruption,
truncation, overlap, padding, CRC/SHA, payload-spoof, R12/debug/feature/manifest/generation
selection, atomic file replacement, CLI extraction, real resident compiler/linker output,
JVM class shape/execution where available, ODS/IRDL round trips and negatives, Python/C/C++
selector parity, strict-warning freestanding builds, sanitizer coverage, and bounded fuzzing.
The ASN.1 seam additionally requires X.680 source compilation parity, DER/COER deterministic
round trips, explicit BER/BASIC-OER admission, native byte identity, malformed projected
metadata refusal, the R24 additive-projection fixture, and generic C X.690 validation.
Downloaded toolchains or unavailable target hardware are clean capability skips, never synthetic
artifacts or claimed execution.
