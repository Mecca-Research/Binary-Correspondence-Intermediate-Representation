"""The C BCAB tool (runtime/c/bcir_bundle_cli.c, PORT-1): the oracle's octets, command for command.

`bcir-bundle` in C prints what `bcir.abi.artifact_tool` prints for `list`, `hexdump`, `select`,
`extract` and the StreamPack and text views of `dis` and `mnemonics`, over bundles whose variants
reach every field of the listing (each flag, a negative priority, features, a manifest digest, the
largest cal_gen, alignment padding, no root and no default) and whose text payloads hold every
whitespace and line-break rule Python's `str` methods apply. Selection is compared over envelopes
that select, that select nothing and that the oracle refuses; over resealed mutations both rails
accept the same bundles. A payload whose view the oracle hands to a resident disassembler, and the
ASN.1 projections, are refused by the C twin, never answered differently.

`check_tool` is the one comparison: this module runs it against a binary it builds from the
manifest closure (held there by `assert_fixture_links`), and tools/c/sections/bundle_listing.sh runs
it against the binary BCIR Make and CMake build.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import os
import random
import subprocess
import sys
import tempfile

from bcir.abi import (
    ArtifactBundle,
    ArtifactFormat,
    ArtifactKind,
    ArtifactVariant,
    BundleError,
    Endianness,
    decode_bundle,
    encode,
    encode_bundle,
)
from bcir.abi.artifact_tool import format_hexdump, format_listing, main as oracle_main
from bcir.examples import multi_histogram
from bcir.gem import hydrate
from bcir.gem.streampack import StreamPack
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.tests.test_artifact_bundle import _elf, _reseal, _three_bundle

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RUNTIME_C = os.path.join(_ROOT, "runtime", "c")
_SOURCES = (
    "bcir_bundle_cli.c",
    "bcir_ab_listing.c",
    "bcir_sp_listing.c",
    "bcir_artifact_bundle.c",
    "bcir_sha256.c",
    "bcir_runtime.c",
)
_WIDTHS = (1, 5, 16, 64)
_U64 = (1 << 64) - 1

# Every rule of str.splitlines / strip / split: CRLF, NEL, LS, IS1 and IS4, NBSP, the ideographic
# space, FF, a comment by each prefix, a lone slash, CR CR, and an unterminated last line.
_TRICKY_TEXT = (
    "  mov a, b\r\n# c\n\x85; x\nnop\u2028.l q\u3000tail\x0cadd\x1c  b\x1fc\n// z\n/x y\n"
    "\xa0\u2000sub\xa0r1\r\rlast \u00e9t\u00e9"
)

# Every code point str.isspace() admits, each at a line's edges and between its words, and four
# that look like whitespace and are not (Mongolian vowel separator, zero-width space, word joiner,
# zero-width no-break space): the C table is held to the interpreter's own set, member by member.
_SPACES = tuple(c for c in range(0x110000) if chr(c).isspace())
_NOT_SPACES = (0x180E, 0x200B, 0x2060, 0xFEFF)
_EVERY_SPACE_TEXT = "\n".join(f"{chr(c)}w{i}{chr(c)}t" for i, c in enumerate(_SPACES + _NOT_SPACES))


def _many() -> ArtifactBundle:
    module = multi_histogram()
    pack = encode(hydrate(module, optimize(module, TargetProfile.x86_avx512(), Theta.cool())))
    return ArtifactBundle(
        (
            ArtifactVariant(
                "a-asm",
                ArtifactKind.ASSEMBLY,
                ArtifactFormat.TEXT,
                _TRICKY_TEXT.encode("utf-8"),
                required_features=("neon", "sve"),
                prohibited_features=("avx512f",),
                priority=-5,
                provenance_digest=7,
                target_manifest_sha256="ab" * 32,
                cal_gen=_U64,
                debug=True,
            ),
            ArtifactVariant(
                "a-ws",
                ArtifactKind.ASSEMBLY,
                ArtifactFormat.TEXT,
                _EVERY_SPACE_TEXT.encode("utf-8"),
            ),
            ArtifactVariant(
                "b-ir",
                ArtifactKind.LLVM_IR,
                ArtifactFormat.TEXT,
                b"define void @f() {\n  ret void\n}",
                priority=3,
                portable=True,
            ),
            ArtifactVariant(
                "c-pack", ArtifactKind.STREAM_PACK, ArtifactFormat.STREAM_PACK, pack, channel="pim0"
            ),
            ArtifactVariant(
                "d-pack0",
                ArtifactKind.STREAM_PACK,
                ArtifactFormat.STREAM_PACK,
                encode(StreamPack(source_plan="empty")),
            ),
            ArtifactVariant(
                "e-elf",
                ArtifactKind.ELF_OBJECT,
                ArtifactFormat.ELF,
                _elf(183),
                triple="aarch64-unknown-linux-gnu",
                architecture="aarch64",
                os_abi="linux-gnu",
                entry_symbol="bcir_kernel",
                required_features=("neon",),
                endianness=Endianness.LITTLE,
                pointer_bits=64,
                e_machine=183,
                priority=9,
                r12_attested=True,
                executable=True,
            ),
            ArtifactVariant("f-raw", ArtifactKind.RAW_BINARY, ArtifactFormat.RAW, bytes(range(13))),
        ),
        "c-pack",
        "b-ir",
        _U64,
        5,
    )


def _single() -> ArtifactBundle:
    return ArtifactBundle(
        (ArtifactVariant("only", ArtifactKind.C_SOURCE, ArtifactFormat.TEXT, b"x"),), "", "", 0, 0
    )


def corpus() -> list[tuple[str, bytes]]:
    """Every bundle the parity is held over, each with its name."""
    return [
        ("three", encode_bundle(_three_bundle())),
        ("many", encode_bundle(_many())),
        ("single", encode_bundle(_single())),
    ]


# Envelopes that select, that select nothing, and that the oracle refuses; with --id the envelope
# options are not consulted at all.
_ALLOW = ["--allow-unattested", "--allow-debug"]
# Envelopes that select, that select nothing, and that the oracle refuses. A refused envelope carries
# both allow flags, so only its own malformation can refuse it: with a malformed field accepted, the
# unconstrained variants would be selected. With --id the envelope options are not consulted at all.
_SELECTORS = (
    [],
    _ALLOW,
    ["--kind", "assembly", *_ALLOW],
    ["--format", "text", "--format", "TEXT", "--allow-unattested"],
    ["--kind", "Stream-Pack", "--allow-unattested"],
    [
        "--triple",
        "aarch64-unknown-linux-gnu",
        "--architecture",
        "aarch64",
        "--os-abi",
        "linux-gnu",
        "--endianness",
        "little",
        "--pointer-bits",
        "64",
        "--machine",
        "183",
        "--feature",
        "neon",
    ],
    [
        "--triple=x86_64-unknown-linux-gnu",
        "--architecture=x86_64",
        "--os-abi=linux-gnu",
        "--channel=host",
        "--feature=avx2",
        "--endianness=little",
        "--pointer-bits=64",
        "--machine=62",
    ],
    ["--feature", "sve", "--feature", "neon", "--feature", "neon", *_ALLOW],
    ["--manifest-sha256", "ab" * 32, *_ALLOW],
    ["--cal-gen", str(_U64), *_ALLOW],
    ["--cal-gen", "0", "--allow-unattested"],
    ["--channel", "pim0", "--allow-unattested"],
    ["--endianness", "big", "--pointer-bits", "32", *_ALLOW],
    ["--manifest-sha256", "00" * 32, *_ALLOW],
    ["--manifest-sha256", "AB" * 32, *_ALLOW],
    ["--manifest-sha256", "ab", *_ALLOW],
    ["--pointer-bits", "16", *_ALLOW],
    ["--machine", str(1 << 32), *_ALLOW],
    ["--cal-gen", str(1 << 64), *_ALLOW],
    ["--kind", "nonsense", *_ALLOW],
    ["--format", "elf_object", *_ALLOW],
    ["--triple", " lead", *_ALLOW],
    ["--channel", "tail ", *_ALLOW],
    ["--os-abi", "x" * 24, *_ALLOW],
    ["--architecture", "x" * 23, *_ALLOW],
    ["--triple", "x" * 48, *_ALLOW],
    ["--feature", "bad,name", *_ALLOW],
    ["--feature", "", *_ALLOW],
    ["--feature", "f" * 64, *_ALLOW],
    ["--id", "b-ir"],
    ["--id", "c-pack", "--kind", "nonsense", "--manifest-sha256", "zz"],
    ["--id", "nowhere"],
    ["--id", " only"],
    ["--id", "x" * 48],
)


def _oracle(args: list[str]) -> tuple[int, bytes]:
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            rc = oracle_main(args)
    except SystemExit as exc:  # argparse's usage error
        rc = int(exc.code or 0)
    return rc, out.getvalue().encode("utf-8")


def _gives_a_reason(stderr: bytes) -> bool:
    """A refusal's last line is `bcir-bundle: <reason>` (a usage error prints the usage first)."""
    lines = stderr.splitlines()
    return bool(lines) and lines[-1].startswith(b"bcir-bundle: ")


def _oracle_accepts(blob: bytes) -> bool:
    try:
        decode_bundle(blob)
    except BundleError:
        return False
    return True


def mutants(blob: bytes, rng: random.Random, count: int) -> list[bytes]:
    """Bundles one change away from `blob`, resealed so the change reaches the directory and the
    payload laws; and a few raw ones, refused by a checksum."""
    out = [blob[: rng.randrange(len(blob))], blob + bytes(8)]
    for _ in range(count):
        b = bytearray(blob)
        at = rng.randrange(len(b))
        b[at] = rng.choice((b[at] ^ (1 << rng.randrange(8)), 0, 0xFF, rng.randrange(256)))
        out.append(_reseal(b) if rng.random() < 0.85 else bytes(b))
    return out


def check_tool(tool: str, work: str, *, mutations: int = 8, seed: int = 20261110) -> list[str]:
    """Hold the C `bcir-bundle` at `tool` to the oracle; return the failures (empty when it agrees)."""
    failures: list[str] = []
    path = os.path.join(work, "bundle.bcab")

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run([tool, *args], capture_output=True, timeout=60)

    def same(label: str, args: list[str]) -> None:
        got = run(*args)
        rc, want = _oracle(args)
        if got.returncode != rc or got.stdout != want:
            failures.append(
                f"{label}: {args[0]} {args[2:]}: C exit {got.returncode} vs {rc}, "
                f"{'same' if got.stdout == want else 'different'} output {got.stderr[:120]!r}"
            )
        elif rc and (got.stdout or not _gives_a_reason(got.stderr)):
            failures.append(f"{label}: {args[0]} {args[2:]}: a refusal printed output or no reason")

    rng = random.Random(seed)
    for name, blob in corpus():
        with open(path, "wb") as f:
            f.write(blob)
        got = run("list", path)
        if got.returncode != 0 or got.stdout != format_listing(blob).encode():
            failures.append(f"{name}: list differs (exit {got.returncode})")
        for width in _WIDTHS:
            got = run("hexdump", path, "--width", str(width))
            if got.returncode != 0 or got.stdout != format_hexdump(blob, width=width).encode():
                failures.append(f"{name}: hexdump --width {width} differs")
        for selector in _SELECTORS:
            same(name, ["select", path, *selector])
        bundle = decode_bundle(blob)
        for variant in bundle.variants:
            vid = variant.variant_id
            c_out, py_out = os.path.join(work, "c.out"), os.path.join(work, "py.out")
            got = run("extract", path, c_out, "--id", vid)
            rc, want = _oracle(["extract", path, py_out, "--id", vid])
            with open(c_out, "rb") as c_file, open(py_out, "rb") as py_file:
                if got.returncode != rc or got.stdout != want or c_file.read() != py_file.read():
                    failures.append(f"{name}: extract {vid} differs")
            for command in ("dis", "mnemonics"):
                viewable = (
                    variant.kind == ArtifactKind.STREAM_PACK
                    or variant.format == ArtifactFormat.TEXT
                )
                if viewable:
                    same(name, [command, path, "--id", vid])
                else:
                    got = run(command, path, "--id", vid)
                    if got.returncode != 2 or got.stdout or b"disassembler" not in got.stderr:
                        failures.append(f"{name}: {command} {vid} is not refused as delegated")
        for j, mutant in enumerate(mutants(blob, rng, mutations)):
            with open(path, "wb") as f:
                f.write(mutant)
            got = run("list", path)
            if _oracle_accepts(mutant):
                if got.returncode != 0 or got.stdout != format_listing(mutant).encode():
                    failures.append(f"{name} mutant {j}: the oracle lists it, the C tool does not")
            elif got.returncode != 2 or got.stdout or not _gives_a_reason(got.stderr):
                failures.append(
                    f"{name} mutant {j}: the oracle refuses it, the C tool exits {got.returncode}"
                )
    return failures


def _build(tmp: str) -> str | None:
    from bcir.toolchain import host_c_compiler

    cc = host_c_compiler()
    if cc is None:
        return None
    exe = os.path.join(tmp, "bcir-bundle.exe" if os.name == "nt" else "bcir-bundle")
    sources = [os.path.join(_RUNTIME_C, s) for s in _SOURCES]
    last = None
    for std in ("-std=c23", "-std=c2x"):  # GCC 13 knows C23 only as c2x
        last = subprocess.run(
            [cc, std, "-O2", "-Wall", "-Wextra", "-I", _RUNTIME_C, *sources, "-o", exe],
            capture_output=True,
            text=True,
        )
        if last.returncode == 0:
            return exe
    raise AssertionError(f"bcir-bundle did not build: {last.stderr if last else ''}")


def test_the_fixture_builds_what_the_gate_links():
    from bcir.tests.gate_links import assert_fixture_links

    assert_fixture_links("tools", "bcir-bundle", _SOURCES)


def test_the_corpus_reaches_every_field_of_the_listing():
    bundles = [decode_bundle(blob) for _, blob in corpus()]
    variants = [v for b in bundles for v in b.variants]
    assert any(v.priority < 0 for v in variants) and any(v.cal_gen == _U64 for v in variants)
    assert all(
        any(getattr(v, flag) for v in variants)
        for flag in ("executable", "r12_attested", "portable", "debug")
    )
    assert any(v.target_manifest_sha256 for v in variants)
    assert any(v.prohibited_features for v in variants)
    assert any(not b.root_variant_id and not b.default_variant_id for b in bundles)
    assert any(len(v.payload) % 8 for v in variants), "a payload that needs alignment padding"
    views = {v.kind for v in variants} | {v.format for v in variants}
    assert ArtifactKind.STREAM_PACK in views and ArtifactFormat.TEXT in views
    assert ArtifactFormat.ELF in views, "a payload whose view is delegated"


def test_the_c_tool_prints_and_selects_as_the_oracle_does():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return  # no C compiler on this tier (quick hides it); the c-runtime tier and the gate run it
        failures = check_tool(exe, tmp, mutations=24 if os.environ.get("BCIR_THOROUGH") else 8)
        assert not failures, "\n".join(failures[:20])


def test_the_two_command_lines_write_a_text_payload_as_utf8_whatever_the_console():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        path = os.path.join(tmp, "bundle.bcab")
        with open(path, "wb") as f:
            f.write(encode_bundle(_many()))
        env = dict(os.environ, PYTHONIOENCODING="ascii", PYTHONPATH=_ROOT)
        shown = {}
        for command in ("dis", "mnemonics"):
            args = [command, path, "--id", "a-asm"]
            oracle = subprocess.run(
                [sys.executable, "-m", "bcir.abi.artifact_tool", *args],
                capture_output=True,
                env=env,
                timeout=60,
            )
            native = subprocess.run([exe, *args], capture_output=True, timeout=60)
            assert oracle.returncode == 0, oracle.stderr
            assert native.returncode == 0, native.stderr
            assert oracle.stdout == native.stdout
            shown[command] = native.stdout
        assert "\u00e9t\u00e9".encode() in shown["dis"]  # the text's own octets, not an escape
        assert shown["mnemonics"].endswith(b"\nlast\n")


def test_the_c_command_line_refuses_what_it_does_not_twin():
    blob = encode_bundle(_single())
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        path = os.path.join(tmp, "bundle.bcab")
        with open(path, "wb") as f:
            f.write(blob)
        cases = (
            (["--max-bytes", "127", "list", path], b"--max-bytes must be in"),
            ([f"--max-bytes={(1 << 30) + 1}", "list", path], b"--max-bytes must be in"),
            ([f"--max-bytes={len(blob) - 1}", "list", path], b"exceeds --max-bytes"),
            (["list", os.path.join(tmp, "nowhere")], b"cannot open"),
            (["hexdump", path, "--width", "0"], b"width"),
            (["to-der", path, os.path.join(tmp, "o")], b"ASN.1"),
            (["from-oer", path, os.path.join(tmp, "o")], b"ASN.1"),
            (["list", path, "--id", "only"], b"unknown option"),
            (["extract", path], b"OUTPUT"),
            (["select", path, "--endianness", "middle"], b"selector"),
            (["select", path, "--pointer-bits", "-1"], b"selector"),
            (["inspect", path], b"command"),
        )
        for args, reason in cases:
            got = subprocess.run([exe, *args], capture_output=True, timeout=60)
            assert got.returncode == 2 and not got.stdout, (args, got)
            assert got.stderr.splitlines()[-1].startswith(b"bcir-bundle: "), (args, got.stderr)
            assert reason in got.stderr, (args, got.stderr)
        ok = subprocess.run(
            [exe, f"--max-bytes={len(blob)}", "select", path, "--id", "only"],
            capture_output=True,
            timeout=60,
        )
        assert ok.returncode == 0 and ok.stdout == b"only\n"
