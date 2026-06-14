"""The library API facade (library-first): plan -> a deployable kernel artifact
(C source + ABI header + metadata + R12 attestation), usable AOT or embedded."""

import shutil

from bcir.api import build_artifact, compile_kernel


def test_artifact_carries_metadata_and_r12_attestation():
    a = build_artifact("vector_add", target="x86_avx512", theta="cool")
    assert a.program == "vector_add" and a.width == 16 and a.op == "+"
    assert a.score == 7808 and a.manifest_digest != 0
    assert a.attested and a.diagnostics == ()           # R12 clean


def test_header_declares_the_kernel_abi():
    a = build_artifact("vector_add")
    assert "#ifndef BCIR_BCIR_KERNEL_H" in a.header_c
    assert "void bcir_kernel(const float *restrict A" in a.header_c
    assert "size_t n);" in a.header_c


def test_kernel_is_the_c23_backend_output():
    a = build_artifact("vector_add", theta="hot")        # hot -> vec8
    assert a.width == 8 and "width=8" in a.kernel_c
    assert "restrict" in a.kernel_c and "static_assert" in a.kernel_c


def test_metadata_json_excludes_source():
    a = build_artifact("vector_add")
    md = a.metadata()
    assert "kernel_c" not in md and "header_c" not in md
    assert md["attested"] is True and md["width"] == 16


def test_to_files_writes_kernel_and_header(tmp_path=None):
    import tempfile, os
    a = build_artifact("vector_add")
    d = tempfile.mkdtemp(prefix="bcir-api-")
    try:
        paths = a.to_files(d)
        assert os.path.exists(paths["kernel"]) and os.path.exists(paths["header"])
        assert "bcir_kernel" in open(paths["kernel"]).read()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_compile_kernel_aot_self_checks():
    if shutil.which("clang") is None and shutil.which("cc") is None and shutil.which("gcc") is None:
        return  # skip cleanly without a C compiler
    art, (ok, out) = compile_kernel("vector_add", run=True, theta="cool")
    assert art.attested and ok, out
    assert "OK" in out
