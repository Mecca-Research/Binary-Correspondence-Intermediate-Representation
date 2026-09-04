"""Phase 4 — cross-platform ABI matrix (segment 2a). The frontend's type layout is parameterized by a
`TargetABI` instead of a single hard-coded host data model: `long` / pointer / `long double` follow
the selected target (LP64 vs LLP64 vs ILP32), while the host default stays byte-identical to before.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront.abi import HOST, TARGETS, target
from bcir.frontends.cfront.ctype_model import AggregateBuilder, pointer, scalar


def test_matrix_data_models_match_the_documented_abis():
    expect = {  # target -> (long, pointer, long double size, long double align)
        "x86_64-linux": (8, 8, 16, 16),
        "aarch64-linux": (8, 8, 16, 16),
        "riscv64-linux": (8, 8, 16, 16),
        "x86_64-windows": (4, 8, 8, 8),  # LLP64: long is 4
        "i386-linux": (4, 4, 12, 4),  # ILP32: pointers are 4, long double 12/4-aligned
    }
    for name, (lng, ptr, lds, lda) in expect.items():
        t = target(name)
        assert scalar("long", t).size == lng
        assert pointer(scalar("int", t), t).size == ptr
        ld = scalar("long double", t)
        assert (ld.size, ld.align) == (lds, lda)


def test_host_abi_is_byte_identical_to_the_default():
    # scalar(name) / pointer(of) with no ABI must equal the HOST-parameterized result, so nothing
    # that already uses the type model changes behaviour.
    for nm in (
        "void",
        "_Bool",
        "char",
        "short",
        "int",
        "long",
        "long long",
        "unsigned long",
        "size_t",
        "intptr_t",
        "float",
        "double",
        "long double",
        "uint32_t",
    ):
        assert scalar(nm) == scalar(nm, HOST)
    assert pointer(scalar("int")) == pointer(scalar("int"), HOST)


def test_the_named_lp64_targets_have_identical_layouts():
    # x86-64 / AArch64 / RISC-V differ in long-double format + calling convention (delegated), not in
    # the layout the frontend computes -- so their scalar sizes coincide.
    lp64 = [TARGETS[n] for n in ("x86_64-linux", "aarch64-linux", "riscv64-linux")]
    for nm in ("long", "long double", "size_t", "intptr_t"):
        sizes = {scalar(nm, t).size for t in lp64}
        assert len(sizes) == 1


def _struct_long_char(abi) -> int:
    b = AggregateBuilder("struct", "S")
    b.members.append(("a", scalar("long", abi), 0, 0))  # (name, CType, bit_width, req_align)
    b.members.append(("b", scalar("char", abi), 0, 0))
    return b.build().size


def test_struct_layout_follows_the_selected_data_model():
    assert _struct_long_char(target("x86_64-linux")) == 16  # long 8 + char + 7 pad
    assert _struct_long_char(target("x86_64-windows")) == 8  # long 4 + char + 3 pad (LLP64)
    assert _struct_long_char(target("i386-linux")) == 8  # long 4 + char + 3 pad (ILP32)


def test_unknown_target_is_rejected():
    try:
        target("sparc-solaris")
        raise AssertionError("expected KeyError for an unknown target")
    except KeyError:
        pass


def _clang_sizes():
    """(sizeof long, void*, long double) from the host C compiler, or None if unavailable."""
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return None
    src = (
        "#include <stdio.h>\nint main(void){"
        'printf("%zu %zu %zu\\n", sizeof(long), sizeof(void*), sizeof(long double));return 0;}\n'
    )
    with tempfile.TemporaryDirectory() as d:
        s, exe = os.path.join(d, "a.c"), os.path.join(d, "a.out")
        with open(s, "w") as f:
            f.write(src)
        if subprocess.run([cc, "-O0", s, "-o", exe], capture_output=True).returncode != 0:
            return None
        run = subprocess.run([exe], capture_output=True, text=True)
        if run.returncode != 0:
            return None
        return tuple(int(x) for x in run.stdout.split())


def test_host_abi_matches_clang_sizeof():
    sizes = _clang_sizes()
    if sizes is None:
        return  # no C compiler -> skip cleanly
    lng, ptr, ldbl = sizes
    assert HOST.long_size == lng and HOST.pointer_size == ptr
    assert HOST.long_double_size == ldbl  # LP64 host: 16 on x86-64 and AArch64


# --- segment 2b: target selection threaded through compile_unit -----------------------------------

_STRUCT = "struct S { long a; char b; }; unsigned f(void){ struct S s; return sizeof(s); }\n"


def test_compile_unit_defaults_to_the_host_and_checks_equivalence():
    from bcir.frontends.cfront import compile_unit

    r = compile_unit(_STRUCT, check_clang=True)
    assert r.target == "x86_64-linux" and r.is_clean
    assert r.lowered.aggregates["S"].size == 16  # LP64 layout
    assert r.equivalence in ("match", "skip") or r.equivalence.startswith("skip")


def test_compile_unit_lays_out_for_a_selected_target():
    from bcir.frontends.cfront import compile_unit

    rw = compile_unit(_STRUCT, check_clang=True, target="x86_64-windows")
    assert rw.target == "x86_64-windows" and rw.is_clean  # still R1-R18 clean
    assert rw.lowered.aggregates["S"].size == 8  # LLP64: long is 4
    assert rw.equivalence == "skip:cross-target:x86_64-windows"  # not host-clang-compatible
    ri = compile_unit(_STRUCT, check_clang=False, target="i386-linux")
    assert ri.lowered.aggregates["S"].size == 8 and ri.target == "i386-linux"


def test_lp64_targets_compile_to_identical_layouts():
    from bcir.frontends.cfront import compile_unit

    sizes = {
        t: compile_unit(_STRUCT, check_clang=False, target=t).lowered.aggregates["S"].size
        for t in ("x86_64-linux", "aarch64-linux", "riscv64-linux")
    }
    assert set(sizes.values()) == {16}  # the LP64 trio coincide


def test_compile_unit_rejects_an_unknown_target():
    from bcir.frontends.cfront import compile_unit

    try:
        compile_unit(_STRUCT, check_clang=False, target="sparc-solaris")
        raise AssertionError("expected KeyError for an unknown target")
    except KeyError:
        pass
