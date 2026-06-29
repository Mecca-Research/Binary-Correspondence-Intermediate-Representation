"""ASM3 — memory-fence (hardware barrier) intrinsics as typed, kinded, BARRIERED edges with REAL per-ISA
assembly emit behind `--target`.

The fence intrinsic is already a recognized `barriered` claim (ASM1/§5.8 set the pattern); ASM3 deepens it
two ways, mirroring ASM2's port-I/O slice:

  * TYPED KINDS — the FULL (seq_cst) fence keeps the BACKWARD-COMPATIBLE op string `c.fence` (so the
    existing `__atomic_thread_fence` / `__sync_synchronize` claims, and the C-twin parity corpus, are
    unchanged — no digest/parity churn), and two NEW op strings carry the lighter kinds: `c.fence.acquire`
    (load fence) and `c.fence.release` (store fence). The kind is read off the x86-conventional intrinsic
    NAME (`_mm_mfence`/`_mm_lfence`/`_mm_sfence`) — no `memory_order` argument is parsed.
  * PER-ISA EMIT — the bare portable `__atomic_thread_fence(__ATOMIC_SEQ_CST);` is replaced by the real
    hardware-barrier instruction behind a GNU `__asm__ __volatile__ (... ::: "memory")`, keyed off the
    function's `--target` (x86 → mfence/lfence/sfence; aarch64 → dmb ish/ishld/ishst; riscv64 → fence
    rw,rw / r,rw / rw,w). The `"memory"` clobber is the REQUIRED compiler-barrier half of the fence. Every
    ISA HAS a fence, so a target outside the three families keeps the portable emit as an honest default
    (NOT a fallback — unlike port I/O, there is no unsupported diagnostic here).

**The carried-forward lesson (per-ISA assemble, host-arch-gated).** Barriers are per-ISA, and you cannot
cross-assemble (the aarch64 CI runner has no x86 sysroot, and vice-versa). So this gate ASSEMBLES each fence
for the HOST's OWN native arch only (`gcc`/`clang -c`, assemble-only — these are plain memory barriers, but
the assemble proves the emitted mnemonic is valid for the toolchain), and for NON-native targets asserts the
emit TEXT (arch-independent — e.g. the aarch64 emit contains `dmb ish`) WITHOUT assembling. This gives real
assembled coverage on EVERY CI lane (x86 lanes assemble mfence/lfence/sfence; the aarch64 lane assembles dmb
ish/ishld/ishst) — strictly better than ASM2 (which could only run on x86). Self-skips cleanly when no
compiler is present.
"""

import os
import platform
import re
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront import compile_unit
from bcir.frontends.cfront.emit import emit_function
from bcir.model import Opcode

_GCC = shutil.which("gcc")
_CLANG = shutil.which("clang")
_CC = _CLANG or shutil.which("cc") or _GCC

# The host arch -> the matching `--target` for the NATIVE assemble. A fence is emitted as native assembly, so
# it can only be assembled for the host's own ISA (the aarch64 CI lane has no x86 sysroot, and vice-versa).
# Pick the native target by `platform.machine()`; a host outside the three families assembles nothing (the
# emit-text assertions are arch-independent and cover every kind regardless).
_MACHINE = platform.machine().lower()
if _MACHINE in ("x86_64", "amd64", "x86", "i386", "i486", "i586", "i686"):
    _NATIVE_TARGET = "x86_64-linux"
elif _MACHINE in ("aarch64", "arm64"):
    _NATIVE_TARGET = "aarch64-linux"
elif _MACHINE in ("riscv64",):
    _NATIVE_TARGET = "riscv64-linux"
else:
    _NATIVE_TARGET = None

# the per-target native fence mnemonics per kind (full / acquire / release) -- the arch-independent emit-TEXT
# contract (asserted on every host; only the native family is also assembled). The three x86 ABIs share the
# x86 mnemonics (emit.py keys fences by ISA FAMILY, not the specific ABI), but each is listed explicitly so a
# per-target lookup asserts every target against its OWN entry (and a future per-ABI mnemonic split has a slot).
_X86_FENCE = {"full": "mfence", "acquire": "lfence", "release": "sfence"}
_FENCE_MNEM = {
    "x86_64-linux":  _X86_FENCE, "i386-linux": _X86_FENCE, "x86_64-windows": _X86_FENCE,
    "aarch64-linux": {"full": "dmb ish", "acquire": "dmb ishld", "release": "dmb ishst"},
    "riscv64-linux": {"full": "fence rw,rw", "acquire": "fence r,rw", "release": "fence rw,w"}}

# the intrinsic name -> (op string, kind) it lowers to. The full-fence names keep `c.fence` (backward-compat);
# the x86-conventional _mm_{l,s}fence names carry the lighter kinds.
_INTRINSIC_KIND = {
    "__sync_synchronize": ("c.fence", "full"),
    "__atomic_thread_fence": ("c.fence", "full"),
    "atomic_thread_fence": ("c.fence", "full"),               # C11 <stdatomic.h>
    "_mm_mfence": ("c.fence", "full"),
    "_mm_lfence": ("c.fence.acquire", "acquire"),
    "_mm_sfence": ("c.fence.release", "release")}


# --- helpers ------------------------------------------------------------------------------------

def _lf(src: str, fn: str | None = None, target: str | None = None):
    """Compile `src` (no Clang check) and return the (result, lowered-function) for `fn` (default: the
    last function)."""
    r = compile_unit(src, check_clang=False, target=target)
    fns = r.lowered.functions
    lf = fns[fn] if fn else fns[next(reversed(fns))]
    return r, lf


def _fence_claims(lf):
    return [c for c in lf.claims if c.op == "c.fence" or c.op.startswith("c.fence.")]


def _emit_clean(lf) -> str:
    """The emitted function body (attestation comment stripped)."""
    return re.sub(r"/\*.*?\*/\n?", "", emit_function(lf), flags=re.S)


def _call_of(intrinsic: str) -> str:
    """A one-line fixture invoking `intrinsic` (the *_thread_fence forms take a memory-order arg)."""
    arg = "5" if intrinsic.endswith("thread_fence") else ""    # __ATOMIC_SEQ_CST / memory_order_seq_cst == 5
    return f"void f(void){{ {intrinsic}({arg}); }}"


def _assemble_native(c_text: str) -> str:
    """ASSEMBLE `c_text` (the NATIVE-target fence emit) to an object under every available CC at
    `-std=c11 -pedantic -c` (assemble-only). The emit is native assembly for THIS host's ISA (the target is
    `_NATIVE_TARGET`), so a native `gcc`/`clang` assembles it without a cross sysroot. Returns "ok" iff every
    available compiler builds the object, "skip" if no compiler or no native target, else "FAIL:<cc>:<detail>".
    Non-native fence emits are NOT assembled here (the aarch64 lane has no x86 sysroot, and vice-versa) -- their
    validity is proven on their own CI lane, and their emit TEXT is asserted arch-independently below."""
    if _NATIVE_TARGET is None:
        return "skip"                                          # a host outside the three families: text-only
    ccs = [(n, p) for n, p in (("gcc", _GCC), ("clang", _CLANG)) if p]
    if not ccs:
        if not _CC:
            return "skip"
        ccs = [("cc", _CC)]
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "barrier.c")
        with open(src, "w", encoding="utf-8") as f:
            f.write(c_text)
        for name, cc in ccs:
            obj = os.path.join(d, f"barrier_{name}.o")
            b = subprocess.run([cc, "-std=c11", "-pedantic", "-c", src, "-o", obj],  # -c: assemble only
                               capture_output=True, text=True)
            if b.returncode != 0:
                return f"FAIL:{name}:assemble:{(b.stderr or b.stdout).strip().splitlines()[-1:]}"
            if not os.path.exists(obj):
                return f"FAIL:{name}:no-object"
    return "ok"


# --- (1) recognition + kind --------------------------------------------------------------------

def test_all_six_intrinsics_recognized_with_the_right_op_string():
    for intrinsic, (op, _kind) in _INTRINSIC_KIND.items():
        _r, lf = _lf(_call_of(intrinsic))
        cs = _fence_claims(lf)
        assert len(cs) == 1, f"{intrinsic}: expected exactly one fence claim, got {len(cs)}"
        assert cs[0].op == op, f"{intrinsic}: expected op {op!r}, got {cs[0].op!r}"


def test_full_fence_keeps_the_backward_compatible_c_fence_op_string():
    # the full (seq_cst) fence MUST stay `c.fence` -- the existing __atomic_thread_fence / __sync_synchronize
    # corpus + the C-twin parity digest depend on it being unchanged (no new op string for the full kind).
    for intrinsic in ("__sync_synchronize", "__atomic_thread_fence", "atomic_thread_fence", "_mm_mfence"):
        _r, lf = _lf(_call_of(intrinsic))
        assert _fence_claims(lf)[0].op == "c.fence", intrinsic


def test_lighter_kinds_get_the_new_op_strings():
    _r, lf = _lf(_call_of("_mm_lfence"))
    assert _fence_claims(lf)[0].op == "c.fence.acquire"
    _r, lf = _lf(_call_of("_mm_sfence"))
    assert _fence_claims(lf)[0].op == "c.fence.release"


def test_c11_atomic_thread_fence_is_recognized():
    # the C11 <stdatomic.h> `atomic_thread_fence` was NOT recognized before ASM3 (only the __ builtins were).
    _r, lf = _lf("void f(void){ atomic_thread_fence(5); }")
    assert _fence_claims(lf)[0].op == "c.fence", "C11 atomic_thread_fence must lower to the full fence"


# --- (2) the barriered / opcode / lane contract ------------------------------------------------

def test_every_fence_kind_is_barriered():
    for intrinsic in _INTRINSIC_KIND:
        _r, lf = _lf(_call_of(intrinsic))
        c = _fence_claims(lf)[0]
        assert c.hazard == "barriered", f"{intrinsic}: a memory fence must be barriered, got {c.hazard}"


def test_every_fence_is_a_barrier_opcode_on_lane_a():
    for intrinsic in _INTRINSIC_KIND:
        _r, lf = _lf(_call_of(intrinsic))
        c = _fence_claims(lf)[0]
        assert c.opcode == Opcode.BARRIER, f"{intrinsic}: a fence must be the BARRIER opcode, got {c.opcode}"
        assert c.lane.name == "A", f"{intrinsic}: a scalar fence lowers on lane A, got {c.lane.name}"


def test_fence_is_off_the_legality_value_path():
    # a trusted opaque effect: no R-law verdict, the unit is clean (like c.asm / c.portio / c.call.libm).
    r, _lfn = _lf("void f(void){ _mm_mfence(); _mm_lfence(); _mm_sfence(); }")
    assert r.is_clean and not r.diagnostics


def test_unused_fence_is_not_eliminated():
    # a side-effecting barrier whose result temp is unused must still emit (the emit never drops a claim).
    # An explicit target realizes the native asm form (the portable default emit is exercised separately).
    r, lf = _lf("void f(void){ __sync_synchronize(); }", target="x86_64-linux")
    assert _fence_claims(lf), "the fence claim was dropped from the lowered graph"
    assert "__asm__ __volatile__" in r.emitted["f"], "the fence was eliminated from the emit"


# --- (3) per-target emit text (arch-independent — asserted on every host) -----------------------

def test_x86_targets_emit_the_real_fence_instruction():
    for target in ("x86_64-linux", "i386-linux", "x86_64-windows"):
        for intrinsic, (_op, kind) in _INTRINSIC_KIND.items():
            _r, lf = _lf(_call_of(intrinsic), target=target)
            body = _emit_clean(lf)
            mnem = _FENCE_MNEM[target][kind]               # each x86 ABI asserts against its own entry
            assert f'__asm__ __volatile__ ("{mnem}" ::: "memory");' in body, (target, intrinsic, body)


def test_aarch64_target_emits_the_dmb_barrier():
    for intrinsic, (_op, kind) in _INTRINSIC_KIND.items():
        _r, lf = _lf(_call_of(intrinsic), target="aarch64-linux")
        body = _emit_clean(lf)
        mnem = _FENCE_MNEM["aarch64-linux"][kind]
        assert f'__asm__ __volatile__ ("{mnem}" ::: "memory");' in body, (intrinsic, body)


def test_riscv64_target_emits_the_fence_barrier():
    for intrinsic, (_op, kind) in _INTRINSIC_KIND.items():
        _r, lf = _lf(_call_of(intrinsic), target="riscv64-linux")
        body = _emit_clean(lf)
        mnem = _FENCE_MNEM["riscv64-linux"][kind]
        assert f'__asm__ __volatile__ ("{mnem}" ::: "memory");' in body, (intrinsic, body)


def test_every_emit_carries_the_required_memory_clobber():
    # the `"memory"` clobber is the compiler-barrier half of the fence -- it must be present for every kind
    # on every ISA family (else the compiler could reorder accesses across the hardware fence).
    for target in ("x86_64-linux", "aarch64-linux", "riscv64-linux"):
        for intrinsic in _INTRINSIC_KIND:
            _r, lf = _lf(_call_of(intrinsic), target=target)
            assert '::: "memory"' in _emit_clean(lf), (target, intrinsic)


def test_unknown_target_keeps_the_portable_default():
    # every ISA HAS a fence, so an EXPLICIT target outside the three families is NOT an unsupported diagnostic
    # (unlike port I/O) -- it keeps the portable __atomic_thread_fence(__ATOMIC_SEQ_CST) as an honest default.
    # The shipping ABIs are all in the three families, so this default is reached only by a hand-set target.
    r, lf = _lf("void f(void){ __sync_synchronize(); }")       # default (host) target
    lf.target = "mips-linux"                                   # outside x86/aarch64/riscv64 -> portable default
    lf.target_explicit = True                                  # force the explicit-target path so we test the
    body = _emit_clean(lf)                                     #   per-ISA dispatch falling through to portable
    assert "__atomic_thread_fence(__ATOMIC_SEQ_CST);" in body, body
    assert "__asm__" not in body, "the portable default must not emit inline asm"


def test_default_host_target_emits_the_portable_fence():
    # with NO explicit --target, the fence emits the PORTABLE __atomic_thread_fence -- so the default emit
    # compiles on ANY host (a cross-arch native compile never sees foreign asm). Native per-ISA asm is the
    # opt-in behaviour of an explicit `--target` (asserted by the per-target tests above).
    r, lf = _lf("void f(void){ __sync_synchronize(); _mm_lfence(); _mm_sfence(); }")   # no target
    body = _emit_clean(lf)
    assert body.count("__atomic_thread_fence(__ATOMIC_SEQ_CST);") == 3, body
    assert "__asm__" not in body, "the host-default fence emit must be portable, not inline asm"


# --- (4) source-order ordering (the barriered contract is realized by in-order emit) ------------

def test_two_fences_stay_in_source_order_in_the_emit():
    # the cfront emit walks the body in source order and never reorders/drops a claim -- so a release then an
    # acquire fence emit in that order (the `barriered` ordering is realized by source-order emission).
    _r, lf = _lf("void f(void){ _mm_sfence(); _mm_lfence(); }", target="x86_64-linux")
    body = _emit_clean(lf)
    assert body.index("sfence") < body.index("lfence"), f"the fences were reordered in the emit:\n{body}"


# --- (5) a user-defined function named like an intrinsic stays a normal call --------------------

def test_user_defined_atomic_thread_fence_is_not_treated_as_the_intrinsic():
    # if the unit DEFINES its own `atomic_thread_fence`, the call resolves to it (a real R18 edge), not the
    # fence intrinsic -- the same shadowing guard the stdlib / port-I/O intrinsics use (callee in func_rets).
    src = ("unsigned atomic_thread_fence(unsigned o){ return o + 1u; } "
           "unsigned f(void){ return atomic_thread_fence(5u); }")
    _r, lf = _lf(src, fn="f")
    assert not _fence_claims(lf), "a user-defined atomic_thread_fence must not lower to a fence edge"
    assert any(c.op == "c.call:atomic_thread_fence" for c in lf.claims), "the user fn should be a normal call"


# --- (6) the emitted NATIVE-arch fence ASSEMBLES under gcc + clang ------------------------------

_PROG = """
void b_full(void){ __sync_synchronize(); }
void b_full_c11(void){ atomic_thread_fence(5); }
void b_mfence(void){ _mm_mfence(); }
void b_lfence(void){ _mm_lfence(); }
void b_sfence(void){ _mm_sfence(); }
"""


def test_emitted_native_fence_assembles_under_gcc_and_clang():
    # the honest seam: the emitted NATIVE-arch hardware barrier is REAL assembly the toolchain accepts
    # (assemble-only, `-c`). On an x86 lane this assembles mfence/lfence/sfence; on the aarch64 lane it
    # assembles dmb ish/ishld/ishst. NON-native targets are NOT assembled here (no cross sysroot) -- their
    # emit text is asserted above. Self-skips if no compiler / a host outside the three families.
    if _NATIVE_TARGET is None:
        return
    r = compile_unit(_PROG, check_clang=False, target=_NATIVE_TARGET)
    emit = "#include <stdint.h>\n" + "\n\n".join(
        _emit_clean(lf) for lf in r.lowered.functions.values())
    verdict = _assemble_native(emit)
    assert verdict in ("ok", "skip"), f"emitted native fence did not assemble: {verdict}"


def test_each_native_fence_kind_assembles_individually():
    # assemble each KIND on its own (full / acquire / release) for the native target, so a single bad mnemonic
    # is isolated rather than masked by a passing sibling in the combined program above.
    if _NATIVE_TARGET is None:
        return
    for intrinsic in ("_mm_mfence", "_mm_lfence", "_mm_sfence"):
        r = compile_unit(_call_of(intrinsic), check_clang=False, target=_NATIVE_TARGET)
        emit = "#include <stdint.h>\n" + _emit_clean(r.lowered.functions["f"])
        verdict = _assemble_native(emit)
        assert verdict in ("ok", "skip"), f"{intrinsic}: native fence did not assemble: {verdict}"


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    tests = sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))
    passed = failed = 0
    for name in tests:
        try:
            getattr(mod, name)()
            passed += 1
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
