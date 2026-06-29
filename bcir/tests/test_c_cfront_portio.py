"""ASM2 — port-mapped I/O intrinsics (inb/inw/inl / outb/outw/outl) as a typed, isolated, BARRIERED
I/O-port edge in the cfront C subset.

Unlike raw inline asm (ASM1's opaque pass-through), port I/O has KNOWN semantics — so BCIR models it as
a **typed port-access trusted edge** carrying (port number, access width, direction), isolated under the
I/O address space (the MMIO domain, so a port access never aliases normal memory), `barriered` (volatile
+ ordered: two port ops must never be reordered/fused/eliminated), and OFF the legality value-path (a
trusted effect, no R-law verdict). The per-ISA `in`/`out` instruction is emitted behind the `--target`
flag: x86 targets emit the real instruction (as a GNU `__asm__ __volatile__`, reusing the ASM1 trusted-
edge + barrier machinery); non-x86 targets (aarch64 / riscv64) have NO port I/O (only MMIO), so they raise
an honest unsupported diagnostic → the LLVM fallback. This gate covers:

  * recognition of all six intrinsics as CALLs (no new syntax), with the right result types (u8/u16/u32)
    and the void writes;
  * the Linux `out*(value, port)` argument order (value FIRST, port SECOND) — pinned so a reversed
    convention would fail;
  * arity + operand-type diagnostics;
  * the edge is `barriered` + ISOLATED (port access in the MMIO I/O-port space, not aliasing a normal
    pointer; two port ops do not commute);
  * off the legality value-path (no Diagnostic, `is_clean`);
  * per-target emit — x86 emits the right `in`/`out` instruction (asserted on the emitted text); a non-x86
    target (`--target aarch64-linux`) raises the honest unsupported diagnostic;
  * the emitted x86 asm ASSEMBLES under gcc + clang (`-c`, assemble-only — NOT linked/run).

**Honest boundary (privileged execution).** Executing `in`/`out` from userspace traps (it needs iopl/
ioperm + ring-0). So this gate ASSEMBLES the emitted asm (`gcc -c` / `clang -c`, `-std=c11 -pedantic`) to
prove the emitted instruction is VALID x86 the toolchain accepts — it does NOT link or run it. That is the
honest seam: the emitted instruction is real and assembles; execution is privileged and gated (like the
SYCL device path). Self-skips cleanly when no compiler is present.
"""

import os
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront import compile_unit, diagnose
from bcir.frontends.cfront.emit import emit_function
from bcir.frontends.cfront.pipeline import compile_with_fallback
from bcir.model import Domain

import platform

_GCC = shutil.which("gcc")
_CLANG = shutil.which("clang")
_CC = _CLANG or shutil.which("cc") or _GCC
# The emitted asm is x86 `in`/`out`, so it must be assembled FOR an x86 target regardless of the host arch
# (the CI matrix includes an aarch64 lane, whose native assembler rejects x86 instructions). clang
# cross-assembles via `-target x86_64-linux-gnu` on ANY host; a native gcc can only assemble x86 on an x86 host.
_HOST_X86 = platform.machine().lower() in ("x86_64", "amd64", "x86", "i386", "i486", "i586", "i686")


# --- helpers ------------------------------------------------------------------------------------

def _lf(src: str, fn: str | None = None, target: str | None = None):
    """Compile `src` (no Clang check) and return the (result, lowered-function) for `fn` (default: the
    last function)."""
    r = compile_unit(src, check_clang=False, target=target)
    fns = r.lowered.functions
    lf = fns[fn] if fn else fns[next(reversed(fns))]
    return r, lf


def _portio_claims(lf):
    return [c for c in lf.claims if c.op.startswith("c.portio.")]


def _emit_clean(lf) -> str:
    """The emitted function body (attestation comment stripped)."""
    import re

    return re.sub(r"/\*.*?\*/\n?", "", emit_function(lf), flags=re.S)


def _assemble_only(c_text: str) -> str:
    """ASSEMBLE `c_text` to an object under BOTH gcc AND clang at `-std=c11 -pedantic -c` (assemble-only —
    NEVER linked or run, because `in`/`out` are privileged ring-0 instructions). The emitted asm is x86
    `in`/`out`, so it is assembled FOR an x86 target regardless of the host arch: clang cross-assembles via
    `-target x86_64-linux-gnu` on any host (incl. the aarch64 CI lane); a native gcc can only assemble x86 on
    an x86 host. Returns "ok" iff every x86-capable compiler builds the object, "skip" if none is available,
    else "FAIL:<cc>:<detail>". This is the honest seam: the emitted x86 `in`/`out` must be VALID assembly."""
    ccs = []
    if _CLANG:
        ccs.append(("clang", _CLANG, ["-target", "x86_64-linux-gnu"]))   # clang cross-assembles on any host
    if _GCC and _HOST_X86:
        ccs.append(("gcc", _GCC, []))                                    # native gcc only on an x86 host
    if not ccs:
        if _CC and _HOST_X86:                                            # a non-clang `cc` only if x86
            ccs = [("cc", _CC, [])]
        else:
            return "skip"                                                # no x86-capable assembler here
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "portio.c")
        with open(src, "w", encoding="utf-8") as f:
            f.write(c_text)
        for name, cc, tflag in ccs:
            obj = os.path.join(d, f"portio_{name}.o")
            b = subprocess.run([cc, *tflag, "-std=c11", "-pedantic", "-c", src, "-o", obj],  # -c: assemble only
                               capture_output=True, text=True)
            if b.returncode != 0:
                return f"FAIL:{name}:assemble:{(b.stderr or b.stdout).strip().splitlines()[-1:]}"
            if not os.path.exists(obj):
                return f"FAIL:{name}:no-object"
    return "ok"


# --- (1) recognition + result types -------------------------------------------------------------

def test_all_six_intrinsics_are_recognized_as_portio_edges():
    src = ("unsigned a(void){ return inb(0x60); } unsigned b(void){ return inw(0x64); }"
           "unsigned c(void){ return inl(0x1f0); } void d(unsigned v){ outb(v, 0x60); }"
           "void e(unsigned v){ outw(v, 0x64); } void f(unsigned v){ outl(v, 0x1f0); }")
    r = compile_unit(src, check_clang=False)
    ops = set()
    for lf in r.lowered.functions.values():
        for c in _portio_claims(lf):
            ops.add(c.op.split(":")[0])
    assert ops == {"c.portio.in.b", "c.portio.in.w", "c.portio.in.l",
                   "c.portio.out.b", "c.portio.out.w", "c.portio.out.l"}, ops


def test_read_result_types_are_u8_u16_u32():
    for fn, width, name in (("inb", 1, "uint8_t"), ("inw", 2, "uint16_t"), ("inl", 4, "uint32_t")):
        _r, lf = _lf(f"unsigned f(void){{ return {fn}(0x60); }}")
        c = _portio_claims(lf)[0]
        rt = lf.rid_types[c.wr[0]]
        assert rt.name == name and rt.size == width, (fn, rt.name, rt.size)
        assert c.imm == (width,)                    # the access width rides in imm


def test_writes_are_void_and_produce_no_result_temp():
    for fn in ("outb", "outw", "outl"):
        _r, lf = _lf(f"void f(unsigned v){{ {fn}(v, 0x60); }}")
        c = _portio_claims(lf)[0]
        # a void write: the I/O-port space is the only write (no result temp), value+port+space are reads.
        assert c.op.startswith(f"c.portio.out.{ {'outb':'b','outw':'w','outl':'l'}[fn] }:")
        assert len(c.wr) == 1 and len(c.rd) == 3    # wr = (io space); rd = (value, port, io space)


# --- (2) the Linux out*(value, port) argument order ---------------------------------------------

def test_out_argument_order_is_value_then_port():
    # the Linux <asm/io.h> convention: out*(value, port). value is the FIRST arg, port is the SECOND. A
    # reversed convention (port, value) would put `port` on rd[0] and `value` on rd[1], failing this.
    _r, lf = _lf("void f(unsigned val, unsigned prt){ outb(val, prt); }")
    c = _portio_claims(lf)[0]
    val_rid = next(rid for n, rid, _ct in lf.params if n == "val")
    prt_rid = next(rid for n, rid, _ct in lf.params if n == "prt")
    # rd = (value, port, io-space) -- value FIRST, port SECOND (the Linux order, NOT reversed).
    assert c.rd[0] == val_rid, "out*(value, port): value must be the first operand (rd[0])"
    assert c.rd[1] == prt_rid, "out*(value, port): port must be the second operand (rd[1])"


def test_out_order_is_visible_in_the_x86_emit():
    # the emit pairs value -> the accumulator "a" and port -> the "Nd" port. A reversed convention would
    # send the port to the accumulator and the value to the port (a miscompile a real driver would hit).
    _r, lf = _lf("void f(unsigned val, unsigned prt){ outw(val, prt); }")
    body = _emit_clean(lf)
    assert '"a" (val), "Nd" (prt)' in body, body          # value->accumulator, port->Nd, in source order


# --- (3) arity / type diagnostics ---------------------------------------------------------------

def test_wrong_arity_read_is_an_honest_diagnostic():
    rep = diagnose("unsigned f(void){ return inb(0x60, 0x61); }")   # inb takes exactly 1 arg
    assert not rep.ok and rep.diagnostics
    assert "inb" in rep.diagnostics[0].message and "1 argument" in rep.diagnostics[0].message


def test_wrong_arity_write_is_an_honest_diagnostic():
    rep = diagnose("void f(unsigned v){ outb(v); }")                # outb takes exactly 2 args
    assert not rep.ok and rep.diagnostics
    assert "outb" in rep.diagnostics[0].message and "2 argument" in rep.diagnostics[0].message


def test_float_port_is_an_honest_diagnostic():
    rep = diagnose("unsigned f(double p){ return inb(p); }")        # a float port is invalid
    assert not rep.ok and rep.diagnostics
    assert "port" in rep.diagnostics[0].message


def test_float_value_is_an_honest_diagnostic():
    rep = diagnose("void f(double v){ outl(v, 0x1f0); }")           # a float value is invalid
    assert not rep.ok and rep.diagnostics
    assert "value" in rep.diagnostics[0].message


# --- (4) barriered + isolated semantics ---------------------------------------------------------

def test_portio_edge_is_barriered():
    for fn, arg in (("inb", "0x60"), ("outb", "v, 0x60")):
        sig = "unsigned f(unsigned v)" if fn == "outb" else "unsigned f(void)"
        _r, lf = _lf(f"{sig}{{ {fn}({arg}); return 0; }}")
        c = _portio_claims(lf)[0]
        assert c.hazard == "barriered", f"{fn}: port I/O must be barriered, got {c.hazard}"


def test_portio_edge_is_in_the_mmio_io_domain():
    # the edge is ISOLATED under the I/O address space: domain MMIO + lane H, so it sits in the device-I/O
    # tier and never on the normal-memory legality value-path.
    _r, lf = _lf("unsigned f(void){ return inw(0x64); }")
    c = _portio_claims(lf)[0]
    assert c.domain == Domain.MMIO
    assert c.lane.name == "H"                              # the hazard/barrier lane


def test_port_access_does_not_alias_a_normal_pointer():
    # the port access rides the dedicated __ioport resource (rid 990000, MMIO); a normal pointer parameter
    # is a RAM resource in the function-local rid band. The two rid sets are DISJOINT — no aliasing.
    _r, lf = _lf("unsigned f(unsigned *p){ *p = 1u; return inb(0x60); }")
    c = _portio_claims(lf)[0]
    prid = next(rid for n, rid, _ct in lf.params if n == "p")
    io_rids = set(c.rd) | set(c.wr)
    assert prid not in io_rids, "the port access must not reference the normal pointer rid"
    io_res = lf.resources[990000]
    assert io_res.domain == Domain.MMIO and io_res.name == "__ioport"
    assert lf.resources[prid].domain == Domain.RAM        # the pointer is normal memory, disjoint


def test_two_port_ops_share_the_io_space_so_they_do_not_commute():
    # two port ops both read AND write the single __ioport resource, so there is a WAW/RAW between them — a
    # scheduler can never reorder/fuse them. (This is what keeps `outb(a,p); v=inb(q);` ordered.)
    _r, lf = _lf("unsigned f(unsigned a){ outb(a, 0x70); return inb(0x71); }")
    cs = _portio_claims(lf)
    assert len(cs) == 2
    io = 990000
    # the write writes io; the read reads AND writes io -> a dependency chain on the shared resource.
    assert io in cs[0].wr and io in cs[1].rd and io in cs[1].wr


def test_two_port_ops_stay_in_source_order_in_the_emit():
    # the cfront emit walks the body in source order and never reorders/drops a claim — so an out then an in
    # emit in that order (the `barriered` ordering is realized by source-order emission, like ASM1).
    _r, lf = _lf("unsigned f(unsigned a){ outb(a, 0x70); return inb(0x71); }")
    body = _emit_clean(lf)
    out_i = body.index("outb")
    in_i = body.index("inb")
    assert out_i < in_i, f"the out/in port ops were reordered in the emit:\n{body}"


# --- (5) off the legality value-path ------------------------------------------------------------

def test_portio_is_off_the_legality_value_path():
    # a trusted opaque effect: no R-law verdict, the unit is clean (like c.asm / c.call.libm).
    r, _lfn = _lf("unsigned f(void){ outb(0xffu, 0x80); return inb(0x60); }")
    assert r.is_clean and not r.diagnostics


def test_unused_read_result_is_not_eliminated():
    # a side-effecting port READ whose value is unused must still emit (the emit never drops a claim).
    r, lf = _lf("void f(void){ (void)inb(0x60); }")
    assert _portio_claims(lf), "the port read claim was dropped from the lowered graph"
    assert "inb" in r.emitted["f"], "the port read was eliminated from the emit"


# --- (6) per-target emit ------------------------------------------------------------------------

def test_x86_targets_emit_the_real_in_out_instruction():
    expect = {
        "inb": '"inb %w1, %b0" : "=a"', "inw": '"inw %w1, %w0" : "=a"', "inl": '"inl %w1, %k0" : "=a"',
        "outb": '"outb %b0, %w1" :  : "a"', "outw": '"outw %w0, %w1" :  : "a"',
        "outl": '"outl %k0, %w1" :  : "a"'}
    for target in ("x86_64-linux", "i386-linux", "x86_64-windows"):
        for fn, snippet in expect.items():
            if fn.startswith("in"):
                src = f"unsigned f(void){{ return {fn}(0x60); }}"
            else:
                src = f"void f(unsigned v){{ {fn}(v, 0x60); }}"
            _r, lf = _lf(src, target=target)
            body = _emit_clean(lf)
            assert "__asm__ __volatile__" in body, (target, fn, body)
            assert snippet in body, f"{target}/{fn}: expected {snippet!r} in\n{body}"


def test_non_x86_target_raises_the_honest_unsupported_diagnostic():
    # the honest diagnostic is raised at the per-ISA EMIT (reached when the target lays out non-x86), so the
    # unit routes to the LLVM fallback — port-mapped I/O genuinely does not exist on ARM / RISC-V (only MMIO).
    for target in ("aarch64-linux", "riscv64-linux"):
        r = compile_with_fallback("unsigned f(void){ return inb(0x60); }", check_clang=False, target=target)
        assert r.needs_fallback, f"{target}: port I/O should fall back (no port I/O on this ISA)"
        assert "requires an x86 target" in r.fallback and target in r.fallback
        assert "use MMIO" in r.fallback


def test_non_x86_write_also_falls_back():
    r = compile_with_fallback("void f(unsigned v){ outl(v, 0x1f0); }", check_clang=False,
                              target="aarch64-linux")
    assert r.needs_fallback and "requires an x86 target" in r.fallback


# --- (7) the emitted x86 asm ASSEMBLES under gcc + clang (honest boundary: -c, never run) --------

_PROG = """
unsigned p_inb(unsigned port){ return inb(port); }
unsigned p_inw(unsigned port){ return inw(port); }
unsigned p_inl(unsigned port){ return inl(port); }
void p_outb(unsigned v, unsigned port){ outb(v, port); }
void p_outw(unsigned v, unsigned port){ outw(v, port); }
void p_outl(unsigned v, unsigned port){ outl(v, port); }
"""


def test_emitted_x86_portio_assembles_under_gcc_and_clang():
    # the honest seam: the emitted x86 `in`/`out` is REAL assembly the toolchain accepts (assemble-only,
    # `-c`). It is NOT linked or run — `in`/`out` are privileged (ring-0 / iopl). Self-skips if no compiler.
    r = compile_unit(_PROG, check_clang=False)             # default target x86_64-linux
    emit = "#include <stdint.h>\n" + "\n\n".join(
        _emit_clean(lf) for lf in r.lowered.functions.values())
    verdict = _assemble_only(emit)
    assert verdict in ("ok", "skip"), f"emitted port-I/O did not assemble: {verdict}"


def test_emitted_x86_portio_assembles_with_a_literal_immediate_port():
    # an immediate port (`inb(0x60)`) exercises the `"Nd"` immediate-or-dx constraint's immediate leg; a
    # variable port exercises the dx leg (above). Both must assemble.
    r = compile_unit("unsigned f(void){ return inb(0x60); } void g(unsigned v){ outb(v, 0x80); }",
                     check_clang=False)
    emit = "#include <stdint.h>\n" + "\n\n".join(
        _emit_clean(lf) for lf in r.lowered.functions.values())
    verdict = _assemble_only(emit)
    assert verdict in ("ok", "skip"), f"immediate-port port-I/O did not assemble: {verdict}"


# --- (8) a user-defined function named like an intrinsic stays a normal call --------------------

def test_user_defined_inb_is_not_treated_as_the_intrinsic():
    # if the unit DEFINES its own `inb`, the call resolves to it (a real R18 edge), not the port intrinsic —
    # the same shadowing guard the stdlib/free intrinsics use (callee in self.func_rets).
    src = "unsigned inb(unsigned p){ return p + 1u; } unsigned f(void){ return inb(5u); }"
    _r, lf = _lf(src, fn="f")
    assert not _portio_claims(lf), "a user-defined inb must not lower to a port-I/O edge"
    assert any(c.op == "c.call:inb" for c in lf.claims), "the user inb should be a normal call"


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
