"""Blueprint slice U2: the polled UART driver, the simulated 16550 and the linked, running binary.

`docs/kernel/BCIR_UART_DRIVER_BLUEPRINT.md` section 5 (U2) and section 8. Three C units make one
program: the driver (`runtime/c/uart_driver.c`, its API `runtime/c/uart_driver.h`), the device
(`runtime/c/sim16550.c`) and the harness (`runtime/c/uart_main.c`). The headline: the driver, emitted
by either BCIR rail with `--linkable` and linked with the other two, prints the transcript the program
built from the driver's source prints, on every placement; and the axi16550 placement's transcript --
PG143's part and programming vector -- is the checked-in golden `runtime/c/uart_main.golden`. Each
rail's `--linkable` artifact is self-contained (CF-LINKEMIT, CF-LINKTWIN), so each is compiled alone,
as `tools/c/sections/uartdrv.sh` compiles it.

The sim's spec-fidelity tests come first, as the blueprint orders them: the device the driver runs
against is held to section 8 and to the documents' pages before anything is held to it. Each drives the
device through the harness's script console (`uart_main --script`) and drives in the failure the rule
exists to prevent. Then the driver's: both rails compile it clean and law-clean with its contracts; the
probe, the init order, the TX bursts, the RX triage and the self-test are read off the claim graph and
off the device's own view of the bus (the transcript's trace); the probe leaves the FIFOs off, and its
verdict is acted on -- a part without a working FIFO runs in 16450 mode, which the transcript drives by
forcing the verdict, since the sim answers as a 16550A; init refuses a format with set break or DLAB,
uart_tx a state no init has filled, and the self-test drains what the receiver holds first, bounded (a
stale byte; a peer that never stops sending); each of the driver's five waits gives up at its one bound,
UART_SPIN_BUDGET polls, under the sim's two fault knobs (a transmitter that never drains, a loopback
that delivers nothing), their unchanging polls collapsed to one trace line each; the FCR shadow is never
fed by a device read and never holds the self-clearing FIFO resets (H-E); a driver laid out for another
placement is refused (H-A); an untrapped build is refused.

The register window reaches the device through Clang's load/store tracing on the driver unit (see
`uart_main.c`), so the tests that run the driver need a clang (`_TRACE_CC`); the sim's need any C
compiler (`_CC`). The quick tier hides both and these tests return early;
`tools/testing/faults/uart-driver.json` requires both.

`compile_unit(..., check_clang=False)` throughout, as U0 and U1 do: `check_clang=True`'s equivalence
harness models MMIO as a 256-byte buffer, which the AXI placement's offsets (to 0x101C) leave, and which
cannot answer a polled driver's waits at all. The behaviour equivalence of the emitted driver is the
linked transcript instead: the same program, against the same device, byte for byte.
"""

import atexit
import dataclasses
import os
import re
import shutil
import subprocess
import sys
import tempfile

from bcir.frontends.cfront import compile_unit
from bcir.frontends.devices import uart_laws as ul
from bcir.frontends.devices import uart_model as um
from bcir.model import Domain
from bcir.toolchain import build_units

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_GOLDEN = os.path.join(_C, "uart_main.golden")
#: the placement the golden transcript is the reference build's for: PG143's own part
_GOLDEN_PLACEMENT = "axi16550"
_CC = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
#: load/store tracing (-fsanitize-coverage=trace-loads,trace-stores) is Clang's
_TRACE_CC = shutil.which("clang")
_TRACE = "-fsanitize-coverage=func,trace-loads,trace-stores"
_UNITS = ("uart_driver.c", "uart_driver.h", "uart_main.c", "sim16550.c", "sim16550.h")
_FUNCTIONS = {"uart_probe", "uart_init", "uart_tx", "uart_rx_drain", "uart_flush", "uart_selftest"}
_TIMEOUT = 120
#: the transcript's fixed vector: 56000 baud from the placement's reference by PG143's truncating
#: formula, 8E2, trigger 8, and the TX buffer
_BAUD = 56000
_LCR_8E2 = 0x1F
_MESSAGE = b"U2: the polled 16550 driver, emitted by both rails\r\n"
#: FCR[2:1], the FIFO resets, which clear themselves when written (PC16550D p.22): no shadow holds them
_FCR_RESETS = 0x06
#: LSR's receiver errors, OE PE FE BI: a read clears them (PC16550D p.21)
_LSR_ERRORS = 0x1E
#: the transcript's 16450-mode runs: the probe's verdicts it forces after the probe -- 0, no FIFO (a 16450
#: or 8250), and 1, the original 16550's broken FIFO -- and the buffer each run sends under a stale cap
_NO_FIFO = (0, 1)
_NO_FIFO_MESSAGE = b"16450 mode: a byte per THRE, none lost\r\n"

_SESSION = None
_BUILT: dict = {}
_TRANSCRIPTS: dict = {}
_COMPILED: dict = {}


# --- the tree, the builds and the runs ----------------------------------------------------------------


def _tree() -> bool:
    """Whether this tree carries the U2 units; an installed one (no runtime/c) has none to build."""
    if all(os.path.isfile(os.path.join(_C, unit)) for unit in _UNITS):
        return True
    from bcir.tests.run_all import _is_source_checkout  # noqa: PLC0415

    assert not _is_source_checkout(), "runtime/c lacks the U2 units"
    return False


def _read(unit: str) -> str:
    with open(os.path.join(_C, unit), encoding="utf-8") as fh:
        return fh.read()


def _session() -> str:
    global _SESSION
    if _SESSION is None:
        _SESSION = tempfile.mkdtemp(prefix="bcir_uart_driver_")
        atexit.register(shutil.rmtree, _SESSION, ignore_errors=True)
    return _SESSION


def _header(placement: str) -> str:
    return um.emit_header(um.PLACEMENTS[placement])


def _scratch(placement: str, tag: str = "") -> str:
    """A directory holding this tree's U2 units beside `placement`'s generated header -- the units
    as they are on disk now, so a fault injected into one is what builds."""
    d = os.path.join(_session(), f"{tag or 'units'}-{placement}")
    if not os.path.isdir(d):
        os.makedirs(d)
        for unit in _UNITS:
            shutil.copyfile(os.path.join(_C, unit), os.path.join(d, unit))
        with open(os.path.join(d, "uart16550_regs.h"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(_header(placement))
    return d


def _compile(cc: str, include: str, source: str, obj: str, *, traced: bool, extra=()) -> None:
    flags = ["-std=c11", "-O2", *([_TRACE] if traced else []), *extra]
    build_units(
        [cc, *flags, "-I", include, "-c", source, "-o", obj],
        f"{os.path.basename(source)} build",
        timeout=_TIMEOUT,
    )


def _program(cc: str, d: str, driver_c: str, name: str, *, traced: bool = True, extra=()) -> str:
    """Link `driver_c` -- compiled beside the generated header in its own directory, with `extra`,
    and traced so that its loads and stores reach the device -- with the harness and the device built
    in `d` for `d`'s placement, never traced; the program's path, in `d`."""
    objects = []
    for unit in ("uart_main.c", "sim16550.c"):
        obj = os.path.join(d, f"{unit[:-2]}.{os.path.basename(cc)}.o")
        if not os.path.isfile(obj):
            _compile(cc, d, os.path.join(d, unit), obj, traced=False)
        objects.append(obj)
    exe = os.path.join(d, name)
    _compile(cc, os.path.dirname(driver_c), driver_c, exe + ".o", traced=traced, extra=extra)
    build_units([cc, exe + ".o", *objects, "-o", exe], f"{name} link", timeout=_TIMEOUT)
    return exe


def _run(exe: str, *args: str, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [exe, *args], input=stdin, capture_output=True, timeout=_TIMEOUT, check=False
    )


def _bcir_cc() -> str:
    from bcir.tests.test_c_cfront import _build_bcir_cc  # noqa: PLC0415

    return _build_bcir_cc(_session())


def _oracle_linkable(d: str) -> str:
    """The oracle's `--linkable` emit of the driver in `d`, through its CLI: a self-contained unit
    (types, the contract literal and the functions, external linkage)."""
    out = os.path.join(d, "driver_oracle.c")
    env = dict(os.environ, PYTHONPATH=_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    done = subprocess.run(
        [sys.executable, "-m", "bcir.frontends.cfront", "--linkable", "-o", out, "uart_driver.c"],
        cwd=d,
        env=env,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=False,
    )
    assert done.returncode == 0, ("the oracle's --linkable emit", done.stderr[-800:])
    return out


def _twin_linkable(d: str) -> str:
    """bcir-cc's `--linkable` emit of the driver in `d`, as it prints it: like the oracle's, the whole
    artifact -- the types, a prototype for every function, the globals and the headers its text
    names (CF-LINKTWIN) -- so it is compiled alone."""
    driver = os.path.join(d, "uart_driver.c")
    done = subprocess.run(
        [_bcir_cc(), "--linkable", driver],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=False,
    )
    assert done.returncode == 0, ("bcir-cc --linkable", done.stderr[-800:])
    out = os.path.join(d, "driver_twin.c")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(done.stdout)
    return out


def _transcript(placement: str, rail: str = "source") -> bytes:
    """What the program prints with the driver from its source, or as `rail` ("oracle", "twin")
    emits it, laid out for `placement` and run against the device configured for it."""
    key = (placement, rail)
    if key not in _TRANSCRIPTS:
        d = _scratch(placement)
        if rail == "source":
            driver = os.path.join(d, "uart_driver.c")
        elif rail == "oracle":
            driver = _oracle_linkable(d)
        else:
            driver = _twin_linkable(d)
        exe = _program(_TRACE_CC, d, driver, f"uart_main_{rail}")
        done = _run(exe)
        assert done.returncode == 0, (placement, rail, done.returncode, done.stderr[-800:])
        _TRANSCRIPTS[key] = done.stdout
    return _TRANSCRIPTS[key]


def _phases(transcript: bytes) -> dict:
    """The transcript's phases, by name: the lines under each `-- <name> (t=N)` header."""
    out: dict = {}
    name = None
    for line in transcript.decode("ascii").splitlines():
        head = re.fullmatch(r"-- (.+) \(t=\d+\)", line)
        if head:
            name = head.group(1)
            out[name] = []
        elif name is not None:
            out[name].append(line)
    return out


_EVENT = re.compile(r"  ([RW]) 0x([0-9a-f]{4}) ([A-Z]{3}) 0x([0-9a-f]{2})")


def _events(lines: list) -> list:
    """The bus accesses a phase shows: (R|W, offset, register, value)."""
    out = []
    for line in lines:
        m = _EVENT.fullmatch(line)
        if m:
            out.append((m.group(1), int(m.group(2), 16), m.group(3), int(m.group(4), 16)))
    return out


#: an access line that may stand for several: the timeout phases collapse unchanging polls (" xN")
_RUN = re.compile(r"  ([RW]) 0x([0-9a-f]{4}) ([A-Z]{3}) 0x([0-9a-f]{2})(?: x([0-9]+))?")


def _trace(lines: list) -> list:
    """The bus accesses a phase shows with their counts: (R|W, offset, register, value, count) -- a
    collapsed line stands for `count` identical reads, any other line for one access."""
    out = []
    for line in lines:
        m = _RUN.fullmatch(line)
        if m:
            count = int(m.group(5)) if m.group(5) else 1
            out.append((m.group(1), int(m.group(2), 16), m.group(3), int(m.group(4), 16), count))
    return out


def _between(lines: list, after: str, before: str) -> list:
    """A phase's lines after the one starting `after` and before the one starting `before`."""
    start = [k for k, line in enumerate(lines) if line.startswith(after)]
    end = [k for k, line in enumerate(lines) if line.startswith(before)]
    assert len(start) == 1 and len(end) == 1 and start[0] < end[0], (after, before, lines[:6])
    return lines[start[0] + 1 : end[0]]


def _spin_budget() -> int:
    """UART_SPIN_BUDGET, read off the driver's API header rather than restated here."""
    m = re.search(r"#define UART_SPIN_BUDGET\s+([0-9]+)u", _read("uart_driver.h"))
    assert m, "uart_driver.h defines no UART_SPIN_BUDGET"
    return int(m.group(1))


def _phase(transcript: bytes, prefix: str) -> list:
    names = [n for n in _phases(transcript) if n == prefix or n.startswith(prefix + ":")]
    assert len(names) == 1, (prefix, names)
    return _phases(transcript)[names[0]]


def _summary(lines: list, head: str) -> str:
    found = [line for line in lines if line.startswith(head)]
    assert len(found) == 1, (head, lines[-5:])
    return found[0]


def _printed(data: bytes) -> str:
    """`data` as the harness's print_bytes() spells it: quoted, \\r \\n \\" \\\\ escaped, printable ASCII
    as is, every other byte \\xNN."""
    out = []
    for b in data:
        c = chr(b)
        if c == "\r":
            out.append("\\r")
        elif c == "\n":
            out.append("\\n")
        elif c in '"\\':
            out.append("\\" + c)
        elif 0x20 <= b < 0x7F:
            out.append(c)
        else:
            out.append(f"\\x{b:02x}")
    return '"' + "".join(out) + '"'


def _off(placement: str, reg: str) -> int:
    return um.reg_offset(um.PLACEMENTS[placement], um.REGS[reg])


# --- the device alone: its script console ------------------------------------------------------------


def _console() -> str:
    """The harness built untraced, for its script console (no driver runs in it)."""
    if "console" not in _BUILT:
        d = _scratch(um.DEFAULT_PLACEMENT, "console")
        _BUILT["console"] = _program(
            _CC, d, os.path.join(d, "uart_driver.c"), "uart_console", traced=False
        )
    return _BUILT["console"]


def _console_run(lines) -> subprocess.CompletedProcess:
    return _run(_console(), "--script", stdin=("\n".join(lines) + "\n").encode("ascii"))


def _sim(*lines: str) -> list:
    done = _console_run(lines)
    assert done.returncode == 0, (done.returncode, done.stdout[-800:], done.stderr[-800:])
    return done.stdout.decode("ascii").splitlines()


def _config(p) -> str:
    knobs = (
        p.stride, p.base_offset, p.reg_width, p.fcr_reset, int(p.fcr_gated), p.fcr_impl_mask,
        p.fifo_depth, *p.rx_triggers, int(p.has_scr), int(p.has_afe), int(p.has_modem),
    )  # fmt: skip
    return f"config {p.name} " + " ".join(str(k) for k in knobs)


def _reads(lines: list) -> list:
    """The values of the console's `r` lines, in order."""
    return [int(line.split()[2], 16) for line in lines if line.startswith("r ")]


def _field(lines: list, kind: str, name: str) -> int:
    """`name=value` from the last `kind` line (status, counters)."""
    line = [ln for ln in lines if ln.startswith(kind + " ")][-1]
    value = dict(item.split("=") for item in line.split()[1:])[name]
    return int(value, 16) if value.startswith("0x") else int(value)


def _w(p, reg: str, value: int) -> str:
    return f"w {um.reg_offset(p, um.REGS[reg]):#x} {value:#x}"


def _r(p, reg: str) -> str:
    return f"r {um.reg_offset(p, um.REGS[reg]):#x}"


def _peek(p, reg: str) -> str:
    return f"peek {um.reg_offset(p, um.REGS[reg]):#x}"


def _8bit(p) -> str:
    """LCR = 03, 8 data bits: LCR resets to 00, 5-bit characters (PC16550D Table 3, p.19), and only the
    word length's data bits are sent and received (p.18), so a script that moves 8-bit data sets it."""
    return _w(p, "LCR", 0x03)


# --- the sim's spec-fidelity tests (blueprint section 8) ---------------------------------------------


def test_sim_reset_states():
    """Reset per section 1.1 (PC16550D Table 3 p.19): IER, LCR, MCR 00, LSR 60 (both transmitter
    registers empty), MSR 00 with no modem input asserted, SCR 00; IIR 01 -- or C1 where the
    placement's FCR reset turns the FIFOs on (Core16550: DS50003798A p.5). The FCR reset is read from
    the placement record, never assumed: pc16550d given Core16550's reset reads C1 too. The scratch
    register holds what is written (section 3.1's probe tells a 16450 from an 8250 by it), and a
    record without one (has_scr False: Lattice's, H-L) reads 0 whatever is written."""
    if not (_CC and _tree()):
        return
    want_iir = {"pc16550d": 0x01, "core16550_apb": 0xC1, "axi16550": 0x01}
    for name, p in um.PLACEMENTS.items():
        regs = ("IER", "IIR", "LCR", "MCR", "LSR", "MSR", "SCR")
        out = _sim(_config(p), *(_r(p, reg) for reg in regs), _w(p, "SCR", 0xA5), _r(p, "SCR"))
        assert out[0] == f"config {name}", out
        want = [0x00, want_iir[name], 0x00, 0x00, 0x60, 0x00, 0x00, 0xA5]
        assert _reads(out) == want, (name, out)
    fifo_on = dataclasses.replace(um.PLACEMENTS["pc16550d"], fcr_reset=0x01)
    assert _reads(_sim(_config(fifo_on), _r(fifo_on, "IIR"))) == [0xC1]
    no_scr = dataclasses.replace(um.PLACEMENTS["pc16550d"], has_scr=False)
    assert _reads(_sim(_config(no_scr), _w(no_scr, "SCR", 0xA5), _r(no_scr, "SCR"))) == [0x00]


def test_sim_trigger_and_timeout():
    """Section 8 and PC16550D 8.4.1 (p.16). With IER[0] set the received-data indication (IIR 04)
    asserts when the FIFO reaches the trigger level -- exactly there, for each of the four levels --
    and clears as it drops below; the character timeout (IIR 0C) comes due after exactly 4 character
    times with a byte held and no arrival and no RBR read, an arrival or one RBR read resets its
    timer, and an empty FIFO has none. In the polled mode (IER 0) neither shows in IIR (p.17) while
    the condition itself holds."""
    if not (_CC and _tree()):
        return
    p = um.PLACEMENTS["pc16550d"]
    iir = _r(p, "IIR")
    for code, level in enumerate(p.rx_triggers):
        lines = [_config(p), _8bit(p), _w(p, "FCR", 0x07 | (code << 6)), _w(p, "IER", 0x01)]
        for k in range(level):
            lines += [f"inject {0x40 + k:#x} 0", iir]
        values = _reads(_sim(*lines, _r(p, "RBR"), iir))
        # IIR after each arrival, the first byte read back, IIR once one byte has gone
        assert values == [0xC1] * (level - 1) + [0xC4, 0x40, 0xC1], (level, values)
    timeout = [
        _config(p), _8bit(p), _w(p, "FCR", 0x87), _w(p, "IER", 0x01),
        *(f"inject {0x61 + k:#x} 0" for k in range(7)),
        "tick 3", iir, "tick 1", iir,               # due after exactly 4 idle character times
        _r(p, "RBR"), iir, "tick 3", iir, "tick 1", iir,   # one read clears it and resets the timer
        "inject 0x7a 0", "tick 3", iir, "tick 1", iir,     # an arrival resets the timer too
    ]  # fmt: skip
    values = _reads(_sim(*timeout))
    assert values == [0xC1, 0xCC, 0x61, 0xC1, 0xC1, 0xCC, 0xC1, 0xCC], values
    one = [
        _config(p), _8bit(p), _w(p, "FCR", 0x87), _w(p, "IER", 0x01), "inject 0x31 0",
        "tick 3", iir, "tick 1", iir, _r(p, "RBR"), "tick 10", iir,
    ]  # fmt: skip
    assert _reads(_sim(*one)) == [0xC1, 0xCC, 0x31, 0xC1]
    polled = _sim(
        _config(p), _8bit(p), _w(p, "FCR", 0x87), "inject 0x31 0", "tick 4", iir, "status"
    )
    assert _reads(polled) == [0xC1] and _field(polled, "status", "timeout") == 1, polled


def test_sim_overrun_discards_shift_byte_not_fifo():
    """Section 1.2's overrun rule (PC16550D p.21; ST16C550 p.18): with the FIFO full, a character
    completing in the shift register sets OE at once and is lost -- the 16 bytes held are intact and
    come out in order -- and an LSR read clears OE. In 16450 mode the holding register is the one
    byte, and the next character destroys it (p.21)."""
    if not (_CC and _tree()):
        return
    p = um.PLACEMENTS["pc16550d"]
    lines = [_config(p), _8bit(p), _w(p, "FCR", 0x07)]
    lines += [f"inject {0x40 + k:#x} 0" for k in range(16)]
    lines += ["inject 0x99 0", _r(p, "LSR"), _r(p, "LSR")]
    lines += [_r(p, "RBR") for _ in range(16)] + [_r(p, "LSR"), "counters"]
    out = _sim(*lines)
    values = _reads(out)
    assert values[:2] == [0x63, 0x61], values[:2]  # DR|OE|THRE|TEMT, then OE cleared by the read
    assert values[2:18] == list(range(0x40, 0x50)), values[2:18]  # the FIFO intact, the 17th lost
    assert values[18] == 0x60 and _field(out, "counters", "rx_overruns") == 1, out[-2:]
    legacy = _sim(
        _config(p), _8bit(p), "inject 0x41 0", "inject 0x42 0", _r(p, "LSR"), _r(p, "RBR")
    )
    assert _reads(legacy) == [0x63, 0x42], legacy


def test_sim_loopback_mirrors_mcr_to_msr():
    """Section 1.7 (PC16550D pp.23-24). With MCR[4] set, MSR[7:4] mirror MCR[3:0] -- DTR on DSR, RTS
    on CTS, OUT1 on RI, OUT2 on DCD -- each one alone, with the deltas each change raises (TERI when
    RI goes off); the external inputs are disconnected until the loop is left. A byte written to THR
    comes back through the receiver one character time later, nothing goes out on SOUT, and a
    character arriving on SIN meanwhile is lost."""
    if not (_CC and _tree()):
        return
    p = um.PLACEMENTS["pc16550d"]
    msr = _r(p, "MSR")
    mirror = [_config(p)]
    for mcr in (0x10, 0x11, 0x12, 0x14, 0x18, 0x10):
        mirror += [_w(p, "MCR", mcr), msr]
    assert _reads(_sim(*mirror)) == [0x00, 0x22, 0x13, 0x41, 0x8C, 0x08]
    # the external inputs all on: DCTS, DDSR and DDCD (RI coming on is no TERI); loopback with MCR[3:0]
    # clear takes them all off, TERI with them; leaving it brings the inputs back
    lines = _sim(_config(p), "modem 0xf0", msr, _w(p, "MCR", 0x10), msr, _w(p, "MCR", 0x00), msr)
    assert _reads(lines) == [0xFB, 0x0F, 0xFB], lines
    data = _sim(
        _config(p), _8bit(p), _w(p, "FCR", 0x07), _w(p, "MCR", 0x10), _w(p, "THR", 0x5A),
        _peek(p, "LSR"), "tick 1", _r(p, "LSR"), _r(p, "RBR"),
        "inject 0x33 0", _r(p, "LSR"), "line", "counters",
    )  # fmt: skip
    assert _reads(data) == [0x61, 0x5A, 0x60], data
    peek = f"peek {_off('pc16550d', 'LSR'):#x} 0x20"  # THRE without TEMT: the byte is shifting
    assert [ln for ln in data if ln.startswith("peek ")] == [peek], data
    assert "line 0" in data and _field(data, "counters", "rx_dropped") == 1, data


def test_sim_dlab_mux_and_fcr_gating():
    """Section 1.1's mux on every placement's layout: with DLAB set, slots 0 and 1 are DLL and DLM --
    a write there reaches neither THR nor IER -- and with it clear they are RBR/THR and IER; slot 2
    reads IIR at either DLAB. An offset that is no register lane (below the AXI base, between lanes)
    is a bus error. FCR gating (PC16550D p.22, every placement): in 16450 mode a write with FCR[0]
    clear programs neither the RX reset nor the trigger, while an ungated record's does; FCR[5:4] are
    not implemented and the resets clear themselves."""
    if not (_CC and _tree()):
        return
    for name, p in um.PLACEMENTS.items():
        slot0, slot1 = um.reg_offset(p, um.REGS["RBR"]), um.reg_offset(p, um.REGS["IER"])
        lines = [
            _config(p), _w(p, "LCR", 0x80), f"w {slot0:#x} 0x34", f"w {slot1:#x} 0x12",
            f"r {slot0:#x}", f"r {slot1:#x}", _peek(p, "LSR"), _r(p, "IIR"),
            _w(p, "LCR", 0x03), f"r {slot1:#x}", f"w {slot1:#x} 0x05", f"r {slot1:#x}",
            _w(p, "LCR", 0x83), f"r {slot1:#x}", _r(p, "IIR"), _w(p, "LCR", 0x03),
            f"w {slot0:#x} 0x41", _peek(p, "LSR"),
            f"r {p.base_offset + 1:#x}" if p.stride > 1 else f"r {p.base_offset + 8:#x}",
            "counters",
        ]  # fmt: skip
        out = _sim(*lines)
        iir = 0xC1 if p.fcr_reset & 1 else 0x01
        assert _reads(out) == [0x34, 0x12, iir, 0x00, 0x05, 0x12, iir, 0x00], (name, out)
        peeks = [int(ln.split()[2], 16) for ln in out if ln.startswith("peek ")]
        # the DLL write was no THR write (LSR 60); the THR write was one (the byte shifting: 20)
        assert peeks == [0x60, 0x20], (name, peeks)
        assert _field(out, "counters", "bus_errors") == 1, (name, out)
        if p.base_offset:
            below = _sim(_config(p), "r 0x0", "counters")
            assert _field(below, "counters", "bus_errors") == 1, (name, below)
        for record, programmed in ((p, False), (dataclasses.replace(p, fcr_gated=False), True)):
            gate = _sim(
                _config(record), _w(record, "FCR", 0x00), "inject 0x41 0",
                _w(record, "FCR", 0xC2), _peek(record, "LSR"), "status",
            )  # fmt: skip
            lsr = int([ln for ln in gate if ln.startswith("peek ")][0].split()[2], 16)
            assert ((lsr & 0x01) == 0) is programmed, (name, programmed, gate)
            assert (_field(gate, "status", "fcr") == 0xC0) is programmed, (name, programmed, gate)
        impl = _sim(_config(p), _w(p, "FCR", 0xFF), "status")
        assert _field(impl, "status", "fcr") == 0xC9, (name, impl)


def test_sim_thre_semantics():
    """Section 8 and PC16550D pp.16, 21-22 (TL16C550D p.41). A byte written to an idle transmitter
    moves into the shift register at once; 16 written fill it and 15 wait in the FIFO, so THRE and
    TEMT clear. THRE sets when the FIFO empties -- the last byte in the shift register -- and TEMT one
    character later; the line carries the bytes in order. The THRE interrupt (IER[1]) is pending after
    the FIFO mode is entered, a read of IIR that reports it clears it, a THR write clears it, the FIFO
    emptying raises it again. A byte written to a full transmitter is lost (16 in the FIFO with the
    shift register busy; one in the holding register in 16450 mode)."""
    if not (_CC and _tree()):
        return
    p = um.PLACEMENTS["pc16550d"]
    iir, lsr = _r(p, "IIR"), _peek(p, "LSR")
    lines = [_config(p), _8bit(p), _w(p, "FCR", 0x07), _w(p, "IER", 0x02), iir, iir]
    lines += [_w(p, "THR", 0x30 + k) for k in range(16)]
    lines += [lsr, iir, "tick 14", lsr, "tick 1", lsr, iir, "tick 1", lsr, "line", "counters"]
    out = _sim(*lines)
    assert _reads(out) == [0xC2, 0xC1, 0xC1, 0xC2], out
    peeks = [int(ln.split()[2], 16) for ln in out if ln.startswith("peek ")]
    assert peeks == [0x00, 0x00, 0x20, 0x60], peeks  # TEMT one character after THRE
    assert [ln for ln in out if ln.startswith("line ")] == [
        "line 16 " + " ".join(f"{0x30 + k:02x}" for k in range(16))
    ]
    assert _field(out, "counters", "tx_overflows") == 0
    full = [_config(p), _w(p, "FCR", 0x07)] + [_w(p, "THR", 0x41) for _ in range(18)] + ["counters"]
    assert _field(_sim(*full), "counters", "tx_overflows") == 1  # 1 shifting + 16 held + 1 lost
    legacy = [_config(p)] + [_w(p, "THR", 0x41) for _ in range(3)] + ["counters"]
    assert _field(_sim(*legacy), "counters", "tx_overflows") == 1  # 1 shifting + 1 held + 1 lost


def test_sim_lsr_errors_ride_the_fifo_and_clear_on_read():
    """PE, FE and BI ride the FIFO with their byte and show in LSR when it reaches the head (PC16550D
    p.21); an LSR read clears them (pp.20-21) -- a second read of the same head shows none; a break
    loads one zero character whatever was on the line; LSR[7] is set while an error byte is in the
    FIFO and clears at the LSR read after the last one has left (p.22); in 16450 mode it is 0."""
    if not (_CC and _tree()):
        return
    p = um.PLACEMENTS["pc16550d"]
    lsr, rbr = _r(p, "LSR"), _r(p, "RBR")
    out = _sim(
        _config(p), _8bit(p), _w(p, "FCR", 0x07),
        "inject 0x61 0", "inject 0x62 0x04", "inject 0x63 0x08", "inject 0x64 0x10",
        lsr, rbr, lsr, lsr, rbr, lsr, rbr, lsr, rbr, lsr, lsr,
    )  # fmt: skip
    assert _reads(out) == [0xE1, 0x61, 0xE5, 0xE1, 0x62, 0xE9, 0x63, 0xF1, 0x00, 0xE0, 0x60], out
    legacy = _sim(_config(p), "inject 0x50 0x04", lsr)
    assert _reads(legacy) == [0x65], legacy


def test_sim_sends_and_receives_only_the_word_lengths_data_bits():
    """LCR[1:0] set the number of bits in each transmitted or received character, 5 to 8 (PC16550D p.18),
    bit 0 first (p.17), and LCR resets to 00, 5 bits (Table 3, p.19). At each word length a byte written
    to THR goes out as its low bits -- on SOUT, and in loopback into the receiver -- and a character
    arriving on SIN is that many bits; RBR's bits above them read 0 here (the PC16550D does not state
    them, and the driver compares only the data bits). The review of U2 found the sim forwarding all
    eight bits at every word length, which hid a self-test that failed at 5, 6 and 7."""
    if not (_CC and _tree()):
        return
    p = um.PLACEMENTS["pc16550d"]
    assert _reads(_sim(_config(p), _r(p, "LCR"))) == [0x00]
    for wls in range(4):
        mask = 0xFF >> (3 - wls)
        out = _sim(
            _config(p), _w(p, "LCR", wls), _w(p, "THR", 0xFF), "tick 1", "line",
            _w(p, "MCR", 0x10), _w(p, "THR", 0xAA), "tick 1", _r(p, "RBR"),
            _w(p, "MCR", 0x00), "inject 0xff 0", _r(p, "RBR"),
        )  # fmt: skip
        assert [ln for ln in out if ln.startswith("line ")] == [f"line 1 {mask:02x}"], (wls, out)
        assert _reads(out) == [0xAA & mask, 0xFF & mask], (wls, out)


def test_sim_refuses_a_config_it_cannot_model():
    """The console and the device fail closed (laws.md L1): each configuration the sim cannot model
    is refused by name, a command before any configuration and a malformed line end the script with
    status 2 and the line, and the configuration after a refusal works."""
    if not (_CC and _tree()):
        return
    pc = um.PLACEMENTS["pc16550d"]
    for change, why in (
        ({"stride": 3}, "the stride is not 1, 2, 4 or 8"),
        ({"reg_width": 4}, "the register lane does not fit in the stride"),
        ({"stride": 4, "reg_width": 4, "base_offset": 0x1002}, "the base offset is not a multiple of the stride"),
        ({"fifo_depth": 300}, "the FIFO depth is not 1..256"),
        ({"rx_triggers": (1, 4, 14, 8)}, "the RX triggers are not four increasing byte counts within the FIFO"),
        ({"rx_triggers": (1, 4, 8, 32)}, "the RX triggers are not four increasing byte counts within the FIFO"),
        ({"fcr_impl_mask": 0xC6}, "FCR[0] is not implemented (a part with its FIFOs always on is a U9 profile)"),
        ({"fcr_reset": 0x10}, "the FCR reset sets a bit the part does not implement"),
    ):  # fmt: skip
        record = dataclasses.replace(pc, **change)
        out = _sim(_config(record), _config(pc), _r(pc, "LSR"))
        assert out == [f"config refused: {why}", "config pc16550d", "r 0x5 0x60"], (change, out)
    for lines, line in (
        (["r 0x5"], "error: line 1: no device: config first"),
        ([_config(pc), "r"], "error: line 2: unknown command or bad operands"),
        ([_config(pc), "w 0x5 -1"], "error: line 2: unknown command or bad operands"),
        (["config pc16550d 1 0"], "error: line 1: config takes a name and 14 numbers"),
    ):
        done = _console_run(lines)
        assert done.returncode == 2, (lines, done.returncode)
        assert done.stdout.decode("ascii").splitlines()[-1] == line, (lines, done.stdout)


# --- the driver ----------------------------------------------------------------------------------------


def _compile_driver(placement: str, source: str | None = None):
    """The oracle's compile of `source` laid out for `placement`, under the driver's API header as it
    is on disk; of the driver on disk when `source` is None, compiled once per placement per process
    (a fault sweep runs each fault in its own)."""
    if source is not None:
        includes = {"uart16550_regs.h": _header(placement), "uart_driver.h": _read("uart_driver.h")}
        return compile_unit(source, includes=includes, check_clang=False)
    if placement not in _COMPILED:
        _COMPILED[placement] = _compile_driver(placement, _read("uart_driver.c"))
    return _COMPILED[placement]


def _accesses(res, fn: str, placement: str) -> list:
    """`fn`'s register accesses in claim order, (R|W, register), named as the U1 laws name them --
    by offset, direction and DLAB, the bracket opened and closed by the LCR stores' known bit 7."""
    lf = res.lowered.functions[fn]
    facts = ul._Facts(lf)  # the laws' own reading of offsets and values (one predicate)
    p = um.PLACEMENTS[placement]
    dlab, out = 0, []
    for c in lf.claims:
        if c.op not in ("c.load", "c.store") or c.domain != Domain.MMIO:
            continue
        write = c.op == "c.store"
        reg = ul.register_at(p, facts.offsets.get(c.id), write, dlab)
        if write and reg == "LCR":
            dlab = 1 if facts.known(c.rd[-1])[0] & 0x80 else 0
        out.append(("W" if write else "R", reg))
    return out


_POLL = """    /* (1) reconfigure only at idle: the line may be mid-character. Each poll's receiver errors are saved:
     * the read clears them (PC16550D p.21), and a wait that gives up leaves the receiver as it was */
    lsr = u[UART_LANE(UART_LSR_OFF)];
    st->lsr_saved = st->lsr_saved | (lsr & UART_LSR_ERRORS);
    while ((lsr & UART_LSR_TEMT_MASK) == 0u) {
        spins = spins + 1u;
        if (spins == UART_SPIN_BUDGET) {
            return UART_ERR_BUSY;
        }
        lsr = u[UART_LANE(UART_LSR_OFF)];
        st->lsr_saved = st->lsr_saved | (lsr & UART_LSR_ERRORS);
    }
"""


def test_driver_compiles_clean_on_both_rails_with_contracts():
    """The driver through the oracle on every placement: clean, law-clean (zero uart_diagnostics),
    every function with MMIO-domain claims and its call-ABI contract, the RegMap contract attested;
    no call and no global. The laws are not vacuous over it: without section 1.4's TEMT wait the
    same unit draws law 4. With a C compiler: warning-clean freestanding C under -Wall -Wextra -Wpedantic
    -Werror; bcir-cc compiles it to the oracle's parity summary line; both rails' --linkable emits
    exit 0."""
    if not _tree():
        return
    source = _read("uart_driver.c")
    for name, placement in um.PLACEMENTS.items():
        res = _compile_driver(name)
        assert res.is_clean, (name, res.diagnostics)
        assert res.uart_diagnostics == [], (name, res.uart_diagnostics)
        assert res.regmap_contract == um.regmap_contract_for(placement), name
        assert set(res.lowered.functions) == _FUNCTIONS, sorted(res.lowered.functions)
        assert set(res.abi_contracts) == _FUNCTIONS and res.abi_diagnostics == [], name
        assert res.lowering_diagnostics == [], (name, res.lowering_diagnostics)
        for fn, lf in res.lowered.functions.items():
            assert any(c.domain == Domain.MMIO for c in lf.claims), (name, fn)
            assert not any(c.op.startswith("c.call") for c in lf.claims), (name, fn)
            assert not lf.globals_used, (name, fn, lf.globals_used)
    assert source.count(_POLL) == 1
    faulty = _compile_driver("axi16550", source.replace(_POLL, ""))
    assert [d.message for d in faulty.uart_diagnostics] == [
        "uart_init: R3-uart: line reconfiguration without a dominating LSR (TEMT) check"
    ]
    if not _CC:
        return
    from bcir.tests.test_c_cfront import _summary_line  # noqa: PLC0415

    twin = _bcir_cc()
    for name in um.PLACEMENTS:
        d = _scratch(name)
        driver = os.path.join(d, "uart_driver.c")
        host = subprocess.run(
            [_CC, "-std=c11", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-ffreestanding", "-c",
             driver, "-o", os.path.join(d, "freestanding.o")],
            capture_output=True, text=True, timeout=_TIMEOUT, check=False,
        )  # fmt: skip
        assert host.returncode == 0, (name, host.stderr)
        done = subprocess.run(
            [twin, driver], capture_output=True, text=True, timeout=_TIMEOUT, check=False
        )
        assert done.returncode == 0, (name, done.stderr)
        line = _summary_line(_compile_driver(name))
        assert done.stdout.splitlines()[0] == f"{driver}: {line}", (name, done.stdout[:300])
        assert os.path.getsize(_oracle_linkable(d)) > 0 and os.path.getsize(_twin_linkable(d)) > 0


#: the API functions each rail's --linkable artifact declares with another type than the header's: the
#: emit's parameter types keep no qualifier below a parameter's top level (the open defect
#: bcir/tests/test_cfront_link.py pins as `_LINKABLE_OPEN`), so uart_tx's `const uint8_t *buf` is
#: declared and defined `uint8_t *buf`, a type C does not call compatible with the header's (C11
#: 6.7.6.3p15, 6.7.6.1p2), and a program whose units disagree on it is undefined (6.2.7p2). The
#: harness calls through the header; every ABI the program builds on passes the two alike.
_LINKABLE_API_OPEN = {"uart_tx"}
_PROTOTYPE = re.compile(r"^uint32_t (uart_\w+)\(.*\);$", re.MULTILINE)
_CONFLICT = re.compile(r"conflicting types for ['\u2018](\w+)['\u2019]")


def _api_conflicts(d: str, artifact: str) -> set:
    """The functions whose prototype in `artifact` conflicts with the driver's API header, as the C
    compiler finds them in one translation unit: the header, then the artifact's prototypes."""
    protos = _PROTOTYPE.findall(artifact)
    assert set(protos) == _FUNCTIONS, protos
    unit = os.path.join(d, "api_check.c")
    with open(unit, "w", encoding="utf-8", newline="\n") as fh:
        fh.write('#include "uart_driver.h"\n')
        fh.writelines(m.group(0) + "\n" for m in _PROTOTYPE.finditer(artifact))
    done = subprocess.run(
        [_CC, "-std=c11", "-fsyntax-only", "-I", d, unit],
        capture_output=True, text=True, timeout=_TIMEOUT, check=False,
        env={**os.environ, "LC_ALL": "C"},
    )  # fmt: skip
    found = set(_CONFLICT.findall(done.stderr))
    assert (done.returncode == 0) == (not found), (done.returncode, done.stderr[-600:])
    return found


def test_the_emitted_driver_declares_the_headers_api_but_the_open_defect():
    """Each rail's --linkable driver against the driver's API header, in one translation unit: the
    compiler names every API function whose emitted declaration is not the header's. On both rails
    that is exactly `_LINKABLE_API_OPEN`, uart_tx's TX buffer defined without its `const`. The check
    reads the declarations: the same artifact with the `const` restored conflicts nowhere, and with
    uart_flush's count taken as an `int32_t *` it conflicts at uart_flush too."""
    if not (_CC and _tree()):
        return
    d = _scratch(_GOLDEN_PLACEMENT)
    for rail, path in (("oracle", _oracle_linkable(d)), ("twin", _twin_linkable(d))):
        with open(path, encoding="utf-8") as fh:
            artifact = fh.read()
        assert _api_conflicts(d, artifact) == _LINKABLE_API_OPEN, rail
        tx = [m.group(0) for m in _PROTOTYPE.finditer(artifact) if m.group(1) == "uart_tx"]
        assert len(tx) == 1 and tx[0].count("uint8_t *") == 1, (rail, tx)
        fixed = artifact.replace(tx[0], tx[0].replace("uint8_t *", "const uint8_t *"))
        assert _api_conflicts(d, fixed) == set(), rail
        flush = [m.group(0) for m in _PROTOTYPE.finditer(artifact) if m.group(1) == "uart_flush"]
        count = re.sub(r", uint32_t \*(\s*\w*)\);$", r", int32_t *\1);", flush[0])
        assert len(flush) == 1 and count != flush[0], (rail, flush)
        wrong = artifact.replace(flush[0], count)
        assert _api_conflicts(d, wrong) == _LINKABLE_API_OPEN | {"uart_flush"}, rail


def test_probe_identifies_the_sim_as_16550a():
    """Section 3.1 on every placement: the probe writes FCR = 0xC1 and only then reads IIR (so a part
    whose FIFOs are off at reset -- pc16550d, axi16550 -- answers 11 because the probe turned them
    on), then writes FCR = 0, and writes and reads back SCR twice; the device answers as a 16550A with
    a working scratch register, and the state records it with the shadow of that last FCR write, 0."""
    if not (_TRACE_CC and _tree()):
        return
    for name in um.PLACEMENTS:
        res = _compile_driver(name)
        assert _accesses(res, "uart_probe", name) == [
            ("W", "FCR"), ("R", "IIR"), ("W", "FCR"),
            ("W", "SCR"), ("R", "SCR"), ("W", "SCR"), ("R", "SCR"),
        ]  # fmt: skip
        lines = _phase(_transcript(name), "probe")
        fcr, iir, scr = _off(name, "FCR"), _off(name, "IIR"), _off(name, "SCR")
        assert _events(lines) == [
            ("W", fcr, "FCR", 0xC1), ("R", iir, "IIR", 0xC1), ("W", fcr, "FCR", 0x00),
            ("W", scr, "SCR", 0xA5), ("R", scr, "SCR", 0xA5),
            ("W", scr, "SCR", 0x5A), ("R", scr, "SCR", 0x5A),
        ], (name, lines)  # fmt: skip
        assert _summary(lines, "probe:") == (
            "probe: returns 2 (16550A); fifo_ok 2, scr_ok 1, fcr_shadow 0x00"
        )


def test_the_probe_leaves_the_fifos_off():
    """The probe's end state, whatever its verdict: once IIR[7:6] are read it writes FCR = 0 and
    shadows 0, so a broken FIFO is not left on and init programs FCR from a known state -- as Linux
    leaves a port after autoconfig() (drivers/tty/serial/8250/8250_port.c: serial8250_clear_fifos()
    ends with FCR = 0 on a part with UART_CAP_FIFO; for one without, serial8250_set_fcr() writes the
    uart_config[] FCR, 0 for PORT_16450 and PORT_16550). After the probe alone the device has its FIFOs
    off (IIR[7:6] = 00) on every placement -- the Core16550's, on at reset, included -- and a uart_tx
    before any init, under a plan's cap, bursts one byte per THRE: the probe's shadow says no FIFO
    mode, so the device, whose holding register is one byte, loses none. No init has set the word
    length, so the line carries each byte at LCR's reset one, 5 data bits (Table 3, p.19): its low five
    bits (p.18)."""
    if not (_TRACE_CC and _tree()):
        return
    early = b"pre-init\r\n"
    line = f"{_printed(bytes(b & 0x1F for b in early))}, the buffer at 5 data bits"
    for name, p in um.PLACEMENTS.items():
        transcript = _transcript(name)
        probe = _phase(transcript, "probe")
        assert _summary(probe, "  device:") == "  device: fifos off, iir 0x01", (name, probe)
        assert _summary(probe, "probe:").endswith("fcr_shadow 0x00"), (name, probe)
        tx = _phase(transcript, "tx before init")
        events = _events(tx)
        runs = _runs(events)
        assert [count for _lsr, count in runs] == [1] * len(early), (name, runs)
        assert all(lsr & 0x20 for lsr, _count in runs), (name, runs)
        assert bytes(value for _k, _o, reg, value in events if reg == "THR") == early, (name, tx)
        want = f"tx: returns 0 (ok); {len(early)} of {len(early)} bytes sent under a plan burst cap"
        assert _summary(tx, "tx:") == f"{want} of {p.fifo_depth}; tx overflows 0", (name, tx)
        flush = _summary(tx, "flush:")
        assert flush.endswith(f"the line carried {len(early)} bytes {line}"), (name, flush)


def test_init_follows_the_canonical_sequence():
    """Section 1.4, statically and on the bus, on every placement. The claim order is LSR read (the
    first poll, then the loop's next) < LCR (DLAB=1) < DLL < DLM < LCR (DLAB=0) < FCR < MCR < IER < one read each of LSR, RBR, IIR, MSR.
    On the bus, with PG143's vector -- 56000 baud by its truncating formula from the placement's
    reference (100 MHz: 0x6F, PG143 p.52; 1.8432 MHz: 2, p.53), 8E2 = 0x1F -- the values are LCR 0x9F,
    DLL then DLM low byte first, LCR 0x1F, FCR 0x87 (enable, both resets, trigger 8 in one write), MCR
    DTR|RTS, IER 0, and the state records the shadow 0x81 -- the write less the resets, which clear
    themselves (H-E) -- a trigger of 8 bytes and a burst of the FIFO's 16. Each argument init refuses --
    divisor 0 or past 16 bits, a format with DLAB or with set break, a trigger code that is none -- is
    refused before the device is touched."""
    if not (_TRACE_CC and _tree()):
        return
    for name, p in um.PLACEMENTS.items():
        res = _compile_driver(name)
        assert _accesses(res, "uart_init", name) == [
            ("R", "LSR"), ("R", "LSR"), ("W", "LCR"), ("W", "DLL"), ("W", "DLM"), ("W", "LCR"), ("W", "FCR"),
            ("W", "MCR"), ("W", "IER"), ("R", "LSR"), ("R", "RBR"), ("R", "IIR"), ("R", "MSR"),
        ], name  # fmt: skip
        divisor = p.ref_clk_hz // (16 * _BAUD)
        off = {reg: _off(name, reg) for reg in um.REGS}
        lines = _phase(_transcript(name), "init")
        assert _events(lines) == [
            ("R", off["LSR"], "LSR", 0x60), ("W", off["LCR"], "LCR", 0x80 | _LCR_8E2),
            ("W", off["DLL"], "DLL", divisor & 0xFF), ("W", off["DLM"], "DLM", divisor >> 8),
            ("W", off["LCR"], "LCR", _LCR_8E2), ("W", off["FCR"], "FCR", 0x87),
            ("W", off["MCR"], "MCR", 0x03), ("W", off["IER"], "IER", 0x00),
            ("R", off["LSR"], "LSR", 0x60), ("R", off["RBR"], "RBR", 0x00),
            ("R", off["IIR"], "IIR", 0xC1), ("R", off["MSR"], "MSR", 0x00),
        ], (name, lines)  # fmt: skip
        init = _summary(lines, "init:")
        assert init.startswith(
            f"init: returns 0 (ok); divisor {divisor} for 56000 baud from {p.ref_clk_hz} Hz"
        ), lines
        assert init.endswith("trigger 8 bytes, fcr_shadow 0x81, tx_burst 16"), lines
        assert _summary(lines, "  device:") == (
            f"  device: divisor latch {divisor:#06x}, lcr 0x1f, mcr 0x03, ier 0x00, fifos on, "
            "trigger 8 bytes"
        ), lines
        refusals = _phase(_transcript(name), "init refusals")
        assert _events(refusals) == [], refusals
        assert [ln for ln in refusals if ln.startswith("init(")] == [
            "init(divisor 0): returns 2, the device untouched",
            "init(divisor 0x10000): returns 2, the device untouched",
            "init(format 0x83, with DLAB): returns 3, the device untouched",
            "init(format 0x43, with set break): returns 3, the device untouched",
            "init(trigger code 0x81): returns 4, the device untouched",
        ], refusals
    assert um.PLACEMENTS["axi16550"].ref_clk_hz // (16 * _BAUD) == 0x6F  # PG143 p.52


def test_init_refuses_a_format_with_break_or_dlab():
    """lcr_format is the line format, LCR[5:0] (UART_FORMAT_BITS). LCR[6] is Break Control, which holds
    SOUT at spacing until it is cleared, and LCR[7] is DLAB (PC16550D Table 1, pp.17-18): an init that
    took either as part of the format would leave the line in break, or leave the divisor latches where
    RBR, THR and IER should be. So init refuses 0x43 (8N1 with set break) and 0x83 (8N1 with DLAB) with
    UART_ERR_FORMAT, and the transcript shows neither call reaching the device."""
    if not (_TRACE_CC and _tree()):
        return
    for name in um.PLACEMENTS:
        refusals = _phase(_transcript(name), "init refusals")
        assert _events(refusals) == [], (name, refusals)
        formats = [ln for ln in refusals if ln.startswith("init(format ")]
        assert formats == [
            "init(format 0x83, with DLAB): returns 3, the device untouched",
            "init(format 0x43, with set break): returns 3, the device untouched",
        ], (name, refusals)


def _runs(events: list) -> list:
    """The TX phase's bursts: (the LSR value read just before, the THR writes that followed)."""
    runs: list = []
    for kind, _offset, reg, value in events:
        if kind == "R" and reg == "LSR":
            runs.append([value, 0])
        elif kind == "W" and reg == "THR":
            assert runs, "a THR write before any LSR read"
            runs[-1][1] += 1
    return [(lsr, n) for lsr, n in runs if n]


def test_tx_bursts_never_exceed_fifo_and_never_write_without_thre():
    """The TX law and the review's FIFO-headroom bound in its polled case (blueprint section 12), in FIFO
    mode (the 16450-mode half is test_tx_without_a_working_fifo_bursts_one_byte_per_thre): every
    burst of THR writes follows, directly, an LSR read with THRE set -- the TX FIFO just seen empty, so
    the free FIFO is UART_CAP_FIFO_DEPTH -- and is no longer than that free FIFO; the plan's cap of 64
    is clamped, so the 52-byte buffer goes as 16, 16, 16, 4. The device loses nothing (no TX overflow)
    and the line carries the buffer, in order. Statically: one LSR read and the THR store, nothing
    else."""
    if not (_TRACE_CC and _tree()):
        return
    for name, p in um.PLACEMENTS.items():
        assert _accesses(_compile_driver(name), "uart_tx", name) == [("R", "LSR"), ("W", "THR")]
        transcript = _transcript(name)
        tx = _phase(transcript, "tx")
        runs = _runs(_events(tx))
        assert [n for _lsr, n in runs] == [16, 16, 16, 4], (name, runs)
        assert all(lsr & 0x20 and n <= p.fifo_depth for lsr, n in runs), (name, runs)
        assert _summary(tx, "tx:") == (
            f"tx: returns 0 (ok); {len(_MESSAGE)} of {len(_MESSAGE)} bytes sent under a plan "
            "burst cap of 64; tx overflows 0"
        )
        flush = _summary(_phase(transcript, "flush"), "flush:")
        line = '"U2: the polled 16550 driver, emitted by both rails\\r\\n", the buffer'
        assert flush.endswith(line), flush


def test_tx_refuses_a_state_without_a_burst_cap():
    """A state no init has filled has no TX burst cap (tx_burst 0), and uart_tx refuses it with
    UART_ERR_NOT_READY before the device is touched, reporting 0 bytes sent, where it would otherwise
    read the 0 as some cap and burst. The transcript's zeroed state shows it on every placement: the
    call returns 8 and makes no register access."""
    if not (_TRACE_CC and _tree()):
        return
    for name in um.PLACEMENTS:
        lines = _phase(_transcript(name), "tx refused")
        assert _events(lines) == [], (name, lines)
        want = "tx(a zeroed state): returns 8 (not ready), 0 bytes sent, the device untouched"
        assert _summary(lines, "tx(") == want, (name, lines)


def test_selftest_refuses_a_state_without_a_word_length():
    """The self-test's patterns are the word length's data bits, which uart_init records (word_bits); a
    state no init has filled has none, and the test refuses it before it touches the device:
    UART_ERR_NOT_READY with no register access, as uart_tx refuses a state with no burst cap."""
    if not (_TRACE_CC and _tree()):
        return
    for name in um.PLACEMENTS:
        lines = _phase(_transcript(name), "selftest refused")
        assert _events(lines) == [], (name, lines)
        assert _summary(lines, "selftest(") == (
            "selftest(a zeroed state): returns 8 (not ready), the device untouched"
        ), (name, lines)


def test_rx_drain_triage_counts_the_injected_errors_exactly():
    """One LSR read per drain iteration feeds both the DR test and the triage (statically: the drain's
    register accesses are one LSR read and one RBR read), and on the bus every RBR read follows exactly
    one LSR read that shows DR. The script's errors are counted exactly -- the parity error once, the
    overrun once, nothing else -- and the bytes are the script's but the one the overrun lost. The
    device saw the gap bring the character timeout due at 4 character times, not 3, and the burst
    reach the trigger level at its 8th byte, not its 7th."""
    if not (_TRACE_CC and _tree()):
        return
    want = (
        ("rx a", '"abc"', (0, 0, 0, 0)),
        ("rx b", '"RX-burst"', (0, 0, 0, 0)),
        ("rx c", '"ABCDE"', (0, 1, 0, 0)),
        ("rx d", '"0123456789abcdef"', (1, 0, 0, 0)),
    )
    for name in um.PLACEMENTS:
        assert _accesses(_compile_driver(name), "uart_rx_drain", name) == [
            ("R", "LSR"), ("R", "RBR"),
        ]  # fmt: skip
        transcript = _transcript(name)
        for phase, data, counts in want:
            lines = _phase(transcript, phase)
            oe, pe, fe, bi = counts
            assert _summary(lines, "drain:") == (
                f"drain: {len(data) - 2} bytes {data}; OE {oe} PE {pe} FE {fe} BI {bi}; "
                "lsr_saved 0x00 after it"
            ), (name, phase, lines)
            events = [(kind, reg, value) for kind, _o, reg, value in _events(lines)]
            for k, (kind, reg, _value) in enumerate(events):
                if reg == "RBR":
                    assert k > 0 and events[k - 1][1] == "LSR" and events[k - 1][2] & 0x01, events
            lsr_reads = sum(1 for e in events if e[1] == "LSR")
            assert lsr_reads == sum(1 for e in events if e[1] == "RBR") + 1, events
        gap = [ln for ln in _phase(transcript, "rx a") if ln.startswith("  device:")]
        assert "timeout due: no" in gap[0] and "timeout due: yes" in gap[1], gap
        burst = [ln for ln in _phase(transcript, "rx b") if ln.startswith("  device:")]
        assert "(8) reached: no" in burst[0] and "(8) reached: yes" in burst[1], burst
        assert _summary(_phase(transcript, "end"), "verdict:").endswith(
            "rx 36 bytes with OE 1 PE 3 FE 2 BI 0"
        )


def test_tx_and_flush_polls_keep_the_receivers_errors_for_the_drain():
    """A read of LSR clears OE, PE, FE and BI, and PE, FE and BI describe the byte at the FIFO head
    (PC16550D p.21), so an LSR read uart_tx or uart_flush makes while a received byte waits takes that
    byte's error from the drain that later takes the byte. The driver keeps what those reads saw in the
    state (lsr_saved) and the drain counts it with its first read, as Linux keeps up->lsr_saved_flags
    (drivers/tty/serial/8250/8250.h: serial_lsr_in). On the bus: with a byte held under a parity error,
    TX's LSR read shows PE (0xE5) and the device's next reads show none -- the read cleared it -- yet the
    state holds PE, and the drain returns the byte with PE counted once and the state cleared; with a
    framing error, the flush's one read clears FE the same way and the drain counts it. Before the
    review's fix the drain returned both bytes with no error counted."""
    if not (_TRACE_CC and _tree()):
        return
    for name in um.PLACEMENTS:
        transcript = _transcript(name)
        off = {reg: _off(name, reg) for reg in um.REGS}
        for phase, byte, err, first, after, saved, counted in (
            ("rx e", 0x42, "PE", 0xE5, [0xA1, 0xE1], 0x04, "OE 0 PE 1 FE 0 BI 0"),
            ("rx f", 0x46, "FE", 0xE9, [], 0x08, "OE 0 PE 0 FE 1 BI 0"),
        ):
            lines = _phase(transcript, phase)
            before = [
                v
                for _k, o, reg, v in _events(_between(lines, "  device:", "  state:"))
                if reg == "LSR"
            ]
            assert before == [first, *after], (name, phase, before)
            assert not any(v & _LSR_ERRORS for v in after), (name, phase, after)
            assert _summary(lines, "  state:") == f"  state: lsr_saved {saved:#04x}", (name, lines)
            assert _events(_between(lines, "  state:", "drain:")) == [
                ("R", off["LSR"], "LSR", 0xE1), ("R", off["RBR"], "RBR", byte),
                ("R", off["LSR"], "LSR", 0xE0),
            ], (name, phase, lines)  # fmt: skip
            assert _summary(lines, "drain:") == (
                f'drain: 1 bytes "{chr(byte)}"; {counted}; lsr_saved 0x00 after it'
            ), (name, phase, err, lines)


def test_selftest_passes_in_loopback():
    """Section 1.7 on the bus, every placement: an LSR read showing TEMT first (the transmitter idle,
    so nothing uart_tx accepted is looped back); MCR saved; the receiver drained of the byte it held
    from before the test; with LOOP set, OUT2|RTS come back as DCD|CTS (MSR 0x9_) and OUT1|DTR as
    RI|DSR (0x6_); each pattern byte goes out after a THRE read and comes back after a DR read; TEMT
    again before the loopback is left; both FIFOs reset from the FCR shadow while still in it -- the
    shadow 0x81 with both resets, FCR = 0x87 -- the shadow staying 0x81, since the resets clear
    themselves (H-E); MCR restored and MSR read once; nothing reached the line, and the test returns
    0."""
    if not (_TRACE_CC and _tree()):
        return
    for name in um.PLACEMENTS:
        off = {reg: _off(name, reg) for reg in um.REGS}
        lines = _phase(_transcript(name), "selftest")
        assert _events(lines) == [
            ("R", off["LSR"], "LSR", 0x61), ("R", off["MCR"], "MCR", 0x03),
            ("R", off["LSR"], "LSR", 0x61), ("R", off["RBR"], "RBR", 0x5A),
            ("R", off["LSR"], "LSR", 0x60),
            ("W", off["MCR"], "MCR", 0x1A),
            ("R", off["MSR"], "MSR", 0x99), ("W", off["MCR"], "MCR", 0x15),
            ("R", off["MSR"], "MSR", 0x6B),
            ("R", off["LSR"], "LSR", 0x60), ("W", off["THR"], "THR", 0x55),
            ("R", off["LSR"], "LSR", 0x61), ("R", off["RBR"], "RBR", 0x55),
            ("R", off["LSR"], "LSR", 0x60), ("W", off["THR"], "THR", 0xAA),
            ("R", off["LSR"], "LSR", 0x61), ("R", off["RBR"], "RBR", 0xAA),
            ("R", off["LSR"], "LSR", 0x60), ("W", off["FCR"], "FCR", 0x87),
            ("W", off["MCR"], "MCR", 0x03), ("R", off["MSR"], "MSR", 0x06),
        ], (name, lines)  # fmt: skip
        assert _summary(lines, "selftest:") == (
            "selftest: returns 0 (ok); mcr 0x03 after it, fcr_shadow 0x81"
        )


def test_selftest_drains_the_receiver_first():
    """A byte the receiver holds from before the self-test would come back in place of a pattern, so
    the test first drains RBR while LSR shows DR, bounded at UART_CAP_FIFO_DEPTH + 1 reads -- a full
    FIFO and the character completing in the shift register behind it -- as Linux bounds its own drains
    (drivers/tty/serial/8250/8250_port.c: size_fifo(), serial8250_rx_chars()). With a stale byte placed
    in the receiver, the test reads it (0x5A) before it enters the loopback, and passes. Under a peer
    that never stops sending -- a character arrives with each character time, so DR never clears -- it
    reads exactly UART_CAP_FIFO_DEPTH + 1 bytes, each after an LSR read showing DR, sees DR once more and
    fails with UART_ERR_RX_STUCK without entering the loopback; it still resets the FIFOs (the receiver
    is empty after it), restores MCR and reads MSR. Each run's first access is the poll that finds the
    transmitter idle (TEMT), then the MCR read the test restores from."""
    if not (_TRACE_CC and _tree()):
        return
    for name, p in um.PLACEMENTS.items():
        transcript = _transcript(name)
        off = {reg: _off(name, reg) for reg in um.REGS}
        main = _phase(transcript, "selftest")
        assert _events(main)[2:5] == [
            ("R", off["LSR"], "LSR", 0x61), ("R", off["RBR"], "RBR", 0x5A),
            ("R", off["LSR"], "LSR", 0x60),
        ], (name, main)  # fmt: skip
        assert _summary(main, "selftest:").startswith("selftest: returns 0 (ok)"), (name, main)
        lines = _phase(transcript, "stuck selftest")
        reads = p.fifo_depth + 1  # UART_CAP_FIFO_DEPTH + 1, the drain's bound
        drain = [("R", off["LSR"], "LSR", 0x61), ("R", off["RBR"], "RBR", 0x7E)] * reads
        assert _events(lines) == [
            ("R", off["LSR"], "LSR", 0x61), ("R", off["MCR"], "MCR", 0x03), *drain,
            ("R", off["LSR"], "LSR", 0x61), ("W", off["FCR"], "FCR", 0x87),
            ("W", off["MCR"], "MCR", 0x03), ("R", off["MSR"], "MSR", 0x00),
        ], (name, lines)  # fmt: skip
        assert _summary(lines, "selftest:") == (
            "selftest: returns 9 (the receiver never emptied); mcr 0x03 after it, fcr_shadow 0x81"
        )
        assert _summary(lines, "  device:") == "  device: the receiver holds 0 after it"


def _no_fifo(transcript: bytes, verdict: int, step: str) -> list:
    """`step`'s phase of the 16450-mode run whose probe verdict the transcript forced to `verdict`."""
    return _phase(transcript, f"fifo_ok {verdict} {step}")


def test_init_without_a_working_fifo_programs_16450_mode():
    """Section 3.1's verdict acted on (blueprint section 3 item 1), on every placement. The sim answers
    the probe as a 16550A, so the transcript forces the verdict after the probe -- once to 0, no FIFO
    (a 16450 or 8250), once to 1, the original 16550's broken FIFO -- and init must enable no FIFO the
    probe did not find working. The probe has left the FIFOs off; init runs section 1.4 with FCR = 0 in
    step 5, which asks for nothing FCR[0] gates (PC16550D p.22) and keeps the part in 16450 mode --
    IIR[7:6] read 00 after it -- and records the shadow 0, a one-byte trigger and a one-byte TX burst:
    what Linux's 8250 core runs PORT_16450 and PORT_16550 with (drivers/tty/serial/8250/8250_port.c,
    uart_config[]: fifo_size 1, tx_loadsz 1, no FCR enable)."""
    if not (_TRACE_CC and _tree()):
        return
    for name, p in um.PLACEMENTS.items():
        transcript = _transcript(name)
        divisor = p.ref_clk_hz // (16 * _BAUD)
        off = {reg: _off(name, reg) for reg in um.REGS}
        for verdict in _NO_FIFO:
            probe = _no_fifo(transcript, verdict, "probe")
            summary = _summary(probe, "probe:")
            assert summary.startswith(f"probe: returns 2; fifo_ok forced to {verdict} ("), summary
            device = _summary(probe, "  device:")
            assert device == "  device: fifos off, iir 0x01", (name, verdict, probe)
            lines = _no_fifo(transcript, verdict, "init")
            # step 8's RBR read returns the byte read last before it (the sim's choice: sim16550.h)
            seen = [(k, o, reg, None if reg == "RBR" else v) for k, o, reg, v in _events(lines)]
            assert seen == [
                ("R", off["LSR"], "LSR", 0x60), ("W", off["LCR"], "LCR", 0x80 | _LCR_8E2),
                ("W", off["DLL"], "DLL", divisor & 0xFF), ("W", off["DLM"], "DLM", divisor >> 8),
                ("W", off["LCR"], "LCR", _LCR_8E2), ("W", off["FCR"], "FCR", 0x00),
                ("W", off["MCR"], "MCR", 0x03), ("W", off["IER"], "IER", 0x00),
                ("R", off["LSR"], "LSR", 0x60), ("R", off["RBR"], "RBR", None),
                ("R", off["IIR"], "IIR", 0x01), ("R", off["MSR"], "MSR", 0x00),
            ], (name, verdict, lines)  # fmt: skip
            want = "init: returns 0 (ok); trigger 1 bytes, fcr_shadow 0x00, tx_burst 1"
            assert _summary(lines, "init:") == want, (name, verdict, lines)
            device = _summary(lines, "  device:")
            assert device == "  device: fifos off, iir 0x01", (name, verdict, lines)


def test_tx_without_a_working_fifo_bursts_one_byte_per_thre():
    """The TX law and the review's headroom bound in 16450 mode (blueprint section 12; U2's notes):
    THRE says the one-byte holding register is empty (PC16550D p.22), so a THRE read shows one byte
    free. Each 16450-mode run hands uart_tx a stale plan cap of UART_CAP_FIFO_DEPTH -- what a plan made
    for the FIFO leaves -- and the driver clamps it to one byte, since its FCR shadow says no FIFO mode:
    the 40-byte buffer goes as 40 one-byte bursts, each directly after an LSR read with THRE set; the
    device loses none (no TX overflow, where a 16-byte burst after THRE loses 14 or 15 of its bytes) and
    the line carries the buffer, in order."""
    if not (_TRACE_CC and _tree()):
        return
    n = len(_NO_FIFO_MESSAGE)
    line = '"16450 mode: a byte per THRE, none lost\\r\\n", the buffer'
    for name, p in um.PLACEMENTS.items():
        transcript = _transcript(name)
        for verdict in _NO_FIFO:
            tx = _no_fifo(transcript, verdict, "tx")
            events = _events(tx)
            runs = _runs(events)
            assert [count for _lsr, count in runs] == [1] * n, (name, verdict, runs)
            assert all(lsr & 0x20 for lsr, _count in runs), (name, verdict, runs)
            sent = bytes(value for _k, _o, reg, value in events if reg == "THR")
            assert sent == _NO_FIFO_MESSAGE, (name, verdict, sent)
            cap = f"a stale plan burst cap of {p.fifo_depth}"
            want = f"tx: returns 0 (ok); {n} of {n} bytes sent under {cap}; tx overflows 0"
            assert _summary(tx, "tx:") == want, (name, verdict, tx)
            flush = _summary(_no_fifo(transcript, verdict, "flush"), "flush:")
            assert flush.endswith(f"the line carried {n} bytes {line}"), (name, verdict, flush)


def test_selftest_resets_the_fifos_only_in_fifo_mode():
    """FCR[2:1] go with FCR[0] in the same write or not at all: with FCR[0] clear a gated part programs
    nothing else (PC16550D p.22; the U1 law refuses such a write where it can prove bit 0 clear), and
    16450 mode has no FIFO to reset -- Linux's serial8250_clear_fifos resets them only on a part with
    UART_CAP_FIFO (drivers/tty/serial/8250/8250_port.c). So the self-test resets the FIFOs only when
    its shadow says FIFO mode: in each 16450-mode run it passes in loopback with no FCR write at all,
    its shadow still 0, and leaves it only after TEMT, with an LSR read for the looped character RBR may
    still hold (none: both were read back); and on every placement no FCR write in the transcript sets
    FCR[2:1] with FCR[0] clear."""
    if not (_TRACE_CC and _tree()):
        return
    for name in um.PLACEMENTS:
        transcript = _transcript(name)
        off = {reg: _off(name, reg) for reg in um.REGS}
        writes = [
            value
            for lines in _phases(transcript).values()
            for _k, _o, reg, value in _events(lines)
            if reg == "FCR"
        ]
        gated = [value for value in writes if value & _FCR_RESETS and not value & 0x01]
        assert writes and not gated, (name, writes)
        for verdict in _NO_FIFO:
            lines = _no_fifo(transcript, verdict, "selftest")
            assert _events(lines) == [
                ("R", off["LSR"], "LSR", 0x60), ("R", off["MCR"], "MCR", 0x03),
                ("R", off["LSR"], "LSR", 0x60), ("W", off["MCR"], "MCR", 0x1A),
                ("R", off["MSR"], "MSR", 0x99), ("W", off["MCR"], "MCR", 0x15),
                ("R", off["MSR"], "MSR", 0x6B),
                ("R", off["LSR"], "LSR", 0x60), ("W", off["THR"], "THR", 0x55),
                ("R", off["LSR"], "LSR", 0x61), ("R", off["RBR"], "RBR", 0x55),
                ("R", off["LSR"], "LSR", 0x60), ("W", off["THR"], "THR", 0xAA),
                ("R", off["LSR"], "LSR", 0x61), ("R", off["RBR"], "RBR", 0xAA),
                ("R", off["LSR"], "LSR", 0x60), ("R", off["LSR"], "LSR", 0x60),
                ("W", off["MCR"], "MCR", 0x03), ("R", off["MSR"], "MSR", 0x06),
            ], (name, verdict, lines)  # fmt: skip
            want = "selftest: returns 0 (ok); mcr 0x03 after it, fcr_shadow 0x00"
            assert _summary(lines, "selftest:") == want, (name, verdict, lines)


def test_tx_gives_up_when_thre_never_comes():
    """uart_tx's wait for THRE is bounded. Against a transmitter that never drains -- the sim's
    sim_stall_tx: its shift register never finishes, so THRE stays off once a byte waits behind it --
    TX after init in FIFO mode writes one burst after a THRE read, the FIFO depth's worth that the FIFO
    and the shift register accept, then polls LSR with THRE clear UART_SPIN_BUDGET times, the bound
    every wait of the driver gives up at, and returns UART_ERR_TIMEOUT, *nsent the bytes accepted.
    Nothing reached the line meanwhile; once the transmitter resumes, those bytes go out, in order."""
    if not (_TRACE_CC and _tree()):
        return
    budget = _spin_budget()
    for name, p in um.PLACEMENTS.items():
        lines = _phase(_transcript(name), "tx timeout")
        off = {reg: _off(name, reg) for reg in um.REGS}
        accepted = _MESSAGE[: p.fifo_depth]
        assert _summary(lines, "init:") == "init: returns 0 (ok); fcr_shadow 0x81, tx_burst 16"
        assert _trace(_between(lines, "init:", "tx:")) == [
            ("R", off["LSR"], "LSR", 0x60, 1),
            *(("W", off["THR"], "THR", byte, 1) for byte in accepted),
            ("R", off["LSR"], "LSR", 0x00, budget),
        ], (name, lines)  # fmt: skip
        assert _summary(lines, "tx:") == (
            f"tx: returns 7 (timed out); {len(accepted)} of {len(_MESSAGE)} bytes sent under "
            f"init's burst cap of {p.fifo_depth}; tx overflows 0"
        )
        assert _summary(lines, "  device:") == (
            f"  device: the line carried 0 bytes meanwhile; {len(accepted) - 1} wait in the "
            "TX FIFO, 1 in the shift register"
        )
        flush = _summary(lines, "flush:")
        line = f'the line carried {len(accepted)} bytes "{accepted.decode("ascii")}", the buffer'
        assert flush.endswith(line), (name, flush)


def test_selftest_gives_up_when_the_loopback_delivers_nothing():
    """The self-test's wait for its pattern is bounded. Against a loopback that delivers nothing -- the
    sim's sim_drop_loopback: in loopback a finished character never reaches the receiver -- the test
    finds nothing to drain, sees the MSR mirror, writes 0x55 after a THRE read, then polls LSR with DR
    clear UART_SPIN_BUDGET times and returns UART_ERR_TIMEOUT with its loopback undone: one more LSR
    read finds the transmitter idle (TEMT; the dropped pattern left it), the FIFOs are reset from the
    shadow while still in the loopback, MCR restored and MSR read once. The loopback kept exactly that
    one character."""
    if not (_TRACE_CC and _tree()):
        return
    budget = _spin_budget()
    for name in um.PLACEMENTS:
        lines = _phase(_transcript(name), "selftest timeout")
        off = {reg: _off(name, reg) for reg in um.REGS}
        assert _trace(lines) == [
            ("R", off["LSR"], "LSR", 0x60, 1), ("R", off["MCR"], "MCR", 0x03, 1),
            ("R", off["LSR"], "LSR", 0x60, 1),
            ("W", off["MCR"], "MCR", 0x1A, 1), ("R", off["MSR"], "MSR", 0x99, 1),
            ("W", off["MCR"], "MCR", 0x15, 1), ("R", off["MSR"], "MSR", 0x6B, 1),
            ("R", off["LSR"], "LSR", 0x60, 1), ("W", off["THR"], "THR", 0x55, 1),
            ("R", off["LSR"], "LSR", 0x60, budget + 1),  # the DR wait, then the TEMT poll
            ("W", off["FCR"], "FCR", 0x87, 1), ("W", off["MCR"], "MCR", 0x03, 1),
            ("R", off["MSR"], "MSR", 0x06, 1),
        ], (name, lines)  # fmt: skip
        assert _summary(lines, "selftest:") == (
            "selftest: returns 7 (timed out); mcr 0x03 after it, fcr_shadow 0x81"
        )
        assert _summary(lines, "  device:") == (
            "  device: the loopback kept 1 characters from the receiver"
        )


def test_selftest_enters_the_loopback_only_with_the_transmitter_idle():
    """In the loopback the transmitter's shift register feeds the receiver and SOUT is held marking
    (PC16550D p.23), so bytes uart_tx accepted but the line has not taken would be looped back -- lost,
    and read in place of a pattern -- and a pattern left in the transmitter would go out on SOUT once
    MCR is restored. The review of U2 reproduced both: TX of "ABC", the self-test, then a flush gave a
    self-test failure and the line 0x41 0x55. So the test waits for TEMT before it enters the loopback:
    after TX of "ABC" its first LSR polls find the transmitter busy (THRE and TEMT clear, then THRE set
    with TEMT clear) until TEMT, and only then does it read MCR; it passes, and the line carries "ABC"
    exactly -- no pattern. TX's poll saved the parity error of a byte the receiver held (lsr_saved 0x04),
    which the test drains and discards with its error (0x00 after it). With the transmitter stalled
    (sim_stall_tx) and a character waiting behind the one shifting, the test polls LSR with TEMT clear
    UART_SPIN_BUDGET times and returns UART_ERR_BUSY having touched nothing else -- no MCR access, no
    FIFO reset -- and once the transmitter resumes, both characters go out; a framing error that arrived
    after TX's poll, which the test's first poll showed and cleared (0x89), it saved for the drain. (Its waits for THRE before a
    pattern stay bounded, but no sim knob reaches them now: TEMT, which implies THRE, is established
    first, and each pattern is read back before the next is written.)"""
    if not (_TRACE_CC and _tree()):
        return
    budget = _spin_budget()
    for name in um.PLACEMENTS:
        transcript = _transcript(name)
        off = {reg: _off(name, reg) for reg in um.REGS}
        lines = _phase(transcript, "selftest after tx")
        assert _summary(lines, "tx:").startswith("tx: returns 0 (ok); 3 of 3 bytes sent"), lines
        test = _events(_between(lines, "tx:", "selftest:"))  # the trace lines only
        first_mcr = next(k for k, e in enumerate(test) if e[2] == "MCR")
        polls = [v for _k, _o, reg, v in test[:first_mcr]]
        assert polls[-1] & 0x40 and not any(v & 0x40 for v in polls[:-1]) and len(polls) > 1, polls
        assert all(reg == "LSR" for _k, _o, reg, _v in test[:first_mcr]), test[:first_mcr]
        thr = [v for kind, _o, reg, v in test if kind == "W" and reg == "THR"]
        assert thr == [0x55, 0xAA], (name, thr)
        assert _summary(lines, "selftest:") == (
            "selftest: returns 0 (ok); mcr 0x03 after it, fcr_shadow 0x81"
        ), lines
        states = [ln for ln in lines if ln.startswith("  state:")]
        assert states == ["  state: lsr_saved 0x04", "  state: lsr_saved 0x00"], (name, states)
        assert _summary(lines, "flush:").endswith('the line carried 3 bytes "ABC", the buffer'), (
            lines
        )
        lines = _phase(transcript, "selftest stalled")
        assert _summary(lines, "tx:") == (
            "tx: returns 0 (ok); 2 of 2 bytes sent under init's burst cap of 16; tx overflows 0"
        )
        assert _trace(_between(lines, "tx:", "selftest:")) == [
            ("R", off["LSR"], "LSR", 0x89, 1), ("R", off["LSR"], "LSR", 0x81, budget - 1),
        ], (name, lines)  # fmt: skip
        assert _summary(lines, "selftest:") == (
            "selftest: returns 1 (busy); mcr 0x03 after it, fcr_shadow 0x81"
        )
        assert _summary(lines, "flush:").endswith('the line carried 2 bytes "<>", the buffer'), (
            lines
        )
        drain = _summary(lines, "drain:")
        assert drain == 'drain: 1 bytes "G"; OE 0 PE 0 FE 1 BI 0; lsr_saved 0x00 after it', drain


def test_selftest_passes_at_every_word_length():
    """The self-test's patterns are 0x55 and 0xAA at the word length init programmed: LCR[1:0] set the
    number of bits in each character (PC16550D p.18), so at 5, 6 and 7 bits a full 8-bit pattern cannot
    come back, and the review of U2 found the test failing every word length but 8 (status 6). After an
    init at each of 5, 6 and 7 data bits -- the state records word_bits, and the init has cleared the
    break a flush saved just before it (lsr_saved 0x10, then 0x00) -- the test writes 0x15/0x0A,
    0x15/0x2A and 0x55/0x2A, reads each back and passes."""
    if not (_TRACE_CC and _tree()):
        return
    for name in um.PLACEMENTS:
        lines = _phase(_transcript(name), "selftest at 5, 6 and 7 data bits")
        assert _summary(lines, "  state:") == "  state: lsr_saved 0x10", lines
        inits = [ln for ln in lines if ln.startswith("init:")]
        tests = [ln for ln in lines if ln.startswith("selftest:")]
        assert inits == [
            f"init: returns 0 (ok); format {w:#04x}, {5 + w} data bits: word_bits {5 + w}, lsr_saved 0x00"
            for w in range(3)
        ], (name, inits)
        assert tests == [
            f"selftest: returns 0 (ok) at {5 + w} data bits; mcr 0x03 after it" for w in range(3)
        ], (name, tests)
        events = _events(lines)
        thr = [v for kind, _o, reg, v in events if kind == "W" and reg == "THR"]
        assert thr == [0x15, 0x0A, 0x15, 0x2A, 0x55, 0x2A], (name, thr)
        back = [events[k + 1][3] for k, e in enumerate(events) if e[2] == "LSR" and e[3] & 0x01
                and k + 1 < len(events) and events[k + 1][2] == "RBR" and k > 0
                and any(x[2] == "THR" for x in events[:k])]  # fmt: skip
        assert back[-6:] == thr, (name, back)


def test_init_gives_up_when_the_line_never_goes_idle():
    """init's wait for TEMT (section 1.4 step 1: reconfigure only at idle) is bounded. With one character
    shifting forever (sim_stall_tx), init polls LSR -- THRE set, TEMT clear -- UART_SPIN_BUDGET times and
    returns UART_ERR_BUSY having touched nothing else: no LCR, divisor or FCR write, and the state as it
    was. A byte that arrived with a parity error after TX's poll shows it at init's first poll (0xA5),
    which clears it (PC16550D p.21); init saved it, and the drain after the transmitter resumes returns
    the byte with PE counted."""
    if not (_TRACE_CC and _tree()):
        return
    budget = _spin_budget()
    for name in um.PLACEMENTS:
        lines = _phase(_transcript(name), "init busy")
        off = {reg: _off(name, reg) for reg in um.REGS}
        assert _summary(lines, "tx:") == (
            "tx: returns 0 (ok); 1 of 1 bytes sent under init's burst cap of 16; tx overflows 0"
        )
        assert _trace(_between(lines, "tx:", "init:")) == [
            ("R", off["LSR"], "LSR", 0xA5, 1), ("R", off["LSR"], "LSR", 0xA1, budget - 1),
        ], (name, lines)  # fmt: skip
        drain = _summary(_phase(_transcript(name), "flush busy"), "drain:")
        assert drain == 'drain: 1 bytes "I"; OE 0 PE 1 FE 0 BI 0; lsr_saved 0x00 after it', drain
        assert _summary(lines, "init:") == "init: returns 1 (busy); fcr_shadow 0x81, tx_burst 16"


def test_flush_gives_up_when_the_line_never_goes_idle():
    """The flush's wait for TEMT is bounded. With the character still shifting forever it polls LSR --
    THRE set, TEMT clear -- UART_SPIN_BUDGET times and returns UART_ERR_BUSY, *nbusy the budget; when
    the transmitter resumes, the next flush sees TEMT at its first poll and the line has carried the
    character. (The receiver holds a byte meanwhile, DR set, its parity error saved by init's polls:
    test_init_gives_up_when_the_line_never_goes_idle.)"""
    if not (_TRACE_CC and _tree()):
        return
    budget = _spin_budget()
    for name in um.PLACEMENTS:
        lines = _phase(_transcript(name), "flush busy")
        off = {reg: _off(name, reg) for reg in um.REGS}
        flushes = lines[: next(k for k, ln in enumerate(lines) if ln.startswith("flush:"))]
        assert _trace(flushes) == [
            ("R", off["LSR"], "LSR", 0xA1, budget), ("R", off["LSR"], "LSR", 0xE1, 1),
        ], (name, lines)  # fmt: skip
        stalled = _summary(lines, "stalled flush:")
        assert stalled == f"stalled flush: returns 1 (busy); {budget} busy polls", (name, stalled)
        assert _summary(lines, "flush:") == (
            'flush: 0 busy polls; the line carried 1 bytes "<", the buffer'
        )


def _fed_by_device(lf, rid: int) -> bool:
    """Whether an MMIO-domain load feeds `rid`, through every claim that writes it, transitively."""
    seen, todo = set(), [rid]
    while todo:
        r = todo.pop()
        if r in seen:
            continue
        seen.add(r)
        for c in lf.claims:
            if r in c.wr:
                if c.op == "c.load" and c.domain == Domain.MMIO:
                    return True
                todo.extend(c.rd)
    return False


def test_the_driver_never_reads_fcr_back():
    """Hazard H-E, on the claim graph of every placement: each store to the FCR shadow (uart_state's
    first member) takes a value no device read feeds -- probe, init and the self-test each make one --
    and the unit's only reads at the FCR/IIR slot are the two IIR reads section 3.1 and section 1.4
    step 8 call for. On the bus, on every placement, each shadow the transcript reports is the value
    last written to FCR less FCR[2:1], the FIFO resets, which clear themselves when written (PC16550D
    p.22): no shadow carries them, so no read-modify-write from it can reset a FIFO unasked."""
    if not _tree():
        return
    for name in um.PLACEMENTS:
        res = _compile_driver(name)
        stores = {}
        slot2 = {}
        for fn, lf in res.lowered.functions.items():
            facts = ul._Facts(lf)  # the laws' own reading of pointers and offsets
            st = [rid for pname, rid, _ct in lf.params if pname == "st"]
            for c in lf.claims:
                if not st or c.op != "c.store" or c.domain == Domain.MMIO or not c.rd:
                    continue
                root, off = facts._root(c.rd[0])
                if root == st[0] and off is not None and off + (c.imm[0] if c.imm else 0) == 0:
                    stores.setdefault(fn, []).append(_fed_by_device(lf, c.rd[-1]))
            reads = [
                c for c in lf.claims
                if c.op == "c.load" and c.domain == Domain.MMIO
                and facts.offsets.get(c.id) == _off(name, "IIR")
            ]  # fmt: skip
            if reads:
                slot2[fn] = len(reads)
        assert stores == {
            "uart_probe": [False], "uart_init": [False], "uart_selftest": [False],
        }, (name, stores)  # fmt: skip
        assert slot2 == {"uart_probe": 1, "uart_init": 1}, (name, slot2)
    if not _TRACE_CC:
        return
    for name in um.PLACEMENTS:
        last, reported = None, []
        for line in _transcript(name).decode("ascii").splitlines():
            for _k, _o, reg, value in _events([line]):
                if reg == "FCR":
                    last = value
            for shadow in re.findall(r"fcr_shadow (0x[0-9a-f]{2})", line):
                reported.append((last, int(shadow, 16)))
        assert reported and all(w is not None for w, _s in reported), (name, reported)
        # each shadow is the last FCR write less FCR[2:1] (the self-test writes 0x87, keeps 0x81) ...
        assert all(s == w & ~_FCR_RESETS for w, s in reported), (name, reported)
        # ... so none carries the resets
        assert not [s for _w, s in reported if s & _FCR_RESETS], (name, reported)
        # the probe (it leaves the FIFOs off), init, the self-test, the one after TX and the stuck one;
        # each 16450-mode run's probe, init and self-test; init again before the TX timeout, the
        # self-test's timeout; then the stalled self-test (busy before it touched FCR) and the init that
        # found the line busy (each state as it was)
        timeouts = [0x81, 0x81, 0x81, 0x81]
        want = [0x00, 0x81, 0x81, 0x81, 0x81] + [0x00, 0x00, 0x00] * len(_NO_FIFO) + timeouts
        assert [s for _w, s in reported] == want, (name, reported)


def test_emitted_to_emitted_binary_matches_reference_behavior():
    """THE HEADLINE (blueprint U2): on every placement, the program linked from the driver as the
    oracle emits it with --linkable, and from the driver as bcir-cc emits it, prints the transcript of
    the same program built from the driver's source -- byte for byte, the bus trace included -- and
    the axi16550 placement's is the checked-in golden. The transcript is a function of the layout:
    the placements' differ, and their verdicts agree."""
    if not (_TRACE_CC and _tree()):
        return
    verdicts = set()
    for name in um.PLACEMENTS:
        source = _transcript(name)
        assert _transcript(name, "oracle") == source, f"{name}: the oracle's emit is not the source"
        assert _transcript(name, "twin") == source, f"{name}: the twin's emit is not the source"
        verdicts.add(_summary(_phase(source, "end"), "verdict:"))
    assert len({_transcript(name) for name in um.PLACEMENTS}) == 3
    assert verdicts == {
        "verdict: probe 16550A, init ok, selftest ok, tx 52/52 on the line, rx 36 bytes with "
        "OE 1 PE 3 FE 2 BI 0"
    }, verdicts
    with open(_GOLDEN, "rb") as fh:
        golden = fh.read()
    assert _transcript(_GOLDEN_PLACEMENT) == golden, (
        "runtime/c/uart_main.golden is not the reference build's transcript: review the "
        "change, then python3 -m bcir.tests.test_uart_driver --transcript axi16550 "
        "> runtime/c/uart_main.golden"
    )


def test_a_driver_laid_out_for_another_placement_is_refused():
    """Hazard H-A at run time: the driver compiled for the byte-strided pc16550d and run against the
    AXI core (stride 4, base 0x1000) -- or the other way round -- does not fit the device's lanes,
    and the harness refuses it (exit 4) after the probe instead of printing a transcript. The driver
    is built beside its own placement's generated header, the harness and the device beside the
    other's."""
    if not (_TRACE_CC and _tree()):
        return
    for driver, device in (("pc16550d", "axi16550"), ("axi16550", "pc16550d")):
        laid_out = os.path.join(_scratch(driver, "laid-out"), "uart_driver.c")
        d = _scratch(device, f"misplaced-{driver}")
        done = _run(_program(_TRACE_CC, d, laid_out, "uart_main_misplaced"))
        assert done.returncode == 4, (driver, device, done.returncode, done.stderr)
        assert f"do not fit placement {device}".encode("ascii") in done.stderr, done.stderr


def test_an_untrapped_build_is_refused():
    """The trap must be able to fail (laws.md L2): the program with the driver compiled WITHOUT the
    load/store tracing reaches the device with no access, and the harness says so (exit 3) instead
    of printing a transcript of a plain array."""
    if not (_CC and _tree()):
        return
    d = _scratch(_GOLDEN_PLACEMENT, "untrapped")
    done = _run(_program(_CC, d, os.path.join(d, "uart_driver.c"), "uart_main_plain", traced=False))
    assert done.returncode == 3, (done.returncode, done.stdout[-300:])
    assert b"the register window is not trapped" in done.stderr, done.stderr
    assert b"-- init" not in done.stdout


if __name__ == "__main__":
    if sys.argv[1:2] == ["--transcript"]:
        # the reference build's transcript, for the golden: python3 -m bcir.tests.test_uart_driver
        # --transcript axi16550 > runtime/c/uart_main.golden
        assert len(sys.argv) == 3 and sys.argv[2] in um.PLACEMENTS, "--transcript PLACEMENT"
        assert _TRACE_CC and _tree(), "needs clang and a source checkout"
        sys.stdout.buffer.write(_transcript(sys.argv[2]))
        sys.exit(0)
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
