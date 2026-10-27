"""The UART-16550 device registry (blueprint slice U0): one logical model, N placements, a RegMap contract.

`docs/kernel/BCIR_UART_DRIVER_BLUEPRINT.md` section 1.1 is the model, section 2 the variant matrix
and section 5 (U0) the shape of this module. `UART16550` is the one logical register file; a
`UartPlacement` says where a part puts it (stride, base offset, lane width) and what the part does
differently (the FCR reset value and gating, autoflow, modem lines, the interrupt output, the baud
reference). Every view is generated from those two and spelled nowhere else:

  * `emit_header(placement)` -- the vendor-style C header for a placement, deterministic text.
    `runtime/c/uart16550_regs.h` is the default placement's, and `fixtures/` holds one golden header
    per placement; both are byte-compared against a fresh emission by `bcir/tests/test_uart_model.py`.
  * `regmap_contract_for(placement)` -- the RegMap contract: the laid-out register facts a driver
    unit attests, the MMIO counterpart of the R12 call-ABI contract (`cfront/abi.py::AbiContract`).
  * `verify_regmap_contract(contract, placement)` -- its law: [] when the contract is the placement's,
    one message per lie otherwise (the pipeline wraps them, as it wraps `verify_abi_contract`'s).

Errata and documentation errors are registry data too (`ERRATA`): a part's published defects, and
the errors found in the documents while recording them, which are kept as provenance notes and never
reproduced as facts. `PROVENANCE`, at the end, records which document and page each fact was checked
against.

Regenerate the checked-in headers with `python3 -m bcir.frontends.devices.uart_model --write`;
`--check` exits 1 when one is stale. Dependency-free (stdlib only) and deterministic.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, fields
from fractions import Fraction


@dataclass(frozen=True)
class UartReg:
    """One logical register: where it sits (the A2:A0 slot), which direction reaches it, and the
    LCR[7] (DLAB) value it is reachable at -- None when either value reaches it."""

    name: str  # "RBR", "THR", "IER", "IIR", "FCR", "LCR", "MCR", "LSR", "MSR", "SCR", "DLL", "DLM"
    slot: int  # 0..7 (the A2:A0 address)
    access: str  # "r" | "w" | "rw"
    dlab: int | None  # None = don't care; 0/1 = only reachable at that DLAB value
    fields: tuple  # ((bit_hi, bit_lo, field_name, doc), ...) from section 1.1


# THE normative table (blueprint section 1.1; PC16550D SNLS378C Tables 1-2, pp.17-18). LSR and MSR
# are read-only here: their writes exist for factory test only (PC16550D p.22). Reserved bits --
# IER[7:4], IIR[5:4], FCR[5:4], MCR[7:5] where MCR[5] is not AFE -- are absent from the fields.
UART16550: tuple[UartReg, ...] = (
    UartReg("RBR", 0, "r", 0, ((7, 0, "DATA", "the received byte (bit 0 arrived first)"),)),
    UartReg("THR", 0, "w", 0, ((7, 0, "DATA", "the byte to transmit (bit 0 is sent first)"),)),
    UartReg(
        "IER",
        1,
        "rw",
        0,
        (
            (0, 0, "ERBFI", "received data available (and FIFO timeout) interrupt"),
            (1, 1, "ETBEI", "transmitter holding register empty interrupt"),
            (2, 2, "ELSI", "receiver line status interrupt"),
            (3, 3, "EDSSI", "modem status interrupt"),
        ),
    ),
    UartReg(
        "IIR",
        2,
        "r",
        None,
        (
            (0, 0, "INTPEND", "0 = an interrupt is pending (active low)"),
            (3, 1, "INTID", "the pending source (IIR_SOURCES); bit 3 = FIFO timeout"),
            (7, 6, "FIFOEN", "11 in FIFO mode (FCR[0] = 1), 00 in 16450 mode"),
        ),
    ),
    UartReg(
        "FCR",
        2,
        "w",
        None,
        (
            (0, 0, "FIFOEN", "enable both FIFOs; the other bits need it set in the same write"),
            (1, 1, "RXFIFORST", "clear the RX FIFO (self-clearing; shift register kept)"),
            (2, 2, "TXFIFORST", "clear the TX FIFO (self-clearing; shift register kept)"),
            (3, 3, "DMAMODE", "RXRDY/TXRDY signalling: 0 = mode 0, 1 = mode 1"),
            (7, 6, "RXTRIG", "RX FIFO trigger level (uart_rx_trigger_t)"),
        ),
    ),
    UartReg(
        "LCR",
        3,
        "rw",
        None,
        (
            (1, 0, "WLS", "word length: 00..11 = 5..8 bits"),
            (2, 2, "STB", "stop bits: 0 = 1; 1 = 1.5 for 5-bit words, else 2"),
            (3, 3, "PEN", "parity enable"),
            (4, 4, "EPS", "even parity select"),
            (5, 5, "SP", "stick parity"),
            (6, 6, "SB", "set break: SOUT forced low until cleared"),
            (7, 7, "DLAB", "divisor latch access: DLL/DLM over slots 0 and 1"),
        ),
    ),
    UartReg(
        "MCR",
        4,
        "rw",
        None,
        (
            (0, 0, "DTR", "data terminal ready (1 drives DTR low)"),
            (1, 1, "RTS", "request to send (1 drives RTS low)"),
            (2, 2, "OUT1", "user output 1 (1 drives OUT1 low)"),
            (3, 3, "OUT2", "user output 2; gates the IRQ line on PC/AT wiring"),
            (4, 4, "LOOP", "loopback: SOUT high, MCR[3:0] mirrored to MSR[7:4]"),
            (5, 5, "AFE", "autoflow enable (TL16C550C/D class only)"),
        ),
    ),
    UartReg(
        "LSR",
        5,
        "r",
        None,
        (
            (0, 0, "DR", "data ready: at least one byte received"),
            (1, 1, "OE", "overrun error (the shift-register byte is lost)"),
            (2, 2, "PE", "parity error of the byte at the FIFO head"),
            (3, 3, "FE", "framing error of the byte at the FIFO head"),
            (4, 4, "BI", "break interrupt at the FIFO head"),
            (5, 5, "THRE", "transmitter holding register / TX FIFO empty"),
            (6, 6, "TEMT", "transmitter empty: TX FIFO and shift register"),
            (7, 7, "FIFOERR", "an error byte anywhere in the RX FIFO"),
        ),
    ),
    UartReg(
        "MSR",
        6,
        "r",
        None,
        (
            (0, 0, "DCTS", "CTS changed since MSR was last read"),
            (1, 1, "DDSR", "DSR changed since MSR was last read"),
            (2, 2, "TERI", "trailing edge of RI"),
            (3, 3, "DDCD", "DCD changed since MSR was last read"),
            (4, 4, "CTS", "clear to send (RTS in loopback)"),
            (5, 5, "DSR", "data set ready (DTR in loopback)"),
            (6, 6, "RI", "ring indicator (OUT1 in loopback)"),
            (7, 7, "DCD", "data carrier detect (OUT2 in loopback)"),
        ),
    ),
    UartReg("SCR", 7, "rw", None, ((7, 0, "DATA", "scratch byte; no effect on the UART"),)),
    UartReg("DLL", 0, "rw", 1, ((7, 0, "DIV", "divisor bits 7:0"),)),
    UartReg("DLM", 1, "rw", 1, ((7, 0, "DIV", "divisor bits 15:8"),)),
)

#: The normative table by register name (a view of UART16550, never a second spelling).
REGS: dict[str, UartReg] = {r.name: r for r in UART16550}

#: A field that exists only on placements with a capability: (register, field) -> the
#: `UartPlacement` flag that makes it exist. MCR[5] is reserved zero on a part without autoflow
#: (PC16550D p.24, ST16C550 p.18, Core16550 p.9, PG143 p.34) and AFE on the TL16C550C/D.
CAPABILITY_FIELDS: dict[tuple[str, str], str] = {("MCR", "AFE"): "has_afe"}

#: IIR[3:0] with no interrupt pending (PC16550D Table 5, p.21).
IIR_NO_INTERRUPT = 0x1

#: The interrupt priority ladder, highest first (PC16550D Table 5, p.21):
#: (IIR[3:0], name, priority level, source, what clears it).
IIR_SOURCES: tuple = (
    (0x6, "RLS", 1, "receiver line status: OE, PE, FE or BI", "reading LSR"),
    (0x4, "RDA", 2, "received data available (trigger level)", "reading RBR below the trigger"),
    (0xC, "CTI", 2, "character timeout (FIFO mode only)", "reading RBR"),
    (0x2, "THRE", 3, "transmitter holding register empty", "reading IIR or writing THR"),
    (0x0, "MS", 4, "modem status: CTS, DSR, RI or DCD", "reading MSR"),
)

#: FCR[3]'s two DMA signalling modes (section 1.5; PC16550D p.6, the RXRDY/TXRDY pins): pin
#: handshakes only, documented and not driven by any slice yet (blueprint section 9).
DMA_MODES: tuple = (
    (0, "single transfer: RXRDY while a byte is held, TXRDY while nothing is held to send"),
    (1, "multi-transfer: RXRDY from the trigger level or a timeout until the RX FIFO is empty"),
)

#: The classic asynchronous tolerance bound the divisor law holds an achieved baud to
#: (blueprint section 1.3: |error| <= 3%).
BAUD_ERROR_LIMIT = Fraction(3, 100)


@dataclass(frozen=True)
class UartPlacement:
    """Where a part puts the register file and what it does differently (blueprint section 2)."""

    name: str  # "pc16550d" | "core16550_apb" | "axi16550"
    stride: int  # 1 | 4
    base_offset: int  # 0 | 0 | 0x1000
    reg_width: int  # 1 | 4 | 4 (bytes per register lane)
    fcr_reset: int  # 0x00 | 0x01 | 0x00
    fcr_gated: bool  # an FCR write programs bits 1..7 only with bit 0 set in it (H-C)
    has_afe: bool  # MCR5 autoflow exists
    has_modem: bool  # modem lines wired (H-NOMSR)
    ref_clk_hz: int  # the baud reference the divisor law uses
    irq_level: bool  # level-sensitive interrupt output (AXI) vs pin
    # -- 16750-family capability fields (U8; every U0-era placement takes the defaults,
    #    so U8 is non-disturbing by construction -- H-F/H-K/H-L are encoded HERE) --
    fifo_depth: int = 16  # 16 | 64 | 128 | synthesis value (H-K: data, never probed)
    rx_triggers: tuple = (1, 4, 8, 14)  # BYTE VALUES per FCR[7:6] encoding (H-F/H-G)
    rx_triggers_alt: tuple = ()  # 64-byte/alternate-mode byte values, () = no alt mode
    tx_triggers: tuple = ()  # FCR[5:4] TX trigger spaces (750E), () = none
    mode64_key: str = ""  # "" | "dlab" (classic 750: FCR[5] under LCR[7]=1)
    flow_mech: str = "none"  # "none" | "mcr5_afe" | "efr_tcr" (750E) -- the U6/U9 dispatch
    has_tlr: bool = False  # programmable x8 triggers behind EFR[4]+MCR[6]
    has_frac_divisor: bool = False  # DLF present (750E)
    has_sleep: bool = False  # EFR[4]&IER[4] sleep mode
    has_scr: bool = True  # False on Lattice (H-L: probe must consult this)
    # Which FCR bits exist. FCR[5:4] are reserved on the 16550 class (PC16550D p.22, Core16550
    # p.6, PG143 p.27), so the default is 0xCF; the blueprint's Lattice value 0xC6 is this with
    # bits 0 and 3 cleared.
    fcr_impl_mask: int = 0xCF
    reset_honest: bool = True  # False when reset values are synthesis attributes /
    #                            registers survive reset (Lattice, 750E DLL/DLH; H-M)
    errata: tuple = ()  # ids into ERRATA this placement's part is subject to (none in U0)


# The three U0 placements (blueprint section 2; the facts and their pages are in PROVENANCE).
# Every one gates FCR: the gating sentence is the PC16550D's own (p.22), repeated by TL16C550C/D,
# ST16C550 and Core16550, and PG143/DS748 defer to the PC16550D data sheet for register behaviour.
PLACEMENTS: dict[str, UartPlacement] = {
    "pc16550d": UartPlacement(
        name="pc16550d",
        stride=1,
        base_offset=0,
        reg_width=1,
        fcr_reset=0x00,
        fcr_gated=True,
        has_afe=False,
        has_modem=True,
        ref_clk_hz=1_843_200,
        irq_level=False,
    ),
    "core16550_apb": UartPlacement(
        name="core16550_apb",
        stride=4,
        base_offset=0,
        reg_width=4,
        fcr_reset=0x01,
        fcr_gated=True,
        has_afe=False,
        has_modem=True,
        ref_clk_hz=18_432_000,
        irq_level=False,
    ),
    "axi16550": UartPlacement(
        name="axi16550",
        stride=4,
        base_offset=0x1000,
        reg_width=4,
        fcr_reset=0x00,
        fcr_gated=True,
        has_afe=False,
        has_modem=True,
        ref_clk_hz=100_000_000,
        irq_level=True,
    ),
}

#: The placement `runtime/c/uart16550_regs.h` is generated for: the byte-strided silicon.
DEFAULT_PLACEMENT = "pc16550d"


# --- errata: a part's published defects, and the documents' own errors -----------------------------


@dataclass(frozen=True)
class DocNote:
    """A documentation error found while recording a source -- provenance, never a fact.

    `where` is the location, `printed` what the document prints there, `corrected` what the
    arithmetic or the unit gives, `note` the reading; `cells` names the table cells the note covers
    as (table, baud, k) keys, k the divisor decrement, () for prose."""

    where: str
    printed: str
    corrected: str
    note: str
    cells: tuple = ()


@dataclass(frozen=True)
class Erratum:
    """One published defect of a part: what it applies to, when it bites, what to do about it."""

    id: str  # the source's identifier
    parts: tuple  # the part numbers it names
    revisions: str  # the silicon revisions it applies to, as the source states them
    condition: str  # when the defect bites
    workarounds: tuple  # each workaround the source gives, in its order
    severity: str  # the source's own grade
    source: str  # document, issue and revision dates
    pages: str  # where in the source
    provenance: tuple = ()  # DocNote, ... -- documentation errors in the source


_SLLZ058A_RATES = (1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200)

ERRATA: dict[str, Erratum] = {
    "SLLZ058A": Erratum(
        id="SLLZ058A",
        parts=("TL16C752C", "TL16C754C", "TL16C2752"),
        revisions="the parts as named; the erratum gives no silicon revision or date code",
        condition=(
            "receiving a stream of characters whose STOP bits are shorter than a bit period: "
            "after the STOP bit is sampled and verified the receiver waits about a bit period "
            "before it looks for the next START bit, so a START bit can be missed and a framing "
            "error taken; the shortfall is cumulative, so it can follow a few or many characters"
        ),
        workarounds=(
            "program 1.5 or 2 stop bits",
            "decrease the reference clock divisor by 1, so the received STOP bit appears longer",
        ),
        severity="Moderate",
        source="TI SLLZ058A, October 2009, revised October 2010",
        pages="p.1 (problem, work around, severity); pp.2-3 (Tables 1-4, baud rate vs divisor)",
        provenance=(
            DocNote(
                where="p.1, sections 1 and 2",
                printed="104 ms (also 98 ms and 96-ms)",
                corrected="104.17 us at 9600 baud; 95.5 us at divisor 11 from 1.8432 MHz",
                note=(
                    "bit periods printed in milliseconds: 1/9600 s is 104.17 microseconds, and "
                    "1.8432 MHz/(16 x 11) = 10472.7 baud is a 95.5-microsecond bit (the 96)"
                ),
            ),
            DocNote(
                where="p.2, Table 2 (3.6864-MHz clock), the DIVISOR - 2 columns",
                printed="1200 baud, divisor 190: 12126.316 baud, 910.53%",
                corrected="1212.632 baud, 1.05%",
                note=(
                    "every DIVISOR - 2 rate is printed ten times the true clock/(16 x divisor), "
                    "and its error column is computed from the misprinted rate"
                ),
                cells=tuple((2, baud, 2) for baud in _SLLZ058A_RATES[:-1]),
            ),
            DocNote(
                where="pp.2-3, Tables 3 (14.746 MHz) and 4 (16 MHz), the DIVISOR - 1 and - 2 columns",
                printed="Table 4, 115200 baud, divisor 8: 130198.915 baud, 13.02%",
                corrected="125000 baud, 8.51%",
                note=(
                    "the decremented rates are computed from the unrounded divisor "
                    "clock/(16 x baud) - k, not from the integer divisor the table prints, which "
                    "is what the divisor latch holds: Table 3 moves by under 40 ppm, Table 4 by "
                    "up to 4.8%"
                ),
                cells=tuple(
                    (table, baud, k) for table in (3, 4) for baud in _SLLZ058A_RATES for k in (1, 2)
                ),
            ),
        ),
    ),
}


# --- baud generation (blueprint section 1.3) ---------------------------------------------------------


def achieved_baud(ref_clk_hz: int, divisor: int) -> Fraction:
    """The rate a divisor gives, exactly: ref_clk_hz / (16 x divisor). The divisor latch is 16 bits;
    a divisor of 0 is refused (PC16550D p.17: "not recommended"; the U1 divisor law)."""
    if not _is_int(ref_clk_hz) or ref_clk_hz <= 0:
        raise ValueError(f"reference clock {ref_clk_hz!r} is not a positive integer")
    if not _is_int(divisor) or not 1 <= divisor <= 0xFFFF:
        raise ValueError(f"divisor {divisor!r} is not in 1..65535")
    return Fraction(ref_clk_hz, 16 * divisor)


def baud_error(ref_clk_hz: int, divisor: int, baud: int | Fraction) -> Fraction:
    """The signed relative error of the achieved rate against `baud`, exactly. A rate that is not
    an integer (134.5 baud, PC16550D Table 4) is a Fraction, never a float."""
    if not (_is_int(baud) or isinstance(baud, Fraction)) or baud <= 0:
        raise ValueError(f"baud {baud!r} is not a positive integer or Fraction")
    return achieved_baud(ref_clk_hz, divisor) / baud - 1


# --- the placement record's own law ------------------------------------------------------------------

_NAME = re.compile(r"[a-z][a-z0-9_]*\Z", re.ASCII)
_POWERS = (1, 2, 4, 8)
_INT_FIELDS = (
    "stride",
    "base_offset",
    "reg_width",
    "fcr_reset",
    "ref_clk_hz",
    "fifo_depth",
    "fcr_impl_mask",
)
_BOOL_FIELDS = (
    "fcr_gated",
    "has_afe",
    "has_modem",
    "irq_level",
    "has_tlr",
    "has_frac_divisor",
    "has_sleep",
    "has_scr",
    "reset_honest",
)
_MODE64_KEYS = ("", "dlab")
_FLOW_MECHS = ("none", "mcr5_afe", "efr_tcr")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _trigger_problem(label: str, triggers, limit: int | None, *, required: bool) -> str | None:
    """Four strictly increasing positive byte values (none, where `required` is False), each at most
    `limit` when the record states the FIFO depth they apply to."""
    if triggers == () and not required:
        return None
    if (
        not isinstance(triggers, tuple)
        or len(triggers) != 4
        or not all(_is_int(t) and t > 0 for t in triggers)
        or any(a >= b for a, b in zip(triggers, triggers[1:]))
        or (limit is not None and triggers[-1] > limit)
    ):
        bound = f", at most {limit}" if limit is not None else ""
        return f"{label} {triggers!r} is not four increasing positive byte values{bound}"
    return None


def placement_problems(p) -> list[str]:
    """Every way a placement record is not well formed; [] when it is. Total: a malformed record
    is a list of findings, never an exception."""
    if not isinstance(p, UartPlacement):
        return [f"not a UartPlacement: {type(p).__name__}"]
    out = []
    if not isinstance(p.name, str) or not _NAME.match(p.name):
        out.append(f"placement name {p.name!r} is not a lower-case C identifier")
    out += [
        f"{f} {getattr(p, f)!r} is not an integer"
        for f in _INT_FIELDS
        if not _is_int(getattr(p, f))
    ]
    out += [
        f"{f} {getattr(p, f)!r} is not a bool"
        for f in _BOOL_FIELDS
        if not isinstance(getattr(p, f), bool)
    ]
    if out:
        return out  # the arithmetic below reads these as integers and flags
    if p.stride not in _POWERS:
        out.append(f"stride {p.stride} is not one of {_POWERS}")
    if p.reg_width not in _POWERS or p.reg_width > p.stride:
        out.append(f"reg_width {p.reg_width} does not fit in stride {p.stride}")
    if p.base_offset < 0 or (p.stride in _POWERS and p.base_offset % p.stride):
        out.append(f"base_offset {p.base_offset:#x} is not a non-negative multiple of the stride")
    if not 0 <= p.fcr_impl_mask <= 0xFF:
        out.append(f"fcr_impl_mask {p.fcr_impl_mask:#x} is not a byte")
    if not 0 <= p.fcr_reset <= 0xFF or p.fcr_reset & ~p.fcr_impl_mask:
        out.append(
            f"fcr_reset {p.fcr_reset:#04x} sets a bit outside fcr_impl_mask {p.fcr_impl_mask:#04x}"
        )
    if p.ref_clk_hz <= 0:
        out.append(f"ref_clk_hz {p.ref_clk_hz} is not positive")
    if p.fifo_depth <= 0:
        out.append(f"fifo_depth {p.fifo_depth} is not positive")
    for problem in (
        _trigger_problem("rx_triggers", p.rx_triggers, p.fifo_depth, required=True),
        # the alternate mode's depth is not a field of the record (the classic 750's 64 bytes)
        _trigger_problem("rx_triggers_alt", p.rx_triggers_alt, None, required=False),
        _trigger_problem("tx_triggers", p.tx_triggers, p.fifo_depth, required=False),
    ):
        if problem:
            out.append(problem)
    if p.mode64_key not in _MODE64_KEYS:
        out.append(f"mode64_key {p.mode64_key!r} is not one of {_MODE64_KEYS}")
    if p.flow_mech not in _FLOW_MECHS:
        out.append(f"flow_mech {p.flow_mech!r} is not one of {_FLOW_MECHS}")
    elif (p.flow_mech == "mcr5_afe") != p.has_afe:
        out.append(f"flow_mech {p.flow_mech!r} disagrees with has_afe={p.has_afe}")
    if not isinstance(p.errata, tuple) or len(set(p.errata)) != len(p.errata):
        out.append(f"errata {p.errata!r} is not a tuple of distinct erratum ids")
    else:
        unknown = [e for e in p.errata if e not in ERRATA]
        if unknown:
            out.append(f"errata {unknown!r} are not in ERRATA")
    return out


# U0's generator renders the 16550 core. A placement whose capability fields leave it (the U8
# family: alternate or TX triggers, the enhanced keys, no SCR, another FCR) is refused by name
# rather than rendered without them.
_U0_DEFAULTS = {
    f.name: f.default
    for f in fields(UartPlacement)
    if f.name
    in (
        "rx_triggers_alt",
        "tx_triggers",
        "mode64_key",
        "has_tlr",
        "has_frac_divisor",
        "has_sleep",
        "has_scr",
        "fcr_impl_mask",
    )
}


def _unrendered(p: UartPlacement) -> list[str]:
    out = [name for name, default in _U0_DEFAULTS.items() if getattr(p, name) != default]
    if p.flow_mech == "efr_tcr":
        out.append("flow_mech")
    return out


def _supported(p) -> UartPlacement:
    """The one gate in front of every view: a well-formed record U0's generator can render."""
    problems = placement_problems(p)
    if problems:
        raise ValueError(f"malformed placement: {'; '.join(problems)}")
    unrendered = _unrendered(p)
    if unrendered:
        raise ValueError(
            f"placement {p.name!r} sets {', '.join(unrendered)}, which U0's generator does not "
            "render (the 16750-family rendering is slice U8)"
        )
    return p


def _present_fields(reg: UartReg, p: UartPlacement) -> tuple:
    return tuple(
        f
        for f in reg.fields
        if (reg.name, f[2]) not in CAPABILITY_FIELDS
        or getattr(p, CAPABILITY_FIELDS[(reg.name, f[2])])
    )


def reg_offset(p: UartPlacement, reg: UartReg) -> int:
    """A register's byte offset from the device base: base_offset + slot * stride."""
    return p.base_offset + reg.slot * p.stride


# --- the generated header ----------------------------------------------------------------------------

BANNER = "/* GENERATED by uart_model.py -- do not edit */"
_ACCESS = {"r": "read", "w": "write", "rw": "read/write"}
_ROLE = {
    "RBR": "receiver buffer",
    "THR": "transmitter holding",
    "IER": "interrupt enable",
    "IIR": "interrupt identification",
    "FCR": "FIFO control",
    "LCR": "line control",
    "MCR": "modem control",
    "LSR": "line status",
    "MSR": "modem status",
    "SCR": "scratch",
    "DLL": "divisor latch, low byte",
    "DLM": "divisor latch, high byte",
}


def _define(name: str, value: str, comment: str = "") -> str:
    line = f"#define {name:<30} {value}"
    return f"{line:<48} /* {comment} */" if comment else line


def emit_header(placement: UartPlacement) -> str:
    """The deterministic C header for a placement: an offset per register (stride and base applied),
    a shift and an in-place mask per field, the RX trigger enum named by byte value, and the
    `uart_regmap_contract` literal (name, stride, base, fcr_reset, ref_clk_hz) a driver unit embeds
    -- the RegMap contract's C-side anchor."""
    p = _supported(placement)
    lines = [
        BANNER,
        f'/* uart16550_regs.h -- the 16550 register file laid out for placement "{p.name}".',
        " *",
        " * One logical register file (bcir/frontends/devices/uart_model.py, UART16550), placed with",
        f" * stride {p.stride}, base offset {p.base_offset:#x} and {p.reg_width}-byte register lanes: a register's",
        " * offset is base_offset + slot * stride bytes from the device base. Blueprint slice",
        " * U0 (docs/kernel/BCIR_UART_DRIVER_BLUEPRINT.md); regenerate with",
        " *     python3 -m bcir.frontends.devices.uart_model --write",
        " */",
        "#ifndef UART16550_REGS_H",
        "#define UART16550_REGS_H",
        "",
        "#include <stdint.h>",
        "",
        _define("UART_REG_WIDTH", f"{p.reg_width}u", "bytes per register lane"),
        "",
        "/* Register offsets. DLAB is the LCR[7] value a register is reachable at (x: either). */",
    ]
    for reg in UART16550:
        dlab = "x" if reg.dlab is None else str(reg.dlab)
        comment = f"slot {reg.slot}, {_ACCESS[reg.access]}, DLAB {dlab}"
        lines.append(_define(f"UART_{reg.name}_OFF", f"{reg_offset(p, reg):#x}u", comment))
    lines += ["", "/* Fields: _SHIFT is a field's lowest bit, _MASK the field in place. */"]
    for reg in UART16550:
        present = _present_fields(reg, p)
        lines.append(f"/* {reg.name}: {_ROLE[reg.name]} */")
        for hi, lo, name, doc in present:
            mask = ((1 << (hi - lo + 1)) - 1) << lo
            lines.append(_define(f"UART_{reg.name}_{name}_SHIFT", f"{lo}u"))
            lines.append(_define(f"UART_{reg.name}_{name}_MASK", f"{mask:#04x}u", doc))
    rxtrig = next(f for f in REGS["FCR"].fields if f[2] == "RXTRIG")
    lines += [
        "",
        "/* RX FIFO trigger levels: FCR[7:6] in place, each named by its byte count here. */",
        "typedef enum uart_rx_trigger {",
    ]
    for code, count in enumerate(p.rx_triggers):
        sep = "," if code < len(p.rx_triggers) - 1 else ""
        lines.append(f"    UART_RX_TRIGGER_{count} = {code << rxtrig[1]:#04x}{sep}")
    lines += [
        "} uart_rx_trigger_t;",
        "",
        "/* The RegMap contract's C anchor: the placement this unit was laid out for. */",
        "typedef struct uart_regmap_contract {",
        "    const char *placement;",
        "    uint32_t stride;",
        "    uint32_t base_offset;",
        "    uint32_t fcr_reset;",
        "    uint32_t ref_clk_hz;",
        "} uart_regmap_contract_t;",
        "",
        "static const uart_regmap_contract_t uart_regmap_contract = {",
        f'    "{p.name}", {p.stride}u, {p.base_offset:#x}u, {p.fcr_reset:#04x}u, {p.ref_clk_hz}u',
        "};",
        "",
        "#endif /* UART16550_REGS_H */",
        "",
    ]
    return "\n".join(lines)


# --- the RegMap contract (the R12 call-ABI contract pattern, for the register map) ---------------------


@dataclass(frozen=True)
class RegMapContract:
    """The register map a unit was laid out for -- mirrors `cfront.abi.AbiContract`."""

    placement: str
    stride: int
    base_offset: int
    regs: tuple  # (name, offset, access, dlab) per register -- the laid-out facts


def regmap_contract_for(placement: UartPlacement) -> RegMapContract:
    """Record the contract a placement lays the normative table out as."""
    p = _supported(placement)
    regs = tuple((r.name, reg_offset(p, r), r.access, r.dlab) for r in UART16550)
    return RegMapContract(placement=p.name, stride=p.stride, base_offset=p.base_offset, regs=regs)


def _show(value) -> str:
    return f"{value:#x}" if _is_int(value) else repr(value)


def _same(got, want) -> bool:
    """Equal and of the same types, element by element: `True` is not the stride 1."""
    if isinstance(want, tuple):
        return (
            isinstance(got, tuple)
            and len(got) == len(want)
            and all(_same(g, w) for g, w in zip(got, want))
        )
    return type(got) is type(want) and got == want


def verify_regmap_contract(contract, placement) -> list[str]:
    """The RegMap law: the recorded contract must be the registry's placement laid out by the
    normative table. Returns one message per lie ([] == clean); total over malformed input."""
    if not isinstance(contract, RegMapContract):
        return [f"not a RegMapContract: {type(contract).__name__}"]
    if not isinstance(placement, UartPlacement):
        return [f"not a UartPlacement: {type(placement).__name__}"]
    if not isinstance(contract.placement, str) or contract.placement not in PLACEMENTS:
        return [
            f"unknown placement {contract.placement!r} (registry: {', '.join(sorted(PLACEMENTS))})"
        ]
    if contract.placement != placement.name:
        return [
            f"contract placement {contract.placement!r} != the unit placement {placement.name!r}"
        ]
    if placement != PLACEMENTS[placement.name]:
        return [f"placement record {placement.name!r} is not the registry's"]
    want = regmap_contract_for(placement)
    msgs = []
    if not _same(contract.stride, want.stride):
        msgs.append(f"stride: contract {_show(contract.stride)} != placement {_show(want.stride)}")
    if not _same(contract.base_offset, want.base_offset):
        msgs.append(
            f"base offset: contract {_show(contract.base_offset)} != placement "
            f"{_show(want.base_offset)}"
        )
    if not isinstance(contract.regs, tuple) or len(contract.regs) != len(want.regs):
        size = len(contract.regs) if isinstance(contract.regs, tuple) else contract.regs
        msgs.append(f"registers: contract lays out {size!r}; the model has {len(want.regs)}")
        return msgs
    for got, expected in zip(contract.regs, want.regs):
        if not _same(got, expected):
            msgs.append(f"register {expected[0]}: contract {got!r} != laid out {expected!r}")
    return msgs


# --- the checked-in views: golden fixtures and the runtime header --------------------------------------

#: this package, relative to the repository root
PACKAGE_DIR = os.path.join("bcir", "frontends", "devices")
#: package-relative directory of the golden headers (shipped: pyproject package-data)
FIXTURES = "fixtures"
#: repository-relative path of the default placement's header
RUNTIME_HEADER = os.path.join("runtime", "c", "uart16550_regs.h")


def fixture_name(placement_name: str) -> str:
    return f"uart16550_{placement_name}.h"


def read_fixture(placement_name: str) -> bytes:
    """A golden header's bytes, read through the package (it ships in the wheel)."""
    from importlib import resources  # noqa: PLC0415

    target = resources.files(__package__) / FIXTURES / fixture_name(placement_name)
    return target.read_bytes()


def _repo_root() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )


def generated_files() -> dict[str, bytes]:
    """Every checked-in view, by its path relative to the repository root -> its bytes."""
    out = {
        os.path.join(PACKAGE_DIR, FIXTURES, fixture_name(name)): emit_header(p).encode("utf-8")
        for name, p in PLACEMENTS.items()
    }
    out[RUNTIME_HEADER] = emit_header(PLACEMENTS[DEFAULT_PLACEMENT]).encode("utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    """--check: exit 0 when every checked-in view is current, 1 when one is stale, 2 when this is
    not a source checkout. --write: regenerate them (exit 2 outside a checkout)."""
    args = sys.argv[1:] if argv is None else argv
    if args not in (["--check"], ["--write"]):
        print("usage: python3 -m bcir.frontends.devices.uart_model --check | --write")
        return 2
    root = _repo_root()
    if not os.path.isdir(os.path.join(root, "runtime", "c")):
        print(f"not a source checkout ({root} has no runtime/c): nothing to {args[0][2:]}")
        return 2
    stale = []
    for rel, data in generated_files().items():
        path = os.path.join(root, rel)
        try:
            with open(path, "rb") as fh:
                current = fh.read()
        except OSError:
            current = None
        if current == data:
            continue
        if args == ["--write"]:
            with open(path, "wb") as fh:
                fh.write(data)
            print(f"wrote {rel}")
        else:
            stale.append(rel)
    for rel in stale:
        print(f"stale: {rel} (python3 -m bcir.frontends.devices.uart_model --write)")
    return 1 if stale else 0


# --- provenance: what each fact was checked against ----------------------------------------------------
#
# (fact, documents and pages, reading). Pages are the PDFs' own; every document is one of the
# blueprint's sources (section 0). Documentation errors and cross-document disagreements are
# recorded as such, never adopted.
PROVENANCE: tuple = (
    (
        "the twelve registers' slots, access and DLAB reachability",
        "PC16550D SNLS378C pp.17-18 (Tables 1, 2); TL16C550C SLLS177I pp.21-23 (Tables 7-1, 7-3); "
        "TL16C550D SLLS597E p.32 (Table 1); ST16C550 rev 5.0.1 p.13 (Table 4); Core16550 "
        "DS50003798A p.5 (Table 1-1); PG143 pp.18-19; DS748 p.9 (Table 4)",
        "agree. PC16550D Table 2 prints MODEM Control at A2-A0 = 000 in the extracted text; its "
        "Table 1 and every other source put MCR at slot 4. AXI reads FCR at slot 2 with LCR[7]=1, "
        "so its IIR is reachable at DLAB 0 only (DS748 p.9, PG143 p.30), which PG143's own map "
        "(p.18) prints as 'x'; the portable table keeps PC16550D's IIR, and the DLAB-bracket law "
        "keeps a driver from reading slot 2 inside a bracket",
    ),
    (
        "the field tables (IER, IIR, FCR, LCR, MCR, LSR, MSR) and the IIR priority ladder",
        "PC16550D pp.17-18 (Table 1), pp.20-24 (sections 8.6.2-8.6.9), p.21 (Table 5); TL16C550C "
        "pp.23-24; Core16550 pp.5-10 (Tables 1-4, 1-7 to 1-14); PG143 pp.24-36",
        "agree. Core16550 p.10 states the loopback mirror as CTS = DTR and DSR = RTS, swapped "
        "against PC16550D p.24 (CTS = RTS, DSR = DTR); the model follows the PC16550D",
    ),
    (
        "reset values: IER 00, IIR 01, FCR 00, LCR 00, MCR 00, LSR 60, MSR XXXX0000; SCR and the "
        "divisor latches not reset",
        "PC16550D p.19 (Table 3), p.5 (MR); TL16C550C p.22 (Table 7-2); TL16C550D p.32 (Table 2); "
        "ST16C550 p.13",
        "agree; ST16C550 initializes SPR to FF (p.13)",
    ),
    (
        "FCR[0] gates the other FCR bits in the same write",
        "PC16550D p.22; TL16C550C p.24; TL16C550D p.33; ST16C550 p.15; Core16550 p.6; PG143 p.26 "
        "and DS748 p.10 defer register behaviour to the PC16550D data sheet",
        "every source: fcr_gated is True on all three placements (the blueprint's section 2 "
        "matrix had 'n/a' for the silicon)",
    ),
    (
        "FCR[5:4] reserved on the 16550 class",
        "PC16550D p.22; Core16550 p.6; PG143 p.27",
        "agree: fcr_impl_mask 0xCF",
    ),
    (
        "pc16550d: stride 1, base 0, 1-byte lanes, FCR reset 00, no AFE, modem lines, INTR",
        "PC16550D pp.4-5 (A2-A0, D7-D0, INTR), p.19, p.24 (MCR[7:5] permanently 0)",
        "agree",
    ),
    (
        "pc16550d: ref_clk_hz 1843200",
        "PC16550D p.17 (XIN, DC to 24 MHz), p.19 (Table 4, the 1.8432-MHz crystal column)",
        "the classic crystal; the part takes any clock to 24 MHz",
    ),
    (
        "core16550_apb: stride 4 (PADDR 00..1C), base 0, FCR reset 01, DLL/DLM reset 01/00, IIR "
        "reset C1, MCR[7:4] reserved, PCLK the baud reference",
        "Core16550 DS50003798A p.5 (Table 1-1, section 1.1.5), p.6 (Table 1-5), p.9 (Table 1-11), "
        "p.13 (PCLK)",
        "agree. IIR C1 follows from FCR 01 (FIFO mode at reset). The data ports are 8 bits "
        "(PWDATA/PRDATA[7:0], p.13): reg_width 4 is the APB lane the word-aligned PADDR implies",
    ),
    (
        "core16550_apb: ref_clk_hz 18432000",
        "Core16550 pp.6-7 (Table 1-6, the 18.432-MHz reference clock)",
        "the handbook's own reference; PCLK is the integration's bus clock",
    ),
    (
        "axi16550: stride 4 at base 0x1000 (0x1000..0x101C), 32-bit lanes, FCR reset 00 (readable "
        "at LCR[7]=1), LSR/MSR R/W, level-sensitive IP2INTC_Irpt, 16450 mode a build generic",
        "PG143 pp.17-19, pp.26-28; DS748 p.4 (level sensitive), p.5 (P3), p.7 (G7 C_IS_A_16550), "
        "p.9 (Table 4), p.13 (Table 9)",
        "agree",
    ),
    (
        "axi16550: ref_clk_hz 100000000 (S_AXI_ACLK; external xin below ACLK/2 optional)",
        "DS748 p.7 (G2 C_S_AXI_ACLK_FREQ_HZ default 100000000), p.8 (notes 1-2), p.19 (Example "
        "1); PG143 p.52 (Example 1)",
        "agree",
    ),
    (
        "baud: divisor = reference / (16 x baud), 16-bit, DLL then DLM, either latch reloads the "
        "counter; divisor 0 not recommended",
        "PC16550D p.17; TL16C550C p.31; TL16C550D p.39; ST16C550 p.10; Core16550 p.5; DS748 p.19",
        "agree on the formula. The divisor range is 1..65535 in TL16C550C/D, ST16C550 and "
        "Core16550, and 2..65535 in PC16550D p.17",
    ),
    (
        "the worked divisor examples",
        "PG143 p.52 and DS748 p.19 (100 MHz, 56 kbaud: 0x6F); PG143 p.53 and DS748 p.20 "
        "(1.8432-MHz xin, 56 kbaud: 2); PC16550D p.19 (Table 4); Core16550 pp.6-7 (Table 1-6)",
        "the AXI examples truncate (111 for 111.6), PC16550D Table 4 rounds (21 for 20.6). "
        "PC16550D Table 4 prints 920 for 1200 baud at 18.432 MHz where the formula gives 960, and "
        "Core16550 Table 1-6 copies the 920 with the +4.35% it gives; Core16550 prints the 110 and "
        "134.5 baud errors as the deviation in baud times 100 (-0.2865, 0.0876), not in percent",
    ),
    (
        "the 16-byte FIFOs, 3 error bits per RX byte, the overrun rule, the character timeout",
        "PC16550D p.1, p.16, p.21; ST16C550 p.18 (FIFO data not corrupted); DS748 p.4; Core16550 "
        "p.9",
        "agree",
    ),
    (
        "ST16C550 'enhanced feature register' behind DLAB (hazard H-B)",
        "ST16C550 rev 5.0.1 p.18 (LCR bit 7) against p.13 (Table 4, twelve registers)",
        "named once, in the LCR bit-7 text; the register table documents no such register",
    ),
    (
        "SLLZ058A (ERRATA)",
        "TI SLLZ058A pp.1-3; TL16C754C data sheet SLLS644H (revised January 2014)",
        "recorded with its three documentation errors (DocNote); the TL16C754C data sheet does "
        "not refer to the erratum",
    ),
)


if __name__ == "__main__":
    raise SystemExit(main())
