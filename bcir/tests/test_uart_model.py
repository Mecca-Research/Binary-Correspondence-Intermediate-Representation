"""Blueprint slice U0: the UART16550 registry -- one model, three placements, the RegMap contract.

`docs/kernel/BCIR_UART_DRIVER_BLUEPRINT.md` section 5 (U0) names the first five tests below. The rest
hold the module's facts to the documents they came from -- the blueprint's own rule: validate against
the sources' worked examples, never a round trip -- and hold the errata registry, the placement
record's law, the regeneration command and the shipped package. Pages are the PDFs' own; the
documents are the blueprint's sources (section 0). `uart_model.PROVENANCE` lists every fact checked.
"""

import dataclasses
import os
import re
import shutil
import subprocess
import tempfile
from fractions import Fraction

from bcir.frontends.cfront import compile_unit
from bcir.frontends.devices import uart_model as um
from bcir.model import Domain

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_CC = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _printed_as(value: Fraction, printed: str) -> bool:
    """`value`, rounded half away from zero to the decimals `printed` carries, is `printed`: a
    document is held to its own precision, never to a tolerance of ours."""
    scale = 10 ** len(printed.partition(".")[2])
    magnitude = abs(value) * scale
    whole = int(magnitude) + (1 if magnitude - int(magnitude) >= Fraction(1, 2) else 0)
    return Fraction(whole if value >= 0 else -whole, scale) == Fraction(printed)


def _fmt(value: Fraction, decimals: int) -> str:
    """`value` as the documents print it: rounded to `decimals`, trailing zeros dropped."""
    scale = 10**decimals
    whole = int(abs(value) * scale + Fraction(1, 2))
    text = f"{whole // scale}.{whole % scale:0{decimals}d}" if decimals else str(whole)
    return ("-" if value < 0 else "") + (text.rstrip("0").rstrip(".") if decimals else text)


# --- the five tests the blueprint names ----------------------------------------------------------------


def test_the_normative_table_matches_the_datasheet():
    """Slots, access and DLAB reachability of the twelve registers and their field tables (blueprint
    section 1.1), pinned from PC16550D SNLS378C Table 1 (pp.17-18), Table 2 (p.18), sections
    8.6.2-8.6.9 (pp.20-24) and Table 5 (p.21: the IIR ladder). The same map in TL16C550C SLLS177I
    Tables 7-1 and 7-3 (pp.21-23), TL16C550D SLLS597E Table 1 (p.32), ST16C550 Table 4 (p.13),
    Core16550 Table 1-1 (p.5), PG143 pp.18-19 and DS748 Table 4 (p.9). PC16550D Table 2 prints MODEM
    Control at A2-A0 = 000; its Table 1 and every other source put MCR at slot 4, which is pinned."""
    assert [r.name for r in um.UART16550] == [
        "RBR", "THR", "IER", "IIR", "FCR", "LCR", "MCR", "LSR", "MSR", "SCR", "DLL", "DLM",
    ]  # fmt: skip
    assert {r.name: (r.slot, r.access, r.dlab) for r in um.UART16550} == {
        "RBR": (0, "r", 0),
        "THR": (0, "w", 0),
        "IER": (1, "rw", 0),
        "IIR": (2, "r", None),
        "FCR": (2, "w", None),
        "LCR": (3, "rw", None),
        "MCR": (4, "rw", None),
        "LSR": (5, "r", None),  # writes are factory test only (PC16550D p.22)
        "MSR": (6, "r", None),
        "SCR": (7, "rw", None),
        "DLL": (0, "rw", 1),
        "DLM": (1, "rw", 1),
    }
    bits = {r.name: tuple((hi, lo, name) for hi, lo, name, _doc in r.fields) for r in um.UART16550}
    assert bits["IER"] == ((0, 0, "ERBFI"), (1, 1, "ETBEI"), (2, 2, "ELSI"), (3, 3, "EDSSI"))
    assert bits["IIR"] == ((0, 0, "INTPEND"), (3, 1, "INTID"), (7, 6, "FIFOEN"))
    assert bits["FCR"] == (
        (0, 0, "FIFOEN"), (1, 1, "RXFIFORST"), (2, 2, "TXFIFORST"), (3, 3, "DMAMODE"),
        (7, 6, "RXTRIG"),
    )  # fmt: skip
    assert bits["LCR"] == (
        (1, 0, "WLS"), (2, 2, "STB"), (3, 3, "PEN"), (4, 4, "EPS"), (5, 5, "SP"), (6, 6, "SB"),
        (7, 7, "DLAB"),
    )  # fmt: skip
    assert bits["MCR"] == (
        (0, 0, "DTR"), (1, 1, "RTS"), (2, 2, "OUT1"), (3, 3, "OUT2"), (4, 4, "LOOP"), (5, 5, "AFE"),
    )  # fmt: skip
    assert bits["LSR"] == (
        (0, 0, "DR"), (1, 1, "OE"), (2, 2, "PE"), (3, 3, "FE"), (4, 4, "BI"), (5, 5, "THRE"),
        (6, 6, "TEMT"), (7, 7, "FIFOERR"),
    )  # fmt: skip
    assert bits["MSR"] == (
        (0, 0, "DCTS"), (1, 1, "DDSR"), (2, 2, "TERI"), (3, 3, "DDCD"), (4, 4, "CTS"), (5, 5, "DSR"),
        (6, 6, "RI"), (7, 7, "DCD"),
    )  # fmt: skip
    for name in ("RBR", "THR", "SCR"):
        assert bits[name] == ((7, 0, "DATA"),)
    assert bits["DLL"] == bits["DLM"] == ((7, 0, "DIV"),)
    # MCR[5] is AFE only where the part has autoflow (TL16C550C p.23); reserved zero elsewhere
    assert um.CAPABILITY_FIELDS == {("MCR", "AFE"): "has_afe"}
    # no field leaves the byte or overlaps another of its register
    for reg in um.UART16550:
        used = 0
        for hi, lo, name, doc in reg.fields:
            assert 7 >= hi >= lo >= 0 and doc, (reg.name, name)
            mask = ((1 << (hi - lo + 1)) - 1) << lo
            assert not used & mask, (reg.name, name)
            used |= mask
    assert um.REGS == {r.name: r for r in um.UART16550}
    # the priority ladder, highest first: IIR[3:0], name, level (PC16550D Table 5, p.21)
    assert [(code, name, level) for code, name, level, _src, _clr in um.IIR_SOURCES] == [
        (0x6, "RLS", 1),
        (0x4, "RDA", 2),
        (0xC, "CTI", 2),
        (0x2, "THRE", 3),
        (0x0, "MS", 4),
    ]
    assert um.IIR_NO_INTERRUPT == 0x1
    assert [mode for mode, _doc in um.DMA_MODES] == [0, 1]  # FCR[3] (PC16550D p.6, p.22)


_OFF = re.compile(r"^#define UART_([A-Z]+)_OFF +(0x[0-9a-f]+)u ", re.M)


def test_headers_generate_deterministically_for_all_placements():
    """Same input, same bytes; the three placements put the slots at {0..7} (PC16550D A2-A0,
    pp.4-5), {0x00..0x1C step 4} (Core16550 PADDR, Table 1-1 p.5) and {0x1000..0x101C} (PG143
    pp.18-19, DS748 Table 4 p.9), each register at base_offset + slot * stride; each header is its
    golden fixture, read through the package as the wheel ships it. MCR's AFE field renders exactly
    where the placement has autoflow, and a placement whose capabilities U0's generator does not
    render is refused by name, never rendered without them."""
    want = {
        "pc16550d": set(range(8)),
        "core16550_apb": set(range(0x00, 0x20, 4)),
        "axi16550": set(range(0x1000, 0x1020, 4)),
    }
    assert set(um.PLACEMENTS) == set(want)
    for name, placement in um.PLACEMENTS.items():
        text = um.emit_header(placement)
        assert um.emit_header(dataclasses.replace(placement)) == text
        assert text.startswith(um.BANNER + "\n") and text.endswith(
            "#endif /* UART16550_REGS_H */\n"
        )
        offsets = {reg: int(value, 16) for reg, value in _OFF.findall(text)}
        assert set(offsets) == set(um.REGS), name
        assert set(offsets.values()) == want[name], name
        for reg in um.UART16550:
            assert offsets[reg.name] == placement.base_offset + reg.slot * placement.stride
        assert re.search(rf"^#define UART_REG_WIDTH +{placement.reg_width}u ", text, re.M), name
        assert um.read_fixture(name) == text.encode("utf-8"), f"golden {name} is stale"
        contract = (
            f'    "{name}", {placement.stride}u, {placement.base_offset:#x}u, '
            f"{placement.fcr_reset:#04x}u, {placement.ref_clk_hz}u\n"
        )
        assert contract in text, name
        assert (
            "    UART_RX_TRIGGER_1 = 0x00,\n    UART_RX_TRIGGER_4 = 0x40,\n"
            "    UART_RX_TRIGGER_8 = 0x80,\n    UART_RX_TRIGGER_14 = 0xc0\n"
        ) in text
        assert "UART_MCR_AFE" not in text  # no U0 placement has autoflow
    # the TL16C550C/D class (a record U0 does not register): MCR[5] is AFE
    afe = dataclasses.replace(
        um.PLACEMENTS["pc16550d"], name="tl16c550c", has_afe=True, flow_mech="mcr5_afe"
    )
    text = um.emit_header(afe)
    assert re.search(r"^#define UART_MCR_AFE_SHIFT +5u$", text, re.M)
    assert re.search(r"^#define UART_MCR_AFE_MASK +0x20u ", text, re.M)
    # the 16750 family's capabilities are U8's to render: refused by name
    pc = um.PLACEMENTS["pc16550d"]
    for change, named in (
        ({"rx_triggers_alt": (1, 16, 32, 56), "mode64_key": "dlab"}, "rx_triggers_alt, mode64_key"),
        ({"tx_triggers": (16, 32, 64, 120), "fifo_depth": 128}, "tx_triggers"),
        ({"has_scr": False}, "has_scr"),
        ({"fcr_impl_mask": 0xC6}, "fcr_impl_mask"),
        ({"has_tlr": True}, "has_tlr"),
        ({"has_frac_divisor": True, "has_sleep": True}, "has_frac_divisor, has_sleep"),
        ({"flow_mech": "efr_tcr"}, "flow_mech"),
    ):
        try:
            um.emit_header(dataclasses.replace(pc, **change))
        except ValueError as exc:
            assert str(exc) == (
                f"placement 'pc16550d' sets {named}, which U0's generator does not render "
                "(the 16750-family rendering is slice U8)"
            ), str(exc)
        else:
            raise AssertionError(f"rendered a placement setting {sorted(change)}")


# A probe of the header: the contract literal read as data, and LSR.THRE read through a register
# offset (the MMIO function last, so it is the entry the parity summary counts).
_PROBE = (
    '#include "uart16550_regs.h"\n'
    "uint32_t uart_probe_stride(void) { return uart_regmap_contract.stride; }\n"
    "uint32_t uart_probe_thre(volatile uint8_t *base)\n"
    "{ return ((uint32_t)base[UART_LSR_OFF] & UART_LSR_THRE_MASK) >> UART_LSR_THRE_SHIFT; }\n"
)


def test_generated_header_compiles_through_both_rails():
    """Each placement's header and the probe compile clean through the Python front end
    (`compile_unit`): the register read is an MMIO-domain load at the placement's LSR offset, and the
    contract literal is a global with the placement's values. With a C compiler, the C twin
    (`bcir-cc`) compiles the same two files clean and prints the oracle's parity summary line --
    counts and structural digest, which moves with the offsets -- and the host compiler accepts the
    header with -Wall -Wextra -Wpedantic -Werror."""
    from bcir.tests.test_c_cfront import _summary_line  # noqa: PLC0415

    oracle = {}
    for name, placement in um.PLACEMENTS.items():
        header = um.emit_header(placement)
        res = compile_unit(_PROBE, includes={"uart16550_regs.h": header}, check_clang=False)
        assert res.is_clean, (name, res.diagnostics)
        assert res.abi_diagnostics == [] and res.lowering_diagnostics == [], name
        claims = res.lowered.functions["uart_probe_thre"].claims
        consts = {c.wr[0]: c.imm for c in claims if c.op == "c.const"}
        loads = [c for c in claims if c.op == "c.load"]
        assert [(c.domain, consts[c.rd[1]]) for c in loads] == [
            (Domain.MMIO, (um.reg_offset(placement, um.REGS["LSR"]),))
        ], name
        literal = {g[0]: g[2] for g in res.lowered.globals_decl}["uart_regmap_contract"]
        assert literal == (
            f'{{"{name}", {placement.stride}, {placement.base_offset}, {placement.fcr_reset}, '
            f"{placement.ref_clk_hz}}}",
        ), literal
        oracle[name] = (header, _summary_line(res))
    assert len({line for _h, line in oracle.values()}) == 3  # the digest tells the placements apart
    if not _CC:
        return  # the quick tier hides the compiler
    if not os.path.isdir(os.path.join(_ROOT, "runtime", "c")):
        from bcir.tests.run_all import _is_source_checkout  # noqa: PLC0415

        assert not _is_source_checkout(), "runtime/c is missing from the checkout"
        return  # the wheel ships no runtime/c to build the twin from
    from bcir.tests.test_c_cfront import _build_bcir_cc  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as d:
        bcir_cc = _build_bcir_cc(d)
        for name, (header, line) in oracle.items():
            unit = os.path.join(d, name)
            os.makedirs(unit)
            with open(
                os.path.join(unit, "uart16550_regs.h"), "w", encoding="utf-8", newline="\n"
            ) as fh:
                fh.write(header)
            probe = os.path.join(unit, "probe.c")
            with open(probe, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(_PROBE)
            twin = subprocess.run([bcir_cc, probe], capture_output=True, text=True, timeout=120)
            assert twin.returncode == 0, (name, twin.stdout, twin.stderr)
            assert twin.stdout.splitlines()[0] == f"{probe}: {line}", (name, twin.stdout[:300])
            host = subprocess.run(
                [
                    _CC,
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Wpedantic",
                    "-Werror",
                    "-fsyntax-only",
                    probe,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert host.returncode == 0, (name, host.stderr)


def test_a_tampered_regmap_contract_is_caught():
    """The RegMap law (the R12 call-ABI contract pattern for the register map): every placement's
    own contract is clean, and each lie -- `dataclasses.replace` of the stride, the base offset, one
    register's offset, DLAB or access, the register count, the placement, a name the registry does
    not have, a forged record of a registry name, a type that only compares equal -- is caught with
    its own message, pinned whole."""
    for placement in um.PLACEMENTS.values():
        assert um.verify_regmap_contract(um.regmap_contract_for(placement), placement) == []
    pc, axi = um.PLACEMENTS["pc16550d"], um.PLACEMENTS["axi16550"]
    clean = um.regmap_contract_for(pc)
    assert clean == um.RegMapContract(
        placement="pc16550d",
        stride=1,
        base_offset=0,
        regs=(
            ("RBR", 0, "r", 0), ("THR", 0, "w", 0), ("IER", 1, "rw", 0), ("IIR", 2, "r", None),
            ("FCR", 2, "w", None), ("LCR", 3, "rw", None), ("MCR", 4, "rw", None),
            ("LSR", 5, "r", None), ("MSR", 6, "r", None), ("SCR", 7, "rw", None),
            ("DLL", 0, "rw", 1), ("DLM", 1, "rw", 1),
        ),
    )  # fmt: skip
    assert [r[1] for r in um.regmap_contract_for(axi).regs] == [
        0x1000, 0x1000, 0x1004, 0x1008, 0x1008, 0x100C, 0x1010, 0x1014, 0x1018, 0x101C, 0x1000,
        0x1004,
    ]  # fmt: skip

    def regs_with(name, **change):
        index = [r[0] for r in clean.regs].index(name)
        row = dict(zip(("name", "offset", "access", "dlab"), clean.regs[index]), **change)
        return clean.regs[:index] + (tuple(row.values()),) + clean.regs[index + 1 :]

    verify = um.verify_regmap_contract
    assert verify(dataclasses.replace(clean, stride=4), pc) == [
        "stride: contract 0x4 != placement 0x1"
    ]
    assert verify(dataclasses.replace(clean, base_offset=0x1000), pc) == [
        "base offset: contract 0x1000 != placement 0x0"
    ]
    assert verify(dataclasses.replace(clean, regs=regs_with("LSR", offset=6)), pc) == [
        "register LSR: contract ('LSR', 6, 'r', None) != laid out ('LSR', 5, 'r', None)"
    ]
    assert verify(dataclasses.replace(clean, regs=regs_with("DLL", dlab=0)), pc) == [
        "register DLL: contract ('DLL', 0, 'rw', 0) != laid out ('DLL', 0, 'rw', 1)"
    ]
    assert verify(dataclasses.replace(clean, regs=regs_with("FCR", access="rw")), pc) == [
        "register FCR: contract ('FCR', 2, 'rw', None) != laid out ('FCR', 2, 'w', None)"
    ]
    assert verify(dataclasses.replace(clean, regs=clean.regs[:-1]), pc) == [
        "registers: contract lays out 11; the model has 12"
    ]
    assert verify(um.regmap_contract_for(axi), pc) == [
        "contract placement 'axi16550' != the unit placement 'pc16550d'"
    ]
    assert verify(dataclasses.replace(clean, placement="pc16550x"), pc) == [
        "unknown placement 'pc16550x' (registry: axi16550, core16550_apb, pc16550d)"
    ]
    forged = dataclasses.replace(pc, stride=4, reg_width=4)
    assert verify(dataclasses.replace(clean, stride=4), forged) == [
        "placement record 'pc16550d' is not the registry's"
    ]
    # `True == 1` in Python; a contract is held to its types, not to what compares equal
    assert verify(dataclasses.replace(clean, stride=True), pc) == [
        "stride: contract True != placement 0x1"
    ]
    assert verify(dataclasses.replace(clean, regs=regs_with("LSR", dlab=False)), pc) == [
        "register LSR: contract ('LSR', 5, 'r', False) != laid out ('LSR', 5, 'r', None)"
    ]
    # total over what is not a contract or a placement at all
    assert verify(None, pc) == ["not a RegMapContract: NoneType"]
    assert verify(clean, "pc16550d") == ["not a UartPlacement: str"]
    assert verify(dataclasses.replace(clean, regs=None), pc) == [
        "registers: contract lays out None; the model has 12"
    ]


def test_checked_in_header_is_current():
    """The drift guard (the `gen_status --check` pattern): `runtime/c/uart16550_regs.h` is a fresh
    emission for the default placement, compared as bytes so a CRLF checkout cannot pass through
    universal newlines, its banner first; the module's own `--check` agrees."""
    path = os.path.join(_ROOT, "runtime", "c", "uart16550_regs.h")
    if not os.path.isfile(path):
        from bcir.tests.run_all import _is_source_checkout  # noqa: PLC0415

        assert not _is_source_checkout(), "runtime/c/uart16550_regs.h is missing from the checkout"
        return  # the wheel ships no runtime/c
    with open(path, "rb") as fh:
        data = fh.read()
    assert um.DEFAULT_PLACEMENT == "pc16550d"
    assert data == um.emit_header(um.PLACEMENTS["pc16550d"]).encode("utf-8"), (
        "runtime/c/uart16550_regs.h is stale: python3 -m bcir.frontends.devices.uart_model --write"
    )
    assert data.startswith(b"/* GENERATED by uart_model.py -- do not edit */\n")
    assert um.main(["--check"]) == 0


# --- the facts against the documents -----------------------------------------------------------------


def test_the_placements_match_the_documents():
    """The three U0 placements (blueprint section 2), every field against its page:

    * pc16550d -- 8-bit bus on A2-A0 (PC16550D pp.4-5), FCR reset 00 (Table 3 p.19), MCR[7:5]
      permanently 0 (p.24), modem pins (pp.5-6), INTR (p.5), the 1.8432-MHz crystal of Table 4 (p.19).
    * core16550_apb -- PADDR 00..1C (Core16550 Table 1-1 p.5), FCR reset 01 (p.5, Table 1-4 p.6),
      MCR[7:4] reserved (Table 1-11 p.9), modem pins (Table 4-1 pp.13-14), the 18.432-MHz reference
      of Table 1-6 (pp.6-7); its 8-bit data ports sit in the low byte of a 4-byte APB lane.
    * axi16550 -- 0x1000..0x101C, 32-bit registers (PG143 pp.18-19), FCR reset 0 (PG143 pp.27-28,
      DS748 Table 9 p.13), MCR[7:5] always 000 (PG143 p.34), level-sensitive IP2INTC_Irpt (DS748
      pp.4-5), S_AXI_ACLK at the 100-MHz default (DS748 G2 p.7, notes p.8).

    FCR gating holds on all three: "This bit must be a 1 when other FCR bits are written to or they
    will not be programmed" is PC16550D p.22's own sentence, repeated by TL16C550C p.24, TL16C550D
    p.33, ST16C550 p.15 and Core16550 p.6, and PG143/DS748 defer to the PC16550D data sheet (the
    blueprint's section 2 matrix said "n/a" for the silicon). FCR[5:4] are reserved on all three
    (PC16550D p.22, Core16550 p.6, PG143 p.27): `fcr_impl_mask` 0xCF. Every U8 capability field takes
    its default, and no U0 placement claims an erratum."""
    assert list(um.PLACEMENTS) == ["pc16550d", "core16550_apb", "axi16550"]

    def facts(p):
        return (
            p.stride, p.base_offset, p.reg_width, p.fcr_reset, p.fcr_gated, p.has_afe,
            p.has_modem, p.ref_clk_hz, p.irq_level,
        )  # fmt: skip

    assert facts(um.PLACEMENTS["pc16550d"]) == (1, 0, 1, 0x00, True, False, True, 1_843_200, False)
    assert facts(um.PLACEMENTS["core16550_apb"]) == (
        4,
        0,
        4,
        0x01,
        True,
        False,
        True,
        18_432_000,
        False,
    )
    assert facts(um.PLACEMENTS["axi16550"]) == (
        4,
        0x1000,
        4,
        0x00,
        True,
        False,
        True,
        100_000_000,
        True,
    )
    defaults = {
        f.name: f.default
        for f in dataclasses.fields(um.UartPlacement)
        if f.default is not dataclasses.MISSING
    }
    assert defaults == {
        "fifo_depth": 16,
        "rx_triggers": (1, 4, 8, 14),
        "rx_triggers_alt": (),
        "tx_triggers": (),
        "mode64_key": "",
        "flow_mech": "none",
        "has_tlr": False,
        "has_frac_divisor": False,
        "has_sleep": False,
        "has_scr": True,
        "fcr_impl_mask": 0xCF,
        "reset_honest": True,
        "errata": (),
    }
    for name, placement in um.PLACEMENTS.items():
        assert placement.name == name
        assert {k: getattr(placement, k) for k in defaults} == defaults, name
        assert um.placement_problems(placement) == [], name
    # IIR[7:6] reads 11 whenever FCR[0] is set, so Core16550's IIR resets to C1 (Table 1-1 p.5):
    # the FCR reset is the one fact the placements differ in at reset
    assert {n: p.fcr_reset & 1 for n, p in um.PLACEMENTS.items()} == {
        "pc16550d": 0,
        "core16550_apb": 1,
        "axi16550": 0,
    }


def test_baud_divisors_match_the_documents_worked_examples():
    """`achieved_baud` and `baud_error` against every worked example the sources print:

    * PG143 p.52 and DS748 p.19 (Example 1): 100 MHz, 56 kbaud, divisor 0x6F -- 56306 baud, +0.55%.
      The vendors' formula truncates 111.6 to 111, where 112 would be nearer.
    * PG143 p.53 and DS748 p.20 (Example 2): a 1.8432-MHz xin, 56 kbaud, divisor 2 -- 57600, +2.86%.
    * PC16550D Table 4 (p.19), its printed errors unsigned: every printed divisor is the nearest one
      but one -- 920 for 1200 baud at 18.432 MHz, where the formula gives exactly 960 -- and every
      printed error rounds from the formula but one: 0.628 for 3600 baud at 3.072 MHz (0.6289%).
    * Core16550 Table 1-6 (pp.6-7): it copies the 920 with the +4.3478% that divisor does give, and
      prints the 110 and 134.5 baud rows' "percent error" as the deviation in baud times 100.
    """
    aclk = 100_000_000
    assert um.achieved_baud(aclk, 0x6F) == Fraction(aclk, 16 * 111)
    assert round(um.achieved_baud(aclk, 0x6F)) == 56306
    assert _printed_as(um.baud_error(aclk, 0x6F, 56000) * 100, "0.55")
    assert int(Fraction(aclk, 16 * 56000)) == 0x6F and round(Fraction(aclk, 16 * 56000)) == 112
    assert abs(um.baud_error(aclk, 112, 56000)) < abs(um.baud_error(aclk, 111, 56000))
    assert um.achieved_baud(1_843_200, 2) == 57600
    assert _printed_as(um.baud_error(1_843_200, 2, 56000) * 100, "2.86")
    # PC16550D Table 4: (crystal, baud, divisor, |error| printed)
    for clock, baud, divisor, printed in (
        (1_843_200, 50, 2304, "0"),
        (1_843_200, 110, 1047, "0.026"),
        (1_843_200, Fraction(269, 2), 857, "0.058"),
        (1_843_200, 300, 384, "0"),
        (1_843_200, 2000, 58, "0.69"),
        (1_843_200, 9600, 12, "0"),
        (1_843_200, 38400, 3, "0"),
        (1_843_200, 56000, 2, "2.86"),
        (3_072_000, 110, 1745, "0.026"),
        (3_072_000, Fraction(269, 2), 1428, "0.034"),
        (3_072_000, 1800, 107, "0.312"),
        (3_072_000, 7200, 27, "1.23"),
        (18_432_000, 9600, 120, "0"),
        (18_432_000, 56000, 21, "2.04"),
        (18_432_000, 128000, 9, "0"),
    ):
        assert round(Fraction(clock, 16) / baud) == divisor, (clock, baud)
        assert _printed_as(abs(um.baud_error(clock, divisor, baud)) * 100, printed), (clock, baud)
    assert Fraction(18_432_000, 16 * 1200) == 960 and um.baud_error(18_432_000, 960, 1200) == 0
    assert not _printed_as(abs(um.baud_error(3_072_000, 53, 3600)) * 100, "0.628")
    assert _printed_as(um.baud_error(3_072_000, 53, 3600) * 100, "0.629")
    # Core16550 Table 1-6 (18.432 MHz), signed
    assert _printed_as(um.baud_error(18_432_000, 920, 1200) * 100, "4.3478")
    assert _printed_as(um.baud_error(18_432_000, 21, 56000) * 100, "-2.0408")
    for baud, divisor, printed in ((110, 10473, "-0.2865"), (Fraction(269, 2), 8565, "0.0876")):
        assert _printed_as((um.achieved_baud(18_432_000, divisor) - baud) * 100, printed)
        assert not _printed_as(um.baud_error(18_432_000, divisor, baud) * 100, printed)
    assert um.BAUD_ERROR_LIMIT == Fraction(3, 100)
    # divisor 0 ("not recommended", PC16550D p.17), past the 16-bit latch, or a float: refused
    for clock, divisor in (
        (1_843_200, 0),
        (1_843_200, -1),
        (1_843_200, 0x10000),
        (0, 1),
        (1_843_200, 1.0),
    ):
        try:
            um.achieved_baud(clock, divisor)
        except ValueError:
            continue
        raise AssertionError(f"achieved_baud({clock!r}, {divisor!r}) was accepted")
    try:
        um.baud_error(1_843_200, 12, 9600.0)
    except ValueError:
        pass
    else:
        raise AssertionError("a float baud was accepted")


# TI SLLZ058A pp.2-3 as printed: table -> (clock as labelled, rows); a row is (baud, divisor, the
# (rate, error %) cell at divisor - 1, the cell at divisor - 2), "N/A" where the decremented divisor
# is 0, None where the table has no cell.
_SLLZ058A_PRINTED = {
    1: (
        1_843_200,
        (
            (1200, 96, ("1212.632", "1.05"), ("1225.532", "2.13")),
            (2400, 48, ("2451.064", "2.13"), ("2504.348", "4.35")),
            (4800, 24, ("5008.696", "4.35"), ("5236.364", "9.09")),
            (9600, 12, ("10472.727", "9.09"), ("11520", "20")),
            (19200, 6, ("23040", "20"), ("28800", "50")),
            (38400, 3, ("57600", "50"), ("115200", "200")),
            (57600, 2, ("115200", "100"), "N/A"),
            (115200, 1, "N/A", None),
        ),
    ),
    2: (
        3_686_400,
        (
            (1200, 192, ("1206.283", "0.52"), ("12126.316", "910.53")),
            (2400, 96, ("2425.263", "1.05"), ("24510.638", "921.28")),
            (4800, 48, ("4902.128", "2.13"), ("50086.957", "943.48")),
            (9600, 24, ("10017.391", "4.35"), ("104727.273", "990.91")),
            (19200, 12, ("20945.455", "9.09"), ("230400", "1100")),
            (38400, 6, ("46080", "20"), ("576000", "1400")),
            (57600, 4, ("76800", "33.33"), ("1152000", "1900")),
            (115200, 2, ("230400", "100"), "N/A"),
        ),
    ),
    3: (
        14_746_000,
        (
            (1200, 768, ("1201.564", "0.13"), ("1203.133", "0.26")),
            (2400, 384, ("2406.266", "0.26"), ("2412.565", "0.52")),
            (4800, 192, ("4825.13", "0.52"), ("4850.525", "1.05")),
            (9600, 96, ("9701.05", "1.05"), ("9804.25", "2.13")),
            (19200, 48, ("19608.499", "2.13"), ("20034.759", "4.35")),
            (38400, 24, ("40069.518", "4.35"), ("41890.806", "9.09")),
            (57600, 16, ("61439.889", "6.67"), ("65828.316", "14.29")),
            (115200, 8, ("131656.633", "14.29"), ("153598.611", "33.33")),
        ),
    ),
    4: (
        16_000_000,
        (
            (1200, 833, ("1201.442", "0.12"), ("1202.887", "0.24")),
            (2400, 417, ("2405.774", "0.24"), ("2411.576", "0.48")),
            (4800, 208, ("4823.151", "0.48"), ("4846.527", "0.97")),
            (9600, 104, ("9693.053", "0.97"), ("9787.928", "1.96")),
            (19200, 52, ("19575.856", "1.96"), ("19966.722", "3.99")),
            (38400, 26, ("39933.444", "3.99"), ("41594.454", "8.32")),
            (57600, 17, ("61120.543", "6.11"), ("65099.458", "13.02")),
            (115200, 9, ("130198.915", "13.02"), ("149688.15", "29.94")),
        ),
    ),
}


def test_the_sllz058a_erratum_is_recorded_with_its_misprints_as_provenance():
    """TI SLLZ058A (October 2009, revised October 2010) as registry data: TL16C752C, TL16C754C and
    TL16C2752 take framing errors on a stream of characters with short STOP bits; 1.5 or 2 stop bits,
    or the divisor decreased by 1; Moderate (p.1).

    The witness recomputes the erratum's four tables (pp.2-3) from baud = clock / (16 x divisor)
    through `achieved_baud` -- every true row pinned at the document's own precision -- and the cells
    that disagree are exactly the cells the erratum's provenance notes cover. Each note's reading is
    checked, not trusted: Table 2's DIVISOR - 2 rates are ten times the true rate, with the error
    computed from the misprint; Tables 3 and 4 decrement the unrounded divisor clock/(16 x baud) - k
    rather than the integer divisor they print; and the bit periods on p.1 are microseconds printed
    as milliseconds. The divisor-0 cells are N/A, as `achieved_baud` refuses a divisor of 0."""
    erratum = um.ERRATA["SLLZ058A"]
    assert list(um.ERRATA) == ["SLLZ058A"] and erratum.id == "SLLZ058A"
    assert erratum.parts == ("TL16C752C", "TL16C754C", "TL16C2752")
    assert erratum.severity == "Moderate"
    assert erratum.workarounds == (
        "program 1.5 or 2 stop bits",
        "decrease the reference clock divisor by 1, so the received STOP bit appears longer",
    )
    assert erratum.source == "TI SLLZ058A, October 2009, revised October 2010"
    assert erratum.condition == (
        "receiving a stream of characters whose STOP bits are shorter than a bit period: after the "
        "STOP bit is sampled and verified the receiver waits about a bit period before it looks for "
        "the next START bit, so a START bit can be missed and a framing error taken; the shortfall "
        "is cumulative, so it can follow a few or many characters"
    )
    assert (
        erratum.revisions
        == "the parts as named; the erratum gives no silicon revision or date code"
    )
    units, tenfold, unrounded = erratum.provenance
    assert not set(tenfold.cells) & set(unrounded.cells)
    discrepant, true_cells, cell_of = set(), 0, {}
    worst = {3: Fraction(0), 4: Fraction(0)}
    for table, (clock, rows) in _SLLZ058A_PRINTED.items():
        for baud, divisor, *cells in rows:
            assert divisor == round(Fraction(clock, 16 * baud)), (table, baud)
            for k, cell in zip((1, 2), cells):
                key, d = (table, baud, k), divisor - k
                if cell is None:
                    assert d < 0, key
                    continue
                if cell == "N/A":
                    assert d == 0, key
                    try:
                        um.achieved_baud(clock, d)
                    except ValueError:
                        continue
                    raise AssertionError("a divisor of 0 was given a rate")
                rate, error = cell_of[key] = cell
                true = um.achieved_baud(clock, d)
                if _printed_as(true, rate) and _printed_as((true / baud - 1) * 100, error):
                    true_cells += 1
                    continue
                discrepant.add(key)
                if key in tenfold.cells:
                    assert _printed_as(10 * true, rate), key
                    assert _printed_as((Fraction(rate) / baud - 1) * 100, error), key
                else:
                    unround = Fraction(clock, 16) / (Fraction(clock, 16 * baud) - k)
                    assert _printed_as(unround, rate), key
                    assert _printed_as((unround / baud - 1) * 100, error), key
                    assert not _printed_as(true, rate), key
                    worst[table] = max(worst[table], abs(Fraction(rate) / true - 1))
    assert discrepant == set(tenfold.cells) | set(unrounded.cells)
    assert true_cells == 13 + 8  # Table 1 whole; Table 2 but its DIVISOR - 2 column
    assert len(tenfold.cells) == 7 and len(unrounded.cells) == 32
    # each note's example and claims, from the transcription and the formula
    rate, error = cell_of[(2, 1200, 2)]
    assert tenfold.printed == f"1200 baud, divisor 190: {rate} baud, {error}%"
    true = um.achieved_baud(3_686_400, 190)
    assert tenfold.corrected == f"{_fmt(true, 3)} baud, {_fmt((true / 1200 - 1) * 100, 2)}%"
    rate, error = cell_of[(4, 115200, 1)]
    assert unrounded.printed == f"Table 4, 115200 baud, divisor 8: {rate} baud, {error}%"
    true = um.achieved_baud(16_000_000, 8)
    assert unrounded.corrected == f"{_fmt(true, 3)} baud, {_fmt((true / 115200 - 1) * 100, 2)}%"
    assert worst[3] < Fraction(40, 10**6) and _fmt(worst[4] * 100, 1) == "4.8"
    assert "under 40 ppm" in unrounded.note and "up to 4.8%" in unrounded.note
    bit_9600 = Fraction(10**6, 9600)  # microseconds
    bit_div11 = 1 / (um.achieved_baud(1_843_200, 11) / 10**6)
    assert units.printed == "104 ms (also 98 ms and 96-ms)" and units.cells == ()
    assert units.corrected == (
        f"{_fmt(bit_9600, 2)} us at 9600 baud; {_fmt(bit_div11, 1)} us at divisor 11 from 1.8432 MHz"
    )
    # registry data, non-disturbing: no U0 placement claims an erratum, and a claim changes no header
    assert all(p.errata == () for p in um.PLACEMENTS.values())
    claimed = dataclasses.replace(um.PLACEMENTS["pc16550d"], errata=("SLLZ058A",))
    assert um.placement_problems(claimed) == []
    assert um.emit_header(claimed) == um.emit_header(um.PLACEMENTS["pc16550d"])


def test_a_malformed_placement_is_refused_by_name():
    """`placement_problems` is the record's law, and total: each malformation below is one finding,
    named, and the views refuse a malformed record rather than lay it out."""
    pc = um.PLACEMENTS["pc16550d"]
    cases = (
        ({"name": "PC16550D"}, "placement name 'PC16550D' is not a lower-case C identifier"),
        ({"stride": True}, "stride True is not an integer"),
        ({"has_afe": 1}, "has_afe 1 is not a bool"),
        ({"stride": 3, "reg_width": 1}, "stride 3 is not one of (1, 2, 4, 8)"),
        ({"reg_width": 4}, "reg_width 4 does not fit in stride 1"),
        ({"stride": 4, "reg_width": 4, "base_offset": 0x1002}, "base_offset 0x1002 is not a non-negative multiple of the stride"),
        ({"fcr_impl_mask": 0x1CF}, "fcr_impl_mask 0x1cf is not a byte"),
        ({"fcr_reset": 0x10}, "fcr_reset 0x10 sets a bit outside fcr_impl_mask 0xcf"),
        ({"ref_clk_hz": 0}, "ref_clk_hz 0 is not positive"),
        ({"rx_triggers": (1, 4, 14, 8)}, "rx_triggers (1, 4, 14, 8) is not four increasing positive byte values, at most 16"),
        ({"rx_triggers": (1, 4, 8, 56)}, "rx_triggers (1, 4, 8, 56) is not four increasing positive byte values, at most 16"),
        ({"tx_triggers": (16, 32)}, "tx_triggers (16, 32) is not four increasing positive byte values, at most 16"),
        ({"mode64_key": "lcr"}, "mode64_key 'lcr' is not one of ('', 'dlab')"),
        ({"has_afe": True}, "flow_mech 'none' disagrees with has_afe=True"),
        ({"errata": ("SLLZ999",)}, "errata ['SLLZ999'] are not in ERRATA"),
        ({"errata": ("SLLZ058A", "SLLZ058A")}, "errata ('SLLZ058A', 'SLLZ058A') is not a tuple of distinct erratum ids"),
    )  # fmt: skip
    for change, message in cases:
        record = dataclasses.replace(pc, **change)
        assert um.placement_problems(record) == [message], (change, um.placement_problems(record))
        for view in (um.emit_header, um.regmap_contract_for):
            try:
                view(record)
            except ValueError as exc:
                assert str(exc) == f"malformed placement: {message}", str(exc)
            else:
                raise AssertionError(f"{view.__name__} laid out {change}")
    assert um.placement_problems("pc16550d") == ["not a UartPlacement: str"]
    assert um.verify_regmap_contract(um.regmap_contract_for(pc), pc) == []


def test_the_regeneration_command_reports_and_repairs_drift():
    """`python3 -m bcir.frontends.devices.uart_model --check` exits 1 naming each stale view and 0
    once `--write` has run, on a scratch root; outside a checkout both exit 2, as does a bad
    argument -- every exit is a verdict."""
    import contextlib  # noqa: PLC0415
    import io  # noqa: PLC0415

    def run(*args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = um.main(list(args))
        return code, out.getvalue().splitlines()

    views = sorted(um.generated_files())
    assert views == sorted(
        [os.path.join(um.PACKAGE_DIR, um.FIXTURES, um.fixture_name(n)) for n in um.PLACEMENTS]
        + [um.RUNTIME_HEADER]
    )
    stale = [
        f"stale: {rel} (python3 -m bcir.frontends.devices.uart_model --write)" for rel in views
    ]
    original = um._repo_root
    with tempfile.TemporaryDirectory() as root:
        um._repo_root = lambda: root
        try:
            assert run("--check") == (
                2,
                [f"not a source checkout ({root} has no runtime/c): nothing to check"],
            )
            assert run("--write")[0] == 2
            os.makedirs(os.path.join(root, "runtime", "c"))
            os.makedirs(os.path.join(root, um.PACKAGE_DIR, um.FIXTURES))
            code, lines = run("--check")
            assert (code, sorted(lines)) == (1, stale)
            code, lines = run("--write")
            assert (code, sorted(lines)) == (0, [f"wrote {rel}" for rel in views])
            assert run("--check") == (0, [])
            for rel, data in um.generated_files().items():
                with open(os.path.join(root, rel), "rb") as fh:
                    assert fh.read() == data, rel
            with open(os.path.join(root, um.RUNTIME_HEADER), "ab") as fh:
                fh.write(b"\r\n")  # one stray line ending is drift, bytes not text
            assert run("--check") == (1, [stale[views.index(um.RUNTIME_HEADER)]])
            assert run("--write") == (0, [f"wrote {um.RUNTIME_HEADER}"])
            assert run("--check") == (0, [])
            usage = ["usage: python3 -m bcir.frontends.devices.uart_model --check | --write"]
            assert run() == (2, usage) and run("--check", "--write") == (2, usage)
        finally:
            um._repo_root = original


def test_the_golden_headers_ship_in_the_wheel():
    """The tests read the golden headers through the package (`read_fixture`), so a wheel without
    them fails rather than skips; in a checkout, pyproject's package-data must say so too (L21)."""
    for name in um.PLACEMENTS:
        assert um.read_fixture(name).startswith(um.BANNER.encode("utf-8") + b"\n"), name
    pyproject = os.path.join(_ROOT, "pyproject.toml")
    if not os.path.isfile(pyproject):
        return  # an installed tree: the read above is the witness
    import tomllib  # noqa: PLC0415

    with open(pyproject, "rb") as fh:
        package_data = tomllib.load(fh)["tool"]["setuptools"]["package-data"]
    assert package_data.get("bcir.frontends.devices") == ["fixtures/*.h"], package_data
    on_disk = sorted(os.listdir(os.path.join(_ROOT, um.PACKAGE_DIR, um.FIXTURES)))
    assert on_disk == sorted(um.fixture_name(n) for n in um.PLACEMENTS), on_disk


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
