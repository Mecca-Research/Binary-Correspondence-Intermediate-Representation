"""Blueprint slice U1: the UART protocol laws on the oracle rail (`bcir/frontends/devices/uart_laws.py`).

The positive is a driver written to section 1.4's canonical bring-up (the U2 driver does not exist
yet): it compiles law-clean on every placement. Every negative is that driver with one defect put
in, and asserts the whole list of messages it draws -- the law named, nothing else. The laws are
advisory and vacuous off-UART: the corpus test compiles every cfront fixture and finds nothing.
"""

import dataclasses

from bcir.frontends.cfront import compile_unit
from bcir.frontends.devices import uart_laws as ul
from bcir.frontends.devices import uart_model as um

# Section 1.4, in order: wait for TEMT, open the DLAB bracket with the format, DLL then DLM, close
# it, FCR with bit 0 in the same write, MCR, IER last, then one read each of LSR, RBR, IIR, MSR.
_DRIVER = r"""#include "uart16550_regs.h"
uint32_t uart_init(volatile uint8_t *u, uint32_t divisor, uint32_t lcr_format, uint32_t trigger)
{
    uint32_t spins = 0u;
    while ((u[UART_LSR_OFF] & UART_LSR_TEMT_MASK) == 0u) {
        spins = spins + 1u;
        if (spins > 1000u) {
            return 1u;
        }
    }
    u[UART_LCR_OFF] = (lcr_format & 0x7fu) | UART_LCR_DLAB_MASK;
    u[UART_DLL_OFF] = divisor & 0xffu;
    u[UART_DLM_OFF] = (divisor >> 8) & 0xffu;
    u[UART_LCR_OFF] = lcr_format & 0x7fu;
    u[UART_FCR_OFF] = UART_FCR_FIFOEN_MASK | UART_FCR_RXFIFORST_MASK | UART_FCR_TXFIFORST_MASK
                      | (trigger & UART_FCR_RXTRIG_MASK);
    u[UART_MCR_OFF] = UART_MCR_DTR_MASK | UART_MCR_RTS_MASK | UART_MCR_OUT2_MASK;
    u[UART_IER_OFF] = 0u;
    return (uint32_t)u[UART_LSR_OFF] + u[UART_RBR_OFF] + u[UART_IIR_OFF] + u[UART_MSR_OFF];
}
uint32_t uart_tx(volatile uint8_t *u, const uint8_t *buf, uint32_t len)
{
    uint32_t sent = 0u;
    while (sent < len) {
        if ((u[UART_LSR_OFF] & UART_LSR_THRE_MASK) != 0u) {
            uint32_t burst = 0u;
            while (burst < 16u && sent < len) {
                u[UART_THR_OFF] = buf[sent];
                sent = sent + 1u;
                burst = burst + 1u;
            }
        }
    }
    return sent;
}
uint32_t uart_rx_drain(volatile uint8_t *u, uint8_t *buf, uint32_t cap)
{
    uint32_t n = 0u;
    uint32_t lsr = u[UART_LSR_OFF];
    while ((lsr & UART_LSR_DR_MASK) != 0u && n < cap) {
        buf[n] = u[UART_RBR_OFF];
        n = n + 1u;
        lsr = u[UART_LSR_OFF];
    }
    return n;
}
"""

_POLL = """    while ((u[UART_LSR_OFF] & UART_LSR_TEMT_MASK) == 0u) {
        spins = spins + 1u;
        if (spins > 1000u) {
            return 1u;
        }
    }
"""
_OPEN = "    u[UART_LCR_OFF] = (lcr_format & 0x7fu) | UART_LCR_DLAB_MASK;\n"
_DLL = "    u[UART_DLL_OFF] = divisor & 0xffu;\n"
_DLM = "    u[UART_DLM_OFF] = (divisor >> 8) & 0xffu;\n"
_CLOSE = "    u[UART_LCR_OFF] = lcr_format & 0x7fu;\n"
_FCR = """    u[UART_FCR_OFF] = UART_FCR_FIFOEN_MASK | UART_FCR_RXFIFORST_MASK | UART_FCR_TXFIFORST_MASK
                      | (trigger & UART_FCR_RXTRIG_MASK);
"""


def _with(*edits, source=_DRIVER):
    """The driver with each (old, new) edit applied; every `old` must occur exactly once."""
    for old, new in edits:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source


def _compile(source, placement="pc16550d", header=None):
    header = um.emit_header(um.PLACEMENTS[placement]) if header is None else header
    return compile_unit(source, includes={"uart16550_regs.h": header}, check_clang=False)


def _uart(source, placement="pc16550d"):
    """The R3-uart messages a unit draws (every one tagged R3)."""
    res = _compile(source, placement)
    assert all(d.law == "R3" for d in res.uart_diagnostics), res.uart_diagnostics
    return [d.message for d in res.uart_diagnostics]


def test_the_canonical_driver_is_law_clean_on_every_placement():
    """Section 1.4's bring-up, an LSR-checked TX burst and an RX drain compile clean and law-clean
    on every placement, each attesting its placement's RegMap contract. The laws are advisory: a
    unit they find fault with is still clean, and its other diagnostics are untouched."""
    for name, placement in um.PLACEMENTS.items():
        res = _compile(_DRIVER, name)
        assert res.is_clean, (name, res.diagnostics)
        assert res.uart_diagnostics == [], (name, res.uart_diagnostics)
        assert res.regmap_contract == um.regmap_contract_for(placement), name
    faulty = _compile(_with((_POLL, "")))
    assert faulty.is_clean and faulty.uart_diagnostics, faulty.uart_diagnostics
    clean = _compile(_DRIVER)
    assert (faulty.diagnostics, faulty.abi_diagnostics, faulty.lowering_diagnostics) == (
        clean.diagnostics,
        clean.abi_diagnostics,
        clean.lowering_diagnostics,
    )


def test_law1_only_dll_dlm_and_lcr_inside_a_dlab_bracket():
    """Hazard H-B: inside a DLAB=1 bracket only DLL, DLM and LCR may be touched. Slots 0 and 1 are
    DLL/DLM there by the hardware's own reading, so the registers a bracket can hit by mistake are
    the others: MCR, an IIR read (FCR read-back on the AXI core), LSR, SCR -- and an access whose
    offset the claim graph cannot fix is not provably one of the three."""
    mcr = "    u[UART_MCR_OFF] = 0u;\n"
    assert _uart(_with((_DLL, _DLL + mcr))) == [
        "uart_init: R3-uart: MCR access inside a DLAB=1 bracket"
    ]
    for line, reg in (
        ("    (void)u[UART_IIR_OFF];\n", "IIR"),
        ("    (void)u[UART_LSR_OFF];\n", "LSR"),
        ("    u[UART_SCR_OFF] = 1u;\n", "SCR"),
    ):
        for placement in um.PLACEMENTS:
            assert _uart(_with((_DLL, _DLL + line)), placement) == [
                f"uart_init: R3-uart: {reg} access inside a DLAB=1 bracket"
            ], (reg, placement)
    assert _uart(_with((_DLL, _DLL + "    u[trigger] = 0u;\n"))) == [
        "uart_init: R3-uart: unidentified register access inside a DLAB=1 bracket"
    ]
    # the same accesses after the bracket closes are ordinary register accesses
    assert _uart(_with((_CLOSE, _CLOSE + mcr + "    (void)u[UART_IIR_OFF];\n"))) == []


def test_every_declared_access_idiom_resolves_its_register():
    """The laws read a register off the claim graph for every idiom the module's scope declares,
    on the AXI placement (stride 4, base 0x1000): a byte pointer, 32-bit lanes, pointer arithmetic
    on either (which the front end lowers as an index), a local derived from either (its offset
    scaled by the pointee), a cast, a struct member, a member array element -- the index or offset
    folded from constant arithmetic where the idiom computes it. Each draws exactly the MCR
    message: its LSR read and LCR stores were identified too, or law 4 or the bracket would change
    the list."""
    head = r"""#include "uart16550_regs.h"
typedef struct axi_lanes {
    uint8_t pad[0x1000];
    volatile uint32_t rbr, ier, iir, lcr, mcr, lsr, msr, scr;
} axi_lanes_t;
typedef struct axi_array {
    uint8_t pad[0x1000];
    volatile uint32_t r[8];
} axi_array_t;
"""
    idioms = (
        ("volatile uint8_t *u", "(void)u[UART_LSR_OFF];", "u[UART_LCR_OFF]", "u[UART_MCR_OFF] = 0u;"),
        ("volatile uint32_t *u", "(void)u[UART_LSR_OFF / 4u];", "u[UART_LCR_OFF / 4u]", "u[UART_MCR_OFF / 4u] = 0u;"),
        ("volatile uint8_t *u", "(void)u[UART_LSR_OFF];", "u[UART_LCR_OFF]", "*(u + UART_MCR_OFF) = 0u;"),
        ("volatile uint32_t *u", "(void)u[UART_LSR_OFF / 4u];", "u[UART_LCR_OFF / 4u]", "*(u + UART_MCR_OFF / 4u) = 0u;"),
        ("volatile uint8_t *u", "(void)u[UART_LSR_OFF];", "u[UART_LCR_OFF]", "volatile uint8_t *m = u + UART_MCR_OFF;\n    *m = 0u;"),
        ("volatile uint32_t *u", "(void)u[UART_LSR_OFF / 4u];", "u[UART_LCR_OFF / 4u]", "volatile uint32_t *m = u + UART_MCR_OFF / 4u;\n    *m = 0u;"),
        ("volatile uint8_t *u", "(void)u[UART_LSR_OFF];", "u[UART_LCR_OFF]", "*(volatile uint32_t *)(u + UART_MCR_OFF) = 0u;"),
        ("volatile axi_lanes_t *u", "(void)u->lsr;", "u->lcr", "u->mcr = 0u;"),
        ("volatile axi_array_t *u", "(void)u->r[5];", "u->r[3]", "u->r[(UART_MCR_OFF - 0x1000u) / 4u] = 0u;"),
    )  # fmt: skip
    for param, lsr, lcr, mcr in idioms:
        source = (
            head
            + f"void f({param})\n{{\n    {lsr}\n    {lcr} = 0x83u;\n    {mcr}\n    {lcr} = 0x03u;\n}}\n"
        )
        assert _uart(source, "axi16550") == ["f: R3-uart: MCR access inside a DLAB=1 bracket"], (
            param,
            mcr,
        )


# A bracket whose one access is `@BODY@`, through pointers the resolver must or must not fix.
_UNRESOLVED = r"""#include "uart16550_regs.h"
uint32_t probe(volatile uint8_t *u, volatile uint8_t *w, uint32_t divisor)
{
    volatile uint8_t *p = @INIT@;
    volatile uint8_t *s;
    uint32_t v = 0u;
    while ((u[UART_LSR_OFF] & UART_LSR_TEMT_MASK) == 0u) {
    }
    u[UART_LCR_OFF] = 0x03u | UART_LCR_DLAB_MASK;
@BODY@    u[UART_DLL_OFF] = divisor & 0xffu;
    u[UART_DLM_OFF] = (divisor >> 8) & 0xffu;
    u[UART_LCR_OFF] = 0x03u;
@AFTER@    return v;
}
"""


def test_a_pointer_with_no_one_definition_has_no_offset():
    """The review of #823 (round 2): a local pointer assigned twice has no one definition, and the
    resolver read it as a fresh device base at offset 0 -- `p = u`, then `p = u + UART_LSR_OFF` and
    `*p` inside an open DLAB bracket was read as DLL, and drew nothing. A pointer with no one value is
    no base: its offset is unknown, so is every offset derived from it (`p + 1`, a copy), and an
    access through any of them inside a bracket is unidentified. A parameter's value at entry is a
    definition too, so a parameter the body writes has two, and the same holds; a parameter the body
    never writes is the device base, and a pointer assigned once still resolves. A pointer derived
    from itself has no value either, and the resolver ends on it. Outside a bracket an unidentified
    access is not policed (the module's declared scope)."""
    unidentified = "probe: R3-uart: unidentified register access inside a DLAB=1 bracket"
    lsr = "probe: R3-uart: LSR access inside a DLAB=1 bracket"

    def case(body, init="u", after=""):
        source = _UNRESOLVED.replace("@INIT@", init).replace("@BODY@", body)
        return _uart(source.replace("@AFTER@", after))

    reassigned = "    p = u + UART_LSR_OFF;\n"
    assert case(reassigned + "    v = *p;\n") == [unidentified]  # the review's case
    assert case(reassigned + "    s = p + 1;\n    v = *s;\n") == [unidentified]
    assert case(reassigned + "    s = p;\n    v = *s;\n") == [unidentified]
    assert case("    v = *p;\n", init="u + UART_LSR_OFF") == [lsr]  # assigned once: resolved
    assert case("    v = *p;\n") == []  # a copy of the base: DLL, which the bracket may read
    assert case("    v = *w;\n") == []  # a parameter the body never writes: the base, DLL
    assert case("    v = w[UART_LSR_OFF];\n") == [lsr]
    assert case("    v = *w;\n    w = u + UART_LSR_OFF;\n    v = v + *w;\n") == [unidentified] * 2
    assert case("    s = s + UART_LSR_OFF;\n    v = *s;\n") == [unidentified]  # derived from itself
    assert case("", after=reassigned + "    v = *p;\n") == []  # outside the bracket: not policed


def test_law1_a_dlab_bracket_is_straight_line():
    """A bracket is the straight-line run from an LCR store with bit 7 set to the next with it
    clear; a branch, a loop, a call or a return inside it -- or a bracket left open when the
    function ends -- is refused, conservatively, once."""
    message = ["uart_init: R3-uart: DLAB bracket must be straight-line"]
    assert _uart(_with((_DLM, "    if (divisor > 255u) {\n    " + _DLM + "    }\n"))) == message
    assert (
        _uart(
            _with(
                (
                    _DLM,
                    "    while (trigger > 1u) {\n        trigger = trigger - 1u;\n    }\n" + _DLM,
                )
            )
        )
        == message
    )
    latch = (
        "void uart_latch(volatile uint8_t *u)\n{\n    (void)u[UART_LSR_OFF];\n"
        "    u[UART_LCR_OFF] = 0x83u;\n    u[UART_DLL_OFF] = 1u;\n    u[UART_DLM_OFF] = 0u;\n}\n"
    )
    assert _uart(_DRIVER + latch) == ["uart_latch: R3-uart: DLAB bracket must be straight-line"]
    # never closed: every later register access is inside the bracket, then the return leaves it
    assert (
        _uart(_with((_CLOSE, "")))
        == [
            f"uart_init: R3-uart: {reg} access inside a DLAB=1 bracket"
            for reg in ("FCR", "MCR", "LSR", "IIR", "MSR")
        ]
        + message
    )
    assert (
        _uart(_with((_DLM, _DLM + "    if (trigger == 7u) {\n        return 2u;\n    }\n")))
        == message
    )
    helper = "void uart_scratch(volatile uint8_t *u)\n{\n    u[UART_SCR_OFF] = 0u;\n}\n"
    called = _with(
        ("uint32_t uart_init(", helper + "uint32_t uart_init("),
        (_DLM, _DLM + "    uart_scratch(u);\n"),
    )
    assert _uart(called) == message
    # opened inside an arm and closed after it
    assert _uart(_with((_OPEN, "    if (trigger != 0u) {\n    " + _OPEN + "    }\n"))) == message


def test_law1_every_lcr_store_decides_dlab():
    """No bracket can be read through an LCR store whose bit 7 the claim graph cannot decide."""
    assert _uart(_with((_CLOSE, "    u[UART_LCR_OFF] = lcr_format;\n"))) == [
        "uart_init: R3-uart: LCR store with an undecidable DLAB bit"
    ]
    assert _uart(_with((_OPEN, "    u[UART_LCR_OFF] = lcr_format | 0x80u;\n"))) == []


def test_law2_divisor_zero_is_forbidden():
    """A bracket that closes over constant DLL and DLM stores of 0 (PC16550D p.17: "not
    recommended"; H16750S: illegal); 0x0100 is divisor 256, not 0, and a latch the bracket leaves
    alone is not provably 0."""
    zero = (_DLL, "    u[UART_DLL_OFF] = 0u;\n"), (_DLM, "    u[UART_DLM_OFF] = 0u;\n")
    assert _uart(_with(*zero)) == ["uart_init: R3-uart: divisor 0 is forbidden"]
    assert _uart(
        _with((_DLL + _DLM, "    u[UART_DLM_OFF] = 0u;\n    u[UART_DLL_OFF] = 0u;\n"))
    ) == ["uart_init: R3-uart: divisor 0 is forbidden"]
    assert _uart(_with(zero[0], (_DLM, "    u[UART_DLM_OFF] = 1u;\n"))) == []
    assert _uart(_with(zero[0], (_DLM, ""))) == []


def test_law2_the_baud_bound_holds_a_declared_target():
    """With the unit's `uart_target_baud` and a constant divisor, the rate from the placement's
    reference clock must be within 3% (section 1.3). The worked example is the documents' own: PC16550D
    Table 4 and Core16550 Table 1-6 print 920 for 1200 baud at 18.432 MHz, +4.35%; 960 is exact.
    Without a declared target there is nothing to hold the rate to."""
    target = "static const uint32_t uart_target_baud = 1200u;\n"

    def divisor(value, declare=target):
        return _with(
            ('#include "uart16550_regs.h"\n', '#include "uart16550_regs.h"\n' + declare),
            (_DLL, f"    u[UART_DLL_OFF] = {value & 0xFF}u;\n"),
            (_DLM, f"    u[UART_DLM_OFF] = {value >> 8}u;\n"),
        )

    assert _uart(divisor(920), "core16550_apb") == [
        "uart_init: R3-uart: baud error 4.3% exceeds 3.0%"
    ]
    assert _uart(divisor(960), "core16550_apb") == []
    assert _uart(divisor(920, declare=""), "core16550_apb") == []
    # the same divisor against the PC16550D's 1.8432-MHz crystal misses 1200 baud by 90%
    assert _uart(divisor(920), "pc16550d") == ["uart_init: R3-uart: baud error 89.6% exceeds 3.0%"]
    assert _uart(divisor(96), "pc16550d") == []
    assert _uart(
        divisor(960, declare="static const uint32_t uart_target_baud = 0u;\n"), "core16550_apb"
    ) == ["R3-uart: uart_target_baud '0' is not a positive integer constant"]


def test_law3_an_fcr_write_without_fifo_enable_programs_nothing():
    """Hazard H-C: on a gated placement (all three; PC16550D p.22) an FCR write whose bit 0 is
    provably clear and that asks for anything else programs nothing. FCR = 0 is the 16450-mode
    write and programs what it says; a placement that does not gate is not held to it."""
    resets = "    u[UART_FCR_OFF] = UART_FCR_RXFIFORST_MASK | UART_FCR_TXFIFORST_MASK;\n"
    for name in um.PLACEMENTS:
        assert _uart(_with((_FCR, resets)), name) == [
            f"uart_init: R3-uart: FCR write with FIFO-enable clear programs nothing on {name}"
        ]
        assert _uart(_with((_FCR, "    u[UART_FCR_OFF] = 0u;\n")), name) == []
        assert _uart(_with((_FCR, "    u[UART_FCR_OFF] = trigger | 1u;\n")), name) == []
        assert _uart(_with((_FCR, "    u[UART_FCR_OFF] = trigger;\n")), name) == []  # not provable
    res = _compile(_with((_FCR, resets)))
    ungated = dataclasses.replace(um.PLACEMENTS["pc16550d"], fcr_gated=False)
    assert ul.verify_uart_protocol(res.lowered, ungated) == []
    assert ul.verify_uart_protocol(res.lowered, um.PLACEMENTS["pc16550d"]) == [
        "uart_init: R3-uart: FCR write with FIFO-enable clear programs nothing on pc16550d"
    ]


def test_law4_reconfiguration_waits_for_a_dominating_lsr_read():
    """An LCR, DLL or DLM store needs an LSR read before it with no THR store between (section
    1.4 step 1; the DesignWare busy-detect immunity of section 3.4) -- reported once per run. A
    read on one arm of a branch does not dominate; a read in a loop's condition does."""
    message = ["uart_init: R3-uart: line reconfiguration without a dominating LSR (TEMT) check"]
    assert _uart(_with((_POLL, ""))) == message
    assert _uart(_with((_OPEN, "    u[UART_THR_OFF] = 0u;\n" + _OPEN))) == message
    one_arm = "    if (trigger != 0u) {\n        (void)u[UART_LSR_OFF];\n    }\n"
    assert _uart(_with((_POLL, one_arm))) == message
    both = "    if (trigger != 0u) {\n        (void)u[UART_LSR_OFF];\n    } else {\n        (void)u[UART_LSR_OFF];\n    }\n"
    assert _uart(_with((_POLL, both))) == []
    assert _uart(_with((_POLL, "    (void)u[UART_LSR_OFF];\n"))) == []


def test_the_laws_follow_every_path_through_a_function():
    """Law 4 along the paths a straight-line reading misses: a loop's way back to its head (the next
    trip's LCR store follows this trip's THR store, in a while and a do/while), a `break` and a
    `continue` (each leaves with its own state), a switch (a case falls through into the next;
    without a `default` its head falls out), a `goto` target (reached from anywhere, so nothing is
    known there), a call to one of the unit's own functions (it leaves the check as the callee does)
    -- and not along the exit a constant condition rules out (a `for (;;)` poll leaves only by its
    `break`). One LSR read or a `default` repairs each; a bracket inside one case is straight-line,
    one across a case label is not."""
    reconfigure = ["f: R3-uart: line reconfiguration without a dominating LSR (TEMT) check"]
    lsr = "        (void)u[UART_LSR_OFF];\n"

    def f(body):
        return (
            '#include "uart16550_regs.h"\nvoid f(volatile uint8_t *u, uint32_t n)\n{\n'
            "    (void)u[UART_LSR_OFF];\n" + body + "}\n"
        )

    trip = (
        "        u[UART_LCR_OFF] = 0x03u;\n        u[UART_THR_OFF] = 0x55u;\n        n = n - 1u;\n"
    )
    while_back = "    while (n > 0u) {\n" + trip + "    }\n"
    do_back = "    do {\n" + trip + "    } while (n > 0u);\n"
    broken = """    while (n > 0u) {
        if (n == 1u) {
            u[UART_THR_OFF] = 0u;
            break;
        }
        (void)u[UART_LSR_OFF];
        n = n - 1u;
    }
    u[UART_LCR_OFF] = 0x03u;
"""
    continued = """    while (n > 0u) {
        n = n - 1u;
        if (n == 0u) {
            u[UART_THR_OFF] = 0u;
            continue;
        }
        (void)u[UART_LSR_OFF];
    }
    u[UART_LCR_OFF] = 0x03u;
"""
    fallthrough = """    switch (n) {
    case 1u:
        u[UART_THR_OFF] = 0u;
    case 2u:
        u[UART_LCR_OFF] = 0x03u;
        break;
    default:
        break;
    }
"""
    no_default = """    u[UART_THR_OFF] = 0u;
    switch (n) {
    case 1u:
        (void)u[UART_LSR_OFF];
        break;
    }
    u[UART_LCR_OFF] = 0x03u;
"""
    jumped = """again:
    u[UART_LCR_OFF] = 0x03u;
    u[UART_THR_OFF] = 0u;
    if (n > 0u) {
        n = n - 1u;
        goto again;
    }
"""
    for body, old, new in (
        (while_back, "    while (n > 0u) {\n", "    while (n > 0u) {\n" + lsr),
        (do_back, "    do {\n", "    do {\n" + lsr),
        (broken, "            break;\n", lsr + "            break;\n"),
        (continued, "            continue;\n", lsr + "            continue;\n"),
        (fallthrough, "    case 2u:\n", lsr + "    case 2u:\n"),
        (no_default, "    case 1u:\n", "    default:\n    case 1u:\n"),
        (jumped, "again:\n", "again:\n" + lsr),
    ):
        assert _uart(f(body)) == reconfigure, body
        assert body.count(old) == 1 and _uart(f(body.replace(old, new))) == [], (old, new)
    poll = """    u[UART_THR_OFF] = 0u;
    for (;;) {
        if ((u[UART_LSR_OFF] & UART_LSR_TEMT_MASK) != 0u) {
            break;
        }
    }
    u[UART_LCR_OFF] = 0x03u;
"""
    assert _uart(f(poll)) == []
    assert _uart(f(poll.replace("    for (;;) {\n", "    while (n > 0u) {\n"))) == reconfigure
    calls = r"""#include "uart16550_regs.h"
void uart_wait(volatile uint8_t *u)
{
    while ((u[UART_LSR_OFF] & UART_LSR_TEMT_MASK) == 0u) {
    }
}
void uart_send(volatile uint8_t *u, uint32_t c)
{
    u[UART_THR_OFF] = c;
}
void g(volatile uint8_t *u)
{
    uart_wait(u);
    u[UART_LCR_OFF] = 0x03u;
}
void h(volatile uint8_t *u)
{
    (void)u[UART_LSR_OFF];
    uart_send(u, 0x55u);
    u[UART_LCR_OFF] = 0x03u;
}
"""
    assert _uart(calls) == [
        "h: R3-uart: line reconfiguration without a dominating LSR (TEMT) check"
    ]
    bracket = """    switch (n) {
    case 1u:
        u[UART_LCR_OFF] = 0x83u;
        u[UART_DLL_OFF] = 1u;
        u[UART_DLM_OFF] = 0u;
        u[UART_LCR_OFF] = 0x03u;
        break;
    default:
        break;
    }
"""
    assert _uart(f(bracket)) == []
    across = bracket.replace("        u[UART_DLL_OFF]", "    case 2u:\n        u[UART_DLL_OFF]")
    assert _uart(f(across)) == ["f: R3-uart: DLAB bracket must be straight-line"]


def test_law5_read_effects_are_volatile_and_never_merged():
    """Every register read must be a volatile access (the blueprint's clause), and RBR, IIR, LSR and
    MSR reads clear device state, so the plan must realize every one (the planner's CSE refuses
    them categorically; this is the witness). A non-volatile member beside volatile ones is a
    device access the optimizer may merge, read-clear or not."""
    mixed = r"""#include "uart16550_regs.h"
typedef struct uart_lanes {
    volatile uint8_t rbr;
    uint8_t ier, iir, lcr, mcr, lsr, msr, scr;
} uart_lanes_t;
uint32_t uart_status(uart_lanes_t *u) { return u->lsr; }
uint32_t uart_enables(uart_lanes_t *u) { return u->ier; }
"""
    assert _uart(mixed) == [
        "uart_status: R3-uart: non-volatile access to UART register LSR",
        "uart_enables: R3-uart: non-volatile access to UART register IER",
    ]
    assert _uart(mixed.replace("    uint8_t ier,", "    volatile uint8_t ier,")) == []
    res = _compile(_DRIVER)
    claims = {c.id: c for c in res.lowered.functions["uart_rx_drain"].claims}
    lsr_reads = [
        step for step in res.plans["uart_rx_drain"].steps if claims[step.claim_id].op == "c.load"
    ]
    assert len(lsr_reads) == 3  # the drain's two LSR reads and its RBR read, each its own step
    forged = dict(res.plans)
    forged["uart_rx_drain"] = dataclasses.replace(
        res.plans["uart_rx_drain"],
        steps=[s for s in res.plans["uart_rx_drain"].steps if s is not lsr_reads[0]],
    )
    placement = um.PLACEMENTS["pc16550d"]
    assert ul.verify_uart_protocol(res.lowered, placement, plans=res.plans) == []
    assert ul.verify_uart_protocol(res.lowered, placement, plans=forged) == [
        "uart_rx_drain: R3-uart: two LSR reads merged"
    ]


def test_the_contract_literal_is_attested():
    """The unit's `uart_regmap_contract` literal is the RegMap contract's C anchor: a header edited
    to lie about its stride, its reset or its clock, to name a placement the registry does not have,
    or out of the U0 shape, is caught by the RegMap law; the honest one is recorded."""
    probe = '#include "uart16550_regs.h"\nuint32_t uart_lsr(volatile uint8_t *u) { return u[UART_LSR_OFF]; }\n'
    honest = um.emit_header(um.PLACEMENTS["pc16550d"])
    literal = '    "pc16550d", 1u, 0x0u, 0x00u, 1843200u\n'
    assert literal in honest
    res = _compile(probe, header=honest)
    assert res.regmap_contract == um.regmap_contract_for(um.PLACEMENTS["pc16550d"])
    assert res.uart_diagnostics == []
    for lie, message in (
        ('    "pc16550d", 4u, 0x0u, 0x00u, 1843200u\n', "R3-uart: regmap contract: stride: contract 0x4 != placement 0x1"),
        ('    "pc16550d", 1u, 0x1000u, 0x00u, 1843200u\n', "R3-uart: regmap contract: base offset: contract 0x1000 != placement 0x0"),
        ('    "pc16550d", 1u, 0x0u, 0x01u, 1843200u\n', "R3-uart: regmap contract: fcr_reset 0x01 != placement 0x00"),
        ('    "pc16550d", 1u, 0x0u, 0x00u, 18432000u\n', "R3-uart: regmap contract: ref_clk_hz 18432000 != placement 1843200"),
        ('    "pc16550x", 1u, 0x0u, 0x00u, 1843200u\n', "R3-uart: regmap contract: unknown placement 'pc16550x' (registry: axi16550, core16550_apb, pc16550d)"),
        ('    "PC16550D", 1u, 0x0u, 0x00u, 1843200u\n', "R3-uart: the uart_regmap_contract literal '{\"PC16550D\", 1, 0, 0, 1843200}' is not a U0 header's"),
    ):  # fmt: skip
        res = _compile(probe, header=honest.replace(literal, lie))
        assert [d.message for d in res.uart_diagnostics] == [message], (lie, res.uart_diagnostics)


def test_the_laws_are_vacuous_on_the_existing_corpus():
    """Non-disturbance, measured: every unit of the cfront corpus the oracle compiles
    (`escape_fixtures.corpus()`, the one definition of it: `runtime/c/cfront_*.c`, the CMSIS GPIO
    driver and the vendor-style UART fixture among them) draws no UART diagnostic and attests no
    contract, because none embeds the U0 literal; the canonical driver with a defect draws them."""
    from bcir.tests import escape_fixtures  # noqa: PLC0415
    from bcir.tests.run_all import _is_source_checkout  # noqa: PLC0415

    units = [(name, res) for name, _path, _src, res in escape_fixtures.corpus()]
    if not units:
        assert not _is_source_checkout(), "the cfront corpus compiled nothing in a checkout"
        return
    names = {name for name, _res in units}
    if _is_source_checkout():
        assert {"cfront_driver_gpio.c", "cfront_driver_uart.c"} <= names, sorted(names)[:5]
    for name, res in units:
        assert res.uart_diagnostics == [] and res.regmap_contract is None, (
            name,
            res.uart_diagnostics,
        )
    assert _uart(_with((_POLL, ""))) != []


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
