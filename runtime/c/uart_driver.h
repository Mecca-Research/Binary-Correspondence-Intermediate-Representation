/* uart_driver.h -- the polled 16550 driver's API (blueprint slice U2): its state, its status codes and
 * its six entry points.
 *
 * The one declaration of the driver: uart_driver.c defines what it declares and the harness
 * (uart_main.c) calls it. Each rail's --linkable emit of the driver carries its own copy of the types
 * and prototypes, and builds alone; the one function it declares with another type than this header's
 * is uart_tx, its TX buffer without the `const` (an open defect of both rails' emit, pinned by
 * bcir/tests/test_uart_driver.py; blueprint section 5, U2's notes).
 *
 * It includes the generated register map, uart16550_regs.h (the U0 registry), for the lane type
 * uart_regs_t, and nothing else: through it, that generated header is the driver's only device header.
 * Freestanding C.
 */
#ifndef UART_DRIVER_H
#define UART_DRIVER_H

#include "uart16550_regs.h"

typedef struct uart_state {        /* driver-private, RAM domain */
    uint32_t fcr_shadow;           /* H-E: FCR's persistent bits as last written -- UART_FCR_FIFOEN_MASK |
                                      UART_FCR_DMAMODE_MASK | UART_FCR_RXTRIG_MASK -- never the FIFO resets,
                                      FCR[2:1], which clear themselves (PC16550D p.22); FCR is never read back.
                                      0 after the probe, which leaves the FIFOs off */
    uint32_t trigger;              /* the configured RX trigger level, in bytes (1 in 16450 mode) */
    uint32_t fifo_ok;              /* the probe's verdict: 0 none (8250/16450), 1 broken (16550), 2 working (16550A);
                                      uart_init enables the FIFOs on 2 only, and runs 16450 mode otherwise */
    uint32_t tx_burst;             /* the plan-chosen TX burst cap; the driver clamps it to UART_CAP_FIFO_DEPTH,
                                      and to 1 unless the FCR shadow says FIFO mode. 0 until uart_init sets it:
                                      uart_tx refuses a state without one (UART_ERR_NOT_READY) */
    uint32_t scr_ok;               /* the probe's scratch test: 1 when SCR held both patterns (8250 vs 16450) */
    uint32_t lsr_saved;            /* UART_LSR_ERRORS an LSR read outside uart_rx_drain saw: a read of LSR clears
                                      OE, PE, FE and BI, and PE, FE and BI describe the byte at the FIFO head
                                      (PC16550D p.21), which only the drain takes -- so uart_tx, uart_flush,
                                      init's and the self-test's waits keep them here, and the drain counts
                                      them with its first LSR read (Linux's up->lsr_saved_flags). 0 after
                                      uart_init and after the self-test, which discard what the receiver held */
    uint32_t word_bits;            /* the word length uart_init programmed, 5 to 8 (LCR[1:0], PC16550D p.18): the
                                      self-test's patterns are its data bits. 0 until uart_init sets it:
                                      uart_selftest refuses a state without one (UART_ERR_NOT_READY) */
} uart_state;

#define UART_OK            0u      /* done */
#define UART_ERR_BUSY      1u      /* the transmitter did not go idle (TEMT) within the spin budget (init; flush;
                                      the self-test, before and after its loopback) */
#define UART_ERR_DIVISOR   2u      /* divisor 0 (PC16550D p.17: "not recommended") or past the 16-bit latch */
#define UART_ERR_FORMAT    3u      /* lcr_format has a bit outside UART_FORMAT_BITS: set break, DLAB, or past LCR */
#define UART_ERR_TRIGGER   4u      /* trigger_code is no uart_rx_trigger_t */
#define UART_ERR_LOOP_MSR  5u      /* loopback: MSR[7:4] did not mirror MCR[3:0] */
#define UART_ERR_LOOP_DATA 6u      /* loopback: a pattern byte did not come back */
#define UART_ERR_TIMEOUT   7u      /* THRE or DR did not come within the spin budget (the loopback; uart_tx) */
#define UART_ERR_NOT_READY 8u      /* uart_init has not filled the state: uart_tx (no TX burst cap) and
                                      uart_selftest (no word length) refuse it */
#define UART_ERR_RX_STUCK  9u      /* self-test: LSR.DR still set after UART_CAP_FIFO_DEPTH + 1 RBR reads */
#define UART_SPIN_BUDGET   65536u  /* every wait gives up after exactly this many LSR polls that find it not ready */
#define UART_LSR_ERRORS    (UART_LSR_OE_MASK | UART_LSR_PE_MASK | UART_LSR_FE_MASK | UART_LSR_BI_MASK)

/* uart_init's lcr_format is the line format alone, LCR[5:0]: word length, stop bits, parity enable, even and
 * stick parity. LCR[6] is Break Control, which holds SOUT at spacing until it is cleared, and LCR[7] is DLAB
 * (PC16550D Table 1, pp.17-18): neither is a format, and uart_init refuses both, as it refuses a bit past
 * the 8-bit register. */
#define UART_FORMAT_BITS (UART_LCR_WLS_MASK | UART_LCR_STB_MASK | UART_LCR_PEN_MASK | UART_LCR_EPS_MASK | UART_LCR_SP_MASK)

/* Section 3.1's identification; records fifo_ok and scr_ok, and leaves the FIFOs off -- FCR = 0, the
 * shadow 0 -- whatever it found, for uart_init to program; returns fifo_ok. */
uint32_t uart_probe(volatile uart_regs_t *u, uart_state *st);
/* Section 1.4's bring-up, in FIFO mode where the probe found a working FIFO (fifo_ok 2) and in 16450 mode
 * (FCR = 0) otherwise; returns UART_OK, UART_ERR_BUSY (TEMT did not come within UART_SPIN_BUDGET polls),
 * or the argument it refused before touching the device (UART_ERR_DIVISOR, UART_ERR_FORMAT,
 * UART_ERR_TRIGGER). */
uint32_t uart_init(volatile uart_regs_t *u, uart_state *st, uint32_t divisor, uint32_t lcr_format,
                   uint32_t trigger_code);
/* LSR-checked TX bursts of at most UART_CAP_FIFO_DEPTH bytes, one in 16450 mode, the bytes sent in *nsent;
 * returns UART_OK, UART_ERR_TIMEOUT (a wait for THRE gave up after UART_SPIN_BUDGET polls), or
 * UART_ERR_NOT_READY -- refused before touching the device: the state has no TX burst cap, so uart_init
 * has not run. */
uint32_t uart_tx(volatile uart_regs_t *u, uart_state *st, const uint8_t *buf, uint32_t len,
                 uint32_t *nsent);
/* Drains up to cap bytes, counting OE, PE, FE and BI into err_counts[0..3] -- those the state saved from LSR
 * reads made elsewhere with its first LSR read; returns the bytes drained. */
uint32_t uart_rx_drain(volatile uart_regs_t *u, uart_state *st, uint8_t *buf, uint32_t cap,
                       uint32_t *err_counts);
/* Polls LSR until TEMT, at most UART_SPIN_BUDGET polls, the polls that found it busy in *nbusy, the
 * receiver's errors those polls saw saved in the state; returns UART_OK, or UART_ERR_BUSY when the wait
 * gave up. */
uint32_t uart_flush(volatile uart_regs_t *u, uart_state *st, uint32_t *nbusy);
/* Section 1.7's MCR[4] loopback self-test at the word length init programmed (UART_ERR_NOT_READY on a state
 * without one, the device untouched): the transmitter idle first (TEMT, else UART_ERR_BUSY with the device
 * as it was), the receiver drained (at most UART_CAP_FIFO_DEPTH + 1 reads, else UART_ERR_RX_STUCK), each
 * wait for THRE or DR given up after UART_SPIN_BUDGET polls (UART_ERR_TIMEOUT), the loopback left only
 * with the transmitter idle again (else UART_ERR_BUSY), the FIFOs reset in FIFO mode only; returns UART_OK
 * or the first check that failed. */
uint32_t uart_selftest(volatile uart_regs_t *u, uart_state *st);

#endif /* UART_DRIVER_H */
