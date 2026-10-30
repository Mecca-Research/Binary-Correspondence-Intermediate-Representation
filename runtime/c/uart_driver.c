/* uart_driver.c -- the polled 16550 driver (blueprint slice U2): probe, init, TX, RX drain, flush and
 * a loopback self-test, compiled by BCIR on both rails and linked from their emitted units.
 *
 * Its one include is its API, uart_driver.h (the state, the status codes and the seven prototypes the
 * harness calls), which includes the GENERATED register map, uart16550_regs.h (the U0 registry,
 * bcir/frontends/devices/uart_model.py): the lane type, the offsets and fields, the placement's
 * capabilities and the uart_regmap_contract literal that every unit including it attests -- the
 * driver's only device header. Do not confuse it with runtime/c/uart_regs.h, which with
 * cfront_driver_uart.c is a CMSIS-flavoured COMPILER FIXTURE for the C front -- no 16550 and no
 * driver; the `uart_regs_t` there is that fixture's register struct, not this one's lane type.
 *
 * A register is the lane u[UART_LANE(UART_<REG>_OFF)] of a `volatile uart_regs_t *` at the device base,
 * so every access is an MMIO-domain claim the U1 protocol laws read (bcir/frontends/devices/uart_laws.py),
 * and the unit is law-clean on every placement. It keeps to the C subset both rails compile, calls no
 * function, reads no global and needs no libc (it compiles -ffreestanding).
 *
 * uart_probe leaves the FIFOs off whatever it found: once IIR[7:6] are read it writes FCR = 0 and
 * shadows 0. Linux leaves a port so after its own identification (drivers/tty/serial/8250/8250_port.c:
 * autoconfig() resets an identified port with serial8250_clear_fifos(), which ends with FCR = 0 on a part
 * with UART_CAP_FIFO; a part without one keeps the probe's FCR until serial8250_set_fcr() writes up->fcr,
 * which uart_config[] leaves 0 for PORT_16450 and PORT_16550). uart_init is section 1.4's bring-up, step
 * by step, programming FCR from that state, and it acts on the probe's verdict: FIFO mode only on a FIFO
 * the probe found working (fifo_ok 2, the 16550A). A 16450 or 8250 has no FIFO and the original 16550's
 * does not work, so there -- and on a state no probe has filled -- the part runs in 16450 mode: FCR = 0,
 * a one-byte trigger and one-byte TX bursts, as Linux's 8250 core runs PORT_16450 and PORT_16550
 * (fifo_size 1, tx_loadsz 1 and no FCR enable: uart_config[]). Every argument and state check comes
 * before the device is touched: uart_init refuses a format that is no format (LCR[6], set break, or
 * LCR[7], DLAB), and uart_tx a state no init has filled (no TX burst cap).
 *
 * FCR is written from values and shadowed in uart_state, never read back (hazard H-E: on the 16550 a
 * slot-2 read is IIR, and where FCR does read back -- the AXI core -- it is not relied on). The shadow is
 * FCR's persistent bits as last written -- FIFO enable, DMA mode and the RX trigger -- and never the two
 * FIFO resets, FCR[2:1], which clear themselves the moment they are written (PC16550D p.22): a later
 * read-modify-write from a shadow that kept them (a trigger or DMA-mode change) would reset both FIFOs
 * and drop what they hold. The resets are written when they are meant, and only with FCR[0] in the same
 * write. Linux keeps its FCR shadow the same way (up->fcr: set from uart_config[] and changed by
 * read-modify-write in do_set_rxtrig, while serial8250_clear_fifos writes the resets, with the FIFO
 * enable, on a part with a FIFO only). TX never waits on a THRE interrupt edge (section 3.2): each burst
 * follows an LSR read that shows THRE. Every wait the driver makes -- init's for TEMT, TX's for THRE,
 * the flush's for TEMT, the self-test's for TEMT, for THRE and for DR, the release's for TEMT -- polls
 * LSR and gives up after exactly UART_SPIN_BUDGET polls that find the device not ready.
 *
 * The receiver's errors. A read of LSR clears OE, PE, FE and BI, and in FIFO mode PE, FE and BI describe
 * the character at the FIFO head, revealed when it gets there (PC16550D p.21). The drain is the one call
 * that takes received characters, but uart_tx, uart_flush and the waits of uart_init and the self-test
 * read LSR too: each of those reads ORs the four into st->lsr_saved, and the drain counts them with its
 * first LSR read, so a byte's error survives a TX poll made while it waited at the head. Linux does the
 * same: serial_lsr_in() (drivers/tty/serial/8250/8250.h) keeps LSR_SAVE_FLAGS -- BI, FE, PE and OE
 * (include/linux/serial_8250.h) -- in up->lsr_saved_flags, which the receive path merges and clears
 * (8250_port.c) and startup clears. A flag seen twice before a drain counts once, as there.
 *
 * Buffers and headroom. The driver is synchronous: a buffer handed to uart_tx or uart_rx_drain is the
 * caller's again when the call returns, so the buffer-ownership law of the 2026-10-27 review (blueprint
 * section 12: AVAILABLE -> RESERVED -> DEVICE_OWNED -> COMPLETING -> RETIRED, DEVICE_OWNED(B) => no
 * reuse of B) holds trivially: no buffer is device-owned across a return. And a TX burst never exceeds
 * the free space the driver has just observed: THRE says the transmitter's FIFO is empty -- in 16450
 * mode its one-byte holding register (PC16550D p.22) -- so F_free is UART_CAP_FIFO_DEPTH in FIFO mode
 * and 1 in 16450 mode, and a burst of at most that many bytes after it (the cap clamped to one byte
 * unless the FCR shadow says FIFO mode) is the polled case of the headroom bound
 * F_free >= ceil(R x T_service) + N_inflight + M_safety, with nothing in flight to the FIFO when the
 * burst begins. bcir/tests/test_uart_driver.py's TX tests, one per mode, are its witnesses.
 */
#include "uart_driver.h"

/* Section 3.1's identification: FCR = 0xC1 (FIFOs on, trigger 14), then IIR[7:6] says what answered --
 * 11 a working FIFO (16550A), 10 a broken one (the original 16550), 00 none (16450/8250) -- and FCR = 0
 * turns the FIFOs off again, shadowed as 0, whatever answered: a broken FIFO is not left on, and uart_init
 * starts from a known FCR. The scratch register, written and read back twice, tells a 16450 from an 8250.
 * The verdict is recorded in the state, where uart_init acts on it; returns fifo_ok. */
uint32_t uart_probe(volatile uart_regs_t *u, uart_state *st)
{
    uint32_t fcr = UART_FCR_FIFOEN_MASK | UART_RX_TRIGGER_14;
    uint32_t iir;
    uint32_t first;
    uint32_t second;
    u[UART_LANE(UART_FCR_OFF)] = fcr;
    iir = u[UART_LANE(UART_IIR_OFF)];
    u[UART_LANE(UART_FCR_OFF)] = 0u;
    st->fcr_shadow = 0u;
    u[UART_LANE(UART_SCR_OFF)] = 0xa5u;
    first = u[UART_LANE(UART_SCR_OFF)];
    u[UART_LANE(UART_SCR_OFF)] = 0x5au;
    second = u[UART_LANE(UART_SCR_OFF)];
    st->scr_ok = (first == 0xa5u && second == 0x5au) ? 1u : 0u;
    switch ((iir & UART_IIR_FIFOEN_MASK) >> UART_IIR_FIFOEN_SHIFT) {
    case 3u:
        st->fifo_ok = 2u;
        break;
    case 2u:
        st->fifo_ok = 1u;
        break;
    default:
        st->fifo_ok = 0u;
        break;
    }
    return st->fifo_ok;
}

/* Section 1.4, exactly: (1) wait for TEMT, (2) LCR with DLAB and the format in one write, (3) DLL then
 * DLM, (4) LCR with the format alone, closing the DLAB bracket, (5) FCR -- FIFO enable, both resets and
 * the trigger in one write (FCR[0] must be in it for the rest to take: PC16550D p.22), or 0 where the
 * probe found no working FIFO, (6) MCR, (7) IER last (0: polled), (8) one read each of LSR, RBR, IIR
 * and MSR to clear stale status. Every argument is checked before the device is touched; the format is
 * LCR[5:0] (UART_FORMAT_BITS), and the LCR writes mask it so that the U1 law can read each one's bit 7.
 * The wait for TEMT gives up after exactly UART_SPIN_BUDGET polls that find the line busy, with
 * UART_ERR_BUSY and the device as it was. A loopback the self-test held ends here: with TEMT seen first,
 * the character its shift register held has been looped back, and the FIFO reset (or step 8's RBR read
 * in 16450 mode) drops it, while step 6 restores the line. Returns UART_OK or the error that refused. */
uint32_t uart_init(volatile uart_regs_t *u, uart_state *st, uint32_t divisor, uint32_t lcr_format,
                   uint32_t trigger_code)
{
    uint32_t spins = 0u;
    uint32_t bytes = 0u;
    uint32_t burst;
    uint32_t fcr;
    uint32_t lsr;
    if (divisor == 0u || divisor > 0xffffu) {
        return UART_ERR_DIVISOR;
    }
    if ((lcr_format & ~UART_FORMAT_BITS) != 0u) {
        return UART_ERR_FORMAT;
    }
    switch (trigger_code) {
    case UART_RX_TRIGGER_1:
        bytes = 1u;
        break;
    case UART_RX_TRIGGER_4:
        bytes = 4u;
        break;
    case UART_RX_TRIGGER_8:
        bytes = 8u;
        break;
    case UART_RX_TRIGGER_14:
        bytes = 14u;
        break;
    default:
        return UART_ERR_TRIGGER;
    }
    /* FIFO mode only on a FIFO the probe found working (fifo_ok 2). Anything else runs in 16450 mode:
     * FCR = 0, which asks for nothing FCR[0] gates (the U1 law's reading: it programs what it says), and
     * THRE then means the one-byte holding register is empty (PC16550D p.22), so the trigger and the TX
     * burst are one byte each (uart_config[]'s PORT_16450 and PORT_16550: fifo_size 1, tx_loadsz 1). */
    if (st->fifo_ok == 2u) {
        fcr = UART_FCR_FIFOEN_MASK | UART_FCR_RXFIFORST_MASK | UART_FCR_TXFIFORST_MASK | trigger_code;
        burst = UART_CAP_FIFO_DEPTH;
    } else {
        fcr = 0u;
        bytes = 1u;
        burst = 1u;
    }
    /* (1) reconfigure only at idle: the line may be mid-character. Each poll's receiver errors are saved:
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
    /* (2)-(4) the DLAB bracket, straight-line: DLL then DLM, LSB first */
    u[UART_LANE(UART_LCR_OFF)] = (lcr_format & UART_FORMAT_BITS) | UART_LCR_DLAB_MASK;
    u[UART_LANE(UART_DLL_OFF)] = divisor & 0xffu;
    u[UART_LANE(UART_DLM_OFF)] = (divisor >> 8) & 0xffu;
    u[UART_LANE(UART_LCR_OFF)] = lcr_format & UART_FORMAT_BITS;
    /* (5) one FCR write, shadowed less the resets, which clear themselves (PC16550D p.22): H-E */
    u[UART_LANE(UART_FCR_OFF)] = fcr;
    st->fcr_shadow = fcr & ~(UART_FCR_RXFIFORST_MASK | UART_FCR_TXFIFORST_MASK);
    /* (6) DTR and RTS; OUT2 only gates an interrupt line, which a polled driver does not use */
    u[UART_LANE(UART_MCR_OFF)] = UART_MCR_DTR_MASK | UART_MCR_RTS_MASK;
    /* (7) polled: no interrupt source enabled */
    u[UART_LANE(UART_IER_OFF)] = 0u;
    /* (8) stale status */
    (void)u[UART_LANE(UART_LSR_OFF)];
    (void)u[UART_LANE(UART_RBR_OFF)];
    (void)u[UART_LANE(UART_IIR_OFF)];
    (void)u[UART_LANE(UART_MSR_OFF)];
    st->trigger = bytes;
    st->tx_burst = burst;
    st->word_bits = 5u + ((lcr_format & UART_LCR_WLS_MASK) >> UART_LCR_WLS_SHIFT);
    st->lsr_saved = 0u; /* the receiver's old bytes went with the FIFO reset or the RBR read, and their errors */
    st->loop_held = 0u; /* step 6 wrote MCR, after TEMT: a loopback the self-test held is over */
    return UART_OK;
}

/* LSR-checked bursts: while bytes remain, read LSR; with THRE the transmitter's FIFO is empty -- in 16450
 * mode its one-byte holding register (PC16550D p.22) -- so write up to min(tx_burst,
 * UART_CAP_FIFO_DEPTH, remaining) bytes, and one unless the FCR shadow says FIFO mode: never more than
 * the device was just seen to have free, whatever cap a plan left in the state, and never a byte without
 * a THRE read before its burst. A state with no cap (0) is one uart_init has not filled, and it is
 * refused before the device is touched, not read as some default cap. A wait for THRE gives up after
 * exactly UART_SPIN_BUDGET polls that find it clear. A state whose loopback the self-test holds is
 * refused too: in the loopback the bytes would go to the receiver, never to SOUT (p.23). The bytes sent
 * go to *nsent (0 when refused); returns UART_OK when all len were, UART_ERR_TIMEOUT when a wait gave up,
 * UART_ERR_NOT_READY or UART_ERR_LOOP_HELD. */
uint32_t uart_tx(volatile uart_regs_t *u, uart_state *st, const uint8_t *buf, uint32_t len,
                 uint32_t *nsent)
{
    uint32_t sent = 0u;
    uint32_t spins = 0u;
    uint32_t cap = st->tx_burst;
    *nsent = 0u;
    if (cap == 0u) {
        return UART_ERR_NOT_READY;
    }
    if (st->loop_held != 0u) {
        return UART_ERR_LOOP_HELD;
    }
    if (cap > UART_CAP_FIFO_DEPTH) {
        cap = UART_CAP_FIFO_DEPTH;
    }
    if ((st->fcr_shadow & UART_FCR_FIFOEN_MASK) == 0u) {
        cap = 1u;
    }
    while (sent < len) {
        uint32_t lsr = u[UART_LANE(UART_LSR_OFF)];
        st->lsr_saved = st->lsr_saved | (lsr & UART_LSR_ERRORS); /* the receiver's, for the drain */
        if ((lsr & UART_LSR_THRE_MASK) == 0u) {
            spins = spins + 1u;
            if (spins == UART_SPIN_BUDGET) {
                break;
            }
        } else {
            uint32_t burst = 0u;
            spins = 0u;
            while (burst < cap && sent < len) {
                u[UART_LANE(UART_THR_OFF)] = buf[sent];
                sent = sent + 1u;
                burst = burst + 1u;
            }
        }
    }
    *nsent = sent;
    return sent == len ? UART_OK : UART_ERR_TIMEOUT;
}

/* Drain the RX FIFO into buf, at most cap bytes. ONE LSR read per iteration feeds both the DR test and
 * the error triage: OE, PE, FE and BI of every LSR read the drain makes are counted into
 * err_counts[0..3] (PE, FE and BI describe the byte at the FIFO head, revealed when it got there; OE is
 * no byte's -- it is set when the overrun happens, PC16550D p.21 -- so LSR[7], which counts PE, FE and
 * BI anywhere in the FIFO, cannot gate the triage). The read is made only when a byte could be taken,
 * so an LSR read's errors are never dropped at the cap. The errors another call's LSR read saw
 * (st->lsr_saved) are counted with the first read -- they describe the byte at the FIFO head, the first
 * the drain takes -- and kept when the cap allows no read. A state whose loopback the self-test holds
 * gives nothing: its receiver holds the test's own characters, not a peer's. Returns the bytes drained. */
uint32_t uart_rx_drain(volatile uart_regs_t *u, uart_state *st, uint8_t *buf, uint32_t cap,
                       uint32_t *err_counts)
{
    uint32_t n = 0u;
    uint32_t saved = st->lsr_saved;
    if (st->loop_held != 0u) {
        return 0u;
    }
    while (n < cap) {
        uint32_t lsr = u[UART_LANE(UART_LSR_OFF)] | saved;
        saved = 0u;
        if ((lsr & UART_LSR_OE_MASK) != 0u) {
            err_counts[0] = err_counts[0] + 1u;
        }
        if ((lsr & UART_LSR_PE_MASK) != 0u) {
            err_counts[1] = err_counts[1] + 1u;
        }
        if ((lsr & UART_LSR_FE_MASK) != 0u) {
            err_counts[2] = err_counts[2] + 1u;
        }
        if ((lsr & UART_LSR_BI_MASK) != 0u) {
            err_counts[3] = err_counts[3] + 1u;
        }
        if ((lsr & UART_LSR_DR_MASK) == 0u) {
            break;
        }
        buf[n] = (uint8_t)u[UART_LANE(UART_RBR_OFF)];
        n = n + 1u;
    }
    st->lsr_saved = saved; /* counted with the first read, or kept when the cap allowed none */
    return n;
}

/* Poll LSR until TEMT (the TX FIFO and the shift register both empty: the line is idle), giving up
 * after exactly UART_SPIN_BUDGET polls that find it busy, each poll's receiver errors saved for the
 * drain. The polls that found it busy go to *nbusy; returns UART_OK when TEMT was seen, UART_ERR_BUSY
 * when the wait gave up, and UART_ERR_LOOP_HELD, the device untouched, for a state whose loopback the
 * self-test holds: there TEMT would say a character went to the receiver, not that the line is idle. */
uint32_t uart_flush(volatile uart_regs_t *u, uart_state *st, uint32_t *nbusy)
{
    uint32_t spins = 0u;
    *nbusy = 0u;
    if (st->loop_held != 0u) {
        return UART_ERR_LOOP_HELD;
    }
    while (spins < UART_SPIN_BUDGET) {
        uint32_t lsr = u[UART_LANE(UART_LSR_OFF)];
        st->lsr_saved = st->lsr_saved | (lsr & UART_LSR_ERRORS); /* the receiver's, for the drain */
        if ((lsr & UART_LSR_TEMT_MASK) != 0u) {
            break;
        }
        spins = spins + 1u;
    }
    *nbusy = spins;
    return spins == UART_SPIN_BUDGET ? UART_ERR_BUSY : UART_OK;
}

/* Section 1.7's loopback (MCR[4]): with LOOP set, MCR[3:0] must come back on MSR[7:4] -- DTR on DSR, RTS
 * on CTS, OUT1 on RI, OUT2 on DCD (PC16550D p.24), checked with two patterns that between them set and
 * clear all four (the first is Linux's own existence test in autoconfig(): MCR = LOOP|OUT2|RTS read back
 * as DCD|CTS) -- and each pattern character written to THR must come back through RBR. The patterns are
 * 0x55 and 0xAA at the word length uart_init programmed -- only LCR[1:0]'s data bits are sent and received
 * (p.18), so at 5 bits they are 0x15 and 0x0A -- and only those bits of what RBR returns are compared,
 * since the PC16550D does not state what the bits above them read. A state no init has filled has no
 * word length, and it is refused before the device is touched (UART_ERR_NOT_READY).
 *
 * In the loopback the transmitter's shift register feeds the receiver and SOUT is held marking (p.23), so
 * the test enters it only with the transmitter idle: it waits for TEMT first, since bytes uart_tx accepted
 * are still the line's -- looped back they would be lost and read in place of a pattern -- and it gives up
 * after exactly UART_SPIN_BUDGET polls with UART_ERR_BUSY and the device as it was (the polls' receiver
 * errors saved for the drain). It leaves the loopback the same way, once TEMT shows that whatever it wrote
 * has been looped back rather than left for SOUT; in FIFO mode both FIFOs are reset while it is still in
 * the loopback, so a pattern the transmitter's FIFO still holds is dropped, not sent, and in 16450 mode,
 * which has no FIFO to reset, a looped character the receiver still holds is read out. When TEMT does not
 * come, the shift register still holds a character -- a FIFO reset does not clear it (p.22) -- and
 * leaving the loopback would send it on SOUT, so the test holds the loopback: it keeps the MCR it found
 * in st->mcr_saved, sets st->loop_held and returns UART_ERR_LOOP_HELD, whatever failed first. uart_tx,
 * uart_flush and the self-test then refuse the state and uart_rx_drain takes nothing from it, until
 * uart_selftest_release -- or uart_init -- ends the loopback once TEMT comes. A state whose loopback is
 * held is refused before the device is touched (UART_ERR_LOOP_HELD): MCR then reads the loopback's own
 * value, not the one to restore.
 *
 * A byte the receiver holds from before would come back in place of a pattern, so the receiver is drained
 * first, while LSR shows DR, and bounded: at most UART_CAP_FIFO_DEPTH + 1 reads, a full FIFO and the
 * character completing in the shift register behind it. A receiver still showing data after that has a
 * peer that does not stop sending, and the test fails (UART_ERR_RX_STUCK) without entering the loopback.
 * Linux bounds its drains the same way (8250_port.c: size_fifo() reads RX while LSR shows DR, at most 256
 * times; serial8250_rx_chars() takes at most 256 characters a call). What the test drains it discards, with
 * the errors saved for those bytes (st->lsr_saved is 0 after it). The drain runs before the loopback is
 * entered, so a character completing in the few instructions between them would still be read back in
 * place of a pattern; once in the loopback SIN is disconnected (p.23). The FIFO reset writes FCR[2:1] with
 * FCR[0] in the same write, in FIFO mode only, since with FCR[0] clear a gated part programs nothing else
 * (p.22) (Linux's serial8250_clear_fifos resets them only on a part with UART_CAP_FIFO); MCR is then
 * restored (it is a read/write register; FCR is the one never read back) and MSR read once to clear the
 * deltas the test raised. Each wait for THRE before a pattern and for DR after it gives up after exactly
 * UART_SPIN_BUDGET polls that find it not ready (UART_ERR_TIMEOUT), and the loopback is undone the same
 * way. Returns UART_OK or the first check that failed. */
uint32_t uart_selftest(volatile uart_regs_t *u, uart_state *st)
{
    uint32_t verdict = UART_OK;
    uint32_t round = 0u;
    uint32_t drained = 0u;
    uint32_t spins = 0u;
    uint32_t looped = 0u;
    uint32_t held = 0u;
    uint32_t mask;
    uint32_t mcr;
    uint32_t msr;
    uint32_t lsr;
    uint32_t fcr;
    if (st->word_bits < 5u || st->word_bits > 8u) {
        return UART_ERR_NOT_READY;
    }
    if (st->loop_held != 0u) {
        return UART_ERR_LOOP_HELD;
    }
    mask = (1u << st->word_bits) - 1u;
    /* the transmitter idle before the loopback: what uart_tx accepted is the line's */
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
    mcr = u[UART_LANE(UART_MCR_OFF)];
    /* the receiver drained of what it held before the test, bounded */
    while ((u[UART_LANE(UART_LSR_OFF)] & UART_LSR_DR_MASK) != 0u) {
        if (drained == UART_CAP_FIFO_DEPTH + 1u) {
            verdict = UART_ERR_RX_STUCK;
            break;
        }
        (void)u[UART_LANE(UART_RBR_OFF)];
        drained = drained + 1u;
    }
    st->lsr_saved = 0u; /* the drained bytes are discarded, and the errors saved for them */
    if (verdict == UART_OK) {
        /* MSR[7:4] are the four modem inputs; in loopback they are MCR[3:0] */
        u[UART_LANE(UART_MCR_OFF)] = UART_MCR_LOOP_MASK | UART_MCR_OUT2_MASK | UART_MCR_RTS_MASK;
        looped = 1u;
        msr = u[UART_LANE(UART_MSR_OFF)];
        if ((msr & 0xf0u) != (UART_MSR_DCD_MASK | UART_MSR_CTS_MASK)) {
            verdict = UART_ERR_LOOP_MSR;
        }
        u[UART_LANE(UART_MCR_OFF)] = UART_MCR_LOOP_MASK | UART_MCR_OUT1_MASK | UART_MCR_DTR_MASK;
        msr = u[UART_LANE(UART_MSR_OFF)];
        if (verdict == UART_OK && (msr & 0xf0u) != (UART_MSR_RI_MASK | UART_MSR_DSR_MASK)) {
            verdict = UART_ERR_LOOP_MSR;
        }
    }
    while (verdict == UART_OK && round < 2u) {
        uint32_t pattern = 0x55u;
        uint32_t got;
        if (round != 0u) {
            pattern = 0xaau;
        }
        pattern = pattern & mask;
        spins = 0u;
        while (spins < UART_SPIN_BUDGET && (u[UART_LANE(UART_LSR_OFF)] & UART_LSR_THRE_MASK) == 0u) {
            spins = spins + 1u;
        }
        if (spins == UART_SPIN_BUDGET) {
            verdict = UART_ERR_TIMEOUT;
            break;
        }
        u[UART_LANE(UART_THR_OFF)] = pattern;
        spins = 0u;
        while (spins < UART_SPIN_BUDGET && (u[UART_LANE(UART_LSR_OFF)] & UART_LSR_DR_MASK) == 0u) {
            spins = spins + 1u;
        }
        if (spins == UART_SPIN_BUDGET) {
            verdict = UART_ERR_TIMEOUT;
            break;
        }
        got = u[UART_LANE(UART_RBR_OFF)];
        if ((got & mask) != pattern) {
            verdict = UART_ERR_LOOP_DATA;
        }
        round = round + 1u;
    }
    if (looped != 0u) {
        /* out of the loopback only with the transmitter idle: what the test wrote is looped, not sent */
        spins = 0u;
        while (spins < UART_SPIN_BUDGET && (u[UART_LANE(UART_LSR_OFF)] & UART_LSR_TEMT_MASK) == 0u) {
            spins = spins + 1u;
        }
        if (spins == UART_SPIN_BUDGET) {
            held = 1u; /* the shift register still holds a character: SOUT would carry it */
        }
    }
    if ((st->fcr_shadow & UART_FCR_FIFOEN_MASK) != 0u) {
        /* still in the loopback: a pattern the transmitter holds is dropped, not sent */
        fcr = st->fcr_shadow | UART_FCR_RXFIFORST_MASK | UART_FCR_TXFIFORST_MASK;
        u[UART_LANE(UART_FCR_OFF)] = fcr;
        st->fcr_shadow = fcr & ~(UART_FCR_RXFIFORST_MASK | UART_FCR_TXFIFORST_MASK);
    } else if (looped != 0u && (u[UART_LANE(UART_LSR_OFF)] & UART_LSR_DR_MASK) != 0u) {
        (void)u[UART_LANE(UART_RBR_OFF)]; /* 16450 mode: the one looped character RBR may still hold */
    }
    if (held != 0u) {
        st->mcr_saved = mcr; /* the loopback stays: uart_selftest_release or uart_init ends it */
        st->loop_held = 1u;
        return UART_ERR_LOOP_HELD;
    }
    u[UART_LANE(UART_MCR_OFF)] = mcr;
    (void)u[UART_LANE(UART_MSR_OFF)];
    return verdict;
}

/* Ends a loopback the self-test held (st->loop_held): its shift register held a character when the test
 * gave up its wait for TEMT, and in the loopback that character goes to the receiver, not to SOUT (p.23),
 * so the loopback is left only once TEMT shows it has. The wait gives up after exactly UART_SPIN_BUDGET
 * polls that find the transmitter busy, with UART_ERR_LOOP_HELD and the loopback still held. Then, as the
 * test leaves it: in FIFO mode both FIFOs are reset while still looped -- what the receiver holds is the
 * test's, not a peer's -- and in 16450 mode the one character RBR may hold is read out; MCR is restored
 * (st->mcr_saved) and MSR read once to clear the deltas. A state that holds no loopback is left as it is,
 * the device untouched. Returns UART_OK or UART_ERR_LOOP_HELD. */
uint32_t uart_selftest_release(volatile uart_regs_t *u, uart_state *st)
{
    uint32_t spins = 0u;
    uint32_t fcr;
    if (st->loop_held == 0u) {
        return UART_OK;
    }
    while ((u[UART_LANE(UART_LSR_OFF)] & UART_LSR_TEMT_MASK) == 0u) {
        spins = spins + 1u;
        if (spins == UART_SPIN_BUDGET) {
            return UART_ERR_LOOP_HELD;
        }
    }
    if ((st->fcr_shadow & UART_FCR_FIFOEN_MASK) != 0u) {
        fcr = st->fcr_shadow | UART_FCR_RXFIFORST_MASK | UART_FCR_TXFIFORST_MASK;
        u[UART_LANE(UART_FCR_OFF)] = fcr;
        st->fcr_shadow = fcr & ~(UART_FCR_RXFIFORST_MASK | UART_FCR_TXFIFORST_MASK);
    } else if ((u[UART_LANE(UART_LSR_OFF)] & UART_LSR_DR_MASK) != 0u) {
        (void)u[UART_LANE(UART_RBR_OFF)]; /* 16450 mode: the looped character */
    }
    u[UART_LANE(UART_MCR_OFF)] = st->mcr_saved;
    (void)u[UART_LANE(UART_MSR_OFF)];
    st->loop_held = 0u;
    return UART_OK;
}
