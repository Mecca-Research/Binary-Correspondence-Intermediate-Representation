/* sim16550.c -- the simulated 16550 (blueprint slice U2, section 8); sim16550.h says what it models.
 *
 * The register bits and the IIR source codes are the generated header's (uart16550_regs.h, the U0
 * registry): the field layout is the normative model's and the same on every placement, which the sim
 * takes at run time from its sim16550_config. Pages are the PC16550D's (SNLS378C) unless named.
 */
#include <stddef.h>

#include "sim16550.h"
#include "uart16550_regs.h"

#define LSR_ERRORS (UART_LSR_OE_MASK | UART_LSR_PE_MASK | UART_LSR_FE_MASK | UART_LSR_BI_MASK)
#define RX_ERRORS (UART_LSR_PE_MASK | UART_LSR_FE_MASK | UART_LSR_BI_MASK)
#define MSR_INPUTS (UART_MSR_CTS_MASK | UART_MSR_DSR_MASK | UART_MSR_RI_MASK | UART_MSR_DCD_MASK)
#define IER_BITS (UART_IER_ERBFI_MASK | UART_IER_ETBEI_MASK | UART_IER_ELSI_MASK | UART_IER_EDSSI_MASK)
#define MCR_BITS (UART_MCR_DTR_MASK | UART_MCR_RTS_MASK | UART_MCR_OUT1_MASK | UART_MCR_OUT2_MASK | UART_MCR_LOOP_MASK)
#define MCR_AFE_BIT 0x20u   /* MCR[5], on a part with autoflow (TL16C550C p.23); permanently 0 elsewhere (p.24) */
#define FCR_RESETS (UART_FCR_RXFIFORST_MASK | UART_FCR_TXFIFORST_MASK)

static int is_width(uint32_t w)
{
    return w == 1u || w == 2u || w == 4u || w == 8u;
}

const char *sim_config_problem(const sim16550_config *c)
{
    uint32_t k;
    if (c == NULL) {
        return "no configuration";
    }
    if (!is_width(c->stride)) {
        return "the stride is not 1, 2, 4 or 8";
    }
    if (!is_width(c->reg_width) || c->reg_width > c->stride) {
        return "the register lane does not fit in the stride";
    }
    if (c->base_offset % c->stride != 0u) {
        return "the base offset is not a multiple of the stride";
    }
    if (c->fifo_depth == 0u || c->fifo_depth > SIM16550_FIFO_MAX) {
        return "the FIFO depth is not 1..256";
    }
    for (k = 0u; k < 4u; k++) {
        if (c->rx_triggers[k] == 0u || c->rx_triggers[k] > c->fifo_depth
            || (k > 0u && c->rx_triggers[k] <= c->rx_triggers[k - 1u])) {
            return "the RX triggers are not four increasing byte counts within the FIFO";
        }
    }
    if (c->fcr_impl_mask > 0xffu || (c->fcr_impl_mask & UART_FCR_FIFOEN_MASK) == 0u) {
        return "FCR[0] is not implemented (a part with its FIFOs always on is a U9 profile)";
    }
    if ((c->fcr_reset & ~c->fcr_impl_mask) != 0u) {
        return "the FCR reset sets a bit the part does not implement";
    }
    return NULL;
}

/* The slot an access reaches: offset = base_offset + slot * stride, slot 0..7; -1 for no register. */
static int slot_of(const sim16550 *dev, uint32_t offset)
{
    uint32_t rel;
    if (offset < dev->cfg.base_offset) {
        return -1;
    }
    rel = offset - dev->cfg.base_offset;
    if (rel % dev->cfg.stride != 0u || rel / dev->cfg.stride > 7u) {
        return -1;
    }
    return (int)(rel / dev->cfg.stride);
}

static int fifo_mode(const sim16550 *dev)
{
    return (dev->fcr & UART_FCR_FIFOEN_MASK) != 0u;
}

static int dlab(const sim16550 *dev)
{
    return (dev->lcr & UART_LCR_DLAB_MASK) != 0u;
}

/* The bytes a side holds: the FIFO's depth, or one -- the holding register -- in 16450 mode. */
static uint32_t capacity(const sim16550 *dev)
{
    return fifo_mode(dev) ? dev->cfg.fifo_depth : 1u;
}

uint32_t sim_trigger_bytes(const sim16550 *dev)
{
    return dev->cfg.rx_triggers[(dev->fcr & UART_FCR_RXTRIG_MASK) >> UART_FCR_RXTRIG_SHIFT];
}

/* Received data available: the trigger level reached in FIFO mode (8.4.1), a byte held in 16450 mode. */
uint32_t sim_rda_pending(const sim16550 *dev)
{
    if (!fifo_mode(dev)) {
        return dev->rx_count > 0u ? 1u : 0u;
    }
    return dev->rx_count >= sim_trigger_bytes(dev) ? 1u : 0u;
}

/* The character timeout (8.4.1): FIFO mode, a byte held, and SIM16550_TIMEOUT_CHARS character times
 * since the last arrival and the last RBR read. */
uint32_t sim_timeout_pending(const sim16550 *dev)
{
    return fifo_mode(dev) && dev->rx_count > 0u && dev->idle >= SIM16550_TIMEOUT_CHARS ? 1u : 0u;
}

/* --- the receiver ----------------------------------------------------------------------------------- */

static void rx_clear(sim16550 *dev)
{
    dev->rx_head = 0u;
    dev->rx_count = 0u;
    dev->rx_err_count = 0u;
    dev->idle = 0u;
}

/* A character completes in the receive shift register (from SIN, or from the transmitter in loopback). */
static void rx_arrive(sim16550 *dev, uint32_t byte, uint32_t err)
{
    uint32_t at;
    err &= RX_ERRORS;
    if ((err & UART_LSR_BI_MASK) != 0u) {
        byte = 0u; /* a break loads one zero character (p.21) */
    }
    dev->idle = 0u;
    if (dev->rx_count == capacity(dev)) {
        /* overrun, indicated as soon as it happens (p.21) */
        dev->lsr_err |= UART_LSR_OE_MASK;
        dev->rx_overruns++;
        if (!fifo_mode(dev)) { /* 16450 mode: the character destroys the one held (p.21) */
            if (dev->rx_err[dev->rx_head] != 0u) {
                dev->rx_err_count--;
            }
            dev->rx_data[dev->rx_head] = (uint8_t)byte;
            dev->rx_err[dev->rx_head] = (uint8_t)err;
            if (err != 0u) {
                dev->rx_err_count++;
            }
            dev->lsr_err |= err;
        }
        return; /* FIFO mode: the shift-register character is lost and the FIFO is intact (p.21) */
    }
    at = (dev->rx_head + dev->rx_count) % SIM16550_FIFO_MAX;
    dev->rx_data[at] = (uint8_t)byte;
    dev->rx_err[at] = (uint8_t)err;
    if (err != 0u) {
        dev->rx_err_count++;
        if (fifo_mode(dev)) {
            dev->lsr_fifo_err = 1u; /* LSR[7]: an error byte is in the FIFO (p.22) */
        }
    }
    dev->rx_count++;
    if (dev->rx_count == 1u) {
        dev->lsr_err |= err; /* at the head at once: its errors are revealed (p.21) */
    }
}

static uint32_t rbr_read(sim16550 *dev)
{
    uint32_t byte;
    if (dev->rx_count == 0u) {
        return dev->rbr_last;
    }
    byte = dev->rx_data[dev->rx_head];
    if (dev->rx_err[dev->rx_head] != 0u) {
        dev->rx_err_count--;
    }
    dev->rx_head = (dev->rx_head + 1u) % SIM16550_FIFO_MAX;
    dev->rx_count--;
    if (dev->rx_count > 0u) {
        dev->lsr_err |= dev->rx_err[dev->rx_head]; /* the next byte reaches the head: its errors show */
    }
    dev->rbr_last = byte;
    dev->idle = 0u; /* a CPU read of the FIFO resets the timeout timer (p.16) */
    return byte;
}

/* --- the transmitter -------------------------------------------------------------------------------- */

static void line_put(sim16550 *dev, uint32_t byte)
{
    if (dev->line_len < SIM16550_LINE_MAX) {
        dev->line[dev->line_len] = (uint8_t)byte;
        dev->line_len++;
    } else {
        dev->line_lost++;
    }
}

/* An idle shift register takes the next byte (TL16C550D p.41); the FIFO (the holding register in
 * 16450 mode) emptying is THRE, and its interrupt. */
static void tx_start(sim16550 *dev)
{
    if (dev->tsr_busy != 0u || dev->tx_count == 0u) {
        return;
    }
    dev->tsr = dev->tx_data[dev->tx_head];
    dev->tx_head = (dev->tx_head + 1u) % SIM16550_FIFO_MAX;
    dev->tx_count--;
    dev->tsr_busy = 1u;
    if (dev->tx_count == 0u) {
        dev->thre_int = 1u;
    }
}

static void tx_clear(sim16550 *dev)
{
    if (dev->tx_count != 0u) {
        dev->thre_int = 1u; /* the FIFO empties */
    }
    dev->tx_head = 0u;
    dev->tx_count = 0u;
}

static void thr_write(sim16550 *dev, uint32_t byte)
{
    dev->thre_int = 0u; /* writing THR clears the THRE interrupt (Table 5, p.21) */
    if (dev->tx_count == capacity(dev)) {
        dev->tx_overflows++; /* a full transmitter: the byte is lost */
        return;
    }
    dev->tx_data[(dev->tx_head + dev->tx_count) % SIM16550_FIFO_MAX] = (uint8_t)byte;
    dev->tx_count++;
    tx_start(dev);
}

/* --- the line status, interrupt and modem registers ------------------------------------------------- */

static uint32_t lsr_value(const sim16550 *dev)
{
    uint32_t v = dev->lsr_err & LSR_ERRORS;
    if (dev->rx_count > 0u) {
        v |= UART_LSR_DR_MASK; /* a byte held (p.20) */
    }
    if (dev->tx_count == 0u) {
        v |= UART_LSR_THRE_MASK; /* the FIFO (holding register) is empty (p.22) */
        if (dev->tsr_busy == 0u) {
            v |= UART_LSR_TEMT_MASK; /* ... and the shift register too (p.22) */
        }
    }
    if (fifo_mode(dev) && dev->lsr_fifo_err != 0u) {
        v |= UART_LSR_FIFOERR_MASK; /* 0 in 16450 mode (p.22) */
    }
    return v;
}

static uint32_t lsr_read(sim16550 *dev)
{
    uint32_t v = lsr_value(dev);
    dev->lsr_err = 0u; /* a read clears OE, PE, FE and BI (pp.20-21) */
    /* LSR[7] clears unless error bytes remain in the FIFO (p.22) */
    dev->lsr_fifo_err = fifo_mode(dev) && dev->rx_err_count > 0u ? 1u : 0u;
    return v;
}

/* The highest-priority pending source whose IER bit is set (Table 5, p.21; a disabled source is not
 * indicated, p.23); IIR[7:6] are set with the FIFOs on (p.23). RDA ranks before the timeout, as in
 * Table 5. */
static uint32_t iir_value(const sim16550 *dev)
{
    uint32_t id = UART_IIR_NONE;
    if ((dev->ier & UART_IER_ELSI_MASK) != 0u && (dev->lsr_err & LSR_ERRORS) != 0u) {
        id = UART_IIR_RLS;
    } else if ((dev->ier & UART_IER_ERBFI_MASK) != 0u && sim_rda_pending(dev) != 0u) {
        id = UART_IIR_RDA;
    } else if ((dev->ier & UART_IER_ERBFI_MASK) != 0u && sim_timeout_pending(dev) != 0u) {
        id = UART_IIR_CTI;
    } else if ((dev->ier & UART_IER_ETBEI_MASK) != 0u && dev->thre_int != 0u) {
        id = UART_IIR_THRE;
    } else if ((dev->ier & UART_IER_EDSSI_MASK) != 0u && dev->msr_delta != 0u) {
        id = UART_IIR_MS;
    }
    return id | (fifo_mode(dev) ? UART_IIR_FIFOEN_MASK : 0u);
}

/* MSR[7:4]: the external inputs, or in loopback MCR[3:0] -- DTR on DSR, RTS on CTS, OUT1 on RI, OUT2 on
 * DCD (p.24). A change sets its delta: DCTS, DDSR, DDCD on either edge, TERI when RI goes from on to off
 * (the RI input from low to high, p.24). */
static void modem_update(sim16550 *dev)
{
    uint32_t in = 0u;
    uint32_t changed;
    if ((dev->mcr & UART_MCR_LOOP_MASK) != 0u) {
        if ((dev->mcr & UART_MCR_DTR_MASK) != 0u) {
            in |= UART_MSR_DSR_MASK;
        }
        if ((dev->mcr & UART_MCR_RTS_MASK) != 0u) {
            in |= UART_MSR_CTS_MASK;
        }
        if ((dev->mcr & UART_MCR_OUT1_MASK) != 0u) {
            in |= UART_MSR_RI_MASK;
        }
        if ((dev->mcr & UART_MCR_OUT2_MASK) != 0u) {
            in |= UART_MSR_DCD_MASK;
        }
    } else if (dev->cfg.has_modem != 0u) {
        in = dev->modem_lines & MSR_INPUTS;
    }
    changed = in ^ dev->msr_inputs;
    if ((changed & UART_MSR_CTS_MASK) != 0u) {
        dev->msr_delta |= UART_MSR_DCTS_MASK;
    }
    if ((changed & UART_MSR_DSR_MASK) != 0u) {
        dev->msr_delta |= UART_MSR_DDSR_MASK;
    }
    if ((dev->msr_inputs & UART_MSR_RI_MASK) != 0u && (in & UART_MSR_RI_MASK) == 0u) {
        dev->msr_delta |= UART_MSR_TERI_MASK;
    }
    if ((changed & UART_MSR_DCD_MASK) != 0u) {
        dev->msr_delta |= UART_MSR_DDCD_MASK;
    }
    dev->msr_inputs = in;
}

/* --- the writes ------------------------------------------------------------------------------------- */

/* FCR (p.22). On a gated part FCR[7:1] are programmed only with FCR[0] set in the same write; a write
 * with FCR[0] clear then changes the mode alone. The resets clear their FIFO, not the shift registers,
 * and clear themselves. A mode change clears both FIFOs, and the first THRE interrupt after it is
 * immediate (8.4.1). */
static void fcr_write(sim16550 *dev, uint32_t v)
{
    uint32_t was = dev->fcr & UART_FCR_FIFOEN_MASK;
    v &= dev->cfg.fcr_impl_mask;
    if (dev->cfg.fcr_gated != 0u && (v & UART_FCR_FIFOEN_MASK) == 0u) {
        dev->fcr &= ~UART_FCR_FIFOEN_MASK;
    } else {
        dev->fcr = v & ~FCR_RESETS;
        if ((v & UART_FCR_RXFIFORST_MASK) != 0u) {
            rx_clear(dev);
        }
        if ((v & UART_FCR_TXFIFORST_MASK) != 0u) {
            tx_clear(dev);
        }
    }
    if ((dev->fcr & UART_FCR_FIFOEN_MASK) != was) {
        rx_clear(dev);
        tx_clear(dev);
        dev->thre_int = 1u;
    }
}

/* IER[7:4] are always 0 (p.23). Enabling the THRE interrupt with the transmitter's FIFO (holding
 * register) empty raises it (TL16C550D p.41). */
static void ier_write(sim16550 *dev, uint32_t v)
{
    uint32_t was = dev->ier;
    dev->ier = v & IER_BITS;
    if ((was & UART_IER_ETBEI_MASK) == 0u && (dev->ier & UART_IER_ETBEI_MASK) != 0u
        && dev->tx_count == 0u) {
        dev->thre_int = 1u;
    }
}

static void mcr_write(sim16550 *dev, uint32_t v)
{
    dev->mcr = v & (MCR_BITS | (dev->cfg.has_afe != 0u ? MCR_AFE_BIT : 0u));
    modem_update(dev);
}

/* --- the bus ---------------------------------------------------------------------------------------- */

uint32_t sim_read(sim16550 *dev, uint32_t offset)
{
    uint32_t v;
    switch (slot_of(dev, offset)) {
    case 0:
        return dlab(dev) ? dev->dll : rbr_read(dev);
    case 1:
        return dlab(dev) ? dev->dlm : dev->ier;
    case 2:
        v = iir_value(dev);
        if ((v & (UART_IIR_INTID_MASK | UART_IIR_INTPEND_MASK)) == UART_IIR_THRE) {
            dev->thre_int = 0u; /* reading IIR clears a THRE interrupt it reports (Table 5, p.21) */
        }
        return v;
    case 3:
        return dev->lcr;
    case 4:
        return dev->mcr;
    case 5:
        return lsr_read(dev);
    case 6:
        v = dev->msr_inputs | dev->msr_delta;
        dev->msr_delta = 0u; /* a read clears the deltas (p.24) */
        return v;
    case 7:
        return dev->cfg.has_scr != 0u ? dev->scr : 0u;
    default:
        dev->bus_errors++;
        return 0u;
    }
}

void sim_write(sim16550 *dev, uint32_t offset, uint32_t val)
{
    uint32_t v = val & 0xffu; /* an 8-bit data bus: a wider lane's upper bits are reserved */
    switch (slot_of(dev, offset)) {
    case 0:
        if (dlab(dev)) {
            dev->dll = v;
        } else {
            thr_write(dev, v);
        }
        break;
    case 1:
        if (dlab(dev)) {
            dev->dlm = v;
        } else {
            ier_write(dev, v);
        }
        break;
    case 2:
        fcr_write(dev, v);
        break;
    case 3:
        dev->lcr = v;
        break;
    case 4:
        mcr_write(dev, v);
        break;
    case 5:
    case 6:
        dev->ignored_writes++; /* LSR and MSR writes are for factory test only (p.22) */
        break;
    case 7:
        if (dev->cfg.has_scr != 0u) {
            dev->scr = v;
        }
        break;
    default:
        dev->bus_errors++;
        break;
    }
}

uint32_t sim_peek(const sim16550 *dev, uint32_t offset)
{
    switch (slot_of(dev, offset)) {
    case 0:
        if (dlab(dev)) {
            return dev->dll;
        }
        return dev->rx_count > 0u ? dev->rx_data[dev->rx_head] : dev->rbr_last;
    case 1:
        return dlab(dev) ? dev->dlm : dev->ier;
    case 2:
        return iir_value(dev);
    case 3:
        return dev->lcr;
    case 4:
        return dev->mcr;
    case 5:
        return lsr_value(dev);
    case 6:
        return dev->msr_inputs | dev->msr_delta;
    case 7:
        return dev->cfg.has_scr != 0u ? dev->scr : 0u;
    default:
        return 0u;
    }
}

const char *sim_reg_name(const sim16550 *dev, uint32_t offset, int write)
{
    static const char *const reads[8] = {"RBR", "IER", "IIR", "LCR", "MCR", "LSR", "MSR", "SCR"};
    static const char *const writes[8] = {"THR", "IER", "FCR", "LCR", "MCR", "LSR", "MSR", "SCR"};
    int slot = slot_of(dev, offset);
    if (slot < 0) {
        return "-";
    }
    if (dlab(dev) && slot == 0) {
        return "DLL";
    }
    if (dlab(dev) && slot == 1) {
        return "DLM";
    }
    return write ? writes[slot] : reads[slot];
}

/* --- time, the line and reset ----------------------------------------------------------------------- */

void sim_tick(sim16550 *dev, uint32_t n_char_times)
{
    while (n_char_times > 0u) {
        n_char_times--;
        dev->time++;
        if (dev->rx_count > 0u && dev->idle < SIM16550_TIMEOUT_CHARS) {
            dev->idle++;
        }
        if (dev->tsr_busy != 0u && dev->tx_stall == 0u) {
            dev->tsr_busy = 0u;
            if ((dev->mcr & UART_MCR_LOOP_MASK) == 0u) {
                line_put(dev, dev->tsr);
            } else if (dev->loop_drop != 0u) {
                dev->loop_dropped++; /* the fault knob: the character reaches nothing */
            } else {
                rx_arrive(dev, dev->tsr, 0u); /* looped back into the receiver; SOUT stays marking (p.23) */
            }
        }
        tx_start(dev);
    }
}

void sim_stall_tx(sim16550 *dev, uint32_t on)
{
    dev->tx_stall = on != 0u ? 1u : 0u;
}

void sim_drop_loopback(sim16550 *dev, uint32_t on)
{
    dev->loop_drop = on != 0u ? 1u : 0u;
}

void sim_inject(sim16550 *dev, uint32_t byte, uint32_t err_bits)
{
    if ((dev->mcr & UART_MCR_LOOP_MASK) != 0u) {
        dev->rx_dropped++; /* loopback disconnects SIN (p.23) */
        return;
    }
    rx_arrive(dev, byte & 0xffu, err_bits);
}

void sim_set_modem(sim16550 *dev, uint32_t lines)
{
    dev->modem_lines = lines & MSR_INPUTS;
    modem_update(dev);
}

/* Reset values (Table 3, p.19): IER, LCR, MCR 00; FCR the placement's; LSR 60 (both transmitter
 * registers empty); MSR[3:0] 0 and MSR[7:4] the inputs; IIR 01, or C1 with the FIFOs on at reset
 * (Core16550). */
int sim_init(sim16550 *dev, const sim16550_config *cfg)
{
    unsigned char *bytes = (unsigned char *)dev;
    size_t k;
    if (sim_config_problem(cfg) != NULL) {
        return -1;
    }
    for (k = 0u; k < sizeof *dev; k++) {
        bytes[k] = 0u;
    }
    dev->cfg = *cfg;
    dev->fcr = cfg->fcr_reset & ~FCR_RESETS;
    dev->dll = 0x01u;
    dev->dlm = 0x00u;
    return 0;
}
