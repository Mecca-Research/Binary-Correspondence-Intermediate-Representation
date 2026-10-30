/* sim16550.h -- a deterministic, simulated 16550 for the host (blueprint slice U2, section 8).
 *
 * The device the U2 driver runs against: two FIFOs, the receive side carrying three error bits per byte;
 * the transmit shift register; the LSR, IIR and MSR state machines; the RX trigger level and the
 * four-character timeout; the overrun rule; the MCR[4] loopback; and section 1.1's register mux
 * (direction, DLAB, FIFO mode) over a placement's layout. A placement is configured from the U0 registry
 * (sim16550_config mirrors UartPlacement's knobs; the transcript harness fills it from the generated
 * uart16550_regs.h) and the register bits are the generated header's. It identifies as a 16550A (IIR[7:6]
 * = 11 with the FIFOs on) and its scratch register works.
 *
 * Time is character times and passes only in sim_tick: nothing here reads a clock, so a run is a
 * function of the calls made. The caller owns the state (a sim16550 is plain data); freestanding C.
 *
 * Where the documents leave a choice open, the sim takes one and says so here:
 *   - a byte written to a full transmitter (16 in the FIFO -- the holding register in 16450 mode -- with
 *     the shift register busy) is lost and counted (tx_overflows);
 *   - a byte written to an idle transmitter moves into the shift register at once (TL16C550D p.41), so
 *     THRE stays set; the PC16550D's sub-character delay of that THRE (8.4.1 item 2) is not modeled: time
 *     is whole characters, and the driver never depends on THRE edge timing (blueprint section 3.2);
 *   - an RBR read of an empty receiver returns the last byte read;
 *   - the divisor latches reset to 0x0001 and SCR to 0x00 on every placement (undefined on the PC16550D;
 *     Core16550 resets the latches to 0x01/0x00, the AXI core to its 9600-baud divisor: neither is a
 *     UartPlacement field), and the external modem inputs are inactive until sim_set_modem.
 *
 * Two fault knobs model what no working 16550 does, so that a harness can drive a driver's timeouts; both
 * are off after sim_init, and off they change nothing:
 *   - sim_stall_tx: the transmit shift register never finishes its character, so the transmitter never
 *     drains and THRE stays off once a byte waits behind it (the line receives nothing meanwhile);
 *   - sim_drop_loopback: in loopback a finished character never reaches the receiver -- it is counted in
 *     loop_dropped -- so DR never sets after a loopback write.
 */
#ifndef SIM16550_H
#define SIM16550_H

#include <stdint.h>

/* The deepest FIFO a configuration may ask for (the H16750S's largest synthesis, blueprint section 1.10). */
#define SIM16550_FIFO_MAX 256u
/* The transmitted bytes the sim keeps, in order, for its observer. */
#define SIM16550_LINE_MAX 4096u
/* Character times with a byte held and no arrival and no RBR read that make a character timeout
 * (PC16550D p.16). */
#define SIM16550_TIMEOUT_CHARS 4u

/* sim_inject's err_bits, at their LSR positions (PC16550D pp.20-21). */
#define SIM16550_PE 0x04u
#define SIM16550_FE 0x08u
#define SIM16550_BI 0x10u

/* A placement as the device sees it: the layout and the knobs of uart_model.UartPlacement. */
typedef struct sim16550_config {
    const char *placement;    /* the registry name, for the observer */
    uint32_t stride;          /* bytes between slots: 1, 2, 4 or 8 */
    uint32_t base_offset;     /* slot 0's byte offset from the device base */
    uint32_t reg_width;       /* bytes per register lane; the data bus is the lane's low byte */
    uint32_t fcr_reset;       /* FCR after reset (Core16550: FIFOs on) */
    uint32_t fcr_gated;       /* FCR[7:1] are programmed only with FCR[0] set in the same write */
    uint32_t fcr_impl_mask;   /* the FCR bits that exist; FCR[0] must (a part without it is a U9 profile) */
    uint32_t fifo_depth;      /* bytes per FIFO, 1..SIM16550_FIFO_MAX */
    uint32_t rx_triggers[4];  /* the RX trigger level per FCR[7:6] code, in bytes, increasing */
    uint32_t has_scr;         /* a scratch register at slot 7 (else it reads 0) */
    uint32_t has_afe;         /* MCR[5] exists (stored; autoflow itself is slice U6) */
    uint32_t has_modem;       /* the modem inputs are wired (else MSR[7:4] read 0 outside loopback) */
} sim16550_config;

typedef struct sim16550 {
    sim16550_config cfg;
    /* the registers a write programs */
    uint32_t ier;
    uint32_t lcr;
    uint32_t mcr;
    uint32_t scr;
    uint32_t dll;
    uint32_t dlm;
    uint32_t fcr;             /* as programmed: enable, DMA mode and trigger; the resets never stay set */
    /* the receiver: a ring of fifo_depth bytes (one in 16450 mode), each with its PE/FE/BI bits */
    uint8_t rx_data[SIM16550_FIFO_MAX];
    uint8_t rx_err[SIM16550_FIFO_MAX];
    uint32_t rx_head;
    uint32_t rx_count;
    uint32_t rx_err_count;    /* bytes held that carry PE, FE or BI */
    uint32_t lsr_err;         /* LSR OE/PE/FE/BI as latched: set when it happens or the byte reaches the head */
    uint32_t lsr_fifo_err;    /* LSR[7] as latched */
    uint32_t rbr_last;        /* the last byte read */
    uint32_t idle;            /* character times since the last arrival or RBR read (saturates) */
    /* the transmitter: a ring of fifo_depth bytes (one in 16450 mode) and the shift register */
    uint8_t tx_data[SIM16550_FIFO_MAX];
    uint32_t tx_head;
    uint32_t tx_count;
    uint32_t tsr;
    uint32_t tsr_busy;
    uint32_t thre_int;        /* the THRE interrupt, latched */
    /* the modem side */
    uint32_t modem_lines;     /* the external inputs, at their MSR[7:4] positions */
    uint32_t msr_inputs;      /* MSR[7:4] as the part shows them now */
    uint32_t msr_delta;       /* MSR[3:0] */
    /* what an observer reads; nothing here is a register */
    uint64_t time;            /* character times since reset */
    uint8_t line[SIM16550_LINE_MAX];   /* the bytes sent on SOUT, in order */
    uint32_t line_len;
    uint32_t line_lost;       /* bytes sent past SIM16550_LINE_MAX */
    uint32_t tx_overflows;    /* bytes written to a full transmitter */
    uint32_t rx_overruns;     /* characters lost to an overrun */
    uint32_t rx_dropped;      /* external characters that arrived while loopback disconnected SIN */
    uint32_t bus_errors;      /* accesses at an offset that is no register lane */
    uint32_t ignored_writes;  /* writes to LSR or MSR, which are for factory test only */
    uint32_t loop_dropped;    /* loopback characters sim_drop_loopback kept from the receiver */
    /* the fault knobs (off after sim_init) */
    uint32_t tx_stall;        /* sim_stall_tx: the shift register never finishes */
    uint32_t loop_drop;       /* sim_drop_loopback: loopback delivers nothing */
} sim16550;

/* NULL when the sim can model `cfg`, else why not. */
const char *sim_config_problem(const sim16550_config *cfg);
/* Power-on reset into `cfg`'s placement: 0, or -1 (and dev untouched) for a config it cannot model. */
int sim_init(sim16550 *dev, const sim16550_config *cfg);

/* A bus access at `offset` bytes from the device base, through section 1.1's mux. */
uint32_t sim_read(sim16550 *dev, uint32_t offset);
void sim_write(sim16550 *dev, uint32_t offset, uint32_t val);
/* What a read at `offset` would return, with none of its effects. */
uint32_t sim_peek(const sim16550 *dev, uint32_t offset);
/* The register an access at `offset` reaches now ("-" for none). */
const char *sim_reg_name(const sim16550 *dev, uint32_t offset, int write);

/* Advance time by n character times: the shift register finishes a character and takes the next, and
 * the receiver's timeout counts. */
void sim_tick(sim16550 *dev, uint32_t n_char_times);
/* A character finishes arriving on SIN now, with err_bits (SIM16550_PE | _FE | _BI). */
void sim_inject(sim16550 *dev, uint32_t byte, uint32_t err_bits);
/* The external modem inputs CTS, DSR, RI and DCD, at their MSR[7:4] positions. */
void sim_set_modem(sim16550 *dev, uint32_t lines);
/* The fault knobs above, on (nonzero) or off; a stalled character resumes where it stopped. */
void sim_stall_tx(sim16550 *dev, uint32_t on);
void sim_drop_loopback(sim16550 *dev, uint32_t on);

/* The receiver's conditions, enabled or not: the trigger level reached (IIR 0x4) and the character
 * timeout due (IIR 0xC). IIR shows each only while IER[0] is set -- not in the polled mode (PC16550D
 * p.17, p.23) -- so an observer reads them here. */
uint32_t sim_trigger_bytes(const sim16550 *dev);
uint32_t sim_rda_pending(const sim16550 *dev);
uint32_t sim_timeout_pending(const sim16550 *dev);

#endif /* SIM16550_H */
