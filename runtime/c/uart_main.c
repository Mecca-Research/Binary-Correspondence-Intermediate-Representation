/* uart_main.c -- the U2 harness: the polled UART driver against the simulated 16550 (blueprint slice U2).
 *
 * Three units make the program: the driver (uart_driver.c, or what either BCIR rail emits for it with
 * --linkable), the device (sim16550.c) and this harness, which prints one deterministic transcript.
 *
 * The register window. The driver addresses its registers through a `volatile uart_regs_t *`, and here
 * that pointer is a plain array. Its loads and stores reach the device because the driver unit -- and
 * only it -- is compiled with Clang's load and store tracing (-fsanitize-coverage=func,trace-loads,
 * trace-stores): the compiler calls __sanitizer_cov_loadN / __sanitizer_cov_storeN before each access,
 * and the callbacks below -- the accessor shim -- forward the window's accesses to sim_read and
 * sim_write. A load is answered before it executes (the shim writes the device's value into the window
 * first); a store is forwarded once it has executed (at the next callback, or when the driver call
 * returns), so the device sees every access, in program order, once. The driver itself knows nothing of
 * this: the unit that runs here runs on the silicon. A build without the tracing reaches the device
 * with no access at all, and this harness refuses it (exit 3) rather than print a transcript of a
 * plain array. The harness and the sim must not be traced: their own loads and stores would come back
 * through the shim.
 *
 * Time is the device's character time and passes only in sim_tick. The RX script advances it, and the
 * shim advances it by one character before each LSR read: the polled driver's only way to wait is to
 * poll LSR, so a poll is one character time of waiting. While `streaming` is set the shim also plays a
 * peer that never stops sending: a character arrives with each of those character times, so the
 * receiver's DR never clears. The transcript is a function of the driver's
 * register accesses and of nothing else -- no clock, no address, no host state -- so the program built
 * from the driver's source and from either rail's emit prints it byte for byte
 * (bcir/tests/test_uart_driver.py; tools/c/sections/uartdrv.sh; the golden runtime/c/uart_main.golden is
 * the axi16550 placement's).
 *
 *   uart_main            the transcript: probe, which leaves the FIFOs off; TX refused on a zeroed
 *                        state, then run before init under a plan's cap; five arguments init must
 *                        refuse before it touches the device; init with PG143's sequence and formula
 *                        (56000 baud, 8E2, trigger 8); the loopback self-test, with a byte the receiver
 *                        held from before it; TX of a 52-byte buffer under a plan cap of 64; flush; an
 *                        RX script (a burst then a gap that brings the character timeout due, a burst
 *                        to the trigger level, a parity error, an overrun), each drained; the self-test
 *                        again under a peer that never stops sending; then twice more from the probe,
 *                        its verdict forced to no FIFO (fifo_ok 0) and to a broken one (fifo_ok 1):
 *                        init into 16450 mode, the self-test, TX of a 40-byte buffer under a stale plan
 *                        cap of the FIFO depth, flush; and under the sim's fault knobs, after init in
 *                        FIFO mode again, TX against a transmitter that never drains and the self-test
 *                        against a loopback that delivers nothing, each giving up at the driver's bound
 *                        (their unchanging polls shown " xN" on one line); then, the transmitter stalled
 *                        again, the self-test's wait for THRE, init's wait for TEMT and the flush's,
 *                        each giving up at the same bound
 *   uart_main --script   the device alone, driven by commands on stdin (its spec-fidelity tests)
 *
 * Exit status: 0 the transcript or the script ran; 2 usage, a malformed script line, or a placement the
 * device cannot model; 3 the register window is not trapped; 4 the driver's register accesses do not fit
 * this placement (a driver laid out for another: hazard H-A).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sim16550.h"
#include "uart16550_regs.h"
#include "uart_driver.h"

/* The register window: big enough for every U0 layout (the AXI core's registers end at 0x1020) and for
 * the stray accesses of a driver laid out for another, which land in it as misfits, never past it. The
 * driver's lanes and the shim's memcpy share it the way a device window is shared: through accesses the
 * compiler may not merge or move. */
#define WINDOW_BYTES 0x2000u
_Static_assert(UART_SCR_OFF + UART_REG_WIDTH <= WINDOW_BYTES, "the placement's registers fit the window");

static union {
    uint64_t align;
    unsigned char bytes[WINDOW_BYTES];
} window;

static sim16550 dev;

/* --- the bus: every access the driver makes to the window, in order --------------------------------- */

#define TRACE_MAX 8192u

typedef struct bus_event {
    char kind;          /* R a read, W a write, M a misfit (no whole register lane) */
    uint32_t offset;    /* bytes from the device base */
    uint32_t value;
    uint32_t size;      /* bytes accessed */
    const char *name;   /* the register the device took it for */
    uint32_t repeat;    /* the reads this line stands for: more than 1 only while `collapsing` */
} bus_event;

static bus_event trace[TRACE_MAX];
static uint32_t trace_len;
static uint32_t trace_lost;
static uint32_t trace_shown;
static uint32_t bus_reads;
static uint32_t bus_writes;
static uint32_t bus_misfits;
static int pending;
static unsigned char *pending_at;
static uint32_t pending_size;
static int streaming;
/* While set, a read the same as the one recorded just before it and not yet shown -- a poll that finds
 * nothing changed, which a timeout makes tens of thousands of -- counts on that line, shown " xN", instead
 * of a line of its own; the timeout phases set it, so every other phase's trace is one line per access. */
static int collapsing;

/* What the never-stopping peer sends. */
#define STREAM_BYTE 0x7eu

static void record(char kind, uint32_t offset, uint32_t value, uint32_t size, const char *name)
{
    if (collapsing && kind == 'R' && trace_len > trace_shown) {
        bus_event *last = &trace[trace_len - 1u];
        if (last->kind == 'R' && last->offset == offset && last->value == value && last->size == size
            && strcmp(last->name, name) == 0) {
            last->repeat++;
            return;
        }
    }
    if (trace_len == TRACE_MAX) {
        trace_lost++;
        return;
    }
    trace[trace_len].kind = kind;
    trace[trace_len].offset = offset;
    trace[trace_len].value = value;
    trace[trace_len].size = size;
    trace[trace_len].name = name;
    trace[trace_len].repeat = 1u;
    trace_len++;
}

static uint32_t lane_get(const unsigned char *at, uint32_t size)
{
    uint8_t b;
    uint16_t h;
    uint32_t w;
    uint64_t d;
    switch (size) {
    case 1u:
        memcpy(&b, at, 1u);
        return b;
    case 2u:
        memcpy(&h, at, 2u);
        return h;
    case 4u:
        memcpy(&w, at, 4u);
        return w;
    case 8u:
        memcpy(&d, at, 8u);
        return (uint32_t)d;
    default:
        return 0u;
    }
}

static void lane_put(unsigned char *at, uint32_t size, uint32_t value)
{
    uint8_t b = (uint8_t)value;
    uint16_t h = (uint16_t)value;
    uint32_t w = value;
    uint64_t d = value;
    switch (size) {
    case 1u:
        memcpy(at, &b, 1u);
        break;
    case 2u:
        memcpy(at, &h, 2u);
        break;
    case 4u:
        memcpy(at, &w, 4u);
        break;
    case 8u:
        memcpy(at, &d, 8u);
        break;
    default:
        memset(at, 0, size);
        break;
    }
}

/* The store the driver made last has executed by now: forward it to the device. */
static void bus_flush(void)
{
    uint32_t offset;
    uint32_t value;
    if (!pending) {
        return;
    }
    pending = 0;
    offset = (uint32_t)(pending_at - window.bytes);
    value = lane_get(pending_at, pending_size);
    if (pending_size != UART_REG_WIDTH) {
        bus_misfits++;
        record('M', offset, value, pending_size, "-");
        return;
    }
    bus_writes++;
    record('W', offset, value, pending_size, sim_reg_name(&dev, offset, 1));
    sim_write(&dev, offset, value);
}

static int in_window(const unsigned char *at, uint32_t size)
{
    uintptr_t lo = (uintptr_t)window.bytes;
    uintptr_t a = (uintptr_t)at;
    return a >= lo && a - lo <= (uintptr_t)(WINDOW_BYTES - size);
}

/* One load or store of the traced unit, before it executes. Anything outside the window is the driver's
 * RAM -- its state, the caller's buffers -- and passes through. */
static void bus_access(void *addr, uint32_t size, int write)
{
    unsigned char *at = (unsigned char *)addr;
    uint32_t offset;
    uint32_t value;
    const char *name;
    bus_flush();
    if (!in_window(at, size)) {
        return;
    }
    offset = (uint32_t)(at - window.bytes);
    if (write) {
        pending = 1;
        pending_at = at;
        pending_size = size;
        return;
    }
    if (size != UART_REG_WIDTH) {
        bus_misfits++;
        record('M', offset, 0u, size, "-");
        lane_put(at, size, 0u);
        return;
    }
    if (offset == UART_LSR_OFF) {
        sim_tick(&dev, 1u); /* a poll of LSR waits one character time */
        if (streaming) {
            sim_inject(&dev, STREAM_BYTE, 0u); /* ... in which the peer's next character arrives */
        }
    }
    bus_reads++;
    name = sim_reg_name(&dev, offset, 0);
    value = sim_read(&dev, offset);
    record('R', offset, value, size, name);
    lane_put(at, size, value);
}

/* The callbacks -fsanitize-coverage=trace-loads,trace-stores calls, by access width. */
void __sanitizer_cov_load1(uint8_t *addr);
void __sanitizer_cov_load2(uint16_t *addr);
void __sanitizer_cov_load4(uint32_t *addr);
void __sanitizer_cov_load8(uint64_t *addr);
void __sanitizer_cov_load16(void *addr);
void __sanitizer_cov_store1(uint8_t *addr);
void __sanitizer_cov_store2(uint16_t *addr);
void __sanitizer_cov_store4(uint32_t *addr);
void __sanitizer_cov_store8(uint64_t *addr);
void __sanitizer_cov_store16(void *addr);

void __sanitizer_cov_load1(uint8_t *addr) { bus_access(addr, 1u, 0); }
void __sanitizer_cov_load2(uint16_t *addr) { bus_access(addr, 2u, 0); }
void __sanitizer_cov_load4(uint32_t *addr) { bus_access(addr, 4u, 0); }
void __sanitizer_cov_load8(uint64_t *addr) { bus_access(addr, 8u, 0); }
void __sanitizer_cov_load16(void *addr) { bus_access(addr, 16u, 0); }
void __sanitizer_cov_store1(uint8_t *addr) { bus_access(addr, 1u, 1); }
void __sanitizer_cov_store2(uint16_t *addr) { bus_access(addr, 2u, 1); }
void __sanitizer_cov_store4(uint32_t *addr) { bus_access(addr, 4u, 1); }
void __sanitizer_cov_store8(uint64_t *addr) { bus_access(addr, 8u, 1); }
void __sanitizer_cov_store16(void *addr) { bus_access(addr, 16u, 1); }

/* --- the transcript --------------------------------------------------------------------------------- */

#define NO_ERROR_BYTE 0xffffffffu

/* The placement as the generated header lays it out: the device is the one the driver was written for. */
static void placement_config(sim16550_config *cfg)
{
    static const uint32_t triggers[4] = {UART_CAP_RX_TRIGGER_BYTES};
    uint32_t k;
    memset(cfg, 0, sizeof *cfg);
    cfg->placement = uart_regmap_contract.placement;
    cfg->stride = uart_regmap_contract.stride;
    cfg->base_offset = uart_regmap_contract.base_offset;
    cfg->reg_width = UART_REG_WIDTH;
    cfg->fcr_reset = uart_regmap_contract.fcr_reset;
    cfg->fcr_gated = UART_CAP_FCR_GATED;
    cfg->fcr_impl_mask = UART_CAP_FCR_IMPL;
    cfg->fifo_depth = UART_CAP_FIFO_DEPTH;
    for (k = 0u; k < 4u; k++) {
        cfg->rx_triggers[k] = triggers[k];
    }
    cfg->has_scr = UART_CAP_SCR;
    cfg->has_afe = UART_CAP_AFE;
    cfg->has_modem = UART_CAP_MODEM;
}

static void phase(const char *name)
{
    printf("-- %s (t=%llu)\n", name, (unsigned long long)dev.time);
}

static void show_trace(void)
{
    for (; trace_shown < trace_len; trace_shown++) {
        const bus_event *e = &trace[trace_shown];
        if (e->kind == 'M') {
            printf("  M 0x%04x a %u-byte access: the lanes are %u bytes\n", (unsigned)e->offset,
                   (unsigned)e->size, (unsigned)UART_REG_WIDTH);
        } else if (e->repeat > 1u) {
            printf("  %c 0x%04x %-3s 0x%02x x%u\n", e->kind, (unsigned)e->offset, e->name,
                   (unsigned)e->value, (unsigned)e->repeat);
        } else {
            printf("  %c 0x%04x %-3s 0x%02x\n", e->kind, (unsigned)e->offset, e->name,
                   (unsigned)e->value);
        }
    }
}

static void print_bytes(const uint8_t *bytes, uint32_t n)
{
    uint32_t k;
    putchar('"');
    for (k = 0u; k < n; k++) {
        int c = bytes[k];
        if (c == '\r') {
            fputs("\\r", stdout);
        } else if (c == '\n') {
            fputs("\\n", stdout);
        } else if (c == '"' || c == '\\') {
            printf("\\%c", c);
        } else if (c >= 0x20 && c < 0x7f) {
            putchar(c);
        } else {
            printf("\\x%02x", (unsigned)c);
        }
    }
    putchar('"');
}

static const char *status_text(uint32_t status)
{
    if (status == UART_OK) {
        return "ok";
    }
    if (status == UART_ERR_NOT_READY) {
        return "not ready";
    }
    if (status == UART_ERR_RX_STUCK) {
        return "the receiver never emptied";
    }
    if (status == UART_ERR_TIMEOUT) {
        return "timed out";
    }
    if (status == UART_ERR_BUSY) {
        return "busy";
    }
    return "refused";
}

static const char *probe_text(const uart_state *st)
{
    if (st->fifo_ok == 2u) {
        return "16550A";
    }
    if (st->fifo_ok == 1u) {
        return "16550 with a broken FIFO";
    }
    return st->scr_ok != 0u ? "16450" : "8250";
}

/* An argument uart_init must refuse: its status, and whether it touched the device first. */
static void init_refusal(volatile uart_regs_t *u, uart_state *st, const char *what, uint32_t divisor,
                         uint32_t format, uint32_t trigger)
{
    const uint32_t before = bus_reads + bus_writes + bus_misfits;
    uint32_t status = uart_init(u, st, divisor, format, trigger);
    bus_flush();
    show_trace();
    printf("init(%s): returns %u, %s\n", what, (unsigned)status,
           bus_reads + bus_writes + bus_misfits == before ? "the device untouched" : "the device TOUCHED");
}

static void rx_arrivals(const char *bytes, uint32_t from, uint32_t to, uint32_t error_at,
                        uint32_t error_bits)
{
    uint32_t k;
    for (k = from; k < to; k++) {
        if (k > 0u) {
            sim_tick(&dev, 1u); /* back to back: one character time apart */
        }
        sim_inject(&dev, (uint8_t)bytes[k], k == error_at ? error_bits : 0u);
    }
}

static void rx_status(void)
{
    printf("  device: t=%llu, %u held, trigger level (%u) reached: %s, character timeout due: %s, "
           "lsr 0x%02x\n",
           (unsigned long long)dev.time, (unsigned)dev.rx_count, (unsigned)sim_trigger_bytes(&dev),
           sim_rda_pending(&dev) != 0u ? "yes" : "no", sim_timeout_pending(&dev) != 0u ? "yes" : "no",
           (unsigned)sim_peek(&dev, UART_LSR_OFF));
}

static void no_fifo_phase(uint32_t verdict, const char *step)
{
    printf("-- fifo_ok %u %s (t=%llu)\n", (unsigned)verdict, step, (unsigned long long)dev.time);
}

/* Whether the device has its FIFOs on, and what IIR would read (IIR[7:6] = 11 with them on, 00 off). */
static void fifos_line(void)
{
    printf("  device: fifos %s, iir 0x%02x\n", (dev.fcr & UART_FCR_FIFOEN_MASK) != 0u ? "on" : "off",
           (unsigned)sim_peek(&dev, UART_IIR_OFF));
}

/* A state uart_tx must refuse: its status, the bytes it says it sent, and whether it touched the device
 * first. */
static void tx_refusal(volatile uart_regs_t *u, uart_state *st, const char *what)
{
    static const uint8_t byte[] = "!";
    const uint32_t before = bus_reads + bus_writes + bus_misfits;
    uint32_t sent = 0xffffffffu; /* the driver must say 0 */
    uint32_t status = uart_tx(u, st, byte, 1u, &sent);
    bus_flush();
    show_trace();
    printf("tx(%s): returns %u (%s), %u bytes sent, %s\n", what, (unsigned)status, status_text(status),
           (unsigned)sent,
           bus_reads + bus_writes + bus_misfits == before ? "the device untouched" : "the device TOUCHED");
}

/* uart_tx of a buffer under the cap the state holds: the trace, then its status, the bytes sent and the
 * TX overflows the device counted meanwhile. Returns the bytes sent. */
static uint32_t tx_report(volatile uart_regs_t *u, uart_state *st, const uint8_t *message, uint32_t length,
                          const char *cap_what)
{
    const uint32_t cap = st->tx_burst;
    const uint32_t overflows = dev.tx_overflows;
    uint32_t sent = 0u;
    uint32_t status = uart_tx(u, st, message, length, &sent);
    bus_flush();
    show_trace();
    printf("tx: returns %u (%s); %u of %u bytes sent under %s %u; tx overflows %u\n", (unsigned)status,
           status_text(status), (unsigned)sent, (unsigned)length, cap_what, (unsigned)cap,
           (unsigned)(dev.tx_overflows - overflows));
    return sent;
}

/* uart_flush, then what the line carried from byte `from` on: the trace, the busy polls and the bytes,
 * and the status on a line of its own when the flush gave up. Returns whether they are the buffer. */
static int flush_report(volatile uart_regs_t *u, uint32_t from, const uint8_t *message, uint32_t length)
{
    uint32_t busy = 0u;
    const uint32_t status = uart_flush(u, &busy);
    uint32_t carried;
    int line_ok;
    bus_flush();
    show_trace();
    carried = dev.line_len - from;
    line_ok = carried == length && memcmp(dev.line + from, message, length) == 0;
    printf("flush: %u busy polls; the line carried %u bytes ", (unsigned)busy, (unsigned)carried);
    print_bytes(dev.line + from, carried);
    printf(", %s\n", line_ok ? "the buffer" : "NOT the buffer");
    if (status != UART_OK) {
        printf("  flush: returns %u (%s)\n", (unsigned)status, status_text(status));
    }
    return line_ok;
}

/* A part whose probe found no working FIFO (section 3.1): fifo_ok 0, none -- a 16450 or an 8250 -- or 1,
 * the original 16550's broken FIFO. The sim answers the probe as a 16550A, so this run forces the verdict
 * after the probe, in a state of its own; the probe has left the FIFOs off, and init must keep them off
 * -- 16450 mode, where THRE means the one-byte holding register is empty (PC16550D p.22). The self-test
 * must pass with no FIFO to reset, and TX is handed a stale plan cap -- the FIFO depth, the cap a plan
 * made for the FIFO leaves -- that the driver must clamp to one byte: the line must carry the buffer with
 * no TX overflow. */
static void no_fifo_run(volatile uart_regs_t *u, uint32_t verdict, uint32_t divisor, uint32_t format)
{
    static const uint8_t message[] = "16450 mode: a byte per THRE, none lost\r\n";
    const uint32_t length = (uint32_t)(sizeof message - 1u);
    uart_state st;
    uint32_t status;
    uint32_t from;

    memset(&st, 0, sizeof st);
    no_fifo_phase(verdict, "probe: the sim answers as a 16550A; the run forces the verdict after it");
    status = uart_probe(u, &st);
    bus_flush();
    show_trace();
    st.fifo_ok = verdict;
    printf("probe: returns %u; fifo_ok forced to %u (%s), scr_ok %u, fcr_shadow 0x%02x\n", (unsigned)status,
           (unsigned)st.fifo_ok, probe_text(&st), (unsigned)st.scr_ok, (unsigned)st.fcr_shadow);
    fifos_line();

    no_fifo_phase(verdict, "init");
    status = uart_init(u, &st, divisor, format, UART_RX_TRIGGER_8);
    bus_flush();
    show_trace();
    printf("init: returns %u (%s); trigger %u bytes, fcr_shadow 0x%02x, tx_burst %u\n", (unsigned)status,
           status_text(status), (unsigned)st.trigger, (unsigned)st.fcr_shadow, (unsigned)st.tx_burst);
    fifos_line();

    no_fifo_phase(verdict, "selftest");
    status = uart_selftest(u, &st);
    bus_flush();
    show_trace();
    printf("selftest: returns %u (%s); mcr 0x%02x after it, fcr_shadow 0x%02x\n", (unsigned)status,
           status_text(status), (unsigned)dev.mcr, (unsigned)st.fcr_shadow);

    no_fifo_phase(verdict, "tx");
    st.tx_burst = UART_CAP_FIFO_DEPTH; /* stale: a plan made for the FIFO; the driver clamps it */
    from = dev.line_len;
    (void)tx_report(u, &st, message, length, "a stale plan burst cap of");

    no_fifo_phase(verdict, "flush");
    (void)flush_report(u, from, message, length);
}

static uint32_t rx_drain(volatile uart_regs_t *u, uart_state *st, uint32_t totals[4])
{
    uint8_t got[64];
    uint32_t counts[4] = {0u, 0u, 0u, 0u};
    uint32_t n;
    uint32_t k;
    n = uart_rx_drain(u, st, got, (uint32_t)sizeof got, counts);
    bus_flush();
    show_trace();
    printf("drain: %u bytes ", (unsigned)n);
    print_bytes(got, n);
    printf("; OE %u PE %u FE %u BI %u\n", (unsigned)counts[0], (unsigned)counts[1],
           (unsigned)counts[2], (unsigned)counts[3]);
    for (k = 0u; k < 4u; k++) {
        totals[k] += counts[k];
    }
    return n;
}

static int transcript_main(void)
{
    static const uint8_t message[] = "U2: the polled 16550 driver, emitted by both rails\r\n";
    static const uint8_t early[] = "pre-init\r\n";
    static const uint8_t pair[] = "<>";
    const uint32_t length = (uint32_t)(sizeof message - 1u);
    const uint32_t early_length = (uint32_t)(sizeof early - 1u);
    const uint32_t pair_length = (uint32_t)(sizeof pair - 1u);
    volatile uart_regs_t *u = (volatile uart_regs_t *)window.bytes;
    const uint32_t ref = uart_regmap_contract.ref_clk_hz;
    const uint32_t divisor = ref / (16u * 56000u); /* PG143 pp.52-53: AXI CLK / (16 x baud), truncated */
    const uint32_t format = UART_LCR_WLS_MASK | UART_LCR_STB_MASK | UART_LCR_PEN_MASK | UART_LCR_EPS_MASK;
    sim16550_config cfg;
    uart_state st;
    uart_state zeroed;
    uint32_t totals[4] = {0u, 0u, 0u, 0u};
    uint32_t probe;
    uint32_t init;
    uint32_t selftest;
    uint32_t status;
    uint32_t sent;
    uint32_t from;
    uint32_t reinit;
    uint32_t stalled;
    uint32_t busy = 0u;
    uint32_t received = 0u;
    int line_ok;

    placement_config(&cfg);
    if (sim_init(&dev, &cfg) != 0) {
        fprintf(stderr, "uart_main: the sim cannot model placement %s: %s\n", cfg.placement,
                sim_config_problem(&cfg));
        return 2;
    }
    memset(&st, 0, sizeof st);
    printf("uart_main: the polled 16550 driver against sim16550 (blueprint slice U2)\n");
    printf("layout: placement %s, stride %u, base 0x%x, %u-byte lanes, reference clock %u Hz\n",
           cfg.placement, (unsigned)cfg.stride, (unsigned)cfg.base_offset, (unsigned)cfg.reg_width,
           (unsigned)ref);
    printf("device: fifo %u bytes, rx triggers %u/%u/%u/%u, fcr reset 0x%02x, fcr gated %u, "
           "fcr bits 0x%02x, scr %u, modem %u\n",
           (unsigned)cfg.fifo_depth, (unsigned)cfg.rx_triggers[0], (unsigned)cfg.rx_triggers[1],
           (unsigned)cfg.rx_triggers[2], (unsigned)cfg.rx_triggers[3], (unsigned)cfg.fcr_reset,
           (unsigned)cfg.fcr_gated, (unsigned)cfg.fcr_impl_mask, (unsigned)cfg.has_scr,
           (unsigned)cfg.has_modem);

    phase("probe");
    probe = uart_probe(u, &st);
    bus_flush();
    if (bus_reads + bus_writes + bus_misfits == 0u) {
        fprintf(stderr, "uart_main: the register window is not trapped: the driver unit must be "
                        "compiled with -fsanitize-coverage=func,trace-loads,trace-stores\n");
        return 3;
    }
    show_trace();
    if (bus_misfits != 0u || dev.bus_errors != 0u) {
        fprintf(stderr, "uart_main: the driver's register accesses do not fit placement %s: it was "
                        "laid out for another (hazard H-A)\n",
                cfg.placement);
        return 4;
    }
    printf("probe: returns %u (%s); fifo_ok %u, scr_ok %u, fcr_shadow 0x%02x\n", (unsigned)probe,
           probe_text(&st), (unsigned)st.fifo_ok, (unsigned)st.scr_ok, (unsigned)st.fcr_shadow);
    fifos_line();

    phase("tx refused: a state no init has filled");
    memset(&zeroed, 0, sizeof zeroed);
    tx_refusal(u, &zeroed, "a zeroed state");

    phase("tx before init: the probe left the FIFOs off; a plan's cap in the state");
    st.tx_burst = UART_CAP_FIFO_DEPTH; /* a plan's cap before any init: the probe's shadow says no FIFO */
    from = dev.line_len;
    (void)tx_report(u, &st, early, early_length, "a plan burst cap of");
    (void)flush_report(u, from, early, early_length);

    phase("init refusals");
    init_refusal(u, &st, "divisor 0", 0u, format, UART_RX_TRIGGER_8);
    init_refusal(u, &st, "divisor 0x10000", 0x10000u, format, UART_RX_TRIGGER_8);
    init_refusal(u, &st, "format 0x83, with DLAB", divisor, UART_LCR_WLS_MASK | UART_LCR_DLAB_MASK,
                 UART_RX_TRIGGER_8);
    init_refusal(u, &st, "format 0x43, with set break", divisor, UART_LCR_WLS_MASK | UART_LCR_SB_MASK,
                 UART_RX_TRIGGER_8);
    init_refusal(u, &st, "trigger code 0x81", divisor, format, 0x81u);

    phase("init");
    init = uart_init(u, &st, divisor, format, UART_RX_TRIGGER_8);
    bus_flush();
    show_trace();
    printf("init: returns %u (%s); divisor %u for 56000 baud from %u Hz (PG143's formula): %u baud; "
           "format 0x%02x (8E2); trigger %u bytes, fcr_shadow 0x%02x, tx_burst %u\n",
           (unsigned)init, status_text(init), (unsigned)divisor, (unsigned)ref,
           (unsigned)(divisor != 0u ? ref / (16u * divisor) : 0u), (unsigned)format,
           (unsigned)st.trigger, (unsigned)st.fcr_shadow, (unsigned)st.tx_burst);
    printf("  device: divisor latch 0x%04x, lcr 0x%02x, mcr 0x%02x, ier 0x%02x, fifos %s, trigger %u bytes\n",
           (unsigned)((dev.dlm << 8) | dev.dll), (unsigned)dev.lcr, (unsigned)dev.mcr, (unsigned)dev.ier,
           (dev.fcr & UART_FCR_FIFOEN_MASK) != 0u ? "on" : "off", (unsigned)sim_trigger_bytes(&dev));

    phase("selftest: a byte the receiver held from before it");
    sim_inject(&dev, 0x5au, 0u); /* a character that arrived after init: the drain must take it */
    selftest = uart_selftest(u, &st);
    bus_flush();
    show_trace();
    printf("selftest: returns %u (%s); mcr 0x%02x after it, fcr_shadow 0x%02x\n", (unsigned)selftest,
           status_text(selftest), (unsigned)dev.mcr, (unsigned)st.fcr_shadow);

    phase("tx");
    st.tx_burst = 64u; /* a plan asking for more than the FIFO holds: the driver clamps it */
    from = dev.line_len;
    sent = tx_report(u, &st, message, length, "a plan burst cap of");

    phase("flush");
    line_ok = flush_report(u, from, message, length);

    phase("rx a: a 3-byte burst, then a gap of 4 character times");
    rx_arrivals("abc", 0u, 3u, NO_ERROR_BYTE, 0u);
    sim_tick(&dev, 3u);
    rx_status();
    sim_tick(&dev, 1u);
    rx_status();
    received += rx_drain(u, &st, totals);

    phase("rx b: an 8-byte burst at trigger 8");
    rx_arrivals("RX-burst", 0u, 7u, NO_ERROR_BYTE, 0u);
    rx_status();
    rx_arrivals("RX-burst", 7u, 8u, NO_ERROR_BYTE, 0u);
    rx_status();
    received += rx_drain(u, &st, totals);

    phase("rx c: a 5-byte burst, its third byte with a parity error");
    rx_arrivals("ABCDE", 0u, 5u, 2u, SIM16550_PE);
    rx_status();
    received += rx_drain(u, &st, totals);

    phase("rx d: 17 bytes into the 16-byte FIFO");
    rx_arrivals("0123456789abcdefg", 0u, 17u, NO_ERROR_BYTE, 0u);
    rx_status();
    received += rx_drain(u, &st, totals);

    phase("stuck selftest: a peer that sends a character every character time");
    streaming = 1;
    status = uart_selftest(u, &st);
    streaming = 0;
    bus_flush();
    show_trace();
    printf("selftest: returns %u (%s); mcr 0x%02x after it, fcr_shadow 0x%02x\n", (unsigned)status,
           status_text(status), (unsigned)dev.mcr, (unsigned)st.fcr_shadow);
    printf("  device: the receiver holds %u after it\n", (unsigned)dev.rx_count);

    no_fifo_run(u, 0u, divisor, format);
    no_fifo_run(u, 1u, divisor, format);

    phase("tx timeout: a transmitter that never drains");
    reinit = uart_init(u, &st, divisor, format, UART_RX_TRIGGER_8); /* FIFO mode again, for this state */
    bus_flush();
    show_trace();
    printf("init: returns %u (%s); fcr_shadow 0x%02x, tx_burst %u\n", (unsigned)reinit,
           status_text(reinit), (unsigned)st.fcr_shadow, (unsigned)st.tx_burst);
    sim_stall_tx(&dev, 1u);
    collapsing = 1;
    from = dev.line_len;
    stalled = tx_report(u, &st, message, length, "init's burst cap of");
    collapsing = 0;
    printf("  device: the line carried %u bytes meanwhile; %u wait in the TX FIFO, %u in the shift register\n",
           (unsigned)(dev.line_len - from), (unsigned)dev.tx_count, (unsigned)dev.tsr_busy);
    sim_stall_tx(&dev, 0u); /* the transmitter resumes: what it accepted goes out */
    (void)flush_report(u, from, message, stalled);

    phase("selftest timeout: a loopback that delivers nothing");
    sim_drop_loopback(&dev, 1u);
    collapsing = 1;
    status = uart_selftest(u, &st);
    collapsing = 0;
    sim_drop_loopback(&dev, 0u);
    bus_flush();
    show_trace();
    printf("selftest: returns %u (%s); mcr 0x%02x after it, fcr_shadow 0x%02x\n", (unsigned)status,
           status_text(status), (unsigned)dev.mcr, (unsigned)st.fcr_shadow);
    printf("  device: the loopback kept %u characters from the receiver\n", (unsigned)dev.loop_dropped);

    phase("selftest stalled: a transmitter that never drains, under the self-test");
    sim_stall_tx(&dev, 1u);
    from = dev.line_len;
    (void)tx_report(u, &st, pair, pair_length, "init's burst cap of"); /* one shifting, one waiting */
    collapsing = 1;
    status = uart_selftest(u, &st);
    collapsing = 0;
    bus_flush();
    show_trace();
    printf("selftest: returns %u (%s); mcr 0x%02x after it, fcr_shadow 0x%02x\n", (unsigned)status,
           status_text(status), (unsigned)dev.mcr, (unsigned)st.fcr_shadow);
    sim_stall_tx(&dev, 0u); /* the shifting character goes out; the waiting one went with the FIFO reset */
    (void)flush_report(u, from, pair, 1u);

    phase("init busy: a transmitter that never goes idle");
    sim_stall_tx(&dev, 1u);
    from = dev.line_len;
    (void)tx_report(u, &st, pair, 1u, "init's burst cap of"); /* the one character, shifting forever */
    collapsing = 1;
    status = uart_init(u, &st, divisor, format, UART_RX_TRIGGER_8);
    collapsing = 0;
    bus_flush();
    show_trace();
    printf("init: returns %u (%s); fcr_shadow 0x%02x, tx_burst %u\n", (unsigned)status, status_text(status),
           (unsigned)st.fcr_shadow, (unsigned)st.tx_burst);

    phase("flush busy: the transmitter still never goes idle");
    collapsing = 1;
    status = uart_flush(u, &busy);
    collapsing = 0;
    bus_flush();
    show_trace();
    printf("stalled flush: returns %u (%s); %u busy polls\n", (unsigned)status, status_text(status),
           (unsigned)busy);
    sim_stall_tx(&dev, 0u); /* the character finishes: the line goes idle */
    (void)flush_report(u, from, pair, 1u);

    phase("end");
    show_trace();
    printf("lsr 0x%02x; device: tx overflows %u, rx overruns %u, rx dropped %u, bus errors %u, "
           "ignored writes %u, line lost %u\n",
           (unsigned)sim_peek(&dev, UART_LSR_OFF), (unsigned)dev.tx_overflows, (unsigned)dev.rx_overruns,
           (unsigned)dev.rx_dropped, (unsigned)dev.bus_errors, (unsigned)dev.ignored_writes,
           (unsigned)dev.line_lost);
    printf("bus: %u register accesses (%u reads, %u writes), %u misfits, %u not traced\n",
           (unsigned)(bus_reads + bus_writes), (unsigned)bus_reads, (unsigned)bus_writes,
           (unsigned)bus_misfits, (unsigned)trace_lost);
    printf("verdict: probe %s, init %s, selftest %s, tx %u/%u %s, rx %u bytes with OE %u PE %u FE %u BI %u\n",
           probe_text(&st), status_text(init), status_text(selftest), (unsigned)sent, (unsigned)length,
           line_ok ? "on the line" : "NOT on the line", (unsigned)received, (unsigned)totals[0],
           (unsigned)totals[1], (unsigned)totals[2], (unsigned)totals[3]);
    return 0;
}

/* --- the script console: the device alone ----------------------------------------------------------- */

/* Commands, one per line, numbers in C notation (0x.. hex):
 *   config NAME STRIDE BASE WIDTH FCR_RESET FCR_GATED FCR_IMPL DEPTH T0 T1 T2 T3 SCR AFE MODEM
 *                          reset into that placement: "config NAME", or "config refused: WHY"
 *   r OFF                  sim_read: "r OFF VALUE"
 *   peek OFF               sim_peek: "peek OFF VALUE"
 *   w OFF VALUE            sim_write
 *   tick N                 sim_tick
 *   inject BYTE ERR_BITS   sim_inject
 *   modem LINES            sim_set_modem
 *   status                 the receiver's and transmitter's state, one line
 *   line                   the bytes sent on SOUT, in hex
 *   counters               the device's counters, one line
 * A blank line or one starting with # is skipped. */

#define SCRIPT_TOKENS 20

static int script_error(unsigned long lineno, const char *why)
{
    printf("error: line %lu: %s\n", lineno, why);
    return 2;
}

static int number(const char *text, uint32_t *out)
{
    char *end;
    unsigned long long value;
    if (text[0] < '0' || text[0] > '9') {
        return 0;
    }
    value = strtoull(text, &end, 0);
    if (*end != '\0' || value > 0xffffffffull) {
        return 0;
    }
    *out = (uint32_t)value;
    return 1;
}

static int script_main(void)
{
    static char name[64];
    char text[512];
    char *tok[SCRIPT_TOKENS];
    sim16550_config cfg;
    unsigned long lineno = 0ul;
    int ready = 0;
    uint32_t a = 0u;
    uint32_t b = 0u;
    memset(&cfg, 0, sizeof cfg);
    while (fgets(text, (int)sizeof text, stdin) != NULL) {
        int n = 0;
        char *s = strtok(text, " \t\r\n");
        lineno++;
        while (s != NULL && n < SCRIPT_TOKENS) {
            tok[n] = s;
            n++;
            s = strtok(NULL, " \t\r\n");
        }
        if (n == 0 || tok[0][0] == '#') {
            continue;
        }
        if (strcmp(tok[0], "config") == 0) {
            uint32_t v[14];
            int k;
            if (n != 16) {
                return script_error(lineno, "config takes a name and 14 numbers");
            }
            for (k = 0; k < 14; k++) {
                if (!number(tok[2 + k], &v[k])) {
                    return script_error(lineno, "config takes a name and 14 numbers");
                }
            }
            snprintf(name, sizeof name, "%s", tok[1]);
            cfg.placement = name;
            cfg.stride = v[0];
            cfg.base_offset = v[1];
            cfg.reg_width = v[2];
            cfg.fcr_reset = v[3];
            cfg.fcr_gated = v[4];
            cfg.fcr_impl_mask = v[5];
            cfg.fifo_depth = v[6];
            for (k = 0; k < 4; k++) {
                cfg.rx_triggers[k] = v[7 + k];
            }
            cfg.has_scr = v[11];
            cfg.has_afe = v[12];
            cfg.has_modem = v[13];
            if (sim_init(&dev, &cfg) != 0) {
                printf("config refused: %s\n", sim_config_problem(&cfg));
                ready = 0;
            } else {
                printf("config %s\n", name);
                ready = 1;
            }
            continue;
        }
        if (!ready) {
            return script_error(lineno, "no device: config first");
        }
        if (strcmp(tok[0], "r") == 0 && n == 2 && number(tok[1], &a)) {
            printf("r 0x%x 0x%02x\n", (unsigned)a, (unsigned)sim_read(&dev, a));
        } else if (strcmp(tok[0], "peek") == 0 && n == 2 && number(tok[1], &a)) {
            printf("peek 0x%x 0x%02x\n", (unsigned)a, (unsigned)sim_peek(&dev, a));
        } else if (strcmp(tok[0], "w") == 0 && n == 3 && number(tok[1], &a) && number(tok[2], &b)) {
            sim_write(&dev, a, b);
        } else if (strcmp(tok[0], "tick") == 0 && n == 2 && number(tok[1], &a)) {
            sim_tick(&dev, a);
        } else if (strcmp(tok[0], "inject") == 0 && n == 3 && number(tok[1], &a) && number(tok[2], &b)) {
            sim_inject(&dev, a, b);
        } else if (strcmp(tok[0], "modem") == 0 && n == 2 && number(tok[1], &a)) {
            sim_set_modem(&dev, a);
        } else if (strcmp(tok[0], "status") == 0 && n == 1) {
            printf("status time=%llu rx=%u tx=%u tsr=%u idle=%u rda=%u timeout=%u thre_int=%u "
                   "fcr=0x%02x trigger=%u\n",
                   (unsigned long long)dev.time, (unsigned)dev.rx_count, (unsigned)dev.tx_count,
                   (unsigned)dev.tsr_busy, (unsigned)dev.idle, (unsigned)sim_rda_pending(&dev),
                   (unsigned)sim_timeout_pending(&dev), (unsigned)dev.thre_int, (unsigned)dev.fcr,
                   (unsigned)sim_trigger_bytes(&dev));
        } else if (strcmp(tok[0], "line") == 0 && n == 1) {
            uint32_t k;
            printf("line %u", (unsigned)dev.line_len);
            for (k = 0u; k < dev.line_len; k++) {
                printf(" %02x", (unsigned)dev.line[k]);
            }
            printf("\n");
        } else if (strcmp(tok[0], "counters") == 0 && n == 1) {
            printf("counters tx_overflows=%u rx_overruns=%u rx_dropped=%u bus_errors=%u "
                   "ignored_writes=%u line_lost=%u\n",
                   (unsigned)dev.tx_overflows, (unsigned)dev.rx_overruns, (unsigned)dev.rx_dropped,
                   (unsigned)dev.bus_errors, (unsigned)dev.ignored_writes, (unsigned)dev.line_lost);
        } else {
            return script_error(lineno, "unknown command or bad operands");
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--script") == 0) {
        return script_main();
    }
    if (argc != 1) {
        fprintf(stderr, "usage: uart_main [--script]\n");
        return 2;
    }
    return transcript_main();
}
