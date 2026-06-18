#include <stdint.h>
/* L5: a memory-mapped device register block + bitfields (the register-map/MMIO MVP). */
struct uart_regs {
    volatile uint32_t status;       /* RX/TX status word (MMIO) */
    volatile uint32_t control;      /* control word (MMIO) */
};
struct ctrl_bits {                  /* a packed control word, by value */
    uint32_t enable : 1;
    uint32_t parity : 2;
    uint32_t baud   : 5;
    uint32_t _rsvd  : 24;
};
/* MMIO write path: program the control register, read it back (exercises an MMIO store). */
uint32_t uart_configure(volatile struct uart_regs *regs, uint32_t word)
{
    regs->control = word;
    return regs->control;
}
/* MMIO read + bitfield decode (the entry tested for behaviour-equivalence). */
uint32_t uart_decode(volatile struct uart_regs *regs, struct ctrl_bits cfg)
{
    uint32_t st = regs->status;
    uint32_t rx_ready = (st >> 0) & 1;
    uint32_t tx_empty = (st >> 5) & 1;
    uint32_t speed = cfg.baud * cfg.enable;
    return speed + rx_ready + tx_empty * cfg.parity;
}
