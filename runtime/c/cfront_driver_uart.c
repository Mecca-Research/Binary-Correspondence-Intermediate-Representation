/* Phase D: a real driver, end-to-end through the BCIR plug-in C compiler (no Python).
 *
 * Includes a vendor-style UART register map (uart_regs.h) and drives it: volatile MMIO registers,
 * a bitfield control word, an enum mode, an overlapping-config union, object-macro bit positions,
 * and a bounded status-poll loop. bcir_cpp expands the #include + #defines; bcir_cfront lowers +
 * verifies (R1-R18) + attests (C.2); the closed loop plans / hydrates / executes the straight-line
 * entry. The two rails (Python oracle + C) must agree on the structural summary, and the emitted C
 * must be Clang-behaviour-equivalent.
 */
#include "uart_regs.h"

/* The low 16-bit view of a config word, through the union (members overlap at offset 0). */
uint32_t uart_cfg_low(uart_cfg_t cfg)
{
    return cfg.low;
}

/* Poll the status register until TX-empty (bounded by a timeout), then push a byte; return the
 * spin count. Reads/writes are ordered MMIO; the loop re-reads SR each iteration. */
uint32_t uart_send(volatile uart_regs_t *u, uint32_t byte)
{
    uint32_t spins = 0u;
    uint32_t txe = (u->SR >> UART_SR_TXE) & 1u;
    while (txe == 0u) {
        spins = spins + 1u;
        if (spins > 8u) {
            txe = 1u;                 /* timeout: stop polling */
        }
        txe = txe | ((u->SR >> UART_SR_TXE) & 1u);
    }
    u->DR = byte;
    return spins;
}

/* Program the baud + control registers from a bitfield control word; return the mode field,
 * defaulting an OFF mode up to RX. Straight-line -> this is the entry the execute loop runs. */
uint32_t uart_configure(volatile uart_regs_t *u, struct uart_ctrl ctrl, uint32_t baud)
{
    u->BRR = baud & UART_BAUD_MASK;
    uint32_t cr = (ctrl.enable << UART_CR_EN) | (ctrl.mode << UART_CR_MODE);
    u->CR = cr;
    uint32_t mode = ctrl.mode + (ctrl.mode == UART_MODE_OFF);   /* OFF(0) -> RX(1); else unchanged */
    return mode;
}
