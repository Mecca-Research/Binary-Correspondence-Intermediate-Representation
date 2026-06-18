/* A memory-mapped UART register block + a packed control word (the register-map MVP).
 * Lowered by BOTH the Python oracle (bcir/frontends/cfront) and the C frontend
 * (bcir_cfront.c); the two rails must produce the same structural summary. */
struct uart_regs { volatile uint32_t status; volatile uint32_t control; };
struct ctrl_bits { uint32_t enable : 1; uint32_t parity : 2; uint32_t baud : 5; uint32_t rsvd : 24; };
uint32_t uart_decode(volatile struct uart_regs *regs, struct ctrl_bits cfg)
{
    uint32_t st = regs->status;
    uint32_t rx_ready = (st >> 0) & 1;
    uint32_t tx_empty = (st >> 5) & 1;
    uint32_t speed = cfg.baud * cfg.enable;
    return speed + rx_ready + tx_empty * cfg.parity;
}
