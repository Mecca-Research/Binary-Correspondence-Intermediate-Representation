/* uart_regs.h -- a vendor-style memory-mapped UART register map (CMSIS-flavored).
 *
 * Phase D compiler fixture: this has the shape of a header a real SoC vendor ships. The source beside it
 * (cfront_driver_uart.c) includes it and is driven END-TO-END through the BCIR plug-in C
 * compiler with no Python:  bcir_cpp (the #include + #defines)  ->  bcir_cfront (lex / parse /
 * lower / verify)  ->  claim graph  ->  bcir_plan  ->  bcir_hydrate  ->  bcir_exec.
 *
 * It exercises the accumulated ladder: L7 preprocessor (#ifndef guard, object macros, #include),
 * typedef / enum / union (type-model breadth), L5 volatile MMIO registers, L2 bitfields, and the
 * L8 by-value aggregate ABI.
 */
#ifndef UART_REGS_H
#define UART_REGS_H

/* status-register bit positions (the kind of object macros a real vendor header ships) */
#define UART_SR_RXNE    0u      /* receive buffer not empty */
#define UART_SR_TXE     7u      /* transmit buffer empty    */
#define UART_SR_BUSY    16u     /* line busy                */
/* control-register field positions + a baud mask */
#define UART_CR_EN      0u      /* enable bit               */
#define UART_CR_MODE    1u      /* 2-bit mode field @ bit 1 */
#define UART_BAUD_MASK  0xFFFFu

/* operating modes -- an enum folded to integer constants at parse time (incl. a constant-expr). */
typedef enum uart_mode {
    UART_MODE_OFF    = 0,
    UART_MODE_RX     = 1,
    UART_MODE_TX     = 2,
    UART_MODE_DUPLEX = UART_MODE_RX + UART_MODE_TX   /* 3 -- a constant-expression initializer */
} uart_mode_t;

/* the memory-mapped register block: `volatile` == every access is an ordered MMIO load/store. */
typedef struct uart_regs {
    volatile uint32_t SR;     /* 0x00 status  */
    volatile uint32_t DR;     /* 0x04 data    */
    volatile uint32_t BRR;    /* 0x08 baud    */
    volatile uint32_t CR;     /* 0x0C control */
} uart_regs_t;

/* a control word, addressable as the whole 32-bit word OR its low 16-bit view (the classic
 * overlapping-register union; both members live at offset 0). */
typedef union uart_cfg {
    uint32_t word;
    uint16_t low;
} uart_cfg_t;

/* the bitfield view of the control register (a real header's packed CR layout). */
struct uart_ctrl {
    uint32_t enable : 1;
    uint32_t mode   : 2;
    uint32_t parity : 1;
    uint32_t baud   : 16;
    uint32_t rsvd   : 12;
};

#endif /* UART_REGS_H */
