/* cmsis_gpio.h -- a CMSIS-style GPIO + clock-gate register map (the real ST/ARM header shape).
 *
 * Phase 3 breadth (real fixtures): the idioms an actual CMSIS device header ships that
 * uart_regs.h does not -- the `__IO` volatile qualifier macro, RESERVED padding arrays holding
 * register offsets stable, a WRITE-ONLY atomic set/reset register (BSRR: the hardware-atomic
 * bit-banding idiom -- no read-modify-write on the output register), a 2-bit-per-pin mode
 * field programmed by shift/mask read-modify-write, and a peripheral clock-gate register the
 * driver must touch FIRST (the RCC enable ordering every ST driver carries).
 */
#ifndef CMSIS_GPIO_H
#define CMSIS_GPIO_H

#define __IO volatile           /* the CMSIS I/O qualifier macro, verbatim */

/* pin mode field values (2 bits per pin in MODER) */
#define GPIO_MODE_INPUT   0u
#define GPIO_MODE_OUTPUT  1u
#define GPIO_MODE_AF      2u
#define GPIO_MODE_ANALOG  3u
#define GPIO_MODER_WIDTH  2u

/* the GPIO register block (offsets 0x00..0x2C, CMSIS layout) */
typedef struct gpio_regs {
    __IO uint32_t MODER;        /* 0x00 mode: 2 bits per pin        */
    __IO uint32_t OTYPER;       /* 0x04 output type                 */
    __IO uint32_t OSPEEDR;      /* 0x08 output speed                */
    __IO uint32_t PUPDR;        /* 0x0C pull-up/pull-down           */
    __IO uint32_t IDR;          /* 0x10 input data (read)           */
    __IO uint32_t ODR;          /* 0x14 output data                 */
    __IO uint32_t BSRR;         /* 0x18 bit set/reset (WRITE-ONLY: set = bit n, reset = bit n+16;
                                 *      hardware-atomic -- the driver never RMWs the ODR) */
    __IO uint32_t LCKR;         /* 0x1C lock                        */
    __IO uint32_t AFRL;         /* 0x20 alternate function low      */
    __IO uint32_t AFRH;         /* 0x24 alternate function high     */
    uint32_t RESERVED0[2];      /* 0x28..0x2C pad (keeps a later block's offsets stable) */
} gpio_regs_t;

/* the clock-gate block: a peripheral is DEAD until its AHB enable bit is set (the RCC idiom). */
typedef struct rcc_regs {
    __IO uint32_t CR;           /* 0x00 clock control       */
    uint32_t RESERVED1[11];     /* 0x04..0x2C               */
    __IO uint32_t AHB1ENR;      /* 0x30 AHB1 clock enables  */
} rcc_regs_t;

#endif /* CMSIS_GPIO_H */
