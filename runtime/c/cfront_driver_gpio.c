/* Phase 3 breadth: a CMSIS-style GPIO driver through the plug-in C compiler (no Python).
 *
 * Exercises the Phase-2 laws on the header shapes a real vendor ships (cmsis_gpio.h): __IO
 * volatile MMIO accesses, the RCC clock-gate-first ordering, a 2-bit-per-pin MODER
 * read-modify-write, the WRITE-ONLY hardware-atomic BSRR set/reset (never an ODR RMW), and a
 * bounded IDR poll. Both rails must compile it clean; the emitted C is Clang-behaviour-
 * equivalent; the ABI contract records each function's frame (R12).
 */
#include "cmsis_gpio.h"

/* Gate the port's clock on FIRST (the RCC ordering every ST driver carries), then program the
 * pin's 2-bit mode field: clear-then-set read-modify-write on MODER. Returns the new MODER. */
uint32_t gpio_mode_set(volatile rcc_regs_t *rcc, volatile gpio_regs_t *g,
                       uint32_t pin, uint32_t mode, uint32_t enbit)
{
    rcc->AHB1ENR = rcc->AHB1ENR | (1u << enbit);
    uint32_t shift = pin * GPIO_MODER_WIDTH;
    uint32_t moder = g->MODER;
    moder = moder & ~(3u << shift);
    moder = moder | ((mode & 3u) << shift);
    g->MODER = moder;
    return moder;
}

/* Drive a pin through the WRITE-ONLY BSRR: set = bit n, reset = bit n+16. Hardware-atomic --
 * the driver never read-modify-writes ODR, so concurrent pin owners cannot race. */
void gpio_write_pin(volatile gpio_regs_t *g, uint32_t pin, uint32_t level)
{
    if (level != 0u) {
        g->BSRR = 1u << pin;
    } else {
        g->BSRR = 1u << (pin + 16u);
    }
}

/* Bounded IDR poll: spin until the pin reads `want` or the budget runs out; return the spin
 * count (the uart_send polling idiom on the input register). */
uint32_t gpio_wait_pin(volatile gpio_regs_t *g, uint32_t pin, uint32_t want)
{
    uint32_t spins = 0u;
    uint32_t got = (g->IDR >> pin) & 1u;
    while (got != want) {
        spins = spins + 1u;
        if (spins > 16u) {
            got = want;               /* timeout: stop polling */
        }
        got = got | (((g->IDR >> pin) & 1u) & want);
    }
    return spins;
}
