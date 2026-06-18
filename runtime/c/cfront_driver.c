/* Phase D: a real register-map driver, ingested end-to-end by the plug-in C compiler.
 * Exercises the whole accumulated stack at once -- #include + macros (L7), typedef / enum / union /
 * bitfields (type model), volatile MMIO loads (L5), struct pointers (L3), and the call graph (L4 /
 * R18) -- through one entry function that the loop plans, hydrates, executes, and Clang-checks. */
#include "cfront_driver.h"

/* is the channel quiescent? state == IDLE and no remaining count (a call-graph leaf, R18). */
static uint32_t dma_quiescent(uint32_t status)
{
    uint32_t state = DMA_STATE(status);
    uint32_t remaining = DMA_COUNT(status);
    return (state == DMA_IDLE) & (remaining == 0);
}

/* fold a control word's whole/low views (exercises the union register-view). */
static uint32_t dma_word_fold(union dma_word w)
{
    return w.whole + w.low;
}

/* the entry: read the channel's MMIO status register, decode it with the header's field macros,
 * fold the config bitfields + a control word, and return a packed health code. Pure (reads only),
 * so it is behaviour-equivalent under Clang against the original on a seeded register block. */
uint32_t dma_health(volatile struct dma_channel *ch, struct dma_cfg cfg, union dma_word w)
{
    uint32_t st = ch->status;
    uint32_t quiet = dma_quiescent(st);
    uint32_t state = DMA_STATE(st);
    uint32_t hi = (cfg.prio == DMA_PRIO_HIGH);          /* 1 when high priority, else 0 */
    uint32_t boosted = state + hi * DMA_BUSY;
    uint32_t folded = dma_word_fold(w);
    return quiet + cfg.enable * boosted + folded * cfg.burst + DMA_ERROR;
}
