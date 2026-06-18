/* A vendor-style memory-mapped DMA-channel register map -- the Phase D end-to-end driver target.
 * It is ingested through the FULL plug-in C compiler with no hand-written claim graph:
 *   bcir_cpp (this #include + the field-extract macros) -> bcir_cfront (typedef / enum / union /
 *   bitfields / volatile MMIO / the call graph) -> R1-R18 verify -> emit -> Clang behaviour check.
 * A real header driving the loop is the demonstration the L1-L8 + verifier/type/atomics work built. */
#ifndef CFRONT_DRIVER_H
#define CFRONT_DRIVER_H

typedef volatile uint32_t reg32;          /* a memory-mapped 32-bit register (typedef + volatile) */

enum dma_state { DMA_IDLE = 0, DMA_BUSY = 1, DMA_PAUSED = 2, DMA_ERROR = 4 };  /* status state codes */

/* the channel register block (MMIO): a real device's registers at fixed offsets. */
struct dma_channel {
    reg32 status;        /* +0x00: [2:0] state, [23:8] remaining transfer count */
    reg32 control;       /* +0x04 */
    reg32 count;         /* +0x08 */
};

union dma_word {         /* a control-word view: the whole 32-bit word, or its low half */
    uint32_t whole;
    uint16_t low;
};

struct dma_cfg {         /* the configuration register as named fields (a bitfield struct) */
    uint32_t enable : 1;
    uint32_t dir    : 1;
    uint32_t prio   : 2;
    uint32_t burst  : 4;
    uint32_t rsvd   : 24;
};

#define DMA_STATE(s)   (((s) >> 0) & 0x7u)        /* low 3 bits: the state code */
#define DMA_COUNT(s)   (((s) >> 8) & 0xFFFFu)     /* bits 8..23: remaining transfer count */
#define DMA_PRIO_HIGH  3

#endif /* CFRONT_DRIVER_H */
