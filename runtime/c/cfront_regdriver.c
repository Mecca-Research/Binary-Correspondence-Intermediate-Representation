/* Register-map composition checkpoint: a realistic device driver that exercises the whole register
 * surface together in one function, with no hand-written claim graph -- a `switch` over a status
 * field, a multi-bit bitfield write (`dev->mode = ...`), a bitfield read (`dev->prio`), a register
 * read-modify-write (`dev->ctrl |= ...`), a file-scope lookup table, an `enum`, and a `static`
 * persistent counter. Proves the register-map features (PR #294-#297) compose, not just pass alone. */
typedef uint32_t reg32;

enum dstate { D_IDLE = 0, D_RUN = 1, D_HALT = 2 };

static const uint32_t QUANTA[4] = {1, 2, 4, 8};

struct device {
    reg32 status;
    uint32_t mode : 4;
    uint32_t prio : 4;
    reg32 ctrl;
};

uint32_t step(volatile struct device *dev, uint32_t sel)
{
    static uint32_t halts = 0;
    uint32_t out = 0;
    switch (dev->status & 3u) {
        case D_IDLE:
            dev->mode = sel & 7u;        /* bitfield write */
            out = QUANTA[sel & 3];       /* lookup table */
            break;
        case D_RUN:
            dev->ctrl |= 2u;             /* register read-modify-write */
            out = dev->prio;             /* bitfield read */
            break;
        default:
            halts += 1u;                 /* static persistent counter */
            out = 9u;
            break;
    }
    return out + halts;
}
