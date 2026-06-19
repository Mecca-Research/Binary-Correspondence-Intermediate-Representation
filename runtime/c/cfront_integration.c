/* Integration: a realistic multi-feature driver that composes the Phase-2 surface end-to-end, with
 * no hand-written claim graph -- typedef + enum + an MMIO register-map struct (L5 volatile), a
 * `switch` over an enum status, a `static` fault counter, a `goto` cleanup path, integer casts, a 2D
 * bank lookup, and an inter-procedural call graph (L4 / R18). Every function is Clang-behaviour-
 * equivalent on the C rail. */
typedef uint32_t reg32;

enum sensor_state { S_IDLE = 0, S_READY = 1, S_FAULT = 2 };

struct sensor {
    reg32 status;
    reg32 sample;
    reg32 config;
};

/* decode the low 2-bit state field via a switch over the enum (folded to integer cases). */
static uint32_t decode_state(uint32_t status)
{
    switch (status & 3u) {
        case S_IDLE:  return 0u;
        case S_READY: return 1u;
        case S_FAULT: return 9u;
        default:      return 0u;
    }
}

/* a calibration table lookup: a 2D bank parameter + a narrowing cast (register packing). */
static uint32_t bank_lookup(uint32_t bank[4][8], uint32_t row, uint32_t col)
{
    return (uint16_t)bank[row & 3][col & 7];
}

/* the driver entry: read MMIO status, branch on the decoded state, keep a persistent fault count,
 * and jump to a shared cleanup -- the canonical driver shape. */
uint32_t sensor_read(volatile struct sensor *dev, uint32_t gain)
{
    static uint32_t faults = 0;
    uint32_t st = dev->status;
    uint32_t state = decode_state(st);
    uint32_t out = 0;
    if (state == 9u) {
        faults = faults + 1u;
        goto done;
    }
    out = (uint16_t)(dev->sample * gain);
done:
    return out + faults;
}
