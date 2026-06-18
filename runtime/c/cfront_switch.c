/* switch/case -> a nested if/else-if chain: single cases, a shared multi-label clause, an enum
 * state-machine decode, and a default. The discriminant is a variable (re-evaluation is free). */
enum phase { IDLE = 0, ARMED = 1, RUN = 2, DONE = 4 };

uint32_t decode(uint32_t state, uint32_t x)
{
    uint32_t r = 0;
    switch (state) {
        case 0:
            r = x + 1;
            break;
        case 1:
        case 2:
            r = x * 2;
            break;
        case 7:
            r = x - 3;
            break;
        default:
            r = x & 15;
    }
    return r;
}

/* an enum discriminant -- the canonical driver state-machine decode (enum cases fold to ints). */
uint32_t step_phase(uint32_t p, uint32_t ticks)
{
    uint32_t out;
    switch (p) {
        case IDLE:
            out = 0;
            break;
        case ARMED:
        case RUN:
            out = ticks + p;
            break;
        case DONE:
            out = ticks;
            break;
        default:
            out = 0xFFu;
    }
    return out;
}
