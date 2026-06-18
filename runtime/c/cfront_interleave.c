/* Interleaved top-level declarations: type definitions appear *between* two functions, the way
 * real translation units and vendor headers lay them out. The first function lowers, then an enum
 * and a struct are defined, and only then does the entry consume both -- all in one top-level pass.
 * (The C frontend used to require every typedef/enum/struct/union definition before the first
 * function; the oracle already allowed interleaving, so this gates the two rails back together.) */
uint32_t scale(uint32_t v, uint32_t k)
{
    return v * k;
}

enum mode { OFF = 0, LOW = 1, HIGH = 3 };

struct cfg { uint32_t base; uint32_t span; };

uint32_t plan(struct cfg c, uint32_t sel)
{
    uint32_t pick = (sel == HIGH) ? c.span : c.base;
    return scale(pick, sel) + LOW;
}
