/* File-scope lookup tables: a `static const` array at file scope, indexed at runtime -- the driver
 * calibration / jump-table pattern. The global lowers to a read-only data resource; an access
 * `GAMMA[i]` is an indexed load, emitted verbatim (the global is referenced by name, defined in the
 * original source -- not redeclared). */
static const uint32_t GAMMA[8] = {0, 1, 4, 9, 16, 25, 36, 49};

uint32_t gamma_at(uint32_t i)
{
    return GAMMA[i & 7] + 1u;
}
