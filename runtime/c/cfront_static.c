/* static local variables: a function-local with static storage duration -- it persists across calls
 * and is initialized once. The canonical driver pattern (a running counter / accumulator). The
 * `static T name = init;` initializer is a constant baked into the declaration, so it is NOT a
 * per-call assignment (no init claim is lowered); reads/writes then hit that persistent storage. */
uint32_t tick(uint32_t inc)
{
    static uint32_t total = 0;
    total = total + inc;
    return total;
}
