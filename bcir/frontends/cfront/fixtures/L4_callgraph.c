#include <stdint.h>
/* L4: functions + a call graph (-> R18). */
uint32_t l4_scale(uint32_t v, uint32_t k)
{
    return v * k;
}
uint32_t l4_main(uint32_t a, uint32_t b)
{
    uint32_t s = l4_scale(a, 3);
    uint32_t t = l4_scale(b, 5);
    return s + t;
}
