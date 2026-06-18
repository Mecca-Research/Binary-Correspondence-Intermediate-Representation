#include <stdint.h>
/* L7: object + function macros + conditional compilation. */
#define HW_REV 3
#define WIDTH 5
#define MASK ((1u << WIDTH) - 1u)
#define FIELD(w, sh) (((w) >> (sh)) & MASK)
uint32_t l7_decode(uint32_t word)
{
#if HW_REV >= 2
    return FIELD(word, 0) + FIELD(word, WIDTH);
#elif defined(HW_REV)
    return word & MASK;
#else
    return 0;
#endif
}
