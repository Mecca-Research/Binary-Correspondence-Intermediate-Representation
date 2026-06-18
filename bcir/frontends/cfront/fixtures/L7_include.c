#include <stdint.h>
#include "regmap.h"
/* L7: #include a project header; use its macros + __has_include. */
uint32_t l7_regaddr(uint32_t off)
{
#if defined(REG_BASE) && __has_include("regmap.h")
    return REG_BASE + off * REG_STRIDE;
#else
    return off;
#endif
}
