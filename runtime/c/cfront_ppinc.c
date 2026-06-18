/* L7: #include a project header; use its macros. */
#include "cfront_ppdefs.h"
uint32_t l7_regaddr(uint32_t off)
{
#ifdef REG_BASE
    return REG_BASE + off * REG_STRIDE;
#else
    return off;
#endif
}
