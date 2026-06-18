#include <stdint.h>
/* L7: a C23 #embed binary table indexed at runtime. */
const uint8_t crc_table[] = {
#embed "crc.bin"
};
uint32_t l7_crc_step(uint32_t i, uint32_t acc)
{
    uint32_t hi = crc_table[i & 15];
    uint32_t lo = crc_table[(i >> 4) & 15];
    return (acc ^ hi) + lo;
}
