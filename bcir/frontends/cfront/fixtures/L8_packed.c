#include <stdint.h>
/* L8: a packed (no-padding) wire header read by value — the offsets must match Clang's ABI. */
struct __attribute__((packed)) wire_hdr {
    uint8_t  cmd;
    uint32_t addr;
    uint16_t len;
};
uint32_t l8_wire_fields(struct wire_hdr h)
{
    return h.cmd + h.addr + h.len;
}
