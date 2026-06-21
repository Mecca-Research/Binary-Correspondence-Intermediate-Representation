/* Pointer-element signedness (#ptrsign): a load / store / subscript through a pointer carries the
 * POINTEE's signedness, not just its width. So a deref or subscript of a signed sub-int pointer
 * sign-extends (a negative byte / short reads back negative), an unsigned one zero-extends, and the
 * loaded value drives signed-vs-unsigned division / remainder / shift / comparison / the usual
 * arithmetic conversions. Both rails thread the pointee sign through every pointer-resource path --
 * param, local, struct field, and a pointer-arithmetic result -- so the emit matches Clang over the
 * full signed range (a width-only model would zero-extend a negative pointee and pick the wrong
 * signedness for the arithmetic). Pins the behaviour the integer-promotion / UAC work left as a
 * pointer follow-on; driven by a bespoke harness over negative + boundary pointee values.
 *
 * Uses the fixed-width <stdint.h> types (int8_t/uint8_t/int16_t/uint16_t) rather than `signed char` /
 * `short`: they are unambiguously signed/unsigned on every target, so the differential is exact on
 * ARM too (where plain `char` is unsigned -- a `signed char` spelled emit would diverge there). */

int      ps_s8(int8_t *p)                 { return *p; }          /* sign-extend a negative byte */
unsigned ps_u8(uint8_t *p)                { return *p; }          /* zero-extend */
int      ps_s16(int16_t *p, int i)        { return p[i]; }        /* signed subscript */
unsigned ps_u16(uint16_t *p, int i)       { return p[i]; }        /* unsigned subscript */
int      ps_s8_divrem(int8_t *p)          { return *p / 3 + *p % 3; }   /* signed divide + remainder */
unsigned ps_u8_div(uint8_t *p)            { return *p / 3u; }
int      ps_s8_shr(int8_t *p)             { return *p >> 2; }     /* arithmetic (sign-propagating) shift */
unsigned ps_u8_shr(uint8_t *p)            { return *p >> 2; }     /* logical shift */
int      ps_s8_cmp(int8_t *p)             { return (*p < 0) ? 7 : -7; }
long     ps_s64_div(long *p)              { return *p / 3; }      /* an 8-byte signed pointee divide */
unsigned long ps_u64_div(unsigned long *p){ return *p / 3ul; }   /* an 8-byte unsigned pointee divide */
int      ps_arith(int8_t *p, int i)       { int8_t *q = p + i; return *q >> 1; }  /* sign survives p+i */
int      ps_uac(uint8_t *p, int x)        { return *p - x; }     /* UAC: uint8_t promotes to int */

struct Buf { int8_t *sp; uint8_t *up; };
int      ps_field(struct Buf *b)          { return *(b->sp) + *(b->up); }   /* sign via a loaded ptr field */

void ps_w8(int8_t *p, int v)              { *p = (int8_t)v; }     /* a store truncates to 1 byte */
