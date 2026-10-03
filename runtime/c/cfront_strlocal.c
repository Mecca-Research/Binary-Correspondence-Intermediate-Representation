/* A character array sized by its string literal (C11 6.7.9p14-15/22, CF-STRLOCAL): `char s[] = "abc"` is a
 * local of four elements -- the literal's units plus its NUL -- and the literal initializes it unit by unit,
 * its rest left zero; a wide literal (`u"..."`, `U"..."`, `L"..."`) initializes an array of its code unit;
 * adjacent literals concatenate; a braced literal `{"ab"}` is the same initializer; a literal exactly as
 * long as a sized array drops its NUL; an array of strings `char m[][4] = {...}` takes its row count from
 * the list. Both rails size and fill it the same way, and each emit returns what the original does. */
#include <stdint.h>
#include <stddef.h>

uint32_t sl_plain(uint32_t i) {                     /* "abc" -> char[4]: a, b, c, NUL */
  char s[] = "abc";
  return (uint32_t)sizeof s * 1000u + (uint32_t)s[i % 4u];
}
uint32_t sl_braced(uint32_t i) {                    /* `{"ab"}` initializes the same array */
  char s[] = {"ab"};
  return (uint32_t)sizeof s * 1000u + (uint32_t)s[i % 3u];
}
uint32_t sl_unsigned(uint32_t i) {                  /* an unsigned char array */
  unsigned char s[] = "\xff\x01z";
  return (uint32_t)sizeof s * 1000u + s[i % 4u];
}
uint32_t sl_concat(uint32_t i) {                    /* adjacent literals are one: "ab" "cde" -> 6 */
  char s[] = "ab" "cde";
  return (uint32_t)sizeof s * 1000u + (uint32_t)s[i % 6u];
}
uint32_t sl_escapes(uint32_t i) {                   /* escapes are one unit each: \n \0 \x41 \101 */
  char s[] = "\n\0\x41\101";
  return (uint32_t)sizeof s * 1000u + (uint32_t)s[i % 5u];
}
uint32_t sl_sized_longer(uint32_t i) {              /* a sized array longer than its literal: the rest is 0 */
  char s[8] = "hi";
  return (uint32_t)s[i % 8u] + (uint32_t)s[7] * 7u;
}
uint32_t sl_sized_exact(uint32_t i) {               /* exactly as long as the array: the NUL is dropped */
  char s[3] = "xyz";
  return (uint32_t)sizeof s * 1000u + (uint32_t)s[i % 3u];
}
uint32_t sl_utf16(uint32_t i) {                     /* u"..." -> char16_t units */
  uint16_t w[] = u"hi!";
  return (uint32_t)sizeof w * 100000u + w[i % 4u];
}
uint32_t sl_utf32(uint32_t i) {                     /* U"..." -> char32_t units */
  uint32_t w[] = U"ok";
  return (uint32_t)sizeof w * 100000u + w[i % 3u];
}
uint32_t sl_wide(uint32_t i) {                      /* L"..." -> the target's wchar_t units */
  wchar_t w[] = L"wide";
  return (uint32_t)(sizeof w / sizeof w[0]) * 1000u + (uint32_t)w[i % 5u];
}
uint32_t sl_rows(uint32_t i) {                      /* an array of strings: rows from the list, NULs kept */
  char m[][4] = {"ab", "cde", "f"};
  return (uint32_t)sizeof m * 1000u + (uint32_t)m[i % 3u][i % 4u] + (uint32_t)m[1][2] * 3u;
}
struct sl_rec { char tag[5]; uint8_t n; };
uint32_t sl_struct_member(uint32_t i) {             /* a string member of a local struct */
  struct sl_rec r = {"qrs", 9};
  return (uint32_t)r.tag[i % 5u] + r.n * 1000u;
}
uint32_t sl_write(uint32_t i) {                     /* the array is a mutable local, not the literal */
  char s[] = "mno";
  s[i % 3u] = 'Z';
  return (uint32_t)s[0] + (uint32_t)s[1] * 3u + (uint32_t)s[2] * 5u;
}
uint32_t sl_entry(uint32_t i) {
  return sl_plain(i) + sl_braced(i) * 3u + sl_unsigned(i) * 5u + sl_concat(i) * 7u + sl_escapes(i) * 11u
         + sl_sized_longer(i) * 13u + sl_sized_exact(i) * 17u + sl_utf16(i) * 19u + sl_utf32(i) * 23u
         + sl_wide(i) * 29u + sl_rows(i) * 31u + sl_struct_member(i) * 37u + sl_write(i) * 41u;
}
