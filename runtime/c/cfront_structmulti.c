/* Multi-declarator struct/union members (#structmulti): `unsigned x, y, z;` -- several members off one
 * type-specifier, including multi-declarator bitfields (`unsigned a:3, b:5;`). Each declarator lays out
 * exactly as if written on its own line (Clang treats them identically), so offsets / size + member
 * access match Clang on both rails -- the member-declaration twin of the multi-declarator locals (#374). */
struct Pt { unsigned x, y, z; };
unsigned pt_sum(unsigned a) {
  struct Pt p;
  p.x = a; p.y = a * 2u; p.z = a ^ 0xFFu;
  return p.x + p.y + p.z;
}
struct Flags { unsigned mode : 3, level : 5, mask : 8; };
unsigned fl_pack(unsigned v) {
  struct Flags f;
  f.mode = v; f.level = v >> 1; f.mask = v >> 2;
  return f.mode + f.level + f.mask;
}
