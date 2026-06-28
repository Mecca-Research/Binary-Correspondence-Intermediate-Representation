/* Adversarial SANITIZER fixture (Bug 1, the #532/UAF class): a member / name assignment whose RHS is a
 * statement-expression declaring enough locals to realloc c->env[] MID-RHS-parse. Before the snapshot fix,
 * the env-entry pointer (v / mbase) captured before the RHS dangled into the freed (geometrically grown)
 * c->env[] and reading v->rid / mbase->rid afterward was a heap-use-after-free (env_add's realloc on the
 * free stack). Not a parity fixture -- it pins the memory-safety regression only (run only under ASan). */
struct Senv { int x; };
unsigned f_member(void){
  struct Senv s;
  s.x = ({ int b0; int b1; int b2; int b3; int b4; int b5; int b6; int b7; int b8; int b9;
           int b10; int b11; int b12; int b13; int b14; int b15; int b16; int b17; int b18; int b19;
           int b20; int b21; int b22; int b23; int b24; int b25; int b26; int b27; int b28; int b29;
           int b30; int b31; int b32; int b33; int b34; 7; });
  return (unsigned)s.x;
}
unsigned f_name(void){
  int y;
  int z = (int)(y = ({ int c0; int c1; int c2; int c3; int c4; int c5; int c6; int c7; int c8; int c9;
           int c10; int c11; int c12; int c13; int c14; int c15; int c16; int c17; int c18; int c19;
           int c20; int c21; int c22; int c23; int c24; int c25; int c26; int c27; int c28; int c29;
           int c30; int c31; int c32; int c33; int c34; 9; }));
  return (unsigned)(y + z);
}
unsigned f_index(void){
  int a[4]; int z;
  z = a[({ int d0; int d1; int d2; int d3; int d4; int d5; int d6; int d7; int d8; int d9;
           int d10; int d11; int d12; int d13; int d14; int d15; int d16; int d17; int d18; int d19;
           int d20; int d21; int d22; int d23; int d24; int d25; int d26; int d27; int d28; int d29;
           int d30; int d31; int d32; int d33; int d34; 0; })];
  return (unsigned)z;
}
