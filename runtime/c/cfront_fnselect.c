/* The arms of `?:` that are function designators (CF-FNSEL): a function named as a value decays to a pointer to it
 * (C11 6.3.2.1p4), so `s > 3u ? fs_twice : fs_thrice` is a function pointer (6.5.15p6). Both rails typed the
 * select `uint32_t`, and each emit declared `uint32_t t = (c ? fs_twice : fs_thrice);`, which does not compile.
 * The select is now a pointer to the arms' function type: of two designators, a designator and a function-pointer
 * object, two objects, a designator and a null pointer constant (`c ? f : 0`, whose 0 is that pointer), nested, in
 * a branch whose arm has an effect, of void and of struct-returning functions -- assigned to a local, a typedef'd
 * local and over an earlier value, passed, returned, compared, and called. A pointer that may hold either of two
 * functions is returned to the caller, which calls it (the test's driver): the G10 escape rows count every
 * indirect call no analysis resolves to one function (tools/perf/check_escape.py), so each call made here goes
 * through a pointer that holds one; calling through a select in place is the test's own unit (`_FNSEL_CALLS`). */
#include <stdint.h>

typedef uint32_t (*fs_op)(uint32_t);
typedef void (*fs_act)(uint32_t);
struct fs_pair { uint32_t a, b; };
typedef struct fs_pair (*fs_mk)(uint32_t);

static uint32_t fs_n;                               /* what the actions change, and how many arms had an effect */
static uint32_t fs_twice(uint32_t x) { return x * 2u + 1u; }
static uint32_t fs_thrice(uint32_t x) { return x * 3u + 5u; }
static uint32_t fs_mix(unsigned int x) { return (x ^ 0x5Au) + 7u; }   /* `unsigned int`: uint32_t's type here */
static void fs_add(uint32_t x) { fs_n += x; }
static void fs_sub(uint32_t x) { fs_n -= x * 3u; }
static struct fs_pair fs_lo(uint32_t x) { struct fs_pair p = {x, 1u}; return p; }
static struct fs_pair fs_hi(uint32_t x) { struct fs_pair p = {x + 9u, 2u}; return p; }

fs_op fs_pick(uint32_t s) { return s > 3u ? fs_twice : fs_thrice; }   /* returned: the caller calls it */
fs_op fs_pick_decl(uint32_t s) {                     /* into a declarator local */
  uint32_t (*fp)(uint32_t) = s & 1u ? fs_thrice : fs_mix;
  return fp;
}
fs_op fs_pick_assign(uint32_t s) {                   /* over an earlier value */
  fs_op f = fs_mix;
  f = s > 7u ? fs_thrice : fs_twice;
  return f;
}
fs_op fs_pick_nested(uint32_t s) { return s < 3u ? fs_twice : s < 9u ? fs_thrice : fs_mix; }   /* a select of selects */
fs_op fs_pick_branch(uint32_t s) {                   /* an arm with an effect: a branch into a function-pointer local */
  fs_op f = s > 5u ? (fs_n++, fs_twice) : fs_thrice;
  return f;
}
fs_act fs_pick_act(uint32_t s) { return s & 2u ? fs_add : fs_sub; }  /* pointers to void functions */
fs_mk fs_pick_mk(uint32_t s) { return s > 5u ? fs_lo : fs_hi; }      /* pointers to struct-returning functions */
static uint32_t fs_is_twice(fs_op f) { return f == fs_twice; }
uint32_t fs_pass(uint32_t s) {                       /* passed, and compared where it lands */
  return fs_is_twice(s > 100u ? fs_twice : fs_mix) + fs_is_twice(s & 4u ? fs_mix : fs_twice) * 2u;
}
uint32_t fs_object(uint32_t s) {                     /* a designator and an object, two objects: one function each */
  fs_op g = fs_thrice;
  uint32_t (*h)(uint32_t) = fs_thrice;
  fs_op f = s > 5u ? g : fs_thrice;
  fs_op k = s & 1u ? h : g;
  return f(s) + k(s + 1u) * 7u;
}
uint32_t fs_null(uint32_t s) {                       /* a null pointer constant arm: the other arm's pointer */
  fs_op f = s > 5u ? fs_twice : 0;
  fs_op g = s & 1u ? 0 : fs_mix;
  fs_act a = s & 2u ? fs_add : 0;
  fs_mk m = s > 9u ? 0 : fs_lo;
  uint32_t k = s;
  fs_n = s;
  if (f) k += f(s);
  if (!g) k += 1000u;
  else k += g(s) * 3u;
  if (a) a(s);
  if (m) {
    struct fs_pair q = m(s);
    k += q.a * 5u + q.b;
  }
  return k + fs_n * 11u;
}
uint32_t fs_compare(uint32_t s) {                    /* compared with a designator, a select, each other */
  fs_op f = s > 5u ? fs_twice : fs_thrice;
  fs_op g = s & 1u ? fs_twice : fs_mix;
  uint32_t k = 0u;
  if (f == fs_twice) k += 1u;
  if (f != fs_thrice) k += 2u;
  if (f == g) k += 4u;
  if ((s > 2u ? fs_mix : fs_twice) == g) k += 8u;
  if ((s & 1u ? fs_mix : fs_twice) == fs_mix ? f == fs_twice : g != fs_mix) k += 16u;
  return k;
}
uint32_t fs_entry(uint32_t s) {
  return fs_pass(s) + fs_object(s) * 3u + fs_null(s) * 5u + fs_compare(s) * 7u;
}
