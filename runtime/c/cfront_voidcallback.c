/* A call through a pointer to a function returning void (CF-VOIDCB): a callback held in a local, a typedef'd
 * local, a parameter (a typedef or a declarator), a struct member (declared or typedef'd) reached by `.` or `->`,
 * one taking no arguments, one in an arm of `?:`, one cast to void, one returned from a void function (`return
 * cb(s);`). Both rails lowered each such call as a claim writing a result temp, and each emit declared it
 * `uint32_t t = fp(s);`, which Clang and GCC reject; a call through a void function pointer now has no result, as
 * a direct call to a void function has none. Every function that calls through a pointer parameter is `static`,
 * and every pointer and struct here holds one function, so each indirect call resolves to the one function it
 * reaches: the G10 escape rows count every indirect call no analysis narrows or resolves, and the analysis keeps
 * one set per struct (tools/perf/check_escape.py). A pointer given two functions, a member through a loaded
 * pointer chain and a file-scope ops table are the test's own unit (`_VOIDCB_OPEN`, bcir/tests/test_c_cfront.py). */
#include <stdint.h>

typedef void (*vc_fn)(uint32_t);
struct vc_step { void (*step)(uint32_t); uint32_t k; };   /* a declarator member */
struct vc_reset { uint32_t k; vc_fn reset; };             /* a typedef'd member */
struct vc_tick { void (*tick)(void); };                   /* one taking no arguments */

static uint32_t vc_n;                                /* what the callbacks change: each function starts it at 0 */
static void vc_bump(uint32_t x) { vc_n += x; }
static void vc_twice(uint32_t x) { vc_n += 2u * x + 1u; }
static void vc_mix(uint32_t x) { vc_n = vc_n * 3u + (x ^ 0x5Au); }
static void vc_tock(void) { vc_n += 100u; }

uint32_t vc_local(uint32_t s) {                     /* a local declarator and a typedef'd local */
  void (*fp)(uint32_t) = vc_bump;
  vc_fn f = vc_twice;
  void (*t)(void) = vc_tock;
  vc_n = 0u;
  fp(s);
  f(s + 1u);
  t();
  fp(s ^ 7u);
  return vc_n;
}
static void vc_apply(vc_fn f, uint32_t s) {         /* a parameter, called twice */
  f(s);
  f(s ^ 3u);
}
static void vc_apply_decl(void (*f)(uint32_t), void (*g)(void), uint32_t s) {   /* declarator parameters */
  g();
  f(s);
}
uint32_t vc_param(uint32_t s) {
  vc_n = 0u;
  vc_apply(vc_mix, s);
  vc_apply(vc_mix, s + 2u);
  vc_apply_decl(vc_bump, vc_tock, s);
  return vc_n;
}
uint32_t vc_member(uint32_t s) {                    /* members of struct values, declared and typedef'd */
  struct vc_step o = {vc_bump, 3u};
  struct vc_reset r = {2u, vc_twice};
  struct vc_tick t = {vc_tock};
  vc_n = 0u;
  o.step(s);
  r.reset(s + o.k);
  t.tick();
  o.step(s * 5u + r.k);
  return vc_n;
}
static void vc_run(struct vc_step *o, struct vc_reset *r, struct vc_tick *t, uint32_t s) {   /* through `->` */
  o->step(s);
  r->reset(s + o->k);
  t->tick();
}
uint32_t vc_arrow(uint32_t s) {
  struct vc_step o = {vc_mix, 1u};
  struct vc_reset r = {0u, vc_bump};
  struct vc_tick t = {vc_tock};
  vc_n = 0u;
  vc_run(&o, &r, &t, s);
  return vc_n;
}
uint32_t vc_cond(uint32_t s) {                      /* in an arm of `?:`: the arms are void, one runs */
  vc_fn f = vc_bump;
  struct vc_step o = {vc_twice, 0u};
  struct vc_reset r = {0u, vc_mix};
  struct vc_tick t = {vc_tock};
  vc_n = 0u;
  s > 3u ? f(s) : (void)0;
  s & 1u ? f(1u) : o.step(s);
  s & 2u ? t.tick() : r.reset(s);
  (void)f(s);                                        /* a void value cast to void */
  return vc_n;
}
static void vc_forward(vc_fn f, uint32_t s) { return f(s); }            /* `return cb(s);` in a void function */
static void vc_forward_member(struct vc_reset *r, uint32_t s) { return r->reset(s); }
static void vc_forward_cond(vc_fn f, uint32_t s) { return s > 7u ? f(s) : vc_tock(); }
uint32_t vc_return(uint32_t s) {
  struct vc_reset r = {0u, vc_twice};
  vc_n = 0u;
  vc_forward(vc_mix, s);
  vc_forward_member(&r, s);
  vc_forward_cond(vc_bump, s);
  return vc_n;
}
uint32_t vc_entry(uint32_t s) {
  return vc_local(s) + vc_param(s) * 3u + vc_member(s) * 5u + vc_arrow(s) * 7u + vc_cond(s) * 11u
         + vc_return(s) * 13u;
}
