/*===- bcir_cfront.c - the BCIR plug-in C frontend (C twin of bcir/frontends/cfront) ===
 *
 * A recursive-descent C compiler for the driver/kernel subset, lowering C to the BCIR
 * claim graph (bcir_cir.h) -- the same IR the oracle reasons over. Ported stages:
 *   L1 fixed-width integer expressions   L2 struct/union layout + member access
 *   L3 pointers/arrays (GEP-equivalent)  L4 functions + the call graph -> R18
 *   L5 volatile/MMIO + bitfields
 * It runs an R1-R8 + R18 verifier and emits faithful, compilable C (so a host harness
 * checks behaviour-equivalence against Clang). Host tool (libc); the IR it emits is
 * freestanding. A Python<->C parity test gates the two rails.
 *===----------------------------------------------------------------------===*/
#include "bcir_cfront.h"
#include "bcir_verify.h"

#include <limits.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A function kept out of its callers' frames. The address sanitizer gives every address-taken local its own stack
 * slot, so a local of a rarely taken path that inlines into the recursive descent costs every level of it: the
 * lvalue paths CF-SPLIT2 added grew each level by some 1.9 KiB under GCC's ASan, and the deepest nesting the depth
 * guard admits (`cfront_sec_deepnest.c`) overflowed the stack before the guard refused it. */
#if defined(__GNUC__) || defined(__clang__)
#define BCIR_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define BCIR_NOINLINE __declspec(noinline)
#else
#define BCIR_NOINLINE
#endif

const char *bcir_opcode_name(bcir_opcode op) {
  static const char *N[] = {"nop","load","store","add","sub","mul","atomic_add","atomic_sub",
    "atomic_xor","cmpxchg","barrier","phase_enter","phase_leave","ggg_load","ggg_store","t_macc",
    "gem_dispatch","prov_note"};
  return (op >= 0 && op <= BCIR_OP_PROV_NOTE) ? N[op] : "?";
}

/* --- lexer --------------------------------------------------------------- */
typedef enum { T_ID, T_INT, T_FLT, T_STR, T_PUN, T_END } tkind;
typedef struct { tkind k; const char *s; int n; long long v; } tok;

#define MAXTOK 16384
#define MAXFLD 64        /* members per struct (f[] is embedded in sdef; generous, guarded) */
/* The emitter's scratch spellings (CF-BUF), each sized for the longest it holds now that a name is at most
 * BCIR_CIR_IDENT_MAX characters: a declared name with its `_N` disambiguation; a C type (`_Atomic volatile
 * struct <tag>` and 16 `*`s is 104); an expression of several of each (a bounds guard names four: 331). */
#define BCIR_EMIT_NAME (BCIR_CIR_NAME + 16)
#define BCIR_EMIT_TYPE 256
#define BCIR_EMIT_EXPR 1024

typedef struct { char name[BCIR_CIR_NAME]; int size; int signd; int is_float; int is_complex; int is_bool; int is_plain_char; int bit_width; int byte_off, bit_off, bit_w; int sidx;
                 /* bit_width: a PLAIN (non-bitfield) C23 `_BitInt(N)` member's EXACT width N (0 == a normal
                  * member; >0 == `_BitInt(N)`). `size` is the storage slot (1/2/4/8 bytes, == Clang's), so the
                  * layout matches; the load/store goes through the storage width, and the loaded value carries
                  * the `_BitInt(N)` type so the emit spells it faithfully + same-type arithmetic stays N-bit.
                  * (A `_BitInt` BITFIELD `_BitInt(N) m:W` is OUT of the subset -- rejected at parse.) */
                 int access_bytes;   /* a bitfield's storage-unit byte span (== size, except a PACKED bitfield
                                      * spans only ceil((bit_off+bit_w)/8) bytes -- it may straddle byte/word
                                      * boundaries; `size` stays the DECLARED type width, for read promotion) */
                 int arr_count; int nadims; int adims[3];
                 int is_ptr; int ptee_size; int ptee_float; int ptee_sidx;
                 int ptee_depth;    /* a pointer (or pointer-array element) member's own pointer depth: 1 `T *m`,
                                     * 2 `T **m` -- what `*s->m` is, for `sizeof` (0: not a pointer) */
                 int elem_ptr;      /* arr_count>0 AND the element is a pointer (`T *arr[N]`): each element is
                                     * pointer_size wide, and the ptee_* fields describe what it points at, as
                                     * they do for a pointer member (`is_ptr` stays 0: the member is an array) */
                 int is_volatile;   /* the member's own storage is volatile (`volatile T m`, `volatile T m[N]`, a
                                     * volatile struct member): an access to it is a volatile access */
                 int ptee_volatile; /* a pointer member to volatile storage (`volatile T *regs`): the member is
                                     * plain storage; what it points at is a device region */
                 int is_atomic;     /* the member's own storage is `_Atomic` (`_Atomic T m`, `_Atomic T m[N]`): an
                                     * access to it is an atomic one (CF-ATOMIC) */
                 int ptee_atomic;   /* a pointer member to `_Atomic` storage (`_Atomic T *m`): the member is plain
                                     * storage; an access through it is atomic */
                 int elem_sidx;     /* arr_count>0 AND element is a value struct: its sdef index (array-of-structs,
                                     * for `arr[i].field`); -1 otherwise. Distinct from `sidx` so member_descend
                                     * (which descends a `.` only when sidx>=0) never walks an un-indexed array. */
                 int fp_ret_size; uint8_t fp_ret_signd; uint8_t fp_ret_float;
                 uint8_t fp_ret_void;
                                    /* a funcptr struct member: the captured RETURN type (sign/width/float),
                                     * used to type a c.call.imember result temp; ZERO if not a funcptr / not captured;
                                     * and a `void` return, whose call has no value (CF-VOIDCB; in the padding) */
                 int fp_ret_agg;    /* ... and a struct/union return, as 1 + its sdef index (0: not one) */
                 int fp_sig;        /* ... and its function type, 1 + its index in the context's `sigs` (0: none) */
                 } field;
                 /* is_ptr: a pointer member -- `size` is pointer_size (the ABI layout width), the pointee
                  * (ptee_size width / signd sign / ptee_float / ptee_sidx struct) types the loaded `T *` */
                 /* arr_count > 0: a member array; `size` is the element, arr_count the total element
                  * count, nadims/adims the per-dim sizes (`T m[A][B]` -> nadims 2, adims {A,B}) */
/* An ANONYMOUS struct/union member: its promoted leaves are `f[first..first+n)` of the containing sdef, the
 * member itself the sdef `sidx` at byte offset `off` -- one subobject to an initializer (the oracle's
 * `CType.anon`), so an anonymous union takes one positional value, as in C. */
typedef struct { int first, n, sidx, off; } anon_grp;
typedef struct { char tag[BCIR_CIR_NAME]; field f[MAXFLD]; int nf; int size; int align; int is_union;
                 int vol_storage;   /* the struct holds volatile storage: a volatile member, or a value-struct /
                                     * array-of-structs member that does (a pointer member does not) -- the
                                     * oracle's `CType.volatile_storage` */
                 int nanon; anon_grp anon[MAXFLD];   /* its anonymous members, in declaration order */
                 int incomplete;    /* named but not (yet) laid out -- `struct fwd;`, a pointer to a struct defined
                                     * later or never, the struct whose body is being parsed: a pointer to it is
                                     * complete, anything else needs its layout (the oracle's `CType.incomplete`) */
                 /* how C names an ANONYMOUS aggregate (a synthesized `$anonN` tag no compiler knows; `anon_spelling`):
                  * the file-scope typedef that names it (`typedef struct {...} P;`), else the named member it is the
                  * type of -- `parent`'s member `member`, through `mstars` `*`s and `mdims` array dimensions */
                 char tdname[BCIR_CIR_NAME]; int parent; char member[BCIR_CIR_NAME]; int mstars, mdims;
                 char tdptr[BCIR_CIR_NAME]; int tdstars;   /* ... else a typedef of a pointer to it (`*PP`) */
               } sdef;
typedef struct { char name[BCIR_CIR_NAME]; bcir_ctype ty; int sidx;
                 int nd, dims[3];   /* an array typedef's dims, outer first (`typedef T row_t[6];`): each declarator
                                     * of the alias appends them after its own (CF-SMALL) */
               } tdef;   /* a typedef alias */
typedef struct { char name[BCIR_CIR_NAME]; long long val; } econst;           /* an enum constant */

typedef struct {
  char name[BCIR_CIR_NAME];
  uint32_t rid;
  bcir_ctype type;   /* scalar / struct-by-value / pointer */
  int sidx;          /* struct index for kind 1 or ptr_to_struct */
} venv;

typedef struct { char name[BCIR_CIR_NAME]; bcir_ctype ty; int count;
                 int is_arr;          /* declared with `[...]` (an array of any length) */
                 int nd;              /* its dimensions, outermost first -- `sizeof` measures all of them, where
                                       * `count` keeps the last (CF-SIZEOF); more than 4 are not recorded */
                 long long dims[4];   /* 0: unsized, or not an integer literal */
                 int init_a, init_b;  /* the initializer's token range [init_a, init_b) (empty: none) --
                                       * the escape analysis marks every function it names address-taken */
               } gvar;  /* a file-scope global */

/* The size-varying part of a target's C data model -- the C twin of frontends/cfront/abi.py. `long`,
 * the pointer, and the pointer-tracking `size_t`-class types move across the matrix; `int`, `short`,
 * `char`, the fixed-width <stdint.h> types, and `long long` are fixed by C and the common ABIs.
 * (`long double` takes the ABI's size/align: the twin emits real `long double` C and lets the backend /
 * Clang do the 80/128-bit arithmetic, exactly as it does for float/double -- it never models the bits.) */
typedef struct {
  const char *name;          /* short id, e.g. "x86_64-linux" */
  const char *triple;        /* the Clang target triple (for provenance / -target) */
  const char *data_model;    /* "LP64" | "LLP64" | "ILP32" */
  int long_size;             /* sizeof(long) == sizeof(unsigned long) */
  int pointer_size;          /* sizeof(void *); also size_t / intptr_t / uintptr_t */
  int long_double_size;      /* sizeof(long double): 16 (x86-64), 12 (ILP32), or 8 where it aliases double */
  int long_double_align;
  int atomic_promote_size;   /* the widest `_Atomic` type the ABI promotes (Clang's MaxAtomicPromoteWidth, bytes) */
  int eight_byte_align;      /* an 8-byte scalar's alignment -- `double`, `long long`, `int64_t`, a `_BitInt(33..64)`
                              * and a `double _Complex`'s element (Clang's DoubleAlign / LongLongAlign): 4 on i386 */
  int wchar_size, wchar_signed;   /* `wchar_t` (Clang's __WCHAR_TYPE__), an `L"..."` literal's unit too: a signed
                              * 4-byte `int` on x86-64, RISC-V and i386 Linux, `unsigned int` on AArch64 Linux,
                              * `unsigned short` on Windows */
} bcir_abi;

/* The named matrix (mirrors abi.py TARGETS). x86-64 / AArch64 / RISC-V are all LP64, so their
 * layouts coincide; Windows x64 is LLP64 (long is 4) and 32-bit x86 is ILP32 (pointers are 4) -- the
 * cases that change what the frontend lays out. g_targets[0] is the default (host LP64) model, so
 * --target-less compilation is byte-identical to the layout used before --target existed. */
static const bcir_abi g_targets[] = {
  {"x86_64-linux",   "x86_64-unknown-linux-gnu",  "LP64",  8, 8, 16, 16, 16, 8, 4, 1},
  {"aarch64-linux",  "aarch64-unknown-linux-gnu", "LP64",  8, 8, 16, 16, 16, 8, 4, 0},
  {"riscv64-linux",  "riscv64-unknown-linux-gnu", "LP64",  8, 8, 16, 16, 16, 8, 4, 1},
  {"x86_64-windows", "x86_64-pc-windows-msvc",    "LLP64", 4, 8,  8,  8, 16, 8, 2, 0},
  {"i386-linux",     "i386-unknown-linux-gnu",    "ILP32", 4, 4, 12,  4,  8, 4, 4, 1},
};
#define BCIR_N_TARGETS ((int)(sizeof g_targets / sizeof g_targets[0]))
static const bcir_abi *bcir_abi_host(void){ return &g_targets[0]; }
/* Look up a target ABI by short name; NULL for an unknown name (the driver reports the matrix). */
static const bcir_abi *bcir_abi_by_name(const char *name){
  if(!name) return bcir_abi_host();
  for(int i=0;i<BCIR_N_TARGETS;i++) if(!strcmp(g_targets[i].name,name)) return &g_targets[i];
  return NULL;
}

/* §5.12 one NAME's mutation tally (assignment count + address-taken flag) for the extent-stability
 * pre-pass; an array of these lives on CC, reset per function. */
typedef struct { char name[BCIR_CIR_NAME]; int assigned; int body; int addr; } mutent;

typedef struct { long long lo, hi;                /* a bit range the initializer walk stored (CF-BRACE), and in a */
                 unsigned long long v; int neg; } irange;   /* static's image the value stored there -- its 64-bit
                                                             * two's complement, negative or not (CF-STATICTAB) */
typedef struct { int off, sidx, member; } iunion; /* a union the initializer walk gave a member, and which */
/* A growable, NUL-terminated text the context owns (its allocator; kept across compiles, released by
 * bcir_cfront_context_destroy): `w` bytes written, `cap` held. */
typedef struct { char *s; size_t w, cap; } ctext;
/* The function type a function pointer points at: its return type and its parameters' types (CF-FNSEL, CF-NULLCALL).
 * `alias` is the typedef that spells a pointer to it in the emit -- a source typedef, or the synthesized `__bcir_fpN`
 * of a declarator or of a function designator named `fn` -- and "" for a struct member's, whose struct the source
 * declares. Held in the scratch arena for the compile; a kind-3 ctype names its entry as 1 + its index (`fp_sig`). */
typedef struct { bcir_ctype ret; const bcir_ctype *params; int n_params;
                 int variadic;                       /* a trailing `...`: the named parameters are `params` (CF-FPRET) */
                 char alias[BCIR_CIR_NAME]; char fn[BCIR_CIR_NAME]; } fsig;
typedef struct {
  bcir_host_allocator allocator;
  bcir_host_arena scratch;
  jmp_buf failure_jump;
  int jump_active;
  tok t[MAXTOK]; int nt, i;
  sdef *s; int ns, cap_s;         /* struct/union definitions (grown -- no fixed cap) */
  tdef *td; int ntd, cap_td;      /* typedef aliases (resolved at parse time) */
  econst *ec; int nec, cap_ec;    /* enum constants (folded to literals at parse time) */
  gvar *gv; int ngv, cap_gv;      /* file-scope globals (lookup tables): name -> type + length */
  venv *env; int nenv, cap_env;   /* in-scope local variables */
  bcir_func *fn;
  bcir_unit *unit;   /* the whole unit, so a call can be typed by an earlier-defined callee's return */
  const bcir_abi *abi;  /* the target data model the unit is laid out for (long/ptr widths) */
  uint32_t rid, cid;
  uint32_t cl_ctr;   /* unique anonymous compound-literal locals (`_cl<N>`) */
  ctext anontext;                                    /* the emit with its anonymous aggregates spelled (respell_anon) */
  ctext fpdefs; int n_fpdef;                         /* synthesized `typedef RET (*__bcir_fpN)(PARAMS);`
                                                      * lines for direct funcptr-param declarators (which
                                                      * have no source typedef to print), emitted as a
                                                      * prelude before the function bodies */
  ctext tudefs;                                      /* Phase 3 linking: `extern RET NAME(PARAMS);` lines
                                                      * rendered at each PROTOTYPE, emitted as a prelude so
                                                      * the emitted TU compiles standalone and the host
                                                      * LINKER resolves the cross-TU callee. Both preludes
                                                      * grow (CF-BUF): a signature of any length is emitted,
                                                      * where a fixed line buffer had made its emit impossible */
  int saw_static;                                    /* p_type_base scanned a `static` since p_func last
                                                      * reset it -- captured as fn->static_fn right after
                                                      * the return-type parse (source-static honoring) */
  int saw_extern;                                    /* p_type_base scanned an `extern` (a block-scope
                                                      * declaration reads it: CF-STORAGE) */
  int saw_thread;                                    /* ... a `_Thread_local` / `thread_local`: a block-scope
                                                      * static of thread storage duration (CF-TLS) */
  struct { char name[BCIR_CIR_NAME]; bcir_ctype ret;
           const bcir_ctype *params; int n_params;
           int variadic; } *protos;                  /* prototype table: callee -> return
                                                      * type (for call-result typing) and parameter types (a
                                                      * null pointer argument's, CF-NULLARG -- held in the
                                                      * scratch arena, which lasts the compile); grows
                                                      * geometrically */
  int n_protos, cap_protos;
  struct cc_undecl { struct cc_undecl *next; char name[BCIR_CIR_NAME]; } *undecl, *undecl_last;
                                                     /* the callees called before any declaration of theirs, in
                                                      * call order (the scratch arena): the unit's end refuses
                                                      * one the unit then defines or prototypes (C11 6.5.1p2) */
  /* §5.12 per-function mutation pre-pass (the C twin of _mut_assigned / _mut_addr): over the whole
   * function body, the number of assignments to each NAME and whether its address is ever taken (`&x`).
   * Drives extent-stability -- a recovered count / a bound pointer is trusted only when STABLE (assigned
   * at most its single binding and never aliased). Conservative: an over-approximation never promotes
   * unsoundly. Reset (mut_n=0) per function. */
  mutent mut[512]; int mut_n;
  int ext_ctr;       /* §5.12 unique hidden extent-snapshot locals (`__bcir_extK`); reset per function */
  uint32_t unsized[32]; int n_unsized;   /* local arrays declared `T a[]` that no brace initializer sized (the
                                          * resource keeps a placeholder count of 1): `sizeof` refuses them as
                                          * incomplete, as the oracle does. Reset per function. */
  int stmt_expr_declared_bf; /* terminal bare member of `({ ...; member; })`: Clang retains the
                              * bitfield's declared type instead of applying ordinary promotion */
  /* §5.12 deferred VLA-parameter extent bindings (#vlaparam). A param `T a[n]` decays to a pointer and is
   * bound to the prior integer-scalar param `n` via ptr_extent -- BUT only when `n` is STABLE (unmutated,
   * not address-taken), which the body mutation pre-pass (scan_mutations) determines, and that pre-pass runs
   * only AFTER the whole param list is parsed. So at param time we just RECORD the candidate (the decayed
   * pointer's rid, the size param's rid, and its NAME token for the stability gate); after scan_mutations we
   * resolve each, ptrext_set'ing only the stable ones -- exactly the oracle's gate, evaluated at the same
   * point in the pipeline. Reset per function. */
  struct { uint32_t ptr_rid; uint32_t cnt_rid; tok cnt_tok; } vlaext[16]; int n_vlaext;
  int depth;         /* recursion-depth counter for the recursive-descent grammar -- a pathological
                      * deeply-nested input (e.g. 100000 nested `(`, `{{{...}}}`, `int a[((((...))))]`)
                      * would otherwise exhaust the native stack (a DoS: ASan reports `stack-overflow`).
                      * Bumped at the entry of each recursive cycle's entry point (ENTER_REC / LEAVE_REC),
                      * checked against MAXDEPTH; on exceed the parse cleanly fails ("nesting too deep")
                      * and unwinds -- a clean rc-1 PARSE-ERR, never a crash. Mirrors the oracle's parser
                      * recursion guard (bcir/frontends/cfront/cparse.py) so both rails agree on the
                      * boundary (deep input -> a clean fallback/parse-error, not a segfault). */
  int tok_overflow;  /* set by lex() when the input exceeds MAXTOK tokens -- the entry then fails cleanly
                      * ("input too large") and routes to fallback rather than silently truncating the
                      * token stream and mis-compiling a partial unit (Bug B correctness gap). */
  int call_dropped;  /* p_call dropped an operand past BCIR_CLAIM_MAX_RD: its call-site wrapper marks the call
                      * claim `truncated` (the escape analysis then refuses the unit) */
  int td_nd, td_dims[3];   /* the array dims of the typedef the last p_type_base resolved (0: none), for the
                            * declarator that consumes the type to append after its own */
  irange *iw_rng; int iw_nrng, iw_caprng;   /* the initializer walk's stored ranges and union members: one */
  iunion *iw_un; int iw_nun, iw_capun;      /* walk's records sit above the enclosing walk's (a compound
                                             * literal inside an initializer), and end with it */
  int *pure_memo; int npure, cap_pure;      /* CF-TERNARY: an operand C may leave unevaluated, by the token it
                                             * starts at (`2*tok + pure`) -- a nested chain speculates once */
  struct { const char *s; int n, def; } libdef[24]; int n_libdef;   /* unit_defines: each library name asked
                                             * about, and whether the unit defines it (zeroed per compile) */
  fsig *sigs; int nsig, cap_sig;            /* the function types function pointers point at, in the scratch arena
                                             * (a reset of the context drops them with it) */
  unsigned n_void;                          /* void values made in the unit -- a statement reading one is refused
                                             * (`void_read`; none made, none read) */
  char err[256]; int failed;
} CC;
/* The recursion-depth cap for the recursive-descent parser. Comfortably below the real native-stack
 * limit (the deepest cycle, the expression chain p_expr->...->p_primary->`(`->p_expr, is ~8 frames per
 * nesting level, so ~1200 levels is well under an 8MiB stack at ~Nx that depth) yet far ABOVE any real
 * program's nesting in the fixture corpus (the deepest fixture nests only a handful of levels). The
 * Python oracle's recursion guard uses the same cap so the two rails agree on the over-deep boundary. */
#define BCIR_MAXDEPTH 1200
#define BCIR_MAX_PTR_DEPTH 16   /* exceeds C's minimum translation limit; keeps type spellings bounded */
/* Enter a recursive cycle: bump depth, and on overflow record a clean parse error. Pairs with LEAVE_REC.
 * The `_over` flag lets a caller bail with its own (typed) return value after a failed ENTER_REC. */
#define ENTER_REC(c) (++(c)->depth > BCIR_MAXDEPTH ? (fail((c),"nesting too deep"), 1) : 0)
#define LEAVE_REC(c) (--(c)->depth)
/* Checked, two-phase growth. Failure leaves the original object intact and makes the
 * translation fail; the public entry then destroys every partial result. */
static void cc_raise_oom(CC *c) {
  if(!c->failed){snprintf(c->err,sizeof c->err,"oom");c->failed=1;}
  if(c->jump_active) longjmp(c->failure_jump,1);
}

static int cc_ensure(CC *c, void **allocation, int count, int *capacity, size_t element_size) {
  size_t next;
  if(count < *capacity) return 1;
  if(count < 0 || !bcir_host_grow_capacity((size_t)*capacity,(size_t)count+1u,
                                            element_size,&next) || next>(size_t)INT_MAX ||
     !bcir_host_realloc_array(&c->allocator,allocation,(size_t)*capacity,next,
                              element_size,1)){
    cc_raise_oom(c);
    return 0;
  }
  *capacity=(int)next;
  return 1;
}
#define CC_ENSURE(c, arr, n, cap) \
  cc_ensure((c),(void **)&(arr),(n),&(cap),sizeof *(arr))

static int cc_ensure_size(CC *c, void **allocation, size_t count,
                          size_t *capacity, size_t element_size,
                          size_t initial_capacity) {
  size_t next, minimum;
  if(count < *capacity) return 1;
  if(!bcir_size_add(count,1u,&minimum)) goto oom;
  next=*capacity?*capacity:initial_capacity;
  if(next<minimum && !bcir_host_grow_capacity(next,minimum,element_size,&next)) goto oom;
  if(next<minimum || !bcir_host_realloc_array(&c->allocator,allocation,*capacity,next,
                                               element_size,1)) goto oom;
  *capacity=next;
  return 1;
oom:
  cc_raise_oom(c);
  return 0;
}
/* Append to a context text: measured, grown two-phase through the context's allocator (a failure fails the
 * compile), then written -- never truncated (CF-BUF). No va_list is live across the growth, which may unwind. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf,3,4)))
#endif
static void ctext_putf(CC *c, ctext *t, const char *fmt, ...){
  va_list ap; int k; size_t need;
  va_start(ap,fmt); k=vsnprintf(NULL,0,fmt,ap); va_end(ap);
  if(k<0 || !bcir_size_add(t->w,(size_t)k,&need)){ cc_raise_oom(c); return; }
  if(!cc_ensure_size(c,(void **)&t->s,need,&t->cap,1u,256u)) return;
  va_start(ap,fmt); (void)vsnprintf(t->s+t->w,t->cap-t->w,fmt,ap); va_end(ap);
  t->w=need;
}
/* Append `k` bytes of `p` to `t` (a NUL follows them), as ctext_putf appends its formatted text. */
static void ctext_putn(CC *c, ctext *t, const char *p, size_t k){
  size_t need;
  if(!bcir_size_add(t->w,k,&need)){ cc_raise_oom(c); return; }
  if(!cc_ensure_size(c,(void **)&t->s,need,&t->cap,1u,256u)) return;
  if(k) memcpy(t->s+t->w,p,k);
  t->s[need]=0; t->w=need;
}
/* The active target ABI (defaults to the host LP64 model when the driver set none). */
static const bcir_abi *cc_abi(const CC *c){ return c->abi ? c->abi : bcir_abi_host(); }
/* The alignment of a scalar type of `sz` bytes under the target ABI -- the oracle's `CType.align`: a `long
 * double` takes the ABI's long-double alignment, a complex type its element's (a `long double _Complex` the
 * long double's), an 8-byte scalar the ABI's `eight_byte_align` (4 on i386: `double`, `long long`, `int64_t`,
 * a `_BitInt(33..64)`), every other scalar its size -- the oracle's `ctype_model.scalar_align`. The one answer
 * a member's placement and `_Alignof` share; aligning a complex to its size put a `double _Complex` member at
 * 16 where the ABI puts it at 8. */
static int scalar_align(const CC *c, const bcir_ctype *ty, int sz){
  const bcir_abi *a=cc_abi(c);
  int el=ty->is_complex ? sz/2 : sz;                    /* a complex: the element float */
  if(ty->is_float && el==a->long_double_size && el>8) return a->long_double_align;   /* `long double` */
  if(el==8) return a->eight_byte_align;                 /* an 8-byte scalar (or a `double _Complex`'s element) */
  return el<1 ? 1 : el;
}
/* Where Clang (the Itanium layout) starts a non-packed bitfield of `width` bits at the bit cursor `dbits`, for a
 * type of `size` bytes aligned to `align` -- the oracle's `ctype_model.bitfield_start`: at the cursor, unless it
 * would overflow a storage unit of the type's SIZE that begins at an ALIGNMENT boundary; then at the next
 * alignment boundary. With align == size (every type but i386's 8-byte integers) a field never crosses a unit. */
static long long bitfield_start(long long dbits, int width, int size, int align){
  long long fa=(long long)align*8;
  if(dbits%fa + width > (long long)size*8) dbits+=fa-(dbits%fa);
  return dbits;
}
/* The target ABI's atomic promotion -- the oracle's `with_atomic`: an `_Atomic` type no wider than the ABI's
 * `atomic_promote_size` (Clang's MaxAtomicPromoteWidth) rounds its size up to a power of two and aligns to
 * it, so an `_Atomic float _Complex` aligns to 8 and an `_Atomic double _Complex` to 16 on a 64-bit target; a
 * wider one keeps its layout. A pointer's `is_atomic` qualifies its pointee, so a pointer is left alone. */
static void atomic_layout(const CC *c, const bcir_ctype *ty, int *sz, int *al){
  if(!ty->is_atomic || ty->kind==2 || *sz<1 || *sz>cc_abi(c)->atomic_promote_size) return;
  int p=1; while(p<*sz) p<<=1;
  *sz=p; *al=p;
}
/* A type-name's size and alignment under the target ABI (the oracle's `CType.size` / `CType.align`): a
 * pointer is the ABI's, a struct or union its laid-out size and alignment, a scalar its size and
 * `scalar_align`, and an `_Atomic` one then `atomic_layout`. The one answer `sizeof` and `_Alignof` give. */
static void type_layout(const CC *c, const bcir_ctype *ty, int si, int *sz, int *al){
  if(ty->kind==2){ *sz=*al=cc_abi(c)->pointer_size; return; }
  if(ty->kind==1 && si>=0){ *sz=c->s[si].size; *al=c->s[si].align<1?1:c->s[si].align; }
  else { *sz=ty->size; *al=scalar_align(c,ty,ty->size); }
  atomic_layout(c,ty,sz,al);
}

static int is_idc(int c){return c=='_'||(c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9');}
static int is_id0(int c){return c=='_'||(c>='a'&&c<='z')||(c>='A'&&c<='Z');}

/* An integer constant s[0..n) (C11 6.4.4.1), read as the oracle's `clex.int_literal_parts` reads it: the C23 `'`
 * separators dropped, the trailing run of `u`/`U`/`l`/`L` its suffix, then `0x` hex, `0b` binary, a leading `0`
 * octal, else decimal, each digit checked against its base. Its value exactly -- an unsigned constant past
 * LLONG_MAX keeps its 64 bits (it had saturated at LLONG_MAX, and an octal constant was read as decimal) -- whether
 * it is decimal, and its suffix's `u` and count of `l`. NULL, or the reason it is refused: a digit its base does not
 * have, or a value no type in its list can hold -- past ULLONG_MAX, or a decimal one without `u` past LLONG_MAX,
 * whose type C leaves to the implementation (GCC's `__int128`, Clang's `unsigned long long`). */
typedef struct { unsigned long long v; int decimal, u, lr; } intlit;
static const char int_bad[]="invalid integer literal",
                  int_too_large[]="an integer constant too large for every type its base and suffix allow";
static const char *int_literal(const char *s, int n, intlit *o){
  int e=n, nb=0, base=10, skip=0, nd=0, over=0, k=0; char c0=0, c1=0;
  memset(o,0,sizeof *o);
  while(e>0 && (s[e-1]=='u'||s[e-1]=='U'||s[e-1]=='l'||s[e-1]=='L'||s[e-1]=='\'')){   /* the suffix */
    if(s[e-1]=='u'||s[e-1]=='U') o->u=1; else if(s[e-1]!='\'') o->lr++;
    e--; }
  for(int i=0;i<e;i++) if(s[i]!='\''){ if(nb==0) c0=s[i]; else if(nb==1) c1=s[i]; nb++; }
  if(c0=='0' && (c1=='x'||c1=='X')){ base=16; skip=2; }
  else if(c0=='0' && (c1=='b'||c1=='B')){ base=2; skip=2; }
  else if(c0=='0' && nb>1){ base=8; skip=1; }
  o->decimal=base==10;
  for(int i=0;i<e;i++){ char ch=s[i]; int d;
    if(ch=='\'' || k++<skip) continue;
    d = ch>='0'&&ch<='9' ? ch-'0' : (ch|0x20)>='a'&&(ch|0x20)<='f' ? (ch|0x20)-'a'+10 : base;
    if(d>=base) return int_bad;
    if(o->v > (ULLONG_MAX-(unsigned long long)d)/(unsigned long long)base) over=1;
    else o->v=o->v*(unsigned long long)base+(unsigned long long)d;
    nd++; }
  if(!nd) return int_bad;
  if(over || (o->decimal && !o->u && o->v>(unsigned long long)LLONG_MAX)) return int_too_large;
  return NULL;
}

/* Decode a C character constant 'c' to its int value: a single char is its byte value sign-extended
 * as a (signed) char; a multi-character constant 'AB' packs big-endian (Clang/GCC: ('A'<<8)|'B'),
 * read as a 32-bit int. s[0..n) includes the surrounding quotes. */
static long long parse_char(const char *s, int n) {
  int i = 0;
  if (i<n && (s[i]=='L'||s[i]=='u'||s[i]=='U')) {       /* skip an optional wide/UTF prefix L/u/U/u8 */
    if (s[i]=='u' && i+1<n && s[i+1]=='8') i+=2; else i+=1;
  }
  int e = (n>0 && s[n-1]=='\'') ? n-1 : n;
  if (i<n && s[i]=='\'') i++;                            /* the opening quote */
  int bytes[8], nb=0;
  while (i < e && nb < 8) {
    int b;
    if (s[i]=='\\' && i+1 < e) { char c = s[i+1];
      if (c=='x') { i+=2; int v=0;
        while (i<e && ((s[i]>='0'&&s[i]<='9')||((s[i]|0x20)>='a'&&(s[i]|0x20)<='f'))) {
          int d = (s[i]<='9')?s[i]-'0':((s[i]|0x20)-'a'+10); v=v*16+d; i++; }
        b = v & 0xFF; }
      else if (c>='0'&&c<='7') { i++; int v=0,k=0;
        while (k<3 && i<e && s[i]>='0'&&s[i]<='7'){v=v*8+(s[i]-'0');i++;k++;} b=v&0xFF; }
      else { int v; switch(c){case 'n':v=10;break;case 't':v=9;break;case 'r':v=13;break;
             case '\\':v=92;break;case '\'':v=39;break;case '"':v=34;break;case 'a':v=7;break;
             case 'b':v=8;break;case 'f':v=12;break;case 'v':v=11;break;case '?':v=63;break;
             default:v=(unsigned char)c;} b=v; i+=2; }
    } else { b=(unsigned char)s[i]; i++; }
    bytes[nb++]=b;
  }
  if (nb==0) return 0;
  if (nb==1) return (bytes[0]>=128) ? bytes[0]-256 : bytes[0];          /* a single signed char */
  unsigned long long v=0;
  for (int k=0;k<nb;k++) v = ((v<<8) | (unsigned)bytes[k]) & 0xFFFFFFFFu;
  return (v>=0x80000000u) ? (long long)v-(1LL<<32) : (long long)v;       /* an int32 multichar */
}

/* Bytes in a (possibly concatenated) string literal *excluding* the NUL, decoding escapes (a simple
 * \c, an octal \NNN, or a hex \xHH.. each count as one byte). s[0..n) is the spelling incl. quotes;
 * adjacent literals ("a" "b") are walked quote-aware -- bytes are counted only *inside* quotes and the
 * inter-piece whitespace is ignored, so a hex/octal escape can't merge with the next piece's digit. */
static int str_bytes(const char *s, int n) {
  int i=0, cnt=0, inq=0;
  while (i < n) {
    char ch = s[i];
    if (!inq) { if (ch=='"') inq=1; i++; continue; }          /* between pieces: only `"` opens one */
    if (ch=='"') { inq=0; i++; continue; }                    /* closing quote of this piece */
    if (ch=='\\' && i+1 < n) { char c = s[i+1];
      if (c=='x') { i+=2;
        while (i<n && ((s[i]>='0'&&s[i]<='9')||((s[i]|0x20)>='a'&&(s[i]|0x20)<='f'))) i++; }
      else if (c>='0'&&c<='7') { i++; int k=0; while (k<3 && i<n && s[i]>='0'&&s[i]<='7'){i++;k++;} }
      else i+=2;
    } else i++;
    cnt++;
  }
  return cnt;
}

/* The element size of a (possibly prefixed) string literal: plain/u8 = 1 (char), u = 2 (char16_t),
 * U = 4 (char32_t), L the target's wchar_t (4, or 2 on Windows). s[0..n) is the spelling incl. the prefix. */
static int str_elem_size(const CC *c, const char *s, int n) {
  if (n>0) { if (s[0]=='u') return (n>1 && s[1]=='8') ? 1 : 2; if (s[0]=='U') return 4;
             if (s[0]=='L') return cc_abi(c)->wchar_size; }
  return 1;
}

/* String-literal table (the C twin of the oracle's per-function string globals). It stores the full,
 * NUL-terminated spelling so the emitter can render references as the inline literal regardless of
 * length -- lifting the previous BCIR_CIR_NAME (32-byte) cap of carrying the spelling in the resource
 * name -- and so identical literals in a function share one global (dedup). Copies live in the
 * context's per-operation arena. A host-tool concern only (the freestanding IR never sees this). */
/* The full spelling registered for a string-literal resource (by rid), or NULL if rid is not one. */
static const char *strtab_lookup(const bcir_func *fn, uint32_t rid) {
  for (int k = 0; k < fn->n_host_literals; k++)
    if (fn->host_literals[k].rid == rid) return fn->host_literals[k].spelling;
  return NULL;
}

/* §5.12 recoverable extents (the C twin of LoweredFunc.ptr_extent): a pointer local bound to
 * malloc(N*sizeof(T)) / calloc(N, sizeof(T)) carries the RECOVERED element-count variable -- its `p[i]`
 * accesses promote to `masked` and emit `a[BCIR_CHK(rid, idx, <count var>, "func:ptr")]`. The map is
 * context-owned and keyed by the owning function. Reset per translation unit. */
static void ptrext_set(CC *c, bcir_func *fn, uint32_t ptr_rid, uint32_t cnt_rid) {
  for (int k = 0; k < fn->n_ptr_extents; k++)                 /* an existing binding -> overwrite */
    if (fn->ptr_extents[k].ptr_rid == ptr_rid) { fn->ptr_extents[k].count_rid = cnt_rid; return; }
  if(!CC_ENSURE(c,fn->ptr_extents,fn->n_ptr_extents,fn->cap_ptr_extents)) return;
  fn->ptr_extents[fn->n_ptr_extents].ptr_rid = ptr_rid;
  fn->ptr_extents[fn->n_ptr_extents].count_rid = cnt_rid;
  fn->n_ptr_extents++;
}
/* The recovered count-variable rid bound to a pointer rid (in `fn`), or 0 if the pointer has no extent.
 * (A real rid is never 0 -- the allocator starts at 100 -- so 0 is an unambiguous "no binding".) */
static uint32_t ptrext_get(const bcir_func *fn, uint32_t ptr_rid) {
  for (int k = 0; k < fn->n_ptr_extents; k++)
    if (fn->ptr_extents[k].ptr_rid == ptr_rid) return fn->ptr_extents[k].count_rid;
  return 0;
}

static void fail(CC *c,const char *m);
/* CF-BUF: an identifier or a floating constant longer than BCIR_CIR_IDENT_MAX is refused where it is lexed, as the
 * oracle's `clex` refuses it: a name, a tag, an op and an alias in the graph hold that many characters, and a longer
 * one could only be truncated -- a different callee, member or constant. */
#define BCIR_STR2(x) #x
#define BCIR_STR(x) BCIR_STR2(x)
#define BCIR_LONG_ID "an identifier longer than " BCIR_STR(BCIR_CIR_IDENT_MAX) " characters is not supported"
#define BCIR_LONG_FLT "a floating constant longer than " BCIR_STR(BCIR_CIR_IDENT_MAX) " characters is not supported"
static void lex(CC *c, const char *src) {
  const char *p=src;
  static const char *pu[]={"<<",">>","->","==","!=","<=",">=","&&","||",
                           "++","--",
                           "+=","-=","*=","/=","%=","&=","|=","^=",0};
  while (*p) {
    if (*p==' '||*p=='\t'||*p=='\r'||*p=='\n'){p++;continue;}
    if (p[0]=='/'&&p[1]=='/'){while(*p&&*p!='\n')p++;continue;}
    if (p[0]=='/'&&p[1]=='*'){p+=2;while(*p&&!(p[0]=='*'&&p[1]=='/'))p++;if(*p)p+=2;continue;}
    if (*p=='#'){while(*p&&*p!='\n')p++;continue;}   /* preprocessor: L7 */
    if (c->nt>=MAXTOK-1){ c->tok_overflow=1; break; }   /* over MAXTOK: flag it -- the entry fails cleanly
                                                         * (a fallback the oracle agrees with) instead of
                                                         * SILENTLY truncating + mis-compiling (Bug B). */
    tok *t=&c->t[c->nt];
    if (p[0]=='L'||p[0]=='u'||p[0]=='U'){             /* wide/UTF literal prefix L/u/U/u8 before a quote */
      const char *qp=0;
      if (p[0]=='u'&&p[1]=='8'&&(p[2]=='"'||p[2]=='\'')) qp=p+2;
      else if (p[1]=='"'||p[1]=='\'') qp=p+1;
      if (qp){ char q=*qp; t->s=p; const char *r=qp+1;
        while(*r&&*r!=q){ if(*r=='\\'&&r[1]) r+=2; else r++; }
        if(*r==q) r++; t->n=(int)(r-p); p=r;
        if (q=='"'){ t->k=T_STR; } else { t->k=T_INT; t->v=parse_char(t->s,t->n); }
        c->nt++; continue; }
    }
    if (is_id0(*p)){t->k=T_ID;t->s=p;while(is_idc(*p))p++;t->n=(int)(p-t->s);
      if(t->n>BCIR_CIR_IDENT_MAX){ fail(c,BCIR_LONG_ID); break; }
      c->nt++;continue;}
    /* a decimal float literal (digits with a '.' or exponent; .5 / 1.5 / 1e10 / 3.14f). Hex/binary
     * stay integer; a bare integer falls through to T_INT. */
    if ((( *p>='0'&&*p<='9') && !(p[0]=='0'&&((p[1]|0x20)=='x'||(p[1]|0x20)=='b')))
        || (*p=='.'&&p[1]>='0'&&p[1]<='9')) {
      const char *q=p; int hasdig=0,hasdot=0,hasexp=0;
      while((*q>='0'&&*q<='9')||*q=='\''){q++;hasdig=1;}
      if(*q=='.'){hasdot=1;q++; while((*q>='0'&&*q<='9')||*q=='\''){q++;hasdig=1;}}
      if(hasdig&&((*q|0x20)=='e')){ const char *r=q+1; if(*r=='+'||*r=='-')r++;
        if(*r>='0'&&*r<='9'){hasexp=1;q=r; while(*q>='0'&&*q<='9')q++;}}
      if(hasdig&&(hasdot||hasexp)){
        if(*q=='f'||*q=='F'||*q=='l'||*q=='L')q++;
        t->k=T_FLT;t->s=p;t->n=(int)(q-p);p=q;
        if(t->n>BCIR_CIR_IDENT_MAX){ fail(c,BCIR_LONG_FLT); break; }
        c->nt++;continue; }
    }
    /* a hex float literal: 0x<hex>[.<hex>]p[+/-]<dec>[f/F/l/L] -- the binary exponent is mandatory,
     * so a bare 0xFF (no 'p') stays an integer below (matches the oracle's _float_lit_type). */
    if (p[0]=='0' && (p[1]|0x20)=='x') {
      const char *q=p+2; int hd=0;
      #define _ISHEX(ch) (((ch)>='0'&&(ch)<='9')||(((ch)|0x20)>='a'&&((ch)|0x20)<='f'))
      while(_ISHEX(*q)||*q=='\''){q++;hd=1;}
      if(*q=='.'){q++; while(_ISHEX(*q)||*q=='\''){q++;hd=1;}}
      #undef _ISHEX
      if(hd && (*q|0x20)=='p'){ const char *r=q+1; if(*r=='+'||*r=='-')r++;
        if(*r>='0'&&*r<='9'){ q=r+1; while(*q>='0'&&*q<='9')q++;
          if(*q=='f'||*q=='F'||*q=='l'||*q=='L')q++;
          t->k=T_FLT;t->s=p;t->n=(int)(q-p);p=q;
          if(t->n>BCIR_CIR_IDENT_MAX){ fail(c,BCIR_LONG_FLT); break; }
          c->nt++;continue; } }
    }
    if (*p>='0'&&*p<='9'){t->k=T_INT;t->s=p;while(is_idc(*p)||*p=='\'')p++;t->n=(int)(p-t->s);
      intlit L; const char *why=int_literal(t->s,t->n,&L);
      if(why){ char m[96];                               /* refused where it is lexed (the oracle's `parse_int_literal`) */
        if(why==int_bad) snprintf(m,sizeof m,"%s '%.*s'",why,t->n<48?t->n:48,t->s);
        else snprintf(m,sizeof m,"%s",why);
        fail(c,m); break; }
      t->v = L.v>(unsigned long long)LLONG_MAX ? -(long long)(~L.v)-1 : (long long)L.v;   /* its 64 bits */
      c->nt++;continue;}
    if (*p=='"'){t->k=T_STR;t->s=p;p++;                /* string literal (escapes consumed as a unit) */
                 while(*p&&*p!='"'){ if(*p=='\\'&&p[1]) p+=2; else p++; }
                 if(*p=='"')p++; t->n=(int)(p-t->s); c->nt++; continue;}
    if (*p=='\''){t->k=T_INT;t->s=p;p++;               /* character constant -> a folded int const */
                  while(*p&&*p!='\''){ if(*p=='\\'&&p[1]) p+=2; else p++; }
                  if(*p=='\'')p++; t->n=(int)(p-t->s); t->v=parse_char(t->s,t->n); c->nt++; continue;}
    if((p[0]=='<'||p[0]=='>')&&p[1]==p[0]&&p[2]=='='){   /* <<= / >>= -- 3-char shift-compound-assign */
      t->k=T_PUN;t->s=p;t->n=3;p+=3;c->nt++;continue;}
    if(p[0]=='.'&&p[1]=='.'&&p[2]=='.'){             /* `...` -- the variadic ellipsis (one 3-char token) */
      t->k=T_PUN;t->s=p;t->n=3;p+=3;c->nt++;continue;}
    int m=0; for(int j=0;pu[j];j++) if(p[0]==pu[j][0]&&p[1]==pu[j][1]){
      t->k=T_PUN;t->s=p;t->n=2;p+=2;c->nt++;m=1;break;}
    if (m) continue;
    t->k=T_PUN;t->s=p;t->n=1;p++;c->nt++;
  }
  c->t[c->nt].k=T_END;c->t[c->nt].s="";c->t[c->nt].n=0;
}

/* --- token helpers ------------------------------------------------------- */
/* A read-only T_END sentinel returned for any out-of-range token index (a fixed `tok` with kind T_END,
 * empty spelling). Used to bound every lookahead so a near-MAXTOK token stream cannot read past the
 * fixed `c->t[MAXTOK]` array (Bug B: a global-buffer-overflow). The lexer fills `c->t[0..nt-1]` and a
 * T_END at `c->t[nt]` (nt<=MAXTOK-1), so the only valid readable indices are [0, nt]; anything beyond
 * resolves to this sentinel rather than indexing past the array end. */
static const tok BCIR_TOK_END = { T_END, "", 0, 0 };
/* The token at absolute index `idx`, bounded: an index past the last real token (idx>nt) or a negative
 * index yields the T_END sentinel, never an out-of-array read. Every `c->t[c->i+k]` lookahead and every
 * unbounded scan (`j+=2` member chains) routes through this so the parser is OOB-read-safe at the tail. */
static const tok *tat(CC *c,int idx){ return (idx>=0 && idx<=c->nt) ? &c->t[idx] : &BCIR_TOK_END; }
static tok *pk(CC *c){return (c->i>=0 && c->i<=c->nt) ? &c->t[c->i] : (tok *)&BCIR_TOK_END;}
static int is(CC *c,const char *s){tok *t=pk(c);return (int)strlen(s)==t->n&&!strncmp(t->s,s,t->n);}
static int tok_is(const tok *t,const char *s){return (int)strlen(s)==t->n&&!strncmp(t->s,s,t->n);}
static int isk(CC *c,tkind k){return pk(c)->k==k;}
static tok adv(CC *c){
  if(c->i<0 || c->i>c->nt) return BCIR_TOK_END;   /* never index past the T_END slot at c->t[nt] */
  return c->t[c->i++];                            /* (clamping i to nt keeps a runaway parser in-bounds) */
}
static void fail(CC *c,const char *m){if(!c->failed){snprintf(c->err,sizeof c->err,"%s",m);c->failed=1;}}
static int eat(CC *c,const char *s){if(is(c,s)){c->i++;return 1;}fail(c,s);return 0;}
/* A token's spelling into a name-sized buffer (BCIR_CIR_NAME). The lexer bounds an identifier at
 * BCIR_CIR_IDENT_MAX, so every name fits; a longer token is refused, never truncated (CF-BUF). */
static void idcpy(CC *c,char *d,const tok *t){
  int n=t->n<0?0:t->n;
  if(n>BCIR_CIR_IDENT_MAX){ fail(c,BCIR_LONG_ID); n=0; }
  memcpy(d,t->s,(size_t)n);d[n]=0;}
/* A checked format into a fixed field of the graph or the parser -- a name, a tag, an alias, an op. Each field is
 * sized for its longest spelling (bcir_cir.h), and every part of a spelling is bounded (an identifier or a floating
 * constant by the lexer, a pointer depth, a width), so accepted input never reaches the refusal: a spelling that
 * would not fit fails the compile, never truncates (CF-BUF). */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf,4,5)))
#endif
static int fits(CC *c,char *d,size_t n,const char *fmt,...){
  va_list ap; int k;
  va_start(ap,fmt); k=vsnprintf(d,n,fmt,ap); va_end(ap);
  if(k<0||(size_t)k>=n){ fail(c,"a name too long for the claim graph"); if(n) d[0]=0; return 0; }
  return 1;
}
static void ctype_str(const bcir_ctype *ty,char *o,size_t n);   /* a C type's spelling (the emitter's) */
static void ctype_qstr(const bcir_ctype *ty,char *o,size_t n);  /* ... as a function type spells it (CF-QUALS) */
static unsigned qual_key(const bcir_ctype *t);                  /* ... and compares it (CF-QUALS) */

/* --- types --------------------------------------------------------------- */
static int scalar_size(const char *s,int n) {
  struct {const char *k;int sz;} T[]={{"void",0},{"char",1},{"bool",1},{"_Bool",1},{"short",2},
    {"int",4},{"unsigned",4},{"signed",4},{"long",8},{"uint8_t",1},{"int8_t",1},{"uint16_t",2},{"int16_t",2},
    {"uint32_t",4},{"int32_t",4},{"uint64_t",8},{"int64_t",8},
    {"size_t",8},{"intptr_t",8},{"uintptr_t",8},     /* the pointer-tracking size_t-class types */
    {"wchar_t",4},                                   /* the target's wchar_t (p_type_base resolves it) */
    {"float",4},{"double",8},{0,0}};
  for(int i=0;T[i].k;i++) if((int)strlen(T[i].k)==n&&!strncmp(T[i].k,s,n)) return T[i].sz;
  return -1;
}
/* The inherent signedness of a named scalar type (1 signed / 0 unsigned / -1 none: void/float). */
static int scalar_signed(const char *s,int n) {
  const char *U[]={"unsigned","uint8_t","uint16_t","uint32_t","uint64_t","size_t","uintptr_t","bool","_Bool",0};
  const char *S[]={"char","short","int","long","int8_t","int16_t","int32_t","int64_t","intptr_t","signed",0};
  for(int i=0;U[i];i++) if((int)strlen(U[i])==n&&!strncmp(U[i],s,n)) return 0;
  for(int i=0;S[i];i++) if((int)strlen(S[i])==n&&!strncmp(S[i],s,n)) return 1;
  return -1;
}
/* The base integer types that combine with unsigned/signed (so an explicit keyword wins over them). */
static int is_base_int(const char *s,int n) {
  const char *B[]={"char","short","int","long",0};
  for(int i=0;B[i];i++) if((int)strlen(B[i])==n&&!strncmp(B[i],s,n)) return 1;
  return 0;
}
/* The pointer-tracking integer scalars (size_t / intptr_t / uintptr_t): their width is the data
 * model's pointer size. (ptrdiff_t is omitted to match the oracle, whose _SCALAR has no entry.) */
static int is_ptr_tracking(const char *s,int n) {
  const char *T[]={"size_t","intptr_t","uintptr_t",0};
  for(int i=0;T[i];i++) if((int)strlen(T[i])==n&&!strncmp(T[i],s,n)) return 1;
  return 0;
}
/* The size of a floating type (float = 4, double = 8) or -1 if not a floating type. */
static int scalar_float_size(const char *s,int n) {
  if((int)strlen("float")==n&&!strncmp("float",s,n)) return 4;
  if((int)strlen("double")==n&&!strncmp("double",s,n)) return 8;
  return -1;
}
static int find_struct(CC *c,const char *s,int n){
  for(int i=0;i<c->ns;i++) if((int)strlen(c->s[i].tag)==n&&!strncmp(c->s[i].tag,s,n)) return i;
  return -1;
}
static int find_typedef(CC *c,const char *s,int n){
  for(int i=0;i<c->ntd;i++) if((int)strlen(c->td[i].name)==n&&!strncmp(c->td[i].name,s,n)) return i;
  return -1;
}
static int find_enum(CC *c,const char *s,int n){
  for(int i=0;i<c->nec;i++) if((int)strlen(c->ec[i].name)==n&&!strncmp(c->ec[i].name,s,n)) return i;
  return -1;
}
/* The enumerator a name denotes where it is read, or -1: a name the function binds in an enclosing block scope -- a
 * local, a parameter, a loop's own declaration, all in `env` -- hides a file-scope enumerator of its name (C11 6.2.1p4),
 * which is then no constant (CF-ENUMSCOPE; the oracle's `_enumerator`). Reading the enumerator first had folded it in
 * place of the object: `uint32_t N = s + 1u; uint32_t a[N];` became `a[4]` beside an `enum { N = 4 }`. `env` is empty
 * at file scope (the unit loop clears it). */
static int visible_enum(CC *c,const char *s,int n){
  int e=find_enum(c,s,n);
  if(e>=0) for(int i=c->nenv-1;i>=0;i--) if((int)strlen(c->env[i].name)==n&&!strncmp(c->env[i].name,s,n)) return -1;
  return e;
}
static int find_global(CC *c,const char *s,int n){
  for(int i=0;i<c->ngv;i++) if((int)strlen(c->gv[i].name)==n&&!strncmp(c->gv[i].name,s,n)) return i;
  return -1;
}
/* A struct/union tag named before its definition (or never defined): an incomplete sdef slot, which the
 * definition completes in place (CF-SELFREF). An existing tag keeps its slot. */
/* A tag the front end synthesized for an anonymous aggregate (`$anonN`), which no C compiler knows. */
static int is_anon_tag(const char *tag){ return !strncmp(tag,"$anon",5); }
static int declare_struct(CC *c, const tok *tag, int is_union){
  int si=find_struct(c,tag->s,tag->n);
  if(si>=0) return si;
  if(!CC_ENSURE(c, c->s, c->ns, c->cap_s)) return -1;
  si=c->ns++; sdef *S=&c->s[si];
  S->nf=0; S->size=0; S->align=1; S->is_union=is_union; S->vol_storage=0; S->nanon=0; S->incomplete=1;
  S->tdname[0]=0; S->parent=-1; S->member[0]=0; S->mstars=S->mdims=0; S->tdptr[0]=0; S->tdstars=0;
  idcpy(c,S->tag,tag);
  return si;
}
/* Whether a `*` follows the cursor (past qualifiers): the struct type just named is a pointer's pointee. */
static int ptr_follows(CC *c){
  int j=c->i;
  while(tok_is(tat(c,j),"const")||tok_is(tat(c,j),"volatile")||tok_is(tat(c,j),"restrict")||
        tok_is(tat(c,j),"__restrict")||tok_is(tat(c,j),"__restrict__")) j++;
  return tok_is(tat(c,j),"*");
}
/* A struct completed after a pointer to it was laid out (a member `struct b *pb`, a typedef, a global, a
 * prototype's return): each size captured from the incomplete slot takes the layout's, so a pointer's stride
 * and a typedef'd object are right (the oracle completes its type by tag where it is used). */
/* The typedef'd array dimensions the last p_type_base resolved (outer first), into `dims`, and their count --
 * at most 3, the typedef parse's own bound, restated here so every reader's loop is bounded by it. */
static int td_dims_of(const CC *c, int dims[3]){
  int n = c->td_nd<0 ? 0 : c->td_nd>3 ? 3 : c->td_nd;
  for(int d=0; d<3; d++) dims[d] = d<n ? c->td_dims[d] : 0;
  return n;
}
static void complete_struct_refs(CC *c, int si){
  const sdef *S=&c->s[si]; int sz=S->size;
  for(int k=0;k<c->ns;k++) for(int f=0;f<c->s[k].nf;f++){ field *F=&c->s[k].f[f];
    if(F->ptee_sidx==si && (F->is_ptr||F->elem_ptr) && F->ptee_depth<=1) F->ptee_size=sz; }
  for(int k=0;k<c->ntd;k++) if(c->td[k].sidx==si){ bcir_ctype *t=&c->td[k].ty;
    if(t->kind==1 || (t->kind==2 && t->ptr_to_struct && (t->ptr_depth?t->ptr_depth:1)==1)) t->size=sz; }
  for(int k=0;k<c->ngv;k++){ bcir_ctype *t=&c->gv[k].ty;
    if(t->kind==2 && t->ptr_to_struct && (t->ptr_depth?t->ptr_depth:1)==1 && !strcmp(t->tag,S->tag)) t->size=sz; }
  for(int k=0;k<c->n_protos;k++){ bcir_ctype *t=&c->protos[k].ret;
    if(t->kind==2 && t->ptr_to_struct && (t->ptr_depth?t->ptr_depth:1)==1 && !strcmp(t->tag,S->tag)) t->size=sz; }
}
static venv *use_global(CC *c,const tok *id);   /* fwd: materialize a global's resource on first use */
static uint32_t fp_value_temp(CC *c, const char *alias);   /* fwd: a function-pointer value temp (CF-FPTAB) */
static const char *sig_alias(CC *c, int sig);              /* fwd: a function type's alias (CF-FPTAB) */
static uint32_t temp_ptr(CC *c,const bcir_ctype *ty,int si);  /* fwd: a pointer temp of a pointer type */
static void p_enum_body(CC *c);   /* fwd: `{ A, B=expr, C }` -> register the constants */

/* a parsed type: fills a bcir_ctype + the struct index (sidx, or -1). */
/* `T *volatile p`, `volatile ptr_t p`: a pointer object that is itself volatile, every access of which C performs as
 * written (C11 6.7.3p7) -- which no lowering here does, so it is refused rather than dropped (CF-QUALS). The
 * oracle's `VOLATILE_PTR`. */
#define CC_VOLATILE_PTR "a volatile-qualified pointer is not supported"
/* a qualified `*` past the eighth from the base, which `ptr_const` / `ptr_restrict` have no bit for (CF-QUALS) */
#define CC_QUAL_DEEP "a qualified pointer nested more than 8 deep is not supported"
/* `(uint32_t (*p)(uint32_t))f`, `sizeof(uint32_t (*p)(uint32_t))`: a type name declares no identifier (C11 6.7.7p1) -- the
 * function-pointer type name a cast, `sizeof` and `_Alignof` take is abstract on both rails (CF-RTFP; the oracle's
 * `TYPE_NAME_NAMED`) */
#define CC_TYPE_NAME_NAMED "a type name names an identifier"
/* `(uint32_t[2])s`, `(row_t)p`, `(uint32_t (*[2])(uint32_t))f`: a cast names a scalar type or `void` (C11 6.5.4p2), so a
 * type name of an array type is a cast only as a compound literal's, `(T[N]){...}` (CF-RTFP; the oracle's `CAST_ARRAY`) */
#define CC_CAST_ARRAY "a cast to an array type"
/* `(_Atomic uint32_t){x}`: a compound literal of an `_Atomic` type is an `_Atomic` object, which neither rail models
 * (CF-RTFP; the oracle's `ATOMIC_LITERAL`) */
#define CC_ATOMIC_LITERAL "a compound literal of `_Atomic` type is not supported"
/* `(_Atomic uint32_t)x`: C17 6.5.4p5 gives the cast the unqualified type, Clang the `_Atomic` one, which it then rejects
 * as an operand, so the emit could not be held to the original -- refused, as CF-CALIGN refused it (CF-RTFP; the
 * oracle's `CAST_ATOMIC`). A cast to a pointer to an `_Atomic` object lowers. */
#define CC_CAST_ATOMIC "a cast to an `_Atomic` type is not supported"
/* `+p`, `-p`, `~p` of a pointer, `-f` of a function, `~d` of a real floating value, `__real__ p`: an operand C does not
 * give the unary operator (C11 6.5.3.3p1), which Clang rejects; both rails had lowered them (CF-UNARY; the oracle's
 * `UNARY_NOT`, `unary_operand_ok`). */
#define CC_UNARY_NOT "invalid argument type to unary expression"
/* `switch (p)` of a pointer, a floating value, an array or a function: `switch` takes an integer (C11 6.8.4.2p1), and
 * Clang rejects any other; both rails had lowered them (CF-STRUCTCOND; the oracle's `SWITCH_NOT`, `cond_value_ok`). */
#define CC_SWITCH_NOT "statement requires expression of integer type"
/* The (nonexistent) value of a void expression used as a value (C11 6.3.2.2) -- an initializer, an operand, a
 * condition, an argument, a `return` from a non-void function. The twin parses and lowers in one pass, so a void
 * value is a placeholder no claim writes, and a claim of a statement that reads one is the use: refused at the
 * statement's end (`void_read`), and `return f();` from a function returning a value where it is read (CF-VOIDVAL;
 * the oracle's `VOID_VALUE`, which `_rvalue` raises). */
#define CC_VOID_VALUE "the value of a void expression is used"
/* The qualifiers after a `*` (C11 6.7.6.1p1), consumed: `const` and `restrict` into `*cst` / `*rst`, which a function
 * type spells and compares (CF-QUALS). A `volatile` one is refused (`CC_VOLATILE_PTR`), as the oracle's `_star_quals`
 * refuses it. 1 after a failure. */
static int star_quals(CC *c, int *cst, int *rst){
  *cst=*rst=0;
  for(;;){
    if(is(c,"const")){ *cst=1; c->i++; continue; }
    if(is(c,"restrict")||is(c,"__restrict")||is(c,"__restrict__")){ *rst=1; c->i++; continue; }
    if(is(c,"volatile")){ fail(c,CC_VOLATILE_PTR); return 1; }
    return 0;
  }
}
/* The qualifiers of the `*` at `level` -- 1 the first from the base -- onto `ty` (CF-QUALS). 1 after a failure. */
static int star_level(CC *c, bcir_ctype *ty, int level, int cst, int rst){
  if(!cst && !rst) return 0;
  if(level<1 || level>8){ fail(c,CC_QUAL_DEEP); return 1; }
  if(cst) ty->ptr_const=(uint8_t)(ty->ptr_const|(1u<<(level-1)));
  if(rst) ty->ptr_restrict=(uint8_t)(ty->ptr_restrict|(1u<<(level-1)));
  return 0;
}
/* The `*`s a pointer type has: its depth, the `0 == 1` of a kind-2 ctype made without one counted. */
static int ptr_levels(const bcir_ctype *ty){ return ty->kind==2 ? (ty->ptr_depth?ty->ptr_depth:1) : 0; }
/* Past the `*`s at token `k` and their qualifiers (a lookahead; nothing consumed). */
static int skip_stars(CC *c, int k){
  while(tok_is(tat(c,k),"*")){ k++;
    while(tok_is(tat(c,k),"const")||tok_is(tat(c,k),"volatile")||tok_is(tat(c,k),"restrict")
          ||tok_is(tat(c,k),"__restrict")||tok_is(tat(c,k),"__restrict__")) k++; }
  return k;
}
/* Apply a declarator's leading `*`s to a (base) type: each `*` raises the pointer depth (a struct base
 * becomes a pointer-to-struct), keeping the `const` / `restrict` after it (`star_level`, CF-QUALS). Split out of
 * p_type so a multi-declarator declaration applies stars PER DECLARATOR (`int *p, q;` -> p is `int*`, q is int). */
static void apply_stars(CC *c, bcir_ctype *ty) {
  while(is(c,"*")){c->i++;
    int cst, rst; if(star_quals(c,&cst,&rst)) return;
    if(ty->ptr_depth>=BCIR_MAX_PTR_DEPTH){fail(c,"pointer nesting too deep");return;}
    int level=ptr_levels(ty)+1;
    if(ty->kind==1){ty->ptr_to_struct=1;}
    if(ty->kind==3){ty->ptr_to_fp=1; ty->size=cc_abi(c)->pointer_size;}   /* `op_t *p`: its pointee is the function
      * pointer -- its alias, signature and return ride on, and it is pointer-wide on the target (CF-FPTAB) */
    ty->kind=2; ty->ptr_depth=(uint8_t)level;   /* count `*`s: `T**` -> depth 2 */
    if(star_level(c,ty,level,cst,rst)) return;}
}
/* Parse a type SPECIFIER (the base scalar/struct/union/enum/typedef + qualifiers + the data-model size
 * fixups), WITHOUT the declarator `*`s. p_type folds the stars on top; the multi-declarator paths call
 * this and apply_stars per declarator instead. */
static int p_type(CC *c, bcir_ctype *ty, int *sidx);          /* fwd: typeof(type-name) parses recursively */
static int p_cast_type(CC *c, bcir_ctype *ty, int *si);       /* fwd: a type name, a function pointer's too (CF-RTFP) */
static int fp_name_dims(CC *c, bcir_ctype *ty, long long *dims, int *n, int cap);   /* fwd: its dimensions (CF-RTFP) */
static venv *lookup(CC *c, const tok *t);                     /* fwd: typeof(variable) resolves its type */
static int p_typeof_expr(CC *c, bcir_ctype *ty, int *sidx);   /* fwd: typeof(expression) -- speculative lower */
static int res_sig(CC *c, uint32_t rid);   /* fwd: the function type a function-pointer value points at */
/* Does a cast's type-name start at the cursor -- `( type-name )` rather than `( expression )`? The oracle's
 * `_is_cast`: the answer for a cast and a compound literal. Each of the five sites that asks kept its own
 * copy of this list. `(_Atomic T *)p` is a cast to a pointer to an atomic object, and `(_Atomic T)v` one to `T`
 * (CF-RTFP). */
static int type_name_tok(CC *c, const tok *t){
  return scalar_size(t->s,t->n)>=0 || tok_is(t,"struct")||tok_is(t,"union")||tok_is(t,"enum")
      || tok_is(t,"_Complex")||tok_is(t,"complex")||tok_is(t,"_BitInt")
      || tok_is(t,"const")||tok_is(t,"volatile")||tok_is(t,"_Atomic")
      || tok_is(t,"typeof")||tok_is(t,"__typeof__")||tok_is(t,"typeof_unqual")
      || find_typedef(c,t->s,t->n)>=0;
}
static int starts_type_name(CC *c){ return type_name_tok(c,pk(c)); }
/* ... and where a declaration's type may start (the oracle's `_is_decl_start`): `sizeof`, a size operand
 * and `typeof` take `_Atomic T`, as the oracle folds `sizeof(_Atomic T)`. */
static int decl_type_tok(CC *c, const tok *t){ return type_name_tok(c,t); }
static int starts_decl_type(CC *c){ return decl_type_tok(c,pk(c)); }
/* A block statement that starts with token `t` is a declaration (p_stmt_inner; the CF-PAREN pass). */
static int decl_start_tok(CC *c, const tok *t){
  return t->k==T_ID && (scalar_size(t->s,t->n)>=0||tok_is(t,"static")||tok_is(t,"struct")||tok_is(t,"union")
      ||tok_is(t,"enum")||tok_is(t,"const")||tok_is(t,"volatile")
      ||tok_is(t,"extern")                                   /* refused by p_stmt_inner, not an expression */
      ||tok_is(t,"_Thread_local")||tok_is(t,"thread_local")  /* `_Thread_local static T n;` (CF-TLS) */
      ||tok_is(t,"_Complex")||tok_is(t,"complex")            /* `double _Complex z;` -- a complex local */
      ||tok_is(t,"_BitInt")                                  /* `_BitInt(N) z;` -- a bit-precise local */
      ||tok_is(t,"_Atomic")                                  /* `_Atomic int a;` -- an atomic local */
      ||tok_is(t,"va_list")||tok_is(t,"__builtin_va_list")   /* `va_list ap;` -- a variadic cursor local */
      ||tok_is(t,"typeof")||tok_is(t,"__typeof__")||tok_is(t,"typeof_unqual")
      ||find_typedef(c,t->s,t->n)>=0);
}
/* A qualifier on a pointer typedef qualifies the pointer, not what it points to (C11 6.7.8p3: a typedef is the type
 * it names): `const str_t p` of `typedef char *str_t` is `char *const p`, its outer `*` const -- the function pointer
 * itself for `const op_t g` -- and a `volatile` one is a volatile pointer, refused (CF-QUALS; the oracle's typedef
 * merge). 1 after a failure. */
static int td_ptr_qual(CC *c, bcir_ctype *ty, int cst, int vol){
  if(vol){ fail(c,CC_VOLATILE_PTR); return 1; }
  if(!cst) return 0;
  if(ty->kind==3){ ty->is_const=1; return 0; }
  return star_level(c,ty,ptr_levels(ty),1,0);
}
static int p_type_base(CC *c, bcir_ctype *ty, int *sidx) {
  memset(ty,0,sizeof *ty); ty->kind=0; ty->size=4; ty->signd=1; *sidx=-1; c->td_nd=0;
  int td_ptr=0;                                       /* a pointer or function-pointer typedef was named (CF-QUALS) */
  int seen=0, longs=0, ptrtrk=0, sign_explicit=0, floatkw=0, cplxkw=0;   /* longs: `long` (data-model) vs `long
                                                               * long` (8); floatkw / cplxkw: a float/double or a
                                                               * `_Complex` keyword was scanned in THIS specifier */
  for(;;){
    if(is(c,"volatile")){ty->is_volatile=1;c->i++;continue;}
    if(is(c,"_Atomic")){ c->i++;
      if(is(c,"(")){ c->i++; bcir_ctype inner; int isi;    /* `_Atomic ( type-name )` -- atomic type specifier */
        if(p_type(c,&inner,&isi)) return 1; if(!eat(c,")")) return 1;
        int vol=ty->is_volatile, cn=ty->is_const; *ty=inner; ty->is_atomic=1; if(vol)ty->is_volatile=1;
        ty->is_const=(uint8_t)(ty->is_const||cn); *sidx=isi; seen=1; break; }
      ty->is_atomic=1; continue; }
    if(is(c,"static")){c->saw_static=1;c->i++;continue;}   /* recorded: p_func captures it right after
                                                            * its return-type parse (source-static
                                                            * honoring in --linkable); block-scope
                                                            * statics peek the token BEFORE p_type. */
    if(is(c,"extern")){c->saw_extern=1;c->i++;continue;}
    if(is(c,"_Thread_local")||is(c,"thread_local")){c->saw_thread=1;c->i++;continue;}   /* thread storage (CF-TLS) */
    if(is(c,"const")){ty->is_const=1;c->i++;continue;}   /* spelled by a prototype's pointer parameter */
    if(is(c,"inline")){c->i++;continue;}  /* storage class / qualifier */
    if(is(c,"typeof")||is(c,"__typeof__")||is(c,"typeof_unqual")){       /* typeof(type-name) / typeof(var) */
      c->i++; if(!eat(c,"(")) return 1;
      int is_type = starts_decl_type(c);
      if(is_type){ bcir_ctype inner; int isi;                            /* typeof( type-name ), incl. typeof(int*) */
        if(p_type(c,&inner,&isi)) return 1; *ty=inner; *sidx=isi; }
      else {                                                            /* typeof( expression ) operand */
        venv *v = (isk(c,T_ID) && tok_is(tat(c,c->i+1),")")) ? lookup(c,pk(c)) : NULL;
        if(v){ c->i++; *ty=v->type; *sidx=v->sidx; }                    /* a bare in-scope variable -- exact type */
        else if(p_typeof_expr(c,ty,sidx)) return 1; }                   /* any other operand -- speculative lower */
      if(!eat(c,")")) return 1;
      seen=1; break; }
    if(is(c,"signed")){ty->signd=1;sign_explicit=1;ty->size=4;seen=1;c->i++;continue;}   /* `signed` alone ==
                                                          * `signed int`; a following base (char/long/...) overrides */
    if(is(c,"unsigned")){ty->signd=0;sign_explicit=1;ty->size=4;seen=1;c->i++;continue;}
    if(is(c,"_Complex")||is(c,"complex")){ty->is_complex=1;ty->is_float=1;cplxkw=1;seen=1;c->i++;continue;}  /* C99 _Complex
                                                       * (a modifier on a float base; bare _Complex == double) */
    if(is(c,"_BitInt")){                                /* C23 `_BitInt ( N )` -- a bit-precise integer type */
      c->i++; if(!eat(c,"(")){return 1;}
      if(!isk(c,T_INT)){fail(c,"expected the width N in `_BitInt(N)`");return 1;}
      long long n=adv(c).v; if(!eat(c,")")){return 1;}
      /* the supported subset is a single `_BitInt` with at most one of signed/unsigned and 2<=N<=64; a
       * base int keyword already seen, a second `_BitInt`, or an out-of-range width is rejected (the C
       * twin has no fallback -- a clean failure here routes the Python rail to fallback in parity). */
      if(ty->bit_width || (seen && !sign_explicit)){fail(c,"unsupported `_BitInt` type specifier");return 1;}
      if(n<2||n>64){fail(c,"`_BitInt` width is outside the supported range 2..64");return 1;}
      ty->bit_width=(int)n; ty->size=(n<=8)?1:(n<=16)?2:(n<=32)?4:8;   /* storage slot (1/2/4/8 bytes) */
      if(!sign_explicit) ty->signd=1;                   /* `_BitInt(N)` defaults signed (no signed/unsigned kw) */
      seen=1; continue;
    }
    if(is(c,"struct")||is(c,"union")){int isu=is(c,"union");c->i++;tok tag=adv(c);int si=find_struct(c,tag.s,tag.n);
      int ptr=ptr_follows(c);                        /* a pointer to it needs no layout (C11 6.2.5p22) */
      if(si<0 && ptr && tag.k==T_ID) si=declare_struct(c,&tag,isu);   /* defined later, or never (opaque) */
      if(si<0){fail(c,"unknown struct");return 1;}
      if(c->s[si].incomplete && !ptr){fail(c,"the incomplete struct or union has no layout here");return 1;}
      ty->kind=1;ty->size=c->s[si].size;*sidx=si;
      ty->is_union=(uint8_t)c->s[si].is_union;idcpy(c,ty->tag,&tag);seen=1;break;}
    if(is(c,"enum")){c->i++;if(isk(c,T_ID)&&!is(c,"{"))c->i++;   /* `enum [tag] [{...}]` -> int */
      if(is(c,"{"))p_enum_body(c); ty->kind=0;ty->size=4;ty->signd=1;seen=1;break;}
    if(is(c,"va_list")||is(c,"__builtin_va_list")){              /* the variadic cursor type (<stdarg.h>) -- */
      ty->kind=0;ty->is_valist=1;ty->size=cc_abi(c)->pointer_size;ty->signd=0;c->i++;seen=1;break;}  /* opaque, emit `va_list` */
    if(!seen&&isk(c,T_ID)){int ti=find_typedef(c,pk(c)->s,pk(c)->n);   /* a typedef alias */
      if(ti>=0){int vol=ty->is_volatile, at=ty->is_atomic, cn=ty->is_const;*ty=c->td[ti].ty;
        td_ptr=ty->kind==2 || ty->kind==3;
        if(td_ptr){ if(td_ptr_qual(c,ty,cn,vol)) return 1; }   /* the pointer's own (CF-QUALS) */
        else {
          if(vol)ty->is_volatile=1;
          if(cn)ty->is_const=1;                       /* `const T *` of a typedef T */
        }
        if(at)ty->is_atomic=1;                        /* `_Atomic T` of a typedef stays atomic */
        *sidx=c->td[ti].sidx;c->i++;seen=1;
        c->td_nd=c->td[ti].nd; for(int d=0;d<3;d++) c->td_dims[d]=c->td[ti].dims[d];   /* an array typedef */
        if(c->td_nd && ptr_follows(c)){ fail(c,"a pointer to a typedef'd array is not supported"); return 1; }
        if(ty->kind==1 && *sidx>=0 && c->s[*sidx].incomplete && !ptr_follows(c)){   /* a by-value use of a
          * typedef'd struct not yet defined needs its layout */
          fail(c,"the incomplete struct or union has no layout here"); return 1; }
        break;}}
    if(is(c,"wchar_t") && !seen){ const bcir_abi *a=cc_abi(c);   /* a typedef of the target's integer type
      * (the oracle's `abi.wchar_type`): 4-byte signed on x86-64/RISC-V/i386 Linux, unsigned on AArch64 Linux,
      * a 2-byte unsigned on Windows */
      ty->kind=0; ty->size=a->wchar_size; ty->signd=a->wchar_signed; c->i++; seen=1; break; }
    if(isk(c,T_ID)){int sz=scalar_size(pk(c)->s,pk(c)->n);
      if(sz<0){if(seen)break;fail(c,"unknown type");return 1;}
      if(ty->bit_width){fail(c,"unsupported `_BitInt` type specifier");return 1;}   /* a base int after `_BitInt` */
      ty->size=sz;
      int inh=scalar_signed(pk(c)->s,pk(c)->n);          /* the type name's inherent signedness */
      if(inh>=0 && (!sign_explicit || !is_base_int(pk(c)->s,pk(c)->n))) ty->signd=inh;
      if((int)strlen("long")==pk(c)->n&&!strncmp("long",pk(c)->s,pk(c)->n)) longs++;   /* count `long`s */
      if(is_ptr_tracking(pk(c)->s,pk(c)->n)) ptrtrk=1;
      if(scalar_float_size(pk(c)->s,pk(c)->n)>=0){ty->is_float=1;floatkw=1;}     /* float / double */
      if((pk(c)->n==5&&!strncmp("_Bool",pk(c)->s,5))||(pk(c)->n==4&&!strncmp("bool",pk(c)->s,4)))
        ty->is_bool=1;                                               /* a boolean: a store normalizes to 0/1 */
      if(pk(c)->n==4&&!strncmp("char",pk(c)->s,4)&&!sign_explicit)
        ty->is_plain_char=1;        /* plain `char` (no signed/unsigned): emit `char`, NOT int8_t (ARM) */
      seen=1;c->i++;   /* `long double` / `double _Complex` keep scanning the run */
      if(is(c,"long")||is(c,"int")||is(c,"char")||is(c,"double")||is(c,"_Complex")||is(c,"complex"))continue;break;}
    break;
  }
  /* declaration specifiers come in any order (C11 6.7p1): a qualifier or storage class AFTER a struct tag, a
   * typedef name or a scalar keyword run (`uint32_t static n`, `struct s volatile *p`) is the same specifier
   * it is before one -- the oracle's `_TRAILING_SPEC` scan (CF-STORAGE) */
  while(seen){
    if(is(c,"volatile")){ if(td_ptr && td_ptr_qual(c,ty,0,1)) return 1; ty->is_volatile=1; c->i++; continue; }
    if(is(c,"static")){ c->saw_static=1; c->i++; continue; }
    if(is(c,"extern")){ c->saw_extern=1; c->i++; continue; }
    if(is(c,"_Thread_local")||is(c,"thread_local")){ c->saw_thread=1; c->i++; continue; }
    if(is(c,"const")){                                /* after a pointer typedef, the pointer's (CF-QUALS) */
      if(td_ptr){ if(td_ptr_qual(c,ty,1,0)) return 1; } else ty->is_const=1;
      c->i++; continue; }
    if(is(c,"inline")){ c->i++; continue; }
    break;
  }
  if(!seen){fail(c,"expected a type");return 1;}
  if(ty->is_atomic && ty->kind==1){   /* as the oracle refuses it: its promoted layout would reach every
                                       * declaration, copy and extent, and its accesses must be atomic whole */
    fail(c,"an `_Atomic` struct or union is not supported"); return 1; }
  /* apply the target data model: `size_t`-class -> pointer_size; `long double` -> long_double_size; a
   * single `long` -> long_size (`long long` keeps its fixed 8). On the host LP64 model long/ptr are 8. */
  if(ptrtrk) ty->size=cc_abi(c)->pointer_size;
  else if(longs>=1&&ty->is_float) ty->size=cc_abi(c)->long_double_size;   /* `long double` (80/128-bit) */
  else if(longs==1&&!ty->is_float&&ty->kind==0) ty->size=cc_abi(c)->long_size;
  if(cplxkw) ty->size = (floatkw ? ty->size : 8) * 2;   /* a complex is a pair of the element float; a bare
                                                         * `_Complex` (no float kw) is double. Only a complex
                                                         * spelled HERE: one from a typedef, `typeof` or
                                                         * `_Atomic(T)` already has its size, and doubling it
                                                         * again made a `float _Complex` typedef 16 bytes */
  return 0;
}
/* The full type: the specifier + the (first declarator's) `*`s folded in -- the single-declarator /
 * type-name form (params, casts, sizeof/_Alignof, the first declarator of a declaration). */
static int p_type(CC *c, bcir_ctype *ty, int *sidx) {
  if(ENTER_REC(c)){ LEAVE_REC(c); return 1; }   /* depth guard: typeof/_Atomic re-enter p_type */
  int r=p_type_base(c,ty,sidx); if(r){ LEAVE_REC(c); return r; }
  apply_stars(c,ty); LEAVE_REC(c); return 0;
}
/* The return type `ret` (a struct or union: sdef `ret_si`) of the function a function-pointer type `fp` points
 * at, captured where the declarator, member or typedef is parsed: its sign, width and float to type a call's
 * result, a struct or union as its sdef (CF-STRUCTVAL), and `void` explicitly -- a zero width alone cannot tell
 * `void` from a return that was not captured, and a call through a pointer to a void function has no value
 * (CF-VOIDCB). A `void *` return is a pointer, not `void`. */
static void fp_capture_ret(bcir_ctype *fp, const bcir_ctype *ret, int ret_si){
  fp->fp_ret_size=ret->size; fp->fp_ret_signd=(uint8_t)(ret->signd?1:0); fp->fp_ret_float=(uint8_t)(ret->is_float?1:0);
  fp->fp_ret_agg=ret->kind==1 && ret_si>=0 ? ret_si+1 : 0;
  fp->fp_ret_void=(uint8_t)(ret->kind==0 && ret->size==0 && !ret->is_float && !ret->is_valist && ret->bit_width==0);
}
/* A function type into the context's table: `ret` and the `np` parameter types `params` (in the scratch arena),
 * spelled in the emit by `alias` ("" when the source declares it), the type of the designator `fn` ("" if none).
 * Returns 1 + its index -- a kind-3 ctype's `fp_sig`, a uint16_t -- or 0 after a failure. Each entry is a declarator,
 * a typedef, a member or a function the unit spells in at least five tokens, so MAXTOK keeps the table far below
 * the index's range; a unit past it is refused rather than given another entry's type. */
static int sig_add(CC *c, const bcir_ctype *ret, const bcir_ctype *params, int np, const char *alias, const char *fn){
  if(c->nsig>=UINT16_MAX){ fail(c,"too many function-pointer types"); return 0; }
  if(c->nsig==c->cap_sig){                         /* the table is in the scratch arena too: doubled on growth */
    int cap=c->cap_sig?c->cap_sig*2:16; size_t bytes; fsig *ns;
    if(cap<=c->cap_sig || !bcir_size_mul((size_t)cap,sizeof *ns,&bytes) ||
       !(ns=(fsig *)bcir_host_arena_allocate(&c->scratch,bytes,_Alignof(fsig)))){ cc_raise_oom(c); return 0; }
    if(c->nsig) memcpy(ns,c->sigs,(size_t)c->nsig*sizeof *ns);
    c->sigs=ns; c->cap_sig=cap; }
  fsig *s=&c->sigs[c->nsig]; memset(s,0,sizeof *s);
  s->ret=*ret; s->params=params; s->n_params=np;
  if(!fits(c,s->alias,sizeof s->alias,"%s",alias) || !fits(c,s->fn,sizeof s->fn,"%s",fn)) return 0;
  return ++c->nsig;
}
/* `sig_add` of a function type that is `variadic` -- its parameters `params` and then `...` (CF-FPRET). */
static int sig_addv(CC *c, const bcir_ctype *ret, const bcir_ctype *params, int np, const char *alias, const char *fn,
                    int variadic){
  int s=sig_add(c,ret,params,np,alias,fn);
  if(s) c->sigs[s-1].variadic=variadic;
  return s;
}
/* The parameter-type list of a function-pointer declarator or typedef -- the cursor just past its `(`, left on its
 * `)`: `(void)`, or each parameter's type and an optional name, and a trailing `...` after one (`*variadic`,
 * CF-FPRET) -- into the scratch arena (`*out`, `*n`). With `spell`, each type is also appended to the prelude line a
 * synthesized `__bcir_fpN` typedef is being written on, and the line closed. 1 on a parse failure: the caller takes
 * its partial line back out. */
static int fp_param_list(CC *c, int spell, const bcir_ctype **out, int *n, int *variadic){
  bcir_ctype *v=NULL; int np=0, cap=0;
  *out=NULL; *n=0; *variadic=0;
  if(is(c,"void")&&tat(c,c->i+1)->n==1&&tat(c,c->i+1)->s[0]==')'){ c->i++; }   /* `(void)` */
  else if(!is(c,")")) for(;;){ bcir_ctype pt; int psi;
    if(np && is(c,"...")){ c->i++; *variadic=1;           /* `(T, ...)`: a variadic function */
      if(spell) ctext_putf(c,&c->fpdefs,", ...");
      break; }
    if(p_type(c,&pt,&psi)) return 1;
    if(isk(c,T_ID)) c->i++;                                                /* an optional parameter name (ignored) */
    if(spell){ char ps[BCIR_EMIT_TYPE]; ctype_qstr(&pt,ps,sizeof ps); ctext_putf(c,&c->fpdefs,"%s%s",np?", ":"",ps); }
    if(np==cap){ int nc=cap?cap*2:4; size_t bytes; bcir_ctype *nv;
      if(!bcir_size_mul((size_t)nc,sizeof *nv,&bytes) ||
         !(nv=(bcir_ctype *)bcir_host_arena_allocate(&c->scratch,bytes,_Alignof(bcir_ctype)))){ cc_raise_oom(c); return 1; }
      if(np) memcpy(nv,v,(size_t)np*sizeof *nv);
      v=nv; cap=nc; }
    v[np++]=pt;
    if(is(c,",")){c->i++;continue;} break; }
  if(spell) ctext_putf(c,&c->fpdefs,"%s);\n",np?"":"void");
  *out=v; *n=np; return 0;
}

static long long ce_dim(CC *c,int probe);   /* fwd: an array dimension (CF-ENUMFOLD) */
/* Whether the cursor is at an inline function-pointer declarator (CF-FPTAB): `( * NAME ) (`, a pointer to one
 * `( * * NAME ) (`, an array of them `( * NAME [N] ) (` -- one or more `*`, the name (it may be left out where
 * `abstract`), any dimensions, then the parameter list's `(`. 0 if not; else the index of the declarator's `)`.
 * Nothing is consumed. */
static int fp_decl_at(CC *c, int abstract){
  if(!is(c,"(") || !tok_is(tat(c,c->i+1),"*")) return 0;
  int k=skip_stars(c,c->i+1);
  if(tat(c,k)->k==T_ID) k++; else if(!abstract) return 0;
  while(tok_is(tat(c,k),"[")){ int d=0;
    do{ if(tok_is(tat(c,k),"[")) d++; else if(tok_is(tat(c,k),"]")) d--; k++; }while(d>0 && tat(c,k)->k!=T_END);
    if(d>0) return 0; }
  if(!tok_is(tat(c,k),")") || !tok_is(tat(c,k+1),"(")) return 0;
  return k;
}
/* One `*` more on the function-pointer type `ty` (CF-FPTAB): a pointer to a function pointer, whose pointee rides on --
 * the declarator's own `*`s, as `apply_stars` applies a specifier's. */
static void fp_star(CC *c, bcir_ctype *ty){
  if(ty->kind==3){ ty->ptr_to_fp=1; ty->size=cc_abi(c)->pointer_size; }
  ty->kind=2; ty->ptr_depth++;
}
/* The inline function-pointer declarator at the cursor (`fp_decl_at` said it is one), parsed whole (CF-FPTAB): `ret` is
 * the return type the specifier gave (`ret_si` its struct, else -1). Its function type is given a synthesized prelude
 * typedef `__bcir_fpN` to be spelled by, as a plain declarator's is, and comes back in `*fp` (kind 3); `*stars` counts
 * its `*`s past the first (the caller applies them with `fp_star`), `dims`/`*nd` its dimensions (at most three, as for
 * any array here; each an integer constant expression -- a runtime one is refused, as the oracle refuses an array of
 * function pointers of variable length) and
 * `*nm` its name (none where it was left out). 1 after a failure. */
static int fp_inline_decl(CC *c, const bcir_ctype *ret, int ret_si, bcir_ctype *fp, int *stars, int dims[3], int *nd, tok *nm){
  c->i+=2;                                           /* `( *` */
  int ocst, orst; if(star_quals(c,&ocst,&orst)) return 1;   /* the function pointer's own, `(*const fp)` (CF-QUALS) */
  uint8_t scst=0, srst=0;                            /* ... and each `*` around it, `(**const pp)` */
  *stars=0; while(is(c,"*")){ c->i++; (*stars)++;
    int cst, rst; if(star_quals(c,&cst,&rst)) return 1;
    if((cst||rst) && *stars>8){ fail(c,CC_QUAL_DEEP); return 1; }
    if(cst) scst=(uint8_t)(scst|(1u<<(*stars-1)));
    if(rst) srst=(uint8_t)(srst|(1u<<(*stars-1))); }
  memset(nm,0,sizeof *nm); nm->s="";
  if(isk(c,T_ID)) *nm=adv(c);
  *nd=0;
  while(is(c,"[")){ c->i++;
    long long d=is(c,"]") ? 0 : ce_dim(c,1);
    if(c->failed) return 1;
    if(d<0){ fail(c,"a variable-length array of function pointers is not supported"); return 1; }
    if(*nd>=3){ fail(c,"an array of function pointers of more than 3 dimensions is not supported"); return 1; }
    dims[(*nd)++]=(int)d;
    if(!eat(c,"]")) return 1; }
  if(!eat(c,")")||!eat(c,"(")) return 1;
  char rets[BCIR_EMIT_TYPE]; ctype_qstr(ret,rets,sizeof rets);
  size_t line=c->fpdefs.w; const bcir_ctype *ps; int np;
  ctext_putf(c,&c->fpdefs,"typedef %s (*__bcir_fp%d)(",rets,c->n_fpdef);
  int va;
  if(fp_param_list(c,1,&ps,&np,&va)){ c->fpdefs.w=line; if(c->fpdefs.s) c->fpdefs.s[line]=0; return 1; }
  if(!eat(c,")")) return 1;
  memset(fp,0,sizeof *fp); fp->kind=3; fp->size=cc_abi(c)->pointer_size;
  fp_capture_ret(fp,ret,ret_si);
  snprintf(fp->tag,sizeof fp->tag,"__bcir_fp%d",c->n_fpdef); c->n_fpdef++;
  fp->fp_sig=(uint16_t)sig_addv(c,ret,ps,np,fp->tag,"",va);
  /* its own `const` and its `*`s' qualifiers ride the `fp_star`s the caller applies: bit k-1 for the k-th */
  fp->is_const=(uint8_t)ocst; fp->ptr_const=scst; fp->ptr_restrict=srst;
  (void)orst;                                        /* a `restrict` function pointer: its top level, no type's part */
  return c->failed;
}

/* --- struct layout (Clang-compatible; bitfields LSB-first; packed/aligned, L8) --- */
static void attrs(CC *c,int *packed,int *aligned){
  for(;;){
    if(is(c,"__attribute__")){c->i++;eat(c,"(");eat(c,"(");
      while(!is(c,")")&&!isk(c,T_END)&&!c->failed){
        if(is(c,"packed")||is(c,"__packed__")){*packed=1;c->i++;}
        else if(is(c,"aligned")||is(c,"__aligned__")){c->i++;eat(c,"(");
          if(!isk(c,T_INT)){fail(c,"aligned() needs an integer");return;} *aligned=(int)adv(c).v;eat(c,")");}
        else c->i++;
        if(is(c,","))c->i++;}
      eat(c,")");eat(c,")");
    } else if(is(c,"alignas")||is(c,"_Alignas")){c->i++;eat(c,"(");
      /* only the integer-constant form `alignas(N)` is supported; `alignas(type)` routes to fallback
       * (matching the oracle, which also rejects a non-integer operand) rather than silently mis-aligning. */
      if(!isk(c,T_INT)){fail(c,"alignas() needs an integer");return;} *aligned=(int)adv(c).v;eat(c,")");}
    else break;
  }
}
/* Consume a C23 `[[ ... ]]` attribute run (a `[` is a single-char token, so `[[` is two adjacent `[`).
 * Sets *repro if the run names the value-neutral hint `unsequenced` or `reproducible`; every other token
 * (args, `gnu::` namespaces) is scanned over and dropped. Returns 1 if a run was consumed, else 0 (the
 * cursor is left untouched so the caller's normal parse proceeds). Matches the oracle's `_attributes`. */
static int c23_attrs(CC *c,int *repro){
  int any=0;
  while(is(c,"[") && tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='['){
    c->i+=2; any=1;                                  /* consume the opening `[[` */
    while(!(is(c,"]") && tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]==']')){
      if(isk(c,T_END)){fail(c,"unterminated [[...]] attribute");return any;}
      if(is(c,"unsequenced")||is(c,"reproducible")) *repro=1;   /* both fold to one fusion-legality flag */
      c->i++;                                         /* skip every other token (robust to args/namespaces) */
    }
    c->i+=2;                                          /* eat the closing `]]` */
  }
  return any;
}
/* The attributes written AFTER a body's closing brace -- `struct S { ... } __attribute__((packed));` -- read
 * before its members are laid out: `open` is the `{`. A trailing `packed` changes where every member sits,
 * and the members are placed as they are parsed, so it cannot wait for the `}` (the twin used to read it only
 * there, for the aggregate's alignment, so the members kept their natural offsets). The cursor is restored. */
static void trailing_attrs(CC *c, int open, int *packed, int *aligned){
  int depth=0, k=open;
  for(;; k++){
    const tok *t=tat(c,k);
    if(t->k==T_END) return;                          /* unbalanced: the body parse reports it */
    if(t->k==T_PUN && t->n==1 && t->s[0]=='{') depth++;
    else if(t->k==T_PUN && t->n==1 && t->s[0]=='}' && --depth==0) break;
  }
  int save=c->i; c->i=k+1; attrs(c,packed,aligned); c->i=save;
}
static long long ce_dim(CC *c,int probe);   /* fwd: an array dimension, an integer constant expression (CF-ENUMFOLD) */
/* Parse `struct|union [tag] [attrs] { members } [attrs]` (NO trailing `;`). Registers an sdef and
 * returns its index (-1 on error). An anonymous aggregate (no tag, e.g. `typedef struct {...} N;`)
 * gets a synthesized internal tag so a typedef can alias it. */
static int p_struct_body(CC *c) {
  int is_union = is(c,"union");
  c->i++; int packed=0,aligned=0; attrs(c,&packed,&aligned);
  int my=-1;
  if(isk(c,T_ID)&&!is(c,"{")){ int ex=find_struct(c,pk(c)->s,pk(c)->n);
    if(ex>=0 && c->s[ex].incomplete && c->s[ex].nf==0) my=ex; }   /* a declared tag: complete its slot in place,
                                         * so every pointer that named it (its sdef index) sees the layout */
  if(my<0){
    CC_ENSURE(c, c->s, c->ns, c->cap_s);
    if(c->ns>=c->cap_s){ fail(c,"too many struct definitions"); return -1; }
    my=c->ns++; }                       /* claim our slot NOW: an inline aggregate member recurses into
                                         * p_struct_body and must take a LATER slot (and may realloc c->s). */
  sdef *S=&c->s[my]; S->nf=0; S->align=1; S->is_union=is_union; S->vol_storage=0; S->nanon=0; S->size=0;
  S->tdname[0]=0; S->parent=-1; S->member[0]=0; S->mstars=S->mdims=0; S->tdptr[0]=0; S->tdstars=0;
  S->incomplete=1;                      /* until its closing brace: `struct node *next` is fine, a by-value
                                         * `struct node` member of itself is not */
  if(isk(c,T_ID)&&!is(c,"{")){tok tag=adv(c);idcpy(c,S->tag,&tag);}
  else snprintf(S->tag,sizeof S->tag,"$anon%d",my);   /* anonymous: synth a unique tag */
  attrs(c,&packed,&aligned);
  if(is(c,"{")) trailing_attrs(c,c->i,&packed,&aligned);   /* `} __attribute__((packed))` packs these members */
  if(!eat(c,"{"))return -1;
  long long dbits=0;int maxsz=0;   /* dbits: a bit cursor (Itanium/packed layout) */
  while(!is(c,"}")&&!c->failed){
    int mpk=0,maln=0; attrs(c,&mpk,&maln);            /* member-leading `_Alignas(N)`/`aligned(N)`: over-aligns
                                                       * every declarator off this specifier (mpk ignored) */
    /* an INLINE aggregate member `struct {...}` / `union {...}`: parse + register its body, then either PROMOTE
     * its leaves into S (ANONYMOUS: no declarator) or use it as a value-struct type for a named member. */
    bcir_ctype base; int si=-1; int inl=0;
    if(is(c,"struct")||is(c,"union")){
      int save=c->i,pk_=0,al_=0; c->i++; attrs(c,&pk_,&al_);
      if(isk(c,T_ID)&&!is(c,"{"))c->i++; attrs(c,&pk_,&al_);
      int isdef=is(c,"{"); c->i=save;
      if(isdef){ si=p_struct_body(c); if(si<0)return -1; S=&c->s[my];   /* re-fetch: c->s may have realloced */
        memset(&base,0,sizeof base); base.kind=1; base.size=c->s[si].size; base.signd=1;
        base.is_union=(uint8_t)c->s[si].is_union; snprintf(base.tag,sizeof base.tag,"%s",c->s[si].tag); inl=1; }
    }
    if(inl && is(c,";")){                             /* ANONYMOUS member: promote A's leaves at the anon offset */
      sdef *A=&c->s[si];
      int al = mpk?1:(A->align<1?1:A->align); if(maln>al)al=maln;
      if(al>S->align)S->align=al;
      int anon_off=0;
      if(!is_union){ long long a8=(long long)al*8; if(dbits%a8)dbits+=a8-(dbits%a8);
                     anon_off=(int)(dbits/8); dbits+=(long long)A->size*8; }
      if(A->size>maxsz)maxsz=A->size;
      if(A->nf>0){ anon_grp *g=&S->anon[S->nanon++];      /* one initializable subobject over its leaves */
        g->first=S->nf; g->n=A->nf; g->sidx=si; g->off=anon_off; }
      for(int k=0;k<A->nf;k++){
        if(S->nf>=MAXFLD){ fail(c,"too many struct members"); return -1; }
        field nf=A->f[k]; nf.byte_off+=anon_off; S->f[S->nf++]=nf;   /* shift each leaf's offset into S */
      }
      eat(c,";");
      continue;
    }
    if(!inl && p_type_base(c,&base,&si))return -1;
    int btdd[3], btd=td_dims_of(c,btdd); if(inl) btd=0;   /* a typedef'd array type */
    for(;;){                                          /* one or more declarators off one specifier: */
      if(btd && is(c,"*")){ fail(c,"a pointer to a typedef'd array is not supported"); return -1; }
      bcir_ctype ty=base; apply_stars(c,&ty);   /* per-declarator `*`: `int *p, q;` -> p ptr, q scalar */
      if(is(c,":")){                                  /* an UNNAMED `int :3` / ZERO-WIDTH `int :0` bitfield (no
                                                       * name): positions the cursor, NOT a field, no align bump. */
        c->i++; int w=(int)adv(c).v;
        if(ty.is_atomic && ty.kind!=2){ fail(c,"a bit-field of `_Atomic` type is not supported"); return -1; }
        if(!is_union){ int ta=scalar_align(c,&ty,ty.size); long long fa=(long long)ta*8;
          if(w==0){ if(dbits%fa)dbits+=fa-(dbits%fa); }       /* zero-width -> the next boundary of its type's
                                                               * ALIGNMENT, packed or not (as Clang does) */
          else if(packed){ dbits+=w; }                        /* packed: pack bit-by-bit */
          else { dbits=bitfield_start(dbits,w,ty.size,ta)+w; }
        }
        if(is(c,",")){c->i++;continue;} break;
      }
      tok nm; int mpre_nd=0, mpre_dims[3]={0,0,0};
      int mk=fp_decl_at(c,0);
      if(mk && mk!=c->i+3){               /* `RET (*fn[N])(P)`, `RET (**pp)(P)`: a member table of, or pointer to,
                                          * function pointers (CF-FPTAB), parsed whole; its dims are the member's */
        bcir_ctype rty=ty, fty; int stars;
        if(fp_inline_decl(c,&rty,si,&fty,&stars,mpre_dims,&mpre_nd,&nm)) return -1;
        ty=fty; for(int s=0;s<stars;s++) fp_star(c,&ty);
      } else
      if(is(c,"(") && tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='*'
         && tat(c,c->i+2)->k==T_ID){       /* a function-pointer member `RET (*name)(params)` -> a kind-3 (8-byte)
                                          * field. The struct definition comes from the source (not emitted), so
                                          * no typedef is synthesized; set via a funcptr value + called via
                                          * `o->fn(args)` (the existing c.call.imember machinery). */
        bcir_ctype rty=ty;                                   /* snapshot the parsed RETURN type before the memset */
        c->i+=2; nm=adv(c);                                  /* `( *` then the name */
        if(!eat(c,")")||!eat(c,"("))return -1;
        const bcir_ctype *ps; int np, va;                    /* its parameters type a null pointer argument */
        if(fp_param_list(c,0,&ps,&np,&va)) return -1;
        eat(c,")");                                          /* past the parameter-type list */
        memset(&ty,0,sizeof ty); ty.kind=3; ty.size=cc_abi(c)->pointer_size; ty.signd=0;
        fp_capture_ret(&ty,&rty,si);                         /* its return type types a c.call.imember result */
        ty.fp_sig=sig_addv(c,&rty,ps,np,"","",va);
      } else {
        if(!isk(c,T_ID)){ fail(c,"expected member name"); return -1; }   /* `unsigned x, y, z;` etc. */
        nm=adv(c);
      }
      int arr_count=0,nadims=0,adims[3]={0,0,0};        /* T arr[N] / T m[A][B] -- one or more dims */
      for(int d=0; d<mpre_nd; d++){ int dim=mpre_dims[d];   /* an inline table's own dims (CF-FPTAB) */
        if(nadims<3){ adims[nadims]=dim; }
        nadims++; arr_count = arr_count ? arr_count*dim : dim; }
      while(is(c,"[")){ c->i++; long long d=is(c,"]") ? 0 : ce_dim(c,0);   /* an integer constant expression */
        if(d<0) return -1;
        int dim=(int)d; eat(c,"]");
        if(nadims<3)adims[nadims]=dim; nadims++; arr_count = arr_count ? arr_count*dim : dim; }
      if(inl && si>=0 && is_anon_tag(c->s[si].tag) && c->s[si].parent<0 && !c->s[si].tdname[0]){
        sdef *A=&c->s[si];                              /* `struct {...} m;`: the anonymous type is m's -- its first
                                                         * declarator's (`anon_spelling`) */
        A->parent=my; idcpy(c,A->member,&nm); A->mstars=ty.kind==2?(ty.ptr_depth?ty.ptr_depth:1):0; A->mdims=nadims; }
      for(int d=0; d<btd; d++){ int dim=btdd[d];      /* a typedef'd array's dims follow the declarator's own */
        if(nadims<3)adims[nadims]=dim; nadims++; arr_count = arr_count ? arr_count*dim : dim; }
      if(nadims>3){ fail(c,"member array of more than 3 dimensions"); return -1; }   /* adims[] caps at 3 */
      int width=0; if(is(c,":")){c->i++;width=(int)adv(c).v;}          /* per-declarator bitfield width */
      if(width && ty.is_atomic && ty.kind!=2){          /* as the oracle refuses it: GCC and Clang reject it */
        fail(c,"a bit-field of `_Atomic` type is not supported"); return -1; }
      if(ty.bit_width>0 && width && !(width>=1 && width<=ty.bit_width)){   /* a `_BitInt(N)` BITFIELD: W in 1..N */
        fail(c,"a `_BitInt` bitfield width outside 1..N is not supported"); return -1; }   /* W>N is invalid C */
      /* a `_BitInt(N)` BITFIELD `_BitInt(N) m : W` (1<=W<=N) is first-class: `ty.size` is the Clang storage slot
       * (1/2/4/8 bytes) so it packs into the `_BitInt(N)` storage unit LSB-first exactly like a standard-int
       * bitfield of that size (byte-identical to Clang); a PLAIN `_BitInt(N)` member likewise uses that slot.
       * Either way the member's exact width rides in f->bit_width so the load/store + emit spell `_BitInt(N)`. */
      if(S->nf>=MAXFLD){ fail(c,"too many struct members"); return -1; }   /* f[] embedded; guarded */
      int isptr=(ty.kind==2 && !arr_count);            /* a (non-array) pointer member: ABI pointer_size */
      int elptr=(ty.kind==2 && arr_count);             /* an array of pointers: each ELEMENT is pointer_size -- the
                                                        * pointee's size here put `uint32_t *arr[2]` at 4-byte
                                                        * elements and truncated every stored pointer */
      int sz=(isptr||elptr)?cc_abi(c)->pointer_size:ty.size;
      /* a (array of) value-struct/union member aligns to the NESTED type's alignment, not its size --
       * `struct{int;struct Big t;}` puts t at the struct's align, not at sizeof(Big) (which over-pads). */
      int al = (ty.kind==1 && !ty.ptr_to_struct && si>=0) ? (c->s[si].align<1?1:c->s[si].align)
             : ty.kind==2 ? (sz<1?1:sz)
             : scalar_align(c,&ty,sz);                 /* a scalar: its ABI alignment (a complex its element's,
                                                        * a `long double` the ABI's) */
      atomic_layout(c,&ty,&sz,&al);                   /* an `_Atomic` member: the ABI's atomic promotion */
      int tal=al;                                     /* the type's own alignment: a bitfield's FieldAlign */
      if(packed) al=1;                                /* packed wins over every natural alignment */
      if(maln>al) al=maln;                            /* `_Alignas(N)`/`aligned(N)` over-aligns (survives packed) */
      field *f=&S->f[S->nf++];
      int total=arr_count?sz*arr_count:sz;             /* the bytes the member occupies (array: N*elem) */
      idcpy(c,f->name,&nm);f->size=sz;f->access_bytes=sz;f->signd=ty.signd;f->bit_w=width;f->arr_count=arr_count;
      f->bit_width=(!isptr && !elptr && !arr_count && ty.bit_width>0)?ty.bit_width:0;   /* a C23 `_BitInt(N)` member (plain OR
                                                                              * bitfield): exact N; f->bit_w holds W */
      f->is_float=(!isptr && !elptr && ty.is_float)?1:0;   /* a float/double member loads/stores as itself */
      f->is_complex=(!isptr && !elptr && ty.is_complex)?1:0;   /* a `_Complex` member: load/store as the complex pair,
                                                         * NOT a same-size real (16B would wrongly read as long double) */
      f->is_bool=(!isptr && !elptr && ty.is_bool)?1:0;  /* a _Bool member: a store normalizes any nonzero to 1 */
      f->is_plain_char=(!isptr && !elptr && ty.is_plain_char)?1:0;/* a plain `char` member: read as `char` (impl-defined
                                                         * sign), NOT int8_t -- `char` is UNSIGNED on AArch64 */
      f->nadims=nadims; for(int z=0;z<3;z++) f->adims[z]=adims[z];
      f->is_ptr=isptr; f->elem_ptr=elptr;
      f->ptee_size=(isptr||elptr)?ty.size:0; f->ptee_float=(isptr||elptr)?(ty.is_float?1:0):0;   /* pointee type */
      f->ptee_sidx=((isptr||elptr) && ty.ptr_to_struct)?si:-1;  /* a pointer-to-struct member: the pointee struct tag */
      f->ptee_depth=(isptr||elptr)?(ty.ptr_depth?ty.ptr_depth:1):0;
      f->is_volatile=(ty.kind!=2 && ty.kind!=3 && ty.is_volatile)?1:0;   /* `volatile T m` (a pointer's `volatile`
                                                                           * qualifies its pointee, not itself) */
      f->ptee_volatile=(ty.kind==2 && ty.is_volatile)?1:0;               /* `volatile T *m`: the pointee */
      f->is_atomic=(ty.kind!=2 && ty.kind!=3 && ty.is_atomic)?1:0;       /* `_Atomic T m`: its own storage */
      f->ptee_atomic=(ty.kind==2 && ty.is_atomic)?1:0;                   /* `_Atomic T *m`: the pointee */
      f->sidx = (ty.kind==1 && !ty.ptr_to_struct && !arr_count) ? si : -1;   /* value struct member -> nested */
      f->elem_sidx = (ty.kind==1 && !ty.ptr_to_struct && arr_count) ? si : -1;   /* array-of-structs element struct */
      f->fp_ret_size=ty.fp_ret_size; f->fp_ret_signd=ty.fp_ret_signd; f->fp_ret_float=ty.fp_ret_float;   /* funcptr member: return type */
      f->fp_ret_agg=ty.fp_ret_agg; f->fp_ret_void=ty.fp_ret_void; f->fp_sig=ty.fp_sig;
      if(al>S->align)S->align=al; if(total>maxsz)maxsz=total;
      if(is_union){f->byte_off=0;f->bit_off=0;}        /* union: every member overlaps at offset 0 */
      else if(width){int ub=sz*8;
        if(packed){                                     /* packed: pack bit-by-bit, NO storage-unit reservation
          * (Clang/GCC) -- the field sits at the running bit cursor and its access unit is just the bytes it
          * spans (`access_bytes`), which may straddle byte/word boundaries; the struct stays align 1. */
          int P=(int)dbits; f->byte_off=P/8; f->bit_off=P%8;
          f->access_bytes=(f->bit_off+width+7)/8; dbits+=width;
        }else{                                          /* natural: pack at the bit cursor unless it would
                                                         * overflow its storage unit (`bitfield_start`) */
          dbits=bitfield_start(dbits,width,sz,tal);
          if(tal==sz){                                  /* the size-aligned storage unit holding it */
            int uoff=(int)(dbits/ub)*sz; f->byte_off=uoff;f->bit_off=(int)(dbits-(long long)uoff*8);
          }else{                                        /* an under-aligned type (i386 `long long`): its unit
            * can run past the struct's end (`struct { char c; long long x : 8; }` is 4 bytes), so the field
            * sits at its first byte and is accessed over the bytes it spans, as a packed one is (the oracle's
            * `narrow_bitfield`) */
            f->byte_off=(int)(dbits/8); f->bit_off=(int)(dbits%8); f->access_bytes=(f->bit_off+width+7)/8;
          }
          dbits+=width;
        }
      }else{long long a8=(long long)al*8;if(dbits%a8)dbits+=a8-(dbits%a8);
        f->byte_off=(int)(dbits/8);f->bit_off=0;dbits+=(long long)total*8;}
      if(is(c,",")){c->i++;continue;}                 /* another member off the same specifier */
      break;
    }
    eat(c,";");
  }
  eat(c,"}"); attrs(c,&packed,&aligned);
  S=&c->s[my]; S->vol_storage=0;                       /* volatile storage anywhere in the object (not through a
                                                        * pointer member): an access through a pointer to it is a
                                                        * device access (the registry's region is MMIO) */
  for(int k=0;k<S->nf;k++){ const field *mf=&S->f[k];
    if(mf->is_volatile || (mf->sidx>=0 && c->s[mf->sidx].vol_storage)
       || (mf->elem_sidx>=0 && c->s[mf->elem_sidx].vol_storage)) S->vol_storage=1; }
  int salign = packed ? 1 : S->align; if(aligned>salign) salign=aligned;
  if(salign<1) salign=1;               /* an alignment is at least 1: the size below is rounded to it */
  S->align=salign;
  int total = is_union ? maxsz : (int)((dbits+7)/8);   /* union size = the widest member; struct: bits->bytes */
  if(total%salign)total+=salign-(total%salign); S->size=total;
  S->incomplete=0; complete_struct_refs(c,my);
  return my;
}

/* --- enum + typedef (resolved at parse time so the claim graph carries the folded result) --- */
/* An integer constant expression -- an enumerator's value, a case label, an array dimension, a designator -- folded
 * where it is parsed, in C's own types on the target (the oracle's `lower.fold_constant`; CF-ENUMFOLD): each operation
 * by the predicate a static's initializer folds with (`kbin`, `kun`, `ksel`, `kconvert`), over its tokens where `kfold`
 * reads claims. Its leaves are an integer constant in its C11 6.4.4.1 type, a character or enumeration constant (an
 * int) and a cast to an integer type; an operand C does not evaluate (`live` 0: the right one of `0 && e`, the arm `?:`
 * does not take) is typed, the predicate run on a zero and a one of its operands' types, never refused for its
 * arithmetic. Anything else -- an object, a call, `sizeof`, the comma, a floating constant, a pointer -- is no integer
 * constant expression (6.6p6), and neither is arithmetic C leaves undefined: `long long` arithmetic had folded `-1 > 5`
 * for `~0u > 5`, and a division by zero to 0. */
typedef struct { unsigned long long p; int bits, sgn, kind; } kval;   /* a folded constant: its value's 64-bit two's
  * complement, and its type -- an integer (kind 0) of `bits` bits, signed or not, a _Bool (1), a null pointer (2) */
static kval kint(unsigned long long p,int bits,int sgn);
static long long kll(unsigned long long p);
static int kpromote(const kval *a,kval *o);
static int kbin(const char *suf,const kval *ia,const kval *ib,kval *out);
static int kun(const char *suf,const kval *ia,kval *out);
static int ksel(const kval *c0,const kval *ia,const kval *ib,kval *out);
static int kconvert(const kval *a,const kval *t,kval *out);
static void lit_int_type(const char *s,int n,int lsz,int *size,int *signd);
#define CE_NOTCONST "not an integer constant expression"   /* the oracle's `ICE_NOT` */
#define CE_NOTINT "an enumerator value not representable as int"   /* the oracle's `ENUM_NOT_INT` (6.7.2.2p2) */
#define CE_DIMRANGE "an array dimension outside 0..INT_MAX"   /* the oracle's `DIM_RANGE` */
/* `a OP b` (`kbin`), or `OP a` (`kun`, `b` NULL), of operands C evaluates (`live`) -- else the operation's type alone,
 * on a zero and a one of its operands' types, its value 0. 0 when it is no constant. */
static int ce_op(const char *suf,int live,const kval *a,const kval *b,kval *out){
  if(live) return b ? kbin(suf,a,b,out) : kun(suf,a,out);
  kval z=*a, o=b?*b:*a; z.p=0; o.p=1;
  if(!(b ? kbin(suf,&z,&o,out) : kun(suf,&z,out))) return 0;
  out->p=0; return 1; }
/* A folded constant's value -- an unsigned one past LLONG_MAX held at LLONG_MAX, past every range a caller takes, as
 * its value is (the oracle reads the value itself). */
static long long kvalue(const kval *v){
  return v->sgn ? kll(v->p) : v->p>(unsigned long long)LLONG_MAX ? LLONG_MAX : (long long)v->p; }
/* A constant spelled exactly in its own type, as both emits spell one (the oracle's `_const_spelling`): `N`, `-N`, `Nu`
 * past LLONG_MAX, and the one negative with no positive counterpart as an expression. `neg`: `v` is a negative value's
 * two's complement. */
static int kdigits(char *d,size_t n,unsigned long long v,int neg){
  return !neg ? snprintf(d,n,v>(unsigned long long)LLONG_MAX ? "%lluu" : "%llu",v)
       : v==(1ull<<63) ? snprintf(d,n,"(-9223372036854775807 - 1)") : snprintf(d,n,"-%llu",0ull-v); }
/* A cast's type-name, the cursor past its `(`, as the type `kconvert` takes (the oracle's `_ktype` of
 * `_resolve_member_type`): an integer type only, the one an integer constant expression converts to (6.6p6).
 * BCIR_NOINLINE: its bcir_ctype stays out of the recursive frames (docs/languages/C_MEMORY_DISCIPLINE.md). */
static BCIR_NOINLINE int ce_cast_type(CC *c,kval *t){
  bcir_ctype ty; int si=-1; memset(t,0,sizeof *t);
  if(p_type(c,&ty,&si) || c->td_nd || !is(c,")")) return 0;
  c->i++;
  if(ty.kind!=0 || ty.is_float || ty.is_valist || ty.bit_width || ty.size<1 || ty.size>8) return 0;
  if(ty.is_bool){ t->kind=1; t->bits=1; return 1; }
  t->bits=ty.size*8; t->sgn=ty.signd; return 1; }
static int ce_expr(CC *c,int minp,int live,kval *out);
/* An operand of `ce_expr` -- a unary operator's, a cast, a parenthesized expression, an integer or character constant,
 * an enumerator -- or 0 when it is none of these. */
static int ce_primary(CC *c,int live,kval *out){
  if(ENTER_REC(c)){ LEAVE_REC(c); return 0; }   /* depth guard: a unary operator's or a cast's operand nests here */
  int ok=0; kval a, t;
  const char *suf = is(c,"-") ? "neg" : is(c,"~") ? "bnot" : is(c,"!") ? "lnot" : is(c,"+") ? "" : NULL;
  if(isk(c,T_INT)){ tok k=adv(c); int sz,sg; lit_int_type(k.s,k.n,cc_abi(c)->long_size,&sz,&sg);
    *out=kint((unsigned long long)k.v,sz*8,sg); ok=1; }
  else if(isk(c,T_ID)){ int e=visible_enum(c,pk(c)->s,pk(c)->n);   /* a hidden one is an object: no constant */
    if(e>=0){ c->i++; *out=kint((unsigned long long)c->ec[e].val,32,1); ok=1; } }
  else if(is(c,"(")){ c->i++;
    if(type_name_tok(c,pk(c))) ok = ce_cast_type(c,&t) && ce_primary(c,live,&a) && kconvert(&a,&t,out);
    else if(ce_expr(c,0,live,out) && is(c,")")){ c->i++; ok=1; } }
  else if(suf){ c->i++;
    ok = ce_primary(c,live,&a) && (*suf ? ce_op(suf,live,&a,NULL,out) : kpromote(&a,out)); }
  LEAVE_REC(c); return ok;
}
static int ce_expr(CC *c,int minp,int live,kval *out){
  if(ENTER_REC(c)){ LEAVE_REC(c); return 0; }   /* depth guard: ce_expr<->ce_primary `(...)` cycle */
  /* C's precedence over ||, &&, bit ops, equality, relational, shift, arithmetic; the right operand of `&&` (2) and `||`
   * (1) evaluates only when the left one does not decide, and `?:` evaluates one arm (6.5.13-15) */
  static const struct { const char *t, *s; int p; } P[]={{"||","lor",1},{"&&","land",2},{"|","or",3},{"^","xor",4},
    {"&","and",5},{"==","eq",6},{"!=","ne",6},{"<=","le",7},{">=","ge",7},{"<","lt",7},{">","gt",7},{"<<","shl",8},
    {">>","shr",8},{"+","add",9},{"-","sub",9},{"*","mul",10},{"/","div",10},{"%","mod",10},{0,0,0}};
  int ok=ce_primary(c,live,out);
  while(ok){ int k=-1; kval r;
    for(int i=0;P[i].t;i++) if(is(c,P[i].t)){ k=i; break; }
    if(k<0 || P[k].p<minp) break;
    c->i++;
    ok = ce_expr(c,P[k].p+1,live && (P[k].p>2 || (out->p!=0)==(P[k].p==2)),&r) && ce_op(P[k].s,live,out,&r,out);
  }
  if(ok && minp==0 && is(c,"?")){                 /* the conditional (right-assoc, lowest precedence) */
    kval a, b; c->i++;
    ok = ce_expr(c,0,live && out->p!=0,&a) && is(c,":");
    if(ok){ c->i++; ok = ce_expr(c,0,live && out->p==0,&b) && ksel(out,&a,&b,out); }
  }
  LEAVE_REC(c); return ok;
}
/* The integer constant expression at the cursor, folded: 1 with its value and type; 0 when it is none -- refused
 * (`CE_NOTCONST`) unless `probe`, which leaves the cursor, the failure state and every struct or enumerator the attempt
 * declared as they were, for the caller to read a runtime dimension instead. */
static int ce_fold(CC *c,int probe,kval *out){
  int save=c->i, failed=c->failed, ns=c->ns, nec=c->nec;
  if(ce_expr(c,0,1,out)) return 1;
  if(probe){ c->i=save; c->failed=failed; c->ns=ns; c->nec=nec; return 0; }
  fail(c,CE_NOTCONST); return 0;
}
/* ... its value (`kvalue`) where a designator or a type-name's dimension takes one, 0 with it refused. */
static long long ce_value(CC *c){ kval v; return ce_fold(c,0,&v) ? kvalue(&v) : 0; }
/* An array declarator's dimension, the cursor past its `[` (the oracle's `cparse._dim`): its value when it is an integer
 * constant expression ending at the `]` -- the array then a fixed one, whatever the expression's form (6.7.6.2p4) --
 * refused outside 0..INT_MAX (`CE_DIMRANGE`). -1 when it is none: refused (`CE_NOTCONST`), or with `probe` the cursor
 * left at the dimension, for a block scope's caller to read a runtime one. BCIR_NOINLINE: its locals stay out of the
 * recursive statement parser's frame. */
static BCIR_NOINLINE long long ce_dim(CC *c,int probe){
  int save=c->i; kval v;
  if(!ce_fold(c,probe,&v)) return -1;
  if(!is(c,"]")){ if(probe){ c->i=save; return -1; } fail(c,CE_NOTCONST); return -1; }
  long long d=kvalue(&v);
  if(d<0 || d>INT_MAX){ fail(c,CE_DIMRANGE); return -1; }
  return d;
}
/* `{ A, B = expr, C }`: each enumerator its C value -- the previous one's plus one, or its integer constant expression
 * folded (`ce_fold`) -- registered so a later use reads that constant. An enumeration constant is an int (6.4.4.3): a
 * value no int holds, given or counted on from INT_MAX, is refused (`CE_NOTINT`, 6.7.2.2p2), never cut to one. */
static void p_enum_body(CC *c){
  eat(c,"{"); long long val=0;
  while(!is(c,"}")&&!c->failed){
    tok nm=adv(c);
    if(is(c,"=")){ kval v; c->i++; if(!ce_fold(c,0,&v)) return; val=kvalue(&v); }
    if(val<INT_MIN || val>INT_MAX){ fail(c,CE_NOTINT); return; }
    CC_ENSURE(c,c->ec,c->nec,c->cap_ec);
    if(c->nec<c->cap_ec){idcpy(c,c->ec[c->nec].name,&nm);c->ec[c->nec].val=val;c->nec++;}
    val++;
    if(is(c,","))c->i++;
  }
  eat(c,"}");
}
static void p_typedef(CC *c){
  c->i++;                                            /* `typedef` */
  bcir_ctype ty; int sidx=-1; memset(&ty,0,sizeof ty); ty.size=4; ty.signd=1;
  if(is(c,"struct")||is(c,"union")){                 /* alias an aggregate (named, anon, or by tag) */
    int save=c->i,pk_=0,al_=0; c->i++; attrs(c,&pk_,&al_);
    if(isk(c,T_ID)&&!is(c,"{"))c->i++; attrs(c,&pk_,&al_);
    int isdef=is(c,"{"); c->i=save;
    if(isdef){int si=p_struct_body(c);if(si<0)return; ty.kind=1;ty.size=c->s[si].size;sidx=si;
      ty.is_union=(uint8_t)c->s[si].is_union;snprintf(ty.tag,sizeof ty.tag,"%s",c->s[si].tag);}
    else { int isu=is(c,"union"); c->i++; tok tag=adv(c); int si=find_struct(c,tag.s,tag.n);
      if(si<0 && tag.k==T_ID) si=declare_struct(c,&tag,isu);   /* `typedef struct node node_t;` ahead of it */
      if(si<0){fail(c,"unknown struct in typedef");return;} ty.kind=1;ty.size=c->s[si].size;sidx=si;
      ty.is_union=(uint8_t)c->s[si].is_union;idcpy(c,ty.tag,&tag);}
    int stars=0;
    while(is(c,"*")){c->i++; stars++;
      int cst, rst; if(star_quals(c,&cst,&rst)) return;   /* each `*` keeps its qualifiers (CF-QUALS) */
      if(ty.kind==1) ty.ptr_to_struct=1;
      ty.kind=2; ty.ptr_depth=(uint8_t)stars;
      if(star_level(c,&ty,stars,cst,rst)) return;}
    if(stars && sidx>=0 && is_anon_tag(c->s[sidx].tag) && !c->s[sidx].tdptr[0] && isk(c,T_ID) && !tok_is(tat(c,c->i+1),"[")){
      idcpy(c,c->s[sidx].tdptr,pk(c)); c->s[sidx].tdstars=stars; }   /* `typedef struct {...} *PP;` (anon_spelling) */
  } else if(is(c,"enum")){                            /* alias an enum -> an int scalar */
    c->i++; if(isk(c,T_ID)&&!is(c,"{"))c->i++; if(is(c,"{"))p_enum_body(c);
    ty.kind=0; ty.size=4; ty.signd=1;
  } else {
    if(p_type(c,&ty,&sidx))return;                    /* scalar / pointer / typedef-of-typedef */
  }
  int bdims[3], bnd=td_dims_of(c,bdims);   /* a typedef'd array base */
  if(is(c,"(")){                                       /* typedef RET (*NAME)(PARAMS); -- a funcptr */
    int save=c->i; c->i++;
    if(is(c,"*") && tat(c,c->i+1)->k==T_ID && tok_is(tat(c,c->i+2),")")){
      c->i++; tok nm=adv(c); eat(c,")"); eat(c,"(");
      const bcir_ctype *ps; int np, va;                  /* the parameter-type list: its types ride the typedef too */
      if(fp_param_list(c,0,&ps,&np,&va)) return;
      eat(c,")");
      bcir_ctype fp; memset(&fp,0,sizeof fp); fp.kind=3; fp.size=cc_abi(c)->pointer_size; fp.signd=0; idcpy(c,fp.tag,&nm);
      fp_capture_ret(&fp,&ty,sidx);                      /* its RETURN type rides the typedef to every use */
      fp.fp_sig=sig_addv(c,&ty,ps,np,fp.tag,"",va);     /* spelled by the source's own alias */
      CC_ENSURE(c,c->td,c->ntd,c->cap_td);
      if(c->ntd<c->cap_td){memset(&c->td[c->ntd],0,sizeof c->td[c->ntd]);
        idcpy(c,c->td[c->ntd].name,&nm);c->td[c->ntd].ty=fp;c->td[c->ntd].sidx=-1;c->ntd++;}
      eat(c,";");
      return;
    }
    c->i=save;                                         /* not a funcptr declarator */
    if(fp_decl_at(c,0)){                               /* `typedef R (*NAME[N])(P);`, `(**NAME)(P)`: a table of function
                                                        * pointers, a pointer to one -- its element spelled by a synthesized
                                                        * alias, as an inline declarator's is (CF-RTFP) */
      bcir_ctype fp; int stars, dims[3], nd; tok nm;
      if(fp_inline_decl(c,&ty,sidx,&fp,&stars,dims,&nd,&nm)) return;
      for(int k=0;k<stars;k++) fp_star(c,&fp);
      if(bnd){ fail(c,"an array typedef of more than 3 dimensions"); return; }   /* a typedef'd array's function pointers */
      CC_ENSURE(c,c->td,c->ntd,c->cap_td);
      if(c->ntd<c->cap_td){memset(&c->td[c->ntd],0,sizeof c->td[c->ntd]);
        idcpy(c,c->td[c->ntd].name,&nm);c->td[c->ntd].ty=fp;c->td[c->ntd].sidx=-1;
        c->td[c->ntd].nd=nd; for(int d=0;d<3;d++) c->td[c->ntd].dims[d]=d<nd?dims[d]:0; c->ntd++;}
      eat(c,";");
      return;
    }
  }
  tok nm=adv(c);                                      /* the alias name */
  int tnd=0, tdims[3]={0,0,0};                        /* `typedef T row_t[6];`: its own dims, then its base's */
  while(is(c,"[") && !c->failed){ c->i++; long long d=ce_dim(c,0); if(d<0 || !eat(c,"]")) return;
    if(d==0){ fail(c,"an array typedef needs a positive constant size"); return; }
    if(tnd>=3){ fail(c,"an array typedef of more than 3 dimensions"); return; }
    tdims[tnd++]=(int)d; }
  if(ty.kind==1 && sidx>=0 && !tnd && !bnd && is_anon_tag(c->s[sidx].tag) && !c->s[sidx].tdname[0] && c->s[sidx].parent<0)
    idcpy(c,c->s[sidx].tdname,&nm);                   /* `typedef struct {...} P;`: C names the aggregate `P` */
  for(int d=0; d<bnd; d++){ if(tnd>=3){ fail(c,"an array typedef of more than 3 dimensions"); return; }
    tdims[tnd++]=bdims[d]; }
  CC_ENSURE(c,c->td,c->ntd,c->cap_td);
  if(c->ntd<c->cap_td){memset(&c->td[c->ntd],0,sizeof c->td[c->ntd]);
    idcpy(c,c->td[c->ntd].name,&nm);c->td[c->ntd].ty=ty;c->td[c->ntd].sidx=sidx;
    c->td[c->ntd].nd=tnd; for(int d=0;d<3;d++) c->td[c->ntd].dims[d]=tdims[d]; c->ntd++;}
  eat(c,";");
}

/* --- the IR builder ------------------------------------------------------ */
static uint32_t add_res(CC *c, bcir_domain dom, int elem, int count, int vol, int kind, const char *nm) {
  bcir_func *f=c->fn;
  if(!cc_ensure_size(c,(void **)&f->res,f->n_res,&f->cap_res,sizeof *f->res,16u)) return 0;
  bcir_resource *r=&f->res[f->n_res++]; memset(r,0,sizeof *r);   /* realloc slots are uninitialized */
  r->rid=c->rid++; r->domain=dom; r->elem_bytes=elem<1?1:elem; r->count=count<1?1:count;
  r->is_volatile=(uint8_t)vol; r->read_only=0; r->kind=(uint8_t)kind; r->agg[0]=0;
  if(!fits(c,r->name,sizeof r->name,"%s",nm?nm:"")) r->name[0]=0;
  return r->rid;
}
static bcir_claim *new_claim(CC *c,const char *op,bcir_opcode opc) {
  bcir_func *f=c->fn;
  if(!cc_ensure_size(c,(void **)&f->claims,f->n_claims,&f->cap_claims,
                     sizeof *f->claims,32u)) return NULL;
  bcir_claim *cl=&f->claims[f->n_claims++]; memset(cl,0,sizeof *cl);
  cl->id=c->cid++;cl->opcode=opc;cl->lane=BCIR_LANE_U;cl->stride=BCIR_STRIDE_SCALAR;cl->count=1;
  cl->domain=BCIR_DOM_RAM;cl->hazard=BCIR_HZ_UNIQUE;cl->bounds=BCIR_BND_STRICT;
  if(!fits(c,cl->op,sizeof cl->op,"%s",op)) cl->op[0]=0;
  return cl;
}
/* The casts the emit puts on a call's operands and result (CF-QUALS). A parameter of the callee's function type that
 * keeps a qualifier more than one level below its top -- `const char **`, `const char *const *`, to which C converts no
 * `char **` (C11 6.5.16.1p1) -- takes its argument through a cast to the parameter's type, the emit spelling its own
 * objects without qualifiers; one level down C adds them itself (`const T *` from a `T *`). A return that keeps one
 * below its top level, `const T *`, is cast to the emit's unqualified temp. A function pointer's type is spelled whole
 * wherever it is, so it takes none. The oracle's `_qual_args` / `_qual_result`. */
static int qual_arg_cast(const bcir_ctype *t){
  if(t->kind!=2) return 0;
  int d=ptr_levels(t); if(d<2) return 0;
  if(t->is_const) return 1;                         /* what the pointers point to */
  int lv=d-2>8 ? 8 : d-2;                           /* each `*` under the outermost two */
  unsigned below=lv>0 ? (1u<<lv)-1u : 0u;
  return ((unsigned)(t->ptr_const|t->ptr_restrict)&below)!=0;
}
static void qcast_add(CC *c, bcir_claim *cl, int operand, const char *type){
  bcir_func *f=c->fn;
  if(!CC_ENSURE(c,f->qcasts,f->n_qcasts,f->cap_qcasts)) return;
  bcir_qcast *q=&f->qcasts[f->n_qcasts]; memset(q,0,sizeof *q);
  q->claim_id=cl->id; q->operand=operand;
  if(!fits(c,q->type,sizeof q->type,"%s",type)) return;
  if(!cl->qcast) cl->qcast=(uint32_t)f->n_qcasts+1u;  /* its casts follow one another from here */
  f->n_qcasts++;
}
/* The casts of the call `cl` just lowered: of each of its `na` arguments, at rd[first..], to the parameter of `params`
 * it is passed to, and of its result when `ret`, the callee's return, keeps a qualifier the temp's type lacks. */
static void qcast_call(CC *c, bcir_claim *cl, int first, int na, const bcir_ctype *params, int np,
                       const bcir_ctype *ret){
  if(!cl) return;
  for(int k=0;k<na && k<np && params;k++) if(qual_arg_cast(&params[k])){
    char ty[BCIR_EMIT_TYPE]; ctype_qstr(&params[k],ty,sizeof ty);
    qcast_add(c,cl,first+k,ty); }
  if(ret && cl->n_wr && qual_key(ret)) qcast_add(c,cl,-1,"");
}
/* ... of a call through a pointer to the function type `sig` (1 + its index; 0 unknown), its arguments at rd[1..]. */
static void qcast_sig_call(CC *c, bcir_claim *cl, int sig, int na){
  if(!cl || sig<=0 || sig>c->nsig) return;
  bcir_ctype rt=c->sigs[sig-1].ret;                 /* a copy: the table may grow */
  qcast_call(c,cl,1,na,c->sigs[sig-1].params,c->sigs[sig-1].n_params,&rt);
}
/* The cast the emit puts on operand `k` of the call `cl` -- its result for -1 -- or NULL for none (CF-QUALS). */
static const char *qcast_of(const bcir_func *f, const bcir_claim *cl, int k){
  if(!cl->qcast || cl->qcast>(uint32_t)f->n_qcasts) return NULL;
  for(int i=(int)cl->qcast-1;i<f->n_qcasts && f->qcasts[i].claim_id==cl->id;i++)
    if(f->qcasts[i].operand==k) return f->qcasts[i].type;
  return NULL;
}
/* `(T)`, the cast on operand `k` of the call `cl` -- on its result for -1, to `own`, its temp's own type -- or "". */
static const char *qcast_text(const bcir_func *f, const bcir_claim *cl, int k, const char *own, char *b, size_t n){
  const char *t=qcast_of(f,cl,k);
  if(!t) return "";
  snprintf(b,n,"(%s)",k<0 ? own : t);
  return b;
}
static uint32_t temp(CC *c,int size){return add_res(c,BCIR_DOM_RAM,size?size:4,1,0,BCIR_RK_SCALAR,"");}
/* The value of a void expression: a placeholder temp no claim writes, marked so a `?:` whose arms are void has no
 * value (the oracle's `_VOID_RID`). */
static uint32_t void_temp(CC *c){
  uint32_t t=temp(c,4);
  if(c->fn->n_res && c->fn->res[c->fn->n_res-1].rid==t){ c->fn->res[c->fn->n_res-1].is_void=1; c->n_void++; }
  return t;
}
/* An integer temporary of a given (width, signedness) -- the emit renders the true fixed-width type
 * and the usual arithmetic conversions read it back (the C twin of int_type). */
static uint32_t tempi(CC *c,int size,int signd){ uint32_t r=add_res(c,BCIR_DOM_RAM,size?size:4,1,0,BCIR_RK_SCALAR,"");
  if(c->fn->n_res) c->fn->res[c->fn->n_res-1].is_signed=(uint8_t)(signd?1:0); return r; }
/* A floating temporary (size 4 float / 8 double) -- the emit renders it as float/double, not uint32. */
static uint32_t tempf(CC *c,int size){ uint32_t r=add_res(c,BCIR_DOM_RAM,size,1,0,BCIR_RK_SCALAR,"");
  if(c->fn->n_res) c->fn->res[c->fn->n_res-1].is_float=1; return r; }
/* a `_Complex` temp (size = the full 2x-element bytes): is_float AND is_complex, so the emit spells
 * `<elem> _Complex` (elem width = size/2) and the value is delegated to the backend like any float. */
static uint32_t tempc(CC *c,int size){ uint32_t r=add_res(c,BCIR_DOM_RAM,size,1,0,BCIR_RK_SCALAR,"");
  if(c->fn->n_res){ c->fn->res[c->fn->n_res-1].is_float=1; c->fn->res[c->fn->n_res-1].is_complex=1; } return r; }
/* A struct/union VALUE temp of the sdef `si` (CF-STRUCTVAL): an element of an array of structs, a struct member,
 * `*p`, a select of two structs -- the emit declares it as the aggregate (`tty` spells its `agg`), so it copies,
 * returns, passes and is member-accessed whole. An integer temp of it emitted `int32_t t = ps[i];`, which does not
 * compile, and hid the struct from a brace list, `typeof` and `_Generic`. The oracle's `_temp` of the aggregate. */
static uint32_t tempagg(CC *c,int si){ uint32_t r=add_res(c,BCIR_DOM_RAM,c->s[si].size,1,0,BCIR_RK_AGGREGATE,"");
  if(c->fn->n_res) fits(c,c->fn->res[c->fn->n_res-1].agg,BCIR_CIR_AGG,"%s %s",c->s[si].is_union?"union":"struct",
                            c->s[si].tag);
  return r; }
/* A C23 `_BitInt(N)` temp -- carries the EXACT width N (bit_width) and signedness, so the emit spells
 * `_BitInt(N)` / `unsigned _BitInt(N)` (NO power-of-two canonicalization) and the value is delegated to
 * the backend at the N-bit precision. `size` is the storage slot (1/2/4/8 bytes); is_signed drives the
 * spelling. The C twin of the oracle's `bitint` CType + a `_BitInt` temp. */
static uint32_t tempbi(CC *c,int bit_width,int signd){
  int sz=(bit_width<=8)?1:(bit_width<=16)?2:(bit_width<=32)?4:8;
  uint32_t r=add_res(c,BCIR_DOM_RAM,sz,1,0,BCIR_RK_SCALAR,"");
  if(c->fn->n_res){ c->fn->res[c->fn->n_res-1].is_signed=(uint8_t)(signd?1:0);
    c->fn->res[c->fn->n_res-1].bit_width=bit_width; } return r; }
/* The `_BitInt(N)` width of the value in rid (its exact N), or 0 if rid is not a `_BitInt` scalar. Drives
 * the non-promoting same-type arithmetic + the fallback-on-mix boundary in `binop_result`. */
static int rid_bitint(CC *c,uint32_t rid,int *signd){
  for(size_t i=0;i<c->fn->n_res;i++) if(c->fn->res[i].rid==rid){
    if(c->fn->res[i].bit_width>0){ if(signd)*signd=c->fn->res[i].is_signed; return c->fn->res[i].bit_width; }
    return 0; }
  return 0; }
/* The temp a call returning the function pointer `rt` (kind 3) yields (CF-FPRET): a function-pointer value of its type,
 * spelled by its alias -- the oracle's `_call_result_ct`. A `uint32_t` of it did not compile, and a comparison read a
 * pointer cut to 32 bits. */
static uint32_t fp_ret_temp(CC *c, const bcir_ctype *rt){
  return fp_value_temp(c, rt->fp_sig ? sig_alias(c,rt->fp_sig) : rt->tag);
}
/* The result-temp for a call THROUGH a funcptr (c.call.indirect / c.call.imember), by the funcptr's
 * captured RETURN type -- the C twin of the oracle's _call_result_ct ladder: a float keeps its width;
 * a wide (>4-byte) integer keeps its (width, sign); a SIGNED sub-int return promotes to `int` (so a
 * downstream `>>` / compare / `(long)` widen sign-extends); else uint32. A funcptr whose return type
 * wasn't captured (carriers all 0) -> uint32, today's behaviour. A struct or union return is that aggregate
 * value, as the oracle keeps it (CF-STRUCTVAL): a uint32 of it did not compile. */
static uint32_t fp_result_temp(CC *c, const bcir_ctype *fp){
  if(fp->fp_sig>0 && fp->fp_sig<=c->nsig){            /* the function type is known: a pointer or a function-pointer
                                                       * return is that pointer, never an integer of its width (CF-FPRET) */
    const bcir_ctype *rt=&c->sigs[fp->fp_sig-1].ret;
    if(rt->kind==3) return fp_ret_temp(c,rt);
    if(rt->kind==2) return temp_ptr(c,rt,rt->ptr_to_struct?find_struct(c,rt->tag,(int)strlen(rt->tag)):-1); }
  if(fp->fp_ret_agg) return tempagg(c, fp->fp_ret_agg-1);
  if(fp->fp_ret_float) return tempf(c, fp->fp_ret_size?fp->fp_ret_size:4);
  if(fp->fp_ret_size>4) return tempi(c, fp->fp_ret_size, fp->fp_ret_signd);
  if(fp->fp_ret_signd && fp->fp_ret_size && fp->fp_ret_size<=4) return tempi(c,4,1);
  return temp(c,4);
}
/* The result-temp of a call through the function-pointer member `ff` (`o->fn(x)`, a `c.call.imember`), as through any
 * function pointer of its type (`fp_result_temp`) -- a pointer or function-pointer return that pointer (CF-FPRET). 0 for
 * a void function, which has no result (CF-VOIDCB). */
static uint32_t call_result(CC *c, uint32_t r, const bcir_ctype *rt);   /* fwd: a postfix on a call's value */
/* The postfix on the value `t` a call through the function-pointer member `ff` returned (CF-FPRET): `o.mk(x).a`,
 * `o->get(p)->v`, `o.mk(x)(y)` -- as on any call's value, by the member's function type (`call_result`). */
static uint32_t member_call_result(CC *c, const field *ff, uint32_t t){
  if(ff->fp_sig<=0 || ff->fp_sig>c->nsig) return t;
  bcir_ctype rt=c->sigs[ff->fp_sig-1].ret;          /* a copy: what follows may grow the table */
  return call_result(c,t,&rt);
}
static uint32_t field_call_temp(CC *c, const field *ff){
  if(ff->fp_ret_void) return 0;
  bcir_ctype fp; memset(&fp,0,sizeof fp); fp.kind=3; fp.fp_sig=(uint16_t)ff->fp_sig;
  fp.fp_ret_agg=ff->fp_ret_agg; fp.fp_ret_float=(uint8_t)ff->fp_ret_float; fp.fp_ret_size=ff->fp_ret_size;
  fp.fp_ret_signd=(uint8_t)ff->fp_ret_signd;
  return fp_result_temp(c,&fp);
}
static int rid_complex(CC *c,uint32_t rid){
  for(size_t i=0;i<c->fn->n_res;i++) if(c->fn->res[i].rid==rid) return c->fn->res[i].is_complex; return 0; }
/* The integer (width, signedness) of the value in rid; returns 0 if rid is not a plain integer scalar
 * (float / pointer / aggregate -- the usual arithmetic conversions do not apply to it). */
static int rid_int(CC *c,uint32_t rid,int *size,int *signd){
  for(size_t i=0;i<c->fn->n_res;i++) if(c->fn->res[i].rid==rid){
    const bcir_resource *r=&c->fn->res[i];
    if(r->is_float||r->kind!=BCIR_RK_SCALAR) return 0;
    *size=(int)r->elem_bytes; *signd=r->is_signed; return 1; }
  *size=4; *signd=1; return 1;                          /* unknown -> assume int */
}
/* if rid holds a float/double, returns 1 and its width (else 0) -- a float arm of a select / the usual
 * arithmetic conversions makes the result the wider float. */
static int rid_float(CC *c,uint32_t rid,int *size){
  for(size_t i=0;i<c->fn->n_res;i++) if(c->fn->res[i].rid==rid){
    if(!c->fn->res[i].is_float) return 0; *size=(int)c->fn->res[i].elem_bytes; return 1; }
  return 0;
}
/* integer promotion (§6.3.1.1): a sub-int rank promotes to int. */
static void promote_i(int *size,int *signd){ if(*size<4){*size=4;*signd=1;} }
/* usual arithmetic conversions (§6.3.1.8) in the (width, signedness) value model. */
static void uac_i(int sa,int za,int sb,int zb,int *rs,int *rz){
  promote_i(&sa,&za); promote_i(&sb,&zb);
  if(sa!=sb){ if(sa>sb){*rs=sa;*rz=za;}else{*rs=sb;*rz=zb;} } else { *rs=sa; *rz=za&&zb; }
}
/* The integer type (width, signedness) of an integer constant from its source text (suffix + magnitude,
 * §6.4.4.1), mirroring ctype_model.int_literal_type -- a `long` candidate `lsz` bytes wide, the target's (the
 * oracle's `_lit_type`: on LLP64 / ILP32 a value past a 32-bit `long` is a `long long`). A character constant
 * ('c') has type int. */
static void lit_int_type(const char *s,int n,int lsz,int *size,int *signd){
  *size=4; *signd=1;                                    /* default: int */
  if(n>0 && s[0]=='\'') return;                         /* a character constant is int */
  intlit L; if(int_literal(s,n,&L)) return;            /* (a refused one never reaches here: the lexer refuses it) */
  unsigned long long val=L.v; int u=L.u, lr=L.lr, decimal=L.decimal;   /* the exact value: a 63-character copy cut a
                                                        * long one (`0b` and 64 digits), and with it its type */
  int cs[6],cz[6],nc=0;                                 /* candidate (size, signed) list, in order */
  if(u){ if(lr==0){cs[nc]=4;cz[nc++]=0;} if(lr<2){cs[nc]=lsz;cz[nc++]=0;} cs[nc]=8;cz[nc++]=0; }
  else if(decimal){ if(lr==0){cs[nc]=4;cz[nc++]=1;} if(lr<2){cs[nc]=lsz;cz[nc++]=1;} cs[nc]=8;cz[nc++]=1; }
  else { if(lr==0){cs[nc]=4;cz[nc++]=1; cs[nc]=4;cz[nc++]=0;} if(lr<2){cs[nc]=lsz;cz[nc++]=1; cs[nc]=lsz;cz[nc++]=0;}
         cs[nc]=8;cz[nc++]=1; cs[nc]=8;cz[nc++]=0; }
  for(int i=0;i<nc;i++){
    unsigned long long hi = cz[i] ? ((1ull<<(cs[i]*8-1))-1) : (cs[i]>=8?~0ull:((1ull<<(cs[i]*8))-1));
    if(val<=hi){ *size=cs[i]; *signd=cz[i]; return; }
  }
  *size=cs[nc-1]; *signd=cz[nc-1];
}
/* record an R18 call-graph edge (callee name), growing the per-function call list on demand. */
static void add_call(CC *c, const tok *name) {
  bcir_func *f=c->fn;
  if(!CC_ENSURE(c,f->calls,f->n_calls,f->cap_calls)) return;
  idcpy(c,f->calls[f->n_calls++],name);
}
/* The floating size of the value in rid (4 float / 8 double), or 0 if it is not floating. Drives
 * float result typing (usual arithmetic conversions: the wider float wins) + the emit. */
static int rid_fsize(CC *c,uint32_t rid){
  for(size_t i=0;i<c->fn->n_res;i++)
    if(c->fn->res[i].rid==rid) return (c->fn->res[i].is_float&&c->fn->res[i].kind==BCIR_RK_SCALAR)?(int)c->fn->res[i].elem_bytes:0;
  return 0;
}
/* A string literal -> an anonymous read-only char[] global (decays to a pointer). Identical spellings
 * in the same function share one resource (dedup). The full spelling is kept in result-owned metadata so emit can
 * inline it at any length; the resource name is just a short tag. */
static uint32_t intern_string(CC *c, const char *s, int n) {
  size_t copy_size;
  char *cp;
  for (int k=0;k<c->fn->n_host_literals;k++)                   /* dedup within the current function */
    if ((int)strlen(c->fn->host_literals[k].spelling)==n &&
        !memcmp(c->fn->host_literals[k].spelling,s,(size_t)n))
      return c->fn->host_literals[k].rid;
  int elem = str_elem_size(c,s,n);                            /* element width (wide/UTF prefix) */
  int nunits = str_bytes(s,n)+1;                              /* code units incl. the NUL */
  char nm[BCIR_CIR_NAME]; snprintf(nm,sizeof nm,"__str%d",c->fn->n_host_literals);
  uint32_t rid = add_res(c,BCIR_DOM_RAM,elem,nunits,0,BCIR_RK_POINTER,nm);
  if (c->fn->n_res) c->fn->res[c->fn->n_res-1].read_only=1;   /* a read-only global */
  if(c->failed || n<0 || !bcir_size_add((size_t)n,1u,&copy_size) ||
     !CC_ENSURE(c,c->fn->host_literals,c->fn->n_host_literals,
                c->fn->cap_host_literals)) return rid;
  cp=(char *)bcir_host_allocate(&c->allocator,copy_size);
  if(!cp){cc_raise_oom(c);return rid;}
  memcpy(cp,s,(size_t)n); cp[n]=0;
  c->fn->host_literals[c->fn->n_host_literals].rid=rid;
  c->fn->host_literals[c->fn->n_host_literals].spelling=cp;
  c->fn->n_host_literals++;
  return rid;
}
static venv *lookup(CC *c,const tok *t){
  for(int i=c->nenv-1;i>=0;i--) if((int)strlen(c->env[i].name)==t->n&&!strncmp(c->env[i].name,t->s,t->n))
    return &c->env[i];
  return NULL;
}

/* --- expression lowering (returns rid) ----------------------------------- */
static uint32_t p_expr(CC *c);
static uint32_t p_compound_literal(CC *c, const bcir_ctype *ty, int si);   /* `(type){init}` (defined w/ agg_init) */
static uint32_t p_stmt_expr(CC *c);   /* `({ ... })` -- a GCC statement expression (defined after p_stmt) */
static uint32_t p_array_literal(CC *c, const bcir_ctype *ty, int si, int count, const int *la_dims, int la_nd);    /* `(T[N]){init}` / `(T[A][B]){init}` / `(struct P[]){init}` (defined w/ arr_init) */
static const bcir_resource *res_of(const bcir_func *f,uint32_t rid);   /* (defined with the verifier) */
/* A copy of `rid`'s resource, for a caller about to make a temporary: add_res may reallocate res[], so a
 * pointer into it dangles across the call (CF-UAF). 1 when `rid` names a resource, else 0 and *out zeroed. */
static int res_copy(const CC *c,uint32_t rid,bcir_resource *out){
  const bcir_resource *r=res_of(c->fn,rid);
  if(r) *out=*r; else memset(out,0,sizeof *out);
  return r!=NULL;
}

/* --- volatile: one model on both rails (the oracle: ctype_model.touches_mmio, lower._load_unit/_write,
 * lower._order_device_claims) -------------------------------------------------------------------------
 * R3 holds the base of a device access to a device region, so a resource is MMIO exactly when its declared
 * type holds volatile storage or points (through any number of pointers) at some. An access is a VOLATILE
 * access exactly when the lvalue it reads or writes is volatile; it is then device-domain, lane H,
 * `barriered`, with the claim's volatile bit. Any other claim that touches a device region -- the copy that
 * binds a pointer to volatile storage, the load of a pointer member that points at some -- is made
 * device-domain and ordered by order_device_claims, without the bit. */
static int sdef_vol(const CC *c,int si){ return si>=0 && si<c->ns && c->s[si].vol_storage; }
/* An object of type `ty` (an array of `ty` when `is_arr`; `si` its struct) is a device region. An array of
 * pointers holds addresses: plain storage. */
static int ty_mmio(const CC *c,const bcir_ctype *ty,int si,int is_arr){
  if(ty->kind==3) return 0;
  if(ty->kind==2) return !is_arr && (ty->is_volatile || (ty->ptr_to_struct && sdef_vol(c,si)));
  if(ty->kind==1) return ty->is_volatile || sdef_vol(c,si);
  return ty->is_volatile;
}
/* Mark an access claim: a volatile access (`vol`), or one through a device region (its base is MMIO), is
 * device-domain and ordered; only the former carries the volatile bit (the emit performs exactly it). */
static void mark_access(CC *c,bcir_claim *cl,int vol){
  if(!cl) return;
  const bcir_resource *br = cl->n_rd ? res_of(c->fn,cl->rd[0]) : NULL;
  if(vol || (br && br->domain==BCIR_DOM_MMIO)){ cl->domain=BCIR_DOM_MMIO; cl->lane=BCIR_LANE_H; cl->hazard=BCIR_HZ_BARRIERED; }
  cl->is_volatile=(uint8_t)(vol?1:0);
}
/* An access to `_Atomic` storage is an atomic operation (CF-ATOMIC): lane A and the `atomic` hazard -- lane H
 * when `mark_access` made it a device access -- the oracle's `_access_order`. The emit performs it through an
 * `_Atomic` lvalue, never a byte copy. Called after `mark_access`. */
static void mark_atomic(bcir_claim *cl,int at){
  if(!cl || !at) return;
  cl->hazard=BCIR_HZ_ATOMIC; cl->lane=(uint8_t)(cl->domain==BCIR_DOM_MMIO?BCIR_LANE_H:BCIR_LANE_A);
}
/* R3 over a finished function, one pass: a claim that touches a device region is device-domain and ordered
 * (lane H, `barriered` unless already `atomic`). Its volatile bit is left as the lowering set it. */
static void order_device_claims(bcir_func *f){
  for(size_t i=0;i<f->n_claims;i++){ bcir_claim *cl=&f->claims[i];
    if(cl->domain==BCIR_DOM_MMIO) continue;
    int dev=0;
    for(int k=0;k<cl->n_rd && !dev;k++){ const bcir_resource *r=res_of(f,cl->rd[k]); dev = r && r->domain==BCIR_DOM_MMIO; }
    for(int k=0;k<cl->n_wr && !dev;k++){ const bcir_resource *r=res_of(f,cl->wr[k]); dev = r && r->domain==BCIR_DOM_MMIO; }
    if(dev){ cl->domain=BCIR_DOM_MMIO; cl->lane=BCIR_LANE_H; if(cl->hazard==BCIR_HZ_UNIQUE) cl->hazard=BCIR_HZ_BARRIERED; }
  }
}
/* A named object that is itself volatile (a scalar or a struct; an array decays, a pointer's `volatile`
 * qualifies its pointee): a read or a write of it BY NAME is a volatile access. */
static int obj_vol(CC *c,const venv *v){
  if(!v->type.is_volatile || v->type.kind!=0) return 0;   /* a scalar (a struct copies whole, plain) */
  const bcir_resource *r=res_of(c->fn,v->rid);
  return !(r && r->is_array);
}
/* A resource declared as an ARRAY -- of any length: a one-element `T a[1]` is an array too (its declaration
 * and its subscripts need the brackets, and its subscript a guard), which `count > 1` alone missed. The one
 * predicate the declaration, the bounds class and the guard ask. */
static int decl_array(const bcir_resource *r){ return r->count>1 || r->is_array; }
/* An ARRAY whose elements are pointers -- `T *a[N]` local, static, file-scope (the twin tags a file-scope array
 * POINTER for decay) or a VLA -- so `a[i]` loads a pointer, never a pointee. The one predicate the element's type,
 * its volatility and a subscript chain all ask (a one-element or file-scope array was missed by `count>1`). */
static int ptr_array(CC *c,const venv *b){
  const bcir_resource *br=res_of(c->fn,b->rid);
  return b->type.kind==2 && br && (br->is_array || br->is_vla);
}
/* The element of `b[i]` is volatile: the pointee of a pointer to volatile (a `T **` element is a pointer,
 * an array of pointers holds pointers), or an element of a volatile array. */
static int index_elem_vol(CC *c,const venv *b){
  if(!b->type.is_volatile) return 0;
  if(b->type.kind==2){
    if(ptr_array(c,b)) return 0;                                      /* an array of pointers */
    return (b->type.ptr_depth?b->type.ptr_depth:1)==1;
  }
  return 1;
}
/* The same for `_Atomic` (CF-ATOMIC): `b`'s element or pointee is `_Atomic` storage -- a venv's `is_atomic`
 * qualifies what a pointer points at or an array holds -- so an access to it is an atomic one. */
static int index_elem_atomic(CC *c,const venv *b){
  if(!b->type.is_atomic) return 0;
  if(b->type.kind==2){
    if(ptr_array(c,b)) return 0;                                      /* an array of pointers */
    return (b->type.ptr_depth?b->type.ptr_depth:1)==1;
  }
  return 1;
}
/* The sdef of the struct/union the element of `b[i]` (and `*b`) is -- of an array of structs, or through a pointer to
 * one -- else -1 (a scalar or pointer element): the aggregate its load reads and a store to it takes (CF-STRUCTVAL). */
static int index_elem_sidx(CC *c,const venv *b){
  if(b->type.kind==1) return b->sidx;
  if(b->type.kind==2 && b->type.ptr_to_struct && !ptr_array(c,b) && (b->type.ptr_depth?b->type.ptr_depth:1)==1)
    return b->sidx;
  return -1;
}
/* The sdef an aggregate spelling `struct T` / `union T` (a resource's `agg`) names, else -1. */
static int agg_sidx(CC *c,const char *agg){
  if(!agg[0]) return -1;
  const char *sp=strchr(agg,' '), *tg=sp?sp+1:agg;
  return find_struct(c,tg,(int)strlen(tg));
}
/* The value's type names the struct/union `sidx` (so it initializes that subobject whole, 6.7.9p13). */
static int value_is_struct(CC *c, uint32_t v, int sidx){
  const bcir_resource *r=res_of(c->fn,v);
  char want[BCIR_CIR_AGG];
  if(!r || sidx<0 || !r->agg[0] || r->kind!=BCIR_RK_AGGREGATE) return 0;
  snprintf(want,sizeof want,"%s %s",c->s[sidx].is_union?"union":"struct",c->s[sidx].tag);
  return !strcmp(want,r->agg);
}
/* A struct or union object (of sdef `si`; -1: not one, nothing to check) takes only a value of its own type -- its
 * initializer, when not a brace list, is one expression of it (C11 6.7.9p13), and so is what `=` assigns it
 * (6.5.16.1p1). Anything else -- a string, a scalar, another struct -- is refused with `why` (CF-STRUCTINIT): it had
 * lowered to a copy the emit spells `x = 5;`, which does not compile. The oracle's `_struct_value`. */
static const char *const agg_initialized="a struct or union is initialized by a brace list or a value of its own type";
static const char *const agg_assigned="a struct or union is assigned a value of its own type";
/* ... and a struct or union is no operand of an operator that takes a scalar -- arithmetic, bitwise, shift,
 * relational, equality, logical, a compound assignment, an increment (C11 6.5.2.4p1, 6.5.3.1p1, 6.5.3.3p1,
 * 6.5.5-6.5.14, 6.5.16.2p1-2) -- and converts to no scalar type: an initializer, an assignment, a `return`, an
 * argument, a cast (6.5.16.1p1, 6.5.4p2). Both rails had lowered `a += 5`, `a++` and `uint32_t k = a;` to an emit
 * Clang rejects (CF-STRUCTARITH; the oracle's `_STRUCT_OPERAND`, `_STRUCT_CONVERTED`). */
static const char *const agg_operator="a struct or union is an operand of an arithmetic, bitwise, logical or comparison operator";
static const char *const agg_converted="a struct or union is converted to a scalar type";
/* `v` is a struct or union value -- not an array of them, which decays to a pointer. */
static int agg_rvalue(CC *c, uint32_t v){
  const bcir_resource *r=res_of(c->fn,v);
  return r && r->kind==BCIR_RK_AGGREGATE && !r->is_array && !r->is_vla;
}
/* `v`, an operand of an operator that takes a scalar, is a struct or union: refused, and 1. */
static int agg_operand(CC *c, uint32_t v){
  if(!agg_rvalue(c,v)) return 0;
  fail(c,agg_operator); return 1;
}
/* `v`, taken by an object of a scalar or pointer type, is a struct or union: refused, and 0; else 1. */
static int scalar_value_ok(CC *c, uint32_t v){
  if(!agg_rvalue(c,v)) return 1;
  fail(c,agg_converted); return 0;
}
static int agg_value_ok(CC *c,int si,uint32_t v,const char *why){
  if(si<0 && !is(c,".") && !is(c,"->") && !is(c,"[")) return scalar_value_ok(c,v);   /* a scalar or pointer object
                                                    * takes no struct (CF-STRUCTARITH) */
  if(si<0 || value_is_struct(c,v,si)) return 1;
  if(is(c,".")||is(c,"->")||is(c,"[")) return 1;   /* the value stopped short of a postfix the twin does not apply
                                                    * (`(*p).m`): the statement's own parse error refuses it */
  fail(c,why); return 0;
}
/* The sdef of the struct/union OBJECT `v` names -- not an array of them -- else -1: the type `v = e` assigns. */
static int obj_sidx(CC *c,const venv *v){
  const bcir_resource *r=res_of(c->fn,v->rid);
  return v->type.kind==1 && !(r && (r->is_array || r->is_vla)) ? v->sidx : -1;
}
/* The function-pointer pointee of the pointer resource `r` of type `ty` (CF-FPTAB): its alias, so the emit spells
 * `op_t *p`, and the mark that a read through it is a function-pointer value -- never an integer of its width, whose
 * `uint64_t *` did not compile and was 8 bytes on a 4-byte-pointer target. */
static void ptee_fp(CC *c, bcir_resource *r, const bcir_ctype *ty){
  fits(c,r->agg,BCIR_CIR_AGG,"%s",ty->tag); r->ptee_funcptr=1;
}
/* The alias of the function type `sig` (1 + its index in `sigs`), given one on first use (CF-FPTAB): a member a struct
 * declares by an inline declarator `RET (*fn)(PARAMS)` has none -- the source declares the struct -- so a read of it as
 * a value gets a synthesized `typedef RET (*__bcir_fpN)(PARAMS);` to be declared by, as an inline local's has. "" for
 * no type. */
static const char *sig_alias(CC *c, int sig){
  if(sig<=0 || sig>c->nsig) return "";
  if(!c->sigs[sig-1].alias[0]){
    char al[BCIR_CIR_NAME], rets[BCIR_EMIT_TYPE];
    snprintf(al,sizeof al,"__bcir_fp%d",c->n_fpdef++);
    ctype_qstr(&c->sigs[sig-1].ret,rets,sizeof rets);   /* qualified as the type is (CF-QUALS) */
    ctext_putf(c,&c->fpdefs,"typedef %s (*%s)(",rets,al);
    for(int k=0;k<c->sigs[sig-1].n_params;k++){ char pt[BCIR_EMIT_TYPE]; ctype_qstr(&c->sigs[sig-1].params[k],pt,sizeof pt);
      ctext_putf(c,&c->fpdefs,"%s%s",k?", ":"",pt); }
    ctext_putf(c,&c->fpdefs,"%s);\n",c->sigs[sig-1].variadic?(c->sigs[sig-1].n_params?", ...":"..."):
                                       c->sigs[sig-1].n_params?"":"void");
    fits(c,c->sigs[sig-1].alias,sizeof c->sigs[sig-1].alias,"%s",al); }
  return c->sigs[sig-1].alias;
}
/* A pointer temp of pointer type `ty` (the oracle's `_temp(pointer)`): it carries the pointee -- width, sign,
 * float, plain char, _Bool, struct tag, void, depth and volatility -- and is an MMIO resource when it points
 * at volatile storage, so what is read through it is typed and marked as through a declared pointer. */
static uint32_t temp_ptr(CC *c,const bcir_ctype *ty,int si){
  uint32_t t=add_res(c,ty_mmio(c,ty,si,0)?BCIR_DOM_MMIO:BCIR_DOM_RAM,ty->size>0?ty->size:1,1<<16,ty->is_volatile,
                     BCIR_RK_POINTER,"");
  if(c->fn->n_res){ bcir_resource *pr=&c->fn->res[c->fn->n_res-1];
    pr->is_signed=(uint8_t)(ty->signd?1:0); pr->is_float=(uint8_t)(ty->is_float?1:0);
    pr->is_complex=(uint8_t)(ty->is_complex?1:0); pr->is_bool=(uint8_t)(ty->is_bool?1:0);
    pr->is_plain_char=(uint8_t)(ty->is_plain_char?1:0); pr->ptr_depth=ty->ptr_depth;
    pr->is_atomic=(uint8_t)(ty->is_atomic?1:0);          /* a pointer to `_Atomic` storage (CF-ATOMIC) */
    if(ty->ptr_to_struct) fits(c,pr->agg,BCIR_CIR_AGG,"%s %s",ty->is_union?"union":"struct",ty->tag);
    else if(ty->ptr_to_fp) ptee_fp(c,pr,ty);
    else if(ty->size==0 && !ty->is_float) pr->is_voidptr=1; }
  return t;
}
/* A pointer to an object of type `ty` -- `&v` of a named object, `&(T){...}` of a compound literal -- one level deeper
 * than it, carrying its type: its sign, float, complex, `_Bool`, plain `char` and `_Atomic`, a struct's tag, a function
 * pointer's alias (`&fp`, `&p` of `op_t *p`, CF-FPTAB). The compound literal's had kept one level and no pointee of a
 * pointer, so `&(T *){p}` was a `T *` (CF-RTFP). 0, refused, past BCIR_MAX_PTR_DEPTH. */
static uint32_t addr_temp(CC *c,const bcir_ctype *ty){
  if(ty->kind==2 && (ty->ptr_depth?ty->ptr_depth:1)>=BCIR_MAX_PTR_DEPTH){ fail(c,"pointer nesting too deep"); return 0; }
  uint32_t t=add_res(c,BCIR_DOM_RAM, ty->size?ty->size:4, 1,0,BCIR_RK_POINTER,"");
  if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
    tr->is_signed=(uint8_t)(ty->signd?1:0); tr->is_float=(uint8_t)(ty->is_float?1:0);
    tr->is_complex=(uint8_t)(ty->is_complex?1:0); tr->is_bool=(uint8_t)(ty->is_bool?1:0);
    tr->is_plain_char=(uint8_t)(ty->is_plain_char?1:0);
    tr->is_atomic=(uint8_t)(ty->is_atomic?1:0);         /* `&a` of an `_Atomic` object: `_Atomic T *` */
    tr->ptr_depth=(uint8_t)((ty->kind==2?(ty->ptr_depth?ty->ptr_depth:1):0)+1);
    if(ty->kind==1||ty->ptr_to_struct) fits(c,tr->agg,sizeof tr->agg,"%s %s",ty->is_union?"union":"struct",ty->tag);
    else if(ty->kind==3||ty->ptr_to_fp) ptee_fp(c,tr,ty); }
  return t;
}
/* A fresh ordinary temp of `rid`'s value type -- the value a volatile object's read yields (lvalue
 * conversion drops the qualifier). */
static uint32_t temp_like(CC *c,uint32_t rid){
  bcir_resource snap; res_copy(c,rid,&snap);
  uint32_t t=add_res(c,BCIR_DOM_RAM,(int)(snap.elem_bytes?snap.elem_bytes:4),1,0,
                     snap.kind==BCIR_RK_AGGREGATE?BCIR_RK_AGGREGATE:BCIR_RK_SCALAR,"");
  if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
    tr->is_float=snap.is_float; tr->is_complex=snap.is_complex; tr->is_signed=snap.is_signed;
    tr->is_bool=snap.is_bool; tr->is_plain_char=snap.is_plain_char; tr->bit_width=snap.bit_width;
    if(snap.kind==BCIR_RK_AGGREGATE) fits(c,tr->agg,sizeof tr->agg,"%s",snap.agg); }
  return t;
}
/* A read of the named object `v` as a value: its rid -- or, for a volatile object, a volatile read into an
 * ordinary temp (a c.copy carrying the volatile bit): the oracle's `_rvalue(Name)`. */
/* The value `v` taken by an object of type `ty`: a `c.const 0` taken by a pointer is a null pointer constant
 * (C11 6.3.2.3p3), so its temp becomes that pointer -- the emit declares `T *t = 0u;`, where an integer temp
 * made `p = t;` assign an integer variable to a pointer, which Clang and GCC reject. The claim graph is
 * unchanged: only the temp's C type moves (the oracle's `_null_pointer`, CF-NULLPTR). */
static uint32_t null_pointer(CC *c, uint32_t v, const bcir_ctype *ty){
  if(!c->fn || !ty || !(ty->kind==2 || (ty->kind==3 && ty->tag[0]))) return v;   /* a pointer, a function pointer */
  bcir_resource *r=NULL; for(size_t i=0;i<c->fn->n_res;i++) if(c->fn->res[i].rid==v){ r=&c->fn->res[i]; break; }
  if(!r || r->name[0] || r->kind!=BCIR_RK_SCALAR) return v;         /* a named object is never a constant */
  const bcir_claim *k=NULL;
  for(size_t i=c->fn->n_claims; i-- > 0; ) if(c->fn->claims[i].n_wr && c->fn->claims[i].wr[0]==v){ k=&c->fn->claims[i]; break; }
  if(!k || strcmp(k->op,"c.const") || k->n_imm!=1 || k->imm[0]!=0) return v;
  if(ty->kind==3){ r->is_funcptr=1; fits(c,r->agg,BCIR_CIR_AGG,"%s",ty->tag); return v; }   /* `op_t f = 0;` */
  r->kind=BCIR_RK_POINTER; r->elem_bytes=(uint32_t)(ty->size>0?ty->size:0); r->ptr_depth=ty->ptr_depth;
  r->is_signed=(uint8_t)(ty->signd?1:0); r->is_float=(uint8_t)(ty->is_float?1:0); r->is_complex=(uint8_t)(ty->is_complex?1:0);
  r->is_plain_char=(uint8_t)(ty->is_plain_char?1:0); r->is_bool=(uint8_t)(ty->is_bool?1:0);
  r->is_volatile=(uint8_t)(ty->is_volatile?1:0); r->is_atomic=(uint8_t)(ty->is_atomic?1:0);
  if(ty->ptr_to_struct) fits(c,r->agg,BCIR_CIR_AGG,"%s %s",ty->is_union?"union":"struct",ty->tag);
  else if(ty->ptr_to_fp) ptee_fp(c,r,ty);
  else if(ty->size==0 && !ty->is_float) r->is_voidptr=1;
  return v;
}
/* The actuals of every direct call in the unit, typed by the callee's parameters: an argument converts to its
 * parameter's type as if by assignment (C11 6.5.2.2p7), so a `c.const 0` passed to a pointer parameter is a
 * null pointer of that type (null_pointer). The parser is single pass -- a callee defined after its caller is
 * not known at the call -- so this runs once the unit is parsed, reading the callee's definition in the unit,
 * whichever comes first, else its prototype. A variadic callee's extra actuals have no parameter and keep their
 * type. Only the temps' C types move (the oracle's `_null_pointer_args`, CF-NULLARG). */
static void null_pointer_args(CC *c, bcir_unit *u){
  bcir_func *cur=c->fn;
  for(int i=0;i<u->n_funcs;i++){ c->fn=&u->funcs[i];
    for(size_t k=0;k<c->fn->n_claims;k++){ const bcir_claim *cl=&c->fn->claims[k]; const char *callee;
      if(!strncmp(cl->op,"c.call:",7)) callee=cl->op+7;
      else if(!strncmp(cl->op,"c.call.void:",12)) callee=cl->op+12;
      else if(!strncmp(cl->op,"c.call.tu:",10)) callee=cl->op+10;
      else continue;
      const bcir_func *def=NULL; const bcir_ctype *pt=NULL; int np=0;
      for(int j=0;j<u->n_funcs && !def;j++) if(!strcmp(u->funcs[j].name,callee)) def=&u->funcs[j];
      if(def) np=def->n_params;
      else for(int j=0;j<c->n_protos;j++)
        if(!strcmp(c->protos[j].name,callee)){ pt=c->protos[j].params; np=c->protos[j].n_params; break; }
      for(int a=0;a<cl->n_rd && a<np;a++){ const bcir_ctype *at=def?&def->params[a].type:pt?&pt[a]:NULL;
        if(at && at->kind!=1 && !at->is_valist && !scalar_value_ok(c,cl->rd[a])){ c->fn=cur; return; }   /* `g(a)` of a
                                                          * struct for a scalar parameter (CF-STRUCTARITH) */
        null_pointer(c,cl->rd[a],at); }
    }
  }
  c->fn=cur;
}
/* The actuals of a call through a function pointer whose function type is `sig` (1 + its index in `sigs`; 0: none
 * captured): an argument converts to its parameter's type as if by assignment (C11 6.5.2.2p7), through a pointer as
 * in a direct call, so a null pointer constant passed to a pointer parameter is that parameter's pointer (the
 * oracle's `_null_pointer_args` over the pointer's parameters, CF-NULLCALL). */
static void null_pointer_sig(CC *c, int sig, const uint32_t *args, int na){
  if(sig<=0 || sig>c->nsig) return;
  const fsig *s=&c->sigs[sig-1];
  for(int k=0;k<na && k<s->n_params;k++){
    if(s->params[k].kind!=1 && !s->params[k].is_valist && !scalar_value_ok(c,args[k])) return;   /* a struct argument
                                                          * for a scalar parameter (CF-STRUCTARITH) */
    null_pointer(c,args[k],&s->params[k]); }
}
/* `v`, when it is a `c.const 0` temp, takes the type of the pointer or function-pointer value `res` holds -- a
 * pointer temp, a select, a null pointer typed as one: its resource, less the name (the null pointer constant of
 * `?:` or `==` beside that value). */
static void null_as(CC *c, const bcir_resource *res, uint32_t v){
  bcir_resource tr=*res;                              /* a copy: `res` may sit in the array `v` is retyped in */
  if(tr.kind!=BCIR_RK_POINTER && !(tr.is_funcptr && tr.agg[0])) return;
  bcir_resource *r=NULL;
  for(size_t i=0;i<c->fn->n_res;i++) if(c->fn->res[i].rid==v){ r=&c->fn->res[i]; break; }
  if(!r || r->name[0] || r->kind!=BCIR_RK_SCALAR) return;
  const bcir_claim *k=NULL;
  for(size_t i=c->fn->n_claims; i-- > 0; ) if(c->fn->claims[i].n_wr && c->fn->claims[i].wr[0]==v){ k=&c->fn->claims[i]; break; }
  if(!k || strcmp(k->op,"c.const") || k->n_imm!=1 || k->imm[0]!=0) return;
  bcir_domain dom=r->domain; *r=tr; r->rid=v; r->name[0]=0; r->read_only=0; r->domain=dom;
}
/* The value `v` compared with `p` by `==` or `!=`: a null pointer constant compared with a pointer converts to that
 * pointer (C11 6.5.9p5), so its temp is typed as the pointer -- an `int` temp beside it was a pointer compared with an
 * integer, a constraint violation (6.5.9p2) Clang and GCC only warn about. A named pointer -- a local, a parameter, a
 * global -- by its declared type (an array is no pointer here), any other pointer value by its resource. Only the
 * temp's C type moves (the oracle's `_null_pointer` over both operands, CF-NULLCALL). */
static int designator_sig(CC *c, const char *name);   /* fwd: a designator's function type (CF-FNSEL) */
static void null_compared(CC *c, uint32_t v, uint32_t p){
  { const bcir_resource *pa=res_of(c->fn,p);   /* an array is the pointer it decays to: the 0 a `void *`, which any
                                                  * object pointer compares with (6.5.9p2), whatever the array's rank --
                                                  * an `int` temp was an emit no compiler took (CF-STRUCTCOND) */
    if(pa && (pa->is_array || pa->is_vla)){ bcir_ctype t; memset(&t,0,sizeof t); t.kind=2; t.ptr_depth=1;
      null_pointer(c,v,&t); return; } }
  for(int i=c->nenv;i-- > 0;) if(c->env[i].rid==p){
    const bcir_resource *pr=res_of(c->fn,p); bcir_ctype t=c->env[i].type;
    if((t.kind==2 || t.kind==3) && !(pr && (pr->is_array || pr->is_vla))) null_pointer(c,v,&t);
    return; }
  const bcir_resource *pr=res_of(c->fn,p);
  if(pr && pr->read_only && pr->is_funcptr && pr->name[0]){   /* `f != 0` of a designator: a pointer to its
                                                    * function type, as the oracle's `_fn_type` (CF-EXTDESIG) */
    char fnm[BCIR_CIR_NAME]; snprintf(fnm,sizeof fnm,"%s",pr->name);   /* `pr` may move as the table grows */
    int s=designator_sig(c,fnm);
    if(s>0 && s<=c->nsig && c->sigs[s-1].alias[0]){ bcir_ctype t; memset(&t,0,sizeof t); t.kind=3;
      fits(c,t.tag,sizeof t.tag,"%s",c->sigs[s-1].alias); null_pointer(c,v,&t); }
    return; }
  if(pr && !pr->name[0]) null_as(c,pr,v);
}
static uint32_t named_read(CC *c,const venv *v){
  if(!obj_vol(c,v)) return v->rid;
  uint32_t t=temp_like(c,v->rid);
  bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD);
  if(cl){ cl->n_rd=1; cl->rd[0]=v->rid; cl->n_wr=1; cl->wr[0]=t; mark_access(c,cl,1); }
  return t;
}
/* The pointer temp just created points at volatile storage (`vol`: its pointee is volatile, declared so) or
 * at an aggregate holding some (`sdev`): a device region, like the oracle's typed temp. */
static void last_ptr_to_volatile(CC *c,int vol,int sdev){
  if(!c->fn->n_res || !(vol||sdev)) return;
  bcir_resource *tr=&c->fn->res[c->fn->n_res-1]; tr->domain=BCIR_DOM_MMIO; if(vol) tr->is_volatile=1;
}
/* A write of the named object `v` (a c.copy into it): a volatile access when the object is volatile. */
static void mark_obj_write(CC *c,bcir_claim *cl,const venv *v){ if(cl && obj_vol(c,v)) mark_access(c,cl,1); }

/* --- atomic read-modify-write (CF-ATOMIC): a compound assignment or an increment/decrement of an `_Atomic`
 * object is ONE atomic operation, never a load, an operation and a store (a concurrent write between them was
 * lost). The oracle's `_atomic_rmw`. --- */
/* The opcode of an atomic read-modify-write by its operator: the three the opcode set names, else the
 * compare-and-swap every other one is (the oracle's `_RMW_OPCODE`). */
static bcir_opcode rmw_opcode(const char *kind){
  if(!strcmp(kind,"add")||!strcmp(kind,"preinc")||!strcmp(kind,"postinc")) return BCIR_OP_ATOMIC_ADD;
  if(!strcmp(kind,"sub")||!strcmp(kind,"predec")||!strcmp(kind,"postdec")) return BCIR_OP_ATOMIC_SUB;
  if(!strcmp(kind,"xor")) return BCIR_OP_ATOMIC_XOR;
  return BCIR_OP_CMPXCHG;
}
/* A temp of the VALUE type of an object of type `ty` -- unqualified (6.3.2.1p2), its _Bool, plain char, float,
 * complex or `_BitInt` kind kept: the value an atomic read-modify-write yields, and the type it operates on. */
static uint32_t ctype_value_temp(CC *c,const bcir_ctype *ty){
  bcir_ctype t=*ty; t.is_volatile=0; t.is_atomic=0;
  if(t.kind==2) return temp_ptr(c,&t,-1);
  if(t.is_complex) return tempc(c,t.size);
  if(t.is_float) return tempf(c,t.size);
  if(t.bit_width>0) return tempbi(c,t.bit_width,t.signd?1:0);
  uint32_t r=tempi(c,t.size?t.size:4,t.signd?1:0);
  if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
    if(t.is_bool) tr->is_bool=1;
    if(t.is_plain_char) tr->is_plain_char=1; }
  return r;
}
/* One `c.c11atom.rmw:<kind>` claim on the `_Atomic` object of type `ty` addressed as a store addresses it: the
 * reads (base[, idx][, v]); the imm (off, size) for a member, a dereference or a named object, (off, element
 * size, 0, stride) for an array-of-structs field (`stride` > 0), none for a typed element (`size` 0). `kind` is
 * the operator's suffix (`add`, `shl`, ...) or preinc/predec/postinc/postdec. Returns the value temp: the new
 * value, the old one for a postfix operator. */
static uint32_t emit_rmw(CC *c,const char *kind,const bcir_ctype *ty,uint32_t base,int has_idx,uint32_t idx,
                         int has_v,uint32_t v,long long off,long long size,long long stride,int vol,bcir_bounds bnd){
  uint32_t t=ctype_value_temp(c,ty);
  char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.c11atom.rmw:%s",kind);
  bcir_claim *cl=new_claim(c,op,rmw_opcode(kind)); if(!cl) return t;
  cl->n_rd=0; cl->rd[cl->n_rd++]=base; if(has_idx) cl->rd[cl->n_rd++]=idx; if(has_v) cl->rd[cl->n_rd++]=v;
  cl->n_wr=1; cl->wr[0]=t;
  if(size>0){ cl->n_imm=2; cl->imm[0]=off; cl->imm[1]=size;
    if(stride>0){ cl->imm[2]=0; cl->imm[3]=stride; cl->n_imm=4; } }
  cl->bounds=bnd;
  mark_access(c,cl,vol); mark_atomic(cl,1);
  return t;
}
/* The kind of an increment or decrement: pre/post, inc/dec. */
static const char *incdec_kind(int prefix,char ch){
  return prefix ? (ch=='+'?"preinc":"predec") : (ch=='+'?"postinc":"postdec");
}
/* The value of an assignment to an `_Atomic` object `(E = v)`: the value stored, converted to the object's
 * type -- a re-read would be a second atomic access (the oracle's `_cast_value(v, unqualified(ct))`). */
static uint32_t emit_cast(CC *c, uint32_t v, const bcir_ctype *tyin, int si);   /* fwd */
static uint32_t atomic_assign_value(CC *c,uint32_t v,const bcir_ctype *ty){
  bcir_ctype t=*ty; t.is_volatile=0; t.is_atomic=0;
  return emit_cast(c,v,&t,-1);
}

/* typeof( expression ) -- the operand is UNEVALUATED, so resolve its type by SPECULATIVELY lowering it,
 * reading the produced value's type off its resource, then rolling the whole emission back: the resource
 * and claim arrays, the rid/cid/compound-literal counters, and any call-graph / string-literal / local-env
 * side effects the operand triggered. The twin has no separate AST, so reusing the real lowering -- which
 * already types every value faithfully (fixed-width int + signedness, float, pointer pointee/depth, plain
 * char, _Bool) -- is exactly Clang-equivalent for the supported forms (binary, unary, cast, member, index,
 * deref). Calls / address-of are a deferred follow-on (a wide call return loses signedness on its temp).
 * On entry the cursor is at the operand's first token; on return it is just before the closing `)`. */
/* What a speculative lowering (`typeof`, `_Generic`, `sizeof`) rolls back: the resource and claim arrays, the
 * rid/cid/compound-literal counters, the call graph, the local env and the string literals it added. */
typedef struct { size_t nres, ncl; uint32_t rid, cid, clctr; int ncalls, nenv, nstr; } spec_mark;
static void spec_begin(CC *c, spec_mark *m){
  m->nres=c->fn->n_res; m->ncl=c->fn->n_claims; m->rid=c->rid; m->cid=c->cid; m->clctr=c->cl_ctr;
  m->ncalls=c->fn->n_calls; m->nenv=c->nenv; m->nstr=c->fn->n_host_literals;
}
static void spec_end(CC *c, const spec_mark *m){
  c->fn->n_res=m->nres; c->fn->n_claims=m->ncl;
  c->rid=m->rid; c->cid=m->cid; c->cl_ctr=m->clctr;
  c->fn->n_calls=m->ncalls; c->nenv=m->nenv;
  while(c->fn->n_host_literals>m->nstr){
    c->fn->n_host_literals--;
    bcir_host_deallocate(&c->allocator,
                         c->fn->host_literals[c->fn->n_host_literals].spelling);
    c->fn->host_literals[c->fn->n_host_literals].spelling=NULL;
  }
}
static int p_typeof_expr(CC *c, bcir_ctype *ty, int *sidx){
  spec_mark m; spec_begin(c,&m);
  uint32_t v=p_expr(c);                                  /* parse + speculatively lower the operand */
  const bcir_resource *r=res_of(c->fn,v);                /* the produced value's type lives on its resource */
  memset(ty,0,sizeof *ty); ty->kind=0; ty->size=4; ty->signd=1; *sidx=-1;
  if(r){
    if(r->kind==BCIR_RK_POINTER){                        /* a pointer value: pointee width/sign/float + depth */
      ty->kind=2; ty->size=r->elem_bytes?(int)r->elem_bytes:4; ty->signd=r->is_signed;
      ty->is_float=r->is_float; ty->ptr_depth=(uint8_t)(r->ptr_depth?r->ptr_depth:1);
      ty->is_plain_char=r->is_plain_char;
      if(r->agg[0]){ ty->ptr_to_struct=1;               /* a pointer-to-struct: recover the tag + struct index */
        const char *sp=strchr(r->agg,' '); const char *tg=sp?sp+1:r->agg;
        fits(c,ty->tag,sizeof ty->tag,"%s",tg);
        int si=find_struct(c,tg,(int)strlen(tg)); if(si>=0){ *sidx=si; ty->is_union=(uint8_t)c->s[si].is_union; } }
    } else if(r->kind==BCIR_RK_AGGREGATE){              /* a struct/union by value */
      ty->kind=1; ty->size=(int)r->elem_bytes;
      const char *sp=strchr(r->agg,' '); const char *tg=sp?sp+1:r->agg;
      fits(c,ty->tag,sizeof ty->tag,"%s",tg);
      int si=find_struct(c,tg,(int)strlen(tg)); if(si>=0){ *sidx=si; ty->size=c->s[si].size;
        ty->is_union=(uint8_t)c->s[si].is_union; }
    } else if(r->is_funcptr && res_sig(c,v)){           /* a function-pointer value -- `t[0]` of a table, a member --
                                                         * its function type, as the oracle types it (CF-RTFP) */
      int s=res_sig(c,v); const bcir_ctype *rt=&c->sigs[s-1].ret;
      ty->kind=3; ty->size=cc_abi(c)->pointer_size; ty->signd=0; ty->fp_sig=(uint16_t)s;
      fits(c,ty->tag,sizeof ty->tag,"%s",c->sigs[s-1].alias);
      fp_capture_ret(ty,rt,rt->kind==1 ? find_struct(c,rt->tag,(int)strlen(rt->tag)) : -1);
    } else {                                             /* a scalar (integer / float / _Bool / plain char) */
      ty->kind=0; ty->size=r->elem_bytes?(int)r->elem_bytes:4; ty->signd=r->is_signed;
      ty->is_float=r->is_float; ty->is_bool=r->is_bool; ty->is_plain_char=r->is_plain_char;
    }
  }
  spec_end(c,&m);                                        /* roll the speculative emission fully back */
  return c->failed;
}
/* The `_Generic` type identity (C11 §6.5.1.1, after lvalue/array decay; qualifiers ignored): int / int32_t
 * collapse (same width + signedness), plain `char` stays a distinct type from signed/unsigned char, `_Bool`
 * is its own type, floats key on width, a pointer on its pointee, a struct/union on its tag. Two ctypes
 * match iff their identities are equal. Mirrors the oracle's `_type_key`, so both rails pick the same arm. */
static int ctype_generic_eq(const bcir_ctype *a, const bcir_ctype *b){
  if(a->kind!=b->kind) return 0;
  if(a->kind==1) return a->is_union==b->is_union && !strcmp(a->tag,b->tag);   /* struct/union by tag */
  if(a->kind==2){                                  /* pointer: same depth, then the pointee identity below */
    if((a->ptr_depth?a->ptr_depth:1)!=(b->ptr_depth?b->ptr_depth:1)) return 0;
    if(a->ptr_to_struct||b->ptr_to_struct)
      return a->ptr_to_struct==b->ptr_to_struct && (!a->ptr_to_struct||!strcmp(a->tag,b->tag));
  }
  /* a leaf (a scalar, or a pointer's pointee whose width/sign/float/plain-char ride on the ctype): plain
   * `char` and `_Bool` are their own types; a float keys on width (no sign); an integer on width + sign. */
  if(a->is_plain_char||b->is_plain_char) return a->is_plain_char==b->is_plain_char;
  if(a->is_bool||b->is_bool) return a->is_bool==b->is_bool;
  if(a->is_float||b->is_float) return a->is_float==b->is_float && a->size==b->size;
  return a->size==b->size && a->signd==b->signd;
}

/* A pointer temporary cloned from an existing pointer value -- same pointee (width, signedness, float,
 * struct tag). The result of pointer arithmetic `p + i` carries p's type, so the emit declares a real
 * `T *t = p + i` (a pointee-scaled pointer) instead of a width-truncating uint32. */
static uint32_t tempptr(CC *c, uint32_t src){
  bcir_resource snap; int have=res_copy(c,src,&snap);
  /* an array of pointers `T *a[N]` decays to a `T **`: its pointee is the element's (`ptee_*`), one level deeper */
  int parr=have && snap.kind!=BCIR_RK_POINTER && (snap.is_array||snap.is_vla) && snap.ptr_depth;
  int eb=!have ? 4 : parr ? (int)snap.ptee_bytes : (int)snap.elem_bytes;
  uint32_t r=add_res(c,BCIR_DOM_RAM, eb>0?eb:4, 1, 0, BCIR_RK_POINTER, "");
  if(c->fn->n_res && have){ bcir_resource *t=&c->fn->res[c->fn->n_res-1];
    t->is_signed=parr?snap.ptee_signed:snap.is_signed; t->is_float=parr?snap.ptee_float:snap.is_float;
    t->is_plain_char=parr?snap.ptee_plain_char:snap.is_plain_char;
    fits(c,t->agg,sizeof t->agg,"%s",snap.agg);
    if(snap.kind==BCIR_RK_POINTER || parr){            /* ... and the pointer's own depth: `pp + 1` of a `T **` is a
                                                       * `T **`, which a `T *` temp had made it (CF-IDXARROW) */
      t->ptr_depth=(uint8_t)((snap.ptr_depth?snap.ptr_depth:1)+(parr?1:0));
      t->is_voidptr=snap.is_voidptr; t->ptee_funcptr=snap.ptee_funcptr;
      if(!parr){ t->is_complex=snap.is_complex; t->is_bool=snap.is_bool; } } }
  return r;
}
/* A pointer temporary typed from a pointer struct MEMBER's pointee descriptor -- so a loaded `s->p`
 * carries its real `T *` type (the load reads pointer_size bytes; the emit declares `T *`). */
static uint32_t tempptr_field(CC *c, const field *fld){
  int dev = fld->ptee_volatile || (fld->ptee_sidx>=0 && sdef_vol(c,fld->ptee_sidx));
  uint32_t r=add_res(c,dev?BCIR_DOM_MMIO:BCIR_DOM_RAM, fld->ptee_size?fld->ptee_size:4, 1,
                     fld->ptee_volatile, BCIR_RK_POINTER, "");
  if(c->fn->n_res){ bcir_resource *t=&c->fn->res[c->fn->n_res-1];
    t->is_signed=(uint8_t)(fld->signd?1:0); t->is_float=(uint8_t)(fld->ptee_float?1:0);
    t->is_atomic=(uint8_t)(fld->ptee_atomic?1:0);          /* a pointer to `_Atomic` storage (CF-ATOMIC) */
    if(fld->ptee_sidx>=0) fits(c,t->agg,sizeof t->agg,"%s %s",
      c->s[fld->ptee_sidx].is_union?"union":"struct", c->s[fld->ptee_sidx].tag); }
  if(fld->fp_sig && c->fn->n_res){                       /* `op_t *tab`: a pointer to function pointers (CF-FPTAB) */
    const char *al=sig_alias(c,fld->fp_sig); bcir_resource *t=&c->fn->res[c->fn->n_res-1];
    fits(c,t->agg,sizeof t->agg,"%s",al); t->ptee_funcptr=1; }
  return r;
}

static uint32_t emit_member(CC *c, venv *base, const field *fld, int declared_bf) {
  /* the BITFIELD unit temp is sized to a power of 2 >= its byte span (a packed field that straddles into
   * bits >= 32 needs a 64-bit unit); the load reads only `access_bytes` (the spanned bytes). */
  int usz = fld->bit_w ? (fld->access_bytes<=4?4:8) : fld->size;
  uint32_t t=(fld->fp_sig && !fld->is_ptr)?fp_value_temp(c,sig_alias(c,fld->fp_sig))   /* a function pointer read from a
                                          * member: a function-pointer value of its type (CF-FPTAB) */
            :fld->is_ptr?tempptr_field(c,fld):fld->is_complex?tempc(c,fld->size):fld->is_float?tempf(c,fld->size)
            :fld->bit_w?tempi(c,usz,0)   /* a BITFIELD storage unit: a plain unsigned load (bf.get extracts below),
                                          * even a `_BitInt(N)` bitfield -- its unit is read raw, then masked. */
            :fld->bit_width>0?tempbi(c,fld->bit_width,fld->signd)   /* a PLAIN C23 `_BitInt(N)` member: load at the
                                                                    * storage width, typed `_BitInt(N)` (faithful) */
            :fld->sidx>=0?tempagg(c,fld->sidx)   /* a struct/union member: the aggregate value (CF-STRUCTVAL) */
            :tempi(c,usz,fld->signd);   /* loaded value carries the field's type */
  if(fld->is_plain_char && c->fn->n_res) c->fn->res[c->fn->n_res-1].is_plain_char=1;   /* read as `char`, not int8_t */
  if(fld->is_bool && fld->is_atomic && c->fn->n_res) c->fn->res[c->fn->n_res-1].is_bool=1;   /* an `_Atomic _Bool`
                                                   * is read as one: the atomic load names the object's type */
  bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
  cl->n_rd=1;cl->rd[0]=base->rid;cl->n_wr=1;cl->wr[0]=t;cl->n_imm=2;cl->imm[0]=fld->byte_off;cl->imm[1]=fld->bit_w?fld->access_bytes:fld->size;
  cl->bounds=BCIR_BND_ASSUMED;
  mark_access(c,cl,fld->is_volatile||base->type.is_volatile);
  mark_atomic(cl,fld->is_atomic);                  /* a read of an `_Atomic` member: an atomic load */
  if(fld->bit_w){uint32_t u=t;
    /* integer promotion (6.3.1.1) of a bitfield read, keyed on the BITFIELD WIDTH W (`fld->bit_w`):
     *   * W <= 32  -> promotes to int (int holds all W-bit values), so an UNSIGNED sub-int bitfield reads
     *                 as a SIGNED int (`bf < x` is a signed compare); W==32 unsigned stays unsigned.
     *   * W >  32  -> int can't hold it: a `_BitInt(N)` bitfield stays `_BitInt(N)` (does NOT promote --
     *                 matching Clang's `s.wide + s.wide == _BitInt(N)`); a standard wide field keeps its
     *                 declared 64-bit type. So a `_BitInt(N<=32)` bitfield promotes to int just like a
     *                 standard one -- VERIFIED == Clang via the `_Generic` differential. */
    if(declared_bf) t=fld->bit_width>0?tempbi(c,fld->bit_width,fld->signd):tempi(c,fld->size,fld->signd);
    else if(fld->bit_width>0 && fld->bit_w>32) t=tempbi(c,fld->bit_width,fld->signd);   /* WIDE `_BitInt(N)` bitfield */
    else t=tempi(c,fld->bit_w>32?fld->size:4,(fld->signd||fld->bit_w<32)?1:0);
    bcir_claim *g=new_claim(c,"c.bf.get",BCIR_OP_ADD);if(!g)return t;
    g->n_rd=1;g->rd[0]=u;g->n_wr=1;g->wr[0]=t;g->n_imm=3;g->imm[0]=fld->bit_off;g->imm[1]=fld->bit_w;g->imm[2]=fld->signd;}
  return t;
}
/* `s.arr[idx]` -- a load from a struct member array: the element lands at `&s + member_off + idx*elem`,
 * so the claim carries the base, the index, and (member byte offset, element size) in imm. */
static uint32_t emit_member_index(CC *c, venv *base, const field *fld, uint32_t idx) {
  uint32_t t=(fld->fp_sig && !fld->elem_ptr)?fp_value_temp(c,sig_alias(c,fld->fp_sig))   /* an element of a member
                                                        * table of function pointers (CF-FPTAB) */
            :fld->elem_ptr?tempptr_field(c,fld)        /* an array-of-pointers element: a `T *` temp */
            :fld->elem_sidx>=0?tempagg(c,fld->elem_sidx)   /* an array-of-structs element: the aggregate (CF-STRUCTVAL) */
            :fld->is_complex?tempc(c,fld->size):fld->is_float?tempf(c,fld->size):tempi(c,fld->size,fld->signd);
  if(fld->is_plain_char && c->fn->n_res) c->fn->res[c->fn->n_res-1].is_plain_char=1;   /* `char[]` element: `char` */
  if(fld->is_bool && fld->is_atomic && c->fn->n_res) c->fn->res[c->fn->n_res-1].is_bool=1;   /* `_Atomic _Bool[]` */
  bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
  cl->n_rd=2;cl->rd[0]=base->rid;cl->rd[1]=idx;cl->n_wr=1;cl->wr[0]=t;
  cl->n_imm=2;cl->imm[0]=fld->byte_off;cl->imm[1]=fld->size;cl->bounds=BCIR_BND_ASSUMED;
  mark_access(c,cl,fld->is_volatile||base->type.is_volatile);
  mark_atomic(cl,fld->is_atomic);
  return t;
}
static field member_descend(CC *c, field f);   /* fwd: a nested `.m.k` member chain -> one flattened offset */
/* The member of an array-of-structs element (the element struct is sdef `sidx`) named after the `.`/`->` at the
 * cursor, through any depth of nested struct/union members `.m.k` (CF-NESTMEM): one flattened offset in the
 * element, which the access keeps striding by -- the oracle's `_aos_member`. Only a PLAIN SCALAR leaf is handled
 * (a bitfield/array/struct/pointer leaf is a consistent follow-on -- both rails route to fallback). Returns 1
 * with *sub filled, else 0 (and may have raised via fail()). The cursor is on the `.`/`->`. */
static int sdef_elem_field(CC *c, int sidx, field *sub) {
  c->i++; tok fn=adv(c); sdef *ES=&c->s[sidx]; int fi=-1;
  for(int i=0;i<ES->nf;i++) if((int)strlen(ES->f[i].name)==fn.n&&!strncmp(ES->f[i].name,fn.s,fn.n)) fi=i;
  if(fi<0){ fail(c,"unknown field"); return 0; }
  field s=member_descend(c,ES->f[fi]);
  if(s.bit_w||s.arr_count||s.sidx>=0||s.is_ptr){ fail(c,"array-of-structs non-scalar element field"); return 0; }
  *sub=s; return 1;
}
/* A member access through a subscripted element that is a pointer -- `arr[i]->m`, `pp[i]->m`, `o.p[i]->m`: the oracle's
 * refusal (`_addr` has no `Index` base), the twin's too (`index_member_ok`, `elem_field`; CF-IDXARROW). */
#define CC_INDEX_BASE "unsupported base expression Index"
/* After `arr[i]` on an ARRAY-OF-STRUCTS member (`arr->elem_sidx>=0`) with a trailing `.`/`->`: the element
 * field (`sdef_elem_field`). Returns 1 with *sub filled, else 0 (and may have raised via fail()). */
static int elem_field(CC *c, const field *arr, field *sub) {
  /* `o.p[i]->m` of an array of pointers to structs: refused as the oracle refuses it (CF-IDXARROW) */
  if(arr->elem_ptr && arr->ptee_sidx>=0 && (is(c,".")||is(c,"->"))){ fail(c,CC_INDEX_BASE); return 0; }
  if(arr->elem_sidx<0 || !(is(c,".")||is(c,"->"))) return 0;
  return sdef_elem_field(c,arr->elem_sidx,sub);
}
/* `arr[i].field` (array-of-structs): load `sub->size` bytes at member_off(arr)+offsetof(sub), STRIDING by the
 * element size `arr->size` (imm[2]) -- decoupled from the field copy size (imm[1]). */
static uint32_t emit_member_index_field(CC *c, venv *base, const field *arr, uint32_t idx, const field *sub) {
  uint32_t t=sub->is_float?tempf(c,sub->size):tempi(c,sub->size,sub->signd);
  if(sub->is_plain_char && c->fn->n_res) c->fn->res[c->fn->n_res-1].is_plain_char=1;
  if(sub->is_bool && sub->is_atomic && c->fn->n_res) c->fn->res[c->fn->n_res-1].is_bool=1;   /* `_Atomic _Bool` field */
  bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
  cl->n_rd=2;cl->rd[0]=base->rid;cl->rd[1]=idx;cl->n_wr=1;cl->wr[0]=t;
  cl->n_imm=3;cl->imm[0]=arr->byte_off+sub->byte_off;cl->imm[1]=sub->size;cl->imm[2]=arr->size;
  cl->bounds=BCIR_BND_ASSUMED;
  mark_access(c,cl,sub->is_volatile||arr->is_volatile||base->type.is_volatile);
  mark_atomic(cl,sub->is_atomic);
  return t;
}
/* A member of `base[i]` is a member of the element only when the element is the struct or union itself -- an array
 * of them, or through a pointer to one (`index_elem_sidx`). An element that is a pointer to one -- `arr[i]->m` of an
 * array of pointers, `pp[i]->m` through a `T **` -- is no struct: the oracle has no subscript base (`_addr`) and
 * refuses it (`CC_INDEX_BASE`), and the twin had read the pointer's own slot as the struct it points to, a silent
 * miscompile (CF-IDXARROW). 0 after the refusal. */
static int index_member_ok(CC *c, const venv *base){
  if(index_elem_sidx(c,base)>=0) return 1;
  fail(c,CC_INDEX_BASE); return 0;
}
/* After `a[i]` on a DIRECT local/global ARRAY-OF-STRUCTS variable, or a pointer to structs (`v->sidx>=0`, the
 * element struct), with a trailing `.`/`->`: the element field (`sdef_elem_field`). Returns 1 with *sub filled,
 * else 0 (may raise via fail()); the cursor is past the field chain. */
static int aos_elem_field(CC *c, venv *base, field *sub) {
  if(base->sidx<0 || !(is(c,".")||is(c,"->"))) return 0;
  if(!index_member_ok(c,base)) return 0;             /* an element that is a pointer (CF-IDXARROW) */
  return sdef_elem_field(c,base->sidx,sub);
}
/* `a[i].field` on a DIRECT array-of-structs variable: load `sub->size` bytes at offsetof(sub), STRIDING by
 * the element (struct) size `v->type.size` (imm[2]) -- the base is the array itself (member offset 0). */
static uint32_t emit_index_field(CC *c, venv *base, uint32_t idx, const field *sub) {
  uint32_t t=sub->is_float?tempf(c,sub->size):tempi(c,sub->size,sub->signd);
  if(sub->is_plain_char && c->fn->n_res) c->fn->res[c->fn->n_res-1].is_plain_char=1;
  if(sub->is_bool && sub->is_atomic && c->fn->n_res) c->fn->res[c->fn->n_res-1].is_bool=1;   /* `_Atomic _Bool` field */
  bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
  cl->n_rd=2;cl->rd[0]=base->rid;cl->rd[1]=idx;cl->n_wr=1;cl->wr[0]=t;
  cl->n_imm=3;cl->imm[0]=sub->byte_off;cl->imm[1]=sub->size;cl->imm[2]=base->type.size;
  cl->bounds=BCIR_BND_ASSUMED;
  mark_access(c,cl,sub->is_volatile||base->type.is_volatile);
  mark_atomic(cl,sub->is_atomic);
  return t;
}
/* --- C's assignment conversion at a store the emit spells as a byte copy (CF-MEMCONV) ------------------- */
static void cast_name(CC *c,const bcir_ctype *ty,int signed_int,char *o,size_t n);   /* fwd: a cast's spelling */
/* `(ty)v`: one `c.cast` claim into a temp of the target type -- the explicit cast, and C's assignment
 * conversion at a store (`store_conv`). The oracle's `_cast_value`. */
static uint32_t emit_cast(CC *c, uint32_t v, const bcir_ctype *tyin, int si){
  bcir_ctype ty=*tyin;
  v=null_pointer(c,v,&ty);   /* `(T *)0`, `(op_t)0`: the constant is a null pointer of the type (C11 6.3.2.3p3) -- an
                              * `int` temp cast to a pointer of another width is a cast GCC rejects (CF-RTFP) */
  const bcir_resource *vr=res_of(c->fn,v);
  /* The result temp carries the target's signedness, so a signed (sub-int) target keeps its sign
   * even when the cast value is used directly -- `(signed char)(-5)` stays -5, and `(int)u` reads
   * back signed (an arithmetic `>>`). A float -> signed-int conversion additionally needs a SIGNED
   * cast operator (float -> unsigned is UB / target-divergent). */
  int f2s = vr && vr->is_float && !ty.is_float && ty.kind!=2 && ty.signd && ty.size>0
            && ty.bit_width==0;                                  /* a `_BitInt` keeps its exact spelling */
  uint32_t r = ty.kind==3 ? fp_value_temp(c, ty.tag)            /* a function pointer of the target type (CF-RTFP) */
             : ty.kind==2 ? temp_ptr(c, &ty, si)                /* a pointer cast: a `T *` of the target (first:
                                                                  * a `float *` target is a pointer, not a float) */
             : ty.is_complex ? tempc(c, ty.size)                /* a _Complex cast -> a complex temp */
             : ty.is_float ? tempf(c, ty.size)                  /* a float cast -> a float temp */
             : ty.bit_width>0 ? tempbi(c, ty.bit_width, ty.signd?1:0)   /* a C23 `_BitInt(N)` cast */
                          : tempi(c, ty.size?ty.size:4, ty.signd?1:0);   /* an integer cast */
  if(ty.is_bool && !ty.is_float && ty.kind!=2 && c->fn->n_res)
    c->fn->res[c->fn->n_res-1].is_bool=1;    /* a bool cast -> a _Bool temp (normalizes to 0/1) */
  char op[BCIR_CIR_OP]; cast_name(c,&ty,f2s,op,sizeof op);
  bcir_claim *cl=new_claim(c,op,BCIR_OP_ADD);
  if(cl){cl->n_rd=1;cl->rd[0]=v;cl->n_wr=1;cl->wr[0]=r;}
  return r;
}
/* A value's arithmetic class for C's assignment conversion -- 0 integer, 1 real floating, 2 complex -- or -1
 * when it takes none: a pointer (a file-scope one's slot included), an aggregate, a function pointer, an
 * array. The oracle's `_arith_class` over a scalar value type. */
static int res_arith_class(const bcir_resource *r){
  if(!r || r->kind!=BCIR_RK_SCALAR || r->is_funcptr || r->is_pointer || r->count>1 || r->is_array || r->is_vla)
    return -1;
  return r->is_complex?2:r->is_float?1:0;
}
/* Whether `a` is an operand C gives the unary operator whose claim suffix is `suf` (C11 6.5.3.3p1; the oracle's
 * `_unary_operand`): `+` ("plus") and `-` take an arithmetic operand, `~` an integer one -- or a complex, whose conjugate
 * GCC and Clang take it for -- `!` a scalar, GNU `__real__` / `__imag__` an arithmetic one. A struct or union is refused
 * as every operator's operand (`agg_operand`), anything else as Clang refuses it (`CC_UNARY_NOT`). Checked as each
 * operator lowers, so a speculative lowering -- `sizeof`, `typeof`, `_Generic` -- refuses it too, where a check over the
 * unit's claims once it was parsed never saw them (CF-UNARY). NOINLINE: it stays out of p_unary_inner's frame, which
 * recurs at every level of a nested expression (CF-SPLIT2.1). */
static BCIR_NOINLINE int unary_operand_ok(CC *c, uint32_t a, const char *suf){
  if(c->failed || agg_operand(c,a)) return 0;
  const bcir_resource *r=res_of(c->fn,a);
  if(!r || !strcmp(suf,"lnot")) return 1;
  int k=res_arith_class(r);                         /* -1: a pointer, a function, an array */
  if(k<0 || (k==1 && !strcmp(suf,"bnot"))){ fail(c,CC_UNARY_NOT); return 0; }
  return 1;
}
/* Whether `v` may control `if`, `while`, `for`, `do` or `?:`, which C requires to be a scalar (C11 6.8.4.1p1, 6.8.5p2,
 * 6.5.15p2), or with `sw` a `switch`, an integer (6.8.4.2p1): a struct or union is refused as an operator's operand is
 * (`agg_operand`), a `switch` of a pointer, a floating value, an array or a function as Clang refuses it
 * (`CC_SWITCH_NOT`); a void value at the statement's end, as every use of one (`void_read`). Both rails had lowered `if
 * (a)` of a struct, an emit no compiler takes (the oracle's `_condition` and `_switch_value`, CF-STRUCTCOND). */
static BCIR_NOINLINE int cond_value_ok(CC *c, uint32_t v, int sw){
  if(c->failed || agg_operand(c,v)) return 0;
  const bcir_resource *r=res_of(c->fn,v);
  if(sw && r && !r->is_void && res_arith_class(r)!=0){ fail(c,CC_SWITCH_NOT); return 0; }
  return 1;
}
/* A `++` or `--` the increment's own reading (`incdec_value`) did not take -- after its operand, `rest` the operand's
 * value, or before it (`rest` 0, the cursor on the operator): a struct or union, `(*p)++`, `p->in++`, `g_a[1]++`, `++*p`
 * of a struct (C11 6.5.2.4p1, 6.5.3.1p1), is refused as an operator's operand is (`agg_operand`), any other lvalue form
 * as the oracle's `_incdec_value` refuses those it does not step. Each had been a parse error for want of a `;` or of
 * an expression (CF-STRUCTCOND). NOINLINE beside p_unary_inner, whose frame recurs at every level. */
static uint32_t p_unary(CC *c);   /* fwd: the operand of a prefix step */
static BCIR_NOINLINE uint32_t incdec_rest(CC *c, int prefix, uint32_t rest){
  if(c->failed) return 0;
  if(prefix){ c->i++; rest=p_unary(c); if(c->failed) return 0; }
  else if(!is(c,"++") && !is(c,"--")) return rest;
  if(!agg_operand(c,rest)) fail(c,"inc/dec of this lvalue form is a follow-on");
  return 0;
}
/* `+a` (C11 6.5.3.3p2): a's value, integer-promoted -- a `c.cast` to `int` when the promotion changes its type, as
 * `(int)a` lowers, else `a` itself with no claim (the oracle's `+`). The parser had dropped the `+`, so `sizeof(+c)`,
 * `_Generic(+c, ...)` and `typeof(+c)` read `c`'s own type on both rails (CF-UNARY). */
static BCIR_NOINLINE uint32_t unary_plus(CC *c, uint32_t a){
  if(!unary_operand_ok(c,a,"plus")) return 0;
  const bcir_resource *r=res_of(c->fn,a);
  if(!r || r->is_float || r->bit_width>0 || r->elem_bytes>=4) return a;   /* a float, a `_BitInt`, an `int` or wider */
  bcir_ctype t; memset(&t,0,sizeof t); t.size=4; t.signd=1;
  return emit_cast(c,a,&t,-1);
}
/* The temp of GNU `__real__ a` / `__imag__ a` (the oracle's `_part_type`): a complex's element float; a real operand's
 * own type, not promoted -- an integer one was a `double`, so `sizeof(__real__ c)` of a `uint8_t c` was 8 where Clang
 * gives 1 (CF-UNARY). */
static BCIR_NOINLINE uint32_t part_temp(CC *c, uint32_t a){
  bcir_resource ar;
  if(!res_copy(c,a,&ar)) return tempf(c,8);
  if(ar.is_complex) return tempf(c,(int)ar.elem_bytes/2);
  if(ar.is_float) return tempf(c,(int)ar.elem_bytes);
  uint32_t r = ar.bit_width>0 ? tempbi(c,ar.bit_width,ar.is_signed) : tempi(c,(int)ar.elem_bytes,ar.is_signed);
  if(c->fn->n_res){ bcir_resource *t=&c->fn->res[c->fn->n_res-1]; t->is_bool=ar.is_bool; t->is_plain_char=ar.is_plain_char; }
  return r;
}
/* C's assignment conversion (C23 6.5.17.2) at a store the emit spells as a byte copy -- a member, a
 * member-array element, a field of an array of structs, a bitfield, a store through a pointer, an
 * initializer's member or element: the value converts to the slot's declared type `slot`. The emit picks the
 * stored bytes' type from the VALUE, so a value of another arithmetic class -- an integer into a float slot, a
 * float into an integer one, a real into a complex one -- converts first, through the `c.cast` an explicit
 * `(T)v` lowers to, and the conversion is in the claim graph (the oracle's `_store_conversion`). A width or
 * sign change within a class is the emit's; a `_Bool` slot normalizes by its flag; a pointer, aggregate or
 * unsized slot takes no arithmetic conversion. Returns the value to store: every byte-copy store asks this. */
static uint32_t store_conv(CC *c, uint32_t val, const bcir_ctype *slot){
  if(slot->kind!=0 || slot->is_bool || slot->size<=0) return val;
  int vc=res_arith_class(res_of(c->fn,val));
  int sc=slot->is_complex?2:slot->is_float?1:0;
  if(vc<0 || vc==sc) return val;
  bcir_ctype t=*slot; t.is_volatile=0; t.is_atomic=0;   /* a value is never qualified (6.3.2.1p2) */
  return emit_cast(c,val,&t,-1);
}
/* The declared type of the slot a member store writes -- the member; a member array's element; a bitfield's
 * declared type -- for `store_conv`. A pointer, struct or union member is no arithmetic slot. */
static bcir_ctype field_slot(const field *f){
  bcir_ctype t; memset(&t,0,sizeof t);
  t.kind=(uint8_t)((f->is_ptr||f->elem_ptr)?2:(f->sidx>=0||f->elem_sidx>=0)?1:0);
  t.size=f->size; t.signd=f->signd; t.is_float=(uint8_t)(f->is_float?1:0);
  t.is_complex=(uint8_t)(f->is_complex?1:0); t.is_bool=(uint8_t)(f->is_bool?1:0);
  t.is_plain_char=(uint8_t)(f->is_plain_char?1:0); t.bit_width=f->bit_width;
  t.is_atomic=(uint8_t)(f->is_atomic?1:0);                  /* `_Atomic` storage (CF-ATOMIC) */
  return t;
}
/* The declared type of the object `*p` writes, for `p` of type `*p_ty` (an array's element when it is an
 * array): a pointer to pointers or to a struct is no arithmetic slot. */
static bcir_ctype pointee_slot(const bcir_ctype *p_ty){
  bcir_ctype t=*p_ty;
  if(p_ty->kind==2){
    if((p_ty->ptr_depth?p_ty->ptr_depth:1)>1) return t;       /* the slot holds a pointer (kind 2) */
    t.kind=(uint8_t)(p_ty->ptr_to_struct?1:0); t.ptr_depth=0; }
  t.nadims=0; t.is_volatile=0;
  return t;
}
/* The element type of `v[i]` -- a depth-1 pointer's pointee, an array's element: the type an `_Atomic`
 * element's read-modify-write operates on and the value an assignment to it yields (CF-ATOMIC). */
static bcir_ctype index_elem_ctype(const venv *v){
  bcir_ctype t = v->type.kind==2 ? pointee_slot(&v->type) : v->type;
  t.nadims=0; return t;
}
/* The same for a pointer held in a resource (the general `*(expr) = v` store): its pointee's flags. */
static bcir_ctype res_pointee_slot(const bcir_resource *r){
  bcir_ctype t; memset(&t,0,sizeof t);
  if(!r || r->kind!=BCIR_RK_POINTER || (r->ptr_depth?r->ptr_depth:1)>1 || r->agg[0] || r->is_voidptr){
    t.kind=2; return t; }
  t.size=(int)r->elem_bytes; t.signd=r->is_signed; t.is_float=r->is_float; t.is_complex=r->is_complex;
  t.is_bool=r->is_bool; t.is_plain_char=r->is_plain_char; t.bit_width=r->bit_width;
  return t;
}
/* `a[i].field = val` on a DIRECT array-of-structs variable: store `sub->size` bytes at offsetof(sub),
 * STRIDING by the element (struct) size. The base is the array itself (member offset 0). Mirrors the
 * member-array strided store (imm = [field_off, field_size, _Bool-flag, stride]). */
static uint32_t store_index_field(CC *c, venv *base, uint32_t idx, const field *sub, uint32_t val) {
  bcir_ctype st=field_slot(sub); val=store_conv(c,val,&st);        /* returns the value stored */
  bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
  if(!cl) return val;
  cl->n_rd=3;cl->rd[0]=base->rid;cl->rd[1]=idx;cl->rd[2]=val;cl->n_imm=2;
  cl->imm[0]=sub->byte_off;cl->imm[1]=sub->size;cl->bounds=BCIR_BND_ASSUMED;
  if(sub->is_bool){cl->imm[2]=1;cl->n_imm=3;}
  if(cl->n_imm<3){cl->imm[2]=0;cl->n_imm=3;} cl->imm[3]=base->type.size; cl->n_imm=4;   /* stride imm[3] */
  mark_access(c,cl,sub->is_volatile||base->type.is_volatile);
  mark_atomic(cl,sub->is_atomic);
  return val;
}
/* `s.arr[i] = val` (member array, the element copy size == the element/stride size) OR `s.arr[i].field = val`
 * (member array-of-structs, the field copy size != the element stride): store at member_off + field_off,
 * striding by the element size. `sf` is the stored slot (the element FIELD for AOS, else the array element),
 * `soa` selects which (1 -> AOS field, stride = arr->size; 0 -> plain element, stride == copy size). */
static uint32_t store_member_index(CC *c, venv *base, const field *arr, uint32_t idx,
                                   int soa, const field *sf, uint32_t val) {
  bcir_ctype st=field_slot(sf); val=store_conv(c,val,&st);          /* returns the value stored */
  bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
  if(!cl) return val;
  cl->n_rd=3;cl->rd[0]=base->rid;cl->rd[1]=idx;cl->rd[2]=val;cl->n_imm=2;
  cl->imm[0]= soa ? arr->byte_off+sf->byte_off : arr->byte_off; cl->imm[1]=sf->size;
  cl->bounds=BCIR_BND_ASSUMED;
  if(sf->is_bool){cl->imm[2]=1;cl->n_imm=3;}                      /* a _Bool element/field: normalize on store */
  if(soa){ if(cl->n_imm<3){cl->imm[2]=0;cl->n_imm=3;} cl->imm[3]=arr->size; cl->n_imm=4; }   /* stride imm[3] */
  mark_access(c,cl,sf->is_volatile||arr->is_volatile||base->type.is_volatile);
  mark_atomic(cl,sf->is_atomic);
  return val;
}
/* Parse the `[i]` (or `[i][j][k]`) indices of a member-array access and flatten them row-major into a
 * single linear index (Horner: lin = lin*adims[d] + idx[d]) -- matching the oracle, so 1-D `s.a[i]` and
 * N-D `s.m[i][j]` both reduce to one element-scaled index into the member at its byte offset. */
static uint32_t member_arr_index(CC *c, const field *fld) {
  uint32_t lin=0; int d=0;
  while(is(c,"[")){
    if(fld->elem_ptr && d>=(fld->nadims>0?fld->nadims:1)){   /* `t->arr[i][j]` on `T *arr[N]`: a subscript of the
      * loaded POINTER, not a further dimension -- the oracle refuses it (partial indexing), so the twin does too
      * rather than folding `j` into the element index */
      fail(c,"partial indexing of a struct member array is not yet supported"); return lin; }
    c->i++; uint32_t ix=p_expr(c); eat(c,"]");
    if(d==0){ lin=ix; }
    else { int dim = d<fld->nadims ? fld->adims[d] : 1;
      uint32_t k=temp(c,4); bcir_claim *kc=new_claim(c,"c.const",BCIR_OP_LOAD);
      if(kc){kc->n_wr=1;kc->wr[0]=k;kc->n_imm=1;kc->imm[0]=dim;}
      uint32_t m1=temp(c,4); bcir_claim *mc=new_claim(c,"c.bin.mul",BCIR_OP_MUL);
      if(mc){mc->n_rd=2;mc->rd[0]=lin;mc->rd[1]=k;mc->n_wr=1;mc->wr[0]=m1;}
      uint32_t a1=temp(c,4); bcir_claim *ac=new_claim(c,"c.bin.add",BCIR_OP_ADD);
      if(ac){ac->n_rd=2;ac->rd[0]=m1;ac->rd[1]=ix;ac->n_wr=1;ac->wr[0]=a1;}
      lin=a1; }
    d++; }
  return lin;
}
static int is_compound_op(const tok *t);   /* fwd: a `+= ... >>=` compound-assign punctuator */

/* --- §5.12 recoverable-extent mutation pre-pass (the C twin of lower._scan_mutations) ----------
 * A token-level over-approximation of the oracle's AST walk over the function body: per NAME, the number
 * of assignments to it and whether its address is ever taken. The body is the balanced `{...}` token
 * range starting at `start` (just past the opening brace). It MUST agree with the oracle's AST walk on
 * every fixture (the parity gate enforces it): more mutation seen -> fewer promotions, never an unsound
 * one. */
static mutent *mut_find(CC *c, const tok *id) {
  for (int k = 0; k < c->mut_n; k++)
    if ((int)strlen(c->mut[k].name) == id->n && !strncmp(c->mut[k].name, id->s, (size_t)id->n))
      return &c->mut[k];
  if (c->mut_n < (int)(sizeof c->mut / sizeof c->mut[0])) {
    idcpy(c,c->mut[c->mut_n].name, id);
    c->mut[c->mut_n].assigned = 0; c->mut[c->mut_n].body = 0; c->mut[c->mut_n].addr = 0;
    return &c->mut[c->mut_n++];
  }
  return NULL;
}
static int mut_assigned(CC *c, const tok *id) {            /* TOTAL assignment count of a NAME (0 if unseen) */
  for (int k = 0; k < c->mut_n; k++)
    if ((int)strlen(c->mut[k].name) == id->n && !strncmp(c->mut[k].name, id->s, (size_t)id->n))
      return c->mut[k].assigned;
  return 0;
}
static int mut_body(CC *c, const tok *id) {                /* NON-decl-init assignment count (the count gate) */
  for (int k = 0; k < c->mut_n; k++)
    if ((int)strlen(c->mut[k].name) == id->n && !strncmp(c->mut[k].name, id->s, (size_t)id->n))
      return c->mut[k].body;
  return 0;
}
static int mut_addr(CC *c, const tok *id) {                /* has the NAME's address been taken? */
  for (int k = 0; k < c->mut_n; k++)
    if ((int)strlen(c->mut[k].name) == id->n && !strncmp(c->mut[k].name, id->s, (size_t)id->n))
      return c->mut[k].addr;
  return 0;
}
/* Is token `t` a value-ENDER -- so a following `&` is the BINARY bitwise-and, not a unary address-of?
 * (An identifier / number / string / `)` / `]`.) */
static int is_value_ender(const tok *t) {
  if (t->k == T_ID || t->k == T_INT || t->k == T_FLT || t->k == T_STR) return 1;
  return t->k == T_PUN && t->n == 1 && (t->s[0] == ')' || t->s[0] == ']');
}
static void scan_mutations(CC *c, int start) {
  c->mut_n = 0;
  c->ext_ctr = 0;                                         /* §5.12 reset the per-function snapshot counter */
  int depth = 1;                                           /* `start` is just past the opening `{` */
  for (int i = start; c->t[i].k != T_END && depth > 0; i++) {
    const tok *t = &c->t[i];
    if (t->k == T_PUN && t->n == 1 && t->s[0] == '{') { depth++; continue; }
    if (t->k == T_PUN && t->n == 1 && t->s[0] == '}') { depth--; continue; }
    if (t->k == T_ID) {
      const tok *nx = &c->t[i + 1];                        /* id `=` / id OP= / id++ / id-- -> an assignment */
      int plain = nx->k == T_PUN && nx->n == 1 && nx->s[0] == '=';
      int other = nx->k == T_PUN && (is_compound_op(nx)    /* OP= / postfix ++ / -- : never a decl-init */
          || (nx->n == 2 && (nx->s[0] == '+' || nx->s[0] == '-') && nx->s[1] == nx->s[0]));
      if (plain || other) {
        mutent *m = mut_find(c, t);
        if (m) {
          m->assigned++;
          /* A decl-init (`<type> id = ...`, the type token just before the name) leaves the value stable
           * from the alloc onward; an ORDINARY write (any OP=/++/--, or a plain `=` not in a declarator)
           * may mutate the count AFTER the alloc -> a body assignment that disqualifies it (mirrors the
           * oracle distinguishing a cast.Decl init from a cast.Assign). */
          const tok *pv = (i > start) ? &c->t[i - 1] : NULL;
          int declinit = plain && pv && pv->k == T_ID
              && (scalar_size(pv->s, pv->n) >= 0 || find_typedef(c, pv->s, pv->n) >= 0);
          if (!declinit) m->body++;
        }
      }
      continue;
    }
    if (t->k == T_PUN && t->n == 2 && (t->s[0] == '+' || t->s[0] == '-') && t->s[1] == t->s[0]
        && c->t[i + 1].k == T_ID) {                        /* ++id / --id (the id is NOT a value-ender before it) */
      const tok *pv = (i > start) ? &c->t[i - 1] : NULL;
      if (!(pv && is_value_ender(pv))) {                   /* a postfix x++ is counted by the id-rule above */
        mutent *m = mut_find(c, &c->t[i + 1]);
        if (m) { m->assigned++; m->body++; }              /* a pre-inc/dec is always an ordinary write */
      }
      continue;
    }
    if (t->k == T_PUN && t->n == 1 && t->s[0] == '&') {   /* a UNARY `&x` (address-of): the oracle's rule --
                                                          * the operand is the name itself, parenthesized or not;
                                                          * `&p[i]`, `&s.m`, `&p->m` address what it reaches */
      const tok *pv = (i > start) ? &c->t[i - 1] : NULL;
      int np = 0, j = i + 1;
      while (tok_is(&c->t[j], "(")) { np++; j++; }
      if (c->t[j].k == T_ID && !(pv && is_value_ender(pv))) {   /* unary when the previous token is not a value-ender */
        int k = j + 1, closed = 0;
        while (closed < np && tok_is(&c->t[k], ")")) { closed++; k++; }
        if (closed == np && !tok_is(&c->t[k], "[") && !tok_is(&c->t[k], ".") && !tok_is(&c->t[k], "->")) {
          mutent *m = mut_find(c, &c->t[j]);
          if (m) m->addr = 1;
        }
      }
    }
  }
}

/* --- CF-PAREN: a postfix operator after a parenthesized operand ---------------------------------------------
 * The oracle's parser reads `( e )` as `e` (cparse `_primary`), so `(a)[1]`, `(s).x`, `(p)->x`, `(x)++` and
 * `((uint32_t[3]){s, 7})[1]` are the postfix chains without their parentheses, and it folds `(*p).x` to `p->x`
 * and `(*p)[i]` to `p[0][i]` (`_postfix_tail`). The twin lowers while it parses, from an identifier-rooted chain
 * at each site -- a read, a store, a value assignment, an inc/dec, `&` -- so this pass rewrites a function body's
 * tokens to those spellings before any of it is parsed (the mutation pre-pass has read the source spelling, as
 * the oracle's AST walk sees `(x)++` and `&(p)[i]`). P is a name, a string or a compound literal, then `[..]` `.m`
 * `->m`:
 *     (^n P )^n OP   ->  P OP   (OP one of [ . ->)      (^n * P )^n .  ->  P ->
 *     (^n P )^n ++   ->  (^n P ++ )^n                  (^n * P )^n [  ->  P [ 0 ] [
 * -- `(x)++;` stays the expression statement the oracle's parse keeps (its `x++;` is an assignment) -- and only
 * where the `(` opens a parenthesized expression: not a call's arguments (after a name, `)` or `]`), a cast's type
 * name, an `if`/`while`/`switch` condition, or a declarator (`T (x)[N]`, a form neither rail takes). A pass
 * rewrites the innermost parentheses of a nesting; the caller repeats it until nothing changes. */
typedef struct { uint8_t blk, start, decl, init, lab, forhdr, fstart; int pd, dpd, xpd; } upframe;
#define UP_FRAMES 64
/* The token after the balanced group that opens at `j` (`(`/`[`/`{` ... its closer), or -1 unbalanced. */
static int up_group_end(CC *c, int j){
  char o=tat(c,j)->s[0], cl=o=='('?')':o=='['?']':'}'; int d=0;
  for(int k=j; tat(c,k)->k!=T_END; k++){
    const tok *t=tat(c,k); if(t->k!=T_PUN || t->n!=1) continue;
    if(t->s[0]==o) d++; else if(t->s[0]==cl && --d==0) return k+1;
  }
  return -1;
}
/* The `(` matching the `)` at `j`, scanning back no further than `lo` (-1 when none). */
static int up_match_open(CC *c, int j, int lo){
  int d=0;
  for(int k=j; k>=lo; k--){ const tok *t=tat(c,k);
    if(tok_is(t,")")) d++; else if(tok_is(t,"(") && --d==0) return k; }
  return -1;
}
/* The `(` at `q` opens a type name -- a cast (after another one too) or a compound literal -- and not a `sizeof`/
 * `typeof`/`_Atomic` operand or an attribute's (after a name): no call's arguments or condition start with one. */
static int up_cast_open(CC *c, int q){
  const tok *b=tat(c,q-1), *t=tat(c,q+1);
  if(b->k==T_ID && !tok_is(b,"return") && !tok_is(b,"else") && !tok_is(b,"do")) return 0;
  return t->k==T_ID && decl_type_tok(c,t);
}
/* A name the pass may treat as an operand (not a type, not a keyword). */
static int up_operand_name(CC *c, const tok *t){
  static const char *const kw[]={"sizeof","_Alignof","alignof","_Generic","__real__","__imag__","return","if","else",
    "while","for","do","switch","case","default","break","continue","goto","static","extern","register","auto",
    "inline","typedef","restrict","__restrict","__restrict__","__attribute__","__extension__","asm","__asm__",
    "_Static_assert","static_assert","_Alignas","alignas",0};
  if(t->k!=T_ID || decl_type_tok(c,t)) return 0;
  for(int k=0;kw[k];k++) if(tok_is(t,kw[k])) return 0;
  return 1;
}
/* The end of the postfix operand P at `j` (a name, a string literal -- or, with `cl`, a compound literal -- then
 * its `[..]` `.m` `->m` selectors), -1 when there is none; `*is_name` tells whether it is a name's. */
static int up_operand_end(CC *c, int j, int cl, int *is_name){
  *is_name=up_operand_name(c,tat(c,j));
  if(*is_name) j++;
  else if(tat(c,j)->k==T_STR){ while(tat(c,j)->k==T_STR) j++; }   /* `("abc")[i]`, adjacent pieces included */
  else if(cl && tok_is(tat(c,j),"(") && decl_type_tok(c,tat(c,j+1))){   /* `(T){...}` */
    j=up_group_end(c,j); if(j<0 || !tok_is(tat(c,j),"{")) return -1;
    j=up_group_end(c,j); if(j<0) return -1; }
  else return -1;
  for(;;){
    if(tok_is(tat(c,j),"[")){ j=up_group_end(c,j); if(j<0) return -1; }
    else if((tok_is(tat(c,j),".")||tok_is(tat(c,j),"->")) && tat(c,j+1)->k==T_ID) j+=2;
    else return j;
  }
}
/* The operand P in [a, b) carries a subscript `[..]`. */
static int up_has_subscript(CC *c, int a, int b){
  for(int k=a; k<b; k++) if(tok_is(tat(c,k),"[")) return 1;
  return 0;
}
/* May the `(` about to be written at `w` open a parenthesized operand (the body's first token is at `lo`)? */
static int up_paren_ok(CC *c, int w, int lo, const upframe *f){
  if(f->decl && !f->init && f->xpd<0) return 0;            /* a declarator: `T (x)[N]`, `T *(x)`, `T a, (x)[N]` */
  const tok *p=tat(c,w-1);                                  /* the token before the `(` (the body's `{` first) */
  if(p->k==T_ID) return tok_is(p,"return")||tok_is(p,"else")||tok_is(p,"do");   /* a name: its call, condition or
                                                             * `sizeof`/`typeof` operand */
  if(p->k!=T_PUN || tok_is(p,"]")) return 0;                /* a literal before it, or `a[i](x)`: a call */
  if(!tok_is(p,")")) return 1;
  int q=up_match_open(c,w-1,lo);                            /* after `)`: a condition's, or a cast's */
  if(q<0) return 0;
  const tok *b=tat(c,q-1);
  if(tok_is(b,"if")||tok_is(b,"while")||tok_is(b,"for")||tok_is(b,"switch")) return 1;
  return up_cast_open(c,q);
}
/* Track what the token just written at `w` does to the statement structure (the frames of the braces it is in). */
static void up_track(CC *c, upframe *fr, int *nf, int *deep, int w){
  upframe *f=&fr[*nf-1]; const tok *t=tat(c,w), *pv=tat(c,w-1);
  if(t->k!=T_PUN || t->n!=1) return;
  switch(t->s[0]){
    case '(': case '[':
      if(f->decl && !f->init && f->xpd<0 && (t->s[0]=='[' || tok_is(pv,"typeof")||tok_is(pv,"__typeof__")
         ||tok_is(pv,"typeof_unqual")||tok_is(pv,"sizeof")||tok_is(pv,"_Alignof")||tok_is(pv,"alignof")
         ||tok_is(pv,"_Alignas")||tok_is(pv,"alignas"))) f->xpd=f->pd;   /* an expression inside a declarator */
      f->pd++;
      if(t->s[0]=='(' && f->forhdr){ f->forhdr=0; f->fstart=1; }   /* the for-init follows */
      break;
    case ')': case ']': if(f->pd>0) f->pd--; if(f->pd==f->xpd) f->xpd=-1; break;
    case '{': {
      int blk=1;                                           /* a block (`({` a statement expression's too) -- */
      if(!tok_is(pv,"(")){                                 /* or an initializer's brace list, a compound literal's */
        if(!f->blk || (f->decl && f->init) || tok_is(pv,"=") || tok_is(pv,",")) blk=0;
        else if(tok_is(pv,")")){ int q=up_match_open(c,w-1,0); blk=!(q>=0 && up_cast_open(c,q)); } }
      if(*nf<UP_FRAMES){ upframe *n=&fr[(*nf)++]; memset(n,0,sizeof *n); n->blk=(uint8_t)blk; n->start=(uint8_t)blk; n->xpd=-1; }
      else (*deep)++;
      break; }
    case '}':
      if(*deep){ (*deep)--; break; }
      { int was_blk=f->blk; (*nf)--;
        if(*nf>0 && was_blk){ upframe *o=&fr[*nf-1]; if(o->blk && o->pd==0){ o->start=1; o->decl=0; } } }
      break;
    case ';': if(f->blk && f->pd==0){ f->start=1; f->decl=0; } else if(f->decl && f->pd==f->dpd) f->decl=0; break;
    case '=': if(f->decl && f->pd==f->dpd && f->xpd<0) f->init=1; break;
    case ',': if(f->decl && f->pd==f->dpd) f->init=0; break;
    case ':': if(f->lab && f->pd==0){ f->lab=0; f->start=1; } break;
    default: break;
  }
}
/* `( E ) OP v` where a statement begins (CF-RTFP): E an lvalue of unary `*`s and postfix operators alone -- a name and
 * its `.m`, `->m`, `[i]`, `++`, `--` and calls, parenthesized groups -- and OP an assignment: the parentheses are
 * redundant (C11 6.5.1p5), and the statement forms read E bare, as the oracle reads `(E) = v`. The emit spells an
 * `_Atomic` member store so, `(*(_Atomic T *)((char *)p + K)) = v;`. The index of the `)` closing the `(` at `r`, else
 * -1. */
static int up_paren_lvalue(CC *c, int r){
  int q=up_group_end(c,r)-1;                         /* the `)`: up_group_end gives the token after it */
  if(q<=r+1) return -1;
  const tok *op=tat(c,q+1);
  if(!tok_is(op,"=") && !is_compound_op(op)) return -1;
  int k=r+1;
  while(tok_is(tat(c,k),"*")) k++;
  if(k>=q) return -1;
  while(k<q){
    const tok *t=tat(c,k);
    if(t->k==T_ID || tok_is(t,".") || tok_is(t,"->") || tok_is(t,"++") || tok_is(t,"--")){ k++; continue; }
    if(tok_is(t,"(") || tok_is(t,"[")){ int e=up_group_end(c,k); if(e<0 || e>q) return -1; k=e; continue; }
    return -1;
  }
  return q;
}
/* One pass over the body at `start` (just past its `{`), compacting it in place; 1 when it rewrote something. Of a
 * run of `(`, only the one whose closers directly follow P can open P's parentheses: it is found once per run. */
static int unparen_body(CC *c, int start){
  static const tok lb={T_PUN,"[",1,0}, zero={T_INT,"0",1,0}, rb={T_PUN,"]",1,0}, arrow={T_PUN,"->",2,0};
  upframe fr[UP_FRAMES]; int nf=1, deep=0, np=0, w=start, r=start, changed=0;
  struct { int pos, n, zero; } pend[UP_FRAMES];              /* scheduled closers: dropped, or `[0]` in their place */
  int run_end=start, cand=-1, pe=-1, k=0, star=0, is_name=0;   /* the current run of `(` and its one candidate */
  memset(&fr[0],0,sizeof fr[0]); fr[0].blk=1; fr[0].start=1; fr[0].xpd=-1;
  while(r<c->nt && nf>0){
    if(np && pend[np-1].pos==r){                           /* P's closers */
      if(pend[np-1].zero){ c->t[w++]=lb; c->t[w++]=zero; c->t[w++]=rb; }
      r+=pend[--np].n; continue; }
    tok t=c->t[r]; upframe *f=&fr[nf-1]; int first=0;
    if(f->blk && f->pd==0 && f->start){                    /* a statement starts: a label, or a declaration? */
      f->start=0;
      if(tok_is(&t,"case")||tok_is(&t,"default")||(t.k==T_ID && tok_is(tat(c,r+1),":"))) f->lab=1;
      else { f->decl=(uint8_t)decl_start_tok(c,&t); f->dpd=0; f->init=0; f->xpd=-1; f->forhdr=(uint8_t)tok_is(&t,"for");
        first=!f->decl; } }
    else if(f->fstart){ f->fstart=0; f->decl=(uint8_t)decl_start_tok(c,&t); f->dpd=f->pd; f->init=0; f->xpd=-1; }
    if(first && tok_is(&t,"(") && np<UP_FRAMES && !deep){   /* `( E ) = v;` -> `E = v;` (CF-RTFP) */
      int q=up_paren_lvalue(c,r);
      if(q>0){ pend[np].pos=q; pend[np].n=1; pend[np].zero=0; np++; r++; changed=1; continue; } }
    if(tok_is(&t,"(") && r>=run_end){                      /* a run of `(` starts: P after it, and its closers */
      run_end=r; while(tok_is(tat(c,run_end),"(")) run_end++;
      int ps=run_end;                                      /* P's first token -- the run's last `(` when it opens */
      star=tok_is(tat(c,ps),"*");                          /* a compound literal's type name, `((T){...})[i]` */
      if(!star && decl_type_tok(c,tat(c,ps))) ps--;
      pe=up_operand_end(c,ps+star,!star,&is_name); k=0;
      if(pe>0) while(tok_is(tat(c,pe+k),")")) k++;
      cand = pe>0 && k>=1 && k<=ps-r ? ps-k : -1; }
    if(r==cand && np<UP_FRAMES && !deep && up_paren_ok(c,w,start,f)){
      const tok *op=tat(c,pe+k); int n=k;
      if(!star && (tok_is(op,"[")||tok_is(op,".")||tok_is(op,"->"))){   /* (^n P )^n OP  ->  P OP */
        pend[np].pos=pe; pend[np].n=n; pend[np].zero=0; np++; r+=n; changed=1; continue; }
      if(!star && is_name && tok_is(op,"(") && !up_has_subscript(c,run_end,pe)){   /* (^n P )^n (  ->  P (  : a call
        * through a parenthesized callee, `(fp)(x)`, `(o->fn)(x)` -- the oracle's `fp(x)` / `o->fn(x)` (CF-SPLIT2);
        * `(ops[i])(x)` stays refused, as the oracle's parse refuses it */
        pend[np].pos=pe; pend[np].n=n; pend[np].zero=0; np++; r+=n; changed=1; continue; }
      if(star && is_name && (tok_is(op,".")||tok_is(op,"["))){   /* (^n * P )^n .  ->  P ->  ;  [  ->  P [0] [ */
        int z=tok_is(op,"[");
        if(!z) c->t[pe+n]=arrow;
        pend[np].pos=pe; pend[np].n=n; pend[np].zero=z; np++; r+=n+1; changed=1; continue; }
      if(!star && is_name && (tok_is(op,"++")||tok_is(op,"--"))){   /* (^n P )^n ++  ->  (^n P ++ )^n */
        tok s=c->t[pe+n]; memmove(&c->t[pe+1],&c->t[pe],(size_t)n*sizeof(tok)); c->t[pe]=s; changed=1; }
      else if(!star && is_name && w>start && (tok_is(&c->t[w-1],"++")||tok_is(&c->t[w-1],"--"))
              && !tok_is(op,"(")){                         /* ++ (^n P )^n  ->  (^n ++ P )^n : a prefix step of a
        * parenthesized lvalue, `++(x)`, the expression the oracle reads (CF-STRUCTCOND): the step written at w-1
        * becomes the first `(`, and moves past the others to just before P, which the loop then reads */
        tok pp=c->t[w-1]; c->t[w-1]=c->t[r];
        memmove(&c->t[r],&c->t[r+1],(size_t)(n-1)*sizeof(tok)); c->t[r+n-1]=pp;
        up_track(c,fr,&nf,&deep,w-1); cand=-1; changed=1; continue; }
    }
    c->t[w++]=t; r++;
    up_track(c,fr,&nf,&deep,w-1);
  }
  if(w<r){ memmove(&c->t[w],&c->t[r],(size_t)(c->nt+1-r)*sizeof(tok)); c->nt-=r-w; }
  return changed;
}

/* The byte width of a per-element size operand at token `i` -- `sizeof(type)` or an integer literal --
 * or -1 if it is neither / a malformed sizeof. Does NOT consume (it parses a copy of the cursor). */
static int p_type(CC *c, bcir_ctype *ty, int *sidx);   /* fwd (defined above; re-declared for size_bytes) */
static int rec_size_bytes(CC *c, int i) {
  const tok *t = &c->t[i];
  if (t->k == T_INT) return (int)t->v;
  if (t->k == T_ID && t->n == 6 && !strncmp(t->s, "sizeof", 6) && c->t[i + 1].k == T_PUN
      && c->t[i + 1].n == 1 && c->t[i + 1].s[0] == '(') {
    int save = c->i; c->i = i + 2;                         /* past `sizeof (` -- parse the type-name on a copy */
    int isty = starts_decl_type(c);
    int sz = -1;
    if (isty) { bcir_ctype ty; int si, al; if (!p_type(c, &ty, &si)) type_layout(c, &ty, si, &sz, &al); }
    c->i = save;                                           /* speculative -- never advance the real cursor */
    return sz;
  }
  return -1;
}
/* The token index just past the per-element SIZE operand at `i` -- a `sizeof(...)` balanced group or a single
 * integer literal -- or `i` itself if it is neither (an empty span the caller rejects). Used to confirm the
 * size operand fills its whole side of the `N * sizeof(T)` product (so the OTHER side is the full count). */
static int size_operand_span(CC *c, int i) {
  const tok *t = &c->t[i];
  if (t->k == T_INT) return i + 1;
  if (t->k == T_ID && t->n == 6 && !strncmp(t->s, "sizeof", 6)
      && c->t[i + 1].k == T_PUN && c->t[i + 1].n == 1 && c->t[i + 1].s[0] == '(') {
    int j = i + 1, d = 0;                                   /* walk the balanced `(...)` after sizeof */
    for (; c->t[j].k != T_END; j++) { const tok *u = &c->t[j];
      if (u->k == T_PUN && u->n == 1 && u->s[0] == '(') d++;
      else if (u->k == T_PUN && u->n == 1 && u->s[0] == ')') { d--; if (d == 0) return j + 1; } }
  }
  return i;
}
/* A stable integer-count NAME at token `i` -> its variable rid, else 0. The name must be a bare in-scope
 * integer scalar that is STABLE: assigned at most once and never address-taken (the C twin of
 * _recoverable_alloc.count_rid). */
static uint32_t rec_count_rid(CC *c, int i) {
  const tok *t = &c->t[i];
  if (t->k != T_ID) return 0;
  venv *v = lookup(c, t);                                  /* must be a declared local/param */
  if (!v) return 0;
  if (v->type.kind != 0 || v->type.is_float) return 0;     /* an integer scalar only */
  if (mut_body(c, t) > 0 || mut_addr(c, t)) return 0;       /* not stable: an ordinary (post-alloc) write, or aliased */
  return v->rid;
}
/* §5.12 token-level purity check (the C twin of lower._is_pure): is the value of the expression in the
 * token range [s, e) side-effect-FREE -- so it is safe to RE-EVALUATE for the extent snapshot? Pure iff it
 * is only arithmetic over names / literals / sizeof: no call (`identifier (`, except sizeof/_Alignof/alignof),
 * no assignment / comparison / logical op, no `++`/`--`, and no `*`/`&` used as a deref / address-of. The
 * allowed punctuators are the arithmetic operators (`+ - * / % & | ^ ~ << >>`) and grouping `( )`; a `*`/`&`
 * is binary (allowed) only when the previous token is a value-ender (else it is a unary deref/address-of ->
 * impure). Conservative: anything unrecognized -> impure (stays unmanaged rather than double-run). */
static int is_sizeof_kw(const tok *t) {
  return t->k == T_ID && ((t->n == 6 && !strncmp(t->s, "sizeof", 6))
    || (t->n == 8 && !strncmp(t->s, "_Alignof", 8)) || (t->n == 7 && !strncmp(t->s, "alignof", 7))
    || (t->n == 13 && !strncmp(t->s, "__alignof__", 13)));
}
static int is_pure_range(CC *c, int s, int e) {
  if (e <= s) return 0;
  for (int i = s; i < e; i++) {
    const tok *t = &c->t[i];
    if (t->k == T_INT || t->k == T_FLT || t->k == T_ID) {
      if (t->k == T_ID && !is_sizeof_kw(t)                 /* a call `name (` (sizeof/alignof excepted) */
          && c->t[i + 1].k == T_PUN && c->t[i + 1].n == 1 && c->t[i + 1].s[0] == '(' && i + 1 < e)
        return 0;
      continue;
    }
    if (t->k != T_PUN) return 0;                           /* a string literal etc. -> impure */
    if (t->n == 1) {
      char ch = t->s[0];
      if (ch == '(' || ch == ')' || ch == '+' || ch == '-' || ch == '/' || ch == '%'
          || ch == '|' || ch == '^' || ch == '~') continue;
      if (ch == '*' || ch == '&') {                        /* binary mul/and only when after a value-ender */
        const tok *pv = (i > s) ? &c->t[i - 1] : NULL;
        if (pv && is_value_ender(pv)) continue;
        return 0;                                          /* a unary deref / address-of -> impure */
      }
      return 0;                                            /* `=` `<` `>` `?` `:` `,` ... -> impure */
    }
    if (t->n == 2 && (t->s[0] == '<' || t->s[0] == '>') && t->s[1] == t->s[0]) continue;   /* << >> shifts */
    return 0;                                              /* `==` `&&` `++` `+=` ... -> impure */
  }
  return 1;
}
/* §5.12 _recoverable_alloc: if the call in the token range [start,end) is `calloc(N, sizeof(T))`,
 * `malloc(N*sizeof(T))` / `malloc(sizeof(T)*N)`, or `malloc(N)` for a 1-byte pointee (T the pointee, so
 * N the element COUNT), set the COUNT's token range [*cstart, *cend) and return N's rid when N is a stable
 * integer Name (the fast path), else 0 (the count is an expression the caller may SNAPSHOT, or N is not
 * stable). `*cstart` is set to -1 if the call is not a recoverable alloc form at all. Conservative: any
 * uncertainty about the FORM -> not recognized (no guard, never a false trap). */
static uint32_t recoverable_alloc(CC *c, int start, int end, int pointee_size, int *cstart, int *cend) {
  *cstart = -1; *cend = -1;
  if (pointee_size <= 0) return 0;
  const tok *cal = &c->t[start];
  if (cal->k != T_ID || !(c->t[start + 1].k == T_PUN && c->t[start + 1].n == 1 && c->t[start + 1].s[0] == '('))
    return 0;
  int is_malloc = cal->n == 6 && !strncmp(cal->s, "malloc", 6);
  int is_calloc = cal->n == 6 && !strncmp(cal->s, "calloc", 6);
  if (!is_malloc && !is_calloc) return 0;
  int a0 = start + 2;                                      /* first argument token */
  int close = end - 1;                                     /* end-1 is the call's closing `)` */
  if (is_calloc) {                                         /* calloc(N, sizeof(T)) -- N runs [a0, comma) */
    int j = a0, d = 0, comma = -1;                         /* find the TOP-LEVEL comma separating the two args */
    for (; j < close; j++) { const tok *t = &c->t[j];
      if (t->k == T_PUN && t->n == 1 && (t->s[0] == '(' || t->s[0] == '[')) d++;
      else if (t->k == T_PUN && t->n == 1 && (t->s[0] == ')' || t->s[0] == ']')) d--;
      else if (d == 0 && t->k == T_PUN && t->n == 1 && t->s[0] == ',') { comma = j; break; } }
    if (comma < 0 || comma == a0) return 0;                /* no separator / an empty first arg */
    if (rec_size_bytes(c, comma + 1) != pointee_size) return 0;   /* the second arg must be sizeof(T) / its byte width */
    *cstart = a0; *cend = comma;                           /* the count is the first arg */
    if (comma == a0 + 1 && c->t[a0].k == T_ID) return rec_count_rid(c, a0);   /* a single bare Name -> fast path */
    return 0;                                              /* an expression count -> the caller snapshots */
  }
  /* malloc: the single argument runs [a0, close). Forms: N*S / S*N / N. Find a TOP-LEVEL `*` (depth 0). */
  int j = a0, d = 0, star = -1;
  for (; j < close; j++) { const tok *t = &c->t[j];
    if (t->k == T_PUN && t->n == 1 && (t->s[0] == '(' || t->s[0] == '[')) d++;
    else if (t->k == T_PUN && t->n == 1 && (t->s[0] == ')' || t->s[0] == ']')) d--;
    else if (d == 0 && t->k == T_PUN && t->n == 1 && t->s[0] == '*') { star = j; break; } }
  if (star >= 0) {                                         /* N * S  or  S * N : the `*` splits two operands */
    int lstart = a0, lend = star, rstart = star + 1, rend = close;
    /* exactly ONE side must be the per-element size operand -- `sizeof(T)` (the whole balanced group) or a
     * byte literal -- filling its whole side; the OTHER side is the count (N), which may be an expression. */
    if (rec_size_bytes(c, rstart) == pointee_size && size_operand_span(c, rstart) == rend && lend > lstart) {
      *cstart = lstart; *cend = lend;                      /* N * sizeof(T) : count on the LHS */
    } else if (rec_size_bytes(c, lstart) == pointee_size && size_operand_span(c, lstart) == lend && rend > rstart) {
      *cstart = rstart; *cend = rend;                      /* sizeof(T) * N : count on the RHS */
    } else return 0;                                       /* neither side is the per-element size */
    if (*cend == *cstart + 1 && c->t[*cstart].k == T_ID) return rec_count_rid(c, *cstart);   /* bare Name */
    return 0;                                              /* an expression count -> the caller snapshots */
  }
  /* malloc(N) for a 1-byte pointee: N fills the whole single argument */
  if (pointee_size == 1 && close > a0) {
    *cstart = a0; *cend = close;
    if (close == a0 + 1 && c->t[a0].k == T_ID) return rec_count_rid(c, a0);
    return 0;
  }
  return 0;
}
/* §5.12 snapshot a pure expression COUNT in the token range [cstart, cend): re-lower it (a SECOND
 * evaluation, separate from the malloc arg -- sound because it is pure), copy the value into a fresh hidden
 * IMMUTABLE local `__bcir_extK` (K per-function), and return that snapshot local's rid (the recovered
 * extent), or 0 if the value is not an integer scalar. Mirrors the oracle's _bind_extent snapshot branch. */
static uint32_t snapshot_extent(CC *c, int cstart, int cend) {
  int save = c->i;
  tok stash = c->t[cend];                                  /* terminate the re-lowering exactly at cend */
  c->t[cend].k = T_END; c->t[cend].s = ""; c->t[cend].n = 0;
  c->i = cstart;
  uint32_t v = p_expr(c);                                  /* re-evaluate the count (a SECOND lowering) */
  c->t[cend] = stash; c->i = save;                         /* restore the token + the real cursor */
  const bcir_resource *vr = res_of(c->fn, v);
  if (!vr || vr->kind != BCIR_RK_SCALAR || vr->is_float) return 0;   /* an integer scalar only */
  int vbytes = (int)vr->elem_bytes, vsigned = vr->is_signed;         /* capture before add_res may realloc res[] */
  char nm[BCIR_CIR_NAME]; snprintf(nm, sizeof nm, "__bcir_ext%d", c->ext_ctr++);
  uint32_t ext = add_res(c, BCIR_DOM_RAM, vbytes, 1, 0, BCIR_RK_SCALAR, nm);
  if (c->fn->n_res) c->fn->res[c->fn->n_res - 1].is_signed = (uint8_t)vsigned;   /* mirror the value's signedness */
  bcir_claim *cp = new_claim(c, "c.copy", BCIR_OP_ADD);
  if (cp) { cp->n_rd = 1; cp->rd[0] = v; cp->n_wr = 1; cp->wr[0] = ext; }
  return ext;
}
/* §5.12 _bind_extent: bind a recovered element-count to a malloc/calloc'd pointer local (rid `p_rid`,
 * the resource `pr`, name `p_name`), so its `p[i]` accesses promote to `masked`. Only when p is a POINTER
 * and STABLE -- assigned exactly once (this binding) and never address-taken -- so it still points at that
 * allocation at every access (a `p = realloc(...)` reassigns it, count 2, left unmanaged). The init call
 * is the token range [init_start, init_end). A `void *` takes none: no access through it is subscripted or
 * dereferenced (C11 6.5.2.1p1, 6.5.3.2p4), so there is nothing to bound, and the oracle reads `void` as size 0
 * (`_alloc_count_node`) -- the twin had snapshotted `malloc(4u)`'s 4 as a byte count, two claims nothing read, while
 * R21 reads the allocation's lifetime events alone, which both rails record (CF-STRUCTCOND). */
static void bind_extent(CC *c, uint32_t p_rid, const bcir_resource *pr, const tok *p_name,
                        int init_start, int init_end) {
  if (!pr || pr->kind != BCIR_RK_POINTER || pr->is_voidptr) return;
  if (mut_assigned(c, p_name) != 1 || mut_addr(c, p_name)) return;
  int pointee = (int)pr->elem_bytes;                        /* the pointee element size (`p_ct.of.size`) */
  int cstart, cend;
  uint32_t n_rid = recoverable_alloc(c, init_start, init_end, pointee, &cstart, &cend);
  if (n_rid) { ptrext_set(c,c->fn, p_rid, n_rid); return; }   /* a STABLE integer-count Name -> bound BY NAME */
  if (cstart < 0) return;                                   /* not a recoverable alloc form at all */
  if (cend == cstart + 1 && c->t[cstart].k == T_ID) return; /* a (non-stable) bare Name -> by-name-or-nothing */
  if (!is_pure_range(c, cstart, cend)) return;              /* an impure expression count -> unmanaged */
  uint32_t ext = snapshot_extent(c, cstart, cend);          /* a pure EXPRESSION count -> SNAPSHOT it */
  if (ext) ptrext_set(c,c->fn, p_rid, ext);
}

/* The bounds contract for an indexed access (§5.12 bounds-promotion). A LOCAL/STATIC array OBJECT -- whose
 * extent is statically RECOVERABLE from the resource's element `count` -- is promoted from `assumed` to
 * `masked` (runtime-bounds-checked, the contract the quarantine handler discharges); a pointer base (extent
 * unknown) stays `assumed` -- UNLESS it carries a §5.12 recovered count (ptr_extent). Metadata only -- no
 * emit/behaviour change; `verify` already defaults to bounds. */
static bcir_bounds access_bnd(CC *c, uint32_t rid) {
  const bcir_resource *r = res_of(c->fn, rid);
  if (r && r->domain == BCIR_DOM_MMIO) return BCIR_BND_ASSUMED;   /* a device region (a register block, a
                                                                * volatile array): trusted, like the oracle */
  /* A known-extent ARRAY object (oracle `rt.kind=="array" and rt.count`): a LOCAL/STATIC array (kind !=
   * POINTER), OR a file-scope (static/const) global array -- the twin tags a global array POINTER for
   * index decay, but it carries its real, small element count, whereas a genuine pointer has the SYMBOLIC
   * pointee extent 1<<16 (locals/params) or count 1 (a pointer global / a malloc result). So an array is
   * any base with a small definite count > 1; the pointer extents (65536 / 1) are excluded here and the
   * recovered ones are masked via ptr_extent below. A string-LITERAL base stays assumed_safe (anonymous
   * read-only data -- the oracle excludes str_globals). */
  if (r && decl_array(r) && r->count != (1u << 16) && !strtab_lookup(c->fn,rid))
    return BCIR_BND_MASKED;                                    /* one element included: `T a[1]` is an array */
  if (ptrext_get(c->fn, rid)) return BCIR_BND_MASKED;   /* §5.12 a malloc/calloc pointer with a recovered count */
  return BCIR_BND_ASSUMED;
}
/* The temp an element of `base` loads into: an ARRAY of pointers yields a pointer typed as the element,
 * anything else a value of the element's type. Shared by `a[i]` and `*a`. */
/* A temp holding a function-pointer VALUE whose alias is `alias` (CF-FPTAB): pointer-wide on the target and spelled by
 * the alias, so the emit declares `op_t t` and a call through it compiles -- a `uint64_t` of it did not, and was 8
 * bytes on a 4-byte-pointer target. `res_sig` finds its function type by the alias, as a select's (CF-FNSEL). */
static uint32_t fp_value_temp(CC *c, const char *alias){
  uint32_t t=add_res(c,BCIR_DOM_RAM,cc_abi(c)->pointer_size,1,0,BCIR_RK_SCALAR,"");
  if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1]; tr->is_funcptr=1; fits(c,tr->agg,BCIR_CIR_AGG,"%s",alias); }
  return t;
}
static uint32_t elem_temp(CC *c, venv *base) {
  uint32_t t;
  if(base->type.kind==3 || (base->type.kind==2 && base->type.ptr_to_fp && (base->type.ptr_depth?base->type.ptr_depth:1)==1
                            && !ptr_array(c,base))){
    /* an element of a table of function pointers -- an array of them, or through a pointer to one: a function-pointer
     * value of the table's type (CF-FPTAB) */
    t=fp_value_temp(c,base->type.tag);
  } else if(ptr_array(c,base)){                          /* an ARRAY of pointers `T *a[N]`
    * (a SCALAR array with pointer-wide elements, NOT a pointer variable `T *p`): `a[i]` loads a pointer, typed
    * as the element (its pointee's width, sign and volatility), so indexing it lands on the right objects */
    t=add_res(c,BCIR_DOM_RAM,base->type.size?base->type.size:4,1,0,BCIR_RK_POINTER,"");
    if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
      tr->ptr_depth=base->type.ptr_depth?base->type.ptr_depth:1;
      tr->is_signed=(uint8_t)(base->type.signd?1:0); tr->is_float=(uint8_t)(base->type.is_float?1:0);
      tr->is_plain_char=(uint8_t)(base->type.is_plain_char?1:0);
      tr->is_atomic=(uint8_t)(base->type.is_atomic?1:0);   /* a pointer to `_Atomic` storage (CF-ATOMIC) */
      if(base->type.ptr_to_struct) fits(c,tr->agg,sizeof tr->agg,"%s %s",base->type.is_union?"union":"struct",base->type.tag);
      else if(base->type.ptr_to_fp) ptee_fp(c,tr,&base->type);
      else if(base->type.size==0 && !base->type.is_float) tr->is_voidptr=1; }   /* a `void *` element */
    last_ptr_to_volatile(c,base->type.is_volatile,base->type.ptr_to_struct && sdef_vol(c,base->sidx));
  } else if(base->type.kind==2 && (base->type.ptr_depth?base->type.ptr_depth:1)>1){   /* `pp[i]` through a
    * `T **`: the element is itself a pointer, one level down (the oracle's `base_ct.of`), never a `T` value */
    bcir_ctype et=base->type; et.ptr_depth=(uint8_t)((base->type.ptr_depth?base->type.ptr_depth:1)-1);
    t=temp_ptr(c,&et,base->sidx);
  } else if(index_elem_sidx(c,base)>=0){           /* an element of an array of structs, or through a pointer to
    * one (`ps[i]`, `pp[i]`, `*gp`): the struct/union value itself (CF-STRUCTVAL) */
    t=tempagg(c,index_elem_sidx(c,base));
  } else { int es=base->type.size?base->type.size:4;
    t=base->type.is_float ? tempf(c,es) : tempi(c,es,base->type.signd);   /* float -> a float temp; else keep the sign */
    if(base->type.is_bool && index_elem_atomic(c,base) && c->fn->n_res)   /* an `_Atomic _Bool` element */
      c->fn->res[c->fn->n_res-1].is_bool=1; }
  return t;
}
static const bcir_ctype *callee_ret(CC *c, const tok *id);     /* fwd: a defined function's return type */
static const bcir_ctype *declared_ret(CC *c, const tok *id);   /* fwd: a prototype's */
/* The operand kinds C requires of `*`, `[]`, `.` and `->` (C11 6.5.3.2p2, 6.5.2.1p1, 6.5.2.3p1-2) and of what
 * `sizeof` measures (6.5.3.4p1), each wrong operand refused for one reason on both rails -- the oracle's
 * `DEREF_NOT`, `SUBSCRIPT_NOT`, `DOT_NOT`, `ARROW_NOT`, `FN_NOT_LVALUE` and `SIZEOF_FN` (CF-FPTAB). */
#define CC_DEREF_NOT "dereference of a non-pointer"
#define CC_SUBSCRIPT_NOT "subscripted value is not an array or a pointer to an object"
#define CC_DOT_NOT "member access on an object that is not a struct or union"
#define CC_ARROW_NOT "member access `->` through a value that is not a pointer to a struct or union"
#define CC_FN_NOT_LVALUE "a function designator is not an lvalue"
#define CC_SIZEOF_FN "sizeof of a function designator"
/* Whether the object `v` names is an address `*` or `[]` takes (C11 6.5.3.2p2, 6.5.2.1p1): a pointer to an object --
 * a declared one, or a file-scope one whose slot is a scalar -- or an array of any length, a VLA's too. A function
 * pointer is none (a `*` of one names the function), nor is any other object. Every fast path that reads or writes
 * through a NAMED operand asks this, as `emit_deref_rid` asks it of a value (CF-FPTAB: `*s = 1u`, `*(s + 1u)` and
 * `s[1]` of an integer lowered on both rails). The oracle's `_indexable`. */
static int names_object_ptr(CC *c, const venv *v){
  if(v->type.kind==2) return 1;
  const bcir_resource *r=res_of(c->fn,v->rid);
  return r && (r->kind==BCIR_RK_POINTER || r->is_array || r->is_vla || r->is_pointer);
}
/* The refusal a `*` of the name `nm` -- its object `v`, NULL for none -- gets as an assignment's or an increment's
 * operand (CF-FPTAB): a function pointer or a function names a function, no object (C11 6.3.2.1p1); an object that is
 * neither a pointer nor an array, no address (6.5.3.2p2). 1 after failing for the oracle's reason, else 0. */
static int deref_lvalue_refused(CC *c, const venv *v, const tok *nm){
  if(v ? (v->type.kind==3 && !names_object_ptr(c,v)) : (callee_ret(c,nm) || declared_ret(c,nm))){   /* (a table's */
    fail(c,CC_FN_NOT_LVALUE); return 1; }                                   /* `*t` is its element, an object) */
  if(v && !names_object_ptr(c,v)){ fail(c,CC_DEREF_NOT); return 1; }
  return 0;
}
/* Whether the token at `k` assigns (`=`, a compound `OP=`): what follows an lvalue a store writes. */
static int assign_tok_at(CC *c, int k){ const tok *t=tat(c,k); return tok_is(t,"=") || is_compound_op(t); }
/* Whether `v` is the base its member access takes (C11 6.5.2.3p1-2): `.` a struct or union object, `->` a pointer to
 * one or an array of them, which converts to one (CF-FPTAB: the twin read `s->x` of a struct as `s.x` and `p.x` of a
 * pointer as `p->x`, where the oracle refused both for a third reason). The oracle's `_member_agg`. */
static int member_base_ok(CC *c, const venv *v, int arrow){
  if(v->sidx<0) return 0;
  const bcir_resource *r=res_of(c->fn,v->rid);
  int arr = r && (r->is_array || r->is_vla);
  if(arrow) return v->type.nadims<=1 && ((v->type.kind==2 && v->type.ptr_to_struct && (v->type.ptr_depth?v->type.ptr_depth:1)==1)
                                         || (v->type.kind==1 && arr));
  return v->type.kind==1 && !arr;
}
static uint32_t emit_index(CC *c, venv *base, uint32_t idx) {     /* base[idx] -- GEP load */
  uint32_t t=elem_temp(c,base);
  bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
  cl->n_rd=2;cl->rd[0]=base->rid;cl->rd[1]=idx;cl->n_wr=1;cl->wr[0]=t;cl->bounds=access_bnd(c,base->rid);
  mark_access(c,cl,index_elem_vol(c,base));        /* the element of a pointer to volatile / a volatile array */
  mark_atomic(cl,index_elem_atomic(c,base));       /* ... and of a pointer to / an array of `_Atomic` */
  return t;
}
/* Parse up to `maxd` subscripts `[i][j][k]` on an array variable and Horner-flatten via its declared dims
 * (`v->type.adims`): `m[i][j]` on a `T m[A][B]` -> `i*B + j`. The cursor must be at the first `[`. */
static uint32_t array_index_n(CC *c, venv *v, int maxd) {
  /* SNAPSHOT the env entry: the index p_expr below can declare locals (a `({...})` stmt-expr) and realloc
   * c->env[] -- the incoming `v`, a pointer into that array, would dangle after the move. Reading from the
   * by-value copy is byte-identical (only v->rid / v->type are read). */
  venv vsnap=*v; v=&vsnap;
  if(!names_object_ptr(c,v)){ fail(c,CC_SUBSCRIPT_NOT); return 0; }   /* `s[1]` of an integer, `fp[0]` (CF-FPTAB) */
  uint32_t idxs[3]; int ni=0, taken=0;
  while(taken<maxd && is(c,"[")){ c->i++; uint32_t ix=p_expr(c); eat(c,"]"); if(ni<3)idxs[ni++]=ix; taken++; }
  const bcir_resource *vr=res_of(c->fn,v->rid);    /* a multi-dim VLA -> RUNTIME dim strides (no c.const) */
  int vla=(vr && vr->vla_ndims>0);
  uint32_t lin = ni?idxs[0]:0;
  for(int d=1; d<ni; d++){
    uint32_t k;
    if(vla){ k = (d<(int)vr->vla_ndims) ? vr->vla_strides[d] : 0; }   /* dim d's snapshot rid -- no const */
    else { int dim = d<v->type.nadims ? v->type.adims[d] : 1;
      k=temp(c,4); bcir_claim *kc=new_claim(c,"c.const",BCIR_OP_LOAD);
      if(kc){kc->n_wr=1;kc->wr[0]=k;kc->n_imm=1;kc->imm[0]=dim;} }
    uint32_t m1=temp(c,4); bcir_claim *mc=new_claim(c,"c.bin.mul",BCIR_OP_MUL);
    if(mc){mc->n_rd=2;mc->rd[0]=lin;mc->rd[1]=k;mc->n_wr=1;mc->wr[0]=m1;}
    uint32_t a1=temp(c,4); bcir_claim *ac=new_claim(c,"c.bin.add",BCIR_OP_ADD);
    if(ac){ac->n_rd=2;ac->rd[0]=m1;ac->rd[1]=idxs[d];ac->n_wr=1;ac->wr[0]=a1;}
    lin=a1;
  }
  return lin;
}
static uint32_t array_index(CC *c, venv *v) { return array_index_n(c,v,INT_MAX); }   /* every subscript */
/* The consecutive balanced `[...]` groups starting at token `k`. */
static int sub_group_count(CC *c, int k){
  int n=0;
  while(tok_is(tat(c,k),"[")){
    int d=0;
    do{ if(tok_is(tat(c,k),"[")) d++; else if(tok_is(tat(c,k),"]")) d--; k++; }while(d>0 && tat(c,k)->k!=T_END);
    n++;
  }
  return n;
}
/* The subscripts `v` itself takes: one per dimension of a declared multi-dimensional array (a multi-dim VLA's
 * too), else one. */
static int subscript_dims(CC *c,const venv *v){
  const bcir_resource *vr=res_of(c->fn,v->rid);
  if(vr && vr->vla_ndims>0) return vr->vla_ndims;
  return v->type.nadims>1 ? v->type.nadims : 1;
}
/* A subscript chain `[i]...` on `*v` (cursor at the first `[`) -- the oracle's `_lvalue(Index)`: the base takes
 * its own subscripts (Horner-flattened); while more remain its element is a pointer -- `q[j][i]` on `T *q[N]`,
 * `pp[j][i]` on `T **pp` -- which is loaded, and the rest index what it holds. Flattening them all into the base
 * read `q[j + i]`. On return `*v` is the base the last subscript indexes (the variable, or the loaded pointer's
 * temp) and the result is its linear index. */
/* Load the pointer element `v[lin]` -- of an array of pointers, or through a pointer to pointers -- and make `*v`
 * the loaded pointer (an array's element keeps its depth; through a pointer it is one level down). 0 when the
 * element is not a pointer. The one step a subscript chain and `*(q[j] ...)` both take. */
static int step_to_elem_ptr(CC *c, venv *v, uint32_t lin){
  int arr=ptr_array(c,v), depth=v->type.ptr_depth?v->type.ptr_depth:1;
  if(v->type.kind!=2 || (!arr && depth<2)) return 0;
  uint32_t p=emit_index(c,v,lin);                     /* the pointer element, loaded */
  venv nv=*v; nv.rid=p; nv.type.nadims=0; memset(nv.type.adims,0,sizeof nv.type.adims);
  if(!arr) nv.type.ptr_depth=(uint8_t)(depth-1);
  *v=nv; return 1;
}
/* The dimensions of the file-scope array `v` names (0: not a global, or not an array). */
static int global_ndims(CC *c, const venv *v){
  const bcir_resource *r=res_of(c->fn,v->rid);
  if(!r || !r->read_only || !r->name[0]) return 0;
  int gi=find_global(c,r->name,(int)strlen(r->name));
  return gi>=0 && c->gv[gi].is_arr ? c->gv[gi].nd : 0;
}
static uint32_t index_chain(CC *c, venv *v){
  if(global_ndims(c,v)>1 && !(tok_is(tat(c,c->i),"[") && sub_group_count(c,c->i)>1)){
    /* a row of a multi-dimensional global (`g[i]` of `T g[A][B]`) used as a value: refused, as the oracle does */
    fail(c,"partial indexing of a multi-dimensional array (a row used as a value) is not yet supported"); return 0; }
  for(;;){
    int want=subscript_dims(c,v);
    if(want>1 && sub_group_count(c,c->i)<want){    /* fewer subscripts than dimensions: a ROW, whose value
                                                      * decays to a `T (*)[N]` the value model cannot spell --
                                                      * indexing the flat storage read an element (CF-DECAY) */
      fail(c,"partial indexing of a multi-dimensional array (a row used as a value) is not yet supported"); return 0; }
    uint32_t lin=array_index_n(c,v,want);
    if(c->failed || !is(c,"[")) return lin;
    if(!step_to_elem_ptr(c,v,lin)){ fail(c,"a subscript of an element that is not a pointer"); return lin; }
  }
}
/* A multi-dimensional file-scope array of scalars indexed in full (`g[i][j]` of `T g[A][B]`): the whole object as
 * a member array at offset 0 (the oracle's `whole` index) -- row-major, then an access at a byte offset, which
 * the global's own nested declaration does not care about (CF-SMALL). The cursor is at the first `[`. */
static int global_md_field(CC *c, const venv *v, field *gf){
  const bcir_resource *r=res_of(c->fn,v->rid);
  if(!r || !r->read_only || !r->name[0] || (v->type.kind!=0 && v->type.kind!=3)) return 0;   /* scalars, and function
                                                       * pointers: `g[i][j](x)` of a 2-D table (CF-FPTAB) */
  int gi=find_global(c,r->name,(int)strlen(r->name));
  if(gi<0 || !c->gv[gi].is_arr || c->gv[gi].nd<2 || c->gv[gi].nd>3) return 0;
  const gvar *g=&c->gv[gi];
  if(sub_group_count(c,c->i)!=g->nd) return 0;
  memset(gf,0,sizeof *gf); snprintf(gf->name,sizeof gf->name,"%s",g->name);
  gf->sidx=-1; gf->elem_sidx=-1; gf->ptee_sidx=-1;
  gf->size=v->type.size; gf->signd=v->type.signd; gf->is_float=v->type.is_float; gf->is_complex=v->type.is_complex;
  gf->is_bool=v->type.is_bool; gf->is_plain_char=v->type.is_plain_char; gf->bit_width=v->type.bit_width;
  gf->is_volatile=v->type.is_volatile; gf->is_atomic=v->type.is_atomic; gf->access_bytes=gf->size;
  gf->fp_sig=v->type.kind==3 ? v->type.fp_sig : 0;   /* an element read as the function pointer it is */
  gf->nadims=g->nd; gf->arr_count=1;
  for(int d=0; d<g->nd; d++){ gf->adims[d]=(int)g->dims[d]; gf->arr_count*=(int)g->dims[d]; }
  return 1;
}
/* `a[i].m[j]` on a direct array-of-structs `a` (the element index `lin` lowered, the cursor at `.`): the member
 * array `m` of scalars -- reached through any depth of nested struct/union members, `a[i].s.m[j]` (CF-NESTMEM),
 * which only add their offsets -- its element index folded with the element's: `lin*K + j`, K the element
 * struct's size in `m`'s elements (the oracle's `aos_idx`). 1 with `af` (the member at its offset in the element)
 * and the folded index; 0 when the form is not this one (the cursor unmoved); -1 on a refusal. */
static int aos_member_array(CC *c, venv *v, uint32_t lin, field *af, uint32_t *total){
  if(v->sidx<0 || !is(c,".")) return 0;
  if(!index_member_ok(c,v)) return -1;                 /* an element that is a pointer (CF-IDXARROW) */
  const sdef *ES=&c->s[v->sidx], *S=ES;               /* the element struct; the aggregate a hop names into */
  int j=c->i, off=0, vol=0; field m;
  for(;;){                                             /* peek the `.` chain down to a member array's `[` */
    const tok *fn=tat(c,j+1); int fi=-1;
    if(!tok_is(tat(c,j),".") || fn->k!=T_ID) return 0;
    for(int k=0;k<S->nf;k++) if((int)strlen(S->f[k].name)==fn->n && !strncmp(S->f[k].name,fn->s,(size_t)fn->n)){ fi=k; break; }
    if(fi<0) return 0;
    m=S->f[fi];
    if(m.arr_count && tok_is(tat(c,j+2),"[")) break;
    if(m.sidx<0 || !tok_is(tat(c,j+2),".")) return 0;  /* a scalar member: `a[i].f` (aos_elem_field) */
    off+=m.byte_off; vol|=m.is_volatile; S=&c->s[m.sidx]; j+=2;
  }
  m.byte_off+=off; if(vol) m.is_volatile=1;           /* the member array at its offset in the element */
  int nd=m.nadims>0?m.nadims:1;
  if(m.elem_sidx>=0 || m.elem_ptr || nd>3){ fail(c,"a subscript of an array-of-structs member array is not supported"); return -1; }
  if(sub_group_count(c,j+2)!=nd){ fail(c,"partial indexing of a struct member array is not yet supported"); return -1; }
  if(m.size<=0 || ES->size%m.size){ fail(c,"an array-of-structs member array whose element does not divide the struct"); return -1; }
  c->i=j+2;                                            /* `.s.m` */
  uint32_t lm=member_arr_index(c,&m);
  if(c->failed) return -1;
  uint32_t k=temp(c,4); bcir_claim *kc=new_claim(c,"c.const",BCIR_OP_LOAD);
  if(kc){kc->n_wr=1;kc->wr[0]=k;kc->n_imm=1;kc->imm[0]=ES->size/m.size;}
  uint32_t m1=temp(c,4); bcir_claim *mc=new_claim(c,"c.bin.mul",BCIR_OP_MUL);
  if(mc){mc->n_rd=2;mc->rd[0]=lin;mc->rd[1]=k;mc->n_wr=1;mc->wr[0]=m1;}
  uint32_t a1=temp(c,4); bcir_claim *ac=new_claim(c,"c.bin.add",BCIR_OP_ADD);
  if(ac){ac->n_rd=2;ac->rd[0]=m1;ac->rd[1]=lm;ac->n_wr=1;ac->wr[0]=a1;}
  *af=m; *total=a1; return 1;
}
/* `&base.m[idx]` for a member-array (or whole-object) element `mf` at its byte offset: a `T *` (c.addrof). */
static uint32_t addr_member_elem(CC *c, const venv *v, const field *mf, uint32_t ix){
  int es=mf->size?mf->size:4;
  uint32_t t=add_res(c,BCIR_DOM_RAM, es, 1,0,BCIR_RK_POINTER,"");
  if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
    tr->is_signed=(uint8_t)(mf->signd?1:0); tr->is_float=(uint8_t)(mf->is_float?1:0); tr->ptr_depth=1;
    tr->is_atomic=(uint8_t)(mf->is_atomic?1:0); tr->is_complex=(uint8_t)(mf->is_complex?1:0);
    tr->is_bool=(uint8_t)(mf->is_bool?1:0); tr->is_plain_char=(uint8_t)(mf->is_plain_char?1:0); }
  bcir_claim *cl=new_claim(c,"c.addrof",BCIR_OP_ADD);
  if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=ix;cl->n_wr=1;cl->wr[0]=t;cl->n_imm=2;cl->imm[0]=mf->byte_off;cl->imm[1]=es;}
  return t;
}
static void compound_binop(char ch, const char **suf, bcir_opcode *oc);
static uint32_t binop_result(CC *c, const char *suf, uint32_t a, uint32_t b);
/* The statement `<element> = e;` / `<element> OP= e;` at the cursor, for the member-array (or whole-object) element
 * `f[idx]` of `v`: a compound loads, operates and stores back; an `_Atomic` element is one atomic read-modify-write
 * -- the member-array statement path's shape. 0 when neither follows (the caller re-parses an expression). */
static int store_member_elem_stmt(CC *c, venv *v, const field *f, uint32_t idx){
  uint32_t aval;
  if(f->is_atomic && is_compound_op(&c->t[c->i])){
    char ch=c->t[c->i].s[0]; c->i++; uint32_t rhs=p_expr(c); bcir_ctype ft=field_slot(f);
    const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc); (void)oc;
    (void)emit_rmw(c,suf,&ft,v->rid,1,idx,1,rhs,f->byte_off,f->size,0,f->is_volatile||v->type.is_volatile,
                   BCIR_BND_ASSUMED);
    return 1; }
  if(is_compound_op(&c->t[c->i])){
    char ch=c->t[c->i].s[0]; c->i++;
    uint32_t cur=emit_member_index(c,v,f,idx); uint32_t rhs=p_expr(c);
    const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
    uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.bin.%s",suf);
    bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
    aval=tmp; }
  else if(is(c,"=")){ c->i++; aval=p_expr(c); }
  else return 0;
  (void)store_member_index(c,v,f,idx,0,f,aval);
  return 1;
}
/* The statement `a[i].f = e;` / `a[i].f OP= e;` at the cursor, for the element field `sub` (at its flattened offset
 * in the element) of the array-of-structs base `v`, its element index `idx` resolved once: a store strided by the
 * element; a compound loads, operates and stores back; an `_Atomic` field is one atomic read-modify-write. 0 when
 * neither follows (`a[i].f++;` -- the caller re-parses an expression). */
static int store_elem_field_stmt(CC *c, venv *v, uint32_t idx, const field *sub){
  uint32_t aval;
  if(sub->is_atomic && is_compound_op(&c->t[c->i])){   /* an `_Atomic` field: one atomic read-modify-write */
    char ch=c->t[c->i].s[0]; c->i++; uint32_t rhs=p_expr(c); bcir_ctype ft=field_slot(sub);
    const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc); (void)oc;
    (void)emit_rmw(c,suf,&ft,v->rid,1,idx,1,rhs,sub->byte_off,sub->size,v->type.size,
                   sub->is_volatile||v->type.is_volatile,BCIR_BND_ASSUMED);
    return 1; }
  if(is_compound_op(&c->t[c->i])){                    /* a[i].f OP= expr -> strided load, bin op, strided store */
    char ch=c->t[c->i].s[0]; c->i++;
    uint32_t cur=emit_index_field(c,v,idx,sub); uint32_t rhs=p_expr(c);
    const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
    uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.bin.%s",suf);
    bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
    aval=tmp; }
  else if(is(c,"=")){ c->i++; aval=p_expr(c); }
  else return 0;
  (void)store_index_field(c,v,idx,sub,aval);
  return 1;
}
/* A loaded pointer to structs (`ptr`, its element struct `psidx`, the pointer field `pfld` it came from) as the
 * array-of-structs base a subscript of it is -- `s->p[i].f` strides by the element, as `a[i].f` does. */
static venv ptr_elem_base(CC *c, uint32_t ptr, int psidx, const field *pfld){
  venv b; memset(&b,0,sizeof b); b.rid=ptr; b.sidx=psidx; b.type.kind=2; b.type.ptr_to_struct=1;
  b.type.size=c->s[psidx].size;
  b.type.ptr_depth=(uint8_t)(pfld->ptee_depth>1 ? pfld->ptee_depth : 1);   /* `T **m`: its element is a pointer
                                                       * (`index_member_ok`, CF-IDXARROW) */
  b.type.is_volatile=(uint8_t)(pfld->ptee_volatile?1:0);   /* a pointer to a volatile struct */
  return b;
}
/* Dereference a pointer RVALUE (rid). Depth-aware: `*pp` where pp is `T**` (depth 2) loads a pointer
 * (pointer_size bytes) into a `T*` temp (depth 1); `*p` where p is `T*` loads the base scalar. The
 * general form powers `**pp` (deref the result of `*pp`) and `*(<expr>)`. */
static uint32_t emit_deref_rid(CC *c, uint32_t rid) {
  const bcir_resource *r=res_of(c->fn,rid);
  if(r && r->is_funcptr && r->kind==BCIR_RK_SCALAR) return rid;   /* `*fp`, `*inc`: the function, which converts back
    * to the pointer to it -- no read (C11 6.5.3.2p4, 6.3.2.1p4; CF-FPTAB) */
  if(!r || r->kind!=BCIR_RK_POINTER){ fail(c,CC_DEREF_NOT); return rid; }
  int depth=r->ptr_depth?r->ptr_depth:1, base=r->elem_bytes?(int)r->elem_bytes:4;
  /* SNAPSHOT every field of `r` we still need: the add_res/tempi/tempf calls below allocate a
   * NEW resource, which may realloc (and thus MOVE+free) c->fn->res -- so `r`, a pointer INTO that
   * array, dangles after the first allocation. Reading through it afterward is a use-after-free. */
  uint8_t r_signd=r->is_signed, r_float=r->is_float, r_plain_char=r->is_plain_char, r_vol=r->is_volatile;
  uint8_t r_atom=r->is_atomic, r_bool=r->is_bool, r_fp=r->ptee_funcptr;
  bcir_domain r_dom=r->domain;
  char r_agg[sizeof r->agg]; snprintf(r_agg,sizeof r_agg,"%s",r->agg);
  int r_si=r_fp ? -1 : agg_sidx(c,r_agg);          /* the pointee struct/union (a pointer's `agg`), else -1 */
  uint32_t t;
  if(depth==1 && r_fp) t=fp_value_temp(c,r_agg);   /* `*p` of `op_t *p`: a function-pointer value (CF-FPTAB) */
  else if(depth==1 && r_si>=0) t=tempagg(c,r_si);  /* `*p` of a pointer to a struct: its value (CF-STRUCTVAL) */
  else if(depth>1){                                /* the pointee is itself a pointer (read pointer_size) */
    t=add_res(c,r_dom,base,1,r_vol,BCIR_RK_POINTER,"");   /* still pointing at the same (device) storage */
    if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
      tr->is_signed=r_signd; tr->is_float=r_float; tr->ptr_depth=(uint8_t)(depth-1);
      tr->is_atomic=r_atom;                          /* one level down, still pointing at `_Atomic` storage */
      tr->ptee_funcptr=r_fp;                         /* ... and at function pointers */
      fits(c,tr->agg,sizeof tr->agg,"%s",r_agg); }
  } else { t = r_float ? tempf(c,base) : tempi(c,base,r_signd);
    if(depth==1 && r_plain_char && c->fn->n_res)   /* a `char *` deref loads a plain `char` value */
      c->fn->res[c->fn->n_res-1].is_plain_char=1;
    if(depth==1 && r_atom && r_bool && c->fn->n_res)   /* through `_Atomic _Bool *`: a `_Bool` value */
      c->fn->res[c->fn->n_res-1].is_bool=1; }
  int rd_sz = depth>1 ? cc_abi(c)->pointer_size : base;
  bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
  cl->n_rd=1;cl->rd[0]=rid;cl->n_wr=1;cl->wr[0]=t;cl->bounds=BCIR_BND_ASSUMED;cl->n_imm=2;cl->imm[0]=0;cl->imm[1]=rd_sz;
  mark_access(c,cl,depth==1 && r_vol);             /* the pointee of a pointer to volatile, at its own width */
  mark_atomic(cl,depth==1 && r_atom);              /* the pointee of a pointer to `_Atomic`: an atomic load */
  return t;
}
static uint32_t emit_deref(CC *c, venv *pv) {     /* *p -- a one-read dereference load (named pointer or array) */
  const bcir_resource *r=res_of(c->fn,pv->rid);
  if(pv->type.kind==3 && !(r && (r->is_array || r->ptee_funcptr))) return named_read(c,pv);   /* `*fp`: the
    * function, which converts back to the pointer (C11 6.5.3.2p4) -- the object's value, never a read through it */
  if(r && r->kind!=BCIR_RK_POINTER && r->is_array && pv->type.kind!=1){   /* `*a` on an ARRAY: its first element,
    * one load at offset 0 (the oracle's `mem` lvalue of the element type); volatile when the element is */
    int es = pv->type.kind==2 ? cc_abi(c)->pointer_size : (pv->type.size?pv->type.size:4);
    uint32_t t=elem_temp(c,pv);
    bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
    cl->n_rd=1;cl->rd[0]=pv->rid;cl->n_wr=1;cl->wr[0]=t;cl->bounds=BCIR_BND_ASSUMED;cl->n_imm=2;cl->imm[0]=0;cl->imm[1]=es;
    mark_access(c,cl,index_elem_vol(c,pv));
    mark_atomic(cl,index_elem_atomic(c,pv));
    return t;
  }
  if(r && r->is_pointer && pv->type.kind==2 && (pv->type.ptr_depth?pv->type.ptr_depth:1)==1){   /* `*gp`: a
    * file-scope pointer's slot is modelled as a scalar; the load goes THROUGH the value it holds, typed as the
    * pointee (the oracle's `mem` lvalue at offset 0), volatile when the pointee is */
    int es=pv->type.size?pv->type.size:4;
    uint32_t t=elem_temp(c,pv);
    bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
    cl->n_rd=1;cl->rd[0]=pv->rid;cl->n_wr=1;cl->wr[0]=t;cl->bounds=BCIR_BND_ASSUMED;cl->n_imm=2;cl->imm[0]=0;cl->imm[1]=es;
    mark_access(c,cl,index_elem_vol(c,pv));
    mark_atomic(cl,index_elem_atomic(c,pv));
    return t;
  }
  return emit_deref_rid(c,pv->rid);                /* depth-aware: the resource carries width, sign, float,
                                                    * ptr_depth and the pointee's volatility */
}
/* Nested member access (`o.in.v` / `dev->ctrl.flags` -- a sub-register-block): given a member `f`
 * already resolved (byte_off relative to the access base) and the cursor at a possible further
 * `.`/`->`, descend through nested value-struct members, accumulating byte offsets. Returns the final
 * field with byte_off set to the total offset from the base, so the load/store paths (which read
 * f.byte_off / f.size / f.bit_*) flatten the chain to a single offset access -- matching the oracle. A member
 * of a volatile struct/union member is volatile storage too (the oracle qualifies each hop by its aggregate). */
static field member_descend(CC *c, field f) {
  while((is(c,".")||is(c,"->")) && f.sidx>=0){
    int base_off=f.byte_off, base_vol=f.is_volatile; sdef *S=&c->s[f.sidx]; c->i++;
    tok fn=adv(c); int fi=-1;
    for(int i=0;i<S->nf;i++) if((int)strlen(S->f[i].name)==fn.n&&!strncmp(S->f[i].name,fn.s,fn.n)) fi=i;
    if(fi<0){ fail(c,"unknown field"); return f; }
    f=S->f[fi]; f.byte_off+=base_off; if(base_vol) f.is_volatile=1;
  }
  return f;
}
/* Continue a postfix read chain through a loaded POINTER value (#fieldderef): `s->mid->k`, the two-hop
 * `s->mid->leaf->x`, and the subscript `s->p[i]`. `ptr` is the loaded pointer rid (kind POINTER), `psidx`
 * its pointee struct index (-1 for a pointer-to-scalar), `pfld` the field it came from (its pointee
 * width/sign/float types a scalar deref). Mirrors the oracle: a pointer field used as a base is loaded,
 * then the loaded pointer is the new base for the next `->`/`[` -- the same move as `*q` on a `T**`. */
/* A member array used as a value (`s.a`, `p->a`, `g.a`): it decays to a pointer to its first element -- its
 * ADDRESS, the `c.addrof` `&s.m` emits, typed `T *` -- never a load of that element, which the member read had
 * emitted (an uncompilable `return s.a;`, and a silent element value under a cast to an integer). A
 * multi-dimensional member array would decay to a row pointer the value model cannot spell (CF-DECAY). */
static uint32_t member_array_value(CC *c, const venv *base, const field *mf){
  if(mf->nadims>1){ fail(c,"a multi-dimensional member array used as a value is not yet supported"); return 0; }
  uint32_t t;
  if(mf->elem_ptr){                                 /* an array of pointers: a pointer to its first pointer */
    t=add_res(c,BCIR_DOM_RAM, mf->ptee_size?mf->ptee_size:4, 1,0,BCIR_RK_POINTER,"");
    if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
      tr->is_signed=(uint8_t)(mf->signd?1:0); tr->is_float=(uint8_t)(mf->ptee_float?1:0);
      tr->ptr_depth=(uint8_t)((mf->ptee_depth>0?mf->ptee_depth:1)+1); tr->is_atomic=(uint8_t)(mf->ptee_atomic?1:0);
      if(mf->ptee_sidx>=0) fits(c,tr->agg,sizeof tr->agg,"%s %s",c->s[mf->ptee_sidx].is_union?"union":"struct",c->s[mf->ptee_sidx].tag); }
  } else {                                          /* a pointer to its element, as `&s.m`'s leaf pointer */
    t=add_res(c,BCIR_DOM_RAM, mf->size?mf->size:4, 1,0,BCIR_RK_POINTER,"");
    if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
      tr->is_signed=(uint8_t)(mf->signd?1:0); tr->ptr_depth=1;
      tr->is_float=(uint8_t)(mf->is_float?1:0); tr->is_complex=(uint8_t)(mf->is_complex?1:0);
      tr->is_bool=(uint8_t)(mf->is_bool?1:0); tr->is_plain_char=(uint8_t)(mf->is_plain_char?1:0);
      tr->is_atomic=(uint8_t)(mf->is_atomic?1:0);
      if(mf->elem_sidx>=0) fits(c,tr->agg,sizeof tr->agg,"%s %s",c->s[mf->elem_sidx].is_union?"union":"struct",c->s[mf->elem_sidx].tag); }
  }
  bcir_claim *cl=new_claim(c,"c.addrof",BCIR_OP_ADD);
  if(cl){ cl->n_rd=1; cl->rd[0]=base->rid; cl->n_wr=1; cl->wr[0]=t; cl->n_imm=1; cl->imm[0]=mf->byte_off; }
  return t;
}
static uint32_t postfix_ptr_chain(CC *c, uint32_t ptr, int psidx, field pfld) {
  for(;;){
    if(is(c,"[")){                                   /* `...->p[i]`: index the loaded pointer (base[idx]) */
      c->i++; uint32_t ix=p_expr(c); eat(c,"]");
      if(psidx>=0 && (is(c,".")||is(c,"->"))){       /* `...->p[i].f`: a member of an element of a pointer to
                                                      * structs, strided by the element as `a[i].f` is */
        venv e=ptr_elem_base(c,ptr,psidx,&pfld);
        { field af; uint32_t tot; int r=aos_member_array(c,&e,ix,&af,&tot);   /* `...->p[i].m[j]` */
          if(r<0) return 0;
          if(r>0) return emit_member_index(c,&e,&af,tot); }
        field sub; if(aos_elem_field(c,&e,&sub)) return emit_index_field(c,&e,ix,&sub);
        return 0; }
      venv b; memset(&b,0,sizeof b); b.rid=ptr; b.sidx=-1;
      b.type.size=pfld.ptee_size?pfld.ptee_size:4; b.type.signd=pfld.signd; b.type.is_float=(uint8_t)pfld.ptee_float;
      b.type.is_volatile=(uint8_t)(pfld.ptee_volatile?1:0);   /* `d->regs[i]` through a `volatile T *regs` */
      b.type.is_atomic=(uint8_t)(pfld.ptee_atomic?1:0);       /* ... and `s->ap[i]` through an `_Atomic T *ap` */
      if(pfld.ptee_sidx>=0 && pfld.ptee_depth<=1){ b.type.kind=1; b.sidx=pfld.ptee_sidx; }   /* `s->ps[i]` through a
                                                               * `struct T *ps`: a struct element (CF-STRUCTVAL) */
      else if(pfld.fp_sig && pfld.ptee_depth<=1){           /* `s->tab[i]` through an `op_t *tab`: a function-pointer
                                                             * element (CF-FPTAB) */
        b.type.kind=3; b.type.fp_sig=(uint16_t)pfld.fp_sig; fits(c,b.type.tag,sizeof b.type.tag,"%s",sig_alias(c,pfld.fp_sig)); }
      return emit_index(c,&b,ix);
    }
    if(is(c,"->")||is(c,".")){
      if(psidx<0){ fail(c,"member access through a pointer to a non-struct"); return ptr; }
      c->i++; tok fn=adv(c); sdef *S=&c->s[psidx]; int fi=-1;
      for(int i=0;i<S->nf;i++) if((int)strlen(S->f[i].name)==fn.n&&!strncmp(S->f[i].name,fn.s,fn.n)) fi=i;
      if(fi<0){ fail(c,"unknown field"); return ptr; }
      field mf=member_descend(c,S->f[fi]);           /* flatten any nested value-struct hops */
      venv b; memset(&b,0,sizeof b); b.rid=ptr; b.sidx=psidx; b.type.kind=1;   /* base = the loaded pointer */
      b.type.is_volatile=(uint8_t)(pfld.ptee_volatile?1:0);   /* a pointer to a volatile struct */
      if(is(c,"(")){     /* funcptr-member call through the loaded pointer: `d->ops->fn(args)` (#fnptrchain) */
        c->i++; uint32_t args[BCIR_CLAIM_MAX_RD]; int na=0, dropped=0;
        if(!is(c,")")) for(;;){ uint32_t a=p_expr(c); if(na<BCIR_CLAIM_MAX_RD-1)args[na++]=a; else dropped=1;
          if(is(c,",")){c->i++;continue;} break; }
        eat(c,")");
        field ff=S->f[fi];                               /* the funcptr field carries its captured return type */
        null_pointer_sig(c,ff.fp_sig,args,na);           /* a null pointer argument: its parameter's pointer */
        uint32_t t = field_call_temp(c,&ff);             /* typed by the member's function type (CF-FPRET) */
        char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.call.imember:%s",S->f[fi].name);
        bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
        if(cl){cl->n_rd=(uint8_t)(na+1);cl->rd[0]=ptr;for(int k=0;k<na;k++)cl->rd[k+1]=args[k];
          cl->n_wr=(uint8_t)(ff.fp_ret_void?0:1);cl->wr[0]=t;cl->n_imm=1;cl->imm[0]=1;cl->truncated=(uint8_t)dropped;}   /* imm0=1: `ptr->fn(args)` */
        qcast_sig_call(c,cl,ff.fp_sig,na);               /* its qualified parameters and return (CF-QUALS) */
        return ff.fp_ret_void ? void_temp(c) : member_call_result(c,&ff,t);
      }
      if(mf.is_ptr && (is(c,"->")||is(c,".")||is(c,"["))){    /* another pointer hop: load it, recurse */
        ptr=emit_member(c,&b,&mf,0); psidx=mf.ptee_sidx; pfld=mf; continue;
      }
      if(mf.arr_count && is(c,"[")){ uint32_t ix=member_arr_index(c,&mf);
        field sub; if(elem_field(c,&mf,&sub)) return emit_member_index_field(c,&b,&mf,ix,&sub);
        if(c->failed) return 0;
        return emit_member_index(c,&b,&mf,ix); }
      if(mf.arr_count) return member_array_value(c,&b,&mf);   /* an array member value: its address */
      return emit_member(c,&b,&mf,c->stmt_expr_declared_bf); /* terminal member load through the loaded pointer */
    }
    return ptr;                                       /* no further postfix: the pointer value itself */
  }
}
/* A SCALAR object reached through a pointer the lvalue itself loads (CF-SPLIT2) -- `h.next->v`, `n->next->v`,
 * `s->p[i]`: the oracle's `_lvalue` for an inc/dec or an assignment used as a value, which the twin had lowered
 * only as a statement (`h.next->v += 1;`) and refused as `h.next->v++`, `++h.next->v` or `x = (h.next->v = 5)`.
 * Kind 1: the member `f` of the struct `*b` points at; kind 2: the element `b[idx]`. `b.rid` is the loaded
 * pointer: its accesses carry no known extent (BCIR_BND_ASSUMED), as store_through_ptr's do. */
typedef struct { int kind; venv b; field f; uint32_t idx; } plv;
/* The rest of the lvalue past the loaded pointer `ptr` (`psidx` its pointee struct, or -1; `pfld` the member it
 * was loaded from): the moves postfix_ptr_chain reads, stopping at the object instead of loading it -- a
 * pointer member followed by another `->`/`.`/`[` is loaded and walked again. 1 with the object in `*o`; 0 when
 * the object is not a scalar these paths store (a struct, an array or pointer member, a struct element): the
 * caller rolls back. Iterative, so a long `a->b->c->...` chain costs no stack. */
static int plv_chain(CC *c, uint32_t ptr, int psidx, field pfld, plv *o){
  memset(o,0,sizeof *o);
  while(!c->failed){
    if(is(c,"[")){                                    /* `...->p[i]`: an element of the loaded pointer */
      if(pfld.ptee_sidx>=0 || pfld.ptee_depth>1) return 0;   /* a struct or pointer element: not a scalar */
      c->i++; uint32_t ix=p_expr(c); if(c->failed || !eat(c,"]")) return 0;
      o->kind=2; o->b.rid=ptr; o->b.sidx=-1; o->idx=ix;
      o->b.type.size=pfld.ptee_size?pfld.ptee_size:4; o->b.type.signd=pfld.signd;
      o->b.type.is_float=(uint8_t)pfld.ptee_float;
      o->b.type.is_volatile=(uint8_t)(pfld.ptee_volatile?1:0); o->b.type.is_atomic=(uint8_t)(pfld.ptee_atomic?1:0);
      return !is(c,"[") && !is(c,".") && !is(c,"->");
    }
    if(!(is(c,"->")||is(c,".")) || psidx<0) return 0;
    c->i++; if(!isk(c,T_ID)) return 0;
    tok fn=adv(c); const sdef *S=&c->s[psidx]; int fi=-1;
    for(int k=0;k<S->nf;k++) if((int)strlen(S->f[k].name)==fn.n && !strncmp(S->f[k].name,fn.s,(size_t)fn.n)) fi=k;
    if(fi<0) return 0;
    field mf=member_descend(c,S->f[fi]);             /* flatten any nested value-struct hops */
    venv b; memset(&b,0,sizeof b); b.rid=ptr; b.sidx=psidx; b.type.kind=1;   /* base = the loaded pointer */
    b.type.is_volatile=(uint8_t)(pfld.ptee_volatile?1:0);
    if(mf.is_ptr && (is(c,"->")||is(c,".")||is(c,"["))){   /* another pointer hop: load it, walk on */
      ptr=emit_member(c,&b,&mf,0); psidx=mf.ptee_sidx; pfld=mf; continue;
    }
    if(mf.is_ptr || mf.arr_count || mf.sidx>=0) return 0;   /* a pointer, array or struct member */
    o->kind=1; o->b=b; o->f=mf;
    return 1;
  }
  return 0;
}
/* The object is volatile, or the pointer reaches a device region -- a struct holding volatile storage (the
 * oracle's `_mmio` of the lvalue's base): an extra access on a re-read, so both rails leave the form to the
 * statement paths. */
static int plv_volatile(CC *c, const plv *lv){
  const bcir_resource *r=res_of(c->fn,lv->b.rid);
  if(r && r->domain==BCIR_DOM_MMIO) return 1;
  return lv->kind==1 ? (lv->f.is_volatile || lv->b.type.is_volatile) : index_elem_vol(c,&lv->b);
}
static int plv_atomic(CC *c, const plv *lv){ return lv->kind==1 ? lv->f.is_atomic : index_elem_atomic(c,&lv->b); }
static bcir_ctype plv_slot(const plv *lv){ return lv->kind==1 ? field_slot(&lv->f) : index_elem_ctype(&lv->b); }
static int plv_narrow(CC *c, const plv *lv, uint32_t val){   /* a store truncates `val`: re-read the object */
  const bcir_resource *r=res_of(c->fn,val); int sz = lv->kind==1 ? lv->f.size : lv->b.type.size;
  return (lv->kind==1 && lv->f.bit_w) || sz < (int)(r?r->elem_bytes:4);
}
static uint32_t plv_read(CC *c, plv *lv){
  return lv->kind==1 ? emit_member(c,&lv->b,&lv->f,0) : emit_index(c,&lv->b,lv->idx);
}
/* `&obj` of an object through a loaded pointer (CF-SPLIT2: `&h.next->v`, `&n->next->p[i]`): the pointer to it,
 * a `c.addrof` of the loaded pointer at the member's offset, or of the element (offset 0, the element's size) --
 * the oracle's address of `_lvalue`. A bit-field has no address; a volatile object's stays a follow-on. */
static uint32_t plv_addr(CC *c, const plv *lv){
  if(lv->kind==1 && lv->f.bit_w){ fail(c,"cannot take the address of a bit-field"); return 0; }
  if(plv_volatile(c,lv)){ fail(c,"address-of a volatile object through a loaded pointer is a follow-on"); return 0; }
  bcir_ctype ty=plv_slot(lv); int sz=ty.size?ty.size:4;
  uint32_t t=add_res(c,BCIR_DOM_RAM, sz, 1,0,BCIR_RK_POINTER,"");   /* a `T *` to the object */
  if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
    tr->is_signed=(uint8_t)(ty.signd?1:0); tr->is_float=(uint8_t)(ty.is_float?1:0); tr->ptr_depth=1;
    tr->is_complex=(uint8_t)(ty.is_complex?1:0); tr->is_bool=(uint8_t)(ty.is_bool?1:0);
    tr->is_plain_char=(uint8_t)(ty.is_plain_char?1:0); tr->is_atomic=(uint8_t)(plv_atomic(c,lv)?1:0); }
  bcir_claim *cl=new_claim(c,"c.addrof",BCIR_OP_ADD);
  if(cl){ cl->n_wr=1; cl->wr[0]=t; cl->rd[0]=lv->b.rid;
    if(lv->kind==1){ cl->n_rd=1; cl->n_imm=1; cl->imm[0]=lv->f.byte_off; }
    else { cl->n_rd=2; cl->rd[1]=lv->idx; cl->n_imm=2; cl->imm[0]=0; cl->imm[1]=sz; } }
  return t;
}
/* `*s.a`, `*q->a`, `*s.in.a` (CF-SPLIT2): the dereference of a one-dimensional member array of non-`_Atomic`
 * scalars is its first element -- one access at the member's offset, as the oracle's `_lvalue(Deref)` of the member
 * has it (its `_addr` keeps the member's offset) -- where the twin took the array's address and accessed through
 * it, a claim graph of its own. With the cursor at the name after `*`: 1 with the base and the element as a scalar
 * member, the cursor past the chain; 0 (cursor unmoved) for any other operand. */
static int postfix_follows(CC *c, int k);   /* fwd: a postfix operator follows token `k` */
static int deref_member_array(CC *c, venv *base, field *elem){
  int save=c->i;
  if(!isk(c,T_ID) || !(tok_is(tat(c,c->i+1),".")||tok_is(tat(c,c->i+1),"->")) || tat(c,c->i+2)->k!=T_ID) return 0;
  venv *vp=lookup(c,pk(c)); if(!vp) vp=use_global(c,pk(c));
  if(!vp || vp->sidx<0) return 0;
  venv v=*vp; const sdef *S=&c->s[v.sidx]; const tok *fn=tat(c,c->i+2); int fi=-1;
  for(int k=0;k<S->nf;k++) if((int)strlen(S->f[k].name)==fn->n && !strncmp(S->f[k].name,fn->s,(size_t)fn->n)) fi=k;
  if(fi<0) return 0;
  c->i+=3;
  field mf=member_descend(c,S->f[fi]);               /* nested value members add their offsets */
  if(!mf.arr_count || mf.nadims>1 || mf.elem_ptr || mf.elem_sidx>=0 || mf.is_atomic || mf.bit_w
     || postfix_follows(c,c->i)){ c->i=save; return 0; }
  mf.arr_count=0; mf.nadims=0;                       /* the element, a scalar at the member's offset */
  *base=v; *elem=mf; return 1;
}
/* <math.h> real-valued functions (mirrors the oracle's _LIBM): the result is a floating type fixed
 * by the name suffix -- base -> double, +f -> float. They lower to an opaque external library edge
 * (c.call.libm), so the emit calls the real libm function (the harness links -lm) and R18 sees no
 * callee. (The +l long-double variants need the twin's long-double support and are deferred.) */
static const char *const g_libm[] = {
  "acos","asin","atan","atan2","cos","sin","tan","acosh","asinh","atanh","cosh","sinh","tanh",
  "exp","exp2","expm1","log","log10","log1p","log2","logb","cbrt","fabs","hypot","pow","sqrt",
  "ceil","floor","round","trunc","nearbyint","rint","erf","erfc","lgamma","tgamma",
  "copysign","fdim","fmax","fmin","fmod","remainder","fma","nextafter",
  "ldexp","scalbn","scalbln","nan",               /* + mixed-arg (int/long exponent, tag string) */
  "frexp","modf","remquo", 0 };                   /* + a pointer out-param (rides c.addrof); double result */

/* <math.h> functions with a fixed *integer* result (the f/l suffix types only the argument): ilogb
 * returns int -- exactly the 4-byte value model. (lround/llround/lrint/llrint return long/long long;
 * they need the twin's wide-integer model and are handled in the 8-byte-return slice.) */
static const char *const g_libm_int[] = { "ilogb", 0 };

/* <math.h> functions returning an 8-byte integer: lround/lrint -> long, llround/llrint -> long long.
 * The result temp is 8 bytes (declared uint64_t in the emit -- a lossless round-trip to the function's
 * long/long long return), so the 8-byte value is not truncated to the 4-byte model. */
static const char *const g_libm_long[] = { "lround","llround","lrint","llrint", 0 };

/* Nonzero if s[0..n) is an int-returning libm function (ilogb, ilogbf, ilogbl). */
static int libm_is_int(const char *s, int n) {
  for(int i=0;g_libm_int[i];i++){ int L=(int)strlen(g_libm_int[i]);
    if(L==n && !strncmp(g_libm_int[i],s,(size_t)n)) return 1;
    if((n==L+1) && (s[n-1]=='f'||s[n-1]=='l') && !strncmp(g_libm_int[i],s,(size_t)L)) return 1; }
  return 0;
}

/* Nonzero if s[0..n) is an 8-byte-integer-returning libm function (the f/l suffix types only the arg). */
static int libm_is_long(const char *s, int n) {
  for(int i=0;g_libm_long[i];i++){ int L=(int)strlen(g_libm_long[i]);
    if(L==n && !strncmp(g_libm_long[i],s,(size_t)n)) return 1;
    if((n==L+1) && (s[n-1]=='f'||s[n-1]=='l') && !strncmp(g_libm_long[i],s,(size_t)L)) return 1; }
  return 0;
}

/* The result float size of a <math.h> call s[0..n): 8 (double) for a base name, 4 (float) for an
 * `f`-suffixed variant, or 0 if not a libm function. The full name is matched first so a base that
 * ends in `f` (erf) is not misread as the float variant of `er`. */
static int libm_float_size(const char *s, int n) {
  for(int i=0;g_libm[i];i++){ if((int)strlen(g_libm[i])==n && !strncmp(g_libm[i],s,(size_t)n)) return 8; }
  if(n>1 && s[n-1]=='f')
    for(int i=0;g_libm[i];i++){ if((int)strlen(g_libm[i])==n-1 && !strncmp(g_libm[i],s,(size_t)(n-1))) return 4; }
  return 0;
}
/* a `long double` libm variant -- a base name with an `l` suffix (`sinl`, `sqrtl`, `fabsl`): the result
 * is `long double`, sized by the target ABI (resolved at the call site, where the ABI is in scope). */
static int libm_is_ld(const char *s, int n) {
  if(n<=1 || s[n-1]!='l') return 0;
  for(int i=0;g_libm[i];i++){ if((int)strlen(g_libm[i])==n-1 && !strncmp(g_libm[i],s,(size_t)(n-1))) return 1; }
  return 0;
}
/* <complex.h> functions, lowered like libm (c.call.libm, opaque to R18). The result is the *real*
 * element float for creal/cimag/cabs/carg, or the *complex* type for conj/cproj AND the C99 complex
 * transcendentals (cexp/csqrt/...). The full name is matched first so creal/cimag (which themselves end
 * in l/g) aren't misread as an `l`-suffixed variant. */
static const char *const g_cplx_real[] = { "creal","cimag","cabs","carg", 0 };
static const char *const g_cplx_cplx[] = { "conj","cproj",                            /* algebraic */
  "cexp","clog","csqrt","cpow",                                                       /* exp/log/sqrt/pow */
  "csin","ccos","ctan","casin","cacos","catan",                                       /* circular + inverse */
  "csinh","ccosh","ctanh","casinh","cacosh","catanh", 0 };                            /* hyperbolic + inverse */
static int cplx_name_in(const char *const*set,const char *s,int n){
  for(int i=0;set[i];i++) if((int)strlen(set[i])==n && !strncmp(set[i],s,(size_t)n)) return 1; return 0; }
/* The ELEMENT float size of a <complex.h> call (8/double, 4/+f, long_double/+l), or 0 if not one;
 * *is_cplx is set 1 when the RESULT is itself complex (conj/cproj) vs a real element (creal/cimag/...). */
static int cplx_libm(CC *c,const char *s,int n,int *is_cplx){
  if(cplx_name_in(g_cplx_real,s,n)){ *is_cplx=0; return 8; }
  if(cplx_name_in(g_cplx_cplx,s,n)){ *is_cplx=1; return 8; }
  if(n>1 && (s[n-1]=='f'||s[n-1]=='l')){ int b=n-1, ld=s[n-1]=='l';
    int es = ld ? cc_abi(c)->long_double_size : 4;
    if(cplx_name_in(g_cplx_real,s,b)){ *is_cplx=0; return es; }
    if(cplx_name_in(g_cplx_cplx,s,b)){ *is_cplx=1; return es; } }
  return 0;
}
/* The <complex.h> imaginary unit: `I` is the macro `_Complex_I`, a `const float _Complex` of value i.
 * (Clang doesn't implement `_Imaginary`, so `_Imaginary_I` is intentionally not recognized.) Recognized
 * only when the name is NOT a declared variable/global/enum (a user `I` shadows it) and emitted VERBATIM,
 * so the re-emitted twin resolves it against <complex.h> exactly as the original does. */
static int is_imag_unit(const tok *id){
  return (id->n==1 && id->s[0]=='I') || (id->n==10 && !strncmp(id->s,"_Complex_I",10)); }
/* SEG6.1/SEG7: the C11 `memory_order_*` constants AND the GCC/Clang `__ATOMIC_*` macro spellings (they
 * share the same integer values 0..5). A `memory_order` ARGUMENT spelled as one of these named constants
 * folds to its value. The C twin of the oracle's `_MEMORDER` map -- used both by the identifier->rvalue
 * widening (an unshadowed `memory_order_*` name reads as that int const, like a literal) and by the
 * order-parameterized fence routing (`_fence_order_kind`). Returns 1 and sets *out when `id` matches a
 * known constant; 0 otherwise. */
static int memorder_value(const tok *id, long long *out){
  static const struct{const char *n; long long v;} M[]={
    {"memory_order_relaxed",0},{"memory_order_consume",1},{"memory_order_acquire",2},
    {"memory_order_release",3},{"memory_order_acq_rel",4},{"memory_order_seq_cst",5},
    {"__ATOMIC_RELAXED",0},{"__ATOMIC_CONSUME",1},{"__ATOMIC_ACQUIRE",2},
    {"__ATOMIC_RELEASE",3},{"__ATOMIC_ACQ_REL",4},{"__ATOMIC_SEQ_CST",5},{0,0}};
  for(int i=0;M[i].n;i++) if((int)strlen(M[i].n)==id->n && !strncmp(M[i].n,id->s,(size_t)id->n)){ *out=M[i].v; return 1; }
  return 0;
}
/* SEG7: order value -> fence-kind op string (the C twin of the oracle's `_ORDER_KIND`). acquire(2)/
 * consume(1) -> a load (acquire) fence; release(3) -> a store (release) fence; seq_cst(5)/acq_rel(4)/
 * relaxed(0) and any out-of-range value -> the FULL fence (a sound over-approximation -- a stronger
 * fence never under-synchronizes, so acq_rel and relaxed both conservatively fold to full). */
static const char *order_kind(long long order){
  switch(order){ case 1: case 2: return "c.fence.acquire"; case 3: return "c.fence.release"; default: return "c.fence"; }
}
/* The printf / scanf family of external variadic <stdio.h> functions -- not defined in the unit and not
 * lowered, they emit verbatim (like a libm call, opaque to R18) and return int. (The format string is a
 * read-only char[] literal, already passed through as an argument.) */
static int is_extern_variadic(const char *s, int n) {
  static const char *F[]={"snprintf","vsnprintf","sprintf","vsprintf","printf","fprintf","vprintf",
    "vfprintf","sscanf","vsscanf","scanf","fscanf","dprintf",0};
  for(int i=0;F[i];i++) if((int)strlen(F[i])==n && !strncmp(F[i],s,(size_t)n)) return 1;
  return 0;
}
/* <stdlib.h> memory management -- external libc edges (emitted VERBATIM, opaque to R18, NOT bcir_-renamed),
 * the seam the naked-pointer safety track (§5.12) hangs lifetime annotations on. Returns 1 for an allocator
 * (malloc/calloc/realloc/aligned_alloc -> `void *`), 2 for `free` (-> void), 0 otherwise. */
static int is_stdlib_alloc(const char *s, int n) {
  static const char *A[]={"malloc","calloc","realloc","aligned_alloc",0};
  for(int i=0;A[i];i++) if((int)strlen(A[i])==n && !strncmp(A[i],s,(size_t)n)) return 1;
  if(n==4 && !strncmp("free",s,4)) return 2;
  return 0;
}
/* <string.h> memory routines -- external libc edges like the allocators (verbatim, opaque to R18, NOT
 * bcir_-renamed), each returning its destination, a `void *`. The emit spells every plain memory access as a
 * `memcpy`, so a re-parsed emit calls it (CF-RTWIDE). The oracle's `_STRING_MEM`. */
static int is_string_mem(const char *s, int n) {
  static const char *M[]={"memcpy","memmove","memset",0};
  for(int i=0;M[i];i++) if((int)strlen(M[i])==n && !strncmp(M[i],s,(size_t)n)) return 1;
  return 0;
}

/* B-breadth (#61) LAPACK: nonzero if s[0..n) is a Fortran-ABI LU/solve driver base name with a trailing
 * underscore (e.g. `sgesv_`) -- the symbol a C caller links against. Mirrors linkflags.py's _LAPACK_FORTRAN
 * set EXACTLY (same names, same trailing-underscore convention); the LAPACKE_ C interface is matched by
 * prefix in bcir_lib_for_callee. Kept tiny/explicit so an unrelated `foo_` callee is not swept into it. */
static int lapack_is_fortran(const char *s, int n) {
  static const char *L[]={"sgesv","dgesv","sgetrf","dgetrf","sgetrs","dgetrs",0};
  if(n<2 || s[n-1]!='_') return 0;
  int base=n-1;
  for(int i=0;L[i];i++) if((int)strlen(L[i])==base && !strncmp(L[i],s,(size_t)base)) return 1;
  return 0;
}

/* B1 link-flag derivation -- the byte-identical C twin of bcir/frontends/cfront/linkflags.py. The
 * callee->library classification is the SOURCE OF TRUTH for what an external-call edge links against;
 * both rails must agree (gated in test_c_cfront.py + check_runtime.sh). `s[0..n)` is the external
 * callee name (the suffix of a c.call.libm: / c.call.libm.void: / c.call.extern: claim op).
 *
 * Returns the `-l...` flag, "" for a known-but-implicit libc symbol (NO flag, but EXPLICITLY known),
 * or NULL for an UNKNOWN external callee. Unknown-callee policy (deterministic): NULL contributes no
 * flag -- BCIR does not invent a `-l` it can't justify; an unknown symbol is the build system's to
 * resolve (today's behaviour). NULL is kept DISTINCT from "" so the mapping is a complete statement of
 * what BCIR knows about its own emitted external seams.
 *
 * EXTENSION POINT (roadmap B2): add one branch per newly-wrapped trusted library here, in the SAME
 * ORDER as the oracle's _LIBRARY_RULES (e.g. fftw_*->"-lfftw3", LAPACKE_*->"-llapack", gsl_*->"-lgsl",
 * Sleef_*->"-lsleef", erfcx*->"-lcerf"). First match wins, so order is significant. */
static const char *bcir_lib_for_callee(const char *s, int n) {
  if(n<=0) return NULL;
  /* <math.h> / <complex.h> (incl. the f/l-suffixed + fixed-int/long variants) -> -lm. */
  if(libm_float_size(s,n) || libm_is_int(s,n) || libm_is_long(s,n) || libm_is_ld(s,n)) return "-lm";
  /* libc-implicit (malloc/free/realloc/calloc/aligned_alloc, memcpy/memmove/memset + the printf/scanf family)
   * -> no flag. */
  if(is_stdlib_alloc(s,n) || is_string_mem(s,n) || is_extern_variadic(s,n)) return "";
  /* B5 BLAS: cblas_sgemm and any cblas_* (CBLAS) -> -lcblas (the existing B5 path's choice). */
  if(n>=6 && !strncmp("cblas_",s,6)) return "-lcblas";
  /* B2 FFTW: fftwf_* (single-prec) and fftw_* (double) -> -lfftw3 (the B2 wrap's choice -- fftwf_* also
   * lives in -lfftw3). Matches linkflags.py's fftw rule, in the SAME order (first match wins). */
  if(n>=6 && !strncmp("fftwf_",s,6)) return "-lfftw3";
  if(n>=5 && !strncmp("fftw_",s,5))  return "-lfftw3";
  /* B-breadth (#61) LAPACK: the LAPACKE C interface (LAPACKE_sgesv et al.) and the Fortran-ABI driver
   * symbols (sgesv_/...) -> -llapack (the linear-solve wrap emit_lapack_solve_c calls LAPACKE_sgesv and
   * links -llapacke -llapack; -llapack is the load-bearing dep). Matches linkflags.py's LAPACK rule, in
   * the SAME order (first match wins). */
  if(n>=8 && !strncmp("LAPACKE_",s,8)) return "-llapack";
  if(lapack_is_fortran(s,n))           return "-llapack";
  /* Area-B breadth (#62) GSL: any gsl_* (the GNU Scientific Library -- special functions / statistics) ->
   * -lgsl (the statistics wrap emit_gsl_stats_c calls gsl_stats_mean/variance/sd and links -lgsl
   * -lgslcblas; -lgsl is the load-bearing dep). Matches linkflags.py's GSL rule, in the SAME order. */
  if(n>=4 && !strncmp("gsl_",s,4)) return "-lgsl";
  /* Area-B breadth (#63) SLEEF: any Sleef_* (the SIMD-oriented vectorized math library -- a fast,
   * vectorized libm) -> -lsleef (the vectorized-exp wrap emit_sleef_exp_c calls Sleef_expf1_u10 and links
   * -lsleef). Matches linkflags.py's SLEEF rule, in the SAME order (first match wins). */
  if(n>=6 && !strncmp("Sleef_",s,6)) return "-lsleef";
  /* Area-B breadth (SEG2) libcerf: erfcx / erfcxf (the scaled complementary error function
   * erfcx(x) = e^{x^2}*erfc(x)) -> -lcerf (the erfcx wrap emit_cerf_erfcx_c calls erfcxf and links -lcerf).
   * erfcx is a numerically-robust special function libm LACKS (the naive expf(x*x)*erfcf(x) overflows for
   * large x; libcerf's erfcx stays finite on the full real line). Matches linkflags.py's erfcx rule, in the
   * SAME order (first match wins). */
  if(n>=5 && !strncmp("erfcx",s,5)) return "-lcerf";
  /* --- EXTENSION POINT: one branch per newly-wrapped library, matching the oracle's order. --- */
  return NULL;                                          /* unknown external callee -> no flag */
}

/* The external callee named by a claim op (the suffix of a c.call.libm:/.void:/extern: edge), or NULL
 * if `op` is not an external-call edge; sets *len to the callee length. Mirrors the oracle's
 * _EXTERN_CALL_PREFIXES + _callee_of. */
static const char *bcir_extern_callee(const char *op, int *len) {
  static const char *const P[]={"c.call.libm:","c.call.libm.void:","c.call.extern:",0};
  for(int i=0;P[i];i++){ size_t pl=strlen(P[i]);
    if(!strncmp(op,P[i],pl)){ const char *c=op+pl; *len=(int)strlen(c); return c; } }
  return NULL;
}

/* B1: derive the deduped, STABLY-SORTED linker flags a whole unit's external-call edges need, written
 * one space-separated line to `buf` (e.g. "-lm"; empty for a pure-integer unit). Reproducible (a BCIR
 * hard requirement): the flags are sorted, so the same unit always yields a byte-identical line,
 * independent of claim order. The Python oracle (linkflags.derive_link_flags) produces the identical
 * string. NB: kept tiny -- the flag SET is small (one per linked library), so an insertion-sorted
 * fixed array is exact and allocation-free. */
void bcir_cfront_link_flags(const bcir_unit *u, char *buf, size_t cap) {
  #define BCIR_MAX_LINK_FLAGS 32
  if(!buf||!cap)return;buf[0]=0;if(!u)return;
  const char *flags[BCIR_MAX_LINK_FLAGS]; int nf=0;
  for(int fi=0; fi<u->n_funcs; fi++){ const bcir_func *f=&u->funcs[fi];
    for(size_t ci=0; ci<f->n_claims; ci++){
      int len=0; const char *callee=bcir_extern_callee(f->claims[ci].op,&len);
      if(!callee) continue;
      const char *flag=bcir_lib_for_callee(callee,len);
      if(!flag || !flag[0]) continue;                  /* "" (implicit) and NULL (unknown) add no flag */
      int dup=0; for(int k=0;k<nf;k++) if(!strcmp(flags[k],flag)){ dup=1; break; }
      if(dup) continue;
      if(nf>=BCIR_MAX_LINK_FLAGS) continue;             /* defensive: the live library set is tiny */
      int p=nf;                                         /* insertion sort -> a deterministic sorted set */
      while(p>0 && strcmp(flags[p-1],flag)>0){ flags[p]=flags[p-1]; p--; }
      flags[p]=flag; nf++;
    }
  }
  size_t w=0;
  for(int k=0;k<nf && w<cap;k++)
    w+=(size_t)snprintf(buf+w, w<cap?cap-w:0, "%s%s", k?" ":"", flags[k]);
  if(cap){ if(w>=cap) w=cap-1; buf[w]=0; }
  #undef BCIR_MAX_LINK_FLAGS
}
/* GCC/Clang integer builtins -- emitted verbatim (no bcir_ twin, opaque to R18) with a fixed result type.
 * Returns the result's SIGNED size: -4 a signed int (the bit-count family + abs), -8 a signed long
 * (labs/llabs), or a POSITIVE unsigned size for byte-swap (2/4/8). 0 == not a recognized builtin. */
static int builtin_result(const char *s, int n) {
  static const char *I[]={"__builtin_popcount","__builtin_popcountl","__builtin_popcountll",
    "__builtin_clz","__builtin_clzl","__builtin_clzll","__builtin_ctz","__builtin_ctzl","__builtin_ctzll",
    "__builtin_ffs","__builtin_ffsl","__builtin_ffsll","__builtin_parity","__builtin_parityl",
    "__builtin_parityll","__builtin_clrsb","__builtin_clrsbl","__builtin_clrsbll","__builtin_abs",0};
  for(int i=0;I[i];i++) if((int)strlen(I[i])==n && !strncmp(I[i],s,(size_t)n)) return -4;   /* -> signed int */
  #define _BLT(L) ((int)(sizeof(L)-1)==n && !strncmp(L,s,(size_t)n))
  if(_BLT("__builtin_labs")||_BLT("__builtin_llabs")) return -8;   /* -> signed long / long long */
  if(_BLT("__builtin_bswap16")) return 2;          /* -> unsigned 16 */
  if(_BLT("__builtin_bswap32")) return 4;          /* -> unsigned 32 */
  if(_BLT("__builtin_bswap64")) return 8;          /* -> unsigned 64 */
  #undef _BLT
  return 0;
}

/* The return ctype of a user function defined *earlier* in the unit (so a call can be typed by its
 * callee), or NULL if it is not yet defined (a forward reference / external -> the uint32 default). */
static const bcir_ctype *callee_ret(CC *c, const tok *name) {
  if(!c->unit) return NULL;
  for(int i=0;i<c->unit->n_funcs;i++){ const char *fn=c->unit->funcs[i].name;
    if((int)strlen(fn)==name->n && !strncmp(fn,name->s,(size_t)name->n)) return &c->unit->funcs[i].ret; }
  return NULL;
}

/* An identifier nothing in scope declares -- the oracle's `_lookup` diagnostic, the name quoted (CF-DECLS). Out of
 * the recursive descent's frames (BCIR_NOINLINE), as its buffer would sit in every level. */
static BCIR_NOINLINE void fail_undeclared(CC *c, const tok *id){
  char msg[BCIR_CIR_NAME+40];
  snprintf(msg,sizeof msg,"use of undeclared identifier '%.*s'",id->n<BCIR_CIR_NAME?id->n:BCIR_CIR_NAME,id->s);
  fail(c,msg);
}
/* The return type of the function `name` as a declaration before this point gives it -- an earlier definition
 * (`callee_ret`) or prototype -- or NULL: a later declaration does not declare it here (C11 6.5.1p2; the oracle's
 * `_require_declared`). What a `sizeof` of a call and a function designator are typed by. */
static const bcir_ctype *declared_ret(CC *c, const tok *name) {
  const bcir_ctype *r=callee_ret(c,name);
  if(r) return r;
  for(int k=0;k<c->n_protos;k++)
    if((int)strlen(c->protos[k].name)==name->n && !strncmp(c->protos[k].name,name->s,(size_t)name->n))
      return &c->protos[k].ret;
  return NULL;
}
static int unit_defines(CC *c, const tok *name);   /* fwd: every definition of the unit (below) */

/* Nonzero if the unit DEFINES a function named `name` anywhere -- a file-scope `name ( ... ) {`: the oracle's
 * `func_rets`, which every definition of the unit fills before any body lowers. A library name the unit defines
 * is the unit's own function, never the library's edge; the twin parses in one pass, so `callee_ret` sees only
 * the definitions before the call, and a `malloc` the unit defined had lowered as the libc edge (CF-RTWIDE).
 * Only the library recognizers ask, for their own few names, and each name's scan is cached. */
static int unit_defines(CC *c, const tok *name) {
  for(int k=0;k<c->n_libdef;k++)
    if(c->libdef[k].n==name->n && !strncmp(c->libdef[k].s,name->s,(size_t)name->n)) return c->libdef[k].def;
  int def=0, depth=0;
  for(int i=0;i<c->nt && !def;i++){ const tok *t=&c->t[i];
    if(t->k==T_PUN){ if(tok_is(t,"{")) depth++; else if(tok_is(t,"}") && depth>0) depth--; continue; }
    if(t->k!=T_ID || depth || t->n!=name->n || strncmp(t->s,name->s,(size_t)name->n) || !tok_is(tat(c,i+1),"("))
      continue;
    int j=i+1, pd=0;                       /* the parameter list's `)`, then the body's `{` */
    for(;j<c->nt;j++){ const tok *u=&c->t[j];
      if(u->k!=T_PUN) continue;
      if(tok_is(u,"(")) pd++; else if(tok_is(u,")") && --pd==0) break; }
    def=tok_is(tat(c,j+1),"{");
  }
  if(c->n_libdef<(int)(sizeof c->libdef/sizeof c->libdef[0])){
    c->libdef[c->n_libdef].s=name->s; c->libdef[c->n_libdef].n=name->n; c->libdef[c->n_libdef].def=def;
    c->n_libdef++; }
  return def;
}

static uint32_t p_call(CC *c, const tok *name) {
  if(tok_is(name,"va_arg")){          /* va_arg(ap, TYPE) -- the 2nd arg is a type-name, parsed specially */
    c->i++; /* '(' */
    uint32_t ap=p_expr(c); eat(c,",");
    int k0=c->i; bcir_ctype ty; int si; if(p_type(c,&ty,&si)) return 0;
    int k1=c->i; eat(c,")");
    uint32_t t;                        /* type the result by T so downstream arithmetic/loads are correct */
    if(ty.is_float) t=tempf(c,ty.size);
    else if(ty.kind==1 && si>=0) t=tempagg(c,si);   /* `va_arg(ap, struct T)`: the struct value (CF-STRUCTVAL) */
    else if(ty.kind==2){ t=temp(c,cc_abi(c)->pointer_size); bcir_resource *pr=&c->fn->res[c->fn->n_res-1];
      pr->is_signed=(uint8_t)(ty.signd?1:0); pr->is_float=(uint8_t)(ty.is_float?1:0);
      pr->ptr_depth=ty.ptr_depth?ty.ptr_depth:1; pr->is_plain_char=(uint8_t)(ty.is_plain_char?1:0);
      if(ty.ptr_to_struct) fits(c,pr->agg,sizeof pr->agg,"%s %s",ty.is_union?"union":"struct",ty.tag);
      else if(ty.ptr_to_fp) ptee_fp(c,pr,&ty); }
    else t=tempi(c,ty.size?ty.size:4, ty.signd?1:0);
    /* T rides in the op (the digest strips it: the oracle's op is bare) for the emit's `va_arg(ap, T)`: its tokens,
     * one space apart, or -- when they would not fit -- the spelling of the type they parsed to, which always does.
     * The op once held T's source span cut at 16 characters (`unsigned long long` became `unsigned long lo`, which
     * does not compile), comments and newlines included (CF-BUF). */
    char op[BCIR_CIR_OP]; size_t ow=(size_t)snprintf(op,sizeof op,"c.call.vaarg:");
    for(int k=k0;k<k1 && ow<sizeof op;k++){ const tok *tk=tat(c,k);
      ow+=(size_t)snprintf(op+ow,sizeof op-ow,"%s%.*s",k>k0?" ":"",tk->n,tk->s); }
    if(ow>=sizeof op){ char ts[BCIR_EMIT_TYPE]; ctype_str(&ty,ts,sizeof ts); fits(c,op,sizeof op,"c.call.vaarg:%s",ts); }
    bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
    if(cl){cl->n_rd=1;cl->rd[0]=ap;cl->n_wr=1;cl->wr[0]=t;}
    return t;
  }
  c->i++; /* '(' */
  uint32_t args[BCIR_CLAIM_MAX_RD]; int na=0, dropped=0;
  if(!is(c,")")) for(;;){ uint32_t a=p_expr(c); if(na<BCIR_CLAIM_MAX_RD)args[na++]=a; else dropped=1;
    if(is(c,",")){c->i++;continue;} break; }
  eat(c,")");
  c->call_dropped=dropped;             /* every path below creates the call claim LAST (the site marks it) */
  if(tok_is(name,"va_start")||tok_is(name,"va_end")||tok_is(name,"va_copy")){   /* opaque void variadic builtins */
    char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.call.vabuiltin:%.*s",name->n,name->s);
    bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
    if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=0;}
    return void_temp(c);               /* a void result -- never read */
  }
  int bz = builtin_result(name->s,name->n);
  if(bz){                              /* a GCC/Clang integer builtin -> verbatim, typed, opaque to R18 */
    uint32_t t = bz<0 ? tempi(c,-bz,1) : tempi(c,bz,0);
    char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.call.builtin:%.*s",name->n-10,name->s+10);  /* drop `__builtin_` (op cap) */
    bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
    if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=1;cl->wr[0]=t;}
    return t;                          /* not added to fn->calls (opaque to R18) */
  }
  int cx_is; int cz = cplx_libm(c,name->s,name->n,&cx_is);   /* <complex.h> creal/cimag/conj/... */
  if(cz){                                  /* a typed external complex-library edge (counts as one call) */
    uint32_t t = cx_is ? tempc(c,cz*2) : tempf(c,cz);        /* conj -> complex (2x elem); creal -> real */
    char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.call.libm:%.*s",name->n,name->s);
    bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
    if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=1;cl->wr[0]=t;}
    return t;
  }
  int lz = libm_is_long(name->s,name->n) ? -8
         : libm_is_int(name->s,name->n)  ? -4
         : libm_is_ld(name->s,name->n)   ? cc_abi(c)->long_double_size   /* sinl/sqrtl/... -> long double */
         : libm_float_size(name->s,name->n);
  if(lz){                                  /* a <math.h> call -> a typed external library edge */
    uint32_t t = lz<0 ? temp(c,-lz) : tempf(c,lz);  /* lround -> 8-byte int, ilogb -> 4-byte int, else float */
    char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.call.libm:%.*s",name->n,name->s);
    bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
    if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=1;cl->wr[0]=t;}
    return t;                              /* not added to fn->calls (opaque to R18) */
  }
  int sal=is_stdlib_alloc(name->s,name->n);  /* <stdlib.h> malloc/calloc/realloc/free -- external libc edge */
  if(sal && unit_defines(c,name)) sal=0;     /* ... unless the unit defines it: its own function */
  if(sal){
    if(na>0 && (sal==2 || tok_is(name,"realloc"))){   /* `free(0)`, `realloc(0, n)`: a null `void *` (CF-NULLCALL) */
      bcir_ctype vp; memset(&vp,0,sizeof vp); vp.kind=2; vp.ptr_depth=1; null_pointer(c,args[0],&vp); }
    if(sal==2){                              /* free(p) -> a void external call statement (opaque to R18) */
      char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.call.libm.void:%.*s",name->n,name->s);
      bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
      /* R21 lifetime FREE event (§5.12): the freed pointer it reads dies after this claim, so a later
       * dereference is a use-after-free (or a second free a double-free). Digest-excluded + advisory. */
      if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=0;cl->lifetime=2;}
      return void_temp(c);                   /* a void result -- never read */
    }
    uint32_t t=add_res(c,BCIR_DOM_RAM,cc_abi(c)->pointer_size,1,0,BCIR_RK_POINTER,"");  /* a `void *` result */
    if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1]; tr->ptr_depth=1; }  /* no agg -> `void *` */
    char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.call.libm:%.*s",name->n,name->s);
    bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
    /* R21 lifetime ALLOC event (§5.12): the allocator result (re-)validates the resource it writes, so a
     * pointer reassigned from it is live again after an earlier free. Digest-excluded + advisory. */
    if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=1;cl->wr[0]=t;cl->lifetime=1;}
    return t;                                /* not added to fn->calls (opaque to R18) */
  }
  if(is_string_mem(name->s,name->n) && !unit_defines(c,name)){   /* <string.h> memcpy/memmove/memset */
    uint32_t t=add_res(c,BCIR_DOM_RAM,cc_abi(c)->pointer_size,1,0,BCIR_RK_POINTER,"");  /* the destination */
    if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1]; tr->ptr_depth=1; }  /* no agg -> `void *` */
    char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.call.libm:%.*s",name->n,name->s);
    bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
    if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=1;cl->wr[0]=t;}
    return t;                                /* not added to fn->calls (opaque to R18) */
  }
  const bcir_ctype *rt=callee_ret(c,name);   /* type the result by the callee's return (earlier defs) */
  if(rt && rt->kind==0 && rt->size==0){      /* a void callee -> a bare call statement, no result temp */
    char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.call.void:%.*s",name->n,name->s);
    bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
    if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=0;}
    add_call(c,name);
    return void_temp(c);                     /* an unused placeholder (a void result is never read) */
  }
  if(is_extern_variadic(name->s,name->n) && !unit_defines(c,name)){   /* a printf/scanf-family external -> opaque */
    uint32_t t=tempi(c,4,1);                          /* returns int; emitted verbatim against <stdio.h> */
    char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.call.extern:%.*s",name->n,name->s);
    bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
    if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=1;cl->wr[0]=t;}
    return t;                                         /* not added to fn->calls (opaque to R18) */
  }
  if(!rt){                                            /* Phase 3 LINKING: a PROTOTYPED cross-TU callee */
    const bcir_ctype *ptt=NULL; int pk=-1;
    for(int k=0;k<c->n_protos;k++)
      if((int)strlen(c->protos[k].name)==name->n && !strncmp(c->protos[k].name,name->s,(size_t)name->n)){
        ptt=&c->protos[k].ret; pk=k; break; }
    if(ptt){
      /* A typed external edge the host LINKER resolves from a sibling object: like a libm edge it is
       * opaque to the in-unit R18 call graph (NOT added to fn->calls) and emits verbatim (external
       * linkage) with the prototype's extern declaration in the prelude; unlike libm it derives no -l
       * flag. Result typing mirrors the defined-callee ladder below (the oracle's _call_result_ct). */
      if(ptt->kind==1){ fail(c,"aggregate return through a prototype is not supported"); return temp(c,4); }
      char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.call.tu:%.*s",name->n,name->s);
      if(ptt->kind==0 && ptt->size==0){               /* a void cross-TU callee -> a bare statement */
        bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
        if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=0;}
        qcast_call(c,cl,0,na,c->protos[pk].params,c->protos[pk].n_params,NULL);   /* (CF-QUALS) */
        return void_temp(c);                          /* a void result -- never read */
      }
      uint32_t t = ptt->kind==2 ? temp_ptr(c,ptt,ptt->ptr_to_struct?find_struct(c,ptt->tag,(int)strlen(ptt->tag)):-1)
                 : ptt->kind==3 ? fp_ret_temp(c,ptt)
                 : ptt->is_complex ? tempc(c,ptt->size)
                 : ptt->is_float   ? tempf(c,ptt->size)
                 : (ptt->kind==0 && ptt->bit_width>0) ? tempbi(c,ptt->bit_width,ptt->signd)
                 : (ptt->kind==0 && ptt->size==8) ? tempi(c,8,ptt->signd)
                 : (ptt->kind==0 && ptt->size<=4 && ptt->signd) ? tempi(c,4,1)
                 : temp(c,4);
      bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
      if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=1;cl->wr[0]=t;}
      qcast_call(c,cl,0,na,c->protos[pk].params,c->protos[pk].n_params,ptt);   /* (CF-QUALS) */
      return t;
    }
  }
  if(!rt && !(c->fn && (int)strlen(c->fn->name)==name->n && !strncmp(c->fn->name,name->s,(size_t)name->n))){
    /* no definition or prototype of the callee precedes the call, and it is not the function itself: the
     * unit's end refuses it if the unit declares it later (the oracle's `_require_declared`); a callee the
     * unit never declares stays R18's undefined edge */
    struct cc_undecl *u=(struct cc_undecl *)bcir_host_arena_allocate(&c->scratch,sizeof *u,_Alignof(struct cc_undecl));
    if(!u){ cc_raise_oom(c); return temp(c,4); }
    u->next=NULL; idcpy(c,u->name,name);
    if(c->undecl_last) c->undecl_last->next=u; else c->undecl=u;
    c->undecl_last=u;
  }
  uint32_t t;
  if(rt && rt->kind==1){                              /* a struct/union RETURN: a by-value aggregate temp
                                                       * (`struct P t = mk(x);`) so it copies/passes/member-accesses
                                                       * -- a uint32 temp emitted invalid C `uint32_t t = mk(x)`. */
    t=add_res(c,BCIR_DOM_RAM,rt->size,1,0,BCIR_RK_AGGREGATE,"");
    if(c->fn->n_res) fits(c,c->fn->res[c->fn->n_res-1].agg,BCIR_CIR_AGG,"%s %s",
                              rt->is_union?"union":"struct", rt->tag);
  } else
    t = (rt && rt->kind==2)             ? temp_ptr(c,rt,rt->ptr_to_struct?find_struct(c,rt->tag,(int)strlen(rt->tag)):-1)
                                                            /* a pointer return stays a pointer (`f()->v`, `*f()`):
                                                            * a 4-byte unit truncated it (the oracle's _call_result_ct) */
      : (rt && rt->kind==3)             ? fp_ret_temp(c,rt)   /* a function pointer: that pointer (CF-FPRET) */
      : (rt && rt->is_complex)          ? tempc(c,rt->size)   /* _Complex user return (a float pair) */
      : (rt && rt->is_float)            ? tempf(c,rt->size)   /* float/double user return */
      : (rt && rt->kind==0 && rt->bit_width>0) ? tempbi(c,rt->bit_width,rt->signd)  /* a C23 `_BitInt(N)` return:
                                                            * keep the exact width (it does not promote; same-type
                                                            * arithmetic on the result must stay `_BitInt(N)`) */
      : (rt && rt->kind==0 && rt->size==8) ? tempi(c,8,rt->signd)  /* wide (8-byte) int return: keep its
                                                            * sign so a `>>` on a `long` result stays arithmetic */
      : (rt && rt->kind==0 && rt->size<=4 && rt->signd) ? tempi(c,4,1)  /* a signed char/short/int return
                                                            * promotes to int and sign-extends downstream (a
                                                            * `(long)` widen / compare); else it would go unsigned */
      : temp(c,4);                                            /* unsigned int / pointer / unknown -> 4-byte unit */
  char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.call:%.*s",name->n,name->s);
  bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
  if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=1;cl->wr[0]=t;}
  add_call(c,name);
  return t;
}
/* An indirect call through a function-pointer local/param (HAL dispatch): the target is dynamic, so
 * there is no named callee -- a `c.call.indirect` claim (reads: the pointer value then the actuals).
 * It is *not* added to fn->calls, so R18 leaves it an opaque external edge (no recursion/resolution). */
static uint32_t p_icall(CC *c, const venv *fv) {
  /* SNAPSHOT the funcptr's env entry: the actuals p_expr below can declare locals (a stmt-expr arg) and
   * realloc c->env[] -- `fv`, a pointer into it, would dangle before fv->type / fv->rid are read. */
  venv fvsnap=*fv; fv=&fvsnap;
  c->i++; /* '(' */
  uint32_t args[BCIR_CLAIM_MAX_RD]; int na=0, dropped=0;
  if(!is(c,")")) for(;;){ uint32_t a=p_expr(c); if(na<BCIR_CLAIM_MAX_RD-1)args[na++]=a; else dropped=1;
    if(is(c,",")){c->i++;continue;} break; }
  eat(c,")");
  null_pointer_sig(c,fv->type.fp_sig,args,na);   /* a null pointer argument: its parameter's pointer (CF-NULLCALL) */
  int vd=fv->type.fp_ret_void;              /* a pointer to a void function: no result (CF-VOIDCB) */
  uint32_t t=vd?0:fp_result_temp(c,&fv->type);   /* type by the funcptr's captured return -> a signed return reads back signed */
  bcir_claim *cl=new_claim(c,"c.call.indirect",BCIR_OP_GEM_DISPATCH);
  if(cl){cl->n_rd=(uint8_t)(na+1);cl->rd[0]=fv->rid;for(int k=0;k<na;k++)cl->rd[k+1]=args[k];
    cl->n_wr=(uint8_t)(vd?0:1);cl->wr[0]=t;cl->truncated=(uint8_t)dropped;}
  qcast_sig_call(c,cl,fv->type.fp_sig,na);   /* its qualified parameters and return (CF-QUALS) */
  return vd?void_temp(c):t;                 /* the void value, as a direct void call's (`return cb();`, `c ? cb() : ...`) */
}
#define CC_NOT_CALLABLE "called object is not a function or function pointer"   /* the oracle's `NOT_CALLABLE` */
/* A call through the function-pointer VALUE in `fv` (CF-FPTAB) -- an element of a table, a read through a pointer to one
 * or from a member, a select of them -- with the cursor on its `(`: a `c.call.indirect` like a call through a
 * function-pointer object's (p_icall), its result typed by the value's function type (`res_sig`). The oracle's
 * `_call_ptr` through `_call_through`. */
static uint32_t p_icall_rid(CC *c, uint32_t fv){
  int s=res_sig(c,fv);
  if(!s){ fail(c,CC_NOT_CALLABLE); return 0; }
  venv v; memset(&v,0,sizeof v); v.rid=fv; v.sidx=-1;
  v.type.kind=3; v.type.size=cc_abi(c)->pointer_size; v.type.fp_sig=(uint16_t)s;
  fits(c,v.type.tag,sizeof v.type.tag,"%s",c->sigs[s-1].alias);
  { const bcir_ctype *rt=&c->sigs[s-1].ret;
    fp_capture_ret(&v.type,rt,rt->kind==1 ? find_struct(c,rt->tag,(int)strlen(rt->tag)) : -1); }
  return p_icall(c,&v);
}
/* A postfix call on the value `r` (CF-FPTAB): `ops[i](x)`, `(*p)(x)`, `(ops[i])(x)`, `(c ? f : g)(x)` -- while a `(`
 * follows, `r` is called through and the call's value is the next `r`. A value that is no function pointer is refused,
 * as C refuses it (6.5.2.2p1), for the oracle's one reason. */
static uint32_t postfix_lvalue(CC *c, venv *v);                         /* fwd: `.`, `->`, `[` on a value */
static uint32_t call_value(CC *c, uint32_t r){
  while(is(c,"(") && !c->failed){
    const bcir_resource *rr=res_of(c->fn,r);
    if(!rr || !rr->is_funcptr || rr->kind!=BCIR_RK_SCALAR){ fail(c,CC_NOT_CALLABLE); return 0; }
    int s=res_sig(c,r);
    r=p_icall_rid(c,r);
    if(s && !c->failed && (is(c,".")||is(c,"->")||is(c,"["))){   /* `m(x).a`, `g(x)->v` (CF-FPRET) */
      bcir_ctype rt=c->sigs[s-1].ret;                 /* a copy: what follows may grow the table */
      return call_result(c,r,&rt); }
  }
  return r;
}
/* The postfix on the value `r` a call returned, the function's return type `rt` (CF-FPRET): a struct or union value is
 * addressable -- `mk(x).f`, and `m(x).f` through a function pointer, which the twin refused -- and a pointer takes `->`
 * and `[`; then any call through what that yields. Anything else goes on as `call_value`. */
static uint32_t call_result(CC *c, uint32_t r, const bcir_ctype *rt){
  if(rt && rt->kind==1 && (is(c,".")||is(c,"->")||is(c,"["))){   /* the by-value struct result is addressable */
    venv sv; memset(&sv,0,sizeof sv); sv.rid=r; sv.type=*rt;
    sv.sidx=find_struct(c,rt->tag,(int)strlen(rt->tag));
    if(sv.sidx>=0) return call_value(c,postfix_lvalue(c,&sv));
  }
  if(rt && rt->kind==2 && (is(c,"->")||is(c,"["))){   /* a returned pointer: `f()->v` / `f()[i]` */
    venv sv; memset(&sv,0,sizeof sv); sv.rid=r; sv.type=*rt;
    sv.sidx=rt->ptr_to_struct?find_struct(c,rt->tag,(int)strlen(rt->tag)):-1;
    return call_value(c,postfix_lvalue(c,&sv));
  }
  return call_value(c,r);
}

/* §5.8: GCC/Clang atomic + fence + CAS builtins -> the BCIR ATOMIC_x / BARRIER / CMPXCHG opcodes.
 * kind: 0 = RMW (ptr,val), 1 = fence (no operands), 2 = cmpxchg (ptr,expected,desired). */
enum { AK_RMW=0, AK_FENCE=1, AK_CAS=2, AK_LOAD=3, AK_STORE=4 };
static int atomic_kind(const tok *t,const char **op,bcir_opcode *oc,int *kind){
  struct{const char *n,*op;bcir_opcode oc;int k;} A[]={
    {"__atomic_fetch_add","c.atomic.add",BCIR_OP_ATOMIC_ADD,AK_RMW},
    {"__atomic_fetch_sub","c.atomic.sub",BCIR_OP_ATOMIC_SUB,AK_RMW},
    {"__atomic_fetch_xor","c.atomic.xor",BCIR_OP_ATOMIC_XOR,AK_RMW},
    {"__atomic_thread_fence","c.fence",BCIR_OP_BARRIER,AK_FENCE},   /* SEG6.1/SEG7: order-taking (routes by arg) */
    {"atomic_thread_fence","c.fence",BCIR_OP_BARRIER,AK_FENCE},      /* C11 <stdatomic.h> -- order-parameterized */
    {"__sync_synchronize","c.fence",BCIR_OP_BARRIER,AK_FENCE},
    {"_mm_mfence","c.fence",BCIR_OP_BARRIER,AK_FENCE},               /* x86 mfence -- full (load+store) fence */
    {"_mm_lfence","c.fence.acquire",BCIR_OP_BARRIER,AK_FENCE},       /* x86 lfence -- load (acquire) fence */
    {"_mm_sfence","c.fence.release",BCIR_OP_BARRIER,AK_FENCE},       /* x86 sfence -- store (release) fence */
    {"__sync_val_compare_and_swap","c.cmpxchg.val",BCIR_OP_CMPXCHG,AK_CAS},
    {"__sync_bool_compare_and_swap","c.cmpxchg.bool",BCIR_OP_CMPXCHG,AK_CAS},
    {"atomic_fetch_add","c.c11atom.fetch_add",BCIR_OP_ATOMIC_ADD,AK_RMW},  /* C11 <stdatomic.h> */
    {"atomic_fetch_sub","c.c11atom.fetch_sub",BCIR_OP_ATOMIC_SUB,AK_RMW},
    {"atomic_fetch_xor","c.c11atom.fetch_xor",BCIR_OP_ATOMIC_XOR,AK_RMW},
    {"atomic_exchange","c.c11atom.exchange",BCIR_OP_ATOMIC_ADD,AK_RMW},   /* swap: set + return old */
    {"atomic_load","c.c11atom.load",BCIR_OP_LOAD,AK_LOAD},
    {"atomic_store","c.c11atom.store",BCIR_OP_STORE,AK_STORE},
    {"atomic_compare_exchange_strong","c.c11atom.cas_strong",BCIR_OP_CMPXCHG,AK_CAS},  /* (obj,&exp,des)->_Bool */
    {"atomic_compare_exchange_weak","c.c11atom.cas_weak",BCIR_OP_CMPXCHG,AK_CAS},{0,0,0,0}};
  for(int i=0;A[i].n;i++) if((int)strlen(A[i].n)==t->n&&!strncmp(A[i].n,t->s,t->n)){*op=A[i].op;*oc=A[i].oc;*kind=A[i].k;return 1;}
  return 0;
}
/* SEG7: resolve the order-taking fence's KIND from its FIRST `memory_order` ARGUMENT, mirroring the
 * oracle's `_fence_order_kind` EXACTLY. The cursor is positioned at the first arg token (just past `(`).
 * A bare integer literal -> its value; a bare, UNSHADOWED `memory_order_*` / `__ATOMIC_*` named constant
 * -> its mapped value; ANYTHING else (a non-constant expression, a parenthesized/cast order, or a name
 * shadowed by a declared variable/param/global or a same-named function) -> 5 (seq_cst, the FULL fence).
 * "Bare" == a single int/name token wrapped in any number of BALANCED redundant parens, then `,`/`)` --
 * mirroring the oracle, whose parser strips redundant parens, so `(memory_order_acquire)`, `((2))`, and a
 * macro-expanded `(memory_order_acquire)` all reduce to a bare IntLit/Name and resolve. A `(int)2` (Cast)
 * or `5+0` (Binary) is NOT a single bare token under balanced parens, so it folds to the full fence exactly
 * as the oracle does. The shadow precedence is env (lookup/global) -> func (callee_ret) -> the constant,
 * identical to the rvalue widening, so the kind rail never disagrees with the value rail. Side-effect-free:
 * it only PEEKS (no claim, no cursor move). */
static const char *fence_order_op(CC *c){
  /* strip the TRANSPARENT prefixes: balanced redundant parens, which the oracle's parser drops, AND a leading unary
   * `+`, which keeps the constant's value and which the oracle's `_fence_order_kind` peels (`-`/`~`/`!`/a cast fold
   * to the full fence). Only a `(` needs a matching `)`; a `+` is closer-less. */
  int p=0, i=c->i;
  while(tok_is(tat(c,i),"(") || tok_is(tat(c,i),"+")){ if(tok_is(tat(c,i),"(")) p++; i++; }
  const tok *core=tat(c,i);
  /* EXACTLY p closing parens must follow the core token (balancing the leading ones) -- NOT all consecutive
   * `)` (that would also swallow the call's own closing paren and reject `(memory_order_acquire)`). */
  int closed=1; for(int j=0;j<p;j++) if(!tok_is(tat(c,i+1+j),")")){ closed=0; break; }
  const tok *end=tat(c,i+1+p);                             /* the token right after the p redundant closers */
  int bare = closed && (tok_is(end,",")||tok_is(end,")")); /* a single core token under p balanced parens, then arg-end */
  if(!bare) return "c.fence";                              /* a cast / binary / multi-token order -> full */
  if(core->k==T_INT) return order_kind(core->v);
  if(core->k==T_ID){
    /* match p_primary's value-rail precedence EXACTLY (find_enum -> lookup/global -> func -> memory_order),
     * so the kind rail never disagrees with the value rail or the oracle: an ENUM constant folds to its own
     * value (like the oracle parser's IntLit), a local/param/global/function is a runtime value (-> full),
     * and only THEN does the builtin memory_order name map. */
    int ec=visible_enum(c,core->s,core->n);
    if(ec>=0) return order_kind(c->ec[ec].val);
    if(lookup(c,core) || find_global(c,core->s,core->n)>=0 || callee_ret(c,core)) return "c.fence";
    long long mo;
    if(memorder_value(core,&mo)) return order_kind(mo);
  }
  return "c.fence";                                        /* non-constant / shadowed / unknown -> full */
}
/* A temp of the value type of what pointer `ptr` points at (the oracle's `_atomic_value_type`): its pointee,
 * unqualified, when it is a pointer (depth 1) to a scalar or an array of scalars; else the 4-byte temp it was. */
static uint32_t atomic_value_temp(CC *c,uint32_t ptr){
  bcir_resource p;
  if(!res_copy(c,ptr,&p)) return temp(c,4);
  int ptr1 = p.kind==BCIR_RK_POINTER && (p.ptr_depth?p.ptr_depth:1)==1;
  int arr = p.kind==BCIR_RK_SCALAR && (p.is_array || p.count>1) && !p.ptr_depth;
  if(!(ptr1||arr) || p.agg[0] || p.is_voidptr || p.is_funcptr) return temp(c,4);
  bcir_ctype t; memset(&t,0,sizeof t);
  t.size=(int)p.elem_bytes; t.signd=p.is_signed; t.is_float=p.is_float; t.is_complex=p.is_complex;
  t.is_bool=p.is_bool; t.is_plain_char=p.is_plain_char; t.bit_width=p.bit_width;
  return ctype_value_temp(c,&t);
}
static uint32_t p_atomic(CC *c,const char *op,bcir_opcode oc,int kind,int ordered){
  c->i++; uint32_t args[BCIR_CLAIM_MAX_RD]; int na=0, dropped=0;
  /* SEG7: an order-taking fence (`__atomic_thread_fence`/`atomic_thread_fence`) routes its KIND by the
   * first arg's order value -- peeked HERE, BEFORE the arg is lowered, so the arg's const claim (the
   * value rail) is still emitted in sequence, exactly as the oracle does (digest = [const, fence]). */
  if(ordered && kind==AK_FENCE && !is(c,")")) op=fence_order_op(c);
  if(!is(c,")")) for(;;){uint32_t a=p_expr(c);if(na<BCIR_CLAIM_MAX_RD)args[na++]=a;else dropped=1;
    if(is(c,",")){c->i++;continue;}break;}
  eat(c,")");
  /* the value an atomic reads through its pointer -- atomic_load, a fetch-op, atomic_exchange, a value CAS --
   * has the pointee's type, unqualified (the oracle's `_atomic_value_type`): a uint32 temp truncated a 64-bit
   * counter and converted an `_Atomic float` to an integer (CF-ATOMIC). The bool forms and a fence stay 4-byte. */
  int valued = na>=1 && (kind==AK_RMW || kind==AK_LOAD || !strcmp(op,"c.cmpxchg.val"));
  uint32_t t = valued ? atomic_value_temp(c,args[0]) : temp(c,4);
  if(!strncmp(op,"c.c11atom.cas",13) && c->fn->n_res) c->fn->res[c->fn->n_res-1].is_bool=1;  /* compare_exchange -> _Bool */
  bcir_claim *cl=new_claim(c,op,oc); if(!cl)return t;
  cl->truncated=(uint8_t)dropped;
  cl->lane=BCIR_LANE_A; cl->hazard=kind==AK_FENCE?BCIR_HZ_BARRIERED:BCIR_HZ_ATOMIC;
  if(kind!=AK_FENCE&&na>=1){ bcir_domain dom=BCIR_DOM_RAM;
    for(size_t z=0;z<c->fn->n_res;z++) if(c->fn->res[z].rid==args[0]) dom=c->fn->res[z].domain;
    cl->domain=dom; cl->rd[0]=args[0];
    if(kind==AK_LOAD){ cl->n_rd=1; cl->n_wr=1; cl->wr[0]=t; }              /* atomic_load(p) */
    else if(kind==AK_STORE){ cl->n_rd=2; cl->rd[1]=(na>1)?args[1]:args[0]; }   /* atomic_store(p,v) */
    else if(kind==AK_CAS){ cl->n_wr=1; cl->wr[0]=t;                       /* CMPXCHG: ptr, exp, des */
      cl->n_rd=3; cl->rd[1]=(na>1)?args[1]:args[0]; cl->rd[2]=(na>2)?args[2]:cl->rd[1]; }
    else { cl->n_wr=1; cl->wr[0]=t; cl->n_rd=2; cl->rd[1]=(na>1)?args[1]:args[0]; }   /* RMW: ptr, val */
  }
  return t;
}

/* Concatenate adjacent string-literal tokens (C translation phase 6) into one spelling whose pieces
 * stay adjacent (separated by a space), so a hex/octal escape never merges with the next piece's
 * leading digit. Returns a malloc'd NUL-terminated buffer (caller frees); *out_n is its length. */
static char *gather_strings(CC *c, tok first, int *out_n) {
  size_t total, len;
  int cursor;
  char *buf;
  if(first.n<0 || !bcir_size_add((size_t)first.n,1u,&total)){
    cc_raise_oom(c);*out_n=0;return NULL;
  }
  cursor=c->i;
  while(tat(c,cursor)->k==T_STR){
    const tok *next=tat(c,cursor++);
    if(next->n<0 || !bcir_size_add(total,(size_t)next->n,&total) ||
       !bcir_size_add(total,1u,&total)){
      cc_raise_oom(c);*out_n=0;return NULL;
    }
  }
  buf=(char*)bcir_host_arena_allocate(&c->scratch,total,1u);
  if(!buf){cc_raise_oom(c);*out_n=0;return NULL;}
  memcpy(buf,first.s,(size_t)first.n); len=(size_t)first.n;
  while(isk(c,T_STR)){ tok nx=adv(c);
    buf[len++]=' '; memcpy(buf+len,nx.s,(size_t)nx.n); len+=nx.n; }
  buf[len]=0; *out_n=(int)len; return buf;
}

/* Apply postfix `.field` / `->field` (incl. nested `o.in.v`, a funcptr-member call `o->fn(args)`, a
 * deref-through a loaded pointer field, and a member array `s.arr[i]`) and `[i]` subscripts to an
 * already-resolved lvalue base `v`. Shared by the identifier primary and the compound-literal path -- a
 * synthesized base (rid + struct type + sidx) lets `(struct P){...}.field` read like any struct base. */
static uint32_t postfix_lvalue(CC *c, venv *v){
  /* SNAPSHOT the env entry: the member-funcptr-call args, the `[idx]` subscript, and member_arr_index all
   * re-enter the expression grammar (p_expr), which can declare locals and realloc c->env[] -- the incoming
   * `v`, a pointer into that array, would dangle. The emit/store helpers only READ the venv (by-value identical). */
  venv vsnap=*v; v=&vsnap;
  if(is(c,".")||is(c,"->")){
    int arrow=is(c,"->");
    if(!member_base_ok(c,v,arrow)){ fail(c,arrow?CC_ARROW_NOT:CC_DOT_NOT); return 0; }   /* (CF-FPTAB) */
    c->i++; tok fn=adv(c); sdef *S=&c->s[v->sidx]; int fi=-1;
    for(int i=0;i<S->nf;i++) if((int)strlen(S->f[i].name)==fn.n&&!strncmp(S->f[i].name,fn.s,fn.n)) fi=i;
    if(fi<0){fail(c,"unknown field");return 0;}
    if(is(c,"(")){     /* o->fnptr(args): fused indirect call via a funcptr struct member */
      c->i++; uint32_t args[BCIR_CLAIM_MAX_RD]; int na=0, dropped=0;
      if(!is(c,")")) for(;;){ uint32_t a=p_expr(c); if(na<BCIR_CLAIM_MAX_RD-1)args[na++]=a; else dropped=1;
        if(is(c,",")){c->i++;continue;} break; }
      eat(c,")");
      field ff=S->f[fi];                          /* the funcptr field carries its captured return type */
      null_pointer_sig(c,ff.fp_sig,args,na);      /* a null pointer argument: its parameter's pointer */
      uint32_t t = field_call_temp(c,&ff);        /* typed by the member's function type (CF-FPRET) */
      char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.call.imember:%s",S->f[fi].name);
      bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
      if(cl){cl->n_rd=(uint8_t)(na+1);cl->rd[0]=v->rid;for(int k=0;k<na;k++)cl->rd[k+1]=args[k];
        cl->n_wr=(uint8_t)(ff.fp_ret_void?0:1);cl->wr[0]=t;cl->n_imm=1;cl->imm[0]=arrow;cl->truncated=(uint8_t)dropped;}
      qcast_sig_call(c,cl,ff.fp_sig,na);          /* its qualified parameters and return (CF-QUALS) */
      return ff.fp_ret_void ? void_temp(c) : member_call_result(c,&ff,t);
    }
    field mf=member_descend(c,S->f[fi]);        /* nested `o.in.v` -> one flattened-offset load */
    if(is(c,"[") && !mf.arr_count && !mf.is_ptr){ fail(c,CC_SUBSCRIPT_NOT); return 0; }   /* `o->v[1]` (CF-FPTAB) */
    if(mf.is_ptr && is(c,".")){ fail(c,CC_DOT_NOT); return 0; }   /* `o.p.x` through a pointer member */
    if(mf.is_ptr && (is(c,"->")||is(c,".")||is(c,"["))){   /* deref-through a loaded pointer field (#fieldderef) */
      uint32_t ptr=emit_member(c,v,&mf,0);      /* load the pointer field, then chain through the loaded ptr */
      return postfix_ptr_chain(c,ptr,mf.ptee_sidx,mf); }
    if(mf.arr_count && is(c,"[")){ uint32_t ix=member_arr_index(c,&mf);   /* s.arr[i] / s.m[i][j] load */
      field sub; if(elem_field(c,&mf,&sub)) return emit_member_index_field(c,v,&mf,ix,&sub);   /* arr[i].field */
      if(c->failed) return 0;
      return emit_member_index(c,v,&mf,ix); }
    if(mf.arr_count) return member_array_value(c,v,&mf);   /* an array member value: its address (CF-DECAY) */
    return emit_member(c,v,&mf,c->stmt_expr_declared_bf);
  }
  if(is(c,"[")){                                /* L3: base[i] / m[i][j] (row-major flatten) / q[j][i] (a chain) */
    { field gf; if(global_md_field(c,v,&gf)){       /* g[i][j] of a multi-dimensional global (CF-SMALL) */
        uint32_t ix=member_arr_index(c,&gf); if(c->failed) return 0;
        return emit_member_index(c,v,&gf,ix); } }
    uint32_t lin=index_chain(c,v);
    if(c->failed) return 0;
    if(v->sidx>=0 && (is(c,".")||is(c,"->"))){    /* a[i].field on a DIRECT array-of-structs (strided load) */
      { field af; uint32_t tot; int r=aos_member_array(c,v,lin,&af,&tot);   /* a[i].m[j] (CF-SMALL) */
        if(r<0) return 0;
        if(r>0) return emit_member_index(c,v,&af,tot); }
      field sub; if(aos_elem_field(c,v,&sub)) return emit_index_field(c,v,lin,&sub);
      if(c->failed) return 0; }
    return emit_index(c,v,lin);
  }
  return named_read(c,v);                        /* the value; a volatile object's read is a volatile access */
}
/* `_Generic(ctrl, T1: e1, ..., default: eN)` (C11 §6.5.1.1): the controlling expr is UNEVALUATED -- its
 * static type (read via the speculative typeof machinery, then rolled back) selects the association. The
 * first type-name whose type matches wins, else `default`; only the chosen association's expression is
 * lowered. The two-pass shape (scan all arms to find the chosen token offset, then lower just that one)
 * fits the twin's no-AST, lower-while-parsing model -- each scanned expr is parsed then rolled back. */
/* `_Generic(p, const char *: a, char *: b)`: an association of a qualified type, which this front end does not tell
 * from the unqualified one -- the type of the controlling expression carries no qualifier -- so `p` of `char *`
 * selected `a` (CF-QUALS). The oracle's `GENERIC_QUALIFIED`. */
#define CC_GENERIC_QUALIFIED "a `_Generic` association of a qualified type is not supported"
static uint32_t p_generic(CC *c){
  c->i++;                                              /* _Generic */
  if(!eat(c,"("))return 0;
  bcir_ctype ctrl; int cs; if(p_typeof_expr(c,&ctrl,&cs)) return 0;   /* the controlling type (rolled back) */
  if(!eat(c,","))return 0;
  int sel_at=-1, def_at=-1;
  while(!is(c,")")&&!isk(c,T_END)&&!c->failed){
    int is_def=0; bcir_ctype lty; int lsi=-1;
    if(is(c,"default")){ c->i++; is_def=1; }
    else if(p_type(c,&lty,&lsi)) return 0;             /* a type-name label */
    if(!is_def && (lty.is_const||lty.is_volatile||lty.is_atomic||lty.ptr_const||lty.ptr_restrict)){
      fail(c,CC_GENERIC_QUALIFIED); return 0; }        /* a qualified one (CF-QUALS) */
    if(!eat(c,":"))return 0;
    int expr_at=c->i;                                  /* this association's expression starts here */
    bcir_ctype dump; int ds; if(p_typeof_expr(c,&dump,&ds)) return 0;   /* consume + roll back (not lowered) */
    if(is_def) def_at=expr_at;
    else if(sel_at<0 && ctype_generic_eq(&ctrl,&lty)) sel_at=expr_at;
    if(is(c,",")) c->i++;
  }
  if(!eat(c,")"))return 0;
  int after=c->i;
  if(sel_at<0) sel_at=def_at;
  if(sel_at<0){ fail(c,"no _Generic association matches the controlling type"); return 0; }
  c->i=sel_at; uint32_t r=p_expr(c); c->i=after;       /* lower ONLY the chosen association, for real */
  return r;
}
/* --- sizeof (CF-SIZEOF) ------------------------------------------------------------------------------------ *
 * `sizeof` measures its operand's own type, unevaluated (C11 6.5.3.4p2): an array is its whole array -- a row
 * of a multi-dimensional one a row, a member array its member -- never the pointer it decays to anywhere else,
 * and the result is a `size_t`, the ABI's pointer-wide unsigned integer (p5). The oracle types the operand's
 * AST (`_sizeof_type`). The twin has no AST, so a static walk reads the designator forms -- a name, `[...]`,
 * `.m`, `->m`, `*`, `&`, parentheses, a string literal -- off the tokens WITHOUT lowering them, and any other
 * operand (an operator, a call, a cast, a literal) is lowered speculatively for its value's type and rolled
 * back, as `typeof` is. A walk step returns 1 (typed), 0 (not a designator: lower it) or -1 (refused). */
typedef struct {
  bcir_ctype ty;      /* the element type: a scalar, a struct/union (`si`), a pointer or a function pointer */
  int si;             /* the struct index of a struct value or of a pointer's pointee struct, else -1 */
  int nd;             /* the array dimensions around it, outermost first (0: not an array) */
  long long dims[4];  /* 0: an unsized (incomplete) dimension */
  int addr;           /* `&` levels over the whole: a pointer to it */
  int bf;             /* a bit-field member -- never an operand of `sizeof` (C11 6.5.3.4p1) */
  uint32_t vla;       /* the rid of a stack VLA the operand names bare: its size is a runtime value */
} szt;

static uint32_t p_unary(CC *c);   /* fwd: an operand the walk does not type is lowered speculatively */

/* The largest size an object may have on the target (PTRDIFF_MAX): a type past it has no objects, and Clang
 * refuses it ("array is too large"). The oracle refuses the same bound. */
static long long size_limit(const CC *c){
  int bits=8*cc_abi(c)->pointer_size;
  return bits>=64 ? LLONG_MAX : ((long long)1<<(bits-1))-1;
}
/* sizeof of `t`: the element's layout under the target ABI times every dimension -- -1 for an incomplete type
 * (an unsized dimension, `void`, a struct without a body), -2 for one larger than the target allows. */
static long long szt_bytes(CC *c, const szt *t){
  if(t->addr) return cc_abi(c)->pointer_size;
  int sz, al; type_layout(c,&t->ty,t->si,&sz,&al);
  if(t->ty.kind==3) sz=cc_abi(c)->pointer_size;
  long long n=sz, lim=size_limit(c);
  if(n<=0) return -1;
  for(int d=0; d<t->nd; d++){
    if(t->dims[d]<=0) return -1;
    if(n>lim/t->dims[d]) return -2;
    n*=t->dims[d];
  }
  return n;
}
/* A struct member's type as `sizeof` sees it (the oracle's `agg.field` type): a pointer or an array of them
 * carries its pointee, a nested struct its index, a member array its dimensions. */
static void szt_field(CC *c, szt *t, const field *f){
  memset(t,0,sizeof *t); t->si=-1;
  bcir_ctype *e=&t->ty;
  if(f->is_ptr || f->elem_ptr){
    e->kind=2; e->size=f->ptee_size; e->signd=f->signd; e->is_float=(uint8_t)(f->ptee_float?1:0);
    e->is_atomic=(uint8_t)(f->ptee_atomic?1:0); e->ptr_depth=(uint8_t)(f->ptee_depth>0?f->ptee_depth:1);
    if(f->ptee_sidx>=0){ e->ptr_to_struct=1; t->si=f->ptee_sidx; }
  } else if(f->sidx>=0 || f->elem_sidx>=0){
    e->kind=1; t->si=f->sidx>=0?f->sidx:f->elem_sidx; e->size=c->s[t->si].size;
  } else {                                        /* a scalar: `size` is its laid-out width, `_Atomic` included */
    e->kind=0; e->size=f->size; e->signd=f->signd; e->is_float=(uint8_t)(f->is_float?1:0);
    e->is_complex=(uint8_t)(f->is_complex?1:0); e->is_bool=(uint8_t)(f->is_bool?1:0);
    e->is_plain_char=(uint8_t)(f->is_plain_char?1:0); e->bit_width=f->bit_width;
  }
  if(f->arr_count){
    int n=f->nadims>0?f->nadims:1; if(n>3) n=3;
    for(int d=0; d<n; d++) t->dims[d]=f->nadims>0?f->adims[d]:f->arr_count;
    t->nd=n;
  }
  t->bf=f->bit_w>0;
}
/* `.m` of `t` (the cursor is at the member name): a member of a struct or union value. */
static int sz_member(CC *c, szt *t){
  if(!isk(c,T_ID) || t->addr || t->nd || t->bf || t->ty.kind!=1 || t->si<0) return 0;
  tok fn=adv(c); const sdef *S=&c->s[t->si];
  for(int i=0;i<S->nf;i++)
    if((int)strlen(S->f[i].name)==fn.n && !strncmp(S->f[i].name,fn.s,fn.n)){ szt_field(c,t,&S->f[i]); return 1; }
  return 0;
}
/* One subscript or `*` of `t` (the oracle's `_sizeof_elem`): an array loses its outermost dimension; a pointer
 * gives its pointee -- a row again for a decayed `T m[][N]` or a `T (*m)[N]` (its `adims`). */
static int szt_elem(szt *t){
  if(t->addr){ t->addr--; return 1; }
  if(t->bf) return 0;
  if(t->nd){ for(int d=1; d<t->nd; d++) t->dims[d-1]=t->dims[d]; t->nd--; return 1; }
  bcir_ctype *e=&t->ty;
  if(e->kind!=2) return 0;
  int depth=e->ptr_depth?e->ptr_depth:1;
  if(depth>1){ e->ptr_depth=(uint8_t)(depth-1); e->nadims=0; memset(e->adims,0,sizeof e->adims); return 1; }
  for(int d=1; d<e->nadims && d<3; d++) t->dims[t->nd++]=e->adims[d];
  e->nadims=0; memset(e->adims,0,sizeof e->adims); e->ptr_depth=0;
  if(e->ptr_to_struct){ e->ptr_to_struct=0; e->kind=1; } else e->kind=0;
  return 1;
}
/* A name: a local or a parameter (its env entry and resource: an array's dimensions, a VLA's runtime size), or
 * a file-scope object (its declaration, every dimension). An enumerator, a constant or an undeclared name is
 * lowered; a function designator is refused (C11 6.5.3.4p1). */
static int sz_name(CC *c, szt *t){
  tok id=*pk(c);
  if(tok_is(&id,"sizeof") || tok_is(&id,"_Alignof") || tok_is(&id,"alignof") || tok_is(&id,"_Generic")
     || visible_enum(c,id.s,id.n)>=0) return 0;    /* an operator or an enumerator: lowered */
  venv *v=lookup(c,&id);
  const bcir_resource *r=v?res_of(c->fn,v->rid):NULL;
  int gi=(v && !(r && r->read_only)) ? -1 : find_global(c,id.s,id.n);
  const bcir_ctype *oty = (v && !(r && r->read_only)) ? &v->type : gi>=0 ? &c->gv[gi].ty : NULL;
  if(tok_is(tat(c,c->i+1),"(")){                    /* a call: its declared return type, the arguments skipped */
    bcir_ctype rt;
    if(oty && oty->kind==3){                         /* through a function-pointer object: its captured return */
      memset(&rt,0,sizeof rt); rt.size=oty->fp_ret_size; rt.signd=oty->fp_ret_signd; rt.is_float=oty->fp_ret_float;
    } else if(oty) return 0;                         /* a call of an object that is no function: lowered (and refused) */
    else {
      const bcir_ctype *cr=declared_ret(c,&id);     /* a prototype declares it as a definition does */
      if(!cr){ fail(c,"sizeof of a call to a function whose return type is unknown"); return -1; }
      rt=*cr;
    }
    memset(t,0,sizeof *t); t->ty=rt; t->si=-1;
    if(rt.kind==1 || rt.ptr_to_struct) t->si=find_struct(c,rt.tag,(int)strlen(rt.tag));
    c->i++;
    int d=0;
    do{ if(is(c,"(")) d++; else if(is(c,")")) d--; c->i++; }while(d>0 && !isk(c,T_END));
    return d ? 0 : 1;
  }
  if(v && !(r && r->read_only)){
    c->i++; memset(t,0,sizeof *t); t->ty=v->type; t->si=v->sidx;
    if(r && (r->is_vla || r->is_array)){
      int nd=r->is_vla ? (r->vla_ndims?r->vla_ndims:1) : (v->type.nadims>1?v->type.nadims:1);
      if(nd>3) nd=3;
      for(int d=0; d<nd; d++)
        t->dims[d]=r->is_vla ? 0 : v->type.nadims>1 ? v->type.adims[d] : (long long)r->count;
      t->nd=nd;
      for(int k=0; k<c->n_unsized; k++) if(c->unsized[k]==v->rid) t->dims[0]=0;
      if(r->is_vla) t->vla=v->rid;
      t->ty.nadims=0; memset(t->ty.adims,0,sizeof t->ty.adims);
    }
    return 1;
  }
  if(gi<0){
    if(declared_ret(c,&id)){ fail(c,CC_SIZEOF_FN); return -1; }   /* a prototyped one too (CF-EXTDESIG) */
    return 0;
  }
  const gvar *g=&c->gv[gi];
  c->i++; memset(t,0,sizeof *t); t->ty=g->ty; t->si=-1;
  if(g->ty.kind==1 || g->ty.ptr_to_struct) t->si=find_struct(c,g->ty.tag,(int)strlen(g->ty.tag));
  if(g->is_arr){
    if(g->nd>4){ fail(c,"sizeof of an array of more than four dimensions"); return -1; }
    t->nd=g->nd; for(int d=0; d<g->nd; d++) t->dims[d]=g->dims[d];
    t->ty.nadims=0; memset(t->ty.adims,0,sizeof t->ty.adims);
  }
  return 1;
}
/* An assignment operator at the cursor: `=` (not `==`) or a compound `+=` ... `>>=`. */
static int is_assign_op(CC *c){
  const tok *t=pk(c);
  return t->k==T_PUN && ((t->n==1 && t->s[0]=='=') || is_compound_op(t));
}
static int sz_unary(CC *c, szt *t);
/* Whether the parenthesized type name whose first token is at `k` is a compound literal's: the index of its `)` when a
 * `{` follows it, else 0 (CF-RTFP). Nothing is consumed. */
static int literal_close(CC *c, int k){
  for(int d=0; tat(c,k)->k!=T_END; k++){
    const tok *tk=tat(c,k);
    if(tok_is(tk,"(") || tok_is(tk,"[")) d++;
    else if(tok_is(tk,")") || tok_is(tk,"]")){ if(!d) break; d--; }
  }
  return tok_is(tat(c,k),")") && tok_is(tat(c,k+1),"{") ? k : 0;
}
/* `(type-name){...}` at the cursor, past its `(`, under `sizeof`: a compound literal, of the type name's type -- an
 * array's whole, as `sizeof` reads an array object (C11 6.5.2.5p4, 6.3.2.1p3), where the lowered literal had given the
 * size of the pointer it decays to (CF-RTFP). Its initializer is skipped, never lowered; one whose size the initializer
 * gives, `(T[]){...}`, keeps its unsized dimension, which `sizeof` refuses as the oracle refuses it. 0 for a cast, which
 * is lowered (nothing consumed); 1 with `*t`, the cursor past the `}`; -1 after a failure. */
static BCIR_NOINLINE int sz_literal(CC *c, szt *t){
  int k=literal_close(c,c->i);                       /* the type name's own `)`: a literal when a `{` follows it */
  if(!k) return 0;
  memset(t,0,sizeof *t); t->si=-1;
  if(p_cast_type(c,&t->ty,&t->si) || fp_name_dims(c,&t->ty,t->dims,&t->nd,4)) return -1;
  int btdd[3], btd=td_dims_of(c,btdd);   /* a typedef'd array: its dims follow the type name's own */
  while(is(c,"[")){                                  /* `[N]`, or `[]` the initializer sizes (kept unsized: 0) */
    c->i++; long long d=0;
    if(!is(c,"]")){ d=ce_value(c); if(c->failed) return -1; }
    if(!eat(c,"]")) return -1;
    if(t->nd>=4){ fail(c,"a type-name of more than four array dimensions"); return -1; }
    t->dims[t->nd++]=d; }
  for(int d=0; d<btd; d++){ if(t->nd>=4){ fail(c,"a type-name of more than four array dimensions"); return -1; }
    t->dims[t->nd++]=btdd[d]; }
  if(c->i!=k) return 0;
  if(t->ty.is_atomic && t->ty.kind!=2){ fail(c,CC_ATOMIC_LITERAL); return -1; }   /* an `_Atomic` object */
  c->i=k+1;                                          /* past the `)`: the initializer, unevaluated */
  for(int d=0; !isk(c,T_END); c->i++){
    if(is(c,"{")) d++;
    else if(is(c,"}") && --d==0){ c->i++; break; }
  }
  return c->failed ? -1 : 1;
}
static int sz_postfix(CC *c, szt *t){
  int r;
  if(isk(c,T_ID)) r=sz_name(c,t);
  else if(isk(c,T_STR)){                           /* a (concatenated) literal: its units, NUL included */
    tok st=adv(c); const char *sp=st.s; int sn=st.n;
    if(isk(c,T_STR)){ int cn; char *cb=gather_strings(c,st,&cn); if(cb){ sp=cb; sn=cn; } }
    memset(t,0,sizeof *t); t->si=-1; t->ty.kind=0; t->ty.size=str_elem_size(c,sp,sn);
    t->nd=1; t->dims[0]=(long long)str_bytes(sp,sn)+1; r=1;
  } else if(is(c,"(")){
    c->i++;
    if(is(c,"{")) return 0;                          /* a statement expression */
    if(starts_type_name(c)) r=sz_literal(c,t);       /* a compound literal; a cast is lowered (CF-RTFP) */
    else {
    int last=-1;                                     /* a comma expression: its value is the last operand's */
    for(int k=c->i, d=0; tat(c,k)->k!=T_END; k++){
      const tok *tk=tat(c,k);
      if(tok_is(tk,"(") || tok_is(tk,"[") || tok_is(tk,"{")) d++;
      else if(tok_is(tk,")") || tok_is(tk,"]") || tok_is(tk,"}")){ if(!d) break; d--; }
      else if(!d && tok_is(tk,",")) last=k;
    }
    if(last>=0) c->i=last+1;
    r=sz_unary(c,t);
    if(r!=1) return r;
    if(is_assign_op(c)){                             /* an assignment has its left operand's type (6.5.16p3) */
      if(t->nd || t->addr) return 0;
      for(int d=0; !isk(c,T_END); c->i++){           /* the right operand, unevaluated */
        if(is(c,"(") || is(c,"[") || is(c,"{")) d++;
        else if(is(c,")") || is(c,"]") || is(c,"}")){ if(!d) break; d--; }
      }
      t->bf=0; t->ty.is_atomic=0;
    }
    if(!is(c,")")) return 0;
    c->i++;
    if(last>=0){                                     /* the comma's value: unpromoted, an array decayed */
      t->bf=0; t->ty.is_atomic=0; t->vla=0;
      if(t->nd && !t->addr){ for(int d=1; d<t->nd; d++) t->dims[d-1]=t->dims[d]; t->nd--; t->addr=1; }
    }
    }
  } else return 0;
  if(r!=1) return r;
  for(;;){
    if(is(c,"[")){                                   /* unevaluated: the index is skipped, never lowered */
      int bd=0;
      do{ if(is(c,"[")) bd++; else if(is(c,"]")) bd--; c->i++; }while(bd>0 && !isk(c,T_END));
      if(bd || !szt_elem(t)) return 0;
    } else if(is(c,".")){ c->i++; if(!sz_member(c,t)) return 0; }
    else if(is(c,"->")){ c->i++; if(!szt_elem(t) || !sz_member(c,t)) return 0; }
    else if(is(c,"++") || is(c,"--")){              /* `x++`: the operand's declared type (6.5.2.4) */
      if(t->nd || t->addr || t->ty.kind==1) return 0;
      c->i++; t->bf=0; t->ty.is_atomic=0; t->vla=0;
      return 1;
    }
    else if(is(c,"(")) return 0;                     /* a call through an expression: lowered */
    else return 1;
    t->vla=0;
  }
}
/* Whether the `*`s at `k` name a function -- a function pointer object or a function, no postfix after it: `*fp`,
 * `**inc` (C11 6.5.3.2p4). */
static int star_names_fn(CC *c, int k){
  while(tok_is(tat(c,k),"*")) k++;
  int amp=tok_is(tat(c,k),"&"); if(amp) k++;        /* `sizeof *&f`: the function again (CF-EXTDESIG) */
  const tok *nm=tat(c,k), *nx=tat(c,k+1);
  if(nm->k!=T_ID || tok_is(nx,"(") || tok_is(nx,"[") || tok_is(nx,".") || tok_is(nx,"->")) return 0;
  const venv *v=lookup(c,nm); int gi=v ? -1 : find_global(c,nm->s,nm->n);
  if(amp) return !v && gi<0 && declared_ret(c,nm)!=NULL;   /* `&` of an object is a pointer to it, no function */
  return v ? (v->type.kind==3 && !names_object_ptr(c,v)) : gi>=0 ? (c->gv[gi].ty.kind==3 && !c->gv[gi].is_arr)
           : declared_ret(c,nm)!=NULL;              /* a function a prototype declares too (CF-EXTDESIG) */
}
static int sz_unary(CC *c, szt *t){
  if(ENTER_REC(c)){ LEAVE_REC(c); return -1; }
  int r;
  if(is(c,"++") || is(c,"--")){                     /* `++x`: the operand's declared type (6.5.3.1) */
    c->i++; r=sz_unary(c,t);
    if(r==1 && (t->nd || t->addr || t->ty.kind==1)) r=0;
    if(r==1){ t->bf=0; t->ty.is_atomic=0; t->vla=0; }
  }
  else if(is(c,"*") && star_names_fn(c,c->i)){ fail(c,CC_SIZEOF_FN); r=-1; }   /* `sizeof *fp` (CF-FPTAB) */
  else if(is(c,"*")){ c->i++; r=sz_unary(c,t); if(r==1 && !szt_elem(t)) r=0; if(r==1) t->vla=0; }
  else if(is(c,"&") && tat(c,c->i+1)->k==T_ID && !lookup(c,tat(c,c->i+1)) &&
          find_global(c,tat(c,c->i+1)->s,tat(c,c->i+1)->n)<0 && declared_ret(c,tat(c,c->i+1)) &&
          !tok_is(tat(c,c->i+2),"(") && !tok_is(tat(c,c->i+2),"[")){   /* `&f`: a pointer to the function */
    c->i+=2; memset(t,0,sizeof *t); t->si=-1; t->addr=1; r=1; }
  else if(is(c,"&")){ c->i++; r=sz_unary(c,t); if(r==1 && t->bf) r=0; if(r==1){ t->addr++; t->vla=0; } }
  else r=sz_postfix(c,t);
  LEAVE_REC(c); return r;
}
/* The size of a value the speculative lowering produced: a pointer or a decayed array is the ABI's pointer, a
 * struct or union its laid-out size, a scalar its width. */
static long long spec_value_size(CC *c, uint32_t v){
  const bcir_resource *r=res_of(c->fn,v);
  if(!r) return -1;
  if(r->kind==BCIR_RK_POINTER || r->is_array || r->is_vla || r->is_funcptr || r->is_voidptr)
    return cc_abi(c)->pointer_size;
  if(r->kind==BCIR_RK_AGGREGATE){
    const char *sp=strchr(r->agg,' '); const char *tg=sp?sp+1:r->agg;
    int si=find_struct(c,tg,(int)strlen(tg));
    return si>=0 ? c->s[si].size : (long long)r->elem_bytes;
  }
  return (long long)r->elem_bytes;
}
/* A folded `sizeof` / `_Alignof`: a `c.const` into a `size_t` temp. */
static uint32_t size_result(CC *c, long long v){
  uint32_t r=tempi(c,cc_abi(c)->pointer_size,0);
  bcir_claim *cl=new_claim(c,"c.const",BCIR_OP_LOAD);
  if(cl){ cl->n_wr=1; cl->wr[0]=r; cl->n_imm=1; cl->imm[0]=v; }
  return r;
}
/* A type-name's array dimensions `[N]...` after its specifier and `*`s (an abstract declarator): the type is
 * an array of what precedes them (`sizeof(uint32_t[10])`, `sizeof(uint8_t *[5])`). */
static int type_name_dims(CC *c, szt *t){
  while(is(c,"[")){
    c->i++; long long d=ce_value(c);
    if(!eat(c,"]")) return 1;
    if(t->nd>=4){ fail(c,"a type-name of more than four array dimensions"); return 1; }
    t->dims[t->nd++]=d;
  }
  return c->failed;
}
static uint32_t p_sizeof(CC *c){
  c->i++;                                               /* sizeof */
  szt t; long long size;
  memset(&t,0,sizeof t); t.si=-1;
  if(is(c,"(")){
    int save=c->i; c->i++;
    if(starts_decl_type(c) && !literal_close(c,c->i)){  /* sizeof ( type-name ) -- not `sizeof (T[N]){...}`, a compound
                                                         * literal's unary-expression form (CF-RTFP) */
      if(p_cast_type(c,&t.ty,&t.si) || fp_name_dims(c,&t.ty,t.dims,&t.nd,4)) return 0;   /* `sizeof(R (*)(P))`, and
                                                         * of an array of them (CF-RTFP) */
      int btdd[3], btd=td_dims_of(c,btdd);   /* a typedef'd array */
      if(type_name_dims(c,&t)) return 0;
      for(int d=0; d<btd; d++){ if(t.nd>=4){ fail(c,"a type-name of more than four array dimensions"); return 0; }
        t.dims[t.nd++]=btdd[d]; }
      if(!eat(c,")")) return 0;
      size=szt_bytes(c,&t);
      goto fold;
    }
    c->i=save;
  }
  {
    int at=c->i, r=sz_unary(c,&t);                      /* sizeof unary-expression */
    if(r<0) return 0;
    if(r==1){
      if(t.bf){ fail(c,"sizeof of a bit-field"); return 0; }
      if(t.vla && !t.addr){                             /* a bare stack VLA: its RUNTIME size, the snapshot extent
                                                         * times the element (is_vla -- not merely a recovered
                                                         * extent, which a malloc'd pointer also has) */
        const bcir_resource *vr=res_of(c->fn,t.vla); uint32_t ext=ptrext_get(c->fn,t.vla);
        if(vr && vr->is_vla && ext){
          long long eb=(long long)vr->elem_bytes;        /* read before tempi may grow the resource array */
          uint32_t rr=tempi(c,cc_abi(c)->pointer_size,0);
          bcir_claim *cl=new_claim(c,"c.sizeof.vla",BCIR_OP_ADD);   /* ADD: a cost hint (the emit carries the x) */
          if(cl){ cl->n_rd=1; cl->rd[0]=ext; cl->n_wr=1; cl->wr[0]=rr; cl->n_imm=1; cl->imm[0]=eb; }
          return rr;
        }
      }
      size=szt_bytes(c,&t);
    } else {                                            /* an operator, a call, a cast, a literal: its value */
      spec_mark m;
      c->i=at; spec_begin(c,&m);
      uint32_t v=p_unary(c);
      size=c->failed ? -1 : spec_value_size(c,v);
      spec_end(c,&m);
      if(c->failed) return 0;
    }
  }
fold:
  if(size==-2){ fail(c,"sizeof of a type too large for the target"); return 0; }
  if(size<=0){ fail(c,"sizeof of an incomplete type"); return 0; }
  return size_result(c,size);
}
/* `_Alignof ( type-name )`: the type's alignment -- an array's is its element's -- a folded `size_t`. */
static uint32_t p_alignof(CC *c){
  c->i++;
  if(!eat(c,"(")) return 0;
  szt t; memset(&t,0,sizeof t); t.si=-1;
  if(p_cast_type(c,&t.ty,&t.si) || fp_name_dims(c,&t.ty,t.dims,&t.nd,4) || type_name_dims(c,&t) || !eat(c,")"))
    return 0;                                           /* `_Alignof(R (*)(P))`, `_Alignof(R (*[N])(P))` (CF-RTFP) */
  int sz, al; type_layout(c,&t.ty,t.si,&sz,&al);
  return size_result(c,al);
}
/* A call of the name `id`, the cursor on its `(`: an atomic builtin, a call through a function-pointer object (p_icall)
 * or a direct named call (p_call), a struct or pointer result taking its postfix. `id(x)` -- and `(*id)(x)`, whose `*`
 * of a function names it again (CF-FPTAB). */
static uint32_t p_named_call(CC *c, tok id){
    const char *aop;bcir_opcode aoc;int akind;
    if(atomic_kind(&id,&aop,&aoc,&akind)){    /* atomics/fences/CAS */
      int ordered = (id.n==21 && !strncmp("__atomic_thread_fence",id.s,21))   /* SEG6.1/SEG7: the order-taking */
                 || (id.n==19 && !strncmp("atomic_thread_fence",id.s,19));    /* fence forms route by their arg */
      return p_atomic(c,aop,aoc,akind,ordered);
    }
    { venv *fv=lookup(c,&id);        /* indirect call (funcptr var) vs. direct named call */
      if(fv&&fv->type.kind==3){       /* `fp(x)(y)`: through its value (CF-FPTAB); `fp(x).f`, `fp(x)->v` (CF-FPRET) */
        int s=fv->type.fp_sig;        /* read before the call: its arguments may grow `env` */
        uint32_t r=p_icall(c,fv);
        if(s>0 && s<=c->nsig && !c->failed){ bcir_ctype rt=c->sigs[s-1].ret; return call_result(c,r,&rt); }
        return call_value(c,r); }
      const bcir_ctype *rt=callee_ret(c,&id);     /* a struct-returning call: `mk(x).field` postfixes the result */
      int drop_save=c->call_dropped; c->call_dropped=0;   /* a call nested in an argument restores it */
      uint32_t r=p_call(c,&id);
      if(c->call_dropped && c->fn->n_claims) c->fn->claims[c->fn->n_claims-1].truncated=1;
      c->call_dropped=drop_save;
      return call_result(c,r,rt); }                 /* `mk(x).f`, `f()->v`; `g(x)(y)` through the result, which must be
                                                     * a function pointer */
}
/* `( *... NAME ) (` at the cursor, NAME a function-pointer object (a local, a parameter, a global) or a function: the `*`s
 * name the function again (C11 6.5.3.2p4), so the call is `NAME(...)` -- the oracle's `_call_ptr`, which strips them.
 * The token after the `)` (the call's `(`) on success, else 0. Nothing is consumed or lowered. */
static int deref_named_callee(CC *c, tok *nm){
  int k=c->i+1; while(tok_is(tat(c,k),"*")) k++;
  int amp=tok_is(tat(c,k),"&"); if(amp) k++;        /* `(&f)(x)`, `(*&f)(x)`: the function's address, called --
                                                     * `f(x)` (CF-EXTDESIG) */
  const tok *nt=tat(c,k);
  if(k==c->i+1 || nt->k!=T_ID || !tok_is(tat(c,k+1),")") || !tok_is(tat(c,k+2),"(")) return 0;
  const venv *lv=lookup(c,nt); int gi=lv ? -1 : find_global(c,nt->s,nt->n);
  int fnv = amp ? (!lv && gi<0 && declared_ret(c,nt))   /* `&` of an object is a pointer to it, no function */
          : lv ? (lv->type.kind==3 && !names_object_ptr(c,lv))   /* a table's `*t` is its first element */
          : gi>=0 ? (c->gv[gi].ty.kind==3 && !c->gv[gi].is_arr)
          : declared_ret(c,nt) ? 1 : 0;
  if(!fnv) return 0;
  *nm=*nt; return k+2;
}
/* A function used as a VALUE (function-to-pointer decay, `o->fn = g`; `&g`, C11 6.3.2.1p4, 6.5.3.2p3): a funcptr value
 * emitted as the bare function name (C decays it). No claim. A function declared here by an earlier definition or a
 * prototype -- one this unit defines further on, or one another unit defines, which the emit declares `extern` as it
 * declares every prototype (CF-EXTDESIG) -- as the oracle's `func_rets`, `protos` and `_require_declared` hold it. */
static uint32_t fn_value(CC *c, const tok *id){
  char fnm[BCIR_CIR_NAME]; idcpy(c,fnm,id);
  uint32_t r=add_res(c,BCIR_DOM_RAM,cc_abi(c)->pointer_size,1,0,BCIR_RK_SCALAR,fnm);
  if(c->fn->n_res){ bcir_resource *rr=&c->fn->res[c->fn->n_res-1]; rr->read_only=1; rr->is_funcptr=1; }
  return r;
}
static uint32_t p_primary(CC *c) {
  if(is(c,"_Generic")) return call_value(c,p_generic(c));   /* `_Generic(x, T: f, ...)(y)` (CF-FPTAB) */
  if(isk(c,T_INT)){tok t=adv(c);
    if(is(c,"[")){ fail(c,"a subscript of an integer constant is not supported"); return 0; }   /* `1[p]` (CF-FPTAB) */
    int lsz,lsg; lit_int_type(t.s,t.n,cc_abi(c)->long_size,&lsz,&lsg);   /* the constant's type (§6.4.4.1) */
    uint32_t r=tempi(c,lsz,lsg);
    bcir_claim *cl=new_claim(c,"c.const",BCIR_OP_LOAD);if(!cl)return r;
    cl->n_wr=1;cl->wr[0]=r;cl->n_imm=1;cl->imm[0]=t.v;return call_value(c,r);}   /* `1(2)`: refused (CF-FPTAB) */
  if(isk(c,T_FLT)){tok t=adv(c);                       /* a floating constant -> a typed c.fconst */
    int isf = t.n>0 && (t.s[t.n-1]=='f'||t.s[t.n-1]=='F');   /* f/F -> float(4) */
    int isl = t.n>0 && (t.s[t.n-1]=='l'||t.s[t.n-1]=='L');   /* l/L -> long double, else double(8) */
    uint32_t r=tempf(c, isf?4:isl?cc_abi(c)->long_double_size:8);
    char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.fconst:%.*s",t.n,t.s);
    bcir_claim *cl=new_claim(c,op,BCIR_OP_LOAD); if(cl){cl->n_wr=1;cl->wr[0]=r;} return call_value(c,r);}
  if(isk(c,T_STR)){     /* a string literal -> an anonymous read-only char[] global; value is a ptr */
    tok st=adv(c); uint32_t rid;
    if(isk(c,T_STR)){ int cn; char *cb=gather_strings(c,st,&cn);   /* adjacent literals concatenate */
      rid=intern_string(c, cb?cb:st.s, cb?cn:st.n); }
    else rid=intern_string(c,st.s,st.n);   /* full spelling kept in result-owned metadata; dedup; cap lifted */
    if(is(c,"[")){ c->i++; uint32_t ix=p_expr(c); eat(c,"]");
      venv sv; memset(&sv,0,sizeof sv); sv.rid=rid; sv.type.size=1; sv.sidx=-1;
      return emit_index(c,&sv,ix); }
    return call_value(c,rid);                   /* `"ab"(1)`: refused (CF-FPTAB) */
  }
  if(is(c,"_Alignof")||is(c,"alignof")) return p_alignof(c);   /* the type's alignment, a folded size_t */
  if(is(c,"sizeof")) return p_sizeof(c);   /* the operand's own type (CF-SIZEOF), a folded size_t */
  if(is(c,"(")){
    { tok nm; int at=deref_named_callee(c,&nm); if(at){ c->i=at; return p_named_call(c,nm); } }   /* `(*fp)(x)` */
    c->i++;uint32_t r=p_expr(c);
    while(is(c,",")){c->i++;r=p_expr(c);}    /* the comma OPERATOR (lowest prec): lower each operand for its */
    eat(c,")");return call_value(c,r);}      /* side effects, DISCARD all but the last, yield the last rid; a `(`
                                              * after it calls through the value (CF-FPTAB) */
  if(isk(c,T_ID)){
    tok id=adv(c);
    if(is(c,"(")) return p_named_call(c,id);
    int ec=visible_enum(c,id.s,id.n);             /* an enumerator in scope -> its folded constant (type int) */
    if(ec>=0){uint32_t r=tempi(c,4,1);bcir_claim *cl=new_claim(c,"c.const",BCIR_OP_LOAD);
      if(cl){cl->n_wr=1;cl->wr[0]=r;cl->n_imm=1;cl->imm[0]=c->ec[ec].val;}return r;}
    venv *v=lookup(c,&id); if(!v) v=use_global(c,&id);   /* a file-scope global (lookup table)? */
    if(!v){
      if(declared_ret(c,&id)){                          /* a FUNCTION used as a VALUE (CF-EXTDESIG) */
        if(is_assign_op(c) || is(c,"++") || is(c,"--")){ fail(c,CC_FN_NOT_LVALUE); return 0; }   /* `f = g`, `f++` */
        return fn_value(c,&id); }
      if(is_imag_unit(&id)){                            /* <complex.h> imaginary unit (unless shadowed) */
        uint32_t r=tempc(c,8);                          /* `float _Complex` (value i), emitted verbatim */
        char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.cconst:%.*s",id.n,id.s);
        bcir_claim *cl=new_claim(c,op,BCIR_OP_LOAD); if(cl){cl->n_wr=1;cl->wr[0]=r;}
        return r;
      }
      long long mo;
      if(memorder_value(&id,&mo)){                      /* SEG6.1/SEG7: a `memory_order_*` / `__ATOMIC_*` constant
                                                         * (reached ONLY when NOT a declared var/param/global -- the
                                                         * `if(!v)` guard -- and NOT a defined function -- callee_ret
                                                         * above; EXACTLY the oracle _rvalue precedence env->func->
                                                         * constant) -> an int const claim, byte-identical to a
                                                         * same-valued integer literal (so the kind/value rails agree). */
        uint32_t r=tempi(c,4,1);                        /* signed int, like the oracle's scalar("int") */
        bcir_claim *cl=new_claim(c,"c.const",BCIR_OP_LOAD);
        if(cl){cl->n_wr=1;cl->wr[0]=r;cl->n_imm=1;cl->imm[0]=mo;}
        return r;
      }
      fail_undeclared(c,&id);return 0;
    }
    return call_value(c,postfix_lvalue(c,v));     /* `ops[i](x)`, `t->fn[i](x)`: through the value (CF-FPTAB) */
  }
  fail(c,"expected expression");return 0;
}
/* name a cast's target type by width, so both rails emit the same (uintN_t) spelling. With signed_int
 * set, a width-named integer uses the SIGNED fixed-width spelling -- needed for a float -> signed-int
 * conversion, which is UB/target-divergent if rendered as float -> unsigned. */
static void cast_name(CC *c,const bcir_ctype *ty,int signed_int,char *o,size_t n){
  if(ty->kind==2){   /* a POINTER target, spelled faithfully (the oracle's `_pointer_spelling`): the pointee's own
                      * type -- its sign, a plain `char`, `_Bool`, a float, a struct/union tag or `void` -- behind
                      * its `volatile`, then one `*` per level; the cast yields a real `T *` of that type */
    char base[BCIR_CIR_AGG];
    if(ty->ptr_to_struct) fits(c,base,sizeof base,"%s %s",ty->is_union?"union":"struct",ty->tag);
    else if(ty->ptr_to_fp) snprintf(base,sizeof base,"fnptr");   /* a pointer to a function pointer (CF-RTFP) */
    else if(ty->size==0 && !ty->is_float) snprintf(base,sizeof base,"void");
    else if(ty->bit_width>0) snprintf(base,sizeof base,"%s_BitInt(%d)",ty->signd?"":"unsigned ",ty->bit_width);
    else if(ty->is_complex) snprintf(base,sizeof base,"%s",ty->size==8?"float _Complex":ty->size>16?"long double _Complex":"double _Complex");
    else if(ty->is_float) snprintf(base,sizeof base,"%s",ty->size==4?"float":ty->size>8?"long double":"double");
    else if(ty->is_bool) snprintf(base,sizeof base,"_Bool");
    else if(ty->is_plain_char) snprintf(base,sizeof base,"char");
    else snprintf(base,sizeof base,"%s",ty->signd?(ty->size==1?"int8_t":ty->size==2?"int16_t":ty->size==8?"int64_t":"int32_t")
                                              :(ty->size==1?"uint8_t":ty->size==2?"uint16_t":ty->size==8?"uint64_t":"uint32_t"));
    int d=ty->ptr_depth?ty->ptr_depth:1; if(d>BCIR_MAX_PTR_DEPTH) d=BCIR_MAX_PTR_DEPTH;
    char stars[BCIR_MAX_PTR_DEPTH+1]; for(int k=0;k<d;k++) stars[k]='*'; stars[d]=0;
    fits(c,o,n,"c.cast:%s%s%s %s",ty->is_volatile?"volatile ":"",ty->is_atomic?"_Atomic ":"",base,stars); return; }
  if(ty->kind==3){ fits(c,o,n,"c.cast:fnptr"); return; }   /* a function pointer: its temp carries the type (CF-RTFP) */
  if(ty->kind==0 && ty->bit_width>0){               /* a `_BitInt(N)` cast target -- the exact spelling (faithful) */
    fits(c,o,n,"c.cast:%s_BitInt(%d)",ty->signd?"":"unsigned ",ty->bit_width); return; }
  const char *nm=ty->is_complex ? (ty->size==8?"float _Complex":ty->size>16?"long double _Complex":"double _Complex")
                : ty->is_bool ? "_Bool"           /* a bool cast normalizes any nonzero (full value) to 1 */
                : ty->is_float ? (ty->size==4?"float":ty->size>8?"long double":"double")  /* >8: extended (matches the oracle's `long double`) */
                : signed_int ? (ty->size==1?"int8_t":ty->size==2?"int16_t":ty->size==8?"int64_t":"int32_t")
                : ty->size==1?"uint8_t":ty->size==2?"uint16_t":ty->size==8?"uint64_t":"uint32_t";
  fits(c,o,n,"c.cast:%s",nm);
}
static int incdec_value(CC *c, uint32_t *out);   /* fwd: `++a`/`a++`/`--a`/`a--` in EXPRESSION position */
static uint32_t p_unary_inner(CC *c);
/* Token `k` starts a postfix operator -- `[`, `->`, `.`, `(`, `++` or `--` -- which binds tighter than a prefix
 * operator, so `*name` followed by one dereferences the postfix expression, not the name. */
static int postfix_follows(CC *c, int k){
  const tok *t=tat(c,k);
  if(t->k!=T_PUN) return 0;
  if(t->n==1) return t->s[0]=='[' || t->s[0]=='.' || t->s[0]=='(';
  return t->n==2 && ((t->s[0]=='-' && t->s[1]=='>') || (t->s[0]=='+' && t->s[1]=='+') || (t->s[0]=='-' && t->s[1]=='-'));
}
/* CF-RTVOL: `*(volatile T *)((char *)p + K)` -- one volatile access of `T` at byte offset `K` from the device
 * region `p`, the claim the member access `p->m` lowers to, not a pointer computation and an access at offset
 * 0: the oracle's `_byte_offset_access`, the same predicate. The emit spells a volatile member access so, and
 * driver code a register access by byte offset. Types are read resolved (p_type): the cast's a pointer to a
 * volatile, non-`_Atomic` integer or floating `T`, the byte pointer's a pointer to plain `char`, whatever its
 * qualifiers; `K` an integer literal or an enumerator, no wider than an `int`; `p` a declared pointer or array,
 * or `&s` of a struct or union, whose region holds volatile storage. The oracle matches its AST, so this match
 * skips what the oracle's parser drops: a redundant parenthesis anywhere (a unary `+` it keeps, CF-UNARY). */
static int bo_open(CC *c){                          /* `(`s opening no type-name: the count of `(` */
  int n=0;
  for(;;){
    if(!is(c,"(") || tok_is(tat(c,c->i+1),"{")) return n;   /* `({ ... })` is a statement expression */
    c->i++;
    if(starts_type_name(c)){ c->i--; return n; }    /* a cast's own `(` */
    n++;
  }
}
static int bo_close(CC *c,int n){ for(int k=0;k<n;k++){ if(!is(c,")")) return 0; c->i++; } return 1; }
static int bo_cast(CC *c,bcir_ctype *ty){           /* `( type-name )`: the type, resolved, the cursor past it */
  int si; if(!is(c,"(")) return 0; c->i++;
  if(!starts_type_name(c) || p_cast_type(c,ty,&si) || c->td_nd || !is(c,")")) return 0;
  c->i++; return 1;
}
static int bo_match(CC *c,venv *base,field *f){
  bcir_ctype to, tb; int k1=bo_open(c);             /* around the cast expression */
  if(!bo_cast(c,&to) || to.kind!=2 || (to.ptr_depth?to.ptr_depth:1)!=1 || to.ptr_to_struct || to.is_valist) return 0;
  int fpslot=to.ptr_to_fp && !to.is_volatile;      /* `*(R (**)(P))(...)`: a function-pointer slot (CF-RTFP) */
  if(!fpslot && (to.ptr_to_fp || to.size<=0 || !(to.is_volatile || to.is_atomic)))
    return 0;                                       /* else a pointer to a volatile or `_Atomic` scalar */
  int k2=bo_open(c); if(k2<1) return 0;            /* around the operand, and the byte pointer's own */
  if(!bo_cast(c,&tb) || tb.kind!=2 || (tb.ptr_depth?tb.ptr_depth:1)!=1 || tb.ptr_to_struct || tb.size!=1
     || !tb.is_plain_char || tb.is_atomic) return 0;          /* a pointer to plain `char` */
  int p=bo_open(c), addr=is(c,"&");
  if(addr){ c->i++; p+=bo_open(c); }               /* `&s`: the struct's own storage */
  if(!isk(c,T_ID) || visible_enum(c,pk(c)->s,pk(c)->n)>=0) return 0;
  { tok id=*pk(c); venv *vp=lookup(c,&id); if(!vp) vp=use_global(c,&id); if(!vp) return 0; *base=*vp; c->i++; }
  const bcir_resource *br=res_of(c->fn,base->rid); int arr=br && (br->is_array || br->is_vla);
  if(!br || (to.is_volatile && br->domain!=BCIR_DOM_MMIO)   /* a volatile access only of a device region */
     || (addr ? base->type.kind!=1 || arr : base->type.kind!=2 && !arr)) return 0;
  int cl=0; while(is(c,")")){ c->i++; cl++; }       /* the base's parens, then the byte pointer's own */
  if(cl<p || cl-p>=k2 || !is(c,"+")) return 0;
  c->i++; k2-=cl-p;
  int r=bo_open(c), e; long long k;
  if(isk(c,T_INT)) k=adv(c).v;
  else if(isk(c,T_ID) && (e=visible_enum(c,pk(c)->s,pk(c)->n))>=0){ k=c->ec[e].val; c->i++; }
  else return 0;
  if(k<0 || k>INT_MAX || !bo_close(c,r+k2+k1) || postfix_follows(c,c->i)) return 0;
  memset(f,0,sizeof *f);                            /* a volatile member of T's layout at offset K */
  f->byte_off=(int)k; f->sidx=f->elem_sidx=f->ptee_sidx=-1;
  if(fpslot){                                       /* ... a function pointer of the cast's type (CF-RTFP) */
    f->size=f->access_bytes=cc_abi(c)->pointer_size; f->fp_sig=to.fp_sig; f->fp_ret_size=to.fp_ret_size;
    f->fp_ret_signd=to.fp_ret_signd; f->fp_ret_float=to.fp_ret_float; f->fp_ret_void=to.fp_ret_void;
    f->fp_ret_agg=to.fp_ret_agg; return 1; }
  f->size=f->access_bytes=to.size; f->signd=to.signd; f->is_float=to.is_float; f->is_complex=to.is_complex;
  f->is_bool=to.is_bool; f->is_plain_char=to.is_plain_char; f->bit_width=to.bit_width;
  f->is_volatile=to.is_volatile; f->is_atomic=to.is_atomic;   /* ... or an `_Atomic` one (CF-RTFP) */
  return 1;
}
/* `*(R (**)(P))((char *)p + K) = (R2 (*)(P2))f;` -- the emit's store of a function pointer through a generic slot is the
 * member store it was emitted from, `p->m = f` (CF-RTFP; the oracle's `_assign`): its cast spells the slot, no conversion
 * of its own, so a function or a function-pointer object, named, is stored as it is. 1 with the value in `*out`, the
 * cursor on the `;`; else 0, the cursor unmoved and every speculative effect rolled back. */
static BCIR_NOINLINE int fp_slot_value(CC *c, uint32_t *out){
  int save=c->i, failed=c->failed, ns=c->ns, nec=c->nec; spec_mark m; spec_begin(c,&m);
  bcir_ctype ty; int si;
  if(is(c,"(")){ c->i++;
    if(starts_type_name(c) && !p_cast_type(c,&ty,&si) && ty.kind==3 && !ty.nadims && is(c,")")){
      c->i++;
      const tok *nt=pk(c);
      if(nt->k==T_ID && tok_is(tat(c,c->i+1),";")){
        const venv *lv=lookup(c,nt); int gi=lv ? -1 : find_global(c,nt->s,nt->n);
        int fnv = lv ? lv->type.kind==3 && !names_object_ptr(c,lv)
                : gi>=0 ? c->gv[gi].ty.kind==3 && !c->gv[gi].is_arr
                : declared_ret(c,nt)!=NULL;
        if(fnv){ *out=p_unary(c); return 1; } } } }
  spec_end(c,&m); c->i=save; c->failed=failed; c->ns=ns; c->nec=nec; return 0;
}
/* Called with the cursor just past the `*`. On a match: *base (p's env entry) and *f, the cursor past the
 * operand. Else every speculative effect -- a global bound, a struct or an enumerator a type-name declared, a
 * failed type parse -- is rolled back and the cursor left where it was, for the pointer-value lowering. */
static int byte_off_access(CC *c,venv *base,field *f){
  int save=c->i, failed=c->failed, ns=c->ns, nec=c->nec; spec_mark m; spec_begin(c,&m);
  if(bo_match(c,base,f)) return 1;
  spec_end(c,&m); c->i=save; c->failed=failed; c->ns=ns; c->nec=nec; return 0;
}
/* A cast's type name at the cursor (CF-RTFP): the specifier and its `*`s, then -- `(R (*)(P))f`, `(R (**)(P))p`,
 * `(T *(*)(P))f` -- an abstract function-pointer declarator, as a parameter's is read (`fp_inline_decl`), whose `*`s
 * past the first are a pointer to one. A type name declares no identifier (C11 6.7.7p1): a named one is refused,
 * as the oracle refuses it (`TYPE_NAME_NAMED`). An array of them, `(R (*[N])(P))`, comes back with its dimensions in
 * `nadims`/`adims`, outermost first, which each caller takes as the type name's own (`fp_name_dims`). 1 after a
 * failure. */
static int p_cast_type(CC *c, bcir_ctype *ty, int *si){
  if(p_type(c,ty,si)) return 1;
  if(!fp_decl_at(c,1)) return 0;
  bcir_ctype fp; int stars, dims[3], nd; tok nm;
  if(fp_inline_decl(c,ty,*si,&fp,&stars,dims,&nd,&nm)) return 1;
  if(nm.n){ fail(c,CC_TYPE_NAME_NAMED); return 1; }
  for(int k=0;k<stars;k++) fp_star(c,&fp);
  if(nd){ fp.nadims=(uint8_t)nd; for(int k=0;k<nd;k++) fp.adims[k]=dims[k]; }
  *ty=fp; *si=-1; return 0;
}
/* The dimensions of an array of function pointers `p_cast_type` read, `(R (*[2][3])(P))`, moved off the element type
 * `ty` into `dims` after the `*n` there (outermost first), as the type name's own: the element is the function pointer,
 * the array the type name's (CF-RTFP). 1 after a failure (more than `cap`). */
static int fp_name_dims(CC *c, bcir_ctype *ty, long long *dims, int *n, int cap){
  if(ty->kind!=3) return 0;
  for(int k=0; k<ty->nadims && k<3; k++){
    if(*n>=cap){ fail(c,"a type-name of more than four array dimensions"); return 1; }
    dims[(*n)++]=ty->adims[k]; }
  ty->nadims=0; memset(ty->adims,0,sizeof ty->adims);
  return 0;
}
/* Depth-guarded wrapper: p_unary is a recursive-cycle entry point (p_unary->p_primary->`(`->p_expr->
 * ...->p_unary), so a deeply-nested expression would exhaust the native stack. Bump/check depth once
 * per level here; on overflow fail cleanly ("nesting too deep") and return without recursing. */
static uint32_t p_unary(CC *c) {
  if(ENTER_REC(c)){ LEAVE_REC(c); return 0; }
  uint32_t r=p_unary_inner(c); LEAVE_REC(c); return r;
}
/* The paths of `p_unary_inner` whose locals would otherwise sit in every level of the recursive descent
 * (BCIR_NOINLINE): the address of an object through the pointer member the lvalue loads (`&h.next->v`, CF-SPLIT2), a
 * volatile load at a literal byte offset (CF-RTVOL) and the first element of a member array (`*q->a`, CF-SPLIT2).
 * The loads return 1 with the value in `*out`, or 0 with the cursor unmoved. */
static BCIR_NOINLINE uint32_t addr_through_loaded(CC *c, venv *v, const field *mf){
  plv lv; uint32_t ptr=emit_member(c,v,mf,0);
  if(!plv_chain(c,ptr,mf->ptee_sidx,*mf,&lv)){
    if(!c->failed) fail(c,"address-of this lvalue through a loaded pointer is a follow-on");
    return 0; }
  return plv_addr(c,&lv);
}
static BCIR_NOINLINE int byte_off_load(CC *c, uint32_t *out){
  venv bb; field bf;
  if(!byte_off_access(c,&bb,&bf)) return 0;
  *out=emit_member(c,&bb,&bf,0); return 1;
}
static BCIR_NOINLINE int member_array_first_load(CC *c, uint32_t *out){
  venv mb; field me;
  if(!deref_member_array(c,&mb,&me)) return 0;
  *out=emit_member(c,&mb,&me,0); return 1;
}
static uint32_t p_unary_inner(CC *c) {
  { uint32_t v; if(incdec_value(c,&v)) return v; }   /* PREFIX ++a / POSTFIX a++ (member/array/pointer/scalar) */
  if(is(c,"+")){ c->i++; return unary_plus(c,p_unary(c)); }   /* `+a`: a, promoted (CF-UNARY) */
  if(is(c,"__real__")||is(c,"__imag__")){        /* GNU complex part -> the real element float */
    const char *suf=is(c,"__real__")?"creal":"cimag"; c->i++;
    uint32_t a=p_unary(c); if(!unary_operand_ok(c,a,suf)) return 0;   /* of a struct, a pointer (CF-UNARY) */
    uint32_t r=part_temp(c,a);                    /* not integer-computed; emitted `__real__ x` */
    char op[BCIR_CIR_OP];fits(c,op,sizeof op,"c.un.%s",suf);
    bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);if(cl){cl->n_rd=1;cl->rd[0]=a;cl->n_wr=1;cl->wr[0]=r;}
    return r; }
  if(is(c,"&&")){ c->i++;                          /* `&&label` -- a label's address as a `void *` value (GNU).
    * Safe in unary position: a binary `&&` never STARTS a unary expression, so logical-AND is unaffected. */
    tok lb=adv(c);                                  /* the label identifier */
    uint32_t t=add_res(c,BCIR_DOM_RAM,cc_abi(c)->pointer_size,1,0,BCIR_RK_POINTER,"");   /* a `void *` temp */
    if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1]; tr->ptr_depth=1; tr->is_voidptr=1; }
    char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.labeladdr:%.*s",lb.n,lb.s);   /* emit `void *t = &&L;` */
    bcir_claim *cl=new_claim(c,op,BCIR_OP_LOAD); if(cl){cl->n_wr=1;cl->wr[0]=t;}   /* a LOAD claim, no reads */
    return t; }
  if(is(c,"&")){ c->i++;                          /* address-of: &lvalue -> a pointer value (c.addrof) */
    if(is(c,"*")){ c->i++; uint32_t r=p_unary(c);  /* &*p == p (the pointer itself; &*(p+i) == p+i) -- but `*` */
      const bcir_resource *rr=c->failed?NULL:res_of(c->fn,r);   /* still takes a pointer, an array or a function */
      if(!c->failed && !(rr && (rr->kind==BCIR_RK_POINTER || rr->is_array || rr->is_vla || rr->is_pointer
                                || rr->is_funcptr || rr->is_voidptr))) fail(c,CC_DEREF_NOT);   /* (6.5.3.2p3, CF-FPTAB) */
      return r; }
    if(isk(c,T_ID)){ tok id=*pk(c); venv *vp=lookup(c,&id); if(!vp) vp=use_global(c,&id);
      if(!vp && declared_ret(c,&id) && !tok_is(tat(c,c->i+1),"(")){   /* `&f`: the function's address, the value its
                                                    * designator converts to -- no claim (CF-EXTDESIG) */
        c->i++; return fn_value(c,&id); }
      /* SNAPSHOT the env entry: &s.arr[i] / &arr[i] resolve the index via member_arr_index / p_expr, which
       * can declare locals and realloc c->env[] -- a pointer into it would dangle (the helpers only READ). */
      venv vsnap; venv *v=NULL; if(vp){ vsnap=*vp; v=&vsnap; }
      if(v){ c->i++;
        if((is(c,".")||is(c,"->")) && v->sidx>=0){   /* &s.m / &s->m (scalar/struct member; value OR ptr base) */
          sdef *S=&c->s[v->sidx]; c->i++; tok fn=adv(c); int fi=-1;
          for(int i=0;i<S->nf;i++) if((int)strlen(S->f[i].name)==fn.n&&!strncmp(S->f[i].name,fn.s,fn.n)) fi=i;
          if(fi<0){ fail(c,"unknown field"); return 0; }
          field mf=member_descend(c,S->f[fi]);     /* accumulate the chain's byte offset + the leaf field */
          if(mf.bit_w){ fail(c,"cannot take the address of a bit-field"); return 0; }   /* illegal in C */
          if(mf.is_ptr && !mf.arr_count && (is(c,"->")||is(c,".")||is(c,"[")))   /* `&h.next->v`, `&n->next->p[i]`:
                                                    * the object through the pointer member it loads (CF-SPLIT2) */
            return addr_through_loaded(c,v,&mf);
          if(mf.is_ptr && !mf.arr_count){          /* &s.ptr / &s->ptr -- address of a POINTER member -> a `T **` */
            uint32_t t=add_res(c,BCIR_DOM_RAM, mf.ptee_size?mf.ptee_size:4, 1,0,BCIR_RK_POINTER,"");
            if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];   /* pointee = the member's pointee */
              tr->is_signed=(uint8_t)(mf.signd?1:0); tr->is_float=(uint8_t)(mf.ptee_float?1:0); tr->ptr_depth=2;
              tr->is_atomic=(uint8_t)(mf.ptee_atomic?1:0);   /* still reaching `_Atomic` storage (CF-ATOMIC) */
              if(mf.ptee_sidx>=0) fits(c,tr->agg,sizeof tr->agg,"%s %s",c->s[mf.ptee_sidx].is_union?"union":"struct",c->s[mf.ptee_sidx].tag); }
            last_ptr_to_volatile(c,mf.ptee_volatile,mf.ptee_sidx>=0 && sdef_vol(c,mf.ptee_sidx));
            bcir_claim *cl=new_claim(c,"c.addrof",BCIR_OP_ADD);
            if(cl){cl->n_rd=1;cl->rd[0]=v->rid;cl->n_wr=1;cl->wr[0]=t;cl->n_imm=1;cl->imm[0]=mf.byte_off;}
            return t;
          }
          if(mf.arr_count){                        /* &s.arr[i] / &s->arr[i] / &s.m[i][j] -- a member-array element */
            if(!is(c,"[")){ fail(c,"address-of an array member is a follow-on"); return 0; }
            uint32_t ix=member_arr_index(c,&mf);   /* the row-major flattened element index */
            int es = mf.size?mf.size:4, off=mf.byte_off;   /* element (struct) STRIDE + the member offset */
            int resz = es;                         /* the RESULT pointee size (the field's, for &s.arr[i].field) */
            int rsd=mf.signd, rfl=mf.is_float, rsx=mf.elem_sidx;   /* result-pointer pointee type */
            int rat=mf.is_atomic, rcx=mf.is_complex, rbo=mf.is_bool, rpc=mf.is_plain_char;   /* `_Atomic` (CF-ATOMIC) */
            if(is(c,".")||is(c,"->")){             /* &s.arr[i].field -- array-of-structs element FIELD address */
              if(mf.elem_sidx<0){ fail(c,"address-of a field of a non-struct member-array element"); return 0; }
              sdef *ES=&c->s[mf.elem_sidx]; c->i++; tok efn=adv(c); int efi=-1;
              for(int k=0;k<ES->nf;k++) if((int)strlen(ES->f[k].name)==efn.n&&!strncmp(ES->f[k].name,efn.s,efn.n)) efi=k;
              if(efi<0){ fail(c,"unknown field"); return 0; }
              field ef=member_descend(c,ES->f[efi]);
              if(ef.bit_w||ef.is_ptr||ef.arr_count){ fail(c,"address-of a non-scalar array-of-structs field is a follow-on"); return 0; }
              off += ef.byte_off; resz=ef.size?ef.size:4; rsd=ef.signd; rfl=ef.is_float; rsx=ef.sidx;   /* field at member_off+field_off; stride stays the struct */
              rat=ef.is_atomic; rcx=ef.is_complex; rbo=ef.is_bool; rpc=ef.is_plain_char;
            }
            if(is(c,".")||is(c,"->")||is(c,"[")){  /* a further descent is a follow-on */
              fail(c,"address-of a nested member-array element is a follow-on"); return 0; }
            uint32_t t=add_res(c,BCIR_DOM_RAM, resz, 1,0,BCIR_RK_POINTER,"");   /* an `element/field *` */
            if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
              tr->is_signed=(uint8_t)(rsd?1:0); tr->is_float=(uint8_t)(rfl?1:0); tr->ptr_depth=1;
              tr->is_atomic=(uint8_t)(rat?1:0); tr->is_complex=(uint8_t)(rcx?1:0);
              tr->is_bool=(uint8_t)(rbo?1:0); tr->is_plain_char=(uint8_t)(rpc?1:0);
              if(rsx>=0) fits(c,tr->agg,sizeof tr->agg,"%s %s",c->s[rsx].is_union?"union":"struct",c->s[rsx].tag); }
            bcir_claim *cl=new_claim(c,"c.addrof",BCIR_OP_ADD);
            if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=ix;cl->n_wr=1;cl->wr[0]=t;cl->n_imm=2;cl->imm[0]=off;cl->imm[1]=es;}
            return t;
          }
          uint32_t t=add_res(c,BCIR_DOM_RAM, mf.size?mf.size:4, 1,0,BCIR_RK_POINTER,"");   /* a `leaf *` */
          if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
            tr->is_signed=(uint8_t)(mf.signd?1:0); tr->ptr_depth=1;
            tr->is_float=(uint8_t)(mf.is_float?1:0); tr->is_complex=(uint8_t)(mf.is_complex?1:0);   /* the leaf's own */
            tr->is_bool=(uint8_t)(mf.is_bool?1:0); tr->is_plain_char=(uint8_t)(mf.is_plain_char?1:0);   /* type, as the */
            tr->is_atomic=(uint8_t)(mf.is_atomic?1:0);                                   /* oracle's `T *` */
            if(mf.sidx>=0) fits(c,tr->agg,sizeof tr->agg,"%s %s",c->s[mf.sidx].is_union?"union":"struct",c->s[mf.sidx].tag); }
          bcir_claim *cl=new_claim(c,"c.addrof",BCIR_OP_ADD);
          if(cl){cl->n_rd=1;cl->rd[0]=v->rid;cl->n_wr=1;cl->wr[0]=t;cl->n_imm=1;cl->imm[0]=mf.byte_off;}
          return t;
        }
        if(is(c,"[")){ field gf; if(global_md_field(c,v,&gf)){   /* &g[i][j] of a multi-dimensional global */
            uint32_t ix=member_arr_index(c,&gf); if(c->failed) return 0;
            if(is(c,".")||is(c,"->")||is(c,"[")){ fail(c,"address-of a nested element is a follow-on"); return 0; }
            return addr_member_elem(c,v,&gf,ix); } }
        if(is(c,"[")){                             /* &arr[i] / &p[i] -- a plain element address `(char*)base + i*es` */
          if(!names_object_ptr(c,v)){ fail(c,CC_SUBSCRIPT_NOT); return 0; }   /* `&s[1]` of an integer (CF-FPTAB) */
          int want=subscript_dims(c,v); uint32_t ix;
          if(want>1){                              /* &m[i][j] of a multi-dimensional array (local, VLA, `T m[][N]`
                                                    * parameter): every subscript, Horner-flattened -- the oracle's
                                                    * `_lvalue(Index)`; a row's address is refused, as the oracle does */
            if(sub_group_count(c,c->i)<want){ fail(c,"the address of a row of a multi-dimensional array is not supported"); return 0; }
            ix=array_index_n(c,v,want); if(c->failed) return 0; }
          else { c->i++; ix=p_expr(c); eat(c,"]"); }
          if(is(c,".") && v->sidx>=0){ field af; uint32_t tot; int r=aos_member_array(c,v,ix,&af,&tot);   /* &a[i].m[j] */
            if(r<0) return 0;
            if(r>0){ if(is(c,".")||is(c,"->")||is(c,"[")){ fail(c,"address-of a nested element is a follow-on"); return 0; }
              return addr_member_elem(c,v,&af,tot); } }
          if((is(c,".")||is(c,"->")) && v->sidx>=0){ /* &arr[i].field on a PLAIN array-of-structs base (#495 sibling):
                                                      * the element-field address `(char*)base + i*sizeof(elem) + field_off`,
                                                      * a `field *`. The element struct is v->sidx, the stride v->type.size. */
            field sub; if(!aos_elem_field(c,v,&sub)){ if(c->failed) return 0;
              fail(c,"address-of a plain-base array-of-structs element field is a follow-on"); return 0; }
            if(is(c,".")||is(c,"->")||is(c,"[")){  /* a further descent (nested field / subscript) is a follow-on */
              fail(c,"address-of a nested plain-base array-of-structs element field is a follow-on"); return 0; }
            int strd = v->type.size?v->type.size:4, fz = sub.size?sub.size:4;
            uint32_t t=add_res(c,BCIR_DOM_RAM, fz, 1,0,BCIR_RK_POINTER,"");   /* a `field *` */
            if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
              tr->is_signed=(uint8_t)(sub.signd?1:0); tr->is_float=(uint8_t)(sub.is_float?1:0); tr->ptr_depth=1;
              tr->is_complex=(uint8_t)(sub.is_complex?1:0); tr->is_bool=(uint8_t)(sub.is_bool?1:0);
              tr->is_plain_char=(uint8_t)(sub.is_plain_char?1:0); tr->is_atomic=(uint8_t)(sub.is_atomic?1:0);
              if(sub.sidx>=0) fits(c,tr->agg,sizeof tr->agg,"%s %s",c->s[sub.sidx].is_union?"union":"struct",c->s[sub.sidx].tag); }
            bcir_claim *cl=new_claim(c,"c.addrof",BCIR_OP_ADD);
            if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=ix;cl->n_wr=1;cl->wr[0]=t;cl->n_imm=2;cl->imm[0]=sub.byte_off;cl->imm[1]=strd;}
            return t;
          }
          if(is(c,".")||is(c,"->")||is(c,"[")){    /* &arr[i].field on a non-struct base / nested: a follow-on */
            fail(c,"address-of a plain-base array-of-structs element field is a follow-on"); return 0; }
          int es = v->type.size?v->type.size:4;    /* the pointee / element byte size */
          /* an element that is a pointer -- of an array of pointers, or through a `T **`: its address strides by the
           * pointer and is one level deeper, `T **`, which `&arr[1]` had made an `int32_t *` four bytes in (CF-IDXARROW) */
          int parr=ptr_array(c,v), pel=parr || (v->type.kind==2 && ptr_levels(&v->type)>1);
          int stride=pel ? cc_abi(c)->pointer_size : es;
          uint32_t t=add_res(c,BCIR_DOM_RAM, es, 1,0,BCIR_RK_POINTER,"");
          if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
            tr->is_signed=(uint8_t)(v->type.signd?1:0); tr->is_float=(uint8_t)(v->type.is_float?1:0);
            tr->ptr_depth=(uint8_t)(pel ? ptr_levels(&v->type)+(parr?1:0) : 1);
            if(pel && v->type.size==0 && !v->type.is_float && !v->type.ptr_to_struct && !v->type.ptr_to_fp)
              tr->is_voidptr=1;                     /* `&va[i]` of `void *va[N]`: a `void **` */
            tr->is_complex=(uint8_t)(v->type.is_complex?1:0); tr->is_bool=(uint8_t)(v->type.is_bool?1:0);
            tr->is_plain_char=(uint8_t)(v->type.is_plain_char?1:0);
            tr->is_atomic=(uint8_t)(v->type.is_atomic?1:0);   /* an element of `_Atomic` storage (CF-ATOMIC) */
            if(v->type.kind==1||v->type.ptr_to_struct)        /* `&ps[i]` of an array of structs too: `struct T *`,
                                                              * never the `int32_t *` it had been (CF-STRUCTVAL) */
              fits(c,tr->agg,sizeof tr->agg,"%s %s",v->type.is_union?"union":"struct",v->type.tag);
            else if(v->type.kind==3) ptee_fp(c,tr,&v->type); }   /* `&ops[i]`: a pointer to a function pointer */
          bcir_claim *cl=new_claim(c,"c.addrof",BCIR_OP_ADD);
          if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=ix;cl->n_wr=1;cl->wr[0]=t;cl->n_imm=2;cl->imm[0]=0;cl->imm[1]=stride;}
          return t;
        }
        uint32_t t=addr_temp(c,&v->type); if(c->failed) return 0;   /* a pointer one level deeper: &p (T*) -> T** */
        last_ptr_to_volatile(c,v->type.is_volatile,v->sidx>=0 && sdef_vol(c,v->sidx));   /* &x of volatile storage */
        bcir_claim *cl=new_claim(c,"c.addrof",BCIR_OP_ADD);
        if(cl){cl->n_rd=1;cl->rd[0]=v->rid;cl->n_wr=1;cl->wr[0]=t;} return t; } }
    if(is(c,"(")){ int save=c->i; c->i++;             /* `&(type){...}` -- address of a compound literal */
      int is_type = starts_type_name(c);
      bcir_ctype ty; int si;
      if(is_type && !p_type(c,&ty,&si) && is(c,")")){ c->i++;
        if(is(c,"{") && ty.is_atomic && ty.kind!=2){ fail(c,CC_ATOMIC_LITERAL); return 0; }   /* CF-RTFP */
        if(is(c,"{")){ uint32_t rid=p_compound_literal(c,&ty,si);   /* materialize the anonymous object */
          uint32_t t=addr_temp(c,&ty); if(c->failed) return 0;   /* a `T *` to it -- `T **` of a pointer one (CF-RTFP) */
          bcir_claim *cl=new_claim(c,"c.addrof",BCIR_OP_ADD);
          if(cl){cl->n_rd=1;cl->rd[0]=rid;cl->n_wr=1;cl->wr[0]=t;} return t; } }
      c->i=save; }
    fail(c,"unsupported address-of (only &local/&param/&(compound literal))"); return 0; }
  if(is(c,"-")||is(c,"~")||is(c,"!")){
    const char *suf=is(c,"-")?"neg":is(c,"~")?"bnot":"lnot"; int is_lnot=is(c,"!");
    bcir_opcode oc=(is(c,"-")||is(c,"!"))?BCIR_OP_SUB:BCIR_OP_ADD;c->i++;   /* the oracle's `_UN`: `-` and `!` SUB, `~` ADD */
    uint32_t a=p_unary(c);
    if(!unary_operand_ok(c,a,suf)) return 0;      /* of a struct (CF-STRUCTARITH), a pointer, `~` of a float (CF-UNARY) */
    /* `-`/`~` take the promoted operand type (so negating a `long` stays 64-bit, not a truncated
     * uint32 that widens back to a positive long); `-x` on a float stays float (floats don't promote,
     * and a uint32 temp would truncate -2.5 to a huge integer); a sub-int integer operand promotes to
     * SIGNED int (§6.3.1.1), so `~(unsigned char)0` is -1, not 4294967295; `!` is int. */
    uint32_t r;
    if(is_lnot) r=tempi(c,4,1);
    else { const bcir_resource *ar=res_of(c->fn,a);
           if(ar&&ar->is_complex) r=tempc(c,(int)ar->elem_bytes);        /* `-z`, `~z` of a complex is complex (CF-UNARY:
                                                                         * a real float temp kept only its real part) */
           else if(ar&&ar->is_float) r=tempf(c,(int)ar->elem_bytes);     /* `-x` on a float is float */
           else if(ar&&ar->bit_width>0) r=tempbi(c,ar->bit_width,ar->is_signed);   /* a `_BitInt` does not promote
                                                                         * (C23 6.3.1.1p2; CF-UNARY: it was an `int`) */
           else { int sz=ar?(int)ar->elem_bytes:4, sg=ar?ar->is_signed:1;
                  promote_i(&sz,&sg);                                    /* sub-int -> signed int */
                  r=tempi(c,sz,sg); } }
    char op[BCIR_CIR_OP];fits(c,op,sizeof op,"c.un.%s",suf);
    bcir_claim *cl=new_claim(c,op,oc);if(cl){cl->n_rd=1;cl->rd[0]=a;cl->n_wr=1;cl->wr[0]=r;}return r;}
  if(is(c,"*")){                                   /* pointer dereference: *p / *(p + i) */
    c->i++;
    { uint32_t r; if(byte_off_load(c,&r)) return r; }   /* CF-RTVOL */
    { uint32_t r; if(member_array_first_load(c,&r)) return r; }   /* `*q->a` */
    if(is(c,"(")){ int save=c->i; c->i++;          /* *(p) or *(p + i) */
      if(isk(c,T_ID)){ tok pid=*pk(c); venv *pvp=lookup(c,&pid); if(!pvp) pvp=use_global(c,&pid);
        if(pvp && names_object_ptr(c,pvp)){   /* `*(s + 1u)` of an integer: the general path refuses it (CF-FPTAB) */
          c->i++; venv pvsnap=*pvp; venv *pv=&pvsnap;   /* SNAPSHOT: the `+ i` index p_expr below can realloc c->env[] */
          size_t s_res=c->fn->n_res,s_cl=c->fn->n_claims; uint32_t s_rid=c->rid,s_cid=c->cid,s_clc=c->cl_ctr;
          int through=1;                                /* `*(q[j] ...)`: through the loaded pointer element */
          if(is(c,"[")){ uint32_t lin=index_chain(c,pv); if(c->failed) return 0; through=step_to_elem_ptr(c,pv,lin); }
          if(through && is(c,"+")){ c->i++; uint32_t idx=p_expr(c); eat(c,")"); return emit_index(c,pv,idx); }
          if(through && is(c,")")){ c->i++; return emit_deref(c,pv); }
          c->fn->n_res=s_res;c->fn->n_claims=s_cl;c->rid=s_rid;c->cid=s_cid;c->cl_ctr=s_clc; } }   /* not ours: undo */
      c->i=save;
    } else if(isk(c,T_ID) && !postfix_follows(c,c->i+1)){
      tok pid=*pk(c); venv *pv=lookup(c,&pid); if(!pv) pv=use_global(c,&pid);   /* `*q[j]` is `*(q[j])` and
      * `*t->p` is `*(t->p)`: a postfix operator binds tighter than `*`, so a name followed by one takes the
      * general path below. Taking this one for `*t->arr[i]` dereferenced `t` and failed on the `->` */
      if(pv){ c->i++; return emit_deref(c,pv); } }  /* *p (no sub-parse between lookup and use); a global too */
    return emit_deref_rid(c, p_unary(c));            /* general: `**pp`, `*(<expr>)` -- deref a ptr rvalue */
  }
  if(is(c,"(") && tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='{')
    return p_stmt_expr(c);                          /* `({ ... })` -- a GCC statement expression */
  if(is(c,"(")){                                   /* (type)operand -- a cast binds at the unary level */
    int save=c->i; c->i++;
    int is_type = starts_type_name(c);
    if(is_type){ bcir_ctype ty;int si;
      if(!p_cast_type(c,&ty,&si)){
        int btdd[3], btd=td_dims_of(c,btdd);   /* a typedef'd array type */
        int la_count=0,la_nd=0,la_dims[3]={0,0,0};   /* a `(T[N]...)` array type-name -> an array compound literal */
        if(ty.kind==3){ long long fd[3]; int nf=0;   /* `(R (*[N])(P))`: an array of function pointers (CF-RTFP) */
          if(fp_name_dims(c,&ty,fd,&nf,3)) return 0;
          for(int d=0; d<nf; d++){ int dim=(int)fd[d]; la_dims[la_nd++]=dim; la_count=la_count?la_count*dim:dim; } }
        while(is(c,"[")){ c->i++; int dim=isk(c,T_INT)?(int)adv(c).v:0; eat(c,"]");
          if(la_nd<3)la_dims[la_nd]=dim; la_nd++; la_count=la_count?la_count*dim:dim; }
        for(int d=0; d<btd; d++){ int dim=btdd[d];     /* its dims follow the type-name's own */
          if(la_nd<3)la_dims[la_nd]=dim; la_nd++; la_count=la_count?la_count*dim:dim; }
        /* A SCALAR-element literal `(T[...]){...}` lowers for 1..3 dims; an AGGREGATE-element literal
         * `(struct P[]){...}` / `(struct P[N]){...}` (1-D) AND `(struct P[A][B]){...}` (multi-dim) also lower:
         * the initializer walk (init_object) fills every element, and the indexing venv carries BOTH `sidx`
         * (so `[...].field` descends the element struct via emit_index_field, striding by the struct size)
         * AND `adims`/`nadims` (so `[i][j]` Horner-flattens the outer dims). */
        if(ty.is_atomic && ty.kind!=2 && is(c,")")){   /* an `_Atomic` object, or a cast Clang does not convert */
          fail(c,tok_is(tat(c,c->i+1),"{") ? CC_ATOMIC_LITERAL : CC_CAST_ATOMIC); return 0; }
        if(la_nd && la_nd<=3 && is(c,")") &&
           tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='{'){   /* `(T[...]){...}` */
          c->i++;                                  /* ')' -- p_array_literal/arr_init eats the following `{` */
          uint32_t rid=p_array_literal(c,&ty,si,la_count,la_dims,la_nd);
          for(size_t z=c->fn->n_res;z-->0;) if(c->fn->res[z].rid==rid){ c->fn->res[z].is_array=1; break; }
          if(is(c,"[")){ venv sv; memset(&sv,0,sizeof sv); sv.rid=rid; sv.type=ty; sv.sidx=ty.kind==1?si:-1;
            if(la_nd>1){ for(int z=0;z<3;z++) sv.type.adims[z]=la_dims[z]; sv.type.nadims=la_nd; }
            return call_value(c,postfix_lvalue(c,&sv)); }   /* `(int[]){...}[i]` / `(int[A][B]){...}[i][j]` /
                                                   * `(struct P[A][B]){...}[i][j].f`; `(op_t[2]){f, g}[i](x)` (CF-RTFP) */
          return rid; }
      if(la_nd && is(c,")") && !tok_is(tat(c,c->i+1),"{")){ fail(c,CC_CAST_ARRAY); return 0; }   /* C11 6.5.4p2 */
      if(!la_nd && is(c,")")){
        c->i++;                                    /* ')' */
        if(is(c,"{")){ uint32_t rid=p_compound_literal(c,&ty,si);   /* `(type){init}` -- a compound literal, not a cast */
          if(is(c,".")||is(c,"->")||is(c,"[")){      /* direct postfix on the literal: `(struct P){...}.field` */
            venv sv; memset(&sv,0,sizeof sv); sv.rid=rid; sv.type=ty; sv.sidx=si; return postfix_lvalue(c,&sv); }
          return call_value(c,rid); }              /* `(op_t){f}(x)`: a call through it (CF-RTFP) */
        uint32_t v=p_unary(c);                     /* the operand (right-associative) */
        if(ty.kind==0 && ty.size==0 && !ty.is_float && !ty.bit_width)
          return void_temp(c);                     /* `(void)e`: e for its effects, no value (C11 6.3.2.2) */
        if(ty.kind!=1 && !scalar_value_ok(c,v)) return 0;   /* `(uint32_t)a` of a struct (CF-STRUCTARITH) */
        return emit_cast(c,v,&ty,si);
      } }
    }
    c->i=save;                                     /* not a cast -> a parenthesized expression */
  }
  if(is(c,"++")||is(c,"--")) return incdec_rest(c,1,0);   /* a prefix step `incdec_value` did not take */
  return incdec_rest(c,0,p_primary(c));            /* ... or a postfix one (CF-STRUCTCOND) */
}
static int bin_op(CC *c,char *suf,bcir_opcode *oc) {
  struct {const char *t,*s;bcir_opcode o;} B[]={{"*","mul",BCIR_OP_MUL},{"/","div",BCIR_OP_MUL},
    {"%","mod",BCIR_OP_MUL},{"+","add",BCIR_OP_ADD},{"-","sub",BCIR_OP_SUB},{"<<","shl",BCIR_OP_ADD},
    {">>","shr",BCIR_OP_ADD},{"<","lt",BCIR_OP_SUB},{">","gt",BCIR_OP_SUB},{"<=","le",BCIR_OP_SUB},
    {">=","ge",BCIR_OP_SUB},{"==","eq",BCIR_OP_SUB},{"!=","ne",BCIR_OP_SUB},{"&","and",BCIR_OP_ADD},
    {"^","xor",BCIR_OP_ADD},{"|","or",BCIR_OP_ADD},{"&&","land",BCIR_OP_ADD},{"||","lor",BCIR_OP_ADD},{0,0,0}};
  for(int i=0;B[i].t;i++) if(is(c,B[i].t)){strcpy(suf,B[i].s);*oc=B[i].o;return i;} return -1;
}
static int prec_of(int idx){static const int P[]={10,10,10,9,9,8,8,7,7,7,7,6,6,5,4,3,2,1};return P[idx];}
/* the binary op of a compound assignment `OP=` (its first char): the suffix + cost-class opcode. */
static void compound_binop(char ch,const char **suf,bcir_opcode *oc){
  switch(ch){case '+':*suf="add";*oc=BCIR_OP_ADD;break; case '-':*suf="sub";*oc=BCIR_OP_SUB;break;
    case '*':*suf="mul";*oc=BCIR_OP_MUL;break; case '/':*suf="div";*oc=BCIR_OP_MUL;break;
    case '%':*suf="mod";*oc=BCIR_OP_MUL;break; case '&':*suf="and";*oc=BCIR_OP_ADD;break;
    case '|':*suf="or";*oc=BCIR_OP_ADD;break;  case '<':*suf="shl";*oc=BCIR_OP_ADD;break;  /* <<= */
    case '>':*suf="shr";*oc=BCIR_OP_ADD;break;                                              /* >>= */
    default:*suf="xor";*oc=BCIR_OP_ADD;break;}  /* ^ */
}
/* A compound-assignment operator token: a 2-char `+= -= *= /= %= &= |= ^=` or a 3-char `<<= >>=`
 * (the op char `s[0]` drives compound_binop). Returns 1 if the token is a compound-assign, else 0. */
static int is_compound_op(const tok *t){
  if(t->k!=T_PUN) return 0;
  if(t->n==2 && t->s[1]=='=' && strchr("+-*/%&|^",t->s[0])) return 1;
  if(t->n==3 && t->s[2]=='=' && (t->s[0]=='<'||t->s[0]=='>') && t->s[0]==t->s[1]) return 1;
  return 0;
}
/* lookahead from token `j` (a `.`/`->`/`[` chain following an lvalue): is an `=`/OP= at the end -- i.e. is
 * this a member/element STORE, or a member access used as a VALUE (e.g. `({ s.m; })`)? Skips the whole
 * chain (`.field`, `->field`, balanced `[...]`) without consuming. */
static int member_is_store(CC *c,int j){
  for(;;){
    const tok *t=tat(c,j);   /* bounded: a `j+=2` chain off the tail must not read past c->t[nt] (Bug B) */
    if(t->k==T_END) return 0;
    if(t->k==T_PUN && t->n==1 && t->s[0]=='.'){ j+=2; continue; }              /* .field */
    if(t->k==T_PUN && t->n==2 && t->s[0]=='-' && t->s[1]=='>'){ j+=2; continue; }  /* ->field */
    if(t->k==T_PUN && t->n==1 && t->s[0]=='['){ int d=1; j++;                  /* balanced [...] */
      while(tat(c,j)->k!=T_END && d){ char ch=tat(c,j)->s[0]; if(ch=='[')d++; else if(ch==']')d--; j++; } continue; }
    break;
  }
  return tat(c,j)->k==T_PUN && ((tat(c,j)->n==1 && tat(c,j)->s[0]=='=') || is_compound_op(tat(c,j)));
}
/* The result temp of a binary op `lhs <suf> rhs` -- the usual arithmetic conversions in the (width,
 * signedness) value model: float arithmetic propagates the wider float; a shift keeps the promoted left
 * operand; an integer op the UAC width/sign; a (pointer ± int) stays a pointer (pointee-scaled). Shared
 * by p_binrhs AND the compound-assignment / ++/-- sites, so `long s; s += x` / `double s; s += x` keep
 * their width instead of truncating to a 4-byte uint32. (A relational/logical result -- always int -- is
 * handled at the p_binrhs call site, not here; no compound assignment ever yields one.) */
/* A multi-dimensional array object -- a local `T a[A][B]` (flattened, its dims on the env entry), a multi-dim VLA,
 * a decayed `T m[][N]` / `T (*m)[N]` parameter, a file-scope `T g[A][B]`: as a value it would decay to a row
 * pointer the value model has no spelling for (CF-DECAY). */
static int rid_multidim(CC *c, uint32_t rid){
  const bcir_resource *r=res_of(c->fn,rid);
  if(!r) return 0;
  if(r->vla_ndims>1) return 1;
  for(int i=0;i<c->nenv;i++) if(c->env[i].rid==rid && c->env[i].type.nadims>1) return 1;
  if(r->read_only && r->name[0]){
    int gi=find_global(c,r->name,(int)strlen(r->name));
    if(gi>=0 && c->gv[gi].nd>1) return 1;
  }
  return 0;
}
/* A value that is an address: a pointer, or an array object (which converts to its first element's address). */
static int rid_addr(const bcir_resource *r){ return r && (r->kind==BCIR_RK_POINTER || r->is_array); }
static uint32_t binop_result(CC *c, const char *suf, uint32_t lhs, uint32_t rhs){
  if(agg_operand(c,lhs) || agg_operand(c,rhs)) return temp(c,4);   /* `a + 1`, `a += 5`, `a++` (CF-STRUCTARITH) */
  if(!strcmp(suf,"add") || !strcmp(suf,"sub")){      /* address arithmetic, ahead of the arithmetic types an array's
                                                      * ELEMENT would give it (the integer path typed `la + 1` int) */
    const bcir_resource *lr=res_of(c->fn,lhs), *rr=res_of(c->fn,rhs);
    int lp=rid_addr(lr), rp=rid_addr(rr);
    if(lp || rp){
      if((lp && rid_multidim(c,lhs)) || (rp && rid_multidim(c,rhs))){
        fail(c,"arithmetic on a multi-dimensional array is not yet supported"); return temp(c,4); }
      if(lp && rp){                                  /* p - q: a ptrdiff_t, signed and pointer-wide (CF-PTRDIFF) */
        if(!strcmp(suf,"sub")) return tempi(c,cc_abi(c)->pointer_size,1);
      } else return tempptr(c, lp?lhs:rhs);          /* p + i / i + p / p - i: the pointer (an array decayed) */
    }
  }
  int is_arith=!strcmp(suf,"add")||!strcmp(suf,"sub")||!strcmp(suf,"mul")||!strcmp(suf,"div");
  int is_shift=!strcmp(suf,"shl")||!strcmp(suf,"shr");
  /* C23 `_BitInt(N)` (non-promoting, exact width). The first-class subset carries the result ONLY when it
   * is itself a `_BitInt(N)` -- the bit-precise operand WINS the C23 6.2.5/6.3.1.8 rank (its width strictly
   * exceeds the other operand's post-promotion width). VERIFIED == Clang via the `_Generic` differential.
   *   * same-type `_BitInt(N)` op `_BitInt(N)`           -> `_BitInt(N)`
   *   * a WIDER `_BitInt` op a narrower `_BitInt`/standard-int VARIABLE/constant -> the wider `_BitInt`
   *   * a shift `bi << k`                                -> the (non-promoting) `_BitInt` left operand
   * Any mix whose C23 result is a STANDARD integer type (the `_BitInt` does NOT out-rank: equal/lesser
   * width) cleanly fails -- the twin has no fallback, so a clean failure here keeps the rails in lockstep
   * (the Python rail routes the same form to fallback). A bare integer constant carries its REAL literal
   * type (so `bi64+5`->`_BitInt(64)`, `bi8+5`->`int`->fail), matching Clang -- no const short-circuit. */
  int bsa=0,bsb=0; int ba=rid_bitint(c,lhs,&bsa), bb=rid_bitint(c,rhs,&bsb);
  if(ba||bb){
    if(is_shift){ if(ba) return tempbi(c,ba,bsa); fail(c,"`_BitInt` shift without a `_BitInt` left operand"); return temp(c,4); }
    if(ba&&bb){                                          /* two `_BitInt`s: the WIDER wins (its own sign);
                                                          * equal width combines signedness (unsigned if either). */
      if(ba>bb) return tempbi(c,ba,bsa);
      if(bb>ba) return tempbi(c,bb,bsb);
      return tempbi(c,ba,(bsa&&bsb)?1:0);
    }
    int bw=ba?ba:bb, bs=ba?bsa:bsb; uint32_t other=ba?rhs:lhs;
    /* a `_BitInt` mixed with a FLOAT converts to the float -> the result is FLOATING, not a `_BitInt`. */
    if(rid_fsize(c,other)){ fail(c,"`_BitInt` mixed with a floating type (result is not a `_BitInt`)"); return temp(c,4); }
    int osz=4,osg=1; rid_int(c,other,&osz,&osg);         /* the other operand's (width, sign) ... */
    if(osz<4){osz=4;osg=1;}                               /* ... after integer promotion (a `_BitInt` does not) */
    if(bw>osz*8) return tempbi(c,bw,bs);                  /* `_BitInt` strictly wider -> it wins, own sign */
    fail(c,"`_BitInt` arithmetic whose C23 result is a standard integer type"); return temp(c,4);
  }
  int fa=rid_fsize(c,lhs), fb=rid_fsize(c,rhs);
  if(is_arith&&(fa||fb)){                                      /* float/complex arithmetic */
    int ca=rid_complex(c,lhs), cb=rid_complex(c,rhs);
    if(ca||cb){ int ea=ca?fa/2:fa, eb=cb?fb/2:fb;             /* compare ELEMENT widths (a complex's */
      int ew=ea>eb?ea:eb; return tempc(c,ew*2); }             /* elem_bytes is the full 2x pair) */
    return tempf(c,(fa>fb?fa:fb)); }                          /* real float -> the wider float */
  int sa,za,sb,zb; int ia=rid_int(c,lhs,&sa,&za), ib=rid_int(c,rhs,&sb,&zb);
  if(ia&&ib){ int rs,rz;
    if(is_shift){ promote_i(&sa,&za); rs=sa; rz=za; }          /* a shift result: the promoted left operand */
    else uac_i(sa,za,sb,zb,&rs,&rz);
    return tempi(c,rs,rz);
  }
  return temp(c,4);
}
/* --- the operands C may leave unevaluated (CF-TERNARY; the oracle's `_operand_pure`) ---------------------------
 * C evaluates exactly one arm of `c ? a : b` (C11 6.5.15p4) and the right operand of `&&` / `||` only when the left
 * one does not decide the result (6.5.13p4, 6.5.14p4). An operand may still be computed eagerly -- as a `c.select` or
 * a `c.bin.land` / `c.bin.lor` over values already in hand -- when computing it can neither trap nor change state:
 * every claim it made, from `from` on, is one of these ops, is no volatile access (a device register's included: the
 * lowering marks it volatile as it makes it) and writes no declared (named) variable. An atomic access is a load, a
 * store or a read-modify-write, none of them on the list. Any other operand lowers as a branch that evaluates it only
 * when C does. The rails decide on the claims each made, which parity already holds equal, so they cannot classify an
 * operand differently. */
static int void_value(CC *c, uint32_t v){ const bcir_resource *r=res_of(c->fn,v); return r && r->is_void; }
static int operand_pure(CC *c, size_t from){
  static const char *const ops[]={"c.const","c.copy","c.select","c.addrof","c.ptradd","c.ptrsub","c.sizeof.vla",0};
  static const char *const pre[]={"c.fconst:","c.cconst:","c.cast:","c.labeladdr:",0};
  static const char *const bin[]={"add","sub","mul","and","or","xor","shl","shr","lt","gt","le","ge","eq","ne",
                                  "land","lor",0};   /* not `div` / `mod`: a zero divisor traps */
  static const char *const un[]={"neg","bnot","lnot","creal","cimag",0};
  for(size_t i=from;i<c->fn->n_claims;i++){ const bcir_claim *cl=&c->fn->claims[i]; int ok=0;
    if(cl->opcode==BCIR_OP_NOP) return 0;                       /* a marker: an operand it already branches on */
    if(cl->is_volatile) return 0;                               /* a volatile access: reading one is a side effect */
    for(int k=0;k<cl->n_wr;k++){ const bcir_resource *r=res_of(c->fn,cl->wr[k]); if(r && r->name[0]) return 0; }
    for(int k=0;ops[k] && !ok;k++) ok=!strcmp(cl->op,ops[k]);
    for(int k=0;pre[k] && !ok;k++) ok=!strncmp(cl->op,pre[k],strlen(pre[k]));
    if(!ok && !strncmp(cl->op,"c.bin.",6)) for(int k=0;bin[k] && !ok;k++) ok=!strcmp(cl->op+6,bin[k]);
    if(!ok && !strncmp(cl->op,"c.un.",5)) for(int k=0;un[k] && !ok;k++) ok=!strcmp(cl->op+5,un[k]);
    if(!ok) return 0; }
  return 1;
}
/* Whether the operand starting at token `at` is pure, deciding it the first time by lowering it speculatively (`lower`
 * parses exactly the operand) and rolling that back: memoized by the token, so an operand nested in another is not
 * re-speculated each time the enclosing one is lowered -- a chain `a ? b : c ? d : ...` stays quadratic, not
 * exponential. -1 when the memo cannot grow (the caller then speculates again: slower, never different). */
static int pure_memo(CC *c, int at, int pure){
  for(int k=0;k<c->npure;k++) if(c->pure_memo[k]>>1==at) return c->pure_memo[k]&1;
  if(pure<0) return -1;
  if(CC_ENSURE(c,c->pure_memo,c->npure,c->cap_pure)) c->pure_memo[c->npure++]=at*2+(pure?1:0);
  return pure;
}
static int operand_is_pure(CC *c, uint32_t (*lower)(CC *, int), int arg){
  int at=c->i, pure=pure_memo(c,at,-1);
  if(pure>=0) return pure;
  spec_mark m; spec_begin(c,&m); size_t from=c->fn->n_claims;
  (void)lower(c,arg);
  pure=!c->failed && operand_pure(c,from);
  spec_end(c,&m); c->i=at;
  return pure_memo(c,at,pure);
}
static void marker(CC *c,const char *op,uint32_t cond,int has_cond);
/* The branch such an operand lowers to assigns one named local in each arm -- `if (c) { ..; sel = a; } else { ..;
 * sel = b; }`, the source form both rails already lower and emit (the oracle's `_branch_value`). Its type is set once
 * both arms are lowered (`type_as_select`). */
static uint32_t branch_local(CC *c){ return add_res(c,BCIR_DOM_RAM,4,1,0,BCIR_RK_SCALAR,"sel"); }
static void assign_local(CC *c, uint32_t v, uint32_t sel){
  bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD); if(cl){cl->n_rd=1;cl->rd[0]=v;cl->n_wr=1;cl->wr[0]=sel;}
}
/* The function type of the function `name` the unit defines, used as a value (a designator decays to a pointer to
 * it, C11 6.3.2.1p4), as 1 + its index in `sigs`: its definition's return and parameter types -- a variadic one's named
 * parameters and `...` (CF-FPRET) -- recorded the first time with a synthesized `typedef RET (*__bcir_fpN)(PARAMS);`
 * that spells a pointer to it. 0 for a name the unit has not defined. */
static int designator_sig(CC *c, const char *name){
  for(int k=0;k<c->nsig;k++) if(!strcmp(c->sigs[k].fn,name)) return k+1;
  const bcir_func *df=NULL;
  for(int i=0;c->unit && i<c->unit->n_funcs && !df;i++) if(!strcmp(c->unit->funcs[i].name,name)) df=&c->unit->funcs[i];
  int pk=-1;                                /* else its prototype's: a function another unit defines, or this one
                                             * further on -- the oracle's `func_rets` and `protos` (CF-EXTDESIG) */
  for(int k=0;!df && k<c->n_protos && pk<0;k++) if(!strcmp(c->protos[k].name,name)) pk=k;
  if(!df && pk<0) return 0;
  bcir_ctype ret=df ? df->ret : c->protos[pk].ret;
  int np=df ? df->n_params : c->protos[pk].n_params, va=df ? df->variadic : c->protos[pk].variadic;
  bcir_ctype *ps=NULL; size_t bytes;
  if(np>0){
    if(!bcir_size_mul((size_t)np,sizeof *ps,&bytes) ||
       !(ps=(bcir_ctype *)bcir_host_arena_allocate(&c->scratch,bytes,_Alignof(bcir_ctype)))){ cc_raise_oom(c); return 0; }
    for(int k=0;k<np;k++) ps[k]=df ? df->params[k].type : c->protos[pk].params[k]; }
  char al[BCIR_CIR_NAME], rets[BCIR_EMIT_TYPE];
  snprintf(al,sizeof al,"__bcir_fp%d",c->n_fpdef++);
  ctype_qstr(&ret,rets,sizeof rets);                  /* qualified as the type is (CF-QUALS) */
  ctext_putf(c,&c->fpdefs,"typedef %s (*%s)(",rets,al);
  for(int k=0;k<np;k++){ char pt[BCIR_EMIT_TYPE]; ctype_qstr(&ps[k],pt,sizeof pt);
    ctext_putf(c,&c->fpdefs,"%s%s",k?", ":"",pt); }
  ctext_putf(c,&c->fpdefs,"%s);\n",va?(np?", ...":"..."):np?"":"void");
  return sig_addv(c,&ret,ps,np,al,name,va);
}
/* The function type the function-pointer value in `rid` points at, as 1 + its index in `sigs`, when a typedef spells
 * a pointer to it -- 0 otherwise: a function-pointer object (a local, a parameter, a global) has its declaration's; a
 * temp typed as a function pointer (a select of them, a null pointer constant) its alias's; a designator its
 * function's (`designator_sig`). The oracle's `_fn_type`. */
static int res_sig(CC *c, uint32_t rid){
  int s=0, named=0;
  for(int i=c->nenv;i-- > 0 && !named;) if(c->env[i].rid==rid){ named=1; s=c->env[i].type.kind==3 ? c->env[i].type.fp_sig : 0; }
  if(!named){ const bcir_resource *r=res_of(c->fn,rid);
    if(!r || !r->is_funcptr) return 0;
    if(r->agg[0]){ for(int k=0;k<c->nsig && !s;k++) if(!strcmp(c->sigs[k].alias,r->agg)) s=k+1; }
    else if(r->read_only && r->name[0]) s=designator_sig(c,r->name); }
  return s>0 && s<=c->nsig && c->sigs[s-1].alias[0] ? s : 0;
}
/* Two function types are one when their returns and their parameters are, pairwise, the same types as `_Generic`
 * tells them apart (qualifiers aside) -- the oracle's `_fn_key`, where a `va_list` is its own type, never the
 * integer of its size. */
static int sig_same(const CC *c, int a, int b){
  if(a==b) return 1;
  const fsig *x=&c->sigs[a-1], *y=&c->sigs[b-1];
  if(x->n_params!=y->n_params || x->variadic!=y->variadic || !ctype_generic_eq(&x->ret,&y->ret)) return 0;
  if(qual_key(&x->ret)!=qual_key(&y->ret)) return 0;   /* a `const T *` return is another type (CF-QUALS) */
  for(int k=0;k<x->n_params;k++)
    if(x->params[k].is_valist!=y->params[k].is_valist || !ctype_generic_eq(&x->params[k],&y->params[k])
       || qual_key(&x->params[k])!=qual_key(&y->params[k])) return 0;
  return 1;
}
/* The type `?:` gives its arms' values (C11 6.5.15p3-6), as a temp: over ARITHMETIC arms their common type (the usual
 * arithmetic conversions), NOT a blanket unsigned -- a signed arm keeps its sign and a FLOAT arm makes the result the
 * wider float; a pointer (or array) arm the pointer, an array decayed (CF-DECAY); a struct or union arm that aggregate,
 * copied whole (CF-STRUCTVAL), its arms of one type; a function-pointer arm -- a function designator or a
 * function-pointer object -- that function pointer, its arms of one function type (CF-FNSEL). 0 (after `fail`) for a
 * pair C refuses. */
static uint32_t select_temp(CC *c, uint32_t a, uint32_t b){
  int sa,za,sb,zb,fa,fb;
  int ap=rid_addr(res_of(c->fn,a)), bp=rid_addr(res_of(c->fn,b));
  const bcir_resource *ra=res_of(c->fn,a), *rb=res_of(c->fn,b);
  int ag=ra && ra->kind==BCIR_RK_AGGREGATE, bg=rb && rb->kind==BCIR_RK_AGGREGATE;
  fa=rid_float(c,a,&sa); fb=rid_float(c,b,&sb);
  if(ap||bp){
    if((ap && rid_multidim(c,a)) || (bp && rid_multidim(c,b))){
      fail(c,"a multi-dimensional array operand of `?:` is not yet supported"); return 0; }
    return tempptr(c, ap?a:b); }
  if(ag||bg){
    int si=ag&&bg&&!strcmp(ra->agg,rb->agg) ? agg_sidx(c,ra->agg) : -1;
    if(si<0){ fail(c,"the arms of `?:` are a struct or union and a value of another type"); return 0; }
    return tempagg(c,si); }
  { int ga=res_sig(c,a), gb=res_sig(c,b);           /* a function-pointer arm: that pointer, spelled by its alias */
    if(ga||gb){
      if(ga && gb && !sig_same(c,ga,gb)){ fail(c,"the arms of `?:` point to functions of different types"); return 0; }
      char al[BCIR_CIR_NAME]; snprintf(al,sizeof al,"%s",c->sigs[(ga?ga:gb)-1].alias);
      uint32_t t=add_res(c,BCIR_DOM_RAM,cc_abi(c)->pointer_size,1,0,BCIR_RK_SCALAR,"");
      if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1]; tr->is_funcptr=1; fits(c,tr->agg,BCIR_CIR_AGG,"%s",al); }
      return t; } }
  if(fa||fb){ int w=(fa?sa:0)>(fb?sb:0)?(fa?sa:0):(fb?sb:0); return tempf(c,w); }   /* the wider float */
  if(rid_int(c,a,&sa,&za) && rid_int(c,b,&sb,&zb)){ int rs,rz; uac_i(sa,za,sb,zb,&rs,&rz); return tempi(c,rs,rz); }
  return temp(c,4);
}
/* A null pointer constant arm of `?:` whose value is a pointer or a function pointer takes that type (C11 6.5.15p6):
 * its `c.const 0` temp is declared as the pointer, never an `int` beside it -- in a select as in a branch's local (the
 * oracle's `_null_pointer` over the arms). `res` is the value's resource: the select's temp or the branch's local. */
static void null_arms_as(CC *c, const bcir_resource *res, uint32_t a, uint32_t b){
  bcir_resource tr=*res;                              /* a copy: `res` may sit in the array an arm is retyped in */
  null_as(c,&tr,a); null_as(c,&tr,b);
}
/* The branch's local takes the type a select of its arms has: the temp `select_temp` makes, made and then dropped (no
 * claim names it). A null pointer constant arm is then typed as the pointer, as `null_pointer` types one elsewhere. */
static void type_as_select(CC *c, uint32_t sel, uint32_t a, uint32_t b){
  size_t nres=c->fn->n_res; uint32_t rid=c->rid;
  uint32_t t=select_temp(c,a,b);
  bcir_resource tr; int have=t && !c->failed && res_copy(c,t,&tr);
  c->fn->n_res=nres; c->rid=rid;
  if(!have) return;
  for(size_t i=0;i<c->fn->n_res;i++){ bcir_resource *r=&c->fn->res[i];
    if(r->rid==sel){ char nm[BCIR_CIR_NAME]; snprintf(nm,sizeof nm,"%s",r->name);
      *r=tr; r->rid=sel; snprintf(r->name,sizeof r->name,"%s",nm); break; } }
  null_arms_as(c,&tr,a,b);
}
/* The right operand of a binary op at `prec`: a unary operand, then every op that binds tighter. */
static uint32_t p_binrhs(CC *c,int min_prec,uint32_t lhs);
static uint32_t p_binrhs_operand(CC *c,int prec){
  uint32_t rhs=p_unary(c);
  char s2[BCIR_CIR_NAME];bcir_opcode o2;int nx=bin_op(c,s2,&o2);
  while(nx>=0&&prec_of(nx)>prec){rhs=p_binrhs(c,prec_of(nx),rhs);nx=bin_op(c,s2,&o2);}
  return rhs;
}
/* `lhs && rhs` / `lhs || rhs` (the oracle's `cast.Binary` over `&&` / `||`): a pure right operand is computed eagerly
 * into `c.bin.land` / `c.bin.lor`; any other lowers as a branch -- the arm that evaluates it computes the same op (the
 * left operand is known there, so it is the right one's truth), the arm the left operand decides stores 0 (`&&`) or
 * 1 (`||`) -- into one named `int`. */
static uint32_t p_logical(CC *c,const char *suf,bcir_opcode oc,int prec,uint32_t lhs){
  char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.bin.%s",suf);
  int is_and=!strcmp(suf,"land");
  if(agg_operand(c,lhs)) return lhs;               /* a struct operand (CF-STRUCTARITH) */
  if(operand_is_pure(c,p_binrhs_operand,prec)){
    uint32_t rhs=p_binrhs_operand(c,prec), r=tempi(c,4,1);
    if(agg_operand(c,rhs)) return r;
    bcir_claim *cl=new_claim(c,op,oc);if(cl){cl->n_rd=2;cl->rd[0]=lhs;cl->rd[1]=rhs;cl->n_wr=1;cl->wr[0]=r;}
    return r; }
  uint32_t sel=branch_local(c);
  if(c->fn->n_res) c->fn->res[c->fn->n_res-1].is_signed=1;      /* an `int` */
  marker(c,"c.if",lhs,1);
  for(int arm=0;arm<2;arm++){
    if(arm) marker(c,"c.else",0,0);
    if((arm==0)==is_and){ uint32_t rhs=p_binrhs_operand(c,prec), r=tempi(c,4,1);
      if(agg_operand(c,rhs)) return sel;
      bcir_claim *cl=new_claim(c,op,oc);if(cl){cl->n_rd=2;cl->rd[0]=lhs;cl->rd[1]=rhs;cl->n_wr=1;cl->wr[0]=r;}
      assign_local(c,r,sel); }
    else { uint32_t k=tempi(c,4,1); bcir_claim *kc=new_claim(c,"c.const",BCIR_OP_LOAD);
      if(kc){kc->n_wr=1;kc->wr[0]=k;kc->n_imm=1;kc->imm[0]=is_and?0:1;}
      assign_local(c,k,sel); } }
  marker(c,"c.endif",0,0);
  return sel;
}
static uint32_t p_binrhs(CC *c,int min_prec,uint32_t lhs) {
  for(;;){
    char suf[BCIR_CIR_NAME];bcir_opcode oc;int idx=bin_op(c,suf,&oc);
    if(idx<0||prec_of(idx)<min_prec)return lhs;
    int prec=prec_of(idx);c->i++;
    if(!strcmp(suf,"land")||!strcmp(suf,"lor")){ lhs=p_logical(c,suf,oc,prec,lhs); continue; }   /* CF-TERNARY */
    uint32_t rhs=p_binrhs_operand(c,prec);
    if(agg_operand(c,lhs) || agg_operand(c,rhs)) return lhs;   /* a struct operand (CF-STRUCTARITH) */
    if(!strcmp(suf,"eq")||!strcmp(suf,"ne")){ null_compared(c,rhs,lhs); null_compared(c,lhs,rhs); }   /* `p == 0` */
    /* the result type: a relational op is int; otherwise the usual arithmetic conversions (float propagates the
     * wider float; a shift the promoted LHS; else the integer UAC) -- shared. */
    int is_cmp=!strcmp(suf,"lt")||!strcmp(suf,"gt")||!strcmp(suf,"le")||!strcmp(suf,"ge")
             ||!strcmp(suf,"eq")||!strcmp(suf,"ne");
    uint32_t r = is_cmp ? tempi(c,4,1)              /* relational -> int */
                        : binop_result(c,suf,lhs,rhs);   /* the usual arithmetic conversions (shared) */
    char op[BCIR_CIR_OP];fits(c,op,sizeof op,"c.bin.%s",suf);
    bcir_claim *cl=new_claim(c,op,oc);if(cl){cl->n_rd=2;cl->rd[0]=lhs;cl->rd[1]=rhs;cl->n_wr=1;cl->wr[0]=r;}
    lhs=r;
  }
}
static uint32_t p_binexpr(CC *c){return p_binrhs(c,1,p_unary(c));}
/* the conditional-expression level (binary + ternary). p_expr wraps this with the assignment level below. */
static uint32_t p_expr(CC *c);
static uint32_t cond_arms(CC *c, int unused){   /* both arms of a `?:`, lowered: `a : b` */
  (void)unused; (void)p_expr(c); eat(c,":"); return p_expr(c);
}
/* `cond ? a : b`: C evaluates the condition, then exactly one arm (C11 6.5.15p4). Two pure arms (`operand_pure`) are
 * computed eagerly and chosen by a select -- the emitter renders the real `(cond ? a : b)`; any other pair lowers as a
 * branch that evaluates only the arm C evaluates, into one named local of the select's type (CF-TERNARY: `d ? n / d
 * : 0` divided by zero and `p ? *p : s` read through NULL eagerly). Arms of type void have no value to select or
 * assign. The oracle's `cast.Ternary`. */
static uint32_t p_cond(CC *c){
  uint32_t cond=p_binexpr(c);
  if(!is(c,"?")) return cond;
  c->i++;
  if(!cond_value_ok(c,cond,0)) return 0;   /* `a ? x : y` of a struct (CF-STRUCTCOND) */
  if(operand_is_pure(c,cond_arms,0)){
    uint32_t a=p_expr(c); eat(c,":"); uint32_t b=p_expr(c);
    int va=void_value(c,a);
    if(va!=void_value(c,b)){ fail(c,"one arm of `?:` is void and the other is not"); return 0; }
    if(va) return a;                                   /* two void arms: evaluated for nothing, no value */
    uint32_t t=select_temp(c,a,b); if(!t) return 0;
    { const bcir_resource *tr=res_of(c->fn,t); if(tr) null_arms_as(c,tr,a,b); }   /* `c ? p : 0`: 0 is p's pointer */
    bcir_claim *cl=new_claim(c,"c.select",BCIR_OP_ADD);
    if(cl){cl->n_rd=3;cl->rd[0]=cond;cl->rd[1]=a;cl->rd[2]=b;cl->n_wr=1;cl->wr[0]=t;} return t; }
  /* A void conditional (C11 6.5.15p3: both arms void -- `c ? f() : g();`, an `assert`'s `c ? (void)0 : fail()`)
   * branches for the arm's effects alone: no local, no value. */
  marker(c,"c.if",cond,1);
  uint32_t a=p_expr(c);
  int va=void_value(c,a);
  uint32_t sel=va?0:branch_local(c);
  if(!va) assign_local(c,a,sel);
  marker(c,"c.else",0,0); eat(c,":");
  uint32_t b=p_expr(c);
  if(va!=void_value(c,b)){ fail(c,"one arm of `?:` is void and the other is not"); return 0; }
  if(!va) assign_local(c,b,sel);
  marker(c,"c.endif",0,0);
  if(va) return a;
  type_as_select(c,sel,a,b);
  return sel;
}

/* --- statements + functions ---------------------------------------------- */
static void env_add(CC *c,const tok *nm,uint32_t rid,const bcir_ctype *ty,int sidx){
  CC_ENSURE(c,c->env,c->nenv,c->cap_env);
  if(c->nenv>=c->cap_env)return; venv *v=&c->env[c->nenv++];
  idcpy(c,v->name,nm);v->rid=rid;v->type=*ty;v->sidx=sidx;
}
/* On first use of a file-scope global within a function, materialize a read-only data resource for
 * it (so an access `LUT[i]` lowers to a load) and bind it in the local env.  The resource is marked
 * read-only -- the emitter references the global by name (it is defined in the original source) and
 * does not redeclare it.  Subsequent uses in the same function resolve via the env. */
static venv *use_global(CC *c,const tok *id){
  int gi=find_global(c,id->s,id->n); if(gi<0) return NULL;
  venv *ex=lookup(c,id); if(ex) return ex;
  gvar *g=&c->gv[gi];
  /* a struct or union global -- an object, an array of them, or a pointer to one -- is bound with its
   * definition, as a local is, so `g.m` / `g[i].m` / `g->m` lower through the member paths (CF-GSTRUCT: an
   * index of -1 here made every member access read `c->s[-1]`), and its volatile members make it a device
   * region as a local's do */
  int gsi=(g->ty.kind==1 || g->ty.ptr_to_struct) ? find_struct(c,g->ty.tag,(int)strlen(g->ty.tag)) : -1;
  int kind = g->count>1 ? BCIR_RK_POINTER : (g->ty.kind==1?BCIR_RK_AGGREGATE:BCIR_RK_SCALAR);
  uint32_t rid=add_res(c,ty_mmio(c,&g->ty,gsi,g->is_arr)?BCIR_DOM_MMIO:BCIR_DOM_RAM,   /* a volatile global, or a */
                       g->ty.size,g->count,g->ty.is_volatile,kind,g->name);            /* pointer to volatile storage */
  if(c->fn->n_res){ bcir_resource *gr=&c->fn->res[c->fn->n_res-1];
    gr->read_only=1;                                  /* a global, not a local */
    gr->is_array=(uint8_t)(g->is_arr?1:0);
    gr->ndims=(uint8_t)(g->is_arr && g->nd>1 ? (g->nd<255?g->nd:255) : 0);   /* passed as `&m[0][0]` (CF-GARRAY) */
    gr->is_atomic=(uint8_t)(g->ty.is_atomic?1:0);     /* `_Atomic` storage, or a pointer to it (CF-ATOMIC) */
    gr->is_pointer=(uint8_t)(!g->is_arr&&(g->ty.kind==2||g->ty.kind==3)?1:0);
    if(g->is_arr && g->ty.kind==3) ptee_fp(c,gr,&g->ty);   /* a table of function pointers: its element's alias, as a
                                                         * local table's -- `*g` read an integer of its width (CF-FPTAB) */
    /* the value type rides on the resource, as a local's does: every temp typed from it (a deref of the
     * array, `-g`, the usual arithmetic conversions, a volatile read) is the global's type, not uint32 */
    if(g->ty.kind==0 && kind==BCIR_RK_POINTER){       /* an array: its element, as a pointer local's pointee */
      gr->is_signed=(uint8_t)(g->ty.signd?1:0); gr->is_float=(uint8_t)(g->ty.is_float?1:0);
      gr->is_plain_char=(uint8_t)(g->ty.is_plain_char?1:0); }
    else if(g->ty.kind==2 && kind==BCIR_RK_POINTER){  /* an array of pointers `T *g[N]`: it decays to a `T **` -- what
                                                       * `*g` reads is a pointer, never the `uint32_t` it had read, and
                                                       * `g + 1` is a `T **` (CF-IDXARROW) */
      gr->is_signed=(uint8_t)(g->ty.signd?1:0); gr->is_float=(uint8_t)(g->ty.is_float?1:0);
      gr->is_plain_char=(uint8_t)(g->ty.is_plain_char?1:0);
      gr->ptr_depth=(uint8_t)(ptr_levels(&g->ty)+1);
      if(g->ty.ptr_to_struct) fits(c,gr->agg,BCIR_CIR_AGG,"%s %s",g->ty.is_union?"union":"struct",g->ty.tag);
      else if(g->ty.ptr_to_fp) ptee_fp(c,gr,&g->ty);
      else if(g->ty.size==0 && !g->ty.is_float) gr->is_voidptr=1; }
    else if(g->ty.kind==0){                           /* a scalar: as a scalar local */
      gr->is_signed=(uint8_t)(g->ty.signd?1:0); gr->is_float=(uint8_t)(g->ty.is_float?1:0);
      gr->is_complex=(uint8_t)(g->ty.is_complex?1:0); gr->is_bool=(uint8_t)(g->ty.is_bool?1:0);
      gr->is_plain_char=(uint8_t)(g->ty.is_plain_char?1:0); gr->bit_width=g->ty.bit_width; }
    else if(kind==BCIR_RK_AGGREGATE)                  /* a struct or union: its type, as a local's -- a value of it
                                                       * in a brace list, an initializer or a select (CF-STRUCTVAL) */
      fits(c,gr->agg,BCIR_CIR_AGG,"%s %s",g->ty.is_union?"union":"struct",g->ty.tag); }
  env_add(c,id,rid,&g->ty,gsi);
  return lookup(c,id);
}
/* Is the cursor at a NAMED-variable assignment `name = ...` / `name OP= ...`? Only a bare name -- a member /
 * array / deref lvalue assignment used as a VALUE is a follow-on (both rails fall back). A single `=` is
 * distinguished from `==` (a 2-char token); a compound `+= ... >>=` is an is_compound_op. */
static int name_assign_ahead(CC *c){
  return c->t[c->i].k==T_ID &&
         ((tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='=') || is_compound_op(tat(c,c->i+1)));
}
/* Lookahead (side-effect-free): is the cursor at a SINGLE-LEVEL SCALAR member assignment-expression
 * `name.field = ...` / `name->field OP= ...`? Fills *out (the field) + *base (the struct base venv). A
 * pointer/array/nested-struct field, a nested/array/deref chain, or a non-local base returns 0 (those stay a
 * both-rails follow-on -- the value grammar store+reloads a direct member). A BITFIELD member IS eligible
 * (#bfassignexpr) -- its value is the masked/sign-extended STORED field (a bf.set store + a bf.get reload) --
 * UNLESS the base is volatile/MMIO (the re-read would be an extra access; that stays a fallback). */
static int member_assign_ahead(CC *c, field *out, venv **base){
  if(!isk(c,T_ID)) return 0;
  venv *v=lookup(c,&c->t[c->i]); if(!v || v->sidx<0) return 0;        /* a local/param struct (value or pointer) */
  const tok *op=tat(c,c->i+1);
  if(!(op->k==T_PUN && ((op->n==1 && op->s[0]=='.') || (op->n==2 && op->s[0]=='-' && op->s[1]=='>')))) return 0;
  const tok *fn=tat(c,c->i+2); if(fn->k!=T_ID) return 0;
  sdef *S=&c->s[v->sidx]; int fi=-1;
  for(int i=0;i<S->nf;i++) if((int)strlen(S->f[i].name)==fn->n && !strncmp(S->f[i].name,fn->s,fn->n)) fi=i;
  if(fi<0) return 0;
  field f=S->f[fi];
  if(f.is_ptr || f.arr_count || f.sidx>=0) return 0;                  /* a scalar member (plain or bitfield) only */
  if(f.bit_w && v->type.is_volatile) return 0;                        /* a volatile/MMIO bitfield re-read stays a fallback */
  const tok *as=tat(c,c->i+3);
  if(!(as->k==T_PUN && ((as->n==1 && as->s[0]=='=') || is_compound_op(as)))) return 0;
  *out=f; *base=v; return 1;
}
/* Emit the member store `base.field = val` (the C twin of the oracle's _write for a plain member). */
static uint32_t store_member(CC *c, venv *base, const field *f, uint32_t val){
  bcir_ctype st=field_slot(f); val=store_conv(c,val,&st);           /* returns the value stored */
  bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
  if(cl){cl->n_rd=2;cl->rd[0]=base->rid;cl->rd[1]=val;cl->n_imm=2;cl->imm[0]=f->byte_off;cl->imm[1]=f->size;
    cl->bounds=BCIR_BND_ASSUMED;
    if(f->is_bool){cl->imm[2]=1;cl->n_imm=3;}                         /* a _Bool member normalizes on store */
    mark_access(c,cl,f->is_volatile||base->type.is_volatile);
    mark_atomic(cl,f->is_atomic);}                                    /* an `_Atomic` member: an atomic store */
  return val;
}
/* Emit a BITFIELD member store `base.field = val` (#bfassignexpr): read the storage unit (`access_bytes`
 * spanned bytes, into a pow2 temp), insert the masked bits (c.bf.set), store the unit's spanned bytes back
 * -- the same bf.set machinery the STATEMENT-form bitfield store uses, factored out so the value path reuses
 * it. The reload that yields the assignment's VALUE is a plain emit_member (its c.load + c.bf.get). */
static uint32_t store_member_bf(CC *c, venv *base, const field *f, uint32_t val){
  bcir_ctype st=field_slot(f); val=store_conv(c,val,&st);           /* the field's declared type, then the bits */
  int absz=f->access_bytes<=4?4:8;
  uint32_t unit=temp(c,absz);
  bcir_claim *ld=new_claim(c,"c.load",BCIR_OP_LOAD);
  if(ld){ld->n_rd=1;ld->rd[0]=base->rid;ld->n_wr=1;ld->wr[0]=unit;ld->n_imm=2;ld->imm[0]=f->byte_off;ld->imm[1]=f->access_bytes;ld->bounds=BCIR_BND_ASSUMED;
    mark_access(c,ld,f->is_volatile||base->type.is_volatile);}
  uint32_t nu=temp(c,absz);
  bcir_claim *bs=new_claim(c,"c.bf.set",BCIR_OP_ADD);
  if(bs){bs->n_rd=2;bs->rd[0]=unit;bs->rd[1]=val;bs->n_wr=1;bs->wr[0]=nu;bs->n_imm=2;bs->imm[0]=f->bit_off;bs->imm[1]=f->bit_w;}
  bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
  if(cl){cl->n_rd=2;cl->rd[0]=base->rid;cl->rd[1]=nu;cl->n_imm=3;cl->imm[0]=f->byte_off;cl->imm[1]=f->access_bytes;cl->imm[2]=2;
    cl->bounds=BCIR_BND_ASSUMED;                                      /* a bitfield UNIT store: `_v` takes the unit's full type */
    mark_access(c,cl,f->is_volatile||base->type.is_volatile);}
  return val;
}
static uint32_t p_assign(CC *c);   /* fwd: the rhs of an assignment-as-value is itself an assign (right-assoc) */
static uint32_t plv_write(CC *c, plv *lv, uint32_t val){     /* the value as stored */
  if(lv->kind==1) return lv->f.bit_w ? store_member_bf(c,&lv->b,&lv->f,val) : store_member(c,&lv->b,&lv->f,val);
  bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
  if(cl){cl->n_rd=3;cl->rd[0]=lv->b.rid;cl->rd[1]=lv->idx;cl->rd[2]=val;cl->bounds=BCIR_BND_ASSUMED;
    mark_access(c,cl,index_elem_vol(c,&lv->b)); mark_atomic(cl,index_elem_atomic(c,&lv->b));}
  return val;
}
/* `obj = rhs` / `obj OP= rhs` used as a value, the cursor at the operator: the oracle's memory-lvalue `_assign`
 * -- a plain store then a re-read of the same object; a compound read, operation, store, and the stored value
 * (re-read where the store narrows it); an `_Atomic` object one atomic store or read-modify-write. */
static uint32_t plv_assign_value(CC *c, plv *lv){
  const tok *op=&c->t[c->i];
  int at=plv_atomic(c,lv); bcir_ctype slot=plv_slot(lv);
  if(op->k==T_PUN && op->n==1 && op->s[0]=='='){
    c->i++; uint32_t rhs=p_assign(c);
    uint32_t st=plv_write(c,lv,rhs);
    return at ? atomic_assign_value(c,st,&slot) : plv_read(c,lv);
  }
  char ch=op->s[0]; c->i++;
  const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
  if(at){ uint32_t rhs=p_assign(c);
    return lv->kind==1
      ? emit_rmw(c,suf,&slot,lv->b.rid,0,0,1,rhs,lv->f.byte_off,lv->f.size,0,plv_volatile(c,lv),BCIR_BND_ASSUMED)
      : emit_rmw(c,suf,&slot,lv->b.rid,1,lv->idx,1,rhs,0,0,0,plv_volatile(c,lv),BCIR_BND_ASSUMED); }
  uint32_t cur=plv_read(c,lv); uint32_t rhs=p_assign(c);
  uint32_t tmp=binop_result(c,suf,cur,rhs); char o[BCIR_CIR_OP]; fits(c,o,sizeof o,"c.bin.%s",suf);
  bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
  tmp=plv_write(c,lv,tmp);
  return plv_narrow(c,lv,tmp) ? plv_read(c,lv) : tmp;
}
/* `p[i]` of a POINTER variable (`T *p`, a local, parameter or global) to non-volatile scalar storage, subscripted
 * once, the cursor at the name (CF-SPLIT2): an element the scalar array paths read and write as they do an
 * array's, the oracle's `_lvalue(Index)` -- the twin lowered `p[i] += v;` but refused `p[i]++`, `++p[i]` and
 * `x = (p[i] = v)`. A pointer to structs, to pointers or to rows keeps its own paths. */
static int ptr_elem_lv(CC *c, const venv *v){
  if(v->type.kind!=2 || v->type.is_volatile || v->type.ptr_to_struct || v->type.nadims>1 || v->type.size<=0
     || (v->type.ptr_depth?v->type.ptr_depth:1)!=1 || ptr_array(c,v)) return 0;
  return tok_is(tat(c,c->i+1),"[") && sub_group_count(c,c->i+1)==1;
}
/* A MEMORY-lvalue assignment used as a VALUE (the C twin of the oracle's generalized _is_scalar_member_lv
 * value path, #lvassignexpr): an ARRAY ELEMENT `a[i]`, a pointer DEREF `*p` / `*(p+i)`, or a NESTED struct
 * member `o.in.x` -- as a sub-expression `(a[i]=v)+1`, `(*p=v)*2`, `(o.in.x=v)+3`, and chains `a[0]=b[0]=v`.
 * Mirrors the oracle: the lvalue is resolved ONCE (its index/base captured) and then, for a plain `=`, the
 * rhs is STORED through it and the SAME resolved lvalue is RELOADED (the expression's value); for a compound
 * `OP=`, the current value is read, `cur OP rhs` computed + stored, and the BINOP result is the value.
 * Returns 1 (and sets *out) when it handled an eligible form; 0 (cursor unmoved) otherwise, so an ineligible
 * target -- a BITFIELD, an ARRAY-OF-STRUCTS strided element, a MEMBER-ARRAY element, or a VOLATILE/MMIO
 * lvalue -- falls through to the conditional grammar and PARSE-ERRs, routing the function to fallback exactly
 * as the oracle's CLowerError does (the two rails promote the SAME set of forms). */
/* The VALUE of `<member-array element> = rhs` / `OP= rhs` with the cursor ON the `=`/OP= token, for the element
 * `f[idx]` of `v` (its element FIELD `sub` when `soa`): a STRIDED store at member_off (+ field_off) + idx*element-size
 * with the index resolved ONCE, then the same slot re-read; a compound reads, operates, stores and re-reads a
 * narrowing slot; an `_Atomic` slot is one atomic read-modify-write. The member-array (`s.arr[i]`) and the
 * array-of-structs member-array (`a[i].m[j]`, CF-SMALL) value paths share it. */
static uint32_t member_elem_assign_value(CC *c, venv *v, const field *fp, uint32_t idx, int soa, const field *subp){
  const field f=*fp; field sub; if(soa) sub=*subp; else memset(&sub,0,sizeof sub);
  const field *sf = soa ? &sub : &f;                 /* the stored slot: the element FIELD, or the array element */
  const tok *aop=&c->t[c->i];
  int a_eq = aop->k==T_PUN && aop->n==1 && aop->s[0]=='=';
  if(a_eq){                                          /* plain: store rhs, then RELOAD the same strided slot */
    c->i++; uint32_t rhs=p_assign(c);
    uint32_t st=store_member_index(c,v,&f,idx,soa,sf,rhs);
    if(sf->is_atomic){ bcir_ctype ft=field_slot(sf); return atomic_assign_value(c,st,&ft); }
    return soa ? emit_member_index_field(c,v,&f,idx,&sub) : emit_member_index(c,v,&f,idx);
  }
  char ach=aop->s[0]; c->i++;                         /* compound: read cur, binop, store, value = stored */
  if(sf->is_atomic){                                  /* an `_Atomic` element/field: one atomic read-modify-write */
    uint32_t rhs=p_assign(c); bcir_ctype ft=field_slot(sf);
    const char *suf; bcir_opcode oc; compound_binop(ach,&suf,&oc); (void)oc;
    return emit_rmw(c,suf,&ft,v->rid,1,idx,1,rhs,soa?f.byte_off+sub.byte_off:f.byte_off,sf->size,soa?f.size:0,
                    sf->is_volatile||f.is_volatile,BCIR_BND_ASSUMED);
  }
  uint32_t cur = soa ? emit_member_index_field(c,v,&f,idx,&sub) : emit_member_index(c,v,&f,idx);
  uint32_t rhs=p_assign(c);
  const char *suf; bcir_opcode oc; compound_binop(ach,&suf,&oc);
  uint32_t tmp=binop_result(c,suf,cur,rhs); char o[BCIR_CIR_OP]; fits(c,o,sizeof o,"c.bin.%s",suf);
  bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
  tmp=store_member_index(c,v,&f,idx,soa,sf,tmp);      /* the value as stored (converted) */
  /* a sub-int element/field truncates on store -> RE-READ the same strided slot (#narrowcompound); a
   * full-width element/field needs no re-read (oracle: lv.ct.size < rt.size). */
  const bcir_resource *trr=res_of(c->fn,tmp);
  return ((int)sf->size < (int)(trr?trr->elem_bytes:4))
         ? (soa ? emit_member_index_field(c,v,&f,idx,&sub) : emit_member_index(c,v,&f,idx)) : tmp;
}
static BCIR_NOINLINE int lv_assign_value(CC *c, uint32_t *out){   /* out of p_assign's frame (BCIR_NOINLINE) */
  int save=c->i;
  /* --- a deref `*p = rhs` / `*(p + i) = rhs` / their `OP=` as a VALUE --- */
  if(is(c,"*")){
    spec_mark m; spec_begin(c,&m);                  /* the `*(p + i)` index is lowered to reach the operator */
    c->i++;
    { plv lv; memset(&lv,0,sizeof lv); int ms=c->i;    /* `x = (*q->a = v)`: its first element (CF-SPLIT2) */
      if(deref_member_array(c,&lv.b,&lv.f)){ const tok *aop=pk(c);
        if(tok_is(aop,"=") || is_compound_op(aop)){ lv.kind=1; *out=plv_assign_value(c,&lv); return 1; }
        c->i=ms; } }
    venv pvsnap; venv *pv=NULL; uint32_t idx=0; int has_idx=0, ok=0;
    /* SNAPSHOT the pointer's env entry: the `*(p + i)` index and the RHS p_assign below can declare locals
     * and realloc c->env[] -- a pointer into it dangles. The store/emit helpers only READ the venv. */
    if(is(c,"(")){ c->i++;                              /* *(p) or *(p + i) */
      if(isk(c,T_ID)){ tok pid=*pk(c); venv *pvp=lookup(c,&pid);
        if(pvp){ c->i++; pvsnap=*pvp; pv=&pvsnap;
          if(is(c,"+")){ c->i++; idx=p_expr(c); has_idx=1; if(eat(c,")")) ok=1; }
          else if(is(c,")")){ c->i++; ok=1; } } } }
    else if(isk(c,T_ID)){ tok pid=*pk(c); venv *pvp=lookup(c,&pid);
      if(pvp && names_object_ptr(c,pvp)){ c->i++; pvsnap=*pvp; pv=&pvsnap; ok=1; }   /* *p */
      else if(assign_tok_at(c,c->i+1) && deref_lvalue_refused(c,pvp,&pid)) return 0; }   /* `x = (*fp = v)` (CF-FPTAB) */
    const tok *op=&c->t[c->i];
    int is_eq = op->k==T_PUN && op->n==1 && op->s[0]=='=';
    if(ok && pv && pv->type.kind==2 && !pv->type.is_volatile      /* a non-volatile pointer to a scalar pointee */
       && !pv->type.ptr_to_struct                                  /* (a struct's is the oracle's follow-on too) */
       && pv->type.ptr_depth<=1 && (is_eq || is_compound_op(op))){
      int sz = pv->type.size?pv->type.size:4;
      bcir_ctype pst=pointee_slot(&pv->type);           /* `*p` stores a byte copy: C's conversion first */
      int at=index_elem_atomic(c,pv);                   /* an `_Atomic` pointee: atomic accesses (CF-ATOMIC) */
      if(is_eq){                                       /* plain: rhs FIRST, then store, then RELOAD */
        c->i++; uint32_t rhs=p_assign(c);
        if(!has_idx) rhs=store_conv(c,rhs,&pst);
        bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
        if(cl){ if(has_idx){cl->n_rd=3;cl->rd[0]=pv->rid;cl->rd[1]=idx;cl->rd[2]=rhs;}
          else {cl->n_rd=2;cl->rd[0]=pv->rid;cl->rd[1]=rhs;cl->n_imm=2;cl->imm[0]=0;cl->imm[1]=sz;
                if(pst.kind==0 && pst.is_bool){cl->imm[2]=1;cl->n_imm=3;}}   /* a _Bool pointee normalizes */
          cl->bounds=BCIR_BND_ASSUMED; mark_atomic(cl,at); }
        if(at){ *out=atomic_assign_value(c,rhs,&pst); return 1; }   /* the value stored: no second atomic read */
        *out = has_idx ? emit_index(c,pv,idx) : emit_deref(c,pv);   /* reload the SAME resolved lvalue */
        return 1;
      }
      char ch=op->s[0]; c->i++;                         /* compound: read cur, binop, store, value = stored */
      if(at){                                          /* an `_Atomic` pointee: one atomic read-modify-write */
        uint32_t rhs=p_assign(c);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc); (void)oc;
        *out = has_idx ? emit_rmw(c,suf,&pst,pv->rid,1,idx,1,rhs,0,0,0,0,BCIR_BND_ASSUMED)
                       : emit_rmw(c,suf,&pst,pv->rid,0,0,1,rhs,0,sz,0,0,BCIR_BND_ASSUMED);
        return 1;
      }
      uint32_t cur = has_idx ? emit_index(c,pv,idx) : emit_deref(c,pv);
      uint32_t rhs=p_assign(c);
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
      uint32_t tmp=binop_result(c,suf,cur,rhs); char o[BCIR_CIR_OP]; fits(c,o,sizeof o,"c.bin.%s",suf);
      bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
      if(!has_idx) tmp=store_conv(c,tmp,&pst);          /* the value as stored (converted) */
      bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
      if(cl){ if(has_idx){cl->n_rd=3;cl->rd[0]=pv->rid;cl->rd[1]=idx;cl->rd[2]=tmp;}
        else {cl->n_rd=2;cl->rd[0]=pv->rid;cl->rd[1]=tmp;cl->n_imm=2;cl->imm[0]=0;cl->imm[1]=sz;
              if(pst.kind==0 && pst.is_bool){cl->imm[2]=1;cl->n_imm=3;}}     /* a _Bool pointee normalizes */
        cl->bounds=BCIR_BND_ASSUMED; }
      /* the value of a compound is the STORED (narrowed) value: a sub-int target (`unsigned char *p;
       * (*p += v)`) truncates on store, so RE-READ (#narrowcompound); a full-width target needs no
       * re-read (tmp == the stored value), so `res` byte-unchanged (oracle: lv.ct.size < rt.size). */
      const bcir_resource *trr=res_of(c->fn,tmp);
      *out = (sz < (int)(trr?trr->elem_bytes:4)) ? (has_idx ? emit_index(c,pv,idx) : emit_deref(c,pv)) : tmp;
      return 1;
    }
    c->i=save; spec_end(c,&m);   /* not an eligible deref-assignment value: rewind, and the index lowering with
                                  * it -- kept, it ran the index twice (`*(la + j++)` stepped `j` twice) */
  }
  if(!isk(c,T_ID)) return 0;
  tok id=*pk(c); venv *vp=lookup(c,&id); if(!vp) vp=use_global(c,&id);
  if(!vp){ c->i=save; return 0; }
  /* SNAPSHOT the env entry: the value-store paths below resolve an index / RHS via array_index /
   * member_arr_index / p_assign, which can declare locals and realloc c->env[] mid-parse -- a pointer
   * INTO that array dangles afterward. The store/emit helpers only READ the venv (by-value identical). */
  venv vsnap=*vp; venv *v=&vsnap;
  /* --- a DIRECT ARRAY-OF-STRUCTS element FIELD `a[i].f = rhs` / `a[i].f OP= rhs` as a VALUE (strided) --- */
  if(v->sidx>=0 && !v->type.is_volatile && tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='['){
    size_t s_res=c->fn->n_res,s_cl=c->fn->n_claims; uint32_t s_rid=c->rid,s_cid=c->cid,s_clc=c->cl_ctr;
    int istart=c->i; c->i++;                         /* resolve the index ONCE (Horner-flattened) -- and through */
    uint32_t idx=index_chain(c,v);                   /* an element that is a pointer, `pp[i][j].f`, which a flat
                                                      * index read as `pp[i*1+j]` (CF-IDXARROW) */
    if(c->failed) return 0;
    { field af; uint32_t tot; int r=aos_member_array(c,v,idx,&af,&tot);   /* `a[i].m[j]` =/OP= (CF-SMALL) */
      if(r<0) return 0;
      if(r>0){ const tok *aop=&c->t[c->i];
        if(!((aop->k==T_PUN && aop->n==1 && aop->s[0]=='=') || is_compound_op(aop))){   /* a read -> fall through */
          c->fn->n_res=s_res;c->fn->n_claims=s_cl;c->rid=s_rid;c->cid=s_cid;c->cl_ctr=s_clc;
          c->i=istart; c->i=save; return 0; }
        *out=member_elem_assign_value(c,v,&af,tot,0,NULL);
        return 1; } }
    field sub; int got = (is(c,".")||is(c,"->")) && aos_elem_field(c,v,&sub);
    const tok *op=&c->t[c->i];
    int is_eq = op->k==T_PUN && op->n==1 && op->s[0]=='=';
    if(c->failed){ return 0; }
    if(!got || !(is_eq || is_compound_op(op))){       /* not `a[i].field`=/OP= -> roll back, fall through */
      c->fn->n_res=s_res;c->fn->n_claims=s_cl;c->rid=s_rid;c->cid=s_cid;c->cl_ctr=s_clc;
      c->i=istart; c->i=save; return 0;
    }
    if(is_eq){                                         /* plain: store rhs, then RELOAD the same strided slot */
      c->i++; uint32_t rhs=p_assign(c);
      uint32_t st=store_index_field(c,v,idx,&sub,rhs);
      if(sub.is_atomic){ bcir_ctype ft=field_slot(&sub); *out=atomic_assign_value(c,st,&ft); return 1; }
      *out=emit_index_field(c,v,idx,&sub);             /* reload reuses idx + field offset + element stride */
      return 1;
    }
    char ch=op->s[0]; c->i++;                           /* compound: read cur, binop, store, value = stored */
    if(sub.is_atomic){                                 /* an `_Atomic` field: one atomic read-modify-write */
      uint32_t rhs=p_assign(c); bcir_ctype ft=field_slot(&sub);
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc); (void)oc;
      *out=emit_rmw(c,suf,&ft,v->rid,1,idx,1,rhs,sub.byte_off,sub.size,v->type.size,sub.is_volatile,BCIR_BND_ASSUMED);
      return 1;
    }
    uint32_t cur=emit_index_field(c,v,idx,&sub); uint32_t rhs=p_assign(c);
    const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
    uint32_t tmp=binop_result(c,suf,cur,rhs); char o[BCIR_CIR_OP]; fits(c,o,sizeof o,"c.bin.%s",suf);
    bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
    tmp=store_index_field(c,v,idx,&sub,tmp);              /* the value as stored (converted) */
    /* a sub-int element FIELD (`unsigned char c; (a[i].c += v)`) truncates on store, so RE-READ the same
     * strided slot (#narrowcompound, the fixture's aos_narrow); a full-width field needs no re-read. */
    const bcir_resource *trr=res_of(c->fn,tmp);
    *out = ((int)sub.size < (int)(trr?trr->elem_bytes:4)) ? emit_index_field(c,v,idx,&sub) : tmp;
    return 1;
  }
  /* --- an ARRAY ELEMENT `a[i] = rhs` / `a[i] OP= rhs` as a VALUE --- */
  if(tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='['){
    /* eligible only for a plain SCALAR-element array (kind 0, non-volatile) indexed directly to an `=`/OP=
     * -- NOT an array-of-structs `a[i].f`, NOT a member-array (those are reached as a Member, not here). */
    if((v->type.kind!=0 || v->type.is_volatile) && !ptr_elem_lv(c,v)){ c->i=save; return 0; }
    size_t s_res=c->fn->n_res,s_cl=c->fn->n_claims; uint32_t s_rid=c->rid,s_cid=c->cid,s_clc=c->cl_ctr;
    { field gf; c->i++;                               /* `g[i][j] =`/OP= of a multi-dimensional global: the whole
                                                       * object's element at offset 0, as it is read (CF-SMALL) */
      if(global_md_field(c,v,&gf)){
        uint32_t gix=member_arr_index(c,&gf); if(c->failed) return 0;
        const tok *gop=&c->t[c->i];
        if(!((gop->k==T_PUN && gop->n==1 && gop->s[0]=='=') || is_compound_op(gop))){   /* a read -> fall through */
          c->fn->n_res=s_res;c->fn->n_claims=s_cl;c->rid=s_rid;c->cid=s_cid;c->cl_ctr=s_clc; c->i=save; return 0; }
        *out=member_elem_assign_value(c,v,&gf,gix,0,NULL);
        return 1; }
      c->i--; }
    int istart=c->i; c->i++; uint32_t idx=array_index(c,v);   /* resolve the index ONCE (Horner-flattened) */
    const tok *op=&c->t[c->i];
    int is_eq = op->k==T_PUN && op->n==1 && op->s[0]=='=';
    if(!(is_eq || is_compound_op(op))){                /* `a[i]` not followed by `=`/OP= -> a plain value */
      c->fn->n_res=s_res;c->fn->n_claims=s_cl;c->rid=s_rid;c->cid=s_cid;c->cl_ctr=s_clc;
      c->i=istart-1; c->i=save; return 0;
    }
    int at=index_elem_atomic(c,v);                    /* `_Atomic` elements: atomic accesses (CF-ATOMIC) */
    if(is_eq){                                         /* plain: store rhs, then RELOAD the same index */
      c->i++; uint32_t rhs=p_assign(c);
      bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
      if(cl){cl->n_rd=3;cl->rd[0]=v->rid;cl->rd[1]=idx;cl->rd[2]=rhs;cl->bounds=access_bnd(c,v->rid);mark_atomic(cl,at);}
      if(at){ bcir_ctype et=index_elem_ctype(v); *out=atomic_assign_value(c,rhs,&et); return 1; }   /* the value
                                                        * stored, never a re-read */
      *out=emit_index(c,v,idx);                         /* reload reuses idx */
      return 1;
    }
    char ch=op->s[0]; c->i++;                           /* compound: read cur, binop, store, value = stored */
    if(at){                                            /* an `_Atomic` element: one atomic read-modify-write */
      uint32_t rhs=p_assign(c);
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc); (void)oc;
      bcir_ctype et=index_elem_ctype(v);
      *out=emit_rmw(c,suf,&et,v->rid,1,idx,1,rhs,0,0,0,0,access_bnd(c,v->rid));
      return 1;
    }
    uint32_t cur=emit_index(c,v,idx); uint32_t rhs=p_assign(c);
    const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
    uint32_t tmp=binop_result(c,suf,cur,rhs); char o[BCIR_CIR_OP]; fits(c,o,sizeof o,"c.bin.%s",suf);
    bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
    bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
    if(cl){cl->n_rd=3;cl->rd[0]=v->rid;cl->rd[1]=idx;cl->rd[2]=tmp;cl->bounds=access_bnd(c,v->rid);}
    /* the value is the STORED (narrowed) value: a sub-int element (`unsigned short a[]; (a[i] += v)`)
     * truncates on store, so RE-READ the same index (#narrowcompound); a full-width element needs no
     * re-read (oracle: lv.ct.size < rt.size), so `res` stays byte-identical (the #lvassignexpr cases). */
    const bcir_resource *trr=res_of(c->fn,tmp);
    *out = ((int)v->type.size < (int)(trr?trr->elem_bytes:4)) ? emit_index(c,v,idx) : tmp;
    return 1;
  }
  /* --- a NESTED struct member `o.in.x = rhs` / `s->a.b OP= rhs` as a VALUE --- */
  if(v->sidx>=0 && tat(c,c->i+1)->k==T_PUN
     && (tat(c,c->i+1)->s[0]=='.' || (tat(c,c->i+1)->n==2 && tat(c,c->i+1)->s[0]=='-' && tat(c,c->i+1)->s[1]=='>'))){
    c->i++; tok dot=*pk(c); c->i++; tok fld=adv(c); sdef *S=&c->s[v->sidx]; int fi=-1;
    (void)dot;
    for(int k=0;k<S->nf;k++) if((int)strlen(S->f[k].name)==fld.n && !strncmp(S->f[k].name,fld.s,fld.n)) fi=k;
    if(fi<0){ c->i=save; return 0; }
    field f=member_descend(c,S->f[fi]);                /* flatten `o.in.x` -> one offset; cursor past the chain */
    /* --- through the pointer member it loads: `h.next->v = rhs`, `n->next->v OP= rhs`, `s->p[i] = rhs` as a
     * VALUE (CF-SPLIT2) --- */
    if(f.is_ptr && !v->type.is_volatile && (is(c,"->")||is(c,".")||is(c,"["))){
      size_t m_res=c->fn->n_res,m_cl=c->fn->n_claims; uint32_t m_rid=c->rid,m_cid=c->cid,m_clc=c->cl_ctr;
      plv lv; uint32_t ptr=emit_member(c,v,&f,0);
      int ok=plv_chain(c,ptr,f.ptee_sidx,f,&lv);
      if(c->failed) return 0;
      const tok *aop=&c->t[c->i];
      if(!ok || plv_volatile(c,&lv) || !((aop->k==T_PUN && aop->n==1 && aop->s[0]=='=') || is_compound_op(aop))){
        c->fn->n_res=m_res;c->fn->n_claims=m_cl;c->rid=m_rid;c->cid=m_cid;c->cl_ctr=m_clc;   /* a read: undo */
        c->i=save; return 0; }
      *out=plv_assign_value(c,&lv);
      return 1;
    }
    /* --- a MEMBER-ARRAY element `s.arr[i] = rhs` (and member array-of-structs `s.arr[i].f = rhs`) as a
     * VALUE: a STRIDED store at member_off (+ field_off) + idx*element-size, resolved ONCE, then re-read. --- */
    if(f.arr_count && !v->type.is_volatile && is(c,"[")){
      size_t m_res=c->fn->n_res,m_cl=c->fn->n_claims; uint32_t m_rid=c->rid,m_cid=c->cid,m_clc=c->cl_ctr;
      uint32_t idx=member_arr_index(c,&f);             /* the row-major flattened element index, resolved once */
      field sub; int soa = (is(c,".")||is(c,"->")) && elem_field(c,&f,&sub);   /* arr[i].field on AOS */
      if(c->failed) return 0;
      const tok *aop=&c->t[c->i];
      int a_eq = aop->k==T_PUN && aop->n==1 && aop->s[0]=='=';
      if(!(a_eq || is_compound_op(aop)) || (!soa && f.elem_sidx>=0)){   /* a plain value, not an assignment -- or a
                                                       * struct element, the oracle's follow-on -> undo + fall back */
        c->fn->n_res=m_res;c->fn->n_claims=m_cl;c->rid=m_rid;c->cid=m_cid;c->cl_ctr=m_clc;
        c->i=save; return 0; }
      *out=member_elem_assign_value(c,v,&f,idx,soa,&sub);
      return 1;
    }
    const tok *op=&c->t[c->i];
    int is_eq = op->k==T_PUN && op->n==1 && op->s[0]=='=';
    /* eligible for a SCALAR leaf member (plain OR bitfield), non-volatile -- NOT a pointer/array/struct leaf
     * (a pointer field reaching `->` / an array `[` would not land here; a struct leaf with no `=` is a
     * value), and the base must not be a volatile/MMIO struct (a bitfield re-read there would be an extra
     * access -- it stays a fallback, matching the single-level member path + the oracle's gate). */
    if(!(is_eq || is_compound_op(op)) || f.is_ptr || f.arr_count || f.sidx>=0
       || v->type.is_volatile){ c->i=save; return 0; }
    if(is_eq){                                         /* plain: store rhs, then RELOAD the same member */
      c->i++; uint32_t rhs=p_assign(c);
      if(f.bit_w) store_member_bf(c,v,&f,rhs);          /* a nested bitfield: bf.set */
      else { uint32_t st=store_member(c,v,&f,rhs);
        if(f.is_atomic){ bcir_ctype ft=field_slot(&f); *out=atomic_assign_value(c,st,&ft); return 1; } }
      *out=emit_member(c,v,&f,0);                       /* reload: a bitfield re-reads via bf.get (#bfassignexpr) */
      return 1;
    }
    char ch=op->s[0]; c->i++;                           /* compound: read cur, binop, store, value = stored */
    if(f.is_atomic){                                   /* an `_Atomic` member: one atomic read-modify-write */
      uint32_t rhs=p_assign(c); bcir_ctype ft=field_slot(&f);
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc); (void)oc;
      *out=emit_rmw(c,suf,&ft,v->rid,0,0,1,rhs,f.byte_off,f.size,0,f.is_volatile,BCIR_BND_ASSUMED);
      return 1;
    }
    uint32_t cur=emit_member(c,v,&f,0); uint32_t rhs=p_assign(c);
    const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
    uint32_t tmp=binop_result(c,suf,cur,rhs); char o[BCIR_CIR_OP]; fits(c,o,sizeof o,"c.bin.%s",suf);
    bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
    tmp = f.bit_w ? store_member_bf(c,v,&f,tmp) : store_member(c,v,&f,tmp);     /* a nested bitfield: bf.set;
                                                        * the value as stored (converted) */
    /* the value is the STORED (narrowed) value: a sub-int leaf truncates on store, so RE-READ the same
     * member (#narrowcompound); a full-width leaf needs no re-read (oracle: lv.ct.size < rt.size). A BITFIELD
     * narrows to its BIT width (the byte-size test misses it) -- always RE-READ it. */
    const bcir_resource *trr=res_of(c->fn,tmp);
    *out = (f.bit_w || (int)f.size < (int)(trr?trr->elem_bytes:4)) ? emit_member(c,v,&f,0) : tmp;
    return 1;
  }
  c->i=save; return 0;
}
/* `++a` / `a++` / `--a` / `a--` in EXPRESSION position as a VALUE (the C twin of the oracle's _incdec_value,
 * #incdecexpr). Postfix yields the OLD value, prefix the NEW value; the lvalue is resolved ONCE. Covers a
 * NAMED local/param/global (scalar OR pointer), a single-level SCALAR struct member (plain or bitfield), and a
 * plain SCALAR array element -- exactly the lvalue forms the oracle's gate accepts (a NAMED-LOCAL var, or a
 * scalar memory member; a volatile/MMIO target and any other lvalue stay a both-rails fallback, raising via
 * the conditional grammar's parse-error like the oracle's CLowerError). Returns 1 (and sets *out) when it
 * consumed an inc/dec, else 0 (cursor unmoved). The bare-identifier STATEMENT form `a++;` still routes through
 * p_incdec -> a c.copy (unchanged), so existing fixtures stay byte-identical. */
static uint32_t incdec_emit_const1(CC *c){
  uint32_t one=temp(c,4); bcir_claim *kc=new_claim(c,"c.const",BCIR_OP_LOAD);
  if(kc){kc->n_wr=1;kc->wr[0]=one;kc->n_imm=1;kc->imm[0]=1;} return one;   /* the `1` step (an int literal) */
}
/* After a path has resolved its (one-level) lvalue, settle the step operator. POSTFIX: the trailing token MUST
 * be `++`/`--` (consume it); else a deeper lvalue / a plain access -> not handled here. PREFIX: the operator
 * was already consumed before the operand, so the lvalue must be COMPLETE -- no trailing `.`/`->`/`[` (a deeper
 * chain we only partly walked). Returns 1 (OK to proceed) or 0 (roll back to `save` + fall back). */
static int incdec_settle(CC *c, int prefix, int save){
  if(!prefix){ if(!(is(c,"++")||is(c,"--"))){ c->i=save; return 0; } c->i++; return 1; }
  if(is(c,".")||is(c,"->")||is(c,"[")){ c->i=save; return 0; }   /* prefix: a deeper lvalue -> fall back */
  return 1;
}
/* SNAPSHOT a named local's OLD value via a same-type cast (`c.cast` DECLARES a fresh temp; a plain copy would
 * not) -- _read of a named local hands back the MUTABLE storage rid, which the store below would clobber. The
 * cast SPELLING is the width-named UNSIGNED type (uintN_t / `T *`), matching the oracle's _cast_name; the temp
 * carries the var's OWN (sign/float/pointer) type so it declares + reads back correctly. */
static uint32_t incdec_snapshot(CC *c, venv *v){
  bcir_ctype st=v->type; int sz=st.size?st.size:4;
  uint32_t old;
  if(st.kind==2){                                          /* a POINTER snapshot -> a real `T *` temp (kind ptr) */
    old=add_res(c,BCIR_DOM_RAM, sz?sz:4, 1,0,BCIR_RK_POINTER,"");
    if(c->fn->n_res){ bcir_resource *pr=&c->fn->res[c->fn->n_res-1];   /* carry the pointee (width/sign/struct) */
      pr->is_signed=(uint8_t)(st.signd?1:0); pr->is_float=(uint8_t)(st.is_float?1:0); pr->ptr_depth=st.ptr_depth;
      pr->is_plain_char=(uint8_t)(st.is_plain_char?1:0);
      if(st.ptr_to_struct) fits(c,pr->agg,BCIR_CIR_AGG,"%s %s",st.is_union?"union":"struct",st.tag);
      else if(st.ptr_to_fp) ptee_fp(c,pr,&st); }
  } else {
    old = st.is_float ? tempf(c,sz) : tempi(c,sz,st.signd?1:0);
    if(c->fn->n_res && st.is_plain_char) c->fn->res[c->fn->n_res-1].is_plain_char=1;
  }
  char op[BCIR_CIR_OP]; cast_name(c,&st,0,op,sizeof op);   /* width-named UNSIGNED spelling, like _cast_name */
  bcir_claim *cl=new_claim(c,op,BCIR_OP_ADD); if(cl){cl->n_rd=1;cl->rd[0]=v->rid;cl->n_wr=1;cl->wr[0]=old;}
  return old;
}
/* The VALUE of `++`/`--` (prefix or postfix, already settled) on the member-array element `f[idx]` of `v` (its
 * element FIELD `sub` when `soa`): read, step by one, store back at the same strided slot; a prefix value re-reads
 * a narrowing slot, a postfix one is the old value; an `_Atomic` slot is one atomic read-modify-write. The
 * member-array (`s.arr[i]++`) and array-of-structs member-array (`a[i].m[j]++`, CF-SMALL) inc/dec paths share it. */
static uint32_t member_elem_incdec(CC *c, venv *v, const field *fp, uint32_t idx, int soa, const field *subp,
                                   int prefix, char ch){
  const field f=*fp; field sub; if(soa) sub=*subp; else memset(&sub,0,sizeof sub);
  const field *sf = soa ? &sub : &f;                  /* the stored slot: the element FIELD, or the element */
  const char *suf = ch=='+'?"add":"sub"; bcir_opcode oc = ch=='+'?BCIR_OP_ADD:BCIR_OP_SUB;
  if(sf->is_atomic){                                  /* an `_Atomic` element/field: one atomic RMW */
    bcir_ctype ft=field_slot(sf);
    return emit_rmw(c,incdec_kind(prefix,ch),&ft,v->rid,1,idx,0,0,soa?f.byte_off+sub.byte_off:f.byte_off,sf->size,
                    soa?f.size:0,sf->is_volatile||f.is_volatile,BCIR_BND_ASSUMED);
  }
  uint32_t cur = soa ? emit_member_index_field(c,v,&f,idx,&sub) : emit_member_index(c,v,&f,idx);
  uint32_t one=incdec_emit_const1(c);
  uint32_t nw=binop_result(c,suf,cur,one); char o[BCIR_CIR_OP]; fits(c,o,sizeof o,"c.bin.%s",suf);
  bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=one;b->n_wr=1;b->wr[0]=nw;}
  store_member_index(c,v,&f,idx,soa,sf,nw);
  if(!prefix) return cur;
  const bcir_resource *nr=res_of(c->fn,nw);
  return ((int)sf->size < (int)(nr?nr->elem_bytes:4))
         ? (soa ? emit_member_index_field(c,v,&f,idx,&sub) : emit_member_index(c,v,&f,idx)) : nw;
}
/* `++`/`--` (prefix or postfix, already settled) on an object through a loaded pointer (CF-SPLIT2): read, step by
 * one, store back through the same pointer; a postfix value is the old value, a prefix one the stored new value
 * (re-read where the store narrows it); an `_Atomic` object one atomic read-modify-write. */
static uint32_t plv_incdec(CC *c, plv *lv, int prefix, char ch){
  if(plv_atomic(c,lv)){ bcir_ctype slot=plv_slot(lv);
    return lv->kind==1
      ? emit_rmw(c,incdec_kind(prefix,ch),&slot,lv->b.rid,0,0,0,0,lv->f.byte_off,lv->f.size,0,plv_volatile(c,lv),
                 BCIR_BND_ASSUMED)
      : emit_rmw(c,incdec_kind(prefix,ch),&slot,lv->b.rid,1,lv->idx,0,0,0,0,0,plv_volatile(c,lv),BCIR_BND_ASSUMED); }
  const char *suf = ch=='+'?"add":"sub"; bcir_opcode oc = ch=='+'?BCIR_OP_ADD:BCIR_OP_SUB;
  uint32_t cur=plv_read(c,lv);                           /* the old value: a fresh declared temp */
  uint32_t one=incdec_emit_const1(c);
  uint32_t nw=binop_result(c,suf,cur,one); char o[BCIR_CIR_OP]; fits(c,o,sizeof o,"c.bin.%s",suf);
  bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=one;b->n_wr=1;b->wr[0]=nw;}
  (void)plv_write(c,lv,nw);
  if(!prefix) return cur;
  return plv_narrow(c,lv,nw) ? plv_read(c,lv) : nw;
}
/* `++*p`, `--*p`, `++(*p)`, `(*p)++`, `(*p)--` (CF-SPLIT2): a step of the pointee of a pointer to non-volatile scalar
 * storage, read, stepped and stored back through it as `*p += 1` is -- the oracle's `_lvalue(Deref)`, which the twin
 * had refused. The operand is a named pointer (`*p`), the element spelling `*(p + i)` (stepped as `p[i]` is, as the
 * oracle lowers it), or any other pointer value (`*&a`, `*q[j]`), lowered once. 1 with the value (postfix the old
 * one, prefix the stored new one) for these forms, else 0 with the cursor and the claims unmoved (`*p++` is
 * `*(p++)`, a step of `p`). */
static int deref_tail(const tok *t){   /* a postfix operator after a name: the operand is more than the name */
  return tok_is(t,"[")||tok_is(t,".")||tok_is(t,"->")||tok_is(t,"(")||tok_is(t,"++")||tok_is(t,"--");
}
static int deref_ptr_ok(const venv *v){   /* a named pointer to non-volatile scalar storage */
  return v->type.kind==2 && !v->type.is_volatile && !v->type.ptr_to_struct && (v->type.ptr_depth?v->type.ptr_depth:1)==1
         && v->type.size>0;
}
static int deref_incdec(CC *c, uint32_t *out){
  int k=c->i, prefix=0, paren=0; char ch=0;
  if(is(c,"++")||is(c,"--")){ prefix=1; ch=pk(c)->s[0]; k++; }
  if(tok_is(tat(c,k),"(") && tok_is(tat(c,k+1),"*")){ paren=1; k++; }
  if(!tok_is(tat(c,k),"*") || (!prefix && !paren)) return 0;
  int save=c->i; spec_mark m; spec_begin(c,&m);
  int form=0; venv pv; memset(&pv,0,sizeof pv); uint32_t base=0, idx=0;
  const tok *t1=tat(c,k+1);
  plv mlv; memset(&mlv,0,sizeof mlv);
  venv ab; field af; memset(&ab,0,sizeof ab); memset(&af,0,sizeof af);
  c->i=k+1;
  if(deref_member_array(c,&mlv.b,&mlv.f)){ mlv.kind=1; form=4; }   /* `*q->a`: its first element */
  else c->i=save;
  if(!form){ c->i=k+1;                                           /* `(*(_Atomic T *)((char *)p + K))++`: the atomic slot */
    if(byte_off_access(c,&ab,&af)){                              /* the emit spells an atomic member's step so (CF-RTFP) */
      if(af.is_atomic) form=5;
      else { spec_end(c,&m); spec_begin(c,&m); c->i=save; } }
    else c->i=save; }
  if(form){}
  else if(t1->k==T_ID && !deref_tail(tat(c,k+2))){               /* `*p` */
    venv *vp=lookup(c,t1); if(!vp) vp=use_global(c,t1);
    if(vp && deref_ptr_ok(vp)){ pv=*vp; form=1; c->i=k+2; }
    else if((prefix || (tok_is(tat(c,k+2),")") && (tok_is(tat(c,k+3),"++") || tok_is(tat(c,k+3),"--"))))
            && deref_lvalue_refused(c,vp,t1)) return 0;         /* `++*s`, `(*fp)++` -- not `(*fp)(x)` (CF-FPTAB) */
  } else if(tok_is(t1,"(") && tat(c,k+2)->k==T_ID && tok_is(tat(c,k+3),"+")){   /* `*(p + i)`: `p[i]` */
    venv *vp=lookup(c,tat(c,k+2)); if(!vp) vp=use_global(c,tat(c,k+2));
    if(vp && deref_ptr_ok(vp) && !ptr_array(c,vp)){ pv=*vp; c->i=k+4; idx=p_expr(c);
      if(!c->failed && eat(c,")")) form=2; }
  } else {                                                       /* any other pointer value */
    c->i=k+1; base=p_unary(c);
    const bcir_resource *br=c->failed?NULL:res_of(c->fn,base);
    if(br && br->kind==BCIR_RK_POINTER && (br->ptr_depth?br->ptr_depth:1)==1 && !br->agg[0] && !br->is_voidptr
       && !br->is_volatile && br->domain!=BCIR_DOM_MMIO) form=3;
  }
  if(form && paren){ if(is(c,")")) c->i++; else form=0; }
  if(form && !prefix){ const tok *t=pk(c); if(tok_is(t,"++")||tok_is(t,"--")){ ch=t->s[0]; c->i++; } else form=0; }
  if(!form || c->failed){ if(c->failed) return 0; spec_end(c,&m); c->i=save; return 0; }
  if(form==4){ *out=plv_incdec(c,&mlv,prefix,ch); return 1; }   /* a member array's first element */
  if(form==5){ bcir_ctype ft=field_slot(&af);                    /* an `_Atomic` slot: one atomic read-modify-write */
    *out=emit_rmw(c,incdec_kind(prefix,ch),&ft,ab.rid,0,0,0,0,af.byte_off,af.size,0,af.is_volatile,BCIR_BND_ASSUMED);
    return 1; }
  const char *suf = ch=='+'?"add":"sub"; bcir_opcode oc = ch=='+'?BCIR_OP_ADD:BCIR_OP_SUB;
  if(form==2){                                                   /* an element: the scalar array path's step */
    if(index_elem_atomic(c,&pv)){ bcir_ctype et=index_elem_ctype(&pv);
      *out=emit_rmw(c,incdec_kind(prefix,ch),&et,pv.rid,1,idx,0,0,0,0,0,0,access_bnd(c,pv.rid)); return 1; }
    uint32_t cur=emit_index(c,&pv,idx), one=incdec_emit_const1(c);
    uint32_t nw=binop_result(c,suf,cur,one); char o[BCIR_CIR_OP]; fits(c,o,sizeof o,"c.bin.%s",suf);
    bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=one;b->n_wr=1;b->wr[0]=nw;}
    bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
    if(cl){cl->n_rd=3;cl->rd[0]=pv.rid;cl->rd[1]=idx;cl->rd[2]=nw;cl->bounds=access_bnd(c,pv.rid);}
    if(!prefix){ *out=cur; return 1; }
    const bcir_resource *nr=res_of(c->fn,nw);
    *out = ((int)pv.type.size < (int)(nr?nr->elem_bytes:4)) ? emit_index(c,&pv,idx) : nw;
    return 1;
  }
  int sz; bcir_ctype pst; int at;                                 /* read before new resources move res[] */
  if(form==1){ sz=pv.type.size; pst=pointee_slot(&pv.type); at=index_elem_atomic(c,&pv); }
  else { const bcir_resource *br=res_of(c->fn,base); sz=br->elem_bytes?(int)br->elem_bytes:4;
    pst=res_pointee_slot(br); at=br->is_atomic; }
  uint32_t ptr = form==1 ? pv.rid : base;
  if(at){ *out=emit_rmw(c,incdec_kind(prefix,ch),&pst,ptr,0,0,0,0,0,sz,0,0,BCIR_BND_ASSUMED); return 1; }
  uint32_t cur = form==1 ? emit_deref(c,&pv) : emit_deref_rid(c,base);   /* the old value: a fresh temp */
  uint32_t one=incdec_emit_const1(c);
  uint32_t nw=binop_result(c,suf,cur,one); char o[BCIR_CIR_OP]; fits(c,o,sizeof o,"c.bin.%s",suf);
  bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=one;b->n_wr=1;b->wr[0]=nw;}
  nw=store_conv(c,nw,&pst);                                      /* the value as stored */
  bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
  if(cl){cl->n_rd=2;cl->rd[0]=ptr;cl->rd[1]=nw;cl->n_imm=2;cl->imm[0]=0;cl->imm[1]=sz;cl->bounds=BCIR_BND_ASSUMED;
    if(pst.kind==0 && pst.is_bool){cl->imm[2]=1;cl->n_imm=3;}}   /* a _Bool pointee normalizes */
  if(!prefix){ *out=cur; return 1; }
  const bcir_resource *nr=res_of(c->fn,nw);
  *out = (sz < (int)(nr?nr->elem_bytes:4)) ? (form==1 ? emit_deref(c,&pv) : emit_deref_rid(c,base)) : nw;
  return 1;
}
static BCIR_NOINLINE int incdec_value(CC *c, uint32_t *out){   /* out of p_unary_inner's frame (BCIR_NOINLINE) */
  int prefix=0; char ch=0;
  if(deref_incdec(c,out)) return 1;                        /* `++*p` / `(*p)++` (CF-SPLIT2) */
  if(c->failed) return 0;
  if(is(c,"++")||is(c,"--")){ prefix=1; ch=pk(c)->s[0]; }   /* a PREFIX `++`/`--` -- the operand follows */
  else if(isk(c,T_ID)){
    /* a POSTFIX `name <tail> ++` -- only if a `++`/`--` immediately follows the (bare / member / index)
     * lvalue. Scan past a single `.field`/`->field` or a balanced `[...]` chain off the name; if the next
     * token is not `++`/`--`, this is not an inc/dec -- bail (cursor unmoved) so the value reads normally. */
    int j=c->i+1;
    for(;;){ const tok *t=tat(c,j);   /* bounded: a `j+=2` member chain off the tail must stay in-array (Bug B) */
      if(t->k==T_PUN && t->n==1 && t->s[0]=='.'){ j+=2; continue; }
      if(t->k==T_PUN && t->n==2 && t->s[0]=='-' && t->s[1]=='>'){ j+=2; continue; }
      if(t->k==T_PUN && t->n==1 && t->s[0]=='['){ int d=1; j++;
        while(tat(c,j)->k!=T_END && d){ char k0=tat(c,j)->s[0]; if(k0=='[')d++; else if(k0==']')d--; j++; } continue; }
      break; }
    { const tok *tj=tat(c,j);
      if(!(tj->k==T_PUN && tj->n==2 && (tj->s[0]=='+'||tj->s[0]=='-') && tj->s[1]==tj->s[0]))
        return 0;                                          /* the lvalue is not stepped -> not an inc/dec */
      ch=tj->s[0]; }
  } else return 0;
  int save=c->i;
  if(prefix) c->i++;                                       /* consume the leading `++`/`--` */
  if(!isk(c,T_ID)){ c->i=save; return 0; }                 /* only a named-rooted lvalue is supported */
  tok id=*pk(c); venv *vp=lookup(c,&id); if(!vp) vp=use_global(c,&id);
  if(!vp && prefix && declared_ret(c,&id) && !postfix_follows(c,c->i+1)){   /* `++f`: a function is no object */
    fail(c,CC_FN_NOT_LVALUE); return 0; }                                 /* (CF-EXTDESIG) */
  if(!vp){ c->i=save; return 0; }
  /* SNAPSHOT the env entry: the indexed inc/dec paths below resolve the index via array_index (a p_expr
   * sub-parse) before reading v->rid/sidx/type -- a stmt-expr index can realloc c->env[] and dangle a
   * pointer into it. The by-value copy is byte-identical (the helpers only READ the venv). */
  venv vsnap=*vp; venv *v=&vsnap;
  const char *suf = ch=='+'?"add":"sub"; bcir_opcode oc = ch=='+'?BCIR_OP_ADD:BCIR_OP_SUB;

  /* --- a DIRECT array-of-structs element FIELD `a[i].f` (strided), non-volatile --- */
  if(isk(c,T_ID) && v->sidx>=0 && !v->type.is_volatile
     && tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='['){
    size_t s_res=c->fn->n_res,s_cl=c->fn->n_claims; uint32_t s_rid=c->rid,s_cid=c->cid,s_clc=c->cl_ctr;
    c->i++; uint32_t idx=index_chain(c,v);                 /* resolve the index ONCE (Horner-flattened), and through
                                                            * an element that is a pointer (CF-IDXARROW) */
    if(c->failed) return 0;
    { field af; uint32_t tot; int r=aos_member_array(c,v,idx,&af,&tot);   /* `a[i].m[j]++` (CF-SMALL) */
      if(r<0) return 0;
      if(r>0){ if(!incdec_settle(c,prefix,save)){
          c->fn->n_res=s_res;c->fn->n_claims=s_cl;c->rid=s_rid;c->cid=s_cid;c->cl_ctr=s_clc; c->i=save; return 0; }
        *out=member_elem_incdec(c,v,&af,tot,0,NULL,prefix,ch);
        return 1; } }
    field sub; int got=(is(c,".")||is(c,"->")) && aos_elem_field(c,v,&sub);
    if(c->failed) return 0;
    if(!got || !incdec_settle(c,prefix,save)){             /* not `a[i].field++` -> roll back, fall through */
      c->fn->n_res=s_res;c->fn->n_claims=s_cl;c->rid=s_rid;c->cid=s_cid;c->cl_ctr=s_clc; c->i=save; return 0; }
    if(sub.is_atomic){                                     /* an `_Atomic` field: one atomic read-modify-write */
      bcir_ctype ft=field_slot(&sub);
      *out=emit_rmw(c,incdec_kind(prefix,ch),&ft,v->rid,1,idx,0,0,sub.byte_off,sub.size,v->type.size,sub.is_volatile,
                    BCIR_BND_ASSUMED);
      return 1;
    }
    uint32_t cur=emit_index_field(c,v,idx,&sub);           /* read OLD (a fresh declared temp -- no snapshot) */
    uint32_t one=incdec_emit_const1(c);
    uint32_t nw=binop_result(c,suf,cur,one); char o[BCIR_CIR_OP]; fits(c,o,sizeof o,"c.bin.%s",suf);
    bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=one;b->n_wr=1;b->wr[0]=nw;}
    store_index_field(c,v,idx,&sub,nw);
    if(!prefix){ *out=cur; return 1; }
    const bcir_resource *nr=res_of(c->fn,nw);
    *out = ((int)sub.size < (int)(nr?nr->elem_bytes:4)) ? emit_index_field(c,v,idx,&sub) : nw;
    return 1;
  }

  /* --- a plain SCALAR ARRAY ELEMENT `a[i]` (kind 0, non-volatile, NOT an array-of-structs) --- */
  if(isk(c,T_ID) && ((v->type.kind==0 && !v->type.is_volatile && v->sidx<0) || ptr_elem_lv(c,v))
     && tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='['){
    size_t s_res=c->fn->n_res,s_cl=c->fn->n_claims; uint32_t s_rid=c->rid,s_cid=c->cid,s_clc=c->cl_ctr;
    c->i++;
    { field gf; if(global_md_field(c,v,&gf)){          /* `g[i][j]++` of a multi-dimensional global (CF-SMALL) */
        uint32_t gix=member_arr_index(c,&gf); if(c->failed) return 0;
        if(!incdec_settle(c,prefix,save)){
          c->fn->n_res=s_res;c->fn->n_claims=s_cl;c->rid=s_rid;c->cid=s_cid;c->cl_ctr=s_clc; c->i=save; return 0; }
        *out=member_elem_incdec(c,v,&gf,gix,0,NULL,prefix,ch);
        return 1; } }
    uint32_t idx=array_index(c,v);                         /* resolve the index ONCE (Horner-flattened) */
    if(c->failed){ return 0; }
    if(!incdec_settle(c,prefix,save)){                     /* `a[i]` not stepped -> roll back, fall through */
      c->fn->n_res=s_res;c->fn->n_claims=s_cl;c->rid=s_rid;c->cid=s_cid;c->cl_ctr=s_clc; c->i=save; return 0; }
    if(index_elem_atomic(c,v)){                            /* an `_Atomic` element: one atomic read-modify-write */
      bcir_ctype et=index_elem_ctype(v);
      *out=emit_rmw(c,incdec_kind(prefix,ch),&et,v->rid,1,idx,0,0,0,0,0,0,access_bnd(c,v->rid));
      return 1;
    }
    uint32_t cur=emit_index(c,v,idx);                      /* read OLD (a fresh declared temp -- no snapshot) */
    uint32_t one=incdec_emit_const1(c);
    uint32_t nw=binop_result(c,suf,cur,one); char o[BCIR_CIR_OP]; fits(c,o,sizeof o,"c.bin.%s",suf);
    bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=one;b->n_wr=1;b->wr[0]=nw;}
    bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
    if(cl){cl->n_rd=3;cl->rd[0]=v->rid;cl->rd[1]=idx;cl->rd[2]=nw;cl->bounds=access_bnd(c,v->rid);}
    if(!prefix){ *out=cur; return 1; }                     /* postfix: the OLD value */
    const bcir_resource *nr=res_of(c->fn,nw);              /* prefix: re-read if the element narrows on store */
    *out = ((int)v->type.size < (int)(nr?nr->elem_bytes:4)) ? emit_index(c,v,idx) : nw;
    return 1;
  }

  /* --- a struct MEMBER `s.x` / `p->x` / nested `o.in.x`, OR a member array `s.arr[i]` / `s.arr[i].f`,
   *     non-volatile (the C twin of the oracle's scalar-member-lvalue inc/dec). --- */
  if(isk(c,T_ID) && v->sidx>=0 && !v->type.is_volatile && tat(c,c->i+1)->k==T_PUN
     && ((tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='.')
         || (tat(c,c->i+1)->n==2 && tat(c,c->i+1)->s[0]=='-' && tat(c,c->i+1)->s[1]=='>'))){
    size_t s_res=c->fn->n_res,s_cl=c->fn->n_claims; uint32_t s_rid=c->rid,s_cid=c->cid,s_clc=c->cl_ctr;
    sdef *S=&c->s[v->sidx]; c->i++; (void)adv(c); tok fld=adv(c); int fi=-1;   /* consume `. field` / `-> field` */
    for(int i=0;i<S->nf;i++) if((int)strlen(S->f[i].name)==fld.n && !strncmp(S->f[i].name,fld.s,fld.n)) fi=i;
    if(fi<0){ c->i=save; return 0; }
    field f=member_descend(c,S->f[fi]);                   /* flatten `o.in.x` -> one offset; cursor past the chain */
    if(f.is_ptr && (is(c,"->")||is(c,".")||is(c,"["))){   /* on through the pointer member it loads: `h.next->v++`,
                                                           * `--n->next->v`, `s->p[i]++` (CF-SPLIT2) */
      plv lv; uint32_t ptr=emit_member(c,v,&f,0);
      int ok=plv_chain(c,ptr,f.ptee_sidx,f,&lv);
      if(c->failed) return 0;
      if(!ok || plv_volatile(c,&lv) || !incdec_settle(c,prefix,save)){
        c->fn->n_res=s_res;c->fn->n_claims=s_cl;c->rid=s_rid;c->cid=s_cid;c->cl_ctr=s_clc; c->i=save; return 0; }
      *out=plv_incdec(c,&lv,prefix,ch);
      return 1;
    }
    if(f.arr_count && is(c,"[")){                         /* a MEMBER-ARRAY element `s.arr[i]` / `s.arr[i].f` */
      uint32_t idx=member_arr_index(c,&f);
      field sub; int soa=(is(c,".")||is(c,"->")) && elem_field(c,&f,&sub);
      if(c->failed) return 0;
      if(!soa && f.elem_ptr){                             /* a pointer element steps by its pointee, as a pointer
        * member would: both rails refuse that form (the oracle's `_incdec` lvalue gate), so it is not stepped
        * here as an integer */
        fail(c,"inc/dec of this lvalue form is a follow-on"); return 0; }
      if(!incdec_settle(c,prefix,save)){
        c->fn->n_res=s_res;c->fn->n_claims=s_cl;c->rid=s_rid;c->cid=s_cid;c->cl_ctr=s_clc; c->i=save; return 0; }
      *out=member_elem_incdec(c,v,&f,idx,soa,&sub,prefix,ch);
      return 1;
    }
    if(f.is_ptr || f.arr_count || f.sidx>=0){ c->i=save; return 0; }   /* a non-scalar leaf -> fallback */
    if(!incdec_settle(c,prefix,save)){                    /* a deeper lvalue / not stepped -> roll back */
      c->fn->n_res=s_res;c->fn->n_claims=s_cl;c->rid=s_rid;c->cid=s_cid;c->cl_ctr=s_clc; c->i=save; return 0; }
    if(f.is_atomic){                                      /* an `_Atomic` member: one atomic read-modify-write */
      bcir_ctype ft=field_slot(&f);
      *out=emit_rmw(c,incdec_kind(prefix,ch),&ft,v->rid,0,0,0,0,f.byte_off,f.size,0,f.is_volatile,BCIR_BND_ASSUMED);
      return 1;
    }
    uint32_t cur=emit_member(c,v,&f,0);                   /* read OLD (a fresh declared temp -- no snapshot) */
    uint32_t one=incdec_emit_const1(c);
    uint32_t nw=binop_result(c,suf,cur,one); char o[BCIR_CIR_OP]; fits(c,o,sizeof o,"c.bin.%s",suf);
    bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=one;b->n_wr=1;b->wr[0]=nw;}
    if(f.bit_w) store_member_bf(c,v,&f,nw); else store_member(c,v,&f,nw);
    if(!prefix){ *out=cur; return 1; }                    /* postfix: the OLD value */
    const bcir_resource *nr=res_of(c->fn,nw);             /* prefix: the STORED new value -- re-read if it */
    *out = (f.bit_w || (int)f.size < (int)(nr?nr->elem_bytes:4)) ? emit_member(c,v,&f,0) : nw;   /* narrows */
    return 1;
  }

  /* --- a NAMED local/param/global (scalar OR pointer) `a` -- the bare name followed by the step --- */
  c->i++;                                                  /* consume the name */
  if(!incdec_settle(c,prefix,save)) return 0;              /* postfix: consume `++`/`--`; prefix: lvalue done */
  if(v->type.is_atomic && v->type.kind==0){                /* an `_Atomic` object: one atomic read-modify-write */
    *out=emit_rmw(c,incdec_kind(prefix,ch),&v->type,v->rid,0,0,0,0,0,v->type.size?v->type.size:4,0,
                  v->type.is_volatile,BCIR_BND_ASSUMED);
    return 1;
  }
  if(v->type.is_volatile){ c->i=save; return 0; }          /* a volatile/MMIO target stays a both-rails fallback */
  if(v->type.kind==2){                                     /* a POINTER local steps by element (in place) */
    uint32_t old = !prefix ? incdec_snapshot(c,v) : 0;     /* postfix: snapshot the pre-step pointer first */
    uint32_t one=incdec_emit_const1(c);
    char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.ptr%s",ch=='+'?"add":"sub");
    bcir_claim *cl=new_claim(c,op,BCIR_OP_ADD); if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=one;cl->n_wr=1;cl->wr[0]=v->rid;}
    *out = prefix ? v->rid : old;                          /* prefix: the stepped pointer; postfix: the snapshot */
    return 1;
  }
  /* a SCALAR named local: cur ± 1, stored back into the (addressable) storage via a memory c.store */
  uint32_t old = !prefix ? incdec_snapshot(c,v) : 0;       /* postfix: snapshot OLD before the store clobbers it */
  uint32_t one=incdec_emit_const1(c);
  uint32_t nw=binop_result(c,suf,v->rid,one); char o[BCIR_CIR_OP]; fits(c,o,sizeof o,"c.bin.%s",suf);
  bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=v->rid;b->rd[1]=one;b->n_wr=1;b->wr[0]=nw;}
  int sz=v->type.size?v->type.size:4;
  { bcir_ctype st=v->type; nw=store_conv(c,nw,&st); }      /* a byte copy asks C's conversion (x +/- 1 keeps
                                                            * x's class, so it is the value itself) */
  bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
  if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=nw;cl->n_imm=2;cl->imm[0]=0;cl->imm[1]=sz;cl->bounds=BCIR_BND_ASSUMED;
    if(v->type.is_bool){cl->imm[2]=1;cl->n_imm=3;}}        /* a _Bool local normalizes on store */
  if(!prefix){ *out=old; return 1; }                       /* postfix: the OLD value */
  const bcir_resource *nr=res_of(c->fn,nw);                /* prefix: the STORED new value -- re-read (the var's */
  *out = ((int)sz < (int)(nr?nr->elem_bytes:4)) ? v->rid : nw;   /* storage rid) if a sub-int target narrows */
  return 1;
}
/* The assignment-expression level (lowest precedence, RIGHT-associative): `name = assign` / `name OP= assign`
 * evaluates to the assigned value (the named variable's storage), mirroring the statement forms in p_stmt and
 * the oracle's _assign. Only a NAMED variable target (local/param/global) is handled as a value; anything else
 * falls through to the conditional grammar (so a member/array-lvalue assignment used as a value falls back). */
static uint32_t p_assign(CC *c){
  if(name_assign_ahead(c)){
    venv *vp=lookup(c,&c->t[c->i]); if(!vp) vp=use_global(c,&c->t[c->i]);
    /* SNAPSHOT the env entry into a LOCAL before any RHS sub-parse: p_assign below calls back into the
     * expression grammar, which can declare locals (a `({...})` stmt-expr / a `use_global`) and realloc
     * c->env[] -- a pointer INTO that array dangles after the move. The helpers only READ the venv, so a
     * by-value copy is byte-identical (#532 class). */
    venv vsnap; venv *v=NULL; if(vp){ vsnap=*vp; v=&vsnap; }
    if(v){
      const tok *op=tat(c,c->i+1);
      if(op->n==1 && op->s[0]=='='){                 /* name = rhs  (right-recursive: a = b = c) */
        tok tnm=c->t[c->i]; c->i+=2; int ist=c->i; uint32_t rhs=null_pointer(c,p_assign(c),&v->type); int ien=c->i;
        if(!agg_value_ok(c,obj_sidx(c,v),rhs,agg_assigned)) return 0;   /* a struct takes its own type */
        bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD); if(cl){cl->n_rd=1;cl->rd[0]=rhs;cl->n_wr=1;cl->wr[0]=v->rid;}
        mark_obj_write(c,cl,v);
        bind_extent(c,v->rid,res_of(c->fn,v->rid),&tnm,ist,ien);   /* §5.12: `p = malloc(N*…)` -> N */
        if(v->type.is_atomic && v->type.kind==0)     /* an `_Atomic` object: the value stored, converted -- a */
          return atomic_assign_value(c,rhs,&v->type);   /* re-read by name would be a second atomic access */
        return v->rid;
      }
      char ch=op->s[0];                              /* name OP= rhs */
      if(v->type.kind==2 && (ch=='+'||ch=='-')){     /* pointer += / -= : a single pointer-arith claim */
        c->i+=2; uint32_t rhs=p_assign(c);
        char o[BCIR_CIR_OP]; fits(c,o,sizeof o,"c.ptr%s",ch=='+'?"add":"sub");
        bcir_claim *cl=new_claim(c,o,BCIR_OP_ADD); if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=rhs;cl->n_wr=1;cl->wr[0]=v->rid;}
        return v->rid;
      }
      if(v->type.is_atomic && v->type.kind==0){     /* an `_Atomic` object: one atomic read-modify-write of it */
        c->i+=2; uint32_t rhs=p_assign(c);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc); (void)oc;
        return emit_rmw(c,suf,&v->type,v->rid,0,0,1,rhs,0,v->type.size?v->type.size:4,0,v->type.is_volatile,
                        BCIR_BND_ASSUMED);
      }
      uint32_t cur=named_read(c,v);                 /* the current value (a volatile read, first) */
      c->i+=2; uint32_t rhs=p_assign(c);
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
      uint32_t tmp=binop_result(c,suf,cur,rhs); char o[BCIR_CIR_OP]; fits(c,o,sizeof o,"c.bin.%s",suf);
      bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
      bcir_claim *cp=new_claim(c,"c.copy",BCIR_OP_ADD); if(cp){cp->n_rd=1;cp->rd[0]=tmp;cp->n_wr=1;cp->wr[0]=v->rid;}
      mark_obj_write(c,cp,v);
      return v->rid;
    }
  }
  field mf; venv *mbp;
  if(member_assign_ahead(c,&mf,&mbp)){             /* `p->x = rhs` / `s.x OP= rhs` as a VALUE (store + reload) */
    /* SNAPSHOT the struct base env entry into a LOCAL before the RHS sub-parse below reallocs c->env[]
     * (store_member/emit_member read mbase->rid + mbase->type only -- a by-value copy is identical). */
    venv mbsnap=*mbp; venv *mbase=&mbsnap;
    c->i+=3;                                        /* consume `name . field` (or `name -> field`) */
    const tok *op=&c->t[c->i];
    if(op->n==1 && op->s[0]=='='){                  /* plain: store rhs, then RE-READ -> the converted value */
      c->i++; uint32_t rhs=p_assign(c);             /* right-associative: p->x = s.y = v */
      if(mf.bit_w) store_member_bf(c,mbase,&mf,rhs);   /* a bitfield: bf.set */
      else { uint32_t st=store_member(c,mbase,&mf,rhs);
        if(mf.is_atomic){ bcir_ctype ft=field_slot(&mf); return atomic_assign_value(c,st,&ft); } }   /* no re-read */
      return emit_member(c,mbase,&mf,0);            /* reload: a bitfield re-reads via bf.get (#bfassignexpr) */
    }
    char ch=op->s[0]; c->i++;                       /* compound `OP=`: read-once, binop, store, value = stored */
    if(mf.is_atomic){                               /* an `_Atomic` member: one atomic read-modify-write */
      uint32_t rhs=p_assign(c); bcir_ctype ft=field_slot(&mf);
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc); (void)oc;
      return emit_rmw(c,suf,&ft,mbase->rid,0,0,1,rhs,mf.byte_off,mf.size,0,mf.is_volatile||mbase->type.is_volatile,
                      BCIR_BND_ASSUMED);
    }
    uint32_t cur=emit_member(c,mbase,&mf,0);        /* read FIRST (matches the oracle's claim order) */
    uint32_t rhs=p_assign(c);
    const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
    uint32_t tmp=binop_result(c,suf,cur,rhs); char o[BCIR_CIR_OP]; fits(c,o,sizeof o,"c.bin.%s",suf);
    bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
    tmp = mf.bit_w ? store_member_bf(c,mbase,&mf,tmp) : store_member(c,mbase,&mf,tmp);   /* a bitfield: bf.set;
                                                        * the value as stored (converted) */
    /* the value of a compound is the STORED (narrowed) value: a sub-int member (`unsigned char c;
     * (s.c += v)`) truncates on store, so RE-READ it (#narrowcompound) -- the plain `=` path above
     * already re-reads; a full-width member needs no re-read (oracle: lv.ct.size < rt.size), so the
     * existing #memassignexpr cases stay byte-identical. A BITFIELD narrows to its BIT width (its size is
     * the full underlying type, so the byte-size test misses it) -- always RE-READ it (oracle:
     * narrows = lv.bit_width or lv.ct.size < rt.size). */
    const bcir_resource *trr=res_of(c->fn,tmp);
    return (mf.bit_w || (int)mf.size < (int)(trr?trr->elem_bytes:4)) ? emit_member(c,mbase,&mf,0) : tmp;
  }
  { uint32_t v; if(lv_assign_value(c,&v)) return v; }   /* a[i]/ *p/o.in.x = rhs (store + reload) as a VALUE */
  return p_cond(c);
}
static uint32_t p_expr(CC *c){           /* depth guard: p_expr re-enters via p_primary's `(...)`/call args */
  if(ENTER_REC(c)){ LEAVE_REC(c); return 0; }
  uint32_t r=p_assign(c); LEAVE_REC(c); return r;
}
/* a control-flow marker claim (no realization; the emitter renders it as a brace). carries an
 * optional condition rid as a read so the verifier resolves it and the emitter can test it. */
static void marker(CC *c,const char *op,uint32_t cond,int has_cond){
  bcir_claim *cl=new_claim(c,op,BCIR_OP_NOP);
  if(cl&&has_cond){cl->n_rd=1;cl->rd[0]=cond;}
}
static void p_stmt(CC *c);
/* --- a static's constant image (CF-STATICTAB; the oracle's `_static_init`) -----------------------------------
 * A static's initializer runs once, before the program does -- never as stores at each call. So it is folded, not
 * lowered into the body: the initializer walk runs in constant mode, each entry lowered speculatively (the rolled-back
 * lowering `typeof` and `sizeof` use) and the claims it made evaluated as C evaluates an integer constant expression --
 * each operation in its own type; each scalar the walk initializes is recorded in the static's image, which is
 * rendered as its declaration's initializer (`kimage`). */
#define KV_NOTCONST "a static initializer is not an integer constant expression"   /* the oracle's `_NOT_CONSTANT`: a
  * variable, a load, a call, an address, a floating value -- or arithmetic C leaves undefined, no constant either */
static unsigned long long kmask(int bits){ return bits>=64 ? ~0ull : (1ull<<bits)-1; }
/* `p` reduced to a `bits`-bit integer type: modulo 2^bits, a signed type's two's complement (the oracle's `_kwrap`). */
static unsigned long long kwrap(unsigned long long p,int bits,int sgn){
  p&=kmask(bits); if(sgn && bits<64 && ((p>>(bits-1))&1u)) p|=~kmask(bits); return p; }
static long long kll(unsigned long long p){ return p>>63 ? -(long long)(~p)-1 : (long long)p; }   /* as signed */
static kval kint(unsigned long long p,int bits,int sgn){ kval r; r.p=kwrap(p,bits,sgn); r.bits=bits; r.sgn=sgn; r.kind=0; return r; }
/* An operand's integer promotion: a _Bool or a type narrower than int is an int; a pointer is no integer. */
static int kpromote(const kval *a,kval *o){
  if(a->kind==2) return 0;
  *o=(a->kind==1 || a->bits<32) ? kint(a->p,32,1) : *a; return 1; }
/* The usual arithmetic conversions of two promoted operands: the wider type; of a signed and an unsigned type, the
 * unsigned one unless the signed one is wider (the oracle's `_kcommon`). */
static void kcommon(const kval *a,const kval *b,int *bits,int *sgn){
  if(a->sgn==b->sgn){ *bits=a->bits>b->bits?a->bits:b->bits; *sgn=a->sgn; return; }
  const kval *u=a->sgn?b:a, *s=a->sgn?a:b;
  *bits=u->bits>=s->bits?u->bits:s->bits; *sgn=u->bits<s->bits; }
/* A signed 64-bit `x op y` into *r, or 0 when it overflows (a narrower one cannot overflow a long long). */
static int ks64(char op,long long x,long long y,long long *r){
  if(op=='+'){ if(y>0 ? x>LLONG_MAX-y : x<LLONG_MIN-y) return 0; *r=x+y; return 1; }
  if(op=='-'){ if(y<0 ? x>LLONG_MAX+y : x<LLONG_MIN+y) return 0; *r=x-y; return 1; }
  if(x && y && (x>0 ? (y>0 ? x>LLONG_MAX/y : y<LLONG_MIN/x) : (y>0 ? x<LLONG_MIN/y : y<LLONG_MAX/x))) return 0;
  *r=x*y; return 1; }
/* `a OP b` as C computes it (the oracle's `_kbin`): a comparison or a logical operator gives an int, a shift the
 * promoted left operand's type, the rest the operands' common type -- an unsigned one wrapping, a signed one that
 * overflows undefined. 0 when it is no constant. */
static int kbin(const char *suf,const kval *ia,const kval *ib,kval *out){
  kval a,b; if(!kpromote(ia,&a) || !kpromote(ib,&b)) return 0;
  if(!strcmp(suf,"land") || !strcmp(suf,"lor")){
    int x=a.p!=0, y=b.p!=0; *out=kint((unsigned long long)(suf[1]=='a' ? (x&&y) : (x||y)),32,1); return 1; }
  if(!strcmp(suf,"shl") || !strcmp(suf,"shr")){
    long long n=b.sgn ? kll(b.p) : (b.p>(unsigned long long)LLONG_MAX ? -1 : (long long)b.p);
    if(n<0 || n>=a.bits || n>=64) return 0;                   /* a count outside the promoted width: undefined */
    if(suf[2]=='r'){ *out=kint(a.sgn && (a.p>>63) ? ~(~a.p>>n) : a.p>>n,a.bits,a.sgn); return 1; }   /* arithmetic */
    if(a.sgn && ((a.p>>63) || a.p>(kmask(a.bits-1)>>n))) return 0;   /* a negative or overflowing signed shift */
    *out=kint(a.p<<n,a.bits,a.sgn); return 1; }
  int bits,sgn; kcommon(&a,&b,&bits,&sgn);
  unsigned long long x=kwrap(a.p,bits,sgn), y=kwrap(b.p,bits,sgn), r=0; int cmp=-1;
  if(!strcmp(suf,"eq")) cmp=x==y; else if(!strcmp(suf,"ne")) cmp=x!=y;
  else if(!strcmp(suf,"lt")) cmp=sgn?kll(x)<kll(y):x<y; else if(!strcmp(suf,"gt")) cmp=sgn?kll(x)>kll(y):x>y;
  else if(!strcmp(suf,"le")) cmp=sgn?kll(x)<=kll(y):x<=y; else if(!strcmp(suf,"ge")) cmp=sgn?kll(x)>=kll(y):x>=y;
  if(cmp>=0){ *out=kint((unsigned long long)cmp,32,1); return 1; }
  if(!strcmp(suf,"and")) r=x&y; else if(!strcmp(suf,"or")) r=x|y; else if(!strcmp(suf,"xor")) r=x^y;
  else if(!sgn){                                              /* unsigned: wraps modulo 2^bits */
    if(!strcmp(suf,"add")) r=x+y; else if(!strcmp(suf,"sub")) r=x-y; else if(!strcmp(suf,"mul")) r=x*y;
    else if(!strcmp(suf,"div") || !strcmp(suf,"mod")){ if(!y) return 0; r=suf[0]=='d'?x/y:x%y; }
    else return 0; }
  else { long long sx=kll(x), sy=kll(y), sr;                  /* signed: an overflow is undefined */
    if(!strcmp(suf,"div") || !strcmp(suf,"mod")){             /* truncating; INT_MIN / -1 (and its remainder) */
      if(!sy || (sx==LLONG_MIN && sy==-1)) return 0;
      long long q=sx/sy; if(bits<64 && q>=(1LL<<(bits-1))) return 0;
      sr=suf[0]=='d'?q:sx-q*sy; }
    else { char op=!strcmp(suf,"add")?'+':!strcmp(suf,"sub")?'-':!strcmp(suf,"mul")?'*':0;
      if(!op || (bits>=64 && !ks64(op,sx,sy,&sr))) return 0;
      if(bits<64) sr=op=='+'?sx+sy:op=='-'?sx-sy:sx*sy; }
    if(bits<64 && (sr<-(1LL<<(bits-1)) || sr>=(1LL<<(bits-1)))) return 0;
    r=(unsigned long long)sr; }
  *out=kint(r,bits,sgn); return 1;
}
/* `OP a` (the oracle's `_kun`): `!` gives an int; `-` and `~` the promoted operand's type (`-INT_MIN` overflows). */
static int kun(const char *suf,const kval *ia,kval *out){
  if(!strcmp(suf,"lnot")){ *out=kint(ia->p==0,32,1); return 1; }
  kval a; if(!kpromote(ia,&a)) return 0;
  if(!strcmp(suf,"bnot")){ *out=kint(~a.p,a.bits,a.sgn); return 1; }
  if(strcmp(suf,"neg") || (a.sgn && a.p==kwrap(1ull<<(a.bits-1),a.bits,1))) return 0;
  *out=kint(0ull-a.p,a.bits,a.sgn); return 1; }
/* `c ? a : b` over arithmetic arms (the oracle's `_ksel`): the chosen arm, in the arms' common type. */
static int ksel(const kval *c0,const kval *ia,const kval *ib,kval *out){
  kval a,b; int bits,sgn; if(!kpromote(ia,&a) || !kpromote(ib,&b)) return 0;
  kcommon(&a,&b,&bits,&sgn); *out=kint(c0->p?a.p:b.p,bits,sgn); return 1; }
/* The type a constant claim's temp declares -- a literal's, sizeof's size_t, a cast's target -- as a zero of it (the
 * oracle's `_ktype`); a floating or _BitInt one is no integer constant here. */
static int ktype(const bcir_resource *r,kval *t){
  memset(t,0,sizeof *t);
  if(!r) return 0;
  if(r->kind==BCIR_RK_POINTER || r->is_funcptr){ t->kind=2; return 1; }
  if(r->kind!=BCIR_RK_SCALAR || r->is_float || r->bit_width>0 || r->elem_bytes<1 || r->elem_bytes>8) return 0;
  if(r->is_bool){ t->kind=1; t->bits=1; return 1; }
  t->bits=(int)r->elem_bytes*8; t->sgn=r->is_signed; return 1; }
/* `a` converted to the type `t` (the oracle's `_kconvert`): an integer wraps to its width, a _Bool is whether `a` is
 * nonzero, a pointer takes only the null pointer constant. */
static int kconvert(const kval *a,const kval *t,kval *out){
  *out=*t;
  if(t->kind==1){ out->p=a->p!=0; return 1; }
  if(t->kind==2) return a->p==0;
  *out=kint(a->p,t->bits,t->sgn); return 1; }
/* One claim of an entry, folded (the oracle's `_kfold`): 0 when it is no integer constant expression. */
static int kclaim(CC *c,const bcir_claim *cl,const kval *a,kval *out){
  const char *op=cl->op; kval t;
  if(!strcmp(op,"c.const") && cl->n_rd==0 && cl->n_imm>=1){
    kval x=kint((unsigned long long)cl->imm[0],64,1); return ktype(res_of(c->fn,cl->wr[0]),&t) && kconvert(&x,&t,out); }
  if(!strncmp(op,"c.cast:",7) && cl->n_rd==1) return ktype(res_of(c->fn,cl->wr[0]),&t) && kconvert(&a[0],&t,out);
  if(!strncmp(op,"c.bin.",6) && cl->n_rd==2) return kbin(op+6,&a[0],&a[1],out);
  if(!strncmp(op,"c.un.",5) && cl->n_rd==1) return kun(op+5,&a[0],out);
  if(!strcmp(op,"c.select") && cl->n_rd==3) return ksel(&a[0],&a[1],&a[2],out);
  return 0; }
/* One entry of a static's initializer, folded as C evaluates an integer constant expression (the oracle's
 * `_const_value`): the claims its speculative lowering made from claim `from` on -- a constant (a literal, a sizeof,
 * an enumerator) in its declared type, arithmetic, a cast, a select -- in order, each over values the entry itself
 * produced. Any other claim, or a value produced elsewhere (a variable, a parameter, a string, a function), fails. */
static kval kfold(CC *c,size_t from,uint32_t v){
  kval z=kint(0,32,1); size_t n=c->fn->n_claims-from;
  kval *kv=(kval*)bcir_host_arena_allocate(&c->scratch,(n?n:1)*sizeof *kv,sizeof(unsigned long long));
  if(!kv){ cc_raise_oom(c); return z; }
  for(size_t i=0;i<n;i++){ const bcir_claim *cl=&c->fn->claims[from+i]; kval a[3]; int ok=cl->n_wr==1 && cl->n_rd<=3;
    memset(a,0,sizeof a);
    for(int k=0;k<cl->n_rd && ok;k++){ ok=0;
      for(size_t j=i;j-- > 0;) if(c->fn->claims[from+j].wr[0]==cl->rd[k]){ a[k]=kv[j]; ok=1; break; } }
    if(!ok || !kclaim(c,cl,a,&kv[i])){ fail(c,KV_NOTCONST); return z; } }
  for(size_t i=n;i-- > 0;) if(c->fn->claims[from+i].wr[0]==v) return kv[i];
  fail(c,KV_NOTCONST); return z;
}
/* --- C11 6.7.9 initialization: the current-object walk (CF-BRACE; the oracle's `_init_list`) -----------
 * A braced initializer lowers to the object's zero baseline (emitted `= {}`) plus one store per initialized
 * scalar, in list order. Each brace list has a current object whose subobjects -- array elements, struct members in
 * order, a union's first member -- the positional entries fill in turn, descending into a sub-aggregate whose
 * initializer has no brace of its own (brace elision, 6.7.9p20: it takes only as many entries as it has
 * scalars); a designator re-points the walk, and the entries after it continue from the subobject after the
 * designated one (p17). A store is a typed `base[i]` when it writes a whole element of the declared array its
 * own list initializes, else a store at its byte offset -- the claim shapes the fully braced forms had. */
enum { IK_SCAL, IK_AGG, IK_ARR };
typedef struct {            /* one object or subobject the walk initializes (the oracle's CType + offset) */
  int k;                    /* IK_SCAL (a scalar or pointer), IK_AGG (a struct/union), IK_ARR (an array) */
  int sidx;                 /* IK_AGG: its sdef; IK_ARR: the element struct (-1: a scalar element) */
  int nd, dims[3];          /* IK_ARR: its dims, outer first */
  int es;                   /* IK_ARR: its element's (scalar or struct) size */
  int off, size;            /* its byte offset in the object, and its size */
  int bit_w, bit_off, abytes;   /* IK_SCAL: a bit-field member's width, bit offset and storage-unit span */
  int is_bool, is_atomic;   /* IK_SCAL (and IK_ARR's element): a `_Bool` / `_Atomic` slot */
  bcir_ctype slot;          /* IK_SCAL (and IK_ARR's element): the declared type a stored value converts to */
} iunit;
typedef struct { iunit u; int idx, count, flat, top; } ifrm;   /* idx: the next subobject; count < 0: inferred */
typedef struct { uint32_t rid; int top_array, top_n, inferred, rng0, un0;
                 int konst; kval kv;
                 int skip; const tok *gname; } iwalk;   /* konst: a static's image -- its entries fold (into kv, each before it
                                                 * is placed), its stores record their values (CF-STATICTAB); skip (with
                                                 * konst): a file-scope initializer's shape -- its entries are skipped,
                                                 * its stores record their ranges only (CF-GBRACE); gname: the global */
#define IW_MAXFRAMES 64     /* the subobjects one list may nest into (the oracle's `_INIT_MAX_FRAMES`) */

/* A member's subobject: an array, a struct/union, or a scalar (a bit-field keeps its unit). */
static void iunit_of_field(CC *c, iunit *u, const field *F, int base_off){
  memset(u,0,sizeof *u); u->off=base_off+F->byte_off; u->sidx=-1;
  if(F->arr_count>0){ u->k=IK_ARR; u->sidx=F->elem_sidx; u->nd=F->nadims>0?F->nadims:1;
    if(F->nadims>0){ for(int d=0;d<3;d++) u->dims[d]=F->adims[d]; } else u->dims[0]=F->arr_count;
    u->es=F->size; u->size=F->arr_count*F->size; u->is_bool=F->is_bool; u->is_atomic=F->is_atomic;
    u->slot=field_slot(F); return; }
  if(F->sidx>=0){ u->k=IK_AGG; u->sidx=F->sidx; u->size=c->s[F->sidx].size; return; }
  u->k=IK_SCAL; u->size=F->size; u->bit_w=F->bit_w; u->bit_off=F->bit_off; u->abytes=F->access_bytes;
  u->is_bool=F->is_bool; u->is_atomic=F->is_atomic; u->slot=field_slot(F);
}
/* Element `i` of the array `a`: a row of a multi-dimensional array, else a struct or scalar element. */
static void iunit_elem(iunit *e, const iunit *a, int i){
  int stride=a->es; for(int d=1; d<a->nd; d++) stride*=a->dims[d];
  *e=*a; e->off=a->off+i*stride;
  if(a->nd>1){ e->nd=a->nd-1; for(int d=0;d<3;d++) e->dims[d]=d+1<a->nd?a->dims[d+1]:0; e->size=stride; return; }
  e->nd=0; memset(e->dims,0,sizeof e->dims); e->size=a->es;
  if(a->sidx>=0){ e->k=IK_AGG; return; }
  e->k=IK_SCAL; e->bit_w=0; e->bit_off=0; e->abytes=0;
}
static int iunit_count(CC *c, const iunit *u){ return u->k==IK_ARR ? u->dims[0] : c->s[u->sidx].nf; }
/* The subobject at the frame's position, and the position after it (the oracle's `_init_unit`). An anonymous
 * struct/union member is one subobject over its promoted leaves; a union holds one. */
static int iunit_at(CC *c, const ifrm *fr, iunit *u){
  const iunit *o=&fr->u; int i=fr->idx;
  if(o->k==IK_ARR){ iunit_elem(u,o,i); return i+1; }
  const sdef *S=&c->s[o->sidx];
  for(int g=0; g<S->nanon; g++) if(S->anon[g].first==i){
    memset(u,0,sizeof *u); u->k=IK_AGG; u->sidx=S->anon[g].sidx; u->off=o->off+S->anon[g].off;
    u->size=c->s[u->sidx].size; return S->is_union ? S->nf : i+S->anon[g].n; }
  iunit_of_field(c,u,&S->f[i],o->off);
  return S->is_union ? S->nf : i+1;
}
/* Consume the subobject at the frame's position (the oracle's `_init_take`). A union takes one member per
 * object: giving an initialized union another member is refused, its other bytes would need re-zeroing. */
static int init_take(CC *c, iwalk *W, ifrm *fr, iunit *u){
  int nxt=iunit_at(c,fr,u);
  if(fr->u.k==IK_AGG && c->s[fr->u.sidx].is_union){
    int seen=0;
    for(int k=W->un0; k<c->iw_nun; k++){ const iunion *r=&c->iw_un[k];
      if(r->off==fr->u.off && r->sidx==fr->u.sidx){ seen=1;
        if(r->member!=fr->idx){ fail(c,"an initializer overrides a prior initialization of a subobject"); return 0; } } }
    if(!seen){ if(!CC_ENSURE(c,c->iw_un,c->iw_nun,c->iw_capun)) return 0;
      c->iw_un[c->iw_nun].off=fr->u.off; c->iw_un[c->iw_nun].sidx=fr->u.sidx; c->iw_un[c->iw_nun].member=fr->idx;
      c->iw_nun++; }
  }
  fr->idx=nxt;
  if(fr->top && nxt>W->top_n) W->top_n=nxt;
  return 1;
}
/* Enter the sub-aggregate `u` of `frames[*nf-1]` (brace elision or a designator). */
static ifrm *init_push(CC *c, ifrm *frames, int *nf, const iunit *u){
  if(*nf>=IW_MAXFRAMES){ fail(c,"an initializer nested deeper than 64 subobjects"); return NULL; }
  ifrm *p=&frames[*nf-1], *ch=&frames[*nf];
  ch->u=*u; ch->idx=0; ch->count=iunit_count(c,u); ch->flat=p->flat && u->k==IK_ARR; ch->top=0;
  (*nf)++; return ch;
}
/* The frame whose next subobject the next positional entry fills: filled sub-aggregates entered by elision are
 * left for their parent; past the list's own object is an excess initializer, refused, never dropped. */
static ifrm *init_next(CC *c, ifrm *frames, int *nf){
  for(;;){ ifrm *fr=&frames[*nf-1];
    if(fr->count<0 || fr->idx<fr->count) return fr;
    if(*nf==1){ fail(c,"excess elements in an initializer"); return NULL; }
    (*nf)--; }
}
/* Point the walk at the subobject a designator names (the oracle's `_init_designate`): back to the list's own
 * object, then each `.m` / `[i]` step selects a member or an element, descending into it before the next step;
 * a member of an anonymous struct/union is reached through that member. The cursor ends past the `=`. */
static ifrm *init_designate(CC *c, iwalk *W, ifrm *frames, int *nf){
  *nf=1;
  for(;;){
    ifrm *fr=&frames[*nf-1];
    if(is(c,".")){
      c->i++; tok fld=adv(c);
      if(fr->u.k!=IK_AGG){ fail(c,"a member designator into a non-aggregate"); return NULL; }
      const sdef *S=&c->s[fr->u.sidx]; int j=-1;
      for(int k=0;k<S->nf;k++) if((int)strlen(S->f[k].name)==fld.n && !strncmp(S->f[k].name,fld.s,(size_t)fld.n)){ j=k; break; }
      if(fld.k!=T_ID || j<0){ fail(c,"no member of that name to designate"); return NULL; }
      for(;;){ int g=-1;
        for(int q=0;q<S->nanon;q++) if(S->anon[q].first<=j && j<S->anon[q].first+S->anon[q].n){ g=q; break; }
        if(g<0){ fr->idx=j; break; }
        int first=S->anon[g].first; iunit u;
        fr->idx=first;                                  /* through the anonymous member that holds it */
        if(!init_take(c,W,fr,&u) || !(fr=init_push(c,frames,nf,&u))) return NULL;
        S=&c->s[fr->u.sidx]; j-=first; }
    } else if(is(c,"[")){
      c->i++; long long ix=ce_value(c); if(!eat(c,"]")) return NULL;
      if(fr->u.k!=IK_ARR){ fail(c,"an array designator into a non-array"); return NULL; }
      if(ix<0 || ix>INT_MAX || (fr->count>=0 && ix>=fr->count)){ fail(c,"an array designator outside the array"); return NULL; }
      fr->idx=(int)ix;
    } else break;
    if(is(c,".")||is(c,"[")){                           /* the next step names a subobject of this one */
      iunit u; if(!init_take(c,W,fr,&u)) return NULL;
      if(u.k==IK_SCAL){ fail(c,"a designator into a scalar"); return NULL; }
      if(!init_push(c,frames,nf,&u)) return NULL; }
  }
  if(c->failed || !eat(c,"=")) return NULL;
  return &frames[*nf-1];
}
/* A designator of exactly one `[i]` (the cursor is at its `[`): its entry's store may be a whole element. */
static int desig_single_index(CC *c){
  int j=c->i, d=0;
  if(!tok_is(tat(c,j),"[")) return 0;
  for(; tat(c,j)->k!=T_END; j++){
    if(tok_is(tat(c,j),"[")) d++;
    else if(tok_is(tat(c,j),"]") && --d==0) break; }
  return tok_is(tat(c,j+1),"=");
}
/* A subobject initialized as a whole -- by a brace list or a string literal -- is zero wherever its initializer
 * is silent, which the baseline gives only while nothing has stored into it yet: refuse an earlier store in
 * [lo, hi) bits (the oracle's `_init_overrides`). */
static int init_overrides(CC *c, const iwalk *W, long long lo, long long hi){
  for(int k=W->rng0; k<c->iw_nrng; k++)
    if(c->iw_rng[k].lo<hi && lo<c->iw_rng[k].hi){ fail(c,"an initializer overrides a prior initialization of a subobject"); return 1; }
  return 0;
}
/* The value the static's scalar subobject `u` holds from the constant `a` -- C's conversion as if by assignment --
 * recorded on its store's range `r` (the oracle's `_kleaf`): a pointer holds only the null pointer, a floating one the
 * integer itself, which the rendered initializer converts exactly as the source's does. */
static int kleaf(CC *c,const kval *a,const iunit *u,irange *r){
  const bcir_ctype *s=&u->slot; unsigned long long p=a->p; int neg=a->sgn && (p>>63);
  if(u->k!=IK_SCAL || s->kind==1){ fail(c,KV_NOTCONST); return 0; }
  if(s->kind==2 || s->kind==3){ if(p){ fail(c,KV_NOTCONST); return 0; } neg=0; }
  else if(s->is_float){}                               /* the integer itself, signed as its own type */
  else if(s->is_bool){ p=p!=0; neg=0; }
  else { int bits=u->bit_w?u->bit_w:s->bit_width>0?s->bit_width:s->size*8;
    p=kwrap(p,bits,s->signd); neg=s->signd && (p>>63); }
  r->v=p; r->neg=neg; return 1;
}
/* Store `v` into the scalar (or whole struct/union) subobject `u` (the oracle's `_init_store`): a typed `base[i]`
 * at its flat index when `indexed`, else a store at its byte offset (a bit-field's through its storage unit). In a
 * static's image the folded W->kv is recorded as the value the subobject holds instead, and no claim is made. */
static void init_store(CC *c, iwalk *W, const iunit *u, uint32_t v, int indexed){
  long long lo=(long long)u->off*8+u->bit_off, hi=lo+(u->bit_w?u->bit_w:(long long)u->size*8);
  if(!CC_ENSURE(c,c->iw_rng,c->iw_nrng,c->iw_caprng)) return;
  c->iw_rng[c->iw_nrng].lo=lo; c->iw_rng[c->iw_nrng].hi=hi; c->iw_rng[c->iw_nrng].v=0; c->iw_rng[c->iw_nrng].neg=0;
  if(W->skip){ c->iw_nrng++; return; }              /* a file-scope initializer's shape: the range, no value */
  if(W->konst){ if(kleaf(c,&W->kv,u,&c->iw_rng[c->iw_nrng])) c->iw_nrng++; return; }   /* a static's image: no claim */
  c->iw_nrng++;
  uint32_t rid=W->rid;
  if(u->k==IK_SCAL && !scalar_value_ok(c,v)) return;   /* a scalar subobject takes no struct (CF-STRUCTARITH) */
  if(indexed){                                         /* a whole element of the declared array */
    if(u->k==IK_SCAL){ bcir_ctype et=u->slot; v=null_pointer(c,v,&et); }   /* `T *a[] = {&x, 0}` */
    uint32_t ic=temp(c,4); bcir_claim *kc=new_claim(c,"c.const",BCIR_OP_LOAD);
    if(kc){kc->n_wr=1;kc->wr[0]=ic;kc->n_imm=1;kc->imm[0]=u->size>0?u->off/u->size:0;}
    bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
    if(cl){cl->n_rd=3;cl->rd[0]=rid;cl->rd[1]=ic;cl->rd[2]=v;cl->bounds=access_bnd(c,rid);   /* §5.12 */
      const bcir_resource *ar=res_of(c->fn,rid); mark_atomic(cl,ar && ar->is_atomic);}   /* `_Atomic` elements */
    return; }
  uint32_t val=v;
  if(u->k==IK_SCAL){ bcir_ctype ls=u->slot; v=store_conv(c,v,&ls); val=v; }   /* C's conversion to the slot */
  if(u->bit_w){                                        /* a bitfield member: read unit, set bits, store */
    int absz=u->abytes<=4?4:8;
    uint32_t unit=temp(c,absz);
    bcir_claim *ld=new_claim(c,"c.load",BCIR_OP_LOAD);
    if(ld){ld->n_rd=1;ld->rd[0]=rid;ld->n_wr=1;ld->wr[0]=unit;ld->n_imm=2;ld->imm[0]=u->off;ld->imm[1]=u->abytes;ld->bounds=BCIR_BND_ASSUMED;}
    uint32_t nu=temp(c,absz);
    bcir_claim *bs=new_claim(c,"c.bf.set",BCIR_OP_ADD);
    if(bs){bs->n_rd=2;bs->rd[0]=unit;bs->rd[1]=v;bs->n_wr=1;bs->wr[0]=nu;bs->n_imm=2;bs->imm[0]=u->bit_off;bs->imm[1]=u->bit_w;}
    val=nu; }
  bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
  if(cl){cl->n_rd=2;cl->rd[0]=rid;cl->rd[1]=val;cl->n_imm=2;cl->imm[0]=u->off;cl->imm[1]=u->bit_w?u->abytes:u->size;
    cl->bounds=BCIR_BND_ASSUMED;
    if(u->bit_w){cl->imm[2]=2;cl->n_imm=3;}           /* a bitfield UNIT store: `_v` takes the unit's full type */
    else if(u->is_bool){cl->imm[2]=1;cl->n_imm=3;}     /* a _Bool slot normalizes the value */
    mark_atomic(cl,u->is_atomic && !u->bit_w);}        /* an `_Atomic` slot: an atomic store */
}
/* The code-unit width of a string-literal token's prefix: plain/u8 1, u 2, U 4, L the target's wchar_t. */
static int str_tok_unit(CC *c, const tok *t){ return str_elem_size(c,t->s,t->n); }
static int str_tok_prefix_len(const tok *t){ int k=0; while(k<t->n && t->s[k]!='"') k++; return k; }
/* The array `u` a string literal whose units are `unit` bytes wide initializes (6.7.9p14-15): one-dimensional,
 * of an integer element as wide as the unit (the oracle's `_init_char_array`). */
static int init_char_array(const iunit *u, int unit){
  return u->k==IK_ARR && u->nd==1 && u->sidx<0 && u->slot.kind==0 && !u->slot.is_float && !u->slot.is_bool &&
         u->slot.bit_width==0 && u->es==unit;
}
/* A string-literal entry at the cursor that is its entry's whole initializer (the literal, then `,` or `}`). */
static int str_entry(CC *c){
  int j=c->i; if(tat(c,j)->k!=T_STR) return 0;
  while(tat(c,j)->k==T_STR) j++;
  return tok_is(tat(c,j),",") || tok_is(tat(c,j),"}");
}
/* Decode the (concatenated) string literal at the cursor into its code units, excluding the NUL (the oracle's
 * `str_units`): each unit is a source character or one escape; pieces with different prefixes, a non-ASCII
 * character, a universal character name, or an escape too wide for the unit are refused. Consumes the tokens. */
static int str_units(CC *c, uint32_t **out, int *n_out, int *unit_out){
  int j=c->i, total=0; const tok *first=tat(c,j);
  int pl=str_tok_prefix_len(first), unit=str_tok_unit(c,first);
  for(; tat(c,j)->k==T_STR; j++){ const tok *t=tat(c,j);
    if(str_tok_prefix_len(t)!=pl || strncmp(t->s,first->s,(size_t)pl)){ fail(c,"a string initializer concatenating literals of different prefixes"); return 0; }
    total+=t->n; }
  uint32_t *u=(uint32_t*)bcir_host_arena_allocate(&c->scratch,(size_t)(total>0?total:1)*sizeof *u,sizeof *u);
  if(!u){ cc_raise_oom(c); return 0; }
  int n=0;
  unsigned long long lim = unit>=4 ? 0x100000000ull : (1ull<<(8*unit));
  while(isk(c,T_STR)){ tok t=adv(c); int i=pl+1, end=t.n-1;   /* inside the quotes */
    while(i<end){ unsigned char ch=(unsigned char)t.s[i]; unsigned long long val;
      if(ch>=128){ fail(c,"a non-ASCII character in a string initializer is not supported"); return 0; }
      if(ch!='\\' || i+1>=end){ u[n++]=ch; i++; continue; }
      char e=t.s[i+1];
      if(e=='x'){ int nd=0; val=0; i+=2;
        while(i<end && ((t.s[i]>='0'&&t.s[i]<='9')||((t.s[i]|0x20)>='a'&&(t.s[i]|0x20)<='f'))){
          int h=t.s[i]<='9'?t.s[i]-'0':(t.s[i]|0x20)-'a'+10; if(val<lim) val=val*16+(unsigned)h; i++; nd++; }
        if(!nd){ fail(c,"a \\x escape with no hex digit"); return 0; } }
      else if(e>='0'&&e<='7'){ int k=0; val=0; i++; while(k<3 && i<end && t.s[i]>='0'&&t.s[i]<='7'){ val=val*8+(unsigned)(t.s[i]-'0'); i++; k++; } }
      else if(e=='u'||e=='U'){ fail(c,"a universal character name in a string initializer is not supported"); return 0; }
      else { switch(e){case 'n':val=10;break;case 't':val=9;break;case 'r':val=13;break;case '\\':val=92;break;
               case '\'':val=39;break;case '"':val=34;break;case 'a':val=7;break;case 'b':val=8;break;
               case 'f':val=12;break;case 'v':val=11;break;case '?':val=63;break;default:val=(unsigned char)e;}
             i+=2; }
      if(val>=lim){ fail(c,"an escape sequence out of range for its character type"); return 0; }
      u[n++]=(uint32_t)val; } }
  *out=u; *n_out=n; *unit_out=unit; return 1;
}
/* Initialize the character array `u` (of `count` elements; < 0: the declared array its literal sizes) from the
 * string literal at the cursor: one store per code unit, the NUL and the rest left to the zero baseline (the
 * oracle's `_init_string`); a literal longer than the array is refused. */
static void init_string(CC *c, iwalk *W, const iunit *u, int indexed, int count){
  uint32_t *units; int n, unit;
  if(!str_units(c,&units,&n,&unit)) return;
  if(count<0){ count=n+1; if(count>W->top_n) W->top_n=count; }
  if(n>count){ fail(c,"an initializer-string for a character array is too long"); return; }
  if(init_overrides(c,W,(long long)u->off*8,((long long)u->off+(long long)u->es*count)*8)) return;
  for(int i=0;i<n && !c->failed;i++){
    uint32_t cv=0;
    if(W->konst) W->kv=kint(units[i],32,1);            /* a static's image: the code unit, an int constant */
    else { cv=temp(c,4); bcir_claim *kc=new_claim(c,"c.const",BCIR_OP_LOAD);
      if(kc){kc->n_wr=1;kc->wr[0]=cv;kc->n_imm=1;kc->imm[0]=(int64_t)units[i];} }
    iunit e; iunit_elem(&e,u,i); init_store(c,W,&e,cv,indexed); }
}
static void init_list(CC *c, iwalk *W, const iunit *obj, int top, int flat);
/* An initializer entry's value (the cursor at it), evaluated once, before it is placed: lowered -- or, in a static's
 * image, folded into W->kv; in a file-scope initializer's shape, skipped (the oracle's `_init_value`). */
static void skip_init_expr(CC *c, const tok *nm);   /* fwd: a file-scope initializer's entry, skipped */
static uint32_t init_value(CC *c, iwalk *W){
  if(W->skip){ skip_init_expr(c,W->gname); return 0; }   /* a file-scope initializer's shape: no value (CF-GBRACE) */
  if(!W->konst) return p_expr(c);
  size_t from=c->fn->n_claims; uint32_t v=p_expr(c);
  if(!c->failed) W->kv=kfold(c,from,v);
  return 0;
}
/* A nested brace list for the subobject `u`: it initializes the whole subobject, so no earlier store may lie in
 * it; a scalar takes a braced single expression (6.7.9p11) -- the oracle's `_init_sublist`. */
static void init_sublist(CC *c, iwalk *W, const ifrm *fr, const iunit *u){
  long long lo=(long long)u->off*8+u->bit_off;
  if(init_overrides(c,W,lo,lo+(u->bit_w?u->bit_w:(long long)u->size*8))) return;
  if(u->k==IK_SCAL){
    eat(c,"{");
    if(is(c,"}")){ c->i++; return; }                  /* `{}` is zero, which the baseline already holds */
    if(is(c,".")||is(c,"[")||is(c,"{")){ fail(c,"a braced scalar initializer holds one expression"); return; }
    uint32_t v=init_value(c,W); if(is(c,",")) c->i++;
    if(c->failed) return;
    if(!is(c,"}")){ fail(c,"a braced scalar initializer holds one expression"); return; }
    c->i++; init_store(c,W,u,v,0); return; }
  init_list(c,W,u,0,fr->flat && u->k==IK_ARR);
}
static void init_list_inner(CC *c, iwalk *W, const iunit *obj, int top, int flat){
  if(!eat(c,"{")) return;
  ifrm frames[IW_MAXFRAMES]; int nf=1;
  frames[0].u=*obj; frames[0].idx=0; frames[0].count=(top && W->inferred)?-1:iunit_count(c,obj);
  frames[0].flat=flat; frames[0].top=top;
  if(str_entry(c)){                                   /* `{"abc"}` for a character array: the whole array */
    int j=c->i; while(tat(c,j)->k==T_STR) j++;
    if(tok_is(tat(c,j),",")) j++;
    if(tok_is(tat(c,j),"}") && init_char_array(obj,str_tok_unit(c,pk(c)))){
      init_string(c,W,obj,top && flat,frames[0].count); if(is(c,",")) c->i++; eat(c,"}"); return; } }
  while(!is(c,"}") && !isk(c,T_END) && !c->failed){
    int keyed=is(c,".")||is(c,"[");
    int indexed_ok=top && W->top_array && (!keyed || desig_single_index(c));
    ifrm *fr=keyed ? init_designate(c,W,frames,&nf) : init_next(c,frames,&nf);
    if(!fr) return;
    iunit u;
    if(is(c,"{")){ if(!init_take(c,W,fr,&u)) return; init_sublist(c,W,fr,&u); }
    else if(str_entry(c)){                              /* a character array it initializes, found by elision */
      int unit=str_tok_unit(c,pk(c));
      for(;;){ iunit_at(c,fr,&u);
        if(init_char_array(&u,unit)){ if(!init_take(c,W,fr,&u)) return; init_string(c,W,&u,indexed_ok && fr->flat,u.dims[0]); break; }
        if(u.k==IK_SCAL){ ifrm *hold=fr; if(!init_take(c,W,fr,&u)) return;
          uint32_t v=init_value(c,W); init_store(c,W,&u,v,indexed_ok && hold->flat); break; }
        if(!init_take(c,W,fr,&u) || !init_push(c,frames,&nf,&u) || !(fr=init_next(c,frames,&nf))) return; }
    } else {
      uint32_t v=init_value(c,W);                      /* evaluated once, before it is placed */
      if(c->failed) return;
      for(;;){ iunit_at(c,fr,&u);
        if(u.k==IK_AGG && !W->konst && value_is_struct(c,v,u.sidx)){   /* a struct/union value: the subobject whole
                                                                        * (a folded constant never is one) */
          int flat_here=fr->flat; if(!init_take(c,W,fr,&u)) return; init_store(c,W,&u,v,indexed_ok && flat_here); break; }
        if(u.k==IK_SCAL){ int flat_here=fr->flat; if(!init_take(c,W,fr,&u)) return; init_store(c,W,&u,v,indexed_ok && flat_here); break; }
        if(!init_take(c,W,fr,&u) || !init_push(c,frames,&nf,&u) || !(fr=init_next(c,frames,&nf))) return; }   /* brace elision */
    }
    if(c->failed) return;
    if(is(c,",")) c->i++; else break;
  }
  eat(c,"}");
}
/* Depth-guarded: a nested brace list re-enters init_list (`{{{...}}}`). */
static void init_list(CC *c, iwalk *W, const iunit *obj, int top, int flat){
  if(ENTER_REC(c)){ LEAVE_REC(c); return; }
  init_list_inner(c,W,obj,top,flat); LEAVE_REC(c);
}
/* Lower the initializer at the cursor -- a brace list, or a string literal for a character array -- of the local
 * object `rid` that `obj` describes: the zero baseline plus the walk's stores (the oracle's `_agg_init`).
 * Returns the number of top-level elements reached, which sizes an `inferred` array. */
static int init_object(CC *c, uint32_t rid, const iunit *obj, int inferred){
  iwalk W; memset(&W,0,sizeof W);
  W.rid=rid; W.top_array=obj->k==IK_ARR; W.top_n=0; W.inferred=inferred; W.rng0=c->iw_nrng; W.un0=c->iw_nun;
  for(size_t i=0;i<c->fn->n_res;i++) if(c->fn->res[i].rid==rid){ c->fn->res[i].zinit=1; break; }   /* = {} */
  if(isk(c,T_STR)){                                    /* `char s[] = "ab"` is `char s[] = {"ab"}` (6.7.9p14) */
    if(!init_char_array(obj,str_tok_unit(c,pk(c)))) fail(c,"an array is initialized by a brace list or a string literal");
    else init_string(c,&W,obj,1,inferred?-1:obj->dims[0]); }
  else init_list(c,&W,obj,1,obj->k==IK_ARR);
  c->iw_nrng=W.rng0; c->iw_nun=W.un0;                  /* a nested initializer's records end with it */
  return W.top_n;
}
/* The walk's description of a local struct/union of sdef `si`, or of a local array of `ty` elements (`es` bytes
 * each; a struct element's sdef `si`) with dims `dims[0..nd)`. */
static iunit iunit_struct(CC *c, int si){
  iunit u; memset(&u,0,sizeof u); u.k=IK_AGG; u.sidx=si; u.size=si>=0?c->s[si].size:0; return u;
}
static iunit iunit_array(const bcir_ctype *ty, int si, int es, const int *dims, int nd){
  iunit u; memset(&u,0,sizeof u); u.k=IK_ARR; u.sidx=ty->kind==1?si:-1; u.nd=nd<1?1:nd;
  for(int d=0;d<3 && d<u.nd;d++) u.dims[d]=dims[d];
  u.es=es; u.slot=*ty; u.slot.nadims=0; u.is_bool=ty->is_bool; u.is_atomic=ty->is_atomic;
  u.size=es; for(int d=0; d<u.nd; d++) u.size*=u.dims[d];
  return u;
}
/* ... of a local scalar, pointer or function pointer of type `ty`: one subobject (a static's). */
static iunit iunit_scalar(CC *c, const bcir_ctype *ty){
  iunit u; memset(&u,0,sizeof u); u.k=IK_SCAL; u.sidx=-1; u.slot=*ty; u.slot.nadims=0;
  u.size=ty->kind==2?cc_abi(c)->pointer_size:ty->size; u.is_bool=ty->is_bool; u.is_atomic=ty->is_atomic;
  return u;
}
/* A static's image rendered as its declaration's initializer (the oracle's `_render_image`) -- the brace list both
 * rails render: each scalar the walk stored holds its last value, and a zero one is the static's own zero. An array
 * lists the elements that hold a value, designating one after a gap (`[5] = v`); a struct its members in order, up to
 * the last that holds one; a union the member the walk gave it (`.m = v` for any but the first); a multi-dimensional
 * static its elements flat, as it is declared. */
#define KV_ANONUNION "a static union initialized through an anonymous member other than its first"   /* the oracle's
  * `_ANON_UNION`: C names that member only through its own leaves, which the brace list does not spell */
typedef struct { long long lo, w; unsigned long long v; int neg; int at; } krec;   /* a stored scalar: its bit offset
  * and width, its value, and the order it was stored in (the last store wins) */
typedef struct { krec *r; int n; char *s; size_t len, cap; const iwalk *W; } kimg;   /* the image + the text so far */
static int krec_cmp(const void *x,const void *y){
  const krec *a=(const krec *)x, *b=(const krec *)y;
  return a->lo!=b->lo ? (a->lo<b->lo?-1:1) : a->w!=b->w ? (a->w<b->w?-1:1) : a->at<b->at?-1:a->at>b->at; }
static void kput(CC *c,kimg *m,const char *s,size_t n){   /* append to the text (scratch arena, doubled on growth) */
  if(!m->s || m->cap-m->len<=n){ size_t cap=m->cap?m->cap:256;
    while(cap-m->len<=n){ if(cap>SIZE_MAX/2){ cc_raise_oom(c); return; } cap*=2; }
    char *ns=(char *)bcir_host_arena_allocate(&c->scratch,cap,1u); if(!ns){ cc_raise_oom(c); return; }
    if(m->s && m->len) memcpy(ns,m->s,m->len);
    m->s=ns; m->cap=cap; }
  memcpy(m->s+m->len,s,n); m->len+=n; m->s[m->len]=0; }
static int kfirst(const kimg *m,long long lo){   /* the first stored scalar at bit `lo` or after it */
  int a=0,b=m->n; while(a<b){ int h=a+(b-a)/2; if(m->r[h].lo<lo) a=h+1; else b=h; } return a; }
static void kspell(CC *c,kimg *m,const krec *r){   /* `Nu`, `-N`, and the one negative with no positive counterpart */
  char d[48]; int l = !r->neg ? snprintf(d,sizeof d,"%lluu",r->v)
    : r->v==(1ull<<63) ? snprintf(d,sizeof d,"(-9223372036854775807 - 1)") : snprintf(d,sizeof d,"-%llu",0ull-r->v);
  kput(c,m,d,(size_t)l); }
static int kzero(const char *s,size_t n){ return (n==2 && !memcmp(s,"0u",2)) || (n==3 && !memcmp(s,"{0}",3)); }
/* The rendered initializer of one subobject of the image (the oracle's `_render_unit`). */
static void kunit(CC *c,kimg *m,const iunit *u,int depth){
  long long lo=(long long)u->off*8+u->bit_off;
  if(u->k==IK_SCAL){ long long w=u->bit_w?u->bit_w:(long long)u->size*8;
    for(int i=kfirst(m,lo); i<m->n && m->r[i].lo==lo; i++) if(m->r[i].w==w){ kspell(c,m,&m->r[i]); return; }
    kput(c,m,"0u",2); return; }
  long long hi=lo+(long long)u->size*8; int i=kfirst(m,lo);
  if(i>=m->n || m->r[i].lo>=hi){ kput(c,m,"{0}",3); return; }
  if(depth>=IW_MAXFRAMES){ fail(c,"an initializer nested deeper than 64 subobjects"); return; }
  ifrm fr; memset(&fr,0,sizeof fr); fr.u=*u; fr.count=iunit_count(c,u);
  kput(c,m,"{",1);
  if(u->k==IK_ARR){                                       /* the elements holding a value; a gap is designated */
    long long st=(long long)(u->size/(u->dims[0]>0?u->dims[0]:1))*8; int prev=-1;
    while(i<m->n && m->r[i].lo<hi && !c->failed){ int k=(int)((m->r[i].lo-lo)/st); iunit e;
      if(prev>=0) kput(c,m,", ",2);
      if(k!=prev+1){ char d[32]; int l=snprintf(d,sizeof d,"[%d] = ",k); kput(c,m,d,(size_t)l); }
      iunit_elem(&e,u,k); kunit(c,m,&e,depth+1); prev=k; i=kfirst(m,lo+(long long)(k+1)*st); } }
  else if(c->s[u->sidx].is_union){                        /* the member the walk gave it */
    const sdef *S=&c->s[u->sidx]; int k=0; iunit s;
    for(int q=m->W->un0; q<c->iw_nun; q++) if(c->iw_un[q].off==u->off && c->iw_un[q].sidx==u->sidx){ k=c->iw_un[q].member; break; }
    for(int g=0; g<S->nanon; g++) if(k && S->anon[g].first==k){ fail(c,KV_ANONUNION); return; }
    if(k){ kput(c,m,".",1); kput(c,m,S->f[k].name,strlen(S->f[k].name)); kput(c,m," = ",3); }
    fr.idx=k; iunit_at(c,&fr,&s); kunit(c,m,&s,depth+1); }
  else { size_t keep=m->len; int first=1;                 /* every member in order, up to the last holding a value */
    while(fr.idx<fr.count && !c->failed){ iunit s; fr.idx=iunit_at(c,&fr,&s);
      if(!first) kput(c,m,", ",2);
      first=0; size_t at=m->len; kunit(c,m,&s,depth+1);
      if(!kzero(m->s+at,m->len-at)) keep=m->len; }
    m->len=keep; if(m->s) m->s[keep]=0; }
  kput(c,m,"}",1);
}
/* The rendered initializer of the static the walk `W` imaged, whose object `obj` is -- in the scratch arena; NULL when
 * the image is zero (the declaration's own `{0}` / `0u`). */
static char *kimage(CC *c,const iwalk *W,const iunit *obj){
  int n=c->iw_nrng-W->rng0, m=0;
  if(n<=0) return NULL;
  krec *r=(krec *)bcir_host_arena_allocate(&c->scratch,(size_t)n*sizeof *r,sizeof(long long));
  if(!r){ cc_raise_oom(c); return NULL; }
  for(int k=0;k<n;k++){ const irange *g=&c->iw_rng[W->rng0+k];
    r[k].lo=g->lo; r[k].w=g->hi-g->lo; r[k].v=g->v; r[k].neg=g->neg; r[k].at=k; }
  qsort(r,(size_t)n,sizeof *r,krec_cmp);
  for(int k=0;k<n;k++){                                   /* each scalar's last store, and only a nonzero one */
    if(k+1<n && r[k+1].lo==r[k].lo && r[k+1].w==r[k].w) continue;
    if(r[k].v) r[m++]=r[k]; }
  if(!m) return NULL;
  kimg im; memset(&im,0,sizeof im); im.r=r; im.n=m; im.W=W;
  iunit top=*obj;
  if(top.k==IK_ARR && top.nd>1){ top.dims[0]=top.size/(top.es>0?top.es:1); top.dims[1]=top.dims[2]=0; top.nd=1; }
  kunit(c,&im,&top,0);
  return c->failed ? NULL : im.s;
}
/* A static's initializer at the cursor (the oracle's `_static_init`): the initializer walk in constant mode -- each
 * entry folded, each store recorded in the static's image, no claim left behind (C initializes a static once, before
 * the program runs, never at a call) -- and the image rendered (`kimage`, into *text; NULL for a zero image). A scalar
 * takes one value (`{e}` braced, `{}` zero); an aggregate or an array a brace list, a character array a string
 * literal. Returns the top-level elements reached, which size an inferred array. */
static int init_image(CC *c, const iunit *obj, int inferred, char **text){
  iwalk W; memset(&W,0,sizeof W);
  W.top_array=obj->k==IK_ARR; W.inferred=inferred; W.konst=1; W.rng0=c->iw_nrng; W.un0=c->iw_nun; *text=NULL;
  spec_mark m; spec_begin(c,&m);                          /* the entries' claims: folded, then rolled back */
  if(obj->k==IK_ARR && isk(c,T_STR)){                     /* `char s[] = "ab"` is `{"ab"}` (6.7.9p14) */
    if(!init_char_array(obj,str_tok_unit(c,pk(c)))) fail(c,"an array is initialized by a brace list or a string literal");
    else init_string(c,&W,obj,1,inferred?-1:obj->dims[0]); }
  else if(obj->k!=IK_SCAL && is(c,"{")) init_list(c,&W,obj,1,obj->k==IK_ARR);
  else if(obj->k==IK_ARR) fail(c,"an array is initialized by a brace list or a string literal");
  else if(is(c,"{")){ ifrm fr; memset(&fr,0,sizeof fr); init_sublist(c,&W,&fr,obj); }
  else { uint32_t v=init_value(c,&W); if(!c->failed) init_store(c,&W,obj,v,0); }   /* a scalar's value (a struct's
                                                                                      * is never a constant) */
  spec_end(c,&m);
  if(!c->failed){ iunit sized=*obj;                       /* an inferred `[]`: the rows the walk reached */
    if(inferred && sized.k==IK_ARR){ int n=W.top_n<1?1:W.top_n;
      sized.size=sized.size/(sized.dims[0]>0?sized.dims[0]:1)*n; sized.dims[0]=n; }
    *text=kimage(c,&W,&sized); }
  c->iw_nrng=W.rng0; c->iw_nun=W.un0;                     /* its records end with it */
  return W.top_n;
}
/* Record the static local `nm` (resource `rid`) with its rendered image `text` (NULL: zero); the function owns a copy. */
static void add_static(CC *c, const tok *nm, uint32_t rid, const char *text, int thread){
  bcir_func *f=c->fn; char *cp=NULL;
  if(!CC_ENSURE(c,f->statics,f->n_statics,f->cap_statics)) return;
  if(text){ size_t z=strlen(text)+1; cp=(char *)bcir_host_allocate(&c->allocator,z);
    if(!cp){ cc_raise_oom(c); return; }
    memcpy(cp,text,z); }
  idcpy(c,f->statics[f->n_statics].name,nm); f->statics[f->n_statics].text=cp; f->statics[f->n_statics].rid=rid;
  f->statics[f->n_statics].thread_storage=(uint8_t)(thread?1:0);
  f->n_statics++;
}
/* Re-evaluate the bounds of the stores into `rid` from claim `from` on, once an inferred array has its count. */
static void init_remask(CC *c, uint32_t rid, size_t from){
  bcir_bounds bnd=access_bnd(c,rid);
  for(size_t i=from; i<c->fn->n_claims; i++){ bcir_claim *cl=&c->fn->claims[i];
    if(cl->opcode==BCIR_OP_STORE && cl->n_rd>=1 && cl->rd[0]==rid) cl->bounds=bnd; }
}
/* A C99 compound literal `(type){init}` -- an anonymous local of `type`, initialized exactly like a braced local
 * decl, and yielded as an rvalue rid or addressed under `&`. A struct/union takes the walk; a scalar `(int){v}`
 * copies its one value (`{}` is zero). */
static uint32_t p_compound_literal(CC *c, const bcir_ctype *ty, int si) {
  char nm[BCIR_CIR_NAME]; snprintf(nm,sizeof nm,"_cl%u",++c->cl_ctr);
  if(ty->kind==1){                                   /* a struct/union compound literal */
    uint32_t rid=add_res(c,BCIR_DOM_RAM, si>=0?c->s[si].size:ty->size, 1, 0, BCIR_RK_AGGREGATE, nm);
    if(c->fn->n_res) fits(c,c->fn->res[c->fn->n_res-1].agg,BCIR_CIR_AGG,"%s %s",
                              ty->is_union?"union":"struct", ty->tag);
    iunit obj=iunit_struct(c,si);
    (void)init_object(c, rid, &obj, 0);
    return rid;
  }
  /* a scalar compound literal `(int){v}` -- a named local of its type + a c.copy of the single value: a pointer one a
   * pointer local, as the oracle's (a `uint32_t` one cut the pointer it held, and `*(T *){p}` read no pointer), a
   * function-pointer one declared by its alias (CF-RTFP) */
  uint32_t rid=ty->kind==2 ? temp_ptr(c,ty,si) : add_res(c,BCIR_DOM_RAM, ty->size?ty->size:4, 1, 0, BCIR_RK_SCALAR, nm);
  if(c->fn->n_res){ bcir_resource *rr=&c->fn->res[c->fn->n_res-1];
    if(ty->kind==2) fits(c,rr->name,sizeof rr->name,"%s",nm);
    else { rr->is_signed=(uint8_t)(ty->signd?1:0); rr->is_float=(uint8_t)(ty->is_float?1:0);
      rr->is_bool=(uint8_t)(ty->is_bool?1:0); rr->is_plain_char=(uint8_t)(ty->is_plain_char?1:0);
      if(ty->kind==3 && ty->tag[0]){ rr->is_funcptr=1; fits(c,rr->agg,BCIR_CIR_AGG,"%s",ty->tag); } } }
  eat(c,"{");
  uint32_t v;
  if(is(c,"}")){ v=temp(c,4);                        /* `(int){}` (C23 empty) -> 0 */
    bcir_claim *k=new_claim(c,"c.const",BCIR_OP_LOAD); if(k){k->n_wr=1;k->wr[0]=v;k->n_imm=1;k->imm[0]=0;} }
  else if(is(c,".")||is(c,"[")||is(c,"{")){ fail(c,"a braced scalar initializer holds one expression"); return temp(c,4); }
  else v=p_expr(c);
  if(is(c,",")) c->i++;                              /* a tolerated trailing comma */
  if(!is(c,"}")){ fail(c,"a braced scalar initializer holds one expression"); return v; }
  eat(c,"}");
  v=null_pointer(c,v,ty);                            /* `(T *){0}`: a null pointer of the literal's type */
  bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD);
  if(cl){cl->n_rd=1;cl->rd[0]=v;cl->n_wr=1;cl->wr[0]=rid;}
  return rid;
}
/* An array compound literal `(T[N]){...}` / `(T[]){...}` / `(T[A][B]){...}` / `(struct P[]){...}` -- an anonymous
 * local array of `T`, initialized exactly like a braced local-array decl (the walk), yielded as the array
 * object. An inferred `[]` (or outer `[][B]`) takes its count from the rows the initializer reached, then its
 * stores' bounds are re-evaluated against it. The element store emits real typed `_cl[i] = v`. */
static uint32_t p_array_literal(CC *c, const bcir_ctype *ty, int si, int count, const int *la_dims, int la_nd) {
  char nm[BCIR_CIR_NAME]; snprintf(nm,sizeof nm,"_cl%u",++c->cl_ctr);
  int es = ty->kind==2 ? cc_abi(c)->pointer_size : (ty->size ? ty->size : 4);
  int dims[3]={0,0,0}, nd=la_nd>1?(la_nd<3?la_nd:3):1, inner=1;   /* the caller caps at 3 dims */
  if(la_nd>1){ for(int z=0;z<3;z++) dims[z]=la_dims[z]; } else dims[0]=count;
  for(int d=1; d<nd; d++) inner*=dims[d];
  int inferred = dims[0]<=0;
  if(inferred) dims[0]=1;                            /* provisional: the walk counts the rows */
  uint32_t rid = add_res(c, BCIR_DOM_RAM, es, dims[0]*inner, 0, BCIR_RK_SCALAR, nm);
  int ari = (int)c->fn->n_res - 1;
  if(ty->kind==1) fits(c,c->fn->res[ari].agg,BCIR_CIR_AGG,"%s %s",ty->is_union?"union":"struct",ty->tag);
  else if(ty->kind==3) ptee_fp(c,&c->fn->res[ari],ty);   /* `(op_t[N]){f, g}`: a table of function pointers, declared
                                                         * by its alias as a local table is (CF-RTFP) */
  else if(ty->is_float) c->fn->res[ari].is_float=1;  /* element type flags -> the decl emits `float`/`char`/... */
  else if(ty->kind==0){ c->fn->res[ari].is_signed=(uint8_t)(ty->signd?1:0);
    if(ty->is_bool) c->fn->res[ari].is_bool=1;
    if(ty->is_plain_char) c->fn->res[ari].is_plain_char=1; }
  size_t s_nclaims = c->fn->n_claims;                /* the init stores begin here -- re-mask after sizing */
  iunit obj=iunit_array(ty,si,es,dims,nd);
  int rows=init_object(c, rid, &obj, inferred);      /* note: temp()/new_claim may realloc res[]; re-index ari */
  if(inferred){ c->fn->res[ari].count=(uint32_t)((rows<1?1:rows)*inner); init_remask(c,rid,s_nclaims); }
  return rid;
}
static void p_block(CC *c){            /* `{ stmts }` or a single statement */
  if(ENTER_REC(c)){ LEAVE_REC(c); return; }   /* depth guard: p_block<->p_stmt nesting cycle */
  if(is(c,"{")){c->i++; int env_mark=c->nenv;   /* a block is a scope: its locals do not leak out */
    while(!is(c,"}")&&!isk(c,T_END)&&!c->failed)p_stmt(c);
    eat(c,"}"); c->nenv=env_mark;}              /* pop the block scope -- restore outer name bindings */
  else p_stmt(c);
  LEAVE_REC(c);
}
/* ++i / --i / i++ / i-- (value discarded) -> i = i ± 1 (const 1 + a bin op + a copy).  Returns 1 if
 * it consumed an increment/decrement, 0 (consuming nothing) otherwise. */
static int p_incdec(CC *c) {
  venv *v=NULL; char ch=0;
  if((is(c,"++")||is(c,"--")) && tat(c,c->i+1)->k==T_ID){            /* ++name / --name */
    const tok *nx=tat(c,c->i+2);                        /* ... a bare name: `++h.next->v`, `++p[i]` step the lvalue
                                                         * the expression grammar parses (CF-SPLIT2) */
    if(tok_is(nx,".")||tok_is(nx,"->")||tok_is(nx,"[")||tok_is(nx,"(")||tok_is(nx,"++")||tok_is(nx,"--")) return 0;
    v=lookup(c,tat(c,c->i+1)); if(!v) return 0; ch=c->t[c->i].s[0]; c->i+=2;
  } else if(isk(c,T_ID) && tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==2 &&
            (tat(c,c->i+1)->s[0]=='+'||tat(c,c->i+1)->s[0]=='-') && tat(c,c->i+1)->s[1]==tat(c,c->i+1)->s[0]){
    v=lookup(c,pk(c)); if(!v) return 0; ch=tat(c,c->i+1)->s[0]; c->i+=2;   /* name++ / name-- */
  } else return 0;
  uint32_t one=temp(c,4); bcir_claim *kc=new_claim(c,"c.const",BCIR_OP_LOAD);
  if(kc){kc->n_wr=1;kc->wr[0]=one;kc->n_imm=1;kc->imm[0]=1;}
  if(v->type.is_atomic && v->type.kind==0){             /* `a++;` on an `_Atomic` object is `a += 1` (the oracle's
                                                         * desugaring): one atomic read-modify-write */
    (void)emit_rmw(c,ch=='+'?"add":"sub",&v->type,v->rid,0,0,1,one,0,v->type.size?v->type.size:4,0,v->type.is_volatile,
                   BCIR_BND_ASSUMED);
    return 1;
  }
  if(v->type.kind==2){                                  /* pointer ++/-- : p += 1 / p -= 1 (verbatim) */
    char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.ptr%s",ch=='+'?"add":"sub");
    bcir_claim *cl=new_claim(c,op,BCIR_OP_ADD); if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=one;cl->n_wr=1;cl->wr[0]=v->rid;}
    return 1;
  }
  uint32_t tmp=binop_result(c,ch=='+'?"add":"sub",v->rid,one);   /* keep the operand width (long ++ -> int64) */
  bcir_claim *b=new_claim(c,ch=='+'?"c.bin.add":"c.bin.sub",ch=='+'?BCIR_OP_ADD:BCIR_OP_SUB);
  if(b){b->n_rd=2;b->rd[0]=v->rid;b->rd[1]=one;b->n_wr=1;b->wr[0]=tmp;}
  bcir_claim *cp=new_claim(c,"c.copy",BCIR_OP_ADD); if(cp){cp->n_rd=1;cp->rd[0]=tmp;cp->n_wr=1;cp->wr[0]=v->rid;}
  return 1;
}
/* one simple expression WITHOUT a trailing `;` -- a for-loop step element (each comma-separated piece
 * of `i++, j--, acc += d`). Mirrors the scalar/pointer assignment forms p_stmt handles (plain `=`,
 * compound `OP=`, inc/dec) so a step matches the oracle whether it is one element or a comma list. */
static void ctype_str(const bcir_ctype *ty,char *o,size_t n);   /* used to spell a captured funcptr signature */
static void p_simple(CC *c) {
  if(p_incdec(c)) return;
  if(isk(c,T_ID)){ tok id=*pk(c); venv *vp=lookup(c,&id); if(!vp) vp=use_global(c,&id);   /* a writable global */
    /* SNAPSHOT the env entry before the RHS p_expr below can realloc c->env[] (a stmt-expr / use_global). */
    venv vsnap; venv *v=NULL; if(vp){ vsnap=*vp; v=&vsnap; }
    if(v && tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='='){      /* name = expr */
      c->i+=2; uint32_t val=null_pointer(c,p_expr(c),&v->type);
      if(!agg_value_ok(c,obj_sidx(c,v),val,agg_assigned)) return;   /* a struct takes its own type */
      bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD); if(cl){cl->n_rd=1;cl->rd[0]=val;cl->n_wr=1;cl->wr[0]=v->rid;}
      mark_obj_write(c,cl,v);
      return; }
    if(v && is_compound_op(tat(c,c->i+1))){                                             /* name OP= expr */
      char ch=tat(c,c->i+1)->s[0];
      if(v->type.kind==2 && (ch=='+'||ch=='-')){       /* pointer arithmetic: p += n / p -= n (verbatim) */
        c->i+=2; uint32_t rhs=p_expr(c);
        char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.ptr%s",ch=='+'?"add":"sub");
        bcir_claim *cl=new_claim(c,op,BCIR_OP_ADD); if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=rhs;cl->n_wr=1;cl->wr[0]=v->rid;}
        return; }
      if(v->type.is_atomic && v->type.kind==0){         /* an `_Atomic` object: one atomic read-modify-write */
        c->i+=2; uint32_t rhs=p_expr(c);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc); (void)oc;
        (void)emit_rmw(c,suf,&v->type,v->rid,0,0,1,rhs,0,v->type.size?v->type.size:4,0,v->type.is_volatile,BCIR_BND_ASSUMED);
        return; }
      uint32_t cur=named_read(c,v);                     /* the current value (a volatile read, first) */
      c->i+=2; uint32_t rhs=p_expr(c);                  /* scalar:  name = name OP expr  (bin op + copy) */
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
      uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.bin.%s",suf);
      bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
      bcir_claim *cp=new_claim(c,"c.copy",BCIR_OP_ADD); if(cp){cp->n_rd=1;cp->rd[0]=tmp;cp->n_wr=1;cp->wr[0]=v->rid;}
      mark_obj_write(c,cp,v);
      return; } }
  (void)p_expr(c);
}
/* Store through a loaded POINTER chain (#fieldderef): `s->mid->k = v`, the two-hop `s->mid->leaf->x = v`,
 * the subscript `s->p[i] = v`, and their `OP=` compound forms. `ptr` is the loaded pointer rid, `psidx`
 * its pointee struct (-1 for a pointer-to-scalar), `pfld` the field it came from. Mirrors the member /
 * indexed store path but with the loaded pointer as the base; a further pointer hop loads and recurses.
 * (Bitfields through a pointer chain are not modelled here -- not part of the slice.) */
static void store_through_ptr_inner(CC *c, uint32_t ptr, int psidx, field pfld);
/* Depth-guarded wrapper: store_through_ptr self-recurses per pointer hop (`->`/`.`/`[`), so a long
 * `p->p->p->...->x = v` chain would exhaust the stack. Bump/check depth once per hop. */
static void store_through_ptr(CC *c, uint32_t ptr, int psidx, field pfld) {
  if(ENTER_REC(c)){ LEAVE_REC(c); return; }
  store_through_ptr_inner(c, ptr, psidx, pfld); LEAVE_REC(c);
}
static void store_through_ptr_inner(CC *c, uint32_t ptr, int psidx, field pfld) {
  if(is(c,"[")){                                        /* `...->p[i] = v` -- indexed store via the pointer */
    c->i++; uint32_t idx=p_expr(c); eat(c,"]");
    if(psidx>=0 && (is(c,".")||is(c,"->"))){            /* `...->p[i].f = v`: a member of an element of a pointer
                                                         * to structs, stored strided by the element */
      venv e=ptr_elem_base(c,ptr,psidx,&pfld);
      { field af; uint32_t tot; int r=aos_member_array(c,&e,idx,&af,&tot);   /* `...->p[i].m[j] = v` */
        if(r<0) return;
        if(r>0){ if(!store_member_elem_stmt(c,&e,&af,tot)) fail(c,"="); return; } }
      field sub; if(!aos_elem_field(c,&e,&sub)) return;
      if(!store_elem_field_stmt(c,&e,idx,&sub) && !c->failed) fail(c,"=");
      return; }
    venv b; memset(&b,0,sizeof b); b.rid=ptr; b.sidx=-1;
    b.type.size=pfld.ptee_size?pfld.ptee_size:4; b.type.signd=pfld.signd; b.type.is_float=(uint8_t)pfld.ptee_float;
    b.type.is_volatile=(uint8_t)(pfld.ptee_volatile?1:0);
    b.type.is_atomic=(uint8_t)(pfld.ptee_atomic?1:0);   /* a pointer to `_Atomic` storage (CF-ATOMIC) */
    uint32_t val;
    if(pfld.ptee_atomic && is_compound_op(&c->t[c->i])){   /* on `_Atomic`: one atomic read-modify-write */
      char ch=c->t[c->i].s[0]; c->i++; uint32_t rhs=p_expr(c);
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc); (void)oc;
      (void)emit_rmw(c,suf,&b.type,ptr,1,idx,1,rhs,0,0,0,pfld.ptee_volatile,BCIR_BND_ASSUMED);
      return;
    }
    if(is_compound_op(&c->t[c->i])){ char ch=c->t[c->i].s[0]; c->i++;
      uint32_t cur=emit_index(c,&b,idx); uint32_t rhs=p_expr(c);
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
      uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.bin.%s",suf);
      bcir_claim *bb=new_claim(c,op,oc); if(bb){bb->n_rd=2;bb->rd[0]=cur;bb->rd[1]=rhs;bb->n_wr=1;bb->wr[0]=tmp;}
      val=tmp;
    } else { if(!eat(c,"="))return; val=p_expr(c);
      if(!agg_value_ok(c,pfld.ptee_sidx>=0 && pfld.ptee_depth<=1 ? pfld.ptee_sidx : -1,val,agg_assigned)) return; }
    bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
    if(cl){cl->n_rd=3;cl->rd[0]=ptr;cl->rd[1]=idx;cl->rd[2]=val;cl->bounds=BCIR_BND_ASSUMED;
      mark_access(c,cl,pfld.ptee_volatile); mark_atomic(cl,pfld.ptee_atomic);}
    return;
  }
  if(!(is(c,"->")||is(c,"."))){ fail(c,"expected ->/./[ after a pointer field"); return; }
  if(psidx<0){ fail(c,"member store through a pointer to a non-struct"); return; }
  c->i++; tok fn=adv(c); sdef *S=&c->s[psidx]; int fi=-1;
  for(int i=0;i<S->nf;i++) if((int)strlen(S->f[i].name)==fn.n&&!strncmp(S->f[i].name,fn.s,fn.n)) fi=i;
  if(fi<0){ fail(c,"unknown field"); return; }
  field f=member_descend(c,S->f[fi]);
  venv b; memset(&b,0,sizeof b); b.rid=ptr; b.sidx=psidx; b.type.kind=1;   /* base = the loaded pointer */
  b.type.is_volatile=(uint8_t)(pfld.ptee_volatile?1:0);
  if(f.is_ptr && (is(c,"->")||is(c,".")||is(c,"["))){  /* another pointer hop: load it, recurse */
    uint32_t nptr=emit_member(c,&b,&f,0); store_through_ptr(c,nptr,f.ptee_sidx,f); return;
  }
  uint32_t val;                                        /* terminal member store through the loaded pointer */
  if(f.is_atomic && is_compound_op(&c->t[c->i])){      /* an `_Atomic` member: one atomic read-modify-write */
    char ch=c->t[c->i].s[0]; c->i++; uint32_t rhs=p_expr(c); bcir_ctype ft=field_slot(&f);
    const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc); (void)oc;
    (void)emit_rmw(c,suf,&ft,ptr,0,0,1,rhs,f.byte_off,f.size,0,f.is_volatile||pfld.ptee_volatile,BCIR_BND_ASSUMED);
    return;
  }
  if(is_compound_op(&c->t[c->i])){ char ch=c->t[c->i].s[0]; c->i++;
    uint32_t cur=emit_member(c,&b,&f,0); uint32_t rhs=p_expr(c);
    const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
    uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.bin.%s",suf);
    bcir_claim *bb=new_claim(c,op,oc); if(bb){bb->n_rd=2;bb->rd[0]=cur;bb->rd[1]=rhs;bb->n_wr=1;bb->wr[0]=tmp;}
    val=tmp;
  } else { if(!eat(c,"="))return; val=p_expr(c); if(!agg_value_ok(c,f.sidx,val,agg_assigned)) return; }
  /* the member store every other path takes: C's conversion, a `_Bool` member's flag, a bitfield's unit */
  if(f.bit_w) store_member_bf(c,&b,&f,val); else store_member(c,&b,&f,val);
}
/* A case label, the cursor past `case`: its integer constant expression folded (`ce_fold`) and its marker spelled the
 * value exactly in its type (`kdigits`, the oracle's `_const_spelling`) -- `%lld` had spelled 0xFFFFFFFFFFFFFFFFu as
 * -1. BCIR_NOINLINE: its buffers stay out of the recursive statement parser's frame. */
static BCIR_NOINLINE void case_label(CC *c){
  kval v; char d[32], op[BCIR_CIR_OP];
  if(!ce_fold(c,0,&v)) return;
  kdigits(d,sizeof d,v.p,v.sgn && (v.p>>63));
  eat(c,":"); fits(c,op,sizeof op,"c.case:%s",d); marker(c,op,0,0);
}
static void p_stmt_inner(CC *c);
/* Depth-guarded wrapper: p_stmt is a recursive-cycle entry (p_stmt->p_block->p_stmt, and the stmt-expr
 * `({...})` path p_stmt_expr->p_stmt). Bump/check depth once per statement nesting level. */
/* A claim from `from` on -- those of the statement just parsed -- that reads a void value: refused, and 1 (CF-VOIDVAL). A
 * void value is never written, so any read of one is a use of it; a unit that made none skips the walk. */
static int void_read(CC *c,size_t from){
  if(!c->n_void || !c->fn) return 0;
  for(size_t i=from;i<c->fn->n_claims;i++){ const bcir_claim *cl=&c->fn->claims[i];
    for(int k=0;k<cl->n_rd;k++) if(void_value(c,cl->rd[k])){ fail(c,CC_VOID_VALUE); return 1; } }
  return 0;
}
static void p_stmt(CC *c) {
  if(ENTER_REC(c)){ LEAVE_REC(c); return; }
  size_t from=c->fn?c->fn->n_claims:0;
  p_stmt_inner(c);
  if(!c->failed) (void)void_read(c,from);
  LEAVE_REC(c);
}
static void p_stmt_inner(CC *c) {
  if(is(c,";")){c->i++;return;}          /* empty statement -> a no-op (`for(...);`, `if(c);`, `;;`) */
  if(is(c,"return")){c->i++;
    if(!is(c,";")){uint32_t rv=p_expr(c);
      if(void_value(c,rv)){                            /* `return f();` of a void f: made for its effects, and a void
                                                         * function returns nothing (the oracle's `_VOID_RID`); any
                                                         * other would return the value it has none of */
        const bcir_ctype *rt=&c->fn->ret;
        if(!(rt->kind==0 && rt->size==0 && !rt->is_float && !rt->is_valist && rt->bit_width==0)){ fail(c,CC_VOID_VALUE); return; }
        marker(c,"c.return",0,0); }
      else if(c->fn->ret.kind!=1 && !scalar_value_ok(c,rv)) return;   /* `return a;` of a struct from a function
                                                                        * returning a scalar (CF-STRUCTARITH) */
      else{rv=null_pointer(c,rv,&c->fn->ret);c->fn->return_rid=rv;c->fn->has_return=1;marker(c,"c.return",rv,1);}}
    else marker(c,"c.return",0,0);
    eat(c,";");return;}
  if(is(c,"if")){                      /* L6: if / else -> structured markers */
    c->i++;eat(c,"(");uint32_t cond=p_expr(c);eat(c,")");
    if(!cond_value_ok(c,cond,0)) return;   /* `if (a)` of a struct (CF-STRUCTCOND) */
    marker(c,"c.if",cond,1); p_block(c);
    if(is(c,"else")){c->i++;marker(c,"c.else",0,0);p_block(c);}
    marker(c,"c.endif",0,0); return;
  }
  if(is(c,"while")){                   /* L6: a bounded while loop (cond re-evaluated each iter) */
    c->i++;marker(c,"c.loop",0,0);
    eat(c,"(");uint32_t cond=p_expr(c);eat(c,")");
    if(!cond_value_ok(c,cond,0)) return;
    marker(c,"c.loop.test",cond,1); p_block(c);
    marker(c,"c.cont.tgt",0,0); marker(c,"c.endloop",0,0); return;   /* continue -> re-test (top) */
  }
  if(is(c,"for")){                     /* for(init; cond; step) body == init; while(cond){body; step} */
    c->i++; eat(c,"(");
    int env_mark=c->nenv;             /* for-init + body decls are loop-scoped (no leak past the loop) */
    if(is(c,";")) c->i++;             /* empty init */
    else p_stmt(c);                   /* init: a decl / assignment / expr (consumes its `;`) */
    marker(c,"c.loop",0,0);
    uint32_t cond;
    if(is(c,";")){ cond=temp(c,4); bcir_claim *cl=new_claim(c,"c.const",BCIR_OP_LOAD);
      if(cl){cl->n_wr=1;cl->wr[0]=cond;cl->n_imm=1;cl->imm[0]=1;} }   /* empty cond -> 1 */
    else cond=p_expr(c);
    eat(c,";");
    if(!cond_value_ok(c,cond,0)) return;
    marker(c,"c.loop.test",cond,1);
    int step_start=c->i,pd=1;          /* record the step tokens; skip to the matching `)` */
    while(!isk(c,T_END)&&pd){ if(is(c,"("))pd++; else if(is(c,")")){pd--; if(!pd)break;} c->i++; }
    int step_end=c->i; eat(c,")");
    p_block(c);                        /* the loop body */
    marker(c,"c.cont.tgt",0,0);        /* continue -> run the step, then re-test */
    if(step_end>step_start){ int save=c->i; c->i=step_start;   /* step @ iter end */
      p_simple(c);                                             /* `i++, j--, k = …` -- the comma */
      while(is(c,",")&&!c->failed){ c->i++; p_simple(c); }     /* operator in its dominant position */
      c->i=save; }
    c->nenv=env_mark;                  /* pop the loop scope -- restore outer name bindings */
    marker(c,"c.endloop",0,0); return;
  }
  if(is(c,"do")){                      /* do body while(cond);  == loop { body; if(!cond) break; } */
    c->i++; marker(c,"c.loop",0,0);
    p_block(c);                        /* body runs first */
    marker(c,"c.cont.tgt",0,0);        /* continue -> the bottom test */
    eat(c,"while"); eat(c,"(");
    uint32_t cond=p_expr(c); eat(c,")"); eat(c,";");
    if(!cond_value_ok(c,cond,0)) return;
    marker(c,"c.loop.test",cond,1);    /* the test is at the bottom */
    marker(c,"c.endloop",0,0); return;
  }
  if(is(c,"break")){ c->i++; eat(c,";"); marker(c,"c.break",0,0); return; }
  if(is(c,"continue")){ c->i++; eat(c,";"); marker(c,"c.continue",0,0); return; }
  if(is(c,"goto")){ c->i++;
    if(is(c,"*")){ c->i++;                                       /* `goto *expr;` -- a computed (indirect) goto (GNU) */
      uint32_t tgt=p_expr(c); eat(c,";");                        /* the target is lowered to a void* address */
      marker(c,"c.cgoto",tgt,1); return; }                       /* emit-only (NOP marker), carries the target rid */
    tok lb=adv(c); eat(c,";");                                   /* goto label; -- an emit-only marker */
    char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.goto:%.*s",lb.n,lb.s); new_claim(c,op,BCIR_OP_NOP); return; }
  if(isk(c,T_ID)&&tat(c,c->i+1)->k==T_PUN&&tat(c,c->i+1)->n==1&&tat(c,c->i+1)->s[0]==':'){  /* `name:` -- a label */
    tok lb=adv(c); c->i++; char op[BCIR_CIR_OP];
    fits(c,op,sizeof op,"c.label:%.*s",lb.n,lb.s); new_claim(c,op,BCIR_OP_NOP); return; }
  if(is(c,"switch")){                  /* a real C switch: case labels + fallthrough preserved */
    c->i++; eat(c,"(");
    uint32_t disc=p_expr(c);           /* the discriminant, lowered once */
    if(!cond_value_ok(c,disc,1)) return;   /* an integer (CF-STRUCTCOND) */
    eat(c,")"); eat(c,"{");
    marker(c,"c.switch",disc,1);
    while(!is(c,"}")&&!isk(c,T_END)&&!c->failed){
      if(is(c,"case")){ c->i++; case_label(c); }   /* case <const>: */
      else if(is(c,"default")){ c->i++; eat(c,":"); marker(c,"c.default",0,0); }
      else p_stmt(c);                  /* body statements (break -> c.break, no implicit break) */
    }
    eat(c,"}");
    marker(c,"c.endswitch",0,0);
    return;
  }
  if(is(c,"{")){p_block(c);return;}
  if(decl_start_tok(c,pk(c))){
    c->saw_static=0; c->saw_extern=0; c->saw_thread=0;   /* THIS declaration's own storage classes, wherever spelled */
    bcir_ctype base;int si;if(p_type_base(c,&base,&si))return;   /* the shared specifier (eats `static`, NOT `*`) */
    if(c->saw_extern){ fail(c,"a block-scope extern declaration is not supported"); return; }   /* names an object
      * defined elsewhere, never a new local: binding it as one read an uninitialized object (CF-STORAGE) */
    int is_static = c->saw_static;                 /* `volatile static T n` / `T static n` are static (6.7p1) */
    int is_thread = c->saw_thread;                 /* thread storage is kept (CF-TLS): each thread its own object;
                                                    * at block scope it needs `static` too (C11 6.7.1p3) */
    if(is_thread && !is_static){ fail(c,"a block-scope `_Thread_local` object that is not `static`"); return; }
    int btdd[3], btd=td_dims_of(c,btdd);   /* a typedef'd array specifier */
    /* one or more comma-separated declarators sharing this specifier: `T a = x, b, c = z;`. Each gets
     * its OWN declarator `*`/`[]` shape off a fresh copy of the base, so `int *p, q;` types p as `int*`
     * and q as int (per-declarator, matching the oracle), `int *p, *q;` types both as pointers, and a
     * per-declarator array (`int a[2], b;`) no longer leaks its dims onto the next declarator. */
    for(;;){
      if(btd && is(c,"*")){ fail(c,"a pointer to a typedef'd array is not supported"); return; }
      bcir_ctype ty=base; apply_stars(c,&ty);   /* this declarator's own leading `*`s (none -> base type) */
      int fpre=0, fpre_nd=0, fpre_dims[3]={0,0,0}; tok fpre_nm; memset(&fpre_nm,0,sizeof fpre_nm);
      { int k=fp_decl_at(c,0);
        if(k && k!=c->i+3){   /* `RET (*t[N])(P)`, `RET (**pp)(P)`: an array of, or a pointer to, function pointers
                               * (CF-FPTAB) -- parsed whole, then declared as `ALIAS t[N]` / `ALIAS *pp` are */
          bcir_ctype fty; int stars;
          if(fp_inline_decl(c,&ty,si,&fty,&stars,fpre_dims,&fpre_nd,&fpre_nm)) return;
          ty=fty; for(int s=0;s<stars;s++) fp_star(c,&ty);
          fpre=1; } }
      if(!fpre && is(c,"(") && tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='*'
         && tat(c,c->i+2)->k==T_ID
         && tat(c,c->i+3)->k==T_PUN && tat(c,c->i+3)->n==1 && tat(c,c->i+3)->s[0]==')'
         && tat(c,c->i+4)->k==T_PUN && tat(c,c->i+4)->n==1 && tat(c,c->i+4)->s[0]=='('){
        /* a function-pointer LOCAL `RET (*name)(PARAMS) = fn;` -- the twin of the oracle's Decl funcptr.
         * Like a direct funcptr PARAMETER there is no alias to print, so capture the full signature as a
         * synthesized prelude typedef `__bcir_fpN` and bind the local kind-3 with that tag; the env binding
         * lets p_icall dispatch `f(x)` (typed by the captured return, #signedfnptr) and a bare `f = g;`
         * decay-assign through p_simple. Scalar return + parameter types. */
        bcir_ctype ret=ty;                            /* the already-parsed return type */
        c->i+=2; tok nm=adv(c);                        /* `( *` then the funcptr NAME */
        if(!eat(c,")")||!eat(c,"("))return;            /* `) (` -- into the parameter-type list */
        /* the typedef line is appended to the growable prelude as it is parsed (CF-BUF): a signature of any length
         * is emitted -- a fixed 512-byte line had made a long one's emit impossible (cfront_sec_sigoverflow.c) --
         * and a parameter list that fails to parse takes its partial line back out */
        char rets[BCIR_EMIT_TYPE]; ctype_qstr(&ret,rets,sizeof rets);
        size_t line=c->fpdefs.w; const bcir_ctype *ps; int np;
        ctext_putf(c,&c->fpdefs,"typedef %s (*__bcir_fp%d)(",rets,c->n_fpdef);
        int va;
        if(fp_param_list(c,1,&ps,&np,&va)){ c->fpdefs.w=line; if(c->fpdefs.s) c->fpdefs.s[line]=0; return; }
        if(!eat(c,")"))return;                          /* past the parameter-type list */
        bcir_ctype fty; memset(&fty,0,sizeof fty); fty.kind=3; fty.size=cc_abi(c)->pointer_size; fty.signd=0;
        fp_capture_ret(&fty,&ret,si);                   /* its return type types a c.call.indirect result */
        snprintf(fty.tag,sizeof fty.tag,"__bcir_fp%d",c->n_fpdef); c->n_fpdef++;
        fty.fp_sig=sig_addv(c,&ret,ps,np,fty.tag,"",va);   /* ... and its parameters, a null pointer argument */
        char fnb[BCIR_CIR_NAME]; idcpy(c,fnb,&nm);
        uint32_t frid=add_res(c,BCIR_DOM_RAM,8,1,0,BCIR_RK_SCALAR,fnb);   /* a funcptr-wide scalar local */
        if(c->fn->n_res){ bcir_resource *fr=&c->fn->res[c->fn->n_res-1];
          fr->is_funcptr=1; fits(c,fr->agg,BCIR_CIR_AGG,"%s",fty.tag); }   /* emit `__bcir_fpN f;` up front */
        env_add(c,&nm,frid,&fty,-1);                    /* p_icall finds kind-3 -> a c.call.indirect dispatch */
        if(is_static){ char *st=NULL;                   /* a static function pointer: its image, as any static's */
          if(is(c,"=")){ c->i++; iunit obj=iunit_scalar(c,&fty); (void)init_image(c,&obj,0,&st); if(c->failed) return; }
          add_static(c,&nm,frid,st,is_thread); }
        else if(is(c,"=")){ c->i++;                     /* an init: a function name -> a funcptr value (c.copy) */
          uint32_t v=0;
          if(is(c,"{")){ c->i++;                        /* `= {f}`, `= {f,}` is `= f`, and `= {}` a null pointer (6.7.9p11,
                                                         * CF-RTFP) -- as a typedef'd one's and the oracle's `_braced_scalar` */
            if(is(c,"}")){ v=temp(c,4);
              bcir_claim *k=new_claim(c,"c.const",BCIR_OP_LOAD); if(k){k->n_wr=1;k->wr[0]=v;k->n_imm=1;k->imm[0]=0;} }
            else if(is(c,".")||is(c,"[")||is(c,"{")){ fail(c,"a braced scalar initializer holds one expression"); return; }
            else { v=p_expr(c); if(is(c,",")) c->i++; }
            if(c->failed) return;
            if(!is(c,"}")){ fail(c,"a braced scalar initializer holds one expression"); return; }
            c->i++; }
          else v=p_expr(c);
          v=null_pointer(c,v,&fty);                     /* `= 0`: a null function pointer, not an `int` (CF-NULLPTR; CF-RTFP) */
          bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD); if(cl){cl->n_rd=1;cl->rd[0]=v;cl->n_wr=1;cl->wr[0]=frid;} }
        if(is(c,",")){ c->i++; continue; }              /* another declarator off the same specifier */
        break;
      }
      tok nm;
      if(fpre) nm=fpre_nm;
      else { if(!isk(c,T_ID)){ fail(c,"expected declarator name"); return; } nm=adv(c); }
      char nb[BCIR_CIR_NAME]; idcpy(c,nb,&nm);
      int arr=0,la_nd=0,la_dims[3]={0,0,0};            /* T name[N] / T m[A][B] -- a (multi-dim) local array */
      /* scan ALL `[...]` dims WITHOUT lowering, recording each as either a literal value or a runtime-expr
       * token range; then classify (oracle order): all-literal -> a static array (unchanged); >=1 runtime
       * dim with ONE dim -> the 1-D VLA path (byte-identical); >=2 dims with a runtime dim -> a multi-dim
       * VLA. Deferring the lowering keeps the dim snapshots in canonical dim order (each dim is evaluated
       * then snapshotted, in turn) -- the byte-parity contract with the oracle's Decl branch. */
      int dim_nd=0; int dim_is_lit[8]; int dim_lit[8]; int dim_tok[8]; int any_vla=0;
      for(int d=0; d<fpre_nd; d++){ int dim=fpre_dims[d];   /* an inline table's own dims (CF-FPTAB), all fixed */
        if(dim_nd<8){ dim_is_lit[dim_nd]=1; dim_lit[dim_nd]=dim; }
        if(la_nd<3){ la_dims[la_nd]=dim; }
        la_nd++; arr = arr?arr*dim:dim; dim_nd++; }
      while(is(c,"[")){ c->i++;
        long long cd=is(c,"]") ? 0 : ce_dim(c,1);    /* an integer constant expression: a fixed dim (6.7.6.2p4) */
        if(c->failed) return;
        if(cd<0){                                       /* any other dim -> a runtime VLA dim */
          if(dim_nd<8){ dim_is_lit[dim_nd]=0; dim_tok[dim_nd]=c->i; } any_vla=1;
          int paren=0; while(!(paren==0 && is(c,"]")) && !isk(c,T_END)){   /* skip to the matching `]` */
            if(is(c,"[")||is(c,"(")) paren++; else if(is(c,")")) paren--; c->i++; }
          eat(c,"]"); }
        else { int dim=(int)cd; eat(c,"]");
          if(dim_nd<8){ dim_is_lit[dim_nd]=1; dim_lit[dim_nd]=dim; }
          if(la_nd<3)la_dims[la_nd]=dim; la_nd++; arr = arr?arr*dim:dim; }
        dim_nd++; }
      for(int d=0; d<btd; d++){ int dim=btdd[d];   /* a typedef'd array's dims follow the declarator's own */
        if(dim_nd<8){ dim_is_lit[dim_nd]=1; dim_lit[dim_nd]=dim; }
        if(la_nd<3)la_dims[la_nd]=dim; la_nd++; arr = arr?arr*dim:dim; dim_nd++; }
      int after_dims=c->i;                             /* the cursor past the last `]` (restored after re-parse) */
      if(any_vla){
        if(ty.kind==3){ fail(c,"a variable-length array of function pointers is not supported"); return; }   /* the oracle's */
        if(ty.kind!=0 || ty.is_float){ fail(c,"only an integer-element VLA is supported"); return; }
        if(is_static || is(c,"=")){ fail(c,"a VLA cannot have static storage or an initializer"); return; }
        if(dim_nd>3){ fail(c,"a variable-length array of more than 3 dimensions is not supported"); return; }
      }
      if(any_vla && dim_nd==1){
        /* a 1-D stack VLA `T a[n]` (the C twin of the oracle's Decl VLA branch). The runtime size is
         * evaluated ONCE; snapshot it into an immutable hidden extent `__bcir_extK`, register the array
         * NAMED but with is_vla (so it is declared IN-BODY by `c.vladecl`, not up front -- its size isn't
         * known until execution reaches the decl), and bind ptr_extent so `a[i]` masks against the snapshot. */
        c->i=dim_tok[0]; uint32_t vla_n=p_expr(c); c->i=after_dims;   /* lower the dim ONCE, in C order */
        const bcir_resource *nr=res_of(c->fn,vla_n);    /* the size must be an integer scalar */
        if(!nr || nr->kind!=BCIR_RK_SCALAR || nr->is_float){ fail(c,"a VLA size must be an integer expression"); return; }
        int nbytes=(int)nr->elem_bytes, nsgn=nr->is_signed;   /* capture before add_res may realloc res[] */
        char en[BCIR_CIR_NAME]; snprintf(en,sizeof en,"__bcir_ext%d",c->ext_ctr++);
        uint32_t ext=add_res(c,BCIR_DOM_RAM,nbytes,1,0,BCIR_RK_SCALAR,en);   /* the snapshot: immutable extent */
        if(c->fn->n_res) c->fn->res[c->fn->n_res-1].is_signed=(uint8_t)nsgn;
        { bcir_claim *cp=new_claim(c,"c.copy",BCIR_OP_ADD); if(cp){cp->n_rd=1;cp->rd[0]=vla_n;cp->n_wr=1;cp->wr[0]=ext;} }
        int vmm=ty_mmio(c,&ty,si,1);                  /* volatile elements: a device region, as a sized local array */
        uint32_t arid=add_res(c,vmm?BCIR_DOM_MMIO:BCIR_DOM_RAM,ty.size,1,ty.is_volatile,BCIR_RK_SCALAR,nb);   /* the array (element type, count 0->1) */
        { bcir_resource *ar=&c->fn->res[c->fn->n_res-1];
          ar->is_signed=(uint8_t)(ty.signd?1:0); ar->is_vla=1; ar->ext_var=ext;
          if(ty.is_bool) ar->is_bool=1; if(ty.is_plain_char) ar->is_plain_char=1; }
        { bcir_claim *vd=new_claim(c,"c.vladecl",BCIR_OP_ADD); if(vd){vd->n_rd=1;vd->rd[0]=ext;vd->n_wr=1;vd->wr[0]=arid;
            if(vmm){ vd->domain=BCIR_DOM_MMIO; vd->lane=BCIR_LANE_H; vd->hazard=BCIR_HZ_BARRIERED; } } }   /* a device claim */
        env_add(c,&nm,arid,&ty,si);   /* the venv type is the element type -- `a[i]` indexes via emit_index */
        ptrext_set(c,c->fn,arid,ext);   /* §5.12 mask `a[i]` against the recovered runtime extent */
        if(is(c,",")){ c->i++; continue; }
        break;
      }
      if(any_vla){
        /* a MULTI-dim stack VLA `T a[d0][d1]...` (the C twin of the oracle's vla_dims Decl branch). Snapshot
         * each dim ONCE (canonical order: dims 0..k-1) into a named `__bcir_extK` (runtime -> evaluate + copy;
         * literal -> a const temp + copy), compute the total = product into a final `__bcir_extK`, declare the
         * array IN-BODY as a FLAT `T a[__ext_total];` (same row-major layout as `T a[d0][d1]`), and record the
         * per-dim snapshot rids so `a[i][j]` Horner-flattens with RUNTIME strides masked against the total. */
        uint32_t dim_exts[3]; int nbytes=ty.size, nsgn=ty.signd;
        for(int d=0; d<dim_nd; d++){                  /* 1. snapshot each dim into a named __bcir_extK */
          char en[BCIR_CIR_NAME]; snprintf(en,sizeof en,"__bcir_ext%d",c->ext_ctr++);
          uint32_t ext=add_res(c,BCIR_DOM_RAM,4,1,0,BCIR_RK_SCALAR,en);   /* a uint32_t extent snapshot */
          if(dim_is_lit[d]){                          /* a literal dim -> a const temp, copied into the ext */
            uint32_t kt=temp(c,4);
            { bcir_claim *kc=new_claim(c,"c.const",BCIR_OP_LOAD); if(kc){kc->n_wr=1;kc->wr[0]=kt;kc->n_imm=1;kc->imm[0]=dim_lit[d];} }
            { bcir_claim *cp=new_claim(c,"c.copy",BCIR_OP_ADD); if(cp){cp->n_rd=1;cp->rd[0]=kt;cp->n_wr=1;cp->wr[0]=ext;} }
          } else {                                    /* a runtime dim -> evaluated once, copied in */
            c->i=dim_tok[d]; uint32_t dv=p_expr(c); c->i=after_dims;
            { bcir_claim *cp=new_claim(c,"c.copy",BCIR_OP_ADD); if(cp){cp->n_rd=1;cp->rd[0]=dv;cp->n_wr=1;cp->wr[0]=ext;} }
          }
          dim_exts[d]=ext;
        }
        uint32_t total=dim_exts[0];                   /* 2. total extent = product of all dim snapshots */
        for(int d=1; d<dim_nd; d++){
          uint32_t prod=temp(c,4);
          { bcir_claim *mc=new_claim(c,"c.bin.mul",BCIR_OP_MUL); if(mc){mc->n_rd=2;mc->rd[0]=total;mc->rd[1]=dim_exts[d];mc->n_wr=1;mc->wr[0]=prod;} }
          total=prod;
        }
        char et[BCIR_CIR_NAME]; snprintf(et,sizeof et,"__bcir_ext%d",c->ext_ctr++);
        uint32_t ext_total=add_res(c,BCIR_DOM_RAM,4,1,0,BCIR_RK_SCALAR,et);   /* a stable name for the decl + mask */
        { bcir_claim *cp=new_claim(c,"c.copy",BCIR_OP_ADD); if(cp){cp->n_rd=1;cp->rd[0]=total;cp->n_wr=1;cp->wr[0]=ext_total;} }
        int vmm=ty_mmio(c,&ty,si,1);                  /* volatile elements: a device region, as a sized local array */
        uint32_t arid=add_res(c,vmm?BCIR_DOM_MMIO:BCIR_DOM_RAM,nbytes,1,ty.is_volatile,BCIR_RK_SCALAR,nb);   /* 3. a FLAT runtime-extent array */
        { bcir_resource *ar=&c->fn->res[c->fn->n_res-1];
          ar->is_signed=(uint8_t)(nsgn?1:0); ar->is_vla=1; ar->ext_var=ext_total;
          ar->vla_ndims=(uint8_t)dim_nd; for(int d=0;d<dim_nd;d++) ar->vla_strides[d]=dim_exts[d];
          if(ty.is_bool) ar->is_bool=1; if(ty.is_plain_char) ar->is_plain_char=1; }
        { bcir_claim *vd=new_claim(c,"c.vladecl",BCIR_OP_ADD); if(vd){vd->n_rd=1;vd->rd[0]=ext_total;vd->n_wr=1;vd->wr[0]=arid;
            if(vmm){ vd->domain=BCIR_DOM_MMIO; vd->lane=BCIR_LANE_H; vd->hazard=BCIR_HZ_BARRIERED; } } }   /* a device claim */
        env_add(c,&nm,arid,&ty,si);   /* the venv type is the element type -- `a[i][j]` indexes via emit_index */
        ptrext_set(c,c->fn,arid,ext_total);   /* §5.12 mask `a[i][j]` against the recovered total runtime extent */
        if(is(c,",")){ c->i++; continue; }
        break;
      }
      if(la_nd>3){ fail(c,"local array of more than 3 dimensions"); return; }   /* adims caps at 3 */
      if(la_nd>1){ for(int z=0;z<3;z++) ty.adims[z]=la_dims[z]; ty.nadims=la_nd; }   /* multi-dim flatten */
      int sized_inf=0;                        /* a brace initializer sized the inferred `[]` below */
      int inferred = (la_nd>=1 && la_dims[0]==0);   /* an inferred-size `[]` (or `[][B]`) array: the count
        * comes from the initializer (patched after the init, like p_array_literal). One row for now. */
      if(inferred && !is(c,"=")){ fail(c,"an array of unknown size needs an initializer"); return; }   /* `T a[];`
        * has no size and nothing to count (6.7.9p22) -- it was a zero-length array */
      int is_arr = (arr || inferred);         /* a (possibly inferred-size) ARRAY declarator -> a SCALAR-kind
        * array resource (count>1; a struct-element array carries the struct tag in `agg` for the decl) */
      int rk=is_arr?BCIR_RK_SCALAR:(ty.kind==2?BCIR_RK_POINTER:ty.kind==1?BCIR_RK_AGGREGATE:BCIR_RK_SCALAR);
      int arr_elem = (is_arr && ty.kind==2) ? cc_abi(c)->pointer_size : ty.size;   /* an array of pointers: pointer-wide elements */
      uint32_t rid=add_res(c, ty_mmio(c,&ty,si,is_arr)?BCIR_DOM_MMIO:BCIR_DOM_RAM,
                           is_arr?arr_elem:(ty.kind==2?ty.size:(ty.kind==1?c->s[si].size:ty.size)),
                           is_arr?(arr?arr:1):(ty.kind==2?(1<<16):1), ty.is_volatile, rk, nb);
      if(is_arr && c->fn->n_res) c->fn->res[c->fn->n_res-1].is_array=1;   /* an array object, whatever its length */
      if(c->fn->n_res) c->fn->res[c->fn->n_res-1].is_atomic=(uint8_t)(ty.is_atomic?1:0);   /* `_Atomic` storage: the
        * object, its elements, or a pointer's pointee (CF-ATOMIC) -- declared so, and accessed atomically */
      if(is_arr && ty.kind==1){ bcir_resource *ar=&c->fn->res[c->fn->n_res-1];   /* an ARRAY-OF-STRUCTS local
        * `struct P a[N]`: a SCALAR-kind array of struct-sized elements; carry the struct tag so the decl
        * emits `struct P a[N]` and `a[i].field` strides by the element struct (the venv keeps `si`). */
        fits(c,ar->agg,BCIR_CIR_AGG,"%s %s",ty.is_union?"union":"struct",ty.tag); }
      else if(arr && ty.kind==2){ bcir_resource *ar=&c->fn->res[c->fn->n_res-1];   /* an array of pointers `T *a[N]`: a
        * SCALAR array of pointer-wide elements; the decl + `a[i]` load/store carry the pointee (void* for now) */
        ar->ptr_depth=ty.ptr_depth?ty.ptr_depth:1;
        ar->ptee_bytes=(uint32_t)(ty.size>0?ty.size:0); ar->ptee_signed=(uint8_t)(ty.signd?1:0);   /* the decl's `T` */
        ar->ptee_float=(uint8_t)(ty.is_float?1:0); ar->ptee_plain_char=(uint8_t)(ty.is_plain_char?1:0);
        if(ty.ptr_to_struct) fits(c,ar->agg,BCIR_CIR_AGG,"%s %s",ty.is_union?"union":"struct",ty.tag);
        else if(ty.ptr_to_fp) fits(c,ar->agg,BCIR_CIR_AGG,"%s",ty.tag);   /* `op_t *a[N]`: spelled by the alias */
        else if(ty.size==0 && !ty.is_float) ar->is_voidptr=1; }
      else if(ty.kind==2&&!arr){ bcir_resource *pr=&c->fn->res[c->fn->n_res-1];   /* a pointer local: carry the
        * pointee type (elem_bytes already = pointee size) so the decl emits `T *p`, not a truncating uint32 */
        pr->is_signed=(uint8_t)(ty.signd?1:0); pr->is_float=(uint8_t)(ty.is_float?1:0); pr->ptr_depth=ty.ptr_depth;
        pr->is_plain_char=(uint8_t)(ty.is_plain_char?1:0);   /* a `char *` pointee: the deref load emits `char` */
        pr->is_bool=(uint8_t)(ty.is_bool?1:0);   /* a `_Bool *` pointee: its element is `_Bool` (CF-ATOMIC) */
        if(ty.ptr_to_struct) fits(c,pr->agg,BCIR_CIR_AGG,"%s %s",ty.is_union?"union":"struct",ty.tag);
        else if(ty.ptr_to_fp) ptee_fp(c,pr,&ty);   /* `op_t *p`: a pointer to function pointers (CF-FPTAB) */
        else if(ty.size==0 && !ty.is_float) pr->is_voidptr=1; }   /* a `void *` local (void pointee) -> emit `void *` */
      else if(ty.is_valist) c->fn->res[c->fn->n_res-1].is_valist=1;     /* a `va_list ap;` local -> emit `va_list` */
      else if(ty.is_float){ c->fn->res[c->fn->n_res-1].is_float=1;      /* a float/double (element) local */
        if(ty.is_complex) c->fn->res[c->fn->n_res-1].is_complex=1; }    /* a _Complex local (a float pair) */
      else if(ty.kind==0){ c->fn->res[c->fn->n_res-1].is_signed=(uint8_t)(ty.signd?1:0);  /* (element) signedness */
        if(ty.is_bool) c->fn->res[c->fn->n_res-1].is_bool=1;       /* a _Bool local: emit `_Bool`, store normalizes */
        if(ty.bit_width>0) c->fn->res[c->fn->n_res-1].bit_width=ty.bit_width;   /* a C23 `_BitInt(N)` local */
        if(ty.is_plain_char) c->fn->res[c->fn->n_res-1].is_plain_char=1; }   /* a plain `char` local: emit `char` */
      else if(ty.kind==3 && is_arr){ bcir_resource *ar=&c->fn->res[c->fn->n_res-1];   /* a TABLE of function pointers
        * `op_t ops[N]`: declared by its alias, pointer-wide elements, each read a function-pointer value (CF-FPTAB) */
        ptee_fp(c,ar,&ty); }
      else if(ty.kind==3 && !is_arr){ bcir_resource *fr=&c->fn->res[c->fn->n_res-1];   /* a typedef'd function-pointer
        * local `op_t f`: declared by its alias, as the inline declarator's `__bcir_fpN` is -- not the pointer-wide
        * integer, which a call through it cannot compile */
        fr->is_funcptr=1; fits(c,fr->agg,BCIR_CIR_AGG,"%s",ty.tag); }
      if(ty.kind==1&&!is_arr) fits(c,c->fn->res[c->fn->n_res-1].agg,BCIR_CIR_AGG,"%s %s",ty.is_union?"union":"struct",ty.tag);   /* L8 aggregate local (a NON-array struct/union; the array form set its agg above) */
      env_add(c,&nm,rid,&ty,si);   /* the venv type is the element type -- `a[i]` indexes via emit_index */
      char *stext=NULL;          /* a static's rendered image (CF-STATICTAB) */
      if(is(c,"=") && (is_static || is_arr || (ty.kind==1 && tok_is(tat(c,c->i+1),"{")))){ c->i++;
        /* the C11 initializer walk (the C twin of the oracle's `_agg_init`) -- brace elision, designators, a string
         * literal for a character array -- over an array or a braced struct/union, or, in constant mode, over a static
         * of any type: its image is its declaration's initializer, never a store at a call (`init_image`). An
         * INFERRED `[]` (or outer `[][B]`) takes its count from the rows the walk reached; then the resource extent
         * is patched and the stores re-masked, as p_array_literal does. */
        if(is_arr && !is(c,"{") && !isk(c,T_STR)){ fail(c,"an array is initialized by a brace list or a string literal"); return; }
        int ari=-1; for(size_t i=0;i<c->fn->n_res;i++) if(c->fn->res[i].rid==rid){ ari=(int)i; break; }
        size_t s_nclaims=c->fn->n_claims;                 /* the init stores begin here -- re-mask after sizing */
        int dims[3]={0,0,0}, nd=la_nd>1?la_nd:1, inner=1;
        if(la_nd>1){ for(int z=0;z<3;z++) dims[z]=la_dims[z]; } else dims[0]=arr;
        for(int d=1; d<nd; d++) inner*=dims[d];
        if(inferred) dims[0]=1;                           /* provisional: the walk counts the rows */
        iunit obj = is_arr ? iunit_array(&ty,si,arr_elem,dims,nd) : ty.kind==1 ? iunit_struct(c,si) : iunit_scalar(c,&ty);
        int rows = is_static ? init_image(c,&obj,inferred,&stext) : init_object(c,rid,&obj,inferred);
        if(c->failed) return;
        if(inferred && ari>=0){ int n=rows<1?1:rows;
          c->fn->res[ari].count=(uint32_t)(n*inner); sized_inf=1; init_remask(c,rid,s_nclaims);
          if(la_nd>1){ venv *dv=lookup(c,&nm); if(dv) dv->type.adims[0]=n; } } }   /* `m[][B]`: its row count */
      if(is_static) add_static(c,&nm,rid,stext,is_thread);   /* static storage: its image baked into the decl, the one place it
                                                   * is initialized */
      else if(is(c,"=")){c->i++;
        if(is(c,"{")){                                    /* a braced scalar `T x = {e}` is `T x = e`, and `{}` is
          * zero (6.7.9p11) -- the oracle's `_braced_scalar` */
          c->i++; uint32_t v=0;
          if(is(c,"}")){ v=temp(c,4);
            bcir_claim *k=new_claim(c,"c.const",BCIR_OP_LOAD); if(k){k->n_wr=1;k->wr[0]=v;k->n_imm=1;k->imm[0]=0;} }
          else if(is(c,".")||is(c,"[")||is(c,"{")){ fail(c,"a braced scalar initializer holds one expression"); return; }
          else { v=p_expr(c); if(is(c,",")) c->i++; }
          if(c->failed) return;
          if(!is(c,"}")){ fail(c,"a braced scalar initializer holds one expression"); return; }
          c->i++; v=null_pointer(c,v,&ty);
          if(!scalar_value_ok(c,v)) return;             /* `T x = {a}` of a struct (CF-STRUCTARITH) */
          bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD);if(cl){cl->n_rd=1;cl->rd[0]=v;cl->n_wr=1;cl->wr[0]=rid;}
          { venv *dv=lookup(c,&nm); if(dv) mark_obj_write(c,cl,dv); } }
        else { int ist=c->i; uint32_t v=null_pointer(c,p_expr(c),&ty); int ien=c->i;
          if(!agg_value_ok(c,ty.kind==1?si:-1,v,agg_initialized)) return;   /* a struct: a list or its own type */
          bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD);if(cl){cl->n_rd=1;cl->rd[0]=v;cl->n_wr=1;cl->wr[0]=rid;}
          { venv *dv=lookup(c,&nm); if(dv) mark_obj_write(c,cl,dv); }   /* `volatile T x = v;` */
          bind_extent(c,rid,res_of(c->fn,rid),&nm,ist,ien); } }   /* §5.12: `T *p = malloc(N*sizeof(T))` -> extent N */
      if(inferred && !sized_inf && c->n_unsized<(int)(sizeof c->unsized/sizeof c->unsized[0]))
        c->unsized[c->n_unsized++]=rid;   /* its placeholder count of 1 is no extent: `sizeof` refuses it */
      if(is(c,",")){ c->i++; continue; }   /* another declarator off the same specifier */
      break;
    }
    eat(c,";");return;
  }
  /* L3: store through a pointer  *p = expr  /  *p OP= expr  /  *(p + i) = expr  (write a pointee /
   * output parameter / MMIO location). Mirrors the deref-load forms in p_unary: `*p` is a store at
   * offset 0 (imm = [0, size], like a member store); `*(p + i)` is the indexed store `p[i]` (rd =
   * [ptr, idx, val], like the array store). A compound `OP=` loads first (emit_deref / emit_index). */
  if(is(c,"*")){
    int save=c->i; c->i++;
    spec_mark dm; spec_begin(c,&dm);                   /* a bare `*e;` falls through: every probe below rolls back */
    /* CF-RTVOL: a store `*(volatile T *)((char *)p + K) = v` / `OP= v` at a byte offset of a device region */
    { venv bb; field bf; spec_mark bm; spec_begin(c,&bm); int bns=c->ns, bnec=c->nec;
      if(byte_off_access(c,&bb,&bf)){
        const tok *aop=pk(c); int cmp=is_compound_op(aop);
        if(cmp || tok_is(aop,"=")){ uint32_t val;
          if(cmp && bf.is_atomic){ char ch=aop->s[0]; c->i++;   /* an `_Atomic` slot: one atomic read-modify-write, as
                                                     * an `_Atomic` member's (CF-RTFP) */
            uint32_t rhs=p_expr(c); bcir_ctype ft=field_slot(&bf);
            const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc); (void)oc;
            (void)emit_rmw(c,suf,&ft,bb.rid,0,0,1,rhs,bf.byte_off,bf.size,0,bf.is_volatile,BCIR_BND_ASSUMED);
            eat(c,";"); return; }
          if(cmp){ char ch=aop->s[0]; c->i++;        /* `OP=`: load, the operation, store */
            uint32_t cur=emit_member(c,&bb,&bf,0);
            uint32_t rhs=p_expr(c);
            const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
            val=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.bin.%s",suf);
            bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=val;} }
          else { c->i++; if(!bf.fp_sig || !fp_slot_value(c,&val)) val=p_expr(c); }
          store_member(c,&bb,&bf,val); eat(c,";"); return; }
        spec_end(c,&bm); c->ns=bns; c->nec=bnec; c->i=save+1; } }   /* a read: the expression statement lowers it */
    { plv lv; memset(&lv,0,sizeof lv);                /* `*q->a = v` / `OP= v`: its first element (CF-SPLIT2) */
      if(deref_member_array(c,&lv.b,&lv.f)){
        const tok *aop=pk(c); int cmp=is_compound_op(aop);
        if(cmp || tok_is(aop,"=")){ uint32_t val; lv.kind=1;
          if(cmp){ char ch=aop->s[0]; c->i++;
            uint32_t cur=plv_read(c,&lv); uint32_t rhs=p_expr(c);
            const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
            val=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.bin.%s",suf);
            bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=val;} }
          else { c->i++; val=p_expr(c); }
          (void)plv_write(c,&lv,val); eat(c,";"); return; }
        c->i=save+1; } }                                /* a read: the expression statement lowers it */
    venv pvsnap; venv *pv=NULL; uint32_t idx=0; int has_idx=0, ok=0;
    /* SNAPSHOT the pointer's env entry: the `*(p + i)` index and the RHS p_expr below can declare locals
     * and realloc c->env[] -- a pointer into it dangles. The store/emit helpers only READ the venv. */
    size_t s_res=c->fn->n_res,s_cl=c->fn->n_claims; uint32_t s_rid=c->rid,s_cid=c->cid,s_clc=c->cl_ctr;
    if(is(c,"(")){ c->i++;                              /* *(p) or *(p + i) */
      if(isk(c,T_ID)){ tok pid=*pk(c); venv *pvp=lookup(c,&pid); if(!pvp) pvp=use_global(c,&pid);
        if(pvp && names_object_ptr(c,pvp)){ c->i++; pvsnap=*pvp; pv=&pvsnap;   /* (else the general path: CF-FPTAB) */
          if(is(c,"[")){ uint32_t lin=index_chain(c,pv);   /* `*(q[j] ...)`: through the loaded pointer element */
            if(c->failed) return;
            if(!step_to_elem_ptr(c,pv,lin)) pv=NULL; }
          if(pv && is(c,"+")){ c->i++; idx=p_expr(c); has_idx=1; if(eat(c,")")) ok=1; }
          else if(pv && is(c,")")){ c->i++; ok=1; } } } }
    else if(isk(c,T_ID)){ tok pid=*pk(c); venv *pvp=lookup(c,&pid); if(!pvp) pvp=use_global(c,&pid);   /* *p, *g */
      if(pvp && names_object_ptr(c,pvp)){ c->i++; pvsnap=*pvp; pv=&pvsnap; ok=1; }
      else if(assign_tok_at(c,c->i+1) && deref_lvalue_refused(c,pvp,&pid)) return; }   /* `*fp = v`, `*s = v` */
    if(ok && pv && (is_compound_op(&c->t[c->i]) ||
                    (c->t[c->i].k==T_PUN && c->t[c->i].n==1 && c->t[c->i].s[0]=='='))){
      int sz = (pv->type.ptr_depth>1) ? cc_abi(c)->pointer_size : (pv->type.size?pv->type.size:4); uint32_t val;
      /* `*pp = q` through a `T**` stores a full pointer (pointer_size), not the base scalar width */
      bcir_ctype pst=pointee_slot(&pv->type);           /* `*p` stores a byte copy: C's conversion first */
      int at=index_elem_atomic(c,pv);                   /* an `_Atomic` pointee: atomic accesses (CF-ATOMIC) */
      if(at && is_compound_op(&c->t[c->i])){            /* *p OP= expr on `_Atomic`: one read-modify-write */
        char ch=c->t[c->i].s[0]; c->i++; uint32_t rhs=p_expr(c);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc); (void)oc;
        if(has_idx) (void)emit_rmw(c,suf,&pst,pv->rid,1,idx,1,rhs,0,0,0,index_elem_vol(c,pv),BCIR_BND_ASSUMED);
        else (void)emit_rmw(c,suf,&pst,pv->rid,0,0,1,rhs,0,sz,0,index_elem_vol(c,pv),BCIR_BND_ASSUMED);
        eat(c,";"); return;
      }
      if(is_compound_op(&c->t[c->i])){                  /* *p OP= expr  ->  load, bin op, store */
        char ch=c->t[c->i].s[0]; c->i++;
        uint32_t cur = has_idx ? emit_index(c,pv,idx) : emit_deref(c,pv);
        uint32_t rhs=p_expr(c);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
        uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.bin.%s",suf);
        bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
        val=tmp;
      } else { c->i++; val=p_expr(c);                   /* *p = expr: a struct pointee takes its own type */
        if(!agg_value_ok(c,index_elem_sidx(c,pv),val,agg_assigned)) return; }
      if(!has_idx) val=store_conv(c,val,&pst);
      bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
      if(cl){
        if(has_idx){ cl->n_rd=3; cl->rd[0]=pv->rid; cl->rd[1]=idx; cl->rd[2]=val; }   /* *(p+i) == p[i] */
        else { cl->n_rd=2; cl->rd[0]=pv->rid; cl->rd[1]=val; cl->n_imm=2; cl->imm[0]=0; cl->imm[1]=sz;
               if(pst.kind==0 && pst.is_bool){cl->imm[2]=1;cl->n_imm=3;} }   /* a _Bool pointee normalizes */
        cl->bounds=BCIR_BND_ASSUMED;
        mark_access(c,cl,index_elem_vol(c,pv));          /* `*p` / `*(p+i)`: the pointee */
        mark_atomic(cl,at);
      }
      eat(c,";"); return;
    }
    /* general deref-store: `**pp = v` / `**pp OP= v` / `*(<expr>) = v` -- store through a pointer RVALUE
     * (not a simple named `*p`). Mirrors the general deref-load: parse the operand, then store at *base. */
    c->fn->n_res=s_res;c->fn->n_claims=s_cl;c->rid=s_rid;c->cid=s_cid;c->cl_ctr=s_clc;   /* undo what the shortcut
                                                        * lowered (an element load, an index) before re-parsing */
    c->i=save+1;                                       /* re-parse from just after the leading `*` */
    uint32_t base=p_unary(c); const bcir_resource *br=res_of(c->fn,base);
    if(br && br->kind==BCIR_RK_POINTER && (is_compound_op(&c->t[c->i]) ||
        (c->t[c->i].k==T_PUN && c->t[c->i].n==1 && c->t[c->i].s[0]=='='))){
      int depth=br->ptr_depth?br->ptr_depth:1;
      int sz=(depth>1)?cc_abi(c)->pointer_size:(br->elem_bytes?(int)br->elem_bytes:4); uint32_t val;
      bcir_ctype pst=res_pointee_slot(br);              /* read before the value's parse can move res[] */
      int psi=depth==1 ? agg_sidx(c,br->agg) : -1;      /* a struct pointee (CF-STRUCTINIT), read before it too */
      int at=(depth==1 && br->is_atomic), bvol=(depth==1 && br->is_volatile);   /* `_Atomic` / volatile pointee */
      if(at && is_compound_op(&c->t[c->i])){           /* on `_Atomic`: one atomic read-modify-write */
        char ch=c->t[c->i].s[0]; c->i++; uint32_t rhs=p_expr(c);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc); (void)oc;
        (void)emit_rmw(c,suf,&pst,base,0,0,1,rhs,0,sz,0,bvol,BCIR_BND_ASSUMED);
        eat(c,";"); return;
      }
      if(is_compound_op(&c->t[c->i])){                 /* **pp OP= expr -> load through base, bin, store */
        char ch=c->t[c->i].s[0]; c->i++;
        uint32_t cur=emit_deref_rid(c,base); uint32_t rhs=p_expr(c);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
        uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.bin.%s",suf);
        bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
        val=tmp;
      } else { c->i++; val=p_expr(c);                  /* **pp = expr */
        if(!agg_value_ok(c,psi,val,agg_assigned)) return; }
      val=store_conv(c,val,&pst);                      /* C's conversion to the pointee's type */
      bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
      if(cl){ cl->n_rd=2; cl->rd[0]=base; cl->rd[1]=val; cl->n_imm=2; cl->imm[0]=0; cl->imm[1]=sz; cl->bounds=BCIR_BND_ASSUMED;
        if(pst.kind==0 && pst.is_bool){cl->imm[2]=1;cl->n_imm=3;}   /* a _Bool pointee normalizes (6.3.1.2) */
        const bcir_resource *bb=res_of(c->fn,base); mark_access(c,cl,depth==1 && bb && bb->is_volatile);
        mark_atomic(cl,at); }
      eat(c,";"); return;
    }
    spec_end(c,&dm); c->i=save;   /* not a deref-store -- fall through (e.g. a bare `*p;` expression statement),
                                   * its operand's claims undone, as the oracle lowers the statement once */
  }
  if(isk(c,T_ID)){tok id=*pk(c);venv *vp=lookup(c,&id); if(!vp) vp=use_global(c,&id);   /* a writable file-scope global */
    /* SNAPSHOT the env entry: every assignment form below re-enters the expression grammar (p_expr /
     * array_index / member_arr_index), which can declare locals and realloc c->env[] mid-parse -- a
     * pointer INTO that array then dangles. The store/emit helpers only READ the venv, so the by-value
     * copy is byte-identical (#532 class). */
    venv vsnap; venv *v=NULL; if(vp){ vsnap=*vp; v=&vsnap; }
    /* L8: struct member store  v.field = expr  /  v->field = expr  (only when an `=`/OP= actually follows
     * the access chain -- else `s.m` is a VALUE, e.g. the last item of a `({...})`, and falls through to the
     * expression-statement path below, exactly like a bare `a[i];` subscript). */
    if(v&&v->sidx>=0&&tat(c,c->i+1)->k==T_PUN&&(tat(c,c->i+1)->s[0]=='.'
        ||(tat(c,c->i+1)->n==2&&tat(c,c->i+1)->s[0]=='-'&&tat(c,c->i+1)->s[1]=='>'))   /* `->`, never `-=` or `--` */
        && member_is_store(c,c->i+1)){
      { int arw=tat(c,c->i+1)->n==2;                 /* `s->x = v` of a struct, `p.x = v` of a pointer (CF-FPTAB) */
        if(!member_base_ok(c,v,arw)){ fail(c,arw?CC_ARROW_NOT:CC_DOT_NOT); return; } }
      c->i+=2; tok fld=adv(c); sdef *S=&c->s[v->sidx]; int fi=-1;
      for(int k=0;k<S->nf;k++) if((int)strlen(S->f[k].name)==fld.n&&!strncmp(S->f[k].name,fld.s,fld.n)) fi=k;
      if(fi<0){fail(c,"unknown field");return;}
      field f=member_descend(c,S->f[fi]);         /* nested `o.in.v` -> one flattened-offset store */
      if(is(c,"[") && !f.arr_count && !f.is_ptr){ fail(c,CC_SUBSCRIPT_NOT); return; }   /* `o->v[1] = x` (CF-FPTAB) */
      if(f.is_ptr && is(c,".")){ fail(c,CC_DOT_NOT); return; }   /* `o.p.x = v` through a pointer member */
      if(f.is_ptr && (is(c,"->")||is(c,".")||is(c,"["))){   /* store deref-through a loaded pointer field (#fieldderef) */
        uint32_t ptr=emit_member(c,v,&f,0);       /* load the pointer field, then store through the loaded ptr */
        store_through_ptr(c,ptr,f.ptee_sidx,f); eat(c,";"); return; }
      if(f.arr_count && is(c,"[")){               /* s.arr[i] / s.m[i][j] = expr  /  OP= expr */
        uint32_t idx=member_arr_index(c,&f); uint32_t aval;
        field sub; int soa=elem_field(c,&f,&sub);   /* arr[i].field on an array-of-structs (strided store) */
        if(c->failed) return;
        const field *sf = soa ? &sub : &f;          /* the stored slot: the element FIELD, or the array element */
        if(sf->is_atomic && is_compound_op(&c->t[c->i])){   /* on `_Atomic`: one atomic read-modify-write */
          char ch=c->t[c->i].s[0]; c->i++; uint32_t rhs=p_expr(c); bcir_ctype ft=field_slot(sf);
          const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc); (void)oc;
          (void)emit_rmw(c,suf,&ft,v->rid,1,idx,1,rhs,soa?f.byte_off+sub.byte_off:f.byte_off,sf->size,soa?f.size:0,
                         sf->is_volatile||f.is_volatile||v->type.is_volatile,BCIR_BND_ASSUMED);
          eat(c,";"); return;
        }
        if(is_compound_op(&c->t[c->i])){          /* load element, bin op, store back */
          char ch=c->t[c->i].s[0]; c->i++;
          uint32_t cur=soa?emit_member_index_field(c,v,&f,idx,&sub):emit_member_index(c,v,&f,idx);
          uint32_t rhs=p_expr(c);
          const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
          uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.bin.%s",suf);
          bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
          aval=tmp;
        } else { if(!eat(c,"="))return; aval=p_expr(c);
          if(!soa && !agg_value_ok(c,f.elem_sidx,aval,agg_assigned)) return; }   /* a struct element: its own type */
        store_member_index(c,v,&f,idx,soa,sf,aval);
        eat(c,";");return;
      }
      uint32_t val;
      if(f.is_atomic && is_compound_op(&c->t[c->i])){  /* an `_Atomic` member: one atomic read-modify-write */
        char ch=c->t[c->i].s[0]; c->i++; uint32_t rhs=p_expr(c); bcir_ctype ft=field_slot(&f);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc); (void)oc;
        (void)emit_rmw(c,suf,&ft,v->rid,0,0,1,rhs,f.byte_off,f.size,0,f.is_volatile||v->type.is_volatile,BCIR_BND_ASSUMED);
        eat(c,";"); return;
      }
      if(is_compound_op(&c->t[c->i])){
        /* compound assignment to a member:  r->field OP= expr  (the set/clear-bits driver idiom; a
         * bitfield field reads via c.bf.get, a plain member via a plain load). */
        char ch=c->t[c->i].s[0]; c->i++;
        uint32_t cur=emit_member(c,v,&f,0);        /* the current field value (loaded first) */
        uint32_t rhs=p_expr(c);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
        uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.bin.%s",suf);
        bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
        val=tmp;
      } else { if(!eat(c,"="))return; val=p_expr(c); if(!agg_value_ok(c,f.sidx,val,agg_assigned)) return; }
      /* a bitfield: read the storage unit, insert the masked bits (c.bf.set), store the unit's spanned bytes
       * back; a `_Bool` member normalizes; either converts the value to the member's type first */
      if(f.bit_w) store_member_bf(c,v,&f,val); else store_member(c,v,&f,val);
      eat(c,";");return;}
    /* L3: array element store  a[idx] = expr  /  a[idx] OP= expr  (driver buffer fill / scatter). */
    if(v&&tat(c,c->i+1)->k==T_PUN&&tat(c,c->i+1)->n==1&&tat(c,c->i+1)->s[0]=='['){
      int as_start=c->i;                                /* roll-back point: `a[i]` may be a VALUE, not a store */
      size_t as_res=c->fn->n_res,as_cl=c->fn->n_claims; uint32_t as_rid=c->rid,as_cid=c->cid,as_clc=c->cl_ctr;
      c->i++;
      { field gf; if(global_md_field(c,v,&gf)){         /* g[i][j] = / OP= on a multi-dimensional global */
          uint32_t gix=member_arr_index(c,&gf); if(c->failed) return;
          if(store_member_elem_stmt(c,v,&gf,gix)){ eat(c,";"); return; }
          if(c->failed) return;
          c->fn->n_res=as_res; c->fn->n_claims=as_cl; c->rid=as_rid; c->cid=as_cid; c->cl_ctr=as_clc;
          c->i=as_start; (void)p_expr(c); eat(c,";"); return; } }
      uint32_t idx=index_chain(c,v); uint32_t val;   /* a[i] / m[i][j] (Horner) / q[j][i] (a chain) */
      if(c->failed) return;
      if(v->sidx>=0 && (is(c,".")||is(c,"->"))){        /* a[i].field on a DIRECT array-of-structs (strided store) */
        { field af; uint32_t tot; int r=aos_member_array(c,v,idx,&af,&tot);   /* a[i].m[j] = / OP= (CF-SMALL) */
          if(r<0) return;
          if(r>0){ if(store_member_elem_stmt(c,v,&af,tot)){ eat(c,";"); return; }
            if(c->failed) return;
            c->fn->n_res=as_res; c->fn->n_claims=as_cl; c->rid=as_rid; c->cid=as_cid; c->cl_ctr=as_clc;
            c->i=as_start; (void)p_expr(c); eat(c,";"); return; } }
        field sub; if(!aos_elem_field(c,v,&sub)){ if(c->failed) return;
          c->fn->n_res=as_res;c->fn->n_claims=as_cl;c->rid=as_rid;c->cid=as_cid;c->cl_ctr=as_clc;
          c->i=as_start;(void)p_expr(c);eat(c,";");return; }
        if(store_elem_field_stmt(c,v,idx,&sub)){ eat(c,";"); return; }   /* a[i].f = / OP= (strided store) */
        if(c->failed) return;
        /* `a[i].f++;` / `a[i].m.k;` -- no `=`/OP=: a VALUE (the oracle's expression statement), not a store: undo
         * the speculative index lowering and re-parse it as an expression statement */
        c->fn->n_res=as_res; c->fn->n_claims=as_cl; c->rid=as_rid; c->cid=as_cid; c->cl_ctr=as_clc;
        c->i=as_start; (void)p_expr(c); eat(c,";"); return;
      }
      if(index_elem_atomic(c,v) && is_compound_op(&c->t[c->i])){   /* `_Atomic` elements: one atomic RMW */
        char ch=c->t[c->i].s[0]; c->i++; uint32_t rhs=p_expr(c);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc); (void)oc;
        bcir_ctype et=index_elem_ctype(v);
        (void)emit_rmw(c,suf,&et,v->rid,1,idx,1,rhs,0,0,0,index_elem_vol(c,v),access_bnd(c,v->rid));
        eat(c,";"); return;
      }
      if(is_compound_op(&c->t[c->i])){
        char ch=c->t[c->i].s[0]; c->i++;                /* a[idx] OP= expr -> load, op, store */
        uint32_t cur=emit_index(c,v,idx); uint32_t rhs=p_expr(c);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
        uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.bin.%s",suf);
        bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
        val=tmp;
      } else if(c->t[c->i].k==T_PUN&&c->t[c->i].n==1&&c->t[c->i].s[0]=='='){ c->i++; val=p_expr(c);
        if(!agg_value_ok(c,index_elem_sidx(c,v),val,agg_assigned)) return; }   /* a struct element: its own type */
      else {        /* `a[i]` with no `=`/OP= is a VALUE (e.g. the last item of a `({...})`), not a store: undo
                     * the speculative index lowering and re-parse the whole thing as an expression statement. */
        c->fn->n_res=as_res; c->fn->n_claims=as_cl; c->rid=as_rid; c->cid=as_cid; c->cl_ctr=as_clc;
        c->i=as_start; (void)p_expr(c); eat(c,";"); return; }
      { bcir_ctype et=v->type;                         /* the element: a pointer of an array of pointers, or the
                                                        * pointee of a `T **` -- a null pointer constant's type */
        if(!ptr_array(c,v)){ if(et.kind==2 && (et.ptr_depth?et.ptr_depth:1)>1) et.ptr_depth--; else et=index_elem_ctype(v); }
        val=null_pointer(c,val,&et); }
      bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
      if(cl){cl->n_rd=3;cl->rd[0]=v->rid;cl->rd[1]=idx;cl->rd[2]=val;cl->bounds=access_bnd(c,v->rid);  /* §5.12 promote */
        mark_access(c,cl,index_elem_vol(c,v)); mark_atomic(cl,index_elem_atomic(c,v));}
      eat(c,";");return;}
    if(v&&tat(c,c->i+1)->k==T_PUN&&tat(c,c->i+1)->n==1&&tat(c,c->i+1)->s[0]=='='){
      tok tnm=c->t[c->i]; c->i+=2; int ist=c->i; uint32_t val=null_pointer(c,p_expr(c),&v->type); int ien=c->i;
      if(!agg_value_ok(c,obj_sidx(c,v),val,agg_assigned)) return;   /* a struct takes its own type */
      bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD);if(cl){cl->n_rd=1;cl->rd[0]=val;cl->n_wr=1;cl->wr[0]=v->rid;}
      mark_obj_write(c,cl,v);
      bind_extent(c,v->rid,res_of(c->fn,v->rid),&tnm,ist,ien);   /* §5.12: `p = malloc(N*…)` -> N */
      eat(c,";");return;}
    /* compound assignment  name OP= expr  ->  name = name OP expr  (a bin op + a copy). */
    if(v&&is_compound_op(tat(c,c->i+1))){
      char ch=tat(c,c->i+1)->s[0];
      if(v->type.kind==2 && (ch=='+'||ch=='-')){       /* pointer arithmetic: p += n / p -= n (verbatim) */
        c->i+=2; uint32_t rhs=p_expr(c);
        char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.ptr%s",ch=='+'?"add":"sub");
        bcir_claim *cl=new_claim(c,op,BCIR_OP_ADD); if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=rhs;cl->n_wr=1;cl->wr[0]=v->rid;}
        eat(c,";");return;
      }
      if(v->type.is_atomic && v->type.kind==0){         /* an `_Atomic` object: one atomic read-modify-write */
        c->i+=2; uint32_t rhs=p_expr(c);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc); (void)oc;
        (void)emit_rmw(c,suf,&v->type,v->rid,0,0,1,rhs,0,v->type.size?v->type.size:4,0,v->type.is_volatile,BCIR_BND_ASSUMED);
        eat(c,";");return;
      }
      uint32_t cur=named_read(c,v);                     /* the current value (a volatile read, first) */
      c->i+=2; uint32_t rhs=p_expr(c);
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
      uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_OP]; fits(c,op,sizeof op,"c.bin.%s",suf);
      bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
      bcir_claim *cp=new_claim(c,"c.copy",BCIR_OP_ADD); if(cp){cp->n_rd=1;cp->rd[0]=tmp;cp->n_wr=1;cp->wr[0]=v->rid;}
      mark_obj_write(c,cp,v);
      eat(c,";");return;}}
  if(p_incdec(c)){eat(c,";");return;}    /* ++i / i++ / --i / i-- as a statement */
  (void)p_expr(c);eat(c,";");
}
/* True when [start,end) is exactly an identifier followed by one or more `.`/`->` member hops,
 * modulo parentheses around the whole expression. This intentionally excludes arithmetic, casts,
 * comma expressions, calls, and indexing: those contexts apply normal integer promotion. */
static int bare_member_tokens(CC *c, int start, int end){
  while(end>start && tok_is(&c->t[end-1],";")) end--;
  for(;;){
    if(end-start<3 || !tok_is(&c->t[start],"(") || !tok_is(&c->t[end-1],")")) break;
    int depth=0, wraps=1;
    for(int i=start;i<end;i++){
      if(tok_is(&c->t[i],"(")) depth++;
      else if(tok_is(&c->t[i],")")) depth--;
      if(depth==0 && i<end-1){wraps=0;break;}
    }
    if(!wraps || depth!=0) break;
    start++; end--;
  }
  if(start>=end || c->t[start].k!=T_ID) return 0;
  int i=start+1, hops=0;
  while(i<end && (tok_is(&c->t[i],".")||tok_is(&c->t[i],"->"))){
    if(i+1>=end || c->t[i+1].k!=T_ID) return 0;
    hops++; i+=2;
  }
  return hops>0 && i==end;
}

/* `({ s1; ...; e; })` -- a GCC statement expression (the C twin of cast.StmtExpr): a compound statement
 * in its own scope whose VALUE is the last statement (an expression statement). No AST, so the prefix
 * statements lower in place (p_stmt) and the LAST statement is then rolled back -- the speculative
 * undo p_typeof_expr uses (the resource/claim arrays + the rid/cid/cl_ctr/call/env/string counters) --
 * and re-parsed as an expression to capture its value. The cursor is at the opening `(`. */
static uint32_t p_stmt_expr(CC *c){
  c->i++; c->i++;                        /* consume `(` then `{` */
  int env_mark=c->nenv;                  /* a statement expression is a scope: its locals do not leak */
  int last_save=-1;
  int last_end=-1;
  size_t snap_res=c->fn->n_res, snap_cl=c->fn->n_claims;
  uint32_t snap_rid=c->rid, snap_cid=c->cid, snap_clctr=c->cl_ctr;
  int snap_ncalls=c->fn->n_calls, snap_nenv=c->nenv;
  int snap_nstr=c->fn->n_host_literals;
  while(!is(c,"}")&&!isk(c,T_END)&&!c->failed){
    last_save=c->i;                      /* remember the start + the lowering state before each statement */
    snap_res=c->fn->n_res; snap_cl=c->fn->n_claims; snap_rid=c->rid; snap_cid=c->cid; snap_clctr=c->cl_ctr;
    snap_ncalls=c->fn->n_calls; snap_nenv=c->nenv;
    snap_nstr=c->fn->n_host_literals;
    p_stmt(c); last_end=c->i;
  }
  /* is the LAST statement a VALUE expression statement (so the `({...})` yields it), or a statement-form
   * (an `if`/loop/`{`/label/declaration) -> a VOID statement expression (used in a discarded context)? */
  int is_value = last_save>=0;
  if(last_save>=0){ const tok *lt=&c->t[last_save];
    if(tok_is(lt,"if")||tok_is(lt,"for")||tok_is(lt,"while")||tok_is(lt,"do")||tok_is(lt,"switch")
       ||tok_is(lt,"return")||tok_is(lt,"break")||tok_is(lt,"continue")||tok_is(lt,"goto")||tok_is(lt,"{")) is_value=0;
    else if(lt->k==T_ID && c->t[last_save+1].k==T_PUN && c->t[last_save+1].n==1 && c->t[last_save+1].s[0]==':') is_value=0;
    else if(lt->k==T_ID && (scalar_size(lt->s,lt->n)>=0||tok_is(lt,"struct")||tok_is(lt,"union")||tok_is(lt,"enum")
            ||tok_is(lt,"const")||tok_is(lt,"volatile")||tok_is(lt,"_Atomic")||tok_is(lt,"static")||tok_is(lt,"_BitInt")
            ||find_typedef(c,lt->s,lt->n)>=0)) is_value=0; }
  uint32_t result;
  if(!is_value){ result=void_temp(c); }   /* a void / empty statement expression: the last stmt (if any) is
                                           * already lowered; the value is unused (an unreferenced placeholder) */
  else {                                  /* a value: roll the LAST statement back and re-parse it as the expr */
    c->fn->n_res=snap_res; c->fn->n_claims=snap_cl; c->rid=snap_rid; c->cid=snap_cid; c->cl_ctr=snap_clctr;
    c->fn->n_calls=snap_ncalls; c->nenv=snap_nenv;
    while(c->fn->n_host_literals>snap_nstr){
      c->fn->n_host_literals--;
      bcir_host_deallocate(&c->allocator,
                           c->fn->host_literals[c->fn->n_host_literals].spelling);
      c->fn->host_literals[c->fn->n_host_literals].spelling=NULL;
    }
    c->i=last_save;
    if(name_assign_ahead(c)){ fail(c,"assignment as a statement-expression value"); return temp(c,4); }
    /* ^ a BARE assignment terminal `({ a=b; })` falls back (matches the oracle): the `i++`/`++i` desugar
     *   shares the assignment AST, so its post/pre value would be guessed wrong. (A nested `({ (a=b)+1; })`
     *   is NOT a bare assignment and re-parses fine.) */
    /* A BARE inc/dec terminal `({ a++; })` / `({ ++a; })` ALSO falls back: in STATEMENT position the
     * oracle desugars `a++;` to an Assign (shedding the post/pre distinction), so a stmt-expr whose last
     * item is a bare inc/dec is an `ExprStmt(Assign)` -> the oracle's "assignment as a stmt-expr value"
     * fallback. A nested `({ (a++) + 1; })` is NOT bare (an IncDec inside a Binary) and re-parses fine. */
    { int k=c->i, bare=0;
      if((tok_is(&c->t[k],"++")||tok_is(&c->t[k],"--")) && c->t[k+1].k==T_ID
         && (tok_is(&c->t[k+2],";")||tok_is(&c->t[k+2],"}"))) bare=1;             /* ++a; / --a; */
      else if(c->t[k].k==T_ID && (tok_is(&c->t[k+1],"++")||tok_is(&c->t[k+1],"--"))
              && (tok_is(&c->t[k+2],";")||tok_is(&c->t[k+2],"}"))) bare=1;        /* a++; / a-- ; */
      if(bare){ fail(c,"assignment as a statement-expression value"); return temp(c,4); } }
    int saved_declared_bf=c->stmt_expr_declared_bf;
    c->stmt_expr_declared_bf=bare_member_tokens(c,last_save,last_end);
    result=p_expr(c); if(is(c,";")) c->i++;
    c->stmt_expr_declared_bf=saved_declared_bf;
  }
  eat(c,"}"); c->nenv=env_mark; eat(c,")");   /* pop the scope, close `)` */
  return result;
}

/* `T (*f(P))(Q)`: the declarator of a function that returns a function pointer, which neither rail parses -- the same
 * function with a typedef for its return type lowers (CF-FPRET). The oracle's `FN_RET_FP`. */
#define CC_FN_RET_FP "a function returning a function pointer is not supported without a typedef"
static int fn_ret_fp_at(CC *c){
  if(!is(c,"(") || !tok_is(tat(c,c->i+1),"*")) return 0;
  int k=skip_stars(c,c->i+1);
  return tat(c,k)->k==T_ID && tok_is(tat(c,k+1),"(");
}
static int p_func(CC *c, bcir_func *fn) {
  c->fn=fn; c->nenv=0; c->n_vlaext=0; c->n_unsized=0; c->npure=0;
  c->cl_ctr=0;                                     /* compound literals number per function (`_cl1` ...), as
                                                    * the oracle's _FuncLowerer.cl_ctr does -- they are
                                                    * function-local declarations, so the names never clash */
  c->saw_static=0;                                 /* fresh for THIS definition's return type (a prior
                                                    * body's block-static must not leak into the flag) */
  bcir_ctype rt;int rsi;if(p_type(c,&rt,&rsi))return 1; fn->ret=rt;
  if(fn_ret_fp_at(c)){ fail(c,CC_FN_RET_FP); return 1; }   /* `T (*f(P))(Q)` (CF-FPRET) */
  fn->static_fn=(uint8_t)(c->saw_static!=0);       /* source `static` on the definition (linkable emit) */
  tok nm=adv(c); fits(c,fn->name,sizeof fn->name,"%.*s",nm.n,nm.s);
  if(!eat(c,"("))return 1;
  int unnamed=0;                                   /* a parameter left unnamed: a prototype's (`T g(T *, T);`) */
  if(!is(c,")")) for(;;){
    if(is(c,"void")&&tat(c,c->i+1)->n==1&&tat(c,c->i+1)->s[0]==')'){c->i++;break;}
    if(is(c,"...")){fn->variadic=1;c->i++;break;}   /* a trailing `...` -- the function is variadic */
    bcir_ctype ty;int si;if(p_type(c,&ty,&si))return 1;
    int btdd[3], btd=td_dims_of(c,btdd);   /* a typedef'd array param */
    tok pn; int row_ptr=0;
    /* a direct function-pointer parameter `RET (*name)(PARAMS)`: unlike a typedef'd funcptr param there
     * is no alias to print, so capture the full signature as a synthesized prelude typedef `__bcir_fpN`
     * and type the param kind-3 with that tag. The indirect-call dispatch (p_icall) + the param/emit path
     * then reuse the typedef-funcptr machinery verbatim (ctype_str prints the tag). Scalar ret + params. */
    { int k=fp_decl_at(c,1);
      if(k && k!=c->i+2 && !(k==c->i+3 && tat(c,c->i+2)->k==T_ID)){   /* `RET (**pp)(P)`, `RET (*t[])(P)` (CF-FPTAB): a
        * pointer to function pointers -- an array parameter is a pointer to its element -- parsed whole */
        bcir_ctype fty; int stars, nd, dims[3];
        if(fp_inline_decl(c,&ty,si,&fty,&stars,dims,&nd,&pn)) return 1;
        ty=fty; for(int s=0;s<stars+(nd?1:0);s++) fp_star(c,&ty);
        for(int d=0; d<nd; d++) ty.adims[d]=dims[d];   /* its dims, as a typedef'd table's are kept: a 2-D one */
        ty.nadims=(uint8_t)nd;                         /* indexes its rows flat, as `T m[A][B]` does */
        if(!pn.n) unnamed=1;
        row_ptr=1; } }
    int fp_abstract = !row_ptr && is(c,"(") && tok_is(tat(c,c->i+1),"*") && tok_is(tat(c,c->i+2),")")
                      && tok_is(tat(c,c->i+3),"(");   /* `RET (*)(PARAMS)`: a prototype's, unnamed */
    if(fp_abstract || (!row_ptr && is(c,"(") && tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='*'
       && tat(c,c->i+2)->k==T_ID
       && tat(c,c->i+3)->k==T_PUN && tat(c,c->i+3)->n==1 && tat(c,c->i+3)->s[0]==')'
       && tat(c,c->i+4)->k==T_PUN && tat(c,c->i+4)->n==1 && tat(c,c->i+4)->s[0]=='(')){
      bcir_ctype ret=ty;                            /* the already-parsed return type */
      c->i+=2;                                       /* `( *` then the parameter NAME, if any */
      if(fp_abstract){ memset(&pn,0,sizeof pn); pn.s=""; unnamed=1; } else pn=adv(c);
      if(!eat(c,")")||!eat(c,"("))return 1;          /* `) (` -- into the parameter-type list */
      char rets[BCIR_EMIT_TYPE]; ctype_qstr(&ret,rets,sizeof rets);   /* the growable prelude, as for a local */
      size_t line=c->fpdefs.w; const bcir_ctype *ps; int np;
      ctext_putf(c,&c->fpdefs,"typedef %s (*__bcir_fp%d)(",rets,c->n_fpdef);
      int va;
      if(fp_param_list(c,1,&ps,&np,&va)){ c->fpdefs.w=line; if(c->fpdefs.s) c->fpdefs.s[line]=0; return 1; }
      if(!eat(c,")"))return 1;                        /* past the parameter-type list */
      memset(&ty,0,sizeof ty); ty.kind=3; ty.size=cc_abi(c)->pointer_size; ty.signd=0;
      fp_capture_ret(&ty,&ret,si);                     /* its RETURN type types a c.call.indirect result */
      snprintf(ty.tag,sizeof ty.tag,"__bcir_fp%d",c->n_fpdef); c->n_fpdef++;
      ty.fp_sig=sig_addv(c,&ret,ps,np,ty.tag,"",va);  /* ... and its parameters, a null pointer argument */
      row_ptr=1;                                      /* skip the row-ptr + array-suffix handling below */
    }
    if(!row_ptr && is(c,"(")){    /* (*name)[N]... -- a pointer-to-array "row pointer" (vendor headers); */
      int save=c->i; c->i++; int inner=0;          /* modeled as the equivalent multi-dim array param */
      while(is(c,"*")){inner++;c->i++;            /* its own qualifiers: the parameter's top level (CF-QUALS) */
        int cst, rst; if(star_quals(c,&cst,&rst)) return 1;}
      if(inner==1 && isk(c,T_ID)){ tok cand=adv(c);
        if(is(c,")") && tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='['){
          c->i++;                                  /* consume ) ; the next token is [ */
          pn=cand; int nd=1; ty.adims[0]=0;        /* the outer (pointer) dim is unspecified */
          while(is(c,"[")){ c->i++; long long d=isk(c,T_INT)?(long long)adv(c).v:0;
            if(nd<3)ty.adims[nd]=(int)d; nd++; eat(c,"]"); }
          ty.nadims=nd<3?nd:3; if(ty.kind==0) ty.kind=2;
          else if(ty.kind==1){ ty.kind=2; ty.ptr_to_struct=1; ty.ptr_depth=1; }   /* rows of structs (CF-STRUCTVAL) */
          row_ptr=1;
        } else c->i=save;
      } else c->i=save;
    }
    int vla_have=0; tok vla_tok; memset(&vla_tok,0,sizeof vla_tok);   /* §5.12 a VLA-param extent `a[n]` */
    if(!row_ptr){
    if(is(c,",") || is(c,")") || is(c,"[")){ memset(&pn,0,sizeof pn); pn.s=""; unnamed=1; }   /* no name */
    else pn=adv(c);
    if(is(c,"[") || btd){      /* an array parameter `T name[A][B]...` decays to a flat element ptr */
      int nd=0;
      while(is(c,"[")){ c->i++;
        long long d=is(c,"]") ? 0 : ce_dim(c,1);       /* a static dim `[A]`, `[N + 1]` -- the byte count is recorded */
        if(c->failed) return 1;
        /* §5.12 a VLA-param extent `[n]`: `n` must be a BARE identifier naming a PRIOR in-scope param (source
         * order -- a later param is not yet in env). Capture it for the post-scan stability gate. */
        if(d<0){ d=0;
          if(isk(c,T_ID) && tok_is(tat(c,c->i+1),"]")){
            tok cand=*pk(c);
            if(lookup(c,&cand)){ vla_tok=cand; vla_have=1; adv(c); }
          } }
        if(nd<3)ty.adims[nd]=(int)d; nd++; eat(c,"]"); }   /* a non-int/non-id dim -> 0 today (fallback, no bind) */
      for(int d=0; d<btd; d++){ if(nd<3)ty.adims[nd]=btdd[d]; nd++; }   /* a typedef'd array's dims follow */
      if(nd>3){ fail(c,"an array parameter of more than 3 dimensions"); return 1; }
      ty.nadims=nd<3?nd:3; if(ty.kind==0) ty.kind=2;     /* T[..] -> T* (element size kept in ty.size) */
      else if(ty.kind==1){ ty.kind=2; ty.ptr_to_struct=1; ty.ptr_depth=1; }   /* `struct T a[..]` -> `struct T *`
                                                          * (CF-STRUCTVAL): it had stayed a struct by VALUE */
      else if(ty.kind==2) ty.ptr_depth=(uint8_t)((ty.ptr_depth?ty.ptr_depth:1)+1);   /* T *a[..] -> T ** (`argv`):
                                                          * the decay is one more level, as for a scalar element */
      else if(ty.kind==3) fp_star(c,&ty);               /* `op_t t[N]` -> `op_t *t`: a pointer to function pointers,
                                                          * as the inline `RET (*t[N])(P)` is (CF-FPTAB) -- it had
                                                          * stayed ONE function pointer, an emit no compiler takes */
    }
    }
    char pb[BCIR_CIR_NAME]; idcpy(c,pb,&pn);
    int rk=ty.kind==2?BCIR_RK_POINTER:ty.kind==1?BCIR_RK_AGGREGATE:BCIR_RK_SCALAR;
    uint32_t rid=add_res(c, ty_mmio(c,&ty,si,0)?BCIR_DOM_MMIO:BCIR_DOM_RAM,
                         ty.kind==2?ty.size:(ty.kind==1?c->s[si].size:ty.size),
                         ty.kind==2?(1<<16):1, ty.is_volatile, rk, pb);
    if(c->fn->n_res) c->fn->res[c->fn->n_res-1].is_atomic=(uint8_t)(ty.is_atomic?1:0);   /* `_Atomic` (CF-ATOMIC) */
    if(ty.kind==2){ bcir_resource *pr=&c->fn->res[c->fn->n_res-1];   /* a pointer param: carry the pointee
      * (width/sign/tag/depth) so pointer arithmetic on it (`p + i`) clones the real `T *` type, not uint32 */
      pr->is_signed=(uint8_t)(ty.signd?1:0); pr->is_float=(uint8_t)(ty.is_float?1:0); pr->ptr_depth=ty.ptr_depth;
      pr->is_plain_char=(uint8_t)(ty.is_plain_char?1:0);   /* a `char *` pointee: the deref load emits `char` */
      pr->is_bool=(uint8_t)(ty.is_bool?1:0);   /* a `_Bool *` pointee: its element is `_Bool` (a typed store
                                                * through `_Atomic _Bool *` converts, never a byte copy) */
      if(ty.ptr_to_struct) fits(c,pr->agg,BCIR_CIR_AGG,"%s %s",ty.is_union?"union":"struct",ty.tag);
      else if(ty.ptr_to_fp) ptee_fp(c,pr,&ty); }   /* `op_t *ops`: a pointer to function pointers (CF-FPTAB) */
    else if(ty.is_float){ c->fn->res[c->fn->n_res-1].is_float=1;       /* a float/double parameter */
      if(ty.is_complex) c->fn->res[c->fn->n_res-1].is_complex=1; }     /* a _Complex parameter (a float pair) */
    else if(ty.kind==0){ c->fn->res[c->fn->n_res-1].is_signed=(uint8_t)(ty.signd?1:0);  /* signedness */
      if(ty.is_bool) c->fn->res[c->fn->n_res-1].is_bool=1;       /* a _Bool parameter */
      if(ty.bit_width>0) c->fn->res[c->fn->n_res-1].bit_width=ty.bit_width;   /* a C23 `_BitInt(N)` parameter */
      if(ty.is_plain_char) c->fn->res[c->fn->n_res-1].is_plain_char=1; }   /* a plain `char` parameter */
    else if(ty.kind==3) c->fn->res[c->fn->n_res-1].is_funcptr=1;   /* a funcptr param: stored to a member directly */
    if(ty.kind==1) fits(c,c->fn->res[c->fn->n_res-1].agg,BCIR_CIR_AGG,"%s %s",ty.is_union?"union":"struct",ty.tag);
    if(vla_have && ty.kind==2){          /* §5.12 a VLA param `T a[n]` -> record a deferred extent binding to `n`,
      * resolved (stability-gated) after scan_mutations. Only if `n` is a prior INTEGER-SCALAR param. */
      venv *nv=lookup(c,&vla_tok);
      if(nv && nv->type.kind==0 && !nv->type.is_float && c->n_vlaext<16){
        c->vlaext[c->n_vlaext].ptr_rid=rid; c->vlaext[c->n_vlaext].cnt_rid=nv->rid;
        c->vlaext[c->n_vlaext].cnt_tok=vla_tok; c->n_vlaext++;
      }
    }
    env_add(c,&pn,rid,&ty,si);
    CC_ENSURE(c,fn->params,fn->n_params,fn->cap_params);
    if(fn->n_params<fn->cap_params){bcir_param *pp=&fn->params[fn->n_params++]; memset(pp,0,sizeof *pp);
      idcpy(c,pp->name,&pn);pp->rid=rid;pp->type=ty;}
    if(is(c,",")){c->i++;continue;} break;
  }
  if(!eat(c,")"))return 1;
  if(unnamed && !is(c,";")){ fail(c,"a parameter of the function's definition has no name"); return 1; }
  if(is(c,";")){                       /* a PROTOTYPE `T name(params);` (Phase 3 linking): record the
    * signature for call typing + render the extern declaration -- a cross-TU callee the host LINKER
    * resolves. A same-unit definition WINS: the unit-end rewrite in bcir_cfront_compile_target turns
    * its tu-calls back into ordinary R18 edges. Returns 2 (the unit loop discards the scratch fn). */
    c->i++;
    if(!CC_ENSURE(c,c->protos,c->n_protos,c->cap_protos)) return 1;
    snprintf(c->protos[c->n_protos].name,BCIR_CIR_NAME,"%s",fn->name);
    c->protos[c->n_protos].ret=fn->ret;
    { bcir_ctype *pt=NULL; size_t pbytes;         /* its parameter types, past the scratch fn (CF-NULLARG) */
      if(fn->n_params>0){
        if(!bcir_size_mul((size_t)fn->n_params,sizeof *pt,&pbytes) ||
           !(pt=(bcir_ctype *)bcir_host_arena_allocate(&c->scratch,pbytes,_Alignof(bcir_ctype)))){
          cc_raise_oom(c); return 1; }
        for(int k=0;k<fn->n_params;k++) pt[k]=fn->params[k].type; }
      c->protos[c->n_protos].params=pt; c->protos[c->n_protos].n_params=pt?fn->n_params:0; }
    c->protos[c->n_protos].variadic=fn->variadic;  /* `T f(P, ...);`: a designator's type keeps it (CF-EXTDESIG) */
    c->n_protos++;
    /* every qualifier below a parameter's and the return's top level kept: a `const T *` or `T *const *`
     * parameter is another type than a `T *` or `T **` one, and the declaration's type would conflict (CF-QUALS) */
    char rets[BCIR_EMIT_TYPE]; ctype_qstr(&fn->ret,rets,sizeof rets);   /* the growable prelude (CF-BUF) */
    ctext_putf(c,&c->tudefs,"extern %s %s(",rets,fn->name);
    for(int k=0;k<fn->n_params;k++){ const bcir_ctype *pt=&fn->params[k].type; char ps[BCIR_EMIT_TYPE];
      ctype_qstr(pt,ps,sizeof ps);
      ctext_putf(c,&c->tudefs,"%s%s",k?", ":"",ps); }
    if(fn->variadic) ctext_putf(c,&c->tudefs,"%s...",fn->n_params?", ":"");
    ctext_putf(c,&c->tudefs,"%s);\n",(fn->n_params||fn->variadic)?"":"void");
    return 2;
  }
  if(!eat(c,"{"))return 1;
  scan_mutations(c,c->i);   /* §5.12 extent-stability pre-pass over the body (cursor is just past `{`) */
  for(int k=0;k<c->n_vlaext;k++){          /* §5.12 resolve the deferred VLA-param extent bindings: bind `a` to
    * `n` ONLY when `n` is STABLE -- unmutated in the body and not address-taken (the _bind_extent stable-Name
    * gate, evaluated now that scan_mutations has populated the mutation table). A mutated-size param stays
    * assumed_safe -- matching the oracle so the BCIR_CHK count is identical. */
    if(mut_body(c,&c->vlaext[k].cnt_tok)==0 && !mut_addr(c,&c->vlaext[k].cnt_tok))
      ptrext_set(c,c->fn,c->vlaext[k].ptr_rid,c->vlaext[k].cnt_rid);
  }
  for(int pass=0; pass<8 && unparen_body(c,c->i); pass++){}   /* CF-PAREN: `(a)[i]` -> `a[i]`, before the body parses */
  while(!is(c,"}")&&!isk(c,T_END)&&!c->failed) p_stmt(c);
  eat(c,"}");
  order_device_claims(fn);   /* R3: every claim touching a device region is device-domain and ordered */
  return c->failed;
}

/* --- verify: R1-R18 live in bcir_verify.c (the C twin of bcir/verify) ---- */
static const bcir_resource *res_of(const bcir_func *f,uint32_t rid){
  for(size_t i=0;i<f->n_res;i++) if(f->res[i].rid==rid) return &f->res[i]; return NULL;
}

/* --- faithful C emitter -------------------------------------------------- */
static const char *binop_c(const char *suf){
  struct {const char *s,*c;} M[]={{"add","+"},{"sub","-"},{"mul","*"},{"div","/"},{"mod","%"},
    {"and","&"},{"or","|"},{"xor","^"},{"shl","<<"},{"shr",">>"},{"eq","=="},{"ne","!="},
    {"lt","<"},{"gt",">"},{"le","<="},{"ge",">="},{"lor","||"},{"land","&&"},{0,0}};
  for(int i=0;M[i].s;i++) if(!strcmp(M[i].s,suf)) return M[i].c; return "+";
}
static const char *unop_c(const char *suf){return !strcmp(suf,"neg")?"-":!strcmp(suf,"bnot")?"~":"!";}
static void ctype_str(const bcir_ctype *ty,char *o,size_t n){
  if(ty->kind==3){ snprintf(o,n,"%s",ty->tag); return; }   /* funcptr: the typedef spelling */
  if(ty->is_valist){ snprintf(o,n,"va_list"); return; }    /* a `va_list` param (vprintf-style helpers) */
  if(ty->kind==0 && ty->bit_width>0){                      /* C23 `_BitInt(N)` -- a faithful, exact-width spelling */
    snprintf(o,n,"%s_BitInt(%d)",ty->signd?"":"unsigned ",ty->bit_width); return; }
  int is_struct = (ty->kind==1) || ty->ptr_to_struct;
  const char *kw = ty->is_union ? "union" : "struct";
  const char *base = is_struct || ty->ptr_to_fp ? ty->tag   /* a pointer to a function pointer: its alias */
                   : ty->is_bool ? "_Bool"
                   : ty->is_plain_char ? "char"   /* plain `char`: impl-defined sign (not int8_t -> ARM) */
                   : ty->is_complex ? (ty->size==8?"float _Complex":ty->size>16?"long double _Complex":"double _Complex")
                   : ty->is_float ? (ty->size==4?"float":ty->size>8?"long double":"double")
                   : ty->size==0 ? "void"
                   : ty->signd ? (ty->size==1?"int8_t":ty->size==2?"int16_t":ty->size==8?"int64_t":"int32_t")
                   : (ty->size==1?"uint8_t":ty->size==2?"uint16_t":ty->size==8?"uint64_t":"uint32_t");
  const char *atm = ty->is_atomic ? "_Atomic " : "";
  if(ty->kind==2){ char stars[BCIR_MAX_PTR_DEPTH+2]; int d=ty->ptr_depth?ty->ptr_depth:1, si=0;   /* depth `*`s: `T**` -> ` **` */
    stars[si++]=' '; for(int k=0;k<d;k++) stars[si++]='*'; stars[si]=0;
    snprintf(o,n,"%s%s%s%s%s%s",atm,ty->is_volatile?"volatile ":"",
             ty->ptr_to_struct?kw:"",ty->ptr_to_struct?" ":"",base,stars); }
  else if(ty->kind==1) snprintf(o,n,"%s %s",kw,ty->tag);
  else snprintf(o,n,"%s%s",atm,base);
}
/* The type `ty` as a function type spells a parameter or its return (CF-QUALS): with every qualifier below its top
 * level -- `const char *const *` -- where the emit's own objects are spelled without them (`ctype_str`) and meet it
 * through a cast (`qcast`). A scalar, a struct or a function pointer is its top level whole. The oracle's
 * `_qual_type`. */
static void ctype_qstr(const bcir_ctype *ty,char *o,size_t n){
  if(ty->kind!=2 || n==0){ ctype_str(ty,o,n); return; }
  char b[BCIR_EMIT_TYPE]; ctype_str(ty,b,sizeof b);
  size_t k=strlen(b); while(k && b[k-1]=='*') k--; while(k && b[k-1]==' ') k--; b[k]=0;   /* the type under its `*`s */
  int d=ptr_levels(ty);
  int w=snprintf(o,n,"%s%s ",ty->is_const?"const ":"",b);
  for(int lv=1; lv<=d && w>=0 && (size_t)w<n; lv++){
    int below=lv<d && lv<=8;                         /* the outermost `*` is the parameter's own level */
    int cst=below && ((ty->ptr_const>>(lv-1))&1u), rst=below && ((ty->ptr_restrict>>(lv-1))&1u);
    w+=snprintf(o+w,n-(size_t)w,"*%s%s",cst?"const ":"",rst?"restrict ":"");
  }
}
/* The qualifiers a function type keeps of a parameter or return of type `t`, below its top level (CF-QUALS): what a
 * pointer points to (`is_const`) and each `*` under the outermost -- part of the type's identity (`sig_same`), as the
 * oracle's `_qual_sig` is of its `_fn_key`. 0 when none. */
static unsigned qual_key(const bcir_ctype *t){
  if(t->kind!=2) return 0;
  int d=ptr_levels(t), lv=d-1>8 ? 8 : d-1;
  unsigned below=lv>0 ? (1u<<lv)-1u : 0u;
  return (t->is_const?1u:0u) | ((unsigned)(t->ptr_const&below)<<1) | ((unsigned)(t->ptr_restrict&below)<<9);
}
/* A unique C identifier for a named local. The lowering flattens scopes, so two source locals that
 * shared a name in disjoint scopes (e.g. `i` in two separate `for` loops, or a local shadowing a param)
 * are distinct resources with the same name; declaring both at function scope is a C redefinition. The
 * N-th occurrence of a name (params first, then resources in order) keeps the bare name for the first
 * and gets a `_N` suffix thereafter (`i`, `i_2`, ...) -- the same scheme as the oracle's emitter, used
 * for both the declaration and every reference. An unnamed temp is `t<rid>` -- or, when a declared name
 * spells that (a re-parsed emit's local, a source's parameter), the first free `t<rid>_<k>`: never a second
 * declaration of the name (CF-RTVOL, the oracle's `_Names`). */
static int declared_name(const bcir_func *f,const char *nm);
static const char *uniq_local(const bcir_func *f,uint32_t rid,char *buf){
  const bcir_resource *r=res_of(f,rid);
  if(!r||!r->name[0]){ snprintf(buf,BCIR_EMIT_NAME,"t%u",rid);
    for(int k=2; declared_name(f,buf); k++) snprintf(buf,BCIR_EMIT_NAME,"t%u_%d",rid,k);
    return buf; }
  if(r->read_only){ snprintf(buf,BCIR_EMIT_NAME,"%s",r->name); return buf; }   /* a file-scope global */
  for(int p=0;p<f->n_params;p++)                                              /* a param: keep its name */
    if(f->params[p].rid==rid){ snprintf(buf,BCIR_EMIT_NAME,"%s",r->name); return buf; }
  int occ=0;                                            /* count earlier holders of the bare name */
  for(int p=0;p<f->n_params;p++)
    if(f->params[p].name[0] && !strcmp(f->params[p].name,r->name)) occ++;
  for(size_t i=0;i<f->n_res;i++){
    const bcir_resource *q=&f->res[i];
    if(q->rid==rid){
      if(occ==0) snprintf(buf,BCIR_EMIT_NAME,"%s",r->name);
      else       snprintf(buf,BCIR_EMIT_NAME,"%s_%d",r->name,occ+1);
      return buf;
    }
    if(q->name[0] && !strcmp(q->name,r->name)){         /* an earlier same-named resource... */
      int isp=0; for(int p=0;p<f->n_params;p++) if(f->params[p].rid==q->rid){isp=1;break;}
      if(!isp) occ++;                                   /* ...that is not itself a param (counted above) */
    }
  }
  snprintf(buf,BCIR_EMIT_NAME,"%s",r->name); return buf;
}
/* `nm` is the emitted name of a declared object -- a parameter, a named local or static, a global. Only a
 * resource whose own name `nm` starts with can emit as `nm` (disambiguating appends `_N`). */
static int declared_name(const bcir_func *f,const char *nm){
  for(int p=0;p<f->n_params;p++) if(!strcmp(f->params[p].name,nm)) return 1;
  for(size_t i=0;i<f->n_res;i++){ const bcir_resource *q=&f->res[i]; size_t n=strlen(q->name); char b[BCIR_EMIT_NAME];
    if(n && !strncmp(q->name,nm,n) && !strcmp(uniq_local(f,q->rid,b),nm)) return 1; }
  return 0;
}
static const char *rname(const bcir_func *f,uint32_t rid,char *buf){
  const char *lit=strtab_lookup(f,rid);              /* a string literal -> its full spelling, inline */
  if(lit) return lit;                                /* (returned directly, so length is not capped) */
  return uniq_local(f,rid,buf);                      /* a named local (disambiguated) / a `t<rid>` temp */
}
/* The C type to declare a temporary / local with: float/double for a floating value, else the integer
 * scalar's true fixed-width type from its (width, signedness) -- so the backend does signed-vs-unsigned
 * and width-correct arithmetic (the old flat uint32 model did not). Non-scalar temps (pointer / address
 * paths) stay uint32 here; their declaration goes through the pointee type. */
/* Type-spelling scratch belongs to one emission operation.  Four slots are
 * enough for every currently emitted statement and avoid the old process-global
 * rotating buffer, which made otherwise independent contexts race. */
typedef struct bcir_emit_type_scratch {
  char bitint_ring[4][32];   /* `unsigned _BitInt(N)` for any int N: never truncated */
  unsigned next;
} bcir_emit_type_scratch;
static const char *bitint_spelling(bcir_emit_type_scratch *scratch,int bit_width,int signd){
  char *b=scratch->bitint_ring[scratch->next++&3u];
  snprintf(b,sizeof scratch->bitint_ring[0],"%s_BitInt(%d)",signd?"":"unsigned ",bit_width); return b; }
static const char *tty(bcir_emit_type_scratch *scratch,const bcir_func *f,uint32_t rid){
  const bcir_resource *r=res_of(f,rid);
  if(!r) return "uint32_t";
  if(r->is_valist) return "va_list";   /* a variadic cursor object -- opaque, declared `va_list ap;` */
  if(r->bit_width>0) return bitint_spelling(scratch,r->bit_width,r->is_signed);   /* C23 `_BitInt(N)` -- faithful spelling */
  if(r->is_complex) return r->elem_bytes==8?"float _Complex":r->elem_bytes>16?"long double _Complex":"double _Complex";
  if(r->is_float) return r->elem_bytes==4?"float":r->elem_bytes>8?"long double":"double";   /* 16/12 -> long double */
  if(r->is_bool) return "_Bool";   /* a store into a bool object normalizes any nonzero to 1 (§6.3.1.2) */
  if(r->is_plain_char) return "char";   /* plain `char`: impl-defined sign (not int8_t -> wrong on ARM) */
  if(r->kind==BCIR_RK_AGGREGATE && r->agg[0]) return r->agg;   /* a struct/union value: `struct T` (CF-STRUCTVAL) */
  if(r->kind==BCIR_RK_SCALAR && r->is_funcptr && r->agg[0] && !r->is_array) return r->agg;   /* a function-pointer
    * value -- an element of a table, a read through a pointer to one -- by its alias (CF-FPTAB) */
  if(r->kind==BCIR_RK_SCALAR) switch(r->elem_bytes){
    case 1: return r->is_signed?"int8_t":"uint8_t";
    case 2: return r->is_signed?"int16_t":"uint16_t";
    case 8: return r->is_signed?"int64_t":"uint64_t";
    default:return r->is_signed?"int32_t":"uint32_t";
  }
  return "uint32_t";
}
/* The C type to DECLARE rid with at an emit site. For a pointer resource this composes the real
 * `<pointee> *` (the pointee width/sign/float/tag ride on the resource); for everything else it is the
 * scalar `tty`. Pointer types must be composed (not static strings), so it writes into a caller buffer
 * and returns it -- byte-identical to `tty` for non-pointers, so scalar emit is unchanged. */
static const char *ptr_spelling(char *buf,size_t n,int vol,int voidp,const char *agg,int flt,uint32_t bytes,int signd,int depth,
                                int cplx,int boolp,int pchar,int atom){
  const char *base = voidp ? "void"   /* a `void *` pointee (`&&L`, void-pointee local): no width/sign */
    : agg[0] ? agg
    : cplx ? (bytes==8?"float _Complex":bytes>16?"long double _Complex":"double _Complex")
    : flt ? (bytes==4?"float":bytes>8?"long double":"double")
    : boolp ? "_Bool" : pchar ? "char"   /* the pointee's own type (the oracle's `_cname`) */
    : bytes==1?(signd?"int8_t":"uint8_t") : bytes==2?(signd?"int16_t":"uint16_t")
    : bytes==8?(signd?"int64_t":"uint64_t") : (signd?"int32_t":"uint32_t");
  char stars[BCIR_MAX_PTR_DEPTH+2]; int d=depth?depth:1, si=0;   /* depth `*`s: `T**` at depth 2 */
  if(d>BCIR_MAX_PTR_DEPTH) d=BCIR_MAX_PTR_DEPTH;
  stars[si++]=' '; for(int k=0;k<d;k++) stars[si++]='*'; stars[si]=0;
  snprintf(buf,n,"%s%s%s%s",vol?"volatile ":"",atom?"_Atomic ":"",base,stars);   /* a pointer to volatile /
                                                        * `_Atomic` storage (the oracle's `_cname`) */
  return buf;
}
static const char *decl_ty(bcir_emit_type_scratch *scratch,const bcir_func *f,uint32_t rid,char *buf,size_t n){
  const bcir_resource *r=res_of(f,rid);
  if(r && r->kind==BCIR_RK_POINTER)
    return ptr_spelling(buf,n,r->is_volatile,r->is_voidptr,r->agg,r->is_float,r->elem_bytes,r->is_signed,r->ptr_depth,
                        r->is_complex,r->is_bool,r->is_plain_char,r->is_atomic);
  snprintf(buf,n,"%s",tty(scratch,f,rid));
  return buf;
}
/* The type a function pointer `vr`, spelled `nm`, is stored through at a byte address (CF-RTFP; the oracle's
 * `_fp_store`): a pointer to its own type -- its alias, or, for a function's name, which has none, `__typeof__(&*f)`, a
 * pointer to the function whether `f` names it or holds one (C11 6.5.3.2p3). A store through a generic
 * `void (**)(void)` slot, read back as the member's type, is undefined, and GCC at -O2 dropped it. */
static const char *fp_slot_ty(const bcir_resource *vr,const char *nm,char *buf,size_t n){
  if(vr->agg[0]) snprintf(buf,n,"%s *",vr->agg); else snprintf(buf,n,"__typeof__(&*%s) *",nm);
  return buf;
}
/* A pointer to a volatile slot of C type `t`: the conventional `volatile T *`, or `T volatile *` when `t` is
 * itself a pointer type (a qualifier in front would qualify its pointee, not the slot). The oracle's
 * `_volatile_ptr`, byte-identical. */
static const char *vol_ptr(const char *t,char *buf,size_t n){
  size_t k=strlen(t); while(k && t[k-1]==' ') k--;
  if(k && t[k-1]=='*') snprintf(buf,n,"%s volatile *",t); else snprintf(buf,n,"volatile %s *",t);
  return buf;
}
/* A pointer to an `_Atomic` object of C type `t` (CF-ATOMIC) -- the oracle's `_atomic_ptr`: `_Atomic T *`,
 * `volatile` too for a device access, `T _Atomic *` when `t` is itself a pointer type. */
static const char *atomic_ptr(const char *t,int vol,char *buf,size_t n){
  size_t k=strlen(t); while(k && t[k-1]==' ') k--;
  const char *q=vol?"volatile _Atomic":"_Atomic";
  if(k && t[k-1]=='*') snprintf(buf,n,"%s %s *",t,q); else snprintf(buf,n,"%s %s *",q,t);
  return buf;
}
/* The type a volatile LOAD reads into temp `trid`: the value's own type, at the slot's width `w` for an
 * integer (a bitfield's storage unit) -- the oracle reads `et`, or the unit's unsigned type. A struct or union
 * is read whole, as itself (CF-STRUCTVAL). */
static const char *vol_load_ty(bcir_emit_type_scratch *scratch,const bcir_func *f,uint32_t trid,long long w,char *buf,size_t n){
  const bcir_resource *r=res_of(f,trid);
  if(!r || r->kind==BCIR_RK_POINTER || r->kind==BCIR_RK_AGGREGATE || r->is_float || r->is_complex || r->bit_width>0
     || r->is_bool || r->is_plain_char)
    return decl_ty(scratch,f,trid,buf,n);
  if(w==1||w==2||w==4||w==8){ snprintf(buf,n,"%sint%lld_t",r->is_signed?"":"u",w*8); return buf; }
  return decl_ty(scratch,f,trid,buf,n);
}
/* The C type of the `sz`-byte slot a volatile STORE of `vrid` writes -- the oracle's `_slot_ctype`: a `_Bool`
 * slot (`flag` 1), a float of the slot's width, a pointer or an aggregate as itself, else the unsigned integer
 * of the slot's width (a bitfield's unit, `flag` 2, included): one access of exactly the slot. */
static const char *vol_slot_ty(bcir_emit_type_scratch *scratch,const bcir_func *f,uint32_t vrid,long long sz,int flag,char *buf,size_t n){
  const bcir_resource *vr=res_of(f,vrid);
  if(flag==1) return "_Bool";
  if(vr && vr->is_complex) return sz==8?"float _Complex":sz>16?"long double _Complex":"double _Complex";
  if(vr && vr->is_float) return sz==4?"float":sz>8?"long double":"double";
  if(vr && vr->kind==BCIR_RK_POINTER) return decl_ty(scratch,f,vrid,buf,n);
  if(vr && vr->kind==BCIR_RK_AGGREGATE && vr->agg[0]) return vr->agg;
  return sz==1?"uint8_t":sz==2?"uint16_t":sz==8?"uint64_t":"uint32_t";
}
/* The `&`-or-not prefix that turns a BASE resource into a `(char *)`-castable pointer (mirrors the oracle's
 * _base_ptr): a POINTER value decays to itself (`(char *)p`), and so does an ARRAY name (a scalar resource
 * with count>1 -- a DIRECT array-of-structs `a[i].f`, `(char *)a`); a struct/scalar VALUE is addressed
 * (`(char *)&s`). NOTE: the addrof path keeps its own `&`-for-array rule (it addresses `&arr[i]`). */
/* A base whose value IS the address an access goes through: a pointer resource, or a file-scope `T *g`
 * whose slot is modelled as a scalar (`is_pointer`) -- `(char *)g`, never `(char *)&g`, which would address
 * the pointer variable itself. The one predicate every base-address spelling below uses. */
static int holds_pointer(const bcir_resource *br){ return br && (br->kind==BCIR_RK_POINTER || br->is_pointer); }
static const char *base_amp(const bcir_resource *br){
  if(holds_pointer(br)) return "";
  if(br && br->kind==BCIR_RK_SCALAR && br->count>1) return "";   /* a direct array name decays */
  return "&";
}
static int is_named_local(const bcir_func *f,uint32_t rid){
  const bcir_resource *r=res_of(f,rid); if(!r||!r->name[0]) return 0;
  if(r->read_only) return 0;                                          /* a global, defined in source */
  for(int i=0;i<f->n_params;i++) if(f->params[i].rid==rid) return 0;   /* a param, not a local */
  return 1;
}
/* a file-scope global referenced by name (defined in the source): a write to it is a bare assignment
 * `g = v;`, never a `uint32_t g = v;` declaration (the storage is external, not a fresh temp). */
static int is_global_ref(const bcir_func *f,uint32_t rid){
  const bcir_resource *r=res_of(f,rid); return r && r->name[0] && r->read_only;
}
/* a parameter, already declared in the signature: a write to it (`a = v;`) is a bare assignment too,
 * never a `uint32_t a = v;` declaration -- that would redeclare the parameter (invalid C). */
static int is_param_ref(const bcir_func *f,uint32_t rid){
  for(int i=0;i<f->n_params;i++) if(f->params[i].rid==rid) return 1; return 0;
}
static const char *guard_idx(const bcir_func *f, const bcir_claim *cl, char *buf, size_t bn, int is_write);   /* fwd */
/* The `_Atomic` lvalue an atomic access performs (CF-ATOMIC) -- the oracle's `_atomic_object` --
 * `(*(_Atomic T *)ADDR)` at the address the claim's first `naddr` reads and its imm name, exactly where the
 * plain access lands: a typed element `base[idx]` keeps its bounds guard (a write site's unless the claim is a
 * load); a member-array element or an array-of-structs field lands at base + off + idx*stride (the stride at
 * imm[stride_at], else the element size); a member, a dereference or a named object at base + off. Never a
 * byte copy: a `memcpy` of an atomic object is not an atomic access, and it tears. */
static const char *atomic_object(const bcir_func *f,const bcir_claim *cl,const char *t,int naddr,int stride_at,
                                 char *buf,size_t n){
  char pb[BCIR_EMIT_TYPE+32], a[BCIR_EMIT_NAME], b[BCIR_EMIT_NAME], gb[BCIR_EMIT_EXPR];
  atomic_ptr(t,cl->is_volatile,pb,sizeof pb);
  if(naddr==2 && !cl->n_imm){
    snprintf(buf,n,"(*(%s)&%s[%s])",pb,rname(f,cl->rd[0],a),guard_idx(f,cl,gb,sizeof gb,strcmp(cl->op,"c.load")!=0));
    return buf; }
  const char *amp=base_amp(res_of(f,cl->rd[0])); long long off=cl->n_imm?cl->imm[0]:0;
  if(naddr==2){ long long es=cl->n_imm>1?cl->imm[1]:4, stride=cl->n_imm>stride_at?cl->imm[stride_at]:es;
    snprintf(buf,n,"(*(%s)((char *)%s%s + %lld + (size_t)%s * %lld))",pb,amp,rname(f,cl->rd[0],a),off,
             rname(f,cl->rd[1],b),stride);
    return buf; }
  snprintf(buf,n,"(*(%s)((char *)%s%s + %lld))",pb,amp,rname(f,cl->rd[0],a),off);
  return buf;
}
/* The C type of an element of the pointer or array `rid` (the oracle's `_elem_ctype`): the slot an atomic
 * store into a typed element `base[idx]` writes. A pointer's pointee is its spelling without the ` *`. */
static const char *elem_spelling(bcir_emit_type_scratch *scratch,const bcir_func *f,uint32_t rid,char *buf,size_t n){
  const bcir_resource *r=res_of(f,rid);
  if(r && r->kind==BCIR_RK_POINTER && (r->ptr_depth?r->ptr_depth:1)==1){
    ptr_spelling(buf,n,0,r->is_voidptr,r->agg,r->is_float,r->elem_bytes,r->is_signed,1,r->is_complex,r->is_bool,
                 r->is_plain_char,0);
    size_t k=strlen(buf); while(k && (buf[k-1]=='*' || buf[k-1]==' ')) buf[--k]=0;
    return buf; }
  snprintf(buf,n,"%s",tty(scratch,f,rid));
  return buf;
}
/* The index expression for `base[idx]`: a MASKED (runtime-bounds-checked, §5.12) access into a known-extent
 * array is wrapped in a bounds guard -- in-bounds returns idx (behaviour-identical to the raw `a[i]`),
 * out-of-bounds calls the bounds-quarantine handler; the numeric `rid` is the access provenance and
 * `"<func>:<array>"` is the source-site handle the debugger / ML-layer reads (a site->source table realized
 * inline). Any other access -> the bare index. Result written into `buf`.
 *
 * READ vs WRITE (§5.12): a READ index site uses `BCIR_CHK` (the handler MAY clamp an OOB read to a valid
 * element -- a load mutates nothing); a WRITE index site (`is_write`) uses `BCIR_CHK_W`, whose handler is
 * `noreturn` and NEVER clamps -- a clamped OOB store would silently redirect the write onto a[extent-1] and
 * corrupt it, so an OOB store always fails-fast. This MUST mirror emit.py's `_idx(write=...)` so the two
 * rails stay parity-identical. */
static const char *guard_idx(const bcir_func *f, const bcir_claim *cl, char *buf, size_t bn, int is_write){
  const bcir_resource *br=res_of(f,cl->rd[0]);
  const char *chk = is_write ? "BCIR_CHK_W" : "BCIR_CHK";
  if(cl->bounds==BCIR_BND_MASKED){
    char ib[BCIR_EMIT_NAME], nb[BCIR_EMIT_NAME];
    uint32_t ext=ptrext_get(f,cl->rd[0]);                  /* §5.12 a naked pointer with a RECOVERED runtime extent */
    if(ext){ char eb[BCIR_EMIT_NAME];
      snprintf(buf,bn,"%s(%u, %s, %s, \"%s:%s\")",chk,(unsigned)cl->rd[0],rname(f,cl->rd[1],ib),
               rname(f,ext,eb),f->name,rname(f,cl->rd[0],nb));   /* the count VARIABLE, re-emitted by name */
      return buf;
    }
    if(br && decl_array(br)){                              /* a known-extent local/static array (constant N,
                                                            * one included) */
      snprintf(buf,bn,"%s(%u, %s, %lluu, \"%s:%s\")",chk,(unsigned)cl->rd[0],rname(f,cl->rd[1],ib),
               (unsigned long long)br->count,f->name,rname(f,cl->rd[0],nb));
      return buf;
    }
  }
  return rname(f,cl->rd[1],buf);
}
/* Nonzero if a label of the function's own spells `name` (a source label, or one a re-parsed emit kept). */
static int label_taken(const bcir_func *f, const char *name){
  for(size_t i=0;i<f->n_claims;i++)
    if(!strncmp(f->claims[i].op,"c.label:",8) && !strcmp(f->claims[i].op+8,name)) return 1;
  return 0;
}
/* The continue label of the function's loop `n`: `__cont_<n>`, or `__cont_<n>_<k>` where a label of the
 * function's own spells that already -- two definitions of one label do not compile (CF-RTWIDE; the oracle's
 * `_cont_labels`). `own` says whether any label of the function's starts `__cont_`, so a function with none
 * scans nothing. Two loops never collide: `<n>_<k>` is no loop number. */
static const char *cont_label(const bcir_func *f, int own, int n, char *buf, size_t cap){
  snprintf(buf,cap,"__cont_%d",n);
  for(int k=2; own && label_taken(f,buf); k++) snprintf(buf,cap,"__cont_%d_%d",n,k);
  return buf;
}
/* A function's signature as its definition spells it -- `static RET bcir_NAME(PARAMS)` -- into `o` (at most `on`
 * bytes; nothing past them) with the whole length returned: the definition's head, and the forward declaration of
 * the function a caller's emit makes. */
static size_t emit_sig(const bcir_func *f,char *o,size_t on){
  size_t w=0; char ty[BCIR_EMIT_TYPE];
  #define SO (w<on?w:on)
  ctype_str(&f->ret,ty,sizeof ty);
  w+=snprintf(o+SO,on-SO,"static %s bcir_%s(",ty,f->name);
  if(f->n_params==0&&!f->variadic) w+=snprintf(o+SO,on-SO,"void");
  for(int i=0;i<f->n_params;i++){char pt[BCIR_EMIT_TYPE];ctype_str(&f->params[i].type,pt,sizeof pt);
    w+=snprintf(o+SO,on-SO,"%s%s %s",i?", ":"",pt,f->params[i].name);}
  if(f->variadic) w+=snprintf(o+SO,on-SO,"%s...",f->n_params?", ":"");   /* a trailing variadic ellipsis */
  w+=snprintf(o+SO,on-SO,")");
  #undef SO
  return w;
}
/* A call's argument as the emit spells it (the oracle's `emit._args`). A file-scope multi-dimensional array is declared
 * nested, as the source declares it, so its name decays to a pointer to its first row (`T (*)[N]`); an emitted
 * parameter taking an array is the flat `T *` (a local array is declared flat already). Such an argument is spelled as
 * its first element's address, `&m[0][0]` -- the same address, of the parameter's type (CF-GARRAY). */
static const char *emit_arg(const bcir_func *f,uint32_t rid,char *buf,char *out,size_t on){
  const char *nm=rname(f,rid,buf); const bcir_resource *r=res_of(f,rid);
  if(!r || !r->read_only || r->ndims<2) return nm;
  size_t w=(size_t)snprintf(out,on,"&%s",nm);
  for(int d=0; d<r->ndims && w<on; d++) w+=(size_t)snprintf(out+w,on-w,"[0]");
  return out;
}
static size_t emit_func(const bcir_func *f,char *o,size_t on){
  size_t w=0; char a[BCIR_EMIT_NAME],b[BCIR_EMIT_NAME],d[BCIR_EMIT_NAME],e[BCIR_EMIT_NAME],ty[BCIR_EMIT_TYPE],tb[BCIR_EMIT_TYPE],gb[BCIR_EMIT_EXPR];
  bcir_emit_type_scratch type_scratch={0};
  /* Clamp the OFFSET (never form o+w / on-w once the buffer is full) for every `snprintf(o+EO,on-EO,...)`
   * below -- the same memory-safety idiom the funcptr-typedef builder uses (see SIG_OFF above). Once
   * `w>=on`, `o+EO` is at most one-past-the-end (a legal pointer) and `on-EO` is 0, so snprintf writes
   * NOTHING but still returns the would-be length, keeping `w` a true running total; without this, the
   * unbounded named-local declaration loop and deep indentation could form `o+w` past the end and
   * underflow `on-w` to a huge size_t, writing out of bounds. Output is byte-identical whenever it fits. */
  #define EO (w<on?w:on)
  ctype_str(&f->ret,ty,sizeof ty);
  w+=emit_sig(f,o+EO,on-EO);
  w+=snprintf(o+EO,on-EO,"\n{\n");
  /* declare named locals up front (mutable storage -- branch merges + loop accumulators) */
  for(size_t i=0;i<f->n_res;i++){const bcir_resource *r=&f->res[i];
    if(r->is_vla) continue;   /* a stack VLA: declared IN-BODY by c.vladecl (size unknown until then), not up front */
    if(is_named_local(f,r->rid)){
      char un[BCIR_EMIT_NAME]; const char *nm=uniq_local(f,r->rid,un);   /* unique vs same-named scopes */
      int sx=-1; for(int k=0;k<f->n_statics;k++) if(f->statics[k].rid==r->rid){sx=k;break;}
      /* a volatile OBJECT (scalar, array of scalars, struct) keeps its qualifier; a pointer's `volatile` is its
       * pointee's and rides in decl_ty; an array of pointers holds plain pointers */
      /* ... and an `_Atomic` object (a scalar, or an array's elements) is declared `_Atomic` (CF-ATOMIC): its reads
       * and writes by name are atomic, and it takes the ABI's atomic alignment */
      char vqb[24]; snprintf(vqb,sizeof vqb,"%s%s",
                             (r->is_volatile && r->kind!=BCIR_RK_POINTER && !r->ptr_depth)?"volatile ":"",
                             (r->is_atomic && r->kind!=BCIR_RK_POINTER && !r->ptr_depth)?"_Atomic ":"");
      const char *vq=vqb;
      /* a static takes its storage class and its constant image (CF-STATICTAB): the rendered initializer, or zero --
       * `{0}` for an array or aggregate, `0u` for a scalar, pointer or function pointer -- in whichever declaration
       * form its type takes (a pointer's extent is no array's: a static pointer is no scalar); a non-static array or
       * aggregate its zero baseline, the empty initializer `= {}` (CF-RTWIDE: `= {0}` also stores 0 to the first
       * scalar, which a re-parsed emit keeps as a store of its own) */
      const char *iv = sx>=0 ? (f->statics[sx].text ? f->statics[sx].text
                                : (r->kind==BCIR_RK_AGGREGATE || (r->kind==BCIR_RK_SCALAR && decl_array(r))) ? "{0}" : "0u")
                             : r->zinit ? "{}" : "";
      const char *sp=sx>=0?(f->statics[sx].thread_storage?"static _Thread_local ":"static "):"", *eq=iv[0]?" = ":"";
      if(r->is_funcptr&&r->agg[0]) w+=snprintf(o+EO,on-EO,"  %s%s %s%s%s;\n",sp,r->agg,nm,eq,iv);   /* a funcptr local: `__bcir_fpN f;` */
      else if(r->kind==BCIR_RK_AGGREGATE&&r->agg[0]) w+=snprintf(o+EO,on-EO,"  %s%s%s %s%s%s;\n",sp,vq,r->agg,nm,eq,iv);
      else if(r->kind==BCIR_RK_SCALAR&&decl_array(r)&&r->is_voidptr) w+=snprintf(o+EO,on-EO,"  %svoid *%s[%u]%s%s;\n",sp,nm,r->count,eq,iv);  /* an array of `void *` */
      else if(r->kind==BCIR_RK_SCALAR&&decl_array(r)&&r->agg[0]&&!r->ptr_depth) w+=snprintf(o+EO,on-EO,"  %s%s%s %s[%u]%s%s;\n",sp,vq,r->agg,nm,r->count,eq,iv);  /* an ARRAY-OF-STRUCTS local `struct P a[N]` */
      else if(r->kind==BCIR_RK_SCALAR&&decl_array(r)&&r->ptr_depth)   /* an ARRAY of pointers `T *a[N]`: each element
        * its pointer type (a pointer to volatile keeps the pointee's qualifier), never the pointer-wide integer */
        w+=snprintf(o+EO,on-EO,"  %s%s%s[%u]%s%s;\n",sp,ptr_spelling(tb,sizeof tb,r->is_volatile,0,r->agg,r->ptee_float,
                    r->ptee_bytes,r->ptee_signed,r->ptr_depth,0,0,r->ptee_plain_char,r->is_atomic),nm,r->count,eq,iv);
      else if(r->kind==BCIR_RK_SCALAR&&decl_array(r)) w+=snprintf(o+EO,on-EO,"  %s%s%s %s[%u]%s%s;\n",sp,vq,tty(&type_scratch,f,r->rid),nm,r->count,eq,iv);  /* a local array */
      else if(r->kind==BCIR_RK_POINTER)               /* a pointer local: `T *p` (the pointee carries width/sign) */
        w+=snprintf(o+EO,on-EO,"  %s%s%s%s%s;\n",sp,decl_ty(&type_scratch,f,r->rid,tb,sizeof tb),nm,eq,iv);
      else w+=snprintf(o+EO,on-EO,"  %s%s%s %s%s%s;\n",sp,vq,tty(&type_scratch,f,r->rid),nm,eq,iv);}}
  int depth=1, lstk[BCIR_MAXDEPTH+1], nls=0, lctr=0;   /* loop-id stack + counter for the `continue` labels */
  int own_cont=0; char cb[48];                          /* a label of the function's own spelled `__cont_...` */
  for(size_t i=0;i<f->n_claims && !own_cont;i++) own_cont=!strncmp(f->claims[i].op,"c.label:__cont_",15);
  #define IND() do{ for(int _k=0;_k<depth;_k++) w+=snprintf(o+EO,on-EO,"  "); }while(0)
  for(size_t i=0;i<f->n_claims;i++){const bcir_claim *cl=&f->claims[i];
    /* L6 control-flow markers (rendered as braces) */
    if(!strcmp(cl->op,"c.if")){IND();w+=snprintf(o+EO,on-EO,"if (%s) {\n",rname(f,cl->rd[0],a));depth++;continue;}
    if(!strcmp(cl->op,"c.else")){depth--;IND();w+=snprintf(o+EO,on-EO,"} else {\n");depth++;continue;}
    if(!strcmp(cl->op,"c.endif")){depth--;IND();w+=snprintf(o+EO,on-EO,"}\n");continue;}
    if(!strcmp(cl->op,"c.loop")){IND();w+=snprintf(o+EO,on-EO,"while (1) {\n");depth++;
      if(nls<BCIR_MAXDEPTH+1)lstk[nls++]=lctr++;continue;}
    if(!strcmp(cl->op,"c.loop.test")){
      /* a loop whose condition is the constant 1 -- `while (1)`, `for (;;)` -- ends no iteration: no test (the
       * oracle's `_const_true`, CF-RTWIDE) */
      const bcir_claim *pv=i?&f->claims[i-1]:NULL;
      if(pv && !strcmp(pv->op,"c.const") && pv->n_wr==1 && cl->n_rd==1 && pv->wr[0]==cl->rd[0] && pv->n_imm==1
         && pv->imm[0]==1) continue;
      IND();w+=snprintf(o+EO,on-EO,"if (!%s) break;\n",rname(f,cl->rd[0],a));continue;}
    if(!strcmp(cl->op,"c.cont.tgt")){IND();
      w+=snprintf(o+EO,on-EO,"%s: ;\n",cont_label(f,own_cont,nls?lstk[nls-1]:0,cb,sizeof cb));continue;}
    if(!strcmp(cl->op,"c.endloop")){depth--;IND();w+=snprintf(o+EO,on-EO,"}\n");if(nls)nls--;continue;}
    if(!strcmp(cl->op,"c.vladecl")){IND();   /* a 1-D stack VLA, declared IN-BODY: `<elem> a[__bcir_extK];` */
      { const bcir_resource *vr=res_of(f,cl->wr[0]);   /* a volatile VLA keeps its qualifier (the oracle's decl) */
        w+=snprintf(o+EO,on-EO,"%s%s %s[%s];\n",vr&&vr->is_volatile?"volatile ":"",tty(&type_scratch,f,cl->wr[0]),rname(f,cl->wr[0],a),rname(f,cl->rd[0],b)); }
      continue;}
    if(!strcmp(cl->op,"c.ptradd")){IND();w+=snprintf(o+EO,on-EO,"%s += %s;\n",rname(f,cl->wr[0],a),rname(f,cl->rd[1],b));continue;}  /* pointer p += n */
    if(!strcmp(cl->op,"c.ptrsub")){IND();w+=snprintf(o+EO,on-EO,"%s -= %s;\n",rname(f,cl->wr[0],a),rname(f,cl->rd[1],b));continue;}  /* pointer p -= n */
    if(!strcmp(cl->op,"c.break")){IND();w+=snprintf(o+EO,on-EO,"break;\n");continue;}
    if(!strcmp(cl->op,"c.switch")){IND();w+=snprintf(o+EO,on-EO,"switch (%s) {\n",rname(f,cl->rd[0],a));depth++;continue;}
    if(!strncmp(cl->op,"c.case:",7)){IND();w+=snprintf(o+EO,on-EO,"case %s:\n",cl->op+7);continue;}  /* a real case label */
    if(!strcmp(cl->op,"c.default")){IND();w+=snprintf(o+EO,on-EO,"default:\n");continue;}
    if(!strcmp(cl->op,"c.endswitch")){depth--;IND();w+=snprintf(o+EO,on-EO,"}\n");continue;}
    if(!strcmp(cl->op,"c.continue")){IND();
      w+=snprintf(o+EO,on-EO,"goto %s;\n",cont_label(f,own_cont,nls?lstk[nls-1]:0,cb,sizeof cb));continue;}
    if(!strncmp(cl->op,"c.goto:",7)){IND();w+=snprintf(o+EO,on-EO,"goto %s;\n",cl->op+7);continue;}
    if(!strcmp(cl->op,"c.cgoto")){IND();w+=snprintf(o+EO,on-EO,"goto *%s;\n",rname(f,cl->rd[0],a));continue;}  /* indirect jump to a label address (GNU) */
    if(!strncmp(cl->op,"c.label:",8)){w+=snprintf(o+EO,on-EO,"%s:;\n",cl->op+8);continue;}
    if(!strcmp(cl->op,"c.return")){IND();
      if(cl->n_rd) w+=snprintf(o+EO,on-EO,"return %s;\n",rname(f,cl->rd[0],a));
      else w+=snprintf(o+EO,on-EO,"return;\n");continue;}
    IND();
    if(!strncmp(cl->op,"c.bin.",6))                       /* decl_ty: a pointer result (`p + i`) declares `T *t` */
      w+=snprintf(o+EO,on-EO,"%s %s = %s %s %s;\n",decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),binop_c(cl->op+6),rname(f,cl->rd[1],b));
    else if(!strncmp(cl->op,"c.labeladdr:",12))            /* `&&L` -- a label's address as a `void *` (GNU) */
      w+=snprintf(o+EO,on-EO,"%s %s = &&%s;\n",decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),cl->op+12);
    else if(!strncmp(cl->op,"c.fconst:",9))                /* a floating constant -> its literal spelling */
      w+=snprintf(o+EO,on-EO,"%s %s = %s;\n",tty(&type_scratch,f,cl->wr[0]),rname(f,cl->wr[0],d),cl->op+9);
    else if(!strncmp(cl->op,"c.cconst:",9))                /* <complex.h> imaginary unit -> verbatim token */
      w+=snprintf(o+EO,on-EO,"%s %s = %s;\n",tty(&type_scratch,f,cl->wr[0]),rname(f,cl->wr[0],d),cl->op+9);
    else if(!strcmp(cl->op,"c.un.creal"))                  /* GNU __real__ z -- the real part (an element float) */
      w+=snprintf(o+EO,on-EO,"%s %s = __real__ %s;\n",tty(&type_scratch,f,cl->wr[0]),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a));
    else if(!strcmp(cl->op,"c.un.cimag"))                  /* GNU __imag__ z -- the imaginary part */
      w+=snprintf(o+EO,on-EO,"%s %s = __imag__ %s;\n",tty(&type_scratch,f,cl->wr[0]),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a));
    else if(!strncmp(cl->op,"c.un.",5))                    /* `-`/`~` keep the operand width (long stays 64) */
      w+=snprintf(o+EO,on-EO,"%s %s = (%s%s);\n",tty(&type_scratch,f,cl->wr[0]),rname(f,cl->wr[0],d),unop_c(cl->op+5),rname(f,cl->rd[0],a));
    else if(!strncmp(cl->op,"c.cast:",7)){                 /* (type)operand -- width / float / pointer cast */
      const char *dt=decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb);   /* decl_ty: a pointer-snapshot cast keeps `T *` */
      const char *to=cl->op+7;
      while(!strncmp(to,"volatile ",9) || !strncmp(to,"_Atomic ",8)) to+=to[0]=='v' ? 9 : 8;
      w+=snprintf(o+EO,on-EO,"%s %s = (%s)%s;\n",dt,rname(f,cl->wr[0],d),
                  strncmp(to,"fnptr",5) ? cl->op+7 : dt,   /* a function pointer, or a pointer to one: the temp's own
                                                            * type, its alias -- `fnptr` names no C type (CF-RTFP) */
                  rname(f,cl->rd[0],a)); }
    else if(!strcmp(cl->op,"c.select"))                    /* ternary: cond ? then : els -- the select's own
                                                            * (signed/unsigned) type, not a hardcoded
                                                            * uint32_t (see the c.const note below); a pointer
                                                            * select is its `T *` (decl_ty, CF-DECAY), a function-
                                                            * pointer select its alias (CF-FNSEL) */
    { const bcir_resource *sr=res_of(f,cl->wr[0]);
      w+=snprintf(o+EO,on-EO,"%s %s = (%s ? %s : %s);\n",sr&&sr->is_funcptr&&sr->agg[0]?sr->agg:decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb),
                  rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),rname(f,cl->rd[1],b),rname(f,cl->rd[2],e)); }
    else if(!strcmp(cl->op,"c.const")){
      /* declare the constant with its OWN type, not a hardcoded uint32_t: a bare integer literal (e.g.
       * `0` in `x < 0`) is signed (int), so emitting `uint32_t = 0u` made a signed comparison promote to
       * unsigned (`int32_t < uint32_t` -> unsigned) -- a miscompile. The literal's (width, signedness)
       * was already recorded on the temp (lit_int_type); render the matching type + suffix. */
      const bcir_resource *cr=res_of(f,cl->wr[0]); int cs=cr&&cr->is_signed;   /* a null pointer constant's
                                                       * temp declares its pointer type (CF-NULLPTR) */
      char kd[32];                                     /* a negative one (an enumerator, a character) signed, not its
                                                        * 64-bit two's complement, which C reads back with a warning */
      if(cs && cl->imm[0]<0) kdigits(kd,sizeof kd,(unsigned long long)cl->imm[0],1);
      else snprintf(kd,sizeof kd,"%llu%s",(unsigned long long)cl->imm[0],cs?"":"u");
      w+=snprintf(o+EO,on-EO,"%s %s = %s;\n",cr&&cr->is_funcptr&&cr->agg[0]?cr->agg:decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),
                  kd); }
    else if(!strcmp(cl->op,"c.sizeof.vla"))                 /* runtime `sizeof a` of a VLA: extent × sizeof(elem).
                                                            * HARDCODE the literal `size_t` (NOT tty(), which
                                                            * returns "uint64_t" for an 8-byte unsigned scalar):
                                                            * the oracle emits literal `size_t` (scalar('size_t')),
                                                            * so the two rails would diverge byte-for-byte. */
      w+=snprintf(o+EO,on-EO,"size_t %s = (size_t)((size_t)%s * %lld);\n",rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),(long long)cl->imm[0]);
    else if(!strcmp(cl->op,"c.copy")){
      if(is_named_local(f,cl->wr[0])||is_global_ref(f,cl->wr[0])||is_param_ref(f,cl->wr[0])) w+=snprintf(o+EO,on-EO,"%s = %s;\n",rname(f,cl->wr[0],d),rname(f,cl->rd[0],a));
      else w+=snprintf(o+EO,on-EO,"%s %s = %s;\n",decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a));   /* decl_ty: a copied pointer temp keeps `T *` */
    }else if(!strcmp(cl->op,"c.load") && cl->hazard==BCIR_HZ_ATOMIC){   /* a read of an `_Atomic` object
                                                       * (CF-ATOMIC): one atomic load through an `_Atomic` lvalue */
      char ob[BCIR_EMIT_EXPR]; const char *et=decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb);
      atomic_object(f,cl,et,cl->n_rd,2,ob,sizeof ob);
      w+=snprintf(o+EO,on-EO,"%s %s = %s;\n",et,rname(f,cl->wr[0],d),ob);
    }else if(!strcmp(cl->op,"c.load")){
      const bcir_resource *br=res_of(f,cl->rd[0]); long long off=cl->n_imm?cl->imm[0]:0;
      if(cl->n_rd==2 && cl->n_imm){       /* s.arr[i] / a[i].f: load at base + off + idx*stride, copy `es` bytes */
        const char *amp=base_amp(br); long long es=cl->n_imm>1?cl->imm[1]:4;
        long long stride=cl->n_imm>2?cl->imm[2]:es;   /* array-of-structs `arr[i].field`: stride sizeof(elem) != es */
        /* the temp carries the element's (width, signedness): memcpy es bytes into it so a signed sub-int
         * element reads sign-extended (the zero-extending uint32 form dropped the sign). */
        if(cl->is_volatile){ char at[BCIR_EMIT_TYPE], vp[BCIR_EMIT_TYPE+16];     /* a volatile element: one access of exactly its type */
          w+=snprintf(o+EO,on-EO,"%s %s = *(%s)((const volatile char *)%s%s + %lld + (size_t)%s * %lld);\n",
            tty(&type_scratch,f,cl->wr[0]),rname(f,cl->wr[0],d),
            vol_ptr(vol_load_ty(&type_scratch,f,cl->wr[0],es,at,sizeof at),vp,sizeof vp),
            amp,rname(f,cl->rd[0],a),off,rname(f,cl->rd[1],b),stride); }
        else
        w+=snprintf(o+EO,on-EO,"%s %s; memcpy(&%s, (const char *)%s%s + %lld + (size_t)%s * %lld, %lld);\n",
          decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),rname(f,cl->wr[0],d),amp,rname(f,cl->rd[0],a),off,rname(f,cl->rd[1],b),stride,es); }   /* decl_ty: an array-of-pointers element is `T *` */
      else if(cl->n_rd==2) w+=snprintf(o+EO,on-EO,"%s %s = %s[%s];\n",decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),guard_idx(f,cl,gb,sizeof gb,0));  /* READ guard; decl_ty: an array-of-pointers element load is `T *` */
      else if(cl->is_volatile){ char at[BCIR_EMIT_TYPE], vp[BCIR_EMIT_TYPE+16];   /* a volatile member / dereference: one ordered access of
                                                       * exactly the accessed type -- never a 32-bit register */
        const char *amp=holds_pointer(br)?"":"&";
        w+=snprintf(o+EO,on-EO,"%s %s = *(%s)((const volatile char *)%s%s + %lld);\n",
          decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),
          vol_ptr(vol_load_ty(&type_scratch,f,cl->wr[0],cl->n_imm>1?cl->imm[1]:0,at,sizeof at),vp,sizeof vp),
          amp,rname(f,cl->rd[0],a),off); }
      else { const char *amp=holds_pointer(br)?"":"&"; long long fsz=cl->n_imm>1?cl->imm[1]:4;
        /* a plain member load: memcpy fsz bytes into the typed temp so a signed sub-int member sign-extends */
        w+=snprintf(o+EO,on-EO,"%s %s; memcpy(&%s, (const char *)%s%s + %lld, %lld);\n",decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),rname(f,cl->wr[0],d),amp,rname(f,cl->rd[0],a),off,fsz); }  /* decl_ty: a pointer member load is `T *t` */
    }else if(!strcmp(cl->op,"c.store") && cl->hazard==BCIR_HZ_ATOMIC){   /* a write of an `_Atomic` object
                                                       * (CF-ATOMIC): one atomic store of exactly its slot's type */
      char ob[BCIR_EMIT_EXPR], st[BCIR_EMIT_TYPE]; int naddr=cl->n_rd-1;
      const char *slot = (naddr==2 && !cl->n_imm) ? elem_spelling(&type_scratch,f,cl->rd[0],st,sizeof st)
        : vol_slot_ty(&type_scratch,f,cl->rd[cl->n_rd-1],cl->n_imm>1?cl->imm[1]:4,(int)(cl->n_imm>2?cl->imm[2]:0),st,sizeof st);
      atomic_object(f,cl,slot,naddr,3,ob,sizeof ob);
      w+=snprintf(o+EO,on-EO,"%s = %s;\n",ob,rname(f,cl->rd[cl->n_rd-1],b));
    }else if(!strcmp(cl->op,"c.store")&&cl->n_rd==3){   /* L3: array element store  a[idx] = value */
      if(cl->n_imm){                      /* s.arr[i]=v / a[i].f=v: store at base + off + idx*stride */
        const bcir_resource *br=res_of(f,cl->rd[0]); const char *amp=base_amp(br);
        long long off=cl->imm[0], es=cl->n_imm>1?cl->imm[1]:4;
        long long stride=cl->n_imm>3?cl->imm[3]:es;    /* array-of-structs `arr[i].field=v`: stride sizeof(elem) */
        const bcir_resource *vr=res_of(f,cl->rd[2]);   /* a float element converts (double->float), not a uint
                                                        * reinterpret; a narrower int widens to the element. */
        const char *vt=(cl->n_imm>2&&cl->imm[2])?"_Bool"   /* a _Bool element: `_Bool _v = x` normalizes to 0/1 */
                      :(vr&&vr->kind==BCIR_RK_POINTER)?decl_ty(&type_scratch,f,cl->rd[2],tb,sizeof tb)   /* a pointer
                       * element (`T *arr[N]`) copies the whole pointer, never a truncating uint32 */
                      :(vr&&(vr->is_array||vr->is_vla))?"const volatile void *"   /* an array decays to its
                       * address (CF-MEMDECAY): the pointer is staged, never an integer holding the array */
                      :(vr&&vr->is_complex)?(es==8?"float _Complex":es>16?"long double _Complex":"double _Complex")
                      :(vr&&vr->is_float)?(es==4?"float":es>8?"long double":"double")
                      :(es==1?"uint8_t":es==2?"uint16_t":es==8?"uint64_t":"uint32_t");
        if(cl->is_volatile){ char st[BCIR_EMIT_TYPE], vp[BCIR_EMIT_TYPE+16];     /* a volatile element: one store of exactly its slot */
          w+=snprintf(o+EO,on-EO,"*(%s)((volatile char *)%s%s + %lld + (size_t)%s * %lld) = %s;\n",
            vol_ptr(vol_slot_ty(&type_scratch,f,cl->rd[2],es,(int)(cl->n_imm>2?cl->imm[2]:0),st,sizeof st),vp,sizeof vp),
            amp,rname(f,cl->rd[0],a),off,rname(f,cl->rd[1],b),stride,rname(f,cl->rd[2],d)); }
        else if(vr && vr->is_funcptr){ char fs[BCIR_EMIT_TYPE+32];   /* an element of a member table of function pointers
          * (`o.t[i] = f`, CF-FPTAB): through a pointer to its own type, as a member's store -- `uint64_t _v = f` does
          * not compile (CF-RTFP; `fp_slot_ty`) */
          w+=snprintf(o+EO,on-EO,"*(%s)((char *)%s%s + %lld + (size_t)%s * %lld) = %s;\n",
            fp_slot_ty(vr,rname(f,cl->rd[2],d),fs,sizeof fs),amp,rname(f,cl->rd[0],a),off,rname(f,cl->rd[1],b),stride,
            rname(f,cl->rd[2],d)); }
        else if(vr && vr->kind==BCIR_RK_AGGREGATE)    /* a struct/union element set from a struct VALUE (`s.v[i] = q`):
          * copy it whole, as the member store does -- a scalar `_v` of it does not compile (CF-STRUCTVAL) */
          w+=snprintf(o+EO,on-EO,"memcpy((char *)%s%s + %lld + (size_t)%s * %lld, &%s, %lld);\n",
            amp,rname(f,cl->rd[0],a),off,rname(f,cl->rd[1],b),stride,rname(f,cl->rd[2],d),es);
        else
        w+=snprintf(o+EO,on-EO,"{ %s _v = %s; memcpy((char *)%s%s + %lld + (size_t)%s * %lld, &_v, %lld); }\n",
          vt,rname(f,cl->rd[2],d),amp,rname(f,cl->rd[0],a),off,rname(f,cl->rd[1],b),stride,es); }
      else                                    /* `base[idx] = v`: through the base's declared type, which
                                                * carries its pointee's `volatile` -- the element's own width */
        w+=snprintf(o+EO,on-EO,"%s[%s] = %s;\n",rname(f,cl->rd[0],a),guard_idx(f,cl,gb,sizeof gb,1),rname(f,cl->rd[2],d));  /* WRITE guard: an OOB store fails-fast, never clamps */
    }else if(!strcmp(cl->op,"c.store")){          /* L8: member store -> memcpy `size` bytes */
      const bcir_resource *br=res_of(f,cl->rd[0]); long long off=cl->imm[0]; long long sz=cl->n_imm>1?cl->imm[1]:4;
      if(cl->is_volatile){ const char *amp=holds_pointer(br)?"":"&";   /* a volatile member /
        * dereference: one ordered store of exactly its slot type -- never a 32-bit register by assumption */
        const bcir_resource *vr=res_of(f,cl->rd[1]);
        if(vr && vr->is_funcptr)
          w+=snprintf(o+EO,on-EO,"*(void (* volatile *)(void))((volatile char *)%s%s + %lld) = (void (*)(void))%s;\n",
                      amp,rname(f,cl->rd[0],a),off,rname(f,cl->rd[1],b));
        else { char st[BCIR_EMIT_TYPE], vp[BCIR_EMIT_TYPE+16];
          w+=snprintf(o+EO,on-EO,"*(%s)((volatile char *)%s%s + %lld) = %s;\n",
            vol_ptr(vol_slot_ty(&type_scratch,f,cl->rd[1],sz,(int)(cl->n_imm>2?cl->imm[2]:0),st,sizeof st),vp,sizeof vp),
            amp,rname(f,cl->rd[0],a),off,rname(f,cl->rd[1],b)); } }
      else { const char *amp=holds_pointer(br)?"":"&";
        /* a pointer value stored into a (pointer_size) member: `_v` carries the real `T *` type so the
         * full pointer is copied -- a `uint32_t _v` would truncate the 8-byte pointer to 4. */
        /* `_v` is sized to the MEMBER width `sz` (not a flat uint32) so a wide scalar member store
         * (`s->m = v` with `long m`) moves all 8 bytes and the value widens/truncates to the member,
         * not over-reads a 4-byte temp; a float member keeps its float type; a pointer its `T *`. */
        const bcir_resource *vr=res_of(f,cl->rd[1]);
        if(vr && vr->is_funcptr){   /* a FUNCTION-POINTER member set from a funcptr value (`o->fn = g` /
          * `o->fn = g_func`): store through a pointer to its own type, its alias, so a function NAME decays to its
          * address (a plain `memcpy(&g_func,8)` would copy the function's CODE; `void *` cannot hold a funcptr) and
          * the call site reads the member as the type it was stored as -- a store through a generic
          * `void (**)(void)` read back as the member's type is undefined, and GCC at -O2 drops it (CF-RTFP; the
          * oracle's `_fp_store`; `fp_slot_ty`). */
          char fs[BCIR_EMIT_TYPE+32];
          w+=snprintf(o+EO,on-EO,"*(%s)((char *)%s%s + %lld) = %s;\n",
                      fp_slot_ty(vr,rname(f,cl->rd[1],b),fs,sizeof fs),amp,rname(f,cl->rd[0],a),off,rname(f,cl->rd[1],b));
        } else if(vr && vr->kind==BCIR_RK_AGGREGATE && !vr->is_array){   /* a struct/union member set from a struct VALUE (a nested
          * `{ ... }` member, `o.p = q`): copy the whole object -- a scalar `uintN _v = <struct>` is a type
          * error, and a too-narrow `_v` would under-read it. memcpy `sz` bytes straight from the source. */
          w+=snprintf(o+EO,on-EO,"memcpy((char *)%s%s + %lld, &%s, %lld);\n",
                      amp,rname(f,cl->rd[0],a),off,rname(f,cl->rd[1],b),sz);
        } else {
        int flag=cl->n_imm>2?cl->imm[2]:0;
        const char *vt=flag==1?"_Bool"                     /* a _Bool member: `_Bool _v = x` normalizes to 0/1 */
                      :flag==2?(vr&&vr->elem_bytes>4?"uint64_t":"uint32_t")   /* a bitfield UNIT: `_v` is the full
                       * unit type (may be wider than the `sz` bytes written -- a packed field spans <= 8 bytes
                       * into a uint64 unit but only its `sz` spanned bytes are memcpy'd back) */
                      :(vr&&vr->kind==BCIR_RK_POINTER)?decl_ty(&type_scratch,f,cl->rd[1],tb,sizeof tb)
                      :(vr&&(vr->is_array||vr->is_vla))?"const volatile void *"   /* an array decays to its address
                       * (C11 6.3.2.1p3, CF-MEMDECAY): `uintN _v = arr` does not compile */
                      :(vr&&vr->is_complex)?(sz==8?"float _Complex":sz>16?"long double _Complex":"double _Complex")
                      :(vr&&vr->is_float)?(sz==4?"float":sz>8?"long double":"double")
                      :(sz==1?"uint8_t":sz==2?"uint16_t":sz==8?"uint64_t":"uint32_t");
        w+=snprintf(o+EO,on-EO,"{ %s _v = %s; memcpy((char *)%s%s + %lld, &_v, %lld); }\n",vt,rname(f,cl->rd[1],b),amp,rname(f,cl->rd[0],a),off,sz); } }
    }else if(!strcmp(cl->op,"c.bf.get")){
      long long off=cl->imm[0],bw=cl->imm[1]; int wide=bw>32;      /* a WIDE bitfield needs 64-bit literals/cast */
      unsigned long long mask=bw>=64?~0ull:(1ull<<bw)-1; const char *sfx=wide?"ull":"u";
      if(cl->n_imm>2&&cl->imm[2]){                       /* a signed bitfield: sign-extend from bit bw-1 */
        unsigned long long sbit=1ull<<(bw-1); const char *cast=wide?"int64_t":"int32_t";
        w+=snprintf(o+EO,on-EO,"%s %s = (%s)((((%s >> %lld) & %llu%s) ^ %llu%s) - %llu%s);\n",
                    tty(&type_scratch,f,cl->wr[0]),rname(f,cl->wr[0],d),cast,rname(f,cl->rd[0],a),off,mask,sfx,sbit,sfx,sbit,sfx);
      } else
        w+=snprintf(o+EO,on-EO,"%s %s = (%s >> %lld) & %llu%s;\n",
                    tty(&type_scratch,f,cl->wr[0]),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),off,mask,sfx); }
    else if(!strcmp(cl->op,"c.bf.set")){          /* (old & ~(mask<<off)) | ((v & mask) << off) */
      long long off=cl->imm[0]; const bcir_resource *ur=res_of(f,cl->rd[0]); int wide=ur&&ur->elem_bytes>4;
      unsigned long long mask=cl->imm[1]>=64?~0ull:(1ull<<cl->imm[1])-1;
      unsigned long long clear=~(mask<<off)&(wide?~0ull:0xFFFFFFFFull); const char *sfx=wide?"ull":"u";
      w+=snprintf(o+EO,on-EO,"%s %s = (%s & %llu%s) | ((%s & %llu%s) << %lld);\n",wide?"uint64_t":"uint32_t",
                  rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),clear,sfx,rname(f,cl->rd[1],b),mask,sfx,off); }
    else if(!strncmp(cl->op,"c.atomic.",9))      /* atomic RMW -> the matching builtin, its value typed by the
                                                  * pointee (a uint32 truncated a 64-bit counter, CF-ATOMIC) */
      w+=snprintf(o+EO,on-EO,"%s %s = __atomic_fetch_%s(%s, %s, __ATOMIC_SEQ_CST);\n",decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb),
                  rname(f,cl->wr[0],d),cl->op+9,rname(f,cl->rd[0],a),rname(f,cl->rd[1],b));
    else if(!strncmp(cl->op,"c.cmpxchg.",10))     /* compare-and-swap -> the __sync CAS builtin (the value form
                                                  * typed by the pointee) */
      w+=snprintf(o+EO,on-EO,"%s %s = __sync_%s_compare_and_swap(%s, %s, %s);\n",decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb),
                  rname(f,cl->wr[0],d),cl->op+10,rname(f,cl->rd[0],a),rname(f,cl->rd[1],b),rname(f,cl->rd[2],e));
    else if(!strcmp(cl->op,"c.fence"))
      w+=snprintf(o+EO,on-EO,"__atomic_thread_fence(__ATOMIC_SEQ_CST);\n");
    else if(!strcmp(cl->op,"c.fence.acquire"))    /* SEG7: order-parameterized acquire fence (PORTABLE -- the */
      w+=snprintf(o+EO,on-EO,"__atomic_thread_fence(__ATOMIC_ACQUIRE);\n");   /* C twin does NO per-ISA asm) */
    else if(!strcmp(cl->op,"c.fence.release"))    /* SEG7: order-parameterized release fence (PORTABLE) */
      w+=snprintf(o+EO,on-EO,"__atomic_thread_fence(__ATOMIC_RELEASE);\n");
    else if(!strncmp(cl->op,"c.c11atom.rmw:",14)){   /* a compound assignment / inc / dec of an `_Atomic` object
                                                       * (CF-ATOMIC): one atomic read-modify-write through an
                                                       * `_Atomic` lvalue, its value the expression's */
      const char *kind=cl->op+14; char ob[BCIR_EMIT_EXPR], ex[BCIR_EMIT_EXPR+BCIR_EMIT_NAME+8];
      int incdec=!strcmp(kind,"preinc")||!strcmp(kind,"predec")||!strcmp(kind,"postinc")||!strcmp(kind,"postdec");
      const char *et=decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb);
      atomic_object(f,cl,et,cl->n_rd-(incdec?0:1),3,ob,sizeof ob);
      if(!strcmp(kind,"preinc")) snprintf(ex,sizeof ex,"++%s",ob);
      else if(!strcmp(kind,"predec")) snprintf(ex,sizeof ex,"--%s",ob);
      else if(!strcmp(kind,"postinc")) snprintf(ex,sizeof ex,"%s++",ob);
      else if(!strcmp(kind,"postdec")) snprintf(ex,sizeof ex,"%s--",ob);
      else snprintf(ex,sizeof ex,"%s %s= %s",ob,binop_c(kind),rname(f,cl->rd[cl->n_rd-1],b));
      w+=snprintf(o+EO,on-EO,"%s %s = (%s);\n",et,rname(f,cl->wr[0],d),ex); }
    else if(!strncmp(cl->op,"c.c11atom.",10)){   /* C11 <stdatomic.h> generics on _Atomic objects */
      const char *fn=cl->op+10;                  /* fetch_add / fetch_sub / fetch_xor / load / store */
      if(!strcmp(fn,"load")) w+=snprintf(o+EO,on-EO,"%s %s = atomic_load(%s);\n",   /* typed by the pointee (CF-ATOMIC) */
                                         decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a));
      else if(!strcmp(fn,"store")) w+=snprintf(o+EO,on-EO,"atomic_store(%s, %s);\n",rname(f,cl->rd[0],a),rname(f,cl->rd[1],b));
      else if(!strncmp(fn,"cas_",4))             /* cas_strong/weak -> _Bool atomic_compare_exchange_<...>(obj,&exp,des) */
        w+=snprintf(o+EO,on-EO,"_Bool %s = atomic_compare_exchange_%s(%s, %s, %s);\n",
                    rname(f,cl->wr[0],d),fn+4,rname(f,cl->rd[0],a),rname(f,cl->rd[1],b),rname(f,cl->rd[2],e));
      else w+=snprintf(o+EO,on-EO,"%s %s = atomic_%s(%s, %s);\n",decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb),
                       rname(f,cl->wr[0],d),fn,rname(f,cl->rd[0],a),rname(f,cl->rd[1],b)); }
    else if(!strcmp(cl->op,"c.addrof")){           /* &lvalue -> a pointer value (decl_ty: `T *`, `T **`, ...) */
      const char *pt=decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb);
      const bcir_resource *br=res_of(f,cl->rd[0]);
      const char *amp=holds_pointer(br)?"":"&";   /* a pointer base decays (`(char*)s`), a value
                                                                 * / array base is addressed (`(char*)&s`) */
      if(cl->n_rd==2)                              /* &base[idx] -> (T *)((char *)base + off + idx*es) */
        w+=snprintf(o+EO,on-EO,"%s%s = (%s)((char *)%s%s + %lld + (size_t)%s * %lld);\n",
          pt,rname(f,cl->wr[0],d),pt,amp,rname(f,cl->rd[0],a),(long long)cl->imm[0],rname(f,cl->rd[1],b),(long long)cl->imm[1]);
      else if(cl->n_imm)                           /* &member -> a typed `(T *)((char *)<amp>base + off)` */
        w+=snprintf(o+EO,on-EO,"%s%s = (%s)((char *)%s%s + %lld);\n",pt,rname(f,cl->wr[0],d),pt,amp,rname(f,cl->rd[0],a),(long long)cl->imm[0]);
      else
        w+=snprintf(o+EO,on-EO,"%s%s = &%s;\n",pt,rname(f,cl->wr[0],d),rname(f,cl->rd[0],a)); }
    else if(!strncmp(cl->op,"c.call.libm:",12)){   /* a <math.h> / <stdlib.h> call -> the real libc function */
      const bcir_resource *wr=res_of(f,cl->wr[0]);            /* an allocator returns `void *`, not a scalar */
      const char *rty=(wr&&wr->kind==BCIR_RK_POINTER)?"void *":tty(&type_scratch,f,cl->wr[0]);
      w+=snprintf(o+EO,on-EO,"%s %s = %s(",rty,rname(f,cl->wr[0],d),cl->op+12);
      for(int k=0;k<cl->n_rd;k++) w+=snprintf(o+EO,on-EO,"%s%s",k?", ":"",rname(f,cl->rd[k],a));
      w+=snprintf(o+EO,on-EO,");\n"); }
    else if(!strncmp(cl->op,"c.call.libm.void:",17)){   /* a void external (free) -> a verbatim call statement */
      w+=snprintf(o+EO,on-EO,"%s(",cl->op+17);
      for(int k=0;k<cl->n_rd;k++) w+=snprintf(o+EO,on-EO,"%s%s",k?", ":"",rname(f,cl->rd[k],a));
      w+=snprintf(o+EO,on-EO,");\n"); }
    else if(!strncmp(cl->op,"c.call.extern:",14)){  /* a printf/scanf-family external variadic -> verbatim */
      w+=snprintf(o+EO,on-EO,"%s %s = %s(",tty(&type_scratch,f,cl->wr[0]),rname(f,cl->wr[0],d),cl->op+14);
      for(int k=0;k<cl->n_rd;k++) w+=snprintf(o+EO,on-EO,"%s%s",k?", ":"",rname(f,cl->rd[k],a));
      w+=snprintf(o+EO,on-EO,");\n"); }
    else if(!strncmp(cl->op,"c.call.tu:",10)){      /* a PROTOTYPED cross-TU callee (Phase 3 linking):
                                                     * verbatim, external linkage -- the prelude declares
                                                     * it; the host LINKER resolves it */
      char cty[BCIR_EMIT_TYPE], qb[BCIR_QCAST_TYPE+2];                                 /* a pointer result declares `T *` (decl_ty) */
      if(cl->n_wr==0) w+=snprintf(o+EO,on-EO,"%s(",cl->op+10);
      else{ const char *dt=decl_ty(&type_scratch,f,cl->wr[0],cty,sizeof cty);   /* a qualified return cast to it */
        w+=snprintf(o+EO,on-EO,"%s %s = %s%s(",dt,rname(f,cl->wr[0],d),qcast_text(f,cl,-1,dt,qb,sizeof qb),cl->op+10); }
      for(int k=0;k<cl->n_rd;k++)                               /* ... an argument to a qualified parameter (CF-QUALS) */
        w+=snprintf(o+EO,on-EO,"%s%s%s",k?", ":"",qcast_text(f,cl,k,NULL,qb,sizeof qb),emit_arg(f,cl->rd[k],a,gb,sizeof gb));
      w+=snprintf(o+EO,on-EO,");\n"); }
    else if(!strncmp(cl->op,"c.call.builtin:",15)){  /* a GCC/Clang integer builtin -> emitted verbatim */
      w+=snprintf(o+EO,on-EO,"%s %s = __builtin_%s(",tty(&type_scratch,f,cl->wr[0]),rname(f,cl->wr[0],d),cl->op+15);
      for(int k=0;k<cl->n_rd;k++) w+=snprintf(o+EO,on-EO,"%s%s",k?", ":"",rname(f,cl->rd[k],a));
      w+=snprintf(o+EO,on-EO,");\n"); }
    else if(!strncmp(cl->op,"c.call.vaarg:",13)){   /* va_arg(ap, T) -- pull the next variadic argument */
      w+=snprintf(o+EO,on-EO,"%s %s = va_arg(%s, %s);\n",
        decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),cl->op+13); }
    else if(!strncmp(cl->op,"c.call.vabuiltin:",17)){   /* va_start / va_end / va_copy -- emitted verbatim, void */
      w+=snprintf(o+EO,on-EO,"%s(",cl->op+17);
      for(int k=0;k<cl->n_rd;k++) w+=snprintf(o+EO,on-EO,"%s%s",k?", ":"",rname(f,cl->rd[k],a));
      w+=snprintf(o+EO,on-EO,");\n"); }
    else if(!strncmp(cl->op,"c.call.void:",12)){   /* a void callee -> a bare call statement */
      w+=snprintf(o+EO,on-EO,"bcir_%s(",cl->op+12);
      for(int k=0;k<cl->n_rd;k++) w+=snprintf(o+EO,on-EO,"%s%s",k?", ":"",emit_arg(f,cl->rd[k],a,gb,sizeof gb));
      w+=snprintf(o+EO,on-EO,");\n"); }
    else if(!strncmp(cl->op,"c.call:",7)){
      const bcir_resource *rr=res_of(f,cl->wr[0]);   /* a struct/union RETURN declares `struct P t = bcir_..`,
                                                      * a pointer return `T *t = bcir_..` (decl_ty) */
      char cty[BCIR_EMIT_TYPE];
      const char *dty=(rr&&rr->kind==BCIR_RK_AGGREGATE&&rr->agg[0])?rr->agg
                      :decl_ty(&type_scratch,f,cl->wr[0],cty,sizeof cty);
      w+=snprintf(o+EO,on-EO,"%s %s = bcir_%s(",dty,rname(f,cl->wr[0],d),cl->op+7);
      for(int k=0;k<cl->n_rd;k++) w+=snprintf(o+EO,on-EO,"%s%s",k?", ":"",emit_arg(f,cl->rd[k],a,gb,sizeof gb));
      w+=snprintf(o+EO,on-EO,");\n"); }
    else if(!strcmp(cl->op,"c.call.indirect")){    /* rd[0] is the function pointer; rd[1..] the args */
      char cty[BCIR_EMIT_TYPE], qb[BCIR_QCAST_TYPE+2];                     /* the result declared by its type -- a pointer return `T *t` (CF-FPRET) */
      if(cl->n_wr==0) w+=snprintf(o+EO,on-EO,"%s(",rname(f,cl->rd[0],a));   /* a void function: a bare call (CF-VOIDCB) */
      else{ const char *dt=decl_ty(&type_scratch,f,cl->wr[0],cty,sizeof cty);   /* result typed by the funcptr's return */
        w+=snprintf(o+EO,on-EO,"%s %s = %s%s(",dt,rname(f,cl->wr[0],d),qcast_text(f,cl,-1,dt,qb,sizeof qb),rname(f,cl->rd[0],a)); }
      for(int k=1;k<cl->n_rd;k++)                               /* its qualified parameters and return (CF-QUALS) */
        w+=snprintf(o+EO,on-EO,"%s%s%s",k>1?", ":"",qcast_text(f,cl,k,NULL,qb,sizeof qb),emit_arg(f,cl->rd[k],b,gb,sizeof gb));
      w+=snprintf(o+EO,on-EO,");\n"); }
    else if(!strncmp(cl->op,"c.call.imember:",15)){   /* o->fn(args): funcptr struct member */
      const char *sep=(cl->n_imm&&cl->imm[0])?"->":".";
      char cty[BCIR_EMIT_TYPE], qb[BCIR_QCAST_TYPE+2];
      if(cl->n_wr==0) w+=snprintf(o+EO,on-EO,"%s%s%s(",rname(f,cl->rd[0],a),sep,cl->op+15);   /* a void member function */
      else{ const char *dt=decl_ty(&type_scratch,f,cl->wr[0],cty,sizeof cty);   /* result typed by the funcptr's return */
        w+=snprintf(o+EO,on-EO,"%s %s = %s%s%s%s(",dt,rname(f,cl->wr[0],d),qcast_text(f,cl,-1,dt,qb,sizeof qb),
                    rname(f,cl->rd[0],a),sep,cl->op+15); }
      for(int k=1;k<cl->n_rd;k++)                               /* its qualified parameters and return (CF-QUALS) */
        w+=snprintf(o+EO,on-EO,"%s%s%s",k>1?", ":"",qcast_text(f,cl,k,NULL,qb,sizeof qb),emit_arg(f,cl->rd[k],b,gb,sizeof gb));
      w+=snprintf(o+EO,on-EO,");\n"); }
  }
  #undef IND
  w+=snprintf(o+EO,on-EO,"}\n");
  #undef EO
  return w;
}

/* Try to parse a top-level type declaration (typedef / enum definition /
 * struct|union definition) at the current token.  Returns 1 if one was consumed,
 * 0 if the current token instead begins a function/global.  Real translation
 * units and vendor headers interleave these with functions, so this is called
 * from the main top-level loop rather than only before the first function. */
static void p_global_declarator(CC *c, const bcir_ctype *base, int si, int btd, const int *btdd);   /* fwd (below) */
static int try_top_decl(CC *c){
  if(c->failed) return 0;
  if(is(c,"typedef")){ p_typedef(c); return 1; }
  if(is(c,"enum")){
    int save=c->i; c->i++; if(isk(c,T_ID)&&!is(c,"{")) c->i++;
    if(is(c,"{")){ p_enum_body(c); eat(c,";"); return 1; }
    c->i=save; return 0;                             /* `enum tag` as a type -> a function follows */
  }
  { /* a struct *definition*?  [storage/qualifiers] struct [attrs] [TAG] [attrs] {  -- lookahead past them. A
     * declaration of objects may define the struct it declares them of, `static struct t { ... } a, b;` (C11
     * 6.7.2.1): the definition, then its declarators, globals (the oracle's `_aggregate_definition`) */
    int save=c->i, vol=0;
    while(is(c,"static")||is(c,"extern")||is(c,"const")||is(c,"volatile")||is(c,"_Thread_local")||is(c,"thread_local")){
      if(is(c,"volatile")) vol=1;
      c->i++; }
    if(is(c,"struct")||is(c,"union")){
      int kw=c->i; c->i++; int pk_=0,al_=0; attrs(c,&pk_,&al_);
      tok tag={T_END,"",0,0}; if(isk(c,T_ID)&&!is(c,"{")) tag=adv(c);   /* the tag */
      attrs(c,&pk_,&al_);
      if(is(c,"{")){
        c->i=kw; int my=p_struct_body(c);
        if(is(c,";")||c->failed||my<0){ eat(c,";"); return 1; }
        if(tag.k!=T_ID){ fail(c,"an object of an untagged struct or union at file scope is not supported"); return 1; }
        bcir_ctype base; memset(&base,0,sizeof base); base.kind=1; base.signd=1; base.size=c->s[my].size;
        base.is_union=(uint8_t)c->s[my].is_union; base.is_volatile=(uint8_t)vol; idcpy(c,base.tag,&tag);
        for(;;){ p_global_declarator(c,&base,my,0,NULL); if(c->failed || !is(c,",")) break; c->i++; }
        eat(c,";"); return 1; }
    }
    c->i=save;
  }
  if(is(c,"struct")||is(c,"union")){
    if(tat(c,c->i+1)->k==T_ID && tok_is(tat(c,c->i+2),";")){   /* `struct tag;` -- a forward declaration */
      int isu=is(c,"union"); c->i++; tok tag=adv(c); if(declare_struct(c,&tag,isu)<0) return 1;
      eat(c,";"); return 1; }
    return 0;                                        /* struct used as a type -> a function follows */
  }
  return 0;
}

/* Lookahead: does the current top-level token begin a file-scope global (a `TYPE NAME ...` that is
 * NOT followed by `(`) rather than a function?  Restores the cursor so the caller re-parses. */
static int looks_global(CC *c){
  int save=c->i, sf=c->failed; bcir_ctype ty; int si; int global=0;
  if(!p_type(c,&ty,&si)){
    if(isk(c,T_ID)){ c->i++; if(!is(c,"(")) global=1; }
    else if(fp_decl_at(c,0)) global=1; }   /* `RET (*g)(P) ...;`: a function-pointer global (CF-FPTAB) */
  c->i=save; c->failed=sf; c->err[0]=0;
  return global;
}
/* The `(` at the cursor opens a call's arguments (the oracle's parser reads a call there): it follows a name that is
 * no type name, `sizeof`, `_Alignof` or `_Generic`, a subscript, or a parenthesized expression that is no cast's type
 * -- `(f)(x)`, `(*fp)(x)` -- where `(T)(x)` casts a parenthesized operand. */
static int init_call_at(CC *c){
  const tok *pv=tat(c,c->i-1);
  if(pv->k==T_ID) return !tok_is(pv,"sizeof") && !tok_is(pv,"_Alignof") && !tok_is(pv,"alignof")
                         && !tok_is(pv,"_Generic") && !decl_type_tok(c,pv);
  if(tok_is(pv,"]")) return 1;
  if(!tok_is(pv,")")) return 0;
  int j=c->i-1;
  for(int d=0; j>=0; j--){ if(tok_is(tat(c,j),")")) d++; else if(tok_is(tat(c,j),"(") && --d==0) break; }
  return j>=0 && !decl_type_tok(c,tat(c,j+1));
}
/* Skip the file-scope initializer expression at the cursor -- to the `,`, `;` or `}` that ends it at its own depth
 * -- without lowering it (the oracle folds it only for its linkable emit, which renders the globals). A constant
 * expression it is in C (C11 6.7.9p4), of any form -- a cast, `sizeof`, a floating or string constant, an address;
 * it had been read by the enum evaluator, which refused all of those (`non-constant enum initializer`). A call in it
 * is refused, as the oracle refuses it (`init_call_at`). `nm` is the global. */
static void skip_init_expr(CC *c, const tok *nm){
  for(int d=0; !isk(c,T_END) && !c->failed; c->i++){
    if(d==0 && (is(c,",")||is(c,";")||is(c,"}"))) break;
    if(is(c,"(") && c->i>0 && init_call_at(c)){
      char m[BCIR_CIR_NAME+80];
      snprintf(m,sizeof m,"file-scope initializer of '%.*s' calls a function (not a constant expression)",
               nm->n<BCIR_CIR_IDENT_MAX?nm->n:BCIR_CIR_IDENT_MAX,nm->s);
      fail(c,m); return; }
    if(is(c,"(")||is(c,"[")||is(c,"{")) d++;
    else if(is(c,")")||is(c,"]")||is(c,"}")) d--;
  }
}
/* A file-scope initializer at the cursor, walked as C walks the current object, for its shape only (CF-GBRACE; the
 * oracle's `_file_scope_shape`): an entry is skipped, neither lowered nor folded -- the source defines the global and
 * the emit names it -- but the walk's constraints hold as for a local or a static (C11 6.7.9p2, p14, p17-19: an excess
 * entry, an override, a string too long, a designator outside its object are refused), and an unsized array takes the
 * extent the walk reaches: `struct pt g[] = {1u, 2u, 3u, 4u}` is two elements, not four. A call in it is refused
 * first, as the oracle's parser refuses it. `ty` / `si`: the declarator's type (an array's element) and its struct;
 * `dims[0..nd)`: an array's dimensions, dims[0] 0 when unsized. The cursor ends past the initializer. Returns the
 * top-level elements an unsized array's walk reached (at least one), else -1. */
static long long global_init_shape(CC *c, const tok *nm, const bcir_ctype *ty, int si, int is_arr, int nd,
                                   const long long *dims){
  int start=c->i; skip_init_expr(c,nm);                /* to the initializer's end: a call in it is refused */
  if(c->failed) return -1;
  int end=c->i; c->i=start;
  int inferred=is_arr && dims[0]==0;
  iunit obj;
  if(is_arr){
    if(nd>3 || !(is(c,"{") || (nd==1 && isk(c,T_STR)))){   /* the twin's walk holds three dimensions, as a local's */
      fail(c,is(c,"{") && nd>3 ? "an initialized file-scope array of more than 3 dimensions is not supported"
                               : "an array is initialized by a brace list or a string literal");
      return -1; }
    int d3[3]={0,0,0};
    for(int d=0; d<nd; d++) d3[d]=dims[d]>0 && dims[d]<=INT_MAX ? (int)dims[d] : 0;
    if(inferred) d3[0]=1;                              /* provisional: the walk counts the rows */
    obj=iunit_array(ty,si,ty->kind==2?cc_abi(c)->pointer_size:ty->size,d3,nd); }
  else if(!is(c,"{")){ c->i=end; return -1; }         /* a scalar's or a struct's expression: nothing to walk */
  else if(ty->kind==1){ if(si<0){ fail(c,"unknown struct"); return -1; } obj=iunit_struct(c,si); }
  else obj=iunit_scalar(c,ty);
  iwalk W; memset(&W,0,sizeof W);
  W.top_array=obj.k==IK_ARR; W.inferred=inferred; W.konst=1; W.skip=1; W.gname=nm;
  W.rng0=c->iw_nrng; W.un0=c->iw_nun;
  if(isk(c,T_STR)){                                    /* `char s[] = "ab"` is `char s[] = {"ab"}` (6.7.9p14) */
    if(!init_char_array(&obj,str_tok_unit(c,pk(c)))) fail(c,"an array is initialized by a brace list or a string literal");
    else init_string(c,&W,&obj,1,inferred?-1:obj.dims[0]); }
  else if(obj.k!=IK_SCAL) init_list(c,&W,&obj,1,obj.k==IK_ARR);
  else { ifrm fr; memset(&fr,0,sizeof fr); init_sublist(c,&W,&fr,&obj); }   /* a braced scalar `T g = {e};` */
  c->iw_nrng=W.rng0; c->iw_nun=W.un0;                  /* its records end with it */
  if(c->failed) return -1;
  return inferred ? (W.top_n<1?1:W.top_n) : -1;
}
/* One declarator of a file-scope declaration off the specifier `base` -- its `*`s, name, dimensions and initializer --
 * registered as a global. The emit names a global, which the source defines, so its initializer is walked only for
 * its shape (`global_init_shape`). `si`: the specifier's struct; `btd` / `btdd`: a typedef'd array specifier's
 * dimensions, inner to the declarator's. */
static void p_global_declarator(CC *c, const bcir_ctype *base, int si, int btd, const int *btdd){
  if(btd && is(c,"*")){ fail(c,"a pointer to a typedef'd array is not supported"); return; }
  bcir_ctype ty=*base; apply_stars(c,&ty);
  if(c->failed) return;
  tok nm; int gpre_nd=0, gpre_dims[3]={0,0,0};
  if(fp_decl_at(c,0)){                /* `RET (*g)(P)`, a table `RET (*t[N])(P)`, `RET (**pp)(P)` (CF-FPTAB) */
    bcir_ctype fty; int stars;
    if(fp_inline_decl(c,&ty,si,&fty,&stars,gpre_dims,&gpre_nd,&nm)) return;
    ty=fty; for(int s=0;s<stars;s++) fp_star(c,&ty); }
  else { if(!isk(c,T_ID)){ fail(c,"expected a declarator"); return; } nm=adv(c); }
  int count=1, is_arr=0, init_a=0, init_b=0, nd=0; long long dims[4]={0,0,0,0}, init_n=-1;
  for(int d=0; d<gpre_nd; d++){ count=gpre_dims[d]; is_arr=1; if(nd<4){ dims[nd]=count; } nd++; }   /* its own dims */
  while(is(c,"[")){ c->i++; long long d=is(c,"]") ? 0 : ce_dim(c,0);   /* an integer constant expression */
    if(d<0) return;
    count=(int)d; eat(c,"]"); is_arr=1;
    if(nd<4){ dims[nd]=count; } nd++; }
  for(int d=0; d<btd; d++){ count=btdd[d]; is_arr=1; if(nd<4){ dims[nd]=count; } nd++; }   /* its dims follow */
  if(is(c,"=")){ c->i++; init_a=c->i;
    /* the initializer walked for its shape (CF-GBRACE): a brace list's -- nested, elided, designated -- or a character
     * array's string literal; an unsized array takes the extent the walk reaches */
    init_n=global_init_shape(c,&nm,&ty,ty.kind==1?si:-1,is_arr,nd,dims);
    if(c->failed) return;
    init_b=c->i;
  }
  if(is_arr && dims[0]==0 && init_n>=0){ dims[0]=init_n; if(nd==1) count=(int)init_n; }   /* `T g[] = ...` / `T g[][B]
                                                      * = ...`: its initializer's extent -- the count a one-dimensional
                                                      * array's accesses are bounded by too, the rows of a nested one */
  CC_ENSURE(c, c->gv, c->ngv, c->cap_gv);
  if(c->ngv<c->cap_gv){ gvar *g=&c->gv[c->ngv++]; idcpy(c,g->name,&nm); g->ty=ty; g->count=count;
    g->is_arr=is_arr; g->init_a=init_a; g->init_b=init_b;
    g->nd=nd; for(int d=0;d<4;d++) g->dims[d]=dims[d]; }
}
/* A file-scope declaration: each declarator off the one specifier a global of its own type -- `T a[3], *p, b = 5;`
 * (C11 6.7p1), a declarator's `*`s its own (the oracle's `_globals`). */
static void p_global(CC *c){
  bcir_ctype base; int si; if(p_type_base(c,&base,&si)) return;
  int btdd[3], btd=td_dims_of(c,btdd);   /* a typedef'd array type */
  for(;;){
    p_global_declarator(c,&base,si,btd,btdd);
    if(c->failed || !is(c,",")) break;
    c->i++;
  }
  eat(c,";");
}

/* --- public entry -------------------------------------------------------- */
static void cfront_free_func(const bcir_host_allocator *allocator, bcir_func *fn) {
  if(!fn) return;
  for(int i=0;i<fn->n_host_literals;i++)
    bcir_host_deallocate(allocator,fn->host_literals[i].spelling);
  for(int i=0;i<fn->n_statics;i++)
    bcir_host_deallocate(allocator,fn->statics[i].text);
  bcir_host_deallocate(allocator,fn->res);
  bcir_host_deallocate(allocator,fn->claims);
  bcir_host_deallocate(allocator,fn->params);
  bcir_host_deallocate(allocator,fn->calls);
  bcir_host_deallocate(allocator,fn->statics);
  bcir_host_deallocate(allocator,fn->host_literals);
  bcir_host_deallocate(allocator,fn->ptr_extents);
  bcir_host_deallocate(allocator,fn->qcasts);
  memset(fn,0,sizeof *fn);
}

void bcir_cfront_result_init(bcir_cfront_result *out) {
  if(out) memset(out,0,sizeof *out);
}

int bcir_cfront_context_init(bcir_cfront_context *context,
                             const bcir_host_allocator *allocator) {
  bcir_host_allocator selected;
  CC *state;
  if(!context) return 1;
  memset(context,0,sizeof *context);
  selected=bcir_host_allocator_or_default(allocator);
  state=(CC *)bcir_host_allocate(&selected,sizeof *state);
  if(!state) return 1;
  memset(state,0,sizeof *state);
  state->allocator=selected;
  (void)bcir_host_arena_init(&state->scratch,&selected,16384u);
  context->allocator=selected;
  context->state=state;
  return 0;
}

void bcir_cfront_context_reset(bcir_cfront_context *context) {
  CC *c;
  bcir_host_allocator allocator;
  bcir_host_arena scratch;
  sdef *s; tdef *td; econst *ec; gvar *gv; venv *env; void *protos; irange *iw_rng; iunion *iw_un; int *pure_memo;
  int cap_s,cap_td,cap_ec,cap_gv,cap_env,cap_protos,iw_caprng,iw_capun,cap_pure;
  ctext fpdefs,tudefs,anontext;
  if(!context||!context->state) return;
  c=(CC *)context->state;
  bcir_host_arena_reset(&c->scratch);
  allocator=c->allocator; scratch=c->scratch;
  s=c->s;cap_s=c->cap_s;td=c->td;cap_td=c->cap_td;ec=c->ec;cap_ec=c->cap_ec;
  gv=c->gv;cap_gv=c->cap_gv;env=c->env;cap_env=c->cap_env;
  protos=c->protos;cap_protos=c->cap_protos;
  iw_rng=c->iw_rng;iw_caprng=c->iw_caprng;iw_un=c->iw_un;iw_capun=c->iw_capun;
  pure_memo=c->pure_memo;cap_pure=c->cap_pure;
  fpdefs=c->fpdefs;tudefs=c->tudefs;anontext=c->anontext;
  memset(c,0,sizeof *c);
  c->allocator=allocator;c->scratch=scratch;
  c->s=s;c->cap_s=cap_s;c->td=td;c->cap_td=cap_td;c->ec=ec;c->cap_ec=cap_ec;
  c->gv=gv;c->cap_gv=cap_gv;c->env=env;c->cap_env=cap_env;
  c->protos=protos;c->cap_protos=cap_protos;
  c->iw_rng=iw_rng;c->iw_caprng=iw_caprng;c->iw_un=iw_un;c->iw_capun=iw_capun;
  c->pure_memo=pure_memo;c->cap_pure=cap_pure;
  c->fpdefs.s=fpdefs.s;c->fpdefs.cap=fpdefs.cap;c->tudefs.s=tudefs.s;c->tudefs.cap=tudefs.cap;   /* kept, emptied */
  c->anontext.s=anontext.s;c->anontext.cap=anontext.cap;
  if(c->fpdefs.s) c->fpdefs.s[0]=0;
  if(c->tudefs.s) c->tudefs.s[0]=0;
}

void bcir_cfront_context_destroy(bcir_cfront_context *context) {
  CC *c;
  bcir_host_allocator allocator;
  if(!context||!context->state){if(context)memset(context,0,sizeof *context);return;}
  c=(CC *)context->state; allocator=c->allocator;
  bcir_host_arena_destroy(&c->scratch);
  bcir_host_deallocate(&allocator,c->s);bcir_host_deallocate(&allocator,c->td);
  bcir_host_deallocate(&allocator,c->ec);bcir_host_deallocate(&allocator,c->gv);
  bcir_host_deallocate(&allocator,c->env);bcir_host_deallocate(&allocator,c->protos);
  bcir_host_deallocate(&allocator,c->iw_rng);bcir_host_deallocate(&allocator,c->iw_un);
  bcir_host_deallocate(&allocator,c->pure_memo);
  bcir_host_deallocate(&allocator,c->fpdefs.s);bcir_host_deallocate(&allocator,c->tudefs.s);
  bcir_host_deallocate(&allocator,c->anontext.s);
  memset(c,0,sizeof *c);bcir_host_deallocate(&allocator,c);
  memset(context,0,sizeof *context);
}

/* The result's verified C grows (CF-BUF): each piece is measured, the block grown two-phase through the
 * result's allocator, then the piece rendered. The text has no fixed capacity, and it is never partial: a
 * failed growth leaves the block as it was and fails the compile. `emit_room` makes room for `need` more
 * bytes and a NUL. */
static int emit_room(bcir_cfront_result *out, size_t need){
  size_t total, next;
  if(!bcir_size_add(out->emitted_len,need,&total) || !bcir_size_add(total,1u,&total)) return 0;
  if(total<=out->_emitted_cap) return 1;
  if(!bcir_host_grow_capacity(out->_emitted_cap?out->_emitted_cap:4096u,total,1u,&next) ||
     !bcir_host_realloc_array(&out->_allocator,(void **)&out->emitted,out->_emitted_cap,next,1u,0)) return 0;
  out->_emitted_cap=next;
  return 1;
}
static const char *const emit_oom="oom", *const emit_unrenderable="cannot render emitted C";
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf,2,3)))
#endif
static const char *emit_putf(bcir_cfront_result *out, const char *fmt, ...){
  va_list ap; int k;
  va_start(ap,fmt); k=vsnprintf(NULL,0,fmt,ap); va_end(ap);
  if(k<0) return emit_unrenderable;
  if(!emit_room(out,(size_t)k)) return emit_oom;
  va_start(ap,fmt);
  (void)vsnprintf(out->emitted+out->emitted_len,out->_emitted_cap-out->emitted_len,fmt,ap);
  va_end(ap);
  out->emitted_len+=(size_t)k;
  return NULL;
}
/* One function's C, rendered into the room left and -- when it needs more -- once more after the block holds it:
 * emit_func writes nothing past its bound and returns the whole length, and it is a pure function of the claims,
 * so the second rendering is the first one whole. */
static const char *emit_function(bcir_cfront_result *out, const bcir_func *f){
  size_t room=out->_emitted_cap-out->emitted_len;
  size_t n=emit_func(f,out->emitted+out->emitted_len,room);
  if(n>=room){
    if(!emit_room(out,n)) return emit_oom;
    room=out->_emitted_cap-out->emitted_len;
    if(emit_func(f,out->emitted+out->emitted_len,room)!=n) return emit_unrenderable;
  }
  out->emitted_len+=n;
  return NULL;
}
/* The forward declaration of each function of the unit `f` calls, before `f`: a callee defined after its caller is
 * otherwise undeclared at the call, which does not compile (the oracle's `emit_function`, given the unit). Each
 * callee once, in the order of its first call; `f` itself and a callee the unit does not define are not declared. */
static const char *emit_callee_decls(bcir_cfront_result *out, const bcir_func *f){
  for(int k=0;k<f->n_calls;k++){
    const char *nm=f->calls[k]; int seen=!strcmp(nm,f->name);
    for(int j=0;j<k && !seen;j++) seen=!strcmp(f->calls[j],nm);
    const bcir_func *g=NULL;
    for(int j=0;j<out->unit.n_funcs && !seen && !g;j++) if(!strcmp(out->unit.funcs[j].name,nm)) g=&out->unit.funcs[j];
    if(!g) continue;
    size_t room=out->_emitted_cap-out->emitted_len;
    size_t n=emit_sig(g,out->emitted+out->emitted_len,room);   /* as emit_function renders: whole, or again */
    if(n>=room){
      if(!emit_room(out,n)) return emit_oom;
      room=out->_emitted_cap-out->emitted_len;
      if(emit_sig(g,out->emitted+out->emitted_len,room)!=n) return emit_unrenderable;
    }
    out->emitted_len+=n;
    const char *e=emit_putf(out,";\n");
    if(e) return e;
  }
  return NULL;
}
/* The C spelling of the anonymous aggregate `si` appended to `t` -- 1, or 0 (nothing appended) when C cannot name
 * it: the name a file-scope typedef gives it (`typedef struct {...} P;`, spelled `P`; `typedef struct {...} *PP;`,
 * spelled `__typeof__(*(PP)0)`, a null pointer's pointee), or the type of the named
 * member it is -- `struct {...} m[N];` in `P`, spelled `__typeof__(((P *)0)->m[0])`, a `*` before the member for each
 * pointer level -- through the spelling of the aggregate that holds the member, up to one a typedef names or that
 * has a tag. The chain is walked without recursion (the oracle's `anon_spelling`). */
static int anon_spelling(CC *c, int si, ctext *t){
  int top=si, k=0;
  while(!c->s[top].tdname[0] && !c->s[top].tdptr[0] && is_anon_tag(c->s[top].tag)){   /* up to a named one */
    if(c->s[top].parent<0) return 0;
    top=c->s[top].parent; k++;
  }
  if(!k && !c->s[si].tdname[0] && !c->s[si].tdptr[0]) return 0;   /* a tagged aggregate: `struct tag` names it */
  for(int x=si; x!=top; x=c->s[x].parent){
    ctext_putf(c,t,"__typeof__(");
    for(int st=0; st<c->s[x].mstars; st++) ctext_putf(c,t,"*");
    ctext_putf(c,t,"((");
  }
  if(c->s[top].tdname[0]) ctext_putf(c,t,"%s",c->s[top].tdname);
  else if(c->s[top].tdptr[0]){                           /* only a pointer typedef names it: a null one's pointee */
    ctext_putf(c,t,"__typeof__(");
    for(int st=0; st<c->s[top].tdstars; st++) ctext_putf(c,t,"*");
    ctext_putf(c,t,"(%s)0)",c->s[top].tdptr); }
  else ctext_putf(c,t,"%s %s",c->s[top].is_union?"union":"struct",c->s[top].tag);
  for(int d=k-1; d>=0; d--){                             /* the members, innermost aggregate first */
    int x=si; for(int j=0;j<d;j++) x=c->s[x].parent;
    ctext_putf(c,t," *)0)->%s",c->s[x].member);
    for(int dd=0; dd<c->s[x].mdims; dd++) ctext_putf(c,t,"[0]");
    ctext_putf(c,t,")");
  }
  return 1;
}
/* The anonymous aggregate a `struct $anonN` / `union $anonN` at `p` (of `n` bytes) names, its end in `*end`; -1 if
 * none starts there. */
static int anon_ref_at(CC *c, const char *p, size_t n, size_t *end){
  size_t kw = n>=7 && !memcmp(p,"struct ",7) ? 7 : n>=6 && !memcmp(p,"union ",6) ? 6 : 0;
  if(!kw || n-kw<6 || memcmp(p+kw,"$anon",5)) return -1;
  size_t j=kw+5; while(j<n && p[j]>='0' && p[j]<='9') j++;
  if(j==kw+5 || (j<n && (is_idc((unsigned char)p[j]) || p[j]=='$'))) return -1;
  int si=find_struct(c,p+kw,(int)(j-kw));
  if(si<0) return -1;
  *end=j; return si;
}
/* Every `struct $anonN` / `union $anonN` of the emitted unit -- outside a string or character literal and a comment
 * -- spelled as C names it (`anon_spelling`): the synthesized tag is no compiler's, so an emit using a typedef'd
 * anonymous struct did not compile (CF-ANON; the oracle's `respell_anon`). One C cannot name stays as it was. */
static const char *respell_anon(CC *c, bcir_cfront_result *out){
  int any=0;
  for(int i=0;i<c->ns && !any;i++)
    any = is_anon_tag(c->s[i].tag) && (c->s[i].tdname[0] || c->s[i].tdptr[0] || c->s[i].parent>=0);
  if(!any || !out->emitted) return NULL;
  ctext *t=&c->anontext; t->w=0;
  const char *src=out->emitted; size_t n=out->emitted_len, run=0;
  for(size_t i=0;i<n;){
    if(src[i]=='"' || src[i]=='\''){                    /* a literal: part of the run, untouched */
      size_t j=i+1; while(j<n && src[j]!=src[i]) j += (src[j]=='\\' && j+1<n) ? 2 : 1;
      i = j<n ? j+1 : n; continue; }
    if(src[i]=='/' && i+1<n && src[i+1]=='*'){         /* a comment: likewise */
      size_t j=i+2; while(j+1<n && !(src[j]=='*' && src[j+1]=='/')) j++;
      i = j+1<n ? j+2 : n; continue; }
    if(src[i]=='/' && i+1<n && src[i+1]=='/'){
      while(i<n && src[i]!='\n') i++;
      continue; }
    size_t e; int si;
    if((i==0 || !(is_idc((unsigned char)src[i-1]) || src[i-1]=='$')) && (si=anon_ref_at(c,src+i,n-i,&e))>=0){
      size_t w0=t->w;
      ctext_putn(c,t,src+run,i-run);
      if(anon_spelling(c,si,t)){ i+=e; run=i; continue; }
      t->w=w0;                                          /* C cannot name it: the run goes on */
    }
    i++;
  }
  ctext_putn(c,t,src+run,n-run);
  if(t->w>out->emitted_len && !emit_room(out,t->w-out->emitted_len)) return emit_oom;
  memcpy(out->emitted,t->s,t->w); out->emitted_len=t->w; out->emitted[t->w]=0;
  return NULL;
}
/* The whole unit's verified C: the C.2 attestation, the preludes, then every function. NULL on success, else
 * the failure's diagnostic. */
static const char *emit_unit(CC *c, bcir_cfront_result *out, const bcir_func *entry, const char *lflags){
  const char *e;
  /* C.2 verified-C attestation: stamp the emitted C with its R-law status + R13 digest + the unit's derived
   * link flags (B1; so --emit-c is self-describing about what it links -- a comment, stripped on re-parse).
   * The link_flags line mirrors the oracle's C.2 attestation. */
  if((e=emit_putf(out,
    "/* BCIR verified-C attestation (C.2) -- generated by bcir_cfront, do not edit.\n"
    " *   R1-R8 + R18  %s\n"
    " *   R9 plan / R10-R11 pack  checked in the compile->execute loop\n"
    " *   R12 lowering-contract  support preserved (emit Clang-behaviour-equivalent)\n"
    " *   R13 provenance digest  %016llx\n"
    " *   R17 accuracy  exact (integer / Q-fixed, 0 ULP)\n"
    " *   link_flags  %s\n */\n",
    out->ok?"clean":"DIRTY", entry?(unsigned long long)bcir_provenance_digest(entry):0ull,
    lflags[0]?lflags:"-"))) return e;
  if(c->fpdefs.w && (e=emit_putf(out,"%s",c->fpdefs.s))) return e;   /* synthesized funcptr-param typedefs */
  if(c->tudefs.w && (e=emit_putf(out,"%s",c->tudefs.s))) return e;   /* extern declarations, cross-TU callees */
  for(int i=0;i<out->unit.n_funcs;i++){
    if((e=emit_callee_decls(out,&out->unit.funcs[i]))) return e;
    if((e=emit_function(out,&out->unit.funcs[i]))) return e;
    if(i+1<out->unit.n_funcs && (e=emit_putf(out,"\n"))) return e;
  }
  return respell_anon(c,out);
}

static int cfront_failure(bcir_cfront_context *context, bcir_cfront_result *out,
                          const char *message) {
  char diagnostic[sizeof out->diag];
  snprintf(diagnostic,sizeof diagnostic,"%s",message&&message[0]?message:"compile failed");
  if(context&&context->state)((CC *)context->state)->jump_active=0;
  bcir_cfront_free(out);
  out->ok=0;out->emitted_ok=0;
  snprintf(out->diag,sizeof out->diag,"%s",diagnostic);
  if(context&&context->state)
    bcir_host_arena_reset(&((CC *)context->state)->scratch);
  return 1;
}

int bcir_cfront_compile_target_context(bcir_cfront_context *context,
                                       const char *src, const char *target,
                                       bcir_cfront_result *out) {
  CC *c;
  if(!out) return 1;
  if(!context||!context->state){
    bcir_cfront_result_init(out);
    snprintf(out->diag,sizeof out->diag,"uninitialized cfront context");
    return 1;
  }
  bcir_cfront_free(out);
  bcir_cfront_result_init(out);
  bcir_cfront_context_reset(context);
  c=(CC *)context->state;
  out->_allocator=c->allocator;out->_owner_tag=0x42434652u;
  if(!src) return cfront_failure(context,out,"invalid source");
  c->abi = bcir_abi_by_name(target);     /* the target data model (NULL name -> host LP64) */
  if(!c->abi){
    char diagnostic[256];
    snprintf(diagnostic,sizeof diagnostic,"unknown target '%s'",target?target:"");
    return cfront_failure(context,out,diagnostic);
  }
  c->rid=100; c->cid=1000; c->unit=&out->unit;
  if(setjmp(c->failure_jump)){
    c->jump_active=0;
    if(out->unit.funcs&&out->unit.n_funcs<out->unit.cap_funcs)
      cfront_free_func(&c->allocator,&out->unit.funcs[out->unit.n_funcs]);
    return cfront_failure(context,out,c->err);
  }
  c->jump_active=1;
  lex(c,src);
  if(c->tok_overflow){                     /* Bug B: more than MAXTOK tokens -> a clean fail (fallback), NOT
                                           * a silent truncation + partial mis-compile. The oracle has no
                                           * token cap but its recursion/size guards route the same oversized
                                           * input to fallback, so both rails agree (neither mis-compiles). */
    return cfront_failure(context,out,"input too large"); }
  while(!isk(c,T_END)&&!c->failed){       /* no fixed function ceiling -- the unit list grows */
    c->nenv=0;                              /* file scope binds no block-scope name: a function's parameters, left in
                                             * `env` after its body, must not hide an enumerator a later file-scope
                                             * declaration reads (`visible_enum`, CF-ENUMSCOPE) */
    /* A1.3: a leading C23 `[[unsequenced]]`/`[[reproducible]]` (or any `[[...]]`) attribute precedes a
     * function/global. Consume it here and carry the value-neutral hint flag into the next p_func (the
     * emit drops it). No run -> repro stays 0, every existing item undisturbed. */
    int lead_repro=0; c23_attrs(c,&lead_repro);
    if(c->failed) break;
    if(try_top_decl(c)) continue;       /* typedef / enum / struct|union defs, interleaved */
    if(isk(c,T_END)||c->failed) break;
    if(looks_global(c)){ p_global(c); continue; }   /* a file-scope global (lookup table) */
    if(!CC_ENSURE(c,out->unit.funcs,out->unit.n_funcs,out->unit.cap_funcs))
      return cfront_failure(context,out,c->err);
    bcir_func *fn=&out->unit.funcs[out->unit.n_funcs]; /* res/claims/params/calls/statics grow lazily */
    c->rid=100+out->unit.n_funcs*1000; c->cid=1000+out->unit.n_funcs*1000;
    int pfr=p_func(c,fn);
    if(pfr==2){                             /* a PROTOTYPE: recorded in c->protos/c->tudefs, no function --
                                             * discard the scratch fn (params were collected into it) */
      cfront_free_func(&c->allocator,fn);
      continue;
    }
    if(pfr){
      /* free the in-progress (uncounted) func's sub-arrays: it was never folded into n_funcs, so
       * bcir_cfront_free's `i<n_funcs` loop would never reach it -> a per-parse-failure leak. */
      cfront_free_func(&c->allocator,fn);
      return cfront_failure(context,out,c->err);
    }
    fn->reproducible=(uint8_t)lead_repro;   /* the C23 hint consumed just above (A1.3); emit drops it */
    out->unit.n_funcs++;
  }
  if(c->failed)return cfront_failure(context,out,c->err);
  /* A call made before any declaration of its callee: C99 dropped the implicit declaration (C11 6.5.1p2), and
   * the call was typed by a default no declaration gives -- a `uint64_t` callee's result read as `uint32_t`. A
   * callee the unit defines or prototypes after the call is refused, in call order (the oracle refuses the
   * first such call it lowers); one the unit never declares stays R18's undefined edge. */
  for(const struct cc_undecl *u=c->undecl; u; u=u->next){
    int later=0;
    for(int j=0;j<out->unit.n_funcs && !later;j++) later=!strcmp(out->unit.funcs[j].name,u->name);
    for(int k=0;k<c->n_protos && !later;k++) later=!strcmp(c->protos[k].name,u->name);
    if(later){ char msg[BCIR_CIR_NAME+40];
      snprintf(msg,sizeof msg,"call to undeclared function '%s'",u->name);
      return cfront_failure(context,out,msg); }
  }
  /* Phase 3 linking, DEFINITION WINS: a call lowered `c.call.tu:` (its callee was only a prototype at
   * the call site -- the parser is single-pass) whose callee IS defined in this unit is an ordinary
   * in-unit call after all: rewrite the op back (`c.call:` / `c.call.void:` by result arity) and record
   * the R18 edge. The extern declaration already rendered stays in the prelude: it declares the
   * unprefixed external name, which the emitted unit neither defines nor references -- harmless. */
  for(int i=0;i<out->unit.n_funcs;i++){ bcir_func *f=&out->unit.funcs[i];
    for(size_t k2=0;k2<f->n_claims;k2++){ bcir_claim *cl=&f->claims[k2];
      if(strncmp(cl->op,"c.call.tu:",10)) continue;
      int def=-1;
      for(int j=0;j<out->unit.n_funcs;j++)
        if(!strcmp(out->unit.funcs[j].name,cl->op+10)){def=j;break;}
      if(def<0) continue;                              /* genuinely cross-TU: the linker's job */
      char callee[BCIR_CIR_NAME];                      /* a defined function's name: it fits (CF-BUF) */
      if(!fits(c,callee,sizeof callee,"%s",cl->op+10) ||
         !fits(c,cl->op,sizeof cl->op,"%s%s",cl->n_wr?"c.call:":"c.call.void:",callee))
        return cfront_failure(context,out,c->err);
      cl->qcast=0;                                     /* the emit's definition spells no qualifier (CF-QUALS) */
      if(!CC_ENSURE(c,f->calls,f->n_calls,f->cap_calls))
        return cfront_failure(context,out,c->err);
      memcpy(f->calls[f->n_calls++],callee,sizeof callee);
    }
  }
  null_pointer_args(c,&out->unit);   /* a null pointer constant argument is its parameter's pointer (CF-NULLARG) */
  if(c->failed) return cfront_failure(context,out,c->err);   /* ... and a struct argument refused (CF-STRUCTARITH) */
  /* G10: a function a file-scope initializer names (an ops table `struct ops t = { handler };`) has its
   * address taken -- callers this unit cannot see may reach it. Every identifier in the initializer counts
   * except a designator's field name (`.fn = ...`); the oracle's twin is LoweredUnit.init_refs. */
  for(int g=0;g<c->ngv;g++) for(int k=c->gv[g].init_a;k<c->gv[g].init_b && k<c->nt;k++){
    const tok *tk=&c->t[k];
    if(tk->k!=T_ID) continue;
    if(k>0 && (tok_is(&c->t[k-1],".") || tok_is(&c->t[k-1],"->"))) continue;
    for(int i=0;i<out->unit.n_funcs;i++){ bcir_func *f=&out->unit.funcs[i];
      if((int)strlen(f->name)==tk->n && !strncmp(f->name,tk->s,(size_t)tk->n)) f->addr_in_init=1; }
  }
  out->ok=bcir_verify_unit_with_allocator(&out->unit,out->diag,sizeof out->diag,&c->allocator);
  if(!out->ok&&!strcmp(out->diag,"oom"))return cfront_failure(context,out,"oom");
  const bcir_func *entry = out->unit.n_funcs ? &out->unit.funcs[out->unit.n_funcs-1] : NULL;
  char lflags[256]; bcir_cfront_link_flags(&out->unit, lflags, sizeof lflags);
  const char *unemitted=emit_unit(c,out,entry,lflags);   /* the verified C, whole or not at all (CF-BUF) */
  if(unemitted)return cfront_failure(context,out,unemitted);
  out->emitted_ok=1;
  c->jump_active=0;
  bcir_host_arena_reset(&c->scratch);
  return 0;
}

void bcir_cfront_free(bcir_cfront_result *out){
  bcir_host_allocator allocator;
  if(!out)return;
  if(out->_owner_tag!=0x42434652u||!bcir_host_allocator_valid(&out->_allocator)){
    memset(out,0,sizeof *out);return;
  }
  allocator=out->_allocator;
  for(int i=0;i<out->unit.n_funcs;i++)cfront_free_func(&allocator,&out->unit.funcs[i]);
  bcir_host_deallocate(&allocator,out->unit.funcs);
  bcir_host_deallocate(&allocator,out->emitted);
  memset(out,0,sizeof *out);
}

int bcir_cfront_compile_context(bcir_cfront_context *context, const char *src,
                                bcir_cfront_result *out) {
  return bcir_cfront_compile_target_context(context,src,NULL,out);
}

static bcir_cfront_context *cfront_legacy_context(bcir_cfront_result *out) {
  static bcir_cfront_context context;
  static int initialized;
  if(!initialized){
    if(bcir_cfront_context_init(&context,NULL)){
      if(out){bcir_cfront_result_init(out);snprintf(out->diag,sizeof out->diag,"oom");}
      return NULL;
    }
    initialized=1;
  }
  return &context;
}

int bcir_cfront_compile_target(const char *src, const char *target,
                               bcir_cfront_result *out) {
  bcir_cfront_result_init(out);
  bcir_cfront_context *context=cfront_legacy_context(out);
  return context?bcir_cfront_compile_target_context(context,src,target,out):1;
}

/* The host-ABI entry (the default target): byte-identical to the layout before --target existed. */
int bcir_cfront_compile(const char *src, bcir_cfront_result *out) {
  bcir_cfront_result_init(out);
  bcir_cfront_context *context=cfront_legacy_context(out);
  return context?bcir_cfront_compile_context(context,src,out):1;
}

void bcir_cfront_summary(const bcir_unit *u,int ok,char *buf,size_t n){
  if(!buf||!n)return;
  if(!u||u->n_funcs<0||(u->n_funcs&& !u->funcs)){snprintf(buf,n,"invalid unit");return;}
  const bcir_func *f = u->n_funcs ? &u->funcs[u->n_funcs-1] : NULL;   /* the entry (last) */
  int mmio=0,bf=0,kn=0,binop=0,calls=0; size_t nc=0;
  if(f){
    for(size_t i=0;i<f->n_claims;i++){const bcir_claim *cl=&f->claims[i];
      if(cl->opcode==BCIR_OP_NOP)continue;       /* control-flow markers are not real claims */
      nc++;
      if(!strcmp(cl->op,"c.load")&&cl->domain==BCIR_DOM_MMIO)mmio++;
      else if(!strcmp(cl->op,"c.bf.get"))bf++;
      else if(!strcmp(cl->op,"c.const"))kn++;
      else if(!strncmp(cl->op,"c.bin.",6))binop++;
      else if(!strncmp(cl->op,"c.call",6))calls++;}}   /* c.call:NAME + c.call.indirect */
  int repro=0;                                   /* A1.3: C23 `[[reproducible]]`/`[[unsequenced]]` hints */
  for(int i=0;i<u->n_funcs;i++) if(u->funcs[i].reproducible) repro++;   /* counted over the WHOLE unit */
  snprintf(buf,n,"funcs=%d claims=%zu mmio=%d bf=%d const=%d binop=%d call=%d repro=%d ok=%d digest=%016llx",
           u->n_funcs,nc,mmio,bf,kn,binop,calls,repro,ok,
           (unsigned long long)bcir_cfront_digest(u));
}

/* --- the cross-rail PER-CLAIM STRUCTURAL DIGEST (the count->structural parity fix) -----------------
 * The 9-integer summary above compares the two cfront rails by COUNTS only, so any corruption that
 * preserves the counts -- swapping operands between two same-op claims, redirecting a call @foo->@bar
 * (both defined), or substituting one c.bin.* op for another -- slips through the gate. This digest
 * closes that gap with a CANONICAL, language-independent serialization of every function's claim
 * DATAFLOW, hashed with FNV-1a (64-bit). The Python oracle (bcir.verify.cfront_structural_digest)
 * builds the SAME records and the SAME hash, so the digests are byte-identical across the whole fixture
 * corpus (proven empirically by --canon: the diff is EMPTY).
 *
 * WHY DATAFLOW VALUE-NUMBERS, not raw positions/rids (the two cfront frontends are NOT byte-identical IR
 * producers -- benign divergences, each measured against the oracle over the whole corpus, defeat a
 * naive positional serialization): (1) the Python rid is the C rid + 1 and absolute rids are
 * rail-private; (2) the sibling sub-expression EVALUATION ORDER differs on 11 fixtures, so claim
 * POSITION is not a cross-rail invariant; (3) a few COMMUTATIVE ops order their two reads differently.
 * The digest is invariant to (1)-(3) yet still a STRUCTURE check:
 *
 *   record(claim) = <op-base>|<opcode-int>|<value-numbers of its reads>|<c.const imm>|<dom>
 *   vn(rid) = <op-base>(<value-numbers of that producer's reads>)  if a claim writes rid; else "in:pj"
 *             if rid is the j-th PARAMETER (position is cross-rail stable -> the two params in `a - b`
 *             are distinguished); else "in" (any other input -- a global/local -- stays anonymous).
 *
 * op-base strips ONLY `c.call.vaarg`'s rail-divergent `:T` suffix (Python emits bare `c.call.vaarg`, the
 * C twin `c.call.vaarg:int`); every OTHER ':' suffix is STRUCTURAL and KEPT -- the c.call callee (a
 * redirect @foo->@bar changes it), the c.cast WIDTH (a type change is caught), the c.fconst VALUE.
 * opcode/domain are their INTEGER values (== the Python IntEnum values by construction). c.const's imm
 * is folded in (a constant tamper is caught). Read order is POSITIONAL by default (so reversing a
 * non-commutative op -- sub/div/mod/shl/shr/lt/gt/le/ge or a c.store -- is caught: emit lowers
 * `ref(rd[0]) op ref(rd[1])` in order); reads are SORTED ONLY for the COMMUTATIVE ops (add/mul/and/or/
 * xor/eq/ne), which is what absorbs (3). The per-function record list is SORTED (absorbs (2)). NOP
 * markers are skipped (matches the count). A duplicate/injected claim id is caught by the unit-wide
 * R1.1 law, not here. */

#define BCIR_VN_MAXDEPTH 96
/* the ONLY op whose ':' suffix is a rail-divergent label (stripped); every other suffix is kept. */
#define BCIR_VN_STRIP(head) (!strcmp(head,"c.call.vaarg"))
/* genuinely commutative ops: their reads are sorted (order cannot change the value); all others stay
 * positional, so reversing a non-commutative op's operands changes the record. */
static int vn_commutative(const char *base){
  return !strcmp(base,"c.bin.add")||!strcmp(base,"c.bin.mul")||!strcmp(base,"c.bin.and")||
         !strcmp(base,"c.bin.or")||!strcmp(base,"c.bin.xor")||!strcmp(base,"c.bin.eq")||
         !strcmp(base,"c.bin.ne");
}

/* op-base: strip ONLY c.call.vaarg's rail-divergent `:T` suffix; else keep the full op (so the callee
 * in c.call:NAME, the width in c.cast:W, and the value in c.fconst:V all survive). */
/* The op identity's buffer: a whole op (a claim holds at most BCIR_CIR_OP - 1 characters) and `!atomic`. */
#define BCIR_VN_OP (BCIR_CIR_OP + 8)
static void vn_base(const char *op, char *out){
  const char *c=strchr(op,':');
  if(c){ size_t h=(size_t)(c-op); char head[BCIR_CIR_OP];
    if(h>=sizeof head) h=sizeof head-1; memcpy(head,op,h); head[h]=0;
    if(BCIR_VN_STRIP(head)){ snprintf(out,BCIR_VN_OP,"%s",head); return; } }
  snprintf(out,BCIR_VN_OP,"%.*s",BCIR_CIR_OP-1,op);
}
/* A claim's op identity in the canon (the oracle's `_vn_op`): `vn_base`, and a load or store of an `_Atomic`
 * object (the `atomic` hazard, CF-ATOMIC) marked `!atomic` -- the same op and operands as a plain access, but
 * not the same access. No other c.load / c.store carries the hazard, so every other record is unchanged. */
static void canon_op(const bcir_claim *cl, char *out){
  vn_base(cl->op,out);
  if(cl->hazard==BCIR_HZ_ATOMIC && (!strcmp(out,"c.load") || !strcmp(out,"c.store"))){
    size_t n=strlen(out); snprintf(out+n,BCIR_VN_OP-n,"!atomic"); }
}

static void *vn_alloc(bcir_host_arena *arena,size_t count,size_t element_size,int zero){
  size_t bytes;void *result;
  if(!bcir_size_mul(count,element_size,&bytes))return NULL;
  result=bcir_host_arena_allocate(arena,bytes?bytes:1u,0u);
  if(result&&zero)memset(result,0,bytes?bytes:1u);
  return result;
}

/* Operation-arena string duplicate. Canonicalization has no independently
 * owned scratch allocations; the complete arena is reset after each function. */
static char *vn_strdup(bcir_host_arena *arena,const char *s){
  size_t z=strlen(s),n;char *r;
  if(!bcir_size_add(z,1u,&n))return NULL;
  r=(char *)bcir_host_arena_allocate(arena,n,1u);if(r)memcpy(r,s,n);return r;
}

/* a small growable byte string (the per-function record buffer + the value-number scratch). */
typedef struct { char *s; size_t n, cap; bcir_host_arena *arena; } sbuf;
static sbuf sbuf_for(bcir_host_arena *arena){sbuf b={NULL,0,0,arena};return b;}
static void sb_add(sbuf *b, const char *p, size_t k){
  if(!p||b->n==SIZE_MAX||k>SIZE_MAX-b->n-1)return;size_t need=b->n+k+1;
  if(need>b->cap){ size_t nc=b->cap?b->cap:256;
    while(nc<need){if(nc>SIZE_MAX/2){nc=need;break;}nc*=2;}
    char *r=(char *)bcir_host_arena_allocate(b->arena,nc,1u);if(!r)return;
    if(b->s&&b->n)memcpy(r,b->s,b->n);b->s=r;b->cap=nc; }
  memcpy(b->s+b->n,p,k); b->n+=k; b->s[b->n]=0;
}
static void sb_str(sbuf *b,const char *s){ if(s)sb_add(b,s,strlen(s)); }

/* The SEMANTIC imm component of a claim's record -- the imm fields that encode WHICH datum a claim
 * touches (a const value, a struct member byte offset, a bitfield bit-off/width/sign), so reading or
 * writing the WRONG member/bitfield is caught. Only the cross-rail-STABLE positions are folded (the
 * Python oracle emits the same bytes); rail-divergent metadata (the c.load bounds `ub`, the c.store
 * trailing _Bool/stride flags) is dropped, preserving --canon byte-identity. Mirrors _vn_imm exactly:
 *   c.const/c.addrof/c.bf.get/c.bf.set/c.call.imember/c.sizeof.vla -> all imm;
 *   c.load -> imm[0] (member byte offset; 0 if absent);  c.store -> imm[0],imm[1] (offset, unit size). */
static void vn_imm(const bcir_func *f, const bcir_claim *cl, sbuf *rec){
  const char *op=cl->op; char nb[24]; int n=cl->n_imm;
  if(!strcmp(op,"c.const")||!strcmp(op,"c.addrof")||!strcmp(op,"c.bf.get")||!strcmp(op,"c.bf.set")||
     !strcmp(op,"c.call.imember")||!strcmp(op,"c.sizeof.vla")){
    /* a constant of an unsigned type is its value in that type: an unsigned 64-bit one past LLONG_MAX keeps its
     * bits in the imm, and the oracle's canon spells the value (`18446744073709551615`, never -1) */
    const bcir_resource *kr=!strcmp(op,"c.const") && cl->n_wr ? res_of(f,cl->wr[0]) : NULL;
    int unsig=kr && kr->kind==BCIR_RK_SCALAR && !kr->is_signed && !kr->is_float;
    for(int k=0;k<n;k++){ if(k)sb_str(rec,",");
      int l=unsig ? snprintf(nb,sizeof nb,"%llu",(unsigned long long)cl->imm[k])
                  : snprintf(nb,sizeof nb,"%lld",(long long)cl->imm[k]);
      sb_add(rec,nb,(size_t)l); }
  } else if(!strcmp(op,"c.load")){
    long long off = n>0 ? (long long)cl->imm[0] : 0;   /* the member byte offset (0 if absent) */
    int l=snprintf(nb,sizeof nb,"%lld",off); sb_add(rec,nb,(size_t)l);
  } else if(!strcmp(op,"c.store") || !strncmp(op,"c.c11atom.rmw:",14)){   /* an atomic read-modify-write: the
                                                       * store's address imm (CF-ATOMIC) */
    int m = n<2 ? n : 2;                               /* (byte offset, unit size); drop the _Bool/stride tail */
    for(int k=0;k<m;k++){ if(k)sb_str(rec,","); int l=snprintf(nb,sizeof nb,"%lld",(long long)cl->imm[k]); sb_add(rec,nb,(size_t)l); }
  }
}

/* sort an array of \0-terminated strings (small n; insertion sort, strcmp order). */
static void sort_strs(char **a, int n){
  for(int i=1;i<n;i++){ char *t=a[i]; int j=i;
    while(j>0 && strcmp(a[j-1]?a[j-1]:"",t?t:"")>0){ a[j]=a[j-1]; j--; } a[j]=t; }
}

/* per-function value-number context: writer[rid]=claim index that first writes it, plus a memo. */
typedef struct {
  const bcir_func *f;
  int *wclaim;          /* parallel to a rid list: index of the first writer claim, or -1 */
  uint32_t *wrid; int nw;
  char **memo;          /* memo[claim] -> its value-number string (lazily built), or NULL */
  bcir_host_arena *arena;
} vnctx;
static int writer_of(vnctx *v, uint32_t rid){
  for(int k=0;k<v->nw;k++) if(v->wrid[k]==rid) return v->wclaim[k]; return -1;
}
/* append vn(rid) to `out`. depth guards recursion; a re-entered claim folds to "cyc". */
static void vn_of(vnctx *v, uint32_t rid, int depth, sbuf *out);
static void vn_claim(vnctx *v, int ci, int depth, sbuf *out){
  if(v->memo[ci]){ sb_str(out,v->memo[ci]); return; }
  if(depth>BCIR_VN_MAXDEPTH){ sb_str(out,"cyc"); return; }
  v->memo[ci]=vn_strdup(v->arena,"cyc");        /* cycle guard: a loop-carried rid resolves to "cyc" */
  const bcir_claim *cl=&v->f->claims[ci];
  char base[BCIR_VN_OP]; canon_op(cl,base);
  /* gather the vns of this claim's reads -- POSITIONAL, except sorted for a commutative op */
  char *parts[BCIR_CLAIM_MAX_RD]; sbuf ps[BCIR_CLAIM_MAX_RD]; int np=cl->n_rd;
  for(int k=0;k<np;k++){ ps[k]=sbuf_for(v->arena); vn_of(v,cl->rd[k],depth+1,&ps[k]); parts[k]=ps[k].s?ps[k].s:(char*)""; }
  if(vn_commutative(base)) sort_strs(parts,np);
  sbuf me=sbuf_for(v->arena); sb_str(&me,base); sb_str(&me,"(");
  for(int k=0;k<np;k++){ if(k)sb_str(&me,","); sb_str(&me,parts[k]); }
  sb_str(&me,")");
  v->memo[ci]=me.s?me.s:vn_strdup(v->arena,"");
  sb_str(out,v->memo[ci]);
}
static void vn_of(vnctx *v, uint32_t rid, int depth, sbuf *out){
  int ci=writer_of(v,rid);
  if(ci<0){                                   /* a function input: a param (positional) or a global/etc */
    const bcir_func *f=v->f;
    for(int j=0;j<f->n_params;j++) if(f->params[j].rid==rid){   /* the j-th parameter -> "in:pj" */
      char b[24]; int l=snprintf(b,sizeof b,"in:p%d",j); sb_add(out,b,(size_t)l); return; }
    sb_str(out,"in"); return;                 /* any other input stays anonymous (rail-private rid out) */
  }
  vn_claim(v,ci,depth,out);
}

/* Build the sorted multiset of per-claim dataflow records for one function; emit each, '\n'-joined.
 * Even a zero-real-claim function (e.g. `int read_a(void){ return a_global; }`) emits its OBSERVABLE-
 * OUTPUT anchor (ret=...|stores=...), exactly as the Python oracle does -- so the byte-identity holds. */
static void canon_func(const bcir_func *f, void (*emit)(void*,const char*,size_t),
                       void *ctx,bcir_host_arena *arena){
  if(!f||!emit)return;
  if(f->n_claims>(size_t)INT_MAX||f->n_claims>(SIZE_MAX-1)/(size_t)BCIR_CLAIM_MAX_WR){
    emit(ctx,"overflow\n",9);return;}
  /* index the non-NOP claims and the first-writer of each rid */
  int nc=0; for(size_t i=0;i<f->n_claims;i++) if(f->claims[i].opcode!=BCIR_OP_NOP) nc++;
  size_t maxw=f->n_claims*(size_t)BCIR_CLAIM_MAX_WR+1;   /* an upper bound on distinct written rids */
  int *cidx=(int *)vn_alloc(arena,(size_t)(nc?nc:1),sizeof *cidx,0);   /* cidx[j] = original claim index of the j-th non-NOP */
  vnctx v={f,NULL,NULL,0,NULL,arena};
  v.wclaim=(int *)vn_alloc(arena,maxw,sizeof(int),0);
  v.wrid=(uint32_t *)vn_alloc(arena,maxw,sizeof(uint32_t),0);
  v.memo=(char **)vn_alloc(arena,f->n_claims?f->n_claims:1,sizeof(char*),1);
  if(!cidx||!v.wclaim||!v.wrid||!v.memo){
    emit(ctx,"oom\n",4);return;}
  /* NB: vn indexes by ORIGINAL claim index (so memo/writer reference the real claim array). */
  int j=0;
  for(size_t i=0;i<f->n_claims;i++){ const bcir_claim *cl=&f->claims[i];
    if(cl->opcode==BCIR_OP_NOP) continue; cidx[j++]=(int)i;
    for(int k=0;k<cl->n_wr;k++){ uint32_t rid=cl->wr[k]; int seen=0;
      for(int m=0;m<v.nw;m++) if(v.wrid[m]==rid){seen=1;break;}
      if(!seen){ v.wrid[v.nw]=rid; v.wclaim[v.nw]=(int)i; v.nw++; } } }
  /* build each non-NOP claim's record */
  char **recs=(char **)vn_alloc(arena,(size_t)nc,sizeof *recs,0);
  if(nc&&!recs){emit(ctx,"oom\n",4);return;}
  for(int r=0;r<nc;r++){ int i=cidx[r]; const bcir_claim *cl=&f->claims[i];
    char base[BCIR_VN_OP]; canon_op(cl,base);
    char *parts[BCIR_CLAIM_MAX_RD]; sbuf ps[BCIR_CLAIM_MAX_RD]; int np=cl->n_rd;
    for(int k=0;k<np;k++){ ps[k]=sbuf_for(arena); vn_of(&v,cl->rd[k],0,&ps[k]); parts[k]=ps[k].s?ps[k].s:(char*)""; }
    if(vn_commutative(base)) sort_strs(parts,np);     /* commutative: sort; else POSITIONAL */
    sbuf rec=sbuf_for(arena); char nb[24];
    sb_str(&rec,base); sb_str(&rec,"|");
    int ol=snprintf(nb,sizeof nb,"%d",(int)cl->opcode); sb_add(&rec,nb,(size_t)ol); sb_str(&rec,"|");
    for(int k=0;k<np;k++){ if(k)sb_str(&rec,","); sb_str(&rec,parts[k]); }
    sb_str(&rec,"|");
    vn_imm(f,cl,&rec);                                /* the semantic imm (member offset / bitfield layout) */
    sb_str(&rec,"|");
    int dl=snprintf(nb,sizeof nb,"%d",(int)cl->domain); sb_add(&rec,nb,(size_t)dl);
    recs[r]=rec.s?rec.s:vn_strdup(arena,"");
  }
  sort_strs(recs,nc);
  for(int r=0;r<nc;r++){ const char *rec=recs[r]?recs[r]:"";emit(ctx,rec,strlen(rec));emit(ctx,"\n",1); }

  /* The OBSERVABLE-OUTPUT anchor (LAST-writer VN -- a use observes the most-recent prior write, which
   * is what the emitted C returns/stores). It pins what the function OUTPUTS: the RETURN value's VN
   * (catches a sink-wr redirect that turns `return t` into `return (a+b)` though no per-claim record
   * changes) and the sorted STORE (dest-VN -> value-VN) pairs (catch a dead/store-target redirect).
   * last==first for a single-write rid, so the anchor stays cross-rail byte-identical. */
  vnctx vl={f,NULL,NULL,0,NULL,arena};
  vl.wclaim=(int *)vn_alloc(arena,maxw,sizeof(int),0);
  vl.wrid=(uint32_t *)vn_alloc(arena,maxw,sizeof(uint32_t),0);
  vl.memo=(char **)vn_alloc(arena,f->n_claims?f->n_claims:1,sizeof(char*),1);
  if(!vl.wclaim||!vl.wrid||!vl.memo){
    emit(ctx,"oom\n",4);return;}
  for(int r=0;r<nc;r++){ int i=cidx[r]; const bcir_claim *cl=&f->claims[i];
    for(int k=0;k<cl->n_wr;k++){ uint32_t rid=cl->wr[k]; int slot=-1;
      for(int m=0;m<vl.nw;m++) if(vl.wrid[m]==rid){slot=m;break;}
      if(slot<0){ vl.wrid[vl.nw]=rid; vl.wclaim[vl.nw]=(int)i; vl.nw++; }
      else vl.wclaim[slot]=(int)i; } }                /* LAST writer wins (overwrite) */
  sbuf anc=sbuf_for(arena); sb_str(&anc,"ret=");
  if(f->has_return){ vn_of(&vl,f->return_rid,0,&anc); } else sb_str(&anc,"void");
  sb_str(&anc,"|stores=");
  /* collect store (dest->value) pairs, sorted */
  int nst=0; for(int r=0;r<nc;r++) if(!strcmp(f->claims[cidx[r]].op,"c.store")) nst++;
  if(nst){ char **sp=(char **)vn_alloc(arena,(size_t)nst,sizeof *sp,1); int si=0;
    if(!sp)sb_str(&anc,"oom");
    else {
    for(int r=0;r<nc;r++){ const bcir_claim *cl=&f->claims[cidx[r]];
      if(strcmp(cl->op,"c.store")) continue;
      sbuf s=sbuf_for(arena);
      if(cl->n_rd){ vn_of(&vl,cl->rd[0],0,&s); sb_str(&s,"->"); vn_of(&vl,cl->rd[cl->n_rd-1],0,&s); }
      else sb_str(&s,"?->?");
      sp[si++]=s.s?s.s:vn_strdup(arena,""); }
    sort_strs(sp,si);
    for(int k=0;k<si;k++){ if(k)sb_str(&anc,";"); sb_str(&anc,sp[k]); }
    }
  }
  emit(ctx,anc.s?anc.s:"ret=void|stores=",anc.s?strlen(anc.s):16); emit(ctx,"\n",1);
  /* A static's constant image (CF-STATICTAB): its initializer runs once, before the program, so no claim writes it (a
   * static reads as an input) -- the initializer both rails render joins the canon instead, one line per static that
   * has one, sorted. Two statics differing only in a value differ here. */
  int nsx=0; for(int k=0;k<f->n_statics;k++) if(f->statics[k].text || f->statics[k].thread_storage) nsx++;
  if(nsx){ char **sl=(char **)vn_alloc(arena,(size_t)nsx,sizeof *sl,1); int si=0;
    if(!sl){ emit(ctx,"oom\n",4); return; }
    for(int k=0;k<f->n_statics;k++) if(f->statics[k].text || f->statics[k].thread_storage){ sbuf s=sbuf_for(arena);
      /* a static of thread storage duration has its line whatever its image (CF-TLS; the oracle's canon) */
      sb_str(&s,f->statics[k].thread_storage?"static _Thread_local ":"static "); sb_str(&s,f->statics[k].name);
      sb_str(&s," = "); sb_str(&s,f->statics[k].text?f->statics[k].text:"0");
      sl[si++]=s.s?s.s:vn_strdup(arena,""); }
    sort_strs(sl,si);
    for(int k=0;k<si;k++){ const char *ln=sl[k]?sl[k]:""; emit(ctx,ln,strlen(ln)); emit(ctx,"\n",1); } }
}

/* The shared canonical serializer: invokes emit(ctx, bytes, len) for each byte of the canon (so the
 * digest and the --canon text dump are GUARANTEED to be the same bytes). One '@' line per function. */
static void canon_walk(const bcir_unit *u, void (*emit)(void*,const char*,size_t), void *ctx,
                       const bcir_host_allocator *allocator){
  bcir_host_arena arena;
  if(!u||!emit)return;
  if(!bcir_host_arena_init(&arena,allocator,4096u)){emit(ctx,"oom\n",4);return;}
  for(int fi=0;fi<u->n_funcs;fi++){
    canon_func(&u->funcs[fi],emit,ctx,&arena);
    bcir_host_arena_reset(&arena);
    emit(ctx,"@\n",2);
  }
  bcir_host_arena_destroy(&arena);
}

typedef struct { uint64_t h; } fnv_ctx;
static void fnv_emit(void *vc, const char *b, size_t n){
  fnv_ctx *c=vc; for(size_t i=0;i<n;i++) c->h=(c->h^(unsigned char)b[i])*1099511628211ull;
}
uint64_t bcir_cfront_digest_with_allocator(const bcir_unit *u,
                                           const bcir_host_allocator *allocator){
  fnv_ctx c={1469598103934665603ull};            /* FNV-1a offset basis (== the Python _DIGEST_OFFSET) */
  if(!u)return c.h;
  canon_walk(u,fnv_emit,&c,allocator);
  return c.h;
}
uint64_t bcir_cfront_digest(const bcir_unit *u){
  bcir_host_allocator allocator=bcir_host_allocator_default();
  return bcir_cfront_digest_with_allocator(u,&allocator);
}

typedef struct { char *buf; size_t cap, w; } buf_ctx;
static void buf_emit(void *vc, const char *b, size_t n){
  buf_ctx *c=vc; for(size_t i=0;i<n;i++){ if(c->w+1<c->cap) c->buf[c->w]=b[i]; c->w++; }
}
/* The canon's allocator: the caller's, noting an allocation it refused -- a canon built past one is not whole (its
 * walk writes `oom` where the work went undone, as the digest hashes it). The arena never reallocates; were it to,
 * the canon would be refused, never cut. */
typedef struct { bcir_host_allocator inner; int refused; } canon_alloc;
static void *canon_allocate(void *vc,size_t size){
  canon_alloc *a=vc; void *p=bcir_host_allocate(&a->inner,size); if(!p) a->refused=1; return p;
}
static void *canon_reallocate(void *vc,void *p,size_t size){
  (void)p; (void)size; ((canon_alloc *)vc)->refused=1; return NULL;
}
static void canon_deallocate(void *vc,void *p){ canon_alloc *a=vc; bcir_host_deallocate(&a->inner,p); }
/* The raw canonical serialization the digest hashes (text, NOT hashed) -- the byte-identity proof
 * (the Python cfront_structural_canon must equal this byte-for-byte on the corpus). Its whole length, as snprintf
 * counts it: a canon longer than the buffer is cut there and the length says so -- the twin's driver held 128 KiB
 * and printed the first 128 KiB of a longer canon, unsaid (CF-LIMITS) -- and one an allocation failed in is
 * SIZE_MAX, with nothing written. */
size_t bcir_cfront_canon_with_allocator(const bcir_unit *u,char *buf,size_t n,
                                        const bcir_host_allocator *allocator){
  canon_alloc a={bcir_host_allocator_or_default(allocator),0};
  bcir_host_allocator noting={&a,canon_allocate,canon_reallocate,canon_deallocate};
  buf_ctx c={buf,buf?n:0,0};canon_walk(u,buf_emit,&c,&noting);
  if(a.refused){ if(buf&&n) buf[0]=0; return SIZE_MAX; }
  if(buf&&n) buf[c.w<n?c.w:n-1]=0;
  return c.w;
}
size_t bcir_cfront_canon(const bcir_unit *u,char *buf,size_t n){
  bcir_host_allocator allocator=bcir_host_allocator_default();
  return bcir_cfront_canon_with_allocator(u,buf,n,&allocator);
}

/* --- G10: escape analysis, indirect-call narrowing and the effect footprint -------------------------
 * The C twin of bcir/frontends/cfront/escape.py -- the same objects, rules and reports, byte for byte
 * (parity-gated over the cfront corpus and generated programs by test_c_cfront.py and check_runtime.sh).
 * Andersen's analysis: inclusion-based, flow-, context- and field-insensitive, OPEN WORLD. The objects:
 *   TOP  unknown memory              STR  every string literal (read-only)
 *   G    a file-scope global (name)  S    a static local (`fn.name`)
 *   L    an automatic local or a temporary (function + rid)
 *   H    a function's heap (everything its allocator calls return)
 *   F    a function (name)
 * pt(o) is the set of objects the pointers stored in storage object o may point to: a bitset over the
 * objects that can be pointed to at all (TOP, STR, F, H, arrays used as values, address-taken objects).
 * Every rule is monotone in pt, so the least fixpoint -- and every report -- is independent of the order
 * the claims are visited in (the rails order sibling expressions differently). The rules, the escape
 * verdicts and the named footprint are the oracle's; each helper names its twin. Hosted tool code: every
 * temporary lives in one arena, released before return. */
enum { EK_SCALAR=0, EK_ARRAY=1, EK_STRUCT=2, EK_POINTER=3 };
enum { EO_TOP=0, EO_STR=1, EO_G=2, EO_S=3, EO_L=4, EO_H=5, EO_F=6 };
enum { EV_NONESCAPING=0, EV_LENT=1, EV_ESCAPING=2 };
static const char *const esc_verdicts[3]={"nonescaping","lent","escaping"};

typedef struct {
  uint8_t kind;              /* EO_* */
  uint8_t taken;             /* F: its value is used (address-taken) */
  int fn;                    /* the owning function (S / L / H), else -1 */
  int def;                   /* F: the defined function it names, else -1 */
  int u;                     /* its index among the pointable objects, or -1 */
  const char *name;          /* G / F: the name; S: `fn.name` */
} esc_obj;

typedef struct {             /* one rid of one function */
  uint32_t rid;
  int obj;                   /* its storage node, or its F / STR object */
  int def;                   /* the first claim writing it, or -1 */
  uint8_t kind;              /* EK_*: how its C type makes it behave */
  uint8_t declared;          /* a variable the source declares (not a temporary) */
  uint8_t memory;            /* storage the program addresses (see esc_build) */
  uint8_t named_local;       /* a named automatic local: reported with a verdict */
  const char *name;
} esc_rid;

typedef struct {
  const bcir_func *f;
  esc_rid *r; int nr;        /* sorted by rid, plus one sentinel entry r[nr] (a rid no claim names) */
  int *params;               /* esc_rid index of each parameter */
  int *rets; int nret;       /* esc_rid indices of the `c.return` operands */
  int heap, fobj;            /* its H object; the F object naming it (-1: its value is never used) */
  int *edges; int nedges;    /* the defined functions it calls, directly or through a known pointer */
} esc_fn;

typedef struct {
  bcir_host_arena *arena;
  const bcir_unit *u;
  esc_fn *fn; int nf;
  esc_obj *obj; int nobj;
  int *named; int nnamed;    /* the G / S / F objects (found by name) */
  int *uobj; int nu, W;      /* the pointable objects (u index -> object) and the bitset width in words */
  uint64_t *pt;              /* nobj x W */
  uint64_t *esc;             /* W: objects stored into unknown memory */
  uint64_t *tmp;             /* ESC_NTMP scratch bitsets of W words */
  int *tg;                   /* nf: an indirect call's targets */
  int changed;
} esc_ctx;
#define ESC_NTMP 8

static void *esc_alloc(esc_ctx *e,size_t count,size_t size){ return vn_alloc(e->arena,count?count:1u,size,1); }
static int esc_has(const uint64_t *b,int u){ return u>=0 && ((b[u>>6]>>(u&63))&1u); }
static void esc_bit(uint64_t *b,int u){ if(u>=0) b[u>>6]|=(uint64_t)1u<<(u&63); }
static void esc_clear(const esc_ctx *e,uint64_t *b){ memset(b,0,(size_t)e->W*sizeof *b); }
static void esc_or(const esc_ctx *e,uint64_t *d,const uint64_t *s){ for(int i=0;i<e->W;i++) d[i]|=s[i]; }
static uint64_t *esc_tmp(const esc_ctx *e,int k){ return e->tmp+(size_t)k*(size_t)e->W; }
static uint64_t *esc_pt(const esc_ctx *e,int o){ return e->pt+(size_t)o*(size_t)e->W; }

static int esc_is(const char *op,const char *prefix){ return !strncmp(op,prefix,strlen(prefix)); }
static int esc_any(const char *op,const char *const *prefixes){
  for(int i=0;prefixes[i];i++) if(esc_is(op,prefixes[i])) return 1;
  return 0;
}
static const char *const esc_ops_noflow[]={"c.const","c.fconst:","c.cconst:","c.labeladdr:","c.sizeof.vla",
                                        "c.fence","c.vladecl",NULL};
static const char *const esc_ops_atomic[]={"c.atomic.","c.c11atom.","c.cmpxchg.",NULL};
static const char *const esc_ops_external[]={"c.call.tu:","c.call.extern:","c.asm:","c.asm.volatile:",NULL};
static const char *const esc_ops_library[]={"c.call.libm:","c.call.libm.void:","c.call.builtin:",
                                         "c.call.vabuiltin:",NULL};
static const char *const esc_ops_allocators[]={"c.call.libm:malloc","c.call.libm:calloc","c.call.libm:realloc",
                                            "c.call.libm:aligned_alloc",NULL};
static int esc_direct(const char *op){ return esc_is(op,"c.call:")||esc_is(op,"c.call.void:"); }
static int esc_indirect(const char *op){ return !strcmp(op,"c.call.indirect")||esc_is(op,"c.call.imember"); }
static int esc_allocator(const char *op){
  for(int i=0;esc_ops_allocators[i];i++) if(!strcmp(op,esc_ops_allocators[i])) return 1;
  return 0;
}
static int esc_pointer_cast(const char *op){   /* `(T *)v`: both rails spell it `c.cast:<pointee> *` */
  size_t n=strlen(op); return esc_is(op,"c.cast:") && n && op[n-1]=='*';
}
static const char *esc_callee(const char *op){ const char *p=strchr(op,':'); return p?p+1:""; }
static int esc_find_fn(const esc_ctx *e,const char *name){
  for(int i=0;i<e->nf;i++) if(!strcmp(e->fn[i].f->name,name)) return i;
  return -1;
}
/* the esc_rid of `rid` (binary search; a rid outside the table is the sentinel r[nr]) */
static esc_rid *esc_R(const esc_fn *F,uint32_t rid){
  int lo=0,hi=F->nr-1;
  while(lo<=hi){ int mid=lo+(hi-lo)/2; if(F->r[mid].rid==rid) return &F->r[mid];
    if(F->r[mid].rid<rid) lo=mid+1; else hi=mid-1; }
  return &F->r[F->nr];
}
static int esc_u32cmp(const void *a,const void *b){
  uint32_t x=*(const uint32_t *)a,y=*(const uint32_t *)b; return (x>y)-(x<y);
}
static int esc_strcmp(const void *a,const void *b){ return strcmp(*(const char *const *)a,*(const char *const *)b); }

/* a new object; a named G / S / F object is found by name first */
static int esc_object(esc_ctx *e,int kind,int fn,const char *name){
  if(name) for(int i=0;i<e->nnamed;i++){ const esc_obj *o=&e->obj[e->named[i]];
    if(o->kind==kind && !strcmp(o->name,name)) return e->named[i]; }
  esc_obj *o=&e->obj[e->nobj]; memset(o,0,sizeof *o);
  o->kind=(uint8_t)kind; o->fn=fn; o->def=-1; o->u=-1; o->name=name;
  if(name) e->named[e->nnamed++]=e->nobj;
  return e->nobj++;
}

/* how a resource's C type makes it behave (the oracle's `_kind` over the rid's CType) */
static int esc_kind(const bcir_func *f,const bcir_resource *r){
  for(int p=0;p<f->n_params;p++) if(f->params[p].rid==r->rid){
    int k=f->params[p].type.kind;                              /* 0 scalar, 1 struct, 2 ptr, 3 funcptr */
    return k==1?EK_STRUCT : (k==2||k==3)?EK_POINTER : EK_SCALAR; }
  if(r->is_array||r->is_vla) return EK_ARRAY;
  if(r->kind==BCIR_RK_AGGREGATE) return EK_STRUCT;
  if(r->kind==BCIR_RK_POINTER||r->is_funcptr||r->is_pointer) return EK_POINTER;
  return EK_SCALAR;
}

/* one function's rids: their objects, kinds and roles (the oracle's `_Unit`). 0 on OOM. */
static int esc_build_fn(esc_ctx *e,int fi){
  esc_fn *F=&e->fn[fi]; const bcir_func *f=&e->u->funcs[fi]; F->f=f;
  size_t nids=f->n_res; for(size_t k=0;k<f->n_claims;k++) nids+=f->claims[k].n_rd+f->claims[k].n_wr;
  uint32_t *ids=esc_alloc(e,nids,sizeof *ids); if(!ids) return 0;
  size_t m=0;
  for(size_t k=0;k<f->n_res;k++) ids[m++]=f->res[k].rid;
  for(size_t k=0;k<f->n_claims;k++){ const bcir_claim *c=&f->claims[k];
    for(int j=0;j<c->n_rd;j++) ids[m++]=c->rd[j];
    for(int j=0;j<c->n_wr;j++) ids[m++]=c->wr[j]; }
  qsort(ids,m,sizeof *ids,esc_u32cmp);
  size_t uniq=0; for(size_t k=0;k<m;k++) if(!uniq||ids[uniq-1]!=ids[k]) ids[uniq++]=ids[k];
  F->nr=(int)uniq; F->r=esc_alloc(e,uniq+1u,sizeof *F->r); if(!F->r) return 0;
  for(size_t k=0;k<=uniq;k++){ F->r[k].rid=k<uniq?ids[k]:0u; F->r[k].def=-1; F->r[k].obj=-1; }
  for(size_t k=0;k<f->n_res;k++){ const bcir_resource *res=&f->res[k]; esc_rid *R=esc_R(F,res->rid);
    if(R->obj>=0) continue;                                    /* a duplicate rid: the first wins */
    int is_param=0; for(int p=0;p<f->n_params;p++) if(f->params[p].rid==res->rid) is_param=1;
    int is_str=0; for(int h=0;h<f->n_host_literals;h++) if(f->host_literals[h].rid==res->rid) is_str=1;
    int st=-1; for(int s=0;s<f->n_statics;s++) if(f->statics[s].rid==res->rid && !res->read_only) st=s;
    R->kind=(uint8_t)esc_kind(f,res); R->name=res->name;
    if(is_str) R->obj=EO_STR;
    else if(res->read_only && res->is_funcptr) R->obj=esc_object(e,EO_F,-1,res->name);
    else if(res->read_only){ R->obj=esc_object(e,EO_G,-1,res->name); R->declared=1; }
    else if(st>=0){
      size_t z=strlen(f->name)+strlen(f->statics[st].name)+2u; char *nm=esc_alloc(e,z,1u); if(!nm) return 0;
      snprintf(nm,z,"%s.%s",f->name,f->statics[st].name);
      R->obj=esc_object(e,EO_S,fi,nm); R->declared=1; }
    else{ R->obj=esc_object(e,EO_L,fi,NULL);
      R->declared=(uint8_t)(is_param||res->name[0]);
      R->named_local=(uint8_t)(!is_param && res->name[0]); } }
  for(int k=0;k<=F->nr;k++) if(F->r[k].obj<0) F->r[k].obj=esc_object(e,EO_L,fi,NULL);   /* no resource */
  F->heap=esc_object(e,EO_H,fi,NULL);
  /* roles: the first writer; memory = used and never written, or an addressed base, or a declared VLA */
  uint8_t *written=esc_alloc(e,(size_t)F->nr+1u,1u), *used=esc_alloc(e,(size_t)F->nr+1u,1u);
  F->params=esc_alloc(e,(size_t)f->n_params,sizeof *F->params);
  F->rets=esc_alloc(e,f->n_claims,sizeof *F->rets);
  if(!written||!used||!F->params||!F->rets) return 0;
  for(int p=0;p<f->n_params;p++){ esc_rid *R=esc_R(F,f->params[p].rid); F->params[p]=(int)(R-F->r); used[R-F->r]=1; }
  for(size_t k=0;k<f->n_claims;k++){ const bcir_claim *c=&f->claims[k];
    for(int j=0;j<c->n_rd;j++) used[esc_R(F,c->rd[j])-F->r]=1;
    for(int j=0;j<c->n_wr;j++){ esc_rid *R=esc_R(F,c->wr[j]); used[R-F->r]=1; written[R-F->r]=1;
      if(R->def<0) R->def=(int)k; }
    if(!strcmp(c->op,"c.return") && c->n_rd==1) F->rets[F->nret++]=(int)(esc_R(F,c->rd[0])-F->r); }
  for(int k=0;k<F->nr;k++) F->r[k].memory=(uint8_t)(used[k]&&!written[k]);
  for(size_t k=0;k<f->n_claims;k++){ const bcir_claim *c=&f->claims[k];
    if(c->n_rd && (!strcmp(c->op,"c.load")||!strcmp(c->op,"c.store")||!strcmp(c->op,"c.addrof")||
                   esc_is(c->op,"c.call.imember"))) esc_R(F,c->rd[0])->memory=1;
    if(!strcmp(c->op,"c.vladecl") && c->n_wr) esc_R(F,c->wr[0])->memory=1; }
  return 1;
}

/* the unit's objects and the pointable universe. 0 on OOM. */
static int esc_build(esc_ctx *e){
  const bcir_unit *u=e->u; size_t cap=2;
  for(int i=0;i<u->n_funcs;i++){ const bcir_func *f=&u->funcs[i];
    cap+=f->n_res+2u; for(size_t k=0;k<f->n_claims;k++) cap+=f->claims[k].n_rd+f->claims[k].n_wr; }
  e->nf=u->n_funcs;
  e->fn=esc_alloc(e,(size_t)e->nf,sizeof *e->fn);
  e->obj=esc_alloc(e,cap,sizeof *e->obj);
  e->named=esc_alloc(e,cap,sizeof *e->named);
  e->tg=esc_alloc(e,(size_t)e->nf,sizeof *e->tg);
  if(!e->fn||!e->obj||!e->named||!e->tg) return 0;
  esc_object(e,EO_TOP,-1,NULL); esc_object(e,EO_STR,-1,NULL);
  for(int fi=0;fi<e->nf;fi++) if(!esc_build_fn(e,fi)) return 0;
  for(int i=0;i<e->nobj;i++) if(e->obj[i].kind==EO_F) e->obj[i].def=esc_find_fn(e,e->obj[i].name);
  for(int fi=0;fi<e->nf;fi++){ e->fn[fi].fobj=-1;
    for(int i=0;i<e->nnamed;i++) if(e->obj[e->named[i]].kind==EO_F && e->obj[e->named[i]].def==fi) e->fn[fi].fobj=e->named[i]; }
  /* the pointable objects: TOP, STR, every F and H, every array used as a value, every addressed base */
  uint8_t *pointable=esc_alloc(e,(size_t)e->nobj,1u); if(!pointable) return 0;
  pointable[EO_TOP]=pointable[EO_STR]=1;
  for(int i=0;i<e->nobj;i++) if(e->obj[i].kind==EO_F||e->obj[i].kind==EO_H) pointable[i]=1;
  for(int fi=0;fi<e->nf;fi++){ const esc_fn *F=&e->fn[fi];
    for(int k=0;k<F->nr;k++) if(F->r[k].kind==EK_ARRAY) pointable[F->r[k].obj]=1;
    for(size_t k=0;k<F->f->n_claims;k++){ const bcir_claim *c=&F->f->claims[k];
      if(!strcmp(c->op,"c.addrof") && c->n_rd) pointable[esc_R(F,c->rd[0])->obj]=1; } }
  e->uobj=esc_alloc(e,(size_t)e->nobj,sizeof *e->uobj); if(!e->uobj) return 0;
  for(int i=0;i<e->nobj;i++) if(pointable[i]){ e->obj[i].u=e->nu; e->uobj[e->nu++]=i; }
  e->W=(e->nu+63)/64;
  e->pt=esc_alloc(e,(size_t)e->nobj*(size_t)e->W,sizeof *e->pt);
  e->esc=esc_alloc(e,(size_t)e->W,sizeof *e->esc);
  e->tmp=esc_alloc(e,(size_t)ESC_NTMP*(size_t)e->W,sizeof *e->tmp);
  return e->pt&&e->esc&&e->tmp;
}

/* what a storage object holds: what the program stored, plus unknown pointers when unknown code can
 * have written it -- a global, a static, an escaped object (the oracle's `held`) */
static void esc_held(const esc_ctx *e,int o,uint64_t *out){
  esc_or(e,out,esc_pt(e,o));
  if(e->obj[o].kind==EO_G||e->obj[o].kind==EO_S||esc_has(e->esc,e->obj[o].u)) esc_bit(out,0);
}
static int esc_in_place(const esc_ctx *e,const esc_rid *R){   /* touched in place, not through it */
  int k=e->obj[R->obj].kind;
  return k==EO_F||k==EO_STR||R->kind==EK_ARRAY||R->kind==EK_STRUCT||(R->kind==EK_SCALAR&&R->declared);
}
/* the objects a rid's VALUE may point to (`val`) */
static void esc_val(esc_ctx *e,int fi,uint32_t rid,uint64_t *out){
  const esc_rid *R=esc_R(&e->fn[fi],rid); esc_obj *o=&e->obj[R->obj];
  if(o->kind==EO_F){ if(!o->taken){ o->taken=1; e->changed=1; } esc_bit(out,o->u); return; }
  if(o->kind==EO_STR||R->kind==EK_ARRAY){ esc_bit(out,o->u); return; }   /* the decay: its own address */
  esc_held(e,R->obj,out);
}
/* what a load or store with base `rid` touches (`deref`): the object itself (its id is returned), or
 * what the pointer holds (-1 is returned, the set is OR-ed into *out) */
static int esc_deref(const esc_ctx *e,int fi,uint32_t rid,uint64_t *out){
  const esc_rid *R=esc_R(&e->fn[fi],rid);
  if(esc_in_place(e,R)) return R->obj;
  esc_held(e,R->obj,out); return -1;
}
/* what `c.addrof` of base `rid` points to (`address`) */
static void esc_address(const esc_ctx *e,int fi,uint32_t rid,uint64_t *out){
  const esc_rid *R=esc_R(&e->fn[fi],rid);
  esc_bit(out,e->obj[R->obj].u);
  if(!esc_in_place(e,R)) esc_held(e,R->obj,out);
}
/* whether `(T *)rid` may make a pointer out of an integer (`forged`) */
static int esc_forged(const esc_ctx *e,int fi,uint32_t rid){
  const esc_fn *F=&e->fn[fi]; const esc_rid *R=esc_R(F,rid); int k=e->obj[R->obj].kind;
  if(k==EO_F||k==EO_STR) return 0;
  if(R->declared) return !(R->kind==EK_POINTER||R->kind==EK_ARRAY);
  if(R->def<0) return 1;
  const bcir_claim *c=&F->f->claims[R->def];
  if(!strcmp(c->op,"c.addrof")||esc_pointer_cast(c->op)) return 0;
  return !(!strcmp(c->op,"c.const") && c->n_imm>=1 && c->imm[0]==0);
}
/* the pointers held by object `one` (if >= 0) and the objects in `objs` (if non-NULL) (`contents`) */
static void esc_contents1(const esc_ctx *e,int o,uint64_t *out){
  int k=e->obj[o].kind;
  if(k==EO_TOP) esc_bit(out,0); else if(k!=EO_F&&k!=EO_STR) esc_held(e,o,out);
}
static void esc_contents(const esc_ctx *e,int one,const uint64_t *objs,uint64_t *out){
  if(one>=0) esc_contents1(e,one,out);
  if(objs) for(int x=0;x<e->nu;x++) if(esc_has(objs,x)) esc_contents1(e,e->uobj[x],out);
}
/* store `objs` into object o (`add`): stored into TOP they escape; code and literals hold nothing */
static void esc_add(esc_ctx *e,int o,const uint64_t *objs){
  int k=e->obj[o].kind;
  if(k==EO_TOP){
    for(int w=0;w<e->W;w++){ uint64_t nb=objs[w]&~e->esc[w]; if(!w) nb&=~(uint64_t)1u;
      if(nb){ e->esc[w]|=nb; e->changed=1; } }
    return; }
  if(k==EO_F||k==EO_STR) return;
  uint64_t *d=esc_pt(e,o);
  for(int w=0;w<e->W;w++){ uint64_t nb=objs[w]&~d[w]; if(nb){ d[w]|=nb; e->changed=1; } }
}
static void esc_add_all(esc_ctx *e,int one,const uint64_t *targets,const uint64_t *objs){
  if(one>=0) esc_add(e,one,objs);
  if(targets) for(int x=0;x<e->nu;x++) if(esc_has(targets,x)) esc_add(e,e->uobj[x],objs);
}
static void esc_add_rid(esc_ctx *e,int fi,uint32_t rid,const uint64_t *objs){
  esc_add(e,esc_R(&e->fn[fi],rid)->obj,objs);
}

/* what an indirect call's function pointer may hold (`call_values`) */
static void esc_call_values(const esc_ctx *e,int fi,const bcir_claim *c,uint64_t *values,uint64_t *scratch){
  if(!strcmp(c->op,"c.call.indirect")){ const esc_rid *R=esc_R(&e->fn[fi],c->rd[0]);
    if(e->obj[R->obj].kind==EO_F) esc_bit(values,e->obj[R->obj].u); else esc_held(e,R->obj,values);
    return; }
  esc_clear(e,scratch); int one=esc_deref(e,fi,c->rd[0],scratch);   /* c.call.imember: the base's member */
  esc_contents(e,one,one<0?scratch:NULL,values);
}
/* the defined functions an indirect call can reach (`targets`): their indices, sorted by name, in
 * e->tg (the count is returned), or -1 when it may reach unknown code or nothing known */
static int esc_targets(esc_ctx *e,int fi,const bcir_claim *c){
  uint64_t *values=esc_tmp(e,6); esc_clear(e,values);
  esc_call_values(e,fi,c,values,esc_tmp(e,7));
  if(esc_has(values,0)) return -1;
  int n=0;
  for(int x=0;x<e->nu;x++) if(esc_has(values,x)){ const esc_obj *o=&e->obj[e->uobj[x]];
    if(o->kind!=EO_F) continue;                   /* a data address is not a callable target */
    if(o->def<0) return -1;                       /* a function of another unit */
    e->tg[n++]=o->def; }
  if(!n) return -1;
  for(int i=1;i<n;i++) for(int j=i;j>0 && strcmp(e->fn[e->tg[j-1]].f->name,e->fn[e->tg[j]].f->name)>0;j--){
    int t=e->tg[j-1]; e->tg[j-1]=e->tg[j]; e->tg[j]=t; }
  return n;
}

/* a call binds actuals to formals and the callee's return values to its results (`bind`) */
static void esc_bind(esc_ctx *e,int fi,int ci,const uint32_t *act,int nact,const bcir_claim *c){
  uint64_t *v=esc_tmp(e,0); const esc_fn *C=&e->fn[ci];
  for(int k=0;k<nact;k++){ esc_clear(e,v); esc_val(e,fi,act[k],v);
    if(k<C->f->n_params) esc_add(e,C->r[C->params[k]].obj,v);
    else esc_add(e,EO_TOP,v); }                    /* a variadic extra: read through va_arg as unknown */
  if(c->n_wr){ esc_clear(e,v);
    for(int k=0;k<C->nret;k++) esc_val(e,ci,C->r[C->rets[k]].rid,v);
    for(int k=0;k<c->n_wr;k++) esc_add_rid(e,fi,c->wr[k],v); }
}
/* an unknown callee receives its actuals into unknown memory and returns unknown pointers (`external`) */
static void esc_external(esc_ctx *e,int fi,const uint32_t *act,int nact,const bcir_claim *c){
  uint64_t *v=esc_tmp(e,0);
  for(int k=0;k<nact;k++){ esc_clear(e,v); esc_val(e,fi,act[k],v); esc_add(e,EO_TOP,v); }
  esc_clear(e,v); esc_bit(v,0);
  for(int k=0;k<c->n_wr;k++) esc_add_rid(e,fi,c->wr[k],v);
}
static int esc_callers_unknown(const esc_ctx *e,int fi){
  const esc_fn *F=&e->fn[fi];
  return !F->f->static_fn || F->f->addr_in_init || (F->fobj>=0 && e->obj[F->fobj].taken);
}

/* one claim's constraints (the oracle's `_Solver.claim`) */
static void esc_claim(esc_ctx *e,int fi,const bcir_claim *c){
  const char *op=c->op; uint64_t *a=esc_tmp(e,1), *b=esc_tmp(e,2), *g=esc_tmp(e,3);
  if(esc_any(op,esc_ops_noflow)) return;
  if(!strcmp(op,"c.load")||!strcmp(op,"c.store")||!strcmp(op,"c.addrof")){
    if(!c->n_rd) return;
    esc_clear(e,a); esc_clear(e,g);
    if(!strcmp(op,"c.store")){
      if(c->n_rd>1) esc_val(e,fi,c->rd[c->n_rd-1],g);
      int one=esc_deref(e,fi,c->rd[0],a); esc_add_all(e,one,one<0?a:NULL,g); return; }
    if(!strcmp(op,"c.load")){ int one=esc_deref(e,fi,c->rd[0],a); esc_contents(e,one,one<0?a:NULL,g); }
    else esc_address(e,fi,c->rd[0],g);
    for(int k=0;k<c->n_wr;k++) esc_add_rid(e,fi,c->wr[k],g);
    return; }
  if(esc_any(op,esc_ops_atomic)){
    int ones[BCIR_CLAIM_MAX_RD]; int n1=0; esc_clear(e,a); esc_clear(e,b); esc_clear(e,g);
    for(int k=0;k<c->n_rd;k++){ int one=esc_deref(e,fi,c->rd[k],a); if(one>=0) ones[n1++]=one;
      esc_val(e,fi,c->rd[k],b); }
    for(int k=0;k<n1;k++) esc_add(e,ones[k],b);
    esc_add_all(e,-1,a,b);
    for(int k=0;k<n1;k++) esc_contents1(e,ones[k],g);
    esc_contents(e,-1,a,g);
    for(int k=0;k<c->n_wr;k++) esc_add_rid(e,fi,c->wr[k],g);
    return; }
  if(esc_direct(op)){ int ci=esc_find_fn(e,esc_callee(op));
    if(ci>=0) esc_bind(e,fi,ci,c->rd,c->n_rd,c); else esc_external(e,fi,c->rd,c->n_rd,c);
    return; }
  if(esc_indirect(op)){           /* monotone: bind every known target; external once anything is unknown */
    if(!c->n_rd) return;
    esc_clear(e,a); esc_val(e,fi,c->rd[0],a);    /* the pointer / dispatch base is used */
    esc_clear(e,g); esc_call_values(e,fi,c,g,b);
    int n=0, unknown=esc_has(g,0);
    for(int x=0;x<e->nu;x++) if(esc_has(g,x)){ const esc_obj *o=&e->obj[e->uobj[x]];
      if(o->kind!=EO_F) continue;
      if(o->def<0) unknown=1; else e->tg[n++]=o->def; }
    for(int k=0;k<n;k++) esc_bind(e,fi,e->tg[k],c->rd+1,c->n_rd-1,c);
    if(unknown) esc_external(e,fi,c->rd+1,c->n_rd-1,c);
    return; }
  if(esc_is(op,"c.call.vaarg")){ esc_clear(e,g); esc_bit(g,0);
    for(int k=0;k<c->n_wr;k++) esc_add_rid(e,fi,c->wr[k],g);
    return; }
  if(esc_any(op,esc_ops_external)){ esc_external(e,fi,c->rd,c->n_rd,c); return; }
  if(esc_any(op,esc_ops_library)){    /* keeps no pointer, returns what it was given (memcpy-like moves kept) */
    const esc_fn *F=&e->fn[fi];
    esc_clear(e,a); esc_clear(e,b);
    for(int k=0;k<c->n_rd;k++) esc_val(e,fi,c->rd[k],a);
    esc_contents(e,-1,a,b); esc_add_all(e,-1,a,b);
    for(int k=0;k<c->n_wr;k++){ const esc_rid *R=esc_R(F,c->wr[k]);
      memcpy(g,a,(size_t)e->W*sizeof *g);
      if(esc_allocator(op)) esc_bit(g,e->obj[F->heap].u);          /* fresh memory: this function's heap */
      else if(R->kind==EK_POINTER) esc_bit(g,0);                   /* a pointer a library routine made */
      esc_add(e,R->obj,g); }
    return; }
  if(esc_pointer_cast(op)){ esc_clear(e,g);
    if(c->n_rd) esc_val(e,fi,c->rd[0],g);
    if(!c->n_rd||esc_forged(e,fi,c->rd[0])) esc_bit(g,0);          /* an integer made into a pointer */
    for(int k=0;k<c->n_wr;k++) esc_add_rid(e,fi,c->wr[k],g);
    return; }
  esc_clear(e,g);                 /* every other op (and a control marker): a value made of its operands */
  for(int k=0;k<c->n_rd;k++) esc_val(e,fi,c->rd[k],g);
  for(int k=0;k<c->n_wr;k++) esc_add_rid(e,fi,c->wr[k],g);
}
static void esc_solve(esc_ctx *e){
  uint64_t *top=esc_tmp(e,5);
  do{ e->changed=0;
    esc_clear(e,top); esc_bit(top,0);
    for(int fi=0;fi<e->nf;fi++) if(esc_callers_unknown(e,fi)){ const esc_fn *F=&e->fn[fi];
      for(int p=0;p<F->f->n_params;p++)            /* a scalar only becomes a pointer through a cast */
        if(F->r[F->params[p]].kind!=EK_SCALAR) esc_add(e,F->r[F->params[p]].obj,top); }
    for(int fi=0;fi<e->nf;fi++)
      for(size_t k=0;k<e->fn[fi].f->n_claims;k++) esc_claim(e,fi,&e->fn[fi].f->claims[k]);
  } while(e->changed);
}

/* the objects reachable from `roots` through what they hold (`_closure`), OR-ed into out. 0 on OOM. */
static int esc_closure(esc_ctx *e,const uint64_t *roots,uint64_t *out){
  int *work=esc_alloc(e,(size_t)e->nu+1u,sizeof *work); if(!work) return 0;
  int n=0;
  for(int x=1;x<e->nu;x++) if(esc_has(roots,x) && !esc_has(out,x)){ esc_bit(out,x); work[n++]=x; }
  while(n){ int o=e->uobj[work[--n]]; int k=e->obj[o].kind;
    if(k==EO_F||k==EO_STR||k==EO_TOP) continue;
    const uint64_t *p=esc_pt(e,o);
    for(int x=1;x<e->nu;x++) if(esc_has(p,x) && !esc_has(out,x)){ esc_bit(out,x); work[n++]=x; } }
  return 1;
}

/* the report writer: snprintf semantics over the caller's buffer, the full length counted */
typedef struct { char *buf; size_t cap, w; } esc_out;
static void esc_emit(esc_out *o,const char *s){
  size_t n=strlen(s);
  if(o->buf && o->cap && o->w<o->cap-1u){ size_t room=o->cap-1u-o->w; memcpy(o->buf+o->w,s,n<room?n:room); }
  o->w+=n;
}
static size_t esc_finish(esc_out *o){
  if(o->buf && o->cap) o->buf[o->w<o->cap?o->w:o->cap-1u]=0;
  return o->w;
}
/* sort, deduplicate and `sep`-join `names` (or "-") */
static void esc_emit_names(esc_out *o,const char **names,int n,const char *sep){
  qsort(names,(size_t)n,sizeof *names,esc_strcmp);
  int m=0; for(int i=0;i<n;i++){ if(m && !strcmp(names[i],names[m-1])) continue;
    if(m) esc_emit(o,sep); esc_emit(o,names[i]); names[m++]=names[i]; }
  if(!m) esc_emit(o,"-");
}
/* the refused unit (a call's operands did not all fit a claim): the oracle's `_refused` reports */
static size_t esc_refused(const bcir_unit *u,esc_out *out,int which){
  if(which) esc_emit(out,"truncated=1\n");
  for(int i=0;i<u->n_funcs;i++){ esc_emit(out,"fn="); esc_emit(out,u->funcs[i].name);
    esc_emit(out,which?" refused\n":" reads=* writes=*\n"); }
  if(!which) for(int i=0;i<u->n_funcs;i++) for(int j=i+1;j<u->n_funcs;j++){
    esc_emit(out,"commute "); esc_emit(out,u->funcs[i].name); esc_emit(out," ");
    esc_emit(out,u->funcs[j].name); esc_emit(out," = 0\n"); }
  return esc_finish(out);
}

/* the escape report: each function's named locals by verdict, then its indirect calls' target sets */
static int esc_escape_report(esc_ctx *e,esc_out *out,const uint64_t *lent,const uint64_t *frame){
  for(int fi=0;fi<e->nf;fi++){ const esc_fn *F=&e->fn[fi]; const uint64_t *fr=frame+(size_t)fi*(size_t)e->W;
    const char **nm=esc_alloc(e,(size_t)F->nr+1u,sizeof *nm), **col=esc_alloc(e,(size_t)F->nr+1u,sizeof *col);
    const char **site=esc_alloc(e,F->f->n_claims,sizeof *site);
    int *vd=esc_alloc(e,(size_t)F->nr+1u,sizeof *vd);
    if(!nm||!col||!site||!vd) return 0;
    int m=0;
    for(int k=0;k<F->nr;k++){ const esc_rid *R=&F->r[k]; if(!R->named_local||!R->memory) continue;
      int uo=e->obj[R->obj].u, v=esc_has(fr,uo)?EV_ESCAPING:esc_has(lent,uo)?EV_LENT:EV_NONESCAPING;
      int at=-1; for(int j=0;j<m;j++) if(!strcmp(nm[j],R->name)){ at=j; break; }
      if(at<0){ nm[m]=R->name; vd[m]=v; m++; }         /* one name in two scopes: the more severe stands */
      else if(v>vd[at]) vd[at]=v; }
    esc_emit(out,"fn="); esc_emit(out,F->f->name);
    for(int v=0;v<3;v++){ int nc=0; for(int j=0;j<m;j++) if(vd[j]==v) col[nc++]=nm[j];
      esc_emit(out," "); esc_emit(out,esc_verdicts[v]); esc_emit(out,"="); esc_emit_names(out,col,nc,","); }
    int ns=0;
    for(size_t k=0;k<F->f->n_claims;k++){ const bcir_claim *c=&F->f->claims[k];
      if(!esc_indirect(c->op)||!c->n_rd) continue;
      int nt=esc_targets(e,fi,c);
      if(nt<0){ site[ns++]="*"; continue; }
      size_t len=1; for(int t=0;t<nt;t++) len+=strlen(e->fn[e->tg[t]].f->name)+1u;
      char *s=esc_alloc(e,len,1u); if(!s) return 0;
      size_t w=0; for(int t=0;t<nt;t++){ const char *fnm=e->fn[e->tg[t]].f->name; size_t z=strlen(fnm);
        if(t) s[w++]=','; memcpy(s+w,fnm,z); w+=z; }
      s[w]=0; site[ns++]=s; }
    qsort(site,(size_t)ns,sizeof *site,esc_strcmp);
    esc_emit(out," icalls="); if(!ns) esc_emit(out,"-");
    for(int j=0;j<ns;j++){ if(j) esc_emit(out,";"); esc_emit(out,site[j]); }
    esc_emit(out,"\n"); }
  return 1;
}

/* a device access: an MMIO-domain claim (the oracle's `escape._device`). R3's pass (`order_device_claims`, the
 * last step of every function's lowering) makes each claim that reads or writes a device region MMIO-domain, so
 * the domain already carries the base resource. The analysis once read a load's base as well, when `p[i]`
 * through a `volatile T *` was an ordinary load here; with R3 on both rails no input reached that second
 * reading, so it went, and the rule lives in one place. */
static int esc_device(const bcir_claim *c){
  if(c->opcode==BCIR_OP_NOP) return 0;   /* a control marker (`c.return`, `c.if`, ...) reads a value and performs
                                          * no access; R3 may still make it device-domain when that value is a
                                          * pointer to volatile storage. The oracle has no marker claims. */
  return c->domain==BCIR_DOM_MMIO;
}

/* the effects report: each function's own accesses and everything it can call, named; the commute matrix */
static int esc_effects_report(esc_ctx *e,esc_out *out,const uint64_t *frame){
  size_t NW=(size_t)(e->nobj+63)/64u;
  uint64_t *own=esc_alloc(e,(size_t)e->nf*2u*NW,sizeof *own);   /* per function: reads, writes (all objects) */
  uint64_t *set=esc_alloc(e,(size_t)e->W,sizeof *set), *acc=esc_alloc(e,2u*NW,sizeof *acc);
  const char ***names=esc_alloc(e,(size_t)e->nf*2u,sizeof *names);
  int *nnames=esc_alloc(e,(size_t)e->nf*2u,sizeof *nnames), *work=esc_alloc(e,(size_t)e->nf,sizeof *work);
  uint8_t *reach=esc_alloc(e,(size_t)e->nf,1u);
  if(!own||!set||!acc||!names||!nnames||!work||!reach) return 0;
#define ESC_SET(bits,o) ((bits)[(size_t)(o)>>6]|=(uint64_t)1u<<((o)&63))
  for(int fi=0;fi<e->nf;fi++){ const esc_fn *F=&e->fn[fi]; uint64_t *rd=own+(size_t)fi*2u*NW, *wr=rd+NW;
    for(size_t k=0;k<F->f->n_claims;k++){ const bcir_claim *c=&F->f->claims[k]; const char *op=c->op;
      for(int j=0;j<c->n_rd;j++){ int o=esc_R(F,c->rd[j])->obj, ok=e->obj[o].kind;
        if(ok==EO_G||ok==EO_S||ok==EO_L) ESC_SET(rd,o); }
      for(int j=0;j<c->n_wr;j++){ int o=esc_R(F,c->wr[j])->obj, ok=e->obj[o].kind;
        if(ok==EO_G||ok==EO_S||ok==EO_L) ESC_SET(wr,o); }
      if(esc_device(c)){ ESC_SET(rd,EO_TOP); ESC_SET(wr,EO_TOP); }   /* a device access */
      int touch=0, one=-1;          /* touch: 1 read, 2 write, 3 both -- the object `one` and those in `set` */
      esc_clear(e,set);
      if(!strcmp(op,"c.load")){ if(c->n_rd){ one=esc_deref(e,fi,c->rd[0],set); touch=1; } }
      else if(!strcmp(op,"c.store")){ if(c->n_rd){ one=esc_deref(e,fi,c->rd[0],set); touch=2; } }
      else if(esc_any(op,esc_ops_atomic)){
        for(int j=0;j<c->n_rd;j++){ int x=esc_deref(e,fi,c->rd[j],set); if(x>=0){ ESC_SET(rd,x); ESC_SET(wr,x); } }
        touch=3; }
      else if(esc_any(op,esc_ops_library)){ for(int j=0;j<c->n_rd;j++) esc_val(e,fi,c->rd[j],set); touch=3; }
      else if(esc_any(op,esc_ops_external)||(esc_direct(op)&&esc_find_fn(e,esc_callee(op))<0)){
        ESC_SET(rd,EO_TOP); ESC_SET(wr,EO_TOP); }
      else if(esc_indirect(op)){ if(c->n_rd){
        if(esc_is(op,"c.call.imember")){ one=esc_deref(e,fi,c->rd[0],set); touch=1; }
        if(esc_targets(e,fi,c)<0){ ESC_SET(rd,EO_TOP); ESC_SET(wr,EO_TOP); } } }
      else if(esc_is(op,"c.call.vaarg")) ESC_SET(rd,EO_TOP);
      if(one>=0){ if(touch&1) ESC_SET(rd,one); if(touch&2) ESC_SET(wr,one); }
      if(touch) for(int x=0;x<e->nu;x++) if(esc_has(set,x)){ int o=e->uobj[x];
        if(touch&1) ESC_SET(rd,o); if(touch&2) ESC_SET(wr,o); } } }
#undef ESC_SET
  for(int fi=0;fi<e->nf;fi++){
    memset(reach,0,(size_t)e->nf); int nw=0; reach[fi]=1; work[nw++]=fi;
    while(nw){ int h=work[--nw]; for(int k=0;k<e->fn[h].nedges;k++){ int g2=e->fn[h].edges[k];
      if(!reach[g2]){ reach[g2]=1; work[nw++]=g2; } } }
    memset(acc,0,2u*NW*sizeof *acc);
    for(int h=0;h<e->nf;h++) if(reach[h]){ const uint64_t *src=own+(size_t)h*2u*NW;
      for(size_t w=0;w<2u*NW;w++) acc[w]|=src[w]; }
    for(int side=0;side<2;side++){ const uint64_t *bits=acc+(size_t)side*NW;
      const char **lst=esc_alloc(e,(size_t)e->nobj+1u,sizeof *lst); if(!lst) return 0;
      int m=0;
      for(int o=0;o<e->nobj;o++){ if(!((bits[(size_t)o>>6]>>(o&63))&1u)) continue; const esc_obj *ob=&e->obj[o];
        if(ob->kind==EO_TOP) lst[m++]="*";
        else if(ob->kind==EO_G||ob->kind==EO_S) lst[m++]=ob->name;
        else if(ob->kind==EO_L||ob->kind==EO_H){  /* private to these activations unless reachable */
          if(!reach[ob->fn]||esc_has(frame+(size_t)ob->fn*(size_t)e->W,ob->u)) lst[m++]="*"; } }
      qsort(lst,(size_t)m,sizeof *lst,esc_strcmp);
      int d=0; for(int i=0;i<m;i++) if(!d||strcmp(lst[i],lst[d-1])) lst[d++]=lst[i];
      names[fi*2+side]=lst; nnames[fi*2+side]=d; } }
  for(int fi=0;fi<e->nf;fi++){
    esc_emit(out,"fn="); esc_emit(out,e->fn[fi].f->name);
    esc_emit(out," reads="); esc_emit_names(out,names[fi*2],nnames[fi*2],",");
    esc_emit(out," writes="); esc_emit_names(out,names[fi*2+1],nnames[fi*2+1],",");
    esc_emit(out,"\n"); }
  for(int i=0;i<e->nf;i++) for(int j=i+1;j<e->nf;j++){
    int conflict=0;                 /* a writes what b reads or writes, or b writes what a reads */
    for(int pass=0;pass<3 && !conflict;pass++){
      const char **x=pass<2?names[i*2+1]:names[j*2+1]; int nx=pass<2?nnames[i*2+1]:nnames[j*2+1];
      const char **y=pass==0?names[j*2]:pass==1?names[j*2+1]:names[i*2];
      int ny=pass==0?nnames[j*2]:pass==1?nnames[j*2+1]:nnames[i*2];
      if(!nx||!ny) continue;
      if(!strcmp(x[0],"*")||!strcmp(y[0],"*")){ conflict=1; break; }   /* `*` sorts first: anything */
      for(int p=0,q=0;p<nx&&q<ny;){ int r=strcmp(x[p],y[q]); if(!r){ conflict=1; break; } if(r<0) p++; else q++; } }
    esc_emit(out,"commute "); esc_emit(out,e->fn[i].f->name); esc_emit(out," ");
    esc_emit(out,e->fn[j].f->name); esc_emit(out,conflict?" = 0\n":" = 1\n"); }
  return 1;
}

/* the analysis and one report: which==0 the effects report, which==1 the escape report. Returns the
 * complete report's length (snprintf semantics: at most n-1 bytes and a NUL are written), or SIZE_MAX
 * when an allocation failed (buf is then empty). */
static size_t esc_report(const bcir_unit *u,char *buf,size_t n,const bcir_host_allocator *allocator,int which){
  bcir_host_arena arena; esc_ctx E; esc_ctx *e=&E; esc_out out={buf,n,0}; int ok=0;
  if(buf&&n) buf[0]=0;
  if(!u) return 0;
  for(int i=0;i<u->n_funcs;i++) for(size_t k=0;k<u->funcs[i].n_claims;k++)
    if(u->funcs[i].claims[k].truncated) return esc_refused(u,&out,which);
  if(!bcir_host_arena_init(&arena,allocator,4096u)) return SIZE_MAX;
  memset(e,0,sizeof *e); e->arena=&arena; e->u=u;
  if(esc_build(e)){
    esc_solve(e);
    /* the call graph, direct and narrowed */
    uint8_t *callee=esc_alloc(e,(size_t)e->nf,1u);
    int graph=callee!=NULL;
    for(int fi=0;fi<e->nf && graph;fi++){ esc_fn *F=&e->fn[fi];
      memset(callee,0,(size_t)e->nf);
      for(size_t k=0;k<F->f->n_claims;k++){ const bcir_claim *c=&F->f->claims[k];
        if(esc_direct(c->op)){ int ci=esc_find_fn(e,esc_callee(c->op)); if(ci>=0) callee[ci]=1; }
        else if(esc_indirect(c->op) && c->n_rd){ int nt=esc_targets(e,fi,c); for(int t=0;t<nt;t++) callee[e->tg[t]]=1; } }
      F->edges=esc_alloc(e,(size_t)e->nf,sizeof *F->edges); if(!F->edges){ graph=0; break; }
      for(int ci=0;ci<e->nf;ci++) if(callee[ci]) F->edges[F->nedges++]=ci; }
    /* escape: what unknown code, a global or a static can reach, and what a function returns */
    uint64_t *roots=esc_alloc(e,(size_t)e->W,sizeof *roots), *escaped=esc_alloc(e,(size_t)e->W,sizeof *escaped);
    uint64_t *lent=esc_alloc(e,(size_t)e->W,sizeof *lent);
    uint64_t *frame=esc_alloc(e,(size_t)e->nf*(size_t)e->W,sizeof *frame);
    if(graph && roots && escaped && lent && frame){
      esc_or(e,roots,e->esc);
      for(int i=0;i<e->nobj;i++) if(e->obj[i].kind==EO_G||e->obj[i].kind==EO_S) esc_or(e,roots,esc_pt(e,i));
      ok=esc_closure(e,roots,escaped);
      esc_clear(e,roots);
      for(int fi=0;fi<e->nf && ok;fi++){ const bcir_func *f=e->fn[fi].f;
        for(size_t k=0;k<f->n_claims;k++){ const bcir_claim *c=&f->claims[k];
          if(!esc_is(c->op,"c.call") && !esc_any(c->op,esc_ops_external)) continue;
          for(int j=esc_indirect(c->op)?1:0;j<c->n_rd;j++) esc_val(e,fi,c->rd[j],roots); } }
      ok=ok && esc_closure(e,roots,lent);
      for(int fi=0;fi<e->nf && ok;fi++){ uint64_t *fr=frame+(size_t)fi*(size_t)e->W; const esc_fn *F=&e->fn[fi];
        esc_clear(e,roots);
        for(int k=0;k<F->nret;k++) esc_val(e,fi,F->r[F->rets[k]].rid,roots);
        ok=esc_closure(e,roots,fr); esc_or(e,fr,escaped); }
      if(ok) ok=which?esc_escape_report(e,&out,lent,frame):esc_effects_report(e,&out,frame);
    }
  }
  bcir_host_arena_destroy(&arena);
  if(!ok){ if(buf&&n) buf[0]=0; return SIZE_MAX; }
  return esc_finish(&out);
}

size_t bcir_cfront_effects_with_allocator(const bcir_unit *u,char *buf,size_t n,
                                          const bcir_host_allocator *allocator){
  return esc_report(u,buf,n,allocator,0);
}
size_t bcir_cfront_effects(const bcir_unit *u,char *buf,size_t n){
  bcir_host_allocator allocator=bcir_host_allocator_default();
  return bcir_cfront_effects_with_allocator(u,buf,n,&allocator);
}
size_t bcir_cfront_escape_with_allocator(const bcir_unit *u,char *buf,size_t n,
                                         const bcir_host_allocator *allocator){
  return esc_report(u,buf,n,allocator,1);
}
size_t bcir_cfront_escape(const bcir_unit *u,char *buf,size_t n){
  bcir_host_allocator allocator=bcir_host_allocator_default();
  return bcir_cfront_escape_with_allocator(u,buf,n,&allocator);
}
