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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
                 int is_volatile;   /* the member's own storage is volatile (`volatile T m`, `volatile T m[N]`, a
                                     * volatile struct member): an access to it is a volatile access */
                 int ptee_volatile; /* a pointer member to volatile storage (`volatile T *regs`): the member is
                                     * plain storage; what it points at is a device region */
                 int elem_sidx;     /* arr_count>0 AND element is a value struct: its sdef index (array-of-structs,
                                     * for `arr[i].field`); -1 otherwise. Distinct from `sidx` so member_descend
                                     * (which descends a `.` only when sidx>=0) never walks an un-indexed array. */
                 int fp_ret_size; uint8_t fp_ret_signd; uint8_t fp_ret_float;
                                    /* a funcptr struct member: the captured RETURN type (sign/width/float),
                                     * used to type a c.call.imember result temp; ZERO if not a funcptr / not captured */
                 } field;
                 /* is_ptr: a pointer member -- `size` is pointer_size (the ABI layout width), the pointee
                  * (ptee_size width / signd sign / ptee_float / ptee_sidx struct) types the loaded `T *` */
                 /* arr_count > 0: a member array; `size` is the element, arr_count the total element
                  * count, nadims/adims the per-dim sizes (`T m[A][B]` -> nadims 2, adims {A,B}) */
typedef struct { char tag[BCIR_CIR_NAME]; field f[MAXFLD]; int nf; int size; int align; int is_union;
                 int vol_storage;   /* the struct holds volatile storage: a volatile member, or a value-struct /
                                     * array-of-structs member that does (a pointer member does not) -- the
                                     * oracle's `CType.volatile_storage` */
               } sdef;
typedef struct { char name[BCIR_CIR_NAME]; bcir_ctype ty; int sidx; } tdef;   /* a typedef alias */
typedef struct { char name[BCIR_CIR_NAME]; long long val; } econst;           /* an enum constant */

typedef struct {
  char name[BCIR_CIR_NAME];
  uint32_t rid;
  bcir_ctype type;   /* scalar / struct-by-value / pointer */
  int sidx;          /* struct index for kind 1 or ptr_to_struct */
} venv;

typedef struct { char name[BCIR_CIR_NAME]; bcir_ctype ty; int count;
                 int is_arr;          /* declared with `[...]` (an array of any length) */
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
} bcir_abi;

/* The named matrix (mirrors abi.py TARGETS). x86-64 / AArch64 / RISC-V are all LP64, so their
 * layouts coincide; Windows x64 is LLP64 (long is 4) and 32-bit x86 is ILP32 (pointers are 4) -- the
 * cases that change what the frontend lays out. g_targets[0] is the default (host LP64) model, so
 * --target-less compilation is byte-identical to the layout used before --target existed. */
static const bcir_abi g_targets[] = {
  {"x86_64-linux",   "x86_64-unknown-linux-gnu",  "LP64",  8, 8, 16, 16},
  {"aarch64-linux",  "aarch64-unknown-linux-gnu", "LP64",  8, 8, 16, 16},
  {"riscv64-linux",  "riscv64-unknown-linux-gnu", "LP64",  8, 8, 16, 16},
  {"x86_64-windows", "x86_64-pc-windows-msvc",    "LLP64", 4, 8,  8,  8},
  {"i386-linux",     "i386-unknown-linux-gnu",    "ILP32", 4, 4, 12,  4},
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
  char fpdefs[4096]; size_t fpdefs_w; int n_fpdef;   /* synthesized `typedef RET (*__bcir_fpN)(PARAMS);`
                                                      * lines for direct funcptr-param declarators (which
                                                      * have no source typedef to print), emitted as a
                                                      * prelude before the function bodies */
  char tudefs[4096]; size_t tudefs_w;                /* Phase 3 linking: `extern RET NAME(PARAMS);` lines
                                                      * rendered at each PROTOTYPE, emitted as a prelude so
                                                      * the emitted TU compiles standalone and the host
                                                      * LINKER resolves the cross-TU callee */
  int emit_overflow;                                  /* a required prelude/body exceeded a fixed emit buffer:
                                                      * reject cleanly, never compile a truncated C artifact */
  int saw_static;                                    /* p_type_base scanned a `static` since p_func last
                                                      * reset it -- captured as fn->static_fn right after
                                                      * the return-type parse (source-static honoring) */
  struct { char name[BCIR_CIR_NAME]; bcir_ctype ret; } *protos;   /* prototype table: callee -> return
                                                      * type (for call-result typing); grows geometrically */
  int n_protos, cap_protos;
  /* §5.12 per-function mutation pre-pass (the C twin of _mut_assigned / _mut_addr): over the whole
   * function body, the number of assignments to each NAME and whether its address is ever taken (`&x`).
   * Drives extent-stability -- a recovered count / a bound pointer is trusted only when STABLE (assigned
   * at most its single binding and never aliased). Conservative: an over-approximation never promotes
   * unsoundly. Reset (mut_n=0) per function. */
  mutent mut[512]; int mut_n;
  int ext_ctr;       /* §5.12 unique hidden extent-snapshot locals (`__bcir_extK`); reset per function */
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
/* The active target ABI (defaults to the host LP64 model when the driver set none). */
static const bcir_abi *cc_abi(const CC *c){ return c->abi ? c->abi : bcir_abi_host(); }

static int is_idc(int c){return c=='_'||(c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9');}
static int is_id0(int c){return c=='_'||(c>='a'&&c<='z')||(c>='A'&&c<='Z');}

static long long parse_int(const char *s, int n) {
  char buf[64]; int j=0;
  for (int k=0;k<n&&j<63;k++) if (s[k]!='\'') buf[j++]=s[k];
  buf[j]=0;
  while (j>0&&(buf[j-1]=='u'||buf[j-1]=='U'||buf[j-1]=='l'||buf[j-1]=='L')) buf[--j]=0;
  if (j>1&&buf[0]=='0'&&(buf[1]=='x'||buf[1]=='X')) return strtoll(buf,NULL,16);
  if (j>1&&buf[0]=='0'&&(buf[1]=='b'||buf[1]=='B')) return strtoll(buf+2,NULL,2);
  return strtoll(buf,NULL,10);
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

/* The element size of a (possibly prefixed) string literal on the Linux/Clang ABI: plain/u8 = 1
 * (char), u = 2 (char16_t), L/U = 4 (wchar_t / char32_t). s[0..n) is the spelling incl. the prefix. */
static int str_elem_size(const char *s, int n) {
  if (n>0) { if (s[0]=='u') return (n>1 && s[1]=='8') ? 1 : 2; if (s[0]=='L'||s[0]=='U') return 4; }
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
    if (is_id0(*p)){t->k=T_ID;t->s=p;while(is_idc(*p))p++;t->n=(int)(p-t->s);c->nt++;continue;}
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
        t->k=T_FLT;t->s=p;t->n=(int)(q-p);p=q;c->nt++;continue; }
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
          t->k=T_FLT;t->s=p;t->n=(int)(q-p);p=q;c->nt++;continue; } }
    }
    if (*p>='0'&&*p<='9'){t->k=T_INT;t->s=p;while(is_idc(*p)||*p=='\'')p++;t->n=(int)(p-t->s);
                          t->v=parse_int(t->s,t->n);c->nt++;continue;}
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
static void idcpy(char *d,const tok *t){int n=t->n<BCIR_CIR_NAME-1?t->n:BCIR_CIR_NAME-1;memcpy(d,t->s,n);d[n]=0;}

/* --- types --------------------------------------------------------------- */
static int scalar_size(const char *s,int n) {
  struct {const char *k;int sz;} T[]={{"void",0},{"char",1},{"bool",1},{"_Bool",1},{"short",2},
    {"int",4},{"unsigned",4},{"signed",4},{"long",8},{"uint8_t",1},{"int8_t",1},{"uint16_t",2},{"int16_t",2},
    {"uint32_t",4},{"int32_t",4},{"uint64_t",8},{"int64_t",8},
    {"size_t",8},{"intptr_t",8},{"uintptr_t",8},     /* the pointer-tracking size_t-class types */
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
static int find_global(CC *c,const char *s,int n){
  for(int i=0;i<c->ngv;i++) if((int)strlen(c->gv[i].name)==n&&!strncmp(c->gv[i].name,s,n)) return i;
  return -1;
}
static venv *use_global(CC *c,const tok *id);   /* fwd: materialize a global's resource on first use */
static void p_enum_body(CC *c);   /* fwd: `{ A, B=expr, C }` -> register the constants */

/* a parsed type: fills a bcir_ctype + the struct index (sidx, or -1). */
/* Apply a declarator's leading `*`s to a (base) type: each `*` raises the pointer depth (a struct base
 * becomes a pointer-to-struct), consuming any cv/restrict qualifier after it. Split out of p_type so a
 * multi-declarator declaration applies stars PER DECLARATOR (`int *p, q;` -> p is `int*`, q is int). */
static void apply_stars(CC *c, bcir_ctype *ty) {
  while(is(c,"*")){c->i++;
    while(is(c,"const")||is(c,"volatile")||is(c,"restrict")||is(c,"__restrict")||is(c,"__restrict__"))c->i++;
    if(ty->ptr_depth>=BCIR_MAX_PTR_DEPTH){fail(c,"pointer nesting too deep");return;}
    if(ty->kind==1){ty->ptr_to_struct=1;} ty->kind=2; ty->ptr_depth++;}   /* count `*`s: `T**` -> depth 2 */
}
/* Parse a type SPECIFIER (the base scalar/struct/union/enum/typedef + qualifiers + the data-model size
 * fixups), WITHOUT the declarator `*`s. p_type folds the stars on top; the multi-declarator paths call
 * this and apply_stars per declarator instead. */
static int p_type(CC *c, bcir_ctype *ty, int *sidx);          /* fwd: typeof(type-name) parses recursively */
static venv *lookup(CC *c, const tok *t);                     /* fwd: typeof(variable) resolves its type */
static int p_typeof_expr(CC *c, bcir_ctype *ty, int *sidx);   /* fwd: typeof(expression) -- speculative lower */
static int p_type_base(CC *c, bcir_ctype *ty, int *sidx) {
  memset(ty,0,sizeof *ty); ty->kind=0; ty->size=4; ty->signd=1; *sidx=-1;
  int seen=0, longs=0, ptrtrk=0, sign_explicit=0, floatkw=0;   /* longs: `long` (data-model) vs `long long` (8);
                                                               * floatkw: a float/double keyword was scanned */
  for(;;){
    if(is(c,"volatile")){ty->is_volatile=1;c->i++;continue;}
    if(is(c,"_Atomic")){ c->i++;
      if(is(c,"(")){ c->i++; bcir_ctype inner; int isi;    /* `_Atomic ( type-name )` -- atomic type specifier */
        if(p_type(c,&inner,&isi)) return 1; if(!eat(c,")")) return 1;
        int vol=ty->is_volatile; *ty=inner; ty->is_atomic=1; if(vol)ty->is_volatile=1; *sidx=isi; seen=1; break; }
      ty->is_atomic=1; continue; }
    if(is(c,"static")){c->saw_static=1;c->i++;continue;}   /* recorded: p_func captures it right after
                                                            * its return-type parse (source-static
                                                            * honoring in --linkable); block-scope
                                                            * statics peek the token BEFORE p_type. */
    if(is(c,"const")||is(c,"inline")||is(c,"extern")
       ||is(c,"_Thread_local")||is(c,"thread_local")){c->i++;continue;}  /* storage class / qualifier */
    if(is(c,"typeof")||is(c,"__typeof__")||is(c,"typeof_unqual")){       /* typeof(type-name) / typeof(var) */
      c->i++; if(!eat(c,"(")) return 1;
      int is_type = scalar_size(pk(c)->s,pk(c)->n)>=0 || is(c,"struct")||is(c,"union")||is(c,"enum")||is(c,"_Complex")||is(c,"complex")||is(c,"_BitInt")
                    || is(c,"const")||is(c,"volatile")
                    || is(c,"typeof")||is(c,"__typeof__")||is(c,"typeof_unqual")
                    || find_typedef(c,pk(c)->s,pk(c)->n)>=0;
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
    if(is(c,"_Complex")||is(c,"complex")){ty->is_complex=1;ty->is_float=1;seen=1;c->i++;continue;}  /* C99 _Complex
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
    if(is(c,"struct")||is(c,"union")){c->i++;tok tag=adv(c);int si=find_struct(c,tag.s,tag.n);
      if(si<0){fail(c,"unknown struct");return 1;} ty->kind=1;ty->size=c->s[si].size;*sidx=si;
      ty->is_union=(uint8_t)c->s[si].is_union;idcpy(ty->tag,&tag);seen=1;break;}
    if(is(c,"enum")){c->i++;if(isk(c,T_ID)&&!is(c,"{"))c->i++;   /* `enum [tag] [{...}]` -> int */
      if(is(c,"{"))p_enum_body(c); ty->kind=0;ty->size=4;ty->signd=1;seen=1;break;}
    if(is(c,"va_list")||is(c,"__builtin_va_list")){              /* the variadic cursor type (<stdarg.h>) -- */
      ty->kind=0;ty->is_valist=1;ty->size=cc_abi(c)->pointer_size;ty->signd=0;c->i++;seen=1;break;}  /* opaque, emit `va_list` */
    if(!seen&&isk(c,T_ID)){int ti=find_typedef(c,pk(c)->s,pk(c)->n);   /* a typedef alias */
      if(ti>=0){int vol=ty->is_volatile;*ty=c->td[ti].ty;if(vol)ty->is_volatile=1;*sidx=c->td[ti].sidx;c->i++;seen=1;break;}}
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
  if(!seen){fail(c,"expected a type");return 1;}
  /* apply the target data model: `size_t`-class -> pointer_size; `long double` -> long_double_size; a
   * single `long` -> long_size (`long long` keeps its fixed 8). On the host LP64 model long/ptr are 8. */
  if(ptrtrk) ty->size=cc_abi(c)->pointer_size;
  else if(longs>=1&&ty->is_float) ty->size=cc_abi(c)->long_double_size;   /* `long double` (80/128-bit) */
  else if(longs==1&&!ty->is_float&&ty->kind==0) ty->size=cc_abi(c)->long_size;
  if(ty->is_complex) ty->size = (floatkw ? ty->size : 8) * 2;   /* a complex is a pair of the element float;
                                                                 * a bare `_Complex` (no float kw) is double */
  return 0;
}
/* The full type: the specifier + the (first declarator's) `*`s folded in -- the single-declarator /
 * type-name form (params, casts, sizeof/_Alignof, the first declarator of a declaration). */
static int p_type(CC *c, bcir_ctype *ty, int *sidx) {
  if(ENTER_REC(c)){ LEAVE_REC(c); return 1; }   /* depth guard: typeof/_Atomic re-enter p_type */
  int r=p_type_base(c,ty,sidx); if(r){ LEAVE_REC(c); return r; }
  apply_stars(c,ty); LEAVE_REC(c); return 0;
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
/* Parse `struct|union [tag] [attrs] { members } [attrs]` (NO trailing `;`). Registers an sdef and
 * returns its index (-1 on error). An anonymous aggregate (no tag, e.g. `typedef struct {...} N;`)
 * gets a synthesized internal tag so a typedef can alias it. */
static int p_struct_body(CC *c) {
  int is_union = is(c,"union");
  c->i++; int packed=0,aligned=0; attrs(c,&packed,&aligned);
  CC_ENSURE(c, c->s, c->ns, c->cap_s);
  if(c->ns>=c->cap_s){ fail(c,"too many struct definitions"); return -1; }
  int my=c->ns++;                       /* claim our slot NOW: an inline aggregate member recurses into
                                         * p_struct_body and must take a LATER slot (and may realloc c->s). */
  sdef *S=&c->s[my]; S->nf=0; S->align=1; S->is_union=is_union; S->vol_storage=0;
  if(isk(c,T_ID)&&!is(c,"{")){tok tag=adv(c);idcpy(S->tag,&tag);}
  else snprintf(S->tag,sizeof S->tag,"$anon%d",my);   /* anonymous: synth a unique tag */
  attrs(c,&packed,&aligned);
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
      for(int k=0;k<A->nf;k++){
        if(S->nf>=MAXFLD){ fail(c,"too many struct members"); return -1; }
        field nf=A->f[k]; nf.byte_off+=anon_off; S->f[S->nf++]=nf;   /* shift each leaf's offset into S */
      }
      eat(c,";");
      continue;
    }
    if(!inl && p_type_base(c,&base,&si))return -1;
    for(;;){                                          /* one or more declarators off one specifier: */
      bcir_ctype ty=base; apply_stars(c,&ty);   /* per-declarator `*`: `int *p, q;` -> p ptr, q scalar */
      if(is(c,":")){                                  /* an UNNAMED `int :3` / ZERO-WIDTH `int :0` bitfield (no
                                                       * name): positions the cursor, NOT a field, no align bump. */
        c->i++; int w=(int)adv(c).v;
        if(!is_union){ int ub=ty.size*8;
          if(w==0){ if(dbits%ub)dbits+=ub-(dbits%ub); }       /* zero-width -> next storage-unit boundary */
          else if(packed){ dbits+=w; }                        /* packed: pack bit-by-bit */
          else { if((int)(dbits%ub)+w>ub)dbits+=ub-(dbits%ub); dbits+=w; }
        }
        if(is(c,",")){c->i++;continue;} break;
      }
      tok nm;
      if(is(c,"(") && tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='*'
         && tat(c,c->i+2)->k==T_ID){       /* a function-pointer member `RET (*name)(params)` -> a kind-3 (8-byte)
                                          * field. The struct definition comes from the source (not emitted), so
                                          * no signature is captured; set via a funcptr value + called via
                                          * `o->fn(args)` (the existing c.call.imember machinery). */
        bcir_ctype rty=ty;                                   /* snapshot the parsed RETURN type before the memset */
        c->i+=2; nm=adv(c);                                  /* `( *` then the name */
        if(!eat(c,")")||!eat(c,"("))return -1;
        for(int dd=1; dd>0 && !isk(c,T_END) && !c->failed;){ if(is(c,"("))dd++; else if(is(c,")"))dd--; if(dd>0)c->i++; }
        eat(c,")");                                          /* past the parameter-type list */
        memset(&ty,0,sizeof ty); ty.kind=3; ty.size=8; ty.signd=0;
        ty.fp_ret_size=rty.size; ty.fp_ret_signd=(uint8_t)(rty.signd?1:0);   /* carry the funcptr's return type, */
        ty.fp_ret_float=(uint8_t)(rty.is_float?1:0);                          /* used to type a c.call.imember result */
      } else {
        if(!isk(c,T_ID)){ fail(c,"expected member name"); return -1; }   /* `unsigned x, y, z;` etc. */
        nm=adv(c);
      }
      int arr_count=0,nadims=0,adims[3]={0,0,0};        /* T arr[N] / T m[A][B] -- one or more dims */
      while(is(c,"[")){ c->i++; int dim=isk(c,T_INT)?(int)adv(c).v:0; eat(c,"]");
        if(nadims<3)adims[nadims]=dim; nadims++; arr_count = arr_count ? arr_count*dim : dim; }
      if(nadims>3){ fail(c,"member array of more than 3 dimensions"); return -1; }   /* adims[] caps at 3 */
      int width=0; if(is(c,":")){c->i++;width=(int)adv(c).v;}          /* per-declarator bitfield width */
      if(ty.bit_width>0 && width && !(width>=1 && width<=ty.bit_width)){   /* a `_BitInt(N)` BITFIELD: W in 1..N */
        fail(c,"a `_BitInt` bitfield width outside 1..N is not supported"); return -1; }   /* W>N is invalid C */
      /* a `_BitInt(N)` BITFIELD `_BitInt(N) m : W` (1<=W<=N) is first-class: `ty.size` is the Clang storage slot
       * (1/2/4/8 bytes) so it packs into the `_BitInt(N)` storage unit LSB-first exactly like a standard-int
       * bitfield of that size (byte-identical to Clang); a PLAIN `_BitInt(N)` member likewise uses that slot.
       * Either way the member's exact width rides in f->bit_width so the load/store + emit spell `_BitInt(N)`. */
      if(S->nf>=MAXFLD){ fail(c,"too many struct members"); return -1; }   /* f[] embedded; guarded */
      int isptr=(ty.kind==2 && !arr_count);            /* a (non-array) pointer member: ABI pointer_size */
      int sz=isptr?cc_abi(c)->pointer_size:ty.size;
      /* a (array of) value-struct/union member aligns to the NESTED type's alignment, not its size --
       * `struct{int;struct Big t;}` puts t at the struct's align, not at sizeof(Big) (which over-pads). */
      int al = packed?1 : (ty.kind==1 && !ty.ptr_to_struct && si>=0) ? (c->s[si].align<1?1:c->s[si].align)
                        : (sz<1?1:sz);
      if(maln>al) al=maln;                            /* `_Alignas(N)`/`aligned(N)` over-aligns (survives packed) */
      field *f=&S->f[S->nf++];
      int total=arr_count?sz*arr_count:sz;             /* the bytes the member occupies (array: N*elem) */
      idcpy(f->name,&nm);f->size=sz;f->access_bytes=sz;f->signd=ty.signd;f->bit_w=width;f->arr_count=arr_count;
      f->bit_width=(!isptr && !arr_count && ty.bit_width>0)?ty.bit_width:0;   /* a C23 `_BitInt(N)` member (plain OR
                                                                              * bitfield): exact N; f->bit_w holds W */
      f->is_float=(!isptr && ty.is_float)?1:0;          /* a float/double member loads/stores as itself */
      f->is_complex=(!isptr && ty.is_complex)?1:0;      /* a `_Complex` member: load/store as the complex pair,
                                                         * NOT a same-size real (16B would wrongly read as long double) */
      f->is_bool=(!isptr && ty.is_bool)?1:0;            /* a _Bool member: a store normalizes any nonzero to 1 */
      f->is_plain_char=(!isptr && ty.is_plain_char)?1:0;/* a plain `char` member: read as `char` (impl-defined
                                                         * sign), NOT int8_t -- `char` is UNSIGNED on AArch64 */
      f->nadims=nadims; for(int z=0;z<3;z++) f->adims[z]=adims[z];
      f->is_ptr=isptr; f->ptee_size=isptr?ty.size:0; f->ptee_float=isptr?(ty.is_float?1:0):0;   /* pointee type */
      f->ptee_sidx=(isptr && ty.ptr_to_struct)?si:-1;  /* a pointer-to-struct member: the pointee struct tag */
      f->is_volatile=(ty.kind!=2 && ty.kind!=3 && ty.is_volatile)?1:0;   /* `volatile T m` (a pointer's `volatile`
                                                                           * qualifies its pointee, not itself) */
      f->ptee_volatile=(ty.kind==2 && ty.is_volatile)?1:0;               /* `volatile T *m`: the pointee */
      f->sidx = (ty.kind==1 && !ty.ptr_to_struct && !arr_count) ? si : -1;   /* value struct member -> nested */
      f->elem_sidx = (ty.kind==1 && !ty.ptr_to_struct && arr_count) ? si : -1;   /* array-of-structs element struct */
      f->fp_ret_size=ty.fp_ret_size; f->fp_ret_signd=ty.fp_ret_signd; f->fp_ret_float=ty.fp_ret_float;   /* funcptr member: return type */
      if(al>S->align)S->align=al; if(total>maxsz)maxsz=total;
      if(is_union){f->byte_off=0;f->bit_off=0;}        /* union: every member overlaps at offset 0 */
      else if(width){int ub=sz*8;
        if(packed){                                     /* packed: pack bit-by-bit, NO storage-unit reservation
          * (Clang/GCC) -- the field sits at the running bit cursor and its access unit is just the bytes it
          * spans (`access_bytes`), which may straddle byte/word boundaries; the struct stays align 1. */
          int P=(int)dbits; f->byte_off=P/8; f->bit_off=P%8;
          f->access_bytes=(f->bit_off+width+7)/8; dbits+=width;
        }else{                                          /* natural: pack at the bit cursor, NOT a fresh unit */
          if((int)(dbits%ub)+width>ub)dbits+=ub-(dbits%ub);   /* would cross a storage-unit boundary -> bump */
          int uoff=(int)(dbits/ub)*sz; f->byte_off=uoff;f->bit_off=(int)(dbits-(long long)uoff*8);dbits+=width;
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
  int salign = packed ? 1 : S->align; if(aligned>salign) salign=aligned; S->align=salign;
  int total = is_union ? maxsz : (int)((dbits+7)/8);   /* union size = the widest member; struct: bits->bytes */
  if(total%salign)total+=salign-(total%salign); S->size=total;
  return my;
}

/* --- enum + typedef (resolved at parse time so the claim graph carries the folded result) --- */
static long long ce_expr(CC *c,int minp);
static long long ce_primary(CC *c){
  if(isk(c,T_INT))return adv(c).v;
  if(is(c,"(")){c->i++;long long v=ce_expr(c,0);eat(c,")");return v;}
  if(is(c,"-")){c->i++;return -ce_primary(c);}
  if(is(c,"+")){c->i++;return ce_primary(c);}
  if(is(c,"~")){c->i++;return ~ce_primary(c);}
  if(is(c,"!")){c->i++;return !ce_primary(c);}
  if(isk(c,T_ID)){int e=find_enum(c,pk(c)->s,pk(c)->n);if(e>=0){c->i++;return c->ec[e].val;}}
  fail(c,"non-constant enum initializer");return 0;
}
static long long ce_expr(CC *c,int minp){
  if(ENTER_REC(c)){ LEAVE_REC(c); return 0; }   /* depth guard: ce_expr<->ce_primary `(...)` cycle */
  /* the SS5.9 integer constant-expression evaluator (the oracle's _const_eval twin): full C
   * precedence over ||, &&, bit ops, equality, relational, shift, arithmetic -- both sides of
   * a logical op evaluate (a constant expression has no side effects to short-circuit away). */
  struct{const char*t;int p;}P[]={{"||",1},{"&&",2},{"|",3},{"^",4},{"&",5},
    {"==",6},{"!=",6},{"<=",7},{">=",7},{"<",7},{">",7},{"<<",8},{">>",8},
    {"+",9},{"-",9},{"*",10},{"/",10},{"%",10},{0,0}};
  long long lhs=ce_primary(c);
  for(;;){int p=-1;const char*op=0;
    for(int i=0;P[i].t;i++) if(is(c,P[i].t)){p=P[i].p;op=P[i].t;break;}
    if(p<minp||p<0)break; c->i++; long long rhs=ce_expr(c,p+1);
    lhs = !strcmp(op,"+")?lhs+rhs:!strcmp(op,"-")?lhs-rhs:!strcmp(op,"*")?lhs*rhs:
          !strcmp(op,"/")?(rhs?lhs/rhs:0):!strcmp(op,"%")?(rhs?lhs%rhs:0):
          !strcmp(op,"&")?lhs&rhs:!strcmp(op,"|")?lhs|rhs:!strcmp(op,"^")?lhs^rhs:
          !strcmp(op,"<<")?lhs<<rhs:!strcmp(op,">>")?lhs>>rhs:
          !strcmp(op,"==")?lhs==rhs:!strcmp(op,"!=")?lhs!=rhs:
          !strcmp(op,"<=")?lhs<=rhs:!strcmp(op,">=")?lhs>=rhs:
          !strcmp(op,"<")?lhs<rhs:!strcmp(op,">")?lhs>rhs:
          !strcmp(op,"&&")?(lhs&&rhs):(lhs||rhs);
  }
  if(minp==0&&is(c,"?")){                        /* the ternary (right-assoc, lowest precedence) */
    c->i++; long long a=ce_expr(c,0); if(!eat(c,":")){LEAVE_REC(c);return 0;}
    long long b=ce_expr(c,0); lhs = lhs?a:b;
  }
  LEAVE_REC(c); return lhs;
}
static void p_enum_body(CC *c){
  eat(c,"{"); long long val=0;
  while(!is(c,"}")&&!c->failed){
    tok nm=adv(c);
    if(is(c,"=")){c->i++;val=ce_expr(c,0);}
    CC_ENSURE(c,c->ec,c->nec,c->cap_ec);
    if(c->nec<c->cap_ec){idcpy(c->ec[c->nec].name,&nm);c->ec[c->nec].val=val;c->nec++;}
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
    else { c->i++; tok tag=adv(c); int si=find_struct(c,tag.s,tag.n);
      if(si<0){fail(c,"unknown struct in typedef");return;} ty.kind=1;ty.size=c->s[si].size;sidx=si;
      ty.is_union=(uint8_t)c->s[si].is_union;idcpy(ty.tag,&tag);}
    while(is(c,"*")){c->i++;
      while(is(c,"const")||is(c,"volatile")||is(c,"restrict")||is(c,"__restrict")||is(c,"__restrict__"))c->i++;
      ty.ptr_to_struct=(ty.kind==1);ty.kind=2;}
  } else if(is(c,"enum")){                            /* alias an enum -> an int scalar */
    c->i++; if(isk(c,T_ID)&&!is(c,"{"))c->i++; if(is(c,"{"))p_enum_body(c);
    ty.kind=0; ty.size=4; ty.signd=1;
  } else {
    if(p_type(c,&ty,&sidx))return;                    /* scalar / pointer / typedef-of-typedef */
  }
  if(is(c,"(")){                                       /* typedef RET (*NAME)(PARAMS); -- a funcptr */
    int save=c->i; c->i++;
    if(is(c,"*")){
      c->i++; tok nm=adv(c); eat(c,")"); eat(c,"(");
      for(int d=1; d>0 && !isk(c,T_END) && !c->failed; ){ /* skip the parameter-type list */
        if(is(c,"(")) d++; else if(is(c,")")) d--;
        if(d>0) c->i++;
      }
      eat(c,")");
      bcir_ctype fp; memset(&fp,0,sizeof fp); fp.kind=3; fp.size=8; fp.signd=0; idcpy(fp.tag,&nm);
      fp.fp_ret_size=ty.size; fp.fp_ret_signd=(uint8_t)(ty.signd?1:0); fp.fp_ret_float=(uint8_t)(ty.is_float?1:0);
                                                         /* carry the funcptr's RETURN type via the typedef to every use */
      CC_ENSURE(c,c->td,c->ntd,c->cap_td);
      if(c->ntd<c->cap_td){idcpy(c->td[c->ntd].name,&nm);c->td[c->ntd].ty=fp;c->td[c->ntd].sidx=-1;c->ntd++;}
      eat(c,";");
      return;
    }
    c->i=save;                                         /* not a funcptr declarator */
  }
  tok nm=adv(c);                                      /* the alias name */
  CC_ENSURE(c,c->td,c->ntd,c->cap_td);
  if(c->ntd<c->cap_td){idcpy(c->td[c->ntd].name,&nm);c->td[c->ntd].ty=ty;c->td[c->ntd].sidx=sidx;c->ntd++;}
  eat(c,";");
}

/* --- the IR builder ------------------------------------------------------ */
static uint32_t add_res(CC *c, bcir_domain dom, int elem, int count, int vol, int kind, const char *nm) {
  bcir_func *f=c->fn;
  if(!cc_ensure_size(c,(void **)&f->res,f->n_res,&f->cap_res,sizeof *f->res,16u)) return 0;
  bcir_resource *r=&f->res[f->n_res++]; memset(r,0,sizeof *r);   /* realloc slots are uninitialized */
  r->rid=c->rid++; r->domain=dom; r->elem_bytes=elem<1?1:elem; r->count=count<1?1:count;
  r->is_volatile=(uint8_t)vol; r->read_only=0; r->kind=(uint8_t)kind; r->agg[0]=0;
  snprintf(r->name,sizeof r->name,"%s",nm?nm:"");
  return r->rid;
}
static bcir_claim *new_claim(CC *c,const char *op,bcir_opcode opc) {
  bcir_func *f=c->fn;
  if(!cc_ensure_size(c,(void **)&f->claims,f->n_claims,&f->cap_claims,
                     sizeof *f->claims,32u)) return NULL;
  bcir_claim *cl=&f->claims[f->n_claims++]; memset(cl,0,sizeof *cl);
  cl->id=c->cid++;cl->opcode=opc;cl->lane=BCIR_LANE_U;cl->stride=BCIR_STRIDE_SCALAR;cl->count=1;
  cl->domain=BCIR_DOM_RAM;cl->hazard=BCIR_HZ_UNIQUE;cl->bounds=BCIR_BND_STRICT;
  snprintf(cl->op,sizeof cl->op,"%s",op); return cl;
}
static uint32_t temp(CC *c,int size){return add_res(c,BCIR_DOM_RAM,size?size:4,1,0,BCIR_RK_SCALAR,"");}
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
/* The result-temp for a call THROUGH a funcptr (c.call.indirect / c.call.imember), by the funcptr's
 * captured RETURN type -- the C twin of the oracle's _call_result_ct ladder: a float keeps its width;
 * a wide (>4-byte) integer keeps its (width, sign); a SIGNED sub-int return promotes to `int` (so a
 * downstream `>>` / compare / `(long)` widen sign-extends); else uint32. A funcptr whose return type
 * wasn't captured (carriers all 0) -> uint32, today's behaviour. Aggregate returns are out of scope. */
static uint32_t fp_result_temp(CC *c, const bcir_ctype *fp){
  if(fp->fp_ret_float) return tempf(c, fp->fp_ret_size?fp->fp_ret_size:4);
  if(fp->fp_ret_size>4) return tempi(c, fp->fp_ret_size, fp->fp_ret_signd);
  if(fp->fp_ret_signd && fp->fp_ret_size && fp->fp_ret_size<=4) return tempi(c,4,1);
  return temp(c,4);
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
 * §6.4.4.1), mirroring ctype_model.int_literal_type. A character constant ('c') has type int. */
static void lit_int_type(const char *s,int n,int *size,int *signd){
  *size=4; *signd=1;                                    /* default: int */
  if(n>0 && s[0]=='\'') return;                         /* a character constant is int */
  char buf[64]; int j=0; for(int k=0;k<n&&j<63;k++) if(s[k]!='\'') buf[j++]=s[k]; buf[j]=0;
  int u=0,lr=0,e=j;
  while(e>0){ char ch=buf[e-1]; if(ch=='u'||ch=='U')u=1; else if(ch=='l'||ch=='L')lr++; else break; e--; }
  buf[e]=0;
  unsigned long long val; int decimal=1;
  if(e>1&&buf[0]=='0'&&(buf[1]=='x'||buf[1]=='X')){ val=strtoull(buf,NULL,16); decimal=0; }
  else if(e>1&&buf[0]=='0'&&(buf[1]=='b'||buf[1]=='B')){ val=strtoull(buf+2,NULL,2); decimal=0; }
  else if(e>1&&buf[0]=='0'){ val=strtoull(buf,NULL,8); decimal=0; }
  else { val=strtoull(buf[0]?buf:"0",NULL,10); decimal=1; }
  int cs[6],cz[6],nc=0;                                 /* candidate (size, signed) list, in order */
  if(u){ if(lr==0){cs[nc]=4;cz[nc++]=0;} cs[nc]=8;cz[nc++]=0; }
  else if(decimal){ if(lr==0){cs[nc]=4;cz[nc++]=1;} cs[nc]=8;cz[nc++]=1; }
  else { if(lr==0){cs[nc]=4;cz[nc++]=1; cs[nc]=4;cz[nc++]=0;} cs[nc]=8;cz[nc++]=1; cs[nc]=8;cz[nc++]=0; }
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
  idcpy(f->calls[f->n_calls++],name);
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
  int elem = str_elem_size(s,n);                              /* element width (wide/UTF prefix) */
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
    if(ty->ptr_to_struct) snprintf(pr->agg,BCIR_CIR_NAME,"%s %s",ty->is_union?"union":"struct",ty->tag);
    else if(ty->size==0 && !ty->is_float) pr->is_voidptr=1; }
  return t;
}
/* A fresh ordinary temp of `rid`'s value type -- the value a volatile object's read yields (lvalue
 * conversion drops the qualifier). */
static uint32_t temp_like(CC *c,uint32_t rid){
  const bcir_resource *r=res_of(c->fn,rid);
  bcir_resource snap; if(r) snap=*r; else memset(&snap,0,sizeof snap);   /* add_res may realloc res[] */
  uint32_t t=add_res(c,BCIR_DOM_RAM,(int)(snap.elem_bytes?snap.elem_bytes:4),1,0,
                     snap.kind==BCIR_RK_AGGREGATE?BCIR_RK_AGGREGATE:BCIR_RK_SCALAR,"");
  if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
    tr->is_float=snap.is_float; tr->is_complex=snap.is_complex; tr->is_signed=snap.is_signed;
    tr->is_bool=snap.is_bool; tr->is_plain_char=snap.is_plain_char; tr->bit_width=snap.bit_width;
    if(snap.kind==BCIR_RK_AGGREGATE) snprintf(tr->agg,sizeof tr->agg,"%s",snap.agg); }
  return t;
}
/* A read of the named object `v` as a value: its rid -- or, for a volatile object, a volatile read into an
 * ordinary temp (a c.copy carrying the volatile bit): the oracle's `_rvalue(Name)`. */
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

/* typeof( expression ) -- the operand is UNEVALUATED, so resolve its type by SPECULATIVELY lowering it,
 * reading the produced value's type off its resource, then rolling the whole emission back: the resource
 * and claim arrays, the rid/cid/compound-literal counters, and any call-graph / string-literal / local-env
 * side effects the operand triggered. The twin has no separate AST, so reusing the real lowering -- which
 * already types every value faithfully (fixed-width int + signedness, float, pointer pointee/depth, plain
 * char, _Bool) -- is exactly Clang-equivalent for the supported forms (binary, unary, cast, member, index,
 * deref). Calls / address-of are a deferred follow-on (a wide call return loses signedness on its temp).
 * On entry the cursor is at the operand's first token; on return it is just before the closing `)`. */
static int p_typeof_expr(CC *c, bcir_ctype *ty, int *sidx){
  size_t s_nres=c->fn->n_res, s_ncl=c->fn->n_claims;
  uint32_t s_rid=c->rid, s_cid=c->cid, s_clctr=c->cl_ctr;
  int s_ncalls=c->fn->n_calls, s_nenv=c->nenv;
  int s_nstr=c->fn->n_host_literals;
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
        snprintf(ty->tag,sizeof ty->tag,"%s",tg);
        int si=find_struct(c,tg,(int)strlen(tg)); if(si>=0){ *sidx=si; ty->is_union=(uint8_t)c->s[si].is_union; } }
    } else if(r->kind==BCIR_RK_AGGREGATE){              /* a struct/union by value */
      ty->kind=1; ty->size=(int)r->elem_bytes;
      const char *sp=strchr(r->agg,' '); const char *tg=sp?sp+1:r->agg;
      snprintf(ty->tag,sizeof ty->tag,"%s",tg);
      int si=find_struct(c,tg,(int)strlen(tg)); if(si>=0){ *sidx=si; ty->size=c->s[si].size;
        ty->is_union=(uint8_t)c->s[si].is_union; }
    } else {                                             /* a scalar (integer / float / _Bool / plain char) */
      ty->kind=0; ty->size=r->elem_bytes?(int)r->elem_bytes:4; ty->signd=r->is_signed;
      ty->is_float=r->is_float; ty->is_bool=r->is_bool; ty->is_plain_char=r->is_plain_char;
    }
  }
  c->fn->n_res=s_nres; c->fn->n_claims=s_ncl;            /* roll the speculative emission fully back */
  c->rid=s_rid; c->cid=s_cid; c->cl_ctr=s_clctr;
  c->fn->n_calls=s_ncalls; c->nenv=s_nenv;
  while(c->fn->n_host_literals>s_nstr){
    c->fn->n_host_literals--;
    bcir_host_deallocate(&c->allocator,
                         c->fn->host_literals[c->fn->n_host_literals].spelling);
    c->fn->host_literals[c->fn->n_host_literals].spelling=NULL;
  }
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
  const bcir_resource *s=res_of(c->fn,src);
  uint32_t r=add_res(c,BCIR_DOM_RAM, s?(int)s->elem_bytes:4, 1, 0, BCIR_RK_POINTER, "");
  if(c->fn->n_res && s){ bcir_resource *t=&c->fn->res[c->fn->n_res-1];
    t->is_signed=s->is_signed; t->is_float=s->is_float; snprintf(t->agg,sizeof t->agg,"%s",s->agg); }
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
    if(fld->ptee_sidx>=0) snprintf(t->agg,sizeof t->agg,"%s %s",
      c->s[fld->ptee_sidx].is_union?"union":"struct", c->s[fld->ptee_sidx].tag); }
  return r;
}

static uint32_t emit_member(CC *c, venv *base, const field *fld, int declared_bf) {
  /* the BITFIELD unit temp is sized to a power of 2 >= its byte span (a packed field that straddles into
   * bits >= 32 needs a 64-bit unit); the load reads only `access_bytes` (the spanned bytes). */
  int usz = fld->bit_w ? (fld->access_bytes<=4?4:8) : fld->size;
  uint32_t t=fld->is_ptr?tempptr_field(c,fld):fld->is_complex?tempc(c,fld->size):fld->is_float?tempf(c,fld->size)
            :fld->bit_w?tempi(c,usz,0)   /* a BITFIELD storage unit: a plain unsigned load (bf.get extracts below),
                                          * even a `_BitInt(N)` bitfield -- its unit is read raw, then masked. */
            :fld->bit_width>0?tempbi(c,fld->bit_width,fld->signd)   /* a PLAIN C23 `_BitInt(N)` member: load at the
                                                                    * storage width, typed `_BitInt(N)` (faithful) */
            :tempi(c,usz,fld->signd);   /* loaded value carries the field's type */
  if(fld->is_plain_char && c->fn->n_res) c->fn->res[c->fn->n_res-1].is_plain_char=1;   /* read as `char`, not int8_t */
  bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
  cl->n_rd=1;cl->rd[0]=base->rid;cl->n_wr=1;cl->wr[0]=t;cl->n_imm=2;cl->imm[0]=fld->byte_off;cl->imm[1]=fld->bit_w?fld->access_bytes:fld->size;
  cl->bounds=BCIR_BND_ASSUMED;
  mark_access(c,cl,fld->is_volatile||base->type.is_volatile);
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
  uint32_t t=fld->is_complex?tempc(c,fld->size):fld->is_float?tempf(c,fld->size):tempi(c,fld->size,fld->signd);
  if(fld->is_plain_char && c->fn->n_res) c->fn->res[c->fn->n_res-1].is_plain_char=1;   /* `char[]` element: `char` */
  bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
  cl->n_rd=2;cl->rd[0]=base->rid;cl->rd[1]=idx;cl->n_wr=1;cl->wr[0]=t;
  cl->n_imm=2;cl->imm[0]=fld->byte_off;cl->imm[1]=fld->size;cl->bounds=BCIR_BND_ASSUMED;
  mark_access(c,cl,fld->is_volatile||base->type.is_volatile);
  return t;
}
/* After `arr[i]` on an ARRAY-OF-STRUCTS member (`arr->elem_sidx>=0`) with a trailing `.`/`->`, parse the
 * element field name and look it up in the element struct. Only a PLAIN SCALAR element field is handled
 * (a bitfield/array/nested/pointer element field is a consistent follow-on -- both rails route to fallback).
 * Returns 1 with *sub filled, else 0 (and may have raised via fail()). The leading `.`/`->` must be present. */
static int elem_field(CC *c, const field *arr, field *sub) {
  if(arr->elem_sidx<0 || !(is(c,".")||is(c,"->"))) return 0;
  c->i++; tok fn=adv(c); sdef *ES=&c->s[arr->elem_sidx]; int fi=-1;
  for(int i=0;i<ES->nf;i++) if((int)strlen(ES->f[i].name)==fn.n&&!strncmp(ES->f[i].name,fn.s,fn.n)) fi=i;
  if(fi<0){ fail(c,"unknown field"); return 0; }
  field s=ES->f[fi];
  if(s.bit_w||s.arr_count||s.sidx>=0||s.is_ptr){ fail(c,"array-of-structs non-scalar element field"); return 0; }
  *sub=s; return 1;
}
/* `arr[i].field` (array-of-structs): load `sub->size` bytes at member_off(arr)+offsetof(sub), STRIDING by the
 * element size `arr->size` (imm[2]) -- decoupled from the field copy size (imm[1]). */
static uint32_t emit_member_index_field(CC *c, venv *base, const field *arr, uint32_t idx, const field *sub) {
  uint32_t t=sub->is_float?tempf(c,sub->size):tempi(c,sub->size,sub->signd);
  if(sub->is_plain_char && c->fn->n_res) c->fn->res[c->fn->n_res-1].is_plain_char=1;
  bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
  cl->n_rd=2;cl->rd[0]=base->rid;cl->rd[1]=idx;cl->n_wr=1;cl->wr[0]=t;
  cl->n_imm=3;cl->imm[0]=arr->byte_off+sub->byte_off;cl->imm[1]=sub->size;cl->imm[2]=arr->size;
  cl->bounds=BCIR_BND_ASSUMED;
  mark_access(c,cl,sub->is_volatile||arr->is_volatile||base->type.is_volatile);
  return t;
}
/* After `a[i]` on a DIRECT local/global ARRAY-OF-STRUCTS variable (`v->sidx>=0`, the element struct) with a
 * trailing `.`/`->`, parse the element field name and look it up in the element struct. Only a PLAIN SCALAR
 * element field is handled (a bitfield/array/nested/pointer field is a both-rails follow-on). Returns 1 with
 * *sub filled, else 0 (may raise via fail()). The leading `.`/`->` must be present; the cursor is past it. */
static int aos_elem_field(CC *c, venv *base, field *sub) {
  if(base->sidx<0 || !(is(c,".")||is(c,"->"))) return 0;
  c->i++; tok fn=adv(c); sdef *ES=&c->s[base->sidx]; int fi=-1;
  for(int i=0;i<ES->nf;i++) if((int)strlen(ES->f[i].name)==fn.n&&!strncmp(ES->f[i].name,fn.s,fn.n)) fi=i;
  if(fi<0){ fail(c,"unknown field"); return 0; }
  field s=ES->f[fi];
  if(s.bit_w||s.arr_count||s.sidx>=0||s.is_ptr){ fail(c,"array-of-structs non-scalar element field"); return 0; }
  *sub=s; return 1;
}
/* `a[i].field` on a DIRECT array-of-structs variable: load `sub->size` bytes at offsetof(sub), STRIDING by
 * the element (struct) size `v->type.size` (imm[2]) -- the base is the array itself (member offset 0). */
static uint32_t emit_index_field(CC *c, venv *base, uint32_t idx, const field *sub) {
  uint32_t t=sub->is_float?tempf(c,sub->size):tempi(c,sub->size,sub->signd);
  if(sub->is_plain_char && c->fn->n_res) c->fn->res[c->fn->n_res-1].is_plain_char=1;
  bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
  cl->n_rd=2;cl->rd[0]=base->rid;cl->rd[1]=idx;cl->n_wr=1;cl->wr[0]=t;
  cl->n_imm=3;cl->imm[0]=sub->byte_off;cl->imm[1]=sub->size;cl->imm[2]=base->type.size;
  cl->bounds=BCIR_BND_ASSUMED;
  mark_access(c,cl,sub->is_volatile||base->type.is_volatile);
  return t;
}
/* --- C's assignment conversion at a store the emit spells as a byte copy (CF-MEMCONV) ------------------- */
static void cast_name(const bcir_ctype *ty,int signed_int,char *o,size_t n);   /* fwd: a cast's spelling */
/* `(ty)v`: one `c.cast` claim into a temp of the target type -- the explicit cast, and C's assignment
 * conversion at a store (`store_conv`). The oracle's `_cast_value`. */
static uint32_t emit_cast(CC *c, uint32_t v, const bcir_ctype *tyin, int si){
  bcir_ctype ty=*tyin;
  const bcir_resource *vr=res_of(c->fn,v);
  /* The result temp carries the target's signedness, so a signed (sub-int) target keeps its sign
   * even when the cast value is used directly -- `(signed char)(-5)` stays -5, and `(int)u` reads
   * back signed (an arithmetic `>>`). A float -> signed-int conversion additionally needs a SIGNED
   * cast operator (float -> unsigned is UB / target-divergent). */
  int f2s = vr && vr->is_float && !ty.is_float && ty.kind!=2 && ty.signd && ty.size>0
            && ty.bit_width==0;                                  /* a `_BitInt` keeps its exact spelling */
  uint32_t r = ty.kind==2 ? temp_ptr(c, &ty, si)                /* a pointer cast: a `T *` of the target (first:
                                                                  * a `float *` target is a pointer, not a float) */
             : ty.is_complex ? tempc(c, ty.size)                /* a _Complex cast -> a complex temp */
             : ty.is_float ? tempf(c, ty.size)                  /* a float cast -> a float temp */
             : ty.bit_width>0 ? tempbi(c, ty.bit_width, ty.signd?1:0)   /* a C23 `_BitInt(N)` cast */
                          : tempi(c, ty.size?ty.size:4, ty.signd?1:0);   /* an integer cast */
  if(ty.is_bool && !ty.is_float && ty.kind!=2 && c->fn->n_res)
    c->fn->res[c->fn->n_res-1].is_bool=1;    /* a bool cast -> a _Bool temp (normalizes to 0/1) */
  char op[BCIR_CIR_NAME]; cast_name(&ty,f2s,op,sizeof op);
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
  t.kind=(uint8_t)(f->is_ptr?2:(f->sidx>=0||f->elem_sidx>=0)?1:0);
  t.size=f->size; t.signd=f->signd; t.is_float=(uint8_t)(f->is_float?1:0);
  t.is_complex=(uint8_t)(f->is_complex?1:0); t.is_bool=(uint8_t)(f->is_bool?1:0);
  t.is_plain_char=(uint8_t)(f->is_plain_char?1:0); t.bit_width=f->bit_width;
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
  return val;
}
/* Parse the `[i]` (or `[i][j][k]`) indices of a member-array access and flatten them row-major into a
 * single linear index (Horner: lin = lin*adims[d] + idx[d]) -- matching the oracle, so 1-D `s.a[i]` and
 * N-D `s.m[i][j]` both reduce to one element-scaled index into the member at its byte offset. */
static uint32_t member_arr_index(CC *c, const field *fld) {
  uint32_t lin=0; int d=0;
  while(is(c,"[")){ c->i++; uint32_t ix=p_expr(c); eat(c,"]");
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
    idcpy(c->mut[c->mut_n].name, id);
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
    if (t->k == T_PUN && t->n == 1 && t->s[0] == '&' && c->t[i + 1].k == T_ID) {  /* a UNARY `&x` (address-of) */
      const tok *pv = (i > start) ? &c->t[i - 1] : NULL;
      if (!(pv && is_value_ender(pv))) {                   /* unary when the previous token is not a value-ender */
        mutent *m = mut_find(c, &c->t[i + 1]);
        if (m) m->addr = 1;
      }
    }
  }
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
    int isty = scalar_size(pk(c)->s, pk(c)->n) >= 0 || is(c, "struct") || is(c, "union") || is(c, "enum")
               || is(c, "_Complex") || is(c, "complex") || is(c, "_BitInt") || is(c, "const") || is(c, "volatile")
               || is(c, "typeof") || is(c, "__typeof__") || is(c, "typeof_unqual")
               || find_typedef(c, pk(c)->s, pk(c)->n) >= 0;
    int sz = -1;
    if (isty) { bcir_ctype ty; int si; if (!p_type(c, &ty, &si))
      sz = ty.kind == 2 ? cc_abi(c)->pointer_size : (ty.kind == 1 ? c->s[si].size : ty.size); }
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
 * is the token range [init_start, init_end). */
static void bind_extent(CC *c, uint32_t p_rid, const bcir_resource *pr, const tok *p_name,
                        int init_start, int init_end) {
  if (!pr || pr->kind != BCIR_RK_POINTER) return;
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
static uint32_t elem_temp(CC *c, venv *base) {
  uint32_t t;
  if(ptr_array(c,base)){                          /* an ARRAY of pointers `T *a[N]`
    * (a SCALAR array with pointer-wide elements, NOT a pointer variable `T *p`): `a[i]` loads a pointer, typed
    * as the element (its pointee's width, sign and volatility), so indexing it lands on the right objects */
    t=add_res(c,BCIR_DOM_RAM,base->type.size?base->type.size:4,1,0,BCIR_RK_POINTER,"");
    if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
      tr->ptr_depth=base->type.ptr_depth?base->type.ptr_depth:1;
      tr->is_signed=(uint8_t)(base->type.signd?1:0); tr->is_float=(uint8_t)(base->type.is_float?1:0);
      tr->is_plain_char=(uint8_t)(base->type.is_plain_char?1:0);
      if(base->type.ptr_to_struct) snprintf(tr->agg,sizeof tr->agg,"%s %s",base->type.is_union?"union":"struct",base->type.tag);
      else if(base->type.size==0 && !base->type.is_float) tr->is_voidptr=1; }   /* a `void *` element */
    last_ptr_to_volatile(c,base->type.is_volatile,base->type.ptr_to_struct && sdef_vol(c,base->sidx));
  } else if(base->type.kind==2 && (base->type.ptr_depth?base->type.ptr_depth:1)>1){   /* `pp[i]` through a
    * `T **`: the element is itself a pointer, one level down (the oracle's `base_ct.of`), never a `T` value */
    bcir_ctype et=base->type; et.ptr_depth=(uint8_t)((base->type.ptr_depth?base->type.ptr_depth:1)-1);
    t=temp_ptr(c,&et,base->sidx);
  } else { int es=base->type.size?base->type.size:4;
    t=base->type.is_float ? tempf(c,es) : tempi(c,es,base->type.signd); }  /* float -> a float temp; else keep the sign */
  return t;
}
static uint32_t emit_index(CC *c, venv *base, uint32_t idx) {     /* base[idx] -- GEP load */
  uint32_t t=elem_temp(c,base);
  bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
  cl->n_rd=2;cl->rd[0]=base->rid;cl->rd[1]=idx;cl->n_wr=1;cl->wr[0]=t;cl->bounds=access_bnd(c,base->rid);
  mark_access(c,cl,index_elem_vol(c,base));        /* the element of a pointer to volatile / a volatile array */
  return t;
}
/* Parse up to `maxd` subscripts `[i][j][k]` on an array variable and Horner-flatten via its declared dims
 * (`v->type.adims`): `m[i][j]` on a `T m[A][B]` -> `i*B + j`. The cursor must be at the first `[`. */
static uint32_t array_index_n(CC *c, venv *v, int maxd) {
  /* SNAPSHOT the env entry: the index p_expr below can declare locals (a `({...})` stmt-expr) and realloc
   * c->env[] -- the incoming `v`, a pointer into that array, would dangle after the move. Reading from the
   * by-value copy is byte-identical (only v->rid / v->type are read). */
  venv vsnap=*v; v=&vsnap;
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
static uint32_t index_chain(CC *c, venv *v){
  for(;;){
    uint32_t lin=array_index_n(c,v,subscript_dims(c,v));
    if(c->failed || !is(c,"[")) return lin;
    if(!step_to_elem_ptr(c,v,lin)){ fail(c,"a subscript of an element that is not a pointer"); return lin; }
  }
}
/* Dereference a pointer RVALUE (rid). Depth-aware: `*pp` where pp is `T**` (depth 2) loads a pointer
 * (pointer_size bytes) into a `T*` temp (depth 1); `*p` where p is `T*` loads the base scalar. The
 * general form powers `**pp` (deref the result of `*pp`) and `*(<expr>)`. */
static uint32_t emit_deref_rid(CC *c, uint32_t rid) {
  const bcir_resource *r=res_of(c->fn,rid);
  if(!r || r->kind!=BCIR_RK_POINTER){ fail(c,"dereference of a non-pointer"); return rid; }
  int depth=r->ptr_depth?r->ptr_depth:1, base=r->elem_bytes?(int)r->elem_bytes:4;
  /* SNAPSHOT every field of `r` we still need: the add_res/tempi/tempf calls below allocate a
   * NEW resource, which may realloc (and thus MOVE+free) c->fn->res -- so `r`, a pointer INTO that
   * array, dangles after the first allocation. Reading through it afterward is a use-after-free. */
  uint8_t r_signd=r->is_signed, r_float=r->is_float, r_plain_char=r->is_plain_char, r_vol=r->is_volatile;
  bcir_domain r_dom=r->domain;
  char r_agg[sizeof r->agg]; snprintf(r_agg,sizeof r_agg,"%s",r->agg);
  uint32_t t;
  if(depth>1){                                     /* the pointee is itself a pointer (read pointer_size) */
    t=add_res(c,r_dom,base,1,r_vol,BCIR_RK_POINTER,"");   /* still pointing at the same (device) storage */
    if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
      tr->is_signed=r_signd; tr->is_float=r_float; tr->ptr_depth=(uint8_t)(depth-1);
      snprintf(tr->agg,sizeof tr->agg,"%s",r_agg); }
  } else { t = r_float ? tempf(c,base) : tempi(c,base,r_signd);
    if(depth==1 && r_plain_char && c->fn->n_res)   /* a `char *` deref loads a plain `char` value */
      c->fn->res[c->fn->n_res-1].is_plain_char=1; }
  int rd_sz = depth>1 ? cc_abi(c)->pointer_size : base;
  bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
  cl->n_rd=1;cl->rd[0]=rid;cl->n_wr=1;cl->wr[0]=t;cl->bounds=BCIR_BND_ASSUMED;cl->n_imm=2;cl->imm[0]=0;cl->imm[1]=rd_sz;
  mark_access(c,cl,depth==1 && r_vol);             /* the pointee of a pointer to volatile, at its own width */
  return t;
}
static uint32_t emit_deref(CC *c, venv *pv) {     /* *p -- a one-read dereference load (named pointer or array) */
  const bcir_resource *r=res_of(c->fn,pv->rid);
  if(r && r->kind!=BCIR_RK_POINTER && r->is_array && pv->type.kind!=1){   /* `*a` on an ARRAY: its first element,
    * one load at offset 0 (the oracle's `mem` lvalue of the element type); volatile when the element is */
    int es = pv->type.kind==2 ? cc_abi(c)->pointer_size : (pv->type.size?pv->type.size:4);
    uint32_t t=elem_temp(c,pv);
    bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
    cl->n_rd=1;cl->rd[0]=pv->rid;cl->n_wr=1;cl->wr[0]=t;cl->bounds=BCIR_BND_ASSUMED;cl->n_imm=2;cl->imm[0]=0;cl->imm[1]=es;
    mark_access(c,cl,index_elem_vol(c,pv));
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
    return t;
  }
  return emit_deref_rid(c,pv->rid);                /* depth-aware: the resource carries width, sign, float,
                                                    * ptr_depth and the pointee's volatility */
}
/* Nested member access (`o.in.v` / `dev->ctrl.flags` -- a sub-register-block): given a member `f`
 * already resolved (byte_off relative to the access base) and the cursor at a possible further
 * `.`/`->`, descend through nested value-struct members, accumulating byte offsets. Returns the final
 * field with byte_off set to the total offset from the base, so the load/store paths (which read
 * f.byte_off / f.size / f.bit_*) flatten the chain to a single offset access -- matching the oracle. */
static field member_descend(CC *c, field f) {
  while((is(c,".")||is(c,"->")) && f.sidx>=0){
    int base_off=f.byte_off; sdef *S=&c->s[f.sidx]; c->i++;
    tok fn=adv(c); int fi=-1;
    for(int i=0;i<S->nf;i++) if((int)strlen(S->f[i].name)==fn.n&&!strncmp(S->f[i].name,fn.s,fn.n)) fi=i;
    if(fi<0){ fail(c,"unknown field"); return f; }
    f=S->f[fi]; f.byte_off+=base_off;
  }
  return f;
}
/* Continue a postfix read chain through a loaded POINTER value (#fieldderef): `s->mid->k`, the two-hop
 * `s->mid->leaf->x`, and the subscript `s->p[i]`. `ptr` is the loaded pointer rid (kind POINTER), `psidx`
 * its pointee struct index (-1 for a pointer-to-scalar), `pfld` the field it came from (its pointee
 * width/sign/float types a scalar deref). Mirrors the oracle: a pointer field used as a base is loaded,
 * then the loaded pointer is the new base for the next `->`/`[` -- the same move as `*q` on a `T**`. */
static uint32_t postfix_ptr_chain(CC *c, uint32_t ptr, int psidx, field pfld) {
  for(;;){
    if(is(c,"[")){                                   /* `...->p[i]`: index the loaded pointer (base[idx]) */
      c->i++; uint32_t ix=p_expr(c); eat(c,"]");
      venv b; memset(&b,0,sizeof b); b.rid=ptr; b.sidx=-1;
      b.type.size=pfld.ptee_size?pfld.ptee_size:4; b.type.signd=pfld.signd; b.type.is_float=(uint8_t)pfld.ptee_float;
      b.type.is_volatile=(uint8_t)(pfld.ptee_volatile?1:0);   /* `d->regs[i]` through a `volatile T *regs` */
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
        uint32_t t = ff.fp_ret_float ? tempf(c, ff.fp_ret_size?ff.fp_ret_size:4)
                   : ff.fp_ret_size>4 ? tempi(c, ff.fp_ret_size, ff.fp_ret_signd)
                   : (ff.fp_ret_signd && ff.fp_ret_size && ff.fp_ret_size<=4) ? tempi(c,4,1)
                   : temp(c,4);                           /* a signed member return reads back signed (mirrors the oracle) */
        char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.call.imember:%s",S->f[fi].name);
        bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
        if(cl){cl->n_rd=(uint8_t)(na+1);cl->rd[0]=ptr;for(int k=0;k<na;k++)cl->rd[k+1]=args[k];
          cl->n_wr=1;cl->wr[0]=t;cl->n_imm=1;cl->imm[0]=1;cl->truncated=(uint8_t)dropped;}   /* imm0=1: `ptr->fn(args)` */
        return t;
      }
      if(mf.is_ptr && (is(c,"->")||is(c,".")||is(c,"["))){    /* another pointer hop: load it, recurse */
        ptr=emit_member(c,&b,&mf,0); psidx=mf.ptee_sidx; pfld=mf; continue;
      }
      if(mf.arr_count && is(c,"[")){ uint32_t ix=member_arr_index(c,&mf);
        field sub; if(elem_field(c,&mf,&sub)) return emit_member_index_field(c,&b,&mf,ix,&sub);
        if(c->failed) return 0;
        return emit_member_index(c,&b,&mf,ix); }
      return emit_member(c,&b,&mf,c->stmt_expr_declared_bf); /* terminal member load through the loaded pointer */
    }
    return ptr;                                       /* no further postfix: the pointer value itself */
  }
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
  /* libc-implicit (malloc/free/realloc/calloc/aligned_alloc + the printf/scanf family) -> no flag. */
  if(is_stdlib_alloc(s,n) || is_extern_variadic(s,n)) return "";
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

static uint32_t p_call(CC *c, const tok *name) {
  if(tok_is(name,"va_arg")){          /* va_arg(ap, TYPE) -- the 2nd arg is a type-name, parsed specially */
    c->i++; /* '(' */
    uint32_t ap=p_expr(c); eat(c,",");
    const char *t0=pk(c)->s; bcir_ctype ty; int si; if(p_type(c,&ty,&si)) return 0;
    const char *t1=c->t[c->i-1].s + c->t[c->i-1].n; eat(c,")");
    uint32_t t;                        /* type the result by T so downstream arithmetic/loads are correct */
    if(ty.is_float) t=tempf(c,ty.size);
    else if(ty.kind==2){ t=temp(c,cc_abi(c)->pointer_size); bcir_resource *pr=&c->fn->res[c->fn->n_res-1];
      pr->is_signed=(uint8_t)(ty.signd?1:0); pr->is_float=(uint8_t)(ty.is_float?1:0);
      pr->ptr_depth=ty.ptr_depth?ty.ptr_depth:1; pr->is_plain_char=(uint8_t)(ty.is_plain_char?1:0);
      if(ty.ptr_to_struct) snprintf(pr->agg,BCIR_CIR_NAME,"%s %s",ty.is_union?"union":"struct",ty.tag); }
    else t=tempi(c,ty.size?ty.size:4, ty.signd?1:0);
    char op[BCIR_CIR_NAME]; int tn=(int)(t1-t0);   /* carry T's exact source spelling for a faithful emit */
    if(tn>(int)sizeof op-16) tn=(int)sizeof op-16; if(tn<0) tn=0;
    snprintf(op,sizeof op,"c.call.vaarg:%.*s",tn,t0);
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
    char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.call.vabuiltin:%.*s",name->n,name->s);
    bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
    if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=0;}
    return temp(c,4);                  /* a void result -- never read */
  }
  int bz = builtin_result(name->s,name->n);
  if(bz){                              /* a GCC/Clang integer builtin -> verbatim, typed, opaque to R18 */
    uint32_t t = bz<0 ? tempi(c,-bz,1) : tempi(c,bz,0);
    char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.call.builtin:%.*s",name->n-10,name->s+10);  /* drop `__builtin_` (op cap) */
    bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
    if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=1;cl->wr[0]=t;}
    return t;                          /* not added to fn->calls (opaque to R18) */
  }
  int cx_is; int cz = cplx_libm(c,name->s,name->n,&cx_is);   /* <complex.h> creal/cimag/conj/... */
  if(cz){                                  /* a typed external complex-library edge (counts as one call) */
    uint32_t t = cx_is ? tempc(c,cz*2) : tempf(c,cz);        /* conj -> complex (2x elem); creal -> real */
    char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.call.libm:%.*s",name->n,name->s);
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
    char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.call.libm:%.*s",name->n,name->s);
    bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
    if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=1;cl->wr[0]=t;}
    return t;                              /* not added to fn->calls (opaque to R18) */
  }
  int sal=is_stdlib_alloc(name->s,name->n);  /* <stdlib.h> malloc/calloc/realloc/free -- external libc edge */
  if(sal){
    if(sal==2){                              /* free(p) -> a void external call statement (opaque to R18) */
      char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.call.libm.void:%.*s",name->n,name->s);
      bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
      /* R21 lifetime FREE event (§5.12): the freed pointer it reads dies after this claim, so a later
       * dereference is a use-after-free (or a second free a double-free). Digest-excluded + advisory. */
      if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=0;cl->lifetime=2;}
      return temp(c,4);                      /* a void result -- never read */
    }
    uint32_t t=add_res(c,BCIR_DOM_RAM,cc_abi(c)->pointer_size,1,0,BCIR_RK_POINTER,"");  /* a `void *` result */
    if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1]; tr->ptr_depth=1; }  /* no agg -> `void *` */
    char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.call.libm:%.*s",name->n,name->s);
    bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
    /* R21 lifetime ALLOC event (§5.12): the allocator result (re-)validates the resource it writes, so a
     * pointer reassigned from it is live again after an earlier free. Digest-excluded + advisory. */
    if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=1;cl->wr[0]=t;cl->lifetime=1;}
    return t;                                /* not added to fn->calls (opaque to R18) */
  }
  const bcir_ctype *rt=callee_ret(c,name);   /* type the result by the callee's return (earlier defs) */
  if(rt && rt->kind==0 && rt->size==0){      /* a void callee -> a bare call statement, no result temp */
    char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.call.void:%.*s",name->n,name->s);
    bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
    if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=0;}
    add_call(c,name);
    return temp(c,4);                        /* an unused placeholder (a void result is never read) */
  }
  if(!rt && is_extern_variadic(name->s,name->n)){   /* a printf/scanf-family external variadic -> opaque */
    uint32_t t=tempi(c,4,1);                          /* returns int; emitted verbatim against <stdio.h> */
    char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.call.extern:%.*s",name->n,name->s);
    bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
    if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=1;cl->wr[0]=t;}
    return t;                                         /* not added to fn->calls (opaque to R18) */
  }
  if(!rt){                                            /* Phase 3 LINKING: a PROTOTYPED cross-TU callee */
    const bcir_ctype *ptt=NULL;
    for(int k=0;k<c->n_protos;k++)
      if((int)strlen(c->protos[k].name)==name->n && !strncmp(c->protos[k].name,name->s,(size_t)name->n)){
        ptt=&c->protos[k].ret; break; }
    if(ptt){
      /* A typed external edge the host LINKER resolves from a sibling object: like a libm edge it is
       * opaque to the in-unit R18 call graph (NOT added to fn->calls) and emits verbatim (external
       * linkage) with the prototype's extern declaration in the prelude; unlike libm it derives no -l
       * flag. Result typing mirrors the defined-callee ladder below (the oracle's _call_result_ct). */
      if(ptt->kind==1){ fail(c,"aggregate return through a prototype is not supported"); return temp(c,4); }
      char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.call.tu:%.*s",name->n,name->s);
      if(ptt->kind==0 && ptt->size==0){               /* a void cross-TU callee -> a bare statement */
        bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
        if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=0;}
        return temp(c,4);                             /* a void result -- never read */
      }
      uint32_t t = ptt->is_complex ? tempc(c,ptt->size)
                 : ptt->is_float   ? tempf(c,ptt->size)
                 : (ptt->kind==0 && ptt->bit_width>0) ? tempbi(c,ptt->bit_width,ptt->signd)
                 : (ptt->kind==0 && ptt->size==8) ? tempi(c,8,ptt->signd)
                 : (ptt->kind==0 && ptt->size<=4 && ptt->signd) ? tempi(c,4,1)
                 : temp(c,4);
      bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
      if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=1;cl->wr[0]=t;}
      return t;
    }
  }
  uint32_t t;
  if(rt && rt->kind==1){                              /* a struct/union RETURN: a by-value aggregate temp
                                                       * (`struct P t = mk(x);`) so it copies/passes/member-accesses
                                                       * -- a uint32 temp emitted invalid C `uint32_t t = mk(x)`. */
    t=add_res(c,BCIR_DOM_RAM,rt->size,1,0,BCIR_RK_AGGREGATE,"");
    if(c->fn->n_res) snprintf(c->fn->res[c->fn->n_res-1].agg,BCIR_CIR_NAME,"%s %s",
                              rt->is_union?"union":"struct", rt->tag);
  } else
    t = (rt && rt->is_complex)          ? tempc(c,rt->size)   /* _Complex user return (a float pair) */
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
  char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.call:%.*s",name->n,name->s);
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
  uint32_t t=fp_result_temp(c,&fv->type);   /* type by the funcptr's captured return -> a signed return reads back signed */
  bcir_claim *cl=new_claim(c,"c.call.indirect",BCIR_OP_GEM_DISPATCH);
  if(cl){cl->n_rd=(uint8_t)(na+1);cl->rd[0]=fv->rid;for(int k=0;k<na;k++)cl->rd[k+1]=args[k];
    cl->n_wr=1;cl->wr[0]=t;cl->truncated=(uint8_t)dropped;}
  return t;
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
  /* strip the oracle parser's TRANSPARENT prefixes: balanced redundant parens AND a leading unary `+` (the
   * parser drops `+x` to `x`, but keeps `-`/`~`/`!`/a cast as a node -> those fold to the full fence). Only
   * a `(` needs a matching `)`; a `+` is closer-less. */
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
    int ec=find_enum(c,core->s,core->n);
    if(ec>=0) return order_kind(c->ec[ec].val);
    if(lookup(c,core) || find_global(c,core->s,core->n)>=0 || callee_ret(c,core)) return "c.fence";
    long long mo;
    if(memorder_value(core,&mo)) return order_kind(mo);
  }
  return "c.fence";                                        /* non-constant / shadowed / unknown -> full */
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
  uint32_t t=temp(c,4);
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
    int arrow=is(c,"->"); c->i++; tok fn=adv(c); sdef *S=&c->s[v->sidx]; int fi=-1;
    for(int i=0;i<S->nf;i++) if((int)strlen(S->f[i].name)==fn.n&&!strncmp(S->f[i].name,fn.s,fn.n)) fi=i;
    if(fi<0){fail(c,"unknown field");return 0;}
    if(is(c,"(")){     /* o->fnptr(args): fused indirect call via a funcptr struct member */
      c->i++; uint32_t args[BCIR_CLAIM_MAX_RD]; int na=0, dropped=0;
      if(!is(c,")")) for(;;){ uint32_t a=p_expr(c); if(na<BCIR_CLAIM_MAX_RD-1)args[na++]=a; else dropped=1;
        if(is(c,",")){c->i++;continue;} break; }
      eat(c,")");
      field ff=S->f[fi];                          /* the funcptr field carries its captured return type */
      uint32_t t = ff.fp_ret_float ? tempf(c, ff.fp_ret_size?ff.fp_ret_size:4)
                 : ff.fp_ret_size>4 ? tempi(c, ff.fp_ret_size, ff.fp_ret_signd)
                 : (ff.fp_ret_signd && ff.fp_ret_size && ff.fp_ret_size<=4) ? tempi(c,4,1)
                 : temp(c,4);                      /* a signed member return reads back signed (mirrors the oracle) */
      char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.call.imember:%s",S->f[fi].name);
      bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
      if(cl){cl->n_rd=(uint8_t)(na+1);cl->rd[0]=v->rid;for(int k=0;k<na;k++)cl->rd[k+1]=args[k];
        cl->n_wr=1;cl->wr[0]=t;cl->n_imm=1;cl->imm[0]=arrow;cl->truncated=(uint8_t)dropped;}
      return t;
    }
    field mf=member_descend(c,S->f[fi]);        /* nested `o.in.v` -> one flattened-offset load */
    if(mf.is_ptr && (is(c,"->")||is(c,".")||is(c,"["))){   /* deref-through a loaded pointer field (#fieldderef) */
      uint32_t ptr=emit_member(c,v,&mf,0);      /* load the pointer field, then chain through the loaded ptr */
      return postfix_ptr_chain(c,ptr,mf.ptee_sidx,mf); }
    if(mf.arr_count && is(c,"[")){ uint32_t ix=member_arr_index(c,&mf);   /* s.arr[i] / s.m[i][j] load */
      field sub; if(elem_field(c,&mf,&sub)) return emit_member_index_field(c,v,&mf,ix,&sub);   /* arr[i].field */
      if(c->failed) return 0;
      return emit_member_index(c,v,&mf,ix); }
    return emit_member(c,v,&mf,c->stmt_expr_declared_bf);
  }
  if(is(c,"[")){                                /* L3: base[i] / m[i][j] (row-major flatten) / q[j][i] (a chain) */
    uint32_t lin=index_chain(c,v);
    if(c->failed) return 0;
    if(v->sidx>=0 && (is(c,".")||is(c,"->"))){    /* a[i].field on a DIRECT array-of-structs (strided load) */
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
static uint32_t p_primary(CC *c) {
  if(is(c,"_Generic")) return p_generic(c);
  if(isk(c,T_INT)){tok t=adv(c);
    int lsz,lsg; lit_int_type(t.s,t.n,&lsz,&lsg);            /* the constant's type (§6.4.4.1) */
    uint32_t r=tempi(c,lsz,lsg);
    bcir_claim *cl=new_claim(c,"c.const",BCIR_OP_LOAD);if(!cl)return r;
    cl->n_wr=1;cl->wr[0]=r;cl->n_imm=1;cl->imm[0]=t.v;return r;}
  if(isk(c,T_FLT)){tok t=adv(c);                       /* a floating constant -> a typed c.fconst */
    int isf = t.n>0 && (t.s[t.n-1]=='f'||t.s[t.n-1]=='F');   /* f/F -> float(4) */
    int isl = t.n>0 && (t.s[t.n-1]=='l'||t.s[t.n-1]=='L');   /* l/L -> long double, else double(8) */
    uint32_t r=tempf(c, isf?4:isl?cc_abi(c)->long_double_size:8);
    char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.fconst:%.*s",t.n,t.s);
    bcir_claim *cl=new_claim(c,op,BCIR_OP_LOAD); if(cl){cl->n_wr=1;cl->wr[0]=r;} return r;}
  if(isk(c,T_STR)){     /* a string literal -> an anonymous read-only char[] global; value is a ptr */
    tok st=adv(c); uint32_t rid;
    if(isk(c,T_STR)){ int cn; char *cb=gather_strings(c,st,&cn);   /* adjacent literals concatenate */
      rid=intern_string(c, cb?cb:st.s, cb?cn:st.n); }
    else rid=intern_string(c,st.s,st.n);   /* full spelling kept in result-owned metadata; dedup; cap lifted */
    if(is(c,"[")){ c->i++; uint32_t ix=p_expr(c); eat(c,"]");
      venv sv; memset(&sv,0,sizeof sv); sv.rid=rid; sv.type.size=1; sv.sidx=-1;
      return emit_index(c,&sv,ix); }
    return rid;
  }
  if(is(c,"_Alignof")||is(c,"alignof")){   /* _Alignof(type) -> the type's alignment, a folded const */
    c->i++; eat(c,"("); bcir_ctype ty;int si; long long al=4;
    if(!p_type(c,&ty,&si)) al = ty.kind==2?cc_abi(c)->pointer_size:(ty.kind==1?c->s[si].align:(ty.size?ty.size:1));
    eat(c,")");
    uint32_t r=temp(c,4); bcir_claim *cl=new_claim(c,"c.const",BCIR_OP_LOAD);
    if(cl){cl->n_wr=1;cl->wr[0]=r;cl->n_imm=1;cl->imm[0]=al;} return r;
  }
  if(is(c,"sizeof")){                  /* sizeof(type) / sizeof expr -> a folded constant (no eval) */
    c->i++; long long size=4; int got=0;
    if(is(c,"(")){ int save=c->i; c->i++;
      int is_type = scalar_size(pk(c)->s,pk(c)->n)>=0 || is(c,"struct")||is(c,"union")||is(c,"enum")||is(c,"_Complex")||is(c,"complex")||is(c,"_BitInt")
                    || is(c,"const")||is(c,"volatile")
                    || is(c,"typeof")||is(c,"__typeof__")||is(c,"typeof_unqual")
                    || find_typedef(c,pk(c)->s,pk(c)->n)>=0;
      if(is_type){ bcir_ctype ty;int si;
        if(!p_type(c,&ty,&si)){ size = ty.kind==2?cc_abi(c)->pointer_size:(ty.kind==1?c->s[si].size:ty.size); got=1; }
        eat(c,")"); }
      else c->i=save;                  /* not a type -> sizeof ( expr ) */
    }
    if(!got){                          /* sizeof <operand>: a variable's static type size */
      int paren=0; if(is(c,"(")){c->i++;paren=1;}
      if(isk(c,T_STR)){ tok st=adv(c);                          /* sizeof a (possibly concatenated) literal */
        if(isk(c,T_STR)){ int cn; char *cb=gather_strings(c,st,&cn); const char *sp=cb?cb:st.s; int sn=cb?cn:st.n;
          size=(long long)(str_bytes(sp,sn)+1)*str_elem_size(sp,sn); }
        else size=(long long)(str_bytes(st.s,st.n)+1)*str_elem_size(st.s,st.n); }   /* units incl. NUL × width */
      else if(isk(c,T_ID)){ tok vid=*pk(c); venv *v=lookup(c,&vid);
        int indexed = tok_is(tat(c,c->i+1),"[");   /* `sizeof a[0]` -- an element, NOT a bare name (stays static) */
        if(v && !indexed){
          const bcir_resource *vr=res_of(c->fn,v->rid); uint32_t ext=ptrext_get(c->fn,v->rid);
          if(vr && vr->is_vla && ext){   /* `sizeof a` of a 1-D stack VLA: a RUNTIME value -- the snapshot
                                          * extent × sizeof(element) (NOT a stale static fold of the 0-size
                                          * array CType). is_vla (NOT merely ptr_extent != 0) gates it: a
                                          * recovered malloc pointer is also in ptr_extent but is not a VLA. */
            c->i++; if(paren) eat(c,")");
            uint32_t r=temp(c,8);   /* an 8-byte size_t result */
            bcir_resource *rr=&c->fn->res[c->fn->n_res-1]; rr->is_signed=0;   /* mark it unsigned size_t */
            bcir_claim *cl=new_claim(c,"c.sizeof.vla",BCIR_OP_ADD);   /* ADD: a cost hint only (emit carries the ×) */
            if(cl){ cl->n_rd=1; cl->rd[0]=ext; cl->n_wr=1; cl->wr[0]=r; cl->n_imm=1; cl->imm[0]=(long long)vr->elem_bytes; }
            return r;
          }
        }
        if(v) size = v->type.kind==2?cc_abi(c)->pointer_size:(v->type.kind==1?c->s[v->sidx].size:v->type.size);
        c->i++;
        if(indexed){   /* `sizeof a[0]`: an element -> the static element size. sizeof is UNEVALUATED,
                        * so SKIP the index tokens by bracket-matching (do NOT lower them -- no claim). */
          int bd=0;
          do{ if(is(c,"[")) bd++; else if(is(c,"]")) bd--; c->i++; }while(bd>0 && !isk(c,T_END));
        }
      }
      if(paren) eat(c,")");
    }
    uint32_t r=temp(c,4); bcir_claim *cl=new_claim(c,"c.const",BCIR_OP_LOAD);
    if(cl){cl->n_wr=1;cl->wr[0]=r;cl->n_imm=1;cl->imm[0]=size;} return r;
  }
  if(is(c,"(")){c->i++;uint32_t r=p_expr(c);
    while(is(c,",")){c->i++;r=p_expr(c);}    /* the comma OPERATOR (lowest prec): lower each operand for its */
    eat(c,")");return r;}                    /* side effects, DISCARD all but the last, yield the last rid */
  if(isk(c,T_ID)){
    tok id=adv(c);
    const char *aop;bcir_opcode aoc;int akind;
    if(is(c,"(")&&atomic_kind(&id,&aop,&aoc,&akind)){    /* atomics/fences/CAS */
      int ordered = (id.n==21 && !strncmp("__atomic_thread_fence",id.s,21))   /* SEG6.1/SEG7: the order-taking */
                 || (id.n==19 && !strncmp("atomic_thread_fence",id.s,19));    /* fence forms route by their arg */
      return p_atomic(c,aop,aoc,akind,ordered);
    }
    if(is(c,"(")){ venv *fv=lookup(c,&id);        /* indirect call (funcptr var) vs. direct named call */
      if(fv&&fv->type.kind==3) return p_icall(c,fv);
      const bcir_ctype *rt=callee_ret(c,&id);     /* a struct-returning call: `mk(x).field` postfixes the result */
      int drop_save=c->call_dropped; c->call_dropped=0;   /* a call nested in an argument restores it */
      uint32_t r=p_call(c,&id);
      if(c->call_dropped && c->fn->n_claims) c->fn->claims[c->fn->n_claims-1].truncated=1;
      c->call_dropped=drop_save;
      if(rt && rt->kind==1 && (is(c,".")||is(c,"->")||is(c,"["))){   /* the by-value struct result is addressable */
        venv sv; memset(&sv,0,sizeof sv); sv.rid=r; sv.type=*rt;
        sv.sidx=find_struct(c,rt->tag,(int)strlen(rt->tag));
        if(sv.sidx>=0) return postfix_lvalue(c,&sv);
      }
      return r; }
    int ec=find_enum(c,id.s,id.n);                /* an enumerator -> its folded constant (type int) */
    if(ec>=0){uint32_t r=tempi(c,4,1);bcir_claim *cl=new_claim(c,"c.const",BCIR_OP_LOAD);
      if(cl){cl->n_wr=1;cl->wr[0]=r;cl->n_imm=1;cl->imm[0]=c->ec[ec].val;}return r;}
    venv *v=lookup(c,&id); if(!v) v=use_global(c,&id);   /* a file-scope global (lookup table)? */
    if(!v){
      if(callee_ret(c,&id)){                             /* a defined FUNCTION used as a VALUE (function-to-pointer
                                                          * decay, e.g. `o->fn = g`): a funcptr value emitted as the
                                                          * bare function name (C decays it). No claim. */
        char fnm[BCIR_CIR_NAME]; idcpy(fnm,&id);
        uint32_t r=add_res(c,BCIR_DOM_RAM,cc_abi(c)->pointer_size,1,0,BCIR_RK_SCALAR,fnm);
        if(c->fn->n_res){ bcir_resource *rr=&c->fn->res[c->fn->n_res-1]; rr->read_only=1; rr->is_funcptr=1; }
        return r;
      }
      if(is_imag_unit(&id)){                            /* <complex.h> imaginary unit (unless shadowed) */
        uint32_t r=tempc(c,8);                          /* `float _Complex` (value i), emitted verbatim */
        char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.cconst:%.*s",id.n,id.s);
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
      fail(c,"undefined identifier");return 0;
    }
    return postfix_lvalue(c,v);
  }
  fail(c,"expected expression");return 0;
}
/* name a cast's target type by width, so both rails emit the same (uintN_t) spelling. With signed_int
 * set, a width-named integer uses the SIGNED fixed-width spelling -- needed for a float -> signed-int
 * conversion, which is UB/target-divergent if rendered as float -> unsigned. */
static void cast_name(const bcir_ctype *ty,int signed_int,char *o,size_t n){
  if(ty->kind==2){   /* a POINTER target, spelled faithfully (the oracle's `_pointer_spelling`): the pointee's own
                      * type -- its sign, a plain `char`, `_Bool`, a float, a struct/union tag or `void` -- behind
                      * its `volatile`, then one `*` per level; the cast yields a real `T *` of that type */
    char base[BCIR_CIR_NAME+8];
    if(ty->ptr_to_struct) snprintf(base,sizeof base,"%s %s",ty->is_union?"union":"struct",ty->tag);
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
    snprintf(o,n,"c.cast:%s%s %s",ty->is_volatile?"volatile ":"",base,stars); return; }
  if(ty->kind==0 && ty->bit_width>0){               /* a `_BitInt(N)` cast target -- the exact spelling (faithful) */
    snprintf(o,n,"c.cast:%s_BitInt(%d)",ty->signd?"":"unsigned ",ty->bit_width); return; }
  const char *nm=ty->is_complex ? (ty->size==8?"float _Complex":ty->size>16?"long double _Complex":"double _Complex")
                : ty->is_bool ? "_Bool"           /* a bool cast normalizes any nonzero (full value) to 1 */
                : ty->is_float ? (ty->size==4?"float":ty->size>8?"long double":"double")  /* >8: extended (matches the oracle's `long double`) */
                : signed_int ? (ty->size==1?"int8_t":ty->size==2?"int16_t":ty->size==8?"int64_t":"int32_t")
                : ty->size==1?"uint8_t":ty->size==2?"uint16_t":ty->size==8?"uint64_t":"uint32_t";
  snprintf(o,n,"c.cast:%s",nm);
}
static int incdec_value(CC *c, uint32_t *out);   /* fwd: `++a`/`a++`/`--a`/`a--` in EXPRESSION position */
static uint32_t p_unary_inner(CC *c);
/* Depth-guarded wrapper: p_unary is a recursive-cycle entry point (p_unary->p_primary->`(`->p_expr->
 * ...->p_unary), so a deeply-nested expression would exhaust the native stack. Bump/check depth once
 * per level here; on overflow fail cleanly ("nesting too deep") and return without recursing. */
static uint32_t p_unary(CC *c) {
  if(ENTER_REC(c)){ LEAVE_REC(c); return 0; }
  uint32_t r=p_unary_inner(c); LEAVE_REC(c); return r;
}
static uint32_t p_unary_inner(CC *c) {
  { uint32_t v; if(incdec_value(c,&v)) return v; }   /* PREFIX ++a / POSTFIX a++ (member/array/pointer/scalar) */
  if(is(c,"+")){ c->i++; return p_unary(c); }    /* unary plus is a no-op */
  if(is(c,"__real__")||is(c,"__imag__")){        /* GNU complex part -> the real element float */
    const char *suf=is(c,"__real__")?"creal":"cimag"; c->i++;
    uint32_t a=p_unary(c); const bcir_resource *ar=res_of(c->fn,a);
    int es = (ar&&ar->is_complex)?(int)ar->elem_bytes/2 : (ar&&ar->is_float)?(int)ar->elem_bytes : 8;
    uint32_t r=tempf(c,es);                       /* not integer-computed; emitted `__real__ x` */
    char op[BCIR_CIR_NAME];snprintf(op,sizeof op,"c.un.%s",suf);
    bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);if(cl){cl->n_rd=1;cl->rd[0]=a;cl->n_wr=1;cl->wr[0]=r;}
    return r; }
  if(is(c,"&&")){ c->i++;                          /* `&&label` -- a label's address as a `void *` value (GNU).
    * Safe in unary position: a binary `&&` never STARTS a unary expression, so logical-AND is unaffected. */
    tok lb=adv(c);                                  /* the label identifier */
    uint32_t t=add_res(c,BCIR_DOM_RAM,cc_abi(c)->pointer_size,1,0,BCIR_RK_POINTER,"");   /* a `void *` temp */
    if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1]; tr->ptr_depth=1; tr->is_voidptr=1; }
    char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.labeladdr:%.*s",lb.n,lb.s);   /* emit `void *t = &&L;` */
    bcir_claim *cl=new_claim(c,op,BCIR_OP_LOAD); if(cl){cl->n_wr=1;cl->wr[0]=t;}   /* a LOAD claim, no reads */
    return t; }
  if(is(c,"&")){ c->i++;                          /* address-of: &lvalue -> a pointer value (c.addrof) */
    if(is(c,"*")){ c->i++; return p_unary(c); }    /* &*p == p (the pointer itself; &*(p+i) == p+i) */
    if(isk(c,T_ID)){ tok id=*pk(c); venv *vp=lookup(c,&id); if(!vp) vp=use_global(c,&id);
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
          if(mf.is_ptr && !mf.arr_count){          /* &s.ptr / &s->ptr -- address of a POINTER member -> a `T **` */
            uint32_t t=add_res(c,BCIR_DOM_RAM, mf.ptee_size?mf.ptee_size:4, 1,0,BCIR_RK_POINTER,"");
            if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];   /* pointee = the member's pointee */
              tr->is_signed=(uint8_t)(mf.signd?1:0); tr->is_float=(uint8_t)(mf.ptee_float?1:0); tr->ptr_depth=2;
              if(mf.ptee_sidx>=0) snprintf(tr->agg,sizeof tr->agg,"%s %s",c->s[mf.ptee_sidx].is_union?"union":"struct",c->s[mf.ptee_sidx].tag); }
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
            if(is(c,".")||is(c,"->")){             /* &s.arr[i].field -- array-of-structs element FIELD address */
              if(mf.elem_sidx<0){ fail(c,"address-of a field of a non-struct member-array element"); return 0; }
              sdef *ES=&c->s[mf.elem_sidx]; c->i++; tok efn=adv(c); int efi=-1;
              for(int k=0;k<ES->nf;k++) if((int)strlen(ES->f[k].name)==efn.n&&!strncmp(ES->f[k].name,efn.s,efn.n)) efi=k;
              if(efi<0){ fail(c,"unknown field"); return 0; }
              field ef=member_descend(c,ES->f[efi]);
              if(ef.bit_w||ef.is_ptr||ef.arr_count){ fail(c,"address-of a non-scalar array-of-structs field is a follow-on"); return 0; }
              off += ef.byte_off; resz=ef.size?ef.size:4; rsd=ef.signd; rfl=ef.is_float; rsx=ef.sidx;   /* field at member_off+field_off; stride stays the struct */
            }
            if(is(c,".")||is(c,"->")||is(c,"[")){  /* a further descent is a follow-on */
              fail(c,"address-of a nested member-array element is a follow-on"); return 0; }
            uint32_t t=add_res(c,BCIR_DOM_RAM, resz, 1,0,BCIR_RK_POINTER,"");   /* an `element/field *` */
            if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
              tr->is_signed=(uint8_t)(rsd?1:0); tr->is_float=(uint8_t)(rfl?1:0); tr->ptr_depth=1;
              if(rsx>=0) snprintf(tr->agg,sizeof tr->agg,"%s %s",c->s[rsx].is_union?"union":"struct",c->s[rsx].tag); }
            bcir_claim *cl=new_claim(c,"c.addrof",BCIR_OP_ADD);
            if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=ix;cl->n_wr=1;cl->wr[0]=t;cl->n_imm=2;cl->imm[0]=off;cl->imm[1]=es;}
            return t;
          }
          uint32_t t=add_res(c,BCIR_DOM_RAM, mf.size?mf.size:4, 1,0,BCIR_RK_POINTER,"");   /* a `leaf *` */
          if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
            tr->is_signed=(uint8_t)(mf.signd?1:0); tr->ptr_depth=1;
            if(mf.sidx>=0) snprintf(tr->agg,sizeof tr->agg,"%s %s",c->s[mf.sidx].is_union?"union":"struct",c->s[mf.sidx].tag); }
          bcir_claim *cl=new_claim(c,"c.addrof",BCIR_OP_ADD);
          if(cl){cl->n_rd=1;cl->rd[0]=v->rid;cl->n_wr=1;cl->wr[0]=t;cl->n_imm=1;cl->imm[0]=mf.byte_off;}
          return t;
        }
        if(is(c,"[")){                             /* &arr[i] / &p[i] -- a plain element address `(char*)base + i*es` */
          c->i++; uint32_t ix=p_expr(c); eat(c,"]");
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
              if(sub.sidx>=0) snprintf(tr->agg,sizeof tr->agg,"%s %s",c->s[sub.sidx].is_union?"union":"struct",c->s[sub.sidx].tag); }
            bcir_claim *cl=new_claim(c,"c.addrof",BCIR_OP_ADD);
            if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=ix;cl->n_wr=1;cl->wr[0]=t;cl->n_imm=2;cl->imm[0]=sub.byte_off;cl->imm[1]=strd;}
            return t;
          }
          if(is(c,".")||is(c,"->")||is(c,"[")){    /* &arr[i].field on a non-struct base / nested: a follow-on */
            fail(c,"address-of a plain-base array-of-structs element field is a follow-on"); return 0; }
          int es = v->type.size?v->type.size:4;    /* the pointee / element byte size */
          uint32_t t=add_res(c,BCIR_DOM_RAM, es, 1,0,BCIR_RK_POINTER,"");
          if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
            tr->is_signed=(uint8_t)(v->type.signd?1:0); tr->is_float=(uint8_t)(v->type.is_float?1:0); tr->ptr_depth=1;
            if(v->type.ptr_to_struct) snprintf(tr->agg,sizeof tr->agg,"%s %s",v->type.is_union?"union":"struct",v->type.tag); }
          bcir_claim *cl=new_claim(c,"c.addrof",BCIR_OP_ADD);
          if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=ix;cl->n_wr=1;cl->wr[0]=t;cl->n_imm=2;cl->imm[0]=0;cl->imm[1]=es;}
          return t;
        }
        /* a pointer one level deeper than the addressed object: */
        if(v->type.kind==2 && (v->type.ptr_depth?v->type.ptr_depth:1)>=BCIR_MAX_PTR_DEPTH){
          fail(c,"pointer nesting too deep");return 0;}
        uint32_t t=add_res(c,BCIR_DOM_RAM, v->type.size?v->type.size:4, 1,0,BCIR_RK_POINTER,"");  /* &p (T*) -> T** */
        if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
          tr->is_signed=(uint8_t)(v->type.signd?1:0); tr->is_float=(uint8_t)(v->type.is_float?1:0);
          tr->ptr_depth=(uint8_t)((v->type.kind==2?(v->type.ptr_depth?v->type.ptr_depth:1):0)+1);
          if(v->type.kind==1||v->type.ptr_to_struct) snprintf(tr->agg,sizeof tr->agg,"%s %s",v->type.is_union?"union":"struct",v->type.tag); }
        last_ptr_to_volatile(c,v->type.is_volatile,v->sidx>=0 && sdef_vol(c,v->sidx));   /* &x of volatile storage */
        bcir_claim *cl=new_claim(c,"c.addrof",BCIR_OP_ADD);
        if(cl){cl->n_rd=1;cl->rd[0]=v->rid;cl->n_wr=1;cl->wr[0]=t;} return t; } }
    if(is(c,"(")){ int save=c->i; c->i++;             /* `&(type){...}` -- address of a compound literal */
      int is_type = scalar_size(pk(c)->s,pk(c)->n)>=0 || is(c,"struct")||is(c,"union")||is(c,"enum")||is(c,"_Complex")||is(c,"complex")||is(c,"_BitInt")
                    || is(c,"const")||is(c,"volatile")
                    || is(c,"typeof")||is(c,"__typeof__")||is(c,"typeof_unqual")
                    || find_typedef(c,pk(c)->s,pk(c)->n)>=0;
      bcir_ctype ty; int si;
      if(is_type && !p_type(c,&ty,&si) && is(c,")")){ c->i++;
        if(is(c,"{")){ uint32_t rid=p_compound_literal(c,&ty,si);   /* materialize the anonymous object */
          uint32_t t=add_res(c,BCIR_DOM_RAM, ty.size?ty.size:4, 1,0,BCIR_RK_POINTER,"");   /* a `T *` to it */
          if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
            tr->is_signed=(uint8_t)(ty.signd?1:0); tr->is_float=(uint8_t)(ty.is_float?1:0); tr->ptr_depth=1;
            tr->is_plain_char=(uint8_t)(ty.is_plain_char?1:0);
            if(ty.kind==1) snprintf(tr->agg,sizeof tr->agg,"%s %s",ty.is_union?"union":"struct",ty.tag); }
          bcir_claim *cl=new_claim(c,"c.addrof",BCIR_OP_ADD);
          if(cl){cl->n_rd=1;cl->rd[0]=rid;cl->n_wr=1;cl->wr[0]=t;} return t; } }
      c->i=save; }
    fail(c,"unsupported address-of (only &local/&param/&(compound literal))"); return 0; }
  if(is(c,"-")||is(c,"~")||is(c,"!")){
    const char *suf=is(c,"-")?"neg":is(c,"~")?"bnot":"lnot"; int is_lnot=is(c,"!");
    bcir_opcode oc=is(c,"-")?BCIR_OP_SUB:BCIR_OP_ADD;c->i++;
    uint32_t a=p_unary(c);
    /* `-`/`~` take the promoted operand type (so negating a `long` stays 64-bit, not a truncated
     * uint32 that widens back to a positive long); `-x` on a float stays float (floats don't promote,
     * and a uint32 temp would truncate -2.5 to a huge integer); a sub-int integer operand promotes to
     * SIGNED int (§6.3.1.1), so `~(unsigned char)0` is -1, not 4294967295; `!` is int. */
    uint32_t r;
    if(is_lnot) r=tempi(c,4,1);
    else { const bcir_resource *ar=res_of(c->fn,a);
           if(ar&&ar->is_float) r=tempf(c,(int)ar->elem_bytes);          /* `-x` on a float is float */
           else { int sz=ar?(int)ar->elem_bytes:4, sg=ar?ar->is_signed:1;
                  promote_i(&sz,&sg);                                    /* sub-int -> signed int */
                  r=tempi(c,sz,sg); } }
    char op[BCIR_CIR_NAME];snprintf(op,sizeof op,"c.un.%s",suf);
    bcir_claim *cl=new_claim(c,op,oc);if(cl){cl->n_rd=1;cl->rd[0]=a;cl->n_wr=1;cl->wr[0]=r;}return r;}
  if(is(c,"*")){                                   /* pointer dereference: *p / *(p + i) */
    c->i++;
    if(is(c,"(")){ int save=c->i; c->i++;          /* *(p) or *(p + i) */
      if(isk(c,T_ID)){ tok pid=*pk(c); venv *pvp=lookup(c,&pid); if(!pvp) pvp=use_global(c,&pid);
        if(pvp){ c->i++; venv pvsnap=*pvp; venv *pv=&pvsnap;   /* SNAPSHOT: the `+ i` index p_expr below can realloc c->env[] */
          size_t s_res=c->fn->n_res,s_cl=c->fn->n_claims; uint32_t s_rid=c->rid,s_cid=c->cid,s_clc=c->cl_ctr;
          int through=1;                                /* `*(q[j] ...)`: through the loaded pointer element */
          if(is(c,"[")){ uint32_t lin=index_chain(c,pv); if(c->failed) return 0; through=step_to_elem_ptr(c,pv,lin); }
          if(through && is(c,"+")){ c->i++; uint32_t idx=p_expr(c); eat(c,")"); return emit_index(c,pv,idx); }
          if(through && is(c,")")){ c->i++; return emit_deref(c,pv); }
          c->fn->n_res=s_res;c->fn->n_claims=s_cl;c->rid=s_rid;c->cid=s_cid;c->cl_ctr=s_clc; } }   /* not ours: undo */
      c->i=save;
    } else if(isk(c,T_ID) && !(tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='[')){
      tok pid=*pk(c); venv *pv=lookup(c,&pid); if(!pv) pv=use_global(c,&pid);   /* `*q[j]` is `*(q[j])`: the */
      if(pv){ c->i++; return emit_deref(c,pv); } }  /* *p (no sub-parse between lookup and use); a global too --
                                                     * a subscript first takes the general path below */
    return emit_deref_rid(c, p_unary(c));            /* general: `**pp`, `*(<expr>)` -- deref a ptr rvalue */
  }
  if(is(c,"(") && tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='{')
    return p_stmt_expr(c);                          /* `({ ... })` -- a GCC statement expression */
  if(is(c,"(")){                                   /* (type)operand -- a cast binds at the unary level */
    int save=c->i; c->i++;
    int is_type = scalar_size(pk(c)->s,pk(c)->n)>=0 || is(c,"struct")||is(c,"union")||is(c,"enum")||is(c,"_Complex")||is(c,"complex")||is(c,"_BitInt")
                  || is(c,"const")||is(c,"volatile")
                    || is(c,"typeof")||is(c,"__typeof__")||is(c,"typeof_unqual")
                    || find_typedef(c,pk(c)->s,pk(c)->n)>=0;
    if(is_type){ bcir_ctype ty;int si;
      if(!p_type(c,&ty,&si)){
        int la_count=0,la_nd=0,la_dims[3]={0,0,0};   /* a `(T[N]...)` array type-name -> an array compound literal */
        while(is(c,"[")){ c->i++; int dim=isk(c,T_INT)?(int)adv(c).v:0; eat(c,"]");
          if(la_nd<3)la_dims[la_nd]=dim; la_nd++; la_count=la_count?la_count*dim:dim; }
        /* A SCALAR-element literal `(T[...]){...}` lowers for 1..3 dims; an AGGREGATE-element literal
         * `(struct P[]){...}` / `(struct P[N]){...}` (1-D) AND `(struct P[A][B]){...}` (multi-dim) also lower:
         * the struct element routes through subagg_init_struct / subagg_init_md_struct, and the indexing venv
         * carries BOTH `sidx` (so `[...].field` descends the element struct via emit_index_field, striding by
         * the struct size) AND `adims`/`nadims` (so `[i][j]` Horner-flattens the outer dims). */
        if(la_nd && la_nd<=3 && is(c,")") &&
           tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='{'){   /* `(T[...]){...}` */
          c->i++;                                  /* ')' -- p_array_literal/arr_init eats the following `{` */
          uint32_t rid=p_array_literal(c,&ty,si,la_count,la_dims,la_nd);   /* multi-dim -> subagg_init_md (row braces) */
          for(size_t z=c->fn->n_res;z-->0;) if(c->fn->res[z].rid==rid){ c->fn->res[z].is_array=1; break; }
          if(is(c,"[")){ venv sv; memset(&sv,0,sizeof sv); sv.rid=rid; sv.type=ty; sv.sidx=ty.kind==1?si:-1;
            if(la_nd>1){ for(int z=0;z<3;z++) sv.type.adims[z]=la_dims[z]; sv.type.nadims=la_nd; }
            return postfix_lvalue(c,&sv); }       /* `(int[]){...}[i]` / `(int[A][B]){...}[i][j]` / `(struct P[A][B]){...}[i][j].f` */
          return rid; }
      if(!la_nd && is(c,")")){
        c->i++;                                    /* ')' */
        if(is(c,"{")){ uint32_t rid=p_compound_literal(c,&ty,si);   /* `(type){init}` -- a compound literal, not a cast */
          if(is(c,".")||is(c,"->")||is(c,"[")){      /* direct postfix on the literal: `(struct P){...}.field` */
            venv sv; memset(&sv,0,sizeof sv); sv.rid=rid; sv.type=ty; sv.sidx=si; return postfix_lvalue(c,&sv); }
          return rid; }
        uint32_t v=p_unary(c);                     /* the operand (right-associative) */
        return emit_cast(c,v,&ty,si);
      } }
    }
    c->i=save;                                     /* not a cast -> a parenthesized expression */
  }
  return p_primary(c);
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
static uint32_t binop_result(CC *c, const char *suf, uint32_t lhs, uint32_t rhs){
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
  const bcir_resource *lr=res_of(c->fn,lhs), *rr=res_of(c->fn,rhs);
  int lp=lr&&lr->kind==BCIR_RK_POINTER, rp=rr&&rr->kind==BCIR_RK_POINTER;
  if((lp^rp) && (!strcmp(suf,"add")||!strcmp(suf,"sub"))) return tempptr(c, lp?lhs:rhs);  /* pointer ± int */
  return temp(c,4);
}
static uint32_t p_binrhs(CC *c,int min_prec,uint32_t lhs) {
  for(;;){
    char suf[BCIR_CIR_NAME];bcir_opcode oc;int idx=bin_op(c,suf,&oc);
    if(idx<0||prec_of(idx)<min_prec)return lhs;
    int prec=prec_of(idx);c->i++;uint32_t rhs=p_unary(c);
    char s2[BCIR_CIR_NAME];bcir_opcode o2;int nx=bin_op(c,s2,&o2);
    while(nx>=0&&prec_of(nx)>prec){rhs=p_binrhs(c,prec_of(nx),rhs);nx=bin_op(c,s2,&o2);}
    /* the result type: a relational/logical op is int; otherwise the usual arithmetic conversions
     * (float propagates the wider float; a shift the promoted LHS; else the integer UAC) -- shared. */
    int is_cmp=!strcmp(suf,"lt")||!strcmp(suf,"gt")||!strcmp(suf,"le")||!strcmp(suf,"ge")
             ||!strcmp(suf,"eq")||!strcmp(suf,"ne")||!strcmp(suf,"land")||!strcmp(suf,"lor");
    uint32_t r = is_cmp ? tempi(c,4,1)              /* relational / logical -> int */
                        : binop_result(c,suf,lhs,rhs);   /* the usual arithmetic conversions (shared) */
    char op[BCIR_CIR_NAME];snprintf(op,sizeof op,"c.bin.%s",suf);
    bcir_claim *cl=new_claim(c,op,oc);if(cl){cl->n_rd=2;cl->rd[0]=lhs;cl->rd[1]=rhs;cl->n_wr=1;cl->wr[0]=r;}
    lhs=r;
  }
}
static uint32_t p_binexpr(CC *c){return p_binrhs(c,1,p_unary(c));}
/* p_expr layers the ternary `cond ? then : els` over the binary expression: a scalar select claim
 * (both arms lowered, then chosen; the emitter renders the real `(cond ? a : b)`). */
/* the conditional-expression level (binary + ternary). p_expr wraps this with the assignment level below. */
static uint32_t p_cond(CC *c){
  uint32_t cond=p_binexpr(c);
  if(is(c,"?")){ c->i++; uint32_t a=p_expr(c); eat(c,":"); uint32_t b=p_expr(c);
    /* a select over ARITHMETIC arms carries their common type (usual arithmetic conversions), NOT a blanket
     * unsigned: a signed arm must keep its sign (else a downstream `>>` / compare goes unsigned) and a FLOAT
     * arm makes the result the wider float (else the double is truncated to int / a nan mis-converts). */
    int sa,za,sb,zb,fa,fb; uint32_t t;
    fa=rid_float(c,a,&sa); fb=rid_float(c,b,&sb);
    if(fa||fb){ int w=(fa?sa:0)>(fb?sb:0)?(fa?sa:0):(fb?sb:0); t=tempf(c,w); }   /* the wider float */
    else if(rid_int(c,a,&sa,&za) && rid_int(c,b,&sb,&zb)){ int rs,rz; uac_i(sa,za,sb,zb,&rs,&rz); t=tempi(c,rs,rz); }
    else t=temp(c,4);
    bcir_claim *cl=new_claim(c,"c.select",BCIR_OP_ADD);
    if(cl){cl->n_rd=3;cl->rd[0]=cond;cl->rd[1]=a;cl->rd[2]=b;cl->n_wr=1;cl->wr[0]=t;} return t; }
  return cond;
}

/* --- statements + functions ---------------------------------------------- */
static void env_add(CC *c,const tok *nm,uint32_t rid,const bcir_ctype *ty,int sidx){
  CC_ENSURE(c,c->env,c->nenv,c->cap_env);
  if(c->nenv>=c->cap_env)return; venv *v=&c->env[c->nenv++];
  idcpy(v->name,nm);v->rid=rid;v->type=*ty;v->sidx=sidx;
}
/* On first use of a file-scope global within a function, materialize a read-only data resource for
 * it (so an access `LUT[i]` lowers to a load) and bind it in the local env.  The resource is marked
 * read-only -- the emitter references the global by name (it is defined in the original source) and
 * does not redeclare it.  Subsequent uses in the same function resolve via the env. */
static venv *use_global(CC *c,const tok *id){
  int gi=find_global(c,id->s,id->n); if(gi<0) return NULL;
  venv *ex=lookup(c,id); if(ex) return ex;
  gvar *g=&c->gv[gi];
  int kind = g->count>1 ? BCIR_RK_POINTER : (g->ty.kind==1?BCIR_RK_AGGREGATE:BCIR_RK_SCALAR);
  uint32_t rid=add_res(c,ty_mmio(c,&g->ty,-1,g->is_arr)?BCIR_DOM_MMIO:BCIR_DOM_RAM,   /* a volatile global, or a */
                       g->ty.size,g->count,g->ty.is_volatile,kind,g->name);            /* pointer to volatile storage */
  if(c->fn->n_res){ bcir_resource *gr=&c->fn->res[c->fn->n_res-1];
    gr->read_only=1;                                  /* a global, not a local */
    gr->is_array=(uint8_t)(g->is_arr?1:0);
    gr->is_pointer=(uint8_t)(!g->is_arr&&(g->ty.kind==2||g->ty.kind==3)?1:0);
    /* the value type rides on the resource, as a local's does: every temp typed from it (a deref of the
     * array, `-g`, the usual arithmetic conversions, a volatile read) is the global's type, not uint32 */
    if(g->ty.kind==0 && kind==BCIR_RK_POINTER){       /* an array: its element, as a pointer local's pointee */
      gr->is_signed=(uint8_t)(g->ty.signd?1:0); gr->is_float=(uint8_t)(g->ty.is_float?1:0);
      gr->is_plain_char=(uint8_t)(g->ty.is_plain_char?1:0); }
    else if(g->ty.kind==0){                           /* a scalar: as a scalar local */
      gr->is_signed=(uint8_t)(g->ty.signd?1:0); gr->is_float=(uint8_t)(g->ty.is_float?1:0);
      gr->is_complex=(uint8_t)(g->ty.is_complex?1:0); gr->is_bool=(uint8_t)(g->ty.is_bool?1:0);
      gr->is_plain_char=(uint8_t)(g->ty.is_plain_char?1:0); gr->bit_width=g->ty.bit_width; } }
  env_add(c,id,rid,&g->ty,-1);
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
    mark_access(c,cl,f->is_volatile||base->type.is_volatile);}
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
static int lv_assign_value(CC *c, uint32_t *out){
  int save=c->i;
  /* --- a deref `*p = rhs` / `*(p + i) = rhs` / their `OP=` as a VALUE --- */
  if(is(c,"*")){
    c->i++;
    venv pvsnap; venv *pv=NULL; uint32_t idx=0; int has_idx=0, ok=0;
    /* SNAPSHOT the pointer's env entry: the `*(p + i)` index and the RHS p_assign below can declare locals
     * and realloc c->env[] -- a pointer into it dangles. The store/emit helpers only READ the venv. */
    if(is(c,"(")){ c->i++;                              /* *(p) or *(p + i) */
      if(isk(c,T_ID)){ tok pid=*pk(c); venv *pvp=lookup(c,&pid);
        if(pvp){ c->i++; pvsnap=*pvp; pv=&pvsnap;
          if(is(c,"+")){ c->i++; idx=p_expr(c); has_idx=1; if(eat(c,")")) ok=1; }
          else if(is(c,")")){ c->i++; ok=1; } } } }
    else if(isk(c,T_ID)){ tok pid=*pk(c); venv *pvp=lookup(c,&pid); if(pvp){ c->i++; pvsnap=*pvp; pv=&pvsnap; ok=1; } }   /* *p */
    const tok *op=&c->t[c->i];
    int is_eq = op->k==T_PUN && op->n==1 && op->s[0]=='=';
    if(ok && pv && pv->type.kind==2 && !pv->type.is_volatile      /* a non-volatile pointer to a scalar pointee */
       && pv->type.ptr_depth<=1 && (is_eq || is_compound_op(op))){
      int sz = pv->type.size?pv->type.size:4;
      bcir_ctype pst=pointee_slot(&pv->type);           /* `*p` stores a byte copy: C's conversion first */
      if(is_eq){                                       /* plain: rhs FIRST, then store, then RELOAD */
        c->i++; uint32_t rhs=p_assign(c);
        if(!has_idx) rhs=store_conv(c,rhs,&pst);
        bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
        if(cl){ if(has_idx){cl->n_rd=3;cl->rd[0]=pv->rid;cl->rd[1]=idx;cl->rd[2]=rhs;}
          else {cl->n_rd=2;cl->rd[0]=pv->rid;cl->rd[1]=rhs;cl->n_imm=2;cl->imm[0]=0;cl->imm[1]=sz;}
          cl->bounds=BCIR_BND_ASSUMED; }
        *out = has_idx ? emit_index(c,pv,idx) : emit_deref(c,pv);   /* reload the SAME resolved lvalue */
        return 1;
      }
      char ch=op->s[0]; c->i++;                         /* compound: read cur, binop, store, value = stored */
      uint32_t cur = has_idx ? emit_index(c,pv,idx) : emit_deref(c,pv);
      uint32_t rhs=p_assign(c);
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
      uint32_t tmp=binop_result(c,suf,cur,rhs); char o[BCIR_CIR_NAME]; snprintf(o,sizeof o,"c.bin.%s",suf);
      bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
      if(!has_idx) tmp=store_conv(c,tmp,&pst);          /* the value as stored (converted) */
      bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
      if(cl){ if(has_idx){cl->n_rd=3;cl->rd[0]=pv->rid;cl->rd[1]=idx;cl->rd[2]=tmp;}
        else {cl->n_rd=2;cl->rd[0]=pv->rid;cl->rd[1]=tmp;cl->n_imm=2;cl->imm[0]=0;cl->imm[1]=sz;}
        cl->bounds=BCIR_BND_ASSUMED; }
      /* the value of a compound is the STORED (narrowed) value: a sub-int target (`unsigned char *p;
       * (*p += v)`) truncates on store, so RE-READ (#narrowcompound); a full-width target needs no
       * re-read (tmp == the stored value), so `res` byte-unchanged (oracle: lv.ct.size < rt.size). */
      const bcir_resource *trr=res_of(c->fn,tmp);
      *out = (sz < (int)(trr?trr->elem_bytes:4)) ? (has_idx ? emit_index(c,pv,idx) : emit_deref(c,pv)) : tmp;
      return 1;
    }
    c->i=save;   /* not an eligible deref-assignment value -- rewind (any speculative index lowering is rolled */
  }              /* back below via the res/claim snapshot when we reach the array path; here nothing was emitted */
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
    int istart=c->i; c->i++; uint32_t idx=array_index(c,v);   /* resolve the index ONCE (Horner-flattened) */
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
      store_index_field(c,v,idx,&sub,rhs);
      *out=emit_index_field(c,v,idx,&sub);             /* reload reuses idx + field offset + element stride */
      return 1;
    }
    char ch=op->s[0]; c->i++;                           /* compound: read cur, binop, store, value = stored */
    uint32_t cur=emit_index_field(c,v,idx,&sub); uint32_t rhs=p_assign(c);
    const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
    uint32_t tmp=binop_result(c,suf,cur,rhs); char o[BCIR_CIR_NAME]; snprintf(o,sizeof o,"c.bin.%s",suf);
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
    if(v->type.kind!=0 || v->type.is_volatile){ c->i=save; return 0; }
    size_t s_res=c->fn->n_res,s_cl=c->fn->n_claims; uint32_t s_rid=c->rid,s_cid=c->cid,s_clc=c->cl_ctr;
    int istart=c->i; c->i++; uint32_t idx=array_index(c,v);   /* resolve the index ONCE (Horner-flattened) */
    const tok *op=&c->t[c->i];
    int is_eq = op->k==T_PUN && op->n==1 && op->s[0]=='=';
    if(!(is_eq || is_compound_op(op))){                /* `a[i]` not followed by `=`/OP= -> a plain value */
      c->fn->n_res=s_res;c->fn->n_claims=s_cl;c->rid=s_rid;c->cid=s_cid;c->cl_ctr=s_clc;
      c->i=istart-1; c->i=save; return 0;
    }
    if(is_eq){                                         /* plain: store rhs, then RELOAD the same index */
      c->i++; uint32_t rhs=p_assign(c);
      bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
      if(cl){cl->n_rd=3;cl->rd[0]=v->rid;cl->rd[1]=idx;cl->rd[2]=rhs;cl->bounds=access_bnd(c,v->rid);}
      *out=emit_index(c,v,idx);                         /* reload reuses idx */
      return 1;
    }
    char ch=op->s[0]; c->i++;                           /* compound: read cur, binop, store, value = stored */
    uint32_t cur=emit_index(c,v,idx); uint32_t rhs=p_assign(c);
    const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
    uint32_t tmp=binop_result(c,suf,cur,rhs); char o[BCIR_CIR_NAME]; snprintf(o,sizeof o,"c.bin.%s",suf);
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
    /* --- a MEMBER-ARRAY element `s.arr[i] = rhs` (and member array-of-structs `s.arr[i].f = rhs`) as a
     * VALUE: a STRIDED store at member_off (+ field_off) + idx*element-size, resolved ONCE, then re-read. --- */
    if(f.arr_count && !v->type.is_volatile && is(c,"[")){
      size_t m_res=c->fn->n_res,m_cl=c->fn->n_claims; uint32_t m_rid=c->rid,m_cid=c->cid,m_clc=c->cl_ctr;
      uint32_t idx=member_arr_index(c,&f);             /* the row-major flattened element index, resolved once */
      field sub; int soa = (is(c,".")||is(c,"->")) && elem_field(c,&f,&sub);   /* arr[i].field on AOS */
      if(c->failed) return 0;
      const field *sf = soa ? &sub : &f;               /* the stored slot: the element FIELD, or the array element */
      const tok *aop=&c->t[c->i];
      int a_eq = aop->k==T_PUN && aop->n==1 && aop->s[0]=='=';
      if(!(a_eq || is_compound_op(aop))){              /* a plain value, not an assignment -> undo + fall back */
        c->fn->n_res=m_res;c->fn->n_claims=m_cl;c->rid=m_rid;c->cid=m_cid;c->cl_ctr=m_clc;
        c->i=save; return 0; }
      if(a_eq){                                        /* plain: store rhs, then RELOAD the same strided slot */
        c->i++; uint32_t rhs=p_assign(c);
        store_member_index(c,v,&f,idx,soa,sf,rhs);
        *out = soa ? emit_member_index_field(c,v,&f,idx,&sub) : emit_member_index(c,v,&f,idx);
        return 1;
      }
      char ach=aop->s[0]; c->i++;                       /* compound: read cur, binop, store, value = stored */
      uint32_t cur = soa ? emit_member_index_field(c,v,&f,idx,&sub) : emit_member_index(c,v,&f,idx);
      uint32_t rhs=p_assign(c);
      const char *suf; bcir_opcode oc; compound_binop(ach,&suf,&oc);
      uint32_t tmp=binop_result(c,suf,cur,rhs); char o[BCIR_CIR_NAME]; snprintf(o,sizeof o,"c.bin.%s",suf);
      bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
      tmp=store_member_index(c,v,&f,idx,soa,sf,tmp);      /* the value as stored (converted) */
      /* a sub-int element/field truncates on store -> RE-READ the same strided slot (#narrowcompound); a
       * full-width element/field needs no re-read (oracle: lv.ct.size < rt.size). */
      const bcir_resource *trr=res_of(c->fn,tmp);
      *out = ((int)sf->size < (int)(trr?trr->elem_bytes:4))
             ? (soa ? emit_member_index_field(c,v,&f,idx,&sub) : emit_member_index(c,v,&f,idx)) : tmp;
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
      if(f.bit_w) store_member_bf(c,v,&f,rhs); else store_member(c,v,&f,rhs);   /* a nested bitfield: bf.set */
      *out=emit_member(c,v,&f,0);                       /* reload: a bitfield re-reads via bf.get (#bfassignexpr) */
      return 1;
    }
    char ch=op->s[0]; c->i++;                           /* compound: read cur, binop, store, value = stored */
    uint32_t cur=emit_member(c,v,&f,0); uint32_t rhs=p_assign(c);
    const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
    uint32_t tmp=binop_result(c,suf,cur,rhs); char o[BCIR_CIR_NAME]; snprintf(o,sizeof o,"c.bin.%s",suf);
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
      if(st.ptr_to_struct) snprintf(pr->agg,BCIR_CIR_NAME,"%s %s",st.is_union?"union":"struct",st.tag); }
  } else {
    old = st.is_float ? tempf(c,sz) : tempi(c,sz,st.signd?1:0);
    if(c->fn->n_res && st.is_plain_char) c->fn->res[c->fn->n_res-1].is_plain_char=1;
  }
  char op[BCIR_CIR_NAME]; cast_name(&st,0,op,sizeof op);   /* width-named UNSIGNED spelling, like _cast_name */
  bcir_claim *cl=new_claim(c,op,BCIR_OP_ADD); if(cl){cl->n_rd=1;cl->rd[0]=v->rid;cl->n_wr=1;cl->wr[0]=old;}
  return old;
}
static int incdec_value(CC *c, uint32_t *out){
  int prefix=0; char ch=0;
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
    c->i++; uint32_t idx=array_index(c,v);                 /* resolve the index ONCE (Horner-flattened) */
    field sub; int got=(is(c,".")||is(c,"->")) && aos_elem_field(c,v,&sub);
    if(c->failed) return 0;
    if(!got || !incdec_settle(c,prefix,save)){             /* not `a[i].field++` -> roll back, fall through */
      c->fn->n_res=s_res;c->fn->n_claims=s_cl;c->rid=s_rid;c->cid=s_cid;c->cl_ctr=s_clc; c->i=save; return 0; }
    uint32_t cur=emit_index_field(c,v,idx,&sub);           /* read OLD (a fresh declared temp -- no snapshot) */
    uint32_t one=incdec_emit_const1(c);
    uint32_t nw=binop_result(c,suf,cur,one); char o[BCIR_CIR_NAME]; snprintf(o,sizeof o,"c.bin.%s",suf);
    bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=one;b->n_wr=1;b->wr[0]=nw;}
    store_index_field(c,v,idx,&sub,nw);
    if(!prefix){ *out=cur; return 1; }
    const bcir_resource *nr=res_of(c->fn,nw);
    *out = ((int)sub.size < (int)(nr?nr->elem_bytes:4)) ? emit_index_field(c,v,idx,&sub) : nw;
    return 1;
  }

  /* --- a plain SCALAR ARRAY ELEMENT `a[i]` (kind 0, non-volatile, NOT an array-of-structs) --- */
  if(isk(c,T_ID) && v->type.kind==0 && !v->type.is_volatile && v->sidx<0
     && tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='['){
    size_t s_res=c->fn->n_res,s_cl=c->fn->n_claims; uint32_t s_rid=c->rid,s_cid=c->cid,s_clc=c->cl_ctr;
    c->i++; uint32_t idx=array_index(c,v);                 /* resolve the index ONCE (Horner-flattened) */
    if(c->failed){ return 0; }
    if(!incdec_settle(c,prefix,save)){                     /* `a[i]` not stepped -> roll back, fall through */
      c->fn->n_res=s_res;c->fn->n_claims=s_cl;c->rid=s_rid;c->cid=s_cid;c->cl_ctr=s_clc; c->i=save; return 0; }
    uint32_t cur=emit_index(c,v,idx);                      /* read OLD (a fresh declared temp -- no snapshot) */
    uint32_t one=incdec_emit_const1(c);
    uint32_t nw=binop_result(c,suf,cur,one); char o[BCIR_CIR_NAME]; snprintf(o,sizeof o,"c.bin.%s",suf);
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
    if(f.arr_count && is(c,"[")){                         /* a MEMBER-ARRAY element `s.arr[i]` / `s.arr[i].f` */
      uint32_t idx=member_arr_index(c,&f);
      field sub; int soa=(is(c,".")||is(c,"->")) && elem_field(c,&f,&sub);
      if(c->failed) return 0;
      const field *sf = soa ? &sub : &f;                  /* the stored slot: the element FIELD, or the element */
      if(!incdec_settle(c,prefix,save)){
        c->fn->n_res=s_res;c->fn->n_claims=s_cl;c->rid=s_rid;c->cid=s_cid;c->cl_ctr=s_clc; c->i=save; return 0; }
      uint32_t cur = soa ? emit_member_index_field(c,v,&f,idx,&sub) : emit_member_index(c,v,&f,idx);
      uint32_t one=incdec_emit_const1(c);
      uint32_t nw=binop_result(c,suf,cur,one); char o[BCIR_CIR_NAME]; snprintf(o,sizeof o,"c.bin.%s",suf);
      bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=one;b->n_wr=1;b->wr[0]=nw;}
      store_member_index(c,v,&f,idx,soa,sf,nw);
      if(!prefix){ *out=cur; return 1; }
      const bcir_resource *nr=res_of(c->fn,nw);
      *out = ((int)sf->size < (int)(nr?nr->elem_bytes:4))
             ? (soa ? emit_member_index_field(c,v,&f,idx,&sub) : emit_member_index(c,v,&f,idx)) : nw;
      return 1;
    }
    if(f.is_ptr || f.arr_count || f.sidx>=0){ c->i=save; return 0; }   /* a non-scalar leaf -> fallback */
    if(!incdec_settle(c,prefix,save)){                    /* a deeper lvalue / not stepped -> roll back */
      c->fn->n_res=s_res;c->fn->n_claims=s_cl;c->rid=s_rid;c->cid=s_cid;c->cl_ctr=s_clc; c->i=save; return 0; }
    uint32_t cur=emit_member(c,v,&f,0);                   /* read OLD (a fresh declared temp -- no snapshot) */
    uint32_t one=incdec_emit_const1(c);
    uint32_t nw=binop_result(c,suf,cur,one); char o[BCIR_CIR_NAME]; snprintf(o,sizeof o,"c.bin.%s",suf);
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
  if(v->type.is_volatile){ c->i=save; return 0; }          /* a volatile/MMIO target stays a both-rails fallback */
  if(v->type.kind==2){                                     /* a POINTER local steps by element (in place) */
    uint32_t old = !prefix ? incdec_snapshot(c,v) : 0;     /* postfix: snapshot the pre-step pointer first */
    uint32_t one=incdec_emit_const1(c);
    char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.ptr%s",ch=='+'?"add":"sub");
    bcir_claim *cl=new_claim(c,op,BCIR_OP_ADD); if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=one;cl->n_wr=1;cl->wr[0]=v->rid;}
    *out = prefix ? v->rid : old;                          /* prefix: the stepped pointer; postfix: the snapshot */
    return 1;
  }
  /* a SCALAR named local: cur ± 1, stored back into the (addressable) storage via a memory c.store */
  uint32_t old = !prefix ? incdec_snapshot(c,v) : 0;       /* postfix: snapshot OLD before the store clobbers it */
  uint32_t one=incdec_emit_const1(c);
  uint32_t nw=binop_result(c,suf,v->rid,one); char o[BCIR_CIR_NAME]; snprintf(o,sizeof o,"c.bin.%s",suf);
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
        tok tnm=c->t[c->i]; c->i+=2; int ist=c->i; uint32_t rhs=p_assign(c); int ien=c->i;
        bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD); if(cl){cl->n_rd=1;cl->rd[0]=rhs;cl->n_wr=1;cl->wr[0]=v->rid;}
        mark_obj_write(c,cl,v);
        bind_extent(c,v->rid,res_of(c->fn,v->rid),&tnm,ist,ien);   /* §5.12: `p = malloc(N*…)` -> N */
        return v->rid;
      }
      char ch=op->s[0];                              /* name OP= rhs */
      if(v->type.kind==2 && (ch=='+'||ch=='-')){     /* pointer += / -= : a single pointer-arith claim */
        c->i+=2; uint32_t rhs=p_assign(c);
        char o[BCIR_CIR_NAME]; snprintf(o,sizeof o,"c.ptr%s",ch=='+'?"add":"sub");
        bcir_claim *cl=new_claim(c,o,BCIR_OP_ADD); if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=rhs;cl->n_wr=1;cl->wr[0]=v->rid;}
        return v->rid;
      }
      uint32_t cur=named_read(c,v);                 /* the current value (a volatile read, first) */
      c->i+=2; uint32_t rhs=p_assign(c);
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
      uint32_t tmp=binop_result(c,suf,cur,rhs); char o[BCIR_CIR_NAME]; snprintf(o,sizeof o,"c.bin.%s",suf);
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
      if(mf.bit_w) store_member_bf(c,mbase,&mf,rhs); else store_member(c,mbase,&mf,rhs);   /* a bitfield: bf.set */
      return emit_member(c,mbase,&mf,0);            /* reload: a bitfield re-reads via bf.get (#bfassignexpr) */
    }
    char ch=op->s[0]; c->i++;                       /* compound `OP=`: read-once, binop, store, value = stored */
    uint32_t cur=emit_member(c,mbase,&mf,0);        /* read FIRST (matches the oracle's claim order) */
    uint32_t rhs=p_assign(c);
    const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
    uint32_t tmp=binop_result(c,suf,cur,rhs); char o[BCIR_CIR_NAME]; snprintf(o,sizeof o,"c.bin.%s",suf);
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
/* A braced aggregate initializer for a local struct/union (the C twin of lower._agg_init): the local
 * is declared with a `= {0}` zero baseline (emit), then each initialized member is a c.store, reusing
 * the member-store path. Positional entries advance a cursor; a `.field=` designator selects by name.
 * (Local arrays + `[i]=` designators need a local array declarator -- a follow-on.) */
static uint32_t p_expr(CC *c);
/* Parse a struct/union designator chain `.field (.field | [const-index] ...)*` -- the C twin of a nested
 * designator list (#designate). `.field` selects a member (descending into a nested value-struct/union),
 * `[i]` folds a constant index into a member array (Horner over its declared dims). Fills the cumulative
 * byte offset, the leaf store size + bitfield position (bit_w 0 if not a bitfield), and the TOP-LEVEL
 * field index (for the positional cursor). The cursor is left at the `=`. (An array-of-struct element --
 * nesting past `[i]` -- is a follow-on: the field model drops the element's struct index.) */
static int designator_chain(CC *c, sdef *S, int *off, int *size, int *bit_w, int *bit_off, int *top_fi,
                            const field **leaf){
  *off=0; *size=4; *bit_w=0; *bit_off=0; *top_fi=0;
  sdef *cur=S; field *F=NULL; int started=0;
  for(;;){
    if(is(c,".")){
      if(!cur){ fail(c,"member designator into a non-aggregate"); return 1; }
      c->i++; tok fld=adv(c); int fi=-1;
      for(int k=0;k<cur->nf;k++) if((int)strlen(cur->f[k].name)==fld.n&&!strncmp(cur->f[k].name,fld.s,fld.n)) fi=k;
      if(fi<0){ fail(c,"unknown field in designator"); return 1; }
      F=&cur->f[fi]; *off+=F->byte_off; *size=F->size; *bit_w=F->bit_w; *bit_off=F->bit_off;
      if(!started){ *top_fi=fi; started=1; }
      cur = (F->sidx>=0) ? &c->s[F->sidx] : NULL;      /* descend a nested value-struct/union */
    } else if(is(c,"[")){
      if(!F || F->arr_count<=0){ fail(c,"array designator into a non-array member"); return 1; }
      int idxs[3], ni=0;                               /* one or more `[i]` -> Horner-flatten via the dims */
      while(is(c,"[")){ c->i++; long long ix=ce_expr(c,0); eat(c,"]"); if(ni<3) idxs[ni++]=(int)ix; }
      int lin=ni?idxs[0]:0;
      for(int d=1; d<ni; d++){ int dim=d<F->nadims?F->adims[d]:1; lin=lin*dim+idxs[d]; }
      *off += lin*F->size; *size=F->size; *bit_w=0; *bit_off=0;   /* an element: never a bitfield */
      cur=NULL;
    } else break;
  }
  if(!started){ fail(c,"empty designator"); return 1; }
  *leaf=F;                                             /* the leaf: a member, or an array member's element */
  return 0;
}
static void agg_init_at(CC *c, uint32_t rid, int sidx, int base_off, int do_zinit);   /* fwd: mutual recursion */
/* A NESTED braced initializer for an array member inside a struct/union init -- `struct S s = { {e0,e1,
 * ..}, n };` (the C twin of lower._init_subagg). Stores each element with an OFFSET-based member c.store
 * at `base_off + idx*es` (so it composes through the enclosing struct's `= {0}` baseline); positional
 * entries advance a cursor, `[i]=` jumps it, gaps zero-fill. Each value converts to the element's declared
 * type `elem` (`store_conv`), exactly like a `s.arr[i] = v` element write. */
static void subagg_init(CC *c, uint32_t rid, int base_off, int es, int is_bool, const bcir_ctype *elem) {
  eat(c,"{");
  int cursor=0;
  while(!is(c,"}")&&!isk(c,T_END)&&!c->failed){
    int idx=cursor;
    if(is(c,"[")){ c->i++; idx=(int)ce_expr(c,0); eat(c,"]"); eat(c,"="); }   /* [const-index] = */
    uint32_t v=store_conv(c,p_expr(c),elem);
    bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
    if(cl){cl->n_rd=2;cl->rd[0]=rid;cl->rd[1]=v;cl->n_imm=2;cl->imm[0]=base_off+idx*es;cl->imm[1]=es;cl->bounds=BCIR_BND_ASSUMED;
      if(is_bool){cl->imm[2]=1;cl->n_imm=3;}}          /* a _Bool[] element init normalizes the value */
    cursor=idx+1;
    if(is(c,",")) c->i++;
  }
  eat(c,"}");
}
/* A NESTED-brace initializer for a MULTI-dim local array `T a[d0][d1]... = { {..}, {..} }` (the C twin of
 * the oracle's _init_subagg multi-dim row descent). The flat resource keeps the same row-major memory layout
 * as `T a[d0][d1]`, so each outer brace descends by ROW: row `r` lands at `base_off + r*stride` where
 * `stride = product(dims[1:]) * es` (the leaf element size `es`). A nested brace recurses with the inner dims
 * (`dims+1`, `nd-1`); a scalar inside the innermost dim stores at its element offset (an OFFSET-based member
 * c.store at `base_off + idx*es`, exactly like subagg_init -- composing through the enclosing `= {0}`
 * baseline), converted to the element's declared type `elem`. Positional entries advance a cursor, `[i]=`
 * jumps it, gaps zero-fill (§6.7.10). */
static void subagg_init_md_inner(CC *c, uint32_t rid, int base_off, const int *dims, int nd, int es, int is_bool,
                                 const bcir_ctype *elem);
/* Depth-guarded wrapper: subagg_init_md self-recurses per nested-row brace, so a deeply nested
 * multi-dim `{{{...}}}` initializer would exhaust the stack. Bump/check depth once per row level. */
static void subagg_init_md(CC *c, uint32_t rid, int base_off, const int *dims, int nd, int es, int is_bool,
                           const bcir_ctype *elem) {
  if(ENTER_REC(c)){ LEAVE_REC(c); return; }
  subagg_init_md_inner(c, rid, base_off, dims, nd, es, is_bool, elem); LEAVE_REC(c);
}
static void subagg_init_md_inner(CC *c, uint32_t rid, int base_off, const int *dims, int nd, int es, int is_bool,
                                 const bcir_ctype *elem) {
  eat(c,"{");
  int stride=es;                                          /* row stride = product(dims[1:]) * es */
  for(int d=1; d<nd; d++) stride*=dims[d];
  int cursor=0;
  while(!is(c,"}")&&!isk(c,T_END)&&!c->failed){
    int idx=cursor;
    if(is(c,"[")){ c->i++; idx=(int)ce_expr(c,0); eat(c,"]"); eat(c,"="); }   /* [const-index] = */
    if(nd>1 && is(c,"{")){                                /* an outer dim takes a nested ROW brace */
      subagg_init_md(c, rid, base_off+idx*stride, dims+1, nd-1, es, is_bool, elem);
    } else {                                              /* innermost dim: a scalar element store */
      uint32_t v=store_conv(c,p_expr(c),elem);
      bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
      if(cl){cl->n_rd=2;cl->rd[0]=rid;cl->rd[1]=v;cl->n_imm=2;cl->imm[0]=base_off+idx*es;cl->imm[1]=es;cl->bounds=BCIR_BND_ASSUMED;
        if(is_bool){cl->imm[2]=1;cl->n_imm=3;}}           /* a _Bool[] element init normalizes the value */
    }
    cursor=idx+1;
    if(is(c,",")) c->i++;
  }
  eat(c,"}");
}
/* A nested braced initializer for an ARRAY-OF-STRUCTS member -- `{ {a,b}, {c,d}, ... }`: each element brace
 * recurses into the element struct's init at `base_off + idx*stride` (the C twin of the oracle's _init_subagg
 * array-of-struct branch). Composes through the enclosing `= {0}` baseline; `[i]=` jumps, gaps zero-fill.
 * Returns the element count reached (max index + 1), so an inferred-size `struct P a[] = {...}` sizes itself. */
static int subagg_init_struct(CC *c, uint32_t rid, int base_off, int elem_sidx, int stride) {
  eat(c,"{");
  int cursor=0, n=0;
  while(!is(c,"}")&&!isk(c,T_END)&&!c->failed){
    int idx=cursor;
    if(is(c,"[")){ c->i++; idx=(int)ce_expr(c,0); eat(c,"]"); eat(c,"="); }   /* [const-index] = */
    if(!is(c,"{")){ fail(c,"array-of-structs element needs a brace initializer"); return n; }
    agg_init_at(c, rid, elem_sidx, base_off+idx*stride, 0);                   /* {a,b} -> the element struct */
    cursor=idx+1; if(cursor>n) n=cursor;
    if(is(c,",")) c->i++;
  }
  eat(c,"}");
  return n;
}
/* A NESTED-brace initializer for a MULTI-dim AGGREGATE-element array `struct P a[d0][d1]... = { {..}, {..} }`
 * (the C twin of the oracle's _array_row / _init_subagg multi-dim struct descent). The flat resource keeps the
 * row-major `struct P a[d0][d1]` layout, so each OUTER brace descends by ROW: row `r` lands at `base_off +
 * r*stride` where `stride = product(dims[1:]) * es` (the struct element size `es`). The INNERMOST dim takes a
 * per-element STRUCT brace `{a,b}` -- routed through agg_init_at at `base_off + idx*es` (per-element field
 * stores at absolute offsets, exactly like subagg_init_struct), composing through the enclosing `= {0}`
 * baseline. Positional entries advance a cursor, `[i]=` jumps it, gaps zero-fill (§6.7.10). `elem_sidx` is the
 * element struct's sdef index. */
static void subagg_init_md_struct_inner(CC *c, uint32_t rid, int base_off, const int *dims, int nd, int es, int elem_sidx);
/* Depth-guarded wrapper: subagg_init_md_struct self-recurses per nested-row brace. */
static void subagg_init_md_struct(CC *c, uint32_t rid, int base_off, const int *dims, int nd, int es, int elem_sidx) {
  if(ENTER_REC(c)){ LEAVE_REC(c); return; }
  subagg_init_md_struct_inner(c, rid, base_off, dims, nd, es, elem_sidx); LEAVE_REC(c);
}
static void subagg_init_md_struct_inner(CC *c, uint32_t rid, int base_off, const int *dims, int nd, int es, int elem_sidx) {
  eat(c,"{");
  int stride=es;                                          /* row stride = product(dims[1:]) * es */
  for(int d=1; d<nd; d++) stride*=dims[d];
  int cursor=0;
  while(!is(c,"}")&&!isk(c,T_END)&&!c->failed){
    int idx=cursor;
    if(is(c,"[")){ c->i++; idx=(int)ce_expr(c,0); eat(c,"]"); eat(c,"="); }   /* [const-index] = */
    if(nd>1){                                             /* an outer dim takes a nested ROW brace */
      if(!is(c,"{")){ fail(c,"multi-dim aggregate literal needs a row brace"); return; }
      subagg_init_md_struct(c, rid, base_off+idx*stride, dims+1, nd-1, es, elem_sidx);
    } else {                                              /* innermost dim: a per-element struct brace `{a,b}` */
      if(!is(c,"{")){ fail(c,"array-of-structs element needs a brace initializer"); return; }
      agg_init_at(c, rid, elem_sidx, base_off+idx*es, 0);
    }
    cursor=idx+1;
    if(is(c,",")) c->i++;
  }
  eat(c,"}");
}
static void agg_init_at_inner(CC *c, uint32_t rid, int sidx, int base_off, int do_zinit);
/* Depth-guarded wrapper: agg_init_at is the aggregate-initializer recursion entry (agg_init_at<->
 * subagg_init_struct/subagg_init_md/subagg_init_md_struct and the nested-brace re-entry), so a deeply
 * nested `{{{...}}}` initializer would exhaust the stack. Bump/check depth once per brace level. */
static void agg_init_at(CC *c, uint32_t rid, int sidx, int base_off, int do_zinit) {
  if(ENTER_REC(c)){ LEAVE_REC(c); return; }
  agg_init_at_inner(c, rid, sidx, base_off, do_zinit); LEAVE_REC(c);
}
static void agg_init_at_inner(CC *c, uint32_t rid, int sidx, int base_off, int do_zinit) {
  eat(c,"{");
  sdef *S = sidx>=0 ? &c->s[sidx] : NULL;
  if(!S){ fail(c,"aggregate initializer needs a struct/union type"); return; }
  if(do_zinit) for(size_t i=0;i<c->fn->n_res;i++) if(c->fn->res[i].rid==rid){ c->fn->res[i].zinit=1; break; }  /* = {0} */
  int cursor=0;
  while(!is(c,"}")&&!isk(c,T_END)&&!c->failed){
    int off=0, size=4, bit_w=0, bit_off=0, top_fi=cursor, skip=0, fbool=0, abytes=4;   /* store target (chain/positional) */
    const field *leaf=NULL;                             /* the member the value lands in: its type, flag, unit */
    if(is(c,".")||is(c,"[")){                           /* a (possibly nested) designator */
      if(designator_chain(c,S,&off,&size,&bit_w,&bit_off,&top_fi,&leaf)) return;
      eat(c,"=");
    } else if(cursor<S->nf){ leaf=&S->f[cursor];        /* positional: the cursor-th member */
      off=leaf->byte_off; size=leaf->size; bit_w=leaf->bit_w; bit_off=leaf->bit_off;
    } else skip=1;                                      /* past the last member -> parse but do not store */
    if(leaf){ fbool=leaf->is_bool; abytes=leaf->access_bytes; }   /* the LEAF's, through a nested designator */
    off += base_off;                                   /* shift into the enclosing object (nested aggregate) */
    if(!skip && is(c,"{")){                             /* a NESTED brace: an aggregate (array OR struct) member */
      field *AF = (top_fi>=0 && top_fi<S->nf) ? &S->f[top_fi] : NULL;
      if(AF && AF->arr_count>0 && AF->elem_sidx>=0){      /* an ARRAY-OF-STRUCTS member: `{ {a,b}, {c,d} }` */
        subagg_init_struct(c, rid, off, AF->elem_sidx, AF->size); cursor=top_fi+1; if(is(c,",")) c->i++; continue; }
      if(AF && AF->arr_count>0){ bcir_ctype afs=field_slot(AF);
        subagg_init(c, rid, off, AF->size, AF->is_bool, &afs); cursor=top_fi+1; if(is(c,",")) c->i++; continue; }
      if(AF && AF->sidx>=0){ agg_init_at(c, rid, AF->sidx, off, 0); cursor=top_fi+1; if(is(c,",")) c->i++; continue; }
      fail(c,"nested initializer for a non-aggregate member"); return;
    }
    uint32_t v=p_expr(c); uint32_t val=v;
    if(skip){ cursor=top_fi+1; if(is(c,",")) c->i++; continue; }
    if(leaf){ bcir_ctype ls=field_slot(leaf); v=store_conv(c,v,&ls); val=v; }   /* C's conversion to the leaf */
    if(bit_w){                                          /* a bitfield member: read unit, set bits, store */
      int absz=abytes<=4?4:8;
      uint32_t unit=temp(c,absz);
      bcir_claim *ld=new_claim(c,"c.load",BCIR_OP_LOAD);
      if(ld){ld->n_rd=1;ld->rd[0]=rid;ld->n_wr=1;ld->wr[0]=unit;ld->n_imm=2;ld->imm[0]=off;ld->imm[1]=abytes;ld->bounds=BCIR_BND_ASSUMED;}
      uint32_t nu=temp(c,absz);
      bcir_claim *bs=new_claim(c,"c.bf.set",BCIR_OP_ADD);
      if(bs){bs->n_rd=2;bs->rd[0]=unit;bs->rd[1]=v;bs->n_wr=1;bs->wr[0]=nu;bs->n_imm=2;bs->imm[0]=bit_off;bs->imm[1]=bit_w;}
      val=nu; }
    bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
    if(cl){cl->n_rd=2;cl->rd[0]=rid;cl->rd[1]=val;cl->n_imm=2;cl->imm[0]=off;cl->imm[1]=bit_w?abytes:size;cl->bounds=BCIR_BND_ASSUMED;
      if(bit_w){cl->imm[2]=2;cl->n_imm=3;}             /* a bitfield UNIT store: `_v` takes the unit's full type */
      else if(fbool){cl->imm[2]=1;cl->n_imm=3;}}       /* a _Bool member init normalizes the value */
    cursor=top_fi+1;
    if(is(c,",")) c->i++;
  }
  eat(c,"}");
}
static void agg_init(CC *c, uint32_t rid, int sidx) { agg_init_at(c, rid, sidx, 0, 1); }
/* A C99 compound literal `(type){init}` -- an anonymous local of `type`, initialized exactly like a
 * braced local decl and yielded as an rvalue rid (a by-value struct arg, an assignment RHS, a scalar
 * value), or addressed under `&`. A struct/union reuses agg_init (the `= {0}` zero baseline + a c.store
 * per initialized member); a scalar `(int){v}` copies the single value in. (Direct postfix on a literal
 * `(struct P){...}.f` and array literals `(int[]){...}` are deferred follow-ons.) */
static uint32_t p_compound_literal(CC *c, const bcir_ctype *ty, int si) {
  char nm[BCIR_CIR_NAME]; snprintf(nm,sizeof nm,"_cl%u",++c->cl_ctr);
  if(ty->kind==1){                                   /* a struct/union compound literal */
    uint32_t rid=add_res(c,BCIR_DOM_RAM, si>=0?c->s[si].size:ty->size, 1, 0, BCIR_RK_AGGREGATE, nm);
    if(c->fn->n_res) snprintf(c->fn->res[c->fn->n_res-1].agg,BCIR_CIR_NAME,"%s %s",
                              ty->is_union?"union":"struct", ty->tag);
    agg_init(c, rid, si);                            /* parses the `{...}`: zinit + a c.store per member */
    return rid;
  }
  /* a scalar compound literal `(int){v}` -- a named scalar local + a c.copy of the single value */
  uint32_t rid=add_res(c,BCIR_DOM_RAM, ty->size?ty->size:4, 1, 0, BCIR_RK_SCALAR, nm);
  if(c->fn->n_res){ bcir_resource *rr=&c->fn->res[c->fn->n_res-1];
    rr->is_signed=(uint8_t)(ty->signd?1:0); rr->is_float=(uint8_t)(ty->is_float?1:0);
    rr->is_bool=(uint8_t)(ty->is_bool?1:0); rr->is_plain_char=(uint8_t)(ty->is_plain_char?1:0); }
  eat(c,"{");
  uint32_t v;
  if(is(c,"}")){ v=temp(c, ty->size?ty->size:4);     /* `(int){}` (C23 empty) -> 0 */
    bcir_claim *k=new_claim(c,"c.const",BCIR_OP_LOAD); if(k){k->n_wr=1;k->wr[0]=v;k->n_imm=1;k->imm[0]=0;} }
  else v=p_expr(c);
  if(is(c,",")) c->i++;                              /* a tolerated trailing comma */
  eat(c,"}");
  bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD);
  if(cl){cl->n_rd=1;cl->rd[0]=v;cl->n_wr=1;cl->wr[0]=rid;}
  return rid;
}
/* A braced initializer for a local array `T a[N] = {...}` (the C twin of the oracle's array _agg_init):
 * a `= {0}` zero baseline (emit) + a c.store per initialized element. Positional entries advance a
 * cursor; a `[i]=` designator (a folded constant index) jumps it -- gaps zero-fill (§6.7.10). Returns
 * the element count reached (max index + 1), so an inferred-size literal `(T[]){...}` can size itself. */
static int arr_init(CC *c, uint32_t rid) {
  eat(c,"{");
  for(size_t i=0;i<c->fn->n_res;i++) if(c->fn->res[i].rid==rid){ c->fn->res[i].zinit=1; break; }
  int cursor=0, n=0;
  while(!is(c,"}")&&!isk(c,T_END)&&!c->failed){
    int idx=cursor;
    if(is(c,"[")){ c->i++; idx=(int)ce_expr(c,0); eat(c,"]"); eat(c,"="); }   /* [const-index] = */
    uint32_t v=p_expr(c);
    uint32_t ic=temp(c,4); bcir_claim *kc=new_claim(c,"c.const",BCIR_OP_LOAD);
    if(kc){kc->n_wr=1;kc->wr[0]=ic;kc->n_imm=1;kc->imm[0]=idx;}
    bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
    if(cl){cl->n_rd=3;cl->rd[0]=rid;cl->rd[1]=ic;cl->rd[2]=v;cl->bounds=access_bnd(c,rid);}  /* §5.12 promote a known-extent array */
    cursor=idx+1; if(cursor>n) n=cursor;
    if(is(c,",")) c->i++;
  }
  eat(c,"}");
  return n;
}
/* Peek the OUTER dim of a braced initializer `{ e0, e1, ... }` (the count of top-level row entries) WITHOUT
 * consuming it -- the cursor must be at the opening `{`. Mirrors the oracle's outer-count inference (lower.py
 * ~1318-1327, max INDEX + 1): each TOP-LEVEL entry sits at a cursor; a positional entry advances it, a leading
 * `[const]=` designator JUMPS the cursor to that constant index; the result is `max(seen index)+1`. Nested
 * `{}` / `()` / `[]` are skipped by depth so a comma -- OR a designator `[..]` -- INSIDE an inner brace is not
 * counted at the top level (a `[k]=` designator is only honored at depth 0 AND at an entry start). */
static int peek_top_entries(CC *c) {
  int j=c->i; if(!tok_is(&c->t[j],"{")) return 0; j++;
  int cursor=0, n=0, depth=0, at_entry_start=1;
  while(c->t[j].k!=T_END){
    if(depth==0 && tok_is(&c->t[j],"}")) break;
    if(depth==0 && tok_is(&c->t[j],",")){            /* a top-level entry separator -> the next entry */
      cursor++; at_entry_start=1; j++; continue;
    }
    if(depth==0 && at_entry_start && tok_is(&c->t[j],"[")
       && c->t[j+1].k==T_INT && tok_is(&c->t[j+2],"]")){   /* a `[const]=` outer designator -> JUMP the cursor */
      cursor=(int)c->t[j+1].v;                       /* the simple integer-literal outer index (a nested chain's */
      j+=3; if(tok_is(&c->t[j],"=")) j++;            /* deeper `[..]` is inside the row, not a top-level designator) */
      if(cursor+1>n) n=cursor+1;                     /* this entry sits at `cursor`, so max index+1 = cursor+1 */
      at_entry_start=0; continue;
    }
    if(depth==0 && at_entry_start){ if(cursor+1>n) n=cursor+1; at_entry_start=0; }   /* a positional entry */
    if(tok_is(&c->t[j],"{")||tok_is(&c->t[j],"(")||tok_is(&c->t[j],"[")) depth++;
    else if(tok_is(&c->t[j],"}")||tok_is(&c->t[j],")")||tok_is(&c->t[j],"]")) depth--;
    j++;
  }
  return n<1?1:n;
}
/* An array compound literal `(T[N]){...}` / `(T[]){...}` / `(T[A][B]){...}` -- an anonymous local array of
 * `T`, initialized exactly like a braced local-array decl (a `= {0}` baseline + a c.store per element) and
 * yielded as the array-object rid. The classic use is a direct subscript `(int[]){...}[i]` -- an inline
 * lookup table. An inferred size `[]` takes its length from the initializer (the max index + 1; gaps
 * zero-fill). A MULTI-dim `(T[A][B]){...}` routes its nested ROW braces through subagg_init_md (the same
 * proven path the regular multi-dim local decl uses), riding a `= {0}` baseline -- so the flat resource is
 * row-major `[A*B]` and `[i][j]` Horner-flattens with the right inner stride. An inferred OUTER dim
 * `(T[][N]){...}` infers the row count from the number of top-level braces. The element store emits real
 * typed `_cl[i] = v`, so any scalar element type converts correctly; a struct element is a deferred
 * follow-on. */
static uint32_t p_array_literal(CC *c, const bcir_ctype *ty, int si, int count, const int *la_dims, int la_nd) {
  char nm[BCIR_CIR_NAME]; snprintf(nm,sizeof nm,"_cl%u",++c->cl_ctr);
  int es = ty->size ? ty->size : 4;
  if(ty->kind==1){                                   /* an AGGREGATE-element literal `(struct P[]){...}` /
    * `(struct P[N]){...}` (1-D) OR `(struct P[A][B]){...}` (multi-dim): a SCALAR-kind array of struct-sized
    * elements (es == the struct size, the per-element stride), carrying the struct tag so the decl emits
    * `struct P _cl[N]` and `_cl[i]...[k].field` strides by the element struct. Each innermost `{...}` element
    * routes through agg_init_at (per-element field stores at absolute offsets, riding a `= {0}` baseline) --
    * the C twin of the oracle's _init_subagg array-of-structs branch (and _array_row row descent for multi-dim).
    * An inferred `[]` patches its count + re-masks the init stores, exactly like the scalar path below. */
    if(la_nd>1){                                      /* a MULTI-dim AGGREGATE literal `(struct P[A][B]){...}` */
      int dims[3]; for(int z=0;z<3;z++) dims[z]=la_dims[z];
      int inner=1; for(int d=1; d<la_nd; d++) inner*=dims[d];   /* product of the FIXED inner dims */
      int outer = dims[0];                            /* `(struct P[][N]){...}` infers the outer dim from the init */
      if(outer<=0) outer = peek_top_entries(c);       /* the row-brace count (known up-front -> stores size right) */
      dims[0]=outer;
      int total = outer*inner;
      uint32_t rid = add_res(c, BCIR_DOM_RAM, es, total>0?total:1, 0, BCIR_RK_SCALAR, nm);
      int ari = (int)c->fn->n_res - 1;
      snprintf(c->fn->res[ari].agg,BCIR_CIR_NAME,"%s %s",ty->is_union?"union":"struct",ty->tag);
      c->fn->res[ari].zinit=1;                        /* a `= {0}` baseline -- unwritten fields/elements zero-fill */
      subagg_init_md_struct(c, rid, 0, dims, la_nd, es, si);   /* descend nested ROW braces; innermost = a struct */
      return rid;
    }
    uint32_t rid = add_res(c, BCIR_DOM_RAM, es, count>0?count:1, 0, BCIR_RK_SCALAR, nm);
    int ari = (int)c->fn->n_res - 1;
    snprintf(c->fn->res[ari].agg,BCIR_CIR_NAME,"%s %s",ty->is_union?"union":"struct",ty->tag);
    c->fn->res[ari].zinit=1;                          /* a `= {0}` baseline -- unwritten fields/elements zero-fill */
    size_t s_nclaims = c->fn->n_claims;               /* the init stores begin here -- re-mask after sizing */
    int nel = subagg_init_struct(c, rid, 0, si, es);  /* per-element struct store at `idx*es` */
    if(count<=0){ c->fn->res[ari].count = (uint32_t)(nel<1?1:nel);   /* an inferred `[]`: size from the init, then
      * re-evaluate access_bnd against the patched extent so the per-element stores mask exactly like a regular
      * array init (matching the oracle's per-element BCIR_CHK). */
      bcir_bounds bnd = access_bnd(c, rid);
      for(size_t i=s_nclaims; i<c->fn->n_claims; i++){ bcir_claim *cl=&c->fn->claims[i];
        if(cl->opcode==BCIR_OP_STORE && cl->n_rd>=1 && cl->rd[0]==rid) cl->bounds=bnd; } }
    return rid;
  }
  if(la_nd>1){                                       /* a MULTI-dim literal `(T[A][B]){...}` */
    int dims[3]; for(int z=0;z<3;z++) dims[z]=la_dims[z];
    int inner=1; for(int d=1; d<la_nd; d++) inner*=dims[d];   /* product of the FIXED inner dims */
    int outer = dims[0];                             /* `(T[][N]){...}` infers the outer dim from the init */
    if(outer<=0) outer = peek_top_entries(c);
    dims[0]=outer;
    int total = outer*inner;
    uint32_t rid = add_res(c, BCIR_DOM_RAM, es, total>0?total:1, 0, BCIR_RK_SCALAR, nm);
    int ari = (int)c->fn->n_res - 1;
    if(ty->is_float) c->fn->res[ari].is_float=1;     /* element type flags -> the decl emits `float`/`char`/... */
    else if(ty->kind==0){ c->fn->res[ari].is_signed=(uint8_t)(ty->signd?1:0);
      if(ty->is_bool) c->fn->res[ari].is_bool=1;
      if(ty->is_plain_char) c->fn->res[ari].is_plain_char=1; }
    c->fn->res[ari].zinit=1;                         /* a `= {0}` baseline -- unwritten elements zero-fill */
    subagg_init_md(c, rid, 0, dims, la_nd, es, ty->is_bool, ty);   /* descend the nested ROW braces */
    return rid;
  }
  uint32_t rid = add_res(c, BCIR_DOM_RAM, es, count>0?count:1, 0, BCIR_RK_SCALAR, nm);
  int ari = (int)c->fn->n_res - 1;                  /* the array resource (stable index; patch count below) */
  if(ty->is_float) c->fn->res[ari].is_float=1;      /* element type flags -> the decl emits `float`/`char`/... */
  else if(ty->kind==0){ c->fn->res[ari].is_signed=(uint8_t)(ty->signd?1:0);
    if(ty->is_bool) c->fn->res[ari].is_bool=1;
    if(ty->is_plain_char) c->fn->res[ari].is_plain_char=1; }
  size_t s_nclaims = c->fn->n_claims;                /* the init stores begin here -- re-mask after sizing */
  int nel = arr_init(c, rid);                        /* note: temp()/new_claim may realloc res[]; re-index ari */
  if(count<=0){ c->fn->res[ari].count = (uint32_t)(nel<1?1:nel);   /* an inferred `[]` size: the count is only
    * KNOWN now (the resource was created count=1), so the per-element c.store claims arr_init already emitted
    * saw count=1 and stayed `assumed`. Re-evaluate access_bnd against the patched extent so a known-extent
    * (count>1) `_clN` masks its init writes exactly like a regular array init -- matching the oracle's per-
    * element BCIR_CHK. (A sized `(T[N]){...}` already had the right count up front, so it is unaffected.) */
    bcir_bounds bnd = access_bnd(c, rid);
    for(size_t i=s_nclaims; i<c->fn->n_claims; i++){ bcir_claim *cl=&c->fn->claims[i];
      if(cl->opcode==BCIR_OP_STORE && cl->n_rd>=1 && cl->rd[0]==rid) cl->bounds=bnd; } }
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
    v=lookup(c,tat(c,c->i+1)); if(!v) return 0; ch=c->t[c->i].s[0]; c->i+=2;
  } else if(isk(c,T_ID) && tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==2 &&
            (tat(c,c->i+1)->s[0]=='+'||tat(c,c->i+1)->s[0]=='-') && tat(c,c->i+1)->s[1]==tat(c,c->i+1)->s[0]){
    v=lookup(c,pk(c)); if(!v) return 0; ch=tat(c,c->i+1)->s[0]; c->i+=2;   /* name++ / name-- */
  } else return 0;
  uint32_t one=temp(c,4); bcir_claim *kc=new_claim(c,"c.const",BCIR_OP_LOAD);
  if(kc){kc->n_wr=1;kc->wr[0]=one;kc->n_imm=1;kc->imm[0]=1;}
  if(v->type.kind==2){                                  /* pointer ++/-- : p += 1 / p -= 1 (verbatim) */
    char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.ptr%s",ch=='+'?"add":"sub");
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
      c->i+=2; uint32_t val=p_expr(c);
      bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD); if(cl){cl->n_rd=1;cl->rd[0]=val;cl->n_wr=1;cl->wr[0]=v->rid;}
      mark_obj_write(c,cl,v);
      return; }
    if(v && is_compound_op(tat(c,c->i+1))){                                             /* name OP= expr */
      char ch=tat(c,c->i+1)->s[0];
      if(v->type.kind==2 && (ch=='+'||ch=='-')){       /* pointer arithmetic: p += n / p -= n (verbatim) */
        c->i+=2; uint32_t rhs=p_expr(c);
        char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.ptr%s",ch=='+'?"add":"sub");
        bcir_claim *cl=new_claim(c,op,BCIR_OP_ADD); if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=rhs;cl->n_wr=1;cl->wr[0]=v->rid;}
        return; }
      uint32_t cur=named_read(c,v);                     /* the current value (a volatile read, first) */
      c->i+=2; uint32_t rhs=p_expr(c);                  /* scalar:  name = name OP expr  (bin op + copy) */
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
      uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.bin.%s",suf);
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
    venv b; memset(&b,0,sizeof b); b.rid=ptr; b.sidx=-1;
    b.type.size=pfld.ptee_size?pfld.ptee_size:4; b.type.signd=pfld.signd; b.type.is_float=(uint8_t)pfld.ptee_float;
    b.type.is_volatile=(uint8_t)(pfld.ptee_volatile?1:0);
    uint32_t val;
    if(is_compound_op(&c->t[c->i])){ char ch=c->t[c->i].s[0]; c->i++;
      uint32_t cur=emit_index(c,&b,idx); uint32_t rhs=p_expr(c);
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
      uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.bin.%s",suf);
      bcir_claim *bb=new_claim(c,op,oc); if(bb){bb->n_rd=2;bb->rd[0]=cur;bb->rd[1]=rhs;bb->n_wr=1;bb->wr[0]=tmp;}
      val=tmp;
    } else { if(!eat(c,"="))return; val=p_expr(c); }
    bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
    if(cl){cl->n_rd=3;cl->rd[0]=ptr;cl->rd[1]=idx;cl->rd[2]=val;cl->bounds=BCIR_BND_ASSUMED;
      mark_access(c,cl,pfld.ptee_volatile);}
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
  if(is_compound_op(&c->t[c->i])){ char ch=c->t[c->i].s[0]; c->i++;
    uint32_t cur=emit_member(c,&b,&f,0); uint32_t rhs=p_expr(c);
    const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
    uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.bin.%s",suf);
    bcir_claim *bb=new_claim(c,op,oc); if(bb){bb->n_rd=2;bb->rd[0]=cur;bb->rd[1]=rhs;bb->n_wr=1;bb->wr[0]=tmp;}
    val=tmp;
  } else { if(!eat(c,"="))return; val=p_expr(c); }
  /* the member store every other path takes: C's conversion, a `_Bool` member's flag, a bitfield's unit */
  if(f.bit_w) store_member_bf(c,&b,&f,val); else store_member(c,&b,&f,val);
}
static void p_stmt_inner(CC *c);
/* Depth-guarded wrapper: p_stmt is a recursive-cycle entry (p_stmt->p_block->p_stmt, and the stmt-expr
 * `({...})` path p_stmt_expr->p_stmt). Bump/check depth once per statement nesting level. */
static void p_stmt(CC *c) {
  if(ENTER_REC(c)){ LEAVE_REC(c); return; }
  p_stmt_inner(c); LEAVE_REC(c);
}
static void p_stmt_inner(CC *c) {
  if(is(c,";")){c->i++;return;}          /* empty statement -> a no-op (`for(...);`, `if(c);`, `;;`) */
  if(is(c,"return")){c->i++;
    if(!is(c,";")){uint32_t rv=p_expr(c);c->fn->return_rid=rv;c->fn->has_return=1;marker(c,"c.return",rv,1);}
    else marker(c,"c.return",0,0);
    eat(c,";");return;}
  if(is(c,"if")){                      /* L6: if / else -> structured markers */
    c->i++;eat(c,"(");uint32_t cond=p_expr(c);eat(c,")");
    marker(c,"c.if",cond,1); p_block(c);
    if(is(c,"else")){c->i++;marker(c,"c.else",0,0);p_block(c);}
    marker(c,"c.endif",0,0); return;
  }
  if(is(c,"while")){                   /* L6: a bounded while loop (cond re-evaluated each iter) */
    c->i++;marker(c,"c.loop",0,0);
    eat(c,"(");uint32_t cond=p_expr(c);eat(c,")");
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
    char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.goto:%.*s",lb.n,lb.s); new_claim(c,op,BCIR_OP_NOP); return; }
  if(isk(c,T_ID)&&tat(c,c->i+1)->k==T_PUN&&tat(c,c->i+1)->n==1&&tat(c,c->i+1)->s[0]==':'){  /* `name:` -- a label */
    tok lb=adv(c); c->i++; char op[BCIR_CIR_NAME];
    snprintf(op,sizeof op,"c.label:%.*s",lb.n,lb.s); new_claim(c,op,BCIR_OP_NOP); return; }
  if(is(c,"switch")){                  /* a real C switch: case labels + fallthrough preserved */
    c->i++; eat(c,"(");
    uint32_t disc=p_expr(c);           /* the discriminant, lowered once */
    eat(c,")"); eat(c,"{");
    marker(c,"c.switch",disc,1);
    while(!is(c,"}")&&!isk(c,T_END)&&!c->failed){
      if(is(c,"case")){ c->i++; long long v=ce_expr(c,0); eat(c,":");   /* case <const>: */
        char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.case:%lld",v); marker(c,op,0,0); }
      else if(is(c,"default")){ c->i++; eat(c,":"); marker(c,"c.default",0,0); }
      else p_stmt(c);                  /* body statements (break -> c.break, no implicit break) */
    }
    eat(c,"}");
    marker(c,"c.endswitch",0,0);
    return;
  }
  if(is(c,"{")){p_block(c);return;}
  int looks_decl=0, is_static=is(c,"static");
  if(isk(c,T_ID)){int sz=scalar_size(pk(c)->s,pk(c)->n);
    looks_decl=sz>=0||is_static||is(c,"struct")||is(c,"union")||is(c,"enum")||is(c,"const")||is(c,"volatile")
               ||is(c,"_Complex")||is(c,"complex")                      /* `double _Complex z;` -- a complex local */
               ||is(c,"_BitInt")                                        /* `_BitInt(N) z;` -- a bit-precise local */
               ||is(c,"_Atomic")                                        /* `_Atomic int a;` -- an atomic local */
               ||is(c,"va_list")||is(c,"__builtin_va_list")              /* `va_list ap;` -- a variadic cursor local */
               ||is(c,"typeof")||is(c,"__typeof__")||is(c,"typeof_unqual")
               ||find_typedef(c,pk(c)->s,pk(c)->n)>=0;}
  if(looks_decl){
    bcir_ctype base;int si;if(p_type_base(c,&base,&si))return;   /* the shared specifier (eats `static`, NOT `*`) */
    /* one or more comma-separated declarators sharing this specifier: `T a = x, b, c = z;`. Each gets
     * its OWN declarator `*`/`[]` shape off a fresh copy of the base, so `int *p, q;` types p as `int*`
     * and q as int (per-declarator, matching the oracle), `int *p, *q;` types both as pointers, and a
     * per-declarator array (`int a[2], b;`) no longer leaks its dims onto the next declarator. */
    for(;;){
      bcir_ctype ty=base; apply_stars(c,&ty);   /* this declarator's own leading `*`s (none -> base type) */
      if(is(c,"(") && tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='*'
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
        char rets[64]; ctype_str(&ret,rets,sizeof rets);
        char sig[512]; size_t sw=0; int np=0;
        /* `sw` accumulates snprintf's RETURN (the would-be length), so on a long signature it can
         * reach/exceed `sizeof sig` -- `sizeof sig - sw` would then underflow size_t to a huge value and
         * the next snprintf would write past sig[512] (a stack-buffer-overflow). CLAMP the size argument
         * to 0 once the buffer is full: snprintf then writes NOTHING but still returns the would-be
         * length, so `sw` tracks the true total for the `fpdefs_w+sw < sizeof fpdefs` guard below (which
         * already drops an over-long prelude) -- the parse CONTINUES and the kind-3 binding + claim graph
         * are byte-identical to the oracle (which has no fixed buffer). The signature text is cosmetic
         * prelude only; this is a memory-safety clamp, NOT a behaviour change. */
        /* Clamp the OFFSET, not just the size: forming `sig+sw` with sw>sizeof sig is itself
         * out-of-bounds-pointer UB (clang-UBSan flags it even when snprintf's size is 0). `sig+SIG_OFF`
         * is at most one-past-the-end (a legal pointer) and `sizeof sig - SIG_OFF` is the remaining space
         * (0 once full); `sw` still accumulates the true would-be length for the fpdefs guard. */
        #define SIG_OFF (sw<sizeof sig?sw:sizeof sig)
        sw+=snprintf(sig+SIG_OFF,sizeof sig-SIG_OFF,"typedef %s (*__bcir_fp%d)(",rets,c->n_fpdef);
        if(is(c,"void")&&tat(c,c->i+1)->n==1&&tat(c,c->i+1)->s[0]==')'){ c->i++; }   /* `(void)` */
        else if(!is(c,")")) for(;;){ bcir_ctype pt; int psi; if(p_type(c,&pt,&psi))return;
          if(isk(c,T_ID)) c->i++;                      /* an optional parameter name (ignored) */
          char ps[64]; ctype_str(&pt,ps,sizeof ps);
          sw+=snprintf(sig+SIG_OFF,sizeof sig-SIG_OFF,"%s%s",np?", ":"",ps); np++;
          if(is(c,",")){c->i++;continue;} break; }
        sw+=snprintf(sig+SIG_OFF,sizeof sig-SIG_OFF,"%s);\n",np?"":"void");
        #undef SIG_OFF
        /* only copy a signature that fit in `sig` (sw<=sizeof sig): a clamped (over-long) one was
         * truncated, so its `sw` overstates the bytes actually in `sig` -- never read past sig[512]. */
        if(sw<sizeof sig && sw<sizeof c->fpdefs-c->fpdefs_w){ memcpy(c->fpdefs+c->fpdefs_w,sig,sw); c->fpdefs_w+=sw; }
        else c->emit_overflow=1;
        if(!eat(c,")"))return;                          /* past the parameter-type list */
        bcir_ctype fty; memset(&fty,0,sizeof fty); fty.kind=3; fty.size=8; fty.signd=0;
        fty.fp_ret_size=ret.size; fty.fp_ret_signd=(uint8_t)(ret.signd?1:0); fty.fp_ret_float=(uint8_t)(ret.is_float?1:0);
        snprintf(fty.tag,sizeof fty.tag,"__bcir_fp%d",c->n_fpdef); c->n_fpdef++;
        char fnb[BCIR_CIR_NAME]; idcpy(fnb,&nm);
        uint32_t frid=add_res(c,BCIR_DOM_RAM,8,1,0,BCIR_RK_SCALAR,fnb);   /* a funcptr-wide scalar local */
        if(c->fn->n_res){ bcir_resource *fr=&c->fn->res[c->fn->n_res-1];
          fr->is_funcptr=1; snprintf(fr->agg,BCIR_CIR_NAME,"%s",fty.tag); }   /* emit `__bcir_fpN f;` up front */
        env_add(c,&nm,frid,&fty,-1);                    /* p_icall finds kind-3 -> a c.call.indirect dispatch */
        if(is(c,"=")){ c->i++;                          /* an init: a function name -> a funcptr value (c.copy) */
          uint32_t v=p_expr(c);
          bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD); if(cl){cl->n_rd=1;cl->rd[0]=v;cl->n_wr=1;cl->wr[0]=frid;} }
        if(is(c,",")){ c->i++; continue; }              /* another declarator off the same specifier */
        break;
      }
      if(!isk(c,T_ID)){ fail(c,"expected declarator name"); return; }
      tok nm=adv(c); char nb[BCIR_CIR_NAME]; idcpy(nb,&nm);
      int arr=0,la_nd=0,la_dims[3]={0,0,0};            /* T name[N] / T m[A][B] -- a (multi-dim) local array */
      /* scan ALL `[...]` dims WITHOUT lowering, recording each as either a literal value or a runtime-expr
       * token range; then classify (oracle order): all-literal -> a static array (unchanged); >=1 runtime
       * dim with ONE dim -> the 1-D VLA path (byte-identical); >=2 dims with a runtime dim -> a multi-dim
       * VLA. Deferring the lowering keeps the dim snapshots in canonical dim order (each dim is evaluated
       * then snapshotted, in turn) -- the byte-parity contract with the oracle's Decl branch. */
      int dim_nd=0; int dim_is_lit[8]; int dim_lit[8]; int dim_tok[8]; int any_vla=0;
      while(is(c,"[")){ c->i++;
        if(!isk(c,T_INT) && !is(c,"]")){               /* a non-constant dim -> a runtime VLA dim */
          if(dim_nd<8){ dim_is_lit[dim_nd]=0; dim_tok[dim_nd]=c->i; } any_vla=1;
          int paren=0; while(!(paren==0 && is(c,"]")) && !isk(c,T_END)){   /* skip to the matching `]` */
            if(is(c,"[")||is(c,"(")) paren++; else if(is(c,")")) paren--; c->i++; }
          eat(c,"]"); }
        else { int dim=isk(c,T_INT)?(int)adv(c).v:0; eat(c,"]");
          if(dim_nd<8){ dim_is_lit[dim_nd]=1; dim_lit[dim_nd]=dim; }
          if(la_nd<3)la_dims[la_nd]=dim; la_nd++; arr = arr?arr*dim:dim; }
        dim_nd++; }
      int after_dims=c->i;                             /* the cursor past the last `]` (restored after re-parse) */
      if(any_vla){
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
        uint32_t arid=add_res(c,BCIR_DOM_RAM,ty.size,1,0,BCIR_RK_SCALAR,nb);   /* the array (element type, count 0->1) */
        { bcir_resource *ar=&c->fn->res[c->fn->n_res-1];
          ar->is_signed=(uint8_t)(ty.signd?1:0); ar->is_vla=1; ar->ext_var=ext;
          if(ty.is_bool) ar->is_bool=1; if(ty.is_plain_char) ar->is_plain_char=1; }
        { bcir_claim *vd=new_claim(c,"c.vladecl",BCIR_OP_ADD); if(vd){vd->n_rd=1;vd->rd[0]=ext;vd->n_wr=1;vd->wr[0]=arid;} }
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
        uint32_t arid=add_res(c,BCIR_DOM_RAM,nbytes,1,0,BCIR_RK_SCALAR,nb);   /* 3. a FLAT runtime-extent array */
        { bcir_resource *ar=&c->fn->res[c->fn->n_res-1];
          ar->is_signed=(uint8_t)(nsgn?1:0); ar->is_vla=1; ar->ext_var=ext_total;
          ar->vla_ndims=(uint8_t)dim_nd; for(int d=0;d<dim_nd;d++) ar->vla_strides[d]=dim_exts[d];
          if(ty.is_bool) ar->is_bool=1; if(ty.is_plain_char) ar->is_plain_char=1; }
        { bcir_claim *vd=new_claim(c,"c.vladecl",BCIR_OP_ADD); if(vd){vd->n_rd=1;vd->rd[0]=ext_total;vd->n_wr=1;vd->wr[0]=arid;} }
        env_add(c,&nm,arid,&ty,si);   /* the venv type is the element type -- `a[i][j]` indexes via emit_index */
        ptrext_set(c,c->fn,arid,ext_total);   /* §5.12 mask `a[i][j]` against the recovered total runtime extent */
        if(is(c,",")){ c->i++; continue; }
        break;
      }
      if(la_nd>3){ fail(c,"local array of more than 3 dimensions"); return; }   /* adims caps at 3 */
      if(la_nd>1){ for(int z=0;z<3;z++) ty.adims[z]=la_dims[z]; ty.nadims=la_nd; }   /* multi-dim flatten */
      int inferred = (la_nd>=1 && arr==0);   /* an inferred-size `[]` array (all dims 0): the count comes
        * from the initializer (patched after the init, like p_array_literal). Created count=1 for now. */
      int is_arr = (arr || inferred);         /* a (possibly inferred-size) ARRAY declarator -> a SCALAR-kind
        * array resource (count>1; a struct-element array carries the struct tag in `agg` for the decl) */
      int rk=is_arr?BCIR_RK_SCALAR:(ty.kind==2?BCIR_RK_POINTER:ty.kind==1?BCIR_RK_AGGREGATE:BCIR_RK_SCALAR);
      int arr_elem = (is_arr && ty.kind==2) ? cc_abi(c)->pointer_size : ty.size;   /* an array of pointers: pointer-wide elements */
      uint32_t rid=add_res(c, ty_mmio(c,&ty,si,is_arr)?BCIR_DOM_MMIO:BCIR_DOM_RAM,
                           is_arr?arr_elem:(ty.kind==2?ty.size:(ty.kind==1?c->s[si].size:ty.size)),
                           is_arr?(arr?arr:1):(ty.kind==2?(1<<16):1), ty.is_volatile, rk, nb);
      if(is_arr && c->fn->n_res) c->fn->res[c->fn->n_res-1].is_array=1;   /* an array object, whatever its length */
      if(is_arr && ty.kind==1){ bcir_resource *ar=&c->fn->res[c->fn->n_res-1];   /* an ARRAY-OF-STRUCTS local
        * `struct P a[N]`: a SCALAR-kind array of struct-sized elements; carry the struct tag so the decl
        * emits `struct P a[N]` and `a[i].field` strides by the element struct (the venv keeps `si`). */
        snprintf(ar->agg,BCIR_CIR_NAME,"%s %s",ty.is_union?"union":"struct",ty.tag); }
      else if(arr && ty.kind==2){ bcir_resource *ar=&c->fn->res[c->fn->n_res-1];   /* an array of pointers `T *a[N]`: a
        * SCALAR array of pointer-wide elements; the decl + `a[i]` load/store carry the pointee (void* for now) */
        ar->ptr_depth=ty.ptr_depth?ty.ptr_depth:1;
        ar->ptee_bytes=(uint32_t)(ty.size>0?ty.size:0); ar->ptee_signed=(uint8_t)(ty.signd?1:0);   /* the decl's `T` */
        ar->ptee_float=(uint8_t)(ty.is_float?1:0); ar->ptee_plain_char=(uint8_t)(ty.is_plain_char?1:0);
        if(ty.ptr_to_struct) snprintf(ar->agg,BCIR_CIR_NAME,"%s %s",ty.is_union?"union":"struct",ty.tag);
        else if(ty.size==0 && !ty.is_float) ar->is_voidptr=1; }
      else if(ty.kind==2&&!arr){ bcir_resource *pr=&c->fn->res[c->fn->n_res-1];   /* a pointer local: carry the
        * pointee type (elem_bytes already = pointee size) so the decl emits `T *p`, not a truncating uint32 */
        pr->is_signed=(uint8_t)(ty.signd?1:0); pr->is_float=(uint8_t)(ty.is_float?1:0); pr->ptr_depth=ty.ptr_depth;
        pr->is_plain_char=(uint8_t)(ty.is_plain_char?1:0);   /* a `char *` pointee: the deref load emits `char` */
        if(ty.ptr_to_struct) snprintf(pr->agg,BCIR_CIR_NAME,"%s %s",ty.is_union?"union":"struct",ty.tag);
        else if(ty.size==0 && !ty.is_float) pr->is_voidptr=1; }   /* a `void *` local (void pointee) -> emit `void *` */
      else if(ty.is_valist) c->fn->res[c->fn->n_res-1].is_valist=1;     /* a `va_list ap;` local -> emit `va_list` */
      else if(ty.is_float){ c->fn->res[c->fn->n_res-1].is_float=1;      /* a float/double (element) local */
        if(ty.is_complex) c->fn->res[c->fn->n_res-1].is_complex=1; }    /* a _Complex local (a float pair) */
      else if(ty.kind==0){ c->fn->res[c->fn->n_res-1].is_signed=(uint8_t)(ty.signd?1:0);  /* (element) signedness */
        if(ty.is_bool) c->fn->res[c->fn->n_res-1].is_bool=1;       /* a _Bool local: emit `_Bool`, store normalizes */
        if(ty.bit_width>0) c->fn->res[c->fn->n_res-1].bit_width=ty.bit_width;   /* a C23 `_BitInt(N)` local */
        if(ty.is_plain_char) c->fn->res[c->fn->n_res-1].is_plain_char=1; }   /* a plain `char` local: emit `char` */
      if(ty.kind==1&&!is_arr) snprintf(c->fn->res[c->fn->n_res-1].agg,BCIR_CIR_NAME,"%s %s",ty.is_union?"union":"struct",ty.tag);   /* L8 aggregate local (a NON-array struct/union; the array form set its agg above) */
      env_add(c,&nm,rid,&ty,si);   /* the venv type is the element type -- `a[i]` indexes via emit_index */
      if(is_static){            /* static storage: a once-only constant init, baked into the decl */
        long long init=0; if(is(c,"=")){c->i++;init=ce_expr(c,0);}
        { bcir_func *f=c->fn;
          CC_ENSURE(c,f->statics,f->n_statics,f->cap_statics);
          if(f->n_statics<f->cap_statics){ idcpy(f->statics[f->n_statics].name,&nm);
            f->statics[f->n_statics].init=init; f->statics[f->n_statics].rid=rid; f->n_statics++; } }
      } else if(is(c,"=")){c->i++;
        if(is(c,"{")){
          int md_nested=0;                                  /* peek: does a MULTI-dim init use a nested ROW brace? */
          if(arr && la_nd>1){ int j=c->i+1;                 /* (skip an optional leading `[const]=` designator) */
            if(tok_is(&c->t[j],"[")){
              int p=0; while(!(p==0 && tok_is(&c->t[j],"]")) && c->t[j].k!=T_END){
                if(tok_is(&c->t[j],"[")) p++; else if(tok_is(&c->t[j],"]")) p--; j++; }
              j++; if(tok_is(&c->t[j],"=")) j++; }
            md_nested = tok_is(&c->t[j],"{"); }
          if(md_nested){                                    /* a MULTI-dim array `T a[d0][d1] = {{..},{..}}`: the
            * flat resource is row-major, so descend each outer brace by ROW (the C twin of the oracle's
            * _agg_init multi-dim row descent). A `= {0}` baseline zero-fills any unwritten element. A FLAT
            * `{e0,e1,..}` multi-dim init (no nested braces) keeps the idx-based arr_init path -- the oracle
            * also flattens that form positionally, so the rails stay byte-identical. */
            for(size_t i=0;i<c->fn->n_res;i++) if(c->fn->res[i].rid==rid){ c->fn->res[i].zinit=1; break; }
            subagg_init_md(c, rid, 0, la_dims, la_nd, ty.size, ty.is_bool, &ty);
          }
          else if(is_arr){                                  /* a 1-D local array init (the C twin of the oracle's
            * array _agg_init). A struct/union ELEMENT array routes each `{...}` element to subagg_init_struct
            * (per-element offset stores at `idx*stride`, riding a `= {0}` baseline) -- byte-identical to the
            * oracle. An INFERRED-size `[]` array (scalar OR struct) infers its count from the initializer and
            * patches the resource extent + re-masks the init stores, exactly as p_array_literal does. */
            int ari=-1; for(size_t i=0;i<c->fn->n_res;i++) if(c->fn->res[i].rid==rid){ ari=(int)i; break; }
            size_t s_nclaims=c->fn->n_claims;               /* the init stores begin here -- re-mask after sizing */
            int nel;
            if(ty.kind==1){                                 /* an ARRAY-OF-STRUCTS local: `{ {a,b}, {c,d}, .. }` */
              if(ari>=0) c->fn->res[ari].zinit=1;           /* the `= {0}` baseline (composes the per-element stores) */
              nel = subagg_init_struct(c, rid, 0, si, ty.size);   /* per-element struct store at `idx*ty.size` */
            } else nel = arr_init(c,rid);                   /* a scalar-element 1-D array */
            if(inferred && ari>=0){ c->fn->res[ari].count=(uint32_t)(nel<1?1:nel);   /* size the inferred `[]` */
              bcir_bounds bnd=access_bnd(c,rid);            /* re-mask the per-element stores vs the patched extent */
              for(size_t i=s_nclaims; i<c->fn->n_claims; i++){ bcir_claim *cl=&c->fn->claims[i];
                if(cl->opcode==BCIR_OP_STORE && cl->n_rd>=1 && cl->rd[0]==rid) cl->bounds=bnd; } }
          } else agg_init(c,rid,si); }   /* {…} struct-union init */
        else { int ist=c->i; uint32_t v=p_expr(c); int ien=c->i;
          bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD);if(cl){cl->n_rd=1;cl->rd[0]=v;cl->n_wr=1;cl->wr[0]=rid;}
          { venv *dv=lookup(c,&nm); if(dv) mark_obj_write(c,cl,dv); }   /* `volatile T x = v;` */
          bind_extent(c,rid,res_of(c->fn,rid),&nm,ist,ien); } }   /* §5.12: `T *p = malloc(N*sizeof(T))` -> extent N */
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
    venv pvsnap; venv *pv=NULL; uint32_t idx=0; int has_idx=0, ok=0;
    /* SNAPSHOT the pointer's env entry: the `*(p + i)` index and the RHS p_expr below can declare locals
     * and realloc c->env[] -- a pointer into it dangles. The store/emit helpers only READ the venv. */
    size_t s_res=c->fn->n_res,s_cl=c->fn->n_claims; uint32_t s_rid=c->rid,s_cid=c->cid,s_clc=c->cl_ctr;
    if(is(c,"(")){ c->i++;                              /* *(p) or *(p + i) */
      if(isk(c,T_ID)){ tok pid=*pk(c); venv *pvp=lookup(c,&pid); if(!pvp) pvp=use_global(c,&pid);
        if(pvp){ c->i++; pvsnap=*pvp; pv=&pvsnap;
          if(is(c,"[")){ uint32_t lin=index_chain(c,pv);   /* `*(q[j] ...)`: through the loaded pointer element */
            if(c->failed) return;
            if(!step_to_elem_ptr(c,pv,lin)) pv=NULL; }
          if(pv && is(c,"+")){ c->i++; idx=p_expr(c); has_idx=1; if(eat(c,")")) ok=1; }
          else if(pv && is(c,")")){ c->i++; ok=1; } } } }
    else if(isk(c,T_ID)){ tok pid=*pk(c); venv *pvp=lookup(c,&pid); if(!pvp) pvp=use_global(c,&pid);   /* *p, *g */
      if(pvp){ c->i++; pvsnap=*pvp; pv=&pvsnap; ok=1; } }
    if(ok && pv && (is_compound_op(&c->t[c->i]) ||
                    (c->t[c->i].k==T_PUN && c->t[c->i].n==1 && c->t[c->i].s[0]=='='))){
      int sz = (pv->type.ptr_depth>1) ? cc_abi(c)->pointer_size : (pv->type.size?pv->type.size:4); uint32_t val;
      /* `*pp = q` through a `T**` stores a full pointer (pointer_size), not the base scalar width */
      bcir_ctype pst=pointee_slot(&pv->type);           /* `*p` stores a byte copy: C's conversion first */
      if(is_compound_op(&c->t[c->i])){                  /* *p OP= expr  ->  load, bin op, store */
        char ch=c->t[c->i].s[0]; c->i++;
        uint32_t cur = has_idx ? emit_index(c,pv,idx) : emit_deref(c,pv);
        uint32_t rhs=p_expr(c);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
        uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.bin.%s",suf);
        bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
        val=tmp;
      } else { c->i++; val=p_expr(c); }                 /* *p = expr */
      if(!has_idx) val=store_conv(c,val,&pst);
      bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
      if(cl){
        if(has_idx){ cl->n_rd=3; cl->rd[0]=pv->rid; cl->rd[1]=idx; cl->rd[2]=val; }   /* *(p+i) == p[i] */
        else { cl->n_rd=2; cl->rd[0]=pv->rid; cl->rd[1]=val; cl->n_imm=2; cl->imm[0]=0; cl->imm[1]=sz; }
        cl->bounds=BCIR_BND_ASSUMED;
        mark_access(c,cl,index_elem_vol(c,pv));          /* `*p` / `*(p+i)`: the pointee */
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
      if(is_compound_op(&c->t[c->i])){                 /* **pp OP= expr -> load through base, bin, store */
        char ch=c->t[c->i].s[0]; c->i++;
        uint32_t cur=emit_deref_rid(c,base); uint32_t rhs=p_expr(c);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
        uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.bin.%s",suf);
        bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
        val=tmp;
      } else { c->i++; val=p_expr(c); }                /* **pp = expr */
      val=store_conv(c,val,&pst);                      /* C's conversion to the pointee's type */
      bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
      if(cl){ cl->n_rd=2; cl->rd[0]=base; cl->rd[1]=val; cl->n_imm=2; cl->imm[0]=0; cl->imm[1]=sz; cl->bounds=BCIR_BND_ASSUMED;
        const bcir_resource *bb=res_of(c->fn,base); mark_access(c,cl,depth==1 && bb && bb->is_volatile); }
      eat(c,";"); return;
    }
    c->i=save;   /* not a deref-store -- fall through (e.g. a bare `*p;` expression statement) */
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
    if(v&&v->sidx>=0&&tat(c,c->i+1)->k==T_PUN&&(tat(c,c->i+1)->s[0]=='.'||(tat(c,c->i+1)->n==2&&tat(c,c->i+1)->s[0]=='-'))
        && member_is_store(c,c->i+1)){
      c->i+=2; tok fld=adv(c); sdef *S=&c->s[v->sidx]; int fi=-1;
      for(int k=0;k<S->nf;k++) if((int)strlen(S->f[k].name)==fld.n&&!strncmp(S->f[k].name,fld.s,fld.n)) fi=k;
      if(fi<0){fail(c,"unknown field");return;}
      field f=member_descend(c,S->f[fi]);         /* nested `o.in.v` -> one flattened-offset store */
      if(f.is_ptr && (is(c,"->")||is(c,".")||is(c,"["))){   /* store deref-through a loaded pointer field (#fieldderef) */
        uint32_t ptr=emit_member(c,v,&f,0);       /* load the pointer field, then store through the loaded ptr */
        store_through_ptr(c,ptr,f.ptee_sidx,f); eat(c,";"); return; }
      if(f.arr_count && is(c,"[")){               /* s.arr[i] / s.m[i][j] = expr  /  OP= expr */
        uint32_t idx=member_arr_index(c,&f); uint32_t aval;
        field sub; int soa=elem_field(c,&f,&sub);   /* arr[i].field on an array-of-structs (strided store) */
        if(c->failed) return;
        const field *sf = soa ? &sub : &f;          /* the stored slot: the element FIELD, or the array element */
        if(is_compound_op(&c->t[c->i])){          /* load element, bin op, store back */
          char ch=c->t[c->i].s[0]; c->i++;
          uint32_t cur=soa?emit_member_index_field(c,v,&f,idx,&sub):emit_member_index(c,v,&f,idx);
          uint32_t rhs=p_expr(c);
          const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
          uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.bin.%s",suf);
          bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
          aval=tmp;
        } else { if(!eat(c,"="))return; aval=p_expr(c); }
        store_member_index(c,v,&f,idx,soa,sf,aval);
        eat(c,";");return;
      }
      uint32_t val;
      if(is_compound_op(&c->t[c->i])){
        /* compound assignment to a member:  r->field OP= expr  (the set/clear-bits driver idiom; a
         * bitfield field reads via c.bf.get, a plain member via a plain load). */
        char ch=c->t[c->i].s[0]; c->i++;
        uint32_t cur=emit_member(c,v,&f,0);        /* the current field value (loaded first) */
        uint32_t rhs=p_expr(c);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
        uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.bin.%s",suf);
        bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
        val=tmp;
      } else { if(!eat(c,"="))return; val=p_expr(c); }
      /* a bitfield: read the storage unit, insert the masked bits (c.bf.set), store the unit's spanned bytes
       * back; a `_Bool` member normalizes; either converts the value to the member's type first */
      if(f.bit_w) store_member_bf(c,v,&f,val); else store_member(c,v,&f,val);
      eat(c,";");return;}
    /* L3: array element store  a[idx] = expr  /  a[idx] OP= expr  (driver buffer fill / scatter). */
    if(v&&tat(c,c->i+1)->k==T_PUN&&tat(c,c->i+1)->n==1&&tat(c,c->i+1)->s[0]=='['){
      int as_start=c->i;                                /* roll-back point: `a[i]` may be a VALUE, not a store */
      size_t as_res=c->fn->n_res,as_cl=c->fn->n_claims; uint32_t as_rid=c->rid,as_cid=c->cid,as_clc=c->cl_ctr;
      c->i++; uint32_t idx=index_chain(c,v); uint32_t val;   /* a[i] / m[i][j] (Horner) / q[j][i] (a chain) */
      if(c->failed) return;
      if(v->sidx>=0 && (is(c,".")||is(c,"->"))){        /* a[i].field on a DIRECT array-of-structs (strided store) */
        field sub; if(!aos_elem_field(c,v,&sub)){ if(c->failed) return;
          c->fn->n_res=as_res;c->fn->n_claims=as_cl;c->rid=as_rid;c->cid=as_cid;c->cl_ctr=as_clc;
          c->i=as_start;(void)p_expr(c);eat(c,";");return; }
        uint32_t aval;
        if(is_compound_op(&c->t[c->i])){               /* a[i].f OP= expr -> strided load, bin op, strided store */
          char ch=c->t[c->i].s[0]; c->i++;
          uint32_t cur=emit_index_field(c,v,idx,&sub); uint32_t rhs=p_expr(c);
          const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
          uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.bin.%s",suf);
          bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
          aval=tmp;
        } else { if(!eat(c,"="))return; aval=p_expr(c); }
        store_index_field(c,v,idx,&sub,aval); eat(c,";"); return;
      }
      if(is_compound_op(&c->t[c->i])){
        char ch=c->t[c->i].s[0]; c->i++;                /* a[idx] OP= expr -> load, op, store */
        uint32_t cur=emit_index(c,v,idx); uint32_t rhs=p_expr(c);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
        uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.bin.%s",suf);
        bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
        val=tmp;
      } else if(c->t[c->i].k==T_PUN&&c->t[c->i].n==1&&c->t[c->i].s[0]=='='){ c->i++; val=p_expr(c); }
      else {        /* `a[i]` with no `=`/OP= is a VALUE (e.g. the last item of a `({...})`), not a store: undo
                     * the speculative index lowering and re-parse the whole thing as an expression statement. */
        c->fn->n_res=as_res; c->fn->n_claims=as_cl; c->rid=as_rid; c->cid=as_cid; c->cl_ctr=as_clc;
        c->i=as_start; (void)p_expr(c); eat(c,";"); return; }
      bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
      if(cl){cl->n_rd=3;cl->rd[0]=v->rid;cl->rd[1]=idx;cl->rd[2]=val;cl->bounds=access_bnd(c,v->rid);  /* §5.12 promote */
        mark_access(c,cl,index_elem_vol(c,v));}
      eat(c,";");return;}
    if(v&&tat(c,c->i+1)->k==T_PUN&&tat(c,c->i+1)->n==1&&tat(c,c->i+1)->s[0]=='='){
      tok tnm=c->t[c->i]; c->i+=2; int ist=c->i; uint32_t val=p_expr(c); int ien=c->i;
      bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD);if(cl){cl->n_rd=1;cl->rd[0]=val;cl->n_wr=1;cl->wr[0]=v->rid;}
      mark_obj_write(c,cl,v);
      bind_extent(c,v->rid,res_of(c->fn,v->rid),&tnm,ist,ien);   /* §5.12: `p = malloc(N*…)` -> N */
      eat(c,";");return;}
    /* compound assignment  name OP= expr  ->  name = name OP expr  (a bin op + a copy). */
    if(v&&is_compound_op(tat(c,c->i+1))){
      char ch=tat(c,c->i+1)->s[0];
      if(v->type.kind==2 && (ch=='+'||ch=='-')){       /* pointer arithmetic: p += n / p -= n (verbatim) */
        c->i+=2; uint32_t rhs=p_expr(c);
        char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.ptr%s",ch=='+'?"add":"sub");
        bcir_claim *cl=new_claim(c,op,BCIR_OP_ADD); if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=rhs;cl->n_wr=1;cl->wr[0]=v->rid;}
        eat(c,";");return;
      }
      uint32_t cur=named_read(c,v);                     /* the current value (a volatile read, first) */
      c->i+=2; uint32_t rhs=p_expr(c);
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
      uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.bin.%s",suf);
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
  if(!is_value){ result=temp(c,4); }      /* a void / empty statement expression: the last stmt (if any) is
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

static int p_func(CC *c, bcir_func *fn) {
  c->fn=fn; c->nenv=0; c->n_vlaext=0;
  c->cl_ctr=0;                                     /* compound literals number per function (`_cl1` ...), as
                                                    * the oracle's _FuncLowerer.cl_ctr does -- they are
                                                    * function-local declarations, so the names never clash */
  c->saw_static=0;                                 /* fresh for THIS definition's return type (a prior
                                                    * body's block-static must not leak into the flag) */
  bcir_ctype rt;int rsi;if(p_type(c,&rt,&rsi))return 1; fn->ret=rt;
  fn->static_fn=(uint8_t)(c->saw_static!=0);       /* source `static` on the definition (linkable emit) */
  tok nm=adv(c); snprintf(fn->name,sizeof fn->name,"%.*s",nm.n,nm.s);
  if(!eat(c,"("))return 1;
  if(!is(c,")")) for(;;){
    if(is(c,"void")&&tat(c,c->i+1)->n==1&&tat(c,c->i+1)->s[0]==')'){c->i++;break;}
    if(is(c,"...")){fn->variadic=1;c->i++;break;}   /* a trailing `...` -- the function is variadic */
    bcir_ctype ty;int si;if(p_type(c,&ty,&si))return 1;
    tok pn; int row_ptr=0;
    /* a direct function-pointer parameter `RET (*name)(PARAMS)`: unlike a typedef'd funcptr param there
     * is no alias to print, so capture the full signature as a synthesized prelude typedef `__bcir_fpN`
     * and type the param kind-3 with that tag. The indirect-call dispatch (p_icall) + the param/emit path
     * then reuse the typedef-funcptr machinery verbatim (ctype_str prints the tag). Scalar ret + params. */
    if(is(c,"(") && tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='*'
       && tat(c,c->i+2)->k==T_ID
       && tat(c,c->i+3)->k==T_PUN && tat(c,c->i+3)->n==1 && tat(c,c->i+3)->s[0]==')'
       && tat(c,c->i+4)->k==T_PUN && tat(c,c->i+4)->n==1 && tat(c,c->i+4)->s[0]=='('){
      bcir_ctype ret=ty;                            /* the already-parsed return type */
      c->i+=2; pn=adv(c);                            /* `( *` then the parameter NAME */
      if(!eat(c,")")||!eat(c,"("))return 1;          /* `) (` -- into the parameter-type list */
      char rets[64]; ctype_str(&ret,rets,sizeof rets);
      char sig[512]; size_t sw=0; int np=0;
      /* Clamp the OFFSET, not just the size: forming `sig+sw` with sw>sizeof sig is itself
       * out-of-bounds-pointer UB (clang-UBSan flags it even when snprintf's size is 0). `sig+SIG_OFF`
       * is at most one-past-the-end (a legal pointer) and `sizeof sig - SIG_OFF` is the remaining space
       * (0 once full); `sw` still accumulates the true would-be length for the fpdefs guard. See Bug 2. */
      #define SIG_OFF (sw<sizeof sig?sw:sizeof sig)
      sw+=snprintf(sig+SIG_OFF,sizeof sig-SIG_OFF,"typedef %s (*__bcir_fp%d)(",rets,c->n_fpdef);
      if(is(c,"void")&&tat(c,c->i+1)->n==1&&tat(c,c->i+1)->s[0]==')'){ c->i++; }   /* `(void)` */
      else if(!is(c,")")) for(;;){ bcir_ctype pt; int psi; if(p_type(c,&pt,&psi))return 1;
        if(isk(c,T_ID)) c->i++;                      /* an optional parameter name (ignored) */
        char ps[64]; ctype_str(&pt,ps,sizeof ps);
        sw+=snprintf(sig+SIG_OFF,sizeof sig-SIG_OFF,"%s%s",np?", ":"",ps); np++;
        if(is(c,",")){c->i++;continue;} break; }
      sw+=snprintf(sig+SIG_OFF,sizeof sig-SIG_OFF,"%s);\n",np?"":"void");
      #undef SIG_OFF
      /* only copy a signature that fit in `sig` (sw<=sizeof sig): a clamped one's `sw` overstates the
       * bytes actually in `sig`, so never read past sig[512]. */
      if(sw<sizeof sig && sw<sizeof c->fpdefs-c->fpdefs_w){ memcpy(c->fpdefs+c->fpdefs_w,sig,sw); c->fpdefs_w+=sw; }
      else c->emit_overflow=1;
      if(!eat(c,")"))return 1;                        /* past the parameter-type list */
      memset(&ty,0,sizeof ty); ty.kind=3; ty.size=8; ty.signd=0;
      ty.fp_ret_size=ret.size; ty.fp_ret_signd=(uint8_t)(ret.signd?1:0); ty.fp_ret_float=(uint8_t)(ret.is_float?1:0);
                                                       /* carry the funcptr param's RETURN type to type a c.call.indirect result */
      snprintf(ty.tag,sizeof ty.tag,"__bcir_fp%d",c->n_fpdef); c->n_fpdef++;
      row_ptr=1;                                      /* skip the row-ptr + array-suffix handling below */
    }
    if(!row_ptr && is(c,"(")){    /* (*name)[N]... -- a pointer-to-array "row pointer" (vendor headers); */
      int save=c->i; c->i++; int inner=0;          /* modeled as the equivalent multi-dim array param */
      while(is(c,"*")){inner++;c->i++;}
      if(inner==1 && isk(c,T_ID)){ tok cand=adv(c);
        if(is(c,")") && tat(c,c->i+1)->k==T_PUN && tat(c,c->i+1)->n==1 && tat(c,c->i+1)->s[0]=='['){
          c->i++;                                  /* consume ) ; the next token is [ */
          pn=cand; int nd=1; ty.adims[0]=0;        /* the outer (pointer) dim is unspecified */
          while(is(c,"[")){ c->i++; long long d=isk(c,T_INT)?(long long)adv(c).v:0;
            if(nd<3)ty.adims[nd]=(int)d; nd++; eat(c,"]"); }
          ty.nadims=nd<3?nd:3; if(ty.kind==0) ty.kind=2; row_ptr=1;
        } else c->i=save;
      } else c->i=save;
    }
    int vla_have=0; tok vla_tok; memset(&vla_tok,0,sizeof vla_tok);   /* §5.12 a VLA-param extent `a[n]` */
    if(!row_ptr){
    pn=adv(c);
    if(is(c,"[")){              /* an array parameter `T name[A][B]...` decays to a flat element ptr */
      int nd=0;
      while(is(c,"[")){ c->i++;
        long long d=0;
        if(isk(c,T_INT)) d=(long long)adv(c).v;          /* a static dim `[A]` -- the byte count is recorded */
        /* §5.12 a VLA-param extent `[n]`: `n` must be a BARE identifier naming a PRIOR in-scope param (source
         * order -- a later param is not yet in env). Capture it for the post-scan stability gate. */
        else if(!is(c,"]") && isk(c,T_ID) && tok_is(tat(c,c->i+1),"]")){
          tok cand=*pk(c);
          if(lookup(c,&cand)){ vla_tok=cand; vla_have=1; adv(c); }
        }
        if(nd<3)ty.adims[nd]=(int)d; nd++; eat(c,"]"); }   /* a non-int/non-id dim -> 0 today (fallback, no bind) */
      ty.nadims=nd<3?nd:3; if(ty.kind==0) ty.kind=2;     /* T[..] -> T* (element size kept in ty.size) */
      else if(ty.kind==2) ty.ptr_depth=(uint8_t)((ty.ptr_depth?ty.ptr_depth:1)+1);   /* T *a[..] -> T ** (`argv`):
                                                          * the decay is one more level, as for a scalar element */
    }
    }
    char pb[BCIR_CIR_NAME]; idcpy(pb,&pn);
    int rk=ty.kind==2?BCIR_RK_POINTER:ty.kind==1?BCIR_RK_AGGREGATE:BCIR_RK_SCALAR;
    uint32_t rid=add_res(c, ty_mmio(c,&ty,si,0)?BCIR_DOM_MMIO:BCIR_DOM_RAM,
                         ty.kind==2?ty.size:(ty.kind==1?c->s[si].size:ty.size),
                         ty.kind==2?(1<<16):1, ty.is_volatile, rk, pb);
    if(ty.kind==2){ bcir_resource *pr=&c->fn->res[c->fn->n_res-1];   /* a pointer param: carry the pointee
      * (width/sign/tag/depth) so pointer arithmetic on it (`p + i`) clones the real `T *` type, not uint32 */
      pr->is_signed=(uint8_t)(ty.signd?1:0); pr->is_float=(uint8_t)(ty.is_float?1:0); pr->ptr_depth=ty.ptr_depth;
      pr->is_plain_char=(uint8_t)(ty.is_plain_char?1:0);   /* a `char *` pointee: the deref load emits `char` */
      if(ty.ptr_to_struct) snprintf(pr->agg,BCIR_CIR_NAME,"%s %s",ty.is_union?"union":"struct",ty.tag); }
    else if(ty.is_float){ c->fn->res[c->fn->n_res-1].is_float=1;       /* a float/double parameter */
      if(ty.is_complex) c->fn->res[c->fn->n_res-1].is_complex=1; }     /* a _Complex parameter (a float pair) */
    else if(ty.kind==0){ c->fn->res[c->fn->n_res-1].is_signed=(uint8_t)(ty.signd?1:0);  /* signedness */
      if(ty.is_bool) c->fn->res[c->fn->n_res-1].is_bool=1;       /* a _Bool parameter */
      if(ty.bit_width>0) c->fn->res[c->fn->n_res-1].bit_width=ty.bit_width;   /* a C23 `_BitInt(N)` parameter */
      if(ty.is_plain_char) c->fn->res[c->fn->n_res-1].is_plain_char=1; }   /* a plain `char` parameter */
    else if(ty.kind==3) c->fn->res[c->fn->n_res-1].is_funcptr=1;   /* a funcptr param: stored to a member directly */
    if(ty.kind==1) snprintf(c->fn->res[c->fn->n_res-1].agg,BCIR_CIR_NAME,"%s %s",ty.is_union?"union":"struct",ty.tag);
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
      idcpy(pp->name,&pn);pp->rid=rid;pp->type=ty;}
    if(is(c,",")){c->i++;continue;} break;
  }
  if(!eat(c,")"))return 1;
  if(is(c,";")){                       /* a PROTOTYPE `T name(params);` (Phase 3 linking): record the
    * signature for call typing + render the extern declaration -- a cross-TU callee the host LINKER
    * resolves. A same-unit definition WINS: the unit-end rewrite in bcir_cfront_compile_target turns
    * its tu-calls back into ordinary R18 edges. Returns 2 (the unit loop discards the scratch fn). */
    c->i++;
    if(!CC_ENSURE(c,c->protos,c->n_protos,c->cap_protos)) return 1;
    snprintf(c->protos[c->n_protos].name,BCIR_CIR_NAME,"%s",fn->name);
    c->protos[c->n_protos].ret=fn->ret; c->n_protos++;
    char rets[64]; ctype_str(&fn->ret,rets,sizeof rets);
    char sig[512]; size_t sw=0;                       /* the fpdefs clamped-offset idiom (see above) */
    #define TU_OFF (sw<sizeof sig?sw:sizeof sig)
    sw+=snprintf(sig+TU_OFF,sizeof sig-TU_OFF,"extern %s %s(",rets,fn->name);
    for(int k=0;k<fn->n_params;k++){ char ps[64]; ctype_str(&fn->params[k].type,ps,sizeof ps);
      sw+=snprintf(sig+TU_OFF,sizeof sig-TU_OFF,"%s%s",k?", ":"",ps); }
    if(fn->variadic) sw+=snprintf(sig+TU_OFF,sizeof sig-TU_OFF,"%s...",fn->n_params?", ":"");
    sw+=snprintf(sig+TU_OFF,sizeof sig-TU_OFF,"%s);\n",(fn->n_params||fn->variadic)?"":"void");
    #undef TU_OFF
    if(sw<sizeof sig && sw<sizeof c->tudefs-c->tudefs_w){
      memcpy(c->tudefs+c->tudefs_w,sig,sw); c->tudefs_w+=sw; }
    else c->emit_overflow=1;
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
  const char *base = is_struct ? ty->tag
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
/* A unique C identifier for a named local. The lowering flattens scopes, so two source locals that
 * shared a name in disjoint scopes (e.g. `i` in two separate `for` loops, or a local shadowing a param)
 * are distinct resources with the same name; declaring both at function scope is a C redefinition. The
 * N-th occurrence of a name (params first, then resources in order) keeps the bare name for the first
 * and gets a `_N` suffix thereafter (`i`, `i_2`, ...) -- the same scheme as the oracle's emitter, used
 * for both the declaration and every reference. Unnamed temps stay `t<rid>`. */
static const char *uniq_local(const bcir_func *f,uint32_t rid,char *buf){
  const bcir_resource *r=res_of(f,rid);
  if(!r||!r->name[0]){ snprintf(buf,BCIR_CIR_NAME,"t%u",rid); return buf; }
  if(r->read_only){ snprintf(buf,BCIR_CIR_NAME,"%s",r->name); return buf; }   /* a file-scope global */
  for(int p=0;p<f->n_params;p++)                                              /* a param: keep its name */
    if(f->params[p].rid==rid){ snprintf(buf,BCIR_CIR_NAME,"%s",r->name); return buf; }
  int occ=0;                                            /* count earlier holders of the bare name */
  for(int p=0;p<f->n_params;p++)
    if(f->params[p].name[0] && !strcmp(f->params[p].name,r->name)) occ++;
  for(size_t i=0;i<f->n_res;i++){
    const bcir_resource *q=&f->res[i];
    if(q->rid==rid){
      if(occ==0) snprintf(buf,BCIR_CIR_NAME,"%s",r->name);
      else       snprintf(buf,BCIR_CIR_NAME,"%s_%d",r->name,occ+1);
      return buf;
    }
    if(q->name[0] && !strcmp(q->name,r->name)){         /* an earlier same-named resource... */
      int isp=0; for(int p=0;p<f->n_params;p++) if(f->params[p].rid==q->rid){isp=1;break;}
      if(!isp) occ++;                                   /* ...that is not itself a param (counted above) */
    }
  }
  snprintf(buf,BCIR_CIR_NAME,"%s",r->name); return buf;
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
  char bitint_ring[4][24];
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
                                int cplx,int boolp,int pchar){
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
  snprintf(buf,n,"%s%s%s",vol?"volatile ":"",base,stars);   /* a pointer to volatile storage */
  return buf;
}
static const char *decl_ty(bcir_emit_type_scratch *scratch,const bcir_func *f,uint32_t rid,char *buf,size_t n){
  const bcir_resource *r=res_of(f,rid);
  if(r && r->kind==BCIR_RK_POINTER)
    return ptr_spelling(buf,n,r->is_volatile,r->is_voidptr,r->agg,r->is_float,r->elem_bytes,r->is_signed,r->ptr_depth,
                        r->is_complex,r->is_bool,r->is_plain_char);
  snprintf(buf,n,"%s",tty(scratch,f,rid));
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
/* The type a volatile LOAD reads into temp `trid`: the value's own type, at the slot's width `w` for an
 * integer (a bitfield's storage unit) -- the oracle reads `et`, or the unit's unsigned type. */
static const char *vol_load_ty(bcir_emit_type_scratch *scratch,const bcir_func *f,uint32_t trid,long long w,char *buf,size_t n){
  const bcir_resource *r=res_of(f,trid);
  if(!r || r->kind==BCIR_RK_POINTER || r->is_float || r->is_complex || r->bit_width>0 || r->is_bool || r->is_plain_char)
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
    char ib[BCIR_CIR_NAME], nb[BCIR_CIR_NAME];
    uint32_t ext=ptrext_get(f,cl->rd[0]);                  /* §5.12 a naked pointer with a RECOVERED runtime extent */
    if(ext){ char eb[BCIR_CIR_NAME];
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
static size_t emit_func(const bcir_func *f,char *o,size_t on){
  size_t w=0; char a[BCIR_CIR_NAME],b[BCIR_CIR_NAME],d[BCIR_CIR_NAME],e[BCIR_CIR_NAME],ty[64],tb[80],gb[192];
  bcir_emit_type_scratch type_scratch={0};
  /* Clamp the OFFSET (never form o+w / on-w once the buffer is full) for every `snprintf(o+EO,on-EO,...)`
   * below -- the same memory-safety idiom the funcptr-typedef builder uses (see SIG_OFF above). Once
   * `w>=on`, `o+EO` is at most one-past-the-end (a legal pointer) and `on-EO` is 0, so snprintf writes
   * NOTHING but still returns the would-be length, keeping `w` a true running total; without this, the
   * unbounded named-local declaration loop and deep indentation could form `o+w` past the end and
   * underflow `on-w` to a huge size_t, writing out of bounds. Output is byte-identical whenever it fits. */
  #define EO (w<on?w:on)
  ctype_str(&f->ret,ty,sizeof ty);
  w+=snprintf(o+EO,on-EO,"static %s bcir_%s(",ty,f->name);
  if(f->n_params==0&&!f->variadic) w+=snprintf(o+EO,on-EO,"void");
  for(int i=0;i<f->n_params;i++){char pt[64];ctype_str(&f->params[i].type,pt,sizeof pt);
    w+=snprintf(o+EO,on-EO,"%s%s %s",i?", ":"",pt,f->params[i].name);}
  if(f->variadic) w+=snprintf(o+EO,on-EO,"%s...",f->n_params?", ":"");   /* a trailing variadic ellipsis */
  w+=snprintf(o+EO,on-EO,")\n{\n");
  /* declare named locals up front (mutable storage -- branch merges + loop accumulators) */
  for(size_t i=0;i<f->n_res;i++){const bcir_resource *r=&f->res[i];
    if(r->is_vla) continue;   /* a stack VLA: declared IN-BODY by c.vladecl (size unknown until then), not up front */
    if(is_named_local(f,r->rid)){
      char un[BCIR_CIR_NAME]; const char *nm=uniq_local(f,r->rid,un);   /* unique vs same-named scopes */
      int sx=-1; for(int k=0;k<f->n_statics;k++) if(!strcmp(f->statics[k].name,r->name)){sx=k;break;}
      /* a volatile OBJECT (scalar, array of scalars, struct) keeps its qualifier; a pointer's `volatile` is its
       * pointee's and rides in decl_ty; an array of pointers holds plain pointers */
      const char *vq=(r->is_volatile && r->kind!=BCIR_RK_POINTER && !r->ptr_depth)?"volatile ":"";
      if(sx>=0){                                     /* a static: its own type (the oracle's `_cname`), not uint32 */
        if(r->kind==BCIR_RK_POINTER) w+=snprintf(o+EO,on-EO,"  static %s%s = %lluu;\n",decl_ty(&type_scratch,f,r->rid,tb,sizeof tb),nm,(unsigned long long)f->statics[sx].init);
        else w+=snprintf(o+EO,on-EO,"  static %s%s %s = %lluu;\n",vq,tty(&type_scratch,f,r->rid),nm,(unsigned long long)f->statics[sx].init); }
      else if(r->is_funcptr&&r->agg[0]) w+=snprintf(o+EO,on-EO,"  %s %s;\n",r->agg,nm);   /* a funcptr local: `__bcir_fpN f;` */
      else if(r->kind==BCIR_RK_AGGREGATE&&r->agg[0]) w+=snprintf(o+EO,on-EO,"  %s%s %s%s;\n",vq,r->agg,nm,r->zinit?" = {0}":"");
      else if(r->kind==BCIR_RK_SCALAR&&decl_array(r)&&r->is_voidptr) w+=snprintf(o+EO,on-EO,"  void *%s[%u]%s;\n",nm,r->count,r->zinit?" = {0}":"");  /* an array of `void *` */
      else if(r->kind==BCIR_RK_SCALAR&&decl_array(r)&&r->agg[0]&&!r->ptr_depth) w+=snprintf(o+EO,on-EO,"  %s%s %s[%u]%s;\n",vq,r->agg,nm,r->count,r->zinit?" = {0}":"");  /* an ARRAY-OF-STRUCTS local `struct P a[N]` */
      else if(r->kind==BCIR_RK_SCALAR&&decl_array(r)&&r->ptr_depth)   /* an ARRAY of pointers `T *a[N]`: each element
        * its pointer type (a pointer to volatile keeps the pointee's qualifier), never the pointer-wide integer */
        w+=snprintf(o+EO,on-EO,"  %s%s[%u]%s;\n",ptr_spelling(tb,sizeof tb,r->is_volatile,0,r->agg,r->ptee_float,
                    r->ptee_bytes,r->ptee_signed,r->ptr_depth,0,0,r->ptee_plain_char),nm,r->count,r->zinit?" = {0}":"");
      else if(r->kind==BCIR_RK_SCALAR&&decl_array(r)) w+=snprintf(o+EO,on-EO,"  %s%s %s[%u]%s;\n",vq,tty(&type_scratch,f,r->rid),nm,r->count,r->zinit?" = {0}":"");  /* a local array */
      else if(r->kind==BCIR_RK_POINTER)               /* a pointer local: `T *p` (the pointee carries width/sign) */
        w+=snprintf(o+EO,on-EO,"  %s%s;\n",decl_ty(&type_scratch,f,r->rid,tb,sizeof tb),nm);
      else w+=snprintf(o+EO,on-EO,"  %s%s %s;\n",vq,tty(&type_scratch,f,r->rid),nm);}}
  int depth=1, lstk[BCIR_MAXDEPTH+1], nls=0, lctr=0;   /* loop-id stack + counter for the `continue` labels */
  #define IND() do{ for(int _k=0;_k<depth;_k++) w+=snprintf(o+EO,on-EO,"  "); }while(0)
  for(size_t i=0;i<f->n_claims;i++){const bcir_claim *cl=&f->claims[i];
    /* L6 control-flow markers (rendered as braces) */
    if(!strcmp(cl->op,"c.if")){IND();w+=snprintf(o+EO,on-EO,"if (%s) {\n",rname(f,cl->rd[0],a));depth++;continue;}
    if(!strcmp(cl->op,"c.else")){depth--;IND();w+=snprintf(o+EO,on-EO,"} else {\n");depth++;continue;}
    if(!strcmp(cl->op,"c.endif")){depth--;IND();w+=snprintf(o+EO,on-EO,"}\n");continue;}
    if(!strcmp(cl->op,"c.loop")){IND();w+=snprintf(o+EO,on-EO,"while (1) {\n");depth++;
      if(nls<BCIR_MAXDEPTH+1)lstk[nls++]=lctr++;continue;}
    if(!strcmp(cl->op,"c.loop.test")){IND();w+=snprintf(o+EO,on-EO,"if (!%s) break;\n",rname(f,cl->rd[0],a));continue;}
    if(!strcmp(cl->op,"c.cont.tgt")){IND();w+=snprintf(o+EO,on-EO,"__cont_%d: ;\n",nls?lstk[nls-1]:0);continue;}
    if(!strcmp(cl->op,"c.endloop")){depth--;IND();w+=snprintf(o+EO,on-EO,"}\n");if(nls)nls--;continue;}
    if(!strcmp(cl->op,"c.vladecl")){IND();   /* a 1-D stack VLA, declared IN-BODY: `<elem> a[__bcir_extK];` */
      w+=snprintf(o+EO,on-EO,"%s %s[%s];\n",tty(&type_scratch,f,cl->wr[0]),rname(f,cl->wr[0],a),rname(f,cl->rd[0],b));continue;}
    if(!strcmp(cl->op,"c.ptradd")){IND();w+=snprintf(o+EO,on-EO,"%s += %s;\n",rname(f,cl->wr[0],a),rname(f,cl->rd[1],b));continue;}  /* pointer p += n */
    if(!strcmp(cl->op,"c.ptrsub")){IND();w+=snprintf(o+EO,on-EO,"%s -= %s;\n",rname(f,cl->wr[0],a),rname(f,cl->rd[1],b));continue;}  /* pointer p -= n */
    if(!strcmp(cl->op,"c.break")){IND();w+=snprintf(o+EO,on-EO,"break;\n");continue;}
    if(!strcmp(cl->op,"c.switch")){IND();w+=snprintf(o+EO,on-EO,"switch (%s) {\n",rname(f,cl->rd[0],a));depth++;continue;}
    if(!strncmp(cl->op,"c.case:",7)){IND();w+=snprintf(o+EO,on-EO,"case %s:\n",cl->op+7);continue;}  /* a real case label */
    if(!strcmp(cl->op,"c.default")){IND();w+=snprintf(o+EO,on-EO,"default:\n");continue;}
    if(!strcmp(cl->op,"c.endswitch")){depth--;IND();w+=snprintf(o+EO,on-EO,"}\n");continue;}
    if(!strcmp(cl->op,"c.continue")){IND();w+=snprintf(o+EO,on-EO,"goto __cont_%d;\n",nls?lstk[nls-1]:0);continue;}
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
    else if(!strncmp(cl->op,"c.cast:",7))                  /* (type)operand -- width / float / pointer cast */
      w+=snprintf(o+EO,on-EO,"%s %s = (%s)%s;\n",decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),cl->op+7,rname(f,cl->rd[0],a));  /* decl_ty: a pointer-snapshot cast keeps `T *` */
    else if(!strcmp(cl->op,"c.select"))                    /* ternary: cond ? then : els -- the select's own
                                                            * (signed/unsigned) type, not a hardcoded
                                                            * uint32_t (see the c.const note below). */
      w+=snprintf(o+EO,on-EO,"%s %s = (%s ? %s : %s);\n",tty(&type_scratch,f,cl->wr[0]),rname(f,cl->wr[0],d),
                  rname(f,cl->rd[0],a),rname(f,cl->rd[1],b),rname(f,cl->rd[2],e));
    else if(!strcmp(cl->op,"c.const")){
      /* declare the constant with its OWN type, not a hardcoded uint32_t: a bare integer literal (e.g.
       * `0` in `x < 0`) is signed (int), so emitting `uint32_t = 0u` made a signed comparison promote to
       * unsigned (`int32_t < uint32_t` -> unsigned) -- a miscompile. The literal's (width, signedness)
       * was already recorded on the temp (lit_int_type); render the matching type + suffix. */
      const bcir_resource *cr=res_of(f,cl->wr[0]); int cs=cr&&cr->is_signed;
      w+=snprintf(o+EO,on-EO,"%s %s = %llu%s;\n",tty(&type_scratch,f,cl->wr[0]),rname(f,cl->wr[0],d),
                  (unsigned long long)cl->imm[0], cs?"":"u"); }
    else if(!strcmp(cl->op,"c.sizeof.vla"))                 /* runtime `sizeof a` of a VLA: extent × sizeof(elem).
                                                            * HARDCODE the literal `size_t` (NOT tty(), which
                                                            * returns "uint64_t" for an 8-byte unsigned scalar):
                                                            * the oracle emits literal `size_t` (scalar('size_t')),
                                                            * so the two rails would diverge byte-for-byte. */
      w+=snprintf(o+EO,on-EO,"size_t %s = (size_t)((size_t)%s * %lld);\n",rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),(long long)cl->imm[0]);
    else if(!strcmp(cl->op,"c.copy")){
      if(is_named_local(f,cl->wr[0])||is_global_ref(f,cl->wr[0])||is_param_ref(f,cl->wr[0])) w+=snprintf(o+EO,on-EO,"%s = %s;\n",rname(f,cl->wr[0],d),rname(f,cl->rd[0],a));
      else w+=snprintf(o+EO,on-EO,"%s %s = %s;\n",decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a));   /* decl_ty: a copied pointer temp keeps `T *` */
    }else if(!strcmp(cl->op,"c.load")){
      const bcir_resource *br=res_of(f,cl->rd[0]); long long off=cl->n_imm?cl->imm[0]:0;
      if(cl->n_rd==2 && cl->n_imm){       /* s.arr[i] / a[i].f: load at base + off + idx*stride, copy `es` bytes */
        const char *amp=base_amp(br); long long es=cl->n_imm>1?cl->imm[1]:4;
        long long stride=cl->n_imm>2?cl->imm[2]:es;   /* array-of-structs `arr[i].field`: stride sizeof(elem) != es */
        /* the temp carries the element's (width, signedness): memcpy es bytes into it so a signed sub-int
         * element reads sign-extended (the zero-extending uint32 form dropped the sign). */
        if(cl->is_volatile){ char at[96], vp[112];     /* a volatile element: one access of exactly its type */
          w+=snprintf(o+EO,on-EO,"%s %s = *(%s)((const volatile char *)%s%s + %lld + (size_t)%s * %lld);\n",
            tty(&type_scratch,f,cl->wr[0]),rname(f,cl->wr[0],d),
            vol_ptr(vol_load_ty(&type_scratch,f,cl->wr[0],es,at,sizeof at),vp,sizeof vp),
            amp,rname(f,cl->rd[0],a),off,rname(f,cl->rd[1],b),stride); }
        else
        w+=snprintf(o+EO,on-EO,"%s %s; memcpy(&%s, (const char *)%s%s + %lld + (size_t)%s * %lld, %lld);\n",
          tty(&type_scratch,f,cl->wr[0]),rname(f,cl->wr[0],d),rname(f,cl->wr[0],d),amp,rname(f,cl->rd[0],a),off,rname(f,cl->rd[1],b),stride,es); }
      else if(cl->n_rd==2) w+=snprintf(o+EO,on-EO,"%s %s = %s[%s];\n",decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),guard_idx(f,cl,gb,sizeof gb,0));  /* READ guard; decl_ty: an array-of-pointers element load is `T *` */
      else if(cl->is_volatile){ char at[96], vp[112];   /* a volatile member / dereference: one ordered access of
                                                       * exactly the accessed type -- never a 32-bit register */
        const char *amp=holds_pointer(br)?"":"&";
        w+=snprintf(o+EO,on-EO,"%s %s = *(%s)((const volatile char *)%s%s + %lld);\n",
          decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),
          vol_ptr(vol_load_ty(&type_scratch,f,cl->wr[0],cl->n_imm>1?cl->imm[1]:0,at,sizeof at),vp,sizeof vp),
          amp,rname(f,cl->rd[0],a),off); }
      else { const char *amp=holds_pointer(br)?"":"&"; long long fsz=cl->n_imm>1?cl->imm[1]:4;
        /* a plain member load: memcpy fsz bytes into the typed temp so a signed sub-int member sign-extends */
        w+=snprintf(o+EO,on-EO,"%s %s; memcpy(&%s, (const char *)%s%s + %lld, %lld);\n",decl_ty(&type_scratch,f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),rname(f,cl->wr[0],d),amp,rname(f,cl->rd[0],a),off,fsz); }  /* decl_ty: a pointer member load is `T *t` */
    }else if(!strcmp(cl->op,"c.store")&&cl->n_rd==3){   /* L3: array element store  a[idx] = value */
      if(cl->n_imm){                      /* s.arr[i]=v / a[i].f=v: store at base + off + idx*stride */
        const bcir_resource *br=res_of(f,cl->rd[0]); const char *amp=base_amp(br);
        long long off=cl->imm[0], es=cl->n_imm>1?cl->imm[1]:4;
        long long stride=cl->n_imm>3?cl->imm[3]:es;    /* array-of-structs `arr[i].field=v`: stride sizeof(elem) */
        const bcir_resource *vr=res_of(f,cl->rd[2]);   /* a float element converts (double->float), not a uint
                                                        * reinterpret; a narrower int widens to the element. */
        const char *vt=(cl->n_imm>2&&cl->imm[2])?"_Bool"   /* a _Bool element: `_Bool _v = x` normalizes to 0/1 */
                      :(vr&&vr->is_complex)?(es==8?"float _Complex":es>16?"long double _Complex":"double _Complex")
                      :(vr&&vr->is_float)?(es==4?"float":es>8?"long double":"double")
                      :(es==1?"uint8_t":es==2?"uint16_t":es==8?"uint64_t":"uint32_t");
        if(cl->is_volatile){ char st[96], vp[112];     /* a volatile element: one store of exactly its slot */
          w+=snprintf(o+EO,on-EO,"*(%s)((volatile char *)%s%s + %lld + (size_t)%s * %lld) = %s;\n",
            vol_ptr(vol_slot_ty(&type_scratch,f,cl->rd[2],es,(int)(cl->n_imm>2?cl->imm[2]:0),st,sizeof st),vp,sizeof vp),
            amp,rname(f,cl->rd[0],a),off,rname(f,cl->rd[1],b),stride,rname(f,cl->rd[2],d)); }
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
        else { char st[96], vp[112];
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
          * `o->fn = g_func`): store through a GENERIC funcptr lvalue so a function NAME decays to its address
          * (a plain `memcpy(&g_func,8)` would copy the function's CODE; `void *` cannot hold a funcptr). The
          * call site reads the member's real type, and function pointers round-trip through the cast. */
          w+=snprintf(o+EO,on-EO,"*(void (**)(void))((char *)%s%s + %lld) = (void (*)(void))%s;\n",
                      amp,rname(f,cl->rd[0],a),off,rname(f,cl->rd[1],b));
        } else if(vr && vr->kind==BCIR_RK_AGGREGATE){   /* a struct/union member set from a struct VALUE (a nested
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
    else if(!strncmp(cl->op,"c.atomic.",9))      /* atomic RMW -> the matching builtin */
      w+=snprintf(o+EO,on-EO,"uint32_t %s = __atomic_fetch_%s(%s, %s, __ATOMIC_SEQ_CST);\n",
                  rname(f,cl->wr[0],d),cl->op+9,rname(f,cl->rd[0],a),rname(f,cl->rd[1],b));
    else if(!strncmp(cl->op,"c.cmpxchg.",10))     /* compare-and-swap -> the __sync CAS builtin */
      w+=snprintf(o+EO,on-EO,"uint32_t %s = __sync_%s_compare_and_swap(%s, %s, %s);\n",
                  rname(f,cl->wr[0],d),cl->op+10,rname(f,cl->rd[0],a),rname(f,cl->rd[1],b),rname(f,cl->rd[2],e));
    else if(!strcmp(cl->op,"c.fence"))
      w+=snprintf(o+EO,on-EO,"__atomic_thread_fence(__ATOMIC_SEQ_CST);\n");
    else if(!strcmp(cl->op,"c.fence.acquire"))    /* SEG7: order-parameterized acquire fence (PORTABLE -- the */
      w+=snprintf(o+EO,on-EO,"__atomic_thread_fence(__ATOMIC_ACQUIRE);\n");   /* C twin does NO per-ISA asm) */
    else if(!strcmp(cl->op,"c.fence.release"))    /* SEG7: order-parameterized release fence (PORTABLE) */
      w+=snprintf(o+EO,on-EO,"__atomic_thread_fence(__ATOMIC_RELEASE);\n");
    else if(!strncmp(cl->op,"c.c11atom.",10)){   /* C11 <stdatomic.h> generics on _Atomic objects */
      const char *fn=cl->op+10;                  /* fetch_add / fetch_sub / fetch_xor / load / store */
      if(!strcmp(fn,"load")) w+=snprintf(o+EO,on-EO,"uint32_t %s = atomic_load(%s);\n",rname(f,cl->wr[0],d),rname(f,cl->rd[0],a));
      else if(!strcmp(fn,"store")) w+=snprintf(o+EO,on-EO,"atomic_store(%s, %s);\n",rname(f,cl->rd[0],a),rname(f,cl->rd[1],b));
      else if(!strncmp(fn,"cas_",4))             /* cas_strong/weak -> _Bool atomic_compare_exchange_<...>(obj,&exp,des) */
        w+=snprintf(o+EO,on-EO,"_Bool %s = atomic_compare_exchange_%s(%s, %s, %s);\n",
                    rname(f,cl->wr[0],d),fn+4,rname(f,cl->rd[0],a),rname(f,cl->rd[1],b),rname(f,cl->rd[2],e));
      else w+=snprintf(o+EO,on-EO,"uint32_t %s = atomic_%s(%s, %s);\n",rname(f,cl->wr[0],d),fn,rname(f,cl->rd[0],a),rname(f,cl->rd[1],b)); }
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
      if(cl->n_wr==0) w+=snprintf(o+EO,on-EO,"%s(",cl->op+10);
      else w+=snprintf(o+EO,on-EO,"%s %s = %s(",tty(&type_scratch,f,cl->wr[0]),rname(f,cl->wr[0],d),cl->op+10);
      for(int k=0;k<cl->n_rd;k++) w+=snprintf(o+EO,on-EO,"%s%s",k?", ":"",rname(f,cl->rd[k],a));
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
      for(int k=0;k<cl->n_rd;k++) w+=snprintf(o+EO,on-EO,"%s%s",k?", ":"",rname(f,cl->rd[k],a));
      w+=snprintf(o+EO,on-EO,");\n"); }
    else if(!strncmp(cl->op,"c.call:",7)){
      const bcir_resource *rr=res_of(f,cl->wr[0]);   /* a struct/union RETURN declares `struct P t = bcir_..` */
      const char *dty=(rr&&rr->kind==BCIR_RK_AGGREGATE&&rr->agg[0])?rr->agg:tty(&type_scratch,f,cl->wr[0]);
      w+=snprintf(o+EO,on-EO,"%s %s = bcir_%s(",dty,rname(f,cl->wr[0],d),cl->op+7);
      for(int k=0;k<cl->n_rd;k++) w+=snprintf(o+EO,on-EO,"%s%s",k?", ":"",rname(f,cl->rd[k],a));
      w+=snprintf(o+EO,on-EO,");\n"); }
    else if(!strcmp(cl->op,"c.call.indirect")){    /* rd[0] is the function pointer; rd[1..] the args */
      w+=snprintf(o+EO,on-EO,"%s %s = %s(",tty(&type_scratch,f,cl->wr[0]),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a));  /* result typed by the funcptr's return */
      for(int k=1;k<cl->n_rd;k++) w+=snprintf(o+EO,on-EO,"%s%s",k>1?", ":"",rname(f,cl->rd[k],b));
      w+=snprintf(o+EO,on-EO,");\n"); }
    else if(!strncmp(cl->op,"c.call.imember:",15)){   /* o->fn(args): funcptr struct member */
      const char *sep=(cl->n_imm&&cl->imm[0])?"->":".";
      w+=snprintf(o+EO,on-EO,"%s %s = %s%s%s(",tty(&type_scratch,f,cl->wr[0]),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),sep,cl->op+15);  /* result typed by the funcptr's return */
      for(int k=1;k<cl->n_rd;k++) w+=snprintf(o+EO,on-EO,"%s%s",k>1?", ":"",rname(f,cl->rd[k],b));
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
static int try_top_decl(CC *c){
  if(c->failed) return 0;
  if(is(c,"typedef")){ p_typedef(c); return 1; }
  if(is(c,"enum")){
    int save=c->i; c->i++; if(isk(c,T_ID)&&!is(c,"{")) c->i++;
    if(is(c,"{")){ p_enum_body(c); eat(c,";"); return 1; }
    c->i=save; return 0;                             /* `enum tag` as a type -> a function follows */
  }
  if(is(c,"struct")||is(c,"union")){
    /* a struct *definition*?  struct [attrs] [TAG] [attrs] {  -- lookahead past attributes. */
    int save=c->i; c->i++; int pk_=0,al_=0; attrs(c,&pk_,&al_);
    if(isk(c,T_ID)&&!is(c,"{")) c->i++;             /* the tag */
    attrs(c,&pk_,&al_);
    int isdef = is(c,"{"); c->i=save;
    if(isdef){ p_struct_body(c); eat(c,";"); return 1; }
    return 0;                                        /* struct used as a type -> a function follows */
  }
  return 0;
}

/* Lookahead: does the current top-level token begin a file-scope global (a `TYPE NAME ...` that is
 * NOT followed by `(`) rather than a function?  Restores the cursor so the caller re-parses. */
static int looks_global(CC *c){
  int save=c->i, sf=c->failed; bcir_ctype ty; int si; int global=0;
  if(!p_type(c,&ty,&si) && isk(c,T_ID)){ c->i++; if(!is(c,"(")) global=1; }
  c->i=save; c->failed=sf; c->err[0]=0;
  return global;
}
/* Parse a file-scope global `[static][const] TYPE NAME [N] [= ...];` and register it.  The
 * initializer is skipped -- the emitter references the global by name (defined in the source), so
 * the claim graph needs only the name + element type + length. */
static void p_global(CC *c){
  bcir_ctype ty; int si; if(p_type(c,&ty,&si)) return; tok nm=adv(c);
  int count=1, is_arr=0, init_a=0, init_b=0;
  while(is(c,"[")){ c->i++; count = isk(c,T_INT)?(int)adv(c).v:0; eat(c,"]"); is_arr=1; }
  if(is(c,"=")){ c->i++; init_a=c->i;
    if(is(c,"{")){ c->i++; int d=1; while(d>0&&!isk(c,T_END)&&!c->failed){ if(is(c,"{"))d++; else if(is(c,"}"))d--; c->i++; } }
    else (void)ce_expr(c,0);
    init_b=c->i;
  }
  eat(c,";");
  CC_ENSURE(c, c->gv, c->ngv, c->cap_gv);
  if(c->ngv<c->cap_gv){ gvar *g=&c->gv[c->ngv++]; idcpy(g->name,&nm); g->ty=ty; g->count=count;
    g->is_arr=is_arr; g->init_a=init_a; g->init_b=init_b; }
}

/* --- public entry -------------------------------------------------------- */
static void cfront_free_func(const bcir_host_allocator *allocator, bcir_func *fn) {
  if(!fn) return;
  for(int i=0;i<fn->n_host_literals;i++)
    bcir_host_deallocate(allocator,fn->host_literals[i].spelling);
  bcir_host_deallocate(allocator,fn->res);
  bcir_host_deallocate(allocator,fn->claims);
  bcir_host_deallocate(allocator,fn->params);
  bcir_host_deallocate(allocator,fn->calls);
  bcir_host_deallocate(allocator,fn->statics);
  bcir_host_deallocate(allocator,fn->host_literals);
  bcir_host_deallocate(allocator,fn->ptr_extents);
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
  sdef *s; tdef *td; econst *ec; gvar *gv; venv *env; void *protos;
  int cap_s,cap_td,cap_ec,cap_gv,cap_env,cap_protos;
  if(!context||!context->state) return;
  c=(CC *)context->state;
  bcir_host_arena_reset(&c->scratch);
  allocator=c->allocator; scratch=c->scratch;
  s=c->s;cap_s=c->cap_s;td=c->td;cap_td=c->cap_td;ec=c->ec;cap_ec=c->cap_ec;
  gv=c->gv;cap_gv=c->cap_gv;env=c->env;cap_env=c->cap_env;
  protos=c->protos;cap_protos=c->cap_protos;
  memset(c,0,sizeof *c);
  c->allocator=allocator;c->scratch=scratch;
  c->s=s;c->cap_s=cap_s;c->td=td;c->cap_td=cap_td;c->ec=ec;c->cap_ec=cap_ec;
  c->gv=gv;c->cap_gv=cap_gv;c->env=env;c->cap_env=cap_env;
  c->protos=protos;c->cap_protos=cap_protos;
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
  memset(c,0,sizeof *c);bcir_host_deallocate(&allocator,c);
  memset(context,0,sizeof *context);
}

static int cfront_failure(bcir_cfront_context *context, bcir_cfront_result *out,
                          const char *message) {
  char diagnostic[sizeof out->diag];
  snprintf(diagnostic,sizeof diagnostic,"%s",message&&message[0]?message:"compile failed");
  if(context&&context->state)((CC *)context->state)->jump_active=0;
  bcir_cfront_free(out);
  out->ok=0;out->emitted_ok=0;out->emitted[0]=0;
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
      char callee[BCIR_CIR_NAME]; snprintf(callee,sizeof callee,"%s",cl->op+10);
      snprintf(cl->op,sizeof cl->op,"%s%s",cl->n_wr?"c.call:":"c.call.void:",callee);
      if(!CC_ENSURE(c,f->calls,f->n_calls,f->cap_calls))
        return cfront_failure(context,out,c->err);
      snprintf(f->calls[f->n_calls++],BCIR_CIR_NAME,"%s",callee);
    }
  }
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
  int emit_impossible=c->emit_overflow;
  out->ok=bcir_verify_unit_with_allocator(&out->unit,out->diag,sizeof out->diag,&c->allocator);
  if(!out->ok&&!strcmp(out->diag,"oom"))return cfront_failure(context,out,"oom");
  /* C.2 verified-C attestation: stamp the emitted C with its R-law status + R13 digest + the unit's
   * derived link flags (B1; so --emit-c is self-describing about what it links -- a comment, stripped
   * on re-parse). The link_flags line mirrors the oracle's C.2 attestation. */
  const bcir_func *entry = out->unit.n_funcs ? &out->unit.funcs[out->unit.n_funcs-1] : NULL;
  char lflags[256]; bcir_cfront_link_flags(&out->unit, lflags, sizeof lflags);
  if(emit_impossible){out->emitted[0]=0;out->emitted_ok=0;
    c->jump_active=0;bcir_host_arena_reset(&c->scratch);return 0;}
  const size_t emit_cap=sizeof out->emitted;
  int head_n=snprintf(out->emitted,emit_cap,
    "/* BCIR verified-C attestation (C.2) -- generated by bcir_cfront, do not edit.\n"
    " *   R1-R8 + R18  %s\n"
    " *   R9 plan / R10-R11 pack  checked in the compile->execute loop\n"
    " *   R12 lowering-contract  support preserved (emit Clang-behaviour-equivalent)\n"
    " *   R13 provenance digest  %016llx\n"
    " *   R17 accuracy  exact (integer / Q-fixed, 0 ULP)\n"
    " *   link_flags  %s\n */\n",
    out->ok?"clean":"DIRTY", entry?(unsigned long long)bcir_provenance_digest(entry):0ull,
    lflags[0]?lflags:"-");
  if(head_n<0)return cfront_failure(context,out,"cannot render emitted C");
  size_t w=(size_t)head_n;
  #define EMIT_OFF (w<emit_cap?w:emit_cap)
  if(c->fpdefs_w){                                      /* synthesized funcptr-param typedefs (prelude) */
    int n=snprintf(out->emitted+EMIT_OFF,emit_cap-EMIT_OFF,"%.*s",(int)c->fpdefs_w,c->fpdefs);
    if(n<0||SIZE_MAX-w<(size_t)n)return cfront_failure(context,out,"cannot render emitted C");
    w+=(size_t)n;
  }
  if(c->tudefs_w){                                      /* extern declarations for cross-TU callees */
    int n=snprintf(out->emitted+EMIT_OFF,emit_cap-EMIT_OFF,"%.*s",(int)c->tudefs_w,c->tudefs);
    if(n<0||SIZE_MAX-w<(size_t)n)return cfront_failure(context,out,"cannot render emitted C");
    w+=(size_t)n;
  }
  for(int i=0;i<out->unit.n_funcs;i++){
    size_t n=emit_func(&out->unit.funcs[i],out->emitted+EMIT_OFF,emit_cap-EMIT_OFF);
    if(SIZE_MAX-w<n)return cfront_failure(context,out,"cannot render emitted C");
    w+=n;
    if(i+1<out->unit.n_funcs){
      int sep=snprintf(out->emitted+EMIT_OFF,emit_cap-EMIT_OFF,"\n");
      if(sep<0||SIZE_MAX-w<(size_t)sep)return cfront_failure(context,out,"cannot render emitted C");
      w+=(size_t)sep;
    }
  }
  #undef EMIT_OFF
  if(w>=emit_cap){out->emitted[0]=0;out->emitted_ok=0;
    c->jump_active=0;bcir_host_arena_reset(&c->scratch);return 0;}
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
static void vn_base(const char *op, char *out){
  const char *c=strchr(op,':');
  if(c){ size_t h=(size_t)(c-op); char head[BCIR_CIR_NAME];
    if(h>=sizeof head) h=sizeof head-1; memcpy(head,op,h); head[h]=0;
    if(BCIR_VN_STRIP(head)){ snprintf(out,BCIR_CIR_NAME,"%s",head); return; } }
  snprintf(out,BCIR_CIR_NAME,"%s",op);
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
static void vn_imm(const bcir_claim *cl, sbuf *rec){
  const char *op=cl->op; char nb[24]; int n=cl->n_imm;
  if(!strcmp(op,"c.const")||!strcmp(op,"c.addrof")||!strcmp(op,"c.bf.get")||!strcmp(op,"c.bf.set")||
     !strcmp(op,"c.call.imember")||!strcmp(op,"c.sizeof.vla")){
    for(int k=0;k<n;k++){ if(k)sb_str(rec,","); int l=snprintf(nb,sizeof nb,"%lld",(long long)cl->imm[k]); sb_add(rec,nb,(size_t)l); }
  } else if(!strcmp(op,"c.load")){
    long long off = n>0 ? (long long)cl->imm[0] : 0;   /* the member byte offset (0 if absent) */
    int l=snprintf(nb,sizeof nb,"%lld",off); sb_add(rec,nb,(size_t)l);
  } else if(!strcmp(op,"c.store")){
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
  char base[BCIR_CIR_NAME]; vn_base(cl->op,base);
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
    char base[BCIR_CIR_NAME]; vn_base(cl->op,base);
    char *parts[BCIR_CLAIM_MAX_RD]; sbuf ps[BCIR_CLAIM_MAX_RD]; int np=cl->n_rd;
    for(int k=0;k<np;k++){ ps[k]=sbuf_for(arena); vn_of(&v,cl->rd[k],0,&ps[k]); parts[k]=ps[k].s?ps[k].s:(char*)""; }
    if(vn_commutative(base)) sort_strs(parts,np);     /* commutative: sort; else POSITIONAL */
    sbuf rec=sbuf_for(arena); char nb[24];
    sb_str(&rec,base); sb_str(&rec,"|");
    int ol=snprintf(nb,sizeof nb,"%d",(int)cl->opcode); sb_add(&rec,nb,(size_t)ol); sb_str(&rec,"|");
    for(int k=0;k<np;k++){ if(k)sb_str(&rec,","); sb_str(&rec,parts[k]); }
    sb_str(&rec,"|");
    vn_imm(cl,&rec);                                  /* the semantic imm (member offset / bitfield layout) */
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
/* The raw canonical serialization the digest hashes (text, NOT hashed) -- the byte-identity proof
 * (the Python cfront_structural_canon must equal this byte-for-byte on the corpus). */
void bcir_cfront_canon_with_allocator(const bcir_unit *u,char *buf,size_t n,
                                      const bcir_host_allocator *allocator){
  if(!buf||!n)return;
  buf_ctx c={buf,n,0};canon_walk(u,buf_emit,&c,allocator);
  if(n) buf[c.w<n?c.w:n-1]=0;
}
void bcir_cfront_canon(const bcir_unit *u,char *buf,size_t n){
  bcir_host_allocator allocator=bcir_host_allocator_default();
  bcir_cfront_canon_with_allocator(u,buf,n,&allocator);
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

/* a device access: an MMIO-domain claim, or a load or store whose BASE resource is MMIO-domain. The base
 * decides, not the claim's spelling -- both rails mark every such access today (`p[i]` through a
 * `volatile T *` was once an ordinary load here), and one predicate over the resource keeps the two rails'
 * answers the same should a spelling drift again. */
static int esc_device(const bcir_func *f,const bcir_claim *c){
  if(c->opcode==BCIR_OP_NOP) return 0;   /* a control marker (`c.return`, `c.if`, ...) reads a value and performs
                                          * no access; R3 may still make it device-domain when that value is a
                                          * pointer to volatile storage. The oracle has no marker claims. */
  if(c->domain==BCIR_DOM_MMIO) return 1;
  if(c->n_rd && (!strcmp(c->op,"c.load")||!strcmp(c->op,"c.store"))){
    const bcir_resource *r=res_of(f,c->rd[0]); return r && r->domain==BCIR_DOM_MMIO; }
  return 0;
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
      if(esc_device(F->f,c)){ ESC_SET(rd,EO_TOP); ESC_SET(wr,EO_TOP); }   /* a device access */
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
