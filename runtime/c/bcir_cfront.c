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
typedef struct { char tag[BCIR_CIR_NAME]; field f[MAXFLD]; int nf; int size; int align; int is_union; } sdef;
typedef struct { char name[BCIR_CIR_NAME]; bcir_ctype ty; int sidx; } tdef;   /* a typedef alias */
typedef struct { char name[BCIR_CIR_NAME]; long long val; } econst;           /* an enum constant */

typedef struct {
  char name[BCIR_CIR_NAME];
  uint32_t rid;
  bcir_ctype type;   /* scalar / struct-by-value / pointer */
  int sidx;          /* struct index for kind 1 or ptr_to_struct */
} venv;

typedef struct { char name[BCIR_CIR_NAME]; bcir_ctype ty; int count; } gvar;  /* a file-scope global */

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
  char err[256]; int failed;
} CC;
/* The recursion-depth cap for the recursive-descent parser. Comfortably below the real native-stack
 * limit (the deepest cycle, the expression chain p_expr->...->p_primary->`(`->p_expr, is ~8 frames per
 * nesting level, so ~1200 levels is well under an 8MiB stack at ~Nx that depth) yet far ABOVE any real
 * program's nesting in the fixture corpus (the deepest fixture nests only a handful of levels). The
 * Python oracle's recursion guard uses the same cap so the two rails agree on the over-deep boundary. */
#define BCIR_MAXDEPTH 1200
/* Enter a recursive cycle: bump depth, and on overflow record a clean parse error. Pairs with LEAVE_REC.
 * The `_over` flag lets a caller bail with its own (typed) return value after a failed ENTER_REC. */
#define ENTER_REC(c) (++(c)->depth > BCIR_MAXDEPTH ? (fail((c),"nesting too deep"), 1) : 0)
#define LEAVE_REC(c) (--(c)->depth)
/* Grow a CC parser-state array geometrically (no fixed cap), zeroing the fresh slots. On OOM the cap
 * is left unchanged, so the append guard (`n < cap`) stops -- a truncated unit the verifier catches. */
#define CC_ENSURE(arr, n, cap) do { \
  if((n) >= (cap)) { int _oc=(cap), _nc=(cap)?(cap)*2:8; void *_p=realloc((arr),(size_t)_nc*sizeof *(arr)); \
    if(_p){ (arr)=_p; (cap)=_nc; \
      memset((char *)(arr)+(size_t)_oc*sizeof *(arr), 0, (size_t)(_nc-_oc)*sizeof *(arr)); } } } while(0)
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
 * name -- and so identical literals in a function share one global (dedup). Owned copies, reset per
 * compile. A host-tool concern only (the freestanding IR it emits never sees this). */
#define BCIR_MAX_STRLITS 512
typedef struct { uint32_t rid; char *s; const bcir_func *fn; } strent;
static strent g_strtab[BCIR_MAX_STRLITS];
static int g_nstr;

static void strtab_reset(void) {
  for (int k = 0; k < g_nstr; k++) { free(g_strtab[k].s); g_strtab[k].s = NULL; }
  g_nstr = 0;
}
/* The full spelling registered for a string-literal resource (by rid), or NULL if rid is not one. */
static const char *strtab_lookup(uint32_t rid) {
  for (int k = 0; k < g_nstr; k++) if (g_strtab[k].rid == rid) return g_strtab[k].s;
  return NULL;
}

/* §5.12 recoverable extents (the C twin of LoweredFunc.ptr_extent): a pointer local bound to
 * malloc(N*sizeof(T)) / calloc(N, sizeof(T)) carries the RECOVERED element-count variable -- its `p[i]`
 * accesses promote to `masked` and emit `a[BCIR_CHK(rid, idx, <count var>, "func:ptr")]`. The map lives
 * on a file-static side table keyed by the OWNING function (whose pointer is stable from p_func through
 * emit_func) so the emitter (guard_idx, which only has a `const bcir_func *`) can read it back -- the
 * bcir_func struct carries no extra field. Reset per translation unit. */
#define BCIR_MAX_PTREXT 256
typedef struct { const bcir_func *fn; uint32_t ptr_rid; uint32_t cnt_rid; } ptrext_ent;
static ptrext_ent g_ptrext[BCIR_MAX_PTREXT];
static int g_nptrext;
static void ptrext_reset(void) { g_nptrext = 0; }
static void ptrext_set(const bcir_func *fn, uint32_t ptr_rid, uint32_t cnt_rid) {
  for (int k = 0; k < g_nptrext; k++)                       /* an existing binding -> overwrite */
    if (g_ptrext[k].fn == fn && g_ptrext[k].ptr_rid == ptr_rid) { g_ptrext[k].cnt_rid = cnt_rid; return; }
  if (g_nptrext < BCIR_MAX_PTREXT) {
    g_ptrext[g_nptrext].fn = fn; g_ptrext[g_nptrext].ptr_rid = ptr_rid;
    g_ptrext[g_nptrext].cnt_rid = cnt_rid; g_nptrext++;
  }
}
/* The recovered count-variable rid bound to a pointer rid (in `fn`), or 0 if the pointer has no extent.
 * (A real rid is never 0 -- the allocator starts at 100 -- so 0 is an unambiguous "no binding".) */
static uint32_t ptrext_get(const bcir_func *fn, uint32_t ptr_rid) {
  for (int k = 0; k < g_nptrext; k++)
    if (g_ptrext[k].fn == fn && g_ptrext[k].ptr_rid == ptr_rid) return g_ptrext[k].cnt_rid;
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
  CC_ENSURE(c->s, c->ns, c->cap_s);
  if(c->ns>=c->cap_s){ fail(c,"too many struct definitions"); return -1; }
  int my=c->ns++;                       /* claim our slot NOW: an inline aggregate member recurses into
                                         * p_struct_body and must take a LATER slot (and may realloc c->s). */
  sdef *S=&c->s[my]; S->nf=0; S->align=1; S->is_union=is_union;
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
    CC_ENSURE(c->ec,c->nec,c->cap_ec);
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
      CC_ENSURE(c->td,c->ntd,c->cap_td);
      if(c->ntd<c->cap_td){idcpy(c->td[c->ntd].name,&nm);c->td[c->ntd].ty=fp;c->td[c->ntd].sidx=-1;c->ntd++;}
      eat(c,";");
      return;
    }
    c->i=save;                                         /* not a funcptr declarator */
  }
  tok nm=adv(c);                                      /* the alias name */
  CC_ENSURE(c->td,c->ntd,c->cap_td);
  if(c->ntd<c->cap_td){idcpy(c->td[c->ntd].name,&nm);c->td[c->ntd].ty=ty;c->td[c->ntd].sidx=sidx;c->ntd++;}
  eat(c,";");
}

/* --- the IR builder ------------------------------------------------------ */
static uint32_t add_res(CC *c, bcir_domain dom, int elem, int count, int vol, int kind, const char *nm) {
  bcir_func *f=c->fn;
  if(f->n_res>=f->cap_res){                 /* grow geometrically -- no fixed resource ceiling */
    size_t nc=f->cap_res?f->cap_res*2:16; bcir_resource *nr=realloc(f->res,nc*sizeof *nr);
    if(!nr){fail(c,"oom");return 0;} f->res=nr; f->cap_res=nc;
  }
  bcir_resource *r=&f->res[f->n_res++]; memset(r,0,sizeof *r);   /* realloc slots are uninitialized */
  r->rid=c->rid++; r->domain=dom; r->elem_bytes=elem<1?1:elem; r->count=count<1?1:count;
  r->is_volatile=(uint8_t)vol; r->read_only=0; r->kind=(uint8_t)kind; r->agg[0]=0;
  snprintf(r->name,sizeof r->name,"%s",nm?nm:"");
  return r->rid;
}
static bcir_claim *new_claim(CC *c,const char *op,bcir_opcode opc) {
  bcir_func *f=c->fn;
  if(f->n_claims>=f->cap_claims){           /* grow geometrically -- no fixed claim ceiling */
    size_t nc=f->cap_claims?f->cap_claims*2:32; bcir_claim *ncl=realloc(f->claims,nc*sizeof *ncl);
    if(!ncl){fail(c,"oom");return NULL;} f->claims=ncl; f->cap_claims=nc;
  }
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
  if(f->n_calls>=f->cap_calls){ int nc=f->cap_calls?f->cap_calls*2:8;
    char (*np)[BCIR_CIR_NAME]=realloc(f->calls,(size_t)nc*sizeof *np);
    if(!np){fail(c,"oom");return;} f->calls=np; f->cap_calls=nc; }
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
 * in the same function share one global (dedup). The full spelling is kept in g_strtab so the emit can
 * inline it at any length; the resource name is just a short tag. */
static uint32_t intern_string(CC *c, const char *s, int n) {
  for (int k=0;k<g_nstr;k++)                                  /* dedup within the current function */
    if (g_strtab[k].fn==c->fn && (int)strlen(g_strtab[k].s)==n && !memcmp(g_strtab[k].s,s,(size_t)n))
      return g_strtab[k].rid;
  int elem = str_elem_size(s,n);                              /* element width (wide/UTF prefix) */
  int nunits = str_bytes(s,n)+1;                              /* code units incl. the NUL */
  char nm[BCIR_CIR_NAME]; snprintf(nm,sizeof nm,"__str%d",g_nstr);
  uint32_t rid = add_res(c,BCIR_DOM_RAM,elem,nunits,0,BCIR_RK_POINTER,nm);
  if (c->fn->n_res) c->fn->res[c->fn->n_res-1].read_only=1;   /* a read-only global */
  if (g_nstr<BCIR_MAX_STRLITS) { char *cp=(char*)malloc((size_t)n+1);
    if (cp){ memcpy(cp,s,(size_t)n); cp[n]=0;
      g_strtab[g_nstr].rid=rid; g_strtab[g_nstr].s=cp; g_strtab[g_nstr].fn=c->fn; g_nstr++; } }
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
  int s_ncalls=c->fn->n_calls, s_nenv=c->nenv, s_nstr=g_nstr;
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
  while(g_nstr>s_nstr){ g_nstr--; free(g_strtab[g_nstr].s); g_strtab[g_nstr].s=NULL; }
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
  uint32_t r=add_res(c,BCIR_DOM_RAM, fld->ptee_size?fld->ptee_size:4, 1, 0, BCIR_RK_POINTER, "");
  if(c->fn->n_res){ bcir_resource *t=&c->fn->res[c->fn->n_res-1];
    t->is_signed=(uint8_t)(fld->signd?1:0); t->is_float=(uint8_t)(fld->ptee_float?1:0);
    if(fld->ptee_sidx>=0) snprintf(t->agg,sizeof t->agg,"%s %s",
      c->s[fld->ptee_sidx].is_union?"union":"struct", c->s[fld->ptee_sidx].tag); }
  return r;
}

static uint32_t emit_member(CC *c, venv *base, const field *fld) {
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
  if(base->type.is_volatile){cl->domain=BCIR_DOM_MMIO;cl->lane=BCIR_LANE_H;cl->hazard=BCIR_HZ_BARRIERED;}
  if(fld->bit_w){uint32_t u=t;
    /* integer promotion (6.3.1.1) of a bitfield read, keyed on the BITFIELD WIDTH W (`fld->bit_w`):
     *   * W <= 32  -> promotes to int (int holds all W-bit values), so an UNSIGNED sub-int bitfield reads
     *                 as a SIGNED int (`bf < x` is a signed compare); W==32 unsigned stays unsigned.
     *   * W >  32  -> int can't hold it: a `_BitInt(N)` bitfield stays `_BitInt(N)` (does NOT promote --
     *                 matching Clang's `s.wide + s.wide == _BitInt(N)`); a standard wide field keeps its
     *                 declared 64-bit type. So a `_BitInt(N<=32)` bitfield promotes to int just like a
     *                 standard one -- VERIFIED == Clang via the `_Generic` differential. */
    if(fld->bit_width>0 && fld->bit_w>32) t=tempbi(c,fld->bit_width,fld->signd);   /* WIDE `_BitInt(N)` bitfield */
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
  if(base->type.is_volatile){cl->domain=BCIR_DOM_MMIO;cl->lane=BCIR_LANE_H;cl->hazard=BCIR_HZ_BARRIERED;}
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
  if(base->type.is_volatile){cl->domain=BCIR_DOM_MMIO;cl->lane=BCIR_LANE_H;cl->hazard=BCIR_HZ_BARRIERED;}
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
  if(base->type.is_volatile){cl->domain=BCIR_DOM_MMIO;cl->lane=BCIR_LANE_H;cl->hazard=BCIR_HZ_BARRIERED;}
  return t;
}
/* `a[i].field = val` on a DIRECT array-of-structs variable: store `sub->size` bytes at offsetof(sub),
 * STRIDING by the element (struct) size. The base is the array itself (member offset 0). Mirrors the
 * member-array strided store (imm = [field_off, field_size, _Bool-flag, stride]). */
static void store_index_field(CC *c, venv *base, uint32_t idx, const field *sub, uint32_t val) {
  bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
  if(!cl) return;
  cl->n_rd=3;cl->rd[0]=base->rid;cl->rd[1]=idx;cl->rd[2]=val;cl->n_imm=2;
  cl->imm[0]=sub->byte_off;cl->imm[1]=sub->size;cl->bounds=BCIR_BND_ASSUMED;
  if(sub->is_bool){cl->imm[2]=1;cl->n_imm=3;}
  if(cl->n_imm<3){cl->imm[2]=0;cl->n_imm=3;} cl->imm[3]=base->type.size; cl->n_imm=4;   /* stride imm[3] */
  if(base->type.is_volatile){cl->domain=BCIR_DOM_MMIO;cl->lane=BCIR_LANE_H;cl->hazard=BCIR_HZ_BARRIERED;}
}
/* `s.arr[i] = val` (member array, the element copy size == the element/stride size) OR `s.arr[i].field = val`
 * (member array-of-structs, the field copy size != the element stride): store at member_off + field_off,
 * striding by the element size. `sf` is the stored slot (the element FIELD for AOS, else the array element),
 * `soa` selects which (1 -> AOS field, stride = arr->size; 0 -> plain element, stride == copy size). */
static void store_member_index(CC *c, venv *base, const field *arr, uint32_t idx,
                               int soa, const field *sf, uint32_t val) {
  bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
  if(!cl) return;
  cl->n_rd=3;cl->rd[0]=base->rid;cl->rd[1]=idx;cl->rd[2]=val;cl->n_imm=2;
  cl->imm[0]= soa ? arr->byte_off+sf->byte_off : arr->byte_off; cl->imm[1]=sf->size;
  cl->bounds=BCIR_BND_ASSUMED;
  if(sf->is_bool){cl->imm[2]=1;cl->n_imm=3;}                      /* a _Bool element/field: normalize on store */
  if(soa){ if(cl->n_imm<3){cl->imm[2]=0;cl->n_imm=3;} cl->imm[3]=arr->size; cl->n_imm=4; }   /* stride imm[3] */
  if(base->type.is_volatile){cl->domain=BCIR_DOM_MMIO;cl->lane=BCIR_LANE_H;cl->hazard=BCIR_HZ_BARRIERED;}
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
  if (n_rid) { ptrext_set(c->fn, p_rid, n_rid); return; }   /* a STABLE integer-count Name -> bound BY NAME */
  if (cstart < 0) return;                                   /* not a recoverable alloc form at all */
  if (cend == cstart + 1 && c->t[cstart].k == T_ID) return; /* a (non-stable) bare Name -> by-name-or-nothing */
  if (!is_pure_range(c, cstart, cend)) return;              /* an impure expression count -> unmanaged */
  uint32_t ext = snapshot_extent(c, cstart, cend);          /* a pure EXPRESSION count -> SNAPSHOT it */
  if (ext) ptrext_set(c->fn, p_rid, ext);
}

/* The bounds contract for an indexed access (§5.12 bounds-promotion). A LOCAL/STATIC array OBJECT -- whose
 * extent is statically RECOVERABLE from the resource's element `count` -- is promoted from `assumed` to
 * `masked` (runtime-bounds-checked, the contract the quarantine handler discharges); a pointer base (extent
 * unknown) stays `assumed` -- UNLESS it carries a §5.12 recovered count (ptr_extent). Metadata only -- no
 * emit/behaviour change; `verify` already defaults to bounds. */
static bcir_bounds access_bnd(CC *c, uint32_t rid) {
  const bcir_resource *r = res_of(c->fn, rid);
  /* A known-extent ARRAY object (oracle `rt.kind=="array" and rt.count`): a LOCAL/STATIC array (kind !=
   * POINTER), OR a file-scope (static/const) global array -- the twin tags a global array POINTER for
   * index decay, but it carries its real, small element count, whereas a genuine pointer has the SYMBOLIC
   * pointee extent 1<<16 (locals/params) or count 1 (a pointer global / a malloc result). So an array is
   * any base with a small definite count > 1; the pointer extents (65536 / 1) are excluded here and the
   * recovered ones are masked via ptr_extent below. A string-LITERAL base stays assumed_safe (anonymous
   * read-only data -- the oracle excludes str_globals). */
  if (r && r->count > 1 && r->count != (1u << 16) && !strtab_lookup(rid)) return BCIR_BND_MASKED;
  if (ptrext_get(c->fn, rid)) return BCIR_BND_MASKED;   /* §5.12 a malloc/calloc pointer with a recovered count */
  return BCIR_BND_ASSUMED;
}
static uint32_t emit_index(CC *c, venv *base, uint32_t idx) {     /* base[idx] -- GEP load */
  const bcir_resource *br=res_of(c->fn,base->rid);
  uint32_t t;
  if(base->type.kind==2 && br && br->kind==BCIR_RK_SCALAR && br->count>1){   /* an ARRAY of pointers `T *a[N]`
    * (a SCALAR array with pointer-wide elements, NOT a pointer variable `T *p`): `a[i]` loads a pointer */
    t=add_res(c,BCIR_DOM_RAM,cc_abi(c)->pointer_size,1,0,BCIR_RK_POINTER,"");
    if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
      tr->ptr_depth=base->type.ptr_depth?base->type.ptr_depth:1;
      if(base->type.ptr_to_struct) snprintf(tr->agg,sizeof tr->agg,"%s %s",base->type.is_union?"union":"struct",base->type.tag);
      else if(base->type.size==0 && !base->type.is_float) tr->is_voidptr=1; }   /* a `void *` element */
  } else { int es=base->type.size?base->type.size:4;
    t=base->type.is_float ? tempf(c,es) : tempi(c,es,base->type.signd); }  /* float -> a float temp; else keep the sign */
  bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
  cl->n_rd=2;cl->rd[0]=base->rid;cl->rd[1]=idx;cl->n_wr=1;cl->wr[0]=t;cl->bounds=access_bnd(c,base->rid);
  return t;
}
/* Parse `[i]` (or `[i][j][k]`) on an array variable and Horner-flatten via its declared dims
 * (`v->type.adims`): `m[i][j]` on a `T m[A][B]` -> `i*B + j`. The cursor must be at the first `[`. */
static uint32_t array_index(CC *c, venv *v) {
  /* SNAPSHOT the env entry: the index p_expr below can declare locals (a `({...})` stmt-expr) and realloc
   * c->env[] -- the incoming `v`, a pointer into that array, would dangle after the move. Reading from the
   * by-value copy is byte-identical (only v->rid / v->type are read). */
  venv vsnap=*v; v=&vsnap;
  uint32_t idxs[3]; int ni=0;
  while(is(c,"[")){ c->i++; uint32_t ix=p_expr(c); eat(c,"]"); if(ni<3)idxs[ni++]=ix; }
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
  uint8_t r_signd=r->is_signed, r_float=r->is_float, r_plain_char=r->is_plain_char;
  char r_agg[sizeof r->agg]; snprintf(r_agg,sizeof r_agg,"%s",r->agg);
  uint32_t t;
  if(depth>1){                                     /* the pointee is itself a pointer (read pointer_size) */
    t=add_res(c,BCIR_DOM_RAM,base,1,0,BCIR_RK_POINTER,"");
    if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
      tr->is_signed=r_signd; tr->is_float=r_float; tr->ptr_depth=(uint8_t)(depth-1);
      snprintf(tr->agg,sizeof tr->agg,"%s",r_agg); }
  } else { t = r_float ? tempf(c,base) : tempi(c,base,r_signd);
    if(depth==1 && r_plain_char && c->fn->n_res)   /* a `char *` deref loads a plain `char` value */
      c->fn->res[c->fn->n_res-1].is_plain_char=1; }
  int rd_sz = depth>1 ? cc_abi(c)->pointer_size : base;
  bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
  cl->n_rd=1;cl->rd[0]=rid;cl->n_wr=1;cl->wr[0]=t;cl->bounds=BCIR_BND_ASSUMED;cl->n_imm=2;cl->imm[0]=0;cl->imm[1]=rd_sz;
  return t;
}
static uint32_t emit_deref(CC *c, venv *pv) {     /* *p -- a one-read dereference load (named pointer) */
  if(pv->type.is_volatile){                        /* MMIO: an ordered volatile load (unchanged) */
    int psz=pv->type.size?pv->type.size:4;
    uint32_t t=tempi(c,psz,pv->type.signd);
    bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
    cl->n_rd=1;cl->rd[0]=pv->rid;cl->n_wr=1;cl->wr[0]=t;cl->bounds=BCIR_BND_ASSUMED;
    cl->n_imm=2;cl->imm[0]=0;cl->imm[1]=psz;
    cl->domain=BCIR_DOM_MMIO;cl->lane=BCIR_LANE_H;cl->hazard=BCIR_HZ_BARRIERED;
    return t;
  }
  return emit_deref_rid(c,pv->rid);                /* depth-aware (the resource carries width + ptr_depth) */
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
      return emit_index(c,&b,ix);
    }
    if(is(c,"->")||is(c,".")){
      if(psidx<0){ fail(c,"member access through a pointer to a non-struct"); return ptr; }
      c->i++; tok fn=adv(c); sdef *S=&c->s[psidx]; int fi=-1;
      for(int i=0;i<S->nf;i++) if((int)strlen(S->f[i].name)==fn.n&&!strncmp(S->f[i].name,fn.s,fn.n)) fi=i;
      if(fi<0){ fail(c,"unknown field"); return ptr; }
      field mf=member_descend(c,S->f[fi]);           /* flatten any nested value-struct hops */
      venv b; memset(&b,0,sizeof b); b.rid=ptr; b.sidx=psidx; b.type.kind=1;   /* base = the loaded pointer */
      if(is(c,"(")){     /* funcptr-member call through the loaded pointer: `d->ops->fn(args)` (#fnptrchain) */
        c->i++; uint32_t args[BCIR_CLAIM_MAX_RD]; int na=0;
        if(!is(c,")")) for(;;){ uint32_t a=p_expr(c); if(na<BCIR_CLAIM_MAX_RD-1)args[na++]=a;
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
          cl->n_wr=1;cl->wr[0]=t;cl->n_imm=1;cl->imm[0]=1;}   /* imm0=1: base is a pointer -> `ptr->fn(args)` */
        return t;
      }
      if(mf.is_ptr && (is(c,"->")||is(c,".")||is(c,"["))){    /* another pointer hop: load it, recurse */
        ptr=emit_member(c,&b,&mf); psidx=mf.ptee_sidx; pfld=mf; continue;
      }
      if(mf.arr_count && is(c,"[")){ uint32_t ix=member_arr_index(c,&mf);
        field sub; if(elem_field(c,&mf,&sub)) return emit_member_index_field(c,&b,&mf,ix,&sub);
        if(c->failed) return 0;
        return emit_member_index(c,&b,&mf,ix); }
      return emit_member(c,&b,&mf);                   /* terminal member load through the loaded pointer */
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
  uint32_t args[BCIR_CLAIM_MAX_RD]; int na=0;
  if(!is(c,")")) for(;;){ uint32_t a=p_expr(c); if(na<BCIR_CLAIM_MAX_RD)args[na++]=a;
    if(is(c,",")){c->i++;continue;} break; }
  eat(c,")");
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
  uint32_t args[BCIR_CLAIM_MAX_RD]; int na=0;
  if(!is(c,")")) for(;;){ uint32_t a=p_expr(c); if(na<BCIR_CLAIM_MAX_RD-1)args[na++]=a;
    if(is(c,",")){c->i++;continue;} break; }
  eat(c,")");
  uint32_t t=fp_result_temp(c,&fv->type);   /* type by the funcptr's captured return -> a signed return reads back signed */
  bcir_claim *cl=new_claim(c,"c.call.indirect",BCIR_OP_GEM_DISPATCH);
  if(cl){cl->n_rd=(uint8_t)(na+1);cl->rd[0]=fv->rid;for(int k=0;k<na;k++)cl->rd[k+1]=args[k];
    cl->n_wr=1;cl->wr[0]=t;}
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
  c->i++; uint32_t args[BCIR_CLAIM_MAX_RD]; int na=0;
  /* SEG7: an order-taking fence (`__atomic_thread_fence`/`atomic_thread_fence`) routes its KIND by the
   * first arg's order value -- peeked HERE, BEFORE the arg is lowered, so the arg's const claim (the
   * value rail) is still emitted in sequence, exactly as the oracle does (digest = [const, fence]). */
  if(ordered && kind==AK_FENCE && !is(c,")")) op=fence_order_op(c);
  if(!is(c,")")) for(;;){uint32_t a=p_expr(c);if(na<BCIR_CLAIM_MAX_RD)args[na++]=a;if(is(c,",")){c->i++;continue;}break;}
  eat(c,")");
  uint32_t t=temp(c,4);
  if(!strncmp(op,"c.c11atom.cas",13) && c->fn->n_res) c->fn->res[c->fn->n_res-1].is_bool=1;  /* compare_exchange -> _Bool */
  bcir_claim *cl=new_claim(c,op,oc); if(!cl)return t;
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
  int cap=first.n+16; char *buf=(char*)malloc((size_t)cap);
  if(!buf){*out_n=0;return NULL;}
  memcpy(buf,first.s,(size_t)first.n); int len=first.n;
  while(isk(c,T_STR)){ tok nx=adv(c); int need=len+1+nx.n+1;
    if(need>cap){ cap=need*2; char *nb=(char*)realloc(buf,(size_t)cap); if(!nb){free(buf);*out_n=0;return NULL;} buf=nb; }
    buf[len++]=' '; memcpy(buf+len,nx.s,(size_t)nx.n); len+=nx.n; }
  buf[len]=0; *out_n=len; return buf;
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
      c->i++; uint32_t args[BCIR_CLAIM_MAX_RD]; int na=0;
      if(!is(c,")")) for(;;){ uint32_t a=p_expr(c); if(na<BCIR_CLAIM_MAX_RD-1)args[na++]=a;
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
        cl->n_wr=1;cl->wr[0]=t;cl->n_imm=1;cl->imm[0]=arrow;}
      return t;
    }
    field mf=member_descend(c,S->f[fi]);        /* nested `o.in.v` -> one flattened-offset load */
    if(mf.is_ptr && (is(c,"->")||is(c,".")||is(c,"["))){   /* deref-through a loaded pointer field (#fieldderef) */
      uint32_t ptr=emit_member(c,v,&mf);        /* load the pointer field, then chain through the loaded ptr */
      return postfix_ptr_chain(c,ptr,mf.ptee_sidx,mf); }
    if(mf.arr_count && is(c,"[")){ uint32_t ix=member_arr_index(c,&mf);   /* s.arr[i] / s.m[i][j] load */
      field sub; if(elem_field(c,&mf,&sub)) return emit_member_index_field(c,v,&mf,ix,&sub);   /* arr[i].field */
      if(c->failed) return 0;
      return emit_member_index(c,v,&mf,ix); }
    return emit_member(c,v,&mf);
  }
  if(is(c,"[")){                                /* L3: base[i] / m[i][j] (row-major flatten) */
    uint32_t idxs[3]; int ni=0;
    while(is(c,"[")){ c->i++; uint32_t ix=p_expr(c); eat(c,"]"); if(ni<3)idxs[ni++]=ix; }
    const bcir_resource *vr=res_of(c->fn,v->rid);    /* a multi-dim VLA -> RUNTIME dim strides (no c.const) */
    int vla=(vr && vr->vla_ndims>0);
    uint32_t lin=idxs[0];
    for(int d=1; d<ni; d++){                     /* lin = lin*dim + idxs[d]  (Horner) */
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
    if(v->sidx>=0 && (is(c,".")||is(c,"->"))){    /* a[i].field on a DIRECT array-of-structs (strided load) */
      field sub; if(aos_elem_field(c,v,&sub)) return emit_index_field(c,v,lin,&sub);
      if(c->failed) return 0; }
    return emit_index(c,v,lin);
  }
  return v->rid;
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
      rid=intern_string(c, cb?cb:st.s, cb?cn:st.n); free(cb); }
    else rid=intern_string(c,st.s,st.n);   /* full spelling kept in g_strtab; dedup; cap lifted */
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
          size=(long long)(str_bytes(sp,sn)+1)*str_elem_size(sp,sn); free(cb); }
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
      uint32_t r=p_call(c,&id);
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
  if(ty->kind==0 && ty->bit_width>0){               /* a `_BitInt(N)` cast target -- the exact spelling (faithful) */
    snprintf(o,n,"c.cast:%s_BitInt(%d)",ty->signd?"":"unsigned ",ty->bit_width); return; }
  const char *nm=ty->is_complex ? (ty->size==8?"float _Complex":ty->size>16?"long double _Complex":"double _Complex")
                : ty->is_bool ? "_Bool"           /* a bool cast normalizes any nonzero (full value) to 1 */
                : ty->is_float ? (ty->size==4?"float":ty->size>8?"long double":"double")  /* >8: extended (matches the oracle's `long double`) */
                : signed_int ? (ty->size==1?"int8_t":ty->size==2?"int16_t":ty->size==8?"int64_t":"int32_t")
                : ty->size==1?"uint8_t":ty->size==2?"uint16_t":ty->size==8?"uint64_t":"uint32_t";
  if(ty->kind==2) snprintf(o,n,"c.cast:%s *",nm); else snprintf(o,n,"c.cast:%s",nm);
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
        uint32_t t=add_res(c,BCIR_DOM_RAM, v->type.size?v->type.size:4, 1,0,BCIR_RK_POINTER,"");  /* &p (T*) -> T** */
        if(c->fn->n_res){ bcir_resource *tr=&c->fn->res[c->fn->n_res-1];
          tr->is_signed=(uint8_t)(v->type.signd?1:0); tr->is_float=(uint8_t)(v->type.is_float?1:0);
          tr->ptr_depth=(uint8_t)((v->type.kind==2?(v->type.ptr_depth?v->type.ptr_depth:1):0)+1);
          if(v->type.kind==1||v->type.ptr_to_struct) snprintf(tr->agg,sizeof tr->agg,"%s %s",v->type.is_union?"union":"struct",v->type.tag); }
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
      if(isk(c,T_ID)){ tok pid=*pk(c); venv *pvp=lookup(c,&pid);
        if(pvp){ c->i++; venv pvsnap=*pvp; venv *pv=&pvsnap;   /* SNAPSHOT: the `+ i` index p_expr below can realloc c->env[] */
          if(is(c,"+")){ c->i++; uint32_t idx=p_expr(c); eat(c,")"); return emit_index(c,pv,idx); }
          if(is(c,")")){ c->i++; return emit_deref(c,pv); } } }
      c->i=save;
    } else if(isk(c,T_ID)){ tok pid=*pk(c); venv *pv=lookup(c,&pid);
      if(pv){ c->i++; return emit_deref(c,pv); } }  /* *p (no sub-parse between lookup and use) */
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
        const bcir_resource *vr=res_of(c->fn,v);
        /* The result temp carries the target's signedness, so a signed (sub-int) target keeps its sign
         * even when the cast value is used directly -- `(signed char)(-5)` stays -5, and `(int)u` reads
         * back signed (an arithmetic `>>`). A float -> signed-int conversion additionally needs a SIGNED
         * cast operator (float -> unsigned is UB / target-divergent). */
        int f2s = vr && vr->is_float && !ty.is_float && ty.kind!=2 && ty.signd && ty.size>0
                  && ty.bit_width==0;                                  /* a `_BitInt` keeps its exact spelling */
        uint32_t r = ty.is_complex ? tempc(c, ty.size)                /* a _Complex cast -> a complex temp */
                   : ty.is_float ? tempf(c, ty.size)                  /* a float cast -> a float temp */
                   : ty.kind==2 ? temp(c, cc_abi(c)->pointer_size)    /* a pointer cast */
                   : ty.bit_width>0 ? tempbi(c, ty.bit_width, ty.signd?1:0)   /* a C23 `_BitInt(N)` cast */
                                : tempi(c, ty.size?ty.size:4, ty.signd?1:0);   /* an integer cast */
        if(ty.is_bool && !ty.is_float && ty.kind!=2 && c->fn->n_res)
          c->fn->res[c->fn->n_res-1].is_bool=1;    /* a bool cast -> a _Bool temp (normalizes to 0/1) */
        char op[BCIR_CIR_NAME]; cast_name(&ty,f2s,op,sizeof op);
        bcir_claim *cl=new_claim(c,op,BCIR_OP_ADD);
        if(cl){cl->n_rd=1;cl->rd[0]=v;cl->n_wr=1;cl->wr[0]=r;} return r;
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
  CC_ENSURE(c->env,c->nenv,c->cap_env);
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
  uint32_t rid=add_res(c,BCIR_DOM_RAM,g->ty.size,g->count,0,kind,g->name);
  if(c->fn->n_res) c->fn->res[c->fn->n_res-1].read_only=1;   /* a global, not a local */
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
static void store_member(CC *c, venv *base, const field *f, uint32_t val){
  bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
  if(cl){cl->n_rd=2;cl->rd[0]=base->rid;cl->rd[1]=val;cl->n_imm=2;cl->imm[0]=f->byte_off;cl->imm[1]=f->size;
    cl->bounds=BCIR_BND_ASSUMED;
    if(f->is_bool){cl->imm[2]=1;cl->n_imm=3;}                         /* a _Bool member normalizes on store */
    if(base->type.is_volatile){cl->domain=BCIR_DOM_MMIO;cl->lane=BCIR_LANE_H;cl->hazard=BCIR_HZ_BARRIERED;}}
}
/* Emit a BITFIELD member store `base.field = val` (#bfassignexpr): read the storage unit (`access_bytes`
 * spanned bytes, into a pow2 temp), insert the masked bits (c.bf.set), store the unit's spanned bytes back
 * -- the same bf.set machinery the STATEMENT-form bitfield store uses, factored out so the value path reuses
 * it. The reload that yields the assignment's VALUE is a plain emit_member (its c.load + c.bf.get). */
static void store_member_bf(CC *c, venv *base, const field *f, uint32_t val){
  int absz=f->access_bytes<=4?4:8;
  uint32_t unit=temp(c,absz);
  bcir_claim *ld=new_claim(c,"c.load",BCIR_OP_LOAD);
  if(ld){ld->n_rd=1;ld->rd[0]=base->rid;ld->n_wr=1;ld->wr[0]=unit;ld->n_imm=2;ld->imm[0]=f->byte_off;ld->imm[1]=f->access_bytes;ld->bounds=BCIR_BND_ASSUMED;
    if(base->type.is_volatile){ld->domain=BCIR_DOM_MMIO;ld->lane=BCIR_LANE_H;ld->hazard=BCIR_HZ_BARRIERED;}}
  uint32_t nu=temp(c,absz);
  bcir_claim *bs=new_claim(c,"c.bf.set",BCIR_OP_ADD);
  if(bs){bs->n_rd=2;bs->rd[0]=unit;bs->rd[1]=val;bs->n_wr=1;bs->wr[0]=nu;bs->n_imm=2;bs->imm[0]=f->bit_off;bs->imm[1]=f->bit_w;}
  bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
  if(cl){cl->n_rd=2;cl->rd[0]=base->rid;cl->rd[1]=nu;cl->n_imm=3;cl->imm[0]=f->byte_off;cl->imm[1]=f->access_bytes;cl->imm[2]=2;
    cl->bounds=BCIR_BND_ASSUMED;                                      /* a bitfield UNIT store: `_v` takes the unit's full type */
    if(base->type.is_volatile){cl->domain=BCIR_DOM_MMIO;cl->lane=BCIR_LANE_H;cl->hazard=BCIR_HZ_BARRIERED;}}
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
      if(is_eq){                                       /* plain: rhs FIRST, then store, then RELOAD */
        c->i++; uint32_t rhs=p_assign(c);
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
    store_index_field(c,v,idx,&sub,tmp);
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
      store_member_index(c,v,&f,idx,soa,sf,tmp);
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
      *out=emit_member(c,v,&f);                         /* reload: a bitfield re-reads via bf.get (#bfassignexpr) */
      return 1;
    }
    char ch=op->s[0]; c->i++;                           /* compound: read cur, binop, store, value = stored */
    uint32_t cur=emit_member(c,v,&f); uint32_t rhs=p_assign(c);
    const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
    uint32_t tmp=binop_result(c,suf,cur,rhs); char o[BCIR_CIR_NAME]; snprintf(o,sizeof o,"c.bin.%s",suf);
    bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
    if(f.bit_w) store_member_bf(c,v,&f,tmp); else store_member(c,v,&f,tmp);     /* a nested bitfield: bf.set */
    /* the value is the STORED (narrowed) value: a sub-int leaf truncates on store, so RE-READ the same
     * member (#narrowcompound); a full-width leaf needs no re-read (oracle: lv.ct.size < rt.size). A BITFIELD
     * narrows to its BIT width (the byte-size test misses it) -- always RE-READ it. */
    const bcir_resource *trr=res_of(c->fn,tmp);
    *out = (f.bit_w || (int)f.size < (int)(trr?trr->elem_bytes:4)) ? emit_member(c,v,&f) : tmp;
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
    uint32_t cur=emit_member(c,v,&f);                     /* read OLD (a fresh declared temp -- no snapshot) */
    uint32_t one=incdec_emit_const1(c);
    uint32_t nw=binop_result(c,suf,cur,one); char o[BCIR_CIR_NAME]; snprintf(o,sizeof o,"c.bin.%s",suf);
    bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=one;b->n_wr=1;b->wr[0]=nw;}
    if(f.bit_w) store_member_bf(c,v,&f,nw); else store_member(c,v,&f,nw);
    if(!prefix){ *out=cur; return 1; }                    /* postfix: the OLD value */
    const bcir_resource *nr=res_of(c->fn,nw);             /* prefix: the STORED new value -- re-read if it */
    *out = (f.bit_w || (int)f.size < (int)(nr?nr->elem_bytes:4)) ? emit_member(c,v,&f) : nw;   /* narrows */
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
      c->i+=2; uint32_t rhs=p_assign(c);
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
      uint32_t tmp=binop_result(c,suf,v->rid,rhs); char o[BCIR_CIR_NAME]; snprintf(o,sizeof o,"c.bin.%s",suf);
      bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=v->rid;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
      bcir_claim *cp=new_claim(c,"c.copy",BCIR_OP_ADD); if(cp){cp->n_rd=1;cp->rd[0]=tmp;cp->n_wr=1;cp->wr[0]=v->rid;}
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
      return emit_member(c,mbase,&mf);              /* reload: a bitfield re-reads via bf.get (#bfassignexpr) */
    }
    char ch=op->s[0]; c->i++;                       /* compound `OP=`: read-once, binop, store, value = stored */
    uint32_t cur=emit_member(c,mbase,&mf);          /* read FIRST (matches the oracle's claim order) */
    uint32_t rhs=p_assign(c);
    const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
    uint32_t tmp=binop_result(c,suf,cur,rhs); char o[BCIR_CIR_NAME]; snprintf(o,sizeof o,"c.bin.%s",suf);
    bcir_claim *b=new_claim(c,o,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
    if(mf.bit_w) store_member_bf(c,mbase,&mf,tmp); else store_member(c,mbase,&mf,tmp);       /* a bitfield: bf.set */
    /* the value of a compound is the STORED (narrowed) value: a sub-int member (`unsigned char c;
     * (s.c += v)`) truncates on store, so RE-READ it (#narrowcompound) -- the plain `=` path above
     * already re-reads; a full-width member needs no re-read (oracle: lv.ct.size < rt.size), so the
     * existing #memassignexpr cases stay byte-identical. A BITFIELD narrows to its BIT width (its size is
     * the full underlying type, so the byte-size test misses it) -- always RE-READ it (oracle:
     * narrows = lv.bit_width or lv.ct.size < rt.size). */
    const bcir_resource *trr=res_of(c->fn,tmp);
    return (mf.bit_w || (int)mf.size < (int)(trr?trr->elem_bytes:4)) ? emit_member(c,mbase,&mf) : tmp;
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
static int designator_chain(CC *c, sdef *S, int *off, int *size, int *bit_w, int *bit_off, int *top_fi){
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
  return 0;
}
static void agg_init_at(CC *c, uint32_t rid, int sidx, int base_off, int do_zinit);   /* fwd: mutual recursion */
/* A NESTED braced initializer for an array member inside a struct/union init -- `struct S s = { {e0,e1,
 * ..}, n };` (the C twin of lower._init_subagg). Stores each element with an OFFSET-based member c.store
 * at `base_off + idx*es` (so it composes through the enclosing struct's `= {0}` baseline); positional
 * entries advance a cursor, `[i]=` jumps it, gaps zero-fill. Element float-ness/width is carried by the
 * stored value's resource (the member-store emit), exactly like a `s.arr[i] = v` element write. */
static void subagg_init(CC *c, uint32_t rid, int base_off, int es, int is_bool) {
  eat(c,"{");
  int cursor=0;
  while(!is(c,"}")&&!isk(c,T_END)&&!c->failed){
    int idx=cursor;
    if(is(c,"[")){ c->i++; idx=(int)ce_expr(c,0); eat(c,"]"); eat(c,"="); }   /* [const-index] = */
    uint32_t v=p_expr(c);
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
 * baseline). Positional entries advance a cursor, `[i]=` jumps it, gaps zero-fill (§6.7.10). */
static void subagg_init_md_inner(CC *c, uint32_t rid, int base_off, const int *dims, int nd, int es, int is_bool);
/* Depth-guarded wrapper: subagg_init_md self-recurses per nested-row brace, so a deeply nested
 * multi-dim `{{{...}}}` initializer would exhaust the stack. Bump/check depth once per row level. */
static void subagg_init_md(CC *c, uint32_t rid, int base_off, const int *dims, int nd, int es, int is_bool) {
  if(ENTER_REC(c)){ LEAVE_REC(c); return; }
  subagg_init_md_inner(c, rid, base_off, dims, nd, es, is_bool); LEAVE_REC(c);
}
static void subagg_init_md_inner(CC *c, uint32_t rid, int base_off, const int *dims, int nd, int es, int is_bool) {
  eat(c,"{");
  int stride=es;                                          /* row stride = product(dims[1:]) * es */
  for(int d=1; d<nd; d++) stride*=dims[d];
  int cursor=0;
  while(!is(c,"}")&&!isk(c,T_END)&&!c->failed){
    int idx=cursor;
    if(is(c,"[")){ c->i++; idx=(int)ce_expr(c,0); eat(c,"]"); eat(c,"="); }   /* [const-index] = */
    if(nd>1 && is(c,"{")){                                /* an outer dim takes a nested ROW brace */
      subagg_init_md(c, rid, base_off+idx*stride, dims+1, nd-1, es, is_bool);
    } else {                                              /* innermost dim: a scalar element store */
      uint32_t v=p_expr(c);
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
    if(is(c,".")||is(c,"[")){                           /* a (possibly nested) designator */
      if(designator_chain(c,S,&off,&size,&bit_w,&bit_off,&top_fi)) return;
      if(top_fi>=0&&top_fi<S->nf){ fbool=S->f[top_fi].is_bool; abytes=S->f[top_fi].access_bytes; }
      eat(c,"=");
    } else if(cursor<S->nf){ field *F=&S->f[cursor];    /* positional: the cursor-th member */
      off=F->byte_off; size=F->size; bit_w=F->bit_w; bit_off=F->bit_off; fbool=F->is_bool; abytes=F->access_bytes;
    } else skip=1;                                      /* past the last member -> parse but do not store */
    off += base_off;                                   /* shift into the enclosing object (nested aggregate) */
    if(!skip && is(c,"{")){                             /* a NESTED brace: an aggregate (array OR struct) member */
      field *AF = (top_fi>=0 && top_fi<S->nf) ? &S->f[top_fi] : NULL;
      if(AF && AF->arr_count>0 && AF->elem_sidx>=0){      /* an ARRAY-OF-STRUCTS member: `{ {a,b}, {c,d} }` */
        subagg_init_struct(c, rid, off, AF->elem_sidx, AF->size); cursor=top_fi+1; if(is(c,",")) c->i++; continue; }
      if(AF && AF->arr_count>0){ subagg_init(c, rid, off, AF->size, AF->is_bool); cursor=top_fi+1; if(is(c,",")) c->i++; continue; }
      if(AF && AF->sidx>=0){ agg_init_at(c, rid, AF->sidx, off, 0); cursor=top_fi+1; if(is(c,",")) c->i++; continue; }
      fail(c,"nested initializer for a non-aggregate member"); return;
    }
    uint32_t v=p_expr(c); uint32_t val=v;
    if(skip){ cursor=top_fi+1; if(is(c,",")) c->i++; continue; }
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
    subagg_init_md(c, rid, 0, dims, la_nd, es, ty->is_bool);   /* descend the nested ROW braces */
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
      return; }
    if(v && is_compound_op(tat(c,c->i+1))){                                             /* name OP= expr */
      char ch=tat(c,c->i+1)->s[0];
      if(v->type.kind==2 && (ch=='+'||ch=='-')){       /* pointer arithmetic: p += n / p -= n (verbatim) */
        c->i+=2; uint32_t rhs=p_expr(c);
        char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.ptr%s",ch=='+'?"add":"sub");
        bcir_claim *cl=new_claim(c,op,BCIR_OP_ADD); if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=rhs;cl->n_wr=1;cl->wr[0]=v->rid;}
        return; }
      c->i+=2; uint32_t rhs=p_expr(c);                  /* scalar:  name = name OP expr  (bin op + copy) */
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
      uint32_t tmp=binop_result(c,suf,v->rid,rhs); char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.bin.%s",suf);
      bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=v->rid;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
      bcir_claim *cp=new_claim(c,"c.copy",BCIR_OP_ADD); if(cp){cp->n_rd=1;cp->rd[0]=tmp;cp->n_wr=1;cp->wr[0]=v->rid;}
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
    uint32_t val;
    if(is_compound_op(&c->t[c->i])){ char ch=c->t[c->i].s[0]; c->i++;
      uint32_t cur=emit_index(c,&b,idx); uint32_t rhs=p_expr(c);
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
      uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.bin.%s",suf);
      bcir_claim *bb=new_claim(c,op,oc); if(bb){bb->n_rd=2;bb->rd[0]=cur;bb->rd[1]=rhs;bb->n_wr=1;bb->wr[0]=tmp;}
      val=tmp;
    } else { if(!eat(c,"="))return; val=p_expr(c); }
    bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
    if(cl){cl->n_rd=3;cl->rd[0]=ptr;cl->rd[1]=idx;cl->rd[2]=val;cl->bounds=BCIR_BND_ASSUMED;}
    return;
  }
  if(!(is(c,"->")||is(c,"."))){ fail(c,"expected ->/./[ after a pointer field"); return; }
  if(psidx<0){ fail(c,"member store through a pointer to a non-struct"); return; }
  c->i++; tok fn=adv(c); sdef *S=&c->s[psidx]; int fi=-1;
  for(int i=0;i<S->nf;i++) if((int)strlen(S->f[i].name)==fn.n&&!strncmp(S->f[i].name,fn.s,fn.n)) fi=i;
  if(fi<0){ fail(c,"unknown field"); return; }
  field f=member_descend(c,S->f[fi]);
  venv b; memset(&b,0,sizeof b); b.rid=ptr; b.sidx=psidx; b.type.kind=1;   /* base = the loaded pointer */
  if(f.is_ptr && (is(c,"->")||is(c,".")||is(c,"["))){  /* another pointer hop: load it, recurse */
    uint32_t nptr=emit_member(c,&b,&f); store_through_ptr(c,nptr,f.ptee_sidx,f); return;
  }
  uint32_t val;                                        /* terminal member store through the loaded pointer */
  if(is_compound_op(&c->t[c->i])){ char ch=c->t[c->i].s[0]; c->i++;
    uint32_t cur=emit_member(c,&b,&f); uint32_t rhs=p_expr(c);
    const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
    uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.bin.%s",suf);
    bcir_claim *bb=new_claim(c,op,oc); if(bb){bb->n_rd=2;bb->rd[0]=cur;bb->rd[1]=rhs;bb->n_wr=1;bb->wr[0]=tmp;}
    val=tmp;
  } else { if(!eat(c,"="))return; val=p_expr(c); }
  bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
  if(cl){cl->n_rd=2;cl->rd[0]=b.rid;cl->rd[1]=val;cl->n_imm=2;cl->imm[0]=f.byte_off;cl->imm[1]=f.size;cl->bounds=BCIR_BND_ASSUMED;}
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
        if(sw<sizeof sig && c->fpdefs_w+sw<sizeof c->fpdefs){ memcpy(c->fpdefs+c->fpdefs_w,sig,sw); c->fpdefs_w+=sw; }
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
        ptrext_set(c->fn,arid,ext);   /* §5.12 mask `a[i]` against the recovered runtime extent */
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
        ptrext_set(c->fn,arid,ext_total);   /* §5.12 mask `a[i][j]` against the recovered total runtime extent */
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
      uint32_t rid=add_res(c, ty.is_volatile?BCIR_DOM_MMIO:BCIR_DOM_RAM,
                           is_arr?arr_elem:(ty.kind==2?ty.size:(ty.kind==1?c->s[si].size:ty.size)),
                           is_arr?(arr?arr:1):(ty.kind==2?(1<<16):1), ty.is_volatile, rk, nb);
      if(is_arr && ty.kind==1){ bcir_resource *ar=&c->fn->res[c->fn->n_res-1];   /* an ARRAY-OF-STRUCTS local
        * `struct P a[N]`: a SCALAR-kind array of struct-sized elements; carry the struct tag so the decl
        * emits `struct P a[N]` and `a[i].field` strides by the element struct (the venv keeps `si`). */
        snprintf(ar->agg,BCIR_CIR_NAME,"%s %s",ty.is_union?"union":"struct",ty.tag); }
      else if(arr && ty.kind==2){ bcir_resource *ar=&c->fn->res[c->fn->n_res-1];   /* an array of pointers `T *a[N]`: a
        * SCALAR array of pointer-wide elements; the decl + `a[i]` load/store carry the pointee (void* for now) */
        ar->ptr_depth=ty.ptr_depth?ty.ptr_depth:1;
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
          if(f->n_statics>=f->cap_statics){ int nc=f->cap_statics?f->cap_statics*2:4;
            bcir_static *ns=realloc(f->statics,(size_t)nc*sizeof *ns); if(ns){f->statics=ns;f->cap_statics=nc;} }
          if(f->n_statics<f->cap_statics){ idcpy(f->statics[f->n_statics].name,&nm);
            f->statics[f->n_statics].init=init; f->n_statics++; } }
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
            subagg_init_md(c, rid, 0, la_dims, la_nd, ty.size, ty.is_bool);
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
    if(is(c,"(")){ c->i++;                              /* *(p) or *(p + i) */
      if(isk(c,T_ID)){ tok pid=*pk(c); venv *pvp=lookup(c,&pid);
        if(pvp){ c->i++; pvsnap=*pvp; pv=&pvsnap;
          if(is(c,"+")){ c->i++; idx=p_expr(c); has_idx=1; if(eat(c,")")) ok=1; }
          else if(is(c,")")){ c->i++; ok=1; } } } }
    else if(isk(c,T_ID)){ tok pid=*pk(c); venv *pvp=lookup(c,&pid); if(pvp){ c->i++; pvsnap=*pvp; pv=&pvsnap; ok=1; } }   /* *p */
    if(ok && pv && (is_compound_op(&c->t[c->i]) ||
                    (c->t[c->i].k==T_PUN && c->t[c->i].n==1 && c->t[c->i].s[0]=='='))){
      int sz = (pv->type.ptr_depth>1) ? cc_abi(c)->pointer_size : (pv->type.size?pv->type.size:4); uint32_t val;
      /* `*pp = q` through a `T**` stores a full pointer (pointer_size), not the base scalar width */
      if(is_compound_op(&c->t[c->i])){                  /* *p OP= expr  ->  load, bin op, store */
        char ch=c->t[c->i].s[0]; c->i++;
        uint32_t cur = has_idx ? emit_index(c,pv,idx) : emit_deref(c,pv);
        uint32_t rhs=p_expr(c);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
        uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.bin.%s",suf);
        bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
        val=tmp;
      } else { c->i++; val=p_expr(c); }                 /* *p = expr */
      bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
      if(cl){
        if(has_idx){ cl->n_rd=3; cl->rd[0]=pv->rid; cl->rd[1]=idx; cl->rd[2]=val; }   /* *(p+i) == p[i] */
        else { cl->n_rd=2; cl->rd[0]=pv->rid; cl->rd[1]=val; cl->n_imm=2; cl->imm[0]=0; cl->imm[1]=sz; }
        cl->bounds=BCIR_BND_ASSUMED;
        if(pv->type.is_volatile){cl->domain=BCIR_DOM_MMIO;cl->lane=BCIR_LANE_H;cl->hazard=BCIR_HZ_BARRIERED;}
      }
      eat(c,";"); return;
    }
    /* general deref-store: `**pp = v` / `**pp OP= v` / `*(<expr>) = v` -- store through a pointer RVALUE
     * (not a simple named `*p`). Mirrors the general deref-load: parse the operand, then store at *base. */
    c->i=save+1;                                       /* re-parse from just after the leading `*` */
    uint32_t base=p_unary(c); const bcir_resource *br=res_of(c->fn,base);
    if(br && br->kind==BCIR_RK_POINTER && (is_compound_op(&c->t[c->i]) ||
        (c->t[c->i].k==T_PUN && c->t[c->i].n==1 && c->t[c->i].s[0]=='='))){
      int depth=br->ptr_depth?br->ptr_depth:1;
      int sz=(depth>1)?cc_abi(c)->pointer_size:(br->elem_bytes?(int)br->elem_bytes:4); uint32_t val;
      if(is_compound_op(&c->t[c->i])){                 /* **pp OP= expr -> load through base, bin, store */
        char ch=c->t[c->i].s[0]; c->i++;
        uint32_t cur=emit_deref_rid(c,base); uint32_t rhs=p_expr(c);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
        uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.bin.%s",suf);
        bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
        val=tmp;
      } else { c->i++; val=p_expr(c); }                /* **pp = expr */
      bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
      if(cl){ cl->n_rd=2; cl->rd[0]=base; cl->rd[1]=val; cl->n_imm=2; cl->imm[0]=0; cl->imm[1]=sz; cl->bounds=BCIR_BND_ASSUMED; }
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
        uint32_t ptr=emit_member(c,v,&f);         /* load the pointer field, then store through the loaded ptr */
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
        uint32_t cur=emit_member(c,v,&f);          /* the current field value (loaded first) */
        uint32_t rhs=p_expr(c);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
        uint32_t tmp=binop_result(c,suf,cur,rhs); char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.bin.%s",suf);
        bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
        val=tmp;
      } else { if(!eat(c,"="))return; val=p_expr(c); }
      if(f.bit_w){
        /* a bitfield store: read the storage unit (`access_bytes` spanned bytes, into a pow2 temp), insert
         * the masked bits (c.bf.set), store the unit's spanned bytes back. */
        int absz=f.access_bytes<=4?4:8;
        uint32_t unit=temp(c,absz);
        bcir_claim *ld=new_claim(c,"c.load",BCIR_OP_LOAD);
        if(ld){ld->n_rd=1;ld->rd[0]=v->rid;ld->n_wr=1;ld->wr[0]=unit;ld->n_imm=2;ld->imm[0]=f.byte_off;ld->imm[1]=f.access_bytes;ld->bounds=BCIR_BND_ASSUMED;
          if(v->type.is_volatile){ld->domain=BCIR_DOM_MMIO;ld->lane=BCIR_LANE_H;ld->hazard=BCIR_HZ_BARRIERED;}}
        uint32_t nu=temp(c,absz);
        bcir_claim *bs=new_claim(c,"c.bf.set",BCIR_OP_ADD);
        if(bs){bs->n_rd=2;bs->rd[0]=unit;bs->rd[1]=val;bs->n_wr=1;bs->wr[0]=nu;bs->n_imm=2;bs->imm[0]=f.bit_off;bs->imm[1]=f.bit_w;}
        val=nu;
      }
      bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
      if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=val;cl->n_imm=2;cl->imm[0]=f.byte_off;cl->imm[1]=f.bit_w?f.access_bytes:f.size;
        cl->bounds=BCIR_BND_ASSUMED;
        if(f.bit_w){cl->imm[2]=2;cl->n_imm=3;}      /* a bitfield UNIT store: `_v` takes the unit's full type */
        else if(f.is_bool){cl->imm[2]=1;cl->n_imm=3;}    /* a _Bool member: emit `_Bool _v` so the store normalizes */
        if(v->type.is_volatile){cl->domain=BCIR_DOM_MMIO;cl->lane=BCIR_LANE_H;cl->hazard=BCIR_HZ_BARRIERED;}}
      eat(c,";");return;}
    /* L3: array element store  a[idx] = expr  /  a[idx] OP= expr  (driver buffer fill / scatter). */
    if(v&&tat(c,c->i+1)->k==T_PUN&&tat(c,c->i+1)->n==1&&tat(c,c->i+1)->s[0]=='['){
      int as_start=c->i;                                /* roll-back point: `a[i]` may be a VALUE, not a store */
      size_t as_res=c->fn->n_res,as_cl=c->fn->n_claims; uint32_t as_rid=c->rid,as_cid=c->cid,as_clc=c->cl_ctr;
      c->i++; uint32_t idx=array_index(c,v); uint32_t val;   /* a[i] / m[i][j] (Horner-flattened) */
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
        if(v->type.is_volatile){cl->domain=BCIR_DOM_MMIO;cl->lane=BCIR_LANE_H;cl->hazard=BCIR_HZ_BARRIERED;}}
      eat(c,";");return;}
    if(v&&tat(c,c->i+1)->k==T_PUN&&tat(c,c->i+1)->n==1&&tat(c,c->i+1)->s[0]=='='){
      tok tnm=c->t[c->i]; c->i+=2; int ist=c->i; uint32_t val=p_expr(c); int ien=c->i;
      bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD);if(cl){cl->n_rd=1;cl->rd[0]=val;cl->n_wr=1;cl->wr[0]=v->rid;}
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
      c->i+=2; uint32_t rhs=p_expr(c);
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
      uint32_t tmp=binop_result(c,suf,v->rid,rhs); char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.bin.%s",suf);
      bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=v->rid;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
      bcir_claim *cp=new_claim(c,"c.copy",BCIR_OP_ADD); if(cp){cp->n_rd=1;cp->rd[0]=tmp;cp->n_wr=1;cp->wr[0]=v->rid;}
      eat(c,";");return;}}
  if(p_incdec(c)){eat(c,";");return;}    /* ++i / i++ / --i / i-- as a statement */
  (void)p_expr(c);eat(c,";");
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
  size_t snap_res=c->fn->n_res, snap_cl=c->fn->n_claims;
  uint32_t snap_rid=c->rid, snap_cid=c->cid, snap_clctr=c->cl_ctr;
  int snap_ncalls=c->fn->n_calls, snap_nenv=c->nenv, snap_nstr=g_nstr;
  while(!is(c,"}")&&!isk(c,T_END)&&!c->failed){
    last_save=c->i;                      /* remember the start + the lowering state before each statement */
    snap_res=c->fn->n_res; snap_cl=c->fn->n_claims; snap_rid=c->rid; snap_cid=c->cid; snap_clctr=c->cl_ctr;
    snap_ncalls=c->fn->n_calls; snap_nenv=c->nenv; snap_nstr=g_nstr;
    p_stmt(c);
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
    while(g_nstr>snap_nstr){ g_nstr--; free(g_strtab[g_nstr].s); g_strtab[g_nstr].s=NULL; }
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
    result=p_expr(c); if(is(c,";")) c->i++;
  }
  eat(c,"}"); c->nenv=env_mark; eat(c,")");   /* pop the scope, close `)` */
  return result;
}

static int p_func(CC *c, bcir_func *fn) {
  c->fn=fn; c->nenv=0; c->n_vlaext=0;
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
      if(sw<sizeof sig && c->fpdefs_w+sw<sizeof c->fpdefs){ memcpy(c->fpdefs+c->fpdefs_w,sig,sw); c->fpdefs_w+=sw; }
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
    }
    }
    char pb[BCIR_CIR_NAME]; idcpy(pb,&pn);
    int rk=ty.kind==2?BCIR_RK_POINTER:ty.kind==1?BCIR_RK_AGGREGATE:BCIR_RK_SCALAR;
    uint32_t rid=add_res(c, ty.is_volatile?BCIR_DOM_MMIO:BCIR_DOM_RAM,
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
    if(fn->n_params>=fn->cap_params){ int nc=fn->cap_params?fn->cap_params*2:4;
      bcir_param *np=realloc(fn->params,(size_t)nc*sizeof *np); if(np){fn->params=np;fn->cap_params=nc;} }
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
    if(c->n_protos>=c->cap_protos){ int nc=c->cap_protos?c->cap_protos*2:8;
      void *np=realloc(c->protos,(size_t)nc*sizeof *c->protos);
      if(!np){fail(c,"oom");return 1;} c->protos=np; c->cap_protos=nc; }
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
    if(sw<sizeof sig && c->tudefs_w+sw<sizeof c->tudefs){
      memcpy(c->tudefs+c->tudefs_w,sig,sw); c->tudefs_w+=sw; }
    return 2;
  }
  if(!eat(c,"{"))return 1;
  scan_mutations(c,c->i);   /* §5.12 extent-stability pre-pass over the body (cursor is just past `{`) */
  for(int k=0;k<c->n_vlaext;k++){          /* §5.12 resolve the deferred VLA-param extent bindings: bind `a` to
    * `n` ONLY when `n` is STABLE -- unmutated in the body and not address-taken (the _bind_extent stable-Name
    * gate, evaluated now that scan_mutations has populated the mutation table). A mutated-size param stays
    * assumed_safe -- matching the oracle so the BCIR_CHK count is identical. */
    if(mut_body(c,&c->vlaext[k].cnt_tok)==0 && !mut_addr(c,&c->vlaext[k].cnt_tok))
      ptrext_set(c->fn,c->vlaext[k].ptr_rid,c->vlaext[k].cnt_rid);
  }
  while(!is(c,"}")&&!isk(c,T_END)&&!c->failed) p_stmt(c);
  eat(c,"}");
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
  if(ty->kind==2){ char stars[10]; int d=ty->ptr_depth?ty->ptr_depth:1, si=0;   /* depth `*`s: `T**` -> ` **` */
    stars[si++]=' '; for(int k=0;k<d&&si<9;k++) stars[si++]='*'; stars[si]=0;
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
  const char *lit=strtab_lookup(rid);                /* a string literal -> its full spelling, inline */
  if(lit) return lit;                                /* (returned directly, so length is not capped) */
  return uniq_local(f,rid,buf);                      /* a named local (disambiguated) / a `t<rid>` temp */
}
/* The C type to declare a temporary / local with: float/double for a floating value, else the integer
 * scalar's true fixed-width type from its (width, signedness) -- so the backend does signed-vs-unsigned
 * and width-correct arithmetic (the old flat uint32 model did not). Non-scalar temps (pointer / address
 * paths) stay uint32 here; their declaration goes through the pointee type. */
/* A short rotating-buffer pool for type spellings that must format a number (`_BitInt(N)`), so a single
 * statement that mentions a couple of `_BitInt` types each get a stable string. (4 slots: more than any
 * one emitted statement needs.) */
static const char *bitint_spelling(int bit_width,int signd){
  static char ring[4][24]; static int rr=0; char *b=ring[rr++&3];
  snprintf(b,sizeof ring[0],"%s_BitInt(%d)",signd?"":"unsigned ",bit_width); return b; }
static const char *tty(const bcir_func *f,uint32_t rid){
  const bcir_resource *r=res_of(f,rid);
  if(!r) return "uint32_t";
  if(r->is_valist) return "va_list";   /* a variadic cursor object -- opaque, declared `va_list ap;` */
  if(r->bit_width>0) return bitint_spelling(r->bit_width,r->is_signed);   /* C23 `_BitInt(N)` -- faithful spelling */
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
static const char *decl_ty(const bcir_func *f,uint32_t rid,char *buf,size_t n){
  const bcir_resource *r=res_of(f,rid);
  if(r && r->kind==BCIR_RK_POINTER){
    const char *base = r->is_voidptr ? "void"   /* a `void *` pointee (`&&L`, void-pointee local): no width/sign */
      : r->agg[0] ? r->agg
      : r->is_float ? (r->elem_bytes==4?"float":r->elem_bytes>8?"long double":"double")
      : r->elem_bytes==1?(r->is_signed?"int8_t":"uint8_t") : r->elem_bytes==2?(r->is_signed?"int16_t":"uint16_t")
      : r->elem_bytes==8?(r->is_signed?"int64_t":"uint64_t") : (r->is_signed?"int32_t":"uint32_t");
    char stars[10]; int d=r->ptr_depth?r->ptr_depth:1, si=0;   /* depth `*`s: `T**` at depth 2 */
    stars[si++]=' '; for(int k=0;k<d&&si<9;k++) stars[si++]='*'; stars[si]=0;
    snprintf(buf,n,"%s%s",base,stars);
  } else snprintf(buf,n,"%s",tty(f,rid));
  return buf;
}
/* The `&`-or-not prefix that turns a BASE resource into a `(char *)`-castable pointer (mirrors the oracle's
 * _base_ptr): a POINTER value decays to itself (`(char *)p`), and so does an ARRAY name (a scalar resource
 * with count>1 -- a DIRECT array-of-structs `a[i].f`, `(char *)a`); a struct/scalar VALUE is addressed
 * (`(char *)&s`). NOTE: the addrof path keeps its own `&`-for-array rule (it addresses `&arr[i]`). */
static const char *base_amp(const bcir_resource *br){
  if(br && br->kind==BCIR_RK_POINTER) return "";
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
    if(br && br->count>1){                                 /* a known-extent local/static array (constant N) */
      snprintf(buf,bn,"%s(%u, %s, %lluu, \"%s:%s\")",chk,(unsigned)cl->rd[0],rname(f,cl->rd[1],ib),
               (unsigned long long)br->count,f->name,rname(f,cl->rd[0],nb));
      return buf;
    }
  }
  return rname(f,cl->rd[1],buf);
}
static size_t emit_func(const bcir_func *f,char *o,size_t on){
  size_t w=0; char a[BCIR_CIR_NAME],b[BCIR_CIR_NAME],d[BCIR_CIR_NAME],e[BCIR_CIR_NAME],ty[64],tb[80],gb[192];
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
      if(sx>=0) w+=snprintf(o+EO,on-EO,"  static uint32_t %s = %lluu;\n",nm,(unsigned long long)f->statics[sx].init);
      else if(r->is_funcptr&&r->agg[0]) w+=snprintf(o+EO,on-EO,"  %s %s;\n",r->agg,nm);   /* a funcptr local: `__bcir_fpN f;` */
      else if(r->kind==BCIR_RK_AGGREGATE&&r->agg[0]) w+=snprintf(o+EO,on-EO,"  %s %s%s;\n",r->agg,nm,r->zinit?" = {0}":"");
      else if(r->kind==BCIR_RK_SCALAR&&r->count>1&&r->is_voidptr) w+=snprintf(o+EO,on-EO,"  void *%s[%u]%s;\n",nm,r->count,r->zinit?" = {0}":"");  /* an array of `void *` */
      else if(r->kind==BCIR_RK_SCALAR&&r->count>1&&r->agg[0]&&!r->ptr_depth) w+=snprintf(o+EO,on-EO,"  %s %s[%u]%s;\n",r->agg,nm,r->count,r->zinit?" = {0}":"");  /* an ARRAY-OF-STRUCTS local `struct P a[N]` */
      else if(r->kind==BCIR_RK_SCALAR&&r->count>1) w+=snprintf(o+EO,on-EO,"  %s %s[%u]%s;\n",tty(f,r->rid),nm,r->count,r->zinit?" = {0}":"");  /* a local array */
      else if(r->kind==BCIR_RK_POINTER)               /* a pointer local: `T *p` (the pointee carries width/sign) */
        w+=snprintf(o+EO,on-EO,"  %s%s;\n",decl_ty(f,r->rid,tb,sizeof tb),nm);
      else w+=snprintf(o+EO,on-EO,"  %s %s;\n",tty(f,r->rid),nm);}}
  int depth=1, lstk[64], nls=0, lctr=0;   /* loop-id stack + counter for the `continue` labels */
  #define IND() do{ for(int _k=0;_k<depth;_k++) w+=snprintf(o+EO,on-EO,"  "); }while(0)
  for(size_t i=0;i<f->n_claims&&w<on-160;i++){const bcir_claim *cl=&f->claims[i];
    /* L6 control-flow markers (rendered as braces) */
    if(!strcmp(cl->op,"c.if")){IND();w+=snprintf(o+EO,on-EO,"if (%s) {\n",rname(f,cl->rd[0],a));depth++;continue;}
    if(!strcmp(cl->op,"c.else")){depth--;IND();w+=snprintf(o+EO,on-EO,"} else {\n");depth++;continue;}
    if(!strcmp(cl->op,"c.endif")){depth--;IND();w+=snprintf(o+EO,on-EO,"}\n");continue;}
    if(!strcmp(cl->op,"c.loop")){IND();w+=snprintf(o+EO,on-EO,"while (1) {\n");depth++;
      if(nls<64)lstk[nls++]=lctr++;continue;}
    if(!strcmp(cl->op,"c.loop.test")){IND();w+=snprintf(o+EO,on-EO,"if (!%s) break;\n",rname(f,cl->rd[0],a));continue;}
    if(!strcmp(cl->op,"c.cont.tgt")){IND();w+=snprintf(o+EO,on-EO,"__cont_%d: ;\n",nls?lstk[nls-1]:0);continue;}
    if(!strcmp(cl->op,"c.endloop")){depth--;IND();w+=snprintf(o+EO,on-EO,"}\n");if(nls)nls--;continue;}
    if(!strcmp(cl->op,"c.vladecl")){IND();   /* a 1-D stack VLA, declared IN-BODY: `<elem> a[__bcir_extK];` */
      w+=snprintf(o+EO,on-EO,"%s %s[%s];\n",tty(f,cl->wr[0]),rname(f,cl->wr[0],a),rname(f,cl->rd[0],b));continue;}
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
      w+=snprintf(o+EO,on-EO,"%s %s = %s %s %s;\n",decl_ty(f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),binop_c(cl->op+6),rname(f,cl->rd[1],b));
    else if(!strncmp(cl->op,"c.labeladdr:",12))            /* `&&L` -- a label's address as a `void *` (GNU) */
      w+=snprintf(o+EO,on-EO,"%s %s = &&%s;\n",decl_ty(f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),cl->op+12);
    else if(!strncmp(cl->op,"c.fconst:",9))                /* a floating constant -> its literal spelling */
      w+=snprintf(o+EO,on-EO,"%s %s = %s;\n",tty(f,cl->wr[0]),rname(f,cl->wr[0],d),cl->op+9);
    else if(!strncmp(cl->op,"c.cconst:",9))                /* <complex.h> imaginary unit -> verbatim token */
      w+=snprintf(o+EO,on-EO,"%s %s = %s;\n",tty(f,cl->wr[0]),rname(f,cl->wr[0],d),cl->op+9);
    else if(!strcmp(cl->op,"c.un.creal"))                  /* GNU __real__ z -- the real part (an element float) */
      w+=snprintf(o+EO,on-EO,"%s %s = __real__ %s;\n",tty(f,cl->wr[0]),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a));
    else if(!strcmp(cl->op,"c.un.cimag"))                  /* GNU __imag__ z -- the imaginary part */
      w+=snprintf(o+EO,on-EO,"%s %s = __imag__ %s;\n",tty(f,cl->wr[0]),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a));
    else if(!strncmp(cl->op,"c.un.",5))                    /* `-`/`~` keep the operand width (long stays 64) */
      w+=snprintf(o+EO,on-EO,"%s %s = (%s%s);\n",tty(f,cl->wr[0]),rname(f,cl->wr[0],d),unop_c(cl->op+5),rname(f,cl->rd[0],a));
    else if(!strncmp(cl->op,"c.cast:",7))                  /* (type)operand -- width / float / pointer cast */
      w+=snprintf(o+EO,on-EO,"%s %s = (%s)%s;\n",decl_ty(f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),cl->op+7,rname(f,cl->rd[0],a));  /* decl_ty: a pointer-snapshot cast keeps `T *` */
    else if(!strcmp(cl->op,"c.select"))                    /* ternary: cond ? then : els -- the select's own
                                                            * (signed/unsigned) type, not a hardcoded
                                                            * uint32_t (see the c.const note below). */
      w+=snprintf(o+EO,on-EO,"%s %s = (%s ? %s : %s);\n",tty(f,cl->wr[0]),rname(f,cl->wr[0],d),
                  rname(f,cl->rd[0],a),rname(f,cl->rd[1],b),rname(f,cl->rd[2],e));
    else if(!strcmp(cl->op,"c.const")){
      /* declare the constant with its OWN type, not a hardcoded uint32_t: a bare integer literal (e.g.
       * `0` in `x < 0`) is signed (int), so emitting `uint32_t = 0u` made a signed comparison promote to
       * unsigned (`int32_t < uint32_t` -> unsigned) -- a miscompile. The literal's (width, signedness)
       * was already recorded on the temp (lit_int_type); render the matching type + suffix. */
      const bcir_resource *cr=res_of(f,cl->wr[0]); int cs=cr&&cr->is_signed;
      w+=snprintf(o+EO,on-EO,"%s %s = %llu%s;\n",tty(f,cl->wr[0]),rname(f,cl->wr[0],d),
                  (unsigned long long)cl->imm[0], cs?"":"u"); }
    else if(!strcmp(cl->op,"c.sizeof.vla"))                 /* runtime `sizeof a` of a VLA: extent × sizeof(elem).
                                                            * HARDCODE the literal `size_t` (NOT tty(), which
                                                            * returns "uint64_t" for an 8-byte unsigned scalar):
                                                            * the oracle emits literal `size_t` (scalar('size_t')),
                                                            * so the two rails would diverge byte-for-byte. */
      w+=snprintf(o+EO,on-EO,"size_t %s = (size_t)((size_t)%s * %lld);\n",rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),(long long)cl->imm[0]);
    else if(!strcmp(cl->op,"c.copy")){
      if(is_named_local(f,cl->wr[0])||is_global_ref(f,cl->wr[0])||is_param_ref(f,cl->wr[0])) w+=snprintf(o+EO,on-EO,"%s = %s;\n",rname(f,cl->wr[0],d),rname(f,cl->rd[0],a));
      else w+=snprintf(o+EO,on-EO,"%s %s = %s;\n",decl_ty(f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a));   /* decl_ty: a copied pointer temp keeps `T *` */
    }else if(!strcmp(cl->op,"c.load")){
      const bcir_resource *br=res_of(f,cl->rd[0]); long long off=cl->n_imm?cl->imm[0]:0;
      if(cl->n_rd==2 && cl->n_imm){       /* s.arr[i] / a[i].f: load at base + off + idx*stride, copy `es` bytes */
        const char *amp=base_amp(br); long long es=cl->n_imm>1?cl->imm[1]:4;
        long long stride=cl->n_imm>2?cl->imm[2]:es;   /* array-of-structs `arr[i].field`: stride sizeof(elem) != es */
        /* the temp carries the element's (width, signedness): memcpy es bytes into it so a signed sub-int
         * element reads sign-extended (the zero-extending uint32 form dropped the sign). */
        w+=snprintf(o+EO,on-EO,"%s %s; memcpy(&%s, (const char *)%s%s + %lld + (size_t)%s * %lld, %lld);\n",
          tty(f,cl->wr[0]),rname(f,cl->wr[0],d),rname(f,cl->wr[0],d),amp,rname(f,cl->rd[0],a),off,rname(f,cl->rd[1],b),stride,es); }
      else if(cl->n_rd==2) w+=snprintf(o+EO,on-EO,"%s %s = %s[%s];\n",decl_ty(f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),guard_idx(f,cl,gb,sizeof gb,0));  /* READ guard; decl_ty: an array-of-pointers element load is `T *` */
      else if(cl->domain==BCIR_DOM_MMIO)
        w+=snprintf(o+EO,on-EO,"uint32_t %s = *(volatile uint32_t *)((const volatile char *)%s + %lld);\n",rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),off);
      else { const char *amp=(br&&br->kind==BCIR_RK_POINTER)?"":"&"; long long fsz=cl->n_imm>1?cl->imm[1]:4;
        /* a plain member load: memcpy fsz bytes into the typed temp so a signed sub-int member sign-extends */
        w+=snprintf(o+EO,on-EO,"%s %s; memcpy(&%s, (const char *)%s%s + %lld, %lld);\n",decl_ty(f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),rname(f,cl->wr[0],d),amp,rname(f,cl->rd[0],a),off,fsz); }  /* decl_ty: a pointer member load is `T *t` */
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
        w+=snprintf(o+EO,on-EO,"{ %s _v = %s; memcpy((char *)%s%s + %lld + (size_t)%s * %lld, &_v, %lld); }\n",
          vt,rname(f,cl->rd[2],d),amp,rname(f,cl->rd[0],a),off,rname(f,cl->rd[1],b),stride,es); }
      else if(cl->domain==BCIR_DOM_MMIO)
        w+=snprintf(o+EO,on-EO,"((volatile uint32_t *)%s)[%s] = %s;\n",rname(f,cl->rd[0],a),rname(f,cl->rd[1],b),rname(f,cl->rd[2],d));
      else
        w+=snprintf(o+EO,on-EO,"%s[%s] = %s;\n",rname(f,cl->rd[0],a),guard_idx(f,cl,gb,sizeof gb,1),rname(f,cl->rd[2],d));  /* WRITE guard: an OOB store fails-fast, never clamps */
    }else if(!strcmp(cl->op,"c.store")){          /* L8: member store -> memcpy `size` bytes */
      const bcir_resource *br=res_of(f,cl->rd[0]); long long off=cl->imm[0]; long long sz=cl->n_imm>1?cl->imm[1]:4;
      if(cl->domain==BCIR_DOM_MMIO)
        w+=snprintf(o+EO,on-EO,"*(volatile uint32_t *)((volatile char *)%s + %lld) = %s;\n",rname(f,cl->rd[0],a),off,rname(f,cl->rd[1],b));
      else { const char *amp=(br&&br->kind==BCIR_RK_POINTER)?"":"&";
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
                      :(vr&&vr->kind==BCIR_RK_POINTER)?decl_ty(f,cl->rd[1],tb,sizeof tb)
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
                    tty(f,cl->wr[0]),rname(f,cl->wr[0],d),cast,rname(f,cl->rd[0],a),off,mask,sfx,sbit,sfx,sbit,sfx);
      } else
        w+=snprintf(o+EO,on-EO,"%s %s = (%s >> %lld) & %llu%s;\n",
                    tty(f,cl->wr[0]),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),off,mask,sfx); }
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
      const char *pt=decl_ty(f,cl->wr[0],tb,sizeof tb);
      const bcir_resource *br=res_of(f,cl->rd[0]);
      const char *amp=(br&&br->kind==BCIR_RK_POINTER)?"":"&";   /* a pointer base decays (`(char*)s`), a value
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
      const char *rty=(wr&&wr->kind==BCIR_RK_POINTER)?"void *":tty(f,cl->wr[0]);
      w+=snprintf(o+EO,on-EO,"%s %s = %s(",rty,rname(f,cl->wr[0],d),cl->op+12);
      for(int k=0;k<cl->n_rd;k++) w+=snprintf(o+EO,on-EO,"%s%s",k?", ":"",rname(f,cl->rd[k],a));
      w+=snprintf(o+EO,on-EO,");\n"); }
    else if(!strncmp(cl->op,"c.call.libm.void:",17)){   /* a void external (free) -> a verbatim call statement */
      w+=snprintf(o+EO,on-EO,"%s(",cl->op+17);
      for(int k=0;k<cl->n_rd;k++) w+=snprintf(o+EO,on-EO,"%s%s",k?", ":"",rname(f,cl->rd[k],a));
      w+=snprintf(o+EO,on-EO,");\n"); }
    else if(!strncmp(cl->op,"c.call.extern:",14)){  /* a printf/scanf-family external variadic -> verbatim */
      w+=snprintf(o+EO,on-EO,"%s %s = %s(",tty(f,cl->wr[0]),rname(f,cl->wr[0],d),cl->op+14);
      for(int k=0;k<cl->n_rd;k++) w+=snprintf(o+EO,on-EO,"%s%s",k?", ":"",rname(f,cl->rd[k],a));
      w+=snprintf(o+EO,on-EO,");\n"); }
    else if(!strncmp(cl->op,"c.call.tu:",10)){      /* a PROTOTYPED cross-TU callee (Phase 3 linking):
                                                     * verbatim, external linkage -- the prelude declares
                                                     * it; the host LINKER resolves it */
      if(cl->n_wr==0) w+=snprintf(o+EO,on-EO,"%s(",cl->op+10);
      else w+=snprintf(o+EO,on-EO,"%s %s = %s(",tty(f,cl->wr[0]),rname(f,cl->wr[0],d),cl->op+10);
      for(int k=0;k<cl->n_rd;k++) w+=snprintf(o+EO,on-EO,"%s%s",k?", ":"",rname(f,cl->rd[k],a));
      w+=snprintf(o+EO,on-EO,");\n"); }
    else if(!strncmp(cl->op,"c.call.builtin:",15)){  /* a GCC/Clang integer builtin -> emitted verbatim */
      w+=snprintf(o+EO,on-EO,"%s %s = __builtin_%s(",tty(f,cl->wr[0]),rname(f,cl->wr[0],d),cl->op+15);
      for(int k=0;k<cl->n_rd;k++) w+=snprintf(o+EO,on-EO,"%s%s",k?", ":"",rname(f,cl->rd[k],a));
      w+=snprintf(o+EO,on-EO,");\n"); }
    else if(!strncmp(cl->op,"c.call.vaarg:",13)){   /* va_arg(ap, T) -- pull the next variadic argument */
      w+=snprintf(o+EO,on-EO,"%s %s = va_arg(%s, %s);\n",
        decl_ty(f,cl->wr[0],tb,sizeof tb),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),cl->op+13); }
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
      const char *dty=(rr&&rr->kind==BCIR_RK_AGGREGATE&&rr->agg[0])?rr->agg:tty(f,cl->wr[0]);
      w+=snprintf(o+EO,on-EO,"%s %s = bcir_%s(",dty,rname(f,cl->wr[0],d),cl->op+7);
      for(int k=0;k<cl->n_rd;k++) w+=snprintf(o+EO,on-EO,"%s%s",k?", ":"",rname(f,cl->rd[k],a));
      w+=snprintf(o+EO,on-EO,");\n"); }
    else if(!strcmp(cl->op,"c.call.indirect")){    /* rd[0] is the function pointer; rd[1..] the args */
      w+=snprintf(o+EO,on-EO,"%s %s = %s(",tty(f,cl->wr[0]),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a));  /* result typed by the funcptr's return */
      for(int k=1;k<cl->n_rd;k++) w+=snprintf(o+EO,on-EO,"%s%s",k>1?", ":"",rname(f,cl->rd[k],b));
      w+=snprintf(o+EO,on-EO,");\n"); }
    else if(!strncmp(cl->op,"c.call.imember:",15)){   /* o->fn(args): funcptr struct member */
      const char *sep=(cl->n_imm&&cl->imm[0])?"->":".";
      w+=snprintf(o+EO,on-EO,"%s %s = %s%s%s(",tty(f,cl->wr[0]),rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),sep,cl->op+15);  /* result typed by the funcptr's return */
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
  int count=1;
  while(is(c,"[")){ c->i++; count = isk(c,T_INT)?(int)adv(c).v:0; eat(c,"]"); }
  if(is(c,"=")){ c->i++;
    if(is(c,"{")){ c->i++; int d=1; while(d>0&&!isk(c,T_END)&&!c->failed){ if(is(c,"{"))d++; else if(is(c,"}"))d--; c->i++; } }
    else (void)ce_expr(c,0);
  }
  eat(c,";");
  CC_ENSURE(c->gv, c->ngv, c->cap_gv);
  if(c->ngv<c->cap_gv){ idcpy(c->gv[c->ngv].name,&nm); c->gv[c->ngv].ty=ty; c->gv[c->ngv].count=count; c->ngv++; }
}

/* --- public entry -------------------------------------------------------- */
int bcir_cfront_compile_target(const char *src, const char *target, bcir_cfront_result *out) {
  static CC c;
  /* `c` is a reused static: preserve the grown parser-state arrays across compiles (counts reset to
   * 0 below, the buffers are reused + grown as needed) so we neither leak per compile nor re-allocate. */
  sdef *sv_s=c.s; int sv_cs=c.cap_s; tdef *sv_td=c.td; int sv_ctd=c.cap_td;
  econst *sv_ec=c.ec; int sv_cec=c.cap_ec; gvar *sv_gv=c.gv; int sv_cgv=c.cap_gv;
  venv *sv_env=c.env; int sv_cenv=c.cap_env;
  void *sv_pr=c.protos; int sv_cpr=c.cap_protos;   /* the prototype table survives the memset (reused) */
  /* free any PRIOR unit before the entry memset zeroes out->unit.funcs: a reused `out` (a static result
   * compiled again without an intervening bcir_cfront_free) would otherwise leak its previous function
   * array + every per-func sub-array. bcir_cfront_free is a no-op on a zero-initialised out (funcs=NULL,
   * n_funcs=0 -> free(NULL)), so this is safe on the first call. */
  bcir_cfront_free(out);
  memset(&c,0,sizeof c); memset(out,0,sizeof *out);
  c.s=sv_s; c.cap_s=sv_cs; c.td=sv_td; c.cap_td=sv_ctd; c.ec=sv_ec; c.cap_ec=sv_cec;
  c.gv=sv_gv; c.cap_gv=sv_cgv; c.env=sv_env; c.cap_env=sv_cenv;
  c.protos=sv_pr; c.cap_protos=sv_cpr;              /* n_protos stays 0 from the memset (fresh unit) */
  c.abi = bcir_abi_by_name(target);     /* the target data model (NULL name -> host LP64) */
  if(!c.abi){ snprintf(out->diag,sizeof out->diag,"unknown target '%s'",target?target:""); return 1; }
  c.rid=100; c.cid=1000; c.unit=&out->unit;
  strtab_reset();                       /* fresh string-literal table per translation unit */
  ptrext_reset();                       /* fresh §5.12 ptr_extent map per translation unit */
  lex(&c,src);
  if(c.tok_overflow){                     /* Bug B: more than MAXTOK tokens -> a clean fail (fallback), NOT
                                           * a silent truncation + partial mis-compile. The oracle has no
                                           * token cap but its recursion/size guards route the same oversized
                                           * input to fallback, so both rails agree (neither mis-compiles). */
    snprintf(out->diag,sizeof out->diag,"input too large"); return 1; }
  while(!isk(&c,T_END)&&!c.failed){       /* no fixed function ceiling -- the unit list grows */
    /* A1.3: a leading C23 `[[unsequenced]]`/`[[reproducible]]` (or any `[[...]]`) attribute precedes a
     * function/global. Consume it here and carry the value-neutral hint flag into the next p_func (the
     * emit drops it). No run -> repro stays 0, every existing item undisturbed. */
    int lead_repro=0; c23_attrs(&c,&lead_repro);
    if(c.failed) break;
    if(try_top_decl(&c)) continue;       /* typedef / enum / struct|union defs, interleaved */
    if(isk(&c,T_END)||c.failed) break;
    if(looks_global(&c)){ p_global(&c); continue; }   /* a file-scope global (lookup table) */
    if(out->unit.n_funcs>=out->unit.cap_funcs){       /* grow the function list geometrically */
      int nc=out->unit.cap_funcs?out->unit.cap_funcs*2:8;
      bcir_func *nf=realloc(out->unit.funcs,(size_t)nc*sizeof *nf);
      if(!nf){snprintf(out->diag,sizeof out->diag,"oom");return 1;}
      memset(nf+out->unit.cap_funcs,0,(size_t)(nc-out->unit.cap_funcs)*sizeof *nf);  /* fresh slots */
      out->unit.funcs=nf; out->unit.cap_funcs=nc;     /* NB: grow only here, never during p_func */
    }
    bcir_func *fn=&out->unit.funcs[out->unit.n_funcs]; /* res/claims/params/calls/statics grow lazily */
    c.rid=100+out->unit.n_funcs*1000; c.cid=1000+out->unit.n_funcs*1000;
    int pfr=p_func(&c,fn);
    if(pfr==2){                             /* a PROTOTYPE: recorded in c.protos/c.tudefs, no function --
                                             * discard the scratch fn (params were collected into it) */
      free(fn->res); free(fn->claims); free(fn->params); free(fn->calls); free(fn->statics);
      memset(fn,0,sizeof *fn);
      continue;
    }
    if(pfr){snprintf(out->diag,sizeof out->diag,"%s",c.err);
      /* free the in-progress (uncounted) func's sub-arrays: it was never folded into n_funcs, so
       * bcir_cfront_free's `i<n_funcs` loop would never reach it -> a per-parse-failure leak. */
      free(fn->res); free(fn->claims); free(fn->params); free(fn->calls); free(fn->statics);
      memset(fn,0,sizeof *fn);
      return 1;}
    fn->reproducible=(uint8_t)lead_repro;   /* the C23 hint consumed just above (A1.3); emit drops it */
    out->unit.n_funcs++;
  }
  if(c.failed){snprintf(out->diag,sizeof out->diag,"%s",c.err);return 1;}
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
      if(f->n_calls>=f->cap_calls){ int nc=f->cap_calls?f->cap_calls*2:8;
        char (*np)[BCIR_CIR_NAME]=realloc(f->calls,(size_t)nc*sizeof *np);
        if(!np){snprintf(out->diag,sizeof out->diag,"oom");return 1;} f->calls=np; f->cap_calls=nc; }
      snprintf(f->calls[f->n_calls++],BCIR_CIR_NAME,"%s",callee);
    }
  }
  out->ok=bcir_verify_unit(&out->unit,out->diag,sizeof out->diag);
  /* C.2 verified-C attestation: stamp the emitted C with its R-law status + R13 digest + the unit's
   * derived link flags (B1; so --emit-c is self-describing about what it links -- a comment, stripped
   * on re-parse). The link_flags line mirrors the oracle's C.2 attestation. */
  const bcir_func *entry = out->unit.n_funcs ? &out->unit.funcs[out->unit.n_funcs-1] : NULL;
  char lflags[256]; bcir_cfront_link_flags(&out->unit, lflags, sizeof lflags);
  size_t w=snprintf(out->emitted,sizeof out->emitted,
    "/* BCIR verified-C attestation (C.2) -- generated by bcir_cfront, do not edit.\n"
    " *   R1-R8 + R18  %s\n"
    " *   R9 plan / R10-R11 pack  checked in the compile->execute loop\n"
    " *   R12 lowering-contract  support preserved (emit Clang-behaviour-equivalent)\n"
    " *   R13 provenance digest  %016llx\n"
    " *   R17 accuracy  exact (integer / Q-fixed, 0 ULP)\n"
    " *   link_flags  %s\n */\n",
    out->ok?"clean":"DIRTY", entry?(unsigned long long)bcir_provenance_digest(entry):0ull,
    lflags[0]?lflags:"-");
  if(c.fpdefs_w && w<sizeof out->emitted-c.fpdefs_w-1)   /* synthesized funcptr-param typedefs (prelude) */
    w+=snprintf(out->emitted+w,sizeof out->emitted-w,"%.*s",(int)c.fpdefs_w,c.fpdefs);
  if(c.tudefs_w && w<sizeof out->emitted-c.tudefs_w-1)   /* extern declarations for cross-TU callees */
    w+=snprintf(out->emitted+w,sizeof out->emitted-w,"%.*s",(int)c.tudefs_w,c.tudefs);
  for(int i=0;i<out->unit.n_funcs && w<sizeof out->emitted-256;i++){
    w+=emit_func(&out->unit.funcs[i],out->emitted+w,sizeof out->emitted-w);
    if(i+1<out->unit.n_funcs) w+=snprintf(out->emitted+w,sizeof out->emitted-w,"\n");
  }
  return 0;
}

/* The host-ABI entry (the default target): byte-identical to the layout before --target existed. */
int bcir_cfront_compile(const char *src, bcir_cfront_result *out) {
  return bcir_cfront_compile_target(src, NULL, out);
}

void bcir_cfront_free(bcir_cfront_result *out){
  for(int i=0;i<out->unit.n_funcs;i++){ bcir_func *f=&out->unit.funcs[i];
    free(f->res); free(f->claims); free(f->params); free(f->calls); free(f->statics); }
  free(out->unit.funcs);
  out->unit.funcs=NULL; out->unit.n_funcs=0; out->unit.cap_funcs=0;
}

void bcir_cfront_summary(const bcir_unit *u,int ok,char *buf,size_t n){
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

/* strdup is C23/POSIX, not C11 -- under the documented -std=c11 fallback build an implicit
 * declaration would truncate the returned pointer (and the later free() crashes). Local twin. */
static char *vn_strdup(const char *s){
  size_t n=strlen(s)+1; char *r=malloc(n); if(r)memcpy(r,s,n); return r;
}

/* a small growable byte string (the per-function record buffer + the value-number scratch). */
typedef struct { char *s; size_t n, cap; } sbuf;
static void sb_add(sbuf *b, const char *p, size_t k){
  if(b->n+k+1>b->cap){ size_t nc=b->cap?b->cap*2:256; while(nc<b->n+k+1)nc*=2;
    char *r=realloc(b->s,nc); if(!r)return; b->s=r; b->cap=nc; }
  memcpy(b->s+b->n,p,k); b->n+=k; b->s[b->n]=0;
}
static void sb_str(sbuf *b,const char *s){ sb_add(b,s,strlen(s)); }

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
    while(j>0 && strcmp(a[j-1],t)>0){ a[j]=a[j-1]; j--; } a[j]=t; }
}

/* per-function value-number context: writer[rid]=claim index that first writes it, plus a memo. */
typedef struct {
  const bcir_func *f;
  int *wclaim;          /* parallel to a rid list: index of the first writer claim, or -1 */
  uint32_t *wrid; int nw;
  char **memo;          /* memo[claim] -> its value-number string (lazily built), or NULL */
} vnctx;
static int writer_of(vnctx *v, uint32_t rid){
  for(int k=0;k<v->nw;k++) if(v->wrid[k]==rid) return v->wclaim[k]; return -1;
}
/* append vn(rid) to `out`. depth guards recursion; a re-entered claim folds to "cyc". */
static void vn_of(vnctx *v, uint32_t rid, int depth, sbuf *out);
static void vn_claim(vnctx *v, int ci, int depth, sbuf *out){
  if(v->memo[ci]){ sb_str(out,v->memo[ci]); return; }
  if(depth>BCIR_VN_MAXDEPTH){ sb_str(out,"cyc"); return; }
  v->memo[ci]=vn_strdup("cyc");                 /* cycle guard: a loop-carried rid resolves to "cyc" */
  const bcir_claim *cl=&v->f->claims[ci];
  char base[BCIR_CIR_NAME]; vn_base(cl->op,base);
  /* gather the vns of this claim's reads -- POSITIONAL, except sorted for a commutative op */
  char *parts[BCIR_CLAIM_MAX_RD]; sbuf ps[BCIR_CLAIM_MAX_RD]; int np=cl->n_rd;
  for(int k=0;k<np;k++){ ps[k]=(sbuf){0,0,0}; vn_of(v,cl->rd[k],depth+1,&ps[k]); parts[k]=ps[k].s?ps[k].s:(char*)""; }
  if(vn_commutative(base)) sort_strs(parts,np);
  sbuf me={0,0,0}; sb_str(&me,base); sb_str(&me,"(");
  for(int k=0;k<np;k++){ if(k)sb_str(&me,","); sb_str(&me,parts[k]); }
  sb_str(&me,")");
  for(int k=0;k<np;k++) free(ps[k].s);
  free(v->memo[ci]); v->memo[ci]=me.s?me.s:vn_strdup("");
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
static void canon_func(const bcir_func *f, void (*emit)(void*,const char*,size_t), void *ctx){
  /* index the non-NOP claims and the first-writer of each rid */
  int nc=0; for(size_t i=0;i<f->n_claims;i++) if(f->claims[i].opcode!=BCIR_OP_NOP) nc++;
  size_t maxw=f->n_claims*(size_t)BCIR_CLAIM_MAX_WR+1;   /* an upper bound on distinct written rids */
  int *cidx=malloc((size_t)(nc?nc:1)*sizeof *cidx);   /* cidx[j] = original claim index of the j-th non-NOP */
  vnctx v={f,NULL,NULL,0,NULL};
  v.wclaim=malloc(maxw*sizeof(int)); v.wrid=malloc(maxw*sizeof(uint32_t));
  v.memo=calloc(f->n_claims?f->n_claims:1,sizeof(char*));
  /* NB: vn indexes by ORIGINAL claim index (so memo/writer reference the real claim array). */
  int j=0;
  for(size_t i=0;i<f->n_claims;i++){ const bcir_claim *cl=&f->claims[i];
    if(cl->opcode==BCIR_OP_NOP) continue; cidx[j++]=(int)i;
    for(int k=0;k<cl->n_wr;k++){ uint32_t rid=cl->wr[k]; int seen=0;
      for(int m=0;m<v.nw;m++) if(v.wrid[m]==rid){seen=1;break;}
      if(!seen){ v.wrid[v.nw]=rid; v.wclaim[v.nw]=(int)i; v.nw++; } } }
  /* build each non-NOP claim's record */
  char **recs=malloc((size_t)nc*sizeof *recs);
  for(int r=0;r<nc;r++){ int i=cidx[r]; const bcir_claim *cl=&f->claims[i];
    char base[BCIR_CIR_NAME]; vn_base(cl->op,base);
    char *parts[BCIR_CLAIM_MAX_RD]; sbuf ps[BCIR_CLAIM_MAX_RD]; int np=cl->n_rd;
    for(int k=0;k<np;k++){ ps[k]=(sbuf){0,0,0}; vn_of(&v,cl->rd[k],0,&ps[k]); parts[k]=ps[k].s?ps[k].s:(char*)""; }
    if(vn_commutative(base)) sort_strs(parts,np);     /* commutative: sort; else POSITIONAL */
    sbuf rec={0,0,0}; char nb[24];
    sb_str(&rec,base); sb_str(&rec,"|");
    int ol=snprintf(nb,sizeof nb,"%d",(int)cl->opcode); sb_add(&rec,nb,(size_t)ol); sb_str(&rec,"|");
    for(int k=0;k<np;k++){ if(k)sb_str(&rec,","); sb_str(&rec,parts[k]); }
    sb_str(&rec,"|");
    vn_imm(cl,&rec);                                  /* the semantic imm (member offset / bitfield layout) */
    sb_str(&rec,"|");
    int dl=snprintf(nb,sizeof nb,"%d",(int)cl->domain); sb_add(&rec,nb,(size_t)dl);
    for(int k=0;k<np;k++) free(ps[k].s);
    recs[r]=rec.s?rec.s:vn_strdup("");
  }
  sort_strs(recs,nc);
  for(int r=0;r<nc;r++){ emit(ctx,recs[r],strlen(recs[r])); emit(ctx,"\n",1); free(recs[r]); }
  free(recs);

  /* The OBSERVABLE-OUTPUT anchor (LAST-writer VN -- a use observes the most-recent prior write, which
   * is what the emitted C returns/stores). It pins what the function OUTPUTS: the RETURN value's VN
   * (catches a sink-wr redirect that turns `return t` into `return (a+b)` though no per-claim record
   * changes) and the sorted STORE (dest-VN -> value-VN) pairs (catch a dead/store-target redirect).
   * last==first for a single-write rid, so the anchor stays cross-rail byte-identical. */
  vnctx vl={f,NULL,NULL,0,NULL};
  vl.wclaim=malloc(maxw*sizeof(int)); vl.wrid=malloc(maxw*sizeof(uint32_t));
  vl.memo=calloc(f->n_claims?f->n_claims:1,sizeof(char*));
  for(int r=0;r<nc;r++){ int i=cidx[r]; const bcir_claim *cl=&f->claims[i];
    for(int k=0;k<cl->n_wr;k++){ uint32_t rid=cl->wr[k]; int slot=-1;
      for(int m=0;m<vl.nw;m++) if(vl.wrid[m]==rid){slot=m;break;}
      if(slot<0){ vl.wrid[vl.nw]=rid; vl.wclaim[vl.nw]=(int)i; vl.nw++; }
      else vl.wclaim[slot]=(int)i; } }                /* LAST writer wins (overwrite) */
  sbuf anc={0,0,0}; sb_str(&anc,"ret=");
  if(f->has_return){ vn_of(&vl,f->return_rid,0,&anc); } else sb_str(&anc,"void");
  sb_str(&anc,"|stores=");
  /* collect store (dest->value) pairs, sorted */
  int nst=0; for(int r=0;r<nc;r++) if(!strcmp(f->claims[cidx[r]].op,"c.store")) nst++;
  if(nst){ char **sp=malloc((size_t)nst*sizeof *sp); int si=0;
    for(int r=0;r<nc;r++){ const bcir_claim *cl=&f->claims[cidx[r]];
      if(strcmp(cl->op,"c.store")) continue;
      sbuf s={0,0,0};
      if(cl->n_rd){ vn_of(&vl,cl->rd[0],0,&s); sb_str(&s,"->"); vn_of(&vl,cl->rd[cl->n_rd-1],0,&s); }
      else sb_str(&s,"?->?");
      sp[si++]=s.s?s.s:vn_strdup(""); }
    sort_strs(sp,nst);
    for(int k=0;k<nst;k++){ if(k)sb_str(&anc,";"); sb_str(&anc,sp[k]); free(sp[k]); }
    free(sp);
  }
  emit(ctx,anc.s?anc.s:"ret=void|stores=",anc.s?strlen(anc.s):16); emit(ctx,"\n",1); free(anc.s);
  for(size_t i=0;i<f->n_claims;i++) free(vl.memo[i]);
  free(vl.memo); free(vl.wclaim); free(vl.wrid);

  for(size_t i=0;i<f->n_claims;i++) free(v.memo[i]);
  free(v.memo); free(v.wclaim); free(v.wrid); free(cidx);
}

/* The shared canonical serializer: invokes emit(ctx, bytes, len) for each byte of the canon (so the
 * digest and the --canon text dump are GUARANTEED to be the same bytes). One '@' line per function. */
static void canon_walk(const bcir_unit *u, void (*emit)(void*,const char*,size_t), void *ctx){
  for(int fi=0;fi<u->n_funcs;fi++){ canon_func(&u->funcs[fi],emit,ctx); emit(ctx,"@\n",2); }
}

typedef struct { uint64_t h; } fnv_ctx;
static void fnv_emit(void *vc, const char *b, size_t n){
  fnv_ctx *c=vc; for(size_t i=0;i<n;i++) c->h=(c->h^(unsigned char)b[i])*1099511628211ull;
}
uint64_t bcir_cfront_digest(const bcir_unit *u){
  fnv_ctx c={1469598103934665603ull};            /* FNV-1a offset basis (== the Python _DIGEST_OFFSET) */
  canon_walk(u,fnv_emit,&c);
  return c.h;
}

typedef struct { char *buf; size_t cap, w; } buf_ctx;
static void buf_emit(void *vc, const char *b, size_t n){
  buf_ctx *c=vc; for(size_t i=0;i<n;i++){ if(c->w+1<c->cap) c->buf[c->w]=b[i]; c->w++; }
}
/* The raw canonical serialization the digest hashes (text, NOT hashed) -- the byte-identity proof
 * (the Python cfront_structural_canon must equal this byte-for-byte on the corpus). */
void bcir_cfront_canon(const bcir_unit *u, char *buf, size_t n){
  buf_ctx c={buf,n,0}; canon_walk(u,buf_emit,&c);
  if(n) buf[c.w<n?c.w:n-1]=0;
}

/* --- module-scope effect / commutation analysis (the C twin of pipeline own_footprint + commute) ---
 * Each function's alias/effect footprint over file-scope globals: the global NAMES it reads and writes
 * (a read references a global rid in a claim operand -- including the c.return / control-condition
 * markers -- a write is a claim result writing a global rid). Callee effects fold into the caller
 * transitively (R18 keeps the graph a DAG). Two functions commute iff their footprints don't conflict:
 * two readers of the same global commute; a writer conflicts with any reader or writer of it. */
#define BCIR_FX_MAXG 32

/* the unit's distinct file-scope global names (read_only named resources), first-seen order. */
static int fx_globals(const bcir_unit *u, char names[][BCIR_CIR_NAME]) {
  int n=0;
  for(int fi=0;fi<u->n_funcs;fi++){ const bcir_func *f=&u->funcs[fi];
    for(size_t i=0;i<f->n_res;i++){ const bcir_resource *r=&f->res[i];
      if(r->name[0] && r->read_only){
        int seen=0; for(int k=0;k<n;k++) if(!strcmp(names[k],r->name)){seen=1;break;}
        if(!seen && n<BCIR_FX_MAXG){ snprintf(names[n],BCIR_CIR_NAME,"%s",r->name); n++; } } } }
  return n;
}
static int fx_index(char names[][BCIR_CIR_NAME], int n, const char *nm){
  for(int k=0;k<n;k++) if(!strcmp(names[k],nm)) return k; return -1;
}
/* function f's OWN read/write global masks (over the names table). */
static void fx_own(const bcir_func *f, char names[][BCIR_CIR_NAME], int ng, uint64_t *rd, uint64_t *wr){
  *rd=0; *wr=0;
  for(size_t i=0;i<f->n_claims;i++){ const bcir_claim *c=&f->claims[i];
    for(int k=0;k<c->n_rd;k++){ const bcir_resource *r=res_of(f,c->rd[k]);
      if(r&&r->name[0]&&r->read_only){ int gi=fx_index(names,ng,r->name); if(gi>=0)*rd|=1ull<<gi; } }
    for(int k=0;k<c->n_wr;k++){ const bcir_resource *r=res_of(f,c->wr[k]);
      if(r&&r->name[0]&&r->read_only){ int gi=fx_index(names,ng,r->name); if(gi>=0)*wr|=1ull<<gi; } } }
}
static int fx_find(const bcir_unit *u, const char *name){
  for(int i=0;i<u->n_funcs;i++) if(!strcmp(u->funcs[i].name,name)) return i; return -1;
}
/* fold function fi's footprint with its callees' (transitively; the `seen` mask guards the DAG). */
static void fx_fold(const bcir_unit *u, int fi, char names[][BCIR_CIR_NAME], int ng,
                    uint64_t *rd, uint64_t *wr, char *seen){      /* seen[]: a visited byte per func */
  if(fi<0 || seen[fi]) return; seen[fi]=1;
  uint64_t ord,owr; fx_own(&u->funcs[fi],names,ng,&ord,&owr); *rd|=ord; *wr|=owr;
  const bcir_func *f=&u->funcs[fi];
  for(int k=0;k<f->n_calls;k++) fx_fold(u,fx_find(u,f->calls[k]),names,ng,rd,wr,seen);
}
/* append the sorted, comma-joined globals selected by `mask` (or "-" if empty). */
static size_t fx_print_names(char *o,size_t cap,size_t w, char names[][BCIR_CIR_NAME], int ng, uint64_t mask){
  int idx[BCIR_FX_MAXG], m=0;
  for(int k=0;k<ng;k++) if(mask&(1ull<<k)) idx[m++]=k;
  for(int i=1;i<m;i++){ int j=i; while(j>0 && strcmp(names[idx[j-1]],names[idx[j]])>0){int t=idx[j-1];idx[j-1]=idx[j];idx[j]=t;j--;} }
  if(m==0) return w + (size_t)snprintf(o+w, w<cap?cap-w:0, "-");
  for(int i=0;i<m;i++) w += (size_t)snprintf(o+w, w<cap?cap-w:0, "%s%s", i?",":"", names[idx[i]]);
  return w;
}

void bcir_cfront_effects(const bcir_unit *u, char *buf, size_t n){
  char names[BCIR_FX_MAXG][BCIR_CIR_NAME]; int ng=fx_globals(u,names);
  int nf=u->n_funcs; size_t af=(size_t)(nf>0?nf:1);   /* per-function footprints (any number of funcs) */
  uint64_t *frd=calloc(af,sizeof *frd), *fwr=calloc(af,sizeof *fwr); char *seen=calloc(af,1);
  if(!frd||!fwr||!seen){ if(n)buf[0]=0; free(frd);free(fwr);free(seen); return; }
  for(int i=0;i<nf;i++){ memset(seen,0,af); uint64_t rd=0,wr=0; fx_fold(u,i,names,ng,&rd,&wr,seen); frd[i]=rd; fwr[i]=wr; }
  size_t w=0;
  for(int i=0;i<u->n_funcs;i++){
    w+=(size_t)snprintf(buf+w,w<n?n-w:0,"fn=%s reads=",u->funcs[i].name);
    w=fx_print_names(buf,n,w,names,ng,frd[i]);
    w+=(size_t)snprintf(buf+w,w<n?n-w:0," writes=");
    w=fx_print_names(buf,n,w,names,ng,fwr[i]);
    w+=(size_t)snprintf(buf+w,w<n?n-w:0,"\n");
  }
  for(int i=0;i<u->n_funcs;i++) for(int j=i+1;j<u->n_funcs;j++){
    uint64_t wa=fwr[i],wb=fwr[j],ra=frd[i],rb=frd[j];
    int conflict = (wa & (rb|wb)) || (wb & (ra|wa));      /* a writes b's footprint, or vice versa */
    w+=(size_t)snprintf(buf+w,w<n?n-w:0,"commute %s %s = %d\n",u->funcs[i].name,u->funcs[j].name,conflict?0:1);
  }
  free(frd); free(fwr); free(seen);
}
