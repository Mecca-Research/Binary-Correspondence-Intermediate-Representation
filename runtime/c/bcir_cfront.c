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
typedef enum { T_ID, T_INT, T_STR, T_PUN, T_END } tkind;
typedef struct { tkind k; const char *s; int n; long long v; } tok;

#define MAXTOK 16384
#define MAXFLD 64
#define MAXENV 256
#define MAXTD 64
#define MAXEC 256

typedef struct { char name[BCIR_CIR_NAME]; int size; int signd; int byte_off, bit_off, bit_w; } field;
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

typedef struct {
  tok t[MAXTOK]; int nt, i;
  sdef s[16]; int ns;
  tdef td[MAXTD]; int ntd;        /* typedef aliases (resolved at parse time) */
  econst ec[MAXEC]; int nec;      /* enum constants (folded to literals at parse time) */
  gvar gv[16]; int ngv;           /* file-scope globals (lookup tables): name -> type + length */
  venv env[MAXENV]; int nenv;
  bcir_func *fn;
  uint32_t rid, cid;
  char err[256]; int failed;
} CC;

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
    if (c->nt>=MAXTOK-1) break;
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
    if (*p>='0'&&*p<='9'){t->k=T_INT;t->s=p;while(is_idc(*p)||*p=='\'')p++;t->n=(int)(p-t->s);
                          t->v=parse_int(t->s,t->n);c->nt++;continue;}
    if (*p=='"'){t->k=T_STR;t->s=p;p++;                /* string literal (escapes consumed as a unit) */
                 while(*p&&*p!='"'){ if(*p=='\\'&&p[1]) p+=2; else p++; }
                 if(*p=='"')p++; t->n=(int)(p-t->s); c->nt++; continue;}
    if (*p=='\''){t->k=T_INT;t->s=p;p++;               /* character constant -> a folded int const */
                  while(*p&&*p!='\''){ if(*p=='\\'&&p[1]) p+=2; else p++; }
                  if(*p=='\'')p++; t->n=(int)(p-t->s); t->v=parse_char(t->s,t->n); c->nt++; continue;}
    int m=0; for(int j=0;pu[j];j++) if(p[0]==pu[j][0]&&p[1]==pu[j][1]){
      t->k=T_PUN;t->s=p;t->n=2;p+=2;c->nt++;m=1;break;}
    if (m) continue;
    t->k=T_PUN;t->s=p;t->n=1;p++;c->nt++;
  }
  c->t[c->nt].k=T_END;c->t[c->nt].s="";c->t[c->nt].n=0;
}

/* --- token helpers ------------------------------------------------------- */
static tok *pk(CC *c){return &c->t[c->i];}
static int is(CC *c,const char *s){tok *t=pk(c);return (int)strlen(s)==t->n&&!strncmp(t->s,s,t->n);}
static int isk(CC *c,tkind k){return pk(c)->k==k;}
static tok adv(CC *c){return c->t[c->i++];}
static void fail(CC *c,const char *m){if(!c->failed){snprintf(c->err,sizeof c->err,"%s",m);c->failed=1;}}
static int eat(CC *c,const char *s){if(is(c,s)){c->i++;return 1;}fail(c,s);return 0;}
static void idcpy(char *d,const tok *t){int n=t->n<BCIR_CIR_NAME-1?t->n:BCIR_CIR_NAME-1;memcpy(d,t->s,n);d[n]=0;}

/* --- types --------------------------------------------------------------- */
static int scalar_size(const char *s,int n) {
  struct {const char *k;int sz;} T[]={{"void",0},{"char",1},{"bool",1},{"_Bool",1},{"short",2},
    {"int",4},{"unsigned",4},{"long",8},{"uint8_t",1},{"int8_t",1},{"uint16_t",2},{"int16_t",2},
    {"uint32_t",4},{"int32_t",4},{"uint64_t",8},{"int64_t",8},{"size_t",8},{0,0}};
  for(int i=0;T[i].k;i++) if((int)strlen(T[i].k)==n&&!strncmp(T[i].k,s,n)) return T[i].sz;
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
static int p_type(CC *c, bcir_ctype *ty, int *sidx) {
  memset(ty,0,sizeof *ty); ty->kind=0; ty->size=4; ty->signd=1; *sidx=-1;
  int seen=0;
  for(;;){
    if(is(c,"volatile")){ty->is_volatile=1;c->i++;continue;}
    if(is(c,"_Atomic")){ty->is_atomic=1;c->i++;continue;}
    if(is(c,"const")||is(c,"static")||is(c,"signed")||is(c,"inline")){c->i++;continue;}
    if(is(c,"unsigned")){ty->signd=0;ty->size=4;seen=1;c->i++;continue;}
    if(is(c,"struct")||is(c,"union")){c->i++;tok tag=adv(c);int si=find_struct(c,tag.s,tag.n);
      if(si<0){fail(c,"unknown struct");return 1;} ty->kind=1;ty->size=c->s[si].size;*sidx=si;
      ty->is_union=(uint8_t)c->s[si].is_union;idcpy(ty->tag,&tag);seen=1;break;}
    if(is(c,"enum")){c->i++;if(isk(c,T_ID)&&!is(c,"{"))c->i++;   /* `enum [tag] [{...}]` -> int */
      if(is(c,"{"))p_enum_body(c); ty->kind=0;ty->size=4;ty->signd=1;seen=1;break;}
    if(!seen&&isk(c,T_ID)){int ti=find_typedef(c,pk(c)->s,pk(c)->n);   /* a typedef alias */
      if(ti>=0){int vol=ty->is_volatile;*ty=c->td[ti].ty;if(vol)ty->is_volatile=1;*sidx=c->td[ti].sidx;c->i++;seen=1;break;}}
    if(isk(c,T_ID)){int sz=scalar_size(pk(c)->s,pk(c)->n);
      if(sz<0){if(seen)break;fail(c,"unknown type");return 1;} ty->size=sz;seen=1;c->i++;
      if(is(c,"long")||is(c,"int")||is(c,"char"))continue;break;}
    break;
  }
  if(!seen){fail(c,"expected a type");return 1;}
  while(is(c,"*")){c->i++;while(is(c,"const")||is(c,"volatile"))c->i++;
    if(ty->kind==1){ty->ptr_to_struct=1;} ty->kind=2;}
  return 0;
}

/* --- struct layout (Clang-compatible; bitfields LSB-first; packed/aligned, L8) --- */
static void attrs(CC *c,int *packed,int *aligned){
  for(;;){
    if(is(c,"__attribute__")){c->i++;eat(c,"(");eat(c,"(");
      while(!is(c,")")&&!isk(c,T_END)&&!c->failed){
        if(is(c,"packed")||is(c,"__packed__")){*packed=1;c->i++;}
        else if(is(c,"aligned")||is(c,"__aligned__")){c->i++;eat(c,"(");*aligned=(int)adv(c).v;eat(c,")");}
        else c->i++;
        if(is(c,","))c->i++;}
      eat(c,")");eat(c,")");
    } else if(is(c,"alignas")||is(c,"_Alignas")){c->i++;eat(c,"(");*aligned=(int)adv(c).v;eat(c,")");}
    else break;
  }
}
/* Parse `struct|union [tag] [attrs] { members } [attrs]` (NO trailing `;`). Registers an sdef and
 * returns its index (-1 on error). An anonymous aggregate (no tag, e.g. `typedef struct {...} N;`)
 * gets a synthesized internal tag so a typedef can alias it. */
static int p_struct_body(CC *c) {
  int is_union = is(c,"union");
  c->i++; int packed=0,aligned=0; attrs(c,&packed,&aligned);
  sdef *S=&c->s[c->ns]; S->nf=0; S->align=1; S->is_union=is_union;
  if(isk(c,T_ID)&&!is(c,"{")){tok tag=adv(c);idcpy(S->tag,&tag);}
  else snprintf(S->tag,sizeof S->tag,"$anon%d",c->ns);   /* anonymous: synth a unique tag */
  attrs(c,&packed,&aligned);
  if(!eat(c,"{"))return -1;
  int off=0,maxsz=0,bf_off=-1,bf_bits=0,bf_unit=0;
  while(!is(c,"}")&&!c->failed){
    bcir_ctype ty;int si;if(p_type(c,&ty,&si))return -1; tok nm=adv(c);
    int width=0; if(is(c,":")){c->i++;width=(int)adv(c).v;} eat(c,";");
    int sz=ty.size,al=packed?1:(sz<1?1:sz); field *f=&S->f[S->nf++];
    idcpy(f->name,&nm);f->size=sz;f->signd=ty.signd;f->bit_w=width;
    if(al>S->align)S->align=al; if(sz>maxsz)maxsz=sz;
    if(is_union){f->byte_off=0;f->bit_off=0;}        /* union: every member overlaps at offset 0 */
    else if(width){int ub=sz*8;
      if(bf_off<0||bf_unit!=sz||bf_bits+width>ub){if(off%al)off+=al-(off%al);bf_off=off;bf_bits=0;bf_unit=sz;off+=sz;}
      f->byte_off=bf_off;f->bit_off=bf_bits;bf_bits+=width;
    }else{bf_off=-1;bf_bits=0;bf_unit=0;if(off%al)off+=al-(off%al);f->byte_off=off;f->bit_off=0;off+=sz;}
  }
  eat(c,"}"); attrs(c,&packed,&aligned);
  int salign = packed ? 1 : S->align; if(aligned>salign) salign=aligned; S->align=salign;
  int total = is_union ? maxsz : off;              /* union size = the widest member */
  if(total%salign)total+=salign-(total%salign); S->size=total;
  return c->ns++;
}

/* --- enum + typedef (resolved at parse time so the claim graph carries the folded result) --- */
static long long ce_expr(CC *c,int minp);
static long long ce_primary(CC *c){
  if(isk(c,T_INT))return adv(c).v;
  if(is(c,"(")){c->i++;long long v=ce_expr(c,0);eat(c,")");return v;}
  if(is(c,"-")){c->i++;return -ce_primary(c);}
  if(is(c,"~")){c->i++;return ~ce_primary(c);}
  if(is(c,"!")){c->i++;return !ce_primary(c);}
  if(isk(c,T_ID)){int e=find_enum(c,pk(c)->s,pk(c)->n);if(e>=0){c->i++;return c->ec[e].val;}}
  fail(c,"non-constant enum initializer");return 0;
}
static long long ce_expr(CC *c,int minp){
  struct{const char*t;int p;}P[]={{"|",1},{"^",2},{"&",3},{"<<",5},{">>",5},
    {"+",6},{"-",6},{"*",7},{"/",7},{"%",7},{0,0}};
  long long lhs=ce_primary(c);
  for(;;){int p=-1;const char*op=0;
    for(int i=0;P[i].t;i++) if(is(c,P[i].t)){p=P[i].p;op=P[i].t;break;}
    if(p<minp||p<0)break; c->i++; long long rhs=ce_expr(c,p+1);
    lhs = !strcmp(op,"+")?lhs+rhs:!strcmp(op,"-")?lhs-rhs:!strcmp(op,"*")?lhs*rhs:
          !strcmp(op,"/")?(rhs?lhs/rhs:0):!strcmp(op,"%")?(rhs?lhs%rhs:0):
          !strcmp(op,"&")?lhs&rhs:!strcmp(op,"|")?lhs|rhs:!strcmp(op,"^")?lhs^rhs:
          !strcmp(op,"<<")?lhs<<rhs:lhs>>rhs;
  }
  return lhs;
}
static void p_enum_body(CC *c){
  eat(c,"{"); long long val=0;
  while(!is(c,"}")&&!c->failed){
    tok nm=adv(c);
    if(is(c,"=")){c->i++;val=ce_expr(c,0);}
    if(c->nec<MAXEC){idcpy(c->ec[c->nec].name,&nm);c->ec[c->nec].val=val;c->nec++;}
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
    while(is(c,"*")){c->i++;ty.ptr_to_struct=(ty.kind==1);ty.kind=2;}
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
      if(c->ntd<MAXTD){idcpy(c->td[c->ntd].name,&nm);c->td[c->ntd].ty=fp;c->td[c->ntd].sidx=-1;c->ntd++;}
      eat(c,";");
      return;
    }
    c->i=save;                                         /* not a funcptr declarator */
  }
  tok nm=adv(c);                                      /* the alias name */
  if(c->ntd<MAXTD){idcpy(c->td[c->ntd].name,&nm);c->td[c->ntd].ty=ty;c->td[c->ntd].sidx=sidx;c->ntd++;}
  eat(c,";");
}

/* --- the IR builder ------------------------------------------------------ */
static uint32_t add_res(CC *c, bcir_domain dom, int elem, int count, int vol, int kind, const char *nm) {
  bcir_func *f=c->fn; if(f->n_res>=f->cap_res) return 0;
  bcir_resource *r=&f->res[f->n_res++];
  r->rid=c->rid++; r->domain=dom; r->elem_bytes=elem<1?1:elem; r->count=count<1?1:count;
  r->is_volatile=vol; r->read_only=0; r->kind=kind; r->agg[0]=0;
  snprintf(r->name,sizeof r->name,"%s",nm?nm:"");
  return r->rid;
}
static bcir_claim *new_claim(CC *c,const char *op,bcir_opcode opc) {
  bcir_func *f=c->fn; if(f->n_claims>=f->cap_claims){fail(c,"too many claims");return NULL;}
  bcir_claim *cl=&f->claims[f->n_claims++]; memset(cl,0,sizeof *cl);
  cl->id=c->cid++;cl->opcode=opc;cl->lane=BCIR_LANE_U;cl->stride=BCIR_STRIDE_SCALAR;cl->count=1;
  cl->domain=BCIR_DOM_RAM;cl->hazard=BCIR_HZ_UNIQUE;cl->bounds=BCIR_BND_STRICT;
  snprintf(cl->op,sizeof cl->op,"%s",op); return cl;
}
static uint32_t temp(CC *c,int size){return add_res(c,BCIR_DOM_RAM,size?size:4,1,0,BCIR_RK_SCALAR,"");}
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

static uint32_t emit_member(CC *c, venv *base, const field *fld) {
  uint32_t t=temp(c,fld->size);
  bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
  cl->n_rd=1;cl->rd[0]=base->rid;cl->n_wr=1;cl->wr[0]=t;cl->n_imm=2;cl->imm[0]=fld->byte_off;cl->imm[1]=fld->size;
  cl->bounds=BCIR_BND_ASSUMED;
  if(base->type.is_volatile){cl->domain=BCIR_DOM_MMIO;cl->lane=BCIR_LANE_H;cl->hazard=BCIR_HZ_BARRIERED;}
  if(fld->bit_w){uint32_t u=t;t=temp(c,4);
    bcir_claim *g=new_claim(c,"c.bf.get",BCIR_OP_ADD);if(!g)return t;
    g->n_rd=1;g->rd[0]=u;g->n_wr=1;g->wr[0]=t;g->n_imm=2;g->imm[0]=fld->bit_off;g->imm[1]=fld->bit_w;}
  return t;
}
static uint32_t emit_index(CC *c, venv *base, uint32_t idx) {     /* base[idx] -- GEP load */
  uint32_t t=temp(c,base->type.size?base->type.size:4);
  bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
  cl->n_rd=2;cl->rd[0]=base->rid;cl->rd[1]=idx;cl->n_wr=1;cl->wr[0]=t;cl->bounds=BCIR_BND_ASSUMED;
  return t;
}
static uint32_t emit_deref(CC *c, venv *pv) {     /* *p -- a one-read dereference load */
  uint32_t t=temp(c,pv->type.size?pv->type.size:4);
  bcir_claim *cl=new_claim(c,"c.load",BCIR_OP_LOAD); if(!cl) return t;
  cl->n_rd=1;cl->rd[0]=pv->rid;cl->n_wr=1;cl->wr[0]=t;cl->bounds=BCIR_BND_ASSUMED;
  if(pv->type.is_volatile){cl->domain=BCIR_DOM_MMIO;cl->lane=BCIR_LANE_H;cl->hazard=BCIR_HZ_BARRIERED;}
  return t;
}
static uint32_t p_call(CC *c, const tok *name) {
  c->i++; /* '(' */
  uint32_t args[BCIR_CLAIM_MAX_RD]; int na=0;
  if(!is(c,")")) for(;;){ uint32_t a=p_expr(c); if(na<BCIR_CLAIM_MAX_RD)args[na++]=a;
    if(is(c,",")){c->i++;continue;} break; }
  eat(c,")");
  uint32_t t=temp(c,4);
  char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.call:%.*s",name->n,name->s);
  bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
  if(cl){cl->n_rd=(uint8_t)na;for(int k=0;k<na;k++)cl->rd[k]=args[k];cl->n_wr=1;cl->wr[0]=t;}
  if(c->fn->n_calls<BCIR_MAX_CALLS) idcpy(c->fn->calls[c->fn->n_calls++],name);
  return t;
}
/* An indirect call through a function-pointer local/param (HAL dispatch): the target is dynamic, so
 * there is no named callee -- a `c.call.indirect` claim (reads: the pointer value then the actuals).
 * It is *not* added to fn->calls, so R18 leaves it an opaque external edge (no recursion/resolution). */
static uint32_t p_icall(CC *c, const venv *fv) {
  c->i++; /* '(' */
  uint32_t args[BCIR_CLAIM_MAX_RD]; int na=0;
  if(!is(c,")")) for(;;){ uint32_t a=p_expr(c); if(na<BCIR_CLAIM_MAX_RD-1)args[na++]=a;
    if(is(c,",")){c->i++;continue;} break; }
  eat(c,")");
  uint32_t t=temp(c,4);
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
    {"__atomic_thread_fence","c.fence",BCIR_OP_BARRIER,AK_FENCE},
    {"__sync_synchronize","c.fence",BCIR_OP_BARRIER,AK_FENCE},
    {"__sync_val_compare_and_swap","c.cmpxchg.val",BCIR_OP_CMPXCHG,AK_CAS},
    {"__sync_bool_compare_and_swap","c.cmpxchg.bool",BCIR_OP_CMPXCHG,AK_CAS},
    {"atomic_fetch_add","c.c11atom.fetch_add",BCIR_OP_ATOMIC_ADD,AK_RMW},  /* C11 <stdatomic.h> */
    {"atomic_fetch_sub","c.c11atom.fetch_sub",BCIR_OP_ATOMIC_SUB,AK_RMW},
    {"atomic_fetch_xor","c.c11atom.fetch_xor",BCIR_OP_ATOMIC_XOR,AK_RMW},
    {"atomic_exchange","c.c11atom.exchange",BCIR_OP_ATOMIC_ADD,AK_RMW},   /* swap: set + return old */
    {"atomic_load","c.c11atom.load",BCIR_OP_LOAD,AK_LOAD},
    {"atomic_store","c.c11atom.store",BCIR_OP_STORE,AK_STORE},{0,0,0,0}};
  for(int i=0;A[i].n;i++) if((int)strlen(A[i].n)==t->n&&!strncmp(A[i].n,t->s,t->n)){*op=A[i].op;*oc=A[i].oc;*kind=A[i].k;return 1;}
  return 0;
}
static uint32_t p_atomic(CC *c,const char *op,bcir_opcode oc,int kind){
  c->i++; uint32_t args[BCIR_CLAIM_MAX_RD]; int na=0;
  if(!is(c,")")) for(;;){uint32_t a=p_expr(c);if(na<BCIR_CLAIM_MAX_RD)args[na++]=a;if(is(c,",")){c->i++;continue;}break;}
  eat(c,")");
  uint32_t t=temp(c,4); bcir_claim *cl=new_claim(c,op,oc); if(!cl)return t;
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

static uint32_t p_primary(CC *c) {
  if(isk(c,T_INT)){tok t=adv(c);uint32_t r=temp(c,4);
    bcir_claim *cl=new_claim(c,"c.const",BCIR_OP_LOAD);if(!cl)return r;
    cl->n_wr=1;cl->wr[0]=r;cl->n_imm=1;cl->imm[0]=t.v;return r;}
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
    if(!p_type(c,&ty,&si)) al = ty.kind==2?8:(ty.kind==1?c->s[si].align:(ty.size?ty.size:1));
    eat(c,")");
    uint32_t r=temp(c,4); bcir_claim *cl=new_claim(c,"c.const",BCIR_OP_LOAD);
    if(cl){cl->n_wr=1;cl->wr[0]=r;cl->n_imm=1;cl->imm[0]=al;} return r;
  }
  if(is(c,"sizeof")){                  /* sizeof(type) / sizeof expr -> a folded constant (no eval) */
    c->i++; long long size=4; int got=0;
    if(is(c,"(")){ int save=c->i; c->i++;
      int is_type = scalar_size(pk(c)->s,pk(c)->n)>=0 || is(c,"struct")||is(c,"union")||is(c,"enum")
                    || is(c,"const")||is(c,"volatile") || find_typedef(c,pk(c)->s,pk(c)->n)>=0;
      if(is_type){ bcir_ctype ty;int si;
        if(!p_type(c,&ty,&si)){ size = ty.kind==2?8:(ty.kind==1?c->s[si].size:ty.size); got=1; }
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
        if(v) size = v->type.kind==2?8:(v->type.kind==1?c->s[v->sidx].size:v->type.size);
        c->i++; }
      if(paren) eat(c,")");
    }
    uint32_t r=temp(c,4); bcir_claim *cl=new_claim(c,"c.const",BCIR_OP_LOAD);
    if(cl){cl->n_wr=1;cl->wr[0]=r;cl->n_imm=1;cl->imm[0]=size;} return r;
  }
  if(is(c,"(")){c->i++;uint32_t r=p_expr(c);eat(c,")");return r;}
  if(isk(c,T_ID)){
    tok id=adv(c);
    const char *aop;bcir_opcode aoc;int akind;
    if(is(c,"(")&&atomic_kind(&id,&aop,&aoc,&akind)) return p_atomic(c,aop,aoc,akind);  /* atomics/fences/CAS */
    if(is(c,"(")){ venv *fv=lookup(c,&id);        /* indirect call (funcptr var) vs. direct named call */
      if(fv&&fv->type.kind==3) return p_icall(c,fv); return p_call(c,&id); }
    int ec=find_enum(c,id.s,id.n);                /* an enumerator -> its folded constant */
    if(ec>=0){uint32_t r=temp(c,4);bcir_claim *cl=new_claim(c,"c.const",BCIR_OP_LOAD);
      if(cl){cl->n_wr=1;cl->wr[0]=r;cl->n_imm=1;cl->imm[0]=c->ec[ec].val;}return r;}
    venv *v=lookup(c,&id); if(!v) v=use_global(c,&id);   /* a file-scope global (lookup table)? */
    if(!v){fail(c,"undefined identifier");return 0;}
    if(is(c,".")||is(c,"->")){
      int arrow=is(c,"->"); c->i++; tok fn=adv(c); sdef *S=&c->s[v->sidx]; int fi=-1;
      for(int i=0;i<S->nf;i++) if((int)strlen(S->f[i].name)==fn.n&&!strncmp(S->f[i].name,fn.s,fn.n)) fi=i;
      if(fi<0){fail(c,"unknown field");return 0;}
      if(is(c,"(")){     /* o->fnptr(args): fused indirect call via a funcptr struct member */
        c->i++; uint32_t args[BCIR_CLAIM_MAX_RD]; int na=0;
        if(!is(c,")")) for(;;){ uint32_t a=p_expr(c); if(na<BCIR_CLAIM_MAX_RD-1)args[na++]=a;
          if(is(c,",")){c->i++;continue;} break; }
        eat(c,")");
        uint32_t t=temp(c,4);
        char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.call.imember:%s",S->f[fi].name);
        bcir_claim *cl=new_claim(c,op,BCIR_OP_GEM_DISPATCH);
        if(cl){cl->n_rd=(uint8_t)(na+1);cl->rd[0]=v->rid;for(int k=0;k<na;k++)cl->rd[k+1]=args[k];
          cl->n_wr=1;cl->wr[0]=t;cl->n_imm=1;cl->imm[0]=arrow;}
        return t;
      }
      return emit_member(c,v,&S->f[fi]);
    }
    if(is(c,"[")){                                /* L3: base[i] / m[i][j] (row-major flatten) */
      uint32_t idxs[3]; int ni=0;
      while(is(c,"[")){ c->i++; uint32_t ix=p_expr(c); eat(c,"]"); if(ni<3)idxs[ni++]=ix; }
      uint32_t lin=idxs[0];
      for(int d=1; d<ni; d++){                     /* lin = lin*dim + idxs[d]  (Horner) */
        int dim = d<v->type.nadims ? v->type.adims[d] : 1;
        uint32_t k=temp(c,4); bcir_claim *kc=new_claim(c,"c.const",BCIR_OP_LOAD);
        if(kc){kc->n_wr=1;kc->wr[0]=k;kc->n_imm=1;kc->imm[0]=dim;}
        uint32_t m1=temp(c,4); bcir_claim *mc=new_claim(c,"c.bin.mul",BCIR_OP_MUL);
        if(mc){mc->n_rd=2;mc->rd[0]=lin;mc->rd[1]=k;mc->n_wr=1;mc->wr[0]=m1;}
        uint32_t a1=temp(c,4); bcir_claim *ac=new_claim(c,"c.bin.add",BCIR_OP_ADD);
        if(ac){ac->n_rd=2;ac->rd[0]=m1;ac->rd[1]=idxs[d];ac->n_wr=1;ac->wr[0]=a1;}
        lin=a1;
      }
      return emit_index(c,v,lin);
    }
    return v->rid;
  }
  fail(c,"expected expression");return 0;
}
/* name a cast's target type by width, so both rails emit the same (uintN_t) spelling. */
static void cast_name(const bcir_ctype *ty,char *o,size_t n){
  const char *nm=ty->size==1?"uint8_t":ty->size==2?"uint16_t":ty->size==8?"uint64_t":"uint32_t";
  if(ty->kind==2) snprintf(o,n,"c.cast:%s *",nm); else snprintf(o,n,"c.cast:%s",nm);
}
static uint32_t p_unary(CC *c) {
  if(is(c,"+")){ c->i++; return p_unary(c); }    /* unary plus is a no-op */
  if(is(c,"-")||is(c,"~")||is(c,"!")){
    const char *suf=is(c,"-")?"neg":is(c,"~")?"bnot":"lnot";
    bcir_opcode oc=is(c,"-")?BCIR_OP_SUB:BCIR_OP_ADD;c->i++;
    uint32_t a=p_unary(c),r=temp(c,4);char op[BCIR_CIR_NAME];snprintf(op,sizeof op,"c.un.%s",suf);
    bcir_claim *cl=new_claim(c,op,oc);if(cl){cl->n_rd=1;cl->rd[0]=a;cl->n_wr=1;cl->wr[0]=r;}return r;}
  if(is(c,"*")){                                   /* pointer dereference: *p / *(p + i) */
    c->i++;
    if(is(c,"(")){ int save=c->i; c->i++;          /* *(p) or *(p + i) */
      if(isk(c,T_ID)){ tok pid=*pk(c); venv *pv=lookup(c,&pid);
        if(pv){ c->i++;
          if(is(c,"+")){ c->i++; uint32_t idx=p_expr(c); eat(c,")"); return emit_index(c,pv,idx); }
          if(is(c,")")){ c->i++; return emit_deref(c,pv); } } }
      c->i=save;
    } else if(isk(c,T_ID)){ tok pid=*pk(c); venv *pv=lookup(c,&pid);
      if(pv){ c->i++; return emit_deref(c,pv); } }  /* *p */
    fail(c,"unsupported dereference"); return 0;
  }
  if(is(c,"(")){                                   /* (type)operand -- a cast binds at the unary level */
    int save=c->i; c->i++;
    int is_type = scalar_size(pk(c)->s,pk(c)->n)>=0 || is(c,"struct")||is(c,"union")||is(c,"enum")
                  || is(c,"const")||is(c,"volatile") || find_typedef(c,pk(c)->s,pk(c)->n)>=0;
    if(is_type){ bcir_ctype ty;int si;
      if(!p_type(c,&ty,&si) && is(c,")")){
        c->i++;                                    /* ')' */
        uint32_t v=p_unary(c);                     /* the operand (right-associative) */
        uint32_t r=temp(c, ty.kind==2?8:(ty.size?ty.size:4));
        char op[BCIR_CIR_NAME]; cast_name(&ty,op,sizeof op);
        bcir_claim *cl=new_claim(c,op,BCIR_OP_ADD);
        if(cl){cl->n_rd=1;cl->rd[0]=v;cl->n_wr=1;cl->wr[0]=r;} return r;
      }
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
    case '|':*suf="or";*oc=BCIR_OP_ADD;break;  default:*suf="xor";*oc=BCIR_OP_ADD;break;}  /* ^ */
}
static uint32_t p_binrhs(CC *c,int min_prec,uint32_t lhs) {
  for(;;){
    char suf[BCIR_CIR_NAME];bcir_opcode oc;int idx=bin_op(c,suf,&oc);
    if(idx<0||prec_of(idx)<min_prec)return lhs;
    int prec=prec_of(idx);c->i++;uint32_t rhs=p_unary(c);
    char s2[BCIR_CIR_NAME];bcir_opcode o2;int nx=bin_op(c,s2,&o2);
    while(nx>=0&&prec_of(nx)>prec){rhs=p_binrhs(c,prec_of(nx),rhs);nx=bin_op(c,s2,&o2);}
    uint32_t r=temp(c,4);char op[BCIR_CIR_NAME];snprintf(op,sizeof op,"c.bin.%s",suf);
    bcir_claim *cl=new_claim(c,op,oc);if(cl){cl->n_rd=2;cl->rd[0]=lhs;cl->rd[1]=rhs;cl->n_wr=1;cl->wr[0]=r;}
    lhs=r;
  }
}
static uint32_t p_binexpr(CC *c){return p_binrhs(c,1,p_unary(c));}
/* p_expr layers the ternary `cond ? then : els` over the binary expression: a scalar select claim
 * (both arms lowered, then chosen; the emitter renders the real `(cond ? a : b)`). */
static uint32_t p_expr(CC *c){
  uint32_t cond=p_binexpr(c);
  if(is(c,"?")){ c->i++; uint32_t a=p_expr(c); eat(c,":"); uint32_t b=p_expr(c);
    uint32_t t=temp(c,4); bcir_claim *cl=new_claim(c,"c.select",BCIR_OP_ADD);
    if(cl){cl->n_rd=3;cl->rd[0]=cond;cl->rd[1]=a;cl->rd[2]=b;cl->n_wr=1;cl->wr[0]=t;} return t; }
  return cond;
}

/* --- statements + functions ---------------------------------------------- */
static void env_add(CC *c,const tok *nm,uint32_t rid,const bcir_ctype *ty,int sidx){
  if(c->nenv>=MAXENV)return; venv *v=&c->env[c->nenv++];
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
/* a control-flow marker claim (no realization; the emitter renders it as a brace). carries an
 * optional condition rid as a read so the verifier resolves it and the emitter can test it. */
static void marker(CC *c,const char *op,uint32_t cond,int has_cond){
  bcir_claim *cl=new_claim(c,op,BCIR_OP_NOP);
  if(cl&&has_cond){cl->n_rd=1;cl->rd[0]=cond;}
}
static void p_stmt(CC *c);
static void p_block(CC *c){            /* `{ stmts }` or a single statement */
  if(is(c,"{")){c->i++;while(!is(c,"}")&&!isk(c,T_END)&&!c->failed)p_stmt(c);eat(c,"}");}
  else p_stmt(c);
}
/* ++i / --i / i++ / i-- (value discarded) -> i = i ± 1 (const 1 + a bin op + a copy).  Returns 1 if
 * it consumed an increment/decrement, 0 (consuming nothing) otherwise. */
static int p_incdec(CC *c) {
  venv *v=NULL; char ch=0;
  if((is(c,"++")||is(c,"--")) && c->t[c->i+1].k==T_ID){            /* ++name / --name */
    v=lookup(c,&c->t[c->i+1]); if(!v) return 0; ch=c->t[c->i].s[0]; c->i+=2;
  } else if(isk(c,T_ID) && c->t[c->i+1].k==T_PUN && c->t[c->i+1].n==2 &&
            (c->t[c->i+1].s[0]=='+'||c->t[c->i+1].s[0]=='-') && c->t[c->i+1].s[1]==c->t[c->i+1].s[0]){
    v=lookup(c,pk(c)); if(!v) return 0; ch=c->t[c->i+1].s[0]; c->i+=2;   /* name++ / name-- */
  } else return 0;
  uint32_t one=temp(c,4); bcir_claim *kc=new_claim(c,"c.const",BCIR_OP_LOAD);
  if(kc){kc->n_wr=1;kc->wr[0]=one;kc->n_imm=1;kc->imm[0]=1;}
  uint32_t tmp=temp(c,4); bcir_claim *b=new_claim(c,ch=='+'?"c.bin.add":"c.bin.sub",ch=='+'?BCIR_OP_ADD:BCIR_OP_SUB);
  if(b){b->n_rd=2;b->rd[0]=v->rid;b->rd[1]=one;b->n_wr=1;b->wr[0]=tmp;}
  bcir_claim *cp=new_claim(c,"c.copy",BCIR_OP_ADD); if(cp){cp->n_rd=1;cp->rd[0]=tmp;cp->n_wr=1;cp->wr[0]=v->rid;}
  return 1;
}
/* a `name = expr` assignment or a bare expression, WITHOUT the trailing `;` (the for-loop step). */
static void p_simple(CC *c) {
  if(p_incdec(c)) return;
  if(isk(c,T_ID)){ tok id=*pk(c); venv *v=lookup(c,&id);
    if(v && c->t[c->i+1].k==T_PUN && c->t[c->i+1].n==1 && c->t[c->i+1].s[0]=='='){
      c->i+=2; uint32_t val=p_expr(c);
      bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD); if(cl){cl->n_rd=1;cl->rd[0]=val;cl->n_wr=1;cl->wr[0]=v->rid;}
      return; } }
  (void)p_expr(c);
}
static void p_stmt(CC *c) {
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
    if(is(c,";")) c->i++;              /* empty init */
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
    if(step_end>step_start){ int save=c->i; c->i=step_start; p_simple(c); c->i=save; }  /* step @ iter end */
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
  if(is(c,"goto")){ c->i++; tok lb=adv(c); eat(c,";");          /* goto label; -- an emit-only marker */
    char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.goto:%.*s",lb.n,lb.s); new_claim(c,op,BCIR_OP_NOP); return; }
  if(isk(c,T_ID)&&c->t[c->i+1].k==T_PUN&&c->t[c->i+1].n==1&&c->t[c->i+1].s[0]==':'){  /* `name:` -- a label */
    tok lb=adv(c); c->i++; char op[BCIR_CIR_NAME];
    snprintf(op,sizeof op,"c.label:%.*s",lb.n,lb.s); new_claim(c,op,BCIR_OP_NOP); return; }
  if(is(c,"switch")){                  /* switch(disc){case V:..;break; default:..} -> if/else-if */
    c->i++; eat(c,"(");
    int disc_start=c->i;               /* re-lower the discriminant per case label (cheap for a var) */
    { int pd=1; while(!isk(c,T_END)&&pd){ if(is(c,"("))pd++; else if(is(c,")")){pd--; if(!pd)break;} c->i++; } }
    eat(c,")"); eat(c,"{");
    int nopen=0, first=1;
    while(!is(c,"}")&&!isk(c,T_END)&&!c->failed){
      if(!first) marker(c,"c.else",0,0);             /* open this clause's scope FIRST, so its */
      uint32_t condrid=0; int have_cond=0, is_default=0;   /* condition claims land in the else, */
      while(is(c,"case")||is(c,"default")){          /* not the previous then-block. */
        if(is(c,"case")){ c->i++; uint32_t valrid=p_expr(c); eat(c,":");
          int save=c->i; c->i=disc_start; uint32_t drid=p_expr(c); c->i=save;   /* disc, re-lowered */
          uint32_t cmp=temp(c,4); bcir_claim *e=new_claim(c,"c.bin.eq",BCIR_OP_SUB);
          if(e){e->n_rd=2;e->rd[0]=drid;e->rd[1]=valrid;e->n_wr=1;e->wr[0]=cmp;}
          if(have_cond){ uint32_t o=temp(c,4); bcir_claim *l=new_claim(c,"c.bin.lor",BCIR_OP_ADD);
            if(l){l->n_rd=2;l->rd[0]=condrid;l->rd[1]=cmp;l->n_wr=1;l->wr[0]=o;} condrid=o; }
          else { condrid=cmp; have_cond=1; }
        } else { c->i++; eat(c,":"); is_default=1; }
      }
      if(!is_default){ marker(c,"c.if",condrid,1); nopen++; }
      while(!is(c,"case")&&!is(c,"default")&&!is(c,"}")&&!isk(c,T_END)&&!c->failed){
        if(is(c,"break")){ c->i++; eat(c,";"); break; }   /* the switch terminator (dropped) */
        p_stmt(c);
      }
      first=0;
    }
    eat(c,"}");
    for(int k=0;k<nopen;k++) marker(c,"c.endif",0,0);     /* close the else-if nesting */
    return;
  }
  if(is(c,"{")){p_block(c);return;}
  int looks_decl=0, is_static=is(c,"static");
  if(isk(c,T_ID)){int sz=scalar_size(pk(c)->s,pk(c)->n);
    looks_decl=sz>=0||is_static||is(c,"struct")||is(c,"union")||is(c,"enum")||is(c,"const")||is(c,"volatile")
               ||find_typedef(c,pk(c)->s,pk(c)->n)>=0;}
  if(looks_decl){
    bcir_ctype ty;int si;if(p_type(c,&ty,&si))return; tok nm=adv(c);   /* p_type eats `static` */
    char nb[BCIR_CIR_NAME]; idcpy(nb,&nm);
    int rk=ty.kind==2?BCIR_RK_POINTER:ty.kind==1?BCIR_RK_AGGREGATE:BCIR_RK_SCALAR;
    uint32_t rid=add_res(c, ty.is_volatile?BCIR_DOM_MMIO:BCIR_DOM_RAM,
                         ty.kind==2?ty.size:(ty.kind==1?c->s[si].size:ty.size),
                         ty.kind==2?(1<<16):1, ty.is_volatile, rk, nb);
    if(ty.kind==1) snprintf(c->fn->res[c->fn->n_res-1].agg,BCIR_CIR_NAME,"%s %s",ty.is_union?"union":"struct",ty.tag);   /* L8 aggregate local */
    env_add(c,&nm,rid,&ty,si);
    if(is_static){            /* static storage: a once-only constant init, baked into the decl */
      long long init=0; if(is(c,"=")){c->i++;init=ce_expr(c,0);}
      if(c->fn->n_statics<8){idcpy(c->fn->statics[c->fn->n_statics].name,&nm);
        c->fn->statics[c->fn->n_statics].init=init;c->fn->n_statics++;}
      eat(c,";");return;
    }
    if(is(c,"=")){c->i++;uint32_t v=p_expr(c);
      bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD);if(cl){cl->n_rd=1;cl->rd[0]=v;cl->n_wr=1;cl->wr[0]=rid;}}
    eat(c,";");return;
  }
  if(isk(c,T_ID)){tok id=*pk(c);venv *v=lookup(c,&id);
    /* L8: struct member store  v.field = expr  /  v->field = expr */
    if(v&&v->sidx>=0&&c->t[c->i+1].k==T_PUN&&(c->t[c->i+1].s[0]=='.'||(c->t[c->i+1].n==2&&c->t[c->i+1].s[0]=='-'))){
      c->i+=2; tok fld=adv(c); sdef *S=&c->s[v->sidx]; int fi=-1;
      for(int k=0;k<S->nf;k++) if((int)strlen(S->f[k].name)==fld.n&&!strncmp(S->f[k].name,fld.s,fld.n)) fi=k;
      if(fi<0){fail(c,"unknown field");return;}
      uint32_t val;
      if(c->t[c->i].k==T_PUN&&c->t[c->i].n==2&&c->t[c->i].s[1]=='='&&strchr("+-*/%&|^",c->t[c->i].s[0])){
        /* compound assignment to a member:  r->field OP= expr  (the set/clear-bits driver idiom; a
         * bitfield field reads via c.bf.get, a plain member via a plain load). */
        char ch=c->t[c->i].s[0]; c->i++;
        uint32_t cur=emit_member(c,v,&S->f[fi]);    /* the current field value (loaded first) */
        uint32_t rhs=p_expr(c);
        const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
        uint32_t tmp=temp(c,4); char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.bin.%s",suf);
        bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=cur;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
        val=tmp;
      } else { if(!eat(c,"="))return; val=p_expr(c); }
      if(S->f[fi].bit_w){
        /* a bitfield store: read the storage unit, insert the masked bits (c.bf.set), store the unit. */
        uint32_t unit=temp(c,S->f[fi].size);
        bcir_claim *ld=new_claim(c,"c.load",BCIR_OP_LOAD);
        if(ld){ld->n_rd=1;ld->rd[0]=v->rid;ld->n_wr=1;ld->wr[0]=unit;ld->n_imm=2;ld->imm[0]=S->f[fi].byte_off;ld->imm[1]=S->f[fi].size;ld->bounds=BCIR_BND_ASSUMED;
          if(v->type.is_volatile){ld->domain=BCIR_DOM_MMIO;ld->lane=BCIR_LANE_H;ld->hazard=BCIR_HZ_BARRIERED;}}
        uint32_t nu=temp(c,S->f[fi].size);
        bcir_claim *bs=new_claim(c,"c.bf.set",BCIR_OP_ADD);
        if(bs){bs->n_rd=2;bs->rd[0]=unit;bs->rd[1]=val;bs->n_wr=1;bs->wr[0]=nu;bs->n_imm=2;bs->imm[0]=S->f[fi].bit_off;bs->imm[1]=S->f[fi].bit_w;}
        val=nu;
      }
      bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
      if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=val;cl->n_imm=2;cl->imm[0]=S->f[fi].byte_off;cl->imm[1]=S->f[fi].size;
        cl->bounds=BCIR_BND_ASSUMED;
        if(v->type.is_volatile){cl->domain=BCIR_DOM_MMIO;cl->lane=BCIR_LANE_H;cl->hazard=BCIR_HZ_BARRIERED;}}
      eat(c,";");return;}
    if(v&&c->t[c->i+1].k==T_PUN&&c->t[c->i+1].n==1&&c->t[c->i+1].s[0]=='='){
      c->i+=2;uint32_t val=p_expr(c);
      bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD);if(cl){cl->n_rd=1;cl->rd[0]=val;cl->n_wr=1;cl->wr[0]=v->rid;}
      eat(c,";");return;}
    /* compound assignment  name OP= expr  ->  name = name OP expr  (a bin op + a copy). */
    if(v&&c->t[c->i+1].k==T_PUN&&c->t[c->i+1].n==2&&c->t[c->i+1].s[1]=='='
       &&strchr("+-*/%&|^",c->t[c->i+1].s[0])){
      char ch=c->t[c->i+1].s[0]; c->i+=2; uint32_t rhs=p_expr(c);
      const char *suf; bcir_opcode oc; compound_binop(ch,&suf,&oc);
      uint32_t tmp=temp(c,4); char op[BCIR_CIR_NAME]; snprintf(op,sizeof op,"c.bin.%s",suf);
      bcir_claim *b=new_claim(c,op,oc); if(b){b->n_rd=2;b->rd[0]=v->rid;b->rd[1]=rhs;b->n_wr=1;b->wr[0]=tmp;}
      bcir_claim *cp=new_claim(c,"c.copy",BCIR_OP_ADD); if(cp){cp->n_rd=1;cp->rd[0]=tmp;cp->n_wr=1;cp->wr[0]=v->rid;}
      eat(c,";");return;}}
  if(p_incdec(c)){eat(c,";");return;}    /* ++i / i++ / --i / i-- as a statement */
  (void)p_expr(c);eat(c,";");
}

static int p_func(CC *c, bcir_func *fn) {
  c->fn=fn; c->nenv=0;
  bcir_ctype rt;int rsi;if(p_type(c,&rt,&rsi))return 1; fn->ret=rt;
  tok nm=adv(c); snprintf(fn->name,sizeof fn->name,"%.*s",nm.n,nm.s);
  if(!eat(c,"("))return 1;
  if(!is(c,")")) for(;;){
    if(is(c,"void")&&c->t[c->i+1].n==1&&c->t[c->i+1].s[0]==')'){c->i++;break;}
    bcir_ctype ty;int si;if(p_type(c,&ty,&si))return 1; tok pn=adv(c);
    if(is(c,"[")){              /* an array parameter `T name[A][B]...` decays to a flat element ptr */
      int nd=0;
      while(is(c,"[")){ c->i++; long long d=isk(c,T_INT)?(long long)adv(c).v:0;
        if(nd<3)ty.adims[nd]=(int)d; nd++; eat(c,"]"); }
      ty.nadims=nd<3?nd:3; if(ty.kind==0) ty.kind=2;     /* T[..] -> T* (element size kept in ty.size) */
    }
    char pb[BCIR_CIR_NAME]; idcpy(pb,&pn);
    int rk=ty.kind==2?BCIR_RK_POINTER:ty.kind==1?BCIR_RK_AGGREGATE:BCIR_RK_SCALAR;
    uint32_t rid=add_res(c, ty.is_volatile?BCIR_DOM_MMIO:BCIR_DOM_RAM,
                         ty.kind==2?ty.size:(ty.kind==1?c->s[si].size:ty.size),
                         ty.kind==2?(1<<16):1, ty.is_volatile, rk, pb);
    if(ty.kind==1) snprintf(c->fn->res[c->fn->n_res-1].agg,BCIR_CIR_NAME,"%s %s",ty.is_union?"union":"struct",ty.tag);
    env_add(c,&pn,rid,&ty,si);
    if(fn->n_params<BCIR_MAX_PARAMS){bcir_param *pp=&fn->params[fn->n_params++];
      idcpy(pp->name,&pn);pp->rid=rid;pp->type=ty;}
    if(is(c,",")){c->i++;continue;} break;
  }
  if(!eat(c,")"))return 1; if(!eat(c,"{"))return 1;
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
  int is_struct = (ty->kind==1) || ty->ptr_to_struct;
  const char *kw = ty->is_union ? "union" : "struct";
  const char *base = is_struct ? ty->tag
                   : (ty->size==1?"uint8_t":ty->size==2?"uint16_t":ty->size==8?"uint64_t":"uint32_t");
  const char *atm = ty->is_atomic ? "_Atomic " : "";
  if(ty->kind==2) snprintf(o,n,"%s%s%s%s%s *",atm,ty->is_volatile?"volatile ":"",
                           ty->ptr_to_struct?kw:"",ty->ptr_to_struct?" ":"",base);
  else if(ty->kind==1) snprintf(o,n,"%s %s",kw,ty->tag);
  else snprintf(o,n,"%s%s",atm,base);
}
static const char *rname(const bcir_func *f,uint32_t rid,char *buf){
  const char *lit=strtab_lookup(rid);                /* a string literal -> its full spelling, inline */
  if(lit) return lit;                                /* (returned directly, so length is not capped) */
  const bcir_resource *r=res_of(f,rid);
  if(r&&r->name[0]){snprintf(buf,BCIR_CIR_NAME,"%s",r->name);return buf;}
  snprintf(buf,BCIR_CIR_NAME,"t%u",rid); return buf;
}
static int is_named_local(const bcir_func *f,uint32_t rid){
  const bcir_resource *r=res_of(f,rid); if(!r||!r->name[0]) return 0;
  if(r->read_only) return 0;                                          /* a global, defined in source */
  for(int i=0;i<f->n_params;i++) if(f->params[i].rid==rid) return 0;   /* a param, not a local */
  return 1;
}
static size_t emit_func(const bcir_func *f,char *o,size_t on){
  size_t w=0; char a[BCIR_CIR_NAME],b[BCIR_CIR_NAME],d[BCIR_CIR_NAME],e[BCIR_CIR_NAME],ty[64];
  ctype_str(&f->ret,ty,sizeof ty);
  w+=snprintf(o+w,on-w,"static %s bcir_%s(",ty,f->name);
  if(f->n_params==0) w+=snprintf(o+w,on-w,"void");
  for(int i=0;i<f->n_params;i++){char pt[64];ctype_str(&f->params[i].type,pt,sizeof pt);
    w+=snprintf(o+w,on-w,"%s%s %s",i?", ":"",pt,f->params[i].name);}
  w+=snprintf(o+w,on-w,")\n{\n");
  /* declare named locals up front (mutable storage -- branch merges + loop accumulators) */
  for(size_t i=0;i<f->n_res;i++){const bcir_resource *r=&f->res[i];
    if(is_named_local(f,r->rid)){
      int sx=-1; for(int k=0;k<f->n_statics;k++) if(!strcmp(f->statics[k].name,r->name)){sx=k;break;}
      if(sx>=0) w+=snprintf(o+w,on-w,"  static uint32_t %s = %lluu;\n",r->name,(unsigned long long)f->statics[sx].init);
      else if(r->kind==BCIR_RK_AGGREGATE&&r->agg[0]) w+=snprintf(o+w,on-w,"  %s %s;\n",r->agg,r->name);
      else w+=snprintf(o+w,on-w,"  uint32_t %s;\n",r->name);}}
  int depth=1, lstk[64], nls=0, lctr=0;   /* loop-id stack + counter for the `continue` labels */
  #define IND() do{ for(int _k=0;_k<depth;_k++) w+=snprintf(o+w,on-w,"  "); }while(0)
  for(size_t i=0;i<f->n_claims&&w<on-160;i++){const bcir_claim *cl=&f->claims[i];
    /* L6 control-flow markers (rendered as braces) */
    if(!strcmp(cl->op,"c.if")){IND();w+=snprintf(o+w,on-w,"if (%s) {\n",rname(f,cl->rd[0],a));depth++;continue;}
    if(!strcmp(cl->op,"c.else")){depth--;IND();w+=snprintf(o+w,on-w,"} else {\n");depth++;continue;}
    if(!strcmp(cl->op,"c.endif")){depth--;IND();w+=snprintf(o+w,on-w,"}\n");continue;}
    if(!strcmp(cl->op,"c.loop")){IND();w+=snprintf(o+w,on-w,"while (1) {\n");depth++;
      if(nls<64)lstk[nls++]=lctr++;continue;}
    if(!strcmp(cl->op,"c.loop.test")){IND();w+=snprintf(o+w,on-w,"if (!%s) break;\n",rname(f,cl->rd[0],a));continue;}
    if(!strcmp(cl->op,"c.cont.tgt")){IND();w+=snprintf(o+w,on-w,"__cont_%d: ;\n",nls?lstk[nls-1]:0);continue;}
    if(!strcmp(cl->op,"c.endloop")){depth--;IND();w+=snprintf(o+w,on-w,"}\n");if(nls)nls--;continue;}
    if(!strcmp(cl->op,"c.break")){IND();w+=snprintf(o+w,on-w,"break;\n");continue;}
    if(!strcmp(cl->op,"c.continue")){IND();w+=snprintf(o+w,on-w,"goto __cont_%d;\n",nls?lstk[nls-1]:0);continue;}
    if(!strncmp(cl->op,"c.goto:",7)){IND();w+=snprintf(o+w,on-w,"goto %s;\n",cl->op+7);continue;}
    if(!strncmp(cl->op,"c.label:",8)){w+=snprintf(o+w,on-w,"%s:;\n",cl->op+8);continue;}
    if(!strcmp(cl->op,"c.return")){IND();
      if(cl->n_rd) w+=snprintf(o+w,on-w,"return %s;\n",rname(f,cl->rd[0],a));
      else w+=snprintf(o+w,on-w,"return;\n");continue;}
    IND();
    if(!strncmp(cl->op,"c.bin.",6))
      w+=snprintf(o+w,on-w,"uint32_t %s = %s %s %s;\n",rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),binop_c(cl->op+6),rname(f,cl->rd[1],b));
    else if(!strncmp(cl->op,"c.un.",5))
      w+=snprintf(o+w,on-w,"uint32_t %s = (%s%s);\n",rname(f,cl->wr[0],d),unop_c(cl->op+5),rname(f,cl->rd[0],a));
    else if(!strncmp(cl->op,"c.cast:",7))                  /* (type)operand -- width cast */
      w+=snprintf(o+w,on-w,"uint32_t %s = (%s)%s;\n",rname(f,cl->wr[0],d),cl->op+7,rname(f,cl->rd[0],a));
    else if(!strcmp(cl->op,"c.select"))                    /* ternary: cond ? then : els */
      w+=snprintf(o+w,on-w,"uint32_t %s = (%s ? %s : %s);\n",rname(f,cl->wr[0],d),
                  rname(f,cl->rd[0],a),rname(f,cl->rd[1],b),rname(f,cl->rd[2],e));
    else if(!strcmp(cl->op,"c.const"))
      w+=snprintf(o+w,on-w,"uint32_t %s = %lluu;\n",rname(f,cl->wr[0],d),(unsigned long long)cl->imm[0]);
    else if(!strcmp(cl->op,"c.copy")){
      if(is_named_local(f,cl->wr[0])) w+=snprintf(o+w,on-w,"%s = %s;\n",rname(f,cl->wr[0],d),rname(f,cl->rd[0],a));
      else w+=snprintf(o+w,on-w,"uint32_t %s = %s;\n",rname(f,cl->wr[0],d),rname(f,cl->rd[0],a));
    }else if(!strcmp(cl->op,"c.load")){
      const bcir_resource *br=res_of(f,cl->rd[0]); long long off=cl->n_imm?cl->imm[0]:0;
      if(cl->n_rd==2) w+=snprintf(o+w,on-w,"uint32_t %s = %s[%s];\n",rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),rname(f,cl->rd[1],b));
      else if(cl->domain==BCIR_DOM_MMIO)
        w+=snprintf(o+w,on-w,"uint32_t %s = *(volatile uint32_t *)((const volatile char *)%s + %lld);\n",rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),off);
      else { const char *amp=(br&&br->kind==BCIR_RK_POINTER)?"":"&"; long long fsz=cl->n_imm>1?cl->imm[1]:4;
        w+=snprintf(o+w,on-w,"uint32_t %s; { uint32_t _v=0; memcpy(&_v, (const char *)%s%s + %lld, %lld); %s = _v; }\n",rname(f,cl->wr[0],d),amp,rname(f,cl->rd[0],a),off,fsz,rname(f,cl->wr[0],d)); }
    }else if(!strcmp(cl->op,"c.store")){          /* L8: member store -> memcpy `size` bytes */
      const bcir_resource *br=res_of(f,cl->rd[0]); long long off=cl->imm[0]; long long sz=cl->n_imm>1?cl->imm[1]:4;
      if(cl->domain==BCIR_DOM_MMIO)
        w+=snprintf(o+w,on-w,"*(volatile uint32_t *)((volatile char *)%s + %lld) = %s;\n",rname(f,cl->rd[0],a),off,rname(f,cl->rd[1],b));
      else { const char *amp=(br&&br->kind==BCIR_RK_POINTER)?"":"&";
        w+=snprintf(o+w,on-w,"{ uint32_t _v = %s; memcpy((char *)%s%s + %lld, &_v, %lld); }\n",rname(f,cl->rd[1],b),amp,rname(f,cl->rd[0],a),off,sz); }
    }else if(!strcmp(cl->op,"c.bf.get"))
      w+=snprintf(o+w,on-w,"uint32_t %s = (%s >> %lld) & %lluu;\n",rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),(long long)cl->imm[0],(1ull<<cl->imm[1])-1);
    else if(!strcmp(cl->op,"c.bf.set")){          /* (old & ~(mask<<off)) | ((v & mask) << off) */
      long long off=cl->imm[0]; unsigned long long mask=(1ull<<cl->imm[1])-1, clear=~(mask<<off)&0xFFFFFFFFull;
      w+=snprintf(o+w,on-w,"uint32_t %s = (%s & %lluu) | ((%s & %lluu) << %lld);\n",
                  rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),clear,rname(f,cl->rd[1],b),mask,off); }
    else if(!strncmp(cl->op,"c.atomic.",9))      /* atomic RMW -> the matching builtin */
      w+=snprintf(o+w,on-w,"uint32_t %s = __atomic_fetch_%s(%s, %s, __ATOMIC_SEQ_CST);\n",
                  rname(f,cl->wr[0],d),cl->op+9,rname(f,cl->rd[0],a),rname(f,cl->rd[1],b));
    else if(!strncmp(cl->op,"c.cmpxchg.",10))     /* compare-and-swap -> the __sync CAS builtin */
      w+=snprintf(o+w,on-w,"uint32_t %s = __sync_%s_compare_and_swap(%s, %s, %s);\n",
                  rname(f,cl->wr[0],d),cl->op+10,rname(f,cl->rd[0],a),rname(f,cl->rd[1],b),rname(f,cl->rd[2],e));
    else if(!strcmp(cl->op,"c.fence"))
      w+=snprintf(o+w,on-w,"__atomic_thread_fence(__ATOMIC_SEQ_CST);\n");
    else if(!strncmp(cl->op,"c.c11atom.",10)){   /* C11 <stdatomic.h> generics on _Atomic objects */
      const char *fn=cl->op+10;                  /* fetch_add / fetch_sub / fetch_xor / load / store */
      if(!strcmp(fn,"load")) w+=snprintf(o+w,on-w,"uint32_t %s = atomic_load(%s);\n",rname(f,cl->wr[0],d),rname(f,cl->rd[0],a));
      else if(!strcmp(fn,"store")) w+=snprintf(o+w,on-w,"atomic_store(%s, %s);\n",rname(f,cl->rd[0],a),rname(f,cl->rd[1],b));
      else w+=snprintf(o+w,on-w,"uint32_t %s = atomic_%s(%s, %s);\n",rname(f,cl->wr[0],d),fn,rname(f,cl->rd[0],a),rname(f,cl->rd[1],b)); }
    else if(!strncmp(cl->op,"c.call:",7)){
      w+=snprintf(o+w,on-w,"uint32_t %s = bcir_%s(",rname(f,cl->wr[0],d),cl->op+7);
      for(int k=0;k<cl->n_rd;k++) w+=snprintf(o+w,on-w,"%s%s",k?", ":"",rname(f,cl->rd[k],a));
      w+=snprintf(o+w,on-w,");\n"); }
    else if(!strcmp(cl->op,"c.call.indirect")){    /* rd[0] is the function pointer; rd[1..] the args */
      w+=snprintf(o+w,on-w,"uint32_t %s = %s(",rname(f,cl->wr[0],d),rname(f,cl->rd[0],a));
      for(int k=1;k<cl->n_rd;k++) w+=snprintf(o+w,on-w,"%s%s",k>1?", ":"",rname(f,cl->rd[k],b));
      w+=snprintf(o+w,on-w,");\n"); }
    else if(!strncmp(cl->op,"c.call.imember:",15)){   /* o->fn(args): funcptr struct member */
      const char *sep=(cl->n_imm&&cl->imm[0])?"->":".";
      w+=snprintf(o+w,on-w,"uint32_t %s = %s%s%s(",rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),sep,cl->op+15);
      for(int k=1;k<cl->n_rd;k++) w+=snprintf(o+w,on-w,"%s%s",k>1?", ":"",rname(f,cl->rd[k],b));
      w+=snprintf(o+w,on-w,");\n"); }
  }
  #undef IND
  w+=snprintf(o+w,on-w,"}\n");
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
  if(c->ngv<16){ idcpy(c->gv[c->ngv].name,&nm); c->gv[c->ngv].ty=ty; c->gv[c->ngv].count=count; c->ngv++; }
}

/* --- public entry -------------------------------------------------------- */
int bcir_cfront_compile(const char *src, bcir_cfront_result *out) {
  static CC c; memset(&c,0,sizeof c); memset(out,0,sizeof *out);
  c.rid=100; c.cid=1000;
  strtab_reset();                       /* fresh string-literal table per translation unit */
  lex(&c,src);
  while(!isk(&c,T_END)&&!c.failed&&out->unit.n_funcs<BCIR_MAX_FUNCS){
    if(try_top_decl(&c)) continue;       /* typedef / enum / struct|union defs, interleaved */
    if(isk(&c,T_END)||c.failed) break;
    if(looks_global(&c)){ p_global(&c); continue; }   /* a file-scope global (lookup table) */
    bcir_func *fn=&out->unit.funcs[out->unit.n_funcs];
    fn->cap_res=256; fn->res=calloc(256,sizeof(bcir_resource));
    fn->cap_claims=4096; fn->claims=calloc(4096,sizeof(bcir_claim));
    if(!fn->res||!fn->claims){snprintf(out->diag,sizeof out->diag,"oom");return 1;}
    c.rid=100+out->unit.n_funcs*1000; c.cid=1000+out->unit.n_funcs*1000;
    if(p_func(&c,fn)){snprintf(out->diag,sizeof out->diag,"%s",c.err);return 1;}
    out->unit.n_funcs++;
  }
  if(c.failed){snprintf(out->diag,sizeof out->diag,"%s",c.err);return 1;}
  out->ok=bcir_verify_unit(&out->unit,out->diag,sizeof out->diag);
  /* C.2 verified-C attestation: stamp the emitted C with its R-law status + R13 digest. */
  const bcir_func *entry = out->unit.n_funcs ? &out->unit.funcs[out->unit.n_funcs-1] : NULL;
  size_t w=snprintf(out->emitted,sizeof out->emitted,
    "/* BCIR verified-C attestation (C.2) -- generated by bcir_cfront, do not edit.\n"
    " *   R1-R8 + R18  %s\n"
    " *   R9 plan / R10-R11 pack  checked in the compile->execute loop\n"
    " *   R12 lowering-contract  support preserved (emit Clang-behaviour-equivalent)\n"
    " *   R13 provenance digest  %016llx\n"
    " *   R17 accuracy  exact (integer / Q-fixed, 0 ULP)\n */\n",
    out->ok?"clean":"DIRTY", entry?(unsigned long long)bcir_provenance_digest(entry):0ull);
  for(int i=0;i<out->unit.n_funcs && w<sizeof out->emitted-256;i++){
    w+=emit_func(&out->unit.funcs[i],out->emitted+w,sizeof out->emitted-w);
    if(i+1<out->unit.n_funcs) w+=snprintf(out->emitted+w,sizeof out->emitted-w,"\n");
  }
  return 0;
}

void bcir_cfront_free(bcir_cfront_result *out){
  for(int i=0;i<out->unit.n_funcs;i++){free(out->unit.funcs[i].res);free(out->unit.funcs[i].claims);}
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
  snprintf(buf,n,"funcs=%d claims=%zu mmio=%d bf=%d const=%d binop=%d call=%d ok=%d",
           u->n_funcs,nc,mmio,bf,kn,binop,calls,ok);
}
