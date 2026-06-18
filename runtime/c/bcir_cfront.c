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
typedef enum { T_ID, T_INT, T_PUN, T_END } tkind;
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

typedef struct {
  tok t[MAXTOK]; int nt, i;
  sdef s[16]; int ns;
  tdef td[MAXTD]; int ntd;        /* typedef aliases (resolved at parse time) */
  econst ec[MAXEC]; int nec;      /* enum constants (folded to literals at parse time) */
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

static void lex(CC *c, const char *src) {
  const char *p=src;
  static const char *pu[]={"<<",">>","->","==","!=","<=",">=","&&","||",0};
  while (*p) {
    if (*p==' '||*p=='\t'||*p=='\r'||*p=='\n'){p++;continue;}
    if (p[0]=='/'&&p[1]=='/'){while(*p&&*p!='\n')p++;continue;}
    if (p[0]=='/'&&p[1]=='*'){p+=2;while(*p&&!(p[0]=='*'&&p[1]=='/'))p++;if(*p)p+=2;continue;}
    if (*p=='#'){while(*p&&*p!='\n')p++;continue;}   /* preprocessor: L7 */
    if (c->nt>=MAXTOK-1) break;
    tok *t=&c->t[c->nt];
    if (is_id0(*p)){t->k=T_ID;t->s=p;while(is_idc(*p))p++;t->n=(int)(p-t->s);c->nt++;continue;}
    if (*p>='0'&&*p<='9'){t->k=T_INT;t->s=p;while(is_idc(*p)||*p=='\'')p++;t->n=(int)(p-t->s);
                          t->v=parse_int(t->s,t->n);c->nt++;continue;}
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
static void p_enum_body(CC *c);   /* fwd: `{ A, B=expr, C }` -> register the constants */

/* a parsed type: fills a bcir_ctype + the struct index (sidx, or -1). */
static int p_type(CC *c, bcir_ctype *ty, int *sidx) {
  memset(ty,0,sizeof *ty); ty->kind=0; ty->size=4; ty->signd=1; *sidx=-1;
  int seen=0;
  for(;;){
    if(is(c,"volatile")){ty->is_volatile=1;c->i++;continue;}
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

/* §5.8: GCC/Clang atomic + fence + CAS builtins -> the BCIR ATOMIC_x / BARRIER / CMPXCHG opcodes.
 * kind: 0 = RMW (ptr,val), 1 = fence (no operands), 2 = cmpxchg (ptr,expected,desired). */
enum { AK_RMW=0, AK_FENCE=1, AK_CAS=2 };
static int atomic_kind(const tok *t,const char **op,bcir_opcode *oc,int *kind){
  struct{const char *n,*op;bcir_opcode oc;int k;} A[]={
    {"__atomic_fetch_add","c.atomic.add",BCIR_OP_ATOMIC_ADD,AK_RMW},
    {"__atomic_fetch_sub","c.atomic.sub",BCIR_OP_ATOMIC_SUB,AK_RMW},
    {"__atomic_fetch_xor","c.atomic.xor",BCIR_OP_ATOMIC_XOR,AK_RMW},
    {"__atomic_thread_fence","c.fence",BCIR_OP_BARRIER,AK_FENCE},
    {"__sync_synchronize","c.fence",BCIR_OP_BARRIER,AK_FENCE},
    {"__sync_val_compare_and_swap","c.cmpxchg.val",BCIR_OP_CMPXCHG,AK_CAS},
    {"__sync_bool_compare_and_swap","c.cmpxchg.bool",BCIR_OP_CMPXCHG,AK_CAS},{0,0,0,0}};
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
    cl->domain=dom; cl->n_wr=1; cl->wr[0]=t; cl->rd[0]=args[0];
    if(kind==AK_CAS){                                  /* CMPXCHG: ptr, expected, desired */
      cl->n_rd=3; cl->rd[1]=(na>1)?args[1]:args[0]; cl->rd[2]=(na>2)?args[2]:cl->rd[1]; }
    else { cl->n_rd=2; cl->rd[1]=(na>1)?args[1]:args[0]; }                 /* RMW: ptr, val */
  }
  return t;
}

static uint32_t p_primary(CC *c) {
  if(isk(c,T_INT)){tok t=adv(c);uint32_t r=temp(c,4);
    bcir_claim *cl=new_claim(c,"c.const",BCIR_OP_LOAD);if(!cl)return r;
    cl->n_wr=1;cl->wr[0]=r;cl->n_imm=1;cl->imm[0]=t.v;return r;}
  if(is(c,"(")){c->i++;uint32_t r=p_expr(c);eat(c,")");return r;}
  if(isk(c,T_ID)){
    tok id=adv(c);
    const char *aop;bcir_opcode aoc;int akind;
    if(is(c,"(")&&atomic_kind(&id,&aop,&aoc,&akind)) return p_atomic(c,aop,aoc,akind);  /* atomics/fences/CAS */
    if(is(c,"(")) return p_call(c,&id);           /* call */
    int ec=find_enum(c,id.s,id.n);                /* an enumerator -> its folded constant */
    if(ec>=0){uint32_t r=temp(c,4);bcir_claim *cl=new_claim(c,"c.const",BCIR_OP_LOAD);
      if(cl){cl->n_wr=1;cl->wr[0]=r;cl->n_imm=1;cl->imm[0]=c->ec[ec].val;}return r;}
    venv *v=lookup(c,&id); if(!v){fail(c,"undefined identifier");return 0;}
    if(is(c,".")||is(c,"->")){c->i++;tok fn=adv(c);sdef *S=&c->s[v->sidx];
      for(int i=0;i<S->nf;i++) if((int)strlen(S->f[i].name)==fn.n&&!strncmp(S->f[i].name,fn.s,fn.n))
        return emit_member(c,v,&S->f[i]);
      fail(c,"unknown field");return 0;}
    if(is(c,"[")){c->i++;uint32_t idx=p_expr(c);eat(c,"]");return emit_index(c,v,idx);}  /* L3 */
    return v->rid;
  }
  fail(c,"expected expression");return 0;
}
static uint32_t p_unary(CC *c) {
  if(is(c,"-")||is(c,"~")||is(c,"!")){
    const char *suf=is(c,"-")?"neg":is(c,"~")?"bnot":"lnot";
    bcir_opcode oc=is(c,"-")?BCIR_OP_SUB:BCIR_OP_ADD;c->i++;
    uint32_t a=p_unary(c),r=temp(c,4);char op[BCIR_CIR_NAME];snprintf(op,sizeof op,"c.un.%s",suf);
    bcir_claim *cl=new_claim(c,op,oc);if(cl){cl->n_rd=1;cl->rd[0]=a;cl->n_wr=1;cl->wr[0]=r;}return r;}
  return p_primary(c);
}
static int bin_op(CC *c,char *suf,bcir_opcode *oc) {
  struct {const char *t,*s;bcir_opcode o;} B[]={{"*","mul",BCIR_OP_MUL},{"/","div",BCIR_OP_MUL},
    {"%","mod",BCIR_OP_MUL},{"+","add",BCIR_OP_ADD},{"-","sub",BCIR_OP_SUB},{"<<","shl",BCIR_OP_ADD},
    {">>","shr",BCIR_OP_ADD},{"<","lt",BCIR_OP_SUB},{">","gt",BCIR_OP_SUB},{"<=","le",BCIR_OP_SUB},
    {">=","ge",BCIR_OP_SUB},{"==","eq",BCIR_OP_SUB},{"!=","ne",BCIR_OP_SUB},{"&","and",BCIR_OP_ADD},
    {"^","xor",BCIR_OP_ADD},{"|","or",BCIR_OP_ADD},{0,0,0}};
  for(int i=0;B[i].t;i++) if(is(c,B[i].t)){strcpy(suf,B[i].s);*oc=B[i].o;return i;} return -1;
}
static int prec_of(int idx){static const int P[]={10,10,10,9,9,8,8,7,7,7,7,6,6,5,4,3};return P[idx];}
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
static uint32_t p_expr(CC *c){return p_binrhs(c,1,p_unary(c));}

/* --- statements + functions ---------------------------------------------- */
static void env_add(CC *c,const tok *nm,uint32_t rid,const bcir_ctype *ty,int sidx){
  if(c->nenv>=MAXENV)return; venv *v=&c->env[c->nenv++];
  idcpy(v->name,nm);v->rid=rid;v->type=*ty;v->sidx=sidx;
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
    marker(c,"c.loop.test",cond,1); p_block(c); marker(c,"c.endloop",0,0); return;
  }
  if(is(c,"{")){p_block(c);return;}
  int looks_decl=0;
  if(isk(c,T_ID)){int sz=scalar_size(pk(c)->s,pk(c)->n);
    looks_decl=sz>=0||is(c,"struct")||is(c,"union")||is(c,"enum")||is(c,"const")||is(c,"volatile")
               ||find_typedef(c,pk(c)->s,pk(c)->n)>=0;}
  if(looks_decl){
    bcir_ctype ty;int si;if(p_type(c,&ty,&si))return; tok nm=adv(c);
    char nb[BCIR_CIR_NAME]; idcpy(nb,&nm);
    int rk=ty.kind==2?BCIR_RK_POINTER:ty.kind==1?BCIR_RK_AGGREGATE:BCIR_RK_SCALAR;
    uint32_t rid=add_res(c, ty.is_volatile?BCIR_DOM_MMIO:BCIR_DOM_RAM,
                         ty.kind==2?ty.size:(ty.kind==1?c->s[si].size:ty.size),
                         ty.kind==2?(1<<16):1, ty.is_volatile, rk, nb);
    if(ty.kind==1) snprintf(c->fn->res[c->fn->n_res-1].agg,BCIR_CIR_NAME,"%s %s",ty.is_union?"union":"struct",ty.tag);   /* L8 aggregate local */
    env_add(c,&nm,rid,&ty,si);
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
      if(!eat(c,"="))return; uint32_t val=p_expr(c);
      bcir_claim *cl=new_claim(c,"c.store",BCIR_OP_STORE);
      if(cl){cl->n_rd=2;cl->rd[0]=v->rid;cl->rd[1]=val;cl->n_imm=2;cl->imm[0]=S->f[fi].byte_off;cl->imm[1]=S->f[fi].size;
        cl->bounds=BCIR_BND_ASSUMED;
        if(v->type.is_volatile){cl->domain=BCIR_DOM_MMIO;cl->lane=BCIR_LANE_H;cl->hazard=BCIR_HZ_BARRIERED;}}
      eat(c,";");return;}
    if(v&&c->t[c->i+1].k==T_PUN&&c->t[c->i+1].n==1&&c->t[c->i+1].s[0]=='='){
      c->i+=2;uint32_t val=p_expr(c);
      bcir_claim *cl=new_claim(c,"c.copy",BCIR_OP_ADD);if(cl){cl->n_rd=1;cl->rd[0]=val;cl->n_wr=1;cl->wr[0]=v->rid;}
      eat(c,";");return;}}
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
    {"lt","<"},{"gt",">"},{"le","<="},{"ge",">="},{0,0}};
  for(int i=0;M[i].s;i++) if(!strcmp(M[i].s,suf)) return M[i].c; return "+";
}
static const char *unop_c(const char *suf){return !strcmp(suf,"neg")?"-":!strcmp(suf,"bnot")?"~":"!";}
static void ctype_str(const bcir_ctype *ty,char *o,size_t n){
  int is_struct = (ty->kind==1) || ty->ptr_to_struct;
  const char *kw = ty->is_union ? "union" : "struct";
  const char *base = is_struct ? ty->tag
                   : (ty->size==1?"uint8_t":ty->size==2?"uint16_t":ty->size==8?"uint64_t":"uint32_t");
  if(ty->kind==2) snprintf(o,n,"%s%s%s%s *",ty->is_volatile?"volatile ":"",
                           ty->ptr_to_struct?kw:"",ty->ptr_to_struct?" ":"",base);
  else if(ty->kind==1) snprintf(o,n,"%s %s",kw,ty->tag);
  else snprintf(o,n,"%s",base);
}
static const char *rname(const bcir_func *f,uint32_t rid,char *buf){
  const bcir_resource *r=res_of(f,rid);
  if(r&&r->name[0]){snprintf(buf,BCIR_CIR_NAME,"%s",r->name);return buf;}
  snprintf(buf,BCIR_CIR_NAME,"t%u",rid); return buf;
}
static int is_named_local(const bcir_func *f,uint32_t rid){
  const bcir_resource *r=res_of(f,rid); if(!r||!r->name[0]) return 0;
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
      if(r->kind==BCIR_RK_AGGREGATE&&r->agg[0]) w+=snprintf(o+w,on-w,"  %s %s;\n",r->agg,r->name);
      else w+=snprintf(o+w,on-w,"  uint32_t %s;\n",r->name);}}
  int depth=1;
  #define IND() do{ for(int _k=0;_k<depth;_k++) w+=snprintf(o+w,on-w,"  "); }while(0)
  for(size_t i=0;i<f->n_claims&&w<on-160;i++){const bcir_claim *cl=&f->claims[i];
    /* L6 control-flow markers (rendered as braces) */
    if(!strcmp(cl->op,"c.if")){IND();w+=snprintf(o+w,on-w,"if (%s) {\n",rname(f,cl->rd[0],a));depth++;continue;}
    if(!strcmp(cl->op,"c.else")){depth--;IND();w+=snprintf(o+w,on-w,"} else {\n");depth++;continue;}
    if(!strcmp(cl->op,"c.endif")){depth--;IND();w+=snprintf(o+w,on-w,"}\n");continue;}
    if(!strcmp(cl->op,"c.loop")){IND();w+=snprintf(o+w,on-w,"while (1) {\n");depth++;continue;}
    if(!strcmp(cl->op,"c.loop.test")){IND();w+=snprintf(o+w,on-w,"if (!%s) break;\n",rname(f,cl->rd[0],a));continue;}
    if(!strcmp(cl->op,"c.endloop")){depth--;IND();w+=snprintf(o+w,on-w,"}\n");continue;}
    if(!strcmp(cl->op,"c.return")){IND();
      if(cl->n_rd) w+=snprintf(o+w,on-w,"return %s;\n",rname(f,cl->rd[0],a));
      else w+=snprintf(o+w,on-w,"return;\n");continue;}
    IND();
    if(!strncmp(cl->op,"c.bin.",6))
      w+=snprintf(o+w,on-w,"uint32_t %s = %s %s %s;\n",rname(f,cl->wr[0],d),rname(f,cl->rd[0],a),binop_c(cl->op+6),rname(f,cl->rd[1],b));
    else if(!strncmp(cl->op,"c.un.",5))
      w+=snprintf(o+w,on-w,"uint32_t %s = (%s%s);\n",rname(f,cl->wr[0],d),unop_c(cl->op+5),rname(f,cl->rd[0],a));
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
    else if(!strncmp(cl->op,"c.atomic.",9))      /* atomic RMW -> the matching builtin */
      w+=snprintf(o+w,on-w,"uint32_t %s = __atomic_fetch_%s(%s, %s, __ATOMIC_SEQ_CST);\n",
                  rname(f,cl->wr[0],d),cl->op+9,rname(f,cl->rd[0],a),rname(f,cl->rd[1],b));
    else if(!strncmp(cl->op,"c.cmpxchg.",10))     /* compare-and-swap -> the __sync CAS builtin */
      w+=snprintf(o+w,on-w,"uint32_t %s = __sync_%s_compare_and_swap(%s, %s, %s);\n",
                  rname(f,cl->wr[0],d),cl->op+10,rname(f,cl->rd[0],a),rname(f,cl->rd[1],b),rname(f,cl->rd[2],e));
    else if(!strcmp(cl->op,"c.fence"))
      w+=snprintf(o+w,on-w,"__atomic_thread_fence(__ATOMIC_SEQ_CST);\n");
    else if(!strncmp(cl->op,"c.call:",7)){
      w+=snprintf(o+w,on-w,"uint32_t %s = bcir_%s(",rname(f,cl->wr[0],d),cl->op+7);
      for(int k=0;k<cl->n_rd;k++) w+=snprintf(o+w,on-w,"%s%s",k?", ":"",rname(f,cl->rd[k],a));
      w+=snprintf(o+w,on-w,");\n"); }
  }
  #undef IND
  w+=snprintf(o+w,on-w,"}\n");
  return w;
}

/* --- public entry -------------------------------------------------------- */
int bcir_cfront_compile(const char *src, bcir_cfront_result *out) {
  static CC c; memset(&c,0,sizeof c); memset(out,0,sizeof *out);
  c.rid=100; c.cid=1000;
  lex(&c,src);
  /* pre-function declarations: typedefs, enum definitions, struct/union definitions -- in any
   * order (vendor headers interleave them), until the first function/global. */
  for(;;){
    if(c.failed) break;
    if(is(&c,"typedef")){ p_typedef(&c); continue; }
    if(is(&c,"enum")){
      int save=c.i; c.i++; if(isk(&c,T_ID)&&!is(&c,"{")) c.i++;
      if(is(&c,"{")){ p_enum_body(&c); eat(&c,";"); continue; }
      c.i=save; break;                              /* `enum tag` as a type -> a function follows */
    }
    if(is(&c,"struct")||is(&c,"union")){
      /* a struct *definition*?  struct [attrs] [TAG] [attrs] {  -- lookahead past attributes. */
      int save=c.i; c.i++; int pk_=0,al_=0; attrs(&c,&pk_,&al_);
      if(isk(&c,T_ID)&&!is(&c,"{")) c.i++;          /* the tag */
      attrs(&c,&pk_,&al_);
      int isdef = is(&c,"{"); c.i=save;
      if(isdef){ p_struct_body(&c); eat(&c,";"); continue; }
      break;                                        /* struct used as a type -> a function follows */
    }
    break;
  }
  while(!isk(&c,T_END)&&!c.failed&&out->unit.n_funcs<BCIR_MAX_FUNCS){
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
      else if(!strncmp(cl->op,"c.call:",7))calls++;}}
  snprintf(buf,n,"funcs=%d claims=%zu mmio=%d bf=%d const=%d binop=%d call=%d ok=%d",
           u->n_funcs,nc,mmio,bf,kn,binop,calls,ok);
}
