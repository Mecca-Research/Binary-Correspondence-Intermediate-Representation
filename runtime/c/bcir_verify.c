/*===- bcir_verify.c - the BCIR verifier in C (the C twin of bcir/verify) --===*/
#include "bcir_verify.h"

#include <stdio.h>   /* snprintf for the diagnostic path (the verdict logic itself is pure) */
#include <stdlib.h>  /* calloc/free for the R18 cycle-detection visited array (no fixed func cap) */

static int slen(const char *s){int n=0;while(s[n])n++;return n;}
static int seq(const char *a,const char *b){int i=0;for(;a[i]&&b[i];i++)if(a[i]!=b[i])return 0;return a[i]==b[i];}

static const bcir_resource *res_of(const bcir_func *f,uint32_t rid){
  for(size_t i=0;i<f->n_res;i++) if(f->res[i].rid==rid) return &f->res[i]; return NULL;
}

/* R6 lane legality: which lanes a declared access-pattern shape admits (mirror bcir/verify's
 * _LEGAL_LANES). Lane A (atomic) is legal for SCALAR (a single-location counter) and RANDOM
 * (a scatter-atomic). */
static int lane_legal(bcir_stride sc,uint8_t lane){
  switch(sc){
    case BCIR_STRIDE_SCALAR:    return lane==BCIR_LANE_U||lane==BCIR_LANE_H||lane==BCIR_LANE_A;
    case BCIR_STRIDE_UNIT:      return lane==BCIR_LANE_U;
    case BCIR_STRIDE_STRIDED:   return lane==BCIR_LANE_U||lane==BCIR_LANE_GGG;
    case BCIR_STRIDE_CACHELINE: return lane==BCIR_LANE_UX||lane==BCIR_LANE_GGG;
    case BCIR_STRIDE_TILE:      return lane==BCIR_LANE_T;
    case BCIR_STRIDE_RANDOM:    return lane==BCIR_LANE_GGG||lane==BCIR_LANE_A;
  }
  return 0;
}

/* --- R1-R8 + R12 + R17 (per function) ------------------------------------ */
static int verify_func(const bcir_func *f,char *diag,size_t dn){
  int effect=0;
  for(size_t i=0;i<f->n_claims;i++){const bcir_claim *cl=&f->claims[i];
    for(int k=0;k<cl->n_rd;k++) if(!res_of(f,cl->rd[k])){snprintf(diag,dn,"R2: claim %u reads undefined rid %u",cl->id,cl->rd[k]);return 0;}
    for(int k=0;k<cl->n_wr;k++) if(!res_of(f,cl->wr[k])){snprintf(diag,dn,"R2: claim %u writes undefined rid %u",cl->id,cl->wr[k]);return 0;}
    int dom_ok=0;
    for(int k=0;k<cl->n_rd;k++){const bcir_resource *r=res_of(f,cl->rd[k]);if(r&&r->domain==cl->domain)dom_ok=1;}
    for(int k=0;k<cl->n_wr;k++){const bcir_resource *r=res_of(f,cl->wr[k]);if(r&&r->domain==cl->domain)dom_ok=1;}
    if((cl->n_rd||cl->n_wr)&&!dom_ok){snprintf(diag,dn,"R3: claim %u domain mismatch",cl->id);return 0;}
    for(int k=0;k<cl->n_wr;k++){const bcir_resource *r=res_of(f,cl->wr[k]);
      if(r&&r->domain==BCIR_DOM_MMIO&&cl->hazard==BCIR_HZ_UNIQUE){snprintf(diag,dn,"R3: MMIO write %u needs a barriered hazard",cl->id);return 0;}}
    /* R5: an atomic lane or an atomic/cmpxchg/barrier opcode needs a non-unique hazard. */
    int atomicish = cl->lane==BCIR_LANE_A || (cl->opcode>=BCIR_OP_ATOMIC_ADD && cl->opcode<=BCIR_OP_BARRIER);
    if(atomicish && cl->hazard==BCIR_HZ_UNIQUE){snprintf(diag,dn,"R5: claim %u (atomic/fence) needs an atomic/barriered hazard",cl->id);return 0;}
    /* R6: the claim's lane must be legal for its declared access-pattern shape. */
    if(!lane_legal(cl->stride,cl->lane)){snprintf(diag,dn,"R6: claim %u lane %u illegal for stride %d",cl->id,cl->lane,(int)cl->stride);return 0;}
    if(!seq(cl->op,"") && cl->opcode!=BCIR_OP_NOP && (cl->n_wr||cl->opcode==BCIR_OP_STORE||cl->opcode==BCIR_OP_BARRIER)) effect=1;
  }
  /* R12 (lowering contract / support): a lowered function preserves an observable effect --
   * a return value or a store/barrier. (A pure no-op function would discharge nothing.) */
  if(f->n_claims && !effect && !f->has_return){snprintf(diag,dn,"R12: function %s discharges no effect",f->name);return 0;}
  /* R17 (accuracy): the C subset is integer/Q-fixed -- exact, 0 ULP -- so it holds by construction. */
  return 1;
}

/* --- R18 (compositional call-graph integrity) ---------------------------- */
static int func_index(const bcir_unit *u,const char *nm){
  for(int i=0;i<u->n_funcs;i++) if(seq(u->funcs[i].name,nm)) return i; return -1;
}
static int has_cycle(const bcir_unit *u,int fi,int *state){
  state[fi]=1;
  for(int k=0;k<u->funcs[fi].n_calls;k++){int ci=func_index(u,u->funcs[fi].calls[k]);
    if(ci<0)continue; if(state[ci]==1)return 1; if(state[ci]==0&&has_cycle(u,ci,state))return 1;}
  state[fi]=2; return 0;
}

int bcir_verify_unit(const bcir_unit *u,char *diag,size_t dn){
  if(dn) diag[0]=0;
  for(int i=0;i<u->n_funcs;i++) if(!verify_func(&u->funcs[i],diag,dn)) return 0;
  for(int i=0;i<u->n_funcs;i++) for(int k=0;k<u->funcs[i].n_calls;k++)
    if(func_index(u,u->funcs[i].calls[k])<0){snprintf(diag,dn,"R18: call to undefined function %s",u->funcs[i].calls[k]);return 0;}
  int *state=calloc((size_t)(u->n_funcs>0?u->n_funcs:1),sizeof *state);   /* visited per func (no cap) */
  if(!state){snprintf(diag,dn,"oom");return 0;}
  for(int i=0;i<u->n_funcs;i++) if(state[i]==0&&has_cycle(u,i,state)){snprintf(diag,dn,"R18: recursive call cycle");free(state);return 0;}
  free(state);
  /* R14-R16 (CIM dispatch / DVFS clock / alloc placement): the scalar C subset emits no such
   * smart-lowering decisions, so these are vacuously satisfied here. */
  (void)slen;
  return 1;
}

/* --- R9 (plan legality) -------------------------------------------------- */
int bcir_verify_plan(const bcir_func *f,const bcir_plan *p,char *diag,size_t dn){
  if(dn) diag[0]=0;
  if(p->n != f->n_claims){snprintf(diag,dn,"R9: plan covers %zu of %zu claims",p->n,f->n_claims);return 0;}
  uint64_t total=0;
  for(size_t i=0;i<p->n;i++){
    if(p->steps[i].claim_id != f->claims[i].id){snprintf(diag,dn,"R9: plan step %zu realizes the wrong claim",i);return 0;}
    if(p->steps[i].width==0){snprintf(diag,dn,"R9: claim %u realized at width 0",p->steps[i].claim_id);return 0;}
    /* each claim realized exactly once */
    for(size_t j=0;j<i;j++) if(p->steps[j].claim_id==p->steps[i].claim_id){snprintf(diag,dn,"R9: claim %u realized more than once",p->steps[i].claim_id);return 0;}
    total+=p->steps[i].cost;
  }
  if(total!=p->total_cost){snprintf(diag,dn,"R9: plan total %llu != sum of step costs %llu",(unsigned long long)p->total_cost,(unsigned long long)total);return 0;}
  return 1;
}

/* --- R10-R11 (StreamPack laws) ------------------------------------------- */
static uint32_t rd32(const uint8_t *p){return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);}
int bcir_verify_pack(const uint8_t *pack,size_t len,uint32_t expect_segs,char *diag,size_t dn){
  if(dn) diag[0]=0;
  bcir_streampack_header h;
  bcir_status st = bcir_sp_validate(pack,len,&h);     /* R11: magic + version + CRC + bounds */
  if(st!=BCIR_OK){snprintf(diag,dn,"R11: StreamPack invalid (status %d)",(int)st);return 0;}
  if(rd32(pack+20)!=expect_segs){snprintf(diag,dn,"R10: pack has %u segments, expected %u",rd32(pack+20),expect_segs);return 0;}
  return 1;
}

/* --- R13 (provenance digest) --------------------------------------------- */
uint64_t bcir_provenance_digest(const bcir_func *f){
  uint64_t h=1469598103934665603ull;                  /* FNV-1a over the claim-graph structure */
  #define MIX(x) do{ h^=(uint64_t)(x); h*=1099511628211ull; }while(0)
  for(size_t i=0;i<f->n_claims;i++){const bcir_claim *cl=&f->claims[i];
    MIX(cl->id); MIX(cl->opcode); MIX(cl->lane); MIX(cl->domain); MIX(cl->hazard);
    for(int k=0;k<cl->n_rd;k++) MIX(cl->rd[k]);
    for(int k=0;k<cl->n_wr;k++) MIX(cl->wr[k]);
    for(int k=0;k<cl->n_imm;k++) MIX((uint64_t)cl->imm[k]);
    for(int k=0;cl->op[k];k++) MIX((unsigned char)cl->op[k]);
  }
  #undef MIX
  return h;
}

/* --- R21 (pointer-lifetime legality: use-after-free / double-free, §5.12) ----
 * The exact mirror of bcir/verify::verify_lifetime. Walks each function's claims IN ORDER with a set
 * `freed` of currently-freed resource rids (reset per function). For each claim: FIRST, for every rid it
 * READS, if that rid is freed, report -- `double-free` if THIS claim is a `free` event, else `use-after-free`.
 * THEN, if this claim is `free`, add its read rids to `freed`. THEN, for every rid it WRITES, discard it
 * from `freed` (a write / alloc re-validates). `alloc` (lifetime==1) and `use` (lifetime==0) are identical
 * for the read-check; only `free` (lifetime==2) distinguishes the kind and marks rids freed.
 *
 * ADVISORY: separate from the bcir_verify_unit verdict -- it never changes r.ok or the summary. Reports
 * each diagnostic via `report(funcname, kind, ctx)`; functions are small, so `freed` is a fixed array. */
void bcir_verify_lifetime(const bcir_unit *u,
                          void (*report)(const char *funcname, const char *kind, void *ctx),
                          void *ctx){
  for(int fi=0;fi<u->n_funcs;fi++){
    const bcir_func *f=&u->funcs[fi];
    uint32_t freed[256]; int nfreed=0;                 /* freed-and-not-re-allocated rids (functions are small) */
    for(size_t i=0;i<f->n_claims;i++){
      const bcir_claim *cl=&f->claims[i];
      int is_free = (cl->lifetime==2);
      for(int k=0;k<cl->n_rd;k++){                      /* a READ of a freed resource is the dangling deref */
        for(int j=0;j<nfreed;j++) if(freed[j]==cl->rd[k]){
          report(f->name, is_free?"double-free":"use-after-free", ctx); break; }
      }
      if(is_free){                                      /* the read resources die after this claim */
        for(int k=0;k<cl->n_rd;k++){
          int seen=0; for(int j=0;j<nfreed;j++) if(freed[j]==cl->rd[k]){seen=1;break;}
          if(!seen && nfreed<(int)(sizeof freed/sizeof freed[0])) freed[nfreed++]=cl->rd[k];
        }
      }
      for(int k=0;k<cl->n_wr;k++){                      /* a WRITE (reassignment / alloc) re-validates */
        for(int j=0;j<nfreed;j++) if(freed[j]==cl->wr[k]){freed[j]=freed[--nfreed];j--;}
      }
    }
  }
}
