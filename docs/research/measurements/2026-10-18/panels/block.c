/* BF16-PANELS-1 probe: the SwiGLU block of one layer at the 9.84M decoder's shapes (n = 1024 tokens,
 * d = 384, f = 1024), as the trainer runs it with binary32 tensors and the GEMMs' packs (old) and
 * with the panels (new), alternated round by round on one pool; outputs compared bit for bit.
 * Forward: SiLU(gate)*up, then the down projection. Backward: the down projection's two VJPs,
 * the SwiGLU VJP, the gate and up VJPs (dx chained, dW apart).
 *   cc -O3 -ffp-contract=off -fno-math-errno -pthread -I runtime/c block.c runtime/c/bcir_tensor.c -lm
 *   ./a.out THREADS REPS */
#define _DEFAULT_SOURCE
#include "bcir_tensor.h"
#include <sys/mman.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static unsigned char *arena_at,*arena_end;
static void *huge(size_t bytes) {
  void *p;
  if (!arena_at) {
    size_t total=(size_t)1<<30; unsigned char *b=mmap(NULL,total+(2u<<20),PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (b==MAP_FAILED) abort();
    arena_at=(unsigned char *)(((uintptr_t)b+(2u<<20)-1)&~(uintptr_t)((2u<<20)-1)); arena_end=arena_at+total;
    (void)madvise(arena_at,total,MADV_HUGEPAGE);
  }
  p=arena_at; arena_at+=(bytes+4095)/4096*4096; if (arena_at>arena_end) abort(); return p;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int cmpd(const void *a,const void *b) { double x=*(const double *)a,y=*(const double *)b; return (x>y)-(x<y); }
static float *fa(size_t n) { float *p=huge((n*4+63)/64*64); size_t i; for (i=0;i<n;i++) p[i]=0.02f*sinf((float)i*0.013f+(float)(n%7)); return p; }
int main(int argc,char **argv) {
  unsigned threads=argc>1 ? (unsigned)atoi(argv[1]) : 4; int reps=argc>2 ? atoi(argv[2]) : 21,r,arm;
  const size_t n=1024,d=384,f=1024,nf=n*f,nd=n*d;
  size_t pb=bcir_tensor_pool_size(threads); void *pm=huge((pb+63)/64*64);
  bcir_tensor_pool *pool=threads>1 ? bcir_tensor_pool_init(pm,pb,threads) : NULL;
  float *gate=fa(nf),*up=fa(nf),*wdown=fa(d*f),*wgate=fa(f*d),*wup=fa(f*d),*n2=fa(nd),*cur=fa(nd);
  float *hid[2],*y[2],*dh[2],*dg[2],*du[2],*dn[2],*gd[2],*gg[2],*gu[2];
  size_t ws=4*bcir_tensor_mm_bf16_workspace(1024,384,1024); void *w=huge((ws+63)/64*64);
  double *tf[2],*tb[2]; int bad;
  for (arm=0;arm<2;arm++) {
    hid[arm]=fa(nf); y[arm]=fa(nd); dh[arm]=fa(nf); dg[arm]=fa(nf); du[arm]=fa(nf); dn[arm]=fa(nd);
    gd[arm]=fa(d*f); gg[arm]=fa(f*d); gu[arm]=fa(f*d);
    tf[arm]=malloc(sizeof(double)*reps); tb[arm]=malloc(sizeof(double)*reps);
  }
  for (r=-2;r<reps;r++) for (int q=0;q<2;q++) {
    double t0,t1,t2; arm=(r&1) ? 1-q : q;
    if (!arm) {  /* binary32 and the packs */
      bcir_tensor_mm_call c,cx,cw,x2[2],w2[2];
      t0=now();
      bcir_tensor_silu_mul_v2_scan_pool(pool,nf,gate,up,hid[0],&bad);
      memset(&c,0,sizeof c); c.tb=1; c.m=n; c.n=d; c.k=f; c.alpha=1.0f; c.a=hid[0]; c.b=wdown; c.c=y[0];
      bcir_tensor_mm_bf16_group_pool(pool,NULL,0,&c,1,w,ws);
      t1=now();
      memset(&cx,0,sizeof cx); cx.m=n; cx.n=f; cx.k=d; cx.alpha=1.0f; cx.a=cur; cx.b=wdown; cx.c=dh[0];
      memset(&cw,0,sizeof cw); cw.ta=1; cw.m=d; cw.n=f; cw.k=n; cw.alpha=1.0f; cw.a=cur; cw.b=hid[0]; cw.c=gd[0];
      bcir_tensor_mm_bf16_group_pool(pool,&cx,1,&cw,1,w,ws);
      bcir_tensor_swiglu_backward_v2_pool(pool,nf,gate,up,dh[0],dg[0],du[0]);
      for (int k=0;k<2;k++) {
        memset(&x2[k],0,sizeof x2[k]); x2[k].m=n; x2[k].n=d; x2[k].k=f; x2[k].alpha=1.0f; x2[k].a=k ? du[0] : dg[0];
        x2[k].b=k ? wup : wgate; x2[k].beta=k ? 1.0f : 0.0f; x2[k].c=dn[0];
        memset(&w2[k],0,sizeof w2[k]); w2[k].ta=1; w2[k].m=f; w2[k].n=d; w2[k].k=n; w2[k].alpha=1.0f;
        w2[k].a=k ? du[0] : dg[0]; w2[k].b=n2; w2[k].c=k ? gu[0] : gg[0];
      }
      bcir_tensor_mm_bf16_group_pool(pool,x2,2,w2,2,w,ws);
      t2=now();
    } else {  /* the panels */
      bcir_tensor_mm_call c,cx,cw,x2[2],w2[2]; bcir_tensor_bf16_panels p1,p2[2],p4[4];
      unsigned short *ha=(unsigned short *)(void *)hid[1],*hb=ha+nf,*ga=(unsigned short *)(void *)dg[1],*gt=ga+nf;
      unsigned short *ua=(unsigned short *)(void *)du[1],*ut=ua+nf;
      t0=now();
      bcir_tensor_silu_mul_panels_pool(pool,1,n,f,gate,up,ha,hb,&bad);
      memset(&c,0,sizeof c); c.tb=1; c.m=n; c.n=d; c.k=f; c.alpha=1.0f; c.b=wdown; c.c=y[1];
      p1.a=ha; p1.b=NULL;
      bcir_tensor_mm_bf16_group_panels_pool(pool,NULL,0,&c,1,&p1,w,ws);
      t1=now();
      memset(&cx,0,sizeof cx); cx.m=n; cx.n=f; cx.k=d; cx.alpha=1.0f; cx.a=cur; cx.b=wdown; cx.c=dh[1];
      memset(&cw,0,sizeof cw); cw.ta=1; cw.m=d; cw.n=f; cw.k=n; cw.alpha=1.0f; cw.a=cur; cw.c=gd[1];
      p2[0].a=p2[0].b=NULL; p2[1].a=NULL; p2[1].b=hb;
      bcir_tensor_mm_bf16_group_panels_pool(pool,&cx,1,&cw,1,p2,w,ws);
      bcir_tensor_swiglu_backward_panels_pool(pool,1,n,f,gate,up,dh[1],ga,gt,ua,ut);
      for (int k=0;k<2;k++) {
        memset(&x2[k],0,sizeof x2[k]); x2[k].m=n; x2[k].n=d; x2[k].k=f; x2[k].alpha=1.0f;
        x2[k].b=k ? wup : wgate; x2[k].beta=k ? 1.0f : 0.0f; x2[k].c=dn[1];
        memset(&w2[k],0,sizeof w2[k]); w2[k].ta=1; w2[k].m=f; w2[k].n=d; w2[k].k=n; w2[k].alpha=1.0f;
        w2[k].b=n2; w2[k].c=k ? gu[1] : gg[1];
        p4[k].a=k ? ua : ga; p4[k].b=NULL; p4[2+k].a=k ? ut : gt; p4[2+k].b=NULL;
      }
      bcir_tensor_mm_bf16_group_panels_pool(pool,x2,2,w2,2,p4,w,ws);
      t2=now();
    }
    if (r>=0) { tf[arm][r]=1e3*(t1-t0); tb[arm][r]=1e3*(t2-t1); }
  }
  {
    int same=!memcmp(y[0],y[1],nd*4) && !memcmp(dh[0],dh[1],nf*4) && !memcmp(dn[0],dn[1],nd*4) &&
        !memcmp(gd[0],gd[1],d*f*4) && !memcmp(gg[0],gg[1],f*d*4) && !memcmp(gu[0],gu[1],f*d*4);
    for (arm=0;arm<2;arm++) { qsort(tf[arm],reps,sizeof(double),cmpd); qsort(tb[arm],reps,sizeof(double),cmpd); }
    printf("{\"threads\": %u, \"reps\": %d, \"old\": {\"forward_ms\": %.3f, \"backward_ms\": %.3f}, \"panels\": {\"forward_ms\": %.3f, \"backward_ms\": %.3f}, "
        "\"forward_ratio\": %.3f, \"backward_ratio\": %.3f, \"block_ratio\": %.3f, \"outputs_identical\": %s}\n",
        threads,reps,tf[0][reps/2],tb[0][reps/2],tf[1][reps/2],tb[1][reps/2],tf[0][reps/2]/tf[1][reps/2],
        tb[0][reps/2]/tb[1][reps/2],(tf[0][reps/2]+tb[0][reps/2])/(tf[1][reps/2]+tb[1][reps/2]),same ? "true" : "false");
  }
  bcir_tensor_pool_stop(pool); return 0;
}
