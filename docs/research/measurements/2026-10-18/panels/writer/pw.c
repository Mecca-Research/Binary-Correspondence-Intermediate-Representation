/* PANELS-WRITER screen: the SwiGLU writers at 1024 x 1024 on huge pages, rotated arm by arm, round
 * by round, in one process: the binary32 entries, the shipped panel writers (two passes, the second
 * panel read back from the A panel), the shipped writers with streamed stores, and the strip writer
 * (v4) at several strip widths and store kinds. Every panel arm's panels are compared with the
 * shipped writer's byte for byte at the end of every round. ./pw THREADS REPS */
#define _DEFAULT_SOURCE
#include "bcir_tensor.h"
#include <sys/mman.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
int bcir_tensor_silu_mul_panels4_pool(bcir_tensor_pool *pool, int v2, size_t r, size_t c,
    const float *x, const float *u, unsigned short *a, unsigned short *b, int *bad, size_t strip, int nt);
int bcir_tensor_swiglu_backward_panels4_pool(bcir_tensor_pool *pool, int v2, size_t r, size_t c,
    const float *g, const float *u, const float *dy, unsigned short *dga, unsigned short *dgt,
    unsigned short *dua, unsigned short *dut, size_t strip, int nt);
int bcir_tensor_silu_mul_panels_nt_pool(bcir_tensor_pool *pool, size_t r, size_t c, const float *x,
    const float *u, unsigned short *a, unsigned short *b, int *bad, int nt);
int bcir_tensor_swiglu_backward_panels_nt_pool(bcir_tensor_pool *pool, size_t r, size_t c,
    const float *g, const float *u, const float *dy, unsigned short *dga, unsigned short *dgt,
    unsigned short *dua, unsigned short *dut, int nt);
static unsigned char *arena_at,*arena_end;
static void *huge(size_t bytes) {
  void *p;
  if (!arena_at) {
    size_t total=(size_t)3<<30; unsigned char *b=mmap(NULL,total+(2u<<20),PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (b==MAP_FAILED) abort();
    arena_at=(unsigned char *)(((uintptr_t)b+(2u<<20)-1)&~(uintptr_t)((2u<<20)-1)); arena_end=arena_at+total;
    (void)madvise(arena_at,total,MADV_HUGEPAGE);
  }
  p=arena_at; arena_at+=(bytes+4095)/4096*4096; if (arena_at>arena_end) abort(); memset(p,0,bytes); return p;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int cmpd(const void *a,const void *b) { double x=*(const double *)a,y=*(const double *)b; return (x>y)-(x<y); }
typedef struct arm { const char *name; int op,kind; size_t strip; int nt; } arm;  /* kind 0 fp32, 1 shipped, 2 shipped+nt, 3 v4 */
static const arm arms[]={
  {"silu*u fp32",0,0,0,0},{"silu*u shipped",0,1,0,0},{"silu*u shipped nt2",0,2,0,2},{"silu*u shipped nt3",0,2,0,3},
  {"silu*u v4 64",0,3,64,0},{"silu*u v4 128",0,3,128,0},{"silu*u v4 256",0,3,256,0},
  {"silu*u v4 128 nt2",0,3,128,2},{"silu*u v4 256 nt2",0,3,256,2},{"silu*u v4 128 nt3",0,3,128,3},
  {"VJP fp32",1,0,0,0},{"VJP shipped",1,1,0,0},{"VJP shipped nt2",1,2,0,2},{"VJP shipped nt3",1,2,0,3},
  {"VJP v4 64",1,3,64,0},{"VJP v4 128",1,3,128,0},{"VJP v4 256",1,3,256,0},
  {"VJP v4 128 nt2",1,3,128,2},{"VJP v4 256 nt2",1,3,256,2},{"VJP v4 128 nt3",1,3,128,3},
};
#define ARMS ((int)(sizeof arms/sizeof arms[0]))
int main(int argc,char **argv) {
  unsigned threads=argc>1 ? (unsigned)atoi(argv[1]) : 4; int reps=argc>2 ? atoi(argv[2]) : 31,r,k,q,bad,same[ARMS],ship[2];
  size_t n=1024,f=1024,nf=n*f,i; size_t pb=bcir_tensor_pool_size(threads); void *pm=huge((pb+63)/64*64);
  bcir_tensor_pool *pool=threads>1 ? bcir_tensor_pool_init(pm,pb,threads) : NULL;
  float *g=huge(nf*4),*u=huge(nf*4),*dy=huge(nf*4),*y[ARMS][2]; unsigned short *p[ARMS][4]; static double t[ARMS][256];
  for (k=0;k<ARMS;k++) {
    same[k]=1;
    if (arms[k].kind==0) { y[k][0]=huge(nf*4); y[k][1]=arms[k].op ? huge(nf*4) : NULL; }
    else for (i=0;i<(arms[k].op ? 4u : 2u);i++) p[k][i]=huge(nf*2);
    if (arms[k].kind==1) ship[arms[k].op]=k;
  }
  for (i=0;i<nf;i++) { g[i]=2.0f*sinf((float)i*0.013f); u[i]=0.5f*cosf((float)i*0.007f); dy[i]=0.01f*sinf((float)i*0.011f); }
  for (r=-2;r<reps;r++) {
    for (q=0;q<ARMS;q++) {
      const arm *a; double t0;
      k=(q+(r<0 ? 0 : r))%ARMS; a=&arms[k]; t0=now();
      if (!a->op) {
        if (a->kind==0) bcir_tensor_silu_mul_v2_scan_pool(pool,nf,g,u,y[k][0],&bad);
        else if (a->kind==1) bcir_tensor_silu_mul_panels_pool(pool,1,n,f,g,u,p[k][0],p[k][1],&bad);
        else if (a->kind==2) bcir_tensor_silu_mul_panels_nt_pool(pool,n,f,g,u,p[k][0],p[k][1],&bad,a->nt);
        else bcir_tensor_silu_mul_panels4_pool(pool,1,n,f,g,u,p[k][0],p[k][1],&bad,a->strip,a->nt);
      } else {
        if (a->kind==0) bcir_tensor_swiglu_backward_v2_pool(pool,nf,g,u,dy,y[k][0],y[k][1]);
        else if (a->kind==1) bcir_tensor_swiglu_backward_panels_pool(pool,1,n,f,g,u,dy,p[k][0],p[k][1],p[k][2],p[k][3]);
        else if (a->kind==2) bcir_tensor_swiglu_backward_panels_nt_pool(pool,n,f,g,u,dy,p[k][0],p[k][1],p[k][2],p[k][3],a->nt);
        else bcir_tensor_swiglu_backward_panels4_pool(pool,1,n,f,g,u,dy,p[k][0],p[k][1],p[k][2],p[k][3],a->strip,a->nt);
      }
      if (r>=0) t[k][r]=1e3*(now()-t0);
    }
    for (k=0;k<ARMS;k++) if (arms[k].kind>1) {
      int s=ship[arms[k].op];
      for (i=0;i<(arms[k].op ? 4u : 2u);i++) same[k]&=!memcmp(p[k][i],p[s][i],nf*2);
    }
  }
  for (k=0;k<ARMS;k++) qsort(t[k],reps,sizeof(double),cmpd);
  printf("threads %u reps %d, median ms (strip: columns a strip; nt: 2 the second panel streamed, 3 both)\n",threads,reps);
  for (k=0;k<ARMS;k++) {
    int s=ship[arms[k].op];
    printf("  %-22s %8.3f", arms[k].name, t[k][reps/2]);
    if (arms[k].kind>1) printf("  x%.3f vs shipped  identical: %s", t[s][reps/2]/t[k][reps/2], same[k] ? "yes" : "NO");
    printf("\n");
  }
  bcir_tensor_pool_stop(pool); return 0;
}
