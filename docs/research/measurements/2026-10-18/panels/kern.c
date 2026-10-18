/* BF16-KERN (2026-10-18): candidates for the bf16 contract's AMX kernel on the packs of the tree it
 * is built against (run.sh builds it on d5bc457's runtime/c, the kernel before BF16-KERN, "shipped"):
 * each computes every 32 x 32 block of C over its whole depth, the depth tiles in order (a split
 * depth spills the accumulators exactly), so each must equal the shipped kernel bit for bit -- a
 * mismatch is printed and fails the run. pfC: the next block's lines of C prefetched for writing
 * during the depth loop (PREFETCHW); Bt1 / At1: the B (A) tiles loaded as streamed (TILELOADDT1);
 * s2: the loads software-pipelined; chunk16: the depth in chunks of 16 tiles, the A chunk kept in L1
 * over the columns; colsOut256: columns outer over 256-row regions, the B block kept. The compute
 * phase only (the panels packed before every timed call, as warm as the step leaves them), the 9.84M
 * decoder's 13 shapes, the candidates rotated round by round, medians in ms.
 *   cc -std=gnu11 -O3 -ffp-contract=off -fno-math-errno -pthread -I BASE/runtime/c kern.c -lm
 *   ./a.out THREADS REPS                                          (includes bcir_tensor.c) */
#include "bcir_tensor.c"
#include <stdio.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int cmpd(const void *a,const void *b) { double x=*(const double *)a,y=*(const double *)b; return (x>y)-(x<y); }
#define AMX __attribute__((target("amx-tile,amx-bf16,avx512f,prfchw")))
AMX static void cfg_tiles(void) {
  bf16_tilecfg cfg; int q; memset(&cfg,0,sizeof cfg); cfg.palette=1;
  for (q=0;q<8;q++) { cfg.rows[q]=16; cfg.colsb[q]=64; }
  _tile_loadconfig(&cfg);
}
/* one 32 x 32 block over depth tiles [q0, q1): start from zero (first, beta 0), from C, store to C;
 * whole blocks only (the shapes' edges go through the shipped kernel) */
static int g_stream,g_sched;
AMX static inline void blk(const bf16_call *c, size_t i, size_t j, size_t q0, size_t q1, int fromzero, int pfc, const float *nextc) {
  const mm_args *x=&c->x; const size_t kt=c->kt,ldc=x->ldc; size_t q;
  const uint16_t *a0=c->ap+((i-c->i0)/16)*kt*BF16_TILE,*a1=a0+kt*BF16_TILE;
  const uint16_t *b0=c->bp+((j-c->j0)/16)*kt*BF16_TILE,*b1=b0+kt*BF16_TILE;
  float *cp=x->c+i*ldc+j;
  if (fromzero) { _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3); }
  else { _tile_loadd(0,cp,ldc*4); _tile_loadd(1,cp+16,ldc*4); _tile_loadd(2,cp+16*ldc,ldc*4); _tile_loadd(3,cp+16*ldc+16,ldc*4); }
  for (q=q0;q<q1;q++) {
    if (pfc && nextc) {  /* the next block's C lines, written soon: 64 lines spread over the depth */
      size_t n=q1-q0,per=(64+n-1)/n,l,l0=(q-q0)*per;
      for (l=l0;l<l0+per && l<64;l++) _m_prefetchw((void *)(nextc+(l>>1)*ldc+(l&1)*16));
    }
    if (g_sched) break;
    if (g_stream==2) { _tile_stream_loadd(4,a0+q*BF16_TILE,64); _tile_stream_loadd(5,a1+q*BF16_TILE,64); }
    else { _tile_loadd(4,a0+q*BF16_TILE,64); _tile_loadd(5,a1+q*BF16_TILE,64); }
    if (g_stream==1) { _tile_stream_loadd(6,b0+q*BF16_TILE,64); _tile_stream_loadd(7,b1+q*BF16_TILE,64); }
    else { _tile_loadd(6,b0+q*BF16_TILE,64); _tile_loadd(7,b1+q*BF16_TILE,64); }
    _tile_dpbf16ps(0,4,6); _tile_dpbf16ps(1,4,7); _tile_dpbf16ps(2,5,6); _tile_dpbf16ps(3,5,7);
  }
  if (g_sched==1) for (q=q0;q<q1;q++) {  /* A0 B0 B1, two products, A1, two products */
    _tile_loadd(4,a0+q*BF16_TILE,64);
    if (g_stream==1) { _tile_stream_loadd(6,b0+q*BF16_TILE,64); _tile_stream_loadd(7,b1+q*BF16_TILE,64); }
    else { _tile_loadd(6,b0+q*BF16_TILE,64); _tile_loadd(7,b1+q*BF16_TILE,64); }
    _tile_dpbf16ps(0,4,6); _tile_dpbf16ps(1,4,7);
    _tile_loadd(5,a1+q*BF16_TILE,64);
    _tile_dpbf16ps(2,5,6); _tile_dpbf16ps(3,5,7);
  }
  if (g_sched==2 && q1>q0) {  /* software pipelined: each tile reloaded as soon as its last reader issued */
    _tile_loadd(4,a0+q0*BF16_TILE,64); _tile_loadd(6,b0+q0*BF16_TILE,64);
    _tile_loadd(7,b1+q0*BF16_TILE,64); _tile_loadd(5,a1+q0*BF16_TILE,64);
    for (q=q0;q+1<q1;q++) {
      _tile_dpbf16ps(0,4,6); _tile_dpbf16ps(1,4,7);
      _tile_loadd(4,a0+(q+1)*BF16_TILE,64);
      _tile_dpbf16ps(2,5,6);
      if (g_stream==1) _tile_stream_loadd(6,b0+(q+1)*BF16_TILE,64); else _tile_loadd(6,b0+(q+1)*BF16_TILE,64);
      _tile_dpbf16ps(3,5,7);
      if (g_stream==1) _tile_stream_loadd(7,b1+(q+1)*BF16_TILE,64); else _tile_loadd(7,b1+(q+1)*BF16_TILE,64);
      _tile_loadd(5,a1+(q+1)*BF16_TILE,64);
    }
    _tile_dpbf16ps(0,4,6); _tile_dpbf16ps(1,4,7); _tile_dpbf16ps(2,5,6); _tile_dpbf16ps(3,5,7);
  }
  if (g_sched==3 && q1>q0) {  /* pipelined, the B tiles first: B0 B1 A0 A1 reloads interleaved */
    _tile_loadd(6,b0+q0*BF16_TILE,64); _tile_loadd(4,a0+q0*BF16_TILE,64);
    _tile_loadd(7,b1+q0*BF16_TILE,64); _tile_loadd(5,a1+q0*BF16_TILE,64);
    for (q=q0;q+1<q1;q++) {
      _tile_dpbf16ps(0,4,6); _tile_dpbf16ps(2,5,6);
      if (g_stream==1) _tile_stream_loadd(6,b0+(q+1)*BF16_TILE,64); else _tile_loadd(6,b0+(q+1)*BF16_TILE,64);
      _tile_dpbf16ps(1,4,7);
      _tile_loadd(4,a0+(q+1)*BF16_TILE,64);
      _tile_dpbf16ps(3,5,7);
      if (g_stream==1) _tile_stream_loadd(7,b1+(q+1)*BF16_TILE,64); else _tile_loadd(7,b1+(q+1)*BF16_TILE,64);
      _tile_loadd(5,a1+(q+1)*BF16_TILE,64);
    }
    _tile_dpbf16ps(0,4,6); _tile_dpbf16ps(2,5,6); _tile_dpbf16ps(1,4,7); _tile_dpbf16ps(3,5,7);
  }
  _tile_stored(0,cp,ldc*4); _tile_stored(1,cp+16,ldc*4); _tile_stored(2,cp+16*ldc,ldc*4); _tile_stored(3,cp+16*ldc+16,ldc*4);
}
typedef struct { bf16_job *j; int kind; size_t rows,cols,kc; size_t tr,tc; int stream,sched; } cand;
/* kind 0: the shipped compute task. kind 1: shipped order + C prefetch. kind 2: columns outer (B block
 * kept), rows inner over a ROWS x COLS region, + C prefetch. kind 3: shipped order, depth in chunks of
 * kc tiles (A chunk kept in L1 over the columns), C spilled between chunks. kind 4: kind 2 with the
 * depth in chunks (B chunk kept over the rows). Edges (partial blocks) by the shipped kernel. */
AMX static void cand_task(void *ctx, size_t t) {
  cand *k=(cand *)ctx; const bf16_call *c=&k->j->call[0]; const mm_args *x=&c->x;
  size_t r0,r1,c0,c1,i,j,q0,q1,kt=c->kt; int zero=x->beta==0.0f;
  if (k->kind==0) { bf16_compute_task(k->j,t); return; }
  r0=(t%k->tr)*k->rows; c0=(t/k->tr)*k->cols;
  r1=r0+k->rows<x->m ? r0+k->rows : x->m; c1=c0+k->cols<x->n ? c0+k->cols : x->n;
  {
    size_t wr=r0+(r1-r0)/32*32,wc=c0+(c1-c0)/32*32;  /* whole blocks; the rest by the shipped kernel */
    cfg_tiles(); g_stream=k->stream; g_sched=k->sched;
    if (k->kind==1 || k->kind==3) {
      size_t kc=k->kind==3 ? k->kc : kt;
      for (i=r0;i<wr;i+=32) for (q0=0;q0<kt;q0+=kc) { q1=q0+kc<kt ? q0+kc : kt;
        for (j=c0;j<wc;j+=32) blk(c,i,j,q0,q1,q0==0 && zero,1,j+32<wc ? x->c+i*x->ldc+j+32 : NULL); }
    } else {
      size_t kc=k->kind==4 ? k->kc : kt;
      for (j=c0;j<wc;j+=32) for (q0=0;q0<kt;q0+=kc) { q1=q0+kc<kt ? q0+kc : kt;
        for (i=r0;i<wr;i+=32) blk(c,i,j,q0,q1,q0==0 && zero,1,i+32<wr ? x->c+(i+32)*x->ldc+j : NULL); }
    }
    _tile_release();
    if (wc<c1) bf16_amx(c,r0,wr,wc,c1,zero);
    if (wr<r1) bf16_amx(c,wr,r1,c0,c1,zero);
  }
}
int main(int argc, char **argv) {
  unsigned threads=argc>1 ? (unsigned)atoi(argv[1]) : 4; int reps=argc>2 ? atoi(argv[2]) : 21;
  static const struct { const char *name; int ta,tb; size_t m,n,k; } S[]={
    {"fwd q",0,1,1024,384,384},{"fwd k",0,1,1024,192,384},{"fwd gate",0,1,1024,1024,384},{"fwd down",0,1,1024,384,1024},
    {"fwd head",0,1,1024,260,384},{"dx q",0,0,1024,384,384},{"dx k",0,0,1024,384,192},{"dx gate",0,0,1024,384,1024},
    {"dx down",0,0,1024,1024,384},{"dW q",1,0,384,384,1024},{"dW k",1,0,192,384,1024},{"dW gate",1,0,1024,384,1024},
    {"dW down",1,0,384,1024,1024}};
  /* candidates: kind, rows, cols, kc */
  static const struct { const char *name; int kind; size_t rows,cols,kc; int stream,sched; } K[]={
    {"shipped",0,0,0,0,0,0},{"pfC",1,32,0,0,0,0},{"pfC+Bt1",1,32,0,0,1,0},{"pfC+At1",1,32,0,0,2,0},
    {"s2+pfC+Bt1",1,32,0,0,1,2},{"chunk16+Bt1",3,32,0,16,1,0},{"colsOut256",2,256,0,0,0,0}};
  enum { NK=sizeof K/sizeof K[0] };
  size_t pb=bcir_tensor_pool_size(threads); void *pm=aligned_alloc(64,(pb+63)/64*64);
  bcir_tensor_pool *pool=threads>1 ? bcir_tensor_pool_init(pm,pb,threads) : NULL;
  double sum[NK]={0}; size_t s,kk; int bad=0;
  if (!(bcir_tensor_mm_bf16_variants()&8)) { printf("no AMX instance\n"); return 2; }
  printf("threads %u reps %d\n%-9s",threads,reps,"shape");
  for (kk=0;kk<NK;kk++) printf(" %10.10s",K[kk].name);
  printf("\n");
  for (s=0;s<sizeof S/sizeof S[0];s++) {
    size_t m=S[s].m,n=S[s].n,k=S[s].k,ws=bcir_tensor_mm_bf16_workspace(m,n,k),i;
    float *a=malloc(m*k*4),*b=malloc(k*n*4),*c=malloc(m*n*4),*ref=malloc(m*n*4); void *w=aligned_alloc(64,(ws+63)/64*64);
    double *tm=malloc(sizeof(double)*reps*NK);
    bcir_tensor_mm_call xx; bf16_job j; bf16_call *cl=&j.call[0]; uint16_t *base; size_t usable=ws;
    for (i=0;i<m*k;i++) a[i]=(float)((i*2654435761u%1000)-500)*1e-3f;
    for (i=0;i<k*n;i++) b[i]=(float)((i*40503u%1000)-500)*1e-3f;
    xx.ta=S[s].ta; xx.tb=S[s].tb; xx.m=m; xx.n=n; xx.k=k; xx.alpha=1.0f; xx.a=a; xx.b=b; xx.beta=0.0f; xx.c=c;
    base=bf16_base(w,&usable); j.ncall=1; j.nchain=0; j.variant=8; bf16_args(cl,&xx);
    cl->wa=base; cl->ap=base; base+=cl->mp*cl->kt*32; cl->pa=xx.ta ? cl->kt : cl->mp/32;
    cl->wb=base; cl->bp=base; cl->pb=xx.tb ? cl->np/32 : cl->kt;
    bf16_prefix(&j); pool_run(pool,bf16_pack_task,&j,j.packs[1]);
    pool_run(pool,bf16_compute_task,&j,j.split+j.tasks[1]); memcpy(ref,c,m*n*4);
    for (int r=-2;r<reps;r++) for (size_t q=0;q<NK;q++) {
      size_t kq=(q+(size_t)(r<0?0:r))%NK; cand cd; double t0,t1;
      cd.j=&j; cd.kind=K[kq].kind; cd.stream=K[kq].stream; cd.sched=K[kq].sched; cd.rows=K[kq].rows; cd.cols=K[kq].cols ? K[kq].cols : cl->cols; cd.kc=K[kq].kc;
      cd.tr=cd.kind ? (m+cd.rows-1)/cd.rows : 0; cd.tc=cd.kind ? (n+cd.cols-1)/cd.cols : 0;
      memset(c,0x7f,m*n*4);
      pool_run(pool,bf16_pack_task,&j,j.packs[1]);  /* the panels as warm as the step leaves them */
      t0=now();
      if (cd.kind==0) pool_run(pool,bf16_compute_task,&j,j.split+j.tasks[1]);
      else pool_run(pool,cand_task,&cd,cd.tr*cd.tc);
      t1=now();
      if (memcmp(c,ref,m*n*4)) { if (!bad) printf("MISMATCH %s %s\n",S[s].name,K[kq].name); bad++; }
      if (r>=0) tm[kq*reps+r]=t1-t0;
    }
    printf("%-9s",S[s].name);
    for (kk=0;kk<NK;kk++) { qsort(tm+kk*reps,reps,sizeof *tm,cmpd); double md=tm[kk*reps+reps/2]; sum[kk]+=md; printf(" %10.4f",md*1e3); }
    printf("\n");
    free(a); free(b); free(c); free(ref); free(w); free(tm);
  }
  printf("%-9s","sum ms"); for (kk=0;kk<NK;kk++) printf(" %10.4f",sum[kk]*1e3); printf("\nmismatches %d\n",bad);
  bcir_tensor_pool_stop(pool); free(pm); return bad!=0;
}
