/* BF16-ATTR probe: the bf16 GEMM's phases at the 9.84M decoder's 13 shapes -- the A pack, the B pack
 * and the compute, each a pool job of its own, alternated with the whole entry -- and the tile unit's
 * peak on this host (TDPBF16PS on resident tiles, one thread and THREADS at once; and with the four
 * tile loads a depth tile of the kernel's inner loop from L1-resident panels).
 *   cc -O3 -ffp-contract=off -fno-math-errno -pthread -I runtime/c phases.c -lm   (includes bcir_tensor.c)
 *   ./a.out THREADS REPS */
#include "bcir_tensor.c"
#include <stdio.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int cmpd(const void *a,const void *b) { double x=*(const double *)a,y=*(const double *)b; return (x>y)-(x<y); }
typedef struct { bf16_job *j; size_t off; } sub_ctx;
static void sub_pack(void *ctx, size_t t) { sub_ctx *s=(sub_ctx *)ctx; bf16_pack_task(s->j,s->off+t); }
#ifdef BCIR_BF16_AMX
static volatile float sink;
__attribute__((target("amx-tile,amx-bf16"))) static double amx_peak(long iters) {
  bf16_tilecfg cfg; long it; double t0,t1; _Alignas(64) static __thread uint16_t z[8][512]; _Alignas(64) float out[256];
  memset(&cfg,0,sizeof cfg); cfg.palette=1;
  for (int q=0;q<8;q++) { cfg.rows[q]=16; cfg.colsb[q]=64; }
  _tile_loadconfig(&cfg);
  for (int q=0;q<8;q++) for (int e=0;e<512;e++) z[q][e]=0x3f80;  /* 1.0 */
  _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
  _tile_loadd(4,z[4],64); _tile_loadd(5,z[5],64); _tile_loadd(6,z[6],64); _tile_loadd(7,z[7],64);
  t0=now();
  for (it=0;it<iters;it++) { _tile_dpbf16ps(0,4,6); _tile_dpbf16ps(1,4,7); _tile_dpbf16ps(2,5,6); _tile_dpbf16ps(3,5,7); }
  t1=now();
  _tile_stored(0,out,64); sink=out[0]; _tile_release();
  return 4.0*16384.0*(double)iters/(t1-t0)/1e9;
}
/* the kernel's inner loop over L1-resident panels: kt depth tiles, four loads and four products each */
__attribute__((target("amx-tile,amx-bf16"))) static double amx_loads(long iters, size_t kt) {
  bf16_tilecfg cfg; long it; size_t q; double t0,t1; _Alignas(64) float out[256];
  uint16_t *p=aligned_alloc(64,4*kt*1024);
  memset(&cfg,0,sizeof cfg); cfg.palette=1;
  for (q=0;q<8;q++) { cfg.rows[q]=16; cfg.colsb[q]=64; }
  for (q=0;q<4*kt*512;q++) p[q]=0x3f80;
  _tile_loadconfig(&cfg);
  _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
  t0=now();
  for (it=0;it<iters;it++) for (q=0;q<kt;q++) {
    _tile_loadd(4,p+q*512,64); _tile_loadd(5,p+(kt+q)*512,64);
    _tile_loadd(6,p+(2*kt+q)*512,64); _tile_loadd(7,p+(3*kt+q)*512,64);
    _tile_dpbf16ps(0,4,6); _tile_dpbf16ps(1,4,7); _tile_dpbf16ps(2,5,6); _tile_dpbf16ps(3,5,7);
  }
  t1=now();
  _tile_stored(0,out,64); sink=out[0]; _tile_release(); free(p);
  return 4.0*16384.0*(double)iters*(double)kt/(t1-t0)/1e9;
}
static double peak_rate[64];
static void peak_task(void *ctx, size_t t) { (void)ctx; peak_rate[t]=amx_peak(2000000); }
#endif
int main(int argc, char **argv) {
  unsigned threads=argc>1 ? (unsigned)atoi(argv[1]) : 4; int reps=argc>2 ? atoi(argv[2]) : 31;
  static const struct { const char *name; int ta,tb; size_t m,n,k; } S[]={
    {"fwd q",0,1,1024,384,384},{"fwd k",0,1,1024,192,384},{"fwd gate",0,1,1024,1024,384},{"fwd down",0,1,1024,384,1024},
    {"fwd head",0,1,1024,260,384},{"dx q",0,0,1024,384,384},{"dx k",0,0,1024,384,192},{"dx gate",0,0,1024,384,1024},
    {"dx down",0,0,1024,1024,384},{"dW q",1,0,384,384,1024},{"dW k",1,0,192,384,1024},{"dW gate",1,0,1024,384,1024},
    {"dW down",1,0,384,1024,1024}};
  size_t pb=bcir_tensor_pool_size(threads); void *pm=aligned_alloc(64,(pb+63)/64*64);
  bcir_tensor_pool *pool=threads>1 ? bcir_tensor_pool_init(pm,pb,threads) : NULL;
  int v=bcir_tensor_mm_bf16_variants(),variant=v&8 ? 8 : v&4 ? 4 : 1;
  double sa=0,sb=0,sc=0,sw=0,sfl=0;
  printf("{\"threads\": %u, \"variant\": %d, \"reps\": %d, \"shapes\": [\n",threads,variant,reps);
  for (size_t s=0;s<sizeof S/sizeof S[0];s++) {
    size_t m=S[s].m,n=S[s].n,k=S[s].k,ws=bcir_tensor_mm_bf16_workspace(m,n,k),i;
    float *a=malloc(m*k*4),*b=malloc(k*n*4),*c=malloc(m*n*4); void *w=aligned_alloc(64,(ws+63)/64*64);
    double *ta=malloc(sizeof(double)*reps),*tb=malloc(sizeof(double)*reps),*tc=malloc(sizeof(double)*reps),*tw=malloc(sizeof(double)*reps);
    bcir_tensor_mm_call x; bf16_job j; bf16_call *cl=&j.call[0]; uint16_t *base; size_t usable=ws; sub_ctx sc_a,sc_b;
    for (i=0;i<m*k;i++) a[i]=(float)((i*2654435761u%1000)-500)*1e-3f;
    for (i=0;i<k*n;i++) b[i]=(float)((i*40503u%1000)-500)*1e-3f;
    x.ta=S[s].ta; x.tb=S[s].tb; x.m=m; x.n=n; x.k=k; x.alpha=1.0f; x.a=a; x.b=b; x.beta=0.0f; x.c=c;
    base=bf16_base(w,&usable);
    j.ncall=1; j.nchain=0; j.variant=variant; bf16_args(cl,&x);
    cl->wa=base; cl->ap=base; base+=cl->mp*cl->kt*32; cl->pa=x.ta ? cl->kt : cl->mp/32;
    cl->wb=base; cl->bp=base; cl->pb=x.tb ? cl->np/32 : cl->kt;
    bf16_prefix(&j); sc_a.j=&j; sc_a.off=0; sc_b.j=&j; sc_b.off=cl->pa;
    for (int r=-3;r<reps;r++) {
      double t0,t1,t2,t3,t4;
      t0=now(); pool_run(pool,sub_pack,&sc_a,cl->pa); t1=now();
      pool_run(pool,sub_pack,&sc_b,cl->pb); t2=now();
      pool_run(pool,bf16_compute_task,&j,j.split+j.tasks[1]); t3=now();
      bcir_tensor_mm_bf16_pool(pool,x.ta,x.tb,m,n,k,1.0f,a,b,0.0f,c,w,ws); t4=now();
      if (r>=0) { ta[r]=t1-t0; tb[r]=t2-t1; tc[r]=t3-t2; tw[r]=t4-t3; }
    }
    qsort(ta,reps,sizeof *ta,cmpd); qsort(tb,reps,sizeof *tb,cmpd); qsort(tc,reps,sizeof *tc,cmpd); qsort(tw,reps,sizeof *tw,cmpd);
    { double fl=2.0*m*n*k,A=ta[reps/2]*1e3,B=tb[reps/2]*1e3,C=tc[reps/2]*1e3,W=tw[reps/2]*1e3;
      printf("  {\"shape\": \"%s\", \"m\": %zu, \"n\": %zu, \"k\": %zu, \"pack_a_ms\": %.4f, \"pack_b_ms\": %.4f, \"compute_ms\": %.4f, "
             "\"entry_ms\": %.4f, \"compute_gflops\": %.0f, \"entry_gflops\": %.0f, \"pack_share\": %.3f, \"pack_tasks\": [%zu, %zu], \"compute_tasks\": %zu}%s\n",
             S[s].name,m,n,k,A,B,C,W,fl/C/1e6,fl/W/1e6,(A+B)/(A+B+C),cl->pa,cl->pb,j.split+j.tasks[1],s+1<sizeof S/sizeof S[0] ? "," : "");
      sa+=A; sb+=B; sc+=C; sw+=W; sfl+=fl; }
    free(a); free(b); free(c); free(w); free(ta); free(tb); free(tc); free(tw);
  }
  printf("], \"sum\": {\"pack_a_ms\": %.3f, \"pack_b_ms\": %.3f, \"compute_ms\": %.3f, \"entry_ms\": %.3f, \"compute_gflops\": %.0f}",sa,sb,sc,sw,sfl/sc/1e6);
#ifdef BCIR_BF16_AMX
  if (variant==8) {
    double one=amx_peak(4000000),l1_8=amx_loads(400000,8),l1_12=amx_loads(250000,12); unsigned t; double all=0;
    double wall=0;
    if (pool) { double t0=now(); pool_run(pool,peak_task,NULL,threads); wall=now()-t0; for (t=0;t<threads;t++) all+=peak_rate[t]; } else all=one;
    printf(",\n \"amx_peak_gflops_one_thread\": %.0f, \"amx_peak_gflops_all_threads_sum\": %.0f, \"amx_peak_gflops_all_threads_wall\": %.0f, \"amx_l1_loads_gflops_kt8\": %.0f, \"amx_l1_loads_gflops_kt12\": %.0f",one,all,pool ? threads*4.0*16384.0*2000000.0/wall/1e9 : one,l1_8,l1_12);
  }
#endif
  printf("}\n");
  bcir_tensor_pool_stop(pool); free(pm); return 0;
}
