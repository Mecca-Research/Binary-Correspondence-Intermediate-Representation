/* RMS-LANES-0: the tensor unit's RMSNorm jobs one at a time -- the forward with its scan (rms_task),
 * the VJP's rows job (rmsb_rows_task) and its gamma-adjoint columns job (rmsb_cols_task) -- against
 * variants that keep every operation in the unit's order: F1, the forward's scale and its scan in
 * AVX-512 (each element (x*inv)*g, the scan's verdict folded into the same pass); R1, the rows job's
 * dx in AVX-512 (((dy*g)*inv) - (x*coeff), plus dx under ADD); C1/C2/C4, the columns job in AVX-512
 * with 16, 32 or 64 columns a task (each column's chain the rows in order from dg). Every output is
 * checked byte for byte first. usage: rms_bench THREADS ROWS WIDTH CALLS ROUNDS */
#define _POSIX_C_SOURCE 200809L
#include "bcir_tensor.c"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static int cmpd(const void *a,const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y ? -1 : x>y; }

/* F1: rms_task with the scale and the scan in AVX-512 */
__attribute__((target("avx512f"))) static void f1_rows16(rms_job *j, size_t r0, unsigned char *bad) {
  float sum[16]; size_t r,c,w=j->width; __mmask16 m=0; const __m512 lim=_mm512_set1_ps(FLT_MAX);
  rms16_stats(w,j->x+r0*w,NULL,NULL,sum,NULL);
  for (r=0;r<16;r++) {
    const float *xr=j->x+(r0+r)*w; float *yr=j->y+(r0+r)*w,inv=1.0f/sqrtf(sum[r]/(float)w+j->eps);
    __m512 iv=_mm512_set1_ps(inv);
    for (c=0;c+16<=w;c+=16) {
      __m512 y=_mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(xr+c),iv),_mm512_loadu_ps(j->g+c));
      _mm512_storeu_ps(yr+c,y);
      m|=_mm512_cmp_ps_mask(_mm512_abs_ps(y),lim,_CMP_NLE_UQ);
    }
    if (c<w) {
      __mmask16 k=(__mmask16)((1u<<(w-c))-1u);
      __m512 y=_mm512_mul_ps(_mm512_mul_ps(_mm512_maskz_loadu_ps(k,xr+c),iv),_mm512_maskz_loadu_ps(k,j->g+c));
      _mm512_mask_storeu_ps(yr+c,k,y);
      m|=_mm512_mask_cmp_ps_mask(k,_mm512_abs_ps(y),lim,_CMP_NLE_UQ);
    }
  }
  if (bad) *bad=(unsigned char)(m!=0);
}
static void f1_task(void *ctx, size_t task) {
  rms_job *j=(rms_job *)ctx; size_t r0=task*BCIR_ROW_CHUNK,r1=r0+BCIR_ROW_CHUNK<j->rows ? r0+BCIR_ROW_CHUNK : j->rows;
  if (j->variant==4 && r1-r0==16) { f1_rows16(j,r0,j->bad ? j->bad+task : NULL); return; }
  bcir_tensor_rms(r1-r0,j->width,j->eps,j->x+r0*j->width,j->g,j->y+r0*j->width);
  if (j->bad) j->bad[task]=(unsigned char)!span_finite_run(j->variant,j->y+r0*j->width,(r1-r0)*j->width);
}
static void rms_v(bcir_tensor_pool *pool, int v, size_t rows, size_t width, float eps,
    const float *x, const float *g, float *y, int *bad) {
  rms_job j; unsigned char bytes[BCIR_TENSOR_SCAN_TASKS]; size_t tasks=(rows+BCIR_ROW_CHUNK-1)/BCIR_ROW_CHUNK;
  j.rows=rows; j.width=width; j.eps=eps; j.x=x; j.g=g; j.y=y; j.variant=mm_widest();
  j.bad=scan_bytes(bad,bytes,tasks);
  pool_run(pool,v ? f1_task : rms_task,&j,tasks);
  if (!bad) return;
  if (j.bad) scan_fold(bad,bytes,tasks);
  else if (!scan_pool(pool,y,rows*width)) *bad=1;
}
/* R1: the rows job's dx in AVX-512 */
__attribute__((target("avx512f"))) static void r1_rows16(rmsb_job *j, size_t r0) {
  float sum[16],dot[16]; size_t r,c,w=j->width;
  rms16_stats(w,j->x+r0*w,j->dy+r0*w,j->g,sum,dot);
  for (r=0;r<16;r++) {
    const float *xr=j->x+(r0+r)*w,*dyr=j->dy+(r0+r)*w; float *dxr=j->dx+(r0+r)*w;
    float inv=1.0f/sqrtf(sum[r]/(float)w+j->eps),coeff=dot[r]*inv*inv*inv/(float)w;
    __m512 iv=_mm512_set1_ps(inv),cv=_mm512_set1_ps(coeff);
    for (c=0;c+16<=w;c+=16) {
      __m512 t=_mm512_sub_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(dyr+c),_mm512_loadu_ps(j->g+c)),iv),
                             _mm512_mul_ps(_mm512_loadu_ps(xr+c),cv));
      if (j->how&BCIR_TENSOR_RMSB_ADD) t=_mm512_add_ps(_mm512_loadu_ps(dxr+c),t);
      _mm512_storeu_ps(dxr+c,t);
    }
    if (c<w) {
      __mmask16 k=(__mmask16)((1u<<(w-c))-1u);
      __m512 t=_mm512_sub_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_maskz_loadu_ps(k,dyr+c),_mm512_maskz_loadu_ps(k,j->g+c)),iv),
                             _mm512_mul_ps(_mm512_maskz_loadu_ps(k,xr+c),cv));
      if (j->how&BCIR_TENSOR_RMSB_ADD) t=_mm512_add_ps(_mm512_maskz_loadu_ps(k,dxr+c),t);
      _mm512_mask_storeu_ps(dxr+c,k,t);
    }
    j->inv[r0+r]=inv;
  }
}
static void r1_task(void *ctx, size_t task) {
  rmsb_job *j=(rmsb_job *)ctx; size_t r0=task*BCIR_ROW_CHUNK,r1=r0+BCIR_ROW_CHUNK<j->rows ? r0+BCIR_ROW_CHUNK : j->rows;
  if (j->variant==4 && r1-r0==16) { r1_rows16(j,r0); return; }
  rmsb_rows_task(ctx,task);
}
/* C1/C2/C4: the columns job in AVX-512, CH columns a task */
static size_t col_chunk=16;
__attribute__((target("avx512f"))) static void cv_task(void *ctx, size_t task) {
  rmsb_job *j=(rmsb_job *)ctx; size_t r,c,w=j->width,ch=col_chunk;
  size_t c0=task*ch,c1=c0+ch<w ? c0+ch : w;
  for (c=c0;c<c1;c+=16) {
    __mmask16 k=(__mmask16)(c1-c>=16 ? 0xffffu : (1u<<(c1-c))-1u);
    __m512 acc=(j->how&BCIR_TENSOR_RMSB_FIRST) ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(k,j->dg+c);
    for (r=0;r<j->rows;r++) {
      __m512 t=_mm512_mul_ps(_mm512_mul_ps(_mm512_maskz_loadu_ps(k,j->dy+r*w+c),_mm512_maskz_loadu_ps(k,j->x+r*w+c)),_mm512_set1_ps(j->inv[r]));
      acc=_mm512_add_ps(acc,t);
    }
    _mm512_mask_storeu_ps(j->dg+c,k,acc);
  }
}
static void rmsb_v(bcir_tensor_pool *pool, int rv, int cvar, size_t rows, size_t width, float eps,
    const float *x, const float *g, const float *dy, float *dx, float *dg, float *inv, unsigned how, double *tr, double *tc) {
  rmsb_job j; double t0;
  j.rows=rows; j.width=width; j.eps=eps; j.x=x; j.g=g; j.dy=dy; j.dx=dx; j.dg=dg; j.inv=inv;
  j.variant=mm_widest(); j.how=how;
  t0=now(); pool_run(pool,rv ? r1_task : rmsb_rows_task,&j,(rows+BCIR_ROW_CHUNK-1)/BCIR_ROW_CHUNK); if (tr) *tr+=now()-t0;
  t0=now();
  if (!cvar) pool_run(pool,rmsb_cols_task,&j,(width+BCIR_COL_CHUNK-1)/BCIR_COL_CHUNK);
  else { col_chunk=(size_t)cvar; pool_run(pool,cv_task,&j,(width+col_chunk-1)/col_chunk); }
  if (tc) *tc+=now()-t0;
}
static float frand(unsigned *s) { *s=*s*1103515245u+12345u; return (float)((*s>>8)&0xffff)/32768.0f-1.0f; }
static void fill(float *a, size_t n, unsigned seed, int kind) {
  size_t i; unsigned s=seed;
  for (i=0;i<n;i++) {
    float v=frand(&s);
    if (kind==1) v*=1e-20f; else if (kind==2) v*=1e18f; else if (kind==3 && i%977==5) v=(i%2) ? INFINITY : NAN;
    a[i]=v;
  }
}
int main(int argc,char **argv) {
  if (argc!=6) return 2;
  unsigned th=(unsigned)atoi(argv[1]); size_t rows=strtoul(argv[2],0,10),w=strtoul(argv[3],0,10),n=rows*w;
  int calls=atoi(argv[4]),rounds=atoi(argv[5]),r,c,k,kind;
  float *x=aligned_alloc(64,n*4),*g=aligned_alloc(64,w*4),*dy=aligned_alloc(64,n*4),*y0=aligned_alloc(64,n*4),*y1=aligned_alloc(64,n*4);
  float *dx0=aligned_alloc(64,n*4),*dx1=aligned_alloc(64,n*4),*dg0=aligned_alloc(64,w*4+64),*dg1=aligned_alloc(64,w*4+64);
  float *iv0=aligned_alloc(64,rows*4+64),*iv1=aligned_alloc(64,rows*4+64);
  size_t ps=bcir_tensor_pool_size(th); void *pm=NULL; if (posix_memalign(&pm,64,(ps+63)/64*64)) return 3;
  bcir_tensor_pool *pool=bcir_tensor_pool_init(pm,ps,th);
  int cvars[4]={16,32,64,0}; unsigned hows[3]={0u,BCIR_TENSOR_RMSB_ADD|BCIR_TENSOR_RMSB_FIRST,BCIR_TENSOR_RMSB_ADD};
  /* values: four kinds of input (uniform, tiny, large, non-finite) under every how */
  for (kind=0;kind<4;kind++) {
    int b0=0,b1=0; size_t hh;
    fill(x,n,11u+kind,kind); fill(g,w,23u,0); fill(dy,n,37u+kind,kind==3 ? 0 : kind);
    rms_v(pool,0,rows,w,1e-5f,x,g,y0,&b0); rms_v(pool,1,rows,w,1e-5f,x,g,y1,&b1);
    if (memcmp(y0,y1,n*4) || b0!=b1) { printf("MISMATCH forward kind=%d bad %d %d\n",kind,b0,b1); return 1; }
    for (hh=0;hh<3;hh++) for (k=0;k<4;k++) {
      fill(dx0,n,51u,0); memcpy(dx1,dx0,n*4); fill(dg0,w,61u,0); memcpy(dg1,dg0,w*4);
      rmsb_v(pool,0,0,rows,w,1e-5f,x,g,dy,dx0,dg0,iv0,hows[hh],NULL,NULL);
      rmsb_v(pool,1,cvars[k],rows,w,1e-5f,x,g,dy,dx1,dg1,iv1,hows[hh],NULL,NULL);
      if (memcmp(dx0,dx1,n*4) || memcmp(dg0,dg1,w*4) || memcmp(iv0,iv1,rows*4)) {
        size_t q,nd=0,nn=0,ng=0,ni=0;
        for (q=0;q<n;q++) if (memcmp(dx0+q,dx1+q,4)) { if (isnan(dx0[q]) && isnan(dx1[q])) nn++; else nd++; }
        for (q=0;q<w;q++) if (memcmp(dg0+q,dg1+q,4)) { if (isnan(dg0[q]) && isnan(dg1[q])) nn++; else ng++; }
        for (q=0;q<rows;q++) if (memcmp(iv0+q,iv1+q,4)) { if (isnan(iv0[q]) && isnan(iv1[q])) nn++; else ni++; }
        if (nd || ng || ni || kind!=3) { printf("MISMATCH backward kind=%d how=%u cols=%d dx %zu dg %zu inv %zu nan-only %zu\n",kind,hows[hh],cvars[k],nd,ng,ni,nn); return 1; }
        if (!hh && !k) printf("kind 3: %zu elements differ only in which NaN they carry\n",nn);
      }
    }
  }
  fill(x,n,11u,0); fill(g,w,23u,0); fill(dy,n,37u,0);
  { double tf[2][64],tr[2][64],tcol[4][64],trr,tcc; int bad=0,v;
    for (r=-1;r<rounds;r++) {
      for (v=0;v<2;v++) { double t0=now(); for (c=0;c<calls;c++) rms_v(pool,v,rows,w,1e-5f,x,g,y0,&bad); if (r>=0) tf[v][r]=(now()-t0)*1e6/calls; }
      for (v=0;v<2;v++) { trr=0; for (c=0;c<calls;c++) rmsb_v(pool,v,0,rows,w,1e-5f,x,g,dy,dx0,dg0,iv0,hows[1],&trr,NULL); if (r>=0) tr[v][r]=trr*1e6/calls; }
      for (k=0;k<4;k++) { tcc=0; int cv=(k==0) ? 0 : cvars[k-1]; for (c=0;c<calls;c++) rmsb_v(pool,0,cv,rows,w,1e-5f,x,g,dy,dx0,dg0,iv0,hows[1],NULL,&tcc); if (r>=0) tcol[k][r]=tcc*1e6/calls; }
    }
    for (v=0;v<2;v++) { qsort(tf[v],(size_t)rounds,sizeof(double),cmpd); qsort(tr[v],(size_t)rounds,sizeof(double),cmpd); }
    for (k=0;k<4;k++) qsort(tcol[k],(size_t)rounds,sizeof(double),cmpd);
    printf("{\"threads\": %u, \"rows\": %zu, \"width\": %zu, \"bytes_equal\": true, \"fwd_us\": [%.2f, %.2f], \"fwd_ratio\": %.3f, "
      "\"rows_us\": [%.2f, %.2f], \"rows_ratio\": %.3f, \"cols_us\": {\"unit16\": %.2f, \"v16\": %.2f, \"v32\": %.2f, \"v64\": %.2f}}\n",
      th,rows,w,tf[0][rounds/2],tf[1][rounds/2],tf[1][rounds/2]/tf[0][rounds/2],tr[0][rounds/2],tr[1][rounds/2],tr[1][rounds/2]/tr[0][rounds/2],
      tcol[0][rounds/2],tcol[1][rounds/2],tcol[2][rounds/2],tcol[3][rounds/2]);
  }
  bcir_tensor_pool_stop(pool);
  return 0;
}
