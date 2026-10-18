/* What bounds the AdamW update at the 9.84M decoder's size on a pool of four, and what NDT-UPDATE
 * takes off it. The trainer's own unit is compiled into this file, so its static functions are the
 * ones timed: the commit as the slice leaves it (commit_avx512: the two shared divisions through
 * div_shared) against the same element body with the divisions (adamw_commit under the same
 * target), and the first pass's pair (pair_avx512) against one chunk a task with its chain after
 * the span (pass_avx512 and the chain, pass_task's loop). Bounds that change the arithmetic, timed
 * only: the commit with its two shared divisions as plain products, with no division or root, and
 * with trivial arithmetic over the same streams (the memory floor); the first pass's span without
 * the chain. Two forms measured and not kept: the commit with streaming stores (the stores bypass
 * the cache; the commit is not bound by its traffic) and the norm chains of eight chunks in eight
 * double lanes from gathers (the gathers' eight streams outrun the prefetchers). Every exact variant
 * is held to the old forms' state or norm, flags and maxima bit for bit before anything is timed.
 *
 *   cc -O3 -ffp-contract=off -fno-math-errno -pthread -I runtime/c update_probe.c runtime/c/bcir_tensor.c -lm
 *   ./a.out [ROUNDS]
 */
#include "bcir_decoder_train.c"
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>

__attribute__((target("avx512f"))) static void commit_old(const adamw_terms *t,size_t n,
    const float *gr,float *wt,float *m1,float *m2) { adamw_commit(t,n,gr,wt,m1,m2); }
/* bounds: the two shared divisions as products (2), no division or root (3), the streams alone (4) */
__attribute__((target("avx512f"))) static void commit_bound(const adamw_terms *t,size_t n,
    const float *gr,float *wt,float *m1,float *m2,int kind) {
  size_t i; double y1=1/(1-t->b1),y2=1/(1-t->b2);
  if (kind==4) {
    for (i=0;i<n;i++) { float m=0.9f*m1[i]+gr[i],v=0.95f*m2[i]+gr[i]*gr[i]; wt[i]-=1e-7f*m; m1[i]=m; m2[i]=v; }
    return;
  }
  for (i=0;i<n;i++) {
    double g=gr[i]*t->scale,mc=t->d1*m1[i]+(1-t->d1)*g,vc=t->d2*m2[i]+(1-t->d2)*g*g;
    float m=(float)mc,v=(float)vc; double a=(double)m*y1,q=(double)v*y2;
    if (kind==2) wt[i]=(float)((double)wt[i]*t->decay-t->lr*a/(sqrt(q)+t->eps));
    else wt[i]=(float)((double)wt[i]*t->decay-t->lr*a*(q*y2+t->eps));
    m1[i]=m; m2[i]=v;
  }
}
/* measured, not kept: the old commit with streaming stores, eight floats a step */
__attribute__((target("avx512f"))) static void commit_stream(const adamw_terms *t,size_t n,
    const float *gr,float *wt,float *m1,float *m2) {
  size_t i=0; __m512d sc=_mm512_set1_pd(t->scale),d1=_mm512_set1_pd(t->d1),e1=_mm512_set1_pd(1-t->d1);
  __m512d d2=_mm512_set1_pd(t->d2),e2=_mm512_set1_pd(1-t->d2),dec=_mm512_set1_pd(t->decay);
  __m512d lr=_mm512_set1_pd(t->lr),eps=_mm512_set1_pd(t->eps),c1=_mm512_set1_pd(1-t->b1),c2=_mm512_set1_pd(1-t->b2);
  if (!((uintptr_t)wt%32) && !((uintptr_t)m1%32) && !((uintptr_t)m2%32))
  for (;i+8<=n;i+=8) {
    __m256 g8=_mm256_loadu_ps(gr+i),w8=_mm256_loadu_ps(wt+i),a8=_mm256_loadu_ps(m1+i),b8=_mm256_loadu_ps(m2+i);
    __m512d g=_mm512_mul_pd(_mm512_cvtps_pd(g8),sc);
    __m512d mc=_mm512_add_pd(_mm512_mul_pd(d1,_mm512_cvtps_pd(a8)),_mm512_mul_pd(e1,g));
    __m512d vc=_mm512_add_pd(_mm512_mul_pd(d2,_mm512_cvtps_pd(b8)),_mm512_mul_pd(_mm512_mul_pd(e2,g),g));
    __m256 m=_mm512_cvtpd_ps(mc),v=_mm512_cvtpd_ps(vc);
    __m512d a=_mm512_div_pd(_mm512_cvtps_pd(m),c1),q=_mm512_div_pd(_mm512_cvtps_pd(v),c2);
    __m512d wc=_mm512_sub_pd(_mm512_mul_pd(_mm512_cvtps_pd(w8),dec),
        _mm512_div_pd(_mm512_mul_pd(lr,a),_mm512_add_pd(_mm512_sqrt_pd(q),eps)));
    _mm256_stream_ps(wt+i,_mm512_cvtpd_ps(wc)); _mm256_stream_ps(m1+i,m); _mm256_stream_ps(m2+i,v);
  }
  _mm_sfence();
  if (i<n) adamw_commit(t,n-i,gr+i,wt+i,m1+i,m2+i);
}
typedef struct cjob2 { const adamw_terms *t; size_t n; const float *gr; float *wt,*m1,*m2; int kind; } cjob2;
static void ctask2(void *ctx,size_t task) {
  cjob2 *j=(cjob2 *)ctx; size_t i0=task*BCIR_DT_CHUNK,i1=i0+BCIR_DT_CHUNK<j->n ? i0+BCIR_DT_CHUNK : j->n,n=i1-i0;
  if (j->kind==0) commit_old(j->t,n,j->gr+i0,j->wt+i0,j->m1+i0,j->m2+i0);
  else if (j->kind==1) commit_avx512(j->t,n,j->gr+i0,j->wt+i0,j->m1+i0,j->m2+i0);
  else if (j->kind==5) commit_stream(j->t,n,j->gr+i0,j->wt+i0,j->m1+i0,j->m2+i0);
  else commit_bound(j->t,n,j->gr+i0,j->wt+i0,j->m1+i0,j->m2+i0,j->kind);
}
static const char *cname[]={"commit, the divisions (before)","commit, div_shared (the slice)",
  "bound: shared divisions as products","bound: no division, no root","bound: the streams alone",
  "not kept: streaming stores"};
/* measured, not kept: eight chunks' chains in eight lanes from gathers, then the other streams */
__attribute__((target("avx512f"))) static void lanes8(size_t chunk,const float *gr,const float *wt,
    const float *m1,const float *m2,double gs,pass_part *parts) {
  __m256i at=_mm256_setr_epi32(0,(int)chunk,(int)(2*chunk),(int)(3*chunk),(int)(4*chunk),(int)(5*chunk),
      (int)(6*chunk),(int)(7*chunk));
  __m512d acc=_mm512_setzero_pd(),scale=_mm512_set1_pd(gs); double norms[8]; size_t i; int k;
  for (i=0;i<chunk;i++) {
    __m512d g=_mm512_mul_pd(_mm512_cvtps_pd(_mm256_i32gather_ps(gr+i,at,4)),scale);
    acc=_mm512_add_pd(acc,_mm512_mul_pd(g,g));
  }
  _mm512_storeu_pd(norms,acc);
  pass_avx512(8*chunk,gr,wt,m1,m2,gs,&parts[0]);
  for (k=0;k<8;k++) { if (k) memset(&parts[k],0,sizeof(parts[k])); parts[k].norm=norms[k]; }
}
typedef struct pjob2 { const float *gr,*wt,*m1,*m2; size_t n,chunk; double gs; pass_part *parts; int kind; } pjob2;
static void ptask2(void *ctx,size_t task) {
  pjob2 *j=(pjob2 *)ctx; size_t chunks=(j->n+j->chunk-1)/j->chunk,c,i;
  size_t per=j->kind==1 ? 2 : j->kind==3 ? 8 : 1,c0=task*per,c1=c0+per<chunks ? c0+per : chunks;
  if (j->kind==1 && c1-c0==2 && c1*j->chunk<=j->n) {
    double norms[2]; pass_part p; size_t i0=c0*j->chunk;
    pair_avx512(j->chunk,j->gr+i0,j->wt+i0,j->m1+i0,j->m2+i0,j->gs,&p,norms);
    p.norm=norms[0]; j->parts[c0]=p; p.bad=0; p.grmax=p.wmax=p.mmax=p.vmax=0.0f; p.norm=norms[1]; j->parts[c0+1]=p;
    return;
  }
  if (j->kind==3 && c1-c0==8 && c1*j->chunk<=j->n) { lanes8(j->chunk,j->gr+c0*j->chunk,j->wt+c0*j->chunk,
      j->m1+c0*j->chunk,j->m2+c0*j->chunk,j->gs,j->parts+c0); return; }
  for (c=c0;c<c1;c++) {
    pass_part p; size_t i0=c*j->chunk,i1=i0+j->chunk<j->n ? i0+j->chunk : j->n;
    pass_avx512(i1-i0,j->gr+i0,j->wt+i0,j->m1+i0,j->m2+i0,j->gs,&p); p.norm=0.0;
    if (j->kind!=2) for (i=i0;i<i1;i++) { double g=(double)j->gr[i]*j->gs; p.norm+=g*g; }
    j->parts[c]=p;
  }
}
static const char *pname[]={"first pass, one chunk a task (before)","first pass, pairs (the slice)",
  "bound: the span without the chain","not kept: eight chains from gathers"};
static float *big(size_t n) {
  size_t bytes=(n*sizeof(float)+(2u<<20)-1)/(2u<<20)*(2u<<20); void *p=NULL;
  if (posix_memalign(&p,2u<<20,bytes)) { perror("posix_memalign"); exit(2); }
  madvise(p,bytes,MADV_HUGEPAGE); memset(p,0,bytes); return (float *)p;
}
static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmp_d(const void *x,const void *y) { double a=*(const double *)x,b=*(const double *)y; return (a>b)-(a<b); }
static uint64_t fnv(const void *p,size_t n) { const unsigned char *b=p; uint64_t h=1469598103934665603ull; size_t i;
  for (i=0;i<n;i++) { h^=b[i]; h*=1099511628211ull; } return h; }
static uint64_t rng=0x243f6a8885a308d3ull;
static double urand(void) { rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (double)(rng>>11)/9007199254740992.0; }
static double nrand(void) { double u=urand()+1e-300,v=urand(); return sqrt(-2*log(u))*cos(6.283185307179586*v); }
static void fold(const pass_part *parts,size_t chunks,double *norm,int *bad,float *mx) {
  size_t i; *norm=0.0; *bad=0; mx[0]=mx[1]=mx[2]=mx[3]=0.0f;
  for (i=0;i<chunks;i++) { *norm+=parts[i].norm; *bad|=parts[i].bad;
    mx[0]=parts[i].grmax>mx[0] ? parts[i].grmax : mx[0]; mx[1]=parts[i].wmax>mx[1] ? parts[i].wmax : mx[1];
    mx[2]=parts[i].mmax>mx[2] ? parts[i].mmax : mx[2]; mx[3]=parts[i].vmax>mx[3] ? parts[i].vmax : mx[3]; }
}
int main(int argc,char **argv) {
  /* the 9.84M decoder's parameter count, the reduce contract's chunk */
  size_t n=9836928,chunk=bcir_decoder_reduce_chunk(9836928),chunks=(n+chunk-1)/chunk,i,k,r;
  int rounds=argc>1 ? atoi(argv[1]) : 15;
  float *g=big(n),*w=big(n),*m1=big(n),*m2=big(n),*w0=big(n),*a0=big(n),*b0=big(n);
  static _Alignas(64) unsigned char pm[1<<16]; bcir_tensor_pool *pool; adamw_terms t;
  static pass_part parts[BCIR_DT_PASS_TASKS]; static double tc[6][64],tp[4][64];
  if (rounds<1 || rounds>64) return 2;
  for (i=0;i<n;i++) { g[i]=(float)(1e-3*nrand()); w0[i]=(float)(0.02*nrand()); a0[i]=(float)(1e-3*nrand());
    b0[i]=(float)(1e-6*urand()); }
  for (i=0;i<n;i+=97) a0[i]=0.0f;
  for (i=5;i<n;i+=1009) { b0[i]=0.0f; a0[i]=-0.0f; }
  t.scale=1.0; t.d1=0.9; t.d2=0.95; t.lr=3e-4; t.eps=1e-8; t.decay=1-3e-4*0.1; t.b1=pow(0.9,100); t.b2=pow(0.95,100);
  pool=bcir_tensor_pool_init(pm,sizeof(pm),4);
  if (!pool) { fprintf(stderr,"pool\n"); return 2; }
  /* exactness: the slice's commit against the divisions over three steps from one state */
  {
    uint64_t h[2]; int kind,s;
    for (kind=0;kind<2;kind++) {
      cjob2 j; memcpy(w,w0,n*4); memcpy(m1,a0,n*4); memcpy(m2,b0,n*4);
      for (s=0;s<3;s++) {
        adamw_terms u=t; u.b1=pow(0.9,1+s*37); u.b2=pow(0.95,1+s*37);
        j.t=&u; j.n=n; j.gr=g; j.wt=w; j.m1=m1; j.m2=m2; j.kind=kind;
        bcir_tensor_pool_run(pool,ctask2,&j,(n+BCIR_DT_CHUNK-1)/BCIR_DT_CHUNK);
      }
      h[kind]=fnv(w,n*4)^(fnv(m1,n*4)*3)^(fnv(m2,n*4)*7);
    }
    printf("commit state after three steps: divisions %016llx, div_shared %016llx -- %s\n",
        (unsigned long long)h[0],(unsigned long long)h[1],h[0]==h[1] ? "the same" : "DIFFERENT");
    if (h[0]!=h[1]) return 1;
  }
  /* exactness: the pairs' folded norm, flag and maxima against one chunk a task */
  {
    double norm[2]; int bad[2]; float mx[2][4]; int kk;
    memcpy(w,w0,n*4); memcpy(m1,a0,n*4); memcpy(m2,b0,n*4);
    for (kk=0;kk<2;kk++) {
      pjob2 j; j.gr=g; j.wt=w; j.m1=m1; j.m2=m2; j.n=n; j.chunk=chunk; j.gs=1.0; j.parts=parts; j.kind=kk;
      bcir_tensor_pool_run(pool,ptask2,&j,kk ? (chunks+1)/2 : chunks);
      fold(parts,chunks,&norm[kk],&bad[kk],mx[kk]);
    }
    printf("first pass over %zu chunks of %zu: norm %a and %a, flags %d %d, maxima %s -- %s\n",chunks,chunk,
        norm[0],norm[1],bad[0],bad[1],memcmp(mx[0],mx[1],sizeof(mx[0])) ? "different" : "the same",
        !memcmp(&norm[0],&norm[1],8) && bad[0]==bad[1] && !memcmp(mx[0],mx[1],sizeof(mx[0])) ? "the same" : "DIFFERENT");
    if (memcmp(&norm[0],&norm[1],8)) return 1;
  }
  memcpy(w,w0,n*4); memcpy(m1,a0,n*4); memcpy(m2,b0,n*4);
  for (r=0;r<(size_t)rounds;r++) {
    for (k=0;k<10;k++) {
      size_t v=(k+r)%10; double t0,t1;
      if (v<6) { cjob2 j; j.t=&t; j.n=n; j.gr=g; j.wt=w; j.m1=m1; j.m2=m2; j.kind=(int)v;
        t0=now_ns(); bcir_tensor_pool_run(pool,ctask2,&j,(n+BCIR_DT_CHUNK-1)/BCIR_DT_CHUNK); t1=now_ns(); tc[v][r]=(t1-t0)/1e6; }
      else { pjob2 j; int kind=(int)(v-6); size_t per=kind==1 ? 2 : kind==3 ? 8 : 1;
        j.gr=g; j.wt=w; j.m1=m1; j.m2=m2; j.n=n; j.chunk=chunk; j.gs=1.0; j.parts=parts; j.kind=kind;
        t0=now_ns(); bcir_tensor_pool_run(pool,ptask2,&j,(chunks+per-1)/per); t1=now_ns(); tp[kind][r]=(t1-t0)/1e6; }
    }
    if (r%4==3) { memcpy(w,w0,n*4); memcpy(m1,a0,n*4); memcpy(m2,b0,n*4); }
  }
  printf("medians of %d rounds on a pool of four, ms a call (min, max):\n",rounds);
  for (k=0;k<6;k++) { double s[64]; memcpy(s,tc[k],rounds*sizeof(double)); qsort(s,rounds,sizeof(double),cmp_d);
    printf("  %-40s %6.2f  (%5.2f, %5.2f)\n",cname[k],s[rounds/2],s[0],s[rounds-1]); }
  for (k=0;k<4;k++) { double s[64]; memcpy(s,tp[k],rounds*sizeof(double)); qsort(s,rounds,sizeof(double),cmp_d);
    printf("  %-40s %6.2f  (%5.2f, %5.2f)\n",pname[k],s[rounds/2],s[0],s[rounds-1]); }
  bcir_tensor_pool_stop(pool);
  return 0;
}
