/* band_place: the transposed product (dV = P^T dY) and the context product (dQ = dS K) over a whole head's P with
 * P in the unit's row-major layout (the unit's att512_transposed / att512_context) or in 16-row bands, rows in
 * lanes (pb[band*16*t + j*16 + u] = P[16*band+u][j]: a band's column one 64-byte line), the same register blocks
 * and the same term order -- so the same bits -- reading P through the band layout; P resident in L3 (NBUF 1) or
 * streamed from DRAM (NBUF buffers in turn); TSC cycles a probability, the median of the calls.
 * usage: band_place NBUF CALLS WHICH(tr|ctx|btr|bctx) */
#include UNIT
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#define PB(pb,t,i,j) ((pb)[((i)>>4)*16*(t)+(j)*16+((i)&15)])
/* att512_tr_block's arithmetic exactly, P(i, j0+r) read from the band layout */
__attribute__((target("avx512f"))) static inline __attribute__((always_inline)) void btr_block(size_t j0, size_t ib,
    size_t ie, size_t t, const float *pb, const float *x, size_t ldx, float *out, size_t ldo, int zero) {
  __m512 a0[8],a1[8]; size_t r,i,di; int diag=j0>=ib;
  for (r=0;r<8;r++) {
    const float *o=out+(j0+r)*ldo;
    a0[r]=zero ? _mm512_setzero_ps() : _mm512_loadu_ps(o); a1[r]=zero ? _mm512_setzero_ps() : _mm512_loadu_ps(o+16);
  }
  for (di=0;diag && di<7;di++) {
    size_t ii=j0+di; const float *xi=x+ii*ldx,*col=pb+(ii>>4)*16*t+(ii&15);
    __m512 b0=_mm512_loadu_ps(xi),b1=_mm512_loadu_ps(xi+16);
    for (r=0;r<8;r++) {
      __mmask16 take=(__mmask16)(r<=di ? 0xffffu : 0u); __m512 mv=_mm512_set1_ps(col[(j0+r)*16]);
      a0[r]=_mm512_mask_add_ps(a0[r],take,a0[r],_mm512_mul_ps(mv,b0)); a1[r]=_mm512_mask_add_ps(a1[r],take,a1[r],_mm512_mul_ps(mv,b1));
    }
  }
  for (i=diag ? j0+7 : ib;i<ie;i++) {
    const float *xi=x+i*ldx,*col=pb+(i>>4)*16*t+(i&15)+j0*16;
    __m512 b0=_mm512_loadu_ps(xi),b1=_mm512_loadu_ps(xi+16);
    for (r=0;r<8;r++) {
      __m512 mv=_mm512_set1_ps(col[r*16]);
      a0[r]=_mm512_add_ps(a0[r],_mm512_mul_ps(mv,b0)); a1[r]=_mm512_add_ps(a1[r],_mm512_mul_ps(mv,b1));
    }
  }
  for (r=0;r<8;r++) { float *o=out+(j0+r)*ldo; _mm512_storeu_ps(o,a0[r]); _mm512_storeu_ps(o+16,a1[r]); }
}
__attribute__((target("avx512f"),noinline)) static void btr(size_t t, const float *pb, const float *x, float *out, int first, size_t band) {
  size_t ib,j0;
  for (ib=0;ib<t;ib+=band) {
    size_t ie=t-ib<band ? t : ib+band;
    for (j0=0;j0<ie;j0+=8) btr_block(j0,ib,ie,t,pb,x,32,out,32,first && j0>=ib);
  }
}
/* att512_ctx_block's arithmetic exactly (nv = 2, every lane), P(i0+r, j) read from the band layout */
__attribute__((target("avx512f"),noinline)) static void bctx(size_t t, const float *pb, const float *v, float *y) {
  size_t i0,r,j,dj;
  for (i0=0;i0<t;i0+=8) {
    __m512 a0[8],a1[8]; const float *row=pb+(i0>>4)*16*t+(i0&15);
    for (r=0;r<8;r++) { a0[r]=_mm512_setzero_ps(); a1[r]=_mm512_setzero_ps(); }
    for (j=0;j<=i0;j++) {
      const float *vj=v+j*32; __m512 b0=_mm512_loadu_ps(vj),b1=_mm512_loadu_ps(vj+16);
      for (r=0;r<8;r++) {
        __m512 pv=_mm512_set1_ps(row[j*16+r]);
        a0[r]=_mm512_add_ps(a0[r],_mm512_mul_ps(pv,b0)); a1[r]=_mm512_add_ps(a1[r],_mm512_mul_ps(pv,b1));
      }
    }
    for (dj=1;dj<8;dj++) {
      const float *vj=v+(i0+dj)*32; __m512 b0=_mm512_loadu_ps(vj),b1=_mm512_loadu_ps(vj+16);
      for (r=0;r<8;r++) {
        __mmask16 take=(__mmask16)(r>=dj ? 0xffffu : 0u); __m512 pv=_mm512_set1_ps(row[(i0+dj)*16+r]);
        a0[r]=_mm512_mask_add_ps(a0[r],take,a0[r],_mm512_mul_ps(pv,b0)); a1[r]=_mm512_mask_add_ps(a1[r],take,a1[r],_mm512_mul_ps(pv,b1));
      }
    }
    for (r=0;r<8;r++) { _mm512_storeu_ps(y+(i0+r)*32,a0[r]); _mm512_storeu_ps(y+(i0+r)*32+16,a1[r]); }
  }
}
static int cmpd(const void *a, const void *b) { double x=*(const double *)a,y=*(const double *)b; return x<y ? -1 : x>y; }
int main(int argc, char **argv) {
  size_t t=2048,d=32,nbuf=strtoul(argv[1],0,10),calls=strtoul(argv[2],0,10),i,j,c,k; const char *w=argv[3];
  int banded=w[0]=='b',ctx=strstr(w,"ctx")!=NULL; size_t band=argc>4 ? strtoul(argv[4],0,10) : 32;
  float **p=malloc(nbuf*sizeof *p),*x=aligned_alloc(64,t*d*4),*out=aligned_alloc(64,t*d*4),*ref=aligned_alloc(64,t*d*4);
  double *cyc=malloc(calls*sizeof *cyc),probs=(double)t*(t+1)/2;
  for (k=0;k<nbuf;k++) {
    float *rm=aligned_alloc(64,t*t*4);
    for (i=0;i<t;i++) for (j=0;j<t;j++) rm[i*t+j]=j<=i ? (float)((i*7+j*3+k)%17)*0.013f-0.07f : 0.0f;
    if (banded) { p[k]=aligned_alloc(64,t*t*4); for (i=0;i<t;i++) for (j=0;j<t;j++) PB(p[k],t,i,j)=rm[i*t+j]; }
    else p[k]=rm;
    if (k==0) {  /* the unit's result on buffer 0, for the bits */
      for (i=0;i<t*d;i++) { x[i]=(float)(i%13)*0.1f-0.5f; ref[i]=0.0f; }
      if (ctx) att512_context(t,d,rm,t,x,d,ref,d,1); else att512_transposed(t,t,d,rm,t,x,d,ref,d,1);
      if (banded) free(rm);
    }
  }
  int rewrite=argc>5 ? atoi(argv[5]) : 0;
  for (c=0;c<calls+2;c++) {
    float *pk=p[c%nbuf];
    if (rewrite==1) { volatile float z=0.0f; for (i=0;i<t*t;i+=16) pk[i]=pk[i]+z; }  /* every line written again */
    if (rewrite==2) { size_t q; for (q=0;q<t;q+=16) { volatile float z=0.0f; size_t lim=(q+16)*16; for (i=0;i<lim;i+=16) pk[q*t+i]=pk[q*t+i]+z; } }  /* the bands' lower parts */
    unsigned long long t0=__rdtsc();
    if (!banded) { if (ctx) att512_context(t,d,pk,t,x,d,out,d,1); else att512_transposed(t,t,d,pk,t,x,d,out,d,1); }
#ifdef PROTO_BTR
    else if (w[1]=='p') att512_btransposed(t,d,pk,x,d,out,d,1);
#endif
    else if (ctx) bctx(t,pk,x,out); else btr(t,pk,x,out,1,band);
    unsigned long long t1=__rdtsc();
    if (c>=2) cyc[c-2]=(double)(t1-t0)/probs;
    if (c==0) for (i=0;i<t*d;i++) if (memcmp(&out[i],&ref[i],4)) { printf("%s: bits differ at %zu\n",w,i); return 1; }
  }
  qsort(cyc,calls,sizeof *cyc,cmpd);
  printf("%-4s band %zu nbuf=%zu: median %.3f min %.3f TSC cycles a probability (v1's rate %.0f%%), buffer 0 bits the unit's\n",
         w,band,nbuf,cyc[calls/2],cyc[0],100*1.443/cyc[calls/2]);
  return 0;
}
