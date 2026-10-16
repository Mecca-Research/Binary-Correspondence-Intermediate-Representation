/* The step's weight-gradient GEMMs (dW = dY^T X: A transposed, ta=1) against the same products with A
 * transposed beforehand through ta=0 (NDT-GEMM-4T's NDT-TA probe): both results bit for bit the
 * same (the kernel's arithmetic does not depend on A's layout), so the times say what A's layout
 * costs. One JSON line per shape: medians of REPS calls each, the explicit transpose timed apart.
 *   cc -std=c11 -O3 -ffp-contract=off -fno-math-errno -pthread -I runtime/c transposed_a.c \
 *       runtime/c/bcir_tensor.c -o transposed_a -lm && ./transposed_a THREADS REPS */
#define _POSIX_C_SOURCE 199309L
#include "bcir_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec*1e-6; }
static int cmpd(const void *a,const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y; }
int main(int argc,char **argv) {
  unsigned threads=argc>1?(unsigned)atoi(argv[1]):1; int reps=argc>2?atoi(argv[2]):30;
  size_t shapes[5][2]={{1024,384},{384,1024},{384,384},{192,384},{260,384}}; const char *names[5]={"gate/up","down","q/o","k/v","head"};
  size_t K=1024,s; void *mem=NULL; bcir_tensor_pool *pool=NULL;
  if (threads>1) { size_t bytes=bcir_tensor_pool_size(threads); mem=aligned_alloc(64,(bytes+63)/64*64); pool=bcir_tensor_pool_init(mem,bytes,threads); }
  for (s=0;s<5;s++) {
    size_t m=shapes[s][0],n=shapes[s][1],i,l; double ta[200],tt[200],tr[200]; int r;
    float *a=aligned_alloc(64,K*m*4),*at=aligned_alloc(64,K*m*4),*b=aligned_alloc(64,K*n*4),*c1=aligned_alloc(64,m*n*4),*c2=aligned_alloc(64,m*n*4),*c0=aligned_alloc(64,m*n*4);
    srand(7); for (i=0;i<K*m;i++) a[i]=(float)rand()/RAND_MAX-0.5f; for (i=0;i<K*n;i++) b[i]=(float)rand()/RAND_MAX-0.5f;
    for (i=0;i<m*n;i++) c0[i]=(float)rand()/RAND_MAX-0.5f;
    for (r=0;r<reps+2;r++) {
      double t0,t1;
      memcpy(c1,c0,m*n*4); t0=now(); bcir_tensor_mm_kblock_pool(pool,1,0,m,n,K,1.0f,a,b,1.0f,c1); t1=now(); if (r>=2) ta[r-2]=t1-t0;
      t0=now(); for (i=0;i<m;i++) for (l=0;l<K;l++) at[i*K+l]=a[l*m+i]; t1=now(); if (r>=2) tr[r-2]=t1-t0;
      memcpy(c2,c0,m*n*4); t0=now(); bcir_tensor_mm_kblock_pool(pool,0,0,m,n,K,1.0f,at,b,1.0f,c2); t1=now(); if (r>=2) tt[r-2]=t1-t0;
    }
    qsort(ta,reps,8,cmpd); qsort(tt,reps,8,cmpd); qsort(tr,reps,8,cmpd);
    printf("{\"shape\": \"%s\", \"threads\": %u, \"m\": %zu, \"n\": %zu, \"k\": %zu, \"ta1_ms\": %.3f, \"ta1_gflops\": %.1f, "
      "\"pretransposed_ms\": %.3f, \"pretransposed_gflops\": %.1f, \"transpose_ms\": %.3f, \"identical\": %s}\n",
      names[s],threads,m,n,K,ta[reps/2],2.0*m*n*K/ta[reps/2]/1e6,tt[reps/2],2.0*m*n*K/tt[reps/2]/1e6,tr[reps/2],
      memcmp(c1,c2,m*n*4) ? "false" : "true");
    free(a); free(at); free(b); free(c1); free(c2); free(c0);
  }
  if (pool) bcir_tensor_pool_stop(pool);
  free(mem); return 0;
}
