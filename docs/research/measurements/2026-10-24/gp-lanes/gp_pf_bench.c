/* GP-PF-0 (gp_pf_bench.c): the lane forward a group of sixteen rows at a time over N rows written to a cold buffer (the model's
 * per-layer features, larger than L2), with and without prefetching the next group's output lines for writing
 * (and its inputs for reading) -- values compared byte for byte; ns a row, median of 9 rounds. usage: pf_bench N */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define GP_RANK 22
#define GP_PROJECTOR 0
#define GP_SYMMETRIC 0
#include "gp_scalar.inc"
#include "gp_lanes_pair.inc"
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static int cmpd(const void *a,const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y; }
GPL_TARGET static void group(const float *z,size_t n,float *p,size_t r0,int pf) {
  size_t l,i; unsigned char ill[GPL]; _Alignas(64) float U[GP_RANK][GPL],V[GP_RANK][GPL],O[GP_P][GPL];
  if (pf && r0+2*GPL<=n) {
    const char *w=(const char *)(p+(r0+GPL)*GP_P),*a=(const char *)(z+(r0+GPL)*GP_RANK),*b=(const char *)(z+(n+r0+GPL)*GP_RANK);
    for (i=0;i<GPL*GP_P*4;i+=64) __builtin_prefetch(w+i,1,3);
    for (i=0;i<GPL*GP_RANK*4;i+=64) { __builtin_prefetch(a+i,0,3); __builtin_prefetch(b+i,0,3); }
  }
  gpl_in(z+r0*GP_RANK,GP_RANK,U); gpl_in(z+(n+r0)*GP_RANK,GP_RANK,V);
  gp_pair_forward_lanes(U,V,O,ill); gpl_out(O,GP_P,p+r0*GP_P,ill);
  for (l=0;l<GPL;l++) if (ill[l]) gp_pair_forward(z+(r0+l)*GP_RANK,z+(n+r0+l)*GP_RANK,p+(r0+l)*GP_P);
}
int main(int argc,char **argv) {
  size_t n=argc>1?strtoul(argv[1],0,10):4096,i,r,g; int k,rounds=9; uint64_t s=88172645463325252ull;
  float *z=malloc(2*n*GP_RANK*4),*p0=malloc(n*GP_P*4),*p1=malloc(n*GP_P*4),*p2=malloc(n*GP_P*4);
  size_t cold=64u<<20; char *flush=malloc(cold); double ts[3][9];
  n=n/GPL*GPL;
  for (i=0;i<2*n*GP_RANK;i++) { s^=s<<13; s^=s>>7; s^=s<<17; z[i]=(float)((double)(s>>11)/9007199254740992.0*2-1); }
  for (k=0;k<rounds;k++) {
    double t0;
    memset(flush,k,cold);  /* the outputs cold: 64 MiB written between rounds */
    t0=now(); for (r=0;r<n;r++) gp_pair_forward(z+r*GP_RANK,z+(n+r)*GP_RANK,p0+r*GP_P); ts[0][k]=(now()-t0)*1e9/n;
    memset(flush,k+1,cold);
    t0=now(); for (g=0;g<n;g+=GPL) group(z,n,p1,g,0); ts[1][k]=(now()-t0)*1e9/n;
    memset(flush,k+2,cold);
    t0=now(); for (g=0;g<n;g+=GPL) group(z,n,p2,g,1); ts[2][k]=(now()-t0)*1e9/n;
  }
  for (k=0;k<3;k++) qsort(ts[k],(size_t)rounds,sizeof(double),cmpd);
  printf("{\"rows\": %zu, \"bytes_equal\": %s, \"ns_a_row\": {\"scalar\": %.1f, \"lanes\": %.1f, \"lanes_prefetch\": %.1f}}\n",
         n,(!memcmp(p0,p1,n*GP_P*4) && !memcmp(p0,p2,n*GP_P*4)) ? "true" : "false",ts[0][rounds/2],ts[1][rounds/2],ts[2][rounds/2]);
  return 0;
}
