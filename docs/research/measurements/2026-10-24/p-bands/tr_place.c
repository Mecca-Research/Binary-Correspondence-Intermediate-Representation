/* tr_place: the unit's own transposed product (dV = P^T dY, att512_transposed) and context product
 * (dQ = dS K, att512_context) over a whole head's P (T x T, the lower triangle written), d = 32, P resident in L3
 * (one buffer, reused) or streamed from DRAM (NBUF buffers in turn); TSC cycles a probability, the median of
 * the calls. The unit is #included (-DUNIT='"copy.c"') so its static kernels are called as the backward calls
 * them. usage: tr_place NBUF CALLS WHICH(tr|ctx) */
#include UNIT
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
static int cmpd(const void *a, const void *b) { double x=*(const double *)a,y=*(const double *)b; return x<y ? -1 : x>y; }
int main(int argc, char **argv) {
  size_t t=2048,d=32,nbuf=strtoul(argv[1],0,10),calls=strtoul(argv[2],0,10),i,j,c,k; int ctx=argv[3][0]=='c';
  float **p=malloc(nbuf*sizeof *p),*x=aligned_alloc(64,t*d*4),*out=aligned_alloc(64,t*d*4);
  double *cyc=malloc(calls*sizeof *cyc),probs=(double)t*(t+1)/2;
  for (k=0;k<nbuf;k++) { p[k]=aligned_alloc(64,t*t*4); for (i=0;i<t;i++) for (j=0;j<t;j++) p[k][i*t+j]=j<=i ? (float)((i*7+j*3+k)%17)*0.01f : 0.0f; }
  for (i=0;i<t*d;i++) { x[i]=(float)(i%13)*0.1f-0.5f; out[i]=0.0f; }
  for (c=0;c<calls+2;c++) {
    float *pk=p[c%nbuf]; unsigned long long t0=__rdtsc();
    if (ctx) att512_context(t,d,pk,t,x,d,out,d,1); else att512_transposed(t,t,d,pk,t,x,d,out,d,1);
    unsigned long long t1=__rdtsc();
    if (c>=2) cyc[c-2]=(double)(t1-t0)/probs;
  }
  double h=0; for (i=0;i<t*d;i++) h+=out[i]*(double)(i%7+1);
  qsort(cyc,calls,sizeof *cyc,cmpd);
  printf("%s nbuf=%zu: median %.3f min %.3f TSC cycles a probability (v1's rate %.0f%%) check %.9g\n",ctx ? "ctx" : "tr",nbuf,
         cyc[calls/2],cyc[0],100*1.443/cyc[calls/2],h);
  return 0;
}
