/* One 16-row group at R, d channels, its phases in sequence as the stored forward runs them (tq+scores, softmax,
 * context), the group's slot SLOT bytes apart in a pool of POOL MiB a thread, a new slot each call (MODE 1: cold,
 * as a stored head's bands are) or the same one (MODE 0); MODE 2: the group in a resident scratch slot, then its
 * (R+16) x 16 floats streamed out to the pool's slot (non-temporal stores, sfence) -- the same values;
 * MODE 3: the same, copied out with ordinary stores. TSC cycles a probability by phase, the median of REPS calls,
 * THREADS copies at once. usage: fwd_cold R D THREADS REPS MODE POOLMIB */
#include "bcir_tensor_tree.c"
#include <pthread.h>
#include <stdio.h>
#include <x86intrin.h>
static size_t gR,gD,gpool; static int greps,gth,gmode; static pthread_barrier_t bar;
static double res[64][8]; static double chk[64];
static int cmpd(const void *a, const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y ? -1 : x>y; }
static void *alloc64(size_t n) { void *p=NULL; if (posix_memalign(&p,64,n ? n : 64)) exit(3); memset(p,0,n ? n : 64); return p; }
__attribute__((target("avx512f"))) static void stream_out(float *dst, const float *src, size_t n, int nt) {
  size_t i;
  if (nt) { for (i=0;i<n;i+=16) _mm512_stream_ps(dst+i,_mm512_load_ps(src+i)); _mm_sfence(); }
  else for (i=0;i<n;i+=16) _mm512_store_ps(dst+i,_mm512_load_ps(src+i));
}
static void *worker(void *arg) {
  int id=(int)(size_t)arg,rep,k; size_t R=gR,d=gD,nk=R+16,i,np=16*R+136,slotb=(size_t)2048*16*4,nslots;
  float *q=alloc64(16*d*4),*kk=alloc64(nk*d*4),*v=alloc64(nk*d*4),*y=alloc64(nk*d*4);
  float *scr=alloc64(nk*16*4),*qt=alloc64(BCIR_ATT_TQ*16*4);
  float *pool=alloc64(gpool<<20);
  _Alignas(64) float sums[16]; float scale=1.0f/sqrtf((float)d);
  double *t[4]; for (k=0;k<4;k++) t[k]=malloc(sizeof(double)*greps);
  nslots=(gpool<<20)/slotb;
  for (i=0;i<16*d;i++) q[i]=sinf((float)(i+id)*0.37f)*1.7f;
  for (i=0;i<nk*d;i++) { kk[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
  pthread_barrier_wait(&bar);
  for (rep=-8;rep<greps;rep++) {
    float *slot=pool+(gmode==0 ? 0 : (size_t)((rep+8)%nslots)*(slotb/4)),*sg=gmode>=2 ? scr : slot;
    unsigned long long c0,c1,c2,c3,c4;
    c0=__rdtsc();
    att512_tq(d,q,d,qt); att512_tscores(R,d,qt,kk,d,sg);
    c1=__rdtsc();
    att512_tsoftmax(R,scale,sg,sums);
    c2=__rdtsc();
    att512_tcontext(R,d,sg,sums,v,d,y,d);
    c3=__rdtsc();
    if (gmode>=2) stream_out(slot,scr,nk*16,gmode==2);
    c4=__rdtsc();
    if (rep>=0) { t[0][rep]=(double)(c1-c0)/np; t[1][rep]=(double)(c2-c1)/np; t[2][rep]=(double)(c3-c2)/np; t[3][rep]=(double)(c4-c3)/np; }
  }
  for (k=0;k<4;k++) { qsort(t[k],(size_t)greps,sizeof(double),cmpd); res[id][k]=t[k][greps/2]; }
  { double s=0; for (i=0;i<nk*16;i++) s+=pool[i]*(double)(i%7+1); for (i=0;i<nk*d;i++) s+=y[i]; chk[id]=s; }
  return NULL;
}
int main(int argc, char **argv) {
  if (argc!=7) return 2;
  gR=strtoul(argv[1],0,10); gD=strtoul(argv[2],0,10); gth=atoi(argv[3]); greps=atoi(argv[4]); gmode=atoi(argv[5]); gpool=strtoul(argv[6],0,10);
  pthread_t th[64]; int i,k;
  pthread_barrier_init(&bar,NULL,(unsigned)gth);
  for (i=0;i<gth;i++) pthread_create(&th[i],NULL,worker,(void*)(size_t)i);
  for (i=0;i<gth;i++) pthread_join(th[i],NULL);
  static const char *nm[4]={"scores","softmax","context","copyout"};
  double sum[4]={0,0,0,0};
  for (k=0;k<4;k++) for (i=0;i<gth;i++) sum[k]+=res[i][k]/gth;
  printf("{\"R\":%zu,\"d\":%zu,\"threads\":%d,\"reps\":%d,\"mode\":%d,\"pool_mib\":%zu",gR,gD,gth,greps,gmode,gpool);
  for (k=0;k<4;k++) printf(",\"%s\":%.3f",nm[k],sum[k]);
  printf(",\"total\":%.3f,\"check\":%.6g}\n",sum[0]+sum[1]+sum[2]+sum[3],chk[0]);
  return 0;
}
