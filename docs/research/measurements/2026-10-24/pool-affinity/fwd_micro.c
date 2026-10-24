/* The banded forward's group kernels (the tree's static att512_tq/tscores/tsoftmax/tcontext, the unit #included)
 * on cache-resident operands: one 16-row group at R with d channels, each phase timed alone (its input restored
 * untimed before each call), TSC cycles a probability, the median of REPS calls; THREADS copies run at once
 * (each its own buffers), a barrier before each phase's batch. usage: fwd_micro R D THREADS REPS */
#include "bcir_tensor_tree.c"
#include <pthread.h>
#include <stdio.h>
#include <x86intrin.h>
static size_t gR,gD; static int greps,gth; static pthread_barrier_t bar;
static double res[64][8];
static int cmpd(const void *a, const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y ? -1 : x>y; }
static void *alloc64(size_t n) { void *p=NULL; if (posix_memalign(&p,64,n ? n : 64)) exit(3); memset(p,0,n ? n : 64); return p; }
static void *worker(void *arg) {
  int id=(int)(size_t)arg,rep,ph; size_t R=gR,d=gD,nk=R+16,i,np=16*R+136;
  float *q=alloc64(16*d*4),*k=alloc64(nk*d*4),*v=alloc64(nk*d*4),*y=alloc64(nk*d*4);
  float *sg=alloc64(nk*16*4),*sc=alloc64(nk*16*4),*ex=alloc64(nk*16*4),*qt=alloc64(BCIR_ATT_TQ*16*4);
  _Alignas(64) float sums[16]; float scale=1.0f/sqrtf((float)d);
  double *t=malloc(sizeof(double)*greps);
  for (i=0;i<16*d;i++) q[i]=sinf((float)(i+id)*0.37f)*1.7f;
  for (i=0;i<nk*d;i++) { k[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
  att512_tq(d,q,d,qt); att512_tscores(R,d,qt,k,d,sc);
  memcpy(ex,sc,nk*16*4); att512_tsoftmax(R,scale,ex,sums);
  for (ph=0;ph<3;ph++) {
    pthread_barrier_wait(&bar);
    for (rep=0;rep<greps;rep++) {
      unsigned long long c0,c1;
      if (ph==1) memcpy(sg,sc,nk*16*4);
      if (ph==2) memcpy(sg,ex,nk*16*4);
      c0=__rdtsc();
      if (ph==0) { att512_tq(d,q,d,qt); att512_tscores(R,d,qt,k,d,sg); }
      else if (ph==1) att512_tsoftmax(R,scale,sg,sums);
      else att512_tcontext(R,d,sg,sums,v,d,y,d);
      c1=__rdtsc();
      t[rep]=(double)(c1-c0)/(double)np;
    }
    qsort(t,(size_t)greps,sizeof(double),cmpd);
    res[id][ph]=t[greps/2];
  }
  return NULL;
}
int main(int argc, char **argv) {
  if (argc!=5) return 2;
  gR=strtoul(argv[1],0,10); gD=strtoul(argv[2],0,10); gth=atoi(argv[3]); greps=atoi(argv[4]);
  pthread_t th[64]; int i,ph;
  pthread_barrier_init(&bar,NULL,(unsigned)gth);
  for (i=0;i<gth;i++) pthread_create(&th[i],NULL,worker,(void*)(size_t)i);
  for (i=0;i<gth;i++) pthread_join(th[i],NULL);
  printf("{\"R\":%zu,\"d\":%zu,\"threads\":%d,\"reps\":%d",gR,gD,gth,greps);
  static const char *nm[3]={"scores","softmax","context"};
  for (ph=0;ph<3;ph++) { printf(",\"%s\":[",nm[ph]); for (i=0;i<gth;i++) printf("%s%.3f",i?",":"",res[i][ph]); printf("]"); }
  printf("}\n");
  return 0;
}
