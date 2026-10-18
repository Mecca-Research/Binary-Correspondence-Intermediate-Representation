/* FMA throughput pinned to CPU sets, chains held in registers (explicit variables, no array): one CPU, each
 * pair, all four. A pair of hyperthreads on one physical core shares its FMA ports: their combined rate is
 * about one CPU's; two separate cores give twice. */
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
typedef struct { int cpu; double gflops; } arg;
__attribute__((target("avx512f"),noinline)) static float burn(long n) {
  __m512 m=_mm512_set1_ps(1.0000001f),c=_mm512_set1_ps(1e-7f);
  __m512 a0=_mm512_set1_ps(0),a1=_mm512_set1_ps(1),a2=_mm512_set1_ps(2),a3=_mm512_set1_ps(3),a4=_mm512_set1_ps(4),
      a5=_mm512_set1_ps(5),a6=_mm512_set1_ps(6),a7=_mm512_set1_ps(7),a8=_mm512_set1_ps(8),a9=_mm512_set1_ps(9),
      a10=_mm512_set1_ps(10),a11=_mm512_set1_ps(11);
  long it;
  for (it=0;it<n;it++) {
    a0=_mm512_fmadd_ps(a0,m,c); a1=_mm512_fmadd_ps(a1,m,c); a2=_mm512_fmadd_ps(a2,m,c); a3=_mm512_fmadd_ps(a3,m,c);
    a4=_mm512_fmadd_ps(a4,m,c); a5=_mm512_fmadd_ps(a5,m,c); a6=_mm512_fmadd_ps(a6,m,c); a7=_mm512_fmadd_ps(a7,m,c);
    a8=_mm512_fmadd_ps(a8,m,c); a9=_mm512_fmadd_ps(a9,m,c); a10=_mm512_fmadd_ps(a10,m,c); a11=_mm512_fmadd_ps(a11,m,c);
  }
  a0=_mm512_add_ps(a0,a1); a2=_mm512_add_ps(a2,a3); a4=_mm512_add_ps(a4,a5); a6=_mm512_add_ps(a6,a7);
  a8=_mm512_add_ps(a8,a9); a10=_mm512_add_ps(a10,a11); a0=_mm512_add_ps(a0,a2); a4=_mm512_add_ps(a4,a6);
  a8=_mm512_add_ps(a8,a10); a0=_mm512_add_ps(a0,a4); a0=_mm512_add_ps(a0,a8);
  return _mm512_reduce_add_ps(a0);
}
static void *spin(void *p) {
  arg *a=(arg *)p; cpu_set_t set; long n=200000000; double t0,t; float s;
  CPU_ZERO(&set); CPU_SET(a->cpu,&set); pthread_setaffinity_np(pthread_self(),sizeof set,&set);
  t0=now(); s=burn(n); t=now()-t0;
  a->gflops=(double)n*12*16*2/t/1e9; if (s==12345.f) printf("!");
  return 0;
}
static void run(const int *cpus, int k) {
  pthread_t th[8]; arg a[8]; int i; double sum=0;
  for (i=0;i<k;i++) { a[i].cpu=cpus[i]; pthread_create(&th[i],0,spin,&a[i]); }
  for (i=0;i<k;i++) { pthread_join(th[i],0); sum+=a[i].gflops; }
  printf("cpus"); for (i=0;i<k;i++) printf(" %d",cpus[i]); printf(": %.1f GFLOP/s total\n",sum);
}
int main(void) {
  int one[1]={0}, p01[2]={0,1}, p02[2]={0,2}, p03[2]={0,3}, p12[2]={1,2}, p13[2]={1,3}, p23[2]={2,3}, all[4]={0,1,2,3};
  run(one,1); run(p01,2); run(p02,2); run(p03,2); run(p12,2); run(p13,2); run(p23,2); run(all,4); run(one,1);
  return 0;
}
