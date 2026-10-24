/* membw: read and copy bandwidth over a buffer of N MiB with T threads (each its own slice), best of R. Each worker
 * reads the TSC around its own pass after the starting barrier, and a rep's time is the latest end less the earliest
 * start: the calling thread, which only waits, reading the clock after a barrier was descheduled behind the workers
 * on a host with no spare core and timed too short an interval (a first run's four-thread rows, discarded). */
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
static size_t per; static float *buf,*dst; static int mode; static double sink[64];
static pthread_barrier_t bar; static unsigned long long t_start[64],t_end[64];
__attribute__((target("avx512f"))) static void *work(void *arg) {
  size_t id=(size_t)arg,i; float *s=buf+id*per,*o=dst+id*per; __m512 a=_mm512_setzero_ps(),b=a,c=a,d=a;
  pthread_barrier_wait(&bar);
  t_start[id]=__rdtsc();
  if (mode==0) { for (i=0;i<per;i+=64) { a=_mm512_add_ps(a,_mm512_load_ps(s+i)); b=_mm512_add_ps(b,_mm512_load_ps(s+i+16)); c=_mm512_add_ps(c,_mm512_load_ps(s+i+32)); d=_mm512_add_ps(d,_mm512_load_ps(s+i+48)); } }
  else if (mode==1) { for (i=0;i<per;i+=16) _mm512_store_ps(o+i,_mm512_load_ps(s+i)); }
  else { for (i=0;i<per;i+=16) _mm512_stream_ps(o+i,_mm512_load_ps(s+i)); }
  __asm__ volatile("" :: "v"(a),"v"(b),"v"(c),"v"(d)); sink[id]=_mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a,b),_mm512_add_ps(c,d)));
  if (mode==2) _mm_sfence();
  t_end[id]=__rdtsc();
  pthread_barrier_wait(&bar);
  return NULL;
}
int main(int argc,char**argv){
  size_t mib=strtoul(argv[1],0,10),n=mib<<18; int th=atoi(argv[2]),reps=atoi(argv[3]),k,r;
  buf=aligned_alloc(64,n*4); dst=aligned_alloc(64,n*4); for (size_t z=0;z<n;z++) buf[z]=1.0f; memset(dst,0,n*4); per=n/th;
  const char *names[3]={"read","copy","copy-nt"};
  for (mode=0;mode<3;mode++) { double best=1e30;
    for (r=0;r<reps;r++) { pthread_t tid[64]; pthread_barrier_init(&bar,NULL,th+1);
      for (k=0;k<th;k++) pthread_create(&tid[k],NULL,work,(void*)(size_t)k);
      pthread_barrier_wait(&bar); pthread_barrier_wait(&bar);
      unsigned long long t0=t_start[0],t1=t_end[0];
      for (k=1;k<th;k++) { if (t_start[k]<t0) t0=t_start[k]; if (t_end[k]>t1) t1=t_end[k]; }
      for (k=0;k<th;k++) pthread_join(tid[k],NULL); pthread_barrier_destroy(&bar);
      double s=(double)(t1-t0)/2.3e9; if (s<best) best=s; }
    double bytes=(double)n*4*(mode?2:1);
    printf("%s %zu MiB %d threads: %.1f GB/s (read+write counted for copies)\n",names[mode],mib,th,bytes/best/1e9);
  }
  return 0;
}
