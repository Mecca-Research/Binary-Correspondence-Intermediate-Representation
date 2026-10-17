/* Pool latency on this host: a job of `tasks` tasks, each `work` ns of spinning, back to back or
 * after a serial gap on the caller; mean microseconds per job over many jobs, best of 5 rounds. */
#define _POSIX_C_SOURCE 199309L
#include "bcir_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static void busy(double ns) { double t0=now_ns(); while (now_ns()-t0<ns) {} }
typedef struct job { double work; volatile unsigned hits[4096]; } job;
static void task(void *ctx, size_t t) { job *j=(job *)ctx; if (j->work>0) busy(j->work); j->hits[t&4095]++; }
static _Alignas(64) unsigned char memory[1<<16];
int main(int argc, char **argv) {
  unsigned threads=argc>1 ? (unsigned)atoi(argv[1]) : 4; int jobs=argc>2 ? atoi(argv[2]) : 4000;
  static job j; bcir_tensor_pool *pool=bcir_tensor_pool_init(memory,sizeof memory,threads);
  struct { const char *name; size_t tasks; double work,gap; } cases[]={
    {"empty x4, back to back",4,0,0}, {"empty x4, 20us gap",4,0,20000}, {"empty x4, 200us gap",4,0,200000},
    {"16 x 25us, back to back",16,25000,0}, {"16 x 25us, 20us gap",16,25000,20000},
    {"64 x 25us, back to back",64,25000,0}, {"13 x 25us (ragged), back to back",13,25000,0},
  };
  size_t c; int r,i;
  if (!pool) { fprintf(stderr,"no pool\n"); return 1; }
  for (c=0;c<sizeof cases/sizeof cases[0];c++) {
    double best=1e30; int n=cases[c].gap>=200000 ? jobs/8 : jobs;
    j.work=cases[c].work;
    for (r=0;r<5;r++) {
      double t=0;
      for (i=0;i<n;i++) {
        double t0;
        if (cases[c].gap>0) busy(cases[c].gap);
        t0=now_ns(); bcir_tensor_pool_run(pool,task,&j,cases[c].tasks); t+=now_ns()-t0;
      }
      if (t/n<best) best=t/n;
    }
    printf("%-36s %9.2f us/job (ideal %8.2f)\n",cases[c].name,best/1e3,
      cases[c].work*(double)((cases[c].tasks+threads-1)/threads)/1e3);
  }
  bcir_tensor_pool_stop(pool);
  return 0;
}
