/* One k-blocked GEMM shape CALLS times back to back on a pool of THREADS, ROUNDS rounds; per worker the mean cycles from
 * the job's dispatch to entering it, inside its tasks, and to leaving it, its tasks a job; the job's mean wall cycles; and the jobs each worker ran on each CPU (since the warm-up).
 * Built against probe_tasks.py's copy of the unit. usage: gemm_tasks THREADS TA TB M N K CALLS ROUNDS */
#define _POSIX_C_SOURCE 200809L
#include "bcir_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void bcir_probe_read(unsigned long long *out);
void bcir_probe_cpus(unsigned long long *out);
static void *alloc64(size_t n) { void *p=NULL; if (posix_memalign(&p,64,n)) exit(3); memset(p,0,n); return p; }
int main(int argc, char **argv) {
  unsigned long long pr[34],acc[34]; unsigned th; int ta,tb,calls,rounds,r,c,w; size_t m,n,k,i;
  if (argc!=9) return 2;
  th=(unsigned)atoi(argv[1]); ta=atoi(argv[2]); tb=atoi(argv[3]); m=strtoul(argv[4],0,10); n=strtoul(argv[5],0,10);
  k=strtoul(argv[6],0,10); calls=atoi(argv[7]); rounds=atoi(argv[8]);
  float *a=alloc64(m*k*4),*b=alloc64(k*n*4),*cc=alloc64(m*n*4);
  for (i=0;i<m*k;i++) a[i]=(float)(i%13)*0.1f; for (i=0;i<k*n;i++) b[i]=(float)(i%7)*0.2f;
  size_t ps=bcir_tensor_pool_size(th); void *pm=alloc64((ps+63)/64*64);
  bcir_tensor_pool *pool=bcir_tensor_pool_init(pm,ps,th);
  for (c=0;c<calls;c++) bcir_tensor_mm_kblock_pool(pool,ta,tb,m,n,k,1.0f,a,b,0.0f,cc);
  bcir_probe_read(pr); memset(acc,0,sizeof acc);
  for (r=0;r<rounds;r++) {
    for (c=0;c<calls;c++) bcir_tensor_mm_kblock_pool(pool,ta,tb,m,n,k,1.0f,a,b,0.0f,cc);
    bcir_probe_read(pr); for (i=0;i<34;i++) acc[i]+=pr[i];
  }
  double jobs=(double)acc[33];
  printf("%d,%d,%zu,%zu,%zu threads %u: wall %.0f cycles a job (%.1f us at the TSC's 2.3 GHz)\n",ta,tb,m,n,k,th,
         acc[32]/jobs,acc[32]/jobs/2300.0);
  for (w=0;w<(int)th;w++)
    printf("  worker %d: enters at %6.0f, busy %6.0f, leaves at %6.0f cycles; %.2f tasks a job\n",w,acc[4*w]/jobs,
           acc[4*w+1]/jobs,acc[4*w+2]/jobs,acc[4*w+3]/jobs);
  {
    unsigned long long cp[64];
    bcir_probe_cpus(cp);
    for (w=0;w<(int)th;w++) {
      printf("  worker %d on cpus:",w);
      for (i=0;i<8;i++) printf(" %llu",cp[8*w+i]);
      printf("\n");
    }
  }
  bcir_tensor_pool_stop(pool);
  return 0;
}
