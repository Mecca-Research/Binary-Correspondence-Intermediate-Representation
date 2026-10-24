/* A sampling profiler for hosts without a PMU, preloaded (LD_PRELOAD): SIGPROF from ITIMER_PROF every SAMPLER_US
 * microseconds of the process's CPU time; the handler stores the interrupted instruction pointer; at exit the
 * pointers and /proc/self/maps go to SAMPLER_OUT (resolved offline by resolve_samples.py). */
#define _GNU_SOURCE 1
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <ucontext.h>
#define CAP (1u<<22)
static unsigned long ips[CAP];
static unsigned long count;
static void on_prof(int sig, siginfo_t *si, void *uc_) {
  ucontext_t *uc=(ucontext_t *)uc_; unsigned long i;
  (void)sig; (void)si;
  i=__atomic_fetch_add(&count,1ul,__ATOMIC_RELAXED);
  if (i<CAP) ips[i]=(unsigned long)uc->uc_mcontext.gregs[REG_RIP];
}
__attribute__((constructor)) static void sampler_start(void) {
  struct sigaction sa; struct itimerval it; const char *us=getenv("SAMPLER_US");
  long period=us ? atol(us) : 200;
  if (!getenv("SAMPLER_OUT")) return;
  memset(&sa,0,sizeof sa); sa.sa_sigaction=on_prof; sa.sa_flags=SA_SIGINFO|SA_RESTART; sigemptyset(&sa.sa_mask);
  sigaction(SIGPROF,&sa,NULL);
  it.it_interval.tv_sec=0; it.it_interval.tv_usec=period; it.it_value=it.it_interval;
  setitimer(ITIMER_PROF,&it,NULL);
}
__attribute__((destructor)) static void sampler_stop(void) {
  const char *out=getenv("SAMPLER_OUT"); struct itimerval it; FILE *f,*m; unsigned long i,n; char line[4096];
  if (!out) return;
  memset(&it,0,sizeof it); setitimer(ITIMER_PROF,&it,NULL);
  f=fopen(out,"w"); if (!f) return;
  m=fopen("/proc/self/maps","r");
  if (m) { while (fgets(line,sizeof line,m)) fprintf(f,"map %s",line); fclose(m); }
  n=count<CAP ? count : CAP;
  for (i=0;i<n;i++) fprintf(f,"ip %lx\n",ips[i]);
  fclose(f);
}
