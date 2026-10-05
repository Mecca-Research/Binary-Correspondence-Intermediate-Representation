/* The out-of-bounds counter under concurrent events (#atomicring, docs §5.12): the probe program the
 * runtime gate wrote out from a heredoc, as a harness the gate and the CMake build compile alike
 * (BUILD-2j, docs/BCIR_BUILD_ROADMAP.md); tools/c/sections/atomicring.sh runs it. Eight writers record
 * 50,000 events each while a reporter snapshots the ring, and the total must come out exact. A build
 * without atomics has a single-threaded contract, and says so instead.
 *
 * Exit 0 with `EXACT <n>` (or `SKIP ...` without atomics), 1 with `LOST <got>/<want>`, 2 when the
 * reporter thread cannot start, 3 when the reporter cannot write its snapshots. */
#include <stdio.h>
#include "bcir_quarantine.h"
#if BCIR_OOB_COUNTER_ATOMIC
#include <pthread.h>
#define NT 8
#define NM 50000
static void *hammer(void *arg){ (void)arg;
  for(long k=0;k<NM;k++) bcir_oob_record_event(1u, (uint64_t)k, 64u, "race:a");
  return 0; }
static void *observe(void *arg){ (void)arg; FILE *f=tmpfile(); if(!f) return (void *)1;
  for(int k=0;k<200;k++){ bcir_quarantine_report(f); rewind(f); }
  return fclose(f) ? (void *)1 : 0; }
int main(void){
  pthread_t th[NT], reader;
  bcir_quarantine_report(NULL); bcir_decide_report(NULL); /* public null handles are harmless */
  if(pthread_create(&reader,0,observe,0)) return 2;
  for(int i=0;i<NT;i++) pthread_create(&th[i],0,hammer,0);
  for(int i=0;i<NT;i++) pthread_join(th[i],0);
  void *reader_rc=0; pthread_join(reader,&reader_rc); if(reader_rc) return 3;
  unsigned long got = bcir_oob_count, want = (unsigned long)NT*NM;
  if(got==want) printf("EXACT %lu\n", got); else printf("LOST %lu/%lu\n", got, want);
  return got==want ? 0 : 1; }
#else
int main(void){ printf("SKIP (no atomics; single-threaded contract)\n"); return 0; }
#endif
