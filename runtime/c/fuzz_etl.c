/*===- fuzz_etl.c - libFuzzer target for the C event transducer -------------===
 * PORT-4: bcir_etl.c is a trust boundary (driver packets, device text). Each input is (1) mini-ROP text, parsed with
 * output capacities its first octet chooses, and (2) a finite-state machine its octets spell -- states, transitions
 * and their names sliced from the input -- compiled and run over the rest. Every verdict is held to its contract: a
 * refusal names offsets inside the input; an accepted parse's resources, claims and rid lists lie inside their
 * buffers and the text; a run's trace is a walk of the machine from its start, its last state the final one.
 * Build: clang -fsanitize=fuzzer,address,undefined fuzz_etl.c bcir_etl.c */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include "bcir_etl.h"

#define REQUIRE(c) do{ if(!(c)) abort(); }while(0)

static void rop(const uint8_t *d, size_t n){
  static bcir_rop_resource res[64]; static bcir_rop_claim cl[64]; static uint32_t rids[64];
  size_t cap=n ? (size_t)(d[0]%64u)+1u : 64u;
  bcir_rop_out o={res,cap,0,cl,cap,0,rids,cap,0,0,0,0,0,0};
  bcir_etl_status st=bcir_rop_parse((const char *)d,n,&o);
  if(st==BCIR_ETL_OK){
    REQUIRE(o.n_res<=cap && o.n_claims<=cap && o.n_rids<=cap);
    for(size_t k=0;k<o.n_res;k++){ REQUIRE(res[k].rid==10u+k && res[k].n>0 && res[k].at+res[k].n<=n); }
    for(size_t k=0;k<o.n_claims;k++){ const bcir_rop_claim *c=&cl[k];
      REQUIRE(c->id==1000u+k && c->name_n>0 && c->name_at+c->name_n<=n && c->lane<6 && c->stride<6);
      REQUIRE(c->rd_first+c->n_rd<=o.n_rids && c->wr_first+c->n_wr<=o.n_rids && c->count<=UINT64_C(0x7fffffffffffffff));
      for(size_t j=0;j<c->n_rd;j++) REQUIRE(rids[c->rd_first+j]>=10u && rids[c->rd_first+j]<10u+o.n_res);
      for(size_t j=0;j<c->n_wr;j++) REQUIRE(rids[c->wr_first+j]>=10u && rids[c->wr_first+j]<10u+o.n_res); } }
  else if(st!=BCIR_ETL_ERR_CAPACITY){
    REQUIRE(st>=BCIR_ETL_ERR_LEX && st<=BCIR_ETL_ERR_REPEATED);
    REQUIRE(o.pos<=n && o.tok_at+o.tok_n<=n); }
}

static void fsm(const uint8_t *d, size_t n){
  if(n<4) return;
  static bcir_fsm_state st[8]; static bcir_fsm_transition tx[16]; static uint32_t sx[16], dx[16];
  static bcir_etl_str sym[256]; static bcir_fsm_step trace[256];
  size_t ns=1u+d[0]%8u, nt=d[1]%17u, at=4;
  /* a name: up to two octets of the input; the empty one too */
  #define NAME(out) do{ size_t len=at<n ? d[at]%3u : 0u; at++; if(at+len>n) len=0; (out).s=(const char *)d+at; (out).n=len; at+=len; }while(0)
  for(size_t k=0;k<ns;k++){ NAME(st[k].name); st[k].accepting=at<n && (d[at]&1); st[k].error=at<n && (d[at]&2); at++; }
  for(size_t k=0;k<nt;k++){ NAME(tx[k].src); NAME(tx[k].dst); NAME(tx[k].on); NAME(tx[k].action); }
  bcir_etl_str start; NAME(start);
  bcir_fsm m; size_t where; int field;
  bcir_etl_status s=bcir_fsm_compile(&m,st,ns,tx,nt,start,sx,dx,&where,&field);
  if(s!=BCIR_ETL_OK){ REQUIRE(s>=BCIR_ETL_ERR_STATE_NAME && s<=BCIR_ETL_ERR_START && field>=0 && field<=2);
    REQUIRE(where<(s==BCIR_ETL_ERR_TX_FIELD||s==BCIR_ETL_ERR_TX_ENDPOINT||s==BCIR_ETL_ERR_TX_DUP ? nt : ns+1)); return; }
  size_t ny=0;
  for(;at<n && ny<256;ny++){ size_t len=d[at]%3u; at++; if(at+len>n) len=n-at; sym[ny].s=(const char *)d+at; sym[ny].n=len; at+=len; }
  size_t steps, fin; int acc;
  s=bcir_fsm_run(&m,sym,ny,trace,256,&steps,&fin,&acc);
  REQUIRE(s==BCIR_ETL_OK && steps<=ny && fin<ns);
  size_t cur=m.start;
  for(size_t k=0;k<steps;k++){ REQUIRE(trace[k].tx<nt && trace[k].sym==k && sx[trace[k].tx]==cur && !st[cur].error);
    cur=dx[trace[k].tx]; }
  REQUIRE(cur==fin);
  REQUIRE(!acc || (steps==ny && st[fin].accepting && !st[fin].error));
  if(ny && steps<ny) REQUIRE(!acc);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size){
  rop(data,size);
  fsm(data,size);
  return 0;
}
