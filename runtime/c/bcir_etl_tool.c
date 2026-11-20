/*===- bcir_etl_tool.c - bcir-etl: the C event transducer, as text ---------===
 * PORT-4's host tool: `bcir-etl rop FILE`, `bcir-etl fsm FILE`, `bcir-etl event FILE`, each verdict spelled in the
 * oracle's words (bcir/etl), so bcir/tests/test_c_etl.py compares the two rails' text exactly.
 *
 *   rop    a mini-ROP program: one `resource <rid> <name>` line per resource, one `claim <id> rop.<name> rd=<rids>
 *          wr=<rids> count=<n> lane=<LANE> stride=<STRIDE>` line per claim, or one `ERR <message>` line
 *   fsm    lines `state NAME [accepting] [error]`, `tx SRC DST ON [ACTION]`, `start NAME` and `run SYM ...` (a run
 *          each); a name `""` is the empty one. `ERR <message>` for a machine refused, else one `run accepted=<0|1>
 *          final=<state> captures=<action:sym,...> trace=<state>/<sym>/<state>,...` line per run
 *   event  lines `stream NAME KIND ENCODING BITS WINDOW`, `kind NAME`, `event KIND BEGIN END`: `ok` or `ERR <message>`
 * A name here is printable ASCII with no quote or backslash, so the oracle's `repr` of it is `'name'`. A refusal of
 * the tool's own (a malformed line, a file it cannot read) is exit 2 with `TOOL <reason>`. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bcir_etl.h"
#include "bcir_host_alloc.h"

/* The file's octets and one more (a NUL the fsm and event modes read to), from the hosted allocator. */
static char *slurp(const bcir_host_allocator *a, const char *path, size_t *n){
  FILE *f=fopen(path,"rb"); if(!f) return NULL;
  size_t cap=4096, len=0; void *b=bcir_host_allocate(a,cap);
  for(;b;){ size_t r=fread((char *)b+len,1,cap-1-len,f); len+=r; if(len<cap-1) break;
    if(cap>SIZE_MAX/2 || !bcir_host_realloc_array(a,&b,cap,cap*2,1,0)){ bcir_host_deallocate(a,b); b=NULL; break; }
    cap*=2; }
  if(b && ferror(f)){ bcir_host_deallocate(a,b); b=NULL; }
  fclose(f);
  if(b) ((char *)b)[len]=0;
  *n=len; return (char *)b;
}
static int tool_fail(const char *why){ printf("TOOL %s\n",why); return 2; }
static void put_repr(bcir_etl_str s){ printf("'%.*s'",(int)s.n,s.s); }

/* --- rop ---------------------------------------------------------------------------------------------------------- */
static void *zeroed(const bcir_host_allocator *a, size_t count, size_t size){
  size_t bytes; return bcir_size_mul(count,size,&bytes)?bcir_host_allocate_zeroed(a,bytes):NULL;
}
static void release(const bcir_host_allocator *a, bcir_rop_out *out){
  bcir_host_deallocate(a,out->res); bcir_host_deallocate(a,out->claims); bcir_host_deallocate(a,out->rids);
}
static int run_rop(const bcir_host_allocator *a, const char *text, size_t n){
  size_t cap=n+1;                                   /* a program of n octets names fewer than n+1 of each */
  bcir_rop_out out; memset(&out,0,sizeof out);
  out.res=zeroed(a,cap,sizeof *out.res); out.claims=zeroed(a,cap,sizeof *out.claims); out.rids=zeroed(a,cap,sizeof *out.rids);
  out.cap_res=out.cap_claims=out.cap_rids=cap;
  if(!out.res||!out.claims||!out.rids){ release(a,&out); return tool_fail("out of memory"); }
  static const char *const lanes[]={"U","UX","T","GGG","A","H"};
  static const char *const strides[]={"SCALAR","UNIT","STRIDED","CACHELINE","TILE","RANDOM"};
  bcir_etl_status st=bcir_rop_parse(text,n,&out);
  const char *tk=out.tok_kind>=0?bcir_rop_tok_name((bcir_rop_tok)out.tok_kind):"?";
  int tn=(int)out.tok_n; const char *ts=text+out.tok_at;
  switch(st){
  case BCIR_ETL_OK:
    for(size_t k=0;k<out.n_res;k++) printf("resource %u %.*s\n",out.res[k].rid,(int)out.res[k].n,text+out.res[k].at);
    for(size_t k=0;k<out.n_claims;k++){ const bcir_rop_claim *c=&out.claims[k];
      printf("claim %u rop.%.*s rd=",c->id,(int)c->name_n,text+c->name_at);
      for(size_t j=0;j<c->n_rd;j++) printf("%s%u",j?",":"",out.rids[c->rd_first+j]);
      printf(" wr=");
      for(size_t j=0;j<c->n_wr;j++) printf("%s%u",j?",":"",out.rids[c->wr_first+j]);
      printf(" count=%llu lane=%s stride=%s\n",(unsigned long long)c->count,lanes[c->lane],strides[c->stride]); }
    break;
  case BCIR_ETL_ERR_LEX: {
    unsigned char ch=(unsigned char)text[out.pos];   /* the oracle's repr of a character: printable ASCII as itself */
    if(ch=='\'') printf("ERR unexpected character \"'\" at offset %zu\n",out.pos);
    else if(ch=='\\') printf("ERR unexpected character '\\\\' at offset %zu\n",out.pos);
    else if(ch>=0x20&&ch<0x7f) printf("ERR unexpected character '%c' at offset %zu\n",ch,out.pos);
    else if(ch<0x80) printf("ERR unexpected character '\\x%02x' at offset %zu\n",ch,out.pos);
    else printf("ERR unexpected non-ASCII character at offset %zu\n",out.pos);
    break; }
  case BCIR_ETL_ERR_EXPECT:
    if(out.want_kind<0) printf("ERR expected 'claim', got %s '%.*s' at offset %zu\n",tk,tn,ts,out.pos);
    else printf("ERR expected '%s', got %s '%.*s' at offset %zu\n",bcir_rop_tok_name((bcir_rop_tok)out.want_kind),tk,tn,ts,
                out.pos);
    break;
  case BCIR_ETL_ERR_SEPARATOR: printf("ERR expected ';' or '}', got %s '%.*s' at offset %zu\n",tk,tn,ts,out.pos); break;
  case BCIR_ETL_ERR_STATEMENT: printf("ERR unknown statement '%.*s' at offset %zu\n",tn,ts,out.pos); break;
  case BCIR_ETL_ERR_REPEATED: printf("ERR repeated statement '%.*s' at offset %zu\n",tn,ts,out.pos); break;
  case BCIR_ETL_ERR_LANE: printf("ERR unknown lane '%.*s' at offset %zu\n",tn,ts,out.pos); break;
  case BCIR_ETL_ERR_STRIDE: printf("ERR unknown stride '%.*s' at offset %zu\n",tn,ts,out.pos); break;
  case BCIR_ETL_ERR_COUNT: printf("ERR count %.*s is past the signed 64-bit wire domain at offset %zu\n",tn,ts,out.pos); break;
  default: printf("TOOL status %d\n",(int)st); release(a,&out); return 2;
  }
  release(a,&out);
  return 0;
}

/* --- fsm and event: whitespace-separated words, `""` the empty one ------------------------------------------------ */
typedef struct { bcir_etl_str w[64]; int n; } words;
static int split(char *line, words *ws){
  ws->n=0;
  for(char *p=line;*p;){
    while(*p==' '||*p=='\t') p++;
    if(!*p) break;
    char *q=p; while(*q && *q!=' ' && *q!='\t') q++;
    if(ws->n>=64) return 0;
    if(q-p==2 && p[0]=='"' && p[1]=='"') { ws->w[ws->n].s=p; ws->w[ws->n].n=0; }
    else { for(char *r=p;r<q;r++) if(*r=='\''||*r=='"'||*r=='\\'||*r<0x21||*r>0x7e) return 0;
      ws->w[ws->n].s=p; ws->w[ws->n].n=(size_t)(q-p); }
    ws->n++;
    if(*q) *q++=0;
    p=q; }
  return 1;
}
static int is_word(bcir_etl_str s, const char *z){ return s.n==strlen(z) && !memcmp(s.s,z,s.n); }
static int to_i64(bcir_etl_str s, long long *v){
  char b[32]; if(s.n==0||s.n>=sizeof b) return 0;
  memcpy(b,s.s,s.n); b[s.n]=0; char *e; *v=strtoll(b,&e,10); return *e==0; }

static int run_fsm(char *text){
  static bcir_fsm_state st[4096]; static bcir_fsm_transition tx[4096]; static uint32_t sx[4096], dx[4096];
  size_t ns=0, nt=0; bcir_etl_str start={NULL,0}; int have_start=0;
  char *runs[4096]; int nr=0;
  for(char *line=strtok(text,"\n");line;line=strtok(NULL,"\n")){
    char *keep=line; words ws;
    if(!strncmp(line,"run",3) && (line[3]==0||line[3]==' ')){ if(nr>=4096) return tool_fail("too many runs"); runs[nr++]=keep; continue; }
    if(!split(line,&ws)) return tool_fail("malformed line");
    if(ws.n==0) continue;
    if(is_word(ws.w[0],"state") && ws.n>=2){
      if(ns>=4096) return tool_fail("too many states");
      st[ns].name=ws.w[1]; st[ns].accepting=st[ns].error=0;
      for(int k=2;k<ws.n;k++){ if(is_word(ws.w[k],"accepting")) st[ns].accepting=1;
        else if(is_word(ws.w[k],"error")) st[ns].error=1; else return tool_fail("malformed state"); }
      ns++; }
    else if(is_word(ws.w[0],"tx") && (ws.n==4||ws.n==5)){
      if(nt>=4096) return tool_fail("too many transitions");
      tx[nt].src=ws.w[1]; tx[nt].dst=ws.w[2]; tx[nt].on=ws.w[3];
      tx[nt].action=ws.n==5?ws.w[4]:(bcir_etl_str){NULL,0}; nt++; }
    else if(is_word(ws.w[0],"start") && ws.n==2){ start=ws.w[1]; have_start=1; }
    else return tool_fail("malformed line"); }
  if(!have_start) return tool_fail("no start");
  bcir_fsm m; size_t where; int field;
  bcir_etl_status s=bcir_fsm_compile(&m,st,ns,tx,nt,start,sx,dx,&where,&field);
  static const char *const fields[]={"src","dst","on"};
  switch(s){
  case BCIR_ETL_OK: break;
  case BCIR_ETL_ERR_STATE_NAME: printf("ERR fsm state: name must be a non-empty string (got '')\n"); return 0;
  case BCIR_ETL_ERR_STATE_KIND: printf("ERR fsm state "); put_repr(st[where].name);
    printf(": a state is accepting or an error, not both\n"); return 0;
  case BCIR_ETL_ERR_TX_FIELD: printf("ERR fsm transition: %s must be a non-empty string (got '')\n",fields[field]); return 0;
  case BCIR_ETL_ERR_STATE_DUP: printf("ERR duplicate fsm state "); put_repr(st[where].name); printf("\n"); return 0;
  case BCIR_ETL_ERR_TX_ENDPOINT: printf("ERR fsm transition "); put_repr(tx[where].src); printf(" -> "); put_repr(tx[where].dst);
    printf(" names unknown state "); put_repr(field?tx[where].dst:tx[where].src); printf("\n"); return 0;
  case BCIR_ETL_ERR_TX_DUP: printf("ERR fsm transition on "); put_repr(tx[where].on); printf(" from state ");
    put_repr(tx[where].src); printf(" is declared twice\n"); return 0;
  case BCIR_ETL_ERR_START: printf("ERR unknown start state "); put_repr(start); printf("\n"); return 0;
  default: printf("TOOL status %d\n",(int)s); return 2; }
  for(int r=0;r<nr;r++){ words ws; if(!split(runs[r]+3,&ws)) return tool_fail("malformed run");
    static bcir_fsm_step trace[64]; size_t steps, fin; int acc;
    s=bcir_fsm_run(&m,ws.w,(size_t)ws.n,trace,64,&steps,&fin,&acc);
    if(s) { printf("TOOL status %d\n",(int)s); return 2; }
    printf("run accepted=%d final=%.*s captures=",acc,(int)st[fin].name.n,st[fin].name.s);
    int first=1;
    for(size_t k=0;k<steps;k++){ const bcir_fsm_transition *t=&tx[trace[k].tx]; if(!t->action.n) continue;
      printf("%s%.*s:%.*s",first?"":",",(int)t->action.n,t->action.s,(int)ws.w[trace[k].sym].n,ws.w[trace[k].sym].s); first=0; }
    printf(" trace=");
    for(size_t k=0;k<steps;k++){ uint32_t t=trace[k].tx;
      printf("%s%.*s/%.*s/%.*s",k?",":"",(int)st[sx[t]].name.n,st[sx[t]].name.s,(int)ws.w[trace[k].sym].n,
             ws.w[trace[k].sym].s,(int)st[dx[t]].name.n,st[dx[t]].name.s); }
    printf("\n"); }
  return 0;
}
static int run_event(char *text){
  for(char *line=strtok(text,"\n");line;line=strtok(NULL,"\n")){
    words ws; long long a, b;
    if(!split(line,&ws)) return tool_fail("malformed line");
    if(ws.n==0) continue;
    if(is_word(ws.w[0],"stream") && ws.n==6 && to_i64(ws.w[4],&a) && to_i64(ws.w[5],&b)){
      bcir_etl_status s=bcir_event_stream_check(ws.w[1],ws.w[2],ws.w[3],a,b);
      if(s==BCIR_ETL_OK){ printf("ok\n"); continue; }
      printf("ERR ");
      if(s==BCIR_ETL_ERR_STREAM_NAME) printf("event stream: name must be a non-empty string (got '')\n");
      else { printf("stream "); put_repr(ws.w[1]);
        if(s==BCIR_ETL_ERR_STREAM_KIND){ printf(": kind must be one of text|binary|telemetry|packet|driver|token (got ");
          put_repr(ws.w[2]); printf(")\n"); }
        else if(s==BCIR_ETL_ERR_STREAM_ENCODING){ printf(": encoding must be one of utf8|bytes|le|be|records|custom (got ");
          put_repr(ws.w[3]); printf(")\n"); }
        else if(s==BCIR_ETL_ERR_STREAM_BITS) printf(": element_bits must be positive (got %lld)\n",a);
        else printf(": max_window must be positive (got %lld)\n",b); } }
    else if(is_word(ws.w[0],"kind") && ws.n==2){
      if(bcir_event_kind_check(ws.w[1])==BCIR_ETL_OK) printf("ok\n");
      else printf("ERR event kind: name must be a non-empty string (got '')\n"); }
    else if(is_word(ws.w[0],"event") && ws.n==4 && to_i64(ws.w[2],&a) && to_i64(ws.w[3],&b)){
      bcir_etl_status s=bcir_event_check(ws.w[1],a,b);
      if(s==BCIR_ETL_OK) printf("ok\n");
      else if(s==BCIR_ETL_ERR_EVENT_KIND) printf("ERR event: kind must be a non-empty string (got '')\n");
      else { printf("ERR event "); put_repr(ws.w[1]); printf(": span must satisfy 0 <= begin <= end (got (%lld, %lld))\n",a,b); } }
    else return tool_fail("malformed line"); }
  return 0;
}

int main(int argc, char **argv){
  if(argc!=3) { fprintf(stderr,"usage: bcir-etl {rop|fsm|event} FILE\n"); return 2; }
  bcir_host_allocator a=bcir_host_allocator_or_default(NULL);
  size_t n; char *text=slurp(&a,argv[2],&n);
  if(!text) return tool_fail("cannot read the file");
  int rc;
  if(!strcmp(argv[1],"rop")) rc=run_rop(&a,text,n);
  else if(memchr(text,0,n)) rc=tool_fail("a NUL in the file");
  else if(!strcmp(argv[1],"fsm")) rc=run_fsm(text);
  else if(!strcmp(argv[1],"event")) rc=run_event(text);
  else rc=tool_fail("unknown mode");
  bcir_host_deallocate(&a,text); return rc;
}
