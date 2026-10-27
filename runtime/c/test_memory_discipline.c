/*===- test_memory_discipline.c - allocator/context/channel regressions -----===*/
#include "bcir_cfront.h"
#include "bcir_cpp.h"
#include "bcir_host_alloc.h"
#include "bcir_llama.h"
#include "bcir_llama_ops.h"
#include "bcir_q8_model.h"
#include "bcir_runtime.h"
#include "bcir_runtime_channel.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct fault_state {
  size_t attempt;
  size_t fail_at;
  size_t live;
} fault_state;

static void *fault_allocate(void *opaque,size_t size){
  fault_state *state=(fault_state *)opaque;
  state->attempt++;
  if(state->fail_at&&state->attempt==state->fail_at)return NULL;
  void *allocation=malloc(size?size:1u);
  if(allocation)state->live++;
  return allocation;
}

static void *fault_reallocate(void *opaque,void *allocation,size_t size){
  fault_state *state=(fault_state *)opaque;
  state->attempt++;
  if(state->fail_at&&state->attempt==state->fail_at)return NULL;
  void *replacement=realloc(allocation,size?size:1u);
  if(replacement&&!allocation)state->live++;
  return replacement;
}

static void fault_deallocate(void *opaque,void *allocation){
  fault_state *state=(fault_state *)opaque;
  if(!allocation)return;
  if(!state->live){fprintf(stderr,"allocator live-count underflow\n");abort();}
  state->live--;
  free(allocation);
}

static bcir_host_allocator fault_allocator(fault_state *state){
  bcir_host_allocator allocator;
  allocator.context=state;allocator.allocate=fault_allocate;
  allocator.reallocate=fault_reallocate;allocator.deallocate=fault_deallocate;
  return allocator;
}

#define CHECK(condition) do { if(!(condition)){ \
  fprintf(stderr,"memory-discipline check failed at %s:%d: %s\n",__FILE__,__LINE__,#condition); \
  return 1; } } while(0)

static int checked_growth_test(void){
  fault_state state={0};
  bcir_host_allocator allocator=fault_allocator(&state);
  bcir_host_allocator invalid={0};
  bcir_host_arena arena;
  unsigned char *bytes=(unsigned char *)bcir_host_allocate(&allocator,4u);
  void *original;
  size_t out;
  CHECK(bytes&&state.live==1u);
  memset(bytes,0x5a,4u);original=bytes;
  state.attempt=0;state.fail_at=1;
  CHECK(!bcir_host_realloc_array(&allocator,(void **)&bytes,4u,16u,1u,0));
  CHECK(bytes==original&&bytes[0]==0x5a&&state.live==1u);
  CHECK(!bcir_size_add(SIZE_MAX,1u,&out));
  CHECK(!bcir_size_mul(SIZE_MAX,2u,&out));
  CHECK(!bcir_size_align_up(SIZE_MAX,8u,&out));
  CHECK(bcir_crc32(NULL,1u)==0u);
  CHECK(!bcir_host_arena_init(&arena,&invalid,128u));
  CHECK(bcir_host_arena_init(&arena,NULL,128u));
  CHECK(bcir_host_arena_allocate(&arena,8u,3u)==NULL);
  CHECK(arena.blocks==NULL);
  CHECK(bcir_host_arena_allocate(&arena,8u,_Alignof(max_align_t))!=NULL);
  bcir_host_arena_destroy(&arena);bcir_host_arena_destroy(&arena);
  bcir_host_deallocate(&allocator,bytes);
  CHECK(state.live==0u);
  return 0;
}

static int cpp_once(fault_state *state,size_t fail_at,size_t *attempts){
  bcir_host_allocator allocator=fault_allocator(state);
  bcir_cpp_context context;
  char output[256],error[128];
  state->fail_at=0;state->attempt=0;
  CHECK(!bcir_cpp_context_init(&context,&allocator));
  state->attempt=0;state->fail_at=fail_at;
  int rc=bcir_cpp_run_ex_context(&context,"#define VALUE 7\nVALUE\n","fault.c",
                                  NULL,0,NULL,0,output,sizeof output,error,sizeof error);
  *attempts=state->attempt;
  if(fail_at){CHECK(rc!=0);CHECK(output[0]==0);}
  else {CHECK(rc==0);CHECK(strstr(output,"7")!=NULL);}
  bcir_cpp_context_destroy(&context);
  bcir_cpp_context_destroy(&context);
  CHECK(state->live==0u);
  return 0;
}

static int cpp_fault_and_isolation_test(void){
  fault_state state={0};size_t attempts=0;
  CHECK(!cpp_once(&state,0,&attempts));
  CHECK(attempts>0);
  for(size_t fail=1;fail<=attempts;fail++){
    memset(&state,0,sizeof state);
    CHECK(!cpp_once(&state,fail,&attempts));
  }
  {
    bcir_cpp_context left,right;
    char a[256],b[4096],error[128],paste[2048];
    const char *bad_entries[1]={NULL};
    const char *too_many_dirs[65];
    for(size_t k=0;k<65u;k++)too_many_dirs[k]=".";
    CHECK(!bcir_cpp_context_init(&left,NULL));
    CHECK(!bcir_cpp_context_init(&right,NULL));
    CHECK(!bcir_cpp_run_context(&left,"#define PRIVATE 11\nPRIVATE\n",NULL,
                                a,sizeof a,error,sizeof error));
    CHECK(!bcir_cpp_run_context(&right,
      "#ifdef PRIVATE\nbad\n#else\ngood\n#endif\n",NULL,
      b,sizeof b,error,sizeof error));
    CHECK(strstr(a,"11")!=NULL&&strstr(b,"good")!=NULL&&strstr(b,"bad")==NULL);
    strcpy(b,"stale");
    CHECK(bcir_cpp_run_context(&right,NULL,NULL,b,sizeof b,error,sizeof error)!=0);
    CHECK(b[0]==0);
    strcpy(b,"stale");
    CHECK(bcir_cpp_run_ex_context(&right,"x\n","bad.c",bad_entries,1,NULL,0,
                                  b,sizeof b,error,sizeof error)!=0);
    CHECK(b[0]==0);
    strcpy(b,"stale");
    CHECK(bcir_cpp_run_ex_context(&right,"x\n","bad.c",NULL,0,bad_entries,1,
                                  b,sizeof b,error,sizeof error)!=0);
    CHECK(b[0]==0);
    strcpy(b,"stale");
    CHECK(bcir_cpp_run_ex_context(&right,"x\n","bad.c",too_many_dirs,65,NULL,0,
                                  b,sizeof b,error,sizeof error)!=0);
    CHECK(b[0]==0);
    {
      /* the paste written a piece at a time, each piece checked whole before the next is placed after it */
      int n=snprintf(paste,sizeof paste,"#define CAT(a,b) a ## b\nCAT(x,");
      size_t w;
      CHECK(n>0&&(size_t)n<sizeof paste);
      w=(size_t)n;
      for(size_t k=0;k<200u;k++){
        n=snprintf(paste+w,sizeof paste-w,"y+");
        CHECK(n>0&&(size_t)n<sizeof paste-w);
        w+=(size_t)n;
      }
      n=snprintf(paste+w,sizeof paste-w,"z)\n");
      CHECK(n>0&&(size_t)n<sizeof paste-w);
      CHECK(!bcir_cpp_run_context(&right,paste,NULL,b,sizeof b,error,sizeof error));
      CHECK(strstr(b,"x")!=NULL&&strstr(b,"z")!=NULL);
    }
    bcir_cpp_context_destroy(&left);bcir_cpp_context_destroy(&right);
  }
  return 0;
}

static const char cfront_source[]=
  "uint32_t cat(uint32_t i) {\n"
  "  uint32_t n = sizeof(\"abc\" \"de\");\n"
  "  uint32_t b = \"hi \" \"there\"[i % 8];\n"
  "  return n + b;\n"
  "}\n";

/* CF-BUF: a unit whose verified C outgrows the emit's first block several times over, and whose function-pointer
 * parameters grow the typedef prelude -- the emit and its prelude grow through the context's allocator, so the
 * sweep below fails each of those growths in turn. `funcs` functions, written into `src`. */
enum { WIDE_FUNCS = 64, WIDE_SOURCE = 32768 };
static int cfront_wide_source(char *src,size_t cap){
  size_t w=0;
  for(int k=0;k<WIDE_FUNCS;k++){
    int n=snprintf(src+w,cap-w,
      "uint32_t wide_%d(uint32_t (*op_%d)(uint32_t, uint32_t), uint32_t a, uint32_t b) {\n"
      "  uint32_t s = a * %du + b;\n  s ^= s >> 3;\n  s += b * %du;\n  return s + %du + op_%d(a, b);\n}\n",
      k,k,k+3,k+5,k+7,k);
    CHECK(n>0&&(size_t)n<cap-w);
    w+=(size_t)n;
  }
  return 0;
}

/* One compile under the fault allocator (fail_at 0: none fails). A failure is the whole compile's: no unit, no
 * emitted C -- never a partial text -- and every block released; a success emits `expect` whole. */
static int cfront_once(const char *source,const char *expect,size_t min_emit,fault_state *state,
                       size_t fail_at,size_t *attempts){
  bcir_host_allocator allocator=fault_allocator(state);
  bcir_cfront_context context;
  bcir_cfront_result result;
  state->fail_at=0;state->attempt=0;
  bcir_cfront_result_init(&result);
  CHECK(!bcir_cfront_context_init(&context,&allocator));
  state->attempt=0;state->fail_at=fail_at;
  int rc=bcir_cfront_compile_context(&context,source,&result);
  *attempts=state->attempt;
  if(fail_at){
    CHECK(rc!=0);CHECK(result.unit.funcs==NULL);CHECK(result.unit.n_funcs==0);
    CHECK(!result.emitted_ok&&result.emitted==NULL&&result.emitted_len==0);
    CHECK(!strcmp(result.diag,"oom"));
  }else{
    CHECK(rc==0&&result.ok&&result.emitted_ok&&result.emitted!=NULL);
    CHECK(result.emitted_len==strlen(result.emitted)&&result.emitted_len>=min_emit);
    CHECK(strstr(result.emitted,expect)!=NULL);
  }
  bcir_cfront_free(&result);
  CHECK(result.unit.funcs==NULL&&result.unit.n_funcs==0&&result.ok==0);
  CHECK(result.emitted_ok==0&&result.emitted==NULL&&result.emitted_len==0&&result.diag[0]==0);
  bcir_cfront_free(&result);
  bcir_cfront_context_destroy(&context);bcir_cfront_context_destroy(&context);
  CHECK(state->live==0u);
  return 0;
}

static int cfront_sweep(const char *source,const char *expect,size_t min_emit,size_t min_attempts){
  fault_state state={0};size_t attempts=0,baseline;
  CHECK(!cfront_once(source,expect,min_emit,&state,0,&attempts));
  baseline=attempts;CHECK(baseline>=min_attempts);
  for(size_t fail=1;fail<=baseline;fail++){
    memset(&state,0,sizeof state);
    CHECK(!cfront_once(source,expect,min_emit,&state,fail,&attempts));
  }
  return 0;
}

static int cfront_fault_test(void){
  char wide[WIDE_SOURCE];
  CHECK(!cfront_sweep(cfront_source,"hi ",1u,6u));
  CHECK(!cfront_wide_source(wide,sizeof wide));
  /* the emit's first block is 4 KiB: this one's verified C needs it grown at least three times */
  CHECK(!cfront_sweep(wide,"typedef uint32_t (*__bcir_fp63)(uint32_t, uint32_t);",8u*4096u+1u,64u));
  return 0;
}

/* a G10 report (bcir_cfront_effects / bcir_cfront_escape), or the canon, under an injected allocation
 * failure: the complete report (the bytes the fault-free run wrote) with its length, or -- when an
 * allocation it made failed -- SIZE_MAX and an empty buffer. Never a partial report, and never a short
 * one that claims to be whole. */
static int report_ok(const char *buf,size_t cap,size_t len,const char *whole,int may_fail){
  if(len==SIZE_MAX){ CHECK(may_fail); CHECK(buf[0]==0); return 0; }
  CHECK(len<cap && strlen(buf)==len && !strcmp(buf,whole));
  return 0;
}

enum { REPORT_CAP = 1024, CANON_CAP = 4096 };

static int cfront_analysis_once(const bcir_unit *unit,fault_state *state,size_t fail_at,
                                size_t *attempts,uint64_t *whole_dg,char *whole_cn,char *whole_fx,char *whole_es){
  bcir_host_allocator allocator=fault_allocator(state);
  char canon[CANON_CAP],cut[16],effects[REPORT_CAP],escape[REPORT_CAP];
  state->attempt=0;state->fail_at=fail_at;
  memset(canon,0xa5,sizeof canon);memset(cut,0xa5,sizeof cut);
  memset(effects,0xa5,sizeof effects);memset(escape,0xa5,sizeof escape);
  uint64_t dg=1;
  int dr=bcir_cfront_digest_with_allocator(unit,&allocator,&dg);   /* the digest, or no digest (CF-PPLIMITS) */
  size_t cn=bcir_cfront_canon_with_allocator(unit,canon,sizeof canon,&allocator);
  size_t mn=bcir_cfront_canon_with_allocator(unit,NULL,0,&allocator);          /* measured (CF-LIMITS) */
  size_t un=bcir_cfront_canon_with_allocator(unit,cut,sizeof cut,&allocator);  /* cut, and saying so */
  size_t fx=bcir_cfront_effects_with_allocator(unit,effects,sizeof effects,&allocator);
  size_t es=bcir_cfront_escape_with_allocator(unit,escape,sizeof escape,&allocator);
  *attempts=state->attempt;
  if(!fail_at){   /* the fault-free run: whole reports, and the reference every failure is held to */
    CHECK(cn>sizeof cut && cn<sizeof canon && fx<sizeof effects && es<sizeof escape && fx>0 && es>0);
    CHECK(dr==0); *whole_dg=dg;
    memcpy(whole_cn,canon,sizeof canon);
    memcpy(whole_fx,effects,sizeof effects);memcpy(whole_es,escape,sizeof escape);
  }
  /* the whole digest, or a refusal with none: never the hash of a canon an allocation failed in */
  CHECK(dr ? fail_at!=0 && dg==0 : dg==*whole_dg);
  CHECK(!report_ok(canon,sizeof canon,cn,whole_cn,fail_at!=0));
  CHECK(mn==SIZE_MAX ? fail_at!=0 : mn==strlen(whole_cn));
  /* a canon longer than its buffer: the buffer holds its first bytes and a NUL, the length the whole one's */
  if(un==SIZE_MAX){ CHECK(fail_at!=0); CHECK(cut[0]==0); }
  else CHECK(un==strlen(whole_cn) && strlen(cut)==sizeof cut-1u && !memcmp(cut,whole_cn,sizeof cut-1u));
  CHECK(!report_ok(effects,sizeof effects,fx,whole_fx,fail_at!=0));
  CHECK(!report_ok(escape,sizeof escape,es,whole_es,fail_at!=0));
  CHECK(state->live==0u);
  return 0;
}

static int cfront_analysis_fault_test(void){
  bcir_cfront_context context;bcir_cfront_result result;
  fault_state state={0};size_t attempts=0,baseline;uint64_t whole_dg=0;
  char whole_cn[CANON_CAP],whole_fx[REPORT_CAP],whole_es[REPORT_CAP];
  bcir_cfront_result_init(&result);CHECK(!bcir_cfront_context_init(&context,NULL));
  CHECK(!bcir_cfront_compile_context(&context,cfront_source,&result));
  CHECK(!cfront_analysis_once(&result.unit,&state,0,&attempts,&whole_dg,whole_cn,whole_fx,whole_es));
  baseline=attempts;CHECK(baseline>=3u);
  for(size_t fail=1;fail<=baseline;fail++){
    memset(&state,0,sizeof state);
    CHECK(!cfront_analysis_once(&result.unit,&state,fail,&attempts,&whole_dg,whole_cn,whole_fx,whole_es));
  }
  bcir_cfront_free(&result);bcir_cfront_context_destroy(&context);
  return 0;
}

static int q8_once(const char *path,fault_state *state,size_t fail_at,
                   size_t *attempts){
  bcir_host_allocator allocator=fault_allocator(state);
  bcir_q8_model model;
  char error[160];
  memset(&model,0xa5,sizeof model);
  state->fail_at=fail_at;state->attempt=0;
  int rc=bcir_q8_model_load_with_allocator(path,&model,error,sizeof error,&allocator);
  *attempts=state->attempt;
  if(fail_at){
    CHECK(rc!=0);CHECK(model.storage==NULL&&model.tensors==NULL);
    CHECK(model.tensor_count==0&&model.storage_size==0);
    CHECK(model.head_pack==NULL&&model.head_pack_size==0);
  }else{
    CHECK(rc==0&&model.storage&&model.tensors&&model.tensor_count>0);
    /* HEAD-PACK: the head packed at load exactly where its rows are a whole number of groups */
    CHECK((model.head_pack!=NULL&&model.head_pack_size>0)==(model.d_model%model.group_size==0));
  }
  bcir_q8_model_free(&model);
  CHECK(model.storage==NULL&&model.tensors==NULL&&model.tensor_count==0);
  CHECK(model.head_pack==NULL&&model.head_pack_size==0);
  bcir_q8_model_free(&model);
  CHECK(state->live==0u);
  return 0;
}

static int q8_and_llama_fault_test(const char *path){
  fault_state state={0};size_t attempts=0,baseline;
  bcir_q8_model invalid,model;
  bcir_host_allocator allocator=fault_allocator(&state);
  char error[160];
  memset(&invalid,0xa5,sizeof invalid);
  CHECK(bcir_q8_model_load_with_allocator(NULL,&invalid,error,sizeof error,&allocator)!=0);
  CHECK(invalid.storage==NULL&&invalid.tensors==NULL&&invalid.tensor_count==0);
  bcir_q8_model_free(&invalid);bcir_q8_model_free(&invalid);
  {
    int32_t prompt[1]={0},generated[1]={-1};
    bcir_q8_model_init(&invalid);
    CHECK(bcir_llama_generate_greedy(&invalid,prompt,1u,1u,generated,NULL)==-1);
    CHECK(generated[0]==-1);
  }
  CHECK(!q8_once(path,&state,0,&attempts));baseline=attempts;CHECK(baseline>=2u);
  for(size_t fail=1;fail<=baseline;fail++){
    memset(&state,0,sizeof state);
    CHECK(!q8_once(path,&state,fail,&attempts));
  }

  bcir_q8_model_init(&model);
  CHECK(!bcir_q8_model_load(path,&model,error,sizeof error));
  CHECK(model.vocab_size==64u);
  {
    bcir_q8_tensor foreign=model.tensors[0];
    foreign.codes=NULL;
    CHECK(isnan(bcir_q8_tensor_value(&model,&foreign,0)));
    CHECK(isnan(bcir_q8_tensor_value(&model,&model.tensors[0],
                                     model.tensors[0].element_count)));
  }
  {
    int32_t prompt[1]={1},generated[1];double logits[64];int rc;
    bcir_q8_model damaged=model;
    damaged.n_heads=0;
    generated[0]=-77;
    CHECK(bcir_llama_generate_greedy(&damaged,prompt,1u,1u,generated,NULL)==-1);
    CHECK(generated[0]==-77);
    damaged=model;damaged.bos_token_id=(int32_t)model.vocab_size;
    CHECK(bcir_llama_generate_greedy(&damaged,prompt,1u,1u,generated,NULL)==-1);
    CHECK(generated[0]==-77);
    damaged=model;damaged.context_length=0u;
    CHECK(bcir_llama_generate_greedy(&damaged,prompt,1u,1u<<20,
                                     generated,NULL)==-3);
    CHECK(generated[0]==-77);
    memset(&state,0,sizeof state);allocator=fault_allocator(&state);
    generated[0]=-77;for(size_t i=0;i<64u;i++)logits[i]=-123.0;
    rc=bcir_llama_generate_greedy_with_allocator(&model,prompt,1u,1u,
                                                  generated,logits,&allocator);
    baseline=state.attempt;
    CHECK(rc==0&&baseline>0u&&generated[0]>=0&&state.live==0u);
    for(size_t fail=1;fail<=baseline;fail++){
      memset(&state,0,sizeof state);state.fail_at=fail;allocator=fault_allocator(&state);
      generated[0]=-77;for(size_t i=0;i<64u;i++)logits[i]=-123.0;
      rc=bcir_llama_generate_greedy_with_allocator(&model,prompt,1u,1u,
                                                    generated,logits,&allocator);
      CHECK(rc==-3&&generated[0]==-77&&logits[0]==-123.0&&state.live==0u);
    }
  }
  bcir_q8_model_free(&model);bcir_q8_model_free(&model);
  return 0;
}

/* DECODE-PASSES. (1) bcir_q8_group_scale: code * scale is ldexp(code, e) bit for bit for every code in [-127, 127]
 * and every exponent in [-300, 300] the format admits, and NaN past them. (2) bcir_q8_tensor_values is
 * bcir_q8_tensor_value bit for bit over every tensor of the model -- whole, across every group's edge and to its end
 * -- and where a group's exponent or an element's code is one the loader refuses (written into the loaded storage and
 * restored: NaN, as the reference); refused whole with `out` untouched for a foreign tensor, a range past the end and
 * a NULL `out`; a zero count writes nothing. (3) bcir_llama_op_argmax picks the scalar loop's token over vocabularies
 * of 1 to 23 logits drawn from ties, -0.0 and +0.0, infinities and NaNs. */
static int same_bits(double a,double b){ return memcmp(&a,&b,sizeof a)==0; }
static int tensor_values_agree(const bcir_q8_model *m,const bcir_q8_tensor *t,uint32_t start,uint32_t count,
                               double *out){
  if(bcir_q8_tensor_values(m,t,start,count,out))return 0;
  for(uint32_t i=0;i<count;i++) if(!same_bits(out[i],bcir_q8_tensor_value(m,t,start+i)))return 0;
  return 1;
}
static uint64_t passes_state=UINT64_C(0x9E3779B97F4A7C15);
static uint64_t passes_next(void){
  passes_state^=passes_state<<13; passes_state^=passes_state>>7; passes_state^=passes_state<<17;
  return passes_state;
}
static int decode_passes_test(const char *path){
  bcir_q8_model model;char error[256];
  for(int e=-300;e<=300;e++){
    double scale=bcir_q8_group_scale(e);
    for(int code=-127;code<=127;code++) CHECK(same_bits((double)code*scale,ldexp((double)code,e)));
  }
  CHECK(isnan(bcir_q8_group_scale(-301))&&isnan(bcir_q8_group_scale(301)));
  CHECK(isnan(bcir_q8_group_scale(-32768))&&isnan(bcir_q8_group_scale(32767)));
  bcir_q8_model_init(&model);
  CHECK(!bcir_q8_model_load(path,&model,error,sizeof error));
  for(uint32_t k=0;k<model.tensor_count;k++){
    const bcir_q8_tensor *t=&model.tensors[k];
    uint32_t n=t->element_count,g=model.group_size;
    double *out=(double *)malloc((size_t)n*sizeof *out);
    CHECK(out!=NULL);
    CHECK(tensor_values_agree(&model,t,0u,n,out));
    for(uint32_t edge=g;edge<n;edge+=g){                 /* across each group's edge, and a group and one */
      CHECK(tensor_values_agree(&model,t,edge-1u,n-edge+1u<3u ? n-edge+1u : 3u,out));
      CHECK(tensor_values_agree(&model,t,edge,n-edge<g+1u ? n-edge : g+1u,out));
    }
    CHECK(tensor_values_agree(&model,t,n-1u,1u,out));
    free(out);
  }
  {
    bcir_q8_tensor *t=&model.tensors[0],foreign=model.tensors[0];
    unsigned char *e0=model.storage+t->exponent_offset,saved[2],code_saved;
    double out[4]={-1.0,-2.0,-3.0,-4.0};
    uint32_t n=t->element_count;
    CHECK(bcir_q8_tensor_values(&model,&foreign,0u,1u,out)==-1&&out[0]==-1.0);
    CHECK(bcir_q8_tensor_values(&model,t,n,1u,out)==-1&&out[0]==-1.0);
    CHECK(bcir_q8_tensor_values(&model,t,n-1u,2u,out)==-1&&out[0]==-1.0);
    CHECK(bcir_q8_tensor_values(&model,t,UINT32_MAX,1u,out)==-1&&out[0]==-1.0);
    CHECK(bcir_q8_tensor_values(&model,t,0u,1u,NULL)==-1);
    CHECK(bcir_q8_tensor_values(&model,t,0u,0u,NULL)==0);
    CHECK(bcir_q8_tensor_values(&model,t,n,0u,out)==0&&out[0]==-1.0);
    CHECK(bcir_q8_tensor_values(NULL,t,0u,1u,out)==-1&&out[0]==-1.0);
    {                                                    /* the model's own entry, its codes off its offset: */
      const unsigned char *codes=t->codes;               /* the shared predicate's spans refuse it in both */
      t->codes=codes+1;
      CHECK(isnan(bcir_q8_tensor_value(&model,t,0u)));
      CHECK(bcir_q8_tensor_values(&model,t,0u,1u,out)==-1&&out[0]==-1.0);
      t->codes=codes;
    }
    memcpy(saved,e0,2u);                                 /* an exponent past [-300, 300]: 301, little-endian */
    e0[0]=(unsigned char)(301&0xff);e0[1]=(unsigned char)(301>>8);
    CHECK(n>=4u&&tensor_values_agree(&model,t,0u,4u,out)&&isnan(out[0])&&isnan(out[3]));
    memcpy(e0,saved,2u);
    code_saved=model.storage[t->code_offset+1u];        /* the -128 code */
    model.storage[t->code_offset+1u]=0x80u;
    CHECK(tensor_values_agree(&model,t,0u,4u,out)&&isnan(out[1])&&!isnan(out[0]));
    model.storage[t->code_offset+1u]=code_saved;
    CHECK(tensor_values_agree(&model,t,0u,4u,out)&&!isnan(out[1]));
  }
  bcir_q8_model_free(&model);
  {
    static const double kinds[]={-0.0,0.0,1.0,1.0,-1.0,2.5,-2.5};
    bcir_q8_model sized;double logits[23];
    memset(&sized,0,sizeof sized);
    for(int trial=0;trial<200000;trial++){
      uint32_t n=1u+(uint32_t)(passes_next()%23u),best=0;
      for(uint32_t i=0;i<n;i++){
        uint64_t pick=passes_next()%12u;
        logits[i]=pick<7u ? kinds[pick] : pick==7u ? INFINITY : pick==8u ? -INFINITY : pick==9u ? NAN :
                  (double)(int64_t)(passes_next()%2001u)/7.0-142.0;
      }
      for(uint32_t i=1;i<n;i++) if(logits[i]>logits[best])best=i;  /* the loop the decoder ran before */
      sized.vocab_size=n;
      CHECK(bcir_llama_op_argmax(&sized,logits)==(int32_t)best);
    }
  }
  return 0;
}

static int channel_test(void){
  bcir_runtime_channel channel={0};
  bcir_runtime_channel_hooks_v1 bad={0};
  bcir_loopback_channel loopback,overwrite;
  bcir_channel_event storage[2],single[1],event;
  bcir_channel_handle session,resource,stale_session;
  bcir_channel_mapping mapping;
  bcir_channel_submission submission;
  bad.abi_version=BCIR_RUNTIME_CHANNEL_ABI_V1;bad.struct_size=1;
  CHECK(bcir_runtime_channel_init(&channel,&bad)==BCIR_CHANNEL_VERSION);
  bad.abi_version=BCIR_RUNTIME_CHANNEL_ABI_V1+1u;bad.struct_size=sizeof bad;
  CHECK(bcir_runtime_channel_init(&channel,&bad)==BCIR_CHANNEL_VERSION);
  CHECK(bcir_loopback_channel_init(&loopback,storage,2,128,
        BCIR_CHANNEL_RING_BACKPRESSURE,&channel)==BCIR_CHANNEL_OK);
  CHECK(bcir_channel_open(&channel,0,&session)==BCIR_CHANNEL_OK);
  CHECK(bcir_channel_sync(&channel,session,0,0)==BCIR_CHANNEL_STALE);
  CHECK(bcir_channel_sync(&channel,session,1,0)==BCIR_CHANNEL_WOULD_BLOCK);
  stale_session=session;
  CHECK(bcir_channel_claim(&channel,session,3,1,&resource)==BCIR_CHANNEL_INVALID);
  CHECK(bcir_channel_claim(&channel,session,3,0,&resource)==BCIR_CHANNEL_OK);
  CHECK(bcir_channel_map(&channel,session,resource,129,1,&mapping)==BCIR_CHANNEL_INVALID);
  CHECK(bcir_channel_map(&channel,session,resource,16,64,&mapping)==BCIR_CHANNEL_OK);
  memset(&submission,0,sizeof submission);submission.mapping=mapping.handle;
  submission.byte_length=8;
  submission.sequence=3;
  CHECK(bcir_channel_submit(&channel,session,&submission)==BCIR_CHANNEL_INVALID);
  submission.sequence=0;submission.flags=1;
  CHECK(bcir_channel_submit(&channel,session,&submission)==BCIR_CHANNEL_INVALID);
  submission.flags=0;submission.reserved=1;
  CHECK(bcir_channel_submit(&channel,session,&submission)==BCIR_CHANNEL_INVALID);
  submission.reserved=0;
  CHECK(bcir_channel_submit(&channel,session,&submission)==BCIR_CHANNEL_OK);
  CHECK(bcir_channel_sync(&channel,session,1,BCIR_CHANNEL_SYNC_CANCEL)==BCIR_CHANNEL_STALE);
  CHECK(bcir_channel_sync(&channel,session,1,
        BCIR_CHANNEL_SYNC_WAIT|BCIR_CHANNEL_SYNC_CANCEL)==BCIR_CHANNEL_INVALID);
  CHECK(bcir_channel_sync(&channel,session,2,8u)==BCIR_CHANNEL_INVALID);
  submission.byte_offset=63;submission.byte_length=2;
  CHECK(bcir_channel_submit(&channel,session,&submission)==BCIR_CHANNEL_INVALID);
  submission.byte_offset=0;submission.byte_length=8;
  CHECK(bcir_channel_submit(&channel,session,&submission)==BCIR_CHANNEL_OK);
  CHECK(bcir_channel_submit(&channel,session,&submission)==BCIR_CHANNEL_BACKPRESSURE);
  CHECK(bcir_channel_next_event(&channel,session,&event)==BCIR_CHANNEL_OK&&event.sequence==1);
  CHECK(bcir_channel_submit(&channel,session,&submission)==BCIR_CHANNEL_OK);
  CHECK(bcir_channel_sync(&channel,session,3,BCIR_CHANNEL_SYNC_WAIT)==BCIR_CHANNEL_OK);
  CHECK(bcir_channel_close(&channel,session)==BCIR_CHANNEL_OK);
  CHECK(bcir_channel_close(&channel,session)==BCIR_CHANNEL_CLOSED);
  CHECK(bcir_channel_claim(&channel,stale_session,1,0,&resource)==BCIR_CHANNEL_CLOSED);
  CHECK(bcir_channel_open(&channel,0,&session)==BCIR_CHANNEL_OK);
  CHECK(bcir_channel_submit(&channel,session,&submission)==BCIR_CHANNEL_STALE);
  CHECK(bcir_channel_close(&channel,session)==BCIR_CHANNEL_OK);

  CHECK(bcir_loopback_channel_init(&overwrite,single,1,32,
        BCIR_CHANNEL_RING_OVERWRITE_OLDEST,&channel)==BCIR_CHANNEL_OK);
  CHECK(bcir_channel_open(&channel,0,&session)==BCIR_CHANNEL_OK);
  CHECK(bcir_channel_claim(&channel,session,0,0,&resource)==BCIR_CHANNEL_OK);
  CHECK(bcir_channel_map(&channel,session,resource,0,32,&mapping)==BCIR_CHANNEL_OK);
  memset(&submission,0,sizeof submission);submission.mapping=mapping.handle;submission.byte_length=1;
  CHECK(bcir_channel_submit(&channel,session,&submission)==BCIR_CHANNEL_OK);
  CHECK(bcir_channel_submit(&channel,session,&submission)==BCIR_CHANNEL_OK);
  CHECK(overwrite.dropped_events==1);
  CHECK(bcir_channel_next_event(&channel,session,&event)==BCIR_CHANNEL_OK&&event.sequence==2);
  CHECK(bcir_channel_next_event(&channel,session,&event)==BCIR_CHANNEL_WOULD_BLOCK);
  overwrite.dropped_events=UINT64_MAX;
  CHECK(bcir_channel_submit(&channel,session,&submission)==BCIR_CHANNEL_OK);
  CHECK(bcir_channel_submit(&channel,session,&submission)==BCIR_CHANNEL_OK);
  CHECK(overwrite.dropped_events==UINT64_MAX);
  CHECK(bcir_channel_close(&channel,session)==BCIR_CHANNEL_OK);

  /* A borrowed hook table becoming incomplete fails closed instead of calling NULL. */
  {
    bcir_channel_submit_fn saved=overwrite.hooks.submit;
    overwrite.hooks.submit=NULL;
    CHECK(bcir_channel_submit(&channel,session,&submission)==BCIR_CHANNEL_INVALID);
    overwrite.hooks.submit=saved;
  }
  overwrite.generation=UINT32_MAX;
  CHECK(bcir_channel_open(&channel,0,&session)==BCIR_CHANNEL_INVALID);
  bcir_runtime_channel_reset(&channel);bcir_runtime_channel_reset(&channel);
  return 0;
}

int main(int argc,char **argv){
  CHECK(argc==2&&argv[1]&&argv[1][0]);
  CHECK(!checked_growth_test());
  CHECK(!cfront_fault_test());
  CHECK(!cfront_analysis_fault_test());
  if(!strcmp(argv[1],"--cfront")){      /* the C frontend's sweeps alone -- no model file (test_c_cfront.py) */
    puts("memory-discipline: cfront ok");
    return 0;
  }
  CHECK(!cpp_fault_and_isolation_test());
  CHECK(!q8_and_llama_fault_test(argv[1]));
  CHECK(!decode_passes_test(argv[1]));
  CHECK(!channel_test());
  puts("memory-discipline: ok");
  return 0;
}
