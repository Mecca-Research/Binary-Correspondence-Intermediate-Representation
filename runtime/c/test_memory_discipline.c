/*===- test_memory_discipline.c - allocator/context/channel regressions -----===*/
#include "bcir_cfront.h"
#include "bcir_cpp.h"
#include "bcir_host_alloc.h"
#include "bcir_llama.h"
#include "bcir_q8_model.h"
#include "bcir_runtime.h"
#include "bcir_runtime_channel.h"

#include <math.h>
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
      size_t w=(size_t)snprintf(paste,sizeof paste,"#define CAT(a,b) a ## b\nCAT(x,");
      for(size_t k=0;k<200u;k++)w+=(size_t)snprintf(paste+w,sizeof paste-w,"y+");
      (void)snprintf(paste+w,sizeof paste-w,"z)\n");
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

static int cfront_once(fault_state *state,size_t fail_at,size_t *attempts){
  bcir_host_allocator allocator=fault_allocator(state);
  bcir_cfront_context context;
  bcir_cfront_result result;
  state->fail_at=0;state->attempt=0;
  bcir_cfront_result_init(&result);
  CHECK(!bcir_cfront_context_init(&context,&allocator));
  state->attempt=0;state->fail_at=fail_at;
  int rc=bcir_cfront_compile_context(&context,cfront_source,&result);
  *attempts=state->attempt;
  if(fail_at){
    CHECK(rc!=0);CHECK(result.unit.funcs==NULL);CHECK(result.unit.n_funcs==0);
    CHECK(!result.emitted_ok&&result.emitted[0]==0);
  }else{
    CHECK(rc==0&&result.ok&&result.emitted_ok);
    CHECK(result.unit.n_funcs==1&&strstr(result.emitted,"hi ")!=NULL);
  }
  bcir_cfront_free(&result);
  CHECK(result.unit.funcs==NULL&&result.unit.n_funcs==0&&result.ok==0);
  CHECK(result.emitted_ok==0&&result.emitted[0]==0&&result.diag[0]==0);
  bcir_cfront_free(&result);
  bcir_cfront_context_destroy(&context);bcir_cfront_context_destroy(&context);
  CHECK(state->live==0u);
  return 0;
}

static int cfront_fault_test(void){
  fault_state state={0};size_t attempts=0,baseline;
  CHECK(!cfront_once(&state,0,&attempts));
  baseline=attempts;CHECK(baseline>5u);
  for(size_t fail=1;fail<=baseline;fail++){
    memset(&state,0,sizeof state);
    CHECK(!cfront_once(&state,fail,&attempts));
  }
  return 0;
}

static int cfront_analysis_once(const bcir_unit *unit,fault_state *state,
                                size_t fail_at,size_t *attempts){
  bcir_host_allocator allocator=fault_allocator(state);
  char canon[4096],effects[1024];
  state->attempt=0;state->fail_at=fail_at;
  memset(canon,0xa5,sizeof canon);memset(effects,0xa5,sizeof effects);
  (void)bcir_cfront_digest_with_allocator(unit,&allocator);
  bcir_cfront_canon_with_allocator(unit,canon,sizeof canon,&allocator);
  bcir_cfront_effects_with_allocator(unit,effects,sizeof effects,&allocator);
  *attempts=state->attempt;
  CHECK(memchr(canon,0,sizeof canon)!=NULL);
  CHECK(memchr(effects,0,sizeof effects)!=NULL);
  CHECK(state->live==0u);
  return 0;
}

static int cfront_analysis_fault_test(void){
  bcir_cfront_context context;bcir_cfront_result result;
  fault_state state={0};size_t attempts=0,baseline;
  bcir_cfront_result_init(&result);CHECK(!bcir_cfront_context_init(&context,NULL));
  CHECK(!bcir_cfront_compile_context(&context,cfront_source,&result));
  CHECK(!cfront_analysis_once(&result.unit,&state,0,&attempts));
  baseline=attempts;CHECK(baseline>=3u);
  for(size_t fail=1;fail<=baseline;fail++){
    memset(&state,0,sizeof state);
    CHECK(!cfront_analysis_once(&result.unit,&state,fail,&attempts));
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
  }else CHECK(rc==0&&model.storage&&model.tensors&&model.tensor_count>0);
  bcir_q8_model_free(&model);
  CHECK(model.storage==NULL&&model.tensors==NULL&&model.tensor_count==0);
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
  CHECK(!cpp_fault_and_isolation_test());
  CHECK(!cfront_fault_test());
  CHECK(!cfront_analysis_fault_test());
  CHECK(!q8_and_llama_fault_test(argv[1]));
  CHECK(!channel_test());
  puts("memory-discipline: ok");
  return 0;
}
