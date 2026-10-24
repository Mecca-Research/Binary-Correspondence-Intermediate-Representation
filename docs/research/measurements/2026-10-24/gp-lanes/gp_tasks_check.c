/* GP-LANES' tasks: a fork's geometric-head entries from two builds -- A (once_level3.py's) and B (gp_lanes_overlay.py's)
 * -- each loaded on its own (RTLD_LOCAL), called on the same inputs, every output compared byte for byte: the prefix
 * cache and the features (gp_wedge_forward), both adjoint blocks (gp_wedge_backward) and, with QUERY, the query
 * contraction and its fused backward (gp_query_contract_forward, gp_query_wedge_backward). Row counts from 1 to 1,024
 * (15, 17, 66, 111 leave rows past the last group of sixteen), serially and on a four-thread pool; every batch's first
 * row has a zero prefix and some rows are near parallel to theirs, so the FP64 fallback runs among the lanes.
 * usage: gp_tasks_check LIB_A LIB_B QUERY(0|1)   (GP_RANK 22, GP_PROJECTOR 0: the study's builds) */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define R 22
#define P ((R*(R-1))/2)
typedef struct bcir_tensor_pool bcir_tensor_pool;
typedef struct lib {
  size_t (*psize)(unsigned); bcir_tensor_pool *(*pinit)(void *,size_t,unsigned); void (*pstop)(bcir_tensor_pool *);
  void (*fwd)(bcir_tensor_pool *,size_t,size_t,const float *,float *);
  void (*bwd)(bcir_tensor_pool *,size_t,size_t,const float *,const float *,float *);
  void (*qfwd)(bcir_tensor_pool *,size_t,const float *,const float *,float *);
  void (*qbwd)(bcir_tensor_pool *,size_t,size_t,const float *,const float *,const float *,float *);
} lib;
static void *sym(void *h,const char *n) { void *f=dlsym(h,n); if (!f) { fprintf(stderr,"missing %s\n",n); exit(2); } return f; }
static lib open_lib(const char *path,int query) {
  lib l; void *h=dlopen(path,RTLD_NOW|RTLD_LOCAL); if (!h) { fprintf(stderr,"dlopen %s: %s\n",path,dlerror()); exit(2); }
  *(void **)&l.psize=sym(h,"bcir_tensor_pool_size"); *(void **)&l.pinit=sym(h,"bcir_tensor_pool_init");
  *(void **)&l.pstop=sym(h,"bcir_tensor_pool_stop"); *(void **)&l.fwd=sym(h,"gp_wedge_forward");
  *(void **)&l.bwd=sym(h,"gp_wedge_backward");
  l.qfwd=NULL; l.qbwd=NULL;
  if (query) { *(void **)&l.qfwd=sym(h,"gp_query_contract_forward"); *(void **)&l.qbwd=sym(h,"gp_query_wedge_backward"); }
  return l;
}
static uint64_t rs=0x2545F4914F6CDD1Dull;
static float unif(void) { rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (float)((double)(rs>>11)/9007199254740992.0*2-1); }
static bcir_tensor_pool *pool_of(const lib *l,unsigned threads,void **mem) {
  size_t b; *mem=NULL; if (threads<2) return NULL;
  b=l->psize(threads); if (posix_memalign(mem,64,(b+63)/64*64)) exit(3);
  return l->pinit(*mem,b,threads);
}
int main(int argc,char **argv) {
  static const size_t shapes[][2]={{1,1},{1,15},{1,16},{1,17},{2,33},{3,37},{4,128},{4,256}};
  size_t s,i,k,t,b,n,checks=0,bad=0; int query; unsigned th,ti; lib A,B;
  if (argc!=4) return 2;
  query=atoi(argv[3]); A=open_lib(argv[1],query); B=open_lib(argv[2],query);
  for (ti=0;ti<2;ti++) for (s=0;s<sizeof shapes/sizeof shapes[0];s++) {
    void *ma,*mb; bcir_tensor_pool *pa,*pb;
    th=ti ? 4u : 1u; pa=pool_of(&A,th,&ma); pb=pool_of(&B,th,&mb);
    b=shapes[s][0]; t=shapes[s][1]; n=b*t;
    float *za=calloc(2*n*R,4),*zb=calloc(2*n*R,4),*pa_=calloc(n*P,4),*pb_=calloc(n*P,4),*dp=malloc(n*P*4),*dc=malloc(n*R*4);
    float *dza=malloc(2*n*R*4),*dzb=malloc(2*n*R*4),*ca=calloc(n*R,4),*cb=calloc(n*R,4);
    for (i=0;i<n;i++) {  /* u rows; every seventh row near parallel to its prefix */
      size_t tt=i%t,r0=i-tt;
      for (k=0;k<R;k++) za[i*R+k]=unif();
      if (tt>0 && tt%7==3) for (k=0;k<R;k++) {
        float sum=0; size_t q; for (q=r0;q<i;q++) sum+=za[q*R+k];
        za[i*R+k]=0.5f*sum+1e-6f*unif();
      }
    }
    for (i=0;i<n*P;i++) dp[i]=unif();
    for (i=0;i<n*R;i++) dc[i]=unif();
    for (i=0;i<2*n*R;i++) dza[i]=dzb[i]=unif();
    memcpy(zb,za,2*n*R*4);
    A.fwd(pa,b,t,za,pa_); B.fwd(pb,b,t,zb,pb_);
    checks+=2; bad+=memcmp(za,zb,2*n*R*4)!=0; bad+=memcmp(pa_,pb_,n*P*4)!=0;
    A.bwd(pa,b,t,za,dp,dza); B.bwd(pb,b,t,zb,dp,dzb);
    checks++; bad+=memcmp(dza,dzb,2*n*R*4)!=0;
    if (query) {
      A.qfwd(pa,n,za,pa_,ca); B.qfwd(pb,n,zb,pb_,cb);
      for (i=0;i<2*n*R;i++) dza[i]=dzb[i]=unif();
      A.qbwd(pa,b,t,za,dp,dc,dza); B.qbwd(pb,b,t,zb,dp,dc,dzb);
      checks+=2; bad+=memcmp(ca,cb,n*R*4)!=0; bad+=memcmp(dza,dzb,2*n*R*4)!=0;
    }
    free(za); free(zb); free(pa_); free(pb_); free(dp); free(dc); free(dza); free(dzb); free(ca); free(cb);
    if (pa) A.pstop(pa);
    if (pb) B.pstop(pb);
    free(ma); free(mb);
  }
  printf("{\"query\": %d, \"shapes\": %zu, \"pools\": [1, 4], \"outputs_compared\": %zu, \"mismatched_outputs\": %zu}\n",
         query,sizeof shapes/sizeof shapes[0],checks,bad);
  return bad ? 1 : 0;
}
