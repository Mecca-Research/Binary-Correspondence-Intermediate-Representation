/* GP-LANES-0: the lane kernels (gp_lanes.inc) against the study fork's scalar kernels (gp_scalar.inc,
 * extracted verbatim), every output byte for byte, over ROWS rows of each kind (uniform; u and v
 * near parallel, so the FP64 fallback; tiny; zero; large; sparse), then each kernel timed both
 * ways on uniform rows. usage: gp_lanes_check ROWS REPS */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef GP_RANK
#define GP_RANK 22
#define GP_PROJECTOR 0
#define GP_SYMMETRIC 0
#endif
#include "gp_scalar.inc"
#include "gp_lanes_pair.inc"
#include "gp_lanes_query.inc"
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static uint64_t rng=0x9e3779b97f4a7c15ull;
static double unif(void) { rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (double)(rng>>11)/9007199254740992.0; }
static void row(int kind,float *u,float *v) {
  size_t i; double a=unif()*2-1;
  for(i=0;i<GP_RANK;i++){
    double x=unif()*2-1,y=unif()*2-1;
    switch(kind){
      case 0: u[i]=(float)x; v[i]=(float)y; break;
      case 1: u[i]=(float)x; v[i]=(float)(a*x+1e-5*y); break;           /* near parallel */
      case 2: u[i]=(float)(x*1e-20); v[i]=(float)(y*1e-20); break;      /* tiny */
      case 3: u[i]=0; v[i]=(i%3)?(float)y:0; break;                      /* zero u */
      case 4: u[i]=(float)(x*1e15); v[i]=(float)(y*1e15); break;        /* large */
      default: u[i]=(i%5==0)?(float)x:0; v[i]=(i%7==1)?(float)y:0; break; /* sparse */
    }
  }
}
int main(int argc,char **argv) {
  size_t rows=argc>1?strtoul(argv[1],0,10):4096,reps=argc>2?strtoul(argv[2],0,10):20,n,r,g,i,c,l,kind,bad=0,ills=0;
  n=rows*6/GPL*GPL;
  float *u=malloc(n*GP_RANK*4),*v=malloc(n*GP_RANK*4),*dp=malloc(n*GP_P*4),*dc=malloc(n*GP_RANK*4);
  float *p1=malloc(n*GP_P*4),*p2=malloc(n*GP_P*4),*du1=malloc(n*GP_RANK*4),*dv1=malloc(n*GP_RANK*4),*du2=malloc(n*GP_RANK*4),*dv2=malloc(n*GP_RANK*4);
  float *c1=malloc(n*GP_RANK*4),*c2=malloc(n*GP_RANK*4),*q1=malloc(n*GP_P*4),*q2=malloc(n*GP_P*4),*e1=malloc(n*GP_RANK*4),*e2=malloc(n*GP_RANK*4);
  if(!__builtin_cpu_supports("avx512f")){printf("no AVX-512F\n");return 77;}
  for(r=0;r<n;r++){kind=(r/GPL)%6; row((int)kind,u+r*GP_RANK,v+r*GP_RANK);
    for(c=0;c<GP_P;c++)dp[r*GP_P+c]=(float)(unif()*2-1);
    for(i=0;i<GP_RANK;i++)dc[r*GP_RANK+i]=(float)(unif()*2-1);}
  /* the scalar kernels, row by row */
  for(r=0;r<n;r++){
    gp_pair_forward(u+r*GP_RANK,v+r*GP_RANK,p1+r*GP_P);
    gp_pair_backward(u+r*GP_RANK,v+r*GP_RANK,dp+r*GP_P,du1+r*GP_RANK,dv1+r*GP_RANK);
    gp_query_contract_pair(u+r*GP_RANK,p1+r*GP_P,c1+r*GP_RANK);
    memcpy(q1+r*GP_P,dp+r*GP_P,GP_P*4);
    gp_query_contract_vjp(u+r*GP_RANK,p1+r*GP_P,dc+r*GP_RANK,q1+r*GP_P,e1+r*GP_RANK);
  }
  /* the lanes, sixteen rows at a time; flagged rows by the scalar kernel */
  for(g=0;g<n;g+=GPL){
    _Alignas(64) float U[GP_RANK][GPL],V[GP_RANK][GPL],O[GP_P][GPL],DP[GP_P][GPL],DU[GP_RANK][GPL],DV[GP_RANK][GPL],C[GP_RANK][GPL],DC[GP_RANK][GPL],E[GP_RANK][GPL];
    unsigned char ill[GPL],ill2[GPL];
    gpl_in(u+g*GP_RANK,GP_RANK,U); gpl_in(v+g*GP_RANK,GP_RANK,V);
    gp_pair_forward_lanes(U,V,O,ill); gpl_out(O,GP_P,p2+g*GP_P,ill);
    for(l=0;l<GPL;l++)if(ill[l]){gp_pair_forward(u+(g+l)*GP_RANK,v+(g+l)*GP_RANK,p2+(g+l)*GP_P);ills++;}
    gpl_in(dp+g*GP_P,GP_P,DP);
    gp_pair_backward_lanes(U,V,DP,DU,DV,ill2); gpl_out(DU,GP_RANK,du2+g*GP_RANK,ill2); gpl_out(DV,GP_RANK,dv2+g*GP_RANK,ill2);
    for(l=0;l<GPL;l++){if(ill2[l]!=ill[l])bad++; if(ill2[l])gp_pair_backward(u+(g+l)*GP_RANK,v+(g+l)*GP_RANK,dp+(g+l)*GP_P,du2+(g+l)*GP_RANK,dv2+(g+l)*GP_RANK);}
    gpl_in(p2+g*GP_P,GP_P,O);  /* the query kernels read the stored features, the fallback's included */
    gp_query_contract_pair_lanes(U,O,C); gpl_out(C,GP_RANK,c2+g*GP_RANK,NULL);
    gpl_in(dc+g*GP_RANK,GP_RANK,DC);
    gp_query_contract_vjp_lanes(U,O,DC,DP,E); gpl_out(DP,GP_P,q2+g*GP_P,NULL); gpl_out(E,GP_RANK,e2+g*GP_RANK,NULL);
  }
  bad+=memcmp(p1,p2,n*GP_P*4)!=0; bad+=memcmp(du1,du2,n*GP_RANK*4)!=0; bad+=memcmp(dv1,dv2,n*GP_RANK*4)!=0;
  bad+=memcmp(c1,c2,n*GP_RANK*4)!=0; bad+=memcmp(q1,q2,n*GP_P*4)!=0; bad+=memcmp(e1,e2,n*GP_RANK*4)!=0;
  printf("{\"rows\": %zu, \"fallback_rows\": %zu, \"mismatched_outputs\": %zu",n,ills,bad);
  /* timing on the uniform rows: kind 0 every row */
  for(r=0;r<n;r++)row(0,u+r*GP_RANK,v+r*GP_RANK);
  {
    double t0,ts[4],tl[4]; size_t rep;
    t0=now(); for(rep=0;rep<reps;rep++)for(r=0;r<n;r++)gp_pair_forward(u+r*GP_RANK,v+r*GP_RANK,p1+r*GP_P); ts[0]=now()-t0;
    t0=now(); for(rep=0;rep<reps;rep++)for(r=0;r<n;r++)gp_pair_backward(u+r*GP_RANK,v+r*GP_RANK,dp+r*GP_P,du1+r*GP_RANK,dv1+r*GP_RANK); ts[1]=now()-t0;
    t0=now(); for(rep=0;rep<reps;rep++)for(r=0;r<n;r++)gp_query_contract_pair(u+r*GP_RANK,p1+r*GP_P,c1+r*GP_RANK); ts[2]=now()-t0;
    t0=now(); for(rep=0;rep<reps;rep++)for(r=0;r<n;r++)gp_query_contract_vjp(u+r*GP_RANK,p1+r*GP_P,dc+r*GP_RANK,q1+r*GP_P,e1+r*GP_RANK); ts[3]=now()-t0;
    for(i=0;i<4;i++)tl[i]=0;
    for(rep=0;rep<reps;rep++)for(g=0;g<n;g+=GPL){
      _Alignas(64) float U[GP_RANK][GPL],V[GP_RANK][GPL],O[GP_P][GPL],DP[GP_P][GPL],DU[GP_RANK][GPL],DV[GP_RANK][GPL],C[GP_RANK][GPL],DC[GP_RANK][GPL],E[GP_RANK][GPL];
      unsigned char ill[GPL];
      t0=now(); gpl_in(u+g*GP_RANK,GP_RANK,U); gpl_in(v+g*GP_RANK,GP_RANK,V); gp_pair_forward_lanes(U,V,O,ill); gpl_out(O,GP_P,p2+g*GP_P,ill); tl[0]+=now()-t0;
      t0=now(); gpl_in(u+g*GP_RANK,GP_RANK,U); gpl_in(v+g*GP_RANK,GP_RANK,V); gpl_in(dp+g*GP_P,GP_P,DP); gp_pair_backward_lanes(U,V,DP,DU,DV,ill); gpl_out(DU,GP_RANK,du2+g*GP_RANK,ill); gpl_out(DV,GP_RANK,dv2+g*GP_RANK,ill); tl[1]+=now()-t0;
      t0=now(); gpl_in(u+g*GP_RANK,GP_RANK,U); gpl_in(p1+g*GP_P,GP_P,O); gp_query_contract_pair_lanes(U,O,C); gpl_out(C,GP_RANK,c2+g*GP_RANK,NULL); tl[2]+=now()-t0;
      t0=now(); gpl_in(u+g*GP_RANK,GP_RANK,U); gpl_in(p1+g*GP_P,GP_P,O); gpl_in(dc+g*GP_RANK,GP_RANK,DC); gpl_in(q1+g*GP_P,GP_P,DP); gp_query_contract_vjp_lanes(U,O,DC,DP,E); gpl_out(DP,GP_P,q2+g*GP_P,NULL); gpl_out(E,GP_RANK,e2+g*GP_RANK,NULL); tl[3]+=now()-t0;
    }
    printf(", \"ns_a_row\": {\"scalar\": [%.1f, %.1f, %.1f, %.1f], \"lanes\": [%.1f, %.1f, %.1f, %.1f]}",
           ts[0]*1e9/(reps*n),ts[1]*1e9/(reps*n),ts[2]*1e9/(reps*n),ts[3]*1e9/(reps*n),tl[0]*1e9/(reps*n),tl[1]*1e9/(reps*n),tl[2]*1e9/(reps*n),tl[3]*1e9/(reps*n));
  }
  printf("}\n");
  return bad?1:0;
}
