/* the lane forward's parts, each on one L1-hot group of 16 rows, TSC cycles a row */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
#define GP_RANK 22
#define GP_PROJECTOR 0
#define GP_SYMMETRIC 0
#include "gp_scalar.inc"
#include "gp_lanes_pair.inc"
#include "gp_lanes_query.inc"
int main(void) {
  static float u[16*GP_RANK],v[16*GP_RANK],p[16*GP_P]; _Alignas(64) float U[GP_RANK][GPL],V[GP_RANK][GPL],O[GP_P][GPL]; unsigned char ill[GPL];
  size_t i,r,N=20000; unsigned long long t0,t; volatile float sink=0;
  for(i=0;i<16*GP_RANK;i++){u[i]=(float)((i*37%101)/50.0-1);v[i]=(float)((i*53%103)/51.0-1);}
  t0=__rdtsc(); for(r=0;r<N;r++){gpl_in(u,GP_RANK,U);gpl_in(v,GP_RANK,V);sink+=U[3][r&15];} t=__rdtsc()-t0; printf("in u,v      %6.1f cycles a row\n",(double)t/N/16);
  t0=__rdtsc(); for(r=0;r<N;r++){gpl_ill(U,V,ill);sink+=ill[r&15];} t=__rdtsc()-t0; printf("ill         %6.1f\n",(double)t/N/16);
  t0=__rdtsc(); for(r=0;r<N;r++){gp_pair_forward_lanes(U,V,O,ill);sink+=O[5][r&15];} t=__rdtsc()-t0; printf("forward     %6.1f (ill included)\n",(double)t/N/16);
  t0=__rdtsc(); for(r=0;r<N;r++){gpl_out(O,GP_P,p,NULL);sink+=p[r&255];} t=__rdtsc()-t0; printf("out p       %6.1f\n",(double)t/N/16);
  t0=__rdtsc(); for(r=0;r<N/16;r++){for(i=0;i<16;i++)gp_pair_forward(u+i*GP_RANK,v+i*GP_RANK,p+i*GP_P);sink+=p[r&255];} t=__rdtsc()-t0; printf("scalar fwd  %6.1f\n",(double)t/(N/16)/16);
  printf("(TSC cycles; sink %g)\n",(double)sink);
  return 0;
}
