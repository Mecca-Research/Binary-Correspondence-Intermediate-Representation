#include "bcir_decoder_train.h"
#include <math.h>
#include <string.h>
#include "global_core.h"
/* Feature mixer only. z already contains the externally computed rotary
 * features. Decoder/FF/optimizer streaming is not implemented by this probe.
 * Caller initializes sums and steps to zero once, then retains them per stream.
 */
void gp_stream_forward(size_t batch,size_t time,const float *z,float *p,double *sums,uint64_t *steps) {
  size_t b,t,k;
  for(b=0;b<batch;b++)for(t=0;t<time;t++) {
    float v[GP_RANK];double scale=steps[b]?1.0/sqrt((double)steps[b]):0;
    const float *u=z+(b*time+t)*GP_RANK;
    for(k=0;k<GP_RANK;k++)v[k]=(float)(sums[b*GP_RANK+k]*scale);
    gp_pair_forward(u,v,p+(b*time+t)*GP_P);
    for(k=0;k<GP_RANK;k++)sums[b*GP_RANK+k]+=u[k];
    steps[b]++;
  }
}
