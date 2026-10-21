/* Experimental causal prefix geometry. Caller owns all arrays. No allocation.
 * z[0:N*R] is projected/rotated input; z[N*R:2*N*R] is prefix cache.
 * dz uses the same two-block layout, with prefix adjoints in its second block.
 * Prefix scan and reverse scan use fixed order, double accumulators.
 */
#if GP_PROJECTOR
#define GP_P ((GP_RANK*(GP_RANK+1))/2)
#else
#define GP_P ((GP_RANK*(GP_RANK-1))/2)
#endif
#define GP_RIDGE 1e-4
typedef struct gp_job {size_t batch,time;const float *z,*dp;float *p,*dz;} gp_job;

static void gp_prefix_task(void *ctx,size_t batch) {
  gp_job *j=ctx; size_t t,k,n=j->batch*j->time;
  const float *z=j->z+batch*j->time*GP_RANK;
  float *prefix=(float *)j->z+n*GP_RANK+batch*j->time*GP_RANK;
  double sum[GP_RANK]={0};
  for(t=0;t<j->time;t++) {
    double scale=t?1.0/sqrt((double)t):0;
    for(k=0;k<GP_RANK;k++) {
      prefix[t*GP_RANK+k]=(float)(sum[k]*scale);
      sum[k]+=(double)z[t*GP_RANK+k];
    }
  }
}
static void gp_pair_forward(const float *u,const float *v,float *out) {
  size_t i,k,c; double aa=GP_RIDGE,bb=0,dd=GP_RIDGE;
#if GP_PROJECTOR
  double inv,a,b,d;
  for(i=0;i<GP_RANK;i++){aa+=(double)u[i]*u[i];bb+=(double)u[i]*v[i];dd+=(double)v[i]*v[i];}
  inv=1.0/(aa*dd-bb*bb);a=dd*inv;b=-bb*inv;d=aa*inv;
  for(i=0,c=0;i<GP_RANK;i++) for(k=i;k<GP_RANK;k++,c++)
    out[c]=(float)((u[i]*(a*u[k]+b*v[k])+v[i]*(b*u[k]+d*v[k]))*0.7071067811865475244);
#else
  double norm=0,inv;
  (void)aa;(void)bb;(void)dd;
  for(i=0,c=0;i<GP_RANK;i++) for(k=i+1;k<GP_RANK;k++,c++) {
#if GP_SYMMETRIC
    out[c]=u[i]*v[k]+u[k]*v[i];
#else
    out[c]=u[i]*v[k]-u[k]*v[i];
#endif
    norm+=(double)out[c]*out[c];
  }
  norm=sqrt(norm);inv=1.0/(norm<1e-6?1e-6:norm);
  for(c=0;c<GP_P;c++)out[c]=(float)((double)out[c]*inv);
#endif
}
static void gp_pair_backward(const float *u,const float *v,const float *dp,float *du,float *dv) {
  size_t i,k,c;
  for(i=0;i<GP_RANK;i++){du[i]=0;dv[i]=0;}
#if GP_PROJECTOR
  double aa=GP_RIDGE,bb=0,dd=GP_RIDGE,inv,a,b,d;
  double yu[GP_RANK]={0},yv[GP_RANK]={0},d00=0,d01=0,d11=0;
  for(i=0;i<GP_RANK;i++){aa+=(double)u[i]*u[i];bb+=(double)u[i]*v[i];dd+=(double)v[i]*v[i];}
  inv=1.0/(aa*dd-bb*bb);a=dd*inv;b=-bb*inv;d=aa*inv;
  /* S is the symmetric matrix adjoint: packed off-diagonals are halved. */
  for(i=0,c=0;i<GP_RANK;i++) for(k=i;k<GP_RANK;k++,c++) {
    double q=dp[c]*0.7071067811865475244;
    if(i==k){yu[i]+=q*u[i];yv[i]+=q*v[i];}
    else {q*=0.5;yu[i]+=q*u[k];yu[k]+=q*u[i];yv[i]+=q*v[k];yv[k]+=q*v[i];}
  }
  for(i=0;i<GP_RANK;i++){d00+=u[i]*yu[i];d01+=u[i]*yv[i];d11+=v[i]*yv[i];}
  {
    double e00=a*a*d00+2*a*b*d01+b*b*d11;
    double e01=a*b*d00+(a*d+b*b)*d01+b*d*d11;
    double e11=b*b*d00+2*b*d*d01+d*d*d11;
    for(i=0;i<GP_RANK;i++) {
      du[i]=(float)(2*(a*yu[i]+b*yv[i]-u[i]*e00-v[i]*e01));
      dv[i]=(float)(2*(b*yu[i]+d*yv[i]-u[i]*e01-v[i]*e11));
    }
  }
#else
  float p[GP_P];double norm=0,dot=0,inv,scale,correction;int clamped;
  for(i=0,c=0;i<GP_RANK;i++) for(k=i+1;k<GP_RANK;k++,c++) {
#if GP_SYMMETRIC
    p[c]=u[i]*v[k]+u[k]*v[i];
#else
    p[c]=u[i]*v[k]-u[k]*v[i];
#endif
    norm+=(double)p[c]*p[c];dot+=(double)p[c]*dp[c];
  }
  norm=sqrt(norm);clamped=norm<=1e-6;inv=1.0/(clamped?1e-6:norm);
  scale=inv;correction=clamped?0:dot*inv*inv;
  for(i=0,c=0;i<GP_RANK;i++) for(k=i+1;k<GP_RANK;k++,c++) {
    float q=(float)(((double)dp[c]-p[c]*correction)*scale);
    du[i]+=q*v[k];dv[k]+=q*u[i];
#if GP_SYMMETRIC
    du[k]+=q*v[i];dv[i]+=q*u[k];
#else
    du[k]-=q*v[i];dv[i]-=q*u[k];
#endif
  }
#endif
}
static void gp_forward_task(void *ctx,size_t row) {
  gp_job *j=ctx;size_t n=j->batch*j->time;
  gp_pair_forward(j->z+row*GP_RANK,j->z+(n+row)*GP_RANK,j->p+row*GP_P);
}
static void gp_backward_task(void *ctx,size_t row) {
  gp_job *j=ctx;size_t n=j->batch*j->time;
  gp_pair_backward(j->z+row*GP_RANK,j->z+(n+row)*GP_RANK,j->dp+row*GP_P,
      j->dz+row*GP_RANK,j->dz+(n+row)*GP_RANK);
}
static void gp_reverse_task(void *ctx,size_t batch) {
  gp_job *j=ctx;size_t n=j->batch*j->time,t,k;
  float *dz=j->dz+batch*j->time*GP_RANK;
  const float *dv=j->dz+n*GP_RANK+batch*j->time*GP_RANK;
  double carry[GP_RANK]={0};
  for(t=j->time;t-->0;) {
    double scale=t?1.0/sqrt((double)t):0;
    for(k=0;k<GP_RANK;k++) {
      dz[t*GP_RANK+k]+=(float)carry[k];
      carry[k]+=(double)dv[t*GP_RANK+k]*scale;
    }
  }
}
void gp_wedge_forward(bcir_tensor_pool *pool,size_t batch,size_t time,const float *z,float *p) {
  gp_job j={batch,time,z,NULL,p,NULL};
  bcir_tensor_pool_run(pool,gp_prefix_task,&j,batch);
  bcir_tensor_pool_run(pool,gp_forward_task,&j,batch*time);
}
void gp_wedge_backward(bcir_tensor_pool *pool,size_t batch,size_t time,const float *z,const float *dp,float *dz) {
  gp_job j={batch,time,z,dp,NULL,dz};
  bcir_tensor_pool_run(pool,gp_backward_task,&j,batch*time);
  bcir_tensor_pool_run(pool,gp_reverse_task,&j,batch);
}
/* Deliberately direct formulation, same pair kernel: quadratic prefix building
 * and adjoint scatter. Reference for scan optimization, not a training path. */
void gp_prefix_reference(size_t batch,size_t time,const float *z,const float *dp,float *p,float *dz) {
  size_t b,t,s,k;memset(dz,0,batch*time*GP_RANK*sizeof(float));
  for(b=0;b<batch;b++)for(t=0;t<time;t++) {
    float v[GP_RANK],du[GP_RANK],dv[GP_RANK];double sum[GP_RANK]={0};
    double scale=t?1.0/sqrt((double)t):0;size_t row=b*time+t;
    for(s=0;s<t;s++)for(k=0;k<GP_RANK;k++)sum[k]+=z[(b*time+s)*GP_RANK+k];
    for(k=0;k<GP_RANK;k++)v[k]=(float)(sum[k]*scale);
    gp_pair_forward(z+row*GP_RANK,v,p+row*GP_P);
    gp_pair_backward(z+row*GP_RANK,v,dp+row*GP_P,du,dv);
    for(k=0;k<GP_RANK;k++)dz[row*GP_RANK+k]+=du[k];
    for(s=0;s<t;s++)for(k=0;k<GP_RANK;k++)dz[(b*time+s)*GP_RANK+k]+=(float)(dv[k]*scale);
  }
}
