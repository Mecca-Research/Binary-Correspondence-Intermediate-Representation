/* Full packed wedge plus query interior product. No dynamic allocation.
 * P_ij=p_ij (i<j), P_ji=-p_ij; c=q^T P, q=u/max(||u||,1e-6).
 * Norms and row reductions use FP64, features and adjoints are FP32.
 * The query clamp is constant below the floor; its norm derivative is zero.
 * The wedge keeps the pinned pair kernel's separate 1e-6 area clamp.
 */
#define GP_QUERY_EPS 1e-6
static double gp_query_unit(const float *u,float *q) {
  size_t i;double norm=0,den;
  for(i=0;i<GP_RANK;i++)norm+=(double)u[i]*u[i];
  norm=sqrt(norm);den=norm>GP_QUERY_EPS?norm:GP_QUERY_EPS;
  for(i=0;i<GP_RANK;i++)q[i]=(float)((double)u[i]/den);
  return norm;
}
void gp_query_contract_pair(const float *u,const float *p,float *c) {
  size_t i,k,a;float q[GP_RANK];double acc[GP_RANK]={0};
  (void)gp_query_unit(u,q);
  for(i=0,a=0;i<GP_RANK;i++)for(k=i+1;k<GP_RANK;k++,a++) {
    acc[k]+=(double)q[i]*p[a];acc[i]-=(double)q[k]*p[a];
  }
  for(i=0;i<GP_RANK;i++)c[i]=(float)acc[i];
}
/* Adds the contraction path to dp and writes its direct query-frame adjoint.
 * du and dc may alias: every dc element is read before the first du write.
 */
void gp_query_contract_vjp(const float *u,const float *p,const float *dc,float *dp,float *du) {
  size_t i,k,a;float q[GP_RANK];double adj[GP_RANK]={0},dot=0;
  double norm=gp_query_unit(u,q),den=norm>GP_QUERY_EPS?norm:GP_QUERY_EPS;
  for(i=0,a=0;i<GP_RANK;i++)for(k=i+1;k<GP_RANK;k++,a++) {
    dp[a]+=(float)((double)q[i]*dc[k]-(double)q[k]*dc[i]);
    adj[i]+=(double)p[a]*dc[k];adj[k]-=(double)p[a]*dc[i];
  }
  /* Use the unrounded unit vector in the norm derivative. */
  if(norm>GP_QUERY_EPS)for(i=0;i<GP_RANK;i++)dot+=adj[i]*((double)u[i]/norm);
  for(i=0;i<GP_RANK;i++)du[i]=(float)((adj[i]-(norm>GP_QUERY_EPS?u[i]/norm*dot:0))/den);
}
typedef struct gp_query_job {
  size_t n;const float *z,*p,*dp,*dc;float *c,*dz;
} gp_query_job;
static void gp_query_forward_task(void *ctx,size_t row) {
  gp_query_job *j=ctx;
  gp_query_contract_pair(j->z+row*GP_RANK,j->p+row*GP_P,j->c+row*GP_RANK);
}
void gp_query_contract_forward(bcir_tensor_pool *pool,size_t n,const float *z,const float *p,float *c) {
  gp_query_job j={n,z,p,NULL,NULL,c,NULL};
  bcir_tensor_pool_run(pool,gp_query_forward_task,&j,n);
}
static void gp_query_backward_task(void *ctx,size_t row) {
  gp_query_job *j=ctx;size_t i;float p[GP_P],dp[GP_P],direct[GP_RANK];
  const float *u=j->z+row*GP_RANK,*v=j->z+(j->n+row)*GP_RANK;
  float *du=j->dz+row*GP_RANK,*dv=j->dz+(j->n+row)*GP_RANK;
  /* Rebuild p locally so frame-rematerialized layouts may already reuse its arena. */
  gp_pair_forward(u,v,p);memcpy(dp,j->dp+row*GP_P,sizeof(dp));
  gp_query_contract_vjp(u,p,j->dc+row*GP_RANK,dp,direct);
  gp_pair_backward(u,v,dp,du,dv);
  for(i=0;i<GP_RANK;i++)du[i]+=direct[i];
}
void gp_query_wedge_backward(bcir_tensor_pool *pool,size_t batch,size_t time,
    const float *z,const float *dp,const float *dc,float *dz) {
  gp_query_job j={batch*time,z,NULL,dp,dc,NULL,dz};
  gp_job reverse={batch,time,z,dp,NULL,dz};
  bcir_tensor_pool_run(pool,gp_query_backward_task,&j,batch*time);
  bcir_tensor_pool_run(pool,gp_reverse_task,&reverse,batch);
}
