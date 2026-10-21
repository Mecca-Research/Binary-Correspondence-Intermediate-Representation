/* New numerical contract: guard ill-conditioned wedges, compute pair/VJP in
 * FP64 there; a plane with norm <= epsilon has zero feature and zero VJP.
 * FP32 input quantization is not reversed by this fallback.
 */
static int wedge_ill_conditioned(const float *u,const float *v) {
  size_t i;double uu=0,vv=0,uv=0,prod,det;
  for(i=0;i<GP_RANK;i++){uu+=(double)u[i]*u[i];vv+=(double)v[i]*v[i];uv+=(double)u[i]*v[i];}
  prod=uu*vv;det=prod-uv*uv;
  return prod==0 || det<=1e-4*prod;
}
static void wedge_precise_forward(const float *u,const float *v,float *out) {
  double p[GP_P],n=0;size_t i,k,c;
  for(i=0,c=0;i<GP_RANK;i++)for(k=i+1;k<GP_RANK;k++,c++){
    p[c]=(double)u[i]*v[k]-(double)u[k]*v[i];n+=p[c]*p[c];}
  n=sqrt(n);
  if(n<=1e-6){for(c=0;c<GP_P;c++)out[c]=0;return;}
  for(c=0;c<GP_P;c++)out[c]=(float)(p[c]/n);
}
static void wedge_precise_backward(const float *u,const float *v,const float *dp,float *du,float *dv) {
  double p[GP_P],gu[GP_RANK]={0},gv[GP_RANK]={0},n=0,dot=0;size_t i,k,c;
  for(i=0,c=0;i<GP_RANK;i++)for(k=i+1;k<GP_RANK;k++,c++){
    p[c]=(double)u[i]*v[k]-(double)u[k]*v[i];n+=p[c]*p[c];dot+=p[c]*dp[c];}
  n=sqrt(n);
  if(n>1e-6)for(i=0,c=0;i<GP_RANK;i++)for(k=i+1;k<GP_RANK;k++,c++){
    double q=((double)dp[c]-p[c]*dot/(n*n))/n;
    gu[i]+=q*v[k];gu[k]-=q*v[i];gv[i]-=q*u[k];gv[k]+=q*u[i];}
  for(i=0;i<GP_RANK;i++){du[i]=(float)gu[i];dv[i]=(float)gv[i];}
}
