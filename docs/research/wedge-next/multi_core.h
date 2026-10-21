/* Direct product of small oriented Grassmann feature spaces. */
#undef GP_P
#define GP_HEAD_R (GP_RANK/GP_HEADS)
#define GP_HEAD_P ((GP_HEAD_R*(GP_HEAD_R-1))/2)
#define GP_P (GP_HEADS*GP_HEAD_P)
static void gp_pair_forward(const float *u,const float *v,float *out) {
  size_t h,i,k,c;double headscale=1.0/sqrt((double)GP_HEADS);
  for(h=0;h<GP_HEADS;h++) {
    double norm=0,inv;size_t start=h*GP_HEAD_R,base=h*GP_HEAD_P;
    for(i=0,c=base;i<GP_HEAD_R;i++)for(k=i+1;k<GP_HEAD_R;k++,c++) {
      out[c]=u[start+i]*v[start+k]-u[start+k]*v[start+i];norm+=(double)out[c]*out[c];
    }
    norm=sqrt(norm);
    if(norm<=1e-6){for(c=base;c<base+GP_HEAD_P;c++)out[c]=0;continue;}
    inv=1.0/norm;
    for(c=base;c<base+GP_HEAD_P;c++)out[c]=(float)((double)out[c]*inv)*((float)headscale);
  }
}
static void gp_pair_backward(const float *u,const float *v,const float *dp,float *du,float *dv) {
  size_t h,i,k,c;float headscale=1.0f/sqrtf((float)GP_HEADS);
  for(i=0;i<GP_RANK;i++){du[i]=0;dv[i]=0;}
  for(h=0;h<GP_HEADS;h++) {
    float p[GP_HEAD_P];double norm=0,dot=0,inv,correction;int clamped;
    size_t start=h*GP_HEAD_R,base=h*GP_HEAD_P;
    for(i=0,c=0;i<GP_HEAD_R;i++)for(k=i+1;k<GP_HEAD_R;k++,c++) {
      p[c]=u[start+i]*v[start+k]-u[start+k]*v[start+i];norm+=(double)p[c]*p[c];dot+=(double)p[c]*(dp[base+c]*headscale);
    }
    norm=sqrt(norm);clamped=norm<=1e-6;if(clamped)continue;
    inv=1.0/norm;correction=dot*inv*inv;
    for(i=0,c=0;i<GP_HEAD_R;i++)for(k=i+1;k<GP_HEAD_R;k++,c++) {
      float q=(float)(((double)(dp[base+c]*headscale)-p[c]*correction)*inv);
      du[start+i]+=q*v[start+k];du[start+k]-=q*v[start+i];dv[start+i]-=q*u[start+k];dv[start+k]+=q*u[start+i];
    }
  }
}
