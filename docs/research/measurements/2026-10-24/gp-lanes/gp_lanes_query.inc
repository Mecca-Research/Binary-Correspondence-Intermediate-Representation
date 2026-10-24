/* GP-LANES: the query frame's kernels, a row a lane (gp_lanes_pair.inc) */
GPL_TARGET static void gp_query_unit_lanes(const float (*U)[GPL],float (*Q)[GPL],gpd *norm_out) {
  size_t i; gpd norm=gpd_set(0),den;
  for(i=0;i<GP_RANK;i++){gpd u=gpd_of(LD(U[i]));norm=gpd_add(norm,gpd_mul(u,u));}
  norm=gpd_sqrt(norm);den=gpd_pick(gpm_gt(norm,gpd_set(GP_QUERY_EPS)),norm,gpd_set(GP_QUERY_EPS));
  for(i=0;i<GP_RANK;i++)ST(Q[i],gpd_float(gpd_div(gpd_of(LD(U[i])),den)));
  *norm_out=norm;
}
GPL_TARGET static void gp_query_contract_pair_lanes(const float (*U)[GPL],const float (*P)[GPL],float (*C)[GPL]) {
  size_t i,k,a; _Alignas(64) float Q[GP_RANK][GPL]; gpd acc[GP_RANK],norm;
  for(i=0;i<GP_RANK;i++)acc[i]=gpd_set(0);
  gp_query_unit_lanes(U,Q,&norm);
  for(i=0,a=0;i<GP_RANK;i++) {
    gpd qi=gpd_of(LD(Q[i]));
    for(k=i+1;k<GP_RANK;k++,a++){
      gpd pa=gpd_of(LD(P[a]));
      acc[k]=gpd_add(acc[k],gpd_mul(qi,pa));acc[i]=gpd_sub(acc[i],gpd_mul(gpd_of(LD(Q[k])),pa));
    }
  }
  for(i=0;i<GP_RANK;i++)ST(C[i],gpd_float(acc[i]));
}
GPL_TARGET static void gp_query_contract_vjp_lanes(const float (*U)[GPL],const float (*P)[GPL],const float (*DC)[GPL],
    float (*DP)[GPL],float (*DU)[GPL]) {
  size_t i,k,a; _Alignas(64) float Q[GP_RANK][GPL]; gpd adj[GP_RANK],dot=gpd_set(0),norm,den; gpm big;
  for(i=0;i<GP_RANK;i++)adj[i]=gpd_set(0);
  gp_query_unit_lanes(U,Q,&norm);
  big=gpm_gt(norm,gpd_set(GP_QUERY_EPS));den=gpd_pick(big,norm,gpd_set(GP_QUERY_EPS));
  for(i=0,a=0;i<GP_RANK;i++) {
    gpd qi=gpd_of(LD(Q[i])),dci=gpd_of(LD(DC[i]));
    for(k=i+1;k<GP_RANK;k++,a++){
      gpd dck=gpd_of(LD(DC[k])),pa=gpd_of(LD(P[a]));
      ST(DP[a],_mm512_add_ps(LD(DP[a]),gpd_float(gpd_sub(gpd_mul(qi,dck),gpd_mul(gpd_of(LD(Q[k])),dci)))));
      adj[i]=gpd_add(adj[i],gpd_mul(pa,dck));adj[k]=gpd_sub(adj[k],gpd_mul(pa,dci));
    }
  }
  for(i=0;i<GP_RANK;i++){  /* if(norm>EPS) dot+=adj[i]*((double)u[i]/norm), lane by lane */
    gpd t=gpd_mul(adj[i],gpd_div(gpd_of(LD(U[i])),norm));
    dot=gpd_pick(big,gpd_add(dot,t),dot);
  }
  for(i=0;i<GP_RANK;i++){
    gpd t=gpd_pick(big,gpd_mul(gpd_div(gpd_of(LD(U[i])),norm),dot),gpd_set(0));
    ST(DU[i],gpd_float(gpd_div(gpd_sub(adj[i],t),den)));
  }
}
