"""Reproducible query-aware C generation from immutable pinned full-head sources."""
from build_next import ROOT,once,compile_source
import hashlib,json

def guarded(s):
    s=once(s,'static void gp_pair_forward(',(ROOT/'precision_core.h').read_text()+'\nstatic void gp_pair_forward(')
    s=once(s,'static void gp_pair_forward(const float *u,const float *v,float *out) {',
        'static void gp_pair_forward(const float *u,const float *v,float *out) {\n  if(wedge_ill_conditioned(u,v)){wedge_precise_forward(u,v,out);return;}')
    return once(s,'static void gp_pair_backward(const float *u,const float *v,const float *dp,float *du,float *dv) {',
        'static void gp_pair_backward(const float *u,const float *v,const float *dp,float *du,float *dv) {\n  if(wedge_ill_conditioned(u,v)){wedge_precise_backward(u,v,dp,du,dv);return;}')

def query_head(s):
    s=once(s,'void gp_wedge_forward(',(ROOT/'query_core.h').read_text()+'\nvoid gp_wedge_forward(')
    s=once(s,'!term(&lp,(size_t)GP_RANK+GP_P,s->width)',
           '!term(&lp,2*(size_t)GP_RANK+GP_P,s->width)')
    s=once(s,'(GP_RANK+GP_P)*g.d','(2*GP_RANK+GP_P)*g.d')
    s=once(s,'float *norm1,*q,*k,*v,*o,*norm2,*gate,*up,*down;',
           'float *norm1,*q,*k,*contract,*v,*o,*norm2,*gate,*up,*down;')
    s=once(s,'p.k=x; x+=g.d*GP_P;',
           'p.k=x; x+=g.d*GP_P;p.contract=x;x+=g.d*GP_RANK;')
    s=once(s,'!term(&la,2,nr)','!term(&la,3,nr)')
    s=once(s,'g.la=5*g.nd+2*g.n*GP_RANK','g.la=5*g.nd+3*g.n*GP_RANK')
    s=once(s,'a.ctx=NULL;','a.ctx=x;x+=g.n*GP_RANK;')
    s=once(s,'3*(size_t)s->width+GP_P+2*GP_RANK',
           '3*(size_t)s->width+GP_P+3*GP_RANK')
    s=once(s,'    linear(s,g.n,GP_P,g.d,a.k,p.k,a.v);',
           '''    linear(s,g.n,GP_P,g.d,a.k,p.k,a.v);
    gp_query_contract_forward(s->pool,g.n,a.q,a.k,a.ctx);
    gemm(s,0,1,g.n,g.d,GP_RANK,1.0f,a.ctx,p.contract,1.0f,a.v);''')
    s=once(s,'    gp_wedge_backward(s->pool,g.b,g.t,a.q,dk,dq);',
           '''    /* dc_query follows the two frame adjoints, disjoint from dy=dv. */
    linear_backward(s,g.n,GP_RANK,g.d,a.ctx,p.contract,dv,dq+2*g.n*GP_RANK,grad.contract,0.0f,wb);
    gp_query_wedge_backward(s->pool,g.b,g.t,a.q,dk,dq+2*g.n*GP_RANK,dq);''')
    return s

if __name__=='__main__':
    compile_source((ROOT/'foundation/compact.c').read_text(),'compact')
    compile_source(query_head((ROOT/'foundation/compact.c').read_text()),'query_compact')
    compile_source((ROOT/'foundation/attention.c').read_text(),'attention')
    compile_source(guarded((ROOT/'foundation/hybrid_compact.c').read_text()),'guard_hybrid_compact')
    for name in ('compact','frame_ff_compact','layer_frame_ff'):
        source=(ROOT/'foundation'/f'{name}.c').read_text()
        compile_source(guarded(source),'guard_'+name)
        compile_source(guarded(query_head(source)),'query_guard_'+name)
    for name in ('query_guard_compact','query_guard_layer_frame_ff'):
        compile_source((ROOT/'build'/f'{name}.c').read_text(),name+'_ubsan',sanitize=True)
    print('Full wedge + query contraction built; no head coordinates removed.')
