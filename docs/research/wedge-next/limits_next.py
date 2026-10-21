"""Additional exact memory endpoint; separate from the frozen main protocol.

Share one rematerialized layer cache AND discard frame/FF intermediates.
All parameters, FP operations within a kernel, and reduction chains stay fixed.
"""
from candidates import *
from build_next import compile_source, once

def build_limit():
    source=(ROOT/'build/frame_ff_compact.c').read_text()
    source=once(source,'!term(&p.activations,s->layers,la)','!add(&p.activations,la)')
    source=source.replace('norm=cs+g.l*g.la','norm=cs+g.la').replace('activations(s,cs+l*g.la,g)','activations(s,cs,g)')
    start=source.index('    bcir_tensor_rms_scan_pool',source.index('static int forward('))
    end=source.index('\n  }\n  bcir_tensor_rms_scan_pool',start)
    body=source[start:end]
    helper='''static void forward_layer(bcir_decoder_state *s,geometry g,layer p,cache a,
    float *x,float *y,int *badp) {
  const bcir_decoder_spec *sp=&s->plan.spec;size_t i;int bad=0;
'''+body+'\n  *badp|=bad;\n}\n'
    source=source[:start]+'    forward_layer(s,g,p,a,x,y,&bad);'+source[end:]
    source=once(source,'static int forward(bcir_decoder_state *s',helper+'\nstatic int forward(bcir_decoder_state *s')
    anchor='    cache a=activations(s,cs,g); float *x=xs+l*g.nd;'
    source=once(source,anchor,anchor+'\n    {int bad=0;forward_layer(s,g,p,a,x,x+g.nd,&bad);if(bad)return BCIR_DT_NUMERIC;}')
    # The whole layer recomputation has already produced these gate/up values.
    source=once(source,'''    {
      size_t out[2]={g.f,g.f};const float *w[2]={p.gate,p.up};float *yy[2]={a.gate,a.up};
      linears(s,g.n,g.d,a.n2,2,out,w,yy);
    }
    swiglu_backward''','''    /* gate and up were just rematerialized by forward_layer. */
    swiglu_backward''')
    # Its cached FP32 prefix is still valid; do not repeat the prefix scan.
    source=once(source,'''    gp_wedge_forward(s->pool,g.b,g.t,a.q,a.k);
    /* Weight VJP''','''    {gp_job j={g.b,g.t,a.q,NULL,a.k,NULL};
      bcir_tensor_pool_run(s->pool,gp_forward_task,&j,g.n);}
    /* Weight VJP''')
    compile_source(source,'layer_frame_ff')
    return source

if __name__=='__main__':
    source=build_limit();rng=np.random.default_rng(67210);rows=[]
    for t in (9,64):
        r=Recipe(kind='global',width=32,layers=3,ff=48,batch=2,context=t)
        x=rng.integers(0,260,(5,2,t),dtype=np.uint32);y=rng.integers(0,260,x.shape,dtype=np.uint32)
        reference=None
        for lib in ('baseline','layer_frame_ff'):
            for th in (1,4):
                with Native(r,threads=th,library_name=lib) as n:
                    p=n.forward(x[0]).copy();loss=n.backward(x[0],y[0]);g=n.buffers['gradients'].copy()
                    n.train(x[:3],y[:3],[.0005]*3);state=n.snapshot();n.train(x[3:],y[3:],[.0005]*2);digest=n.digest()
                    n.restore(state);n.train(x[3:],y[3:],[.0005]*2);assert n.digest()==digest
                    if reference is None:reference=(p,loss,g,digest)
                    assert np.array_equal(p,reference[0]) and loss==reference[1] and np.array_equal(g,reference[2]) and digest==reference[3],(t,lib,th)
                    rows.append({'context':t,'library':lib,'threads':th,'state_sha256':digest,'logit_gradient_state':'bitwise identical'})
    compile_source(source,'layer_frame_ff_ubsan',sanitize=True)
    with Native(r,threads=2,library_name='layer_frame_ff_ubsan') as n:n.train(x,y,[.0005]*5)
    with Native(Recipe(kind='global',ff=257,batch=1,context=1024),threads=1,library_name='layer_frame_ff') as n:
        memory={'activation_bytes':int(n.plan.activations)*4,'scratch_bytes':int(n.plan.scratch)*4,'persistent_bytes':int(n.plan.parameters)*16,
            'arena_bytes':int(n.plan.activations+n.plan.scratch)*4,'parameters':int(n.plan.parameters)}
    (ROOT/'results/limits-validation.json').write_text(json.dumps({'checks':rows,'ubsan':'pass','memory':memory,
        'scope':'Additional exact execution endpoint, after the frozen main learning protocol; not a globally minimal training-memory claim.'},indent=2))
    print(json.dumps(memory))
