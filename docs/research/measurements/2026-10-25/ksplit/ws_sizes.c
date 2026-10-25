/* NEXT-3: the state's workspace figure (bcir_decoder_workspace_need) for the GEO dimensions and the 9.84M decoder,
 * beside the K-split partials the backward's largest GEMM group would need (blocks x m x n floats, summed over a
 * group's weight adjoints at least four K blocks deep). usage: ws_sizes LIB */
#include "bcir_decoder_train.h"
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
typedef int (*plan_fn)(const bcir_decoder_spec *, bcir_decoder_plan *);
typedef size_t (*need_fn)(const bcir_decoder_plan *, uint32_t, unsigned);
static size_t blocks(size_t k) { size_t ls=0,c=0; while (ls<k) { size_t rem=k-ls,len=rem>=896 ? 448 : rem>448 ? (rem/2+15)/16*16 : rem; ls+=len; c++; } return c; }
static size_t part(size_t m, size_t n, size_t k) { size_t b=blocks(k); return b>=4 ? b*m*n*4 : 0; }
int main(int argc, char **argv) {
  void *h=dlopen(argv[1],RTLD_NOW); plan_fn plan=(plan_fn)dlsym(h,"bcir_decoder_make_plan"); need_fn need=(need_fn)dlsym(h,"bcir_decoder_workspace_need");
  struct { const char *name; unsigned v,d,hh,kv,l,f,b,t,contract; } c[]={
    {"geo",260,128,4,4,3,256,4,512,14},{"geo",260,128,4,4,3,256,4,1024,14},{"geo",260,128,4,4,3,256,4,2048,14},
    {"9.84M",260,384,8,4,6,1024,8,128,46},{"9.84M",260,384,8,4,6,1024,8,512,46},{"9.84M",260,384,8,4,6,1024,8,1024,46}};
  size_t i; (void)argc;
  for (i=0;i<sizeof c/sizeof c[0];i++) {
    bcir_decoder_spec s; bcir_decoder_plan p; size_t n,d,f,k,v,qkv,gu,down,head,o,most;
    memset(&s,0,sizeof s); s.vocab=c[i].v; s.width=c[i].d; s.heads=c[i].hh; s.kvheads=c[i].kv; s.layers=c[i].l; s.ff=c[i].f;
    s.batch=c[i].b; s.time=c[i].t; s.tied=1; s.rope_base=10000.0; s.rms_eps=1e-5;
    if (plan(&s,&p)) { printf("%s refused\n",c[i].name); continue; }
    n=(size_t)c[i].b*c[i].t; d=c[i].d; f=c[i].f; v=c[i].v; k=(size_t)c[i].kv*(d/c[i].hh);
    qkv=part(d,d,n)+2*part(k,d,n); gu=2*part(f,d,n); down=part(d,f,n); head=part(v,d,n); o=part(d,d,n);
    most=qkv; if (gu>most) most=gu; if (down>most) most=down; if (head>most) most=head; if (o>most) most=o;
    printf("%-6s B%u T%-5u contract %2u: workspace need %8.2f MiB (4 threads), K-split partials' largest group %8.2f MiB"
           " (qkv %.2f, gate/up %.2f, down %.2f, o %.2f, head %.2f; %zu K blocks)\n",c[i].name,c[i].b,c[i].t,c[i].contract,
           need(&p,c[i].contract,4)/1048576.0,most/1048576.0,qkv/1048576.0,gu/1048576.0,down/1048576.0,o/1048576.0,head/1048576.0,blocks(n));
  }
  return 0;
}
