/* KSPLIT-ROOM: a trainer library's workspace figures at the shapes measured -- the need for any pool
 * (bcir_decoder_workspace_bytes), at one thread and at four (bcir_decoder_workspace_need, what the Python
 * trainer lends), and where the library has it the need without the split room (bcir_decoder_workspace_unsplit)
 * -- beside the split room stated apart from the unit: each weight adjoint's K blocks' partials, blocks x m x n
 * floats to the next 64 bytes from four blocks on, and the most of the backward's GEMM jobs (the head's pair,
 * the down projection's pair, the gate and up group, the output projection's pair, the q, k and v group).
 * One JSON line a shape. usage: room_sizes LIB */
#include "bcir_decoder_train.h"
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
typedef int (*plan_fn)(const bcir_decoder_spec *, bcir_decoder_plan *);
typedef size_t (*any_fn)(const bcir_decoder_plan *, uint32_t);
typedef size_t (*need_fn)(const bcir_decoder_plan *, uint32_t, unsigned);
static size_t blocks(size_t k) {
  size_t ls=0,c=0;
  while (ls<k) { size_t rem=k-ls,len=rem>=896 ? 448 : rem>448 ? (rem/2+15)/16*16 : rem; ls+=len; c++; }
  return c;
}
static size_t room(size_t m, size_t n, size_t k) { size_t b=blocks(k); return b>=4 ? (b*m*n*4+63)/64*64 : 0; }
static size_t most(size_t a, size_t b) { return a>b ? a : b; }
int main(int argc, char **argv) {
  void *h=argc>1 ? dlopen(argv[1],RTLD_NOW|RTLD_LOCAL) : NULL;
  plan_fn plan; any_fn any; need_fn need,unsplit; size_t i;
  struct { const char *name; unsigned v,d,hh,kv,l,f,b,t,contract; } c[]={
    {"corpus",1024,128,4,2,4,344,8,256,14},
    {"geo",260,128,4,4,3,256,4,512,14},{"geo",260,128,4,4,3,256,4,1024,14},{"geo",260,128,4,4,3,256,4,2048,14},
    {"9.84M",260,384,8,4,6,1024,8,128,46},{"9.84M",260,384,8,4,6,1024,8,512,46},{"9.84M",260,384,8,4,6,1024,8,1024,46}};
  if (!h) { fprintf(stderr,"usage: room_sizes LIB\n"); return 2; }
  plan=(plan_fn)dlsym(h,"bcir_decoder_make_plan"); any=(any_fn)dlsym(h,"bcir_decoder_workspace_bytes");
  need=(need_fn)dlsym(h,"bcir_decoder_workspace_need"); unsplit=(need_fn)dlsym(h,"bcir_decoder_workspace_unsplit");
  if (!plan || !any || !need) { fprintf(stderr,"room_sizes: %s lacks the plan's entries\n",argv[1]); return 2; }
  for (i=0;i<sizeof c/sizeof c[0];i++) {
    bcir_decoder_spec s; bcir_decoder_plan p; size_t n,d,f,k,v,head,down,gu,o,qkv,split;
    memset(&s,0,sizeof s); s.vocab=c[i].v; s.width=c[i].d; s.heads=c[i].hh; s.kvheads=c[i].kv; s.layers=c[i].l;
    s.ff=c[i].f; s.batch=c[i].b; s.time=c[i].t; s.tied=1; s.rope_base=10000.0; s.rms_eps=1e-5;
    if (plan(&s,&p)) { printf("{\"case\":\"%s\",\"refused\":true}\n",c[i].name); continue; }
    n=(size_t)c[i].b*c[i].t; d=c[i].d; f=c[i].f; v=c[i].v; k=(size_t)c[i].kv*(d/c[i].hh);
    head=room(v,d,n); down=room(d,f,n); gu=2*room(f,d,n); o=room(d,d,n); qkv=room(d,d,n)+2*room(k,d,n);
    split=most(most(most(head,down),most(gu,o)),qkv);
    printf("{\"case\":\"%s B%u T%u\",\"contract\":%u,\"rows\":%zu,\"k_blocks\":%zu,\"need_any_pool\":%zu,"
           "\"need_1\":%zu,\"need_4\":%zu,",c[i].name,c[i].b,c[i].t,c[i].contract,n,blocks(n),
           any(&p,c[i].contract),need(&p,c[i].contract,1),need(&p,c[i].contract,4));
    if (unsplit) printf("\"unsplit_1\":%zu,\"unsplit_4\":%zu,",unsplit(&p,c[i].contract,1),unsplit(&p,c[i].contract,4));
    else printf("\"unsplit_1\":null,\"unsplit_4\":null,");
    printf("\"split_room\":%zu,\"jobs\":{\"head\":%zu,\"down\":%zu,\"gate_up\":%zu,\"o\":%zu,\"qkv\":%zu}}\n",
           split,head,down,gu,o,qkv);
  }
  return 0;
}
