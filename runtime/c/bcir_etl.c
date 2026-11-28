/*===- bcir_etl.c - the event transducer in C (the C twin of bcir/etl) -----===
 * PORT-4: bcir/etl/fsm.py, bcir/etl/events.py and bcir/etl/parse.py, freestanding (bcir_etl.h). */
#include "bcir_etl.h"

static int str_eq(bcir_etl_str a, bcir_etl_str b){
  if(a.n!=b.n) return 0;
  for(size_t k=0;k<a.n;k++) if(a.s[k]!=b.s[k]) return 0;
  return 1;
}
static int str_is(bcir_etl_str a, const char *z){
  size_t k=0;
  for(;z[k];k++) if(k>=a.n || a.s[k]!=z[k]) return 0;
  return k==a.n;
}
static int str_ok(bcir_etl_str a){ return a.n==0 || a.s!=NULL; }   /* a (pointer, length) the walk may read */

/* --- the finite-state transducer ------------------------------------------------------------------------------- */
static size_t state_index(const bcir_fsm_state *st, size_t n, bcir_etl_str name){
  for(size_t k=0;k<n;k++) if(str_eq(st[k].name,name)) return k;
  return n;
}
bcir_etl_status bcir_fsm_compile(bcir_fsm *m, const bcir_fsm_state *states, size_t n_states,
                                 const bcir_fsm_transition *tx, size_t n_tx, bcir_etl_str start,
                                 uint32_t *src, uint32_t *dst, size_t *where, int *field){
  size_t w=0; int f=0;
  if(where) *where=0;
  if(field) *field=0;
  if(!m || (n_states && !states) || (n_tx && (!tx || !src || !dst)) || !str_ok(start)) return BCIR_ETL_ERR_INVALID;
  if(n_states>UINT32_MAX || n_tx>UINT32_MAX) return BCIR_ETL_ERR_CAPACITY;
  bcir_etl_status st=BCIR_ETL_OK;
  for(size_t k=0;k<n_states && !st;k++){             /* State: a name, and accepting or an error, not both */
    if(!str_ok(states[k].name)) return BCIR_ETL_ERR_INVALID;
    if(states[k].name.n==0){ st=BCIR_ETL_ERR_STATE_NAME; w=k; }
    else if(states[k].accepting && states[k].error){ st=BCIR_ETL_ERR_STATE_KIND; w=k; } }
  for(size_t k=0;k<n_tx && !st;k++){                 /* Transition: a src, a dst and an on, in that order */
    if(!str_ok(tx[k].src)||!str_ok(tx[k].dst)||!str_ok(tx[k].on)||!str_ok(tx[k].action)) return BCIR_ETL_ERR_INVALID;
    if(tx[k].src.n==0){ st=BCIR_ETL_ERR_TX_FIELD; w=k; f=0; }
    else if(tx[k].dst.n==0){ st=BCIR_ETL_ERR_TX_FIELD; w=k; f=1; }
    else if(tx[k].on.n==0){ st=BCIR_ETL_ERR_TX_FIELD; w=k; f=2; } }
  for(size_t k=0;k<n_states && !st;k++)              /* Transducer: each state's name once */
    if(state_index(states,k,states[k].name)<k){ st=BCIR_ETL_ERR_STATE_DUP; w=k; }
  for(size_t k=0;k<n_tx && !st;k++){                 /* ... each transition between declared states, once per key */
    size_t s=state_index(states,n_states,tx[k].src), d=state_index(states,n_states,tx[k].dst);
    if(s==n_states){ st=BCIR_ETL_ERR_TX_ENDPOINT; w=k; f=0; break; }
    if(d==n_states){ st=BCIR_ETL_ERR_TX_ENDPOINT; w=k; f=1; break; }
    src[k]=(uint32_t)s; dst[k]=(uint32_t)d;
    for(size_t j=0;j<k;j++) if(src[j]==src[k] && str_eq(tx[j].on,tx[k].on)){ st=BCIR_ETL_ERR_TX_DUP; w=k; break; } }
  if(!st){ size_t s=state_index(states,n_states,start); if(s==n_states) st=BCIR_ETL_ERR_START; else m->start=s; }
  if(st){ if(where) *where=w; if(field) *field=f; return st; }
  m->states=states; m->n_states=n_states; m->tx=tx; m->n_tx=n_tx; m->src=src; m->dst=dst;
  return BCIR_ETL_OK;
}
static size_t edge_on(const bcir_fsm *m, size_t cur, bcir_etl_str sym){
  for(size_t k=0;k<m->n_tx;k++) if(m->src[k]==cur && str_eq(m->tx[k].on,sym)) return k;
  return m->n_tx;
}
bcir_etl_status bcir_fsm_run(const bcir_fsm *m, const bcir_etl_str *symbols, size_t n,
                             bcir_fsm_step *trace, size_t cap, size_t *n_steps,
                             size_t *final_state, int *accepted){
  static const char star[]="*";
  const bcir_etl_str any={star,1};
  if(!m || (n && !symbols) || (cap && !trace) || !n_steps || !final_state || !accepted || m->start>=m->n_states)
    return BCIR_ETL_ERR_INVALID;
  size_t cur=m->start, steps=0;
  *n_steps=0; *final_state=cur; *accepted=0;
  for(size_t i=0;i<n;i++){
    if(!str_ok(symbols[i])) return BCIR_ETL_ERR_INVALID;
    size_t t=edge_on(m,cur,symbols[i]);
    if(t==m->n_tx) t=edge_on(m,cur,any);             /* the catch-all, `*` */
    if(t==m->n_tx || m->states[cur].error){ *n_steps=steps; *final_state=cur; *accepted=0; return BCIR_ETL_OK; }
    if(steps>=cap) return BCIR_ETL_ERR_CAPACITY;
    trace[steps].tx=(uint32_t)t; trace[steps].sym=i; steps++;
    cur=m->dst[t]; }
  *n_steps=steps; *final_state=cur;
  *accepted=m->states[cur].accepting && !m->states[cur].error;
  return BCIR_ETL_OK;
}

/* --- the event descriptors ------------------------------------------------------------------------------------- */
static int one_of(bcir_etl_str s, const char *const *names, size_t n){
  for(size_t k=0;k<n;k++) if(str_is(s,names[k])) return 1;
  return 0;
}
bcir_etl_status bcir_event_stream_check(bcir_etl_str name, bcir_etl_str kind, bcir_etl_str encoding,
                                        int64_t element_bits, int64_t max_window){
  static const char *const kinds[]={"text","binary","telemetry","packet","driver","token"};
  static const char *const encodings[]={"utf8","bytes","le","be","records","custom"};
  if(!str_ok(name)||!str_ok(kind)||!str_ok(encoding)) return BCIR_ETL_ERR_INVALID;
  if(name.n==0) return BCIR_ETL_ERR_STREAM_NAME;
  if(!one_of(kind,kinds,6)) return BCIR_ETL_ERR_STREAM_KIND;
  if(!one_of(encoding,encodings,6)) return BCIR_ETL_ERR_STREAM_ENCODING;
  if(element_bits<1) return BCIR_ETL_ERR_STREAM_BITS;
  if(max_window<1) return BCIR_ETL_ERR_STREAM_WINDOW;
  return BCIR_ETL_OK;
}
bcir_etl_status bcir_event_kind_check(bcir_etl_str name){
  if(!str_ok(name)) return BCIR_ETL_ERR_INVALID;
  return name.n ? BCIR_ETL_OK : BCIR_ETL_ERR_KIND_NAME;
}
bcir_etl_status bcir_event_check(bcir_etl_str kind, int64_t begin, int64_t end){
  if(!str_ok(kind)) return BCIR_ETL_ERR_INVALID;
  if(kind.n==0) return BCIR_ETL_ERR_EVENT_KIND;
  if(begin<0 || end<begin) return BCIR_ETL_ERR_EVENT_SPAN;
  return BCIR_ETL_OK;
}

/* --- the mini-ROP grammar -------------------------------------------------------------------------------------- */
const char *bcir_rop_tok_name(bcir_rop_tok kind){
  static const char *const names[]={"LBRACE","RBRACE","SEMI","COMMA","INT","IDENT","EOF"};
  return (unsigned)kind<7u ? names[kind] : "?";
}
typedef struct { const char *t; size_t n, i; int kind; size_t at, len; bcir_rop_out *out; } rop_lex;
static int is_digit(char c){ return c>='0' && c<='9'; }
static int is_id0(char c){ return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||c=='_'; }
/* The next token, as the oracle's rules match in order: white space skipped, a brace, `;`, `,`, digits, an
 * identifier; any other character refused at its offset. 0 with the refusal recorded. */
static int rop_next(rop_lex *L){
  while(L->i<L->n && (L->t[L->i]==' '||L->t[L->i]=='\t'||L->t[L->i]=='\r'||L->t[L->i]=='\n')) L->i++;
  L->at=L->i; L->len=0;
  if(L->i>=L->n){ L->kind=BCIR_ROP_EOF; return 1; }
  char c=L->t[L->i];
  if(c=='{'||c=='}'||c==';'||c==','){
    L->kind = c=='{'?BCIR_ROP_LBRACE : c=='}'?BCIR_ROP_RBRACE : c==';'?BCIR_ROP_SEMI : BCIR_ROP_COMMA;
    L->len=1; L->i++; return 1; }
  if(is_digit(c)){ L->kind=BCIR_ROP_INT; while(L->i<L->n && is_digit(L->t[L->i])) L->i++; L->len=L->i-L->at; return 1; }
  if(is_id0(c)){ L->kind=BCIR_ROP_IDENT; while(L->i<L->n && (is_id0(L->t[L->i])||is_digit(L->t[L->i]))) L->i++;
    L->len=L->i-L->at; return 1; }
  L->out->pos=L->i; return 0;
}
/* The oracle tokenizes the whole text before it parses: a character no rule takes is refused first, wherever it is. */
static int rop_lex_all(const char *t, size_t n, bcir_rop_out *out){
  rop_lex L={t,n,0,0,0,0,out};
  do { if(!rop_next(&L)) return 0; } while(L.kind!=BCIR_ROP_EOF);
  return 1;
}
static bcir_etl_status rop_refuse(rop_lex *L, bcir_etl_status st, int want){
  L->out->pos=L->at; L->out->tok_at=L->at; L->out->tok_n=L->len; L->out->tok_kind=L->kind; L->out->want_kind=want;
  return st;
}
static int tok_text_is(const rop_lex *L, const char *z){ bcir_etl_str s={L->t+L->at,L->len}; return str_is(s,z); }
static int tok_text_icase(const rop_lex *L, const char *upper){   /* the oracle's `.lower()` against a model name */
  size_t k=0;
  for(;upper[k];k++){ if(k>=L->len) return 0; char c=L->t[L->at+k]; if(c>='a'&&c<='z') c=(char)(c-'a'+'A');
    if(c!=upper[k]) return 0; }
  return k==L->len;
}
static bcir_etl_status rop_rid(const char *t, bcir_rop_out *out, size_t at, size_t n, uint32_t *rid){
  bcir_etl_str s={t+at,n};
  for(size_t k=0;k<out->n_res;k++){ bcir_etl_str r={t+out->res[k].at,out->res[k].n};
    if(str_eq(r,s)){ *rid=out->res[k].rid; return BCIR_ETL_OK; } }
  if(out->n_res>=out->cap_res || out->n_res>=UINT32_MAX-10u) return BCIR_ETL_ERR_CAPACITY;
  out->res[out->n_res].at=at; out->res[out->n_res].n=n; out->res[out->n_res].rid=(uint32_t)(10u+out->n_res);
  *rid=out->res[out->n_res].rid; out->n_res++;
  return BCIR_ETL_OK;
}
/* `idlist := IDENT ("," IDENT)*`, the rids in out->rids from *first; the cursor on the token after the list. */
static bcir_etl_status rop_idlist(rop_lex *L, size_t *first, size_t *count){
  bcir_rop_out *out=L->out; bcir_etl_status st;
  *first=out->n_rids; *count=0;
  for(;;){
    if(L->kind!=BCIR_ROP_IDENT) return rop_refuse(L,BCIR_ETL_ERR_EXPECT,BCIR_ROP_IDENT);
    if(out->n_rids>=out->cap_rids) return BCIR_ETL_ERR_CAPACITY;
    uint32_t rid; if((st=rop_rid(L->t,out,L->at,L->len,&rid))) return st;
    out->rids[out->n_rids++]=rid; (*count)++;
    (void)rop_next(L);
    if(L->kind!=BCIR_ROP_COMMA) return BCIR_ETL_OK;
    (void)rop_next(L); }
}
bcir_etl_status bcir_rop_parse(const char *text, size_t n, bcir_rop_out *out){
  static const char *const lanes[]={"U","UX","T","GGG","A","H"};           /* bcir.model.Lane, by value */
  static const char *const strides[]={"SCALAR","UNIT","STRIDED","CACHELINE","TILE","RANDOM"};   /* StrideClass */
  static const char *const stmts[]={"reads","writes","count","lane","stride"};
  if(!out || (n && !text) || (out->cap_res && !out->res) || (out->cap_claims && !out->claims) ||
     (out->cap_rids && !out->rids)) return BCIR_ETL_ERR_INVALID;
  out->n_res=out->n_claims=out->n_rids=0; out->pos=out->tok_at=out->tok_n=0; out->tok_kind=-1; out->want_kind=-1;
  if(!rop_lex_all(text,n,out)) return BCIR_ETL_ERR_LEX;
  rop_lex L={text,n,0,0,0,0,out}; bcir_etl_status st;
  (void)rop_next(&L);
  while(L.kind!=BCIR_ROP_EOF){                        /* program := claim+  (the oracle reads claims to the end) */
    if(L.kind!=BCIR_ROP_IDENT || !tok_text_is(&L,"claim")) return rop_refuse(&L,BCIR_ETL_ERR_EXPECT,-1);
    (void)rop_next(&L);
    if(L.kind!=BCIR_ROP_IDENT) return rop_refuse(&L,BCIR_ETL_ERR_EXPECT,BCIR_ROP_IDENT);
    if(out->n_claims>=out->cap_claims || out->n_claims>=UINT32_MAX-1000u) return BCIR_ETL_ERR_CAPACITY;
    bcir_rop_claim *cl=&out->claims[out->n_claims];
    cl->id=(uint32_t)(1000u+out->n_claims); cl->name_at=L.at; cl->name_n=L.len;
    cl->rd_first=cl->wr_first=out->n_rids; cl->n_rd=cl->n_wr=0; cl->count=1; cl->lane=0; cl->stride=1;
    (void)rop_next(&L);
    if(L.kind!=BCIR_ROP_LBRACE) return rop_refuse(&L,BCIR_ETL_ERR_EXPECT,BCIR_ROP_LBRACE);
    (void)rop_next(&L);
    unsigned said=0;
    while(L.kind!=BCIR_ROP_RBRACE){
      if(L.kind!=BCIR_ROP_IDENT) return rop_refuse(&L,BCIR_ETL_ERR_EXPECT,BCIR_ROP_IDENT);
      int s=-1; for(int k=0;k<5;k++) if(tok_text_is(&L,stmts[k])) s=k;
      if(s<0) return rop_refuse(&L,BCIR_ETL_ERR_STATEMENT,-1);
      if(said&(1u<<s)) return rop_refuse(&L,BCIR_ETL_ERR_REPEATED,-1);
      said|=1u<<s;
      (void)rop_next(&L);
      if(s==0){ if((st=rop_idlist(&L,&cl->rd_first,&cl->n_rd))) return st; }
      else if(s==1){ if((st=rop_idlist(&L,&cl->wr_first,&cl->n_wr))) return st; }
      else if(s==2){
        if(L.kind!=BCIR_ROP_INT) return rop_refuse(&L,BCIR_ETL_ERR_EXPECT,BCIR_ROP_INT);
        uint64_t v=0; int big=0;
        for(size_t k=0;k<L.len;k++){ unsigned d=(unsigned)(text[L.at+k]-'0');
          if(v>(UINT64_C(0x7fffffffffffffff)-d)/10u){ big=1; break; } v=v*10u+d; }
        if(big) return rop_refuse(&L,BCIR_ETL_ERR_COUNT,-1);
        cl->count=v; (void)rop_next(&L); }
      else {
        if(L.kind!=BCIR_ROP_IDENT) return rop_refuse(&L,BCIR_ETL_ERR_EXPECT,BCIR_ROP_IDENT);
        const char *const *names=s==3?lanes:strides; int v=-1;
        for(int k=0;k<6;k++) if(tok_text_icase(&L,names[k])) v=k;
        if(v<0) return rop_refuse(&L,s==3?BCIR_ETL_ERR_LANE:BCIR_ETL_ERR_STRIDE,-1);
        if(s==3) cl->lane=(uint8_t)v; else cl->stride=(uint8_t)v;
        (void)rop_next(&L); }
      if(L.kind==BCIR_ROP_SEMI) (void)rop_next(&L);  /* `;`-separated; the one before `}` optional */
      else if(L.kind!=BCIR_ROP_RBRACE) return rop_refuse(&L,BCIR_ETL_ERR_SEPARATOR,-1);
    }
    out->n_claims++;
    (void)rop_next(&L);
  }
  return BCIR_ETL_OK;
}
