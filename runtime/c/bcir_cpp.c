/*===- bcir_cpp.c - the BCIR C preprocessor (L7) ---------------------------===*/
#include "bcir_cpp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* --- macro table --------------------------------------------------------- */
typedef struct {
  char name[64];
  int isfunc;
  int np; char params[16][64]; int variadic;
  char body[1024];
} Macro;
static Macro M[1024]; static int NM;

static int find_macro(const char *n, int len) {
  for (int i = 0; i < NM; i++)
    if ((int)strlen(M[i].name) == len && !strncmp(M[i].name, n, len)) return i;
  return -1;
}
static void undef_macro(const char *n) {
  int i = find_macro(n, (int)strlen(n));
  if (i >= 0) M[i] = M[--NM];
}

/* __FILE__/__LINE__: the file currently being processed + its 1-based logical line number. These
 * are dynamic predefined macros (not stored in the table); a nested #include saves/restores them. */
static const char *g_cur_file = "<source>";
static int g_cur_line;

/* A macro is "defined" (for #ifdef / defined()) if it is in the table OR is a dynamic predefined. */
static int is_defined(const char *n) {
  if (find_macro(n, (int)strlen(n)) >= 0) return 1;
  return !strcmp(n, "__FILE__") || !strcmp(n, "__LINE__");
}

/* --- preprocessing tokenizer --------------------------------------------- */
static int idc(int c) { return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                               (c >= '0' && c <= '9'); }
static int id0(int c) { return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

/* next token from s[*i]; copies into out; returns 'i' ident, 'n' number, 's' string,
 * 'p' punct, 0 end. skips leading spaces. */
static int ntok(const char *s, int *i, char *out, int cap) {
  while (s[*i] == ' ' || s[*i] == '\t') (*i)++;
  int c = (unsigned char)s[*i];
  if (!c) return 0;
  int j = 0;
  if (id0(c)) { while (idc((unsigned char)s[*i]) && j < cap - 1) out[j++] = s[(*i)++]; out[j] = 0; return 'i'; }
  if (c >= '0' && c <= '9') { while ((idc((unsigned char)s[*i]) || s[*i] == '.') && j < cap - 1) out[j++] = s[(*i)++]; out[j] = 0; return 'n'; }
  if (c == '"' || c == '\'') { char q = (char)c; out[j++] = s[(*i)++];
    while (s[*i] && s[*i] != q && j < cap - 2) { if (s[*i] == '\\' && s[*i+1]) out[j++] = s[(*i)++]; out[j++] = s[(*i)++]; }
    if (s[*i] == q) out[j++] = s[(*i)++]; out[j] = 0; return 's'; }
  static const char *ops[] = {"<<=", ">>=", "...", "->", "##", "<<", ">>", "<=", ">=", "==",
                              "!=", "&&", "||", 0};
  for (int k = 0; ops[k]; k++) { int L = (int)strlen(ops[k]);
    if (!strncmp(s + *i, ops[k], (size_t)L)) { memcpy(out, ops[k], (size_t)L); out[L] = 0; *i += L; return 'p'; } }
  out[0] = s[(*i)++]; out[1] = 0; return 'p';
}

/* --- macro expansion (expand a line until stable) ------------------------ */
static void app(char *o, size_t cap, size_t *w, const char *s) {
  size_t n = strlen(s); if (*w + n + 1 < cap) { memcpy(o + *w, s, n); *w += n; o[*w] = 0; }
}
static int needspace(char a, char b) {            /* keep ident/number tokens apart */
  int ai = idc((unsigned char)a), bi = idc((unsigned char)b); return ai && bi;
}

/* substitute a function macro's args into its body; writes to out. */
static void substitute(const Macro *m, char args[][1024], int na, char *out, size_t cap) {
  size_t w = 0; out[0] = 0;
  int i = 0; char t[256], prev[256] = "";
  while (1) {
    int k = ntok(m->body, &i, t, sizeof t); if (!k) break;
    if (!strcmp(t, "#") && k == 'p') {            /* stringize next param */
      char p[256]; ntok(m->body, &i, p, sizeof p);
      int pi = -1; for (int q = 0; q < m->np; q++) if (!strcmp(m->params[q], p)) pi = q;
      app(out, cap, &w, "\""); if (pi >= 0 && pi < na) app(out, cap, &w, args[pi]); app(out, cap, &w, "\"");
      strcpy(prev, "\""); continue;
    }
    if (!strcmp(t, "##") && k == 'p') {           /* paste: glue prev to next, no space */
      char r[256]; int rk = ntok(m->body, &i, r, sizeof r); (void)rk;
      int pi = -1; for (int q = 0; q < m->np; q++) if (!strcmp(m->params[q], r)) pi = q;
      const char *rt = (pi >= 0 && pi < na) ? args[pi] : r;
      app(out, cap, &w, rt); strcpy(prev, rt[0] ? rt : prev); continue;
    }
    int pi = -1; if (k == 'i') for (int q = 0; q < m->np; q++) if (!strcmp(m->params[q], t)) pi = q;
    const char *emit = (pi >= 0 && pi < na) ? args[pi] : t;
    if (w && needspace(prev[strlen(prev) ? strlen(prev) - 1 : 0], emit[0])) app(out, cap, &w, " ");
    app(out, cap, &w, emit);
    strncpy(prev, emit, sizeof prev - 1);
  }
}

static int expand_once(const char *line, char *out, size_t cap, int *changed) {
  size_t w = 0; out[0] = 0; *changed = 0;
  int i = 0; char t[256], prevc = 0;
  while (1) {
    int save = i, k = ntok(line, &i, t, sizeof t); if (!k) break;
    if (k == 'i') {
      int mi = find_macro(t, (int)strlen(t));
      if (mi >= 0 && M[mi].isfunc) {
        int j = i; char nx[8]; int nk = ntok(line, &j, nx, sizeof nx);
        if (nk && !strcmp(nx, "(")) {
          char args[16][1024]; int na = 0; int depth = 1; size_t aw = 0; args[0][0] = 0;
          char a[256];
          while (depth) { int ak = ntok(line, &j, a, sizeof a); if (!ak) break;
            if (!strcmp(a, "(")) depth++;
            else if (!strcmp(a, ")")) { depth--; if (!depth) break; }
            else if (!strcmp(a, ",") && depth == 1) { args[na][aw] = 0; na++; aw = 0; if (na < 16) args[na][0] = 0; continue; }
            if (na < 16) { if (aw && needspace(args[na][aw-1], a[0]) && aw < 1023) args[na][aw++] = ' ';
              for (int z = 0; a[z] && aw < 1023; z++) args[na][aw++] = a[z]; args[na][aw] = 0; } }
          na++;
          char sub[2048]; substitute(&M[mi], args, na, sub, sizeof sub);
          if (w && needspace(prevc, sub[0])) app(out, cap, &w, " ");
          app(out, cap, &w, sub); prevc = sub[0] ? sub[strlen(sub) - 1] : prevc;
          i = j; *changed = 1; continue;
        }
      } else if (mi >= 0) {                        /* object macro */
        if (w && needspace(prevc, M[mi].body[0])) app(out, cap, &w, " ");
        app(out, cap, &w, M[mi].body); prevc = M[mi].body[0] ? M[mi].body[strlen(M[mi].body)-1] : prevc;
        *changed = 1; continue;
      } else if (!strcmp(t, "__LINE__")) {         /* dynamic predefined: current line number */
        char num[16]; snprintf(num, sizeof num, "%d", g_cur_line);
        if (w && needspace(prevc, num[0])) app(out, cap, &w, " ");
        app(out, cap, &w, num); prevc = num[strlen(num) - 1]; *changed = 1; continue;
      } else if (!strcmp(t, "__FILE__")) {         /* dynamic predefined: current file, a string */
        char fl[1100]; size_t fw = 0; fl[fw++] = '"';
        for (const char *q = g_cur_file ? g_cur_file : ""; *q && fw < sizeof fl - 2; q++) {
          if (*q == '\\' || *q == '"') fl[fw++] = '\\';
          fl[fw++] = *q;
        }
        fl[fw++] = '"'; fl[fw] = 0;
        if (w && needspace(prevc, fl[0])) app(out, cap, &w, " ");
        app(out, cap, &w, fl); prevc = '"'; *changed = 1; continue;
      } else if (!strcmp(t, "_Pragma")) {          /* _Pragma("..."): a lowering no-op (like #pragma) */
        int j = i; char nx[8]; int nk = ntok(line, &j, nx, sizeof nx);
        if (nk && !strcmp(nx, "(")) {              /* consume the balanced (...), emit nothing */
          int depth = 1; char a[256];
          while (depth) { int ak = ntok(line, &j, a, sizeof a); if (!ak) break;
            if (!strcmp(a, "(")) depth++; else if (!strcmp(a, ")")) depth--; }
          i = j; *changed = 1; continue;
        }
      }
    }
    (void)save;
    if (w && needspace(prevc, t[0])) app(out, cap, &w, " ");
    app(out, cap, &w, t); prevc = t[strlen(t) - 1];
  }
  return 0;
}
static void expand_line(const char *line, char *out, size_t cap) {
  static char a[8192], b[8192]; strncpy(a, line, sizeof a - 1); a[sizeof a - 1] = 0;
  for (int pass = 0; pass < 64; pass++) { int ch = 0; expand_once(a, b, sizeof b, &ch);
    strncpy(a, b, sizeof a - 1); a[sizeof a - 1] = 0; if (!ch) break; }
  strncpy(out, a, cap - 1); out[cap - 1] = 0;
}

/* --- #if constant-expression evaluation ---------------------------------- */
typedef struct { char t[512][64]; int n, i; } CE;
static long ce_expr(CE *c);
static long lit(const char *s) {
  if (s[0]=='0'&&(s[1]=='x'||s[1]=='X')) return strtol(s,0,16);
  if (s[0]=='0'&&(s[1]=='b'||s[1]=='B')) return strtol(s+2,0,2);
  return strtol(s,0,10);
}
static long ce_prim(CE *c){
  if(c->i>=c->n) return 0; const char *t=c->t[c->i];
  if(!strcmp(t,"(")){c->i++;long v=ce_expr(c);if(c->i<c->n&&!strcmp(c->t[c->i],")"))c->i++;return v;}
  if(!strcmp(t,"!")){c->i++;return !ce_prim(c);}
  if(!strcmp(t,"-")){c->i++;return -ce_prim(c);}
  if(!strcmp(t,"~")){c->i++;return ~ce_prim(c);}
  c->i++;
  if(t[0]>='0'&&t[0]<='9') return lit(t);
  return 0;                                        /* undefined identifier -> 0 */
}
static int prec(const char *o){
  if(!strcmp(o,"||"))return 1; if(!strcmp(o,"&&"))return 2;
  if(!strcmp(o,"|"))return 3; if(!strcmp(o,"^"))return 4; if(!strcmp(o,"&"))return 5;
  if(!strcmp(o,"==")||!strcmp(o,"!="))return 6;
  if(!strcmp(o,"<")||!strcmp(o,">")||!strcmp(o,"<=")||!strcmp(o,">="))return 7;
  if(!strcmp(o,"<<")||!strcmp(o,">>"))return 8;
  if(!strcmp(o,"+")||!strcmp(o,"-"))return 9;
  if(!strcmp(o,"*")||!strcmp(o,"/")||!strcmp(o,"%"))return 10;
  return 0;
}
static long apply(const char *o,long a,long b){
  if(!strcmp(o,"||"))return a||b; if(!strcmp(o,"&&"))return a&&b; if(!strcmp(o,"|"))return a|b;
  if(!strcmp(o,"^"))return a^b; if(!strcmp(o,"&"))return a&b; if(!strcmp(o,"=="))return a==b;
  if(!strcmp(o,"!="))return a!=b; if(!strcmp(o,"<"))return a<b; if(!strcmp(o,">"))return a>b;
  if(!strcmp(o,"<="))return a<=b; if(!strcmp(o,">="))return a>=b; if(!strcmp(o,"<<"))return a<<b;
  if(!strcmp(o,">>"))return a>>b; if(!strcmp(o,"+"))return a+b; if(!strcmp(o,"-"))return a-b;
  if(!strcmp(o,"*"))return a*b; if(!strcmp(o,"/"))return b?a/b:0; if(!strcmp(o,"%"))return b?a%b:0;
  return 0;
}
static long ce_bin(CE *c,int minp){
  long lhs=ce_prim(c);
  while(c->i<c->n){int p=prec(c->t[c->i]); if(p<minp||p==0)break; char op[8];strcpy(op,c->t[c->i]);c->i++;
    long rhs=ce_bin(c,p+1); lhs=apply(op,lhs,rhs);} return lhs;
}
static long ce_expr(CE *c){return ce_bin(c,1);}

static long eval_if(const char *expr) {
  /* replace `defined X` / `defined(X)` first, then expand macros, then evaluate. */
  static char buf[8192]; size_t w=0; buf[0]=0; int i=0; char t[256];
  while(1){int k=ntok(expr,&i,t,sizeof t); if(!k)break;
    if(k=='i'&&!strcmp(t,"defined")){char p[256]; int j=i; int pk=ntok(expr,&j,p,sizeof p);
      int has; char nm[256];
      if(pk&&!strcmp(p,"(")){ntok(expr,&j,nm,sizeof nm);char cl[8];ntok(expr,&j,cl,sizeof cl);i=j;}
      else {strcpy(nm,p);i=j;}
      has = is_defined(nm);
      buf[w++]= has?'1':'0'; buf[w]=0; continue;}
    size_t n=strlen(t); if(w&&needspace(buf[w-1],t[0])){buf[w++]=' ';} memcpy(buf+w,t,n);w+=n;buf[w]=0;}
  static char ex[8192]; expand_line(buf,ex,sizeof ex);
  CE c; c.n=0;c.i=0; int j=0; char tk[64];
  while(c.n<512){int k=ntok(ex,&j,tk,sizeof tk); if(!k)break; strncpy(c.t[c.n++],tk,63);c.t[c.n-1][63]=0;}
  return ce_expr(&c);
}

/* --- directive processing ------------------------------------------------ */
static void define_macro(const char *rest) {
  int i=0; while(rest[i]==' ')i++; int s=i; while(idc((unsigned char)rest[i]))i++;
  if(i==s)return; Macro m; memset(&m,0,sizeof m); int L=i-s; if(L>63)L=63; memcpy(m.name,rest+s,(size_t)L);m.name[L]=0;
  if(rest[i]=='('){m.isfunc=1;i++; while(rest[i]&&rest[i]!=')'){while(rest[i]==' '||rest[i]==',')i++;
      if(rest[i]==')')break; int ps=i; if(!strncmp(rest+i,"...",3)){m.variadic=1;strcpy(m.params[m.np++],"__VA_ARGS__");i+=3;continue;}
      while(idc((unsigned char)rest[i]))i++; int pl=i-ps; if(pl>0&&m.np<16){memcpy(m.params[m.np],rest+ps,(size_t)pl);m.params[m.np][pl]=0;m.np++;}}
    if(rest[i]==')')i++;}
  while(rest[i]==' ')i++;
  strncpy(m.body,rest+i,sizeof m.body-1);
  undef_macro(m.name); if(NM<1024) M[NM++]=m;
}

/* Resolve `name` by trying each search dir in order (the -I path + the source dir); a quoted and an
 * angle include share the path here (a driver MVP). Returns the byte count, or -1 if not found. */
static int read_file_dirs(const char *const *dirs, int ndirs, const char *name, char *out, size_t cap) {
  char path[1024];
  for(int d=0; d<ndirs; d++){ const char *base=dirs[d];
    if(base&&base[0]) snprintf(path,sizeof path,"%s/%s",base,name); else snprintf(path,sizeof path,"%s",name);
    FILE *f=fopen(path,"rb"); if(f){ size_t n=fread(out,1,cap-1,f); out[n]=0; fclose(f); return (int)n; } }
  if(ndirs==0){ FILE *f=fopen(name,"rb"); if(f){ size_t n=fread(out,1,cap-1,f); out[n]=0; fclose(f); return (int)n; } }
  return -1;
}

/* The main directive/text loop. Does NOT reset the macro table -- macros persist across #includes
 * (matching cpp.py, where the same Preprocessor processes nested files); the conditional stack is
 * per-call (a header's #if is self-contained). Recurses into included files, appending to `out` at
 * `*w` and sharing the macro table. */
static int g_cpp_depth;
static int cpp_process(const char *src, const char *curfile, const char *const *dirs, int ndirs,
                       char *out, size_t outcap, size_t *w, char *err, size_t errcap) {
  /* a conditional stack: active flag + taken flag (parent == all enclosing frames active) */
  struct { int active, taken, parent; } cs[64]; int ncs=0;
  const char *p=src; char line[8192]; int presumed=1;
  char filebuf[1024]; const char *curf=curfile;    /* #line may repoint the current file at filebuf */
  while(*p){
    int L=0; while(*p&&*p!='\n'){ if(*p=='\\'&&p[1]=='\n'){p+=2;continue;} if(L<8190)line[L++]=*p; p++; }
    if(*p=='\n')p++; line[L]=0;
    g_cur_file=curf; g_cur_line=presumed; presumed++;   /* __FILE__/__LINE__ for this logical line */
    int q=0; while(line[q]==' '||line[q]=='\t')q++;
    if(line[q]=='#'){
      q++; while(line[q]==' ')q++; int ds=q; while(idc((unsigned char)line[q]))q++;
      char dir[32]; int dl=q-ds; if(dl>31)dl=31; memcpy(dir,line+ds,(size_t)dl);dir[dl]=0;
      const char *rest=line+q; while(*rest==' ')rest++;
      int parent=1; for(int k=0;k<ncs;k++) if(!cs[k].active){parent=0;break;}
      if(!strcmp(dir,"ifdef")||!strcmp(dir,"ifndef")||!strcmp(dir,"if")){
        int tk; if(!strcmp(dir,"ifdef")){char n[64];int j=0;ntok(rest,&j,n,sizeof n);tk=is_defined(n);}
        else if(!strcmp(dir,"ifndef")){char n[64];int j=0;ntok(rest,&j,n,sizeof n);tk=!is_defined(n);}
        else tk=eval_if(rest)!=0;
        if(ncs<64){cs[ncs].active=parent&&tk;cs[ncs].taken=tk;cs[ncs].parent=parent;ncs++;}
      } else if(!strcmp(dir,"endif")){ if(ncs)ncs--; }
      else if(!strcmp(dir,"else")||!strcmp(dir,"elif")||!strcmp(dir,"elifdef")||!strcmp(dir,"elifndef")){
        if(ncs){ int par=cs[ncs-1].parent; int take;
          if(!strcmp(dir,"else")) take=!cs[ncs-1].taken;
          else if(!strcmp(dir,"elifdef")){char n[64];int j=0;ntok(rest,&j,n,sizeof n);take=!cs[ncs-1].taken&&is_defined(n);}
          else if(!strcmp(dir,"elifndef")){char n[64];int j=0;ntok(rest,&j,n,sizeof n);take=!cs[ncs-1].taken&&!is_defined(n);}
          else take=!cs[ncs-1].taken&&par&&(eval_if(rest)!=0);
          cs[ncs-1].active=par&&take; cs[ncs-1].taken=cs[ncs-1].taken||take; }
      }
      else if(parent){                              /* parent == every enclosing frame is active */
        if(!strcmp(dir,"define")) define_macro(rest);
        else if(!strcmp(dir,"undef")){char n[64];int j=0;ntok(rest,&j,n,sizeof n);undef_macro(n);}
        else if(!strcmp(dir,"include")){
          char hd[8192]; int sys=(rest[0]=='<'); char nm[1024]; int j=0;
          while(rest[j]&&rest[j]!='>'&&rest[j]!='"'&&rest[j]!='<')j++; if(rest[j]=='<'||rest[j]=='"')j++;
          int s2=j; while(rest[j]&&rest[j]!='>'&&rest[j]!='"')j++; int nl=j-s2; if(nl>1023)nl=1023; memcpy(nm,rest+s2,(size_t)nl);nm[nl]=0;
          if(read_file_dirs(dirs,ndirs,nm,hd,sizeof hd)<0){ if(!sys){if(err)snprintf(err,errcap,"#include %s not found",nm);return 1;} }
          else { if(g_cpp_depth>64){if(err)snprintf(err,errcap,"#include nesting too deep");return 1;}
                 g_cpp_depth++; int rc=cpp_process(hd,nm,dirs,ndirs,out,outcap,w,err,errcap); g_cpp_depth--;
                 if(rc) return rc; }
        }
        else if(!strcmp(dir,"embed")){
          char nm[1024]; int j=0; while(rest[j]&&rest[j]!='>'&&rest[j]!='"'&&rest[j]!='<')j++;
          if(rest[j]=='<'||rest[j]=='"')j++; int s2=j; while(rest[j]&&rest[j]!='>'&&rest[j]!='"')j++;
          int nl=j-s2; if(nl>1023)nl=1023; memcpy(nm,rest+s2,(size_t)nl);nm[nl]=0;
          static char blob[1<<16]; int bn=read_file_dirs(dirs,ndirs,nm,blob,sizeof blob);
          if(bn<0){if(err)snprintf(err,errcap,"#embed %s not found",nm);return 1;}
          char num[16]; for(int b=0;b<bn;b++){snprintf(num,sizeof num,"%s%u",b?", ":"",(unsigned char)blob[b]);app(out,outcap,w,num);} }
        else if(!strcmp(dir,"line")){          /* #line N ["file"]: presumed line of the NEXT line
                                                  is N (decimal); optional new __FILE__. Operands are
                                                  macro-expanded first. */
          char ex[8192]; expand_line(rest,ex,sizeof ex); int j=0; char t[1024];
          if(ntok(ex,&j,t,sizeof t)=='n'){ presumed=atoi(t);
            if(ntok(ex,&j,t,sizeof t)=='s'){  /* strip the quotes + resolve \ escapes into filebuf */
              size_t fw=0; for(int z=1; t[z] && t[z]!='"' && fw<sizeof filebuf-1; z++){
                if(t[z]=='\\'&&t[z+1])z++; filebuf[fw++]=t[z]; }
              filebuf[fw]=0; curf=filebuf; } } }
        /* #error/#warning/#pragma ignored */
      }
      continue;
    }
    int active=1; for(int k=0;k<ncs;k++) if(!cs[k].active){active=0;break;}
    if(active){ char ex[8192]; expand_line(line,ex,sizeof ex); app(out,outcap,w,ex); app(out,outcap,w,"\n"); }
  }
  if(ncs){if(err)snprintf(err,errcap,"unterminated #if");return 1;}
  return 0;
}

/* __DATE__ ("Mmm dd yyyy", space-padded day) + __TIME__ ("hh:mm:ss"). Frozen from SOURCE_DATE_EPOCH
 * (the reproducible-builds convention, UTC) when it is a plain integer, else the current UTC time --
 * the exact convention cpp.py uses, so the two rails agree byte-for-byte whenever the epoch is set. */
static int all_digits(const char *s) {
  if (!s || !*s) return 0;
  for (; *s; s++) if (*s < '0' || *s > '9') return 0;
  return 1;
}
static void cpp_datetime(char *datebuf, char *timebuf) {
  static const char *mon[] = {"Jan","Feb","Mar","Apr","May","Jun",
                              "Jul","Aug","Sep","Oct","Nov","Dec"};
  const char *e = getenv("SOURCE_DATE_EPOCH");
  time_t t = all_digits(e) ? (time_t)strtoll(e, NULL, 10) : time(NULL);
  struct tm *m = gmtime(&t);
  if (!m) { snprintf(datebuf, 16, "Jan  1 1970"); snprintf(timebuf, 16, "00:00:00"); return; }
  snprintf(datebuf, 16, "%s %2d %d", mon[m->tm_mon], m->tm_mday, m->tm_year + 1900);
  snprintf(timebuf, 16, "%02d:%02d:%02d", m->tm_hour, m->tm_min, m->tm_sec);
}

/* The extended entry point: multiple include dirs (-I) + predefined macros (-D, each "name body").
 * Seeds the predefined + -D macros ONCE; cpp_process then keeps them across nested includes. */
int bcir_cpp_run_ex(const char *src, const char *const *dirs, int ndirs,
                    const char *const *defines, int ndefines,
                    char *out, size_t outcap, char *err, size_t errcap) {
  NM=0; if(err&&errcap)err[0]=0;
  define_macro("__STDC_VERSION__ 202311L"); define_macro("__STDC__ 1");
  define_macro("__STDC_HOSTED__ 1");
  { char db[16], tb[16]; cpp_datetime(db, tb); char def[48];
    snprintf(def, sizeof def, "__DATE__ \"%s\"", db); define_macro(def);
    snprintf(def, sizeof def, "__TIME__ \"%s\"", tb); define_macro(def); }
  for(int d=0; d<ndefines; d++) define_macro(defines[d]);
  out[0]=0; size_t w=0; g_cpp_depth=0;
  return cpp_process(src, "<source>", dirs, ndirs, out, outcap, &w, err, errcap);
}

int bcir_cpp_run(const char *src, const char *basedir, char *out, size_t outcap,
                 char *err, size_t errcap) {
  const char *d[1]; int nd=0; if(basedir&&basedir[0]){ d[0]=basedir; nd=1; }
  return bcir_cpp_run_ex(src, d, nd, NULL, 0, out, outcap, err, errcap);
}
