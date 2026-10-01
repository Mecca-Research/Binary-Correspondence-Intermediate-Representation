/* bcir_intlit.h -- the C front's one reader of an integer constant (C11 6.4.4.1), shared by the lexer
 * (bcir_cfront.c) and the preprocessor's `#if` (bcir_cpp.c), so the two never read one constant two ways: the
 * twin's `#if` had read `0777` as 777 with `strtol` where the lexer, and the oracle's `#if`, read 511, and both
 * took any suffix (CF-SUFFIX). The oracle's counterpart is `clex.int_literal_parts`. Header-only: each includer
 * gets its own static copy, so neither translation unit links against the other. */
#ifndef BCIR_INTLIT_H
#define BCIR_INTLIT_H

#include <limits.h>
#include <string.h>

/* An integer constant s[0..n) (C11 6.4.4.1), read as the oracle's `clex.int_literal_parts` reads it: the C23 `'`
 * separators dropped, the trailing run of `u`/`U`/`l`/`L` its suffix, then `0x` hex, `0b` binary, a leading `0`
 * octal, else decimal, each digit checked against its base. Its value exactly -- an unsigned constant past
 * LLONG_MAX keeps its 64 bits (it had saturated at LLONG_MAX, and an octal constant was read as decimal) -- whether
 * it is decimal, and its suffix's `u` and count of `l`. NULL, or the reason it is refused: a digit its base does not
 * have, or a value no type in its list can hold -- past ULLONG_MAX, or a decimal one without `u` past LLONG_MAX,
 * whose type C leaves to the implementation (GCC's `__int128`, Clang's `unsigned long long`). */
typedef struct { unsigned long long v; int decimal, u, lr, nsuf; char suf[49]; } intlit;
static const char int_bad[]="invalid integer literal", int_bad_suffix[]="invalid suffix",
                  int_too_large[]="an integer constant too large for every type its base and suffix allow";
/* Whether suf[0..n) -- the trailing run of `u`/`U`/`l`/`L` -- is a suffix C spells (C11 6.4.4.1p1): at most one `u`
 * or `U`, first or last, around nothing, one `l`/`L`, or `ll`/`LL`; never `lL`, `uu` or `lul`, which Clang and GCC
 * refuse (the oracle's `clex.int_suffix_ok`, CF-SUFFIX). */
static inline int int_suffix_ok(const char *suf, int n){
  if(n>0 && (suf[0]=='u'||suf[0]=='U')){ suf++; n--; }
  else if(n>0 && (suf[n-1]=='u'||suf[n-1]=='U')) n--;
  return n==0 || (n==1 && (suf[0]=='l'||suf[0]=='L'))
         || (n==2 && ((suf[0]=='l'&&suf[1]=='l')||(suf[0]=='L'&&suf[1]=='L')));
}
static inline const char *int_literal(const char *s, int n, intlit *o){
  int e=n, nb=0, base=10, skip=0, nd=0, over=0, k=0, ns=0; char c0=0, c1=0;
  memset(o,0,sizeof *o);
  while(e>0 && (s[e-1]=='u'||s[e-1]=='U'||s[e-1]=='l'||s[e-1]=='L'||s[e-1]=='\'')){   /* the suffix */
    if(s[e-1]=='u'||s[e-1]=='U') o->u=1; else if(s[e-1]!='\'') o->lr++;
    e--; }
  for(int i=e;i<n;i++) if(s[i]!='\''){ if(ns<48) o->suf[ns]=s[i]; ns++; }   /* its letters, the separators dropped */
  o->nsuf = ns<48 ? ns : 48;
  if(ns>3 || !int_suffix_ok(o->suf,ns)) return int_bad_suffix;   /* (no suffix C spells has more than 3 letters) */
  for(int i=0;i<e;i++) if(s[i]!='\''){ if(nb==0) c0=s[i]; else if(nb==1) c1=s[i]; nb++; }
  if(c0=='0' && (c1=='x'||c1=='X')){ base=16; skip=2; }
  else if(c0=='0' && (c1=='b'||c1=='B')){ base=2; skip=2; }
  else if(c0=='0' && nb>1){ base=8; skip=1; }
  o->decimal=base==10;
  for(int i=0;i<e;i++){ char ch=s[i]; int d;
    if(ch=='\'' || k++<skip) continue;
    d = ch>='0'&&ch<='9' ? ch-'0' : (ch|0x20)>='a'&&(ch|0x20)<='f' ? (ch|0x20)-'a'+10 : base;
    if(d>=base) return int_bad;
    if(o->v > (ULLONG_MAX-(unsigned long long)d)/(unsigned long long)base) over=1;
    else o->v=o->v*(unsigned long long)base+(unsigned long long)d;
    nd++; }
  if(!nd) return int_bad;
  if(over || (o->decimal && !o->u && o->v>(unsigned long long)LLONG_MAX)) return int_too_large;
  return NULL;
}

#endif /* BCIR_INTLIT_H */
