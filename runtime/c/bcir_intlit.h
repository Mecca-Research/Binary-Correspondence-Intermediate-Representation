/* bcir_intlit.h -- the C front's one reader of an integer constant (C11 6.4.4.1), shared by the lexer
 * (bcir_cfront.c) and the preprocessor's `#if` (bcir_cpp.c), so the two never read one constant two ways: the
 * twin's `#if` had read `0777` as 777 with `strtol` where the lexer, and the oracle's `#if`, read 511, and both
 * took any suffix (CF-SUFFIX). The oracle's counterpart is `clex.int_literal_parts`. It holds the reader of a character
 * constant too (`char_literal`, the oracle's `clex.char_constant_units`), which the `#if` reads by (CF-PPARITH).
 * Header-only: each includer
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
/* The one reason the lexer and the preprocessor refuse a byte past ASCII outside a literal for (CF-PPLIMITS; the
 * oracle's `clex.NONASCII`): an identifier here is ASCII, so `café` is no name -- the preprocessor had read the
 * name `caf` out of it, where the oracle read `café`, and the lexer handed the rest on as punctuators. */
static const char nonascii_outside[]="non-ASCII character outside a literal";
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

/* A character constant s[0..n) (C11 6.4.4.4, C23's `u8`), read as the oracle's `clex.char_constant_units` reads one:
 * its encoding prefix -- 0, 'L', 'u', 'U' or '8' (u8) -- and its code units, each an ASCII source character but `'` and
 * `\` (a space, a tab, a vertical tab, a form feed or a printable one) or a simple, octal or hexadecimal escape, its
 * value within its type's code unit: 8 bits for a plain and a `u8` one, 16 for `u`, 32 for `U`, `wchar_bits` (the
 * target's `wchar_t`) for `L`. A plain one may hold several (a multi-character constant; `units` keeps the last 4, as
 * an `int` packs them), a prefixed one one. NULL, or `char_bad` for any other: no character, more than one behind a
 * prefix, an escape past the code unit, a universal character name, an escape C does not define, a source character
 * past ASCII (CF-PPARITH). */
typedef struct { int prefix, n; unsigned long units[4]; } charlit;
static const char char_bad[]="unsupported character constant";
static inline const char *char_literal(const char *s, int n, int wchar_bits, charlit *o){
  int i=0, e;
  unsigned long long limit;
  memset(o,0,sizeof *o);
  if(n>=2 && s[0]=='u' && s[1]=='8'){ o->prefix='8'; i=2; }
  else if(n>=1 && (s[0]=='L'||s[0]=='u'||s[0]=='U')){ o->prefix=s[0]; i=1; }
  if(n-i<2 || s[i]!='\'' || s[n-1]!='\'') return char_bad;
  limit = 1ull << (o->prefix=='u' ? 16 : o->prefix=='U' ? 32 : o->prefix=='L' ? wchar_bits : 8);
  for(i++, e=n-1; i<e; ){
    unsigned long long v=0; char ch=s[i];
    if(ch=='\\'){ char x = i+1<e ? s[i+1] : 0;
      switch(x){
      case '\'': v=39; i+=2; break;  case '"': v=34; i+=2; break;  case '?': v=63; i+=2; break;
      case '\\': v=92; i+=2; break;  case 'a': v=7; i+=2; break;   case 'b': v=8; i+=2; break;
      case 'f': v=12; i+=2; break;   case 'n': v=10; i+=2; break;  case 'r': v=13; i+=2; break;
      case 't': v=9; i+=2; break;    case 'v': v=11; i+=2; break;
      case 'x': { int j=i+2;
        for(; j<e; j++){ int c=s[j], d = c>='0'&&c<='9' ? c-'0' : (c|0x20)>='a'&&(c|0x20)<='f' ? (c|0x20)-'a'+10 : -1;
          if(d<0) break;
          if(v<limit) v=v*16u+(unsigned)d; }        /* past the code unit it stays past it */
        if(j==i+2) return char_bad;
        i=j; break; }
      default:
        if(x>='0' && x<='7'){ int j=i+1;
          for(; j<e && j<i+4 && s[j]>='0' && s[j]<='7'; j++) v=v*8u+(unsigned)(s[j]-'0');
          i=j; break; }
        return char_bad;
      }
    } else if(ch=='\t'||ch=='\v'||ch=='\f'||(ch>=' ' && ch<='~' && ch!='\'' && ch!='\\')){ v=(unsigned char)ch; i++; }
    else return char_bad;
    if(v>=limit) return char_bad;
    if(o->n==4){ memmove(o->units,o->units+1,3*sizeof o->units[0]); o->n=3; }
    o->units[o->n++]=(unsigned long)v;
    if(o->prefix && o->n>1) return char_bad;
  }
  return o->n ? NULL : char_bad;
}

#endif /* BCIR_INTLIT_H */
