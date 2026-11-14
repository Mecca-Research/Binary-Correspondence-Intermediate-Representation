/*===- bcir_cpp.c - the BCIR C preprocessor (L7) ---------------------------===*/
#include "bcir_cpp.h"
#include "bcir_host_alloc.h"
#include "bcir_intlit.h"

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BCIR_CPP_MAX_FILE_BYTES ((size_t)16u << 20)
#define BCIR_CPP_MAX_SOURCE_BYTES ((size_t)64u << 20)
#define BCIR_CPP_MAX_DIRS 64
#define BCIR_CPP_MAX_DEFINES 1024

/* --- macro table --------------------------------------------------------- */
/* A macro name or parameter is at most BCIR_CPP_NAME_MAX characters (CF-LIMITS): C11 5.2.4.1's 63 significant
 * initial characters, the bound both lexers hold an identifier to (CF-BUF; `IDENT_MAX` in cfront/clex.py). A macro
 * name never reaches a lexer -- it is replaced here -- so a longer one is refused wherever a directive reads it, for
 * one reason, as the oracle's `cpp.py` refuses it. */
#define BCIR_CPP_NAME_MAX 63
#define BCIR_CPP_NAME (BCIR_CPP_NAME_MAX + 1)
/* A logical line is at most BCIR_CPP_LINE - 2 bytes (`preprocessor line too long`), and every buffer that holds one
 * token, one macro argument or one substitution holds a line's worth (CF-PPLIMITS): a token is no longer than the
 * line it is in, so none is refused for its own length. They had held 256 bytes -- a string literal, a stringized
 * argument or a `__has_attribute` operand of 256 characters or more was refused (`preprocessor token too long`)
 * where the oracle took it -- and an argument 1 KiB, a substitution 2 KiB. */
#define BCIR_CPP_LINE 8192
#define BCIR_CPP_ARGS 16
#define BCIR_CPP_LONG_NAME "macro name is too long"
typedef struct {
  char name[BCIR_CPP_NAME];
  int isfunc;
  int np; char params[16][BCIR_CPP_NAME]; int variadic;
  char body[1024];
} Macro;

/* A token in macro replacement (the engine below): its spelling (`len` bytes at `off` in the text `src` names), its
 * hide set (an interned set of macro indices, 0 the empty one), its kind as `ntok` reads it ('i', 'n', 's', 'p'; 0 a
 * placemarker) and whether white space came before it. */
typedef struct { uint32_t off, len, hs; uint16_t src; uint8_t kind, ws; } PPTok;
typedef struct { PPTok *v; size_t n, cap; } PPToks;
/* An interned hide set: its sorted elements at hs_el[start..start+len), their hash, its slot in hs_tab. */
typedef struct { uint32_t start, len, hash, slot; } PPHide;

typedef struct CppState {
  bcir_host_allocator allocator;
  bcir_host_arena scratch;
  Macro macros[1024];
  int macro_count;
  const char *limit_error;
  char constant_error[96];   /* the text of a malformed `#if` constant's refusal (`lit`), which limit_error names */
  const char *current_file;
  int current_line;
  int include_depth;
  int long_size, pointer_size, long_double, msvc, flt_eval;   /* the rest of the target the standard headers the
                                                * preprocessor supplies read (`bcir_cpp_context_set_target`) */
  char builtin_text[16384];                  /* one of those headers, built where it is included (never nested) */
  int char_signed, wchar_size, wchar_signed;   /* the target's character types, which a `#if` reads a character
                                                * constant by (CF-PPARITH): kept across runs and resets */
  char conditional_buffer[BCIR_CPP_LINE];
  /* macro replacement (expand_text): the pending tokens, the outputs, the arguments, the replacement being built and
   * the replacement lists being read -- each a stack the nested prescans grow and give back -- the spellings `#`, `##`,
   * `__LINE__` and `__FILE__` make, the expansion as text, the text expanded and the work done; the hide sets */
  PPToks ppx_stack, ppx_out, ppx_args, ppx_sub, ppx_body;
  char *ppx_pool; size_t ppx_pool_n, ppx_pool_cap;
  char *ppx_text; size_t ppx_text_cap;
  const char *ppx_in;
  size_t ppx_work;
  PPHide *hs_v; size_t hs_n, hs_cap;
  uint16_t *hs_el; size_t hs_el_n, hs_el_cap;
  uint32_t *hs_tab; size_t hs_tab_cap;
  uint16_t hs_tmp[1024];                     /* a set being made: at most every macro */
  char ppx_tok[BCIR_CPP_LINE];               /* ntok's copy of the token it reads */
  char line_token[BCIR_CPP_LINE];            /* a `#line` operand (cpp_process recurses per #include: no frame) */
  char *out_buf;                             /* the caller's output buffer, and whether the output filled it */
  int out_full;
  int incomplete;                            /* a source line's expansion ended inside an invocation (1) or on a
                                              * function-like macro's name (2): the oracle's `_Incomplete` (CSUB-1) */
  char join_line[BCIR_CPP_LINE];             /* a line an invocation continues onto (cpp_process, never nested in
                                              * an expansion) */
} CppState;

/* Fixed-size scratch buffers are an implementation limit, never permission to truncate a
 * translation unit.  The old preprocessor silently dropped text when one filled (and one long
 * macro-parameter name overflowed Macro.params outright).  Record the first limit violation and
 * make the public entry return a clean diagnostic after the current bounded operation unwinds. */
/* The white space C reads as such (6.4p3) between tokens and before a directive's `#`: a space, a tab, a form feed, a
 * vertical tab -- the oracle's `_ASCII_SPACE`, which holds a carriage return too: a lone one is white space on both
 * rails, as both lexers skip it (a CRLF end of line is a new-line, read as one below). Once a space or a tab alone,
 * so `\f` in code was an expression's end here and lowered on the oracle (CF-PPSPLITS). */
static int pp_space(int ch) { return ch == ' ' || ch == '\t' || ch == '\f' || ch == '\v' || ch == '\r'; }
static void cpp_limit(CppState *state, const char *message) {
  if (state && !state->limit_error) state->limit_error = message;
}

static int find_macro(CppState *state, const char *n, int len) {
  for (int i = 0; i < state->macro_count; i++)
    if ((int)strlen(state->macros[i].name) == len &&
        !strncmp(state->macros[i].name, n, (size_t)len)) return i;
  return -1;
}
static void undef_macro(CppState *state, const char *n) {
  int i = find_macro(state, n, (int)strlen(n));
  if (i >= 0) state->macros[i] = state->macros[--state->macro_count];
}

/* __FILE__/__LINE__: the file currently being processed + its 1-based logical line number. These
 * are dynamic predefined macros (not stored in the table); a nested #include saves/restores them. */
/* The __has_* feature-test operators usable in #if (and reported as `defined`). */
static int is_has_op(const char *n) {
  return !strcmp(n, "__has_include") || !strcmp(n, "__has_embed") ||
         !strcmp(n, "__has_attribute") || !strcmp(n, "__has_builtin") ||
         !strcmp(n, "__has_c_attribute");
}

/* A macro is "defined" (for #ifdef / defined()) if it is in the table, a dynamic predefined, or a
 * __has_* operator. */
static int is_defined(CppState *state, const char *n) {
  if (find_macro(state, n, (int)strlen(n)) >= 0) return 1;
  return !strcmp(n, "__FILE__") || !strcmp(n, "__LINE__") || is_has_op(n);
}

/* __has_attribute(x): the L8 ABI attributes (GCC __x__ spelling accepted), conservatively. */
static int has_attribute(const char *n) {
  char b[64]; int j = 0;
  for (const char *p = n; *p && j < 63; p++) b[j++] = *p;
  b[j] = 0;
  char *s = b; while (*s == '_') s++;                /* strip surrounding underscores */
  size_t L = strlen(s); while (L > 0 && s[L-1] == '_') L--; s[L] = 0;
  return !strcmp(s, "packed") || !strcmp(s, "aligned");
}

/* --- preprocessing tokenizer --------------------------------------------- */
static int idc(int c) { return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                               (c >= '0' && c <= '9'); }
static int id0(int c) { return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

/* The length of the identifier at s[*i], past the spaces before it (*i is moved past them); 0 when none starts there. */
static int ident_at(const char *s, int *i) {
  int n = 0;
  while (pp_space(s[*i])) (*i)++;
  if (id0((unsigned char)s[*i])) while (idc((unsigned char)s[*i + n])) n++;
  return n;
}
/* A macro name a directive reads -- the one #define or -D defines, #undef removes, #ifdef/#ifndef/#elifdef/#elifndef
 * or defined tests: the identifier at s[*i] into `out` (BCIR_CPP_NAME bytes), *i past it. 0, with the refusal
 * recorded, when none starts there (`why`, the directive's own reason), when it is longer than BCIR_CPP_NAME_MAX,
 * at any length (CF-LIMITS; the oracle's `_macro_name`), or when a byte past ASCII runs on from it: beside `#define
 * caf 5`, `#ifdef café` had tested `caf` here and `café` on the oracle (CF-PPLIMITS). */
static int macro_name(CppState *state, const char *s, int *i, char *out, const char *why) {
  int n = ident_at(s, i);
  if (!n) { cpp_limit(state, why); return 0; }
  *i += n;
  if (n > BCIR_CPP_NAME_MAX) { cpp_limit(state, BCIR_CPP_LONG_NAME); return 0; }
  if ((unsigned char)s[*i] >= 0x80) { cpp_limit(state, nonascii_outside); return 0; }
  memcpy(out, s + *i - n, (size_t)n); out[n] = 0;
  return 1;
}

/* Every multi-character punctuator (C 6.4.6), longest first: `++` is one token, never two `+`. */
static const char *const punct_ops[] = {"<<=", ">>=", "...", "->", "##", "<<", ">>", "<=", ">=", "==", "!=",
                                        "&&", "||", "++", "--", "+=", "-=", "*=", "/=", "%=", "&=", "|=",
                                        "^=", 0};

/* next token from s[*i]; copies into out; returns 'i' ident, 'n' number, 's' string,
 * 'p' punct, 0 end. skips leading spaces. */
/* A string literal or character constant from its opening quote at s[*i] through its closing one, appended at out[*j]. */
static void ntok_quoted(CppState *state, const char *s, int *i, char *out, int cap, int *j) {
  char q = s[*i];
  if (*j < cap - 1) out[(*j)++] = s[*i]; else cpp_limit(state, "preprocessor token too long");
  (*i)++;
  while (s[*i] && s[*i] != q) {
    if (s[*i] == '\\' && s[*i+1]) {
      if (*j < cap - 1) out[(*j)++] = s[*i]; else cpp_limit(state, "preprocessor token too long");
      (*i)++;
    }
    if (*j < cap - 1) out[(*j)++] = s[*i]; else cpp_limit(state, "preprocessor token too long");
    (*i)++;
  }
  if (s[*i] == q) {
    if (*j < cap - 1) out[(*j)++] = s[*i]; else cpp_limit(state, "preprocessor token too long");
    (*i)++;
  }
}
static int ntok(CppState *state, const char *s, int *i, char *out, int cap) {
  if (!s || !i || !out || cap < 2) { cpp_limit(state, "invalid preprocessor token buffer"); return 0; }
  while (pp_space(s[*i])) (*i)++;
  int c = (unsigned char)s[*i];
  if (!c) return 0;
  int j = 0;
  if (id0(c)) {
    while (idc((unsigned char)s[*i])) {
      if (j < cap - 1) out[j++] = s[*i]; else cpp_limit(state, "preprocessor token too long");
      (*i)++;
    }
    out[j] = 0;
    /* An encoding prefix and the character constant it begins are one token (C11 6.4.4.4), so its `L` is no macro
     * name and a `#if` reads it whole (CF-PPARITH; the oracle's `_TOKEN_RE`). */
    if (s[*i] == '\'' && (!strcmp(out, "L") || !strcmp(out, "u") || !strcmp(out, "U") || !strcmp(out, "u8"))) {
      ntok_quoted(state, s, i, out, cap, &j); out[j] = 0; return 's'; }
    return 'i';
  }
  if ((c >= '0' && c <= '9') || (c == '.' && s[*i+1] >= '0' && s[*i+1] <= '9')) {   /* a pp-number (a digit, or
                                                          * `.` and a digit, first): digits, '.', idents, an
                                                          * e/E/p/P binary/decimal exponent's +/- sign, and C23's
                                                          * `'` separators (6.4.8), read as the oracle's `_TOKEN_RE`
                                                          * reads them: the twin had ended the number at one and read
                                                          * a character constant from it (CF-PPARITH) */
    while (idc((unsigned char)s[*i]) || s[*i] == '.' || s[*i] == '\'') {
      char d = s[*i];
      if (j < cap - 1) out[j++] = d; else cpp_limit(state, "preprocessor token too long");
      (*i)++;
      if ((d=='e'||d=='E'||d=='p'||d=='P') && (s[*i]=='+'||s[*i]=='-')) {
        if (j < cap - 1) out[j++] = s[*i]; else cpp_limit(state, "preprocessor token too long");
        (*i)++;
      }
    }
    out[j] = 0; return 'n';
  }
  if (c == '"' || c == '\'') { ntok_quoted(state, s, i, out, cap, &j); out[j] = 0; return 's'; }
  for (int k = 0; punct_ops[k]; k++) { int L = (int)strlen(punct_ops[k]);
    if (!strncmp(s + *i, punct_ops[k], (size_t)L)) {
      memcpy(out, punct_ops[k], (size_t)L); out[L] = 0; *i += L; return 'p'; } }
  out[0] = s[(*i)++]; out[1] = 0; return 'p';
}

/* Whether the next token at s[*i] (past spaces and tabs, as ntok skips them) is the punctuator `c` -- `(` or `)`,
 * which no longer punctuator begins -- and if so *i past it. It copies nothing: a peek through a short buffer had
 * refused whatever long token came next (CF-PPLIMITS). */
static int next_is(const char *s, int *i, char c) {
  int j = *i;
  while (pp_space(s[j])) j++;
  if (s[j] != c) return 0;
  *i = j + 1;
  return 1;
}

/* The reasons both rails give when a function-like macro's invocation runs past its line (C11 6.10.3p10), the
 * oracle's `cpp.UNTERMINATED_ARGS`, `DIRECTIVE_IN_ARGS` and `LINE_IN_CONTINUED_ARGS` word for word (CSUB-1). */
#define PP_UNTERMINATED_ARGS "unterminated macro argument list"
#define PP_DIRECTIVE_IN_ARGS "preprocessing directive inside macro arguments"
#define PP_LINE_IN_CONTINUED "__LINE__ on a continued line of a macro invocation"

/* --- macro expansion (expand a line until stable) ------------------------ */
static void app(CppState *state, char *o, size_t cap, size_t *w, const char *s) {
  size_t n = strlen(s);
  if (*w >= cap || n >= cap - *w) {
    if (o == state->out_buf) state->out_full = 1;   /* the output, not a line: a larger buffer would hold it */
    cpp_limit(state, "preprocessed output too large");
    return;
  }
  memcpy(o + *w, s, n); *w += n; o[*w] = 0;
}
/* The length of the punctuator `s` begins with: its multi-character one (longest first), else 1. */
static int punct_len(const char *s) {
  for (int k = 0; punct_ops[k]; k++) { size_t L = strlen(punct_ops[k]); if (!strncmp(s, punct_ops[k], L)) return (int)L; }
  return 1;
}
/* Whether the text `t`, written right after a token ending with `a`, would lex as part of another token
 * (maximal munch, C 6.4p4): two words run together, a comment opens (`y / *p` without its space, 6.4.9), a
 * punctuator runs on into a longer one (`a + ++g` spelled `a+++g` is `(a++) + g`), two `.` start an ellipsis
 * (`..` is no punctuator, so maximal munch over two tokens cannot see it), a pp-number runs on
 * through a `.`, an exponent's sign or a C23 digit separator `'` (`anum`: the token before is one; `1.5 'c'` had
 * been spelled `1.5'c'`, one pp-number, CSUB-1), or a `.` begins a number. A space
 * between them keeps them two. `prevp` is the token before when it is known: itself when a punctuator,
 * `""` when not one; NULL (unknown) keeps two punctuators apart whenever they might run on. The oracle's
 * `_pastes` (cpp.py), byte for byte where the token is known. */
static int pastes(const char *prevp, char a, const char *t, int anum) {
  static const char pairs[][3] = {"<<", "<=", ">>", ">=", "->", "++", "--", "==", "!=", "&&", "||", "##",
                                  "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^="};
  char b = t[0];
  if (idc((unsigned char)a) && idc((unsigned char)b)) return 1;
  if ((a == '/' && (b == '*' || b == '/')) || (a == '.' && b == '.')) return 1;   /* a comment; an ellipsis */
  for (size_t k = 0; k < sizeof pairs / sizeof pairs[0]; k++)
    if (pairs[k][0] == a && pairs[k][1] == b) {
      if (!prevp) return 1;
      if (!prevp[0]) break;                          /* a word, number or literal: no punctuator runs on */
      char st[8]; snprintf(st, sizeof st, "%.3s%.2s", prevp, t);
      if (punct_len(st) > (int)strlen(prevp)) return 1;
      break;
    }
  if (anum && (b == '.' || b == '\'' || ((a == 'e' || a == 'E' || a == 'p' || a == 'P') && (b == '+' || b == '-'))))
    return 1;
  if (b == '\'' && (a == 'L' || a == 'u' || a == 'U' || a == '8'))   /* a word that may be an encoding prefix */
    return 1;
  return a == '.' && b >= '0' && b <= '9';
}

/* --- macro replacement: Prosser's algorithm (C11 6.10.3.4), the oracle's `_rescan` (CSUB-1) -------------------- */
/* One expansion -- a source line, or the operand of a `#if`, `#elif`, `#line` or `#error` -- is a token stack rescanned
 * once, left to right: a macro's replacement is pushed back and rescanned with the tokens after it (6.10.3.4p1), so a
 * replacement ending in a function-like macro's name is invoked by a `(` that follows it, and a token once passed is
 * output and never looked at again. Each token carries a hide set, the macros it may no longer be replaced by: a token
 * whose set holds its own macro is painted and stays itself (6.10.3.4p2). An object-like macro's replacement adds its
 * name to the set of the name's token; a function-like one's adds it to the intersection of the name's and the closing
 * `)`'s. A parameter is replaced by its argument, completely macro-replaced first as if it were the rest of the file
 * (the prescan, 6.10.3.1), except as an operand of `#` or `##`, where it is the argument as written. Each token also
 * carries whether white space came before it, which is what `#` keeps of the spacing (6.10.3.2p2).
 *
 * This replaces an engine that rewrote the line until it stopped changing, with no hide sets and no prescan:
 * `f(1)(2)` of `#define f(x) (x) + f` came out `(1) + (2) + f` here where C gives `(1) + f(2)`, `f(f)(y)` of `#define
 * f(x) x` came out `y` for `f(y)`, `f LP 1)` of `#define LP (` invoked `f`, `#define EPERM EPERM` was refused (`macro
 * expansion nesting too deep`), a two-level `XSTR(__LINE__)` spelled `"__LINE__"`, `#define H # x` spelled `""`, and a
 * line expanding past 8 KiB was refused. The oracle's `cpp.py` is this engine step for step -- the same hide-set
 * operations in the same order, the same work charged at the same events -- so the two rails take and refuse alike.
 *
 * Bounds, the oracle's in the same words: an argument's prescan nests at most PPX_MAX_DEPTH deep (`macro expansion
 * nesting too deep`), and one expansion does at most PPX_MAX_WORK units of work (`macro expansion is too large`): each
 * token taken from what is pending, each token written into a replacement, each byte of a token a rescan outputs or a
 * `#` or `##` spells, and each element of a hide set first made count one. The work bounds the time and every buffer
 * below, which grow from the context's allocator as they need and are kept for the next expansion. */
#define PPX_MAX_DEPTH 63
#define PPX_MAX_WORK ((size_t)1u << 22)
#define PP_EXPANSION_TOO_LARGE "macro expansion is too large"
#define PP_EXPANSION_TOO_DEEP "macro expansion nesting too deep"

/* Where a token's spelling is: the text expanded, the pool of spellings made here, or macro `src - PPX_BODY`'s
 * replacement list. */
#define PPX_IN 0u
#define PPX_POOL 1u
#define PPX_BODY 2u

static int ppx_grow(CppState *st, void **v, size_t *cap, size_t need, size_t elem) {
  size_t c = *cap ? *cap : 256u;
  void *q;
  if (need <= *cap) return 1;
  while (c < need) {
    if (c > ((size_t)-1) / 4u / elem) { cpp_limit(st, PP_EXPANSION_TOO_LARGE); return 0; }
    c *= 2u;
  }
  q = bcir_host_allocate(&st->allocator, c * elem);
  if (!q) { cpp_limit(st, "oom"); return 0; }
  if (*v && *cap) memcpy(q, *v, *cap * elem);
  bcir_host_deallocate(&st->allocator, *v);
  *v = q; *cap = c;
  return 1;
}
static int toks_room(CppState *st, PPToks *a, size_t extra) {
  return ppx_grow(st, (void **)&a->v, &a->cap, a->n + extra, sizeof *a->v);
}
static int toks_push(CppState *st, PPToks *a, PPTok t) {
  if (!toks_room(st, a, 1u)) return 0;
  a->v[a->n++] = t;
  return 1;
}

static int ppx_charge(CppState *st, size_t n) {
  st->ppx_work += n;
  if (st->ppx_work > PPX_MAX_WORK) { cpp_limit(st, PP_EXPANSION_TOO_LARGE); return 0; }
  return 1;
}
static int ppx_stopped(const CppState *st) { return st->limit_error != NULL || st->incomplete != 0; }

static const char *ppx_s(const CppState *st, const PPTok *t) {
  if (t->src == PPX_IN) return st->ppx_in + t->off;
  if (t->src == PPX_POOL) return st->ppx_pool + t->off;
  return st->macros[t->src - PPX_BODY].body + t->off;
}
static int ppx_is(const CppState *st, const PPTok *t, const char *lit) {
  size_t n = strlen(lit);
  return t->len == n && !memcmp(ppx_s(st, t), lit, n);
}
/* The kind `ntok` gives the token spelled `s` (n bytes), read off its first characters as the oracle reads a token it
 * made (`_is_id`, `_is_ppnum`, a literal's quote): 's' a literal (a prefixed character constant too), 'n' a pp-number,
 * 'i' a word, 'p' anything else. */
static uint8_t ppx_kind(const char *s, size_t n) {
  unsigned char c = n ? (unsigned char)s[0] : 0u;
  if (c == '"' || c == '\'') return 's';
  if ((n > 1 && (c == 'L' || c == 'u' || c == 'U') && s[1] == '\'') || (n > 2 && c == 'u' && s[1] == '8' && s[2] == '\''))
    return 's';
  if ((c >= '0' && c <= '9') || (c == '.' && n > 1 && s[1] >= '0' && s[1] <= '9')) return 'n';
  if (id0(c)) return 'i';
  return 'p';
}
/* A new spelling in the pool: `a` then `b` (the pool may move, so both are copied by offset), its token kind. */
static int ppx_spell(CppState *st, const PPTok *a, const PPTok *b, PPTok *out) {
  size_t la = a->len, lb = b ? b->len : 0u, at = st->ppx_pool_n;
  PPTok ca = *a, cb = b ? *b : *a;
  if (!ppx_grow(st, (void **)&st->ppx_pool, &st->ppx_pool_cap, at + la + lb + 1u, 1u)) return 0;
  memcpy(st->ppx_pool + at, ppx_s(st, &ca), la);
  if (b) memcpy(st->ppx_pool + at + la, ppx_s(st, &cb), lb);
  st->ppx_pool[at + la + lb] = 0;
  st->ppx_pool_n = at + la + lb + 1u;
  out->src = PPX_POOL; out->off = (uint32_t)at; out->len = (uint32_t)(la + lb);
  out->kind = ppx_kind(st->ppx_pool + at, la + lb);
  return 1;
}
static int ppx_spell_text(CppState *st, const char *s, size_t n, PPTok *out) {
  size_t at = st->ppx_pool_n;
  if (!ppx_grow(st, (void **)&st->ppx_pool, &st->ppx_pool_cap, at + n + 1u, 1u)) return 0;
  memcpy(st->ppx_pool + at, s, n); st->ppx_pool[at + n] = 0;
  st->ppx_pool_n = at + n + 1u;
  out->src = PPX_POOL; out->off = (uint32_t)at; out->len = (uint32_t)n; out->kind = ppx_kind(s, n);
  return 1;
}

/* --- hide sets: sorted macro indices, interned per expansion (set 0 the empty one) --- */
static uint32_t hs_hash(const uint16_t *e, uint32_t n) {
  uint32_t h = 2166136261u;
  for (uint32_t k = 0; k < n; k++) { h = (h ^ (e[k] & 0xffu)) * 16777619u; h = (h ^ (e[k] >> 8)) * 16777619u; }
  return h;
}
static void hs_reset(CppState *st) {
  for (size_t k = 1; k < st->hs_n; k++) st->hs_tab[st->hs_v[k].slot] = 0u;
  st->hs_n = 1u; st->hs_el_n = 0u;
}
static int hs_place(CppState *st, uint32_t id) {
  size_t mask = st->hs_tab_cap - 1u, k = st->hs_v[id].hash & mask;
  while (st->hs_tab[k]) k = (k + 1u) & mask;
  st->hs_tab[k] = id + 1u; st->hs_v[id].slot = (uint32_t)k;
  return 1;
}
/* The id of the set of the `n` sorted indices in st->hs_tmp; a set not made before in this expansion is charged an
 * element each, as the oracle's `_hide` charges a new frozenset. */
static int hs_intern(CppState *st, uint32_t n, uint32_t *id) {
  const uint16_t *e = st->hs_tmp;
  uint32_t h;
  if (!n) { *id = 0u; return 1; }
  h = hs_hash(e, n);
  if (st->hs_tab_cap) {
    size_t mask = st->hs_tab_cap - 1u;
    for (size_t k = h & mask; st->hs_tab[k]; k = (k + 1u) & mask) {
      const PPHide *x = &st->hs_v[st->hs_tab[k] - 1u];
      if (x->hash == h && x->len == n && !memcmp(st->hs_el + x->start, e, n * sizeof *e)) {
        *id = st->hs_tab[k] - 1u; return 1;
      }
    }
  }
  if (!ppx_charge(st, n)) return 0;
  if (!ppx_grow(st, (void **)&st->hs_el, &st->hs_el_cap, st->hs_el_n + n, sizeof *st->hs_el)) return 0;
  if (!ppx_grow(st, (void **)&st->hs_v, &st->hs_cap, st->hs_n + 1u, sizeof *st->hs_v)) return 0;
  if ((st->hs_n + 1u) * 2u > st->hs_tab_cap) {          /* keep the table at most half full: rebuild it larger */
    size_t want = st->hs_tab_cap ? st->hs_tab_cap * 2u : 1024u;
    while ((st->hs_n + 1u) * 2u > want) want *= 2u;
    uint32_t *t = (uint32_t *)bcir_host_allocate_zeroed(&st->allocator, want * sizeof *t);
    if (!t) { cpp_limit(st, "oom"); return 0; }
    bcir_host_deallocate(&st->allocator, st->hs_tab);
    st->hs_tab = t; st->hs_tab_cap = want;
    for (size_t k = 1; k < st->hs_n; k++) hs_place(st, (uint32_t)k);
  }
  memcpy(st->hs_el + st->hs_el_n, e, n * sizeof *e);
  st->hs_v[st->hs_n].start = (uint32_t)st->hs_el_n; st->hs_v[st->hs_n].len = n; st->hs_v[st->hs_n].hash = h;
  st->hs_el_n += n;
  *id = (uint32_t)st->hs_n++;
  hs_place(st, *id);
  return 1;
}
static int hs_has(const CppState *st, uint32_t id, int m) {
  const uint16_t *e;
  uint32_t lo = 0u, hi;
  if (!id) return 0;
  e = st->hs_el + st->hs_v[id].start; hi = st->hs_v[id].len;
  while (lo < hi) { uint32_t mid = (lo + hi) / 2u; if (e[mid] < m) lo = mid + 1u; else hi = mid; }
  return lo < st->hs_v[id].len && e[lo] == m;
}
static int hs_add(CppState *st, uint32_t a, int m, uint32_t *out) {        /* a | {m} */
  const uint16_t *e = a ? st->hs_el + st->hs_v[a].start : NULL;
  uint32_t n = a ? st->hs_v[a].len : 0u, k = 0u, w = 0u;
  if (hs_has(st, a, m)) { *out = a; return 1; }
  while (k < n && e[k] < m) st->hs_tmp[w++] = e[k++];
  st->hs_tmp[w++] = (uint16_t)m;
  while (k < n) st->hs_tmp[w++] = e[k++];
  return hs_intern(st, w, out);
}
static int hs_and(CppState *st, uint32_t a, uint32_t b, uint32_t *out) {   /* a & b */
  uint32_t i = 0u, j = 0u, w = 0u, na, nb;
  const uint16_t *x, *y;
  if (a == b) { *out = a; return 1; }
  if (!a || !b) { *out = 0u; return 1; }
  x = st->hs_el + st->hs_v[a].start; y = st->hs_el + st->hs_v[b].start; na = st->hs_v[a].len; nb = st->hs_v[b].len;
  while (i < na && j < nb) {
    if (x[i] < y[j]) i++; else if (y[j] < x[i]) j++; else { st->hs_tmp[w++] = x[i]; i++; j++; }
  }
  return hs_intern(st, w, out);
}
static int hs_or(CppState *st, uint32_t a, uint32_t b, uint32_t *out) {    /* a | b */
  uint32_t i = 0u, j = 0u, w = 0u, na, nb;
  const uint16_t *x, *y;
  if (a == b || !b) { *out = a; return 1; }
  if (!a) { *out = b; return 1; }
  x = st->hs_el + st->hs_v[a].start; y = st->hs_el + st->hs_v[b].start; na = st->hs_v[a].len; nb = st->hs_v[b].len;
  while (i < na || j < nb) {
    if (j >= nb || (i < na && x[i] < y[j])) st->hs_tmp[w++] = x[i++];
    else if (i >= na || y[j] < x[i]) st->hs_tmp[w++] = y[j++];
    else { st->hs_tmp[w++] = x[i]; i++; j++; }
  }
  return hs_intern(st, w, out);
}

/* --- the rescan, the arguments and the replacement --- */
/* An invocation's arguments: each a run of st->ppx_args (only the first BCIR_CPP_ARGS are kept, the count is all of
 * them), whether white space came before each comma between them, and the closing `)`'s hide set. */
typedef struct { size_t start[BCIR_CPP_ARGS], end[BCIR_CPP_ARGS]; uint8_t comma_ws[BCIR_CPP_ARGS]; int n; uint32_t close; } PPArgs;
/* A parameter's two replacements: the argument as written, and -- made once, when first wanted -- prescanned. */
typedef struct { size_t raw0, raw1, pre0, pre1; int prescanned; } PPParam;
typedef struct { int mi, depth; PPParam p[BCIR_CPP_ARGS + 1]; } PPSubst;

static int ppx_rescan(CppState *st, size_t base, int top, int depth);

/* Pop the pending `(` and then through the `)` that closes it (the oracle's `_collect_args`): the arguments appended
 * to st->ppx_args. Running out of the level's tokens (those above `base`) is the next line's for a source line
 * (`top`: st->incomplete = 1), else refused. */
static int ppx_collect(CppState *st, size_t base, int top, PPArgs *a) {
  PPToks *S = &st->ppx_stack;
  int depth = 0;
  a->n = 0;
  if (!ppx_charge(st, 1u)) return 0;
  S->n--;                                                     /* the `(` */
  a->start[0] = st->ppx_args.n;
  while (S->n > base) {
    PPTok t;
    if (!ppx_charge(st, 1u)) return 0;
    t = S->v[--S->n];
    if (t.kind == 'p' && t.len == 1) {
      char c = ppx_s(st, &t)[0];
      if (c == '(') depth++;
      else if (c == ')') {
        if (!depth) {
          if (a->n < BCIR_CPP_ARGS) a->end[a->n] = st->ppx_args.n;
          a->n++; a->close = t.hs;
          return 1;
        }
        depth--;
      } else if (c == ',' && !depth) {
        if (a->n < BCIR_CPP_ARGS) {
          a->end[a->n] = st->ppx_args.n;
          a->comma_ws[a->n] = t.ws;
          if (a->n + 1 < BCIR_CPP_ARGS) a->start[a->n + 1] = st->ppx_args.n;
        }
        a->n++;
        continue;
      }
    }
    if (a->n < BCIR_CPP_ARGS && !toks_push(st, &st->ppx_args, t)) return 0;
  }
  if (top) { st->incomplete = 1; return 0; }
  cpp_limit(st, PP_UNTERMINATED_ARGS);
  return 0;
}

/* As many arguments as parameters (C11 6.10.3p4), the variadic part free to be empty, and never more than
 * BCIR_CPP_ARGS; `M()` of a macro of no parameters passes none (the oracle's `_check_arity`). */
static int ppx_arity(CppState *st, const Macro *m, const PPArgs *a) {
  int need = m->variadic ? m->np - 1 : m->np;
  int given = (a->n == 1 && a->end[0] == a->start[0] && need == 0) ? 0 : a->n;
  if (a->n > BCIR_CPP_ARGS || (!m->variadic && given > m->np)) { cpp_limit(st, "too many macro arguments"); return 0; }
  if (given < need) { cpp_limit(st, "too few macro arguments"); return 0; }
  return 1;
}

/* Append the run v[r0..r1) to the replacement being built (the oracle's `put`): it is charged a unit a token, its first
 * token takes the spacing `ws` (or the white space an empty run before it handed on, *carry), and an empty run hands
 * its white space on. `v` is st->ppx_args or st->ppx_out, or NULL when the run is already in place at the end of
 * st->ppx_sub (`__VA_OPT__`'s content). */
static int ppx_put(CppState *st, const PPToks *v, size_t r0, size_t r1, int ws, int *carry) {
  PPToks *O = &st->ppx_sub;
  size_t at;
  if (!ppx_charge(st, r1 - r0)) return 0;
  if (r0 == r1) { *carry = *carry || ws; return 1; }
  if (v) {
    if (!toks_room(st, O, r1 - r0)) return 0;
    at = O->n;
    memcpy(O->v + O->n, v->v + r0, (r1 - r0) * sizeof *O->v);
    O->n += r1 - r0;
  } else {
    at = r0;
  }
  O->v[at].ws = (uint8_t)(ws || *carry);
  *carry = 0;
  return 1;
}
static int ppx_put1(CppState *st, PPTok t, int ws, int *carry) {
  PPToks one; one.v = &t; one.n = one.cap = 1u;
  return ppx_put(st, &one, 0u, 1u, ws, carry);
}

/* The `#` spelling of the run v[r0..r1) (C11 6.10.3.2p2, the oracle's `_stringize`): a space where white space came
 * between two tokens, a `\` before each `"` and `\` of a literal, the whole in quotes. */
static int ppx_stringize(CppState *st, const PPToks *v, size_t r0, size_t r1, PPTok *out) {
  size_t need = 3u, at;
  char *p;
  for (size_t k = r0; k < r1; k++) {
    const PPTok *t = &v->v[k];
    const char *s = ppx_s(st, t);
    need += t->len + 1u;
    if (t->kind == 's' && t->len >= 2u) for (uint32_t z = 0; z < t->len; z++) need += (s[z] == '"' || s[z] == '\\');
  }
  at = st->ppx_pool_n;
  if (!ppx_grow(st, (void **)&st->ppx_pool, &st->ppx_pool_cap, at + need, 1u)) return 0;
  p = st->ppx_pool + at;
  *p++ = '"';
  for (size_t k = r0; k < r1; k++) {
    const PPTok *t = &v->v[k];
    const char *s = ppx_s(st, t);
    if (k > r0 && t->ws) *p++ = ' ';
    for (uint32_t z = 0; z < t->len; z++) {
      if (t->kind == 's' && t->len >= 2u && (s[z] == '"' || s[z] == '\\')) *p++ = '\\';
      *p++ = s[z];
    }
  }
  *p++ = '"'; *p = 0;
  out->src = PPX_POOL; out->off = (uint32_t)at; out->len = (uint32_t)(p - (st->ppx_pool + at)); out->kind = 's';
  st->ppx_pool_n = at + out->len + 1u;
  return 1;
}

/* Parameter q's prescanned argument (made once): the argument rescanned on its own one level deeper, into st->ppx_out. */
static int ppx_prescan(CppState *st, PPSubst *sb, int q) {
  PPParam *p = &sb->p[q];
  PPToks *S = &st->ppx_stack;
  size_t base = S->n;
  if (p->prescanned) return 1;
  if (sb->depth >= PPX_MAX_DEPTH) { cpp_limit(st, PP_EXPANSION_TOO_DEEP); return 0; }
  if (!toks_room(st, S, p->raw1 - p->raw0)) return 0;
  for (size_t k = p->raw1; k > p->raw0; k--) S->v[S->n++] = st->ppx_args.v[k - 1u];
  p->pre0 = st->ppx_out.n;
  if (!ppx_rescan(st, base, 0, sb->depth + 1)) return 0;
  p->pre1 = st->ppx_out.n;
  p->prescanned = 1;
  return 1;
}

/* The parameter body token `b` names, or -1. */
static int ppx_param(const CppState *st, const Macro *m, const PPTok *b) {
  if (!m->isfunc || b->kind != 'i') return -1;
  for (int q = 0; q < m->np; q++)
    if (strlen(m->params[q]) == b->len && !memcmp(m->params[q], ppx_s(st, b), b->len)) return q;
  return -1;
}

/* The replacement list's tokens body[b0..b1) (st->ppx_body) substituted onto the end of st->ppx_sub (the oracle's
 * `_subst_tokens`): each parameter replaced, `#` stringized, `##` pasted, `__VA_OPT__` taken or dropped. */
static int ppx_subst(CppState *st, PPSubst *sb, size_t b0, size_t b1, int *carry_out) {
  const Macro *m = &st->macros[sb->mi];
  PPToks *O = &st->ppx_sub;
  size_t region = O->n;                     /* this call's own output: what a `##` here can see */
  int carry = 0;                            /* white space an empty run handed on: past the end too (*carry_out) */
  size_t i = b0;
  while (i < b1 && !ppx_stopped(st)) {
    PPTok t = st->ppx_body.v[i];
    const PPTok *nx = i + 1u < b1 ? &st->ppx_body.v[i + 1u] : NULL;
    int q;
    if (t.kind == 'i' && ppx_is(st, &t, "__VA_OPT__") && nx && ppx_is(st, nx, "(")) {
      size_t j = i + 2u, c0 = j;
      int depth = 1;
      while (j < b1) {
        const PPTok *u = &st->ppx_body.v[j];
        if (ppx_is(st, u, "(")) depth++;
        else if (ppx_is(st, u, ")") && !--depth) break;
        j++;
      }
      {
        int vq = m->variadic ? m->np - 1 : -1;
        int taken = vq >= 0 && sb->p[vq].raw1 > sb->p[vq].raw0;
        size_t r0 = O->n;
        int inner = 0;
        if (taken && !ppx_subst(st, sb, c0, j, &inner)) return 0;
        if (!ppx_put(st, NULL, r0, taken ? O->n : r0, t.ws, &carry)) return 0;
        carry = carry || inner;
      }
      i = j < b1 ? j + 1u : j;
      continue;
    }
    if (t.kind == 'p' && t.len == 1u && ppx_s(st, &t)[0] == '#' && nx && (q = ppx_param(st, m, nx)) >= 0) {
      PPTok sp;
      sp.hs = 0u; sp.ws = 0u;
      if (!ppx_stringize(st, &st->ppx_args, sb->p[q].raw0, sb->p[q].raw1, &sp)) return 0;
      if (!ppx_charge(st, sp.len) || !ppx_put1(st, sp, t.ws, &carry)) return 0;
      i += 2u;
      continue;
    }
    if (t.kind == 'p' && ppx_is(st, &t, "##") && nx) {
      int rq = ppx_param(st, m, nx);
      PPToks one; PPTok lit = *nx;
      const PPToks *rv; size_t r0, r1;
      lit.hs = 0u;
      if (rq >= 0) { rv = &st->ppx_args; r0 = sb->p[rq].raw0; r1 = sb->p[rq].raw1; }
      else { one.v = &lit; one.n = one.cap = 1u; rv = &one; r0 = 0u; r1 = 1u; }
      if (O->n > region && O->v[O->n - 1u].len) {             /* a token on the left */
        if (r1 > r0) {
          PPTok left = O->v[O->n - 1u], right = rv->v[r0], glued;
          uint32_t h;
          if (!ppx_spell(st, &left, &right, &glued) || !ppx_charge(st, glued.len)) return 0;
          if (!hs_and(st, left.hs, right.hs, &h)) return 0;
          glued.hs = h; glued.ws = left.ws;
          O->v[O->n - 1u] = glued;
          if (!ppx_put(st, rv, r0 + 1u, r1, r1 > r0 + 1u ? rv->v[r0 + 1u].ws : 0, &carry)) return 0;
        }                                                     /* x ## placemarker: x */
      } else {                                                /* a placemarker on the left takes the right's place */
        int pws = 0;
        if (O->n > region) pws = O->v[--O->n].ws;
        if (r1 > r0) { if (!ppx_put(st, rv, r0, r1, pws, &carry)) return 0; }
        else { PPTok pm; memset(&pm, 0, sizeof pm); if (!ppx_put1(st, pm, pws, &carry)) return 0; }
      }
      i += 2u;
      continue;
    }
    if ((q = ppx_param(st, m, &t)) >= 0) {
      if (nx && ppx_is(st, nx, "##")) {                       /* the left operand of `##`: the argument as written */
        if (sb->p[q].raw1 > sb->p[q].raw0) { if (!ppx_put(st, &st->ppx_args, sb->p[q].raw0, sb->p[q].raw1, t.ws, &carry)) return 0; }
        else { PPTok pm; memset(&pm, 0, sizeof pm); if (!ppx_put1(st, pm, t.ws, &carry)) return 0; }
      } else {
        if (!ppx_prescan(st, sb, q)) return 0;
        if (!ppx_put(st, &st->ppx_out, sb->p[q].pre0, sb->p[q].pre1, t.ws, &carry)) return 0;
      }
      i++;
      continue;
    }
    t.hs = 0u;
    if (!ppx_put1(st, t, t.ws, &carry)) return 0;
    i++;
  }
  *carry_out = carry;
  return !ppx_stopped(st);
}

/* Macro `mi`'s replacement for the invocation `a` (NULL for an object-like macro), each token's hide set joined with
 * `hs`, pushed onto the level's pending stack (the oracle's `_substitute` and `_push`): its first token takes the
 * name's spacing `ws`, and the next pending token above `base` takes the white space an empty run at its end handed
 * on -- or, where it is empty, the name's too. */
static int ppx_replace(CppState *st, int mi, const PPArgs *a, uint32_t hs, int depth, int ws, size_t base) {
  const Macro *m = &st->macros[mi];
  PPSubst sb;
  size_t bmark = st->ppx_body.n, smark = st->ppx_sub.n, omark = st->ppx_out.n, amark = st->ppx_args.n, w;
  int ok = 0, carry = 0;
  memset(&sb, 0, sizeof sb);
  sb.mi = mi; sb.depth = depth;
  for (int q = 0; q < m->np && a; q++) {                      /* each parameter's argument; the variadic one flat */
    if (m->variadic && q == m->np - 1) {
      sb.p[q].raw0 = st->ppx_args.n;
      for (int k = q; k < a->n && k < BCIR_CPP_ARGS; k++) {
        if (k > q) {
          PPTok comma; memset(&comma, 0, sizeof comma);
          if (!ppx_spell_text(st, ",", 1u, &comma)) goto done;
          comma.ws = a->comma_ws[k - 1];
          if (!toks_push(st, &st->ppx_args, comma)) goto done;
        }
        if (!toks_room(st, &st->ppx_args, a->end[k] - a->start[k])) goto done;
        for (size_t z = a->start[k]; z < a->end[k]; z++) st->ppx_args.v[st->ppx_args.n++] = st->ppx_args.v[z];
      }
      sb.p[q].raw1 = st->ppx_args.n;
    } else if (q < a->n) {
      sb.p[q].raw0 = a->start[q]; sb.p[q].raw1 = a->end[q];
    }
  }
  {                                                            /* the replacement list, as tokens */
    const char *body = m->body;
    int i = 0;
    for (;;) {
      PPTok t; int s, k;
      while (pp_space(body[i])) i++;
      s = i;
      k = ntok(st, body, &i, st->ppx_tok, sizeof st->ppx_tok);
      if (!k) break;
      t.src = (uint16_t)(PPX_BODY + (unsigned)mi); t.off = (uint32_t)s; t.len = (uint32_t)(i - s);
      t.kind = (uint8_t)k; t.ws = (uint8_t)(s > 0 && pp_space(body[s - 1])); t.hs = 0u;
      if (!toks_push(st, &st->ppx_body, t)) goto done;
    }
  }
  if (!ppx_subst(st, &sb, bmark, st->ppx_body.n, &carry)) goto done;
  w = smark;                                                   /* the placemarkers go; each token joins `hs` */
  for (size_t k = smark; k < st->ppx_sub.n; k++) {
    PPTok t = st->ppx_sub.v[k];
    uint32_t h;
    if (!t.len) continue;
    if (!hs_or(st, t.hs, hs, &h)) goto done;
    t.hs = h;
    st->ppx_sub.v[w++] = t;
  }
  {
    PPToks *S = &st->ppx_stack;
    size_t n = w - smark;
    if ((carry || (ws && !n)) && S->n > base) S->v[S->n - 1u].ws = 1u;
    if (n) {
      st->ppx_sub.v[smark].ws = (uint8_t)ws;
      if (!toks_room(st, S, n)) goto done;
      for (size_t k = w; k > smark; k--) S->v[S->n++] = st->ppx_sub.v[k - 1u];
    }
  }
  ok = 1;
done:
  st->ppx_body.n = bmark; st->ppx_sub.n = smark; st->ppx_out.n = omark; st->ppx_args.n = amark;
  return ok;
}

/* Rescan the level's pending tokens -- those above `base` on st->ppx_stack -- onto st->ppx_out (the oracle's
 * `_rescan`). `top`: they are a source line's, which the next line may continue: the end of them inside an invocation
 * (st->incomplete = 1), or just after a function-like macro's name (2), stops the expansion for cpp_process to join
 * the next line. */
static int ppx_rescan(CppState *st, size_t base, int top, int depth) {
  PPToks *S = &st->ppx_stack;
  while (S->n > base) {
    PPTok t;
    if (!ppx_charge(st, 1u)) return 0;
    t = S->v[--S->n];
    if (t.kind == 'i') {
      int mi = find_macro(st, ppx_s(st, &t), (int)t.len);
      if (mi >= 0 && !hs_has(st, t.hs, mi)) {
        const Macro *m = &st->macros[mi];
        uint32_t h;
        if (!m->isfunc) {
          if (!hs_add(st, t.hs, mi, &h) || !ppx_replace(st, mi, NULL, h, depth, t.ws, base)) return 0;
          continue;
        }
        if (S->n > base && ppx_is(st, &S->v[S->n - 1u], "(")) {
          PPArgs a;
          size_t amark = st->ppx_args.n;
          uint32_t h0;
          int ok = ppx_collect(st, base, top, &a) && ppx_arity(st, m, &a) && hs_and(st, t.hs, a.close, &h0) &&
                   hs_add(st, h0, mi, &h) && ppx_replace(st, mi, &a, h, depth, t.ws, base);
          st->ppx_args.n = amark;
          if (!ok) return 0;
          continue;
        }
        if (top && S->n == base) { st->incomplete = 2; return 0; }   /* its `(` may open the next line */
      } else if (ppx_is(st, &t, "__LINE__") || ppx_is(st, &t, "__FILE__")) {
        goto dynamic;                       /* the dynamic predefined macros: a #define of either wins unless painted */
      } else if (ppx_is(st, &t, "_Pragma") && S->n > base && ppx_is(st, &S->v[S->n - 1u], "(")) {
        PPArgs a;
        size_t amark = st->ppx_args.n;
        int ok = ppx_collect(st, base, top, &a);            /* _Pragma("..."): a lowering no-op, like #pragma */
        st->ppx_args.n = amark;
        if (!ok) return 0;
        continue;
      }
    }
    goto output;
  dynamic:
    {
      PPTok v = t;
      if (ppx_is(st, &t, "__LINE__")) {
        char num[16]; int n = snprintf(num, sizeof num, "%d", st->current_line);
        if (!ppx_spell_text(st, num, (size_t)n, &v)) return 0;
      } else {
        const char *f = st->current_file ? st->current_file : "";
        size_t n = 2u, at;
        for (const char *q = f; *q; q++) n += (*q == '\\' || *q == '"') ? 2u : 1u;
        at = st->ppx_pool_n;
        if (!ppx_grow(st, (void **)&st->ppx_pool, &st->ppx_pool_cap, at + n + 1u, 1u)) return 0;
        {
          char *p = st->ppx_pool + at;
          *p++ = '"';
          for (const char *q = f; *q; q++) { if (*q == '\\' || *q == '"') *p++ = '\\'; *p++ = *q; }
          *p++ = '"'; *p = 0;
        }
        st->ppx_pool_n = at + n + 1u;
        v.src = PPX_POOL; v.off = (uint32_t)at; v.len = (uint32_t)n; v.kind = 's';
      }
      v.hs = t.hs; v.ws = t.ws;
      t = v;
    }
  output:
    if (!ppx_charge(st, t.len) || !toks_push(st, &st->ppx_out, t)) return 0;
  }
  return 1;
}

/* The expansion of the text `line` -- a source line when `top` -- in st->ppx_text, re-lexable to exactly its tokens:
 * a space only between two that would otherwise lex as another (`pastes`, the oracle's `_join`). NULL where it is
 * refused (st->limit_error) or, for a source line, continues on the next (st->incomplete). */
static const char *expand_text(CppState *st, const char *line, int top) {
  PPToks *S = &st->ppx_stack, *O = &st->ppx_out;
  size_t w = 0u;
  char ptok[4] = "";
  const char *prevp = ptok;
  char prevc = 0;
  int prevnum = 0, i = 0;
  st->ppx_work = 0u; st->ppx_pool_n = 0u; st->incomplete = 0;
  S->n = O->n = st->ppx_args.n = st->ppx_sub.n = st->ppx_body.n = 0u;
  hs_reset(st);
  st->ppx_in = line;
  for (;;) {                                                   /* the line, as tokens, onto ppx_out and then reversed */
    PPTok t; int s, k;
    while (pp_space(line[i])) i++;
    s = i;
    k = ntok(st, line, &i, st->ppx_tok, sizeof st->ppx_tok);
    if (!k) break;
    t.src = PPX_IN; t.off = (uint32_t)s; t.len = (uint32_t)(i - s); t.kind = (uint8_t)k;
    t.ws = (uint8_t)(s > 0 && pp_space(line[s - 1])); t.hs = 0u;
    if (!toks_push(st, O, t)) return NULL;
  }
  if (st->limit_error || !toks_room(st, S, O->n)) return NULL;
  for (size_t k = O->n; k > 0; k--) S->v[S->n++] = O->v[k - 1u];
  O->n = 0u;
  if (!ppx_rescan(st, 0u, top, 0) || ppx_stopped(st)) return NULL;
  if (!ppx_grow(st, (void **)&st->ppx_text, &st->ppx_text_cap, 1u, 1u)) return NULL;
  st->ppx_text[0] = 0;
  for (size_t k = 0; k < O->n; k++) {
    const PPTok *t = &O->v[k];
    const char *s = ppx_s(st, t);
    char head[3];                                              /* `pastes` reads a token's first two bytes */
    int sp;
    head[0] = s[0]; head[1] = t->len > 1u ? s[1] : 0; head[2] = 0;
    sp = w && pastes(prevp, prevc, head, prevnum);
    if (!ppx_grow(st, (void **)&st->ppx_text, &st->ppx_text_cap, w + t->len + 2u, 1u)) return NULL;
    s = ppx_s(st, t);
    if (sp) st->ppx_text[w++] = ' ';
    memcpy(st->ppx_text + w, s, t->len); w += t->len; st->ppx_text[w] = 0;
    prevc = s[t->len - 1u]; prevnum = t->kind == 'n';
    if (t->kind == 'p' && t->len < sizeof ptok) { memcpy(ptok, s, t->len); ptok[t->len] = 0; } else ptok[0] = 0;
    prevp = ptok;
  }
  return st->ppx_text;
}


/* --- the standard headers the preprocessor supplies (CSUB-1) -------------- */
/* The text of `<stdint.h>`, `<stddef.h>`, `<stdbool.h>`, `<limits.h>`, `<float.h>`, `<iso646.h>`, `<stdalign.h>` and
 * `<math.h>`'s constants for the target, read where a `#include <...>` names one no search path holds: the oracle's
 * `bcir/frontends/cfront/stdheaders.py` line for line, whose docstring states what each holds, what it leaves out and
 * why. Each limit has the type C gives it (an object of its type after the integer promotions), and a typedef is
 * marked `__bcir_builtin`, which the linkable emit does not write out again. */
typedef struct { char *o; size_t cap, w; int full; } hbuf;
static void hput(hbuf *b, const char *fmt, ...) {
  if (b->full) return;
  va_list ap; va_start(ap, fmt);
  int n = vsnprintf(b->o + b->w, b->cap - b->w, fmt, ap);
  va_end(ap);
  if (n < 0 || (size_t)n + 2 > b->cap - b->w) { b->full = 1; return; }
  b->w += (size_t)n; b->o[b->w++] = '\n'; b->o[b->w] = 0;
}
static const char *h_top(int bits){ return bits==8?"127":bits==16?"32767":bits==32?"2147483647":"9223372036854775807"; }
static const char *h_utop(int bits){ return bits==8?"255":bits==16?"65535":bits==32?"4294967295":"18446744073709551615"; }
static void h_signed(hbuf *b, const char *name, int bits, const char *sfx) {
  hput(b, "#define %s_MIN (-%s%s-1)", name, h_top(bits), sfx);
  hput(b, "#define %s_MAX %s%s", name, h_top(bits), sfx);
}
static void h_umax(hbuf *b, const char *name, int bits, const char *sfx) {
  hput(b, "#define %s_MAX %s%s", name, h_utop(bits), sfx);
}
static void h_stdint(const CppState *st, hbuf *b) {
  const char *i64 = st->long_size == 8 ? "long" : "long long", *s64 = st->long_size == 8 ? "L" : "LL";
  const char *us64 = st->long_size == 8 ? "UL" : "ULL";
  char n[32];
  for (int bits = 8; bits <= 64; bits *= 2) {
    hput(b, "__bcir_builtin typedef int%d_t int_least%d_t;", bits, bits);
    hput(b, "__bcir_builtin typedef uint%d_t uint_least%d_t;", bits, bits);
  }
  hput(b, "__bcir_builtin typedef %s intmax_t;", i64);
  hput(b, "__bcir_builtin typedef unsigned %s uintmax_t;", i64);
  for (int bits = 8; bits <= 64; bits *= 2) {
    const char *sfx = bits == 64 ? s64 : "", *usfx = bits == 64 ? us64 : bits == 32 ? "U" : "";
    snprintf(n, sizeof n, "INT%d", bits); h_signed(b, n, bits, sfx);
    snprintf(n, sizeof n, "INT_LEAST%d", bits); h_signed(b, n, bits, sfx);
    snprintf(n, sizeof n, "UINT%d", bits); h_umax(b, n, bits, usfx);
    snprintf(n, sizeof n, "UINT_LEAST%d", bits); h_umax(b, n, bits, usfx);
  }
  if (st->pointer_size == 8) {
    h_signed(b, "INTPTR", 64, s64); h_umax(b, "UINTPTR", 64, us64);
    h_signed(b, "PTRDIFF", 64, s64); h_umax(b, "SIZE", 64, us64);
  } else {
    h_signed(b, "INTPTR", 32, ""); h_umax(b, "UINTPTR", 32, "U");
    h_signed(b, "PTRDIFF", 32, ""); h_umax(b, "SIZE", 32, "U");
  }
  h_signed(b, "INTMAX", 64, s64); h_umax(b, "UINTMAX", 64, us64);
  h_signed(b, "SIG_ATOMIC", 32, "");
  if (st->wchar_size == 2) {                     /* an `unsigned short`: promoted to `int` */
    hput(b, "#define WCHAR_MIN 0"); hput(b, "#define WCHAR_MAX 65535");
    hput(b, "#define WINT_MIN 0"); hput(b, "#define WINT_MAX 65535");
  } else {
    if (st->wchar_signed) h_signed(b, "WCHAR", 32, "");
    else { hput(b, "#define WCHAR_MIN 0U"); hput(b, "#define WCHAR_MAX 4294967295U"); }
    hput(b, "#define WINT_MIN 0U"); hput(b, "#define WINT_MAX 4294967295U");
  }
  for (int bits = 8; bits <= 16; bits *= 2) {
    hput(b, "#define INT%d_C(c) c", bits); hput(b, "#define UINT%d_C(c) c", bits);
  }
  hput(b, "#define INT32_C(c) c"); hput(b, "#define UINT32_C(c) c ## U");
  hput(b, "#define INT64_C(c) c ## %s", s64); hput(b, "#define UINT64_C(c) c ## U%s", s64);
  hput(b, "#define INTMAX_C(c) c ## %s", s64); hput(b, "#define UINTMAX_C(c) c ## U%s", s64);
  hput(b, "#if __STDC_VERSION__ >= 202311L");
  for (int bits = 8; bits <= 64; bits *= 2) {
    hput(b, "#define INT%d_WIDTH %d", bits, bits); hput(b, "#define UINT%d_WIDTH %d", bits, bits);
    hput(b, "#define INT_LEAST%d_WIDTH %d", bits, bits); hput(b, "#define UINT_LEAST%d_WIDTH %d", bits, bits);
  }
  { static const char *const w[] = {"INTPTR", "UINTPTR", "PTRDIFF", "SIZE", 0};
    for (int k = 0; w[k]; k++) hput(b, "#define %s_WIDTH %d", w[k], 8 * st->pointer_size); }
  hput(b, "#define INTMAX_WIDTH 64"); hput(b, "#define UINTMAX_WIDTH 64"); hput(b, "#define SIG_ATOMIC_WIDTH 32");
  hput(b, "#define WCHAR_WIDTH %d", 8 * st->wchar_size); hput(b, "#define WINT_WIDTH %d", 8 * st->wchar_size);
  hput(b, "#endif");
}
static void h_stddef(const CppState *st, hbuf *b) {
  hput(b, "__bcir_builtin typedef %s ptrdiff_t;",
       st->pointer_size == 8 ? (st->long_size == 8 ? "long" : "long long") : "int");
  if (st->msvc) hput(b, "__bcir_builtin typedef double max_align_t;");
  else hput(b, "__bcir_builtin typedef struct { _Alignas(8) long long __bcir_max_align_ll; "
               "long double __bcir_max_align_ld; } max_align_t;");
  hput(b, "#define NULL ((void*)0)"); hput(b, "#define offsetof(t, d) __builtin_offsetof(t, d)");
}
static void h_stdbool(hbuf *b) {
  hput(b, "#if __STDC_VERSION__ < 202311L");
  hput(b, "#define bool _Bool"); hput(b, "#define true 1"); hput(b, "#define false 0");
  hput(b, "#endif");
  hput(b, "#define __bool_true_false_are_defined 1");
}
static void h_limits(const CppState *st, hbuf *b) {
  int long_bits = 8 * st->long_size;
  hput(b, "#define CHAR_BIT 8"); hput(b, "#define MB_LEN_MAX %d", st->msvc ? 5 : 16);
  hput(b, "#define SCHAR_MIN (-127-1)"); hput(b, "#define SCHAR_MAX 127"); hput(b, "#define UCHAR_MAX 255");
  if (st->char_signed) { hput(b, "#define CHAR_MIN (-127-1)"); hput(b, "#define CHAR_MAX 127"); }
  else { hput(b, "#define CHAR_MIN 0"); hput(b, "#define CHAR_MAX 255"); }
  hput(b, "#define SHRT_MIN (-32767-1)"); hput(b, "#define SHRT_MAX 32767"); hput(b, "#define USHRT_MAX 65535");
  hput(b, "#define INT_MIN (-2147483647-1)"); hput(b, "#define INT_MAX 2147483647");
  hput(b, "#define UINT_MAX 4294967295U");
  if (long_bits == 64) { h_signed(b, "LONG", 64, "L"); hput(b, "#define ULONG_MAX 18446744073709551615UL"); }
  else { h_signed(b, "LONG", 32, "L"); hput(b, "#define ULONG_MAX 4294967295UL"); }
  h_signed(b, "LLONG", 64, "LL"); hput(b, "#define ULLONG_MAX 18446744073709551615ULL");
  hput(b, "#if __STDC_VERSION__ >= 202311L");
  hput(b, "#define BOOL_WIDTH 1");
  { static const char *const k[] = {"CHAR", "SCHAR", "UCHAR", "SHRT", "USHRT", "INT", "UINT", "LONG", "ULONG",
                                     "LLONG", "ULLONG", 0};
    static const int w[] = {8, 8, 8, 16, 16, 32, 32, 0, 0, 64, 64};
    for (int i = 0; k[i]; i++) hput(b, "#define %s_WIDTH %d", k[i], w[i] ? w[i] : long_bits); }
  hput(b, "#endif");
}
/* <float.h>'s limits, as Clang predefines them (the oracle's `_FLT`, `_DBL` and `_LDBL`): one row a name and a value
 * per format -- `float`, `double`, and `long double`'s x87 80-bit one (x86-64 and i386 Linux) and IEEE binary128
 * (AArch64 and RISC-V Linux); Windows' `long double` is `double`, its floating values spelled with an `L`. */
static const char *const h_fl_keys[12] = {"MANT_DIG", "DIG", "MIN_EXP", "MIN_10_EXP", "MAX_EXP", "MAX_10_EXP", "MAX",
                                          "EPSILON", "MIN", "TRUE_MIN", "DECIMAL_DIG", "HAS_SUBNORM"};
static const char *const h_flt[12] = {"24", "6", "(-125)", "(-37)", "128", "38", "3.40282347e+38F",
                                      "1.19209290e-7F", "1.17549435e-38F", "1.40129846e-45F", "9", "1"};
static const char *const h_dbl[12] = {"53", "15", "(-1021)", "(-307)", "1024", "308", "1.7976931348623157e+308",
                                      "2.2204460492503131e-16", "2.2250738585072014e-308",
                                      "4.9406564584124654e-324", "17", "1"};
static const char *const h_x87[12] = {"64", "18", "(-16381)", "(-4931)", "16384", "4932",
                                      "1.18973149535723176502e+4932L", "1.08420217248550443401e-19L",
                                      "3.36210314311209350626e-4932L", "3.64519953188247460253e-4951L", "21", "1"};
static const char *const h_quad[12] = {"113", "33", "(-16381)", "(-4931)", "16384", "4932",
                                       "1.18973149535723176508575932662800702e+4932L",
                                       "1.92592994438723585305597794258492732e-34L",
                                       "3.36210314311209350626267781732175260e-4932L",
                                       "6.47517511943802511092443895822764655e-4966L", "36", "1"};
static void h_float(const CppState *st, hbuf *b) {
  const char *const *ld = st->long_double == 1 ? h_x87 : st->long_double == 2 ? h_quad : h_dbl;
  int as_dbl = st->long_double == 0;
  hput(b, "#define FLT_EVAL_METHOD %d", st->flt_eval); hput(b, "#define FLT_RADIX 2");
  hput(b, "#define DECIMAL_DIG %s", ld[10]);
  for (int k = 0; k < 12; k++) hput(b, "#define FLT_%s %s", h_fl_keys[k], h_flt[k]);
  for (int k = 0; k < 12; k++) hput(b, "#define DBL_%s %s", h_fl_keys[k], h_dbl[k]);
  for (int k = 0; k < 12; k++)
    hput(b, "#define LDBL_%s %s%s", h_fl_keys[k], ld[k], as_dbl && strchr(ld[k], 'e') ? "L" : "");
}
static void h_iso646(hbuf *b) {
  static const char *const p[][2] = {{"and", "&&"}, {"and_eq", "&="}, {"bitand", "&"}, {"bitor", "|"},
                                      {"compl", "~"}, {"not", "!"}, {"not_eq", "!="}, {"or", "||"},
                                      {"or_eq", "|="}, {"xor", "^"}, {"xor_eq", "^="}};
  for (size_t k = 0; k < sizeof p / sizeof p[0]; k++) hput(b, "#define %s %s", p[k][0], p[k][1]);
}
static void h_stdalign(hbuf *b) {
  hput(b, "#if __STDC_VERSION__ < 202311L");
  hput(b, "#define alignas _Alignas"); hput(b, "#define alignof _Alignof");
  hput(b, "#define __alignas_is_defined 1"); hput(b, "#define __alignof_is_defined 1");
  hput(b, "#endif");
}
static void h_math(hbuf *b) {
  hput(b, "#define NAN (__builtin_nanf(\"\"))"); hput(b, "#define INFINITY (__builtin_inff())");
  hput(b, "#define HUGE_VAL (__builtin_huge_val())"); hput(b, "#define HUGE_VALF (__builtin_huge_valf())");
  hput(b, "#define HUGE_VALL (__builtin_huge_vall())");
  hput(b, "#define isfinite(x) __builtin_isfinite(x)"); hput(b, "#define isnan(x) __builtin_isnan(x)");
  hput(b, "#define isinf(x) __builtin_isinf_sign(x)"); hput(b, "#define signbit(x) __builtin_signbit(x)");
  hput(b, "#define isnormal(x) __builtin_isnormal(x)");
}
static const char *const builtin_names[] = {"stdint.h", "stddef.h", "stdbool.h", "limits.h", "float.h", "iso646.h",
                                            "stdalign.h", "math.h", 0};
static int builtin_known(const char *name) {
  for (int k = 0; builtin_names[k]; k++) if (!strcmp(builtin_names[k], name)) return 1;
  return 0;
}
/* The text of the standard header `name` for the state's target, in state->builtin_text, guarded so a second
 * include reads nothing: 1, or 0 for a header the preprocessor does not supply (or, never, one past the buffer). */
static int builtin_header(CppState *st, const char *name) {
  if (!builtin_known(name)) return 0;
  hbuf b = {st->builtin_text, sizeof st->builtin_text, 0, 0};
  char guard[32]; size_t g = 0;
  g += (size_t)snprintf(guard, sizeof guard, "__BCIR_");
  for (const char *q = name; *q && g + 1 < sizeof guard; q++) guard[g++] = *q == '.' ? '_' : (char)(*q >= 'a' && *q <= 'z' ? *q - 32 : *q);
  guard[g] = 0;
  st->builtin_text[0] = 0;
  hput(&b, "#ifndef %s", guard); hput(&b, "#define %s 1", guard);
  if (!strcmp(name, "stdint.h")) h_stdint(st, &b);
  else if (!strcmp(name, "stddef.h")) h_stddef(st, &b);
  else if (!strcmp(name, "stdbool.h")) h_stdbool(&b);
  else if (!strcmp(name, "limits.h")) h_limits(st, &b);
  else if (!strcmp(name, "float.h")) h_float(st, &b);
  else if (!strcmp(name, "iso646.h")) h_iso646(&b);
  else if (!strcmp(name, "stdalign.h")) h_stdalign(&b);
  else h_math(&b);
  hput(&b, "#endif");
  if (b.full) { cpp_limit(st, "standard header text too large"); return 0; }
  return 1;
}

/* --- #if constant-expression evaluation ---------------------------------- */
/* The reasons a `#if`/`#elif` expression is refused for, the oracle's `cpp.PP_*` word for word (CF-PPARITH). One the
 * grammar does not admit -- nothing, a token no operand or operator is, two operands with no operator between them,
 * an unclosed `(`, a `?` with no `:` -- is malformed wherever it stands; the rest are C's undefined behaviour or
 * constraint violations in an evaluated operand (6.6p3-4, 6.5.5p5-6, 6.5.7p3-4), and an operand C leaves unevaluated
 * -- the right of `&&` after 0 and of `||` after a nonzero, the arm of `?:` not taken -- refuses for none of them. */
#define PP_MALFORMED "malformed #if expression"
#define PP_OVERFLOW "integer overflow in #if expression"
#define PP_SHIFT "invalid shift in #if expression"
#define PP_DIVZERO "division by zero in #if expression"
#define PP_COMMA "comma operator in #if expression"
#define PP_TOO_MANY "too many tokens in #if expression"
#define PP_DEEP "#if expression nested too deeply"
#define PP_MAX_DEPTH 63                     /* nested `(` and `?:`: C11 5.2.4.1's minimum */
typedef struct { char t[512][64]; int n, i; CppState *state; } CE;
/* A `#if` operand (6.10.1p4): its 64 bits, and whether it is a `uintmax_t` (else an `intmax_t`). */
typedef struct { unsigned long long v; int u; } ppv;
static void ce_fail(CE *c, const char *message){ cpp_limit(c->state, message); }
static long long ppv_signed(ppv a){ return a.v>>63 ? -(long long)(~a.v)-1 : (long long)a.v; }
static ppv ce_comma(CE *c, int live, int depth);
/* An integer constant of a `#if`, read as the lexer reads one (`int_literal`, bcir_intlit.h; the oracle's `_int_lit`
 * reads `clex.parse_int_literal`): its digits checked against its base, an octal one read as octal -- `strtol` had read
 * `0777` as 777 where the oracle's `#if` and both lexers read 511, and a malformed digit or suffix (`08`, `1lL`)
 * stopped it short -- refused as the lexer refuses it, in an operand C leaves unevaluated too, for it is no constant
 * at all (CF-SUFFIX), as is one no type holds. It is a `uintmax_t` where its suffix has `u` or its value is past
 * INTMAX_MAX, else an `intmax_t`: it had been a `long`, refused past LONG_MAX (CF-PPARITH). */
static ppv lit(CE *expression, const char *s) {
  intlit L; const char *why=int_literal(s,(int)strlen(s),&L); ppv r={0,0};
  CppState *state=expression->state;
  if(why){
    if(!state->limit_error){
      if(why==int_bad) snprintf(state->constant_error,sizeof state->constant_error,"%s '%.48s'",why,s);
      else if(why==int_bad_suffix) snprintf(state->constant_error,sizeof state->constant_error,"%s '%.*s' on integer constant",why,L.nsuf,L.suf);
      else snprintf(state->constant_error,sizeof state->constant_error,"%s",why);
      cpp_limit(state,state->constant_error); }
    return r; }
  r.v=L.v; r.u=L.u || L.v>(unsigned long long)LLONG_MAX;
  return r;
}
/* A character constant of a `#if` (6.10.1p4), read as Clang and GCC read one: a plain one by the target's plain `char`
 * -- its byte sign-extended, or zero-extended and the operand a `uintmax_t` where `char` is unsigned (so `'a' - 98 < 0`
 * is false on AArch64); a multi-character one packed big-endian into an `int`'s 32 bits and extended likewise -- a `u8`
 * one an `unsigned char`, a `u` and a `U` one `char16_t` and `char32_t`, unsigned, an `L` one the target's `wchar_t`.
 * Both rails had read every one as 0; refused for `char_bad` in any operand (CF-PPARITH; the oracle's `_char_lit`). */
static ppv chr(CE *expression, const char *s) {
  CppState *state=expression->state; charlit L; ppv r={0,0};
  int bits=8*state->wchar_size, sg;
  unsigned long long v;
  if(char_literal(s,(int)strlen(s),bits,&L)){ cpp_limit(state,char_bad); return r; }
  v=L.units[0];
  if(!L.prefix){ bits=8; sg=state->char_signed;
    if(L.n>1){ bits=32; v=0; for(int k=0;k<L.n;k++) v=((v<<8)|L.units[k])&0xFFFFFFFFull; } }
  else if(L.prefix=='L') sg=state->wchar_signed;
  else { bits=L.prefix=='8' ? 8 : L.prefix=='u' ? 16 : 32; sg=0; }
  if(sg && bits<64 && (v>>(bits-1))) v|=~0ull<<bits;    /* sign-extended */
  r.v=v; r.u=!sg;
  return r;
}
static int is_name(const char *t){
  if(!id0((unsigned char)t[0])) return 0;
  for(int k=1;t[k];k++) if(!idc((unsigned char)t[k])) return 0;
  return 1;
}
static int is_char_token(const char *t){
  if(t[0]=='u' && t[1]=='8') t+=2; else if(t[0]=='L'||t[0]=='u'||t[0]=='U') t++;
  return t[0]=='\'';
}
static int ce_is(CE *c, const char *op){ return c->i<c->n && !strcmp(c->t[c->i],op); }
static ppv ce_prim(CE *c, int live, int depth){
  ppv z={0,0}; const char *t;
  if(c->i>=c->n){ ce_fail(c,PP_MALFORMED); return z; }
  t=c->t[c->i++];
  if(!strcmp(t,"(")){
    if(depth>=PP_MAX_DEPTH){ ce_fail(c,PP_DEEP); return z; }
    ppv v=ce_comma(c,live,depth+1);
    if(!ce_is(c,")")){ ce_fail(c,PP_MALFORMED); return z; }
    c->i++; return v; }
  if(t[0]>='0'&&t[0]<='9') return lit(c, t);
  if(is_char_token(t)) return chr(c, t);
  if(is_name(t)){ z.v=!strcmp(t,"true"); return z; }   /* C23 6.10.1p11: `true` is 1, any other identifier 0 */
  ce_fail(c,PP_MALFORMED); return z;
}
/* Prefix operators, read as a run and applied innermost first, so a long run takes no stack. */
static ppv ce_unary(CE *c, int live, int depth){
  int start=c->i, end;
  while(ce_is(c,"+")||ce_is(c,"-")||ce_is(c,"~")||ce_is(c,"!")) c->i++;
  end=c->i;
  ppv v=ce_prim(c,live,depth);
  for(int k=end-1;k>=start;k--){ char op=c->t[k][0];
    if(op=='!'){ v.v=!v.v; v.u=0; }
    else if(op=='~') v.v=~v.v;
    else if(op=='-'){ if(!v.u && v.v==1ull<<63 && live) ce_fail(c,PP_OVERFLOW); v.v=0-v.v; } }
  return v;
}
static int prec(const char *o){
  if(!strcmp(o,"||"))return 1;
  if(!strcmp(o,"&&"))return 2;
  if(!strcmp(o,"|"))return 3;
  if(!strcmp(o,"^"))return 4;
  if(!strcmp(o,"&"))return 5;
  if(!strcmp(o,"==")||!strcmp(o,"!="))return 6;
  if(!strcmp(o,"<")||!strcmp(o,">")||!strcmp(o,"<=")||!strcmp(o,">="))return 7;
  if(!strcmp(o,"<<")||!strcmp(o,">>"))return 8;
  if(!strcmp(o,"+")||!strcmp(o,"-"))return 9;
  if(!strcmp(o,"*")||!strcmp(o,"/")||!strcmp(o,"%"))return 10;
  return 0;
}
/* `a op b` in C's `#if` arithmetic (the oracle's `_apply`); `live` where C evaluates it, so its undefined behaviour
 * refuses the unit -- an unevaluated one's value is never read, but its type still decides a `?:`'s. */
static ppv apply(CE *c, const char *o, ppv a, ppv b, int live){
  ppv r={0,0};
  if(!strcmp(o,"||")){ r.v=a.v||b.v; return r; }
  if(!strcmp(o,"&&")){ r.v=a.v&&b.v; return r; }
  if(!strcmp(o,"<<")||!strcmp(o,">>")){
    r.u=a.u;
    if(b.u ? b.v>=64 : (ppv_signed(b)<0 || ppv_signed(b)>=64)){ if(live) ce_fail(c,PP_SHIFT); return r; }
    unsigned n=(unsigned)b.v;
    if(o[0]=='>'){ r.v = a.u || !(a.v>>63) ? a.v>>n : ~(~a.v>>n); return r; }   /* a negative one: arithmetic */
    if(!a.u && live && (a.v>>63 || (n && a.v>((unsigned long long)LLONG_MAX>>n)))) ce_fail(c,PP_OVERFLOW);
    r.v=a.v<<n; return r;
  }
  int u=a.u||b.u; long long x=ppv_signed(a), y=ppv_signed(b);
  if(!strcmp(o,"==")){ r.v=a.v==b.v; return r; }
  if(!strcmp(o,"!=")){ r.v=a.v!=b.v; return r; }
  if(!strcmp(o,"<")){ r.v=u ? a.v<b.v : x<y; return r; }
  if(!strcmp(o,">")){ r.v=u ? a.v>b.v : x>y; return r; }
  if(!strcmp(o,"<=")){ r.v=u ? a.v<=b.v : x<=y; return r; }
  if(!strcmp(o,">=")){ r.v=u ? a.v>=b.v : x>=y; return r; }
  r.u=u;
  if(!strcmp(o,"&")){ r.v=a.v&b.v; return r; }
  if(!strcmp(o,"|")){ r.v=a.v|b.v; return r; }
  if(!strcmp(o,"^")){ r.v=a.v^b.v; return r; }
  if(!strcmp(o,"/")||!strcmp(o,"%")){
    if(!b.v){ if(live) ce_fail(c,PP_DIVZERO); return r; }
    if(u){ r.v = o[0]=='/' ? a.v/b.v : a.v%b.v; return r; }
    if(x==LLONG_MIN && y==-1){ if(live) ce_fail(c,PP_OVERFLOW); r.v=o[0]=='/' ? a.v : 0; return r; }   /* 6.5.5p6 */
    r.v=(unsigned long long)(o[0]=='/' ? x/y : x%y); return r;
  }
  if(u){ r.v = o[0]=='+' ? a.v+b.v : o[0]=='-' ? a.v-b.v : a.v*b.v; return r; }   /* modulo 2^64 */
  int over = o[0]=='+' ? (y>0 && x>LLONG_MAX-y) || (y<0 && x<LLONG_MIN-y)
           : o[0]=='-' ? (y<0 && x>LLONG_MAX+y) || (y>0 && x<LLONG_MIN+y)
           : (x>0 && ((y>0 && x>LLONG_MAX/y) || (y<0 && y<LLONG_MIN/x)))
             || (x<0 && ((y>0 && x<LLONG_MIN/y) || (y<0 && x<LLONG_MAX/y)));
  if(over && live) ce_fail(c,PP_OVERFLOW);
  r.v = o[0]=='+' ? a.v+b.v : o[0]=='-' ? a.v-b.v : a.v*b.v;   /* the wrapped value, read by nothing live */
  return r;
}
/* Precedence climbing: one frame per operator level, so a long chain takes no stack. */
static ppv ce_bin(CE *c, int minp, int live, int depth){
  ppv lhs=ce_unary(c,live,depth);
  while(c->i<c->n && !c->state->limit_error){ const char *op=c->t[c->i]; int p=prec(op);
    if(!p || p<minp) break;
    c->i++;
    int rlive=live && !(!strcmp(op,"&&") && !lhs.v) && !(!strcmp(op,"||") && lhs.v);
    ppv rhs=ce_bin(c,p+1,rlive,depth);
    lhs=apply(c,op,lhs,rhs,live); }
  return lhs;
}
static ppv ce_cond(CE *c, int live, int depth){
  ppv k=ce_bin(c,1,live,depth), z={0,0};
  if(!ce_is(c,"?")) return k;
  if(depth>=PP_MAX_DEPTH){ ce_fail(c,PP_DEEP); return z; }
  c->i++;
  ppv a=ce_comma(c,live && k.v,depth+1);
  if(!ce_is(c,":")){ ce_fail(c,PP_MALFORMED); return z; }
  c->i++;
  ppv b=ce_cond(c,live && !k.v,depth+1);
  ppv r = k.v ? a : b; r.u=a.u||b.u;
  return r;
}
static ppv ce_comma(CE *c, int live, int depth){
  ppv v=ce_cond(c,live,depth);
  while(ce_is(c,",") && !c->state->limit_error){
    if(live) ce_fail(c,PP_COMMA);
    c->i++; v=ce_cond(c,live,depth); }
  return v;
}

/* Does header `name` resolve against the search dirs (existence only, no read)? Mirrors
 * read_file_dirs' search so __has_include in #if agrees with #include. */
static char *joined_path(CppState *state, const char *base, const char *name);
static int header_exists(CppState *state, const char *const *dirs, int ndirs, const char *name) {
  int tries=ndirs?ndirs:1;
  for(int d=0;d<tries;d++){
    char *path=joined_path(state, ndirs?dirs[d]:NULL,name);
    if(!path){cpp_limit(state, "include path is too large");return 0;}
    FILE *f=fopen(path,"rb");if(f){fclose(f);return 1;}
  }
  return 0;
}

/* Parse the header name out of a `__has_include`/`#include` argument `s[0..e)`: the text between the
 * first <...> or "..." pair. Writes into `nm` (NUL-terminated). */
static void header_name_of(CppState *state, const char *s, int e, char *nm, size_t cap) {
  int p=0; while(p<e && s[p]!='<' && s[p]!='"') p++;
  if(p<e && (s[p]=='<' || s[p]=='"')) p++;
  int q=p; while(q<e && s[q]!='>' && s[q]!='"') q++;
  int nl=q-p; if(nl<0) nl=0; if((size_t)nl>cap-1){cpp_limit(state, "header name is too long");nl=(int)cap-1;}
  memcpy(nm, s+p, (size_t)nl); nm[nl]=0;
}

static int eval_if(CppState *state, const char *expr, const char *const *dirs, int ndirs) {
  /* replace `defined X` / `defined(X)` and the __has_* operators first, then expand, then evaluate. The text written
   * is the oracle's: the expression as it stands, each of those replaced in place by 1 or 0, a space wherever white
   * space came before a token (`ws`), which is all `#` keeps of its spacing (CSUB-1). It had spaced tokens only where
   * two would lex as one, so `defined(A) defined(B)` read as `11` here and as two operands there. */
  char *buf=state->conditional_buffer; size_t w=0; buf[0]=0; int i=0; char t[BCIR_CPP_LINE];
  while(1){{ int q=i; if(ident_at(expr,&q)>BCIR_CPP_NAME_MAX)cpp_limit(state, BCIR_CPP_LONG_NAME); }   /* a name
                                                        * it looks up is a macro name: bounded here and after expansion */
    int ws; { int s0=i; while(pp_space(expr[s0]))s0++; ws=s0>0&&pp_space(expr[s0-1]); }
    int k=ntok(state,expr,&i,t,sizeof t); if(!k)break;
    if(w&&ws)app(state,buf,sizeof state->conditional_buffer,&w," ");
    if(k=='p'&&(unsigned char)t[0]>=0x80){cpp_limit(state,nonascii_outside);break;}   /* past every literal: an
                                                        * identifier and a number are ASCII (CF-PPLIMITS) */
    if(k=='i'&&!strcmp(t,"defined")){int has, j=i; char nm[BCIR_CPP_NAME]="";   /* its operand is a
                                                        * macro name, read as #ifdef reads one */
      while(pp_space(expr[j]))j++;
      if(expr[j]=='('){ j++;
        has=macro_name(state,expr,&j,nm,"malformed defined operator");
        if(has&&!next_is(expr,&j,')')){cpp_limit(state, "malformed defined operator");has=0;}
      } else has=macro_name(state,expr,&j,nm,"malformed defined operator");
      i=j; has=has&&is_defined(state, nm);
      { char bit[2]={(char)(has?'1':'0'),0}; app(state,buf,sizeof state->conditional_buffer,&w,bit); } continue;}
    if(k=='i'&&(!strcmp(t,"__has_attribute")||!strcmp(t,"__has_builtin")||
                !strcmp(t,"__has_c_attribute"))){
      /* feature-test: only the L8 ABI attributes are honoured (no builtins, no [[...]] attrs). */
      char nm[256]=""; int j=i, attr=!strcmp(t,"__has_attribute");
      if(next_is(expr,&j,'(')){ int depth=1; int first=1;   /* the operand's tokens through `t`, whose name is read */
        while(depth){int ak=ntok(state,expr,&j,t,sizeof t); if(!ak)break;
          if(!strcmp(t,"("))depth++; else if(!strcmp(t,")")){depth--; if(!depth)break;}
          else if(first&&ak=='i'){strncpy(nm,t,sizeof nm-1);nm[sizeof nm-1]=0;first=0;} }
        i=j; }
      int yes = attr && has_attribute(nm);
      { char bit[2]={(char)(yes?'1':'0'),0}; app(state,buf,sizeof state->conditional_buffer,&w,bit); } continue;}
    if(k=='i'&&!strcmp(t,"__has_include")){
      /* resolve the header against the include search path (the raw <...>/"..." argument). */
      int j=i; while(expr[j]&&expr[j]!='(') j++; int has=0;
      if(expr[j]=='('){ int s=j+1, depth=1, e=s;
        while(expr[e]&&depth){ if(expr[e]=='(')depth++; else if(expr[e]==')'){depth--; if(!depth)break;} e++; }
        char nm[1024]; header_name_of(state, expr+s, e-s, nm, sizeof nm);
        has = nm[0] ? header_exists(state, dirs, ndirs, nm) : 0;
        if (!has && nm[0]) {                     /* ... or a `<...>` one the preprocessor supplies (CSUB-1) */
          int z = s; while (z < e && pp_space(expr[z])) z++;
          has = expr[z] == '<' && builtin_known(nm); }
        i = (expr[e]==')') ? e+1 : e; }
      { char bit[2]={(char)(has?'1':'0'),0}; app(state,buf,sizeof state->conditional_buffer,&w,bit); } continue;}
    app(state,buf,sizeof state->conditional_buffer,&w,t);}
  if(state->limit_error) return 0;
  const char *ex=expand_text(state,buf,0);
  if(!ex) return 0;
  CE c; c.n=0;c.i=0;c.state=state; int j=0; char tk[64];
  while(1){int q=j; if(ident_at(ex,&q)>BCIR_CPP_NAME_MAX)cpp_limit(state, BCIR_CPP_LONG_NAME);
    int k=ntok(state,ex,&j,tk,sizeof tk); if(!k)break;
    if(c.n>=512){cpp_limit(state, PP_TOO_MANY);break;}
    strncpy(c.t[c.n++],tk,63);c.t[c.n-1][63]=0;}
  if(state->limit_error) return 0;
  ppv v=ce_comma(&c,1,0);
  if(c.i!=c.n) cpp_limit(state,PP_MALFORMED);    /* a token past the expression's end */
  return v.v!=0;
}

/* --- directive processing ------------------------------------------------ */
static void define_macro(CppState *state, const char *rest) {
  int i=0; Macro m; memset(&m,0,sizeof m);
  if(!macro_name(state,rest,&i,m.name,"macro name must be an identifier"))return;
  if(rest[i]=='('){m.isfunc=1;i++; for(;;){
      while(pp_space(rest[i]))i++;
      if(rest[i]==')'){i++;break;}
      if(m.np){
        if(rest[i]!=','){cpp_limit(state, "invalid macro parameter list");return;}
        i++;while(pp_space(rest[i]))i++;
        if(!rest[i]||rest[i]==')'){cpp_limit(state, "invalid macro parameter list");return;}
      }
      if(m.np>=16){cpp_limit(state, "too many macro parameters");return;}
      int ps=i;
      if(!strncmp(rest+i,"...",3)){
        m.variadic=1;strcpy(m.params[m.np++],"__VA_ARGS__");i+=3;
        while(pp_space(rest[i]))i++;
        if(rest[i]!=')'){cpp_limit(state, "variadic macro parameter must be last");return;}
        i++;break;
      }
      if((unsigned char)rest[i]>=0x80){cpp_limit(state, nonascii_outside);return;}   /* a parameter is ASCII, as a
                                                        * name is (CF-PPLIMITS; the oracle's `_params`) */
      if(!id0((unsigned char)rest[i])){cpp_limit(state, "invalid macro parameter");return;}
      int pl;
      while(idc((unsigned char)rest[i]))i++;
      if((unsigned char)rest[i]>=0x80){cpp_limit(state, nonascii_outside);return;}
      pl=i-ps;
      if(pl>BCIR_CPP_NAME_MAX){cpp_limit(state, "macro parameter is too long");return;}
      for(int p=0;p<m.np;p++)if((int)strlen(m.params[p])==pl&&!strncmp(m.params[p],rest+ps,(size_t)pl)){
        cpp_limit(state, "duplicate macro parameter");return;}
      memcpy(m.params[m.np],rest+ps,(size_t)pl);m.params[m.np][pl]=0;m.np++;
      while(pp_space(rest[i]))i++;
      if(rest[i]!=','&&rest[i]!=')'){cpp_limit(state, "invalid macro parameter list");return;}
    }
  }
  while(pp_space(rest[i]))i++;
  if(strlen(rest+i)>=sizeof m.body){cpp_limit(state, "macro replacement is too large");return;}
  strcpy(m.body,rest+i);
  undef_macro(state, m.name);
  if(state->macro_count>=1024){cpp_limit(state, "too many macros");return;}
  state->macros[state->macro_count++]=m;
}

/* Join one include-search directory and name without a path-length truncation. */
static char *joined_path(CppState *state, const char *base, const char *name) {
  size_t bn=base&&base[0]?strlen(base):0, nn=strlen(name), need;
  if(!bcir_size_add(bn,nn,&need)||!bcir_size_add(need,bn?2u:1u,&need))return NULL;
  char *p=(char *)bcir_host_arena_allocate(&state->scratch,need,1u); if(!p)return NULL;
  if(bn)snprintf(p,need,"%s/%s",base,name);else memcpy(p,name,nn+1u);
  return p;
}

/* Read a whole include/embed file.  Returns 0, -1 when no search candidate exists, or -2 on
 * allocation/I/O failure.  Dynamic storage removes the old 8 KiB include and 64 KiB embed
 * truncation; the caller still owns the public output-capacity contract. */
static int read_file_dirs(CppState *state, const char *const *dirs, int ndirs, const char *name,
                          unsigned char **out, size_t *out_len) {
  int tries=ndirs?ndirs:1;
  *out=NULL;*out_len=0;
  for(int d=0;d<tries;d++){
    const char *base=ndirs?dirs[d]:NULL; char *path=joined_path(state,base,name);
    if(!path)return -2;
    FILE *f=fopen(path,"rb");if(!f)continue;
    size_t cap=8192,n=0;unsigned char *buf=(unsigned char *)bcir_host_allocate(&state->allocator,cap);
    if(!buf){fclose(f);return -2;}
    for(;;){
      if(n==cap-1u){
        size_t minimum,nc;
        if(cap>=BCIR_CPP_MAX_FILE_BYTES+1u){
          int extra=fgetc(f);
          if(extra==EOF&&!ferror(f))break;
          bcir_host_deallocate(&state->allocator,buf);fclose(f);return -2;
        }
        if(!bcir_size_add(cap,1u,&minimum)||
           !bcir_host_grow_capacity(cap,minimum,1u,&nc)){
          bcir_host_deallocate(&state->allocator,buf);fclose(f);return -2;}
        if(nc>BCIR_CPP_MAX_FILE_BYTES+1u)nc=BCIR_CPP_MAX_FILE_BYTES+1u;
        if(!bcir_host_realloc_array(&state->allocator,(void **)&buf,cap,nc,1u,0)){
          bcir_host_deallocate(&state->allocator,buf);fclose(f);return -2;}
        cap=nc;
      }
      size_t avail=cap-1u-n, got=fread(buf+n,1,avail,f);n+=got;
      if(got<avail){
        if(ferror(f)){bcir_host_deallocate(&state->allocator,buf);fclose(f);return -2;}
        if(feof(f))break;
        if(!got){bcir_host_deallocate(&state->allocator,buf);fclose(f);return -2;}
      }
    }
    if(fclose(f)){bcir_host_deallocate(&state->allocator,buf);return -2;}
    buf[n]=0;*out=buf;*out_len=n;return 0;
  }
  return -1;
}

/* The main directive/text loop. Does NOT reset the macro table -- macros persist across #includes
 * (matching cpp.py, where the same Preprocessor processes nested files); the conditional stack is
 * per-call (a header's #if is self-contained). Recurses into included files, appending to `out` at
 * `*w` and sharing the macro table. */
/* Translation phase 3: replace every // and block comment with a single space, keeping the newlines
 * a block comment spanned (so __LINE__ and the per-line directive scan stay aligned). String/char
 * literals are copied verbatim; an identifier and a preprocessing number (6.4.2, 6.4.8) are copied whole, so a C23
 * digit separator (`1'000`) -- a `'` inside a pp-number -- is no quote, and the `'` after an identifier (`u8'a'`,
 * `L'a'`) opens a character constant: a `'` flanked by hex digits had been taken for a separator in `u8'a'`, whose
 * closing quote then opened a literal that ran past the next comment (CF-CONSTEXPR2; the oracle's `_strip_comments`).
 * In-place safe (the write cursor never overtakes the read cursor). Mirrors cpp.py _strip_comments,
 * and runs *before* directive processing so a directive inside a comment never fires and the
 * comment's punctuation can't be re-tokenized into the output. */
static void strip_comments(const char *s, char *o, size_t ocap) {
  size_t i=0, w=0;
  #define _DG(ch) ((ch)>='0'&&(ch)<='9')
  #define _IDS(ch) ((((ch)|0x20)>='a'&&((ch)|0x20)<='z') || (ch)=='_')
  #define _IDC(ch) (_IDS(ch) || _DG(ch))
  while(s[i] && w+1<ocap){
    char c=s[i];
    if(_IDC(c) || (c=='.' && _DG(s[i+1]))){        /* an identifier, or a pp-number (a digit or `.digit` first) */
      int num=!_IDS(c);
      o[w++]=c; i++;
      while(s[i] && w+2<ocap){ char ch=s[i];
        if(num && (ch=='e'||ch=='E'||ch=='p'||ch=='P') && (s[i+1]=='+'||s[i+1]=='-')){ o[w++]=ch; o[w++]=s[i+1]; i+=2; }
        else if(num && ch=='\'' && _IDC(s[i+1])){ o[w++]=ch; o[w++]=s[i+1]; i+=2; }   /* a digit separator */
        else if(_IDC(ch) || (num && ch=='.')){ o[w++]=ch; i++; }
        else break; }
      continue; }
    if(c=='"' || c=='\''){
      o[w++]=c; i++;                                  /* a string/char literal: copy verbatim */
      while(s[i] && w+2<ocap){ char ch=s[i];
        if(ch=='\\' && s[i+1]){ o[w++]=ch; o[w++]=s[i+1]; i+=2; continue; }
        o[w++]=ch; i++; if(ch==c) break; }
      continue; }
    if(c=='/' && s[i+1]=='/'){ i+=2; while(s[i] && s[i]!='\n') i++; o[w++]=' '; continue; }
    if(c=='/' && s[i+1]=='*'){ i+=2; int nl=0;
      while(s[i] && !(s[i]=='*'&&s[i+1]=='/')){ if(s[i]=='\n')nl++; i++; }
      if(s[i]) i+=2;                                  /* consume the closing star-slash */
      o[w++]=' '; while(nl>0 && w+1<ocap){ o[w++]='\n'; nl--; } continue; }
    o[w++]=c; i++;
  }
  #undef _DG
  #undef _IDS
  #undef _IDC
  o[w]=0;
}

/* The logical line at *pp into `line` (cap bytes): its backslash-newlines spliced, a CRLF end of line read as a
 * new-line (translation phase 1), *pp past its new-line. Returns its length; *too_long when it outgrew `line`. */
static int read_logical(const char **pp, char *line, int cap, int *too_long) {
  const char *p = *pp; int L = 0; *too_long = 0;
  while(*p&&*p!='\n'){
    if(*p=='\\'&&p[1]=='\n'){p+=2;continue;}
    if(*p=='\\'&&p[1]=='\r'&&p[2]=='\n'){p+=3;continue;}   /* a CRLF end of line is a new-line (translation
                                                           * phase 1): spliced as one, ended as one -- `#if K\r`
                                                           * was malformed here and lowered on the oracle, and a
                                                           * `\\\r\n` splice was none (CF-PPSPLITS) */
    if(*p=='\r'&&p[1]=='\n'){p++;continue;}
    if(L<cap-2)line[L++]=*p;else *too_long=1;
    p++;
  }
  if(*p=='\n')p++;
  line[L]=0; *pp=p; return L;
}
/* Whether `s` holds the identifier token `name` (read as tokens: a literal holding it does not). */
static int has_token(CppState *state, const char *s, const char *name) {
  int i = 0, k; char *t = state->line_token;
  while ((k = ntok(state, s, &i, t, sizeof state->line_token)) != 0) if (k == 'i' && !strcmp(t, name)) return 1;
  return 0;
}
/* The expansion of the source line `line` (a BCIR_CPP_LINE buffer) into `ex`, the lines after it at *pp that it took
 * counted in *extra (the oracle's `_source_line`, CSUB-1). A new-line is white space inside a function-like macro's
 * invocation (C11 6.10.3p10): an argument list still open at the end of the line continues on the next, and a
 * function-like macro's name that ends a line is invoked by a `(` that opens the next line that is not blank. Each
 * line taken is joined to `line` with a space and the whole expanded again, held to the line bound; a directive where
 * the arguments continue is refused (6.10.3p11), and so is `__LINE__` on a continued line, whose line neither rail
 * tracks per token. 1 with `err` set on a refusal. */
static int expand_source_line(CppState *state, char *line, const char **pp, const char **ex, int *extra,
                              char *err, size_t errcap) {
  *extra = 0;
  for (;;) {
    *ex = expand_text(state, line, 1);
    if (state->limit_error) { if(err&&errcap)snprintf(err,errcap,"%s",state->limit_error); return 1; }
    if (!state->incomplete) return 0;
    const char *q = *pp; int taken = 0, too_long = 0;
    char *nx = state->join_line;
    if (state->incomplete == 1) {
      if (!*q) { if(err&&errcap)snprintf(err,errcap,"%s",PP_UNTERMINATED_ARGS); return 1; }
    } else {                                        /* the name: past blank lines, a `(` must open the next one */
      const char *k = q; int blank = 1, paren = 0;
      while (*k && blank) {
        read_logical(&k, nx, BCIR_CPP_LINE, &too_long);
        int z = 0; while (pp_space(nx[z])) z++;
        if (nx[z]) { blank = 0; paren = nx[z] == '('; }
      }
      if (!paren) {                                 /* the name invokes nothing: it is itself (the oracle re-expands) */
        *ex = expand_text(state, line, 0);
        if (!*ex) { if(err&&errcap)snprintf(err,errcap,"%s",state->limit_error?state->limit_error:"preprocessor error"); return 1; }
        return 0;
      }
    }
    do {                                            /* the lines taken: one, or the blank ones and the `(` one */
      read_logical(&q, nx, BCIR_CPP_LINE, &too_long);
      taken++;
      if (too_long) { if(err&&errcap)snprintf(err,errcap,"preprocessor line too long"); return 1; }
      int z = 0; while (pp_space(nx[z])) z++;
      if (nx[z] == '#') { if(err&&errcap)snprintf(err,errcap,"%s",PP_DIRECTIVE_IN_ARGS); return 1; }
      if (find_macro(state, "__LINE__", 8) < 0 && has_token(state, nx, "__LINE__")) {
        if(err&&errcap)snprintf(err,errcap,"%s",PP_LINE_IN_CONTINUED); return 1; }
      size_t a = strlen(line), b = strlen(nx);
      if (a + 1 + b > BCIR_CPP_LINE - 2) { if(err&&errcap)snprintf(err,errcap,"preprocessor line too long"); return 1; }
      line[a] = ' '; memcpy(line + a + 1, nx, b + 1);
      if (state->incomplete == 1) break;
      { int z2 = 0; while (pp_space(nx[z2])) z2++; if (nx[z2]) break; }   /* the `(` line ends the run */
    } while (*q);
    *pp = q; *extra += taken;
  }
}
static int cpp_process(CppState *state, const char *src, const char *curfile,
                       const char *const *dirs, int ndirs,
                       char *out, size_t outcap, size_t *w, char *err, size_t errcap) {
  /* a conditional stack: active flag + taken flag (parent == all enclosing frames active) */
  struct { int active, taken, parent, seen_else; } cs[64]; int ncs=0;
  const char *p=src; char line[8192]; long long presumed=1;
  char filebuf[1024]; const char *curf=curfile;    /* #line may repoint the current file at filebuf */
  while(*p){
    if(state->limit_error){if(err&&errcap)snprintf(err,errcap,"%s",state->limit_error);return 1;}
    int line_too_long=0;
    read_logical(&p,line,(int)sizeof line,&line_too_long);
    if(line_too_long){if(err&&errcap)snprintf(err,errcap,"preprocessor line too long");return 1;}
    if(presumed<1||presumed>INT_MAX){if(err&&errcap)snprintf(err,errcap,"presumed line number is out of range");return 1;}
    state->current_file=curf; state->current_line=(int)presumed; presumed++;
    int q=0; while(pp_space(line[q]))q++;
    if(line[q]=='#'){
      q++; while(pp_space(line[q]))q++; int ds=q; while(idc((unsigned char)line[q]))q++;
      char dir[32]; int dl=q-ds;
      if(dl>31){if(err&&errcap)snprintf(err,errcap,"preprocessor directive name too long");return 1;}
      memcpy(dir,line+ds,(size_t)dl);dir[dl]=0;
      const char *rest=line+q; while(pp_space(*rest))rest++;
      int parent=1; for(int k=0;k<ncs;k++) if(!cs[k].active){parent=0;break;}
      /* a name running on past ASCII -- `#caf\xc3\xa9`, `#\xc2\xa0define` -- in a group read: not a directive name
       * cut at its ASCII start, nor a null directive (CF-PPLIMITS; the oracle's `_directive`) */
      if((unsigned char)line[q]>=0x80&&parent){if(err&&errcap)snprintf(err,errcap,"%s",nonascii_outside);return 1;}
      if(!strcmp(dir,"ifdef")||!strcmp(dir,"ifndef")||!strcmp(dir,"if")){
        int tk=0;
        if(!strcmp(dir,"ifdef")||!strcmp(dir,"ifndef")){   /* a skipped group's operand is never read (C11 6.10.1p6) */
          char n[BCIR_CPP_NAME];int j=0;
          if(parent&&macro_name(state,rest,&j,n,"conditional directive requires an identifier"))
            tk=!strcmp(dir,"ifdef")?is_defined(state,n):!is_defined(state,n);
        } else if(parent)tk=eval_if(state,rest,dirs,ndirs)!=0;
        if(ncs>=64){if(err&&errcap)snprintf(err,errcap,"conditional nesting too deep");return 1;}
        cs[ncs].active=parent&&tk;cs[ncs].taken=tk;cs[ncs].parent=parent;cs[ncs].seen_else=0;ncs++;
      } else if(!strcmp(dir,"endif")){
        if(!ncs){if(err&&errcap)snprintf(err,errcap,"unexpected #endif");return 1;}ncs--;
      }
      else if(!strcmp(dir,"else")||!strcmp(dir,"elif")||!strcmp(dir,"elifdef")||!strcmp(dir,"elifndef")){
        if(!ncs){if(err&&errcap)snprintf(err,errcap,"unexpected #%s",dir);return 1;}
        if(cs[ncs-1].seen_else){if(err&&errcap)snprintf(err,errcap,"#%s after #else",dir);return 1;}
        if(!strcmp(dir,"else"))cs[ncs-1].seen_else=1;
        { int par=cs[ncs-1].parent; int take;
        if(!strcmp(dir,"else")) take=!cs[ncs-1].taken;
        else if(!strcmp(dir,"elifdef")||!strcmp(dir,"elifndef")){   /* read only where no group was taken */
          char n[BCIR_CPP_NAME];int j=0;
          take=!cs[ncs-1].taken&&par&&macro_name(state,rest,&j,n,"conditional directive requires an identifier")&&
               (!strcmp(dir,"elifdef")?is_defined(state,n):!is_defined(state,n));
        }
        else take=!cs[ncs-1].taken&&par&&(eval_if(state,rest,dirs,ndirs)!=0);
        cs[ncs-1].active=par&&take; cs[ncs-1].taken=cs[ncs-1].taken||take; }
      }
      else if(parent){                              /* parent == every enclosing frame is active */
        if(!strcmp(dir,"define")) define_macro(state,rest);
        else if(!strcmp(dir,"undef")){char n[BCIR_CPP_NAME];int j=0;
          if(macro_name(state,rest,&j,n,"#undef requires an identifier"))undef_macro(state,n);}
        else if(!strcmp(dir,"include")){
          int sys=(rest[0]=='<');char close=sys?'>':'"';char nm[1024];int j=0;
          if(rest[0]!='<'&&rest[0]!='"'){if(err&&errcap)snprintf(err,errcap,"malformed #include");return 1;}
          j++;int s2=j;while(rest[j]&&rest[j]!=close)j++;
          if(rest[j]!=close){if(err&&errcap)snprintf(err,errcap,"unterminated #include name");return 1;}
          int nl=j-s2;
          if(nl>1023){if(err&&errcap)snprintf(err,errcap,"#include name too long");return 1;}
          memcpy(nm,rest+s2,(size_t)nl);nm[nl]=0;
          unsigned char *hd=NULL;size_t hn=0;int rr=read_file_dirs(state,dirs,ndirs,nm,&hd,&hn);
          if(rr<0){
            if(rr==-2){if(err&&errcap)snprintf(err,errcap,"cannot read #include %s",nm);return 1;}
            if(!sys){if(err&&errcap)snprintf(err,errcap,"#include %s not found",nm);return 1;}
            if(builtin_header(state,nm)){           /* a standard header the preprocessor supplies (CSUB-1) */
              if(state->include_depth>=64){if(err&&errcap)snprintf(err,errcap,"#include nesting too deep");return 1;}
              state->include_depth++;
              int rc=cpp_process(state,state->builtin_text,nm,dirs,ndirs,out,outcap,w,err,errcap);
              state->include_depth--;
              if(rc)return rc;
            }
            else if(state->limit_error){if(err&&errcap)snprintf(err,errcap,"%s",state->limit_error);return 1;}
          } else {
            if(memchr(hd,0,hn)){bcir_host_deallocate(&state->allocator,hd);if(err&&errcap)snprintf(err,errcap,"#include %s contains NUL",nm);return 1;}
            if(state->include_depth>=64){bcir_host_deallocate(&state->allocator,hd);if(err&&errcap)snprintf(err,errcap,"#include nesting too deep");return 1;}
            strip_comments((char *)hd,(char *)hd,hn+1u);    /* phase 3 for the header (in-place safe) */
            state->include_depth++;
            int rc=cpp_process(state,(char *)hd,nm,dirs,ndirs,out,outcap,w,err,errcap);
            state->include_depth--;
            bcir_host_deallocate(&state->allocator,hd);if(rc)return rc;
          }
        }
        else if(!strcmp(dir,"embed")){
          int sys=(rest[0]=='<');char close=sys?'>':'"';char nm[1024];int j=0;
          if(rest[0]!='<'&&rest[0]!='"'){if(err&&errcap)snprintf(err,errcap,"malformed #embed");return 1;}
          j++;int s2=j;while(rest[j]&&rest[j]!=close)j++;
          if(rest[j]!=close){if(err&&errcap)snprintf(err,errcap,"unterminated #embed name");return 1;}
          int nl=j-s2;if(nl>1023){if(err&&errcap)snprintf(err,errcap,"#embed name too long");return 1;}
          memcpy(nm,rest+s2,(size_t)nl);nm[nl]=0;
          unsigned char *blob=NULL;size_t bn=0;int rr=read_file_dirs(state,dirs,ndirs,nm,&blob,&bn);
          if(rr<0){if(err&&errcap)snprintf(err,errcap,rr==-2?"cannot read #embed %s":"#embed %s not found",nm);return 1;}
          char num[16];for(size_t b=0;b<bn&&!state->limit_error;b++){
            snprintf(num,sizeof num,"%s%u",b?", ":"",(unsigned)blob[b]);app(state,out,outcap,w,num);
          }
          bcir_host_deallocate(&state->allocator,blob);
        }
        else if(!strcmp(dir,"line")){          /* #line N ["file"]: presumed line of the NEXT line
                                                  is N (decimal); optional new __FILE__. Operands are
                                                  macro-expanded first. */
          const char *ex=expand_text(state,rest,0); int j=0; char *t=state->line_token;
          if(!ex){if(err&&errcap)snprintf(err,errcap,"%s",state->limit_error?state->limit_error:"preprocessor error");return 1;}
          { int digits=ntok(state,ex,&j,t,sizeof state->line_token)=='n';   /* a decimal digit sequence (C11 6.10.4p3):
                                                                             * `12abc` and `0x10` are none; both rails
                                                                             * refuse where each once read 12 or nothing */
            for(const char *dp=t; digits && *dp; dp++) if(*dp<'0'||*dp>'9') digits=0;
            if(!digits){                                     /* a character past ASCII in the operand is its own
                                                              * refusal (NONASCII), read before the digits are */
              if(err&&errcap)snprintf(err,errcap,"%s",state->limit_error?state->limit_error:"#line number is not a decimal digit sequence");
              return 1; }
            errno=0; long long line_no=strtoll(t,NULL,10);
            if(errno||line_no<1||line_no>INT_MAX){
              if(err&&errcap)snprintf(err,errcap,"#line number is out of range");
              return 1;
            }
            presumed=line_no;
            if(ntok(state,ex,&j,t,sizeof state->line_token)=='s'){  /* strip the quotes + resolve \ escapes into filebuf */
              size_t fw=0; for(int z=1; t[z] && t[z]!='"'; z++){
                if(t[z]=='\\'&&t[z+1])z++;
                if(fw>=sizeof filebuf-1){if(err&&errcap)snprintf(err,errcap,"#line file name too long");return 1;}
                filebuf[fw++]=t[z]; }
              filebuf[fw]=0; curf=filebuf; } } }
        else if(!strcmp(dir,"error")){           /* `#error text`: the compile's failure, the text macro-expanded
                                                  * (C11 6.10.5), as the oracle's; once ignored here (CF-PPSPLITS) */
          const char *ex=expand_text(state,rest,0);
          if(!ex){if(err&&errcap)snprintf(err,errcap,"%s",state->limit_error?state->limit_error:"preprocessor error");return 1;}
          size_t el=strlen(ex);
          while(el && pp_space((unsigned char)ex[el-1])) el--;
          if(err&&errcap)snprintf(err,errcap,"#error %.*s",(int)el,ex); return 1; }
        else if(!strcmp(dir,"warning")||!strcmp(dir,"pragma")||!dir[0]){ /* accepted, no effect on lowering */ }
        else { if(err&&errcap)snprintf(err,errcap,"unknown directive #%s",dir); return 1; }   /* C11 6.10p9: a
                                                  * non-directive's execution is undefined; the oracle refuses it */
      }
      continue;
    }
    int active=1; for(int k=0;k<ncs;k++) if(!cs[k].active){active=0;break;}
    if(active){ const char *ex=NULL; int extra=0;
      if(expand_source_line(state,line,&p,&ex,&extra,err,errcap)) return 1;
      app(state,out,outcap,w,ex); app(state,out,outcap,w,"\n");
      for(int k=0;k<extra;k++){ presumed++; app(state,out,outcap,w,"\n"); }   /* each line it took keeps its place */
    }
  }
  if(state->limit_error){if(err&&errcap)snprintf(err,errcap,"%s",state->limit_error);return 1;}
  if(ncs){if(err&&errcap)snprintf(err,errcap,"unterminated #if");return 1;}
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
  time_t t = (e && all_digits(e)) ? (time_t)strtoll(e, NULL, 10) : time(NULL);
  struct tm *m = gmtime(&t);
  if (!m) { snprintf(datebuf, 16, "Jan  1 1970"); snprintf(timebuf, 16, "00:00:00"); return; }
  snprintf(datebuf, 16, "%s %2d %d", mon[m->tm_mon], m->tm_mday, m->tm_year + 1900);
  snprintf(timebuf, 16, "%02d:%02d:%02d", m->tm_hour, m->tm_min, m->tm_sec);
}

/* The extended entry point: the source name for __FILE__ (NULL/"" -> "<source>"), multiple include
 * dirs (-I) + predefined macros (-D, each "name body"). Seeds the predefined + -D macros ONCE;
 * cpp_process then keeps them across nested includes. */
int bcir_cpp_context_init(bcir_cpp_context *context,
                          const bcir_host_allocator *allocator) {
  CppState *state;
  bcir_host_allocator selected;
  if (!context) return 1;
  memset(context, 0, sizeof *context);
  selected = bcir_host_allocator_or_default(allocator);
  state = (CppState *)bcir_host_allocate_zeroed(&selected, sizeof *state);   /* 2.3 MB: zero pages, not a memset */
  if (!state) return 1;
  state->allocator = selected;
  state->char_signed = 1; state->wchar_size = 4; state->wchar_signed = 1;   /* x86-64 Linux's */
  state->long_size = 8; state->pointer_size = 8; state->long_double = 1; state->msvc = 0; state->flt_eval = 0;
  (void)bcir_host_arena_init(&state->scratch, &selected, 16384u);
  context->allocator = selected;
  context->state = state;
  return 0;
}

void bcir_cpp_context_reset(bcir_cpp_context *context) {
  CppState *state;
  if (!context || !context->state) return;
  state = (CppState *)context->state;
  bcir_host_arena_reset(&state->scratch);
  state->macro_count = 0;
  state->limit_error = NULL;
  state->current_file = "<source>";
  state->current_line = 0;
  state->include_depth = 0;
}

void bcir_cpp_context_destroy(bcir_cpp_context *context) {
  CppState *state;
  bcir_host_allocator allocator;
  if (!context || !context->state) {
    if (context) memset(context, 0, sizeof *context);
    return;
  }
  state = (CppState *)context->state;
  allocator = state->allocator;
  bcir_host_arena_destroy(&state->scratch);
  {                                                  /* macro replacement's buffers (expand_text) */
    void *blocks[] = {state->ppx_stack.v, state->ppx_out.v, state->ppx_args.v, state->ppx_sub.v, state->ppx_body.v,
                      state->ppx_pool, state->ppx_text, state->hs_v, state->hs_el, state->hs_tab};
    for (size_t k = 0; k < sizeof blocks / sizeof blocks[0]; k++) bcir_host_deallocate(&allocator, blocks[k]);
  }
  memset(state, 0, sizeof *state);
  bcir_host_deallocate(&allocator, state);
  memset(context, 0, sizeof *context);
}

void bcir_cpp_context_set_target(bcir_cpp_context *context, int long_size, int pointer_size, int long_double,
                                 int msvc, int flt_eval) {
  CppState *state = context ? (CppState *)context->state : NULL;
  if (!state) return;
  state->long_size = long_size == 4 ? 4 : 8;
  state->pointer_size = pointer_size == 4 ? 4 : 8;
  state->long_double = long_double >= 0 && long_double <= 2 ? long_double : 1;
  state->msvc = msvc ? 1 : 0;
  state->flt_eval = flt_eval;
}
void bcir_cpp_context_set_chars(bcir_cpp_context *context, int char_signed, int wchar_size, int wchar_signed) {
  CppState *state;
  if (!context || !context->state) return;
  state = (CppState *)context->state;
  state->char_signed = char_signed ? 1 : 0;
  state->wchar_size = wchar_size == 2 ? 2 : 4;
  state->wchar_signed = wchar_signed ? 1 : 0;
}

int bcir_cpp_run_ex_context(bcir_cpp_context *context, const char *src,
                            const char *srcname, const char *const *dirs, int ndirs,
                            const char *const *defines, int ndefines,
                            char *out, size_t outcap, char *err, size_t errcap) {
  CppState *state;
  int rc = 1;
  if(err&&errcap)err[0]=0;
  if(out&&outcap)out[0]=0;
  if(!context || !context->state){
    if(err&&errcap)snprintf(err,errcap,"uninitialized preprocessor context");
    return 1;
  }
  state=(CppState *)context->state;
  bcir_cpp_context_reset(context);
  state->out_buf=out; state->out_full=0;
  if(!src||!out||!outcap||ndirs<0||ndirs>BCIR_CPP_MAX_DIRS||
     ndefines<0||ndefines>BCIR_CPP_MAX_DEFINES||(ndirs&&!dirs)||(ndefines&&!defines)){
    if(err&&errcap)snprintf(err,errcap,"invalid preprocessor arguments");
    return 1;
  }
  for(int d=0;d<ndirs;d++)if(!dirs[d]){
    if(err&&errcap)snprintf(err,errcap,"invalid preprocessor include directory");return 1;}
  for(int d=0;d<ndefines;d++)if(!defines[d]){
    if(err&&errcap)snprintf(err,errcap,"invalid preprocessor definition");return 1;}
  define_macro(state,"__STDC_VERSION__ 202311L"); define_macro(state,"__STDC__ 1");
  define_macro(state,"__STDC_HOSTED__ 1");
  { char db[16], tb[16]; cpp_datetime(db, tb); char def[48];
    snprintf(def, sizeof def, "__DATE__ \"%s\"", db); define_macro(state,def);
    snprintf(def, sizeof def, "__TIME__ \"%s\"", tb); define_macro(state,def); }
  for(int d=0; d<ndefines; d++) define_macro(state,defines[d]);
  if(state->limit_error){if(err&&errcap)snprintf(err,errcap,"%s",state->limit_error);goto cleanup;}
  {
    size_t sl=0, scratch_size, w=0;
    char *stripped;
    while(sl<=BCIR_CPP_MAX_SOURCE_BYTES&&src[sl])sl++;
    if(sl>BCIR_CPP_MAX_SOURCE_BYTES){
      if(err&&errcap)snprintf(err,errcap,"source exceeds %zu bytes",BCIR_CPP_MAX_SOURCE_BYTES);
      goto cleanup;
    }
    if(!bcir_size_add(sl,1u,&scratch_size)){
      if(err&&errcap)snprintf(err,errcap,"source is too large");
      goto cleanup;
    }
    stripped=(char *)bcir_host_arena_allocate(&state->scratch,scratch_size,1u);
    if(!stripped){if(err&&errcap)snprintf(err,errcap,"oom");goto cleanup;}
    strip_comments(src, stripped, scratch_size);
    rc=cpp_process(state,stripped, (srcname && srcname[0]) ? srcname : "<source>",
                   dirs, ndirs, out, outcap, &w, err, errcap);
  }
  if(!rc&&state->limit_error){if(err&&errcap)snprintf(err,errcap,"%s",state->limit_error);rc=1;}
cleanup:
  bcir_host_arena_reset(&state->scratch);
  return rc;
}

/* The preprocessed text in a block grown as it needs (CF-PPLIMITS): the run is repeated with a buffer four times the
 * last while the output alone filled it, to BCIR_CPP_MAX_OUTPUT_BYTES -- each run starts from a reset context, so
 * only the last one is seen. The twin's drivers had held the text in 64 KiB: a unit of some 6 000 globals was refused
 * there that the oracle lowered. */
#define BCIR_CPP_MAX_OUTPUT_BYTES ((size_t)64u << 20)
int bcir_cpp_run_ex_alloc_context(bcir_cpp_context *context, const char *src, const char *srcname,
                                  const char *const *dirs, int ndirs, const char *const *defines, int ndefines,
                                  char **out, size_t *outlen, char *err, size_t errcap) {
  size_t cap=(size_t)1u<<16;
  if(out)*out=NULL;
  if(outlen)*outlen=0;
  if(!context||!context->state||!out){
    if(err&&errcap)snprintf(err,errcap,"invalid preprocessor arguments");
    return 1;
  }
  for(;;){
    char *buf=(char *)bcir_host_allocate(&context->allocator,cap);
    int rc;
    if(!buf){if(err&&errcap)snprintf(err,errcap,"oom");return 1;}
    rc=bcir_cpp_run_ex_context(context,src,srcname,dirs,ndirs,defines,ndefines,buf,cap,err,errcap);
    if(!rc){*out=buf;if(outlen)*outlen=strlen(buf);return 0;}
    bcir_host_deallocate(&context->allocator,buf);
    if(!((CppState *)context->state)->out_full||cap>BCIR_CPP_MAX_OUTPUT_BYTES)return rc;
    cap=cap>BCIR_CPP_MAX_OUTPUT_BYTES/4u?BCIR_CPP_MAX_OUTPUT_BYTES+1u:cap*4u;   /* the bound, and a NUL */
  }
}

/* The whole translation unit, NUL-terminated, in a block `allocator` holds (CF-LIMITS, CF-PPLIMITS): it grows two-phase,
 * each size checked, to BCIR_CPP_MAX_SOURCE_BYTES; a file that cannot be handed on whole -- past that bound,
 * unreadable, or holding a NUL a C string would stop at -- is refused, never cut. The twin's three drivers read
 * through it: the loop driver had read the first 64 KiB of a source and lowered that prefix. */
const char *bcir_cpp_read_source(FILE *fp, const bcir_host_allocator *allocator, char **out) {
  bcir_host_allocator heap=bcir_host_allocator_or_default(allocator);
  size_t cap=(size_t)1u<<16, n=0;
  char *buf;
  if(!out) return "invalid arguments";
  *out=NULL;
  if(!fp) return "cannot read the source";
  buf=(char *)bcir_host_allocate(&heap,cap);
  if(!buf) return "out of memory";
  for(;;){
    if(n==cap-1u){                          /* full: past the bound one byte more is refused, else the block grows */
      size_t want, grown;
      if(cap>=BCIR_CPP_MAX_SOURCE_BYTES+1u){
        int extra=fgetc(fp);
        if(extra==EOF&&!ferror(fp)) break;
        bcir_host_deallocate(&heap,buf);
        return extra==EOF?"cannot read the source":"the source is larger than 64 MiB";
      }
      if(!bcir_size_add(cap,1u,&want)||!bcir_host_grow_capacity(cap,want,1u,&grown)){
        bcir_host_deallocate(&heap,buf); return "out of memory"; }
      if(grown>BCIR_CPP_MAX_SOURCE_BYTES+1u) grown=BCIR_CPP_MAX_SOURCE_BYTES+1u;
      if(!bcir_host_realloc_array(&heap,(void **)&buf,cap,grown,1u,0)){
        bcir_host_deallocate(&heap,buf); return "out of memory"; }
      cap=grown;
    }
    size_t avail=cap-1u-n, got=fread(buf+n,1,avail,fp);
    n+=got;
    if(got<avail){
      if(ferror(fp)||(!feof(fp)&&!got)){ bcir_host_deallocate(&heap,buf); return "cannot read the source"; }
      if(feof(fp)) break;
    }
  }
  if(memchr(buf,0,n)){ bcir_host_deallocate(&heap,buf); return "the source holds a NUL byte"; }
  buf[n]=0; *out=buf;
  return NULL;
}

int bcir_cpp_run_context(bcir_cpp_context *context, const char *src,
                         const char *basedir, char *out, size_t outcap,
                         char *err, size_t errcap) {
  const char *d[1]; int nd=0; if(basedir&&basedir[0]){ d[0]=basedir; nd=1; }
  return bcir_cpp_run_ex_context(context,src,"<source>",d,nd,NULL,0,out,outcap,err,errcap);
}

static bcir_cpp_context *legacy_context(char *err, size_t errcap) {
  static bcir_cpp_context context;
  static int initialized;
  if(!initialized){
    if(bcir_cpp_context_init(&context,NULL)){
      if(err&&errcap)snprintf(err,errcap,"oom");
      return NULL;
    }
    initialized=1;
  }
  return &context;
}

int bcir_cpp_run_ex(const char *src, const char *srcname, const char *const *dirs, int ndirs,
                    const char *const *defines, int ndefines,
                    char *out, size_t outcap, char *err, size_t errcap) {
  bcir_cpp_context *context=legacy_context(err,errcap);
  return context?bcir_cpp_run_ex_context(context,src,srcname,dirs,ndirs,defines,ndefines,
                                          out,outcap,err,errcap):1;
}

int bcir_cpp_run(const char *src, const char *basedir, char *out, size_t outcap,
                 char *err, size_t errcap) {
  bcir_cpp_context *context=legacy_context(err,errcap);
  return context?bcir_cpp_run_context(context,src,basedir,out,outcap,err,errcap):1;
}

int bcir_cpp_run_ex_alloc(const char *src, const char *srcname, const char *const *dirs, int ndirs,
                          const char *const *defines, int ndefines, char **out, size_t *outlen,
                          char *err, size_t errcap) {
  bcir_cpp_context *context=legacy_context(err,errcap);
  if(out)*out=NULL;
  return context?bcir_cpp_run_ex_alloc_context(context,src,srcname,dirs,ndirs,defines,ndefines,
                                                out,outlen,err,errcap):1;
}

int bcir_cpp_run_alloc(const char *src, const char *basedir, char **out, size_t *outlen, char *err, size_t errcap) {
  const char *d[1]; int nd=0; if(basedir&&basedir[0]){ d[0]=basedir; nd=1; }
  return bcir_cpp_run_ex_alloc(src,"<source>",d,nd,NULL,0,out,outlen,err,errcap);
}

void bcir_cpp_set_target(int long_size, int pointer_size, int long_double, int msvc, int flt_eval) {
  bcir_cpp_context *context = legacy_context(NULL, 0);
  if(context) bcir_cpp_context_set_target(context,long_size,pointer_size,long_double,msvc,flt_eval);
}
void bcir_cpp_set_chars(int char_signed, int wchar_size, int wchar_signed) {
  bcir_cpp_context *context=legacy_context(NULL,0);
  if(context) bcir_cpp_context_set_chars(context,char_signed,wchar_size,wchar_signed);
}
