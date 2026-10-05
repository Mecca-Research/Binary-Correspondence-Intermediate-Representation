/*===- bcir_make.c - bcir-make, the C twin of BCIR Make's judge and planner ===*/
/*
 * bcir-make --dry-run [-f BCIRfile] [--root DIR] [--state FILE] [--workers N] [--check-tools]
 *
 * The C twin of the oracle's `bcir-make --dry-run` (bcir/make/, BUILD-6; docs/BCIR_BUILD_ROADMAP.md
 * §8): the version-1 grammar (MK0), the laws MK1-MK5 over the lowered graph, every target's
 * generation tag, the run-or-reuse decision against a recorded state, and the waves of at most
 * --workers targets, printed as the oracle prints them, byte for byte. tools/build/make_parity.py
 * holds the two rails to that over a generated corpus of BCIRfiles (BUILD-8). The twin judges and
 * plans; the oracle's runner (python -m bcir.make) runs.
 *
 * It follows the oracle step by step, in the same orders: the grammar's checks line by line and in
 * the parser's order, the findings law by law and target by target, the phase ids in file order,
 * a phase's dependencies in the order its reads and `after` name them, and the IR's iterative
 * depth-first walks (bcir/model/graph.py) for the cycle check, the cycle's witness and the
 * canonical order. Paths are compared as bytes, which is the oracle's order for the ASCII the
 * grammar admits; a tree file whose path the grammar cannot spell is a finding on both rails
 * (MK4), so a tag never names one.
 *
 * Hosted tool (runtime/c/MEMORY_CLASSIFICATION.txt): every allocation is one arena's, over the host
 * allocator, released at exit. Every exit is a verdict (docs/security/laws.md L1): 0 with a plan,
 * 1 when the BCIRfile breaks a law, 2 when nothing can be judged (an unreadable file, a state that
 * is not one, a bad option, no memory), with the reason on stderr. POSIX only, as BCIR Make is.
 */
#define _XOPEN_SOURCE 700 /* opendir, lstat and realpath under a strict -std */

#include "bcir_host_alloc.h"
#include "bcir_sha256.h"

#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define MK_MAX_BYTES (16u * 1024u * 1024u)
#define MK_MAX_LINE 65536u
#define MK_MAX_TARGETS 100000u
#define MK_MAX_STATE (64u * 1024u * 1024u)
#define MK_TAG_HEADER "bcir-make generation 1"

/* ------------------------------------------------------------------ memory, strings, maps */

typedef struct mk_ctx {
  bcir_host_arena arena;
  const char *root; /* the resolved root, no trailing slash */
  int check_tools;
} mk_ctx;

static void mk_unusable(const char *what, const char *detail) {
  fprintf(stderr, "bcir-make: UNUSABLE: %s%s%s\n", what, detail ? ": " : "", detail ? detail : "");
  exit(2);
}

static void *mk_alloc(mk_ctx *mk, size_t size) {
  void *p = bcir_host_arena_allocate(&mk->arena, size, 0u);
  if (!p) mk_unusable("out of memory", NULL);
  return p;
}

static char *mk_strndup(mk_ctx *mk, const char *s, size_t n) {
  char *p = (char *)mk_alloc(mk, n + 1u);
  memcpy(p, s, n);
  p[n] = '\0';
  return p;
}

static char *mk_strdup(mk_ctx *mk, const char *s) { return mk_strndup(mk, s, strlen(s)); }

typedef struct mk_buf {
  char *p;
  size_t n, cap;
} mk_buf;

static void buf_put(mk_ctx *mk, mk_buf *b, const char *s, size_t n) {
  if (b->n + n + 1u > b->cap) {
    size_t cap = b->cap ? b->cap : 256u;
    char *np;
    while (cap < b->n + n + 1u) {
      if (cap > SIZE_MAX / 2u) mk_unusable("out of memory", NULL);
      cap *= 2u;
    }
    np = (char *)mk_alloc(mk, cap);
    if (b->n) memcpy(np, b->p, b->n);
    b->p = np;
    b->cap = cap;
  }
  memcpy(b->p + b->n, s, n);
  b->n += n;
  b->p[b->n] = '\0';
}

static void buf_puts(mk_ctx *mk, mk_buf *b, const char *s) { buf_put(mk, b, s, strlen(s)); }

static void buf_putc(mk_ctx *mk, mk_buf *b, char c) { buf_put(mk, b, &c, 1u); }

static void buf_printf(mk_ctx *mk, mk_buf *b, const char *fmt, ...) {
  char small[512];
  va_list ap;
  int n;
  va_start(ap, fmt);
  n = vsnprintf(small, sizeof small, fmt, ap);
  va_end(ap);
  if (n < 0) mk_unusable("a message could not be formatted", NULL);
  if ((size_t)n < sizeof small) {
    buf_put(mk, b, small, (size_t)n);
    return;
  }
  {
    char *big = (char *)mk_alloc(mk, (size_t)n + 1u);
    va_start(ap, fmt);
    (void)vsnprintf(big, (size_t)n + 1u, fmt, ap);
    va_end(ap);
    buf_put(mk, b, big, (size_t)n);
  }
}

/* Python's repr() of a printable-ASCII string, which is all a BCIRfile token can be: single quotes
 * unless the text holds one and no double quote, a backslash doubled, the quote escaped. */
static void buf_repr(mk_ctx *mk, mk_buf *b, const char *s) {
  char q = (strchr(s, '\'') && !strchr(s, '"')) ? '"' : '\'';
  buf_putc(mk, b, q);
  for (; *s; s++) {
    if (*s == '\\') {
      buf_puts(mk, b, "\\\\");
    } else if (*s == q) {
      buf_putc(mk, b, '\\');
      buf_putc(mk, b, q);
    } else {
      buf_putc(mk, b, *s);
    }
  }
  buf_putc(mk, b, q);
}

typedef struct mk_strs {
  char **v;
  size_t n, cap;
} mk_strs;

static void strs_push(mk_ctx *mk, mk_strs *s, char *x) {
  if (s->n == s->cap) {
    size_t cap = s->cap ? s->cap * 2u : 8u;
    char **nv = (char **)mk_alloc(mk, cap * sizeof *nv);
    if (s->n) memcpy(nv, s->v, s->n * sizeof *nv);
    s->v = nv;
    s->cap = cap;
  }
  s->v[s->n++] = x;
}

static int strs_has(const mk_strs *s, const char *x) {
  size_t i;
  for (i = 0; i < s->n; i++)
    if (!strcmp(s->v[i], x)) return 1;
  return 0;
}

static int cmp_str(const void *a, const void *b) {
  return strcmp(*(char *const *)a, *(char *const *)b);
}

/* A sorted copy, duplicates dropped (Python's sorted(set(...))). */
static mk_strs strs_sorted_unique(mk_ctx *mk, char **v, size_t n) {
  mk_strs out = {0};
  size_t i;
  char **copy;
  if (!n) return out;
  copy = (char **)mk_alloc(mk, n * sizeof *copy);
  memcpy(copy, v, n * sizeof *copy);
  qsort(copy, n, sizeof *copy, cmp_str);
  for (i = 0; i < n; i++)
    if (!out.n || strcmp(out.v[out.n - 1u], copy[i])) strs_push(mk, &out, copy[i]);
  return out;
}

typedef struct mk_ints {
  size_t *v;
  size_t n, cap;
} mk_ints;

static void ints_push(mk_ctx *mk, mk_ints *s, size_t x) {
  if (s->n == s->cap) {
    size_t cap = s->cap ? s->cap * 2u : 8u;
    size_t *nv = (size_t *)mk_alloc(mk, cap * sizeof *nv);
    if (s->n) memcpy(nv, s->v, s->n * sizeof *nv);
    s->v = nv;
    s->cap = cap;
  }
  s->v[s->n++] = x;
}

static int ints_has(const mk_ints *s, size_t x) {
  size_t i;
  for (i = 0; i < s->n; i++)
    if (s->v[i] == x) return 1;
  return 0;
}

/* string -> pointer, open addressing; a key's bytes are compared with its length, so a key holding
 * a NUL (a state's "\u0000") never equals a name. */
typedef struct mk_entry {
  const char *key; /* NULL: an empty slot */
  size_t len;
  void *val;
} mk_entry;

typedef struct mk_map {
  mk_entry *slots; /* NULL until the first insertion; cap is a power of two */
  size_t cap, n;
} mk_map;

static uint64_t mk_hash(const char *s, size_t n) {
  uint64_t h = 1469598103934665603ull;
  size_t i;
  for (i = 0; i < n; i++) {
    h ^= (unsigned char)s[i];
    h *= 1099511628211ull;
  }
  return h;
}

static void **map_slot(mk_ctx *mk, mk_map *m, const char *k, size_t n, int create) {
  size_t i;
  if (create && (!m->slots || (m->n + 1u) * 10u >= m->cap * 7u)) {
    mk_map grown;
    size_t j;
    grown.cap = m->slots ? m->cap * 2u : 64u;
    grown.n = 0;
    grown.slots = (mk_entry *)mk_alloc(mk, grown.cap * sizeof *grown.slots);
    memset(grown.slots, 0, grown.cap * sizeof *grown.slots);
    for (j = 0; m->slots && j < m->cap; j++)
      if (m->slots[j].key)
        *map_slot(mk, &grown, m->slots[j].key, m->slots[j].len, 1) = m->slots[j].val;
    *m = grown;
  }
  if (!m->slots) return NULL;
  i = (size_t)(mk_hash(k, n) & (m->cap - 1u));
  for (;;) {
    mk_entry *e = &m->slots[i];
    if (!e->key) {
      if (!create) return NULL;
      e->key = k;
      e->len = n;
      e->val = NULL;
      m->n++;
      return &e->val;
    }
    if (e->len == n && !memcmp(e->key, k, n)) return &e->val;
    i = (i + 1u) & (m->cap - 1u);
  }
}

static void *map_get(const mk_map *m, const char *k) {
  void **slot = map_slot(NULL, (mk_map *)m, k, strlen(k), 0);
  return slot ? *slot : NULL;
}

static int map_has(const mk_map *m, const char *k) {
  return map_slot(NULL, (mk_map *)m, k, strlen(k), 0) != NULL;
}

static void map_put(mk_ctx *mk, mk_map *m, const char *k, void *v) {
  *map_slot(mk, m, k, strlen(k), 1) = v;
}

/* ------------------------------------------------------------------ the grammar (MK0) */

typedef struct mk_tool {
  char *name, *path, *identity;
} mk_tool;

typedef struct mk_cmd {
  char **argv;
  size_t argc;
} mk_cmd;

typedef struct mk_target {
  char *name;
  size_t index; /* file order; the phase id is index + 1 */
  mk_strs reads, reads_tree, writes, after, uses;
  mk_cmd *runs;
  size_t nruns, cap_runs;
  mk_ints deps; /* target indices, in the order lower() appends them */
} mk_target;

typedef struct mk_file {
  mk_tool **tools;
  size_t ntools, cap_tools;
  mk_target **targets;
  size_t ntargets, cap_targets;
  mk_map tool_by_name, target_by_name;
} mk_file;

static int is_name(const char *s) {
  const char *p;
  if (!((*s >= 'A' && *s <= 'Z') || (*s >= 'a' && *s <= 'z') || (*s >= '0' && *s <= '9')))
    return 0;
  for (p = s + 1; *p; p++)
    if (!((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') || (*p >= '0' && *p <= '9') ||
          *p == '_' || *p == '.' || *p == '+' || *p == '-'))
      return 0;
  return 1;
}

static int is_segment_char(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' ||
         c == '.' || c == '+' || c == '-';
}

/* bcir.make.grammar.is_repo_path: non-empty segments of [A-Za-z0-9_.+-] joined by '/', never '.'
 * or '..'. */
static int is_repo_path_n(const char *s, size_t n) {
  size_t i = 0;
  for (;;) {
    size_t start = i;
    while (i < n && s[i] != '/') {
      if (!is_segment_char(s[i])) return 0;
      i++;
    }
    if (i == start) return 0;
    if ((i - start == 1u && s[start] == '.') ||
        (i - start == 2u && s[start] == '.' && s[start + 1u] == '.'))
      return 0;
    if (i == n) return 1;
    i++; /* the '/' */
  }
}

static int is_repo_path(const char *s) { return is_repo_path_n(s, strlen(s)); }

static int is_tool_path(const char *s) {
  if (s[0] == '/') return s[1] != '\0' && is_repo_path(s + 1);
  return is_repo_path(s);
}

static int is_identity(const char *s) {
  size_t i;
  if (strncmp(s, "sha256:", 7u)) return 0;
  for (i = 0; i < 64u; i++) {
    char c = s[7u + i];
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return 0;
  }
  return s[71] == '\0';
}

typedef struct mk_error {
  size_t line; /* 0: the file as a whole */
  mk_buf message;
} mk_error;

static const char *const mk_attributes[] = {"reads", "reads-tree", "writes", "after", "uses", "run"};

static int mk_parse(mk_ctx *mk, const char *data, size_t len, mk_file *out, mk_error *err) {
  size_t i, number = 0, start;
  int header_seen = 0;
  mk_target *current = NULL;
  memset(out, 0, sizeof *out);
  memset(err, 0, sizeof *err);
  if (len > MK_MAX_BYTES) {
    buf_printf(mk, &err->message, "the file is over the bound of %u bytes", MK_MAX_BYTES);
    return 0;
  }
  for (i = 0; i < len; i++) {
    unsigned char b = (unsigned char)data[i];
    if (b > 0x7Eu || (b < 0x20u && b != 0x0Au)) {
      size_t k, line = 1;
      for (k = 0; k < i; k++)
        if (data[k] == '\n') line++;
      err->line = line;
      buf_printf(mk, &err->message, "byte 0x%02x is not printable ASCII or a line feed", b);
      return 0;
    }
  }
  if (len && data[len - 1u] != '\n') {
    size_t k, line = 1;
    for (k = 0; k < len; k++)
      if (data[k] == '\n') line++;
    err->line = line;
    buf_puts(mk, &err->message, "the last line has no line feed");
    return 0;
  }
  start = 0;
  while (start < len) {
    size_t end = start, n;
    char *line, *body, **tokens, *key;
    size_t ntok, k, nargs;
    int indented;
    while (data[end] != '\n') end++;
    n = end - start;
    number++;
    line = mk_strndup(mk, data + start, n);
    start = end + 1u;
    if (n > MK_MAX_LINE) {
      err->line = number;
      buf_printf(mk, &err->message, "the line is %zu bytes; the bound is %u", n, MK_MAX_LINE);
      return 0;
    }
    if (!n) continue;
    if (line[n - 1u] == ' ') {
      err->line = number;
      buf_puts(mk, &err->message, "trailing space");
      return 0;
    }
    {
      const char *p = line;
      while (*p == ' ') p++;
      if (*p == '#') continue;
    }
    if (!header_seen) {
      if (strcmp(line, "bcirfile 1")) {
        err->line = number;
        buf_puts(mk, &err->message, "the first line is not `bcirfile 1`");
        return 0;
      }
      header_seen = 1;
      continue;
    }
    indented = n >= 2u && line[0] == ' ' && line[1] == ' ';
    body = indented ? line + 2 : line;
    if (body[0] == ' ') {
      err->line = number;
      buf_puts(mk, &err->message, "indentation is exactly two spaces, on a target's attribute");
      return 0;
    }
    /* split on single spaces; an empty token is two spaces in a row */
    ntok = 1;
    for (k = 0; body[k]; k++)
      if (body[k] == ' ') ntok++;
    tokens = (char **)mk_alloc(mk, ntok * sizeof *tokens);
    {
      char *p = body;
      size_t t = 0;
      tokens[t++] = p;
      for (; *p; p++)
        if (*p == ' ') {
          *p = '\0';
          tokens[t++] = p + 1;
        }
      for (t = 0; t < ntok; t++)
        if (!tokens[t][0]) {
          err->line = number;
          buf_puts(mk, &err->message, "tokens are separated by exactly one space");
          return 0;
        }
    }
    key = tokens[0];
    nargs = ntok - 1u;
    if (!indented) {
      if (!strcmp(key, "tool")) {
        mk_tool *tool;
        if (nargs != 3u) {
          err->line = number;
          buf_puts(mk, &err->message, "`tool NAME PATH IDENTITY`");
          return 0;
        }
        if (!is_name(tokens[1])) {
          err->line = number;
          buf_puts(mk, &err->message, "tool name ");
          buf_repr(mk, &err->message, tokens[1]);
          buf_puts(mk, &err->message, " is not [A-Za-z0-9][A-Za-z0-9_.+-]*");
          return 0;
        }
        if (!is_tool_path(tokens[2])) {
          err->line = number;
          buf_puts(mk, &err->message, "tool path ");
          buf_repr(mk, &err->message, tokens[2]);
          buf_puts(mk, &err->message, " is not a path");
          return 0;
        }
        if (!is_identity(tokens[3])) {
          err->line = number;
          buf_puts(mk, &err->message, "tool identity ");
          buf_repr(mk, &err->message, tokens[3]);
          buf_puts(mk, &err->message, " is not sha256:<64 hex>");
          return 0;
        }
        if (map_has(&out->tool_by_name, tokens[1]) || map_has(&out->target_by_name, tokens[1])) {
          err->line = number;
          buf_printf(mk, &err->message, "%s is declared twice", tokens[1]);
          return 0;
        }
        tool = (mk_tool *)mk_alloc(mk, sizeof *tool);
        tool->name = tokens[1];
        tool->path = tokens[2];
        tool->identity = tokens[3];
        if (out->ntools == out->cap_tools) {
          size_t cap = out->cap_tools ? out->cap_tools * 2u : 8u;
          mk_tool **nv = (mk_tool **)mk_alloc(mk, cap * sizeof *nv);
          if (out->ntools) memcpy(nv, out->tools, out->ntools * sizeof *nv);
          out->tools = nv;
          out->cap_tools = cap;
        }
        out->tools[out->ntools++] = tool;
        map_put(mk, &out->tool_by_name, tool->name, tool);
        current = NULL;
      } else if (!strcmp(key, "target")) {
        mk_target *t;
        if (nargs != 1u) {
          err->line = number;
          buf_puts(mk, &err->message, "`target NAME`");
          return 0;
        }
        if (!is_name(tokens[1])) {
          err->line = number;
          buf_puts(mk, &err->message, "target name ");
          buf_repr(mk, &err->message, tokens[1]);
          buf_puts(mk, &err->message, " is not [A-Za-z0-9][A-Za-z0-9_.+-]*");
          return 0;
        }
        if (map_has(&out->target_by_name, tokens[1]) || map_has(&out->tool_by_name, tokens[1])) {
          err->line = number;
          buf_printf(mk, &err->message, "%s is declared twice", tokens[1]);
          return 0;
        }
        if (out->ntargets >= MK_MAX_TARGETS) {
          err->line = number;
          buf_printf(mk, &err->message, "more than %u targets", MK_MAX_TARGETS);
          return 0;
        }
        t = (mk_target *)mk_alloc(mk, sizeof *t);
        memset(t, 0, sizeof *t);
        t->name = tokens[1];
        t->index = out->ntargets;
        if (out->ntargets == out->cap_targets) {
          size_t cap = out->cap_targets ? out->cap_targets * 2u : 8u;
          mk_target **nv = (mk_target **)mk_alloc(mk, cap * sizeof *nv);
          if (out->ntargets) memcpy(nv, out->targets, out->ntargets * sizeof *nv);
          out->targets = nv;
          out->cap_targets = cap;
        }
        out->targets[out->ntargets++] = t;
        map_put(mk, &out->target_by_name, t->name, t);
        current = t;
      } else {
        err->line = number;
        buf_repr(mk, &err->message, key);
        buf_puts(mk, &err->message, " is not `tool` or `target`");
        return 0;
      }
      continue;
    }
    if (!current) {
      err->line = number;
      buf_puts(mk, &err->message, "an attribute outside a target");
      return 0;
    }
    {
      size_t a;
      mk_strs *bucket = NULL;
      for (a = 0; a < sizeof mk_attributes / sizeof mk_attributes[0]; a++)
        if (!strcmp(key, mk_attributes[a])) break;
      if (a == sizeof mk_attributes / sizeof mk_attributes[0]) {
        err->line = number;
        buf_repr(mk, &err->message, key);
        buf_puts(mk, &err->message, " is not one of reads, reads-tree, writes, after, uses, run");
        return 0;
      }
      if (!nargs) {
        err->line = number;
        buf_printf(mk, &err->message, "`%s` names nothing", key);
        return 0;
      }
      if (!strcmp(key, "run")) {
        mk_cmd *cmd;
        if (!is_name(tokens[1])) {
          err->line = number;
          buf_puts(mk, &err->message, "a command starts with a tool name, not ");
          buf_repr(mk, &err->message, tokens[1]);
          return 0;
        }
        if (current->nruns == current->cap_runs) {
          size_t cap = current->cap_runs ? current->cap_runs * 2u : 4u;
          mk_cmd *nv = (mk_cmd *)mk_alloc(mk, cap * sizeof *nv);
          if (current->nruns) memcpy(nv, current->runs, current->nruns * sizeof *nv);
          current->runs = nv;
          current->cap_runs = cap;
        }
        cmd = &current->runs[current->nruns++];
        cmd->argv = tokens + 1;
        cmd->argc = nargs;
        continue;
      }
      if (!strcmp(key, "reads")) bucket = &current->reads;
      else if (!strcmp(key, "reads-tree")) bucket = &current->reads_tree;
      else if (!strcmp(key, "writes")) bucket = &current->writes;
      else if (!strcmp(key, "after")) bucket = &current->after;
      else bucket = &current->uses;
      for (k = 1; k < ntok; k++) {
        const char *token = tokens[k];
        if (!strcmp(key, "after") || !strcmp(key, "uses")) {
          if (!is_name(token)) {
            err->line = number;
            buf_printf(mk, &err->message, "`%s` names %s, not ", key,
                       !strcmp(key, "after") ? "a target" : "a tool");
            buf_repr(mk, &err->message, token);
            return 0;
          }
        } else if (!is_repo_path(token)) {
          err->line = number;
          buf_repr(mk, &err->message, token);
          buf_puts(mk, &err->message, " is not a repo-relative path");
          return 0;
        }
        if (strs_has(bucket, token)) {
          err->line = number;
          buf_printf(mk, &err->message, "%s %s %s twice", current->name, key, token);
          return 0;
        }
        strs_push(mk, bucket, tokens[k]);
      }
    }
  }
  if (!header_seen) {
    err->line = 0;
    buf_puts(mk, &err->message, "the file has no `bcirfile 1` line");
    return 0;
  }
  return 1;
}

/* ------------------------------------------------------------------ the tree */

static char *mk_join(mk_ctx *mk, const char *base, const char *rel) {
  size_t a = strlen(base), b = strlen(rel);
  char *p = (char *)mk_alloc(mk, a + b + 2u);
  memcpy(p, base, a);
  p[a] = '/';
  memcpy(p + a + 1u, rel, b + 1u);
  return p;
}

/* Path.is_file / is_dir: stat, which follows a symlink */
static int is_file_at(mk_ctx *mk, const char *rel) {
  struct stat st;
  return stat(mk_join(mk, mk->root, rel), &st) == 0 && S_ISREG(st.st_mode);
}

static int is_dir_at(mk_ctx *mk, const char *rel) {
  struct stat st;
  return stat(mk_join(mk, mk->root, rel), &st) == 0 && S_ISDIR(st.st_mode);
}

typedef struct mk_walk {
  mk_strs files; /* repo paths, sorted */
  size_t unspellable, unlistable;
} mk_walk;

/* bcir.make.laws.walk_tree: every regular file -- never a symlink -- under the directory, skipping
 * a name that starts with '.' and a `__pycache__` directory; a file whose repo path the grammar
 * cannot spell is counted, as is a directory that cannot be listed. An explicit stack, no
 * recursion: a deep tree is data. */
static mk_walk walk_tree(mk_ctx *mk, const char *directory) {
  mk_walk w;
  mk_strs stack = {0};
  memset(&w, 0, sizeof w);
  strs_push(mk, &stack, mk_strdup(mk, directory));
  while (stack.n) {
    char *rel = stack.v[--stack.n];
    DIR *dir = opendir(mk_join(mk, mk->root, rel));
    struct dirent *entry;
    if (!dir) {
      w.unlistable++;
      continue;
    }
    for (;;) {
      struct stat st;
      char *child;
      errno = 0;
      entry = readdir(dir);
      if (!entry) {
        if (errno) w.unlistable++;
        break;
      }
      if (entry->d_name[0] == '.') continue; /* also . and .. */
      child = mk_join(mk, rel, entry->d_name);
      if (lstat(mk_join(mk, mk->root, child), &st) != 0) continue;
      if (S_ISDIR(st.st_mode)) {
        if (strcmp(entry->d_name, "__pycache__")) strs_push(mk, &stack, child);
      } else if (S_ISREG(st.st_mode)) {
        if (is_repo_path(child)) strs_push(mk, &w.files, child);
        else w.unspellable++;
      }
    }
    closedir(dir);
  }
  if (w.files.n) qsort(w.files.v, w.files.n, sizeof *w.files.v, cmp_str);
  return w;
}

/* sha256 of a file's bytes, as 64 lowercase hex digits; 0 when it cannot be read */
static int file_digest(const char *path, char hex[65], int *error) {
  static const char digits[] = "0123456789abcdef";
  unsigned char chunk[65536];
  uint8_t out[32];
  bcir_sha256 h;
  FILE *f = fopen(path, "rb");
  size_t n, i;
  if (!f) {
    *error = errno;
    return 0;
  }
  bcir_sha256_init(&h);
  /* fread returns short only at the end of the file or on an error (C11 7.21.8.1), so a short read
   * is the last one -- never read a stream again in that state -- and ferror tells the two apart */
  do {
    n = fread(chunk, 1u, sizeof chunk, f);
    if (n) bcir_sha256_update(&h, chunk, n);
  } while (n == sizeof chunk);
  if (ferror(f)) {
    *error = errno ? errno : EIO;
    fclose(f);
    return 0;
  }
  fclose(f);
  bcir_sha256_final(&h, out);
  for (i = 0; i < 32u; i++) {
    hex[2u * i] = digits[out[i] >> 4];
    hex[2u * i + 1u] = digits[out[i] & 15u];
  }
  hex[64] = '\0';
  return 1;
}

static void sha256_hex(const char *data, size_t n, char hex[65]) {
  static const char digits[] = "0123456789abcdef";
  uint8_t out[32];
  size_t i;
  bcir_sha256_digest((const uint8_t *)data, n, out);
  for (i = 0; i < 32u; i++) {
    hex[2u * i] = digits[out[i] >> 4];
    hex[2u * i + 1u] = digits[out[i] & 15u];
  }
  hex[64] = '\0';
}

/* ------------------------------------------------------------------ lowering and the IR's walks */

typedef struct mk_lowered {
  mk_map writer;  /* path -> the first target that writes it */
  mk_map writers; /* path -> mk_ints* of every target that writes it, in file order */
} mk_lowered;

static void mk_lower(mk_ctx *mk, mk_file *bf, mk_lowered *lw) {
  size_t i, k;
  memset(lw, 0, sizeof *lw);
  for (i = 0; i < bf->ntargets; i++) {
    mk_target *t = bf->targets[i];
    for (k = 0; k < t->writes.n; k++) {
      void **slot = map_slot(mk, &lw->writers, t->writes.v[k], strlen(t->writes.v[k]), 1);
      if (!*slot) {
        *slot = mk_alloc(mk, sizeof(mk_ints));
        memset(*slot, 0, sizeof(mk_ints));
        map_put(mk, &lw->writer, t->writes.v[k], t);
      }
      ints_push(mk, (mk_ints *)*slot, i);
    }
  }
  for (i = 0; i < bf->ntargets; i++) {
    mk_target *t = bf->targets[i];
    for (k = 0; k < t->reads.n; k++) {
      mk_target *producer = (mk_target *)map_get(&lw->writer, t->reads.v[k]);
      if (producer && producer != t && !ints_has(&t->deps, producer->index))
        ints_push(mk, &t->deps, producer->index);
    }
    for (k = 0; k < t->after.n; k++) {
      mk_target *named = (mk_target *)map_get(&bf->target_by_name, t->after.v[k]);
      if (named && !ints_has(&t->deps, named->index)) ints_push(mk, &t->deps, named->index);
    }
  }
}

typedef struct mk_frame {
  size_t node, next;
} mk_frame;

/* bcir.model.graph.phase_graph_has_cycle */
static int has_cycle(mk_ctx *mk, mk_file *bf) {
  unsigned char *color = (unsigned char *)mk_alloc(mk, bf->ntargets + 1u);
  mk_frame *stack = (mk_frame *)mk_alloc(mk, (bf->ntargets + 1u) * sizeof *stack);
  size_t r;
  memset(color, 0, bf->ntargets + 1u);
  for (r = 0; r < bf->ntargets; r++) {
    size_t top = 0;
    if (color[r]) continue;
    color[r] = 1;
    stack[top].node = r;
    stack[top++].next = 0;
    while (top) {
      mk_frame *f = &stack[top - 1u];
      mk_target *t = bf->targets[f->node];
      int pushed = 0;
      while (f->next < t->deps.n) {
        size_t dep = t->deps.v[f->next++];
        if (color[dep] == 1) return 1;
        if (!color[dep]) {
          color[dep] = 1;
          stack[top].node = dep;
          stack[top++].next = 0;
          pushed = 1;
          break;
        }
      }
      if (!pushed) {
        color[f->node] = 2;
        top--;
      }
    }
  }
  return 0;
}

/* bcir.make.laws._cycle_witness: one cycle's targets, in order */
static mk_ints cycle_witness(mk_ctx *mk, mk_file *bf) {
  unsigned char *color = (unsigned char *)mk_alloc(mk, bf->ntargets + 1u);
  mk_frame *stack = (mk_frame *)mk_alloc(mk, (bf->ntargets + 1u) * sizeof *stack);
  mk_ints path = {0}, none = {0};
  size_t r;
  memset(color, 0, bf->ntargets + 1u);
  for (r = 0; r < bf->ntargets; r++) {
    size_t top = 0;
    if (color[r]) continue;
    path.n = 0;
    stack[top].node = r;
    stack[top++].next = 0;
    color[r] = 1;
    ints_push(mk, &path, r);
    while (top) {
      mk_frame *f = &stack[top - 1u];
      mk_target *t = bf->targets[f->node];
      int pushed = 0;
      while (f->next < t->deps.n) {
        size_t dep = t->deps.v[f->next++];
        if (color[dep] == 1) {
          mk_ints cycle = {0};
          size_t at = 0, k;
          while (path.v[at] != dep) at++;
          for (k = at; k < path.n; k++) ints_push(mk, &cycle, path.v[k]);
          return cycle;
        }
        if (!color[dep]) {
          color[dep] = 1;
          ints_push(mk, &path, dep);
          stack[top].node = dep;
          stack[top++].next = 0;
          pushed = 1;
          break;
        }
      }
      if (!pushed) {
        top--;
        path.n--;
        color[f->node] = 2;
      }
    }
  }
  return none;
}

/* bcir.model.graph.topological_phase_ids: dependency-first, roots in file order */
static mk_ints topo_order(mk_ctx *mk, mk_file *bf) {
  unsigned char *color = (unsigned char *)mk_alloc(mk, bf->ntargets + 1u);
  mk_frame *stack = (mk_frame *)mk_alloc(mk, (bf->ntargets + 1u) * sizeof *stack);
  mk_ints order = {0};
  size_t r;
  memset(color, 0, bf->ntargets + 1u);
  for (r = 0; r < bf->ntargets; r++) {
    size_t top = 0;
    if (color[r]) continue;
    color[r] = 1;
    stack[top].node = r;
    stack[top++].next = 0;
    while (top) {
      mk_frame *f = &stack[top - 1u];
      mk_target *t = bf->targets[f->node];
      int pushed = 0;
      while (f->next < t->deps.n) {
        size_t dep = t->deps.v[f->next++];
        if (!color[dep]) {
          color[dep] = 1;
          stack[top].node = dep;
          stack[top++].next = 0;
          pushed = 1;
          break;
        }
      }
      if (!pushed) {
        color[f->node] = 2;
        ints_push(mk, &order, f->node);
        top--;
      }
    }
  }
  return order;
}

/* ------------------------------------------------------------------ the laws (MK1-MK5) */

static void finding(mk_ctx *mk, mk_strs *out, const char *code, const char *target, mk_buf *msg) {
  mk_buf line = {0};
  if (target && target[0]) buf_printf(mk, &line, "%s %s: %s", code, target, msg->p ? msg->p : "");
  else buf_printf(mk, &line, "%s: %s", code, msg->p ? msg->p : "");
  strs_push(mk, out, line.p);
  msg->n = 0;
  if (msg->p) msg->p[0] = '\0';
}

/* bcir.make.laws._paths_in */
static size_t paths_in(mk_ctx *mk, const char *token, char *out[3]) {
  size_t n = 0, len = strlen(token);
  if (is_repo_path(token)) out[n++] = (char *)token;
  if (token[0] == '-') {
    char c = len > 2u ? token[1] : '\0';
    if (len > 2u && ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) && is_repo_path(token + 2))
      out[n++] = (char *)token + 2;
    {
      const char *eq = strchr(token, '=');
      if (eq && is_repo_path(eq + 1)) out[n++] = mk_strdup(mk, eq + 1);
    }
  }
  return n;
}

static int under_tree(const char *path, const mk_strs *trees) {
  size_t i;
  for (i = 0; i < trees->n; i++) {
    size_t n = strlen(trees->v[i]);
    if (!strcmp(path, trees->v[i]) || (!strncmp(path, trees->v[i], n) && path[n] == '/')) return 1;
  }
  return 0;
}

static const char *tool_path(mk_ctx *mk, const char *path) {
  return path[0] == '/' ? path : mk_join(mk, mk->root, path);
}

static mk_strs mk_check(mk_ctx *mk, mk_file *bf, mk_lowered *lw) {
  mk_strs out = {0};
  mk_buf msg = {0};
  mk_map walks = {0}; /* one walk per tree per check */
  mk_map used = {0};
  size_t i, k, j, m;
  /* MK1 claims */
  for (i = 0; i < bf->ntargets; i++) {
    mk_target *t = bf->targets[i];
    if (!t->writes.n) {
      buf_puts(mk, &msg, "writes nothing: a target is a claim on its outputs");
      finding(mk, &out, "MK1", t->name, &msg);
    }
    if (!t->nruns) {
      buf_puts(mk, &msg, "runs no command");
      finding(mk, &out, "MK1", t->name, &msg);
    }
    for (k = 0; k < t->writes.n; k++) {
      mk_ints *ws = (mk_ints *)map_get(&lw->writers, t->writes.v[k]);
      if (ws->n > 1u && ws->v[0] == i) {
        buf_printf(mk, &msg, "%s has %zu writers (", t->writes.v[k], ws->n);
        for (j = 0; j < ws->n; j++) {
          if (j) buf_puts(mk, &msg, ", ");
          buf_puts(mk, &msg, bf->targets[ws->v[j]]->name);
        }
        buf_putc(mk, &msg, ')');
        finding(mk, &out, "MK1", t->name, &msg);
      }
    }
    {
      mk_strs both = {0}, sorted;
      for (k = 0; k < t->reads.n; k++)
        if (strs_has(&t->writes, t->reads.v[k])) strs_push(mk, &both, t->reads.v[k]);
      sorted = strs_sorted_unique(mk, both.v, both.n);
      for (k = 0; k < sorted.n; k++) {
        buf_printf(mk, &msg, "reads %s, which it writes", sorted.v[k]);
        finding(mk, &out, "MK1", t->name, &msg);
      }
    }
    for (k = 0; k < t->after.n; k++) {
      if (!map_has(&bf->target_by_name, t->after.v[k])) {
        buf_printf(mk, &msg, "after %s, which is no target", t->after.v[k]);
        finding(mk, &out, "MK1", t->name, &msg);
      } else if (!strcmp(t->after.v[k], t->name)) {
        buf_puts(mk, &msg, "is after itself");
        finding(mk, &out, "MK1", t->name, &msg);
      }
    }
  }
  /* MK2 anti-cycle */
  if (has_cycle(mk, bf)) {
    mk_ints w = cycle_witness(mk, bf);
    buf_puts(mk, &msg, "the targets form a cycle: ");
    for (k = 0; k < w.n; k++) {
      if (k) buf_puts(mk, &msg, " -> ");
      buf_puts(mk, &msg, bf->targets[w.v[k]]->name);
    }
    if (w.n) {
      buf_puts(mk, &msg, " -> ");
      buf_puts(mk, &msg, bf->targets[w.v[0]]->name);
    }
    finding(mk, &out, "MK2", w.n ? bf->targets[w.v[0]]->name : "", &msg);
  }
  /* MK3 footprint (static) */
  for (i = 0; i < bf->ntargets; i++) {
    mk_target *t = bf->targets[i];
    mk_strs named = {0};
    for (k = 0; k < t->nruns; k++) {
      for (j = 1; j < t->runs[k].argc; j++) {
        char *cand[3];
        size_t nc = paths_in(mk, t->runs[k].argv[j], cand);
        for (m = 0; m < nc; m++) {
          if (map_has(&lw->writers, cand[m]) || is_file_at(mk, cand[m])) {
            if (!strs_has(&named, cand[m])) strs_push(mk, &named, cand[m]);
            if (!strs_has(&t->reads, cand[m]) && !strs_has(&t->writes, cand[m]) &&
                !under_tree(cand[m], &t->reads_tree)) {
              buf_printf(mk, &msg, "a command names %s, which it neither reads nor writes", cand[m]);
              finding(mk, &out, "MK3", t->name, &msg);
            }
          }
        }
      }
    }
    for (k = 0; k < t->writes.n; k++)
      if (!strs_has(&named, t->writes.v[k])) {
        buf_printf(mk, &msg, "writes %s, which none of its commands names", t->writes.v[k]);
        finding(mk, &out, "MK3", t->name, &msg);
      }
  }
  /* MK4 tags: every input is something */
  for (i = 0; i < bf->ntargets; i++) {
    mk_target *t = bf->targets[i];
    for (k = 0; k < t->reads.n; k++)
      if (!map_has(&lw->writers, t->reads.v[k]) && !is_file_at(mk, t->reads.v[k])) {
        buf_printf(mk, &msg, "reads %s: no file and no target's output", t->reads.v[k]);
        finding(mk, &out, "MK4", t->name, &msg);
      }
    for (k = 0; k < t->reads_tree.n; k++) {
      const char *d = t->reads_tree.v[k];
      mk_walk *w;
      if (!is_dir_at(mk, d)) {
        buf_printf(mk, &msg, "reads-tree %s: no such directory", d);
        finding(mk, &out, "MK4", t->name, &msg);
        continue;
      }
      w = (mk_walk *)map_get(&walks, d);
      if (!w) {
        w = (mk_walk *)mk_alloc(mk, sizeof *w);
        *w = walk_tree(mk, d);
        map_put(mk, &walks, d, w);
      }
      if (w->unspellable) {
        buf_printf(mk, &msg, "reads-tree %s: %zu file(s) under it have paths the grammar cannot spell",
                   d, w->unspellable);
        finding(mk, &out, "MK4", t->name, &msg);
      }
      if (w->unlistable) {
        buf_printf(mk, &msg, "reads-tree %s: %zu director(ies) under it cannot be listed", d,
                   w->unlistable);
        finding(mk, &out, "MK4", t->name, &msg);
      }
    }
  }
  /* MK5 tools */
  for (i = 0; i < bf->ntargets; i++) {
    mk_target *t = bf->targets[i];
    for (k = 0; k < t->nruns; k++) {
      if (!map_has(&bf->tool_by_name, t->runs[k].argv[0])) {
        buf_printf(mk, &msg, "runs %s, which is no declared tool", t->runs[k].argv[0]);
        finding(mk, &out, "MK5", t->name, &msg);
      }
      map_put(mk, &used, t->runs[k].argv[0], t);
    }
    for (k = 0; k < t->uses.n; k++) {
      if (!map_has(&bf->tool_by_name, t->uses.v[k])) {
        buf_printf(mk, &msg, "uses %s, which is no declared tool", t->uses.v[k]);
        finding(mk, &out, "MK5", t->name, &msg);
      }
      map_put(mk, &used, t->uses.v[k], t);
    }
  }
  for (i = 0; i < bf->ntools; i++) {
    mk_tool *tool = bf->tools[i];
    if (!map_has(&used, tool->name)) {
      buf_puts(mk, &msg, "is declared and never run");
      finding(mk, &out, "MK5", tool->name, &msg);
    } else if (mk->check_tools) {
      char hex[65];
      int error = 0;
      if (!file_digest(tool_path(mk, tool->path), hex, &error)) {
        buf_printf(mk, &msg, "%s cannot be read (%s)", tool->path, strerror(error));
        finding(mk, &out, "MK5", tool->name, &msg);
        continue;
      }
      if (strcmp(tool->identity + 7, hex)) {
        buf_printf(mk, &msg, "%s is sha256:%s, not its declared %s", tool->path, hex, tool->identity);
        finding(mk, &out, "MK5", tool->name, &msg);
      }
    }
  }
  return out;
}

/* ------------------------------------------------------------------ tags, waves, decisions */

static const char *digest_of(mk_ctx *mk, mk_map *cache, const char *rel) {
  char *hex = (char *)map_get(cache, rel);
  int error = 0;
  if (hex) return hex;
  hex = (char *)mk_alloc(mk, 65u);
  if (!file_digest(mk_join(mk, mk->root, rel), hex, &error))
    mk_unusable(mk_join(mk, mk->root, rel), strerror(error));
  map_put(mk, cache, rel, hex);
  return hex;
}

/* bcir.make.plan.tag_texts, hashed: each target's generation tag, in topological order */
static char **mk_tags(mk_ctx *mk, mk_file *bf, mk_lowered *lw, const mk_ints *order) {
  char **tags = (char **)mk_alloc(mk, (bf->ntargets + 1u) * sizeof *tags);
  mk_map files = {0}, trees = {0};
  size_t o, k;
  memset(tags, 0, (bf->ntargets + 1u) * sizeof *tags);
  for (o = 0; o < order->n; o++) {
    mk_target *t = bf->targets[order->v[o]];
    mk_buf text = {0};
    mk_strs names = {0}, sorted;
    buf_puts(mk, &text, MK_TAG_HEADER "\n");
    buf_printf(mk, &text, "target %s\n", t->name);
    for (k = 0; k < t->nruns; k++) strs_push(mk, &names, t->runs[k].argv[0]);
    for (k = 0; k < t->uses.n; k++) strs_push(mk, &names, t->uses.v[k]);
    sorted = strs_sorted_unique(mk, names.v, names.n);
    for (k = 0; k < sorted.n; k++) {
      mk_tool *tool = (mk_tool *)map_get(&bf->tool_by_name, sorted.v[k]);
      buf_printf(mk, &text, "tool %s %s\n", sorted.v[k], tool ? tool->identity : "undeclared");
    }
    for (k = 0; k < t->nruns; k++) {
      size_t a;
      buf_puts(mk, &text, "run");
      for (a = 0; a < t->runs[k].argc; a++) {
        buf_putc(mk, &text, ' ');
        buf_puts(mk, &text, t->runs[k].argv[a]);
      }
      buf_putc(mk, &text, '\n');
    }
    sorted = strs_sorted_unique(mk, t->reads.v, t->reads.n);
    for (k = 0; k < sorted.n; k++) {
      mk_target *producer = (mk_target *)map_get(&lw->writer, sorted.v[k]);
      if (producer && producer != t && tags[producer->index])
        buf_printf(mk, &text, "read %s tag:%s\n", sorted.v[k], tags[producer->index]);
      else
        buf_printf(mk, &text, "read %s file:%s\n", sorted.v[k], digest_of(mk, &files, sorted.v[k]));
    }
    sorted = strs_sorted_unique(mk, t->reads_tree.v, t->reads_tree.n);
    for (k = 0; k < sorted.n; k++) {
      char *hex = (char *)map_get(&trees, sorted.v[k]);
      if (!hex) {
        mk_walk w = walk_tree(mk, sorted.v[k]);
        bcir_sha256 h;
        uint8_t out[32];
        size_t f, i;
        static const char digits[] = "0123456789abcdef";
        bcir_sha256_init(&h);
        for (f = 0; f < w.files.n; f++) {
          char file_hex[65];
          int error = 0;
          if (!file_digest(mk_join(mk, mk->root, w.files.v[f]), file_hex, &error))
            mk_unusable(mk_join(mk, mk->root, w.files.v[f]), strerror(error));
          bcir_sha256_update(&h, (const uint8_t *)w.files.v[f], strlen(w.files.v[f]));
          bcir_sha256_update(&h, (const uint8_t *)"\0", 1u);
          bcir_sha256_update(&h, (const uint8_t *)file_hex, 64u);
          bcir_sha256_update(&h, (const uint8_t *)"\n", 1u);
        }
        bcir_sha256_final(&h, out);
        hex = (char *)mk_alloc(mk, 65u);
        for (i = 0; i < 32u; i++) {
          hex[2u * i] = digits[out[i] >> 4];
          hex[2u * i + 1u] = digits[out[i] & 15u];
        }
        hex[64] = '\0';
        map_put(mk, &trees, sorted.v[k], hex);
      }
      buf_printf(mk, &text, "tree %s %s\n", sorted.v[k], hex);
    }
    sorted = strs_sorted_unique(mk, t->writes.v, t->writes.n);
    for (k = 0; k < sorted.n; k++) buf_printf(mk, &text, "write %s\n", sorted.v[k]);
    tags[t->index] = (char *)mk_alloc(mk, 65u);
    sha256_hex(text.p, text.n, tags[t->index]);
  }
  return tags;
}

/* bcir.make.plan.schedule: waves of at most `workers` in the canonical order */
static mk_ints *mk_schedule(mk_ctx *mk, mk_file *bf, const mk_ints *order, size_t workers,
                            size_t *nwaves) {
  size_t *wave_of = (size_t *)mk_alloc(mk, (bf->ntargets + 1u) * sizeof *wave_of);
  mk_ints *waves = (mk_ints *)mk_alloc(mk, (bf->ntargets + 1u) * sizeof *waves);
  size_t o, k, n = 0;
  memset(waves, 0, (bf->ntargets + 1u) * sizeof *waves);
  for (o = 0; o < order->n; o++) {
    mk_target *t = bf->targets[order->v[o]];
    size_t w = 0;
    for (k = 0; k < t->deps.n; k++)
      if (wave_of[t->deps.v[k]] + 1u > w) w = wave_of[t->deps.v[k]] + 1u;
    while (w < n && waves[w].n >= workers) w++;
    if (w >= n) n = w + 1u;
    ints_push(mk, &waves[w], t->index);
    wave_of[t->index] = w;
  }
  *nwaves = n;
  return waves;
}

/* ------------------------------------------------------------------ the recorded state */

typedef struct mk_json {
  const unsigned char *p, *end;
} mk_json;

static int utf8_valid(const unsigned char *s, size_t n) {
  size_t i = 0;
  while (i < n) {
    unsigned char c = s[i];
    size_t need;
    uint32_t cp;
    if (c < 0x80u) {
      i++;
      continue;
    }
    if (c >= 0xC2u && c <= 0xDFu) {
      need = 1;
      cp = c & 0x1Fu;
    } else if (c >= 0xE0u && c <= 0xEFu) {
      need = 2;
      cp = c & 0x0Fu;
    } else if (c >= 0xF0u && c <= 0xF4u) {
      need = 3;
      cp = c & 0x07u;
    } else {
      return 0;
    }
    if (n - i <= need) return 0;
    {
      size_t k;
      for (k = 1; k <= need; k++) {
        if ((s[i + k] & 0xC0u) != 0x80u) return 0;
        cp = (cp << 6) | (s[i + k] & 0x3Fu);
      }
    }
    if ((need == 2u && (cp < 0x800u || (cp >= 0xD800u && cp <= 0xDFFFu))) ||
        (need == 3u && (cp < 0x10000u || cp > 0x10FFFFu)))
      return 0;
    i += need + 1u;
  }
  return 1;
}

static void json_ws(mk_json *j) {
  while (j->p < j->end && (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r')) j->p++;
}

static int hex4(const unsigned char *p, uint32_t *out) {
  uint32_t v = 0;
  int i;
  for (i = 0; i < 4; i++) {
    unsigned char c = p[i];
    v <<= 4;
    if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
    else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
    else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
    else return 0;
  }
  *out = v;
  return 1;
}

static void put_cp(mk_ctx *mk, mk_buf *b, uint32_t cp) {
  char u[4];
  if (cp < 0x80u) {
    u[0] = (char)cp;
    buf_put(mk, b, u, 1u);
  } else if (cp < 0x800u) {
    u[0] = (char)(0xC0u | (cp >> 6));
    u[1] = (char)(0x80u | (cp & 0x3Fu));
    buf_put(mk, b, u, 2u);
  } else if (cp < 0x10000u) { /* a lone surrogate too: it can equal only itself */
    u[0] = (char)(0xE0u | (cp >> 12));
    u[1] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
    u[2] = (char)(0x80u | (cp & 0x3Fu));
    buf_put(mk, b, u, 3u);
  } else {
    u[0] = (char)(0xF0u | (cp >> 18));
    u[1] = (char)(0x80u | ((cp >> 12) & 0x3Fu));
    u[2] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
    u[3] = (char)(0x80u | (cp & 0x3Fu));
    buf_put(mk, b, u, 4u);
  }
}

/* A JSON string (RFC 8259, as Python's json reads it: no raw control character, the eight short
 * escapes and \uXXXX with surrogate pairs joined), decoded. */
static int json_string(mk_ctx *mk, mk_json *j, mk_buf *out) {
  out->n = 0;
  buf_put(mk, out, "", 0u);
  if (j->p >= j->end || *j->p != '"') return 0;
  j->p++;
  while (j->p < j->end) {
    unsigned char c = *j->p++;
    if (c == '"') return 1;
    if (c < 0x20u) return 0;
    if (c != '\\') {
      buf_put(mk, out, (const char *)&c, 1u);
      continue;
    }
    if (j->p >= j->end) return 0;
    c = *j->p++;
    switch (c) {
    case '"': buf_putc(mk, out, '"'); break;
    case '\\': buf_putc(mk, out, '\\'); break;
    case '/': buf_putc(mk, out, '/'); break;
    case 'b': buf_putc(mk, out, '\b'); break;
    case 'f': buf_putc(mk, out, '\f'); break;
    case 'n': buf_putc(mk, out, '\n'); break;
    case 'r': buf_putc(mk, out, '\r'); break;
    case 't': buf_putc(mk, out, '\t'); break;
    case 'u': {
      uint32_t cp, lo;
      if (j->end - j->p < 4 || !hex4(j->p, &cp)) return 0;
      j->p += 4;
      if (cp >= 0xD800u && cp <= 0xDBFFu && j->end - j->p >= 6 && j->p[0] == '\\' &&
          j->p[1] == 'u' && hex4(j->p + 2, &lo) && lo >= 0xDC00u && lo <= 0xDFFFu) {
        cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
        j->p += 6;
      }
      put_cp(mk, out, cp);
      break;
    }
    default:
      return 0;
    }
  }
  return 0;
}

/* bcir.make.plan.load_state: a JSON object of target name -> 64 lowercase hex digits, each name
 * once. Anything else -- a value that is no string, a key recorded twice, invalid UTF-8, data after
 * the object -- is refused (exit 2), as the oracle refuses it. */
static mk_map load_state(mk_ctx *mk, const char *path) {
  mk_map state = {0};
  mk_buf key = {0}, value = {0};
  mk_json j;
  unsigned char *data;
  size_t n = 0, cap = 1u << 16;
  FILE *f = fopen(path, "rb");
  if (!f) mk_unusable(path, strerror(errno));
  data = (unsigned char *)mk_alloc(mk, cap);
  for (;;) {
    size_t want, got;
    if (n == cap) {
      unsigned char *grown;
      if (cap >= MK_MAX_STATE) {
        fclose(f);
        mk_unusable(path, "the state is over its bound");
      }
      grown = (unsigned char *)mk_alloc(mk, cap * 2u);
      memcpy(grown, data, n);
      data = grown;
      cap *= 2u;
    }
    want = cap - n;
    got = fread(data + n, 1u, want, f);
    n += got;
    if (got < want) break; /* the end of the file or an error, as file_digest reads: ferror below */
  }
  if (ferror(f)) {
    fclose(f);
    mk_unusable(path, "it cannot be read");
  }
  fclose(f);
  if (!utf8_valid(data, n)) mk_unusable(path, "it is not UTF-8");
  j.p = data;
  j.end = data + n;
  json_ws(&j);
  if (j.p >= j.end || *j.p != '{') goto refused;
  j.p++;
  json_ws(&j);
  if (j.p < j.end && *j.p == '}') {
    j.p++;
  } else {
    for (;;) {
      void **slot;
      char *k, *v;
      size_t i;
      if (!json_string(mk, &j, &key)) goto refused;
      json_ws(&j);
      if (j.p >= j.end || *j.p != ':') goto refused;
      j.p++;
      json_ws(&j);
      if (!json_string(mk, &j, &value)) goto refused; /* no string: no state, whatever follows */
      if (value.n != 64u) goto refused;
      for (i = 0; i < 64u; i++)
        if (!((value.p[i] >= '0' && value.p[i] <= '9') || (value.p[i] >= 'a' && value.p[i] <= 'f')))
          goto refused;
      k = mk_strndup(mk, key.p, key.n);
      v = mk_strndup(mk, value.p, 64u);
      slot = map_slot(mk, &state, k, key.n, 1);
      if (*slot) goto refused; /* recorded twice */
      *slot = v;
      json_ws(&j);
      if (j.p < j.end && *j.p == ',') {
        j.p++;
        json_ws(&j);
        continue;
      }
      if (j.p < j.end && *j.p == '}') {
        j.p++;
        break;
      }
      goto refused;
    }
  }
  json_ws(&j);
  if (j.p != j.end) goto refused;
  return state;
refused:
  mk_unusable(path, "it is not a JSON object of target -> sha256 tag");
  return state;
}

/* ------------------------------------------------------------------ the dry run */

/* str(pathlib.PurePosixPath(p)): repeated slashes collapse (a leading pair stays), `.` parts and a
 * trailing slash go, nothing becomes `.` */
static char *posix_path_str(mk_ctx *mk, const char *p) {
  mk_buf out = {0};
  const char *s = p;
  int any = 0;
  if (p[0] == '/' && p[1] == '/' && p[2] != '/') {
    buf_puts(mk, &out, "//");
    s = p + 2;
  } else if (p[0] == '/') {
    buf_putc(mk, &out, '/');
    while (*s == '/') s++;
  }
  while (*s) {
    const char *e = s;
    while (*e && *e != '/') e++;
    if (e > s && !(e - s == 1 && s[0] == '.')) {
      if (any) buf_putc(mk, &out, '/');
      buf_put(mk, &out, s, (size_t)(e - s));
      any = 1;
    }
    s = e;
    while (*s == '/') s++;
  }
  if (!out.n) buf_putc(mk, &out, '.');
  return out.p;
}

static int dry_run(mk_ctx *mk, const char *data, size_t len, const char *source, mk_map *state,
                   int have_state, size_t workers, const char *workers_text) {
  mk_file bf;
  mk_error err;
  mk_lowered lw;
  mk_strs findings;
  mk_ints order, *waves;
  char **tags;
  size_t i, k, nwaves, runs = 0;
  mk_buf out = {0};
  if (!mk_parse(mk, data, len, &bf, &err)) {
    buf_printf(mk, &out, "bcir-make: %s: not a BCIRfile\nlaws: FAIL (1 finding)\n  MK0: ", source);
    if (err.line) buf_printf(mk, &out, "line %zu: ", err.line);
    buf_puts(mk, &out, err.message.p ? err.message.p : "");
    buf_putc(mk, &out, '\n');
    fwrite(out.p, 1u, out.n, stdout);
    return 1;
  }
  buf_printf(mk, &out, "bcir-make: %s: %zu tools, %zu targets (dry run)\n", source, bf.ntools,
             bf.ntargets);
  mk_lower(mk, &bf, &lw);
  findings = mk_check(mk, &bf, &lw);
  if (findings.n) {
    buf_printf(mk, &out, "laws: FAIL (%zu finding(s))\n", findings.n);
    for (i = 0; i < findings.n; i++) buf_printf(mk, &out, "  %s\n", findings.v[i]);
    fwrite(out.p, 1u, out.n, stdout);
    return 1;
  }
  buf_puts(mk, &out, "laws: PASS (MK0-MK5)\n");
  order = topo_order(mk, &bf);
  tags = mk_tags(mk, &bf, &lw, &order);
  /* decide: run when the recorded generation differs or an output is missing, reuse otherwise */
  {
    const char **action = (const char **)mk_alloc(mk, (bf.ntargets + 1u) * sizeof *action);
    mk_buf *why = (mk_buf *)mk_alloc(mk, (bf.ntargets + 1u) * sizeof *why);
    memset(why, 0, (bf.ntargets + 1u) * sizeof *why);
    for (i = 0; i < bf.ntargets; i++) {
      mk_target *t = bf.targets[i];
      const char *recorded = have_state ? (const char *)map_get(state, t->name) : NULL;
      const char *missing = NULL;
      for (k = 0; k < t->writes.n && !missing; k++)
        if (!is_file_at(mk, t->writes.v[k])) missing = t->writes.v[k];
      if (!recorded || strcmp(recorded, tags[i])) {
        action[i] = "run";
        buf_puts(mk, &why[i], recorded ? "its generation tag changed" : "no recorded generation");
        runs++;
      } else if (missing) {
        action[i] = "run";
        buf_printf(mk, &why[i], "%s is missing", missing);
        runs++;
      } else {
        action[i] = "reuse";
        buf_puts(mk, &why[i], "same generation, outputs present");
      }
    }
    waves = mk_schedule(mk, &bf, &order, workers, &nwaves);
    buf_printf(mk, &out, "plan: %zu to run, %zu to reuse, %s worker(s), %zu wave(s)\n", runs,
               bf.ntargets - runs, workers_text, nwaves);
    for (i = 0; i < nwaves; i++) {
      buf_printf(mk, &out, "wave %zu:", i + 1u);
      for (k = 0; k < waves[i].n; k++) {
        buf_putc(mk, &out, ' ');
        buf_puts(mk, &out, bf.targets[waves[i].v[k]]->name);
      }
      buf_putc(mk, &out, '\n');
    }
    for (i = 0; i < bf.ntargets; i++)
      buf_printf(mk, &out, "%s %s %.16s (%s)\n", action[i], bf.targets[i]->name, tags[i],
                 why[i].p);
  }
  fwrite(out.p, 1u, out.n, stdout);
  return 0;
}

/* ------------------------------------------------------------------ the command line */

static void usage(void) {
  fputs("usage: bcir-make --dry-run [-f BCIRfile] [--root DIR] [--state FILE] [--workers N] "
        "[--check-tools]\n",
        stderr);
}

/* --workers, decimal: the digits as the oracle's int() prints them, and the count it schedules
 * with (a cap past the number of targets schedules like any larger one) */
static int parse_workers(const char *s, size_t *value, char **text, mk_ctx *mk) {
  const char *p = s;
  size_t v = 0;
  int big = 0;
  if (*p == '+') p++;
  if (!*p) return 0;
  while (*p == '0' && p[1]) p++;
  *text = mk_strdup(mk, p);
  for (; *p; p++) {
    if (*p < '0' || *p > '9') return 0;
    if (v > (SIZE_MAX - 9u) / 10u) big = 1;
    else v = v * 10u + (size_t)(*p - '0');
  }
  *value = big ? SIZE_MAX : v;
  return 1;
}

int main(int argc, char **argv) {
  mk_ctx mk;
  const char *file = "BCIRfile", *root = ".", *state_path = NULL, *workers_arg = "2";
  int dry = 0, i, rc;
  size_t workers;
  char *workers_text, resolved[PATH_MAX], *source, *data;
  size_t len = 0;
  mk_map state = {0};
  FILE *f;
  memset(&mk, 0, sizeof mk);
  if (!bcir_host_arena_init(&mk.arena, NULL, 1u << 20)) mk_unusable("out of memory", NULL);
  for (i = 1; i < argc; i++) {
    const char *a = argv[i];
    const char **slot = NULL;
    const char *value = NULL;
    if (!strcmp(a, "--dry-run")) {
      dry = 1;
      continue;
    }
    if (!strcmp(a, "--check-tools")) {
      mk.check_tools = 1;
      continue;
    }
    if (!strcmp(a, "-f") || !strcmp(a, "--file") || !strncmp(a, "--file=", 7u)) slot = &file;
    else if (!strcmp(a, "--root") || !strncmp(a, "--root=", 7u)) slot = &root;
    else if (!strcmp(a, "--state") || !strncmp(a, "--state=", 8u)) slot = &state_path;
    else if (!strcmp(a, "--workers") || !strncmp(a, "--workers=", 10u)) slot = &workers_arg;
    if (!slot) {
      usage();
      return 2;
    }
    value = strchr(a, '=');
    if (value && a[1] == '-') {
      value++;
    } else {
      if (i + 1 >= argc) {
        usage();
        return 2;
      }
      value = argv[++i];
    }
    *slot = value;
  }
  if (!dry) {
    fputs("bcir-make: UNUSABLE: the twin judges and plans (--dry-run); the oracle runs "
          "(python -m bcir.make)\n",
          stderr);
    return 2;
  }
  if (!parse_workers(workers_arg, &workers, &workers_text, &mk)) {
    usage();
    return 2;
  }
  if (workers < 1u) mk_unusable("--workers must be at least 1", NULL);
  source = posix_path_str(&mk, file);
  f = fopen(source, "rb");
  if (!f) mk_unusable(source, strerror(errno));
  data = (char *)mk_alloc(&mk, MK_MAX_BYTES + 2u);
  len = fread(data, 1u, MK_MAX_BYTES + 1u, f); /* one byte past the bound decides (L3) */
  if (ferror(f)) {
    int error = errno ? errno : EIO;
    fclose(f);
    mk_unusable(source, strerror(error));
  }
  fclose(f);
  if (!realpath(posix_path_str(&mk, root), resolved)) {
    fprintf(stderr, "bcir-make: UNUSABLE: %s is no directory\n", root);
    return 2;
  }
  {
    struct stat st;
    if (stat(resolved, &st) != 0 || !S_ISDIR(st.st_mode)) {
      fprintf(stderr, "bcir-make: UNUSABLE: %s is no directory\n", resolved);
      return 2;
    }
  }
  mk.root = !strcmp(resolved, "/") ? "" : resolved;
  if (state_path) state = load_state(&mk, posix_path_str(&mk, state_path));
  rc = dry_run(&mk, data, len, source, &state, state_path != NULL, workers, workers_text);
  fflush(stdout);
  bcir_host_arena_destroy(&mk.arena);
  return rc;
}
