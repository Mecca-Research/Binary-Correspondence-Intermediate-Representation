/*===- bcir_cpp.h - the BCIR C preprocessor (L7, the C twin of cfront/cpp.py) ===
 *
 * A C preprocessor for the plug-in compiler: object-, function- and variadic-like #define
 * macros (with # stringize, ## paste, __VA_ARGS__ and C23 __VA_OPT__), #undef, conditional
 * compilation (#if / #ifdef /
 * #ifndef / #elif / #elifdef / #elifndef / #else / #endif with an integer
 * constant-expression evaluator + `defined` + `__has_include` + `__has_attribute`/
 * `__has_builtin`/`__has_c_attribute`), the predefined macros __FILE__ / __LINE__ /
 * __DATE__ / __TIME__ (and the static __STDC__ / __STDC_VERSION__ / __STDC_HOSTED__), the #line
 * directive, the _Pragma operator, #include of project headers, and C23 #embed. It runs before
 * bcir_cfront's lexer, emitting
 * fully-expanded C text -- so a real vendor register-map header (with its REG #defines)
 * ingests through the C compiler.
 *
 * Host tool (libc). #include / #embed resolve files relative to `basedir`. A Python<->C
 * parity test gates it against bcir/frontends/cfront/cpp.py.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_CPP_H
#define BCIR_CPP_H

#include <stddef.h>
#include <stdio.h>

#include "bcir_host_alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Re-entrant hosted preprocessor context. `state` is owned by the context;
 * `allocator` and its context pointer are borrowed until destroy. Independent
 * contexts may run concurrently. A context must be initialized before use.
 */
typedef struct bcir_cpp_context {
  bcir_host_allocator allocator;
  void *state;
} bcir_cpp_context;

/* Zero-initialize and allocate a context. NULL allocator selects libc. */
int bcir_cpp_context_init(bcir_cpp_context *context,
                          const bcir_host_allocator *allocator);

/* Release operation scratch while preserving a reusable initialized context. */
void bcir_cpp_context_reset(bcir_cpp_context *context);

/* Release all owned state. Safe on NULL and safe to call repeatedly. */
void bcir_cpp_context_destroy(bcir_cpp_context *context);

/* The target's character types, which a character constant in `#if` reads by (CF-PPARITH): whether plain `char`
 * is signed -- Clang and GCC read a plain one as a `uintmax_t` where it is not -- and `wchar_t`'s size in bytes (2
 * or 4) and signedness, an `L` one's. A context starts with x86-64 Linux's (a signed `char`, a signed 4-byte
 * `wchar_t`) and keeps the last one set across runs and resets; `bcir_cfront_target_chars` gives a target's. */
void bcir_cpp_context_set_chars(bcir_cpp_context *context, int char_signed, int wchar_size, int wchar_signed);

/* Context-based forms. All input/output pointers are borrowed. */
int bcir_cpp_run_context(bcir_cpp_context *context, const char *src,
                         const char *basedir, char *out, size_t outcap,
                         char *err, size_t errcap);
int bcir_cpp_run_ex_context(bcir_cpp_context *context, const char *src,
                            const char *srcname, const char *const *dirs,
                            int ndirs, const char *const *defines,
                            int ndefines, char *out, size_t outcap,
                            char *err, size_t errcap);

/* Compatibility wrapper over one process-static context. It is intentionally
 * NON-THREAD-SAFE; concurrent callers must use bcir_cpp_run_context instead.
 * Preprocess `src` into `out[0..outcap)` (NUL-terminated). `basedir` is the directory
 * #include/#embed resolve against (NULL/"" -> only <system> headers are no-ops, no file
 * reads). Returns 0 on success; nonzero on error with a message in `err`. */
int bcir_cpp_run(const char *src, const char *basedir, char *out, size_t outcap,
                 char *err, size_t errcap);

/* NON-THREAD-SAFE compatibility wrapper. The compiler-driver entry: `srcname` is the translation unit's name for __FILE__ (NULL/"" ->
 * "<source>"); resolve `#include`/`#embed` against `dirs[0..ndirs)` (the source dir + each -I path,
 * tried in order) and seed `defines[0..ndefines)` predefined macros (each a "name body" string, e.g.
 * "WIDE 1" or "REG_BASE 0x4000"). Macros persist across nested includes. Returns 0 on success;
 * nonzero with a message in `err`. */
int bcir_cpp_run_ex(const char *src, const char *srcname, const char *const *dirs, int ndirs,
                    const char *const *defines, int ndefines,
                    char *out, size_t outcap, char *err, size_t errcap);

/* The preprocessed text in a block grown as it needs, to 64 MiB (CF-PPLIMITS): *out the NUL-terminated text and
 * *outlen its length, released through the context's allocator (`context->allocator`); the legacy forms' through the
 * default host allocator. 0, or nonzero with the reason in `err` and *out NULL. A fixed-capacity run refuses a unit
 * whose text outgrows its buffer (`preprocessed output too large`); these never do below the bound. */
int bcir_cpp_run_ex_alloc_context(bcir_cpp_context *context, const char *src, const char *srcname,
                                  const char *const *dirs, int ndirs, const char *const *defines, int ndefines,
                                  char **out, size_t *outlen, char *err, size_t errcap);
int bcir_cpp_run_ex_alloc(const char *src, const char *srcname, const char *const *dirs, int ndirs,
                          const char *const *defines, int ndefines, char **out, size_t *outlen,
                          char *err, size_t errcap);
int bcir_cpp_run_alloc(const char *src, const char *basedir, char **out, size_t *outlen, char *err, size_t errcap);
/* A whole translation unit read from `fp`, NUL-terminated, in a block `allocator` holds (NULL: libc), to 64 MiB:
 * NULL with *out set, or the reason it is refused -- past the bound, unreadable, or holding a NUL -- with *out NULL.
 * Never a prefix (CF-LIMITS, CF-PPLIMITS). */
const char *bcir_cpp_read_source(FILE *fp, const bcir_host_allocator *allocator, char **out);
/* NON-THREAD-SAFE: `bcir_cpp_context_set_chars` for the compatibility wrappers' process-static context. */
void bcir_cpp_set_chars(int char_signed, int wchar_size, int wchar_signed);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_CPP_H */
