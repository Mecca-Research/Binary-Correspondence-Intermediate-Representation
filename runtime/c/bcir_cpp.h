/*===- bcir_cpp.h - the BCIR C preprocessor (L7, the C twin of cfront/cpp.py) ===
 *
 * A C preprocessor for the plug-in compiler: object- and function-like #define macros
 * (with # stringize and ## paste), #undef, conditional compilation (#if / #ifdef /
 * #ifndef / #elif / #elifdef / #elifndef / #else / #endif with an integer
 * constant-expression evaluator + `defined`), #include of project headers, and C23
 * #embed. It runs before bcir_cfront's lexer, emitting fully-expanded C text -- so a real
 * vendor register-map header (with its REG #defines) ingests through the C compiler.
 *
 * Host tool (libc). #include / #embed resolve files relative to `basedir`. A Python<->C
 * parity test gates it against bcir/frontends/cfront/cpp.py.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_CPP_H
#define BCIR_CPP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Preprocess `src` into `out[0..outcap)` (NUL-terminated). `basedir` is the directory
 * #include/#embed resolve against (NULL/"" -> only <system> headers are no-ops, no file
 * reads). Returns 0 on success; nonzero on error with a message in `err`. */
int bcir_cpp_run(const char *src, const char *basedir, char *out, size_t outcap,
                 char *err, size_t errcap);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_CPP_H */
