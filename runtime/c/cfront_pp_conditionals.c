/* L7 adversarial preprocessor corpus -- conditional compilation (C 6.10.1).
 * Deep `#if`/`#elif`/`#else` nesting; `#if` arithmetic with `defined(X)`, `&&`/`||` short-circuit,
 * operator precedence, a ternary, and a wrap-around at the evaluator's integer width; `#ifdef`/`#ifndef`.
 * The short-circuit case `defined(UNDEF) && (1/0)` must NOT evaluate the divide. Single-valued: no
 * impl-defined predefined value is read, so clang and gcc select the identical branches. */
#define ON 1
#define ZERO 0
#define LEVEL 2
#if defined(ON) && ON
top_on
#  if LEVEL == 2
    level_two
#  elif LEVEL == 1
    level_one
#  else
    level_other
#  endif
#endif
#if ZERO || (1 + 2 * 3 == 7)
arith_ok
#endif
#if defined(UNDEF) && (1 / 0)
divide_reached
#else
short_circuit_ok
#endif
#if (1 ? 10 : 20) == 10
ternary_ok
#endif
#ifndef NOT_DEFINED
ifndef_ok
#endif
