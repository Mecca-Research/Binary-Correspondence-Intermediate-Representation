/* L7 adversarial preprocessor corpus -- the IMPLEMENTATION-DEFINED predefined macro value
 * `__STDC_VERSION__`. This fixture exists to EXERCISE THE FAIRNESS GATE: clang -E and gcc -E themselves
 * disagree on the value (clang reports 202311L for its C23 mode, gcc 202000L for -std=c2x), so it is NOT
 * a fair single-truth differential and the gate must EXCLUDE it with a recorded reason -- never blame
 * cpp.preprocess for matching one and not the other. It is pinned in _PP_PIN_STDC_VERSION in the test so
 * the excluded set stays explicit and minimal. (cpp.py's own static value is 202311L.) */
#define STR(x) #x
#define XSTR(x) STR(x)
stdc_version_value XSTR(__STDC_VERSION__)
#if __STDC_VERSION__ >= 201112L
at_least_c11
#endif
