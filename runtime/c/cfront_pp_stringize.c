/* L7 adversarial preprocessor corpus -- the stringize operator `#` (C 6.10.4.2) and the classic
 * two-level `XSTR`. `STR(x)` stringizes its RAW argument; `XSTR(x)` first macro-expands its argument
 * (argument prescan) and then stringizes -- so `XSTR(MEANING)` spells the VALUE, not the name. Runs of
 * whitespace in an argument collapse to one space; leading/trailing space is dropped; a `\` or `"` that
 * is part of a string/char literal token is escaped, but a bare backslash is emitted verbatim.
 * Single-valued (no impl-defined macro, no __LINE__): clang and gcc agree exactly. */
#define STR(x) #x
#define XSTR(x) STR(x)
#define MEANING 42
STR(a    b   c)
STR(plain)
XSTR(MEANING)
STR(a\b)
STR(C:\tmp\x)
STR("q")
STR("a\nb")
