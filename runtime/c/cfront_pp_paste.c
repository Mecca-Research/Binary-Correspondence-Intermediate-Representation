/* L7 adversarial preprocessor corpus -- the token-paste operator `##` (C 6.10.4.3).
 * Pasting builds a new single token from the two operands; the result is then RESCANNED, so a paste
 * that forms a macro name re-expands. `##` pastes ANY two adjacent replacement-list tokens, NOT only
 * parameter-adjacent ones -- so an OBJECT-macro body `a##c` / `1##2` / `A##B` pastes too (the common case
 * a token-based expander forgets, leaving a literal `##`). An empty argument is a placemarker: `a ## b`
 * with an empty side yields just the non-empty side and NO literal `##`. Building an identifier and a
 * number are checked. Single-valued: clang and gcc agree exactly. */
#define CAT(a, b) a ## b
#define MKID(a, b) a ## b
#define FOO 99
#define PREFIX_value 7
CAT(1, 2)
CAT(0x, FF)
MKID(F, OO)
MKID(PREFIX_, value)
[CAT(, x)] [CAT(x, )] [CAT(, )]
/* object-macro `##` paste: the body's `##` glues its neighbours (no parameters involved). */
#define OBJ_ID a ## c
#define OBJ_NUM 1 ## 2
#define AB xyz
#define OBJ_RESCAN A ## B
#define M2(p, q) p q
#define Y_OBJ M2(a ## c, z)
OBJ_ID
OBJ_NUM
OBJ_RESCAN
Y_OBJ
