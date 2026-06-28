/* L7 adversarial preprocessor corpus -- line continuations and comments in tricky positions
 * (translation phases 2 and 3, C 6.10). A backslash-continued #define is one logical line; a block
 * comment splitting a macro body or sitting between an argument's tokens becomes a single space (so the
 * tokens never glue); a line comment ends the directive. Single-valued: clang and gcc agree. */
#define WIDE(a, b) \
    ((a) << 4 | \
     (b))
#define GLUE(a, b) a/* joiner */b
#define HEAD 1 // trailing line comment is dropped
WIDE(0xA, 0x3)
GLUE(con, cat)
HEAD
#define SPLIT_VALUE/* comment here */55
SPLIT_VALUE
