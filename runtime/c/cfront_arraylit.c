/* Array compound literals (#arraylit): an anonymous array object `(T[N]){...}` / `(T[]){...}`, used by a
 * direct subscript -- the inline lookup-table idiom `(int[]){...}[i]` (a jump/offset table materialized
 * at the use site). Mirrors the local-array path: a `= {0}` zero baseline + a c.store per element, then an
 * indexed load for the subscript. An inferred `[]` size is taken from the initializer (the max index + 1;
 * gaps zero-fill, §6.7.10); the typed element store (`_cl[i] = v`) converts to any scalar element type
 * (int / char / long). Both rails agree structurally and the emit is Clang-behaviour-equivalent. (A
 * struct-element literal -- per-element aggregate init -- and array-to-pointer decay are follow-ons.) */

/* an inferred-size lookup table indexed at the use site -- the canonical idiom */
int  weekday(unsigned m)  { return (int[]){0,3,2,5,0,3,5,1,4,6,2,4}[m % 12u]; }
/* an explicit array size */
int  sized(unsigned i)    { return (int[4]){10,20,30,40}[i & 3u]; }
/* a char-element table: the element store/load is plain `char` (the load widens to int on return) */
int  charlut(unsigned i)  { return (char[]){'a','b','c','d'}[i & 3u]; }
/* `[i]=` designators mixed with a positional entry: {0,7,9,0,100} (gaps zero-fill) */
int   desig(unsigned i)   { return (int[5]){[4]=100, [1]=7, 9}[i % 5u]; }
/* a float-element table: the element load keeps its float type (a `uint32_t` temp would truncate it) */
float flut(unsigned i)    { return (float[]){1.5f, 2.5f, 3.5f, 4.5f}[i & 3u]; }
/* a wide `long` element -- the typed array store moves the full 8 bytes per element */
long  widelut(unsigned i) { return (long[]){1L<<40, 2L<<40, 3L<<40}[i % 3u]; }
