/* enum breadth: enumerators resolve to their integer values at parse time (IDLE=0, RUN=3,
 * HALT=4), so each use lowers to a c.const -- the claim graph carries the folded constants. */
enum mode { IDLE, RUN = 3, HALT };

unsigned int pick(unsigned int x)
{
    unsigned int base = RUN;
    unsigned int top = HALT + IDLE;
    return x + base * top;
}
