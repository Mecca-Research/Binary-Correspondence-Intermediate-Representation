/* typedef breadth: a scalar alias and an alias-of-an-alias resolve to the underlying type,
 * so the claim graph is identical to spelling `unsigned int` out everywhere. */
typedef unsigned int u32;
typedef u32 word;

u32 scale(word x, u32 k)
{
    u32 acc = x * k;
    word bump = acc + k;
    return bump;
}
