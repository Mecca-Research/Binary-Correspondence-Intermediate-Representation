/* union breadth: all members overlap at offset 0 (size = the widest member). A register read
 * as a whole word vs. its low half -- the classic overlapping register-view pattern. */
union reg {
    unsigned int word;
    unsigned short half;
    unsigned char byte;
};

unsigned int low(union reg r)
{
    unsigned int w = r.word;
    unsigned int h = r.half;
    return w + h;
}
