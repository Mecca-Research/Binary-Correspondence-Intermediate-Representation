/* NDT-BF16's region width (bcir_tensor.c's bf16_cols), as it first shipped and as counted in whole
 * 32-column blocks after clang's analyzer (18 and 23) found that a call with no rows or no columns
 * divided by zero through the grouped entries: the two agree on every non-empty shape of the grid
 * below, so the partition of every call that computes anything is unchanged, and the new one gives
 * every empty shape a positive multiple of 32 (no regions, no division by zero).
 *   gcc -O2 -std=c11 cols_check.c -o cols_check && ./cols_check */
#include <stdio.h>
#include <stddef.h>
#define BF16_ROWS 32
#define BF16_PANEL (1024u*1024u)
#define BF16_TASKS 16
static size_t bf16_pad(size_t x) { return (x+31)/32*32; }
static size_t old_cols(size_t kt, size_t m, size_t n) {
  size_t blocks=kt ? BF16_PANEL/(kt*32*32*2) : 1,cols,tr=(m+BF16_ROWS-1)/BF16_ROWS,want;
  cols=(blocks ? blocks : 1)*32;
  want=tr<BF16_TASKS ? (BF16_TASKS+tr-1)/tr : 1;
  if ((n+cols-1)/cols<want) { size_t w=bf16_pad((n+want-1)/want); cols=w<cols ? w : cols; }
  return cols;
}
static size_t new_cols(size_t kt, size_t m, size_t n) {
  size_t panel=kt*32*32*2,width=panel ? BF16_PANEL/panel : 1,tr=(m+BF16_ROWS-1)/BF16_ROWS,nb=(n+31)/32;
  if (!width) width=1;
  if (tr && tr<BF16_TASKS) {
    size_t want=(BF16_TASKS+tr-1)/tr,per=(nb+want-1)/want;
    if ((nb+width-1)/width<want && per && per<width) width=per;
  }
  return width*32;
}
int main(void) {
  static const size_t kts[]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,24,31,32,33,40,48,63,64,
      65,96,100,127,128,129,200,255,256,257,300,341,342,400,511,512,513,600,1000,1024,2048,5000,100000};
  size_t i,m,n,checked=0,differ=0,empty=0,bad=0;
  for (i=0;i<sizeof kts/sizeof kts[0];i++) {
    for (m=1;m<=600;m++) for (n=1;n<=4200;n++) {
      size_t a=old_cols(kts[i],m,n),b=new_cols(kts[i],m,n); checked++;
      if (a!=b) { if (differ<5) printf("differ kt=%zu m=%zu n=%zu old=%zu new=%zu\n",kts[i],m,n,a,b); differ++; }
    }
    for (m=0;m<=600;m++) { size_t b=new_cols(kts[i],m,0); empty++; bad+=b<32||b%32; }
    for (n=0;n<=4200;n++) { size_t b=new_cols(kts[i],0,n); empty++; bad+=b<32||b%32; }
  }
  printf("checked %zu non-empty shapes: %zu differ; %zu empty shapes: %zu widths not a positive multiple of 32\n",
      checked,differ,empty,bad);
  return differ||bad;
}
