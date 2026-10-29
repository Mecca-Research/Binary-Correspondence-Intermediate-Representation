/* GATE-SPAN's premise, exhaustively: the tensor unit's span exponential (bcir_tensor_exp, the widest instance, and
 * each instance bcir_tensor_exp_variant admits) against the scalar one (bcir_tensor_expf, exp1) on every one of the
 * 2^32 binary32 bit patterns -- NaNs (every payload), both infinities, both zeros, subnormals and the clamps --
 * compared bit for bit, in spans of 4,096 and in spans of every length from 1 to 80 (the tails a 512-element task
 * of the geometric heads' gate never reaches, and the masked tail). Prints the count of inputs and of mismatches
 * per instance; exits 1 on any mismatch. usage: exp_span_check */
#include "bcir_tensor.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define SPAN 4096
int main(void) {
  static float xs[SPAN],ys[SPAN],want[SPAN]; int variants[3]={1,2,4},v,ran[3]={0,0,0};
  unsigned long long checked=0,bad[4]={0,0,0,0},tails=0; uint64_t base; size_t i,n;
  for (base=0;base<(1ull<<32);base+=SPAN) {
    for (i=0;i<SPAN;i++) { uint32_t b=(uint32_t)(base+i); memcpy(&xs[i],&b,4); want[i]=bcir_tensor_expf(xs[i]); }
    bcir_tensor_exp(SPAN,xs,ys);
    for (i=0;i<SPAN;i++) bad[0]+=memcmp(&ys[i],&want[i],4)!=0;
    for (v=0;v<3;v++) {
      if (!bcir_tensor_exp_variant(variants[v],SPAN,xs,ys)) continue;
      ran[v]=1;
      for (i=0;i<SPAN;i++) bad[1+v]+=memcmp(&ys[i],&want[i],4)!=0;
    }
    checked+=SPAN;
    if ((base&((1ull<<28)-1))==0) {  /* every 2^28: the span lengths 1..80 at this block's start */
      for (n=1;n<=80;n++) {
        bcir_tensor_exp(n,xs,ys);
        for (i=0;i<n;i++) bad[0]+=memcmp(&ys[i],&want[i],4)!=0;
        tails+=n;
      }
    }
  }
  {  /* the special values at every lane of spans 1..80 from offsets 0..15: the masked tail sees each of them */
    static const uint32_t special[]={0x00000000u,0x80000000u,0x7f800000u,0xff800000u,0x7fc00000u,0x7f800001u,
      0xffffffffu,0x7fbfffffu,0x42b17218u,0x42b17217u,0x42b17219u,0xc2cff1b5u,0xc2cff1b4u,0xc2cff1b6u,0x42b20000u,
      0xc2d00000u,0x7f7fffffu,0xff7fffffu,0x00800000u,0x00000001u,0x80000001u,0x3f800000u,0xbf800000u};
    size_t ns=sizeof special/sizeof special[0],off;
    for (i=0;i<SPAN;i++) { uint32_t b=special[i%ns]; memcpy(&xs[i],&b,4); want[i]=bcir_tensor_expf(xs[i]); }
    for (off=0;off<16;off++) for (n=1;n<=80;n++) {
      bcir_tensor_exp(n,xs+off,ys);
      for (i=0;i<n;i++) bad[0]+=memcmp(&ys[i],&want[off+i],4)!=0;
      tails+=n;
    }
  }
  printf("exp_span_check: %llu inputs; widest %llu mismatches (+%llu tail inputs); instance 1 %s %llu, 2 %s %llu, "
         "4 %s %llu\n",checked,bad[0],tails,ran[0] ? "ran" : "absent",bad[1],ran[1] ? "ran" : "absent",bad[2],
         ran[2] ? "ran" : "absent",bad[3]);
  return bad[0]||bad[1]||bad[2]||bad[3] ? 1 : 0;
}
