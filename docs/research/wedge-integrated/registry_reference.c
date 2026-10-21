/* Task-specific reference for the declared synthetic key/value grammar.
 * This is an exact parser/lookup control, not a trained neural architecture.
 * BOS=256, record=(258,key32..47,value0..15), final query=(257,key).
 * Repeated keys, if supplied, use the last complete assignment.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef struct registry_state {uint32_t values[16];uint16_t seen;} registry_state;
size_t registry_state_bytes(void){return sizeof(registry_state);}

int registry_predict(const uint32_t *x,size_t batch,size_t time,uint32_t *out){
  size_t b,i;int failed=0;
  if(!x||!out||time<5||!batch||batch>SIZE_MAX/time)return 1;
  for(b=0;b<batch;b++){
    const uint32_t *row=x+b*time;registry_state s;uint32_t q;int valid=1;
    memset(&s,0,sizeof(s));out[b]=UINT32_MAX;
    if(row[0]!=256||row[time-2]!=257||row[time-1]<32||row[time-1]>=48){failed=1;continue;}
    for(i=1;i+2<time-2;i++)if(row[i]==258){
      uint32_t key=row[i+1],value=row[i+2];
      if(key<32||key>=48||value>=16){valid=0;break;}
      s.values[key-32]=value;s.seen|=(uint16_t)(1u<<(key-32));i+=2;
    }
    q=row[time-1]-32;
    if(valid&&(s.seen&(uint16_t)(1u<<q)))out[b]=s.values[q];else failed=1;
  }
  return failed;
}
