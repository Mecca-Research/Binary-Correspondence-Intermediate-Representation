"""Instrument master-thread kernel calls; no nested providers used."""
from candidates import *
from run_experiments import bytes_data,lm_batches
from build_next import compile_source

def wrap(source,name,cat):
    markers=[f'static void {name}(',f'void {name}(']
    start=next(source.index(m) for m in markers if m in source);a=source.index('{',start);depth=1;b=a+1
    while depth:
        depth+=int(source[b]=='{')-int(source[b]=='}');b+=1
    body=source[a+1:b-1]
    return source[:a+1]+f'double profile_start=profile_now();\n'+body+f'\nprofile_ns[{cat}]+=profile_now()-profile_start;profile_count[{cat}]++;\n'+source[b-1:]

if __name__=='__main__':
    import argparse
    pa=argparse.ArgumentParser();pa.add_argument('--threads',type=int,default=1);args=pa.parse_args()
    source=(ROOT/'build/fast.c').read_text()
    header='''#define _POSIX_C_SOURCE 200809L
#include <time.h>
static double profile_ns[5]={0};static unsigned long long profile_count[5]={0};
static double profile_now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec*1e9+t.tv_nsec;}
void wedge_profile_reset(void){unsigned i;for(i=0;i<5;i++){profile_ns[i]=0;profile_count[i]=0;}}
double wedge_profile_get(unsigned i){return i<5?profile_ns[i]:0;}
unsigned long long wedge_profile_calls(unsigned i){return i<5?profile_count[i]:0;}
'''
    source=header+source
    for name,cat in [('linear',0),('linears',0),('linear_backward',1),('linears_backward',1),('gp_wedge_forward',2),('gp_wedge_backward',3),('silu_mul',4),('swiglu_backward',4),('rms_backward',4)]:source=wrap(source,name,cat)
    compile_source(source,'profile')
    data,_=bytes_data();x,y=lm_batches(data['validation'],33,1,1024,123);rows=[]
    for threads in (args.threads,):
        r=Recipe(kind='global',ff=257,batch=1,context=1024)
        with Native(r,seed=100,threads=threads,library_name='profile') as n:
            n.train(x[:3],y[:3],[.0001]*3);n.lib.wedge_profile_reset();n.lib.wedge_profile_get.restype=ct.c_double;n.lib.wedge_profile_calls.restype=ct.c_ulonglong
            seconds,_=n.train(x[3:],y[3:],[.0001]*30)
            cats={name:{'ms_per_update':n.lib.wedge_profile_get(i)/30e6,'calls_per_update':n.lib.wedge_profile_calls(i)/30}
                  for i,name in enumerate(['forward_matrices','backward_matrices','geometry_forward','geometry_backward','activation_and_norm_backward'])}
            geometry=cats['geometry_forward']['ms_per_update']+cats['geometry_backward']['ms_per_update']
            total=seconds*1000/30
            row={'threads':threads,'context':1024,'batch':1,'ms_per_update':total,'categories':cats,
                'unassigned_ms_per_update':total-sum(v['ms_per_update'] for v in cats.values()),
                'geometry_fraction':geometry/total,'geometry_infinite_speed_max_total_speedup':1/(1-geometry/total),
                'state_sha256':n.digest(),'linear_matmul_flops_per_update':2*3*(3*(128*22+128*231+2*128*128+3*128*257)+260*128)*1024,
                'note':'Serial kernel boundaries; instrumentation includes geometry scans. Amdahl bound holds other work fixed. FLOPs omit scalar/normalization/optimizer work.'}
            with Native(r,seed=100,threads=threads,library_name='fast') as base:
                base.train(x,y,[.0001]*33);assert base.digest()==n.digest()
            rows.append(row);print(json.dumps(row),flush=True)
    (ROOT/'results'/f'profile-{args.threads}thread.json').write_text(json.dumps(rows,indent=2))
