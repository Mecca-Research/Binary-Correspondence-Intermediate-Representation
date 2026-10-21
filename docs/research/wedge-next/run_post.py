"""All heavy validation, compilation and measurements remain serialized."""
from pathlib import Path
import subprocess,sys,json,os
root=Path(__file__).resolve().parent;env=os.environ|{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
jobs=[]
for mode in ('execution','architecture'):
    for th in (1,4):
        cmd=(["taskset","-c","0"] if th==1 else [])+[sys.executable,'benchmark_next.py','--mode',mode,'--threads',str(th)]
        jobs.append((f'benchmark-{mode}-{th}',cmd))
jobs += [('profile-1',['taskset','-c','0',sys.executable,'profile_next.py','--threads','1']),
         ('profile-4',[sys.executable,'profile_next.py','--threads','4']),
         ('precision',[sys.executable,'precision_next.py']),
         ('budget',[sys.executable,'budget_next.py']),('ham',[sys.executable,'ham_next.py']),
         ('length-stress',[sys.executable,'assess_next.py']),('svd',[sys.executable,'svd_next.py'])]
for name,cmd in jobs:
    print(json.dumps({'job':name,'state':'start'}),flush=True)
    with (root/f'{name}.log').open('w') as log:
        result=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=1200)
    if result.returncode:raise RuntimeError(f'{name}: exit {result.returncode}; see log')
    print(json.dumps({'job':name,'state':'pass'}),flush=True)
