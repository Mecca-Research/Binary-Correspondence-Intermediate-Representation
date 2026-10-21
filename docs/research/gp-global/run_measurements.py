"""Serialize all model/probe/build measurements; never overlap heavy gates."""
from pathlib import Path
import subprocess,sys,json,datetime
root=Path(__file__).resolve().parent
jobs=[('baselines',[sys.executable,'evaluate_baselines.py']),
      ('scaling-1thread',['taskset','-c','0',sys.executable,'benchmark.py','--threads','1','--reps','6']),
      ('scaling-4thread',[sys.executable,'benchmark.py','--threads','4','--reps','6']),
      ('kernel-1thread',['taskset','-c','0',sys.executable,'kernel_benchmark.py','--threads','1']),
      ('streaming',[sys.executable,'streaming_probe.py']),
      ('subspace',[sys.executable,'subspace_probe.py']),
      ('common-factor',[sys.executable,'common_factor_check.py']),
      ('summary',[sys.executable,'summarize_extended.py']),
      ('report',[sys.executable,'write_report_extended.py'])]
for name,cmd in jobs:
    print(json.dumps({'job':name,'state':'started','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}),flush=True)
    with (root/f'{name}.log').open('w') as log:
        result=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    if result.returncode:raise RuntimeError(f'{name}: exit {result.returncode}; see {name}.log')
    print(json.dumps({'job':name,'state':'passed','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}),flush=True)
