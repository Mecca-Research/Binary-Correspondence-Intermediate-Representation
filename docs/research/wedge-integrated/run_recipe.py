"""Run one reusable, full-global recipe in a new result directory."""
import argparse,json,hashlib
from pathlib import Path
import train_integrated as study
from policy import BudgetPolicy,ROOT

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Native-C full-global recipe with a controlled buffer budget.')
    p.add_argument('--case',choices=list(study.CASES),default='ordered_bucket_global')
    p.add_argument('--seed',type=int,default=20261021)
    p.add_argument('--budget-mib',type=int)
    p.add_argument('--output',type=Path,default=Path('runs/single-recipe'))
    p.add_argument('--describe',action='store_true',help='Show admitted plans and exit without training.')
    p.add_argument('--no-calibration',action='store_true',help='Use deterministic feasible-plan fallback without timing lookup.')
    args=p.parse_args();case=dict(study.CASES[args.case])
    if args.budget_mib is not None:case['budget_mib']=args.budget_mib
    cal=json.loads((ROOT/'results/calibration.json').read_text())
    cpu=next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name'))
    lookup=cal['lookup_ms'] if cpu==cal['host_cpu'] and not args.no_calibration else {}
    policy=BudgetPolicy(study.base_recipe(case['family']),case['family'],study.shapes(case['task']),case['budget_mib']*2**20,4,lookup,micro_mode='full')
    manifest={'case':args.case,'seed':args.seed,'case_configuration':case,'policy':policy.to_dict(),
        'calibration_host_matches':cpu==cal['host_cpu'],'calibration_used':bool(lookup),
        'calibration_sha256':hashlib.sha256((ROOT/'results/calibration.json').read_bytes()).hexdigest(),
        'parent_protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),
        'scope':'A new user run of the retained recipe. Budget controls the declared arrays/control/staging storage, not RSS.'}
    if args.describe:print(json.dumps(manifest,indent=2));raise SystemExit
    output=args.output.resolve();(output/'results').mkdir(parents=True,exist_ok=True);(output/'checkpoints').mkdir(exist_ok=True)
    (output/'recipe.json').write_text(json.dumps(manifest,indent=2))
    data,_=study.bytes_data();study.CASES={args.case:case};study.ROOT=output
    study.run(args.case,args.seed,data,lookup)
    print(output/'results'/f'train-{args.case}-{args.seed}.json')
