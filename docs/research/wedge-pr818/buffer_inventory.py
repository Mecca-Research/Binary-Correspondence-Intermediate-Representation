"""Enumerate actual C-plan/packing sizes, without executing or timing a model."""
from study import *
from dataclasses import replace


if __name__=='__main__':
    freeze();rows=[]
    for model in MODELS:
        for contract in (14,26):
            r=replace(recipe(model),contract=contract)
            for t in (64,128,256,512,1024):
                for variant in FAMILIES[family(model)]:
                    c=candidate(r,4,t,4,variant,4,4*t)
                    rows.append({'model':model,'contract':contract,**c})
    estimates=[]
    for model in MODELS[:-1]:
        for contract in (14,26):
            ds=[r for r in rows if r['model']==model and r['contract']==contract and r['context']==256]
            cached,frame=ds[:2]
            per_layer=(cached['activations']-frame['activations'])*4/recipe(model).layers
            slack=32*2**20-frame['resident_bound_bytes']
            estimates.append({'model':model,'contract':contract,'context':256,
                'frame_resident_bound_bytes':frame['resident_bound_bytes'],
                'cached_minus_frame_activation_bytes_per_layer':per_layer,
                'cached_scratch_bytes':cached['scratch']*4,'frame_scratch_bytes':frame['scratch']*4,
                '32MiB_slack_bytes':slack,
                'arithmetic_only_cached_layer_upper_estimate':min(recipe(model).layers,int(slack//per_layer)),
                'qualification':'Inventory-derived idea, not an implemented or timed mixed-layer plan. A real plan must prove all mixed scratch/packing lifetimes.'})
    out={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'scope':'Actual native make_plan/workspace queries, B4, four threads, per-shape staging. Controlled bytes, not RSS or minimum-memory proofs.',
         'rows':rows,'mixed_layer_research_estimates':estimates}
    (ROOT/'results/buffer-inventory.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(estimates,indent=2))
