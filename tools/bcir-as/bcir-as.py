#!/usr/bin/env python3
import argparse, re
from pathlib import Path

OPMAP={"LOAD_UX":(1,1),"ADD_UX":(3,1),"STORE_U":(2,0),"ATOMIC_ADD":(32,4)}
VERIFY_BITS={"none":0,"bounds":1,"exact":2}
HAZARD_MODE={"none":0,"unique":1,"atomic":2,"barriered":3}
DOMAIN_BITS={"ram":0,"vram":1,"nvm":2,"mmio":3,"cxl":4,"hbm":5}
COST_BITS={"none":0,"bandwidth":1,"compute":2,"latency":3}

def parse(src:str):
    module = re.search(r'MODULE\s+(\w+)', src).group(1)
    rid_map={m.group(1):int(m.group(2)) for m in re.finditer(r'RECORD\s+(\w+)\s*:\s*RES\s*\{[^}]*rid\s*=\s*(\d+)',src,re.S)}
    claims=[]; phase=0; lines=src.splitlines(); i=0
    while i<len(lines):
        ln=lines[i].strip()
        pm=re.match(r'PHASE\s+(\d+)\s*\{',ln)
        if pm: phase=int(pm.group(1))
        cm=re.match(r'CLAIM\s+(\d+)\s*:\s*(\w+)\((.*?)\)\s*::\s*\{',ln)
        if cm:
            cid=int(cm.group(1)); op=cm.group(2)
            if op not in OPMAP: raise ValueError(f"unsupported op {op}")
            args={k:v for k,v in re.findall(r'(\w+)\s*=\s*([\w\d]+)',cm.group(3))}
            block=[]; i+=1
            while i<len(lines) and '}' not in lines[i]: block.append(lines[i]); i+=1
            c='\n'.join(block)
            req=["rd","wr","hazard","domain","verify","cost"]
            for k in req:
                if not re.search(rf'\b{k}\s*=',c): raise ValueError(f"missing contract field {k} for claim {cid}")
            rd=[int(x) for x in re.search(r'rd\s*=\s*\[([^\]]*)\]',c).group(1).replace(' ','').split(',') if x]
            wr=[int(x) for x in re.search(r'wr\s*=\s*\[([^\]]*)\]',c).group(1).replace(' ','').split(',') if x]
            hazard=re.search(r'hazard\s*=\s*(\w+)',c).group(1)
            domain=re.search(r'domain\s*=\s*(\w+)',c).group(1)
            verify=re.search(r'verify\s*=\s*(\w+)',c).group(1)
            cost=re.search(r'cost\s*=\s*(\w+)',c).group(1)
            claims.append(dict(id=cid,phase=phase,op=op,args=args,rd=rd,wr=wr,hazard=hazard,domain=domain,verify=verify,cost=cost))
        i+=1
    return module,rid_map,claims

def control(opcode,lane,phase,epoch=0,flags=0,stride=1):
    return opcode | (lane<<8) | (phase<<16) | (epoch<<28) | (flags<<40) | (stride<<48)

def encode_hazard(c):
    mode=HAZARD_MODE.get(c['hazard'],0)
    dom=DOMAIN_BITS.get(c['domain'],0)
    verify=VERIFY_BITS.get(c['verify'],0)
    cost=COST_BITS.get(c['cost'],0)
    return mode | (dom<<4) | (verify<<8) | (cost<<16)

def emit(module,rids,claims):
    out=['source_filename = "bcir_examples_phase4_generated.ll"','target triple = "unknown-unknown-unknown"','target datalayout = ""','',
    '%bcir.claim = type { i64, [4 x i32], [4 x i32], i64, [2 x i64] }','%bcir.res = type { i32, i32, ptr, i64, i64, i64, i64 }','%bcir.batch = type { i32, i32, i32, i32, i32, i32, i64, i64, i64 }','%bcir.phase.range = type { i32, i32, i64, i64 }','']
    crows=[]
    for c in claims:
        opcode,lane=OPMAP[c['op']]
        imm0=int(c['args'].get('off',0)); imm1=int(c['args'].get('count',8))
        rd4=(c['rd']+[0,0,0,0])[:4]; wr4=(c['wr']+[0,0,0,0])[:4]
        ctrl=control(opcode,lane,c['phase']); hz=encode_hazard(c)
        crows.append(f'  %bcir.claim {{ i64 {ctrl}, [4 x i32] [i32 {rd4[0]}, i32 {rd4[1]}, i32 {rd4[2]}, i32 {rd4[3]}], [4 x i32] [i32 {wr4[0]}, i32 {wr4[1]}, i32 {wr4[2]}, i32 {wr4[3]}], i64 {hz}, [2 x i64] [i64 {imm0}, i64 {imm1}] }}')
    out += [f'@bcir.generated.claims = global [{len(crows)} x %bcir.claim] [', ',\n'.join(crows), '], align 64','']
    rrows=[f'  %bcir.res {{ i32 {rid}, i32 {DOMAIN_BITS.get("ram",0)}, ptr null, i64 4096, i64 0, i64 0, i64 0 }}' for _,rid in sorted(rids.items(), key=lambda kv:kv[1])]
    out += [f'@bcir.generated.resources = global [{len(rrows)} x %bcir.res] [', ',\n'.join(rrows), '], align 64','']

    # dynamic batches / phase ranges
    batch_rows=[]; phase_ranges=[]
    i=0; bidx=0
    while i<len(claims):
        j=i+1
        op,lane=OPMAP[claims[i]['op']]; phase=claims[i]['phase']
        while j<len(claims) and claims[j]['phase']==phase and OPMAP[claims[j]['op']]==(op,lane): j+=1
        batch_rows.append(f'  %bcir.batch {{ i32 0, i32 {phase}, i32 {lane}, i32 {op}, i32 0, i32 0, i64 {i}, i64 {j-i}, i64 0 }}')
        bidx+=1; i=j
    # phase ranges from batches
    if batch_rows:
        bs=[]
        for b,txt in enumerate(batch_rows):
            m=re.search(r'i32 (\d+), i32 (\d+), i32',txt); bs.append(int(m.group(2)))
        pstart=0
        while pstart<len(bs):
            pend=pstart+1
            while pend<len(bs) and bs[pend]==bs[pstart]: pend+=1
            phase_ranges.append(f'  %bcir.phase.range {{ i32 0, i32 {bs[pstart]}, i64 {pstart}, i64 {pend} }}')
            pstart=pend
    out += [f'@bcir.generated.batches = global [{len(batch_rows)} x %bcir.batch] [', ',\n'.join(batch_rows), '], align 64','']
    out += [f'@bcir.generated.phase_ranges = global [{len(phase_ranges)} x %bcir.phase.range] [', ',\n'.join(phase_ranges), '], align 64','']
    out += ['!bcir.generated.module = !{!900}','!900 = !{!"module", !"'+module+'", !"claim_layout", !"BCIR_ClaimV2", !"schedule", !"global_indexed"}']
    return '\n'.join(out)+'\n'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input'); ap.add_argument('-o','--output',required=True)
    a=ap.parse_args(); m,r,c=parse(Path(a.input).read_text()); Path(a.output).write_text(emit(m,r,c))

if __name__=='__main__': main()
