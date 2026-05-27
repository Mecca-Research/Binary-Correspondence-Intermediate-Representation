#!/usr/bin/env python3
import argparse, re
from pathlib import Path

def parse(src: str):
    module = re.search(r'MODULE\s+(\w+)', src).group(1)
    rid_map = {m.group(1): int(m.group(2)) for m in re.finditer(r'RECORD\s+(\w+)\s*:\s*RES\s*\{[^}]*rid\s*=\s*(\d+)', src, re.S)}
    claims = []
    phase = 0
    lines = src.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        m = re.match(r'PHASE\s+(\d+)\s*\{', ln)
        if m:
            phase = int(m.group(1))
        cm = re.match(r'CLAIM\s+(\d+)\s*:\s*(\w+)\((.*?)\)\s*::\s*\{', ln)
        if cm:
            op = cm.group(2)
            args = {k:v for k,v in re.findall(r'(\w+)\s*=\s*([\w\d]+)', cm.group(3))}
            block=[]
            i += 1
            while i < len(lines) and '}' not in lines[i]:
                block.append(lines[i])
                i += 1
            cbody='\n'.join(block)
            rdm = re.search(r'rd\s*=\s*\[([^\]]*)\]', cbody)
            wrm = re.search(r'wr\s*=\s*\[([^\]]*)\]', cbody)
            rd = [int(x) for x in (rdm.group(1).replace(' ','').split(',') if rdm else []) if x]
            wr = [int(x) for x in (wrm.group(1).replace(' ','').split(',') if wrm else []) if x]
            claims.append((phase,op,args,rd,wr))
        i += 1
    return module,rid_map,claims

def op_lane(op):
    if op == 'LOAD_UX': return 1,1
    if op == 'ADD_UX': return 3,1
    return 0,0

def control(opcode,lane,phase,epoch=0,flags=0,stride=1):
    return opcode | (lane<<8) | (phase<<16) | (epoch<<28) | (flags<<40) | (stride<<48)

def emit(module,rids,claims):
    out=[]
    out += ['source_filename = "bcir_examples_phase4_generated.ll"','target triple = "unknown-unknown-unknown"','target datalayout = ""','']
    out += ['%bcir.claim = type { i64, [4 x i32], [4 x i32], i64, [2 x i64] }','%bcir.res = type { i32, i32, ptr, i64, i64, i64, i64 }','%bcir.batch = type { i32, i32, i32, i32, i32, i32, i64, i64, i64 }','%bcir.phase.range = type { i32, i32, i64, i64 }','']
    crows=[]
    for ph,op,args,rd,wr in claims:
      opcode,lane=op_lane(op)
      imm0=int(args.get('off',0)); imm1=int(args.get('count',8))
      rd4=rd+[0]*(4-len(rd)); wr4=wr+[0]*(4-len(wr))
      c=control(opcode,lane,ph)
      crows.append(f'  %bcir.claim {{ i64 {c}, [4 x i32] [i32 {rd4[0]}, i32 {rd4[1]}, i32 {rd4[2]}, i32 {rd4[3]}], [4 x i32] [i32 {wr4[0]}, i32 {wr4[1]}, i32 {wr4[2]}, i32 {wr4[3]}], i64 0, [2 x i64] [i64 {imm0}, i64 {imm1}] }}')
    out += [f'@bcir.generated.claims = global [{len(crows)} x %bcir.claim] [', ',\n'.join(crows), '], align 64','']
    rrows=[]
    for name,rid in sorted(rids.items(), key=lambda kv: kv[1]):
      rrows.append(f'  %bcir.res {{ i32 {rid}, i32 0, ptr null, i64 4096, i64 0, i64 0, i64 0 }}')
    out += [f'@bcir.generated.resources = global [{len(rrows)} x %bcir.res] [', ',\n'.join(rrows), '], align 64','']
    out += ['@bcir.generated.batches = global [2 x %bcir.batch] [','  %bcir.batch { i32 0, i32 0, i32 1, i32 1, i32 0, i32 0, i64 0, i64 1, i64 0 },','  %bcir.batch { i32 0, i32 1, i32 1, i32 3, i32 0, i32 0, i64 1, i64 1, i64 0 }','], align 64','']
    out += ['@bcir.generated.phase_ranges = global [2 x %bcir.phase.range] [','  %bcir.phase.range { i32 0, i32 0, i64 0, i64 1 },','  %bcir.phase.range { i32 0, i32 1, i64 1, i64 2 }','], align 64','']
    out += ['!bcir.generated.module = !{!900}','!900 = !{!"module", !"'+module+'", !"claim_layout", !"BCIR_ClaimV2", !"schedule", !"global_indexed"}']
    return '\n'.join(out)+'\n'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input'); ap.add_argument('-o','--output',required=True)
    a=ap.parse_args(); src=Path(a.input).read_text(); m,r,c=parse(src); Path(a.output).write_text(emit(m,r,c))

if __name__=='__main__': main()
