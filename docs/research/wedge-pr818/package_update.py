"""Manifest and reviewable ZIP of completed standalone research, never the repo."""
from pathlib import Path
import json,hashlib,zipfile

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'BCIR_PR818_Geometric_Head_2026-10-18.zip'


if __name__=='__main__':
    audit=json.loads((ROOT/'results/before-after-audit.json').read_text())
    assert audit['all_training_checks_pass'] and audit['all_regression_rows_equal'] and audit['evaluated_runs']==51
    assert len(list((ROOT/'bf16/results').glob('train-*.json')))==15
    assert len(list((ROOT/'bf16/results').glob('eval-*.json')))==15
    assert (ROOT/'BCIR_PR818_Geometric_Head_Report_2026-10-18.md').exists()
    skip={'__pycache__','.git','.venv'}
    names={'MANIFEST_SHA256.json','pre-evaluation-audit.log'}
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and not skip.intersection(p.relative_to(ROOT).parts) and p.name not in names and p.suffix!='.pyc')
    manifest={'format':1,'scope':'Completed standalone PR818 head study, excluding transient caches and the externally synced git repository.',
              'files':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}}
    mp=ROOT/'MANIFEST_SHA256.json';mp.write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(OUT,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
        for p in files+[mp]:z.write(p,arcname=str(Path(ROOT.name)/p.relative_to(ROOT)))
    with zipfile.ZipFile(OUT) as z:
        saved=json.loads(z.read('wedge-pr818/MANIFEST_SHA256.json'))
        assert len(z.namelist())==len(saved['files'])+1
        for name,entry in saved['files'].items():
            # ZipFile.read validates CRC while the manifest check binds the bytes.
            content=z.read('wedge-pr818/'+name)
            assert len(content)==entry['bytes'] and hashlib.sha256(content).hexdigest()==entry['sha256'],name
    sha=hashlib.sha256(OUT.read_bytes()).hexdigest()
    OUT.with_suffix('.zip.sha256').write_text(f'{sha}  {OUT.name}\n')
    print(json.dumps({'archive':str(OUT),'bytes':OUT.stat().st_size,'sha256':sha,'manifest_files':len(files),'selected_learning_checkpoints':66},indent=2))
