"""Preserve source trials while combining the two frozen parameter domains."""
from pathlib import Path
import argparse,json,hashlib
from run_experiments import dump

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--test',type=Path,required=True);ap.add_argument('--robustness',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise ValueError('Use a fresh combined output folder')
    rows=[];inputs={};reference=None
    for expected,folder in [('test',a.test),('robustness',a.robustness)]:
        folder=folder.resolve();manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        complete=json.loads((folder/'execution_complete.json').read_text(encoding='utf-8'))
        records=json.loads((folder/'trials.json').read_text(encoding='utf-8'))
        if manifest['split']!=expected or len(records)!=manifest['expected_trials'] or complete['trials']!=len(records) or not complete['source_unchanged']:
            raise ValueError('Confirmation output incomplete')
        if reference is None:reference=manifest
        elif any(manifest[k]!=reference[k] for k in ('code_sha256','settings','methods','scenarios','benchmark_id','physics_timestep')):
            raise ValueError('Parameter domains do not share the same frozen methods')
        inputs[str(folder)]={name:sha(folder/name) for name in ('manifest.json','trials.json','execution_complete.json')}
        for row in records:
            item=dict(row);item['trace_file']=str((folder/row['trace_file']).resolve());rows.append(item)
    keys={(r['split'],r['scenario'],r['method'],r['seed']) for r in rows}
    if len(keys)!=len(rows):raise ValueError('Duplicate records')
    a.out.mkdir(parents=True);dump(a.out/'trials.json',rows);dump(a.out/'combined_manifest.json',dict(input_sha256=inputs,trials=len(rows),code_sha256=reference['code_sha256'],benchmark_id=reference['benchmark_id']))
    print(json.dumps(dict(trials=len(rows),out=str(a.out.resolve())),ensure_ascii=False))
if __name__=='__main__':main()
