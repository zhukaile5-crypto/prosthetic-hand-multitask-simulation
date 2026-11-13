"""Deterministic, common-budget candidate choice; no confirmation data read."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parent
PRIMARY=('fixed','ratio','independent','unified')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prefix',required=True);ap.add_argument('--apply',action='store_true');args=ap.parse_args()
    candidates=[];inputs={}
    for i in range(3):
        folder=ROOT/'output'/f'{args.prefix}_{i}'
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        complete=json.loads((folder/'execution_complete.json').read_text(encoding='utf-8'))
        rows=json.loads((folder/'trials.json').read_text(encoding='utf-8'))
        if manifest['split']!='development' or len(rows)!=120 or complete['trials']!=120 or not complete['source_unchanged']:
            raise ValueError('Each candidate must have the same completed 120 development trials')
        inputs[str(folder/'trials.json')]=sha(folder/'trials.json')
        if i==0: common={k:manifest[k] for k in ('seeds','methods','scenarios','code_sha256')}
        elif any(manifest[k]!=v for k,v in common.items()):raise ValueError('Candidate inputs differ')
        for method in PRIMARY:
            a=[x for x in rows if x['method']==method]
            if len(a)!=30 or not all(x['complete_evidence'] for x in a):raise ValueError('Incomplete development evidence')
            eligible=[x for x in a if x['static_window']['eligible']]
            release=[x for x in a if x['release_required']]
            candidates.append(dict(method=method,index=i,gains=manifest['settings']['controllers'][method],
                joint_pass=sum(x['task_success'] and x['force_safety']['force_protection_pass'] for x in a),
                task_pass=sum(x['task_success'] for x in a),force_pass=sum(x['force_safety']['force_protection_pass'] for x in a),
                release_pass=sum(x['release_success'] for x in release),release_trials=len(release),
                eligible=len(eligible),A_S=float(np.mean([x['restricted_unstable_time'] for x in a])),
                median_TV=float(np.median([x['static_window']['metrics']['finger_tv_rate'] for x in eligible])) if eligible else None,
                mean_force=float(np.mean([x['static_window']['metrics']['mean_force'] for x in eligible])) if eligible else None,
                local_excess=float(np.mean([x['force_safety']['local_excess_integral'] for x in a])),
                total_excess=float(np.mean([x['force_safety']['total_excess_integral'] for x in a]))))
    selected={}
    for method in PRIMARY:
        a=[x for x in candidates if x['method']==method]
        best=max((x['joint_pass'],x['task_pass'],x['release_pass'],x['eligible']) for x in a)
        a=[x for x in a if (x['joint_pass'],x['task_pass'],x['release_pass'],x['eligible'])==best]
        fastest=min(x['A_S'] for x in a);a=[x for x in a if x['A_S']<=fastest+.20+1e-12]
        selected[method]=min(a,key=lambda x:(float('inf') if x['median_TV'] is None else x['median_TV'],float('inf') if x['mean_force'] is None else x['mean_force'],x['index']))
    result=dict(prefix=args.prefix,inputs_sha256=inputs,development_inputs=common,candidates=candidates,selected=selected,
        selection='joint task and actual force; then task, release, window coverage; A_S within .20s; median TV, mean force, index',
        formal_data_used=False)
    path=ROOT/'review'/f'{args.prefix}_selection.json';write(path,result)
    if args.apply:
        settings=json.loads((ROOT/'settings.json').read_text(encoding='utf-8'))
        settings['controllers']={m:x['gains'] for m,x in selected.items()}
        for method in ('unified_no_continuity','unified_no_geometry','unified_fixed_stage'):settings['controllers'][method]=selected['unified']['gains'].copy()
        settings['selection_source']=str(path.relative_to(ROOT));settings['selection_sha256']=sha(path)
        write(ROOT/'settings.json',settings)
    print(json.dumps(selected,ensure_ascii=False))

if __name__=='__main__':main()
