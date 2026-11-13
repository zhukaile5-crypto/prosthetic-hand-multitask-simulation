"""Equal-condition development tuning; never reads formal or robustness results."""
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from physics import ROOT
from scenarios import DEV_NAMES
from controllers import Gains
from run_experiments import file_worker_pool, code_hashes, time_stamp


CANDIDATES = {
 'event': [dict(step_rate=k) for k in (4.,9.,14.)],
 'ratio': [dict(ratio_kp=kp,ratio_ki=ki) for kp,ki in ((3.,3.),(7.,6.),(12.,9.))],
 'adaptive': [{}],
 'fusion': [dict(ratio_kp=kp,ratio_ki=ki,fusion_slip_rate=sr,fusion_floor=floor,predict_horizon=h)
            for kp,ki,sr,floor,h in [(2.,1.,16.,1.45,.005),(3.,2.,24.,1.45,.015),
                                    (3.,2.,32.,1.45,.04),(6.,3.,16.,1.25,.04),
                                    (3.,2.,16.,1.25,.01),(7.,6.,32.,1.45,.04),
                                    (2.,.5,24.,1.45,.005),(3.,1.,24.,1.45,.025)]]}


if __name__ == '__main__':
    out = ROOT/'output/calibration_v1'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'trials.json').exists():
        raise RuntimeError('Calibration results already exist. Preserve them and use a new version.')
    if (out/'manifest.json').exists():
        manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        assert manifest['candidates']==CANDIDATES
    else:
        manifest = dict(started=time_stamp(),code_sha256=code_hashes(),seeds=[1001,1002,1003],
                        scenes=list(DEV_NAMES),candidates=CANDIDATES,
                        selection='highest safe completion, then lowest drop count, then median retained displacement, then excess impulse')
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    jobs=[]
    for method,candidates in CANDIDATES.items():
        for ci,gains in enumerate(candidates):
            dest=out/f'{method}_{ci}'
            dest.mkdir(exist_ok=True)
            for name in DEV_NAMES:
                for seed in manifest['seeds']:
                    jobs.append(dict(method=method,scenario=name,seed=seed,gains=gains,
                                     output=str(dest),save_trace=False))
    existing={}
    for pattern in ('worker_*_results.json','worker_*_results.tmp'):
        for path in out.rglob(pattern):
            items=json.loads(path.read_text(encoding='utf-8'))
            for item in items:
                key=(item['method'],item['scenario'],item['seed'],json.dumps(item['gains'],sort_keys=True))
                existing[key]=item
    pending=[]
    for job in jobs:
        key=(job['method'],job['scenario'],job['seed'],json.dumps(asdict(Gains(**job['gains'])),sort_keys=True))
        if key not in existing:
            pending.append(job)
    manifest['resume_completed_trials']=len(existing)
    manifest['resumed_code_sha256']=code_hashes()
    print('Recovered',len(existing),'completed trials; pending',len(pending),flush=True)
    results=list(existing.values())+file_worker_pool(pending,4,out/'resume_1' if existing else out)
    (out/'trials.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    selection={}
    scores=[]
    for method,candidates in CANDIDATES.items():
        choices=[]
        for index,overrides in enumerate(candidates):
            expected=asdict(Gains(**overrides))
            rows=[r for r in results if r['method']==method and r['gains']==expected]
            assert len(rows)==len(DEV_NAMES)*len(manifest['seeds'])
            safe=sum(r['safe_completed'] for r in rows)
            drops=sum(r['dropped'] or not r['setup_ok'] for r in rows)
            shifts=[r['max_slip_mm'] for r in rows if r['max_slip_mm'] is not None]
            excess=sum(r['force_excess_Ns']+r['budget_excess_Ns'] for r in rows)
            score=(-safe,drops,float(np.median(shifts)) if shifts else 1e9,excess)
            summary=dict(method=method,index=index,safe=safe,drops=drops,
                         median_shift_mm=score[2],excess_Ns=excess,gains=overrides)
            scores.append(summary)
            choices.append((score,index,overrides))
        _,index,best=min(choices,key=lambda x:x[0])
        selection[method]=best
    selection['fusion_no_prediction']=selection['fusion'].copy()
    selection['fusion_no_distribution']=selection['fusion'].copy()
    selection['constant']={}
    (ROOT/'selected_gains.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    (out/'scores.json').write_text(json.dumps(scores,indent=2),encoding='utf-8')
    manifest['ended']=time_stamp()
    manifest['final_code_sha256']=code_hashes()
    manifest['unchanged']=manifest['final_code_sha256']==manifest['code_sha256']
    manifest['dynamics_unchanged']=all(manifest['final_code_sha256'][n]==manifest['code_sha256'][n]
                                       for n in ('physics.py','controllers.py','scenarios.py'))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(selected=selection,scores=scores),indent=2),flush=True)
