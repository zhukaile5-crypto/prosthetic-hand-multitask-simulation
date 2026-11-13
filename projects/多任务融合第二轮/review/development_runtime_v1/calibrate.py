"""Development-only tuning with explicit provenance and equal safety metrics."""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import numpy as np
from physics import ROOT, DT
from scenarios import DEV_NAMES
from controllers import Gains
from run_experiments import file_worker_pool, code_hashes, time_stamp, sha256


BASELINE_CANDIDATES = {
    'event': [dict(base=b,step_rate=r) for b in (1.2,1.45,1.8,2.2) for r in (4.,14.)],
    'ratio': [dict(base=b,ratio_kp=kp,ratio_ki=ki,ratio_floor=floor)
              for b in (1.2,1.45) for kp,ki,floor in
              ((.25,.05,1.45),(1.,.3,1.45),(1.,.3,1.8),(3.,2.,1.8))],
    'fusion': [dict(base=b,ratio_kp=kp,ratio_ki=ki,fusion_slip_rate=sr,fusion_floor=floor,predict_horizon=h)
               for b in (1.2,1.45) for kp,ki,sr,floor,h in
               ((3.,2.,24.,1.45,.015),(7.,6.,32.,1.45,.04),
                (1.,.3,16.,1.8,.015),(3.,2.,24.,1.8,.04))]}
COORD_SETTINGS = ((6.,1.,.15,1.),(12.,2.,.25,2.),(12.,2.,.15,1.),(24.,4.,.15,1.),
                  (6.,1.,.30,2.),(12.,2.,.30,3.),(0.,0.,.25,2.),(12.,1.,.25,2.))
CANDIDATES = BASELINE_CANDIDATES


def calibration_hashes():
    return dict(code_hashes(),**{'calibrate.py':sha256(ROOT/'calibrate.py')})


def trial_key(item):
    return (item['method'],item['scenario'],item['seed'],
            json.dumps(asdict(Gains(**item['gains'])),sort_keys=True))


def select_gains(results,seeds):
    """Rank full-task safe completion first; preserve all component scores."""
    selection,scores = {},[]
    for method,candidates in CANDIDATES.items():
        choices=[]
        for index,overrides in enumerate(candidates):
            expected=asdict(Gains(**overrides))
            rows=[r for r in results if r['method']==method and r['gains']==expected]
            expected_keys={(name,seed) for name in DEV_NAMES for seed in seeds}
            assert {(r['scenario'],r['seed']) for r in rows} == expected_keys
            assert len(rows)==len(expected_keys)
            safe=sum(r['safe_completed'] for r in rows)
            drops=sum(r['dropped'] or not r['setup_ok'] for r in rows)
            shifts=[r['max_slip_mm'] for r in rows if r['max_slip_mm'] is not None]
            # Includes acquisition as well as holding, matching safe_completed.
            force_excess=sum(r['total_force_excess_Ns'] for r in rows)
            budget_excess=sum(r['total_budget_excess_Ns'] for r in rows)
            shift=float(np.median(shifts)) if shifts else 1e9
            score=(-safe,drops,force_excess+budget_excess,shift)
            scores.append(dict(method=method,index=index,trials=len(rows),safe=safe,drops=drops,
                               median_shift_mm=shift,force_excess_Ns=force_excess,
                               budget_excess_Ns=budget_excess,excess_Ns=force_excess+budget_excess,
                               gains=overrides))
            choices.append((score,index,overrides))
        _,_,best=min(choices,key=lambda x:x[0])
        selection[method]=best
    # Ablations isolate mechanisms using the selected full method's gains.
    if 'coordinated' in selection:
        for name in ('coordinated_no_guard','coordinated_no_redistribution'):
            selection[name]=selection['coordinated'].copy()
    return selection,scores


def main():
    global CANDIDATES
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='baseline_calibration')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--stage',choices=('baseline','coordinated'),default='baseline')
    parser.add_argument('--baseline-gains',type=Path)
    parser.add_argument('--timestep',type=float,default=.0005)
    parser.add_argument('--seeds',nargs='+',type=int,default=[1101,1102,1103])
    args=parser.parse_args()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    if args.timestep <= 0 or abs(DT/args.timestep-round(DT/args.timestep)) > 1e-9:
        raise ValueError('Physics timestep must divide the control interval.')
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= set(range(1101,1106)):
        raise ValueError('Calibration accepts unique development seeds 1101-1105 only.')
    inherited = {}
    if args.stage == 'coordinated':
        if args.baseline_gains is None:
            raise ValueError('Coordinated tuning must inherit the selected baseline gains.')
        inherited=json.loads(args.baseline_gains.read_text(encoding='utf-8'))
        core=inherited['ratio']
        CANDIDATES={'coordinated':[dict(core,coord_boost_rate=r,coord_boost_max=m,
                                      coord_guard_margin=s,coord_guard_gain=g)
                                   for r,m,s,g in COORD_SETTINGS]}
    else:
        CANDIDATES=BASELINE_CANDIDATES
    out=(ROOT/'output'/args.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'trials.json').exists():
        raise RuntimeError('Completed calibration already exists; preserve it and use a new --out.')
    current_hashes=calibration_hashes()
    if (out/'manifest.json').exists():
        manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if (manifest['candidates'] != CANDIDATES or manifest['seeds'] != args.seeds or
                manifest['code_sha256'] != current_hashes or manifest['physics_timestep'] != args.timestep):
            raise RuntimeError('Cannot resume with changed code, candidates or seeds. Use a new --out.')
    else:
        manifest=dict(started=time_stamp(),stage=args.stage,baseline_gains=inherited,physics_timestep=args.timestep,
                      code_sha256=current_hashes,seeds=args.seeds,
                      scenes=list(DEV_NAMES),candidates=CANDIDATES,
                      selection='highest full-task safe completion, then lowest drop/setup-failure count, '
                                'then total acquisition+hold force/budget excess impulse, then median retained displacement',
                      safety='run_episode safe_completed; acquisition and hold share the same persistent limit rule')
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    jobs=[]
    for method,candidates in CANDIDATES.items():
        for ci,gains in enumerate(candidates):
            dest=out/f'{method}_{ci}'
            dest.mkdir(exist_ok=True)
            for name in DEV_NAMES:
                for seed in args.seeds:
                    jobs.append(dict(method=method,scenario=name,seed=seed,gains=gains,
                                     output=str(dest),save_trace=False,timestep=args.timestep))
    wanted={trial_key(job) for job in jobs}
    existing={}
    for pattern in ('worker_*_results.json','worker_*_results.tmp'):
        for path in sorted(out.rglob(pattern)):
            try:
                items=json.loads(path.read_text(encoding='utf-8'))
            except (json.JSONDecodeError,PermissionError):
                continue
            for item in items:
                key=trial_key(item)
                if key in wanted:
                    if key in existing and existing[key] != item:
                        raise RuntimeError(f'Conflicting recovered trial {key}; preserve and inspect both files.')
                    existing[key]=item
    pending=[job for job in jobs if trial_key(job) not in existing]
    manifest['resume_completed_trials']=len(existing)
    manifest['expected_trials']=len(jobs)
    manifest['workers']=args.workers
    print('Recovered',len(existing),'completed trials; pending',len(pending),flush=True)
    # New resume directories preserve prior worker evidence, including partial files.
    work_out=out
    if existing:
        resume_index=1
        while (out/f'resume_{resume_index}').exists():
            resume_index+=1
        work_out=out/f'resume_{resume_index}'
    results=list(existing.values())+file_worker_pool(pending,args.workers,work_out)
    if len(results) != len(jobs) or {trial_key(r) for r in results} != wanted:
        raise RuntimeError('Calibration is incomplete or contains duplicate trial identities.')
    selection,scores=select_gains(results,args.seeds)
    selection={**inherited,**selection}
    final_hashes=calibration_hashes()
    if final_hashes != current_hashes:
        raise RuntimeError('Code changed during calibration. Preserve worker outputs and rerun in a new --out.')
    (out/'trials.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'scores.json').write_text(json.dumps(scores,indent=2),encoding='utf-8')
    (out/'selected_gains.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    manifest['ended']=time_stamp()
    manifest['final_code_sha256']=final_hashes
    manifest['unchanged']=True
    manifest['dynamics_unchanged']=True
    manifest['completed_trials']=len(results)
    manifest['selected_gains_sha256']=sha256(out/'selected_gains.json')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    # The active file is updated only after every development trial and hash check passed.
    (ROOT/'selected_gains.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print(json.dumps(dict(selected=selection,scores=scores),indent=2),flush=True)


if __name__ == '__main__':
    main()

