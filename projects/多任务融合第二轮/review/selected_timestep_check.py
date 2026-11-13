"""Diagnostic timestep comparison after final second-round gains are selected.

This script never edits core code or gains. It reports all paired results without
turning qualitative label agreement into a numerical-convergence certificate.
Run only after the parent has frozen selected_gains.json.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
import math
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CORE_NAMES = ('physics.py','controllers.py','scenarios.py','run_experiments.py')
SCENES = ('friction_drop','local_limit','release_slip','sensor_dropout','collision')
METHODS = ('ratio','coordinated')
TIMESTEPS = (.0005,.00025)
SEED = 1104
LABELS = ('setup_ok','completed','safe_completed','peak_limit_completed')
NUMERICS = ('physics_peak_force_N','physics_peak_penetration_mm',
            'physics_force_excess_Ns','physics_budget_excess_Ns',
            'max_slip_mm','unrestricted_shift_mm','release_latency_s')

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fingerprints():
    return {name:sha256(ROOT/name) for name in CORE_NAMES}

def run(job):
    sys.path.insert(0,str(ROOT))
    from run_experiments import run_episode
    row=run_episode(job)
    row['physics_timestep']=job['timestep']
    return row

def clean_json(value):
    if isinstance(value,dict):
        return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):
        return None
    return value

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gains-sha256',required=True,
        help='Parent-supplied expected hash of the final selected_gains.json.')
    parser.add_argument('--workers',type=int,choices=(1,2,3,4),default=4)
    args=parser.parse_args()
    gains_path=ROOT/'selected_gains.json'
    gains_hash=sha256(gains_path)
    if gains_hash != args.gains_sha256:
        raise SystemExit('Final gains hash mismatch; no simulation was started.')
    selected=json.loads(gains_path.read_text(encoding='utf-8'))
    missing=[m for m in METHODS if m not in selected]
    if missing:
        raise SystemExit(f'Final selected gains missing methods {missing}; no simulation was started.')
    before=fingerprints()
    jobs=[dict(scenario=s,method=m,seed=SEED,timestep=dt,
               gains=selected[m],save_trace=False,
               output=str(ROOT/'review'/'selected_timestep_diagnostic'))
          for s in SCENES for m in METHODS for dt in TIMESTEPS]
    rows=[];errors=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run,j):j for j in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:
                row=future.result();rows.append(row)
                print(json.dumps({k:row.get(k) for k in
                    ('scenario','method','physics_timestep',*LABELS,*NUMERICS)},
                    ensure_ascii=False),flush=True)
            except Exception:
                errors.append(dict(job=job,exception=traceback.format_exc()))
                print(f"Diagnostic failed: {job['scenario']} {job['method']} dt={job['timestep']}",flush=True)
    rows.sort(key=lambda r:(r['scenario'],r['method'],r['physics_timestep']))
    by_key={(r['scenario'],r['method'],r['physics_timestep']):r for r in rows}
    pairs=[]
    for scene in SCENES:
        for method in METHODS:
            coarse=by_key.get((scene,method,.0005))
            fine=by_key.get((scene,method,.00025))
            pair=dict(scenario=scene,method=method,seed=SEED)
            if coarse is None or fine is None:
                pair['incomplete']=True;pairs.append(pair);continue
            pair['labels']={k:dict(coarse=coarse.get(k),fine=fine.get(k),
                agrees=coarse.get(k)==fine.get(k)) for k in LABELS}
            pair['numeric_differences']={}
            for metric in NUMERICS:
                a,b=coarse.get(metric),fine.get(metric)
                values=dict(coarse=a,fine=b,delta_fine_minus_coarse=None,
                            relative_absolute_difference=None)
                if a is not None and b is not None and math.isfinite(a) and math.isfinite(b):
                    values['delta_fine_minus_coarse']=b-a
                    values['relative_absolute_difference']=abs(b-a)/max(abs(a),abs(b),1e-12)
                pair['numeric_differences'][metric]=values
            pairs.append(pair)
    after=fingerprints()
    gains_after=sha256(gains_path)
    complete=len(rows)==len(jobs) and not errors and len(by_key)==len(jobs)
    unchanged=before==after and gains_hash==gains_after
    disagreements=[dict(scenario=p['scenario'],method=p['method'],
        labels=[k for k,v in p['labels'].items() if not v['agrees']])
        for p in pairs if 'labels' in p and any(not v['agrees'] for v in p['labels'].values())]
    max_absolute_differences={}
    for metric in NUMERICS:
        values=[abs(p['numeric_differences'][metric]['delta_fine_minus_coarse'])
                for p in pairs if 'numeric_differences' in p and
                p['numeric_differences'][metric]['delta_fine_minus_coarse'] is not None]
        max_absolute_differences[metric]=max(values) if values else None
    conclusion=(f'{len(rows)}/20 个开发诊断完成；{len(disagreements)}/10 对出现离散判定差异。'
        '连续差异逐项保留，未设置任意数值阈值将其自动判为收敛；'
        '此检查不属于正式留出性能或统计优势证据。')
    manifest=dict(purpose='Selected-controller diagnostic timestep sensitivity; no tuning or formal proof',
        seed=SEED,seed_role='fixed development diagnostic; not formal heldout',
        scenarios=SCENES,methods=METHODS,timesteps=TIMESTEPS,max_workers=args.workers,
        expected_trials=len(jobs),expected_pairs=10,code_hashes_before=before,
        code_hashes_after=after,gains_source=str(gains_path),
        gains_sha256_before=gains_hash,gains_sha256_after=gains_after,
        selected_method_gains={m:selected[m] for m in METHODS},
        script_source=str(Path(__file__).resolve()),script_sha256=sha256(__file__),
        source_model_sha256=sha256(ROOT.parent/'五指假肢抓握仿真'/'model.xml'),
        platform=platform.platform(),python=sys.version,wall_seconds=time.perf_counter()-start)
    report=dict(manifest=manifest,results=rows,pairs=pairs,errors=errors,
        complete=complete,code_and_gains_unchanged=unchanged,
        label_disagreements=disagreements,
        max_absolute_numeric_differences=max_absolute_differences,
        numerical_convergence_certified=False,conclusion=conclusion)
    target=ROOT/'review'/'selected_timestep_validation.json'
    target.write_text(json.dumps(clean_json(report),indent=2,ensure_ascii=False,
                                 allow_nan=False),encoding='utf-8')
    print(conclusion,flush=True)
    if not complete or not unchanged:
        raise SystemExit('Diagnostic incomplete or inputs changed; all partial evidence retained in output JSON.')

if __name__=='__main__':
    main()
