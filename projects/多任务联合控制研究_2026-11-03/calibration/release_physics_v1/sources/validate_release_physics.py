"""Development-only physical probes of the explicitly selected release mode."""
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,time,os,shutil
import numpy as np
import research_imports
from research_imports import ROOT
from session import sha,jsonable

def probe(job):
    from runtime import ResearchSession
    from lifecycle_plant import LifecyclePlant
    from lifecycle_release import LifecycleRelease
    from hand_plant import CommonPreparation
    from evaluate_runs import evaluate_trace
    from evaluation import evaluate_research
    s=ResearchSession(method='ratio_release',scenario='nominal',shape=job['shape'],seed=610120,split='development')
    if abs(s.config.timestep-job['dt'])>1e-12:
        s.config=replace(s.config,timestep=job['dt'])
        s.plant=LifecyclePlant(s.config);s.preparation=CommonPreparation(s.plant)
        s.release_driver=LifecycleRelease(s.plant);s.last_truth=s.plant.truth()
    began=time.perf_counter()
    while s.running:s.step_one_sensor_tick()
    target=Path(job['directory']);s.save(target)
    result=evaluate_trace(target/'trace.npz');result['research_metrics']=evaluate_research(target/'trace.npz')
    result['probe_dt']=job['dt'];result['wall_seconds']=time.perf_counter()-began
    (target/'result.json').write_text(json.dumps(jsonable(result),ensure_ascii=False,indent=2),encoding='utf-8')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--shapes',nargs='+',default=['capsule','sphere','box']);p.add_argument('--dts',nargs='+',type=float,default=[.0005,.00025]);a=p.parse_args()
    target=(ROOT/a.out).resolve()
    if not target.is_relative_to(ROOT) or target.exists():raise ValueError('Fresh independent probe output required')
    sources={n:sha(ROOT/n) for n in ('validate_release_physics.py','runtime.py','research_imports.py','lifecycle_plant.py','lifecycle_release.py','evaluation.py','protocol.json')}
    target.mkdir(parents=True);(target/'sources').mkdir()
    for n in sources:shutil.copy2(ROOT/n,target/'sources'/n)
    jobs=[dict(shape=shape,dt=dt,directory=str(target/f'{shape}__{dt:.5f}')) for shape in a.shapes for dt in a.dts]
    (target/'manifest.json').write_text(json.dumps(dict(phase='development_physical_probe',sources=sources,jobs=jobs,superiority_claim=False),indent=2),encoding='utf-8')
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(probe,j) for j in jobs]):
            r=future.result();results.append(r)
            print(json.dumps(dict(shape=r['shape'],dt=r['probe_dt'],success=r['task_success'],
                force_pass=r['safety']['force_protection_pass'],peak=max(r['safety']['finger_peak']),
                release=r['research_metrics']['release']),ensure_ascii=False),flush=True)
    final=dict(execution_complete=len(results)==len(jobs),results=results,
        sources_unchanged=all(sha(ROOT/n)==value for n,value in sources.items()),
        scope='Three standard shapes, baseline holding plus new release, numerical step comparison only; no confirmation performance claim')
    (target/'summary.json').write_text(json.dumps(jsonable(final),ensure_ascii=False,indent=2),encoding='utf-8')
    if not final['sources_unchanged']:raise RuntimeError('Executed release dependencies changed')

if __name__=='__main__':main()
