"""Run an admitted, preregistered paired matrix in a fresh output directory."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,os,time

from session import TrialSession,ROOT,SCENE_BY_NAME,check_admission,load_json,jsonable,sha
from evaluate_runs import evaluate_trace

def jobs_from_plan(plan,phase):
    phase_config=plan['phases'][phase]
    if 'jobs' in phase_config:return [dict(job,split=phase) for job in phase_config['jobs']]
    methods=phase_config.get('methods',plan.get('methods',['ratio','unified']))
    scenarios=phase_config.get('scenarios',list(SCENE_BY_NAME))
    seeds=phase_config['seeds']
    return [dict(method=method,scenario=scenario,shape=SCENE_BY_NAME[scenario]['shape'],seed=int(seed),split=phase)
            for seed in seeds for scenario in scenarios for method in methods]

def key(job):return f"{job['scenario']}__{job['shape']}__{job['method']}__{job['seed']}"

def run_job(job):
    directory=Path(job['directory']);began=time.perf_counter();session=None
    try:
        session=TrialSession(method=job['method'],shape=job['shape'],scenario=job['scenario'],
            seed=job['seed'],mode='scheduled',split=job['split'])
        while session.running:session.step_one_sensor_tick()
        metadata=session.save(directory)
    except Exception as exc:
        # Admission failure is not an experimental result and must abort dispatch.
        if not check_admission()['passed']:raise
        if session is not None and session.trace and not directory.exists():
            session.running=False;session.completed=False;session.failure_reason='runtime_failure';session.failure_at=session.time
            metadata=session.save(directory)
        else:
            directory.mkdir(parents=True,exist_ok=True)
            metadata=dict(method=job['method'],shape=job['shape'],scenario=SCENE_BY_NAME[job['scenario']],
                seed=job['seed'],split=job['split'],mode='scheduled',completed=False)
        metadata.update(runtime_failure=type(exc).__name__,error=str(exc))
        (directory/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        result=dict(method=job['method'],shape=job['shape'],scenario=job['scenario'],seed=job['seed'],
            split=job['split'],mode='scheduled',boundary=bool(SCENE_BY_NAME[job['scenario']].get('boundary',False)),
            task_success=False,A_S=1.4,record_complete=False,static_eligible=False,
            release_planned=True,release_exposed=False,release_success=False,
            failure_reason='runtime_failure',error=str(exc),wall_seconds=time.perf_counter()-began)
        (directory/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        return result
    # An evaluator error is a software defect to repair/review, not fabricated
    # evidence that the policy failed a physical task. Preserve the raw trial.
    result=evaluate_trace(directory/'trace.npz')
    result['wall_seconds']=time.perf_counter()-began
    (directory/'result.json').write_text(json.dumps(jsonable(result),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','confirmation','robustness'],required=True)
    p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--resume',action='store_true')
    p.add_argument('--limit',type=int,help='Development smoke only; never a complete confirmation matrix.')
    a=p.parse_args();admission=check_admission()
    if not admission['passed']:raise SystemExit(' '.join(admission['problems']))
    plan_path=ROOT/admission['record']['evidence']['experiment_plan']['path'];plan=load_json(plan_path)
    jobs=jobs_from_plan(plan,a.phase)
    full_count=len(jobs)
    if a.limit is not None:
        if a.phase!='development':raise ValueError('Only development may select a smoke prefix')
        jobs=jobs[:a.limit]
    destination=(ROOT/a.out).resolve()
    if not destination.is_relative_to(ROOT):raise ValueError('Output must stay in independent integration directory')
    manifest=dict(schema='paired_shadow_run_matrix_v1',phase=a.phase,planned_jobs=jobs,
        phase_total_jobs=full_count,complete_phase_requested=len(jobs)==full_count,
        experiment_plan_sha256=sha(plan_path),admission_sha256=admission['manifest_sha256'],
        source_sha256=admission['record']['source_sha256'])
    if destination.exists():
        if not a.resume:raise FileExistsError('Choose a fresh output or explicitly resume matching evidence')
        if load_json(destination/'manifest.json')!=manifest:raise ValueError('Resume manifest does not match frozen run')
    else:
        (destination/'runs').mkdir(parents=True)
        (destination/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    pending=[];results=[]
    for job in jobs:
        directory=destination/'runs'/key(job)
        if (directory/'result.json').exists():
            results.append(load_json(directory/'result.json'));continue
        if directory.exists():raise ValueError(f'Incomplete trial directory requires explicit review: {directory}')
        pending.append({**job,'directory':str(directory)})
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
    with ProcessPoolExecutor(max_workers=max(1,a.workers)) as pool:
        futures={pool.submit(run_job,job):job for job in pending}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            print(json.dumps(dict(completed=len(results),planned=len(jobs),trial=key(futures[future]),
                success=result['task_success'],failure=result.get('failure_reason')),ensure_ascii=False),flush=True)
    order={key(job):i for i,job in enumerate(jobs)}
    results.sort(key=lambda r:order[f"{r['scenario']}__{r['shape']}__{r['method']}__{r['seed']}"])
    (destination/'trials.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    certificate=dict(execution_complete=len(results)==len(jobs),trials=len(results),planned=len(jobs),
        complete_phase_requested=manifest['complete_phase_requested'],phase=a.phase,
        runtime_failures=sum(r.get('failure_reason')=='runtime_failure' for r in results),
        trials_sha256=sha(destination/'trials.json'),manifest_sha256=sha(destination/'manifest.json'))
    (destination/'execution_complete.json').write_text(json.dumps(certificate,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(certificate,ensure_ascii=False))

if __name__=='__main__':main()
