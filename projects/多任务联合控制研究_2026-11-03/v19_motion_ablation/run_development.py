"""Fresh-output V19MotionAblation development runner; original runner/evaluator stay intact.

Import and zero-step construction do not advance a plant. Only main dispatches
physical trials. Every assigned job and runtime failure remains in the ledger.
"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,json,os,time,platform,hashlib,traceback
from importlib.metadata import version,PackageNotFoundError
import numpy as np
from v19_motion_ablation.runtime import ReleaseSession as SupportSession,METHODS,MOTION_MODES,source_files,source_identity
from research_imports import FOUNDATION
from v13_a1.run_bridge import environment as pinned_environment
from session import SCENE_BY_NAME,check_admission,jsonable
from evaluate_runs import evaluate_trace
from evaluation import evaluate_research


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def package_version(name):
    try:return version(name)
    except PackageNotFoundError:return None


def output_path(value):
    out=(ROOT/value).resolve();allowed=(ROOT/'output').resolve()
    if not out.is_relative_to(allowed) or out==allowed:raise ValueError('V19MotionAblation output must be a fresh strict descendant of ROOT/output')
    if out.exists():raise FileExistsError('All development attempts retained; choose a new output')
    return out


def snapshot_sources(destination,identity,methods):
    records={};files=source_files(methods)
    if set(files)!=set(identity):raise RuntimeError('V19MotionAblation source set changed before snapshot')
    for name,path in files.items():
        payload=path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=identity[name]:raise RuntimeError('V19MotionAblation source changed: '+name)
        target=destination/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
        records[name]=dict(original_path=str(path),snapshot_path=target.relative_to(destination).as_posix(),sha256=identity[name],bytes=len(payload))
    write(destination/'source_snapshot_manifest.json',dict(schema='v19_motion_ablation_source_snapshot_v1',files=records,all_snapshot_bytes_verified=True))


def run_one(job,expected_identity=None):
    """One assigned job; preserve partial traces and explicit exceptions."""
    session=None;destination=Path(job['destination']);began=time.perf_counter()
    try:
        if expected_identity is not None and source_identity([job['method']])!=expected_identity:
            raise RuntimeError('Frozen V19MotionAblation worker source identity mismatch before construction')
        session=SupportSession(method=job['method'],scenario=job['scenario'],seed=job['seed'],split=job['split'],motion_mode=job['motion_mode'],task_policy=job['task_policy'],receiver_policy=job['receiver_policy'])
        if expected_identity is not None and session.research_sources!=expected_identity:
            raise RuntimeError('V19MotionAblation session loaded a different source identity')
        while session.running:session.step_one_sensor_tick()
        session.save(destination)
        result=evaluate_trace(destination/'trace.npz')
        result['research_metrics']=evaluate_research(destination/'trace.npz')
        result['motion_mode']=job['motion_mode']
        if expected_identity is not None and source_identity([job['method']])!=expected_identity:
            raise RuntimeError('Frozen V19MotionAblation worker sources changed during execution')
        result['wall_seconds']=time.perf_counter()-began
        write(destination/'result.json',result)
        return result
    except Exception as exc:
        failure=dict(type=type(exc).__name__,error=str(exc),job=job,
                     time_s=None if session is None else session.time,wall_seconds=time.perf_counter()-began,traceback=traceback.format_exc())
        if session is not None:
            session.running=False;session.completed=False
            session.failure_reason='v19_motion_ablation_runtime_or_evaluation_failure';session.failure_at=session.time
            if session.trace and not destination.exists():
                try:session.save(destination)
                except Exception as save_exc:
                    failure['partial_save_error']=dict(type=type(save_exc).__name__,error=str(save_exc),traceback=traceback.format_exc())
        destination.mkdir(parents=True,exist_ok=True)
        write(destination/'runtime_error.json',failure)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--registration',required=True)
    parser.add_argument('--task-policy',choices=('legacy','integrated'),default='integrated')
    parser.add_argument('--receiver-policy',choices=('blind','guarded','coordinated'),default='coordinated')
    parser.add_argument('--motion-mode',choices=MOTION_MODES,default='scenario_default')
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=['ratio_effort','fixed_effort','fusion_effort'])
    parser.add_argument('--scenarios',nargs='+',default=['nominal','dynamic_transport','fragile_conflict','collision'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[610212])
    parser.add_argument('--physical-randomization',action='store_true')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args(argv)
    if any(s not in SCENE_BY_NAME for s in args.scenarios):raise ValueError('Unknown admitted scenario')
    if any(not 610000<=s<620000 for s in args.seeds):raise ValueError('Development seed must stay within610000..619999')
    if len(set(args.methods))!=len(args.methods) or len(set(args.scenarios))!=len(args.scenarios) or len(set(args.seeds))!=len(args.seeds):raise ValueError('Duplicate assigned dimensions forbidden')
    if not 1<=args.workers<=16:raise ValueError('Explicit worker count must be1..16')
    destination=output_path(args.out);admission=check_admission()
    if not admission['passed']:raise RuntimeError(admission['problems'])
    identity=source_identity(args.methods)
    registration_path=(ROOT/args.registration).resolve()
    registration=json.loads(registration_path.read_text(encoding='utf-8-sig'))
    expected=dict(methods=args.methods,scenarios=args.scenarios,seeds=args.seeds,motion_mode=args.motion_mode,physical_randomization=args.physical_randomization,task_policy=args.task_policy,receiver_policy=args.receiver_policy)
    if registration['source_sha256']!=identity or registration['environment']!=pinned_environment() or registration['assignment']!=expected or not registration['software_gate_passed']:
        raise ValueError('Predeclared joint source/environment/software/assignment pin changed')
    split='robustness' if args.physical_randomization else 'development'
    jobs=[dict(method=m,scenario=s,seed=seed,split=split,motion_mode=args.motion_mode,task_policy=args.task_policy,receiver_policy=args.receiver_policy,destination=str(destination/'runs'/f'{s}__{m}__{seed}'))
          for seed in args.seeds for s in args.scenarios for m in args.methods]
    destination.mkdir(parents=True,exist_ok=False)
    plan=dict(schema='v19_motion_ablation_development_plan_v1',status='development_execution_requested',phase='development',
              performance_claim_allowed=False,registration=str(registration_path),registration_sha256=sha(registration_path),methods=args.methods,scenarios=args.scenarios,seeds=args.seeds,
              jobs=jobs,planned=len(jobs),task_policy=args.task_policy,receiver_policy=args.receiver_policy,physical_randomization=args.physical_randomization,workers=args.workers,motion_mode=args.motion_mode,
              schedule=dict(duration_s=16.,handoff_s=7.5,release_s=12.,sensor_period_s=.005),
              observer_warmup_s=dict(all_methods=.2),
              preview_control='Assigned FullV19 combined motion-effort-task ablation; same holding, progress, normal/reference QP, governor, branch, devices, plant, permissions, rates, metrics and16s schedule; actual observations retained and no force prediction',
              physical_random_stream='unchanged foundation seed+511000; same normalized draws within seed',
              observation_random_stream='unchanged foundation seed+910000; independent receiver SeedSequence seed,0x56313552 with fixed6Gaussian+1uniform per200Hz tick',
              admission='V19MotionAblation development only; no confirmation or global model admission')
    write(destination/'plan.json',plan)
    try:
        snapshot_sources(destination,identity,args.methods)
    except Exception as exc:
        failure=dict(stage='source_snapshot',error=type(exc).__name__+': '+str(exc))
        write(destination/'trials.json',[]);write(destination/'runtime_failures.json',[failure])
        write(destination/'assignment_ledger.json',[dict(job=j,status='not_executed_due_to_source_snapshot_failure') for j in jobs])
        write(destination/'execution_complete.json',dict(execution_complete=False,trials=0,planned=len(jobs),
            runtime_failures=[failure],sources_unchanged=False,trials_sha256=sha(destination/'trials.json'),
            phase='development',performance_claim_allowed=False))
        return 1
    manifest=dict(schema='v19_motion_ablation_independent_manifest_v1',phase='development',performance_claim_allowed=False,
                  jobs=jobs,created_utc=datetime.now(timezone.utc).isoformat(),research_source_sha256=identity,motion_mode=args.motion_mode,
                  foundation=str(FOUNDATION),foundation_admission_sha256=admission['manifest_sha256'],
                  physical_randomization=args.physical_randomization,plan_sha256=sha(destination/'plan.json'),
                  source_snapshot_manifest_sha256=sha(destination/'source_snapshot_manifest.json'),
                  source_namespace='Assigned motion-task ablation with complete frozen FullV19 source parent plus four independent namespace files',
                  note='All prior V6/V6.1/confirmation material is seen development evidence; no independent superiority claim.')
    write(destination/'manifest.json',manifest)
    write(destination/'environment.json',dict(python=sys.version,executable=sys.executable,numpy=np.__version__,
          mujoco=package_version('mujoco'),platform=platform.platform(),entry_module='v19_motion_ablation.run_development',pinned_environment=pinned_environment()))
    results=[];failures=[];began=time.perf_counter()
    try:
        for method in args.methods:
            try:
                session=SupportSession(method=method,scenario=args.scenarios[0],seed=args.seeds[0],split=split,motion_mode=args.motion_mode,task_policy=args.task_policy,receiver_policy=args.receiver_policy)
                if session.time!=0. or session.tick!=0 or session.trace:raise RuntimeError('Construction unexpectedly advanced or recorded physics')
            except Exception as exc:
                failure=dict(stage='zero_step_preflight',method=method,error=type(exc).__name__+': '+str(exc))
                failures.append(failure);write(destination/'preflight_error.json',failure);raise
        os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(run_one,job,identity):job for job in jobs}
            for future in as_completed(futures):
                job=futures[future]
                try:
                    result=future.result();results.append(result)
                    print(json.dumps(dict(completed=len(results),planned=len(jobs),method=job['method'],scenario=job['scenario'],seed=job['seed'],task_success=result['task_success']),ensure_ascii=False),flush=True)
                except Exception as exc:
                    failure=dict(job=job,error=type(exc).__name__+': '+str(exc));failures.append(failure)
                    print(json.dumps(failure,ensure_ascii=False),flush=True)
    except Exception as exc:
        if not failures:failures.append(dict(stage='dispatch',error=type(exc).__name__+': '+str(exc)))
    finally:
        write(destination/'trials.json',results);write(destination/'runtime_failures.json',failures)
        try:final_identity=source_identity(args.methods)
        except Exception as exc:final_identity={};failures.append(dict(stage='final_source_check',error=str(exc)));write(destination/'runtime_failures.json',failures)
        sources_unchanged=identity==final_identity and registration['environment']==pinned_environment()
        completed_keys={(r['scenario'],r['method'],r['seed']) for r in results}
        assignment=[dict(job=job,status='evaluated' if (job['scenario'],job['method'],job['seed']) in completed_keys
                         else 'runtime_failed' if any(f.get('job')==job for f in failures) else 'not_executed_due_to_preflight_or_dispatch_failure') for job in jobs]
        write(destination/'assignment_ledger.json',assignment)
        finish=dict(execution_complete=len(results)==len(jobs) and not failures and sources_unchanged,
                    trials=len(results),planned=len(jobs),runtime_failures=failures,sources_unchanged=sources_unchanged,
                    trials_sha256=sha(destination/'trials.json'),assignment_ledger_sha256=sha(destination/'assignment_ledger.json'),
                    plan_sha256=sha(destination/'plan.json'),manifest_sha256=sha(destination/'manifest.json'),
                    phase='development',performance_claim_allowed=False,wall_seconds=time.perf_counter()-began,
                    ended_utc=datetime.now(timezone.utc).isoformat())
        write(destination/'execution_complete.json',finish)
    print(json.dumps(finish,ensure_ascii=False),flush=True)
    return 0 if finish['execution_complete'] else 1


if __name__=='__main__':raise SystemExit(main())
