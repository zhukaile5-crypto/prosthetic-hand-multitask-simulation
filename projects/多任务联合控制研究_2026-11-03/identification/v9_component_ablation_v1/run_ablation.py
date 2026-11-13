"""Eight assigned full16s trials; --execute is intentionally required."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,importlib.metadata,json,os,platform,shutil,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from identification.v9_component_ablation_v1.ablation_runtime import ComponentSession,PACKAGE,source_files,source_identity,read_plan,sha
from evaluate_runs import evaluate_trace
from evaluation import evaluate_research

PHYSICAL_FIELDS=('time','normal','skin_normal','shear','palm','hand_other','all_hand_object','support',
 'position','relative','rotation','relative_velocity','angular_velocity','lift','lift_ref',
 'wrist_position','requested_wrist_reference','qpos','qvel','ctrl_before','ctrl','alpha_before','alpha',
 'executed_action','warning_count','collision','actual_motion_target','executed_motion_reference',
 'penetration','world_penetration','substep_time','substep_duration','substep_normal','substep_skin_normal',
 'substep_palm','substep_hand_other','substep_all_hand_object','substep_support','substep_penetration','substep_world_penetration')
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def versions():
    def version(n):
        try:return importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:return None
    return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
        packages={n:version(n) for n in ('numpy','mujoco')},
        task_environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','PYTHONIOENCODING','PYTHONDONTWRITEBYTECODE')})
def worker(job,identity):
    session=None;dest=Path(job['destination']);began=time.perf_counter()
    try:
        if source_identity()!=identity:raise RuntimeError('Pinned source mismatch before construction')
        session=ComponentSession(component_variant=job['component_variant'],scenario=job['scenario'],seed=job['seed'],split='robustness')
        if session.method!='joint' or session.research_sources!=identity:raise RuntimeError('Routing/source identity changed')
        while session.running:session.step_one_sensor_tick()
        session.save(dest);result=evaluate_trace(dest/'trace.npz')
        result['research_metrics']=evaluate_research(dest/'trace.npz')
        result.update(component_variant=job['component_variant'],component_flags=job['component_flags'],motion_mode='transport',wall_seconds=time.perf_counter()-began)
        if source_identity()!=identity:raise RuntimeError('Source changed during execution')
        write(dest/'result.json',result);return result,None
    except Exception as exc:
        failure=dict(job=job,type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
            time_s=None if session is None else session.time,wall_seconds=time.perf_counter()-began)
        if session is not None:
            session.running=False;session.completed=False;session.failure_reason='component_ablation_runtime_failure';session.failure_at=session.time
            try:
                if session.trace and not dest.exists():session.save(dest)
            except Exception as save_exc:failure['partial_save_error']=repr(save_exc)
        dest.mkdir(parents=True,exist_ok=True);write(dest/'runtime_error.json',failure)
        return dict(method='joint',scenario=job['scenario'],seed=job['seed'],component_variant=job['component_variant'],
            task_success=False,record_complete=False,runtime_failure=True,research_metrics=None),failure

def endpoint_checks(output,plan):
    results=[]
    for scene in plan['environments']:
        for variant,method in [('P0H0','ratio_preview'),('P1H1','joint')]:
            new=output/'runs'/f'{scene}__{variant}__610212'/'trace.npz'
            old=ROOT/'output/development_v9_transport_v1/runs'/f'{scene}__{method}__610212'/'trace.npz'
            detail={};missing=[]
            if new.is_file():
                with np.load(new,allow_pickle=False) as a,np.load(old,allow_pickle=False) as b:
                    for field in PHYSICAL_FIELDS:
                        if field not in a.files or field not in b.files:missing.append(field);continue
                        x=a[field];y=b[field];detail[field]=x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
            else:missing=['trace.npz']
            results.append(dict(scenario=scene,component_variant=variant,reference_method=method,
                reference_trace_sha256=sha(old),all_physical_arrays_bit_exact=not missing and all(detail.values()),missing=missing,fields=detail))
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output',default='output/causal8_v1');args=p.parse_args()
    plan=read_plan();identity=source_identity()
    if not args.execute:
        print(json.dumps(dict(preflight_passed=True,planned=8,source_count=len(identity),physics_executed=False,
            instruction='Run verify.py first. Only the root executor should use --execute.')));return 0
    out=(PACKAGE/args.output).resolve()
    if not out.is_relative_to(PACKAGE.resolve()) or out==PACKAGE.resolve() or out.exists():raise ValueError('Fresh output inside ablation package required')
    if not 1<=args.workers<=8:raise ValueError('workers must be1..8')
    out.mkdir(parents=True);jobs=[]
    for scene in plan['environments']:
        for variant,flags in plan['variants'].items():jobs.append(dict(method='joint',scenario=scene,seed=610212,split='robustness',motion_mode='transport',component_variant=variant,component_flags=flags,destination=str(out/'runs'/f'{scene}__{variant}__610212')))
    snapshots={}
    for key,path in source_files().items():
        dest=out/'sources'/key;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        if sha(dest)!=identity[key]:raise RuntimeError('Snapshot byte mismatch')
        snapshots[key]=dict(original_path=str(path),snapshot_path=str(dest.relative_to(out)),sha256=identity[key],bytes=path.stat().st_size)
    write(out/'source_snapshot_manifest.json',dict(files=snapshots,all_snapshot_bytes_verified=True))
    write(out/'plan.json',plan);write(out/'environment.json',versions())
    manifest=dict(schema='v9_component_ablation_execution_v1',phase='development_identification',performance_claim_allowed=False,
        jobs=jobs,research_source_sha256=identity,source_snapshot_manifest_sha256=sha(out/'source_snapshot_manifest.json'),registration_sha256=sha(PACKAGE/'plan.json'))
    write(out/'manifest.json',manifest);ledger=[dict(job=j,status='assigned') for j in jobs];write(out/'assignment_ledger.json',ledger)
    results=[];failures=[];began=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        work={pool.submit(worker,j,identity):j for j in jobs}
        for future in as_completed(work):
            j=work[future]
            try:result,failure=future.result()
            except Exception as exc:
                failure=dict(job=j,type=type(exc).__name__,error=str(exc));result=dict(method='joint',scenario=j['scenario'],seed=610212,component_variant=j['component_variant'],task_success=False,record_complete=False,runtime_failure=True,research_metrics=None)
            results.append(result)
            if failure:failures.append(failure)
            next(x for x in ledger if x['job']==j)['status']='runtime_failed' if failure else 'evaluated'
            write(out/'assignment_ledger.json',ledger);write(out/'runtime_failures.json',failures)
            print(json.dumps(dict(completed=len(results),planned=8,scenario=j['scenario'],variant=j['component_variant'],runtime_failure=bool(failure)),ensure_ascii=False),flush=True)
    results.sort(key=lambda x:(x['scenario'],x['component_variant']));write(out/'trials.json',results)
    stable=source_identity()==identity;endpoints=endpoint_checks(out,plan);write(out/'endpoint_execution_acceptance.json',dict(checks=endpoints,passed=all(x['all_physical_arrays_bit_exact'] for x in endpoints)))
    finish=dict(execution_complete=len(results)==8 and not failures and stable,trials=len(results),planned=8,runtime_failures=failures,
        sources_unchanged=stable,trials_sha256=sha(out/'trials.json'),manifest_sha256=sha(out/'manifest.json'),assignment_ledger_sha256=sha(out/'assignment_ledger.json'),
        endpoint_execution_acceptance_passed=all(x['all_physical_arrays_bit_exact'] for x in endpoints),
        endpoint_acceptance_sha256=sha(out/'endpoint_execution_acceptance.json'),wall_seconds=time.perf_counter()-began,ended_utc=datetime.now(timezone.utc).isoformat(),
        phase='development_identification',performance_claim_allowed=False)
    write(out/'execution_complete.json',finish);print(json.dumps(finish,ensure_ascii=False));return 0 if finish['execution_complete'] and finish['endpoint_execution_acceptance_passed'] else 1
if __name__=='__main__':raise SystemExit(main())
