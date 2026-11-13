"""Root-only fresh-output physical behavior bridge. Standard -m entry on Windows."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,json,hashlib,sys,time,shutil,traceback
import numpy as np
from .runtime import V13A1Session,source_files,source_identity,METHODS
from research_imports import ROOT,FOUNDATION
from session import jsonable,check_admission

PLAN=ROOT/'review/v13_a1_behavior_bridge_plan_v1.json'
LOCK=Path(__file__).with_name('source_lock.json')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(jsonable(value),f,ensure_ascii=False,indent=2,allow_nan=False)
def environment():
    import mujoco
    binaries={}
    for package in (mujoco,np):
        for pattern in ('*.pyd','*.dll'):
            for p in Path(package.__file__).parent.rglob(pattern):binaries[str(p)]=sha(p)
    return dict(python=sys.version,executable=sys.executable,executable_sha256=sha(sys.executable),mujoco=mujoco.__version__,numpy=np.__version__,native_binaries_sha256=dict(sorted(binaries.items())))
def original_jobs():
    plan=read(PLAN);native_reg=ROOT/plan['current_native80_registration']
    if sha(native_reg)!=plan['current_native80_registration_sha256']:raise ValueError('Predeclared parent registration changed')
    parent=read(native_reg);current=source_identity(METHODS)
    if any(current.get(k)!=v for k,v in parent['source_sha256'].items()):raise ValueError('Frozen native80 changed')
    available=[]
    for group in parent['groups']:
        folder=ROOT/group['output'];terminal=read(folder/'execution_complete.json')
        if not terminal['execution_complete']:raise ValueError('Original cohort not terminal complete')
        available.extend(read(folder/'manifest.json')['jobs'])
    jobs=[]
    for context in plan['physical_gate_contexts']:
        for method in plan['physical_gate_methods']:
            found=[j for j in available if j['method']==method and all(j[k]==v for k,v in context.items())]
            if len(found)!=1:raise ValueError('Original context missing or duplicated')
            job=found[0];folder=Path(job['destination']);meta=read(folder/'metadata.json')
            if not meta['completed'] or sha(folder/'trace.npz')!=meta['trace_sha256']:raise ValueError('Original complete trace identity mismatch')
            jobs.append(dict(method=method,**context,split=job['split'],original_trace=str(folder/'trace.npz'),original_trace_sha256=meta['trace_sha256']))
    if len(jobs)!=9:raise ValueError('Exactly the predeclared9 assignments required')
    return jobs

def require_lock():
    lock=read(LOCK)
    if lock['source_sha256']!=source_identity(METHODS) or lock['environment']!=environment() or lock['plan_sha256']!=sha(PLAN):raise ValueError('A1 source/environment/plan changed after pin')
    if lock['jobs']!=original_jobs():raise ValueError('Original bridge inputs changed')
    if sha(lock['software_gate'])!=lock['software_gate_sha256']:raise ValueError('Software gate changed after pin')
    return lock

def compare_trace(actual_path,original_path):
    # The two explicit exclusions are provenance and measured wall time only.
    # Every other physical/public/decision/force column must be byte-identical.
    fields={};excluded=('metadata','control_ms')
    with np.load(actual_path,allow_pickle=False) as a,np.load(original_path,allow_pickle=False) as b:
        same_keys=set(a.files)==set(b.files)
        for name in sorted(set(a.files)|set(b.files)):
            if name in excluded:continue
            if name not in a.files or name not in b.files:
                fields[name]=dict(exact=False,missing=True);continue
            x,y=a[name],b[name];equal=x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()
            detail=dict(exact=equal,actual_shape=list(x.shape),original_shape=list(y.shape),actual_dtype=str(x.dtype),original_dtype=str(y.dtype))
            if not equal and x.shape==y.shape and x.dtype.kind in 'fiu' and y.dtype.kind in 'fiu':
                finite=np.isfinite(x)&np.isfinite(y);detail['max_finite_abs_difference']=float(np.max(np.abs(x[finite]-y[finite]))) if finite.any() else None
            fields[name]=detail
    return dict(passed=same_keys and bool(fields) and all(v['exact'] for v in fields.values()),same_column_keys=same_keys,excluded_columns=list(excluded),fields=fields,scope='All complete physical, common sensor and decision columns; provenance and walltime excluded explicitly')

def worker(job,expected_sources):
    target=Path(job['destination']);session=None;began=time.perf_counter()
    try:
        if source_identity(METHODS)!=expected_sources:raise ValueError('Worker source mismatch before construction')
        session=V13A1Session(method=job['method'],scenario=job['scenario'],seed=job['seed'],split=job['split'],motion_mode=job['motion_mode'])
        while session.running:session.step_one_sensor_tick()
        session.save(target)
        comparison=compare_trace(target/'trace.npz',job['original_trace']);write(target/'complete_native_equivalence.json',comparison)
        same=source_identity(METHODS)==expected_sources and sha(job['original_trace'])==job['original_trace_sha256']
        result=dict(job=job,complete=True,all_trace_columns_exact=comparison['passed'],sources_and_input_unchanged=same,wall_seconds=time.perf_counter()-began,trace_sha256=sha(target/'trace.npz'),physical_steps=len(session.plant.substep_records),actor_write_count=session.plant.actor21.write_count)
    except Exception as exc:
        if session is not None and session.trace and not target.exists():
            session.running=False;session.completed=False;session.failure_reason='v13_a1_bridge_runtime_failure';session.save(target)
        target.mkdir(parents=True,exist_ok=True)
        result=dict(job=job,complete=False,error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-began)
    write(target/'result.json',result);return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pin-sources',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--software-gate');parser.add_argument('--out');parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    if args.pin_sources:
        if not args.software_gate:parser.error('--software-gate required')
        gate=Path(args.software_gate).resolve();proof=read(gate);identity=source_identity(METHODS)
        if not proof['passed'] or proof['source_sha256']!=identity or not proof['physical_not_started']:raise ValueError('Software gate failed or stale')
        value=dict(schema='v13_a1_behavior_bridge_source_lock_v1',created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=identity,environment=environment(),plan_sha256=sha(PLAN),jobs=original_jobs(),software_gate=str(gate),software_gate_sha256=sha(gate))
        write(LOCK,value);print(json.dumps(dict(pinned=len(identity),native_binaries=len(value['environment']['native_binaries_sha256']),sha256=sha(LOCK))));return 0
    if not args.out or not 1<=args.workers<=16:parser.error('fresh --out and1..16 workers required')
    admission=check_admission()
    if not admission['passed']:raise ValueError(admission['problems'])
    lock=require_lock();out=(ROOT/args.out).resolve();allowed=(ROOT/'output').resolve()
    if not out.is_relative_to(allowed) or out==allowed:raise ValueError('Strict output descendant required')
    out.mkdir(parents=True,exist_ok=False)
    for name,p in source_files(METHODS).items():
        target=out/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        if sha(target)!=lock['source_sha256'][name]:raise ValueError('Snapshot mismatch')
    write(out/'source_lock.json',lock)
    jobs=[]
    for j in lock['jobs']:
        jobs.append(dict(**j,destination=str(out/'runs'/f"{j['scenario']}__{j['method']}__{j['seed']}__{j['motion_mode']}")))
    write(out/'manifest.json',dict(schema='v13_a1_full_native_behavior_bridge_jobs_v1',jobs=jobs,planned=9,source_lock_sha256=sha(LOCK),research_source_sha256=lock['source_sha256'],performance_claim_allowed=False))
    began=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(worker,j,lock['source_sha256']):j for j in jobs}
        for future in as_completed(futures):
            try:r=future.result()
            except Exception as exc:r=dict(job=futures[future],complete=False,error=repr(exc),traceback=traceback.format_exc())
            results.append(r);write(out/f'completion_{len(results):02d}.json',r)
            print(json.dumps(dict(completed=len(results),planned=9,method=r['job']['method'],scenario=r['job']['scenario'],complete=r['complete'],exact=r.get('all_trace_columns_exact')),ensure_ascii=False),flush=True)
    results.sort(key=lambda r:(r['job']['scenario'],r['job']['seed'],r['job']['method']))
    same=source_identity(METHODS)==lock['source_sha256']
    complete=len(results)==9 and all(r['complete'] for r in results)
    value=dict(schema='v13_a1_full_behavior_bridge_terminal_v1',execution_complete=complete,all9_exact=complete and all(r['all_trace_columns_exact'] for r in results),sources_unchanged=same,source_lock_sha256=sha(LOCK),results=results,wall_seconds=time.perf_counter()-began,ended_utc=datetime.now(timezone.utc).isoformat(),performance_claim_allowed=False)
    write(out/'execution_complete.json',value);print(json.dumps({k:v for k,v in value.items() if k!='results'},ensure_ascii=False))
    return 0 if value['execution_complete'] and value['all9_exact'] and same else 1
if __name__=='__main__':sys.exit(main())
