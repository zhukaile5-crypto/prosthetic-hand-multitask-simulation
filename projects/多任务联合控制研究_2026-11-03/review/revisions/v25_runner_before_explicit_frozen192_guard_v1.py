"""Explicit, registered V25 capability/bridge/five-probe experiment runner.

Import does not construct or advance a model. Physical modes run only when
explicitly invoked by the sole experiment scheduler. There is no21-axis or
mixed expansion mode in this first protocol.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
import argparse, gzip, hashlib, json, time, traceback
from numbers import Real, Integral
import numpy as np
from .runtime import ResponseSession, source_identity, source_files, plain
from .design import SCHEMA, PROBE_ARMS, ORIGIN_TRIGGERS, ProbeSpec, TIMING_COLUMNS, DEBUG_TIMING_KEYS
from .forks import fork_session
from v22_joint_task_fusion.numerics import numerical_identity
from v13_a1.run_bridge import environment


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def fresh_output(value):
    destination=(ROOT/value).resolve();allowed=(ROOT/'output').resolve()
    if destination==allowed or not destination.is_relative_to(allowed):raise ValueError('Fresh strict ROOT/output descendant required')
    if destination.exists():raise FileExistsError('Retain old attempts; use a fresh output')
    return destination


def _debug_payload(value):
    data=json.loads(str(value));timing=data.get('solver_timing_ms')
    if timing is not None:
        if not isinstance(timing,dict):raise ValueError('debug timing dictionary required')
        for key in DEBUG_TIMING_KEYS:
            if key in timing:
                v=timing[key]
                if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0:raise ValueError('Invalid measured timing')
                timing[key]='__validated_measured_wall_time__'
    return json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=True).encode('utf-8')


def compare_trace(actual,original):
    """Frozen v3 measured timing whitelist; full Unicode payload, no padding test."""
    checks={}
    with np.load(actual,allow_pickle=False) as a,np.load(original,allow_pickle=False) as b:
        same=set(a.files)==set(b.files)
        for name in sorted(set(a.files)|set(b.files)):
            if name not in a or name not in b:checks[name]=False;continue
            x,y=a[name],b[name]
            if name=='metadata':continue
            if name in TIMING_COLUMNS:
                checks[name]=bool(x.shape==y.shape and x.dtype==y.dtype and x.dtype.kind in 'fi' and np.isfinite(x).all() and np.isfinite(y).all() and (x>=0).all() and (y>=0).all())
            elif name=='debug_json':
                checks[name]=bool(x.shape==y.shape and x.dtype.kind==y.dtype.kind=='U' and all(_debug_payload(u)==_debug_payload(v) for u,v in zip(x.flat,y.flat)))
            elif x.dtype.kind==y.dtype.kind=='U':
                checks[name]=bool(x.shape==y.shape and all(str(u).encode('utf-8')==str(v).encode('utf-8') for u,v in zip(x.flat,y.flat)))
            else:checks[name]=bool(x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes())
        full=bool(np.array_equal(a['time'],np.arange(3201)*.005) and np.array_equal(b['time'],np.arange(3201)*.005))
    return dict(passed=same and full and all(checks.values()),same_column_keys=same,full16=full,checks=checks,
        actual_trace_sha256=sha(actual),reference_trace_sha256=sha(original),measured_timing_columns=TIMING_COLUMNS,
        debug_timing_paths=['solver_timing_ms.'+k for k in DEBUG_TIMING_KEYS],provenance_only_column='metadata',
        no_algorithm_or_public_measurement_fields_removed=True,unicode_rule='fullUTF8payload_exact')


def _gzip_payload(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:yield json.dumps(json.loads(line),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')


def compare_public_logs(left,right):
    from itertools import zip_longest
    rows={}
    for name in ('response_observations.jsonl.gz','response_receipts.jsonl.gz','response_input_intervals.jsonl.gz'):
        count=0;passed=True
        for a,b in zip_longest(_gzip_payload(left/name),_gzip_payload(right/name)):
            count+=1
            if a is None or b is None or a!=b:passed=False
        rows[name]=dict(passed=passed,records=count)
    return dict(passed=all(r['passed'] for r in rows.values()),streams=rows)


def public_trigger(session,name):
    if name not in ORIGIN_TRIGGERS:raise ValueError('Unregistered public stage')
    frame=session.last_obs;graph=session.v16_supervisor
    if frame is None or graph is None or session.plant.joint_release_entry is None:return False
    if not session.running or not float(frame.time)<session.time-1e-9:return False
    p=graph.public_state
    if name=='receiver_transfer':return p.get('phase')=='TRANSFER' and p.get('receiver_fresh_streak',0)>=2
    if name=='unload_window':return p.get('phase')=='TRANSFER' and p.get('transfer',{}).get('continuous_support_evidence',{}).get('window_complete') is True
    r=frame.receiver_surface
    acq=r.get('acquisition_time_s');valid=r.get('valid') is True
    return bool(valid and acq is not None and acq>=session.plant.joint_release_entry.started_s-1e-9
        and np.asarray(frame.valid,bool).all() and np.any(np.asarray(frame.contact_present,bool))
        and r.get('normal_load_N') is not None and r.get('uncertainty6') is not None
        and r['normal_load_N']<=r['uncertainty6'][2])


def advance_to_origin(session,assignment):
    if 'origin_time_s' in assignment:
        raw=assignment['origin_time_s']
        if isinstance(raw,(bool,np.bool_)) or not isinstance(raw,Real) or not np.isfinite(raw):raise ValueError('Numeric finite origin clock required')
        t=float(raw)
        if not 0<=t<16 or abs(t/.005-round(t/.005))>1e-8:raise ValueError('Registered origin must be an existing PRE_OBSERVE tick before16')
        while session.running and session.time<t-1e-9:session.step_one_sensor_tick()
        if abs(session.time-t)>1e-9:raise RuntimeError('Registered origin not reached')
        return dict(available=True,time_s=session.time,trigger='registered_public_clock')
    trigger=assignment['origin_trigger']
    while session.running:
        if public_trigger(session,trigger):
            return dict(available=True,time_s=session.time,trigger=trigger,trigger_read_time_s=float(session.last_obs.time),
                phase=session.v16_supervisor.phase,next_PRE_OBSERVE_boundary=True)
        session.step_one_sensor_tick()
    return dict(available=False,reason='registered_public_trigger_not_exposed',trigger=trigger)


def construct(assignment):
    return ResponseSession(method=assignment['method'],scenario=assignment['scenario'],seed=assignment['seed'],
        split=assignment.get('split','robustness'),motion_mode=assignment['motion_mode'],
        task_policy='integrated',receiver_policy='coordinated',release_variant='canonical_joint')


def validate_assignment(assignment,selected):
    allowed={'mode','method','scenario','seed','motion_mode','split','origin_time_s','origin_trigger'}
    if set(assignment)-allowed or assignment['mode']!=selected:raise ValueError('Only the declared experiment assignment fields are allowed')
    seed=assignment['seed']
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,Integral) or not 610000<=seed<620000:raise ValueError('Explicit development physical seed610000..619999 required')
    if assignment.get('split','robustness') not in ('robustness','development'):raise ValueError('This experiment cannot consume confirmation data')
    if ('origin_time_s' in assignment)==('origin_trigger' in assignment):raise ValueError('Exactly one public-clock origin or public trigger required')
    if 'origin_trigger' in assignment and assignment['origin_trigger'] not in ORIGIN_TRIGGERS:raise ValueError('Unregistered public trigger')
    if 'origin_time_s' in assignment:
        t=assignment['origin_time_s']
        if isinstance(t,(bool,np.bool_)) or not isinstance(t,Real) or not np.isfinite(t) or not 0<=t<16 or abs(t/.005-round(t/.005))>1e-8:raise ValueError('Strict finite5ms public origin before16 required')


def save_full(session,path):
    while session.running:session.step_one_sensor_tick()
    session.save(path)
    if session.time!=16. or not session.completed:raise RuntimeError('Original full16 episode did not complete')
    from evaluate_runs import evaluate_trace
    from evaluation import evaluate_research
    result=evaluate_trace(path/'trace.npz');result['research_metrics']=evaluate_research(path/'trace.npz');write(path/'result.json',result)
    return result


def snapshot_sources(out,identity):
    rows={}
    for key,path in source_files().items():
        payload=path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=identity[key]:raise RuntimeError('Source changed before snapshot:'+key)
        destination=out/'sources'/key;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(payload)
        rows[key]=dict(original_path=str(path),snapshot_path=destination.relative_to(out).as_posix(),sha256=identity[key])
    write(out/'source_snapshot_manifest.json',dict(schema=SCHEMA,files=rows))


def capability_zero(out):
    import mujoco,hand_plant
    from unittest.mock import patch
    steps=0;forwards=0;original=mujoco.mj_forward
    def deny(*a,**kw):
        nonlocal steps
        steps+=1;raise RuntimeError('Physics forbidden during capability-zero')
    def forward(*a,**kw):
        nonlocal forwards
        forwards+=1;return original(*a,**kw)
    with patch.object(mujoco,'mj_step',deny),patch.object(mujoco,'mj_step1',deny),patch.object(mujoco,'mj_step2',deny),patch.object(hand_plant.HandPlant,'advance',deny),patch.object(mujoco,'mj_forward',forward):
        s=construct(dict(method='ratio_effort',scenario='nominal',seed=619741,motion_mode='hold'))
        passed=s.time==s.tick==s.sensors.samples==0 and not s.trace and s.response_observer is not None and callable(fork_session)
    return dict(passed=passed,physics_steps=steps,initialization_forward_calls=forwards,
        source_count=len(s.research_sources),production_source_sha256=s.research_sources,numerical_identity=s.numerical_sources,
        scope='Actual passive constructor and API capability only; fork reproduction and all remaining physical behaviour not tested')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--capability-zero',action='store_true');mode.add_argument('--bridge-fork',action='store_true');mode.add_argument('--probe-five',action='store_true')
    ap.add_argument('--out',required=True);ap.add_argument('--registration');args=ap.parse_args(argv)
    out=fresh_output(args.out);identity=source_identity();numeric=numerical_identity();env=environment()
    if len(identity)!=198:raise ValueError('Complete original192 plus all six V25 sources required')
    selected='capability-zero' if args.capability_zero else 'bridge-fork' if args.bridge_fork else 'probe-five'
    reg=None;reg_path=None
    if not args.capability_zero:
        if not args.registration:raise ValueError('Explicit source/environment/software/assignment registration required')
        reg_path=(ROOT/args.registration).resolve();reg=read(reg_path)
        if (reg['source_sha256']!=identity or reg['environment']!=env or reg['numerical_identity']!=numeric
            or reg.get('software_gate_passed') is not True or reg['assignment']['mode']!=selected):raise ValueError('Registered experiment changed')
        validate_assignment(reg['assignment'],selected)
        if selected=='probe-five':
            pin=reg['required_bridge_gate'];path=ROOT/pin['path'];bridge=read(path)
            if (sha(path)!=pin['sha256'] or bridge.get('schema')!='v25_full_remaining_fork_bridge_v1'
                or bridge.get('full_remaining_episode_compared') is not True or bridge.get('passed') is not True
                or bridge.get('source_sha256')!=identity or bridge.get('numerical_identity')!=numeric):raise ValueError('Actual full16 fork bridge of this exact198 required before probes')
    assigned_keys=('zero_capability',) if args.capability_zero else ('donor','clone') if args.bridge_fork else PROBE_ARMS
    out.mkdir(parents=True,exist_ok=False)
    try:snapshot_sources(out,identity)
    except Exception as exc:
        failure=dict(stage='source_snapshot',error=repr(exc),traceback=traceback.format_exc())
        write(out/'runtime_error.json',failure);write(out/'runtime_failures.json',[failure])
        write(out/'assignment_ledger.json',[dict(key=k,status='not_executed_due_to_source_snapshot_failure') for k in assigned_keys])
        write(out/'execution_complete.json',dict(schema=SCHEMA,execution_complete=False,passed=False,planned=len(assigned_keys),completed=0,runtime_failures=[failure],sources_unchanged=False))
        return 1
    write(out/'plan.json',dict(schema=SCHEMA,mode=selected,registration=None if reg_path is None else str(reg_path),
        registration_sha256=None if reg_path is None else sha(reg_path),source_sha256=identity,numerical_identity=numeric,environment=env,
        assignment=None if reg is None else reg['assignment'],all_expansion_requires_new_registration=True,performance_claim_allowed=False))
    write(out/'environment.json',dict(pinned_environment=env,numerical_identity=numeric,executable=sys.executable))
    write(out/'manifest.json',dict(schema=SCHEMA,source_sha256=identity,numerical_identity=numeric,
        plan_sha256=sha(out/'plan.json'),source_snapshot_manifest_sha256=sha(out/'source_snapshot_manifest.json'),
        original192_unchanged=True,performance_claim_allowed=False))
    rows=[];failures=[];began=time.perf_counter();passed=False;base=clone=None
    try:
        if args.capability_zero:
            result=capability_zero(out);write(out/'capability_zero.json',result);passed=result['passed'];rows.append(dict(key='zero_capability',status='assessed'))
        else:
            base=construct(reg['assignment']);origin=advance_to_origin(base,reg['assignment']);write(out/'origin_public.json',origin)
            if not origin['available']:
                rows=[dict(key=k,status='not_exposed',reason=origin['reason']) for k in (('donor','clone') if args.bridge_fork else PROBE_ARMS)]
                base.save(out/'unexposed_donor')
            elif args.bridge_fork:
                clone,fork_meta=fork_session(base);write(out/'opaque_fork_metadata.json',fork_meta)
                for name,s in (('donor',base),('clone',clone)):
                    result=save_full(s,out/name);rows.append(dict(key=name,status='completed_full16',result=result))
                fork_bridge=compare_trace(out/'clone/trace.npz',out/'donor/trace.npz');public=compare_public_logs(out/'clone',out/'donor')
                reference=reg['frozen_v24_reference'];reference_run=Path(reference['run'])
                if any(sha(reference_run/name)!=digest for name,digest in reference['input_sha256'].items()):raise ValueError('Frozen V24 reference changed')
                v24_bridge=compare_trace(out/'donor/trace.npz',reference_run/'trace.npz')
                bridge=dict(schema='v25_full_remaining_fork_bridge_v1',passed=fork_bridge['passed'] and public['passed'] and v24_bridge['passed'],source_sha256=identity,numerical_identity=numeric,
                    fork_trace_bridge=fork_bridge,clone_public_logs=public,frozen_V24_trace_bridge=v24_bridge,
                    opaque_state_never_exposed_to_controller=True,full_remaining_episode_compared=True)
                write(out/'bridge_complete.json',bridge);passed=bridge['passed']
            else:
                for arm in PROBE_ARMS:
                    branch=None
                    try:
                        branch,fork_meta=fork_session(base);branch.arm_probe(ProbeSpec(arm,origin['time_s']))
                        required_end=origin['time_s']+.2+float(branch.sensor_characteristics['tactile_delay_s'])+.005
                        end=min(16.,required_end)
                        while branch.running and branch.time<end-1e-9:branch.step_one_sensor_tick()
                        destination=out/'branches'/arm;branch.save(destination);write(destination/'opaque_fork_metadata.json',fork_meta)
                        probe=branch.probe_coordinator
                        rows.append(dict(key=arm,status='short_window_recorded' if branch.time>=required_end-1e-9 else 'truncated_at_original16s',terminal_probe_status=probe.terminal_status,
                            reference_origin_s=origin['time_s'],recorded_until_s=branch.time,actual_receipt_required=True,
                            requested_last_read_s=required_end,full_requested_window_recorded=branch.time>=required_end-1e-9,
                            exposed=any(r['active'] for r in probe.records),zero_control=arm=='zero',
                            original_full16_task_metrics_available=False,response_fit_or_model_admission=False))
                    except Exception as exc:
                        failure=dict(key=arm,error=repr(exc),traceback=traceback.format_exc());failures.append(failure);rows.append(dict(key=arm,status='runtime_failed'))
                        destination=out/'branches'/arm;destination.mkdir(parents=True,exist_ok=True);write(destination/'runtime_error.json',failure)
                        if branch is not None and branch.trace and not (destination/'trace.npz').exists():branch.save(destination)
                passed=not failures and len(rows)==5 and all(r['status']=='short_window_recorded' for r in rows)
    except Exception as exc:
        failures.append(dict(error=repr(exc),traceback=traceback.format_exc()));write(out/'runtime_error.json',failures[-1])
        for name,s in (('donor',base),('clone',clone)):
            if s is not None and s.trace and not (out/name/'trace.npz').exists():
                try:s.save(out/name)
                except Exception as save_error:failures.append(dict(stage='partial_save',key=name,error=repr(save_error)))
    seen={r['key'] for r in rows}
    rows.extend(dict(key=k,status='not_executed_due_to_runtime_failure') for k in assigned_keys if k not in seen)
    try:sources_unchanged=identity==source_identity()
    except Exception as exc:sources_unchanged=False;failures.append(dict(stage='final_source_check',error=repr(exc)))
    try:numeric_unchanged=numeric==numerical_identity() and env==environment()
    except Exception as exc:numeric_unchanged=False;failures.append(dict(stage='final_numerical_check',error=repr(exc)))
    planned=1 if args.capability_zero else 2 if args.bridge_fork else 5
    complete=dict(schema=SCHEMA,execution_complete=not failures and sources_unchanged and numeric_unchanged,
        passed=passed and not failures and sources_unchanged and numeric_unchanged,mode=selected,
        planned=planned,completed=sum(r['status'] in ('assessed','completed_full16','short_window_recorded') for r in rows),
        all_assigned_accounted=bool(rows) and len(rows)==planned,
        rows=rows,runtime_failures=failures,sources_unchanged=sources_unchanged,numerical_unchanged=numeric_unchanged,
        source_sha256=identity,numerical_identity=numeric,environment=env,wall_seconds=time.perf_counter()-began,
        physics_steps=0 if args.capability_zero else None,identification_response_validated=False,
        no_force_or_response_certificate=True)
    write(out/'assignment_ledger.json',rows);write(out/'runtime_failures.json',failures)
    complete.update(assignment_ledger_sha256=sha(out/'assignment_ledger.json'),plan_sha256=sha(out/'plan.json'),manifest_sha256=sha(out/'manifest.json'))
    write(out/'execution_complete.json',complete)
    print(json.dumps(dict(output=str(out),execution_complete=complete['execution_complete'],passed=complete['passed'],rows=len(rows),runtime_failures=len(failures)),ensure_ascii=False),flush=True)
    return 0 if complete['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
