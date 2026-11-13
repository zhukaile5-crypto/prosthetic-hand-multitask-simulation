"""Isolated six-run release basis pilot. Physics requires explicit --execute.

Default --preflight and --pin-sources do not call advance/mj_step.  The author
does not execute the physical pilot; root owns scheduling after source freeze.
"""
from pathlib import Path
from dataclasses import is_dataclass,fields
from collections import deque
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,base64,enum,hashlib,json,os,sys,time,traceback
import numpy as np
PILOT=Path(__file__).resolve().parent;ROOT=PILOT.parents[1]
sys.path.insert(0,str(PILOT));sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from decoder import ReleasePath2D,make_plant_class,COMMAND_AXES,PATHS
from source_pinning import identity,require_lock,snapshot,environment,sha


def freeze(value,stack=None):
    """Canonical full Python state, with actual RNG/queued arrays, no repr()."""
    stack=set() if stack is None else stack
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,(float,np.floating)):
        return {'float_hex':float(value).hex()}
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,(bytes,bytearray)):return dict(bytes_b64=base64.b64encode(bytes(value)).decode('ascii'))
    if isinstance(value,np.dtype):return dict(numpy_dtype=value.str)
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject:raise TypeError('Object arrays require an explicit snapshot encoder')
        a=np.ascontiguousarray(value)
        return dict(array_dtype=a.dtype.str,shape=list(a.shape),bytes_b64=base64.b64encode(a.tobytes()).decode('ascii'))
    if isinstance(value,np.random.Generator):return dict(numpy_generator=type(value.bit_generator).__name__,state=freeze(value.bit_generator.state,stack))
    if isinstance(value,np.random.RandomState):return dict(numpy_random_state=freeze(value.get_state(),stack))
    if isinstance(value,enum.Enum):return dict(enum=type(value).__qualname__,value=freeze(value.value,stack))
    if isinstance(value,Path):return dict(path=str(value))
    objid=id(value)
    if objid in stack:raise TypeError('Cyclic mutable state needs an explicit snapshot adapter')
    active=stack|{objid}
    if isinstance(value,dict):return {str(k):freeze(v,active) for k,v in sorted(value.items(),key=lambda x:str(x[0]))}
    if isinstance(value,(list,tuple,deque)):return dict(sequence_type=type(value).__name__,items=[freeze(v,active) for v in value])
    if isinstance(value,(set,frozenset)):
        items=[freeze(v,active) for v in value]
        return dict(set_type=type(value).__name__,items=sorted(items,key=lambda v:json.dumps(v,sort_keys=True)))
    if is_dataclass(value):return dict(class_name=type(value).__module__+'.'+type(value).__qualname__,fields={f.name:freeze(getattr(value,f.name),active) for f in fields(value)})
    if hasattr(value,'__dict__'):return dict(class_name=type(value).__module__+'.'+type(value).__qualname__,state=freeze(value.__dict__,active))
    raise TypeError('Unregistered exact state type: '+type(value).__module__+'.'+type(value).__qualname__)


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()
def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def make_session(shape,path):
    # Retained baseline construction/config/randomization. This decoder does
    # not inspect object state or infer a force target from physical labels.
    from runtime import ResearchSession
    from hand_plant import CommonPreparation
    from skin_geometry import DelayedSkinGeometry
    session=ResearchSession(method='ratio',scenario='nominal',shape=shape,seed=610120,split='development')
    c=session.config
    if c.shape!=shape or c.mass!=.12 or c.friction!=.6 or c.timestep!=.0005 or c.control_dt!=.005:
        raise ValueError('Actual standard physical config differs from the registered pilot')
    if session.release_driver is not None:raise ValueError('Native baseline release required')
    session.plant=make_plant_class()(c,path)
    session.preparation=CommonPreparation(session.plant);session.last_truth=session.plant.truth()
    session.skin_geometry=DelayedSkinGeometry(session.plant)
    return session


def entry_snapshot(s,target):
    import mujoco
    p=s.plant;data=p.data
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(p.model,spec));mujoco.mj_getState(p.model,data,state,spec)
    arrays=dict(integration_state=state)
    for name in ('qpos','qvel','act','ctrl','qacc_warmstart','qacc','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat','eq_active','userdata','plugin_state'):
        if hasattr(data,name):arrays[name]=np.array(getattr(data,name),copy=True)
    np.savez_compressed(target/'entry_physics.npz',**arrays)
    # These explicit exclusions are immutable robot/model links, captured by
    # full integration state and pinned robot assets, NOT omitted mutable RNG.
    def independent(obj):
        return {k:v for k,v in obj.__dict__.items() if k not in ('plant','model','data','basis_path','basis_decoder','basis_rows')}
    side=dict(controller=s.controller,sensors=s.sensors,preparation=independent(s.preparation),
        skin_geometry=independent(s.skin_geometry),
        session_clock={k:getattr(s,k) for k in ('running','last_command','release_at','handoff_passed','manual_release',
            'last_action','last_desired','last_prediction','reference_relative','local_lower','local_upper','release_supported') if hasattr(s,k)},
        plant_side={k:getattr(p,k) for k in ('control_steps','alpha','phase','lift_ref','wrist_ref','release_started','release_thumb','release_lift','release_alpha','release_ctrl','env','max_hidden_clip')})
    encoded=freeze(side);write_json(target/'entry_side_state.json',encoded)
    # Retain full state above. Only explicitly noncausal wall-clock diagnostic
    # keys may be ignored in equality, never simulation/observation clocks.
    ignored=('qp_seconds','control_ms','wall_seconds')
    def causal(value):
        if isinstance(value,dict):return {k:causal(v) for k,v in value.items() if k not in ignored}
        if isinstance(value,list):return [causal(v) for v in value]
        return value
    signature=dict(time=float(p.time),physical_time=float(data.time),integration_state_spec=int(spec),
        integration_state_sha256=digest(freeze(state)),physics_arrays_sha256=digest(freeze(arrays)),
        python_causal_side_state_sha256=digest(causal(encoded)),
        ignored_noncausal_wallclock_keys=list(ignored),
        excluded_immutable_links=['plant/model/data pointers; actual state and pinned model identity are stored separately'])
    write_json(target/'entry_signature.json',signature)
    return signature


def prefix_digest(trace_path):
    keys=('time','qpos','qvel','alpha_before','ctrl_before','normal','skin_normal','observation_json','validity','observed_normal')
    with np.load(trace_path,allow_pickle=False) as archive:
        data={k:archive[k] for k in keys}
    selected=data['time']<12.-1e-9
    return {k:digest(freeze(value[selected])) for k,value in data.items()}


def save_decoder(s,target):
    p=s.plant;rows=np.array(p.basis_rows,float)
    width=3+42+2*p.model.nu
    if not len(rows):rows=np.empty((0,width))
    np.savez_compressed(target/'decoder_commands.npz',control_time=rows[:,0],physical_time=rows[:,1],
        path_time=rows[:,2],actual_command21=rows[:,3:24],native_command21=rows[:,24:45],
        actual_ctrl=rows[:,45:45+p.model.nu],native_ctrl=rows[:,45+p.model.nu:45+2*p.model.nu],command_axes=np.asarray(COMMAND_AXES))
    report=dict(applications=p.basis_apply_count,mismatches=p.basis_mismatch_count,first_mismatch=p.basis_first_mismatch,
        path=p.basis_path,decoder_exactness_scope='native2d fullctrl/alpha; fixed_height identical18finger+2wrist/alpha with lift-only difference',
        no_release_response_model=True,force_safety_certified=False)
    write_json(target/'decoder_validation.json',report)
    return report


def probe(job):
    target=Path(job['directory']);target.mkdir(parents=True,exist_ok=False)
    s=None;entry=None;began=time.perf_counter()
    try:
        require_lock();s=make_session(job['shape'],job['path'])
        write_json(target/'job.json',job)
        while s.running:
            if abs(s.plant.time-12.)<1e-9 and entry is None:
                if s.plant.release_started is not None:raise ValueError('Entry must precede the first release command')
                entry=entry_snapshot(s,target)
            s.step_one_sensor_tick()
        if entry is None or abs(s.plant.time-16.)>1e-8:raise ValueError('Full16s and exact12s entry required')
        s.save(target)
        from evaluate_runs import evaluate_trace
        from evaluation import evaluate_research
        from session import jsonable
        result=evaluate_trace(target/'trace.npz');result['research_metrics']=evaluate_research(target/'trace.npz')
        result.update(pilot_path=job['path'],pilot_role='decoder/lift_direction_development_only',wall_seconds=time.perf_counter()-began)
        write_json(target/'result.json',jsonable(result))
        decoder=save_decoder(s,target);prefix=prefix_digest(target/'trace.npz')
        write_json(target/'prefix_signature.json',prefix)
        pilot_meta=dict(environment=environment(),source_lock_sha256=job['lock_sha256'],source_manifest='../../manifest.json',
            actual_config=jsonable(s.config.to_dict() if hasattr(s.config,'to_dict') else vars(s.config)),entry=entry,
            public_decoder_inputs='actual release-entry robot alpha/thumbctrl/lift/wrist and real actuator bounds only; no object truth',
            command_record_sha256=sha(target/'decoder_commands.npz'),no_response_model=True)
        write_json(target/'pilot_metadata.json',pilot_meta)
        return dict(job_id=job['job_id'],shape=job['shape'],path=job['path'],status='complete',entry=entry,prefix=prefix,
            decoder=decoder,trace_sha256=sha(target/'trace.npz'),task_success=result['task_success'],
            guard_pass=result['safety']['force_protection_pass'],finger_peak=result['safety']['finger_peak'],
            elapsed_s=time.perf_counter()-began)
    except Exception as exc:
        failure=dict(job_id=job['job_id'],shape=job['shape'],path=job['path'],status='failed',error=repr(exc),
            traceback=traceback.format_exc(),elapsed_s=time.perf_counter()-began,entry_captured=entry is not None)
        if s is not None:
            try:s.save(target);failure['partial_trace_retained']=(target/'trace.npz').is_file()
            except Exception as sub:failure['partial_save_error']=repr(sub)
            try:save_decoder(s,target)
            except Exception as sub:failure['decoder_save_error']=repr(sub)
        write_json(target/'failure.json',failure)
        return failure


def preflight():
    reports=[]
    for shape in ('capsule','box','sphere'):
        for path in PATHS:
            s=make_session(shape,path);p=s.plant
            p.set_closure(np.full(5,.75));p.set_motion(.1,[0.,0.]);p.begin_release()
            # Re-evaluate command formulas only; no dynamic integration.
            for tick in (0,1,19,20,55,56,64,65,119,120,159,160,279,280,439,440,700,800):
                p.control_steps=tick;p._apply_ctrl()
            if p.data.time!=0. or p.basis_mismatch_count:raise AssertionError('Zero-step decoder preflight failed')
            reports.append(dict(shape=shape,path=path,applications=p.basis_apply_count,physical_time=float(p.data.time),exact=True))
    return dict(passed=True,scope='zero dynamics; actual standard robot/model construction and decoder formulas only',reports=reports)


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pin-sources',action='store_true');mode.add_argument('--preflight',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--out');parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.pin_sources:
        lock=dict(schema='release_basis_source_lock_v1',sources=identity(),environment=environment(),physics_started=False)
        write_json(PILOT/'dependency_lock.json',lock);print(json.dumps(dict(pinned=len(lock['sources']),path=str(PILOT/'dependency_lock.json'))));return
    if args.preflight:
        report=preflight();print(json.dumps(report,ensure_ascii=False));return
    if not args.out:parser.error('--execute requires fresh --out')
    target=(ROOT/args.out).resolve()
    if not target.is_relative_to(ROOT.resolve()) or target.exists():raise ValueError('Fresh output within research root required')
    if args.workers<1 or args.workers>6:raise ValueError('One to six independent workers')
    lock=require_lock();target.mkdir(parents=True);snapshot(target/'sources',lock)
    jobs=[dict(job_id=f'{shape}__{path}',shape=shape,path=path,directory=str(target/'runs'/f'{shape}__{path}'),
        lock_sha256=sha(PILOT/'dependency_lock.json')) for shape in ('capsule','box','sphere') for path in PATHS]
    write_json(target/'manifest.json',dict(schema='release_basis_six_run_manifest_v1',plan=json.loads((PILOT/'plan.json').read_text(encoding='utf-8')),
        sources=lock['sources'],environment=environment(),jobs=jobs,physics_started=True,performance_claim_allowed=False))
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    results=[];began=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        submitted={pool.submit(probe,job):job for job in jobs}
        for future in as_completed(submitted):
            job=submitted[future]
            try:r=future.result()
            except Exception as exc:r=dict(job_id=job['job_id'],shape=job['shape'],path=job['path'],status='worker_exception',error=repr(exc),traceback=traceback.format_exc())
            results.append(r);print(json.dumps(r,ensure_ascii=False),flush=True)
    pairs=[]
    for shape in ('capsule','box','sphere'):
        a=next(x for x in results if x['shape']==shape and x['path']=='native2d');b=next(x for x in results if x['shape']==shape and x['path']=='fixed_height')
        complete=a['status']==b['status']=='complete'
        equal=bool(complete and a['entry']==b['entry'] and a['prefix']==b['prefix'])
        pairs.append(dict(shape=shape,both_complete=complete,full_entry_and_prefix_exact=equal,
            causal_lift_comparison_allowed=equal,reason=None if equal else 'No causal direction inference: incomplete or unequal full entry/prefix'))
    unchanged=identity()==lock['sources']
    final=dict(execution_complete=len(results)==6,completed=sum(x['status']=='complete' for x in results),planned=6,
        failures=[x for x in results if x['status']!='complete'],results=results,pairs=pairs,sources_unchanged=unchanged,
        passed_decoder_and_pairing=bool(unchanged and all(x['full_entry_and_prefix_exact'] for x in pairs) and all(x.get('decoder',{}).get('mismatches',1)==0 for x in results)),
        wall_seconds=time.perf_counter()-began,performance_claim_allowed=False,release_model_admitted=False)
    write_json(target/'summary.json',final)
    if not final['passed_decoder_and_pairing']:raise SystemExit(2)


if __name__=='__main__':main()
