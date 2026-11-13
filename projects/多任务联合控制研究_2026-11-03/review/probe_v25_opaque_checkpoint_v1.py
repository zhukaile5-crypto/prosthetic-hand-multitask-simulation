"""Experiment-only clone capability; never a controller input or physics claim."""
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import argparse, copy, hashlib, json, sys, traceback, types
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def walk(value,path=(),seen=None):
    seen=set() if seen is None else seen
    yield path,value
    if id(value) in seen:return
    seen.add(id(value))
    if isinstance(value,dict):items=list(value.items())
    elif isinstance(value,(tuple,list)):items=list(enumerate(value))
    elif isinstance(value,np.ndarray) or isinstance(value,(str,bytes,int,float,bool,type(None),type,types.ModuleType,types.FunctionType,types.MethodType)):return
    else:
        try:items=list(vars(value).items())
        except TypeError:return
    for key,child in items:
        if isinstance(key,(str,int)):
            yield from walk(child,path+(key,),seen)

def resolve(value,path):
    for key in path:
        value=value[key] if isinstance(value,(dict,list,tuple)) else vars(value)[key]
    return value

def clone_with_delegation_memo(value):
    """Preserve normal deepcopy; seed only delegated attribute wrapper shells.

    Delegated __getattr__ may recurse while generic deepcopy probes an empty
    instance. The protocol is review-only and writes no production method.
    """
    memo={};wrappers=[]
    for path,obj in walk(value):
        cls=type(obj)
        if isinstance(obj,types.ModuleType):
            if not (obj.__name__=='mujoco' or obj.__name__.startswith('mujoco.') or obj.__name__=='numpy' or obj.__name__.startswith('numpy.')):
                raise ValueError('Unreviewed stored API module '+obj.__name__)
            memo[id(obj)]=obj
        if cls.__module__ in ('v17_contact_transfer.motion_domain_sensors','v10.controller','v9.controller') and '__getattr__' in cls.__dict__ and id(obj) not in memo:
            new=object.__new__(cls);memo[id(obj)]=new;wrappers.append((obj,new,path))
    for old,new,_ in wrappers:vars(new).update(copy.deepcopy(vars(old),memo))
    return copy.deepcopy(value,memo),[list(path) for _,_,path in wrappers]

def main(out):
    if out.exists():raise FileExistsError(out)
    from v24_canonical_task_fusion.runtime import ReleaseSession,source_identity
    from v22_joint_task_fusion.numerics import numerical_identity
    import mujoco
    parent=json.loads((ROOT/'review/v24_small_plan_v1.json').read_text(encoding='utf-8'))
    pins=source_identity(['ratio_effort']);numeric=numerical_identity()
    assert pins==parent['source_sha256'] and numeric==parent['numerical_identity']
    calls={n:0 for n in ('mj_step','mj_step1','mj_step2')};checks=[];issues=[];attempts=[]
    def ck(name,passed,detail=None):
        checks.append(dict(name=name,passed=bool(passed),detail=detail))
        if not passed:issues.append(name)
    def forbidden(name):
        def reject(*a,**kw):calls[name]+=1;raise RuntimeError('No dynamics in clone capability gate '+name)
        return reject
    s=ReleaseSession(method='ratio_effort',scenario='nominal',seed=619741,split='robustness',motion_mode='transport',task_policy='integrated',receiver_policy='coordinated',release_variant='canonical_joint')
    with ExitStack() as stack:
        # Construct with real native function handles. Installing mocks first
        # can cause the native servo's saved API handles to capture mocks.
        for name in calls:stack.enter_context(patch.object(mujoco,name,forbidden(name)))
        try:
            cloned=copy.deepcopy(s);route='standard_deepcopy';seeded=[]
            attempts.append(dict(route=route,available=True))
        except Exception as exc:
            attempts.append(dict(route='standard_deepcopy',available=False,type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
            cloned,seeded=clone_with_delegation_memo(s);route='delegation_shell_memo_deepcopy'
            attempts.append(dict(route=route,available=True,seeded_wrapper_paths=seeded))
        ck('zero_no_acquisition_or_ticks',s.time==cloned.time==0. and s.tick==cloned.tick==0 and s.sensors.samples==cloned.sensors.samples==0)
        ck('plant_model_data_independent',s.plant is not cloned.plant and s.plant.model is not cloned.plant.model and s.plant.data is not cloned.plant.data)
        sig=mujoco.mjtState.mjSTATE_INTEGRATION
        def state(session):
            model=session.plant.model;v=np.empty(mujoco.mj_stateSize(model,sig));mujoco.mj_getState(model,session.plant.data,v,sig);return v
        ck('opaque_complete_integration_state_exact',state(s).tobytes()==state(cloned).tobytes())
        model_arrays=('qpos0','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','body_mass','geom_friction')
        for name in model_arrays:
            x=getattr(s.plant.model,name);y=getattr(cloned.plant.model,name)
            ck('model_'+name+'_exact_independent',x.tobytes()==y.tobytes() and not np.shares_memory(x,y))
        original_nodes=list(walk(s));groups={}
        for path,obj in original_nodes:
            if isinstance(obj,(mujoco.MjModel,mujoco.MjData)):
                groups.setdefault(id(obj),[]).append(path)
        for i,paths in enumerate(groups.values()):
            replacements=[resolve(cloned,p) for p in paths]
            ck('native_alias_group_'+str(i),len({id(v) for v in replacements})==1 and replacements[0] is not resolve(s,paths[0]),[list(p) for p in paths])
        for path,obj in original_nodes:
            if not isinstance(obj,np.ndarray):continue
            for field in ('qpos','qvel','ctrl'):
                if np.shares_memory(obj,getattr(s.plant.data,field)):
                    ck('cached_data_view_'+str(path),np.shares_memory(resolve(cloned,path),getattr(cloned.plant.data,field)),dict(path=list(path),field=field))
        from v13_a1.codec import Reference21
        before=s.plant.data.ctrl.copy();before_count=s.plant.actor21.write_count
        ref=cloned.plant.reference_snapshot()
        cloned.plant.actor21.apply(cloned.plant.data,Reference21(ref.values,ref.time_s))
        ck('sole_actor_clone_does_not_write_original',before.tobytes()==s.plant.data.ctrl.tobytes() and before_count==s.plant.actor21.write_count)
        ck('source192_and_numerical_identity_preserved',source_identity(['ratio_effort'])==pins and numerical_identity()==numeric)
        ck('no_dynamics_calls',all(v==0 for v in calls.values()))
    report=dict(schema='v25_opaque_checkpoint_capability_v1',passed=not issues,checks=checks,issues=issues,
        attempts=attempts,clone_route=route,source_sha256=pins,numerical_identity=numeric,
        checker_sha256=sha(__file__),physics_steps=0,dynamics_guard_calls=calls,
        scope='Zero-step actual constructor/copy capability only; not full16 behaviour bridge, future response, model calibration or controller access to physics state')
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(out=str(out),passed=report['passed'],checks=len(checks),issues=issues,route=route),ensure_ascii=False))
    return 0 if report['passed'] else 1

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();raise SystemExit(main(ROOT/args.out))
