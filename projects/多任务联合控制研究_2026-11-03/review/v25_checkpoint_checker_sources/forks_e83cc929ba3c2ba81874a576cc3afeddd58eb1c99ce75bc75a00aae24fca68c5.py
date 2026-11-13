"""Opaque experiment checkpoint copies. Never imported by a decision policy.

The experiment coordinator may fork a PRE_OBSERVE session. Physics state is
copied internally and represented externally only by an equality fingerprint;
it must not be a public observation, regression feature or control input.
All model/data, devices, RNGs, observer and controller state are independent.
Only pinned native API modules/functions are shared as immutable code handles.
"""
from pathlib import Path
from collections import defaultdict, deque
import copy, hashlib, json, types
import numpy as np
import mujoco

DELEGATED_WRAPPERS=('v17_contact_transfer.motion_domain_sensors','v10.controller','v9.controller')

def _walk(value,path=(),seen=None):
    seen=set() if seen is None else seen
    yield path,value
    if id(value) in seen:return
    seen.add(id(value))
    if isinstance(value,dict):items=list(value.items())
    elif isinstance(value,(list,tuple,deque)):items=list(enumerate(value))
    elif isinstance(value,(np.ndarray,str,bytes,int,float,bool,type(None),type,types.ModuleType,types.FunctionType,types.MethodType)):return
    else:
        try:items=list(vars(value).items())
        except TypeError:return
    for key,child in items:
        if isinstance(key,(int,str)):yield from _walk(child,path+(key,),seen)

def _resolve(value,path):
    for key in path:value=value[key] if isinstance(value,(dict,list,tuple,deque)) else vars(value)[key]
    return value

def _boundary(session):
    now=float(session.time)
    if not np.isfinite(now) or now<0 or abs(now-float(session.plant.time))>1e-10:
        raise ValueError('session and physics boundary clocks differ')
    if abs(now-int(session.tick)*.005)>1e-10 or getattr(session.plant,'_delivery_inside_advance',False):
        raise ValueError('PRE_OBSERVE complete5ms boundary required')
    last=getattr(session.sensors,'_last_time',None)
    if (now==0. and last is not None) or (now>0. and (last is None or abs(float(last)-(now-.005))>1e-9)):
        raise ValueError('fork must precede this tick sensor acquisition')
    if abs(float(session.plant.model.opt.timestep)-.0005)>1e-15:
        raise ValueError('frozen500us physics timestep required')
    return now

def _integration_digest(session):
    signature=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(session.plant.model,signature))
    mujoco.mj_getState(session.plant.model,session.plant.data,state,signature)
    return hashlib.sha256(state.tobytes()).hexdigest()

def _copy(session):
    nodes=list(_walk(session));memo={};wrappers=[];modules=[]
    for path,obj in nodes:
        if isinstance(obj,types.ModuleType):
            name=obj.__name__
            if not (name=='mujoco' or name.startswith('mujoco.') or name=='numpy' or name.startswith('numpy.')):
                raise ValueError('Unreviewed stored API module: '+name)
            memo[id(obj)]=obj;modules.append(name)
        cls=type(obj)
        if cls.__module__ in DELEGATED_WRAPPERS and '__getattr__' in cls.__dict__ and id(obj) not in memo:
            shell=object.__new__(cls);memo[id(obj)]=shell;wrappers.append((obj,shell,path))
    for old,new,_ in wrappers:vars(new).update(copy.deepcopy(vars(old),memo))
    cloned=copy.deepcopy(session,memo)
    # deepcopy does not preserve NumPy readonly flags. Preserve declarations,
    # including immutable public packets, without changing any numeric value.
    for path,obj in nodes:
        if isinstance(obj,np.ndarray) and not obj.flags.writeable:
            _resolve(cloned,path).setflags(write=False)
    return cloned,nodes,dict(shared_api_modules=sorted(set(modules)),
        delegated_wrapper_shell_paths=[list(path) for _,_,path in wrappers])

def fork_session(session):
    """Return an isolated session and evidence at the SAME pre-observe tick.

    This function makes no sensor call, action, forward, integration or model
    mutation. A successful copy is not yet a behavioural bridge: every new
    runtime must separately replay an unmodified complete remaining episode.
    """
    now=_boundary(session);before=_integration_digest(session)
    cloned,nodes,metadata=_copy(session);checks=[]
    def check(name,passed,detail=None):
        checks.append(dict(name=name,passed=bool(passed),detail=detail))
        if not passed:raise RuntimeError('opaque checkpoint independence failed: '+name)
    check('pre_observe_clock_preserved',_boundary(cloned)==now)
    check('opaque_integration_exact_and_original_unchanged',before==_integration_digest(cloned)==_integration_digest(session))
    check('plant_independent',cloned.plant is not session.plant)
    groups=defaultdict(list)
    for path,obj in nodes:
        if isinstance(obj,(mujoco.MjModel,mujoco.MjData,np.random.Generator,np.random.RandomState)):
            groups[id(obj)].append(path)
    for i,paths in enumerate(groups.values()):
        old=_resolve(session,paths[0]);values=[_resolve(cloned,path) for path in paths]
        check('native_or_rng_alias_group_'+str(i),len({id(x) for x in values})==1 and values[0] is not old,
            dict(type=type(old).__name__,paths=[list(path) for path in paths]))
        if isinstance(old,np.random.Generator):
            check('generator_state_'+str(i),json.dumps(old.bit_generator.state,sort_keys=True)==json.dumps(values[0].bit_generator.state,sort_keys=True))
        if isinstance(old,np.random.RandomState):
            left=old.get_state();right=values[0].get_state()
            check('randomstate_state_'+str(i),left[0]==right[0] and np.array_equal(left[1],right[1]) and left[2:]==right[2:])
    callbacks=[]
    for path,obj in nodes:
        if isinstance(obj,types.MethodType):
            other=_resolve(cloned,path)
            callbacks.append(dict(path=list(path),independent_owner=other.__self__ is not obj.__self__,same_function=other.__func__ is obj.__func__))
    check('bound_python_callbacks_point_to_cloned_owners',all(row['independent_owner'] and row['same_function'] for row in callbacks),callbacks)
    # No cached Python ndarray may accidentally point into the parent's data.
    # If it was a view of the original live data, it must still be a view of
    # the clone's corresponding data rather than an unrelated copied array.
    array_count=0;view_count=0;alias_errors=[]
    for path,obj in nodes:
        if not isinstance(obj,np.ndarray):continue
        other=_resolve(cloned,path);array_count+=1
        for field in ('qpos','qvel','ctrl'):
            oldbuf=getattr(session.plant.data,field);newbuf=getattr(cloned.plant.data,field)
            if np.shares_memory(other,oldbuf):alias_errors.append(dict(path=list(path),field=field,issue='parent_buffer_shared'))
            if np.shares_memory(obj,oldbuf):
                view_count+=1
                if not np.shares_memory(other,newbuf):alias_errors.append(dict(path=list(path),field=field,issue='copied_view_detached_from_clone_buffer'))
    check('all_cached_ndarray_buffer_aliases_preserved',not alias_errors,
        dict(ndarrays_assessed=array_count,live_data_views_assessed=view_count,issues=alias_errors))
    metadata.update(schema='v25_opaque_preobserve_session_fork_v1',time_s=now,tick=int(session.tick),
        opaque_integration_sha256=before,checks=checks,passed=True,
        physics_steps=0,sensor_samples_added=0,source='experiment_coordinator_only',
        checkpoint_state_exposed_to_policy=False,response_model_admitted=False,
        scope='Independent copy capability at this boundary, not complete remaining behaviour bridge or physical safety')
    return cloned,metadata
