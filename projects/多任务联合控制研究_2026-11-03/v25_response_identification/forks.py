"""Opaque experiment checkpoint copies. Never imported by a decision policy.

The experiment coordinator may fork a PRE_OBSERVE session. Physics state is
copied internally and represented externally only by an equality fingerprint;
it must not be a public observation, regression feature or control input.
All model/data, devices, RNGs, observer and controller state are independent.
Only pinned native API modules/functions are shared as immutable code handles.
"""
from pathlib import Path
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from numbers import Real, Integral
from bisect import bisect_left, bisect_right
import copy, hashlib, json, types, sys
import numpy as np
import mujoco

DELEGATED_WRAPPERS=('v17_contact_transfer.motion_domain_sensors','v10.controller','v9.controller')

def _walk(value,path=(),seen=None):
    seen=set() if seen is None else seen
    yield path,value
    if id(value) in seen:return
    seen.add(id(value))
    if isinstance(value,Mapping):items=list(value.items())
    elif isinstance(value,(list,tuple,deque)):items=list(enumerate(value))
    elif isinstance(value,(np.ndarray,str,bytes,int,float,bool,type(None),type,types.ModuleType,types.MethodType)):return
    elif isinstance(value,types.FunctionType):items=[('__defaults__',value.__defaults__),('__kwdefaults__',value.__kwdefaults__),('__closure__',value.__closure__)]
    elif isinstance(value,types.CellType):items=[('cell_contents',value.cell_contents)]
    elif is_dataclass(value):
        # The frozen SensorFrame deliberately removes legacy model-only
        # fields after acquisition. Preserve actual absence; never recreate
        # a missing force predictor from a dataclass declaration/default.
        items=[(f.name,getattr(value,f.name)) for f in fields(value) if hasattr(value,f.name)]
    else:
        try:items=list(vars(value).items())
        except TypeError:items=[]
        names={key for key,_ in items}
        for cls in type(value).__mro__:
            slots=cls.__dict__.get('__slots__',())
            slots=(slots,) if isinstance(slots,str) else slots
            for name in slots:
                if name in ('__dict__','__weakref__'):continue
                actual='_'+cls.__name__.lstrip('_')+name if name.startswith('__') and not name.endswith('__') else name
                if actual not in names and hasattr(value,actual):items.append((actual,getattr(value,actual)));names.add(actual)
    for key,child in items:
        if isinstance(key,(int,str)):yield from _walk(child,path+(key,),seen)

def _resolve(value,path):
    for key in path:value=value[key] if isinstance(value,(Mapping,list,tuple,deque)) else getattr(value,key)
    return value

def _boundary(session):
    def clock(value):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value) or value<0:
            raise ValueError('finite nonnegative real boundary clock required')
        return float(value)
    now=clock(session.time);physical_now=clock(session.plant.time)
    if abs(now-physical_now)>1e-10:
        raise ValueError('session and physics boundary clocks differ')
    if isinstance(session.tick,(bool,np.bool_)) or not isinstance(session.tick,Integral) or session.tick<0:
        raise ValueError('explicit nonnegative integer sensor tick required')
    if abs(now-int(session.tick)*.005)>1e-10 or getattr(session.plant,'_delivery_inside_advance',False):
        raise ValueError('PRE_OBSERVE complete5ms boundary required')
    last=getattr(session.sensors,'_last_time',None)
    if (now==0. and last is not None) or (now>0. and (last is None or abs(clock(last)-(now-.005))>1e-9)):
        raise ValueError('fork must precede this tick sensor acquisition')
    if abs(clock(session.plant.model.opt.timestep)-.0005)>1e-15:
        raise ValueError('frozen500us physics timestep required')
    return now

def _integration_digest(session):
    signature=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(session.plant.model,signature))
    mujoco.mj_getState(session.plant.model,session.plant.data,state,signature)
    return hashlib.sha256(state.tobytes()).hexdigest()

def _copy(session):
    nodes=list(_walk(session));memo={};wrappers=[];modules=[];proxies=[];owned_callbacks=[]
    plant_module=sys.modules.get('hand_plant')
    plant_type=None if plant_module is None else getattr(plant_module,'HandPlant',None)
    lookup_code=() if plant_type is None else tuple(c for c in plant_type.__init__.__code__.co_consts if isinstance(c,types.CodeType) and c.co_name=='<lambda>')
    prepared_module=sys.modules.get('v13_joint.prepared_native_effort21')
    prepared_type=None if prepared_module is None else getattr(prepared_module,'_FixedSnapshotServo',None)
    def code_children(code):
        result=[code]
        for child in code.co_consts:
            if isinstance(child,types.CodeType):result.extend(code_children(child))
        return result
    guard_codes=() if prepared_type is None else tuple(c for c in code_children(prepared_type.__init__.__code__) if c.co_name=='<lambda>')
    def immutable_default(value):
        return value is None or isinstance(value,(str,bytes,int,float,bool,complex,type)) or isinstance(value,(tuple,frozenset)) and all(immutable_default(x) for x in value)
    for path,obj in nodes:
        if isinstance(obj,types.ModuleType):
            name=obj.__name__
            if name not in ('mujoco','numpy') or sys.modules.get(name) is not obj:
                raise ValueError('Unreviewed stored API module: '+name)
            memo[id(obj)]=obj;modules.append(name)
        if isinstance(obj,types.FunctionType):
            module=sys.modules.get(obj.__module__)
            exported=module is not None and getattr(module,obj.__name__,None) is obj
            plant_lookup=bool(path==('plant','bid') or path==('plant','aid')) and obj.__module__=='hand_plant' and any(obj.__code__ is code for code in lookup_code) and obj.__code__.co_freevars==('self',) and len(obj.__closure__ or ())==1 and obj.__closure__[0].cell_contents is session.plant
            guard_lookup=prepared_type is not None and obj.__module__=='v13_joint.prepared_native_effort21' and any(obj.__code__ is code for code in guard_codes) and obj.__code__.co_freevars==('self',) and len(obj.__closure__ or ())==1 and isinstance(obj.__closure__[0].cell_contents,prepared_type)
            if plant_lookup or guard_lookup:
                # Frozen HandPlant stores two name->id lambdas closing over
                # self. Rebind their ORIGINAL code to a child-owner cell;
                # otherwise even an apparently successful deepcopy retains
                # a hidden callback into the parent's native model.
                cell=(lambda value:lambda:value)(None).__closure__[0]
                new=types.FunctionType(obj.__code__,obj.__globals__,obj.__name__,None,(cell,))
                new.__qualname__=obj.__qualname__;memo[id(obj)]=new;owned_callbacks.append((obj,new,cell,obj.__closure__[0].cell_contents,path))
            else:
                # A module-exported function can mutate its module globals;
                # export identity is not a purity proof. The concrete frozen
                # session needs only the two reviewed closure families above.
                raise ValueError('Stored function callback outside reviewed frozen owner/code families: '+str(path))
        if isinstance(obj,types.MappingProxyType) and id(obj) not in memo:
            backing={};memo[id(obj)]=types.MappingProxyType(backing);proxies.append((obj,backing,path))
        cls=type(obj)
        if cls.__module__ in DELEGATED_WRAPPERS and '__getattr__' in cls.__dict__ and id(obj) not in memo:
            shell=object.__new__(cls);memo[id(obj)]=shell;wrappers.append((obj,shell,path))
        if isinstance(obj,np.ndarray) and not obj.flags.writeable:
            base=obj
            while isinstance(base,np.ndarray) and base.base is not None:base=base.base
            if isinstance(base,bytes):
                if obj.dtype.hasobject:raise ValueError('Object pointers cannot use immutable raw-byte clone backing')
                origin=np.frombuffer(base,dtype=np.uint8).__array_interface__['data'][0]
                offset=obj.__array_interface__['data'][0]-origin
                backing=memoryview(base).tobytes()
                memo[id(obj)]=np.ndarray(obj.shape,dtype=obj.dtype,buffer=backing,offset=offset,strides=obj.strides)
    for old,new,_ in wrappers:vars(new).update(copy.deepcopy(vars(old),memo))
    for old,backing,_ in proxies:
        backing.update(copy.deepcopy(dict(old),memo))
    cloned=copy.deepcopy(session,memo)
    for old,new,cell,owner,_ in owned_callbacks:
        if id(owner) not in memo:raise ValueError('Reviewed callback owner missing from cloned session')
        cell.cell_contents=memo[id(owner)];new.__defaults__=copy.deepcopy(old.__defaults__,memo)
        new.__kwdefaults__=copy.deepcopy(old.__kwdefaults__,memo);new.__annotations__=copy.deepcopy(old.__annotations__,memo)
    # deepcopy does not preserve NumPy readonly flags. Preserve declarations,
    # including immutable public packets, without changing any numeric value.
    for path,obj in nodes:
        if isinstance(obj,np.ndarray) and not obj.flags.writeable:
            _resolve(cloned,path).setflags(write=False)
    return cloned,nodes,dict(shared_api_modules=sorted(set(modules)),
        delegated_wrapper_shell_paths=[list(path) for _,_,path in wrappers],
        independently_copied_readonly_mapping_paths=[list(path) for _,_,path in proxies],
        rebound_frozen_owned_callback_paths=[list(path) for _,_,_,_,path in owned_callbacks])

def _native_buffers(nodes):
    """Inspect every encountered native model/data array, not just qpos/ctrl.

    Cached live native views are outside this copier's supported domain: a
    normal ndarray deepcopy detaches such a view from the cloned native owner.
    Reject them explicitly rather than pretending to rebind an arbitrary view.
    """
    records=[];seen=set()
    for path,obj in nodes:
        if not isinstance(obj,(mujoco.MjModel,mujoco.MjData)) or id(obj) in seen:continue
        seen.add(id(obj))
        for field in dir(obj):
            if field.startswith('_'):continue
            try:array=getattr(obj,field)
            except (AttributeError,ValueError):continue
            if not isinstance(array,np.ndarray) or not array.size:continue
            low,high=np.lib.array_utils.byte_bounds(array)
            records.append((low,high,array,list(path),field))
    records.sort(key=lambda r:r[0]);starts=[r[0] for r in records];maximum=[];value=0
    for row in records:value=max(value,row[1]);maximum.append(value)
    return records,starts,maximum

def _native_overlaps(array,buffers):
    if not array.size:return []
    records,starts,maximum=buffers;low,high=np.lib.array_utils.byte_bounds(array)
    left=bisect_right(maximum,low);right=bisect_left(starts,high)
    return [row for row in records[left:right] if np.shares_memory(array,row[2])]

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
    # No detached native view is silently accepted, including qacc, ctrl and
    # private servo/FK arrays. The supported concrete session must contain no
    # cached Python ndarray view into native buffers; real owner aliases above
    # remain fully preserved. This is a checked domain restriction, not a claim
    # that arbitrary Python object graphs can be independently cloned.
    array_count=0;alias_errors=[];buffers=_native_buffers(nodes)
    for path,obj in nodes:
        if not isinstance(obj,np.ndarray):continue
        other=_resolve(cloned,path);array_count+=1
        for row in _native_overlaps(obj,buffers):alias_errors.append(dict(path=list(path),native_owner=row[3],field=row[4],issue='unsupported_cached_live_native_view'))
        for row in _native_overlaps(other,buffers):alias_errors.append(dict(path=list(path),native_owner=row[3],field=row[4],issue='parent_native_buffer_shared'))
    check('no_cached_native_views_or_parent_buffer_aliases',not alias_errors,
        dict(ndarrays_assessed=array_count,native_arrays_assessed=len(buffers[0]),issues=alias_errors))
    metadata.update(schema='v25_opaque_preobserve_session_fork_v1',time_s=now,tick=int(session.tick),
        opaque_integration_sha256=before,checks=checks,passed=True,
        physics_steps=0,sensor_samples_added=0,source='experiment_coordinator_only',
        checkpoint_state_exposed_to_policy=False,response_model_admitted=False,
        scope='Independent copy capability at this boundary, not complete remaining behaviour bridge or physical safety')
    return cloned,metadata
