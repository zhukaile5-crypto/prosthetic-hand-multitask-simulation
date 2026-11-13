"""Path-aware actual-state encoder; module API handles are not mutable state.

Unexpected mappingproxy/classes/callables are rejected WITH THEIR OBJECT PATH,
not expanded into a namespace or silently converted to empty dictionaries.
The real virtual-skin MuJoCo API handle is the single audited exception and
has pinned origin/wheel/native-binding identity. RNG/queues/FK MjData are data.
"""
from pathlib import Path
from dataclasses import is_dataclass,fields
from collections import deque,defaultdict
from types import ModuleType,MappingProxyType
import base64,enum,json,hashlib
import numpy as np
from source_pinning import module_reference,instance_reference


class StateSerializationError(TypeError):
    def __init__(self,path,value,reason):
        self.object_path=path;self.type_name=type(value).__module__+'.'+type(value).__qualname__
        self.reason=reason
        super().__init__(f'{reason}; object_path={path}; exact_type={self.type_name}')


def freeze(value,stack=None,path='$',code_references=None):
    stack=set() if stack is None else stack
    references=[] if code_references is None else code_references
    def child(v,p):return freeze(v,stack|{id(value)},p,references)
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,(float,np.floating)):return {'float_hex':float(value).hex()}
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,(complex,np.complexfloating)):
        return dict(complex_real_hex=float(value.real).hex(),complex_imag_hex=float(value.imag).hex())
    if isinstance(value,(bytes,bytearray)):return dict(bytes_b64=base64.b64encode(bytes(value)).decode('ascii'))
    if isinstance(value,np.dtype):return dict(numpy_dtype=value.str,descr=value.descr if value.fields else None)
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject:
            if id(value) in stack:raise StateSerializationError(path,value,'Cyclic object array needs an explicit adapter')
            return dict(array_dtype=value.dtype.str,shape=list(value.shape),object_items=[child(v,f'{path}.flat[{i}]') for i,v in enumerate(value.flat)])
        a=np.ascontiguousarray(value)
        return dict(array_dtype=a.dtype.str,descr=a.dtype.descr if a.dtype.fields else None,shape=list(a.shape),bytes_b64=base64.b64encode(a.tobytes()).decode('ascii'))
    if isinstance(value,np.random.Generator):return dict(numpy_generator=type(value.bit_generator).__name__,state=child(value.bit_generator.state,path+'.bit_generator.state'))
    if isinstance(value,np.random.RandomState):return dict(numpy_random_state=child(value.get_state(),path+'.get_state'))
    if isinstance(value,enum.Enum):return dict(enum=type(value).__qualname__,value=child(value.value,path+'.value'))
    if isinstance(value,Path):return dict(path=str(value))
    if isinstance(value,ModuleType):
        try:description=module_reference(value,path)
        except Exception as exc:raise StateSerializationError(path,value,'Module handle not audited/pinned: '+str(exc)) from exc
        references.append(dict(object_path=path,**description));return description
    if isinstance(value,type):
        raise StateSerializationError(path,value,'Class reference needs an explicit pinned implementation adapter; do not traverse class.__dict__')
    if isinstance(value,MappingProxyType):
        raise StateSerializationError(path,value,'Unexpected mappingproxy view; identify its owning state/code path before adding an adapter')
    if id(value) in stack:raise StateSerializationError(path,value,'Cyclic mutable state needs an explicit adapter')
    if isinstance(value,dict):
        if isinstance(value,defaultdict):
            # A default_factory can affect future state; never lose it merely
            # by treating a defaultdict as a plain dictionary.
            factory=child(value.default_factory,path+'.default_factory')
        else:factory=None
        items=[]
        for i,(k,v) in enumerate(value.items()):
            key=child(k,f'{path}.key[{i}]')
            suffix='.'+k if isinstance(k,str) and k.isidentifier() else '['+json.dumps(key,sort_keys=True)+']'
            items.append([key,child(v,path+suffix)])
        items.sort(key=lambda pair:json.dumps(pair[0],ensure_ascii=False,sort_keys=True))
        result=dict(mapping_type=type(value).__module__+'.'+type(value).__qualname__,mapping_items=items)
        if isinstance(value,defaultdict):result['default_factory']=factory
        return result
    if isinstance(value,(list,tuple,deque)):
        result=dict(sequence_type=type(value).__module__+'.'+type(value).__qualname__,items=[child(v,f'{path}[{i}]') for i,v in enumerate(value)])
        if isinstance(value,deque):result['maxlen']=value.maxlen
        return result
    if isinstance(value,(set,frozenset)):
        values=[child(v,path+'.set_item') for v in value]
        return dict(set_type=type(value).__name__,items=sorted(values,key=lambda v:json.dumps(v,sort_keys=True)))
    if is_dataclass(value):
        try:implementation=instance_reference(value,path)
        except Exception as exc:raise StateSerializationError(path,value,'Unpinned dataclass implementation: '+str(exc)) from exc
        declared={f.name:child(getattr(value,f.name),path+'.'+f.name) for f in fields(value)}
        extra={k:child(v,path+'.'+k) for k,v in vars(value).items() if k not in declared} if hasattr(value,'__dict__') else {}
        return dict(implementation=implementation,fields=declared,extra_attributes=extra)
    if hasattr(value,'__dict__'):
        try:implementation=instance_reference(value,path)
        except Exception as exc:raise StateSerializationError(path,value,'Unpinned mutable instance implementation: '+str(exc)) from exc
        return dict(implementation=implementation,state=child(vars(value),path))
    raise StateSerializationError(path,value,'Unregistered exact mutable state type')


def mjdata_state(model,data):
    import mujoco
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    integration=np.empty(mujoco.mj_stateSize(model,spec));mujoco.mj_getState(model,data,integration,spec)
    result=dict(integration_state=integration,integration_state_spec=int(spec),time=float(data.time))
    for name in ('qpos','qvel','act','ctrl','qacc_warmstart','qacc','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat',
                 'eq_active','userdata','plugin_state','actuator_force','qfrc_actuator','qfrc_constraint','qacc_smooth','sensordata','efc_force','efc_state'):
        if hasattr(data,name):result[name]=np.array(getattr(data,name),copy=True)
    return result


def side_state(s):
    """Real causal side state; FK MjData is explicitly retained, not omitted."""
    p=s.plant
    preparation={k:v for k,v in vars(s.preparation).items() if k!='plant'}
    preparation['plant_reference']='covered by entry_physics and plant_side'
    skin={k:v for k,v in vars(s.skin_geometry).items() if k not in ('model','data')}
    skin['fk_data']=mjdata_state(s.skin_geometry.model,s.skin_geometry.data)
    skin['model_reference']=dict(same_as_actual_plant=s.skin_geometry.model is p.model,robot_model_sha256=p.model_sha256)
    keys=('tick','running','paused','completed','saved','handoff_attempted','handoff_passed','handoff_alpha','reference_relative',
        'release_at','release_supported','preparation_failure','failure_reason','failure_at','manual_force_until','manual_squeeze_until',
        'manual_release','last_obs','last_desired','last_prediction','last_action','last_debug','last_command','environment_signature',
        'task_request','motion_reference_bounds','user_trajectory','sensor_characteristics','lifecycle_enabled','release_selection',
        'trust_radius','limits','budget','gains')
    return dict(controller=s.controller,sensors=s.sensors,preparation=preparation,skin_geometry=skin,
        session_causal_state={k:getattr(s,k) for k in keys if hasattr(s,k)},
        plant_side={k:getattr(p,k) for k in ('control_steps','alpha','phase','lift_ref','wrist_ref','release_started','release_thumb',
            'release_lift','release_alpha','release_ctrl','env','max_hidden_clip')})


WALLCLOCK_KEYS=('qp_seconds','control_ms','wall_seconds')
def without_wallclock(encoded):
    if isinstance(encoded,dict):
        result={k:without_wallclock(v) for k,v in encoded.items()}
        if 'mapping_items' in result:
            result['mapping_items']=[pair for pair in result['mapping_items'] if not isinstance(pair[0],str) or pair[0] not in WALLCLOCK_KEYS]
        return result
    if isinstance(encoded,list):return [without_wallclock(v) for v in encoded]
    return encoded


def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
