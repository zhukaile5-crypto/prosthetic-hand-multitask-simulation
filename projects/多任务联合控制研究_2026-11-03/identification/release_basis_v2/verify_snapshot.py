"""Exact-state regression checks, no plant or physical stepping.

For REAL filled sensor/controller/FK state use run_pilot.py --preflight, whose
implementation samples once at time0 and never calls a dynamics step.
"""
from pathlib import Path
from dataclasses import dataclass
from collections import deque
from types import MappingProxyType
import json,sys
import numpy as np
PILOT=Path(__file__).resolve().parent;ROOT=PILOT.parents[1]
sys.path.insert(0,str(PILOT));sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import research_imports
import mujoco
from state_snapshot import freeze,digest,without_wallclock,StateSerializationError
from force_model import ForceResponse


@dataclass
class MutableFixture:
    scalar:float


def verify():
    checks={};references=[]
    module=freeze({'skin_geometry':{'mj':mujoco}},code_references=references)
    checks['actual_skin_mujoco_handle_has_precise_path_and_native_file_identity']=bool(
        len(references)==1 and references[0]['object_path']=='$.skin_geometry.mj' and references[0]['module_name']=='mujoco'
        and references[0]['origin_sha256'] and references[0]['covered_native_binding_sha256'])
    checks['module_namespace_is_not_traversed']=references[0]['namespace_not_serialized']
    for name,value,wanted in [('mappingproxy',MappingProxyType({'mutable':[1]}),'builtins.mappingproxy'),
        ('class_reference',MutableFixture,'builtins.type'),('random_module',np.random,'builtins.module')]:
        try:freeze({'outer':{'actual':value}})
        except StateSerializationError as exc:
            checks[name+'_fails_at_exact_path']=bool(exc.object_path=='$.outer.actual' and exc.type_name==wanted)
        else:checks[name+'_fails_at_exact_path']=False
    rng=np.random.default_rng(71);before=digest(freeze(rng));rng.normal(size=3)
    checks['actual_rng_mutation_changes_exact_snapshot']=before!=digest(freeze(rng))
    same=np.random.default_rng(71);same.normal(size=3)
    checks['same_rng_state_has_same_exact_snapshot']=digest(freeze(same))==digest(freeze(rng))
    queue=deque([np.array([1.,2.])],maxlen=7);a=freeze(queue)
    queue[0][0]=9.;b=freeze(queue)
    checks['queued_arrays_not_replaced_by_shape_only']=digest(a)!=digest(b)
    checks['deque_capacity_is_retained']=a['maxlen']==7
    item=MutableFixture(1.);item.extra={'dynamic_sensor_geometry':[1,2]};first=freeze(item)
    item.extra['dynamic_sensor_geometry'].append(3)
    checks['dataclass_dynamic_attributes_not_omitted']=bool('extra' in first['extra_attributes'] and digest(first)!=digest(freeze(item)))
    response=freeze(ForceResponse(np.eye(5),.005))
    checks['actual_force_response_implementation_is_in_source_closure']=bool(response['implementation']['origin_sha256'])
    fields={'time':12.,'qp_seconds':.1,'nested':{'wall_seconds':4.,'sample_time':11.99}}
    a=digest(without_wallclock(freeze(fields)));fields['qp_seconds']=9.;fields['nested']['wall_seconds']=10.
    checks['only_explicit_wallclock_data_ignored']=a==digest(without_wallclock(freeze(fields)))
    fields['time']=12.005
    checks['simulation_clock_never_ignored']=a!=digest(without_wallclock(freeze(fields)))
    checks['typed_dictionary_keys_not_conflated']=digest(freeze({1:'a','1':'b'}))!=digest(freeze({'1':'b'}))
    cycle={};cycle['self']=cycle
    try:freeze(cycle)
    except StateSerializationError as exc:checks['cyclic_state_is_rejected_with_path']=exc.object_path=='$.self'
    else:checks['cyclic_state_is_rejected_with_path']=False
    if not all(checks.values()):raise AssertionError(checks)
    return dict(schema='release_basis_snapshot_pure_v2',passed=True,checks=checks,physics_steps=0,plant_constructions=0,
        code_reference_paths=[r['object_path'] for r in references],scope='exact codecs and actual pinned ForceResponse/MuJoCo API identity; no physical acceptance')


if __name__=='__main__':
    result=verify();target=PILOT/'snapshot_pure_validation_v2.json'
    if target.exists():raise FileExistsError(target)
    with target.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))
