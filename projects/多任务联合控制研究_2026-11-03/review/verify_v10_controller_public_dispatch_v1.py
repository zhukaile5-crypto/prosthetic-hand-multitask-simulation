"""Actual controller dispatch on archived public frames; zero plant/physics.

Robot FK/bank construction are replaced by unavailable stubs. This validates
the retained force state/action, fallback and lifecycle, not geometry or model
construction (separately checked by the root constructor smoke).
"""
from pathlib import Path
from dataclasses import fields
from copy import deepcopy
from types import SimpleNamespace, MethodType
import argparse
import hashlib
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import research_imports
from contracts import SensorFrame
from controller_adapter import create_controller
import v10.controller as subject
from v10.directional_allocator import ROBOT_AXES25


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame(value):
    core = {x.name for x in fields(SensorFrame)}
    f = SensorFrame(**{k: np.asarray(v) if isinstance(v,list) else v for k,v in value.items() if k in core})
    for k,v in value.items():
        if k not in core:setattr(f,k,np.asarray(v) if k in ('executed_motion_reference','motion_reference_rate_limits') else v)
    return f


class UnavailableRobotGeometry:
    raw_joint_names = ROBOT_AXES25[:24]
    axes25 = ROBOT_AXES25
    def __init__(self,*args):pass


class UnavailableBank:
    entries = ()
    def __init__(self,*args):pass
    def select(self,*args,**kwargs):
        return SimpleNamespace(response25x8=None, debug={'fixture':'unavailable bank, not a measured model'}, status='UNAVAILABLE_FIXTURE', reasons=('independent_missing_model_fixture',))


def construct(metadata):
    old_geometry,old_bank=subject.RobotContactSpace,subject.Bank
    try:
        subject.RobotContactSpace=UnavailableRobotGeometry;subject.Bank=UnavailableBank
        return subject.create_joint(metadata['calibration'],metadata['limits'],metadata['budget'],metadata['gains'],robot_model=object())
    finally:subject.RobotContactSpace,subject.Bank=old_geometry,old_bank


def task(value):
    result=deepcopy(value)
    # V10 newly declares actual robot-reference authority. Constant/moving
    # archived requests remain identical; these public bounds do not modify it.
    result.setdefault('motion_reference_bounds',{'lower':[0.,-.523599,-.698132],'upper':[.1,.174533,.488692]})
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'review/v10_controller_public_dispatch_v1.json');args=parser.parse_args()
    paths=[ROOT/'v10/controller.py',ROOT/'v10/runtime.py'];before={str(p):sha(p) for p in paths}
    checks=[];records=[]
    def check(name,value,**extra):checks.append(dict(name=name,passed=bool(value),**extra))
    last=None
    for motion in ('hold','transport'):
        base=ROOT/f'output/development_v9_{motion}_v1/runs/nominal__ratio_preview__610212'
        metadata=json.loads((base/'metadata.json').read_text(encoding='utf-8'))
        baseline=create_controller('ratio',metadata['calibration'],metadata['limits'],metadata['budget'],metadata['gains'])
        controller=construct(metadata)
        with np.load(base/'trace.npz',allow_pickle=False) as z:
            times=z['time'].copy();observations=z['observation_json'].copy();tasks=z['task_request_json'].copy();archived_motion=z['executed_motion_reference'].copy()
        action_diff=state_diff=motion_diff=0.;ticks=0;active=0
        for i in np.flatnonzero((times>=7.3-1e-9)&(times<12.-1e-9)):
            a=frame(json.loads(str(observations[i])));b=frame(json.loads(str(observations[i])))
            if times[i]<7.5-1e-9:baseline.state.update(a,.005);controller.state.update(b,.005);continue
            controller.set_task_request(task(json.loads(str(tasks[i]))))
            x=baseline.step(a);y=controller.step(b)
            action_diff=max(action_diff,float(np.max(abs(x.delta_closure-y.delta_closure))))
            for name in ('normal','alpha','latent_alpha','force_uncertainty'):
                state_diff=max(state_diff,float(np.max(abs(np.asarray(getattr(baseline.state,name))-np.asarray(getattr(controller.state,name))))))
            motion_diff=max(motion_diff,float(np.max(abs(controller.motion_reference-archived_motion[i]))))
            active+=int(controller.plan_active);ticks+=1;last=(metadata,deepcopy(b),task(json.loads(str(tasks[i]))))
        check(motion+'/900_actual_public_ticks',ticks==900)
        check(motion+'/one_exact_baseline_force_state',state_diff==0.,maximum_difference=state_diff)
        check(motion+'/healthy_nominal_action_bitexact',action_diff==0.,maximum_difference=action_diff)
        check(motion+'/public_preview_actual_reference_exact',motion_diff==0.,maximum_difference=motion_diff)
        check(motion+'/no_unrequested_geometry_intervention',active==0)
        check(motion+'/authoritative_state_is_unique',controller.nominal.state is controller.state and controller.state.force_state is not controller.state)
        records.append(dict(motion=motion,trace_sha256=sha(base/'trace.npz'),ticks=ticks,action_difference=action_diff,state_difference=state_diff,motion_reference_difference=motion_diff,geometry_intervention_ticks=active))
    metadata,f,request=last
    c=construct(metadata);r=create_controller('ratio',metadata['calibration'],metadata['limits'],metadata['budget'],metadata['gains'])
    f.time=7.5;f.normal=np.full(5,6.);f.lift_velocity_m_s=None
    request['time_s']=7.5;request['motion_target']=np.asarray(f.executed_motion_reference).tolist();request['executed_motion_reference']=np.asarray(f.executed_motion_reference).tolist();request['motion_error']=[0.,0.,0.]
    # Build a coherent current public request instead of backdating old preview.
    from task_contract import build_task_request
    request=build_task_request(7.5,'hold',request['motion_target'],request['executed_motion_reference'],12.,metadata['limits'],metadata['budget'],tracking_tolerances=[.01,.015,.015],motion_reference_bounds=request['motion_reference_bounds'],sensor_characteristics=f.sensor_characteristics)
    c.set_task_request(request);x=r.step(deepcopy(f));y=c.step(deepcopy(f))
    check('risk_missing_model_returns_nominal_action',np.array_equal(x.delta_closure,y.delta_closure) and c.debug['status']=='BASELINE_ROBOT_RESPONSE_UNAVAILABLE')
    check('risk_missing_model_does_not_add_motion_freeze',y.pause_transport==x.pause_transport and c.debug['extra_motion_freeze'] is False)
    c=construct(metadata)
    def force_plan(self,before):
        self.plan_active=True;self.plan_start=float(self.latest_frame.time);self.plan_reference=np.r_[self.state.alpha,self.executed_motion]
        self.plan_delta=np.r_[[-.004,0.,0.,0.,0.],[0.,0.,0.]];self.reference_before=deepcopy(before)
        self.debug={'forced_plan_fixture':True,'reference_memory_rollback':False}
    c._decide=MethodType(force_plan,c);c.set_task_request(request)
    before_memory={name:deepcopy(getattr(c.nominal,name)) for name in ('integral','boost')}
    y=c.step(deepcopy(f));check('accepted_plan_rolls_back_reference_integral_boost',all(np.array_equal(getattr(c.nominal,name),value) for name,value in before_memory.items()) and c.debug['reference_memory_rollback'])
    release_frame=deepcopy(f);release_frame.time=7.505
    release_request=build_task_request(7.505,'release',request['motion_target'],request['executed_motion_reference'],11.005,metadata['limits'],metadata['budget'],tracking_tolerances=[.01,.015,.015],motion_reference_bounds=request['motion_reference_bounds'],sensor_characteristics=f.sensor_characteristics)
    c.set_task_request(release_request);z=c.step(release_frame,command='hold')
    check('release_immediately_cancels_finite_plan',not c.plan_active and c.release_latched and z.debug.get('release_authority') is True and np.all(z.delta_closure<=0.))
    later=deepcopy(release_frame);later.time+=.005
    c.set_task_request(build_task_request(later.time,'hold',request['motion_target'],request['executed_motion_reference'],12.,metadata['limits'],metadata['budget'],tracking_tolerances=[.01,.015,.015],motion_reference_bounds=request['motion_reference_bounds'],sensor_characteristics=f.sensor_characteristics))
    z=c.step(later,command='hold');check('release_latched_no_reclose',c.release_latched and np.all(z.delta_closure<=0.) and not c.plan_active)
    after={str(p):sha(p) for p in paths};check('reviewed_sources_unchanged_during_pure_replay',before==after)
    result=dict(schema='independent_v10_controller_public_dispatch_v1',passed=all(x['passed'] for x in checks),checks=checks,records=records,source_sha256=before,
        physics_steps=0,plant_constructions=0,robot_geometry_and_modelbank_constructors='unavailable stubs solely for force/dispatch replay; no geometry/model admission claim',
        scope='1800 actual healthy public policy ticks plus synthetic unavailable-model, antiwindup and release dispatch; no counterfactual physical rollout')
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']],records=records,physics_steps=0),ensure_ascii=False))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
