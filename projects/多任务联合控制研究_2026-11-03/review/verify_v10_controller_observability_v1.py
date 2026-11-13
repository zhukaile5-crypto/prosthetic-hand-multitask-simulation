"""Independent decision-level baseline observability guard and intent bounds."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'review'))
import verify_v10_controller_public_dispatch_v1 as helpers
from controller_adapter import create_controller


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    before=sha(ROOT/'v10/controller.py');checks=[];examples=[]
    def check(name,value,**extra):checks.append(dict(name=name,passed=bool(value),**extra))
    base=ROOT/'output/development_v9_hold_v1/runs/nominal__ratio_preview__610212'
    metadata=json.loads((base/'metadata.json').read_text(encoding='utf-8'))
    with np.load(base/'trace.npz',allow_pickle=False) as z:
        i=int(np.flatnonzero(abs(z['time']-7.5)<1e-9)[0]);original=helpers.frame(json.loads(str(z['observation_json'][i])));request=helpers.task(json.loads(str(z['task_request_json'][i])))
    for count in (0,1,2):
        c=helpers.construct(metadata);b=create_controller('ratio',metadata['calibration'],metadata['limits'],metadata['budget'],metadata['gains'])
        f=deepcopy(original);f.normal=np.full(5,6.);f.valid=np.arange(5)<count
        c.set_task_request(request)
        x=b.step(deepcopy(f));y=c.step(deepcopy(f))
        check(f'valid{count}/stabilizer_action_exact',np.array_equal(x.delta_closure,y.delta_closure))
        check(f'valid{count}/baseline_pause_retained',bool(x.pause_transport) and bool(y.pause_transport) and c.debug['stabilizer_pause_transport'] and not c.debug['stabilizer_pause_replaced_by_directional_plan'])
        check(f'valid{count}/no_geometry_acceptance_before_observability',not c.plan_active and c.debug['status']=='BASELINE_RETAINED_GRIP_OBSERVABILITY_REQUIREMENT')
        check(f'valid{count}/motion_stays_actual',np.array_equal(c.motion_reference,c.executed_motion))
        examples.append(dict(valid_count=count,action=y.delta_closure.tolist(),pause=bool(y.pause_transport),debug=y.debug))
    # A previously committed plan must be reconsidered at the next PUBLIC
    # decision. No new faster validity/pause rule is silently attributed to old
    # baseline between its20ms decision clocks.
    c=helpers.construct(metadata);c.set_task_request(request);c.step(deepcopy(original))
    c.plan_active=True;c.plan_start=7.5;c.plan_reference=np.r_[c.state.alpha,c.executed_motion];c.plan_delta=np.r_[[-.004,0.,0.,0.,0.],[-.0004,0.,0.]]
    c.reference_before={name:deepcopy(getattr(c.nominal,name)) for name in ('integral','boost')}
    f=deepcopy(original);f.time=7.52;f.normal=np.full(5,6.);f.valid=np.array([True,True,False,False,False])
    from task_contract import build_task_request
    req=build_task_request(7.52,'hold',request['motion_target'],request['executed_motion_reference'],12.,metadata['limits'],metadata['budget'],tracking_tolerances=[.01,.015,.015],motion_reference_bounds=request['motion_reference_bounds'],sensor_characteristics=f.sensor_characteristics)
    c.set_task_request(req);y=c.step(f)
    check('next_decision_cancels_old_plan_under3valid',not c.plan_active and y.pause_transport and c.debug['status']=='BASELINE_RETAINED_GRIP_OBSERVABILITY_REQUIREMENT')
    c=helpers.construct(metadata);c.set_task_request(request);c.step(deepcopy(original))
    c.nominal.interval_action=np.array([9.,-9.,9.,-9.,9.])
    target,lo,hi,uniform=c._motion_plan()
    check('uniform_model_intent_legal_actual_bounds',np.all(uniform>=lo) and np.all(uniform<=hi))
    check('uniform_intent_does_not_change_nominal_interval_state',np.array_equal(c.nominal.interval_action,[9.,-9.,9.,-9.,9.]))
    after=sha(ROOT/'v10/controller.py');check('source_stable',before==after)
    result=dict(schema='v10_controller_observability_guard_independent_v1',passed=all(c['passed'] for c in checks),checks=checks,examples=examples,
        controller_sha256=before,checker_sha256=sha(__file__),helper_sha256=sha(Path(helpers.__file__)),physics_steps=0,plant_constructions=0,
        scope='actual controller force/dispatch with unavailable geometry/bank constructors; baseline decision-level authority retained, no claim of intraperiod instantaneous stop or model/physics admission')
    output=ROOT/'review/v10_controller_observability_independent_v1.json'
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']],source=before,physics_steps=0),ensure_ascii=False))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
