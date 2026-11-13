"""Zero-dynamics interface oracles using an old public sensor packet.

This does not replay a physical response to a new action, and proves no quality
improvement. Robot construction, isolated FK and actuation algebra only.
"""
from pathlib import Path
import sys,json,hashlib
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import research_imports
from session import TrialSession,jsonable
from v12_native.controller import create_controller,METHODS


def main():
    out=Path(sys.argv[1]);assert not out.exists()
    source=next((ROOT/'output/development_v10_hold_v1/runs').glob('nominal__ratio_preview__*/trace.npz'))
    data=np.load(source,allow_pickle=False);idx=int(np.flatnonzero(abs(data['time']-7.5)<1e-9)[0])
    raw=json.loads(str(data['observation_json'][idx]));request=json.loads(str(data['task_request_json'][idx]))
    session=TrialSession(method='ratio',scenario='nominal',seed=610212,split='development')
    checks=[];details=[]
    def check(name,test):
        checks.append(dict(name=name,passed=bool(test)))
        if not test:raise AssertionError(name)
    check('construction_has_no_time_advance',session.time==0 and session.tick==0 and not session.trace)
    for method in METHODS:
        c=create_controller(method,dict(config=dict(shape='capsule')),session.limits,session.budget,session.gains,session.plant.model)
        obs=json.loads(json.dumps(raw));obs['closure_min']=session.plant.closure_min.tolist();obs['closure_max']=session.plant.closure_max.tolist()
        current=np.r_[obs['alpha'],request['executed_motion_reference']]
        obs['robot_actuator_target21']=c.tracker._port(current)
        obs['robot_control_time_s']=obs['time']
        c.set_task_request(request);frame=SimpleNamespace(**obs)
        result=c.step(frame,phase='HOLDING')
        check(method+'_finite_five_increment',result.delta_closure.shape==(5,) and np.isfinite(result.delta_closure).all())
        check(method+'_native_per_tick_rate',np.all(result.delta_closure<=c.g.close_rate*.005+1e-12) and np.all(result.delta_closure>=-c.g.open_rate*.005-1e-12))
        check(method+'_no_pressure_forecast',np.isnan(result.predicted_normal).all() and c.debug['pressure_prediction_available'] is False)
        check(method+'_actual_effort_interface_available',c.debug['effort_tracking']['increment8'] is not None)
        details.append(dict(method=method,force_intent=c.debug['force_intent'],tracker_status=c.debug['effort_tracking']['status'],delta=result.delta_closure.tolist()))
        for phase in ('RELEASE','RELEASING','MONITOR'):
            c2=create_controller(method,dict(config=dict(shape='capsule')),session.limits,session.budget,session.gains,session.plant.model)
            c2.set_task_request(request);before=c2.intent.checkpoint() if hasattr(c2.intent,'checkpoint') else None
            r=c2.step(SimpleNamespace(**obs),phase=phase)
            check(method+'_'+phase+'_exclusive_release',np.all(r.delta_closure<=0) and np.array_equal(c2.desired,np.zeros(5)) and c2.plan_start is None and c2.release_latched)
            check(method+'_'+phase+'_no_intent_integration',before is None or json.dumps(jsonable(before),sort_keys=True)==json.dumps(jsonable(c2.intent.checkpoint()),sort_keys=True))
        # Unknown pressure cannot manufacture a positive contact-search step.
        c3=create_controller(method,dict(config=dict(shape='capsule')),session.limits,session.budget,session.gains,session.plant.model)
        missing=json.loads(json.dumps(raw));missing.update(closure_min=obs['closure_min'],closure_max=obs['closure_max'],
            valid=[False]*5,age=[None]*5,normal=[None]*5,skin_patches=[],robot_actuator_target21=obs['robot_actuator_target21'],robot_control_time_s=obs['time'])
        missing['age']=[float('inf')]*5;missing['normal']=[float('nan')]*5
        c3.set_task_request(request);r=c3.step(SimpleNamespace(**missing),phase='HOLDING')
        check(method+'_unknown_not_zero_search',np.all(r.delta_closure<=0))
        json.dumps(jsonable(c.state_dict()),allow_nan=False)
        check(method+'_causal_export_json',True)
    check('all_zero_dynamics_kept_plant_at_zero',session.time==0 and float(session.plant.data.time)==0.)
    report=dict(passed=all(c['passed'] for c in checks),checks=checks,details=details,
        physics_advanced=False,quality_claim_allowed=False,
        public_packet_source=str(source),public_packet_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        controller_sha256=hashlib.sha256((ROOT/'v12_native/controller.py').read_bytes()).hexdigest())
    out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(jsonable(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),details=jsonable(details)),ensure_ascii=False))


if __name__=='__main__':main()
