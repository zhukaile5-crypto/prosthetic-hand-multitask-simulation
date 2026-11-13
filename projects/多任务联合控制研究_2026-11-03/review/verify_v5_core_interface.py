"""Independent V5 real-controller synthetic checks; no hand/dynamics import."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import argparse,hashlib,json,sys,runpy
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import coordinator as c
from task_contract import build_task_request
EXPECTED={'coordinator.py':'156260395048515780542224d862560684cd579817160282102745b66cc17beb',
 'coordinator_model.py':'550c6a8ec5846d7a6cfb1a7b28e8261d0230612fd6547729270c4ab3d7c56918',
 'coordinator_contacts.py':'1759fdd94398bc32d1cda6a6790ba8327b72f67fe85909ba8ec4e41d70d7637c'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def request(target,executed,time=8.,mode='transport'):
    return build_task_request(time,mode,target,executed,12.,[5.]*5,18.,[.01,.015,.015])

def fixture():
    # Motion changes pressure by more than the old0.15N anchor tube while
    # pressure/shear remain physical and well inside5N/18N guards.
    G=np.zeros((16,8));G[:5,:5]=np.eye(5)*10.;G[5:10,:5]=np.eye(5)*6.
    G[:5,5]=[400.,300.,200.,150.,100.];G[:5,6]=[2.,-1.,1.,-2.,.5];G[:5,7]=[-.5,.4,-.6,.8,-1.]
    G[5:10,5:]=.6*G[:5,5:];G[10:13,5:]=np.eye(3)*2.
    K=np.asarray([G*(1-np.exp(-t/.02)) for t in np.arange(101)*.005])
    cal=dict(force_matrix=np.eye(5)*10.,force_tau=.02,reference_alpha=np.zeros(5),response_error_bound=np.full(5,.02),config={'shape':'capsule','lift_target':.1})
    ctl=c.JointCoordinator(cal,np.full(5,5.),18.,model_document=c.synthetic_document(K));p=np.full(5,2.);tau=.6*p;y=np.r_[p,tau,[0.,0.,1.2,0.,0.,0.]]
    state=SimpleNamespace(normal=p.copy(),alpha=np.zeros(5),closure_min=np.full(5,-.04),closure_max=np.full(5,.04),valid=np.ones(5,bool),closure_permission=np.ones(5,bool),contact_model_active=np.ones(5,bool),contact=np.ones(5,bool),confidence=np.ones(5),slip_magnitude=np.zeros(5),auxiliary_wrench=np.zeros(6),auxiliary_valid=True,load=1.2,load_uncertainty=.5,up=np.array([0.,0.,1.]),force_gain=np.full(5,10.),slip_speed=0.)
    ctl.state=state
    ctl.model.record_input(7.98,np.r_[np.zeros(5),.1,0.,0.]);ctl.model.record_input(8.,np.r_[np.zeros(5),.1,0.,0.]);ctl.model.beta=y.copy()
    ctl.model.snapshot=c.OutputForecast(y,np.zeros(16),np.full(16,.001),np.full(16,.01),np.full(16,50),np.ones(16),np.ones(16,bool),y.copy(),np.full(16,8.))
    for k in range(51):ctl.contacts.update(SimpleNamespace(time=7.75+k*.005,normal=p,shear_magnitude=tau,contact_pos=np.zeros((5,3))),state,ctl.model.snapshot)
    ctl.set_task_request(request([.1,0.,0.],[.1,0.,0.],mode='hold'));ctl._decision()
    return ctl

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);a=p.parse_args();before={k:sha(ROOT/k) for k in EXPECTED}
    if before!=EXPECTED:raise RuntimeError('Frozen V5 source changed; not certifying another revision')
    checks={};provided=c.self_test();checks.update({'provided/'+k:bool(v) for k,v in provided['checks'].items()})
    ctl=fixture();target=np.array([.098,.003,-.002]);ctl.set_task_request(request(target,[.1,0.,0.]));ctl._decision()
    expected=np.array([-.0007,.003,-.002]);nominal=np.r_[np.zeros(5),expected]
    checks['independent_safe_three_axis_motion_is_not_silenced']=ctl.debug['nominal_task_preserved'] and np.allclose(ctl.interval_motion,expected,atol=1e-12,rtol=0)
    checks['nominal_pressure_change_exceeds_old_anchor_tube']=abs(ctl.predicted[0]-2.)>.15
    checks['no_alpha_or_relative_correction_for_safe_nominal']=np.array_equal(ctl.interval_action,np.zeros(5)) and np.allclose(ctl.debug['correction_8'],0.,atol=1e-12,rtol=0)
    reference_plan=ctl.previous_plan.copy();old_force=np.asarray(ctl.debug['nominal_force_prediction'])
    ctl.contacts.anchor_pressure[:]=9.;ctl.contacts.anchor_shear[:]=.05;ctl.contacts.anchor_ratio[:]=.01
    ctl.contacts.snapshot.anchor_pressure[:]=9.;ctl.contacts.snapshot.anchor_shear[:]=.05;ctl.contacts.snapshot.anchor_ratio[:]=.01
    ctl._decision();checks['stale_historical_pressure_anchor_cannot_pin_motion']=ctl.debug['nominal_task_preserved'] and np.allclose(ctl.previous_plan,reference_plan,atol=1e-12,rtol=0)
    # Run the actual step/execution interface for all four5ms ticks.
    ctl.state.update=lambda f,dt:None;ctl.last_decision=8.;ctl.previous_time=7.995;executed=np.array([.1,0.,0.]);deltas=[];motions=[]
    for k in range(4):
        ctl.set_task_request(request(target,executed,8.+k*.005));out=ctl.step(SimpleNamespace(time=8.+k*.005,phase='HOLDING'));deltas.append(out.delta_closure.copy());ctl.state.alpha+=out.delta_closure
        motions.append(ctl.motion_reference-executed);executed=ctl.motion_reference.copy()
    checks['actual_four_tick_motion_total_matches_nominal_20ms_plan']=np.allclose(np.sum(motions,axis=0),expected,atol=1e-12,rtol=0)
    checks['actual_four_tick_finger_total_stays_zero']=np.array_equal(np.sum(deltas,axis=0),np.zeros(5))
    # Unknown object/scene/future diagnostic fields cannot change actual plan.
    one=fixture();two=fixture();public=request(target,[.1,0.,0.]);poison=deepcopy(public);poison.update(scenario='future_scene',object_pose=[99.,0.,0.],true_mass=99.,true_mu=.001,is_target=False,future_disturbance=[99.,99.,99.])
    one.set_task_request(public);two.set_task_request(poison);one._decision();two._decision()
    checks['actual_plan_invariant_to_hidden_object_scene_future_fields']=np.array_equal(one.previous_plan,two.previous_plan) and np.array_equal(one.interval_motion,two.interval_motion)
    # The regularizer's previous correction must come from actual history,
    # rather than from the previous desired or preferred command.
    history=fixture();actual=np.r_[[.001,0.,0.,0.,0.],.0998,.001,0.];history.model.record_input(8.,actual);history.model.beta=history.model.snapshot.output_now-history.model.input_effect(8.,8.)
    history.state.alpha=actual[:5].copy();history.previous_nominal=np.r_[np.full(5,.003),-.0004,.002,0.];expected_previous=actual-np.r_[np.zeros(5),.1,0.,0.]-history.previous_nominal
    history.set_task_request(request([.0998,.001,0.],actual[5:]));history._decision()
    checks['previous_correction_uses_executed_eight_input_history']=np.allclose(history.debug['actual_previous_interval_correction_8'],expected_previous,atol=1e-12,rtol=0)
    # Even an erroneous later HOLD request cannot revoke release ownership.
    monitor=fixture();monitor.state.update=lambda f,dt:None;monitor.set_task_request(request([.1,0.,0.],[.1,0.,0.],mode='monitor'))
    monitor._decision=lambda:(_ for _ in ()).throw(AssertionError('Holding decision called after monitor latch'))
    first=monitor.step(SimpleNamespace(time=8.,phase='HOLDING'));monitor.set_task_request(request(target,[.1,0.,0.],8.005,mode='hold'));last=monitor.step(SimpleNamespace(time=8.005,phase='HOLDING'))
    checks['monitor_latches_no_holding_reentry_even_if_later_task_says_hold']=monitor.release_latched and first.debug['release_authority'] and last.debug['release_authority'] and np.isnan(last.predicted_normal).all() and (last.delta_closure<=0.).all()
    checks['stable_sources_during_actual_core_checks']=all(sha(ROOT/k)==v for k,v in before.items())
    checks={k:bool(v) for k,v in checks.items()}
    result=dict(schema='independent_v5_actual_core_synthetic_interface_v1',passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],source_sha256=before,reviewer_sha256=sha(__file__),
        nominal_input8=nominal.tolist(),nominal_pressure_prediction=old_force.tolist(),executed_four_tick_motion_increment=np.sum(motions,axis=0).tolist(),actual_previous_correction_expected=expected_previous.tolist(),
        physics_steps_executed=0,development_interface_ready=all(checks.values()),confirmation_admitted=False,model_admitted=False,
        scope='Actual frozen controller on independent synthetic response/measurement witnesses. No hand model or dynamics imported. Local algebra/interfaces are ready for paired experimental development, not proof of empirical superiority or all-domain physical safety.')
    if a.out:
        out=a.out.resolve()
        if not out.is_relative_to(ROOT/'review') or out.exists():raise ValueError('Fresh own review output required')
        out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
