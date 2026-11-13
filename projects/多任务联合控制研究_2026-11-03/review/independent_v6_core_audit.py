"""Targeted independent core checks: algebra/interfaces, no physical plant.

No complete saved-flow QP replay is repeated. A real closed-loop experiment is
required to evaluate the alternative commands produced by this controller.
"""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coordinator import JointCoordinator,make_gains
from coordinator_contacts import ForceQualityTask
from coordinator_model import ContactConditionedModel,OutputForecast,synthetic_document
from task_trajectory import public_transport_trajectory,unknown_future_point


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


class ForceResponse:
    def inverse(self,active):return np.diag(np.asarray(active,float)*.1)


def state(t=0.,pressure=1.):
    return SimpleNamespace(time=t,valid=np.ones(5,bool),contact_model_active=np.ones(5,bool),
        tangential=np.tile([.2,0.],(5,1)),delayed_normal=np.ones(5),normal=np.full(5,pressure),
        nominal_weights=np.ones(5),force_model=ForceResponse(),closure_permission=np.ones(5,bool),
        model_reference_alpha=np.full(5,.5),alpha=np.full(5,.5))


def calibration():
    return dict(reference_alpha=[.5]*5,response_error_bound=[.01]*5,
        force_matrix=(np.eye(5)*10.).tolist(),force_tau=.04,config=dict(shape='capsule',lift_target=.1))


def main():
    sources=['coordinator.py','coordinator_model.py','coordinator_contacts.py','grasp_contacts.py',
        'task_contract.py','task_trajectory.py','runtime.py','research_imports.py','calibration/public_contact_model_phase2_v1.json']
    identity={f:sha(ROOT/f) for f in sources};checks={};details={}
    gains=make_gains(dict(base=2.,baseline_tracking_gain=1.))
    member=SimpleNamespace(mode=np.full(5,'INCIPIENT',dtype='<U20'))
    q=ForceQualityTask(gains);s=state();mask=np.ones(5,bool);limits=np.full(5,5.)
    before=(q.integral,q.boost);action,_,_=q.propose(s,member,mask,limits,18.)
    q.reconcile(np.zeros(8))
    checks['blocked_nominal_does_not_commit_PI_or_slip_boost']=(q.integral,q.boost)==before
    s.time+=.02;action,_,_=q.propose(s,member,mask,limits,18.);q.reconcile(np.r_[action,np.zeros(3)])
    checks['actual_realized_slip_boost_is_committed']=q.boost>before[1]
    q=ForceQualityTask(gains);q.target_ratio=.1;q.integral=.2;member.mode[:]='STICK_CONFIRMED'
    action,_,_=q.propose(s,member,mask,limits,18.);q.reconcile(np.zeros(8))
    checks['blocked_ratio_PI_integral_stays_committed_value']=q.integral==.2
    s.time+=.02;action,_,_=q.propose(s,member,mask,limits,18.);q.reconcile(np.r_[-action,np.zeros(3)])
    checks['opposite_actual_increment_has_no_nominal_integral_credit']=q.integral==.2
    s.time+=.02;action,_,_=q.propose(s,member,mask,limits,18.);q.reconcile(None)
    checks['unavailable_execution_history_cannot_commit_integral']=q.integral==.2
    s.time+=.02;action,_,_=q.propose(s,member,mask,limits,18.);q.reconcile(np.r_[action,np.zeros(3)])
    checks['actual_realized_ratio_PI_integral_is_committed']=q.integral>.2
    q=ForceQualityTask(gains);s=state(0.);action,_,_=q.propose(s,member,mask,limits,18.)
    q.reconcile(np.r_[action,np.zeros(3)]);s.time=1.2
    action,_,_=q.propose(s,member,mask,limits,18.)
    checks['initial_ratio_learning_cannot_bridge_unobserved_decision_gap']=q.target_ratio is None
    q=ForceQualityTask(gains)
    for k in range(61):
        s=state(k*.02);action,_,_=q.propose(s,member,mask,limits,18.);q.reconcile(np.r_[action,np.zeros(3)])
    checks['continuous_actual_contact_observations_can_initialize_ratio']=q.target_ratio is not None and q.target_ratio>0.
    q1=ForceQualityTask(gains);q2=ForceQualityTask(gains);s=state()
    low,_,_=q1.propose(s,member,mask,limits,18.,noise_sigma=np.zeros(5))
    high,_,_=q2.propose(s,member,mask,limits,18.,noise_sigma=np.full(5,.1))
    checks['public_noise_deadzone_reduces_unresolved_force_correction']=np.linalg.norm(high)<np.linalg.norm(low)
    q=ForceQualityTask(gains);s=state();action,_,debug=q.propose(s,member,np.zeros(5,bool),limits,18.)
    checks['unaffiliated_below_limit_pressure_is_not_support_regulation']=np.allclose(action,0.) and not any(debug['persistent_feedback_mask'])
    s.normal[:]=6.;action,_,debug=q.propose(s,member,np.zeros(5,bool),limits,18.)
    checks['unaffiliated_pressure_is_still_protected_against_limits']=bool(np.all(action<0.))
    q=ForceQualityTask(gains);s=state(pressure=2.);motion=np.array([.1,-.2,.3,-.4,.5])
    action,target,debug=q.propose(s,member,mask,limits,18.,motion_pressure=motion,noise_sigma=np.zeros(5))
    checks['force_quality_separates_only_given_motion_component']=np.allclose(debug['force_quality_signal_N'],s.normal-motion) and np.allclose(target,np.asarray(debug['force_quality_base_target_N'])+motion)
    model=ContactConditionedModel(calibration(),document=synthetic_document())
    for t,value in [(0.,np.zeros(8)),(.02,np.array([.01,-.02,.015,0.,.02,.003,-.004,.002])),(.04,np.array([.02,-.01,.01,.01,.015,.001,.003,-.002])),(.06,np.array([.015,0.,.02,.005,.01,.002,.001,-.001]))]:
        model.record_input(t,value)
    full=model.input_effect(.08,.06);grip=model.input_effect(.08,.06,input_columns=range(5));motion=model.input_effect(.08,.06,input_columns=(5,6,7))
    checks['same_history_full_response_equals_grip_plus_public_motion']=np.max(abs(full-grip-motion))<1e-12
    checks['actual_increment_comes_from_recorded_input_endpoints']=np.allclose(model.actual_increment(.02,.06),np.array([.005,.02,.005,.005,-.01,-.001,.005,-.003]))
    checks['missing_input_history_is_not_reported_as_zero_execution']=model.actual_increment(-.01,.02) is None and model.actual_increment(.02,.08) is None
    model.snapshot=OutputForecast(np.r_[np.full(5,2.),np.full(5,.5),np.zeros(6)],np.zeros(16),np.zeros(16),
        np.full(16,.001),np.full(16,10),np.ones(16),np.ones(16,bool),np.zeros(16),np.zeros(16))
    raw=np.tile(model.snapshot.output_now,(3,1));raw[0,5]=-.01;copy=raw.copy();errors=model.error_by_horizon.copy()
    assessed=model.assess(raw)
    checks['negative_unsigned_shear_is_unknown_model_not_extra_force_constraint']=not assessed['usable'] and 'unsigned_shear_forecast_outside_mode' in assessed['reasons']
    checks['assessment_does_not_clip_prediction_or_calibration_error']=np.array_equal(raw,copy) and np.array_equal(errors,model.error_by_horizon)
    ctl=JointCoordinator(calibration(),limits,18.,model_document=synthetic_document())
    ctl.state.alpha=np.full(5,.5);ctl.state.valid=np.array([True,True,False,True,True]);ctl.state.closure_permission=np.array([True,True,True,True,False])
    ctl.executed_motion=np.array([.05,0.,0.]);ctl.motion_anchor=ctl.executed_motion.copy()
    nominal=np.full((3,8),.03);positive=np.full(8,.01);negative=np.full(8,.02)
    lower=np.full(5,.4);upper=np.full(5,.6);mlower=np.tile([.04,-.015,-.015],(3,1));mupper=np.tile([.06,.015,.015],(3,1))
    ctl._feedback_mode(nominal,positive,negative,lower,upper,mlower,mupper,'synthetic_unknown',{},dict(usable=False))
    checks['unknown_model_retains_bounded_observed_feedback']=np.any(ctl.interval_action>0.) and np.isnan(ctl.predicted).all()
    checks['unknown_model_does_not_claim_model_safety']=not ctl.debug['model_based_force_safety_certified'] and not ctl.debug['model_admitted'] and ctl.debug['final_authority_residual']<=1e-8
    checks['unknown_model_blocked_channels_cannot_close']=ctl.interval_action[2]<=0. and ctl.interval_action[4]<=0.
    checks['unknown_model_projection_obeys_first_interval_limits']=np.all(ctl.interval_action<=positive[:5]+1e-10) and np.all(ctl.interval_action>=-negative[:5]-1e-10)
    ctl.state.valid[:]=True;ctl.state.closure_permission[:]=True
    ctl.set_task_request(dict(time_s=8.,mode='hold',motion_target=[.05,0.,0.],executed_motion_reference=[.05,0.,0.],tracking_tolerances=[.01,.015,.015]))
    user,preview,lo,hi=ctl._motion_preview(8.,3,.02,positive,negative)
    checks['public_hold_tube_allows_identified_motion_correction_axes']=bool(np.all(lo[:,1:]<ctl.executed_motion[1:]) and np.all(hi[:,1:]>ctl.executed_motion[1:]) and np.all(lo[:,0]<ctl.executed_motion[0]))
    checks['unmeasured_positive_lift_still_cannot_exceed_visited_anchor']=bool(np.all(hi[:,0]<=ctl.motion_anchor[0]+1e-10))
    trajectory=public_transport_trajectory();now=10.;current=trajectory.position_at(now)
    ctl.set_task_request(dict(time_s=now,mode='transport',motion_target=current,executed_motion_reference=current,tracking_tolerances=[.01,.015,.015],motion_preview=trajectory.preview(now)))
    user,preview,lo,hi=ctl._motion_preview(now,3,.02,positive,negative)
    checks['known_user_preview_uses_next_interval_endpoint']=np.allclose(user[0],trajectory.position_at(now+.02),atol=1e-14,rtol=0.)
    ctl.set_task_request(dict(time_s=now,mode='transport',motion_target=current,executed_motion_reference=current,motion_preview=unknown_future_point(now,current)))
    user,_,_,_=ctl._motion_preview(now,3,.02,positive,negative)
    checks['unknown_user_future_does_not_use_fabricated_waypoints']=np.array_equal(user,np.tile(current,(3,1)))
    checks['source_identity_stable_during_independent_audit']=identity=={f:sha(ROOT/f) for f in sources}
    checks={k:bool(v) for k,v in checks.items()}
    details['failed']=[k for k,v in checks.items() if not v]
    result=dict(schema='independent-v6-core-targeted-audit-v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(checks.values()),
        checks=checks,count=len(checks),sources=identity,details=details,scope='Pure function and synthetic public interface checks only. No new physics, complete saved-flow replay, closed-loop advantage or global safety guarantee.')
    out=ROOT/'review'/('independent_v6_core_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=details['failed'],sources=identity,path=str(out)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
