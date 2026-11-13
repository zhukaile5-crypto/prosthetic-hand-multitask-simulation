"""Small independent V8 set/no-intervention and supervisory oracles; no plant."""
from pathlib import Path
from dataclasses import replace
from datetime import datetime,timezone
from types import SimpleNamespace
from copy import deepcopy
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from review.verify_v7_reachable_task import fixture,freeze_except,clean
from review.verify_v7_controller_independent import CAL,GAINS,DEVICE,public_frame,task
from review.verify_v7_belief_independent import document

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run():
    from v8.feedback import ReachableTaskPlanner
    from v8.controller import JointTaskController,AggregateGripReference
    sources={n:sha(ROOT/n) for n in ('v8/controller.py','v8/feedback.py','v7/belief.py')}
    planner=ReachableTaskPlanner();checks=[];outputs={}
    def ck(k,v,detail=None):checks.append(dict(name=k,passed=bool(v),detail=detail))
    def solve(name,f):
        r=planner.solve(**deepcopy(f));a=np.asarray(r.first_action8);plan=np.asarray(r.plan)
        ck(name+'/finite_bounded_actual_action',np.isfinite(plan).all() and np.all(plan>=f['step_lower8']-2e-7) and np.all(plan<=f['step_upper8']+2e-7))
        alpha=f['alpha5']+np.cumsum(plan[:,:5],0)
        ck(name+'/cumulative_alpha_domain',np.all(alpha>=f['alpha_lower5']-2e-7) and np.all(alpha<=f['alpha_upper5']+2e-7))
        ck(name+'/no_empirical_forecast_certification',not np.isfinite(r.predicted_pressure).any() and r.debug['prediction_available'] is False and r.debug['model_based_force_safety_certified'] is False)
        outputs[name]=dict(action=a.tolist(),debug=r.debug);return r
    f=fixture();f['pressure_map'][:5,:5]=np.array([[2.,1.,0.,0.,0.],[0.,2.,1.,0.,0.],[0.,0.,2.,1.,0.],[0.,0.,0.,2.,1.],[1.,0.,0.,0.,2.]])
    f['force_total_target']=9.;f['persistent5'][:]=True
    inside=solve('inside_safe_minimum_set',f)
    ck('inside_set_is_exact_zero_not_a_small_correction',np.array_equal(inside.first_action8,np.zeros(8)))
    f2=deepcopy(f);f2['pressure_covariance5x5']*=1e-6;f2['calibration_error5H'][:]=.508;f2['previous_action8']=np.array([.01,-.01,.01,-.01,0.,0.,0.,0.])
    unchanged=solve('smaller_noise_larger_calibration_error_and_old_action',f2)
    ck('posterior_precision_does_not_create_micro_action',np.array_equal(unchanged.first_action8,np.zeros(8)))
    ck('full_calibration_error_stays_diagnostic_508',np.all(np.asarray(unchanged.debug['calibration_error_preserved'])==.508))
    f=freeze_except(fixture(),{1});f['pressure_map'][0,1]=10.;f['force_total_target']=10.1
    deficit=solve('one_free_FF_minimum_deficit',f)
    ck('minimum_deficit_has_known_analytic_action',abs(deficit.first_action8[1]-.01)<1e-6)
    nextf=deepcopy(f);nextp=f['pressure_now5']+f['pressure_map']@deficit.first_action8
    nextf['pressure_now5']=nextp;nextf['observed_pressure5']=nextp;nextf['pressure_free5H']=nextp
    nextf['alpha5']+=deficit.first_action8[:5];nextf['force_total_target']=9.
    surplus=solve('relaxed_minimum_does_not_unload_surplus',nextf)
    ck('relaxed_minimum_no_unload_then_reload',np.array_equal(surplus.first_action8,np.zeros(8)))
    f=freeze_except(fixture(),{0,1});f['alpha5'][0]=1.;f['step_upper8'][0]=0.
    f['pressure_map'][:2,:2]=[[1.,1.],[1.,2.]];f['force_total_target']=10.03
    sat=solve('saturated_TH_coupled_FF_reserve',f)
    ck('saturated_TH_excluded_from_positive_allocation',sat.first_action8[0]<=0. and abs(sat.first_action8[1]-.01)<2e-6)
    f=freeze_except(fixture(),set(range(5)));f['alpha5'][:]=1.;f['step_upper8'][:5]=0.;f['pressure_map'][:5,:5]=np.eye(5);f['force_total_target']=10.3
    impossible=solve('all_fingers_saturated_reserve_unreachable',f)
    ck('unreachable_minimum_has_zero_activity_and_visible_slack',np.max(abs(impossible.first_action8))<1e-9 and impossible.debug['aggregate_reserve_slack_N']>=.29999)
    f=freeze_except(fixture(),{1});f['pressure_map'][0,1]=10.;f['force_total_target']=10.1;f['step_upper8'][1]=0.
    one=solve('unavailable_positive_direction',f)
    ck('single_sided_unreachable_reserve_not_inverse_clip',np.max(abs(one.first_action8))<1e-9 and one.debug['aggregate_reserve_slack_N']>=.09999)
    f=freeze_except(fixture(),{1});f['pressure_map'][0,1]=10.;f['force_total_target']=10.1
    f['pressure_valid5'][4]=False;f['observed_pressure5'][4]=np.nan
    missing=solve('missing_group_contribution',f)
    ck('missing_does_not_shrink_denominator_or_create_deficit',np.array_equal(missing.first_action8,np.zeros(8)) and missing.debug['aggregate_task_available'] is False and missing.debug['grip_support_mask']==[True]*5 and missing.debug['aggregate_reserve_slack_N'] is None)
    f=freeze_except(fixture(),{1});f['pressure_now5'][0]=5.1;f['observed_pressure5'][0]=5.1;f['pressure_free5H'][0]=5.1;f['limits5'][0]=5.;f['pressure_map'][0,1]=-10.;f['force_total_target']=13.2
    risk=solve('material_risk_has_priority_over_reserve',f)
    after=f['pressure_free5H']+f['pressure_map']@risk.first_action8
    ck('reserve_cannot_restore_overforce',after[0]<=5.00001 and risk.debug['aggregate_reserve_slack_N']>=.1999)
    f=freeze_except(fixture(),{1});f['pressure_map'][0,1]=-10.;f['observed_pressure5'][0]=6.;f['limits5'][0]=5.;f['force_total_target']=9.
    raw=solve('raw_overforce_not_filtered_away',f)
    ck('latest_raw_risk_uses_bounded_unloading',raw.first_action8[1]>=.0399 and raw.debug['raw_pressure_excess_N'][0]==1.)
    # Direct reference contract: learning never turns the reserve into a point.
    ctl=JointTaskController(CAL,[5.]*5,18.,GAINS,model_document=document())
    for k in range(81):ctl.state.update(public_frame(k*.005),.005)
    ref=AggregateGripReference(ctl.g)
    b=SimpleNamespace(mean16=np.r_[np.full(5,2.),np.full(5,.5),np.zeros(6)],valid16=np.ones(16,bool),covariance=np.eye(16)*1e-8)
    scalars=[]
    for k in range(61):
        scalar,_=ref.propose(k*.02,b,np.ones(5,bool),np.ones(5,bool),np.zeros(5),DEVICE,True,slip_variance=np.full(5,1e-10),weight_sum=5.)
        scalars.append(scalar)
    ck('one_second_learning_retains_minimum_floor_not_10N_point',np.array_equal(np.array(scalars),np.full(61,GAINS['ratio_floor'])))
    slips=np.array([.01,0.,0.,0.,0.])
    scalar,slip_debug=ref.propose(1.22,b,np.ones(5,bool),np.ones(5,bool),slips,DEVICE,True,slip_variance=np.full(5,1e-10),weight_sum=5.)
    ck('first_resolved_slip_raises_minimum_above_current_grip',slip_debug['slip_evidence'] and scalar*5.>10. and abs(scalar-(2.+GAINS['slip_boost_rate']*.02))<1e-12)
    # Real controller with explicit provided public posterior. This isolates
    # supervisory hysteresis from response physics; no hidden state truth exists.
    motion=np.array([.05,0.,0.]);ctl.set_task_request(task(.4,motion,'transport',[.05,.01,0.]))
    original_task=json.dumps(ctl.task_request,sort_keys=True)
    base=ctl.belief.snapshot
    def posterior(first,std=.01,missing=False):
        mean=base.mean16.copy();mean[:5]=2.;mean[0]=first;cov=base.covariance.copy();cov[:5,:]=0.;cov[:,:5]=0.;cov[:5,:5]=np.eye(5)*std**2
        observed=base.observed_pressure5.copy();observed[:]=2.;observed[0]=first;observed_valid=base.observed_pressure_valid5.copy();observed_valid[4]=not missing
        ctl.state.valid[:]=True;ctl.state.valid[4]=not missing
        ctl.belief.snapshot=replace(base,mean16=mean,covariance=cov,observed_pressure5=observed,observed_pressure_valid5=observed_valid)
        ctl._decision();return ctl.interval_action.copy()
    a=posterior(6.)
    ck('resolved_pressure_risk_immediately_holds_motion',ctl.recovery_motion_hold and np.max(abs(a[5:]))<1e-10 and a[0]<0.)
    posterior(4.99)
    ck('ambiguous_pressure_band_keeps_recovery_latch',ctl.recovery_motion_hold)
    posterior(4.9,missing=True)
    ck('missing_raw_does_not_fake_complete_clearance',ctl.recovery_motion_hold)
    a=posterior(4.9)
    ck('complete_upper_bound_clearance_resumes_original_motion',not ctl.recovery_motion_hold and a[6]>0.)
    ctl.state.slip=np.array([.01,0.,0.,0.,0.]);ctl.state.slip_variance=np.full(5,1e-10)
    a=posterior(2.)
    ck('resolved_slip_holds_motion_and_requests_reserve',ctl.recovery_motion_hold and np.max(abs(a[5:]))<1e-10 and np.any(a[:5]>0.))
    ctl.state.slip[0]=.003;posterior(2.)
    ck('slip_between_enter_exit_thresholds_stays_held',ctl.recovery_motion_hold)
    ctl.state.slip[0]=.001;a=posterior(2.)
    ck('low_upper_slip_and_clear_pressure_release_motion_latch',not ctl.recovery_motion_hold and a[6]>0.)
    ck('recovery_never_redefines_original_user_target_or_deadline',json.dumps(ctl.task_request,sort_keys=True)==original_task)
    ctl.set_task_request(task(.405,motion,'release'));released=ctl.step(public_frame(.405),'HOLDING','release')
    ck('release_preempts_reserve_and_recovery',released.debug.get('release_authority') is True and np.all(released.delta_closure<=0.) and not np.isfinite(released.predicted_normal).any())
    ctl.set_task_request(task(.410,motion,'hold'));again=ctl.step(public_frame(.410),'HOLDING','hold')
    ck('release_latch_does_not_regrasp',again.debug.get('release_authority') is True and np.all(again.delta_closure<=0.))
    ck('all_tested_sources_unchanged',sources=={n:sha(ROOT/n) for n in sources})
    ck('no_plant_or_physics_imported',not any(n=='mujoco' or n.startswith('mujoco.') or n=='hand_plant' for n in sys.modules))
    return dict(schema='independent_v8_minimum_task_set_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(x['passed'] for x in checks),checks=checks,outputs=outputs,
        source_sha256=sources,checker_sha256=sha(__file__),physics_executed=False,plant_constructed=False,
        scope='10 analytic set-allocation fixtures and actual reference/controller interfaces with provided public posterior, risk/slip hysteresis and release; no physical performance or global safety certification',
        limitations=['Response surrogate and movement hold are reactive engineering policy, not a predictive material bound','No-op is mathematical when the measured task set is already satisfied; actual physical drift may still occur','Reuses frozen V7 belief and full-model source assumptions; no fresh model identification'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args();r=run()
    if args.output:
        if args.output.exists():raise FileExistsError(args.output)
        args.output.write_text(json.dumps(clean(r),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=r['passed'],checks=len(r['checks']),failed=[x for x in r['checks'] if not x['passed']],output=str(args.output) if args.output else None),ensure_ascii=False))
    return 0 if r['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
