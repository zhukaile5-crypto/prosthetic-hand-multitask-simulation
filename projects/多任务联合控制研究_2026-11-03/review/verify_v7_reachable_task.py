"""Independent V7 analytic fixtures. No plant, simulation, or production edits.

Fixture self-checks establish the analytic oracles only. They are deliberately
separate from checks of the actual v7.feedback.ReachableTaskPlanner.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(h=1):
    p = np.full(5, 2.0)
    motion = np.array([.05, 0., 0.])
    return dict(
        pressure_now5=p.copy(), pressure_covariance5x5=np.eye(5)*1e-8,
        pressure_map=np.zeros((5*h, 8*h)), pressure_free5H=np.tile(p, h),
        calibration_error5H=np.zeros(5*h), observed_pressure5=p.copy(),
        pressure_valid5=np.ones(5, dtype=bool), limits5=np.full(5, 100.),
        budget=500., alpha5=np.full(5, .5), alpha_lower5=np.zeros(5),
        alpha_upper5=np.ones(5), step_lower8=np.array([-.04]*5+[-.0007,-.002,-.002]),
        step_upper8=np.array([.04]*5+[.0007,.002,.002]), motion_now3=motion,
        user_targetsH3=np.tile(motion,(h,1)), motion_lower3=np.array([0.,-.4,-.4]),
        motion_upper3=np.array([.1,.4,.4]), tracking_tolerances3=np.array([.01,.015,.015]),
        persistent5=np.zeros(5,dtype=bool), force_total_target=10.,
        previous_action8=np.zeros(8), model_usable=True,
    )


def freeze_except(f, free):
    for j in range(8):
        if j not in free:
            f['step_lower8'][j]=f['step_upper8'][j]=0.
    return f


def cases():
    out={}
    # FF only redistributes TH/FF pressure; aggregate cannot change.
    f=freeze_except(fixture(),{0,1});f['alpha5'][0]=1.;f['step_upper8'][0]=0.
    f['pressure_map'][:2,:2]=[[1.,1.],[1.,-1.]]
    f['force_total_target']=10.02
    out['saturated_thumb_redistribution_only']=f
    # Free FF has aggregate gain 3; requested .03 N needs +.01 alpha.
    f=freeze_except(fixture(),{0,1});f['alpha5'][0]=1.;f['step_upper8'][0]=0.
    f['pressure_map'][:2,:2]=[[1.,1.],[1.,2.]]
    f['force_total_target']=10.03
    out['saturated_thumb_reachable_aggregate']=f
    f=freeze_except(fixture(),set()); f['force_total_target']=10.3
    f['pressure_map'][:5,:5]=np.eye(5)
    out['all_inputs_fixed_unreachable_force']=f
    f=freeze_except(fixture(),set(range(5)));f['alpha5'][:]=1.
    f['step_upper8'][:5]=0.;f['pressure_map'][:5,:5]=np.eye(5)
    f['force_total_target']=10.3
    out['all_fingers_upper_saturated_open_direction_available']=f
    f=freeze_except(fixture(),{1}); f['pressure_map'][0,1]=1.
    f['step_lower8'][1]=-.04; f['step_upper8'][1]=0.
    f['force_total_target']=10.02
    out['one_sided_positive_unavailable']=f
    # A feasible .5mm next motion adds .3 N; simultaneous FF -.03 cancels it.
    f=freeze_except(fixture(),{1,5}); f['pressure_map'][0,1]=10.
    f['pressure_map'][0,5]=600.
    f['user_targetsH3'][0,0]+=.0005
    f['tracking_tolerances3'][0]=0.
    out['joint_motion_force_compensation']=f
    f=freeze_except(fixture(),{5}); f['motion_now3'][0]=.1
    f['user_targetsH3'][0,0]=.08; f['tracking_tolerances3'][0]=.005
    out['rate_and_user_tube_disjoint']=f
    # Holding means a tube, not an immovable old anchor. Safety needs +.1mm.
    f=freeze_except(fixture(),{5}); f['pressure_now5'][0]=5.1
    f['observed_pressure5'][0]=5.1; f['pressure_free5H'][0]=5.1
    f['force_total_target']=13.1; f['limits5'][0]=5.
    f['pressure_map'][0,5]=-1000.
    out['safe_motion_inside_hold_tube']=f
    f=freeze_except(fixture(),{1});f['pressure_now5'][0]=.2
    f['observed_pressure5'][0]=.2;f['pressure_free5H'][0]=.2
    f['pressure_map'][0,1]=10.;f['limits5'][0]=.1
    f['persistent5'][0]=True;f['force_total_target']=.1
    out['material_priority_over_incompatible_contact_floor']=f
    f=freeze_except(fixture(),{1});f['pressure_now5'][0]=.01
    f['observed_pressure5'][0]=.01;f['pressure_free5H'][0]=.01
    f['pressure_map'][:2,1]=[-1.,-1.]
    f['force_total_target']=float(f['pressure_now5'].sum()-.06)
    out['unsigned_pressure_cannot_supply_fictitious_negative_force']=f
    f=freeze_except(fixture(),{1});f['observed_pressure5'][0]=6.
    f['limits5'][0]=5.;f['pressure_map'][0,1]=-10.
    out['raw_material_overforce_not_hidden_by_filtered_mean']=f
    f=freeze_except(fixture(2),{0})
    f['alpha5'][0]=.99
    f['pressure_map'][0,0]=f['pressure_map'][5,0]=1.
    f['pressure_map'][5,8]=1.
    f['force_total_target']=10.1
    out['cumulative_alpha_upper_across_ramps']=f
    f=freeze_except(fixture(),{1}); f['pressure_map'][0,1]=1.
    f['force_total_target']=10.02
    out['small_calibration_error']=f
    f=copy.deepcopy(f); f['calibration_error5H'][:]=.508
    out['large_calibration_error_nonbinding_caps']=f
    f=freeze_except(fixture(),{1,2});f['pressure_map'][0,1]=1.;f['pressure_map'][1,2]=1.
    C=np.eye(5)*1e-6;C[:2,:2]=np.array([[1.,1.5],[1.5,4.]])*1e-6
    f['pressure_covariance5x5']=C;f['forecast_covariance']=C.copy()
    f['joint_pressure_covariance']=np.block([[C,.8*C],[.8*C,C]])
    f['force_total_target']=10.03
    out['correlated_posterior_mahalanobis_allocation']=f
    f=freeze_except(fixture(),{1}); f['pressure_map'][0,1]=1.
    f['force_total_target']=10.02; f['pressure_valid5'][4]=False
    f['observed_pressure5'][4]=np.nan
    out['partial_observation_missing']=f
    f=copy.deepcopy(f); f['observed_pressure5'][4]=12345.
    out['partial_observation_arbitrary_invalid_payload']=f
    f=copy.deepcopy(out['partial_observation_missing']);f['force_total_target']=10.
    out['partial_missing_must_not_manufacture_total_grip_deficit']=f
    f=freeze_except(fixture(),{1}); f['pressure_map'][0,1]=1.
    f['force_total_target']=10.02; f['model_usable']=False
    out['unknown_model_bounded_fallback']=f
    return out


def check_oracles():
    c=cases(); checks=[]
    def ck(name, ok, detail=None):
        checks.append(dict(name=name,passed=bool(ok),detail=detail))
    f=c['saturated_thumb_redistribution_only']; u=np.zeros(8); u[1]=.02
    dp=f['pressure_map']@u
    ck('redistribution_aggregate_gain_exact_zero',dp.sum()==0. and np.abs(dp).sum()==.04)
    f=c['saturated_thumb_reachable_aggregate']; u=np.zeros(8);u[1]=.01
    ck('free_finger_attains_aggregate',abs((f['pressure_free5H']+f['pressure_map']@u).sum()-10.03)<1e-12)
    # The unconstrained per-finger inverse and clip has the wrong direction.
    bad=np.linalg.solve(np.array([[1.,1.],[1.,2.]]),np.array([.03,0.]))
    ck('inverse_then_clip_counterexample',bad[0]>.04 and bad[1]<0.)
    f=c['joint_motion_force_compensation'];u=np.zeros(8);u[1]=-.03;u[5]=.0005
    ck('simultaneous_motion_and_force_analytic_solution',np.max(abs(f['pressure_map']@u))<1e-12)
    f=c['rate_and_user_tube_disjoint']
    gap=f['motion_now3'][0]+f['step_lower8'][5]-(f['user_targetsH3'][0,0]+f['tracking_tolerances3'][0])
    ck('physical_rate_tube_gap_14_3_mm',abs(gap-.0143)<1e-12,gap)
    f=c['safe_motion_inside_hold_tube'];u=np.zeros(8);u[5]=.0001
    ck('nonzero_hold_tube_safety_action',abs((f['pressure_free5H']+f['pressure_map']@u)[0]-5.)<1e-12)
    ck('fixture_has_no_oracle_or_future_disturbance_fields',not any(any(x in k for x in ('mass','friction','object','scenario','disturbance')) for f in c.values() for k in f))
    ck('public_calibration_error_preserved_508',np.all(c['large_calibration_error_nonbinding_caps']['calibration_error5H']==.508))
    return checks


def asarray(result,name):
    return np.asarray(getattr(result,name) if hasattr(result,name) else result[name],dtype=float)


def run_planner():
    sys.path.insert(0,str(ROOT))
    from v7.feedback import ReachableTaskPlanner
    source=ROOT/'v7'/'feedback.py'; before=sha(source)
    checks=[]; outputs={}; planner=ReachableTaskPlanner()
    sig=inspect.signature(planner.solve)
    def ck(name,ok,detail=None):
        checks.append(dict(name=name,passed=bool(ok),detail=detail))
    for name,original in cases().items():
        f=copy.deepcopy(original)
        # This alias is the sole compatibility accommodation. Missing semantics
        # are not silently filtered out of the actual production interface.
        if 'user_targets' in sig.parameters and 'user_targetsH3' not in sig.parameters:
            f['user_targets']=f.pop('user_targetsH3')
        try:
            r=planner.solve(**f)
            a=asarray(r,'first_action8'); plan=asarray(r,'plan').reshape(-1,8)
            pred=asarray(r,'predicted_pressure').reshape(-1,5)
            debug=getattr(r,'debug',{})
            outputs[name]=dict(action=a.tolist(),plan=plan.tolist(),predicted=pred.tolist(),debug=debug)
            ck(name+':finite_actual_action',a.shape==(8,) and np.isfinite(a).all())
            ck(name+':first_action_matches_actual_plan',np.allclose(a,plan[0],atol=1e-12,rtol=0))
            ck(name+':per_ramp_bounds',np.all(plan>=original['step_lower8']-2e-7) and np.all(plan<=original['step_upper8']+2e-7))
            alpha=original['alpha5']+np.cumsum(plan[:,:5],axis=0)
            motion=original['motion_now3']+np.cumsum(plan[:,5:],axis=0)
            ck(name+':cumulative_actual_alpha_bounds',np.all(alpha>=original['alpha_lower5']-2e-7) and np.all(alpha<=original['alpha_upper5']+2e-7))
            ck(name+':cumulative_actual_motion_bounds',np.all(motion>=original['motion_lower3']-2e-7) and np.all(motion<=original['motion_upper3']+2e-7))
            if original['model_usable']:
                oracle=(original['pressure_free5H']+original['pressure_map']@plan.ravel()).reshape(-1,5)
                ck(name+':full_execution_map_residual',np.max(abs(pred-oracle))<2e-7,float(np.max(abs(pred-oracle))))
            else:
                ck(name+':unknown_model_no_safety_forecast',not np.isfinite(pred).any())
        except Exception as e:
            ck(name+':production_interface',False,type(e).__name__+': '+str(e))
    def action(name):
        return np.asarray(outputs[name]['action'])
    def have(*names):return all(n in outputs for n in names)
    if have('saturated_thumb_redistribution_only'):
        ck('unreachable_total_does_not_create_distribution_TV',np.max(abs(action('saturated_thumb_redistribution_only')))<2e-5)
    if have('saturated_thumb_reachable_aggregate'):
        ck('saturated_TH_free_FF_solve_not_inverse_clip',abs(action('saturated_thumb_reachable_aggregate')[1]-.01)<2e-4)
    if have('all_inputs_fixed_unreachable_force'):
        ck('all_saturated_actual_zero',np.max(abs(action('all_inputs_fixed_unreachable_force')))<1e-10)
    if have('all_fingers_upper_saturated_open_direction_available'):
        ck('all_upper_saturated_no_manufactured_closing_or_redistribution',np.max(abs(action('all_fingers_upper_saturated_open_direction_available')))<2e-5)
    for name,gap in (('saturated_thumb_redistribution_only',.02),('all_inputs_fixed_unreachable_force',.3)):
        if have(name):
            debug=outputs[name]['debug']
            try:
                requested=np.asarray(debug['requested_aggregate_gap'],float)
                projected=np.asarray(debug['reachable_projection_gap'],float)
                ck(name+':original_requested_gap_is_not_hidden',np.isfinite(requested).all() and np.max(abs(requested-gap))<8e-4)
                ck(name+':reachable_projection_residual_separate',np.isfinite(projected).all() and np.max(abs(projected))<8e-4)
            except (KeyError,TypeError,ValueError) as e:
                ck(name+':separate_requested_and_projected_gap_fields',False,str(e))
    if have('one_sided_positive_unavailable'):
        ck('unavailable_positive_direction_respected',action('one_sided_positive_unavailable')[1]<=2e-7)
    if have('joint_motion_force_compensation'):
        a=action('joint_motion_force_compensation')
        ck('finite_ramp_joint_motion_plus_pressure_compensation',abs(a[5]-.0005)<2e-7 and abs(a[1]+.03)<3e-4)
    if have('safe_motion_inside_hold_tube'):
        a=action('safe_motion_inside_hold_tube')
        ck('hold_tube_does_not_pin_old_motion_anchor',a[5]>=.0001-3e-6)
    if have('material_priority_over_incompatible_contact_floor'):
        r=outputs['material_priority_over_incompatible_contact_floor']
        ck('incompatible_contact_cannot_restore_material_overforce',r['predicted'][0][0]<=.10001)
        ck('incompatible_contact_slack_remains_visible',float(r['debug']['contact_slack_N'][0])>=.0499)
    if have('unsigned_pressure_cannot_supply_fictitious_negative_force'):
        r=outputs['unsigned_pressure_cannot_supply_fictitious_negative_force']
        ck('unsigned_pressure_domain_preserved_before_total_force_projection',np.min(r['predicted'])>=-2e-7,
           'A negative affine pressure is outside this unsigned response mode; it cannot buy a smaller total-force target.')
    if have('raw_material_overforce_not_hidden_by_filtered_mean'):
        r=outputs['raw_material_overforce_not_hidden_by_filtered_mean']
        ck('valid_raw_overforce_gets_bounded_reactive_unloading_priority',r['action'][1]>=.0399)
        ck('valid_raw_overforce_not_relabelled_zero_risk',abs(float(r['debug'].get('raw_pressure_excess_N',[np.nan])[0])-1.)<1e-12)
    if have('rate_and_user_tube_disjoint'):
        name='rate_and_user_tube_disjoint';f=cases()[name]
        plan=np.asarray(outputs[name]['plan']);motion=f['motion_now3']+np.cumsum(plan[:,5:],axis=0)
        gap=np.maximum(0.,abs(motion-f['user_targetsH3'])-f['tracking_tolerances3'])
        ck('unreachable_original_motion_tube_not_redefined',gap[0,0]>=.0143-2e-7)
        try:
            declared=np.asarray(outputs[name]['debug']['tube_violation'],float)
            ck('declared_tube_violation_matches_original_user_request',declared.shape==gap.shape and np.allclose(declared,gap,atol=2e-7,rtol=0))
        except (KeyError,TypeError,ValueError) as e:
            ck('explicit_original_tube_conflict_field',False,str(e))
    if have('small_calibration_error','large_calibration_error_nonbinding_caps'):
        ck('calibration_error_not_measurement_deadzone_when_safety_inactive',np.max(abs(action('small_calibration_error')-action('large_calibration_error_nonbinding_caps')))<2e-5)
    if have('correlated_posterior_mahalanobis_allocation'):
        r=outputs['correlated_posterior_mahalanobis_allocation'];a=np.asarray(r['action'])
        total=a[1]+a[2]
        # min d'C^-1d under 1'd=s has d=C1*s/(1'C1).
        expected_share=(1.+1.5)/(1.+4.+2*1.5)
        ck('correlated_noise_changes_quality_allocation_not_component_deadzone',total>.01 and abs(a[1]/total-expected_share)<5e-4)
        ck('actual_current_future_covariance_scope_reported','joint' in r['debug'].get('quality_covariance_role',''))
    if have('partial_observation_missing','partial_observation_arbitrary_invalid_payload'):
        ck('invalid_observation_payload_cannot_enter_force_task',np.max(abs(action('partial_observation_missing')-action('partial_observation_arbitrary_invalid_payload')))<2e-7)
    if have('partial_missing_must_not_manufacture_total_grip_deficit'):
        a=action('partial_missing_must_not_manufacture_total_grip_deficit')
        ck('missing_contribution_does_not_create_new_total_force_request',np.max(abs(a))<2e-5,
           'The complete five-channel target is 10N, not a 10N target for the observed four. Missing is not confirmed zero or a new deficit.')
    bad=freeze_except(fixture(),set());bad['pressure_free5H'][0]=-.2
    try:
        planner.solve(**bad)
        domain_rejected=False;reason='returned an affine negative-pressure mode'
    except Exception as e:
        domain_rejected=type(e).__name__=='ModelDomainUnavailable'
        reason=type(e).__name__+': '+str(e)
    ck('unrecoverable_affine_mode_is_not_called_physical_infeasibility',domain_rejected,reason)
    ck('tested_production_source_unchanged',sha(source)==before)
    ck('no_physics_module_imported',not any(n=='mujoco' or n.startswith('mujoco.') or n in ('hand_plant','runtime') for n in sys.modules))
    return checks,outputs,dict(feedback_sha256=before,signature=str(sig))


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fixture-self-test',action='store_true')
    ap.add_argument('--output',type=Path);args=ap.parse_args()
    checks=check_oracles();outputs={};identity={}
    if not args.fixture_self_test:
        more,outputs,identity=run_planner();checks+=more
    result=dict(schema='independent_v7_reachable_task_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),
                scope='pure numerical planner and analytic fixtures; no physics, performance, model admission, or release-pressure certification',
                production_tested=not args.fixture_self_test,passed=all(c['passed'] for c in checks),
                checks=checks,outputs=outputs,source_identity=identity,checker_sha256=sha(__file__),
                outstanding=['controller same-belief integration and measurement-time continuity','release authority and no holding re-entry','actual four 5ms ramp execution equality','posterior covariance construction is reviewed separately'])
    if args.output:
        if args.output.exists():raise FileExistsError(args.output)
        args.output.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],production_tested=result['production_tested'],checks=len(checks),failures=[c for c in checks if not c['passed']],output=str(args.output) if args.output else None),ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
