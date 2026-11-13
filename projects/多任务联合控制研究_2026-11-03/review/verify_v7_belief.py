"""Author algebra/clock/statistical checks; no plant or physics construction."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from v7.belief import UnifiedBelief,MeasuredResponseHistory
from coordinator_model import synthetic_document,ContactConditionedModel

DEFAULT_SPECS=dict(force_sigma_N=.01,tangential_sigma_N=.005,position_sigma_m=.0001,
    auxiliary_force_sigma_N=.01,auxiliary_moment_sigma_Nm=.0005)


def calibration_fixture(error=.02):
    return dict(reference_alpha=np.zeros(5),response_error_bound=np.full(5,error),selected_radius=.04,
        force_matrix=np.eye(5)*10.,force_tau=.02,config=dict(shape='capsule'))


def base_output():return np.r_[np.full(5,2.),np.full(5,1.2),[0.,0.,1.2,0.,0.,0.]]


def frame_fixture(time,output=None,alpha=None,motion=None,sample_time=None,specs=None):
    y=base_output() if output is None else np.asarray(output,float)
    M=np.zeros((6,5));M[2]=1.;M[4]=np.linspace(-.02,.02,5)
    T=np.zeros((5,6));A=y[10:]-M@y[:5]
    return SimpleNamespace(time=float(time),sample_time=max(0.,time-.01) if sample_time is None else float(sample_time),
        alpha=np.zeros(5) if alpha is None else np.asarray(alpha,float),
        normal=y[:5].copy(),shear_magnitude=y[5:10].copy(),valid=np.ones(5,bool),contact_present=np.ones(5,bool),
        normal_wrench_map=M,finger_non_normal_by_finger=T,
        tangent_basis=np.tile(np.array([[1.,0.,0.],[0.,1.,0.]]),(5,1,1)),
        contact_pos=np.column_stack((np.linspace(-.02,.02,5),np.zeros(5),np.zeros(5))),
        auxiliary_wrench=A,auxiliary_valid=True,tangential=np.zeros((5,2)),
        sensor_characteristics=dict(DEFAULT_SPECS if specs is None else specs))


def build_constant_belief(error=.02,through=.4,specs=None):
    spec=dict(DEFAULT_SPECS if specs is None else specs)
    belief=UnifiedBelief(calibration_fixture(error),sensor_specs=spec,model_document=synthetic_document())
    for tick in range(round(through/.005)+1):belief.update(frame_fixture(tick*.005,specs=spec),np.array([.1,0.,0.]))
    return belief


def verify():
    checks={};metrics={}
    original_update=ContactConditionedModel.update;original_trajectory=ContactConditionedModel.trajectory
    def forbidden(*args,**kwargs):raise AssertionError('A second legacy estimator was invoked')
    try:
        ContactConditionedModel.update=forbidden;ContactConditionedModel.trajectory=forbidden
        belief=build_constant_belief()
    finally:ContactConditionedModel.update=original_update;ContactConditionedModel.trajectory=original_trajectory
    s=belief.snapshot;f=belief.forecast()
    checks['no_legacy_estimator_or_Coupled_state_is_run']=bool(np.allclose(s.mean16,base_output(),atol=1e-10))
    checks['constant_force_and_full_clock_reached']=s.current_time==.4 and np.isclose(s.observation_time,.39) and s.mean_time==s.current_time
    checks['known_grip_unknown_motion_scope_is_explicit']=bool(np.array_equal(f.known_input8,[True]*5+[False]*3) and np.array_equal(f.usable_input_mask,f.known_input8))
    checks['constant_static_local_model_can_be_used_for_known_subset']=f.model_usable
    checks['one_posterior_has_full_cross_output_covariance']=bool(np.isfinite(s.covariance).all() and np.linalg.eigvalsh(s.covariance).min()>-1e-10 and s.covariance[0,12]>0.)
    checks['joint_current_future_covariance_first_block_equals_snapshot']=bool(f.joint_covariance.shape==(64,64) and np.allclose(f.joint_covariance[:16,:16],s.covariance,atol=1e-12))
    packets=list(belief.packets);times=np.array([x['time'] for x in packets]);R=packets[-1]['R']
    fade=np.exp(-(times[-1]-times)/.1);center=fade@times/fade.sum();X=np.column_stack((np.ones(len(times)),times-center))
    operator=np.linalg.solve(X.T@(fade[:,None]*X),X.T*fade)
    weights=[np.array([1.,t-center])@operator for t in [.4,.42,.44,.46]]
    oracle=np.block([[(a@b)*R for b in weights] for a in weights])
    checks['independent_WLS_weight_oracle_matches_cross_horizon_covariance']=bool(np.allclose(oracle,f.joint_covariance,rtol=1e-8,atol=1e-12))
    metrics['maximum_covariance_oracle_difference']=float(abs(oracle-f.joint_covariance).max())
    big=build_constant_belief(error=.508)
    checks['large_calibration_error_neither_enters_posterior_cov_nor_gets_capped']=bool(
        np.array_equal(big.snapshot.covariance,s.covariance) and np.min(big.forecast().calibration_error.reshape(3,16)[:,:5])>=.508)
    # A 20ms actual ramp plus a true linear residual has an independent closed form.
    b=UnifiedBelief(calibration_fixture(),DEFAULT_SPECS,synthetic_document());delta=np.r_[.003,-.002,.001,0.,-.001,0.,0.,0.]
    drift=np.r_[.2,np.zeros(15)];start=.10
    for k in range(81):
        now=k*.005;sample=max(0.,now-.01);value=delta*np.clip((now-start)/.02,0.,1.)
        response=b.response.kernel_at(max(0.,sample-start))@delta if sample>=start else np.zeros(16)
        y=base_output()+drift*sample+response
        b.update(frame_fixture(now,y,value[:5],sample_time=sample),np.array([.1,0.,0.]))
    expected=base_output()+drift*.4+b.response.kernel_at(.4-start)@delta
    forecast=b.forecast();future=np.concatenate([base_output()+drift*t+b.response.kernel_at(t-start)@delta for t in (.42,.44,.46)])
    checks['delayed_linear_residual_repropagates_to_current_same_force_posterior']=bool(np.allclose(b.snapshot.mean16,expected,atol=1e-9))
    checks['future_forecast_uses_same_level_slope_and_actual_ramp_history']=bool(np.allclose(forecast.c,future,atol=1e-9))
    before=belief.snapshot.mean16.copy();old_cov=belief.snapshot.covariance.copy();history=np.asarray(belief.input_history).copy()
    duplicate=frame_fixture(.4);duplicate.motion_preview={'future_targets':[[999.,999.,999.]]}
    belief.update(duplicate,[.1,0.,0.])
    checks['future_preview_does_not_enter_actual_history_or_posterior']=bool(np.array_equal(history,np.asarray(belief.input_history)) and np.array_equal(before,belief.snapshot.mean16) and np.array_equal(old_cov,belief.snapshot.covariance))
    memory=MeasuredResponseHistory(calibration_fixture(),synthetic_document())
    for k in range(9):memory.record_input(k*.005,np.r_[delta[:5]*min(1.,k/4.),.1,0.,0.])
    past=memory.input_effect(.0125,.0125).copy();past_increment=memory.actual_increment(.005,.0125).copy()
    memory.record_input(.045,np.ones(8))
    checks['later_actual_samples_cannot_rewrite_off_clock_past_effect_or_increment']=bool(np.array_equal(past,memory.input_effect(.0125,.0125)) and np.array_equal(past_increment,memory.actual_increment(.005,.0125)))
    scope=MeasuredResponseHistory(calibration_fixture(),synthetic_document())
    for k in range(13):scope.record_input(k*.005,np.r_[np.zeros(5),.1,.003*min(1.,k/4.),0.])
    checks['unknown_motion_history_is_not_point_subtracted']=bool(np.allclose(scope.input_effect(.06,.06),0.))
    for k in range(13,21):scope.record_input(k*.005,np.r_[np.zeros(5),.1,.003,0.],np.ones(8,bool))
    checks['later_scope_cannot_retroactively_claim_old_motion']=bool(np.allclose(scope.input_effect(.10,.10),0.))
    for k in range(21,29):scope.record_input(k*.005,np.r_[np.zeros(5),.1,.003+.002*min(1.,(k-20)/4.),0.],np.ones(8,bool))
    checks['subsequent_full_known_ramp_can_use_measured_response']=bool(np.max(abs(scope.input_effect(.14,.14)))>0.)
    moving=build_constant_belief();mf=frame_fixture(.405);moving.update(mf,[.1,.001,0.])
    checks['unknown_motion_change_uses_total_measured_posterior_at_explicit_mean_time']=bool(moving.snapshot.mean_time==mf.sample_time and np.allclose(moving.snapshot.mean16,base_output()) and 'unidentified_input_changed_since_acquisition' in moving.forecast().reasons)
    missing=build_constant_belief();mf=frame_fixture(.405);mf.valid[4]=False;mf.contact_present[4]=False;mf.normal[4]=np.nan;mf.shear_magnitude[4]=np.nan
    missing.update(mf,[.1,0.,0.])
    checks['missing_finger_is_not_new_absence_or_zero_force']=bool(missing.snapshot.active_contact5[4] and np.isnan(missing.snapshot.observed_pressure5[4]) and missing.snapshot.mean16[4]>1.9)
    lost=build_constant_belief();mf=frame_fixture(.405);mf.valid[:]=False;mf.normal[:]=np.nan;mf.shear_magnitude[:]=np.nan
    lost.update(mf,[.1,0.,0.]);lf=lost.forecast()
    checks['all_observation_missing_is_separate_from_model_unknown']=bool(not lf.observation_available and 'observation_unavailable' in lf.reasons and not lf.model_usable)
    inactive=build_constant_belief(error=.508);mf=frame_fixture(.405);mf.contact_present[1]=False;mf.normal[1]=.01;mf.shear_magnitude[1]=.01
    inactive.update(mf,[.1,0.,0.]);fi=inactive.forecast()
    checks['explicit_inactive_mode_zero_does_not_clip_active_model_or_error']=bool(inactive.snapshot.mean16[1]==0. and inactive.snapshot.mean16[6]==0. and fi.calibration_error[1]>=.508)
    falling=UnifiedBelief(calibration_fixture(),DEFAULT_SPECS,synthetic_document())
    for k in range(81):
        now=k*.005;sample=max(0.,now-.01);y=base_output();y[0]=.34-.8*sample;y[5]=.6*y[0]
        falling.update(frame_fixture(now,y,sample_time=sample),[.1,0.,0.])
    ff=falling.forecast()
    checks['negative_active_forecast_is_raw_unknown_not_clipped']=bool(ff.c.reshape(3,16)[:,:5].min()<0. and not ff.model_usable and 'negative_active_pressure_forecast' in ff.reasons)
    impact=build_constant_belief();mf=frame_fixture(.405);mf.normal[4]=8.;mf.shear_magnitude[4]=4.8
    impact.update(mf,[.1,0.,0.]);fi=impact.forecast()
    checks['impact_is_observed_high_force_not_inflated_covariance_demand']=bool(impact.snapshot.mean16[4]>7.9 and not fi.model_usable and impact.snapshot.covariance[4,4]<.001 and impact.snapshot.diagnostics['mode_event']['innovation_change_outputs'])
    try:belief.update(frame_fixture(.405,specs={**DEFAULT_SPECS,'force_sigma_N':.08}),[.1,0.,0.])
    except ValueError:checks['constructor_frame_noise_mismatch_rejected']=True
    else:checks['constructor_frame_noise_mismatch_rejected']=False
    if not all(checks.values()):raise AssertionError(checks)
    return dict(schema='v7_unified_belief_author_validation_v1',passed=True,checks=checks,metrics=metrics,
        scope='pure public algebra/clock/covariance; no plant or physical stepping',
        source_sha256=hashlib.sha256((ROOT/'v7/belief.py').read_bytes()).hexdigest())


if __name__=='__main__':
    target=ROOT/'review/v7_belief_author_validation_v1.json'
    if target.exists():raise FileExistsError(target)
    result=verify()
    with target.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False,indent=2))
