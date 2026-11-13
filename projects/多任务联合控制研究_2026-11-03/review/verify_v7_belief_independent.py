"""Small independent measurement-time/covariance oracles; no plant or QP."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def document():
    times=np.arange(21)*.005
    errors={str(float(h)):dict(max_abs_per_output=np.zeros(16).tolist()) for h in (.02,.04,.06,.1)}
    kernel=dict(response_kernel_times_s=times.tolist(),
                response_to_20ms_interval_increment=np.zeros((21,16,8)).tolist(),
                identified_inputs=[True]*8,holdout_complete_causal_by_horizon=errors,
                holdout_differential_response=dict(max_abs_per_output=np.zeros(16).tolist()))
    return dict(execution_complete=True,shapes=[dict(shape='capsule',execution_complete=True,
        coordinates=dict(current_palm=kernel),
        input_availability=[dict(axis=i,plus=True,minus=True) for i in range(8)],
        measured_input_delta_min=[-.04]*5+[-.001,-.005,-.005],
        measured_input_delta_max=[.04]*5+[.001,.005,.005])])


SPECS=dict(force_sigma_N=.01,tangential_sigma_N=.005,position_sigma_m=.001,
           auxiliary_force_sigma_N=.01,auxiliary_moment_sigma_Nm=.0005)


def frame(now,valid=None,present=None):
    sample=round(now-.01,10)
    p=np.full(5,2.+.2*sample)
    x=np.zeros((5,3));x[:,0]=np.arange(5)*.01
    M=np.zeros((6,5));M[2]=1.;M[4]=-x[:,0]
    basis=np.tile(np.array([[1.,0.,0.],[0.,1.,0.]]),(5,1,1))
    return SimpleNamespace(time=round(now,10),sample_time=sample,alpha=np.full(5,.5),
        normal=p,shear_magnitude=np.full(5,.5),
        valid=np.ones(5,bool) if valid is None else np.asarray(valid,bool),
        contact_present=np.ones(5,bool) if present is None else np.asarray(present,bool),
        auxiliary_valid=True,auxiliary_wrench=np.array([0.,0.,.1,0.,0.,0.]),
        normal_wrench_map=M,tangent_basis=basis,contact_pos=x,
        finger_non_normal_by_finger=np.zeros((5,6)),tangential=np.zeros((5,2)),
        sensor_characteristics=deepcopy(SPECS))


def weights(times,query):
    """Independent direct two-column weighted least squares, centred at zero.

    The production implementation centres timestamps at their weighted mean;
    this oracle deliberately uses an ordinary time design instead.
    """
    t=np.asarray(times);A=np.column_stack([np.ones(len(t)),t])
    fade=np.exp((t-t[-1])/.1)
    return np.array([1.,query])@np.linalg.solve(A.T@(fade[:,None]*A),A.T*fade)


def run():
    from v7.belief import UnifiedBelief,MeasuredResponseHistory
    source=ROOT/'v7/belief.py';identity=sha(source);checks=[]
    def ck(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    def make(error,doc=None):
        return UnifiedBelief(dict(reference_alpha=[.5]*5,response_error_bound=[error]*5,
                                  shape='capsule',trust_radius=.05,model_trust_radius=.05,selected_radius=.05),
                             sensor_specs=SPECS,model_document=document() if doc is None else doc)
    low=make(0.);high=make(.508)
    # One immutable sequence of actual references. No future user request is
    # supplied to this estimator and no object/world state exists in the frame.
    for tick in range(61):
        now=.01+.005*tick
        s=low.update(frame(now),np.array([.05,0.,0.]))
        sh=high.update(frame(now),np.array([.05,0.,0.]))
    now=s.current_time
    times=np.array([p['time'] for p in low.packets if p['covered'] and p['valid'][0]])
    W=weights(times,now)
    expected_pressure=2.+.2*now
    ck('linear_drift_acquisition_to_current_time',np.max(abs(s.mean16[:5]-expected_pressure))<2e-10)
    ck('pressure_posterior_variance_direct_WLS',np.max(abs(np.diag(s.covariance)[:5]-.01**2*(W@W)))<2e-12)
    ck('calibration_error_never_measurement_covariance',np.array_equal(s.covariance,sh.covariance,equal_nan=True))
    ck('calibration_error_never_current_posterior_mean',np.array_equal(s.mean16,sh.mean16,equal_nan=True))
    # Shared pressure noise makes pressure_i and wrench_Fz correlated. Aux
    # noise adds variance to Fz without becoming a second force observation.
    ck('pressure_net_force_cross_covariance',abs(s.covariance[0,12]-.01**2*(W@W))<2e-12)
    ck('net_force_variance_counts_five_fingers_plus_aux',abs(s.covariance[12,12]-6*.01**2*(W@W))<2e-12)
    forecast=low.forecast(3);fh=high.forecast(3)
    for j in range(3):
        for k in range(3):
            wj=weights(times,now+.02*(j+1));wk=weights(times,now+.02*(k+1))
            expected=.01**2*(wj@wk)
            ck(f'cross_horizon_pressure_covariance_{j}_{k}',abs(forecast.covariance[16*j,16*k]-expected)<2e-12)
    ck('forecast_measured_statistical_covariance_independent_of_cal_error',np.array_equal(forecast.covariance,fh.covariance,equal_nan=True))
    ck('full_public_error_508_in_every_future_pressure',np.all(fh.calibration_error.reshape(3,16)[:,:5]==.508))
    ck('future_mean_same_posterior_linear_extrapolation',np.max(abs(forecast.c.reshape(3,16)[:,:5]-(2.+.2*(now+.02*np.arange(1,4)))[:,None]))<2e-10)
    ck('measurement_covariance_symmetric_PSD',np.max(abs(s.covariance-s.covariance.T))<1e-12 and np.linalg.eigvalsh(s.covariance).min()>-1e-12)
    ck('future_covariance_symmetric_PSD',np.max(abs(forecast.covariance-forecast.covariance.T))<1e-12 and np.linalg.eigvalsh(forecast.covariance).min()>-1e-12)
    original_len=len(low.input_history);original_mean=s.mean16.copy()
    again=low.update(frame(now),np.array([.05,0.,0.]))
    ck('same_actual_packet_idempotent',len(low.input_history)==original_len and np.array_equal(again.mean16,original_mean))
    changed=frame(now);changed.normal[0]+=.1
    try:low.update(changed,np.array([.05,0.,0.]));rejected=False
    except ValueError:rejected=True
    ck('cannot_rewrite_acquired_packet',rejected)
    missing=np.ones(5,bool);missing[4]=False
    nextf=frame(now+.005,valid=missing);nextf.normal[4]=12345.
    sm=low.update(nextf,np.array([.05,0.,0.]))
    ck('missing_pressure_is_not_new_zero_or_payload',abs(sm.mean16[4]-(2.+.2*(now+.005)))<2e-10)
    ck('missing_observation_mask_separate_from_recent_posterior',not sm.observed_pressure_valid5[4] and sm.valid16[4] and sm.age16[4]>.01)
    ck('unknown_wrench_does_not_become_new_complete_observation',not low._last_measurement_valid[10:].any())
    doc=document();kernel=np.asarray(doc['shapes'][0]['coordinates']['current_palm']['response_to_20ms_interval_increment'])
    kernel[:,0,6]=10.*np.minimum(1.,np.arange(21)*.005/.02)
    doc['shapes'][0]['coordinates']['current_palm']['response_to_20ms_interval_increment']=kernel.tolist()
    history=MeasuredResponseHistory(dict(reference_alpha=[.5]*5,response_error_bound=[0.]*5,shape='capsule'),doc)
    for tick in range(13):
        history.record_input(tick*.005,np.r_[np.full(5,.5),.05,.002*min(1.,tick/4.),0.])
    unknown_effect=history.input_effect(.06,.06).copy()
    ck('unknown_motion_not_point_subtracted',np.max(abs(unknown_effect))<1e-12)
    for tick in range(13,21):
        history.record_input(tick*.005,np.r_[np.full(5,.5),.05,.002,0.],np.ones(8,bool))
    ck('later_scope_never_reclassifies_old_unknown_motion',np.max(abs(history.input_effect(.1,.1)))<1e-12)
    for tick in range(21,29):
        history.record_input(tick*.005,np.r_[np.full(5,.5),.05,.002+.002*min(1.,(tick-20)/4.),0.],np.ones(8,bool))
    ck('subsequent_complete_known_ramp_gain_preserved',abs(history.input_effect(.14,.14)[0]-.02)<1e-12)
    ck('future_actual_records_cannot_rewrite_past_acquisition_effect',np.array_equal(history.input_effect(.06,.06),unknown_effect))
    moving=make(0.,doc)
    for tick in range(61):
        now=.01+.005*tick;smove=moving.update(frame(now),np.array([.05,.002*np.sin(now),0.]))
    fm=moving.forecast(3)
    ck('unknown_recent_motion_keeps_explicit_acquisition_mean_time',smove.mean_time==smove.observation_time and smove.mean_time<smove.current_time)
    ck('unknown_motion_same_total_posterior_not_fake_correction',np.max(abs(smove.mean16[:5]-(2.+.2*smove.observation_time)))<2e-10)
    ck('unknown_gain_not_zeroed_out_of_measured_evidence',abs(fm.P[0,6])>0. and not fm.known_input8[6] and not fm.model_usable)
    zig=make(0.)
    for tick in range(61):zig.update(frame(.005*tick),np.array([.05,0.,0.]))
    for k in range(1,5):
        fz=frame(.3+.005*k);fz.alpha[1]=.501 if k==1 else .5
        mask=np.ones(8,bool);mask[1]=k!=1
        sz=zig.update(fz,np.array([.05,0.,0.]),known_history_input8=mask)
    net=zig.actual_increment(.3,.32)
    ck('unknown_return_fixture_really_has_zero_net_delta',np.max(abs(net))<1e-12)
    ck('net_cancellation_does_not_certify_unknown_executed_return',not zig.forecast().model_usable and sz.mean_time==sz.observation_time)
    ck('unknown_actual_subinterval_preserved_in_diagnostics',bool(sz.diagnostics['unknown_executed_increments']))
    original=zig.response.input_scope[-1].copy();different=original.copy();different[1]=False
    try:zig.response.record_input(.32,np.r_[np.full(5,.5),.05,0.,0.],different);rejected=False
    except ValueError:rejected=True
    ck('same_executed_timestamp_scope_cannot_be_rewritten',rejected)
    ck('production_source_unchanged',sha(source)==identity)
    ck('no_physics_imported',not any(n=='mujoco' or n.startswith('mujoco.') or n in ('hand_plant','runtime') for n in sys.modules))
    return dict(schema='independent_v7_belief_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),
                passed=all(c['passed'] for c in checks),checks=checks,source_sha256=dict(belief=identity,checker=sha(__file__)),
                scope='synthetic exact linear drift and known-device-noise conditional covariance; no plant, QP, performance or model admission',
                limitations=['signed shear vector response unavailable','clipped or switching/noisy geometry does not yield exact Gaussian safety intervals','calibrated finite-ramp/model masks reviewed separately'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args();r=run()
    if args.output:
        if args.output.exists():raise FileExistsError(args.output)
        args.output.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=r['passed'],checks=len(r['checks']),failures=[c for c in r['checks'] if not c['passed']],output=str(args.output) if args.output else None),ensure_ascii=False))
    return 0 if r['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
