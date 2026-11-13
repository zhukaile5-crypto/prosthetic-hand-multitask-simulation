"""Independent synthetic oracles for the shared signed/coupled observer."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib,json
import numpy as np

from contracts import SensorFrame
from state_estimator import CoupledTactileState
from vendor.frozen_controllers import Gains, TactileState

ROOT=Path(__file__).resolve().parent


def matrix():
    return np.array([[-2.,1.2,.1,0.,0.], [.4,3.,-.2,0.,0.],
                     [0.,.2,2.,-.3,0.], [0.,0.,.1,2.5,.2], [.1,0.,0.,-.2,1.8]])


def frame(t=0.,normal=None,alpha=None,ages=0.,B=None):
    B=matrix() if B is None else B
    return SensorFrame(time=t,normal=np.full(5,3.) if normal is None else np.asarray(normal,float),
        tangential=np.zeros((5,2)),contact_pos=np.zeros((5,3)),
        contact_normal=np.tile([1.,0.,0.],(5,1)),
        tangent_basis=np.tile([[0.,0.,1.],[0.,-1.,0.]],(5,1,1)),slip_local=np.zeros((5,2)),
        joint=np.full(5,.5),joint_velocity=np.zeros(5),valid=np.ones(5,bool),age=ages,
        alpha=np.full(5,.5) if alpha is None else np.asarray(alpha,float),phase='HOLDING',
        lift=.05,lift_ref=.1,support_axis=np.array([0.,0.,1.]),raw_joint=np.zeros(24),raw_joint_velocity=np.zeros(24),
        force_gain=np.maximum(.01,np.linalg.norm(B,axis=1)),closure_permission=np.ones(5,bool),
        closure_max=np.full(5,1.5),closure_min=np.full(5,-1.),nominal_force_weights=np.ones(5),
        motor_effort=10.,handload_prior=10.,load_estimate=0.,load_uncertainty=.5,load_valid=False,load_confidence=0.,
        force_tau=.015,force_matrix=B.copy(),model_reference_alpha=np.full(5,.5),response_error_bound=np.full(5,.2),
        aggregation_valid=np.ones(5,bool),aggregation_error=np.zeros(5),sample_time=t-float(np.max(ages)),
        auxiliary_wrench=np.zeros(6),auxiliary_pressure=0.,auxiliary_valid=True,
        auxiliary_age=float(np.max(ages)),auxiliary_uncertainty=.1)


def main():
    checks={};details={}
    def check(name,value,detail=None):
        checks[name]=bool(value)
        if detail is not None:details[name]=detail
        if not checks[name]:raise AssertionError(name)

    g=Gains();state=CoupledTactileState(g);B=matrix();reference=np.full(5,.5)
    times=[];latent=[];actual=[];errors=[];last=None
    age_ticks=np.array([4,1,3,2,0]);age=age_ticks*.005
    for k in range(240):
        t=k*.005
        command=reference+np.array([.11*np.sin(t*3.),.12*np.cos(t*2.)-.12,
            .09*np.sin(t*4.),-.08*np.sin(t*2.5),.1*np.sin(t*1.7)])
        # Independent physical-model recurrence; no implementation helper used.
        z=command.copy() if last is None else last+(1.-np.exp(-.005/.015))*(command-last)
        n=3.+B@(z-reference)
        latent.append(z.copy());actual.append(n.copy());times.append(t);last=z.copy()
        delayed=np.array([actual[max(0,k-int(age_ticks[i]))][i] for i in range(5)])
        fr=frame(t,delayed,command,ages=np.minimum(age,t))
        state.update(fr,.005)
        errors.append(float(np.max(np.abs(state.normal-n))))
        oracle_rows=np.array([latent[max(0,k-int(age_ticks[i]))] for i in range(5)])
        check(f'all_delayed_coordinates_rowwise_tick_{k}',np.allclose(state.last_delayed_coordinates,oracle_rows,atol=1e-12))
    check('signed_coupled_current_force_reconstruction',max(errors)<1e-11,{'max_error_N':max(errors)})
    check('negative_gain_and_cross_terms_preserved',state.force_matrix[0,0]<0. and state.force_matrix[0,1]==1.2)
    check('normalization_scale_not_used_as_physical_response',not np.allclose(np.diag(state.force_gain),state.force_matrix))
    check('calibration_error_is_uncertainty_floor',np.all(state.force_uncertainty>=.2))

    # No observed contact: arbitrary closure must not produce inferred force.
    s=CoupledTactileState(g)
    for k in range(30):
        fr=frame(k*.005,normal=np.zeros(5),alpha=np.full(5,.5+k*.01))
        fr.contact_normal[:]=0.;s.update(fr,.005)
    check('free_closure_never_invents_contact_force',not np.any(s.normal) and not np.any(s.contact_model_active))

    # Observe real contact, then explicit contact loss; eventual force model resets.
    s=CoupledTactileState(g)
    for k in range(10):s.update(frame(k*.005),.005)
    check('contact_initializes_only_after_measurement',s.contact_model_active.all())
    for k in range(10,90):
        fr=frame(k*.005,normal=np.zeros(5));fr.contact_normal[:]=0.;s.update(fr,.005)
    check('sustained_contact_loss_resets_model',not np.any(s.contact_model_active) and not np.any(s.normal))
    s.update(frame(.45),.005)
    # Filtering may delay reacquisition briefly; exercise the declared lifecycle.
    for k in range(91,105):s.update(frame(k*.005),.005)
    check('recontact_reinitializes_bias',s.contact_model_active.all() and np.max(abs(s.normal-3.))<1e-12)

    # Missing observations must not be interpreted as a newly measured zero.
    s=CoupledTactileState(g)
    for k in range(20):s.update(frame(k*.005),.005)
    old=s.delayed_normal.copy()
    missing=frame(.1);missing.valid[:]=False;missing.normal[:]=np.nan
    missing.tangential[:]=np.nan;missing.slip_local[:]=np.nan
    s.update(missing,.005)
    check('missing_observation_does_not_measure_zero',np.array_equal(old,s.delayed_normal) and not np.any(s.valid))
    old_mu=s.mu.copy()
    check('nonsliding_friction_not_falsely_identified',np.array_equal(old_mu,np.full(5,g.nominal_mu)))
    stale=frame(.105,ages=.2);s.update(stale,.005)
    check('stale_observations_are_invalid',not np.any(s.valid))
    bad_age=frame(.11,ages=np.array([np.nan,0.,0.,0.,0.]));s.update(bad_age,.005)
    check('invalid_age_has_zero_confidence',not s.valid[0] and s.confidence[0]==0.)

    # A diagonal public model reproduces the frozen state for ordinary valid input.
    Bdiag=np.diag([2.,3.,1.5,2.2,1.8]);gdiag=Gains(force_gain=tuple(np.diag(Bdiag)),force_tau=.015)
    old=TactileState(gdiag);new=CoupledTactileState(gdiag)
    max_diag=0.
    for k in range(100):
        t=k*.005;alpha=np.full(5,.5)+.02*np.sin(t*3.)*np.arange(1.,6.)
        fr=frame(t,normal=np.full(5,3.)+.02*np.sin(t)*np.arange(1.,6.),alpha=alpha,B=Bdiag)
        fr.response_error_bound[:]=0.;old.update(fr,.005);new.update(fr,.005)
        max_diag=max(max_diag,float(np.max(abs(old.normal-new.normal))))
    check('diagonal_limit_matches_frozen_observer',max_diag<1e-12,{'max_error_N':max_diag})
    check('diagonal_limit_preserves_friction_and_contact_rules',np.allclose(old.mu,new.mu) and np.allclose(old.contact,new.contact))

    # Full-hand load must combine signed finger and auxiliary contributions
    # ONCE before averaging with total motor load, never add the load channels.
    whole=CoupledTactileState(g)
    for k in range(600):
        fr=frame(k*.005);fr.tangential[:,0]=-.2
        fr.auxiliary_wrench=np.array([0.,0.,3.,0.,0.,0.]);fr.auxiliary_pressure=3.
        fr.load_estimate=2.;fr.load_uncertainty=.3;fr.load_valid=True;fr.load_confidence=.9
        whole.update(fr,.005)
    check('whole_hand_support_sums_signed_finger_and_auxiliary_once',abs(whole.finger_support_signed+1.)<1e-10 and abs(whole.auxiliary_support-3.)<1e-10 and abs(whole.tactile_load-2.)<1e-7,
          {'finger_signed_N':whole.finger_support_signed,'auxiliary_signed_N':whole.auxiliary_support_signed,'total_tactile_N':whole.tactile_load})
    check('total_tactile_and_motor_load_are_fused_not_added',abs(whole.load-2.)<1e-7 and abs(whole.motor_load-2.)<1e-12)
    old_tactile=whole.tactile_load
    fr=frame(3.);fr.auxiliary_valid=False;fr.auxiliary_wrench[:]=np.nan
    fr.load_estimate=2.;fr.load_uncertainty=.3;fr.load_valid=True;fr.load_confidence=.9
    whole.update(fr,.005)
    check('missing_auxiliary_contributes_no_fresh_support_or_fake_zero_load',not whole.auxiliary_valid and whole.auxiliary_support==0. and whole.tactile_load==old_tactile and abs(whole.load-2.)<1e-12)
    fr=frame(3.005);fr.auxiliary_wrench=np.array([0.,0.,9.,0.,0.,0.]);fr.auxiliary_pressure=9.;fr.auxiliary_age=.2
    whole.update(fr,.005)
    check('stale_auxiliary_cannot_contribute_passive_support',not whole.auxiliary_valid and whole.auxiliary_support==0. and np.isinf(whole.auxiliary_support_uncertainty))
    signed=CoupledTactileState(g)
    for k in range(300):
        fr=frame(k*.005);fr.tangential[:,0]=.2
        fr.auxiliary_wrench=np.array([0.,0.,-3.,0.,0.,0.]);fr.auxiliary_pressure=3.
        signed.update(fr,.005)
    check('downward_auxiliary_is_not_flipped_into_support',signed.auxiliary_support_signed<0. and signed.auxiliary_support==0. and signed.tactile_load<1e-10)
    # Multi-patch presence and scalar traction/slip may not be replaced by
    # a cancelling mean normal/vector when the distributed contract is valid.
    from sensors import HandSensors
    from verify_sensors import raw,calibration,patch
    sensors=HandSensors(calibration());distributed=CoupledTactileState(g)
    for k in range(100):
        patches=[]
        for digit in range(3):
            patches.extend([patch(digit,n=2.,force=(2.,.4,0.),slip=(0.,.01,0.)),
                            patch(digit,n=2.,force=(-2.,-.4,0.),direction=(-1.,0.,0.),slip=(0.,-.01,0.))])
        fr=sensors.sample(raw(k*.005,patches));distributed.update(fr,.005)
    check('distributed_contact_does_not_require_mean_normal',distributed.contact[:3].all() and np.linalg.norm(fr.contact_normal[0])<1e-12)
    check('distributed_scalar_shear_and_slip_drive_state',distributed.shear_magnitude[0]>.79 and distributed.slip_magnitude[0]>.0099 and distributed.slip_speed>.0099)
    check('distributed_scalar_signals_drive_friction_identifiability',distributed.mu_observable[:3].all() and distributed.sliding_since[0]>.08 and distributed.mu[0]<g.nominal_mu)
    check('distributed_geometry_drives_state_wrench',np.allclose(distributed.normal_wrench_map,fr.normal_wrench_map) and np.allclose(distributed.normal_support_coeff,fr.normal_support_coeff))

    changed=frame(.5,B=Bdiag.copy());changed.force_matrix[0,1]=.1
    rejected=False
    try:new.update(changed,.005)
    except ValueError:rejected=True
    check('calibration_switch_requires_new_observer',rejected)
    rejected=False
    try:CoupledTactileState(g).update(frame(),0.)
    except ValueError:rejected=True
    check('nonpositive_update_interval_rejected',rejected)

    files=('state_estimator.py','verify_state_estimator.py','force_model.py','vendor/frozen_controllers.py')
    report=dict(passed=all(checks.values()),check_count=len(checks),checks=checks,details=details,
        scope='Synthetic signed/coupled observer math and lifecycle, not physical hand acceptance or algorithm performance.',
        retained='Frozen TactileState filtering, contact lifecycle, friction-identifiability gates, local slip, and confidence-weighted load fusion structure.',
        changed='Full signed force matrix, row-wise all-coordinate delayed interpolation, calibration reference and residual uncertainty floor, explicit calibration lifecycle, signed full-hand external tactile support before total motor fusion.',
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})
    dest=ROOT/'review/observation_coupled_estimator.json';dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'checks':len(checks),'certificate':str(dest)},ensure_ascii=False))

if __name__=='__main__':main()
