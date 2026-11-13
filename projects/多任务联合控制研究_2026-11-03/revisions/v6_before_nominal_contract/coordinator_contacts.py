"""Contact-conditioned set membership, not an invented friction reserve.

Sticking aggregate measurements imply only a necessary friction lower bound.
They do not identify exact material friction, per-patch cone margins or an
upper bound. Stable measured contact is an empirical local viable point. Its
pressure/shear corridor, not a discounted default mu, defines holding tasks.
"""
from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class ContactMembership:
    mode: np.ndarray
    mu_lower: np.ndarray
    mu_upper: np.ndarray
    prior_conflict: np.ndarray
    ratio: np.ndarray
    ratio_uncertainty: np.ndarray
    stable_duration: np.ndarray
    anchor_pressure: np.ndarray
    anchor_shear: np.ndarray
    anchor_ratio: np.ndarray
    anchor_valid: np.ndarray
    measurement_error_pressure: np.ndarray
    measurement_error_shear: np.ndarray
    speed_upper: np.ndarray
    speed_lower: np.ndarray
    speed_bias: np.ndarray
    speed_uncertainty: np.ndarray


class ContactCorridorObserver:
    def __init__(self,nominal_mu=.45,stable_dwell=.12):
        self.nominal_mu=float(nominal_mu);self.stable_dwell=float(stable_dwell)
        self.stable_duration=np.zeros(5);self.sliding_duration=np.zeros(5)
        self.unstable_duration=np.zeros(5);self.stick_latched=np.zeros(5,bool)
        self.mu_lower=np.zeros(5);self.mode=np.full(5,'UNKNOWN',dtype='<U16')
        self.anchor_pressure=np.zeros(5);self.anchor_shear=np.zeros(5)
        self.anchor_ratio=np.zeros(5);self.anchor_valid=np.zeros(5,bool)
        self.last_time=None;self.last_positions=None;self.snapshot=None
        self.speed_history=[deque(maxlen=80) for _ in range(5)]
        self.speed_bias=np.zeros(5);self.speed_sigma=np.full(5,.0002);self.noise_frozen=False
        self.reference_updated_at=None;self.reference_updates=0

    def update(self,frame,state,prediction):
        now=float(frame.time);dt=.005 if self.last_time is None else max(0.,min(.05,now-self.last_time))
        if self.last_time is not None and abs(now-self.last_time)<1e-10:return self.snapshot
        self.last_time=now
        p=np.asarray(frame.normal,float);tau=np.asarray(frame.shear_magnitude,float)
        valid=np.asarray(state.valid,bool)&np.isfinite(p)&np.isfinite(tau)&(p>.15)
        if getattr(frame,'contact_present',None) is not None:valid&=np.asarray(frame.contact_present,bool)
        positions=np.asarray(frame.contact_pos,float)
        jump=np.zeros(5,bool) if self.last_positions is None else np.linalg.norm(positions-self.last_positions,axis=1)>.012
        self.last_positions=positions.copy()
        ep=np.maximum(.009,np.asarray(prediction.fit_error[:5],float))
        et=np.maximum(.009,np.asarray(prediction.fit_error[5:10],float))
        speed=np.maximum(0.,np.asarray(state.slip_magnitude,float))
        # The nonnegative max-slip channel has a positive noise bias. Estimate
        # its filtered baseline using only the common public warm-up, then
        # freeze it; later physical slip cannot be learned away as sensor noise.
        if not self.noise_frozen:
            for i in np.flatnonzero(valid&(speed<.002)):
                self.speed_history[i].append(float(speed[i]))
                values=np.asarray(self.speed_history[i]);median=float(np.median(values))
                self.speed_bias[i]=median
                self.speed_sigma[i]=max(.00005,1.4826*float(np.median(np.abs(values-median))))
        speed_error=3.*self.speed_sigma
        speed_upper=np.maximum(0.,speed-self.speed_bias)+speed_error
        speed_lower=np.maximum(0.,speed-self.speed_bias-speed_error)
        sticking=valid&~jump&(speed_upper<.002)
        sliding=valid&(speed_lower>.004)
        self.stable_duration=np.where(sticking,self.stable_duration+dt,0.)
        self.sliding_duration=np.where(sliding,self.sliding_duration+dt,0.)
        self.unstable_duration=np.where(valid&(speed_lower>.002),self.unstable_duration+dt,0.)
        reset=jump|~valid
        self.mu_lower[reset]=0.;self.anchor_valid[reset]=False
        self.stick_latched[reset]=False
        lower=np.maximum(0.,tau-et)/np.maximum(.15,p+ep)
        ratio=tau/np.maximum(.15,p)
        ratio_error=(et+ratio*ep)/np.maximum(.15,p-ep)
        confirmed=sticking&(self.stable_duration>=self.stable_dwell)
        self.stick_latched[confirmed]=True
        self.stick_latched[self.unstable_duration>=.03]=False
        # Finite-window evidence can be revoked by changed contact/sliding;
        # do not keep an everlasting maximum across an unobserved friction drop.
        self.mu_lower[confirmed]=lower[confirmed]
        self.mu_lower[sliding]=np.minimum(self.mu_lower[sliding],lower[sliding])
        self.mode[:]='UNKNOWN';self.mode[valid]='TRANSIENT'
        self.mode[self.stick_latched&valid]='STICK_CONFIRMED'
        self.mode[self.unstable_duration>=.03]='INCIPIENT'
        self.mode[sliding]='INCIPIENT'
        self.mode[self.sliding_duration>=.08]='SLIDING'
        new_anchor=confirmed&~self.anchor_valid
        self.anchor_pressure[new_anchor]=p[new_anchor]
        self.anchor_shear[new_anchor]=tau[new_anchor]
        self.anchor_ratio[new_anchor]=ratio[new_anchor]
        self.anchor_valid[new_anchor]=True
        self.snapshot=ContactMembership(self.mode.copy(),self.mu_lower.copy(),np.full(5,np.inf),
            confirmed&(self.mu_lower>self.nominal_mu),ratio,ratio_error,self.stable_duration.copy(),
            self.anchor_pressure.copy(),self.anchor_shear.copy(),self.anchor_ratio.copy(),self.anchor_valid.copy(),
            ep,et,speed_upper,speed_lower,self.speed_bias.copy(),speed_error)
        return self.snapshot

    def freeze_handoff(self,prediction):
        """One common public handoff anchor, not a future selected success."""
        if self.snapshot is None:raise RuntimeError('Contact warm-up is required')
        self.noise_frozen=True
        p=np.asarray(prediction.output_now[:5]);tau=np.asarray(prediction.output_now[5:10])
        available=np.asarray(prediction.output_valid[:5],bool)&np.asarray(prediction.output_valid[5:10],bool)&(p>.15)
        self.anchor_pressure[~available]=0.;self.anchor_shear[~available]=0.;self.anchor_ratio[~available]=0.
        self.anchor_valid[~available]=False
        self.anchor_pressure[available]=p[available];self.anchor_shear[available]=tau[available]
        self.anchor_ratio[available]=tau[available]/np.maximum(.15,p[available]);self.anchor_valid[available]=True
        self.snapshot.anchor_pressure=self.anchor_pressure.copy();self.snapshot.anchor_shear=self.anchor_shear.copy()
        self.snapshot.anchor_ratio=self.anchor_ratio.copy();self.snapshot.anchor_valid=self.anchor_valid.copy()
        self.reference_updated_at=self.last_time

    def adopt_stable_reference(self,prediction,limits,budget,now,last_intervention):
        """Causal contact reference update, never a model-domain expansion.

        A post-intervention state must persist with observed contact/slip and
        acceptable measured force. Joint stillness alone is not force settling.
        The reference is descriptive and is never a pressure point target.
        """
        if not self.noise_frozen or self.snapshot is None:return False
        if now-last_intervention<self.stable_dwell-1e-9:return False
        if self.reference_updated_at is not None and now-self.reference_updated_at<self.stable_dwell-1e-9:return False
        p=np.asarray(prediction.output_now[:5],float);tau=np.asarray(prediction.output_now[5:10],float)
        measured=np.asarray(prediction.last_measured[:5],float)
        if not np.isfinite(np.r_[p,tau,measured]).all() or np.any(p<0.) or np.any(tau<0.):return False
        if np.any(p>limits) or np.any(measured>limits) or max(float(p.sum()),float(measured.sum()))>budget:return False
        good=(self.snapshot.mode=='STICK_CONFIRMED')&prediction.output_valid[:5]&prediction.output_valid[5:10]&(p>.15)
        if not good[0] or np.count_nonzero(good)<3:return False
        settling=np.abs(prediction.drift_rate[:10])*self.stable_dwell<=prediction.fit_error[:10]
        good&=settling[:5]&settling[5:10]
        if not good[0] or np.count_nonzero(good)<3:return False
        self.anchor_pressure[good]=p[good];self.anchor_shear[good]=tau[good]
        self.anchor_ratio[good]=tau[good]/p[good];self.anchor_valid[good]=True
        self.snapshot.anchor_pressure=self.anchor_pressure.copy();self.snapshot.anchor_shear=self.anchor_shear.copy()
        self.snapshot.anchor_ratio=self.anchor_ratio.copy();self.snapshot.anchor_valid=self.anchor_valid.copy()
        self.reference_updated_at=float(now);self.reference_updates+=1
        return True


class ForceQualityTask:
    """Measured force regulation is part of the task, not a fixed handoff force.

    The mature friction-ratio target supplies a nominal grip distribution.
    Known task-motion response is separated by the SAME input-history kernel.
    Integral changes are reconciled against actual executed increments. No
    optimizer request, unobserved interval or blocked channel earns integration.
    """
    def __init__(self,gains):
        self.g=gains;self.scalar=float(gains.base);self.boost=0.;self.integral=0.
        self.target_ratio=None;self.samples=deque(maxlen=5);self.stable_since=None
        self.pending_integral=None;self.pending_boost=None;self.pending_action=None;self.last_realization=None
        self.last_proposal_time=None

    def observation_missing(self):
        self.stable_since=None;self.samples.clear()
        self.pending_integral=None;self.pending_boost=None;self.pending_action=None
        self.last_realization=None

    def reconcile(self,actual_increment):
        self.last_realization=None
        if self.pending_integral is None:return
        if actual_increment is not None and self.pending_action is not None:
            actual=np.asarray(actual_increment,float)[:5];requested=self.pending_action
            norm=float(requested@requested)
            fraction=float(actual@requested/max(1e-18,norm))
            orthogonal=np.linalg.norm(actual-fraction*requested)
            realized=(norm<1e-16 and np.linalg.norm(actual)<1e-8) or (
                abs(fraction-1.)<=.05 and orthogonal<=.05*np.sqrt(norm)+1e-8)
            self.last_realization=dict(actual_increment=actual.tolist(),
                requested_nominal_increment=requested.tolist(),fraction=fraction,
                integral_committed=bool(realized))
            if realized:
                self.integral=float(self.pending_integral)
                if self.pending_boost is not None:self.boost=float(self.pending_boost)
        self.pending_integral=None;self.pending_boost=None;self.pending_action=None

    def propose(self,state,membership,persistent,limits,budget,motion_pressure=None,noise_sigma=None):
        from vendor.frozen_controllers import project_force
        period=float(self.g.decision_period)
        if self.last_proposal_time is not None and state.time-self.last_proposal_time>1.5*period:
            self.observation_missing()
        self.last_proposal_time=float(state.time)
        supported=np.asarray(persistent,bool)&state.valid&state.contact_model_active
        shear=np.linalg.norm(state.tangential[supported],axis=1).sum()
        ratio=float(shear/max(.25,float(state.delayed_normal[supported].sum()))) if supported.any() else 0.
        if supported.sum()>=2:self.samples.append(ratio)
        measured_ratio=float(np.mean(self.samples)) if self.samples else 0.
        slipping=bool(np.any(np.isin(membership.mode[supported],('INCIPIENT','SLIDING'))))
        pending=self.integral;candidate_boost=self.boost
        if self.target_ratio is None:
            if slipping or supported.sum()<3:
                self.stable_since=None
                if slipping:candidate_boost=min(5.-self.g.base,self.boost+self.g.slip_boost_rate*period)
            else:
                self.boost*=np.exp(-period/1.5)
                candidate_boost=self.boost
                if self.stable_since is None:self.stable_since=float(state.time)
                if state.time-self.stable_since>=1.-1e-9:
                    self.target_ratio=max(.025,measured_ratio);self.scalar=self.g.base+self.boost
            scalar=self.g.base+candidate_boost
        else:
            error=measured_ratio-self.target_ratio
            candidate=float(np.clip(self.integral+error*period,-.35,.75))
            raw=self.scalar+self.g.ratio_kp*error+self.g.ratio_ki*candidate
            if self.g.ratio_floor<=raw<=5. or raw>5. and error<0. or raw<self.g.ratio_floor and error>0.:
                pending=candidate
            scalar=float(np.clip(raw,self.g.ratio_floor,5.))
        target=project_force(scalar*state.nominal_weights,np.maximum(0.,limits-self.g.force_margin),max(0.,budget-2.*self.g.force_margin))
        task_response=np.zeros(5) if motion_pressure is None else np.asarray(motion_pressure,float)
        # The force quality error excludes only publicly explained task motion.
        # Full safety always uses actual pressure, not the residual signal.
        signal=state.normal-task_response
        sigma=np.full(5,.01) if noise_sigma is None else np.broadcast_to(np.asarray(noise_sigma,float),(5,))
        error=target-signal
        statistically_resolved=np.sign(error)*np.maximum(0.,np.abs(error)-2.*sigma)
        action=self.g.baseline_tracking_gain*(state.force_model.inverse(active=supported)@statistically_resolved)
        # Every instrumented contact is protected, even when it has no grasp
        # affiliation. Such contact is never credited as supporting the task.
        protective=state.valid&state.contact_model_active&(state.normal>limits-self.g.force_margin)
        if np.any(protective):
            overload=np.minimum(0.,limits-self.g.force_margin-state.normal)
            action=self.g.baseline_tracking_gain*(state.force_model.inverse(active=state.valid&state.contact_model_active)@overload)
        absent=state.valid&~state.contact_model_active&state.closure_permission
        action+=np.where(absent,np.clip(state.model_reference_alpha-state.alpha,
            -self.g.open_rate*period,self.g.close_rate*period),0.)
        self.pending_integral=pending;self.pending_boost=candidate_boost;self.pending_action=action.copy()
        return action,target+task_response,dict(task='measured_force_quality_with_friction_ratio_reference',
            target_ratio=self.target_ratio,measured_ratio=measured_ratio,ratio_scalar=scalar,
            force_quality_base_target_N=target.tolist(),explained_motion_pressure_N=task_response.tolist(),
            force_quality_signal_N=signal.tolist(),force_quality_error_N=error.tolist(),
            force_quality_resolved_error_N=statistically_resolved.tolist(),
            force_quality_noise_sigma_N=sigma.tolist(),anti_windup_actual_realization=self.last_realization,
            integral=self.integral,persistent_feedback_mask=supported.tolist())


def verify_contact_membership():
    from types import SimpleNamespace
    observer=ContactCorridorObserver();state=SimpleNamespace(valid=np.ones(5,bool),slip_magnitude=np.zeros(5))
    forecast=SimpleNamespace(fit_error=np.full(16,.01),output_valid=np.ones(16,bool),
        output_now=np.r_[np.full(5,2.),np.full(5,1.2),np.zeros(6)])
    for k in range(50):
        frame=SimpleNamespace(time=k*.005,normal=np.full(5,2.),shear_magnitude=np.full(5,1.2),contact_pos=np.zeros((5,3)))
        observer.update(frame,state,forecast)
    snap=observer.snapshot
    checks=dict(sticking_implies_lower_not_exact_friction=bool(np.all(snap.mu_lower>.58) and np.isinf(snap.mu_upper).all()),
        inconsistent_prior_is_diagnosis_not_physical_deficit=bool(snap.prior_conflict.all()),
        stable_point_is_retained=bool(snap.anchor_valid.all() and np.allclose(snap.anchor_ratio,.6)))
    observer.freeze_handoff(forecast)
    state.slip_magnitude[:]=.008
    for k in range(50,71):frame.time=k*.005;observer.update(frame,state,forecast)
    checks['sliding_is_not_called_sticking']=bool(np.all(observer.snapshot.mode=='SLIDING'))
    frame.time+=.005;frame.contact_pos=np.ones((5,3));observer.update(frame,state,forecast)
    checks['changed_contact_revokes_old_anchor']=bool(not observer.snapshot.anchor_valid.any())
    checks['slip_noise_baseline_frozen_before_later_sliding']=bool(np.all(observer.snapshot.speed_bias==0.))
    settled=ContactCorridorObserver();state.slip_magnitude[:]=0.
    for k in range(50):
        frame.time=8.+k*.005;frame.contact_pos=np.zeros((5,3));settled.update(frame,state,forecast)
    settled.freeze_handoff(forecast)
    forecast.output_now[:10]*=.8;forecast.last_measured=np.r_[forecast.output_now[:5],np.zeros(11)]
    forecast.drift_rate=np.zeros(16)
    early=settled.adopt_stable_reference(forecast,np.full(5,5.),18.,8.30,8.25)
    later=settled.adopt_stable_reference(forecast,np.full(5,5.),18.,8.50,8.25)
    checks['post_intervention_reference_requires_causal_dwell']=bool(not early and later)
    checks['new_lower_stable_reference_is_adopted_without_old_force_goal']=bool(np.allclose(settled.anchor_pressure,1.6))
    if not all(checks.values()):raise AssertionError(checks)
    return checks
