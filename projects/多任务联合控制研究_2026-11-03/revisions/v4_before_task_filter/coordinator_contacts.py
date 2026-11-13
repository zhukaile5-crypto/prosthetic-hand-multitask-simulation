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
    if not all(checks.values()):raise AssertionError(checks)
    return checks
